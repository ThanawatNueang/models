"""Procedural texture painting in UV space + material setup.

Every triangle is rasterized into its UV layout, so each texel knows its 3D
rest-pose position P, normal N and interpolated vertex attributes (skin
weights, ambient occlusion).  "Texture shaders" below are plain numpy
functions of (P, N, ...) that return albedo / roughness / height; the height
is turned into a tangent-space normal map from the UV-space gradient, which
is exactly the tangent frame glTF and Blender use.
"""
import os

import bpy
import numpy as np
from mathutils import Vector
from mathutils.bvhtree import BVHTree
from scipy import ndimage

import noise
from noise import smoothstep as ss

RES = {"skin": 4096, "eye": 512, "shirt": 2048, "pants": 2048, "boots": 1024, "hair": 1024}


# ------------------------------------------------------------ raster

class Raster:
    """UV-space raster of one material of one mesh object."""

    def __init__(self, obj, res, mat_index=0, attrs=None):
        me = obj.data
        me.calc_loop_triangles()
        tris = me.loop_triangles
        n = len(tris)
        vi = np.empty(n * 3, np.int64)
        tris.foreach_get("vertices", vi)
        li = np.empty(n * 3, np.int64)
        tris.foreach_get("loops", li)
        mi = np.empty(n, np.int64)
        tris.foreach_get("material_index", mi)
        vi, li = vi.reshape(-1, 3), li.reshape(-1, 3)
        uv = np.empty(len(me.loops) * 2)
        me.uv_layers.active.data.foreach_get("uv", uv)
        uv = uv.reshape(-1, 2)
        co = np.empty(len(me.vertices) * 3)
        me.vertices.foreach_get("co", co)
        co = co.reshape(-1, 3)
        vn = np.empty(len(me.vertices) * 3)
        me.vertices.foreach_get("normal", vn)
        vn = vn.reshape(-1, 3)
        sel = mi == mat_index
        # skip the inner shell of solidified cloth: it shares the outer UVs
        inner = np.zeros(len(me.vertices), bool)
        vg = obj.vertex_groups.get("_shell_inner")
        if vg is not None:
            for v in me.vertices:
                for g in v.groups:
                    if g.group == vg.index and g.weight > 0.5:
                        inner[v.index] = True
            sel &= ~inner[vi].all(axis=1)
        vi, li = vi[sel], li[sel]
        self.res = res
        self.vi = vi
        tid, bary = _raster(uv[li] * res, res)
        self.valid = tid >= 0
        m = self.valid
        self.tid = tid[m]
        self.bary = bary[m]
        self.co, self.vn = co, vn
        self.P = self.interp(co)
        N = self.interp(vn)
        self.N = N / np.maximum(np.linalg.norm(N, axis=1, keepdims=True), 1e-9)
        self.attrs = attrs or {}

    def interp(self, arr):
        v = self.vi[self.tid]  # (k,3)
        if arr.ndim == 1:
            return (arr[v] * self.bary).sum(axis=1)
        return (arr[v] * self.bary[:, :, None]).sum(axis=1)

    def image(self, vals, fill=0.0):
        """Scatter per-texel values back into a padded (res,res,C) image."""
        vals = np.asarray(vals, dtype=np.float32)
        if vals.ndim == 1:
            vals = vals[:, None]
        img = np.full((self.res, self.res, vals.shape[1]), fill, np.float32)
        img[self.valid] = vals
        return pad(img, self.valid)


def _raster(tri_px, res):
    """Rasterize triangles given in pixel coords -> (tid, bary) images."""
    tid = np.full((res, res), -1, np.int64)
    bary = np.zeros((res, res, 3), np.float32)
    lo = np.floor(tri_px.min(axis=1)).astype(np.int64)
    hi = np.ceil(tri_px.max(axis=1)).astype(np.int64)
    size = np.maximum(hi - lo, 1).max(axis=1)
    a, b, c = tri_px[:, 0], tri_px[:, 1], tri_px[:, 2]
    det = (b[:, 1] - c[:, 1]) * (a[:, 0] - c[:, 0]) + (c[:, 0] - b[:, 0]) * (a[:, 1] - c[:, 1])
    ok = np.abs(det) > 1e-12
    buckets = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
    prev = 0
    for S in buckets:
        sel = np.nonzero(ok & (size > prev) & (size <= S))[0]
        prev = S
        if not len(sel):
            continue
        for chunk in np.array_split(sel, max(1, len(sel) * S * S // 4_000_000 + 1)):
            if not len(chunk):
                continue
            off = np.arange(S + 1)
            xs = lo[chunk, 0][:, None, None] + off[None, None, :]
            ys = lo[chunk, 1][:, None, None] + off[None, :, None]
            xs = np.broadcast_to(xs, (len(chunk), S + 1, S + 1))
            ys = np.broadcast_to(ys, (len(chunk), S + 1, S + 1))
            px, py = xs + 0.5, ys + 0.5
            A, B, C, D = a[chunk], b[chunk], c[chunk], det[chunk]
            l1 = ((B[:, 1] - C[:, 1])[:, None, None] * (px - C[:, 0][:, None, None]) +
                  (C[:, 0] - B[:, 0])[:, None, None] * (py - C[:, 1][:, None, None])) / D[:, None, None]
            l2 = ((C[:, 1] - A[:, 1])[:, None, None] * (px - C[:, 0][:, None, None]) +
                  (A[:, 0] - C[:, 0])[:, None, None] * (py - C[:, 1][:, None, None])) / D[:, None, None]
            l3 = 1 - l1 - l2
            eps = -1e-4
            inside = (l1 >= eps) & (l2 >= eps) & (l3 >= eps) & \
                (xs >= 0) & (xs < res) & (ys >= 0) & (ys < res)
            k, yy, xx = np.nonzero(inside)
            X, Y = xs[k, yy, xx], ys[k, yy, xx]
            tid[Y, X] = chunk[k]
            bary[Y, X] = np.stack([l1[k, yy, xx], l2[k, yy, xx], l3[k, yy, xx]], axis=1)
    return tid, bary


def pad(img, valid):
    """Fill every invalid texel with its nearest valid texel (no seams in mips)."""
    if valid.all() or not valid.any():
        return img
    _, (ri, ci) = ndimage.distance_transform_edt(~valid, return_indices=True)
    return img[ri, ci]


def height_to_normal(r, h, strength=1.0):
    """Tangent-space normal map (OpenGL / glTF convention) from a per-texel
    height (metres), using the UV-space derivative of the 3D position."""
    res = r.res
    H = np.zeros((res, res), np.float32)
    H[r.valid] = h
    Pimg = np.zeros((res, res, 3), np.float32)
    Pimg[r.valid] = r.P
    V = r.valid

    def deriv(axis):
        fwd = np.roll(V, -1, axis) & V
        dh = np.roll(H, -1, axis) - H
        dP = np.linalg.norm(np.roll(Pimg, -1, axis) - Pimg, axis=2)
        texel = np.median(dP[fwd]) if fwd.any() else 1e-3
        good = fwd & (dP < texel * 4) & (dP > 1e-9)
        g = np.where(good, dh / np.maximum(dP, 1e-9), 0.0)
        # central difference: average with the backward derivative
        g_b = np.roll(g, 1, axis)
        good_b = np.roll(good, 1, axis)
        return np.where(good & good_b, (g + g_b) / 2, np.where(good, g, g_b))

    du = deriv(1)   # columns = +U
    dv = deriv(0)   # rows (bottom-up) = +V
    n = np.stack([-du * strength, -dv * strength, np.ones_like(du)], axis=2)
    n /= np.linalg.norm(n, axis=2, keepdims=True)
    return pad((n * 0.5 + 0.5).astype(np.float32), V)


# ----------------------------------------------------------- images

def save_image(name, arr, out_dir, colorspace="sRGB", fmt="PNG"):
    """arr: (H,W,C) float in [0,1], rows bottom-up (Blender convention)."""
    h, w, c = arr.shape
    rgba = np.ones((h, w, 4), np.float32)
    if c == 1:
        rgba[..., :3] = arr
    else:
        rgba[..., :c] = arr[..., :4]
    img = bpy.data.images.new(name, w, h, alpha=(c == 4))
    img.pixels.foreach_set(np.clip(rgba, 0, 1).ravel())
    ext = "png" if fmt == "PNG" else "jpg"
    path = os.path.join(out_dir, "textures", f"{name}.{ext}")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    img.filepath_raw = path
    img.file_format = "PNG" if fmt == "PNG" else "JPEG"
    if fmt != "PNG":
        bpy.context.scene.render.image_settings.quality = 92
    img.save()
    img.source = "FILE"
    img.filepath = path
    img.reload()
    img.colorspace_settings.name = colorspace
    return img


def srgb(c):
    return np.asarray(c, dtype=np.float64)


def mix(a, b, t):
    t = np.asarray(t)[..., None] if np.ndim(t) else t
    return a * (1 - t) + b * t


# --------------------------------------------------------------- AO

def vertex_ao(objs, rays=20, dist=0.18):
    """Per-vertex ambient occlusion by ray casting against all parts (rest pose)."""
    verts, polys, spans = [], [], []
    off = 0
    for o in objs:
        me = o.data
        co = np.empty(len(me.vertices) * 3)
        me.vertices.foreach_get("co", co)
        co = co.reshape(-1, 3)
        verts.append(co)
        polys += [[i + off for i in p.vertices] for p in me.polygons]
        spans.append((off, len(co)))
        off += len(co)
    allv = np.concatenate(verts)
    bvh = BVHTree.FromPolygons([Vector(v) for v in allv], polys)
    rng = np.random.default_rng(5)
    # cosine-weighted hemisphere directions around +Z
    u1, u2 = rng.random(rays), rng.random(rays)
    r = np.sqrt(u1)
    th = 2 * np.pi * u2
    local = np.stack([r * np.cos(th), r * np.sin(th), np.sqrt(1 - u1)], axis=1)
    out = {}
    for o, (start, n) in zip(objs, spans):
        me = o.data
        vn = np.empty(n * 3)
        me.vertices.foreach_get("normal", vn)
        vn = vn.reshape(-1, 3)
        co = allv[start:start + n]
        ao = np.ones(n)
        for i in range(n):
            nz = vn[i]
            t = np.cross(nz, [0, 0, 1] if abs(nz[2]) < 0.9 else [1, 0, 0])
            t /= np.linalg.norm(t)
            b = np.cross(nz, t)
            dirs = local[:, :1] * t + local[:, 1:2] * b + local[:, 2:3] * nz
            origin = co[i] + nz * 0.0015
            hit = 0
            for d in dirs:
                loc, _, _, _ = bvh.ray_cast(Vector(origin), Vector(d), dist)
                if loc is not None:
                    hit += 1
            ao[i] = 1 - hit / rays
        out[o.name] = ao
    return out


# ------------------------------------------------------ skin shader

def skin_maps(r, h, look, ao):
    lm = h.lm
    P, N = r.P, r.N
    zombie = look["zombie"]
    n_tex = len(P)
    base = np.tile(srgb(look["skin"]), (n_tex, 1))

    # large-scale tone variation + fine mottling
    tone = noise.fbm(P, 4.0, 3, seed=1) - 0.5
    mott = noise.fbm(P, 45.0, 3, seed=2) - 0.5
    col = base * (1 + look["skin_var"] * 2.0 * tone[:, None] + 0.06 * mott[:, None])

    # --- facial landmarks (A-pose rest coords)
    def gauss(center, sx, sy, sz):
        d = (P - center) / np.array([sx, sy, sz])
        return np.exp(-(d ** 2).sum(axis=1))

    eye_l, eye_r, mouth, nose = lm.eye_l, lm.eye_r, lm.mouth, lm.nose
    head_w = r.interp(h.W_head)
    face_front = ss(-0.1, -0.5, N[:, 1]) * ss(0.3, 0.7, head_w)
    cheeks = sum(gauss(e + np.array([0.03 * s, -0.005, -0.03]), 0.022, 0.03, 0.02)
                 for e, s in ((eye_l, 1), (eye_r, -1)))
    nose_red = gauss(nose, 0.016, 0.03, 0.02)
    ears = ss(0.058, 0.075, np.abs(P[:, 0])) * ss(0.4, 0.8, head_w) * \
        ss(0.05, 0.0, np.abs(P[:, 2] - lm.eye[2] + 0.005))
    joints = np.zeros(n_tex)
    for key in ("knuckle",):
        joints += r.interp(h.W_knuckle)
    red_mask = np.clip(cheeks * 0.7 + nose_red * 0.08 + ears * 0.6 + joints * 0.25, 0, 1)
    red = srgb((0.78, 0.40, 0.36))
    col = mix(col, red, red_mask * look["redness"] * 0.6)

    # --- lips
    dx = (P[:, 0] - mouth[0]) / 0.027
    dz = (P[:, 2] - mouth[2]) / 0.013
    lips = ss(1.05, 0.75, np.sqrt(dx ** 2 + dz ** 2)) * face_front * (P[:, 1] < mouth[1] + 0.02)
    lip_col = srgb((0.62, 0.36, 0.33)) if not zombie else srgb((0.34, 0.28, 0.30))
    col = mix(col, lip_col, lips * 0.5)
    lip_line = np.exp(-((P[:, 2] - mouth[2]) / 0.0012) ** 2) * ss(1.0, 0.7, np.abs(dx)) * face_front
    col = mix(col, srgb((0.25, 0.12, 0.1)), lip_line * 0.6)

    # --- stubble (jaw, chin, upper lip, cheeks lower half)
    jaw_zone = (ss(lm.nose[2] - 0.002, lm.nose[2] - 0.016, P[:, 2]) *
                ss(lm.mouth[2] - 0.085, lm.mouth[2] - 0.045, P[:, 2]) *
                ss(0.3, 0.7, head_w) * ss(0.075, 0.055, np.abs(P[:, 0])))
    cheek_zone = ss(lm.eye[2] - 0.035, lm.eye[2] - 0.055, P[:, 2]) * \
        ss(0.035, 0.05, np.abs(P[:, 0])) * ss(0.075, 0.06, np.abs(P[:, 0])) * ss(0.3, 0.7, head_w)
    neck_zone = ss(lm.mouth[2] - 0.045, lm.mouth[2] - 0.075, P[:, 2]) * \
        ss(lm.neck[2] + 0.01, lm.neck[2] + 0.05, P[:, 2]) * ss(-0.0, -0.3, N[:, 1])
    beard = np.clip(jaw_zone + cheek_zone * 0.8 + neck_zone * 0.6, 0, 1) * (1 - lips)
    dots = ss(0.6, 0.8, noise.value(P, 2600.0, seed=4))
    shadow = srgb((0.30, 0.27, 0.26))
    col = mix(col, shadow, beard * look["stubble"] * (0.18 + 0.35 * dots))

    # --- eyebrows (arched hair strokes)
    brow = np.zeros(n_tex)
    for e, s in ((eye_l, 1), (eye_r, -1)):
        u = (P[:, 0] - e[0]) * s   # + towards temple
        t = (u + 0.022) / 0.05     # 0 inner .. 1 outer
        arch = lm.brow_z + 0.004 + 0.006 * np.sin(np.clip(t, 0, 1) * np.pi * 0.8) - 0.004 * t
        thick = 0.0085 * (1 - 0.5 * np.clip(t, 0, 1))
        band = ss(thick, thick * 0.4, np.abs(P[:, 2] - arch)) * ss(-0.05, 0.08, t) * ss(1.05, 0.9, t)
        strokes = noise.stretched(P, np.array([s * 0.95, 0, 0.3]) / np.linalg.norm([0.95, 0, 0.3]),
                                  150.0, 1800.0, 2, seed=9 + s)
        core = ss(thick * 0.9, thick * 0.2, np.abs(P[:, 2] - arch))
        brow = np.maximum(brow, band * np.clip(ss(0.3, 0.55, strokes) + core * 0.5, 0, 1) * face_front)
    col = mix(col, srgb(look["brow"]), brow * 0.6)

    # --- eyes: lash line, lids, under-eye
    lash = np.zeros(n_tex)
    under = np.zeros(n_tex)
    for e in (eye_l, eye_r):
        d = np.linalg.norm(P - e, axis=1)
        ring = ss(lm.eye_radius * 1.22, lm.eye_radius * 1.02, d)
        upper = ss(e[2] - 0.002, e[2] + 0.004, P[:, 2])
        lash = np.maximum(lash, ring * (0.35 + 0.65 * upper))
        under = np.maximum(under, gauss(e + np.array([0, -0.004, -0.014]), 0.018, 0.02, 0.008))
    col = mix(col, srgb((0.08, 0.05, 0.045)), lash * face_front * 0.85)
    col = mix(col, col * srgb((0.8, 0.72, 0.78)), under * 0.5)

    # --- scalp under the hair: dark root stubble (blends the hairline)
    hairline = h.meta["hairline"]
    above = hairline.line(P) - P[:, 2]
    scalp = ss(0.004, -0.012, above) * ss(0.4, 0.8, head_w)
    hair_dots = ss(0.45, 0.7, noise.value(P, 1200.0, seed=6))
    col = mix(col, srgb(look["hair"]) * 1.4, scalp * (0.45 + 0.4 * hair_dots))

    # --- fingernails
    nail = r.interp(h.W_nail) * ss(0.45, 0.8, N[:, 2])
    nail = ss(0.35, 0.6, nail)
    nail_col = srgb((0.86, 0.67, 0.62)) if not zombie else srgb((0.45, 0.42, 0.3))
    col = mix(col, nail_col, nail * 0.85)

    # --- veins (subtle on survivor forearms / strong zombie network)
    fore = r.interp(h.W_forearm)
    veins = noise.ridged(P, 18.0, 3, seed=12)
    if zombie:
        vmask = ss(0.35, 0.75, veins) * (0.5 + 0.5 * face_front + fore)
        col = mix(col, srgb((0.28, 0.25, 0.33)), np.clip(vmask, 0, 1) * 0.55)
    else:
        col = mix(col, col * srgb((0.82, 0.86, 0.95)), ss(0.5, 0.85, veins) * fore * 0.4)

    # --- freckles / moles
    moles = ss(0.975, 0.99, noise.value(P, 160.0, seed=21)) * face_front
    col = mix(col, col * 0.55, moles * 0.6)

    # --- dirt / grime
    grime = ss(0.55, 0.8, noise.fbm(P, 9.0, 4, seed=30)) * look["dirt"]
    hands = r.interp(h.W_hands)
    grime = np.clip(grime * (0.35 + 0.8 * hands + 0.3 * face_front), 0, 1)
    col = mix(col, srgb((0.30, 0.24, 0.18)), grime * 0.5)

    height = np.zeros(n_tex)
    rough = np.full(n_tex, 0.52)
    # T-zone shine
    tzone = gauss(nose, 0.02, 0.05, 0.03) + gauss(lm.eye + np.array([0, -0.01, 0.04]), 0.04, 0.05, 0.02)
    rough -= 0.12 * np.clip(tzone, 0, 1)
    rough = rough * (1 - lips) + 0.36 * lips
    rough = rough * (1 - nail) + 0.22 * nail

    if zombie:
        col = zombie_skin(r, h, P, N, col, face_front, hands, look)
        rough += 0.05

    # blood
    if look["blood"]:
        mouth_blood = gauss(mouth + np.array([0, -0.01, -0.03]), 0.035, 0.05, 0.05)
        drips = ss(0.4, 0.8, noise.stretched(P, np.array([0, 0, 1.0]), 25, 220, 2, seed=40))
        chin = mouth_blood * (0.6 + 0.6 * drips) * ss(-0.0, -0.3, N[:, 1])
        spl = ss(0.62, 0.72, noise.fbm(P, 14.0, 4, seed=41)) * (0.4 + hands)
        bm = np.clip(chin + spl, 0, 1) * look["blood"]
        col = mix(col, blood_color(P), bm * 0.9)
        rough = rough * (1 - bm) + 0.25 * bm

    # ambient occlusion baked lightly into albedo (cavities read at distance)
    col = col * (0.55 + 0.45 * ao[:, None])

    # --- height: pores, wrinkles, knuckle creases, lip lines
    pores = (noise.value(P, 1800.0, seed=50) - 0.5) * 0.00010
    pores += (noise.value(P, 700.0, seed=51) - 0.5) * 0.00012
    height += pores * (0.5 + 1.2 * face_front)
    forehead = ss(lm.brow_z + 0.012, lm.brow_z + 0.02, P[:, 2]) * \
        ss(lm.brow_z + 0.055, lm.brow_z + 0.035, P[:, 2]) * face_front
    lines = np.sin(P[:, 2] * 2 * np.pi / 0.0075 + noise.fbm(P, 60, 2, seed=52) * 6)
    age_k = 0.5 if not zombie else 1.4
    height -= forehead * ss(0.6, 1.0, lines) * 0.00018 * age_k
    for e, s in ((eye_l, 1), (eye_r, -1)):
        crow = gauss(e + np.array([0.028 * s, 0.012, 0]), 0.008, 0.02, 0.012)
        ang = np.arctan2(P[:, 2] - e[2], (P[:, 0] - e[0]) * s)
        height -= crow * ss(0.7, 1.0, np.sin(ang * 18)) * 0.00015 * age_k
        naso = gauss(nose + np.array([0.022 * s, 0.01, -0.02]), 0.006, 0.02, 0.02)
        height -= naso * 0.0006 * age_k
    height -= lips * ss(0.6, 1.0, np.sin(P[:, 0] * 2 * np.pi / 0.002)) * 0.00006
    height -= joints * ss(0.7, 1.0, np.sin(P[:, 0] * 2 * np.pi / 0.0025 + P[:, 1] * 900)) * 0.0001
    height += nail * 0.0002
    height -= beard * dots * 0.00004
    if zombie:
        height += h.zombie_height

    col = np.clip(col, 0, 1)
    return col, np.clip(rough, 0.05, 1), height


def zombie_skin(r, h, P, N, col, face_front, hands, look):
    """Decay: desaturation, purple bruising, dark sunken eyes, wounds, bite."""
    lm = h.lm
    grey = col.mean(axis=1, keepdims=True)
    col = col * 0.55 + grey * 0.45
    bruise = ss(0.55, 0.8, noise.fbm(P, 6.0, 4, seed=60))
    col = mix(col, srgb((0.38, 0.30, 0.38)), bruise * 0.5)
    yellow = ss(0.6, 0.8, noise.fbm(P, 5.0, 3, seed=61))
    col = mix(col, srgb((0.62, 0.58, 0.36)), yellow * 0.35)
    # sunken dark eye sockets
    for e in (lm.eye_l, lm.eye_r):
        d = np.linalg.norm((P - e) / np.array([1.0, 1.4, 1.1]), axis=1)
        col = mix(col, srgb((0.22, 0.16, 0.18)), ss(0.028, 0.012, d) * 0.7)
    # wounds: a few open gashes (dark red core, swollen purple rim)
    wound_field = noise.fbm(P, 7.0, 3, seed=62)
    core = ss(0.73, 0.78, wound_field)
    rim = ss(0.66, 0.73, wound_field) * (1 - core)
    col = mix(col, srgb((0.45, 0.22, 0.25)), rim * 0.8)
    col = mix(col, srgb((0.22, 0.03, 0.03)), core)
    # neck bite (left side)
    bite_c = lm.neck + np.array([0.045, -0.02, 0.03])
    bd = np.linalg.norm((P - bite_c) / np.array([1.0, 1.0, 0.8]), axis=1)
    teeth = ss(0.5, 0.8, np.sin(np.arctan2(P[:, 2] - bite_c[2], P[:, 1] - bite_c[1]) * 7))
    bite = ss(0.028, 0.018, bd) * (0.5 + 0.5 * teeth)
    col = mix(col, srgb((0.25, 0.02, 0.02)), bite)
    h.zombie_height = -core * 0.0025 + rim * 0.0008 - bite * 0.003
    return col


def blood_color(P):
    dry = noise.fbm(P, 30.0, 2, seed=70)
    return mix(np.tile(srgb((0.30, 0.02, 0.02)), (len(P), 1)), srgb((0.16, 0.03, 0.02)), dry)


# ------------------------------------------------------ eye shader

def eye_maps(r, h, look):
    lm = h.lm
    P = r.P
    col = np.zeros((len(P), 3))
    for e in (lm.eye_l, lm.eye_r):
        sel = np.linalg.norm(P - e, axis=1) < lm.eye_radius * 1.3
        d = P[sel] - e
        d /= np.linalg.norm(d, axis=1, keepdims=True)
        fwd = np.array([0.0, -1.0, 0.0])
        th = np.degrees(np.arccos(np.clip(d @ fwd, -1, 1)))
        phi = np.arctan2(d[:, 2], d[:, 0])
        iris_r, pupil_r = 27.0, 9.0
        polar = np.stack([np.cos(phi) * 30, np.sin(phi) * 30, th * 0.02], axis=1)
        fibers = noise.fbm(polar * np.array([1, 1, 40]), 1.0, 3, seed=80)
        iris_c = srgb(look["iris"])
        iris = iris_c * (0.65 + 0.7 * fibers[:, None])
        iris = mix(iris, iris_c * 1.6 + 0.05, ss(pupil_r + 6, pupil_r, th) * 0.5)   # collarette
        iris = mix(iris, iris_c * 0.35, ss(iris_r - 4, iris_r, th))                  # limbal ring
        sclera = np.tile(srgb((0.86, 0.82, 0.78)), (len(d), 1))
        veins = ss(0.55, 0.9, noise.ridged(P[sel], 250.0, 2, seed=81)) * ss(40, 85, th)
        sclera = mix(sclera, srgb((0.78, 0.45, 0.42)), veins * (0.5 if not look["zombie"] else 1.0))
        sclera = mix(sclera, sclera * srgb((0.95, 0.85, 0.8)), ss(40, 90, th))
        if look["zombie"]:
            sclera = mix(sclera, srgb((0.75, 0.68, 0.45)), 0.35)
        c = mix(sclera, iris, ss(iris_r + 1.0, iris_r - 1.0, th))
        pupil = ss(pupil_r + 0.8, pupil_r - 0.8, th)
        c = mix(c, srgb((0.02, 0.02, 0.02)) if not look["zombie"] else srgb((0.55, 0.57, 0.55)), pupil)
        if look["zombie"]:  # milky cataract
            c = mix(c, srgb((0.78, 0.8, 0.76)), ss(iris_r + 2, 0, th) * 0.55)
        col[sel] = c
    return np.clip(col, 0, 1)


# --------------------------------------------------- cloth shaders

def _cut_distance(cuts, P, arm=None):
    """Distance (m) from each texel to the nearest garment edge (hem/cuff/collar)."""
    ds = []
    for c in cuts:
        if c.where is None:
            f = (P - c.p) @ c.n if isinstance(c, _outfit().Cut) else c.field(P)
            ds.append(np.abs(f))
        else:
            f = (P - c.p) @ c.n
            on_arm = (P[:, 0] * c.side > 0) & ((arm > 0.3) if arm is not None else True)
            ds.append(np.where(on_arm, np.abs(f), 1.0))
    return np.min(np.stack(ds), axis=0)


def _outfit():
    import outfit
    return outfit


def stitch_line(dist, at, width=0.0007, dash=0.004, P=None):
    line = ss(width * 1.8, width * 0.6, np.abs(dist - at))
    if P is not None:
        s = np.sin((P[:, 0] + P[:, 1] + P[:, 2]) * 2 * np.pi / dash)
        line = line * ss(-0.2, 0.3, s)
    return line


def shirt_maps(r, h, look, ao):
    P, N = r.P, r.N
    lm = h.lm
    n = len(P)
    base = srgb(look["shirt"])
    heather = noise.fbm(P, 300.0, 2, seed=90) - 0.5
    col = np.tile(base, (n, 1)) * (1 + 0.10 * heather[:, None])
    fade = noise.fbm(P, 5.0, 3, seed=91) - 0.5
    col *= (1 + 0.12 * fade[:, None])
    cuts = h.meta["shirt_cuts"]
    dist = _cut_distance(cuts, P, r.interp(h.W_arm))
    # rib collar + hem bands, double stitching
    collar = cuts[1].field(P)
    rib = ss(-0.018, -0.014, collar)
    height = np.zeros(n)
    height += rib * np.sin(np.arctan2(P[:, 0], -P[:, 1] + 0.0) * 260) * 0.00025
    col = mix(col, col * 0.9, rib * 0.5)
    hem_band = ss(0.02, 0.017, dist) * (1 - rib)
    st = stitch_line(dist, 0.018, P=P) + stitch_line(dist, 0.0215, P=P)
    height -= np.clip(st, 0, 1) * 0.00025 + hem_band * 0.0001
    col = mix(col, col * 0.8, np.clip(st, 0, 1) * 0.6)
    # side + shoulder seams
    az = np.degrees(np.arctan2(P[:, 1] - lm.spine2[1], np.abs(P[:, 0])))
    torso = r.interp(h.W_torso)
    side_seam = ss(4, 0, np.abs(az)) * torso
    shoulder_seam = np.exp(-((P[:, 1] - lm.shoulder_l[1] + 0.01) / 0.003) ** 2) * \
        ss(0.6, 0.9, N[:, 2]) * ss(0.06, 0.09, np.abs(P[:, 0]))
    seam = np.clip(side_seam + shoulder_seam, 0, 1)
    height -= seam * 0.0004
    col = mix(col, col * 0.78, seam * 0.5)
    # wrinkles: horizontal compression at waist + armpit folds
    wr = noise.stretched(P, np.array([1.0, 0, 0]), 18.0, 90.0, 3, seed=92)
    waist = ss(h.meta["hem_z"] + 0.18, h.meta["hem_z"] + 0.04, P[:, 2])
    height += (wr - 0.5) * 0.0012 * (0.4 + waist)
    arm = r.interp(h.W_arm)
    fold = noise.stretched(P, np.array([0, 0, 1.0]), 20.0, 70.0, 2, seed=93)
    height += (fold - 0.5) * 0.001 * arm
    # sweat + dirt
    pits = sum(np.exp(-(((P - (getattr(lm, f"shoulder_{s}") + np.array([-0.04 * k, 0.0, -0.09])))
                         / np.array([0.04, 0.05, 0.05])) ** 2).sum(axis=1))
               for s, k in (("l", 1), ("r", -1)))
    col = mix(col, col * 0.78, np.clip(pits, 0, 1) * 0.5 * look["dirt"])
    dirt = ss(0.5, 0.8, noise.fbm(P, 8.0, 4, seed=94)) * look["dirt"]
    col = mix(col, srgb((0.28, 0.23, 0.17)), dirt * 0.55)
    rough = np.full(n, 0.85)
    if look["blood"]:
        splat = ss(0.58, 0.66, noise.fbm(P, 10.0, 5, seed=95))
        collar_soak = ss(0.12, 0.0, P[:, 2] * 0 + np.linalg.norm(P - (lm.neck + [0.04, -0.03, -0.04]), axis=1))
        front = ss(0.0, -0.4, N[:, 1])
        drips = noise.stretched(P, np.array([0, 0, 1.0]), 12, 160, 3, seed=96)
        run = ss(lm.neck[2] - 0.35, lm.neck[2] - 0.05, P[:, 2]) * ss(0.45, 0.7, drips) * front * 0.8
        bm = np.clip(splat + collar_soak * 0.9 + run, 0, 1) * look["blood"]
        col = mix(col, blood_color(P), bm * 0.85)
        rough = rough * (1 - bm) + 0.55 * bm
    col = col * (0.5 + 0.5 * ao[:, None])
    return np.clip(col, 0, 1), rough, height


def pants_maps(r, h, look, ao):
    P, N = r.P, r.N
    lm = h.lm
    n = len(P)
    base = srgb(look["pants"])
    # twill weave (diagonal)
    tw = np.sin((P[:, 2] * 0.8 + P[:, 0] + P[:, 1]) * 2 * np.pi / 0.0016)
    col = np.tile(base, (n, 1)) * (1 + 0.05 * tw[:, None])
    wash = noise.fbm(P, 6.0, 4, seed=100) - 0.5
    col *= (1 + 0.18 * wash[:, None])
    height = tw * 0.00005
    knee_z = lm.knee_l[2]
    knee = ss(0.06, 0.0, np.abs(P[:, 2] - knee_z - 0.01)) * ss(-0.2, -0.6, N[:, 1])
    col = mix(col, col * 1.25 + 0.03, knee * 0.5)                      # worn knees
    # waistband
    wz = h.meta["waist_z"]
    band = ss(wz - 0.042, wz - 0.038, P[:, 2])
    height += band * 0.0006
    st = stitch_line(P[:, 2], wz - 0.036, P=P) + stitch_line(P[:, 2], wz - 0.006, P=P)
    # belt loops
    loops_az = np.degrees(np.arctan2(P[:, 0], -P[:, 1]))
    loopm = sum(ss(3.0, 2.0, np.abs(loops_az - a)) for a in (-110, -60, 60, 110, 180, -180)) * band
    height += loopm * 0.0012
    # fly (front centre, J-stitch)
    fly = ss(0.035, 0.03, np.abs(P[:, 0] + 0.012)) * ss(wz - 0.2, wz - 0.17, P[:, 2]) * (P[:, 1] < 0) * (1 - band)
    st += stitch_line(np.abs(P[:, 0] + 0.012), 0.03, P=P) * ss(wz - 0.2, wz - 0.17, P[:, 2]) * (P[:, 1] < 0) * (1 - band)
    height += fly * 0.0002
    # leg seams (outseam / inseam) via angle around each leg axis
    leg_c = np.where(P[:, 0:1] > 0, lm.hip_l, lm.hip_r)
    ang = np.degrees(np.arctan2(P[:, 1] - leg_c[:, 1], np.abs(P[:, 0] - leg_c[:, 0])))
    below_crotch = P[:, 2] < lm.hips[2] - 0.08
    outseam = ss(3.5, 0.5, np.abs(ang)) * (np.abs(P[:, 0]) > np.abs(leg_c[:, 0])) * below_crotch
    inseam = ss(3.5, 0.5, np.abs(ang)) * (np.abs(P[:, 0]) < np.abs(leg_c[:, 0])) * below_crotch
    height -= np.clip(outseam + inseam, 0, 1) * 0.0005
    st += (outseam + inseam) * 0.6
    # cargo pocket on the outer thigh
    pz0, pz1 = knee_z + 0.10, knee_z + 0.29
    outer = (np.abs(P[:, 0]) > np.abs(leg_c[:, 0])) & below_crotch
    pang = np.abs(ang - 5)
    pocket = ss(pz0, pz0 + 0.004, P[:, 2]) * ss(pz1, pz1 - 0.004, P[:, 2]) * ss(34, 31, pang) * outer
    flap = pocket * ss(pz1 - 0.055, pz1 - 0.05, P[:, 2])
    edge = pocket * (1 - ss(pz0 + 0.004, pz0 + 0.008, P[:, 2]) * ss(pz1 - 0.004, pz1 - 0.008, P[:, 2])
                     * ss(31, 28, pang))
    height += pocket * 0.0015 + flap * 0.0008 - edge * 0.0004
    col = mix(col, col * 0.85, flap * 0.3 + edge * 0.5)
    st += edge * 0.5
    # back pockets
    back = (P[:, 1] > 0) & (P[:, 2] > lm.hips[2] - 0.12) & (P[:, 2] < wz - 0.045)
    bp = ss(0.03, 0.035, np.abs(P[:, 0])) * ss(0.11, 0.105, np.abs(P[:, 0])) * back * \
        ss(lm.hips[2] - 0.12, lm.hips[2] - 0.115, P[:, 2])
    height += bp * 0.0006
    height -= np.clip(st, 0, 1) * 0.0002
    col = mix(col, col * 0.75, np.clip(st, 0, 1) * 0.5)
    # folds: knee bunching + crotch whiskers
    fold = noise.stretched(P, np.array([1.0, 0, 0]), 22.0, 110.0, 3, seed=101)
    bunch = ss(0.1, 0.0, np.abs(P[:, 2] - knee_z)) + ss(h.meta["cuff_z"] + 0.12, h.meta["cuff_z"] + 0.03, P[:, 2])
    height += (fold - 0.5) * 0.0015 * (0.3 + bunch)
    whisk = ss(0.55, 0.85, noise.stretched(P, np.array([0.8, 0, 0.6]), 20, 200, 2, seed=102)) * \
        ss(lm.hips[2] - 0.2, lm.hips[2] - 0.08, P[:, 2]) * ss(lm.hips[2] + 0.02, lm.hips[2] - 0.05, P[:, 2])
    col = mix(col, col * 1.2, whisk * 0.4)
    dirt = ss(0.45, 0.8, noise.fbm(P, 7.0, 4, seed=103)) * look["dirt"]
    dirt = np.clip(dirt * (0.5 + ss(0.45, 0.15, P[:, 2]) + knee), 0, 1)
    col = mix(col, srgb((0.30, 0.25, 0.18)), dirt * 0.6)
    rough = np.full(n, 0.88)
    if look["blood"]:
        bm = ss(0.6, 0.68, noise.fbm(P, 9.0, 5, seed=104)) * look["blood"]
        col = mix(col, blood_color(P), bm * 0.8)
        rough = rough * (1 - bm) + 0.6 * bm
    col = col * (0.5 + 0.5 * ao[:, None])
    return np.clip(col, 0, 1), rough, height


def boots_maps(r, h, look, ao):
    P, N = r.P, r.N
    lm = h.lm
    n = len(P)
    base = srgb(look["boots"])
    grain = noise.value(P, 900.0, seed=110)
    col = np.tile(base, (n, 1)) * (0.9 + 0.2 * noise.fbm(P, 20.0, 3, seed=111)[:, None])
    height = (grain - 0.5) * 0.00012
    rough = 0.62 + 0.2 * (noise.fbm(P, 12.0, 3, seed=112) - 0.5)
    # sole: rubber band + tread
    sole = ss(0.024, 0.02, P[:, 2])
    welt = stitch_line(P[:, 2], 0.027, P=P)
    col = mix(col, srgb((0.07, 0.065, 0.06)), sole)
    rough = rough * (1 - sole) + 0.8 * sole
    tread = ss(0.2, 0.6, np.sin(P[:, 1] * 2 * np.pi / 0.012)) * ss(0.005, 0.0, P[:, 2])
    height += sole * 0.0003 - tread * 0.002 - welt * 0.0003
    col = mix(col, col * 0.6, welt * 0.7)
    # toe cap scuffs + ankle flex creases
    side = np.where(P[:, 0:1] > 0, lm.ankle_l, lm.ankle_r)
    fwd = side[:, 1] - P[:, 1]
    toe = ss(0.10, 0.16, fwd) * (1 - sole)
    col = mix(col, col * 1.18 + 0.01, toe * ss(0.5, 0.8, noise.fbm(P, 60.0, 3, seed=113)) * 0.5)
    crease_zone = ss(0.02, 0.06, fwd) * ss(0.12, 0.07, fwd) * ss(0.03, 0.06, P[:, 2]) * ss(-0.2, 0.5, N[:, 2])
    creases = ss(0.6, 0.95, noise.stretched(P, np.array([1.0, 0, 0]), 30, 250, 2, seed=114))
    height -= crease_zone * creases * 0.0006
    col = mix(col, col * 0.7, crease_zone * creases * 0.5)
    # laces: front of the shaft / instep
    lx = P[:, 0] - side[:, 0]
    lace_zone = (N[:, 1] < -0.25) & (P[:, 2] > 0.06) & (P[:, 2] < h.meta["boot_z"] - 0.005) & (fwd < 0.11)
    lz = lace_zone * ss(0.019, 0.016, np.abs(lx))
    s = P[:, 2] + fwd * 0.6
    period = 0.014
    ph = (s / period) % 1.0
    cross = np.minimum(np.abs(lx / 0.018 - (ph * 2 - 1)), np.abs(lx / 0.018 + (ph * 2 - 1)))
    lace = lz * ss(0.35, 0.15, cross)
    eyelets = lace_zone * ss(0.004, 0.002, np.abs(np.abs(lx) - 0.017)) * ss(0.25, 0.1, np.abs(ph - 0.5))
    tongue = lz * (1 - lace)
    col = mix(col, col * 0.75, tongue * 0.6)
    col = mix(col, srgb((0.08, 0.07, 0.06)), lace)
    col = mix(col, srgb((0.55, 0.52, 0.45)), eyelets)
    height += lace * 0.0015 - tongue * 0.0005 + eyelets * 0.0006
    rough = rough * (1 - eyelets) + 0.25 * eyelets
    # collar padding at the top
    top = ss(h.meta["boot_z"] - 0.02, h.meta["boot_z"] - 0.012, P[:, 2])
    height += top * 0.001
    mud = ss(0.35, 0.7, noise.fbm(P, 10.0, 4, seed=115)) * ss(0.09, 0.0, P[:, 2]) * (0.4 + look["dirt"])
    col = mix(col, srgb((0.26, 0.21, 0.15)), np.clip(mud, 0, 1) * 0.7)
    rough = rough * (1 - mud * 0.5) + 0.9 * mud * 0.5
    col = col * (0.55 + 0.45 * ao[:, None])
    return np.clip(col, 0, 1), np.clip(rough, 0.1, 1), height


def hair_maps(r, h, look, ao):
    P, N = r.P, r.N
    lm = h.lm
    n = len(P)
    # flow direction: back and down from the crown, forward-down at the fringe
    crown = np.array([0.0, lm.head[1] + 0.03, lm.top - 0.01])
    flow = P - crown
    flow[:, 2] = -np.abs(flow[:, 2]) - 0.02
    flow -= (flow * N).sum(axis=1, keepdims=True) * N
    flow /= np.maximum(np.linalg.norm(flow, axis=1, keepdims=True), 1e-6)
    # strands: stretched noise evaluated in a flow-aligned frame
    strands = np.zeros(n)
    for k, (sa, sx) in enumerate(((60, 2500), (30, 900))):
        q = P - (P * flow).sum(axis=1, keepdims=True) * flow
        pos = q * sx + (P * flow).sum(axis=1, keepdims=True) * flow * sa
        strands += noise.fbm(pos, 1.0, 2, seed=120 + k) * (0.6 if k == 0 else 0.4)
    base = srgb(look["hair"])
    col = np.tile(base, (n, 1)) * (0.6 + 0.9 * strands[:, None])
    col = mix(col, base * 2.2 + 0.03, ss(0.62, 0.8, strands) * 0.35)   # highlights
    above = h.meta["hairline"].line(P) - P[:, 2]
    edge = ss(0.006, -0.03, above)
    alpha = np.clip(edge * (0.55 + 0.6 * strands) + ss(-0.014, -0.02, above), 0, 1)
    alpha = np.where(alpha > 0.98, 1.0, alpha)
    if look["zombie"]:
        col = mix(col, srgb((0.30, 0.28, 0.25)), ss(0.55, 0.8, noise.fbm(P, 12, 3, seed=125)) * 0.4)
        col = mix(col, blood_color(P), ss(0.66, 0.74, noise.fbm(P, 9, 4, seed=126)) * 0.8)
    height = (strands - 0.5) * 0.0005
    rough = 0.42 + 0.15 * (1 - strands)
    col = col * (0.6 + 0.4 * ao[:, None])
    return np.clip(np.concatenate([col, alpha[:, None]], axis=1), 0, 1), rough, height


# ---------------------------------------------------------- materials

def principled(mat, color_img=None, rough_img=None, normal_img=None, color=None,
               rough=None, normal_strength=1.0, **extra):
    mat.use_nodes = True
    nt = mat.node_tree
    for nd in list(nt.nodes):
        nt.nodes.remove(nd)
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    out.location = (600, 0)
    bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
    bsdf.location = (250, 0)
    nt.links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])
    y = 300
    if color_img:
        t = nt.nodes.new("ShaderNodeTexImage")
        t.image = color_img
        t.location = (-400, y)
        nt.links.new(t.outputs["Color"], bsdf.inputs["Base Color"])
        if color_img.depth == 32 or extra.get("alpha"):
            nt.links.new(t.outputs["Alpha"], bsdf.inputs["Alpha"])
    elif color is not None:
        bsdf.inputs["Base Color"].default_value = tuple(np.asarray(color) ** 2.2) + (1,)
    y -= 300
    if rough_img:
        t = nt.nodes.new("ShaderNodeTexImage")
        t.image = rough_img
        t.location = (-400, y)
        nt.links.new(t.outputs["Color"], bsdf.inputs["Roughness"])
    elif rough is not None:
        bsdf.inputs["Roughness"].default_value = rough
    y -= 300
    if normal_img:
        t = nt.nodes.new("ShaderNodeTexImage")
        t.image = normal_img
        t.location = (-400, y)
        nm = nt.nodes.new("ShaderNodeNormalMap")
        nm.location = (-100, y)
        nm.inputs["Strength"].default_value = normal_strength
        nt.links.new(t.outputs["Color"], nm.inputs["Color"])
        nt.links.new(nm.outputs["Normal"], bsdf.inputs["Normal"])
    for k, v in extra.items():
        if k in bsdf.inputs:
            bsdf.inputs[k].default_value = v
    if extra.get("alpha"):
        try:
            mat.surface_render_method = "DITHERED"
        except Exception:
            mat.blend_method = "HASHED"
    return bsdf


# -------------------------------------------------------------- paint

def _pack_uvs(obj):
    bpy.context.view_layer.objects.active = obj
    for o in bpy.context.view_layer.objects:
        o.select_set(o == obj)
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.uv.select_all(action="SELECT")
    bpy.ops.uv.pack_islands(rotate=True, margin=0.004)
    bpy.ops.object.mode_set(mode="OBJECT")


def _weights_per_vertex(obj, h, bone_keys):
    """Summed skin weight per vertex of `obj` for bones containing any key."""
    n = len(obj.data.vertices)
    w = np.zeros(n)
    names = {vg.index: vg.name for vg in obj.vertex_groups}
    want = {i for i, nm in names.items() if any(k in nm for k in bone_keys)}
    for v in obj.data.vertices:
        for g in v.groups:
            if g.group in want:
                w[v.index] += g.weight
    return np.clip(w, 0, 1)


def paint(h, parts, spec, out_dir):
    look = spec["look"]
    name = spec["name"]
    mats = h.mats
    for k in ("shirt", "pants", "hair"):
        _pack_uvs(parts[k])
    print("  ambient occlusion...", flush=True)
    ao = vertex_ao(list(parts.values()))

    body = parts["body"]
    h.W_head = _weights_per_vertex(body, h, ["Head"])
    h.W_hands = _weights_per_vertex(body, h, ["Hand"])
    h.W_knuckle = _weights_per_vertex(body, h, ["Index1", "Middle1", "Ring1", "Pinky1",
                                               "Index2", "Middle2", "Ring2", "Pinky2"]) * 0.5
    h.W_nail = _weights_per_vertex(body, h, ["Index3", "Middle3", "Ring3", "Pinky3", "Thumb3"])
    h.W_forearm = _weights_per_vertex(body, h, ["ForeArm"])

    print("  skin...", flush=True)
    r = Raster(body, RES["skin"], 0)
    col, rough, height = skin_maps(r, h, look, r.interp(ao[body.name]))
    ci = save_image(f"{name}_skin_color", r.image(col), out_dir, fmt="JPEG")
    ri = save_image(f"{name}_skin_rough", r.image(rough), out_dir, "Non-Color", fmt="JPEG")
    ni = save_image(f"{name}_skin_normal", height_to_normal(r, height), out_dir, "Non-Color")
    principled(mats["skin"], ci, ri, ni, **{"Subsurface Weight": 1.0,
                                           "Subsurface Scale": 0.006,
                                           "Subsurface Radius": (1.0, 0.35, 0.2),
                                           "Specular IOR Level": 0.45})

    print("  eyes...", flush=True)
    r = Raster(body, RES["eye"], 1)
    ei = save_image(f"{name}_eye_color", r.image(eye_maps(r, h, look)), out_dir, fmt="JPEG")
    principled(mats["eye"], ei, rough=0.04, **{"Coat Weight": 1.0, "Coat Roughness": 0.02})
    principled(mats["teeth"], color=(0.86, 0.83, 0.74) if not look["zombie"] else (0.6, 0.52, 0.36),
               rough=0.25)
    principled(mats["tongue"], color=(0.62, 0.32, 0.32), rough=0.3,
               **{"Subsurface Weight": 0.5, "Subsurface Scale": 0.003})

    for key, fn, strength, extra in (
            ("shirt", shirt_maps, 1.0, {"Sheen Weight": 0.4, "Sheen Roughness": 0.4}),
            ("pants", pants_maps, 1.0, {"Sheen Weight": 0.2}),
            ("boots", boots_maps, 1.0, {}),
            ("hair", hair_maps, 1.0, {"alpha": True, "Sheen Weight": 0.3})):
        print(f"  {key}...", flush=True)
        o = parts[key]
        if key == "shirt":
            h.W_torso = _weights_per_vertex(o, h, ["Spine"])
            h.W_arm = _weights_per_vertex(o, h, ["Arm", "Shoulder"])
        r = Raster(o, RES[key], 0)
        col, rough, height = fn(r, h, look, r.interp(ao[o.name]))
        fmt = "PNG" if col.shape[1] == 4 else "JPEG"
        ci = save_image(f"{name}_{key}_color", r.image(col), out_dir, fmt=fmt)
        ri = save_image(f"{name}_{key}_rough", r.image(np.broadcast_to(rough, (len(r.P),))),
                        out_dir, "Non-Color", fmt="JPEG")
        ni = save_image(f"{name}_{key}_normal", height_to_normal(r, height), out_dir, "Non-Color")
        extra = dict(extra)
        principled(mats[key], ci, ri, ni, **extra)
