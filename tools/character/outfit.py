"""Clothing + hair shells grown out of the MakeHuman body surface.

A garment is a subset of body faces chosen by weight masks and cut planes.
Boundary vertices are snapped onto the cut plane (clean hems instead of a
quad staircase), the shell is Taubin-smoothed (removes toes / nipples / abs
from under the cloth), pushed out along the normal by a per-vertex offset
(loose at the hem, tight at the collar), given low-frequency fold noise and
finally a Solidify for real fabric thickness.  Because every vertex is a body
vertex, the garment inherits the body's skin weights and deforms with it.
Body faces fully hidden under a garment are deleted (no poke-through, fewer
triangles).
"""
import bmesh
import bpy
import numpy as np
from mathutils import Vector

import mh
import noise


# ------------------------------------------------------------ landmarks

class Landmarks:
    def __init__(self, h):
        co = h.co
        g = h.groups
        self.eye_l = co[g["helper-l-eye"]].mean(axis=0)
        self.eye_r = co[g["helper-r-eye"]].mean(axis=0)
        self.eye = (self.eye_l + self.eye_r) / 2
        self.eye_radius = np.linalg.norm(co[g["helper-l-eye"]] - self.eye_l, axis=1).max()
        # MakeHuman's joint-mouth sits at nose height; the lips part between the teeth
        ut, lt = co[g["helper-upper-teeth"]], co[g["helper-lower-teeth"]]
        body_v = g["body"]
        mz = (ut[:, 2].min() + lt[:, 2].max()) / 2
        near = body_v[(np.abs(co[body_v, 0]) < 0.004) & (np.abs(co[body_v, 2] - mz) < 0.012)]
        self.mouth = np.array([0.0, co[near, 1].min(), mz])
        self.neck = np.array(h.joint("joint-neck"))
        self.head = np.array(h.bone_head("mixamorig:Head"))
        self.hips = np.array(h.bone_head("mixamorig:Hips"))
        self.spine2 = np.array(h.bone_head("mixamorig:Spine2"))
        body = g["body"]
        self.top = co[body, 2].max()
        self.body_idx = body
        for s, S in (("l", "Left"), ("r", "Right")):
            setattr(self, f"shoulder_{s}", np.array(h.bone_head(f"mixamorig:{S}Arm")))
            setattr(self, f"elbow_{s}", np.array(h.bone_head(f"mixamorig:{S}ForeArm")))
            setattr(self, f"wrist_{s}", np.array(h.bone_head(f"mixamorig:{S}Hand")))
            setattr(self, f"knee_{s}", np.array(h.bone_head(f"mixamorig:{S}Leg")))
            setattr(self, f"ankle_{s}", np.array(h.bone_head(f"mixamorig:{S}Foot")))
            setattr(self, f"hip_{s}", np.array(h.bone_head(f"mixamorig:{S}UpLeg")))
        # nose tip: most forward head vertex near the midline
        head_v = body[(np.abs(co[body, 0]) < 0.01) & (co[body, 2] > self.mouth[2])
                      & (co[body, 2] < self.eye[2])]
        self.nose = co[head_v[np.argmin(co[head_v, 1])]]
        self.brow_z = self.eye[2] + 0.022


# --------------------------------------------------------------- helpers

def vertex_normals(co, faces):
    n = np.zeros_like(co)
    for f in faces:
        f = np.asarray(f)
        p = co[f]
        # Newell normal (works for quads and tris)
        nx = ((p[:, 1] - np.roll(p[:, 1], -1)) * (p[:, 2] + np.roll(p[:, 2], -1))).sum()
        ny = ((p[:, 2] - np.roll(p[:, 2], -1)) * (p[:, 0] + np.roll(p[:, 0], -1))).sum()
        nz = ((p[:, 0] - np.roll(p[:, 0], -1)) * (p[:, 1] + np.roll(p[:, 1], -1))).sum()
        n[f] += (nx, ny, nz)
    ln = np.linalg.norm(n, axis=1, keepdims=True)
    ln[ln == 0] = 1
    return n / ln


def neighbours(n_verts, faces):
    nb = [set() for _ in range(n_verts)]
    for f in faces:
        k = len(f)
        for i in range(k):
            a, b = f[i], f[(i + 1) % k]
            nb[a].add(b)
            nb[b].add(a)
    return nb


def boundary_verts(faces):
    count = {}
    for f in faces:
        k = len(f)
        for i in range(k):
            e = tuple(sorted((f[i], f[(i + 1) % k])))
            count[e] = count.get(e, 0) + 1
    b = set()
    for e, c in count.items():
        if c == 1:
            b.update(e)
    return b


def taubin(co, faces, idx, iters, fixed=(), lam=0.5, mu=-0.53, weight=None):
    """Taubin (non-shrinking) smoothing of vertices `idx`."""
    co = co.copy()
    nb = neighbours(len(co), faces)
    idx = np.array([i for i in idx if i not in fixed and nb[i]])
    if not len(idx) or iters <= 0:
        return co
    # CSR-ish neighbour lists
    lists = [np.fromiter(nb[i], dtype=np.int64) for i in idx]
    lens = np.array([len(l) for l in lists])
    flat = np.concatenate(lists)
    owner = np.repeat(np.arange(len(idx)), lens)
    w = np.ones(len(idx)) if weight is None else weight[idx]
    for _ in range(iters):
        for f in (lam, mu):
            acc = np.zeros((len(idx), 3))
            np.add.at(acc, owner, co[flat])
            avg = acc / lens[:, None]
            co[idx] += (avg - co[idx]) * f * w[:, None]
    return co


def relax_with_clearance(co, faces, idx, fixed, clearance, bvh, iters):
    """Inflate & relax: alternately Laplacian-smooth the shell and push every
    vertex back out so it keeps `clearance` metres from the body surface.
    Removes toes/abs/nipples from under cloth without ever intersecting."""
    co = co.copy()
    nb = neighbours(len(co), faces)
    free = np.array([i for i in idx if i not in fixed and nb[i]])
    lists = [np.fromiter(nb[i], dtype=np.int64) for i in free]
    lens = np.array([len(l) for l in lists])
    flat = np.concatenate(lists)
    owner = np.repeat(np.arange(len(free)), lens)

    def push():
        for i in idx:
            loc, n, _, _ = bvh.find_nearest(Vector(co[i]))
            if loc is None:
                continue
            d = (Vector(co[i]) - loc).dot(n)
            if d < clearance[i]:
                co[i] += np.array(n) * (clearance[i] - d)

    push()
    for _ in range(max(iters, 1)):
        for _ in range(2):
            acc = np.zeros((len(free), 3))
            np.add.at(acc, owner, co[flat])
            co[free] += (acc / lens[:, None] - co[free]) * 0.6
        push()
    return co


def faces_in(faces, mask):
    return [f for f in faces if mask[f].all()]


# --------------------------------------------------------------- garment

class Cut:
    """Half-space cut: keep where dot(v - point, normal) <= 0.
    `where` limits the cut to a vertex mask (e.g. one arm)."""

    def __init__(self, point, normal, where=None):
        self.p = np.asarray(point, dtype=np.float64)
        n = np.asarray(normal, dtype=np.float64)
        self.n = n / np.linalg.norm(n)
        self.where = where

    def field(self, co):
        f = (co - self.p) @ self.n
        if self.where is not None:
            f = np.where(self.where, f, -1.0)
        return f


class NecklineCut:
    """Crew neck: a ring around the neck, lower at the front, rising towards
    the shoulders so the trapezius stays covered."""

    def __init__(self, neck, drop_front=0.03, radius=0.075):
        self.c = np.asarray(neck)
        self.drop = drop_front
        self.r = radius
        self.n = np.array([0.0, 0.0, 1.0])
        self.where = None

    def field(self, co):
        d = co - self.c
        r = np.hypot(d[:, 0], d[:, 1] * 1.25)
        front = np.clip(-d[:, 1] / 0.09, 0, 1)
        z_cut = self.c[2] - 0.012 - self.drop * front + 1.4 * np.maximum(r - self.r, 0) ** 1.2
        return co[:, 2] - z_cut


class HairlineCut:
    """Keep vertices above an azimuth-dependent hairline (in z)."""

    def __init__(self, center, table):
        self.c = np.asarray(center)
        self.table = np.array(table)  # [(azimuth_deg, z)]
        self.n = np.array([0.0, 0.0, -1.0])
        self.where = None

    def line(self, co):
        d = co - self.c
        az = np.degrees(np.abs(np.arctan2(d[:, 0], -d[:, 1])))
        return np.interp(az, self.table[:, 0], self.table[:, 1])

    def field(self, co):
        return self.line(co) - co[:, 2]


def build_garment(h, name, allowed, cuts, offset, material, smooth=3,
                  thickness=0.003, snap=0.03, delete_margin=0.02, keep_body=None,
                  folds=0.0, fold_scale=8.0, drop_faces=None):
    """Create a garment object; returns (obj, faces, covered_body_face_mask)."""
    co = h.co
    body_faces = h.body_faces
    fields = np.stack([c.field(co) for c in cuts]) if cuts else np.full((1, len(co)), -1.0)
    fmax = fields.max(axis=0)
    inside = allowed & (fmax <= 0)
    faces = [(f, t) for f, t in zip(body_faces, h.body_uvfaces) if inside[f].all()]
    if drop_faces is not None:
        faces = [(f, t) for f, t in faces if not drop_faces(co[f].mean(axis=0))]
    vf = [f for f, _ in faces]
    new = co.copy()
    bset = boundary_verts(vf)
    bidx = np.array(sorted(bset))
    # snap boundary verts onto the nearest active cut
    if len(bidx):
        fb = fields[:, bidx]
        which = fb.argmax(axis=0)
        for k, c in enumerate(cuts):
            sel = bidx[(which == k) & (fb[k] > -snap)]
            if len(sel):
                new[sel] -= fields[k, sel][:, None] * c.n[None, :]
    used = np.unique(np.concatenate([np.asarray(f) for f in vf]))
    target = offset(co, vertex_normals(co, vf)) if callable(offset) else np.full(len(co), offset)
    new = relax_with_clearance(new, vf, used, bset, target, h.body_bvh, smooth)
    if folds:
        nrm = vertex_normals(new, vf)
        fn = noise.fbm(new, fold_scale, 3, seed=abs(hash(name)) % 97) - 0.5
        new = new + nrm * (fn * folds)[:, None]

    obj = mh.make_mesh_object(name, new, vf, [t for _, t in faces], h.uv, h.W, h.bones,
                              materials=[material])
    mh.remove_loose(obj)
    if thickness:
        mod = obj.modifiers.new("Solidify", "SOLIDIFY")
        mod.thickness = thickness
        mod.offset = -1
        mod.use_rim = True
        mod.use_even_offset = False
        mod.use_quality_normals = True
        mod.shell_vertex_group = "_shell_inner"
        mod.rim_vertex_group = "_shell_rim"
        _apply(obj, mod)
    # body faces fully under the garment (with margin) are hidden -> delete
    deep = allowed & (fmax <= -delete_margin)
    if keep_body is not None:
        deep &= ~keep_body
    covered = np.array([deep[f].all() for f in body_faces])
    return obj, covered


def _apply(obj, mod):
    with bpy.context.temp_override(object=obj, active_object=obj, selected_objects=[obj]):
        bpy.ops.object.modifier_apply(modifier=mod.name)


# ------------------------------------------------------------ materials

def material(name):
    m = bpy.data.materials.get(name) or bpy.data.materials.new(name)
    m.use_nodes = True
    return m


# ---------------------------------------------------------------- build

def build(h, spec):
    """Build body + outfit objects. Returns dict of part name -> object."""
    lm = Landmarks(h)
    h.lm = lm
    co = h.co
    W = h.W
    bw = lambda *names: W[:, [h.bones.index("mixamorig:" + n) for n in names]].sum(axis=1)

    h.body_faces = [f for f, t, g in h.faces if g == "body"]
    h.body_uvfaces = [t for f, t, g in h.faces if g == "body"]
    from mathutils.bvhtree import BVHTree
    h.body_bvh = BVHTree.FromPolygons([Vector(v) for v in co], h.body_faces)
    is_body = np.zeros(len(co), bool)
    is_body[lm.body_idx] = True

    hands = bw(*[f"{s}Hand{f}{j}" for s in ("Left", "Right")
                 for f in ("Thumb", "Index", "Middle", "Ring", "Pinky") for j in (1, 2, 3)],
               "LeftHand", "RightHand")
    arms = bw("LeftArm", "RightArm", "LeftForeArm", "RightForeArm") + hands
    legs = bw("LeftUpLeg", "RightUpLeg", "LeftLeg", "RightLeg", "LeftFoot", "RightFoot",
              "LeftToeBase", "RightToeBase")
    feet = bw("LeftFoot", "RightFoot", "LeftToeBase", "RightToeBase")
    head = bw("Head")
    upleg = bw("LeftUpLeg", "RightUpLeg")

    wear = spec["outfit"]
    mats = {k: material(f"{spec['name']}_{k}") for k in ("skin", "eye", "teeth", "tongue",
                                                         "shirt", "pants", "boots", "hair")}
    parts = {}
    covered = np.zeros(len(h.body_faces), bool)
    rng = np.random.default_rng(spec.get("seed", 1))

    tear = None
    if wear.get("torn"):
        tear_fn = lambda p: noise.fbm(p, 5.0, 3, seed=7)
        tear = lambda c: tear_fn(c[None, :])[0] > 0.68
        tear_mask = noise.fbm(co, 5.0, 3, seed=7) > 0.64  # body kept under holes

    # ---------------- shirt (short sleeve crew-neck t-shirt)
    hem_z = lm.hips[2] - wear.get("shirt_hem", 0.07)
    neck_pt = lm.neck + np.array([0, 0, -0.035])
    cuts = [Cut((0, 0, hem_z), (0, 0, -1)),
            NecklineCut(lm.neck)]
    sleeve = wear.get("sleeve", 0.45)
    for s in ("l", "r"):
        sh, el = getattr(lm, f"shoulder_{s}"), getattr(lm, f"elbow_{s}")
        d = (el - sh) / np.linalg.norm(el - sh)
        S = "Left" if s == "l" else "Right"
        side = (co[:, 0] * (1 if s == "l" else -1) > 0) & (bw(f"{S}Arm", f"{S}ForeArm") > 0.3)
        c = Cut(sh + (el - sh) * sleeve, d, where=side)
        c.side = 1 if s == "l" else -1
        cuts.append(c)
    allowed = is_body & (hands < 0.05) & (head < 0.3) & (upleg < 0.8)

    def shirt_offset(p, n):
        belly = noise.smoothstep(lm.hips[2] - 0.1, lm.hips[2] + 0.12, p[:, 2]) * \
            noise.smoothstep(lm.spine2[2] + 0.05, lm.hips[2] + 0.12, p[:, 2])
        front_back = np.clip(np.abs(n[:, 1]), 0, 1)
        o = 0.006 + 0.007 * belly * front_back
        hem = noise.smoothstep(hem_z + 0.12, hem_z, p[:, 2])
        o += 0.008 * hem
        for s in ("l", "r"):
            sh, el = getattr(lm, f"shoulder_{s}"), getattr(lm, f"elbow_{s}")
            d = (el - sh) / np.linalg.norm(el - sh)
            t = ((p - sh) @ d) / np.linalg.norm(el - sh)
            side = p[:, 0] * (1 if s == "l" else -1) > 0.12
            o += np.where(side, 0.009 * noise.smoothstep(0.05, sleeve, t), 0)
        collar = noise.smoothstep(neck_pt[2] - 0.08, neck_pt[2], p[:, 2])
        o = o * (1 - 0.4 * collar)
        # hang over the pants waistband: stay outside the pants shell
        over_pants = noise.smoothstep(lm.hips[2] + 0.13, lm.hips[2] + 0.07, p[:, 2])
        return np.maximum(o, 0.022 * over_pants)

    shirt, cov = build_garment(h, "Shirt", allowed, cuts, shirt_offset, mats["shirt"],
                               smooth=6, thickness=0.0035, folds=0.006, fold_scale=9,
                               keep_body=tear_mask if tear else None, drop_faces=tear)
    parts["shirt"] = shirt
    h.meta = {"shirt_cuts": cuts, "hem_z": hem_z}
    covered |= cov

    # ---------------- cargo pants
    waist_z = lm.hips[2] + 0.075
    boot_top = lm.ankle_l[2] + 0.15
    cuff_z = boot_top - 0.035
    cuts = [Cut((0, 0, waist_z), (0, 0, 1)), Cut((0, 0, cuff_z), (0, 0, -1))]
    allowed = is_body & (arms < 0.05) & (head < 0.05)

    def pants_offset(p, n):
        knee_z = lm.knee_l[2]
        t = noise.smoothstep(waist_z, knee_z, p[:, 2])
        o = 0.007 + 0.010 * t
        # blousing just above the boot, tight where tucked inside it
        o += 0.010 * noise.smoothstep(knee_z, boot_top + 0.05, p[:, 2])
        o *= noise.smoothstep(boot_top - 0.035, boot_top + 0.02, p[:, 2]) * 0.8 + 0.2
        # cargo pockets bulge on the outer thigh
        thigh = noise.smoothstep(0.03, 0.09, np.abs(p[:, 0])) * \
            np.exp(-((p[:, 2] - (knee_z + 0.18)) / 0.08) ** 2) * (np.abs(n[:, 0]) > 0.5)
        o += 0.008 * thigh
        return o

    pants, cov = build_garment(h, "Pants", allowed, cuts, pants_offset, mats["pants"],
                               smooth=8, thickness=0.004, folds=0.008, fold_scale=7,
                               keep_body=tear_mask if tear else None, drop_faces=tear)
    parts["pants"] = pants
    h.meta.update(pants_cuts=cuts, waist_z=waist_z, cuff_z=cuff_z)
    covered |= cov

    # ---------------- boots (voxel-closed shell: toes merge into a toe box)
    boot_z = lm.ankle_l[2] + 0.15
    boots = build_boots(h, lm, boot_z, feet_w=legs, mat=mats["boots"])
    parts["boots"] = boots
    h.meta["boot_z"] = boot_z
    deep = is_body & (legs > 0.5) & (co[:, 2] < boot_z - 0.02)
    covered |= np.array([deep[f].all() for f in h.body_faces])

    # ---------------- hair (short crew cut)
    hc = np.array([0.0, lm.head[1], lm.eye[2] + 0.04])
    eye_z, nape = lm.eye[2], lm.neck[2]
    hl = wear.get("hairline", 0.0)
    table = [(0, lm.brow_z + 0.050 + hl), (25, lm.brow_z + 0.046 + hl),
             (45, lm.brow_z + 0.034 + hl), (62, eye_z + 0.020), (75, eye_z - 0.005),
             (84, eye_z + 0.012), (98, eye_z + 0.030), (112, eye_z + 0.018),
             (128, eye_z - 0.010), (150, nape + 0.065), (180, nape + 0.055)]
    hair_cut = HairlineCut(hc, table)
    allowed = is_body & (head + bw("Neck") > 0.5) & (co[:, 2] > lm.neck[2] + 0.03) & \
        (np.abs(co[:, 0]) < 0.12)
    top = lm.top
    hair_len = wear.get("hair_len", 0.012)

    def hair_offset(p, n):
        above = p[:, 2] - hair_cut.line(p)
        o = 0.0008 + hair_len * noise.smoothstep(0.0, 0.07, above) * \
            noise.smoothstep(top - 0.20, top - 0.02, p[:, 2])
        return o + 0.002 * noise.smoothstep(0.005, 0.03, above)

    bald = None
    if wear.get("bald_patches"):
        bald = lambda c: noise.fbm(c[None, :], 14.0, 2, seed=3)[0] > 0.66
    hair, _ = build_garment(h, "Hair", allowed, [hair_cut], hair_offset, mats["hair"],
                            smooth=4, thickness=0.0012, snap=0.02, folds=0.004,
                            fold_scale=30, drop_faces=bald)
    parts["hair"] = hair
    h.meta["hairline"] = hair_cut

    # ---------------- body (skin + eyes + teeth + tongue)
    face_list, uv_list, mat_idx = [], [], []
    order = {"skin": 0, "eye": 1, "teeth": 2, "tongue": 3}
    bi = 0
    for f, t, g in h.faces:
        kind = mh.KEEP_GROUPS.get(g, None) if g in mh.KEEP_GROUPS else None
        if g == "body":
            if covered[bi]:
                bi += 1
                continue
            bi += 1
        if kind is None:
            continue
        face_list.append(f)
        uv_list.append(t)
        mat_idx.append(order[kind])
    body = mh.make_mesh_object("Body", co, face_list, uv_list, h.uv, h.W, h.bones,
                               mat_index=mat_idx,
                               materials=[mats["skin"], mats["eye"], mats["teeth"],
                                          mats["tongue"]])
    mh.remove_loose(body)
    parts["body"] = body
    h.obj = body
    h.mats = mats

    for name, o in parts.items():
        mh.parent_to_rig(o, h.armature)
        sub = o.modifiers.new("Subdivision", "SUBSURF")
        sub.levels = 1
        sub.render_levels = 2
        o.modifiers.move(len(o.modifiers) - 1, 0)  # smooth before skinning
    return parts


def build_boots(h, lm, boot_z, feet_w, mat, voxel=0.0025):
    """Boots via signed distance field closing + marching cubes.

    1. SDF of the foot surface from the nearest body vertex (+ normal sign),
    2. morphological closing (dilate R, erode R-offset) fills the toe gaps,
    3. flat sole clip, bigger toe box, marching cubes, relax, decimate,
    4. weights copied from the nearest foot vertex.
    """
    from scipy import ndimage
    from scipy.spatial import cKDTree
    from skimage import measure

    co = h.co
    nrm_all = vertex_normals(co, h.body_faces)
    objs = []
    for sgn, S in ((1, "Left"), (-1, "Right")):
        sel = np.nonzero((feet_w > 0.3) & (co[:, 2] < boot_z + 0.06) & (co[:, 0] * sgn > 0.0))[0]
        sel = np.intersect1d(sel, lm.body_idx)
        pts, nrm = co[sel], nrm_all[sel]
        lo = pts.min(axis=0) - 0.035
        hi = pts.max(axis=0) + 0.035
        lo[2] = -0.004
        shape = np.ceil((hi - lo) / voxel).astype(int) + 1
        gx, gy, gz = [lo[i] + np.arange(shape[i]) * voxel for i in range(3)]
        G = np.stack(np.meshgrid(gx, gy, gz, indexing="ij"), axis=-1).reshape(-1, 3)
        tree = cKDTree(pts)
        d, i = tree.query(G, k=4)
        sign = np.einsum("nkj,nkj->nk", G[:, None, :] - pts[i], nrm[i]).mean(axis=1)
        inside = (sign < 0) & (d[:, 0] < 0.05)
        inside = inside.reshape(shape)
        inside = ndimage.binary_fill_holes(inside)
        # closing radius (m) and final offset from the foot
        R, off = 0.032, 0.010
        dil = ndimage.distance_transform_edt(~inside) * voxel <= R
        sdf = (ndimage.distance_transform_edt(dil) - ndimage.distance_transform_edt(~dil)) * voxel
        # toe box + heel counter a bit roomier; everything above boot top removed
        Gs = G.reshape(tuple(shape) + (3,))
        toe = noise.smoothstep(lm.ankle_l[1] - 0.06, lm.ankle_l[1] - 0.16, Gs[..., 1])
        sole = noise.smoothstep(0.03, 0.018, Gs[..., 2])
        level = (R - off) - 0.007 * toe - 0.004 * sole
        field = sdf - level
        field = np.minimum(field, boot_z - Gs[..., 2])        # open-top cut
        field = np.minimum(field, Gs[..., 2] - 0.0)            # flat sole on floor
        field = ndimage.gaussian_filter(field, 1.6)
        verts, faces, _, _ = measure.marching_cubes(field, 0.0, spacing=(voxel,) * 3)
        verts += lo
        me = bpy.data.meshes.new(f"Boot_{S}")
        me.from_pydata(verts.tolist(), [], faces.tolist())
        me.update()
        o = bpy.data.objects.new(f"Boot_{S}", me)
        bpy.context.scene.collection.objects.link(o)
        # the marching-cubes cap at the top is removed -> open shaft
        bm = bmesh.new()
        bm.from_mesh(me)
        top = [f for f in bm.faces if f.calc_center_median().z > boot_z - voxel * 1.5
               and f.normal.z > 0.5]
        bmesh.ops.delete(bm, geom=top, context="FACES")
        bm.to_mesh(me)
        bm.free()
        for p in me.polygons:
            p.use_smooth = True
        dec = o.modifiers.new("Decimate", "DECIMATE")
        dec.ratio = min(1.0, 2600 / max(len(me.polygons), 1))
        _apply(o, dec)
        sm = o.modifiers.new("Smooth", "SMOOTH")
        sm.factor = 0.5
        sm.iterations = 4
        _apply(o, sm)
        sol = o.modifiers.new("Solidify", "SOLIDIFY")
        sol.thickness = 0.004
        sol.offset = -1
        _apply(o, sol)
        objs.append(o)
    # join both boots
    with bpy.context.temp_override(active_object=objs[0], selected_editable_objects=objs,
                                   object=objs[0]):
        bpy.ops.object.join()
    boots = objs[0]
    boots.name = "Boots"
    boots.data.materials.append(mat)
    with bpy.context.temp_override(active_object=boots, object=boots, selected_objects=[boots],
                                   edit_object=boots):
        bpy.context.view_layer.objects.active = boots
        bpy.ops.object.mode_set(mode="EDIT")
        bpy.ops.mesh.select_all(action="SELECT")
        bpy.ops.uv.smart_project(angle_limit=1.15, island_margin=0.01)
        bpy.ops.object.mode_set(mode="OBJECT")
    # weights from nearest foot/leg body vertex
    bv = np.empty(len(boots.data.vertices) * 3)
    boots.data.vertices.foreach_get("co", bv)
    bv = bv.reshape(-1, 3)
    cand = np.intersect1d(np.nonzero(feet_w > 0.3)[0], lm.body_idx)
    _, nn = cKDTree(co[cand]).query(bv)
    Wb = np.zeros((len(co), h.W.shape[1]))
    Wb = h.W[cand[nn]]
    mh.set_weights(boots, Wb, h.bones)
    return boots


def _flatten_sole(obj):
    me = obj.data
    co = np.empty(len(me.vertices) * 3)
    me.vertices.foreach_get("co", co)
    co = co.reshape(-1, 3)
    low = co[:, 2] < 0.028
    # widen the sole edge a little (boots are wider than feet)
    for sgn in (1, -1):
        side = low & (co[:, 0] * sgn > 0)
        if side.any():
            cx = np.median(co[side, 0])
            co[side, 0] += (co[side, 0] - cx) * 0.12
    co[:, 2] = np.where(co[:, 2] < 0.012, 0.0, co[:, 2])
    co[low, 2] = np.maximum(co[low, 2] - 0.004, 0.0)
    me.vertices.foreach_set("co", co.ravel())
    me.update()
