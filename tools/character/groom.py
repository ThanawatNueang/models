"""Procedural hair groom -> Blender Curves (hair) object.

Studio-groom workflow in code:
  1. roots are scattered on the outer shell of the hair cap (area weighted),
  2. a *length map* and a *flow field* define the style (e.g. short faded
     sides, longer textured top with a lifted front = "textured quiff"),
  3. ~3k guide strands are grown step by step: they start lifted off the
     scalp, bend towards the flow direction and are kept above the scalp by
     collision against the cap (with a random layer height -> volume),
  4. ~45k child strands are interpolated from the nearest guide with
     clumping (children pulled towards the guide towards the tip) and frizz.
The curves object is parented to the Head bone, so it follows the rig.
Games get the textured mesh cap (glTF/FBX have no strands).
"""
import bpy
import numpy as np
from mathutils import Matrix, Vector
from mathutils.bvhtree import BVHTree
from scipy.spatial import cKDTree

from noise import smoothstep as ss
import noise
import strands


def _outer_triangles(cap):
    me = cap.data
    me.calc_loop_triangles()
    n = len(me.loop_triangles)
    vi = np.empty(n * 3, np.int64)
    me.loop_triangles.foreach_get("vertices", vi)
    vi = vi.reshape(-1, 3)
    co = np.empty(len(me.vertices) * 3)
    me.vertices.foreach_get("co", co)
    co = co.reshape(-1, 3)
    vn = np.empty(len(me.vertices) * 3)
    me.vertices.foreach_get("normal", vn)
    vn = vn.reshape(-1, 3)
    inner = np.zeros(len(co))
    for name in ("_shell_inner", "_shell_rim"):
        vg = cap.vertex_groups.get(name)
        if vg is None:
            continue
        for v in me.vertices:
            for g in v.groups:
                if g.group == vg.index:
                    inner[v.index] = max(inner[v.index], g.weight)
    keep = inner[vi].max(axis=1) < 0.5
    return co, vn, vi[keep]


def _scatter(co, vn, tris, n, rng):
    a, b, c = co[tris[:, 0]], co[tris[:, 1]], co[tris[:, 2]]
    area = np.linalg.norm(np.cross(b - a, c - a), axis=1) / 2
    pick = rng.choice(len(tris), n, p=area / area.sum())
    u, v = rng.random(n), rng.random(n)
    flip = u + v > 1
    u[flip], v[flip] = 1 - u[flip], 1 - v[flip]
    w = 1 - u - v
    t = tris[pick]
    p = co[t[:, 0]] * w[:, None] + co[t[:, 1]] * u[:, None] + co[t[:, 2]] * v[:, None]
    nn = vn[t[:, 0]] * w[:, None] + vn[t[:, 1]] * u[:, None] + vn[t[:, 2]] * v[:, None]
    nn /= np.linalg.norm(nn, axis=1, keepdims=True)
    return p, nn


def style_fields(p, n, h, style, rng):
    """Length (m), lift (0..1), flow direction (tangent) per root."""
    lm = h.lm
    eye_z = lm.eye[2]
    side_z = eye_z + 0.05                         # fade line: short below, long above
    top = ss(side_z, side_z + 0.04, p[:, 2])
    front = ss(lm.head[1] + 0.03, lm.head[1] - 0.08, p[:, 1])
    above = h.meta["hairline"].line(p) - p[:, 2]
    edge = ss(0.0, -0.015, above)                  # short at the hairline
    if style == "messy":
        L = (0.02 + 0.03 * rng.random(len(p))) * (0.4 + 0.6 * edge)
        lift = 0.1 + 0.2 * rng.random(len(p))
        flow = np.stack([rng.normal(0, 0.6, len(p)), rng.normal(0.3, 0.6, len(p)),
                         -np.ones(len(p))], axis=1)
    else:  # textured quiff: 6-8 mm faded sides, ~5 cm on top, lifted front
        L = 0.007 + (0.042 + 0.012 * front) * top
        L = L * (0.35 + 0.65 * edge) * (0.85 + 0.3 * rng.random(len(p)))
        lift = 0.10 + 0.30 * front * top + 0.10 * rng.random(len(p)) * top
        back = np.array([0.18, 1.0, 0.25])          # swept back, slightly to one side
        down = np.array([0.0, 0.45, -1.0])
        flow = back[None, :] * top[:, None] + down[None, :] * (1 - top[:, None])
        # fringe goes up then back
        flow[:, 2] += 0.8 * front * top
    flow -= (flow * n).sum(axis=1, keepdims=True) * n
    flow /= np.maximum(np.linalg.norm(flow, axis=1, keepdims=True), 1e-6)
    return L, lift, flow


def grow(p, n, L, lift, flow, bvh, layer, segs, gravity):
    """Grow strands (vectorised over strands, collision per point)."""
    N = len(p)
    pts = np.zeros((N, segs + 1, 3))
    pts[:, 0] = p
    d = n * lift[:, None] + flow * (1 - lift[:, None])
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    step = (L / segs)[:, None]
    min_h = 0.0015 + layer
    for k in range(1, segs + 1):
        t = k / segs
        q = pts[:, k - 1] + d * step
        # collision: stay `min_h` above the cap
        for i in range(N):
            loc, nrm, _, _ = bvh.find_nearest(Vector(q[i]))
            if loc is None:
                continue
            off = (Vector(q[i]) - loc).dot(nrm)
            want = min_h[i] * min(1.0, t * 3)
            if off < want:
                q[i] += np.array(nrm) * (want - off)
        pts[:, k] = q
        # comb: bend towards the flow and back down onto the head; lifted
        # strands (the front quiff) hold their lift longer
        hold = lift[:, None]
        d = d + flow * (1.3 / segs * 2) - n * (0.9 / segs * 2) * (1 - hold) * t \
            + np.array([0, 0, -gravity]) * t
        d /= np.linalg.norm(d, axis=1, keepdims=True)
    return pts


def _resample(guide, ratio):
    """Evaluate a guide polyline at fraction ratio*t (extrapolating past the tip)."""
    segs = guide.shape[1] - 1
    t = np.linspace(0, 1, segs + 1)[None, :] * ratio[:, None] * segs
    i0 = np.clip(np.floor(t).astype(int), 0, segs - 1)
    f = t - i0
    a = np.take_along_axis(guide, i0[..., None].repeat(3, 2), 1)
    b = np.take_along_axis(guide, (i0 + 1)[..., None].repeat(3, 2), 1)
    return a + (b - a) * f[..., None]


def build(h, parts, look, style="textured_quiff", n_guides=3000, n_children=45000, segs=10,
          seed=7):
    cap = parts["hair"]
    rig = h.armature
    rng = np.random.default_rng(seed)
    co, vn, tris = _outer_triangles(cap)
    bvh = BVHTree.FromPolygons([Vector(v) for v in co], tris.tolist())

    gp, gn = _scatter(co, vn, tris, n_guides, rng)
    L, lift, flow = style_fields(gp, gn, h, style, rng)
    top = ss(h.lm.eye[2] + 0.05, h.lm.eye[2] + 0.09, gp[:, 2])
    layer = rng.random(n_guides) * (0.003 + 0.012 * top)
    gravity = 0.25 if style == "messy" else 0.08
    guides = grow(gp, gn, L, lift, flow, bvh, layer, segs, gravity)

    cp, cn = _scatter(co, vn, tris, n_children, rng)
    Lc, _, _ = style_fields(cp, cn, h, style, rng)
    _, gi = cKDTree(gp).query(cp)
    ratio = np.clip(Lc / np.maximum(L[gi], 1e-4), 0.3, 1.4)
    g = _resample(guides[gi], ratio)
    t = np.linspace(0, 1, segs + 1)[None, :, None]
    clump = 0.55 if style != "messy" else 0.35
    offset = (cp - gp[gi])[:, None, :]
    pts = g + offset * (1 - clump * t)
    frizz = rng.normal(0, 1, pts.shape) * 0.0009 * t ** 1.5
    wave = noise.fbm(pts.reshape(-1, 3), 60.0, 2, seed=9).reshape(len(cp), segs + 1)[..., None] - 0.5
    pts = pts + frizz + cn[:, None, :] * wave * 0.003 * t
    # final collision for children (keep roots on the surface)
    flat = pts[:, 1:].reshape(-1, 3)
    for i in range(len(flat)):
        loc, nrm, _, _ = bvh.find_nearest(Vector(flat[i]))
        if loc is not None:
            off = (Vector(flat[i]) - loc).dot(nrm)
            if off < 0.001:
                flat[i] += np.array(nrm) * (0.001 - off)
    pts[:, 1:] = flat.reshape(len(cp), segs, 3)
    pts[:, 0] = cp

    allpts = np.concatenate([guides, pts])
    hc = bpy.data.hair_curves.new("HairStrands")
    hc.add_curves([segs + 1] * len(allpts))
    hc.position_data.foreach_set("vector", allpts.astype(np.float32).ravel())
    rad = hc.attributes.get("radius") or hc.attributes.new("radius", "FLOAT", "POINT")
    r = np.linspace(0.000055, 0.00002, segs + 1)
    rad.data.foreach_set("value", np.tile(r, len(allpts)).astype(np.float32))
    mat = strands.hair_material(f"{rig.name}_hair_strands", look)
    hc.materials.append(mat)
    obj = bpy.data.objects.new("HairStrands", hc)
    bpy.context.scene.collection.objects.link(obj)
    # follow the head bone
    bone = rig.data.bones["mixamorig:Head"]
    obj.parent = rig
    obj.parent_type = "BONE"
    obj.parent_bone = bone.name
    obj.matrix_parent_inverse = (rig.matrix_world @ bone.matrix_local @
                                 Matrix.Translation((0, bone.length, 0))).inverted()
    return obj
