"""MakeHuman base mesh -> shaped, rigged Blender human.

Everything here reads the CC0 MakeHuman data in ./makehuman/.
"""
import gzip
import json
import math
import os

import bmesh
import bpy
import numpy as np
from mathutils import Quaternion, Vector

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "makehuman")

# Face groups of base.obj that become real geometry. Everything else
# (joint cubes, tights/skirt/hair helpers) is only used for measurements.
KEEP_GROUPS = {
    "body": "skin",
    "helper-l-eye": "eye",
    "helper-r-eye": "eye",
    "helper-upper-teeth": "teeth",
    "helper-lower-teeth": "teeth",
    "helper-tongue": "tongue",
    "helper-genital": None,  # dropped: covered by clothing
}


class Human:
    """Container for everything derived from the base mesh."""

    def __init__(self):
        self.co = None          # (N,3) blender coords (m, Z up), all MH verts
        self.faces = []         # list of (verts, uvs, group)
        self.uv = None          # (M,2)
        self.groups = {}        # group name -> np.array of vertex indices
        self.obj = None         # body mesh object
        self.armature = None
        self.weights = {}       # bone -> (idx array, weight array)
        self.W = None           # (N, B) dense weights
        self.bones = []

    def joint(self, cube):
        idx = self.groups[cube]
        return Vector(self.co[idx].mean(axis=0))

    def bone_head(self, bone):
        return Vector(self.armature.data.bones[bone].head_local)

    def bone_tail(self, bone):
        return Vector(self.armature.data.bones[bone].tail_local)


# --------------------------------------------------------------------- OBJ

def load_obj(human):
    verts, uvs, faces = [], [], []
    group = None
    groups = {}
    with open(os.path.join(DATA, "base.obj")) as f:
        for line in f:
            if line.startswith("v "):
                verts.append([float(t) for t in line.split()[1:4]])
            elif line.startswith("vt "):
                uvs.append([float(t) for t in line.split()[1:3]])
            elif line.startswith("g "):
                group = line.split()[1]
            elif line.startswith("f "):
                vs, ts = [], []
                for tok in line.split()[1:]:
                    p = tok.split("/")
                    vs.append(int(p[0]) - 1)
                    ts.append(int(p[1]) - 1 if len(p) > 1 and p[1] else 0)
                faces.append((vs, ts, group))
                groups.setdefault(group, set()).update(vs)
    human.mh_co = np.array(verts, dtype=np.float64)
    human.uv = np.array(uvs, dtype=np.float64)
    human.faces = faces
    human.groups = {k: np.array(sorted(v)) for k, v in groups.items()}


# ----------------------------------------------------------------- targets

def _load_target(path):
    idx, d = [], []
    with gzip.open(path, "rt") as f:
        for line in f:
            if not line.strip() or line[0] == "#":
                continue
            p = line.split()
            idx.append(int(p[0]))
            d.append([float(p[1]), float(p[2]), float(p[3])])
    return np.array(idx, dtype=np.int64), np.array(d)


def _pair(v, lo, mid, hi):
    """MakeHuman macro slider split: 0..0.5..1 -> {lo, mid, hi} weights."""
    if v < 0.5:
        a = (0.5 - v) * 2
        return {lo: a, mid: 1 - a}
    a = (v - 0.5) * 2
    return {mid: 1 - a, hi: a}


def macro_targets(body):
    """Return [(relative target path, weight)] like MakeHuman's MacroModifier."""
    g = body["gender"]
    genders = {"female": 1 - g, "male": g}
    a = body["age"]  # 0.5 = 25 years, 1.0 = 90 years
    ages = {"young": 1.0} if a <= 0.5 else {"young": 1 - (a - 0.5) * 2,
                                           "old": (a - 0.5) * 2}
    muscles = _pair(body["muscle"], "minmuscle", "averagemuscle", "maxmuscle")
    weights = _pair(body["weight"], "minweight", "averageweight", "maxweight")
    h = body["height"]
    heights = {"maxheight": (h - 0.5) * 2} if h >= 0.5 else {"minheight": (0.5 - h) * 2}
    p = body["proportions"]
    props = ({"idealproportions": (p - 0.5) * 2} if p >= 0.5
             else {"uncommonproportions": (0.5 - p) * 2})
    out = []
    for gn, gw in genders.items():
        for an, aw in ages.items():
            for race, rw in body["race"].items():
                out.append((f"{race}-{gn}-{an}.target.gz", rw * gw * aw))
            for mn, mw in muscles.items():
                for wn, ww in weights.items():
                    base = gw * aw * mw * ww
                    out.append((f"universal-{gn}-{an}-{mn}-{wn}.target.gz", base))
                    for hn, hw in heights.items():
                        out.append((f"height/{gn}-{an}-{mn}-{wn}-{hn}.target.gz", base * hw))
                    for pn, pw in props.items():
                        out.append((f"proportions/{gn}-{an}-{mn}-{wn}-{pn}.target.gz", base * pw))
    return [(t, w) for t, w in out if w > 1e-4]


def apply_targets(human, body):
    co = human.mh_co.copy()
    for rel, w in macro_targets(body):
        path = os.path.join(DATA, "targets", rel)
        if not os.path.exists(path):
            print("  missing target", rel)
            continue
        idx, d = _load_target(path)
        if len(idx):
            co[idx] += d * w
    # detail targets (face / neck / torso shaping): {"chin/chin-prominent-incr": 0.3}
    for rel, w in body.get("detail", {}).items():
        path = os.path.join(DATA, "targets", "detail", rel + ".target.gz")
        if not os.path.exists(path):
            print("  missing detail target", rel)
            continue
        idx, d = _load_target(path)
        if len(idx):
            co[idx] += d * w
    # MakeHuman: decimetres, Y up, face towards +Z.  Blender: metres, Z up, face -Y.
    bl = np.empty_like(co)
    bl[:, 0] = co[:, 0] * 0.1
    bl[:, 1] = -co[:, 2] * 0.1
    bl[:, 2] = co[:, 1] * 0.1
    body_idx = human.groups["body"]
    bl[:, 2] -= bl[body_idx, 2].min()
    human.co = bl


# --------------------------------------------------------------------- rig

def build_armature(human, name):
    rig_def = json.load(open(os.path.join(DATA, "rig", "rig.mixamo.json")))["bones"]

    def pos(spec):
        if spec["strategy"] == "CUBE":
            return human.joint(spec["cube_name"])
        return Vector(human.co[spec["vertex_indices"]].mean(axis=0))

    arm = bpy.data.armatures.new(f"{name}_rig")
    arm.display_type = "OCTAHEDRAL"
    rig = bpy.data.objects.new(f"{name}", arm)
    bpy.context.scene.collection.objects.link(rig)
    bpy.context.view_layer.objects.active = rig
    rig.select_set(True)
    bpy.ops.object.mode_set(mode="EDIT")
    for bname, b in rig_def.items():
        eb = arm.edit_bones.new(bname)
        eb.head = pos(b["head"])
        eb.tail = pos(b["tail"])
        eb.roll = b.get("roll", 0.0)
    for bname, b in rig_def.items():
        if b["parent"]:
            eb = arm.edit_bones[bname]
            eb.parent = arm.edit_bones[b["parent"]]
            eb.use_connect = b.get("use_connect", False)
    bpy.ops.object.mode_set(mode="OBJECT")
    human.armature = rig
    human.bones = list(rig_def)

    wdef = json.load(open(os.path.join(DATA, "rig", "weights.mixamo.json")))["weights"]
    n = len(human.co)
    W = np.zeros((n, len(human.bones)))
    for bi, bname in enumerate(human.bones):
        pairs = wdef.get(bname, [])
        if pairs:
            a = np.array(pairs)
            W[a[:, 0].astype(int), bi] = a[:, 1]
    s = W.sum(axis=1, keepdims=True)
    s[s == 0] = 1
    human.W = W / s


def bone_weight(human, *names):
    """Summed normalized weight per vertex of all bones whose name contains any key."""
    cols = [i for i, b in enumerate(human.bones) if any(k in b for k in names)]
    return human.W[:, cols].sum(axis=1)


# -------------------------------------------------------------------- mesh

def make_mesh_object(name, co, faces, uv_faces, uv, W, bones, mat_index=None,
                     materials=()):
    """Create a mesh keeping ALL vertices (indices == MH indices); caller
    removes loose verts at the very end."""
    me = bpy.data.meshes.new(name)
    me.from_pydata(co.tolist(), [], [list(f) for f in faces])
    me.update()
    uvl = me.uv_layers.new(name="UVMap")
    loop_uv = np.concatenate([uv[np.array(t)] for t in uv_faces]) if uv_faces else np.zeros((0, 2))
    uvl.data.foreach_set("uv", loop_uv.ravel())
    for m in materials:
        me.materials.append(m)
    if mat_index is not None:
        me.polygons.foreach_set("material_index", np.array(mat_index, dtype=np.int32))
    for p in me.polygons:
        p.use_smooth = True
    obj = bpy.data.objects.new(name, me)
    bpy.context.scene.collection.objects.link(obj)
    set_weights(obj, W, bones)
    return obj


def set_weights(obj, W, bones, limit=4):
    """Write dense weights as vertex groups, keeping the top `limit` per vertex
    (glTF / game engines skin with 4 influences)."""
    W = W.copy()
    if limit and W.shape[1] > limit:
        drop = np.argsort(W, axis=1)[:, :-limit]
        np.put_along_axis(W, drop, 0.0, axis=1)
        s = W.sum(axis=1, keepdims=True)
        s[s == 0] = 1
        W /= s
    for bi, b in enumerate(bones):
        idx = np.nonzero(W[:, bi] > 1e-4)[0]
        if not len(idx):
            continue
        vg = obj.vertex_groups.new(name=b)
        for i in idx:
            vg.add([int(i)], float(W[i, bi]), "REPLACE")


def remove_loose(obj):
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    loose = [v for v in bm.verts if not v.link_faces]
    bmesh.ops.delete(bm, geom=loose, context="VERTS")
    bm.to_mesh(obj.data)
    bm.free()


def parent_to_rig(obj, rig):
    obj.parent = rig
    mod = obj.modifiers.new("Armature", "ARMATURE")
    mod.object = rig


def build_human(body, name="human"):
    h = Human()
    load_obj(h)
    apply_targets(h, body)
    build_armature(h, name)
    return h


# ------------------------------------------------------------------ poses

def pose_relaxed(rig, strength=1.0):
    """Arms down by the sides, slight natural finger curl (preview pose)."""
    pb = rig.pose.bones
    for p in pb:
        p.rotation_mode = "XYZ"
        p.rotation_euler = (0, 0, 0)
    if strength == 0:
        return
    bpy.context.view_layer.update()
    for side, sgn in (("Left", 1), ("Right", -1)):
        _aim(rig, f"mixamorig:{side}Arm", Vector((0.30 * sgn, 0.04, -1)), strength)
        bpy.context.view_layer.update()
        _aim(rig, f"mixamorig:{side}ForeArm", Vector((0.16 * sgn, -0.22, -1)), strength)
        bpy.context.view_layer.update()
        # natural relaxed hand: graded curl (index least, pinky most), thumb
        # resting along the index finger instead of sticking out
        curl = {"Index": (8, 14, 8), "Middle": (13, 20, 10), "Ring": (17, 24, 12),
                "Pinky": (21, 27, 14), "Thumb": (22, 12, 10)}
        for f, angs in curl.items():
            for j, ang in zip((1, 2, 3), angs):
                b = pb.get(f"mixamorig:{side}Hand{f}{j}")
                if b:
                    b.rotation_mode = "XYZ"
                    b.rotation_euler = (math.radians(ang * strength), 0, 0)
    bpy.context.view_layer.update()


def _aim(rig, bone, direction, strength):
    """Rotate a pose bone so its Y axis points along `direction` (armature space)."""
    pb = rig.pose.bones[bone]
    cur = (pb.tail - pb.head).normalized()
    want = direction.normalized()
    q = cur.rotation_difference(want)
    if strength < 1:
        q = Quaternion().slerp(q, strength)
    m = pb.matrix.to_quaternion()
    pb.rotation_mode = "QUATERNION"
    pb.rotation_quaternion = pb.rotation_quaternion @ (m.inverted() @ q @ m)
