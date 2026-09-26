"""Strand hair (Blender particle hair, rendered as curves in Cycles/EEVEE).

The textured mesh hair cap is what games get (glTF has no strands); inside
Blender the cap is covered with short strands, plus eyebrow and eyelash
strands, which is what makes close-ups read as real hair.
The particle systems sit after the Armature modifier, so they follow the rig.
"""
import bpy
import numpy as np

import noise
from noise import smoothstep as ss


def _co(obj):
    a = np.empty(len(obj.data.vertices) * 3)
    obj.data.vertices.foreach_get("co", a)
    return a.reshape(-1, 3)


def _normals(obj):
    a = np.empty(len(obj.data.vertices) * 3)
    obj.data.vertices.foreach_get("normal", a)
    return a.reshape(-1, 3)


def _group(obj, name, w):
    vg = obj.vertex_groups.get(name) or obj.vertex_groups.new(name=name)
    for i in np.nonzero(w > 1e-3)[0]:
        vg.add([int(i)], float(min(w[i], 1.0)), "REPLACE")
    return vg


def hair_material(name, look):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    nt = m.node_tree
    for n in list(nt.nodes):
        nt.nodes.remove(n)
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    hb = nt.nodes.new("ShaderNodeBsdfHairPrincipled")
    try:
        hb.parametrization = "MELANIN"
    except Exception:
        pass
    c = np.asarray(look["hair"])
    lum = float(c.mean())
    melanin = float(np.clip(1.0 - lum * 0.6, 0.3, 0.99))
    hb.inputs["Melanin"].default_value = melanin
    hb.inputs["Melanin Redness"].default_value = 0.12 if not look["zombie"] else 0.08
    hb.inputs["Roughness"].default_value = 0.35
    hb.inputs["Radial Roughness"].default_value = 0.4
    if "Random Color" in hb.inputs:
        hb.inputs["Random Color"].default_value = 0.15
    if "Random Roughness" in hb.inputs:
        hb.inputs["Random Roughness"].default_value = 0.2
    if look["zombie"] and "Tint" in hb.inputs:
        hb.inputs["Tint"].default_value = (0.8, 0.8, 0.75, 1)
    nt.links.new(hb.outputs[0], out.inputs["Surface"])
    return m


def _system(obj, name, group, count, length, mat_name, *, children=0, normal=1.0,
            align=(0, 0, 0), radius=0.00005, clump=0.0, rough=0.0, length_group=None,
            display=0.3, rand_len=0.3):
    mod = obj.modifiers.new(name, "PARTICLE_SYSTEM")
    ps = mod.particle_system
    st = ps.settings
    st.name = name
    st.type = "HAIR"
    st.use_advanced_hair = True
    st.count = count
    st.hair_length = length
    st.emit_from = "FACE"
    st.use_emit_random = True
    st.use_even_distribution = True
    # advanced hair: strand length comes from the emission velocity (m)
    st.normal_factor = normal * length
    st.object_align_factor = tuple(a * length / 0.01 for a in align)
    st.factor_random = 0.0
    st.length_random = rand_len
    st.material = list(obj.data.materials).index(bpy.data.materials[mat_name]) + 1
    st.display_percentage = int(display * 100)
    st.hair_step = 4
    st.display_step = 3
    st.render_step = 4
    st.radius_scale = radius
    st.root_radius = 1.0
    st.tip_radius = 0.3
    st.use_close_tip = True
    if children:
        st.child_type = "INTERPOLATED"
        st.child_percent = max(1, children // 3)
        st.rendered_child_count = children
        st.clump_factor = clump
        st.roughness_1 = rough
        st.roughness_1_size = 1.0
        st.roughness_endpoint = rough * 0.5
        st.child_length = 0.9
        st.child_length_threshold = 0.3
    ps.vertex_group_density = group
    if length_group:
        ps.vertex_group_length = length_group
    return ps


def add(h, parts, look):
    body = parts["body"]
    cap = parts["hair"]
    lm = h.lm
    mat = hair_material(f"{body.name}_strands", look)
    for o in (body, cap):
        o.data.materials.append(mat)

    # ---- scalp strands grow out of the cap's outer shell
    co = _co(cap)
    inner = np.zeros(len(co))
    vg_in = cap.vertex_groups.get("_shell_inner")
    vg_rim = cap.vertex_groups.get("_shell_rim")
    for v in cap.data.vertices:
        for g in v.groups:
            if (vg_in and g.group == vg_in.index) or (vg_rim and g.group == vg_rim.index):
                inner[v.index] = max(inner[v.index], g.weight)
    hairline = h.meta["hairline"]
    above = hairline.line(co) - co[:, 2]
    dens = (1 - inner) * ss(0.012, -0.01, above)
    if look["zombie"]:
        dens *= ss(0.66, 0.6, noise.fbm(co, 14.0, 2, seed=3))
    _group(cap, "strand_density", dens)
    top = ss(lm.top - 0.12, lm.top - 0.02, co[:, 2])
    _group(cap, "strand_length", 0.35 + 0.65 * top)
    _system(cap, "ScalpHair", "strand_density", 16000, 0.006, mat.name, children=5,
            normal=0.8, align=(0.0, 0.006, -0.004), radius=0.00005, clump=0.3, rough=0.0008,
            length_group="strand_length", display=0.25)

    # ---- eyebrows + eyelashes on the face
    bco = _co(body)
    bn = _normals(body)
    brow = np.zeros(len(bco))
    lash = np.zeros(len(bco))
    for e, s in ((lm.eye_l, 1), (lm.eye_r, -1)):
        u = (bco[:, 0] - e[0]) * s
        t = (u + 0.022) / 0.05
        arch = lm.brow_z + 0.004 + 0.006 * np.sin(np.clip(t, 0, 1) * np.pi * 0.8) - 0.004 * t
        thick = 0.0085 * (1 - 0.5 * np.clip(t, 0, 1))
        band = ss(thick * 1.1, thick * 0.3, np.abs(bco[:, 2] - arch)) * ss(-0.05, 0.08, t) * ss(1.05, 0.9, t)
        brow = np.maximum(brow, band * (bn[:, 1] < -0.2))
        d = np.linalg.norm(bco - e, axis=1)
        lid = ss(lm.eye_radius * 1.25, lm.eye_radius * 1.08, d) * ss(e[2] - 0.001, e[2] + 0.004, bco[:, 2])
        lash = np.maximum(lash, lid * (bco[:, 1] < e[1]))
    _group(body, "brows", brow)
    _group(body, "lashes", lash)
    _system(body, "Eyebrows", "brows", 1800, 0.007, mat.name, normal=0.25,
            align=(0.0, 0.0, 0.0), radius=0.00004, display=1.0, rand_len=0.4)
    brows_ps = body.particle_systems["Eyebrows"].settings
    brows_ps.tangent_factor = 0.9 * 0.007
    brows_ps.tangent_phase = 0.0
    _system(body, "Eyelashes", "lashes", 260, 0.008, mat.name, normal=0.6,
            align=(0.0, -0.004, 0.006), radius=0.00004, display=1.0, rand_len=0.25)
    return mat
