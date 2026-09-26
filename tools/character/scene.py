"""Scene setup, preview rendering and export."""
import math
import os

import bpy
from mathutils import Vector

import mh


def reset():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    try:
        bpy.ops.preferences.addon_enable(module="cycles")
    except Exception:
        pass
    sc = bpy.context.scene
    bpy.context.preferences.filepaths.save_version = 0
    sc.unit_settings.system = "METRIC"
    sc.render.fps = 30


def finalize(rig, name):
    for o in bpy.data.objects:
        o.select_set(False)
    rig.select_set(True)
    bpy.context.view_layer.objects.active = rig
    rig.data.pose_position = "REST"
    rig.show_in_front = False


def export_glb(rig, path):
    bpy.ops.object.select_all(action="DESELECT")
    rig.select_set(True)
    for c in rig.children:
        c.select_set(True)
    bpy.context.view_layer.objects.active = rig
    rig.data.pose_position = "REST"
    bpy.ops.export_scene.gltf(
        filepath=path,
        export_format="GLB",
        use_selection=True,
        export_apply=False,
        export_skins=True,
        export_all_influences=False,
        export_animations=False,
        export_yup=True,
        export_image_format="AUTO",
        export_texcoords=True,
        export_normals=True,
        export_tangents=True,
    )


# ------------------------------------------------------------------ preview

def _world(sc):
    w = bpy.data.worlds.new("studio")
    sc.world = w
    w.use_nodes = True
    nt = w.node_tree
    bg = nt.nodes["Background"]
    sky = nt.nodes.new("ShaderNodeTexGradient")
    coord = nt.nodes.new("ShaderNodeTexCoord")
    ramp = nt.nodes.new("ShaderNodeValToRGB")
    mapn = nt.nodes.new("ShaderNodeMapping")
    mapn.inputs["Rotation"].default_value = (0, math.radians(-90), 0)
    nt.links.new(coord.outputs["Generated"], mapn.inputs["Vector"])
    nt.links.new(mapn.outputs["Vector"], sky.inputs["Vector"])
    nt.links.new(sky.outputs["Fac"], ramp.inputs["Fac"])
    ramp.color_ramp.elements[0].color = (0.10, 0.10, 0.11, 1)
    ramp.color_ramp.elements[1].color = (0.30, 0.31, 0.33, 1)
    nt.links.new(ramp.outputs["Color"], bg.inputs["Color"])
    bg.inputs["Strength"].default_value = 0.6


def _area(name, loc, target, energy, size, color=(1, 1, 1)):
    l = bpy.data.lights.new(name, "AREA")
    l.energy = energy
    l.size = size
    l.color = color
    o = bpy.data.objects.new(name, l)
    bpy.context.scene.collection.objects.link(o)
    o.location = loc
    d = Vector(target) - Vector(loc)
    o.rotation_euler = d.to_track_quat("-Z", "Y").to_euler()
    return o


def setup_render(sc, w=900, h=1200, samples=None):
    samples = samples or int(os.environ.get("CHAR_SAMPLES", 128))
    sc.render.engine = "CYCLES"
    sc.cycles.device = "CPU"
    sc.cycles.samples = samples
    sc.cycles.use_denoising = True
    try:
        sc.cycles.denoiser = "OPENIMAGEDENOISE"
    except Exception:
        pass
    sc.cycles.max_bounces = 6
    sc.render.resolution_x = w
    sc.render.resolution_y = h
    sc.render.film_transparent = False
    sc.view_settings.view_transform = "AgX"
    sc.view_settings.look = "AgX - Medium High Contrast"


def render_previews(rig, spec, out_dir, name):
    sc = bpy.context.scene
    setup_render(sc)
    _world(sc)
    # floor
    bpy.ops.mesh.primitive_plane_add(size=12, location=(0, 0, 0))
    floor = bpy.context.active_object
    floor.name = "_floor"
    fm = bpy.data.materials.new("_floor")
    fm.use_nodes = True
    fm.node_tree.nodes["Principled BSDF"].inputs["Base Color"].default_value = (0.2, 0.2, 0.21, 1)
    fm.node_tree.nodes["Principled BSDF"].inputs["Roughness"].default_value = 0.8
    floor.data.materials.append(fm)

    height = max((rig.matrix_world @ b.tail_local).z for b in rig.data.bones)
    key = _area("_key", (-2.2, -3.0, 2.8), (0, 0, 1.2), 380, 2.0, (1.0, 0.96, 0.9))
    fill = _area("_fill", (2.8, -2.0, 1.4), (0, 0, 1.0), 120, 3.0, (0.85, 0.9, 1.0))
    rim = _area("_rim", (0.8, 3.0, 2.4), (0, 0, 1.3), 320, 1.5, (0.9, 0.95, 1.0))

    cam_data = bpy.data.cameras.new("_cam")
    cam = bpy.data.objects.new("_cam", cam_data)
    sc.collection.objects.link(cam)
    sc.camera = cam

    def shoot(fname, loc, target, lens, w=900, h=1200):
        sc.render.resolution_x, sc.render.resolution_y = w, h
        cam.location = loc
        d = Vector(target) - Vector(loc)
        cam.rotation_euler = d.to_track_quat("-Z", "Y").to_euler()
        cam_data.lens = lens
        sc.render.filepath = os.path.join(out_dir, "renders", fname)
        bpy.ops.render.render(write_still=True)
        print("  rendered", fname, flush=True)

    rig.data.pose_position = "POSE"
    mh.pose_relaxed(rig)
    mid = height * 0.52
    shoot("front.png", (0, -5.2, mid), (0, 0, mid), 50)
    shoot("three_quarter.png", (3.4, -3.9, mid + 0.1), (0, 0, mid), 50)
    shoot("side.png", (5.2, 0, mid), (0, 0, mid), 50)
    shoot("back.png", (0, 5.2, mid), (0, 0, mid), 50)
    head = rig.pose.bones["mixamorig:Head"]
    hz = (rig.matrix_world @ head.head).z + 0.09
    shoot("face.png", (0.35, -0.95, hz + 0.02), (0, 0, hz), 85, 900, 900)
    shoot("face_side.png", (1.0, -0.25, hz + 0.02), (0, 0, hz), 85, 900, 900)
    hand = rig.pose.bones["mixamorig:LeftHand"]
    hp = rig.matrix_world @ ((hand.head + hand.tail) / 2)
    shoot("hand.png", (hp.x + 0.55, hp.y - 0.75, hp.z + 0.08), hp, 85, 900, 900)
    rig.data.pose_position = "POSE"
    mh.pose_relaxed(rig, 0.0)
    shoot("apose.png", (0, -6.0, mid), (0, 0, mid), 50)

    for o in (floor, key, fill, rim, cam):
        bpy.data.objects.remove(o, do_unlink=True)
    rig.data.pose_position = "REST"
