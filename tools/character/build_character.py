"""Build a realistic, rigged human character (survivor + zombie variants).

Pipeline (run headless with the `bpy` module or inside Blender):

    python3 tools/character/build_character.py            # all variants
    python3 tools/character/build_character.py survivor   # one variant

Base anatomy comes from the MakeHuman HM08 base mesh (CC0), shaped with
MakeHuman macro targets (gender/age/muscle/weight/height/proportions/race).
The skeleton + skin weights are the MPFB "mixamo" rig, so bone names are
`mixamorig:Hips`, `mixamorig:LeftHandIndex1`, ... and Mixamo animations
play on it directly.

On top of the anatomy this script builds (all procedurally, no external art):
  * clothing shells (t-shirt, cargo pants, boots) grown out of the body surface,
    so they inherit the rig weights and deform with the body,
  * a short hair cap, eyebrows, stubble, lips, nails,
  * texture maps (base color, roughness, tangent-space normal) that are painted
    per texel from 3D position -> numpy "texture shaders" (see textures.py),
  * eyes (sclera, iris, pupil, wet cornea look), teeth, tongue.

Outputs go to `characters/<variant>/`: `<variant>.blend`, `<variant>.glb`,
textures, and preview renders.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import bpy  # noqa: E402

import mh  # noqa: E402
import outfit  # noqa: E402
import textures  # noqa: E402
import scene  # noqa: E402
import strands  # noqa: E402
import groom  # noqa: E402
import variants  # noqa: E402

REPO = os.path.dirname(os.path.dirname(HERE))
OUT = os.path.join(REPO, "characters")


def build(name):
    spec = variants.VARIANTS[name]
    out_dir = os.path.join(OUT, name)
    os.makedirs(out_dir, exist_ok=True)
    scene.reset()

    human = mh.build_human(spec["body"], name=name)
    parts = outfit.build(human, spec)
    textures.paint(human, parts, spec, out_dir)
    rig = human.armature
    # glTF first (mesh hair cap only), then strand hair for Blender renders
    scene.export_glb(rig, os.path.join(out_dir, f"{name}.glb"))
    strands.add(human, parts, spec["look"], scalp=False)   # eyebrows + eyelashes
    groom.build(human, parts, spec["look"], style=spec["outfit"].get("hair_style", "textured_quiff"))

    scene.finalize(rig, name)
    bpy.ops.file.pack_all()  # self-contained .blend (textures embedded)
    bpy.ops.wm.save_as_mainfile(filepath=os.path.join(out_dir, f"{name}.blend"))
    scene.render_previews(rig, spec, out_dir, name)
    # Save again so the .blend keeps the relaxed preview pose off (rest pose).
    bpy.ops.wm.save_as_mainfile(filepath=os.path.join(out_dir, f"{name}.blend"))


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    if "--" in sys.argv:
        args = sys.argv[sys.argv.index("--") + 1:]
    names = args or list(variants.VARIANTS)
    for n in names:
        print(f"=== building {n} ===", flush=True)
        build(n)


if __name__ == "__main__":
    main()
