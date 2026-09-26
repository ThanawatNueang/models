# Realistic character builder

`build_character.py` generates rigged, textured human characters (a survivor and
a zombie) fully procedurally inside Blender. It borrows proven techniques from
open-source and production pipelines:

| Part | Technique | Source of the idea |
|---|---|---|
| Anatomy | MakeHuman HM08 base mesh (real 5-finger hands, face, ears, eyes, teeth, tongue) shaped with MakeHuman *macro targets* (gender / age / muscle / weight / height / proportions / ethnicity blend) | MakeHuman / MPFB2 (CC0 assets) |
| Rig | MPFB **Mixamo** skeleton + skin weights (`mixamorig:Hips`, `mixamorig:LeftHandIndex1`, ...) → Mixamo animations retarget 1:1 | MPFB2 |
| Clothes | Garments are grown out of the body surface: face selection by skin weights + cut planes, boundary snapping (clean hems), *inflate & relax* (Laplacian smoothing that is pushed back out to keep a clearance from the body, as in cloth tools), fold noise and Solidify for real fabric thickness. They inherit the body's skin weights, so they deform with it. Hidden body faces are deleted. | Marvelous / game-outfit workflow |
| Boots | Signed distance field of the foot → morphological closing (fills the toe gaps) → marching cubes → decimate → Smart UV | ZBrush DynaMesh / voxel remesh |
| Textures | Custom UV-space rasterizer: every texel knows its 3D position, normal and skin weights, so textures are painted with 3D "texture shaders" (skin tone, cheeks, lips, stubble, eyebrows, lash line, nails, veins, pores, forehead lines, crow's feet; iris fibres, limbal ring; fabric heather, collar rib, double stitching, side seams, cargo pockets, belt loops, fly, knee wear; boot laces, eyelets, sole tread, welt stitch, scuffs, mud; zombie decay, wounds, bite, blood). Height → tangent-space normal map from the UV gradient. Ray-cast ambient occlusion. | Substance-style procedural texturing |
| Hair | Textured mesh cap with a soft alpha hairline (game/glTF) + particle strand hair, eyebrows and eyelashes on top (Blender/Cycles) | standard real-time vs offline hair |
| Shading | Principled BSDF with subsurface skin, coat on eyes, sheen on cloth, Principled Hair BSDF for strands | |

## Output (`characters/<variant>/`)

* `<variant>.blend` – open in Blender (4.2+ / 5.x). Rest pose is A-pose; strand hair included.
* `<variant>.glb` – game-ready glTF (mesh + skin + PBR textures, no strands).
* `textures/` – color / roughness / normal maps.
* `renders/` – Cycles previews (relaxed pose, face, hand, A-pose).

## Rebuild

```
pip install bpy==4.2.0 scipy scikit-image   # or run inside Blender's Python
python3 tools/character/build_character.py            # survivor + zombie
python3 tools/character/build_character.py zombie     # one variant
CHAR_SAMPLES=32 python3 tools/character/build_character.py   # faster previews
```

Tweak bodies, outfits and colors in `variants.py`
(macro sliders are 0..1, 0.5 = average).

## Files

* `mh.py` – base mesh loading, macro targets, rig + weights, poses
* `outfit.py` – shirt, cargo pants, boots, hair cap, body clean-up
* `textures.py` – rasterizer, texture shaders, AO, materials
* `strands.py` – particle hair, eyebrows, eyelashes
* `scene.py` – lighting, preview renders, glTF export
* `noise.py` – vectorized 3D value noise / fbm / ridged noise
* `makehuman/` – the CC0 MakeHuman data used (see `LICENSE-CC0.md`)
