"""Character variants. Body values are MakeHuman macro sliders (0..1, 0.5 = average)."""

FACE = {
    "chin/chin-prominent-incr": 0.5, "chin/chin-width-incr": 0.5, "chin/chin-height-incr": 0.2,
    "head/head-square": 0.5,
    "cheek/l-cheek-bones-incr": 0.4, "cheek/r-cheek-bones-incr": 0.4,
    "cheek/l-cheek-volume-decr": 0.3, "cheek/r-cheek-volume-decr": 0.3,
    "nose/nose-point-width-decr": 0.4, "nose/nose-flaring-decr": 0.3,
    "nose/nose-scale-vert-incr": 0.3, "nose/nose-hump-incr": 0.2,
    "mouth/mouth-lowerlip-volume-incr": 0.3, "mouth/mouth-upperlip-volume-incr": 0.1,
    "eyes/l-eye-height2-decr": 0.2, "eyes/r-eye-height2-decr": 0.2,
    "eyes/l-eye-eyefold-down": 0.3, "eyes/r-eye-eyefold-down": 0.3,
    "eyes/l-eye-scale-incr": 0.2, "eyes/r-eye-scale-incr": 0.2,
    "neck/neck-scale-horiz-incr": 0.5, "neck/measure-neck-height-decr": 0.5,
    "torso/measure-shoulder-dist-incr": 0.3, "torso/torso-muscle-pectoral-incr": 0.3,
    "torso/torso-muscle-dorsi-incr": 0.3,
}

VARIANTS = {
    "survivor": {
        "name": "survivor",
        "seed": 3,
        "body": {
            "gender": 1.0, "age": 0.52, "muscle": 0.68, "weight": 0.48,
            "height": 0.58, "proportions": 0.85,
            "race": {"asian": 0.65, "caucasian": 0.25, "african": 0.10},
            "detail": FACE,
        },
        # style: "underwear" (base body) or "clothes" (tee + cargo pants + boots)
        "outfit": {"style": "underwear", "underwear": (0.10, 0.11, 0.13),
                   "sleeve": 0.5, "shirt_hem": 0.08, "hair_len": 0.022,
                   "hair_style": "textured_quiff"},
        "look": {
            "skin": (0.74, 0.54, 0.42),      # sRGB albedo
            "skin_var": 0.06,
            "redness": 0.35,
            "stubble": 0.45,
            "hair": (0.045, 0.035, 0.03),
            "brow": (0.06, 0.045, 0.035),
            "iris": (0.23, 0.14, 0.07),
            "shirt": (0.30, 0.34, 0.24),     # olive tee
            "pants": (0.26, 0.23, 0.18),     # dark khaki cargo
            "boots": (0.23, 0.15, 0.09),     # brown leather
            "dirt": 0.25,
            "blood": 0.0,
            "zombie": False,
        },
    },
    "zombie": {
        "name": "zombie",
        "seed": 11,
        "body": {
            "gender": 1.0, "age": 0.62, "muscle": 0.42, "weight": 0.36,
            "height": 0.55, "proportions": 0.6,
            "race": {"asian": 0.4, "caucasian": 0.5, "african": 0.1},
            "detail": {k: w * 0.5 for k, w in FACE.items()},
        },
        "outfit": {"style": "underwear", "underwear": (0.30, 0.28, 0.25),
                   "sleeve": 0.45, "shirt_hem": 0.06, "hair_len": 0.012,
                   "hair_style": "messy",
                   "torn": True, "bald_patches": True, "hairline": 0.012},
        "look": {
            "skin": (0.56, 0.58, 0.48),
            "skin_var": 0.12,
            "redness": 0.0,
            "stubble": 0.6,
            "hair": (0.05, 0.045, 0.04),
            "brow": (0.07, 0.06, 0.05),
            "iris": (0.55, 0.55, 0.50),
            "shirt": (0.55, 0.53, 0.47),     # stained white tee
            "pants": (0.16, 0.20, 0.27),     # faded jeans
            "boots": (0.16, 0.13, 0.11),
            "dirt": 0.7,
            "blood": 1.0,
            "zombie": True,
        },
    },
}
