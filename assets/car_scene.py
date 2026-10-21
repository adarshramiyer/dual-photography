from pathlib import Path

import mitsuba as mi

MESH_DIR = Path(__file__).resolve().parent / "objs" / "sep_by_mtl"

CAR_TO_WORLD_TRANSFORM = mi.ScalarTransform4f.rotate(axis=[0, 1, 0], angle=150)
PLANE_SCALE = 4

CAR_BASE_SCENE_DICT = {
    "type": "scene",
    "integrator": {"type": "path"},  # Path tracer
    # "ambient": {
    #     "type": "constant",
    #     "radiance": {
    #         "type": "rgb",
    #         "value": 0.3,
    #     },
    # },
    "ground_plane": {
        "type": "cube",
        "to_world": mi.ScalarTransform4f.translate([0, 0, -5])
        @ mi.ScalarTransform4f.scale([PLANE_SCALE, 0.001, PLANE_SCALE * 2]),
        "bsdf": {
            "type": "diffuse",
            "reflectance": {"type": "rgb", "value": [0.1, 0.1, 0.1]},
        },
    },
    "left_plane": {
        "type": "cube",
        "to_world": mi.ScalarTransform4f.translate([-4, 2, -4])
        @ mi.ScalarTransform4f.scale([0.001, PLANE_SCALE / 2, PLANE_SCALE + 3]),
        "bsdf": {
            "type": "diffuse",
            "reflectance": {"type": "rgb", "value": [0.1, 0.1, 0.1]},
        },
    },
    "right_plane": {
        "type": "cube",
        "to_world": mi.ScalarTransform4f.translate([0, 2, -8])
        @ mi.ScalarTransform4f.scale([PLANE_SCALE, PLANE_SCALE / 2, 0.001]),
        "bsdf": {
            "type": "diffuse",
            "reflectance": {"type": "rgb", "value": [0.1, 0.1, 0.1]},
        },
    },
    "body": {
        "type": "obj",
        "filename": str(MESH_DIR / "body.obj"),
        "to_world": CAR_TO_WORLD_TRANSFORM,
        "material": {
            "type": "roughplastic",
            "diffuse_reflectance": {"type": "rgb", "value": [0.2, 0.02, 0.02]},
            "alpha": 0.05,
            "distribution": "ggx",
            "nonlinear": False,
        },
    },
    "chrome": {
        "type": "obj",
        "filename": str(MESH_DIR / "chrome.obj"),
        "to_world": CAR_TO_WORLD_TRANSFORM,
        "material": {
            "type": "roughconductor",
            "material": "Cr",
            "alpha": 0.3,
            "distribution": "ggx",
        },
    },
    "white": {
        "type": "obj",
        "filename": str(MESH_DIR / "default_white.obj"),
        "to_world": CAR_TO_WORLD_TRANSFORM,
        "material": {
            "type": "roughplastic",
            "diffuse_reflectance": {"type": "rgb", "value": [0.9, 0.9, 0.9]},
            "alpha": 0.9,
            "distribution": "ggx",
            "nonlinear": False,
        },
    },
    "logo": {
        "type": "obj",
        "filename": str(MESH_DIR / "logo_green.obj"),
        "to_world": CAR_TO_WORLD_TRANSFORM,
        "material": {
            "type": "roughplastic",
            "diffuse_reflectance": {"type": "rgb", "value": [0.0, 0.27, 0.14]},
            "alpha": 0.9,
            "distribution": "ggx",
            "nonlinear": False,
        },
    },
    "matte": {
        "type": "obj",
        "filename": str(MESH_DIR / "matte.obj"),
        "to_world": CAR_TO_WORLD_TRANSFORM,
        "material": {
            "type": "roughplastic",
            "diffuse_reflectance": {"type": "rgb", "value": [0.0, 0.0, 0.0]},
            "alpha": 1.0,
            "distribution": "ggx",
            "nonlinear": False,
        },
    },
    "windows": {
        "type": "obj",
        "filename": str(MESH_DIR / "tinted_windows.obj"),
        "to_world": CAR_TO_WORLD_TRANSFORM,
        "material": {"type": "dielectric", "int_ior": 1.52, "ext_ior": 1.0},
    },
    "tire_rims": {
        "type": "obj",
        "filename": str(MESH_DIR / "tire_rims.obj"),
        "to_world": CAR_TO_WORLD_TRANSFORM,
        "material": {
            "type": "roughplastic",
            "diffuse_reflectance": {"type": "rgb", "value": [1.0, 1.0, 1.0]},
            "alpha": 0.6,
            "distribution": "ggx",
            "nonlinear": False,
        },
    },
    "tires": {
        "type": "obj",
        "filename": str(MESH_DIR / "tires.obj"),
        "to_world": CAR_TO_WORLD_TRANSFORM,
        "material": {
            "type": "roughplastic",
            "diffuse_reflectance": {"type": "rgb", "value": [0.2, 0.2, 0.2]},
            "alpha": 0.9,
            "distribution": "ggx",
            "nonlinear": False,
        },
    },
    "windshield": {
        "type": "obj",
        "filename": str(MESH_DIR / "windshield.obj"),
        "to_world": CAR_TO_WORLD_TRANSFORM,
        "material": {"type": "dielectric", "int_ior": 1.52, "ext_ior": 1.0},
    },
}
