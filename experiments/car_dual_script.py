from pathlib import Path

import mitsuba as mi
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

from utils.utils import render_scene, image_stats, unroll_image, roll_image

MI_HARDWARE_TYPE = "cuda_ad"
MI_COLOR_TYPE = "rgb"
DOUBLE_PRECISION = ""  # float32
mi.set_variant(f"{MI_HARDWARE_TYPE}_{MI_COLOR_TYPE}{DOUBLE_PRECISION}")
from assets.car_scene import CAR_BASE_SCENE_DICT

OUT_DIR = Path(__file__).resolve().parents[1] / "results" / "car" / "brute_force"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def map_range(nparr):
    return nparr.clip(0, 1)


# constants
LIGHT_H = 128
LIGHT_W = 128
CAM_H = 128
CAM_W = 128
SAMPLES = 4096
EMITTER_RADIANCE = 2000
FORWARD_AREA_LIGHT_PATTERN = np.ones([LIGHT_H, LIGHT_W, 3]) * EMITTER_RADIANCE
BACKWARD_AREA_LIGHT_PATTERN = np.ones([CAM_H, CAM_W, 3]) * EMITTER_RADIANCE
FOV = 45

# projector pixels per camera pixel; brightens the dual image to match ground truth
RES_RATIO = (LIGHT_H * LIGHT_W) / (CAM_H * CAM_W)

CAMERA_LOC = [5, 10, 12]
CAMERA_TARGET = [0, 1, 0]
LIGHT_LOC = [0, 3, 14]
LIGHT_TARGET = CAMERA_TARGET
UP = [0, 1, 0]


forward_scene = {}
for k, v in CAR_BASE_SCENE_DICT.items():
    forward_scene[k] = v

forward_scene["sensor"] = {
    "type": "perspective",
    "fov": FOV,
    "to_world": mi.ScalarTransform4f.look_at(
        origin=CAMERA_LOC, target=CAMERA_TARGET, up=UP
    ),
    "film": {
        "type": "hdrfilm",
        "pixel_format": "rgb",
        "width": CAM_W,
        "height": CAM_H,
        "rfilter": {"type": "box"},
    },
    "sampler": {"type": "independent", "sample_count": SAMPLES},
}

# SET AREA LIGHT
forward_scene["textured_light"] = {
    "type": "projector",
    "to_world": mi.ScalarTransform4f.look_at(
        origin=LIGHT_LOC, target=LIGHT_TARGET, up=UP
    ),
    "fov": FOV,
    "irradiance": {
        "type": "bitmap",
        "bitmap": mi.Bitmap(FORWARD_AREA_LIGHT_PATTERN),
        "raw": True,
        "filter_type": "nearest",  # CRUCIAL to avoid extra illumination at corners
    },
}

forward_luminance = render_scene(forward_scene)
plt.imsave(OUT_DIR / "forward_image.png", map_range(forward_luminance))
print("Saved forward image as forward_image.png")
print(image_stats(forward_luminance))


backward_scene = {}
for k, v in CAR_BASE_SCENE_DICT.items():
    backward_scene[k] = v

backward_scene["sensor"] = {
    "type": "perspective",
    "to_world": mi.ScalarTransform4f.look_at(
        origin=LIGHT_LOC, target=LIGHT_TARGET, up=UP
    ),
    "fov": FOV,
    "film": {
        "type": "hdrfilm",
        "pixel_format": "rgb",
        "width": LIGHT_W,
        "height": LIGHT_H,
        "rfilter": {"type": "box"},
    },
    "sampler": {"type": "independent", "sample_count": SAMPLES},
}

# SET AREA LIGHT
backward_scene["textured_light"] = {
    "type": "projector",
    "to_world": mi.ScalarTransform4f.look_at(
        origin=CAMERA_LOC, target=CAMERA_TARGET, up=UP
    ),
    "fov": FOV,
    "irradiance": {
        "type": "bitmap",
        "bitmap": mi.Bitmap(BACKWARD_AREA_LIGHT_PATTERN),
        "raw": True,
        "filter_type": "nearest",  # CRUCIAL to avoid extra illumination at corners
    },
}

backward_luminance = render_scene(backward_scene)
plt.imsave(OUT_DIR / "backward_image.png", map_range(backward_luminance))
print("Saved backward image as backward_image.png")
print(image_stats(backward_luminance))


def get_light_bitmap_at_row_col(row: int, col: int):
    tex = np.zeros([LIGHT_H, LIGHT_W])
    tex[row][col] = EMITTER_RADIANCE
    return mi.Bitmap(tex)


def update_area_light(scene_dict: dict, bitmap: mi.Bitmap):
    scene_dict["textured_light"] = {
        "type": "projector",
        "to_world": mi.ScalarTransform4f.look_at(
            origin=LIGHT_LOC, target=LIGHT_TARGET, up=UP
        ),
        "fov": FOV,
        "irradiance": {
            "type": "bitmap",
            "bitmap": bitmap,
            "raw": True,
            "filter_type": "nearest",  # CRUCIAL to avoid extra illumination at corners
        },
    }
    return scene_dict


scene_dict = {}
for key, val in CAR_BASE_SCENE_DICT.items():
    scene_dict[key] = val

scene_dict["sensor"] = {
    "type": "perspective",
    "fov": FOV,
    "to_world": mi.ScalarTransform4f.look_at(
        origin=CAMERA_LOC, target=CAMERA_TARGET, up=UP
    ),
    "film": {
        "type": "hdrfilm",
        "pixel_format": "rgb",
        "width": CAM_W,
        "height": CAM_H,
        "rfilter": {"type": "box"},
    },
    "sampler": {"type": "independent", "sample_count": SAMPLES},
}


T_matrix = np.zeros([CAM_H * CAM_W, LIGHT_H * LIGHT_W, 3])

# scan light
for row in range(LIGHT_H):
    for col in tqdm(range(LIGHT_W), desc=f"row {row}"):
        bitmap = get_light_bitmap_at_row_col(row, col)
        scene_dict = update_area_light(scene_dict, bitmap)

        luminance = render_scene(scene_dict)

        T_col = unroll_image(luminance)
        T_index = row * LIGHT_W + col
        T_matrix[:, T_index] = T_col


p = np.ones([LIGHT_H * LIGHT_W, 3])
c = np.einsum("hwc, wc -> hc", T_matrix, p)
recon = roll_image(c, CAM_H, CAM_W, 3)

diff = forward_luminance - recon

fig, ax = plt.subplots(1, 3, figsize=(10, 5))
ax[0].imshow(map_range(recon))
ax[1].imshow(map_range(forward_luminance))
ax[2].imshow(map_range(diff))
for a in ax:
    a.axis("off")
plt.tight_layout()
plt.savefig(OUT_DIR / "reconstruction_comparison.png")
plt.close(fig)
print("Saved reconstruction comparison as reconstruction_comparison.png")

dual_c = np.ones([CAM_H * CAM_W, 3])
T_transpose = T_matrix.swapaxes(0, 1)  # now shape = (W, H, 3)
dual_p = np.einsum("hwc, wc -> hc", T_transpose, dual_c) * RES_RATIO
dual_image = roll_image(dual_p, LIGHT_H, LIGHT_W, 3)

plt.imsave(OUT_DIR / "dual_image.png", map_range(dual_image))
print("Saved dual image as dual_image.png")
np.save(OUT_DIR / "T_matrix.npy", T_matrix)

# For the dual image comparison
fig, ax = plt.subplots(1, 3, figsize=(12, 4))
ax[0].imshow(map_range(forward_luminance))
ax[0].set_title("primal image")
ax[1].imshow(map_range(dual_image))
ax[1].set_title("dual image")
ax[2].imshow(map_range(backward_luminance))
ax[2].set_title("ground truth dual image")
for a in ax:
    a.axis("off")
plt.tight_layout()
plt.savefig(OUT_DIR / "dual_image_comparison.png")
plt.close(fig)
print("Saved dual image comparison as dual_image_comparison.png")
