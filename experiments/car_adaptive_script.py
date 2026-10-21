from pathlib import Path

import mitsuba as mi
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
import imageio.v2 as imageio

from utils.utils import render_scene, image_stats, unroll_image, roll_image

MI_HARDWARE_TYPE = "cuda_ad"
MI_COLOR_TYPE = "rgb"
DOUBLE_PRECISION = ""  # float32
mi.set_variant(f"{MI_HARDWARE_TYPE}_{MI_COLOR_TYPE}{DOUBLE_PRECISION}")
from assets.car_scene import CAR_BASE_SCENE_DICT
from sampling_algorithms.adaptive_multiplexing import AdaptiveMultiplexing

OUT_DIR = Path(__file__).resolve().parents[1] / "results" / "car" / "adaptive"
for d in [OUT_DIR / "beam_frames", OUT_DIR / "ambient_frames"]:
    d.mkdir(parents=True, exist_ok=True)


def map_range(nparr):
    return nparr.clip(0, 1)


# constants
LIGHT_H = 512
LIGHT_W = 512
CAM_H = 128
CAM_W = 128
SAMPLES = 4096
EMITTER_RADIANCE = 1000
FORWARD_AREA_LIGHT_PATTERN = np.ones([LIGHT_H, LIGHT_W, 3]) * EMITTER_RADIANCE
BACKWARD_AREA_LIGHT_PATTERN = np.ones([CAM_H, CAM_W, 3]) * EMITTER_RADIANCE
FOV = 45
THRESHOLD = 0.0005

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
        "filter_type": "nearest",
    },
}

backward_luminance = render_scene(backward_scene)
plt.imsave(OUT_DIR / "backward_image.png", map_range(backward_luminance))
print("Saved backward image as backward_image.png")
print(image_stats(backward_luminance))


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
            "filter_type": "nearest",
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

# scan light
scanner = AdaptiveMultiplexing(
    camera_resolution=(CAM_H, CAM_W, 3), projector_resolution=(LIGHT_H, LIGHT_W)
)

total_patterns_projected = 0

while not scanner.finished:
    patterns = scanner.get_next_patterns()
    if not patterns:
        break

    current_round_captures = []
    total_patterns_projected += len(patterns)

    for pattern in tqdm(
        patterns, desc=f"Scanning Phase (Blocks: {len(scanner.blocks_to_process)})"
    ):
        radiance_pattern = pattern * EMITTER_RADIANCE
        radiance_pattern = np.stack([radiance_pattern] * 3, axis=-1)

        bitmap = mi.Bitmap(radiance_pattern)
        scene_dict = update_area_light(scene_dict, bitmap)

        luminance = render_scene(scene_dict)
        current_round_captures.append(luminance)

    scanner.process_captured_images(current_round_captures, threshold=THRESHOLD)

print(f"Total Patterns Projected: {total_patterns_projected}")

T_matrix_flat = scanner.get_T_matrix()
# Reshape assuming flat T_matrix is (CAM_PIXELS*3, PROJ_PIXELS)
# AdaptiveMultiplexing t_entries stores flattened camera pixel index including RGB
T_matrix = T_matrix_flat.reshape(CAM_H * CAM_W, 3, LIGHT_H * LIGHT_W).transpose(0, 2, 1)


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


# RELIGHTING


def get_beam_light_pattern(index):
    pattern = np.ones([CAM_H, CAM_W, 3]) * 0.5
    pixel_width = max(CAM_W // 20, 1)
    for i in range(index - pixel_width, index + pixel_width):
        if i < 0 or i >= CAM_W:
            continue
        pattern[:, i] = np.ones([CAM_H, 3]) * 2
    return unroll_image(pattern)


def save_gif(frames, path, fps=10):
    """
    Save a list of HWC numpy arrays as a GIF.

    Frames can be float or uint; output will be uint8.
    """
    duration = 1.0 / fps
    out = []

    for f in frames:
        f = np.asarray(f)
        if f.dtype != np.uint8:
            # assume float in [0,1] or [0,255]
            if f.max() <= 1.0:
                f = (f * 255).clip(0, 255)
            f = f.astype(np.uint8)
        out.append(f)

    imageio.mimsave(path, out, format="GIF", duration=duration, loop=0)


frames = []
for beam_index in tqdm(range(CAM_W)):
    dual_c = get_beam_light_pattern(beam_index)
    dual_p = np.einsum("hwc, wc -> hc", T_transpose, dual_c)
    dual_image = roll_image(dual_p, LIGHT_H, LIGHT_W, 3) * RES_RATIO
    dual_image = np.clip(dual_image, 0, 1)
    frames.append(dual_image)

for i, frame in enumerate(frames):
    plt.imsave(OUT_DIR / "beam_frames" / f"frame_{i}.png", frame)

save_gif(frames, OUT_DIR / "beam_sweep.gif")


def procedural_ambient_texture_colorful(size=512, t=None):
    # Coordinate grid from -4..4 (gives nice scale)
    x = np.linspace(-4, 4, size)
    y = np.linspace(-4, 4, size)
    x, y = np.meshgrid(x, y)

    # Three slowly drifting cloud centers
    cx1 = np.cos(t * 0.61) * 2.2
    cy1 = np.sin(t * 0.73) * 1.9
    cx2 = np.sin(t * 0.47) * 2.7
    cy2 = np.cos(t * 0.58) * 2.3
    cx3 = np.cos(t * 0.83) * 1.8
    cy3 = np.sin(t * 0.52) * 2.4

    # Distance from each center → soft radial waves
    d1 = np.hypot(x - cx1, y - cy1)
    d2 = np.hypot(x - cx2, y - cy2)
    d3 = np.hypot(x - cx3, y - cy3)

    # Animated waves
    wave1 = np.sin(d1 * 1.1 - t * 0.9)
    wave2 = np.sin(d2 * 1.3 + t * 0.7)
    wave3 = np.sin(d3 * 0.9 - t * 1.1)

    # Shared luminance (very soft clouds)
    lum = 0.35 + 0.65 * (
        np.sin(wave1 * 1.4) * 0.50
        + np.sin(wave2 * 1.2) * 0.40
        + np.sin(wave3 * 1.6) * 0.35
    )
    lum = np.clip(lum, 0, 1)
    lum = np.power(lum, 1.4)  # creamy contrast

    # Colour: slow hue rotation + subtle spatial variation
    hue = (t * 0.07) % 1.0
    hue += wave1 * 0.07 + wave2 * 0.05 + wave3 * 0.06
    hue = hue % 1.0

    saturation = 0.4 + 0.3 * np.sin(wave1 + wave2 * 0.8)  # gently breathing saturation

    # Fast vectorised HSV → RGB
    h6 = hue * 6.0
    c = saturation * lum  # chroma
    x = c * (1 - np.abs((h6 % 2) - 1))

    # Six segments of the hue wheel
    idx = h6.astype(int)

    conds = [idx == 0, idx == 1, idx == 2, idx == 3, idx == 4, idx == 5]
    choices_r = [c, x, np.zeros_like(c), np.zeros_like(c), x, c]
    choices_g = [x, c, c, x, np.zeros_like(c), np.zeros_like(c)]
    choices_b = [np.zeros_like(c), np.zeros_like(c), x, c, c, x]

    r = np.select(conds, choices_r)
    g = np.select(conds, choices_g)
    b = np.select(conds, choices_b)

    # Add brightness back
    m = lum - c
    r, g, b = r + m, g + m, b + m

    rgb = np.stack([r, g, b], axis=-1)
    rgb = np.power(rgb, 0.9)  # final softness

    return np.clip(rgb, 0, 1).astype(np.float32)


frames = []
for t_index in tqdm(range(500)):
    t = t_index / 10.0
    img = (procedural_ambient_texture_colorful(size=CAM_H, t=t) + 0.5) * RES_RATIO
    dual_c = unroll_image(img)
    dual_p = np.einsum("hwc, wc -> hc", T_transpose, dual_c)
    dual_image = roll_image(dual_p, LIGHT_H, LIGHT_W, 3)
    dual_image = np.clip(dual_image, 0, 1)
    frames.append(dual_image)

for i, frame in enumerate(frames):
    plt.imsave(OUT_DIR / "ambient_frames" / f"frame_{i}.png", frame)

save_gif(frames, OUT_DIR / "ambient.gif")
