import numpy as np
import mitsuba as mi

def unroll_image(image: np.ndarray):
    """
    Convert an image of shape (H, W) or (H, W, C) into a flat array of shape (H*W*C,).
    Grayscale is treated as C=1.
    """
    if image.ndim == 2:
        h, w = image.shape
        return image.reshape(h * w)
    elif image.ndim == 3:
        h, w, c = image.shape
        return image.reshape(h * w, c)
    else:
        raise ValueError("Image must have shape (H, W) or (H, W, C).")


def roll_image(flat_image: np.ndarray, h: int, w: int, c: int):
    """
    Convert a flat array of length H*W*C back into an (H, W) grayscale image (if C=1)
    or an (H, W, C) color image otherwise.
    """
    assert flat_image.size == h * w * c

    if c == 1:
        return flat_image.reshape(h, w)
    else:
        return flat_image.reshape(h, w, c)


def image_stats(image_array: np.ndarray) -> str:
    """
    Calculates key statistical metrics for a NumPy array representing an image 
    and returns the results as a concise, formatted string.

    Args:
        image_array: A NumPy array (e.g., loaded image data).
        
    Returns:
        A multi-line formatted string containing the image statistics.
    """
    if image_array is None or image_array.size == 0:
        return "Error: Input array is empty or None."

    # Flatten and convert data type for accurate statistical calculation
    calc_array = image_array.flatten().astype(np.float64)

    # Core Metrics
    mean_val = np.mean(calc_array)
    std_dev_val = np.std(calc_array)
    min_val = np.min(calc_array)
    max_val = np.max(calc_array)
    median_val = np.median(calc_array)
    q1 = np.percentile(calc_array, 25)
    q3 = np.percentile(calc_array, 75)
    
    # Array Properties
    shape_val = image_array.shape
    dtype_val = image_array.dtype
    
    stats_string = (
        f"Shape: {shape_val}\n"
        f"DType: {dtype_val}\n"
        f"-------------------\n"
        f"Mean:   {mean_val:.8f}\n"
        f"StdDev: {std_dev_val:.8f}\n"
        f"Median: {median_val:.8f}\n"
        f"Min/Max: {min_val:.8f} / {max_val:.8f}\n"
        f"Range:  {(max_val - min_val):.8f}\n"
        f"Q1/Q3:  {q1:.8f} / {q3:.8f}"
    )
    
    return stats_string

def linear_to_srgb(x):
    # x is linear RGB in [0, ∞)
    a = 0.055
    srgb = np.where(
        x <= 0.0031308,
        12.92 * x,
        (1 + a) * np.power(x, 1/2.4) - a
    )
    return np.clip(srgb, 0, 1)

def render_scene(scene: dict):
    scene = mi.load_dict(scene)
    image = mi.render(scene)
    image_np = mi.TensorXf(image).numpy()
    return image_np
    
