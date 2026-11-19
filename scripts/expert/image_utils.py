import numpy as np
from PIL import Image
from scripts.expert import settings

def to_dataset_image(raw: np.ndarray) -> np.ndarray:
    img = np.asarray(raw)
    if img.dtype != np.uint8:
        img = np.clip(img * (255.0 if img.max() <= 1.0 else 1.0), 0, 255).astype(np.uint8)
    expected = (settings.dataset_render_height, settings.dataset_render_width, 3)
    if img.shape != expected:
        raise ValueError(f"Unexpected render shape {img.shape}, expected {expected}")

    img = np.flipud(img)

    x0f, y0f, sf = settings.dataset_crop
    h, w = img.shape[:2]
    s, x0, y0 = int(round(sf * w)), int(round(x0f * w)), int(round(y0f * h))
    img = np.ascontiguousarray(img[y0:y0 + s, x0:x0 + s])

    out = Image.fromarray(img).resize(
        (settings.dataset_image_width, settings.dataset_image_height), Image.LANCZOS)
    return np.ascontiguousarray(np.asarray(out))