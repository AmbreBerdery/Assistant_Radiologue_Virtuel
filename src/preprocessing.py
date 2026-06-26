from __future__ import annotations

from pathlib import Path
from typing import Any

from PIL import Image

ALLOWED_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp"}
DICOM_SUFFIXES = {".dcm", ".dicom"}
SUPPORTED_SUFFIXES = ALLOWED_SUFFIXES | DICOM_SUFFIXES
DEFAULT_IMAGE_SIZE = (512, 512)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SYNTHETIC_DIR = PROJECT_ROOT / "data" / "sample_images"
DEFAULT_RSNA_DIR = PROJECT_ROOT / "data" / "raw" / "rsna"


def load_image(path: str | Path, size: tuple[int, int] = DEFAULT_IMAGE_SIZE) -> Image.Image:
    """Load an image safely for the educational prototype.

    Supports standard image formats used by the toy dataset:
    PNG, JPG, JPEG, BMP.

    Also supports DICOM files for RSNA if pydicom and numpy are installed.
    """
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix in DICOM_SUFFIXES:
        return load_dicom_image(path, size=size)

    if suffix not in ALLOWED_SUFFIXES:
        raise ValueError(f"Unsupported image format: {path.suffix}")

    img = Image.open(path).convert("RGB")
    return img.resize(size)


def load_dicom_image(path: str | Path, size: tuple[int, int] = DEFAULT_IMAGE_SIZE) -> Image.Image:
    """Load and preprocess a DICOM image as an RGB PIL image.

    pydicom and numpy are imported lazily so smoke tests can run without
    medical-imaging dependencies installed.
    """
    try:
        import numpy as np
        import pydicom
    except ModuleNotFoundError as exc:
        raise ImportError(
            "DICOM loading requires optional dependencies. "
            "Install them with: pip install numpy pydicom"
        ) from exc

    path = Path(path)
    ds = pydicom.dcmread(str(path))

    pixels = ds.pixel_array.astype("float32")

    if getattr(ds, "PhotometricInterpretation", "") == "MONOCHROME1":
        pixels = pixels.max() - pixels

    min_value = float(np.min(pixels))
    max_value = float(np.max(pixels))

    if max_value > min_value:
        pixels = (pixels - min_value) / (max_value - min_value)
    else:
        pixels = np.zeros_like(pixels)

    pixels = (pixels * 255).clip(0, 255).astype("uint8")

    img = Image.fromarray(pixels).convert("RGB")
    return img.resize(size)


def preprocess_image(
    path: str | Path,
    size: tuple[int, int] = DEFAULT_IMAGE_SIZE,
    as_numpy: bool = False,
) -> Image.Image | Any:
    """Load and preprocess an image for the VLM pipeline.

    By default, returns a PIL RGB image resized to 512x512.

    If as_numpy=True, returns a float32 array scaled between 0 and 1.
    """
    img = load_image(path, size=size)

    if not as_numpy:
        return img

    try:
        import numpy as np
    except ModuleNotFoundError as exc:
        raise ImportError(
            "as_numpy=True requires numpy. Install it with: pip install numpy"
        ) from exc

    return np.asarray(img).astype("float32") / 255.0


def preprocess_for_vlm(
    path: str | Path,
    size: tuple[int, int] = DEFAULT_IMAGE_SIZE,
    output: str = "pil",
) -> Image.Image | Any:
    """Compatibility wrapper for VLM preprocessing."""
    if output == "pil":
        return preprocess_image(path, size=size, as_numpy=False)

    if output == "numpy":
        return preprocess_image(path, size=size, as_numpy=True)

    raise ValueError("output must be either 'pil' or 'numpy'")


def basic_quality_flag(path: str | Path) -> str:
    """Toy quality flag based on filename metadata."""
    name = Path(path).name.lower()
    if "uncertain" in name or "limited" in name:
        return "limited"
    return "good"


def is_rsna_available(rsna_dir: str | Path = DEFAULT_RSNA_DIR) -> bool:
    """Return True if a local RSNA directory seems to contain image files."""
    rsna_dir = Path(rsna_dir)

    if not rsna_dir.exists():
        return False

    for pattern in ("*.dcm", "*.dicom", "*.png", "*.jpg", "*.jpeg"):
        if any(rsna_dir.rglob(pattern)):
            return True

    return False


def find_image_files(
    root_dir: str | Path,
    limit: int | None = None,
) -> list[Path]:
    """Find supported image files in a directory."""
    root_dir = Path(root_dir)

    if not root_dir.exists():
        return []

    files = [
        path
        for path in root_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES
    ]

    files = sorted(files)

    if limit is not None:
        return files[:limit]

    return files


def find_rsna_images(
    rsna_dir: str | Path = DEFAULT_RSNA_DIR,
    limit: int | None = None,
) -> list[Path]:
    """Find RSNA image files in the expected local dataset directory."""
    return find_image_files(rsna_dir, limit=limit)


def get_demo_images(
    rsna_dir: str | Path = DEFAULT_RSNA_DIR,
    synthetic_dir: str | Path = DEFAULT_SYNTHETIC_DIR,
    limit: int = 5,
) -> list[Path]:
    """Return RSNA images if available, otherwise fallback to synthetic images."""
    rsna_images = find_image_files(rsna_dir, limit=limit)

    if rsna_images:
        return rsna_images

    return find_image_files(synthetic_dir, limit=limit)


def get_demo_image_paths(
    rsna_root: str | Path = DEFAULT_RSNA_DIR,
    synthetic_root: str | Path = DEFAULT_SYNTHETIC_DIR,
    limit: int = 5,
) -> tuple[list[Path], str]:
    """Return demo image paths and a source label."""
    rsna_images = find_image_files(rsna_root, limit=limit)

    if rsna_images:
        return rsna_images, "rsna"

    return find_image_files(synthetic_root, limit=limit), "synthetic"


def rsna_patient_id_from_path(path: str | Path) -> str:
    """Extract the RSNA patient ID from an image filename."""
    return Path(path).stem
