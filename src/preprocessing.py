from __future__ import annotations

from pathlib import Path
from typing import Literal

from PIL import Image

try:
except ImportError:  # pragma: no cover - dependency is optional at import time
    pydicom = None


ALLOWED_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".dcm", ".dicom"}
STANDARD_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp"}
DICOM_SUFFIXES = {".dcm", ".dicom"}


def _as_window_value(value: object) -> float | None:
    """Convert DICOM window metadata to a float when possible.

    Some DICOM tags can be stored as a single value or as a sequence.
    This helper keeps the main DICOM loading function readable.
    """
    if value is None:
        return None

    try:
        if isinstance(value, (list, tuple)):
            return float(value[0])
        # pydicom MultiValue behaves like a sequence but is not always a list/tuple.
        if hasattr(value, "__iter__") and not isinstance(value, (str, bytes)):
            return float(list(value)[0])
        return float(value)
    except (TypeError, ValueError, IndexError):
        return None


def _normalize_to_uint8(array: np.ndarray) -> np.ndarray:
    """Normalize a medical image array into uint8 [0, 255].

    The percentile clipping avoids one or two extreme pixels dominating the
    visualization. This is a pragmatic preprocessing step for a VLM prototype,
    not a clinically validated radiology preprocessing protocol.
    """
    array = array.astype(np.float32)

    if not np.isfinite(array).all():
        array = np.nan_to_num(array, nan=0.0, posinf=0.0, neginf=0.0)

    low, high = np.percentile(array, (1, 99))

    if high <= low:
        min_value = float(np.min(array))
        max_value = float(np.max(array))
        if max_value <= min_value:
            return np.zeros(array.shape, dtype=np.uint8)
        low, high = min_value, max_value

    array = np.clip(array, low, high)
    array = (array - low) / (high - low)
    array = (array * 255.0).clip(0, 255).astype(np.uint8)
    return array


def load_dicom_image(path: str | Path, size: tuple[int, int] = (512, 512)) -> Image.Image:
    """Load a RSNA/CXR DICOM image and convert it to RGB PIL format.

    Parameters
    path:
        Path to a DICOM file, usually from RSNA under
        ``data/raw/rsna/stage_2_train_images/*.dcm``.
    size:
        Target image size expected by the VLM pipeline.

    Returns
    PIL.Image.Image
        RGB image resized to ``size``.

    Notes
    This function is designed for an educational prototype. It handles common
    DICOM details such as rescale slope/intercept, optional windowing metadata,
    MONOCHROME1 inversion, uint8 conversion and resizing. It does not replace a
    clinically validated medical-imaging pipeline.
    """
    if pydicom is None:
        raise ImportError(
            "pydicom is required to load DICOM files. Install it with: pip install pydicom"
        )

    path = Path(path)
    dataset = pydicom.dcmread(str(path))

    pixels = dataset.pixel_array.astype(np.float32)

    slope = float(getattr(dataset, "RescaleSlope", 1.0))
    intercept = float(getattr(dataset, "RescaleIntercept", 0.0))
    pixels = pixels * slope + intercept

    window_center = _as_window_value(getattr(dataset, "WindowCenter", None))
    window_width = _as_window_value(getattr(dataset, "WindowWidth", None))

    if window_center is not None and window_width is not None and window_width > 0:
        lower = window_center - window_width / 2
        upper = window_center + window_width / 2
        pixels = np.clip(pixels, lower, upper)

    photometric = str(getattr(dataset, "PhotometricInterpretation", "")).upper()
    pixels_uint8 = _normalize_to_uint8(pixels)

    if photometric == "MONOCHROME1":
        pixels_uint8 = 255 - pixels_uint8

    image = Image.fromarray(pixels_uint8).convert("RGB")
    return image.resize(size)


def load_standard_image(path: str | Path, size: tuple[int, int] = (512, 512)) -> Image.Image:
    """Load a standard image file such as PNG/JPG/BMP and convert it to RGB."""
    path = Path(path)
    image = Image.open(path).convert("RGB")
    return image.resize(size)


def load_image(path: str | Path, size: tuple[int, int] = (512, 512)) -> Image.Image:
    """Load an image safely for the educational prototype.

    This function keeps backward compatibility with the original synthetic
    pipeline while adding support for real RSNA DICOM files.

    Supported formats:
    - synthetic/standard images: PNG, JPG, JPEG, BMP
    - RSNA medical images: DICOM, DCM

    For real CXR work, DICOM metadata, windowing, projection and acquisition
    details should be handled explicitly and documented. This implementation is
    a clean prototype-level baseline, not a diagnostic medical pipeline.
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")

    suffix = path.suffix.lower()

    if suffix not in ALLOWED_SUFFIXES:
        raise ValueError(
            f"Unsupported image format: {suffix}. "
            f"Supported formats are: {sorted(ALLOWED_SUFFIXES)}"
        )

    if suffix in DICOM_SUFFIXES:
        return load_dicom_image(path, size=size)

    return load_standard_image(path, size=size)


def preprocess_for_vlm(
    path: str | Path,
    size: tuple[int, int] = (512, 512),
    output: Literal["pil", "numpy"] = "pil",
) -> Image.Image | np.ndarray:
    """Preprocess an image for the VLM pipeline.

    The current notebooks and toy inference pipeline work naturally with PIL
    images. ``output='numpy'`` is available for quick inspection or future ML
    code that expects an array.

    Parameters
    path:
        Image path, either synthetic PNG/JPG or RSNA DICOM.
    size:
        Target resize resolution.
    output:
        ``"pil"`` for an RGB PIL image, ``"numpy"`` for a float32 array in
        [0, 1] with shape (H, W, 3).
    """
    image = load_image(path, size=size)

    if output == "pil":
        return image

    if output == "numpy":
        return np.asarray(image).astype(np.float32) / 255.0

    raise ValueError("output must be either 'pil' or 'numpy'")


def find_image_files(root: str | Path) -> list[Path]:
    """Return all supported image files under a directory."""
    root = Path(root)
    if not root.exists():
        return []

    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in ALLOWED_SUFFIXES
    )


def find_rsna_images(rsna_root: str | Path = "data/raw/rsna") -> list[Path]:
    """Find RSNA DICOM files in the expected local dataset structure.

    Expected examples:
    - data/raw/rsna/stage_2_train_images/*.dcm
    - data/raw/rsna/stage_2_test_images/*.dcm
    """
    rsna_root = Path(rsna_root)
    return sorted(rsna_root.rglob("*.dcm")) + sorted(rsna_root.rglob("*.dicom"))


def get_demo_image_paths(
    rsna_root: str | Path = "data/raw/rsna",
    synthetic_root: str | Path = "data/sample_images",
    limit: int = 5,
) -> tuple[list[Path], str]:
    """Return RSNA images if available, otherwise fallback to synthetic images.

    Returns
    tuple[list[Path], str]
        A list of image paths and a source label: ``"rsna"`` or ``"synthetic"``.
    """
    rsna_images = find_rsna_images(rsna_root)
    if rsna_images:
        return rsna_images[:limit], "rsna"

    synthetic_images = find_image_files(synthetic_root)
    return synthetic_images[:limit], "synthetic"


def rsna_patient_id_from_path(path: str | Path) -> str:
    """Extract the RSNA patient ID from a DICOM filename."""
    return Path(path).stem


def basic_quality_flag(path: str | Path) -> str:
    """Toy quality flag based on filename metadata.

    Replace this with real image-quality checks in a serious implementation.
    The function is kept for backward compatibility with the current toy
    inference pipeline.
    """
    name = Path(path).name.lower()
    if "uncertain" in name or "limited" in name:
        return "limited"
    return "good"
