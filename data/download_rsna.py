from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

COMPETITION_NAME = "rsna-pneumonia-detection-challenge"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "raw" / "rsna"


def kaggle_credentials_available() -> bool:
    """Return True if Kaggle credentials seem to be configured locally.

    Credentials must stay outside the GitHub repository.
    """
    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    env_credentials = bool(os.getenv("KAGGLE_USERNAME") and os.getenv("KAGGLE_KEY"))
    return kaggle_json.exists() or env_credentials


def kaggle_cli_available() -> bool:
    """Return True if the Kaggle command-line tool is available."""
    return shutil.which("kaggle") is not None


def validate_kaggle_setup() -> None:
    """Validate local Kaggle setup before launching a download."""
    if not kaggle_cli_available():
        raise RuntimeError(
            "Kaggle CLI is not installed. Install it with: pip install kaggle"
        )

    if not kaggle_credentials_available():
        raise RuntimeError(
            "Kaggle credentials were not found. Create a Kaggle API token and place "
            "kaggle.json in ~/.kaggle/kaggle.json, or set KAGGLE_USERNAME and "
            "KAGGLE_KEY as environment variables. Never commit kaggle.json."
        )


def run_command(command: list[str]) -> None:
    """Run a shell command safely and raise a readable error if it fails."""
    print("+", " ".join(command))
    completed = subprocess.run(command, check=False)

    if completed.returncode != 0:
        raise RuntimeError(
            "The Kaggle download command failed. Make sure you have accepted the "
            "RSNA competition rules on Kaggle and that your API credentials are valid."
        )


def extract_zip_files(output_dir: Path) -> None:
    """Extract all zip files found in output_dir."""
    zip_files = sorted(output_dir.glob("*.zip"))

    if not zip_files:
        print("No zip files found to extract.")
        return

    for zip_path in zip_files:
        print(f"Extracting {zip_path.name} ...")
        with zipfile.ZipFile(zip_path, "r") as archive:
            archive.extractall(output_dir)

    print("Extraction complete.")


def download_rsna(
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    unzip: bool = True,
    force: bool = False,
) -> None:
    """Download the RSNA Pneumonia Detection Challenge dataset locally.

    The downloaded data must not be committed to GitHub.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    validate_kaggle_setup()

    command = [
        "kaggle",
        "competitions",
        "download",
        "-c",
        COMPETITION_NAME,
        "-p",
        str(output_dir),
    ]

    if force:
        command.append("--force")

    print("Downloading RSNA Pneumonia Detection Challenge dataset.")
    print(f"Output directory: {output_dir}")
    print("Reminder: do not commit downloaded medical data to GitHub.")

    run_command(command)

    if unzip:
        extract_zip_files(output_dir)

    print("Done.")
    print("Expected local files may include:")
    print(f"- {output_dir / 'stage_2_train_images'}")
    print(f"- {output_dir / 'stage_2_test_images'}")
    print(f"- {output_dir / 'stage_2_train_labels.csv'}")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Download RSNA Pneumonia Detection Challenge data locally."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Download directory. Default: data/raw/rsna",
    )
    parser.add_argument(
        "--no-unzip",
        action="store_true",
        help="Download the archive without extracting it.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download if Kaggle finds existing files.",
    )
    return parser.parse_args()


def main() -> int:
    """Command-line entrypoint."""
    args = parse_args()

    try:
        download_rsna(
            output_dir=args.output_dir,
            unzip=not args.no_unzip,
            force=args.force,
        )
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
