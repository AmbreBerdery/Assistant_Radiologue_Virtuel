from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path


COMPETITION = "rsna-pneumonia-detection-challenge"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "raw" / "rsna"


def has_kaggle_credentials() -> bool:
    """Check whether Kaggle credentials are likely configured locally."""
    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    env_credentials = bool(os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"))
    return kaggle_json.exists() or env_credentials


def check_kaggle_cli() -> None:
    """Fail early with a clear message if the Kaggle CLI is missing."""
    if shutil.which("kaggle") is None:
        raise RuntimeError(
            "Kaggle CLI not found.\n"
            "Install it with:\n"
            "  pip install kaggle\n"
            "Then configure your Kaggle API token. Do not commit kaggle.json."
        )

    if not has_kaggle_credentials():
        raise RuntimeError(
            "Kaggle credentials not found.\n"
            "Create an API token from your Kaggle account and place kaggle.json in:\n"
            "  ~/.kaggle/kaggle.json\n"
            "or set KAGGLE_USERNAME and KAGGLE_KEY as environment variables.\n"
            "Never commit kaggle.json to GitHub."
        )


def run_command(command: list[str]) -> None:
    """Run a shell command and raise a readable error if it fails."""
    print("+", " ".join(command))
    completed = subprocess.run(command, check=False)

    if completed.returncode != 0:
        raise RuntimeError(
            "Kaggle download failed.\n"
            "Possible causes:\n"
            "- Kaggle API token is missing or invalid.\n"
            "- You have not accepted the competition rules on Kaggle.\n"
            "- The competition data is not accessible from your account.\n"
            "- Network or storage issue."
        )


def unzip_archives(output_dir: Path) -> None:
    """Extract every zip file found in the output directory."""
    archives = sorted(output_dir.glob("*.zip"))

    if not archives:
        print("No zip archive found to extract.")
        return

    for archive in archives:
        print(f"Extracting {archive.name} ...")
        with zipfile.ZipFile(archive, "r") as zip_file:
            zip_file.extractall(output_dir)

    print("Extraction complete.")


def download_rsna(output_dir: Path, unzip: bool = True, force: bool = False) -> None:
    """Download RSNA Pneumonia Detection Challenge data with Kaggle CLI."""
    output_dir.mkdir(parents=True, exist_ok=True)

    check_kaggle_cli()

    command = [
        "kaggle",
        "competitions",
        "download",
        "-c",
        COMPETITION,
        "-p",
        str(output_dir),
    ]

    if force:
        command.append("--force")

    print("Downloading RSNA Pneumonia Detection Challenge dataset.")
    print(f"Target directory: {output_dir}")
    print("Reminder: do not commit downloaded medical data to GitHub.")
    print("You must have accepted the Kaggle competition rules before running this script.")

    run_command(command)

    if unzip:
        unzip_archives(output_dir)

    print("\nDone.")
    print("Expected local structure may include:")
    print(f"  {output_dir / 'stage_2_train_images'}")
    print(f"  {output_dir / 'stage_2_test_images'}")
    print(f"  {output_dir / 'stage_2_train_labels.csv'}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download RSNA Pneumonia Detection Challenge data locally."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Local output directory. Default: data/raw/rsna",
    )
    parser.add_argument(
        "--no-unzip",
        action="store_true",
        help="Download zip files but do not extract them.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download if files already exist.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    try:
        download_rsna(
            output_dir=args.output_dir,
            unzip=not args.no_unzip,
            force=args.force,
        )
    except Exception as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
