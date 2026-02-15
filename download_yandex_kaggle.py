"""Download Yandex Personalized Web Search Challenge logs via Kaggle CLI."""
import os
import shutil
import subprocess
import sys
from pathlib import Path


def _has_kaggle_credentials() -> bool:
    """Check for Kaggle API credentials."""
    env_ok = bool(os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"))
    if env_ok:
        return True

    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    return kaggle_json.exists()


def main() -> int:
    """Download Kaggle competition logs into data/yandex."""
    dest_dir = Path("data") / "yandex"
    dest_dir.mkdir(parents=True, exist_ok=True)

    if not _has_kaggle_credentials():
        print("Kaggle credentials not found.")
        print("Set KAGGLE_USERNAME and KAGGLE_KEY or create ~/.kaggle/kaggle.json.")
        print("See: https://www.kaggle.com/docs/api")
        return 1

    if shutil.which("kaggle") is None:
        print("Kaggle CLI not found. Install with: pip install kaggle")
        return 1

    competition = "yandex-personalized-web-search-challenge"
    print(f"Downloading Kaggle competition: {competition}")

    cmd = [
        "kaggle",
        "competitions",
        "download",
        "-c",
        competition,
        "-p",
        str(dest_dir),
        "--force",
    ]

    try:
        subprocess.check_call(cmd)
    except subprocess.CalledProcessError as exc:
        print(f"Download failed: {exc}")
        return 1

    # Unzip all archives in destination
    for zip_path in dest_dir.glob("*.zip"):
        print(f"Extracting {zip_path.name}...")
        shutil.unpack_archive(str(zip_path), str(dest_dir))

    print("Download complete.")
    print(f"Files are in: {dest_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
