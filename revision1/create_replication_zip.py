"""Create the minimal corrected replication archive for Zenodo.

The corrected source and results already validated for the revision are taken
from the revision-one code archive. Only the modules needed to rerun the
synthetic experiments or the descriptive logged-support replay are retained;
manuscript files, paper-asset builders, utils, and raw logs are excluded.
"""

from __future__ import annotations

import hashlib
import json
import os
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE_ZIP = Path(os.environ.get(
    "EDE_REVISION_SOURCE_ZIP",
    r"C:\Users\Arun\AppData\Local\Temp\ede_revision_one_inspect\revision1\FINAL\01_Zenodo_upload\EDE_revision_one_code_and_results.zip",
))
OUTPUT = ROOT / "revision1" / "FINAL_minimal" / "Zenodo_upload" / "EDE_revision1.zip"

SELECTED = [
    "LICENSE",
    "revision1/data/source.json",
    "revision1/fetch_replay_data.py",
    "revision1/requirements-revision.txt",
    "revision1/results/protocol.json",
    "revision1/results/replay/manifest.json",
    "revision1/results/replay/summary.csv",
    "revision1/results/test_per_seed.csv",
    "revision1/results/test_summary.csv",
    "revision1/results/validation_per_seed.csv",
    "revision1/results/validation_summary.csv",
    "src/ede_real_replay/__init__.py",
    "src/ede_real_replay/revision_replay.py",
    "src/ede_sanity/__init__.py",
    "src/ede_sanity/config.py",
    "src/ede_sanity/entropy.py",
    "src/ede_sanity/experiment.py",
    "src/ede_sanity/metrics.py",
    "src/ede_sanity/policy.py",
    "src/ede_sanity/ranker.py",
    "src/ede_sanity/revision_experiments.py",
    "src/ede_sanity/revision_policy.py",
    "src/ede_sanity/simulate.py",
    "tests/test_revision.py",
]

README = """# EDE revision one minimal replication archive

This archive contains only the corrected code and numerical evidence needed to
rerun the revision experiments and the descriptive Yandex logged-support
replay. It excludes manuscript files, figure and table upload assets, the
separate `utils` repository, raw Yandex logs, and paper-specific builders.

From the archive root, install the tested dependencies and run the focused
tests:

```text
python -m pip install -r revision1/requirements-revision.txt
python -m pytest tests -q
```

The complete corrected synthetic experiment suite can be rerun with:

```text
python -m src.ede_sanity.revision_experiments --workers 4
```

The optional logged-support replay requires the author's own Kaggle access:

```text
python revision1/fetch_replay_data.py
python -m src.ede_real_replay.revision_replay --log revision1/data/train_first_200000_sessions.txt
```

The acquisition script saves the first 200,000 complete sessions and a
SHA-256 source manifest; the raw log is not redistributed. The included
results were produced with validation seeds 100-109 and held-out seeds
1000-1019. The frozen operating point is lambda=0.4, eta=0.65, m=2,
top_L=50, alpha=0.5, and tau=50. The replay uses 2,000 session bootstrap
resamples with seed 20260911 and is a descriptive ten-candidate
logged-support diagnostic, not causal engagement or cold-start evidence.

`SHA256.json` records the SHA-256 hash of every archived file other than the
manifest itself.
"""


def write_entry(archive: zipfile.ZipFile, name: str, data: bytes) -> None:
    info = zipfile.ZipInfo(name, date_time=(2026, 9, 12, 0, 0, 0))
    info.compress_type = zipfile.ZIP_DEFLATED
    info.create_system = 0
    archive.writestr(info, data)


def main() -> None:
    if not SOURCE_ZIP.exists():
        raise FileNotFoundError(SOURCE_ZIP)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(SOURCE_ZIP) as source:
        source_names = set(source.namelist())
        missing = [name for name in SELECTED if name not in source_names]
        if missing:
            raise FileNotFoundError(f"Missing source entries: {missing}")
        files = {name: source.read(name) for name in SELECTED}

    hashes = {name: hashlib.sha256(data).hexdigest() for name, data in files.items()}
    files["README.md"] = README.encode("utf-8")
    hashes["README.md"] = hashlib.sha256(files["README.md"]).hexdigest()
    manifest = json.dumps(hashes, indent=2, sort_keys=True).encode("utf-8") + b"\n"
    files["SHA256.json"] = manifest

    with zipfile.ZipFile(OUTPUT, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name in sorted(files):
            write_entry(archive, name, files[name])

    print(f"Created {OUTPUT} with {len(files)} files")


if __name__ == "__main__":
    main()
