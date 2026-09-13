"""
Packaging utility to bundle code/, evaluation/, and README.md into code.zip for submission.

Constraints:
- Bundles: code/, evaluation/, and README.md.
- Excludes: __pycache__, *.pyc, *.pyo, dataset/, output.csv, .git/, tests/, scratch/.
"""

from __future__ import annotations

import os
import zipfile
from pathlib import Path

_ROOT_DIR = Path(__file__).resolve().parent
_OUTPUT_ZIP = _ROOT_DIR / "code.zip"

# Items to include in the zip archive
_INCLUDE_TARGETS = [
    "code",
    "evaluation",
    "README.md",
]

# Patterns and folders to exclude
_EXCLUDE_DIR_NAMES = {
    "__pycache__",
    ".git",
    ".gemini",
    ".system_generated",
    "scratch",
    "dataset",
}

_EXCLUDE_EXTENSIONS = {
    ".pyc",
    ".pyo",
    ".pyd",
    ".DS_Store",
}


def create_submission_zip(zip_path: Path = _OUTPUT_ZIP) -> None:
    """Create the submission zip file."""
    print(f"Creating submission package at: {zip_path.resolve()}")

    if zip_path.exists():
        zip_path.unlink()
        print("Removed existing code.zip archive.")

    total_files = 0
    total_uncompressed_bytes = 0

    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for target_name in _INCLUDE_TARGETS:
            target_path = _ROOT_DIR / target_name
            if not target_path.exists():
                print(f"Warning: Target {target_name} not found at {target_path}")
                continue

            if target_path.is_file():
                arcname = target_name
                zf.write(target_path, arcname=arcname)
                f_size = target_path.stat().st_size
                total_files += 1
                total_uncompressed_bytes += f_size
                print(f"  + Added file: {arcname} ({f_size:,} bytes)")
            elif target_path.is_dir():
                for root, dirs, files in os.walk(target_path):
                    # Prune excluded directories in-place
                    dirs[:] = [d for d in dirs if d not in _EXCLUDE_DIR_NAMES]

                    for f in files:
                        file_ext = os.path.splitext(f)[1].lower()
                        if file_ext in _EXCLUDE_EXTENSIONS or f.startswith("."):
                            continue

                        full_path = Path(root) / f
                        arcname = full_path.relative_to(_ROOT_DIR).as_posix()
                        zf.write(full_path, arcname=arcname)
                        f_size = full_path.stat().st_size
                        total_files += 1
                        total_uncompressed_bytes += f_size
                        print(f"  + Added file: {arcname} ({f_size:,} bytes)")

    zip_size = zip_path.stat().st_size
    print("-" * 65)
    print(f"SUCCESS: Created {zip_path.name}")
    print(f"Total files packaged: {total_files}")
    print(f"Uncompressed size: {total_uncompressed_bytes:,} bytes")
    print(f"Compressed size:   {zip_size:,} bytes ({zip_size / 1024:.1f} KB)")
    print("-" * 65)

    # Verification check: inspect archive contents
    print("Verifying code.zip contents:")
    with zipfile.ZipFile(zip_path, "r") as zf:
        namelist = zf.namelist()
        for name in namelist:
            info = zf.getinfo(name)
            print(f"  - {name} ({info.file_size:,} bytes)")

        # Invariant checks
        assert not any("dataset" in n for n in namelist), "Error: dataset folder included in zip!"
        assert not any("output.csv" in n for n in namelist), "Error: output.csv included in zip!"
        assert not any("__pycache__" in n for n in namelist), "Error: __pycache__ included in zip!"
        assert "evaluation/usage_report.md" in namelist or "code/evaluation/usage_report.md" in namelist, "Error: usage_report.md missing!"
        assert "README.md" in namelist, "Error: README.md missing!"

    print("\nAll submission integrity checks passed successfully!")


if __name__ == "__main__":
    create_submission_zip()
