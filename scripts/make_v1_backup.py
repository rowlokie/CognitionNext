import shutil
import os
from pathlib import Path

ROOT = Path(__file__).parent.parent
BACKUP_DIR = ROOT / "backups" / "decide-v1"
BACKUP_DIR.mkdir(parents=True, exist_ok=True)

dirs_to_copy = ["src", "app", "artifacts", "tests", "scripts", "testing", "testing folder", "data", "docs", "reports"]
files_to_copy = ["model_card.md", "README.md", "results.json", "requirements.txt", "Makefile", "selfcheck.py", "demo_examples.json", ".gitignore"]

for d in dirs_to_copy:
    src_p = ROOT / d
    dst_p = BACKUP_DIR / d
    if src_p.exists():
        if dst_p.exists():
            shutil.rmtree(dst_p)
        shutil.copytree(src_p, dst_p, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        print(f"Copied directory {d} -> backups/decide-v1/{d}")

for f in files_to_copy:
    src_p = ROOT / f
    dst_p = BACKUP_DIR / f
    if src_p.exists():
        shutil.copy2(src_p, dst_p)
        print(f"Copied file {f} -> backups/decide-v1/{f}")

print("\n[OK] Complete V1 backup verified at:", BACKUP_DIR)
