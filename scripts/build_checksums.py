#!/usr/bin/env python3
"""Hash the tracked release tree after staging changes (excluding checksums.json itself)."""
import hashlib
import json
from pathlib import Path
import subprocess

root = Path(__file__).resolve().parents[1]
names = subprocess.check_output(["git", "ls-files", "-z"], cwd=root).decode().split("\0")
checksums = {}
for name in sorted(n for n in names if n and n != "checksums.json"):
    path = root / name
    if path.is_file():
        checksums[name] = hashlib.sha256(path.read_bytes()).hexdigest()
(root / "checksums.json").write_text(json.dumps(checksums, indent=2) + "\n")
print(f"Hashed {len(checksums)} tracked release files")
