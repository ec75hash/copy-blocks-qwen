#!/usr/bin/env python3
"""Hash only this release tree, without depending on a parent Git repository."""
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
EXCLUDED={'.git','.cache','.venv','__pycache__','reproduction','dist','build'}

def files():
    for path in sorted(ROOT.rglob('*')):
        rel=path.relative_to(ROOT)
        if any(part in EXCLUDED for part in rel.parts):continue
        if path.is_file() and path.name!='checksums.json' and path.suffix!='.pyc':
            yield rel.as_posix(),path

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(16*1024*1024),b''):h.update(b)
    return h.hexdigest()

if __name__=='__main__':
    result={name:sha(path) for name,path in files()}
    (ROOT/'checksums.json').write_text(json.dumps(result,indent=2)+'\n')
    print(f'Hashed {len(result)} release files')
