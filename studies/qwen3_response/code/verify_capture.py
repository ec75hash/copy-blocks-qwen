"""Verify frozen capture data without writing into it or running a model."""
from pathlib import Path
import argparse
import json
import tempfile
import shutil
import sys

HERE=Path(__file__).resolve().parent
STUDY=HERE.parent
sys.path.insert(0,str(HERE))
# Original numerical validator is retained, with only its output path redirected.
import importlib.util
spec=importlib.util.spec_from_file_location('as_run_verify', HERE/'as_run/verify.py')
module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,default=STUDY.parents[1]/'reproduction/qwen3_response/CAPTURE_VALIDATION.json')
    args=parser.parse_args()
    out=args.out.resolve()
    if out==STUDY or STUDY in out.parents:
        raise ValueError('Validation output must be outside released study')
    def redirected(path,record):
        out.parent.mkdir(parents=True,exist_ok=True)
        out.write_text(json.dumps(record,indent=2)+'\n')
    module.atomic_json=redirected
    record=module.verify_run(STUDY/'data/capture')
    print(json.dumps(record,indent=2))

if __name__=='__main__':main()
