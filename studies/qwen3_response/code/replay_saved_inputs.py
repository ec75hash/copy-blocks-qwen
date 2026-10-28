"""Fresh macOS/Metal inference on exact frozen tokens; no corpus parser rerun.

The wrapper was prepared for portability; a new full replay was not run for v0.2.0.
"""
from pathlib import Path
import argparse
import json
import os
import shutil
import sys
import numpy as np

STUDY=Path(__file__).resolve().parents[1]

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--model',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--limit',type=int,default=10000,help='Use a separate output directory for a short instrumentation replay')
    args=ap.parse_args()
    out=args.out.resolve()
    if not 1<=args.limit<=10000:ap.error('--limit must be 1..10000')
    if out.exists():ap.error('Use a new output directory; frozen evidence and previous captures must not be overwritten')
    if out==STUDY or STUDY in out.parents:ap.error('Output must be outside released study')
    if not args.model.is_file():ap.error('Pinned Q8 model file not found')
    os.environ['QWEN_Q8_MODEL']=str(args.model.resolve())
    from backend import Capture,check_model
    from core import Moments,open_stores,commit,passage_arrays,atomic_npz
    frozen=STUDY/'data/capture'
    layout=json.loads((frozen/'layout.json').read_text())
    full_columns=json.loads((frozen/'raw_columns.json').read_text())
    columns=[r['flat_column'] for r in full_columns]
    keep=set(json.loads((frozen/'raw_indices.json').read_text()))
    tokens=np.load(frozen/'tokens.npy',mmap_mode='r')
    check_model()
    out.mkdir(parents=True)
    (out/'raw').mkdir()
    shutil.copy2(frozen/'layout.json',out/'layout.json')
    shutil.copy2(frozen/'inputs.json',out/'ORIGINAL_INPUTS.json')
    stores=open_stores(out,args.limit,layout['total_channels'],True)
    moments=Moments(layout['pairs']);raw_count=0
    engine=Capture(layout,256)
    try:
        engine.validate(tokens[0],out)
        for i in range(args.limit):
            values,_=engine.evaluate(tokens[i+1])
            peak,pos,count,local=passage_arrays(values)
            for key,value in [('peaks',peak),('positions',pos),('sign_counts',count),('passage_moments',local)]:
                stores[key][i]=value
            moments.update(values,layout)
            if i in keep:
                atomic_npz(out/'raw'/f'p{i:05d}.npz',ids=np.array(tokens[i+1]),values=values[:,:,columns],passage_index=np.array(i))
                raw_count+=1
            if (i+1)%16==0 or i+1==args.limit:
                commit(out,stores,moments,i+1,raw_count)
                print(f'{i+1}/{args.limit} windows committed',flush=True)
    finally:engine.close()
    (out/'REPLAY.json').write_text(json.dumps({'completed':args.limit,'source':'Exact released token windows, no parser rerun',
        'fresh_capture':True,'resume_supported':False,'original_data_modified':False},indent=2)+'\n')

if __name__=='__main__':main()
