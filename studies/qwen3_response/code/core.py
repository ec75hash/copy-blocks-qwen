"""Storage and descriptive statistics for the unattended Q8 collection."""
from pathlib import Path
import hashlib
import json
import os
import random

import numpy as np

SCHEMA = 1
COMPONENTS = ['gate_pre_silu', 'value', 'activation']
REGIMES = ['positive_gate_positive_value', 'positive_gate_negative_value',
           'negative_gate_positive_value', 'negative_gate_negative_value']

def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(16*1024*1024),b''):h.update(b)
    return h.hexdigest()

def atomic_json(path,value):
    path=Path(path);tmp=path.with_suffix(path.suffix+'.tmp')
    with tmp.open('w') as f:
        json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
        f.flush();os.fsync(f.fileno())
    tmp.replace(path)

def atomic_npz(path,**arrays):
    path=Path(path);tmp=path.with_suffix(path.suffix+'.tmp')
    with tmp.open('wb') as f:
        np.savez_compressed(f,**arrays);f.flush();os.fsync(f.fileno())
    tmp.replace(path)

def layout(selection,controls,seed):
    """The existing 12 independently selected indices at each source layer."""
    rng=random.Random(seed);pairs=[];layers={};offset=0
    for source in range(8,32):
        rows=[r for r in selection['targets'] if r['layer_a']==source]
        ids=sorted(r['neuron'] for r in rows)
        if len(ids)!=12:raise ValueError(f'Expected 12 selected neurons at layer {source}')
        available=[i for i in range(25600) if i not in ids]
        bank=sorted(rng.sample(available,controls))
        target_ids=sorted(ids+bank)
        pairs.append({'source':source,'target':source+24,'neurons':ids,
                      'segments':{str(r['neuron']):r['segment'] for r in rows},
                      'controls':bank,'target_columns':[target_ids.index(i) for i in ids],
                      'control_columns':[target_ids.index(i) for i in bank]})
        layers[source]={'ids':ids};layers[source+24]={'ids':target_ids}
    for layer in sorted(layers):
        item=layers[layer];item['start']=offset;offset+=len(item['ids']);item['end']=offset
    return {'pairs':pairs,'layers':{str(k):v for k,v in sorted(layers.items())},
            'total_channels':offset,'controls_shared_within_each_layer_pair':True}

def bytes_required(passages,tokens,raw_passages,channels=2976):
    # Four sign regimes; peak product plus its gate/value; token positions;
    # per-regime counts; per-component mean/RMS. Raw uses 576 paired channels.
    dense=passages*channels*(4*3*4 + 4*2 + 4*2 + 2*3*4)
    raw=min(passages,raw_passages)*tokens*576*3*4
    return dense+raw+250_000_000  # provenance, text, summaries, checkpoints

class Moments:
    """Merge centered batch covariance in float64; no full token archive needed."""
    def __init__(self,pairs,components=3):
        self.n=0;self.pairs=pairs
        ns=len(pairs[0]['neurons']);nt=ns+len(pairs[0]['controls'])
        shape=(len(pairs),components)
        self.mean_x=np.zeros(shape+(ns,),np.float64)
        self.mean_y=np.zeros(shape+(nt,),np.float64)
        self.m2_x=np.zeros_like(self.mean_x);self.m2_y=np.zeros_like(self.mean_y)
        self.cross=np.zeros(shape+(ns,nt),np.float64)
    def update(self,values,lay):
        # values = [token, component, flat_channel], always finite float32.
        count=values.shape[0];new=self.n+count
        factor=self.n*count/new
        for pi,p in enumerate(self.pairs):
            xloc=lay['layers'][str(p['source'])];yloc=lay['layers'][str(p['target'])]
            for c in range(3):
                x=values[:,c,xloc['start']:xloc['end']].astype(np.float64)
                y=values[:,c,yloc['start']:yloc['end']].astype(np.float64)
                bx=x.mean(0);by=y.mean(0);dx=bx-self.mean_x[pi,c];dy=by-self.mean_y[pi,c]
                x-=bx;y-=by
                self.cross[pi,c]+=x.T@y+np.outer(dx,dy)*factor
                self.m2_x[pi,c]+=(x*x).sum(0)+dx*dx*factor
                self.m2_y[pi,c]+=(y*y).sum(0)+dy*dy*factor
                self.mean_x[pi,c]+=dx*count/new;self.mean_y[pi,c]+=dy*count/new
        self.n=new
    def correlations(self):
        denom=np.sqrt(np.maximum(self.m2_x,0)[...,None]*np.maximum(self.m2_y,0)[...,None,:])
        return np.divide(self.cross,denom,out=np.full_like(self.cross,np.nan),where=denom>0)
    def arrays(self):
        return {k:getattr(self,k) for k in ['mean_x','mean_y','m2_x','m2_y','cross']}
    def restore(self,z):
        self.n=int(z['tokens_seen'])
        for k in self.arrays():getattr(self,k)[:]=z[k]

def passage_arrays(values):
    """Peak actual activation per sign regime, with gate/value at that token."""
    tokens,_,channels=values.shape
    g=values[:,0];v=values[:,1];a=values[:,2]
    masks=[(g>=0)&(v>=0)&(a!=0),(g>=0)&(v<0)&(a!=0),
           (g<0)&(v>=0)&(a!=0),(g<0)&(v<0)&(a!=0)]
    peaks=np.full((4,3,channels),np.nan,np.float32)
    positions=np.full((4,channels),65535,np.uint16)
    counts=np.zeros((4,channels),np.uint16)
    columns=np.arange(channels)
    for r,mask in enumerate(masks):
        counts[r]=mask.sum(0)
        score=np.where(mask,np.abs(a),-np.inf)
        idx=score.argmax(0);exists=mask.any(0)
        positions[r,exists]=idx[exists]
        for c in range(3):peaks[r,c,exists]=values[idx[exists],c,columns[exists]]
    # Axis 0 mean and RMS; component order gate, value, activation.
    x=values.astype(np.float64)
    local=np.stack([x.mean(0),np.sqrt((x*x).mean(0))]).astype(np.float32)
    return peaks,positions,counts,local

def open_stores(run,n,channels,new):
    specs={'peaks':((n,4,3,channels),np.float32),
           'positions':((n,4,channels),np.uint16),
           'sign_counts':((n,4,channels),np.uint16),
           'passage_moments':((n,2,3,channels),np.float32)}
    stores={}
    for name,(shape,dtype) in specs.items():
        p=Path(run)/(name+'.npy')
        if new and not p.exists():
            stores[name]=np.lib.format.open_memmap(p,mode='w+',dtype=dtype,shape=shape)
        else:
            stores[name]=np.load(p,mmap_mode='r+')
            if stores[name].shape!=shape or stores[name].dtype!=dtype:raise ValueError(f'Storage schema mismatch: {p}')
    return stores

def commit(run,stores,moments,completed,raw_written):
    for v in stores.values():v.flush()
    atomic_npz(Path(run)/'checkpoint.npz',completed=np.array(completed,np.int64),
               raw_written=np.array(raw_written,np.int64),tokens_seen=np.array(moments.n,np.int64),
               **moments.arrays())

def top_indices(values,k,negative=False):
    score=-np.asarray(values) if negative else np.asarray(values)
    valid=np.flatnonzero(np.isfinite(score)&(score>0))
    if len(valid)<k:return None,'insufficient_signed_passages'
    order=valid[np.argsort(score[valid],kind='stable')[-k:]]
    cutoff=score[order[0]]
    if (score[valid]>=cutoff).sum()>k:return None,'tied_cutoff'
    return set(order.tolist()),None

def finite(value):
    return float(value) if np.isfinite(value) else None

def rank(value,others):
    if value is None or not np.isfinite(value):return None
    others=np.asarray(others);others=others[np.isfinite(others)]
    eq=np.isclose(others,value,rtol=0,atol=1e-12)
    return float(1+np.sum((others>value)&~eq)+.5*eq.sum())
