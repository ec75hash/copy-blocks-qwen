"""Write descriptive tables from committed data only; does not infer meanings."""
from pathlib import Path
import csv
import json
import math
import numpy as np
from core import Moments,COMPONENTS,REGIMES,finite,rank,top_indices,atomic_json

def write_csv(path,rows,fields):
    tmp=path.with_suffix(path.suffix+'.tmp')
    with tmp.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
    tmp.replace(path)

def make_report(run):
    run=Path(run)
    if not (run/'checkpoint.npz').exists():raise ValueError('No committed captures yet')
    lay=json.loads((run/'layout.json').read_text());inputs=json.loads((run/'inputs.json').read_text())
    moments=Moments(lay['pairs'])
    with np.load(run/'checkpoint.npz') as z:
        completed=int(z['completed']);raw=int(z['raw_written']);moments.restore(z)
    out=run/'reports';out.mkdir(exist_ok=True)
    if not completed:
        summary={'committed_passages':0,'planned_passages':inputs['settings']['passages'],'mode':'No activation passages committed'}
        atomic_json(out/'summary.json',summary)
        (out/'READOUT.txt').write_text('No activation passages committed; resume the same run to collect data.\n')
        return summary
    corr=moments.correlations();np.save(out/'correlations.npy',corr)
    rows=[]
    for pi,p in enumerate(lay['pairs']):
        for c,component in enumerate(COMPONENTS):
            for i,neuron in enumerate(p['neurons']):
                true=finite(corr[pi,c,i,p['target_columns'][i]])
                controls=corr[pi,c,i,p['control_columns']];controls=controls[np.isfinite(controls)]
                rows.append({'source_layer':p['source'],'target_layer':p['target'],'neuron':neuron,
                    'segment':p['segments'][str(neuron)],'component':component,'paired_correlation':true,
                    'control_mean':finite(controls.mean()) if len(controls) else None,
                    'paired_rank':rank(true,controls),'usable_controls':len(controls),
                    'token_observations':moments.n})
    write_csv(out/'correlations.csv',rows,list(rows[0]))
    peaks=np.load(run/'peaks.npy',mmap_mode='r')
    # Product extrema are retained separately for each gate/value sign combination.
    overlap=[];topk_values=[k for k in [10,100,500] if k<=completed//2]
    for pi,p in enumerate(lay['pairs']):
        xl=lay['layers'][str(p['source'])];yl=lay['layers'][str(p['target'])]
        x=peaks[:completed,:,2,xl['start']:xl['end']]
        y=peaks[:completed,:,2,yl['start']:yl['end']]
        views=[('positive_all_regimes',False,[0,3]),('negative_all_regimes',True,[1,2])]
        views += [(name,r in [1,2],[r]) for r,name in enumerate(REGIMES)]
        for label,negative,regimes in views:
            def scores(a):
                subset=np.asarray(a[:,regimes,:])
                v=np.max(np.where(np.isfinite(subset),-subset if negative else subset,-np.inf),axis=1)
                return np.where(np.isfinite(v),-v if negative else v,np.nan)
            sx=scores(x);sy=scores(y)
            for k in topk_values:
                sets_y=[top_indices(sy[:,j],k,negative) for j in range(sy.shape[1])]
                for i,neuron in enumerate(p['neurons']):
                    a,issue=top_indices(sx[:,i],k,negative)
                    b,partner_issue=sets_y[p['target_columns'][i]]
                    true=None if a is None or b is None else len(a&b)/len(a|b)
                    others=[];counts=[]
                    if a is not None:
                        for j in p['control_columns']:
                            s,_=sets_y[j]
                            if s is not None:others.append(len(a&s)/len(a|s));counts.append(len(a&s))
                    overlap.append({'source_layer':p['source'],'target_layer':p['target'],'neuron':neuron,
                        'segment':p['segments'][str(neuron)],'response_group':label,'top_k':k,
                        'shared_passages':None if a is None or b is None else len(a&b),
                        'paired_jaccard':true,'control_shared_mean':float(np.mean(counts)) if counts else None,
                        'control_jaccard_mean':float(np.mean(others)) if others else None,
                        'paired_rank':rank(true,others),'usable_controls':len(others),
                        'source_issue':issue,'partner_issue':partner_issue})
    fields=['source_layer','target_layer','neuron','segment','response_group','top_k','shared_passages',
            'paired_jaccard','control_shared_mean','control_jaccard_mean','paired_rank','usable_controls','source_issue','partner_issue']
    write_csv(out/'passage_overlap.csv',overlap,fields)
    # References to strongest positive/negative passages, gate and value at that token.
    pos=np.load(run/'positions.npy',mmap_mode='r');examples=[]
    for p in lay['pairs']:
        for layer in [p['source'],p['target']]:
            loc=lay['layers'][str(layer)]
            for neuron in p['neurons']:
                col=loc['start']+loc['ids'].index(neuron)
                for r,label in enumerate(REGIMES):
                    vals=np.asarray(peaks[:completed,r,2,col]);valid=np.flatnonzero(np.isfinite(vals))
                    order=valid[np.argsort(np.abs(vals[valid]))[-25:][::-1]]
                    for index in order:
                        examples.append({'layer':layer,'neuron':neuron,'sign_regime':label,'passage_index':int(index),
                            'passage_id':f'p{index:05d}','token_position':int(pos[index,r,col]),
                            'gate_pre_silu':float(peaks[index,r,0,col]),'value':float(peaks[index,r,1,col]),
                            'activation':float(vals[index])})
    write_csv(out/'strong_passage_references.csv',examples,
        ['layer','neuron','sign_regime','passage_index','passage_id','token_position','gate_pre_silu','value','activation'])
    status=json.loads((run/'status.json').read_text()) if (run/'status.json').exists() else {}
    summary={'mode':'descriptive measurements only; no predictions, significance tests or semantic labels',
        'committed_passages':completed,'planned_passages':inputs['settings']['passages'],
        'tokens_per_neuron':moments.n,'neuron_pairs':288,'layer_pairs':24,
        'raw_passages_retained':raw,'correlation_rows':len(rows),'overlap_rows':len(overlap),
        'top_k_values':topk_values,'underlying_captured_components':COMPONENTS,
        'sign_regimes':REGIMES,'control_design':'100 shared random other-index controls per layer pair' if inputs['settings']['controls']==100 else f"{inputs['settings']['controls']} shared controls per layer pair",
        'selection_note':'Four selected neuron indices per segment per layer pair; report segments separately rather than treating equal sampling as population weights',
        'status_at_report':status.get('phase')}
    atomic_json(out/'summary.json',summary)
    text=['OVERNIGHT QWEN3-32B Q8 COLLECTION — MEASUREMENTS ONLY','',
        f"Committed passages: {completed:,} of {inputs['settings']['passages']:,}",
        f'Token observations per neuron: {moments.n:,}',
        '288 corresponding neuron pairs across all 24 layer pairs; original zero-based indices.',
        f"{inputs['settings']['controls']} shared random different-index controls per layer pair.",
        f'Detailed raw paired-neuron recordings retained for {raw} preselected passages.','',
        'correlations.csv: separate gate, value and final-activation response correlations.',
        'passage_overlap.csv: positive/negative response overlap and four gate/value sign regimes.',
        'strong_passage_references.csv: exact passage/token locations and measured gate/value/product.',
        'No neuron meanings, predictions, significance tests or pass/fail decisions are assigned.','',
        'The .npy files have space reserved for the whole planned run.',
        'Only rows 0 through committed_passages-1 contain committed observations.',
        'Raw files contain paired channels only; controls retain streaming correlations and per-passage summaries.',
        'Exact text is in passages.jsonl; exact token IDs are in tokens.npy (row 0 is instrumentation).']
    (out/'READOUT.txt').write_text('\n'.join(text)+'\n')
    return summary
