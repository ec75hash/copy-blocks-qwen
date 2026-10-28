#!/usr/bin/env python3
"""Check the v0.2.0 study reductions, exact examples and source preservation offline."""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import numpy as np

ROOT=Path(__file__).resolve().parents[1]

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(16*1024*1024),b''):h.update(b)
    return h.hexdigest()

def equal(a,b,path='root'):
    if isinstance(a,dict):
        assert isinstance(b,dict) and a.keys()==b.keys(),path
        for k in a:equal(a[k],b[k],path+'.'+k)
    elif isinstance(a,list):
        assert len(a)==len(b),path
        for i,(x,y) in enumerate(zip(a,b)):equal(x,y,f'{path}[{i}]')
    elif isinstance(a,(float,int)):
        assert np.isclose(a,b,rtol=1e-11,atol=1e-12), (path,a,b)
    else:assert a==b,(path,a,b)

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--reproduced',type=Path,default=ROOT/'reproduction')
    args=ap.parse_args();redo=args.reproduced.resolve()
    source=json.loads((ROOT/'provenance/update_20261027.json').read_text())
    for rec in source['copies']:
        assert sha(ROOT/rec['destination'])==rec['sha256'],rec['destination']
    for name,expected in source['original_survey_unchanged'].items():
        assert sha(ROOT/name)==expected,name
    for rec in source.get('source_copy_amendments',[]):
        assert sha(ROOT/rec['path'])==rec['released_sha256'],rec['path']
    for rec in source.get('original_survey_description_amendments',[]):
        assert sha(ROOT/rec['path'])==rec['sha256_after'],rec['path']
    amendments=json.loads((ROOT/'docs/AMENDMENTS.json').read_text())
    for rec in amendments['entries']+amendments['presentation_changes']:
        assert sha(ROOT/rec['path'])==rec['sha256_after'],rec['path']
    weights=json.loads((redo/'qwen3_weights_readout/summary.json').read_text())
    expected=json.loads((ROOT/'studies/qwen3_weights/analysis/summary.json').read_text())
    equal(weights,expected,'weight_reduction')
    # The additional HF viewer tables must be exact projections of the retained source tables.
    study=ROOT/'studies/qwen3_weights'
    table=list(csv.DictReader((study/'analysis/mlp_layer_pairs.csv').open()))
    assert len(table)==6048
    lookup={(r['family'],r['layer_a'],r['layer_b']):r for r in table}
    assert len(lookup)==6048
    for fam in ['gate_proj','up_proj','down_proj']:
        for row in csv.DictReader((study/'data'/fam/'pairs.csv').open()):
            got=lookup['mlp.'+fam+'.weight',row['layer_a'],row['layer_b']]
            assert all(got[k]==v for k,v in row.items())
            assert got['revision']=='9216db5781bf21249d130ec9da846c4624c16137'
    table=list(csv.DictReader((study/'analysis/cross_checkpoint_gate_rows.csv').open()))
    assert len(table)==64
    cross=json.loads((study/'data/cross_qwen3_14b/mlp_gate/results.json').read_text())
    lookup={(int(r['layer_a']),int(r['layer_b'])):r for r in table}
    for row in cross['results']:
        for layer,data in row['candidates'].items():
            got=lookup[row['L14'],int(layer)]
            assert float(got['aligned_row_mean'])==data['row_mean']
            assert float(got['flat_cosine_shared_rows'])==data['flat_cosine_shared_rows']
    qwen25=json.loads((redo/'qwen25_weights_readout/summary.json').read_text())
    expected25=json.loads((ROOT/'studies/qwen25_weights/analysis/summary.json').read_text())
    equal(qwen25,expected25,'qwen25_reduction')
    assert qwen25['status']=='passed' and qwen25['cross_pair_rows_checked']==18432
    assert qwen25['mapped_summary_rows_checked']==24 and qwen25['endpoint_residual_rows_checked']==96
    assert qwen25['payload_manifest_entries_checked']==672
    response=ROOT/'studies/qwen3_response'
    released=json.loads((response/'analysis/summary.json').read_text())
    recomputed=json.loads((redo/'qwen3_response/summary.json').read_text())
    equal(released,recomputed,'response_reduction')
    detail=json.loads((redo/'qwen3_response/validation.json').read_text())
    assert detail['correlation_rows_recomputed']==864 and detail['overlap_rows_recomputed']==5184
    assert detail['correlation_ranks_recomputed']==864
    assert detail['max_correlation_summary_error']==0 and detail['max_control_overlap_mean_error']==0
    capture=json.loads((redo/'qwen3_response/CAPTURE_VALIDATION.json').read_text())
    assert capture['status']=='passed' and capture['raw_passages_independently_reduced']==32
    assert capture['committed_passages']==10000 and capture['retained_raw_passages']==512
    run=response/'data/capture'
    layout=json.loads((run/'layout.json').read_text())
    tokens=np.load(run/'tokens.npy',mmap_mode='r')
    peaks=np.load(run/'peaks.npy',mmap_mode='r')
    positions=np.load(run/'positions.npy',mmap_mode='r')
    passages=[json.loads(x) for x in (run/'passages.jsonl').read_text().splitlines()][1:]
    assert len(passages)==len({x['text'] for x in passages})==10000
    cases=json.loads((response/'analysis/examples.json').read_text());count=0
    for case in cases:
        events=[e for m in case['members'] for e in m['events']]
        events += [e for pair in case['shared_window_examples'] for e in [pair['source'],pair['partner']]]
        for event in events:
            layer=layout['layers'][str(event['layer'])]
            col=layer['start']+layer['ids'].index(event['neuron'])
            i,r=event['passage_index'],event['sign_regime']
            assert event['token_position']==int(positions[i,r,col])
            assert event['token_id']==int(tokens[i+1,event['token_position']])
            assert event['full_window']==passages[i]['text']
            for comp,name in enumerate(['gate_pre_silu','value','activation']):
                assert event[name]==float(peaks[i,r,comp,col])
            count+=1
    assert count==52
    pins=json.loads((run/'inputs.json').read_text())
    assert pins['model_revision']=='938a7432affaec9157f883a87164e2646ae17555'
    assert pins['model_sha256']=='2c50eb8aad05047dbf24fa014eb621adf552e14176cabe0c5db4ef38c91e2169'
    assert pins['dataset_revision']=='b08601e04326c79dfdd32d625aee71d232d685c3'
    result={'status':'passed','source_copies_checked':len(source['copies']),
            'original_survey_files_unchanged':len(source['original_survey_unchanged']),
            'source_copy_amendments_checked':len(source.get('source_copy_amendments',[])),
            'public_metadata_amendments_checked':len(amendments['entries']),
            'annotated_events_checked':count,'response_rows_checked':864+5184,
            'qwen25_cross_pair_rows_checked':18432,
            'new_inference':False,'remote_mutation':False}
    (redo/'UPDATE_VALIDATION.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
