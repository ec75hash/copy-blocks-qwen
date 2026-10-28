#!/usr/bin/env python3
"""Check saved Base-study tables and regenerate RESULTS.md offline (no raw weights)."""
from pathlib import Path
import argparse, csv, json
from collections import defaultdict
import numpy as np
S=Path(__file__).resolve().parents[1]
def table(name):
    with (S/'data'/name).open(newline='') as f:return list(csv.DictReader(f))
def close(a,b):
    np.testing.assert_allclose(a,b,rtol=0,atol=1e-12)
def mean(rows,key):return float(np.mean([float(r[key]) for r in rows]))
def primary(r):return r['map_group']=='plus24' and 8<=int(r['source_layer'])<32

def check():
    summary=json.loads((S/'data/summary.json').read_text())
    plan=json.loads((S/'FETCH_PLAN.json').read_text())
    sample=np.array(plan['sample_rows']);chunks=sample.reshape(17,32)
    assert len(set(sample))==544 and sample.min()>=0 and sample.max()<17408
    assert np.all(chunks==chunks[:,0,None]+np.arange(32))
    assert (sample<8192).sum()==256 and (sample>=8192).sum()==288
    attention=table('attention_mapped_pairs.csv');assert len(attention)==336
    gate=table('gate_mapped_pairs.csv');assert len(gate)==168
    expected_pairs={(i,i,'same_index') for i in range(32)}|{(i,i+24,'plus24') for i in range(8,40)}
    for rows in (attention,gate):
        families=sorted({r.get('family','gate') for r in rows})
        for family in families:
            rr=[r for r in rows if r.get('family','gate')==family]
            for label in ('base14_to_32','it14_to_32','base14_to_it14'):
                pairs={(int(r['source_layer']),int(r['target_layer']),r['map_group']) for r in rr if r['comparison']==label}
                assert pairs==(expected_pairs if label!='base14_to_it14' else {(i,i,'same_layer') for i in range(40)})
    for family,stem in [('self_attn.k_proj.weight','k_proj'),('self_attn.v_proj.weight','v_proj')]:
        rr=[r for r in attention if r['family']==family];s=summary[family];mats={}
        for label in ('base14_to_32','it14_to_32','base14_to_it14'):
            m=np.load(S/'data'/f'{stem}_{label}.npy',allow_pickle=False)
            assert m.shape==((40,40) if label=='base14_to_it14' else (40,64)) and np.isfinite(m).all()
            assert np.max(np.abs(m))<=1+1e-9;mats[label]=m
            for r in rr:
                if r['comparison']==label:
                    # Full-tensor BLAS and row-wise einsum use different summation orders.
                    np.testing.assert_allclose(m[int(r['source_layer']),int(r['target_layer'])],float(r['flat_cosine']),rtol=0,atol=1e-10)
        for label,key in [('base14_to_32','base_mean'),('it14_to_32','it_mean')]:
            pp=[r for r in rr if r['comparison']==label and primary(r)]
            assert len(pp)==s['primary_pairs']==24;close(mean(pp,'flat_cosine'),s[key])
        close(s['base_mean']-s['it_mean'],s['paired_difference_mean'])
        close(np.diag(mats['base14_to_it14']).mean(),s['base_to_it_same_layer_mean'])
        for label,key in [('base14_to_32','base_best_targets'),('it14_to_32','it_best_targets')]:
            assert np.argmax(mats[label],axis=1).tolist()==s[key]
        assert s['base_best_targets']==s['it_best_targets']
    groups=defaultdict(list)
    neurons=table('gate_neuron_pairs.csv');assert len(neurons)==91392
    for r in neurons:groups[(r['comparison'],int(r['source_layer']),int(r['target_layer']))].append(r)
    assert len(groups)==168
    for r in gate:
        rr=groups[(r['comparison'],int(r['source_layer']),int(r['target_layer']))]
        assert len(rr)==int(r['valid_rows'])==int(r['total_rows'])==544
        assert sorted(int(x['neuron']) for x in rr)==sorted(sample.tolist())
        values=np.array([float(x['row_cosine']) for x in rr]);assert np.isfinite(values).all()
        close(values.mean(),float(r['row_mean']));close(np.median(values),float(r['row_median']))
    sg=summary['sampled_gate'];assert sg['rows']==544 and sg['primary_pairs']==24
    for label in ('base14_to_32','it14_to_32'):
        rr=[r for r in gate if r['comparison']==label and primary(r)];assert len(rr)==24
        for key in ('flat_cosine','row_mean','row_median'):close(mean(rr,key),sg[label][key])
    close(sg['base14_to_32']['row_mean']-sg['it14_to_32']['row_mean'],sg['paired_row_mean_difference'])
    close(mean([r for r in gate if r['comparison']=='base14_to_it14'],'row_mean'),sg['base_to_it_same_layer_row_mean'])
    pairs=table('base30_within_pairs.csv');offsets=table('base30_offsets.csv')
    assert len(pairs)==2256 and len(offsets)==94
    for family in ('self_attn.k_proj.weight','mlp.gate.weight'):
        pp=[r for r in pairs if r['family']==family];ss=summary['base30_'+family]
        assert {(int(r['layer_a']),int(r['layer_b'])) for r in pp}=={(a,b) for a in range(48) for b in range(a+1,48)}
        for r in pp:
            a,b=int(r['layer_a']),int(r['layer_b']);assert int(r['offset'])==b-a
            aligned=6<=a<18 and b-a in (12,24) or 18<=a<30 and b-a==12
            assert (r['both_inside_region']=='True')==aligned
        eligible=[]
        for r in (x for x in offsets if x['family']==family):
            k=int(r['offset']);rr=[x for x in pp if int(x['offset'])==k]
            assert len(rr)==int(r['n_pairs']);close(mean(rr,'cosine'),float(r['mean_cosine']))
            close(mean(rr,'row_mean_cosine'),float(r['mean_row_mean_cosine']))
            assert (r['rank_eligible']=='True')==(len(rr)>=6)
            if len(rr)>=6:eligible.append(r)
        eligible.sort(key=lambda r:float(r['mean_cosine']),reverse=True)
        assert [int(r['offset']) for r in eligible[:4]]==[r['offset'] for r in ss['top_offsets']]
        close(mean(pp,'cosine'),ss['global_pair_mean'])
        for k in (12,24):
            rr=[r for r in pp if int(r['offset'])==k];p=ss['periods'][str(k)]
            inside=[r for r in rr if r['both_inside_region']=='True'];outside=[r for r in rr if r['both_inside_region']=='False']
            assert len(rr)==p['n_pairs'] and len(inside)==p['n_inside'] and len(outside)==p['n_outside']
            close(mean(rr,'cosine'),p['all_valid_mean'])
            for group,label in [(inside,'inside'),(outside,'outside')]:
                close(mean(group,'cosine'),p[label+'_mean']);close(mean(group,'row_mean_cosine'),p[label+'_row_mean'])
            close(min(float(r['cosine']) for r in inside),p['inside_min']);close(max(float(r['cosine']) for r in inside),p['inside_max'])
    return summary, {'attention_pairs':len(attention),'gate_pairs':len(gate),'gate_rows':len(neurons),'base30_pairs':len(pairs),'base30_offsets':len(offsets),'status':'passed'}

def render(s):
    lines=['# Base-checkpoint follow-up: measured results','',
    'This separate weight study changes the 14B reference while keeping Qwen3-32B fixed. It does not measure neuron responses.','',
    'The recorded prediction was: “PT vs IT numbers will show PT retains higher cosine similarities between weights, but not neurons”. Here PT means the published Base checkpoint, IT means the published non-Base checkpoint, and the neuron measurement is sampled gate-weight rows. No threshold or prediction score is assigned.','',
    'The primary comparison uses 24 layer pairs: 14B layers 8–31 against 32B layers 32–55. Both references use the same 544 gate indices in 17 contiguous 32-row clusters (seed 20261027).','',
    '| Quantity | Base 14B reference | IT 14B reference | Base minus IT |','|---|---:|---:|---:|']
    for fam in ('self_attn.k_proj.weight','self_attn.v_proj.weight'):
        x=s[fam];lines.append(f"| {fam} flattened cosine | {x['base_mean']:.6f} | {x['it_mean']:.6f} | {x['paired_difference_mean']:+.6f} |")
    g=s['sampled_gate'];lines.append(f"| Sampled gate aligned-row mean | {g['base14_to_32']['row_mean']:.6f} | {g['it14_to_32']['row_mean']:.6f} | {g['paired_row_mean_difference']:+.6f} |")
    lines+=['','The mean changes are less than 0.0002. The best matching target layer for every source layer is unchanged in both full K/V matrices. These measurements do not determine construction order or shared activation features. The gate result covers the sampled rows.','',
    '## Reference calibration','','| Base 14B versus IT 14B, same layer | Mean cosine |','|---|---:|']
    for fam in ('self_attn.k_proj.weight','self_attn.v_proj.weight'):lines.append(f"| {fam} flattened | {s[fam]['base_to_it_same_layer_mean']:.6f} |")
    lines.append(f"| Sampled gate aligned-row mean | {g['base_to_it_same_layer_row_mean']:.6f} |")
    lines+=['','## Published Base 30B checkpoint, within itself','',
    'The declared spacing K is 12 layers. The three blocks are layers 6–17, 18–29 and 30–41 (zero-based, inclusive). At K=12 there are 24 corresponding pairs and 12 other pairs. This is a within-checkpoint comparison, separate from the 14B/32B map.','',
    '| Family | Within K=12 | Other K=12 pairs | Aligned-row within | Aligned-row other | Top eligible offset |','|---|---:|---:|---:|---:|---:|']
    for fam in ('self_attn.k_proj.weight','mlp.gate.weight'):
        x=s['base30_'+fam];p=x['periods']['12'];lines.append(f"| {fam} | {p['inside_mean']:.6f} | {p['outside_mean']:.6f} | {p['inside_row_mean']:.6f} | {p['outside_row_mean']:.6f} | {x['top_offsets'][0]['offset']} |")
    lines+=['','All reported means weight layer pairs equally. Eligible offsets contain at least six pairs; this is a reporting rule, not a significance test. Full pairs, both layer maps, row values and denominators are in [data](data/); see [column definitions](COLUMNS.md) and [methods and reproduction](README.md).','']
    return '\n'.join(lines)

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--out',type=Path,required=True);args=ap.parse_args()
    out=args.out.resolve()
    if out==S or S in out.parents:raise ValueError('Choose an output directory outside the released study')
    s,report=check();out.mkdir(parents=True,exist_ok=True)
    (out/'RESULTS.md').write_text(render(s));(out/'verification.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
if __name__=='__main__':main()
