"""Independent, offline review of October 25 outputs; writes only to the separate output directory.

No historical scanner is imported or executed. Full saved-table reaggregation and
bounded BF16 payload checks, not a fresh full-model scan or functional experiment.
"""
from pathlib import Path
import csv
import hashlib
import json
import numpy as np

import argparse
STUDY = Path(__file__).resolve().parents[1]
ROOT = STUDY.parents[1]
parser = argparse.ArgumentParser(description='Independent offline reduction of released Qwen3 weight measurements')
parser.add_argument('--out', type=Path, default=ROOT/'reproduction/qwen3_weights_readout')
args = parser.parse_args()
OUT = args.out.resolve()
if OUT == STUDY or STUDY in OUT.parents:
    raise ValueError('Output must be outside released study')
OUT.mkdir(parents=True, exist_ok=True)
SRC = STUDY/'data' 
CACHE = Path.home() / '.cache/copy-blocks-qwen-tensors'
REV32 = '9216db5781bf21249d130ec9da846c4624c16137'
REV14 = '40c069824f4251a91eefaf281ebe4c544efd3e18'
report = {'scope': __doc__, 'within': {}, 'cross': {}, 'payloads': [], 'raw_checks': [], 'unavailable_payloads': []}
loadjson = lambda p: json.loads(p.read_text())

def close(a, b, tol=1e-7):
    assert abs(float(a)-float(b)) < tol, (a, b)

for fam in ('gate_proj', 'up_proj', 'down_proj'):
    p = SRC / fam
    rows = list(csv.DictReader((p / 'pairs.csv').open()))
    keys = {(int(r['layer_a']), int(r['layer_b'])) for r in rows}
    assert keys == {(i,j) for i in range(64) for j in range(i+1,64)}
    assert len(rows) == 2016
    for r in rows:
        i,j = int(r['layer_a']),int(r['layer_b'])
        assert int(r['offset']) == j-i
        assert (r['corresponding']=='True') == (8 <= i <= 31 and j==i+24)
        assert all(np.isfinite(float(r[k])) and -1.000001 <= float(r[k]) <= 1.000001
                   for k in ('cosine','row_mean_cosine','row_median_cosine','shuffled_row_mean'))
    s = loadjson(p/'summary.json')['by_period']['24']
    within = [r for r in rows if r['corresponding']=='True']
    outside = [r for r in rows if int(r['offset'])==24 and r['corresponding']=='False']
    assert (len(within),len(outside)) == (24,16)
    for group,prefix in ((within,'within'),(outside,'off')):
        for col,suffix in (('cosine','cosine'),('row_mean_cosine','row_mean'),('frac_row_gt_0.5','frac_gt_0.5')):
            close(np.mean([float(r[col]) for r in group]),s[prefix+'_'+suffix],1e-12)
    offsets = list(csv.DictReader((p/'offsets.csv').open()))
    for off in offsets:
        rs = [r for r in rows if r['offset']==off['offset']]
        assert len(rs)==int(off['n_pairs'])
        close(np.mean([float(r['cosine']) for r in rs]),off['mean_cosine'],1e-12)
    assert max(offsets,key=lambda r:float(r['mean_cosine']))['offset']=='24'
    with np.load(p/'neuron_cosines_offset24.npz') as z:
        assert len(z.files)==80
        for r in (r for r in rows if int(r['offset'])==24):
            key=f"L{r['layer_a']}_L{r['layer_b']}"
            v=z[key].astype(np.float64); sh=z[key+'_shuffled'].astype(np.float64)
            assert v.shape==(25600,) and np.isfinite(v).all()
            close(v.mean(),r['row_mean_cosine']);close(np.median(v),r['row_median_cosine'])
            close((v>.5).mean(),r['frac_row_gt_0.5']);close(sh.mean(),r['shuffled_row_mean'])
        v=np.stack([z[f'L{i}_L{i+24}'] for i in range(8,32)]).astype(np.float64)
    report['within'][fam] = dict(s, median=float(np.median(v)),p95=float(np.quantile(v,.95)),
        count_gt_0_7=int((v>.7).sum()),count_gt_0_5=int((v>.5).sum()),n=v.size,
        segment_means=[float(v[:,a:b].mean()) for a,b in ((0,8192),(8192,17408),(17408,25600))],
        saved_pairs_checked=len(rows),saved_neuron_vectors_checked=40)
    print('Saved within checked:',fam,flush=True)

cross = loadjson(SRC/'cross_qwen3_14b/summary.json')
for fam in ('k_proj','v_proj'):
    c=np.load(SRC/f'cross_qwen3_14b/cosine_{fam}.npy')
    s=cross['summary'][f'self_attn.{fam}.weight']
    assert c.shape==(40,64) and np.isfinite(c).all()
    assert c.argmax(1).tolist()==s['best_match_per_14B_layer']
    assert c.argmax(1).tolist()==[i if i<10 else i+24 for i in range(40)]
    close(np.mean([c[i,i+24] for i in range(8,32)]),s['twin_region_32B_L+24_mean'],1e-12)
    report['cross'][fam]={'later_mean':s['twin_region_32B_L+24_mean'],
                         'earlier_mean':float(np.mean([c[i,i] for i in range(8,32)]))}
mlp=loadjson(SRC/'cross_qwen3_14b/mlp_gate/results.json')
with np.load(SRC/'cross_qwen3_14b/mlp_gate/per_neuron_cosines.npz') as z:
    n=0
    for r in mlp['results']:
        for j,s in r['candidates'].items():
            v=z[f"L14_{r['L14']}_vs_L32_{j}"].astype(np.float64)
            assert v.shape==(17408,)
            close(v.mean(),s['row_mean']);close(np.median(v),s['row_median'])
            close((v>.5).mean(),s['frac_gt_0.5'])
            n+=1
    assert len(z.files)==n
report['cross']['mlp_saved_vectors_checked']=n
group=loadjson(SRC/'cross_qwen3_14b/heads/group_aware_mapping.json')
recipe=[0,1,2,0,1,2,3,4]
exact=total=0; exceptions=[]
for r in group:
    assert r['best_parent_in_same_group_frac']==1
    assert sum(r['slot_map_counts'].values())==64
    n=sum(r['slot_map_counts'].get(f'{i}->{j}',0) for i,j in enumerate(recipe))
    exact+=n;total+=64
    if n!=64:exceptions.append({'family':r['family'],'L14':r['L14'],'L32':r['L32'],'exceptions':64-n})
report['cross']['head_recipe']={'exact':exact,'total':total,'exceptions':exceptions,'sampled_layer_pairs_per_family':7}

report['status']='passed'
report['raw_scope']='Saved-table and vector reductions only; prior bounded BF16 checks are retained in audit/validation.json.'
(OUT/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
fig,axes=plt.subplots(1,3,figsize=(14,4.7),gridspec_kw={'width_ratios':[1.15,1.15,1]})
for ax,fam,letter in [(axes[0],'k_proj','A'),(axes[1],'v_proj','B')]:
 c=np.load(SRC/f'cross_qwen3_14b/cosine_{fam}.npy')
 im=ax.imshow(c,aspect='auto',vmin=0,vmax=1,cmap='viridis',origin='upper')
 ax.set(xlabel='Qwen3-32B decoder layer',ylabel='Qwen3-14B decoder layer',title=f'{letter}  Cross-checkpoint {fam}')
 fig.colorbar(im,ax=ax,fraction=.045,pad=.03,label='Full-tensor cosine')
dup=loadjson(SRC/'cross_qwen3_14b/mlp_gate/neuron_duplication_test.json')
axes[2].bar(range(5),[100*r['best_is_A_j'] for r in dup],color=['#9eafbd','#9eafbd','#245B88','#245B88','#245B88'])
axes[2].set_xticks(range(5),[str(r['L32']) for r in dup])
axes[2].set(xlabel='Qwen3-32B decoder layer',ylabel='Sample matching its leading index (%)',ylim=(0,108),title='C  Higher-index gate rows')
for i,r in enumerate(dup):axes[2].text(i,100*r['best_is_A_j']+2,f"{100*r['best_is_A_j']:.2f}",ha='center',fontsize=8)
fig.suptitle('Qwen3-14B / 32B: measured weight correspondences',fontsize=15)
fig.text(.02,.025,'A/B: pinned BF16 K/V tensors, all 40 × 64 layer pairs. C: 400 sampled higher-index rows per layer; nearest other row by absolute cosine.\nC_j = row 17408+j; leading counterpart A_j = row j. Early-layer exceptions shown. Construction history remains an interpretation.',fontsize=8,color='#444')
fig.tight_layout(rect=[0,.13,1,.94])
fig.savefig(OUT/'weight_correspondence.png',dpi=180)
fig.savefig(OUT/'weight_correspondence.svg')
print('Weight reductions and figure complete:',OUT)
