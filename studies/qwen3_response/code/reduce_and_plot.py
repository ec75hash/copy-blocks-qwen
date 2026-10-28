"""Exploratory summary of the frozen overnight Q8 collection; no new inference."""
from pathlib import Path
import csv
import hashlib
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import argparse
STUDY=Path(__file__).resolve().parents[1]
ROOT=STUDY.parents[1]
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--out',type=Path,default=ROOT/'reproduction/qwen3_response')
args=parser.parse_args()
HERE=args.out.resolve()
RUN=STUDY/'data/capture'
if HERE==STUDY or STUDY in HERE.parents:
    raise ValueError('Use an output directory outside the released study to preserve evidence')
HERE.mkdir(parents=True,exist_ok=True)
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(16*1024*1024),b''):h.update(b)
    return h.hexdigest()
def save(name,value):
    (HERE/name).write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
def number(s):return None if s=='' else float(s)
def top(v,k):
    valid=np.flatnonzero(np.isfinite(v)&(v>0))
    if len(valid)<k:return None,'insufficient_signed_passages'
    cutoff=np.partition(v[valid],-k)[-k]
    ids=np.flatnonzero(np.isfinite(v)&(v>=cutoff))
    return (set(ids.tolist()),None) if len(ids)==k else (None,'tied_cutoff')

completion=json.loads((RUN/'completion.json').read_text())
assert completion['phase']=='complete' and completion['validation_status']=='passed'
validation=json.loads((RUN/'VALIDATION.json').read_text())
hashes={}
for name,expected in validation['dense_array_sha256'].items():
    p=RUN/(name+'.npy');got=sha(p);assert got==expected,name;hashes[str(p.relative_to(ROOT))]=got
assert sha(RUN/'checkpoint.npz')==validation['checkpoint_sha256']
c=list(csv.DictReader((RUN/'reports/correlations.csv').open()))
o=list(csv.DictReader((RUN/'reports/passage_overlap.csv').open()))
lay=json.loads((RUN/'layout.json').read_text())
with np.load(RUN/'checkpoint.npz') as z:
    assert int(z['completed'])==10000 and int(z['tokens_seen'])==2560000
    corr=z['cross']/np.sqrt(z['m2_x'][...,None]*z['m2_y'][...,None,:])
ci={(int(r['source_layer']),int(r['neuron']),r['component']):r for r in c}
components=['gate_pre_silu','value','activation'];max_error=0.
for pi,pair in enumerate(lay['pairs']):
    for comp,component in enumerate(components):
        for ni,neuron in enumerate(pair['neurons']):
            row=ci[pair['source'],neuron,component];v=corr[pi,comp,ni,pair['target_columns'][ni]]
            controls=corr[pi,comp,ni,pair['control_columns']]
            error=max(abs(v-float(row['paired_correlation'])),abs(controls.mean()-float(row['control_mean'])))
            assert error<1e-12;max_error=max(max_error,float(error))
            expected_rank=1+int(np.sum(controls>v))+.5*int(np.sum(controls==v))
            assert expected_rank==float(row['paired_rank'])

peaks=np.load(RUN/'peaks.npy',mmap_mode='r');positions=np.load(RUN/'positions.npy',mmap_mode='r')
oi={(int(r['source_layer']),int(r['neuron']),r['response_group'],int(r['top_k'])):r for r in o}
groups=[('positive_all_regimes',[0,3],False),('negative_all_regimes',[1,2],True),
        ('positive_gate_positive_value',[0],False),('positive_gate_negative_value',[1],True),
        ('negative_gate_positive_value',[2],True),('negative_gate_negative_value',[3],False)]
overlap_checked=0;overlap_max_error=0.
for pair in lay['pairs']:
    xl=lay['layers'][str(pair['source'])];yl=lay['layers'][str(pair['target'])]
    x=np.asarray(peaks[:,:,2,xl['start']:xl['end']]);y=np.asarray(peaks[:,:,2,yl['start']:yl['end']])
    for name,regimes,negative in groups:
        def score(a):
            b=a[:,regimes,:];return np.max(np.where(np.isfinite(b),-b if negative else b,-np.inf),axis=1)
        sx,sy=score(x),score(y)
        for k in [10,100,500]:
            target=[top(sy[:,j],k) for j in range(sy.shape[1])]
            for ni,neuron in enumerate(pair['neurons']):
                row=oi[pair['source'],neuron,name,k];a,issue=top(sx[:,ni],k);b,bi=target[pair['target_columns'][ni]]
                assert (row['source_issue'] or None)==issue and (row['partner_issue'] or None)==bi
                n=None if a is None or b is None else len(a&b)
                assert number(row['shared_passages'])==n
                others=[]
                if a is not None:
                    others=[len(a&t[0]) for j in pair['control_columns'] if (t:=target[j])[0] is not None]
                assert int(row['usable_controls'])==len(others)
                if others:
                    error=abs(np.mean(others)-float(row['control_shared_mean']));assert error<1e-12
                    overlap_max_error=max(overlap_max_error,float(error))
                overlap_checked+=1

summary={'scope':'288 sampled within-checkpoint Q8 neuron pairs; descriptive and exploratory',
    'corpus_unit':'10,000 unique 256-token WikiText windows; parsed labels are not all genuine article titles',
    'components':{},'overlap':{},'layer_pairs':[]}
def cs(rows):
    a=np.array([float(r['paired_correlation']) for r in rows]);b=np.array([float(r['control_mean']) for r in rows]);ranks=np.array([float(r['paired_rank']) for r in rows])
    return {'n':len(rows),'paired_mean':float(a.mean()),'paired_median':float(np.median(a)),
        'control_mean':float(b.mean()),'control_mean_median':float(np.median(b)),
        'above_control_mean':int((a>b).sum()),'partner_rank_1':int((ranks==1).sum()),
        'partner_rank_le_10':int((ranks<=10).sum()),'paired_min':float(a.min()),'paired_max':float(a.max())}
for component in components:
    rows=[r for r in c if r['component']==component]
    summary['components'][component]=cs(rows)
    summary['components'][component]['segments']={s:cs([r for r in rows if r['segment']==s]) for s in ['A','B','C']}
for name,_,_ in groups:
    summary['overlap'][name]={}
    for k in [10,100,500]:
        rows=[r for r in o if r['response_group']==name and int(r['top_k'])==k]
        valid=[r for r in rows if r['shared_passages']!='' and r['control_shared_mean']!='']
        a=np.array([float(r['shared_passages']) for r in valid]);b=np.array([float(r['control_shared_mean']) for r in valid])
        summary['overlap'][name][str(k)]={'total_pairs':288,'usable_pairs':len(valid),
            'excluded_pairs':288-len(valid),'paired_mean_shared':float(a.mean()),'paired_median_shared':float(np.median(a)),
            'control_mean_shared':float(b.mean()),'control_mean_median_shared':float(np.median(b)),
            'above_control_mean':int((a>b).sum()),'zero_overlap_pairs':int((a==0).sum()),
            'minimum_usable_controls':min(int(r['usable_controls']) for r in valid)}
activation=[r for r in c if r['component']=='activation']
for layer in range(8,32):summary['layer_pairs'].append({'source':layer,'target':layer+24,**cs([r for r in activation if int(r['source_layer'])==layer])})

# Examples chosen by declared numerical rules, then inspected for context.
positive=[r for r in o if r['response_group']=='positive_all_regimes' and r['top_k']=='100' and r['shared_passages']!='']
negative=[r for r in o if r['response_group']=='negative_all_regimes' and r['top_k']=='100' and r['shared_passages']!='']
def key(r):return int(r['source_layer']),int(r['neuron'])
bestpos=max(positive,key=lambda r:(float(r['shared_passages']),float(ci[*key(r),'activation']['paired_correlation'])))
bestneg=max(negative,key=lambda r:(float(r['shared_passages']),float(ci[*key(r),'activation']['paired_correlation'])))
median=summary['components']['activation']['paired_median']
typical=min(activation,key=lambda r:(abs(float(r['paired_correlation'])-median),*key(r)))
lowest=min(activation,key=lambda r:(float(r['paired_correlation']),*key(r)))
choices=[('highest_positive_top100_overlap',bestpos,'positive'),('highest_negative_top100_overlap',bestneg,'negative'),
         ('nearest_median_activation_correlation',typical,'positive'),('lowest_signed_activation_correlation',lowest,'positive')]
cases=[]
for criterion,r,pole in choices:
    layer,neuron=key(r);pair=next(x for x in lay['pairs'] if x['source']==layer)
    case={'selection_rule':criterion,'source_layer':layer,'target_layer':layer+24,'neuron':neuron,
          'correlation':ci[layer,neuron,'activation'],'overlap':{p:oi[layer,neuron,p+'_all_regimes',100] for p in ['positive','negative']},
          'display_pole':pole,'members':[]}
    for L in [layer,layer+24]:
        loc=lay['layers'][str(L)];col=loc['start']+loc['ids'].index(neuron);reg=[0,3] if pole=='positive' else [1,2]
        vals=np.asarray(peaks[:,reg,2,col]);s=np.max(np.where(np.isfinite(vals),vals if pole=='positive' else -vals,-np.inf),axis=1)
        order=np.argsort(-s,kind='stable')[:5]
        events=[]
        for index in order:
            rr=reg[int(np.nanargmax(vals[index] if pole=='positive' else -vals[index]))]
            events.append({'passage_index':int(index),'passage_id':f'p{index:05d}',
                'token_position':int(positions[index,rr,col]),'sign_regime':rr,
                'gate_pre_silu':float(peaks[index,rr,0,col]),'value':float(peaks[index,rr,1,col]),
                'activation':float(peaks[index,rr,2,col]),'strength_tied_windows':int(np.sum(s==s[index]))})
        case['members'].append({'layer':L,'events':events})
    cases.append(case)
save('example_selection.json',cases)

# Post-hoc prefix sensitivity on retained raw passages only; controls have no raw archive.
raw_columns=json.loads((RUN/'raw_columns.json').read_text());raw_map={(r['layer'],r['neuron']):i for i,r in enumerate(raw_columns)}
source=[raw_map[int(r['source_layer']),int(r['neuron'])] for r in activation]
target=[raw_map[int(r['target_layer']),int(r['neuron'])] for r in activation]
acc={drop:{'n':0,'sx':np.zeros(288),'sy':np.zeros(288),'xx':np.zeros(288),'yy':np.zeros(288),'xy':np.zeros(288)} for drop in [0,8,32]}
for file in sorted((RUN/'raw').glob('*.npz')):
    with np.load(file) as z:v=z['values'][:,2,:].astype(np.float64)
    for drop,d in acc.items():
        x=v[drop:,source];y=v[drop:,target];d['n']+=len(x)
        for name,val in [('sx',x.sum(0)),('sy',y.sum(0)),('xx',(x*x).sum(0)),('yy',(y*y).sum(0)),('xy',(x*y).sum(0))]:d[name]+=val
sensitivity={};sensitivity_rows=[]
for drop,d in acc.items():
    n=d['n'];r=(d['xy']-d['sx']*d['sy']/n)/np.sqrt((d['xx']-d['sx']**2/n)*(d['yy']-d['sy']**2/n))
    sensitivity[str(drop)]={'raw_passages':512,'token_observations_per_neuron':n,'median_paired_correlation':float(np.nanmedian(r))}
    for i,row in enumerate(activation):sensitivity_rows.append({'source_layer':row['source_layer'],'neuron':row['neuron'],'drop_first_tokens':drop,'paired_correlation':float(r[i])})
with (HERE/'prefix_sensitivity.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(sensitivity_rows[0]));w.writeheader();w.writerows(sensitivity_rows)
summary['posthoc_prefix_sensitivity']=sensitivity
summary['posthoc_prefix_scope']='Paired raw channels on 512 preselected windows only; no corresponding raw-control comparison is available.'

# One complete-sample figure; thresholds 10 and 100 both shown, avoiding only favorable tails.
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
fig,axes=plt.subplots(1,3,figsize=(14.8,4.7),gridspec_kw={'width_ratios':[1.2,1,1]})
ordered=sorted(activation,key=lambda r:float(r['paired_correlation']));x=np.arange(1,289)
axes[0].scatter(x,[float(r['paired_correlation']) for r in ordered],s=12,color='#245B88',label='Same-index pair',zorder=3)
axes[0].plot(x,[float(r['control_mean']) for r in ordered],lw=.8,color='#B65E33',label='Mean of 100 controls')
axes[0].axhline(0,color='#888',lw=.6);axes[0].set(xlabel='All 288 pairs, sorted by paired correlation',ylabel='Signed token-response correlation',title='A  Final neuron activation')
axes[0].legend(frameon=False,fontsize=8,loc='upper left')
axes[0].text(.02,.70,'185/288 partners rank first\nMedian paired r = 0.056',transform=axes[0].transAxes,fontsize=9)
rng=np.random.default_rng(20261027)
for ax,k,letter in [(axes[1],10,'B'),(axes[2],100,'C')]:
    data=[]
    for group in ['positive_all_regimes','negative_all_regimes']:
        valid=[r for r in o if r['response_group']==group and int(r['top_k'])==k and r['shared_passages']!='' and r['control_shared_mean']!='']
        data += [np.array([float(r['shared_passages']) for r in valid]),np.array([float(r['control_shared_mean']) for r in valid])]
    for j,v in enumerate(data):
        color='#245B88' if j%2==0 else '#B65E33';ax.scatter(j+rng.uniform(-.14,.14,len(v)),v,s=9,alpha=.35,color=color)
        ax.plot([j-.2,j+.2],[np.median(v)]*2,color='black',lw=1.7)
        ax.scatter([j],[v.mean()],marker='D',s=30,color=color,edgecolor='white',linewidth=.5,zorder=4)
    ax.set_xticks(range(4),['Pair\npositive','Control\npositive','Pair\nnegative','Control\nnegative'],fontsize=8)
    ax.set(ylabel=f'Shared windows in top-{k} lists',title=f'{letter}  Strongest {k} windows')
    ax.grid(axis='y',alpha=.15)
fig.suptitle('Qwen3-32B Q8: same-index neuron pairs 24 layers apart',fontsize=15,y=.99)
fig.text(.02,.015,'10,000 fixed 256-token WikiText windows. Descriptive sampled comparisons; controls shared within layer pair.\nB/C: dots = usable pairs, black bars = medians, diamonds = means; tied cutoffs/insufficient signed responses excluded with counts in READOUT.txt.',fontsize=8,color='#444')
fig.tight_layout(rect=[0,.105,1,.94]);fig.savefig(HERE/'neuron_pairing.png',dpi=200);fig.savefig(HERE/'neuron_pairing.svg');plt.close(fig)
for name in ['inputs.json','layout.json','passages.jsonl','tokens.npy','checkpoint.npz','reports/correlations.csv','reports/passage_overlap.csv','code_identity.json']:
    hashes[str((RUN/name).relative_to(ROOT))]=sha(RUN/name)
save('summary.json',summary)
save('validation.json',{'status':'passed','frozen_dense_hashes_checked':4,'correlation_rows_recomputed':864,'correlation_ranks_recomputed':864,
    'max_correlation_summary_error':max_error,'overlap_rows_recomputed':overlap_checked,
    'max_control_overlap_mean_error':overlap_max_error,'input_sha256':hashes,'new_inference':False})
print(json.dumps(summary,indent=2))
