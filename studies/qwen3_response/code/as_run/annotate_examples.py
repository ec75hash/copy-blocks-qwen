"""Attach exact saved-token context to numerically selected examples; no inference."""
from pathlib import Path
import csv,json,sys
import numpy as np

HERE=Path(__file__).resolve().parent;RUN=HERE.parent/'runs/overnight_20261026_q8'
sys.path.insert(0,str(HERE.parent/'overnight'))
from backend import vocabulary

cases=json.loads((HERE/'example_selection.json').read_text());lay=json.loads((RUN/'layout.json').read_text())
texts=[json.loads(x) for x in (RUN/'passages.jsonl').read_text().splitlines()][1:]
ids=np.load(RUN/'tokens.npy',mmap_mode='r');peaks=np.load(RUN/'peaks.npy',mmap_mode='r');pos=np.load(RUN/'positions.npy',mmap_mode='r')
vocab=vocabulary()
def event(layer,neuron,index,pole):
    loc=lay['layers'][str(layer)];col=loc['start']+loc['ids'].index(neuron);reg=[0,3] if pole=='positive' else [1,2]
    vals=np.asarray(peaks[index,reg,2,col]);rr=reg[int(np.nanargmax(vals if pole=='positive' else -vals))]
    t=int(pos[index,rr,col]);tokens=ids[index+1].tolist()
    before=vocab.detokenize(tokens[:t]).decode('utf-8',errors='replace')
    token=vocab.detokenize(tokens[t:t+1]).decode('utf-8',errors='replace')
    after=vocab.detokenize(tokens[t+1:]).decode('utf-8',errors='replace')
    return {'layer':layer,'neuron':neuron,'passage_index':index,'passage_id':f'p{index:05d}',
        'parsed_source_label':texts[index]['article'],'source_rows':[texts[index]['row_start'],texts[index]['row_end']],
        'token_position':t,'token_id':tokens[t],'token_text':token,
        'context':before[-150:]+'[['+token+']]'+after[:210],
        'gate_pre_silu':float(peaks[index,rr,0,col]),'value':float(peaks[index,rr,1,col]),
        'activation':float(peaks[index,rr,2,col]),'sign_regime':rr,'full_window':texts[index]['text']}
try:
    for case in cases:
        L=case['source_layer'];n=case['neuron'];pole=case['display_pole'];rankings=[]
        for member in case['members']:
            old=member['events'];member['events']=[dict(event(member['layer'],n,e['passage_index'],pole),strength_tied_windows=e['strength_tied_windows']) for e in old]
            loc=lay['layers'][str(member['layer'])];col=loc['start']+loc['ids'].index(n);reg=[0,3] if pole=='positive' else [1,2]
            vals=np.asarray(peaks[:,reg,2,col]);score=np.max(np.where(np.isfinite(vals),vals if pole=='positive' else -vals,-np.inf),axis=1)
            order=np.argsort(-score,kind='stable');rankings.append(order)
        common=set(rankings[0][:100])&set(rankings[1][:100])
        ranks=[{int(v):i for i,v in enumerate(a)} for a in rankings]
        chosen=sorted(common,key=lambda i:(ranks[0][int(i)]+ranks[1][int(i)],int(i)))[:2]
        case['shared_window_examples']=[{'selection':'lowest sum of source/partner ranks within their shared top-100 windows',
            'source':event(L,n,int(i),pole),'partner':event(L+24,n,int(i),pole)} for i in chosen]
finally:vocab.close()
(HERE/'examples.json').write_text(json.dumps(cases,indent=2,ensure_ascii=False)+'\n')
lines=['NUMERICALLY SELECTED NEURON EXAMPLES — QWEN3-32B Q8','Exploratory examples selected after collection; no neuron-meaning labels are assigned.',
       'Selection includes both high-overlap cases, a near-median case, and the lowest signed correlation.',
       '[[...]] marks the saved token at the measured peak; token indices are zero-based.',
       'Source labels reproduce the original parser labels; some are equation fragments, not article titles.','']
for c in cases:
    r=c['correlation'];lines += [f"SELECTION: {c['selection_rule']}",f"L{c['source_layer']} / L{c['target_layer']}, neuron {c['neuron']}",
        f"Full-token correlation {float(r['paired_correlation']):.6f}; mean control {float(r['control_mean']):.6f}; rank {r['paired_rank']}/101."]
    for pole in ['positive','negative']:
        o=c['overlap'][pole];lines.append(f"Top-100 {pole}: {o['shared_passages']} shared; control mean {float(o['control_shared_mean']):.3f}; usable controls {o['usable_controls']}.")
    for m in c['members']:
        lines += ['',f"L{m['layer']} — strongest {c['display_pole']} windows (ties ordered by stored passage index):"]
        for e in m['events'][:3]:
            lines += [f"{e['passage_id']} | label: {e['parsed_source_label']} | token {e['token_position']} ({e['token_text']!r})",
                f"gate={e['gate_pre_silu']:.5f}, value={e['value']:.5f}, activation={e['activation']:.5f}, regime={e['sign_regime']}; equally strong windows={e['strength_tied_windows']}",e['context']]
    for shared in c['shared_window_examples']:
        lines+=['','Same-window comparison:']
        for side in ['source','partner']:
            e=shared[side];lines += [f"L{e['layer']} {e['passage_id']} token {e['token_position']} ({e['token_text']!r}), activation={e['activation']:.5f}",e['context']]
    lines+=['','-'*72,'']
(HERE/'PASSAGE_EXAMPLES.txt').write_text('\n'.join(lines)+'\n')
for c in cases:
    print(c['selection_rule'],c['source_layer'],c['neuron'])
    for m in c['members']:
        print(' L',m['layer'])
        for e in m['events'][:3]:print(' ',e['passage_id'],repr(e['token_text']),e['context'][:310])
