"""Freeze text and neuron choices before the unattended activation capture."""
from pathlib import Path
import json
import random
import subprocess
import numpy as np
import pyarrow.parquet as pq

from core import atomic_json,digest,layout,bytes_required,SCHEMA
from backend import BASE,vocabulary,log,MODEL_SHA,MODEL_REV

HERE=Path(__file__).resolve().parent
DATA_REV='b08601e04326c79dfdd32d625aee71d232d685c3'
DATA_FILE='wikitext-103-raw-v1/train-00000-of-00002.parquet'
DATA_SHA='74da360f23826045b3e6ac6375411fdb15f003030aa74f2596ed08b857cb9212'

def source():
    path=HERE/'assets'/DATA_FILE
    if not path.exists():
        log('Downloading pinned WikiText-103 text shard (157 MB)')
        subprocess.run(['hf','download','Salesforce/wikitext',DATA_FILE,'--repo-type','dataset',
            '--revision',DATA_REV,'--local-dir',str(HERE/'assets')],check=True)
    if digest(path)!=DATA_SHA:raise ValueError('Text source checksum mismatch')
    return path

def articles(path):
    result=[];title=None;parts=[];chars=0;start=0;row=0
    def save(end):
        if title and chars>=1500:
            result.append({'article':title,'row_start':start,'row_end':end,'prefix':'\n\n'.join(parts)})
    for batch in pq.ParquetFile(path).iter_batches(columns=['text']):
        for text in batch.column(0).to_pylist():
            t=text.strip()
            if t.startswith('= ') and t.endswith(' =') and not t.startswith('= ='):
                save(row-1);title=t.strip('= ').strip();parts=[];chars=0;start=row
            elif t and not t.startswith('='):
                if chars<12000:parts.append(t[:max(0,12000-chars)])
                chars+=len(t)
            row+=1
    save(row-1)
    return result

def prepare_run(run,settings):
    run=Path(run);run.mkdir(parents=True,exist_ok=True)
    manifest=run/'inputs.json'
    if manifest.exists():
        old=json.loads(manifest.read_text())
        if old['settings']!=settings:raise ValueError('Run settings differ from the existing frozen run; use a new --run-dir')
        for name,h in old['input_sha256'].items():
            if digest(run/name)!=h:raise ValueError(f'Frozen input changed: {name}')
        return old
    path=source();candidates=articles(path)
    smoke=BASE/'runs/smoke_20261026_local_q8/source_passages.json'
    excluded={p['article'].casefold() for p in json.loads(smoke.read_text())} if smoke.exists() else set()
    seen=set();eligible=[]
    for a in candidates:
        key=a['article'].casefold()
        if key not in seen and key not in excluded:eligible.append(a);seen.add(key)
    random.Random(settings['text_seed']).shuffle(eligible)
    if len(eligible)<settings['passages']+1:raise ValueError(f'Only {len(eligible)} eligible distinct articles in the fixed shard')
    selection=json.loads((BASE/'selection.json').read_text())
    lay=layout(selection,settings['controls'],settings['control_seed'])
    atomic_json(run/'layout.json',lay)
    count=settings['passages']+1;length=settings['tokens']
    token_path=run/'tokens.npy';tmp=run/'tokens.tmp.npy'
    ids=np.lib.format.open_memmap(tmp,mode='w+',dtype=np.int32,shape=(count,length))
    vocab=vocabulary();used=0
    text_tmp=run/'passages.jsonl.tmp'
    try:
        with text_tmp.open('w') as out:
            for a in eligible:
                tokens=vocab.tokenize(a['prefix'].encode(),add_bos=False,special=False)
                if len(tokens)<length:continue
                tokens=tokens[:length];ids[used]=tokens
                record={k:a[k] for k in ['article','row_start','row_end']}
                record.update(id='instrumentation' if used==0 else f'p{used-1:05d}',
                              text=vocab.detokenize(tokens).decode('utf-8',errors='replace'))
                out.write(json.dumps(record,ensure_ascii=False)+'\n');used+=1
                if used%1000==0:log('Prepared passages',used-1,'/',settings['passages'])
                if used==count:break
            out.flush()
        if used!=count:raise ValueError(f'Insufficient full-length passages: {used-1}')
        ids.flush();del ids;tmp.replace(token_path);text_tmp.replace(run/'passages.jsonl')
    finally:vocab.close()
    raw_indices=sorted(random.Random(settings['raw_seed']).sample(range(settings['passages']),min(settings['raw_passages'],settings['passages'])))
    atomic_json(run/'raw_indices.json',raw_indices)
    record={'schema':SCHEMA,'settings':settings,'model_repo':'Qwen/Qwen3-32B-GGUF',
        'model_revision':MODEL_REV,'model_sha256':MODEL_SHA,
        'q8_scope':'Q8 activations; no byte identity to earlier BF16 weights is asserted',
        'dataset_repo':'Salesforce/wikitext','dataset_revision':DATA_REV,'dataset_file':DATA_FILE,
        'dataset_sha256':DATA_SHA,'dataset_local_path':str(path),'dataset_license':'CC-BY-SA-3.0',
        'text_selection':'Random distinct articles; fixed first token window; no activation-based selection',
        'excluded_smoke_articles':len(excluded),'excluded_smoke_input_sha256':digest(smoke) if smoke.exists() else None,
        'selection_input_sha256':digest(BASE/'selection.json'),
        'raw_retention':'Raw float32 gate/value/product at the 576 paired channels for a preselected random passage subset',
        'estimated_maximum_output_bytes':bytes_required(settings['passages'],settings['tokens'],settings['raw_passages'],lay['total_channels']),
        'input_sha256':{n:digest(run/n) for n in ['layout.json','tokens.npy','passages.jsonl','raw_indices.json']},
        'no_predictions':True,'no_significance_tests':True,'no_interpretation':True}
    atomic_json(manifest,record)
    log('Inputs frozen:',settings['passages'],'passages,',lay['total_channels'],'captured channels')
    return record
