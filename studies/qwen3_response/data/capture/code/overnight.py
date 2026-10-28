#!/usr/bin/env python3
"""Unattended local Q8 data collection. No predictions or interpretation."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import traceback

import numpy as np
from core import atomic_json,atomic_npz,digest,open_stores,Moments,passage_arrays,commit
from backend import BASE,MODEL,MODEL_SHA,log,check_model,Capture
from prepare import prepare_run
from report import make_report

HERE=Path(__file__).resolve().parent
DEFAULT_RUN=BASE/'runs/overnight_20261026_q8'
IMPLEMENTATION=['core.py','backend.py','prepare.py','report.py','overnight.py']

def settings(a):
    if not 1<=a.tokens<=4096:raise ValueError('--tokens must be 1..4096')
    if not 1<=a.controls<=1000:raise ValueError('--controls must be 1..1000')
    if a.passages<1 or a.raw_passages<0:raise ValueError('Invalid passage count')
    if a.checkpoint_every<1 or a.max_hours<=0 or a.reserve_gb<=0:raise ValueError('Invalid runtime limit')
    return {'passages':a.passages,'tokens':a.tokens,'controls':a.controls,
            'raw_passages':min(a.raw_passages,a.passages),'text_seed':20261027,
            'control_seed':20261028,'raw_seed':20261029}

def code_identity(run):
    hashes={n:digest(HERE/n) for n in IMPLEMENTATION}
    path=run/'code_identity.json'
    if path.exists():
        if json.loads(path.read_text())['sha256']!=hashes:
            raise ValueError('Capture code changed since this run began; keep the old run and use a new --run-dir')
    else:
        dest=run/'code';dest.mkdir(exist_ok=True)
        for n in IMPLEMENTATION:shutil.copy2(HERE/n,dest/n)
        atomic_json(path,{'sha256':hashes,'python':sys.version,'executable':sys.executable})

def status(run):
    p=run/'status.json'
    d=json.loads(p.read_text()) if p.exists() else {'phase':'prepared' if (run/'inputs.json').exists() else 'not_started','run_dir':str(run)}
    if (run/'inputs.json').exists():
        d.setdefault('planned_passages',json.loads((run/'inputs.json').read_text())['settings']['passages'])
    if (run/'checkpoint.npz').exists():
        with np.load(run/'checkpoint.npz') as z:
            d.update(committed_passages=int(z['completed']),token_observations_per_neuron=int(z['tokens_seen']))
    d['free_disk_GB']=round(shutil.disk_usage(run if run.exists() else BASE).free/1e9,2)
    print(json.dumps(d,indent=2))

def run_capture(a,run):
    cfg=settings(a)
    # Preparing in its own process uses only the vocabulary and releases it before GPU load.
    if not (run/'inputs.json').exists():
        command=[sys.executable,str(__file__),'prepare','--run-dir',str(run),
                 '--passages',str(a.passages),'--tokens',str(a.tokens),
                 '--controls',str(a.controls),'--raw-passages',str(a.raw_passages)]
        subprocess.run(command,check=True)
    with (run/'run.lock').open('a+') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise RuntimeError('This run is already active; use status or stop')
        inputs=prepare_run(run,cfg);code_identity(run)
        lay=json.loads((run/'layout.json').read_text())
        moments=Moments(lay['pairs']);completed=raw_written=0
        checkpoint=run/'checkpoint.npz'
        if checkpoint.exists():
            with np.load(checkpoint) as z:
                completed=int(z['completed']);raw_written=int(z['raw_written']);moments.restore(z)
            if moments.n!=completed*a.tokens:raise ValueError('Checkpoint token count mismatch')
        if completed==a.passages:
            log('All planned passages already committed; refreshing descriptive reports')
            make_report(run)
            from verify import verify_run
            verify_run(run)
            return
        used=sum(p.stat().st_blocks*512 for p in run.rglob('*') if p.is_file())
        remaining=max(0,inputs['estimated_maximum_output_bytes']-used)
        free=shutil.disk_usage(run).free
        if free-remaining<a.reserve_gb*1e9:
            raise RuntimeError(f'Output budget needs {remaining/1e9:.2f} GB more plus {a.reserve_gb:g} GB reserve; only {free/1e9:.2f} GB free')
        def record(phase,**extra):
            atomic_json(run/'status.json',{'phase':phase,'pid':os.getpid(),'run_dir':str(run),
                'updated_unix':time.time(),'planned_passages':a.passages,'committed_passages':completed,
                'tokens_per_passage':a.tokens,'free_disk_GB':round(shutil.disk_usage(run).free/1e9,2),**extra})
        stopping={'value':False}
        def request_stop(signum,frame):stopping['value']=True
        signal.signal(signal.SIGINT,request_stop);signal.signal(signal.SIGTERM,request_stop)
        stop=run/'STOP_REQUESTED'
        atomic_json(run/f'invocation_{time.time_ns()}.json',{'pid':os.getpid(),'start_completed':completed,
            'max_capture_hours':a.max_hours,'checkpoint_every':a.checkpoint_every,
            'reserve_gb':a.reserve_gb,'stop_after':a.stop_after})
        record('checking_model');check_model()
        new=not checkpoint.exists()
        # A first process can die after allocating files but before its first commit.
        # Those files contain zero committed passages and are safely reused by shape.
        if new and all((run/(n+'.npy')).exists() for n in ['peaks','positions','sign_counts','passage_moments']):new=False
        stores=open_stores(run,a.passages,lay['total_channels'],new)
        if not checkpoint.exists():commit(run,stores,moments,0,0)
        tokens=np.load(run/'tokens.npy',mmap_mode='r')
        raw_set=set(json.loads((run/'raw_indices.json').read_text()))
        raw_columns=[]
        for layer,loc in lay['layers'].items():
            paired=next(p['neurons'] for p in lay['pairs'] if int(layer) in [p['source'],p['target']])
            raw_columns.extend(loc['start']+loc['ids'].index(i) for i in paired)
        atomic_json(run/'raw_columns.json',[{'layer':int(layer),'neuron':i,'flat_column':loc['start']+loc['ids'].index(i)}
            for layer,loc in lay['layers'].items()
            for i in next(p['neurons'] for p in lay['pairs'] if int(layer) in [p['source'],p['target']])])
        raw_dir=run/'raw';raw_dir.mkdir(exist_ok=True)
        engine=None;start=time.monotonic();deadline=start+a.max_hours*3600
        start_completed=completed;last_committed=completed;reason='complete'
        try:
            record('loading_model');engine=Capture(lay,a.tokens)
            record('checking_instrumentation');engine.validate(tokens[0],run)
            record('capturing')
            log(f'Capturing from passage {completed}/{a.passages}; {a.max_hours:g} hour limit; {a.reserve_gb:g} GB disk reserve')
            while completed<a.passages:
                if stopping['value'] or stop.exists():reason='requested_stop';break
                if time.monotonic()>=deadline:reason='time_limit';break
                if a.stop_after and completed-start_completed>=a.stop_after:reason='requested_batch_limit';break
                if shutil.disk_usage(run).free<a.reserve_gb*1e9:reason='disk_reserve';break
                values,_=engine.evaluate(tokens[completed+1])
                peaks,positions,counts,local=passage_arrays(values)
                stores['peaks'][completed]=peaks;stores['positions'][completed]=positions
                stores['sign_counts'][completed]=counts;stores['passage_moments'][completed]=local
                moments.update(values,lay)
                if completed in raw_set:
                    atomic_npz(raw_dir/f'p{completed:05d}.npz',ids=np.array(tokens[completed+1]),
                               values=values[:,:,raw_columns],passage_index=np.array(completed))
                    raw_written+=1
                completed+=1
                if completed%a.checkpoint_every==0 or completed==a.passages:
                    commit(run,stores,moments,completed,raw_written);last_committed=completed
                    elapsed=time.monotonic()-start;rate=(completed-start_completed)/elapsed
                    log(f'{completed}/{a.passages} passages committed; {elapsed/60:.1f} min; {shutil.disk_usage(run).free/1e9:.2f} GB free')
                    record('capturing',raw_passages_written=raw_written,
                           estimated_remaining_capture_hours=(a.passages-completed)/rate/3600 if rate else None)
            if completed!=last_committed:commit(run,stores,moments,completed,raw_written)
        except BaseException as e:
            # Never commit moments if a passage failed partway through an update.
            record('failed',error=repr(e),committed_passages=last_committed,last_safe_checkpoint=last_committed,
                   uncommitted_work_will_be_recomputed=True)
            raise
        finally:
            if engine is not None:engine.close()
        try:
            record('writing_measurements',stop_reason=reason)
            summary=make_report(run)
            record('verifying_saved_data',stop_reason=reason)
            from verify import verify_run
            validation=verify_run(run)
        except BaseException as e:
            record('postprocessing_failed',error=repr(e),stop_reason=reason,
                   captures_preserved=True)
            raise
        phase='complete' if completed==a.passages else 'paused'
        if reason=='requested_stop' and stop.exists():
            # Consume a fulfilled stop request here, never at startup: a stop
            # issued just after --detach must not be erased by initialization.
            stop.unlink()
        record(phase,stop_reason=reason,raw_passages_written=raw_written,
               reports=str(run/'reports/READOUT.txt'),elapsed_seconds=time.monotonic()-start)
        atomic_json(run/'completion.json',{'phase':phase,'stop_reason':reason,'summary':summary,
            'validation_status':validation['status'],
            'files_to_analyze':['checkpoint.npz','peaks.npy','positions.npy','sign_counts.npy','passage_moments.npy','raw/','passages.jsonl','tokens.npy','reports/'],
            'no_predictions':True,'no_interpretation':True})
        log(phase.upper(),f'{completed}/{a.passages} passages;',run/'reports/READOUT.txt')

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('action',nargs='?',default='run',choices=['run','prepare','status','stop','report','verify'])
    ap.add_argument('--run-dir',type=Path,default=DEFAULT_RUN)
    ap.add_argument('--passages',type=int,default=10000);ap.add_argument('--tokens',type=int,default=256)
    ap.add_argument('--controls',type=int,default=100);ap.add_argument('--raw-passages',type=int,default=512)
    ap.add_argument('--max-hours',type=float,default=8);ap.add_argument('--reserve-gb',type=float,default=4)
    ap.add_argument('--checkpoint-every',type=int,default=16)
    ap.add_argument('--stop-after',type=int,default=0,help='Optional count for a single invocation; mainly validation')
    ap.add_argument('--detach',action='store_true',help='Run in the background, with output in run.log')
    a=ap.parse_args();run=a.run_dir.expanduser().resolve()
    if a.action=='status':status(run);return
    if a.action=='stop':
        run.mkdir(parents=True,exist_ok=True);(run/'STOP_REQUESTED').write_text('Stop after the current passage and checkpoint.\n')
        print('Stop requested; data will be checkpointed.');return
    if a.detach:
        if a.action!='run':raise ValueError('--detach applies to run only')
        run.mkdir(parents=True,exist_ok=True)
        command=[sys.executable,str(__file__),*[x for x in sys.argv[1:] if x!='--detach']]
        with (run/'run.log').open('a') as f:
            child=subprocess.Popen(['/usr/bin/caffeinate','-i',*command],stdout=f,stderr=subprocess.STDOUT,
                                   stdin=subprocess.DEVNULL,start_new_session=True)
        print(f'Background runner PID {child.pid}\nRun: {run}\nLog: {run/"run.log"}\nUse the same command with status or stop.');return
    if a.action=='run':run_capture(a,run);return
    if a.action=='verify':
        from verify import verify_run
        with (run/'run.lock').open('a+') as lock:
            try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:raise RuntimeError('Run is active; use status or stop before verification')
            print(json.dumps(verify_run(run),indent=2));return
    run.mkdir(parents=True,exist_ok=True)
    with (run/'run.lock').open('a+') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise RuntimeError('Run is active; use status or stop first')
        if a.action=='prepare':prepare_run(run,settings(a))
        else:print(json.dumps(make_report(run),indent=2))

if __name__=='__main__':
    try:main()
    except Exception as e:
        print(f'ERROR: {e}',file=sys.stderr,flush=True)
        traceback.print_exc()
        raise SystemExit(1)
