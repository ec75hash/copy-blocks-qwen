"""Check committed capture structure and raw-to-summary consistency."""
from pathlib import Path
import json
import numpy as np
from core import digest,atomic_json,Moments,passage_arrays

def verify_run(run):
    run=Path(run);inputs=json.loads((run/'inputs.json').read_text());lay=json.loads((run/'layout.json').read_text())
    for name,h in inputs['input_sha256'].items():
        if digest(run/name)!=h:raise ValueError(f'Frozen input mismatch: {name}')
    code=json.loads((run/'code_identity.json').read_text())
    for name,h in code['sha256'].items():
        if digest(run/'code'/name)!=h:raise ValueError(f'As-run code mismatch: {name}')
    moments=Moments(lay['pairs'])
    with np.load(run/'checkpoint.npz') as z:
        n=int(z['completed']);raw_count=int(z['raw_written']);moments.restore(z)
    cfg=inputs['settings'];t=cfg['tokens']
    if moments.n!=n*t:raise ValueError('Token count mismatch')
    stores={k:np.load(run/(k+'.npy'),mmap_mode='r') for k in ['peaks','positions','sign_counts','passage_moments']}
    for start in range(0,n,128):
        end=min(n,start+128)
        peak=np.asarray(stores['peaks'][start:end]);pos=np.asarray(stores['positions'][start:end])
        count=np.asarray(stores['sign_counts'][start:end]);local=np.asarray(stores['passage_moments'][start:end])
        exists=pos!=65535
        if not np.all((pos[exists]>=0)&(pos[exists]<t)):raise ValueError('Invalid peak token index')
        if not np.array_equal(exists,np.isfinite(peak).all(axis=2)):raise ValueError('Peak presence mismatch')
        if not np.array_equal(exists,count>0):raise ValueError('Sign counts disagree with peak presence')
        if not np.all(count.sum(axis=1)<=t):raise ValueError('Sign regime counts exceed passage length')
        if not np.isfinite(local).all():raise ValueError('Nonfinite passage moments')
        for r in range(4):
            valid=exists[:,r,:]
            g=peak[:,r,0,:][valid];v=peak[:,r,1,:][valid];a=peak[:,r,2,:][valid]
            conditions=[(g>=0)&(v>=0),(g>=0)&(v<0),(g<0)&(v>=0),(g<0)&(v<0)]
            if not conditions[r].all():raise ValueError('Peak has the wrong sign regime')
            reconstructed=g.astype(np.float64)/(1+np.exp(-np.clip(g.astype(np.float64),-80,80)))*v
            if not np.allclose(reconstructed,a,rtol=2e-4,atol=2e-5):raise ValueError('Peak product does not reconstruct')
    columns=json.loads((run/'raw_columns.json').read_text());idx=[x['flat_column'] for x in columns]
    raw_indices=[i for i in json.loads((run/'raw_indices.json').read_text()) if i<n]
    if len(raw_indices)!=raw_count:raise ValueError('Raw retention count differs from checkpoint')
    chosen=raw_indices if len(raw_indices)<=32 else [raw_indices[i] for i in np.linspace(0,len(raw_indices)-1,32,dtype=int)]
    tokens=np.load(run/'tokens.npy',mmap_mode='r')
    for i in chosen:
        with np.load(run/'raw'/f'p{i:05d}.npz') as z:
            if not np.array_equal(z['ids'],tokens[i+1]):raise ValueError('Raw token IDs differ')
            peaks,pos,count,local=passage_arrays(z['values'])
        np.testing.assert_array_equal(peaks,np.asarray(stores['peaks'][i])[:,:,idx])
        np.testing.assert_array_equal(pos,np.asarray(stores['positions'][i])[:,idx])
        np.testing.assert_array_equal(count,np.asarray(stores['sign_counts'][i])[:,idx])
        np.testing.assert_allclose(local,np.asarray(stores['passage_moments'][i])[:,:,idx],rtol=2e-6,atol=1e-7)
    # Independent aggregate check from passage mean/RMS, stored in float32.
    largest_mean_error=0.;largest_variance_error=0.
    for pi,p in enumerate(lay['pairs']):
        for side,layer in [('x',p['source']),('y',p['target'])]:
            loc=lay['layers'][str(layer)]
            mu=np.asarray(stores['passage_moments'][:n,0,:,loc['start']:loc['end']],dtype=np.float64)
            rms=np.asarray(stores['passage_moments'][:n,1,:,loc['start']:loc['end']],dtype=np.float64)
            if n:
                mean=mu.mean(0);m2=(np.mean(rms*rms,axis=0)-mean*mean)*moments.n
                original_mean=getattr(moments,'mean_'+side)[pi];original_m2=getattr(moments,'m2_'+side)[pi]
                np.testing.assert_allclose(mean,original_mean,rtol=3e-5,atol=2e-7)
                np.testing.assert_allclose(m2,original_m2,rtol=2e-4,atol=1e-5)
                largest_mean_error=max(largest_mean_error,float(np.max(np.abs(mean-original_mean))))
                largest_variance_error=max(largest_variance_error,float(np.max(np.abs(m2-original_m2))))
    record={'status':'passed','committed_passages':n,'token_observations_per_neuron':moments.n,
        'retained_raw_passages':raw_count,'raw_passages_independently_reduced':len(chosen),
        'all_committed_peaks_signs_positions_and_counts_checked':True,
        'max_global_mean_difference_from_float32_passage_summaries':largest_mean_error,
        'max_global_m2_difference_from_float32_passage_summaries':largest_variance_error,
        'checkpoint_sha256':digest(run/'checkpoint.npz'),
        'dense_array_sha256':{k:digest(run/(k+'.npy')) for k in stores},
        'no_interpretation':True}
    atomic_json(run/'VALIDATION.json',record)
    return record
