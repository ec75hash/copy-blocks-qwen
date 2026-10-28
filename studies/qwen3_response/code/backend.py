"""The locally validated llama.cpp Metal tap, generalized to 48 layers."""
from pathlib import Path
import ctypes
import os
import json
import time
import numpy as np

from core import atomic_json,digest

BASE=Path(__file__).resolve().parents[1]
MODEL=Path(os.environ.get('QWEN_Q8_MODEL',str(BASE/'models/Qwen3-32B-Q8_0.gguf')))
MODEL_SHA='2c50eb8aad05047dbf24fa014eb621adf552e14176cabe0c5db4ef38c91e2169'
MODEL_REV='938a7432affaec9157f883a87164e2646ae17555'

def log(*args):print(time.strftime('%H:%M:%S'),*args,flush=True)

def check_model():
    if not MODEL.exists():raise FileNotFoundError(f'Existing local Q8 model required: {MODEL}')
    log('Verifying local model SHA-256')
    if digest(MODEL)!=MODEL_SHA:raise ValueError('Model hash differs from the verified official Q8 file')

def vocabulary():
    import llama_cpp
    from llama_cpp._internals import LlamaModel
    if llama_cpp.__version__!='0.3.36':raise RuntimeError('This runner was validated with llama-cpp-python 0.3.36')
    llama_cpp.llama_backend_init()
    params=llama_cpp.llama_model_default_params();params.vocab_only=True;params.n_gpu_layers=0
    return LlamaModel(path_model=str(MODEL),params=params,verbose=False)

class Capture:
    def __init__(self,lay,tokens):
        import llama_cpp
        import llama_cpp.llama_cpp as low
        import gguf
        if llama_cpp.__version__!='0.3.36':raise RuntimeError('Validated llama-cpp-python 0.3.36 required')
        self.lay=lay;self.llama_cpp=llama_cpp;self.on=False;self.errors=[];self.buffer=None
        self.nodes={};self.offsets={}
        self.channels={int(k):np.array(v['ids']) for k,v in lay['layers'].items()}
        for layer in self.channels:
            for c,name in enumerate(['ffn_gate','ffn_up','ffn_swiglu']):
                self.nodes[f'{name}-{layer}']=(layer,c)
        reader=gguf.GGUFReader(str(MODEL))
        for k,v in [('qwen3.block_count',64),('qwen3.embedding_length',5120),('qwen3.feed_forward_length',25600)]:
            if int(reader.fields[k].contents())!=v:raise ValueError(f'Wrong model metadata: {k}')
        del reader
        self.lib=ctypes.CDLL(str(Path(llama_cpp.__file__).parent/'lib/libggml-base.dylib'))
        self.lib.ggml_get_name.restype=ctypes.c_char_p;self.lib.ggml_get_name.argtypes=[ctypes.c_void_p]
        self.lib.ggml_nelements.restype=ctypes.c_int64;self.lib.ggml_nelements.argtypes=[ctypes.c_void_p]
        self.lib.ggml_nbytes.restype=ctypes.c_size_t;self.lib.ggml_nbytes.argtypes=[ctypes.c_void_p]
        self.lib.ggml_backend_tensor_get.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_size_t,ctypes.c_size_t]
        self.lib.ggml_backend_tensor_get.restype=None
        self.cb=llama_cpp.ggml_backend_sched_eval_callback(self.hook)
        original=low.llama_context_default_params
        def patched():
            p=original();p.cb_eval=self.cb;p.cb_eval_user_data=None;return p
        low.llama_context_default_params=patched;llama_cpp.llama_context_default_params=patched
        try:
            log('Loading existing Q8 model on Metal')
            self.model=llama_cpp.Llama(model_path=str(MODEL),n_ctx=max(256,tokens),n_batch=tokens,
                n_ubatch=min(tokens,128),n_gpu_layers=99,flash_attn=True,use_mmap=True,
                logits_all=False,verbose=False)
        finally:
            low.llama_context_default_params=original;llama_cpp.llama_context_default_params=original
    def hook(self,t,ask,user):
        try:
            name=self.lib.ggml_get_name(t).decode()
            if ask:return self.on and name in self.nodes
            if not self.on or name not in self.nodes:return True
            n=int(self.lib.ggml_nelements(t));nb=int(self.lib.ggml_nbytes(t))
            if nb!=n*4 or n%25600:raise ValueError(f'Unexpected tensor layout {name}: {n}/{nb}')
            raw=np.empty(n,np.float32)
            self.lib.ggml_backend_tensor_get(t,ctypes.c_void_p(raw.ctypes.data),0,nb)
            layer,c=self.nodes[name];data=raw.reshape(-1,25600)[:,self.channels[layer]]
            i=self.offsets[name];j=i+len(data);loc=self.lay['layers'][str(layer)]
            if j>len(self.buffer):raise ValueError(f'Too many token rows for {name}')
            self.buffer[i:j,c,loc['start']:loc['end']]=data;self.offsets[name]=j
            return True
        except BaseException as e:
            self.errors.append(repr(e));return False
    def evaluate(self,ids,tapped=True):
        self.errors=[];self.offsets={name:0 for name in self.nodes}
        self.buffer=np.empty((len(ids),3,self.lay['total_channels']),np.float32) if tapped else None
        self.on=tapped
        try:
            self.model.reset();self.model.eval([int(t) for t in ids])
        finally:self.on=False
        if self.errors:raise RuntimeError(self.errors)
        if tapped:
            if any(n!=len(ids) for n in self.offsets.values()):
                raise ValueError({k:v for k,v in self.offsets.items() if v!=len(ids)})
            if not np.isfinite(self.buffer).all():raise ValueError('Nonfinite captured activation')
        return self.buffer,np.array(self.model.scores[-1],copy=True)
    def validate(self,ids,run):
        _,baseline=self.evaluate(ids,False)
        values,logits=self.evaluate(ids,True)
        if not np.allclose(baseline,logits,rtol=1e-5,atol=1e-4):raise ValueError('Capture changed model logits')
        g=values[:,0].astype(np.float64);v=values[:,1].astype(np.float64)
        reconstructed=g/(1+np.exp(-np.clip(g,-80,80)))*v
        if not np.allclose(reconstructed,values[:,2],rtol=2e-4,atol=2e-5):raise ValueError('Captured channel does not reconstruct from gate/value')
        record={'tokens':len(ids),'layers_checked':len(self.channels),'captured_channels':self.lay['total_channels'],
                'logit_max_absolute_difference':float(np.abs(baseline-logits).max()),
                'product_max_absolute_error':float(np.abs(reconstructed-values[:,2]).max()),
                'all_capture_nodes_have_exact_token_count':True,
                'library_sha256':{n:digest(Path(self.llama_cpp.__file__).parent/'lib'/n)
                    for n in ['libllama.dylib','libggml-base.dylib','libggml-metal.dylib']}}
        atomic_json(Path(run)/'instrumentation.json',record)
        return record
    def close(self):self.model.close()
