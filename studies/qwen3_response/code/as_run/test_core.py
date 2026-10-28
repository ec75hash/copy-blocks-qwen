import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from core import Moments,layout,passage_arrays,top_indices,open_stores,commit,bytes_required

class CoreTests(unittest.TestCase):
    def small(self):
        pair={'source':8,'target':32,'neurons':[0,1],'controls':[2],
              'target_columns':[0,1],'control_columns':[2]}
        lay={'pairs':[pair],'total_channels':5,
             'layers':{'8':{'start':0,'end':2},'32':{'start':2,'end':5}}}
        return pair,lay
    def test_centered_streaming_correlation_and_restore(self):
        pair,lay=self.small();rng=np.random.default_rng(42)
        x=rng.normal(size=(37,3,5)).astype(np.float32)
        x[:,:,2]=x[:,:,0]*3+7;x[:,:,3]=-x[:,:,1]
        m=Moments([pair]);m.update(x[:7],lay);m.update(x[7:],lay)
        actual=m.correlations()
        for c in range(3):
            for i in range(2):
                for j in range(3):
                    expected=np.corrcoef(x[:,c,i],x[:,c,j+2])[0,1]
                    self.assertAlmostEqual(actual[0,c,i,j],expected,places=12)
        with tempfile.TemporaryDirectory() as tmp:
            stores=open_stores(tmp,2,5,True)
            commit(tmp,stores,m,1,0)
            restored=Moments([pair])
            with np.load(Path(tmp)/'checkpoint.npz') as z:restored.restore(z)
            self.assertEqual(restored.n,37)
            np.testing.assert_array_equal(restored.correlations(),actual)
    def test_constant_channel_is_undefined(self):
        pair,lay=self.small();x=np.ones((12,3,5),np.float32)
        m=Moments([pair]);m.update(x,lay)
        self.assertTrue(np.isnan(m.correlations()).all())
    def test_four_sign_regimes_and_original_values(self):
        gate=np.array([[2,0],[2,0],[-2,0],[-2,0]],np.float32)
        value=np.array([[3,0],[-3,0],[3,0],[-3,0]],np.float32)
        act=gate/(1+np.exp(-gate))*value
        full=np.stack([gate,value,act],axis=1)
        peak,pos,count,local=passage_arrays(full)
        np.testing.assert_array_equal(pos[:,0],np.arange(4))
        np.testing.assert_array_equal(count[:,0],[1,1,1,1])
        self.assertTrue(np.isnan(peak[:,:,1]).all())
        np.testing.assert_array_equal(pos[:,1],np.full(4,65535))
        for r in range(4):np.testing.assert_array_equal(peak[r,:,0],full[r,:,0])
        np.testing.assert_allclose(local[0],full.mean(0),atol=1e-7)
    def test_top_sets_do_not_fill_with_zeros_or_break_ties(self):
        self.assertEqual(top_indices(np.arange(-20,21),10,True)[0],set(range(10)))
        self.assertEqual(top_indices(np.zeros(30),10),(None,'insufficient_signed_passages'))
        self.assertEqual(top_indices(np.ones(30),10),(None,'tied_cutoff'))
    def test_interrupted_allocation_can_be_reopened(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'peaks.npy'
            a=np.lib.format.open_memmap(p,mode='w+',dtype=np.float32,shape=(2,4,3,5));del a
            stores=open_stores(tmp,2,5,True)
            self.assertEqual(set(stores),{'peaks','positions','sign_counts','passage_moments'})
    def test_layout_excludes_targets_and_preserves_frozen_ids(self):
        source=Path(__file__).resolve().parents[1]/'selection.json'
        lay=layout(json.loads(source.read_text()),100,20261028)
        self.assertEqual(len(lay['pairs']),24);self.assertEqual(lay['total_channels'],2976)
        for p in lay['pairs']:
            self.assertEqual(p['target']-p['source'],24)
            self.assertFalse(set(p['neurons'])&set(p['controls']))
            self.assertEqual(len(set(p['controls'])),100)
    def test_default_budget_fits_four_gigabytes(self):
        self.assertLess(bytes_required(10000,256,512),4_000_000_000)

if __name__=='__main__':unittest.main()
