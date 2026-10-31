"""Regression controls for sequence extension without changing central evidence."""
import json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from dense_experiment import cached_pair,masked_inputs,pair_indices
from sequence_geometry import propose_edge,transform
from run_dense_long import remapped_proposal,partition_hashes,register_long


class StableExtensionTests(unittest.TestCase):
    def observations(self):
        rng=np.random.default_rng(71);p=rng.uniform(40,250,(230,2)).astype(np.float32)
        H=np.array([[1.01,.03,12],[-.01,.99,4],[.00002,-.00001,1]])
        q=(transform(H,p)+rng.normal(0,.1,p.shape)).astype(np.float32)
        q[-30:]=rng.uniform(40,250,(30,2))
        return p,q

    def test_relabel_retains_training_and_heldout(self):
        p,q=self.observations();old,info=propose_edge(1,2,p,q)
        new,got=remapped_proposal(4,5,p,q,[1,2])
        self.assertIsNotNone(old);self.assertEqual(partition_hashes(old),partition_hashes(new))
        self.assertEqual((new['i'],new['j']),(4,5));self.assertEqual(got,dict(info,i=4,j=5))
        naive,_=propose_edge(4,5,p,q)
        self.assertNotEqual(partition_hashes(old)['vp'],partition_hashes(naive)['vp'])

    def test_negative_seeds_forbidden(self):
        p,q=self.observations()
        with self.assertRaises(ValueError):remapped_proposal(0,1,p,q,[-3,-2])

    def test_topology_has_seven_preserved_pairs(self):
        pairs=pair_indices(11,5);central={(i-3,j-3) for i,j in pairs if 3<=i<=7 and 3<=j<=7}
        self.assertEqual(central,set(pair_indices(5,2)));self.assertEqual(len(pairs),25)

    def test_hidden_rgb_is_irrelevant_to_cache(self):
        rng=np.random.default_rng(2);images=[rng.integers(0,256,(96,96,3),dtype=np.uint8) for _ in range(2)]
        allowed=[np.ones((96,96),bool) for _ in images];hidden=np.zeros((96,96),bool);hidden[30:60,30:60]=1
        a,ma=masked_inputs(images,allowed,hidden,0);changed=[x.copy() for x in images];changed[0][hidden]^=255
        b,mb=masked_inputs(changed,allowed,hidden,0)
        for x,y in zip(a,b):np.testing.assert_array_equal(x,y)
        points=np.array([[15,15],[75,75]],np.float32)
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'pair.npz'
            cached_pair(path,0,1,a,ma,lambda *x:(points,points,{}),{'model':'fixture'})
            def forbidden(*args):self.fail('Hidden truth change caused re-inference')
            _,_,_,resumed=cached_pair(path,0,1,b,mb,forbidden,{'model':'fixture'})
            self.assertTrue(resumed)

    def test_cache_rejects_visible_rgb_mask_and_model_changes(self):
        a=[np.ones((96,96,3),np.uint8) for _ in range(2)];allowed=[np.ones((96,96),bool) for _ in a]
        p=np.array([[15,15]],np.float32)
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'pair.npz';cached_pair(path,0,1,a,allowed,lambda *x:(p,p,{}),{'model':'A'})
            b=[x.copy() for x in a];b[0][20,20]=42
            masks=[x.copy() for x in allowed];masks[0][20,20]=False
            for ims,als,model in [(b,allowed,{'model':'A'}),(a,masks,{'model':'A'}),(a,allowed,{'model':'B'})]:
                with self.assertRaises(RuntimeError):cached_pair(path,0,1,ims,als,None,model)

    def test_unsanitized_input_rejected_before_matching(self):
        images=[np.ones((8,8,3),np.uint8) for _ in range(11)];allowed=[np.zeros((8,8),bool) for _ in images]
        with self.assertRaisesRegex(ValueError,'Unsanitized'):
            register_long(images,allowed,5,160,{},Path('unused'),None,{})


if __name__=='__main__':unittest.main()
