"""Controls for scoring denominators, abstention and hidden-RGB isolation."""
import unittest
import numpy as np
from area_recovery_metrics import extended_metrics,same_pixels
from audit_expanded_baseline import expected_extended,expected_same,verify_extended
from dense_experiment import masked_inputs
from validate_dense_run_v2 import compare


class ExpandedBaselineTests(unittest.TestCase):
    def setUp(self):
        self.mask=np.array([[True,True],[True,False]])
        self.take=np.array([[True,False],[True,False]])
        self.weights=np.array([[1.,2.],[7.,10.]])
        self.texture=np.array([[True,True],[False,False]])
        self.truth=np.full((2,2,3),30,np.uint8)
        self.pred=self.truth.copy();self.pred[0,0]=[0,0,0];self.pred[1,0]=[33,33,33]

    def metrics(self,take=None,truth=True):
        return extended_metrics(self.pred,self.truth if truth else None,self.take if take is None else take,self.mask,self.weights,self.texture if truth else None)

    def test_angular_coverage_is_not_pixel_coverage(self):
        m=self.metrics();self.assertAlmostEqual(m['angular_coverage'],.8);self.assertEqual(m['accepted_pixels'],2)
        self.assertAlmostEqual(m['angular_rgb_mae_255'],51/8);self.assertAlmostEqual(m['angular_rgb_mae_le_20_fraction_of_mask'],.7)
        self.assertEqual(m['accepted_rgb_mae_le_threshold_pixels'],{'10':1,'20':1})

    def test_texture_has_separate_population_and_error(self):
        m=self.metrics()['strata'];self.assertEqual(m['textured']['mask_pixels'],2);self.assertEqual(m['textured']['coverage'],.5)
        self.assertEqual(m['textured']['rgb_mae_255'],30);self.assertEqual(m['textured']['over_20_fraction'],1)
        self.assertEqual(m['low_texture']['rgb_mae_255'],3)

    def test_abstention_is_not_zero_error(self):
        m=self.metrics(np.zeros((2,2),bool));self.assertEqual(m['angular_abstention'],1)
        self.assertIsNone(m['angular_rgb_mae_255']);self.assertIsNone(m['strata']['textured']['rgb_mae_255'])

    def test_natural_support_has_no_truth_fields(self):
        m=self.metrics(truth=False);self.assertNotIn('strata',m);self.assertNotIn('angular_rgb_mae_255',m)
        with self.assertRaises(ValueError):extended_metrics(self.pred,None,self.take,self.mask,self.weights,self.texture)

    def test_invalid_denominators_or_acceptance_rejected(self):
        for w in [np.zeros((2,2)),np.full((2,2),np.nan),np.full((2,2),-1.)]:
            with self.assertRaises(ValueError):extended_metrics(self.pred,self.truth,self.take,self.mask,w,self.texture)
        with self.assertRaises(ValueError):extended_metrics(self.pred,self.truth,np.ones((2,2),bool),self.mask,self.weights,self.texture)
        with self.assertRaises(ValueError):extended_metrics(self.pred,self.truth,np.zeros((2,2),bool),np.zeros((2,2),bool),self.weights,self.texture)

    def test_independent_metric_recomputation_and_corruption_rejection(self):
        actual=self.metrics();expected=expected_extended(self.pred,self.truth,self.take,self.mask,self.weights,self.texture)
        verify_extended(actual,expected)
        actual['angular_coverage']+=.01
        with self.assertRaises(ValueError):verify_extended(actual,expected)

    def test_same_pixels_keeps_wins_losses_and_empty(self):
        control=self.truth.copy();control[1,0]=40
        m=same_pixels(self.pred,control,self.truth,self.take);compare(m,expected_same(self.pred,control,self.truth,self.take))
        self.assertEqual((m['source_better_pixels'],m['source_worse_pixels'],m['equal_pixels']),(1,1,0))
        self.assertIsNone(same_pixels(self.pred,control,self.truth,np.zeros((2,2),bool))['source_mae_255'])

    def test_hidden_truth_and_nuisance_rgb_do_not_reach_registration_input(self):
        images=[np.full((20,20,3),i*30,np.uint8) for i in range(5)];allowed=[np.ones((20,20),bool) for _ in images]
        for a in allowed:a[:4,:]=False
        hidden=np.zeros((20,20),bool);hidden[8:12,8:12]=True
        before,ma=masked_inputs(images,allowed,hidden,2);changed=[im.copy() for im in images]
        for im,a in zip(changed,allowed):im[~a]=255
        changed[2][hidden]=255;after,mb=masked_inputs(changed,allowed,hidden,2)
        self.assertTrue(all(np.array_equal(x,y) for x,y in zip(before,after)));self.assertTrue(all(np.array_equal(x,y) for x,y in zip(ma,mb)))


if __name__=='__main__':unittest.main()
