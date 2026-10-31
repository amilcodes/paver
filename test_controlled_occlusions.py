import unittest
import numpy as np
from controlled_occlusions import make_mask,damage,FAMILIES,mask_shape,appearance_bands
from dense_experiment import masked_inputs


class ControlledTests(unittest.TestCase):
    def setUp(self):
        self.basis=np.zeros((240,320),bool);self.basis[20:220,30:290]=True
        self.image=np.random.default_rng(9).integers(0,256,(240,320,3),dtype=np.uint8)

    def test_masks_inside_reviewed_ground_and_nested(self):
        self.basis[60:90,70:100]=False
        for shape in ['disc','vehicle','strip']:
            a=make_mask(self.basis,shape,'mild');b=make_mask(self.basis,shape,'severe')
            self.assertGreater(a.sum(),500);self.assertFalse((a&~b).any());self.assertFalse((b&~self.basis).any())
            np.testing.assert_array_equal(b,make_mask(self.basis,shape,'severe'))

    def test_damage_deterministic_and_local(self):
        for family in FAMILIES:
            for severity in ['mild','severe']:
                mask=make_mask(self.basis,mask_shape(family),severity);out=damage(self.image,mask,family,severity)
                np.testing.assert_array_equal(out[~mask],self.image[~mask]);self.assertTrue((out[mask]!=self.image[mask]).any())
                np.testing.assert_array_equal(out,damage(self.image,mask,family,severity))

    def test_hidden_truth_invariance_for_every_condition(self):
        for family in FAMILIES:
            for severity in ['mild','severe']:
                mask=make_mask(self.basis,mask_shape(family),severity);out=damage(self.image,mask,family,severity)
                altered=self.image.copy();altered[mask]=255-self.image[mask]
                clean1,allowed1=masked_inputs([out],[self.basis],mask,0)
                clean2,allowed2=masked_inputs([altered],[self.basis],mask,0)
                np.testing.assert_array_equal(clean1,clean2);np.testing.assert_array_equal(allowed1,allowed2)
                self.assertFalse(clean1[0][mask].any())

    def test_bands_include_abstention_in_denominator(self):
        truth=np.zeros_like(self.image);pred=truth.copy();mask=self.basis
        take=mask.copy();take[:120]=False;pred[take]=10
        result=appearance_bands(pred,truth,take,mask)
        self.assertEqual(result['fraction_of_mask_with_rgb_mae_le_threshold']['10'],take.sum()/mask.sum())

    def test_bad_spec_rejected(self):
        with self.assertRaises(ValueError):make_mask(self.basis,'invented','mild')
        with self.assertRaises(ValueError):damage(self.image,self.basis,'invented','mild')


if __name__=='__main__':unittest.main()
