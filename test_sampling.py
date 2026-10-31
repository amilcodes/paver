import unittest
import numpy as np
from dense_matching import prepare_masked,restore_points,filter_matches
from sampling_matcher import MAX_SIDE


class SamplingTests(unittest.TestCase):
    def test_crop_resize_coordinate_round_trip(self):
        image=np.zeros((1600,1800,3),np.uint8);mask=np.zeros(image.shape[:2],bool);mask[125:1437,187:1675]=True
        prepared=prepare_masked(image,mask,MAX_SIDE);p=np.array([[350.2,410.6],[1020.3,1110.7]])
        raw=(p+.5-np.asarray(prepared['origin']))*np.asarray(prepared['scale'])
        np.testing.assert_allclose(restore_points(raw,prepared),p,atol=1e-10)
        self.assertTrue(max(prepared['gray'].shape)<=MAX_SIDE);self.assertTrue(max(prepared['scale'])<1)

    def test_excluded_rgb_does_not_affect_preprocessing(self):
        rng=np.random.default_rng(18);image=rng.integers(0,256,(1500,1400,3),dtype=np.uint8);mask=np.ones(image.shape[:2],bool);mask[:140]=False;mask[700:900,600:800]=False
        a=prepare_masked(image,mask,MAX_SIDE);image[~mask]=255-image[~mask];b=prepare_masked(image,mask,MAX_SIDE)
        for key in ['gray','mask','origin','scale']:np.testing.assert_equal(a[key],b[key])

    def test_repeated_endpoint_cannot_inflate_match_support(self):
        a=np.ones((256,256),bool);a[:20]=False
        p=np.array([[80,80],[80.1,80.1],[120,120],[140,140]],float);q=np.array([[85,85],[86,86],[160,160],[160.1,160.1]],float)
        pp,qq,c=filter_matches(p,q,np.array([.8,.9,.8,.7]),a,a)
        self.assertEqual(len(pp),2);self.assertEqual(len(set(map(tuple,np.rint(pp/6)))),2);self.assertEqual(len(set(map(tuple,np.rint(qq/6)))),2)

    def test_mask_margin_still_in_original_coordinates(self):
        mask=np.ones((256,256),bool);mask[:40]=False
        p=np.array([[100,65],[100,74],[150,90]],float)
        pp,_,_=filter_matches(p,p,np.ones(3),mask,mask)
        self.assertEqual(len(pp),2);self.assertTrue((pp[:,1]>=74).all())


if __name__=='__main__':unittest.main()
