import unittest
import numpy as np
from prepare_area_benchmark import solid_angles,assemble
from dense_experiment import masked_inputs
from controlled_occlusions import make_mask,FAMILIES,SEVERITIES,damage,mask_shape
from audit_area_annotations import verify_source_masks,quadrature


class AreaBenchmarkTests(unittest.TestCase):
    def test_exact_square_solid_angle_and_resolution_invariance(self):
        for n in [20,100,400]:
            w=solid_angles(n,90)
            self.assertAlmostEqual(w.sum(),2*np.pi/3,places=12)
            self.assertTrue((w>0).all());np.testing.assert_allclose(w,w[::-1,::-1],atol=1e-15)
            self.assertGreater(w[n//2,n//2],w[0,0])

    def test_independent_spherical_triangle_reference(self):
        n=80;fov=163.302054963582;f=n/(2*np.tan(np.deg2rad(fov/2)));w=solid_angles(n,fov)
        for x,y in [(0,0),(39,39),(7,63)]:
            rays=np.array([[x-n/2,y-n/2,f],[x+1-n/2,y-n/2,f],[x+1-n/2,y+1-n/2,f],[x-n/2,y+1-n/2,f]],float)
            rays/=np.linalg.norm(rays,axis=1)[:,None]
            def triangle(a,b,c):return 2*np.arctan2(abs(np.dot(a,np.cross(b,c))),1+np.dot(a,b)+np.dot(b,c)+np.dot(c,a))
            reference=triangle(*rays[:3])+triangle(rays[0],rays[2],rays[3])
            self.assertAlmostEqual(w[y,x],reference,places=13)

    def test_bad_projection(self):
        for n,fov in [(0,90),(10,180),(10,-10),(2.5,90)]:
            with self.assertRaises(ValueError):solid_angles(n,fov)

    def test_every_corruption_has_identical_sanitized_input(self):
        rng=np.random.default_rng(321);im=rng.integers(0,256,(240,240,3),dtype=np.uint8);allowed=np.ones((240,240),bool);basis=np.zeros_like(allowed);basis[40:200,40:200]=True
        for family in FAMILIES:
            for severity in SEVERITIES:
                mask=make_mask(basis,mask_shape(family),severity);changed=damage(im,mask,family,severity)
                a,ma=masked_inputs([im],[allowed],mask,0);b,mb=masked_inputs([changed],[allowed],mask,0)
                np.testing.assert_array_equal(a,b);np.testing.assert_array_equal(ma,mb)

    def annotation(self):
        def box(x0,y0,x1,y1):return [[x0,y0],[x1,y0],[x1,y1],[x0,y1]]
        f=dict(ground=[box(0,0,1,1)],ego=[box(.4,.4,.6,.6)],moving=[box(.1,.75,.2,.85)],shadow=[box(.4,0,1,.2)],privacy=[],uncertain=[])
        return dict(projection=dict(size=240,fov=145),boundary_margin_px=2,ground_margin_px=3,frame_order=['target'],frames=dict(target=f),regions=dict(rear=[box(.1,.7,.9,.95)]))

    def test_independent_validator_accepts_valid_masks(self):
        a=self.annotation();parts,_=assemble(a);verify_source_masks(a,parts)

    def test_independent_validator_rejects_ego_or_moving_leak(self):
        a=self.annotation();parts,_=assemble(a)
        for name in ['ego','moving']:
            bad={k:v.copy() for k,v in parts.items()};y,x=np.argwhere(bad['target_'+name])[0];bad['target_allowed'][y,x]=True
            with self.assertRaisesRegex(ValueError,'allowed mask'):verify_source_masks(a,bad)

    def test_quadrature_matches_exact_angular_integration(self):
        np.testing.assert_allclose(quadrature(200,90),solid_angles(200,90),rtol=1e-9,atol=1e-12)


if __name__=='__main__':unittest.main()
