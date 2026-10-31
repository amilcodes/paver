"""Behavioral checks for unsafe transfers and evaluation leakage."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import cv2
import numpy as np
import pytest
from repair import Registration, fuse, register, nadir, paste_nadir, score


def identity(shape):
    return Registration(np.eye(3),np.ones(shape,bool),
                        {'inliers':30,'median_reprojection_px':0.2})


def test_masked_donor_cannot_fill_and_unmasked_target_is_unchanged():
    target=np.full((64,64,3),20,np.uint8)
    source=np.full_like(target,180)
    region=np.zeros((64,64),bool);region[20:44,20:44]=True
    allowed=np.ones((64,64),bool);allowed[24:40,24:40]=False
    result=fuse(target,region,[source],[allowed],[identity(region.shape)])
    assert not result['accepted'][24:40,24:40].any()
    assert np.array_equal(result['image'][~region],target[~region])
    assert np.all(result['image'][result['accepted']]==180)


def test_conflicting_views_abstain_instead_of_averaging():
    target=np.zeros((32,32,3),np.uint8);mask=np.ones((32,32),bool)
    result=fuse(target,mask,[target+60,target+200],[mask,mask],
                [identity(mask.shape),identity(mask.shape)])
    assert not result['accepted'].any()
    assert np.all(result['provenance']==-1)


def test_single_view_is_distinguished_from_two_view_agreement():
    target=np.zeros((32,32,3),np.uint8);mask=np.ones((32,32),bool)
    one=fuse(target,mask,[target+60],[mask],[identity(mask.shape)],min_views=2)
    two=fuse(target,mask,[target+60,target+65],[mask,mask],
             [identity(mask.shape),identity(mask.shape)],min_views=2)
    assert not one['accepted'].any()
    assert two['accepted'].all()
    assert np.all(two['count']==2)


def test_no_source_and_blank_images_abstain():
    im=np.zeros((128,128,3),np.uint8);mask=np.ones((128,128),bool)
    assert register(im,im,mask,mask).homography is None
    assert not fuse(im,mask,[],[],[])['accepted'].any()


def test_registration_recovers_known_motion_without_hidden_pixel_leakage():
    rng=np.random.default_rng(97)
    source=rng.integers(0,256,(512,512,3),dtype=np.uint8)
    source=cv2.GaussianBlur(source,(3,3),.7)
    true_H=np.array([[1.,0.,24.],[0.,1.,-13.],[0.,0.,1.]])
    target=cv2.warpPerspective(source,true_H,(512,512))
    src_mask=np.ones((512,512),bool)
    target_mask=np.ones((512,512),bool)
    target_mask[:40]=False;target_mask[-40:]=False
    target_mask[:,:40]=False;target_mask[:,-40:]=False
    target_mask[180:300,180:300]=False
    variant=target.copy();variant[~target_mask]=rng.integers(0,256,(int((~target_mask).sum()),3))
    a=register(source,target,src_mask,target_mask)
    b=register(source,variant,src_mask,target_mask)
    assert a.homography is not None
    assert np.allclose(a.homography,b.homography)
    points=np.float32([[[100,100]],[[400,400]],[[256,256]]])
    assert np.max(np.abs(cv2.perspectiveTransform(points,a.homography)-
                         cv2.perspectiveTransform(points,true_H)))<1


def test_score_only_counts_requested_region():
    truth=np.zeros((10,10,3),np.uint8);prediction=truth.copy()
    prediction[0,0]=10;mask=np.zeros((10,10),bool);mask[0,0]=True
    assert score(prediction,truth,mask)['mae_255']==10
    assert score(prediction,truth,np.zeros_like(mask))['psnr_db'] is None


def test_projection_handles_pole_and_invalid_input():
    # All rows differ; a downward optical axis must sample the bottom row.
    panorama=np.broadcast_to(np.arange(64,dtype=np.uint8)[:,None,None],(64,128,3)).copy()
    projected=nadir(panorama,65,120)
    assert projected[32,32,0]==63
    with pytest.raises(ValueError):nadir(np.zeros((50,50,3),np.uint8))
    with pytest.raises(ValueError):nadir(panorama,fov=180)


def test_panorama_export_preserves_all_unsupported_pixels_and_upper_hemisphere():
    original=np.full((128,256,3),40,np.uint8)
    patch=np.full((128,128,3),170,np.uint8)
    mask=np.zeros((128,128),bool);mask[30:100,30:100]=True
    output,support=paste_nadir(original,patch,mask,145)
    assert support.any()
    assert not support[:64].any()
    assert np.array_equal(output[~support],original[~support])
    assert np.all(output[support]==170)
