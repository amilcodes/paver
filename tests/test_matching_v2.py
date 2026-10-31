import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import cv2
import numpy as np
from matching_v2 import fit_registration,SIFTMatcher,register_learned
from benchmark import policy_mask,error_metrics,masks

def grid():
    x,y=np.meshgrid(np.linspace(30,225,9),np.linspace(30,225,9))
    return np.c_[x.ravel(),y.ravel()].astype(np.float32)
def test_planar_fit_recovers_known_motion_and_rejects_reflection():
    points=grid();h=np.array([[1.,0.,5],[0.,1.,7],[0.,0.,1.]])
    image=np.full((256,256,3),100,np.uint8);mask=np.ones((256,256),bool)
    target=cv2.warpPerspective(image,h,(256,256))
    target_allowed=mask.copy();target_allowed[:10]=False;target_allowed[:,:10]=False
    mapped=cv2.perspectiveTransform(points[:,None],h)[:,0]
    result=fit_registration(points,mapped,image,target,mask,target_allowed)
    assert result.homography is not None
    assert np.max(np.abs(result.homography-h))<1e-3
    reflected=points.copy();reflected[:,0]=255-reflected[:,0]
    rejected=fit_registration(points,reflected,image,image,mask,mask)
    assert rejected.homography is None
def test_geometrically_consistent_wrong_appearance_is_rejected():
    image=np.full((256,256,3),40,np.uint8);mask=np.ones((256,256),bool)
    result=fit_registration(grid(),grid(),image,image+100,mask,mask)
    assert result.homography is None
    assert 'photometric' in result.diagnostics['reason']
def test_audit_does_not_read_excluded_pixels():
    image=np.full((256,256,3),100,np.uint8);mask=np.ones((256,256),bool)
    mask[90:170,90:170]=False;variant=image.copy();variant[~mask]=250
    a=fit_registration(grid(),grid(),image,image,mask,mask)
    b=fit_registration(grid(),grid(),image,variant,mask,mask)
    assert a.diagnostics['audit_median_rgb']==b.diagnostics['audit_median_rgb']==0
    assert a.homography is not None and b.homography is not None
def test_strict_policy_never_promotes_single_views_or_conflicts():
    count=np.array([[0,1,2,3]],np.uint8);difference=np.array([[0,0,9,4]],np.float32)
    base={'count':count,'disagreement':difference};mask=np.ones(count.shape,bool)
    accepted=policy_mask(base,mask,{'min_views':2,'max_disagreement':8})
    assert accepted.tolist()==[[False,False,False,True]]
def test_error_fraction_counts_only_accepted_pixels():
    truth=np.zeros((2,2,3),np.uint8);pred=truth.copy();pred[0,0]=21
    mask=np.zeros((2,2),bool);mask[0,0]=True
    assert error_metrics(pred,truth,mask)['over_20_fraction']==1
    assert error_metrics(pred,truth,~mask)['over_20_fraction']==0
    assert error_metrics(pred,truth,np.zeros_like(mask))['over_20_fraction'] is None

def test_learned_extraction_cannot_see_masked_target_values():
    import os,pytest
    if os.environ.get('RUN_LEARNED_TESTS')!='1':
        pytest.skip('Optional local model integration check; set RUN_LEARNED_TESTS=1')
    from matching_v2 import LearnedMatcher
    from PIL import Image
    root=Path(__file__).resolve().parents[1]
    image=np.array(Image.open(root/'benchmark_v2/results/dev_marked/sift/original.jpg').resize((512,512)))
    allowed=np.ones((512,512),bool);allowed[180:340,180:340]=False
    changed=image.copy();changed[~allowed]=255-image[~allowed]
    matcher=LearnedMatcher('auto',max_keypoints=256)
    a=matcher.describe(image,allowed);matcher.cache.clear()
    b=matcher.describe(changed,allowed)
    assert matcher.torch.equal(a['keypoints'],b['keypoints'])
    assert matcher.torch.equal(a['descriptors'],b['descriptors'])
    assert a['keypoints'].shape[1]>10
