"""Actual MPS resize/crop/projective/isolation validation before road scoring."""
import argparse,json
from pathlib import Path
from datetime import datetime,timezone
import cv2
import numpy as np
from benchmark import ROOT,digest
from sampling_matcher import SamplingMatcher,signature
from dense_experiment import immutable_json,array_hash
from run_dense import heavy_job,verify_installed,ref,read_ref

OUT=ROOT/'sampling_v1'


def fixture():
    rng=np.random.default_rng(731991);small=cv2.GaussianBlur(rng.integers(10,245,(880,880,3),dtype=np.uint8),(5,5),0)
    for _ in range(230):
        x,y=rng.integers(20,860,2);color=tuple(int(v) for v in rng.integers(0,255,3));cv2.circle(small,(int(x),int(y)),int(rng.integers(3,16)),color,2)
    image=cv2.resize(small,(1760,1760));allowed=np.zeros(image.shape[:2],bool);allowed[80:-80,64:-96]=True;allowed[600:1000,670:1050]=False
    angle=np.deg2rad(1.5);scale=1.025
    H=np.array([[scale*np.cos(angle),-scale*np.sin(angle),24.],[scale*np.sin(angle),scale*np.cos(angle),-18.],[8e-6,-6e-6,1.]])
    return image,allowed,H


def check_certificate(reference):
    c=json.loads(read_ref(reference).read_text())
    if not c.get('passed') or c['signature']!=signature() or c['validator_sha256']!=digest(Path(__file__)):raise RuntimeError('Invalid or stale sampling certificate')
    if c.get('device')!='mps' or not c.get('excluded_rgb_correspondence_invariance'):raise RuntimeError('Model/device/isolation check missing')
    if any(r['matches']<32 or r['median_error_px']>=1.5 or r['p90_error_px']>=3 for r in c['transforms']):raise RuntimeError('Geometric fixture failed')
    return c


def run(weights):
    with heavy_job():
        verify_installed();sig=signature();model=SamplingMatcher(weights,sig['checkpoint_sha256'],root=ROOT)
        image,a,projective=fixture();rows=[];isolation=True
        for name,H in [('translation',np.array([[1.,0.,32.],[0.,1.,24.],[0.,0.,1.]])),('projective',projective)]:
            b=cv2.warpPerspective(a.astype(np.uint8),H,(1760,1760),flags=cv2.INTER_NEAREST).astype(bool);other=cv2.warpPerspective(image,H,(1760,1760))
            p,q,extra=model.correspondences(image,other,a,b);expected=cv2.perspectiveTransform(p[None].astype(np.float64),H)[0];error=np.linalg.norm(q-expected,axis=1)
            changed=image.copy();changed[~a]=255-image[~a];changed2=other.copy();changed2[~b]=255-other[~b]
            pp,qq,_=model.correspondences(changed,changed2,a,b);same=np.array_equal(p,pp) and np.array_equal(q,qq);isolation&=same
            rows.append(dict(name=name,matches=len(p),median_error_px=float(np.median(error)) if len(p) else None,p90_error_px=float(np.percentile(error,90)) if len(p) else None,
                hidden_rgb_invariant=same,transform=H.tolist(),matcher=extra,fixture_hashes={k:array_hash(v) for k,v in dict(image=image,other=other,allowed0=a,allowed1=b).items()},correspondence_hashes=dict(p=array_hash(p),q=array_hash(q))))
        passed=isolation and all(r['matches']>=32 and r['median_error_px']<1.5 and r['p90_error_px']<3 for r in rows)
        record=dict(created_at_utc=datetime.now(timezone.utc).isoformat(),signature=sig,validator_sha256=digest(Path(__file__)),device='mps',actual_model_executed=True,
            passed=passed,excluded_rgb_correspondence_invariance=isolation,transforms=rows,
            limitation='Known synthetic transforms validate coordinate restoration and excluded-input invariance, not road geometry or confidence calibration.')
        immutable_json(OUT/'model_validation_001.json',record);print(json.dumps(record,indent=2))
        if not passed:raise RuntimeError('Synthetic validation failed; no road inference permitted')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--weights',required=True);args=parser.parse_args();run(args.weights)
