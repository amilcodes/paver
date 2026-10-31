"""Actual MPS smoke/translation/isolation check; never substitutes for road scores."""
import argparse,json
from pathlib import Path
from datetime import datetime,timezone
import cv2
import numpy as np
from benchmark import ROOT,digest
from dense_matching import MaskedLoFTR
from dense_experiment import array_hash,immutable_json
from run_dense import heavy_job,verify_installed,certificate_signature


def fixture():
    rng=np.random.default_rng(41029);size=512
    image=cv2.GaussianBlur(rng.integers(20,235,(size,size,3),dtype=np.uint8),(3,3),0)
    for _ in range(100):
        x,y=rng.integers(30,size-30,2);color=tuple(int(v) for v in rng.integers(0,255,3))
        cv2.circle(image,(int(x),int(y)),int(rng.integers(3,15)),color,2)
    shift=np.array([16.,8.]);H=np.array([[1.,0,shift[0]],[0,1,shift[1]],[0,0,1.]])
    other=cv2.warpPerspective(image,H,(size,size))
    a=np.zeros((size,size),bool);a[24:-24,24:-24]=True;a[190:290,190:290]=False
    b=cv2.warpPerspective(a.astype(np.uint8),H,(size,size),flags=cv2.INTER_NEAREST).astype(bool)
    return image,other,a,b,shift


def run(weights,output):
    with heavy_job():
        verify_installed();sig=certificate_signature()
        model=MaskedLoFTR(weights,sig['model']['checkpoint_sha256'],root=ROOT)
        image,other,a,b,shift=fixture();p,q,extra=model.correspondences(image,other,a,b)
        changed=image.copy();changed[~a]=255-image[~a]
        changed2=other.copy();changed2[~b]=255-other[~b]
        pp,qq,_=model.correspondences(changed,changed2,a,b)
        invariant=np.array_equal(p,pp) and np.array_equal(q,qq)
        error=np.linalg.norm((q-p)-shift,axis=1)
        median=float(np.median(error)) if len(error) else None;p90=float(np.percentile(error,90)) if len(error) else None
        passed=len(p)>=32 and median<1.5 and p90<3 and invariant
        record=dict(signature=sig,passed=passed,actual_model_executed=True,
                    created_at_utc=datetime.now(timezone.utc).isoformat(),translation_matches=len(p),
                    translation_median_error_px=median,translation_p90_error_px=p90,
                    excluded_rgb_correspondence_invariance=invariant,matcher=extra,
                    fixture_sha256={k:array_hash(v) for k,v in dict(image=image,other=other,allowed0=a,allowed1=b,shift=shift).items()},
                    correspondence_sha256=dict(p=array_hash(p),q=array_hash(q)),
                    limitation='Tests translation accuracy and excluded-pixel invariance only. Does not validate absolute coordinate centers under projective transforms, dense-match calibration, or recovery accuracy on roads.')
        immutable_json(output,record);print(json.dumps(record,indent=2))
        if not passed:raise RuntimeError('Actual model preflight failed; inspect saved record before any road inference')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--weights',required=True);p.add_argument('--output',required=True)
    args=p.parse_args();run(Path(args.weights).resolve(),Path(args.output).resolve())
