"""Deterministic controlled damage and scoring-only appearance diagnostics."""
import cv2
import numpy as np
from benchmark import error_metrics
from metric_plane import gradient_error

FAMILIES=['opaque_disc','opaque_vehicle','missing_strip','blur','clipping','jpeg']
SEVERITIES=['mild','severe']


def mask_shape(family):
    if family not in FAMILIES:raise ValueError('Unknown corruption family')
    return 'vehicle' if family=='opaque_vehicle' else 'strip' if family=='missing_strip' else 'disc'


def make_mask(basis,shape,severity):
    if basis.dtype!=bool or not basis.any() or severity not in SEVERITIES:raise ValueError('Invalid mask specification')
    y,x=np.nonzero(basis);cx=(x.min()+x.max())/2;cy=(y.min()+y.max())/2
    scale=.55 if severity=='mild' else .9;w=(x.max()-x.min()+1)*scale;h=(y.max()-y.min()+1)*scale
    out=np.zeros(basis.shape,np.uint8)
    if shape=='disc':cv2.ellipse(out,(round(cx),round(cy)),(max(1,round(w/2)),max(1,round(h/2))),0,0,360,1,-1)
    elif shape=='vehicle':
        points=np.array([[-.36,-.5],[.36,-.5],[.42,-.3],[.5,-.3],[.5,.12],[.42,.12],[.42,.5],[-.42,.5],[-.42,.12],[-.5,.12],[-.5,-.3],[-.42,-.3]])
        cv2.fillPoly(out,[np.rint(points*[w,h]+[cx,cy]).astype(np.int32)],1)
    elif shape=='strip':
        width=(x.max()-x.min()+1)*(.18 if severity=='mild' else .4)
        cv2.rectangle(out,(round(cx-width/2),y.min()),(round(cx+width/2),y.max()),1,-1)
    else:raise ValueError('Unknown mask shape')
    return (out>0)&basis


def damage(image,mask,family,severity):
    if family not in FAMILIES or severity not in SEVERITIES or image.dtype!=np.uint8 or mask.dtype!=bool:raise ValueError('Invalid damage request')
    out=image.copy();severe=severity=='severe'
    if family in ['opaque_disc','missing_strip']:out[mask]=0
    elif family=='opaque_vehicle':out[mask]=32
    elif family=='blur':out[mask]=cv2.GaussianBlur(image,(0,0),32 if severe else 12)[mask]
    elif family=='clipping':out[mask]=np.clip(image[mask].astype(float)*(3 if severe else 1.8)+(100 if severe else 60),0,255).astype(np.uint8)
    elif family=='jpeg':
        ok,encoded=cv2.imencode('.jpg',cv2.cvtColor(image,cv2.COLOR_RGB2BGR),[cv2.IMWRITE_JPEG_QUALITY,5 if severe else 20])
        if not ok:raise RuntimeError('JPEG encoder failed')
        out[mask]=cv2.cvtColor(cv2.imdecode(encoded,cv2.IMREAD_COLOR),cv2.COLOR_BGR2RGB)[mask]
    return out


def appearance_bands(pred,truth,accepted,mask):
    """Evaluation only: thresholds define measured appearance, not confidence."""
    error=np.abs(pred.astype(np.int16)-truth.astype(np.int16)).mean(2)
    gray=cv2.cvtColor(truth,cv2.COLOR_RGB2GRAY).astype(np.float32)
    gx=cv2.Sobel(gray,cv2.CV_32F,1,0,ksize=3,scale=1/8);gy=cv2.Sobel(gray,cv2.CV_32F,0,1,ksize=3,scale=1/8)
    textured=np.hypot(gx,gy)>=4
    counts={str(t):int((accepted&(error<=t)).sum()) for t in [10,20]}
    return dict(mask_pixels=int(mask.sum()),accepted_pixels=int(accepted.sum()),
        accepted_rgb_mae_le_threshold_pixels=counts,
        fraction_of_mask_with_rgb_mae_le_threshold={k:n/max(1,int(mask.sum())) for k,n in counts.items()},
        accepted_textured_pixels=int((accepted&textured).sum()),accepted_low_texture_pixels=int((accepted&~textured).sum()))


def baseline_stats(pred,truth,mask):
    return dict(coverage=1.,abstention=0.,accepted_pixels=int(mask.sum()),error=error_metrics(pred,truth,mask),
        gradient=gradient_error(pred,truth,mask),appearance=appearance_bands(pred,truth,mask,mask))
