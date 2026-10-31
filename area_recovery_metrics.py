"""Scoring-only angular/texture diagnostics. Not confidence or selection gates."""
import numpy as np


def extended_metrics(pred,truth,take,mask,weights,texture=None):
    if mask.dtype!=bool or take.dtype!=bool or (take&~mask).any():raise ValueError('Acceptance must lie inside boolean mask')
    if weights.shape!=mask.shape or not np.isfinite(weights).all() or (weights<=0).any():raise ValueError('Invalid angular weights')
    n=int(mask.sum());k=int(take.sum());omega=float(weights[mask].sum());used=float(weights[take].sum())
    if not n:raise ValueError('Empty evaluation mask')
    out=dict(mask_pixels=n,accepted_pixels=k,mask_solid_angle_sr=omega,accepted_solid_angle_sr=used,angular_coverage=used/omega,angular_abstention=1-used/omega)
    if truth is None:
        if texture is not None:raise ValueError('Natural obstruction cannot receive truth strata')
        return out
    if texture is None or texture.dtype!=bool or texture.shape!=mask.shape:raise ValueError('Missing scoring-only texture strata')
    e=np.abs(pred[take].astype(np.int16)-truth[take].astype(np.int16)).sum(1).astype(np.float64)/3
    w=weights[take];out.update(angular_rgb_mae_255=float(np.dot(e,w)/used) if used else None,
        angular_rgb_mae_le_20_fraction_of_mask=float(w[e<=20].sum()/omega),
        accepted_rgb_mae_le_threshold_pixels={str(t):int((e<=t).sum()) for t in [10,20]},strata={})
    for name,stratum in [('textured',texture),('low_texture',~texture)]:
        total=int((mask&stratum).sum());chosen=take&stratum;count=int(chosen.sum())
        errors=np.abs(pred[chosen].astype(np.int16)-truth[chosen].astype(np.int16)).sum(1).astype(np.float64)/3
        out['strata'][name]=dict(mask_pixels=total,accepted_pixels=count,coverage=count/total if total else None,
            rgb_mae_255=float(errors.mean()) if count else None,over_20_fraction=float((errors>20).mean()) if count else None)
    return out


def same_pixels(pred,baseline,truth,take):
    a=np.abs(pred[take].astype(np.int16)-truth[take].astype(np.int16)).sum(1)
    b=np.abs(baseline[take].astype(np.int16)-truth[take].astype(np.int16)).sum(1)
    n=len(a)
    return dict(pixels=n,source_mae_255=float(a.mean()/3) if n else None,control_mae_255=float(b.mean()/3) if n else None,
        source_better_pixels=int((a<b).sum()),source_worse_pixels=int((a>b).sum()),equal_pixels=int((a==b).sum()))
