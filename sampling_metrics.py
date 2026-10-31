"""Post-registration comparisons on identical, gained and lost pixels."""
import numpy as np
from benchmark import policy_mask


def error_summary(values):
    n=len(values);e=values.astype(np.float64)/3
    return dict(pixels=n,mae_255=float(e.mean()) if n else None,p90_rgb=float(np.percentile(e,90)) if n else None,over_20_fraction=float((e>20).mean()) if n else None)


def partitions(new,old,policies,natural=False):
    mask=new['mask']
    if not np.array_equal(mask,old['mask']):raise ValueError('Control mask differs')
    result={}
    for name,p in policies.items():
        a=policy_mask(new,mask,p)[mask];b=policy_mask(old,mask,p)[mask];rows={}
        for label,take in [('common',a&b),('gained',a&~b),('lost',b&~a)]:
            v=dict(pixels=int(take.sum()),fraction_of_mask=float(take.sum()/mask.sum()))
            if not natural:
                if label!='lost':v['new']=error_summary(new['absolute_rgb_error_sum_on_mask'][take])
                if label!='gained':v['control']=error_summary(old['absolute_rgb_error_sum_on_mask'][take])
            rows[label]=v
        result[name]=rows
    return result
