"""Scoring-only comparison on identical accepted pixels; never an output gate.

Post-hoc descriptive diagnostic of the frozen baseline, not a predeclared model
selection test. Correlated conditions/pixels are not independent samples.
"""
import json
from pathlib import Path
from collections import defaultdict
import numpy as np
from benchmark import ROOT,digest,policy_mask
from run_dense import heavy_job,read_ref,ref
from run_controlled_benchmark import checked_plan,load_area,load_evidence
from dense_experiment import immutable_json


def compare():
    plan=checked_plan();out=ROOT/'controlled_v1';audit=json.loads((out/'audit_001.json').read_text())
    if audit.get('passed') is not True or audit['plan_sha256']!=digest(out/'execution_plan.json'):raise RuntimeError('Complete audit required')
    aggregate=defaultdict(lambda:dict(pixels=0,source_error_sum=0,baseline_error_sum=0,source_better_pixels=0,source_worse_pixels=0,equal_pixels=0));details=[]
    with heavy_job():
        for area in plan['areas']:
            images,allowed,bundle=load_area(area);truth=images[area['target']]
            for s in area['scenarios']:
                record=json.loads((out/area['area']/'scenarios'/(s['key']+'.json')).read_text());g=json.loads(read_ref(record['geometry']).read_text());mask=bundle[s['mask_key']];t=truth[mask].astype(np.int16)
                baseline={name:np.abs(load_evidence(d['evidence'])['prediction_on_mask'].astype(np.int16)-t).sum(1) for name,d in record['baselines'].items()}
                for method,data in g['methods'].items():
                    e=load_evidence(data['evidence']);errors=np.abs(e['prediction_on_mask'].astype(np.int16)-t).sum(1)
                    if not np.array_equal(errors,e['absolute_rgb_error_sum_on_mask']):raise RuntimeError('Source errors differ from audited truth')
                    for name,p in plan['policies'].items():
                        select=policy_mask(e,mask,p)[mask];n=int(select.sum());x=errors[select]
                        for control,ce in baseline.items():
                            y=ce[select];values=dict(pixels=n,source_error_sum=int(x.sum()),baseline_error_sum=int(y.sum()),source_better_pixels=int((x<y).sum()),source_worse_pixels=int((x>y).sum()),equal_pixels=int((x==y).sum()))
                            details.append(dict(area=area['area'],scenario=s['key'],method=method,policy=name,control=control,**values))
                            ag=aggregate[(area['area'],s['family'],method,name,control)]
                            for k,v in values.items():ag[k]+=v
            del images,truth,bundle
    rows=[]
    for key,v in aggregate.items():
        n=v['pixels'];rows.append(dict(zip(['area','family','method','policy','control'],key),**v,
            source_mae_255=v['source_error_sum']/(3*n) if n else None,control_mae_255=v['baseline_error_sum']/(3*n) if n else None,
            source_minus_control_mae_255=(v['source_error_sum']-v['baseline_error_sum'])/(3*n) if n else None))
    result=dict(source=ref(Path(__file__)),plan=ref(out/'execution_plan.json'),audit=ref(out/'audit_001.json'),
        scope='Descriptive post-hoc same-pixel control; no output/policy/registration changed. All evaluation truth stays scoring-only.',aggregates=rows,conditions=details)
    immutable_json(out/'same_pixel_controls.json',result)
    for r in rows:
        if r['area']=='gironde' and r['method']=='loftr_metric_free' and r['policy']=='candidate':print({k:r[k] for k in ['family','control','source_mae_255','control_mae_255','source_minus_control_mae_255']})


if __name__=='__main__':compare()
