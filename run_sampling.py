"""Bounded sampling ablation over every frozen area and hard-road control."""
import argparse,gc,json,time
from datetime import datetime,timezone
from pathlib import Path
import cv2
import numpy as np
from benchmark import ROOT,digest,policy_mask
from dense_experiment import immutable_json,masked_inputs,register_sequence,score_transfer
from run_dense import SOURCES,ref,read_ref,save_evidence,heavy_job,verify_installed,load_case
from run_controlled_benchmark import load_area,load_evidence
from run_expanded_baseline import load_scene
from area_recovery_metrics import extended_metrics,same_pixels
from prepare_area_benchmark import solid_angles
from resource_guard import require_storage
from sampling_matcher import SamplingMatcher,signature
from validate_sampling_model import check_certificate
from sampling_metrics import partitions

OUT=ROOT/'sampling_v1'


def build_plan():
    scenes=[];expanded=json.loads((ROOT/'controlled_v2/baseline_v1/execution_plan.json').read_text());initial=json.loads((ROOT/'controlled_v1/execution_plan.json').read_text());hard=json.loads((ROOT/'sequence_v6/execution_plan.json').read_text())
    natural=json.loads((ROOT/'natural_track_v1/summary.json').read_text());natural_refs={a['area']:a['record'] for a in natural['results']}
    for s in expanded['scenes']:
        folder=ROOT/'controlled_v2/baseline_v1'/s['scene']
        jobs=[dict(key=j['key'],kind=j['kind'],mask=s['masks'],mask_key=j['mask_key'],control=ref(folder/'groups'/(j['key']+'.json'))) for j in s['jobs']]
        conditions=[dict(spec=c,job=c['mask_key'].removeprefix('hidden_'),baseline=ref(folder/'conditions'/(c['key']+'.json'))) for c in s['conditions']]
        scenes.append(dict(scene=s['scene'],area=s['area'],split=s['split'],loader='expanded',source=s,projection=s['projection'],target=s['target'],frames=s['frames'],cache_group=s['scene'],jobs=jobs,conditions=conditions))
    for s in initial['areas']:
        folder=ROOT/'controlled_v1'/s['area'];jobs=[dict(key=m['key'],kind='controlled',mask=s['masks'],mask_key=m['key'],control=ref(folder/'groups'/(m['key']+'.json'))) for m in s['mask_specs']]
        jobs.append(dict(key='footprint',kind='natural',mask=s['masks'],mask_key='footprint',control=natural_refs[s['area']]))
        conditions=[dict(spec=c,job=c['mask_key'],baseline=ref(folder/'scenarios'/(c['key']+'.json'))) for c in s['scenarios']]
        scenes.append(dict(scene=s['area']+'_original',area=s['area'],split=s['split'],loader='initial',source=s,projection=s['projection'],target=s['target'],frames=s['frames'],cache_group=s['area']+'_original',jobs=jobs,conditions=conditions))
    for s in hard['cases']:
        if s['case'] not in ['gironde_0','alsace_fields_0']:continue
        area='gironde' if s['case']=='gironde_0' else 'alsace';jobs=[]
        for j in s['jobs']:
            if area=='gironde' and j['region']=='footprint':continue # Same frozen sources/mask already included in original natural track.
            jobs.append(dict(key=j['region'],kind='natural' if j['region']=='footprint' else 'hard_control',mask=j['mask'],mask_key=j['mask_key'],control=ref(ROOT/'sequence_v6/results'/s['case']/(j['region']+'.json'))))
        scenes.append(dict(scene=area+'_hard',area=area,split=s['split'],loader='hard',source=s,projection=s['projection'],target=2,frames=s['frames'],cache_group=area+'_original' if area=='gironde' else 'alsace_hard',jobs=jobs,conditions=[]))
    assert sum(len(s['jobs']) for s in scenes)==126 and sum(len(s['conditions']) for s in scenes)==216
    sources=list(dict.fromkeys(SOURCES+['run_controlled_benchmark.py','run_expanded_baseline.py','area_recovery_metrics.py','prepare_area_benchmark.py',
        'sampling_matcher.py','sampling_metrics.py','validate_sampling_model.py','run_sampling.py','audit_sampling.py','summarize_sampling.py','test_sampling.py',
        'audit_expanded_baseline.py','audit_area_annotations.py','audit_controlled_benchmark.py','validate_dense_run_v2.py']))
    plan=dict(protocol=ref(OUT/'protocol.json'),signature={n:digest(ROOT/n) for n in sources},model=signature(),certificate=ref(OUT/'model_validation_001.json'),policies=expanded['policies'],scenes=scenes,
        baseline_plans=[ref(ROOT/p) for p in ['controlled_v2/baseline_v1/execution_plan.json','controlled_v1/execution_plan.json','sequence_v6/execution_plan.json']],
        baseline_audits=[ref(ROOT/p) for p in ['controlled_v2/baseline_v1/audit_001.json','controlled_v1/audit_001.json','natural_track_v1/summary.json','sequence_v6/gironde_audit_002.json','sequence_v6/alsace_audit_001.json']],
        groups=126,methods=['loftr_homography','loftr_metric_free'],conditions=216,independent_test=False,
        alias='Original Gironde footprint also covers the identical hard-road footprint control; compute once. Donor-only caches shared for identical Gironde source windows.')
    check_certificate(plan['certificate']);immutable_json(OUT/'execution_plan.json',plan);return plan


def check_plan():
    p=json.loads((OUT/'execution_plan.json').read_text())
    for f,h in p['signature'].items():read_ref(dict(path=f,sha256=h))
    for r in p['baseline_plans']+p['baseline_audits']+[p['protocol']]:read_ref(r)
    check_certificate(p['certificate'])
    if p['model']!=signature():raise RuntimeError('Treatment differs')
    return p


def inputs(scene):
    if scene['loader']=='expanded':images,allowed,_=load_scene(scene['source'])
    elif scene['loader']=='initial':images,allowed,_=load_area(scene['source'])
    else:
        images,allowed,target=load_case(scene['source'])
        if target!=scene['target']:raise ValueError('Wrong target')
    return images,allowed


def texture_for_scoring(scene,truth):
    if scene['loader']=='expanded':return load_evidence(scene['source']['scoring_only_strata'])['textured']
    gray=cv2.cvtColor(truth,cv2.COLOR_RGB2GRAY).astype(np.float32);x=cv2.Sobel(gray,cv2.CV_32F,1,0,ksize=3,scale=1/8);y=cv2.Sobel(gray,cv2.CV_32F,0,1,ksize=3,scale=1/8)
    return (x*x+y*y)>=16


def prediction(e):
    out=np.zeros((*e['mask'].shape,3),np.uint8);out[e['mask']]=e['prediction_on_mask'];return out


def run(args):
    plan=check_plan();sig=dict(plan_sha256=digest(OUT/'execution_plan.json'),certificate=plan['certificate']);completed=0
    with heavy_job():
        verify_installed();matcher=None
        for scene in plan['scenes']:
            folder=OUT/scene['scene'];groups=folder/'groups';groups.mkdir(parents=True,exist_ok=True);images=None;texture=None;weights=None
            for job in scene['jobs']:
                dest=groups/(job['key']+'.json')
                if dest.exists():
                    old=json.loads(dest.read_text())
                    if old['signature']!=sig:raise RuntimeError('Saved run signature differs')
                    for m in old['methods'].values():read_ref(m['evidence'])
                    continue
                if completed>=args.limit:return
                require_storage(ROOT);start=time.monotonic()
                if images is None:images,allowed=inputs(scene)
                mask=load_evidence(job['mask'])[job['mask_key']];natural=job['kind']=='natural';target=scene['target']
                if ((mask&allowed[target]).any() if natural else (mask&~allowed[target]).any()):raise ValueError('Truth visibility failure')
                clean,als=masked_inputs(images,allowed,mask,target)
                if matcher is None:matcher=SamplingMatcher(args.weights,plan['model']['checkpoint_sha256'],root=ROOT)
                def match(i,j,a,b,ma,mb):return matcher.correspondences(a,b,ma,mb)
                regs,graphs,pairs=register_sequence(clean,als,target,scene['projection']['fov'],job['key'],OUT/'pairs'/scene['cache_group'],match,plan['model'])
                # No clean target or texture is passed to the matcher, graph fit or donor ranking.
                truth=None if natural else images[target]
                if weights is None:weights=solid_angles(**scene['projection'])
                if not natural and texture is None:texture=texture_for_scoring(scene,truth)
                control=json.loads(read_ref(job['control']).read_text());result=dict(scene=scene['scene'],area=scene['area'],job=job,signature=sig,graphs=graphs,correspondences=pairs,methods={},created_at_utc=datetime.now(timezone.utc).isoformat())
                for method,r in regs.items():
                    stats,e=score_transfer(clean,als,mask,target,r,plan['policies'],truth);ep=groups/(job['key']+'_'+method+'.npz');save_evidence(ep,e);pred=prediction(e)
                    old=load_evidence(control['methods'][method]['evidence']);oldpred=prediction(old)
                    extra={n:extended_metrics(pred,truth,policy_mask(e,mask,p),mask,weights,None if natural else texture) for n,p in plan['policies'].items()}
                    oldextra={n:extended_metrics(oldpred,truth,policy_mask(old,mask,p),mask,weights,None if natural else texture) for n,p in plan['policies'].items()}
                    result['methods'][method]=dict(policies=stats,extended=extra,control_policies=control['methods'][method]['policies'],control_extended=oldextra,partitions=partitions(e,old,plan['policies'],natural),evidence=ref(ep),registrations=[v.diagnostics for v in r.values()])
                    del e,old,pred,oldpred;gc.collect()
                result['runtime_seconds']=time.monotonic()-start;immutable_json(dest,result);completed+=1
                print('GROUP',scene['scene'],job['key'],{m:dict(coverage=d['policies']['candidate']['coverage'],mae=d['policies']['candidate'].get('error',{}).get('mae_255')) for m,d in result['methods'].items()},flush=True)
                del clean,als,regs;gc.collect()
            folder.joinpath('conditions').mkdir(exist_ok=True)
            for condition in scene['conditions']:
                dest=folder/'conditions'/(condition['spec']['key']+'.json')
                if dest.exists():
                    if json.loads(dest.read_text())['signature']!=sig:raise RuntimeError('Condition signature differs')
                    continue
                require_storage(ROOT)
                if images is None:images,allowed=inputs(scene)
                gp=groups/(condition['job']+'.json');g=json.loads(gp.read_text());baseline=json.loads(read_ref(condition['baseline']).read_text());comparisons={}
                for method,data in g['methods'].items():
                    e=load_evidence(data['evidence']);mask=e['mask'];truth=images[scene['target']][mask];comparisons[method]={}
                    for name,p in plan['policies'].items():
                        take=policy_mask(e,mask,p)[mask]
                        comparisons[method][name]={b:same_pixels(e['prediction_on_mask'],load_evidence(v['evidence'])['prediction_on_mask'],truth,take) for b,v in baseline['baselines'].items()}
                    del e
                immutable_json(dest,dict(scene=scene['scene'],condition=condition,signature=sig,geometry=ref(gp),same_pixel_controls=comparisons));gc.collect()
            print('SCENE_COMPLETE',scene['scene'],flush=True)
            if images is not None:del images,allowed
            del texture,weights;gc.collect()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--prepare',action='store_true');p.add_argument('--weights');p.add_argument('--limit',type=int,default=4);args=p.parse_args()
    if not 1<=args.limit<=126:p.error('limit must be1..126')
    if args.prepare:print(json.dumps({k:v for k,v in build_plan().items() if k in ['groups','conditions']}))
    else:
        if not args.weights:p.error('Local weights required')
        run(args)
