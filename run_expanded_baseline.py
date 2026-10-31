"""Frozen six-anchor baseline, bounded/resumable, with separate natural track."""
import argparse,gc,json,time
from pathlib import Path
from datetime import datetime,timezone
import cv2
import numpy as np
from PIL import Image
from benchmark import ROOT,digest,policy_mask
from repair import nadir
from resource_guard import require_storage
from run_dense import SOURCES,model_signature,verify_installed,verify_certificate,heavy_job,read_ref,ref,save_evidence
from dense_experiment import immutable_json,masked_inputs,register_sequence,score_transfer
from controlled_occlusions import damage,baseline_stats
from prepare_area_benchmark import solid_angles
from area_recovery_metrics import extended_metrics,same_pixels

OUT=ROOT/'controlled_v2';RUN=OUT/'baseline_v1'


def load_npz(reference):
    with np.load(read_ref(reference),allow_pickle=False) as z:return {k:z[k] for k in z.files}


def predict(e):
    p=np.zeros((*e['mask'].shape,3),np.uint8);p[e['mask']]=e['prediction_on_mask'];return p


def build_plan():
    protocol=json.loads((OUT/'benchmark_protocol.json').read_text());audit=json.loads((OUT/'annotation_audit_001.json').read_text())
    if audit.get('passed') is not True or audit['protocol_sha256']!=digest(OUT/'benchmark_protocol.json'):raise RuntimeError('Frozen annotation audit required')
    old=json.loads(read_ref(protocol['existing_controlled_v1']).read_text());scenes=[]
    for r in protocol['prepared']:
        m=json.loads(read_ref(r).read_text());a=json.loads(read_ref(m['annotation']).read_text())
        if a['split'] not in ['development','train'] or a['status']!='reviewed_frozen_before_scoring':raise RuntimeError('Invalid annotation split/status')
        frames=[dict(name=n,id=a['frames'][n]['id'],path=a['frames'][n]['image'],sha256=a['frames'][n]['sample_sha256']) for n in a['frame_order']]
        jobs=[dict(key=s['key'],mask_key='hidden_'+s['key'],kind='controlled',mask_sha256=s['sha256']) for s in m['mask_specs']]
        jobs.append(dict(key='footprint',mask_key='region_footprint',kind='natural'))
        scenes.append(dict(scene=m['scene_id'],area=m['area'],split=m['split'],frames=frames,target=a['frame_order'].index('target'),projection=m['projection'],
            manifest=r,annotation=m['annotation'],masks=m['masks'],scoring_only_strata=m['scoring_only_strata'],jobs=jobs,conditions=m['conditions']))
    sources=list(dict.fromkeys(SOURCES+['controlled_occlusions.py','prepare_area_benchmark.py','prepare_annotated_case_v2.py','area_recovery_metrics.py',
        'run_expanded_baseline.py','audit_expanded_baseline.py','summarize_expanded_baseline.py','test_expanded_baseline.py',
        'validate_dense_run_v2.py','audit_area_annotations.py','audit_controlled_benchmark.py']))
    plan=dict(version=1,status='frozen_before_scoring',signature={n:digest(ROOT/n) for n in sources},
        protocol=ref(OUT/'benchmark_protocol.json'),annotation_audit=ref(OUT/'annotation_audit_001.json'),candidate_graph=protocol['graph'],
        model=model_signature(),policies=old['policies'],scenes=scenes,controlled_masks=72,natural_masks=6,conditions=144,
        comparison_controls=dict(previous_three_anchor_battery=ref(ROOT/'controlled_v1/summary.json'),previous_control_audit=ref(ROOT/'controlled_v1/audit_001.json'),
            alsace_footprint=ref(ROOT/'sequence_v6/results/alsace_fields_0/footprint.json')),
        methods=['loftr_homography','loftr_metric_free'],baselines=['as_damaged','telea_r3'],
        input_isolation='Only sanitized RGB/allowed masks enter registration. Evaluation RGB and scoring-only texture strata enter after registration and never choose donors.',
        natural_track='Source support/abstention only; no clean target truth. Five annotated source images per query.',
        correlation='Disc/blur/clipping/JPEG sharing a mask reuse the same donor prediction; conditions/pixels are not independent trials.',
        promotion='No automatic promotion; preserve zero-coverage scenes, low-texture limitations, angular mismatch and same-pixel controls.')
    immutable_json(RUN/'execution_plan.json',plan);return plan


def check_plan():
    plan=json.loads((RUN/'execution_plan.json').read_text())
    for p,h in plan['signature'].items():read_ref(dict(path=p,sha256=h))
    for key in ['protocol','annotation_audit','candidate_graph']:read_ref(plan[key])
    for r in plan['comparison_controls'].values():read_ref(r)
    return plan


def load_scene(scene):
    read_ref(scene['manifest']);read_ref(scene['annotation']);parts=load_npz(scene['masks'])
    images=[nadir(np.asarray(Image.open(read_ref(r)).convert('RGB')),**scene['projection']) for r in scene['frames']]
    allowed=[parts[r['name']+'_allowed'] for r in scene['frames']]
    return images,allowed,parts


def run(args):
    plan=check_plan();cert=ROOT/'sequence_v6/model_validation_001.json';signature=dict(plan_sha256=digest(RUN/'execution_plan.json'),certificate=ref(cert))
    with heavy_job():
        verify_installed();verify_certificate(cert)
        from dense_matching import MaskedLoFTR
        matcher=None;done=0
        for scene in plan['scenes']:
            folder=RUN/scene['scene'];groups=folder/'groups';groups.mkdir(parents=True,exist_ok=True);images=None;texture=None;weights=None
            for job in scene['jobs']:
                path=groups/(job['key']+'.json')
                if path.exists():
                    prior=json.loads(path.read_text())
                    if prior['signature']!=signature:raise RuntimeError('Existing signature differs')
                    for data in prior['methods'].values():read_ref(data['evidence'])
                    continue
                if done>=args.limit:return
                require_storage(ROOT);start=time.monotonic()
                if images is None:images,allowed,parts=load_scene(scene)
                mask=parts[job['mask_key']];target=scene['target'];natural=job['kind']=='natural'
                visibility_violated=(mask&allowed[target]).any() if natural else (mask&~allowed[target]).any()
                if visibility_violated:raise ValueError('Truth visibility violated')
                clean,als=masked_inputs(images,allowed,mask,target)
                if matcher is None:matcher=MaskedLoFTR(args.weights,plan['model']['checkpoint_sha256'],root=ROOT)
                def match(i,j,a,b,ma,mb):return matcher.correspondences(a,b,ma,mb)
                regs,graphs,pairs=register_sequence(clean,als,target,scene['projection']['fov'],job['key'],folder/'pairs',match,plan['model'])
                # Scoring inputs are accessed only after the correspondence/pose stage.
                if weights is None:weights=solid_angles(**scene['projection'])
                if not natural and texture is None:texture=load_npz(scene['scoring_only_strata'])['textured']
                truth=None if natural else images[target]
                result=dict(scene=scene['scene'],area=scene['area'],job=job,signature=signature,graphs=graphs,correspondences=pairs,methods={},created_at_utc=datetime.now(timezone.utc).isoformat())
                for method,r in regs.items():
                    stats,e=score_transfer(clean,als,mask,target,r,plan['policies'],truth);ep=groups/(job['key']+'_'+method+'.npz');save_evidence(ep,e);pred=predict(e)
                    extras={name:extended_metrics(pred,truth,policy_mask(e,mask,p),mask,weights,None if natural else texture) for name,p in plan['policies'].items()}
                    result['methods'][method]=dict(policies=stats,extended=extras,registrations=[v.diagnostics for v in r.values()],evidence=ref(ep))
                    del e,pred;gc.collect()
                result['runtime_seconds']=time.monotonic()-start;immutable_json(path,result);done+=1
                print('GROUP',scene['scene'],job['key'],{m:dict(coverage=d['policies']['candidate']['coverage'],mae=d['policies']['candidate'].get('error',{}).get('mae_255')) for m,d in result['methods'].items()},flush=True)
                del regs,clean,als;gc.collect()
            dest=folder/'conditions';dest.mkdir(exist_ok=True)
            for spec in scene['conditions']:
                path=dest/(spec['key']+'.json')
                if path.exists():
                    if json.loads(path.read_text())['signature']!=signature:raise RuntimeError('Condition signature differs')
                    continue
                require_storage(ROOT)
                if images is None:images,allowed,parts=load_scene(scene)
                if weights is None:weights=solid_angles(**scene['projection'])
                if texture is None:texture=load_npz(scene['scoring_only_strata'])['textured']
                mask=parts[spec['mask_key']];truth=images[scene['target']];damaged=damage(truth,mask,spec['family'],spec['severity']);blank=damaged.copy();blank[mask]=0
                controls=dict(as_damaged=damaged,telea_r3=cv2.inpaint(blank,mask.astype(np.uint8),3,cv2.INPAINT_TELEA));baselines={}
                for name,pred in controls.items():
                    ep=dest/(spec['key']+'_'+name+'.npz');save_evidence(ep,dict(prediction_on_mask=pred[mask]))
                    baselines[name]=dict(metrics=baseline_stats(pred,truth,mask),extended=extended_metrics(pred,truth,mask,mask,weights,texture),evidence=ref(ep))
                gp=groups/(spec['mask_key'].removeprefix('hidden_')+'.json');geometry=json.loads(gp.read_text());comparisons={}
                for method,data in geometry['methods'].items():
                    e=load_npz(data['evidence']);pred=predict(e);comparisons[method]={}
                    for name,p in plan['policies'].items():
                        take=policy_mask(e,mask,p);comparisons[method][name]={control:same_pixels(pred,value,truth,take) for control,value in controls.items()}
                    del e,pred
                immutable_json(path,dict(scene=scene['scene'],area=scene['area'],condition=spec,signature=signature,geometry=ref(gp),baselines=baselines,same_pixel_controls=comparisons))
                del damaged,blank,controls;gc.collect()
            print('SCENE_COMPLETE',scene['scene'],flush=True)
            if images is not None:del images,parts
            del texture,weights;gc.collect()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--prepare',action='store_true');p.add_argument('--weights');p.add_argument('--limit',type=int,default=3);args=p.parse_args()
    if not 1<=args.limit<=78:p.error('limit must be1..78')
    if args.prepare:
        plan=build_plan();print(json.dumps(dict(scenes=len(plan['scenes']),controlled_masks=72,natural_masks=6,conditions=144,plan_sha256=digest(RUN/'execution_plan.json'))))
    else:
        if not args.weights:p.error('Explicit local weights required')
        run(args)
