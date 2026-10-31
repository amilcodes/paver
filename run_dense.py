"""Bounded, resumable LoFTR experiment. Planning never loads pixels or weights."""
import argparse,fcntl,gc,json,os,subprocess,time
from contextlib import contextmanager
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
from PIL import Image
from benchmark import ROOT,digest,masks,policy_mask
from repair import nadir
from resource_guard import require_storage
from dense_experiment import immutable_json,masked_inputs,register_sequence,score_transfer,same_pixel_error

SOURCES=['run_dense.py','dense_experiment.py','dense_matching.py','validate_dense_model.py',
         'sequence_geometry.py','constrained_sequence.py','metric_plane.py','trajectory_matching.py',
         'matching_v2.py','repair.py','benchmark.py','resource_guard.py']


def ref(path):
    path=Path(path);return dict(path=str(path.relative_to(ROOT)),sha256=digest(path))


def read_ref(record):
    path=ROOT/record['path']
    if digest(path)!=record['sha256']:raise RuntimeError(f'Input hash changed: {path}')
    return path


def model_signature():
    prep=json.loads((ROOT/'sequence_v6/model_preparation.json').read_text())
    return dict(checkpoint_sha256=prep['outdoor_file']['lfs']['sha256'],
                adapter_sha256=digest(ROOT/'dense_matching.py'),packages=prep['packages'],
                installed_sources=prep['installed_implementation_sources'])


def verify_installed():
    from importlib.metadata import version
    sig=model_signature();workspace=ROOT.parents[1]
    for name,expected in sig['packages'].items():
        if version(name)!=expected:raise RuntimeError(f'Installed package changed: {name}')
    for name,expected in sig['installed_sources'].items():
        if digest(workspace/name)!=expected:raise RuntimeError(f'Installed model source changed: {name}')


def certificate_signature():
    return dict(model=model_signature(),validator_sha256=digest(ROOT/'validate_dense_model.py'),
                thresholds=dict(minimum_matches=32,median_error_px=1.5,p90_error_px=3.0),
                device='mps')


def verify_certificate(path):
    c=json.loads(Path(path).read_text())
    if c.get('passed') is not True or c.get('signature')!=certificate_signature():
        raise RuntimeError('Missing, unsuccessful or stale actual-model certificate')
    if not c.get('actual_model_executed') or not c.get('excluded_rgb_correspondence_invariance'):
        raise RuntimeError('Preprocessing-only validation cannot authorize model experiment')
    if c.get('translation_matches',0)<32 or c.get('translation_median_error_px',float('inf'))>=1.5 or c.get('translation_p90_error_px',float('inf'))>=3:
        raise RuntimeError('Actual-model translation validation failed')


@contextmanager
def heavy_job():
    require_storage(ROOT)
    # Check older runners that predate this shared lock, including absolute paths.
    names={'run_dense.py','validate_dense_model.py','run_annotated.py','run_metric.py',
           'run_constrained.py','run_sequence.py','run_trajectory.py','run_long_sequence.py','preflight_dense.py'}
    for line in subprocess.check_output(['ps','-ax','-o','pid=,command='],text=True).splitlines():
        parts=line.split()
        if len(parts)<3 or int(parts[0])==os.getpid() or 'python' not in Path(parts[1]).name.lower():continue
        if any(Path(p).name in names for p in parts[2:]):raise RuntimeError('Another heavy research process is active: '+line.strip())
    with (ROOT/'autoresearch/heavy_job.lock').open('a') as handle:
        fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
        yield


def build_plan():
    protocol=json.loads((ROOT/'sequence_v6/protocol.json').read_text())
    config=json.loads((ROOT/'benchmark_v2/protocol.json').read_text());cases=[]
    for case in protocol['cases']:
        if case=='bourges_turn_001':
            annotation=ROOT/'benchmark_v3/annotations_train_001.json';a=json.loads(annotation.read_text())
            prepared=ROOT/'benchmark_v3/prepared'/case;manifest=json.loads((prepared/'manifest.json').read_text())
            if manifest['annotation_sha256']!=digest(annotation) or manifest['mask_sha256']!=digest(prepared/'masks.npz'):
                raise RuntimeError('Frozen annotation changed')
            if a['status']!='reviewed_frozen_before_scoring' or a['split']!='train':raise RuntimeError('Not a frozen train scene')
            names=a['frame_order'];images=[dict(name=n,id=a['frames'][n]['id'],**ref(ROOT/a['frames'][n]['image'])) for n in names]
            for r,n in zip(images,names):
                if r['sha256']!=a['frames'][n]['sample_sha256']:raise RuntimeError('Source image changed')
            regions=['footprint',*a['regions']];controls=ROOT/'benchmark_v3/results'/case
            methods=['joint_homography','metric_free'];projection=a['projection']
            inputs=[ref(annotation),ref(prepared/'manifest.json'),ref(prepared/'masks.npz'),ref(ROOT/'dataset_v1/splits_v1.json')]
            split=json.loads((ROOT/'dataset_v1/splits_v1.json').read_text())['collections'][a['collection']]['split']
            if split!='train':raise RuntimeError('Train assignment changed')
        else:
            data=ROOT/'benchmark_v2/data'/case;manifest_path=data/'manifest.json';manifest=json.loads(manifest_path.read_text())
            names=json.loads((ROOT/'sequence_v1/protocol.json').read_text())['frames'];records={r['name']:r for r in manifest['images']}
            images=[dict(name=n,id=records[n]['id'],**ref(data/(n+'.jpg'))) for n in names]
            for r in images:
                if r['sha256']!=records[r['name']]['sample_sha256']:raise RuntimeError('Source image changed')
            regions=['footprint','hidden_front','hidden_rear','local_front','local_rear'];methods=['homography_converged','metric_free']
            controls=ROOT/'sequence_v5/results/sequence_v1'/case;projection={k:config[k] for k in ['size','fov']}
            inputs=[ref(manifest_path),ref(ROOT/'sequence_v1/protocol.json')];split='development'
        if len(images)!=5 or len({r['id'] for r in images})!=5:raise RuntimeError('Need five distinct frozen source images')
        jobs=[]
        for region in regions:
            source=controls/(region+'.json');record=json.loads(source.read_text())
            if record['case']!=case or record['region']!=region:raise RuntimeError('Wrong control identity')
            control={m:dict(record=ref(source),evidence=ref(controls/(region+'_'+m+'_evidence.npz'))) for m in methods}
            mask=ref(ROOT/'benchmark_v2/results'/case/'disk_lightglue'/(region+'_evidence.npz')) if split=='development' else ref(prepared/'masks.npz')
            jobs.append(dict(region=region,mask=mask,mask_key='mask' if split=='development' else 'region_'+region,controls=control))
        cases.append(dict(case=case,split=split,frames=images,projection=projection,inputs=inputs,jobs=jobs))
    return dict(experiment='sequence_v6_dense_matching',status='execution_plan_not_model_result',
                signature={n:digest(ROOT/n) for n in SOURCES},
                protocols=[ref(ROOT/'sequence_v6/protocol.json'),ref(ROOT/'sequence_v6/execution_protocol.json'),ref(ROOT/'benchmark_v2/protocol.json')],
                model=model_signature(),policies=config['policies'],cases=cases)


def load_case(case):
    for r in case['inputs']:read_ref(r)
    images=[nadir(np.asarray(Image.open(read_ref(r)).convert('RGB')),**case['projection']) for r in case['frames']]
    names=[r['name'] for r in case['frames']]
    if case['split']=='development':
        cfg=json.loads((ROOT/'benchmark_v2/protocol.json').read_text());_,allowed,_=masks(cfg,cfg['cases'][case['case']])
        allowed=[allowed[n] for n in names]
    else:
        with np.load(ROOT/'benchmark_v3/prepared'/case['case']/'masks.npz',allow_pickle=False) as z:
            allowed=[z[n+'_allowed'] for n in names]
    return images,allowed,names.index('target')


def save_evidence(path,evidence):
    if path.exists():
        with np.load(path,allow_pickle=False) as old:
            if set(old.files)!=set(evidence) or any(not np.array_equal(old[k],v,equal_nan=True) for k,v in evidence.items()):
                raise RuntimeError('Existing evidence differs; preserve and version the experiment')
        return
    tmp=path.with_suffix('.npz.tmp')
    with tmp.open('wb') as f:np.savez_compressed(f,**evidence)
    tmp.replace(path)


def run(args,plan):
    with heavy_job():
        verify_installed();verify_certificate(args.certificate)
        signature=dict(plan_sha256=digest(ROOT/'sequence_v6/execution_plan.json'),certificate=ref(Path(args.certificate).resolve()))
        from dense_matching import MaskedLoFTR
        matcher=None;completed=0
        for case in plan['cases']:
            if args.case and case['case'] not in args.case:continue
            images=None
            for job in case['jobs']:
                region=job['region']
                if args.region and region not in args.region:continue
                out=ROOT/'sequence_v6/results'/case['case'];dest=out/(region+'.json')
                if dest.exists():
                    old=json.loads(dest.read_text())
                    if old['signature']!=signature:raise RuntimeError('Immutable run signature changed')
                    for method in old['methods'].values():read_ref(method['evidence'])
                    print('RESUME',case['case'],region,flush=True);continue
                if completed>=args.limit:return
                require_storage(ROOT);start=time.monotonic();out.mkdir(parents=True,exist_ok=True)
                if images is None:images,allowed,target=load_case(case)
                with np.load(read_ref(job['mask']),allow_pickle=False) as z:mask=z[job['mask_key']]
                if region=='footprint':
                    if (mask&allowed[target]).any():raise RuntimeError('Footprint overlaps allowed target')
                elif (mask&~allowed[target]).any():raise RuntimeError('Hidden truth is outside annotated visible ground')
                clean,als=masked_inputs(images,allowed,mask,target)
                if matcher is None:matcher=MaskedLoFTR(args.weights,plan['model']['checkpoint_sha256'],root=ROOT)
                def match(i,j,a,b,ma,mb):return matcher.correspondences(a,b,ma,mb)
                regs,graphs,pairs=register_sequence(clean,als,target,case['projection']['fov'],region,out/'pairs',match,plan['model'])
                result=dict(case=case['case'],split=case['split'],region=region,signature=signature,
                            created_at_utc=datetime.now(timezone.utc).isoformat(),mask_pixels=int(mask.sum()),
                            source_order=[r['name'] for i,r in enumerate(case['frames']) if i!=target],
                            graphs=graphs,correspondences=pairs,methods={},controls={},same_pixel_control={},promotion='none')
                # The original hidden RGB enters only after all registration has finished.
                truth=None if region=='footprint' else images[target]
                for method,r in regs.items():
                    require_storage(ROOT)
                    stats,evidence=score_transfer(clean,als,mask,target,r,plan['policies'],truth)
                    ep=out/(region+'_'+method+'_evidence.npz');save_evidence(ep,evidence)
                    result['methods'][method]=dict(policies=stats,registrations=[v.diagnostics for v in r.values()],evidence=ref(ep))
                    for control,refs in job['controls'].items():
                        old=json.loads(read_ref(refs['record']).read_text())['methods'][control]['policies']
                        result['controls'][control]=dict(references=refs,policies={k:dict(v,abstention=1-v['coverage']) for k,v in old.items()})
                        with np.load(read_ref(refs['evidence']),allow_pickle=False) as z:
                            cached={k:z[k] for k in z.files}
                        if not np.array_equal(mask,cached['mask']):raise RuntimeError('Cached control mask differs')
                        for name,p in plan['policies'].items():
                            if int(policy_mask(cached,mask,p).sum())!=old[name]['accepted_pixels']:raise RuntimeError('Cached control count differs')
                        if truth is not None:result['same_pixel_control'].setdefault(method,{})[control]=same_pixel_error(evidence,cached,plan['policies'])
                        del cached
                    del evidence;gc.collect()
                result['runtime_seconds']=time.monotonic()-start;immutable_json(dest,result);completed+=1
                print('RESULT',case['case'],region,{m:v['policies']['candidate'] for m,v in result['methods'].items()},flush=True)
                del clean,als,regs;gc.collect()
            del images;gc.collect()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',action='store_true');p.add_argument('--case',action='append',choices=['gironde_0','alsace_fields_0','bourges_turn_001']);p.add_argument('--region',action='append')
    p.add_argument('--limit',type=int,default=1);p.add_argument('--weights');p.add_argument('--certificate');args=p.parse_args()
    if not 1<=args.limit<=5:p.error('--limit must be between 1 and 5 regions')
    if args.run and (not args.weights or not args.certificate):p.error('--run requires explicit local --weights and passed --certificate')
    plan=build_plan();immutable_json(ROOT/'sequence_v6/execution_plan.json',plan)
    if args.run:run(args,plan)
    else:print(json.dumps(dict(plan='sequence_v6/execution_plan.json',cases=len(plan['cases']),regions=sum(len(c['jobs']) for c in plan['cases']),model_executed=False)))
