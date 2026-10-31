"""Status/resume entry point for the frozen baseline. Does not tune experiments.

With --advance it executes at most --limit missing geometry masks, then audits
only once the full battery exists. Existing immutable completed work is reused.
"""
import argparse,fcntl,json,os,subprocess,sys
from pathlib import Path
from datetime import datetime,timezone
from benchmark import ROOT,save_json
from resource_guard import admission
from run_dense import heavy_job

OUT=ROOT/'controlled_v1'


def status():
    jobs=[]
    names={'acquire_area_sets.py','run_controlled_benchmark.py','audit_controlled_benchmark.py','controlled_workflow.py','evaluate_natural_anchors.py','compare_controlled_outputs.py',
        'preview_area_annotations.py','prepare_area_benchmark.py','audit_area_annotations.py',
        'run_expanded_baseline.py','audit_expanded_baseline.py','validate_sampling_model.py','run_sampling.py','audit_sampling.py','run_solver_budget.py','audit_solver_budget.py',
        'run_dense.py','run_long_sequence.py','validate_dense_model.py','run_annotated.py','run_metric.py','run_constrained.py','run_sequence.py'}
    for line in subprocess.check_output(['ps','-ax','-o','pid=,command='],text=True).splitlines():
        parts=line.split()
        if len(parts)<3 or int(parts[0])==os.getpid() or 'python' not in Path(parts[1]).name.lower():continue
        if any(Path(p).name in names for p in parts[2:]):jobs.append(dict(pid=int(parts[0]),command=' '.join(parts[1:])))
    pp=OUT/'execution_plan.json';plan=json.loads(pp.read_text());areas={}
    for a in plan['areas']:
        folder=OUT/a['area'];areas[a['area']]=dict(geometry_masks_complete=sum((folder/'groups'/(s['key']+'.json')).exists() for s in a['mask_specs']),
            geometry_masks_planned=len(a['mask_specs']),scenarios_complete=sum((folder/'scenarios'/(s['key']+'.json')).exists() for s in a['scenarios']),scenarios_planned=len(a['scenarios']))
    complete=all(a['geometry_masks_complete']==a['geometry_masks_planned'] and a['scenarios_complete']==a['scenarios_planned'] for a in areas.values())
    audit=json.loads((OUT/'audit_001.json').read_text()) if (OUT/'audit_001.json').exists() else None
    resources=admission(ROOT,reserve_bytes=512*1024**2)
    lock_busy=False
    with (ROOT/'autoresearch/heavy_job.lock').open('a') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:lock_busy=True
    if jobs or lock_busy:resources.update(safe_to_start_heavy_job=False,safe_to_start_acquisition=False)
    result=dict(observed_at_utc=datetime.now(timezone.utc).isoformat(),areas=areas,running_jobs=jobs,resources=resources,
        results_complete=complete,independent_audit_passed=bool(audit and audit.get('passed') is True),summary_complete=(OUT/'summary.json').exists(),heavy_lock_busy=lock_busy)
    expansion=ROOT/'controlled_v2/annotation_audit_001.json'
    if expansion.exists():
        e=json.loads(expansion.read_text());result['annotation_expansion']=dict(anchors=e['anchors'],source_frames=e['reviewed_source_frames'],conditions=e['conditions'],audit_passed=e['passed'],model_runs_at_annotation_freeze=e['new_recovery_runs'])
    expanded=ROOT/'controlled_v2/baseline_v1';ep=expanded/'execution_plan.json'
    if ep.exists():
        p=json.loads(ep.read_text());scenes={}
        for scene in p['scenes']:
            folder=expanded/scene['scene'];scenes[scene['scene']]=dict(
                geometry_groups=sum((folder/'groups'/(j['key']+'.json')).exists() for j in scene['jobs']),planned_groups=len(scene['jobs']),
                conditions=sum((folder/'conditions'/(c['key']+'.json')).exists() for c in scene['conditions']),planned_conditions=len(scene['conditions']))
        ea=json.loads((expanded/'audit_001.json').read_text()) if (expanded/'audit_001.json').exists() else {}
        result['expanded_baseline']=dict(scenes=scenes,results_complete=all(v['geometry_groups']==v['planned_groups'] and v['conditions']==v['planned_conditions'] for v in scenes.values()),
            independent_audit_passed=ea.get('passed') is True,summary_complete=(expanded/'summary.json').exists())
    sampling=ROOT/'sampling_v1';sp=sampling/'execution_plan.json'
    if sp.exists():
        p=json.loads(sp.read_text());groups=sum((sampling/s['scene']/'groups'/(j['key']+'.json')).exists() for s in p['scenes'] for j in s['jobs'])
        conditions=sum((sampling/s['scene']/'conditions'/(c['spec']['key']+'.json')).exists() for s in p['scenes'] for c in s['conditions'])
        sa=json.loads((sampling/'audit_001.json').read_text()) if (sampling/'audit_001.json').exists() else {}
        result['sampling_ablation']=dict(groups=groups,planned_groups=p['groups'],conditions=conditions,planned_conditions=p['conditions'],results_complete=groups==p['groups'] and conditions==p['conditions'],
            independent_audit_passed=sa.get('passed') is True,summary_complete=(sampling/'summary.json').exists())
    solver=ROOT/'solver_budget_v1';bp=solver/'execution_plan.json'
    if bp.exists():
        p=json.loads(bp.read_text());selected=[r for r in p['groups'] if r['changed']];conditions=[r for r in p['conditions'] if r['changed']]
        ng=sum((solver/r['scene']/'groups'/(r['job']+'.json')).exists() for r in selected)
        nc=sum((solver/r['scene']/'conditions'/(r['condition']+'.json')).exists() for r in conditions)
        sa=json.loads((solver/'audit_001.json').read_text()) if (solver/'audit_001.json').exists() else {}
        result['solver_budget_ablation']=dict(changed_groups=ng,planned_changed_groups=len(selected),changed_conditions=nc,planned_changed_conditions=len(conditions),
            results_complete=ng==len(selected) and nc==len(conditions) and (solver/'registry.json').exists(),independent_audit_passed=sa.get('passed') is True,summary_complete=(solver/'summary.json').exists())
    save_json(OUT/'status.json',result);return result


def advance(limit):
    s=status()
    if s['running_jobs'] or not s['resources']['safe_to_start_heavy_job']:return s
    env=dict(os.environ,PYTORCH_ENABLE_MPS_FALLBACK='0',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1')
    if not s['results_complete']:
        log=OUT/('resume_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')+'.log')
        with log.open('x') as f:
            subprocess.run([sys.executable,str(ROOT/'run_controlled_benchmark.py'),'--weights',str(ROOT.parents[1]/'work/model-cache/loftr_outdoor_21f5bec59681.ckpt'),'--limit',str(limit)],env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
        s=status()
    if s['results_complete'] and not s['independent_audit_passed']:
        with heavy_job():subprocess.run([sys.executable,str(ROOT/'audit_controlled_benchmark.py')],env=env,check=True)
        s=status()
    if s['independent_audit_passed'] and not s['summary_complete']:
        subprocess.run([sys.executable,str(ROOT/'summarize_controlled_benchmark.py')],env=env,check=True)
    s=status();expanded=s.get('expanded_baseline')
    if expanded and not expanded['results_complete']:
        folder=ROOT/'controlled_v2/baseline_v1';log=folder/('resume_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')+'.log')
        with log.open('x') as f:
            subprocess.run([sys.executable,str(ROOT/'run_expanded_baseline.py'),'--weights',str(ROOT.parents[1]/'work/model-cache/loftr_outdoor_21f5bec59681.ckpt'),'--limit',str(limit)],env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
        expanded=status()['expanded_baseline']
    if expanded and expanded['results_complete'] and not expanded['independent_audit_passed']:
        subprocess.run([sys.executable,str(ROOT/'audit_expanded_baseline.py')],env=env,check=True)
        expanded=status()['expanded_baseline']
    if expanded and expanded['independent_audit_passed'] and not expanded['summary_complete']:
        subprocess.run([sys.executable,str(ROOT/'summarize_expanded_baseline.py')],env=env,check=True)
    sampling=status().get('sampling_ablation')
    if sampling and not sampling['results_complete']:
        folder=ROOT/'sampling_v1';log=folder/('resume_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')+'.log')
        with log.open('x') as f:
            subprocess.run([sys.executable,str(ROOT/'run_sampling.py'),'--weights',str(ROOT.parents[1]/'work/model-cache/loftr_outdoor_21f5bec59681.ckpt'),'--limit',str(limit)],env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
        sampling=status()['sampling_ablation']
    if sampling and sampling['results_complete'] and not sampling['independent_audit_passed']:
        subprocess.run([sys.executable,str(ROOT/'audit_sampling.py')],env=env,check=True)
        sampling=status()['sampling_ablation']
    if sampling and sampling['independent_audit_passed'] and not sampling['summary_complete']:
        subprocess.run([sys.executable,str(ROOT/'summarize_sampling.py')],env=env,check=True)
    solver=status().get('solver_budget_ablation')
    if solver and not solver['results_complete']:
        folder=ROOT/'solver_budget_v1';log=folder/('resume_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')+'.log')
        for row in json.loads((folder/'execution_plan.json').read_text())['groups']:
            if row['changed']:(folder/row['scene']/'groups').mkdir(parents=True,exist_ok=True)
        with log.open('x') as f:
            subprocess.run([sys.executable,str(ROOT/'run_solver_budget.py'),'--limit',str(min(limit,6))],env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
        solver=status()['solver_budget_ablation']
    if solver and solver['results_complete'] and not solver['independent_audit_passed']:
        subprocess.run([sys.executable,str(ROOT/'audit_solver_budget.py')],env=env,check=True)
        solver=status()['solver_budget_ablation']
    if solver and solver['independent_audit_passed'] and not solver['summary_complete']:
        subprocess.run([sys.executable,str(ROOT/'summarize_solver_budget.py')],env=env,check=True)
    return status()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--advance',action='store_true');p.add_argument('--limit',type=int,default=3);args=p.parse_args()
    if not 1<=args.limit<=36:p.error('limit must be1..36')
    print(json.dumps(advance(args.limit) if args.advance else status(),indent=2))
