"""Compact, live acquisition status; image counts require a matching checksum."""
import json,os,subprocess
from pathlib import Path
from datetime import datetime,timezone
from benchmark import ROOT,digest,save_json
from resource_guard import admission


def snapshot():
    out=ROOT/'area_sets_v1';areas={};jobs=[]
    for line in subprocess.check_output(['ps','-ax','-o','pid=,command='],text=True).splitlines():
        parts=line.split()
        if len(parts)<3 or int(parts[0])==os.getpid() or 'python' not in Path(parts[1]).name.lower():continue
        if any(Path(p).name in ['acquire_area_sets.py','audit_area_sets.py'] for p in parts[2:]):jobs.append(dict(pid=int(parts[0]),command=' '.join(parts[1:])))
    for name in ['gironde','nantes','bourges']:
        path=out/(name+'_plan.json')
        if not path.exists():areas[name]=dict(status='metadata_pending');continue
        plan=json.loads(path.read_text());cached=[];failures=[]
        for r in plan['frames']:
            image=ROOT/'dataset_v1/images'/(r['id']+'.jpg');record=image.with_suffix('.json')
            if not image.exists() or not record.exists():continue
            expected=json.loads(record.read_text())['sample_sha256']
            if digest(image)==expected:cached.append(r['id'])
            else:failures.append(r['id'])
        manifest=out/(name+'_manifest.json');complete=manifest.exists()
        areas[name]=dict(planned=len(plan['frames']),verified_cached=len(cached),missing=len(plan['frames'])-len(cached),
            checksum_failures=failures,manifest_complete=complete,full_collection=plan['full_collection'],
            path_length_m=plan['path_length_m'],split=plan['selection']['split'],
            source_collection=json.loads((out/(name+'_collection.json')).read_text())['stats:items']['count'],
            viewer_url='https://explore.panoramax.fr/?pic='+plan['selection']['target']+'&focus=pic&xyz=0/-90/0')
    resources=admission(ROOT,reserve_bytes=512*1024**2)
    if jobs:
        resources.update(safe_to_start_heavy_job=False,safe_to_start_acquisition=False)
        resources['blockers'].append('An area acquisition or audit job is already running; reuse it')
    return dict(observed_at_utc=datetime.now(timezone.utc).isoformat(),areas=areas,running_jobs=jobs,
        resources=resources,planned_total=sum(a.get('planned',0) for a in areas.values()),
        verified_cached_total=sum(a.get('verified_cached',0) for a in areas.values()),
        all_manifests_complete=all(a.get('manifest_complete',False) for a in areas.values()),
        independent_area_audit_complete=(out/'audit_001.json').exists(),source_sha256=digest(Path(__file__)))


if __name__=='__main__':
    status=snapshot();save_json(ROOT/'area_sets_v1/status.json',status);print(json.dumps(status,indent=2))
