"""Resumable acquisition restricted to the frozen cohort and admitted routes.

Dry-run reads metadata only. Real acquisition retains the original rate limits,
license checks, image cache cap, provenance and three-attempt failure records.
"""
import argparse
import fcntl
import hashlib
import json
from pathlib import Path
from dataset_admission import ROOT, digest, effective_splits
from resource_guard import admission, require_storage


def pending_targets(data, cohort, split, limit):
    if limit<1:
        raise ValueError('Positive acquisition limit required')
    data=Path(data); effective=effective_splits(data)
    assert digest(data/'splits_v1.json')==cohort['base_sha256']
    for entry in cohort['admission_ledgers']:
        assert digest(data/'admissions'/entry['name'])==entry['sha256']
    quarantine=set(effective['quarantined_collections'])
    existing=set()
    for p in (data/'windows').glob('*.json'):
        w=json.loads(p.read_text());c=w['collection']
        if c in effective['effective_groups']:
            existing.add(effective['effective_groups'][c])
    candidates=[]
    for t in cohort['targets']:
        c=t['collection']
        if t['split']!=split or c in quarantine:
            continue
        assert effective['collections'][c]['split']==split
        path=data/'windows'/(c+'.json')
        if path.exists():
            w=json.loads(path.read_text())
            assert w['target_id']==t['id'], 'Cached target differs from frozen cohort'
            # Keep incomplete sequences explicit. Repair requires a separate
            # bounded task; never repeatedly count a singleton as new success.
            continue
        failure=data/'failures'/(c+'.json')
        if failure.exists() and json.loads(failure.read_text()).get('attempts',0)>=3:
            continue
        assert digest(data/t['metadata_path'])==t['metadata_sha256']
        candidates.append(dict(t,group=effective['effective_groups'][c]))
    candidates.sort(key=lambda t:(t['group'] in existing, hashlib.sha256(t['group'].encode()).hexdigest(),t['collection']))
    selected=[];groups=set()
    for t in candidates:
        if t['group'] in groups:
            continue
        groups.add(t['group']);selected.append(t)
        if len(selected)==limit:
            break
    return selected


def acquire(args):
    data=Path(args.data);cohort=json.loads(Path(args.cohort).read_text())
    targets=pending_targets(data,cohort,args.split,args.limit)
    resources=admission(data.parent,reserve_bytes=512*1024**2)
    if args.dry_run:
        result=dict(dry_run=True, split=args.split, targets=targets, resources=resources,
                    images_downloaded=0,not_a_completion_count=True)
        print(json.dumps(result,indent=2));return result
    require_storage(data.parent)
    import dataset
    if dataset.DATA.resolve()!=data.resolve():
        raise ValueError('Actual acquisition is restricted to the project catalog')
    with (data/'.acquisition.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        from audit_effective_dataset import audit_current
        audit_current(data)
        db=dataset.connection()
        for t in targets:
            require_storage(data.parent)
            c=t['collection'];failure=data/'failures'/(c+'.json')
            attempts=json.loads(failure.read_text()).get('attempts',0) if failure.exists() else 0
            try:
                dataset.window(db,t['id'],2,c);failure.unlink(missing_ok=True)
            except Exception as e:
                dataset.save(failure,dict(error=str(e),target=t,at=dataset.now(),attempts=attempts+1))
                print('FAILED',c,str(e),flush=True)
            # Neighbor metadata can reveal a bridge to another frozen split.
            # Stop immediately after this window if current observations fail.
            audit_current(data)
        dataset.summary(db)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',default=str(ROOT/'dataset_v1'))
    p.add_argument('--cohort',default=str(ROOT/'dataset_v1/cohorts/initial_100_v1.json'))
    p.add_argument('--split',choices=['train','validation','test'],default='train')
    p.add_argument('--limit',type=int,default=5);p.add_argument('--dry-run',action='store_true')
    acquire(p.parse_args())
