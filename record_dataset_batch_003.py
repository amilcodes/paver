"""Freeze provenance and completion counts for the bounded scaling batch."""
import json
from datetime import datetime, timezone
from pathlib import Path
from benchmark import ROOT, digest
from run_dense import ref
from dense_experiment import immutable_json
from dataset_admission import effective_splits
from resource_guard import admission


def record():
    data=ROOT/'dataset_v1'; out=data/'admission_audits/003'
    selection=json.loads((out/'acquisition_selection_postcommit.json').read_text())
    validation=json.loads((out/'validation.json').read_text())
    effective=effective_splits(data)
    audit=json.loads((data/'effective_audit.json').read_text())
    assert validation['passed'] is True and audit['integrity_pass'] and audit['spatial_split_audit_pass']
    assert audit['ledger_chain']==effective['ledger_chain'] and not audit['quarantine'] and not audit['integrity_failures']
    assert digest(data/'cohorts/initial_100_v1.json')==validation['cohort_sha256']
    cohort=json.loads((data/'cohorts/initial_100_v1.json').read_text())
    windows={r['collection']:r for r in audit['windows']}
    completed=incomplete=pending=0
    for t in cohort['targets']:
        c=t['collection']; w=windows.get(c)
        if w is None: pending+=1
        elif w['complete_window']: completed+=1
        else: incomplete+=1
        if t['split'] in ['validation','test']: assert w is None, 'Held-out imagery touched'
    selected=[]
    for t in selection['targets']:
        c=t['collection']; path=data/'windows'/(c+'.json'); w=json.loads(path.read_text())
        assert effective['collections'][c]['split']=='train' and windows[c]['complete_window'] and windows[c]['images']==5
        assert w['target_id']==t['id'] and all(r['license']=='etalab-2.0' for r in w['images'])
        selected.append(dict(collection=c, split='train', manifest=ref(path), frames=5, license='etalab-2.0'))
    summary=json.loads((data/'summary.json').read_text())
    assert len(selected)==5 and completed==25 and incomplete==1 and pending==74
    immutable_json(out/'effective_audit_snapshot.json',audit)
    immutable_json(out/'catalog_summary_snapshot.json',summary)
    sources=['dataset.py','dataset_admission.py','audit_admission_next.py','audit_admission.py','audit_effective_dataset.py','acquire_admitted.py','cohort.py','resource_guard.py']
    result=dict(status='completed_integrity_and_split_audited',created_at_utc=datetime.now(timezone.utc).isoformat(),source=ref(Path(__file__).resolve()),source_sha256={s:digest(ROOT/s) for s in sources},
                new_catalog_queries=20,new_collections=38,admitted_collections=len(effective['collections']),effective_spatial_groups=len(set(effective['effective_groups'].values())),
                new_images_downloaded=25,new_complete_train_windows=5,selected_complete_windows=completed,selected_incomplete_windows=incomplete,selected_pending_windows=pending,
                windows=selected,validation=ref(out/'validation.json'),proposal=ref(out/'proposal.json'),effective_audit=ref(out/'effective_audit_snapshot.json'),catalog=ref(out/'catalog_summary_snapshot.json'),
                acquisition_selection=ref(out/'acquisition_selection_postcommit.json'),acquisition_log=ref(out/'acquisition_001.log'),catalog_log=ref(data/'catalog_batch_003.log'),
                cohort_sha256=validation['cohort_sha256'],previous_assignments_preserved=315,actual_active_cross_split_edges=validation['actual_active_cross_split_edges'],
                unsafe_independent_assignment_cross_split_edges=validation['unsafe_independent_assignment_cross_split_edges'],quarantined_collections=[],resources=admission(ROOT,reserve_bytes=512*1024**2),
                commands=['dataset.catalog(queries=20,ring=1) under shared heavy/acquisition locks','dataset_admission.py --output <absolute003>',
                          'audit_admission_next.py --proposal <absolute003/proposal.json> --output <absolute003/validation.json>',
                          'dataset_admission.py --output <absolute003> --commit','acquire_admitted.py --dry-run --split train --limit5 (postcommit selection verified)',
                          'acquire_admitted.acquire(split=train,limit=5) under shared heavy lock; acquisition lock, rate limit and current split audit before/after each window'],
                limitations=['Twenty-five selected windows complete; one incomplete and74pending, not100finished sequences.',
                             'The25new training images remain unreviewed, unannotated and unevaluated.', 'Held-out validation/test images remain untouched.',
                             'Catalog spatial separation uses sampled coordinates, not complete route geometry; no independent test performance claim.',
                             'Catalog count353 is a metadata collection count, not353downloaded sequences.'])
    immutable_json(out/'decision.json',result)
    print(json.dumps({k:result[k] for k in ['status','admitted_collections','effective_spatial_groups','selected_complete_windows','selected_incomplete_windows','selected_pending_windows']}))


if __name__=='__main__':record()
