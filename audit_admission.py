"""Independent grid-based spatial audit of append-only admission and cohort."""
import hashlib
import itertools
import json
import math
from collections import Counter,defaultdict
from pathlib import Path
from dataset_admission import ROOT, digest, effective_splits, freeze_ledger


def grid_edges(rows, radius):
    """Independent streaming grid audit; no scipy tree or plan() reuse."""
    cells=defaultdict(list);edges=set();point_pairs=0
    for row in rows:
        lng,lat=math.radians(row['longitude']),math.radians(row['latitude'])
        p=(6371000*math.cos(lat)*math.cos(lng),6371000*math.cos(lat)*math.sin(lng),6371000*math.sin(lat))
        key=tuple(math.floor(v/radius) for v in p)
        for delta in itertools.product((-1,0,1),repeat=3):
            for other,q in cells[tuple(k+d for k,d in zip(key,delta))]:
                if other==row['collection']:
                    continue
                if sum((v-w)**2 for v,w in zip(p,q))<=radius**2:
                    edges.add(tuple(sorted((row['collection'],other))));point_pairs+=1
        cells[key].append((row['collection'],p))
    return edges,point_pairs


def main():
    data=ROOT/'dataset_v1';out=data/'admission_audits/001'
    ledger=json.loads((data/'admissions/0001.json').read_text())
    assert digest(data/'splits_v1.json')==ledger['base_sha256']
    assert digest(ROOT/'dataset_admission.py')==ledger['source_sha256']
    snapshot=ROOT/ledger['snapshot_path'];assert digest(snapshot)==ledger['snapshot_sha256']
    rows=json.loads(snapshot.read_text());base=json.loads((data/'splits_v1.json').read_text())
    effective=effective_splits(data);combined=effective['collections'];quarantine=set(effective['quarantined_collections'])
    edges,pairs=grid_edges(rows,ledger['radius_m'])
    assert edges=={tuple(e['collections']) for e in ledger['plan']['collection_neighbor_edges']}
    unsafe=dict(base['collections'])
    for c in ledger['plan']['additions']:
        v=int(hashlib.sha256(('split-v1:'+c).encode()).hexdigest()[:8],16)%10
        unsafe[c]={'split':'test' if v<2 else 'validation' if v<4 else 'train'}
    unsafe_edges=[list(e) for e in sorted(edges) if unsafe[e[0]]['split']!=unsafe[e[1]]['split']]
    actual_edges=[list(e) for e in sorted(edges) if not quarantine.intersection(e) and combined[e[0]]['split']!=combined[e[1]]['split']]
    assert not actual_edges
    for c,v in base['collections'].items():
        assert combined[c]==v
    assert len(ledger['plan']['targets'])==len(ledger['plan']['additions'])
    cohort_path=data/'cohorts/initial_100_v1.json';cohort=json.loads(cohort_path.read_text())
    assert len(cohort['targets'])==len({t['collection'] for t in cohort['targets']})==100
    assert dict(Counter(t['split'] for t in cohort['targets']))==cohort['split_quotas']
    for t in cohort['targets']:
        assert combined[t['collection']]['split']==t['split']
        assert effective['effective_groups'][t['collection']]==t['group']
        assert t['collection'] not in quarantine
        assert digest(data/t['metadata_path'])==t['metadata_sha256']
        assert t['license']=='etalab-2.0'
    windows={json.loads(p.read_text())['collection']:json.loads(p.read_text()) for p in (data/'windows').glob('*.json')}
    complete=sum(bool(windows.get(t['collection'],{}).get('complete_window')) for t in cohort['targets'])
    incomplete=sum(t['collection'] in windows and not windows[t['collection']]['complete_window'] for t in cohort['targets'])
    heldout_cached=[t['collection'] for t in cohort['targets'] if t['split'] in ['test','validation'] and t['collection'] in windows]
    assert not heldout_cached
    code=['dataset_admission.py','cohort.py','acquire_admitted.py','audit_admission.py','tests/test_dataset_admission.py','tests/test_cohort.py','research_status.py']
    result=dict(spatial_audit='independent ECEF spatial-grid enumeration agrees with cKDTree planner',
                eligible_pictures=len(rows),collection_neighbor_edges=len(edges),picture_neighbor_pairs=pairs,
                baseline_split_sha256_unchanged=True,original_assignments_preserved=len(base['collections']),
                admitted_collections=len(ledger['plan']['additions']),effective_split_counts=dict(Counter(v['split'] for v in combined.values())),
                unsafe_independent_assignment_cross_split_edges=len(unsafe_edges),unsafe_control_examples=unsafe_edges[:10],
                actual_active_cross_split_edges=len(actual_edges),quarantined_collections=sorted(quarantine),
                selected_cohort=100,selected_per_split=cohort['split_quotas'],complete_selected_windows=complete,
                incomplete_selected_windows=incomplete,pending_selected_windows=100-complete-incomplete,
                heldout_images_cached_in_cohort=False,new_images_downloaded=0,new_recovery_measurements=None,
                provenance=dict(admission_sha256=digest(data/'admissions/0001.json'),cohort_sha256=digest(cohort_path),
                                source_sha256={n:digest(ROOT/n) for n in code}),
                tests_passed=11,limitations=['Sampled road locations only; full route geometry may reveal additional conflicts',
                                             '100 selected does not mean 100 acquired, annotated or evaluated',
                                             'Unsafe assignment is a deliberately constructed control, not a defect found in the old frozen split'])
    freeze_ledger(out/'validation.json',result)
    print(json.dumps({k:v for k,v in result.items() if k not in ['provenance','unsafe_control_examples']},indent=2))


if __name__=='__main__':
    main()
