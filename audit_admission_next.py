"""Audit an uncommitted append-only proposal against its exact preceding ledger.

Spatial enumeration is the independent grid implementation used for admission1.
Original split assignments and the frozen acquisition cohort are never rewritten.
"""
import argparse,hashlib,json
from pathlib import Path
from collections import Counter
from dataset_admission import ROOT,digest,effective_splits,freeze_ledger
from audit_admission import grid_edges


def audit(proposal,output):
    data=ROOT/'dataset_v1';proposal=Path(proposal).resolve();ledger=json.loads(proposal.read_text())
    prior=effective_splits(data)
    assert prior['ledger_chain']==ledger['prior_ledgers'],'Audit requires exact uncommitted predecessor'
    assert digest(data/'splits_v1.json')==ledger['base_sha256']
    assert digest(ROOT/'dataset_admission.py')==ledger['source_sha256']
    snapshot=ROOT/ledger['snapshot_path'];assert digest(snapshot)==ledger['snapshot_sha256']
    rows=json.loads(snapshot.read_text());assert len({r['id'] for r in rows})==len(rows)
    for r in rows:assert r['license']=='etalab-2.0' and digest(data/r['metadata_path'])==r['metadata_sha256']
    additions=ledger['plan']['additions'];assert not(set(additions)&set(prior['collections']))
    combined=dict(prior['collections'],**additions);quarantine=set(prior['quarantined_collections'])|set(ledger['plan']['quarantined_collections'])
    edges,pairs=grid_edges(rows,ledger['radius_m'])
    assert edges=={tuple(r['collections']) for r in ledger['plan']['collection_neighbor_edges']}
    def conflicts(assignments):
        return [list(e) for e in sorted(edges) if not quarantine.intersection(e) and assignments[e[0]]['split']!=assignments[e[1]]['split']]
    actual=conflicts(combined);assert not actual
    assert all(combined[c]==a for c,a in prior['collections'].items())
    assert len(ledger['plan']['targets'])==len(additions)
    assert {t['collection'] for t in ledger['plan']['targets']}==set(additions)
    unsafe=dict(combined)
    for c in additions:
        v=int(hashlib.sha256(('split-v1:'+c).encode()).hexdigest()[:8],16)%10
        unsafe[c]={'split':'test' if v<2 else 'validation' if v<4 else 'train'}
    unsafe_edges=conflicts(unsafe)
    cohortpath=data/'cohorts/initial_100_v1.json';cohort=json.loads(cohortpath.read_text())
    assert len(cohort['targets'])==len({t['collection'] for t in cohort['targets']})==100
    assert dict(Counter(t['split'] for t in cohort['targets']))==cohort['split_quotas']
    windows={json.loads(p.read_text())['collection']:json.loads(p.read_text()) for p in (data/'windows').glob('*.json')}
    for t in cohort['targets']:
        assert combined[t['collection']]['split']==t['split'] and t['collection'] not in quarantine
        assert digest(data/t['metadata_path'])==t['metadata_sha256']
        if t['split'] in ['test','validation']:assert t['collection'] not in windows
    complete=sum(bool(windows.get(t['collection'],{}).get('complete_window')) for t in cohort['targets'])
    incomplete=sum(t['collection'] in windows and not windows[t['collection']]['complete_window'] for t in cohort['targets'])
    result=dict(passed=True,scope='Independent spatial-grid audit of uncommitted admission; does not claim complete route geometry.',
        proposal_sha256=digest(proposal),prior_ledgers=ledger['prior_ledgers'],snapshot_sha256=digest(snapshot),
        independent_grid_source_sha256=digest(ROOT/'audit_admission.py'),validator_sha256=digest(Path(__file__)),
        input_pictures=len(rows),collection_neighbor_edges=len(edges),picture_neighbor_pairs=pairs,
        previous_assignments_preserved=len(prior['collections']),new_admissions=len(additions),quarantined_collections=sorted(quarantine),
        actual_active_cross_split_edges=len(actual),unsafe_independent_assignment_cross_split_edges=len(unsafe_edges),
        unsafe_control_examples=unsafe_edges[:10],effective_split_counts=dict(Counter(a['split'] for a in combined.values())),
        cohort_sha256=digest(cohortpath),selected_cohort=100,complete_selected_windows=complete,
        incomplete_selected_windows=incomplete,pending_selected_windows=100-complete-incomplete,
        heldout_images_cached_in_cohort=False,new_image_scoring=False)
    freeze_ledger(Path(output).resolve(),result)
    print(json.dumps({k:v for k,v in result.items() if k not in ['prior_ledgers','unsafe_control_examples']},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--proposal',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    audit(a.proposal,a.output)
