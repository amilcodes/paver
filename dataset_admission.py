"""Append-only route admission with transitive geographic conflict quarantine.

No images are decoded, downloaded, scored or used for selection. Original split
labels and targets remain immutable. Quarantine is an exclusion, not relabeling.
"""
import argparse
import hashlib
import json
import math
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parent


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def xyz(lon, lat):
    lon, lat = math.radians(lon), math.radians(lat)
    return [6371000*math.cos(lat)*math.cos(lon), 6371000*math.cos(lat)*math.sin(lon), 6371000*math.sin(lat)]


def plan(rows, frozen, baseline=(), radius=5000):
    """Rows require id, collection, longitude, latitude. Frozen is an effective view."""
    rows = sorted(rows, key=lambda r: r['id'])
    if len({r['id'] for r in rows}) != len(rows):
        raise ValueError('Duplicate panorama IDs')
    assigned = frozen['collections']
    collections = sorted({r['collection'] for r in rows} | set(assigned))
    parent = {c:c for c in collections}
    def find(c):
        while c != parent[c]:
            parent[c] = parent[parent[c]]; c = parent[c]
        return c
    def union(a, b):
        a, b = find(a), find(b)
        if a != b:
            parent[max(a, b)] = min(a, b)
    # Prior spatial grouping must remain transitive even with sparse snapshots.
    previous_groups = defaultdict(list)
    for c, value in assigned.items():
        previous_groups[frozen.get('effective_groups', {}).get(c, value['group'])].append(c)
    for members in previous_groups.values():
        for c in members[1:]:
            union(members[0], c)
    positions = np.asarray([xyz(r['longitude'], r['latitude']) for r in rows]).reshape(-1, 3)
    tree = cKDTree(positions)
    edges, examples = set(), {}
    # Stream neighborhoods instead of allocating all point-pair combinations.
    for i, point in enumerate(positions):
        for j in tree.query_ball_point(point, radius):
            a, b = rows[i]['collection'], rows[j]['collection']
            if a == b:
                continue
            edge = tuple(sorted((a, b)))
            if edge in edges:
                continue
            edges.add(edge); union(*edge)
            examples[edge] = dict(collections=list(edge), picture_ids=[rows[i]['id'], rows[j]['id']],
                                  chord_distance_m=float(np.linalg.norm(positions[i]-positions[j])))
    dev_collections = set()
    for item in baseline:
        if item['collection'] in parent:
            dev_collections.add(item['collection'])
        for i in tree.query_ball_point(xyz(*item['coordinates'][:2]), radius):
            dev_collections.add(rows[i]['collection'])
    components = defaultdict(list)
    for c in collections:
        components[find(c)].append(c)
    additions, targets, conflicts, effective_groups = {}, [], [], {}
    quarantine = set(frozen.get('quarantined_collections', []))
    by_collection = defaultdict(list)
    for r in rows:
        by_collection[r['collection']].append(r)
    for group, members in sorted(components.items()):
        labels = {assigned[c]['split'] for c in members if c in assigned}
        touches_development = bool(dev_collections.intersection(members))
        if touches_development:
            labels.add('development')
        if len(labels) > 1 or quarantine.intersection(members):
            quarantine.update(members)
            conflicts.append(dict(group=group, members=members, inherited_splits=sorted(labels),
                                  reason='transitive component crosses frozen splits' if len(labels)>1 else 'prior quarantine',
                                  new_collections=[c for c in members if c not in assigned]))
            continue
        if labels:
            split = next(iter(labels)); reason = 'inherits connected frozen split or known development neighborhood'
        else:
            value = int(hashlib.sha256(('split-v1:'+group).encode()).hexdigest()[:8], 16) % 10
            split = 'test' if value<2 else 'validation' if value<4 else 'train'
            reason = 'new isolated component; deterministic assignment before image review'
        for c in members:
            effective_groups[c] = group
            if c in assigned:
                continue
            additions[c] = dict(group=group, split=split, reason=reason)
            chosen = min(by_collection[c], key=lambda r:hashlib.sha256(('target-v1:'+r['id']).encode()).hexdigest())
            targets.append(dict(id=chosen['id'], collection=c, group=group, split=split))
    combined = dict(assigned, **additions)
    active_cross_split_edges = [list(e) for e in edges if not quarantine.intersection(e)
                               and e[0] in combined and e[1] in combined
                               and combined[e[0]]['split'] != combined[e[1]]['split']]
    assert not active_cross_split_edges
    # Do not quietly admit a collection without an eligible target.
    assert {t['collection'] for t in targets} == set(additions)
    return dict(additions=additions, targets=targets, quarantined_collections=sorted(quarantine),
                conflicts=conflicts, effective_groups=effective_groups,
                collection_neighbor_edges=[examples[e] for e in sorted(edges)],
                active_cross_split_edges=active_cross_split_edges,
                metrics=dict(input_pictures=len(rows), input_collections=len(collections),
                             previously_assigned=len(assigned), newly_observed=len(set(collections)-set(assigned)),
                             admitted=len(additions), quarantined_new=len(quarantine-set(assigned)),
                             quarantined_existing=len(quarantine.intersection(assigned)),
                             additions_by_split=dict(Counter(a['split'] for a in additions.values())),
                             effective_active_groups=len(set(effective_groups.values())),
                             collection_neighbor_edges=len(edges)))


def effective_splits(data):
    """Validate immutable ledger chain; exclude quarantines from usable targets."""
    data = Path(data); base = data/'splits_v1.json'
    frozen = json.loads(base.read_text()); result = json.loads(json.dumps(frozen))
    result['effective_groups'] = {c:v['group'] for c,v in result['collections'].items()}
    quarantine, prior = set(), []
    for path in sorted((data/'admissions').glob('*.json')):
        ledger = json.loads(path.read_text())
        assert ledger['base_sha256'] == digest(base), 'Frozen base changed'
        assert ledger['prior_ledgers'] == prior, 'Admission chain changed'
        assert not (set(ledger['plan']['additions']) & set(result['collections'])), 'Reassignment forbidden'
        result['collections'].update(ledger['plan']['additions'])
        result['targets'].extend(ledger['plan']['targets'])
        result['effective_groups'].update(ledger['plan']['effective_groups'])
        quarantine.update(ledger['plan']['quarantined_collections'])
        prior.append(dict(name=path.name, sha256=digest(path)))
    result['quarantined_collections'] = sorted(quarantine)
    result['targets'] = [dict(t, group=result['effective_groups'].get(t['collection'], t['group']))
                         for t in result['targets'] if t['collection'] not in quarantine]
    result['ledger_chain'] = prior
    assert all(result['collections'][c] == v for c,v in frozen['collections'].items())
    return result


def freeze_ledger(path, ledger):
    """Idempotent only for an exactly identical record; never overwrite."""
    path = Path(path)
    if path.exists():
        if canonical(json.loads(path.read_text())) != canonical(ledger):
            raise RuntimeError('Immutable admission already exists with different contents')
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as f:
        f.write(json.dumps(ledger, indent=2, allow_nan=False)+'\n')


def build(data, output, commit=False):
    data, output = Path(data), Path(output)
    frozen = effective_splits(data)
    with sqlite3.connect(f'file:{data / "catalog.sqlite"}?mode=ro', uri=True) as db:
        db.row_factory = sqlite3.Row
        rows = [dict(r) for r in db.execute('SELECT id,collection,longitude,latitude,datetime,license,producer,metadata_path,metadata_sha256 FROM pictures WHERE eligible=1 ORDER BY id')]
    for r in rows:
        assert r['license'] == 'etalab-2.0'
        assert digest(data/r['metadata_path']) == r['metadata_sha256']
    baseline = [json.loads(p.read_text()) for p in (data.parent/'benchmark_v2/data').glob('*/target_record.json')]
    result = plan(rows, frozen, baseline)
    output.mkdir(parents=True, exist_ok=True)
    snapshot = output/'eligible_snapshot.json'
    if snapshot.exists():
        assert json.loads(snapshot.read_text()) == rows, 'Snapshot exists with different input'
    else:
        snapshot.write_bytes(canonical(rows)+b'\n')
    ledger = dict(version=1, base_sha256=digest(data/'splits_v1.json'), prior_ledgers=frozen['ledger_chain'],
                  snapshot_path=str(snapshot.relative_to(data.parent)), snapshot_sha256=digest(snapshot),
                  source_sha256=digest(__file__), radius_m=5000, plan=result,
                  limitations=['Sampled points, not complete route geometry',
                               'Excluded bridge components require independent recollection or review; frozen labels are not rewritten',
                               'No new images viewed; labels, licenses and metadata are inherited from the catalog'],
                  frozen_before_image_review=True)
    freeze_ledger(output/'proposal.json', ledger)
    if commit:
        path = data/'admissions'/f'{len(frozen["ledger_chain"])+1:04d}.json'
        freeze_ledger(path, ledger)
        effective = effective_splits(data)
        assert all(effective['collections'][c] == a for c,a in frozen['collections'].items())
    print(json.dumps(result['metrics'], indent=2))
    return ledger


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--data', default=str(ROOT/'dataset_v1'))
    p.add_argument('--output', default=str(ROOT/'dataset_v1/admission_audits/001'))
    p.add_argument('--commit', action='store_true'); a = p.parse_args()
    build(a.data, a.output, a.commit)
