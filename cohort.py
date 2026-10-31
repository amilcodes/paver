"""Freeze a result-independent 100-collection acquisition cohort."""
import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from dataset_admission import ROOT, digest, effective_splits, freeze_ledger

QUOTAS = {'train':60, 'validation':20, 'test':20}


def choose_targets(effective, cached_collections, quotas=QUOTAS):
    selected = []
    for split, quota in quotas.items():
        candidates = [t for t in effective['targets'] if t['split']==split]
        cached = sorted((t for t in candidates if t['collection'] in cached_collections), key=lambda t:t['collection'])
        if len(cached)>quota or len(candidates)<quota:
            raise ValueError(f'Cannot fill {split} quota without dropping cached routes or inventing targets')
        selected.extend(cached); chosen = {t['collection'] for t in cached}
        groups = Counter(t['group'] for t in cached)
        while len(chosen)<quota:
            remaining = [t for t in candidates if t['collection'] not in chosen]
            t = min(remaining, key=lambda t:(groups[t['group']], hashlib.sha256(('cohort-100-v1:'+t['group']).encode()).hexdigest(), t['collection']))
            selected.append(t); chosen.add(t['collection']); groups[t['group']]+=1
    assert len({t['collection'] for t in selected}) == sum(quotas.values())
    return selected


def freeze_cohort(data, output):
    data, output = Path(data), Path(output)
    if output.exists():
        manifest = json.loads(output.read_text())
        assert manifest['base_sha256'] == digest(data/'splits_v1.json')
        for entry in manifest['admission_ledgers']:
            assert digest(data/'admissions'/entry['name']) == entry['sha256']
        print('RESUME', output); return manifest
    effective = effective_splits(data)
    cached, sources = set(), []
    for p in sorted((data/'windows').glob('*.json')):
        w = json.loads(p.read_text())
        if w['collection'] in effective['collections']:
            cached.add(w['collection']); sources.append(dict(path=str(p.relative_to(data)), sha256=digest(p)))
    selected = choose_targets(effective, cached)
    # Metadata only; source pixels are never opened for selection.
    for t in selected:
        p = data/'metadata'/(t['id']+'.json'); d = json.loads(p.read_text())
        assert d['id']==t['id'] and d['collection']==t['collection']
        assert d['properties']['license']=='etalab-2.0'
        t.update(metadata_path=str(p.relative_to(data)), metadata_sha256=digest(p),
                 license=d['properties']['license'], producer=d['properties'].get('geovisio:producer'),
                 capture_time=d['properties'].get('datetime'), coordinates=d['geometry']['coordinates'],
                 was_cached_at_selection=t['collection'] in cached)
    result = dict(name='initial_100_v1', selected_collections=len(selected), target_window_frames=5,
                  split_quotas=QUOTAS, base_sha256=digest(data/'splits_v1.json'),
                  admission_ledgers=effective['ledger_chain'], selection_source_sha256=digest(__file__),
                  cached_window_inputs=sources, targets=selected,
                  selection='Retain existing cached non-development routes; fill quotas by least-represented geographic group then deterministic hash. No quality or reconstruction scores.',
                  frozen_before_new_image_review=True,
                  limitations=['This is a selected acquisition cohort, not 100 downloaded or annotated sequences',
                               'Cached training images have already been inspected in some cases; no heldout images reviewed',
                               'Spatial independence uses sampled 5 km neighborhoods, not complete routes',
                               'Never tune on test collections or use heldout recovery results for selection'])
    freeze_ledger(output, result)
    print(json.dumps(dict(selected=len(selected), per_split=QUOTAS,
                          geographic_groups={s:len({t['group'] for t in selected if t['split']==s}) for s in QUOTAS}), indent=2))
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',default=str(ROOT/'dataset_v1'))
    p.add_argument('--output',default=str(ROOT/'dataset_v1/cohorts/initial_100_v1.json'))
    a=p.parse_args();freeze_cohort(a.data,a.output)
