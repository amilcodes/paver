"""Audit current metadata/windows against all admitted splits before acquisition.

Unlike the frozen admission audit, this includes newly discovered observations.
Cross-split components remain quarantined even if a later snapshot omits them.
"""
import argparse,json,sqlite3
from pathlib import Path
from datetime import datetime,timezone
from dataset_admission import ROOT,digest,effective_splits,plan


def audit_current(data):
    data=Path(data);effective=effective_splits(data);failures=[]
    with sqlite3.connect(f'file:{data/"catalog.sqlite"}?mode=ro',uri=True) as db:
        db.row_factory=sqlite3.Row
        rows=[dict(r) for r in db.execute('SELECT id,collection,longitude,latitude,metadata_path,metadata_sha256 FROM pictures WHERE eligible=1 ORDER BY id')]
    lookup={r['id']:r for r in rows}
    images=set();windows=[];legacy_metadata=0
    for path in sorted((data/'windows').glob('*.json')):
        w=json.loads(path.read_text());seen=set()
        for rec in w['images']:
            if rec['id'] in seen or rec['collection']!=w['collection']:
                failures.append([path.name,'duplicate image or mixed route'])
            seen.add(rec['id']);images.add(rec['id'])
            image=data/'images'/(rec['id']+'.jpg');metadata=data/'metadata'/(rec['id']+'.json')
            if not image.exists() or digest(image)!=rec['sample_sha256']:
                failures.append([rec['id'],'image checksum mismatch'])
            expected_metadata=rec.get('metadata_sha256')
            if expected_metadata is None:
                expected_metadata=lookup.get(rec['id'],{}).get('metadata_sha256')
                legacy_metadata+=1
            if not expected_metadata or not metadata.exists() or digest(metadata)!=expected_metadata:
                failures.append([rec['id'],'metadata checksum mismatch']);continue
            m=json.loads(metadata.read_text())
            if rec['license']!='etalab-2.0' or m['properties'].get('license')!='etalab-2.0' or m['collection']!=rec['collection']:
                failures.append([rec['id'],'license or collection mismatch'])
            if rec['id'] not in lookup:
                xy=m['geometry']['coordinates']
                row=dict(id=rec['id'],collection=m['collection'],longitude=xy[0],latitude=xy[1],metadata_path=str(metadata.relative_to(data)),metadata_sha256=digest(metadata))
                lookup[rec['id']]=row;rows.append(row)
        expected={'target'}|{f'{direction}{i}' for direction in ['prev','next'] for i in range(1,w.get('radius',0)+1)}
        complete=bool(w['complete_window']) if 'complete_window' in w else len(seen)==len(expected) and {r['name'] for r in w['images']}==expected
        windows.append(dict(collection=w['collection'],images=len(seen),complete_window=complete,
                            completeness_inferred_from_radius='complete_window' not in w))
    for row in rows:
        if digest(data/row['metadata_path'])!=row['metadata_sha256']:
            failures.append([row['id'],'catalog metadata checksum mismatch'])
    baseline=[json.loads(p.read_text()) for p in (data.parent/'benchmark_v2/data').glob('*/target_record.json')]
    proposed=plan(rows,effective,baseline)
    quarantine_path=data/'active_quarantine.json'
    prior=json.loads(quarantine_path.read_text()) if quarantine_path.exists() else {'collections':[]}
    quarantine=set(prior['collections'])|set(proposed['quarantined_collections'])
    if quarantine:
        quarantine_path.write_text(json.dumps(dict(collections=sorted(quarantine),reason='Current or previously detected transitive split conflict; never silently unquarantine'),indent=2)+'\n')
    result=dict(checked_at=datetime.now(timezone.utc).isoformat(),eligible_and_cached_pictures=len(rows),cached_images=len(images),
                admitted_collections=len(effective['collections']),unassigned_observed_collections=proposed['metrics']['newly_observed'],
                legacy_window_metadata_verified_against_catalog=legacy_metadata,
                quarantine=sorted(quarantine),conflicts=proposed['conflicts'],integrity_failures=failures,
                integrity_pass=not failures,spatial_split_audit_pass=not quarantine,
                integrity_allows_acquisition=not failures and not quarantine,windows=windows,
                base_sha256=digest(data/'splits_v1.json'),ledger_chain=effective['ledger_chain'],source_sha256=digest(__file__),
                note='Checks current metadata and cached window sources; sampled coordinates are not complete route geometry. Does not admit any new collection.')
    path=data/'effective_audit.json';tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(result,indent=2)+'\n');tmp.replace(path)
    if failures or quarantine:
        raise RuntimeError('Dataset integrity/split audit failed; see effective_audit.json; no further acquisition allowed')
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',default=str(ROOT/'dataset_v1'));a=p.parse_args()
    result=audit_current(a.data);print(json.dumps({k:v for k,v in result.items() if k not in ['windows','ledger_chain']},indent=2))
