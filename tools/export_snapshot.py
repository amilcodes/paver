"""Create a portable public derivative without modifying the research source.

Usage: python export_snapshot.py SOURCE DESTINATION
Unchanged large assets use hardlinks locally; GitHub release archives copy bytes.
"""
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sqlite3
import sys
from datetime import datetime, timezone

SOURCE, DEST = map(lambda s: Path(s).resolve(), sys.argv[1:3])
if DEST.exists():
    raise SystemExit('Destination must not already exist')
if shutil.disk_usage(SOURCE).free < 8.15 * 2**30:
    raise SystemExit('Need 8 GiB reserve plus 0.15 GiB publication headroom')
EXCLUDED = {
    'autoresearch/state.json', 'autoresearch/HANDOFF.txt', 'autoresearch/environment.txt',
    'autoresearch/cleanup_2026-10-31.json', 'autoresearch/events.jsonl',
    'network_v1/automation_update.json', 'area_sets_v1/automation_update.json',
}
TEXT = {'.py', '.json', '.jsonl', '.md', '.txt', '.html', '.log', '.geojson', '.xml', '.yml'}
rows, contents, original, omitted = [], {}, {}, []
DEST.mkdir(parents=True)
def sha(b): return hashlib.sha256(b).hexdigest()
for src in sorted(SOURCE.rglob('*')):
    if not src.is_file(): continue
    rel = src.relative_to(SOURCE).as_posix()
    if any(x in {'.git', '__pycache__', '.pytest_cache'} for x in src.relative_to(SOURCE).parts): continue
    if rel in EXCLUDED or rel.startswith('area_sets_v1/research_sources/') or src.name.endswith('.lock') or src.name in {'.DS_Store', '.gitignore'}:
        omitted.append(rel); continue
    if src.is_symlink(): raise RuntimeError('Unexpected symlink: '+rel)
    name = 'docs/LEGACY_DEMO.md' if rel == 'README.md' else rel
    out = DEST / name; out.parent.mkdir(parents=True, exist_ok=True)
    b = src.read_bytes(); digest = sha(b)
    release = src.suffix.lower() in {'.jpg', '.jpeg', '.png', '.npz', '.sqlite'} or rel.startswith(('dataset_v1/metadata/', 'dataset_v1/queries/'))
    row = dict(path=name, source_path=rel, source_sha256=digest, sha256=digest,
               size=len(b), source_mtime_utc=datetime.fromtimestamp(src.stat().st_mtime, timezone.utc).isoformat(),
               storage='release' if release else 'git', transformations=[])
    if src.suffix in TEXT and not release:
        # Original numerical results are unchanged. Paths become ROOT-relative.
        clean = b.replace((str(SOURCE)+'/').encode(), b'').replace(str(SOURCE).encode(), b'.')
        if b'/Users/' in clean: raise RuntimeError('Unreviewed host path in '+rel)
        original[name] = b; contents[name] = clean
        if clean != b: row['transformations'].append('research-root paths made relative')
    elif src.suffix == '.sqlite':
        c = sqlite3.connect(f'file:{src}?mode=ro', uri=True); d = sqlite3.connect(out)
        c.backup(d); d.close(); c.close()
        # SQLite backup can change physical layout. No table values are changed.
        row['sha256'] = sha(out.read_bytes()); row['size'] = out.stat().st_size
        if row['sha256'] != digest: row['transformations'].append('consistent SQLite backup')
    else:
        os.link(src, out)
    rows.append(row)

# Resolve the artifact DAG from the original bytes on every pass. Never hash an
# already-rewritten hash as an input identity. Publication ledger keeps both.
digest_map = {r['source_sha256']: r['sha256'] for r in rows}
base = contents.copy()
token = re.compile(rb'(?<![0-9a-f])[0-9a-f]{64}(?![0-9a-f])')
for iteration in range(50):
    changed = False
    for row in rows:
        name = row['path']
        if name not in base: continue
        b = base[name]
        if name.endswith(('.json', '.jsonl')):
            b = token.sub(lambda m: digest_map.get(m[0].decode(), m[0].decode()).encode(), b)
        contents[name] = b
        h = sha(b)
        if h != row['sha256']: changed = True
        row['sha256'], row['size'] = h, len(b)
    digest_map = {r['source_sha256']: r['sha256'] for r in rows}
    if not changed: break
else: raise RuntimeError('Artifact hash references contain a cycle; do not publish')

for row in rows:
    name = row['path']
    if name in contents:
        if contents[name] != base[name]: row['transformations'].append('derived artifact hash references updated')
        (DEST/name).write_bytes(contents[name])
        stamp = datetime.fromisoformat(row['source_mtime_utc']).timestamp()
        os.utime(DEST/name, (stamp, stamp))
    # Audit transformations against a whitelist; numeric payload is untouched.
    if name.endswith('.json') and name in contents:
        def normalized(x):
            if isinstance(x, str):
                x = x.replace(str(SOURCE)+'/', '').replace(str(SOURCE), '.')
                return token.sub(b'<SHA256>', x.encode()).decode()
            if isinstance(x, dict): return {normalized(k): normalized(v) for k,v in x.items()}
            if isinstance(x, list): return [normalized(v) for v in x]
            return x
        assert normalized(json.loads(original[name])) == normalized(json.loads(contents[name])), name

pub = DEST/'publication'; pub.mkdir()
(pub/'snapshot.jsonl').write_text(''.join(json.dumps(r, sort_keys=True)+'\n' for r in rows))
(pub/'export.json').write_text(json.dumps(dict(
    format=1, created_at_utc=datetime.now(timezone.utc).isoformat(),
    files=len(rows), git_files=sum(r['storage']=='git' for r in rows),
    release_files=sum(r['storage']=='release' for r in rows),
    omitted=omitted, hash_reference_passes=iteration+1,
    numeric_payloads_unchanged=True, source_files_modified=False,
    note='Portable derivative, not a byte-identical original archive. Original/export hashes are paired in snapshot.jsonl. No historical Git snapshots existed; current recorded file bytes are imported using recorded modification times.'
), indent=2)+'\n')
print(json.dumps(dict(files=len(rows), git_bytes=sum(r['size'] for r in rows if r['storage']=='git'),
                     release_bytes=sum(r['size'] for r in rows if r['storage']=='release'),
                     transformed=sum(bool(r['transformations']) for r in rows), passes=iteration+1)))
