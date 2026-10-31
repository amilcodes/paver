"""Verify all exported snapshot files (or just Git files) against the ledger."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(1024*1024), b''): h.update(b)
    return h.hexdigest()

def main():
    p=argparse.ArgumentParser(); p.add_argument('--git-only', action='store_true'); a=p.parse_args()
    rows=[json.loads(s) for s in (ROOT/'publication/snapshot.jsonl').read_text().splitlines()]
    checked=0; missing=[]; corrupt=[]
    for r in rows:
        if a.git_only and r['storage']!='git': continue
        path=ROOT/r['path']
        if not path.is_file(): missing.append(r['path']); continue
        if path.stat().st_size!=r['size'] or digest(path)!=r['sha256']: corrupt.append(r['path'])
        checked+=1
    result=dict(passed=not missing and not corrupt, checked=checked, missing=missing, corrupt=corrupt)
    print(json.dumps(result)); return 0 if result['passed'] else 1

if __name__=='__main__': raise SystemExit(main())
