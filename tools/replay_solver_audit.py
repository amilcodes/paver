"""Replay the frozen audit, redirecting only its final immutable output.

No matching, optimization, checkpoint loading or historical-file writes.
"""
import fcntl
import json
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

def main():
    import audit_solver_budget as audit
    from resource_guard import require_storage
    require_storage(ROOT)
    destination=ROOT/'publication/replay-audit.json'
    if destination.exists(): raise SystemExit('Replay output already exists; inspect it before running again')
    # Optional shared lock for a live research directory elsewhere on this host.
    shared=Path(os.environ.get('PAVER_SHARED_LOCK', ROOT/'autoresearch/heavy_job.lock'))
    shared.parent.mkdir(parents=True, exist_ok=True)
    with shared.open('a') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX|fcntl.LOCK_NB)
        def publish(path, result):
            if path != audit.OUT/'audit_001.json': raise RuntimeError('Unexpected audit output')
            result['publication_replay']=True
            result['historical_audit_modified']=False
            destination.write_text(json.dumps(result, indent=2)+'\n')
        audit.immutable_json=publish
        audit.audit()

if __name__=='__main__': main()
