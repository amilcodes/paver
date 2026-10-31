"""Resume a checksummed public release without trusting archive paths or links."""
import argparse
import fcntl
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import shutil
import tarfile
import urllib.request

ROOT=Path(__file__).resolve().parents[1]
GIB=2**30

def sha(data): return hashlib.sha256(data).hexdigest()
def file_sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024), b''): h.update(b)
    return h.hexdigest()

def safe_target(root, name):
    p=PurePosixPath(name)
    if p.is_absolute() or '..' in p.parts or '\\' in name: raise ValueError('Unsafe asset path')
    target=root/name
    if not target.resolve().is_relative_to(root.resolve()): raise ValueError('Asset escapes root')
    return target

def extract_verified(payload, entries, root):
    """Validate all members before writing; never replace an existing file."""
    expected={r['path']:r for r in entries}
    if len(expected)!=len(entries): raise ValueError('Duplicate manifest paths')
    with tarfile.open(fileobj=io.BytesIO(payload), mode='r:gz') as t:
        members=t.getmembers()
        if len(members)!=len(expected) or {m.name for m in members}!=set(expected): raise ValueError('Archive population differs')
        for m in members:
            dest=safe_target(root, m.name); r=expected[m.name]
            if not m.isfile() or m.size!=r['size']: raise ValueError('Unexpected member type or size')
            data=t.extractfile(m).read()
            if sha(data)!=r['sha256']: raise ValueError('Payload checksum mismatch: '+m.name)
            if dest.exists() and (not dest.is_file() or file_sha(dest)!=r['sha256']): raise ValueError('Existing file differs: '+m.name)
        for m in members:
            dest=safe_target(root, m.name)
            if dest.exists(): continue
            if shutil.disk_usage(root).free-m.size < 8*GIB: raise RuntimeError('8 GiB disk reserve')
            dest.parent.mkdir(parents=True,exist_ok=True)
            # Exclusive create makes interruption resumable at a file boundary.
            tmp=dest.with_name(dest.name+'.paver-partial')
            if tmp.exists():
                if tmp.is_symlink() or not tmp.is_file(): raise ValueError('Unsafe partial file')
                # This reserved filename is written only by the fetcher. The
                # verified archive remains the authority after an interruption.
                tmp.unlink()
            with tmp.open('xb') as f: shutil.copyfileobj(t.extractfile(m), f)
            tmp.rename(dest)

def main():
    p=argparse.ArgumentParser();p.add_argument('--part', action='append');a=p.parse_args()
    manifest=json.loads((ROOT/'publication/release-assets.json').read_text())
    ledger={r['path']:r for r in map(json.loads,(ROOT/'publication/snapshot.jsonl').read_text().splitlines())}
    names={part['name'] for part in manifest['parts']}
    if a.part and not set(a.part)<=names: p.error('Unknown part; see publication/release-assets.json')
    cache=ROOT/'.asset-cache';cache.mkdir(exist_ok=True)
    with (cache/'acquisition.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        for part in manifest['parts']:
            if a.part and part['name'] not in a.part: continue
            entries=[ledger[n] for n in part['files']]
            missing=[r for r in entries if not (ROOT/r['path']).is_file()]
            if not missing and all(file_sha(ROOT/r['path'])==r['sha256'] for r in entries):
                print('Verified',part['name'],flush=True);continue
            if shutil.disk_usage(ROOT).free-sum(r['size'] for r in missing) < 8*GIB: raise RuntimeError('8 GiB disk reserve')
            with urllib.request.urlopen(part['url'],timeout=120) as response: payload=response.read(part['size']+1)
            if len(payload)!=part['size'] or sha(payload)!=part['sha256']: raise ValueError('Archive checksum mismatch')
            extract_verified(payload,entries,ROOT)
            print('Installed',part['name'],flush=True)

if __name__=='__main__': main()
