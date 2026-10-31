"""Admission checks for new research work; no deletion or implicit cleanup."""
import shutil
from pathlib import Path

GIB = 1024 ** 3


def admission(root, jobs=(), *, minimum_free_gib=8, cache_cap_gib=8,
              reserve_bytes=0, free_bytes=None, cache_bytes=None):
    root = Path(root)
    if free_bytes is None:
        free_bytes = shutil.disk_usage(root).free
    if cache_bytes is None:
        cache_bytes = sum(p.stat().st_size for p in (root / 'dataset_v1/images').glob('*') if p.is_file())
    storage_ok = free_bytes - reserve_bytes >= minimum_free_gib * GIB
    cache_ok = cache_bytes <= cache_cap_gib * GIB
    heavy_running = any('run_' in job or 'preflight_dense' in job for job in jobs)
    acquisition_running = any('dataset.py' in job for job in jobs)
    blockers = []
    if not storage_ok:
        blockers.append('free disk after reservation is below the configured minimum')
    if not cache_ok:
        blockers.append('image cache exceeds the configured cap')
    if heavy_running:
        blockers.append('another heavy research job is running')
    if acquisition_running:
        blockers.append('an acquisition worker is running')
    return dict(free_disk_gib=free_bytes / GIB, image_cache_gib=cache_bytes / GIB,
                reserved_gib=reserve_bytes / GIB, minimum_free_gib=minimum_free_gib,
                safe_to_start_heavy_job=storage_ok and cache_ok and not heavy_running,
                safe_to_start_acquisition=storage_ok and cache_ok and not acquisition_running,
                blockers=blockers)


def require_storage(root, reserve_bytes=512 * 1024 ** 2):
    state = admission(root, reserve_bytes=reserve_bytes)
    if not state['safe_to_start_heavy_job']:
        raise RuntimeError(f"Research storage guard: {state}")
    return state
