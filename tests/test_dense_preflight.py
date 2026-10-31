import sys
from pathlib import Path
import numpy as np
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dense_matching import prepare_masked, restore_points, filter_matches, MaskedLoFTR
from resource_guard import admission, GIB


def test_excluded_rgb_never_changes_resampled_model_input():
    rng = np.random.default_rng(10)
    im = rng.integers(0, 256, (233, 355, 3), dtype=np.uint8)
    allowed = np.zeros((233, 355), bool); allowed[19:211, 21:333] = True
    allowed[71:142, 123:178] = False
    changed = im.copy(); changed[~allowed] = 255 - im[~allowed]
    a, b = [prepare_masked(x, allowed, max_side=128) for x in [im, changed]]
    assert np.array_equal(a['gray'], b['gray'])
    assert np.array_equal(a['mask'], b['mask'])
    assert a['gray'].shape[0] % 8 == a['gray'].shape[1] % 8 == 0
    assert a['mask'].any()
    changed = im.copy(); changed[allowed] = 0
    assert not np.array_equal(a['gray'], prepare_masked(changed, allowed, 128)['gray'])


def test_reference_pixel_centers_roundtrip_resize_and_crop():
    im = np.zeros((367, 511, 3), np.uint8)
    mask = np.zeros((367, 511), bool); mask[91:309, 117:430] = True
    prep = prepare_masked(im, mask, 128)
    native = np.array([[152., 150.], [203.25, 212.75], [400., 290.]])
    # OpenCV resize maps integer pixel centers with a half-pixel offset.
    resized_cv = (native - prep['origin'] + .5) * prep['scale'] - .5
    reference_centers = resized_cv + .5
    assert np.allclose(restore_points(reference_centers, prep), native)


def test_empty_masks_cannot_offer_match_support():
    result = prepare_masked(np.full((80, 96, 3), 255, np.uint8), np.zeros((80, 96), bool))
    assert not result['gray'].any() and not result['mask'].any()


def test_matches_reject_hidden_points_nonfinite_and_duplicate_support():
    mask = np.zeros((180, 180), bool); mask[10:170, 10:170] = True
    mask[70:90, 70:90] = False
    p = np.array([[45., 45.], [45.5, 45.5], [80., 80.], [-10., 40.], [np.nan, 50.], [130., 130.]])
    q = p.copy(); q[-1] = [80., 80.]
    kept, _, conf = filter_matches(p, q, np.array([.8, .9, .95, .99, .99, .9]), mask, mask, margin=12)
    assert len(kept) == 1
    assert np.allclose(kept[0], p[1]) and conf[0] == .9


def test_low_disk_blocks_even_when_no_jobs_run(tmp_path):
    d = admission(tmp_path, free_bytes=int(5.96*GIB), cache_bytes=0)
    assert not d['safe_to_start_heavy_job'] and not d['safe_to_start_acquisition']
    assert d['blockers']
    exact = admission(tmp_path, free_bytes=8*GIB, cache_bytes=0)
    assert exact['safe_to_start_heavy_job']
    reserved = admission(tmp_path, free_bytes=8*GIB, cache_bytes=0, reserve_bytes=1)
    assert not reserved['safe_to_start_heavy_job']


def test_jobs_and_cache_cap_gate_admission(tmp_path):
    d = admission(tmp_path, ['python run_dense.py'], free_bytes=10*GIB, cache_bytes=0)
    assert not d['safe_to_start_heavy_job'] and d['safe_to_start_acquisition']
    d = admission(tmp_path, ['python dataset.py download'], free_bytes=10*GIB, cache_bytes=0)
    assert d['safe_to_start_heavy_job'] and not d['safe_to_start_acquisition']
    d = admission(tmp_path, free_bytes=10*GIB, cache_bytes=8*GIB+1)
    assert not d['safe_to_start_heavy_job'] and not d['safe_to_start_acquisition']


def test_blocked_constructor_never_imports_or_downloads_model(tmp_path, monkeypatch):
    import resource_guard
    def blocked(*args, **kwargs):
        raise RuntimeError('disk floor')
    monkeypatch.setattr(resource_guard, 'require_storage', blocked)
    with pytest.raises(RuntimeError, match='disk floor'):
        MaskedLoFTR(tmp_path/'absent.ckpt', 'not-a-checksum', root=tmp_path)

