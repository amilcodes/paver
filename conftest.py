"""Keep asset-dependent replay tests explicit in a lightweight checkout.

The signed historical test files remain unchanged. Full tests run automatically
once the documented release assets are installed.
"""
import json
from pathlib import Path
import pytest

ROOT=Path(__file__).resolve().parent
DATA_TESTS={
    'test_refreezing_identical_masks_does_not_rewrite_artifacts':'benchmark_v3/annotations_train_001.json',
    'test_frozen_case_cannot_be_changed_or_rewritten':'benchmark_v3/annotations_train_002.json',
}

def pytest_collection_modifyitems(items):
    for item in items:
        annotation=DATA_TESTS.get(item.name)
        if not annotation: continue
        data=json.loads((ROOT/annotation).read_text())
        if any(not (ROOT/frame['image']).is_file() for frame in data['frames'].values()):
            item.add_marker(pytest.mark.skip(reason='Release image integration test: run python tools/fetch_assets.py first'))
