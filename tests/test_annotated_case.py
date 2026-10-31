import json,sys
from pathlib import Path
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from benchmark import ROOT,digest
from prepare_annotated_case import prepare

def test_refreezing_identical_masks_does_not_rewrite_artifacts():
 path=ROOT/'benchmark_v3/annotations_train_001.json';folder=ROOT/'benchmark_v3/prepared/bourges_turn_001'
 before={n:(digest(folder/n),(folder/n).stat().st_mtime_ns) for n in ['masks.npz','manifest.json']}
 prepare(path,freeze=True)
 after={n:(digest(folder/n),(folder/n).stat().st_mtime_ns) for n in before}
 assert before==after

def test_changed_annotation_cannot_overwrite_frozen_masks(tmp_path):
 original=ROOT/'benchmark_v3/annotations_train_001.json';data=json.loads(original.read_text());data['regions']['local_front'][0][0][0]+=.01
 variant=tmp_path/'changed.json';variant.write_text(json.dumps(data));folder=ROOT/'benchmark_v3/prepared/bourges_turn_001';before=digest(folder/'masks.npz')
 with pytest.raises(RuntimeError,match='Frozen annotation changed'):prepare(variant,freeze=True)
 assert digest(folder/'masks.npz')==before
