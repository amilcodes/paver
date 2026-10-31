import copy,json,sys
from pathlib import Path
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from prepare_annotated_case_v2 import compile_masks,prepare,CATEGORIES
from benchmark import ROOT,digest


def rectangle(x0,y0,x1,y1):
    return [[x0,y0],[x1,y0],[x1,y1],[x0,y1]]


def fixture():
    parts={k:[] for k in CATEGORIES}
    parts.update(ego=[rectangle(.4,.2,.6,.4)],ground=[rectangle(.01,.01,.99,.99)],
                 moving=[rectangle(.1,.1,.2,.2)],shadow=[rectangle(.4,.4,.6,.5)],
                 privacy=[rectangle(.8,.1,.9,.2)])
    return dict(projection={'size':256},frame_order=['target'],frames={'target':parts},boundary_margin_px=2,
                regions={'large':[rectangle(.1,.55,.9,.95)]})


def test_movers_shadows_and_blurs_are_never_footprint_or_valid_truth():
    a=fixture();m,_,_=compile_masks(a)
    nuisance=m['target_moving']|m['target_shadow']|m['target_privacy']
    assert not (m['target_allowed']&nuisance).any()
    assert np.array_equal(m['region_footprint'],m['target_ego'])
    assert not (m['region_large']&m['target_excluded']).any()
    b=copy.deepcopy(a);b['frames']['target']['moving'].append(rectangle(.2,.6,.4,.8))
    changed,_,_=compile_masks(b)
    assert np.array_equal(changed['region_footprint'],m['region_footprint'])
    assert changed['region_large'].sum()<m['region_large'].sum()


def test_bad_polygon_cannot_silently_create_ground_truth():
    a=fixture();a['frames']['target']['ground']=[rectangle(-.1,0,1,1)]
    with pytest.raises(ValueError,match='Invalid'):compile_masks(a)
    a=fixture();a['regions']['footprint']=[rectangle(0,0,1,1)]
    with pytest.raises(ValueError,match='Cannot replace'):compile_masks(a)


def test_margin_excludes_uncertain_boundaries_without_inflating_ego():
    a=fixture();m,_,_=compile_masks(a);a['boundary_margin_px']=8;n,_,_=compile_masks(a)
    assert np.array_equal(m['region_footprint'],n['region_footprint'])
    assert n['target_allowed'].sum()<m['target_allowed'].sum()


def test_frozen_case_cannot_be_changed_or_rewritten(tmp_path):
    p=ROOT/'benchmark_v3/annotations_train_002.json';folder=ROOT/'benchmark_v3/prepared/nantes_van_002'
    before={n:(digest(folder/n),(folder/n).stat().st_mtime_ns) for n in ['manifest.json','masks.npz']}
    prepare(p,freeze=True)
    assert before=={n:(digest(folder/n),(folder/n).stat().st_mtime_ns) for n in before}
    changed=json.loads(p.read_text());changed['frames']['target']['shadow'][0][0][0]+=.01
    q=tmp_path/'modified.json';q.write_text(json.dumps(changed))
    with pytest.raises(RuntimeError,match='Frozen annotation changed'):prepare(q,freeze=True)
    assert before=={n:(digest(folder/n),(folder/n).stat().st_mtime_ns) for n in before}
