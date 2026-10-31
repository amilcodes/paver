"""One-variable sampling ablation; base preprocessing and gates unchanged."""
from dense_matching import MaskedLoFTR
from benchmark import ROOT,digest
from run_dense import model_signature

MAX_SIDE=960


class SamplingMatcher(MaskedLoFTR):
    name='masked_loftr_outdoor_960'

    def __init__(self,weights,expected_sha256,*,root):
        super().__init__(weights,expected_sha256,root=root,max_side=MAX_SIDE)


def signature():
    base=model_signature()
    return dict(**base,treatment_adapter_sha256=digest(ROOT/'sampling_matcher.py'),
        treatment=dict(max_side=MAX_SIDE,base_max_side=640,crop='unchanged allowed-ground bounding box',
            margin_original_pixels=32,endpoint_dedup_cell_original_pixels=6,maximum_matches=2048,
            tiling=False,confidence_and_geometry_thresholds='unchanged'))
