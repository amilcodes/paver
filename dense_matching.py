"""Masked LoFTR adapter. Local pinned weights only; MPS inference only.

The preprocessing is independently testable without torch or model downloads.
Inference remains unvalidated until the recorded resource blocker is cleared.
"""
import hashlib
from pathlib import Path
import cv2
import numpy as np


def prepare_masked(image, allowed, max_side=640, margin=32):
    if image.shape[:2] != allowed.shape or allowed.dtype != np.bool_:
        raise ValueError('Expected matching image and boolean mask dimensions')
    if max_side < 32 or margin < 0:
        raise ValueError('Invalid preprocessing limits')
    # Sanitize BEFORE grayscale, crop, resize, backbone and global attention.
    clean = image.copy()
    clean[~allowed] = 0
    ys, xs = np.nonzero(allowed)
    if not len(xs):
        return dict(gray=np.zeros((8, 8), np.float32), mask=np.zeros((8, 8), bool),
                    origin=[0, 0], scale=[1., 1.], original_shape=list(allowed.shape))
    h, w = allowed.shape
    x0, x1 = max(0, int(xs.min()) - margin), min(w, int(xs.max()) + margin + 1)
    y0, y1 = max(0, int(ys.min()) - margin), min(h, int(ys.max()) + margin + 1)
    scale = min(1., max_side / max(x1 - x0, y1 - y0))
    rw, rh = max(1, round((x1 - x0) * scale)), max(1, round((y1 - y0) * scale))
    gray = cv2.cvtColor(clean[y0:y1, x0:x1], cv2.COLOR_RGB2GRAY).astype(np.float32) / 255
    gray = cv2.resize(gray, (rw, rh), interpolation=cv2.INTER_AREA)
    safe = cv2.distanceTransform(allowed.astype(np.uint8), cv2.DIST_L2, 5) > margin
    safe = cv2.resize(safe[y0:y1, x0:x1].astype(np.float32), (rw, rh), interpolation=cv2.INTER_AREA) > .999
    ph, pw = ((rh + 7) // 8) * 8, ((rw + 7) // 8) * 8
    return dict(gray=np.pad(gray, ((0, ph-rh), (0, pw-rw))),
                mask=np.pad(safe, ((0, ph-rh), (0, pw-rw))),
                origin=[x0, y0], scale=[rw/(x1-x0), rh/(y1-y0)],
                original_shape=list(allowed.shape))


def restore_points(raw_points, prepared):
    # Installed Kornia LoFTR uses reference-image pixel centers at x+0.5.
    # Convert to integer-center OpenCV coordinates, then undo resize and crop.
    return np.asarray(raw_points, np.float64) / np.asarray(prepared['scale']) - .5 + prepared['origin']


def filter_matches(p, q, confidence, allowed0, allowed1, margin=32, limit=2048):
    p, q, confidence = np.asarray(p), np.asarray(q), np.asarray(confidence)
    keep = np.isfinite(p).all(1) & np.isfinite(q).all(1) & np.isfinite(confidence)
    for points, allowed in [(p, allowed0), (q, allowed1)]:
        h, w = allowed.shape
        finite = np.nan_to_num(points, nan=-1, posinf=-1, neginf=-1)
        ix, iy = np.rint(finite[:, 0]).astype(int), np.rint(finite[:, 1]).astype(int)
        inside = (ix >= 0) & (ix < w) & (iy >= 0) & (iy < h)
        d = cv2.distanceTransform(allowed.astype(np.uint8), cv2.DIST_L2, 5)
        keep &= inside & (d[np.clip(iy, 0, h-1), np.clip(ix, 0, w-1)] > margin)
    selected, seen0, seen1 = [], set(), set()
    for i in np.argsort(-confidence, kind='stable'):
        if not keep[i]:
            continue
        a, b = tuple(np.rint(p[i]/6).astype(int)), tuple(np.rint(q[i]/6).astype(int))
        if a in seen0 or b in seen1:
            continue
        selected.append(i); seen0.add(a); seen1.add(b)
        if len(selected) >= limit:
            break
    return p[selected].astype(np.float32), q[selected].astype(np.float32), confidence[selected]


class MaskedLoFTR:
    name = 'masked_loftr_outdoor_640'

    def __init__(self, weights, expected_sha256, *, root, max_side=640):
        from resource_guard import require_storage
        require_storage(root)
        weights = Path(weights)
        if not weights.is_file() or hashlib.sha256(weights.read_bytes()).hexdigest() != expected_sha256:
            raise ValueError('Local LoFTR checkpoint missing or checksum mismatch')
        import torch
        from kornia.feature import LoFTR
        if not torch.backends.mps.is_available():
            raise RuntimeError('MPS required; no CPU inference fallback')
        self.torch, self.max_side, self.root = torch, max_side, root
        self.model = LoFTR(pretrained=None).eval()
        self.model.load_state_dict(torch.load(weights, map_location='cpu', weights_only=True)['state_dict'])
        self.model.to('mps')

    def correspondences(self, image0, image1, allowed0, allowed1):
        from resource_guard import require_storage
        require_storage(self.root)
        a, b = [prepare_masked(im, mask, self.max_side) for im, mask in [(image0, allowed0), (image1, allowed1)]]
        if not a['mask'].any() or not b['mask'].any():
            return np.empty((0, 2), np.float32), np.empty((0, 2), np.float32), dict(method=self.name, reason='no safe ground')
        t = self.torch
        data = {}
        for i, item in enumerate([a, b]):
            data[f'image{i}'] = t.from_numpy(item['gray'])[None, None].to('mps')
            data[f'mask{i}'] = t.from_numpy(item['mask'])[None].to('mps')
        with t.inference_mode():
            out = self.model(data)
        p, q = [restore_points(out[f'keypoints{i}'].cpu().numpy(), item) for i, item in enumerate([a, b])]
        p, q, confidence = filter_matches(p, q, out['confidence'].cpu().numpy(), allowed0, allowed1)
        del data, out
        t.mps.empty_cache()
        return p, q, dict(method=self.name, matches=len(p), max_side=self.max_side,
                         minimum_confidence=float(confidence.min()) if len(confidence) else None,
                         weights='explicit local checkpoint; caller must retain SHA-256')
