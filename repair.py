"""Geometry-first, selective panorama repair. CPU-only experimental baseline.

RGB uint8 images; masks are boolean, True = allowed/repair depending on name.
Homographies map source nadir pixels to target nadir pixels. A single plane
is assumed; this is NOT a general 3D scene reconstruction method.
"""
from dataclasses import dataclass
import cv2
import numpy as np


def nadir(panorama, size=1024, fov=145):
    """Equirectangular -> downward pinhole view, with horizontal seam wrapping.

    Longitude zero is the panorama's center column, pointing to the top of
    the output. Rays are (x, -1, z), y up; fov is horizontal in degrees.
    """
    if panorama.ndim != 3 or panorama.shape[1] != 2 * panorama.shape[0]:
        raise ValueError("Expected a full 2:1 equirectangular RGB panorama")
    if size < 8 or not 0 < fov < 180:
        raise ValueError("size >= 8 and 0 < fov < 180 required")
    h, w = panorama.shape[:2]
    f = size / (2 * np.tan(np.deg2rad(fov / 2)))
    y, x = np.mgrid[:size, :size]
    dx, dz = (x - (size - 1) / 2) / f, -(y - (size - 1) / 2) / f
    lon = np.arctan2(dx, dz)
    lat = np.arctan2(-np.ones_like(dx), np.hypot(dx, dz))
    u = ((lon / (2 * np.pi) + .5) * w - .5).astype(np.float32)
    v = np.clip((.5 - lat / np.pi) * h - .5, 0, h - 1).astype(np.float32)
    return cv2.remap(panorama, u, v, cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)


def polygon_mask(shape, polygons):
    """Normalized [0,1] polygon coordinates; multiple polygons form a union."""
    h, w = shape[:2]
    mask = np.zeros((h, w), np.uint8)
    for polygon in polygons:
        points = np.rint(np.asarray(polygon) * [w - 1, h - 1]).astype(np.int32)
        cv2.fillPoly(mask, [points], 255)
    return mask > 0


def paste_nadir(panorama, patch, accepted, fov):
    """Return a full panorama, changing only safely supported patch samples.

    All unaccepted pixels retain their exact input bytes. The acceptance mask
    is eroded so bilinear sampling cannot blend a repair with an unresolved hole.
    """
    h,w=panorama.shape[:2]
    n=patch.shape[0]
    if patch.shape[1] != n or accepted.shape != (n,n) or not 0 < fov < 180:
        raise ValueError('Expected square patch/mask and 0 < fov < 180')
    f=n/(2*np.tan(np.deg2rad(fov/2)))
    y,x=np.mgrid[:h,:w]
    lon=(x+.5)/w*(2*np.pi)-np.pi
    lat=np.pi/2-(y+.5)/h*np.pi
    down=-np.sin(lat)
    denominator=np.maximum(down,1e-8)
    u=(f*np.cos(lat)*np.sin(lon)/denominator+(n-1)/2).astype(np.float32)
    v=(-f*np.cos(lat)*np.cos(lon)/denominator+(n-1)/2).astype(np.float32)
    safe=cv2.erode(accepted.astype(np.uint8),np.ones((5,5),np.uint8),
                   borderType=cv2.BORDER_CONSTANT,borderValue=0)
    supported=(cv2.remap(safe,u,v,cv2.INTER_NEAREST,borderMode=cv2.BORDER_CONSTANT)>0)&(down>0)
    transferred=cv2.remap(patch,u,v,cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT)
    result=panorama.copy()
    result[supported]=transferred[supported]
    return result,supported


@dataclass
class Registration:
    homography: object
    support: np.ndarray
    diagnostics: dict


def register(source, target, source_allowed, target_allowed, seed=7):
    """SIFT mutual ratio matching + RANSAC, gated by spatial support.

    Excluded image content is zeroed BEFORE feature extraction, then accepted
    keypoints require their full approximate descriptor support to be allowed.
    Thus a withheld patch cannot supply target descriptors indirectly.
    """
    cv2.setRNGSeed(seed)
    def describe(image, allowed):
        clean = image.copy()
        clean[~allowed] = 0
        gray = cv2.cvtColor(clean, cv2.COLOR_RGB2GRAY)
        sift = cv2.SIFT_create(nfeatures=10000, contrastThreshold=.015)
        kp, desc = sift.detectAndCompute(gray, allowed.astype(np.uint8) * 255)
        if desc is None:
            return [], None
        distance = cv2.distanceTransform(allowed.astype(np.uint8), cv2.DIST_L2, 5)
        keep = [i for i, k in enumerate(kp)
                if distance[min(int(k.pt[1]), image.shape[0]-1),
                            min(int(k.pt[0]), image.shape[1]-1)] > 6 * k.size + 2]
        return [kp[i] for i in keep], desc[keep] if keep else None

    sk, sd = describe(source, source_allowed)
    tk, td = describe(target, target_allowed)
    info = dict(source_features=len(sk), target_features=len(tk), matches=0,
                inliers=0, accepted=False)
    empty = np.zeros(target.shape[:2], bool)
    if sd is None or td is None or min(len(sd), len(td)) < 2:
        info['reason'] = 'insufficient features'
        return Registration(None, empty, info)
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    def ratio(a, b):
        return {m.queryIdx: m.trainIdx for pair in matcher.knnMatch(a, b, k=2)
                if len(pair) == 2 for m, n in [pair] if m.distance < .75 * n.distance}
    forward, backward = ratio(sd, td), ratio(td, sd)
    pairs = [(i, j) for i, j in forward.items() if backward.get(j) == i]
    info['matches'] = len(pairs)
    if len(pairs) < 12:
        info['reason'] = 'fewer than 12 mutual matches'
        return Registration(None, empty, info)
    sp = np.float32([sk[i].pt for i, _ in pairs])
    tp = np.float32([tk[j].pt for _, j in pairs])
    H, inlier_mask = cv2.findHomography(sp, tp, cv2.RANSAC, 3.0,
                                      maxIters=10000, confidence=.999)
    if H is None or not np.isfinite(H).all():
        info['reason'] = 'homography estimation failed'
        return Registration(None, empty, info)
    inliers = inlier_mask.ravel().astype(bool)
    predicted = cv2.perspectiveTransform(sp[:, None, :], H)[:, 0]
    errors = np.linalg.norm(predicted - tp, axis=1)
    hull = cv2.convexHull(tp[inliers])
    hull_fraction = cv2.contourArea(hull) / target.shape[0] / target.shape[1]
    info.update(inliers=int(inliers.sum()), inlier_ratio=float(inliers.mean()),
                median_reprojection_px=float(np.median(errors[inliers])),
                inlier_hull_fraction=float(hull_fraction), homography=H.tolist())
    if inliers.sum() < 10 or inliers.mean() < .35 or hull_fraction < .015:
        info['reason'] = 'weak or spatially concentrated geometric support'
        return Registration(None, empty, info)
    support = np.zeros(target.shape[:2], np.uint8)
    cv2.fillConvexPoly(support, np.rint(hull).astype(np.int32), 255)
    info.update(accepted=True, reason='accepted planar registration')
    return Registration(H, support > 0, info)


def fuse(target, repair_mask, sources, source_allowed, registrations,
         agreement_threshold=18.0, min_views=1):
    """Transfer supported source observations; abstain on conflict/no coverage.

    One-view transfers remain explicitly labeled single_view. Two views must
    agree in mean RGB absolute difference <= threshold (0..255). This is a
    heuristic consistency gate, NOT proof of geometry or calibrated confidence.
    Select a single observed donor per pixel to preserve auditable provenance.
    """
    if min_views < 1 or not np.isfinite(agreement_threshold) or agreement_threshold < 0:
        raise ValueError('Invalid fusion thresholds')
    h, w = target.shape[:2]
    candidates, valid, quality = [], [], []
    for src, allowed, reg in zip(sources, source_allowed, registrations, strict=True):
        if reg.homography is None:
            candidates.append(np.zeros_like(target))
            valid.append(np.zeros((h, w), bool))
            quality.append(0.)
            continue
        # Erode before bilinear sampling so blocked content cannot bleed in.
        safe = cv2.erode(allowed.astype(np.uint8), np.ones((5, 5), np.uint8))
        warped = cv2.warpPerspective(src, reg.homography, (w, h), flags=cv2.INTER_LINEAR)
        visible = cv2.warpPerspective(safe, reg.homography, (w, h),
                                     flags=cv2.INTER_NEAREST) > 0
        candidates.append(warped)
        valid.append(visible & reg.support & repair_mask)
        quality.append(reg.diagnostics['inliers'] /
                       (1 + reg.diagnostics['median_reprojection_px']))
    output = target.copy()
    provenance = np.full((h, w), -1, np.int16)
    count = np.zeros((h, w), np.uint8)
    disagreement = np.zeros((h, w), np.float32)
    if not sources:
        return dict(image=output, provenance=provenance, count=count,
                    accepted=np.zeros_like(repair_mask), disagreement=disagreement)
    stack, usable = np.stack(candidates), np.stack(valid)
    count = usable.sum(axis=0).astype(np.uint8)
    # Conservative maximum pairwise disagreement. No majority voting yet.
    for i in range(len(sources)):
        for j in range(i + 1, len(sources)):
            common = usable[i] & usable[j]
            diff = np.abs(stack[i].astype(np.float32) - stack[j]).mean(axis=2)
            disagreement[common] = np.maximum(disagreement[common], diff[common])
    accepted = repair_mask & (count >= min_views) & (disagreement <= agreement_threshold)
    scores = np.asarray(quality)[:, None, None] * usable
    donor = np.argmax(scores, axis=0)
    yy, xx = np.nonzero(accepted)
    output[yy, xx] = stack[donor[yy, xx], yy, xx]
    provenance[accepted] = donor[accepted]
    return dict(image=output, provenance=provenance, count=count,
                accepted=accepted, disagreement=disagreement)


def score(prediction, truth, mask):
    """Metrics only on explicitly specified pixels; never whole-image PSNR."""
    if not mask.any():
        return {'pixels': 0, 'mae_255': None, 'psnr_db': None}
    difference = prediction[mask].astype(np.float64) - truth[mask]
    mse = np.mean(difference ** 2)
    return dict(pixels=int(mask.sum()), mae_255=float(np.abs(difference).mean()),
                psnr_db=float(10 * np.log10(255 ** 2 / max(mse, 1e-12))))
