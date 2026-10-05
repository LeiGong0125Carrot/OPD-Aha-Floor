# Copyright 2026 Vision-OPD / OPD-Aha-Floor authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
"""Arm Amis -- "mismatched-crop" visual null for OPD-Aha target reconstruction.

OPD-Aha (arm A) builds the null view as [full image, mean-colour block] (null_scope=last). The
log-ratio u = log p+ - log p0 then mixes two channels: the CONTENT of the crop (evidence) and the
mere PRESENCE of a real zoom image (style: "zoomed/cropped" wording, explain-before-answer, ...).
Here the null view is [full image, crop of ANOTHER prompt in the same batch]: a real zoom image is
present in both forwards, so the style channel cancels while the evidence channel (which only the
own crop carries) is kept. Everything else of the A target is unchanged (same beta, JSD, support).

Hard constraint (dp_actor): the null forward re-uses the real teacher's input_ids / position_ids
and swaps only the image inputs, so the donor crop MUST be resized to the own crop's pixel size
(identical `image_grid_thw`). ray_trainer additionally asserts prefix-tensor equality.

Pure-python helpers live here so they can be unit-tested on CPU without importing ray.
"""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np

__all__ = ["sattolo_derangement", "pick_donor_indices", "mismatch_crop_images", "average_null_log_probs"]


def sattolo_derangement(n: int, rng: np.random.Generator) -> np.ndarray:
    """Return a permutation of range(n) with NO fixed point (single cycle; Sattolo's algorithm).
    n must be >= 2."""
    if n < 2:
        raise ValueError(f"mismatch_crop: need at least 2 distinct prompts in the batch, got {n}")
    perm = np.arange(n)
    for i in range(n - 1, 0, -1):
        j = int(rng.integers(0, i))          # j in [0, i): never i itself -> single cycle, no fixed point
        perm[i], perm[j] = perm[j], perm[i]
    return perm


def pick_donor_indices(uids, seed: int, step: int, num_views: int = 1) -> list[np.ndarray]:
    """For every sample i (uids[i] = prompt id shared by its n rollouts) choose `num_views` donor
    sample indices whose prompt differs from uids[i]. All rollouts of one prompt get the same donors.
    Deterministic in (seed, step). Returns a list of `num_views` int arrays of shape [batch].
    With num_views == 2 the second donor is also different from the first."""
    uids = list(uids)
    uniq = list(dict.fromkeys(uids))                      # first-occurrence order
    n = len(uniq)
    if num_views not in (1, 2):
        raise ValueError(f"mismatch_crop: num_views must be 1 or 2, got {num_views}")
    if n < 2 or (num_views == 2 and n < 3):
        raise ValueError(
            f"mismatch_crop: need >= {2 if num_views == 1 else 3} distinct prompts for num_views={num_views}, got {n}"
        )
    first_idx = {}
    for i, u in enumerate(uids):
        first_idx.setdefault(u, i)
    pos = {u: k for k, u in enumerate(uniq)}
    rng = np.random.default_rng(int(seed) * 1_000_003 + int(step))
    perm1 = sattolo_derangement(n, rng)
    perms = [perm1]
    if num_views == 2:
        perm2 = sattolo_derangement(n, rng)               # independent draw from the same stream
        # if a prompt got the same donor twice, move the second donor one step along perm1's cycle
        # (perm1[perm1[k]] != k for n >= 3 since perm1 is a single n-cycle, and != perm1[k])
        same = perm2 == perm1
        perm2 = np.where(same, perm1[perm1], perm2)
        perms.append(perm2)
    out = []
    for perm in perms:
        donors = np.array([first_idx[uniq[perm[pos[u]]]] for u in uids], dtype=np.int64)
        out.append(donors)
    for donors in out:
        for i, d in enumerate(donors):
            if uids[int(d)] == uids[i]:
                raise RuntimeError("mismatch_crop: donor has the same prompt as the sample (derangement bug)")
    if num_views == 2:
        for i in range(len(uids)):
            if uids[int(out[0][i])] == uids[int(out[1][i])]:
                raise RuntimeError("mismatch_crop: the two donors share a prompt")
    return out


def mismatch_crop_images(teacher_images: list, donor_crop, normalize: Callable, resample=None) -> list:
    """null_scope=last with a mismatched donor: keep every image but the last (normalised, as the
    mean-colour path does); replace the last by the donor crop resized to the own crop's exact size."""
    if len(teacher_images) < 1:
        raise ValueError("mismatch_crop: teacher_images is empty")
    from PIL import Image
    own = normalize(teacher_images[-1])
    donor = normalize(donor_crop)
    if donor is own or (
        isinstance(donor_crop, dict) and isinstance(teacher_images[-1], dict)
        and donor_crop.get("path") is not None and donor_crop.get("path") == teacher_images[-1].get("path")
    ):
        raise RuntimeError("mismatch_crop: donor crop is the sample's own crop")
    if resample is None:
        resample = Image.BICUBIC
    out = [normalize(img) for img in teacher_images[:-1]]
    out.append(donor.resize(own.size, resample) if donor.size != own.size else donor.copy())
    if out[-1].size != own.size:
        raise RuntimeError("mismatch_crop: resized donor size mismatch")
    return out


def average_null_log_probs(a, b):
    """Two-view null (num_views=2): geometric mean in probability space = arithmetic mean of log-probs,
    consistent with u = log p+ - mean_j log p0_j. For top-k log-probs the mass stays <= 1 (geometric
    mean <= arithmetic mean of two valid sub-distributions). `None` passes through if both are None."""
    if a is None and b is None:
        return None
    if a is None or b is None:
        raise ValueError("average_null_log_probs: both views must be present")
    if a.shape != b.shape:
        raise ValueError(f"average_null_log_probs: shape mismatch {tuple(a.shape)} vs {tuple(b.shape)}")
    return 0.5 * (a + b)
