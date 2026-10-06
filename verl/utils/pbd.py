"""P5 PBD -- privileged-view branch distillation (docs/proposals_2026-10-06/12_P5_branch_distillation.md, 13 §15-§19).

V0 base (teacher = same model + crop, no null forward). For each student rollout y^S a fixed-ratio prefix
y_{<t*} (ratio of |y^S|, snapped to a word boundary) is continued by the rollout engine under the CROP VIEW
([full image][crop] + question, no hint sentence) -> y^T. A BRANCH ROW = student prompt + y_{<t*} + y^T is
appended to the training batch: the student (full image) is scored on it with grad, the frozen teacher
(crop view, usual teacher_prompt) gives the target; loss positions = t >= t* (prefix masked: duplicate of V0),
minus crop-wording (leak) tokens. Keep: parent rows keep their full loss mask; Replace: parent rows lose t >= t*.

Pure helpers here (CPU-testable); the trainer glue is RayPPOTrainer._maybe_append_pbd_branch_rows.
"""
from __future__ import annotations

import math
import re
from typing import Any, Optional

import numpy as np
import torch

LEAK_RE = re.compile(r"zoom|close-up|closeup|crop|enlarg|magnif|inset|second image|zoomed", re.I)


def snap_prefix_len(tokenizer, ids: list[int], ratio: float, min_len: int = 1) -> Optional[int]:
    """k = ceil(ratio*T) snapped FORWARD to the next token that starts a word (leading space / newline).
    Returns None when the response is too short to branch (k would cover (almost) everything)."""
    T = len(ids)
    if T < 4:
        return None
    k = max(min_len, int(math.ceil(ratio * T)))
    while k < T and not tokenizer.decode([ids[k]]).startswith((" ", "\n")):
        k += 1
    if k >= T - 1:   # nothing left to continue
        return None
    return k


def leak_token_mask(tokenizer, ids: list[int]) -> list[bool]:
    """True at token positions inside a crop-wording match (char-offset alignment of a re-tokenisation of the
    decoded text; positions are marked by char overlap). Falls back to all-False if alignment is impossible."""
    if not ids:
        return []
    text = tokenizer.decode(ids)
    spans = [m.span() for m in LEAK_RE.finditer(text)]
    if not spans:
        return [False] * len(ids)
    enc = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    off = enc["offset_mapping"]
    if len(off) != len(ids):
        # token boundaries differ after decode/re-encode; approximate by char position of each original token
        out, pos = [], 0
        for t in ids:
            piece = tokenizer.decode([t]); a, b = pos, pos + len(piece); pos = b
            out.append(any(a < e and b > s for s, e in spans))
        return out
    return [any(a < e and b > s for s, e in spans) for (a, b) in off]


def crop_view_messages(raw_prompt: list[dict], crop_image: Any) -> list[dict]:
    """Student messages (user turn with image items + text) -> crop-view user turn:
    [all original images][crop image] + original text segments. No hint sentence is added."""
    if not raw_prompt or raw_prompt[-1].get("role") != "user":
        raise ValueError("crop_view_messages expects the last message to be the user turn")
    msg = raw_prompt[-1]
    content = msg["content"]
    if isinstance(content, str):
        imgs, texts = [], [{"type": "text", "text": content}]
    else:
        imgs = [c for c in content if c.get("type") == "image"]
        texts = [c for c in content if c.get("type") != "image"]
    if not imgs:
        raise ValueError("student prompt has no image item; PBD needs the full image in the student view")
    # same normalisation as RLHFDataset._build_messages: dict -> {"type": "image", **d}, path -> "image"
    if isinstance(crop_image, dict):
        d = dict(crop_image)
        if "type" in d and "image" in d:
            crop_item = d
        else:
            d.pop("type", None)
            if d.get("bytes") is not None:
                from io import BytesIO
                from PIL import Image as _Image
                d["image"] = _Image.open(BytesIO(d.pop("bytes")))
            elif "image" not in d and "path" in d:
                d["image"] = d["path"]
            if "image" not in d:
                raise ValueError(f"crop image dict has neither 'image' nor 'path': {list(d)}")
            crop_item = {"type": "image", **d}
    else:
        crop_item = {"type": "image", "image": crop_image}
    new_content = list(imgs) + [crop_item] + list(texts)
    return list(raw_prompt[:-1]) + [{"role": "user", "content": new_content}]


def unpad_response(responses_row: torch.Tensor, attn_row: torch.Tensor) -> list[int]:
    return responses_row[attn_row.bool()].tolist()


def compute_position_ids(processor, input_ids: torch.Tensor, attention_mask: torch.Tensor, multi_modal_inputs: dict) -> torch.Tensor:
    """Replicates AgentLoopWorker._compute_position_ids for one row: text positions + mrope (processor) -> [1, 4, L],
    or plain incremental positions [1, L] without a processor."""
    from verl.utils.model import compute_position_id_with_mask
    if processor is None:
        return compute_position_id_with_mask(attention_mask)
    mm = dict(multi_modal_inputs)
    kwargs = {"image_grid_thw": mm.get("image_grid_thw"), "video_grid_thw": mm.get("video_grid_thw")}
    # the parent's multi_modal_inputs had mm_token_type_ids popped by the agent loop; Qwen3.5's get_rope_index
    # needs it, so rebuild it from input_ids whenever the processor exposes an image token id
    if getattr(processor, "image_token_id", None) is not None:
        tt = torch.zeros_like(input_ids)
        tt[0][input_ids[0] == processor.image_token_id] = 1
        vid = getattr(processor, "video_token_id", None)
        if vid is not None:
            tt[0][input_ids[0] == vid] = 2
        kwargs["mm_token_type_ids"] = tt
    vis, _ = processor.get_rope_index(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
    vis = vis.transpose(0, 1)
    valid = attention_mask[0].bool()
    txt = torch.ones((1, input_ids.shape[1]), dtype=torch.long)
    txt[0, valid] = torch.arange(int(valid.sum().item()))
    return torch.cat((txt.unsqueeze(0), vis), dim=1)


def make_branch_row_tensors(
    prompts_row: torch.Tensor, prompt_attn_row: torch.Tensor, prefix_ids: list[int], cont_ids: list[int],
    response_length: int, pad_token_id: int, leak_flags: Optional[list[bool]],
) -> dict[str, torch.Tensor]:
    """Build responses / attention / response_mask / pbd_loss_mask for ONE branch row (all [response_length])."""
    resp = (list(prefix_ids) + list(cont_ids))[:response_length]
    n = len(resp); k = min(len(prefix_ids), n)
    responses = torch.full((response_length,), pad_token_id, dtype=prompts_row.dtype)
    responses[:n] = torch.tensor(resp, dtype=prompts_row.dtype)
    resp_attn = torch.zeros(response_length, dtype=prompt_attn_row.dtype); resp_attn[:n] = 1
    loss = torch.zeros(response_length, dtype=torch.float32); loss[k:n] = 1.0
    if leak_flags:
        for j, f in enumerate(leak_flags[: max(0, n - k)]):
            if f:
                loss[k + j] = 0.0
    return {
        "responses": responses, "response_attention": resp_attn,
        "input_ids": torch.cat([prompts_row, responses]), "attention_mask": torch.cat([prompt_attn_row, resp_attn]),
        "response_mask": resp_attn.clone(),          # verl semantics: 1 on real response tokens (also used as teacher attention)
        "pbd_loss_mask": loss, "pbd_t_star": torch.tensor(k, dtype=torch.long), "pbd_n_resp": torch.tensor(n, dtype=torch.long),
    }


def parent_loss_mask(response_attn_row: torch.Tensor, k: Optional[int], mode: str) -> torch.Tensor:
    """Keep: all real response tokens; Replace: only t < t* on branched rows (k=None -> not branched -> all)."""
    m = response_attn_row.to(torch.float32).clone()
    if mode == "replace" and k is not None:
        m[k:] = 0.0
    return m


ANSWER_RE = re.compile(r"<answer>\s*[^<]{1,40}\s*</answer>", re.I)


def missing_answer(text: str) -> bool:
    return ANSWER_RE.search(text or "") is None


def leak_rel_positions(flags: list[bool]) -> list[float]:
    n = len(flags)
    return [j / n for j, f in enumerate(flags) if f] if n else []


def smoke_dump(dirname: Optional[str], name: str, obj: Any) -> None:
    """JSON dump for the pre-run integration test (14_P5_pre_run_integration_test_plan.md); no-op when dirname is None."""
    if not dirname:
        return
    import json, os
    def conv(x):
        if torch.is_tensor(x): return x.detach().cpu().tolist()
        if isinstance(x, np.ndarray): return x.tolist()
        if isinstance(x, (np.integer,)): return int(x)
        if isinstance(x, (np.floating,)): return float(x)
        if isinstance(x, dict): return {str(k): conv(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)): return [conv(v) for v in x]
        return x
    os.makedirs(dirname, exist_ok=True)
    with open(os.path.join(dirname, name), "w") as f:
        json.dump(conv(obj), f)


def coverage_select(n: int, coverage: float, seed: int, step: int) -> np.ndarray:
    """Indices of rollouts to branch: all for coverage>=1, else a seeded random subset (no targeting, §13.4/§14.3)."""
    if coverage >= 1.0:
        return np.arange(n)
    m = max(1, int(round(coverage * n)))
    rng = np.random.default_rng(seed * 1_000_003 + step)
    return np.sort(rng.choice(n, size=m, replace=False))
