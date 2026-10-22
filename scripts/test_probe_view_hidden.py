#!/usr/bin/env python
"""CPU tests for scripts/probe_view_hidden.py helpers + analyze on synthetic tokens. 用法: python scripts/test_probe_view_hidden.py"""
import json, os, sys, tempfile, types
import numpy as np, torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
import probe_view_hidden as V

fails = []
def check(name, cond):
    print(("PASS " if cond else "FAIL ") + name); cond or fails.append(name)

check("cos shape [L,T] from [L,T,D]", tuple(V.cos_rows(torch.randn(3, 5, 4), torch.randn(3, 5, 4)).shape) == (3, 5))
a = torch.tensor([[[1.0, 0, 100], [0, 2.0, 100]]]); b = torch.tensor([[[0, 1.0, 100], [0, 2.0, 100]]])
check("ndist: shared huge dim cancels, scale divides", torch.allclose(V.ndist(a, b, torch.tensor([2.0])), torch.tensor([[1.0, 0.0]])))
keys = [("img0", "c0"), ("img0", "c1"), ("img1", "c2"), ("img2", "c2")]
check("other_row skips same source image", V.other_row(0, keys) == 2)
check("other_row skips same crop path", V.other_row(2, keys) == 0 and V.other_row(3, keys) == 0)
try:
    V.other_row(0, [("a", "x"), ("a", "y")]); check("other_row no valid donor raises", False)
except ValueError:
    check("other_row no valid donor raises", True)
check("PAIRS explicit", V.PAIRS["S2C"] == ("S2", "C") and V.PAIRS["SS2"] == ("S", "S2") and V.PAIRS["CM"] == ("C", "M"))
t = np.array([0, 1, 7, 8, 9, 10, 11, 0, 3, 9, 2, 12]); c = np.array([10] * 7 + [-1, -1, -1, 2, 2])
bm = V.bucket_masks(t, c)
check("buckets", list(np.where(bm["t=0 (生成前)"])[0]) == [0, 7] and list(np.where(bm["承诺前 (≥8)"])[0]) == [3, 4]
      and list(np.where(bm["承诺 token (≥8)"])[0]) == [5] and list(np.where(bm["承诺后 (≥8)"])[0]) == [6, 11]
      and list(np.where(bm["承诺未知 (≥8)"])[0]) == [9] and list(np.where(bm["1–7"])[0]) == [1, 2, 8, 10])
check("buckets disjoint and cover all", (sum(m.astype(int) for m in bm.values()) == 1).all())

# forward(): fake model whose logits / hidden encode the absolute position -> pins the P-1 slice and logits_to_keep
class Fake:
    device = torch.device("cpu")
    def __call__(self, input_ids, attention_mask, output_hidden_states, logits_to_keep, use_cache, **kw):
        S = input_ids.shape[1]; pos = torch.arange(S, dtype=torch.float32)
        hs = tuple(pos.view(1, S, 1).expand(1, S, 3) + 1000 * l for l in range(33))
        lg = torch.zeros(1, S, 5); lg[0, torch.arange(S), (pos.long() % 5)] = 10.0
        return types.SimpleNamespace(hidden_states=hs, logits=lg[:, S - logits_to_keep:])
enc = dict(input_ids=torch.zeros(1, 6, dtype=torch.long), attention_mask=torch.ones(1, 6, dtype=torch.long))
ls, hid = V.forward(Fake(), enc, [1, 2, 3], (1, 32))
check("forward: hidden at P-1..P+T-2", hid.shape == (2, 3, 3) and hid[0, :, 0].tolist() == [1005.0, 1006.0, 1007.0] and hid[1, 0, 0] == 32005.0)
check("forward: logits aligned with hidden positions", ls.argmax(-1).tolist() == [0, 1, 2])

# analyze on synthetic: nd_SC < nd_SM (content share 0.5), training lowers nd (S2C < SC)
rng = np.random.RandomState(0); N, L = 2000, len(V.LAYERS); d = tempfile.mkdtemp(); z = {}
noise = rng.rand(N, L) * 0.01
for p, nd in (("SC", 0.5), ("SM", 1.0), ("CM", 1.0), ("S2C", 0.4), ("S2C2", 0.3), ("SS2", 0.1)):
    z[f"cos_{p}"] = (1 - nd / 2 + noise).astype(np.float32); z[f"nd_{p}"] = (nd + noise).astype(np.float32)
z["nd_SC"][:, 0] += np.linspace(0, 0.01, N).astype(np.float32)
z.update(jsd_SC=rng.rand(N) * 0.1, jsd_SM=rng.rand(N) * 0.2, jsd_S2C=rng.rand(N) * 0.05,
         row=np.repeat(np.arange(N // 20), 20), s=np.zeros(N, int), t=np.tile(np.arange(20), N // 20), tok=np.zeros(N, int), commit=np.full(N, 10))
np.savez_compressed(os.path.join(d, "tokens.npz"), **z)
vs = {f"{v}_{k}": dict(mean=1.0, median=1.0) for v in "SCM" for k in ("prompt", "img")}
json.dump(dict(skipped={}, n_rows=N // 20, trained="x", K=100, layers=list(V.LAYERS), warm_tokens=1, views=vs), open(os.path.join(d, "meta.json"), "w"))
V.part_analyze(types.SimpleNamespace(out=d)); s = json.load(open(os.path.join(d, "summary.json")))
check("analyze: content share ~0.5", abs(s["layer"][str(V.LAYERS[1])]["content_share"] - 0.5) < 0.02)
check("analyze: S2C < SC at every bucket/layer", all(v[str(l)]["S2C"] < v[str(l)]["SC"] for v in s["pos"].values() for l in (8, 16, 24, 31)))
check("analyze: only the populated buckets", set(s["pos"]) == {"t=0 (生成前)", "1–7", "承诺前 (≥8)", "承诺 token (≥8)", "承诺后 (≥8)"})
check("analyze: within-trajectory spearman computed", all(np.isfinite(v["within"]) or np.isnan(v["within"]) for v in s["out_vs_hidden"].values()) and s["out_vs_hidden"][str(V.LAYERS[0])]["n_traj"] == N // 20)
print("\nALL PASS" if not fails else f"\n{len(fails)} FAIL: {fails}")
sys.exit(1 if fails else 0)
