import glob, numpy as np, torch
from verl.trainer.ppo.core_algos import compute_self_distillation_loss
class Cfg(dict): __getattr__ = dict.get
def cfg(**kw):
    d = {"full_logit_distillation": True, "distillation_topk": 100, "distillation_add_tail": True,
         "renorm_topk_log_probs": False, "alpha": 0.5, "is_clip": None,
         "counterfactual_null_mode": "mean_color", "counterfactual_extrapolation_beta": 4.0,
         "counterfactual_u_clip_pos": False, "counterfactual_st_enable": False,
         "counterfactual_floor_alpha": 0.0, "counterfactual_tanh_scale": 0.0,
         "counterfactual_target_gamma": 1.0, "log_prob_dump_dir": None}
    d.update(kw); return Cfg(d)

recs = []
for f in sorted(glob.glob("results/rd_dump/*.npz"))[:24]:
    z = np.load(f)
    if "lp_stu" in z and "lp_pos" in z:
        recs.append((z["lp_stu"], z["lp_pos"], z["lp_null"]))
B = len(recs); T = max(r[0].shape[0] for r in recs); K = recs[0][0].shape[1]
def pad(a):
    out = np.full((B, T, K), -30.0, np.float32)
    for i, x in enumerate(a): out[i, :x.shape[0]] = x
    return torch.from_numpy(out)
stu, pos, nul = pad([r[0] for r in recs]), pad([r[1] for r in recs]), pad([r[2] for r in recs])
mask = torch.zeros(B, T)
for i, r in enumerate(recs): mask[i, :r[0].shape[0]] = 1.0
lp = torch.randn(B, T) * 0.1 - 1.0

def run(c):
    return compute_self_distillation_loss(
        student_log_probs=lp, teacher_log_probs=lp.clone(), response_mask=mask,
        self_distillation_config=c, old_log_probs=lp.clone(),
        student_topk_log_probs=stu, teacher_topk_log_probs=pos,
        teacher_null_topk_log_probs=nul,
        self_distillation_mask=torch.ones(B), loss_agg_mode="token-mean")

print(f"B={B} rollout, T={T}, K={K}, 有效 token={int(mask.sum())}")
for name, c in (("A (γ=1)", cfg()),
                ("floor α=0.1", cfg(counterfactual_floor_alpha=0.1)),
                ("tanh τ=0.5", cfg(counterfactual_tanh_scale=0.5)),
                ("γ=50 硬标签", cfg(counterfactual_target_gamma=50.0))):
    _, m = run(c)
    print(f"  {name:<14} max_prob={m['self_distillation/target_max_prob']:.4f}  "
          f"entropy={m['self_distillation/target_entropy']:.4f}  "
          f"tv={m['self_distillation/counterfactual_target_tv']:.4f}  "
          f"jsd={m['self_distillation/raw_jsd_token_mean']:.4f}")
print("  训练日志里 A 族: target_tv≈0.31, raw_jsd≈0.19 (用于判断口径是否对齐)")

# --- 真实数据的尾桶质量: 决定 γ=50 的 q_max 上界 ---
import torch.nn.functional as F
def add_tail_t(x):
    s = torch.logsumexp(x, -1, keepdim=True).clamp(max=-1e-7)
    return torch.cat([x, torch.log(-torch.expm1(s))], -1)
lp_p = add_tail_t(pos); lp_n = add_tail_t(nul)
u = lp_p - lp_n
logq = F.log_softmax(lp_p + 4.0 * u, -1)
q = logq.exp()
m3 = mask.unsqueeze(-1) > 0
tail = q[..., -1][mask > 0]
print(f"\n  真实张量上 (K=100, n={int(mask.sum())} tok):")
print(f"    尾桶质量  均值={tail.mean():.5f}  p50={tail.median():.5f}  p90={tail.quantile(0.9):.5f}")
print(f"    => γ=50 时 q_max 的上界 ≈ 1-尾桶 = {1-tail.mean():.5f}")
print(f"    argmax 落在尾桶的比例 = {100*(q[mask>0].argmax(-1)==q.shape[-1]-1).float().mean():.2f}%")
