#!/usr/bin/env python
"""γ-sharpening 臂单测 —— 7 组, 全 CPU。

判据见 docs/gamma_sharpening_plan.md:
  log q_γ = log_softmax(γ·log q), 加在 tilt/floor/sup/ST **全部算完之后**。
  γ=1 必须 bit-identical; γ 大 -> onehot(argmax q); γ<1 -> 更平。

组 5 是**真数值对拍**(独立参照实现 -> JSD 标量 vs 生产 loss)。
上次 tanh 的 code-review 抓到我把方向性断言当成对拍, 这次一开始就写真的。
"""
import math, os, sys, torch, torch.nn.functional as F

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from verl.trainer.ppo.core_algos import compute_self_distillation_loss  # noqa: E402

FAIL = []
def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"   {detail}" if detail else ""))
    if not cond: FAIL.append(name)


class Cfg(dict):
    __getattr__ = dict.get


def cfg(**kw):
    d = {"full_logit_distillation": True, "distillation_topk": 8, "distillation_add_tail": True,
         "renorm_topk_log_probs": False, "alpha": 0.5, "is_clip": None,
         "counterfactual_null_mode": "mean_color", "counterfactual_extrapolation_beta": 4.0,
         "counterfactual_u_clip_pos": False, "counterfactual_st_enable": False,
         "counterfactual_floor_alpha": 0.0, "counterfactual_tanh_scale": 0.0,
         "counterfactual_target_gamma": 1.0, "log_prob_dump_dir": None}
    d.update(kw); return Cfg(d)


def add_tail(x):
    s = torch.logsumexp(x, -1, keepdim=True).clamp(max=-1e-7)
    return torch.cat([x, torch.log(-torch.expm1(s))], -1)


def ref_target(log_h, log_f, beta, gamma=1.0, clip_pos=False, floor_alpha=0.0):
    """独立参照实现: tilt (+可选 floor/sup) -> 再 γ。不调用生产代码。"""
    u = log_h - log_f
    if floor_alpha > 0:
        valley = log_h < math.log(floor_alpha) + log_h.max(dim=-1, keepdim=True).values
        u = torch.where(valley & (u < 0), torch.zeros_like(u), u)
    elif clip_pos:
        u = torch.clamp(u, max=0.0)
    log_q = torch.log_softmax(log_h + beta * u, dim=-1)
    if gamma != 1.0:
        # 只锐化显式列, 尾桶(最后一列)的质量原样保留 —— 见 core_algos 的同名注释
        exp_ = torch.ones_like(log_q, dtype=torch.bool); exp_[..., -1] = False
        ninf = torch.finfo(log_q.dtype).min
        mass = torch.logsumexp(log_q.masked_fill(~exp_, ninf), -1, keepdim=True)
        sc = gamma * log_q
        z = torch.logsumexp(sc.masked_fill(~exp_, ninf), -1, keepdim=True)
        log_q = torch.where(exp_, sc - z + mass, log_q)
    return log_q


def jsd_ref(log_q, log_s, alpha=0.5):
    q, s = log_q.exp(), log_s.exp()
    m = alpha * s + (1 - alpha) * q
    logm = (m + 1e-30).log()
    return (alpha * (s * (log_s - logm)).sum(-1) + (1 - alpha) * (q * (log_q - logm)).sum(-1)).mean()


print("=" * 74); print("γ-sharpening 臂单测"); print("=" * 74)
BETA = 4.0

# ---- 共用 fixture (与 test_tanh/test_floor 同构) ----
torch.manual_seed(11)
B, T, K = 2, 4, 8
def topk_logps():
    return torch.log_softmax(torch.randn(B, T, K + 5), -1).sort(-1, descending=True).values[..., :K]
student0, real0, null0 = topk_logps(), topk_logps(), topk_logps()
lp = torch.randn(B, T) * 0.1 - 1.0
mask = torch.ones(B, T)
log_h, log_f, log_s = add_tail(real0), add_tail(null0), add_tail(student0)
def run(c):
    return compute_self_distillation_loss(
        student_log_probs=lp, teacher_log_probs=lp.clone(), response_mask=mask,
        self_distillation_config=c, old_log_probs=lp.clone(),
        student_topk_log_probs=student0, teacher_topk_log_probs=real0,
        teacher_null_topk_log_probs=null0,
        self_distillation_mask=torch.ones(B), loss_agg_mode="token-mean")

# ---- 1. γ=1 恒等 ----
print("\n[1] γ=1 恒等 (A 臂回归保护)")
l_base, m_base = run(cfg())
l_one, m_one = run(cfg(counterfactual_target_gamma=1.0))
check("γ=1 与不设时 bit-identical", torch.allclose(l_base, l_one, atol=0))
c_absent = cfg(); del c_absent["counterfactual_target_gamma"]
l_abs, _ = run(c_absent)
check("缺 key 回落到 γ=1 (保护旧 config/ckpt)", torch.allclose(l_base, l_abs, atol=0))
check("γ=1 时参照实现也一致",
      abs(l_base.item() - float(jsd_ref(ref_target(log_h, log_f, BETA), log_s))) < 1e-5)

# ---- 2. 大 γ -> 硬标签 ----
print("\n[2] 大 γ -> onehot(argmax q)")
q1 = ref_target(log_h, log_f, BETA)
q50 = ref_target(log_h, log_f, BETA, gamma=50.0)
# 尾桶质量守恒后, 语义是"在显式列内部锐化":
K_idx = q1.shape[-1] - 1                      # 尾桶列
was_tail = q1.argmax(-1) == K_idx
check("argmax 在显式列时位置不变 (只改锐度, 不改偏向谁)",
      bool((q1.argmax(-1) == q50.argmax(-1))[~was_tail].all()),
      f"{int((~was_tail).sum())} 个位置的 argmax 本就在显式列")
check("argmax 原在尾桶时会移入显式列 (符合设计: 不要对'不在 top-k'下硬标签)",
      bool((q50.argmax(-1) != K_idx)[was_tail].all()) if was_tail.any() else True,
      f"{int(was_tail.sum())} 个位置原本 argmax=尾桶")
tail_mass = q1[..., K_idx].exp()
ratio = q50.exp().amax(-1) / (1 - tail_mass).clamp(min=1e-9)
# 硬不变式: 尾桶质量守恒 + argmax 在显式列 => q_max <= 1 - tail
check("硬不变式 q_max ≤ 1−尾桶质量", bool((ratio <= 1.0 + 1e-6).all()),
      f"最大 ratio = {ratio.max():.6f}")
# 多数位置会塌成 one-hot; 例外是 top-2 近似并列处(已知性质, 见 §一)
check("多数位置在显式列内塌成 one-hot (中位数口径)", float(ratio.median()) > 0.999,
      f"ratio 中位数 {ratio.median():.6f}, 最小 {ratio.min():.4f} (最小者=近似并列处)")
check("锐度显著提升", q50.exp().amax(-1).mean() > q1.exp().amax(-1).mean() + 0.3,
      f"q_max 均值 {q1.exp().amax(-1).mean():.4f} -> {q50.exp().amax(-1).mean():.4f} "
      f"(上界 1−尾桶 = {1-float(tail_mass.mean()):.4f})")
# **已知性质(不是 bug)**: 任何有限 γ 都压不硬"近似并列"的位置 ——
# 若 top-2 在 γ=1 时比值为 1+ε, 则 γ 次幂后仍是 (1+ε)^γ, ε 很小时仍接近并列。
# 本 fixture 是随机 top-8 (q_max 仅 0.4686, 远比真实训练平), 故存在这类位置。
worst = q50.exp().amax(-1).min().item()
check("近似并列位置会保留软质量 (已知性质, 需写入文档)", worst < 1.0,
      f"最硬位 1.000 / 最软位 {worst:.6f} —— 后者是 top-2 近似并列处")
# 真实训练的量级: q_max=0.9824 (实测), 次高至多 1-0.9824=0.0176
r = (0.9824 / 0.0176) ** 50
check("真实训练量级下 γ=50 确是硬标签", r > 1e80,
      f"首/次 token 比 ≈ {r:.1e} (用实测 q_max=0.9824)")
# 直接在"真实锐度"的 fixture 上验证
real_like = torch.log(torch.tensor([[[0.9824, 0.0100, 0.0040, 0.0020, 0.0016]]]))
rl50 = torch.log_softmax(50.0 * torch.log_softmax(real_like, -1), -1)
check("真实锐度 fixture 上 γ=50 -> q_max > 0.9999", rl50.exp().amax(-1).min().item() > 0.9999,
      f"q_max = {rl50.exp().amax(-1).min().item():.8f}, 熵 = {(-(rl50.exp()*rl50).sum(-1)).max().item():.2e}")

# ---- 3. γ<1 -> 更平 ----
print("\n[3] γ<1 -> 更平 (反方向)")
q05 = ref_target(log_h, log_f, BETA, gamma=0.5)
check("q_max 下降", q05.exp().amax(-1).mean() < q1.exp().amax(-1).mean(),
      f"{q1.exp().amax(-1).mean():.4f} -> {q05.exp().amax(-1).mean():.4f}")
check("熵上升", (-(q05.exp()*q05).sum(-1)).mean() > (-(q1.exp()*q1).sum(-1)).mean(),
      f"{(-(q1.exp()*q1).sum(-1)).mean():.4f} -> {(-(q05.exp()*q05).sum(-1)).mean():.4f}")

# ---- 4. 数值安全 ----
print("\n[4] 数值安全 (硬标签不会炸)")
for g in (0.1, 1.0, 10.0, 50.0, 500.0):
    lg = ref_target(log_h, log_f, BETA, gamma=g)
    ok = bool(torch.isfinite(lg).all()) and abs(float(lg.exp().sum(-1).mean()) - 1.0) < 1e-5
    check(f"γ={g:<5} 有限且归一", ok)
# JSD 对 one-hot 目标的上界 log2 (计划里推导过)
l50, _ = run(cfg(counterfactual_target_gamma=50.0))
check("硬标签下 loss ≤ log2 (有界, 无 log(0))", l50.item() <= math.log(2) + 1e-6,
      f"loss={l50.item():.6f}  log2={math.log(2):.6f}")

# ---- 5. 生产函数数值对拍 ----
print("\n[5] 生产函数数值对拍")
for g in (0.5, 10.0, 50.0):
    l, _ = run(cfg(counterfactual_target_gamma=g))
    r = float(jsd_ref(ref_target(log_h, log_f, BETA, gamma=g), log_s))
    check(f"γ={g:<5} loss 与参照一致 (<1e-5)", abs(l.item() - r) < 1e-5,
          f"生产 {l.item():.8f} vs 参照 {r:.8f}")
# 剂量对拍: 若 γ 误加在 log_h 上(而非最终 q), 必须被抓住
wrong = float(jsd_ref(torch.log_softmax(50.0 * log_h + BETA * (log_h - log_f), -1), log_s))
l50v = run(cfg(counterfactual_target_gamma=50.0))[0].item()
check("能区分'γ 误加在 p⁺ 上'的实现", abs(l50v - wrong) > 1e-3,
      f"正确 {l50v:.6f} vs 错版 {wrong:.6f}")

# ---- 6. 指标 ----
print("\n[6] 新增指标")
_, m50 = run(cfg(counterfactual_target_gamma=50.0))
_, m05 = run(cfg(counterfactual_target_gamma=0.5))
for k in ("target_top1_prob", "target_entropy"):
    check(f"{k} 在 γ=1 也上报 (可横向对照)", f"self_distillation/{k}" in m_base)
q_ref = ref_target(log_h, log_f, BETA, gamma=50.0).exp()
check("target_top1_prob 与手算一致",
      abs(m50["self_distillation/target_top1_prob"] - float(q_ref.amax(-1).mean())) < 1e-5,
      f"{m50['self_distillation/target_top1_prob']:.6f}")
lq = ref_target(log_h, log_f, BETA, gamma=50.0)
check("target_entropy 与手算一致",
      abs(m50["self_distillation/target_entropy"] - float((-(lq.exp()*lq).sum(-1)).mean())) < 1e-5)
check("γ=50 熵 < γ=1 熵 < γ=0.5 熵 (单调)",
      m50["self_distillation/target_entropy"] < m_base["self_distillation/target_entropy"]
      < m05["self_distillation/target_entropy"],
      f"{m50['self_distillation/target_entropy']:.4f} < "
      f"{m_base['self_distillation/target_entropy']:.4f} < "
      f"{m05['self_distillation/target_entropy']:.4f}")

# ---- 7. 组合与互斥 ----
print("\n[7] 组合与互斥")
l_fl, _ = run(cfg(counterfactual_target_gamma=50.0, counterfactual_floor_alpha=0.1))
check("γ + floor 与参照一致 (γ 加在 floor 之后)",
      abs(l_fl.item() - float(jsd_ref(ref_target(log_h, log_f, BETA, gamma=50.0, floor_alpha=0.1), log_s))) < 1e-5)
l_sp, _ = run(cfg(counterfactual_target_gamma=50.0, counterfactual_u_clip_pos=True))
check("γ + sup 与参照一致",
      abs(l_sp.item() - float(jsd_ref(ref_target(log_h, log_f, BETA, gamma=50.0, clip_pos=True), log_s))) < 1e-5)
from verl.workers.config.actor import SelfDistillationConfig as SDC  # noqa: E402
for kw, why in (({"counterfactual_st_enable": True}, "与 ST 互斥"),
                ({"counterfactual_target_gamma": -1.0}, "γ<=0 被拦"),
                ({"counterfactual_null_mode": None}, "无 null_mode 被拦")):
    base = {"counterfactual_null_mode": "mean_color", "counterfactual_target_gamma": 50.0}
    base.update(kw)
    ok = False
    try: SDC(**base)
    except Exception: ok = True
    check(why, ok)

# ---- 8. 尾桶不变性 与 运行时守卫 (code-review #4 / #2) ----
print("\n[8] 尾桶不变性 + 运行时守卫")
for g in (0.5, 50.0):
    lg = ref_target(log_h, log_f, BETA, gamma=g)
    l1 = ref_target(log_h, log_f, BETA)
    check(f"γ={g:<5} 尾桶质量不变 (它是截断产物, 不是 token)",
          torch.allclose(lg[..., -1], l1[..., -1], atol=1e-6),
          f"tail {l1[...,-1].exp().mean():.3e} -> {lg[...,-1].exp().mean():.3e}")
    check(f"γ={g:<5} 仍是合法分布", abs(float(lg.exp().sum(-1).mean()) - 1.0) < 1e-5)
# 若误把 γ 也作用到尾桶上, γ=0.5 会把尾桶抬高几个数量级 —— 这条要能抓住
naive = torch.log_softmax(0.5 * ref_target(log_h, log_f, BETA), dim=-1)
correct = ref_target(log_h, log_f, BETA, gamma=0.5)
check("能区分'γ 也作用于尾桶'的实现",
      not torch.allclose(naive[..., -1], correct[..., -1], atol=1e-4),
      f"错版 tail {naive[...,-1].exp().mean():.3e} vs 正确 {correct[...,-1].exp().mean():.3e}")
# 生产路径对拍(含尾桶处理)
for g in (0.5, 50.0):
    l, _ = run(cfg(counterfactual_target_gamma=g))
    r = float(jsd_ref(ref_target(log_h, log_f, BETA, gamma=g), log_s))
    check(f"γ={g:<5} 生产与参照一致(含尾桶口径)", abs(l.item() - r) < 1e-5,
          f"{l.item():.8f} vs {r:.8f}")
# 运行时守卫: dataclass 校验在本仓库训练路径上是死代码, 必须在 loss 函数里拦
for bad, why in ((-1.0, "γ=-1 (会反转目标)"), (0.0, "γ=0 (会退化成恒等)")):
    ok = False
    try: run(cfg(counterfactual_target_gamma=bad))
    except ValueError as e: ok = "must be > 0" in str(e)
    check(f"运行时拦住 {why}", ok)
ok = False
try: run(cfg(counterfactual_target_gamma=50.0, counterfactual_st_enable=True))
except ValueError as e: ok = "incompatible" in str(e)
check("运行时拦住 γ + ST 同时开", ok)

print("\n" + "=" * 74)
print("全部通过" if not FAIL else "失败: " + ", ".join(FAIL))
sys.exit(1 if FAIL else 0)
