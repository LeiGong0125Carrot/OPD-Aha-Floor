# nhC 第一次跑 case study (2026-10-02)

TB 峰 47.90 (47.16/44.94/47.90), V* 峰 91.10。池化口径 = 各臂所有 {30,40,50} ckpt 逐题平均。

## 1. 正确的参照是 B 不是 A
| 臂 | ckpt 数 | TB 池化准确率 |
|---|---|---|
| A (ahaAn2/rr1/rr2) | 9 | 49.2 |
| B (supn2/supn2r2) | 6 | 47.7 |
| C (nhC r1) | 3 | 46.7 |
C−B = −1.0 (噪声内); C 峰 47.90 落在 B 两次峰 50.37/47.41 之间。"C 差"的大部分来自 suppression-only 底座本身就低于 A。

## 2. 类别迁移 (C−A, 单次跑, 类别 SE≈5pt, 仅提示性)
升: Perception/Attributes +9.2, Physical State +6.8
降: Spatial Containment −8.4, Ordering −7.2, OCR −5.9, Contact&Occlusion −5.4
→ 更强抑制偏向局部属性识别, 伤关系/结构推理。

## 3. 输出形态无变化
平均词数 188 (=A), 字母分布与 A/B 同, 无效字母 2.1%, 疑截断 2.1% → 不是格式塌陷。

## 4. 训练动力学: C ≈ "加大剂量的 B"
target_tv 0.083–0.090 vs B 0.066–0.070 (+~27%); raw_jsd 略高; grad_norm/max_prob/长度同 B。
β_t 均值 6.6→6.1, 历史状态几十 token 即饱和 → β_t 在轨迹内近似常数,
C 实际上主要是**剂量**变化, 不是时间自适应。

## 5. 翻转题 (A 稳对 9/9 而 C 错): 21 题, 反向 11 题
典型模式 = 自信地编造细节: q85 "holding a camera" (GT: 没在拍照); q154 "small but distinct
space" (GT: 手指直接接触)。

## 建议
用"B + 常数 β=6.3"做对照, 分离"剂量"与"历史信息": 若常数版 ≈ C, 历史模块无独立贡献。
