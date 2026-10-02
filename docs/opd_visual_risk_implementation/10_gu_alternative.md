# 10｜Visual GU：独立的局部负向学习对照

**目的：** 保留已解释的另一条优化路线，用于与temporal-risk PPO区分。
**默认不启用：** history Phi、future returns、LOO advantage、PPO clipping。GU与主分支互斥。
**来源状态：** 以下GU构造来自本轮讨论 [C]；这里作为可独立推导的候选视觉迁移，不复述外部NSD性能。

## 1. Gate

$$G_t=[p_t^0(y_t)-p_t^+(y_t)]_+.$$

取实际生成token的完整teacher概率差。Real=.60、null=.80→G=.20。Real=.30、null=.10→G=0。不做top-k排名、不采样新的候选。

## 2. Loss

$$L_{\rm GU,t}=\frac{G_t}{2-p_t^S(y_t)}.$$

s=$p_t^S(y_t)$ 是当前student在配置明确的完整raw分布中的概率。Gate固定，s保留梯度。

G=.20时：s=.70→L=.153846；s=.40→L=.125；s=.90→L=.181818。最小化该项会压低gate激活的token。

其来源可直接化简：

$$\sigma[-\log(1-s)]=\frac1{1+\exp[\log(1-s)]}=\frac1{2-s}.$$

分母中的2不是额外训练超参数。Loss在s→0时趋向G/2而不是0，但固定G时减去G/2不改变梯度。

## 3. 与主线的接口差异

GU直接使用gate→local loss，没有比较后续路径“相对更好”。所有激活位置均产生局部负向作用；gate为0无该项梯度。若加anchor，写：

$$L=L_{\rm GU}+\lambda\operatorname{JSD}(p^+,p^S).$$

不能默认写GU+PPO+Phi同时训练，并称仍是已解释过的方案。若将G替换为future return，也已经改变GU监督含义，不属于本实施分支。

## 4. Reduction与数值

默认对所有有效response位置求平均，而不是只对G>0位置求平均。后者会改变稀疏gate的有效尺度，应另命名。

用闭式G/(2-exp(logpS))避免先计算-log(1-s)在s接近1时的中间数值问题。fp32下运算；真实logpS必须<=0。Invalid/pad先mask，不使用0*NaN。

## 5. 为什么不能保证解决循环

对sampled-token logit z 的导数为：

$$\frac{\partial L}{\partial z_y}=G\frac{s(1-s)}{(2-s)^2}.$$

s→1或s→0时导数都趋近0。保护高置信度结构的同时，可能难以改变高置信度错误。

如果重复的“red”在real条件反而更被支持，G=0，循环可能完全没有GU惩罚。Loss有界不意味着训练行为不会塌缩。

## 6. 验收与对照公平性

Gate detach、student有梯度；gate0loss/grad0；s增大loss增大；sigmoid与闭式一致；不因最小概率筛选而排除head错误。

保持与主线相同的两视图、数据、初始化、rollout预算和评估。GU没有LOO需要，不应因此私自修改每题rollout数让成本比较失真。

**参考实现：** `gu_loss`。**配置分支：** `mode: gu_local`。
