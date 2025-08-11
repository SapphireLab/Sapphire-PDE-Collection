# [DeepMVBSDE](../../docs/BSDE/Han2022DeepMVBSDE.md)

## 问题描述

**McKean-Vlasov Forward Backward Stochastic Differential Equations (MV-FBSDEs)** 具有如下形式:
$$
\begin{cases}
\text{d} X_t = b(t, \Theta_t, \mathcal{L}(\Theta_t)) \text{d} t + \sigma(t, \Theta_t, \mathcal{L}(\Theta_t)) \text{d} W_t,  &X_0 = x_0,  \\
\text{d} Y_t = -h(t, \Theta_t, \mathcal{L}(\Theta_t)) \text{d} t + Z_t \text{d} W_t, &Y_T = g(X_T, \mathcal{L}(X_T)),
\end{cases}
$$

- $\mathcal{L}(\cdot)$: 表示随机过程的边缘概率分布 (Marginal Law);
- $(b, \sigma, h, g)$: 具有兼容维数 (Compatible Dimension, 实现某种运算需要满足的维度条件) 的可测函数.
- $\Theta_t=(X_t, Y_t, Z_t)$: 后续简化符号.

当 $\sigma$ 和 $Z_t$, $\mathcal{L}(Z_t)$ 无关时, 在合适条件下, 上述 MV-FBSDEs 具有解耦场的解:

$$
Y_t = u(t, X_t), \quad Z_t = v(t, X_t) = \partial_x u(t,x) \sigma(t, x, u(t, x), (I_d, u(t, \cdot))(\mathcal{L}(X_t)))
$$

解耦场 (Decoupling Field): 解耦了 $Y_t$ 和 $Z_t$ 对其他变量的依赖.

## 算法描述

DeepMVBSDE 算法结合了**虚拟博弈 (Fictitious Play)** 思想和现有的 BSDE 求解器 (DeepBSDE, DBDP), 用于求解具有一般分布依赖性的 MV-FBSDEs.

上述问题求解时会面临两个问题:
1. 分布 $\mathcal{L}$ 先验未知, 需要通过不动点来确定 → 虚拟博弈思想;
2. 如何建模系数函数关于分布 $\mathcal{L}$ 的依赖关系 → 引入中间变量.

DeepMVBSDE 假设系数函数 $(b, \sigma, h, g)$ 具有如下结构, 用于简化复杂的依赖关系建模:

$$
\begin{aligned}
b(t, \Theta_t, \mathcal{L}(\Theta_t)) &:= b(t, \Theta_t, \textcolor{red}{m_1(t,X_t,\mathcal{L}(\Theta_t))})\\
\sigma(t, \Theta_t, \mathcal{L}(\Theta_t)) &:= \sigma(t,X_t)\\
h(t,\Theta_t,\mathcal{L}(\Theta_t)) &:= h(t,\Theta_t, \textcolor{blue}{m_2(\mathcal{L}(x, X_t, \Theta_t))})\\
g(X_T, \mathcal{L}(X_T)) &:= g(X_T, \textcolor{green}{m_3(X_T, \mathcal{L}(X_T))})
\end{aligned}
$$

- $\textcolor{red}{m_1}$: $[0,T]\times \mathbb{R}^d \times \mathcal{P}^2(\mathbb{R}^{\theta})\to \mathbb{R}^l$ 向量值函数;
- $\textcolor{blue}{m_2}$: $[0,T]\times \mathbb{R}^d \times \mathcal{P}^2(\mathbb{R}^{\theta})\to \mathbb{R}^l$ 向量值函数;
- $\textcolor{green}{m_3}$: $\mathbb{R}^d \times \mathcal{P}^2(\mathbb{R}^{\theta})\to \mathbb{R}^l$ 向量值函数.

由于 $\mathcal{L}$ 未知, 结合虚拟博弈的思想, 将原问题转化为不动点迭代求解过程.
每次迭代分为三步: 1. 更新经验状态分布; 2. 更新分布依赖函数; 3. 求解标准 FBSDEs.

### 1. 更新经验状态分布

在第 $(k)$ 次虚拟博弈迭代时, 根据前一次迭代解 $Y_0\approx u^{(k-1)}(0, X_0)$ 和 $Z_t\approx v^{(k-1)}(t,X_t)$, 和分布依赖函数 $\textcolor{red}{\hat{m}_1^{(k-1)}}$, $\textcolor{blue}{\hat{m}_2^{(k-1)}}$, $\textcolor{green}{\hat{m}_3^{(k-1)}}$, 通过 SDE 获得轨迹 $(\tilde{X}_t^{(k-1)}, Y_t^{(k-1)})$:

$$
\begin{aligned}
\text{d} \tilde{X}_t^{(k-1)} &= b(t, \tilde{\Theta}_t^{(k-1)}, \textcolor{red}{\hat{m}_1^{(k-1)}}) \text{d} t + \sigma(t, \tilde{X}_t) \text{d} W_t,  &\tilde{X}_0^{(k-1)} &= x_0,  \\
\text{d} \tilde{Y}_t^{(k-1)} &= -h(t, \tilde{\Theta}_t^{(k-1)}, \textcolor{blue}{\hat{m}_2^{(k-1)}}) \text{d} t + v^{(k-1)}(t, \tilde{X}_t^{(k-1)}) \text{d} W_t, &\tilde{Y}_0^{(k-1)} &= u^{(k-1)}(0, \tilde{X}_0^{(k-1)}),
\end{aligned}
$$

然后更新 $\textcolor{orange}{\mathcal{L}(\Theta_t)\approx \nu_t^{(k-1)} = \mathcal{L}(\tilde{\Theta}_t^{(k-1)})}$ 和 $\textcolor{purple}{\mathcal{L}(X_T)\approx \mu_T^{(k-1)} = \mathcal{L}(\tilde{X}_T^{(k-1)})}$.

### 2. 更新分布依赖函数

上一步更新了 $\mathcal{L}(\Theta_t)$ 和 $\mathcal{L}(X_T)$, 固定该结果, 使用监督学习实现分布依赖函数的学习:

$$
\begin{aligned}
\inf_{\hat{m}_1} &\int_0^T \mathbb{E} \| m_1(t, \tilde{X}_t^{(k-1)}, \textcolor{orange}{\nu_t^{(k-1)}}) - \hat{m}_1(t, \tilde{X}_t^{(k-1)})\|^2,\\
\inf_{\hat{m}_2} &\int_0^T \mathbb{E} \| m_2(t, \tilde{X}_t^{(k-1)}, \textcolor{orange}{\nu_t^{(k-1)}}) - \hat{m}_2(t, \tilde{X}_t^{(k-1)})\|^2,\\
\inf_{\hat{m}_3} &\mathbb{E} \| m_3(\tilde{X}_T^{(k-1)}, \textcolor{purple}{\mu_T^{(k-1)}}) - \hat{m}_3(\tilde{X}_T^{(k-1)})\|^2.
\end{aligned}
$$

优化后得到 $\textcolor{red}{\hat{m}_1^{(k)}}$, $\textcolor{blue}{\hat{m}_2^{(k)}}$, $\textcolor{green}{\hat{m}_3^{(k)}}$.

不采用神经网络近似时, 可以采用经验分布估计 (Monte Carlo) 获得估计, 不过非常耗时.

### 3. 求解标准 FBSDEs

根据前一步的分布依赖函数求解标准 FBSDEs:

$$
\begin{cases}
\text{d} X_t^{(k)} = b(t, \Theta_t^{(k)}, \textcolor{red}{\hat m_1^{(k)}(t, X_t^{k})})\text{d} t + \sigma(t, X_t^{(k)})\text{d} W_t, & X^{(k)}_0 = x_0,\\
\text{d} Y_t^{(k)} = - h(t, \Theta_t^{(k)}, \textcolor{blue}{\hat m_2^{(k)}(t, X_t^{k})})\text{d} t + Z^{(k)}_t\text{d} W_t, & Y^{(k)}_T= g(X^{(k)}_T, \textcolor{green}{\hat m_3^{(k)}(X^{k}_T)}),
\end{cases}
$$

使用 DeepBSDE 或 DBDP 算法求解解函数 $Y_0 \approx u^{(k)}(0, X_0; \psi^{(k)})$ 和 $Z_t \approx v^{(k)}(t,X_t;\phi^{(k)})$.

---

注: 第二步和第三步采用的轨迹不同导致分布不匹配, 可能导致理论和数值困难. 这也是设置简化分布函数的原因之一, 在该假设下, $X_t^{(k)}$ 的分布和 $X_t^{(k-1)}$ 的分布可以由 Girsanov 定理确保绝对连续, 从而可以通过前一步的误差来控制下一步的误差.
还可以使用替换轨迹确保每步的分布匹配, 但要获得第 k 步的轨迹, 得先在相同的布朗运动下仿真前 $0~k-1$ 步, 计算并不高效. ?

## 算例 1: 具有显式解的基准算例

对于 $(X_t, Y_t, Z_t)$ 考虑如下 MV-FBSDEs:

$$
\begin{dcases}
\text{d} X_t^i = \left[\sin(\textcolor{red}{\tilde{\mathbb{E}}_{x_t'\sim\mu_t} e^{-\dfrac{\|X_t- x_t'\|^2}{d}}} - e^{-\dfrac{\|X_t\|^2}{d+2t}}\Big(\dfrac{d}{d+2t}\Big)^\dfrac{d}{2})+ \dfrac{1}{2}(\textcolor{blue}{m_t^Y} - \sin(t) e^{-\dfrac{t}{2}}) \right]\text{d} t + \text{d} W_t^i, \\
\text{d} Y_t = \left[\frac{Z_t^1 + \dots + Z_t^d}{\sqrt{d}}- \frac{Y_t}{2}+ \sqrt{Y_t^2 + \|Z_t\|^2 + 1}-\sqrt{2}\right]\text{d} t + Z_t\text{d} W_t, \\
X_0^i = 0,\\
Y_T = \sin(T + \dfrac{X_T^1 + \dots + X_T^d}{\sqrt{d}}),\\
\end{dcases}
$$

- $X_t^i$: $d$ 维前向过程 $X_t$ 的第 $i$ 个分量;
- $\mu_t=\mathcal{L}(X_t)$;
- $\textcolor{blue}{m_t^Y}=\mathbb{E}[Y_t]$: **根据真解该值为 0**;
- $p=1$;
- $q=d$;
- $\tilde{\mathbb{E}}$ 只和 $x_t'\sim\mu_t$ 有关;

相应的解为：
$$
\begin{aligned}
X_t &= W_t;\\
Y_t &= \sin(t + \dfrac{X_t^1 + \dots + X_t^d}{\sqrt{d}});\\
Z_t &= \dfrac{1}{\sqrt{d}}\cos(t + \dfrac{X_t^1 + \dots + X_t^d}{\sqrt{d}})
\end{aligned}
$$

问题配置:
- $d$: 问题维度, 取值 5, 10;
- $T$: 终止时刻, 取值 0.5;
- $N_t$: 离散时间区间数, 取值 20;

---

DeepMVBSDE 算法的相应的分布依赖函数定义为：

$$
\begin{aligned}
m_1(t,x,\mathcal{L}(\Theta_t)) &= \tilde{\mathbb{E}}_{x_t'\sim \mathcal{L}(X_t)} e^{-\dfrac{\|x- x'\|^2}{d}} +\dfrac{1}{2} m_t^Y\\
m_2 &= 0\\
m_3 &= 0\\
\end{aligned}
$$

### 实验 1

两层宽度为 24 的隐藏层 + ReLU 激活函数的全连接神经网络 + $N$: 仿真轨迹数, 取值 1500 + 虚拟博弈迭代 30 次:

| 基本算法 | 问题维度 $d$ | $Y_0$ 绝对误差 | $m_t^Y$ L2 误差 |
| --- | --- | --- | --- |
| DeepBSDE | 5 | 0.54% | 0.43% |
| DeepBSDE | 10 | 1.01% | 0.80% |
| DBDP | 5 | 1.60% | 0.92% |
| DBDP | 10 | 1.75% | 1.09% |

- 结论: DeepBSDE 比 DBDP 的性能稍好, 符合 [^Germain2019Numerical] 给出的结论.

### 实验 2

根据问题维度修改网络宽度和仿真轨迹数, 虚拟博弈迭代 80 次, 学习率调度和其他超参数不变:

| 基本算法 | 问题维度 $d$ | 网络宽度 | 仿真轨迹数 $N$ | $Y_0$ 绝对误差 |
| --- | --- | --- | --- | --- |
| DeepBSDE | 5 | 12 | 500 | < 0.01 |
| DeepBSDE | 8 | 18 | 800 | < 0.02 |
| DeepBSDE | 10 | 24 | 1000 | < 0.02 |
| DeepBSDE | 12 | 30 | 1200 | < 0.02 |
| DeepBSDE | 15 | 36 | 1500 | < 0.02 |

- 结论:
  1. 在保持合适的计算成本增长下可以保持性能;
  2. 维度越大需要的虚拟博弈越多, 但增长并不剧烈.

[^Germain2019Numerical]: [Numerical Resolution of McKean-Vlasov FBSDEs Using Neural Networks.](../../docs/BSDE/Germain2019Numerical.md)