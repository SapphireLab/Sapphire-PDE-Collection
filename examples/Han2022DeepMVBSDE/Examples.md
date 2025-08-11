# [DeepMVBSDE](../../docs/BSDE/Han2022DeepMVBSDE.md)

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

### DeepMVBSDE 算法

相应的分布依赖函数定义为：

$$
\begin{aligned}
m_1(t,x,\mathcal{L}(\Theta_t)) &= \tilde{\mathbb{E}}_{x_t'\sim \mathcal{L}(X_t)} e^{-\dfrac{\|x- x'\|^2}{d}} +\dfrac{1}{2} m_t^Y\\
m_2 &= 0\\
m_3 &= 0\\
\end{aligned}
$$

- 神经网络: 两层隐藏层 + ReLU 激活函数的全连接神经网络, 用于近似解和 $\mathbf{m}_1$;
- $\mathcal{L}(\Theta_t)$: 状态分布.

### 实验 1

网络隐藏层宽度 24 + $N$: 仿真轨迹数, 取值 1500 + 虚拟博弈 (Fictitious Play) 30 个阶段:

| 基本算法 | 问题维度 $d$ | $Y_0$ 绝对误差 | $m_t^Y$ L2 误差 |
| --- | --- | --- | --- |
| DeepBSDE | 5 | 0.54% | 0.43% |
| DeepBSDE | 10 | 1.01% | 0.80% |
| DBDP | 5 | 1.60% | 0.92% |
| DBDP | 10 | 1.75% | 1.09% |

- 结论: DeepBSDE 比 DBDP 的性能稍好, 符合 [^Germain2019Numerical] 给出的结论.

### 实验 2

根据维度修改网络宽度和仿真轨迹数, 虚拟博弈 (Fictitious Play) 80 个阶段, 学习率调度和其他超参数不变:

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
