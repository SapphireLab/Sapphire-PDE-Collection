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

### 1 更新经验状态分布

在第 $(k)$ 次虚拟博弈迭代时, 根据前一次迭代解 $Y_0\approx u^{(k-1)}(0, X_0)$ 和 $Z_t\approx v^{(k-1)}(t,X_t)$, 和分布依赖函数 $\textcolor{red}{\hat{m}_1^{(k-1)}}$, $\textcolor{blue}{\hat{m}_2^{(k-1)}}$, $\textcolor{green}{\hat{m}_3^{(k-1)}}$, 通过 SDE 获得轨迹 $(\tilde{X}_t^{(k-1)}, Y_t^{(k-1)})$:

$$
\begin{aligned}
\text{d} \tilde{X}_t^{(k-1)} &= b(t, \tilde{\Theta}_t^{(k-1)}, \textcolor{red}{\hat{m}_1^{(k-1)}}) \text{d} t + \sigma(t, \tilde{X}_t) \text{d} W_t,  &\tilde{X}_0^{(k-1)} &= x_0,  \\
\text{d} \tilde{Y}_t^{(k-1)} &= -h(t, \tilde{\Theta}_t^{(k-1)}, \textcolor{blue}{\hat{m}_2^{(k-1)}}) \text{d} t + v^{(k-1)}(t, \tilde{X}_t^{(k-1)}) \text{d} W_t, &\tilde{Y}_0^{(k-1)} &= u^{(k-1)}(0, \tilde{X}_0^{(k-1)}),
\end{aligned}
$$

然后更新 $\textcolor{orange}{\mathcal{L}(\Theta_t)\approx \nu_t^{(k-1)} = \mathcal{L}(\tilde{\Theta}_t^{(k-1)})}$ 和 $\textcolor{purple}{\mathcal{L}(X_T)\approx \mu_T^{(k-1)} = \mathcal{L}(\tilde{X}_T^{(k-1)})}$.

### 2 更新分布依赖函数

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

### 3 求解标准 FBSDEs

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

对于 $(X_t, Y_t, Z_t)\in \mathbb{H}^2(\mathbb{R}^d\times \mathbb{R}^p \times \mathbb{R}^{p\times q})$ 考虑如下 MV-FBSDEs:

$$
\begin{cases}
\text{d} X_t^i = \left[\sin(\textcolor{red}{\tilde{\mathbb{E}}_{x_t'\sim\mu_t} e^{-\dfrac{\|X_t- x_t'\|^2}{d}}} - e^{-\dfrac{\|X_t\|^2}{d+2t}}\Big(\dfrac{d}{d+2t}\Big)^\dfrac{d}{2})+ \dfrac{1}{2}(\textcolor{blue}{m_t^Y} - \sin(t) e^{-\dfrac{t}{2}}) \right]\text{d} t + \text{d} W_t^i, \\
\text{d} Y_t = \left[\frac{Z_t^1 + \dots + Z_t^d}{\sqrt{d}}- \frac{Y_t}{2}+ \sqrt{Y_t^2 + \|Z_t\|^2 + 1}-\sqrt{2}\right]\text{d} t + Z_t\text{d} W_t, \\
X_0^i = 0,\\
Y_T = \sin(T + \dfrac{X_T^1 + \dots + X_T^d}{\sqrt{d}}),\\
\end{cases}
$$

- $X_t^i$: $d$ 维前向过程 $X_t$ 的第 $i$ 个分量;
- $\mu_t=\mathcal{L}(X_t)$;
- $\textcolor{blue}{m_t^Y}=\mathbb{E}[Y_t]$;
- $p=1$: $Y_t$ 维度;
- $q=d$: $Z_t$ 维度;
- $\tilde{\mathbb{E}}$ 只和 $x_t'\sim\mu_t$ 有关;

相应的解为：
$$
\begin{aligned}
X_t &= W_t, \\
Y_t &= \sin(t + \dfrac{X_t^1 + \dots + X_t^d}{\sqrt{d}}), \\
Z_t^i &= \dfrac{1}{\sqrt{d}}\cos(t + \dfrac{X_t^1 + \dots + X_t^d}{\sqrt{d}})
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

注: 代码中 $m_1$ 由两部分实现, 第一部分期望由 `drift_model` 预测, 第二部分 $m_t^Y$ 的实现是单独更新 `mean_y_estimate` 变量.

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

---

### 个人补充

#### $X_t$ 真解验证

真解 $X_t$ 为 $W_t$, 因此相应的随机微分方程中的漂移项应为 0, 下面验证该结论.

首先第一部分为:
$$
\sin(\textcolor{red}{\tilde{\mathbb{E}}_{x_t'\sim\mu_t} e^{-\dfrac{\|X_t- x_t'\|^2}{d}}} - e^{-\dfrac{\|X_t\|^2}{d+2t}}\Big(\dfrac{d}{d+2t}\Big)^\dfrac{d}{2})
$$

代入真解后 $\mu_t = \mathcal{L}(X_t) =\mathcal{N}(\mathbf{0}, t\mathbf{I}_d)$
然后展开指数项:
$$
\|W_t-x_t'\|^2 = \sum_{i=1}^{d} (W_t^i-x_t'^i)^2 = \sum_{i=1}^{d}[(W_t^i)^2-2W_t^ix_t'^i+(x_t'^i)^2]
$$

$$
e^{-\dfrac{\|X_t-x_t'\|^2}{d+2t}} = e^{-\dfrac{\|W_t\|^2}{d}}\cdot e^{\dfrac{2W_t^{\mathsf{T}} x_t'}{d}}\cdot e^{-\dfrac{\|x_t'\|^2}{d}}
$$

所以期望变为:
$$
\begin{aligned}
\textcolor{red}{\tilde{\mathbb{E}}_{x_t'\sim\mu_t} e^{-\dfrac{\|X_t- x_t'\|^2}{d}}}
&=\mathbb{E}_{x_t'}[e^{-\dfrac{\|W_t\|^2}{d}}\cdot e^{\dfrac{2W_t^{\mathsf{T}} x_t'}{d}}\cdot e^{-\dfrac{\|x_t'\|^2}{d}}]\\
&=e^{-\dfrac{\|W_t\|^2}{d}}\cdot \mathbb{E}_{x_t'}[e^{\dfrac{2W_t^{\mathsf{T}} x_t'}{d}-\dfrac{\|x_t'\|^2}{d}}]\\
\end{aligned}
$$

> 高斯线性变换积分公式:
> $\mathbf{x}\sim \mathcal{N}(\mathbf{0},\sigma^2\mathbf{I}_d)$ 是 $d$ 维高斯随机变量, $\mathbf{a}\in \mathbb{R}^d$ 是常数变量, $b>0$ 是常数, 则下述期望有:
$$
\mathbb{E}_{\mathbf{x}}[e^{\mathbf{a}^{\mathsf{T}}\mathbf{x}-b\| \mathbf{x}\|^2}]=\dfrac{1}{(1+2b\sigma^2)^{d/2}}\exp(\dfrac{\sigma^2 \|\mathbf{a}\|^2}{2(1+2b\sigma^2)}).
$$

<details>
<summary>推导</summary>
$$
p(\mathbf{x})=\dfrac{1}{(2\pi\sigma^2)^{d/2}}e^{-\dfrac{\|\mathbf{x}\|^2}{2\sigma^2}}
$$
>
> $$
\begin{aligned}
\mathbb{E}[e^{\mathbf{a}^{\mathsf{T}}\mathbf{x}-b\| \mathbf{x}\|^2}] &= \int e^{\mathbf{a}^{\mathsf{T}}\mathbf{x}-b\| \mathbf{x}\|^2}p(\mathbf{x}) \text{d}\mathbf{x}\\
&= \int \dfrac{1}{(2\pi\sigma^2)^{d/2}}e^{-\frac{\|\mathbf{x}\|^2}{2\sigma^2}} e^{\mathbf{a}^{\mathsf{T}}\mathbf{x}-b\| \mathbf{x}\|^2} \text{d}\mathbf{x}\\
&= \dfrac{1}{(2\pi\sigma^2)^{d/2}} \int e^{\mathbf{a}^{\mathsf{T}}\mathbf{x}-b\| \mathbf{x}\|^2-\frac{\|\mathbf{x}\|^2}{2\sigma^2}} \text{d}\mathbf{x}\\
&= \dfrac{1}{(2\pi\sigma^2)^{d/2}} \int e^{\mathbf{a}^{\mathsf{T}}\mathbf{x} + (-b -\frac{1}{2\sigma^2})\|\mathbf{x}\|^2} \text{d}\mathbf{x}\\
&= \dfrac{1}{(2\pi\sigma^2)^{d/2}} \int e^{-(b+\frac{1}{2\sigma^2})\|\mathbf{x}\|^2+\mathbf{a}^{\mathsf{T}}\mathbf{x}} \text{d}\mathbf{x}\\
&= \dfrac{1}{(2\pi\sigma^2)^{d/2}} \int e^{-(b+\frac{1}{2\sigma^2})\|\mathbf{x}-\frac{\mathbf{a}}{2(b+\frac{1}{2\sigma^2})}\|^2+\frac{\| \mathbf{a}\|^2}{4(b+\frac{1}{2\sigma^2})}} \text{d}\mathbf{x}\\
&= \dfrac{1}{(2\pi\sigma^2)^{d/2}}e^{\frac{\| \mathbf{a}\|^2}{4(b+\frac{1}{2\sigma^2})}} \int e^{-(b+\frac{1}{2\sigma^2})\|\mathbf{x}-\frac{\mathbf{a}}{2(b+\frac{1}{2\sigma^2})}\|^2} \text{d}\mathbf{x}\\
&= \dfrac{1}{(2\pi\sigma^2)^{d/2}}e^{\frac{\| \mathbf{a}\|^2}{4(b+\frac{1}{2\sigma^2})}} \cdot (\frac{2\pi}{2(b+\frac{1}{2\sigma^2})})^{d/2}\\
&= \dfrac{1}{(2\pi\sigma^2)^{d/2}}e^{\frac{\sigma^2\| \mathbf{a}\|^2}{2(2b\sigma^2+1)}} \cdot (\frac{2\pi\sigma^2}{2b\sigma^2+1})^{d/2}\\
&= \frac{1}{(1+2b\sigma^2)^{d/2}} \exp (\frac{\sigma^2\| \mathbf{a}\|^2}{2(1+2b\sigma^2)})
\end{aligned}
$$
</details>

因此:
$$
a = \dfrac{2W_t}{d}, b=\dfrac{1}{d}, \sigma^2=t
$$
所以结果为:
$$
\begin{aligned}
\mathbb{E}_{x_t'}[e^{\dfrac{2W_t^{\mathsf{T}} x_t'}{d}-\dfrac{\|x_t'\|^2}{d}}]
&= \dfrac{1}{(1+2t/d)^{d/2}}\exp(\dfrac{t\|\dfrac{2W_t}{d}\|^2}{2(1+2t/d)})\\
&= (\dfrac{d}{d+2t})^{d/2}\exp(\dfrac{dt\|\dfrac{2W_t}{d}\|^2}{2(d+2t)})\\
&= (\dfrac{d}{d+2t})^{d/2}\exp(\dfrac{2t\|W_t\|^2}{d(d+2t)})\\
\end{aligned}
$$

合并结果后为:
$$
\begin{aligned}
\textcolor{red}{\tilde{\mathbb{E}}_{x_t'\sim\mu_t} e^{-\dfrac{\|X_t- x_t'\|^2}{d}}}
&= e^{-\dfrac{\|W_t\|^2}{d}}\cdot \mathbb{E}_{x_t'}[e^{\dfrac{2W_t^{\mathsf{T}} x_t'}{d}-\dfrac{\|x_t'\|^2}{d}}]\\
&= \exp(-\dfrac{\|W_t\|^2}{d}) (\dfrac{d}{d+2t})^{d/2}\exp(\dfrac{2t\|W_t\|^2}{d(d+2t)})\\
&= (\dfrac{d}{d+2t})^{d/2}\exp(\dfrac{2t\|W_t\|^2-(d+2t)\|W_t\|^2}{d(d+2t)})\\
&= (\dfrac{d}{d+2t})^{d/2}\exp(\dfrac{-\|W_t\|^2}{d+2t})\\
\end{aligned}
$$

可以得到第一项的理论结果为 0.

---

第二部分为:
$$
\begin{aligned}
\textcolor{blue}{m_t^Y} - \sin(t) e^{-\dfrac{t}{2}}
&= \mathbb{E}[Y_t] - \sin(t) e^{-\dfrac{t}{2}}\\
\end{aligned}
$$

由 $Y_t = \sin(t + \dfrac{W_t^1 + \dots + W_t^d}{\sqrt{d}})$ 可得:
设 $S_t=\dfrac{W_t^1 + \dots + W_t^d}{\sqrt{d}}$, 它服从均值为 $0$, 方差为 $t$ 的高斯分布 $S_t\sim \mathcal{N}(0, t)$, 则由高斯分布的特征函数可得:

$\mathbb{E}[e^{i\lambda S_t}] = \mathbb{E}[\cos(\lambda S_t)+i\sin(\lambda S_t)] = e^{-\dfrac{\lambda^2 t}{2}}$

取 $\lambda = 1$ 的实部和虚部: $\mathbb{E}[\cos(S_t)]=\exp(-\dfrac{t}{2})$, $\mathbb{E}[\sin(S_t)]=0$.

所以
$$
\begin{aligned}
\mathbb{E}[Y_t] &= \mathbb{E}[\sin(t+S_t)]\\
&= \mathbb{E}[\sin (t)\cos(S_t) + \cos(t)\sin(S_t)]\\
&= \sin(t)\mathbb{E}[\cos(S_t)] + \cos(t)\mathbb{E}[\sin(S_t)]\\
&= \sin(t)\exp(-\dfrac{t}{2}) + \cos(t)0\\
&= \sin(t)e^{-\dfrac{t}{2}}\\
\end{aligned}
$$

所以第二项的理论结果为 0.

综上所述, 当 $X_t=W_t$ 时, 漂移项为 0, 满足方程.

---

<details>
<summary>特征函数补充内容</summary>

对于随机变量 $X$, 其特征函数 $\phi_X(\lambda)$ 定义为:
$$
\phi_X(\lambda) = \mathbb{E}\left[e^{i\lambda X}\right], \quad \lambda \in \mathbb{R}
$$

- 其中 $i$ 是虚数单位;
- **特征函数是概率密度函数 (PDF)的傅里叶变换**.

对于高斯分布情形, 设 $X \sim \mathcal{N}(\mu, \sigma^2)$, 其概率密度函数为:
$$
f_X(x) = \frac{1}{\sqrt{2\pi}\sigma} \exp(-\frac{(x-\mu)^2}{2\sigma^2})
$$

特征函数 $\phi_X(\lambda)$ 则为:
$$
\phi_X(\lambda) = \int_{-\infty}^{\infty} e^{i\lambda x} f_X(x) \, dx = \frac{1}{\sqrt{2\pi}\sigma} \int_{-\infty}^{\infty} e^{i\lambda x} e^{-\frac{(x-\mu)^2}{2\sigma^2}} \, dx
$$

合并指数部分：
$$
i\lambda x - \frac{(x-\mu)^2}{2\sigma^2} = -\frac{x^2 - 2\mu x + \mu^2 - 2i\lambda x \sigma^2}{2\sigma^2}
$$

对 $x$ 配方：
$$
x^2 - 2(\mu + i\lambda \sigma^2)x + \mu^2 = \left(x - (\mu + i\lambda \sigma^2)\right)^2 - (\mu + i\lambda \sigma^2)^2 + \mu^2
$$
因此, 指数部分变为：
$$
-\frac{(x - (\mu + i\lambda \sigma^2))^2}{2\sigma^2} + \frac{(\mu + i\lambda \sigma^2)^2 - \mu^2}{2\sigma^2}
$$

特征函数可拆分为：
$$
\phi_X(\lambda) = \frac{1}{\sqrt{2\pi}\sigma} e^{\frac{(\mu + i\lambda \sigma^2)^2 - \mu^2}{2\sigma^2}} \int_{-\infty}^{\infty} e^{-\frac{(x - (\mu + i\lambda \sigma^2))^2}{2\sigma^2}} \, dx
$$

- **积分项**：是一个高斯积分（即使中心是复数）, 其值为 $\sqrt{2\pi}\sigma$（解析延拓保证）。
- **常数项**：化简指数部分：
  $$
  \frac{(\mu + i\lambda \sigma^2)^2 - \mu^2}{2\sigma^2} = \frac{2i\lambda \mu \sigma^2 - \lambda^2 \sigma^4}{2\sigma^2} = i\lambda \mu - \frac{\lambda^2 \sigma^2}{2}
  $$

最后化简得到:
$$
\phi_X(\lambda) = e^{i\lambda \mu - \frac{\lambda^2 \sigma^2}{2}}
$$

---

1. 零均值高斯分布 ($\mu = 0$)
$$
\phi_X(\lambda) = e^{-\frac{\lambda^2 \sigma^2}{2}}
$$
- **实部**：$\mathbb{E}[\cos(\lambda X)] = e^{-\frac{\lambda^2 \sigma^2}{2}}$
- **虚部**：$\mathbb{E}[\sin(\lambda X)] = 0$（奇函数对称性）。

2. 多维高斯分布 ($\mathbf{X} \sim \mathcal{N}(\mathbf{\mu}, \mathbf{\Sigma})$)
$$
\phi_{\mathbf{X}}(\boldsymbol{\lambda}) = e^{i\boldsymbol{\lambda}^T \mathbf{\mu} - \frac{1}{2} \boldsymbol{\lambda}^T \mathbf{\Sigma} \boldsymbol{\lambda}}
$$

3. 特征函数可以生成各阶矩。例如：
$$
\mathbb{E}[X^n] = \frac{1}{i^n} \left. \frac{d^n}{d\lambda^n} \phi_X(\lambda) \right|_{\lambda=0}
$$

</details>

## 算例 2: Cucker-Smale 集群模型的平均场博弈

考虑 Cucker-Smale 集群模型 (Flocking Model) 的平均场问题 39 40.
Cucker-Smale 模型在建模和分析集群行为中十分重要, 集群行为是指一群自驱动个体所表现出的集体运动, 当个体数量非常庞大时, 很自然地会考虑其平均场极限系统.
下面考虑 Cucker-Smale 模型的平均场博弈版本, 每个智能体通过选择自身的加速度, 来最小化加速成本以及位置和速度的偏差.

通过以下过程定义代表性智能体动态:

$$
\begin{aligned}
\text{d} x_t &= v_t \text{d} t,\\
\text{d} v_t &= u_t \text{d} t + C \text{d} W_t,\\
\end{aligned}
$$

- $x_t\in\mathbb{R}^n$: 智能体位置;
- $v_t\in\mathbb{R}^n$: 智能体速度;
- $u_t\in\mathbb{R}^n$: 加速度控制输入;
- $W_t\in\mathcal{R}^p$: 布朗运动.

智能体在可行的 $u_t$ 下最小化目标:
$$
\begin{aligned}
&\mathbb{E} \int_0^T \|u_t\|_R^2 + \mathcal{C}(x_t,v_t;f_t)\text{d}t,\\
&\mathcal{C}(x,v;f) = \| \int_{\mathbb{R}^{2n}} w(\| x-x'\|)(v'-v)f(x',v')\text{d}x' v'\|_Q^2\\
&\qquad\qquad=\| \tilde{\mathbb{E}}_{(x',v')\sim f} [w(\|x-x'\|)(v'-v)]\|_Q^2.\\
&w(x):=\dfrac{1}{(1+x^2)^{\beta}},\quad \beta\geq 0.
\end{aligned}
$$
- 第一项为加速成本;
- 第二项描述给定分布 $f_t$ 下智能体的位置偏差和速度偏差;
- $Q, R$: 具有兼容维度的对称正定矩阵, 使得 $\|x\|_Q:= (x^{\mathsf{T}}Qx)^{1/2}$.

使用**随机最大化原则 (Stochastic Maximization Principle)** 通过 MV-FBSDEs 刻画平均场均衡.
为此, 定义 Hamiltonian $H$ 为:
$$
H(t,x,v,f,y,u) = (v^{\mathsf{T}}, u^{\mathsf{T}}) y + \mathcal{C}(x,v;f) + \|u\|_R^2,
$$
- $y\in \mathbb{R}^{2n}$

$$
\hat{u} = -\dfrac{1}{2} R^{-1} y_{n+1:2n}
$$

$$
\begin{aligned}
\partial_x H &= \partial_x \mathcal{C}(x,v;f)\\
&=2\textcolor{red}{\tilde{\mathbb{E}}_{(x',v'\sim f)}[\partial_x w(\|x-x'\|)(v'-v)]^{\mathsf{T}}} Q \textcolor{blue}{\tilde{\mathbb{E}}_{(x',v'\sim f)}[w(\|x-x'\|)(v'-v)]} \\
\partial_v H &= y_{1:n} + \partial_v \mathcal{C}(x,v;f)\\
&= y_{1:n} + 2Q\textcolor{blue}{\tilde{\mathbb{E}}_{(x',v'\sim f)}[w(\|x-x'\|)(v'-v)]}\textcolor{green}{\tilde{\mathbb{E}}_{(x',v'\sim f)}[-w(\|x-x'\|)]} \\
\end{aligned}
$$

通过随机最大化原则, $(\hat{u}_t, \hat{f}_t)_{0\leq t\leq T}$ 是 MFG 均衡当且仅当 $\hat{u}_t = -\dfrac{1}{2}R^{-1}Y_t^2$, $\hat{f}_t = \mathcal{L}(x_t,v_t)$, 且有如下 MV-FBSDEs:

$$
\begin{cases}
\text{d} x_t = v_t \text{d} t,\text{d} v_t = -\dfrac{1}{2}R^{-1}Y_t^2\text{d}t + C\text{d}W_t, &(x_0, v_0)=\xi,\\
\text{d} Y_t = -\begin{pmatrix}\partial_x H\\ \partial_v H\end{pmatrix}(t,x_t,v_t,\mathcal{L}(x_t,v_t), Y_t, \hat{u}_t)\text{d}t + Z_t\text{d}W_t, & Y_T = 0.
\end{cases}
$$

[^Germain2019Numerical]: [Numerical Resolution of McKean-Vlasov FBSDEs Using Neural Networks.](../../docs/BSDE/Germain2019Numerical.md)