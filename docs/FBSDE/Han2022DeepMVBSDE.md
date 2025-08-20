# 学习具有一般分布依赖性的高维 McKean-Vlasov 正倒向随机微分方程

- 标题: Learning High-Dimensional McKean-Vlasov Forward-Backward Stochastic Differential Equations with General Distribution Dependence.
- 作者: Jiequn Han, Ruimeng Hu, Jihao Long
- 发表: [SIAM Journal on Numerical Analysis, 2024](https://epubs.siam.org/doi/abs/10.1137/22M151861X)
- 预印: https://arxiv.org/abs/2204.11924v3
- 代码: https://github.com/frankhan91/DeepMVBSDE
- 引用: [Han2022DeepMVBSDE]
- 复现: [Example](../../examples/Han2022DeepMVBSDE/Examples.md)

## 基本概念

- **平均场**：将复杂系统中大量的个体相互作用简化为单个个体与群体平均行为的交互。关键特点是使用群体效应代替个体交互，降低了复杂度。
- **平均场博弈**：研究大量相似理性个体在相互影响下的决策行为，每个个体的策略取决于群体的整体状态，群体整体状态由所有个体的策略共同形成，即个体行为影响群体，群体状态又反过来影响个体决策的均衡问题。
  - 个体最优控制问题：HJB 方程 + 群体分布演化 Fokker-Planck 方程
- **平均场控制**：存在中央控制者，优化全局目标
- **Cucker-Smale 模型**：描述自驱动粒子群体（如鸟群）自组织形成一致运动的数学模型
  - 个体通过局部交互调整自身速度，逐渐与邻居趋同；
  - 无需中央控制，仅依靠简单规则即可涌现群体同步行为；
- **虚拟博弈 (Fictitious Play)**: 用于求解博弈论中均衡策略的迭代学习算法，用于模拟玩家如何通过观察对手的历史行为选择自己的最优响应，最终收敛到纳什均衡。
- **连续统 (Continuum)**: 连续不断的数集；
- **Pontryagin 最大值原理**：作为现代控制理论的核心工具，将最优控制问题转化为一个哈密顿系统的极值问题，通过引入协态变量 (Costate)，将动态优化问题转化为一组微分方程（正向状态方程+倒向协态方程）的求解问题。提供了必要条件，在凸性问题中可保证全局最优。

## 摘要

**平均场 (Mean-Field Control)** 和**平均场博弈 (Mean-Field Game)** 的核心问题之一是求解相应的 McKean-Vlasov 正倒向随机微分方程 (MV-FBSDEs)。
大多数现有方法仅适用于一些特殊情形，其平均场相互作用仅依赖于期望或其他矩量，因此当平均场相互作用具有完全分布依赖性时不能适当地求解问题。

本文提出了一个新式深度学习方法，用于计算具有平均场相互作用一般形式的 MV-FBSDE。
具体来说，基于**虚拟博弈 (Fictitious Play)**，我们将原问题转化为重复求解具有显式系数函数的标准 FBSDEs。
这些系数函数用于近似具有完全分布依赖性的 MV-FBSDE 的模型系数，并通过利用前次迭代的 FBSDE 的解仿真的训练数据求解另外的监督学习问题来更新这些系数。
我们使用深度神经网络求解标准的 BSDEs 和近似系数函数以求解高维的 MV-FBSDEs。

对学习到的函数进行适当的假设，我们使用 [《A Class of Dimension-free Metrics for the Convergence of Empirical Measures》](https://arxiv.org/abs/2104.12036) 提出的一类积分概率度量，证明了所提方法的收敛性不受维度灾难的影响。
证明的定理展示了方法在高维情形下的优势。

我们在高维 MV-FBSDE 问题展示了数值性能，包括著名的 Cucker-Smale 模型的平均场博弈示例，其成本依赖于前向过程的全分布。

## 引言

**McKean-Vlasov 正倒向随机微分方程 (MV-FBSDEs)** 是**平均场博弈 (MFGs)** 和**平均场控制 (MFC)** 问题的自然表述形式。

- 由 Lasry-Lions [^33] [^34] [^35] 和 Huang-Malhame-Caines [^30] [^31] 首次提出， MFG 研究同质智能体**连续统**的策略决策问题，其中每个智能体都旨在最大化自身目标。
由于每个智能体都是无限小 (infinitesimal) 的，在优化其自身的成本泛函时，状态过程随机变量的概率分布被视为固定的。
因此，这只是一个标准的最优控制问题加上一个不动点论证（解的分布等于初始分布）。

- 从稍有不同的角度来看，MFC 分析 McKean-Vlasov 型随机微分方程（SDEs）的最优控制，这可以解释为大群体智能体之间合作博弈的极限状态。
这个目标为不动点增加了一个额外的优化层，即状态过程的分布随着目标优化而变化。

尽管存在差异，这两个问题都可以通过使用 **Pontryagin 最大值原理**分析 MV-FBSDEs 来解决 [^09] [^11]，并且这两个解都可以作为交互方式和目标函数为平均场类型的大群体个体的近似均衡策略。

关于它们差异和关系的更多解释，我们推荐读者参阅 [^11] 第6节。

---

本文中，我们致力于开发有效的深度学习方法并分析以下 MV-FBSDEs 的数值收敛性：

$$
\begin{cases}
\mathrm{d} X_t = \textcolor{cyan}{b(t, X_t, Y_t, Z_t, \mathcal{L}(X_t, Y_t, Z_t))} \text{d} t + \textcolor{orange}{\sigma(t, X_t, Y_t, Z_t, \mathcal{L}(X_t, Y_t, Z_t))} \text{d} W_t,  &X_0 = x_0,  \\
\mathrm{d} Y_t = -\textcolor{red}{h(t, X_t, Y_t, Z_t, \mathcal{L}(X_t, Y_t, Z_t))} \text{d} t + Z_t \text{d} W_t, &Y_T = g(X_T, \mathcal{L}(X_T)),
\end{cases}
$$

- 在有限时间区间 $[0,T]$上，
- $\mathcal{L}(\cdot)$ 表示过程的边缘概率分布，
- $(b, \sigma, h, g)$ 是具有兼容维数的可测函数，具体形式详见第 2 节。

特别地，我们将重点计算具有一般形式的平均场相互作用的高维 MV-FBSDEs，而现有方法大多仅处理平均场相互作用通过期望或其他矩量描述的特殊情况。

---

理论上，MV-FBSDEs 的存在性和唯一性结果最近已经发展起来 [^08] [^09] [^10]，主要通过紧致性论证和不动点定理解决。

特别地，当 $\sigma$ 不依赖于 $Z_t$ 和 $\mathcal{L}(Z_t)$时，在适当条件下 [^11] (定理 4.29，备注 4.30)，MV-FBSDEs 允许具有解耦场的解：

$$
    Y_t = u(t, X_t), \quad Z_t = v(t, X_t)
$$

- 其中 $v(t, x) = \partial_x u(t, x) \textcolor{orange}{\sigma(t, x, u(t, x), \textcolor{red}{(I_d, u(t, \cdot))(\mathcal{L}(X_t))})}$。

已经设计了几种数值算法来求解 MV-FBSDEs，包括:
- 递归局部 Picard 迭代方法 [^13];
- 解耦设置下的基于 cubature 的算法 [^15];
- 以及深度学习方法 [^12] [^21]。

大多数工作，无论是否有分析，都只展示了一种特殊类型平均场相互作用的数值例子，即 $(b, \sigma, h, g)$ 仅通过其期望依赖于$\mathcal{L}$。

在相关主题上，MFGs 和 MFC 问题也已通过偏微分方程和使用神经网络解决高维问题；例如参见 [^02] [^45] [^44] [^01]。

---

与我们的工作最密切相关的是 [^21]。

它使用深度学习解决 MV-FBSDEs，并且 $(b, \sigma, h, g)$ 相对于 $\mathcal{L}(X_t, Y_t, Z_t)$ 的依赖仅以 $(\mathbb{E}[\varphi_1(X_t)], \mathbb{E}[\varphi_2(Y_t)], \mathbb{E}[\varphi_3(Z_t)])$ 的形式出现，其中 $\varphi_i$ 是一些连续函数。
为此，本质上是在估计一些均衡数 $\int_y \varphi_i(y) \,\mu(\mathrm{d}y)$，其中 $\mu$ 是 MV-FBSDE 解的概率分布。
超出仅依赖于矩量的范围（本文的范围），目标是学习以非线性方式依赖于分布的均衡函数，例如形式为 $\int_y \varphi(t, x, y) \, \mu(\mathrm{d}y)$ 的 $(t,x)$ 函数。这在算法设计和表示这些量的方法方面都带来了更大的挑战。

我们提出了一种新颖的深度学习方法来求解高维 MV-FBSDEs，利用虚拟博弈的思想 [^06] [^07]和现有的机器学习BSDE求解器 [^16] [^26] [^27] [^32]，具有处理一般平均场依赖（不仅仅是矩量）的能力。

主要贡献包括：

1. 受 MV-FBSDEs 的不动点性质启发，我们使用虚拟博弈并求解
$$
\begin{dcases}
  \text{d} X_t^{(k)} = \textcolor{cyan}{b(t, \Theta_t^{(k)}, \mathcal{L}(\Theta_t^{(k-1)}))} \text{d} t + \textcolor{orange}{\sigma(t, \Theta_t^{(k)}, \mathcal{L}(\Theta_t^{(k-1)}))} \text{d} W_t,  &X_0 = x_0,  \\
  \text{d} Y_t^{(k)} = -\textcolor{red}{h(t, \Theta_t^{k}, \mathcal{L}(\Theta_t^{(k-1)}))} \text{d} t + Z_t^{(k)} \text{d} W_t, &Y_T = g(X_T^{(k)}, \mathcal{L}(X_T^{(k-1)})),
\end{dcases}
$$

在解耦场形式的解中（即寻找一些确定性函数 $u^{(k)}, v^{(k)}$，它们被视为对上述解耦场的解中定义的 $u, v$ 的近似，使得 $Y_t^{(k)} = u^{(k)}(t, X_t^{(k)}), Z_t^{(k)} = v^{(k)}(t, X_t^{(k)})$），通过现有的 BSDE 求解器迭代求解。
为了加速BSDE求解器对 $(X^{(k)}, Y^{(k)})$ 的模拟，我们提出使用神经网络通过监督学习学习映射 $(t, X_t, Y_t, Z_t) \mapsto (b, \sigma, h, g)$，当它们完全依赖于任何分布时。在数值上，函数值（也称为"标签"）通过用上一次迭代解的相应经验分布替换 $\mathcal{L}(\cdot)$ 来近似，并且学习的映射在每次迭代后更新。

1. 我们证明了所提算法的收敛性。
使用 [25] 中提出的积分概率度量，学习过程与解之间的差异可以足够小且不受维度灾难的影响，前提是学习函数具有足够的平滑性、虚拟游戏的足够迭代次数以及足够小的时间步长。

1. 我们引入了一个 Cucker-Smale 模型的平均场博弈，其中成本依赖于正向过程的完整分布。我们提供了数值基准，并希望这将促进对MV-FBSDEs数值算法的进一步研究。

---

本文的其余部分组织如下。
- 第 2 节回顾了虚拟博弈的思想和两种现有的基于深度学习（DL）的BSDE求解器：Deep BSDE 和 DBDP，并描述了我们提出的算法。
- 第 3 节提供了所提算法的收敛性分析。
- 第 4 节展示了两个数值例子。
  - 第一个是十维的，具有解析解作为基准。值得一提的是，第一个例子不是文献中广泛使用的与线性二次问题相关的 MV-FBSDEs。
  - 在第二个例子中，我们研究了 Cucker-Smale 模型下的平均场群集问题。平均场相互作用涉及状态过程的整个分布。
- 第 5 节总结本文工作。

## 用于 MV-FBSDEs 的深度学习算法

相比标准的 FBSDE 问题，求解 MV-FBSDEs 面临两个额外的困难：
$$
\begin{cases}
\mathrm{d} X_t = \textcolor{cyan}{b(t, X_t, Y_t, Z_t, \mathcal{L}(X_t, Y_t, Z_t))} \text{d} t + \textcolor{orange}{\sigma(t, X_t, Y_t, Z_t, \mathcal{L}(X_t, Y_t, Z_t))} \text{d} W_t,  &X_0 = x_0,  \\
\mathrm{d} Y_t = -\textcolor{red}{h(t, X_t, Y_t, Z_t, \mathcal{L}(X_t, Y_t, Z_t))} \text{d} t + Z_t \text{d} W_t, &Y_T = g(X_T, \mathcal{L}(X_T)),
\end{cases}
$$

1. $\mathcal{L}(X_t, Y_t, Z_t)$ 概率分布无法得知先验，而是作为不动点被确定；
2. 即便给定了 $\mathcal{L}(X_t, Y_t, Z_t)$，系数函数 $(b, \sigma, h, g)$ 在 $\mathcal{L}$ 上的依赖也可以非常复杂，导致 FBSDE 在计算上难以求解。

为了克服第一个困难，我们使用虚拟博弈的思想。
虚拟博弈最初由 Brown [^06] [^07] 提出，作为静态博弈中寻找纳什均衡的学习过程，后广泛应用与机器学习算法和理论研究中 [^29] [^23] [^24] [^18] [^47] [^48] [^37] [^28]。
这类问题通常需要寻找某种 "均衡" 的量 $Q$。
从对这一均衡的初始猜测 $Q^{(0)}$ 出发，用于求解此问题，然后更新猜测 $Q^{(1)}$，然后循环求解，产生一个猜测序列 $Q^{(0)},\cdots,Q^{(k)},\cdots $ 并期望有极限 $Q^*$。
在 N 人博弈中，这个量为纳什均衡策略，在平均场博弈中，这是最优状态过程的概率分布，在本文中为 $(X_t,Y_t,Z_t)$ 的测度流。

注意虚拟博弈的思想和 MV-FBSDEs 解的理论构造高度契合——后者通常将问题转化为关于不动点问题来求解：
首先使用一些固定分布作为输入，然后作为标准的 FBSDE 进行求解，目标是在合适的函数和测度空间中找到一个固定点，将输入分布映射到标准 FBSDE 的概率分布。

为了克服第二个困难，我们观察到 $\mathcal{L}(X_t,Y_t,Z_t)$ 是关于 $t$ 的确定性函数，所以我们可以将后续的 $m_1$, $m_2$ 视为 $(t,x)$ 的函数，类似地 $m_3$ 视为 $x$ 的函数。
因此，我们给定 $\mathcal{L}(X_t, Y_t, Z_t)$ 的最新估计，使用监督学习来直接学习这些映射。

---

## Deep MV-FBSDE 算法

首先假设系数函数 $(b, \sigma, h, g)$ 具有如下结构：
令 $m_1,m_2: [0,T]\times \mathbb{R}^d\times \mathcal{P}^2(\mathbb{R}^\theta)\to \mathbb{R}^l$ 和 $m_3: \mathbb{R}^d\times \mathcal{P}^2(\mathbb{R}^d)\to \mathbb{R}^l$ 为向量值函数，使得

$$
\begin{aligned}
b &= b(t, \Theta_t, m_1(t, X_t, \mathcal{L}(\Theta_t))),\\
\sigma &= \sigma(t, X_t),\\
h &= b(t, \Theta_t, m_2(t, X_t, \mathcal{L}(\Theta_t))),\\
g &= g(X_T, m_3(X_T, \mathcal{L}(X_T))),\\
\end{aligned}
$$

我们的算法在解 $(X_t, Y_t, Z_t)$ 和他们的经验测度上迭代。

在每次迭代中遵循以下三步：

### 更新对分布 $\mathcal{L}(\Theta)$ 的近似：

给定对解的最新近似：$Y_0\approx u^{(k-1)}(0, X_0), Z_t\approx v^{(k-1)}(t,X_t)$；
给定对分布依赖的近似：$\hat{m}_1^{(k-1)}, \hat{m}_2^{(k-1)}, \hat{m}_3^{(k-1)}$；

我们考虑关于 $(\tilde{X}_{t}^{(k-1)}, \tilde{Y}_{t}^{(k-1)})$ 的如下前向 SDE：

$$
\begin{dcases}
\mathrm{d} \tilde X_t^{(k-1)} = \textcolor{cyan}{b(t, \tilde \Theta_t^{(k-1)}, \hat{m}_1^{(k-1)}(t, \tilde X^{(k-1)}_t))} \text{d} t + \textcolor{orange}{\sigma(t, \tilde X_t^{(k-1)})} \text{d} W_t,  & \tilde X_0^{(k-1)} = \xi,\\
\mathrm{d} \tilde Y_t^{(k-1)} = - \textcolor{red}{h(t, \tilde \Theta_t^{(k-1)}, \hat{m}_2^{(k-1)}(t, \tilde X^{(k-1)}_t))} \text{d} t + \textcolor{yellow}{v^{(k-1)}(t, \tilde X^{(k-1)}_t)} \text{d} W_t, & \tilde Y_0^{(k-1)}= u^{(k-1)}(0, \tilde X_0^{(k-1)}).
\end{dcases}
$$

并使用相应的分布来近似 $\mathcal{L}(\Theta)$ 和 $\mathcal{L}(X_T)$：
$$
\nu_t^{(k-1)} = \mathcal{L}(\tilde{\Theta}_t^{(k-1)}); \mu_T^{(k-1)} = \mathcal{L}(\tilde{X}_T^{(k-1)})
$$

### 近似概率依赖 $m_1, m_2, m_3$：

给定最新的 $\mathcal{L}(\Theta)$ 和 $\mathcal{L}(X_T)$ 的估计，我们可以将 $m_1, m_2, m_3$ 分别视为关于 $(t,x)$ 和 $x$ 的函数。
自然地，我们使用神经网络近似这些函数并通过监督学习优化。

在第 $(k-1)$ 阶段中，我们优化：

$$
\begin{aligned}
&\inf_{\mathfrak{m}_1} \int_0^T\mathbb{E}\|m_1(t, \tilde X_t^{(k-1)}, \nu_t^{(k-1)}) - \mathfrak{m}_1(t, \tilde X_t^{(k-1)})\|^2 \text{d} t, \\
&\inf_{\mathfrak{m}_2} \int_0^T\mathbb{E}\|m_2(t, \tilde X_t^{(k-1)}, \nu_t^{(k-1)}) - \mathfrak{m}_2(t, \tilde X_t^{(k-1)})\|^2 \text{d} t, \\
& \inf_{\mathfrak{m}_3}  \mathbb{E}\|m_3(\tilde X_T^{(k-1)}, \mu_T^{(k-1)}) - \mathfrak{m}_3(\tilde X_T^{(k-1)})\|^2,
\end{aligned}
$$

其中 $\mathfrak{m}_1, \mathfrak{m}_2, \mathfrak{m}_3$ 在一类 NN 上搜索。
优化后的神经网络记为 $\hat{m}_1, \hat{m}_2, \hat{m}_3$。

### 求解标准 FBSDE

在第 $(k)$ 阶段求解标准 FBSDE：

$$
\begin{cases}
\text{d} X_t^{(k)} = b(t, \Theta_t^{(k)}, \hat m_1^{(k)}(t, X_t^{k}))\text{d} t + \sigma(t, X_t^{(k)})\text{d} W_t, & X^{(k)}_0 = \xi,\\
\text{d} Y^{(k)}_t = - h(t, \Theta_t^{(k)}, \hat m_2^{(k)}(t, X_t^{k}))\text{d} t + Z^{(k)}_t\text{d} W_t, & Y^{(k)}_T= g(X^{(k)}_T, \hat m_3^{(k)}(X^{k}_T)),
\end{cases}
$$

目标是找到最优的网络 $\psi^{(k)}$ 和 $\phi^{(k)}$ 分别参数化 $Y_0^{(k)}$ 和 $Z_t^{(k)}$。
这两个函数是作为解耦场 $u^{(k)}$ 和 $v^{(k)}$ 的近似。

注 1：第二步使用了对 $m_i$ 的监督学习，用于快速提供对 $m_i$ 的估计，这对第三步很重要。因为不进行此运算，那么由于 $m$ 依赖于 $\mathcal{L}$，那么此时需要通过经验分布的方式计算，非常耗时。

注 2：第二步训练时使用的是 $X_t^{(k-1)}$，而第三步计算时使用的是 $X_t^{(k)}$，随机变量的分布的不匹配可能会导致理论和数值困难，这也是我们为什么将系数函数进行了限制。za
在此假设下，通过 Girsanov 定理，$X_t^{(k)}$ 的分布关于 $X_t^{(k-1)}$ 的分布是绝对连续的，因此可以用后者的误差来控制前者（引理 3.8）。

一种替代的方式是所有将另外两步的变量使用同一个 $(k)$ 以保持一致。然而如果需要仿真一个路径，那么就需要先仿真之前的所有步，计算上并不高效。

因此这两种方案存在一个权衡。

### 具体实现

首先考虑在区间 $[0,T]$ 上的一个划分 $\pi$: $0=t_0 < t_1 < \cdots < t_{N_T} = T$, $\delta t_i = t_{i+1} - t_i$.

为了更新第一步中的分布，我们生成初始条件的 $N$ 个样本 $\xi^{n,(k-1)}$，和 $N$ 个布朗运动 $W_t^{n,(k-1)}$ 以离散时间的方式仿真正向 SDE：

$$
\begin{dcases}
\tilde X_{t_{i+1}}^{n, (k-1)} = \tilde X_{t_{i}}^{n, (k-1)} + \textcolor{cyan}{b(t_i, \tilde \Theta_{t_i}^{n,(k-1)}, \hat{m}_1^{(k-1)}(t_i, \tilde X_{t_i}^{n, (k-1)}))} \Delta t_i + \textcolor{orange}{\sigma(t_i, \tilde X_{t_i}^{n, (k-1)})} \Delta W_{t_i}^{n,(k-1)}, \\
\tilde Y_{t_{i+1}}^{n, (k-1)} = \tilde Y_{t_{i}}^{n, (k-1)} - \textcolor{red}{h(t_i, \tilde \Theta_{t_i}^{n, (k-1)}, \hat{m}_2^{(k-1)}(t_i, \tilde X_{t_i}^{n, (k-1)}))} \Delta t_i + \textcolor{yellow}{v^{(k-1)}(t_i, \tilde X_{t_i}^{n, (k-1)})} \Delta W_{t_i}^{n,(k-1)},
\end{dcases}
$$

初始条件：
$$
\tilde{X}_0^{n,(k-1)} = \xi^{n,(k-1)}, \quad \tilde{Y}_0^{n,(k-1)} = u^{(k-1)}(0, \tilde{X}_0^{n,(k-1)}),
$$

并定义用于近似 $\mathcal{L}(\Theta_t)$ 在时刻 $t_i$ 和 $\mathcal{L}(X_T)$ 的经验测度（狄拉克函数）:

$$
\nu_{t_i}^{k-1} = \frac{1}{N} \sum_{n=1}^N \delta_{\tilde \Theta_{t_i}^{n, (k-1)}}, \quad \mu_T^{k-1} = \frac{1}{N} \sum_{n=1}^N \delta_{\tilde X_T^{n, (k-1)}},
$$

然后使用前面仿真出来的数据用于构造对分布依赖 $m_i$ 的监督学习任务。
通过蒙特卡洛采样，对应的损失函数为：

$$
\begin{aligned}
&\inf_{\mathfrak{m}_1} \sum_{1\leq n \leq N}\sum_{0\leq i \leq N_T-1}\|m_1(t_i, \tilde X_{t_i}^{n,(k-1)}, \nu_{t_i}^{(k-1)}) - \mathfrak{m}_1(t_i, \tilde X_{t_i}^{n,(k-1)})\|^2, \\
&\inf_{\mathfrak{m}_2} \sum_{1\leq n \leq N}\sum_{0\leq i \leq N_T-1} \|m_2(t_i, \tilde X_{t_i}^{n, (k-1)}, \nu_{t_i}^{(k-1)}) - \mathfrak{m}_2(t_i, \tilde X_{t_i}^{n,(k-1)})\|^2, \\
& \inf_{\mathfrak{m}_3}  \sum_{1\leq n \leq N}\|m_3(\tilde X_T^{n,(k-1)}, \mu_T^{(k-1)}) - \mathfrak{m}_3(\tilde X_T^{n,(k-1)})\|^2.
\end{aligned}
$$

训练完成后，即可快速推理 $m_i^{(k)}$, 然后使用 DeepBSDE/DBDP 求解后续的 FBSDE。

---

## 代码解析

代码基本结构为：

- `configs/`
  - `flock_d3.json`
  - `sinebm_d2.json`
  - `sinebm_d10.json`
  - `sinebm_d15.json`
- `equation.py`
- `main.py`
- `solver.py`

`main.py` 使用 `absl` 用于命令行参数解析和日志处理，使用 `munch` 来转换配置文件属性，使用 tensorflow 作为基本计算框架。
在 `solver.py` 中定义了三个 Solver: SineBMSolver, SineBMDBDPSolver, FlockSolver.
在 `equation.py` 中定义了三个类：`Equation`, `SineBM`, `Flock`。

基本的方程类包含：
- 属性：`eqn_config`, 维度 `dim`, 终值时间 `total_time`, 时间区间数 `num_time_interval`, `delta_t`, `t_grid`, `sqrt_delta_t`, Y 初值 `y_init`
- 方法：采样 `sample()`；`f_tf()` 生成函数；`g_tf()` 终值条件

---

`SineBM` 类（意思是算例 4.1 的正弦基准 Sine Benchmark）
代码中包含：
- `create_model()`：创建 drift model，
- `update_drift_nn()`：训练 drift 神经网络，计算误差的对象似乎不正确？
- `update_drift_mc()`: 蒙特卡洛
- `sample()`：仿真正向 SDE 轨迹：
  - drift_true = -exp(sum(x[:,i]^2)/(d+2t)) * (d/(d+2t)) ** (d/2)
  - drift_nn = drift model.predict(x)
  - x_t+1 = x_t + sin(drift_nn - drift_true) * dt + dw
  - 当方程类型为 3 时，增加一项 couple_coeff * (mean_y_estimate - mean_y) * dt
    - couple_coeff = 0.5
    - mean_y_estimate = mean_y * 0
    - mean_y = sin(t_grid) * exp(-t_grid/2)
