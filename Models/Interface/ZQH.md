## 2021·SGFEM for Interfacial Crack Problems in Bi-Materials

标题: Stable Generalized Finite Element Methods (SGFEM) for Interfacial Crack Problems in Bi-Materials
作者: Hong Li, Cu Cui, Qinghui Zhang.
时间: Received 2021-06-16 - Accepted 2022-01-13.

---

在许多工程应用中，问题的解具有非光滑性质，例如跳跃 (Jumps)、尖角 (Kinks)、角奇性 (Corner Singularities)、高梯度 (High Gradients) 和振荡 (Oscillations)。
此外，非光滑性质可能随时间变化。

传统的有限元方法 (FEM) 对于这类问题需要一个网格来适应非光滑性以产生合理的解。
如果解涉及到奇性点，例如裂纹尖端 (Crack Tips) 和重入角 (Re-entrant Corners)，FEM 中的网格需要在这些点的附近进行细化以提高解的精度。
这样的网格操作——网格适应/细化——十分耗时且难以实现，特别是对于时间依赖和三维问题。
因此，简化网格生成是至关重要的。

**广义/扩展有限元 (Generalized/Extended Finite Element, GFEM/XFEM)** 使用一个通常简单、固定且与问题的非光滑特征无关的网格。
GFEM/XFEM 的主要概念是在简单网格上对 FEM 进行增广，使用特殊函数模拟局部的非光滑性。
GFEM 和 XFEM 本质上都是**单元分解方法 (Partition of Unity Methods, PUMs)** [^1] [^2]，其中特殊函数使用**单元分解 (Partition of Unity, PU) 函数**"粘贴"在一起。
GFEM/XFEM 已经广泛应用于各种工程问题，如裂纹扩展 (Crack Growth)、材料建模 (Material Modeling)、多相流 (Multiphase Flows) 和流体结构相互作用 (Fluid-Structure Interaction)。
对于 GFEM/XFEM 的各个方面可参阅综述文章 [^3] [^4] [^5] [^6] [^7]。

GFEM/XFEM 已经成功应用于裂纹问题，并在近几十年中取得了许多重要的进展。
有关它们的综述和近期进展，可参阅 [^8] [^9] [^10] [^11] [^12] [^13] [^14] [^15] [^16] [^17] [^18]。

裂纹问题涉及到两种非光滑特性: **沿裂纹线的不连续性**和**裂纹尖端附近的奇异性**。
这些特性在 GFEM/XFEM 中通常分别使用阶跃函数/赫维赛德函数 (Heaviside Function) 和分支函数 (Branch Function) 处理。
基于阶跃函数和分支函数的 GFEM/XFEM 在裂纹问题中的最优收敛性已有研究。
然而，GFEM/XFEM 的条件性 (Conditioning) 比标准 FEM 要差得多 [^4] [^5] [^19] [^20]。
这主要是因为 FEM 形状函数与阶跃函数和分支函数之间**几乎线性相关 (Almost Linear Dependence, ALD)**。
例如，对于一个具有奇性解的一维特定问题，数学上已经证明 GFEM 的刚度矩阵的**缩放条件数 (Scaled Condition Number, SCN)** [^19] 是 $\mathcal{O}(h^{-4})$，远大于标准 FEM 的 $\mathcal{O}(h^{-2})$。
这种不良条件性可能会导致在求解底层线性系统时消除法中的灾难性舍入误差或降低迭代格式的收敛率。
许多工作致力于改善 GFEM/XFEM 的条件性，例如局部调整节点或界面曲线的位置来平衡曲线和元素相交生成的两个体积的比例 [^21] [^22] [^23]，预条件刚度矩阵或正交化 [^20] [^24] [^25] [^26] [^27] [^28]，使用插值函数修正富集函数 [^15] [^17] [^19] [^29] [^30] [^31] [^32]。

在一系列最近的研究中，**稳定广义有限元方法 (Stable Generalized Finite Element, SGFEM)** 被提出用于解决 GFEM/XFEM 在裂纹问题中的条件性困难。
如果 GFEM/XFEM 满足：(a) 达到最优收敛性；(b) 其条件性和 FEM 同阶; (c) 当裂纹线接近元素边界时，收敛性和条件性不恶化，那么认为它是稳定的，即 SGFEM。
最后一个特点代表了 GFEM/XFEM 的稳健性。
SGFEM 的概念主要基于一个简单的修改方案，即从富集函数中减去他们的有限元插值函数 [^19] [^30] [^31]。
前面提及的研究表明 GFEM/XFEM 在条件性和稳健性方面的困难可以被克服。

上面讨论的裂纹问题是同质材料，双材料裂纹问题相对更加复杂。
除了同质裂纹问题中具有的不连续性和奇异性之外，还存在由 $\sin(\epsilon \ln r)$ 和 $\cos(\epsilon \ln r)$ 描述的**振荡奇异性 (Oscillatory Singularity)** 和沿着结合界面未开裂部分**弱不连续性 (Weak Discontinuity)**[^34] [^35]。
弱不连续性意味着位移是连续的，而梯度不是。
额外的非光滑性需要使用其他富集函数来处理。

用于同质材料的传统分支函数 [^5] [^8] [^9] 在双材料裂纹问题中作为富集函数 [^36] [^37] [^38]。
然而，这些函数不能产生有效的近似，因为它们没有反应出振荡奇异性的信息。
文献[^34] 基于双材料问题的解的理论展开[^39] [^40]，提出了 12-Fold 奇性函数，被后续的研究使用 [^41] [^42] [^43] [^44] [^45]。
然而，这种富集函数无法有效地表达展开的主要部分，如[^35]所述。
因此，对于双材料裂纹问题，使用 12-Fold 富集函数的 XFEM 无法达到最优收敛阶。
为了克服这一问题，基于解的展开设计了 8-Fold 材料依赖的奇性函数 [^39] [^46] [^47] [^48]。
这些奇性函数可以精确地表示展开的主要部分。
此外，文献探索了 16-Fold 奇性函数的开发 [^49] [^50]。

对于上述提及的弱不连续性，如果网格线和界面贴合，则无需特别处理。
然而，如果网格和界面不一致，则需要适当处理弱不连续性，以避免收敛性下降。
在 [^41] [^43] [^44] [^50]，使用了修改的距离或水平集函数。
在 [^35] 中使用了相同的 8-Fold 奇性函数来处理弱不连续性，只有不一致网格考虑拓扑富集，而所谓的混合单元误差 (blending element errors) 可能出现在几何富集 [12]。
GFEM/XFEM 在正交双材料、压电材料、泊松界面问题中的应用可参考 [^51] [^52] [^53] [^54] [^55] [^56] [^57] [^58]。

GFEM/XFEM 在双材料问题中的条件性困难比同质材料问题更严重。
首先，有更多的非光滑特性：不连续性、奇异性、弱不连续性。
此外，在一个富集节点中可能有两类非光滑特性，这会显著增加 ALD。
其次，在一个特定节点中具有更多的富集函数 (8, 12, 甚至 16)。
因此，局部 ALD 在这个节点变得更强。
局部 ALD 严重影响整个刚度矩阵的条件性，如 [^17],[^18],[^57] 所述。
数项研究则开始研究 GFEM/XFEM 在双材料问题中的条件性问题。
据我们所示，具有 (a) (b) (c) 特点的 SGFEM 尚未开发。
此外，现有工作主要考虑网格和界面之间的简单关系，如网格是否贴合或平行于界面，然而网格和界面之间的任意相对位置则未被讨论。

本研究提出满足 (a) (b) (c) 特点的 SGFEM 来解决 GFEM/XFEM 在双材料裂纹问题中的条件性困难。
SGFEM 中通过富集 8-Fold 奇性函数[^35]来处理径向和振荡奇异性，因为这些函数可以精确地展开解地主要部分，和传统分支函数和 12-Fold 富集函数相比 [^10]。
线性阶跃函数 [^15] 被用于近似沿裂纹线的不连续性。
对于弱不连续性，则富集一个一般的距离函数或绝对值水平集函数 [^29] [^55] [^57] [^58]。
SGFEM 建立在从富集函数中减去有限元插值函数的基本概念上 [^19],[^30],[^31]。
两种稳定性技术 —— 改变 PU 函数和局部猪肠粉分析 (Local Principal Component Analysis, LPCA) —— 被用于处理条件性问题。
LPCA 用于确定一个节点内多重富集的"贡献"，并过滤掉具有非常小"贡献"的冗余信息，使得局部 ALD 可以减小，从而提高条件性。
数值实验表明和传统 GFEM/XFEM 相比，提出的 SGFEM 具有最优收敛阶 $\mathcal{O}(h)$，并且条件性与 FEM 同阶。
构造了稳健性测试以表明 SGFEM 的收敛性和条件性不会随着界面线接近元素边界而变差。
还研究了材料系数比值对条件性的影响，结果表明 SGFEM 的条件性对不同的材料系数不敏感。
使用类似 [^17],[^33],[^55] 的技术，可以证明 SGFEM 的最优收敛阶是 $\mathcal{O}(h)$，在本文中进行了简化。

---

被引: (8)
1. Review on Interpretations, Applications, and Developments of Numerical Methods in Studying Interface Fracture. [Springer·Acta Mechanica](https://link.springer.com/article/10.1007/s00707-024-04212-6), 2024.11.02.
2. **ADN** Adaptive Deep Neural Networks for Solving Corner Singular Problems. [Elsevier·Eng. Ana. Bound.](https://doi.org/10.1016/j.enganabound.2023.11.022), 2023.06.23. (Self Cited)
3. **C-GFEM** Consistent Generalized Finite Element Method: An Accurate and Robust Mesh-Based Method Even in Distorted Meshes. [Elsevier·Eng. Ana. Bound.](https://doi.org/10.1016/j.enganabound.2024.106084) 2024.08.09.
4. A Novel Application of Recovered Stresses in Stress Intensity Factors Extraction Methods for the Generalized/Extended Finite Element Method. [Wiley]
5. Extended Isogeometric Analysis: A Two-Scale Coupling FEM/IGA for 2D Elastic Fracture Problems. [Springer·Computational Mechanics](https://doi.org/10.1007/s00466-023-02383-y), 2023.02.28.
6. Mass Lumping of Stable Generalized Finite Element Methods for Parabolic Interface Problems. [Wiley] (Self Cited)
7. Numerical Investigation of Convergence in the L inf Norm for Modified SGFEM Applied to Elliptic Interface Problems. [AIMS·Mathematics](https://doi.org/10.3934/math.20241507), 2024.08.23. (Cited by Students)
8. On the Implementation of SGFEM Simulation of Cohesive Crack Propagation Problems. 2022.11.21.

---
