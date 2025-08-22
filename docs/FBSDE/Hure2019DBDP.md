# Deep Backward Schemes for \\ High-Dimensional Nonlinear PDEs \Thanks{This Work Is Supported by FiME, Laboratoire De Finance Des March\'es De L'Energie, and the ''Finance and Sustainable Development'' EDF - CACIB Chair.

## Abstract

We propose new machine learning schemes for solving high dimensional nonlinear partial differential equations (PDEs). Relying on the classical backward stochastic di\-fferential equation (BSDE) representation of PDEs, our algorithms estimate simultaneously the solution and its gradient by deep neural networks. These approximations are performed at each time step from the minimization of loss functions defined recursively by backward induction. The metho\-dology is extended to variational inequalities ari\-sing in optimal stopping problems. We analyze the convergence of the deep learning schemes and provide error estimates in terms of the universal approximation of neural networks. Numerical results show that our algorithms give very good results till dimension 50 (and certainly above), for both PDEs and variational inequalities problems. For the PDEs resolution, our results are very similar to those obtained by the recent method in [^Weinan2017deep] when the latter converges to the right solution or does not diverge. Numerical tests indicate that the proposed methods are not stuck in poor local minima as it can be the case with the algorithm designed in [^Weinan2017deep], and no divergence is experienced. The only limitation seems to be due to the inability of the considered deep neural networks to represent a solution with a too complex structure in high dimension.

## 1·Introduction

This paper is devoted to the resolution in high dimension of nonlinear parabolic partial differential equations (PDEs) of the form 

$$
\label{eq:PDEInit} \left\{ 

$$
\begin{aligned}
\partial_t u + \Lc u + f(.,.,u,\sigma\trans D_x u) & = 0 , \;\;\;\;\;\; \mbox{ on } [ 0,T)\times\R^d, \\ u(T,.) &=g, \;\;\;\;\; \mbox{ on } \R^d, 
\end{aligned}
$$

\right. 
$$

with a non-linearity in the solution and its gradient via the function $f(t,x,y,z)$ defined on $[0,T]\times\R^d\times\R\times\R^d$, a terminal condition $g$, and a second-order generator $\Lc$ defined by 

$$
\begin{aligned}
\label{eq:PDE} \mathcal{L}u & := \frac{1}{2} \Tr \big(\sigma \sigma\trans D_x^2 u \big) + \mu.D_x u. 
\end{aligned}
$$

Here $\mu$ is a function defined on $[0,T] \times \R^d$ with values in $\R^d$, $\sigma$ is a function defined on $[0,T] \times \R^d$ with values in $\M^d$ the set of $d \times d$ matrices, and $\mathcal{L}$ is the generator associated to the forward diffusion process: 
\begin{flalign} \Xc_t = x_0 + \int_0^t \mu(s,\Xc_s) \diff s+ \int_0^t \sigma(s,\Xc_s) \diff W_s, \;\;\; 0 \leq t \leq T, \label{eq:SDE} \end{flalign}
with $W$ a $d$-dimensional Brownian motion on some probability space $(\Omega,\Fc,\P)$ equipped with a filtration $\F$ $=$ $(\Fc_t)_{0\leq t\leq T}$ satisfying the usual conditions.

Due to the so called "curse of dimensionality", the resolution of nonlinear PDEs in high dimension has always been a challenge for scientists.

Until recently, only the BSDE (Backward Stochastic Differential Equation) approach first developed in [^Pardoux1990adapted] was available to tackle this problem: using the time discretization scheme proposed in [^Bouchard2004discrete], some effective algorithms based on regressions manage to solve non linear PDEs in dimension above 4 (see [^Gobet2005regression], [^Lemor2006rate]).

However this approach is still not implementable in dimension above 6 or 7 : the number of basis functions used for the regression still explodes with the dimension.

Quite recently some new methods have been developed for this problem, and several methodologies have emerged: 

-  Some are based on the Feyman-Kac representation of the PDE.

Branching techniques [^Henry2016branching] have been studied and shown to be convergent but only for small maturities and some small nonlinearities.

Some effective techniques based on nesting Monte Carlo have been studied in [^Warin2018nestingsMC], [^Warin2018monte]: the convergence is proved for semi-linear equations.

Still based on this Feyman-Kac representation some machine learning techniques permitting to solve a fixed point problem have been used recently in [^Chan2018machine]: numerical results show that it is efficient and some partial demonstrations justify why it is effective. 
-  Multilevel Picard methods have been developed in [^Hutetal18] and [^Hutzenthaler2018overcoming] with algorithms based on Picard iterations, multi-level techniques and automatic differentiation.

These methods permit to handle some high dimensional PDEs with non linearity in $u$ and its gradient $D_x u$, with convergence results as well as numerous numerical examples showing their efficiency in high dimension. 
-  Another class of methods is based on the BSDE approach and the curse of dimensionality issue is partially avoided by using some machine learning techniques.

The pioneering papers [^Han2017overcoming], [^Weinan2017deep] propose a neural-networks based technique called \emph{Deep BSDE}, which was the first serious attempt for using machine learning methods to solve high dimensional PDEs.

Based on an Euler discretization of the forward underlying SDE $\Xc_t$, the idea is to view the BSDE as a forward SDE, and the algorithm tries to learn the values $u$ and $z$ $=$ $\sigma\trans Du$ at each time step of the Euler scheme by minimizing a global loss function between the forward simulation of $u$ till maturity $T$ and the target $g(\Xc_T)$.

This deep learning approximation has been extended to the case of fully nonlinear PDE and second order BSDE in [^Becketal19]. 
-  At last, using some machine learning representation of the solution, [^Sirignano2018dgm] proposes with the so-called Deep Galerkin Method to use the automatic numerical differentiation of the solution to solve the PDE on a finite domain.

The authors prove the convergence of their method but without information on the rate of convergence. 

Like the second methodology, our approach relies on BSDE representation of the PDE and deep learning approximations: we first discretize the BSDE associated to the PDE by an Euler scheme, but in contrast with [^Weinan2017deep], we adopt a classical backward resolution technique.

On each time step, we propose to use some machine learning techniques to estimate simultaneously the solution and its gradient by minimizing a loss function defined recursively by backward induction, and solving this local problem by a stochastic gradient algorithm.

Two different schemes are designed to deal with the local problems: 

- [(1)] The first one tries the estimate the solution and its gradient by a neural network. 
- [(2)] The second one tries only to approximate the solution by a neural network while its gradient is estimated directly with some numerical differentiation techniques. 

The proposed methodology is then extended to solve some variational inequalities, i.e., free boundary problems related to optimal stopping problems.

We mention that the related recent paper [^Becker] also proposes deep learning method for solving optimal stopping problems, but differently from our method, it relies on the approximation of (randomised) stopping decisions with a sequence of multilayer feedforward neural networks.

Convergence analysis of the two schemes for PDEs and variational inequalities is provided and shows that the approximation error goes to zero as we increase the number of time steps and the number of neurons/layers whenever the gradient descent method used to solve the local problems is not trapped in a local minimum.

Notice that similar convergence result for the deep BSDE method has been also obtained in [^Hanlong18] with a posteriori error estimation of the solution in terms of the universal approximation capability of global neural networks.

In the last part of the paper, we test our algorithms on different examples.

When the solution is easy to represent by a neural network, we can solve the problem in quite high dimension (at least $50$ in our numerical tests).

We show that the proposed methodology improves the algorithm proposed in [^Han2017overcoming] that sometimes does not converge or is trapped in a local minimum far away from the true solution.

We then show that when the solution has a very complex structure, we can still solve the problem but only in moderate dimension: the neural network used is not anymore able to represent the solution accurately in very high dimension.

Finally, we illustrate numerically that the method is effective to solve some system of variational inequalities: we consider the problem of American options and show that it can be solved very accurately in high dimension (we tested until $40$).

The outline of the paper is organized as follows.

In Section [secNN](#secNN), we give a brief and useful reminder for neural networks.

We describe in Section [secalgo](#secalgo) our two numerical schemes and compare with the algorithm in [^Han2017overcoming].

Section [secconv](#secconv) is devoted to the convergence analysis of our machine learning algorithms, and we present in Section [secnum](#secnum) several numerical tests.

