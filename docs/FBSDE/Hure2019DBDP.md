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

## 2·Neural networks as function approximators

\label{secNN}

Multilayer (also called deep) neural networks are designed to approximate unknown or large class of functions.

In contrast to additive approximation theory with weighted sum over basis functions, e.g. polynomials, neural networks rely on the composition of simple functions, and appear to provide an efficient way to handle high-dimensional approximation problems, in particular thanks to the increase in computer power for finding the "optimal" parameters by (stochastic) gradient descent methods.

We shall consider feedforward (or artificial) neural networks, which represent the basic type of deep neural networks.

Let us recall some notation and basic definitions that will be useful in our context.

We fix the input dimension $d_0$ $=$ $d$ (here the dimension of the state variable $x$), the output dimension $d_1$ (here $d_1$ $=$ $1$ for approximating the real-valued solution to the PDE, or $d_1$ $=$ $d$ for approximating the vector-valued gradient function), the global number $L+1$ $\in$ $\N\setminus\{1,2\}$ of layers with $m_\ell$, $\ell$ $=$ $0,\ldots,L$, the number of neurons (units or nodes) on each layer: the first layer is the input layer with $m_0$ $=$ $d$, the last layer is the output layer with $m_L$ $=$ $d_1$, and the $L-1$ layers between are called hidden layers, where we choose for simplicity the same dimension $m_\ell$ $=$ $m$, $\ell$ $=$ $1,\ldots,L-1$.

A feedforward neural network is a function from $\R^{d}$ to $\R^{d_1}$ defined as the composition 

$$
\begin{aligned}
\label{defNN} x \in \R^d & \longmapsto \; A_L \circ \varrho \circ A_{L - 1} \circ \ldots \circ \varrho \circ A_1(x) \; \in \; \R^{d_1}. 
\end{aligned}
$$

Here $A_\ell$, $\ell$ $=$ $1,\ldots,L$ are affine transformations: $A_1$ maps from $\R^d$ to $\R^m$, $A_2,\ldots,A_{L-1}$ map from $\R^m$ to $\R^m$, and $A_L$ maps from $\R^m$ to $\R^{d_1}$, represented by 

$$
\begin{aligned}

A_\ell (x) &= \; \Wc_\ell x + \beta_\ell, 
\end{aligned}
$$

for a matrix $\Wc_\ell$ called weight, and a vector $\beta_\ell$ called bias term, $\varrho$ $:$ $\R$ $\rightarrow$ $\R$ is a nonlinear function, called activation function, and applied component-wise on the outputs of $A_\ell$, i.e., $\varrho(x_1,\ldots,x_m)$ $=$ $(\varrho(x_1),\ldots,\varrho(x_m))$.

Standard examples of activation functions are the sigmoid, the ReLu, the Elu, $\tanh$.

All these matrices $\Wc_\ell$ and vectors $\beta_\ell$, $\ell$ $=$ $1,\ldots,L$, are the parameters of the neural network, and can be identified with an element $\theta$ $\in$ $\R^{N_m}$, where $N_m$ $=$ $\sum_{\ell=0}^{L-1} m_\ell (1+m_{\ell+1})$ $=$ $d(1+m)+m(1+m)(L-2)+m(1+d_1)$ is the number of parameters, where we fix $d_0$, $d_1$, $L$, but allow growing number $m$ of hidden neurons.

We denote by $\Theta_m$ the set of possible parameters: in the sequel, we shall consider either the case when there are no constraints on parameters, i.e., $\Theta_m$ $=$ $\R^{N_m}$, or when the total variation norm of the neural networks is smaller than $\gamma_m$, i.e., 

$$
\begin{aligned}
\Theta_m & = \Theta_m^\gamma \; := \; \big\{ \theta = (\Wc_\ell,\beta_\ell)_\ell: |\Wc_l| \leq \gamma_m, \;\; \ell = 1,\ldots,L \big\}, \;\; \mbox{ with} \; \gamma_m \nearrow \infty, \mbox{ as } m \rightarrow \infty. 
\end{aligned}
$$

We denote by $\Phi_{_m}(.;\theta)$ the neural network function defined in \eqref{defNN}, and by $\Nc\Nc_{d,d_1,L,m}^\varrho(\Theta_m)$ the set of all such neural networks $\Phi_{_m}(.;\theta)$ for $\theta$ $\in$ $\Theta_m$, and set 
\begin{eqnarray*} \Nc\Nc_{d,d_1,L}^\varrho &= & \bigcup_{m \in \N} \Nc\Nc_{d,d_1,L,m}^\varrho(\Theta_m) \; = \; \bigcup_{m \in \N} \Nc\Nc_{d,d_1,L,m}^\varrho(\R^{N_m}), \end{eqnarray*}
as the class of all neural networks within a fixed structure given by $d$, $d_1$, $L$ and $\varrho$.

The fundamental result of Hornick et al. [^Horetal89] justifies the use of neural networks as function approximators: \vspace{1mm} \noindent {\bf Universal approximation theorem (I)}: $\Nc\Nc_{d,d_1,L}^\varrho$ is dense in $L^2(\nu)$ for any finite measure $\nu$ on $\R^d$, whenever $\varrho$ is continuous and non-constant. \vspace{1mm}

Moreover, we have a universal approximation result for the derivatives in the case of a single hidden layer, i.e. $L$ $=$ $2$, and when the activation function is a smooth function, see [^Hor90universal]. \vspace{1mm} \noindent {\bf Universal approximation theorem (II)}: Assume that $\varrho$ is a (non constant) $C^k$ function.

Then, $\Nc\Nc_{d,d_1,2}^\varrho$ approximates any function and its derivatives up to order $k$, arbitrary well on any compact set of $\R^d$.

## 3·Deep learning-based schemes for semi-linear PDEs

\label{secalgo}

The starting point for our probabilistic numerical schemes to the PDE \eqref{eq:PDEInit} is the well-known (see [^Pardoux1990adapted]) nonlinear Feynman-Kac formula via the pair $(Y,Z)$ of $\F$-adapted processes valued in $\R\times\R^d$, solution to the BSDE 

$$
\begin{aligned}
\label{eqBSDE}

Y_t &= g(\Xc_T) + \int_t^T f(s,\Xc_s,Y_s,Z_s) \diff s - \int_t^T Z_s\trans \diff W_s, \;\;\; 0 \leq t \leq T, 
\end{aligned}
$$

related to the solution $u$ of \eqref{eq:PDEInit} via 

$$
\begin{aligned}

Y_t & = u(t,\Xc_t), \;\;\; 0 \leq t \leq T, 
\end{aligned}
$$

and when $u$ is smooth: 

$$
\begin{aligned}

Z_t & = \; \sigma\trans(t,\Xc_t)D_x u(t,\Xc_t), \;\;\; 0 \leq t \leq T. 
\end{aligned}
$$

### The deep BSDE scheme of [^Han2017overcoming]

The DBSDE algorithm proposed in [^Han2017overcoming], [^Weinan2017deep] starts from the BSDE representation \eqref{eqBSDE} of the solution to \eqref{eq:PDEInit}, but rewritten in forward form as: 

$$
\begin{aligned}
\label{eq:bsde} u(t,\Xc_t) = & \;\; u(0, x_0)- \int_0^t f(s,\Xc_s,u(s,\Xc_s),\sigma\trans(s,\Xc_s)D_xu(s,\Xc_s)) \diff s \\ & \;\;\; + \int_0^t D_xu(s,\Xc_s)\trans\sigma(s,\Xc_s) \diff W_s, \;\;\;\;\; 0 \leq t \leq T. 
\end{aligned}
$$

The forward process $\Xc$ in equation \eqref{eq:SDE}, when it is not simulatable, is numerically approximated by an Euler scheme $X$ $=$ $X^\pi$ on a time grid: $\pi$ $=$ $\{t_0=0<t_1< \ldots < t_N = T\}$, with modulus $|\pi|$ $=$ $\max_{i=0,\ldots,N-1}\Delta t_i$, $\Delta t_i$ $:=$ $t_{i+1}-t_i$, and defined as 

$$

X_{t_{i+1}} \; = \; X_{t_{i}} + \mu(t_i, X_{t_i}) \Delta t_{i} + \sigma(t_i, X_{t_i}) \Delta W_{t_i}, \;\;\; i=0,\ldots,N-1, \; X_0 = x_0, \label{eq:eulerSDE} 
$$

where we set $\Delta W_{t_i}$ $:=$ $W_{t_{i+1}} - W_{t_{i}}$.

To alleviate notations, we omit the dependence of $X$ $=$ $X^\pi$ on the time grid $\pi$ as there is no ambiguity (recall that we use the notation $\Xc$ for the forward diffusion process).

The approximation of equation \eqref{eq:PDEInit} is then given formally from the Euler scheme associated to the forward representation \eqref{eq:bsde} by 

$$
\begin{aligned}
\label{uF} u(t_{i+1}, X_{t_{i+1}}) & \approx F(t_i,X_{t_i},u(t_i,X_{t_i}),\sigma\trans(t_i,X_{t_i})D_x u(t_{i}, X_{t_{i}}),\Delta t_{i},\Delta W_{t_i}) 
\end{aligned}
$$

with 

$$
\begin{aligned}

F(t,x,y,z,h,\Delta) & := y - f(t,x,y,z)h + z\trans\Delta. \label{defF} 
\end{aligned}
$$

In [^Han2017overcoming], [^Weinan2017deep], the numerical approximation of $u(t_i,X_{t_i})$ is designed as follows: starting from an estimation $\Uc_0$ of $u(0,X_0)$, and then using at each time step $t_i$, $i$ $=$ $0,\ldots,N-1$, a multilayer neural network $x$ $\in$ $\R^d$ $\mapsto$ $\Zc_i(x;\theta_i)$ with parameter $\theta_i$ for the approximation of $x$ $\mapsto$ $\sigma\trans(t_i,x)D_x u(t_i,x)$: 

$$
\begin{aligned}
\label{NNti} \Zc_i(x;\theta_i) & \approx \sigma\trans(t_i,x)D_x u(t_i,x), 
\end{aligned}
$$

one computes estimations $\Uc_i$ of $u(t_i;X_{t_i})$ by forward induction via: 

$$
\begin{aligned}
\Uc_{i+1} &= F(t_i,X_{t_i},\Uc_i,\Zc_i(X_{t_i};\theta_i),\Delta t_i,\Delta W_{t_i}), 
\end{aligned}
$$

for $i$ $=$ $0,\ldots,N-1$.

This algorithm forms a global deep neural network composed of the neural networks \eqref{NNti} of each period, by taking as input data (in machine learning language) the paths of $(X_{t_i})_{i=0,\ldots,N}$ and $(W_{t_i})_{i=0,\ldots,N}$, and giving as output $\Uc_N$ $=$ $\Uc_N(\theta)$, which is a function of the input and of the total set of parameters $\theta$ $=$ $(\Uc_0,\theta_0,\ldots,\theta_{N-1})$.

The output aims to match the terminal condition $g(X_{t_N})$ of the BSDE, and one then optimizes over the parameter $\theta$ the expected square loss function: 

$$
\begin{aligned}
\theta & \mapsto \; \E \big| g(X_{t_N}) - \Uc_N(\theta) \big|^2. 
\end{aligned}
$$

This is obtained by stochastic gradient descent-type (SGD) algorithms relying on training input data. 

### New schemes: DBDP1 and DBDP2

\label{secnewML}

The proposed scheme is defined from a backward dynamic programming type relation, and has two versions: 

- [(1)] First version: 
\begin{itemize} 
- [-] Initialize from an estimation $\widehat\Uc_N^{(1)}$ of $u(t_N,.)$ with $\widehat\Uc_N^{(1)}$ $=$ $g$ 
- [-] For $i$ $=$ $N-1,\ldots,0$, given $\widehat\Uc_{i+1}^{(1)}$, use a pair of deep neural networks $(\Uc_i(.;\theta),\Zc_i(.;\theta))$ $\in$ $\Nc\Nc_{d,1,L,m}^\varrho(\R^{N_m})\times\Nc\Nc_{d,d,L,m}^\varrho(\R^{N_m})$ for the approximation of $(u(t_i,.),\sigma\trans(t_i,.)D_x u(t_i,.))$, and compute (by SGD) the minimizer of the expected quadratic loss function 

$$
\label{eq:scheme1} \left\{ 

$$
\begin{aligned}
\hat L_i^{(1)}(\theta) & := \E \Big| \widehat\Uc_{i+1}^{(1)}(X_{t_{i+1}}) - F(t_i,X_{t_i},\Uc_i(X_{t_i};\theta),\Zc_i(X_{t_i};\theta),\Delta t_i,\Delta W_{t_i}) \Big|^2 \\ \theta_i^* & \in {\rm arg}\min_{\theta\in\R^{N_m}} \hat L_i^1(\theta). 
\end{aligned}
$$

\right. 
$$

Then, update: $\widehat\Uc_i^{(1)}$ $=$ $\Uc_i(.;\theta_i^*)$, and set $\widehat\Zc_i^{(1)}$ $=$ $\Zc_i(.;\theta_i^*)$. 

- [(2)] Second version: 

-  Initialize with $\widehat\Uc_N^{(2)}$ $=$ $g$ 
-  For $i$ $=$ $N-1,\ldots,0$, given $\widehat\Uc_{i+1}^{(2)}$, use a deep neural network $\Uc_i(.;\theta)$ $\in$ $\Nc\Nc_{d,1,L,m}^\varrho(\Theta_m)$, and compute (by SGD) the minimizer of the expected quadratic loss function 

$$
\label{eq:scheme2} \left\{ 

$$
\begin{aligned}
\hat L_i^{(2)}(\theta) & := \E \Big| \widehat\Uc_{i+1}^{(2)}(X_{t_{i+1}}) - \\ & \quad \quad F(t_i,X_{t_i},\Uc_i(X_{t_i};\theta),\sigma\trans(t_i,X_{t_i}) \hat D_x \Uc_i(X_{t_i};\theta),\Delta t_i,\Delta W_{t_i}) \Big|^2 \\ \theta_i^* & \in {\rm arg}\min_{\theta\in\Theta_m} \hat L_i^2(\theta), 
\end{aligned}
$$

\right. 
$$

where $\hat D_x \Uc_i(.;\theta)$ is the numerical differentiation of $\Uc_i(.;\theta)$.

Then, update: $\widehat\Uc_i^{(2)}$ $=$ $\Uc_i(.;\theta_i^*)$, and set $\widehat\Zc_i^{(2)}$ $=$ $\sigma\trans(t_i,.) \hat D_x \Uc_i(.;\theta_i^*)$. 

\end{itemize}

\begin{Remark} \label{remNN} {\rm For the first version of the scheme, one can use independent neural networks, respectively for the approximation of $u(t_i,.)$ and for the approximation of $\sigma\trans(t_i,.)D_xu(t_i,.)$.

In other words, the parameters are divided into a pair $\theta$ $=$ $(\xi,\eta)$ and we consider neural networks $\Uc_i(.;\xi)$ and $\Zc_i(.;\eta)$. } \ep \end{Remark}

In the sequel, we refer to the first and second version of the new scheme above as DBDP1 and DBDP2, where the acronym DBDP stands for deep learning backward dynamic programming.

The intuition behind DBDP1 and DBDP2 is the following.

For simplicity, take $f$ $=$ $0$, so that $F(t,x,y,z,h,\Delta)$ $=$ $y+z\trans\Delta$.

The solution $u$ to the PDE \eqref{eq:PDEInit} should then approximately satisfy (see \eqref{uF}) 

$$
\begin{aligned}
u(t_{i+1}, X_{t_{i+1}}) & \approx \; u(t_i,X_{t_i}) + D_x u(t_i,X_{t_i})\trans \sigma(t_i,X_{t_i}) \Delta W_{t_i}. 
\end{aligned}
$$

Consider the first scheme DBDP1, and suppose that at time $i+1$, $\widehat\Uc_{i+1}^{(1)}$ is an estimation of $u(t_{i+1,.})$.

The quadratic loss function at time $i$ is then approximately equal to 

$$
\begin{aligned}
\hat L_i^{(1)}(\theta) & \approx \; \E \Big| u(t_{i+1}, X_{t_{i+1}}) - \Uc_i(X_{t_i};\theta) - \Zc_i(X_{t_i};\theta)\trans\Delta W_{t_i} \Big|^2 \\ & \approx \; \E \Big[ \big| u(t_i,X_{t_i}) - \Uc_i(X_{t_i};\theta) \big|^2 + \Delta t_i \big| \sigma\trans(t_i,X_{t_i}) D_x u(t_i,X_{t_i}) - \Zc_i(X_{t_i};\theta) \big|^2 \Big]. 
\end{aligned}
$$

Therefore, by minimizing over $\theta$ this quadratic loss function, via SGD based on simulations of $(X_{t_i},X_{t_{i+1}},\Delta W_{t_i})$ (called training data in the machine learning language), one expects the neural networks $\Uc_i$ and $\Zc_i$ to learn/approximate better and better the functions $u(t_i,.)$ and $\sigma\trans(t_i,)D_x u(t_i,)$ in view of the universal approximation theorem [^Hor90universal].

Similarly, the second scheme DPDP2, which uses only neural network on the value functions, learns $u(t_i,.)$ by means of the neural network $\Uc_i$, and $\sigma\trans(t_i,)D_x u(t_i,)$ via $\sigma\trans(t_i,)\hat D_x \Uc_i$.

The rigorous arguments for the convergence of these schemes will be derived in the next section. \vspace{2mm}

The advantages of our two schemes, compared to the Deep BSDE algorithm, are the following: 

-  by decomposing the global problem into smaller ones, we may expect to help the gradient descent method to provide estimations closer to the real solution.

The memory needed in [^Han2017overcoming] can be a problem when taking too many time steps. 
-  at each time step, we initialize the weights and bias of the neural network to the weights and bias of the previous time step treated : this trick is commonly used in iterative solvers of PDE, and allows us to start with a value close to the solution, hence avoiding local minima which are too far away from the true solution.

Besides the number of gradient iterations to achieve is rather small after the first resolution step. 

The small disadvantage is due to the Tensorflow structure.

As it is done in python, the global graph creation takes much time as it is repeated for each time step and the global resolution is a little bit time consuming : as the dimension of the problem increases, the time difference decreases and it becomes hard to compare the computational time for a given accuracy when the dimension is above 5. 

### Extension to variational inequalities: scheme RDBDP

\label{sec:varIneq}

Let us consider a variational inequality in the form 

$$
\label{eq:IQV} \left\{ 

$$
\begin{aligned}
\min \big[ - \partial_t u - \Lc u - f(t,x,u,\sigma\trans D_x u) , u - g \big] & = 0 , \;\;\;\;\;\;\; t \in [0,T), \; x \in \R^d, \\ u(T,x) &=g(x), \;\;\; x\in\R^d. 
\end{aligned}
$$

\right. 
$$

which arises, e.g., in optimal stopping problem and American option pricing in finance.

It is known, see e.g. [^Elk97], that such variational inequality is related to reflected BSDE of the form 

$$
\begin{aligned}
\label{RBSDE}

Y_t &= \; g(\Xc_T) + \int_t^T f(s,\Xc_s,Y_s,Z_s) \diff s - \int_t^T Z_s\trans \diff W_s + K_T - K_t, \\ Y_t & \geq \; g(X_t), \;\;\; 0 \leq t \leq T, 
\end{aligned}
$$

where $K$ is an adapted non-decreasing process satisfying 

$$
\begin{aligned}
\int_0^T \big(Y_t - g(X_t) \big) dK_t & = \; 0. 
\end{aligned}
$$

The extension of our DBDP1 scheme for such variational inequality, and refereed to as RDBDP scheme, becomes 

-  Initialize $\widehat\Uc_N$ $=$ $g$ 
-  For $i$ $=$ $N-1,\ldots,0$, given $\widehat\Uc_{i+1}$, use a pair of (multilayer) neural network $(\Uc_i(.;\theta),\Zc_i(.;\theta))$ $\in$ $\Nc\Nc_{d,1,L,m}^\varrho(\R^{N_m})\times\Nc\Nc_{d,d,L,m}^\varrho(\R^{N_m})$, and compute (by SGD) the minimizer of the expected quadratic loss function 

$$
\label{eq:schemeVI} \left\{ 

$$
\begin{aligned}
\hat L_i(\theta) & := \E \big| \widehat\Uc_{i+1}(X_{t_{i+1}}) - F(t_i,X_{t_i},\Uc_i(X_{t_i};\theta),\Zc_i(X_{t_i};\theta),\Delta t_i,\Delta W_{t_i}) \big|^2 \\ \theta_i^* & \in {\rm arg}\min_{\theta\in\R^{N_m}} \hat L_i(\theta). 
\end{aligned}
$$

\right. 
$$

Then, update: $\widehat\Uc_i$ $=$ $\max\big[\Uc_i(.;\theta_i^*),g]$, and set $\hat\Zc_i$ $=$ $\Zc(.;\theta_i ^*)$. 

## 4·Convergence analysis

\label{secconv}

The main goal of this section is to prove convergence of the DBDP schemes towards the solution $(Y,Z)$ to the BSDE \eqref{eqBSDE} (or reflected BSDE \eqref{RBSDE} for variational inequalities), and to provide a rate of convergence that depends on the approximation errors by neural networks. 

### Convergence of DBDP1

We assume the standard Lipschitz conditions on $\mu$ and $\sigma$, which ensures the existence and uniqueness of an adapted solution $\Xc$ to the forward SDE \eqref{eq:SDE} satisfying for any $p$ $>$ $1$, 

$$
\begin{aligned}
\label{integX} \E \big[ \sup_{0\leq t \leq T} |\Xc_t|^p \big] & \; < C_p(1 + |x_0|^p), 
\end{aligned}
$$

for some constant $C_p$ depending only on $p$, $b$, $\sigma$ and $T$.

Moreover, we have the well-known error estimate with the Euler scheme $X$ $=$ $X^\pi$ defined in \eqref{eq:eulerSDE} with a time grid $\pi$ $=$ $\{t_0=0<t_1< \ldots < t_N = T\}$, with modulus $|\pi|$ s.t. $N|\pi|$ is bounded by a constant depending only on $T$ (hence independent of $N$): 

$$
\begin{aligned}
\label{estimEulerX} \max_{i=0,\ldots,N-1} \E \Big[ |\Xc_{t_{i+1}} - X_{t_{i+1}}|^2 + \sup_{t\in[t_i,t_{i+1}]} | \Xc_t - X_{t_i}|^2 \Big] & = \; O(|\pi|). 
\end{aligned}
$$

Here, the standard notation $O(|\pi|)$ means that $\limsup_{|\pi| \rightarrow 0} \; |\pi|^{-1}

O(|\pi|)$ $<$ $\infty$. \vspace{1mm}

We shall make the standing usual assumptions on the driver $f$ and the terminal data $g$. \vspace{2mm} \noindent \textbf{(H1)} (i) There exists a constant $[f]_{_L}>0$ such that the driver $f$ satisfies: 

$$
\left| f(t_2,x_2,y_2,z_2) - f(t_1,x_1,y_1,z_1) \right| \leq [f]_{_L} \left( |t_2-t_1 |^{1/2} + |x_2-x_1| +|y_2-y_1| +|z_2-z_1| \right), 
$$

for all $(t_1,x_1,y_1,z_1)$ and $(t_2,x_2,y_2,z_2)$ $\in [0,T] \times \R^d \times \R \times \R^d$.

Moreover, 

$$
\sup_{0 \leq t \leq T} |f(t,0,0,0)| < \infty. 
$$

(ii) The function $g$ satisfies a linear growth condition. \vspace{3mm}

Recall that Assumption \textbf{(H1)} ensures the existence and uniqueness of an adapted solution $(Y,Z)$ to \eqref{eqBSDE} satisfying 

$$
\begin{aligned}
\label{estiYZ} \E \Big[ \sup_{0\leq t \leq T} |Y_t|^2 + \int_0^T |Z_t|^2 \diff t \Big] & < \; \infty. 
\end{aligned}
$$

From the linear growth condition on $f$ in {\bf (H1)}, and \eqref{integX}, we also see that 

$$
\begin{aligned}
\label{integf2} \E \Big[ \int_0^T |f(t,\Xc_t,Y_t,Z_t)|^2 \diff t \Big] & < \; \infty. 
\end{aligned}
$$

Moreover, we have the standard $L^2$-regularity result on $Y$: 

$$
\begin{aligned}
\label{regulY} \max_{i=0,\ldots,N-1} \E \Big[ \sup_{t\in[t_i,t_{i+1}]} | Y_t - Y_{t_i}|^2 \Big] & = \; O(|\pi|). 
\end{aligned}
$$

Let us also introduce the $L^2$-regularity of $Z$: 

$$
\begin{aligned}
\eps^Z(\pi) & := \; \E \bigg[ \sum_{i=0}^{N-1} \int_{t_i}^{t_{i+1}} |Z_t - \bar Z_{t_i}|^2 dt \bigg], \;\;\; \mbox{ with } \; \bar Z_{t_i} \; := \; \frac{1}{\Delta t_i} \E_i \Big[ \int_{t_i}^{t_{i+1}}

Z_t dt \Big], 
\end{aligned}
$$

where $\E_i$ denotes the conditional expectation given $\Fc_{t_i}$.

Since $\bar Z$ is a $L^2$-projection of $Z$, we know that $\eps^Z(\pi)$ converges to zero when $|\pi|$ goes to zero.

Moreover, as shown in [^Zhang04numerical], when the terminal condition $g$ is also Lipschitz, we have 

$$
\begin{aligned}
\eps^Z(\pi) & = \; O(|\pi|). 
\end{aligned}
$$

Let us first investigate the convergence of the scheme DBDP1 in \eqref{eq:scheme1}, and define (implicitly) 

$$
\label{defVCZ} \left\{ 
\begin{array}{rcl} \widehat\Vc_{t_i} & := & \E_i \big[ \widehat\Uc_{i+1}^{(1)}(X_{t_{i+1}}) \big] + f(t_i,X_{t_i},\widehat\Vc_{t_i},\overline{{\widehat Z_{t_i}}}) \Delta t_i \\ \overline{{\widehat Z_{t_i}}} & := & \frac{1}{\Delta t_i} \E_i\left[ \widehat\Uc_{i+1}^{(1)}(X_{t_{i+1}}) \Delta W_{t_i} \right], \end{array}
\right. 
$$

for $i$ $=$ $0,\ldots,N-1$.

Notice that $\widehat\Vc_{t_i}$ is well-defined for $|\pi|$ small enough (recall that $f$ is Lipschitz) by a fixed point argument.

By the Markov property of the discretized forward process $(X_{t_i})_{i=0,\ldots,N}$, we note that there exists some deterministic functions $\hat v_i$ and $\overline{{\hat z_i}}$ s.t. 

$$
\begin{aligned}
\label{defhatv1} \widehat\Vc_{t_i}^{} \; = \; \hat v_i^{}(X_{t_i}), & \mbox{ and } \;\; \overline{{\widehat Z_{t_i}}^{}} \; = \; \overline{{\hat z_i}^{}}(X_{t_i}), \;\;\;\;\; i =0,\ldots,N-1. 
\end{aligned}
$$

Moreover, by the martingale representation theorem, there exists an $\R^d$-valued square integrable process $(\widehat Z_t)_t$ such that 

$$
\begin{aligned}
\label{FBSDE} \widehat\Uc_{i+1}^{(1)}(X_{t_{i+1}}) & = \; \widehat\Vc_{t_i} - f(t_i,X_{t_i},\widehat\Vc_{t_i} ,\overline{{\widehat Z_{t_i}}}) \Delta t_i + \int_{t_i}^{t_{i+1}} \widehat Z_s\trans \diff W_s, 
\end{aligned}
$$

and by It\^o isometry, we have 

$$
\begin{aligned}
\label{ZbarZ} \overline{{\widehat Z_{t_i}}} &= \; \frac{1}{\Delta t_i} \E_i \Big[ \int_{t_i}^{t_{i+1}} \widehat Z_s \diff s \Big], \;\;\;\;\; i=0,\ldots,N-1. 
\end{aligned}
$$

\vspace{2mm}

Let us now define a measure of the (squared) error for the DBDP1 scheme by 

$$
\begin{aligned}
\Ec\big[(\widehat\Uc^{(1)},\widehat\Zc^{(1)}),(Y,Z)\big] & := \; \max_{i=0,\ldots,N-1} \E \big|Y_{t_i}- \widehat\Uc_i^{(1)}(X_{t_i})\big|^2 + \E \bigg[ \sum_{i=0}^{N-1} \int_{t_i}^{t_{i+1}} \big| Z_t - \widehat\Zc_i^{(1)}(X_{t_i}) \big|^2 dt \bigg]. 
\end{aligned}
$$

Our first main result gives an error estimate of the DBDP1 scheme in terms of the $L^2$-approximation errors of $\hat v_i$ and $\overline{{\hat z_i}}$ by neural networks $\Uc_i$ and $\Zc_i$, $i=0,\ldots,N-1$, assumed to be independent (see Remark [remNN](#remNN)), and defined as 

$$
\begin{aligned}
\eps_i^{\Nc,v} \; := \; \inf_{\xi} \E \big|\hat v_i(X_{t_i}) - \Uc_i(X_{t_i};\xi) \big|^2 , \hspace{7mm} \eps_i^{\Nc,z} \; := \; \inf_{\eta} \E \big|\overline{{\hat z_i}^{}}(X_{t_i}) - \Zc_i(X_{t_i};\eta) \big|^2. 
\end{aligned}
$$

Here, we fix the structure of the neural networks with input dimension $d$, output dimension $d_1$ $=$ $1$ for $\Uc_i$, and $d_1$ $=$ $d$ for $\Zc_i$, number of layers $L$, and $m$ neurons for the hidden layers, and the parameters vary in the whole set $\R^{N_m}$ where $N_m$ is the number of parameters.

From the universal approximation theorem (I) ([^Horetal89]), we know that $\eps_i^{N N,v}$ and $\eps_i^{NN,z}$ converge to zero as $m$ goes to infinity, hence can be made arbitrary small for sufficiently large number of neurons. 
\begin{Theorem} \emph{(Consistency of DBDP1)} \label{theo:scheme1_1}

Under {\bf (H1)}, there exists a constant $C>0$, independent of $\pi$, such that 

$$
\begin{aligned}
\Ec\big[(\widehat\Uc^{(1)},\widehat\Zc^{(1)}),(Y,Z)\big] & \leq \; C \Big( \E \big|g(\Xc_{T}) - g(X_T) \big|^2 + |\pi| + \eps^Z(\pi) \\ & \hspace{9mm} + \; \sum_{i=0}^{N-1} \big(N \eps_i^{\Nc,v} + \eps_i^{\Nc,z}\big) \Big). \label{eq:theo1_scheme1} 
\end{aligned}
$$

\end{Theorem}

\begin{Remark} {\rm The error contributions for the DBDP1 scheme in the r.h.s. of estimation \eqref{eq:theo1_scheme1} consists of four terms.

The first three terms correspond to the time discretization of BSDE, similarly as in [^Bouchard2004discrete], [^Gobet2005regression], namely (i) the strong approximation of the terminal condition (depending on the forward scheme and the terminal data $g$), and converging to zero, as $|\pi|$ goes to zero, with a rate $|\pi|$ when $g$ is Lipschitz by \eqref{estimEulerX} (see [^Avi09] for irregular $g$), (ii) the strong approximation of the forward Euler scheme, and the $L^2$-regularity of $Y$, which gives a convergence of order $|\pi|$, (iii) the $L^2$-regularity of $Z$, which converges to zero, as $|\pi|$ goes to zero, with a rate $|\pi|$ when $g$ is Lipschitz.

Finally, the better the neural networks are able to approximate/learn the functions $\hat v_i$ and $\overline{{\hat z_i}}$ at each time $i$ $=$ $0,\ldots,N-1$, the smaller is the last term in the error estimation.

Moreover, given a prescribed accuracy for the neural network approximation error, the number of parameters of the employed deep neural networks grows at most polynomially in the PDE dimension, as recently proved in [^Hutetal19] in the case of semi-linear heat equations. } \ep \end{Remark}
\vspace{3mm} \noindent {\bf Proof of Theorem [theo:scheme1_1](#theo:scheme1_1).} \noindent In the following, $C$ will denote a positive generic constant independent of $\pi$, and that may take different values from line to line. \vspace{1mm} \noindent {\it Step 1}.

Fix $i$ $\in$ $\{0,\ldots,N-1\}$, and observe by \eqref{eqBSDE}, \eqref{defVCZ} that 

$$
\begin{aligned}
\label{reldifY} \hspace{-5mm}

Y_{t_i} - \widehat\Vc_{t_i} & = \E_i\big[ Y_{t_{i+1}} - \widehat\Uc_{i+1}^{(1)}(X_{t_{i+1}}) \big] + \E_i\Big[ \int_{t_i}^{t_{i+1}} f(t,\Xc_t,Y_t,Z_t) - f(t_i,X_{t_i},\widehat\Vc_{t_i},\overline{{\widehat Z_{t_i}}}) \diff t \Big]. 
\end{aligned}
$$

By using Young inequality: $(a+b)^2$ $\leq$ $(1+\gamma \Delta t_i)a^2$ $+$ $(1+\frac{1}{\gamma \Delta t_i})b^2$ for some $\gamma$ $>$ $0$ to be chosen later, Cauchy-Schwarz inequality, the Lipschitz condition on $f$ in {\bf (H1)}, and the estimation \eqref{estimEulerX} on the forward process, we then have 

$$
\begin{aligned}
\E\big| Y_{t_i} - \widehat\Vc_{t_i} \big|^2 & \leq \; (1 +\gamma\Delta t_i) \E \Big| \E_i\big[ Y_{t_{i+1}} - \widehat\Uc_{i+1}^{(1)}(X_{t_{i+1}}) \big] \Big|^2 \\ & \;\;\; + 4 [f]^2_{_L} \Delta t_i \big(1+\frac{1}{\gamma \Delta t_i}\big) \Big\{ |\Delta t_i|^2 + \E\Big[ \int_{t_i}^{t_{i+1}} \big| Y_t - \widehat\Vc_{t_i} \big|^2 \diff t \Big] \\ & \hspace{4.3cm} + \; \E\Big[ \int_{t_i}^{t_{i+1}} \big| Z_t - \overline{{\widehat Z_{t_i}}} \big|^2 \diff t \Big] \Big\} \\ & \leq \; (1 +\gamma\Delta t_i) \E \Big| \E_i\big[ Y_{t_{i+1}} - \hat\Uc_{i+1}^{(1)}(X_{t_{i+1}}) \big] \Big|^2 \label{inegYVi} \\ & \;\;\; + 4 \frac{[f]^2_{_L}}{\gamma} (1+ \gamma \Delta t_i) \Big\{ C |\pi|^2 + 2 \Delta t_i \E\big| Y_{t_i} - \widehat\Vc_{t_i} \big|^2 + \E\Big[ \int_{t_i}^{t_{i+1}} \big| Z_t - \overline{{\widehat Z_{t_i}}} \big|^2 \diff t \Big] \Big\}, 
\end{aligned}
$$

where we use in the last inequality the $L^2$-regularity \eqref{regulY} of $Y$.

Recalling the definition of $\bar Z$ as a $L^2$-projection of $Z$, we observe that 

$$
\begin{aligned}
\label{Pythagore} \E\Big[ \int_{t_i}^{t_{i+1}} \big| Z_t - \overline{{\widehat Z_{t_i}}} \big|^2 \diff t \Big] & = \; \E \Big[\int_{t_i}^{t_{i+1}} \big| Z_t - \bar Z_{t_i}\big|^2 \diff t \Big] + \Delta t_i \E \big|\bar Z_{t_i} - \overline{{\widehat Z_{t_i}}} \big|^2. 
\end{aligned}
$$

By multiplying equation \eqref{eqBSDE} between $t_i$ and $t_{i+1}$ by $\Delta W_{t_i}$, and using It\^o isometry, we have together with \eqref{defVCZ} 

$$
\begin{aligned}
\Delta t_i \big( \bar Z_{t_i} - \overline{{\widehat Z_{t_i}}} \big) & = \; \E_i \big[ \Delta W_{t_i} \big( Y_{t_{i+1}} - \widehat\Uc_{i+1}^{(1)}(X_{t_{i+1}}) \big) \big] + \E_i \Big[ \Delta W_{t_i} \int_{t_i}^{t_{i+1}} f(t,\Xc_t,Y_t,Z_t) \diff t \Big] \\ & = \; \E_i \Big[ \Delta W_{t_i} \Big( Y_{t_{i+1}} - \widehat\Uc_{i+1}^{(1)}(X_{t_{i+1}}) - \E_i\big[ Y_{t_{i+1}} - \widehat\Uc_{i+1}^{(1)}(X_{t_{i+1}})\big] \Big) \Big] \\ & \;\;\;\; + \; \E_i \Big[ \Delta W_{t_i} \int_{t_i}^{t_{i+1}} f(t,\Xc_t,Y_t,Z_t) \diff t \Big]. 
\end{aligned}
$$

By Cauchy-Schwarz inequality, and law of iterated conditional expectations, this implies 

$$
\begin{aligned}
\Delta t_i \E \big| \bar Z_{t_i} - \overline{{\widehat Z_{t_i}}} \big|^2 & \leq \; 2 d \Big( \E \big|Y_{t_{i+1}} - \widehat\Uc_{i+1}^{(1)}(X_{t_{i+1}}) \big|^2 - \E \Big| \E_i\big[ Y_{t_{i+1}} - \widehat\Uc_{i+1}^{(1)}(X_{t_{i+1}})\big] \Big|^2 \Big) \\ & \;\;\;\;\; + 2 d \Delta t_i \E \Big[ \int_{t_i}^{t_{i+1}} |f(t,\Xc_t,Y_t,Z_t)|^2 \diff t \Big]. \label{inegZi} 
\end{aligned}
$$

Then, by plugging \eqref{Pythagore} and \eqref{inegZi} into \eqref{inegYVi}, and choosing $\gamma$ $=$ $8 d [f]^2_{_L}$, we have 

$$
\begin{aligned}
\E\big| Y_{t_i} - \widehat\Vc_{t_i} \big|^2 & \leq \; C \Delta t_i \E\big| Y_{t_i} - \widehat\Vc_{t_i} \big|^2 + (1 +\gamma\Delta t_i) \E \big|Y_{t_{i+1}} - \widehat\Uc_{i+1}^{(1)}(X_{t_{i+1}}) \big|^2 + C |\pi|^2 \\ & \;\;\; + \; C \E \Big[\int_{t_i}^{t_{i+1}} \big| Z_t - \bar Z_{t_i}\big|^2 \diff t \Big] + C \Delta t_i \E \Big[ \int_{t_i}^{t_{i+1}} |f(t,\Xc_t,Y_t,Z_t)|^2 \diff t \Big], 
\end{aligned}
$$

and thus for $|\pi|$ small enough: 

$$
\begin{aligned}
\E\big| Y_{t_i} - \widehat\Vc_{t_i} \big|^2 & \leq \; (1 + C |\pi|) \E \big|Y_{t_{i+1}} - \widehat\Uc_{i+1}^{(1)}(X_{t_{i+1}}) \big|^2 + C |\pi|^2 \\ & \hspace{.2cm}+ C \E \Big[\int_{t_i}^{t_{i+1}} \big| Z_t - \bar Z_{t_i}\big|^2 \diff t \Big] + C |\pi| \E \Big[ \int_{t_i}^{t_{i+1}} |f(t,\Xc_t,Y_t,Z_t)|^2 \diff t \Big]. \label{interYUV} 
\end{aligned}
$$

\vspace{1mm} \noindent {\it Step 2.}

By using Young inequality in the form: $(a+b)^2$ $\geq$ $(1- |\pi|)a^2$ $+$ $(1-\frac{1}{|\pi|})b^2$ $\geq$ $(1- |\pi|)a^2$ $-$ $\frac{1}{ |\pi|}b^2$, we have 

$$
\begin{aligned}
\E\big| Y_{t_i} - \widehat\Vc_{t_i} \big|^2 & = \E\big| Y_{t_i} - \widehat\Uc_{i}^{(1)}(X_{t_{i}}) + \widehat\Uc_{i}^{(1)}(X_{t_{i}}) - \widehat\Vc_{t_i} \big|^2 \\ & \geq \; (1- |\pi|) \E\big| Y_{t_i} - \widehat\Uc_{i}^{(1)}(X_{t_{i}}) \big|^2 - \frac{1}{ |\pi|} \E \big| \widehat\Uc_{i}^{(1)}(X_{t_{i}}) - \widehat\Vc_{t_i} \big|^2. \label{YUinter} 
\end{aligned}
$$

By plugging this last inequality into \eqref{interYUV}, we then get for $|\pi|$ small enough 

$$
\begin{aligned}
\E\big| Y_{t_i} - \widehat\Uc_{i}^{(1)}(X_{t_{i}}) \big|^2 & \leq \; (1 + C |\pi|) \E \big|Y_{t_{i+1}} - \widehat\Uc_{i+1}^{(1)}(X_{t_{i+1}}) \big|^2 + C |\pi|^2 \\ & \;\;\;\;\; + \; C \E \Big[\int_{t_i}^{t_{i+1}} \big| Z_t - \bar Z_{t_i}\big|^2 \diff t \Big] + C |\pi| \E \Big[ \int_{t_i}^{t_{i+1}} |f(t,\Xc_t,Y_t,Z_t)|^2 \diff t \Big] \\ & \;\;\;\;\; + \; C N \E \big| \widehat\Vc_{t_i} - \widehat\Uc_{i}^{(1)}(X_{t_{i}}) \big|^2. 
\end{aligned}
$$

From discrete Gronwall's lemma (or by induction), and recalling the terminal condition $Y_{t_N}$ $=$ $g(\Xc_T)$, $\widehat\Uc_{i}^{(1)}(X_{t_{N}})$ $=$ $g(X_T)$, the definition $\eps^Z(\pi)$ of the $L^2$-regularity of $Z$, and \eqref{integf2}, this yields 

$$
\begin{aligned}
\max_{i=0,\ldots,N-1} \E\big| Y_{t_i} - \widehat\Uc_{i}^{(1)}(X_{t_{i}}) \big|^2 & \leq \; C \E \big|g(\Xc_{T}) - g(X_T) \big|^2 + C |\pi| + C \eps^Z(\pi) \\ & \hspace{5mm} + \; C N \sum_{i=0}^{N-1} \E \big| \widehat\Vc_{t_i} - \widehat\Uc_{i}^{(1)}(X_{t_{i}}) \big|^2. \label{estimYinter} 
\end{aligned}
$$

\vspace{1mm} \noindent {\it Step 3.}

Fix $i$ $\in$ $\{0,\ldots,N-1\}$.

By using relation \eqref{FBSDE} in the expression of the expected quadratic loss function in \eqref{eq:scheme1}, and recalling the definition of $\overline{{\widehat Z_{t_i}}}$ as a $L^2$-projection of $\widehat Z_t$, we have for all parameters $\theta$ $=$ $(\xi,\eta)$ of the neural networks $\Uc_i(.;\xi)$ and $\Zc_i(.;\eta)$ 

$$
\begin{aligned}
\label{LtildeL} \hat L_i^{(1)}(\theta) &= \; \tilde L_i(\theta) + \E \Big[ \int_{t_i}^{t_{i+1}} \big| \widehat Z_t - \overline{{\widehat Z_{t_i}}} \big|^2 \diff t \Big] 
\end{aligned}
$$

with 

$$
\begin{aligned}
\tilde L_i(\theta) & := \; \E \Big| \widehat\Vc_{t_i} - \Uc_i(X_{t_i};\xi) + \big( f(t_i,X_{t_i},\Uc_i(X_{t_i};\xi),\Zc_i(X_{t_i};\eta)) - f(t_i,X_{t_i},\widehat\Vc_{t_i},\overline{{\widehat Z_{t_i}}}) \big) \Delta t_i \Big|^2 \\ & \hspace{.6cm} + \; \Delta t_i \E \big| \overline{{\widehat Z_{t_i}}} - \Zc_i(X_{t_i};\eta) \big|^2. 
\end{aligned}
$$

By using Young inequality: $(a+b)^2$ $\leq$ $(1+\gamma \Delta t_i)a^2$ $+$ $(1+\frac{1}{\gamma \Delta t_i})b^2$, together with the Lipschitz condition on $f$ in {\bf (H1)}, we clearly see that 

$$
\begin{aligned}
\label{L<} \tilde L_i(\theta) & \leq \; (1 + C \Delta t_i) \E \big| \widehat\Vc_{t_i} - \Uc_i(X_{t_i};\xi) \big|^2 + C \Delta t_i \E \big| \overline{{\widehat Z_{t_i}}} - \Zc_i(X_{t_i};\eta) \big|^2. 
\end{aligned}
$$

On the other hand, using Young inequality in the form: $(a+b)^2$ $\geq$ $(1- \gamma \Delta t_i)a^2$ $+$ $(1-\frac{1}{\gamma \Delta t_i})b^2$ $\geq$ $(1- \gamma \Delta t_i)a^2$ $-$ $\frac{1}{\gamma \Delta t_i}b^2$, together with the Lipschitz condition on $f$, we have 

$$
\begin{aligned}
\tilde L_i(\theta) & \geq \; (1 - \gamma \Delta t_i) \E \big| \widehat\Vc_{t_i} - \Uc_i(X_{t_i};\xi) \big|^2 - \frac{2 \Delta t_i [f]^2_{_L}}{\gamma} \Big( \E \big| \widehat\Vc_{t_i} - \Uc_i(X_{t_i};\xi) \big|^2 + \E \big| \overline{{\widehat Z_{t_i}}} - \Zc_i(X_{t_i};\eta) \big|^2 \Big) \nonumber \\ & \hspace{.6cm} + \; \Delta t_i \E \big| \overline{{\widehat Z_{t_i}}} - \Zc_i(X_{t_i};\eta) \big|^2. 
\end{aligned}
$$

By choosing $\gamma$ $=$ $4[f]_{_L}^2$, this yields 

$$
\begin{aligned}
\label{L>} \tilde L_i(\theta) & \geq \; (1 - C \Delta t_i) \E \big| \widehat\Vc_{t_i} - \Uc_i(X_{t_i};\xi) \big|^2 + \frac{\Delta t_i}{2} \E \big| \overline{{\widehat Z_{t_i}}} - \Zc_i(X_{t_i};\eta) \big|^2. 
\end{aligned}
$$

\vspace{1mm} \noindent {\it Step 4.}

Fix $i$ $\in$ $\{0,\ldots,N-1\}$, and take $\theta_i^*$ $=$ $(\xi_i^*,\eta_i^*)$ $\in$ ${\rm arg}\min_\theta \hat L_i^{(1)}(\theta)$ so that $\widehat\Uc_i^{(1)}$ $=$ $\Uc_i(.;\xi_i^*)$, and $\widehat\Zc_i^{(1)}$ $=$ $\Zc_i(.;\eta_i^*)$.

By \eqref{LtildeL}, notice that $\theta_i^*$ $\in$ ${\rm arg}\min_\theta \tilde L_i(\theta)$.

From \eqref{L>} and \eqref{L<}, we then have for all $\theta$ $=$ $(\xi,\eta)$ 

$$
\begin{aligned}
(1 - C \Delta t_i) \E \big| \widehat\Vc_{t_i} - \widehat\Uc_i^{(1)}(X_{t_i}) \big|^2 + \frac{\Delta t_i}{2} \E \big| \overline{{\widehat Z_{t_i}}} - \widehat\Zc_i^{(1)}(X_{t_i}) \big|^2 \\ & \hspace{-8cm} \leq \tilde L_i(\theta_i^*) \; \leq \tilde L_i(\theta) \; \leq \; (1 + C \Delta t_i) \E \big| \widehat\Vc_{t_i} - \Uc_i(X_{t_i};\xi) \big|^2 + C \Delta t_i \E \big| \overline{{\widehat Z_{t_i}}} - \Zc_i(X_{t_i};\eta) \big|^2. \nonumber 
\end{aligned}
$$

For $|\pi|$ small enough, and recalling \eqref{defhatv1}, this implies 

$$
\begin{aligned}
\label{estimVU} \E \big| \widehat\Vc_{t_i} - \widehat\Uc_i^{(1)}(X_{t_i}) \big|^2 + \Delta t_i \E \big| \overline{{\widehat Z_{t_i}}} - \widehat\Zc_i^{(1)}(X_{t_i}) \big|^2 & \leq \; C \eps_i^{\Nc,v} + C \Delta t_i \eps_i^{\Nc,z}. 
\end{aligned}
$$

Plugging this last inequality into \eqref{estimYinter}, we obtain 

$$
\begin{aligned}
\max_{i=0,\ldots,N-1} \E\big| Y_{t_i} - \widehat\Uc_{i}^{(1)}(X_{t_{i}}) \big|^2 & \leq \; C \E \big|g(\Xc_{T}) - g(X_T) \big|^2 + C |\pi| + C \eps^Z(\pi) \\ & \hspace{6mm} + \; C \sum_{i=0}^{N-1} \big( N \eps_i^{\Nc,v} + \eps_i^{\Nc,z} \big), \label{estimYfin} 
\end{aligned}
$$

which proves the consistency of the $Y$-component in \eqref{eq:theo1_scheme1}. \vspace{1mm} \noindent {\it Step 5.}

Let us finally prove the consistency of the $Z$-component.

From \eqref{Pythagore} and \eqref{inegZi}, we have for any $i$ $=$ $0,\ldots,N-1$: 

$$
\begin{aligned}
\E\Big[ \int_{t_i}^{t_{i+1}} \big| Z_t - \overline{{\widehat Z_{t_i}}} \big|^2 \diff t \Big] & \leq \; \E\Big[ \int_{t_i}^{t_{i+1}} \big| Z_t - \bar Z_{t_i} \big|^2 \diff t \Big] + 2d |\pi| \E \Big[ \int_{t_i}^{t_{i+1}} |f(t,\Xc_t,Y_t,Z_t)|^2 \diff t \Big] \\ & \hspace{.6cm} + \; 2d \Big( \E \big|Y_{t_{i+1}} - \widehat\Uc_{i+1}^{(1)}(X_{t_{i+1}}) \big|^2 - \E \Big| \E_i\big[ Y_{t_{i+1}} - \widehat\Uc_{i+1}^{(1)}(X_{t_{i+1}})\big] \Big|^2 \Big) 
\end{aligned}
$$

By summing over $i$ $=$ $0,\ldots,N-1$, we get (recall \eqref{integf2}) 

$$
\begin{aligned}
\E \Big[ \sum_{i=0}^{N-1} \int_{t_i}^{t_{i+1}} \big| Z_t - \overline{{\widehat Z_{t_i}}} \big|^2 \diff t \Big] & \leq \; \eps^Z(\pi) + C |\pi| + 2d \E \big|g(\Xc_{T}) - g(X_T) \big|^2 \label{Zinter} \\ & \; + \: 2d \sum_{i=0}^{N-1} \Big( \E \big|Y_{t_{i}} - \widehat\Uc_{i}^{(1)}(X_{t_{i}}) \big|^2 - \E \Big| \E_i\big[ Y_{t_{i+1}} - \widehat\Uc_{i+1}^{(1)}(X_{t_{i+1}})\big] \Big|^2 \Big) \nonumber 
\end{aligned}
$$

where we change the indices in the last summation.

Now, from \eqref{inegYVi}, \eqref{YUinter}, we have 

$$
\begin{aligned}
2d \Big( \E \big|Y_{t_{i}} - \widehat\Uc_{i}^{(1)}(X_{t_{i}}) \big|^2 - \E \Big| \E_i\big[ Y_{t_{i+1}} - \widehat\Uc_{i+1}^{(1)}(X_{t_{i+1}})\big] \Big|^2 \Big) \\ & \hspace{-7cm}\leq \Big(\frac{1 +\gamma |\pi|}{1 - |\pi|} - 1 \Big)\E \Big| \E_i\big[ Y_{t_{i+1}} - \hat\Uc_{i+1}^{(1)}(X_{t_{i+1}}) \big] \Big|^2 \\ & \hspace{-6.4cm}+ \; \frac{8d[f]^2_{_L}}{\gamma} \frac{1+ \gamma |\pi|}{1 - |\pi|} \Big\{ C |\pi|^2 + |\pi| \E\big| Y_{t_i} - \widehat\Vc_{t_i} \big|^2 + \E\Big[ \int_{t_i}^{t_{i+1}} \big| Z_t - \overline{{\widehat Z_{t_i}}} \big|^2 \diff t \Big] \Big\} \\ & \hspace{-6.4cm}+ \; \frac{2d}{|\pi|(1-|\pi|)} \E \big| \widehat\Uc_{i}^{(1)}(X_{t_{i}}) - \widehat\Vc_{t_i} \big|^2. 
\end{aligned}
$$

We now choose $\gamma$ $=$ $24d[f]^2_{_L}$ so that $\frac{8d[f]^2_{_L}}{\gamma} (1+ \gamma |\pi|)/(1 - |\pi|)$ $\leq$ $1/2$ for $|\pi|$ small enough, and by plugging into \eqref{Zinter}, we obtain (note also that $\big[(1 +\gamma |\pi|)/(1 - |\pi|) - 1 \big]$ $=$ $O(|\pi|)$): 

$$
\begin{aligned}
\frac{1}{2} \E \Big[ \sum_{i=0}^{N-1} \int_{t_i}^{t_{i+1}} \big| Z_t - \overline{{\widehat Z_{t_i}}} \big|^2 \diff t \Big] & \leq \; \eps^Z(\pi) + C |\pi| + C \max_{i=0,\ldots,N} \E\big| Y_{t_i} - \widehat\Uc_{i}^{(1)}(X_{t_{i}}) \big|^2 \\ & \hspace{.6cm}+ \; \frac{1}{2} |\pi| \sum_{i=0}^{N-1} \E\big| Y_{t_i} - \widehat\Vc_{t_i} \big|^2 + CN \sum_{i=0}^{N-1} \E \big| \widehat\Uc_{i}^{(1)}(X_{t_{i}}) - \widehat\Vc_{t_i} \big|^2 \\ & \leq \; C \eps^Z(\pi) + C |\pi| + C \max_{i=0,\ldots,N} \E\big| Y_{t_i} - \widehat\Uc_{i}^{(1)}(X_{t_{i}}) \big|^2 \\ & \hspace{.6cm} + \; CN \sum_{i=0}^{N-1} \E \big| \widehat\Uc_{i}^{(1)}(X_{t_{i}}) - \widehat\Vc_{t_i} \big|^2 \\ & \leq \; C \E \big|g(\Xc_{T}) - g(X_T) \big|^2 + C |\pi| + C \eps^Z(\pi) \\ & \hspace{6mm} + \; C \sum_{i=0}^{N-1} \big( N \eps_i^{\Nc,v} + \eps_i^{\Nc,z} \big), \label{estimZhatZ} 
\end{aligned}
$$

where we used \eqref{interYUV} and \eqref{integf2} in the second inequality, and \eqref{estimVU} and \eqref{estimYfin} in the last inequality.

By writing that 

$$
\begin{aligned}
\E\Big[ \int_{t_i}^{t_{i+1}} \big| Z_t - \widehat\Zc_i^{(1)}(X_{t_i}) \big|^2 \diff t \Big] & \leq \; 2 \E\Big[ \int_{t_i}^{t_{i+1}} \big| Z_t - \overline{{\widehat Z_{t_i}}} \big|^2 \diff t \Big] + 2 \Delta t_i \E \big| \overline{{\widehat Z_{t_i}}} - \widehat\Zc_i^{(1)}(X_{t_i}) \big|^2, 
\end{aligned}
$$

and using \eqref{estimVU}, \eqref{estimZhatZ}, we obtain after summation over $i$ $=$ $0,\ldots,N-1$, the required error estimate for the $Z$-component as in \eqref{estimYfin}, and this ends the proof. \ep 

### Convergence of DBDP2

We shall consider neural networks with one hidden layer, $m$ neurons with total variation smaller than $\gamma_m$ (see Section [secNN](#secNN)), a $C^3$ activation function $\varrho$ with linear growth condition, and bounded derivatives, e.g., a sigmoid activation function, or a $\tanh$ function: this class of neural networks is then represented by the parametric set of functions 

$$
\begin{aligned}
\Nc\Nc_{d,1,2,m}^\varrho(\Theta_m^\gamma) &:= \left\{ x \in \R^d \mapsto \mathcal{U}(x;\theta) = \sum_{i=1}^{m} c_i \varrho(a_i.x+b_i) + b_0, \; \theta=(a_i,b_i,c_i,b_0)_{i=1}^m \; \in \; \Theta^\gamma_m \right\}, \nonumber 
\end{aligned}
$$

with 

$$
\begin{aligned}
\Theta^\gamma_m &:= \left\{ \theta=(a_i,b_i,c_i,b_0)_{i=1}^m: \; \max_{i=1,\ldots,m} |a_i| \leq \gamma_m, \; \sum_{i=1}^{m} |c_i| \leq \gamma_m \right\}, 
\end{aligned}
$$

for some sequence $(\gamma_m)_m$ converging to $\infty$, as $m$ goes to infinity, and such that 

$$
\label{convgamma} 
\begin{array}{cc} \frac{\gamma_{m}^6}{N} \xrightarrow[m,N \to \infty]{} 0. \end{array}

$$

Notice that the neural networks in $\Nc\Nc_{d,1,2,m}^\varrho(\Theta_m^\gamma)$ have their first, second and third derivatives uniformly bounded w.r.t. the state variable $x$.

More precisely, there exists some constant $C$ depending only on $d$ and the derivatives of $\varrho$ s.t. for any $\Uc$ $\in$ $\Nc\Nc_{d,1,2,m}^\varrho(\Theta_m^\gamma)$, 

$$
\label{bounderivU} \left\{ 

$$
\begin{aligned}
\sup_{x\in\R^d,\theta\in\Theta_m^\gamma} \Big| D_x \mathcal{U}(x;\theta) \Big| \; \leq \; C \gamma_{m}^2, \quad \sup_{x\in\R^d,\theta\in\Theta_m^\gamma} \Big| D_x^2 \mathcal{U}(x;\theta) \Big| \; \leq \; C \gamma_{m}^3, \\ \text{ and } \;\;\; \sup_{x\in\R^d,\theta\in\Theta_m^\gamma} \Big| D_x^3 \mathcal{U}(x;\theta) \Big| \; \leq \; C \gamma_{m}^4. 
\end{aligned}
$$

\right. 
$$

\vspace{1mm}

Let us investigate the convergence of the scheme DBDP2 in \eqref{eq:scheme2} with neural networks in $\Nc\Nc_{d,1,2,m}^\varrho(\Theta_m^\gamma)$, and define for $i$ $=$ $0,\ldots,N-1$: 

$$
\label{defVCZ2} \left\{ 
\begin{array}{rcl} \widehat\Vc_{t_i} & := & \E_i \big[ \widehat\Uc_{i+1}^{(2)}(X_{t_{i+1}}) \big] + f(t_i,X_{t_i},\widehat\Vc_{t_i},\overline{{\widehat Z_{t_i}}}) \Delta t_i \; = \; \hat v_i^{}(X_{t_i}), \\ \overline{{\widehat Z_{t_i}}} & := & \frac{1}{\Delta t_i} \E_i\left[ \widehat\Uc_{i+1}^{(2)}(X_{t_{i+1}}) \Delta W_{t_i} \right] \; = \; \overline{{\hat z_i}^{}}(X_{t_i}). \end{array}
\right. 
$$

\vspace{1mm}

A measure of the (squared) error for the DBDP2 scheme is defined similarly as in DBDP1 scheme: 

$$
\begin{aligned}
\Ec\big[(\widehat\Uc^{(2)},\widehat\Zc^{(2)}),(Y,Z)\big] & := \; \max_{i=0,\ldots,N-1} \E \big|Y_{t_i}- \widehat\Uc_i^{(2)}(X_{t_i})\big|^2 + \E \bigg[ \sum_{i=0}^{N-1} \int_{t_i}^{t_{i+1}} \big| Z_t - \widehat\Zc_i^{(2)}(X_{t_i}) \big|^2 dt \bigg]. 
\end{aligned}
$$

Our second main result gives an error estimate of the DBDP2 scheme in terms of the $L^2$-approximation errors of $\hat v_i$ and its derivative (which exists under assumption detailed below) by neural networks $\Uc_i$ $\in$ $\Nc\Nc_{d,1,2,m}^\varrho(\Theta_m^\gamma)$, $i=0,\ldots,N-1$, and defined as 

$$
\begin{aligned}
\eps_i^{\Nc,m} \; := \; \inf_{\theta \in \Theta_m^\gamma}\Big\{ \E \big|\hat v_i(X_{t_i}) - \Uc_i(X_{t_i};\theta) \big|^2 + \Delta t_i \E \big| \sigma\trans(t_i,X_{t_i}) \big( D_x \hat v_i(X_{t_i}) - D_x \Uc_i(X_{t_i};\theta) \big) \big|^2 \Big\}, \nonumber 
\end{aligned}
$$

which are expected to be small in view of the universal approximation theorem (II), see discussion in Remark [remNN_approxError](#remNN_approxError). \vspace{1mm}

We also require the additional conditions on the coefficients: \vspace{2mm} \noindent {\bf (H2)} (i) The functions $x$ $\mapsto$ $\mu(t,.)$, $\sigma(t,.)$ are $C^1$ with bounded derivatives uniformly w.r.t. $(t,x)$ $\in$ $[0,T]\times\R^d$. \vspace{1mm} \noindent (ii) The function $(x,y,z)$ $\mapsto$ $f(t,.)$ is $C^1$ with bounded derivatives uniformly w.r.t. $(t,x,y,z)$ in $[0,T]\times \R^d\times\R\times\R^d$. \vspace{2mm} 
\begin{Theorem} \label{theoconv2} \emph{(Consistency of DBDP2)} \label{theo:scheme2}

Under {\bf (H1)}-{\bf (H2)}, there exists a constant $C>0$, independent of $\pi$, such that 

$$
\begin{aligned}
\Ec\big[(\widehat\Uc^{(2)},\widehat\Zc^{(2)}),(Y,Z)\big] & \leq \; C \Big( \E \big|g(\Xc_{T}) - g(X_T) \big|^2 + \frac{\gamma_{m}^6}{N} + \eps^Z(\pi) + N \sum_{i=0}^{N-1} \eps_i^{\Nc,m} \Big). \label{eq:theo2_scheme2} 
\end{aligned}
$$

\end{Theorem}
\noindent {\bf Proof.}

For simplicity of notations, we assume $d$ $=$ $1$, and only detail the arguments that differ from the proof of Theorem [eq:theo1_scheme1](#eq:theo1_scheme1).

From \eqref{defVCZ2}, and the Euler scheme \eqref{eq:eulerSDE}, we have 

$$
\begin{aligned}
\hat v_i(x) & = \; \tilde v_i(x) + \Delta t_i f(t_i,x,\hat v_i(x), \overline{{\hat z_i}}(x)), \;\;\; \tilde v_i(x) \; := \; \E\big[ \hat u_{i+1}(X_{t_{i+1}}^x) \big], \;\; x \in \R^d, \\ \overline{{\hat z_i}}(x) & = \; \frac{1}{\Delta t_i} \E \big[ \hat u_{i+1}(X_{t_{i+1}}^x) \Delta W_{t_i} \big], \;\;\; X_{t_{i+1}}^x \; = \; x + \mu(t_i,x) \Delta t_i + \sigma(t_i,x) \Delta W_{t_i}. 
\end{aligned}
$$

Under assumption {\bf (H2)}(i), and recalling that $\hat u_{i+1}$ $=$ $\Uc_{i+1}(.;\theta_{i+1}^*)$ is $C^2$ with bounded derivatives, we see that $\tilde v_i$ is $C^1$ with 

$$
\begin{aligned}

D_x \tilde v_i(x) & = \; \E \Big[ \big(1 + D_x \mu(t_i,x) \Delta t_i + D_x \sigma(t_i,x) \Delta W_{t_i} \big) D_x \hat u_{i+1}(X_{t_{i+1}}^x) \Big] \\ & = \; \E \big[ D_x \hat u_{i+1}(X_{t_{i+1}}^x) \big] + \Delta t_i \; R_i(x) \label{derivtildev} \\ R_i(x) & := \; D_x \mu(t_i,x) \E \big[ D_x \hat u_{i+1}(X_{t_{i+1}}^x) \big] + \sigma(t_i,x) D_x \sigma(t_i,x) \E \big[ D_x^2 \hat u_{i+1}(X_{t_{i+1}}^x) \big], 
\end{aligned}
$$

where we use integration by parts in the second equality.

Similarly, we have 

$$
\left\{ 

$$
\begin{aligned}
\overline{{\hat z_i}}(x) & = \; \sigma(t_i,x)\E\big[ D_x \hat u_{i+1}(X_{t_{i+1}}^x) \big], \label{IPPZ} \\ D_x \overline{{\hat z_i}}(x) & = \; D_x \sigma(t_i,x)\E\big[ D_x \hat u_{i+1}(X_{t_{i+1}}^x) \big] + \sigma(t_i,x)\E\big[ D_x^2 \hat u_{i+1}(X_{t_{i+1}}^x) \big] + \Delta t_i \; \sigma(t_i,x) G_i(x) \\ G_i(x) & := \; D_x \mu(t_i,x) \E \big[ D_x^2 \hat u_{i+1}(X_{t_{i+1}}^x) \big] + \sigma(t_i,x) D_x \sigma(t_i,x) \E \big[ D_x^3 \hat u_{i+1}(X_{t_{i+1}}^x) \big]. 
\end{aligned}
$$

\right. 
$$

Denoting by $\hat f_i(x)$ $=$ $f(t_i,x,\hat v_i(x), \overline{{\hat z_i}}(x))$, it follows by the implicit function theorem, and for $|\pi|$ small enough, that $\hat v_i$ is $C^1$ with derivative given by 

$$
\begin{aligned}

D_x \hat v_i(x) & = \; D_x \tilde v_i(x) + \Delta t_i \Big( D_x \hat f_i(x) + D_y \hat f_i(x) D_x \hat v_i(x) + D_z \hat f_i(x) D_x \overline{{\hat z_i}}(x) \Big) 
\end{aligned}
$$

and thus by \eqref{derivtildev}-\eqref{IPPZ} 

$$
\begin{aligned}
\big( 1 - \Delta t_i D_y \hat f_i(x) \big) \sigma(t_i,x) D_x \hat v_i(x) & = \; \overline{{\hat z_i}}(x) + \Delta t_i \sigma(t_i,x) \Big( R_i(x) + D_x \hat f_i(x) + D_z \hat f_i(x) D_x \overline{{\hat z_i}}(x) \Big). 
\end{aligned}
$$

Under {\bf (H2)}, by the linear growth condition on $\sigma$, and using the bounds on the derivatives of the neural networks in $\Nc\Nc_{d,1,2,m}^\varrho(\Theta_m^\gamma)$ in \eqref{bounderivU}, we then have 

$$
\begin{aligned}
\label{interzv} \E \Big| \sigma(t_i,X_{t_i}) D_x \hat v_i(X_{t_i}) - \overline{{\widehat Z_{t_i}}} \Big|^2 & \leq \; C( \gamma_{m}^6 + |\pi|^2 \gamma_{m}^8) |\pi|^2. 
\end{aligned}
$$

Next, by the same arguments as in Steps 3 and 4 in the proof of Theorem [theo:scheme1_1](#theo:scheme1_1) (see in particular \eqref{estimVU}), we have for $|\pi|$ small enough, 

$$
\begin{aligned}
\E \big| \widehat\Vc_{t_i} - \widehat\Uc_i^{(2)}(X_{t_i}) \big|^2 + \Delta t_i \E \big| \overline{{\widehat Z_{t_i}}} - \widehat\Zc_i^{(2)}(X_{t_i}) \big|^2 \\ & \hspace{-5cm}\leq C \E\big[ \big|\hat v_i(X_{t_i}) - \Uc_i(X_{t_i};\theta) \big|^2 \big] + C \Delta t_i \E \big| \overline{{\widehat Z_{t_i}}} - \sigma(t_i,X_{t_i}) \hat D_x \Uc_i(X_{t_i};\theta) \big|^2, 
\end{aligned}
$$

for all $\theta$ $\in$ $\Theta^N$, and then with \eqref{interzv}, and by definition of $\eps_i^{NN,v,2}$: 

$$
\begin{aligned}
\label{estimU2} \E \big| \widehat\Vc_{t_i} - \widehat\Uc_i^{(2)}(X_{t_i}) \big|^2 + \Delta t_i \E \big| \overline{{\widehat Z_{t_i}}} - \widehat\Zc_i^{(2)}(X_{t_i}) \big|^2 & \leq \; C \eps_i^{NN,v,2} + C ( \gamma_{m}^6 + |\pi|^2 \gamma_{m}^8) |\pi|^3. 
\end{aligned}
$$

On the other hand, by the same arguments as in Steps 1 and 2 in the proof of Theorem [theo:scheme1_1](#theo:scheme1_1) (see in particular \eqref{estimYinter}), we have 

$$
\begin{aligned}
\max_{i=0,\ldots,N-1} \E\big| Y_{t_i} - \widehat\Uc_{i}^{(2)}(X_{t_{i}}) \big|^2 & \leq \; C \E \big|g(\Xc_{T}) - g(X_T) \big|^2 + C |\pi| + C \eps^Z(\pi) \\ & \hspace{6mm} + \; C N \sum_{i=0}^{N-1} \E \big| \widehat\Vc_{t_i} - \widehat\Uc_{i}^{(2)}(X_{t_{i}}) \big|^2. 
\end{aligned}
$$

Plugging \eqref{estimU2} into this last inequality, together with \eqref{convgamma}, gives the required estimation \eqref{eq:theo2_scheme2} for the $Y$-component.

Finally, by following the same arguments as in Step 5 in the proof of \eqref{theo:scheme1_1}, we obtain the estimation \eqref{eq:theo2_scheme2} for the $Z$-component. \ep \vspace{3mm} 
\begin{Remark} \label{remNN_approxError} {\rm The universal approximation theorem (II) [^Hor90universal] is valid on compact sets, and one cannot conclude {\it a priori} that the error of network approximation $\eps_i^{\Nc,m}$ converge to zero as $m$ goes to infinity.

Instead, we have to proceed into two steps: 

- [(i)] Localize the error by considering 

$$
\begin{aligned}
\eps_i^{\Nc,m,K} & := \; \inf_{\theta\in\Theta_m^{\gamma}} \E \big[ \Delta_i(X_{t_i};\theta) 1_{|X_{t_i}| \leq K} \big], 
\end{aligned}
$$

where we set $\Delta_i(x;\theta)$ $:=$ $|\hat v_i(x)-\Uc_i(x;\theta)|^2$ + $\Delta t_i \big|\sigma\trans(t_i,x) \big( D_x \hat v_i(x) - D_x \Uc_i(x;\theta) \big) \big|^2$. 
- [(ii)] Consider an increasing family of neural networks $\Theta_m^{\gamma^{N-1}}$ $\subset$ $\ldots$ $\subset$ $\Theta_m^{\gamma^{i}}$ $\subset$ $\ldots$ $\subset$ $\Theta_m^{\gamma^0}$ on which to minimize the approximation errors by backward induction at times $t_i$, $i$ $=$ $N-1,\ldots,0$, and where, $\gamma_m^i$ is defined by $$ \gamma^i_m:=\gamma_{\varphi^{N-1-i}(m)}, $$ with $\varphi: \mathbb{N} \to \mathbb{N}$ an increasing function, and where we use the notation $\varphi^k:=\varphi \circ ... \circ \varphi$ (composition $k$ times).

The localized approximation error at time $t_i$, for $0\leq i \leq N-1$, should then be rewritten as 

$$
\begin{aligned}
\eps_{i,N}^{\Nc,m,K} & := \; \inf_{\theta\in\Theta_m^{\gamma^i}} \E \big[ \Delta_i(X_{t_i};\theta) 1_{|X_{t_i}| \leq K} \big], \nonumber 
\end{aligned}
$$

and the non-localized one as 

$$
\begin{aligned}
\eps_{i,N}^{\Nc,m} & := \; \inf_{\theta\in\Theta_m^{\gamma^i}} \E \big[ \Delta_i(X_{t_i};\theta)\big]. \nonumber 
\end{aligned}
$$

Note that $\eps_{i,N}^{\Nc,m,K}$ converges to zero, as $m$ goes to infinity, for any $K$ $>$ $0$, as claimed by the universal approximation theorem (II) [^Hor90universal].

On the other hand, from the expressions of $\hat v_i$, $D_x\hat v_i$ in the above proof of Theorem [theoconv2](#theoconv2), we see under {\bf (H1)}-{\bf (H2)}, and from \eqref{bounderivU} that for all $x$ $\in$ $\R^d$, $\theta$ $\in$ $\Theta_m^{\gamma^i}$, $i$ $=$ $0,\ldots,N-1$: 

$$
\begin{aligned}
|\Delta_i(x;\theta)| & \leq \; C(1 +|x|^2)\gamma_{\varphi^{N-1}(m)}^4, 
\end{aligned}
$$

for some positive constant $C$ independent of $m,\pi$.

We deduce by Cauchy-Schwarz and Chebyshev's inequalities that for all $K$ $>$ $0$, and $\theta$ $\in$ $\Theta_m^{\gamma^i}$, $i$ $=$ $0,\ldots,N-1$, 

$$
\begin{aligned}
\label{deltainter} \E \big[ \Delta_i(X_{t_i};\theta) 1_{|X_{t_i}| > K} \big] & \leq \; \Big\|\Delta_i(X_{t_i};\theta) \Big\|_{_2} \frac{\big\| X_{t_i} \big\|_{_2}}{K} \; \leq C(1 +|x_0|^3)\frac{\gamma_{\varphi^{N-1}(m)}^4}{K}, 
\end{aligned}
$$

where we used \eqref{integX} in the last inequality.

This shows that 

$$
\begin{aligned}
\eps_{i,N}^{\Nc,m} & \leq \; \eps_{i,N}^{\Nc,m,K} + C \frac{\gamma_{\varphi^{N-1}(m)}^4}{K}, \;\;\; \forall K > 0, 
\end{aligned}
$$

and thus, in theory, the error $\eps_{i,N}^{\Nc,m}$ can be made arbitrary small by suitable choices of large $m$ and $K$. \ep } \end{Remark}

### Convergence of RDBDP

In this paragraph, we study the convergence of machine learning schemes for the variational inequality \eqref{eq:IQV}.

We first consider the case when $f$ does not depend on $z$, so that the component $Y_t$ $=$ $u(t,\Xc_t)$ solution to the reflected BSDE \eqref{RBSDE} admits a Snell envelope representation, and we shall focus on the error on $Y$ by proposing an alternative to scheme \eqref{eq:schemeVI}, refereed to as RDBDPbis scheme, which only uses neural network for learning the function $u$: 

-  Initialize $\widehat\Uc_N$ $=$ $g$ 
-  For $i$ $=$ $N-1,\ldots,0$, given $\widehat\Uc_{i+1}$, use a deep neural network $\Uc_i(.;\theta)$ $\in$ $\Nc\Nc_{d,1,L,m}^\varrho(\R^{N_m})$, and compute (by SGD) the minimizer of the expected quadratic loss function 

$$
\label{eq:schemeVIbis} \left\{ 

$$
\begin{aligned}
\bar L_i(\theta) & := \E \big| \widehat\Uc_{i+1}(X_{t_{i+1}}) - \Uc_i(X_{t_i};\theta) + f(t_i,X_{t_i},\Uc_i(X_{t_i};\theta)) \Delta t_i \big|^2 \\ \theta_i^* & \in {\rm arg}\min_{\theta\in\R^{N_m}} \bar L_i(\theta). 
\end{aligned}
$$

\right. 
$$

Then, update: $\widehat\Uc_i$ $=$ $\max\big[\Uc_i(.;\theta_i^*),g]$. 

Let us also define from the scheme \eqref{eq:schemeVIbis} 

$$
\label{defV3} \left\{ 

$$
\begin{aligned}
\tilde\Vc_{t_i} & := \; \E_i \big[ \widehat\Uc_{i+1}^{}(X_{t_{i+1}}) \big] + f(t_i,X_{t_i},\tilde\Vc_{t_i}) \Delta t_i \; = \; \tilde v_i(X_{t_i}), \\ \widehat\Vc_{t_i} & := \; \max[\tilde\Vc_{t_i} ; g(X_{t_i}) ], \;\;\; i =0,\ldots,N-1. 
\end{aligned}
$$

\right. 
$$

Our next result gives an error estimate of the scheme \eqref{eq:schemeVIbis} in terms of the $L^2$-approximation errors of $\tilde v_i$ by neural networks $\Uc_i$, $i=0,\ldots,N-1$, and defined as 

$$
\begin{aligned}
\tilde\eps_i^{\Nc} & := \; \inf_{\theta\in\R^{N_m}} \E \big|\tilde v_i(X_{t_i}) - \Uc_i(X_{t_i};\theta) \big|^2. 
\end{aligned}
$$

\begin{Theorem} \emph{(Case $f$ independent of $z$: Consistency of RDBDPbis)} \label{theo:scheme3bis}

Let Assumption {\bf (H1)} hold, with $g$ Lipschitz.

Then, there exists a constant $C>0$, independent of $\pi$, such that 

$$
\begin{aligned}
\max_{i=0,\ldots,N-1} \big\| Y_{t_i} - \hat\Uc_i(X_{t_i}) \big\|_{_2} & \leq \; C \Big( |\pi|^{\frac{1}{2}} + \sum_{i=0}^{N-1} \sqrt{\tilde \eps_i^{\Nc}} \Big), \label{eq:theo3bis} 
\end{aligned}
$$

where $\|.\|_{_2}$ is the $L^2$-norm on $(\Omega,\Fc,\P)$. \end{Theorem}

\begin{Remark} \label{rem:estimRDBDP} {\rm The estimation \eqref{eq:theo3bis} implies that 

$$
\begin{aligned}
\max_{i=0,\ldots,N-1} \E\big| Y_{t_i} - \hat\Uc_i(X_{t_i}) \big|^2 & \leq \; C \Big( |\pi| + N \sum_{i=0}^{N-1} \tilde \eps_i^{\Nc} \Big), \label{eq:theo3_scheme3} 
\end{aligned}
$$

which is of the same order than the error estimate in Theorem [theo:scheme1_1](#theo:scheme1_1) when $g$ is Lipschitz. } \ep \end{Remark}
\noindent {\bf Proof.}

Let us introduce the discrete-time approximation of the reflected BSDE 

$$
\label{defYpi} \left\{ 

$$
\begin{aligned}

Y_{t_N}^\pi & = \; g(X_{t_N}) \\ \tilde Y_{t_i}^\pi & = \; \E_i[ Y_{t_{i+1}}^\pi ] + f(t_i,X_{t_i},\tilde Y_{t_i}^\pi) \Delta t_i \\ Y_{t_i}^\pi & = \; \max \big[ \tilde Y_{t_i}^\pi ; g(X_{t_i}) \big] , \;\;\; i =0,\ldots,N-1. 
\end{aligned}
$$

\right. 
$$

It is known, see [^Balpag03], [^Bouchard2004discrete] that 

$$
\begin{aligned}
\label{estimRBSDE} \max_{i=0,\ldots,N-1} \big\| Y_{t_i} - Y_{t_i}^\pi \big\|_{_2} & = \; O(|\pi|^{\frac{1}{2}}). 
\end{aligned}
$$

Fix $i$ $=$ $0,\ldots,N-1$.

From \eqref{defV3}, \eqref{defYpi}, we have 

$$
\begin{aligned}
| \tilde Y_{t_i}^\pi - \tilde\Vc_{t_i} | & \leq \; \E_i \big| Y_{t_{i+1}}^\pi - \widehat\Uc_{i+1}^{}(X_{t_{i+1}}) \big| + \Delta t_i \big| f(t_i,X_{t_i},\tilde Y_{t_i}^\pi) - f(t_i,X_{t_i},\tilde\Vc_{t_i}) \big| \\ & \leq \; \E_i \big| Y_{t_{i+1}}^\pi - \widehat\Uc_{i+1}^{}(X_{t_{i+1}}) \big| + [f]_{_L} \Delta t_i | \tilde Y_{t_i}^\pi - \tilde\Vc_{t_i} |, 
\end{aligned}
$$

from the Lipschitz condition on $f$ in {\bf (H1)}, and then for $|\pi|$ small enough 

$$
\begin{aligned}
\big\| \tilde Y_{t_i}^\pi - \tilde\Vc_{t_i} \big\|_{_2} & \leq \; (1 + C |\pi|) \big\| Y_{t_{i+1}}^\pi - \widehat\Uc_{i+1}^{}(X_{t_{i+1}}) \big\|_{_2}. 
\end{aligned}
$$

By Minkowski inequality, this yields for all $\theta$ 

$$
\begin{aligned}
\label{YVxi} \big\| \tilde Y_{t_i}^\pi - \Uc_{i}^{}(X_{t_{i}};\theta) \big\|_{_2} & \leq \; (1 + C |\pi|) \big\| Y_{t_{i+1}}^\pi - \widehat\Uc_{i+1}^{}(X_{t_{i+1}}) \big\|_{_2} + \big\| \tilde\Vc_{t_i} - \Uc_{i}^{}(X_{t_{i}};\theta) \big\|_{_2}. 
\end{aligned}
$$

On the other hand, by the martingale representation theorem, there exists an $\R^d$-valued square integrable process $(\tilde Z_t)_t$ such that 

$$
\begin{aligned}
\label{FBSDE3} \widehat\Uc_{i+1}^{}(X_{t_{i+1}}) & = \; \tilde\Vc_{t_i} - f(t_i,X_{t_i},\tilde\Vc_{t_i}) \Delta t_i + \int_{t_i}^{t_{i+1}} \tilde Z_s\trans \diff W_s, 
\end{aligned}
$$

and the expected squared loss function of the DBDP3 scheme can be written as 

$$
\begin{aligned}
\label{LtildeL3} \bar L_i^{}(\theta) &= \; \tilde L_i(\theta) + \E \Big[ \int_{t_i}^{t_{i+1}} \big| \tilde Z_t \big|^2 \diff t \Big] 
\end{aligned}
$$

with 

$$
\begin{aligned}
\sqrt{ \tilde L_i(\theta) } & := \; \Big\| \tilde\Vc_{t_i} - \Uc_i(X_{t_i};\theta) + \big( f(t_i,X_{t_i},\Uc_i(X_{t_i};\theta)) - f(t_i,X_{t_i},\tilde\Vc_{t_i}) \big) \Delta t_i \Big\|_{_2}. 
\end{aligned}
$$

From the Lipschitz condition on $f$, and by Minkowski inequality, we have for all $\theta$ 
\begin{eqnarray*} (1 - [f]_{_L} \Delta t_i) \big\| \tilde\Vc_{t_i} - \Uc_i(X_{t_i};\theta) \big\|_{_2} & \leq & \sqrt{\tilde L_i(\theta)} \; \leq \; (1 + [f]_{_L} \Delta t_i) \big\| \tilde \Vc_{t_i} - \Uc_i(X_{t_i};\theta) \big\|_{_2}. \end{eqnarray*}

Take now $\theta_i^*$ $\in$ ${\rm arg}\min_\theta \bar L_i(\theta)$ $=$ ${\rm arg}\min_\theta \tilde L_i(\theta)$.

Then, from the above relations, we have 
\begin{eqnarray*} (1 - [f]_{_L} \Delta t_i) \big\| \tilde\Vc_{t_i} - \Uc_i(X_{t_i};\theta_i^*) \big\|_{_2} & \leq & (1 + [f]_{_L} \Delta t_i) \big\| \tilde \Vc_{t_i} - \Uc_i(X_{t_i};\theta) \big\|_{_2}, \end{eqnarray*}
for all $\theta$, and so 

$$
\begin{aligned}
\label{VUxi} \big\| \tilde\Vc_{t_i} - \Uc_i(X_{t_i};\xi_i^*) \big\|_{_2} & \leq \; (1 + C|\pi|) \sqrt{ \tilde\eps_i^{\Nc} }. 
\end{aligned}
$$

By taking $\theta$ $=$ $\theta_i^*$ in \eqref{YVxi}, recalling that $\widehat\Uc_i(X_{t_i})$ $=$ $\max[\Uc_i(X_{t_i};\theta_i^*); g(X_{t_i})]$, $Y_{t_i}^\pi$ $=$ $\max[\tilde Y_{t_i}^\pi;g(X_{t_i})]$, and since $|\max(a,c) - \max(b,c)|$ $\leq$ $|a-b|$, we obtain by using \eqref{VUxi} 

$$
\begin{aligned}
\big\| Y_{t_i}^\pi - \widehat \Uc_{i}^{}(X_{t_{i}}) \big\|_{_2} & \leq \; (1 + C |\pi|) \Big( \big\| Y_{t_{i+1}}^\pi - \widehat\Uc_{i+1}^{}(X_{t_{i+1}}) \big\|_{_2} + \sqrt{ \tilde\eps_i^{\Nc} } \Big). 
\end{aligned}
$$

By induction, this yields 

$$
\begin{aligned}
\max_{i=0,\ldots,N-1} \big\| Y_{t_i}^\pi - \widehat \Uc_{i}^{}(X_{t_{i}}) \big\|_{_2} & \leq \; C \sum_{i=0}^{N-1} \sqrt{ \tilde\eps_i^{\Nc} }, 
\end{aligned}
$$

and we conclude with \eqref{estimRBSDE}. \ep \vspace{5mm}

We finally turn to the general case when $f$ may depend on $z$, and study the convergence of the RDBDP scheme \eqref{eq:schemeVI} towards the variational inequality \eqref{eq:IQV} related to the solution $(Y,Z)$ of the reflected BSDE \eqref{RBSDE} by showing an error estimate for 

$$
\begin{aligned}
\Ec\big[(\widehat\Uc^{},\widehat\Zc^{}),(Y,Z)\big] & := \; \max_{i=0,\ldots,N-1} \E \big|Y_{t_i}- \widehat\Uc_i^{}(X_{t_i})\big|^2 + \E \bigg[ \sum_{i=0}^{N-1} \int_{t_i}^{t_{i+1}} \big| Z_t - \widehat\Zc_i^{}(X_{t_i}) \big|^2 dt \bigg]. 
\end{aligned}
$$

Let us define from the scheme \eqref{eq:schemeVI} 

$$
\label{defVZ3} \left\{ 

$$
\begin{aligned}
\tilde\Vc_{t_i} & := \; \E_i \big[ \widehat\Uc_{i+1}(X_{t_{i+1}}) \big] + f(t_i,X_{t_i},\tilde\Vc_{t_i},\overline{{\tilde Z_{t_i}}}) \Delta t_i \; = \; \tilde v_i(X_{t_i}), \\ \overline{{\tilde Z_{t_i}}} & := \; \frac{1}{\Delta t_i} \E_i\left[ \widehat\Uc_{i+1}(X_{t_{i+1}}) \Delta W_{t_i} \right] \; = \; \tilde z_i(X_{t_i}), \\ \widehat\Vc_{t_i} & := \; \max[\tilde\Vc_{t_i} ; g(X_{t_i}) ], \;\;\; i =0,\ldots,N-1. 
\end{aligned}
$$

\right. 
$$

Our final main result gives an error estimate of the RDBDP scheme in terms of the $L^2$-approximation errors of $\tilde v_i$ and $\tilde z_i$ by neural networks $\Uc_i$ and $\Zc_i$, $i=0,\ldots,N-1$, assumed to be independent (see Remark [remNN](#remNN)), and defined as 

$$
\begin{aligned}
\eps_i^{\Nc,\tilde v} \; := \; \inf_{\xi} \E \big|\tilde v_i(X_{t_i}) - \Uc_i(X_{t_i};\xi) \big|^2, \hspace{7mm} \eps_i^{\Nc,\tilde z} \; := \; \inf_{\eta} \E\big|\tilde z_i(X_{t_i}) - \Zc_i(X_{t_i};\eta) \big|^2. 
\end{aligned}
$$

The result is obtained under one of the following additional assumptions \vspace{2mm} \noindent {\bf (H3)} $g$ is $C^1$, and $g$, $D_x g$ are Lipschitz. or \noindent {\bf (H4)} $\sigma$ is $C^1$, with $\sigma$, $D_x \sigma$ both Lipschitz, and $g$ is $C^2$, with $g$, $D_x g$, $D_x^2 g$ all Lipschitz. 
\begin{Theorem} \emph{(Consistency of RDBDP)} \label{theo:scheme3}

Let Assumption {\bf (H1)} hold.

There exists a constant $C>0$, independent of $\pi$, such that 

$$
\begin{aligned}
\Ec\big[(\widehat\Uc^{},\widehat\Zc^{}),(Y,Z)\big] & \leq \; C \Big( \eps(\pi) + \sum_{i=0}^{N-1} \big(N \eps_i^{\Nc,\tilde v} + \eps_i^{\Nc,\tilde z}\big) \Big), \label{eq:theo3} 
\end{aligned}
$$

with $\eps(\pi)$ $=$ $O(|\pi|^{\frac{1}{2}})$ under {\bf (H3)}, and $\eps(\pi)$ $=$ $O(|\pi|)$ under {\bf (H4)}. \end{Theorem}
\noindent {\bf Proof.}

Let us introduce the discrete-time approximation of the reflected BSDE 

$$
\label{defYZpi} \left\{ 

$$
\begin{aligned}

Y_{t_N}^\pi & = \; g(X_{t_N}) \\ Z_{t_i}^\pi &= \; \frac{1}{\Delta t_i} \E_i \big[ Y_{t_{i+1}}^\pi \Delta W_{t_i} \big], \\ \tilde Y_{t_i}^\pi & = \; \E_i[ Y_{t_{i+1}}^\pi ] + f(t_i,X_{t_i},\tilde Y_{t_i}^\pi,Z_{t_i}^\pi) \Delta t_i \\ Y_{t_i}^\pi & = \; \max \big[ \tilde Y_{t_i}^\pi ; g(X_{t_i}) \big] , \;\;\; i =0,\ldots,N-1. 
\end{aligned}
$$

\right. 
$$

It is known from [^Boucha08] that 

$$
\label{estimBC} \left\{ 

$$
\begin{aligned}
\max_{i=0,\ldots,N-1} \E \big| Y_{t_i} - Y_{t_i}^\pi \big|^2 & = \; \eps(\pi) \\ \E \bigg[ \sum_{i=0}^{N-1} \int_{t_i}^{t_{i+1}} \big| Z_t - Z_{t_i}^\pi \big|^2 dt \bigg] & = \; O(|\pi|^{\frac{1}{2}}), 
\end{aligned}
$$

\right. 
$$

with $\eps(\pi)$ $=$ $O(|\pi|^{\frac{1}{2}})$ under {\bf (H3)}, and $\eps(\pi)$ $=$ $O(|\pi|)$ under {\bf (H4)}. \vspace{1mm}

Fix $i$ $=$ $0,\ldots,N-1$.

By writing that 

$$
\begin{aligned}
\tilde Y_{t_i}^\pi - \tilde\Vc_{t_i} & = \E_i\big[ Y_{t_{i+1}} - \widehat\Uc_{i+1}^{}(X_{t_{i+1}}) \big] + \Delta t_i \Big( f(t_i,X_{t_i},\tilde Y_{t_i}^\pi,Z_{t_i}^\pi) - f(t_i,X_{t_i},\tilde\Vc_{t_i},\overline{{\tilde Z_{t_i}}}) \Big), 
\end{aligned}
$$

and proceeding similarly as in Step 1 in the proof of Theorem [theo:scheme1_1](#theo:scheme1_1), we have by Young inequality and Lipschitz condition on $f$ 

$$
\begin{aligned}
\E\big| \tilde Y_{t_i}^\pi - \tilde\Vc_{t_i} \big|^2 & \leq \; (1 +\gamma\Delta t_i) \E \Big| \E_i\big[ Y_{t_{i+1}}^\pi - \widehat\Uc_{i+1}^{}(X_{t_{i+1}}) \big] \Big|^2 \\ & \;\;\; + 2 \frac{[f]^2_{_L}}{\gamma} \big(1+ \gamma \Delta t_i\big) \Big\{ \Delta t_i \E \big| \tilde Y_{t_i}^\pi - \tilde\Vc_{t_i} \big|^2 + \Delta t_i \E \big| Z_{t_i}^\pi - \overline{{\tilde Z_{t_i}}} \big|^2 \Big\}. \label{Ypiinter} 
\end{aligned}
$$

From \eqref{defVZ3}, \eqref{defYZpi}, Cauchy-Schwarz inequality, and law of iterated conditional expectations, we have similarly as in Step 1 in the proof of Theorem [theo:scheme1_1](#theo:scheme1_1): 

$$
\begin{aligned}
\Delta t_i \E \big| Z_{t_i}^\pi - \overline{{\tilde Z_{t_i}}} \big|^2& \leq \; 2 d \Big( \E \big|Y_{t_{i+1}}^\pi- \widehat\Uc_{i+1}^{}(X_{t_{i+1}}) \big|^2 - \E \Big| \E_i\big[ Y_{t_{i+1}}^\pi - \widehat\Uc_{i+1}^{}(X_{t_{i+1}})\big] \Big|^2 \Big). 
\end{aligned}
$$

Then, by plugging into \eqref{Ypiinter} and choosing $\gamma$ $=$ $4 d [f]^2_{_L}$, we have for $|\pi|$ small enough: 

$$
\begin{aligned}
\E\big| \tilde Y_{t_i}^\pi - \tilde\Vc_{t_i} \big|^2 & \leq \; (1 + C |\pi|) \E \big|Y_{t_{i+1}}^\pi - \widehat\Uc_{i+1}^{}(X_{t_{i+1}}) \big|^2. 
\end{aligned}
$$

Next, by using Young inequality as in Step 2 in the proof of Theorem [theo:scheme1_1](#theo:scheme1_1), we obtain for all $\theta$ $=$ $(\xi,\zeta)$: 

$$
\begin{aligned}
\label{tildeYU} \hspace{-5mm} \E\big| \tilde Y_{t_i}^\pi - \Uc_{i}(X_{t_i};\xi) \big|^2 & \leq \; (1 + C |\pi|) \E \big|Y_{t_{i+1}}^\pi - \widehat\Uc_{i+1}^{}(X_{t_{i+1}}) \big|^2 + CN \E\big| \tilde\Vc_{t_i} - \Uc_{i}(X_{t_i};\xi) \big|^2. 
\end{aligned}
$$

On the other hand, by the martingale representation theorem, there exists an $\R^d$-valued square integrable process $(\tilde Z_t)_t$ such that 

$$
\begin{aligned}
\label{RFBSDE} \widehat\Uc_{i+1}^{}(X_{t_{i+1}}) & = \; \tilde\Vc_{t_i} - f(t_i,X_{t_i},\tilde\Vc_{t_i},\overline{{\tilde Z_{t_i}}}) \Delta t_i + \int_{t_i}^{t_{i+1}} \tilde Z_s\trans \diff W_s, 
\end{aligned}
$$

and the expected squared loss function of the RDBDP scheme can be written as 

$$
\begin{aligned}
\hat L_i(\theta) &= \; \tilde L_i(\theta) + \E \Big[ \int_{t_i}^{t_{i+1}} \big| \tilde Z_t - \overline{{\tilde Z_{t_i}}} \big|^2 \diff t \Big], 
\end{aligned}
$$

where we notice by It\^o isometry that $\overline{{\tilde Z_{t_i}}}$ $=$ $\frac{1}{\Delta t_i}\E_i\Big[ \int_{t_i}^{t_{i+1}} \tilde Z_t dt \Big]$, and 

$$
\begin{aligned}
\tilde L_i(\theta) & := \; \E \Big| \tilde\Vc_{t_i} - \Uc_i(X_{t_i};\xi) + \big( f(t_i,X_{t_i},\Uc_i(X_{t_i};\xi), \Zc_i(X_{t_i};\eta)) -f(t_i,X_{t_i},\tilde\Vc_{t_i},\overline{{\tilde Z_{t_i}}}) \big) \Delta t_i \Big|^2 \\ & \hspace{.6cm} + \; \Delta t_i \E \big| \overline{{\tilde Z_{t_i}}} - \Zc_i(X_{t_i};\eta) \big|^2. 
\end{aligned}
$$

By the same arguments as in Step 3 in the proof of Theorem [theo:scheme1_1](#theo:scheme1_1), using Lipschitz condition on $f$ and Young inequality, we show that for all $\theta$ $=$ $(\xi,\eta)$ 

$$
\begin{aligned}
(1 - C \Delta t_i) \E \big| \tilde\Vc_{t_i} - \Uc_i(X_{t_i};\xi) \big|^2 + \frac{\Delta t_i}{2} \E \big| \overline{{\tilde Z_{t_i}}} - \Zc_i(X_{t_i};\eta) \big|^2 \\ & \hspace{-7cm}\leq \tilde L_i(\theta) \; \leq \; (1 + C \Delta t_i) \E \big| \tilde\Vc_{t_i} - \Uc_i(X_{t_i};\xi) \big|^2 + C \Delta t_i \E \big| \overline{{\tilde Z_{t_i}}} - \Zc_i(X_{t_i};\eta) \big|^2. 
\end{aligned}
$$

By taking $\theta_i^*$ $=$ $(\xi_i^*,\eta_i^*)$ $\in$ ${\rm arg}\min_\theta \hat L_i^{}(\theta)$ $=$ ${\rm arg}\min_\theta \tilde L_i^{}(\theta)$, it follows that for $|\pi|$ small enough 

$$
\begin{aligned}
\E \big| \tilde\Vc_{t_i} - \Uc_i^{}(X_{t_i};\xi_i^*) \big|^2 + \Delta t_i \E \big| \overline{{\tilde Z_{t_i}}} - \Zc_i^{}(X_{t_i};\eta_i^*) \big|^2 & \leq \; C \eps_i^{\Nc,\tilde v} + C \Delta t_i \eps_i^{\Nc,\tilde z}. 
\end{aligned}
$$

By plugging into \eqref{tildeYU}, recalling that $\widehat\Uc_i(X_{t_i})$ $=$ $\max[\Uc_i(X_{t_i};\xi_i^*); g(X_{t_i})]$, $Y_{t_i}^\pi$ $=$ $\max[\tilde Y_{t_i}^\pi;g(X_{t_i})]$, and since $|\max(a,c) - \max(b,c)|$ $\leq$ $|a-b|$, we obtain 

$$
\begin{aligned}
\E\big| Y_{t_i}^\pi - \widehat\Uc_{i}(X_{t_i}) \big|^2 & \leq \; (1 + C |\pi|) \E \big|Y_{t_{i+1}}^\pi - \widehat\Uc_{i+1}^{}(X_{t_{i+1}}) \big|^2 + CN \big( \eps_i^{\Nc,\tilde v} + \Delta t_i \eps_i^{\Nc,\tilde z} \big), 
\end{aligned}
$$

and then by induction 

$$
\begin{aligned}
\max_{i=0,\ldots,N-1} \E\big| Y_{t_i}^\pi - \widehat\Uc_{i}(X_{t_i}) \big|^2 & \leq \; C \sum_{i=0}^{N-1} \big( N \eps_i^{\Nc,\tilde v} + \eps_i^{\Nc,\tilde z} \big). 
\end{aligned}
$$

Combining with \eqref{estimBC}, this proves the error estimate \eqref{eq:theo3} for the $Y$-component.

The error estimate \eqref{eq:theo3} for the $Z$-component is proved along the same arguments as in Step 5 in the proof of Theorem [theo:scheme1_1](#theo:scheme1_1), and is omitted here. \ep

## 5·Numerical results

\label{secnum}

In the first two subsections, we compare our schemes DBDP1 \eqref{eq:scheme1}, DBDP2 \eqref{eq:scheme2} and the scheme proposed by [^Han2017overcoming] on some examples of PDEs and BSDEs.

We first test our algorithms on some PDEs with bounded solutions and quite a simple structure (see section [sec:numeric1](#sec:numeric1)), and then try to solve some PDEs with unbounded solutions and more complex structures (see section [sec:numeric2](#sec:numeric2)).

Our goal is to emphasize that solutions with simple structure easily represented by a neural network can be evaluated by our method even in very high-dimension, whereas the solution with complex structure can only be evaluated in moderate dimension.

Finally, we apply the scheme described in section [sec:varIneq](#sec:varIneq) to an American option problem and show its accuracy in high dimension (see section [sec:numeric3](#sec:numeric3)).

If not specified, we use in the sequel a fully connected feedforward network with two hidden layers, and $d+10$ neurons on each hidden layer, to implement our schemes \eqref{eq:scheme1} and \eqref{eq:scheme2}.

We choose tanh as activation function for the hidden layers in order to avoid some explosion while calculating the numerical gradient $Z$ in scheme \eqref{eq:scheme2} and choose identity function as activation function for the output layer.

We renormalize the data before entering the network.

We use Adam Optimizer, implemented in TensorFlow and mini-batch with $1000$ trajectories for the stochastic gradient descent. 

### PDEs with bounded solution and simple structure

\label{sec:numeric1}

We begin with a simple example in dimension one.

It is not hard to find test cases where the scheme proposed in [^Han2017overcoming] fails even in dimension one.

In fact the latter scheme works well for small maturities and with a starting point close to the solution.

It is always interesting to start by testing schemes in dimension one as one can easily compare graphically the numerical results to the theoretical solution.

Then we take some examples in higher dimensions and show that our method seems to work well when the dimension increases higher. 

#### An example in 1D

We take the following parameters for the BSDE problem defined by \eqref{eq:SDE} and \eqref{eqBSDE}: 

$$
\sigma = 1 , \; \mu = 0.2 , \; T = 2 , \; d=1, \label{coeff:pb_simple} 
$$

$$

\begin{array}{rclrclrclrcl} f(t,x,y,z) &= & (\cos(x) (e^{\frac{T-t}{2}}+ \frac{\sigma^2}{2}) + \mu \sin(x)) e^{\frac{T-t}{2}} - \frac{1}{2} \left( \sin(x)\cos(x) e^{T-t} \right)^2 + \frac{1}{2}(yz)^2& & & & & & & & & \\ g(x)&= &\cos(x). & & & & & & & \end{array}

$$

for which, the explicit analytic solution is equal to $u(t,x) =e^{\frac{T-t}{2}} \cos(x)$.

We want to estimate the solution $u$ and its gradient $D_x u$ from our schemes.

This example is interesting, because with $T=1$, the method proposed in [^Han2017overcoming], initializing $u(0,.)$ as the solution of the associated linear problem associated ($f=0$) and randomly initializing $D_x u(0,.)$ works very well.

However, for $T=2$, the method in [^Han2017overcoming] always fails on our test whatever the choice of the initialization: the algorithm is either trapped in a local minimum when the initial learning rate associated to the gradient method is too small or explodes when the learning rate is taken higher.

This numerical failure is not dependent on the considered network: using some LSTM networks as in [^Chan2018machine] gives the same result.

Because of the high non-linearity, we discretize the BSDE using $N$ $=$ $240$ time steps, and implemented hidden layers with $d+10$ $=$ $11$ neurons.

Figure [fig:quentinCase1DS1](#fig:quentinCase1DS1) (resp.

Figure [fig:quentinCase1DS2](#fig:quentinCase1DS2)) depicts the estimated functions $u(t,.)$ and $D_x u(t,.)$ estimated from DBDP1 (resp.

DBDP2) scheme. 
![](PDEQuentd1ndt240AUTOFalse_U_120.png)

<a id='fig:quentinCase1DS1'>$u(t,.)$ and its estimate at time $t=1.$</a>

![](PDEQuentd1ndt240AUTOTrue_U_120.png)

<a id='fig:quentinCase1DS2'>$u(t,.)$ and its estimate at time $t=1.$</a>

 |
| | Averaged value | Standard deviation  |
| DBDP1 | 1.46332 | 0.01434  |
| DBDP2 | 1.4387982 | 0.01354  |
<a id=tab:sol1D>Estimate of $u(0,x_0)$ where $d$ $=$ $1$ and $x_0$ $=$ $1$.

Average and standard deviation observed over 10 independent runs are reported.

The theoretical solution is $1.4686938$.</a>

\clearpage 

#### Increasing the dimension

We extend the example from the previous section to the following $d$-dimensional problem: 
\begin{flalign*} d\geq 1, \quad & \sigma = \frac{1}{\sqrt{d}} \I_d, \quad \mu = \frac{0.2}{d} \un_d , \quad T = 1 , \end{flalign*}

$$

\begin{array}{rcl} f(t,x,y,z) & = & (\cos( \bar x) (e^{\frac{T-t}{2}}+ \frac{1}{2}) + 0.2 \sin( \bar x)) e^{\frac{T-t}{2}} - \frac{1}{2} \left( \sin(\bar x)\cos(\bar x) e^{T-t} \right)^2 + \frac{1}{2d}(u(\un_d.z))^2,\\ g(x) & = &\cos(\bar x), \end{array}

$$

with $\bar x = \sum_{i=1}^d x_i$.

We take $N$ $=$ $120$ in the Euler scheme, and $d+10$ neurons for each hidden layer.

We take $1000$ trajectories in mini batch, use data renormalization, and check the loss convergence every $50$ iterations.

For this small maturity, the scheme [^Han2017overcoming] generally converges, and we give the results obtained with the same network and initializing the scheme with the linear solution of the problem.

Results in dimension 5 to 50 are given in Tables [tab:sol5Dsimple](#tab:sol5Dsimple), [tab:sol10Dsimple](#tab:sol10Dsimple), [tab:sol20D](#tab:sol20D) and [tab:sol50D](#tab:sol50D).

Both schemes \eqref{eq:scheme1} and \eqref{eq:scheme2} work well with results very close to the solution and close to the results calculated by the scheme [^Han2017overcoming].

As the dimension increases, scheme \eqref{eq:scheme1} seems to be the most accurate. 
\begin{Remark} {\rm In dimension $50$, the initial learning rate in scheme [^Han2017overcoming] is taken small in order to avoid a divergence of the method.

In fact, running the test 3 times (with 10 runs each time), we observed convergence of the algorithm two times, and in the last test: one of the ten run exploded, and another one clearly converged to a wrong solution. } \ep \end{Remark}

 |
| | Averaged value | Standard deviation  |
| DBDP1 | 0.4637038 | 0.004253  |
| DBDP2 | 0.46335 | 0.00137  |
| Scheme cite{han2017overcoming} | 0.46562 | 0.0035 |
<a id=tab:sol5Dsimple>Estimate of $u(0,x_0)$ where $d$ $=$ $5$ and $x_0$ $=$ $\un_5$.

Average and standard deviation observed over 10 independent runs are reported.

The theoretical solution is $0.46768$. </a>

 |
| | Averaged value | Standard deviation  |
| DBDP1 | - 1.3895 | 0.00148 |
| DBDP2 | -1.3913 | 0.000583  |
| Scheme cite{han2017overcoming} | -1.3880 | 0.00155 |
<a id=tab:sol10Dsimple> Estimate of $u(0,x_0)$ where $d$ $=$ $10$ and $x_0$ $=$ $\un_{10}$.

Average and standard deviation observed over 10 independent runs are reported.

The theoretical solution is $-1.383395$. </a>

 |
| | Averaged value | Standard deviation  |
| DBDP1 | 0.6760 | 0.00274  |
| DBDP2 |0.67102 | 0.00559  |
| Scheme cite{han2017overcoming} | 0.68686 | 0.002402  |
<a id=tab:sol20D>

Estimate of $u(0,x_0)$ where $d$ $=$ $20$ and $x_0$ $=$ $\un_{20}$.

Average and standard deviation observed over 10 independent runs are reported.

The theoretical solution is $0.6728135$. </a>

 |
| | Averaged value | Standard deviation  |
| DBDP1 | 1.5903 | 0.006276  |
| DBDP2 | 1.58762| 0.00679  |
| Scheme cite{han2017overcoming} | 1.583023 | 0.0361  |
<a id=tab:sol50D>

Estimate of $u(0,x_0)$ where $d$ $=$ $50$ and $x_0$ $=$ $\un_{50}$.

Average and standard deviation observed over 10 independent runs are reported.

The theoretical solution is $1.5909$. </a>

### PDEs with unbounded solution and more complex structure

\label{sec:numeric2}

In this section with take the following parameters 
\begin{flalign} \sigma & = \; \frac{1}{\sqrt{d}} \I_d, \quad \mu = 0, \quad T = 1 , \\ f(x,y,z) & = \; k(x) + \frac{1}{2 \sqrt{d}} y(\un_d.z) + \frac{ y^2}{2} \label{coeff:pb_complex} \end{flalign}
where the function $k$ is chosen such that the solution to the PDE is equal to 
\begin{flalign*} u(t,x) = \frac{T-t}{d} \sum_{i=1}^d ( \sin(x_i) 1_{x_i<0} + x_i 1_{x_1 \ge 0} ) + \cos \left(\sum_{i=1}^d i x_i \right). \end{flalign*}

Notice that the structure of the solution is more complex than in the first example.

We aim at evaluating the solution at $x=0.5 \un_d$.

We take $120$ time steps for the Euler time discretization and $d+10$ neurons in each hidden layers.

As shown in Figures [fig:huyenCase1DS1](#fig:huyenCase1DS1) and [fig:huyenCase1DS2](#fig:huyenCase1DS2) as well as in Table [tab:sol1D_complex](#tab:sol1D_complex), the three schemes provide accurate and stable results in dimension $d$ $=$ $1$. 
![](PDEHuyend1ndt120AUTOFalse_U_60.png)

<a id='fig:huyenCase1DS1'>$u(t,.)$ and its estimate at time $t=0.5$.</a>

![](PDEHuyend1ndt120AUTOTrue_U_60.png)

<a id='fig:huyenCase1DS2'>$u(t,.)$ and its estimate at time $t=0.5$.</a>

 |
| | Averaged value | Standard deviation  |
| DBDP1 | 1.3720 | 0.00301  |
| DBDP2 | 1.37357 | 0.0022  |
| Scheme cite{han2017overcoming} | 1.37238 | 0.00045  |
<a id=tab:sol1D_complex> Estimate of $u(0,x_0)$, where $d$ $=$ $1$ and $x_0$ $=$ $0.5$.

Average and standard deviation observed over 10 independent runs are reported.

The theoretical solution is $1.37758$. </a>

In dimension 2, the three schemes provide very accurate and stable results, as shown in Figures [fig:huyenCase2DS1](#fig:huyenCase2DS1) and [fig:huyenCase2DS2](#fig:huyenCase2DS2), as well as in Table [tab:sol2D](#tab:sol2D). 
![](PDEHuyend2ndt120AUTOFalse_U_60.png)

<a id='fig:huyenCase2DS1'>Error on solution at date $t=0.5$.</a>

![](PDEHuyend2ndt120AUTOTrue_U_60.png)

<a id='fig:huyenCase2DS2'>Error on solution at date $t=0.5$.</a>

 |
| | Averaged value | Standard deviation  |
| DBDP1 | 0.5715359 | 0.0038  |
| DBDP2 | 0.5707974 | 0.00235 |
| Scheme cite{han2017overcoming} | 0.57145 | 0.0006  |
<a id=tab:sol2D>

Estimate of $u(0,x_0)$, where $d$ $=$ $2$ and $x_0$ $=$ $0.5\un_{2}$.

Average and standard deviation observed over 10 independent runs are reported.

The theoretical solution is $0.570737$. </a>

Above dimension 3, the scheme [^Han2017overcoming] always explodes no matter the chosen initial learning rate and the activation function for the hidden layers (among the $\tanh$, ELU, ReLu and sigmoid ones).

Besides, taking $3$ or $4$ hidden layers does not improve the results.

We reported the results obtained in dimension $d$ $=$ $5$ and $8$ in Table [tab:sol5D](#tab:sol5D) and [tab:sol8D](#tab:sol8D).

Scheme \eqref{eq:scheme1} seems to work better than scheme \eqref{eq:scheme2} as the dimension increases.

Note that the standard deviation increases with the dimension of the problem. 
 |
| | Averaged value | Standard deviation  |
| DBDP1 | 0.8666 | 0.013  |
| DBDP2 | 0.83646 | 0.00453  |
| Scheme cite{han2017overcoming} | NC | NC  |
<a id=tab:sol5D>

Estimate of $u(0,x_0)$, where $d$ $=$ $5$ and $x_0$ $=$ $0.5\un_{5}$.

Average and standard deviation observed over 10 independent runs are reported.

The theoretical solution is $0.87715$. </a>

 |
| | Averaged value | Standard deviation  |
| DBDP1 | 1.169441 | 0.02537  |
| DBDP2 | 1.0758344 | 0.00780  |
| Scheme cite{han2017overcoming} | NC | NC  |
<a id=tab:sol8D>

Estimate of $u(0,x_0)$, where $d$ $=$ $8$ and $x_0$ $=$ $0.5\un_{8}$.

Average and standard deviation observed over 10 independent runs are reported.

The theoretical solution is $1.1603167$. </a>

When $d\geq10$, schemes \eqref{eq:scheme1} and \eqref{eq:scheme2} both fail at providing correct estimates of the solution, as shown in Table [tab:sol10D](#tab:sol10D).

Increasing the number of layers or neurons does not improve the result. 
 |
| | Averaged value | Standard deviation  |
| DBDP1 | -0.3105 | 0.02296  |
| DBDP2 | -0.3961| 0.0139 |
| Scheme cite{han2017overcoming} | NC | NC  |
<a id=tab:sol10D>

Estimate of $u(0,x_0)$, where $d$ $=$ $10$ and $x_0$ $=$ $0.5\un_{10}$.

Average and standard deviation observed over 10 independent runs are reported.

The theoretical solution is $-0.2148861$. </a>

#