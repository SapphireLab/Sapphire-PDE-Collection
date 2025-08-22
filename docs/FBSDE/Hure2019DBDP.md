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

#