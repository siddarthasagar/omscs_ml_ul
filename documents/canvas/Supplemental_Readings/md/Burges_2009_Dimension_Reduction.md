# Dimension Reduction: A Guided Tour

**Christopher J.C. Burges**  
Microsoft Research, One Microsoft Way, Redmond, WA 98052-6399, USA.  
MSR Tech Report MSR-TR-2009-2013

---

## Abstract

We give a tutorial overview of several geometric methods for dimension reduction. We divide the methods into projective methods and methods that model the manifold on which the data lies. For projective methods, we review projection pursuit, principal component analysis (PCA), kernel PCA, probabilistic PCA, canonical correlation analysis, oriented PCA, and several techniques for sufficient dimension reduction. For the manifold methods, we review multidimensional scaling (MDS), landmark MDS, Isomap, locally linear embedding, Laplacian eigenmaps and spectral clustering. The Nyström method, which links several of the manifold algorithms, is also reviewed. The goal is to provide a self-contained overview of key concepts underlying many of these algorithms, and to give pointers for further reading.

---

## Contents

1. [Introduction](#1-introduction)
2. [Projective Methods](#2-projective-methods)
   - 2.1 [Principal Components Analysis (PCA)](#21-principal-components-analysis-pca)
   - 2.2 [Probabilistic PCA (PPCA)](#22-probabilistic-pca-ppca)
   - 2.3 [Kernel PCA](#23-kernel-pca)
   - 2.4 [Canonical Correlation Analysis](#24-canonical-correlation-analysis)
   - 2.5 [Oriented PCA and Distortion Discriminant Analysis](#25-oriented-pca-and-distortion-discriminant-analysis)
   - 2.6 [Sufficient Dimension Reduction](#26-sufficient-dimension-reduction)
3. [Manifold Modeling](#3-manifold-modeling)
   - 3.1 [The Nyström method](#31-the-nyström-method)
   - 3.2 [Multidimensional Scaling](#32-multidimensional-scaling)
   - 3.3 [Isomap](#33-isomap)
   - 3.4 [Locally Linear Embedding](#34-locally-linear-embedding)
   - 3.5 [Graphical Methods](#35-graphical-methods)
   - 3.6 [Pulling the Threads Together](#36-pulling-the-threads-together)
4. [Conclusion](#4-conclusion)
- [Acknowledgements](#acknowledgements)
- [References](#references)

---

## 1 Introduction

Dimension reduction[^1] is the mapping of data to a lower dimensional space such that uninformative variance in the data is discarded, or such that a subspace in which the data lives is detected. Dimension reduction has a long history as a method for data visualization, and for extracting key low dimensional features (for example, the 2-dimensional orientation of an object, from its high dimensional image representation). In some cases the desired low dimensional features depend on the task at hand. Apart from teaching us about the data, dimension reduction can lead us to better models for inference. The need for dimension reduction also arises for other pressing reasons. [Stone, 1982] showed that, under certain regularity assumptions (including that the samples be IID), the optimal rate of convergence[^2] for nonparametric regression varies as $m^{-p/(2p+d)}$, where $m$ is the sample size, the data lies in $\mathcal{R}^d$, and where the regression function is assumed to be $p$ times differentiable. We can get a very rough idea of the impact of sample size on the rate of convergence as follows. Consider a particular point in the sequence of values corresponding to the optimal rate of convergence: $m = 10,000$ samples, for $p = 2$ and $d = 10$. Suppose that $d$ is increased to 20; what number of samples in the new sequence gives the same value? The answer is approximately 10 million. If our data lie (approximately) on a low dimensional manifold $\mathcal{L}$ that happens to be embedded in a high dimensional manifold $\mathcal{H}$, then modeling the data directly in $\mathcal{L}$ rather than in $\mathcal{H}$ may turn an infeasible problem into a feasible one.

The purpose of this review is to describe the mathematics and key ideas underlying the methods, and to provide some links to the literature for those interested in pursuing a topic further[^3]. The subject of dimension reduction is vast, so we use the following criterion to limit the discussion: we restrict our attention to the case where the inferred feature values are continuous. The observables, on the other hand, may be continuous or discrete. Thus this review does not address clustering methods, or, for example, feature selection for discrete data, such as text. Furthermore implementation details, and important theoretical details such as consistency and rates of convergence of sample quantities to their population values, although important, are not discussed.

Regarding notation: vectors are denoted by boldface, whereas components are denoted by $x_a$, or by $(\mathbf{x}_i)_a$ for the $a$'th component of the $i$'th vector. Random variables are denoted by upper case; we use $E[X|y]$ as shorthand for the function $E[X|Y = y]$, in contrast to the random variable $E[X|Y]$. Following [Horn and Johnson, 1985], the set of $p$ by $q$ matrices is denoted $M_{pq}$, the set of (square) $p$ by $p$ matrices by $M_p$, and the set of symmetric $p$ by $p$ matrices by $S_p$ (all matrices considered are real). **e** with no subscript is used to denote the vector of all ones; on the other hand $\mathbf{e}_a$ denotes the $a$'th eigenvector. We denote sample size by $m$, and dimension usually by $d$ or $d'$, with typically $d' \ll d$. $\delta_{ij}$ is the Kronecker delta (the $ij$'th component of the unit matrix). We generally reserve indices $i, j$, to index vectors and $a, b$ to index dimension.

We place dimension reduction techniques into two broad categories: methods that rely on projections (Chapter 2) and methods that attempt to model the manifold on which the data lies (Chapter 3). Chapter 2 gives a detailed description of principal component analysis; apart from its intrinsic usefulness, PCA is interesting because it serves as a starting point for many modern algorithms, some of which (kernel PCA, probabilistic PCA, and oriented PCA) are also described here. However it has clear limitations: it is easy to find even low dimensional examples where the PCA directions are far from optimal for feature extraction [Duda and Hart, 1973], and PCA ignores correlations in the data that are higher than second order. We end Chapter 2 with a brief look at projective methods for dimension reduction of labeled data: sliced inverse regression, and kernel dimension reduction. Chapter 3 starts with an overview of the Nyström method, which can be used to extend, and link, several of the algorithms described in this paper. We then examine some methods for dimension reduction which assume that the data lie on a low dimensional manifold embedded in a high dimensional space, namely locally linear embedding, multidimensional scaling, Isomap, Laplacian eigenmaps, and spectral clustering.

[^1]: We follow both the lead of the statistics community and the spirit of the paper to reduce "dimensionality reduction" and "dimensional reduction" to "dimension reduction".
[^2]: For convenience we reproduce Stone's definitions [Stone, 1982]. A "rate of convergence" is defined as a sequence of numbers, indexed by sample size. Let $\theta$ be the unknown regression function, $\Theta$ the collection of functions to which $\theta$ belongs, $\hat{T}_n$ an estimator of $\theta$ using $n$ samples, and $\{b_n\}$ a sequence of positive constants. Then $\{b_n\}$ is called a lower rate of convergence if there exists $c > 0$ such that $\lim_n \inf_{\hat{T}_n} \sup_\Theta P(\|\hat{T}_n - \theta\| \geq cb_n) = 1$, and it is called an achievable rate of convergence if there is a sequence of estimators $\{\hat{T}_n\}$ and $c > 0$ such that $\lim_n \sup_\Theta P(\|\hat{T}_n - \theta\| \geq cb_n) = 0$; $\{b_n\}$ is called an optimal rate of convergence if it is both a lower rate of convergence and an achievable rate of convergence. Here the $\inf_{\hat{T}_n}$ is over all possible estimators $\hat{T}_n$.
[^3]: This paper is a revised and extended version of [Burges, 2005].

---

## 2 Projective Methods

If dimension reduction is so desirable, how should we go about it? Perhaps the simplest approach is to attempt to find low dimensional *projections* that extract useful information from the data, by maximizing a suitable objective function. This is the idea of projection pursuit [Friedman and Tukey, 1974]. The name 'pursuit' arises from the iterative version, where the currently optimal projection is found in light of previously found projections (in fact originally this was done manually[^pp1]). Apart from handling high dimensional data, projection pursuit methods can be robust to noisy or irrelevant features [Huber, 1985], and have been applied to regression [Friedman and Stuetzle, 1981], where the regression is expressed as a sum of 'ridge functions' (functions of the one dimensional projections) and at each iteration the projection is chosen to minimize the residuals; to classification; and to density estimation [Friedman et al., 1984]. How are the interesting directions found? One approach is to search for projections such that the projected data departs from normality [Huber, 1985]. One might think that, since a distribution is normal if and only if all of its one dimensional projections are normal, if the least normal projection of some dataset is still approximately normal, then the dataset is also necessarily approximately normal, but this is not true; Diaconis and Freedman have shown that most projections of high dimensional data are approximately normal [Diaconis and Freedman, 1984] (see also below). Given this, finding projections along which the density departs from normality, if such projections exist, should be a good exploratory first step.

[^pp1]: See J.H. Friedman's interesting response to [Huber, 1985] in the same issue.

The sword of Diaconis and Freedman cuts both ways, however. If most projections of most high dimensional datasets are approximately normal, perhaps projections are not always the best way to find low dimensional representations. Let's review their results in a little more detail. The main result can be stated informally as follows: consider a model where the data, the dimension $d$, and the sample size $m$ depend on some underlying parameter $\nu$, such that as $\nu$ tends to infinity, so do $m$ and $d$. Suppose that as $\nu$ tends to infinity, the fraction of vectors which are not approximately the same length tends to zero, and suppose further that under the same conditions, the fraction of pairs of vectors which are not approximately orthogonal to each other also tends to zero[^df1]. Then ([Diaconis and Freedman, 1984], Theorem 1.1) the empirical distribution of the projections along any given unit direction tends to $N(0, \sigma^2)$ weakly in probability[^df2]. However, if the conditions are not fulfilled, as for some long-tailed distributions, then the opposite result can hold — that is, most projections are *not* normal (for example, most projections of Cauchy distributed data[^df3] will be Cauchy [Diaconis and Freedman, 1984]).

[^df1]: More formally, the conditions are: for $\sigma^2$ positive and finite, and for any positive $\epsilon$, $(1/m)\text{card}\{j \leq m : \|\mathbf{x}_j\|^2 - \sigma^2 d| > \epsilon d\} \to 0$ and $(1/m^2)\text{card}\{1 \leq j, k \leq m : |\mathbf{x}_j \cdot \mathbf{x}_k| > \epsilon d\} \to 0$ [Diaconis and Freedman, 1984].
[^df2]: Some authors refer to convergence 'weakly in probability' simply as convergence in probability. A sequence $X_n$ of random variables is said to converge in probability to a random variable $X$ if $\lim_{n\to\infty} P(|X_n - X| > \epsilon) = 0$ for all $\epsilon > 0$ [Grimmet and Stirzaker, 2001].
[^df3]: The Cauchy distribution in one dimension has density $c/(c^2 + x^2)$ for constant $c$.

As a concrete example, consider data uniformly distributed over the unit $n+1$-sphere $S^{n+1}$ for odd[^sphere] $n$. Let's compute the density projected along any line $\mathcal{I}$ passing through the origin. By symmetry, the result will be independent of the direction we choose. If the distance along the projection is parameterized by $\xi \equiv \cos\theta$, where $\theta$ is the angle between $\mathcal{I}$ and the line from the origin to a point on the sphere, then the density at $\xi$ is proportional to the volume of an $n$-sphere of radius $\sin\theta$: $\rho(\xi) = C(1 - \xi^2)^{\frac{n-1}{2}}$. Requiring that $\int_{-1}^{1} \rho(\xi)d\xi = 1$ gives the constant $C$:

$$C = 2^{-\frac{1}{2}(n+1)} \frac{n!!}{(\frac{1}{2}(n-1))!} \tag{2.1}$$

[^sphere]: The story for even $n$ is similar but the formulae are slightly different.

Let's plot this density and compare against a one dimensional Gaussian density fitted using maximum likelihood. For that we just need the variance, which can be computed analytically: $\sigma^2 = \frac{1}{n+2}$, and the mean, which is zero. Figure 2.1 shows the result for the 20-sphere. Although data uniformly distributed on $S^{20}$ is far from Gaussian, its projection along any direction is close to Gaussian for all such directions, and we cannot hope to uncover such structure using one dimensional projections.

*Fig. 2.1 Dotted line: a Gaussian with zero mean and variance 1/21. Solid line: the density projected from data distributed uniformly over the 20-sphere, to any line passing through the origin.*

The notion of searching for non-normality, which is at the heart of projection pursuit (the goal of which is dimension reduction), is also a key idea underlying independent component analysis (ICA) [Hyvärinen et al., 2001]. ICA views the data as being generated by a mixture of unknown latent variables, and although typically the number of latent variables is assumed to equal the dimension of the data, the method has parallels with dimension reduction, so we briefly describe it here. ICA searches for projections such that the probability distributions of the data along those projections are statistically independent. Consider for example the case of two speakers speaking into two microphones, where each microphone captures sound from both speakers. The microphone signals may be written $\mathbf{y} = A\mathbf{x}$, $\mathbf{x}, \mathbf{y} \in \mathbb{R}^2$, where the components of $\mathbf{x}$ are the (assumed statistically independent and zero mean) signals from each individual speaker, and where $A$ is a fixed two dimensional mixing matrix. In principle, we could separate out the source signals by finding $A$ and inverting it. However, both $A$ and $\mathbf{x}$ are unknown here, and any invertible scaling of each component of $\mathbf{x}$, followed by any permutation of the components of the rescaled $\mathbf{x}$ (the net result of which is another pair of statistically independent variables) can be compensated by redefining $A$. We can remove the scaling degrees of freedom from the problem by whitening the data $\mathbf{y}$ and then assuming that $A$ is a rotation matrix, which amounts to choosing a coordinate system in which $\mathbf{x}$ is white (which, since the $x_i$ are independent and zero mean, is equivalent to just rescaling the $x_i$). Note that this also means that if $\mathbf{x}$ happens to be normally distributed, then ICA fails, since $A$ can then be any orthogonal matrix (since any orthogonal matrix applied to independent, unit variance Gaussian variables results in independent, unit variance Gaussian variables). To give nontrivial results, ICA therefore requires that the original signals be non-Gaussian (or more precisely, that at most one is Gaussian distributed), and in fact it turns out that finding the maximally non-Gaussian component (under the assumptions that the $\mathbf{x}$ are IID, zero mean, and unit variance) will yield an independent component [Hyvärinen et al., 2001]. ICA components may also be found by searching for components with minimum mutual information, since zero mutual information corresponds to statistical independence. Such functions — whose optimization leads to the desired independent components — are called contrast functions. [Bach and Jordan, 2002] generalize ICA by proposing contrast functions based on canonical correlation analysis (CCA) in Reproducing Kernel Hilbert Spaces (RKHSs); we will encounter CCA, and RKHS's used in similar ways, below.

---

### 2.1 Principal Components Analysis (PCA)

#### 2.1.1 PCA: Finding an Informative Direction

Given data $\mathbf{x}_i \in \mathcal{R}^d$, $i = 1, \cdots, m$, suppose you'd like to find a direction $\mathbf{v} \in \mathcal{R}^d$ for which the projection $\mathbf{x}_i \cdot \mathbf{v}$ gives a good one dimensional representation of your original data: that is, informally, the act of projecting loses as little information about your expensively-gathered data as possible (we will examine the information theoretic view of this below). Suppose that unbeknownst to you, your data in fact lies along a line $\mathcal{I}$ embedded in $\mathcal{R}^d$, that is, $\mathbf{x}_i = \boldsymbol{\mu} + \theta_i \mathbf{n}$, where $\boldsymbol{\mu}$ is the sample mean[^pca1], $\theta_i \in \mathcal{R}$, $\sum_i \theta_i = 0$, and $\mathbf{n} \in \mathcal{R}^d$ has unit length. The sample variance of the projection along $\mathbf{n}$ is then[^pca2]

$$v_n \equiv \frac{1}{m} \sum_{i=1}^{m} ((\mathbf{x}_i - \boldsymbol{\mu}) \cdot \mathbf{n})^2 = \frac{1}{m} \sum_{i=1}^{m} \theta_i^2 \tag{2.2}$$

and that along some other unit direction $\mathbf{n}'$ is

$$v'_n \equiv \frac{1}{m} \sum_{i=1}^{m} ((\mathbf{x}_i - \boldsymbol{\mu}) \cdot \mathbf{n}')^2 = \frac{1}{m} \sum_{i=1}^{m} \theta_i^2 (\mathbf{n} \cdot \mathbf{n}')^2 \tag{2.3}$$

[^pca1]: Note that if all $x_i$ lie on a given line then so does $\mu$.
[^pca2]: When the choice is immaterial to the argument, we use denominator $m$ (sample viewed as the whole population) rather than $m-1$ (unbiased estimator of population variance).

Since $(\mathbf{n} \cdot \mathbf{n}')^2 = \cos^2\phi$, where $\phi$ is the angle between $\mathbf{n}$ and $\mathbf{n}'$, we see that the projected variance is maximized if and only if $\mathbf{n} = \pm\mathbf{n}'$. Hence in this case, finding the projection for which the projected variance is maximized gives you the direction you are looking for, namely $\mathbf{n}$, *regardless of the distribution of the data along* $\mathbf{n}$, as long as the data has finite variance. You would then quickly find that the variance along all directions orthogonal to $\mathbf{n}$ is zero, and conclude that your data in fact lies along a one dimensional manifold embedded in $\mathcal{R}^d$. This is one of several basic results of PCA that hold for arbitrary distributions, as we shall see.

Even if the underlying physical process generates data that ideally lies along $\mathcal{I}$, noise will usually modify the data at various stages up to and including the measurements themselves, and so your data will very likely not lie exactly along $\mathcal{I}$. If the overall noise is much smaller than the signal, it makes sense to try to find $\mathcal{I}$ by searching for that projection along which the projected data has maximal variance. If instead your data lies in a two (or higher) dimensional subspace, the above argument can be repeated, picking off the highest variance directions in turn. Let's see how that works.

#### 2.1.2 PCA: Ordering by Variance

We have seen that directions of maximum variance can be interesting, but how can we find them? From here on, unless otherwise stated, we allow the $\mathbf{x}_i$ to be arbitrarily distributed. The sample variance along an arbitrary unit vector $\mathbf{n}$ is $\mathbf{n}^T C \mathbf{n}$ where $C$ is the sample covariance matrix. Since $C$ is positive semidefinite, its eigenvalues are positive or zero; let us choose the indexing such that the (unit norm) eigenvectors $\mathbf{e}_a$, $a = 1, \ldots, d$ are arranged in order of decreasing size of the corresponding eigenvalues $\lambda_a$. Since the $\{\mathbf{e}_a\}$ span the space (or can be so chosen, if several share the same eigenvalue), we can expand any $\mathbf{n}$ in terms of them: $\mathbf{n} = \sum_{a=1}^d \alpha_a \mathbf{e}_a$, and we would like to find the $\alpha_a$ that maximize $\mathbf{n}^T C \mathbf{n} = \mathbf{n}^T \sum_a \alpha_a C\mathbf{e}_a = \sum_a \lambda_a \alpha_a^2$, subject to $\sum_a \alpha_a^2 = 1$ (to give unit normed $\mathbf{n}$). This is just a convex combination of the $\lambda$'s, and since a convex combination of any set of numbers is maximized by taking the largest, the optimal $\mathbf{n}$ is just $\mathbf{e}_1$, the principal eigenvector (or any one of the principal eigenvectors, if the principal eigenvalue has geometric multiplicity greater than one), and furthermore, the sample variance of the projection of the data along $\mathbf{n}$ is then just $\lambda_1$.

The above construction captures the variance of the data along the direction $\mathbf{n}$. To characterize the remaining variance of the data, let's find that direction $\mathbf{m}$ which is both orthogonal to $\mathbf{n}$, and along which the projected data again has maximum variance. Since the eigenvectors of $C$ form an orthonormal basis (or can be so chosen), we can expand $\mathbf{m}$ in the subspace $\mathcal{R}^{d-1}$ orthogonal to $\mathbf{n}$ as $\mathbf{m} = \sum_{a=2}^d \beta_a \mathbf{e}_a$. Just as above, we wish to find the $\beta_a$ that maximize $\mathbf{m}^T C \mathbf{m} = \sum_{a=2}^d \lambda_a \beta_a^2$, subject to $\sum_{a=2}^d \beta_a^2 = 1$, and by the same argument, the desired direction is given by the (or any) remaining eigenvector with largest eigenvalue, and the corresponding variance is just that eigenvalue. Repeating this argument gives $d$ orthogonal directions, in order of monotonically decreasing projected variance. PCA for feature extraction thus amounts to projecting the data to a lower dimensional space: given an input vector $\mathbf{x}$, the mapping consists of computing the projections of $\mathbf{x}$ along the $\mathbf{e}_a$, $a = 1, \ldots, d'$, thereby constructing the components of the projected $d'$-dimensional feature vectors. Finally, since the $d$ directions are orthogonal, they also provide a complete basis. Thus if one uses all $d$ directions, no information is lost; and as we'll see below, given that one wants to project to a $d' < d$ dimensional space, if one uses the $d'$ principal directions, then the mean squared error introduced by representing the data by their projections along these directions is minimized.

#### 2.1.3 PCA Decorrelates the Data

Now suppose we've performed PCA on our samples, and instead of using it to construct low dimensional features, we simply use the full set of orthonormal eigenvectors as a choice of basis. In the old basis, a given input vector $\mathbf{x}$ is expanded as $\mathbf{x} = \sum_{a=1}^d x_a \mathbf{u}_a$ for some orthonormal set $\{\mathbf{u}_a\}$, and in the new basis, the same vector is expanded as $\mathbf{x} = \sum_{b=1}^d \tilde{x}_b \mathbf{e}_b$, so $\tilde{x}_a \equiv \mathbf{x} \cdot \mathbf{e}_a = \mathbf{e}_a \cdot \sum_b x_b \mathbf{u}_b$. The mean $\boldsymbol{\mu} \equiv \frac{1}{m} \sum_i \mathbf{x}_i$ has components $\tilde{\mu}_a = \boldsymbol{\mu} \cdot \mathbf{e}_a$ in the new basis. The sample covariance matrix depends on the choice of basis: if $C$ is the covariance matrix in the old basis, then the corresponding covariance matrix in the new basis is $\tilde{C}_{ab} \equiv \frac{1}{m} \sum_i (\tilde{x}_{ia} - \tilde{\mu}_a)(\tilde{x}_{ib} - \tilde{\mu}_b) = \mathbf{e}_a' C \mathbf{e}_b = \lambda_b \delta_{ab}$. Hence in the new basis the covariance matrix is diagonal and the samples are uncorrelated. It's worth emphasizing two points: first, although the covariance matrix can be viewed as a geometric object in that it transforms as a tensor (since it is a summed outer product of vectors, which themselves have a meaning independent of coordinate system), nevertheless, the notion of correlation is basis-dependent (data can be correlated in one basis and uncorrelated in another). Second, no assumptions regarding the distribution of $X$ has been made here.

#### 2.1.4 PCA: Reconstruction with Minimum Squared Error

The basis provided by the eigenvectors of the covariance matrix is also optimal for dimension reduction in the following sense. Again consider some arbitrary orthonormal basis $\{\mathbf{u}_a, a = 1, \ldots, d\}$, and take the first $d'$ of these to perform the dimension reduction: $\tilde{\mathbf{x}} \equiv \sum_{a=1}^{d'} (\mathbf{x} \cdot \mathbf{u}_a)\mathbf{u}_a$. The chosen $\mathbf{u}_a$ form a basis for $\mathcal{R}^{d'}$, so we may take the components of the dimensionally reduced vectors to be $\mathbf{x} \cdot \mathbf{u}_a$, $a = 1, \ldots, d'$ (although here we leave $\tilde{\mathbf{x}}$ with dimension $d$). Define the reconstruction error summed over the dataset as $\sum_{i=1}^m \|\mathbf{x}_i - \tilde{\mathbf{x}}_i\|^2$. Again assuming that the eigenvectors $\{\mathbf{e}_a\}$ of the covariance matrix are indexed in order of non-increasing eigenvalues, then choosing those eigenvectors as basis vectors will give minimal reconstruction error, as we will show. If the data is not centered, then the mean should be subtracted first, the dimension reduction performed, and the mean then added back[^pca3]; thus in this case, the dimensionally reduced data will still lie in the subspace $\mathcal{R}^{d'}$, but that subspace will be offset from the origin by the mean. Bearing this caveat in mind, to prove the claim we can assume that the data is centered. Expanding $\mathbf{u}_a \equiv \sum_{p=1}^d \beta_{ap} \mathbf{e}_p$, we have

$$\frac{1}{m} \sum_i \|\mathbf{x}_i - \tilde{\mathbf{x}}_i\|^2 = \frac{1}{m} \sum_i \|\mathbf{x}_i\|^2 - \frac{1}{m} \sum_{a=1}^{d'} \sum_i (\mathbf{x}_i \cdot \mathbf{u}_a)^2 \tag{2.4}$$

with orthogonality constraints $\sum_{p=1}^d \beta_{ap}\beta_{bp} = \delta_{ab}$. The second term on the right is

$$-\sum_{a=1}^{d'} \mathbf{u}_a^T C \mathbf{u}_a = -\sum_{a=1}^{d'} \left(\sum_{p=1}^d \beta_{ap} \mathbf{e}_p^T\right) C \left(\sum_{q=1}^d \beta_{aq} \mathbf{e}_q\right) = -\sum_{a=1}^{d'} \sum_{p=1}^d \lambda_p \beta_{ap}^2 \tag{2.5}$$

[^pca3]: The principal eigenvectors are not necessarily the directions that give minimal reconstruction error if the data is not centered: imagine data whose mean is both orthogonal to the principal eigenvector and far from the origin. The single direction that gives minimal reconstruction error will be close to the mean.

Introducing Lagrange multipliers $\omega_{ab}$ to enforce the orthogonality constraints [Burges, 2004], in order to minimize the reconstruction error we must maximize

$$F = \sum_{a=1}^{d'} \sum_{p=1}^d \lambda_p \beta_{ap}^2 - \sum_{a,b=1}^{d'} \omega_{ab} \left(\sum_{p=1}^d \beta_{ap}\beta_{bp} - \delta_{ab}\right) \tag{2.6}$$

Choosing[^pca4] $\omega_{ab} \equiv \omega_a \delta_{ab}$ and taking derivatives with respect to $\beta_{cq}$ gives $\lambda_q \beta_{cq} = \omega_c \beta_{cq}$. Both this and the constraints can be satisfied by choosing $\omega_a = \lambda_a$ and $\beta_{ap} = \delta_{ap}$ for $p \leq d'$, $\beta_{ap} = 0$ otherwise. The objective function then simply becomes $\sum_{p=1}^{d'} \lambda_p$, which is maximized by choosing the first $d'$ largest $\lambda_p$. Note that this also amounts to a proof that, for projections that give minimal reconstruction error, the 'greedy' approach to PCA dimension reduction — solve for a single optimal direction (which gives the principal eigenvector as first basis vector), then project your data into the subspace orthogonal to that, then repeat — also results in the global optimal solution, found by solving for all directions at once. The same observation applies to finding projections that maximally reduce the residual variance. Again, note that this argument is distribution independent.

[^pca4]: Recall that Lagrange multipliers can be chosen in any way that results in a solution satisfying the constraints.

#### 2.1.5 PCA Maximizes Mutual Information on Gaussian Data

Now consider some proposed set of projections $W \in M_{d'd}$, where the rows of $W$ are orthonormal, so that the projected data is $\mathbf{y} \equiv W\mathbf{x}$, $\mathbf{y} \in \mathcal{R}^{d'}$, $\mathbf{x} \in \mathcal{R}^d$, $d' \leq d$. Suppose that $X \sim \mathcal{N}(0, C)$. Then since the $\mathbf{y}$'s are linear combinations of the $\mathbf{x}$'s, they are also normally distributed, with zero mean and sample covariance $C_y \equiv (1/m) \sum_i \mathbf{y}_i \mathbf{y}_i' = (1/m) W(\sum_i \mathbf{x}_i \mathbf{x}_i')W' = WCW'$. It's interesting to ask how $W$ can be chosen so that the mutual information between the distribution of $X$ and that of $Y$ is maximized [Baldi and Hornik, 1995, Diamantaras and Kung, 1996]. Since the mapping $W$ is deterministic, the conditional entropy $H(Y|X)$ vanishes, and the mutual information is just $I(X, Y) = H(Y) - H(Y|X) = H(Y)$. Using a small, fixed bin size, we can approximate this by the differential entropy,

$$H(Y) = -\int p(\mathbf{y}) \log_2 p(\mathbf{y}) d\mathbf{y} = \frac{1}{2} \log_2(e(2\pi)^{d'}) + \frac{1}{2} \log_2 \det(C_y) \tag{2.7}$$

This is maximized by maximizing $\det(C_y) = \det(WCW')$ over choice of $W$, subject to the constraint that the rows of $W$ are orthonormal. The general solution to this is $W = UE$, where $U$ is an arbitrary $d'$ by $d'$ orthogonal matrix, and where the rows of $E \in M_{d'd}$ are formed from the first $d'$ principal eigenvectors of $C$, and at the solution, $\det(C_y)$ is just the product of the first $d'$ principal eigenvalues. Clearly, the choice of $U$ does not affect the entropy, since $\det(UECE'U') = \det(U)\det(ECE')\det(U') = \det(ECE')$. In the special case where $d' = 1$, so that $E$ consists of a single, unit length vector $\mathbf{e}$, we have $\det(ECE') = \mathbf{e}'C\mathbf{e}$, which is maximized by choosing $\mathbf{e}$ to be the principal eigenvector of $C$, as shown above. We refer the reader to [Wilks, 1962] for a proof for the general case $1 < d' < d$.

---

### 2.2 Probabilistic PCA (PPCA)

Suppose you've applied PCA to obtain low dimensional feature vectors for your data, but that you have also somehow found a partition of the data such that the PCA projections you obtain on each subset are quite different from those obtained on the other subsets. It would be tempting to perform PCA on each subset and use the relevant projections on new data, but how do you determine what is 'relevant', and how in general would you even find such subsets? These problems could be addressed if we could learn a mixture of generative models for the data, where each model corresponded to its own PCA decomposition. [Tipping and Bishop, 1999A, Tipping and Bishop, 1999B] proposed such a model — "Probabilistic PCA" — building on earlier work linking PCA decomposition to factor analysis. The advantages of a probabilistic model are numerous: for example, the weight that each mixture component gives to the posterior probability of a given data point can be computed, solving the 'relevance' problem stated above. In this section we briefly review PPCA.

The approach is in fact a form of factor analysis, which itself is a classical dimension reduction technique. Factor analysis first appeared in the behavioral sciences community over a century ago, when Spearman hypothesised that intelligence could be reduced to a single underlying factor [Spearman, 1904]. If, given an $n$ by $n$ correlation matrix between variables $X_i \in \mathbb{R}$, $i = 1, \cdots, n$, there is a single variable $g$ such that the conditional correlation between $X_i$ and $X_j$ vanishes for $i \neq j$ given the value of $g$, then $g$ is the underlying 'factor' and the off-diagonal elements of the correlation matrix can be written as the corresponding off-diagonal elements of $\mathbf{z}\mathbf{z}'$ for some $\mathbf{z} \in \mathbb{R}^n$ [Darlington, 1997]. Modern factor analysis usually considers a model where the underlying factors $X \in \mathcal{R}^{d'}$ are Gaussian, and where a Gaussian noise term $\boldsymbol{\epsilon} \in \mathcal{R}^d$ is added:

$$Y = WX + \boldsymbol{\mu} + \boldsymbol{\epsilon} \tag{2.8}$$
$$X \sim \mathcal{N}(0, \mathbf{1})$$
$$\boldsymbol{\epsilon} \sim \mathcal{N}(0, \Psi)$$

Here $Y \in \mathcal{R}^d$ are the observations, the parameters of the model are $W \in M_{dd'}$ ($d' \leq d$), $\Psi$ and $\boldsymbol{\mu}$, and $\Psi$ is assumed to be diagonal. By construction, $Y$ has mean $\boldsymbol{\mu}$ and 'model covariance' $WW' + \Psi$. For this model, given $X$, the vectors $Y - \boldsymbol{\mu}$ become uncorrelated, and $\epsilon_i$ captures the variance that is unique to $Y_i$. Since $X$ and $\boldsymbol{\epsilon}$ are Gaussian distributed, so is $Y$, and so the maximum likelihood estimate of $\boldsymbol{\mu}$ is just the empirical expectation of the $\mathbf{y}$'s. However, in general, $W$ and $\Psi$ must be estimated iteratively, using for example EM. There is an instructive exception to this [Basilevsky, 1994]. Suppose that $\Psi = \sigma^2 \mathbf{1}$, so that the $d - d'$ smallest eigenvalues of the model covariance are the same and are equal to $\sigma^2$. Suppose also that $S$, the sample covariance of the $\mathbf{y}$'s, is equal to the model covariance; we can then read off $d'$ as the multiplicity of the smallest eigenvalue $\sigma^2$ of $S$. Let $\mathbf{e}^{(j)}$ be the $j$'th orthonormal eigenvector of $S$ with eigenvalue $\lambda_j$. Then it is straightforward to check that $W_{ij} = \sqrt{(\lambda_j - \sigma^2)} e_i^{(j)}$, $i = 1, \cdots, d$, $j = 1, \cdots, d'$ satisfies $WW' + \Psi = S$ if the $\mathbf{e}^{(j)}$ are in principal order. The model thus arrives at the PCA directions, but in a probabilistic way. *Probabilistic* PCA (PPCA) assumes a model of the form (2.8) with $\Psi = \sigma^2 \mathbf{1}$, but it drops the above assumption that the model and sample covariances are equal (which in turn means that $\sigma^2$ must now be estimated). The resulting maximum likelihood estimates of $W$ and $\sigma^2$ can be written in closed form, as [Tipping and Bishop, 1999A]

$$W_{ML} = U(\Lambda - \sigma^2 \mathbf{1})R \tag{2.9}$$

$$\sigma^2_{ML} = \frac{1}{d - d'} \sum_{i=d'+1}^{d} \lambda_i \tag{2.10}$$

where $U \in M_{dd'}$ is the matrix of the $d'$ principal column eigenvectors of $S$, $\Lambda$ is the corresponding diagonal matrix of principal eigenvalues, and $R \in M_{d'}$ is an arbitrary orthogonal matrix. Thus $\sigma^2$ captures the variance lost in the discarded projections and the PCA directions appear in the maximum likelihood estimate of $W$ (and in fact re-appear in the expression for the expectation of $X$ given $Y$, in the limit $\sigma \to 0$, in which case the components of $X$ become the PCA projections of $Y$). This closed form result is rather striking in view of the fact that for general factor analysis (for example, for diagonal but non-isotropic $\Psi$) we must resort to an iterative algorithm. The probabilistic formulation makes PCA amenable to a rich variety of probabilistic methods: for example, PPCA allows one to perform PCA when some of the data has missing components; and $d'$ (which so far we've assumed known) can itself be estimated using Bayesian arguments [Bishop, 1999]. Returning to the problem posed at the beginning of this Section, a mixture of PPCA models, each with weight $\pi_i \geq 0$, $\sum_i \pi_i = 1$, can be computed for the data using maximum likelihood and EM, thus giving a principled approach to combining several local PCA models [Tipping and Bishop, 1999B].

---

### 2.3 Kernel PCA

PCA is a linear method, in the sense that the reduced dimension representation is generated by linear projections (although the eigenvectors and eigenvalues depend non-linearly on the data), and this can severely limit the usefulness of the approach. Several versions of nonlinear PCA have been proposed (see e.g. [Diamantaras and Kung, 1996]) in the hope of overcoming this problem. In this section we describe one such algorithm called kernel PCA [Schölkopf et al., 1998]. Kernel PCA relies on the "kernel trick", the essence of which rests on the following observation: suppose you have an algorithm (for example, $k$'th nearest neighbour) which depends only on dot products of the data. Consider using the same algorithm on transformed data: $\mathbf{x} \to \Phi(\mathbf{x}) \in \mathcal{F}$, where $\mathcal{F}$ is a (possibly infinite dimensional) vector space, which we will call feature space. Operating in $\mathcal{F}$, your algorithm depends only on the dot products $\Phi(\mathbf{x}_i) \cdot \Phi(\mathbf{x}_j)$. Now suppose there exists a (symmetric) 'kernel' function $k(\mathbf{x}_i, \mathbf{x}_j)$ such that for all $\mathbf{x}_i, \mathbf{x}_j \in \mathcal{R}^d$, $k(\mathbf{x}_i, \mathbf{x}_j) = \Phi(\mathbf{x}_i) \cdot \Phi(\mathbf{x}_j)$. Then since your algorithm depends only on these dot products, you never have to compute $\Phi(\mathbf{x})$ explicitly; you can always just substitute the kernel form.

In fact this 'trick' is very general, and since it is widely used, we summarize it briefly here. Consider a Hilbert space $\mathcal{H}$ (a complete vector space for which an inner product is defined). We will take $\mathcal{H}$ to be a space whose elements are real valued functions defined over $\mathcal{R}^d$, for concreteness. Consider the set of linear evaluation functionals $I_x: f \in \mathcal{H} \to f(x) \in \mathbb{R}$, indexed by $x \in \mathcal{R}^d$. If every such linear functional is continuous, then there is a special function $k_x$ associated with $\mathcal{H}$, also indexed by $x$, and called a reproducing kernel, for which $\langle f, k_x \rangle = f(x)$. Such Hilbert spaces are called Reproducing Kernel Hilbert Spaces (RKHSs) and this particular relation is called the reproducing property. In particular, the function $k_{x_1}$ evaluated at some other point $x_2$ is defined as $k(x_1, x_2) \equiv k_{x_1}(x_2)$, and using the reproducing property on $k_x$ itself yields $\langle k_{x_1}, k_{x_2} \rangle = k(x_1, x_2)$. From this follow the usual properties we associate with kernels — they are symmetric in their arguments, and are positive definite functions.

Kernel PCA applies the kernel trick to create a nonlinear version of PCA in sample space by performing ordinary PCA in $\mathcal{F}$. It's striking that, since projections are being performed in a space whose dimension can be much larger than $d$, the number of useful such projections can actually exceed $d$. [Schölkopf et al., 1998] show how the problem can indeed be cast entirely in terms of inner products. They make two key observations: first, that the eigenvectors of the covariance matrix in $\mathcal{F}$ lie in the span of the (centered) mapped data, and second, that therefore no information in the eigenvalue equation is lost if the equation is replaced by $m$ equations, formed by taking the inner product of each side of the eigenvalue equation with each (centered) mapped data point. The covariance matrix of the mapped data in feature space is

$$C \equiv \frac{1}{m} \sum_{i=1}^m (\Phi_i - \boldsymbol{\mu})(\Phi_i - \boldsymbol{\mu})^T \tag{2.11}$$

where $\Phi_i \equiv \Phi(\mathbf{x}_i)$ and $\boldsymbol{\mu} \equiv \frac{1}{m} \sum_i \Phi_i$. We are looking for eigenvector solutions $\mathbf{v}$ of

$$C\mathbf{v} = \lambda\mathbf{v} \tag{2.12}$$

Since this can be written $\frac{1}{m} \sum_{i=1}^m (\Phi_i - \boldsymbol{\mu})[(\Phi_i - \boldsymbol{\mu}) \cdot \mathbf{v}] = \lambda\mathbf{v}$, the eigenvectors $\mathbf{v}$ lie in the span of the $\Phi_i - \boldsymbol{\mu}$'s, or

$$\mathbf{v} = \sum_i \alpha_i (\Phi_i - \boldsymbol{\mu}) \tag{2.13}$$

for some $\alpha_i$. We will denote the vector whose $i$th component is $\alpha_i$ by $\boldsymbol{\alpha} \in \mathcal{R}^m$. We can replace Eq. (2.12) with the $m$ equations

$$(\Phi_i - \boldsymbol{\mu})^T C\mathbf{v} = \lambda(\Phi_i - \boldsymbol{\mu})^T \mathbf{v} \tag{2.14}$$

We can easily compute the kernel matrix $K_{ij}$, the matrix of inner products[^gram] in $\mathcal{F}$: $K_{ij} \equiv \langle \Phi_i, \Phi_j \rangle = k(\mathbf{x}_i, \mathbf{x}_j)$, $i, j = 1, \ldots, m$. We need the centered kernel matrix, $K^C_{ij} \equiv \langle(\Phi_i - \boldsymbol{\mu}), (\Phi_j - \boldsymbol{\mu})\rangle$. Note that we do not need to compute $\boldsymbol{\mu}$ explicitly since any $m$ by $m$ inner product matrix can be centered by left- and right-multiplying by the projection matrix $P \equiv \mathbf{1} - \frac{1}{m}\mathbf{e}\mathbf{e}'$, where $\mathbf{1}$ is the unit matrix in $M_m$ and where $\mathbf{e}$ is the $m$-vector of all ones (see Section 3.2 for further discussion of centering). Thus $K^C = PKP$ and Eq. (2.14) becomes

[^gram]: A matrix of inner products is called a Gram matrix. Any Gram matrix $G$ is necessarily positive semidefinite, as is easily seen from $\mathbf{z}'K\mathbf{z} = \sum_{ij} z_i z_j \langle \Phi_i, \Phi_j \rangle = \|\sum_i z_i \Phi_i\|^2$.

$$K^C K^C \boldsymbol{\alpha} = \nu K^C \boldsymbol{\alpha} \tag{2.15}$$

where $\nu \equiv m\lambda$. Now clearly any solution of

$$K^C \boldsymbol{\alpha} = \nu \boldsymbol{\alpha} \tag{2.16}$$

is also a solution of (2.15). Finally, to use the eigenvectors $\mathbf{v}$ to compute principal components in $\mathcal{F}$, we need $\mathbf{v}$ to have unit length, that is, $\mathbf{v} \cdot \mathbf{v} = 1 = \nu \boldsymbol{\alpha} \cdot \boldsymbol{\alpha}$ (using (2.13) and (2.16)), so the $\boldsymbol{\alpha}$ must be normalized to have length $1/\sqrt{\nu}$.

The recipe for extracting the $i$'th principal component in $\mathcal{F}$ using kernel PCA is therefore:

1. Compute the $i$'th principal eigenvector of $K^C$, with eigenvalue $\nu$.
2. Normalize the corresponding eigenvector, $\boldsymbol{\alpha}$, to have length $1/\sqrt{\nu}$.
3. For a training point $\mathbf{x}_k$, the principal component is then just $(\Phi(\mathbf{x}_k) - \boldsymbol{\mu}) \cdot \mathbf{v} = \nu\alpha_k \tag{2.17}$
4. For a general test point $\mathbf{x}$, the principal component is $(\Phi(\mathbf{x}) - \boldsymbol{\mu}) \cdot \mathbf{v} = \sum_i \alpha_i k(\mathbf{x}, \mathbf{x}_i) - \frac{1}{m}\sum_{i,j} \alpha_i k(\mathbf{x}, \mathbf{x}_j) - \frac{1}{m}\sum_{i,j} \alpha_i k(\mathbf{x}_i, \mathbf{x}_j) + \frac{1}{m^2}\sum_{i,j,n} \alpha_i k(\mathbf{x}_j, \mathbf{x}_n)$ where the last two terms can be dropped since they don't depend on $\mathbf{x}$.

Kernel PCA may be viewed as a way of putting more effort into the up-front computation of features, rather than putting the onus on the classifier or regression algorithm. Kernel PCA followed by a linear SVM on a pattern recognition problem has been shown to give similar results to using a nonlinear SVM using the same kernel [Schölkopf et al., 1998]. Classical PCA has the significant limitation that it depends only on first and second moments of the data, whereas kernel PCA does not (for example, a polynomial kernel $k(\mathbf{x}_i, \mathbf{x}_j) = (\mathbf{x}_i \cdot \mathbf{x}_j + b)^p$ contains powers up to order $2p$, which is particularly useful for e.g. image classification, where one expects that products of several pixel values will be informative as to the class). Kernel PCA has the computational limitation of having to compute eigenvectors for square matrices of side $m$, but again this can be addressed, for example by using a subset of the training data, or by using the Nyström method for approximating the eigenvectors of a large Gram matrix (see below).

---

### 2.4 Canonical Correlation Analysis

Suppose we have two datasets $\mathbf{x}_{1i} \in S_1$, $\mathbf{x}_{2j} \in S_2$, where $S_1 \equiv \mathcal{R}^{d_1}$, $S_2 \equiv \mathcal{R}^{d_2}$, and $i = 1, \ldots, m_1$, $j = 1, \ldots, m_2$. Note that $d_1$ may not equal $d_2$ and that $m_1$ may not equal $m_2$. Canonical Correlation Analysis (CCA) [Hotelling, 1936] finds paired directions $\{\mathbf{w}_1, \mathbf{w}_2\}$, $\mathbf{w}_1 \in S_1$, $\mathbf{w}_2 \in S_2$ such that the projection of the first dataset along $\mathbf{w}_1$ is maximally correlated with the projection of the second dataset along $\mathbf{w}_2$. In addition, for $i \neq j$, the projections along the pairs $\{\mathbf{w}_{1i}, \mathbf{w}_{1j}\}$, $\{\mathbf{w}_{2i}, \mathbf{w}_{2j}\}$ and $\{\mathbf{w}_{1i}, \mathbf{w}_{2j}\}$ are all uncorrelated. Furthermore, the values of the $\mathbf{w} \cdot \mathbf{x}$'s themselves are invariant to invertible affine transformations of the data, which gives CCA a coordinate independent meaning, in contrast to ordinary correlation analysis.

CCA may be summarized as follows. We are given two random variables, $\mathbf{X}_1$, $\mathbf{X}_2$ with sample spaces $\Omega_1 \in \mathcal{R}^{d_1}$ and $\Omega_2 \in \mathcal{R}^{d_2}$. Introduce random variables $U \equiv \mathbf{X}_1 \cdot \mathbf{w}_1$ and $V \equiv \mathbf{X}_2 \cdot \mathbf{w}_2$. We wish to find $\mathbf{w}_1 \in \mathcal{R}^{d_1}$, $\mathbf{w}_2 \in \mathcal{R}^{d_2}$, such that the correlation

$$\rho \equiv \frac{E[UV]}{\sqrt{E[U^2]E[V^2]}} = \frac{\mathbf{w}_1' C_{12} \mathbf{w}_2}{\sqrt{(\mathbf{w}_1' C_{11} \mathbf{w}_1)(\mathbf{w}_2' C_{22} \mathbf{w}_2)}} \equiv \frac{A_{12}}{\sqrt{A_{11} A_{22}}} \tag{2.18}$$

is maximized, where $C_{pq} \equiv E[\mathbf{X}_p \mathbf{X}_q']$ and we have introduced scalars $A_{pq} \equiv \mathbf{w}_p' C_{pq} \mathbf{w}_q$. Setting the derivative of $\rho^2$ with respect to $w_{pa}$ equal to zero for $p = \{1, 2\}$ gives

$$C_{11}^{-1} C_{12} \mathbf{w}_2 = \frac{A_{12}}{A_{11}} \mathbf{w}_1 \tag{2.19}$$

$$C_{11}^{-1} C_{12} C_{22}^{-1} C_{21} \mathbf{w}_1 = \rho^2 \mathbf{w}_1 \tag{2.20}$$

#### 2.4.1 CCA Decorrelates the Data

CCA shares with PCA the property that the projections decorrelate the data. For CCA, the data is decorrelated both within $S_1$ and $S_2$ and between $S_1$ and $S_2$, and the directions are conjugate with respect to the covariance matrices. Thus in the new basis, the variables are uncorrelated:

$$E[U_i U_j'] = E[\mathbf{w}_{1i} \cdot \mathbf{X}_1 \mathbf{w}_{1j} \cdot \mathbf{X}_1] = \mathbf{w}_{1i}' C_{11} \mathbf{w}_{1j} = 0 \text{ for } i \neq j \tag{2.25}$$

and similarly $E[V_i V_j'] = E[U_i V_j'] = 0$ if $i \neq j$.

#### 2.4.2 CCA is Invariant under Invertible Affine Transformations

One of the strengths of CCA is that the projected values are invariant under invertible affine transformations $\mathbf{x} \in \mathcal{R}^d \to B\mathbf{x} + \mathbf{b}$, $B \in M_d$, $\mathbf{b} \in \mathcal{R}^d$, provided the $\mathbf{w}$'s are appropriately transformed. Invariance with respect to translations follows directly from the definition of $\rho$, since covariance matrices are functions of the centered data. Note that the above two properties — decorrelation and affine invariance — are not shared by ordinary correlation analysis.

#### 2.4.3 CCA in Practice; Kernel CCA

In practice CCA is applied to paired datasets, $\mathbf{x}_{1i}$, $\mathbf{x}_{2j}$, $i, j = 1, \ldots, m$. Kernel CCA follows kernel PCA in spirit. The data $\mathbf{x}_1 \in \mathcal{R}^{d_1}$, $\mathbf{x}_2 \in \mathcal{R}^{d_2}$ are mapped to feature spaces $\mathcal{F}_1$ and $\mathcal{F}_2$ by maps $\Phi_1$, $\Phi_2$ respectively. Following the analysis in the spaces $\mathcal{F}_p$ yields

$$\rho = \max_{\boldsymbol{\alpha}_1, \boldsymbol{\alpha}_2} \frac{\boldsymbol{\alpha}_1' K_1 K_2 \boldsymbol{\alpha}_2}{\sqrt{\boldsymbol{\alpha}_1' K_1^2 \boldsymbol{\alpha}_1 \boldsymbol{\alpha}_2' K_2^2 \boldsymbol{\alpha}_2}} \tag{2.30}$$

where each $K_p$ is a square matrix of side $m$. We refer the reader to [Hardoon, Szedmak and Shawe-Taylor, 2004] for details.

---

### 2.5 Oriented PCA and Distortion Discriminant Analysis

Before leaving projective methods, we describe another extension of PCA, which has proven very effective at extracting robust features from audio [Burges et al., 2002, Burges et al., 2003]. We first describe the method of oriented PCA (OPCA) [Diamantaras and Kung, 1996]. Suppose we are given a set of 'signal' vectors $\mathbf{x}_i \in \mathcal{R}^d$, $i = 1, \ldots, m$, where each $\mathbf{x}_i$ represents an undistorted data point, and suppose that for each $\mathbf{x}_i$, we have a set of $N$ distorted versions $\tilde{\mathbf{x}}_i^k$, $k = 1, \ldots, N$. Define the corresponding 'noise' difference vectors to be $\mathbf{z}_i^k \equiv \tilde{\mathbf{x}}_i^k - \mathbf{x}_i$. Roughly speaking, we wish to find linear projections which are as orthogonal as possible to the difference vectors, but along which the variance of the signal data is simultaneously maximized.

The first OPCA direction is defined as that direction $\mathbf{n}$ that maximizes the generalized Rayleigh quotient $q_0 = \frac{\mathbf{n}' C_1 \mathbf{n}}{\mathbf{n}' C_2 \mathbf{n}}$, where $C_1$ is the covariance matrix of the signal and $C_2$ that of the noise. For $d'$ directions collected into a column matrix $\mathcal{N} \in M_{dd'}$, we instead maximize $\frac{\det(\mathcal{N}' C_1 \mathcal{N})}{\det(\mathcal{N}' C_2 \mathcal{N})}$. Explicitly:

$$C \equiv \frac{1}{m} \sum_i (\mathbf{x}_i - E[\mathbf{x}])(\mathbf{x}_i - E[\mathbf{x}])' \tag{2.31}$$

$$R \equiv \frac{1}{mN} \sum_{i,k} \mathbf{z}_i^k (\mathbf{z}_i^k)' \tag{2.32}$$

and maximize $q = \frac{\mathbf{n}' C \mathbf{n}}{\mathbf{n}' R \mathbf{n}}$. Setting $\nabla q = 0$ gives the generalized eigenvalue problem $C\mathbf{n} = qR\mathbf{n}$.

'Distortion discriminant analysis' [Burges et al., 2002, Burges et al., 2003] uses layers of OPCA projectors both to reduce dimensionality and to make the features more robust.

---

### 2.6 Sufficient Dimension Reduction

This section explores dimension reduction techniques where the data consists of predictor-response pairs $\{(\mathbf{x}_i, y_i)\}$, $i = 1, \ldots, m$. We consider models of the form

$$y = f(\mathbf{a}_1'\mathbf{x}, \mathbf{a}_2'\mathbf{x}, \ldots, \mathbf{a}_k'\mathbf{x}, \epsilon), \quad \mathbf{a}_i, \mathbf{x} \in \mathcal{R}^d, \epsilon \in \mathcal{R} \tag{2.33}$$

where the $\epsilon$'s model the noise and are assumed independent of $X$. An alternative way of writing (2.33) is

$$Y \perp\!\!\!\perp X \mid \mathbf{a}^T X \tag{2.34}$$

Following [Cook, 1998], who defines a *minimum dimension-reduction subspace* (minimum DRS) as a space $S_{\mathbf{a}}$ satisfying Eq. (2.34) for which $k$ is minimal. A *central subspace* is defined as the intersection of all DRS's, if that intersection is itself a DRS. The goal of Sufficient Dimension Reduction[^sdr] is to estimate the central subspace, when it exists.

[^sdr]: The phrase *Sufficient Dimension Reduction* was introduced to the statistics community by [Cook and Lee, 1999]. The phrase *Sufficient Dimensionality Reduction* was introduced to the machine learning community by [Globerson and Tishby, 2003]. The approaches are quite different; we briefly summarize the latter below.

#### 2.6.1 Sliced Inverse Regression

Sliced Inverse Regression (SIR) was introduced in a seminal paper by [Li, 1991a]. Normal (forward) regression estimates $E[Y|\mathbf{x}]$. Inverse regression instead estimates $E[X|y]$, which is a much easier problem since it amounts to solving $d$ one dimensional regression problems.

**Theorem 2.1.** *Given Eq. (2.33), further assume that $E[X|\mathbf{a}_1'\mathbf{x}, \mathbf{a}_2'\mathbf{x}, \ldots, \mathbf{a}_k'\mathbf{x}]$ lies in the subspace spanned by $\Sigma_{XX}\mathbf{a}_i$, where $\Sigma_{XX}$ is the covariance matrix of $X$. Then the centered inverse regression curve $E[X|y] - E[X]$ lies in that subspace.*

The SIR algorithm consists of grouping the measured $\mathbf{x}_i$ by their corresponding values of $y$ (binned if necessary), computing the mean for each group, and performing a weighted PCA on the resulting set of vectors in order to estimate $S_{\mathbf{a}}$.

#### 2.6.2 Kernel Dimension Reduction

Kernel dimension reduction (KDR) [Fukumizu, Bach and Jordan, 2009] addresses limitations of SIR by working directly with the defining condition for Sufficient Dimension Reduction: $Y \perp\!\!\!\perp X|\alpha^T \mathbf{x}$. Associate with the random variables $X$ and $Y$, Reproducing Kernel Hilbert Spaces (RKHSs), $H_X$ and $H_Y$. A "cross covariance" operator $\Sigma_{YX}: H_X \to H_Y$ can be defined so that

$$\langle g, \Sigma_{YX} f \rangle = E_{XY}[(f(X) - E_X[f(X)])(g(Y) - E_Y[g(Y)])] \tag{2.41}$$

A conditional covariance operator $\Sigma_{YY|X} \equiv \Sigma_{YY} - \Sigma_{YX}\Sigma_{XX}^{-1}\Sigma_{XY}$ is then defined. The authors write a sample version of the objective function as

$$\text{Tr}[G_Y (G_X^B + m\epsilon I_m)^{-1}] \quad \text{subject to } B^T B = 1 \tag{2.42}$$

where $m$ is the sample size and $\epsilon$ a regularization parameter. $B$ is then found using gradient descent.

#### 2.6.3 Sufficient Dimensionality Reduction

Here we briefly describe Sufficient Dimensionality Reduction (SDR'), a similarly named but quite different technique [Globerson and Tishby, 2003]. SDR' is not a supervised method. Rather than searching for a subspace that satisfies Eq. (2.34), SDR' models the density $p(X)$, parameterized by $y$, using two-way contingency tables. The key idea of SDR' is to identify feature mappings $\phi(x)$ such that the $y$'s can be described by a small set of such features.

---

## 3 Manifold Modeling

In Chapter 2 we gave an example of data with a particular geometric structure which would not be immediately revealed by examining one dimensional projections in input space. How, then, can such underlying structure be found? This section outlines some methods designed to accomplish this. However we first describe the Nyström method, which provides a thread linking several of the algorithms we describe.

---

### 3.1 The Nyström method

Suppose that $K \in M_n$ and that the rank of $K$ is $r \ll n$. Nyström gives a way of approximating the eigenvectors and eigenvalues of $K$ using those of a small submatrix $A$. If $A$ has rank $r$, then the decomposition is exact. This is a powerful method that can be used to speed up kernel algorithms [Williams and Seeger, 2001], to efficiently extend some algorithms to out-of-sample (test) points [Bengio et al., 2004], and in some cases, to make an otherwise infeasible algorithm feasible [Fowlkes et al., 2004].

#### 3.1.1 Original Nyström

The Nyström method originated as a method for approximating the solution of Fredholm integral equations of the second kind [Press et al., 1992]. The homogeneous $d$-dimensional form with density $p(\mathbf{x})$, $\mathbf{x} \in \mathcal{R}^d$ has the form

$$\int k(\mathbf{x}, \mathbf{y}) u(\mathbf{y}) p(\mathbf{y}) d\mathbf{y} = \lambda u(\mathbf{x}) \tag{3.1}$$

The integral is approximated using the quadrature rule

$$\lambda u(\mathbf{x}) \approx \frac{1}{m} \sum_{i=1}^m k(\mathbf{x}, \mathbf{x}_i) u(\mathbf{x}_i) \tag{3.2}$$

which when applied to the sample points becomes a matrix equation $K_{mm} \mathbf{u}_m = m\lambda\mathbf{u}_m$.

#### 3.1.2 Exact Nyström Eigendecomposition

Suppose that a kernel matrix $\tilde{K}_{mm}$ has rank $r < m$. Writing the matrix $K$ as

$$K_{mm} \equiv \begin{bmatrix} A_{rr} & B_{rn} \\ B'_{nr} & C_{nn} \end{bmatrix} \tag{3.3}$$

Since $A$ is of full rank and $K$ is of rank $r$, one can show that $C = B'A^{-1}B$. Thus $K$ must be of the form

$$K_{mm} = \begin{bmatrix} A & B \\ B' & B'A^{-1}B \end{bmatrix} = \begin{bmatrix} A \\ B' \end{bmatrix}_{mr} A^{-1}_{rr} \begin{bmatrix} A & B \end{bmatrix}_{rm} \tag{3.5}$$

Using the eigendecomposition of $A$, $A = U\Lambda U'$, one can show that the exact eigendecomposition of $K_{mm}$ for its nonvanishing eigenvalues is given by the orthogonal column eigenvectors:

$$V_{mr} \equiv \begin{bmatrix} A \\ B' \end{bmatrix} A^{-1/2} U_Q \Lambda_Q^{-1/2} \tag{3.7}$$

so that $K_{mm} = V\Lambda_Q V'$ and $V'V = 1_{rr}$ [Fowlkes et al., 2004].

#### 3.1.3 Approximate Nyström Eigendecomposition

Two kinds of approximation naturally arise. The first occurs if $K$ is only approximately low rank, that is, its spectrum decays rapidly, but not to exactly zero. In this case, $B'A^{-1}B$ will only approximately equal $C$, and the approximation can be quantified as $\|C - B'A^{-1}B\|$ for some matrix norm $\|\cdot\|$, where the difference is known as the Schur complement of $A$ for the matrix $K$ [Golub and Van Loan, 1996].

The second kind of approximation addresses the need to compute the eigendecomposition just once, to speed up test phase. The idea is simply to take Equation (3.2), sum over $d$ elements on the right hand side where $d \ll m$ and $d > r$, and approximate the eigenvector of the full kernel matrix $K_{mm}$ by evaluating the left hand side at all $m$ points [Williams and Seeger, 2001].

---

### 3.2 Multidimensional Scaling

MDS starts with a measure of dissimilarity between each pair of data points in the dataset. Given this, MDS searches for a mapping of the (possibly further transformed) dissimilarities to a low dimensional Euclidean space such that the (transformed) pair-wise dissimilarities become squared distances.

We start with the fundamental theorem upon which 'classical MDS' is built. Let $\mathbf{e}$ be the column vector of $m$ ones. Consider the 'centering' matrix $P^e \equiv \mathbf{1} - \frac{1}{m}\mathbf{e}\mathbf{e}'$.

**Theorem 3.1.** *Consider the class of symmetric matrices $A \in S_n$ such that $A_{ij} \geq 0$ and $A_{ii} = 0$ $\forall i, j$. Then $\bar{A} \equiv -P^e A P^e$ is positive semidefinite if and only if $A$ is a distance matrix (with embedding space $\mathcal{R}^d$ for some $d$). Given that $A$ is a distance matrix, the minimal embedding dimension $d$ is the rank of $\bar{A}$, and the embedding vectors are any set of Gram vectors of $\bar{A}$, scaled by a factor of $\frac{1}{\sqrt{2}}$.*

The fraction of 'unexplained residuals' is

$$\sum_{i=1}^m \|\hat{\mathbf{y}}_i - \mathbf{y}_i\|^2 = \sum_{a=p+1}^r \lambda_a \tag{3.12}$$

in analogy to the fraction of 'unexplained variance' in PCA.

#### 3.2.3 Landmark MDS

MDS is computationally expensive: since the distances matrix is not sparse, the computational complexity of the eigendecomposition is $O(m^3)$. This can be significantly reduced by using a method called Landmark MDS (LMDS) [Silva and Tenenbaum, 2002]. In LMDS the idea is to choose $q$ points, called 'landmarks', where $q > r$ but $q \ll m$, and to perform MDS on landmarks, mapping them to $\mathcal{R}^d$. The remaining points are then mapped to $\mathcal{R}^d$ using only their distances to the landmark points. Landmark MDS has two significant advantages: first, it reduces the computational complexity from $O(m^3)$ to $O(q^3 + q^2(m - q) = q^2 m)$; and second, it can be applied to any non-landmark point, and so gives a method of extending MDS (using Nyström) to out-of-sample data.

---

### 3.3 Isomap

MDS is valuable for extracting low dimensional representations for some kinds of data, but it does not attempt to explicitly model the underlying manifold. Isomap (isometric feature map) [Tenenbaum, 1998] assumes that the quantity of interest, when comparing two points, is the distance along the manifold between the two points. The basic idea is to construct a graph whose nodes are the data points, where a pair of nodes are adjacent only if the two points are close in $\mathcal{R}^d$, and then to approximate the geodesic distance along the manifold between any two points as the shortest path in the graph, computed using the Floyd algorithm [Gondran and Minoux, 1984]; and finally to use MDS to extract the low dimensional representation.

Isomap shares with the other manifold mapping techniques the property that it does not provide a direct functional form for the mapping $I: \mathcal{R}^d \to \mathcal{R}^{d'}$ that can simply be applied to new data. The eigenvector computation is $O(m^3)$, and the Floyd algorithm also $O(m^3)$. Landmark Isomap simply employs landmark MDS [Silva and Tenenbaum, 2002] to address this problem, reducing the computational complexity to $O(q^2 m)$ for the LMDS step, and to $O(hqm\log m)$ for the shortest path step.

---

### 3.4 Locally Linear Embedding

Locally linear embedding (LLE) [Roweis and Saul, 2000] models the manifold by treating it as a union of linear patches, in analogy to using coordinate charts to parameterize a manifold in differential geometry. Suppose that each point $\mathbf{x}_i \in \mathcal{R}^d$ has a small number of close neighbours indexed by the set $\mathcal{N}(i)$, and let $\mathbf{y}_i \in \mathcal{R}^{d'}$ be the low dimensional representation of $\mathbf{x}_i$. The idea is to express each $\mathbf{x}_i$ as a linear combination of its neighbours, and then construct the $\mathbf{y}_i$ so that they can be expressed as the same linear combination of their corresponding neighbours. The objective function to minimize is

$$F \equiv \sum_i F_i \equiv \sum_i \left(\frac{1}{2}\left\|\mathbf{x}_i - \sum_{j \in \mathcal{N}(i)} W_{ij}\mathbf{x}_j\right\|^2 - \lambda_i\left(\sum_{j \in \mathcal{N}(i)} W_{ij} - 1\right)\right)$$

Given the $W$'s, the second step minimizes $\sum_i \|\mathbf{y}_i - \sum_{j \in \mathcal{N}(i)} W_{ij}\mathbf{y}_j\|^2$ with respect to the $\mathbf{y}$'s, keeping the $W$'s fixed. This leads to the matrix equation

$$(\mathbf{1} - W)'(\mathbf{1} - W)Y = \frac{1}{m} Y\Lambda \tag{3.18}$$

Choosing the columns of $Y$ to be the next $d'$ eigenvectors of $(\mathbf{1} - W)'(\mathbf{1} - W)$ with the smallest eigenvalues gives the low dimensional representations. LLE requires a two-step procedure. The first step (finding the $W$'s) has $O(n^3 m)$ computational complexity; the second requires eigendecomposing the product of two sparse matrices in $M_m$.

---

### 3.5 Graphical Methods

Let's start by defining a simple mapping from a dataset to an undirected graph $G$ by forming a one-to-one correspondence between nodes in the graph and data points. If two nodes $i$, $j$ are connected by an arc, associate with it a positive arc weight $W_{ij}$, where $W_{ij}$ is a similarity measure between points $\mathbf{x}_i$ and $\mathbf{x}_j$. The Laplacian matrix for any weighted, undirected graph is defined [Chung, 1997] by $\mathcal{L} \equiv D^{-1/2}LD^{-1/2}$, where $L_{ij} \equiv D_{ij} - W_{ij}$ and where $D_{ij} \equiv \delta_{ij}(\sum_k W_{ik})$.

For any vector $\mathbf{z} \in \mathcal{R}^m$, since $W_{ij} \geq 0$:

$$0 \leq \frac{1}{2}\sum_{i,j}(z_i - z_j)^2 W_{ij} = \sum_i z_i^2 D_{ii} - \sum_{i,j} z_i W_{ij} z_j = \mathbf{z}' L \mathbf{z}$$

#### 3.5.1 Laplacian Eigenmaps

The Laplacian eigenmaps algorithm [Belkin and Niyogi, 2003] uses $W_{ij} = \exp^{-\|\mathbf{x}_i - \mathbf{x}_j\|^2/2\sigma^2}$. We would like to find $\mathbf{y}$'s that minimize $\sum_{i,j} \|\mathbf{y}_i - \mathbf{y}_j\|^2 W_{ij}$, since then if two points are similar, their $\mathbf{y}$'s will be close. We have:

$$\sum_{i,j} \|\mathbf{y}_i - \mathbf{y}_j\|^2 W_{ij} = 2\text{Tr}(Y'LY) \tag{3.19}$$

Minimizing $\text{Tr}(Y'LY)$ subject to the constraint $Y'DY = \mathbf{1}$ results in the simple generalized eigenvalue problem $L\mathbf{y} = \lambda D\mathbf{y}$ [Belkin and Niyogi, 2003].

#### 3.5.2 Spectral Clustering

Although spectral clustering is a clustering method, it is very closely related to dimension reduction. The solution to the normalized mincut problem is given by [Shi and Malik, 2000]

$$\min_\mathbf{y} \frac{\mathbf{y}' L \mathbf{y}}{\mathbf{y}' D \mathbf{y}} \quad \text{such that } y_i \in \{1, -b\} \text{ and } \mathbf{y}' D\mathbf{e} = 0 \tag{3.20}$$

This problem is solved by relaxing $\mathbf{y}$ to take real values: the problem then becomes finding the second smallest eigenvector of the generalized eigenvalue problem $L\mathbf{y} = \lambda D\mathbf{y}$, which is exactly the same problem found by Laplacian eigenmaps.

---

### 3.6 Pulling the Threads Together

At this point the reader is probably struck by how similar the mathematics underlying all of these approaches is. We've used essentially the same Lagrange multiplier trick to enforce constraints three times; all of the methods in this Chapter (and most in this review) rely heavily on eigendecompositions. Isomap, LLE, Laplacian eigenmaps, and spectral clustering all share the property that in their original forms, they do not provide a direct functional form for the dimension-reducing mapping, so the extension to new data requires re-training. Landmark Isomap solves this problem; the other algorithms could also use Nyström to solve it (as pointed out by [Bengio et al., 2004]).

Isomap is often called a 'global' dimension reduction algorithm, because it attempts to preserve all geodesic distances; by contrast, LLE, spectral clustering and Laplacian eigenmaps are local. Although they start from different geometrical considerations, LLE, Laplacian eigenmaps, spectral clustering and MDS all look quite similar under the hood: the first three use the dual eigenvectors of a symmetric matrix as their low dimensional representation, and MDS uses the dual eigenvectors with components scaled by square roots of eigenvalues.

The connection between MDS and PCA goes further than the form taken by the 'unexplained residuals' in Eq. (3.12). If $X \in M_{md}$ is the matrix of $m$ (zero-mean) sample vectors, then PCA diagonalizes the covariance matrix $X'X$, whereas MDS diagonalizes the kernel matrix $XX'$; but $XX'$ has the same eigenvalues as $X'X$ [Horn and Johnson, 1985], and $m - d$ additional zero eigenvalues (if $m > d$). [Williams, 2001] has pointed out that kernel PCA can itself be viewed as performing MDS in feature space.

---

## 4 Conclusion

Dimension reduction has benefited from a great deal of work in both the statistics and machine learning communities. In this review I've limited the scope largely to geometric methods, so I'd like to alert the reader to three other interesting leads. The first is the method of principal curves, where the idea is to find that smooth curve that passes through the data in such a way that the sum of shortest distances from each point to the curve is minimized [Hastie and Stuetzle, 1989]. Second, the Information Bottleneck method [Tishby et al., 1999], which aims to distill the information in a random (predictor) variable $X$ that is needed to describe a (response) variable $Y$, using a model variable $Z$, maximizes the (weighted) difference in mutual information between $Y$ and $Z$, and between $X$ and $Z$. Finally, we note that the subject of feature selection, where one wants to select an optimal subset of features, is a restricted form of dimension reduction. Along those lines, [Viola and Jones, 2001] showed that boosted stump models can be very effective for finding a small set of good features from a large set of possible features. For more on feature selection, we refer the reader to [Guyon, 2003].

---

## Acknowledgements

Thanks to Michael Jordan, John Platt, and Dennis Cook, for valuable comments and suggestions.

---

## References

- [Aizerman et al., 1964] M.A. Aizerman, E.M. Braverman, and L.I. Rozoner. Theoretical foundations of the potential function method in pattern recognition learning. *Automation and Remote Control*, 25:821–837, 1964.
- [Bach and Jordan, 2002] F. R. Bach and M. I. Jordan. Kernel Independent Component Analysis. *Journal of machine learning research*, 3:1–48, 2002.
- [Baldi and Hornik, 1995] P.F. Baldi and K. Hornik. Learning in linear neural networks: A survey. *IEEE Transactions on Neural Networks*, 6(4):837–858, July 1995.
- [Basilevsky, 1994] A. Basilevsky. *Statistical Factor Analysis and Related Methods*. Wiley, New York, 1994.
- [Belkin and Niyogi, 2003] M. Belkin and P. Niyogi. Laplacian eigenmaps for dimensionality reduction and data representation. *Neural Computation*, 15(6):1373–1396, 2003.
- [Bengio et al., 2004] Y. Bengio, J. Paiement, and P. Vincent. Out-of-sample extensions for LLE, Isomap, MDS, Eigenmaps and spectral clustering. In *Advances in Neural Information Processing Systems 16*. MIT Press, 2004.
- [Berg et al., 1984] C. Berg, J.P.R. Christensen, and P. Ressel. *Harmonic Analysis on Semigroups*. Springer-Verlag, 1984.
- [Bishop, 1999] C. M. Bishop. Bayesian PCA. In *Advances in Neural Information Processing Systems*, volume 11, pages 382–388. The MIT Press, 1999.
- [Borg and Groenen, 1997] I. Borg and P. Groenen. *Modern Multidimensional Scaling: Theory and Applications*. Springer, 1997.
- [Boser et al., 1992] B. E. Boser, I. M. Guyon, and V. Vapnik. A training algorithm for optimal margin classifiers. In *Fifth Annual Workshop on Computational Learning Theory*, pages 144–152. ACM, 1992.
- [Burges, 2004] C.J.C. Burges. Some Notes on Applied Mathematics for Machine Learning. In *Advanced Lectures on Machine Learning*, pages 21–40. Springer, 2004.
- [Burges, 2005] C.J.C. Burges. Geometric Methods for Feature Selection and Dimensional Reduction. *Data Mining and Knowledge Discovery Handbook*. Kluwer Academic, 2005.
- [Burges et al., 2002] C.J.C. Burges, J.C. Platt, and S. Jana. Extracting noise-robust features from audio. In *Proc. IEEE Conference on Acoustics, Speech and Signal Processing*, pages 1021–1024. IEEE, 2002.
- [Burges et al., 2003] C.J.C. Burges, J.C. Platt, and S. Jana. Distortion discriminant analysis for audio fingerprinting. *IEEE Transactions on Speech and Audio Processing*, 11(3):165–174, 2003.
- [Chung, 1997] F.R.K. Chung. *Spectral Graph Theory*. American Mathematical Society, 1997.
- [Cook and Weisberg, 1991] R.D. Cook and S. Weisberg. Sliced Inverse Regression for Dimension Reduction: Comment. *Journal of the American Statistical Association*, 86(414):328–332, 1991.
- [Cook, 1998] R.D. Cook. *Regression Graphics*. Wiley, 1998.
- [Cook and Lee, 1999] R.D. Cook and H. Lee. Dimension Reduction in Binary Response Regression. *Journal of the American Statistical Association*, 94(448):1187–1200, 1999.
- [Cox and Cox, 2001] T.F. Cox and M.A.A. Cox. *Multidimensional Scaling*. Chapman and Hall, 2001.
- [Darlington, 1997] R.B. Darlington. Factor analysis. Technical report, Cornell University, 1997.
- [Silva and Tenenbaum, 2002] V. de Silva and J.B. Tenenbaum. Global versus local methods in nonlinear dimensionality reduction. In *Advances in Neural Information Processing Systems 15*, pages 705–712. MIT Press, 2002.
- [Diaconis and Freedman, 1984] P. Diaconis and D. Freedman. Asymptotics of graphical projection pursuit. *Annals of Statistics*, 12:793–815, 1984.
- [Diamantaras and Kung, 1996] K.I. Diamantaras and S.Y. Kung. *Principal Component Neural Networks*. John Wiley, 1996.
- [Duda and Hart, 1973] R.O. Duda and P.E. Hart. *Pattern Classification and Scene Analysis*. John Wiley, 1973.
- [Fowlkes et al., 2004] C. Fowlkes, S. Belongie, F. Chung, and J. Malik. Spectral grouping using the Nyström method. *IEEE Trans. Pattern Analysis and Machine Intelligence*, 26(2), 2004.
- [Friedman and Stuetzle, 1981] J.H. Friedman and W. Stuetzle. Projection pursuit regression. *Journal of the American Statistical Association*, 76(376):817–823, 1981.
- [Friedman et al., 1984] J.H. Friedman, W. Stuetzle, and A. Schroeder. Projection pursuit density estimation. *J. Amer. Statistical Assoc.*, 79:599–608, 1984.
- [Friedman and Tukey, 1974] J.H. Friedman and J.W. Tukey. A projection pursuit algorithm for exploratory data analysis. *IEEE Transactions on Computers*, c-23(9):881–890, 1974.
- [Fukumizu, Bach and Jordan, 2009] K. Fukumizu, F.R. Bach and M.I. Jordan. Kernel Dimension Reduction in Regression. *Annals of Statistics*, to appear.
- [Globerson and Tishby, 2003] A. Globerson and N. Tishby. Sufficient Dimensionality Reduction. *Journal of Machine Learning Research*, 3, 2003.
- [Golub and Van Loan, 1996] G.H. Golub and C.F. Van Loan. *Matrix Computations*. Johns Hopkins, third edition, 1996.
- [Gondran and Minoux, 1984] M. Gondran and M. Minoux. *Graphs and Algorithms*. John Wiley and Sons, 1984.
- [Grimmet and Stirzaker, 2001] G. Grimmet and D. Stirzaker. *Probability and Random Processes*. Oxford University Press, third edition, 2001.
- [Guyon, 2003] I. Guyon. NIPS 2003 workshop on feature extraction.
- [Ham et al., 2004] J. Ham, D.D. Lee, S. Mika, and B. Schölkopf. A kernel view of dimensionality reduction of manifolds. In *Proceedings of the International Conference on Machine Learning*, 2004.
- [Hardoon, Szedmak and Shawe-Taylor, 2004] D.R. Hardoon, S. Szedmak and J. Shawe-Taylor. Canonical correlation analysis: an overview with application to learning methods. *Neural Computation*, 12(16):2639–2664, 2004.
- [Hastie and Stuetzle, 1989] T.J. Hastie and W. Stuetzle. Principal curves. *Journal of the American Statistical Association*, 84(406):502–516, 1989.
- [Hsing and Ren, 2009] T. Hsing and H. Ren. An RKHS formulation of the inverse regression dimension-reduction problem. *Annals of Statistics*, 37(2):726–755, 2009.
- [Horn and Johnson, 1985] R.A. Horn and C.R. Johnson. *Matrix Analysis*. Cambridge University Press, 1985.
- [Hotelling, 1936] H. Hotelling. Relations between two sets of variates. *Biometrika*, 28:321–377, 1936.
- [Huber, 1985] P.J. Huber. Projection pursuit. *Annals of Statistics*, 13(2):435–475, 1985.
- [Hyvärinen et al., 2001] A. Hyvärinen, J. Karhunen, and E. Oja. *Independent Component Analysis*. Wiley, 2001.
- [Kelly, 1928] T. L. Kelly. *Crossroads in the Mind of Man; A study of Differentiable Mental Abilities*. Stanford University Press, 1928.
- [Kimeldorf and Wahba, 1971] G.S. Kimeldorf and G. Wahba. Some results on Tchebycheffian Spline Functions. *J. Mathematical Analysis and Applications*, 33:82–95, 1971.
- [Li, 1991a] C-K. Li. Sliced Inverse Regression for Dimension Reduction. *Journal of the American Statistical Association*, 86(414):316–327, 1991.
- [Li, 1991b] C-K. Li. Sliced Inverse Regression for Dimension Reduction: Rejoinder. *Journal of the American Statistical Association*, 86(414):337–342, 1991.
- [Li, 1992] C-K. Li. On Principal Hessian Directions for Data Visualization and Dimension Reduction. *Journal of the American Statistical Association*, 87(420):1025–1039, 1992.
- [Li, Zha and Chiaromonte, 2005] B. Li, H. Zha and F. Chiaromonte. Contour Regression: A General Approach to Dimension Reduction. *The Annals of Statistics*, 33(4):1580–1616, 2005.
- [Meila and Shi, 2000] M. Meila and J. Shi. Learning segmentation by random walks. In *Advances in Neural Information Processing Systems*, pages 873–879, 2000.
- [Mika et al., 1999] S. Mika, B. Schölkopf, A. J. Smola, K.-R. Müller, M. Scholz, and G. Rätsch. Kernel PCA and de-noising in feature spaces. In *Advances in Neural Information Processing Systems 11*. MIT Press, 1999.
- [Ng et al., 2002] A. Y. Ng, M. I. Jordan, and Y. Weiss. On spectral clustering: analysis and an algorithm. In *Advances in Neural Information Processing Systems 14*. MIT Press, 2002.
- [Nilsson, Sha and Jordan 2007] J. Nilsson and F. Sha and M.I. Jordan. Regression on Manifolds using Kernel Dimension Reduction. *Proceedings of the 24th International Conference on Machine Learning*, 2007.
- [Platt, 2005] J. Platt. Fastmap, MetricMap, and Landmark MDS are all Nyström algorithms. In *Proc. 10th International Conference on Artificial Intelligence and Statistics*, 2005.
- [Press et al., 1992] W.H. Press, B.P. Flannery, S.A. Teukolsky, and W.T. Vettering. *Numerical recipes in C: the art of scientific computing*. Cambridge University Press, 2nd edition, 1992.
- [Ross and Pekőz, 2007] S.M. Ross and E.A. Pekőz. *A Second Course in Probability*. ProbabilityBookstore.com, Boston, MA, 2007.
- [Roweis and Saul, 2000] S.T. Roweis and L.K. Saul. Nonlinear dimensionality reduction by locally linear embedding. *Science*, 290(22):2323–2326, 2000.
- [Schoenberg, 1935] I.J. Schoenberg. Remarks to Maurice Frechet's article. *Annals of Mathematics*, 36:724–732, 1935.
- [Schölkopf, 2001] B. Schölkopf. The kernel trick for distances. In *Advances in Neural Information Processing Systems 13*, pages 301–307. MIT Press, 2001.
- [Schölkopf and Smola, 2002] B. Schölkopf and A. Smola. *Learning with Kernels*. MIT Press, 2002.
- [Schölkopf et al., 1998] B. Schölkopf, A. Smola, and K-R. Muller. Nonlinear component analysis as a kernel eigenvalue problem. *Neural Computation*, 10(5):1299–1319, 1998.
- [Shi and Malik, 2000] J. Shi and J. Malik. Normalized cuts and image segmentation. *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 22(8):888–905, 2000.
- [Spearman, 1904] C.E. Spearman. 'General intelligence' objectively determined and measured. *American Journal of Psychology*, 5:201–293, 1904.
- [Stone, 1982] C.J. Stone. Optimal global rates of convergence for nonparametric regression. *Annals of Statistics*, 10(4):1040–1053, 1982.
- [Tenenbaum, 1998] J.B. Tenenbaum. Mapping a manifold of perceptual observations. In *Advances in Neural Information Processing Systems*, volume 10. The MIT Press, 1998.
- [Tipping and Bishop, 1999A] M.E. Tipping and C.M. Bishop. Probabilistic principal component analysis. *Journal of the Royal Statistical Society*, 61(3):611, 1999.
- [Tipping and Bishop, 1999B] M.E. Tipping and C.M. Bishop. Mixtures of probabilistic principal component analyzers. *Neural Computation*, 11(2):443–482, 1999.
- [Tishby et al., 1999] N. Tishby, F.C. Pereira and William Bialek. The information bottleneck method. *Proc. of the 37th Annual Allerton Conference on Communication, Control and Computing*, 368-377, 1999.
- [Viola and Jones, 2001] P. Viola and M. Jones. Robust real-time object detection. In *Second international workshop on statistical and computational theories of vision*, 2001.
- [Wilks, 1962] S. Wilks. *Mathematical Statistics*. John Wiley, 1962.
- [Williams, 2001] C.K.I. Williams. On a Connection between Kernel PCA and Metric Multidimensional Scaling. In *Advances in Neural Information Processing Systems 13*, pages 675–681. MIT Press, 2001.
- [Williams and Seeger, 2001] C.K.I. Williams and M. Seeger. Using the Nyström method to speed up kernel machines. In *Advances in Neural Information Processing Systems 13*, pages 682–688. MIT Press, 2001.
