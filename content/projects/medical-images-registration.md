---
title: "Medical images registration"
date: 2024-07-31T14:00:00Z
draft: false
cover: 
    image: "projects/registration/cover.jpg"
    alt: "source : https://viso.ai/computer-vision/image-registration/"
---

A key challenge in biomedical image analysis and particularly in brain image analysis is **registration**; human anatomical variability is the norm, necessitating the transformation of image data into standard coordinates. This standardization facilitates analysis and allows results to be generalized to a larger population.

**Affine registration**, which uses linear transformations, provides a robust initial alignment of overall brain structures. This is followed by **diffeomorphic registration**, which models smooth, non-linear deformations to capture intricate anatomical details. Combining these methods ensures both global and local alignment, enhancing the correlation of structural and functional information and improving clinical and research outcomes.

I'll explain the theory behind affine and diffeomorphic registration, applied to 3D brain images registration, and their implementation in `DIPY` [0].

## Affine registration
This part is based on the following paper : *Mattes, D., Haynor, D. R., Vesselle, H., Lewellen, T. K., & Eubank*, W. PET-CT image registration in the chest using free-form deformations. IEEE Transactions on Medical Imaging, 22(1), 120-8, 2003. 

### Introduction


We assume that an image $I$ is described by a set of samples $I\_i = I(\mathbf{x}\_i)$ where $\mathbf{x}\_i \in \Omega \subseteq \mathbb{R}^3$ defined on a cartesian grid with integer spacing. 

We have a reference/fixed/static image $I\_S$ and a moving image $I\_M$. We want to register $I\_M$ on $I\_S$. We therefore need to find a function $\mathbf{T}\_{\phi}$ describing the transformation from $\Omega\_M$ (codomain) to $\Omega\_S$ (domain), with $\phi$ a set of transformation parameters to be determined. We can formulate this as an optimization problem. To align the reference image $I\_S$ with the transformed image $\mathbf{T}\_{\phi} \circ I\_M$, we seek the set of transformation parameters $\phi$ that minimizes a discrepancy function $S$: 

$
\phi^*= \argmin\limits_{\phi} S(I\_S,   \mathbf{T}\_{\phi} \circ I\_M)
$

In the next sections we will answer to the following questions:
- What is $\mathbf{T}\_{\phi}$ in the case of affine registration ? 
- How is the registration implemented in practice (what is the discrepancy function $S$, what is the optimization scheme)? 

### Affine registration parameters

In 3D, an affine transformation can be represented by a $4 \times 4$ matrix that contains 3 parameters for translations and 9 parameters for linear transformations (scaling, rotation, reflection, shear).

The following matrices constitute the basis affine transforms in 3D:

- Translate: $\begin{pmatrix}
    1 & 0 & 0 & t\_x\\\
    0 & 1 & 0 & t\_y\\\
    0 & 0 & 1 & t\_z\\\
    0 & 0 & 0 & 1\\\
\end{pmatrix}$

- Scale: $\begin{pmatrix}
    s\_x & 0 & 0 & 0\\\
    0 & s\_y & 0 & 0\\\
    0 & 0 & s\_z & 0\\\
    0 & 0 & 0 & 1\\\
\end{pmatrix}$

- Shear: $\begin{pmatrix}
    1 & h\_{xy} & h\_{xz} & 0\\\
    h\_{yx} & 1 & h\_{yz} & 0\\\
    h\_{zx} & h\_{zy} & 1 & 0\\\
    0 & 0 & 0 & 1\\\
\end{pmatrix}$

-Rotate along $x$ axis: $\begin{pmatrix}
    1 & 0 & 0 & 0\\\
    0 & \cos(\theta\_x) & -\sin(\theta\_x) & 0\\\
    0 & \sin(\theta\_x) & \cos(\theta\_x) & 0\\\
    0 & 0 & 0 & 1\\\
\end{pmatrix}$

-Rotate along $y$ axis: $\begin{pmatrix}
    \cos(\theta\_y) & 0 & \sin(\theta\_y) & 0\\\
    0 & 1 & 0 & 0\\\
    -\sin(\theta\_y) & 0 & \cos(\theta\_y) & 0\\\
    0 & 0 & 0 & 1\\\
\end{pmatrix}$

-Rotate along $z$ axis:  $\begin{pmatrix}
    \cos(\theta\_z) & -\sin(\theta\_z) & 0 & 0\\\
    \sin(\theta\_z) & \cos(\theta\_z) & 0 & 0\\\
    0 & 0 & 1 & 0\\\
    0 & 0 & 0 & 1\\\
\end{pmatrix}$

Given a point $\mathbf{x} = (x,y,z)^T$, its new position $\mathbf{x'} = (x',y',z')^T$ after an affine transformation will be:


$
\begin{pmatrix}
    x'\\\
    y'\\\
    z'\\\
    1\\\
\end{pmatrix} = \begin{pmatrix}
    a\_0 & a\_1 & a\_2 & a\_3\\\
    a\_4 & a\_5 & a\_6 & a\_7\\\
    a\_8 & a\_9 & a\_{10} & a\_{11}\\\
    0 & 0 & 0 & 1\\\
\end{pmatrix} \begin{pmatrix}
    x'\\\
    y'\\\
    z'\\\
    1\\\
\end{pmatrix} 
\quad (1)
$

Here we have rephrased the basis affine transforms parameters in homogeneous coordinates, which allows us to combine the rotation/scale/sheer/translation in a single matrix.

### Registration process and DIPY implementation
The affine registration is implemented in `DIPY` in `dipy/align/imaffine.py`
#### Brain MRI images format

A common format for medical images, in our case, brain MRI images, is `nifti`. 3D images are stored in cartesian grids of shape $(q\_x, q\_y, q\_z) \in \mathbb{N}^3$. An affine transform (matrix of shape $4 \times 4$) is also stored. This matrix represents the voxel-to-world transformation. That is, if we apply $(1)$ to a voxel of the grid, the resulting coordinates will be the coordinates of this point but in the referential of the scanner ($x, y, z$ distance from the origin, which is located in the center of the brain, in mm) [1]. We write $\mathbf{A\_S}$ for the domain voxel-to-world affine (static image) and $\mathbf{A\_M}$ for the codomain voxel-to-world affine (moving image).

#### What does it mean to map a 3D image (=a grid/domain) on another 3D image ? 

The parameters of the transform $\mathbf{T}\_{\phi}$ that we try to optimize come down to an affine matrix $\mathbf{A}$ with $\phi = (a\_0,  a\_1,  a\_2, a\_3, a\_4, a\_5, a\_6, a\_7, a\_8, a\_9, a\_{10}, a\_{11})$ the parameters of the affine transformation that we seek to optimize (see matrix in $(1)$). 

The codomain $\Omega\_M$ (moving image) and the domain $\Omega\_S$ (static image)can have different shapes $(q\_{M,x}, q\_{M,y}, q\_{M,z})$ and $(q\_{S,x}, q\_{S,y}, q\_{S,z})$. If we register the moving image on the static image, the resulting moved image will have the shape of the static image. We therefore have a grid, the domain $\Omega\_S$, which we want to fill with values. For each voxel of the domain which we want to fill with a value, we fill it with the value that we retrieve in the codomain with the help of $\mathbf{T}\_{\phi} = \mathbf{A}$, $\mathbf{A\_S}$ and $\mathbf{A\_M}$. 

Given $\mathbf{x} \in \Omega\_S = \llbracket 0, q\_{S,x} \rrbracket \times \llbracket 0, q\_{S,y} \rrbracket \times \llbracket 0, q\_{S,z} \rrbracket$, we have:

 $\underbrace{\mathbf{T}\_{\phi}\circ I\_M}_{\text{transformed image}} (\mathbf{x}) = \mathcal{L}(\mathbf{A\_M}^{-1}\mathbf{A}\mathbf{A\_S}\mathbf{x})$.


This means that in order to transform a moving image towards a static image, we first map each voxel $\mathbf{x} = (i,j,k)$ of the static image to world coordinates $(x,y,z)$ by applying $\mathbf{A\_S}$. Then we apply the affine transform $\mathbf{A}$ to $(x,y,z)$ obtaining $(x', y', z')$ in moving image's world coordinates. Finally, $(x', y', z')$ is mapped to voxel coordinates $(i', j', k')$ in the moving image by multiplying by the inverse of $\mathbf{A\_M}$. 

$\mathcal{L}: \mathbb{R}^3 \longrightarrow \mathbb{R}$ is a trilinear interpolation function [2]. It linearly interpolates the values of the 8 voxels of $\Omega_M$ that are the closest to $\mathbf{A\_M}^{-1}\mathbf{A}\mathbf{A\_S}\mathbf{x} \in \mathbb{R}^3$. This is useful when the codomain's resolution is very small compared with the domain's resolution ($q\_M \ll q\_S$) as it allows to move the moving image in a smoother way into the domain. 


In `DIPY`, this is done by the `_apply_transform` function in the `AffineMap` class. 

#### How are the coefficients of $A$ computed ? 

The algorithmic core for computing and optimizing $S$  and finding the best matrix $A$ relies on **Limited-memory BFGS** [3], which is an optimization algorithm in the family of quasi-Newton methods. 

In a classical Newton method, we would find $\phi^*$ with the following iterative scheme :

$
\phi\_{k+1} = \phi\_{k} - \underbrace{[\frac{\partial^2S}{\partial\phi^2}]^{-1}}\_{\text{Hessian} \in \mathbb{R}^{12 \times 12}} \underbrace{\frac{\partial S}{\partial\phi}}\_{\text{Gradient} \in \mathbb{R}^{12}} = \phi\_{k} - [\nabla^2S]^{-1}{\nabla S}
$

However finding the inverse of the Hessian in high dimensions to compute the Newton direction can be an expensive operation. In quasi-Newton methods the Hessian matrix does not need to be computed. The Hessian is updated by analyzing successive gradient vectors instead.


In `DIPY`, this optimization is done in the `òptimize` function of the `ÀffineRegistration` class.

#### What is $S$ and how to compute $\nabla S$ ? 

The image discrepancy measure used in `DIPY` is the **negative mutual information**. 
To understand mutual information, we need to define the joint histogram between $I\_S$ and $\mathbf{T}\_{\phi}\circ I\_M$.
From now on, we consider note $I\_T = \mathbf{T}\_{\phi}\circ I\_M$ the transformed image. $I\_S$ and $I\_T$ are in the same space $\Omega\_S$ through the process that has been described before. 

As implemented in `DIPY`, it is possible to use masks so that all voxels $\mathbf{x}$ are not used to compute the mutual information between images. It is also possible to undersample (for instance only select 30% of the voxels). We therefore note $\Omega$ the space of voxels that are used to actually compute the metric. 

Let $L\_S$ and $L\_T$ be specified numbers of uniformly sized bins
along the respective dimensions of the joint histogram of the
static and moving images. The joint histogram is a matrix $\mathbf{H} \in \mathbb{N}^{L\_S \times L\_T}$. The value $(a,b)$ in $\mathbf{H}$ is equal to the number of voxels $\mathbf{x} \in \Omega$ that have intensity $a$ in $I\_S$ and intensity $b$ in $I\_T$. By dividing by the total of number of pixels $|\Omega|$, we normalize the histogram so that it behaves like a traditional probability distribution. The histogram bins are indexed by integer values $\kappa$, $0 \leq \kappa < L\_S$ and $\iota$, $0 \leq \iota < L\_T$.

A Parzen window is used to generate continuous estimates of the underlying image distributions, thereby reducing the effects of quantization from interpolation and discretization from binning the data. It also makes the distributions differentiable which allows us to compute a gradient. 

Let $\beta^{(3)}$ be a cubic spline Parzen window and $\beta^{(0)}$ be a zero-order spline Parzen window (centered unit pulse). We will come bac on the calculation of the coefficients of the spline later. The smoothed joint histogram of $(I\_S, I\_T)$ is given by:

$
p(\iota, \kappa | \phi) = \alpha \displaystyle\sum\_{\mathbf{x}\in\Omega}\beta^{(0)}(\kappa - \frac{I\_S(\mathbf{x})-I\_S^0}{\Delta b\_S}) \times \beta^{(3)}(\iota - \frac{I\_T(\mathbf{x})-I\_T^0}{\Delta b\_T}) \quad (2)
$

where $\alpha$ is a normalization factor that ensures $\sum p(\iota, \kappa) = 1$. Each contributing image value is normalized by the minimum intensity value, $I\_S^0$ and $I\_T^0$ and the intensity range of each bin, $\Delta b\_S$ and $\Delta b\_T$ to fit into the specified number of bins ($L\_S$ or $L\_T$) in the intensity distribution. 
The marginal smoothed histogram for the test image is computed from the joint histogram:

$
p\_T(\iota | \phi) = \displaystyle\sum\_{\kappa}p(\iota, \kappa | \phi)
$


The static image smoothed histogram is computed as:

$
p\_S(\kappa) = \alpha \displaystyle\sum\_{\mathbf{x}\in \Omega} \beta^{(0)}(\kappa - \frac{I\_S(\mathbf{x})-I\_S^0}{\Delta b_S})
$


The **negative of mutual information $S$** between the static image and the moved image is expressed as a function of the transformation parameters $\phi$: 

$
S(I\_S, \mathbf{T}\_{\phi}\circ I\_M) = -\displaystyle\sum\_{\iota}\displaystyle\sum\_{\kappa}p(\iota,\kappa | \phi)\log \frac{p(\iota,\kappa | \phi)}{p\_M(\iota | \phi)p\_S(\kappa)}
$

It can be seen as the Kullback-Leibler divergence between $p$, the smoothed joint histogram, and $p\_S \otimes p\_T$. It measures the cost for considering $I\_T$ and $I\_S$ as independant random variables, when in reality they are not. 

Calculation of the gradient of the cost function is necessary as seen earlier:

$
\nabla S = [\frac{\partial S}{\partial a\_0},..., \frac{\partial S}{\partial a\_{11}}]^T
$

A single component of the gradient requires differentiation of the joint distribution $(2)$. After applying the chain rule, the ith partial derivative of the joint distribution is given as:

$
\frac{\partial p(\iota, \kappa)}{\partial a\_i} = \frac{1}{\Delta b\_T |\Omega|}\displaystyle\sum\_{\mathbf{x}\in \Omega}\beta^{(0)}(\kappa - \frac{I\_S(\mathbf{x})-I\_S^0}{\Delta b\_S}) \times \frac{\partial \beta^{(3)}(u)}{\partial u}\bigg|\_{u=\iota - \frac{I\_T(\mathbf{x})-I\_T^0}{\Delta b\_T}} \times (\frac{-\partial I\_M(\mathbf{t})}{\partial \mathbf{t}} \bigg|\_{\mathbf{t}=\mathbf{T}\_{\phi}(\mathbf{x})})^T \frac{\partial \mathbf{T}\_{\phi}(\mathbf{x})}{\partial a\_i}
$

In `DIPY`, all the metric calculations of $S$ and $\nabla S$ is done in the `MutualInformationMetric` class. In our case, the `transform` argument is the 3D affine transformation. 

#### B-Spline coefficients calculation

As explained in the introduction, $I$ is described by a set of samples $I\_i = I(\mathbf{x}\_i)$ where $\mathbf{x}\_i \in \Omega$.

The calculation of $I(\mathbf{x})$ at points not on the grid requires an interpolation method based on the samples $I\_i$ and their locations $\mathbf{x}\_i$. In `DIPY`, an interpolation scheme taht represents the underlying continuous image by a **B-Spline basis** is used. The expansion of coefficients $c\_i$ of the basis are computed from the image samples $I\_i$ through an efficient recursive filtering algorithm [4]. Values of $I(\mathbf{x})$ that do not lie on the lattice can be interpolated: 

$
I(\mathbf{x}) = \displaystyle\sum\_{i}c\_i \beta^{(3)}(\mathbf{x}-\mathbf{x\_i})
$ 

where $\mathbf{x} = (x,y,z)^T$ is any real-valued voxel location in the volume, $\mathbf{x\_i} = (x\_i,y\_i,z\_i)^T$ is the coordinate vector of a lattice point, and $\beta^{(3)}(\mathbf{x})= \beta^{(3)}(x)\beta^{(3)}(y)\beta^{(3)}(z)$ is a separable convolution kernel. The argument of the spline window is the sampled cubic B-Spline:

$
\beta^{(3)}(x) = \begin{cases} 
\frac{1}{6}(4-6x^2+3|x|^3) & \text{if } 0 \leq |x| < 1, \\\
\frac{1}{6}(2-|x|)^3 & \text{if } 1 \leq |x| < 2, \\\
0 & \text{if } 2 \leq |x|.
\end{cases}
$

The gradient of the interpolated image at any location can therefore be calculated : 

$
\frac{\partial I(\mathbf{x})}{\partial x} = \displaystyle\sum\_{i}c\_i(\frac{d\beta^{(3)}(u)}{du}\bigg|\_{u=x-x\_i}\beta^{(3)}(y-y\_i)\beta^{(3)}(z-z\_i))
$

with similar formulas for $\frac{\partial I}{\partial y}$ and $\frac{\partial I}{\partial z}$. The cubic spline window $\beta^{(3)}$ can be differentiated explicitely and after simplification redices to the difference of two shifted second-order splines:

$
\frac{d\beta^{(3)}(u)}{du} = \beta^{(2)}(u+\frac{1}{2}) - \beta^{(2)}(u-\frac{1}{2})
$

#### Multiresolution framework

In order to avoid local minima and to decrease computatio time, a hierarchical multiresolution optimization scheme is used. This means that $\mathbf{A}$ is initially calculated for our images with a downsized resolution, then as the resolution is increased, fine misalignments are recovered. The low resolution images are smoothed with a gaussian kernel. 

In `DIPY`, this corresponds to the `level\_iters`, `sigmas` and `factors` argument that are used to initialize the `ÀffineRegistration` class. The `level\_iters` argument is the number of iterations at each scale of the scale space (by default, a 3-level scale space with iterations sequence equal to $[10000, 1000, 100]$ is used). The `sigmas` argument is a custom smoothing parameter to build the scale space (standard deviation of the smoothing gaussian kernels, defaults to $[3,1,0]$). The `factors` argument defines the resolution of the scale spaces at each level($\Omega$), it defaults to $[4,2,1]$.

#### Image registration result example

![affine_reg](../../projects/registration/affine_moving.png)

You can see on this overlay that after affine registration, both brains are aligned but we can still see some spots where they need to be morphed in a **non-linear** way. That's what is explained in the next part. 

## Diffeomorphic map registration 

## RSL registration code



[0]: https://dipy.org/
[1]: https://nipy.org/nibabel/coordinate_systems.html
[2]: https://en.wikipedia.org/wiki/Trilinear_interpolation
[3]: https://en.wikipedia.org/wiki/Limited-memory_BFGS
[4]: M. Unser, A. Aldroubi, and M. Eden, “Fast B-spline transforms for continuous image representation and interpo-lation,” IEEE Trans. Pattern
Anal. Machine Intell., vol. 13, pp. 277–285, Mar. 1991.