---
title: "Diffusion-weighted imaging theory"
date: 2024-07-23T14:00:00Z
draft: false
cover: 
    image: "projects/dwi/dwi.png"
    alt: "csd-msmt"
---


Here's an overview of how MRI works:

- **Magnetic Alignment**: The MRI scanner generates a strong magnetic field that aligns the protons in the body's hydrogen atoms, which are abundant in water and fat.

- **Radiofrequency Pulses**: The scanner then emits RF pulses at a specific frequency that matches the resonance frequency of the aligned protons. These pulses provide energy, causing the protons to tip out of their alignment with the magnetic field (for example, a 90° RF pulse tips the aligned protons by 90 degrees from their inital alignment with the main magnetic field (B0), from $ z $-axis to transverse plane ($x-y$ plane), this tipping creates a detectable signal in the MRI receiver coil).

- **Relaxation**: After the RF pulses are turned off, the protons begin to relax back to their original alignment. As they do, they release the absorbed energy. This relaxation occurs at different rates in different tissues, described by T1 and T2 relaxation times.

- **Signal Detection**: The released energy during relaxation generates signals that are detected by the MRI scanner's receivers. These signals are then processed by a computer to construct detailed images of the body's internal structures.

By carefully controlling the timing and strength of the RF pulses, MRI can produce various types of images. Two common types of MRI sequences are T1-weighted and T2-weighted images. T1-weighted sequences provide high-resolution images where fat appears bright and water appears dark, making them useful for visualizing anatomical details. T2-weighted sequences, on the other hand, highlight differences in water content, with water appearing bright and fat appearing dark, which is particularly useful for detecting edema, inflammation, and other fluid-related abnormalities. Together, T1 and T2 sequences offer complementary information, aiding in comprehensive diagnostic evaluations.


Diffusion-weighted magnetic resonance imaging (DWI or dMRI) uses specific MRI sequences and software to generate images based on the diffusion of water molecules, creating contrast in MR images. This technique enables the mapping of the diffusion process of molecules, primarily water, in biological tissues, in vivo and non-invasively.

The brain is composed of white matter and gray matter. White matter mainly consists of myelinated axons responsible for transmitting signals between different brain regions. Gray matter contains neuronal cell bodies, dendrites, and unmyelinated axons, involved in processing and regulating information.

Molecular diffusion in brain tissues is not random but reflects interactions with various obstacles such as axons, cell membranes, and other neural components, especially in white matter. Therefore, the diffusion patterns of water molecules can reveal microscopic details about tissue architecture, whether normal or diseased.


![white matter](../../projects/dwi/wm.png)

### The Physics of Diffusion
#### Isotropic Diffusion

Diffusion, as described by Fick's laws, involves the movement of particles driven by concentration differentials. Fick's first law states that the flux $F(r, t)$ of particles is proportional to the negative gradient of concentration $C(r, t)$, where $D$ is the diffusion coefficient:

$
F(r, t) = -D \nabla C(r, t) \quad 
$

Fick's second law, derived from the conservation of mass, links flux to changes in concentration over time:

$
D \nabla^2 C(r, t) = \frac{\partial C(r, t)}{\partial t} \quad 
$

The probability density $P(r, t)$ of finding a particle per unit volume obeys a similar diffusion equation:

$
D \nabla^2 P(r, t) = \frac{\partial P(r, t)}{\partial t} \quad 
$

For unrestricted isotropic diffusion, the solution for $P(r|r_0, t)$, the probability density function of finding a particle at position $r$ given it started at $r_0$ at time $t = 0$, is:

$
P(r|r_0, t) = \left( \frac{4 \pi Dt}{N} \right)^{-\frac{3}{2}} \exp \left( -\frac{|r - r0|^2}{4Dt} \right) 
$

The mean square displacement $\langle \Delta r \rangle^2$ of diffusing particles over time $t$ is given by:

$
\langle \Delta r \rangle^2 = 2NDt 
$

where $N$ is the dimensionality of diffusion (1 for line, 2 for plane, 3 for free space), reflecting Einstein's description of diffusion behavior.

#### Anisotropic Diffusion


When Brownian motion exhibits non-uniform spatial characteristics, a single scalar diffusion coefficient becomes insufficient to fully characterize the system. Instead, the flux of particles $F$ can be described using a diffusion tensor $\mathbf{D}$:

$
F = -D \nabla C 
$

Here, $D$ is a symmetric $3 \times 3$ matrix that accounts for the anisotropic nature of diffusion. Due to the requirement of real values, the diffusion tensor $\textbf{D}$ satisfies $D_{ij} = D_{ji}$, with only six independent elements:

$$
\textbf{D} = \begin{pmatrix}
    D_{xx} & D_{xy} & D_{xz} \\\
    D_{xy} & D_{yy} & D_{yz} \\\
    D_{xz} & D_{yz} & D_{zz}
\end{pmatrix} 
$$

For a compartment governed by the diffusion tensor $D$, the probability that a particle initially at position $\mathbf{r_0}$ reaches position $\textbf{r}$ at time $t$ is given by:

$
P(\mathbf{r}|\mathbf{r_0}, t) = \frac{1}{\sqrt{|\mathbf{D}|(4\pi t)^3}} \exp \left( -\frac{(\mathbf{r} - \mathbf{r_0})^T \mathbf{D}^{-1} (\mathbf{r} - \mathbf{r_0})}{4t} \right) 
$

With, $|\mathbf{D}|$ denotes the determinant of $\mathbf{D}$, encapsulating the volume scaling factor of the diffusion process. 
In this model, introduced by Basser et al, diffusion is described by a multivariate normal distribution. 


#### dMRI Physics

The decrease in the spin echo signal, caused by the loss of coherence among nuclear spins due to both their translational movement and the application of well-defined spatial gradient pulses, serves as a method to quantify motion. A widely utilized technique for measuring diffusion involves modifying the spin echo sequence, known as pulsed gradient spin echo (PGSE), originally pioneered by Stejskal and Tanner. 

![pulse](../../projects/dwi/pulse.png)

We can show that for such sequence, the measured signal will verify the following equation: 

$
\frac{S}{S_0} = \exp(-\gamma^2G^2\delta^2(\Delta-\frac{1}{3}\delta)D)
$

Where $\delta$ is the duration of the dradient, $\Delta$ is the diffusion time (the time between the two leading edges of the gradient pulses), $G$ is the gradient strength, $S$ is the measured signal, $S_0$ is the measured signal in the absence of diffusion weighting gradient, $\gamma$ is the gyromagnetic ratio and $D$ is the diffusion coefficient. 

We can rewrite this equation: 

$
S = S_0e^{-bD}
$

where $b$ is the b-value, defined by: 

$
b = \gamma^2G^2\delta^2(\Delta-\frac{1}{3}\delta)
$

This value has units s/mm&sup2; and represents the sensitivity of the sequence to motion. A high b-value indicates a greater diffusion weighting of the image.


#### DTI 
[MR Diffusion Tensor Spectroscopy and Imaging](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC1275686/pdf/biophysj00080-0261.pdf) (1994)

If the diffusion is anisotropic, Basser et al [MR Diffusion Tensor Spectroscopy and Imaging] derived a formula that relates the signal attenuation to the diffusion tensor $\textbf{D}$ by:

$
\frac{S}{S_0} = e^{-\textbf{B}:\textbf{D}}
$

with :

$
\textbf{B} = \begin{bmatrix}
br_x^2 & br_xr_y & br_xr_z \\\
br_xr_y & br_y^2 & br_yr_z \\\
br_xr_z & br_yr_z & br_z^2
\end{bmatrix} 
$

where $\textbf{r} = (r_x, r_y, r_z)$ is the gradient direction. 

We can then write : 

$
S(b,\textbf{r}) = S_0e^{-br_x^2D_{xx}-br_y^2D_{yy}-br_z^2D_{zz}-2br_xr_yD_{xy}-2br_xr_zD_{xz}-2br_yr_zD_{yz}} = S_0e^{—b\textbf{r}^T\textbf{Dr}}
$

We therefore need a minimum of six non-collinear diffusion measurements to fully estimate the diffusion tensor coefficients : 

$
\mathbf{d} = \mathbf{H}^{-1}\mathbf{Y}
$

with $\mathbf{d} = \begin{bmatrix}
D_{xx} \\\
D_{yy} \\\
D_{zz} \\\
D_{xy} \\\
D_{xz} \\\
D_{yz} \\\
\end{bmatrix}$, $\quad \mathbf{H} = \begin{bmatrix}
r_{x1}^2 & r_{y1}^2 & r_{z1}^2 & 2r_{x1}r_{y1} & 2r_{x1}r_{z1} & 2r_{y1}r_{z1}\\\
& & &.\\\
& & &.\\\
& & &.\\\
& & &.\\\
r_{x6}^2 & r_{y6}^2 & r_{z6}^2 & 2r_{x6}r_{y6} & 2r_{x6}r_{z6} & 2r_{y6}r_{z6} \\\
\end{bmatrix}$, $\quad \textbf{Y} = \begin{bmatrix}
\frac{\ln(\frac{S_0}{S_1})}{b} \\\
. \\\
. \\\
. \\\
. \\\
\frac{\ln(\frac{S_0}{S_6})}{b}
\end{bmatrix}$

In each voxel of 3D DWI image, the diffusion tensor may be visualized as an ellipsoid with the eigenvectors defining the directions of the principal axes and the ellipsoidal radii defined by the eigenvalues. When the diffusion is isotropic, all the eigenvalues are nearly equal. Otherwise, the eigenvalues are significantly different in magnitude. Tissue injury, diseases or normal physiological changes like aging modify the local tissue microstructure and therefore the eigenvalue magnitudes. The diffusion tensor is therefore a sensitive probe for characterizing both normal and abnormal tissue microstructure.  

![source : https://medium.com/@ayt.hsueh/%E6%93%B4%E6%95%A3%E5%BC%B5%E9%87%8F%E5%BD%B1%E5%83%8F%E8%99%95%E7%90%86-diffusion-tensor-imaging-dti-processing-%E4%B8%80-%E5%90%8D%E8%A9%9E%E4%BB%8B%E7%B4%B9-501d936be611 ](../../projects/dwi/ellipse.png)


![3d_screenshot_CSD](../../projects/dwi/3d_screenshot_CSD.png)

![3d_screenshot_DTI](../../projects/dwi/3d_screenshot_DTI.png)


[Modelling white matter with spherical deconvolution: How and why?](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6585735/pdf/NBM-32-na.pdf)

[Diffusion and MRI](https://imaging.mrc-cbu.cam.ac.uk/imaging/AnalyzingDiffusion?action=AttachFile&do=get&target=Diffusion_and_MRI.pdf)

https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2041910/pdf/13311_2011_Article_40300316.pdf