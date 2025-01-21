The mathematical foundation of the paper "Maximizing Self-supervision from Thermal Image for Effective Self-supervised Learning of Depth and Ego-motion" can be explained step by step as follows:

* **Histogram Calculation (Equation 1):**

  * The paper proposes to rearrange the thermal radiation values according to the observation number of each sub-histogram.
  * The first step is to calculate the histogram for raw temperature measurements. The histogram is represented by *h(i)* and is defined as the number of samples, denoted by  *ni* , within a specific temperature range [bi, bi+1).
  * The temperature ranges are calculated by dividing the global minimum and maximum values ( *tmin* ,  *tmax* ) of the raw images by a hyperparameter,  *Nbin* , which represents the number of bins in the histogram.

  **Formula:**

  * *h(i) = ni* , for *i* =  *b0* ,  *b1* , ..., *bmax*

  **Where:**

  * *ni* = number of samples within the range [ *bi* ,  *bi+1* )
  * *b0* , ..., *bi* = values calculated by dividing the global min-max values ( *tmin* ,  *tmax* ) by the bin number, *Nbin*
* **Thermal Radiation Rearrangement (Equation 2):**

  * The next step involves rearranging the raw measurement values within each sub-histogram to enhance image information.
  * Each raw measurement value,  *x* , within the sub-histogram [ *bi* ,  *bi+1* ) is rearranged proportionally to the number of samples in that sub-histogram using the formula below.

  **Formula:**

  * *x' = αi * ((x - bi) / (bi+1 - bi)) + b'i*

  **Where:**

  * *αi* = scaling factor proportional to the number of samples in the sub-histogram, defined as *αi = ni / (∑max j=0 nj)*
  * *b'i* = new offset of each scaled sub-histogram, defined as *b'i = ∑i−1 j=0 αj*
  * This process stretches or squeezes each sub-histogram based on its sample count, resulting in improved image contrast and detail.
* **Local Detail Enhancement (Equation 3):**

  * After the rearrangement process, local image details are enhanced to further improve the self-supervisory signal.
  * The paper adopts Contrast Limited Adaptive Histogram Equalization (CLAHE) to enhance local details without amplifying noise.

  **Formula:**

  * *Ieh = fLDE(I')*

  **Where:**

  * *Ieh* = locally enhanced image
  * *fLDE* = CLAHE function
  * *I'* = rearranged thermal image
* **Self-Supervised Training Loss (Equation 4):**

  * The overall self-supervised training loss function is used to train the single-view depth and multiple-view pose estimation network.

  **Formula:**

  * *Ltotal = Lrec + λgcLgc + λsmLsm*

  **Where:**

  * *Lrec* = thermal image reconstruction loss
  * *Lgc* = geometric consistency loss
  * *Lsm* = edge-aware depth smoothness loss
  * *λgc* and *λsm* = hyperparameters
* **Temperature Consistency Loss (Equation 5):**

  * This loss is calculated to ensure consistency between the synthesized thermal image and the original target thermal image.

  **Formula:**

  * *Lpe(Ieht, Ĩeht) = (γ/2) * (1− SSIM(Ieht, Ĩeht)) + (1− γ) * ||Ieht − Ĩeht||1*

  **Where:**

  * *γ* = scale factor between SSIM and L1 loss
  * *SSIM* = Structural Similarity Index Map
* **Smoothness Loss (Equation 6):**

  * The smoothness loss regularizes the estimated depth map to ensure smooth properties, especially in low-texture areas.

  **Formula:**

  * *Lsm = ∑ |∇Dt| * e^(-|∇Ieht|)

  **Where:**

  * *∇* = first differential operator along the spatial direction
* **Geometric Consistency Loss (Equation 7, 8):**

  * This loss enforces scale consistency between estimated depth maps by minimizing geometric inconsistencies.

  **Formulas:**

  * *Gdiff = |D̃t - D't| / (D̃t + D't)*
  * *Lgc = (1 / |Vp|) * ∑Vp Gdiff*

  **Where:**

  * *Gdiff* = geometric inconsistency map
  * *D̃t* = synthesized depth map
  * *D't* = interpolated depth map
  * *Vp* = valid points
  * *|Vp|* = number of points in Vp
* **Invalid Pixel Masking (Equation 9):**

  * Invalid pixel differences are excluded from the reconstruction loss to improve accuracy.

  **Formulas:**

  * *Lrec = (1 / |Vp|) * ∑Vp Mgp * Msp * Lpe(Ieht, Ĩeht)*
  * *Mgp = 1−Gdiff*
  * *Msp = [ Lpe(Ieht, Ĩeht) < Lpe(Ieht, Iehs) ]*

  **Where:**

  * *Mgp* = geometric inconsistent pixel mask
  * *Msp* = static pixel mask
  * *[ ]* = Iverson bracket

These equations and explanations provide a detailed understanding of the mathematical foundation of the paper, covering key aspects like thermal image enhancement, loss functions, and the techniques used to address challenges in self-supervised depth and pose estimation from thermal images.
