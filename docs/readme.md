When evaluating the quality of predicted depth maps compared to ground truth (original) depth maps, several metrics are commonly used. These metrics assess the accuracy, consistency, and overall performance of the depth prediction. Here are the most frequently used comparison metrics:

---

### **1. Absolute Error Metrics**

* **Mean Absolute Error (MAE):**

  MAE=1N∑i=1N∣di−di∗∣\text{MAE} = \frac{1}{N} \sum_{i=1}^N |d_i - d_i^*|
  Measures the average absolute difference between predicted (did_i) and ground truth (di∗d_i^*) depths.
* **Root Mean Square Error (RMSE):**

  RMSE=1N∑i=1N(di−di∗)2\text{RMSE} = \sqrt{\frac{1}{N} \sum_{i=1}^N (d_i - d_i^*)^2}
  Penalizes larger errors more than smaller ones, providing insight into overall prediction quality.
* **Mean Absolute Relative Error (MRE):**

  MRE=1N∑i=1N∣di−di∗∣di∗\text{MRE} = \frac{1}{N} \sum_{i=1}^N \frac{|d_i - d_i^*|}{d_i^*}
  Captures the relative error between predicted and ground truth values.

---

### **2. Accuracy Threshold Metrics**

These metrics evaluate how many predicted depths fall within a certain threshold of the ground truth.

* **Threshold Accuracy (δ\delta):**
  δ=max⁡(didi∗,di∗di)\delta = \max \left( \frac{d_i}{d_i^*}, \frac{d_i^*}{d_i} \right)
  Accuracy is often reported for thresholds like δ<1.25\delta < 1.25, δ<1.252\delta < 1.25^2, and δ<1.253\delta < 1.25^3.

---

### **3. Logarithmic Error Metrics**

* **Mean Logarithmic Error (Log-MAE):**

  Log-MAE=1N∑i=1N∣log⁡(di)−log⁡(di∗)∣\text{Log-MAE} = \frac{1}{N} \sum_{i=1}^N |\log(d_i) - \log(d_i^*)|
* **Root Mean Square Log Error (Log-RMSE):**

  Log-RMSE=1N∑i=1N(log⁡(di)−log⁡(di∗))2\text{Log-RMSE} = \sqrt{\frac{1}{N} \sum_{i=1}^N (\log(d_i) - \log(d_i^*))^2}
  These are particularly useful when depth values vary over several orders of magnitude.

---

### **4. Structural Similarity Metrics**

* **Structural Similarity Index Measure (SSIM):**

  Evaluates the perceptual similarity between the predicted and ground truth depth maps. It takes into account luminance, contrast, and structure:
  SSIM=(2μxμy+C1)(2σxy+C2)(μx2+μy2+C1)(σx2+σy2+C2)\text{SSIM} = \frac{(2 \mu_x \mu_y + C_1)(2 \sigma_{xy} + C_2)}{(\mu_x^2 + \mu_y^2 + C_1)(\sigma_x^2 + \sigma_y^2 + C_2)}

---

### **5. Scale-Invariant Metrics**

* **Scale-Invariant Logarithmic Error:**

  This metric accounts for scale ambiguities in depth maps:
  Scale-Invariant Error=12N∑i=1N(log⁡(di)−log⁡(di∗)+α)2−α22\text{Scale-Invariant Error} = \frac{1}{2N} \sum_{i=1}^N \left( \log(d_i) - \log(d_i^*) + \alpha \right)^2 - \frac{\alpha^2}{2}
  where α=1N∑i=1N(log⁡(di)−log⁡(di∗))\alpha = \frac{1}{N} \sum_{i=1}^N \left( \log(d_i) - \log(d_i^*) \right).

---

### **6. Gradient and Normal-Based Metrics**

* **Gradient Error:**

  Compares the gradients of the predicted and ground truth depth maps to evaluate consistency in surface details:

  Gradient Error=1N∑i=1N∥∇di−∇di∗∥\text{Gradient Error} = \frac{1}{N} \sum_{i=1}^N \|\nabla d_i - \nabla d_i^*\|
* **Surface Normal Error:**

  Measures the angular deviation between the surface normals derived from the predicted and ground truth depth maps.

---

### **7. Point Cloud Metrics**

If the depth maps are converted to 3D point clouds:

* **Chamfer Distance:**

  Measures the distance between two sets of 3D points generated from depth maps.
* **Earth Mover's Distance (EMD):**

  Measures the minimum cost of transforming one point cloud into another.

---

By combining these metrics, you can get a comprehensive evaluation of the predicted depth map's quality. The choice of metrics often depends on the specific application and the importance of scale, relative accuracy, or structural similarity.
