# Ch.5 — Feature Matching & Correspondence: Technical Lineage

Ch.4 established the mathematical framework of state estimation. But whether we use a Kalman filter or a factor graph, the **data association** problem — "which observation corresponds to which landmark?" — must be solved first. This chapter covers the technical lineage of its core: feature matching and correspondence search.

> **Purpose of this chapter**: Almost every component of sensor fusion — Visual Odometry, calibration, loop closure, point cloud registration — depends on **correspondence**. This chapter traces the technical flow starting from mutual information and continuing through to RoMa, clearly showing which limitation of the previous generation each method addressed.

---

## 5.1 What Is the Correspondence Problem

### 5.1.1 The Problem of Finding "the Same Thing"

The correspondence problem is the problem of identifying **physically identical points, regions, or structures** across two or more observations. To understand why this problem is fundamental in robotics, we must recognize that virtually every stage of the sensor fusion pipeline presupposes correspondence.

- **Visual Odometry**: Camera motion can only be estimated by finding the 2D projections of the same 3D point across consecutive frames.
- **Calibration**: Estimating the camera-LiDAR extrinsic parameters requires identifying the same physical point observed by both sensors.
- **Loop Closure**: Recognizing a previously visited place requires confirming correspondences between current observations and past observations.
- **Point Cloud Registration**: Alignment of two scans is the process of estimating a rigid transformation based on corresponding point pairs.

### 5.1.2 Three Types of Correspondence

#### 2D-2D Correspondence

This is the problem of finding the projections of the same 3D point across two images. It forms the basis of Visual Odometry, stereo matching, and image stitching.

When the point $\mathbf{p}_1 = (u_1, v_1)$ in image $I_1$ and the point $\mathbf{p}_2 = (u_2, v_2)$ in image $I_2$ are projections of the same 3D point $\mathbf{X}$, the pair $(\mathbf{p}_1, \mathbf{p}_2)$ is called a correspondence.

The geometric relationship between the two images is expressed by the **epipolar constraint**:

$$\mathbf{p}_2^\top \mathbf{F} \mathbf{p}_1 = 0$$

Here, $\mathbf{F}$ is the fundamental matrix (3×3, rank 2). When the intrinsic parameters are known, we use the essential matrix $\mathbf{E} = \mathbf{K}_2^\top \mathbf{F} \mathbf{K}_1$:

$$\hat{\mathbf{p}}_2^\top \mathbf{E} \hat{\mathbf{p}}_1 = 0$$

Here, $\hat{\mathbf{p}} = \mathbf{K}^{-1} \mathbf{p}$ are the normalized image coordinates.

#### 2D-3D Correspondence

The correspondence between a 2D point in an image and a 3D map point. This is the basis of the **PnP (Perspective-n-Point)** problem and is central to visual localization and map-based tracking in SLAM.

Relationship between a 3D point $\mathbf{X} = (X, Y, Z)^\top$ and its 2D projection $\mathbf{p} = (u, v)^\top$:

$$s \begin{bmatrix} u \\ v \\ 1 \end{bmatrix} = \mathbf{K} [\mathbf{R} | \mathbf{t}] \begin{bmatrix} X \\ Y \\ Z \\ 1 \end{bmatrix}$$

With a minimum of 6 (DLT), 4 (P3P+1), or 3 (P3P) correspondences, the camera pose $(\mathbf{R}, \mathbf{t})$ is estimated.

#### 3D-3D Correspondence

The problem of finding the same physical point between two point clouds. It is used for LiDAR scan-to-scan registration, multi-LiDAR calibration, and point cloud registration during loop closure.

Estimating the rigid transformation $(\mathbf{R}, \mathbf{t})$ between two point clouds $\mathcal{P} = \{\mathbf{p}_i\}$ and $\mathcal{Q} = \{\mathbf{q}_i\}$:

$$\min_{\mathbf{R}, \mathbf{t}} \sum_{i} \| \mathbf{q}_i - (\mathbf{R} \mathbf{p}_i + \mathbf{t}) \|^2$$

The closed-form solution to this optimization problem can be obtained via SVD (ICP, see Ch.7).

### 5.1.3 Why Correspondence Is Central to Sensor Fusion

In sensor fusion, correspondence is not merely preprocessing but **the bottleneck that determines the accuracy of the entire system**. A single incorrect correspondence (outlier) can completely ruin pose estimation, and if correspondences cannot be found (as in textureless environments) the system itself fails to operate. The remainder of this chapter traces the technical lineage of how this problem has been solved.

---

## 5.2 Traditional Feature Detection & Description

The traditional correspondence pipeline consists of three stages: **detect → describe → match**. This section covers the first two stages — where to find keypoints (detection) and how to represent those keypoints (description).

### 5.2.1 Corner Detection: Harris → FAST → ORB

#### Harris Corner Detector (1988)

The Harris corner detector defines a corner as **a point where the intensity changes significantly in all directions when an image patch is shifted**.

The intensity change when the window around a point $(x, y)$ in image $I$ is shifted by $(\Delta u, \Delta v)$:

$$E(\Delta u, \Delta v) = \sum_{x, y} w(x, y) [I(x + \Delta u, y + \Delta v) - I(x, y)]^2$$

$w(x, y)$ is a Gaussian window. Applying a first-order Taylor expansion:

$$E(\Delta u, \Delta v) \approx \begin{bmatrix} \Delta u & \Delta v \end{bmatrix} \mathbf{M} \begin{bmatrix} \Delta u \\ \Delta v \end{bmatrix}$$

Here, the **structure tensor** (or second moment matrix) $\mathbf{M}$ is:

$$\mathbf{M} = \sum_{x, y} w(x, y) \begin{bmatrix} I_x^2 & I_x I_y \\ I_x I_y & I_y^2 \end{bmatrix}$$

$I_x, I_y$ are the image gradients in the x and y directions. Depending on the eigenvalues $(\lambda_1, \lambda_2)$ of $\mathbf{M}$:

- $\lambda_1 \approx 0, \lambda_2 \approx 0$: flat region (no intensity change)
- $\lambda_1 \gg \lambda_2 \approx 0$: edge (change in only one direction)
- Both $\lambda_1, \lambda_2$ large: corner (change in all directions)

Instead of computing the eigenvalues directly, Harris defines a **corner response function**:

$$R = \det(\mathbf{M}) - k \cdot \text{tr}(\mathbf{M})^2 = \lambda_1 \lambda_2 - k(\lambda_1 + \lambda_2)^2$$

Here, $k$ is typically 0.04-0.06. Points with $R > \text{threshold}$ are selected as corners, and non-maximum suppression is applied.

**Limitations of Harris**: It is rotation invariant but **not scale invariant**. As the camera moves closer, a corner may appear as an edge. This limitation motivated the scale-space approach of SIFT.

```python
import cv2
import numpy as np

# Harris Corner Detection
img = cv2.imread('scene.jpg', cv2.IMREAD_GRAYSCALE)
img_float = np.float32(img)

# blockSize: neighborhood size, ksize: Sobel kernel, k: Harris parameter
harris_response = cv2.cornerHarris(img_float, blockSize=2, ksize=3, k=0.04)

# Non-maximum suppression & threshold
corners = harris_response > 0.01 * harris_response.max()
```

#### FAST (Features from Accelerated Segment Test, 2006)

FAST is a detector that pursues **extreme speed** over the accuracy of Harris. It was proposed by [Rosten & Drummond (2006)](https://arxiv.org/abs/0810.2434) to meet the real-time demand of detecting keypoints at tens of FPS in robot vision.

The algorithm is surprisingly simple:

1. Place 16 pixels on a circle of radius 3 centered on a candidate pixel $p$ (Bresenham circle).
2. If $N$ consecutive pixels on the circle (typically $N=12$) are all brighter or all darker than $p$, then $p$ is a corner.
3. Fast reject: first check only the 4 pixels at positions 1, 5, 9, 13 — if at least 3 of them fail the condition, immediately reject.

$$\text{FAST condition: } \exists \text{ contiguous arc of } N \text{ pixels on circle, all } > I_p + t \text{ or all } < I_p - t$$

Here, $t$ is the intensity threshold.

**Decision tree learning**: FAST additionally uses machine learning (ID3 decision tree) to optimize the inspection order. It learns which pixels should be inspected first to reject non-corners as quickly as possible.

**Limitations of FAST**: It has no orientation or scale information, and does not generate a descriptor. ORB addresses these shortcomings.

```python
# FAST Corner Detection
fast = cv2.FastFeatureDetector_create(threshold=20, nonmaxSuppression=True)
keypoints = fast.detect(img, None)
print(f"Detected {len(keypoints)} keypoints")
```

#### ORB (Oriented FAST and Rotated BRIEF, 2011)

ORB was proposed by [Rublee et al. (2011)](https://ieeexplore.ieee.org/document/6126544) to simultaneously address the patent and speed problems of SIFT/SURF. It is a combination that **adds orientation to the FAST detector and rotation invariance to the BRIEF descriptor**.

**oFAST (Oriented FAST)**:
- Orientations are assigned to keypoints detected by FAST using the **intensity centroid** method.
- Image moments of the patch around a keypoint are computed:

$$m_{pq} = \sum_{x, y} x^p y^q I(x, y)$$

- Centroid: $\mathbf{C} = (m_{10}/m_{00}, m_{01}/m_{00})$
- Orientation: $\theta = \text{atan2}(m_{01}, m_{10})$

**rBRIEF (Rotated BRIEF)**:
- BRIEF generates a binary descriptor by comparing the intensities of random point pairs $(x_i, y_i)$ within a patch:

$$\tau(\mathbf{p}; x_i, y_i) = \begin{cases} 1 & \text{if } I(\mathbf{p}, x_i) < I(\mathbf{p}, y_i) \\ 0 & \text{otherwise} \end{cases}$$

- 256 comparisons → 256-bit binary descriptor.
- ORB rotates the comparison point pairs according to the keypoint orientation $\theta$ to secure **rotation invariance**.
- In addition, the comparison point pairs are selected greedily to minimize their correlation, increasing discriminability.

**Multi-scale**: An image pyramid (typically 8 levels) is constructed and FAST is performed at each level, approximating scale invariance.

ORB is **the core feature of the ORB-SLAM series**, and thanks to its binary descriptor, matching is performed with Hamming distance, making it very fast.

```python
# ORB Detection & Description
orb = cv2.ORB_create(nfeatures=1000)
keypoints, descriptors = orb.detectAndCompute(img, None)
# descriptors.shape: (N, 32) — 256-bit = 32 bytes
```

### 5.2.2 Blob Detection: SIFT → SURF

#### SIFT (Scale-Invariant Feature Transform, 2004)

SIFT, proposed by [Lowe (2004)](https://link.springer.com/article/10.1023/B:VISI.0000029664.99615.94), is a keypoint detection and description algorithm that is **invariant to scale, rotation, and illumination changes**. It was the de facto standard for feature matching for 20 years, and since the patent expired in 2020 it can be used freely.

**Stage 1 — Scale-Space Extrema Detection (DoG)**:

A scale-space is constructed by blurring the image with Gaussians at various scales:

$$L(x, y, \sigma) = G(x, y, \sigma) * I(x, y)$$

Here, $G(x, y, \sigma) = \frac{1}{2\pi\sigma^2} \exp\left(-\frac{x^2 + y^2}{2\sigma^2}\right)$.

As an efficient approximation of the Laplacian of Gaussian (LoG), the **Difference of Gaussians (DoG)** is used:

$$D(x, y, \sigma) = L(x, y, k\sigma) - L(x, y, \sigma) \approx (k-1)\sigma^2 \nabla^2 G$$

In the DoG images, extrema are found by comparing against 26 neighbors (spatial: 8 neighbors + scale: 9 each above and below).

**Stage 2 — Keypoint Localization**:

Sub-pixel/sub-scale localization in scale-space using a Taylor expansion:

$$D(\mathbf{x}) = D + \frac{\partial D}{\partial \mathbf{x}}^\top \mathbf{x} + \frac{1}{2} \mathbf{x}^\top \frac{\partial^2 D}{\partial \mathbf{x}^2} \mathbf{x}$$

Refined position of the extremum: $\hat{\mathbf{x}} = -\frac{\partial^2 D}{\partial \mathbf{x}^2}^{-1} \frac{\partial D}{\partial \mathbf{x}}$

Low-contrast keypoint removal: remove if $|D(\hat{\mathbf{x}})| < 0.03$.
Edge response removal: unstable extrema on edges are removed by the ratio of eigenvalues of the Hessian matrix.

**Stage 3 — Orientation Assignment**:

The gradient magnitudes and orientations around a keypoint are computed, and a 36-bin orientation histogram is built (Gaussian weighted, $\sigma = 1.5 \times$ keypoint scale). If there is a peak at least 80% of the maximum peak, a separate keypoint is created for that orientation as well to secure **rotation invariance**.

**Stage 4 — Keypoint Descriptor**:

The $16 \times 16$ region around a keypoint is divided into 16 blocks of $4 \times 4$. An 8-bin orientation histogram is built for each block:

$$\text{Descriptor} = 4 \times 4 \times 8 = 128\text{-dimensional vector}$$

After L2 normalization, values above 0.2 are clipped and renormalized, making it robust to nonlinear illumination changes.

```python
# SIFT Detection & Description
sift = cv2.SIFT_create(nfeatures=2000)
keypoints, descriptors = sift.detectAndCompute(img, None)
# descriptors.shape: (N, 128) — 128-dimensional float32
```

#### SURF (Speeded-Up Robust Features, 2006)

SURF was proposed by [Bay et al. (2006)](https://link.springer.com/chapter/10.1007/11744023_32) to address the speed issue of SIFT. Key ideas:

- Approximate LoG with box filters using the **integral image**. Any box filter size can be computed in O(1).
- Use the **Hessian determinant** as the detection criterion (instead of DoG):

$$\det(\mathbf{H}) = D_{xx} D_{yy} - (0.9 \cdot D_{xy})^2$$

- A 64-dimensional descriptor (half the dimensionality of SIFT's 128): sums and sums-of-absolute-values of Haar wavelet responses.
- 3-7× faster than SIFT with comparable accuracy.

Due to patent issues SURF is rarely used in recent practice; in real-time applications ORB is preferred, while for accuracy-critical applications SIFT or learning-based methods are preferred.

### 5.2.3 Binary Descriptors: BRIEF, ORB, BRISK

Binary descriptors generate a 0/1 bit string by comparing the intensities of point pairs within a patch. Since matching is done with **Hamming distance**, they are tens of times faster than float descriptors.

| Descriptor | Bits | Features |
|-----------|--------|------|
| BRIEF | 128/256/512 | Random point pairs, not rotation invariant |
| ORB | 256 | Learned point pairs, rotation invariant |
| BRISK | 512 | Concentric sampling, scale invariant |

BRIEF's comparison operation:

$$b_i = \begin{cases} 1 & \text{if } I(\mathbf{p}_i) < I(\mathbf{q}_i) \\ 0 & \text{otherwise} \end{cases}, \quad \text{descriptor} = \sum_{1 \le i \le n} 2^{i-1} b_i$$

Hamming distance is computed with XOR + popcount in a single CPU instruction:

$$d_H(\mathbf{a}, \mathbf{b}) = \text{popcount}(\mathbf{a} \oplus \mathbf{b})$$

### 5.2.4 Technical Lineage: A History of the Accuracy vs. Speed Trade-off

The history of traditional feature points is a continuous effort to optimize **the trade-off between accuracy and speed**:

```
Harris (1988)     — Mathematical definition of corner detection
    ↓ need for scale invariance
SIFT (2004)       — scale-space + 128D float descriptor (accurate but slow)
    ↓ speed improvement
SURF (2006)       — integral image + 64D (3-7× faster, still float)
    ↓ real-time demand
FAST (2006)       — extreme-speed detection (no descriptor)
    ↓ unifying detection+description
ORB (2011)        — oFAST + rBRIEF, 256-bit binary (Hamming matching)
```

This trade-off continues in the deep learning era as well, and SuperPoint aimed to achieve SIFT-level accuracy at ORB-level speed.

---

## 5.3 Traditional Matching & Outlier Rejection

After detecting keypoints and extracting descriptors, a matching stage is needed **to decide which feature-point pairs actually correspond to the same 3D point**.

### 5.3.1 Brute-Force Matching

The simplest method. Every descriptor in image A is compared with every descriptor in image B, and the closest one is matched.

- Float descriptors (SIFT): L2 distance
- Binary descriptors (ORB): Hamming distance
- Time complexity: $O(N \cdot M \cdot D)$ — $N, M$ are the number of keypoints per image, $D$ is the descriptor dimension.

```python
# Brute-Force Matching
bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)  # SIFT
# bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)  # ORB
matches = bf.match(desc1, desc2)
matches = sorted(matches, key=lambda x: x.distance)
```

`crossCheck=True` applies the **mutual nearest neighbor** condition: a match is accepted only when the nearest neighbor from A to B and from B to A agree.

### 5.3.2 FLANN (Fast Library for Approximate Nearest Neighbors)

For large descriptor sets, brute-force is impractical, so **approximate nearest neighbor (ANN) search** is used. FLANN automatically selects among kd-tree, randomized kd-tree, hierarchical k-means, etc.

```python
# FLANN Matching for SIFT
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)  # limit on number of nodes explored

flann = cv2.FlannBasedMatcher(index_params, search_params)
matches = flann.knnMatch(desc1, desc2, k=2)  # k=2 for ratio test
```

### 5.3.3 Lowe's Ratio Test

A match filtering technique proposed by Lowe (2004) in the SIFT paper. A match is accepted only if the **ratio of the nearest distance to the second-nearest distance** is below a threshold:

$$\frac{d(\mathbf{f}, \mathbf{f}_1)}{d(\mathbf{f}, \mathbf{f}_2)} < \tau$$

Here, $\mathbf{f}_1, \mathbf{f}_2$ are the nearest and second-nearest descriptors respectively, and $\tau$ is typically 0.7-0.8.

Intuition: a correct match has a clearly close nearest neighbor, so $d_1 \ll d_2$. An incorrect match has multiple similar candidates, so $d_1 \approx d_2$.

```python
# Ratio test (Lowe's)
good_matches = []
for m, n in matches:  # m: best, n: second best
    if m.distance < 0.75 * n.distance:
        good_matches.append(m)
```

### 5.3.4 The RANSAC Family: RANSAC → PROSAC → MAGSAC++

Even after the matching stage, **outliers (incorrect matches)** remain. Estimating a geometric model (fundamental/essential matrix) while simultaneously rejecting outliers is the role of the RANSAC family.

#### RANSAC (Random Sample Consensus, 1981)

A robust estimation paradigm proposed by [Fischler & Bolles (1981)](https://dl.acm.org/doi/10.1145/358669.358692):

1. Randomly sample the minimum $n$ matches required for the model (e.g., fundamental matrix with 8, 7, or 5 points)
2. Estimate the model from the sampled points
3. Form a consensus set of points (inliers) whose error to the model is within a threshold $t$ over all matches
4. Select the model with the largest consensus set
5. Re-estimate the model using all inliers

**Required number of iterations**:

$$k = \frac{\log(1 - p)}{\log(1 - w^n)}$$

- $p$: desired success probability (typically 0.99)
- $w$: inlier ratio
- $n$: minimum number of points required by the model

Example: with 50% inliers, an 8-point model, and 99% confidence → $k \approx 1177$ iterations.

```python
# Estimate the fundamental matrix with RANSAC
pts1 = np.float32([kp1[m.queryIdx].pt for m in good_matches])
pts2 = np.float32([kp2[m.trainIdx].pt for m in good_matches])

F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC,
                                  ransacReprojThreshold=3.0,
                                  confidence=0.99)
inlier_matches = [m for m, flag in zip(good_matches, mask.ravel()) if flag]
```

#### PROSAC (Progressive Sample Consensus, 2005)

Proposed by Chum & Matas. While RANSAC samples uniformly at random, PROSAC **sorts matches by quality (e.g., descriptor distance)** and progressively samples from the top.

Intuition: better matches have higher probability of being inliers, so trying the model on the top matches first finds a good model more quickly. Tens of times faster convergence than RANSAC.

#### MAGSAC / MAGSAC++ (2019/2020)

Proposed by Barath et al. One of the core problems of RANSAC is the **manual setting of the threshold $t$**. MAGSAC automates this:

- Marginalize the quality of the model over all possible thresholds $\sigma$:

$$Q(\theta) = \int_0^{\sigma_{\max}} q(\theta, \sigma) f(\sigma) d\sigma$$

Here, $q(\theta, \sigma)$ is the quality of the model $\theta$ at threshold $\sigma$, and $f(\sigma)$ is the prior over thresholds.

- MAGSAC++ implements this more efficiently and adds a weighted least-squares fit based on $\sigma$-consensus.
- As a result, sensitivity to threshold selection is substantially reduced.

```python
# OpenCV's USAC (includes MAGSAC++)
F, mask = cv2.findFundamentalMat(
    pts1, pts2,
    cv2.USAC_MAGSAC,           # use MAGSAC++
    ransacReprojThreshold=1.0,  # less sensitive
    confidence=0.999,
    maxIters=10000
)
```

### 5.3.5 Fundamental / Essential Matrix Estimation

The process of estimating the geometric relationship between two cameras from 2D-2D matches:

**Fundamental Matrix** ($\mathbf{F}$, 7 DOF):
- Used when intrinsic parameters are unknown
- 8-point algorithm: solve the linear system from at least 8 correspondences and enforce the rank-2 constraint
- 7-point algorithm: at least 7 correspondences, up to 3 solutions as roots of a cubic polynomial

**Essential Matrix** ($\mathbf{E}$, 5 DOF):
- Used when intrinsic parameters are known
- $\mathbf{E} = [\mathbf{t}]_\times \mathbf{R}$, $\mathbf{E} = \mathbf{K}_2^\top \mathbf{F} \mathbf{K}_1$
- 5-point algorithm (Nistér, 2004): at least 5 correspondences, up to 10 solutions via a 10th-order polynomial

Used in combination with RANSAC:

```python
# Essential matrix estimation (calibrated camera)
E, mask = cv2.findEssentialMat(pts1, pts2, cameraMatrix=K,
                                method=cv2.RANSAC,
                                prob=0.999, threshold=1.0)

# Recover R, t from the essential matrix
_, R, t, mask_pose = cv2.recoverPose(E, pts1, pts2, cameraMatrix=K)
```

### 5.3.6 Technical Lineage: The Evolution of Robust Estimation

```
Least Squares (vulnerable to outliers)
    ↓ handling outliers
RANSAC (1981)   — the first robust estimation paradigm
    ↓ exploiting prior information
PROSAC (2005)   — progressive sampling based on match quality
    ↓ threshold automation
MAGSAC++ (2020) — threshold-free robust estimation
    ↓ eliminating learning-based routines
GeoTransformer (2022) — direct transformation estimation without RANSAC
```

---

## 5.4 Mutual Information & Intensity-Based Registration

### 5.4.1 Definition and Intuition of MI

**Mutual Information (MI)** is an information-theoretic measure of how much information two random variables share about each other.

Mutual information of two random variables $X, Y$:

$$I(X; Y) = \sum_{x \in X} \sum_{y \in Y} p(x, y) \log \frac{p(x, y)}{p(x) p(y)}$$

For continuous variables:

$$I(X; Y) = \int \int p(x, y) \log \frac{p(x, y)}{p(x) p(y)} \, dx \, dy$$

Equivalent expression using entropy:

$$I(X; Y) = H(X) + H(Y) - H(X, Y)$$

Here:
- $H(X) = -\sum_x p(x) \log p(x)$: entropy of $X$
- $H(X, Y) = -\sum_{x,y} p(x, y) \log p(x, y)$: joint entropy

**Intuition**: when two images are correctly aligned, knowing a pixel value in one image allows better prediction of the corresponding pixel in the other. That is, $I(X; Y)$ is maximized. When alignment drifts, the relationship between the two images weakens and $I(X; Y)$ decreases.

Key property: MI measures **nonlinear statistical dependence** between two variables. This means it can capture relationships that a simple correlation coefficient cannot.

### 5.4.2 MI-Based Multi-Modality Registration

The true value of MI lies in its ability **to register sensor data across different modalities**. This is related to the fact that the method was originally developed for medical imaging to register CT-MRI.

Why does MI work across multiple modalities?

- CT and MRI of the same object have completely different intensity distributions (bone can be bright in CT and dark in MRI).
- A simple intensity difference (SSD) or correlation (NCC) cannot model such nonlinear relationships.
- MI measures only the **statistical dependence** between intensity values, so it works for monotonic or non-monotonic relationships between intensities.

Application in robotics: **camera-LiDAR registration**. LiDAR intensity images and camera images measure completely different physical quantities, but they reflect the same structures in the same scene, so MI is high.

### 5.4.3 Practical Computation of MI and NMI

When applying MI to image registration, the probability distributions $p(x), p(y), p(x, y)$ are estimated from a **joint histogram**.

When images $A$ and $B$ are aligned by the transformation $T$:
1. At each common location $(u, v)$, collect intensity pairs $A(u, v)$ and $B(T(u, v))$.
2. Estimate the joint distribution $p(a, b)$ from a 2D histogram.
3. Marginal distributions $p(a), p(b)$ are the row/column sums of the histogram.
4. Compute MI.

**Normalized Mutual Information (NMI)** normalizes MI to reduce sensitivity to overlap area size:

$$NMI(A, B) = \frac{H(A) + H(B)}{H(A, B)}$$

Or:

$$NMI(A, B) = \frac{2 \cdot I(A; B)}{H(A) + H(B)}$$

NMI remains stable when the overlap area varies, so it is preferred over MI in practice.

### 5.4.4 MI Gradient Computation

To use MI as the objective for registration, we must compute the gradient with respect to the transformation parameters.

The gradient of MI with respect to a transformation $T_\xi$ (parameters $\xi$):

$$\frac{\partial I}{\partial \xi} = \sum_{a, b} \frac{\partial p(a, b)}{\partial \xi} \left(1 + \log \frac{p(a, b)}{p(a) p(b)}\right)$$

When the joint histogram is discrete the gradient does not exist, so differentiable histogram estimation methods based on a **Parzen window (kernel density estimation)** or **B-splines** are used.

In practice, derivative-free optimization such as **Nelder-Mead simplex** is often used instead of gradient-based optimization (used in the calibration tool of Koide et al., 2023).

### 5.4.5 Why MI Is Used for Calibration

In the targetless camera-LiDAR calibration of Ch.3, MI (or NMI, NID) is used as the core cost function:

1. Project the LiDAR point cloud onto the camera image with the current extrinsic estimate
2. Compute the MI between the projected LiDAR intensity and the camera pixel intensity
3. Adjust the extrinsic parameters to maximize MI

**Normalized Information Distance (NID)** is an MI-based distance metric used in the general-purpose calibration tool of Koide et al. (2023):

$$NID(A, B) = 1 - \frac{I(A; B)}{H(A, B)} = \frac{H(A|B) + H(B|A)}{H(A, B)}$$

NID takes values from 0 (full dependence) to 1 (full independence) and satisfies the properties of a metric space.

```python
import numpy as np
from sklearn.metrics import mutual_info_score

def compute_mi(img_a, img_b, bins=256):
    """Compute the mutual information of two images."""
    # Joint histogram
    hist_2d, _, _ = np.histogram2d(
        img_a.ravel(), img_b.ravel(), bins=bins
    )
    # Joint probability
    pxy = hist_2d / hist_2d.sum()
    px = pxy.sum(axis=1)
    py = pxy.sum(axis=0)
    
    # MI = H(X) + H(Y) - H(X,Y)
    hx = -np.sum(px[px > 0] * np.log(px[px > 0]))
    hy = -np.sum(py[py > 0] * np.log(py[py > 0]))
    hxy = -np.sum(pxy[pxy > 0] * np.log(pxy[pxy > 0]))
    
    return hx + hy - hxy

def compute_nid(img_a, img_b, bins=256):
    """Compute the Normalized Information Distance."""
    mi = compute_mi(img_a, img_b, bins)
    hist_2d, _, _ = np.histogram2d(
        img_a.ravel(), img_b.ravel(), bins=bins
    )
    pxy = hist_2d / hist_2d.sum()
    hxy = -np.sum(pxy[pxy > 0] * np.log(pxy[pxy > 0]))
    
    return 1.0 - mi / hxy if hxy > 0 else 1.0
```

---

## 5.5 Learning-Based Feature Detection & Description

Traditional feature points rely on **low-level visual cues** such as intensity gradients, corners, and blobs. For this reason they are vulnerable to illumination changes, viewpoint changes, and weather changes. Deep learning has overcome this limitation by learning more robust feature representations from large amounts of data.

### 5.5.1 SuperPoint (2018): Self-Supervised Integration of Detection and Description

[DeTone et al. (2018)](https://arxiv.org/abs/1712.07629)'s SuperPoint is the first practical deep learning pipeline to **unify keypoint detection and descriptor extraction into a single network**.

#### Homographic Adaptation: The Key Training Strategy

The most important technical contribution of SuperPoint is **a method for training a highly repeatable keypoint detector without labels**.

1. Apply 100+ random homographies to a single image
2. Detect keypoints in each transformed image
3. Inversely transform the detection results back to the original coordinate frame and aggregate
4. Adopt only the points **consistently detected across transformations** as pseudo ground-truth

Through this process, a highly repeatable keypoint detector is trained without manual labels.

#### Two-Stage Training: MagicPoint → SuperPoint

- **Stage 1**: Pretrain a corner/junction detector (**MagicPoint**) on the Synthetic Shapes dataset composed of synthetic geometric figures (triangles, rectangles, line segments, etc.).
- **Stage 2**: Apply MagicPoint with Homographic Adaptation to real images such as MS-COCO to generate pseudo ground-truth in real scenes and train SuperPoint.

#### Architecture

A VGG-style encoder (shared backbone) → branches into two decoder heads:

**Interest Point Decoder**: 
- Divide the input image into a grid of 8×8 cells
- Perform a 65-channel (64 positions + 1 "no interest point") softmax in each cell
- Generate a pixel-level keypoint heatmap by directly predicting the position within the cell

**Descriptor Decoder**: 
- Output a 256-dimensional descriptor map from the shared backbone's feature map
- Sample at detected keypoint positions using bi-cubic interpolation
- Apply L2 normalization

#### Training Loss

- Keypoint detection: cross-entropy loss
- Descriptor: since correspondences are known via the homography, hinge loss on positive/negative pairs:

$$L_{desc} = \sum_{(i,j) \in \text{pos}} \max(0, m_p - \mathbf{d}_i^\top \mathbf{d}_j) + \sum_{(i,j) \in \text{neg}} \max(0, \mathbf{d}_i^\top \mathbf{d}_j - m_n)$$

Here, $m_p, m_n$ are the positive/negative margins.

**Performance**: detection and description are performed jointly in a single forward pass. About 70 FPS on a 640×480 image (GPU).

```python
import torch
# SuperPoint usage example (hloc / kornia)
from kornia.feature import SuperPoint as KorniaSuperPoint

# Load model
sp = KorniaSuperPoint(max_num_keypoints=2048)
sp = sp.eval()

# Inference
with torch.no_grad():
    img_tensor = torch.from_numpy(img).float().unsqueeze(0).unsqueeze(0) / 255.0
    pred = sp(img_tensor)
    keypoints = pred['keypoints']        # (1, N, 2)
    descriptors = pred['descriptors']    # (1, 256, N)
    scores = pred['scores']              # (1, N)
```

### 5.5.2 D2-Net (2019): Detect-and-Describe Jointly

[D2-Net (Dusmanu et al., 2019)](https://arxiv.org/abs/1905.03561) is a method that integrates detection and description even more aggressively. Whereas SuperPoint still separates the detection head and the description head, D2-Net **performs detection and description simultaneously from the same feature map**.

Key idea: using the intermediate feature map $\mathbf{F} \in \mathbb{R}^{H \times W \times C}$ of VGG16:
- **Detection**: take the maximum value along the channel axis at each position and apply spatial NMS to select keypoints
- **Description**: use the $C$-dimensional vector at the same location as the descriptor

Advantage: uses higher-level semantic features, so robust to large appearance changes.
Disadvantage: detection repeatability may be lower than SuperPoint, and localization is only possible up to 1/4 of the input resolution.

### 5.5.3 R2D2 (2019): Reliable and Repeatable Detector-Descriptor

[R2D2 (Revaud et al., 2019)](https://arxiv.org/abs/1906.06195) analyzes the limitations of SuperPoint and D2-Net and proposes a method that **explicitly trains for repeatability and reliability**.

- **Repeatability**: is the same point detected across various viewpoints?
- **Reliability**: is the descriptor of the detected point useful for matching? (A keypoint in a textureless area may be repeatable but useless for matching.)

R2D2 predicts the two as separate confidence maps and multiplies them to determine the final keypoint score.

### 5.5.4 DISK (2020)

[DISK (Tyszkiewicz et al., 2020)](https://arxiv.org/abs/2006.13566) trains keypoint detection from a **reinforcement learning** perspective. The detector is trained by rewarding successful matches and penalizing failed ones.

Key differentiator: by directly optimizing matching accuracy, it takes one more step toward end-to-end optimization of detection and matching.

### 5.5.5 Advantages Over Traditional Methods: Improved Illumination/Viewpoint Invariance

Specific scenarios in which learning-based feature points have an edge over traditional methods:

| Scenario | SIFT/ORB Limitations | SuperPoint/D2-Net Improvements |
|---------|-------------|---------------------|
| Extreme illumination change (day-night) | DoG/gradient-based detection fails | Learned features capture high-level structures |
| Wide viewpoint change | Limitations of affine approximation | Viewpoint invariance learned from large training data |
| Repetitive patterns | Similar descriptors lead to ambiguous matches | Context information captured for discrimination |
| Motion blur | Weakened gradients → detection failure | CNN learns robustness to blur |

However, learning-based methods also have limitations: they depend on the training data domain and performance can degrade in entirely new environments (e.g., underwater, Mars).

---

## 5.6 Learning-Based Feature Matching

### 5.6.1 SuperGlue (2020): Attention-Based Matching

[Sarlin et al. (2020)](https://arxiv.org/abs/1911.11763)'s SuperGlue recast keypoint matching as a learnable problem using **graph neural networks (GNN) and the attention mechanism**. It is the first production-grade system to replace traditional nearest-neighbor matching with learning-based matching.

#### Problem Definition: Partial Assignment

Between the keypoint sets $\mathcal{A} = \{(\mathbf{p}_i, \mathbf{d}_i)\}_{i=1}^{N}$ and $\mathcal{B} = \{(\mathbf{p}_j, \mathbf{d}_j)\}_{j=1}^{M}$ extracted from two images, we seek correspondences, but **not every keypoint has a correspondence**. To handle this, a virtual node called a "dustbin" is added to explicitly handle unmatchable points.

#### Keypoint Encoder

The keypoint position $(x, y)$ and detection confidence $c$ are embedded via an MLP and added to the descriptor vector:

$$\mathbf{f}_i^{(0)} = \mathbf{d}_i + \text{MLP}_{\text{enc}}([\mathbf{p}_i, c_i])$$

In this way, geometric information is baked into the feature.

#### Attentional Graph Neural Network

Keypoints are represented as nodes of a graph, and self-attention and cross-attention are applied alternately:

**Self-Attention (intra-image)**: learns relationships among keypoints within the same image. For example, it captures structural information such as building corners being collinear.

$$\text{message}_i^{\text{self}} = \sum_{j \in \mathcal{A}} \text{softmax}\left(\frac{\mathbf{q}_i^\top \mathbf{k}_j}{\sqrt{d}}\right) \mathbf{v}_j$$

**Cross-Attention (inter-image)**: learns relationships between keypoints across the two images. It infers by attention "which keypoint in the other image is this keypoint similar to?"

$$\text{message}_i^{\text{cross}} = \sum_{j \in \mathcal{B}} \text{softmax}\left(\frac{\mathbf{q}_i^\top \mathbf{k}_j}{\sqrt{d}}\right) \mathbf{v}_j$$

This is alternated $L$ times (9 in the paper), progressively refining the matching information.

#### Matching via Optimal Transport

The final matching score matrix is computed as the inner product of the GNN outputs, and the **Sinkhorn algorithm** (a soft form of the Hungarian algorithm) is applied.

The score matrix $\mathbf{S} \in \mathbb{R}^{(N+1) \times (M+1)}$ (including dustbin):

$$S_{ij} = \langle \mathbf{f}_i^{(L)}, \mathbf{f}_j^{(L)} \rangle, \quad S_{i,M+1} = S_{N+1,j} = z$$

Here, $z$ is a learnable dustbin score.

Sinkhorn normalization is applied iteratively (about 100 times):

$$\mathbf{S} \leftarrow \text{row-normalize}(\mathbf{S}), \quad \mathbf{S} \leftarrow \text{col-normalize}(\mathbf{S})$$

After convergence, the soft assignment matrix is thresholded to produce the final matches.

#### Training

End-to-end training by maximizing the negative log-likelihood over ground-truth correspondences (generated from a homography or from relative pose + depth map):

$$L = -\sum_{(i,j) \in \mathcal{M}} \log \hat{P}_{ij} - \sum_{i \in \mathcal{U}_A} \log \hat{P}_{i, M+1} - \sum_{j \in \mathcal{U}_B} \log \hat{P}_{N+1, j}$$

Here, $\mathcal{M}$ is the set of matched pairs and $\mathcal{U}_A, \mathcal{U}_B$ are the sets of unmatched keypoints.

```python
# SuperGlue usage example (hloc)
from hloc import match_features, extract_features
from hloc.utils.io import list_h5_names

# SuperPoint feature extraction
feature_conf = extract_features.confs['superpoint_aachen']
features_path = extract_features.main(feature_conf, images_dir)

# SuperGlue matching
match_conf = match_features.confs['superglue']
matches_path = match_features.main(match_conf, pairs_path, feature_conf['output'], features_path)
```

#### Position in the Technical Lineage

If SuperPoint learned detection+description, SuperGlue **learned the matching stage**. With this, **all three stages** of the detect-then-describe-then-match pipeline **were replaced by deep learning**. However, the sequential three-stage structure of the pipeline itself is still maintained. It is LoFTR that broke this structural limitation.

### 5.6.2 LightGlue (2023): Making SuperGlue Efficient

[Lindenberger et al. (2023)](https://arxiv.org/abs/2306.13643)'s LightGlue maintains SuperGlue's accuracy while drastically improving speed through **adaptive computation**.

#### Diagnosis of SuperGlue's Problems

- Always performs a fixed 9 GNN layers and 100 Sinkhorn iterations → unnecessarily many operations even for easy matches.
- $O(N^2)$ attention is repeated over the number of keypoints $N$, so it slows down drastically as the number of keypoints grows.

#### Key Improvements: Adaptive Depth & Width

**Adaptive Depth (early layer exit)**:
- A lightweight classifier (MLP) after each layer predicts matching confidence.
- If confidence is sufficiently high, the network terminates early.
- Easy image pairs are processed in only 2-3 layers, while difficult pairs use all 9 layers.

**Adaptive Width (keypoint pruning)**:
- Keypoints deemed unmatchable are pruned in intermediate layers.
- The sequence length of attention gradually decreases, lightening the computation of later layers.

**Removing Sinkhorn**: 
Instead of optimal transport, matching is done with simple **dual-softmax + mutual nearest neighbor**. The iteration cost of Sinkhorn is eliminated entirely with virtually no performance degradation.

$$P_{ij} = \text{softmax}_j(S_{ij}) \cdot \text{softmax}_i(S_{ij})$$

#### Training Strategy

- During training, adaptive termination is not used; the full network is run and supervision is applied to the output of each layer (**deep supervision**).
- Adaptive termination is activated only at inference.
- Designed for general compatibility with various local feature detectors including SuperPoint, DISK, and ALIKED.

**Performance**: at accuracy on par with SuperGlue, it is **3-5× faster**. On easy pairs, speedups of up to 10× or more are achieved.

```python
# LightGlue usage example (kornia / hloc)
from kornia.feature import LightGlue as KorniaLightGlue

# SuperPoint + LightGlue combination
lg = KorniaLightGlue(features='superpoint')
lg = lg.eval()

with torch.no_grad():
    # pred0, pred1: SuperPoint outputs
    matches = lg({'image0': pred0, 'image1': pred1})
    # matches['matches']: (K, 2) — pairs of matched keypoint indices
    # matches['scores']: (K,) — match confidence
```

### 5.6.3 Technical Flow: Deep-Learning Replacement of the Separated Detect → Describe → Match Pipeline

```
[Traditional pipeline]
SIFT detect → SIFT descriptor → BF/FLANN + ratio test
                                        ↓
[Learning-based replacements]
SuperPoint detect+describe → SuperGlue attention matching → LightGlue efficiency
    (2018)                       (2020)                        (2023)
```

The core narrative of this evolution: **each stage of the pipeline is replaced one by one with deep learning, while the three-stage serial structure itself is preserved**. The advantage of this structure is modularity and interpretability; the disadvantage is that a failure in the detection stage leads to failure of the entire pipeline. The next section's detector-free paradigm breaks this fundamental limitation.

---

## 5.7 Detector-Free Matching (Paradigm Shift)

### 5.7.1 Why Detector-Free Is Needed

The detect-then-match pipeline has the fundamental limitation that **matching is impossible if the detector fails to find keypoints**. This is critical in real-world environments such as:

- **Textureless regions**: white walls, concrete floors, sky — weak gradients cause corner/blob detection to fail
- **Repetitive patterns**: tiles, window grids — detection succeeds but descriptors are similar, leading to ambiguous matches
- **Wide viewpoint changes**: at extreme viewpoint differences, the appearance of local patches changes completely

The detector-free approach eliminates the detection stage and **treats every position in the image as a potential matching candidate**, overcoming this limitation.

### 5.7.2 LoFTR (2021): Coarse-to-Fine Transformer Matching

[Sun et al. (2021)](https://arxiv.org/abs/2104.00680)'s LoFTR is **a transformer-based architecture that performs dense matching between two images without a detector**, opening a new paradigm of detector-free matching.

#### Architecture

**1. Local Feature CNN**: a ResNet-18-based FPN (Feature Pyramid Network) extracts coarse (1/8 resolution) and fine (1/2 resolution) feature maps from the two images.

**2. Coarse-Level Matching (Transformer)**:

The 1/8 feature maps are flattened into 1D sequences, and a transformer module with self-attention + cross-attention applied alternately $N$ times (typically 4) is used.

Positional encoding: sinusoidal positional encoding preserves spatial information.

A score matrix is computed by inner products of the output features, and **dual-softmax** produces a confidence matrix:

$$P_{ij} = \text{softmax}_j(\mathbf{f}_i^A \cdot \mathbf{f}_j^B) \cdot \text{softmax}_i(\mathbf{f}_i^A \cdot \mathbf{f}_j^B)$$

Coarse matches are extracted by thresholding + the mutual nearest neighbor condition.

**3. Fine-Level Refinement**:

For each coarse match, a $w \times w$ window (typically 5×5) around the corresponding position is cropped from the 1/2-resolution feature map.
Cross-attention-based correlation is again performed within the window to regress the **sub-pixel-accurate** match location.

#### Linear Transformer

To reduce computation, **kernel-based linear attention** is used instead of standard softmax attention:

Standard attention: $O(N^2)$, $\text{Attn}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{softmax}(\mathbf{Q}\mathbf{K}^\top / \sqrt{d}) \mathbf{V}$

Linear attention: $O(N)$, $\text{Attn}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \phi(\mathbf{Q})(\phi(\mathbf{K})^\top \mathbf{V})$

Here, $\phi$ is an ELU-based kernel function. By changing the associativity of the matrix products, $O(N)$ complexity is achieved.

However, subsequent work revealed that standard attention has the edge in accuracy.

#### Training

Correspondences generated from ground-truth pose + depth map are used as supervision:
- Coarse level: cross-entropy loss
- Fine level: L2 regression loss
- Trained on ScanNet (indoor) and MegaDepth (outdoor) datasets

```python
# LoFTR usage example (kornia)
from kornia.feature import LoFTR as KorniaLoFTR

loftr = KorniaLoFTR(pretrained='outdoor')
loftr = loftr.eval()

with torch.no_grad():
    input_dict = {
        'image0': img0_tensor,  # (1, 1, H, W) grayscale
        'image1': img1_tensor,
    }
    result = loftr(input_dict)
    
    mkpts0 = result['keypoints0']     # (K, 2) — matched coordinates in image 0
    mkpts1 = result['keypoints1']     # (K, 2) — matched coordinates in image 1
    confidence = result['confidence'] # (K,) — match confidence
```

#### Position in the Technical Lineage

LoFTR is a **paradigm shift**. It breaks away from the detect-then-match pipeline that flowed from SuperPoint to SuperGlue and removes the detector entirely. The key insight is **that the transformer's attention can perform detection and matching simultaneously**. Since every position in one image communicates directly (via cross-attention) with every position in the other, matching is possible without a separate detection stage.

### 5.7.3 QuadTree Attention: Making LoFTR Efficient

LoFTR's coarse-level transformer flattens every location in the image into a sequence, so the sequence length grows rapidly as resolution increases. QuadTree Attention (Tang et al., 2022) addresses this.

Key idea: perform attention hierarchically.

1. Perform full attention at the coarsest resolution
2. Select only the regions with high attention scores
3. Perform the next resolution's attention only in the selected regions
4. Repeat to realize hierarchical attention that concentrates on relevant regions

This reduces complexity from $O(N^2)$ to $O(N \log N)$ while maintaining LoFTR-level accuracy.

### 5.7.4 ASpanFormer (2022): Adaptive Span Attention

ASpanFormer (Chen et al., 2022) addresses another limitation of LoFTR: **must every location have the same attention span?**

Key idea: adaptively adjust the **attention span** per location.

- Texture-rich regions: precise matching with a narrow span
- Texture-poor regions: context-aware matching with a wide span

This mitigates the matching accuracy degradation LoFTR exhibited in textureless regions.

### 5.7.5 RoMa (2024): Maturation of DINOv2 + Dense Matching

[Edstedt et al. (2024)](https://arxiv.org/abs/2305.15404)'s RoMa represents the **maturation** of detector-free matching. Through two key evolutions it substantially surpasses the LoFTR family.

#### Leveraging a Foundation Model

Unlike LoFTR, which was trained from scratch, RoMa **uses a pretrained DINOv2 ViT-Large as the feature extractor** (weights frozen).

DINOv2 is a general-purpose visual feature learned by large-scale self-supervised learning, and it already contains rich semantic information. This is a philosophical shift: "leave feature learning to the general model and only learn the matching logic."

#### Architecture

**1. Frozen DINOv2 Backbone**: extracts patch features at 1/14 resolution from a pretrained DINOv2 ViT-Large.

**2. Coarse Matching (Warp Estimation)**:
- Cross-attention (transformer decoder) is applied to DINOv2 features.
- For each location in image A, the corresponding location in image B is predicted as a **probability distribution**:

$$p(\mathbf{x}_B | \mathbf{x}_A) = \sum_{k} w_k \mathcal{N}(\mathbf{x}_B; \mu_k, \Sigma_k)$$

By predicting a probability distribution rather than a single point, the uncertainty of ambiguous matches is explicitly represented.

**3. Fine Matching (Iterative Refinement)**:
- Starting from the coarse warp, CNN-based fine-level features are used to iteratively refine.
- At each refinement step, the resolution is increased; based on the previous warp, local correlation is computed and the residual is predicted.
- The philosophy is similar to RAFT's iterative refinement, but applied to sparse-to-dense warp rather than optical flow.

**4. Certainty Estimation**: certainty is predicted together with each match, so that only high-confidence matches can be used selectively in post-processing.

#### Robust Regression

Instead of regressing the matching position with a simple L2 loss, the **negative log-likelihood** between the predicted probability distribution and the ground truth is optimized:

$$L = -\sum_{(\mathbf{x}_A, \mathbf{x}_B^*)} \log p(\mathbf{x}_B^* | \mathbf{x}_A)$$

This approach is robust to outliers: even if there are incorrect ground-truth correspondences, they are absorbed into the tail of the distribution, keeping training stable.

#### Performance

On the MegaDepth and ScanNet benchmarks it substantially surpasses LoFTR, ASpanFormer, and others. The performance improvement is especially pronounced under **wide baselines (large viewpoint changes)**.

```python
# RoMa usage example
from romatch import roma_outdoor

# Load model (includes DINOv2 backbone)
roma_model = roma_outdoor(device='cuda')
roma_model.eval()

# Matching
warp, certainty = roma_model.match(img0_path, img1_path)

# Extract only high-confidence matches
matches, certainty_scores = roma_model.sample(
    warp, certainty,
    num=5000  # maximum number of matches
)
# matches: (K, 4) — [x0, y0, x1, y1] normalized coordinates
```

#### Position in the Technical Lineage

RoMa embodies two key transitions:

1. **Leveraging a foundation model**: training from scratch → training only the matching logic on top of features from a large pretrained model
2. **Probabilistic matching**: deterministic point prediction → probability distribution prediction that explicitly handles uncertain matches

It combines RAFT's iterative refinement idea with LoFTR's detector-free mindset while introducing the powerful pretrained features of DINOv2 — a comprehensive advance.

### 5.7.6 Recent Developments: 3D-Aware Dense Matching (2024-2025)

In 2024-2025, a paradigm has emerged that goes beyond 2D matching to **directly predict 3D geometry while performing matching**.

**DUSt3R (Leroy et al., 2024)**: [DUSt3R](https://arxiv.org/abs/2312.14132) is a method that directly regresses a 3D pointmap from an arbitrary image pair without any calibration or pose information. Whereas existing matching pipelines followed the order "2D matching → 3D reconstruction," DUSt3R reverses this to **directly predict the 3D structure itself and treat correspondences as a natural byproduct obtained in 3D space**.

**MASt3R (Leroy et al., 2024)**: [MASt3R](https://arxiv.org/abs/2406.09756) adds a dense local feature head to DUSt3R, **substantially strengthening dense matching performance** along with 3D pointmap prediction. On the map-free localization benchmark it achieves a 30%p (absolute) VCRE AUC improvement over the previous best method.

**VGGT (Wang et al., 2025)**: [VGGT](https://arxiv.org/abs/2503.11651) (Visual Geometry Grounded Transformer) is the CVPR 2025 Best Paper, which **simultaneously infers camera parameters, pointmap, depth map, and 3D point tracks in a single feed-forward pass** from one or more images. With inference time under one second, it achieves accuracy exceeding existing methods that require post-processing optimization.

This trend dismantles the long-standing assumption that "matching is a 2D problem" and forms a new paradigm that **directly reasoning about 3D geometry ultimately produces more robust matching**.

---

## 5.8 3D-3D Correspondence

In parallel with the technical lineage of 2D image matching, **research on finding correspondences between 3D point clouds** has also evolved. It is central to LiDAR scan-to-scan registration, multi-session map merging, and loop closure.

### 5.8.1 FPFH (Fast Point Feature Histograms)

[FPFH (Rusu et al., 2009)](https://ieeexplore.ieee.org/document/5152473) is a handcrafted descriptor that **encodes the geometric features of a 3D point cloud as a histogram**.

#### SPFH (Simple Point Feature Histogram)

A local coordinate frame is set up between a query point $\mathbf{p}$ and its neighbor $\mathbf{p}_k$, and three angular features are computed.

From the normal vectors $\mathbf{n}_p, \mathbf{n}_k$ and the direction vector $\mathbf{d} = \mathbf{p}_k - \mathbf{p}$:

$$\mathbf{u} = \mathbf{n}_p$$
$$\mathbf{v} = \mathbf{u} \times \frac{\mathbf{d}}{\|\mathbf{d}\|}$$
$$\mathbf{w} = \mathbf{u} \times \mathbf{v}$$

Three angular features:
$$\alpha = \mathbf{v} \cdot \mathbf{n}_k, \quad \phi = \mathbf{u} \cdot \frac{\mathbf{d}}{\|\mathbf{d}\|}, \quad \theta = \text{atan2}(\mathbf{w} \cdot \mathbf{n}_k, \mathbf{u} \cdot \mathbf{n}_k)$$

Each feature is quantized into a $B$-bin histogram.

#### FPFH: An Accelerated Version of SPFH

SPFH computes features over all neighbor pairs within radius $r$, so its complexity is $O(k^2)$. FPFH approximates this to reduce it to $O(k)$:

$$\text{FPFH}(\mathbf{p}) = \text{SPFH}(\mathbf{p}) + \frac{1}{k} \sum_{i=1}^{k} \frac{1}{w_i} \text{SPFH}(\mathbf{p}_i)$$

Here, $w_i = \|\mathbf{p}_i - \mathbf{p}\|$. A 33-dimensional histogram (11 bins × 3 angles).

```python
import open3d as o3d

# Compute FPFH
pcd = o3d.io.read_point_cloud("scan.ply")
pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

fpfh = o3d.pipelines.registration.compute_fpfh_feature(
    pcd,
    o3d.geometry.KDTreeSearchParamHybrid(radius=0.25, max_nn=100)
)
# fpfh.data.shape: (33, N)
```

### 5.8.2 3DMatch (2017): The Beginning of Learned 3D Descriptors

[Zeng et al. (2017)](https://arxiv.org/abs/1603.08182)'s 3DMatch is the first system to **extract 3D matching descriptors through learning** on RGB-D data.

- **Training data**: correspondence pairs automatically generated from RGB-D reconstructions of 62 indoor scenes
- **Architecture**: a 3D CNN with a 3D TDF (Truncated Distance Function) volume as input
- **Output**: 512-dimensional local descriptor

3DMatch is the starting point of learned 3D descriptors, and it also provided a standard benchmark called the **3DMatch Benchmark**, serving as the evaluation standard for subsequent research.

### 5.8.3 FCGF (Fully Convolutional Geometric Features, 2019)

[Choy et al. (2019)](https://arxiv.org/abs/1904.09793)'s FCGF uses **sparse convolution** to extract descriptors for all points in the entire point cloud in a single forward pass.

While 3DMatch processes each keypoint's local volume individually, FCGF processes the entire point cloud at once, so it is **tens to hundreds of times faster**. With a 32-dimensional descriptor — more compact than 3DMatch's 512 dimensions — it achieved even higher accuracy.

### 5.8.4 Predator (2021): Overlap-Aware 3D Matching

[Huang et al. (2021)](https://arxiv.org/abs/2011.13005)'s Predator is an approach that **explicitly predicts the overlap region** of two point clouds.

Existing methods extract descriptors uniformly over all regions of both point clouds, but in reality only parts of them overlap. Predator:

1. **Overlap attention**: predicts the degree to which each point overlaps with the other point cloud via cross-attention
2. **Matchability score**: separately scores points useful for matching (distinctive geometric structures) within the overlap region
3. Achieves substantial gains in difficult scenarios with low overlap (10-30%)

### 5.8.5 GeoTransformer (2022): Geometric Transformer

[Qin et al. (2022)](https://arxiv.org/abs/2202.06688)'s GeoTransformer is an innovative method in 3D point cloud registration that simultaneously achieves **learning of geometric invariant features and removal of RANSAC**.

#### Keypoint-Free Superpoint Matching

Instead of repeatable keypoint detection, correspondences are found on downsampled **superpoints**, then propagated to dense points.

#### Geometric Transformer Architecture

The core is learning **geometric features invariant to rigid transformations**. Two geometric encodings are used:

**1. Pairwise Distance Encoding**:

The Euclidean distance $d_{ij} = \|\mathbf{p}_i - \mathbf{p}_j\|$ between superpoints $\mathbf{p}_i, \mathbf{p}_j$ is invariant to rigid transformations. It is used as a positional encoding:

$$\text{PE}(d_{ij}) = [\sin(\omega_1 d_{ij}), \cos(\omega_1 d_{ij}), \ldots, \sin(\omega_K d_{ij}), \cos(\omega_K d_{ij})]$$

**2. Triplet Angle Encoding**:

The angle formed by three points $\mathbf{p}_i, \mathbf{p}_j, \mathbf{p}_k$ is also invariant to rigid transformations. This angle information is used as an additional encoding to more richly capture geometric structure.

These geometric encodings are injected as attention biases into the transformer:

$$\text{Attn}_{ij} = \frac{\mathbf{q}_i^\top \mathbf{k}_j}{\sqrt{d}} + b(\text{PE}(d_{ij}))$$

#### RANSAC-Free Transformation Estimation

Because robust correspondences at the superpoint level achieve high inlier ratios, **the transformation can be estimated directly without RANSAC**. This yields a **100× speedup**.

#### Performance

On the 3DLoMatch benchmark, the inlier ratio improves by 17-30%p and registration recall by over 7%p. In particular, large improvements over existing methods in the **low-overlap (10-30%)** regime.

```python
# GeoTransformer usage example
from geotransformer.utils.pointcloud import apply_transform
import torch

# Load two point clouds (N, 3)
src_points = torch.from_numpy(src_pcd.points).float().cuda()
ref_points = torch.from_numpy(ref_pcd.points).float().cuda()

# Downsample + extract superpoints
# ... (voxel downsampling)

# GeoTransformer inference
output = model(src_points, ref_points, src_feats, ref_feats)
# output['estimated_transform']: (4, 4) transformation matrix
# output['src_corr_points'], output['ref_corr_points']: correspondences
```

### 5.8.6 Technical Lineage of 3D-3D Correspondence

```
[Handcrafted 3D Descriptors]
FPFH (2009)          — geometric angular histograms
SHOT (2010)          — orientation histograms
    ↓ learning-based
[Learned 3D Descriptors]
3DMatch (2017)       — first learned 3D descriptor + benchmark
    ↓ efficiency
FCGF (2019)          — sparse conv, processes entire point cloud at once
    ↓ overlap-aware
Predator (2021)      — overlap-aware cross-attention
    ↓ RANSAC removal
GeoTransformer (2022) — geometric invariant transformer, RANSAC-free
```

---

## 5.9 Cross-Modal Correspondence

### 5.9.1 2D-3D Matching: Camera-LiDAR

The most common cross-modal correspondence problem is **matching between 2D images and 3D point clouds**. Application scenarios:

- **Camera-LiDAR extrinsic calibration**: find the same physical point observed by both sensors to estimate the extrinsic parameters
- **Visual localization against a LiDAR map**: localize a camera image against a LiDAR map
- **Loop closure**: cross-validate camera and LiDAR observations

### 5.9.2 Image-to-Point Cloud Matching Approaches

#### Projection-Based Approach

The most straightforward method is to project the 3D point cloud onto the 2D image plane to reduce the problem to 2D-2D matching:

1. Render the LiDAR point cloud into a virtual camera view to generate depth/intensity images
2. Perform 2D matching (e.g., SuperGlue) between the generated 2D image and the camera image
3. Back-project the 2D matching results to 3D coordinates

The calibration tool of Koide et al. (2023) uses this approach: the LiDAR dense point cloud is rendered with a virtual camera, and SuperGlue is used to detect cross-modal 2D-3D correspondences with the camera image.

#### Learning-Based Direct Matching

LCD (LiDAR-Camera Descriptor): learns a common embedding space for 2D image patches and 3D point cloud patches.

P2-Net (Yu et al., 2021): learns patch-to-point matching to infer direct correspondences between 2D image patches and 3D points.

### 5.9.3 Why Cross-Modal Is Difficult: The Representation Gap

Why 2D-3D cross-modal matching is fundamentally difficult:

1. **Representation heterogeneity**: 2D images are intensity/color values on a regular grid, while 3D point clouds are irregularly distributed coordinates + intensity. The data structures themselves are different.

2. **Information asymmetry**: images provide rich texture information but lack depth, while point clouds provide accurate geometric information but have sparse texture.

3. **Density gap**: the resolution of camera images (millions of pixels) differs greatly from the density of LiDAR point clouds (tens of thousands to hundreds of thousands of points), and LiDAR density changes drastically with distance.

4. **Appearance domain gap**: even for the same object, camera albedo and LiDAR reflection intensity measure different physical quantities.

Because of these difficulties, cross-modal correspondence is still a less mature research area than unimodal (2D-2D or 3D-3D) matching. The MI-based approach (Section 5.4) is a strategy that statistically bypasses this domain gap, while the projection-based approach is a strategy that reduces the problem to the same modality.

---

## 5.10 Dense Matching & Optical Flow

The methods discussed so far have focused on **sparse correspondence**. This section covers **dense matching**, which finds correspondences for every pixel of an image.

### 5.10.1 Classical Optical Flow: Lucas-Kanade, Horn-Schunck

#### Definition of Optical Flow

Optical flow represents the **apparent motion of each pixel** in an image sequence as a 2D vector field:

$$\mathbf{u}(x, y) = (u(x, y), v(x, y))$$

**Brightness constancy assumption**:

$$I(x, y, t) = I(x + u, y + v, t + 1)$$

A first-order Taylor expansion yields the **optical flow constraint equation**:

$$I_x u + I_y v + I_t = 0$$

Here, $I_x = \frac{\partial I}{\partial x}$, $I_y = \frac{\partial I}{\partial y}$, $I_t = \frac{\partial I}{\partial t}$.

Because this single equation has two unknowns $(u, v)$, an additional constraint is needed (aperture problem).

#### Lucas-Kanade (1981)

**Local consistency assumption**: optical flow is assumed constant within a small window $\Omega$. For all pixels in the window:

$$\begin{bmatrix} I_{x_1} & I_{y_1} \\ I_{x_2} & I_{y_2} \\ \vdots & \vdots \\ I_{x_n} & I_{y_n} \end{bmatrix} \begin{bmatrix} u \\ v \end{bmatrix} = -\begin{bmatrix} I_{t_1} \\ I_{t_2} \\ \vdots \\ I_{t_n} \end{bmatrix}$$

That is, $\mathbf{A} \mathbf{u} = -\mathbf{b}$. Since this is an overdetermined system, least squares gives:

$$\begin{bmatrix} u \\ v \end{bmatrix} = (\mathbf{A}^\top \mathbf{A})^{-1} \mathbf{A}^\top (-\mathbf{b})$$

Here, $\mathbf{A}^\top \mathbf{A} = \begin{bmatrix} \sum I_x^2 & \sum I_x I_y \\ \sum I_x I_y & \sum I_y^2 \end{bmatrix}$ is identical to the structure tensor $\mathbf{M}$ of the Harris corner detector.

**Limitation**: for large displacements the local linear approximation fails. To address this, it is applied coarse-to-fine over an **image pyramid** (Pyramidal LK).

```python
# Pyramidal Lucas-Kanade (OpenCV)
lk_params = dict(
    winSize=(21, 21),
    maxLevel=3,       # pyramid levels
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
)

# Select points to track (typically FAST/Shi-Tomasi corners)
p0 = cv2.goodFeaturesToTrack(prev_gray, maxCorners=200, qualityLevel=0.3, minDistance=7)

# Lucas-Kanade tracking
p1, status, err = cv2.calcOpticalFlowPyrLK(prev_gray, next_gray, p0, None, **lk_params)
# status: whether tracking succeeded (1/0)
```

#### Horn-Schunck (1981)

**Global smoothness assumption**: optical flow is assumed to vary smoothly across the image, and the following energy function is minimized:

$$E = \iint \left[ (I_x u + I_y v + I_t)^2 + \lambda^2 (\|\nabla u\|^2 + \|\nabla v\|^2) \right] dx \, dy$$

The first term is the data term (brightness constancy) and the second is smoothness regularization. The larger $\lambda$, the smoother the flow.

Solving the Euler-Lagrange equations yields an iterative update formula. It produces a dense flow, but has the limitation of not handling discontinuities at boundaries well.

### 5.10.2 Learning-Based Optical Flow: FlowNet → RAFT → FlowFormer → UniMatch

#### FlowNet / FlowNet 2.0 (2015/2017)

[Dosovitskiy et al. (2015)](https://arxiv.org/abs/1504.06852)'s FlowNet is the first deep learning method to **directly predict optical flow with a CNN**. An encoder-decoder structure that takes two images as input and outputs the flow field.

[FlowNet 2.0 (Ilg et al., 2017)](https://arxiv.org/abs/1612.01925) stacks multiple FlowNets to substantially improve accuracy.

#### RAFT (2020): A New Standard for Optical Flow

[Teed & Deng (2020)](https://arxiv.org/abs/2003.12039)'s RAFT presents a new architectural paradigm of **4D correlation volume + iterative GRU update**, winning the ECCV 2020 Best Paper.

**Architecture in 3 stages**:

**1. Feature Encoder**: a CNN (ResNet variant) is applied to each of the two input images to extract feature maps $\mathbf{g}_1, \mathbf{g}_2 \in \mathbb{R}^{H/8 \times W/8 \times D}$ at 1/8 resolution. A separate Context Encoder extracts the initial hidden state and context feature for the GRU from the first image.

**2. Correlation Volume Construction**: the inner product is computed for every pixel pair of the two feature maps:

$$C_{ijkl} = \sum_d g_1(i, j, d) \cdot g_2(k, l, d)$$

Generates a 4D correlation volume $\mathbf{C} \in \mathbb{R}^{H \times W \times H \times W}$. This is average-pooled over the last two dimensions to build a 4-level **correlation pyramid** (scales 1, 2, 4, 8).

Key insight: **rather than coarse-to-fine, multi-scale lookup is performed at a single resolution**.

**3. Iterative Update (GRU)**: a ConvGRU iteratively refines the current flow estimate:

At each iteration $k$:
1. Based on the current flow estimate $\mathbf{f}^k$, look up values from the correlation pyramid (references a local window around the current correspondence location)
2. Combine the correlation feature, current flow, and context feature
3. ConvGRU updates the hidden state $\mathbf{h}^k$
4. Predict the flow residual $\Delta \mathbf{f}$ from the hidden state
5. Flow update: $\mathbf{f}^{k+1} = \mathbf{f}^k + \Delta \mathbf{f}$

12 iterations at training, 12-32 at inference. It has the property that **increasing the number of iterations improves accuracy** (test-time adaptability).

**All-Pairs vs. Coarse-to-Fine**: existing methods such as PWC-Net and FlowNet first estimate coarse flow at the pyramid and then progressively refine, so large displacements missed at the coarse level cannot be recovered. RAFT **keeps all correlations at full resolution at once**, so large displacements are not missed.

**Training**: an L1 loss with ground-truth flow is applied to the predictions at every iteration, with exponentially increasing weights on later iterations:

$$L = \sum_{k=1}^{K} \gamma^{K-k} \| \mathbf{f}^k - \mathbf{f}^{gt} \|_1$$

Here, $\gamma = 0.8$.

```python
# RAFT usage example (torchvision)
from torchvision.models.optical_flow import raft_large, Raft_Large_Weights

model = raft_large(weights=Raft_Large_Weights.DEFAULT)
model = model.eval().cuda()

with torch.no_grad():
    # img1, img2: (1, 3, H, W) float tensors, range [-1, 1]
    flow_predictions = model(img1, img2)
    # flow_predictions[-1]: final flow (1, 2, H, W)
    flow = flow_predictions[-1]  # (u, v) per pixel
```

#### FlowFormer (2022)

[FlowFormer (Huang et al., 2022)](https://arxiv.org/abs/2203.16194) is a method that **replaces RAFT's GRU update with a transformer**. It tokenizes the cost volume and uses transformer self-attention to capture global context, achieving accuracy surpassing RAFT.

#### UniMatch (2023)

[Xu et al. (2023)](https://arxiv.org/abs/2211.05783)'s UniMatch handles **optical flow, stereo matching, and depth estimation in a single unified framework**. The key idea is that all three tasks reduce to the common problem of "dense correspondence between two observations."

### 5.10.3 Dense Stereo: SGM → RAFT-Stereo → UniMatch

Stereo matching is the problem of estimating the **horizontal disparity** between left and right camera images. It can be viewed as a special case of optical flow (vertical flow = 0).

#### SGM (Semi-Global Matching, Hirschmuller 2005)

The representative of traditional dense stereo. Key ideas:
- Compute the matching cost for every disparity candidate at each pixel
- Aggregate costs by 1D dynamic programming along 8 (or 16) directions
- Select the minimum of the aggregated costs to determine the disparity

$$L_r(p, d) = C(p, d) + \min\begin{cases} L_r(p-r, d) \\ L_r(p-r, d-1) + P_1 \\ L_r(p-r, d+1) + P_1 \\ \min_i L_r(p-r, i) + P_2 \end{cases}$$

Here, $P_1, P_2$ are smoothness penalties.

#### RAFT-Stereo (2021)

Applies the RAFT architecture to stereo matching. The 4D correlation volume is reduced to 3D (H×W×D), and an iterative GRU refines the disparity.

#### Unification in UniMatch

UniMatch unifies flow, stereo, and depth, so a single model adjusts the direction and range of cross-attention depending on the task:

- **Stereo**: 1D horizontal correlation
- **Flow**: 2D all-pairs correlation
- **Depth**: depth regression from monocular features

### 5.10.4 Technical Flow: From Sparse Feature to Dense Correspondence

```
[Sparse Correspondence]
Harris → SIFT → SuperPoint → SuperGlue → LightGlue
    ↓ densification
[Dense Correspondence]
Lucas-Kanade (1981)  — local window, sparse-to-dense
Horn-Schunck (1981)  — global optimization, per-pixel flow
    ↓ deep learning
FlowNet (2015)       — CNN direct prediction
[PWC-Net](https://arxiv.org/abs/1709.02371) (2018)       — cost volume + coarse-to-fine
    ↓ paradigm shift
RAFT (2020)          — all-pairs correlation + iterative GRU
    ↓ transformer
FlowFormer (2022)    — cost volume tokenization + transformer
    ↓ task unification
UniMatch (2023)      — flow + stereo + depth unified
```

RAFT's all-pairs correlation and iterative refinement ideas directly influenced not only dense matching but also **detector-free feature matching** (LoFTR, RoMa) and **learned SLAM** (DROID-SLAM).

---

## Technical Lineage Summary

We consolidate the flow of all technologies covered in this chapter into a single diagram:

```
═══════════════════════════════════════════════════════════════════════════════
                    FEATURE MATCHING & CORRESPONDENCE TECHNICAL LINEAGE
═══════════════════════════════════════════════════════════════════════════════

[2D Detection & Description]
                                                                              
Harris (1988) ─────→ SIFT (2004) ───→ FAST (2006) ─→ ORB (2011)             
  corner detection    scale-space       extreme speed    FAST+BRIEF             
  structure tensor    DoG + 128D        detection only   binary descriptor       
                      float descriptor                                        
         │                │                                                   
         │   history of the accuracy vs. speed trade-off                      
         ▼                ▼                                                   
[Learning-Based Detection & Description]                                      
                                                                              
    SuperPoint (2018) ──→ D2-Net (2019) ──→ R2D2 (2019) ──→ DISK (2020)      
      self-supervised      detect=describe    reliability      RL-based         
      homographic adapt    joint feature map  + repeatability                  
         │                                                                    
         │                                                                    
         ▼                                                                    
[Learning-Based Matching — Pipeline Preserved]                                
                                                                              
    SuperGlue (2020) ──────────────→ LightGlue (2023)                        
      GNN + cross-attention           adaptive depth/width                    
      Sinkhorn optimal transport      dual-softmax (Sinkhorn removed)         
      O(N²) × 9 layers                early exit, 3-5× faster                 
                                                                              
═══════════════════════ Paradigm Shift ═══════════════════════════════════       
                                                                              
[Detector-Free Matching — Pipeline Dissolved]                                 
                                                                              
    LoFTR (2021) ──→ QuadTree (2022) ──→ ASpanFormer (2022) ──→ RoMa (2024) 
      transformer         O(N log N)        adaptive span          DINOv2     
      coarse-to-fine       efficiency        texture-adaptive       probabilistic matching 
      detector fully removed                                        foundation 
                                                                    model leveraged
                                                                        │
═══════════════════════ 3D-Aware Shift ═══════════════════════════════════
                                                                        │
[3D-Aware Dense Matching — Beyond 2D Matching]                          ▼
                                                                              
    DUSt3R (2024) ──→ MASt3R (2024) ──→ VGGT (2025, CVPR Best Paper)  
      3D pointmap         dense local        feed-forward 3D inference 
      direct regression   feature added      camera+depth+pointmap     
      matching = 3D byproduct  stronger matching  unified single-pass inference
═══════════════════════════════════════════════════════════════════════════════

[Dense Matching & Optical Flow — Parallel Evolution]                          
                                                                              
    LK (1981) ──→ Horn-Schunck (1981) ──→ FlowNet (2015)                     
      local window   global smoothness      CNN direct prediction              
         │                                      │                             
         ▼                                      ▼                             
    PWC-Net (2018) ──→ RAFT (2020) ──→ FlowFormer (2022) ──→ UniMatch (2023)
      cost volume       4D all-pairs      transformer           flow+stereo   
      coarse-to-fine    iterative GRU     cost tokenization     +depth unified
                            │                                                 
                            │ RAFT's ideas propagate                           
                            ├──→ LoFTR (all-pairs attention)                  
                            ├──→ RoMa (iterative refinement)                  
                            └──→ DROID-SLAM (correlation + DBA)               
                                                                              
═══════════════════════════════════════════════════════════════════════════════

[3D-3D Correspondence — Independent Evolution]                                
                                                                              
    FPFH (2009) ──→ SHOT (2010) ──→ 3DMatch (2017) ──→ FCGF (2019)          
      geometric histograms  orientation histograms  learned 3D descriptor  sparse conv
                                                │                             
                                                ▼                             
                                         Predator (2021)                      
                                           overlap-aware                      
                                                │                             
                                                ▼                             
                                        GeoTransformer (2022)                 
                                           geometric transformer              
                                           RANSAC-free                        
                                                                              
═══════════════════════════════════════════════════════════════════════════════

[Cross-Modal — MI-Based Bypass Strategy]                                      
                                                                              
    MI (information theory) ──→ NMI ──→ NID (Koide et al. 2023)              
      multi-modality              normalization  LiDAR-Camera calibration    
      statistical dependence                                                  
                                                                              
═══════════════════════════════════════════════════════════════════════════════

Key narratives:
  1. A flow that replaces the traditional three-stage pipeline
     (detect → describe → match) with deep learning one step at a time:
     SIFT → SuperPoint → SuperGlue → LightGlue

  2. A flow that dismantles the pipeline itself and transitions to
     end-to-end dense matching: RAFT → LoFTR → RoMa

  3. The rise of leveraging foundation models: using DINOv2 features as
     the basis for matching — a paradigm shift of "leave feature learning
     to the general model and only learn the matching logic."

  4. The emergence of 3D-aware matching (2024-2025): in the flow from
     DUSt3R → MASt3R → VGGT, the reverse intuition of "directly inferring
     3D lets matching follow naturally" — rather than "2D matching then
     3D reconstruction" — is achieving great success.

  The latter (2, 3, 4) is increasingly dominant, but the efficiency and
  interpretability of the former (1) still hold practical value.
```

The matching techniques covered in this chapter are used in earnest from the next chapter onward. In Ch.6 we examine, through concrete systems, how these techniques are deployed in the frontend of Visual Odometry, and how the state estimation methods covered in Ch.4 are combined in the backend.
