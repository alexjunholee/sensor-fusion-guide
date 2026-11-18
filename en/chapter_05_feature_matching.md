# Chapter 5. Feature Matching and Correspondence

The Kalman filters and factor graphs of Chapter 4 estimate a robot's state from sensor measurements, but they rest on a prior assumption: that the system already knows which observation corresponds to which physical landmark or earlier frame (→ Chapter 4). Answering that **data association** question is the role of feature matching and correspondence search, which bridge raw sensor streams and the odometry, calibration, and mapping pipelines ahead.

> Almost every component of sensor fusion — visual odometry, extrinsic calibration, loop closure, and point cloud registration — rests on **correspondence**. Over the past four decades, the field has progressed from handcrafted local detectors and descriptors to robust geometric filtering, information-theoretic cross-modal alignment, learned keypoint matchers, detector-free dense transformers, and direct 3D-aware inference, with each generation tackling a bottleneck left open by the last.

---

## 5.1 What Is the Correspondence Problem

### 5.1.1 The Problem of Finding "the Same Thing"

At its core, the correspondence problem asks how to identify **the same physical point, region, or structure** across two or more observations. Nearly every stage of a sensor-fusion pipeline relies on getting those associations right:

- **Visual Odometry**: Camera motion can only be recovered by tracking the 2D projections of the same 3D scene points across consecutive frames.
- **Calibration**: Estimating camera-LiDAR extrinsics requires linking what both sensors see. Target-based calibration matches the same physical corner or plane, whereas targetless mutual-information methods (§5.4) measure statistical dependence across their overlapping field of view.
- **Loop Closure**: Recognizing a revisited place demands verifiable correspondences between the current observation and a stored past keyframe or submap.
- **Point Cloud Registration**: Aligning two 3D scans means estimating a rigid transformation from paired geometric points or surfaces.

### 5.1.2 Three Types of Correspondence

Depending on the sensors involved and whether a map already exists, correspondence problems fall into three geometric regimes.

#### 2D-2D Correspondence

When two cameras—or one moving camera at two instants—view the same scene, matching seeks the 2D image projections of shared 3D points, forming the foundation of visual odometry, stereo disparity, and image stitching.

If point $\mathbf{p}_1 = (u_1, v_1)$ in image $I_1$ and point $\mathbf{p}_2 = (u_2, v_2)$ in image $I_2$ are projections of the same 3D point $\mathbf{X}$, the pair $(\mathbf{p}_1, \mathbf{p}_2)$ forms a correspondence. In the projective equations below, image points are written in homogeneous coordinates $(u, v, 1)^\top$ so that multiplication by a $3 \times 3$ matrix and inversion via $\mathbf{K}^{-1}$ are both well defined.

The two views are tied together geometrically by the **epipolar constraint**, which requires the ray through $\mathbf{p}_1$ and the ray through $\mathbf{p}_2$ to be coplanar:

$$\mathbf{p}_2^\top \mathbf{F} \mathbf{p}_1 = 0$$

Here $\mathbf{F}$ is the $3 \times 3$, rank-2 fundamental matrix. When camera intrinsics are known, multiplying out the calibration matrices gives the essential matrix $\mathbf{E} = \mathbf{K}_2^\top \mathbf{F} \mathbf{K}_1$, and the constraint takes the calibrated form:

$$\hat{\mathbf{p}}_2^\top \mathbf{E} \hat{\mathbf{p}}_1 = 0$$

where $\hat{\mathbf{p}} = \mathbf{K}^{-1} \mathbf{p}$ denotes normalized image coordinates.

#### 2D-3D Correspondence

Once a 3D map has been built, localization switches from comparing two images to matching 2D image observations against 3D map points. This formulation defines the **Perspective-n-Point (PnP)** problem that powers visual localization and map-based SLAM tracking.

Projecting a 3D point $\mathbf{X} = (X, Y, Z)^\top$ to its 2D pixel location $\mathbf{p} = (u, v)^\top$ through a calibrated camera gives:

$$s \begin{bmatrix} u \\ v \\ 1 \end{bmatrix} = \mathbf{K} [\mathbf{R} | \mathbf{t}] \begin{bmatrix} X \\ Y \\ Z \\ 1 \end{bmatrix}$$

With a minimal set of 6 correspondences (DLT), 4 correspondences (P3P with one disambiguating point), or 3 correspondences (P3P yielding up to four candidates), we can solve for the camera pose $(\mathbf{R}, \mathbf{t})$.

#### 3D-3D Correspondence

When both observations already live in metric 3D space—as in LiDAR scan-to-scan odometry, multi-LiDAR calibration, or point-cloud loop closure—the task becomes pairing the same physical points across two point clouds.

Given corresponding pairs between $\mathcal{P} = \{\mathbf{p}_i\}$ and $\mathcal{Q} = \{\mathbf{q}_i\}$, estimating the rigid transformation $(\mathbf{R}, \mathbf{t})$ reduces to minimizing the sum of squared Euclidean residuals:

$$\min_{\mathbf{R}, \mathbf{t}} \sum_{i} \| \mathbf{q}_i - (\mathbf{R} \mathbf{p}_i + \mathbf{t}) \|^2$$

Because the correspondences are already paired, this least-squares problem admits a closed-form solution via SVD, which forms the inner step of Iterative Closest Point (ICP; → Chapter 7).

### 5.1.3 The Role of Correspondence in Sensor Fusion

Downstream pose accuracy rises or falls with correspondence quality. Even a handful of false matches (outliers) can pull an estimator far off course, whereas scenes with too few reliable matches—such as blank walls or repetitive facades—leave motion underconstrained. How the field learned to extract distinctive features, reject outliers, and eventually move beyond local keypoints is the subject of the sections that follow.

---

## 5.2 Traditional Feature Detection & Description

Comparing every pixel in one image against every pixel in another is both computationally prohibitive and ambiguous on uniform surfaces. Classical correspondence pipelines therefore compress each image into sparse, repeatable locations through three stages: **detect → describe → match**. Detection pinpoints salient keypoints, and description encodes each keypoint's neighborhood into a compact vector so pairs can be compared across frames.

### 5.2.1 Corner Detection: Harris → FAST → ORB

#### Harris Corner Detector (1988)

Not every pixel makes a good landmark. Shifting a small window over a flat wall produces almost no intensity change in any direction, while shifting along a straight edge produces change only perpendicular to the edge—leaving its exact position along the edge ambiguous. A **corner** is distinct precisely because intensity changes sharply no matter which direction the window moves.

To formalize that directional sensitivity, the Harris detector measures the weighted sum of squared intensity differences when a window centered at $(x, y)$ in image $I$ is shifted by $(\Delta u, \Delta v)$:

$$E(\Delta u, \Delta v) = \sum_{x, y} w(x, y) [I(x + \Delta u, y + \Delta v) - I(x, y)]^2$$

where $w(x, y)$ is a Gaussian smoothing window. For small shifts, a first-order Taylor expansion turns this nonlinear difference into a quadratic form:

$$E(\Delta u, \Delta v) \approx \begin{bmatrix} \Delta u & \Delta v \end{bmatrix} \mathbf{M} \begin{bmatrix} \Delta u \\ \Delta v \end{bmatrix}$$

This approximation isolates the local gradient geometry inside a $2 \times 2$ symmetric matrix called the **structure tensor** (or second moment matrix) $\mathbf{M}$:

$$\mathbf{M} = \sum_{x, y} w(x, y) \begin{bmatrix} I_x^2 & I_x I_y \\ I_x I_y & I_y^2 \end{bmatrix}$$

where $I_x$ and $I_y$ are the horizontal and vertical image gradients. Diagonalizing $\mathbf{M}$ reveals the local shape through its eigenvalues $(\lambda_1, \lambda_2)$:

- $\lambda_1 \approx 0, \lambda_2 \approx 0$: a flat region with negligible intensity change in all directions.
- $\lambda_1 \gg \lambda_2 \approx 0$: an edge, where intensity varies strongly across the boundary but stays flat along it.
- Both $\lambda_1$ and $\lambda_2$ large: a corner, where intensity changes significantly in every direction.

Computing exact eigenvalues at every pixel was expensive on late-1980s hardware, so Harris replaced them with a **corner response function** built solely from the determinant and trace of $\mathbf{M}$:

$$R = \det(\mathbf{M}) - k \cdot \text{tr}(\mathbf{M})^2 = \lambda_1 \lambda_2 - k(\lambda_1 + \lambda_2)^2$$

With $k$ typically set to 0.04–0.06, $R$ is large and positive only when both eigenvalues are large. Pixels with $R > \text{threshold}$ undergo non-maximum suppression so only local response peaks survive as corners.

**Limitations of Harris**: Although rotation invariant, the fixed window size makes Harris **not scale invariant**: zoom in closer, and a sharp corner blurs into a gently curved edge. Overcoming that scale sensitivity later motivated SIFT's scale-space pyramid. In practice, OpenCV computes the raw response map with a single function call, leaving thresholding and non-maximum suppression to the caller:

```python
import cv2
import numpy as np

# Harris Corner Detection
img = cv2.imread('scene.jpg', cv2.IMREAD_GRAYSCALE)
img_float = np.float32(img)

# blockSize: neighborhood size, ksize: Sobel kernel, k: Harris parameter
harris_response = cv2.cornerHarris(img_float, blockSize=2, ksize=3, k=0.04)

# Threshold only — cornerHarris returns the response map, so NMS is a separate step
corners = harris_response > 0.01 * harris_response.max()
```

#### FAST (Features from Accelerated Segment Test, 2006)

While Harris offered a clean mathematical model of corners, computing Sobel gradients and structure tensors at every pixel was still too heavy for early real-time robotics running at tens of frames per second. FAST—proposed by [Rosten & Drummond (ECCV 2006; the link is the extended version by Rosten, Porter & Drummond, TPAMI 2010)](https://arxiv.org/abs/0810.2434)—trades gradient calculus for simple intensity comparisons to achieve **extreme detection speed**.

Instead of derivatives, FAST inspects a 16-pixel Bresenham circle of radius 3 around each candidate pixel $p$:

1. Examine the 16 pixels along the ring around $p$.
2. Declare $p$ a corner if $N$ contiguous pixels on the circle (typically $N=12$) are all brighter than $I_p + t$ or all darker than $I_p - t$.
3. Apply a fast-reject pre-test when $N=12$: check only the four compass pixels at positions 1, 5, 9, and 13 first, and discard $p$ immediately unless at least three of them pass the same bright or dark threshold. Because that four-point shortcut is mathematically valid only for $N=12$ and does not generalize to variants such as $N=9$, FAST trains a decision tree to choose the optimal test order in general.

Formally, the detector tests whether such a bright or dark arc exists around $p$:

$$\text{FAST condition: } \exists \text{ contiguous arc of } N \text{ pixels on circle, all } > I_p + t \text{ or all } < I_p - t$$

where $t$ is the intensity threshold.

**Decision tree learning**: To squeeze out additional speed, FAST trains an ID3 decision tree on representative images to learn which ring pixel to inspect next so non-corners are rejected in as few comparisons as possible.

**Limitations of FAST**: Raw speed comes at the expense of invariance and matching capability—FAST provides neither orientation nor scale, and it outputs only keypoint coordinates without a descriptor. ORB was designed specifically to fill those two gaps without sacrificing FAST's speed:

```python
# FAST Corner Detection
fast = cv2.FastFeatureDetector_create(threshold=20, nonmaxSuppression=True)
keypoints = fast.detect(img, None)
print(f"Detected {len(keypoints)} keypoints")
```

#### ORB (Oriented FAST and Rotated BRIEF, 2011)

By 2011, SIFT and SURF were accurate but both computationally heavy and encumbered by patents. [Rublee et al. (2011)](https://ieeexplore.ieee.org/document/6126544) introduced ORB as a fast, open alternative that **augments the FAST detector with orientation and pyramid scale, and upgrades the BRIEF binary descriptor with rotation invariance**.

To supply the orientation missing from FAST (**oFAST**, or Oriented FAST), ORB computes the **intensity centroid** of the patch surrounding each detected corner. Using the image moments of the patch:

$$m_{pq} = \sum_{x, y} x^p y^q I(x, y)$$

the intensity-weighted centroid sits at $\mathbf{C} = (m_{10}/m_{00}, m_{01}/m_{00})$, and the vector from the keypoint center to $\mathbf{C}$ defines the dominant orientation $\theta = \text{atan2}(m_{01}, m_{10})$. To handle scale changes as well, ORB constructs an image pyramid (typically 8 levels) and runs FAST independently at each level.

For description (**rBRIEF**, or Rotated BRIEF), BRIEF converts a patch into a binary string by evaluating pairwise intensity tests at sampled offsets $(x_i, y_i)$:

$$\tau(\mathbf{p}; x_i, y_i) = \begin{cases} 1 & \text{if } I(\mathbf{p}, x_i) < I(\mathbf{p}, y_i) \\ 0 & \text{otherwise} \end{cases}$$

Stacking 256 such tests yields a compact 256-bit (32-byte) descriptor. Naive BRIEF fails as soon as the camera rolls in-plane, so ORB steers the sampling pairs by the keypoint orientation $\theta$ to achieve **rotation invariance**. Moreover, rather than choosing point pairs completely at random, ORB selects them greedily during offline training to maximize variance while minimizing pairwise correlation, sharpening discriminability.

Because ORB pairs fast pyramid FAST corners with a binary descriptor matched via Hamming distance, it became **the core feature of the ORB-SLAM series** and remains a staple of CPU-constrained SLAM pipelines:

```python
# ORB Detection & Description
orb = cv2.ORB_create(nfeatures=1000)
keypoints, descriptors = orb.detectAndCompute(img, None)
# descriptors.shape: (N, 32) — 256-bit = 32 bytes
```

### 5.2.2 Blob Detection: SIFT → SURF

While corner detectors look for sharp intersections of edges, blob detectors search across scales for regions whose intensity stands out from their surroundings—allowing them to handle large zoom changes that break fixed-window corner detectors.

#### SIFT (Scale-Invariant Feature Transform, 2004)

Proposed by [Lowe (2004)](https://link.springer.com/article/10.1023/B:VISI.0000029664.99615.94), SIFT is a landmark keypoint detector and descriptor built from the ground up for invariance to scale and in-plane rotation, while keeping its normalized gradient descriptor resilient to illumination shifts and moderate affine distortion. It served as the field's accuracy benchmark for over a decade, and its US patent expired in 2020. SIFT operates in four stages:

**Stage 1 — Scale-Space Extrema Detection (DoG)**: To find features at their natural physical size, SIFT builds a continuous scale-space by convolving the image with Gaussians of increasing width $\sigma$:

$$L(x, y, \sigma) = G(x, y, \sigma) * I(x, y)$$

where $G(x, y, \sigma) = \frac{1}{2\pi\sigma^2} \exp\left(-\frac{x^2 + y^2}{2\sigma^2}\right)$. True scale-normalized Laplacian of Gaussian ($\sigma^2 \nabla^2 G$) responses identify blobs across scales, and SIFT approximates them cheaply as the **Difference of Gaussians (DoG)** between adjacent blur levels:

$$D(x, y, \sigma) = L(x, y, k\sigma) - L(x, y, \sigma) \approx (k-1)\sigma^2 \nabla^2 G$$

Candidate keypoints are local maxima or minima in $(x, y, \sigma)$, found by comparing each pixel in a DoG level against its 26 neighbors—8 in the same image plane plus 9 each in the scales immediately above and below.

**Stage 2 — Keypoint Localization**: Discrete grid sampling only pins down keypoints to integer pixels and discrete scale steps. SIFT refines each candidate to sub-pixel and sub-scale accuracy using a second-order Taylor expansion of $D(\mathbf{x})$:

$$D(\mathbf{x}) = D + \frac{\partial D}{\partial \mathbf{x}}^\top \mathbf{x} + \frac{1}{2} \mathbf{x}^\top \frac{\partial^2 D}{\partial \mathbf{x}^2} \mathbf{x}$$

Setting the derivative to zero locates the interpolated extremum at $\hat{\mathbf{x}} = -\frac{\partial^2 D}{\partial \mathbf{x}^2}^{-1} \frac{\partial D}{\partial \mathbf{x}}$. Candidates with weak contrast ($|D(\hat{\mathbf{x}})| < 0.03$) are discarded as noise, and candidates lying along straight edges—which have one large principal curvature and one tiny one—are filtered out by thresholding the ratio of eigenvalues of the 2D spatial Hessian.

**Stage 3 — Orientation Assignment**: To make the descriptor rotation invariant, SIFT accumulates gradient magnitudes and orientations within a Gaussian-weighted neighborhood ($\sigma = 1.5 \times$ keypoint scale) into a 36-bin histogram. The dominant peak sets the keypoint's canonical orientation, and any secondary peak reaching at least 80% of the maximum spawns an additional keypoint at that orientation.

**Stage 4 — Keypoint Descriptor**: Once rotated into its canonical frame, a $16 \times 16$ patch around the keypoint is partitioned into a $4 \times 4$ grid of subregions, and each subregion builds an 8-bin gradient orientation histogram:

$$\text{Descriptor} = 4 \times 4 \times 8 = 128\text{-dimensional vector}$$

SIFT L2-normalizes this 128-dimensional vector to remove linear contrast changes, then clips any component above 0.2 and renormalizes—a simple trick that bounds the impact of specular highlights and nonlinear illumination shifts. In OpenCV, detecting and describing SIFT features takes only two lines:

```python
# SIFT Detection & Description
sift = cv2.SIFT_create(nfeatures=2000)
keypoints, descriptors = sift.detectAndCompute(img, None)
# descriptors.shape: (N, 128) — 128-dimensional float32
```

#### SURF (Speeded-Up Robust Features, 2006)

SIFT's accuracy came with heavy Gaussian convolutions and 128-dimensional floating-point histograms. [Bay et al. (2006)](https://link.springer.com/chapter/10.1007/11744023_32) proposed SURF to keep SIFT's scale-and-rotation invariance while cutting the computational cost through three design choices:

- **Box filters over integral images**: Rather than smoothing with expensive Gaussians across scales, SURF approximates second-order Gaussian derivatives with rectangular box filters evaluated in $O(1)$ time via an **integral image**.
- **Hessian determinant detection**: Instead of DoG extrema, SURF detects blobs where the approximated **Hessian determinant** peaks:

$$\det(\mathbf{H}) = D_{xx} D_{yy} - (0.9 \cdot D_{xy})^2$$

- **Compact 64-dimensional descriptor**: Rather than 128 histogram bins, SURF sums Haar wavelet responses and their absolute values over a $4 \times 4$ grid, cutting the floating-point vector in half to 64 dimensions.

While SURF was engineered to approximate Hessian scale-space responses efficiently with integral images and box filters, actual speed and accuracy margins depend on the implementation, image resolution, and hardware. Because of historical patent restrictions, SURF sees little use in modern codebases; real-time systems favor ORB, whereas accuracy-sensitive pipelines turn to SIFT or learned features.

### 5.2.3 Binary Descriptors: BRIEF, ORB, BRISK

Floating-point descriptors such as SIFT's 128-D vector demand 512 bytes per keypoint and floating-point L2 distance evaluations. Binary descriptors take a radically lighter route: they encode local appearance as a compact bit string by comparing pairs of pixel intensities, allowing matching via Hamming distance—implemented with bitwise XOR and `popcount`—which is typically far cheaper than floating-point L2 norms, though the exact speedup depends on descriptor length, library implementation, and CPU architecture. The table below compares three widely used binary descriptors:

| Descriptor | Bits | Features |
|-----------|--------|------|
| BRIEF | 128/256/512 | Random point pairs, not rotation invariant |
| ORB | 256 | Learned point pairs, rotation invariant |
| BRISK | 512 | Concentric sampling, scale invariant |

At the core of BRIEF is a simple pairwise intensity comparison folded into an $n$-bit integer:

$$b_i = \begin{cases} 1 & \text{if } I(\mathbf{p}_i) < I(\mathbf{q}_i) \\ 0 & \text{otherwise} \end{cases}, \quad \text{descriptor} = \sum_{1 \le i \le n} 2^{i-1} b_i$$

Computing the Hamming distance between two descriptors $\mathbf{a}$ and $\mathbf{b}$ reduces to counting differing bits via XOR and `popcount`. Because `popcount` operates as a native 64-bit word instruction on modern CPUs, comparing two 256-bit descriptors requires only four 64-bit XOR-and-`popcount` operations plus their sum (and eight for a 512-bit descriptor):

$$d_H(\mathbf{a}, \mathbf{b}) = \text{popcount}(\mathbf{a} \oplus \mathbf{b})$$

### 5.2.4 Technical Lineage: A History of the Accuracy vs. Speed Trade-off

Taken together, classical feature design reads as a sustained push-and-pull between geometric invariance and real-time throughput:

```
Harris (1988)     — Mathematical definition of corner detection
    ↓ need for scale invariance
SIFT (2004)       — scale-space + 128D float descriptor (accurate but slow)
    ↓ speed improvement
SURF (2006)       — integral image + 64D (speed-oriented, still float)
    ↓ real-time demand
FAST (2006)       — extreme-speed detection (no descriptor)
    ↓ unifying detection+description
ORB (2011)        — oFAST + rBRIEF, 256-bit binary (Hamming matching)
```

That same tension persists in the deep-learning era: SuperPoint, for instance, jointly learns detection and description to handle severe viewpoint and illumination shifts, yet its roughly 70 FPS throughput on 640×480 images (§5.5.1) remains tied to GPU hardware rather than lightweight CPU bit operations. Once keypoints and descriptors are extracted, however, the next challenge is deciding which pairs actually match—and filtering out the ones that do not.

---

## 5.3 Traditional Matching & Outlier Rejection

Detecting keypoints and computing descriptors in two images only sets the stage; a matching step is still needed **to decide which pairs actually depict the same 3D scene point**—and a geometric verification step must weed out the inevitable false positives.

### 5.3.1 Brute-Force Matching

The most direct way to pair descriptors is brute-force search: compare every descriptor in image A against every descriptor in image B and return the closest neighbor. Floating-point descriptors such as SIFT are compared by L2 distance, whereas binary descriptors such as ORB use Hamming distance. For $N$ and $M$ keypoints of dimension $D$, an exhaustive search takes $O(N \cdot M \cdot D)$ operations.

Setting `crossCheck=True` enforces a **mutual nearest neighbor** check, accepting a pair $(i, j)$ only when $j$ is $i$'s closest match from A to B and $i$ is simultaneously $j$'s closest match from B to A:

```python
# Brute-Force Matching
bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)  # SIFT
# bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)  # ORB
matches = bf.match(desc1, desc2)
matches = sorted(matches, key=lambda x: x.distance)
```

### 5.3.2 FLANN (Fast Library for Approximate Nearest Neighbors)

Once an image contains thousands of keypoints—or a localization system searches against a map with hundreds of thousands of descriptors—exhaustive $O(N \cdot M \cdot D)$ comparison becomes too slow. FLANN solves this by switching to **approximate nearest neighbor (ANN) search**, automatically selecting among indexing structures such as randomized kd-trees or hierarchical k-means trees to trade exactness for sub-linear lookup time. Requesting `k=2` neighbors per query sets up the ratio test described next:

```python
# FLANN Matching for SIFT
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)  # limit on number of nodes explored

flann = cv2.FlannBasedMatcher(index_params, search_params)
matches = flann.knnMatch(desc1, desc2, k=2)  # k=2 for ratio test
```

### 5.3.3 Lowe's Ratio Test

Nearest-neighbor distance alone is a surprisingly poor filter: in scenes with repetitive facades, foliage, or low texture, a wrong match can still sit close in descriptor space if dozens of nearly identical patches exist. Proposed by Lowe (2004) alongside SIFT, the ratio test asks not whether the closest match is small in absolute terms, but whether it is **distinctively better than the runner-up**. A match is kept only if the ratio between the nearest distance and the second-nearest distance falls below a threshold $\tau$ (typically 0.7–0.8):

$$\frac{d(\mathbf{f}, \mathbf{f}_1)}{d(\mathbf{f}, \mathbf{f}_2)} < \tau$$

where $\mathbf{f}_1$ and $\mathbf{f}_2$ are the nearest and second-nearest descriptors to $\mathbf{f}$, respectively. Intuitively, a true match stands out clearly from the crowd ($d_1 \ll d_2$), whereas an ambiguous match on a repeating tile or blank wall has multiple lookalike candidates ($d_1 \approx d_2$). In code, filtering a `knnMatch(..., k=2)` output takes a single loop:

```python
# Ratio test (Lowe's)
good_matches = []
for m, n in matches:  # m: best, n: second best
    if m.distance < 0.75 * n.distance:
        good_matches.append(m)
```

### 5.3.4 The RANSAC Family: RANSAC → PROSAC → MAGSAC++

Even after mutual checks and Lowe's ratio test, **outliers (false correspondences)** survive—especially when moving objects, repeating structures, or wide baseline changes fool local descriptors. Ordinary least squares breaks completely under even a single severe outlier, so estimating a geometric model (such as a fundamental or essential matrix) requires a robust estimator that discovers inliers and model parameters together.

#### RANSAC (Random Sample Consensus, 1981)

Introduced by [Fischler & Bolles (1981)](https://dl.acm.org/doi/10.1145/358669.358692), RANSAC flips the usual estimation logic on its head: rather than fitting to all data and hoping outliers wash out, it repeatedly guesses minimal subsets, tests how many points agree with each guess, and refines on the winners:

1. Randomly sample the minimum $n$ matches needed to fit the model (for example, 8 or 7 points for a fundamental matrix, or 5 points for an essential matrix).
2. Solve for a candidate geometric model from that minimal sample.
3. Score the candidate by counting its consensus set (inliers)—all matches whose geometric error lies within an inlier threshold $t$.
4. Keep the candidate with the largest consensus set.
5. Re-estimate the final model using all inliers in that winning consensus set.

How many random draws does RANSAC need? If the true inlier fraction is $w$ and each minimal sample draws $n$ points independently, the chance of picking an all-inlier sample in one try is $w^n$. Requiring at least one clean sample with overall confidence $p$ (typically 0.99) sets the iteration budget to:

$$k = \frac{\log(1 - p)}{\log(1 - w^n)}$$

where $p$ is the desired success probability, $w$ is the inlier ratio, and $n$ is the minimal sample size. With 50% inliers ($w=0.5$), an 8-point fundamental-matrix solver ($n=8$), and 99% confidence ($p=0.99$), RANSAC needs roughly $k \approx 1177$ iterations—and that count climbs rapidly as $w$ drops or $n$ grows. OpenCV bundles minimal sampling, consensus scoring, and re-estimation into `cv2.findFundamentalMat`:

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

Standard RANSAC treats every candidate match as equally plausible, sampling uniformly across the entire list even though descriptor matching already told us which pairs look strongest. Proposed by Chum & Matas, PROSAC **orders matches by a quality score (such as descriptor distance or ratio-test margin)** and grows its sampling pool progressively from the highest-ranked matches outward. When that quality score correlates well with true inlier probability, PROSAC hits an uncorrupted minimal sample far earlier than uniform sampling, though the actual speedup depends on how well calibrated the scores are and how inliers are distributed in the ranking.

#### MAGSAC / MAGSAC++ (2019/2020)

Classic RANSAC also suffers from a brittle hyperparameter: **the hard inlier threshold $t$**. Set $t$ too tight, and noisy true correspondences get thrown away; set $t$ too loose, and false matches slip into the consensus set and bias the fit. Barath et al. removed this brittle hard cutoff in MAGSAC and MAGSAC++ by marginalizing model quality over a continuum of noise scales $\sigma \in [0, \sigma_{\max}]$:

$$Q(\theta) = \int_0^{\sigma_{\max}} q(\theta, \sigma) f(\sigma) d\sigma$$

where $q(\theta, \sigma)$ measures the quality of model $\theta$ at threshold $\sigma$, and $f(\sigma)$ is a prior over noise levels.

Building on this idea, MAGSAC++ speeds up the marginalization and replaces the final unweighted refit with a weighted least-squares step ($\sigma$-consensus) that downweights borderline points smoothly. Rather than relying on a sharp inlier/outlier boundary, sensitivity to the cutoff drops substantially—though an upper integration bound $\sigma_{\max}$ still remains. In OpenCV's `USAC_MAGSAC`, that bound is taken as the larger of `ransacReprojThreshold` and a model-specific internal default (2.5 for the fundamental matrix), while `ransacReprojThreshold` still governs the termination check and the binary `mask` returned to the caller:

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

In 2D-2D visual odometry, RANSAC's inner solver recovers the two-view epipolar geometry from minimal correspondence sets, choosing between two parameterizations depending on whether camera calibration is available:

- **Fundamental Matrix** ($\mathbf{F}$, 7 DOF), used when camera intrinsics are unknown. The 8-point algorithm solves a linear system from at least 8 correspondences and projects onto rank 2, whereas the 7-point algorithm uses 7 correspondences and returns up to 3 real roots of a cubic polynomial.
- **Essential Matrix** ($\mathbf{E}$, 5 DOF), used when intrinsics $\mathbf{K}_1, \mathbf{K}_2$ are known, satisfying $\mathbf{E} = [\mathbf{t}]_\times \mathbf{R}$ and $\mathbf{E} = \mathbf{K}_2^\top \mathbf{F} \mathbf{K}_1$. Nistér's 5-point algorithm (2004) solves from just 5 correspondences via a 10th-order polynomial yielding up to 10 candidate solutions.

Pairing the 5-point solver with RANSAC and decomposing the resulting essential matrix directly recovers relative rotation $\mathbf{R}$ and unit-norm translation $\mathbf{t}$:

```python
# Essential matrix estimation (calibrated camera)
E, mask = cv2.findEssentialMat(pts1, pts2, cameraMatrix=K,
                                method=cv2.RANSAC,
                                prob=0.999, threshold=1.0)

# Recover R, t from the essential matrix
_, R, t, mask_pose = cv2.recoverPose(E, pts1, pts2, cameraMatrix=K)
```

### 5.3.6 Technical Lineage: The Evolution of Robust Estimation

Robust geometric estimation has progressed steadily from brittle least-squares fitting to learned end-to-end transformation networks:

```
Least Squares (vulnerable to outliers)
    ↓ handling outliers
RANSAC (1981)   — a landmark random-sampling paradigm for robust estimation
    ↓ exploiting prior information
PROSAC (2005)   — progressive sampling based on match quality
    ↓ threshold automation
MAGSAC++ (2020) — robust estimation without an inlier threshold
    ↓ learning-based rejection
GeoTransformer (2022) — direct transformation estimation without RANSAC
```

Every method up to this point assumes the two observations share comparable local texture or geometry. What happens when two sensors observe completely different physical quantities—such as an RGB camera and a LiDAR intensity scan? That requires stepping away from local keypoints entirely and turning to information theory.

---

## 5.4 Mutual Information & Intensity-Based Registration

### 5.4.1 Definition and Intuition of MI

When two sensors measure different physical phenomena, corresponding pixels rarely have equal or even linearly correlated intensities. Instead of comparing appearance directly, **Mutual Information (MI)** borrows from Shannon information theory to ask a purely statistical question: how much uncertainty about one variable disappears once we observe the other?

For two discrete random variables $X$ and $Y$, mutual information quantifies the Kullback-Leibler divergence between their joint distribution $p(x, y)$ and the product of their marginals $p(x)p(y)$:

$$I(X; Y) = \sum_{x \in X} \sum_{y \in Y} p(x, y) \log \frac{p(x, y)}{p(x) p(y)}$$

For continuous variables, the sums become double integrals:

$$I(X; Y) = \int \int p(x, y) \log \frac{p(x, y)}{p(x) p(y)} \, dx \, dy$$

Equivalently, MI can be written directly in terms of marginal and joint Shannon entropies:

$$I(X; Y) = H(X) + H(Y) - H(X, Y)$$

where $H(X) = -\sum_x p(x) \log p(x)$ is the marginal entropy of $X$ and $H(X, Y) = -\sum_{x,y} p(x, y) \log p(x, y)$ is their joint entropy.

**Intuition**: When two observations of the same scene are properly aligned, knowing a pixel's value in image $A$ sharply narrows the likely values at the corresponding location in image $B$. That predictability collapses the joint entropy $H(X, Y)$ and drives $I(X; Y)$ to a peak. Misalignment scrambles the pairing, spreads out the joint distribution, and causes $I(X; Y)$ to fall. Because MI captures **arbitrary nonlinear statistical dependence**, it succeeds even where linear correlation fails.

### 5.4.2 MI-Based Multi-Modality Registration

Originally pioneered in medical imaging to align CT and MRI scans, MI excels at **registering observations across distinct sensor modalities** because it makes no assumption about how intensities map from one sensor to the other.

Consider why simpler photometric scores break down across modalities: in CT and MRI of the same anatomy, cortical bone appears bright in CT yet dark in MRI, and soft tissues invert or remap nonlinearly. Comparing raw intensity differences (SSD) or linear correlation (NCC) fails under such non-monotonic intensity relationships. MI sidesteps that hurdle altogether because it only measures whether intensities covary consistently, regardless of whether the mapping is inverted, curved, or many-to-one.

In robotics, this exact property makes MI natural for **camera-LiDAR alignment**: LiDAR reflectance intensity and camera pixel brightness capture completely different physical properties, yet both trace the same object boundaries and surface materials in the shared scene, producing high mutual information when properly registered.

### 5.4.3 Practical Computation of MI and NMI

In image or scan registration, the underlying probability distributions $p(x)$, $p(y)$, and $p(x, y)$ are unknown and must be approximated empirically from a **2D joint histogram** under a candidate transformation $T$:

1. At every overlapping pixel $(u, v)$, pair the intensity $A(u, v)$ with the transformed intensity $B(T(u, v))$.
2. Bin those pairs into a normalized 2D histogram to estimate the joint distribution $p(a, b)$.
3. Sum along the rows and columns of the histogram to obtain the marginals $p(a)$ and $p(b)$.
4. Evaluate $I(A; B)$ from the marginal and joint entropies.

Raw mutual information has a subtle flaw during optimization: changing the transformation $T$ also changes the size of the overlapping region, which can artificially raise or lower $I(A; B)$ even when alignment worsens. **Normalized Mutual Information (NMI)** dampens that sensitivity to overlap area, either as the ratio of marginal entropies to joint entropy:

$$NMI(A, B) = \frac{H(A) + H(B)}{H(A, B)}$$

or equivalently as the symmetric uncertainty coefficient:

$$NMI(A, B) = \frac{2 \cdot I(A; B)}{H(A) + H(B)}$$

Because NMI stays stable as the overlap region expands or contracts, practitioners almost always prefer it over unnormalized MI.

### 5.4.4 MI Gradient Computation

Optimizing an MI-based objective with gradient descent or Gauss-Newton requires differentiating the objective with respect to the transformation parameters $\xi$ of $T_\xi$:

$$\frac{\partial I}{\partial \xi} = \sum_{a, b} \frac{\partial p(a, b)}{\partial \xi} \left(1 + \log \frac{p(a, b)}{p(a) p(b)}\right)$$

Standard hard-binned histograms are piecewise constant and therefore non-differentiable. To make $\partial p(a, b) / \partial \xi$ smooth, gradient-based implementations replace hard binning with continuous **Parzen window (kernel density estimation)** or **B-spline** histogram weights. Even then, MI objectives often exhibit local ripples from interpolation artifacts, so practical calibration tools frequently bypass derivatives altogether and use derivative-free search such as **Nelder-Mead simplex** (as in the calibration toolbox of Koide et al., 2023).

### 5.4.5 Why MI Is Used for Calibration

In targetless camera-LiDAR extrinsic calibration (→ Chapter 3), MI variants serve as the core alignment objective without requiring checkerboards or fiducials:

1. Project the LiDAR point cloud onto the camera plane using the current extrinsic estimate.
2. Evaluate the statistical dependence between projected LiDAR intensity and sampled camera grayscale intensity.
3. Optimize the 6-DOF extrinsics to maximize that dependence.

For optimization stability and clean metric behavior, modern calibration pipelines such as Koide et al. (2023) often minimize **Normalized Information Distance (NID)** rather than maximizing raw MI:

$$NID(A, B) = 1 - \frac{I(A; B)}{H(A, B)} = \frac{H(A|B) + H(B|A)}{H(A, B)}$$

Bounded cleanly between 0 (perfect statistical dependence) and 1 (complete independence), NID satisfies the triangle inequality and other axioms of a proper metric space. Both MI and NID can be computed concisely from a 2D histogram in Python:

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

Returning from cross-modal alignment to camera-to-camera matching, classical detectors and descriptors share a structural limitation: they rely on **low-level visual cues** such as local intensity gradients, corners, and DoG blobs. When day turns to night, viewpoints swing past 45°, or motion blur smears edges, those hand-crafted rules break down. Deep learning replaces brittle gradient heuristics with representations learned from large datasets under heavy photometric and geometric variation.

### 5.5.1 SuperPoint (2018): Self-Supervised Integration of Detection and Description

Unlike object classification, keypoint detection has no objective human ground truth—annotators disagree on which pixels in a natural scene count as "good corners." [DeTone et al. (2018)](https://arxiv.org/abs/1712.07629) sidestepped manual labeling in SuperPoint by **computing keypoint locations and descriptors in a single fully convolutional network** trained via self-supervised Homographic Adaptation.

#### Homographic Adaptation Training Strategy

Rather than asking humans to label corners, SuperPoint generates pseudo ground truth on real images through geometric self-consistency:

1. Warp a single unlabeled image with 100 or more random homographies.
2. Run the current detector independently on every warped view.
3. Project all detected keypoints back into the original image frame and aggregate their heatmaps.
4. Retain only those points **detected consistently across transformations** as pseudo ground-truth keypoints.

By demanding repeatability under synthetic viewpoint warps, the detector learns to fire on stable geometric landmarks without human labels.

#### Two-Stage Training: MagicPoint → SuperPoint

Bootstrapping this self-supervised loop requires an initial detector that already knows what a geometric corner looks like, so training proceeds in two stages. First, a base corner/junction detector (**MagicPoint**) is trained on the Synthetic Shapes dataset—rendered triangles, rectangles, stars, and line segments whose exact corner coordinates are known analytically. Second, MagicPoint is applied with Homographic Adaptation to unlabeled real photographs such as MS-COCO to harvest real-scene pseudo ground truth, which then trains the full **SuperPoint** network jointly for detection and description.

#### Architecture

A shared VGG-style convolutional encoder downsamples the image and splits into two decoder heads:

- **Interest Point Decoder**: Partitions the image into an $8 \times 8$ pixel grid and predicts a 65-channel softmax per cell—64 bins for the sub-cell pixel coordinates plus 1 "no interest point" dustbin channel—reshaping the non-dustbin probabilities into a full-resolution keypoint heatmap.
- **Descriptor Decoder**: Outputs a 256-dimensional descriptor map at reduced resolution, samples descriptor vectors at detected keypoint coordinates via bicubic interpolation, and applies L2 normalization.

#### Training Loss

The detector head minimizes cross-entropy against the pseudo ground-truth cell labels, while the descriptor head exploits the fact that every training pair is related by a known homography $\mathcal{H}$. Because ground-truth correspondences across the warped pair are exact, SuperPoint trains descriptors with a margin-based hinge loss that pulls matching pairs closer and pushes non-matching pairs apart:

$$L_{desc} = \sum_{(i,j) \in \text{pos}} \max(0, m_p - \mathbf{d}_i^\top \mathbf{d}_j) + \sum_{(i,j) \in \text{neg}} \max(0, \mathbf{d}_i^\top \mathbf{d}_j - m_n)$$

where $m_p$ and $m_n$ are the positive and negative similarity margins.

Because both heads share one forward pass, inference is fast: the SuperPoint paper reports roughly 70 FPS on 640×480 images in the authors' GPU implementation, though actual throughput varies with modern GPUs and keypoint budgets. Through the `lightglue` package, running SuperPoint takes only a few lines of PyTorch:

```python
import torch
# SuperPoint usage example (cvg/LightGlue package; kornia.feature has no SuperPoint class)
from lightglue import SuperPoint

# Load model
sp = SuperPoint(max_num_keypoints=2048)
sp = sp.eval()

# Inference
with torch.no_grad():
    img_tensor = torch.from_numpy(img).float().unsqueeze(0).unsqueeze(0) / 255.0
    pred = sp({'image': img_tensor})   # returned keys: keypoints, keypoint_scores, descriptors
    keypoints = pred['keypoints']        # (1, N, 2)
    descriptors = pred['descriptors']    # (1, N, 256)
    scores = pred['keypoint_scores']              # (1, N)
```

### 5.5.2 D2-Net (2019): Detect-and-Describe Jointly

SuperPoint shares a backbone yet still separates detection and description into distinct decoder branches, meaning low-level corners can be detected even where high-level descriptors struggle. [D2-Net (Dusmanu et al., 2019)](https://arxiv.org/abs/1905.03561) pushes integration a step further by **performing detection and description simultaneously from the very same feature map**.

Starting from an intermediate VGG16 feature tensor $\mathbf{F} \in \mathbb{R}^{H \times W \times C}$, D2-Net selects keypoints where a channel achieves both a spatial local maximum and a channel-wise peak (followed by spatial NMS), and directly takes the $C$-dimensional vector at that location as its descriptor. Because those deeper CNN activations encode broader semantic context, D2-Net holds up remarkably well across drastic day-to-night or seasonal appearance changes. The trade-off is spatial precision: detection repeatability can trail SuperPoint, and because the feature map sits at 1/4 of the input resolution, raw localization is quantized to that grid until the paper's local refinement step recovers sub-pixel coordinates.

### 5.5.3 R2D2 (2019): Reliable and Repeatable Detector-Descriptor

D2-Net exposed a subtle flaw in standard keypoint scoring: repeatability alone does not guarantee good matches. A point along a repetitive railing or a faint cloud edge might be detected reliably across views (high **repeatability**), yet its descriptor looks identical to dozens of neighbors and fails during matching (low **reliability**).

Analyzing both SuperPoint and D2-Net, [R2D2 (Revaud et al., 2019)](https://arxiv.org/abs/1906.06195) explicitly decouples **repeatability** from **reliability**, predicting two separate confidence maps and multiplying them to select keypoints that are both geometrically stable and distinctive in descriptor space.

### 5.5.4 DISK (2020)

Even with reliability scores, training detectors via proxy repeatability objectives leaves a gap between what the network optimizes and what downstream pose estimation actually wants: accurate matches. [DISK (Tyszkiewicz et al., 2020)](https://arxiv.org/abs/2006.13566) closes that gap by framing keypoint extraction through **reinforcement learning**—sampling feature detections probabilistically and rewarding pairs that produce correct geometric matches while penalizing false ones, thereby optimizing detection and matching end-to-end.

### 5.5.5 Advantages Over Traditional Methods: Improved Illumination/Viewpoint Invariance

Across challenging visual regimes, learned local features address the specific failure modes that trip up gradient-based pipelines:

| Scenario | SIFT/ORB Limitations | SuperPoint/D2-Net Improvements |
|---------|-------------|---------------------|
| Extreme illumination change (day-night) | DoG/gradient-based detection fails | Learned features capture high-level structures |
| Wide viewpoint change | Limitations of affine approximation | Viewpoint invariance learned from large training data |
| Repetitive patterns | Similar descriptors lead to ambiguous matches | Context information captured for discrimination |
| Motion blur | Weakened gradients → detection failure | CNN learns robustness to blur |

That generalization is not free, however: learned detectors and descriptors inherit the biases of their training distributions and can falter in out-of-domain settings such as underwater imagery or extraterrestrial terrain. Even more importantly, better local descriptors alone still leave the matching stage blind to spatial layout—a limitation that motivated learning the matcher itself.

---

## 5.6 Learning-Based Feature Matching

### 5.6.1 SuperGlue (2020): Attention-Based Matching

Classical nearest-neighbor matching compares each descriptor independently, ignoring where keypoints sit relative to one another. Under repetitive windows or texture shifts, two patches can look identical locally even when their surrounding geometric constellation makes the true pairing obvious. [Sarlin et al. (2020)](https://arxiv.org/abs/1911.11763) introduced SuperGlue to replace hand-crafted nearest-neighbor heuristics and ratio tests with a **graph neural network, alternating attention, and differentiable optimal transport** that solves matching and outlier rejection jointly.

#### Problem Definition: Partial Assignment

Given two sets of keypoints and descriptors $\mathcal{A} = \{(\mathbf{p}_i, \mathbf{d}_i)\}_{i=1}^{N}$ and $\mathcal{B} = \{(\mathbf{p}_j, \mathbf{d}_j)\}_{j=1}^{M}$, many keypoints in one image are occluded or outside the field of view in the other, meaning **not every keypoint has a valid match**. SuperGlue models this partial assignment cleanly by augmenting the matching matrix with an extra "dustbin" row and column that absorb unmatchable points.

#### Keypoint Encoder

To let the matcher reason over both appearance and spatial geometry, each keypoint's 2D coordinates $(x, y)$ and detection confidence $c$ are projected through an MLP and added directly to its visual descriptor:

$$\mathbf{f}_i^{(0)} = \mathbf{d}_i + \text{MLP}_{\text{enc}}([\mathbf{p}_i, c_i])$$

#### Attentional Graph Neural Network

Treating keypoints as nodes of a complete graph, SuperGlue alternates two types of attention across $L$ layers ($L=9$ in the paper) to progressively refine feature representations:

- **Self-Attention (intra-image)** aggregates spatial and visual context among keypoints within the *same* image—encoding structural cues such as collinear building corners or coplanar facades:

$$\text{message}_i^{\text{self}} = \sum_{j \in \mathcal{A}} \text{softmax}\left(\frac{\mathbf{q}_i^\top \mathbf{k}_j}{\sqrt{d}}\right) \mathbf{v}_j$$

- **Cross-Attention (inter-image)** exchanges information *between* the two images, letting each keypoint inspect candidate counterparts across the other frame:

$$\text{message}_i^{\text{cross}} = \sum_{j \in \mathcal{B}} \text{softmax}\left(\frac{\mathbf{q}_i^\top \mathbf{k}_j}{\sqrt{d}}\right) \mathbf{v}_j$$

#### Matching via Optimal Transport

Once the GNN layers finish refining node features, pairwise inner products populate a score matrix $\mathbf{S} \in \mathbb{R}^{(N+1) \times (M+1)}$ whose extra row and column hold a learnable dustbin score $z$:

$$S_{ij} = \langle \mathbf{f}_i^{(L)}, \mathbf{f}_j^{(L)} \rangle, \quad S_{i,M+1} = S_{N+1,j} = z$$

Discrete bipartite matching via the Hungarian algorithm is non-differentiable, so SuperGlue solves a soft, entropy-regularized optimal transport relaxation using the **Sinkhorn algorithm**, alternating row and column normalizations (around 100 iterations):

$$\mathbf{S} \leftarrow \text{row-normalize}(\mathbf{S}), \quad \mathbf{S} \leftarrow \text{col-normalize}(\mathbf{S})$$

At inference, thresholding the converged assignment matrix $\hat{\mathbf{P}}$ yields crisp mutual correspondences while routing unmatchable keypoints to the dustbin.

#### Training

Because every step is differentiable, SuperGlue trains end-to-end by minimizing the negative log-likelihood of ground-truth correspondences (derived from homographies or known camera poses and depth maps) alongside dustbin assignments for unmatched points:

$$L = -\sum_{(i,j) \in \mathcal{M}} \log \hat{P}_{ij} - \sum_{i \in \mathcal{U}_A} \log \hat{P}_{i, M+1} - \sum_{j \in \mathcal{U}_B} \log \hat{P}_{N+1, j}$$

where $\mathcal{M}$ denotes matched pairs and $\mathcal{U}_A, \mathcal{U}_B$ denote keypoints with no counterpart in the other view. In visual localization pipelines such as `hloc`, SuperPoint extraction and SuperGlue matching plug together directly:

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

Where SuperPoint learned detection and description, SuperGlue **learned the matching stage itself**, completing the deep-learning replacement of all three stages of the classical detect-then-describe-then-match pipeline. Still, two practical hurdles remained: SuperGlue was computationally expensive on large keypoint sets, and it remained chained to a fixed keypoint detector—limitations tackled respectively by LightGlue and detector-free transformers.

### 5.6.2 LightGlue (2023): Making SuperGlue Efficient

[Lindenberger et al. (2023)](https://arxiv.org/abs/2306.13643) revisited SuperGlue's architecture and asked why an easy, high-overlap image pair should pay the same compute bill as an extreme wide-baseline pair. By introducing **adaptive computation**, LightGlue achieves accuracy comparable to SuperGlue while delivering an average 3–5× speedup in the authors' evaluation, with the exact gain varying by keypoint count and hardware.

#### Diagnosis of SuperGlue's Problems

Two bottlenecks slowed SuperGlue down: its public configuration always executes all 9 GNN layers and full Sinkhorn normalization regardless of scene difficulty, and every layer incurs $O(N^2)$ attention cost over all $N$ keypoints—causing runtime to balloon rapidly as keypoint counts rise.

#### Key Improvements: Adaptive Depth & Width

LightGlue attacks both bottlenecks together while simplifying the assignment head:

- **Adaptive Depth (early layer exit)**: After each attention layer, a lightweight MLP predicts per-point confidence. When enough points are already confident, inference stops early—for example, in Figure 2 of the paper, an easy pair exits after 3 layers while a difficult pair runs 8 of the 9 layers.
- **Adaptive Width (keypoint pruning)**: Keypoints predicted to be unmatchable with high confidence are dropped from intermediate layers, progressively shrinking the $O(N^2)$ attention sequence length in deeper layers.
- **Removing Sinkhorn**: Rather than iterating Sinkhorn optimal transport, LightGlue computes assignment probabilities directly via **dual-softmax** plus a mutual nearest-neighbor check (with the accuracy trade-off depending on dataset and configuration):

$$P_{ij} = \text{softmax}_j(S_{ij}) \cdot \text{softmax}_i(S_{ij})$$

#### Training Strategy

During training, LightGlue disables early exit and runs the full 9-layer stack with **deep supervision** applied at every intermediate layer so that early predictions remain well calibrated for test-time exit. It is trained as a drop-in matcher across multiple local extractors including SuperPoint, DISK, and ALIKED, delivering an average **3–5×** speedup in the authors' benchmarks (and even larger gains on easy pairs, depending on keypoint count and hardware). In Kornia, pairing SuperPoint outputs with LightGlue requires just a few lines:

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

The evolution from SIFT to LightGlue shows how deep networks replaced each stage of the classical three-step architecture one piece at a time:

```
[Traditional pipeline]
SIFT detect → SIFT descriptor → BF/FLANN + ratio test
                                        ↓
[Learning-based replacements]
SuperPoint detect+describe → SuperGlue attention matching → LightGlue efficiency
    (2018)                       (2020)                        (2023)
```

Keeping the three stages modular makes pipelines easy to inspect and swap, yet leaves one fundamental single point of failure: if the detector fails to fire on repeatable keypoints in the first place, even the strongest matcher has nothing to work with. Detector-free methods overcome that bottleneck by removing the detector altogether (→ §5.7).

---

## 5.7 Detector-Free Matching

### 5.7.1 Why Detector-Free Is Needed

In any detect-then-match pipeline, **matching stops before it even begins if the detector fails to fire on repeatable points**. That failure happens precisely in the scenes robotics encounters most often:

- **Textureless regions**: Painted walls, concrete floors, or overcast skies lack crisp intensity gradients, leaving corner and blob detectors empty-handed.
- **Repetitive patterns**: Tile grids and office facades yield plenty of detections, yet every local patch looks identical without global context.
- **Wide viewpoint changes**: Under extreme perspective shifts, local image patches warp beyond recognition.

Detector-free methods drop the keypoint detector altogether and **treat every spatial position in the image grid as a potential match candidate**, letting global context decide which regions correspond.

### 5.7.2 LoFTR (2021): Coarse-to-Fine Transformer Matching

Proposed by [Sun et al. (2021)](https://arxiv.org/abs/2104.00680), LoFTR replaces explicit keypoint extraction with **a coarse-to-fine transformer that finds dense correspondences across raw feature maps**.

#### Architecture

LoFTR splits dense matching into three stages so it can reason globally without drowning in full-resolution pixel pairs:

1. **Local Feature CNN**: A ResNet-18 Feature Pyramid Network (FPN) extracts both coarse feature maps (at 1/8 resolution) and fine feature maps (at 1/2 resolution) from each image.
2. **Coarse-Level Matching (Transformer)**: The 1/8 feature maps are flattened into 1D token sequences augmented with sinusoidal positional encodings so spatial layout survives flattening. A transformer stack alternates self-attention and cross-attention $N$ times (typically 4), letting every coarse patch gather context within its own image and across the other view. Inner products of the refined features fill a score matrix, and **dual-softmax** converts it into mutual confidence scores:

$$P_{ij} = \text{softmax}_j(\mathbf{f}_i^A \cdot \mathbf{f}_j^B) \cdot \text{softmax}_i(\mathbf{f}_i^A \cdot \mathbf{f}_j^B)$$

Thresholding and a mutual nearest-neighbor check select confident coarse correspondences.

3. **Fine-Level Refinement**: Centered on each coarse match, LoFTR crops a $w \times w$ window (typically 5×5) from the 1/2-resolution feature maps, applies local cross-attention correlation, and regresses a **sub-pixel-accurate** match position.

#### Linear Transformer

Running standard scaled dot-product attention over a flattened 2D feature map of $N$ tokens scales quadratically as $O(N^2)$—already prohibitive at even modest image sizes. To make dense coarse attention tractable, LoFTR replaces standard softmax attention with **kernel-based linear attention**:

Standard attention: $O(N^2)$, $\text{Attn}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{softmax}(\mathbf{Q}\mathbf{K}^\top / \sqrt{d}) \mathbf{V}$

Linear attention: $O(N)$, $\text{Attn}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \phi(\mathbf{Q})(\phi(\mathbf{K})^\top \mathbf{V})$

Using an ELU-based feature map $\phi$ lets us reassociate the matrix product so $\phi(\mathbf{K})^\top \mathbf{V}$ is computed once in $O(N)$ time rather than forming the full $N \times N$ attention matrix. That speedup comes with a trade-off: smoothing attention through a kernel softens its ability to focus sharply on fine spatial details—the exact weakness that QuadTree Attention and ASpanFormer subsequently addressed by altering attention structure.

#### Training

LoFTR is trained end-to-end on ScanNet (indoor) and MegaDepth (outdoor) using ground-truth correspondences projected from known camera poses and depth maps, combining a cross-entropy loss on the coarse assignment matrix with an L2 regression loss on the fine sub-pixel offsets. In Kornia, running pretrained outdoor LoFTR takes only a few lines:

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

LoFTR severed the dependency on an upfront detector that still constrained SuperPoint and SuperGlue. Because cross-attention lets every spatial token in one image communicate directly with every token in the other, detection and matching fuse into a single context-aware operation.

### 5.7.3 QuadTree Attention: Making LoFTR Efficient

Even with linear attention, flattening an entire coarse feature map limits spatial resolution and blurs sharp local peaks. QuadTree Attention (Tang et al., 2022) restores sharper focus by organizing cross-attention hierarchically:

1. Compute full attention across all tokens at the coarsest scale.
2. Retain only the top-scoring regions that show strong correspondence evidence.
3. Refine attention at the next finer resolution strictly inside those selected quadrants.
4. Repeat down the pyramid so computation concentrates on plausible regions.

Compared against standard $O(N^2)$ attention, QuadTree Attention cuts asymptotic complexity to $O(N)$ while keeping fine-grained selectivity; relative to LoFTR's already-$O(N)$ linear attention, the authors report higher matching accuracy at comparable runtime cost.

### 5.7.4 ASpanFormer (2022): Adaptive Span Attention

ASpanFormer (Chen et al., 2022) tackled a complementary flaw in LoFTR: **should every pixel attend over the exact same spatial neighborhood?** Using estimated flow uncertainty, ASpanFormer adapts its **attention span** per location—tightening around a narrow local window in texture-rich regions for crisp localization, while widening across broad context in textureless regions so nearby structures can disambiguate blank walls. This adaptive focus mitigates LoFTR's accuracy drop across low-texture surfaces.

### 5.7.5 RoMa (2024): DINOv2 + Dense Matching

While LoFTR, QuadTree Attention, and ASpanFormer trained their convolutional backbones from scratch on geometry datasets, [Edstedt et al. (2024)](https://arxiv.org/abs/2305.15404) took a qualitatively different leap in RoMa by pairing **frozen foundation-model features** with **probabilistic dense warp regression**.

#### Leveraging a Foundation Model

Instead of training a feature extractor solely on matching pairs, RoMa **adopts a pretrained, frozen DINOv2 ViT-Large backbone**. Trained via large-scale self-supervised learning on diverse imagery, DINOv2 supplies rich semantic and geometric representations that hold up under drastic viewpoint, weather, and seasonal shifts, leaving RoMa to learn only the cross-view matching and refinement heads on top.

#### Architecture

RoMa pairs coarse semantic reasoning with fine geometric precision in four tightly linked components:

1. **Frozen DINOv2 Backbone**: Extracts dense patch descriptors at 1/14 input resolution from a frozen DINOv2 ViT-Large.
2. **Coarse Matching (Warp Estimation)**: A transformer decoder applies cross-attention over the DINOv2 tokens, and rather than regressing a single deterministic point in image B for each location in image A, predicts a multimodal **probability distribution**:

$$p(\mathbf{x}_B | \mathbf{x}_A) = \sum_{k} w_k \mathcal{N}(\mathbf{x}_B; \mu_k, \Sigma_k)$$

Representing correspondences as a mixture distribution explicitly captures multimodal ambiguity on symmetric or repetitive surfaces.

3. **Fine Matching (Iterative Refinement)**: Starting from the coarse warp field, dedicated CNN fine features iteratively refine the displacement across increasing resolutions by computing local correlations and predicting residual updates—echoing RAFT's recurrent refinement philosophy, adapted from optical flow to wide-baseline dense warp.
4. **Certainty Estimation**: Alongside the warp field, the network outputs a pixel-wise certainty score so downstream RANSAC or pose solvers can sample only reliable correspondences.

#### Robust Regression

Instead of penalizing coordinate errors with an unweighted L2 loss, RoMa trains the warp distribution by minimizing the **negative log-likelihood** of the ground-truth target $\mathbf{x}_B^*$:

$$L = -\sum_{(\mathbf{x}_A, \mathbf{x}_B^*)} \log p(\mathbf{x}_B^* | \mathbf{x}_A)$$

Predicting a full distribution allows uncertain regions to spread probability mass rather than skewing a single deterministic point. How robust that loss is to noisy or occluded ground truth still depends on the chosen distribution family: light-tailed Gaussians still penalize large errors quadratically, so curbing extreme outliers requires heavy-tailed distributions, an explicit outlier component, or discretizing regression into classification bins.

#### Performance

Across relative-pose benchmarks in the RoMa paper, the model outperforms the tested LoFTR- and ASpanFormer-family baselines on MegaDepth and ScanNet (with margins depending on dataset and angular threshold), and stands out on extreme wide-baseline and appearance changes with a reported 36% relative improvement on WxBS. Using the `romatch` package, running dense matching and sampling high-certainty pairs takes only a few lines:

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

RoMa fuses two major ideas—**foundation-model representations** (freezing DINOv2 rather than learning features from scratch) and **probabilistic warp modeling** (predicting uncertainty-aware distributions instead of single coordinates)—effectively uniting LoFTR's detector-free transformer matching, RAFT's iterative refinement, and large-scale self-supervised visual priors.

### 5.7.6 Recent Developments: 3D-Aware Dense Matching (2024-2025)

Up through RoMa, dense matchers still followed a strictly 2D-first mindset: match pixels in image space, then triangulate 3D geometry afterward. Between 2024 and 2025, a new wave of architectures inverted that pipeline by **predicting 3D structure directly in a feed-forward pass and treating 2D correspondences as a byproduct**:

- **DUSt3R (Wang et al., 2024)**: [DUSt3R](https://arxiv.org/abs/2312.14132) takes an uncalibrated image pair with no known poses and directly regresses aligned 3D pointmaps in a shared coordinate frame. Rather than matching in 2D first and reconstructing later, correspondences fall straight out of nearest-neighbor or reciprocal lookup in the predicted 3D pointmaps.
- **MASt3R (Leroy et al., 2024)**: [MASt3R](https://arxiv.org/abs/2406.09756) augments DUSt3R with a dense local-feature head trained specifically for metric correspondence, reporting a 30-percentage-point VCRE AUC gain over the prior baseline under the paper's map-free localization protocol.
- **VGGT (Wang et al., 2025)**: [VGGT](https://arxiv.org/abs/2503.11651) (Visual Geometry Grounded Transformer), winner of the CVPR 2025 Best Paper award, scales feed-forward 3D inference to one or more views in a single pass—simultaneously predicting camera parameters, depth maps, 3D pointmaps, and 3D point tracks. Under its reported evaluation settings, VGGT reconstructs scenes in under a second and outperforms tested methods that rely on downstream geometric optimization.

Together, DUSt3R, MASt3R, and VGGT mark a conceptual shift: instead of asking "which 2D pixels match?" and hoping 3D geometry follows, modern networks solve the 3D scene directly and read correspondences off the resulting geometry.

---

## 5.8 3D-3D Correspondence

Alongside 2D image matching, a parallel branch of research addresses **finding correspondences directly between 3D point clouds**. Whenever a robot aligns consecutive LiDAR scans, merges maps from multiple sessions, or verifies a loop closure in 3D geometry, it faces the 3D-3D correspondence problem—where points carry coordinates and surface normals rather than rich RGB texture.

### 5.8.1 FPFH (Fast Point Feature Histograms)

Proposed by [Rusu et al. (2009)](https://ieeexplore.ieee.org/document/5152473), FPFH is a classic handcrafted 3D descriptor that **summarizes local surface curvature and normal geometry around a point as an SE(3)-invariant histogram**.

#### SPFH (Simple Point Feature Histogram)

To make the descriptor invariant to rigid 3D rotation and translation, FPFH first defines an intrinsic Darboux-style local coordinate frame $(\mathbf{u}, \mathbf{v}, \mathbf{w})$ between a query point $\mathbf{p}$ (with surface normal $\mathbf{n}_p$) and each neighbor $\mathbf{p}_k$ (with normal $\mathbf{n}_k$ and displacement $\mathbf{d} = \mathbf{p}_k - \mathbf{p}$):

$$\mathbf{u} = \mathbf{n}_p$$
$$\mathbf{v} = \mathbf{u} \times \frac{\mathbf{d}}{\|\mathbf{d}\|}$$
$$\mathbf{w} = \mathbf{u} \times \mathbf{v}$$

Projecting the neighbor's normal into this local frame yields three rigid-motion-invariant angular features:

$$\alpha = \mathbf{v} \cdot \mathbf{n}_k, \quad \phi = \mathbf{u} \cdot \frac{\mathbf{d}}{\|\mathbf{d}\|}, \quad \theta = \text{atan2}(\mathbf{w} \cdot \mathbf{n}_k, \mathbf{u} \cdot \mathbf{n}_k)$$

Quantizing each angle into a $B$-bin histogram forms the Simple Point Feature Histogram (SPFH) around $\mathbf{p}$.

#### FPFH: An Accelerated Version of PFH

The original Point Feature Histogram (PFH) evaluated angular triplets across *every pair* within a $k$-neighborhood, costing $O(k^2)$ per point. FPFH cuts that complexity to $O(k)$ by computing SPFH only between the query and its immediate neighbors, then blending in the precomputed SPFHs of those neighbors weighted by inverse distance $w_i = \|\mathbf{p}_i - \mathbf{p}\|$:

$$\text{FPFH}(\mathbf{p}) = \text{SPFH}(\mathbf{p}) + \frac{1}{k} \sum_{i=1}^{k} \frac{1}{w_i} \text{SPFH}(\mathbf{p}_i)$$

With 11 bins per angle, FPFH produces a compact 33-dimensional descriptor (3 angles × 11 bins). In Open3D, computing normals and FPFH descriptors takes only a few lines:

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

Just as SIFT eventually gave way to learned 2D descriptors, handcrafted surface histograms struggled with noisy, sparse, or partially occluded scans. [Zeng et al. (2017)](https://arxiv.org/abs/1603.08182) ushered in learned 3D features with **3DMatch**, mining training correspondences automatically from RGB-D reconstructions across 62 indoor scenes, feeding local 3D Truncated Distance Function (TDF) voxel volumes into a 3D CNN, and outputting a 512-dimensional local descriptor. Beyond the network itself, its **3DMatch Benchmark** became the standard proving ground for 3D registration research.

### 5.8.3 FCGF (Fully Convolutional Geometric Features, 2019)

Processing a separate 3D voxel patch around every keypoint made 3DMatch slow on dense scans. [Choy et al. (2019)](https://arxiv.org/abs/1904.09793) solved that bottleneck in FCGF by switching to **sparse 3D convolutions**, extracting compact 32-dimensional descriptors for an entire point cloud in a single fully convolutional pass. Under the paper's implementation, hardware, and benchmarks, this whole-cloud design delivered major runtime speedups alongside strong registration accuracy.

### 5.8.4 Predator (2021): Overlap-Aware 3D Matching

Even strong 3D descriptors fail when two scans barely overlap: extracting features uniformly across non-overlapping walls floods the matcher with unmatchable points. [Huang et al. (2021)](https://arxiv.org/abs/2011.13005) addressed this directly in Predator by **explicitly predicting where two point clouds overlap** before matching:

1. **Overlap attention**: Cross-attention between the two clouds estimates each point's probability of lying in the shared region.
2. **Matchability score**: A second score filters for geometrically distinctive points within that overlap zone.
3. **Low-overlap gains**: Focusing sampling on overlapping, salient geometry produces major improvements in challenging 10–30% overlap regimes.

### 5.8.5 GeoTransformer (2022): Geometric Transformer

Traditional 3D registration still leaned heavily on RANSAC to survive false point matches, which becomes slow as point counts rise or inlier ratios collapse. [Qin et al. (2022)](https://arxiv.org/abs/2202.06688) introduced GeoTransformer to **learn rigid-transformation-invariant features and recover the 3D alignment directly without RANSAC**.

#### Keypoint-Free Superpoint Matching

Rather than relying on brittle 3D keypoint detectors, GeoTransformer first matches coarse downsampled **superpoints** and then propagates those matches down to dense local points.

#### Geometric Transformer Architecture

Absolute $(x, y, z)$ coordinates change whenever a scan rotates or translates, so a vanilla positional encoding would break rigid invariance. GeoTransformer instead builds two **SE(3)-invariant geometric encodings** from relative distances and angles:

**1. Pairwise Distance Encoding**: Because Euclidean distance $d_{ij} = \|\mathbf{p}_i - \mathbf{p}_j\|$ between superpoints $\mathbf{p}_i$ and $\mathbf{p}_j$ is invariant under any rigid motion, it enters as a sinusoidal embedding:

$$\text{PE}(d_{ij}) = [\sin(\omega_1 d_{ij}), \cos(\omega_1 d_{ij}), \ldots, \sin(\omega_K d_{ij}), \cos(\omega_K d_{ij})]$$

**2. Triplet Angle Encoding**: Encoding the angles formed by point triplets $\mathbf{p}_i, \mathbf{p}_j, \mathbf{p}_k$ adds higher-order local shape awareness while preserving full rigid invariance.

Injecting these encodings directly as learned attention biases inside the transformer self-attention layers keeps geometric structure front and center:

$$\text{Attn}_{ij} = \frac{\mathbf{q}_i^\top \mathbf{k}_j}{\sqrt{d}} + b(\text{PE}(d_{ij}))$$

#### RANSAC-Free Transformation Estimation

From the matched superpoints, GeoTransformer generates local transformation candidates over small patches and selects the globally consistent rigid transform via local-to-global registration, removing the need for an external RANSAC loop. The authors report substantial speedups under their evaluation setup, though performance on differently contaminated correspondence sets still depends on the data distribution.

#### Performance

On the low-overlap 3DLoMatch benchmark under the paper's evaluation protocol, GeoTransformer reports a 17-to-30 percentage-point gain in inlier ratio over compared baselines and more than 7 points of improvement in registration recall—metrics tied to that benchmark's low-overlap regime. Running inference in PyTorch produces both correspondences and the estimated $4 \times 4$ rigid transform directly:

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

Over roughly a decade, 3D point-cloud correspondence traced an arc remarkably parallel to 2D matching—progressing from handcrafted local histograms to sparse whole-cloud CNNs, overlap-aware attention, and RANSAC-free geometric transformers:

```
[Handcrafted 3D Descriptors]
FPFH (2009)          — geometric angular histograms
SHOT (2010)          — orientation histograms
    ↓ learning-based
[Learned 3D Descriptors]
3DMatch (2017)       — early learned 3D descriptor + benchmark
    ↓ efficiency
FCGF (2019)          — sparse conv, processes entire point cloud at once
    ↓ overlap-aware
Predator (2021)      — overlap-aware cross-attention
    ↓ RANSAC removal
GeoTransformer (2022) — geometric invariant transformer, RANSAC-free
```

Yet both 2D-2D and 3D-3D matching still enjoy one luxury: both inputs speak the same modality. What happens when a camera image must be matched directly against a 3D LiDAR point cloud?

---

## 5.9 Cross-Modal Correspondence

### 5.9.1 2D-3D Matching: Camera-LiDAR

In multi-sensor robotics, one of the hardest correspondence problems crosses sensor boundaries altogether: **matching a 2D camera image directly against a 3D LiDAR point cloud**. Solving that link unlocks three core capabilities:

- **Camera-LiDAR extrinsic calibration**: Identifying the same physical corners, edges, or planes in both sensors to estimate their relative 6-DOF transform.
- **Visual localization against a LiDAR map**: Localizing an onboard camera image directly inside a pre-built 3D LiDAR map via 2D-3D PnP.
- **Cross-modal loop closure**: Validating place recognition between camera keyframes and LiDAR submaps.

### 5.9.2 Image-to-Point Cloud Matching Approaches

Because images and point clouds live in fundamentally different spaces, systems bridge the gap either by projecting 3D geometry into 2D or by learning a shared latent embedding.

#### Projection-Based Approach

The most practical strategy reduces 2D-3D cross-modal matching back into a 2D-2D problem via virtual rendering:

1. Project the 3D LiDAR point cloud onto a virtual camera plane to synthesize 2D depth and reflectance-intensity images.
2. Run a learned 2D matcher such as SuperGlue between that rendered view and the real camera frame.
3. Lift the surviving 2D pixel matches back onto their underlying 3D LiDAR points.

The calibration toolbox of Koide et al. (2023) relies on this exact recipe, rendering dense LiDAR intensity images and matching them to camera frames with SuperGlue to gather automatic 2D-3D calibration correspondences.

#### Learning-Based Direct Matching

An alternative line of work bypasses intermediate rendering and learns cross-modal embeddings end-to-end. **LCD (Learned Cross-Domain Descriptors)** maps 2D image patches and 3D point-cloud patches into a shared metric space, while **P2-Net (Wang et al., 2021)** directly infers pixel-to-point correspondences between raw image pixels and 3D points.

### 5.9.3 Why Cross-Modal Is Difficult: The Representation Gap

Four structural mismatches explain why 2D-3D cross-modal correspondence lags behind unimodal (2D-2D or 3D-3D) matching:

1. **Representation heterogeneity**: Images store color or brightness on a regular 2D pixel grid, whereas LiDAR scans store irregular $(x, y, z)$ coordinates paired with sparse intensity values—leaving standard CNNs and point networks incompatible without translation.
2. **Information asymmetry**: Cameras capture dense photometric texture without direct depth, while LiDAR captures metric 3D geometry with almost no visual texture.
3. **Density gap**: A camera packs millions of pixels uniformly across its field of view, whereas LiDAR emits tens to hundreds of thousands of returns whose spatial spacing thins rapidly with range.
4. **Appearance domain gap**: The two sensors measure distinct physical quantities of the same surface. A camera records brightness shaped by ambient illumination and surface reflectance, whereas LiDAR measures active laser return intensity at a narrow infrared wavelength; neither sensor isolates true surface albedo directly.

Because of this four-way gap, cross-modal matching remains less mature than unimodal pipelines. In practice, systems either bypass local appearance altogether with information-theoretic alignment (Section 5.4) or project 3D scans into synthetic 2D views so mature 2D matchers can operate.

---

## 5.10 Dense Matching & Optical Flow

Up to this point, most pipelines have focused on **sparse correspondences**—whether handcrafted corners or learned keypoints—with RoMa and DUSt3R extending into wide-baseline dense warps. When consecutive video frames or stereo pairs share small, continuous motion, however, robotics often needs **dense per-pixel correspondence**, known as **optical flow** across time and **dense disparity** across calibrated stereo views.

### 5.10.1 Classical Optical Flow: Lucas-Kanade, Horn-Schunck

#### Definition of Optical Flow

Optical flow models the **apparent 2D motion of brightness patterns across consecutive frames** as a vector field over image coordinates $(x, y)$:

$$\mathbf{u}(x, y) = (u(x, y), v(x, y))$$

Classical formulations start from the **brightness constancy assumption**, which assumes a scene point keeps the same pixel intensity as it moves by $(u, v)$ over a small step from frame $t$ to $t+1$:

$$I(x, y, t) = I(x + u, y + v, t + 1)$$

For small displacements, taking a first-order Taylor expansion and canceling $I(x, y, t)$ yields the linear **optical flow constraint equation**:

$$I_x u + I_y v + I_t = 0$$

where $I_x = \frac{\partial I}{\partial x}$, $I_y = \frac{\partial I}{\partial y}$, and $I_t = \frac{\partial I}{\partial t}$ are the spatial and temporal intensity derivatives. Because a single equation cannot pin down two unknowns $(u, v)$—known geometrically as the **aperture problem**, since a small window over a straight edge only sees motion perpendicular to the edge—classical algorithms add spatial constraints to solve for $(u, v)$.

#### Lucas-Kanade (1981)

Lucas and Kanade resolve the aperture problem through a **local constancy assumption**: instead of solving per pixel in isolation, they assume every pixel inside a small local window $\Omega$ moves with the same velocity $(u, v)$. Stacking the optical flow constraint across all $n$ pixels in $\Omega$ turns one underdetermined equation into an overdetermined linear system:

$$\begin{bmatrix} I_{x_1} & I_{y_1} \\ I_{x_2} & I_{y_2} \\ \vdots & \vdots \\ I_{x_n} & I_{y_n} \end{bmatrix} \begin{bmatrix} u \\ v \end{bmatrix} = -\begin{bmatrix} I_{t_1} \\ I_{t_2} \\ \vdots \\ I_{t_n} \end{bmatrix}$$

Writing this concisely as $\mathbf{A} \mathbf{u} = -\mathbf{b}$, the least-squares solution is:

$$\begin{bmatrix} u \\ v \end{bmatrix} = (\mathbf{A}^\top \mathbf{A})^{-1} \mathbf{A}^\top (-\mathbf{b})$$

Notice the matrix being inverted: $\mathbf{A}^\top \mathbf{A} = \begin{bmatrix} \sum I_x^2 & \sum I_x I_y \\ \sum I_x I_y & \sum I_y^2 \end{bmatrix}$ is algebraically identical to the Harris corner **structure tensor** $\mathbf{M}$. That connection explains why Lucas-Kanade tracking succeeds only where $\mathbf{A}^\top \mathbf{A}$ is well-conditioned—meaning corners with strong gradients in both directions—and why practical trackers select initial points with `goodFeaturesToTrack` (Shi-Tomasi/Harris corners).

Because the first-order Taylor approximation breaks down once frame-to-frame motion exceeds a pixel or two, practical implementations run **Pyramidal Lucas-Kanade**, solving coarse-to-fine across a Gaussian image pyramid so large displacements shrink to sub-pixel steps at the coarsest level:

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

Rather than tracking isolated corner patches, Horn and Schunck impose a **global smoothness assumption** across the entire image, minimizing a variational energy that balances brightness constancy against spatial flow smoothness:

$$E = \iint \left[ (I_x u + I_y v + I_t)^2 + \lambda^2 (\|\nabla u\|^2 + \|\nabla v\|^2) \right] dx \, dy$$

The first term penalizes violations of brightness constancy, while the second penalizes sharp spatial variations in flow weighted by $\lambda$. Solving the corresponding Euler-Lagrange equations yields an iterative dense update rule across every pixel—including textureless interiors where Lucas-Kanade fails—though global quadratic smoothness tends to blur sharp motion discontinuities along foreground object boundaries.

### 5.10.2 Learning-Based Optical Flow: FlowNet → RAFT → FlowFormer → UniMatch

Classical variational flow struggles whenever brightness constancy fails under shadows, motion blur, or fast multi-pixel jumps. Deep optical flow networks replaced hand-crafted gradient equations with learned feature matching and recurrent refinement.

#### FlowNet / FlowNet 2.0 (2015/2017)

[Dosovitskiy et al. (2015)](https://arxiv.org/abs/1504.06852) demonstrated with FlowNet that an encoder-decoder CNN trained end-to-end on synthetic rendered sequences can regress dense optical flow directly from two raw images. [FlowNet 2.0 (Ilg et al., 2017)](https://arxiv.org/abs/1612.01925) subsequently cascaded multiple FlowNet modules with intermediate image warping, closing much of the accuracy gap against classical variational baselines.

#### RAFT (2020): A New Standard for Optical Flow

Coarse-to-fine pyramid networks suffered from a persistent blind spot: if a small, fast-moving object vanished or mismatched at the coarsest scale, finer levels could rarely recover it. [Teed & Deng (2020)](https://arxiv.org/abs/2003.12039) reshaped the field in **RAFT**—winner of the ECCV 2020 Best Paper award—by pairing an **all-pairs 4D correlation volume with recurrent ConvGRU updates** at a fixed high resolution:

1. **Feature Encoder**: A ResNet-style CNN extracts 1/8-resolution feature maps $\mathbf{g}_1, \mathbf{g}_2 \in \mathbb{R}^{H/8 \times W/8 \times D}$ from both frames, while a separate Context Encoder extracts initial hidden states and context features from the first frame.
2. **Correlation Volume Construction**: Rather than restricting search to a local window, RAFT evaluates the inner product between every pixel pair across the two feature maps:

$$C_{ijkl} = \sum_d g_1(i, j, d) \cdot g_2(k, l, d)$$

This produces a 4D correlation volume $\mathbf{C} \in \mathbb{R}^{H/8 \times W/8 \times H/8 \times W/8}$ that is average-pooled over its last two axes to form a 4-level **correlation pyramid** (strides 1, 2, 4, 8). Crucially, **multi-scale correlation lookup happens at a single 1/8 spatial resolution**, rather than stepping through a coarse-to-fine image pyramid.
3. **Iterative Update (GRU)**: Starting from zero flow, a ConvGRU operator refines the flow field over $K$ steps: at each iteration $k$, it looks up a local neighborhood around the current estimate $\mathbf{f}^k$ across all levels of the correlation pyramid, fuses the correlation slice with current flow and context features, updates the hidden state $\mathbf{h}^k$, predicts a residual $\Delta \mathbf{f}$, and sets $\mathbf{f}^{k+1} = \mathbf{f}^k + \Delta \mathbf{f}$.

The reference configuration runs 12 iterations during training and 32 on Sintel or 24 on KITTI at evaluation; while extra iterations reduced error in some paper experiments, gains are not guaranteed past convergence, so iteration budgets should be tuned against latency and validation error.

**All-Pairs vs. Coarse-to-Fine**: By retaining all-pairs feature correlations and querying them across multiple pooling scales at every recurrent step, RAFT keeps large-displacement matches accessible instead of losing fast objects at coarse pyramid levels—though mismatches can still occur if local features are ambiguous or the recurrent update settles on the wrong peak.

**Training**: Supervision applies an L1 loss against ground-truth flow at every iteration with exponentially increasing weights on later refinements:

$$L = \sum_{k=1}^{K} \gamma^{K-k} \| \mathbf{f}^k - \mathbf{f}^{gt} \|_1$$

with $\gamma = 0.8$. In `torchvision`, running pretrained `raft_large` takes only a few lines:

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

[FlowFormer (Huang et al., 2022)](https://arxiv.org/abs/2203.16194) upgrades RAFT's correlation lookup by tokenizing the 4D cost volume into a compact latent cost memory with a transformer encoder, then querying that memory with a recurrent transformer decoder to report higher accuracy than RAFT.

#### UniMatch (2023)

Recognizing that optical flow, rectified stereo, and two-view depth all solve dense correspondence, [Xu et al. (2023)](https://arxiv.org/abs/2211.05783) unified **optical flow, stereo matching, and depth estimation within a single transformer framework** in UniMatch, switching only the geometry of cross-view matching across tasks.

### 5.10.3 Dense Stereo: SGM → RAFT-Stereo → UniMatch

In calibrated stereo rigs, epipolar rectification collapses correspondence search onto horizontal scanlines: estimating **horizontal disparity** is simply optical flow constrained to zero vertical displacement.

#### SGM (Semi-Global Matching, Hirschmuller 2005)

Long the workhorse of classical real-time stereo, SGM estimates dense disparity in three deterministic stages: compute per-pixel matching costs across every candidate disparity, aggregate those costs via 1D dynamic programming along 8 (or 16) radial scanline paths, and pick the minimum aggregated cost per pixel. Along path direction $r$, the recursive path cost is:

$$L_r(p, d) = C(p, d) + \min\begin{cases} L_r(p-r, d) \\ L_r(p-r, d-1) + P_1 \\ L_r(p-r, d+1) + P_1 \\ \min_i L_r(p-r, i) + P_2 \end{cases}$$

where $P_1$ penalizes small one-pixel disparity steps along slanted surfaces and $P_2$ penalizes sharp depth jumps at object boundaries.

#### RAFT-Stereo (2021)

RAFT-Stereo adapts RAFT directly to rectified stereo: because matching restricts itself to horizontal epipolar lines, the 4D all-pairs correlation volume collapses to a 3D volume of size $H \times W \times D$, over which a multi-level GRU refines disparity iteratively.

#### Unification in UniMatch

UniMatch completes the unification by sharing one transformer backbone and tailoring cross-attention correlation solely to the geometric constraint of each task:

- **Stereo**: 1D horizontal correlation along epipolar scanlines.
- **Flow**: 2D all-pairs correlation across unconstrained image motion.
- **Depth**: Cross-view matching constrained by known camera poses.

### 5.10.4 Technical Flow: From Sparse Feature to Dense Correspondence

The progression from sparse corners to unified dense matchers illustrates how local tracking and global correlation eventually converged:

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
    ↓ global correlation + iterative updates
RAFT (2020)          — all-pairs correlation + iterative GRU
    ↓ transformer
FlowFormer (2022)    — cost volume tokenization + transformer
    ↓ task unification
UniMatch (2023)      — flow + stereo + depth unified
```

RAFT's core design—all-pairs feature correlation paired with iterative residual updates—reached far beyond optical flow, directly inspiring **detector-free feature matchers** (LoFTR, RoMa) and **learned dense SLAM** systems such as DROID-SLAM.

---

## Summary

The diagram below brings together every branch of correspondence covered in this chapter—from classical 2D corners and RANSAC to mutual information, 3D-3D point-cloud transformers, dense optical flow, and 3D-aware foundation matchers:

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

═══════════════════════ Shift to Detector-Free Matching ═══════════════════

[Detector-Free Matching]

    LoFTR (2021) ──→ QuadTree (2022) ──→ ASpanFormer (2022) ──→ RoMa (2024)
      transformer         O(N)        adaptive span          DINOv2
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

Summary:
  1. A flow that replaces the traditional three-stage pipeline
     (detect → describe → match) with deep learning one step at a time:
     SIFT → SuperPoint → SuperGlue → LightGlue

  2. A flow that dismantles the pipeline itself and transitions to
     end-to-end dense matching: RAFT → LoFTR → RoMa

  3. Foundation models: fixing DINOv2 features as the basis for matching
     and training the matching logic on top of them.

  4. The emergence of 3D-aware matching (2024-2025): DUSt3R → MASt3R →
     VGGT reverse the usual order. They infer 3D structure directly and
     derive correspondences from that structure, rather than matching in
     2D before reconstruction.

  Approaches 2, 3, and 4 extend matching beyond a fixed detector-descriptor
  pipeline, while approach 1 remains useful when efficiency and
  interpretability matter.
```

With the state-estimation machinery of Chapter 4 and the correspondence methods of this chapter in place, the next chapters put both pieces to work inside complete front-end/back-end systems: visual and visual-inertial odometry (→ Chapter 6) build directly on ORB descriptors, Lucas-Kanade optical flow, and RANSAC geometric verification, while LiDAR odometry (→ Chapter 7) extends 3D-3D correspondence and scan registration into real-time SLAM.
