# Ch.6 — Visual Odometry & Visual-Inertial Odometry

Ch.4 covered the state estimation framework and Ch.5 covered feature matching techniques. We now examine the first system in which these two are actually combined: Visual Odometry (VO) and Visual-Inertial Odometry (VIO).

Visual Odometry (VO) estimates the ego-motion of a camera from image data alone, while Visual-Inertial Odometry (VIO) couples it with an IMU to gain scale observability and robustness. This chapter examines the internal structure of VO/VIO and the design choices involved in depth.

The origin of VO traces back to [Nistér et al. (2004)](https://doi.org/10.1109/CVPR.2004.1315094). That paper first defined the term "Visual Odometry" and presented a real-time ego-motion estimation system for stereo and monocular cameras. The stereo approach triangulated 3D points from the left/right cameras and then estimated the rigid-body transform between frames with a 3-point algorithm; the monocular approach estimated the Essential Matrix with a 5-point algorithm. This basic pipeline — feature detection → matching → RANSAC → motion estimation — still forms the skeleton of feature-based VO two decades later.

VO/VIO systems can be classified along three main axes:

1. **Feature-based vs Direct**: whether geometric features (corners, edges) are extracted and matched, or pixel intensities themselves are used directly.
2. **Filter vs Optimization**: whether state estimation relies on Kalman-filter-family methods or nonlinear optimization.
3. **Loosely coupled vs Tightly coupled**: whether IMU and camera are processed independently and their results fused, or their raw measurements are placed in a single optimization problem.

The combinations of these three axes have produced a wide variety of systems. This chapter dissects representative systems one by one, analyzing the rationale and consequences of each design choice.

---

## 6.1 Feature-based Visual Odometry

Feature-based VO extracts geometric features from images, establishes inter-frame correspondences, and estimates camera motion from them. It is the oldest and best-understood VO paradigm, and it remains the most widely deployed today through ORB-SLAM3.

### 6.1.1 Frontend: Detection, Tracking, Outlier Rejection

The frontend of a feature-based VO performs three core tasks.

**Feature Detection**

This step finds trackable points in a frame. Ideal feature points must have high repeatability — the same 3D point should be detected at similar locations when imaged from different viewpoints.

The Harris corner detector detects corners based on the autocorrelation matrix (also called the second moment matrix) $\mathbf{M}$ of an image patch:

$$\mathbf{M} = \sum_{(x,y) \in W} w(x,y) \begin{bmatrix} I_x^2 & I_xI_y \\ I_xI_y & I_y^2 \end{bmatrix}$$

Here $I_x, I_y$ are the image gradients, $W$ is the window, and $w$ is the weighting function. A pixel is labeled a corner if both eigenvalues of $\mathbf{M}$ are large. The Harris response function is:

$$R = \det(\mathbf{M}) - k \cdot \text{tr}(\mathbf{M})^2 = \lambda_1\lambda_2 - k(\lambda_1 + \lambda_2)^2$$

FAST (Features from Accelerated Segment Test) is a detector optimized for speed. It declares a candidate pixel $p$ a corner if at least $n$ (typically $n=9$) contiguous pixels on a radius-3 Bresenham circle around $p$ are all brighter or all darker than $p$. A pre-test examines only pixels 1, 5, 9, and 13 first to quickly reject non-corners. It is tens of times faster than Harris but is noise-sensitive and lacks orientation/scale invariance.

ORB (Oriented FAST and Rotated BRIEF) augments FAST detection with orientation and rotates the BRIEF descriptor accordingly, yielding features suited to real-time SLAM. Orientation is computed from the intensity centroid of the image patch:

$$\theta = \text{atan2}(m_{01}, m_{10}), \quad m_{pq} = \sum_{x,y} x^p y^q I(x,y)$$

The ORB-SLAM family extracts ORB features from an image pyramid (typically 8 levels, scale factor 1.2) to obtain scale invariance.

**Feature Tracking**

There are two ways to establish inter-frame correspondences:

1. **Descriptor matching**: detect features independently in each frame and match them by descriptor distance (Hamming distance for binary descriptors, L2 for float descriptors). ORB-SLAM3 uses this approach. A DBoW2 vocabulary tree restricts match candidates to speed up matching.

2. **Optical Flow Tracking**: the Lucas-Kanade (LK) tracker tracks where each feature point of the previous frame moves in the current frame. VINS-Mono uses this approach. Under the brightness constancy assumption:

$$I(x + u, y + v, t + \Delta t) = I(x, y, t)$$

A first-order Taylor expansion followed by least squares within a window $W$ yields:

$$\begin{bmatrix} u \\ v \end{bmatrix} = \left(\sum_W \begin{bmatrix} I_x^2 & I_xI_y \\ I_xI_y & I_y^2 \end{bmatrix}\right)^{-1} \sum_W \begin{bmatrix} -I_xI_t \\ -I_yI_t \end{bmatrix}$$

Running this coarse-to-fine over an image pyramid makes it possible to track even large displacements.

Trade-off between the two approaches: descriptor matching is strong under wide baselines (large viewpoint changes) and can be reused for loop closure, but detection + description + matching costs time. LK tracking is fast and offers subpixel accuracy, but tends to fail under large viewpoint changes.

**Outlier Rejection**

Mismatches (outliers) are inevitable in matching results. If they are not removed, motion estimation can be thrown off badly.

The basic approach is RANSAC (Random Sample Consensus). A model is estimated from a minimal sample, inliers are counted across the full data, and the model with the most inliers is retained. In VO:

- **2D-2D**: estimate the Essential Matrix $\mathbf{E}$ with the 5-point algorithm. Decompose $\mathbf{E} = [\mathbf{t}]_\times \mathbf{R}$ to obtain the relative pose $(R, t)$.
- **3D-2D**: PnP (Perspective-n-Point) algorithm + RANSAC. Estimate the absolute pose from correspondences between already triangulated 3D points and current 2D observations.
- **3D-3D**: an ICP variant estimates the rigid-body transform.

The core equation for outlier rejection using the epipolar constraint is:

$$\mathbf{p}_2^T \mathbf{E} \mathbf{p}_1 = 0$$

where $\mathbf{p}_1, \mathbf{p}_2$ are normalized image coordinates. Correspondences that fail to satisfy this equation are labeled outliers.

ORB-SLAM3 estimates the Fundamental Matrix and a Homography in parallel and picks the appropriate model according to scene structure (planar vs non-planar). For planar scenes, the Homography has fewer degrees of freedom and is thus more stable.

### 6.1.2 Backend: PnP, Motion-only BA, Local BA

Once the frontend provides correspondences, the backend uses them to estimate camera poses and 3D structure.

**PnP (Perspective-n-Point)**

Given correspondences between already triangulated 3D map points $\mathbf{P}_j \in \mathbb{R}^3$ and 2D observations $\mathbf{u}_j \in \mathbb{R}^2$ in the current frame, the problem is to estimate the camera pose $\mathbf{T} = (\mathbf{R}, \mathbf{t}) \in SE(3)$:

$$\mathbf{u}_j = \pi(\mathbf{R}\mathbf{P}_j + \mathbf{t})$$

Here $\pi: \mathbb{R}^3 \to \mathbb{R}^2$ is the camera projection function. A minimal solution uses four points (three points plus one for verification in the case of P3P), combined with RANSAC to handle outliers.

Once an initial solution is obtained, a nonlinear refinement is performed over all inliers.

**Motion-only Bundle Adjustment**

After obtaining an initial pose from PnP, this step fixes the 3D point positions and optimizes only the camera pose:

$$\mathbf{T}^* = \underset{\mathbf{T}}{\arg\min} \sum_{j} \rho\left(\left\|\mathbf{u}_j - \pi(\mathbf{T} \cdot \mathbf{P}_j)\right\|^2_{\Sigma_j}\right)$$

where $\rho(\cdot)$ is a robust kernel (Huber, etc.) and $\Sigma_j$ is the observation covariance. The Huber function is:

$$\rho(s) = \begin{cases} s & \text{if } s \leq \delta^2 \\ 2\delta\sqrt{s} - \delta^2 & \text{if } s > \delta^2 \end{cases}$$

This optimization is 6-DoF (optimization on SE(3)), so we update via an increment $\boldsymbol{\xi} \in \mathbb{R}^6$ in the Lie algebra $\mathfrak{se}(3)$:

$$\mathbf{T} \leftarrow \exp(\boldsymbol{\xi}^{\wedge}) \cdot \mathbf{T}$$

**Local Bundle Adjustment**

This jointly optimizes the most recent $N$ keyframes together with the map points they observe:

$$\{\mathbf{T}_i^*, \mathbf{P}_j^*\} = \underset{\{\mathbf{T}_i, \mathbf{P}_j\}}{\arg\min} \sum_{i,j} \rho\left(\left\|\mathbf{u}_{ij} - \pi(\mathbf{T}_i \cdot \mathbf{P}_j)\right\|^2_{\Sigma_{ij}}\right)$$

This is the canonical form of Bundle Adjustment (BA). "Bundle" refers to the bundles of rays emanating from each 3D point toward the cameras. The normal equations of BA have a special sparse structure (Schur complement structure):

$$\begin{bmatrix} \mathbf{H}_{cc} & \mathbf{H}_{cp} \\ \mathbf{H}_{pc} & \mathbf{H}_{pp} \end{bmatrix} \begin{bmatrix} \delta\boldsymbol{\xi} \\ \delta\mathbf{p} \end{bmatrix} = \begin{bmatrix} \mathbf{b}_c \\ \mathbf{b}_p \end{bmatrix}$$

$\mathbf{H}_{pp}$ is block-diagonal (each point is independent), so eliminating the point variables via the Schur complement yields:

$$(\mathbf{H}_{cc} - \mathbf{H}_{cp}\mathbf{H}_{pp}^{-1}\mathbf{H}_{pc})\delta\boldsymbol{\xi} = \mathbf{b}_c - \mathbf{H}_{cp}\mathbf{H}_{pp}^{-1}\mathbf{b}_p$$

The size of this reduced system depends only on the number of cameras (independent of the number of points) and can be solved efficiently. This is the key reason BA handles tens of thousands of points while still approaching real-time speed.

### 6.1.3 ORB-SLAM3 Architecture Deep Dive

[ORB-SLAM3 (Campos et al., 2021)](https://doi.org/10.1109/TRO.2021.3075644) is the current de facto standard for feature-based visual(-inertial) SLAM. In a single framework it supports monocular, stereo, and RGB-D cameras as well as IMUs, and it accommodates both pinhole and fisheye lens models.

**Overall Architecture**

ORB-SLAM3 consists of three parallel threads:

1. **Tracking Thread**: extracts ORB features in every frame, matches them with existing map points to estimate the current pose, and refines the pose with motion-only BA.
2. **Local Mapping Thread**: when a new keyframe is inserted, it triangulates new map points and performs local BA. Redundant keyframes and map points are culled to keep the map compact.
3. **Loop Closing & Map Merging Thread**: detects loop candidates with DBoW2 and verifies them via Sim(3) (or SE(3)) registration. Once confirmed, pose graph optimization is run and then a full BA.

**Tracking Thread in Detail**

1. Initial pose prediction from the previous frame: predict the current pose with a constant velocity model or IMU preintegration.
2. Using the predicted pose, project map points onto the current frame and find correspondences via ORB matching.
3. Refine the pose with motion-only BA.
4. Decide whether to insert a keyframe. Keyframe conditions: (a) a minimum number of frames has elapsed since the last keyframe, (b) the fraction of tracked map points in the current frame is below a threshold, and (c) the current frame observes a sufficient number of map points.

**Visual-Inertial Mode**

The visual-inertial mode of ORB-SLAM3 is based on the IMU preintegration of [Forster et al. (2017)](https://doi.org/10.1109/TRO.2016.2623335). The key is MAP (Maximum-a-Posteriori) estimation: visual residuals and IMU residuals are optimized jointly in a single cost function:

$$\mathcal{C} = \sum_{i,j} \rho\left(\left\|\mathbf{e}_{ij}^{\text{vis}}\right\|^2_{\Sigma_{ij}}\right) + \sum_k \left\|\mathbf{e}_k^{\text{IMU}}\right\|^2_{\Sigma_k^{\text{IMU}}} + \left\|\mathbf{e}^{\text{prior}}\right\|^2_{\Sigma^{\text{prior}}}$$

The IMU residual $\mathbf{e}_k^{\text{IMU}}$ is defined as the difference between the preintegrated IMU measurement between keyframes $k$ and $k+1$ and the state estimate. The state vector for each keyframe includes:

$$\mathbf{x}_k = [{}^W\mathbf{R}_k, {}^W\mathbf{p}_k, {}^W\mathbf{v}_k, \mathbf{b}_k^g, \mathbf{b}_k^a]$$

Using the preintegrated measurements derived in the previous chapter (Ch.4):

$$\Delta\mathbf{R}_{k,k+1}, \quad \Delta\mathbf{v}_{k,k+1}, \quad \Delta\mathbf{p}_{k,k+1}$$

the IMU residual is constructed as:

$$\mathbf{e}^{\text{IMU}}_{\Delta R} = \text{Log}\left(\Delta\hat{\mathbf{R}}_{k,k+1}^T \cdot \mathbf{R}_k^T \mathbf{R}_{k+1}\right)$$
$$\mathbf{e}^{\text{IMU}}_{\Delta v} = \mathbf{R}_k^T(\mathbf{v}_{k+1} - \mathbf{v}_k - \mathbf{g}\Delta t) - \Delta\hat{\mathbf{v}}_{k,k+1}$$
$$\mathbf{e}^{\text{IMU}}_{\Delta p} = \mathbf{R}_k^T(\mathbf{p}_{k+1} - \mathbf{p}_k - \mathbf{v}_k\Delta t - \frac{1}{2}\mathbf{g}\Delta t^2) - \Delta\hat{\mathbf{p}}_{k,k+1}$$

**Multi-Map System (Atlas)**

One of the most important contributions of ORB-SLAM3 is the Atlas structure. When tracking fails in regions with insufficient visual information (fast rotation, occlusion, etc.), existing systems reinitialize and lose the connection to the previous map. The Atlas of ORB-SLAM3:

1. Creates a new map (sub-map) when tracking fails.
2. Maintains each sub-map independently.
3. When place recognition (DBoW2) detects a previously visited sub-map, automatically merges the two maps.
4. Performs Sim(3) registration for monocular or SE(3) registration for stereo/VI upon merging.

Thanks to this structure, ORB-SLAM3 can recover gracefully from tracking failure and also supports multi-session SLAM that reuses maps from past sessions.

**ORB-SLAM3 Performance**: the stereo-inertial configuration achieves 3.6 cm accuracy on the EuRoC MAV dataset and 9 mm on TUM-VI. This showcases the strengths of the feature-based approach — precise geometric constraints and stable loop closure.

```python
# ORB-SLAM3 Tracking Thread pseudocode
def track(frame, map, last_keyframe):
    # 1. Extract ORB features (8-level image pyramid)
    keypoints, descriptors = extract_orb(frame, n_features=1000, n_levels=8, scale=1.2)
    
    # 2. Initial pose prediction
    if imu_available:
        T_predict = imu_preintegrate(last_frame.T, imu_measurements)
    else:
        T_predict = constant_velocity_model(last_frame.T, last_frame.velocity)
    
    # 3. Project map points onto the current frame → match
    projected_points = project_map_points(map.local_points, T_predict, frame.camera)
    matches = match_by_projection(keypoints, descriptors, projected_points, radius=15)
    
    # 4. Motion-only BA (optimize pose only, fix map points)
    T_refined = motion_only_BA(
        pose_init=T_predict,
        observations=[(kp, mp) for kp, mp in matches],
        robust_kernel='huber',
        n_iterations=10
    )
    
    # 5. Keyframe decision
    tracked_ratio = len(matches) / len(last_keyframe.observations)
    if tracked_ratio < 0.9 and len(matches) > 50:
        insert_keyframe(frame, T_refined, matches)
    
    return T_refined
```

---

## 6.2 Direct Visual Odometry

Direct methods use pixel intensity itself as the observation, without extracting feature points. The basic idea is: a 3D point $\mathbf{P}$ should have the same intensity in two frames (brightness constancy).

### 6.2.1 Photometric Error

The core residual of direct VO is the photometric error. For camera poses $\mathbf{T}_i, \mathbf{T}_j$ and a 3D point $\mathbf{P}$ (or a pixel $\mathbf{u}$ and inverse depth $d^{-1}$ in the host frame):

$$e_{\text{photo}} = I_j\left(\pi(\mathbf{T}_j \mathbf{T}_i^{-1} \pi^{-1}(\mathbf{u}_i, d_i^{-1}))\right) - I_i(\mathbf{u}_i)$$

where:
- $\pi^{-1}(\mathbf{u}, d^{-1})$: reconstructing the 3D point from a 2D point and inverse depth (unprojection)
- $\pi(\cdot)$: 3D → 2D projection
- $I_i, I_j$: intensity images of frames $i, j$

Pose and depth are estimated by minimizing this residual. The key differences from the reprojection error of feature-based methods are:

| | Reprojection error | Photometric error |
|---|---|---|
| Observation | Feature coordinates (2D) | Pixel intensity (1D or patch) |
| Data association | Explicit (matching required) | Implicit (computed via warping) |
| Gradient | Geometric | Image gradient |
| Texture requirement | Corners/edges needed | Only gradient needed |
| Illumination change | Invariant (geometric) | Sensitive (correction needed) |

The advantage of the photometric error is that explicit feature matching is unnecessary. Every region with a gradient across the entire image can be exploited, so the method can operate even in environments that lack texture but have weak gradients (e.g., the subtle texture of a white wall).

There are two limitations:
1. **Brightness constancy violation**: illumination changes, auto exposure, and lens vignetting cause the intensity of the same 3D point to vary across frames. Without correction, accuracy drops sharply.
2. **Narrow basin of convergence**: since optimization is based on image gradients, a poor initial pose estimate leads to local minima. Typically an initial alignment within 1–2 pixels is required.

### 6.2.2 DSO (Direct Sparse Odometry) Architecture Deep Dive

[DSO (Engel et al., 2018)](https://doi.org/10.1109/TPAMI.2017.2658577) is a landmark VO system that combines a direct method with a sparse representation. Prior to DSO there was an implicit equation of "direct = dense" (LSD-SLAM) and "sparse = indirect" (ORB-SLAM), but DSO recombined these two axes in a new way.

**Core Design Principles of DSO**

1. **Direct**: uses pixel intensity directly, without feature points.
2. **Sparse**: instead of using the entire image, samples points evenly in regions with gradient.
3. **Joint Optimization**: simultaneously optimizes poses, inverse depths, and camera intrinsic parameters (affine brightness parameters).

**Full Photometric Calibration**

One of the most important contributions of DSO is its systematic treatment of photometric calibration. In a real camera, the observed intensity $I'$ relates to the true scene irradiance $B$ as:

$$I'(\mathbf{u}) = G(t \cdot V(\mathbf{u}) \cdot B(\mathbf{u}))$$

where:
- $G(\cdot)$: the camera's nonlinear response function
- $t$: exposure time
- $V(\mathbf{u})$: lens vignetting — intensity decreases as we move away from the image center

Correcting all three factors yields a photometrically corrected image:

$$I(\mathbf{u}) = t^{-1} \cdot G^{-1}(I'(\mathbf{u})) / V(\mathbf{u})$$

Without this correction, the same 3D point has different intensities at the image center versus the edge, or in frames with different exposures, so the photometric error becomes inaccurate.

**Affine Brightness Transfer Function**

For cases in which complete photometric calibration is infeasible, DSO further compensates for inter-frame brightness changes with an affine model:

$$e_{\mathbf{p}j} = \sum_{\mathbf{p} \in \mathcal{N}_\mathbf{p}} w_\mathbf{p} \left\| (I_j[\mathbf{p}'] - b_j) - \frac{t_j e^{a_j}}{t_i e^{a_i}}(I_i[\mathbf{p}] - b_i) \right\|_\gamma$$

Here $a_i, b_i, a_j, b_j$ are per-frame affine brightness parameters optimized jointly. $\|\cdot\|_\gamma$ is the Huber norm.

**Point Selection Strategy**

DSO divides the image into a grid and selects the point with the largest gradient magnitude in each cell. The key is "uniform distribution" — it prevents features from concentrating in a single region. About 2000 points are selected, and the gradient threshold is adaptively adjusted to secure points even in low-texture regions.

**Sliding Window Optimization**

DSO jointly optimizes the most recent 5–7 keyframes and the inverse depths of the points belonging to them within a sliding window. The optimization variables are:

$$\boldsymbol{\theta} = \{\mathbf{T}_1, \ldots, \mathbf{T}_n, d_1^{-1}, \ldots, d_m^{-1}, a_1, b_1, \ldots, a_n, b_n\}$$

that is, camera poses (SE(3)), inverse depths, and affine brightness parameters.

Frames/points that drop out of the window are marginalized via the Schur complement and remain as a prior. This marginalization follows the same Schur-complement-based principle covered in Ch.4.7, differing only in that only visual residuals are involved.

**Limitations and Extensions of DSO**

The original DSO design lacks loop closure. This is not a fundamental limitation of direct methods but a design choice. LDSO (Loop-closing DSO) addresses this by combining DBoW with direct alignment. Similarly, VI-DSO, BASALT, and others are VIO variants that couple DSO with an IMU.

```python
# DSO core flow pseudocode
def dso_track(frame, window, camera):
    # 1. Apply photometric calibration
    frame.I = apply_photometric_correction(frame.raw, camera.G_inv, camera.V, frame.exposure)
    
    # 2. Initial pose estimate via direct image alignment (coarse-to-fine)
    T_init = direct_alignment_pyramid(
        ref_frame=window.latest_keyframe,
        cur_frame=frame,
        initial_guess=constant_velocity_predict(),
        levels=[4, 3, 2, 1]  # image pyramid levels
    )
    
    # 3. Point selection (gradient-based, uniform distribution)
    candidate_points = select_points_gradient_based(frame, n_blocks=32*32, n_per_block=1)
    
    # 4. Inverse depth initialization (epipolar search)
    for p in candidate_points:
        p.inv_depth = epipolar_search(window.keyframes, frame, p.u)
    
    # 5. If keyframe: insert into sliding window and run joint optimization
    if is_keyframe(frame, window):
        window.add(frame, candidate_points)
        # Joint optimization of pose + inverse depth + affine parameters
        gauss_newton_optimize(
            residuals=photometric_residuals(window),
            variables=[T, inv_depth, a, b],
            max_iter=6
        )
        # Marginalize the oldest frame
        if len(window) > 7:
            marginalize_oldest(window)
    
    return T_init
```

### 6.2.3 Semi-Direct: SVO

[SVO (Semi-direct Visual Odometry, Forster et al., 2017)](https://doi.org/10.1109/TRO.2016.2623335) is a hybrid approach that combines the advantages of feature-based and direct methods. The name "semi-direct" comes from the fact that tracking uses a direct method while mapping uses a feature-based method.

**Core Ideas of SVO**

1. **Sparse Model-based Image Alignment**: existing 3D map points are projected onto the current frame, and the frame pose is estimated by minimizing the photometric error over a patch around each projected point. This uses image gradients directly, as DSO does, but only around already-known map points rather than the whole image, so it is very fast.

2. **Feature Alignment**: once the pose is estimated, the projected location of each map point is refined to subpixel accuracy. Patch-based direct alignment is again used here.

3. **Structure & Motion Refinement**: the refined 2D locations are treated as "virtual feature points" and BA (reprojection error minimization) jointly optimizes the pose and 3D structure.

Thanks to this three-stage decomposition SVO is extremely fast — it reaches 200–400 Hz even on high-resolution images, making it suitable for agile robots such as high-speed drones. On the other hand, it lacks loop closure and is vulnerable to pure rotation (rotation in place).

---

## 6.3 Tightly-Coupled Visual-Inertial Odometry

VO alone has two fundamental limitations: (1) scale ambiguity with a monocular camera, and (2) tracking failure under fast motion or texture scarcity. Combining an IMU can resolve both problems simultaneously. Tightly-coupled VIO processes the raw measurements of camera and IMU within a single estimation framework.

### 6.3.1 VINS-Mono Architecture in Detail

[VINS-Mono (Qin et al., 2018)](https://doi.org/10.1109/TRO.2018.2853729) is a complete VIO system that achieves robust 6-DoF state estimation using only a monocular camera and a low-cost IMU. It integrates the entire pipeline — initialization, odometry, loop closure, and map reuse — in a single system.

**System Architecture**

VINS-Mono consists of three main modules:

1. **Frontend (Measurement Preprocessing)**: KLT-based feature tracking + IMU preintegration
2. **Backend (Estimator)**: nonlinear-optimization-based tightly-coupled VIO
3. **Loop Closure (Relocalization)**: DBoW2-based place recognition + relocalization

**Robust Initialization**

The greatest challenge for monocular VIO is initialization. A monocular camera alone cannot observe metric scale, and IMU biases are also unknown. VINS-Mono's initialization solves this with a loosely-coupled approach:

**Step 1: Vision-only SfM**

For the first few frames, Structure from Motion is run using vision alone. The Essential Matrix is estimated with the 5-point algorithm and triangulation recovers 3D points and camera poses. Scale at this stage is arbitrary.

**Step 2: Visual-Inertial Alignment**

The SfM result is aligned with IMU preintegration. During this process, the following are estimated:

(a) The gyro bias $\mathbf{b}_g$: compare the SfM rotation $\mathbf{R}_{c_k c_{k+1}}^{\text{sfm}}$ between consecutive keyframe pairs with the IMU preintegrated rotation $\Delta\hat{\mathbf{R}}_{k,k+1}$:

$$\min_{\mathbf{b}_g} \sum_k \left\| \text{Log}\left(\Delta\hat{\mathbf{R}}_{k,k+1}(\mathbf{b}_g)^T \cdot \mathbf{R}_{c_k}^{c_{k+1}} \right) \right\|^2$$

Using the first-order approximation to bias change (the preintegration Jacobian from Ch.4.6), this problem becomes linear.

(b) The gravity direction, velocities, and metric scale are estimated jointly. This reduces to the following linear system. For each keyframe pair $(k, k+1)$:

$$s\mathbf{R}_{c_0}^w \mathbf{p}_{c_{k+1}}^{c_0} - s\mathbf{R}_{c_0}^w \mathbf{p}_{c_k}^{c_0} - \mathbf{v}_k^w \Delta t_k + \frac{1}{2}\mathbf{g}^w\Delta t_k^2 = \mathbf{R}_k^w \Delta\hat{\mathbf{p}}_{k,k+1}$$

The unknowns are $s$ (scale), $\mathbf{g}^w$ (gravity vector), and $\{\mathbf{v}_k^w\}$ (velocities). The constraint $\|\mathbf{g}\| = 9.81$ on the magnitude of gravity is added to improve accuracy.

Once this loosely-coupled initialization converges, the system transitions to tightly-coupled optimization.

**Tightly-Coupled Nonlinear Optimization**

The VINS-Mono backend jointly optimizes three kinds of residuals within the sliding window:

$$\min_{\mathcal{X}} \left\{ \left\|\mathbf{r}_p - \mathbf{H}_p \mathcal{X}\right\|^2 + \sum_{k \in \mathcal{B}} \left\|\mathbf{r}_{\mathcal{B}}(\hat{\mathbf{z}}_{b_k b_{k+1}}, \mathcal{X})\right\|^2_{\mathbf{P}_{b_k b_{k+1}}^{-1}} + \sum_{(l,j) \in \mathcal{C}} \left\|\mathbf{r}_{\mathcal{C}}(\hat{\mathbf{z}}_{l}^{c_j}, \mathcal{X})\right\|^2_{(\mathbf{P}_{l}^{c_j})^{-1}} \right\}$$

where:
- $\mathbf{r}_p$: marginalization prior residual
- $\mathbf{r}_{\mathcal{B}}$: IMU preintegration residual
- $\mathbf{r}_{\mathcal{C}}$: visual reprojection error residual
- $\mathcal{X}$: state variables (keyframe poses, velocities, IMU biases in the sliding window, and feature inverse depths)

**Detailed Definition of the Visual Residual**:

For a feature $l$ first observed in keyframe $i$ and parameterized by inverse depth $\lambda_l$, the reprojection error in keyframe $j$ is:

$$\mathbf{r}_{\mathcal{C}} = \begin{bmatrix} \bar{u}_l^{c_j} - u_l^{c_j} / z_l^{c_j} \\ \bar{v}_l^{c_j} - v_l^{c_j} / z_l^{c_j} \end{bmatrix}$$

where $\begin{bmatrix} u_l^{c_j} & v_l^{c_j} & z_l^{c_j} \end{bmatrix}^T = \mathbf{R}_{b_j}^{c} (\mathbf{R}_{w}^{b_j}(\mathbf{R}_{b_i}^{w}(\mathbf{R}_{c}^{b_i} \frac{1}{\lambda_l}\begin{bmatrix} \bar{u}_l^{c_i} \\ \bar{v}_l^{c_i} \\ 1 \end{bmatrix} + \mathbf{p}_c^b) + \mathbf{p}_{b_i}^w - \mathbf{p}_{b_j}^w) - \mathbf{p}_c^b)$

**Sliding Window Management and Marginalization**

VINS-Mono fixes the window size (typically 10 keyframes) and applies one of two marginalization strategies depending on whether the newest frame is a keyframe:

1. **If the newest frame is a keyframe**: marginalize the oldest keyframe. All measurements connected to that frame are converted to priors via the Schur complement.

2. **If the newest frame is not a keyframe**: marginalize the previous frame (second-newest). In this case only the visual measurements are discarded, since the IMU measurements are included in the preintegration between adjacent keyframes and thus information is preserved.

The key point of both strategies is **information preservation**. Marginalization removes a variable while retaining the information it contributed as a prior. The mathematical mechanism of the Schur complement is treated in detail in Ch.4.7.

**4-DoF Pose Graph Optimization**

Once a loop closure is detected, VINS-Mono optimizes the pose graph in 4-DoF (yaw + 3D translation) rather than 6-DoF. Why 4-DoF? Because the IMU accelerometer observes the direction of gravity, roll and pitch are already sufficiently observable via the IMU. Drift accumulates only in yaw and position. Therefore, on loop closure it is physically sensible to leave roll/pitch untouched and correct only yaw and position.

```python
# VINS-Mono sliding window optimization pseudocode
class VINSEstimator:
    def __init__(self, window_size=10):
        self.window_size = window_size
        self.states = []  # [pose, velocity, bias_g, bias_a] per keyframe
        self.features = {}  # feature_id -> inverse_depth
        self.prior = None  # marginalization prior
    
    def optimize(self):
        # Construct the cost function
        cost = 0.0
        
        # 1. Marginalization prior
        if self.prior is not None:
            cost += self.prior.evaluate(self.states)
        
        # 2. IMU preintegration residual
        for k in range(len(self.states) - 1):
            preint = self.imu_preintegrations[k]
            r_imu = compute_imu_residual(
                self.states[k], self.states[k+1], preint
            )
            cost += r_imu.T @ preint.info_matrix @ r_imu
        
        # 3. Visual reprojection error
        for feat_id, inv_depth in self.features.items():
            for obs in self.observations[feat_id]:
                r_vis = compute_reprojection_error(
                    self.states[obs.host_frame], self.states[obs.target_frame],
                    inv_depth, obs.uv, self.T_cam_imu
                )
                cost += huber(r_vis.T @ obs.info_matrix @ r_vis)
        
        # Gauss-Newton / Levenberg-Marquardt optimization
        solve_nonlinear_least_squares(cost, max_iterations=8)
    
    def marginalize(self, is_keyframe):
        if is_keyframe:
            # Marginalize the oldest frame
            self.prior = schur_complement_marginalize(
                self.states[0], connected_factors(self.states[0])
            )
            self.states.pop(0)
        else:
            # Marginalize the previous frame (visual only; IMU preserved)
            self.prior = schur_complement_marginalize(
                self.states[-2], visual_factors(self.states[-2])
            )
            self.states.pop(-2)
```

### 6.3.2 OKVIS

[OKVIS (Open Keyframe-based Visual-Inertial SLAM, Leutenegger et al., 2015)](https://doi.org/10.1177/0278364914554813) is a tightly-coupled VIO that preceded VINS-Mono and presented an early form of keyframe-based sliding window optimization.

**Key Features of OKVIS**:

1. **Harris corner + BRISK descriptor**: uses Harris corners and BRISK descriptors instead of ORB.
2. **Keyframe-based marginalization**: performs marginalization within a sliding window, similarly to VINS-Mono.
3. **Speed error term**: rather than IMU preintegration, it directly integrates IMU over a short time interval and uses it as a velocity constraint. OKVIS2 later switched to preintegration.
4. **Ceres Solver-based**: uses Ceres Solver for optimization.

OKVIS has simpler initialization than VINS-Mono (it assumes a stereo camera by default), but VINS-Mono is superior for robust initialization in the monocular case.

### 6.3.3 MSCKF: Multi-State Constraint Kalman Filter

[MSCKF (Mourikis & Roumeliotis, 2007)](https://doi.org/10.1109/ROBOT.2007.364024) is the representative algorithm for filter-based VIO. Its approach is fundamentally different from optimization-based systems (VINS-Mono, ORB-SLAM3), and it remains actively used in certain applications today.

**Core Idea of MSCKF: Do Not Put Landmarks in the State**

EKF-SLAM includes landmarks (3D points) in the state vector. With $N$ landmarks, the state vector has size $3N + 15$ and the covariance matrix has size $(3N+15)^2$, requiring $O(N^2)$ space and $O(N^3)$ time in the number of landmarks. This is fatal for real-time processing.

The key insight of MSCKF is: **it is possible to exclude landmarks from the state vector while still preserving the geometric constraint information they provide**.

**State Vector Structure**

The MSCKF state vector comprises the IMU error state and $N$ camera poses within a sliding window:

$$\tilde{\mathbf{x}} = [\tilde{\mathbf{x}}_{IMU}^T, \tilde{\mathbf{x}}_{C_1}^T, \ldots, \tilde{\mathbf{x}}_{C_N}^T]^T$$

where the IMU error state is:

$$\tilde{\mathbf{x}}_{IMU} = [\delta\boldsymbol{\theta}^T, \tilde{\mathbf{b}}_g^T, {}^G\tilde{\mathbf{v}}_I^T, \tilde{\mathbf{b}}_a^T, {}^G\tilde{\mathbf{p}}_I^T]^T \in \mathbb{R}^{15}$$

and each camera pose error state is:

$$\tilde{\mathbf{x}}_{C_k} = [\delta\boldsymbol{\theta}_{C_k}^T, {}^G\tilde{\mathbf{p}}_{C_k}^T]^T \in \mathbb{R}^6$$

The total size of the state vector is $15 + 6N$, independent of the number of landmarks.

**Null-Space Projection: The Core Mathematics**

Suppose a single static feature $\mathbf{p}_f$ is observed across $M$ camera poses. Linearizing the observation equation:

$$\mathbf{r} = \mathbf{H}_X \tilde{\mathbf{x}} + \mathbf{H}_f \tilde{\mathbf{p}}_f + \mathbf{n}$$

where:
- $\mathbf{r} \in \mathbb{R}^{2M}$: residual vector (observation − prediction)
- $\mathbf{H}_X \in \mathbb{R}^{2M \times (15+6N)}$: Jacobian with respect to the state
- $\mathbf{H}_f \in \mathbb{R}^{2M \times 3}$: Jacobian with respect to the feature position
- $\tilde{\mathbf{p}}_f \in \mathbb{R}^3$: feature position error

QR-decompose $\mathbf{H}_f$:

$$\mathbf{H}_f = \begin{bmatrix} \mathbf{Q}_1 & \mathbf{Q}_2 \end{bmatrix} \begin{bmatrix} \mathbf{R}_1 \\ \mathbf{0} \end{bmatrix}$$

$\mathbf{Q}_2$ spans the left null space of $\mathbf{H}_f$ ($\mathbf{Q}_2^T \mathbf{H}_f = \mathbf{0}$). Left-multiplying both sides by $\mathbf{Q}_2^T$:

$$\mathbf{r}_o = \mathbf{Q}_2^T \mathbf{r} = \mathbf{Q}_2^T \mathbf{H}_X \tilde{\mathbf{x}} + \mathbf{Q}_2^T \mathbf{n} = \mathbf{H}_o \tilde{\mathbf{x}} + \mathbf{n}_o$$

The feature position $\tilde{\mathbf{p}}_f$ has been fully eliminated. The EKF update can be performed using only $\mathbf{r}_o$ and $\mathbf{H}_o$. This is the "multi-state constraint" of MSCKF — exploiting the geometric constraint that a single feature imposes across multiple camera poses directly, while excluding the feature itself from the state.

**Computational Complexity**: the state vector size is $15 + 6N$ (in the number of camera poses $N$), independent of the number of landmarks $M$. EKF-SLAM includes $M$ landmarks in the state so the state size is $O(M)$ and the covariance update requires $O(M^2)$. MSCKF excludes landmarks from the state and therefore depends only on the number of cameras $N \ll M$ — this is its core advantage.

**MSCKF Update Procedure**

1. **IMU propagation**: when a new IMU measurement arrives, propagate the state and update the covariance:
   $$\hat{\mathbf{x}}_{k+1|k} = f(\hat{\mathbf{x}}_{k|k}, \mathbf{u}_k)$$
   $$\mathbf{P}_{k+1|k} = \boldsymbol{\Phi}_k \mathbf{P}_{k|k} \boldsymbol{\Phi}_k^T + \mathbf{G}_k \mathbf{Q} \mathbf{G}_k^T$$

2. **State augmentation**: when a new image arrives, copy the current IMU pose to add a camera pose to the state. The covariance matrix is also expanded.

3. **MSCKF update**: for a feature whose tracking has ended (no longer observed):
   - Estimate $\hat{\mathbf{p}}_f$ by triangulation.
   - Compute $\mathbf{r}_o, \mathbf{H}_o$ via null-space projection.
   - Perform the standard EKF update:
     $$\mathbf{K} = \mathbf{P} \mathbf{H}_o^T (\mathbf{H}_o \mathbf{P} \mathbf{H}_o^T + \sigma^2 \mathbf{I})^{-1}$$
     $$\hat{\mathbf{x}} \leftarrow \hat{\mathbf{x}} + \mathbf{K} \mathbf{r}_o$$
     $$\mathbf{P} \leftarrow (\mathbf{I} - \mathbf{K}\mathbf{H}_o)\mathbf{P}$$

4. **State pruning**: remove old camera poses from the sliding window and shrink the covariance accordingly.

```cpp
// MSCKF core update pseudocode (C++)
void MSCKF::msckf_update(const Feature& feature) {
    // 1. Estimate the 3D position of the feature via triangulation
    Vector3d p_f = triangulate(feature.observations, cam_states);
    
    // 2. Compute observation Jacobians
    int N_obs = feature.observations.size();
    MatrixXd H_X(2*N_obs, state_dim);  // Jacobian with respect to the state
    MatrixXd H_f(2*N_obs, 3);          // Jacobian with respect to the feature
    VectorXd r(2*N_obs);               // residual
    
    for (int i = 0; i < N_obs; i++) {
        auto& obs = feature.observations[i];
        auto& cam = cam_states[obs.cam_id];
        
        Vector3d p_c = cam.R_w2c * (p_f - cam.p_w);
        double X = p_c(0), Y = p_c(1), Z = p_c(2);
        
        // Projection Jacobian (pinhole)
        Matrix<double, 2, 3> J_proj;
        J_proj << 1.0/Z, 0, -X/(Z*Z),
                  0, 1.0/Z, -Y/(Z*Z);
        
        // Compute H_f
        H_f.block<2,3>(2*i, 0) = J_proj * cam.R_w2c;
        
        // Compute H_X (blocks for camera poses)
        // ... (rotation, translation Jacobians)
        
        // Residual
        r.segment<2>(2*i) = obs.uv - Vector2d(X/Z, Y/Z);
    }
    
    // 3. Null-space projection via QR decomposition
    // H_f = Q * [R1; 0] → Q2^T * H_f = 0
    HouseholderQR<MatrixXd> qr(H_f);
    MatrixXd Q = qr.householderQ();
    MatrixXd Q2 = Q.rightCols(2*N_obs - 3);
    
    MatrixXd H_o = Q2.transpose() * H_X;
    VectorXd r_o = Q2.transpose() * r;
    
    // 4. Standard EKF update
    MatrixXd S = H_o * P * H_o.transpose() + sigma2 * MatrixXd::Identity(r_o.size(), r_o.size());
    MatrixXd K = P * H_o.transpose() * S.inverse();
    
    VectorXd dx = K * r_o;
    // Update the state (on-manifold)
    apply_correction(dx);
    // Update the covariance
    P = (MatrixXd::Identity(state_dim, state_dim) - K * H_o) * P;
}
```

### 6.3.4 OpenVINS

[OpenVINS (Geneva et al., 2020)](https://doi.org/10.1109/ICRA40945.2020.9196524) is the most complete open-source implementation of MSCKF-based VIO. Beyond being a straightforward implementation, it aims to be a research platform where various VIO algorithm variants can be modularly compared and experimented with.

**Key Features of OpenVINS**:

1. **On-Manifold Sliding Window EKF**: a sliding window Kalman filter based on MSCKF. Rotations are handled on the SO(3) manifold.

2. **Online calibration**: camera intrinsics, camera-IMU extrinsics, and the temporal offset are estimated automatically at runtime. Temporal offset estimation is particularly important because imperfect synchronization of camera and IMU timestamps severely degrades performance.

3. **SLAM landmark support**: pure MSCKF does not include features in the state, but OpenVINS can optionally include a subset of landmarks in the state as SLAM features. SLAM features are long-tracked points, parameterized as anchored inverse depth.

4. **First-Estimates Jacobian (FEJ)**: a technique for addressing the consistency problem of the EKF. The standard EKF recomputes Jacobians at the latest state estimate at every update, which violates observability properties and causes the covariance to shrink excessively. FEJ computes Jacobians only at the first estimate, preserving correct observability.

5. **Simulator**: a simulator for testing VIO algorithms is included. It supports a variety of trajectories, environment configurations, and IMU noise models, enabling quantitative evaluation via comparison with ground truth.

### 6.3.5 Basalt

[Basalt (Usenko et al., 2020)](https://doi.org/10.1109/LRA.2019.2961227) is a tightly-coupled VIO similar to VINS-Mono but differs in several design choices.

**Key Features of Basalt**:

1. **Visual-only Frontend**: instead of KLT, it performs patch-based direct alignment (similar to SVO) for subpixel-accurate feature tracking.

2. **Non-linear Factor Recovery (NFR)**: an alternative to marginalization. Marginalization leaves a prior that depends on the linearization point, and information distortion occurs if that linearization point later changes significantly. Basalt's NFR approximates the marginalized information as a nonlinear factor, enabling relinearization.

3. **Efficient Implementation**: Basalt exploits the structure of the factor graph for an efficient implementation, achieving higher processing speed than VINS-Mono.

4. **Stereo/Multi-camera support**: it naturally integrates visual information from multiple cameras.

---

## 6.4 VIO Design Choices

When designing a VIO system, several core design choices arise. This section analyzes the pros and cons of each option and the scenarios to which it applies.

### 6.4.1 Filter vs Optimization

This is one of the oldest debates in the VIO field.

**Filter-based (MSCKF, OpenVINS)**

- Maintains only the current state and updates it sequentially as new measurements arrive
- Past states are "absorbed" into the current state distribution (mean + covariance)
- Computational complexity: $O(N^2)$ per update ($N$ is the state dimension)
- Advantages: constant per-update cost, simple implementation
- Disadvantages: accumulated linearization error (once linearized, it cannot be corrected), consistency problems

**Optimization-based (VINS-Mono, ORB-SLAM3, Basalt)**

- Maintains multiple states within a sliding window and optimizes them jointly with all measurements
- Iterative re-linearization is possible
- Computational complexity: $O(N^3)$ per iteration ($N$ is the window size), but made efficient with the Schur complement
- Advantages: more accurate (iterative relinearization reduces linearization error), easy integration of diverse measurements
- Disadvantages: higher computational cost, sensitivity to window size

**Empirical Comparison**: under equal conditions, optimization-based methods are generally more accurate than filter-based ones. However, when FEJ and proper observability analysis are applied, MSCKF closes much of the gap. In severely resource-constrained environments (ultra-low-power MCUs, etc.), filter-based approaches remain attractive.

### 6.4.2 Keyframe Selection Strategies

Keyframe selection has a large impact on VIO performance. Inserting keyframes too often increases the computational burden; too rarely, and information is lost.

Common keyframe selection criteria:

1. **Time-based**: a minimum time has elapsed since the last keyframe.
2. **Parallax-based**: the mean feature parallax between the current frame and the last keyframe exceeds a threshold. VINS-Mono uses this criterion.
3. **Tracking quality-based**: insert a keyframe when the number of tracked features falls below a threshold. ORB-SLAM3 uses this criterion.
4. **Information gain-based**: decide based on the estimated information gain (amount of information) that a new keyframe would provide. Theoretically the most principled but computationally expensive.

Keyframe selection is closely tied to marginalization. The two-way marginalization strategy of VINS-Mono (see Section 6.3.1) switches the direction of marginalization depending on whether the frame is a keyframe, illustrating this connection clearly.

### 6.4.3 Feature Parameterization

How 3D points are parameterized is another important design choice.

**XYZ (Euclidean 3D coordinates)**

The most intuitive option. $\mathbf{P} = [X, Y, Z]^T \in \mathbb{R}^3$. However, it is unstable for far points — small angular changes cause large variations in $Z$.

**Inverse Depth**

A point is expressed as "observation direction in the host frame + inverse depth":

$$\boldsymbol{\lambda} = [\theta, \phi, \rho]^T$$

Here $\theta, \phi$ are the azimuth/elevation in the host frame and $\rho = 1/d$ is the inverse depth. Advantages:

1. **Handling far points**: as $d \to \infty$, $\rho \to 0$ is numerically stable. In XYZ, $Z \to \infty$ leads to instability.
2. **Improved linearity**: when depth is uncertain in monocular initialization, the uncertainty distribution of inverse depth is closer to Gaussian.

VINS-Mono and OpenVINS use the inverse depth parameterization.

**Anchored Inverse Depth**

The inverse depth is anchored to a specific "anchor" keyframe:

$$\mathbf{P} = \mathbf{T}_{\text{anchor}} \cdot \frac{1}{\rho} [\bar{u}, \bar{v}, 1]^T$$

where $(\bar{u}, \bar{v})$ are the normalized coordinates in the anchor frame and $\rho$ is the inverse depth. The advantage of this parameterization is that even when the anchor frame's pose changes, the inverse depth itself does not, partially reducing linearization error. It is used for SLAM features in ORB-SLAM3 and OpenVINS.

---

## 6.5 Learning-Based VO/VIO

Traditional VO/VIO relies on a "human-designed pipeline": feature detection → matching → RANSAC → BA. Learning-based approaches aim to replace part or all of this pipeline with neural networks.

### 6.5.1 Supervised: the DeepVO Family

Early learning-based VO ([DeepVO, Wang et al., 2017](https://doi.org/10.1109/ICRA.2017.7989236)) trained an end-to-end network that takes consecutive image pairs as input and directly predicts the relative pose. A CNN extracts visual features and an LSTM models temporal dependencies.

The limitations are clear:
- Overfitting to the environment of the training data (poor generalization)
- Not exploiting geometric constraints (e.g., epipolar geometry), so accuracy falls short of traditional methods
- Severe scale drift

### 6.5.2 Self-supervised: Limitations and Current Position

Self-supervised VO (SfMLearner, Monodepth2, etc.) learns depth and pose jointly using photometric loss via view synthesis. It has the advantage of training without labels, but struggles with moving objects, texture scarcity, and occlusion.

Current position: self-supervised VO has achieved great success in monocular depth estimation, but as a standalone VO/VIO system it falls well short of traditional methods. In particular it has not resolved the accumulated drift problem.

### 6.5.3 Hybrid: DROID-SLAM

[DROID-SLAM (Teed & Deng, 2021)](https://arxiv.org/abs/2108.10869) is a system that integrates the geometric rigor of traditional BA with the robust matching ability of deep learning in a single differentiable pipeline. It was the first to demonstrate that learning-based SLAM can surpass traditional systems on every metric.

**Architecture**

The core of DROID-SLAM is the combination of two components:

1. **RAFT-Inspired Iterative Update Operator**

A structure inspired by RAFT (Recurrent All-Pairs Field Transforms, Teed & Deng, 2020). For a frame pair $(i, j)$:
- A 4D correlation volume is computed from the two frames' feature maps.
- Features are indexed from the correlation volume using the correspondence field derived from the current pose/depth estimate.
- A 3×3 Convolutional GRU takes correlation features, the current flow, and context features as input and outputs a **flow revision** $\Delta\mathbf{f}_{ij}$ and a **confidence weight** $\mathbf{w}_{ij}$.

$$(\Delta\mathbf{f}_{ij}, \mathbf{w}_{ij}) = \text{ConvGRU}(\text{corr}_{ij}, \mathbf{f}_{ij}^{\text{curr}}, \text{context}_i)$$

This update is applied iteratively to progressively refine the correspondences.

2. **Differentiable Dense Bundle Adjustment (DBA) Layer**

A layer that turns the flow revision output by the GRU into a geometric update. The key idea: define a reprojection error for every pixel and minimize it via Gauss-Newton over camera poses $\mathbf{T}_i \in SE(3)$ and inverse depths $d_i$:

$$\sum_{(i,j)} \sum_{\mathbf{p}} \left\| \mathbf{w}_{ij}^{\mathbf{p}} \circ (\mathbf{p}^{*}_{ij} - \pi(\mathbf{T}_j \circ \mathbf{T}_i^{-1} \circ \pi^{-1}(\mathbf{p}, d_i^{\mathbf{p}}))) \right\|^2$$

where $\mathbf{p}^{*}_{ij} = \mathbf{p} + \mathbf{f}_{ij}^{\text{curr}} + \Delta\mathbf{f}_{ij}$ is the target correspondence and $\mathbf{w}_{ij}^{\mathbf{p}}$ is the confidence weight.

Deriving the Gauss-Newton update gives the normal equations:

$$\begin{bmatrix} \mathbf{B} & \mathbf{E} \\ \mathbf{E}^T & \mathbf{C} \end{bmatrix} \begin{bmatrix} \boldsymbol{\xi} \\ \delta\mathbf{d} \end{bmatrix} = \begin{bmatrix} \mathbf{v} \\ \mathbf{w} \end{bmatrix}$$

where $\boldsymbol{\xi}$ is the pose update (in the SE(3) tangent space), $\delta\mathbf{d}$ is the inverse depth update, $\mathbf{B}$ is the pose block, $\mathbf{C}$ is the depth block (diagonal), and $\mathbf{E}$ is the cross block.

The Schur complement is applied as in traditional BA:

$$(\mathbf{B} - \mathbf{E}\mathbf{C}^{-1}\mathbf{E}^T)\boldsymbol{\xi} = \mathbf{v} - \mathbf{E}\mathbf{C}^{-1}\mathbf{w}$$

Since $\mathbf{C}$ is diagonal, $\mathbf{C}^{-1}$ is trivial. The reduced system depends only on the number of cameras and is therefore efficient.

The core innovation is that the entire Gauss-Newton solver is **differentiable**. Backpropagation trains the GRU's parameters to output "good correspondences."

**Frame Graph and Loop Closure**

DROID-SLAM builds a co-visibility-based frame graph dynamically. When a new frame is added, edges are connected to its neighbors; when co-visibility with past frames is detected, long-range edges are added to perform loop closure. The backend runs a global BA over the entire keyframe history.

**Multi-Modality Support**

Although DROID-SLAM is trained only on monocular video, it can directly use stereo and RGB-D inputs at inference time:
- **Stereo**: treat left/right frames as separate frames while fixing their relative pose at the known baseline.
- **RGB-D**: use depth information as the initial inverse depth and incorporate depth observations as additional constraints.

**Performance**

- TartanAir: 62% error reduction versus the previous best
- EuRoC (monocular): 82% reduction
- ETH-3D: 30 out of 32 sequences succeed (previous best: 19)
- Trained only on synthetic data (TartanAir), it achieves SOTA on all four real datasets

**Why It Matters**

DROID-SLAM is the first practical system that combines geometric rigor (BA) with data-driven robustness (learned correspondence). It operates stably even in environments where traditional methods fail (repetitive patterns, texture scarcity, abrupt illumination change).

It has limitations, however:
- **Real time**: a GPU is required, and it currently runs slower than real time.
- **No IMU**: it is a vision-only system; coupling an IMU remains an open research problem.
- **Memory**: feature maps and correlation volumes for every frame must be retained, so memory usage is high.

Follow-up research is addressing these limitations one by one, and learning-based VO/VIO is rapidly catching up with traditional methods.

### 6.5.4 Recent Trends (2023-2025)

[DPVO (Teed & Deng, 2023)](https://arxiv.org/abs/2208.04726) replaces DROID-SLAM's dense flow with sparse patch-based matching, reducing memory by a factor of three and improving speed threefold while achieving comparable or better accuracy. Combining a patch-wise recurrent update operator with a differentiable BA, it realizes near-real-time learning-based VO.

[MAC-VO (Qu et al., 2024)](https://arxiv.org/abs/2409.09479) introduces learning-based matching uncertainty (metrics-aware covariance) into stereo VO, using the uncertainty to determine keypoint selection and the residual weights in pose graph optimization. It outperforms existing VO/SLAM systems in environments with illumination changes and texture scarcity, and was selected as ICRA 2025 Best Paper.

---

## Chapter 6 Summary

| System | Type | Estimation | Sensors | Key Features |
|--------|------|-----------|------|-----------|
| ORB-SLAM3 | Feature-based | Optimization (BA) | Mono/Stereo/RGBD + IMU | Multi-map Atlas, fisheye support |
| DSO | Direct | Optimization (windowed) | Mono | Photometric calibration, sparse sampling |
| SVO | Semi-direct | Optimization (BA) | Mono/Stereo | 200-400 Hz, suitable for high-speed drones |
| VINS-Mono | Feature-based | Optimization (sliding window) | Mono + IMU | Robust initialization, 4-DoF loop closure |
| MSCKF | Feature-based | EKF (sliding window) | Mono/Stereo + IMU | Excludes landmarks from state, null-space projection |
| OpenVINS | Feature-based | EKF (MSCKF) | Mono/Stereo + IMU | Online calibration, FEJ, research platform |
| Basalt | Semi-direct | Optimization (sliding window) | Stereo + IMU | NFR, efficient implementation |
| DROID-SLAM | Learned | Differentiable BA | Mono/Stereo/RGBD | Differentiable BA, trained on synthetic data |
| DPVO | Learned (sparse) | Differentiable BA | Mono | 3x faster than DROID with 1/3 memory |
| MAC-VO | Learned + Opt. | Pose graph opt. | Stereo | Metrics-aware covariance, ICRA 2025 Best Paper |

The next chapter covers LiDAR-based odometry and LiDAR-Inertial fusion systems, analyzing their complementary relationship with camera-based systems.
