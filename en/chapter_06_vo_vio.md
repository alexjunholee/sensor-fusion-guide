# Chapter 6. VO and VIO

Visual Odometry (VO) and Visual-Inertial Odometry (VIO) bring together the state estimation machinery of Chapter 4 and the visual correspondence pipelines of Chapter 5 (→ Chapters 4, 5). Where Chapter 5 asked how to match pixels and features across views, this chapter asks how to turn those correspondences—and, when available, inertial measurements—into a continuous 6-DoF camera trajectory.

Pure VO estimates a camera's ego-motion from images alone. Adding an IMU turns the system into VIO, which recovers metric scale for a single camera and bridges brief visual dropouts during rapid motion or textureless stretches. The two families share a common heritage, yet they diverge sharply in internal structure and design trade-offs.

Modern VO traces back to [Nistér et al. (2004)](https://doi.org/10.1109/CVPR.2004.1315094), who coined the term "Visual Odometry" and demonstrated real-time ego-motion estimation for both stereo and monocular cameras. Their stereo pipeline triangulated 3D landmarks from left–right pairs and recovered inter-frame rigid-body motion via a 3-point solver, whereas their monocular pipeline estimated the Essential Matrix with a 5-point algorithm. Two decades later, that core progression—feature detection, correspondence matching, RANSAC outlier rejection, and motion estimation—still underpins feature-based VO.

Across this landscape, VO and VIO architectures divide along three primary axes:

1. **Feature-based vs. Direct**: whether the estimator extracts and matches discrete geometric primitives (corners, edges) or minimizes residuals directly over raw pixel intensities.
2. **Filter vs. Optimization**: whether the backend propagates a recursive Kalman-filter state or solves nonlinear least squares over a sliding window of keyframes.
3. **Loosely coupled vs. Tightly coupled**: whether visual and inertial pipelines produce separate estimates that are merged downstream, or combine raw image and IMU residuals inside a single joint estimator.

Different combinations along these three axes yield distinct system architectures, computational footprints, and failure modes.

---

## 6.1 Feature-based Visual Odometry

Feature-based VO converts each incoming image into a sparse set of repeatable geometric landmarks, tracks or matches them across frames, and recovers camera motion from projective geometry. Because this paradigm has matured over decades, its failure modes and numerical behavior are thoroughly understood—which is why production-grade reference systems such as ORB-SLAM3 continue to build on it.

### 6.1.1 Frontend: Detection, Tracking, Outlier Rejection

The visual frontend transforms raw pixel arrays into reliable geometric correspondences across time through three stages: detecting salient points, associating them between frames, and purging false matches.

**Feature Detection**

A good visual feature must exhibit high *repeatability*: the projection of the same physical 3D point should be re-detected reliably even as viewpoint and illumination shift. Geometrically, a point along a straight edge suffers from the aperture problem—motion parallel to the edge cannot be observed—so trackers seek corners where image intensity changes sharply in two orthogonal directions.

The Harris corner detector formalizes this intuition using the local autocorrelation matrix (second-moment matrix) $\mathbf{M}$ summed over an image patch $W$ with weights $w(x,y)$:

$$\mathbf{M} = \sum_{(x,y) \in W} w(x,y) \begin{bmatrix} I_x^2 & I_xI_y \\ I_xI_y & I_y^2 \end{bmatrix}$$

Here $I_x$ and $I_y$ denote the horizontal and vertical image gradients. Geometrically, the eigenvalues $\lambda_1, \lambda_2$ of $\mathbf{M}$ measure intensity variation along the two principal directions of the patch: flat regions have two small eigenvalues, edges have one large and one small eigenvalue, and corners have two large eigenvalues. Rather than computing eigenvalues explicitly, Harris evaluates the corner response:

$$R = \det(\mathbf{M}) - k \cdot \text{tr}(\mathbf{M})^2 = \lambda_1\lambda_2 - k(\lambda_1 + \lambda_2)^2$$

When speed is paramount, FAST (Features from Accelerated Segment Test) replaces gradient convolutions with direct intensity comparisons. A candidate pixel $p$ is classified as a corner if at least $n$ contiguous pixels (typically $n=9$) on a radius-3 Bresenham circle of 16 surrounding pixels are all brighter or all darker than $p$ by a margin. While a quick four-point pre-test on pixels 1, 5, 9, and 13 works as a shortcut for $n=12$, it does not hold for $n=9$; instead, a learned decision tree orders the pixel tests to reject non-corners early. FAST's speed advantage over Harris depends on implementation and hardware, and raw FAST lacks both orientation and scale invariance.

ORB (Oriented FAST and Rotated BRIEF) bridges that gap by pairing FAST detection with an orientation estimate and steering the binary BRIEF descriptor accordingly, producing compact features well suited to real-time SLAM. Patch orientation comes from the intensity centroid:

$$\theta = \text{atan2}(m_{01}, m_{10}), \quad m_{pq} = \sum_{x,y} x^p y^q I(x,y)$$

To achieve scale invariance as the camera approaches or recedes from surfaces, the ORB-SLAM family extracts ORB features across an image pyramid (typically 8 levels with a scale factor of 1.2).

**Feature Tracking**

Once features are detected, two complementary strategies establish correspondences across frames:

1. **Descriptor matching**: Features are detected independently in each frame and paired by descriptor distance (Hamming distance for binary descriptors such as ORB, L2 distance for floating-point descriptors). ORB-SLAM3 follows this route, using a DBoW2 vocabulary tree to narrow candidate matches and keep matching fast.

2. **Optical Flow Tracking**: The Lucas-Kanade (LK) tracker—used by VINS-Mono—follows each corner from the previous frame into the current image without re-extracting descriptors. Assuming that a physical surface point maintains constant brightness over a small time step $\Delta t$,

$$I(x + u, y + v, t + \Delta t) = I(x, y, t)$$

a first-order Taylor expansion over a local window $W$ gives a linear least-squares system for the pixel displacement $(u, v)$:

$$\begin{bmatrix} u \\ v \end{bmatrix} = \left(\sum_W \begin{bmatrix} I_x^2 & I_xI_y \\ I_xI_y & I_y^2 \end{bmatrix}\right)^{-1} \sum_W \begin{bmatrix} -I_xI_t \\ -I_yI_t \end{bmatrix}$$

Evaluating this update coarse-to-fine across an image pyramid enables tracking even across large inter-frame pixel motions.

Each approach carries a clear trade-off: descriptor matching handles wide baselines (large viewpoint shifts) and produces descriptors that can be reused directly for loop closure, at the cost of per-frame detection and matching overhead. LK optical flow is lightweight and achieves subpixel precision, but breaks down under large viewpoint changes or rapid appearance shifts.

**Outlier Rejection**

Even well-tuned matchers produce false correspondences (outliers), and a single gross mismatch can pull a least-squares pose estimate far off course. The standard remedy is RANSAC (Random Sample Consensus), which repeatedly fits a minimal geometric model to tiny random subsets, counts how many remaining matches agree with each hypothesis, and keeps the model with the largest inlier consensus. Depending on what 3D information is available, VO applies RANSAC to one of three geometric formulations:

- **2D-2D**: estimate the Essential Matrix $\mathbf{E}$ with the 5-point algorithm, then decompose $\mathbf{E} = [\mathbf{t}]_\times \mathbf{R}$ to recover relative pose $(R, t)$ up to scale.
- **3D-2D**: solve PnP (Perspective-n-Point) inside RANSAC to recover absolute camera pose from correspondences between already-triangulated 3D map points and current 2D observations.
- **3D-3D**: estimate the rigid-body transform between two 3D point sets via an ICP variant.

In the 2D-2D case, valid correspondences between normalized image coordinates $\mathbf{p}_1, \mathbf{p}_2$ must lie on corresponding epipolar lines and therefore satisfy the epipolar constraint:

$$\mathbf{p}_2^T \mathbf{E} \mathbf{p}_1 = 0$$

Matches that violate this condition beyond a geometric tolerance are discarded as outliers.

When initializing a monocular map, ORB-SLAM3 estimates both a Fundamental Matrix $\mathbf{F}$ and a planar Homography $\mathbf{H}$ in parallel, choosing between them according to their relative model scores. Note that choosing $\mathbf{H}$ on planar scenes is *not* because $\mathbf{H}$ has fewer parameters—in fact, $\mathbf{F}$ has 7 degrees of freedom whereas $\mathbf{H}$ has 8. Rather, when all observed 3D points lie on a single plane (or when the camera undergoes pure rotation), $\mathbf{F}$ becomes geometrically degenerate and is no longer uniquely determined.

### 6.1.2 Backend: PnP, Motion-only BA, Local BA

With clean correspondences from the frontend, the backend recovers camera poses and 3D scene structure in three stages—progressing from a fast minimal pose estimate, to single-frame pose refinement, to joint optimization over a local window of keyframes and landmarks.

**PnP (Perspective-n-Point)**

Given correspondences between triangulated 3D map points $\mathbf{P}_j \in \mathbb{R}^3$ and their 2D image observations $\mathbf{u}_j \in \mathbb{R}^2$, PnP recovers the camera pose $\mathbf{T} = (\mathbf{R}, \mathbf{t}) \in SE(3)$ consistent with perspective projection:

$$\mathbf{u}_j = \pi(\mathbf{R}\mathbf{P}_j + \mathbf{t})$$

Here $\pi: \mathbb{R}^3 \to \mathbb{R}^2$ maps 3D camera-frame coordinates onto the image plane. A minimal solver such as P3P uses three points to generate candidate poses and a fourth to disambiguate them, running inside RANSAC to reject outliers before refining the surviving pose over all inliers.

**Motion-only Bundle Adjustment**

Starting from the PnP estimate, motion-only Bundle Adjustment holds the 3D map points fixed and refines only the current 6-DoF camera pose by minimizing weighted reprojection error:

$$\mathbf{T}^* = \underset{\mathbf{T}}{\arg\min} \sum_{j} \rho\left(\left\|\mathbf{u}_j - \pi(\mathbf{T} \cdot \mathbf{P}_j)\right\|^2_{\Sigma_j}\right)$$

where $\Sigma_j$ is the observation covariance and $\rho(\cdot)$ is a robust loss such as the Huber kernel, which tempers the influence of lingering outliers by switching from quadratic to linear growth beyond a threshold $\delta$:

$$\rho(s) = \begin{cases} s & \text{if } s \leq \delta^2 \\ 2\delta\sqrt{s} - \delta^2 & \text{if } s > \delta^2 \end{cases}$$

Because $\mathbf{T}$ evolves on the $SE(3)$ manifold rather than Euclidean space, each Gauss-Newton or Levenberg-Marquardt step updates the pose via a 6-DoF increment $\boldsymbol{\xi} \in \mathbb{R}^6$ in the Lie algebra $\mathfrak{se}(3)$:

$$\mathbf{T} \leftarrow \exp(\boldsymbol{\xi}^{\wedge}) \cdot \mathbf{T}$$

**Local Bundle Adjustment**

While motion-only BA keeps the frame rate high during tracking, any drift in map point positions would persist unless landmarks and past poses are refined together. Local BA jointly optimizes the most recent $N$ keyframes and every 3D point they observe:

$$\{\mathbf{T}_i^*, \mathbf{P}_j^*\} = \underset{\{\mathbf{T}_i, \mathbf{P}_j\}}{\arg\min} \sum_{i,j} \rho\left(\left\|\mathbf{u}_{ij} - \pi(\mathbf{T}_i \cdot \mathbf{P}_j)\right\|^2_{\Sigma_{ij}}\right)$$

The term "bundle" refers to the bundles of light rays converging from each 3D landmark onto the observing cameras. Because observations link cameras only to points (never directly camera-to-camera or point-to-point in pure visual BA), the resulting normal equations exhibit a distinctive arrow-head sparsity:

$$\begin{bmatrix} \mathbf{H}_{cc} & \mathbf{H}_{cp} \\ \mathbf{H}_{pc} & \mathbf{H}_{pp} \end{bmatrix} \begin{bmatrix} \delta\boldsymbol{\xi} \\ \delta\mathbf{p} \end{bmatrix} = \begin{bmatrix} \mathbf{b}_c \\ \mathbf{b}_p \end{bmatrix}$$

Crucially, $\mathbf{H}_{pp}$ is block-diagonal with one $3\times 3$ block per landmark because 3D points do not interact directly with one another. Inverting $\mathbf{H}_{pp}$ is therefore trivial, and eliminating $\delta\mathbf{p}$ via the Schur complement collapses the system to:

$$(\mathbf{H}_{cc} - \mathbf{H}_{cp}\mathbf{H}_{pp}^{-1}\mathbf{H}_{pc})\delta\boldsymbol{\xi} = \mathbf{b}_c - \mathbf{H}_{cp}\mathbf{H}_{pp}^{-1}\mathbf{b}_p$$

This reduced system scales only with the number of keyframes—completely independent of the number of 3D points—allowing local BA to optimize tens of thousands of landmarks near real time.

### 6.1.3 ORB-SLAM3 Architecture Deep Dive

[ORB-SLAM3 (Campos et al., 2021)](https://doi.org/10.1109/TRO.2021.3075644) stands as a comprehensive reference architecture for feature-based visual and visual-inertial SLAM. Within a unified codebase, it handles monocular, stereo, and RGB-D cameras—with or without an IMU—under both pinhole and Kannala-Brandt fisheye projections.

**Overall Architecture**

To separate frame-rate pose tracking from heavier map refinement and global loop closure, ORB-SLAM3 runs three concurrent threads:

1. **Tracking Thread**: extracts ORB features from every incoming frame, matches them against projected local map points, refines the pose with motion-only BA, and decides whether the frame warrants creating a new keyframe.
2. **Local Mapping Thread**: inserts new keyframes, triangulates fresh 3D map points across co-visible views, runs local BA (or local visual-inertial BA), and culls redundant keyframes and unstable points to keep map size bounded.
3. **Loop Closing & Map Merging Thread**: queries a DBoW2 bag-of-words database to detect revisited places or overlaps with earlier sub-maps, verifies candidate matches geometrically in Sim(3) (monocular) or SE(3) (stereo/VI), and runs pose-graph optimization followed by global BA to eliminate accumulated drift.

**Tracking Thread in Detail**

At each frame, the tracking thread executes a tight four-step loop. It first predicts an initial pose from the previous frame using either a constant-velocity motion model or IMU preintegration. Projecting local map points into the image with that prediction narrows the ORB search window for fast correspondence matching. Motion-only BA then refines the 6-DoF pose against those matches, after which the thread triggers keyframe insertion if enough frames have elapsed, the fraction of tracked map points has dropped below a threshold, and sufficient map points remain visible to anchor the new keyframe.

**Visual-Inertial Mode**

In visual-inertial mode, ORB-SLAM3 adopts the on-manifold IMU preintegration of [Forster et al. (2017)](https://doi.org/10.1109/TRO.2016.2597321) inside a Maximum-a-Posteriori (MAP) estimator, minimizing visual reprojection residuals, IMU preintegration residuals, and any prior in one joint objective:

$$\mathcal{C} = \sum_{i,j} \rho\left(\left\|\mathbf{e}_{ij}^{\text{vis}}\right\|^2_{\Sigma_{ij}}\right) + \sum_k \left\|\mathbf{e}_k^{\text{IMU}}\right\|^2_{\Sigma_k^{\text{IMU}}} + \left\|\mathbf{e}^{\text{prior}}\right\|^2_{\Sigma^{\text{prior}}}$$

Each keyframe carries its pose, world velocity, and gyroscope/accelerometer biases in the state vector:

$$\mathbf{x}_k = [{}^W\mathbf{R}_k, {}^W\mathbf{p}_k, {}^W\mathbf{v}_k, \mathbf{b}_k^g, \mathbf{b}_k^a]$$

Between consecutive keyframes $k$ and $k+1$, the preintegrated inertial measurements (→ §4.6)

$$\Delta\mathbf{R}_{k,k+1}, \quad \Delta\mathbf{v}_{k,k+1}, \quad \Delta\mathbf{p}_{k,k+1}$$

define the IMU residual $\mathbf{e}_k^{\text{IMU}}$ through three error blocks measuring discrepancies in rotation, velocity, and position:

$$\mathbf{e}^{\text{IMU}}_{\Delta R} = \text{Log}\left(\Delta\hat{\mathbf{R}}_{k,k+1}^T \cdot \mathbf{R}_k^T \mathbf{R}_{k+1}\right)$$
$$\mathbf{e}^{\text{IMU}}_{\Delta v} = \mathbf{R}_k^T(\mathbf{v}_{k+1} - \mathbf{v}_k - \mathbf{g}\Delta t) - \Delta\hat{\mathbf{v}}_{k,k+1}$$
$$\mathbf{e}^{\text{IMU}}_{\Delta p} = \mathbf{R}_k^T(\mathbf{p}_{k+1} - \mathbf{p}_k - \mathbf{v}_k\Delta t - \frac{1}{2}\mathbf{g}\Delta t^2) - \Delta\hat{\mathbf{p}}_{k,k+1}$$

**Multi-Map System (Atlas)**

Earlier VO/SLAM pipelines often discarded the active map or froze whenever severe motion blur, occlusion, or rapid rotation broke visual tracking. ORB-SLAM3 resolves this with the **Atlas** multi-map manager: whenever tracking is lost and cannot relocalize immediately, the system spawns a fresh, independent sub-map rather than overwriting prior geometry. As soon as DBoW2 recognizes a previously mapped location inside another sub-map, the loop-closing thread aligns the two maps via a Sim(3) (monocular) or SE(3) (stereo/VI) weld and fuses them into a single consistent map—enabling both seamless recovery from tracking dropouts and multi-session mapping across separate runs.

**Scope of the ORB-SLAM3 evaluation**: The paper benchmarks its stereo-inertial configuration on EuRoC MAV and TUM-VI, reporting trajectory accuracy under each dataset's protocol. Any quantitative comparison should specify the error metric (ATE or RPE), the exact sequence set and aggregation rule, and how initialization delays or tracking failures are scored; when deploying on a new camera–IMU rig, remeasure ATE/RPE, failure rate, runtime, and memory directly on target sequences.

The following pseudocode summarizes how the tracking thread ties feature extraction, pose prediction, projection matching, motion-only BA, and keyframe insertion together on every incoming frame:

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

Instead of distilling an image down to discrete corners and matching descriptors, direct methods treat pixel intensities themselves as the geometric observations. The underlying physical assumption is brightness constancy: the same 3D surface point $\mathbf{P}$ should project to roughly the same intensity in consecutive frames, allowing motion and depth to be recovered directly by warping pixels and driving intensity differences to zero.

### 6.2.1 Photometric Error

Direct VO formulates its optimization around *photometric error*. Given camera poses $\mathbf{T}_i, \mathbf{T}_j$ and a pixel $\mathbf{u}_i$ in host frame $i$ parameterized by inverse depth $d_i^{-1}$, unprojecting $\mathbf{u}_i$ into 3D, transforming it into frame $j$, and reprojecting it yields a predicted intensity difference:

$$e_{\text{photo}} = I_j\left(\pi(\mathbf{T}_j \mathbf{T}_i^{-1} \pi^{-1}(\mathbf{u}_i, d_i^{-1}))\right) - I_i(\mathbf{u}_i)$$

Here $\pi^{-1}(\mathbf{u}, d^{-1})$ lifts a 2D pixel and its inverse depth back to a 3D point, $\pi(\cdot)$ projects 3D points onto the image plane, and $I_i, I_j$ denote the intensity images of frames $i$ and $j$.

Minimizing this photometric residual shifts data association from discrete descriptor matching to continuous image warping. The table below contrasts the resulting formulation with feature-based reprojection error:

| | Reprojection error | Photometric error |
|---|---|---|
| Observation | Feature coordinates (2D) | Pixel intensity (1D or patch) |
| Data association | Explicit (matching required) | Implicit (computed via warping) |
| Gradient | Geometric | Image gradient |
| Texture requirement | Corners/edges needed | Only gradient needed |
| Illumination change | Invariant (geometric) | Sensitive (correction needed) |

By avoiding explicit feature extraction and matching, direct methods can exploit every pixel with a measurable intensity gradient—allowing them to keep tracking along gentle shading variations on a blank wall or along smooth curves where corner detectors fire sparsely.

That flexibility comes with two structural vulnerabilities:

1. **Brightness constancy violation**: Real cameras rarely deliver pure irradiance. Scene lighting changes, auto-exposure shifts, and lens vignetting alter the recorded intensity of the same 3D point across frames; left uncorrected, they introduce systematic bias into the photometric residual.
2. **Limited basin of convergence**: Because image intensity varies non-monotonically over just a few pixels, gradient-based photometric alignment easily settles into the wrong local minimum when the initial pose guess is far off. The tolerable initialization error depends on pyramid level, texture frequency, viewpoint change, and photometric variation, so systems rely on coarse-to-fine image pyramids and residual checks rather than a fixed pixel cutoff.

### 6.2.2 DSO (Direct Sparse Odometry) Architecture Deep Dive

[DSO (Engel et al., 2018)](https://doi.org/10.1109/TPAMI.2017.2658577) broke an unwritten convention in early direct SLAM. Before DSO, the field largely conflated residual type with representation density—assuming that direct methods had to be dense (DTAM, Newcombe et al., ICCV 2011) or semi-dense ([LSD-SLAM, Engel et al., 2014](https://doi.org/10.1007/978-3-319-10605-2_54)), while sparse methods had to be feature-based (ORB-SLAM). DSO decoupled those two axes, pairing a direct photometric formulation with a sparse set of well-distributed points.

Three design pillars define DSO: it is **direct** (optimizing raw intensities with no corner descriptors), **sparse** (sampling points uniformly across image regions with gradient rather than processing entire dense depth maps), and **jointly optimized** (solving simultaneously for camera poses, point inverse depths, camera intrinsics, and per-frame affine brightness parameters).

**Full Photometric Calibration**

To keep brightness constancy valid across real camera hardware, DSO explicitly models the physical imaging pipeline relating true scene irradiance $B(\mathbf{u})$ to observed pixel value $I'(\mathbf{u})$:

$$I'(\mathbf{u}) = G(t \cdot V(\mathbf{u}) \cdot B(\mathbf{u}))$$

where $G(\cdot)$ is the camera's nonlinear response function, $t$ is the exposure time, and $V(\mathbf{u})$ captures lens vignetting (radial brightness falloff toward image borders). Inverting all three effects recovers an irradiance-proportional image:

$$I(\mathbf{u}) = t^{-1} \cdot G^{-1}(I'(\mathbf{u})) / V(\mathbf{u})$$

Without this calibration, a point sliding from the image center toward the vignette-darkened edge, or viewed under shifting auto-exposure, would register a spurious intensity drop that the optimizer could misinterpret as camera motion.

**Affine Brightness Transfer Function**

When full photometric calibration is unavailable or scene illumination fluctuates globally, DSO absorbs residual inter-frame brightness shifts through an affine brightness transfer model inside the Huber-weighted patch residual:

$$e_{\mathbf{p}j} = \sum_{\mathbf{p} \in \mathcal{N}_\mathbf{p}} w_\mathbf{p} \left\| (I_j[\mathbf{p}'] - b_j) - \frac{t_j e^{a_j}}{t_i e^{a_i}}(I_i[\mathbf{p}] - b_i) \right\|_\gamma$$

Here $a_i, b_i, a_j, b_j$ are per-frame affine brightness parameters optimized jointly alongside geometry, and $\|\cdot\|_\gamma$ denotes the Huber norm.

**Point Selection Strategy**

Rather than letting high-contrast clutter monopolize the point budget, DSO partitions each image into a regular grid and picks the pixel with the highest gradient magnitude inside each cell—adapting local thresholds downward when needed to retain roughly 2,000 evenly distributed points even in low-texture scenes.

**Sliding Window Optimization**

DSO maintains a sliding window of the most recent 5–7 keyframes and jointly optimizes all active poses, inverse depths, camera intrinsics $\mathbf{c}$, and affine brightness terms:

$$\boldsymbol{\theta} = \{\mathbf{T}_1, \ldots, \mathbf{T}_n, d_1^{-1}, \ldots, d_m^{-1}, \mathbf{c}, a_1, b_1, \ldots, a_n, b_n\}$$

As keyframes and points exit the window, DSO marginalizes them via the Schur complement into a dense quadratic prior—applying the same marginalization mechanism covered in §4.7 (→ §4.7), specialized here to purely photometric factors.

**Limitations and Extensions of DSO**

By design, vanilla DSO operates purely as odometry without loop closure. That omission reflects a deliberate scope choice rather than a limitation of direct residuals: LDSO adds loop closing via DBoW alongside direct alignment, while VI-DSO couples DSO tightly with an IMU. Note that Basalt (→ §6.3.5) is an independent system built on a patch-based direct frontend and Non-linear Factor Recovery rather than a DSO derivative.

The pseudocode below outlines DSO's per-frame loop from photometric correction and coarse-to-fine alignment through point sampling and windowed Gauss-Newton updates:

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

[SVO (Semi-direct Visual Odometry, Forster et al., 2017)](https://doi.org/10.1109/TRO.2016.2623335) bridges feature-based and direct paradigms: it tracks motion directly on pixel intensities, then hands refined coordinates to a feature-based bundle adjustment backend.

Every frame passes through three tightly staged steps. First, **Sparse Model-based Image Alignment** projects known 3D map points into the current image and recovers an initial pose by minimizing photometric error only over small patches around those projections—capturing DSO's gradient-based alignment without scanning the whole image. Second, **Feature Alignment** relaxes each projected point independently against its local patch to subpixel accuracy, compensating for slight drift in the initial pose or 3D coordinates. Third, **Structure & Motion Refinement** treats those photometrically aligned subpixel coordinates as "virtual feature points" and runs reprojection-error bundle adjustment to jointly optimize poses and 3D structure.

In the authors' hardware and resolution setup, SVO reaches frame rates of several hundred hertz; actual throughput depends on image resolution, map size, CPU architecture, and implementation, so it should always be profiled on the target platform. The baseline SVO pipeline omits loop closure, and pure camera rotation provides zero translation baseline for standard depth-filter updates unless handled explicitly.

---

## 6.3 Tightly-Coupled Visual-Inertial Odometry

Pure visual odometry leaves two fundamental gaps: a single camera cannot observe metric scale, and visual tracking stalls whenever rapid motion blurs frames or texture vanishes. An IMU complements the camera on both fronts—accelerometers and gravity anchor metric scale along with roll and pitch, while high-rate inertial propagation bridges brief visual outages. Tightly-coupled VIO folds raw visual observations and inertial measurements directly into one shared estimator so each sensor continuously calibrates and constrains the other.

### 6.3.1 VINS-Mono Architecture in Detail

[VINS-Mono (Qin et al., 2018)](https://doi.org/10.1109/TRO.2018.2853729) delivers full 6-DoF state estimation from a single monocular camera and a low-cost MEMS IMU, integrating initialization, sliding-window odometry, loop closure, and map reuse in one pipeline.

**System Architecture**

The pipeline separates into three stages: a **Frontend** for KLT optical-flow tracking and IMU preintegration, a **Backend** running tightly-coupled nonlinear sliding-window optimization, and a **Loop Closure** module performing DBoW2 place recognition and relocalization.

**Robust Initialization**

Monocular VIO faces a chicken-and-egg startup problem: a single camera gives poses only up to an unknown scale factor, yet inertial integration needs metric scale, initial velocity, gravity direction, and unknown IMU biases before it can constrain poses. VINS-Mono breaks this circularity with a two-step loosely-coupled initialization.

First, **Vision-only SfM** processes a short window of initial frames using images alone—estimating the Essential Matrix with the 5-point algorithm and triangulating 3D points to recover up-to-scale camera poses.

Second, **Visual-Inertial Alignment** brings that up-to-scale SfM trajectory into agreement with IMU preintegration in two sub-steps:

(a) *Gyroscope bias calibration*: Because rotations do not depend on metric scale, the relative rotation $\mathbf{R}_{c_k c_{k+1}}^{\text{sfm}}$ between consecutive SfM keyframes can be compared directly against the preintegrated gyro measurement $\Delta\hat{\mathbf{R}}_{k,k+1}$:

$$\min_{\mathbf{b}_g} \sum_k \left\| \text{Log}\left(\Delta\hat{\mathbf{R}}_{k,k+1}(\mathbf{b}_g)^T \cdot \mathbf{R}_{c_k}^{c_{k+1}} \right) \right\|^2$$

Applying the first-order preintegration Jacobian (→ §4.6) turns this rotation-alignment problem into a linear solve for $\mathbf{b}_g$.

(b) *Gravity, velocity, and scale estimation*: With $\mathbf{b}_g$ calibrated, preintegrated velocities and positions align linearly against the SfM translations. Across each keyframe pair $(k, k+1)$,

$$s\mathbf{R}_{c_0}^w \mathbf{p}_{c_{k+1}}^{c_0} - s\mathbf{R}_{c_0}^w \mathbf{p}_{c_k}^{c_0} - \mathbf{v}_k^w \Delta t_k - \frac{1}{2}\mathbf{g}^w\Delta t_k^2 = \mathbf{R}_k^w \Delta\hat{\mathbf{p}}_{k,k+1}$$

which forms a linear system in the unknown scale $s$, gravity vector $\mathbf{g}^w$, and frame velocities $\{\mathbf{v}_k^w\}$. Note that our sign convention follows $\mathbf{g}=[0,0,-9.81]^\top$ (→ Chapter 4); the original VINS-Mono paper adopts the opposite sign for $\mathbf{g}^w$, flipping the gravity term's sign relative to the equation above. Enforcing the known magnitude $\|\mathbf{g}\| = 9.81$ further sharpens gravity alignment and scale recovery. Once initialization converges, VINS-Mono hands the state off to the tightly-coupled nonlinear estimator.

**Tightly-Coupled Nonlinear Optimization**

Inside its sliding window, the VINS-Mono backend jointly minimizes three weighted residual blocks—a marginalization prior, IMU preintegration factors, and visual reprojection factors:

$$\min_{\mathcal{X}} \left\{ \left\|\mathbf{r}_p - \mathbf{H}_p \mathcal{X}\right\|^2 + \sum_{k \in \mathcal{B}} \left\|\mathbf{r}_{\mathcal{B}}(\hat{\mathbf{z}}_{b_k b_{k+1}}, \mathcal{X})\right\|^2_{\mathbf{P}_{b_k b_{k+1}}^{-1}} + \sum_{(l,j) \in \mathcal{C}} \left\|\mathbf{r}_{\mathcal{C}}(\hat{\mathbf{z}}_{l}^{c_j}, \mathcal{X})\right\|^2_{(\mathbf{P}_{l}^{c_j})^{-1}} \right\}$$

Here $\mathbf{r}_p$ captures the marginalization prior, $\mathbf{r}_{\mathcal{B}}$ measures the IMU preintegration discrepancy across body frames $b_k$ and $b_{k+1}$, $\mathbf{r}_{\mathcal{C}}$ is the visual reprojection residual, and $\mathcal{X}$ stacks all active keyframe poses, velocities, IMU biases, and feature inverse depths.

**Detailed Definition of the Visual Residual**:

To represent landmarks without blowing up on distant points, each feature $l$ is anchored to the keyframe $i$ that first observed it and parameterized by a scalar inverse depth $\lambda_l$. Projecting that 3D point through camera–IMU extrinsics into another keyframe $j$ gives the normalized-plane reprojection error:

$$\mathbf{r}_{\mathcal{C}} = \begin{bmatrix} \bar{u}_l^{c_j} - u_l^{c_j} / z_l^{c_j} \\ \bar{v}_l^{c_j} - v_l^{c_j} / z_l^{c_j} \end{bmatrix}$$

where $\begin{bmatrix} u_l^{c_j} & v_l^{c_j} & z_l^{c_j} \end{bmatrix}^T = \mathbf{R}_{b_j}^{c} (\mathbf{R}_{w}^{b_j}(\mathbf{R}_{b_i}^{w}(\mathbf{R}_{c}^{b_i} \frac{1}{\lambda_l}\begin{bmatrix} \bar{u}_l^{c_i} \\ \bar{v}_l^{c_i} \\ 1 \end{bmatrix} + \mathbf{p}_c^b) + \mathbf{p}_{b_i}^w - \mathbf{p}_{b_j}^w) - \mathbf{p}_c^b)$

**Sliding Window Management and Marginalization**

VINS-Mono's reference configuration bounds computation by keeping roughly ten frame states in a fixed-length window and selecting between two marginalization rules depending on whether the newest frame brings enough parallax to become a keyframe:

1. **When the newest frame is a keyframe**, the oldest keyframe leaves the window, and all visual and inertial factors tied to it are folded into the prior via the Schur complement.
2. **When the newest frame is not a keyframe**, the second-newest frame is dropped instead. In this branch, its visual measurements are simply discarded rather than converted into a prior, while its IMU measurements stay fused into the preintegrated factor spanning the remaining adjacent keyframes—preserving inertial continuity without filling the window with redundant low-parallax views (see §4.7 for the Schur complement mechanics).

**4-DoF Pose Graph Optimization**

When loop closure fires, VINS-Mono runs a 4-DoF pose-graph optimization over yaw and 3D translation rather than all 6 DoF. Because gravity aligns roll and pitch continuously whenever the IMU has sufficient excitation and well-modeled biases, only global yaw and 3D position retain unobservable gauge freedom; fixing local roll and pitch lets global loop closure close drift cleanly without warping gravity alignment.

The following Python-style sketch illustrates how `VINSEstimator` assembles the three residual terms and switches between the two marginalization branches:

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
            # Drop the previous (non-keyframe) frame: discard its visual measurements and chain the IMU via preintegration
            drop_visual_factors(self.states[-2])
            merge_imu_preintegration(self.states[-3], self.states[-2], self.states[-1])
            self.states.pop(-2)
```

### 6.3.2 OKVIS

Predating VINS-Mono, [OKVIS (Open Keyframe-based Visual-Inertial SLAM, Leutenegger et al., 2015)](https://doi.org/10.1177/0278364914554813) pioneered keyframe-based sliding-window visual-inertial optimization inside Ceres Solver. Rather than ORB or optical flow, OKVIS pairs Harris corners with binary BRISK descriptors and marginalizes old keyframes out of a bounded window much like VINS-Mono. The key difference lies in its inertial term: instead of preintegration, the original OKVIS repropagated IMU kinematics directly between frames to build a 15-dimensional residual spanning orientation, velocity, position, and biases ("speed and bias" denotes the state block $[\mathbf{v}, \mathbf{b}_g, \mathbf{b}_a]$, not a separate error term; OKVIS2 later adopted preintegration). Because OKVIS targets stereo rigs by default, its initialization is straightforward compared with VINS-Mono, whereas VINS-Mono excels at bootstrapping monocular-inertial motion from scratch.

### 6.3.3 MSCKF: Multi-State Constraint Kalman Filter

While VINS-Mono and ORB-SLAM3 rely on nonlinear graph optimization, [MSCKF (Mourikis & Roumeliotis, 2007)](https://doi.org/10.1109/ROBOT.2007.364024) remains the canonical filter-based VIO algorithm—and is still widely deployed where bounded, predictable filter updates are prized.

**MSCKF Excludes Landmarks from the State**

Standard EKF-SLAM places every tracked 3D landmark directly inside the filter state vector. With $L$ landmarks, the state grows to dimension $3L + 15$ and the dense covariance matrix to $(3L+15)^2$, imposing $O(L^2)$ memory and $O(L^2)$ covariance-update work per step. Even though the innovation matrix stays small (matching measurement dimension rather than full state dimension) so updates do not require factoring the full covariance, quadratic growth in $L$ quickly overwhelms real-time budgets in feature-rich scenes.

MSCKF sidesteps that bottleneck with an algebraic insight: **exclude all 3D landmarks from the filter state while retaining the geometric constraints they induce across camera poses**.

**State Vector Structure**

Instead of landmarks, the MSCKF state stacks the current 15-DoF IMU error state alongside a short sliding window of $N$ cloned camera poses:

$$\tilde{\mathbf{x}} = [\tilde{\mathbf{x}}_{IMU}^T, \tilde{\mathbf{x}}_{C_1}^T, \ldots, \tilde{\mathbf{x}}_{C_N}^T]^T$$

Here the IMU error state tracks orientation, gyro bias, velocity, accelerometer bias, and position,

$$\tilde{\mathbf{x}}_{IMU} = [\delta\boldsymbol{\theta}^T, \tilde{\mathbf{b}}_g^T, {}^G\tilde{\mathbf{v}}_I^T, \tilde{\mathbf{b}}_a^T, {}^G\tilde{\mathbf{p}}_I^T]^T \in \mathbb{R}^{15}$$

and each cloned camera pose adds just 6 error states:

$$\tilde{\mathbf{x}}_{C_k} = [\delta\boldsymbol{\theta}_{C_k}^T, {}^G\tilde{\mathbf{p}}_{C_k}^T]^T \in \mathbb{R}^6$$

The entire state dimension is fixed at $15 + 6N$, completely independent of how many landmarks are tracked.

**Null-Space Projection**

How does MSCKF turn visual observations of a feature $\mathbf{p}_f$ into pose constraints without putting $\mathbf{p}_f$ in $\tilde{\mathbf{x}}$? Suppose $\mathbf{p}_f$ is observed across $M$ cloned camera poses. Linearizing the stacked reprojection residuals around the current state and triangulated feature estimate yields:

$$\mathbf{r} = \mathbf{H}_X \tilde{\mathbf{x}} + \mathbf{H}_f \tilde{\mathbf{p}}_f + \mathbf{n}$$

where $\mathbf{r} \in \mathbb{R}^{2M}$ is the measurement residual (observation minus prediction), $\mathbf{H}_X \in \mathbb{R}^{2M \times (15+6N)}$ is the Jacobian with respect to the state, $\mathbf{H}_f \in \mathbb{R}^{2M \times 3}$ is the Jacobian with respect to the 3D feature position, and $\tilde{\mathbf{p}}_f \in \mathbb{R}^3$ is the unknown feature triangulation error.

Taking a QR decomposition of $\mathbf{H}_f$ separates the column space $\mathbf{Q}_1$ from its left null space $\mathbf{Q}_2$:

$$\mathbf{H}_f = \begin{bmatrix} \mathbf{Q}_1 & \mathbf{Q}_2 \end{bmatrix} \begin{bmatrix} \mathbf{R}_1 \\ \mathbf{0} \end{bmatrix}$$

Because $\mathbf{Q}_2^T \mathbf{H}_f = \mathbf{0}$ by construction, left-multiplying the linearized measurement equation by $\mathbf{Q}_2^T$ annihilates the $\tilde{\mathbf{p}}_f$ term cleanly:

$$\mathbf{r}_o = \mathbf{Q}_2^T \mathbf{r} = \mathbf{Q}_2^T \mathbf{H}_X \tilde{\mathbf{x}} + \mathbf{Q}_2^T \mathbf{n} = \mathbf{H}_o \tilde{\mathbf{x}} + \mathbf{n}_o$$

Geometrically, projecting onto $\mathbf{Q}_2^T$ removes the 3 degrees of freedom of the 3D point and leaves the $2M - 3$ independent epipolar/multi-view constraints linking all $M$ observing camera poses—the "multi-state constraint" that gives MSCKF its name.

**Computational Complexity**: Because the filter state has dimension $15 + 6N$ (with $N \ll L$ cloned camera poses), covariance updates avoid the $O(L)$ state growth and $O(L^2)$ update cost that saddle EKF-SLAM as the landmark count $L$ scales up.

**MSCKF Update Procedure**

In operation, MSCKF cycles through four steps:

1. **IMU propagation**: Each incoming IMU sample advances the nominal state and covariance:
   $$\hat{\mathbf{x}}_{k+1|k} = f(\hat{\mathbf{x}}_{k|k}, \mathbf{u}_k)$$
   $$\mathbf{P}_{k+1|k} = \boldsymbol{\Phi}_k \mathbf{P}_{k|k} \boldsymbol{\Phi}_k^T + \mathbf{G}_k \mathbf{Q} \mathbf{G}_k^T$$

2. **State augmentation**: Whenever an image arrives, the current IMU pose is transformed to the camera frame and cloned into the state vector, expanding $\mathbf{P}$ accordingly.

3. **MSCKF update**: Once a tracked feature leaves the field of view (or reaches the window limit), its 3D position $\hat{\mathbf{p}}_f$ is triangulated, projected into the null space to form $(\mathbf{r}_o, \mathbf{H}_o)$, and assimilated via a standard EKF update:
   $$\mathbf{K} = \mathbf{P} \mathbf{H}_o^T (\mathbf{H}_o \mathbf{P} \mathbf{H}_o^T + \sigma^2 \mathbf{I})^{-1}$$
   $$\hat{\mathbf{x}} \leftarrow \hat{\mathbf{x}} + \mathbf{K} \mathbf{r}_o$$
   $$\mathbf{P} \leftarrow (\mathbf{I} - \mathbf{K}\mathbf{H}_o)\mathbf{P}$$

4. **State pruning**: Old camera clones are dropped from the sliding window (marginalized implicitly by removing their rows and columns from the mean and covariance).

The C++ pseudocode below traces step 3—from triangulation and Jacobian assembly through Householder QR null-space projection to the final on-manifold EKF correction:

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

[OpenVINS (Geneva et al., 2020)](https://doi.org/10.1109/ICRA40945.2020.9196524) provides a modular open-source research platform built around an on-manifold sliding-window MSCKF (representing rotations directly on SO(3)) and a built-in trajectory simulator with configurable environments and IMU noise models for controlled benchmarking against ground truth.

Beyond the vanilla MSCKF formulation, OpenVINS introduces three practical capabilities:

1. **Online calibration**: When enabled, camera intrinsics, camera–IMU extrinsics, and time offset can be co-estimated in the state vector—though these parameters stay stable only under sufficiently exciting motion and rich scene geometry, so hardware-synchronized, well-calibrated rigs often keep them fixed.
2. **Hybrid SLAM landmarks**: Long-lived features that outlast the sliding pose window can optionally be promoted into the state as anchored inverse-depth SLAM features, boosting accuracy when revisiting static landmarks.
3. **First-Estimates Jacobian (FEJ)**: By evaluating selected Jacobians at initial state estimates rather than freshly updated states, FEJ prevents unobservable gauge directions (such as global yaw) from falsely gaining spurious information—an important consistency safeguard, though not a cure-all for unmodeled dynamics, mistuned noise covariances, or large linearization errors.

### 6.3.5 Basalt

[Basalt (Usenko et al., 2020)](https://doi.org/10.1109/LRA.2019.2961227) shares VINS-Mono's tightly-coupled sliding-window optimization philosophy while rethinking both the visual frontend and marginalization. On the frontend, it replaces point-wise KLT with inverse-compositional patch-wise KLT tracking under an SE(2) warp—improving subpixel tracking stability across stereo and multi-camera rigs. On the backend, it replaces standard Schur-complement marginalization with **Non-linear Factor Recovery (NFR)**: whereas a standard marginalization prior locks in a fixed linearization point (distorting information if later updates move the state far away), NFR distills the marginalized sub-graph into compact nonlinear factors that can still be relinearized cleanly. Combined with sparse factor-graph engineering, Basalt runs efficiently on stereo+IMU hardware—though any speed comparison against mono+IMU pipelines like VINS-Mono only holds when sensor count, image resolution, and compute hardware are matched.

---

## 6.4 VIO Design Choices

Beyond choosing a frontend or sensor suite, three architectural decisions dictate a VIO system's accuracy, memory footprint, and real-time headroom: how the backend estimates states, when it promotes frames to keyframes, and how it parameterizes 3D points.

### 6.4.1 Filter vs Optimization

Whether to formulate the backend as a recursive filter or a sliding-window optimizer remains one of VIO's longest-running trade-offs.

**Filter-based backends (MSCKF, OpenVINS)** update a compact state vector—current IMU pose, velocity, biases, and a short queue of cloned camera poses—sequentially as measurements arrive. When older camera poses are pruned, their information remains folded into the maintained Gaussian mean and covariance. Because updates require $O(N^2)$ work in the small state dimension $N$, filter-based VIO delivers deterministic, low-latency per-frame runtimes and clean implementation. The price is single-shot linearization: once a measurement updates the state, its linearization point is locked in, leaving the filter vulnerable to accumulated linearization error and observability-consistency drift.

**Optimization-based backends (VINS-Mono, ORB-SLAM3, Basalt)** retain a sliding window of recent keyframes and solve a joint nonlinear least-squares problem across all active visual and inertial factors. Although naive dense Gauss-Newton scales as $O(N^3)$ per iteration in window size $N$, exploiting factor sparsity via the Schur complement keeps solver runtimes tractable while allowing iterative relinearization across the active window and straightforward insertion of diverse residual types. The trade-off is higher peak compute and sensitivity to window management.

Under matched conditions, optimization-based estimators generally edge out filters in raw accuracy because they can relinearize past states. Yet once First-Estimates Jacobians and careful observability constraints are enforced, modern MSCKF systems close much of that gap—keeping filters a compelling choice on tightly constrained compute such as ultra-low-power microcontrollers.

### 6.4.2 Keyframe Selection Strategies

Keyframe selection acts as the throttle on both accuracy and compute: promoting frames too eagerly swells the sliding window and marginalization overhead, whereas spacing them too sparsely breaks visual overlap and discards parallax.

In practice, VIO systems trigger new keyframes using four main criteria:

1. **Time-based**: insert a keyframe once a minimum interval has elapsed since the previous one, bounding inertial integration gaps.
2. **Parallax-based**: insert a keyframe when the average feature parallax relative to the last keyframe crosses a threshold (used by VINS-Mono), guaranteeing geometric baseline for triangulation.
3. **Tracking quality-based**: insert a keyframe whenever the count or fraction of tracked map features falls below a threshold (used by ORB-SLAM3), refreshing the local map before tracking degrades.
4. **Information gain-based**: select frames by estimating how much Fisher information a candidate keyframe adds—theoretically the cleanest rule, though rarely worth its extra compute at frame rate.

Because every new keyframe eventually forces an older state out of the window, keyframe policy is tightly coupled to marginalization—as seen in VINS-Mono's two-branch scheme (→ §6.3.1), which drops redundant non-keyframes immediately while summarizing true keyframes into the prior.

### 6.4.3 Feature Parameterization

How 3D landmarks are parameterized strongly shapes numerical conditioning, especially for monocular systems operating across wide depth ranges.

**XYZ (Euclidean 3D coordinates)**

Representing a point directly as $\mathbf{P} = [X, Y, Z]^T \in \mathbb{R}^3$ is straightforward, but it becomes ill-conditioned for distant scenery: when a landmark is far from the camera, tiny angular jitter in ray direction produces huge swings in $Z$, skewing linearization and covariance.

**Inverse Depth**

Inverse depth avoids that blow-up by storing a landmark's viewing direction from its host frame alongside the reciprocal of distance:

$$\boldsymbol{\lambda} = [\theta, \phi, \rho]^T$$

where $\theta$ and $\phi$ give azimuth and elevation in the host frame and $\rho = 1/d$ is inverse depth. Two geometric properties make this representation standard in VINS-Mono and OpenVINS:

1. **Bounded behavior at long range**: as $d \to \infty$, $\rho \to 0$ smoothly without numerical divergence, whereas Euclidean $Z \to \infty$ destabilizes the coordinates—letting distant sky-line or mountain features contribute clean bearing constraints.
2. **Better Gaussianity under monocular uncertainty**: during early initialization when depth is poorly constrained, uncertainty in $1/d$ is far closer to Gaussian than uncertainty in $d$, improving linearized filter and Gauss-Newton updates.

**Anchored Inverse Depth**

Anchored inverse depth ties $\rho$ directly to a designated anchor keyframe pose $\mathbf{T}_{\text{anchor}}$ and normalized image coordinates $(\bar{u}, \bar{v})$:

$$\mathbf{P} = \mathbf{T}_{\text{anchor}} \cdot \frac{1}{\rho} [\bar{u}, \bar{v}, 1]^T$$

Because the point's coordinates remain fixed relative to its anchor frame, updating the anchor pose during optimization shifts the 3D point rigidly without distorting $\rho$, curbing linearization error. OpenVINS adopts this parameterization for its promoted SLAM features, whereas ORB-SLAM3 stores and optimizes map points directly as Euclidean 3D coordinates in the world frame.

---

## 6.5 Learning-Based VO/VIO

Classical VO and VIO pipelines rely on handcrafted stages—corner detection, descriptor matching, RANSAC, and bundle adjustment. Learning-based methods seek to replace individual stages, or the entire pipeline, with neural networks trained directly on motion and image streams.

### 6.5.1 Supervised: the DeepVO Family

Early end-to-end systems such as [DeepVO (Wang et al., 2017)](https://doi.org/10.1109/ICRA.2017.7989236) bypassed explicit geometry altogether: a CNN extracted visual features from consecutive frame pairs, and an LSTM modeled temporal dynamics to regress relative 6-DoF poses directly. Although conceptually simple, pure black-box pose regression suffers from three persistent weaknesses: it overfits to the visual appearance and motion statistics of training scenes, ignores hard geometric constraints such as epipolar geometry—leaving accuracy behind classical estimators—and accumulates severe scale drift.

### 6.5.2 Self-supervised: Limitations and Current Position

Self-supervised frameworks such as SfMLearner and Monodepth2 eliminate ground-truth pose labels by jointly predicting monocular depth and ego-motion and training on view-synthesis photometric reconstruction losses. While effective for monocular depth estimation, standalone self-supervised VO still lags classical pipelines in trajectory accuracy, struggles around dynamic objects, textureless surfaces, and occlusions, and leaves long-term scale and pose drift unresolved.

### 6.5.3 Hybrid: DROID-SLAM

Rather than discarding projective geometry, [DROID-SLAM (Teed & Deng, 2021)](https://arxiv.org/abs/2108.10869) fuses learned dense correspondence updates directly with a differentiable bundle adjustment solver. Across the benchmarks and protocols reported in the paper—including TartanAir, EuRoC, and TUM-RGBD—this hybrid architecture delivers markedly lower trajectory errors and far fewer catastrophic tracking losses than prior learned and classical baselines.

**Architecture**

DROID-SLAM iterates between a learned optical-flow update module and a geometric solver:

1. **RAFT-Inspired Iterative Update Operator**

Borrowing from RAFT (Recurrent All-Pairs Field Transforms, Teed & Deng, 2020), each co-visible frame pair $(i, j)$ constructs a 4D all-pairs correlation volume from dense image feature maps. Indexing into this volume with the current projected correspondence field, a 3×3 Convolutional GRU takes correlation features, current flow, and image context and predicts a **flow revision** $\Delta\mathbf{f}_{ij}$ together with per-pixel **confidence weights** $\mathbf{w}_{ij}$:

$$(\Delta\mathbf{f}_{ij}, \mathbf{w}_{ij}) = \text{ConvGRU}(\text{corr}_{ij}, \mathbf{f}_{ij}^{\text{curr}}, \text{context}_i)$$

Repeated iterations progressively snap correspondences onto consistent scene motion.

2. **Differentiable Dense Bundle Adjustment (DBA) Layer**

Every GRU revision feeds directly into a geometric Gauss-Newton solver that refines camera poses $\mathbf{T}_i \in SE(3)$ and dense per-pixel inverse depths $d_i$ by minimizing confidence-weighted reprojection error across all pixels:

$$\sum_{(i,j)} \sum_{\mathbf{p}} \left\| \mathbf{w}_{ij}^{\mathbf{p}} \circ (\mathbf{p}^{*}_{ij} - \pi(\mathbf{T}_j \circ \mathbf{T}_i^{-1} \circ \pi^{-1}(\mathbf{p}, d_i^{\mathbf{p}}))) \right\|^2$$

where $\mathbf{p}^{*}_{ij} = \mathbf{p} + \mathbf{f}_{ij}^{\text{curr}} + \Delta\mathbf{f}_{ij}$ denotes the GRU-revised target pixel and $\mathbf{w}_{ij}^{\mathbf{p}}$ weights each pixel's reliability. Linearizing this objective gives the same block-structured Gauss-Newton normal equations as classical BA:

$$\begin{bmatrix} \mathbf{B} & \mathbf{E} \\ \mathbf{E}^T & \mathbf{C} \end{bmatrix} \begin{bmatrix} \boldsymbol{\xi} \\ \delta\mathbf{d} \end{bmatrix} = \begin{bmatrix} \mathbf{v} \\ \mathbf{w} \end{bmatrix}$$

Here $\boldsymbol{\xi}$ is the $SE(3)$ pose increment, $\delta\mathbf{d}$ is the dense inverse-depth update, $\mathbf{B}$ is the pose Hessian block, $\mathbf{C}$ is the diagonal depth block, and $\mathbf{E}$ couples poses to depths. Eliminating $\delta\mathbf{d}$ via the Schur complement mirrors classical BA:

$$(\mathbf{B} - \mathbf{E}\mathbf{C}^{-1}\mathbf{E}^T)\boldsymbol{\xi} = \mathbf{v} - \mathbf{E}\mathbf{C}^{-1}\mathbf{w}$$

Because $\mathbf{C}$ is strictly diagonal, $\mathbf{C}^{-1}$ is element-wise and cheap, keeping the reduced linear system proportional only to the number of keyframes. Crucially, the entire Gauss-Newton step is **differentiable**: end-to-end backpropagation teaches the ConvGRU to produce correspondences and weights that yield clean geometric poses rather than just low 2D flow error.

**Frame Graph and Loop Closure**

DROID-SLAM dynamically maintains a co-visibility graph over keyframes—linking new frames to recent neighbors and inserting long-range edges whenever co-visibility with earlier keyframes re-emerges, so global BA closes loops across the full trajectory.

**Multi-Modality Support**

Even though DROID-SLAM trains purely on monocular video, its geometric DBA layer seamlessly accepts stereo or RGB-D inputs at inference time: stereo pairs enter as two frames linked by a fixed known baseline, while RGB-D depth seeds initial inverse depths and adds direct depth constraints.

**Performance**

Under the comparison tables and evaluation protocols reported in the DROID-SLAM paper:

- TartanAir: the DROID-SLAM paper reports a 62% error reduction against its previous compared method
- EuRoC monocular: the same paper reports an 82% reduction under its protocol
- ETH-3D: 30 out of 32 sequences succeed (previous best: 19)
- After training only on synthetic data (TartanAir), it reports lower error than the compared methods on the selected metrics of four real-world datasets

**Geometric Backend and Learned Correspondence**

By pairing recurrent learned matching with exact differentiable bundle adjustment, DROID-SLAM stays resilient across repetitive patterns, low-texture regions, and lighting shifts where handcrafted corner matchers drop tracking. Three practical limits remain, however:

- **Compute**: the public implementation relies on a GPU, with throughput governed by image resolution, active keyframe count, hardware, and configuration.
- **No IMU**: the baseline architecture is vision-only; tightly integrating inertial measurements remains an active research topic.
- **Memory**: retaining dense feature maps and 4D correlation volumes for keyframes carries a heavy memory footprint.

Subsequent work has focused on trimming that memory and compute overhead with sparse patches and adding inertial constraints.

### 6.5.4 Recent Trends (2023-2025)

Building directly on DROID-SLAM's differentiable BA paradigm, [DPVO (Teed, Lipson & Deng, 2023)](https://arxiv.org/abs/2208.04726) trades dense all-pairs correlation for sparse patch tracking coupled with a patch-wise recurrent update operator and differentiable BA. Under its evaluation setup, the paper reports roughly one-third the memory footprint and about three times the throughput of DROID-SLAM while preserving competitive accuracy.

Taking a complementary angle on learned uncertainty, [MAC-VO (Qiu et al., 2024)](https://arxiv.org/abs/2409.09479) predicts metrics-aware matching covariance for stereo VO, using those learned uncertainties both to select reliable keypoints and to weight residuals inside pose-graph optimization. The paper reports lower trajectory error than compared baselines on benchmarks featuring severe lighting changes and sparse texture, earning the ICRA 2025 Best Paper Award on Robot Perception.

---

## Summary

| System | Type | Estimation | Sensors | Features |
|--------|------|-----------|------|-----------|
| ORB-SLAM3 | Feature-based | Optimization (BA) | Mono/Stereo/RGBD + IMU | Multi-map Atlas, fisheye support |
| DSO | Direct | Optimization (windowed) | Mono | Photometric calibration, sparse sampling |
| SVO | Semi-direct | Optimization (BA) | Mono/Stereo | Semi-direct frontend, sparse depth, designed for high-rate processing |
| VINS-Mono | Feature-based | Optimization (sliding window) | Mono + IMU | Robust initialization, 4-DoF loop closure |
| MSCKF | Feature-based | EKF (sliding window) | Mono/Stereo + IMU | Excludes landmarks from state, null-space projection |
| OpenVINS | Feature-based | EKF (MSCKF) | Mono/Stereo + IMU | Online calibration, FEJ, research platform |
| Basalt | Semi-direct | Optimization (sliding window) | Stereo + IMU | NFR, efficient implementation |
| DROID-SLAM | Learned | Differentiable BA | Mono/Stereo/RGBD | Differentiable BA, trained on synthetic data |
| DPVO | Learned (sparse) | Differentiable BA | Mono | Sparse patches; paper reports lower compute and memory cost than DROID-SLAM under its setup |
| MAC-VO | Learned + Opt. | Pose graph opt. | Stereo | Metrics-aware covariance, ICRA 2025 Best Paper on Robot Perception |

Across these ten systems, one overarching pattern stands out: whether correspondence comes from ORB descriptors, optical flow, photometric gradients, or recurrent neural updates, camera-based odometry remains bound by projective geometry—and by passive reliance on ambient light. Switching from passive 2D cameras to active 3D LiDAR sensing transforms the geometry of ego-motion estimation once again, trading photometric sensitivity and scale ambiguity for dense metric range measurements, continuous scan distortion, and point cloud registration—the focus of Chapter 7.
