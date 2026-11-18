# Chapter 1. Why Sensor Fusion

Every autonomous vehicle, drone, or service robot starts with the same perceptual challenge: choosing which sensors to trust to understand the world. Because no single sensor sees everything reliably under all conditions, sensor fusion begins where individual sensors fall short. This chapter examines the physical limits of core robotic sensors, classifies the ways their measurements can be combined, compares system coupling architectures, and maps out how deep learning has reshaped—and where it still relies on—classical estimation theory.

---

## 1.1 Limitations of a Single Sensor

Every sensor observes the environment through a specific physical principle, and that very principle dictates both its strengths and its blind spots. Understanding these failure modes is the first step toward combining sensors effectively.

### Camera Limitations

Cameras capture rich semantic and geometric detail at low cost, yet passive optical imaging breaks down under several common operating conditions.

**Illumination dependence.** Because a camera passively detects reflected ambient light, image quality deteriorates sharply at night, inside tunnels, or against strong backlighting. Auto-exposure helps within limits, but once a scene's contrast exceeds the sensor's dynamic range, saturation or underexposure becomes unavoidable.

**Scale ambiguity.** Projecting a 3D scene onto a 2D image plane discards depth. Without additional constraints, a 1 m object at 2 m and a 10 m object at 20 m occupy the same footprint in the image. Consequently, monocular visual odometry cannot recover absolute metric scale from projective geometry alone; doing so requires stereo baselines, fusion with metric sensors such as an IMU, or prior knowledge of object size.

**Textureless environments.** Feature-based pipelines rely on distinct corners and edges, which vanish along blank white walls, long uniform corridors, or wide asphalt roads. Direct visual odometry suffers similarly whenever intensity gradients are too weak to constrain motion.

**Motion blur.** Rapid vehicle motion or sharp angular rotation smears pixel intensities across the frame, degrading both keypoint extraction and descriptor matching.

### LiDAR Limitations

Unlike passive cameras, LiDAR (Light Detection And Ranging) actively emits laser pulses and measures time-of-flight to deliver accurate 3D geometry independent of ambient light. Even so, laser ranging carries distinct trade-offs.

**Absence of texture information.** While LiDAR captures 3D structure with high precision, it records virtually no color or visual texture—and although some units report return intensity, that signal is far sparser than an RGB image. As a result, place recognition often struggles in geometrically repetitive environments such as street canyons lined with nearly identical buildings.

**Weather and environmental sensitivity.** Rain, fog, snow, and airborne dust scatter laser pulses, generating dense clouds of ghost points and sharply shrinking effective range. Returns also become erratic on dark, low-reflectivity targets or specular surfaces such as glass and polished metal.

**Resolution and cost.** Mechanical spinning units span 16 to 32 vertical channels in entry-level models, 64 channels in reference sensors like the Velodyne HDL-64E used in KITTI, and up to 128 channels at the high end—where prices range from several thousand to tens of thousands of dollars. Solid-state LiDARs bring costs down significantly, though typically at the expense of a narrower field of view (FoV).

### IMU Limitations

An inertial measurement unit (IMU) measures linear acceleration and angular velocity at high rates (typically 100 Hz–1 kHz) without relying on external signals or lighting. That environmental independence comes at a steep mathematical price: errors accumulate whenever inertial measurements are integrated over time.

**Drift.** Computing velocity and position from raw inertial readings requires integrating once for velocity and twice for position, causing sensor biases and noise to compound rapidly. In particular, a constant accelerometer bias drives position error quadratically in $t^2$. Because real-world drift depends heavily on sensor grade, thermal compensation, initialization quality, motion profile, and external aiding, an IMU's grade alone never guarantees a fixed error bound after a given interval:

$$\delta \mathbf{p}(t) \approx \frac{1}{2} \mathbf{b}_a \, t^2 + \frac{1}{\sqrt{3}} \sigma_a \, t^{3/2}$$

where $\mathbf{b}_a$ is the accelerometer bias and $\sigma_a$ is the acceleration noise density. Even a modest bias of $0.01\,\text{m/s}^2$ produces 0.5 m of position error in just 10 seconds.

**Absence of an absolute reference.** Because an IMU senses only relative motion, it cannot determine global position or heading on its own. Stationary periods—or intervals where the accelerometer cleanly observes gravity—allow roll and pitch alignment, yet yaw remains unobservable to a standalone IMU. Anchoring heading requires an external reference such as a magnetometer, GNSS course, or alignment against a camera or LiDAR map.

### GNSS Limitations

Global Navigation Satellite Systems (GNSS) fill that global-reference gap outdoors, but satellite signals are fragile and slow to update.

**Obstructed environments.** Indoors, inside tunnels, or beneath tall buildings in dense urban canyons, satellite signals are blocked or bounce off structures via multipath propagation, easily introducing tens of meters of position error.

**Update rate.** Most receivers output fixes at just 1–10 Hz, too slow to capture agile maneuvers or rapid platform dynamics on their own.

**Precision limits.** Standard single-point positioning offers only meter-level accuracy. Real-Time Kinematic (RTK) processing narrows that error to the centimeter scale, but demands a nearby base station or correction stream and requires initial convergence time before achieving a fixed solution.

### Complementarity of Sensor Limitations

Comparing these four core sensors side by side reveals a clear pattern: nearly every weakness in one modality is offset by a strength in another.

| Property | Camera | LiDAR | IMU | GNSS |
|------|--------|-------|-----|------|
| Absolute position | ✗ | ✗ | ✗ | ✓ |
| Relative motion (short-term) | ✓ | ✓ | ✓ | ✗ |
| Relative motion (long-term) | △ (drift) | △ (drift) | ✗ (diverges) | ✓ |
| High-frequency motion capture | ✗ | ✗ | ✓ | ✗ |
| Illumination independence | ✗ | ✓ | ✓ | ✓ |
| Weather robustness | △ | ✗ | ✓ | ✓ |
| 3D geometric information | △ (depth ambiguous) | ✓ | ✗ | ✗ |
| Texture / semantic information | ✓ | ✗ | ✗ | ✗ |
| Indoor operation | ✓ | ✓ | ✓ | ✗ |
| Cost | Low | High | Medium–Low | Medium |

No single sensor covers every row of this table reliably. Sensor fusion provides a principled framework for turning these complementary traits into a consistent state estimate—which raises the next question: in what distinct ways can sensors be combined?

---

## 1.2 Taxonomy of Sensor Fusion

Sensor fusion combines measurements from multiple sensors to achieve accuracy and robustness beyond what any standalone sensor can deliver. Depending on how their physical observations relate, fusion strategies fall into three primary categories.

### Complementary Fusion

Complementary fusion pairs sensors that measure different physical quantities so each compensates for what the other misses. Because each modality observes a distinct subset of the system state, combining them reveals state variables that remain unobservable to either sensor alone.

**Representative example: Camera + IMU (Visual-Inertial Odometry)**

- The camera tracks 6-DoF relative motion well, yet lacks metric scale and struggles during fast maneuvers.
- The IMU supplies high-rate acceleration and angular velocity to bridge gaps between image frames; given sufficient motion excitation, metric acceleration paired with visual constraints renders scale observable, while gravity measurements anchor roll and pitch.
- Together, the IMU supplies metric scale and inter-frame dynamics, while visual observations continuously bound inertial drift.

**Representative example: GNSS + IMU**

- GNSS provides low-rate (1–10 Hz) absolute global positions.
- The IMU tracks high-rate (hundreds of Hz) local motion between fixes.
- When satellite signals drop out inside a tunnel, the IMU bridges the outage over the short term; once GNSS returns, global fixes reset the accumulated inertial drift.

The core benefit of complementary fusion is **extended observability**: state components hidden from one sensor become directly observable through another.

### Competitive Fusion

Competitive fusion deploys multiple sensors measuring the same physical quantity to increase fault tolerance and reduce estimation variance.

**Representative example: multi-camera systems**

- Two forward-facing cameras independently track visual features across overlapping scenes.
- If one lens is occluded, fouled, or fails outright, the remaining camera keeps the system operational.
- Combining both independent estimates reduces variance below what either camera achieves individually.

**Statistical foundation.** Suppose two sensors produce unbiased, independent Gaussian observations of the same quantity, $z_1 \sim \mathcal{N}(\mu, \sigma_1^2)$ and $z_2 \sim \mathcal{N}(\mu, \sigma_2^2)$. Weighting each measurement by its inverse variance yields the minimum-variance estimate:

$$\hat{\mu} = \frac{\sigma_2^2 z_1 + \sigma_1^2 z_2}{\sigma_1^2 + \sigma_2^2}, \quad \sigma_{\text{fused}}^2 = \frac{\sigma_1^2 \sigma_2^2}{\sigma_1^2 + \sigma_2^2}$$

Provided the underlying assumptions hold—unbiased, independent Gaussian measurements with accurately calibrated variances—the fused variance satisfies $\sigma_{\text{fused}}^2 < \min(\sigma_1^2, \sigma_2^2)$. Ignoring cross-correlation or mis-specifying variances breaks that guarantee. Notice that this inverse-variance weighting is simply a scalar Kalman measurement update for an independent observation.

**Representative example: multi-LiDAR systems**

Mounting multiple LiDAR units around a vehicle widens total coverage and creates redundant observations in overlapping zones. In practice, sensor count and placement balance detection range, blind spots, hardware cost, and redundancy requirements.

### Cooperative Fusion

Cooperative fusion synthesizes **new information** from raw multi-sensor data that no single sensor could generate on its own.

**Representative example: Stereo Vision**

- Two horizontally offset cameras image the same scene simultaneously, allowing pixel disparity to be matched across views.
- Triangulating that disparity recovers dense 3D depth.
- Neither camera can resolve depth from a single frame alone, yet their cooperation creates a new geometric quantity.

**Representative example: Camera + LiDAR → colored point cloud**

- Accurate 3D LiDAR points are projected onto synchronized camera frames to sample RGB values.
- The resulting colored point cloud unites LiDAR's metric geometry with the camera's visual texture—an output unattainable from either sensor individually.
- Real-time systems such as [R3LIVE (Lin et al., 2022)](https://arxiv.org/abs/2109.07982) maintain and update such dense colored maps online.

**Representative example: Radar + Camera → adverse-weather perception**

- Fusing radar Doppler measurements with camera imagery enables simultaneous object classification and radial velocity estimation even in heavy fog or rain.

These three categories are not mutually exclusive; production systems routinely weave all three together. A self-driving stack, for instance, may combine cooperative camera–LiDAR fusion for colored mapping, complementary camera–IMU fusion for VIO, and competitive multi-LiDAR coverage for redundancy.

---

## 1.3 Classification by Coupling Level: Loosely vs Tightly vs Ultra-tightly Coupled

Beyond *what* physical quantities sensors observe, fusion architectures differ in *where* their data meets inside the estimation pipeline. This coupling depth directly governs accuracy, engineering complexity, and failure behavior.

### Loosely Coupled

In a loosely coupled architecture, each sensor runs its own self-contained estimator and only high-level outputs—such as poses or velocities—are merged downstream.

**Structure:**
```
Sensor A → [Subsystem A] → Estimate A ─┐
                                        ├─→ [Fusion] → Final estimate
Sensor B → [Subsystem B] → Estimate B ─┘
```

**Representative example: Independent VO + independent LiDAR odometry → EKF fusion**

- Visual odometry produces an independent pose stream.
- LiDAR odometry produces a separate pose stream.
- A downstream EKF blends both pose outputs into a unified trajectory estimate.

**Advantages:**
- **Modularity**: individual subsystems can be built, validated, and swapped out independently.
- **Simplicity**: the top-level fusion filter remains lightweight and easy to reason about.
- **Partial failure isolation**: if one pipeline stalls or fails, the fusion node can fall back on the surviving subsystems.

**Disadvantages:**
- **Information loss**: compressing raw measurements into summary poses discards fine-grained detail, such as per-feature uncertainties.
- **Double counting**: when separate subsystems ingest the same upstream signal (for example, shared IMU readings), their outputs become correlated; ignoring that correlation causes overconfident covariance estimates.
- **Suboptimality**: unless intermediate outputs form sufficient statistics or carry full cross-covariances, accuracy lags behind an estimator that processes raw measurements jointly.

### Tightly Coupled

Tightly coupled systems bypass intermediate pose estimates altogether, feeding **raw measurements** from every sensor into one unified estimator.

**Structure:**
```
Sensor A → raw measurement A ─┐
                                ├─→ [Single Estimator] → Final estimate
Sensor B → raw measurement B ─┘
```

**Representative example: [VINS-Mono (Qin et al., 2018)](https://arxiv.org/abs/1708.03852)**

- Raw 2D feature tracks from the camera and raw inertial measurements from the IMU enter a single sliding-window nonlinear optimizer.
- The joint objective minimizes visual reprojection residuals alongside IMU preintegration residuals:

$$\min_{\mathcal{X}} \left\{ \sum_{(i,j) \in \mathcal{B}} \| \mathbf{r}_{\text{IMU}}(\mathbf{x}_i, \mathbf{x}_j) \|^2_{\mathbf{P}_{ij}} + \sum_{(i,l) \in \mathcal{C}} \| \mathbf{r}_{\text{cam}}(\mathbf{x}_i, \mathbf{f}_l) \|^2_{\mathbf{\Sigma}_l} \right\}$$

where $\mathbf{r}_{\text{IMU}}$ is the IMU preintegration residual, $\mathbf{r}_{\text{cam}}$ is the visual reprojection residual, and $\mathcal{X}$ collects the full state (poses, velocities, IMU biases, and 3D landmarks).

**Representative example: [FAST-LIO2 (Xu et al., 2022)](https://arxiv.org/abs/2107.06829)**

- Raw LiDAR points and IMU measurements are fused directly inside a single iterated error-state Kalman filter.

**Representative example: [LIO-SAM (Shan et al., 2020)](https://arxiv.org/abs/2007.00258)**

- LiDAR feature constraints, IMU preintegration factors, and GNSS position factors are jointly optimized in one factor graph.

**Advantages:**
- **Information preservation**: retaining raw residuals and individual measurement covariances avoids premature compression loss, though final accuracy still hinges on modeling fidelity, linearization points, and outlier rejection.
- **Cross-state calibration**: when extrinsics, time offsets, or sensor biases are augmented into the state under sufficient motion excitation, visual or LiDAR residuals directly help refine IMU biases and calibration parameters.
- **Graceful degradation potential**: when one modality temporarily drops out, remaining sensors can still constrain shared states—provided the estimator incorporates explicit failure detection and observability checks rather than relying on tight coupling alone.

**Disadvantages:**
- **Implementation complexity**: every sensor's observation and noise model must live inside one codebase, complicating design and debugging.
- **Computational load**: larger state vectors and dense raw measurement sets increase runtime demands.
- **Rigid coupling**: adding, removing, or swapping a sensor typically requires modifying the core estimator.

### Ultra-tightly Coupled

Ultra-tightly coupled (or deeply coupled) fusion pushes integration down to the **signal level**, a paradigm used primarily in GNSS/INS navigation.

**Representative example: GNSS/INS ultra-tight integration**

- Whereas tightly coupled GNSS/INS feeds decoded pseudoranges from the receiver into a navigation filter, ultra-tight integration feeds the INS's predicted Doppler and velocity back into the receiver's internal code and carrier tracking loops.
- Knowing platform dynamics externally allows the receiver to narrow its tracking-loop bandwidth, rejecting noise and maintaining satellite lock under heavy interference or weak signals.

**Analogous concepts in vision:**

In visual-inertial systems, the closest parallel is using IMU predictions to restrict feature-matching search windows or to compensate motion blur directly at the image level. By contrast, VINS-Mono does not fall into this category: its front end tracks keypoints with standard KLT optical flow, while IMU measurements enter through preintegration in the back end and initialization.

### Comparison of Coupling Levels

The following table contrasts the trade-offs across all three coupling architectures.

| Property | Loosely Coupled | Tightly Coupled | Ultra-tightly Coupled |
|------|----------------|-----------------|----------------------|
| Fusion level | Output | Measurement | Signal |
| Information utilization | Low | High | Highest |
| Implementation complexity | Low | Medium–High | Very high |
| Modularity | High | Low | Very low |
| Partial failure handling | Easy | Requires design | Difficult |
| Representative systems | Independent VO + LO → EKF | VINS-Mono, LIO-SAM, FAST-LIO2, ORB-SLAM3 | GNSS/INS deep integration |

Most open-source VIO and LIO pipelines adopt a **tightly coupled** design. Loosely coupled architectures remain attractive when rapid modularity matters, since swapping subsystems is straightforward despite the information lost to intermediate summaries. Ultra-tight coupling demands low-level access to internal receiver loops or sensor hardware, limiting its use to specialized domains. Pushing tight coupling across three modalities, [FAST-LIVO2 (Zheng et al., 2024)](https://arxiv.org/abs/2408.14035) unifies LiDAR, IMU, and camera measurements inside one framework and reports gains in both accuracy and throughput under its benchmark evaluation protocol.

---

## 1.4 Classical vs Learning-based: What Deep Learning Changed and What It Did Not

For decades, sensor fusion rested on **classical** probabilistic estimation (Kalman filters, factor graphs) paired with geometric algorithms (epipolar geometry, ICP). Since the mid-2010s, deep learning has transformed parts of that stack—especially front-end perception—while leaving the mathematical core of state estimation surprisingly intact.

### What Deep Learning Changed

**Feature extraction and matching.** Classical pipelines relied on handcrafted descriptors such as SIFT and ORB. [SuperPoint (DeTone et al., 2018)](https://arxiv.org/abs/1712.07629) learns keypoint detection and description jointly via self-supervision, holding up far better across lighting and viewpoint shifts. Building on learned keypoints, [SuperGlue (Sarlin et al., 2020)](https://arxiv.org/abs/1911.11763) uses graph neural networks (GNNs) and attention to solve feature matching with fewer mismatches than nearest-neighbor search on handcrafted descriptors. **Detector-free** matchers such as [LoFTR (Sun et al., 2021)](https://arxiv.org/abs/2104.00680) and [RoMa (Edstedt et al., 2024)](https://arxiv.org/abs/2305.15404) dispense with explicit keypoint detection altogether, establishing dense correspondences even across low-texture surfaces. While learned matchers outperform classical baselines on many public benchmarks, real-world deployment still demands evaluating inference latency, memory footprint, and domain shift.

**Place Recognition.** [NetVLAD (Arandjelović et al., 2016)](https://arxiv.org/abs/1511.07247) replaces hand-engineered bag-of-words pooling with a differentiable VLAD layer over CNN features, achieving stronger recall on evaluated image-retrieval benchmarks. More recently, [AnyLoc (Keetha et al., 2023)](https://arxiv.org/abs/2308.00688) pairs foundation-model DINOv2 features with unsupervised VLAD across diverse environments. Because learned descriptors differ sharply from DBoW2 in both input representation and compute cost, fair comparisons require testing them within the same dataset and geometric-verification pipeline.

**Monocular depth estimation.** Although single-view projective geometry cannot resolve absolute scale, data-driven priors can infer plausible depth cues from context alone. [Depth Anything (Yang et al., 2024)](https://arxiv.org/abs/2401.10891) learns robust relative depth from massive training sets; [Depth Anything V2 (Yang et al., 2024)](https://arxiv.org/abs/2406.09414) sharpens fine detail using synthetic supervision and large-scale pseudo-labeling; and [Metric3D v2 (Hu et al., 2024)](https://arxiv.org/abs/2404.15506) targets zero-shot metric depth prediction. Even so, before feeding learned depth into a fusion filter or optimizer, practitioners must characterize its scale bias and uncertainty on the target camera and operating domain.

**Map representations.** Neural Radiance Fields (NeRF) encode scenes implicitly in neural network weights, whereas 3D Gaussian Splatting models scenes as collections of optimized 3D Gaussians. NeRF-SLAM and Gaussian Splatting SLAM bring photorealistic rendering directly into mapping, offering an expressive alternative to traditional sparse point clouds and voxel grids.

**Event cameras.** Alongside deep learning, neuromorphic event cameras have matured as an alternative sensing modality. Rather than capturing full frames at fixed intervals, each pixel asynchronously reports brightness changes with microsecond resolution and wide dynamic range. As surveyed in [event camera survey (Chakravarthi et al., 2024)](https://arxiv.org/abs/2408.13627), event-based VIO and SLAM fuse event streams with conventional frame cameras to survive extreme high-speed motion and low-light conditions.

### What Deep Learning Did Not Change

**State estimation backends.** Probabilistic frameworks such as Kalman filters and factor graph optimization remain the backbone of multi-sensor estimation for four enduring reasons:

1. **Explicit uncertainty propagation**: Kalman filters and factor graphs propagate state covariances systematically under assumed noise models. Consistency still requires guarding against model mismatch, linearization errors, and unmodeled correlations—and learned front-end observations need explicit uncertainty calibration—but the mathematical machinery for fusing heterogeneous uncertainties stays intact.
2. **Physical models and constraints**: rigid-body kinematics and inertial dynamics can be written directly into state transitions and residual factors. While soft priors alone cannot prevent every unphysical estimate, hard physical constraints can be enforced cleanly through manifold parameterizations and constrained optimization.
3. **Training data vs. physical calibration**: classical estimators do not require massive supervised datasets to operate, though they still depend on accurate calibration, noise tuning, and validation logs. Learned modules add the burden of curating training distributions and managing supervision assumptions.
4. **Generalization**: end-to-end learned odometry often degrades when deployed outside its training domain. Geometric pipelines are hardly immune to poor texture, harsh lighting, dynamic clutter, or sensor noise, yet their failure modes are interpretable and they deploy to new platforms without retraining.

**LiDAR odometry.** Since [LOAM (Zhang & Singh, 2014)](https://frc.ri.cmu.edu/~zhangji/publications/RSS_2014.pdf), geometric point-cloud registration—including ICP, GICP, and NDT—has remained the standard workhorse for LiDAR odometry. Its mathematical models and degeneracy conditions (such as featureless tunnels) are well understood, and it transfers across sensors without retraining. While learned LiDAR odometry shows competitive numbers on specific benchmarks, its robustness across new LiDAR geometries and environments still requires separate validation.

**Calibration.** The planar-checkerboard formulation of [Zhang (2000)](https://doi.org/10.1109/34.888718) remains the standard starting point for camera intrinsic calibration. Although targetless and learning-based calibration methods are active research frontiers, practitioners still benchmark their precision and parameter observability directly against classical target-based protocols.

### Hybrid Approach

Rather than choosing between classical and learned paradigms, modern systems commonly adopt a **hybrid** architecture that pairs a learned front end with a classical back end:

```
[Learning-based frontend]         [Classical backend]
 SuperPoint/SuperGlue          Factor Graph / EKF
 Mono Depth Estimation    →    Nonlinear Optimization
 Semantic Segmentation         Kalman Filtering
 Place Recognition              Pose Graph SLAM
```

- **At the front end**, neural networks extract resilient visual quantities—keypoints, correspondences, monocular depth priors, or semantic masks—from raw sensor streams.
- **At the back end**, geometric and probabilistic estimators enforce multi-view consistency, propagate uncertainty, and fuse inertial or ranging measurements.

[DROID-SLAM (Teed & Deng, 2021)](https://arxiv.org/abs/2108.10869) exemplifies this synthesis: it predicts dense optical flow and confidence weights with a neural network, then solves for camera poses and depth via differentiable bundle adjustment.

### Technical Lineage Summary

Throughout this guide, each topic follows a consistent progression: **classical foundations → what deep learning changed → where classical rigor remains essential**. The table below maps that landscape across core fusion subtasks.

| Area | Classical | Learning-based | Relationship in practice |
|------|-----------|---------------|----------|
| Feature matching | SIFT/ORB + BF/FLANN | SuperPoint+SuperGlue → LoFTR → RoMa | Hybrid |
| Visual odometry | Feature-based (ORB) / Direct (DSO) | DROID-SLAM, DPV-SLAM | Choice depends on accuracy, compute, and domain |
| LiDAR odometry | ICP/LOAM | DeepLO-family | Geometry-centered, sometimes with learned components |
| Place recognition | BoW/VLAD | NetVLAD → AnyLoc | Mixed retrieval frontends, followed by geometric verification |
| Depth estimation | Stereo matching | Mono depth (Depth Anything) | Separate roles for metric stereo and learned priors |
| Calibration | Target-based | Targetless + learning | Select target or targetless methods by requirements |
| Map representation | Occupancy/TSDF | NeRF / 3DGS | Coexistence |
| State estimation backend | KF / Factor Graph | End-to-end attempts | Model-based estimator with learned observations |

In practice, learned modules increasingly power perception front ends while model-based estimators bind geometry and uncertainty together at the back end. Rather than drawing a rigid boundary, this split helps engineers cleanly separate each module's inputs, failure modes, and validation metrics.

---

## 1.5 Scope and Organization

### Relation to robotics-practice

While robotics-practice surveys Spatial AI broadly, this guide goes deep into **sensor fusion, localization, and retrieval**.

- Where robotics-practice summarizes EKF and particle filters in one or two pages, this guide develops full mathematical derivations from the Kalman filter through the error-state Kalman filter (ESKF), IMU preintegration, and factor graph optimization (→ Chapter 4).
- Where robotics-practice introduces sensors conceptually, this guide derives each sensor's **noise model and observation equations** explicitly in mathematical and executable form.
- Introductory material already covered in robotics-practice is cross-referenced rather than repeated, keeping the focus on analytical depth and implementation detail.

### Target Audience

This guide is written for robotics engineers and researchers with working familiarity in four areas:

- **Linear algebra**: matrix operations, eigendecomposition, and singular value decomposition (SVD)
- **Probability**: probability distributions, conditional probability, Bayes' theorem, and multivariate Gaussians
- **Basic optimization**: least squares and gradient descent
- **Python**: reading and running numerical code built on NumPy and SciPy

Every major topic moves from physical intuition to equations, then verifies the math with runnable Python code and concrete numerical checks. Next, Chapter 2 puts this approach to work by formulating the observation and noise models for each sensor modality—the measurement functions that any fusion filter or optimizer relies on.
