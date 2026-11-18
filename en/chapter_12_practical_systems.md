# Chapter 12. Practical Systems and Benchmarks

The preceding chapters built the core pieces of a spatial perception stack—from physical sensor models and calibration to state estimation, odometry, place recognition, loop closure, and the hierarchical spatial representations of Chapter 11. On a physical robot, however, those pieces can no longer be tuned in isolation. Weight, power, compute, latency, and safety margins force hard architectural trade-offs across the entire pipeline.

This chapter examines how complete systems bring these components together on three representative platforms—autonomous vehicles, drones, and handheld or backpack mapping rigs—before turning to the standardized benchmarks, trajectory metrics (ATE and RPE), and open-source tools needed to evaluate and reproduce them fairly.

---

## 12.1 Autonomous Driving Perception Stack

On public roads, high vehicle speeds and strict safety requirements leave no margin for single-point failures. An autonomous-driving safety architecture must detect single-sensor faults and bound the resulting risk: whether the vehicle continues operating or transitions to a minimal-risk condition depends on its Operational Design Domain (ODD) and safety case, and combining complementary sensors provides the redundancy and cross-checks that make such guarantees possible.

### 12.1.1 Sensor Suite Configuration Examples

Different platforms balance redundancy, field of view, and hardware cost in markedly different ways, as illustrated by three widely cited configurations:

**Waymo** (5th generation):
- 1 × long-range LiDAR (360°, up to 300 m)
- 4 × short-range LiDAR (covering near-field blind spots)
- 29 × cameras (360° coverage, varied fields of view)
- 6 × radar (long-range velocity measurement)
- IMU, GNSS, wheel encoder

**[nuScenes](https://arxiv.org/abs/1903.11027) collection vehicle**:
- 1 × 32-beam spinning LiDAR
- 6 × surround cameras (360° coverage)
- 5 × radar
- IMU, GNSS

**Camera-centric production example**:
- Some production systems anchor perception around multiple surround cameras and neural networks rather than LiDAR.
- Because sensor suites and radar fitments change across vehicle generations, regional markets, and model years, no manufacturer should be generalized as having one fixed "vision-only" configuration.

### 12.1.2 Production-Level Fusion Pipeline

Once sensor streams are hardware-synchronized, a production autonomous-driving stack routes each modality through specialized preprocessing before fusing them for downstream perception and planning. The diagram below sketches a conceptual dataflow rather than a rigid dependency graph: localization, in particular, does not sit downstream of tracking and prediction, but runs in parallel from GNSS, IMU, LiDAR, and wheel measurements so its state estimates feed perception, prediction, and planning alike.

```
Sensor synchronization (HW trigger + PTP)
    ↓
[LiDAR processing]     [Camera processing]   [Radar processing]
- Motion compensation  - Object detection    - Doppler velocity
- Ground segmentation  - Semantic seg        - Long-range detection
- 3D object detection  - Lane detection
    ↓                      ↓                     ↓
         Late Fusion / Deep Fusion
    ↓
[Tracking & Prediction]
- Multi-object tracking (Kalman/JPDA)
- Trajectory prediction
    ↓
[Localization]
- HD map matching
- GNSS/IMU/LiDAR integration
    ↓
[Planning & Control]
```

A central architectural choice in this pipeline is **where** modalities meet—specifically, whether to combine final detections or intermediate features:

- **Late fusion**: Each modality independently proposes 3D bounding boxes, which are merged afterward via Non-Maximum Suppression (NMS). While this modular boundary is easy to inspect and debug, it discards low-level cross-sensor cues before fusion even begins.

- **Deep fusion**: Modalities interact at the learned feature level, though architectures differ in where that interaction occurs. [BEVFusion](https://arxiv.org/abs/2205.13542) (MIT) projects camera and LiDAR features into a shared Bird's Eye View (BEV) plane and combines them there, whereas TransFusion initializes object queries from LiDAR BEV features and cross-attends to image features inside a transformer decoder. Both strategies exploit cross-sensor complementarity more effectively than late fusion, at the cost of requiring large-scale labeled multi-sensor datasets for end-to-end training.

To make BEV feature alignment concrete, the following conceptual implementation illustrates how camera features are lifted along predicted depth distributions and splatted onto the same top-down grid as voxelized LiDAR features:

```python
# BEV Fusion conceptual diagram (pseudo-code)

def bev_fusion_pipeline(lidar_points, camera_images, calibrations):
    """
    Fusion pipeline that combines LiDAR and camera features in BEV space.
    """
    # 1. LiDAR → BEV feature
    # Voxelize the LiDAR points, process them with a 3D backbone, and
    # collapse the z axis to produce a BEV feature map
    lidar_voxels = voxelize(lidar_points, voxel_size=0.1)
    lidar_3d_features = sparse_3d_cnn(lidar_voxels)  # (X, Y, Z, C)
    lidar_bev = lidar_3d_features.max(dim='z')        # (X, Y, C)
    
    # 2. Camera → BEV feature
    # Extract features from each camera image and transform them
    # into BEV space via depth estimation
    camera_features = []
    for img, calib in zip(camera_images, calibrations):
        feat_2d = image_backbone(img)  # (H', W', C)
        depth_dist = depth_net(feat_2d)  # (H', W', D) — depth probability distribution
        
        # Lift: 2D feature to 3D (LSS scheme)
        feat_3d = outer_product(feat_2d, depth_dist)  # (H', W', D, C)
        
        # Splat: aggregate 3D features into BEV pillars
        feat_bev = splat_to_bev(feat_3d, calib)  # (X, Y, C)
        camera_features.append(feat_bev)
    
    camera_bev = sum(camera_features)  # sum BEV features across all cameras
    
    # 3. Fusion: concatenate or attention in BEV space
    fused_bev = concat_and_conv(lidar_bev, camera_bev)  # (X, Y, C')
    
    # 4. Detection head
    detections = detection_head(fused_bev)  # 3D bounding boxes
    
    return detections
```

### 12.1.3 Localization Stack

Because urban canyons introduce severe GNSS multipath errors of several meters while local scan matching can fall into wrong local minima without a coarse prior, autonomous driving typically splits localization into two complementary layers:

1. **Global localization**: RTK or PPP GNSS supplies a coarse pose estimate on the map, anchoring the vehicle globally even though multipath limits its standalone reliability in dense cities.

2. **Map-relative localization**: Current LiDAR scans are registered against an HD point-cloud map via NDT or ICP. Given accurate prior maps, sufficient overlap, distinct geometry, and reliable initialization, scan matching locks down position where GNSS degrades—though claiming centimeter-level accuracy still requires validation under the specific map and evaluation protocol.

A factor-graph backend unifies these asynchronous measurements with high-rate inertial prediction and wheel odometry:

$$\mathbf{x}^* = \arg\min \underbrace{f_{\text{IMU}}}_{\text{prediction}} + \underbrace{f_{\text{LiDAR}}}_{\text{map matching}} + \underbrace{f_{\text{GNSS}}}_{\text{global anchor}} + \underbrace{f_{\text{wheel}}}_{\text{velocity}}$$

---

## 12.2 Drones/UAVs

Unlike passenger vehicles that carry generous battery capacity, payload margins, and multi-GPU compute, aerial platforms operate under tight budgets for mass, power, and aerodynamic drag while executing agile 6-DoF maneuvers. Sensor fusion on drones must adapt to those constraints from the ground up.

### 12.2.1 Visual-Inertial-Centric Systems

Because traditional spinning LiDARs are too heavy and power-hungry for small multirotors, most drones rely on a camera–IMU pair, though compact solid-state LiDARs such as the Livox Mid-360 are beginning to broaden what small airframes can carry. On any rotorcraft, high-frequency propeller vibration injects substantial noise into inertial measurements, making mechanical vibration-isolating mounts and software filtering equally essential.

Three visual-inertial odometry (VIO) architectures see wide deployment on drones, trading off accuracy against onboard compute:
- **VINS-Mono/Fusion**: Tightly coupled nonlinear optimization that integrates smoothly with flight stacks such as PX4.
- **MSCKF/OpenVINS**: Filter-based estimation whose low computational footprint suits resource-constrained embedded boards.
- **Basalt**: Visual-inertial mapping built around nonlinear factor recovery for consistent long-term trajectory estimation.

### 12.2.2 GPS-Denied Navigation

Autonomous flight indoors, inside tunnels, beneath forest canopy, or under electronic-warfare interference deprives a drone of reliable satellite positioning. How the system bounds drift depends on what prior maps or external infrastructure are available:

1. **VIO alone**: Without external anchors, drift grows over time; the allowable mission duration therefore follows directly from motion dynamics, scene texture, calibration accuracy, loop-closure opportunities, and the vehicle's error budget.
2. **VIO + terrain matching**: Visual observations are matched against a pre-built terrain or building map, which eliminates unbounded drift at the cost of requiring an up-to-date prior map.
3. **VIO + UWB**: Ultra-wideband anchors placed in the workspace provide range measurements that bound position drift, provided the environment can be instrumented beforehand.
4. **VIO + barometer**: Air-pressure changes supply a useful relative-altitude cue, but weather shifts, ambient temperature, and propeller downwash introduce time-varying bias that prevents barometry alone from removing absolute z-axis drift.

### 12.2.3 Real-Time Constraints

A drone's sensor rates and latency budget cannot be picked from rules of thumb; they follow from peak angular rates and linear accelerations, flight-controller bandwidth, exposure time, estimator runtime, and telemetry delay. The IMU sampling rate must be high enough to avoid aliasing and integration error under aggressive maneuvers, while camera exposure must balance allowable pixel motion against low-light sensor noise. Likewise, the end-to-end state-estimation latency budget must be derived from closed-loop stability analysis and flight testing rather than assuming a universal figure such as 30 ms.

For high-agility flight with LiDAR, **Point-LIO** lowers latency by updating the state on individual LiDAR points as they arrive instead of waiting for a full scan sweep to finish.

The configuration and quality-gating logic below illustrate how these constraints appear in practice—pairing higher IMU noise densities to reflect airframe vibration with feature-count caps and motion-blur rejection so corrupted frames never enter the estimator:

```python
class DroneVIOConfig:
    """Example VIO system configuration for drones."""
    
    # Sensor configuration
    camera_fps = 30
    camera_resolution = (640, 480)  # low resolution to reduce compute
    imu_rate = 400  # Hz
    
    # VIO parameters
    max_features = 150  # cap on number of features (compute)
    keyframe_interval = 5  # one keyframe every 5 frames
    sliding_window_size = 10  # optimization window size
    
    # Drone-specific settings
    gravity_magnitude = 9.81
    max_angular_velocity = 10.0  # rad/s — handle aggressive rotations
    motion_blur_threshold = 0.3  # drop heavily blurred frames
    
    # IMU noise (large values since drones vibrate heavily)
    gyro_noise_density = 0.004  # rad/s/sqrt(Hz)
    accel_noise_density = 0.05  # m/s^2/sqrt(Hz)
    gyro_random_walk = 0.0002   # rad/s^2/sqrt(Hz)
    accel_random_walk = 0.003   # m/s^3/sqrt(Hz)
    
    # Safety
    max_allowed_drift_m = 0.5   # warn if drift exceeds this value
    min_tracked_features = 20   # warn about tracking quality if below


def check_image_quality(image, angular_velocity, exposure_time):
    """
    Drone camera image quality check.
    Exclude heavily motion-blurred frames from VIO.
    """
    # Motion-blur estimate: angular velocity × exposure time × focal length
    blur_pixels = abs(angular_velocity) * exposure_time * 300  # approximate focal length
    
    if blur_pixels > 5.0:  # blur exceeds 5 pixels
        return False, f"Excessive motion blur: {blur_pixels:.1f} pixels"
    
    # Brightness check
    mean_brightness = image.mean()
    if mean_brightness < 20:
        return False, f"Too dark: mean={mean_brightness:.0f}"
    if mean_brightness > 240:
        return False, f"Too bright: mean={mean_brightness:.0f}"
    
    return True, "OK"
```

---

## 12.3 Handheld/Backpack Mapping

Compared with micro aerial vehicles, handheld and backpack-mounted rigs carry more weight and battery power, yet they operate in a different regime altogether—walking through GPS-denied buildings, stairwells, and industrial plants for surveying, Building Information Modeling (BIM), and digital-twin reconstruction.

### 12.3.1 SLAM as a Service

Commercial mobile mapping platforms package multi-sensor SLAM into turnkey workflows, though their sensor mounts and scanning mechanisms differ considerably:

- **Leica BLK2GO**: A handheld scanner performing real-time SLAM by fusing LiDAR, IMU, and camera streams. Manufacturer accuracy specifications must be read alongside their stated test conditions (scan length, whether loops were closed, and validation methodology) and checked separately against the project criteria in 12.3.2 below.
- **NavVis VLX**: A wearable backpack rig combining four cameras and LiDAR, tailored to rapid indoor architectural mapping.
- **GeoSLAM ZEB**: A handheld mobile mapper whose scanning mechanism evolved across hardware generations—early units mounted a 2D LiDAR on a passive spring so walking motion swept the beam, whereas models from ZEB-REVO onward drive the head with a motor.

Although none of these commercial products publishes its proprietary internal code, the conceptual pipeline below maps their workflow onto open-source methods that fill equivalent functional roles:

```
LiDAR + IMU → LIO (FAST-LIO2 or similar)
     ↓
Loop closure (Scan Context, etc.)
     ↓
Global optimization (iSAM2)
     ↓
Dense colorized point cloud (camera color overlaid)
     ↓
Post-processing (cloud cleanup, mesh generation)
```

### 12.3.2 Survey-Grade Mapping

Delivering survey-grade maps requires anchoring system design to explicit contract specifications and independent validation procedures:

- **Absolute accuracy**: Horizontal and vertical error bounds and confidence levels must be specified upfront and verified against independent checkpoints using GNSS, a total station, or ground control points (GCPs) as site conditions dictate.
- **Relative accuracy**: Loop closure and global pose-graph optimization preserve local geometric consistency across the reconstructed map.
- **Point density**: Point spacing and completeness follow from minimum target feature size and deliverable requirements; a 1 cm spacing threshold is a project-specific target rather than a universal survey standard.
- **Color quality**: Accurate HDR colorization demands tight hardware time synchronization and precise camera–LiDAR extrinsic calibration.

Even with high-grade sensors, field deployments regularly confront three failure modes that single-modality pipelines struggle to survive:

1. **Degenerate environments**: Featureless corridors and empty rooms starve LiDAR scan matching along the travel direction. Tightly coupled multi-modal systems such as R3LIVE and FAST-LIVO2 counter this drift using camera photometric constraints and inertial preintegration.
2. **Multi-story buildings**: Without GNSS, vertical (z-axis) drift accumulates as an operator climbs between floors. Revisiting the same floor allows loop closure to collapse the error, whereas single-pass routes with no loops require external vertical constraints such as barometric cues, floor-height or floor-plan priors, or surveyed control points.
3. **Glass and mirrors**: Transparent panes transmit laser pulses while mirrors produce virtual ghost geometry behind walls, forcing systems to fuse visual cues or filter out specular reflections.

---

## 12.4 Benchmarks & Evaluation

Because architectural claims mean little without reproducible comparisons across shared sequences, standardized datasets and clearly defined error metrics form the backbone of sensor-fusion engineering.

### 12.4.1 Major Datasets

The table below summarizes eight widely used benchmarks spanning autonomous driving, aerial flight, construction sites, and handheld campus mapping:

| Dataset | Year | Environment | Sensors | Features |
|---------|------|-------------|---------|----------|
| **[KITTI](https://doi.org/10.1177/0278364913491297)** | 2012 | Outdoor (autonomous driving) | Stereo, LiDAR, GPS/IMU | 11 training + 11 test sequences for odometry evaluation |
| **[EuRoC](https://doi.org/10.1177/0278364915620033)** | 2016 | Indoor (MAV) | Stereo, IMU | VIO sequences in Machine Hall + Vicon Room |
| **[TUM-RGBD](https://doi.org/10.1109/IROS.2012.6385773)** | 2012 | Indoor | RGB-D | Kinect v1 visual-SLAM sequences and evaluation tools |
| **TUM-VI** | 2018 | Indoor + outdoor | Stereo, IMU | VIO benchmark. Diverse motion patterns |
| **[Hilti](https://arxiv.org/abs/2109.11316)** | 2021–2023 challenge editions | Construction sites | LiDAR, Camera, IMU | Industrial-environment-specific. Challenging conditions |
| **[HeLiPR](https://arxiv.org/abs/2309.14590)** | 2023 | Outdoor (urban) | Heterogeneous LiDAR, IMU, GNSS/INS | For place recognition research across heterogeneous LiDARs. Ouster+Velodyne+Livox+Aeva. Check whether cameras are included in the per-release sensor list |
| **[nuScenes](https://arxiv.org/abs/1903.11027)** | 2020 | Outdoor (autonomous driving) | Camera, LiDAR, Radar, GPS/IMU | 1000 scenes, 23-class 3D annotations, 360° surround sensors |
| **Newer College** | 2020 | Outdoor (campus) | LiDAR, Camera, IMU | New College grounds, Oxford. Handheld single session, many loops |

Each benchmark tests a distinct operational regime and comes with its own ground-truth caveats:

**KITTI** — Released in 2012, KITTI remains a standard reference for autonomous-driving odometry and SLAM, pairing a Velodyne 64-channel LiDAR with stereo cameras and GPS/IMU. Even so, its sensors are now dated, individual sequences are relatively short, and its GPS/INS reference trajectories do not guarantee centimeter-level accuracy in every segment.

**EuRoC** — Captured on a micro aerial vehicle with stereo cameras and an IMU, EuRoC serves as the classic VIO benchmark. Ground truth comes from either a Vicon motion-capture system or a Leica laser tracker, and its 11 sequences step systematically from easy to medium to difficult motion profiles.

**Hilti** — Designed for active construction sites, Hilti stresses SLAM pipelines with airborne dust, tool vibration, and repetitive unfinished structures. Challenge editions ran in 2021, 2022, and 2023, with sensor configurations and evaluation rules changing across editions.

**HeLiPR** — Released in 2023, HeLiPR mounts spinning, solid-state, and FMCW LiDARs on the same platform simultaneously, enabling research on inter-LiDAR place recognition across heterogeneous scan patterns.

**Newer College** — Collected in a single walking session across New College, Oxford using a handheld rig, this dataset revisits the same courtyards repeatedly and captures aggressive handheld motion, making it well suited to loop-closure testing. Studying long-term appearance shifts across hours of the day or changing seasons, however, calls for multi-session datasets such as Boreas.

Since 2022, newer benchmarks have broadened coverage to millimeter-grade industrial reference trajectories, multi-season driving, and 4D imaging radar:

- **[Hilti-Oxford](https://arxiv.org/abs/2208.09825)** (2022): A construction-environment SLAM benchmark providing millimeter-level ground truth for the 2022 challenge.
- **[Boreas](https://arxiv.org/abs/2203.10168)** (Burnett et al. 2023): An autonomous-driving dataset recorded by driving the same route repeatedly across all four seasons with LiDAR, radar, and cameras.
- **[Snail-Radar](https://arxiv.org/abs/2407.11705)** (Huai et al., IJRR 2025): A multi-platform benchmark evaluating 4D radar odometry and SLAM across diverse environments.

### 12.4.2 Evaluation Metrics

Evaluating a trajectory requires distinguishing global consistency from local drift: a system with small frame-to-frame error can still accumulate large end-to-end distortion without loop closure, while a globally aligned trajectory may hide jittery local estimates. Two complementary trajectory metrics capture these behaviors.

**ATE (Absolute Trajectory Error)** quantifies the global positional discrepancy between the estimated trajectory and ground truth:

$$\text{ATE} = \sqrt{\frac{1}{N} \sum_{i=1}^{N} \| \text{trans}(\mathbf{T}_{\text{gt},i}^{-1} \cdot \mathbf{T}_{\text{est},i}) \|^2}$$

Before computing ATE, the estimated trajectory is usually aligned to ground truth. Crucially, the allowable alignment transform depends on which degrees of freedom are physically unobservable and what the benchmark protocol specifies—not merely on the sensor name. Sim(3) alignment is used when metric scale is unobservable (as in pure monocular VO), whereas SE(3) alignment is used when scale is observable (as in stereo, LiDAR, or monocular–inertial fusion). Other protocols skip Sim(3) alignment entirely, as in the official KITTI odometry evaluation, or measure raw absolute-coordinate error with no post-hoc alignment at all, as in the survey-grade checks discussed earlier. When Sim(3) alignment is needed, the optimal similarity transform

$$\mathbf{S}^* = \arg\min_{\mathbf{S} \in \text{Sim}(3)} \sum_i \| \mathbf{p}_{\text{gt},i} - \mathbf{S} \cdot \mathbf{p}_{\text{est},i} \|^2$$

admits a closed-form solution via the Umeyama algorithm.

**RPE (Relative Pose Error)** instead measures local pose accuracy over fixed intervals, directly exposing odometry drift:

$$\text{RPE}(\Delta) = \sqrt{\frac{1}{M} \sum_{i=1}^{M} \| \text{trans}((\mathbf{T}_{\text{gt},i}^{-1} \mathbf{T}_{\text{gt},i+\Delta})^{-1} (\mathbf{T}_{\text{est},i}^{-1} \mathbf{T}_{\text{est},i+\Delta})) \|^2}$$

Here $\Delta$ denotes a frame-index offset: short $\Delta$ captures local odometry accuracy, while long $\Delta$ reflects accumulated drift. Evaluating RPE over fixed travel distances—such as KITTI's 100–800 m segments—requires first stepping forward from each start frame until accumulated path length reaches the target distance and pairing those two frames.

When a system includes loop closure or relocalization, trajectory metrics are paired with **place recognition metrics**:
- **Recall@N**: The fraction of queries whose true location appears among the top-N retrieved candidates, with Recall@1 serving as the strictest operating point.
- **Precision-Recall curve**: The trade-off between retrieval precision and recall as the acceptance threshold varies.
- **AUC**: The area under the Precision-Recall curve, summarizing retrieval quality across thresholds.

The Python implementation below puts these definitions into code, covering Umeyama alignment in both SE(3) and Sim(3), per-frame ATE, and frame-interval translation and rotation RPE:

```python
import numpy as np
from scipy.spatial.transform import Rotation

def compute_ate(poses_gt, poses_est, align='se3'):
    """
    Compute Absolute Trajectory Error.
    
    Args:
        poses_gt: list of ground truth poses [(4, 4), ...]
        poses_est: list of estimated poses [(4, 4), ...]
        align: 'se3' or 'sim3'
        
    Returns:
        ate_rmse: ATE RMSE (meters)
        ate_per_frame: per-frame ATE (N,)
    """
    positions_gt = np.array([T[:3, 3] for T in poses_gt])  # (N, 3)
    positions_est = np.array([T[:3, 3] for T in poses_est])  # (N, 3)
    
    # Umeyama alignment
    if align == 'sim3':
        S, R, t = umeyama_alignment(positions_est, positions_gt, 
                                     with_scale=True)
        positions_aligned = S * (R @ positions_est.T).T + t
    else:  # se3
        _, R, t = umeyama_alignment(positions_est, positions_gt, 
                                     with_scale=False)
        positions_aligned = (R @ positions_est.T).T + t
    
    # Per-frame error
    errors = np.linalg.norm(positions_gt - positions_aligned, axis=1)
    
    ate_rmse = np.sqrt(np.mean(errors ** 2))
    
    return ate_rmse, errors


def compute_rpe(poses_gt, poses_est, delta=10):
    """
    Compute Relative Pose Error.
    
    Args:
        poses_gt: list of ground truth poses
        poses_est: list of estimated poses
        delta: evaluation interval (in frames)
        
    Returns:
        rpe_trans: RPE translation RMSE (meters)
        rpe_rot: RPE rotation RMSE (degrees)
    """
    trans_errors = []
    rot_errors = []
    
    for i in range(len(poses_gt) - delta):
        # Ground truth relative transform
        T_gt_rel = np.linalg.inv(poses_gt[i]) @ poses_gt[i + delta]
        
        # Estimated relative transform
        T_est_rel = np.linalg.inv(poses_est[i]) @ poses_est[i + delta]
        
        # Error
        T_error = np.linalg.inv(T_gt_rel) @ T_est_rel
        
        # Translation error
        trans_err = np.linalg.norm(T_error[:3, 3])
        trans_errors.append(trans_err)
        
        # Rotation error
        rot = Rotation.from_matrix(T_error[:3, :3])
        rot_err = np.linalg.norm(rot.as_rotvec()) * 180 / np.pi  # in degrees
        rot_errors.append(rot_err)
    
    rpe_trans = np.sqrt(np.mean(np.array(trans_errors) ** 2))
    rpe_rot = np.sqrt(np.mean(np.array(rot_errors) ** 2))
    
    return rpe_trans, rpe_rot


def umeyama_alignment(source, target, with_scale=True):
    """
    Umeyama alignment: compute the optimal similarity/rigid
    transform that aligns source to target.
    
    Args:
        source: (N, 3) source points
        target: (N, 3) target points
        with_scale: True for Sim(3), False for SE(3)
        
    Returns:
        scale: scale (1.0 if with_scale=False)
        rotation: (3, 3) rotation matrix
        translation: (3,) translation vector
    """
    n = source.shape[0]
    
    # Recenter
    mu_source = np.mean(source, axis=0)
    mu_target = np.mean(target, axis=0)
    
    source_centered = source - mu_source
    target_centered = target - mu_target
    
    # Covariance matrix
    sigma_source = np.sum(source_centered ** 2) / n
    H = (target_centered.T @ source_centered) / n
    
    # SVD
    U, D, Vt = np.linalg.svd(H)
    
    # Reflection correction
    d = np.linalg.det(U) * np.linalg.det(Vt)
    S = np.diag([1, 1, np.sign(d)])
    
    rotation = U @ S @ Vt
    
    if with_scale:
        scale = np.trace(np.diag(D) @ S) / sigma_source
    else:
        scale = 1.0
    
    translation = mu_target - scale * rotation @ mu_source
    
    return scale, rotation, translation
```

### 12.4.3 Difficulties of Fair Comparison

Even with shared datasets and identical metric formulas, headline leaderboard numbers can easily mislead if five reporting pitfalls go unchecked:

1. **Parameter tuning**: An algorithm's accuracy shifts dramatically with feature thresholds, keyframe spacing, or noise covariances. A score obtained after tuning specifically for one dataset reflects that tuned configuration rather than out-of-the-box generalization, so tuning protocols must be reported alongside the numbers.
2. **Hardware dependence**: "Real-time" means very different things on a desktop GPU versus an embedded ARM board; runtime claims are meaningless without hardware and thread-count context.
3. **Completeness**: A system that loses tracking on hard segments and computes ATE only over surviving frames hides its failure rate. **Completeness**—the fraction of sequences or frames successfully tracked—must be reported alongside trajectory error.
4. **Initialization differences**: In VIO systems, small differences in initialization strategy or stationary warm-up duration can produce noticeably different errors on the exact same sequence.
5. **Whether loop closure is included**: Results with loop closure enabled must never be compared directly against pure odometry, because closing even one loop can cut global ATE dramatically. Naming conventions are not reliable either: DSO is a pure VO system without loop closure, whereas LDSO adds one, and ORB-SLAM3 can be executed with loop closure disabled—so always verify the actual runtime configuration.

---

## 12.5 Open-Source Tool Guide

Building and evaluating a sensor-fusion pipeline rarely starts from scratch; a mature ecosystem of open-source solvers, calibration suites, trajectory evaluators, and middleware underpins both academic baselines and deployed systems.

### 12.5.1 Optimization Libraries

Three nonlinear optimization libraries dominate SLAM backends, differing primarily in how they structure variables and constraints:

**GTSAM** (Georgia Tech Smoothing and Mapping) provides a factor-graph abstraction with built-in iSAM2 incremental smoothing, C++ speed with mature Python bindings (`gtsam`), and rich domain-specific factors that power systems such as LIO-SAM. Its workflow centers on attaching prior, odometry, and loop-closure factors directly to pose nodes, as shown in the minimal 3-pose example below:

```python
# GTSAM Python usage example (conceptual code)

import gtsam
import numpy as np

def simple_pose_graph_gtsam():
    """Simple pose graph optimization with GTSAM."""
    
    # Create the factor graph
    graph = gtsam.NonlinearFactorGraph()
    
    # Initial values
    initial = gtsam.Values()
    
    # Prior factor: fix the first pose
    prior_noise = gtsam.noiseModel.Diagonal.Sigmas(
        np.array([0.01, 0.01, 0.01, 0.01, 0.01, 0.01])  # (rx, ry, rz, tx, ty, tz)
    )
    graph.add(gtsam.PriorFactorPose3(
        0, gtsam.Pose3(), prior_noise
    ))
    initial.insert(0, gtsam.Pose3())
    
    # Odometry factors
    odom_noise = gtsam.noiseModel.Diagonal.Sigmas(
        np.array([0.05, 0.05, 0.05, 0.1, 0.1, 0.1])
    )
    
    # Pose 1: 1 m forward
    T_01 = gtsam.Pose3(gtsam.Rot3(), gtsam.Point3(1.0, 0.0, 0.0))
    graph.add(gtsam.BetweenFactorPose3(0, 1, T_01, odom_noise))
    initial.insert(1, gtsam.Pose3(gtsam.Rot3(), gtsam.Point3(1.0, 0.1, 0.0)))
    
    # Pose 2: 90-degree left turn + 1 m forward
    T_12 = gtsam.Pose3(
        gtsam.Rot3.Rz(np.pi / 2), 
        gtsam.Point3(0.0, 1.0, 0.0)   # translation is expressed in pose 1's frame, so forward after the turn is +y
    )
    graph.add(gtsam.BetweenFactorPose3(1, 2, T_12, odom_noise))
    initial.insert(2, gtsam.Pose3(
        gtsam.Rot3.Rz(np.pi / 2), 
        gtsam.Point3(1.0, 1.1, 0.0)
    ))
    
    # Loop closure: pose 2 → pose 0
    loop_noise = gtsam.noiseModel.Diagonal.Sigmas(
        np.array([0.1, 0.1, 0.1, 0.2, 0.2, 0.2])
    )
    T_20 = gtsam.Pose3(
        gtsam.Rot3.Rz(-np.pi / 2),
        gtsam.Point3(-1.0, 1.0, 0.0)   # pose2.between(pose0): from pose2 = (Rz(90°), (1,1,0)) back to the origin
    )
    graph.add(gtsam.BetweenFactorPose3(2, 0, T_20, loop_noise))
    
    # Optimize with iSAM2
    params = gtsam.ISAM2Params()
    isam = gtsam.ISAM2(params)
    isam.update(graph, initial)
    result = isam.calculateEstimate()
    
    # Print results
    for i in range(3):
        pose = result.atPose3(i)
        print(f"Pose {i}: x={pose.x():.3f}, y={pose.y():.3f}, "
              f"z={pose.z():.3f}")
    
    return result
```

**Ceres Solver**, developed by Google, targets general nonlinear least-squares optimization in C++ (with limited Python bindings). Rather than enforcing a factor-graph abstraction, it lets users define residual cost functions directly and supports full automatic differentiation, making it a staple in tightly coupled VIO pipelines such as VINS-Mono.

**g2o** (General Graph Optimization), introduced by Kümmerle et al. (2011), is a C++ library specialized for graph optimization. It ships with predefined vertex and edge types (`SE2`, `SE3`, `Sim3`, and others) under a direct vertex/(hyper-)edge graph model rather than GTSAM's factor hierarchy. Because runtime differences between solvers can flip depending on problem size, variable ordering, sparse linear solver, and compiler build flags, throughput should always be benchmarked on the target graph.

The table below summarizes how the three solvers compare across abstraction, differentiation, incremental solving, and ecosystem support:

| Property | GTSAM | Ceres | g2o |
|----------|-------|-------|-----|
| Abstraction level | Factor graph | Cost function | Graph vertex/edge |
| Automatic differentiation | Partial (Expression-based) | Full | Numerical differentiation by default; automatic differentiation available in some versions |
| Incremental | iSAM2 | Unsupported | Unsupported |
| Python support | Good | Limited | Limited |
| Representative use | LIO-SAM | VINS-Mono | ORB-SLAM |

### 12.5.2 Calibration Tools

Reliable multi-sensor fusion starts with accurate spatial and temporal calibration, supported by three widely used toolkits:

- **Kalibr** (ethz-asl): Performs camera–IMU, camera–camera, and multi-IMU calibration using continuous-time B-spline trajectories and an AprilGrid target; it remains a standard for visual–inertial rigs despite tricky ROS-dependent installation.
- **OpenCalib** (2022): Offers a suite of target-based and targetless calibration modules tailored to autonomous-driving stacks—covering camera intrinsics, LiDAR–camera, LiDAR–IMU, radar–camera, and LiDAR–LiDAR pairs—though it does not cover every possible combination across all four modalities.
- **[direct_visual_lidar_calibration](https://arxiv.org/abs/2302.05094)** (Koide et al. 2023): Performs targetless single-shot LiDAR–camera calibration by initializing correspondences with SuperGlue and refining extrinsics via Normalized Information Distance (NID) registration; for sparse spinning LiDARs, an accumulated point cloud may still be needed to reach sufficient point density.

### 12.5.3 Evaluation Tools

**evo** (MH Grupp) is the de facto Python toolkit for trajectory evaluation, computing and plotting ATE and RPE across TUM, KITTI, and EuRoC log formats via both CLI commands and a Python API.

For quick experiment checks from the shell, `evo_ape`, `evo_rpe`, and `evo_traj` handle alignment and plotting directly:

```bash
# evo usage examples
# Compute ATE
evo_ape tum groundtruth.txt estimated.txt -va --plot --plot_mode xz

# Compute RPE
evo_rpe tum groundtruth.txt estimated.txt -va --delta 100 --delta_unit f

# Compare two systems
evo_traj tum system_a.txt system_b.txt --ref groundtruth.txt -p --plot_mode xz
```

When integrating trajectory checks into automated benchmark suites or ablation scripts, the Python API exposes the same synchronization, alignment, and error statistics programmatically:

```python
# evo Python API usage example

from evo.core import metrics, sync
from evo.core.trajectory import PosePath3D, PoseTrajectory3D
from evo.tools import file_interface
import numpy as np
import copy

def evaluate_trajectory(gt_file, est_file, align=True):
    """
    Trajectory evaluation using evo.
    
    Args:
        gt_file: path to ground truth file (TUM format)
        est_file: path to estimated trajectory file
        align: whether to perform SE(3) alignment
    """
    # Load trajectories
    traj_gt = file_interface.read_tum_trajectory_file(gt_file)
    traj_est = file_interface.read_tum_trajectory_file(est_file)
    
    # Synchronize timestamps
    traj_gt, traj_est = sync.associate_trajectories(traj_gt, traj_est)
    
    # Compute ATE
    ate_metric = metrics.APE(metrics.PoseRelation.translation_part)
    
    if align:
        # Umeyama alignment
        traj_est_aligned = copy.deepcopy(traj_est)
        traj_est_aligned.align(traj_gt, correct_scale=False)   # align transforms in place; the return value is (R, t, s)
        ate_metric.process_data((traj_gt, traj_est_aligned))
    else:
        ate_metric.process_data((traj_gt, traj_est))
    
    stats = ate_metric.get_all_statistics()
    
    print(f"ATE RMSE: {stats['rmse']:.4f} m")
    print(f"ATE Mean: {stats['mean']:.4f} m")
    print(f"ATE Median: {stats['median']:.4f} m")
    print(f"ATE Max: {stats['max']:.4f} m")
    
    return stats


def compare_systems(gt_file, system_files, system_names):
    """Compare the ATE of multiple systems."""
    traj_gt = file_interface.read_tum_trajectory_file(gt_file)
    
    results = {}
    for name, est_file in zip(system_names, system_files):
        traj_est = file_interface.read_tum_trajectory_file(est_file)
        traj_gt_sync, traj_est_sync = sync.associate_trajectories(
            traj_gt, traj_est
        )
        
        traj_est_aligned = copy.deepcopy(traj_est_sync)
        traj_est_aligned.align(traj_gt_sync, correct_scale=False)
        
        ate = metrics.APE(metrics.PoseRelation.translation_part)
        ate.process_data((traj_gt_sync, traj_est_aligned))
        
        results[name] = ate.get_all_statistics()
    
    # Print comparison table
    print(f"{'System':<20} {'RMSE (m)':<12} {'Mean (m)':<12} {'Max (m)':<12}")
    print("-" * 56)
    for name, stats in results.items():
        print(f"{name:<20} {stats['rmse']:<12.4f} "
              f"{stats['mean']:<12.4f} {stats['max']:<12.4f}")
    
    return results
```

### 12.5.4 Other Essential Tools

Rounding out the practical engineering stack are four infrastructure and geometry tools that bridge raw sensor drivers, reconstruction, and offline inspection:

- **ROS 2** (Robot Operating System): Serves as the standard integration middleware for sensor drivers, time synchronization, and inter-process message passing, with most open-source SLAM pipelines shipping as ROS packages.
- **Open3D**: Provides Python-accessible 3D data structures—point clouds, triangle meshes, and TSDF volumes—alongside geometric primitives such as ICP, RANSAC, FPFH, and interactive visualization.
- **CloudCompare**: Offers an interactive GUI for point-cloud editing, filtering, downsampling, registration, and cloud-to-cloud (C2C) or cloud-to-mesh (C2M) distance inspection.
- **COLMAP**: Implements a full Structure-from-Motion (SfM) and Multi-View Stereo (MVS) pipeline, frequently used in sensor fusion to recover camera intrinsics or generate image-based reference maps.

---

These deployed architectures, datasets, metrics, and open-source tools represent the established engineering foundation of multi-sensor fusion. Yet the field is far from static: foundation models, differentiable end-to-end SLAM, persistent scene graphs, cross-modal embeddings, event cameras, and 4D imaging radar are actively reshaping how future systems perceive and remember the world—the **research frontiers** explored in Chapter 13.
