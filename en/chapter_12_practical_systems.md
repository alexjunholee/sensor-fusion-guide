# Ch.12 — Practical Systems & Benchmarks

Real platforms combine sensor modeling, state estimation, odometry, place recognition, and spatial representation into a single system.

Representative platforms include autonomous driving, drones, and handheld mapping, and each system is evaluated with standard benchmarks and tools.

---

## 12.1 Autonomous Driving Perception Stack

An autonomous-driving safety design must detect single-sensor failures and limit risk. Whether the vehicle continues or transitions to a minimal-risk condition depends on its ODD and safety case; multiple sensors can provide redundancy and cross-checks.

### 12.1.1 Sensor Suite Configuration Examples

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
- A system may center perception on multiple surround cameras and neural models.
- Installed sensors and radar use vary by vehicle generation, market, and date, so do not generalize an entire manufacturer as one fixed "vision-only" configuration.

### 12.1.2 Production-Level Fusion Pipeline

A production autonomous-driving system typically uses the following sensor fusion pipeline:

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

**Late Fusion vs Deep Fusion**:

- **Late fusion**: Each sensor independently detects 3D bounding boxes, which are then combined via NMS (Non-Maximum Suppression). It is modular and easy to debug, but has difficulty exploiting cross-sensor complementarity.

- **Deep fusion**: Features from multiple sensors are combined directly in BEV (Bird's Eye View) space. Representative systems include [BEVFusion](https://arxiv.org/abs/2205.13542) (MIT/Nvidia) and TransFusion. The network learns to exploit complementary cross-sensor information, but end-to-end training requires large-scale labeled data.

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

Autonomous driving localization is typically organized in two stages:

1. **Global localization**: GNSS (RTK or PPP) provides an initial position on the map. In urban environments, multipath can cause errors of several meters, so GNSS alone is insufficient.

2. **Map-relative localization**: register the current LiDAR scan to an HD point-cloud map with NDT or ICP. With adequate map accuracy, overlap, geometry, and initialization, this constrains position where GNSS is weak; centimeter-level error must be demonstrated under the actual map and protocol.

Factor-graph-based integration:
$$\mathbf{x}^* = \arg\min \underbrace{f_{\text{IMU}}}_{\text{prediction}} + \underbrace{f_{\text{LiDAR}}}_{\text{map matching}} + \underbrace{f_{\text{GNSS}}}_{\text{global anchor}} + \underbrace{f_{\text{wheel}}}_{\text{velocity}}$$

---

## 12.2 Drones/UAVs

Sensor fusion for drones operates under substantially different constraints than autonomous driving.

### 12.2.1 Visual-Inertial-Centric Systems

**Camera + IMU** is a common sensor combination on drones. Reasons:

- **Weight/size constraints**: Small drones cannot easily carry a LiDAR (though this is changing with compact solid-state LiDARs such as the Livox Mid-360).
- **Power constraints**: Cameras and IMUs consume little power.
- **Vibration**: Propeller vibration adds noise to IMU data. Vibration-isolating mounts and software filtering are required.

Representative VIO systems used on drones include the following:
- **VINS-Mono/Fusion**: tightly-coupled optimization-based. Can be integrated with PX4.
- **MSCKF/OpenVINS**: filter-based. Low computational cost makes it suitable for embedded boards.
- **Basalt**: visual-inertial mapping with non-linear factor recovery.

### 12.2.2 GPS-Denied Navigation

Autonomous flight in GPS-denied environments — indoors, tunnels, under forest canopy, and in electronic-warfare conditions — remains a challenge for drones.

The available approaches depend on whether prior infrastructure is available:

1. **VIO alone**: drift accumulates, so allowable mission duration follows from motion, texture, calibration, loop closure, and the error budget.
2. **VIO + terrain matching**: Match current camera observations against a pre-built terrain/building map. A prior map is required.
3. **VIO + UWB**: Install UWB anchors in the environment and use ranging measurements to correct drift. Prior infrastructure is required.
4. **VIO + barometer**: pressure change provides a relative-altitude cue. Weather, temperature, and prop wash introduce bias, so it does not remove absolute z drift by itself.

### 12.2.3 Real-Time Constraints

Set a drone's sensor rates and latency budget from maximum angular rate and acceleration, control bandwidth, exposure, estimator delay, and communication delay.

- **IMU rate**: high enough to limit aliasing and integration error under the specified motion.
- **Camera exposure**: short enough for the allowed image motion, balanced against low-light noise.
- **Processing latency**: an end-to-end bound derived from closed-loop stability analysis and flight testing; 30 ms is not universal.
- **Point-LIO**: An LIO that reduces latency by processing points individually without waiting for scan completion. It can be used for high-agility drone maneuvers.

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

Mapping environments with handheld or backpack-mounted sensors is widely used in surveying, BIM (Building Information Modeling), and digital twin construction.

### 12.3.1 SLAM as a Service

Examples of commercial handheld mapping devices:

- **Leica BLK2GO**: Handheld LiDAR scanner. Performs real-time SLAM by fusing LiDAR + IMU + camera. Survey-grade accuracy.
- **NavVis VLX**: Backpack-mounted. Four cameras + LiDAR. Specialized for indoor mapping.
- **GeoSLAM ZEB**: Handheld mobile mapping. A 2D LiDAR is rotated manually to produce 3D scans.

These devices generally use the following pipeline:

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

Survey mapping begins with the contract specification and validation procedure:

- **Absolute accuracy**: specify horizontal and vertical error and confidence, then validate against independent checkpoints. Use GNSS, total station, or GCPs as site conditions require.
- **Relative accuracy**: Internal consistency of the map. Loop closure and global optimization establish this consistency.
- **Point density**: specify spacing and completeness from object size and deliverable. A 1 cm value may be a project requirement, not a universal survey standard.
- **Color quality**: Accurate HDR color mapping. Camera-LiDAR time synchronization and extrinsic calibration must be precise.

Deployed sensor fusion systems must address the following problems:

1. **Degenerate environments**: Long corridors, empty rooms, and other environments lacking geometric features. Drift that occurs in LiDAR-only systems is compensated by cameras or IMU. Multi-modal systems such as R3LIVE and FAST-LIVO2 are effective.

2. **Multi-story buildings**: Loop closure is essential when moving between floors via elevators or stairs. With no GNSS, z-axis drift is especially problematic. A barometer serves as a useful auxiliary sensor.

3. **Glass/mirrors**: LiDAR beams either pass through or reflect. Compensate with cameras or filter out reflected points.

---

## 12.4 Benchmarks & Evaluation

Fair comparison of sensor fusion systems requires standardized datasets and evaluation metrics.

### 12.4.1 Major Datasets

| Dataset | Year | Environment | Sensors | Features |
|---------|------|-------------|---------|----------|
| **[KITTI](https://doi.org/10.1177/0278364913491297)** | 2012 | Outdoor (autonomous driving) | Stereo, LiDAR, GPS/IMU | 11 training + 11 test sequences for odometry evaluation |
| **[EuRoC](https://doi.org/10.1177/0278364915620033)** | 2016 | Indoor (MAV) | Stereo, IMU | VIO sequences in Machine Hall + Vicon Room |
| **[TUM-RGBD](https://doi.org/10.1109/IROS.2012.6385773)** | 2012 | Indoor | RGB-D | Kinect v1 visual-SLAM sequences and evaluation tools |
| **TUM-VI** | 2018 | Indoor + outdoor | Stereo, IMU | VIO benchmark. Diverse motion patterns |
| **[Hilti](https://arxiv.org/abs/2109.11316)** | 2021–2023 challenge editions | Construction sites | LiDAR, Camera, IMU | Industrial-environment-specific. Challenging conditions |
| **[HeLiPR](https://arxiv.org/abs/2309.14590)** | 2023 | Outdoor (urban) | Heterogeneous LiDAR, Camera, IMU, GNSS | For heterogeneous LiDAR fusion research. Ouster+Velodyne+Livox+Aeva |
| **[nuScenes](https://arxiv.org/abs/1903.11027)** | 2020 | Outdoor (autonomous driving) | Camera, LiDAR, Radar, GPS/IMU | 1000 scenes, 23-class 3D annotations, 360° surround sensors |
| **Newer College** | 2020 | Outdoor + indoor | LiDAR, Camera, IMU | Oxford University campus. Multi-session |

Characteristics and uses of each dataset:

**KITTI** — Released in 2012, it remains a long-standing reference benchmark for autonomous-driving odometry and SLAM. It provides a Velodyne 64-channel LiDAR, stereo cameras, and GPS/IMU. Its sensors are dated, its sequences are relatively short, and its GPS/INS-based ground truth does not guarantee centimeter-level accuracy in every segment.

**EuRoC** — Stereo camera + IMU data captured on a drone (MAV). It is a widely used VIO benchmark. Depending on the sequence, ground truth was recorded with a Vicon motion-capture system or a Leica laser tracker. The 11 sequences are classified as easy → medium → difficult.

**Hilti** — Evaluates SLAM in challenging construction-site environments with dust, vibration, and repetitive structures. Public sources document challenge editions in 2021, 2022, and 2023; the sensor suites and evaluation rules differ by edition.

**HeLiPR** — Released in 2023, it mounts different types of LiDAR (spinning, solid-state, FMCW) simultaneously. It is used for research on heterogeneous LiDAR fusion.

**Newer College** — Collected by visiting the Oxford University campus multiple times, this dataset is well suited to multi-session SLAM and long-term mapping research. Captured with a handheld LiDAR, it contains challenging motion patterns.

**Recent benchmark trends (2024-2025)**:

- **[Hilti-Oxford](https://arxiv.org/abs/2208.09825)** (2022): A construction-environment SLAM benchmark with millimeter-level ground truth used for the 2022 challenge.
- **[Boreas](https://arxiv.org/abs/2203.10168)** (Burnett et al. 2023): An autonomous driving dataset collected by repeatedly driving the same route over a year. It includes LiDAR, radar, and cameras and captures all four seasons as well as adverse weather conditions.
- **[Snail-Radar](https://arxiv.org/abs/2407.11705)** (Huai et al., IJRR 2025): A large-scale benchmark for evaluating 4D radar SLAM that systematically compares 4D radar odometry/SLAM across diverse environments and platforms.

### 12.4.2 Evaluation Metrics

**ATE (Absolute Trajectory Error)**: measures the global difference between the estimated trajectory and the ground truth trajectory.

$$\text{ATE} = \sqrt{\frac{1}{N} \sum_{i=1}^{N} \| \text{trans}(\mathbf{T}_{\text{gt},i}^{-1} \cdot \mathbf{T}_{\text{est},i}) \|^2}$$

Before evaluation, the two trajectories must be aligned in Sim(3) or SE(3). Monocular VO has scale ambiguity, so Sim(3) is used; for stereo/LiDAR, SE(3) is used.

$$\mathbf{S}^* = \arg\min_{\mathbf{S} \in \text{Sim}(3)} \sum_i \| \mathbf{p}_{\text{gt},i} - \mathbf{S} \cdot \mathbf{p}_{\text{est},i} \|^2$$

This alignment admits a closed-form solution via the Umeyama algorithm.

**RPE (Relative Pose Error)**: measures relative accuracy over short intervals. It reflects drift tendency.

$$\text{RPE}(\Delta) = \sqrt{\frac{1}{M} \sum_{i=1}^{M} \| \text{trans}((\mathbf{T}_{\text{gt},i}^{-1} \mathbf{T}_{\text{gt},i+\Delta})^{-1} (\mathbf{T}_{\text{est},i}^{-1} \mathbf{T}_{\text{est},i+\Delta})) \|^2}$$

$\Delta$ is the evaluation interval (in frames or distance). RPE at short $\Delta$ reflects odometry accuracy, while RPE at long $\Delta$ reflects drift.

**Place recognition metrics**:
- **Recall@N**: the fraction of queries for which the correct place is included among the top-N candidates. Recall@1 is the strictest.
- **Precision-Recall curve**: the precision vs recall trade-off as a function of threshold.
- **AUC**: the area under the PR curve.

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

The following caveats apply when interpreting benchmark results:

1. **Parameter tuning**: The same algorithm can perform very differently depending on its parameters. Tuning for a specific dataset reduces generality.

2. **Hardware dependence**: Real-time performance depends heavily on hardware. The definition of "real-time" varies across papers (desktop GPU vs embedded ARM).

3. **Completeness**: Some systems experience tracking loss on difficult sequences; computing ATE only over successful segments fails to reflect the failure rate. **Completeness** (= the fraction of sequences that succeed) should be reported alongside.

4. **Initialization differences**: Different initialization methods and times in VIO systems can yield different results on the same sequence.

5. **Whether loop closure is included**: VO (no loop closure) vs SLAM (with loop closure) must be distinguished. Loop closure can substantially improve ATE.

---

## 12.5 Open-Source Tool Guide

Sensor fusion research and practice commonly use the following open-source tools.

### 12.5.1 Optimization Libraries

**GTSAM** (Georgia Tech Smoothing and Mapping):
- Factor-graph-based optimization library
- Includes an iSAM2 implementation
- C++ with Python bindings (gtsam)
- Backend of many SLAM systems including LIO-SAM
- Strengths: intuitive factor graph construction, many built-in factor types

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
        gtsam.Point3(1.0, 0.0, 0.0)
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
        gtsam.Rot3.Rz(np.pi / 2),
        gtsam.Point3(0.0, -1.0, 0.0)
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

**Ceres Solver**:
- A nonlinear least squares optimization library developed by Google
- C++ only (Python bindings are limited)
- Supports automatic differentiation
- Used by VINS-Mono, ORB-SLAM, and others
- Defines the optimization problem directly, without a factor graph abstraction

**g2o** (General Graph Optimization):
- Developed by Kümmerle et al. (2011)
- C++ library specialized for graph optimization
- Predefines various vertex/edge types (SE2, SE3, Sim3, etc.)
- Lighter and faster than GTSAM but less flexible

Comparison of the three libraries:

| Property | GTSAM | Ceres | g2o |
|----------|-------|-------|-----|
| Abstraction level | Factor graph | Cost function | Graph vertex/edge |
| Automatic differentiation | Partial | Full | None |
| Incremental | iSAM2 | Unsupported | Unsupported |
| Python support | Good | Limited | Limited |
| Representative use | LIO-SAM | VINS-Mono | ORB-SLAM |

### 12.5.2 Calibration Tools

**Kalibr** (ethz-asl):
- Camera-IMU, Camera-Camera, and multi-IMU calibration
- Continuous-time B-spline trajectory-based
- Uses an AprilGrid target
- Widely used for camera-IMU calibration, though installation is tricky (ROS dependency)

**OpenCalib** (2023):
- Unified calibration across the full autonomous driving sensor stack
- Supports all combinations among Camera, LiDAR, Radar, and IMU
- Covers both target-based and targetless methods

**[direct_visual_lidar_calibration](https://arxiv.org/abs/2302.05094)** (Koide et al. 2023):
- NID-based targetless LiDAR-camera calibration
- Initial estimate via SuperGlue, refined via NID registration
- Operates from a single capture

### 12.5.3 Evaluation Tools

**evo** (MH Grupp):
- Python-based trajectory evaluation tool
- Computes and visualizes ATE and RPE
- Supports a variety of formats including TUM, KITTI, and EuRoC
- Provides both command-line tools and a Python API

```bash
# evo usage examples
# Compute ATE
evo_ape tum groundtruth.txt estimated.txt -va --plot --plot_mode xz

# Compute RPE
evo_rpe tum groundtruth.txt estimated.txt -va --delta 100 --delta_unit f

# Compare two systems
evo_traj tum system_a.txt system_b.txt --ref groundtruth.txt -p --plot_mode xz
```

```python
# evo Python API usage example

from evo.core import metrics, sync
from evo.core.trajectory import PosePath3D, PoseTrajectory3D
from evo.tools import file_interface
import numpy as np

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
        traj_est_aligned = traj_est.align(traj_gt, correct_scale=False)
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
        
        traj_est_aligned = traj_est_sync.align(
            traj_gt_sync, correct_scale=False
        )
        
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

**ROS 2** (Robot Operating System):
- Integration framework for sensor fusion systems
- Provides sensor drivers, time synchronization, and message-passing infrastructure
- Most open-source SLAM systems are distributed as ROS packages

**Open3D**:
- Python library for 3D data processing
- Point cloud, mesh, and TSDF processing
- Built-in geometric algorithms such as ICP, RANSAC, and FPFH
- Excellent visualization

**CloudCompare**:
- GUI tool for point cloud comparison/editing
- Distance comparison between two point clouds (C2C, C2M)
- Point cloud registration, filtering, and downsampling

**COLMAP**:
- Structure from Motion (SfM) + Multi-View Stereo (MVS) pipeline
- 3D reconstruction from image collections
- Used in sensor fusion for camera intrinsic estimation and ground truth map construction

---

The sensor-fusion methods covered here reach products and research through these systems and benchmarks. Foundation models, event cameras, 4D radar, and end-to-end SLAM remain active **research frontiers**.
