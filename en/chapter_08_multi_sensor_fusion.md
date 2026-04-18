# Ch.8 — Multi-Sensor Fusion Architectures

> A design discipline that goes beyond individual odometry to address **how multiple sensors are integrated**.
> Having covered Visual Odometry and LiDAR Odometry separately in previous chapters, this chapter provides an in-depth analysis of the architectures that weave them into a single system.

---

## 8.1 Taxonomy of Fusion Architectures

The first decision in designing a multi-sensor fusion system is **at what level the sensor data will be combined**. The depth of this coupling fundamentally determines the system's complexity, performance, and failure modes.

### 8.1.1 Loosely Coupled

**Intuition**: Each sensor is viewed as an independent "expert." Each expert draws its own conclusion (pose, velocity, etc.) from its own data, and a higher-level stage then synthesizes these conclusions.

Concretely, the LiDAR odometry module independently estimates $\mathbf{T}_{L}$ from LiDAR scans and the Visual odometry module independently estimates $\mathbf{T}_{V}$ from images, and a higher-level fusion module combines these two estimates.

$$
\hat{\mathbf{x}} = \arg\min_{\mathbf{x}} \left\| \mathbf{x} - \mathbf{x}_{LiDAR} \right\|^2_{\mathbf{P}_{L}^{-1}} + \left\| \mathbf{x} - \mathbf{x}_{Visual} \right\|^2_{\mathbf{P}_{V}^{-1}}
$$

Here $\mathbf{P}_{L}$ and $\mathbf{P}_{V}$ are the covariances reported by each subsystem.

**Advantages**:
- High modularity. Each sensor module can be swapped or upgraded independently.
- Easier debugging. It is easy to trace which module a problem came from.
- If one sensor fails, the others continue to operate.

**Disadvantages**:
- Because each subsystem combines its output after information has already been lost, the full benefit of complementary interaction between sensors cannot be exploited. For example, LiDAR's precise geometric information can resolve a camera's scale ambiguity, but in a loosely coupled design this interaction is limited.
- The consistency of the covariances reported by each subsystem is not guaranteed. If a subsystem reports an overly optimistic covariance, the fusion result is distorted.

### 8.1.2 Tightly Coupled

**Intuition**: All sensors' **raw measurements** are fed directly into a single estimator. Instead of consulting "experts," a single "chief analyst" inspects all raw data directly.

From a factor graph perspective, each sensor's raw measurements are inserted as independent factors:

$$
\hat{\mathbf{x}} = \arg\min_{\mathbf{x}} \sum_{i} \left\| \mathbf{r}^{\text{IMU}}_i(\mathbf{x}) \right\|^2_{\boldsymbol{\Sigma}^{-1}_{\text{IMU}}} + \sum_{j} \left\| \mathbf{r}^{\text{LiDAR}}_j(\mathbf{x}) \right\|^2_{\boldsymbol{\Sigma}^{-1}_{\text{LiDAR}}} + \sum_{k} \left\| \mathbf{r}^{\text{cam}}_k(\mathbf{x}) \right\|^2_{\boldsymbol{\Sigma}^{-1}_{\text{cam}}}
$$

Here $\mathbf{r}^{\text{IMU}}_i$ is the IMU preintegration residual, $\mathbf{r}^{\text{LiDAR}}_j$ is the point-to-plane residual, and $\mathbf{r}^{\text{cam}}_k$ is the reprojection error.

**Advantages**:
- Sensor-to-sensor interaction is exploited to the fullest. For example, the IMU corrects the LiDAR's motion distortion, and the LiDAR anchors the VIO's scale.
- Fusion is close to information-theoretically optimal.

**Disadvantages**:
- System complexity is high. The observation models, noise models, and time synchronization of all sensors must be managed within a single framework.
- Anomalous data from one sensor can contaminate the entire estimate (outlier handling is essential).
- Achieving real-time performance is difficult.

**Representative systems**: [LIO-SAM](https://arxiv.org/abs/2007.00258) (LiDAR+IMU+GPS), VINS-Mono (Camera+IMU), [R3LIVE](https://arxiv.org/abs/2109.07982) (Camera+LiDAR+IMU).

### 8.1.3 Ultra-Tightly Coupled (Signal-Level Coupling)

**Intuition**: Not the sensor's measurements but the **signals themselves** are combined. This is the most extreme form of integration.

A representative example is GNSS-INS ultra-tight coupling. A typical GNSS receiver extracts pseudoranges from satellite signals and then combines them with the INS, but in the ultra-tight approach the position and velocity estimated by the INS directly aid the GNSS receiver's code/carrier tracking loops. Doing so lets the receiver keep tracking GNSS signals longer even in weak-signal environments (urban canyons, right after entering indoors).

$$
\text{NCO frequency} = f_{\text{nominal}} + \Delta f_{\text{INS-aided}}
$$

Here the frequency of the NCO (Numerically Controlled Oscillator) is corrected by the Doppler shift predicted by the INS, widening the receiver's tracking range.

**Reality**: Ultra-tight coupling requires access to the GNSS receiver at the hardware/firmware level, so it is rarely seen outside military and aviation applications. For most robotics systems, tightly coupled is the practical limit.

### 8.1.4 Comparison of the Three Levels

```
Measurement flow:

[Loosely]    Sensor A → Subsystem A → Pose A ─┐
                                              ├→ Fusion → Final pose
             Sensor B → Subsystem B → Pose B ─┘

[Tightly]    Sensor A → raw meas. A ──┐
                                      ├→ Single Optimizer → Final pose
             Sensor B → raw meas. B ──┘

[Ultra-Tight] Sensor A signal ←→ Sensor B estimate (bidirectional signal-level coupling)
```

```python
import numpy as np
from scipy.linalg import inv

def loosely_coupled_fusion(x_lidar, P_lidar, x_visual, P_visual):
    """
    Loosely coupled fusion: combine independent estimates from two subsystems
    via a covariance-weighted average.

    Parameters:
        x_lidar: state estimate from the LiDAR subsystem (n,)
        P_lidar: covariance of the LiDAR estimate (n, n)
        x_visual: state estimate from the Visual subsystem (n,)
        P_visual: covariance of the Visual estimate (n, n)

    Returns:
        x_fused: fused state estimate (n,)
        P_fused: fused covariance (n, n)
    """
    # Convert to information form
    I_lidar = inv(P_lidar)
    I_visual = inv(P_visual)

    # Sum of information matrices
    I_fused = I_lidar + I_visual
    P_fused = inv(I_fused)

    # Information-weighted mean
    x_fused = P_fused @ (I_lidar @ x_lidar + I_visual @ x_visual)

    return x_fused, P_fused


# Example: 2D position estimation
x_lidar = np.array([10.1, 5.2])      # Position estimated by LiDAR
P_lidar = np.diag([0.01, 0.01])       # LiDAR is precise and isotropic
x_visual = np.array([10.0, 5.0])      # Position estimated by Visual
P_visual = np.diag([0.1, 0.05])       # Visual is less precise in the vertical direction

x_fused, P_fused = loosely_coupled_fusion(x_lidar, P_lidar, x_visual, P_visual)
print(f"LiDAR:  {x_lidar}, P_diag: {np.diag(P_lidar)}")
print(f"Visual: {x_visual}, P_diag: {np.diag(P_visual)}")
print(f"Fused:  {x_fused}, P_diag: {np.diag(P_fused)}")
# The fused result is closer to the LiDAR estimate (since its covariance is smaller)
```

<!-- DEMO: fusion_architecture_comparison.html -->

---

## 8.2 Camera + LiDAR + IMU Fusion

The combination of camera, LiDAR, and IMU currently constitutes the most information-rich sensor suite in autonomous driving and robotics. The camera provides texture and color information, the LiDAR provides precise 3D geometry, and the IMU provides high-rate inertial measurements. These three sensors complement one another's weaknesses:

| Situation | Camera | LiDAR | IMU |
|------|--------|-------|-----|
| Dark environment | ✗ | ✓ | ✓ |
| Textureless wall | ✗ | ✓ | ✓ |
| Geometric degeneracy (long corridor) | ✓ | ✗ | ✓ |
| High-speed rotation | ✗ | ✗ | ✓ |
| Scale observability | ✗ (monocular) | ✓ | ✗ |
| Color/semantics | ✓ | ✗ | ✗ |

We analyze recent state-of-the-art systems that integrate these three sensors.

### 8.2.1 R3LIVE / R3LIVE++

[R3LIVE](https://arxiv.org/abs/2109.07982) (Lin et al., 2022) is a system that tightly couples two subsystems: LiDAR-Inertial Odometry (LIO) and Visual-Inertial Odometry (VIO).

**Core architectural idea**:

R3LIVE adopts a **dual-subsystem** architecture. The LIO subsystem is responsible for geometry and the VIO subsystem is responsible for photometric (texture) information, and the two subsystems are tightly coupled by **sharing a single state**.

```
LiDAR scan ──→ [LIO subsystem] ──→ state update (geometry)
                     ↓                       ↓
                 IMU data ───────────→ shared state vector
                     ↑                       ↑
Camera image ──→ [VIO subsystem] ──→ state update (photometric)
```

**LIO subsystem**: Identical to FAST-LIO2, raw LiDAR points are directly point-to-plane registered to an ikd-Tree-based map. The state is updated by an Iterated EKF.

**VIO subsystem**: This is where R3LIVE's originality shines. A typical VIO minimizes the reprojection error of feature points, whereas R3LIVE uses the **photometric error**. Specifically, each point in the 3D map built by LIO is assigned an RGB color, and when a new camera image arrives, these map points are projected into the image to minimize the **difference between the observed color and the color stored in the map**:

$$
\mathbf{r}^{\text{photo}}_i = \mathbf{I}(\pi(\mathbf{T}_{CW} \mathbf{p}^W_i)) - \mathbf{c}_i^{\text{map}}
$$

Here $\mathbf{I}(\cdot)$ is the pixel intensity of the image, $\pi(\cdot)$ is the 3D→2D projection function, $\mathbf{T}_{CW}$ is the world-to-camera transformation, $\mathbf{p}^W_i$ is the 3D coordinate of a map point, and $\mathbf{c}_i^{\text{map}}$ is the color stored in the map for that point.

**Robustness**: The key design benefit is that if either the LiDAR or the camera temporarily fails, the system continues to operate with the remaining sensors. When the LiDAR is occluded, VIO+IMU operate; when the camera is dark, LIO+IMU operate.

**Result**: It produces a survey-grade colored 3D map in real time while performing SLAM.

### 8.2.2 LVI-SAM

[LVI-SAM](https://arxiv.org/abs/2104.10831) (Shan et al., 2021) is an extension of LIO-SAM that couples a Visual-Inertial subsystem and a LiDAR-Inertial subsystem **bidirectionally**.

**The essence of bidirectional coupling**:

- **VIS → LIS direction**: The pose estimated by the Visual-Inertial subsystem is used as the initial guess for LiDAR scan matching. Especially when the LiDAR alone yields an inaccurate initial guess (high-speed rotation, featureless environments), VIS provides the initial guess and helps LiDAR registration converge.

- **LIS → VIS direction**: The depth information estimated by the LiDAR is assigned to Visual feature points, accelerating depth initialization in the Visual subsystem. In monocular VIO, feature depth is estimated by triangulation, but depth is inaccurate until sufficient parallax accumulates. By providing this depth directly from the LiDAR, immediate initialization becomes possible.

```
         ┌─── VIS initial pose ───→ LIS initial guess
         │                            │
  [Visual-Inertial]            [LiDAR-Inertial]
         │                            │
         └←── LiDAR depth ────────────┘

              ↓ both factors ↓
           [Factor Graph (GTSAM/iSAM2)]
                     ↓
              Final optimized pose
```

**Factor graph design**: The following factors are inserted into the LVI-SAM factor graph:
- IMU preintegration factor (between successive keyframes)
- LiDAR odometry factor (scan matching result)
- Visual odometry factor (feature tracking result)
- GPS factor (when available)
- Loop closure factor (upon revisit detection)

### 8.2.3 FAST-LIVO / FAST-LIVO2

[FAST-LIVO2](https://arxiv.org/abs/2408.14035) (Zheng et al., 2024) is a direct Camera+LiDAR+IMU fusion system developed by the FAST-LIO2 team (HKU MARS Lab). "Direct" means raw data is used without feature extraction.

**Key innovation 1 — Sequential Update**:

Measurements from heterogeneous sensors have different dimensionalities. LiDAR provides 3D point-to-plane residuals, while the camera provides 2D photometric residuals. Stacking them into a single large residual vector and optimizing simultaneously complicates the Jacobian matrix structure and can be numerically unstable.

FAST-LIVO2 solves this problem with **sequential Bayesian updates**:

1. Predict the state with the IMU (prediction)
2. Update the state with LiDAR measurements (first update)
3. Update the state again with camera measurements (second update)

Theoretically, if the measurements are independent, sequential updating gives the same result as simultaneous updating:

$$
p(\mathbf{x} | \mathbf{z}_L, \mathbf{z}_C) = p(\mathbf{x} | \mathbf{z}_C, \mathbf{z}_L) \propto p(\mathbf{z}_C | \mathbf{x}) \cdot p(\mathbf{z}_L | \mathbf{x}) \cdot p(\mathbf{x})
$$

Sequentially:
$$
\underbrace{p(\mathbf{x} | \mathbf{z}_L)}_{\text{after LiDAR update}} \propto p(\mathbf{z}_L | \mathbf{x}) \cdot p(\mathbf{x})
$$
$$
\underbrace{p(\mathbf{x} | \mathbf{z}_L, \mathbf{z}_C)}_{\text{after camera update}} \propto p(\mathbf{z}_C | \mathbf{x}) \cdot p(\mathbf{x} | \mathbf{z}_L)
$$

In the second equation, $p(\mathbf{x} | \mathbf{z}_L)$ serves as the prior, and the final result is mathematically equivalent to the simultaneous update.

**Key innovation 2 — Unified adaptive voxel map**:

FAST-LIVO2 uses a single voxel map based on a hash table plus an octree. The LiDAR module builds the geometric structure (3D coordinates, normal vectors), and the Visual module attaches image patches to the same map points. Geometry and texture are thus managed consistently within a single map.

**Key innovation 3 — Affine warping using LiDAR normals**:

When comparing image patches in a camera direct method, affine warping that accounts for surface tilt improves accuracy. FAST-LIVO2 leverages the planar normal vectors extracted from the LiDAR to perform accurate affine warping without any separate normal estimation. This is a concrete example of LiDAR-camera complementarity.

**Key innovation 4 — Real-time exposure compensation**:

In environments with rapidly changing illumination (entering/exiting a tunnel), FAST-LIVO2 estimates the exposure time online and corrects the photometric error accordingly.

```python
import numpy as np

def sequential_ekf_update(x_pred, P_pred, z_lidar, H_lidar, R_lidar, z_cam, H_cam, R_cam):
    """
    Sequential EKF update in the order LiDAR → Camera.
    Mathematically equivalent to a simultaneous update but avoids the
    dimensionality-mismatch issue.

    Parameters:
        x_pred: predicted state (n,)
        P_pred: predicted covariance (n, n)
        z_lidar: LiDAR measurement (m_L,)
        H_lidar: LiDAR observation Jacobian (m_L, n)
        R_lidar: LiDAR measurement noise (m_L, m_L)
        z_cam: camera measurement (m_C,)
        H_cam: camera observation Jacobian (m_C, n)
        R_cam: camera measurement noise (m_C, m_C)

    Returns:
        x_updated: final updated state
        P_updated: final updated covariance
    """
    # Step 1: LiDAR update
    S_L = H_lidar @ P_pred @ H_lidar.T + R_lidar
    K_L = P_pred @ H_lidar.T @ np.linalg.inv(S_L)
    y_L = z_lidar - H_lidar @ x_pred  # innovation
    x_after_lidar = x_pred + K_L @ y_L
    P_after_lidar = (np.eye(len(x_pred)) - K_L @ H_lidar) @ P_pred

    # Step 2: Camera update (use the post-LiDAR result as the prior)
    S_C = H_cam @ P_after_lidar @ H_cam.T + R_cam
    K_C = P_after_lidar @ H_cam.T @ np.linalg.inv(S_C)
    y_C = z_cam - H_cam @ x_after_lidar  # innovation
    x_updated = x_after_lidar + K_C @ y_C
    P_updated = (np.eye(len(x_pred)) - K_C @ H_cam) @ P_after_lidar

    return x_updated, P_updated
```

### 8.2.4 Comparison of Multimodal Factor Graph Designs

We compare the designs of the three systems from a factor graph perspective:

| Aspect | R3LIVE | LVI-SAM | FAST-LIVO2 |
|------|--------|---------|------------|
| Backend | IEKF (dual subsystem) | iSAM2 (factor graph) | IEKF (sequential) |
| LiDAR processing | Direct (point-to-plane) | Feature-based (LOAM) | Direct (point-to-plane) |
| Camera processing | Direct (photometric) | Feature-based (ORB) | Direct (photometric) |
| Map representation | ikd-Tree + RGB | Voxel map | Hash+Octree voxel map |
| Feature extraction | Not required | Required (edge/planar, ORB) | Not required |
| GPS integration | None | Integrated as factor | None |
| Loop closure | None | Integrated as factor | None |
| Embedded support | Limited | Limited | ARM real-time capable |

**Selection criteria**:
- If loop closure and GPS are needed: LVI-SAM
- If the highest-precision colored map is needed: R3LIVE
- If real-time operation on embedded platforms is needed: FAST-LIVO2
- In environments with scarce feature points (textureless walls, interiors of structures): direct methods (R3LIVE, FAST-LIVO2)

---

## 8.3 GNSS Integration

GNSS (Global Navigation Satellite System) is the only sensor that provides a global position reference. No matter how precise IMU+LiDAR+camera are, they all provide only **relative** measurements, so drift accumulates over long-duration operation. GNSS serves as an anchor that corrects this drift.

### 8.3.1 GNSS Factor in the Factor Graph (LIO-SAM Approach)

[LIO-SAM](https://arxiv.org/abs/2007.00258) (Shan et al., 2020) shows a clean way to integrate GNSS into a factor graph. When the GNSS receiver reports a position, it is connected to a pose node as a **unary factor**:

$$
\mathbf{r}^{\text{GPS}}_i = \mathbf{T}^{-1}_{\text{ENU→map}} \cdot \mathbf{p}^{\text{ENU}}_{\text{GPS}} - \mathbf{p}^{\text{map}}_i - \mathbf{R}^{\text{map}}_i \cdot \mathbf{l}_{\text{antenna}}
$$

where:
- $\mathbf{p}^{\text{ENU}}_{\text{GPS}}$ is the ENU coordinate reported by the GNSS
- $\mathbf{T}_{\text{ENU→map}}$ is the transformation from the ENU frame to the SLAM map frame
- $\mathbf{p}^{\text{map}}_i$ is the robot position estimated by SLAM
- $\mathbf{l}_{\text{antenna}}$ is the lever-arm vector between the GNSS antenna and the robot's body frame
- $\mathbf{R}^{\text{map}}_i$ is the robot's rotation

**Frame alignment issue**: The SLAM local map frame and the GNSS global frame (WGS84/ENU) are different. At the first GNSS reception, the ENU origin is set, and the map↔ENU transformation is estimated using the initial poses. This transformation has 6-DoF (3 translation + 3 rotation), but because the IMU provides the gravity direction, only 4-DoF (yaw + 3 translation) need to be estimated in practice.

### 8.3.2 Loosely vs Tightly Coupled GNSS

**Loosely Coupled GNSS-INS**:
The position/velocity solution (PVT solution) already computed by the GNSS receiver is combined with the IMU estimate by an EKF. Most commercial systems use this approach.

```python
def gnss_loose_coupling_ekf_update(x_ins, P_ins, gnss_position, R_gnss):
    """
    Loosely coupled GNSS-INS: correct the INS state using the GNSS PVT solution.

    x_ins: INS state [position(3), velocity(3), attitude(3), biases(6)] = 15 dim
    gnss_position: position computed by GNSS (3,)
    R_gnss: covariance of the GNSS position (3, 3) — typically HDOP * sigma_uere
    """
    n = len(x_ins)
    # Observation matrix: GNSS observes position only
    H = np.zeros((3, n))
    H[0:3, 0:3] = np.eye(3)  # only the position portion is observed

    # Innovation
    y = gnss_position - H @ x_ins

    # Kalman gain
    S = H @ P_ins @ H.T + R_gnss
    K = P_ins @ H.T @ np.linalg.inv(S)

    # Update
    x_updated = x_ins + K @ y
    P_updated = (np.eye(n) - K @ H) @ P_ins

    return x_updated, P_updated
```

**Tightly Coupled GNSS-INS**:
Instead of the GNSS receiver's PVT solution, the raw pseudorange and Doppler measurements are used directly. The pseudorange to each satellite is inserted as an individual factor:

$$
\rho_i = \| \mathbf{p}_{\text{sat},i} - \mathbf{p}_{\text{rx}} \| + c \cdot \delta t_{\text{rx}} + I_i + T_i + \epsilon_i
$$

Here $\rho_i$ is the pseudorange to satellite $i$, $c \cdot \delta t_{\text{rx}}$ is the receiver clock bias, and $I_i$ and $T_i$ are the ionospheric/tropospheric delays.

The advantage of tight coupling is that even when fewer than four satellites are visible — so that GNSS cannot produce a solution on its own — the pseudoranges from the available satellites can still be exploited. Because buildings frequently occlude satellites in urban environments, this advantage is substantial in practice.

### 8.3.3 Handling GNSS-Denied → GNSS-Available Transitions

In real robot operation, GNSS signals are repeatedly lost and recovered (tunnels, underground parking lots, under overpasses). Handling these transitions reliably is a central challenge of system design.

**Transition considerations**:
1. **Avoid coordinate jumps**: Immediately after GNSS recovery, there can be a large discrepancy between the GNSS position and the IMU/LiDAR-estimated position. Correcting this abruptly introduces discontinuities in the map. The solution is to initially set the GNSS uncertainty to a large value and decrease it gradually.

2. **GNSS quality verification**: Measurements in the first few seconds after recovery can lose accuracy due to multipath and similar effects. Include them as factors only after verifying PDOP/HDOP, satellite count, carrier phase status, and the like to ensure sufficient reliability.

3. **Map frame correction**: If drift has accumulated during a prolonged GNSS outage, the map frame itself may need to be corrected upon recovery. This is handled by pose graph optimization similar to loop closure.

---

## 8.4 Radar Fusion

### 8.4.1 Radar Revisited

Traditionally, automotive radar was considered unsuitable for SLAM/odometry because of its low resolution. However, the emergence of **4D imaging radar** is rapidly changing the situation.

**What is 4D radar**: Whereas conventional automotive radars measured three quantities — range, Doppler velocity, and azimuth — 4D imaging radar adds **elevation** to produce a 3D point cloud. The resolution is nowhere near that of LiDAR (hundreds to thousands of points vs. hundreds of thousands), but it has unique strengths.

**Unique advantages of radar**:

1. **Adverse-weather penetration**: It penetrates rain, snow, fog, and dust. LiDAR (905 nm/1550 nm lasers) degrades sharply under such conditions, whereas radar (mm-wave) is nearly unaffected. From the standpoint of autonomous-driving safety, this is decisive.

2. **Direct velocity measurement**: FMCW (Frequency-Modulated Continuous Wave) radar uses the Doppler effect to **measure the relative velocity of objects directly**. Cameras and LiDAR must infer velocity indirectly by comparing consecutive frames, whereas radar obtains velocity from a single measurement.

3. **Low cost**: Automotive radar chipsets are produced at mass-market scale, making them at least an order of magnitude cheaper than LiDAR.

### 8.4.2 Radar Odometry

Odometry using 4D radar is a rapidly emerging field. The key idea is to exploit radar Doppler measurements directly for ego-motion estimation.

Each measurement point of an FMCW radar provides $(r, \theta, \phi, v_d)$ — range, azimuth, elevation, Doppler velocity. Given the robot's linear velocity $\mathbf{v}$ and angular velocity $\boldsymbol{\omega}$, the Doppler velocity observed at a point in direction $\mathbf{d}_i = [\cos\phi_i \cos\theta_i, \cos\phi_i \sin\theta_i, \sin\phi_i]^T$ is:

$$
v_{d,i} = -\mathbf{d}_i^T (\mathbf{v} + \boldsymbol{\omega} \times \mathbf{p}_i) + n_i
$$

where $\mathbf{p}_i = r_i \mathbf{d}_i$ is the 3D position of the point. Using only static points (after removing moving objects), $(\mathbf{v}, \boldsymbol{\omega})$ can be estimated from this set of equations.

```python
import numpy as np

def radar_ego_velocity(radar_points, doppler_velocities):
    """
    Estimate ego-velocity from radar Doppler measurements.

    radar_points: (N, 3) — 3D coordinates of each point (r*d_i)
    doppler_velocities: (N,) — observed Doppler velocity of each point

    Returns:
        v_ego: (3,) — ego linear velocity
    """
    # Direction vectors (unit vectors) for each point
    norms = np.linalg.norm(radar_points, axis=1, keepdims=True)
    directions = radar_points / (norms + 1e-8)  # (N, 3)

    # v_d = -d^T @ v_ego  (simple case: ignoring angular velocity)
    # => A @ v_ego = b, where A = -directions, b = doppler_velocities
    A = -directions
    b = doppler_velocities

    # After removing dynamic objects via RANSAC, apply least squares
    # Simplified version: least squares over the full data
    v_ego, residuals, _, _ = np.linalg.lstsq(A, b, rcond=None)

    return v_ego
```

### 8.4.3 4D Radar + Camera Fusion

The combination of 4D radar and camera is drawing attention as a compelling alternative for "LiDAR-free" autonomous driving. The complementarity of the two sensors is as follows:

| Property | Camera | 4D Radar |
|------|--------|----------|
| Resolution | Very high | Low |
| Adverse weather | Weak | Robust |
| Direct depth measurement | ✗ | ✓ |
| Direct velocity measurement | ✗ | ✓ |
| Semantic understanding | Strong | Weak |
| Cost | Very low | Low |

Fusion approaches:
- **Early fusion**: Project radar points into the image and use them as sparse depth cues. Used as scale anchors for monocular depth estimation.
- **Mid-level fusion**: Combine camera features and radar features inside a network. Fusion in the BEV (Bird's Eye View) space is common.
- **Late fusion**: Detect objects independently with each sensor and then combine the results.

### 8.4.4 Boreas Benchmark

[Boreas](https://arxiv.org/abs/2203.10168) (Burnett et al., 2023) is a multi-sensor dataset collected under diverse weather conditions (clear, rain, snow), and it is particularly important for benchmarking radar odometry. It simultaneously mounts a camera, LiDAR (Velodyne Alpha Prime), and 4D radar (Navtech CIR304-H), and repeatedly drives the same route across different times/seasons, making it useful for long-term localization research as well.

---

## 8.5 Multi-Robot / Decentralized Fusion

Moving beyond single-robot fusion, the problem of having **multiple robots cooperatively perceive the environment** is a level more difficult. This is because communication constraints, the absence of a common relative reference frame, and the difficulty of data association are added.

### 8.5.1 Core Challenges of Multi-Robot SLAM

1. **Inter-Robot Relative Pose**: Each robot runs SLAM in its own local frame. To merge the maps of two robots, their relative coordinate transformation must first be known. This is solved by cross-robot place recognition plus geometric verification.

2. **Communication Constraint**: Transmitting the full map or raw sensor data is often impossible due to bandwidth limits. Thus **what information to compress and share** is a key design decision.

3. **Distributed Optimization**: Collecting all data at a central server for optimization has communication-bottleneck and single-point-of-failure problems. It is desirable for each robot to perform local optimization in a distributed fashion, exchanging only limited information with neighboring robots.

### 8.5.2 Kimera-Multi

[Kimera-Multi](https://arxiv.org/abs/2106.14386) (Rosinol et al., 2021) is a distributed multi-robot SLAM system developed by MIT's SPARK Lab.

**Architecture**:
- Each robot runs Kimera and performs local metric-semantic SLAM
- Upon rendezvous between robots, common places are detected by **DBoW2**-based inter-robot place recognition
- Detected inter-robot loop closures are incorporated into the distributed pose graph optimization
- A **GNC (Graduated Non-Convexity)** solver robustly rejects outlier loop closures

**Distributed optimization**: Each robot maintains its own pose graph and exchanges only inter-robot factors with neighboring robots. A distributed optimization algorithm such as Riemannian block-coordinate descent is used to reach convergence.

### 8.5.3 Swarm-SLAM

[Swarm-SLAM](https://arxiv.org/abs/2301.06230) (Lajoie et al., 2024) is a distributed SLAM for large-scale robot swarms that places particular emphasis on communication efficiency.

**Core design**:
- **Place recognition descriptor exchange**: Only place recognition descriptors (NetVLAD, Scan Context, etc.), rather than the full map, are exchanged to minimize bandwidth
- **Inter-robot loop closure**: Candidates are found by descriptor matching, and only a minimal amount of geometric information (feature points or a point cloud) is exchanged for verification
- **Peer-to-peer communication between neighboring robots**: Direct communication between adjacent robots without a central server
- **LiDAR/Visual/Multimodal support**: Robots with camera-only, LiDAR-only, or mixed sensor configurations can participate simultaneously

```python
# Conceptual implementation of distributed pose graph optimization
import numpy as np

class DistributedPoseGraphNode:
    """
    A single robot node in the distributed pose graph.
    Each robot maintains its own local graph and
    exchanges only inter-robot factors with neighbors.
    """
    def __init__(self, robot_id):
        self.robot_id = robot_id
        self.local_poses = []           # Own poses (local frame)
        self.local_factors = []          # Local odometry factors
        self.inter_robot_factors = []    # Loop closure factors with other robots
        self.neighbor_info = {}          # Boundary info received from neighbors

    def add_odometry(self, delta_pose, covariance):
        """Add a local odometry factor"""
        self.local_factors.append({
            'type': 'odom',
            'from': len(self.local_poses) - 1,
            'to': len(self.local_poses),
            'measurement': delta_pose,
            'covariance': covariance
        })
        self.local_poses.append(self.local_poses[-1] @ delta_pose)

    def add_inter_robot_factor(self, other_robot_id, other_pose_idx,
                                relative_pose, covariance):
        """Add a loop closure factor with another robot"""
        self.inter_robot_factors.append({
            'type': 'inter_robot',
            'robot': other_robot_id,
            'local_idx': len(self.local_poses) - 1,
            'remote_idx': other_pose_idx,
            'measurement': relative_pose,
            'covariance': covariance
        })

    def exchange_boundary_info(self, neighbor_node):
        """
        Exchange boundary information (estimates and covariances of
        boundary variables) with a neighbor node. Only variables related
        to inter-robot factors are exchanged, not the full map.
        """
        boundary_poses = []
        for factor in self.inter_robot_factors:
            if factor['robot'] == neighbor_node.robot_id:
                idx = factor['local_idx']
                boundary_poses.append({
                    'idx': idx,
                    'pose': self.local_poses[idx]
                })
        neighbor_node.neighbor_info[self.robot_id] = boundary_poses
```

---

## 8.6 System Design in Practice

Beyond theory and algorithms, we address the practical problems encountered when designing and deploying a real multi-sensor fusion system.

### 8.6.1 Sensor Suite Selection Guide

The choice of sensor suite is dictated by the operational environment:

| Environment | Recommended minimum | Optional additional sensors |
|------|---------------|---------------|
| Indoor (office/warehouse) | Camera + IMU | LiDAR (for precise mapping) |
| Urban autonomous driving | Camera + LiDAR + IMU + GNSS | 4D Radar, Wheel Odom |
| Off-road/outdoor | LiDAR + IMU + GNSS | Camera (semantics), Radar |
| Underground/tunnel | LiDAR + IMU | Camera, UWB |
| Underwater | IMU + DVL (Doppler Velocity Log) | Sonar, Pressure |
| Aerial/drone | Camera + IMU + GNSS | LiDAR (for mapping) |
| Adverse weather (rain/snow) | Radar + IMU | Camera, LiDAR |

**Example configurations by budget**:
- **Under $500**: Stereo Camera + IMU (Intel RealSense D435i)
- **Under $2,000**: + 2D LiDAR (RPLidar)
- **Under $10,000**: + 3D LiDAR (Livox Mid-360) + GNSS RTK
- **$30,000+**: Multi-LiDAR + Multi-Camera + 4D Radar + GNSS RTK

### 8.6.2 Timing Architecture (Time Synchronization Design)

In a multi-sensor system, **time synchronization** is a decisive factor for accuracy. On a vehicle moving at 100 km/h, a 1 ms time error corresponds to roughly a 2.8 cm position error.

**Hardware Sync**:

The most precise method is to use hardware triggers:

- **PPS (Pulse Per Second)**: The GNSS receiver outputs a precise pulse once per second. This pulse is wired into the synchronization input of other sensors. Precision: ~50 ns.
- **PTP (Precision Time Protocol, IEEE 1588)**: Ethernet-based time synchronization. Supported by LiDARs (Velodyne, Ouster, etc.). Precision: ~μs.
- **External trigger**: A microcontroller simultaneously triggers the camera shutter and captures the IMU timestamp.

**Software Sync**:

When hardware synchronization is not possible, the time offset is estimated in software:

- **Kalibr approach**: Represent the continuous-time trajectory with a B-spline and include the inter-sensor time offset as an optimization variable, estimating everything jointly.
- **Correlation-based**: Compute the cross-correlation between the motion estimates of two sensors to estimate the time delay.

$$
\hat{\tau} = \arg\max_{\tau} \int \mathbf{a}_{\text{IMU}}(t) \cdot \dot{\mathbf{v}}_{\text{camera}}(t + \tau) \, dt
$$

```python
import numpy as np
from scipy.signal import correlate

def estimate_time_offset(timestamps_a, signal_a, timestamps_b, signal_b, max_offset_ms=100):
    """
    Estimate the time offset between two sensor signals via cross-correlation.

    Example: IMU angular velocity vs. inter-frame rotation rate of the camera
    """
    # Resample to a common timeline (1 kHz)
    dt = 0.001  # 1 ms
    t_common = np.arange(
        max(timestamps_a[0], timestamps_b[0]),
        min(timestamps_a[-1], timestamps_b[-1]),
        dt
    )
    sig_a = np.interp(t_common, timestamps_a, signal_a)
    sig_b = np.interp(t_common, timestamps_b, signal_b)

    # Remove the mean
    sig_a -= np.mean(sig_a)
    sig_b -= np.mean(sig_b)

    # Cross-correlation
    correlation = correlate(sig_a, sig_b, mode='full')
    lags = np.arange(-len(sig_b) + 1, len(sig_a)) * dt

    # Find the maximum correlation within the max_offset range
    mask = np.abs(lags) <= max_offset_ms / 1000
    valid_corr = correlation[mask]
    valid_lags = lags[mask]

    best_idx = np.argmax(valid_corr)
    estimated_offset = valid_lags[best_idx]

    return estimated_offset  # in seconds

# Example: IMU gyro Z-axis vs. Camera rotation rate
# offset = estimate_time_offset(imu_times, gyro_z, cam_times, cam_rotation_rate)
```

### 8.6.3 Failure Modes and Degradation Handling

In real systems, sensors inevitably fail. A robust system must achieve **graceful degradation** — that is, it must continue to operate with the remaining sensors, even at reduced performance, when one sensor fails.

**Key failure modes and responses**:

| Failure mode | Symptom | Detection | Response |
|-----------|------|-----------|------|
| Camera over-/under-exposure | Entire image is bright or dark | Histogram analysis | Disable camera factor, operate with LIO only |
| LiDAR geometric degeneracy | Long corridor, wide flat plane | Eigenvalue analysis of the information matrix | Relax LiDAR constraint on the affected DoF, compensate with VIO |
| IMU saturation | Measurement range exceeded under high-speed impact | Detect ADC maximum values | Increase IMU preintegration uncertainty for the affected interval |
| GNSS multipath | Large error due to reflections from buildings | RAIM, residual check | Increase the covariance of the affected GNSS factor or remove it |
| Total sensor dropout | No data received | Watchdog timer | Disable all factors for the affected sensor |

**Detecting LiDAR geometric degeneracy**:

In LiDAR scan matching, geometric degeneracy can be detected by eigenvalue analysis of the information matrix (Hessian) $\mathbf{H} = \mathbf{J}^T \mathbf{J}$. If the eigenvalue along one direction is significantly smaller than those along the others, the constraint in that direction is weak.

$$
\mathbf{H} = \mathbf{U} \boldsymbol{\Lambda} \mathbf{U}^T, \quad \lambda_{\min} / \lambda_{\max} < \epsilon \Rightarrow \text{degenerate}
$$

For instance, in a long corridor the constraint along the corridor axis becomes weak, so the LiDAR constraint in that direction is relaxed and complemented by the camera's optical flow.

```python
import numpy as np

def check_lidar_degeneracy(jacobian, threshold=0.01):
    """
    Check for geometric degeneracy using the eigenvalues of the Hessian
    from LiDAR scan matching.

    jacobian: (m, 6) — 6-DoF Jacobian of m point-to-plane residuals
    threshold: threshold for the min-to-max eigenvalue ratio

    Returns:
        is_degenerate: bool
        degenerate_directions: (k, 6) — eigenvectors of degenerate directions
        eigenvalues: (6,) — eigenvalues of the information matrix
    """
    # Information matrix (approximate Hessian)
    H = jacobian.T @ jacobian

    # Eigen-decomposition
    eigenvalues, eigenvectors = np.linalg.eigh(H)

    # Eigenvalue ratio check
    ratio = eigenvalues / (eigenvalues.max() + 1e-10)
    degenerate_mask = ratio < threshold

    is_degenerate = np.any(degenerate_mask)
    degenerate_directions = eigenvectors[:, degenerate_mask].T

    if is_degenerate:
        print(f"[Warning] Geometric degeneracy detected!")
        print(f"  Eigenvalues: {eigenvalues}")
        print(f"  Number of degenerate directions: {degenerate_mask.sum()}")

    return is_degenerate, degenerate_directions, eigenvalues


def adaptive_fusion_weight(lidar_eigenvalues, camera_track_quality,
                            lidar_min_eig_threshold=100.0):
    """
    Adaptively adjust the camera weight according to the degree of LiDAR degeneracy.
    """
    min_eig = lidar_eigenvalues.min()

    if min_eig > lidar_min_eig_threshold:
        # LiDAR is sufficiently constrained → default weights
        lidar_weight = 1.0
        camera_weight = 0.3
    else:
        # LiDAR is degenerate → increase the camera weight
        decay = min_eig / lidar_min_eig_threshold
        lidar_weight = decay
        camera_weight = 1.0

    return lidar_weight, camera_weight
```

### 8.6.5 Notable Recent Research (2024-2025)

- **[Gaussian-LIC (Lang et al., ICRA 2025)](https://arxiv.org/abs/2404.06926)**: A system that integrates 3D Gaussian Splatting into tightly-coupled LiDAR-Inertial-Camera SLAM. By fusing the precise geometric information from the LiDAR with the camera's texture using a Gaussian representation, it achieves photo-realistic scene reconstruction concurrently with SLAM.
- **[Snail-Radar (Huai et al., IJRR 2025)](https://arxiv.org/abs/2407.11705)**: A large-scale diversity benchmark for evaluating 4D radar SLAM. It systematically compares 4D radar-based odometry/SLAM algorithms across diverse environments (indoor/outdoor, urban/suburban) and platforms.

### 8.6.4 System Design Checklist

Items that must always be checked when designing a real multi-sensor fusion system:

**Calibration**:
- [ ] Extrinsic calibration completed for every sensor pair
- [ ] Time-synchronization offsets measured/estimated
- [ ] Calibration results verified for reproducibility (at least three repetitions)
- [ ] Mechanism in place for online calibration drift correction

**Data flow**:
- [ ] Each sensor's data rate matches the system's processing rate
- [ ] Temporal alignment method between sensors is finalized
- [ ] Buffer sizes and latency are analyzed

**Robustness**:
- [ ] Failure modes of each sensor identified
- [ ] Degradation handling strategy established
- [ ] Outlier rejection mechanisms (robust kernel, chi-square test)
- [ ] Tested under extreme conditions (darkness, rain, vibration, geometric degeneracy)

**Performance**:
- [ ] Target accuracy (ATE/RPE) defined
- [ ] Real-time constraints satisfied (worst-case latency)
- [ ] Memory usage (accumulation over long-duration operation)
- [ ] CPU/GPU utilization

---

## Chapter 8 Summary

Multi-sensor fusion architectures are broadly classified as loosely/tightly/ultra-tightly coupled, and in modern robotics **tightly coupled** is the mainstream choice. Triple Camera+LiDAR+IMU fusion has reached a mature stage through systems such as R3LIVE, LVI-SAM, and FAST-LIVO2, each representing a distinct design philosophy — dual subsystem, factor graph, and sequential update, respectively.

GNSS integration provides a global coordinate anchor that resolves long-term drift, and 4D radar is rapidly rising thanks to its unique advantages of adverse-weather robustness and direct velocity measurement. Multi-robot fusion faces the core challenges of distributed optimization and cross-robot place recognition under communication constraints, with Kimera-Multi and Swarm-SLAM leading this area.

Finally, in practical system design, sensor selection, time synchronization, and failure-mode handling are as important as the algorithms themselves, and the engineering capability to address these systematically determines successful deployment.

The odometry/fusion systems covered in Ch.6-8 are highly accurate locally, but drift accumulates over long-duration operation. To correct this drift, the ability to "recognize places previously visited" is required. The next chapter addresses this key component, **Place Recognition**.
