# Ch.8 — Multi-Sensor Fusion Architectures

> A design discipline that goes beyond individual odometry to address **how multiple sensors are integrated**.
> Combining Visual Odometry and LiDAR Odometry in one system requires choices about coupling level and estimator structure.

---

## 8.1 Taxonomy of Fusion Architectures

The first decision in designing a multi-sensor fusion system is **at what level the sensor data will be combined**. The depth of this coupling determines the system's complexity, performance, and failure modes.

### 8.1.1 Loosely Coupled

Each sensor is viewed as an independent "expert." Each expert produces an independent estimate (pose, velocity, etc.) from its own data, and a higher-level stage then combines the results.

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
- Because fusion occurs only after each subsystem has already compressed its measurements into an output estimate, the full benefit of complementary interaction between sensors cannot be exploited. For example, LiDAR's precise geometric information can resolve a camera's scale ambiguity, but in a loosely coupled design this interaction is limited.
- The consistency of the covariances reported by each subsystem is not guaranteed. If a subsystem reports an overly optimistic covariance, the fusion result is distorted.

### 8.1.2 Tightly Coupled

All sensors' **raw measurements** directly constrain a single shared state. Instead of running "experts" and merging their outputs, the raw data enters the state estimate itself. What defines the level is where the measurements enter, not whether there is only one optimizer. A design in which two subsystems update one shared state (R3LIVE) or one that applies sequential updates (FAST-LIVO2) is tightly coupled as well. Even when two subsystems keep separate graphs, as in LVI-SAM, the design is classified at the same level if they exchange measurements (LiDAR depth) and state (initial guesses) directly with each other.

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

**Representative systems**: [LIO-SAM (Shan et al. 2020)](https://arxiv.org/abs/2007.00258) (LiDAR+IMU+GPS), VINS-Mono (Camera+IMU), [R3LIVE (Lin et al. 2022)](https://arxiv.org/abs/2109.07982) (Camera+LiDAR+IMU).

### 8.1.3 Ultra-Tightly Coupled (Signal-Level Coupling)

Not the sensor's measurements but the **signals themselves** are combined. This is the most extreme form of integration.

A representative example is GNSS-INS ultra-tight coupling. A typical tightly coupled GNSS-INS system combines receiver-derived pseudoranges with the inertial navigation solution, but in the ultra-tight approach the position and velocity estimated by the INS directly aid the GNSS receiver's code/carrier tracking loops. Doing so lets the receiver keep tracking GNSS signals longer even in weak-signal environments (urban canyons, right after entering indoors).

$$
\text{NCO frequency} = f_{\text{nominal}} + \Delta f_{\text{INS-aided}}
$$

Here the frequency of the NCO (Numerically Controlled Oscillator) is corrected by the Doppler shift predicted by the INS, widening the receiver's tracking range.

**Implementation boundary**: ultra-tight coupling requires access to correlators or tracking loops and cannot be built from a generic measurement API alone. It appears in integrated GNSS/INS products and research platforms with receiver SDK, FPGA, or firmware access; ordinary robotics projects more readily use tightly coupled raw pseudorange, carrier, and Doppler measurements.

### 8.1.4 Comparison of the Three Levels

```
Measurement flow:

[Loosely]    Sensor A → Subsystem A → Pose A ─┐
                                              ├→ Fusion → Final pose
             Sensor B → Subsystem B → Pose B ─┘

[Tightly]    Sensor A → raw meas. A ──┐
                                      ├→ Shared-state estimator → Final pose
             Sensor B → raw meas. B ──┘   (a single optimizer, sequential updates,
                                           and two subsystems sharing one state
                                           all belong to this level)

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
P_visual = np.diag([0.1, 0.05])       # Visual precision differs per axis (less precise along x)

x_fused, P_fused = loosely_coupled_fusion(x_lidar, P_lidar, x_visual, P_visual)
print(f"LiDAR:  {x_lidar}, P_diag: {np.diag(P_lidar)}")
print(f"Visual: {x_visual}, P_diag: {np.diag(P_visual)}")
print(f"Fused:  {x_fused}, P_diag: {np.diag(P_fused)}")
# The fused result is closer to the LiDAR estimate (since its covariance is smaller)
```

<!-- DEMO: fusion_architecture_comparison.html -->

---

## 8.2 Camera + LiDAR + IMU Fusion

Camera, LiDAR, and IMU together provide texture and color, 3D range, and high-rate inertia. Cost, power, weather, range, and correlated failures still matter; installing all three does not by itself guarantee robustness:

| Situation | Camera | LiDAR | IMU |
|------|--------|-------|-----|
| Dark environment | ✗ | ✓ | ✓ |
| Textureless wall | ✗ | ✓ | ✓ |
| Geometric degeneracy (long corridor) | ✓ | ✗ | ✓ |
| High-speed rotation | ✗ | ✗ | ✓ |
| Scale observability | ✗ (monocular) | ✓ | ✓ (acceleration is in metric units; requires motion excitation) |
| Color/semantics | ✓ | ✗ | ✗ |

The following systems integrate these three sensors.

### 8.2.1 R3LIVE / R3LIVE++

R3LIVE (Lin et al., 2022) is a system that tightly couples two subsystems: LiDAR-Inertial Odometry (LIO) and Visual-Inertial Odometry (VIO).

R3LIVE adopts a **dual-subsystem** architecture. The LIO subsystem is responsible for geometry and the VIO subsystem is responsible for photometric (texture) information, and the two subsystems are tightly coupled by **sharing a single state**.

```
LiDAR scan ──→ [LIO subsystem] ──→ state update (geometry)
                     ↓                       ↓
                 IMU data ───────────→ shared state vector
                     ↑                       ↑
Camera image ──→ [VIO subsystem] ──→ state update (photometric)
```

The LIO subsystem directly registers raw LiDAR points to an ikd-Tree-based map with point-to-plane matching, as FAST-LIO2 does. An iterated EKF updates the state.

The VIO subsystem distinguishes R3LIVE. A typical VIO minimizes the reprojection error of feature points, whereas R3LIVE uses the **photometric error**. Each point in the 3D map built by LIO is assigned an RGB color. When a new camera image arrives, these map points are projected into the image to minimize the **difference between the observed color and the color stored in the map**:

$$
\mathbf{r}^{\text{photo}}_i = \mathbf{I}(\pi(\mathbf{T}_{CW} \mathbf{p}^W_i)) - \mathbf{c}_i^{\text{map}}
$$

Here $\mathbf{I}(\cdot)$ is the pixel intensity of the image, $\pi(\cdot)$ is the 3D→2D projection function, $\mathbf{T}_{CW}$ is the world-to-camera transformation, $\mathbf{p}^W_i$ is the 3D coordinate of a map point, and $\mathbf{c}_i^{\text{map}}$ is the color stored in the map for that point.

R3LIVE's shared state can continue receiving valid updates when one modality temporarily contributes fewer usable residuals. The duration of visual updates during LiDAR blockage depends on visibility of the existing colored map; LIO during dark imagery depends on LiDAR geometry and IMU quality. The paper demonstrates online colored 3D mapping with this structure.

### 8.2.2 LVI-SAM

[LVI-SAM](https://arxiv.org/abs/2104.10831) (Shan et al., 2021) is an extension of LIO-SAM that couples a Visual-Inertial subsystem and a LiDAR-Inertial subsystem **bidirectionally**.

**Bidirectional coupling**:

- **VIS → LIS direction**: The pose estimated by the Visual-Inertial subsystem is used as the initial guess for LiDAR scan matching. Especially when the LiDAR alone yields an inaccurate initial guess (high-speed rotation, featureless environments), VIS provides the initial guess and helps LiDAR registration converge.

- **LIS → VIS direction**: assign LiDAR depth to visual features with valid spatial and temporal correspondence. With correct extrinsics, synchronization, visibility, and occlusion handling, this provides a metric-depth prior without waiting for triangulation parallax.

```
         ┌─── VIS initial pose ───→ LIS initial guess
         │                            │
  [Visual-Inertial]            [LiDAR-Inertial]
         │                            │
         └←── LiDAR depth ────────────┘

              ↓ LiDAR·IMU factors ↓
           [Factor Graph (GTSAM/iSAM2)]
                     ↓
              Final optimized pose
```

**Factor graph design**: LVI-SAM consists of two subsystems, a visual-inertial system (VIS) and a LiDAR-inertial system (LIS), and the factor graph is maintained by the LIS, which inherits it from LIO-SAM. The factors it holds are:
- IMU preintegration factor (between successive keyframes)
- LiDAR odometry factor (scan matching result)
- GPS factor (when available)
- Loop closure factor (upon revisit detection)

Rather than inserting its own odometry into this graph as a factor, the VIS couples in by supplying the LIS with an initial guess and by taking LiDAR depth to fix its own scale.

### 8.2.3 FAST-LIVO / FAST-LIVO2

[FAST-LIVO2](https://arxiv.org/abs/2408.14035) (Zheng et al., 2024) is a direct Camera+LiDAR+IMU fusion system developed by the FAST-LIO2 team (HKU MARS Lab). "Direct" means raw data is used without feature extraction.

**Design 1 — Sequential Update**:

Measurements from heterogeneous sensors differ greatly in number and in nature. LiDAR provides one point-to-plane residual per point, while the camera provides one photometric residual per patch pixel; the two measurement sets differ in size, units, and noise magnitude, and the original paper cites this dimension mismatch as the reason for choosing sequential updates.

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

**Design 2 — Unified adaptive voxel map**:

FAST-LIVO2 uses a single voxel map based on a hash table plus an octree. The LiDAR module builds the geometric structure (3D coordinates, normal vectors), and the Visual module attaches image patches to the same map points. Geometry and texture are thus managed consistently within a single map.

**Design 3 — Affine warping using LiDAR normals**:

When comparing image patches in a camera direct method, affine warping that accounts for surface tilt improves accuracy. FAST-LIVO2 leverages the planar normal vectors extracted from the LiDAR to perform accurate affine warping without any separate normal estimation. This illustrates LiDAR-camera complementarity.

**Design 4 — Real-time exposure compensation**:

In environments with rapidly changing illumination (entering/exiting a tunnel), FAST-LIVO2 estimates the exposure time online and corrects the photometric error accordingly.

```python
import numpy as np

def sequential_ekf_update(x_pred, P_pred, z_lidar, H_lidar, R_lidar, z_cam, H_cam, R_cam):
    """
    Sequential EKF update in the order LiDAR → Camera.
    Mathematically equivalent to a simultaneous update, but the dimension
    mismatch of the heterogeneous measurements need not be handled at once.

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
| Camera processing | Direct (photometric) | Feature-based (Shi-Tomasi + KLT) | Direct (photometric) |
| Map representation | ikd-Tree + RGB | Voxel map | Hash+Octree voxel map |
| Feature extraction | Not required | Required (LiDAR edge/planar, visual corners) | Not required |
| GPS integration | None | Integrated as factor | None |
| Loop closure | None | Integrated as factor | None |
| Embedded validation | Benchmark on target hardware | Benchmark on target hardware | ARM implementations reported; revalidate under the target setup |

**Criteria for narrowing the candidates**:
- Evaluate LVI-SAM when integrated loop closure and GPS factors are required. Also check the sensor configuration, ROS dependencies, and revisit performance on the target data.
- Evaluate R3LIVE when direct photometric updates and colored mapping are required. Compare color, trajectory, and map-quality metrics in the target scenes.
- Evaluate FAST-LIVO2 as an ARM deployment candidate, but measure throughput, worst-case latency, and memory at the target sensor resolution and configuration.
- In environments with scarce visual or LOAM features, include direct methods such as R3LIVE and FAST-LIVO2 as candidates, then validate photometric calibration and sensitivity to dynamic objects.

---

## 8.3 GNSS Integration

GNSS (Global Navigation Satellite System) provides an absolute position reference in a global coordinate frame. The IMU, LiDAR odometry, and visual odometry discussed here primarily provide **relative** motion, so drift accumulates over long-duration operation. GNSS can serve as an anchor that corrects this drift. Surveyed landmarks, UWB beacons, and motion-capture systems can also provide absolute references, so GNSS is not the only possible global sensor.

### 8.3.1 GNSS Factor in the Factor Graph (LIO-SAM Approach)

[LIO-SAM](https://arxiv.org/abs/2007.00258) (Shan et al., 2020) connects each GNSS position report to a factor-graph pose node as a **unary factor**:

$$
\mathbf{r}^{\text{GPS}}_i = \mathbf{T}_{\text{ENU→map}} \cdot \mathbf{p}^{\text{ENU}}_{\text{GPS}} - \mathbf{p}^{\text{map}}_i - \mathbf{R}^{\text{map}}_i \cdot \mathbf{l}_{\text{antenna}}
$$

where:
- $\mathbf{p}^{\text{ENU}}_{\text{GPS}}$ is the ENU coordinate reported by the GNSS
- $\mathbf{T}_{\text{ENU→map}}$ is the transformation from the ENU frame to the SLAM map frame. The equation above brings the GNSS position into the map frame for comparison, so this transformation is applied as is; the $\mathbf{T}^{-1}$ form is what one writes when using the opposite direction, $\mathbf{T}_{\text{map→ENU}}$
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
    R_gnss: covariance of the GNSS position (3, 3) — put squared standard deviations
            on the diagonal. Horizontally, sigma_h = HDOP * sigma_uere; vertically,
            use VDOP. That is, diag(sigma_E**2, sigma_N**2, sigma_U**2).
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

### 8.4.1 Radar and 4D Imaging

Traditionally, automotive radar was considered unsuitable for SLAM/odometry because of its low resolution. However, the emergence of **4D imaging radar** is changing the situation.

**What is 4D radar**: Whereas conventional automotive radars measured three quantities — range, Doppler velocity, and azimuth — 4D imaging radar adds **elevation** to produce a 3D point cloud. Its point count is lower than that of LiDAR (hundreds to thousands of points vs. hundreds of thousands).

Radar provides the following properties:

1. **Adverse-weather tolerance**: mm-wave radar is often less affected by fog, rain, and snow than visible cameras or some LiDARs. Heavy precipitation, wet-road multipath, attenuation, and clutter remain, so performance must be tested by condition.

2. **Direct radial-velocity measurement**: FMCW Doppler gives the line-of-sight component of relative velocity from the phase change across the several chirps within one frame. A single chirp alone yields only range information, with velocity not yet separated out. It does not directly provide a point's full 3D velocity or an object's motion; multiple directions, time, tracking, or another sensor are needed.

3. **Different cost structure**: mass-market radar chipsets can be inexpensive, but imaging-radar and LiDAR module prices overlap depending on channels, antennas, compute, and production volume. Compare current quotes against required performance rather than assuming a fixed ratio.

### 8.4.2 Radar Odometry

The number of papers on odometry using 4D radar has grown rapidly since 2022. These methods directly use radar Doppler measurements for ego-motion estimation.

Each measurement point of an FMCW radar provides $(r, \theta, \phi, v_d)$ — range, azimuth, elevation, Doppler velocity. Given the robot's linear velocity $\mathbf{v}$ and angular velocity $\boldsymbol{\omega}$, the Doppler velocity observed at a point in direction $\mathbf{d}_i = [\cos\phi_i \cos\theta_i, \cos\phi_i \sin\theta_i, \sin\phi_i]^T$ is:

$$
v_{d,i} = -\mathbf{d}_i^T \mathbf{v}_S + n_i
$$

Here $\mathbf{v}_S$ is the linear velocity of the radar phase center. If static returns span enough directions, the equations constrain $\mathbf{v}_S$. Point Doppler from a single monostatic radar does not independently determine angular velocity about the sensor origin; transform to a body origin with a known lever arm and angular rate from an IMU or scan registration.

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

Combining 4D radar with cameras can produce a "LiDAR-free" autonomous-driving system. The two sensors complement each other as follows:

| Property | Camera | 4D Radar |
|------|--------|----------|
| Resolution | Very high | Low |
| Adverse weather | Weak | Robust |
| Direct depth measurement | ✗ | ✓ |
| Radial-velocity measurement | ✗ | ✓ |
| Semantic understanding | Strong | Weak |
| Cost | Depends on lenses, camera count, and compute | Depends on antennas, channels, and compute |

Fusion uses three approaches:
- **Early fusion**: Project radar points into the image and use them as sparse depth cues. Used as scale anchors for monocular depth estimation.
- **Mid-level fusion**: Combine camera features and radar features inside a network. Fusion in the BEV (Bird's Eye View) space is common.
- **Late fusion**: Detect objects independently with each sensor and then combine the results.

### 8.4.4 Boreas Benchmark

[Boreas](https://arxiv.org/abs/2203.10168) (Burnett et al., 2023) repeats multi-sensor routes across clear, rainy, snowy, and seasonal conditions. It synchronizes cameras, a Velodyne Alpha Prime LiDAR, and a Navtech CIR304-H **planar scanning radar**, among other sensors. That Navtech unit is not an automotive 4D imaging radar with elevation, so Boreas radar-odometry results should not be presented as 4D-radar performance.

---

## 8.5 Multi-Robot / Decentralized Fusion

For **multiple robots to perceive the environment cooperatively**, they must address communication constraints, the absence of a common reference frame, and data association.

### 8.5.1 Challenges of Multi-Robot SLAM

1. **Inter-Robot Relative Pose**: Each robot runs SLAM in its own local frame. To merge the maps of two robots, their relative coordinate transformation must first be known. This is solved by cross-robot place recognition plus geometric verification.

2. **Communication Constraint**: Transmitting the full map or raw sensor data is often impossible due to bandwidth limits. The system must therefore choose **what information to compress and share**.

3. **Distributed Optimization**: Collecting all data at a central server for optimization has communication-bottleneck and single-point-of-failure problems. It is desirable for each robot to perform local optimization in a distributed fashion, exchanging only limited information with neighboring robots.

### 8.5.2 Kimera-Multi

[Kimera-Multi](https://arxiv.org/abs/2106.14386) (Tian et al., 2021) is a distributed multi-robot SLAM system developed by MIT's SPARK Lab.

**Architecture**:
- Each robot runs Kimera and performs local metric-semantic SLAM
- Upon rendezvous between robots, common places are detected by **DBoW2**-based inter-robot place recognition
- Detected inter-robot loop closures are incorporated into the distributed pose graph optimization
- A **GNC (Graduated Non-Convexity)** solver robustly rejects outlier loop closures

**Distributed optimization**: Each robot maintains its own pose graph. When a loop is detected, the inter-robot factor is shared; from then on a distributed optimizer such as Riemannian block-coordinate descent converges by exchanging, at every iteration, the current estimates of the separator poses it shares with its neighbors. Sharing a constraint once and repeatedly exchanging optimization variables are two different kinds of communication.

### 8.5.3 Swarm-SLAM

[Swarm-SLAM](https://arxiv.org/abs/2301.06230) (Lajoie et al., 2024) is a distributed SLAM for large-scale robot swarms that places particular emphasis on communication efficiency.

**Swarm-SLAM design**:
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
    exchanges inter-robot factors and separator pose estimates with neighbors.
    """
    def __init__(self, robot_id):
        self.robot_id = robot_id
        self.local_poses = [np.eye(4)]  # Own poses (local frame); seeded with the identity pose at the origin
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

Designing and deploying a real multi-sensor fusion system also introduces problems beyond theory and algorithms.

### 8.6.1 Sensor Suite Selection Guide

The following table is a starting point. Final selection depends on the operational environment, safety requirements, range, power and mass, and failure scenarios:

| Environment | Recommended minimum | Optional additional sensors |
|------|---------------|---------------|
| Indoor (office/warehouse) | Camera + IMU | LiDAR (for precise mapping) |
| Urban autonomous driving | Camera + LiDAR + IMU + GNSS | 4D Radar, Wheel Odom |
| Off-road/outdoor | LiDAR + IMU + GNSS | Camera (semantics), Radar |
| Underground/tunnel | LiDAR + IMU | Camera, UWB |
| Underwater | IMU + DVL (Doppler Velocity Log) | Sonar, Pressure |
| Aerial/drone | Camera + IMU + GNSS | LiDAR (for mapping) |
| Adverse weather (rain/snow) | Radar + IMU | Camera, LiDAR |

**Costing sequence**: sensor prices vary by date, region, and volume. First specify the required measurement, accuracy, rate, and FoV. Then price the complete system: sensors, cables, synchronization, GNSS corrections, compute, mounts, spares, and integration time. A low-cost prototype may begin with camera and IMU; add 2D or 3D LiDAR, RTK GNSS, radar, or redundancy only for failure modes and error budgets established by testing.

### 8.6.2 Timing Architecture (Time Synchronization Design)

In a multi-sensor system, **time synchronization** directly affects accuracy. On a vehicle moving at 100 km/h, a 1 ms time error corresponds to roughly a 2.8 cm position error.

**Hardware Sync**:

A shared clock or trigger can make the latency path clearer than software receive timestamps. Pulse accuracy and measurement-timestamp accuracy are not identical, so validate the path end to end:

- **PPS (Pulse Per Second)**: use a GNSS receiver's second-boundary pulse as a common clock reference. Check the receiver time-pulse specification, cable and input delay, and timestamp reference point.
- **PTP (Precision Time Protocol, IEEE 1588)**: synchronize Ethernet clocks. Actual error depends on hardware timestamps, switches, PTP profile, path asymmetry, and lock state.
- **External trigger**: A microcontroller simultaneously triggers the camera shutter and captures the IMU timestamp.

**Software Sync**:

When hardware synchronization is not possible, the time offset is estimated in software:

- **Kalibr approach**: Represent the continuous-time trajectory with a B-spline and include the inter-sensor time offset as an optimization variable, estimating everything jointly.
- **Correlation-based**: Estimate the time delay from the cross-correlation between two signals in which the sensors observed the same motion. Using acceleration requires removing gravity from the IMU measurement and bringing both signals into the same frame, and with a monocular camera the scale of the velocity is undetermined. Practice therefore mostly compares gyro angular rate against the camera's rotation rate, which needs none of those three conditions.

$$
\hat{\tau} = \arg\max_{\tau} \int \boldsymbol{\omega}_{\text{IMU}}(t) \cdot \boldsymbol{\omega}_{\text{camera}}(t + \tau) \, dt
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

In real systems, sensors inevitably fail. A robust system must achieve **graceful degradation**: when one sensor fails, it must continue operating with the remaining sensors, even at reduced performance.

**Failure modes and responses**:

| Failure mode | Symptom | Detection | Response |
|-----------|------|-----------|------|
| Camera over-/under-exposure | Entire image is bright or dark | Histogram analysis | Disable camera factor, operate with LIO only |
| LiDAR geometric degeneracy | Long corridor, wide flat plane | Eigenvalue analysis of the information matrix | Relax LiDAR constraint on the affected DoF, compensate with VIO |
| IMU saturation | Measurement range exceeded under high-speed impact | Detect ADC maximum values | Increase IMU preintegration uncertainty for the affected interval |
| GNSS multipath | Large error due to reflections from buildings | RAIM, residual check | Increase the covariance of the affected GNSS factor or remove it |
| Total sensor dropout | No data received | Watchdog timer | Stop adding new factors. If the failure time is uncertain, down-weight or remove only the interval after the suspect point |

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

### 8.6.4 Recent Systems and Research (2024-2025)

- **[Gaussian-LIC (Lang et al., ICRA 2025)](https://arxiv.org/abs/2404.06926)**: A system that integrates 3D Gaussian Splatting into tightly-coupled LiDAR-Inertial-Camera SLAM. By fusing the precise geometric information from the LiDAR with the camera's texture using a Gaussian representation, it achieves photo-realistic scene reconstruction concurrently with SLAM.
- **[Snail-Radar (Huai et al., IJRR 2025)](https://arxiv.org/abs/2407.11705)**: A large-scale diversity benchmark for evaluating 4D radar SLAM. It systematically compares 4D radar-based odometry/SLAM algorithms across diverse environments (indoor/outdoor, urban/suburban) and platforms.

### 8.6.5 System Design Checklist

Items that must always be checked when designing a real multi-sensor fusion system:

**Calibration**:
- [ ] Extrinsic calibration completed for every sensor pair
- [ ] Time-synchronization offsets measured/estimated
- [ ] Calibration results verified for reproducibility (at least three repetitions)
- [ ] Procedure in place for monitoring and responding to calibration changes (online estimation or periodic recalibration). Online extrinsic estimation requires sufficient motion excitation for observability, so it is not a mandatory condition for every system

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

Multi-sensor fusion architectures are broadly classified as loosely/tightly/ultra-tightly coupled, and in modern robotics **tightly coupled** is the mainstream choice. Three-sensor fusion with camera, LiDAR, and IMU is implemented in systems such as R3LIVE, LVI-SAM, and FAST-LIVO2, which use a dual subsystem, factor graph, and sequential update, respectively.

GNSS integration constrains drift with a global-coordinate observation, while 4D radar can add weather tolerance and radial-velocity measurements. Multi-robot systems such as Kimera-Multi and Swarm-SLAM combine distributed estimation with cross-robot place recognition under communication constraints.

In practical system design, sensor selection, time synchronization, and failure-mode handling matter as much as the algorithms. Engineering decisions in these areas affect deployment results.

The odometry/fusion systems covered in Ch.6-8 are highly accurate locally, but drift accumulates over long-duration operation. Correcting that drift through loop closure requires the ability to recognize previously visited places: **Place Recognition**.
