# Chapter 8. Multi-Sensor Fusion Architectures

> A design discipline that goes beyond individual odometry to address how multiple sensors are integrated.
> Combining Visual Odometry and LiDAR Odometry in a single system requires deliberate choices about coupling level and estimator structure.

Visual-inertial odometry (Chapter 6) and LiDAR-inertial odometry (Chapter 7) each deliver accurate local trajectories, yet any pipeline built around a single exteroceptive modality inherits its blind spots. Cameras lose tracking in the dark or against textureless walls, while LiDAR registration degrades in geometrically featureless corridors or heavy precipitation. Closing those gaps on a shared platform starts with two architectural decisions: how deeply the sensor streams couple, and what estimator structure solves the joint state.

---

## 8.1 Taxonomy of Fusion Architectures

The first decision in designing a multi-sensor fusion system is where in the pipeline sensor data should merge—whether subsystems exchange summarized poses, pool raw measurements into one estimator, or feed state estimates back into hardware signal-tracking loops. That coupling depth governs system complexity, statistical optimality, and how failures propagate when one sensor degrades.

### 8.1.1 Loosely Coupled

In a loosely coupled architecture, each sensor subsystem acts as an independent estimator that reduces its own observations to a pose or velocity estimate before a higher-level fusion stage reconciles the outputs. Concretely, a LiDAR odometry module estimates $\mathbf{T}_{L}$ from point-cloud scans while a visual odometry module estimates $\mathbf{T}_{V}$ from images; the fusion layer then combines both estimates via covariance-weighted least squares:

$$
\hat{\mathbf{x}} = \arg\min_{\mathbf{x}} \left\| \mathbf{x} - \mathbf{x}_{LiDAR} \right\|^2_{\mathbf{P}_{L}^{-1}} + \left\| \mathbf{x} - \mathbf{x}_{Visual} \right\|^2_{\mathbf{P}_{V}^{-1}}
$$

Here $\mathbf{P}_{L}$ and $\mathbf{P}_{V}$ denote the covariances reported by each subsystem, so their inverses act as information-matrix weights.

The main appeal of loose coupling is clean modularity. Individual sensor modules can be swapped, upgraded, or debugged independently, and if one sensor drops out, the remaining subsystems continue producing estimates. That isolation comes at the cost of information loss, however, because fusion happens only after each subsystem has compressed its raw measurements into a summary pose. Subsystems cannot assist each other at the measurement level—LiDAR's metric 3D geometry, for example, cannot directly resolve monocular camera scale inside the visual tracker. Equally important, covariance consistency is never guaranteed: if either subsystem reports an overly optimistic covariance, it pulls the fused state away from the truth.

### 8.1.2 Tightly Coupled

Rather than merging summarized poses, a tightly coupled system constrains a single shared state directly with the raw measurements from all sensors. What defines tight coupling is where measurements enter the estimator, not whether the implementation uses a single monolithic optimizer. Architectures where two subsystems update one shared state (R3LIVE) or apply sequential updates (FAST-LIVO2) are tightly coupled, as are dual-graph designs like LVI-SAM that exchange raw measurements (LiDAR depth) and state estimates (initial guesses) directly between subsystems.

Expressed as a factor graph, every sensor contributes its raw observations as residual factors over the shared state:

$$
\hat{\mathbf{x}} = \arg\min_{\mathbf{x}} \sum_{i} \left\| \mathbf{r}^{\text{IMU}}_i(\mathbf{x}) \right\|^2_{\boldsymbol{\Sigma}^{-1}_{\text{IMU}}} + \sum_{j} \left\| \mathbf{r}^{\text{LiDAR}}_j(\mathbf{x}) \right\|^2_{\boldsymbol{\Sigma}^{-1}_{\text{LiDAR}}} + \sum_{k} \left\| \mathbf{r}^{\text{cam}}_k(\mathbf{x}) \right\|^2_{\boldsymbol{\Sigma}^{-1}_{\text{cam}}}
$$

In this objective, $\mathbf{r}^{\text{IMU}}_i$ is the IMU preintegration residual, $\mathbf{r}^{\text{LiDAR}}_j$ is the point-to-plane residual, and $\mathbf{r}^{\text{cam}}_k$ is the visual reprojection error.

By bringing raw residuals into one state space, sensors reinforce one another directly: high-rate IMU measurements undistort LiDAR scans, while LiDAR geometry anchors VIO scale, enabling fusion close to information-theoretic optimality. The trade-off is substantially higher engineering complexity. Observation models, noise covariances, and time synchronization across every sensor must be calibrated within a single framework, and because an unrejected outlier from one modality can corrupt the shared state, robust outlier rejection and careful compute budgeting for real-time execution become essential. Representative systems include [LIO-SAM (Shan et al. 2020)](https://arxiv.org/abs/2007.00258) (LiDAR+IMU+GPS), VINS-Mono (Camera+IMU), and [R3LIVE (Lin et al. 2022)](https://arxiv.org/abs/2109.07982) (Camera+LiDAR+IMU).

### 8.1.3 Ultra-Tightly Coupled (Signal-Level Coupling)

At the deepest level of integration, fusion reaches past decoded measurements into the raw sensor signals themselves. The canonical example is GNSS-INS ultra-tight coupling. Whereas a tightly coupled GNSS-INS estimator combines receiver-decoded pseudoranges with inertial navigation, an ultra-tight design feeds INS-predicted position and velocity directly into the GNSS receiver's code and carrier tracking loops. Specifically, the receiver's Numerically Controlled Oscillator (NCO) adjusts its tracking frequency using the Doppler shift predicted by the INS:

$$
\text{NCO frequency} = f_{\text{nominal}} + \Delta f_{\text{INS-aided}}
$$

Compensating for platform motion inside the tracking loop narrows the effective dynamics the receiver must follow, letting it maintain lock through weak-signal environments such as urban canyons or immediately after entering indoors.

In practice, ultra-tight coupling faces a sharp hardware boundary: it requires direct access to correlators or tracking loops and cannot be implemented through a standard measurement API. Such architectures appear mainly in integrated GNSS/INS products and research platforms with receiver SDK, FPGA, or firmware access; most robotics systems instead rely on tightly coupled pipelines built on raw pseudorange, carrier, and Doppler outputs.

### 8.1.4 Comparison of the Three Levels

Comparing how sensor streams flow through an estimator makes the distinction between the three coupling regimes concrete:

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

The Python function below implements the simplest case—loosely coupled fusion in information form—converting each subsystem covariance into an information matrix so the fused estimate naturally shifts toward the more certain sensor:

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

With these coupling regimes established, the next section examines how modern robotics pipelines put tight coupling into practice across cameras, LiDAR, and IMUs.

---

## 8.2 Camera + LiDAR + IMU Fusion

Pairing a camera, a LiDAR, and an IMU brings three complementary measurement types onto one rig: visual texture and color, metric 3D range, and high-rate inertial motion. Even with all three mounted, cost, power, weather, range limits, and correlated failures still shape system reliability, so installing every sensor does not automatically yield robustness. The table below summarizes where each modality holds up and where it fails:

| Situation | Camera | LiDAR | IMU |
|------|--------|-------|-----|
| Dark environment | ✗ | ✓ | ✓ |
| Textureless wall | ✗ | ✓ | ✓ |
| Geometric degeneracy (long corridor) | ✓ | ✗ | ✓ |
| High-speed rotation | ✗ | ✗ | ✓ |
| Scale observability | ✗ (monocular) | ✓ | ✓ (acceleration is in metric units; requires motion excitation) |
| Color/semantics | ✓ | ✗ | ✗ |

To exploit this complementarity in real time, modern Camera+LiDAR+IMU systems organize the estimator around different coupling patterns—a shared-state dual filter in R3LIVE, bidirectionally coupled subsystems in LVI-SAM, and sequential direct updates in FAST-LIVO2.

### 8.2.1 R3LIVE / R3LIVE++

R3LIVE (Lin et al., 2022) tightly couples LiDAR-Inertial Odometry (LIO) and Visual-Inertial Odometry (VIO) through a **dual-subsystem** architecture that converges on a **shared state vector**. The LIO subsystem handles 3D geometry while the VIO subsystem handles photometric texture, with IMU propagation linking both pipelines:

```
LiDAR scan ──→ [LIO subsystem] ──→ state update (geometry)
                     ↓                       ↓
                 IMU data ───────────→ shared state vector
                     ↑                       ↑
Camera image ──→ [VIO subsystem] ──→ state update (photometric)
```

Following FAST-LIO2, the LIO subsystem registers raw LiDAR points directly against an ikd-Tree map using point-to-plane residuals and updates the shared state through an iterated EKF.

Where typical VIO pipelines minimize feature reprojection error, R3LIVE's VIO subsystem minimizes **photometric error** directly against the colored 3D map. Each point in the LIO-built map stores an RGB color; as each new camera frame arrives, those map points project into the image and drive an update that reduces the discrepancy between the observed pixel intensity and the stored map color:

$$
\mathbf{r}^{\text{photo}}_i = \mathbf{I}(\pi(\mathbf{T}_{CW} \mathbf{p}^W_i)) - \mathbf{c}_i^{\text{map}}
$$

Here $\mathbf{I}(\cdot)$ is the pixel intensity of the image, $\pi(\cdot)$ is the 3D→2D projection function, $\mathbf{T}_{CW}$ is the world-to-camera transformation, $\mathbf{p}^W_i$ is the 3D coordinate of a map point, and $\mathbf{c}_i^{\text{map}}$ is the color stored in the map for that point.

Because both subsystems write to the same state vector, valid updates can continue even when one modality temporarily yields few usable residuals. How long visual updates sustain during a LiDAR outage depends on whether previously colored map points remain visible; conversely, LIO operation in dark imagery depends on LiDAR geometry and IMU quality. Using this structure, the paper demonstrates online colored 3D mapping alongside odometry.

### 8.2.2 LVI-SAM

[LVI-SAM](https://arxiv.org/abs/2104.10831) (Shan et al., 2021) extends LIO-SAM by coupling a visual-inertial subsystem (VIS) and a LiDAR-inertial subsystem (LIS) **bidirectionally** rather than merging them into one monolithic filter.

Information flows both ways between the two pipelines:

- **From VIS to LIS**, the pose estimated by the visual-inertial subsystem serves as the initial guess for LiDAR scan matching, helping registration converge under fast rotations or sparse geometry where LiDAR alone struggles.
- **From LIS to VIS**, LiDAR depth is assigned to visual features with valid spatial and temporal correspondence. Given accurate extrinsics, synchronization, visibility checks, and occlusion handling, this supplies a metric depth prior without waiting for triangulation parallax.

The diagram below illustrates this exchange alongside the downstream factor graph:

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

On the backend, the LIS owns the GTSAM/iSAM2 factor graph inherited from LIO-SAM, maintaining IMU preintegration factors between successive keyframes, LiDAR odometry factors from scan matching, and loop closure factors when a revisit is detected. In the public implementation, rather than inserting visual odometry as an explicit graph factor, the VIS couples through the initial-guess handoff and LiDAR depth association (while the paper also describes visual-odometry graph constraints, the released code disables that factor).

### 8.2.3 FAST-LIVO / FAST-LIVO2

Developed by the FAST-LIO2 team (HKU MARS Lab), [FAST-LIVO2](https://arxiv.org/abs/2408.14035) (Zheng et al., 2024) takes a direct, feature-free approach to Camera+LiDAR+IMU fusion, operating straight on raw scans and pixel patches.

Combining raw LiDAR and raw image measurements in one joint update creates a severe dimension, unit, and noise-scale mismatch: LiDAR yields one point-to-plane residual per point, whereas the camera yields one photometric residual per patch pixel. To avoid balancing heterogeneous blocks inside a single Jacobian solve, FAST-LIVO2 adopts **sequential Bayesian updates**—predicting the state with the IMU, updating first with LiDAR measurements, and updating again with camera measurements. Under conditional independence of the sensor noises, sequential updating produces the same posterior as simultaneous fusion:

$$
p(\mathbf{x} | \mathbf{z}_L, \mathbf{z}_C) = p(\mathbf{x} | \mathbf{z}_C, \mathbf{z}_L) \propto p(\mathbf{z}_C | \mathbf{x}) \cdot p(\mathbf{z}_L | \mathbf{x}) \cdot p(\mathbf{x})
$$

Factoring the update into two stages makes the post-LiDAR posterior $p(\mathbf{x} | \mathbf{z}_L)$ the prior for the camera step, yielding the same result as a simultaneous update:

$$
\underbrace{p(\mathbf{x} | \mathbf{z}_L)}_{\text{after LiDAR update}} \propto p(\mathbf{z}_L | \mathbf{x}) \cdot p(\mathbf{x})
$$
$$
\underbrace{p(\mathbf{x} | \mathbf{z}_L, \mathbf{z}_C)}_{\text{after camera update}} \propto p(\mathbf{z}_C | \mathbf{x}) \cdot p(\mathbf{x} | \mathbf{z}_L)
$$

Alongside sequential estimation, three map and imaging mechanisms keep the two modalities tightly aligned:

- **Unified adaptive voxel map**: A single voxel structure backed by a hash table and an octree stores both LiDAR-derived geometry (3D coordinates and surface normals) and camera-attached image patches on the same map points, keeping geometry and texture consistent in one representation.
- **Affine warping using LiDAR normals**: Direct photometric patch matching loses accuracy on slanted surfaces unless perspective tilt is modeled; FAST-LIVO2 reuses the planar normals estimated from LiDAR points to perform accurate affine patch warping without separate normal estimation.
- **Real-time exposure compensation**: When illumination shifts abruptly—such as entering or exiting a tunnel—the estimator tracks exposure time online and adjusts the photometric residual accordingly.

The function below implements the linear-algebraic core of this sequential EKF update in the LiDAR-then-camera order:

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

Comparing R3LIVE, LVI-SAM, and FAST-LIVO2 side by side clarifies how backend formulation, feature handling, and map structure trade off across these three systems:

| Aspect | R3LIVE | LVI-SAM | FAST-LIVO2 |
|------|--------|---------|------------|
| Backend | IEKF (dual subsystem) | iSAM2 (factor graph) | IEKF (sequential) |
| LiDAR processing | Direct (point-to-plane) | Feature-based (LOAM) | Direct (point-to-plane) |
| Camera processing | Direct (photometric) | Feature-based (Shi-Tomasi + KLT) | Direct (photometric) |
| Map representation | ikd-Tree + RGB | Voxel map | Hash+Octree voxel map |
| Feature extraction | Not required | Required (LiDAR edge/planar, visual corners) | Not required |
| GPS integration | None | None (GPS-factor code inherited from LIO-SAM is disabled by default) | None |
| Loop closure | None | Integrated as factor | None |
| Embedded validation | Benchmark on target hardware | Benchmark on target hardware | ARM implementations reported; revalidate under the target setup |

When narrowing candidates for a specific deployment, match the architecture to the operational requirements and verify assumptions on target data. LVI-SAM is the natural candidate when integrated loop closure is needed, provided its sensor layout, ROS dependencies, and revisit behavior are checked against the mission profile. R3LIVE fits workflows centered on direct photometric updates and online colored mapping, judged by color, trajectory, and map-quality metrics in the target scene. For ARM platforms, FAST-LIVO2 is a strong candidate, though throughput, worst-case latency, and memory still require measurement at the target resolution and configuration. In scenes with scarce visual corners or LOAM edges, direct pipelines such as R3LIVE and FAST-LIVO2 merit evaluation alongside checks on photometric calibration and sensitivity to dynamic objects.

---

## 8.3 GNSS Integration

Even tightly coupled visual-inertial and LiDAR-inertial pipelines observe only relative motion, so small estimation errors inevitably accumulate into drift over long distances. Global Navigation Satellite System (GNSS) receivers counter that drift by supplying absolute position references in a global coordinate frame. While surveyed landmarks, UWB beacons, and motion-capture systems can play an analogous anchoring role in bounded spaces, GNSS remains the primary global reference for outdoor robotics.

### 8.3.1 GNSS Factor in the Factor Graph (LIO-SAM Approach)

[LIO-SAM](https://arxiv.org/abs/2007.00258) (Shan et al., 2020) anchors the local SLAM trajectory by attaching each GNSS position fix to the corresponding pose node as a unary residual factor:

$$
\mathbf{r}^{\text{GPS}}_i = \mathbf{T}_{\text{ENU→map}} \cdot \mathbf{p}^{\text{ENU}}_{\text{GPS}} - \mathbf{p}^{\text{map}}_i - \mathbf{R}^{\text{map}}_i \cdot \mathbf{l}_{\text{antenna}}
$$

where:
- $\mathbf{p}^{\text{ENU}}_{\text{GPS}}$ is the ENU position reported by the GNSS receiver
- $\mathbf{T}_{\text{ENU→map}}$ transforms coordinates from the ENU frame into the SLAM map frame; because this residual compares both positions in the map frame, this transformation applies directly (using the inverse $\mathbf{T}_{\text{map→ENU}}$ would require $\mathbf{T}^{-1}$)
- $\mathbf{p}^{\text{map}}_i$ is the robot position estimated by SLAM
- $\mathbf{l}_{\text{antenna}}$ is the lever-arm offset from the robot body frame to the GNSS antenna
- $\mathbf{R}^{\text{map}}_i$ is the robot's rotation in the map frame

Aligning those frames raises an initialization problem: the SLAM local map frame does not coincide with the GNSS global frame (WGS84/ENU). Upon receiving the first valid GNSS fix, the system fixes the ENU origin and estimates the map-to-ENU transform from the initial trajectory. Although a general rigid transform has 6 DoF (three translations and three rotations), gravity observed by the IMU already anchors roll and pitch, leaving only 4 DoF—yaw plus three translations—to be solved.

### 8.3.2 Loosely vs Tightly Coupled GNSS

Just as with vision and LiDAR, GNSS can enter the state estimator either as a pre-solved position fix or as raw satellite measurements.

In **loosely coupled GNSS-INS**, an EKF fuses the position-velocity-time (PVT) solution computed internally by the receiver with the inertial navigation state—the standard pattern in most commercial navigation units. The following function illustrates a single position update against a 15-state INS vector:

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

**Tightly coupled GNSS-INS** bypasses the receiver's PVT solver and inserts raw pseudorange and Doppler observations directly as per-satellite factors:

$$
\rho_i = \| \mathbf{p}_{\text{sat},i} - \mathbf{p}_{\text{rx}} \| + c \cdot \delta t_{\text{rx}} + I_i + T_i + \epsilon_i
$$

Here $\rho_i$ is the pseudorange to satellite $i$, $c \cdot \delta t_{\text{rx}}$ is the receiver clock bias, and $I_i$ and $T_i$ denote the ionospheric and tropospheric delays. Unlike loose coupling—which stalls whenever fewer than four satellites are locked—tight coupling continues extracting information from whatever satellites remain visible, a decisive advantage in urban canyons where buildings routinely block half the sky.

### 8.3.3 Handling GNSS-Denied → GNSS-Available Transitions

Field robots repeatedly cross between open sky and GNSS-denied environments such as tunnels, underground garages, and highway underpasses. Recovering cleanly when signals return requires guarding against three failure modes:

1. **Avoid coordinate jumps**: Immediately after GNSS returns, accumulated odometry drift can leave a sizable gap between the GNSS fix and the local INS/LiDAR pose. Applying a tight covariance right away snaps the trajectory abruptly and tears the map; starting with a large GNSS uncertainty and tightening it gradually prevents discontinuities.
2. **Verify GNSS quality**: Measurements in the first few seconds after reacquisition are often degraded by multipath reflections. Gate new factors on PDOP/HDOP, satellite count, and carrier-phase fix status before trusting them.
3. **Correct the map frame**: When a prolonged outage has accumulated significant drift across the local map, recovering global consistency requires re-optimizing the trajectory itself via pose graph optimization, much like a loop closure.

---

## 8.4 Radar Fusion

Optical cameras and LiDARs both struggle when heavy fog, rain, or snow scatters visible and near-infrared light. Millimeter-wave radar penetrates those particulates far more effectively and measures radial velocity directly, offering an orthogonal failure mode for all-weather autonomy.

### 8.4.1 Radar and 4D Imaging

Conventional automotive radar measured only range, Doppler velocity, and azimuth—too coarse in elevation to support general 3D SLAM or odometry. Adding **elevation** transforms **4D imaging radar** into a 3D point-cloud sensor, albeit much sparser than LiDAR (hundreds to thousands of returns per scan versus hundreds of thousands).

Three physical and economic characteristics govern how radar fits into a fusion stack:

1. **Adverse-weather tolerance**: Millimeter-wave radar typically suffers less attenuation in fog, rain, and snow than visible cameras or many LiDARs. Heavy precipitation, wet-road multipath, clutter, and attenuation still degrade returns, so performance must be verified condition by condition.
2. **Direct radial-velocity measurement**: FMCW Doppler extracts the line-of-sight component of relative velocity from phase changes across multiple chirps within one frame; a single chirp yields range only, with velocity unresolved. Because one return measures only radial projection rather than full 3D target velocity, recovering full motion requires returns across multiple directions, temporal tracking, or an aiding sensor.
3. **Cost structure**: Mass-market radar chipsets can be inexpensive, yet complete imaging-radar and LiDAR module prices overlap depending on channel count, antenna array, compute, and volume. Decisions should weigh actual vendor quotes against target specifications rather than assuming a fixed cost ratio.

### 8.4.2 Radar Odometry

Work on 4D radar odometry has expanded steadily since 2022, largely by exploiting Doppler velocity directly for ego-motion estimation. Each FMCW return provides $(r, \theta, \phi, v_d)$—range, azimuth, elevation, and Doppler velocity. Given a robot moving with linear velocity $\mathbf{v}$ and angular velocity $\boldsymbol{\omega}$, the Doppler velocity observed at a static return along unit direction $\mathbf{d}_i = [\cos\phi_i \cos\theta_i, \cos\phi_i \sin\theta_i, \sin\phi_i]^T$ satisfies:

$$
v_{d,i} = -\mathbf{d}_i^T \mathbf{v}_S + n_i
$$

where $\mathbf{v}_S$ is the linear velocity of the radar phase center. Provided static returns span sufficiently diverse directions, stacking these scalar equations constrains $\mathbf{v}_S$. Point Doppler from a single monostatic radar does not independently observe angular velocity about the sensor origin, so transforming motion to the vehicle body frame requires a known lever arm paired with angular rates from an IMU or scan registration.

The function below estimates ego linear velocity by solving this linear Doppler system with least squares; in deployment, RANSAC outlier rejection precedes the solve to strip out moving vehicles and pedestrians:

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

Pairing 4D radar with cameras targets a LiDAR-free perception stack by trading dense visual semantics against sparse metric depth and Doppler velocity:

| Property | Camera | 4D Radar |
|------|--------|----------|
| Resolution | Very high | Low |
| Adverse weather | Weak | Robust |
| Direct depth measurement | ✗ | ✓ |
| Radial-velocity measurement | ✗ | ✓ |
| Semantic understanding | Strong | Weak |
| Cost | Depends on lenses, camera count, and compute | Depends on antennas, channels, and compute |

Architectures combine these streams at three stages: **early fusion** projects radar returns into the image plane as sparse metric depth anchors for monocular depth estimation; **mid-level fusion** merges camera and radar feature maps inside a neural network, commonly in Bird's Eye View (BEV) space; and **late fusion** runs object detection independently on each sensor before associating and merging the bounding outputs.

### 8.4.4 Boreas Benchmark

[Boreas](https://arxiv.org/abs/2203.10168) (Burnett et al., 2023) records repeated multi-sensor traverses across clear weather, rain, snow, and seasonal change, synchronizing cameras, a Velodyne Alpha Prime LiDAR, and a Navtech CIR304-H **planar scanning radar**. Because that Navtech sensor provides 360-degree azimuth-range scans without elevation rather than an automotive 4D imaging radar, Boreas radar-odometry results should not be conflated with 4D-radar benchmarks.

---

## 8.5 Multi-Robot / Decentralized Fusion

Scaling fusion from one platform to a fleet introduces three coupled constraints: robots start without a shared coordinate frame, wireless links cannot carry full maps or raw sensor streams, and central servers create both communication bottlenecks and single points of failure.

### 8.5.1 Challenges of Multi-Robot SLAM

Cooperative mapping therefore centers on three problems. First, because each robot runs SLAM in its own local coordinate frame, merging two maps requires recovering their **inter-robot relative pose** via cross-robot place recognition and geometric verification. Second, **communication constraints** prevent streaming raw scans or dense maps, forcing systems to compress what they share into compact descriptors and loop constraints. Third, **distributed optimization** replaces central aggregation so each robot solves its local graph while exchanging only minimal boundary variables with nearby peers.

### 8.5.2 Kimera-Multi

Developed by MIT's SPARK Lab, [Kimera-Multi](https://arxiv.org/abs/2106.14386) (Tian et al., 2021) addresses those constraints in a distributed metric-semantic SLAM pipeline. Each robot runs Kimera locally; upon rendezvous, **DBoW2**-based inter-robot place recognition identifies overlapping places, and detected inter-robot loop closures enter distributed pose graph optimization guarded by a **GNC (Graduated Non-Convexity)** solver that rejects outlier loops.

Each robot maintains its own pose graph. Once a loop is verified, the inter-robot factor is shared; from that point on, a distributed optimizer such as Riemannian block-coordinate descent iterates to convergence by exchanging current estimates of the separator poses shared with neighbors. Sharing a loop constraint once and repeatedly exchanging separator variables during optimization represent two distinct communication costs.

### 8.5.3 Swarm-SLAM

[Swarm-SLAM](https://arxiv.org/abs/2301.06230) (Lajoie et al., 2024) extends distributed SLAM to larger robot swarms with an explicit focus on bandwidth efficiency. Rather than transmitting full maps, robots broadcast compact place-recognition descriptors (such as NetVLAD or Scan Context) peer-to-peer to neighbors without a central server, exchanging minimal geometric data—selected keypoints or point clouds—only after descriptor matching flags a candidate inter-robot loop closure. Because the coordination layer decouples retrieval from sensing modality, camera-only, LiDAR-only, and mixed-sensor robots can participate in the same swarm.

The class sketch below illustrates how each robot in a decentralized pose graph tracks its local trajectory while exchanging only separator poses tied to inter-robot factors:

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

## 8.6 Sensor Selection, Timing Sync, and Failure Handling

Even mathematically sound fusion estimators fail in deployment without careful hardware selection, clock synchronization, and fault handling.

### 8.6.1 Sensor Suite Selection Guide

The table below gives a practical baseline by operating environment; final hardware choices still depend on safety margins, range, power and mass budgets, and expected failure scenarios:

| Environment | Recommended minimum | Optional additional sensors |
|------|---------------|---------------|
| Indoor (office/warehouse) | Camera + IMU | LiDAR (for precise mapping) |
| Urban autonomous driving | Camera + LiDAR + IMU + GNSS | 4D Radar, Wheel Odom |
| Off-road/outdoor | LiDAR + IMU + GNSS | Camera (semantics), Radar |
| Underground/tunnel | LiDAR + IMU | Camera, UWB |
| Underwater | IMU + DVL (Doppler Velocity Log) | Sonar, Pressure |
| Aerial/drone | Camera + IMU + GNSS | LiDAR (for mapping) |
| Adverse weather (rain/snow) | Radar + IMU | Camera, LiDAR |

Because sensor prices shift by date, region, and volume, budgeting should work backward from measurement requirements—accuracy, rate, and FoV—to total system cost, including cabling, synchronization hardware, GNSS correction services, compute, mounts, spares, and integration engineering. A prototype can start lean with a camera and IMU, adding 2D or 3D LiDAR, RTK GNSS, radar, or sensor redundancy only where testing uncovers concrete failure modes or error-budget shortfalls.

### 8.6.2 Timing Architecture (Time Synchronization Design)

In multi-sensor fusion, **time synchronization** directly bounds trajectory accuracy: on a vehicle traveling at 100 km/h, a 1 ms clock offset translates into roughly 2.8 cm of spatial error.

Hardware synchronization eliminates the jitter of software receive timestamps by distributing a shared clock or trigger, though pulse precision still differs from end-to-end measurement-timestamp accuracy and must be validated across the full pipeline:

- **PPS (Pulse Per Second)** uses a GNSS receiver's one-second pulse as a common clock edge, requiring verification of receiver pulse specs, cable and input delays, and the internal timestamp reference point.
- **PTP (Precision Time Protocol, IEEE 1588)** aligns Ethernet clocks across sensors and hosts, with realized accuracy governed by hardware timestamping, network switches, PTP profile, path asymmetry, and lock stability.
- **External trigger** links a microcontroller directly to both the camera shutter and IMU timestamp capture so exposure and inertial sampling align on hardware edges.

When hardware synchronization is unavailable, software methods estimate the inter-sensor offset from recorded motion. The **Kalibr approach** parameterizes the continuous-time trajectory as a B-spline and jointly optimizes the time offset alongside calibration parameters. **Correlation-based** methods instead recover the delay by cross-correlating motion signals observed by both sensors. Matching linear acceleration requires removing gravity from the IMU and rotating signals into a common frame, while monocular cameras leave velocity scale unobservable; in practice, engineers therefore correlate the norm of the gyroscope angular rate against the norm of the camera rotation rate, avoiding gravity removal, scale ambiguity, and extrinsic frame alignment (whereas component-wise vector comparison would require rotating one signal by the camera-IMU extrinsic rotation):

$$
\hat{\tau} = \arg\max_{\tau} \int \left\| \boldsymbol{\omega}_{\text{IMU}}(t) \right\| \, \left\| \boldsymbol{\omega}_{\text{camera}}(t + \tau) \right\| \, dt
$$

The Python function below implements this discrete cross-correlation estimator after resampling both streams onto a common timeline:

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

In field operation, sensors inevitably saturate, drop packets, or encounter degenerate scenes. Robust fusion requires **graceful degradation**—detecting impaired measurements early and continuing estimation on the remaining healthy modalities rather than letting one corrupted stream collapse the entire state. The table below pairs common sensor faults with practical detection signals and mitigation actions:

| Failure mode | Symptom | Detection | Response |
|-----------|------|-----------|------|
| Camera over-/under-exposure | Entire image is bright or dark | Histogram analysis | Disable camera factor, operate with LIO only |
| LiDAR geometric degeneracy | Long corridor, wide flat plane | Eigenvalue analysis of the information matrix | Relax LiDAR constraint on the affected DoF, compensate with VIO |
| IMU saturation | Measurement range exceeded under high-speed impact | Detect ADC maximum values | Increase IMU preintegration uncertainty for the affected interval |
| GNSS multipath | Large error due to reflections from buildings | RAIM, residual check | Increase the covariance of the affected GNSS factor or remove it |
| Total sensor dropout | No data received | Watchdog timer | Stop adding new factors. If the failure time is uncertain, down-weight or remove only the interval after the suspect point |

Among these faults, LiDAR geometric degeneracy illustrates how complementary sensing rescues a single axis of observability. Eigen-decomposing the scan-matching information matrix (Hessian) $\mathbf{H} = \mathbf{J}^T \mathbf{J}$ reveals whether constraints along any motion direction have collapsed:

$$
\mathbf{H} = \mathbf{U} \boldsymbol{\Lambda} \mathbf{U}^T, \quad \lambda_{\min} / \lambda_{\max} < \epsilon \Rightarrow \text{degenerate}
$$

Inside a long corridor, for example, eigenvalues along the hallway axis drop sharply; the estimator can then relax the LiDAR constraint in that degenerate subspace while leaning on visual optical flow to hold longitudinal motion. The two functions below detect weak Hessian directions and scale LiDAR versus camera weights accordingly:

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

Recent fusion work extends beyond sparse landmarks and voxel maps toward photorealistic scene reconstruction and standardized radar benchmarks:

- **[Gaussian-LIC (Lang et al., ICRA 2025)](https://arxiv.org/abs/2404.06926)** embeds 3D Gaussian Splatting directly inside a tightly coupled LiDAR-inertial-camera SLAM pipeline, uniting metric LiDAR geometry with camera appearance for real-time photorealistic reconstruction alongside state estimation.
- **[Snail-Radar (Huai et al., IJRR 2025)](https://arxiv.org/abs/2407.11705)** introduces a large-scale diversity benchmark for 4D radar SLAM, evaluating radar-based odometry and SLAM methods across varied indoor, outdoor, urban, and suburban platforms.

### 8.6.5 System Design Checklist

Before deploying a multi-sensor fusion stack, verify each stage from calibration through runtime resource budgets:

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

## Summary

Multi-sensor fusion architectures span loose, tight, and signal-level coupling, with **tight coupling** serving as the standard choice across modern robotics pipelines. Combining cameras, LiDAR, and IMUs takes distinct engineering shapes across R3LIVE (shared-state dual filter), LVI-SAM (bidirectional factor graph), and FAST-LIVO2 (sequential direct updates).

When local odometry alone cannot bound long-range error, GNSS anchors trajectories in global coordinates, while 4D imaging radar adds Doppler-based radial velocity and weather resilience—provided its noise and multipath characteristics are validated per operating condition. Multi-robot systems such as Kimera-Multi and Swarm-SLAM extend these principles to decentralized fleets by coupling distributed graph optimization with cross-robot place recognition under tight bandwidth budgets. Across all of these architectures, sensor selection, clock synchronization, and graceful degradation matter just as much as the estimator math.

Even tightly fused odometry accumulates drift whenever absolute references such as GNSS are unavailable. Eliminating that residual drift globally requires recognizing previously visited places and closing loops across the trajectory—the subjects of Chapter 9 and Chapter 10.
