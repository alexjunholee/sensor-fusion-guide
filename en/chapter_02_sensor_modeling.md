# Chapter 2. Sensor Modeling

Chapter 1 showed *why* multi-sensor fusion is necessary; this chapter specifies *what* each sensor actually measures. How a sensor perceives the physical world is formalized by its mathematical observation model and error distribution—the exact ingredients a Kalman filter or factor graph uses as its measurement function $h(\mathbf{x})$ (→ Chapter 4).

> Every fusion algorithm relies on an explicit mathematical mapping from state variables to raw measurements, paired with a realistic model of measurement noise (sensor overview → robotics-practice Chapter 2).

---

## 2.1 Camera Observation Model

A camera projects 3D scene points onto a 2D image plane. To use image features inside a state estimator, we need a forward projection model that maps 3D coordinates into pixels, a distortion model that accounts for real lens optics, and a timing model when pixel rows are exposed sequentially.

### 2.1.1 Pinhole Camera Model

The pinhole camera model is the simplest geometric abstraction of image formation: it assumes every 3D point projects onto the image plane along a straight ray passing through a single optical center.

Consider a 3D point $\mathbf{P}_c = [X_c, Y_c, Z_c]^\top$ expressed in the camera coordinate frame. Its ideal pixel projection $\mathbf{p} = [u, v]^\top$ is given by perspective division followed by an affine scaling into pixel coordinates:

$$\begin{bmatrix} u \\ v \\ 1 \end{bmatrix} = \frac{1}{Z_c} \mathbf{K} \begin{bmatrix} X_c \\ Y_c \\ Z_c \end{bmatrix}$$

where $\mathbf{K}$ is the **camera intrinsic matrix**:

$$\mathbf{K} = \begin{bmatrix} f_x & 0 & c_x \\ 0 & f_y & c_y \\ 0 & 0 & 1 \end{bmatrix}$$

Each intrinsic parameter captures a physical property of the lens–sensor assembly:
- $f_x, f_y$: focal lengths expressed in pixels, obtained by dividing the physical focal length $f$ by pixel pitch $(\Delta x, \Delta y)$: $f_x = f / \Delta x$ and $f_y = f / \Delta y$. Square pixels ($\Delta x = \Delta y$) give $f_x = f_y$, whereas rectangular pixels produce distinct values.
- $c_x, c_y$: principal point, where the optical axis pierces the image plane. Ideally this coincides with the image center, though manufacturing tolerances typically shift it by several pixels.

When the 3D landmark $\mathbf{P}_w = [X_w, Y_w, Z_w, 1]^\top$ lives in the world frame rather than the camera frame, homogeneous coordinates combine the rigid-body transformation and intrinsic projection into a single linear-up-to-scale relation:

$$s \begin{bmatrix} u \\ v \\ 1 \end{bmatrix} = \mathbf{K} [\mathbf{R} | \mathbf{t}] \begin{bmatrix} X_w \\ Y_w \\ Z_w \\ 1 \end{bmatrix}$$

Here $[\mathbf{R} | \mathbf{t}]$ denotes the extrinsic transform from the world frame to the camera frame, and the projective scale factor $s = Z_c$ equals the point's depth along the camera optical axis.

Writing the nonlinear camera projection explicitly as $\pi(\cdot)$:

$$\mathbf{p} = \pi(\mathbf{P}_c) = \begin{bmatrix} f_x \frac{X_c}{Z_c} + c_x \\ f_y \frac{Y_c}{Z_c} + c_y \end{bmatrix}$$

Because state estimators must propagate uncertainty and compute Gauss-Newton updates through this nonlinear mapping, they evaluate its analytical Jacobian directly:

$$\frac{\partial \pi}{\partial \mathbf{P}_c} = \begin{bmatrix} \frac{f_x}{Z_c} & 0 & -\frac{f_x X_c}{Z_c^2} \\ 0 & \frac{f_y}{Z_c} & -\frac{f_y Y_c}{Z_c^2} \end{bmatrix}$$

This compact $2 \times 3$ Jacobian linearizes the camera measurement update in EKF-based VIO and supplies the analytical derivatives inside nonlinear bundle adjustment.

### 2.1.2 Lens Distortion Model

Real glass lenses bend light nonlinearly away from the optical axis, causing systematic departures from ideal pinhole geometry. Ignoring lens distortion can inflate reprojection errors from under a pixel to tens of pixels—enough to destabilize triangulation and bias pose estimation—so accurate fusion requires modeling distortion explicitly.

#### Radial-Tangential Distortion (Brown-Conrady Model)

The Brown-Conrady radial-tangential model is the default convention in OpenCV and standard pinhole SLAM/VIO pipelines. (For ultra-wide fields of view, fisheye or omnidirectional models below are more appropriate.)

Starting from normalized image coordinates $\mathbf{p}_n = [x_n, y_n]^\top = [X_c/Z_c, \, Y_c/Z_c]^\top$, we define the squared radial distance from the optical axis:

$$r^2 = x_n^2 + y_n^2$$

**Radial distortion** arises from lens curvature and grows rapidly toward the image periphery:

$$x_r = x_n (1 + k_1 r^2 + k_2 r^4 + k_3 r^6)$$
$$y_r = y_n (1 + k_1 r^2 + k_2 r^4 + k_3 r^6)$$

When $k_1 < 0$, the lens exhibits barrel distortion (straight lines bow outward around a magnified center); when $k_1 > 0$, it exhibits pincushion distortion (magnification stretches toward the corners).

**Tangential distortion** occurs when the lens assembly is not perfectly parallel to the imaging sensor (decentering):

$$x_t = 2p_1 x_n y_n + p_2 (r^2 + 2x_n^2)$$
$$y_t = p_1 (r^2 + 2y_n^2) + 2p_2 x_n y_n$$

Although tangential terms are typically much smaller than radial terms, neglecting them introduces measurable bias in high-precision visual odometry.

Summing both contributions yields the **distorted normalized coordinates**, which are then scaled by $\mathbf{K}$ into pixel coordinates:

$$\begin{bmatrix} x_d \\ y_d \end{bmatrix} = \begin{bmatrix} x_r + x_t \\ y_r + y_t \end{bmatrix}$$

$$\begin{bmatrix} u \\ v \end{bmatrix} = \begin{bmatrix} f_x x_d + c_x \\ f_y y_d + c_y \end{bmatrix}$$

In practice, the parameter vector $[k_1, k_2, p_1, p_2, k_3]$ is calibrated alongside $\mathbf{K}$ using planar checkerboard targets as in [Zhang (2000)](https://doi.org/10.1109/34.888718) (→ Chapter 3).

#### Fisheye (Equidistant) Model

Wide-angle fisheye lenses (180° FoV or wider) break the Brown-Conrady polynomial: as the ray angle approaches 90° ($Z_c \to 0$), the normalized radius $r = \sqrt{x_n^2 + y_n^2}$ explodes and the polynomial diverges.

To avoid dividing by $Z_c$, the generic fisheye model of [Kannala & Brandt (2006)](https://doi.org/10.1109/TPAMI.2006.153) parameterizes projection directly in terms of the incidence angle $\theta$ measured from the optical axis:

$$\theta = \operatorname{atan2}\left(\sqrt{X_c^2 + Y_c^2},\; Z_c\right)$$

Using the two-argument $\operatorname{atan2}$ cleanly supports rays in the rear hemisphere ($Z_c < 0$, beyond 180°), and normalizing the radial direction by $\sqrt{X_c^2+Y_c^2}$ avoids any division by $Z_c$—the exact convention followed in the Python code below.

The distorted radius $r_d$ is modeled as an odd polynomial in $\theta$:

$$r_d = k_1 \theta + k_2 \theta^3 + k_3 \theta^5 + k_4 \theta^7 + k_5 \theta^9$$

Here $r_d$ is a dimensionless normalized radius; a pure equidistant projection corresponds to $r_d = \theta$ ($k_1 = 1$, $k_2 = k_3 = \cdots = 0$). Scaling by the pixel focal lengths $f_x$ and $f_y$ gives the final image coordinates:

$$\begin{bmatrix} u \\ v \end{bmatrix} = \begin{bmatrix} f_x \cdot r_d \cdot \frac{X_c}{\sqrt{X_c^2 + Y_c^2}} + c_x \\ f_y \cdot r_d \cdot \frac{Y_c}{\sqrt{X_c^2 + Y_c^2}} + c_y \end{bmatrix}$$

Because wide-FoV fisheye cameras see more of the environment and retain visual features longer during turns, they are supported natively in VIO pipelines such as [VINS-Mono](https://arxiv.org/abs/1708.03852) and Basalt, and are commonly calibrated with OCamCalib ([Scaramuzza et al., 2006](https://rpg.ifi.uzh.ch/docs/IROS06_scaramuzza.pdf)) or Kalibr.

The two Python functions below implement both the pinhole Brown-Conrady projection and OpenCV's 4-parameter equidistant fisheye projection.

```python
import numpy as np

def project_pinhole(P_c, K, dist_coeffs=None):
    """3D to 2D projection with the pinhole camera model.
    
    Args:
        P_c: (3,) 3D point in the camera frame [X, Y, Z]
        K: (3,3) intrinsic matrix
        dist_coeffs: (5,) distortion coefficients [k1, k2, p1, p2, k3] or None
    
    Returns:
        (2,) image coordinates [u, v]
    """
    X, Y, Z = P_c
    # Normalized coordinates
    x_n = X / Z
    y_n = Y / Z
    
    if dist_coeffs is not None:
        k1, k2, p1, p2, k3 = dist_coeffs
        r2 = x_n**2 + y_n**2
        r4 = r2**2
        r6 = r2 * r4
        
        # Radial distortion
        radial = 1 + k1 * r2 + k2 * r4 + k3 * r6
        x_d = x_n * radial + 2 * p1 * x_n * y_n + p2 * (r2 + 2 * x_n**2)
        y_d = y_n * radial + p1 * (r2 + 2 * y_n**2) + 2 * p2 * x_n * y_n
    else:
        x_d = x_n
        y_d = y_n
    
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    
    u = fx * x_d + cx
    v = fy * y_d + cy
    return np.array([u, v])


def project_fisheye(P_c, K, fisheye_coeffs):
    """3D to 2D projection with the fisheye (equidistant) camera model.
    
    Uses the model of the OpenCV fisheye module:
      theta_d = theta * (1 + k1*theta^2 + k2*theta^4 + k3*theta^6 + k4*theta^8)
    This has a different parameterization from the 5-parameter Kannala-Brandt
    model above.
    
    Args:
        P_c: (3,) 3D point in the camera frame [X, Y, Z]
        K: (3,3) intrinsic matrix
        fisheye_coeffs: (4,) distortion coefficients [k1, k2, k3, k4] (OpenCV convention)
    
    Returns:
        (2,) image coordinates [u, v]
    """
    X, Y, Z = P_c
    r_xyz = np.sqrt(X**2 + Y**2)
    theta = np.arctan2(r_xyz, Z)
    
    k1, k2, k3, k4 = fisheye_coeffs
    theta2 = theta**2
    r_d = theta * (1 + k1 * theta2 + k2 * theta2**2 + k3 * theta2**3 + k4 * theta2**4)
    
    if r_xyz < 1e-10:
        return np.array([K[0, 2], K[1, 2]])
    
    u = K[0, 0] * r_d * (X / r_xyz) + K[0, 2]
    v = K[1, 1] * r_d * (Y / r_xyz) + K[1, 2]
    return np.array([u, v])
```

### 2.1.3 Reprojection Error

Nearly every vision-based fusion backend optimizes poses and landmarks by minimizing **reprojection error**—the pixel-space discrepancy between where a 3D map point is predicted to land and where the detector actually found it.

Given a 3D landmark $\mathbf{P}_w$ and a world-to-camera pose $\mathbf{T}_{cw} = [\mathbf{R}|\mathbf{t}]$, the reprojection error compares the observed feature coordinate $\mathbf{p}_{\text{obs}}$ against the predicted projection $\hat{\mathbf{p}} = \pi(\mathbf{T}_{cw} \cdot \mathbf{P}_w)$:

$$\mathbf{e}_{\text{reproj}} = \mathbf{p}_{\text{obs}} - \pi(\mathbf{T}_{cw} \cdot \mathbf{P}_w)$$

where $\pi(\cdot)$ includes both pinhole projection and lens distortion.

In **bundle adjustment**, all camera poses and 3D landmarks are jointly refined by minimizing the Mahalanobis-weighted sum of squared reprojection errors across every observation pair $(i, j)$:

$$\min_{\{\mathbf{T}_i\}, \{\mathbf{P}_j\}} \sum_{i,j} \rho\left(\| \mathbf{p}_{ij} - \pi(\mathbf{T}_i \cdot \mathbf{P}_j) \|^2_{\mathbf{\Sigma}_{ij}}\right)$$

where:
- $\mathbf{p}_{ij}$: detected image coordinates of landmark $j$ in camera frame $i$
- $\mathbf{\Sigma}_{ij}$: measurement noise covariance (often modeled isotropically as $\sigma^2 \mathbf{I}_2$ with $\sigma \approx 1$ pixel)
- $\rho(\cdot)$: robust loss kernel (such as Huber or Cauchy) that down-weights mismatched outliers

Feature detection noise is typically modeled as zero-mean Gaussian with $\sigma = 0.5 \sim 2$ pixels, shrinking toward $\sigma \approx 0.5$ pixel when sub-pixel corner refinement is used. More recently, monocular depth foundation models such as [Depth Anything V2 (Yang et al., 2024)](https://arxiv.org/abs/2406.09414) and [Metric3D v2 (Hu et al., 2024)](https://arxiv.org/abs/2404.15506) have opened the door to augmenting classic 2D reprojection residuals with dense 3D depth priors.

### 2.1.4 Rolling Shutter Model

Most consumer CMOS cameras—including those on smartphones and low-cost webcams—use a **rolling shutter** rather than a global shutter. Instead of exposing the entire sensor array simultaneously, a rolling shutter reads out rows sequentially from top to bottom, meaning the top and bottom of a single image observe the world at slightly different instants.

The exposure timestamp of row $k$ is shifted linearly by the row readout time $t_r$:

$$t_k = t_0 + k \cdot t_r$$

where $t_0$ is the timestamp of the first row. Across an image of height $H$, the start-time offset between the first and last rows is $(H-1) \cdot t_r$ (distinct from each row's integration duration), giving a total readout period of roughly $H \cdot t_r$—typically several to tens of milliseconds.

When a rolling-shutter camera moves rapidly during readout, two closely coupled distortions corrupt the image:
- **Geometric "jello" distortion**: vertical edges skew diagonally and dynamic objects appear to wobble.
- **Pose-inconsistent features**: because landmarks on different rows were imaged under different camera poses, fitting a single global-shutter pose $\mathbf{T}_{cw}$ introduces systematic reprojection bias.

A rolling-shutter-aware observation model therefore evaluates the camera pose at each feature's exact row timestamp $t_{v_i}$:

$$\mathbf{p}_i = \pi\left(\mathbf{T}(t_{v_i}) \cdot \mathbf{P}_i\right)$$

where $\mathbf{T}(t_{v_i})$ is the world-to-camera transform at row $v_i$. Given poses at the first and last rows ($H>1$), constant-twist $SE(3)$ interpolation approximates intermediate poses smoothly along the manifold:

$$\mathbf{T}(t_{v_i}) = \mathbf{T}(t_0) \cdot \text{Exp}\left(\frac{v_i}{H-1} \cdot \text{Log}(\mathbf{T}(t_0)^{-1} \mathbf{T}(t_0 + (H-1) \cdot t_r))\right)$$

Here $\text{Exp}$ and $\text{Log}$ denote the exponential and logarithmic maps on $SE(3)$. For convenience, the Python function below accepts camera-to-world transforms and decouples motion into rotational SLERP plus linear translation interpolation.

Rolling-shutter compensation is available optionally in systems such as [VINS-Mono](https://arxiv.org/abs/1708.03852) (via a calibrated readout-time parameter, whereas ORB-SLAM3 assumes a global shutter), and becomes essential whenever fast maneuvers meet consumer CMOS sensors on drones or handheld devices.

```python
import numpy as np
from scipy.spatial.transform import Rotation, Slerp

def rolling_shutter_project(P_w, T_start, T_end, K, H, v_row):
    """Projection under a rolling-shutter camera model.
    
    Interpolates between the poses at the first row (T_start) and last row
    (T_end) of the image, and projects the 3D point using the pose of the
    corresponding row.
    
    Args:
        P_w: (3,) 3D point in world coordinates
        T_start: (4,4) camera-to-world transform at the exposure of the first row
        T_end: (4,4) camera-to-world transform at the exposure of the last row
        K: (3,3) intrinsic matrix
        H: image height (number of rows)
        v_row: row coordinate of the feature point to project
    
    Returns:
        (2,) image coordinates [u, v]
    """
    alpha = v_row / (H - 1) if H > 1 else 0.0  # first row 0, last row 1
    
    # Rotation interpolation (SLERP)
    R_start = Rotation.from_matrix(T_start[:3, :3])
    R_end = Rotation.from_matrix(T_end[:3, :3])
    slerp = Slerp([0, 1], Rotation.concatenate([R_start, R_end]))
    R_interp = slerp(alpha).as_matrix()
    
    # Translation interpolation (linear)
    t_interp = (1 - alpha) * T_start[:3, 3] + alpha * T_end[:3, 3]
    
    # World-to-camera transform, then project
    P_c = R_interp.T @ (P_w - t_interp)
    
    u = K[0, 0] * P_c[0] / P_c[2] + K[0, 2]
    v = K[1, 1] * P_c[1] / P_c[2] + K[1, 2]
    return np.array([u, v])
```

---

## 2.2 LiDAR Observation Model

Whereas cameras infer geometry indirectly from ambient light, LiDAR is an active sensor that emits laser pulses and measures time-of-flight or phase shift to recover 3D structure directly. Modeling LiDAR accurately means capturing both its native spherical measurement noise and the temporal distortion introduced as the beam sweeps across a moving scene.

### 2.2.1 Range-Bearing Model

Every LiDAR return originates as a **range** and **bearing** measurement along an individual laser beam. For a 3D LiDAR, each point observation is naturally parameterized in spherical coordinates $(r, \alpha, \omega)$—range $r$, azimuth $\alpha$, and elevation $\omega$.

Given a true 3D point $\mathbf{P}_L = [x, y, z]^\top$ in the LiDAR frame, the noisy spherical observation follows:

$$\begin{aligned}
r &= \sqrt{x^2 + y^2 + z^2} + n_r \\
\alpha &= \arctan2(y, x) + n_\alpha \\
\omega &= \arcsin\left(\frac{z}{\sqrt{x^2 + y^2 + z^2}}\right) + n_\omega
\end{aligned}$$

where $n_r$, $n_\alpha$, and $n_\omega$ denote the range, azimuth, and elevation measurement errors, respectively.

Understanding how these noise terms scale with distance is critical when weighting LiDAR residuals inside a filter or factor graph:
- **Range noise** ($n_r$): typically $\sigma_r \approx 1\text{–}3\,\text{cm}$ on mechanical spinning units and $\sigma_r \approx 2\text{–}5\,\text{cm}$ on solid-state units, increasing at long range due to beam spreading and weaker return energy.
- **Angular noise** ($n_\alpha, n_\omega$): governed by encoder and mirror precision, typically $\sigma_\alpha, \sigma_\omega \approx 0.01°\text{–}0.1°$. Although tiny in angle, lever-arm geometry amplifies angular uncertainty laterally with distance—at 50 m, a $0.1°$ error shifts a point laterally by roughly $8.7\,\text{cm}$.

Converting the spherical return back into Cartesian coordinates yields the point cloud coordinates consumed by registration algorithms:

$$\mathbf{P}_L = \begin{bmatrix} r \cos\omega \cos\alpha \\ r \cos\omega \sin\alpha \\ r \sin\omega \end{bmatrix}$$

**Beam divergence and mixed pixels.** A physical laser beam is not an infinitely thin ray; it diverges as a narrow cone, typically $0.1°\text{–}0.5°$ wide. At long range, the illuminated spot grows large enough that a single return averages reflections across that patch. Whenever a beam straddles the silhouette edge of an object—hitting both foreground and background—the receiver blends both echoes into a phantom point floating in mid-air between them, an artifact known as a **mixed pixel**.

### 2.2.2 Motion Distortion

Just as a rolling-shutter camera exposes pixel rows sequentially, a mechanical spinning LiDAR sweeps 360° over a finite scan period—roughly 100 ms per revolution on a Velodyne VLP-16. Whenever the robot moves or turns during that sweep, points within a "single" scan are actually observed from continuously changing poses. Correcting this warping is the **motion distortion** (or **ego-motion compensation**) problem.

To reconstruct a geometrically consistent snapshot at a common reference timestamp $t_0$, each point $\mathbf{P}_L^{(t_i)}$ captured at $t_i$ must be warped back using the sensor motion between $t_0$ and $t_i$:

$$\mathbf{P}_L^{(t_0)} = \mathbf{T}(t_0)^{-1} \cdot \mathbf{T}(t_i) \cdot \mathbf{P}_L^{(t_i)}$$

where $\mathbf{T}(t_i)$ is the LiDAR-to-world pose at timestamp $t_i$.

In practice, estimators obtain the intra-scan trajectory $\mathbf{T}(t_i)$ in one of three ways:

1. **IMU-based interpolation**: propagate high-rate inertial measurements across the scan window to interpolate each point's relative pose—the standard approach in LIO pipelines such as LIO-SAM and FAST-LIO2.
2. **Constant-velocity odometry**: extrapolate smoothly across the scan using the linear and angular velocity estimated from the preceding sweep.
3. **Continuous-time trajectories**: represent the pose trajectory directly as a smooth spline (such as a B-spline) and query $\mathbf{T}(t_i)$ analytically at every point's timestamp, as in [CT-ICP (Dellenbach et al., 2022)](https://arxiv.org/abs/2109.12979).

The Python snippet below demonstrates IMU-style deskewing by interpolating between the scan-start and scan-end poses with rotational SLERP and linear translation.

```python
import numpy as np
from scipy.spatial.transform import Rotation, Slerp

def undistort_scan(points, timestamps, T_start, T_end, t_start, t_end):
    """IMU-based motion distortion correction.
    
    Interpolates between the scan-start and scan-end poses and transforms each
    point into the coordinate frame of the start time.
    
    Args:
        points: (N, 3) LiDAR point cloud
        timestamps: (N,) timestamp of each point
        T_start: (4,4) LiDAR pose at the start of the scan (lidar-to-world)
        T_end: (4,4) LiDAR pose at the end of the scan (lidar-to-world)
        t_start, t_end: scan start/end times
    
    Returns:
        (N, 3) corrected point cloud
    """
    R_start = Rotation.from_matrix(T_start[:3, :3])
    R_end = Rotation.from_matrix(T_end[:3, :3])
    slerp = Slerp([t_start, t_end], Rotation.concatenate([R_start, R_end]))
    
    corrected = np.zeros_like(points)
    for i in range(len(points)):
        alpha = (timestamps[i] - t_start) / (t_end - t_start)
        alpha = np.clip(alpha, 0, 1)
        
        # Pose interpolation at that time
        R_i = slerp(np.clip(timestamps[i], t_start, t_end)).as_matrix()
        t_i = (1 - alpha) * T_start[:3, 3] + alpha * T_end[:3, 3]
        
        # Frame at that time -> frame at start time
        p_world = R_i @ points[i] + t_i
        corrected[i] = T_start[:3, :3].T @ (p_world - T_start[:3, 3])
    
    return corrected
```

### 2.2.3 Impact of Spinning vs Solid-State LiDAR on Fusion

How a LiDAR steers its laser beams shapes the downstream registration algorithm just as strongly as its range precision.

**Mechanical spinning LiDARs** (such as Velodyne, Ouster, and Hesai) sweep a full 360° horizontal FoV, producing an ordered annular point cloud organized into concentric scan rings. Classic LOAM-family pipelines exploit this regular ring geometry directly, extracting edge and planar keypoints along each horizontal scan line to estimate 6-DoF pose.

**Limited-FoV Livox LiDARs** (Mid-40/70, Avia, HAP, and related units) follow a different trade-off. Rather than forming uniform horizontal rings, they support non-repetitive scanning modes—though units like the Mid-40 still steer beams with internal rotating prisms rather than having zero moving parts, and the Avia also offers repetitive scanning. In non-repetitive mode, the laser traces rosette or Lissajous trajectories inside a narrower FoV (roughly 70.4° circular on the Mid-70), steadily filling in scene density over longer integration windows.

The table below contrasts these two scanning paradigms and their algorithmic consequences:

| Property | Spinning LiDAR | Limited-FoV Non-repetitive Mode |
|------|-------------|-------------------|
| FoV | 360° horizontal | Limited (40° to 120°) |
| Scan pattern | Repetitive (horizontal lines) | Non-repetitive (rosette, Lissajous, etc.) |
| Single-frame density | Uniform | Non-uniform (grows over time) |
| Feature extraction | Scan-line-based feasible | No scan-line structure |
| Suitable algorithms | LOAM, LeGO-LOAM | FAST-LIO/LIO2 (point-wise processing) |

Because non-repetitive scans lack fixed scan-line rings, LOAM's line-based feature extractor cannot be applied directly. Pipelines such as FAST-LIO and [FAST-LIO2](https://arxiv.org/abs/2107.06829) handle this naturally by feeding point-wise residuals across the scan into an iterated EKF; in particular, FAST-LIO2 **registers raw points directly against the map** without extracting features at all. Going a step further, Point-LIO in Chapter 7 updates the filter state sequentially point by point. Extending this direct iterated-EKF architecture across three modalities, [FAST-LIVO2 (Zheng et al., 2024)](https://arxiv.org/abs/2408.14035) fuses LiDAR, inertial, and camera measurements in a unified framework without separate feature extraction for either points or images.

---

## 2.3 IMU Model

Whereas cameras and LiDAR observe external landmarks at 10–30 Hz, an Inertial Measurement Unit (IMU) pairs a 3-axis accelerometer and a 3-axis gyroscope to measure self-motion internally at 100 Hz to 1 kHz. Those high-rate measurements bridge the gaps between visual or LiDAR frames, deskew scans, and provide metric scale and gravity observability during initialization. Exploiting inertial data safely, however, requires modeling its bias dynamics and stochastic noise precisely.

### 2.3.1 Gyroscope Error Model

A gyroscope measures 3-axis angular velocity $\boldsymbol{\omega}$ in the IMU body frame. In practice, the raw reading $\tilde{\boldsymbol{\omega}}$ is corrupted by a slowly drifting offset and broadband noise:

$$\tilde{\boldsymbol{\omega}} = \boldsymbol{\omega} + \mathbf{b}_g + \mathbf{n}_g$$

where:
- $\boldsymbol{\omega}$: true angular velocity in the IMU body frame
- $\mathbf{b}_g$: **bias**, a quasi-constant offset that drifts slowly over time
- $\mathbf{n}_g$: **measurement noise**, modeled as additive white Gaussian noise (AWGN)

**Bias dynamics.** Because temperature shifts and mechanical stress cause the gyroscope bias to wander gradually, estimators model its evolution as a **random walk**:

$$\dot{\mathbf{b}}_g = \mathbf{n}_{bg}$$

driven by zero-mean Gaussian noise $\mathbf{n}_{bg} \sim \mathcal{N}(\mathbf{0}, \sigma_{bg}^2 \mathbf{I})$. Discretizing over sampling interval $\Delta t$ gives:

$$\mathbf{b}_{g,k+1} = \mathbf{b}_{g,k} + \sigma_{bg} \sqrt{\Delta t} \cdot \mathbf{w}_k, \quad \mathbf{w}_k \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$$

To calibrate these gyroscope parameters reliably for a specific sensor and operating profile:

- **White-noise density**: cross-check the device datasheet against the $-1/2$ slope region of an Allan deviation plot, verifying that continuous-time density units match the discrete covariance conversion used in code.
- **In-run bias stability**: read the nearly flat minimum of an Allan deviation computed from a sufficiently long static log. Because temperature, mounting stress, vibration, and recording length all shift this floor, never copy a datasheet headline figure blindly.
- **Bias-random-walk driving noise**: extract $\sigma_{bg}$ from the $+1/2$ long-term slope, keeping measurement noise $\sigma_g$ strictly separate from bias driving noise $\sigma_{bg}$ and documenting the continuous-to-discrete covariance conversion.

### 2.3.2 Accelerometer Error Model

An accelerometer measures 3-axis **specific force** $\mathbf{a}$—the non-gravitational acceleration exerted on the sensor, which equals true linear acceleration minus gravity. Consequently, the raw measurement $\tilde{\mathbf{a}}$ takes the form:

$$\tilde{\mathbf{a}} = \mathbf{R}_{bw}(\mathbf{a}_w - \mathbf{g}_w) + \mathbf{b}_a + \mathbf{n}_a$$

where:
- $\mathbf{a}_w$: true linear acceleration in the world frame
- $\mathbf{g}_w = [0, 0, -g]^\top$: world-frame gravity vector ($g \approx 9.81\,\text{m/s}^2$)
- $\mathbf{R}_{bw}$: rotation matrix transforming vectors from the world frame to the body frame
- $\mathbf{b}_a$: accelerometer bias
- $\mathbf{n}_a \sim \mathcal{N}(\mathbf{0}, \sigma_a^2 \mathbf{I})$: measurement noise. Because $\sigma_a$ is specified as a continuous-time noise density ($\text{m/s}^2/\sqrt{\text{Hz}}$), discrete samples at period $\Delta t$ carry standard deviation $\sigma_a/\sqrt{\Delta t}$ (in contrast to bias random walk, which scales with $\sqrt{\Delta t}$).

**Role of gravity.** The fact that an accelerometer senses gravity plays a decisive role in inertial fusion. Even when stationary, a z-up sensor reads $[0, 0, g]^\top$ rather than zero, letting the estimator observe pitch and roll directly from the gravity vector. Yaw, however, rotates around the gravity axis and leaves the gravity projection unchanged—making global yaw unobservable from an IMU alone. Standalone VIO and LIO therefore fix an arbitrary initial yaw and track relative heading, relying on a magnetometer, GNSS course, or prior map alignment whenever global yaw is required.

**Bias dynamics.** Just like the gyroscope bias, accelerometer bias is modeled as a continuous-time random walk:

$$\dot{\mathbf{b}}_a = \mathbf{n}_{ba}, \quad \mathbf{n}_{ba} \sim \mathcal{N}(\mathbf{0}, \sigma_{ba}^2 \mathbf{I})$$

Parameterizing the accelerometer follows the same discipline as the gyroscope:

- **White-noise density**: reconcile the datasheet with the $-1/2$ Allan deviation slope, stating explicit unit conversions among $\text{m/s}^2/\sqrt{\text{Hz}}$, $\text{mg}/\sqrt{\text{Hz}}$, and VRW.
- **In-run bias stability**: inspect the flat Allan floor and repeatability across static thermal runs, taking care that unmodeled gravity-projection errors or mounting tilts do not masquerade as bias.
- **Bias-random-walk driving noise**: estimate $\sigma_{ba}$ from the positive long-term Allan slope and convert it cleanly to discrete covariance at the filter's sampling rate.

### 2.3.3 Allan Variance

How do we separate white noise, flicker instability, and bias random walk from a single sensor stream? **Allan variance** is the standard time-domain tool for characterizing IMU noise. By recording several hours of static data and computing the variance of consecutive cluster averages across varying averaging times $\tau$, distinct noise processes reveal themselves at different timescales:

$$\sigma^2(\tau) = \frac{1}{2} \langle (\bar{y}_{k+1} - \bar{y}_k)^2 \rangle$$

where $\bar{y}_k$ is the average of samples within the $k$-th non-overlapping cluster of duration $\tau$.

Plotted on log-log axes, the slope of the Allan deviation $\sigma(\tau)$ identifies the dominant noise mechanism at each timescale:

| Slope | Noise type | Physical meaning |
|--------|-----------|-----------|
| $-1/2$ | Angle/velocity random walk (ARW/VRW) | White noise $\mathbf{n}_g, \mathbf{n}_a$ |
| $0$ | Bias instability | Flicker noise; the minimum is the bias stability |
| $+1/2$ | Rate random walk (RRW) | Bias random walk $\mathbf{n}_{bg}, \mathbf{n}_{ba}$ |

**Reading an IMU datasheet alongside an Allan plot.** Four headline quantities map directly from vendor datasheets and Allan curves into filter noise matrices:

1. **Angular random walk (ARW)**: reported in $°/\sqrt{\text{hr}}$ or $\text{rad/s}/\sqrt{\text{Hz}}$ and read at $\tau = 1\,\text{s}$ along the $-1/2$ slope line; this sets $\sigma_g$.
2. **Velocity random walk (VRW)**: reported in $\text{m/s}/\sqrt{\text{hr}}$ or $\text{m/s}^2/\sqrt{\text{Hz}}$, giving the accelerometer white-noise density $\sigma_a$.
3. **In-run bias stability**: often quoted by vendors as the minimum of the Allan deviation curve. Under IEEE Std 952, the bias-instability coefficient $B$ satisfies $\sigma(\tau) \approx 0.664B$ in the flat region, so $B$ equals the curve minimum divided by 0.664 (matching the `bias_instability` value computed in Chapter 3). Remember that this metric reflects static time-averaged stability rather than a lower bound on dynamic bias estimation error inside a fusion filter.
4. **Rate random walk**: captures long-term bias drift from the $+1/2$ slope region, yielding $\sigma_{bg}$ and $\sigma_{ba}$.

The following implementation computes the non-overlapping Allan deviation curve from static IMU logs and extracts the white-noise and bias-instability metrics.

```python
import numpy as np

def compute_allan_variance(data, dt, max_cluster_size=None):
    """Compute the Allan variance.
    
    Args:
        data: (N,) data for one IMU axis collected at rest
        dt: sampling period (seconds)
        max_cluster_size: maximum cluster size (defaults to N//10 when None: at least 10 clusters even at the longest tau)
    
    Returns:
        taus: array of cluster times
        adevs: array of Allan deviations
    """
    N = len(data)
    if max_cluster_size is None:
        max_cluster_size = N // 10  # with only 2-3 clusters at long tau, the estimate is so noisy that the minimum is underestimated
    
    # Cluster sizes generated on a log scale
    cluster_sizes = np.unique(np.logspace(
        0, np.log10(max_cluster_size), num=100
    ).astype(int))
    cluster_sizes = cluster_sizes[cluster_sizes > 0]
    
    taus = []
    adevs = []
    
    for m in cluster_sizes:
        tau = m * dt
        # Non-overlapping mean
        n_clusters = N // m
        if n_clusters < 2:
            break
        
        # Mean of each cluster
        truncated = data[:n_clusters * m].reshape(n_clusters, m)
        cluster_means = truncated.mean(axis=1)
        
        # Allan variance
        diff = np.diff(cluster_means)
        avar = 0.5 * np.mean(diff**2)
        
        taus.append(tau)
        adevs.append(np.sqrt(avar))
    
    return np.array(taus), np.array(adevs)


def extract_imu_params(taus, adevs):
    """Extract IMU noise parameters from an Allan deviation plot.
    
    Args:
        taus: array of cluster times
        adevs: array of Allan deviations
    
    Returns:
        dict: {
            'white_noise': value at tau=1 (ARW or VRW),
            'bias_instability': minimum value,
            'bias_instability_tau': tau at the minimum
        }
    """
    # White noise: Allan deviation at tau=1 (slope -1/2 region)
    idx_1s = np.argmin(np.abs(taus - 1.0))
    white_noise = adevs[idx_1s]
    
    # Bias instability: minimum of the Allan deviation
    idx_min = np.argmin(adevs)
    bias_instability = adevs[idx_min]
    bias_instability_tau = taus[idx_min]
    
    return {
        'white_noise': white_noise,
        'bias_instability': bias_instability,
        'bias_instability_tau': bias_instability_tau
    }
```

### 2.3.4 Strapdown Navigation Equation

Having characterized IMU errors, we now turn to the equations that propagate raw inertial readings into a full 3D navigation state (attitude, velocity, and position). These are the **strapdown navigation equations**—so named because the IMU is rigidly bolted ("strapped down") to the vehicle body, shifting frame alignment from a mechanical gimbal into software.

Under the standard VIO/LIO assumption that neglects Earth rotation and treats the local world frame as inertial, the continuous-time kinematics of the world-frame state $[\mathbf{R}, \mathbf{v}, \mathbf{p}]$ evolve as follows (whereas Earth-fixed navigation frames such as NED/ENU add Earth-rate and transport-rate correction terms):

$$\dot{\mathbf{R}} = \mathbf{R} \cdot [\tilde{\boldsymbol{\omega}} - \mathbf{b}_g - \mathbf{n}_g]_\times$$

$$\dot{\mathbf{v}} = \mathbf{R} (\tilde{\mathbf{a}} - \mathbf{b}_a - \mathbf{n}_a) + \mathbf{g}$$

$$\dot{\mathbf{p}} = \mathbf{v}$$

where:
- $\mathbf{R} \in SO(3)$: body-to-world rotation matrix
- $\mathbf{v} \in \mathbb{R}^3$: world-frame velocity
- $\mathbf{p} \in \mathbb{R}^3$: world-frame position
- $[\cdot]_\times$: skew-symmetric matrix operator representing the vector cross product as matrix multiplication:

$$[\boldsymbol{\omega}]_\times = \begin{bmatrix} 0 & -\omega_z & \omega_y \\ \omega_z & 0 & -\omega_x \\ -\omega_y & \omega_x & 0 \end{bmatrix}$$

**Discrete-time (Euler) integration.** Over a short sampling interval $\Delta t$ from $t_k$ to $t_{k+1} = t_k + \Delta t$, first-order Euler integration propagates the nominal state via:

$$\mathbf{R}_{k+1} = \mathbf{R}_k \cdot \text{Exp}\left((\tilde{\boldsymbol{\omega}}_k - \mathbf{b}_{g,k}) \Delta t\right)$$

$$\mathbf{v}_{k+1} = \mathbf{v}_k + \mathbf{g} \Delta t + \mathbf{R}_k (\tilde{\mathbf{a}}_k - \mathbf{b}_{a,k}) \Delta t$$

$$\mathbf{p}_{k+1} = \mathbf{p}_k + \mathbf{v}_k \Delta t + \frac{1}{2} \mathbf{g} \Delta t^2 + \frac{1}{2} \mathbf{R}_k (\tilde{\mathbf{a}}_k - \mathbf{b}_{a,k}) \Delta t^2$$

Here $\text{Exp}(\boldsymbol{\phi})$ is the $SO(3)$ exponential map converting a rotation vector $\boldsymbol{\phi}$ into a rotation matrix via Rodrigues' formula:

$$\text{Exp}(\boldsymbol{\phi}) = \mathbf{I} + \frac{\sin\|\boldsymbol{\phi}\|}{\|\boldsymbol{\phi}\|} [\boldsymbol{\phi}]_\times + \frac{1 - \cos\|\boldsymbol{\phi}\|}{\|\boldsymbol{\phi}\|^2} [\boldsymbol{\phi}]_\times^2$$

**Midpoint integration.** First-order Euler integration assumes constant angular rate and acceleration across $\Delta t$. Averaging consecutive IMU samples achieves second-order accuracy with minimal extra cost:

$$\bar{\boldsymbol{\omega}} = \frac{1}{2}(\tilde{\boldsymbol{\omega}}_k + \tilde{\boldsymbol{\omega}}_{k+1}) - \mathbf{b}_{g,k}$$

$$\mathbf{R}_{k+1} = \mathbf{R}_k \cdot \text{Exp}(\bar{\boldsymbol{\omega}} \Delta t)$$

$$\bar{\mathbf{a}} = \frac{1}{2}\left(\mathbf{R}_k(\tilde{\mathbf{a}}_k - \mathbf{b}_{a,k}) + \mathbf{R}_{k+1}(\tilde{\mathbf{a}}_{k+1} - \mathbf{b}_{a,k})\right)$$

$$\mathbf{v}_{k+1} = \mathbf{v}_k + (\bar{\mathbf{a}} + \mathbf{g}) \Delta t$$

$$\mathbf{p}_{k+1} = \mathbf{p}_k + \mathbf{v}_k \Delta t + \frac{1}{2}(\bar{\mathbf{a}} + \mathbf{g}) \Delta t^2$$

Midpoint integration is widely used in VIO pipelines such as VINS-Mono, though every estimator has its own discretization details (for example, check FAST-LIO2's propagation equations directly in its implementation). Even a high IMU output rate does not eliminate discretization error under intense angular acceleration or sampling jitter, so choose among midpoint, RK4, and preintegration based on your platform's motion profile and error budget.

The Python implementation below performs midpoint strapdown propagation given gyroscope and accelerometer streams along with fixed bias estimates.

```python
import numpy as np
from scipy.spatial.transform import Rotation

def skew(v):
    """Skew-symmetric matrix of a 3D vector."""
    return np.array([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0]
    ])

def exp_so3(phi):
    """SO(3) exponential map: rotation vector -> rotation matrix (Rodrigues' formula)."""
    angle = np.linalg.norm(phi)
    if angle < 1e-10:
        return np.eye(3) + skew(phi)
    
    axis = phi / angle
    K = skew(axis)
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * K @ K

def imu_strapdown(gyro_data, accel_data, dt, R0, v0, p0, bg, ba, gravity):
    """Strapdown inertial navigation (midpoint integration).
    
    Args:
        gyro_data: (N, 3) gyroscope measurements [rad/s]
        accel_data: (N, 3) accelerometer measurements [m/s^2]
        dt: sampling period [s]
        R0: (3,3) initial rotation matrix (body-to-world)
        v0: (3,) initial velocity [m/s] (world frame)
        p0: (3,) initial position [m] (world frame)
        bg: (3,) gyroscope bias
        ba: (3,) accelerometer bias
        gravity: (3,) gravity vector (e.g., [0, 0, -9.81])
    
    Returns:
        Rs: (N+1, 3, 3) rotation history
        vs: (N+1, 3) velocity history
        ps: (N+1, 3) position history
    """
    N = len(gyro_data)
    Rs = np.zeros((N + 1, 3, 3))
    vs = np.zeros((N + 1, 3))
    ps = np.zeros((N + 1, 3))
    
    Rs[0] = R0
    vs[0] = v0
    ps[0] = p0
    
    for k in range(N - 1):
        # Midpoint gyroscope
        omega_k = gyro_data[k] - bg
        omega_k1 = gyro_data[k + 1] - bg
        omega_mid = 0.5 * (omega_k + omega_k1)
        
        # Rotation update
        Rs[k + 1] = Rs[k] @ exp_so3(omega_mid * dt)
        
        # Midpoint acceleration (world frame)
        a_k = Rs[k] @ (accel_data[k] - ba)
        a_k1 = Rs[k + 1] @ (accel_data[k + 1] - ba)
        a_mid = 0.5 * (a_k + a_k1)
        
        # Velocity/position update
        vs[k + 1] = vs[k] + (a_mid + gravity) * dt
        ps[k + 1] = ps[k] + vs[k] * dt + 0.5 * (a_mid + gravity) * dt**2
    
    # Last step (simple Euler)
    k = N - 1
    omega_k = gyro_data[k] - bg
    Rs[k + 1] = Rs[k] @ exp_so3(omega_k * dt)
    a_k = Rs[k] @ (accel_data[k] - ba)
    vs[k + 1] = vs[k] + (a_k + gravity) * dt
    ps[k + 1] = ps[k] + vs[k] * dt + 0.5 * (a_k + gravity) * dt**2
    
    return Rs, vs, ps
```

**How fast does uncorrected drift grow?** A quick numerical check shows why unaided inertial navigation cannot run open-loop for long. Suppose a low-cost MEMS accelerometer carries an uncompensated bias of just $b_a = 0.01\,\text{m/s}^2$ (roughly $1\,\text{mg}$):

- After 1 s: $\frac{1}{2} \times 0.01 \times 1^2 = 0.005\,\text{m}$ (5 mm)
- After 10 s: $\frac{1}{2} \times 0.01 \times 100 = 0.5\,\text{m}$
- After 60 s: $\frac{1}{2} \times 0.01 \times 3600 = 18\,\text{m}$

Within a single minute, position error blows past typical robotic tolerances. How long an IMU remains useful open-loop depends on sensor quality, thermal stability, motion profile, initialization, and the mission's error budget—and although higher-grade INS units stretch that window, quadratic drift still wins eventually. That is why VIO and LIO pipelines **augment the state vector with the biases $\mathbf{b}_g, \mathbf{b}_a$** and refine them continuously from external camera or LiDAR updates. In parallel, learning-based inertial odometry is also advancing: [AirIO (Qiu et al., 2025)](https://arxiv.org/abs/2501.15659) enhances IMU feature observability and reports over 50% accuracy gains compared with earlier learned inertial methods on drone benchmarks.

### 2.3.5 IMU Grade Classification

Marketing labels such as *navigation grade*, *tactical grade*, *industrial*, and *consumer* lack standardized boundaries across vendors and application domains—and "MEMS" describes a silicon fabrication process, not a performance tier. Rather than picking an IMU from a generic grade chart or static price table, evaluate candidate hardware against explicit performance criteria and on-platform tests:

| Selection criterion | What to inspect | Validation evidence | Design impact |
|----------|------------|----------|----------|
| White noise | Gyro/accelerometer noise density and ARW/VRW units | Device datasheet + Allan test | Short-term propagation error |
| Bias | In-run stability, random walk, temperature coefficient | Repeated static/temperature tests + Allan test | Unaided drift and aiding rate |
| Dynamic range | Range, bandwidth, saturation, and clipping | Target motion/vibration profile | Observation loss during aggressive motion |
| Timing | Output rate, timestamps, and clock/trigger support | Latency and jitter measurements | Temporal calibration and fusion consistency |
| Environmental tolerance | Temperature, vibration, shock, and mounting stress | Vendor qualification + target-platform test | Repeatability and long-term stability |

Feed those measured noise and timing parameters directly into the mission error budget to determine what external sensors and aiding rates are needed. Because pricing fluctuates with volume and procurement date, weigh current hardware quotes alongside the engineering cost of calibration and time synchronization.

---

## 2.4 GNSS Model

Because inertial integration drifts without bound, outdoor systems often pair IMUs with a Global Navigation Satellite System (GNSS)—a family encompassing GPS, GLONASS, Galileo, and BeiDou—to obtain global position fixes.

### 2.4.1 Pseudorange Observation Model

At the most basic level, a GNSS receiver measures the apparent travel time of a coded radio signal from each visible satellite, multiplying by the speed of light $c$ to obtain a **pseudorange** $\rho$—equal to the true geometric distance plus clock offsets and atmospheric delays:

$$\rho^s = r^s + c \cdot \delta t_r - c \cdot \delta t^s + I^s + T^s + \epsilon_\rho$$

where:
- $r^s = \|\mathbf{p}_r - \mathbf{p}^s\|$: true geometric distance between receiver position $\mathbf{p}_r$ and satellite $s$ position $\mathbf{p}^s$
- $c \cdot \delta t_r$: unknown receiver clock bias, solved jointly with position
- $c \cdot \delta t^s$: satellite clock bias, broadcast and corrected via the navigation message
- $I^s$: ionospheric delay caused by free electrons in the upper atmosphere (largely eliminated with dual-frequency L1/L2 receivers)
- $T^s$: tropospheric delay caused by dry air and water vapor in the lower atmosphere (typically compensated via models such as Saastamoinen)
- $\epsilon_\rho$: unmodeled residual error from receiver thermal noise and multipath, typically $\sigma_\rho \approx 1\text{–}5\,\text{m}$ for single-point positioning and $\sigma_\rho \approx 0.1\text{–}0.3\,\text{m}$ with dual-frequency corrections

**Positioning principle.** Solving for the three receiver coordinates $(x, y, z)$ plus the unknown receiver clock offset $\delta t_r$ requires pseudoranges from at least four satellites. Solving these nonlinear equations via least squares or a Kalman filter constitutes standard Single Point Positioning (SPP).

**Multipath.** In dense urban streets, satellite signals bounce off glass and concrete facades before reaching the antenna, arriving along indirect paths that add meters—or tens of meters—of positive bias to the pseudorange. Because multipath shifts erratically with vehicle motion and surrounding geometry, robust outlier gating is indispensable whenever GNSS enters a fusion estimator.

### 2.4.2 Carrier Phase Observation Model

Beyond code pseudoranges, precision receivers track the fractional phase of the underlying RF carrier wave. While carrier-phase noise sits at the millimeter level—roughly two orders of magnitude cleaner than pseudorange noise—that alone does not yield millimeter positioning. On the L1 band (about 1575.42 MHz), one carrier cycle spans roughly 19 cm, meaning the receiver must still resolve or model the unknown integer cycle count, cycle slips, multipath, and atmospheric delays:

$$\Phi^s = r^s + c \cdot \delta t_r - c \cdot \delta t^s + \lambda N^s - I^s + T^s + \epsilon_\Phi$$

where:
- $\lambda$: carrier wavelength
- $N^s$: **integer ambiguity**, the unknown whole number of carrier cycles between satellite and receiver (resolved to integers in fixed RTK and PPP-AR, or floated as real values in conventional PPP)
- $\epsilon_\Phi \approx 1\text{–}5\,\text{mm}$: carrier-phase measurement noise (roughly 1/100 of pseudorange noise)

Notice that the ionospheric delay $I^s$ enters with the opposite sign here compared with the pseudorange equation (reflecting group velocity versus phase velocity). Dual-frequency combinations eliminate or estimate first-order ionospheric error, leaving higher-order atmospheric effects and local errors to be managed.

### 2.4.3 RTK (Real-Time Kinematic)

To unlock centimeter-level precision from carrier phase in real time, **RTK** combines observations from the moving rover with corrections from a nearby base station or network. Because atmospheric delays remain strongly correlated over short baselines, differencing cancels common errors—though that spatial correlation weakens as baseline length grows, and centimeter-level accuracy still depends on fixing the integer ambiguities cleanly under good signal quality.

**Double differencing.** Forming the difference across both the rover and base station as well as a pair of satellites (target $s$ and a reference satellite) gives the double-difference carrier-phase observation:

$$\nabla\Delta\Phi_{br}^{sr} = \nabla\Delta r_{br}^{sr} + \lambda \nabla\Delta N_{br}^{sr} + \epsilon$$

Double differencing eliminates both receiver and satellite clock offsets completely and nearly cancels ionospheric and tropospheric delays on short baselines. What remains are the geometric range difference (a function of rover position) and the integer double-difference ambiguities, which dedicated integer least-squares solvers such as LAMBDA resolve rapidly.

### 2.4.4 PPP (Precise Point Positioning)

**PPP** achieves centimeter-level accuracy using a single standalone receiver—no local base station required. Instead of differencing against a nearby reference, PPP ingests precise satellite orbit and clock products from an external network service to strip out satellite-side errors, while estimating tropospheric and ionospheric delays directly inside the navigation state.

The table below compares the practical trade-offs between RTK and PPP:

| Property | RTK | PPP |
|------|-----|-----|
| Base station required | Yes (typically within 10–20 km of a single base) | No |
| Convergence time | seconds to tens of seconds | tens of minutes (traditional), minutes (PPP-AR) |
| Precision (after convergence) | $\sim 2\,\text{cm}$ | $\sim 5\,\text{cm}$ |
| Coverage | Near the base station | Global |

**Integrating GNSS into sensor fusion.** Global GNSS fixes anchor long-term VIO or LIO drift, though residual bias and outages still arise under multipath, frame misalignment, lever-arm offsets, or clock skew. Frameworks such as [LIO-SAM (Shan et al., 2020)](https://arxiv.org/abs/2007.00258) selectively insert qualified GNSS factors into a factor graph, guided by three essential checks:

1. **Coordinate frame transformation**: receivers report geodetic coordinates in WGS84 (latitude, longitude, ellipsoidal height), whereas robotics estimators operate in local Cartesian frames such as ENU (East-North-Up) or NED (North-East-Down), requiring a local tangent-plane conversion.
2. **Proper covariance scaling**: transform the receiver's 3D position covariance into the local frame before weighting measurements. Note that DOP (Dilution of Precision) is a dimensionless geometric factor, not a covariance matrix—it must be multiplied by an empirical measurement-error scale to reflect true uncertainty.
3. **Outlier rejection**: because multipath can corrupt fixes by tens of meters, apply robust loss kernels or $\chi^2$ gating to reject degraded fixes before they pull the trajectory off course.

The two Python functions below illustrate a simplified ECEF pseudorange measurement model (along with its Jacobian with respect to position and clock bias) and a local tangent-plane conversion from WGS84 to ENU.

```python
import numpy as np

def pseudorange_model(p_receiver, p_satellites, clock_bias):
    """Pseudorange observation model (simplified).
    
    Args:
        p_receiver: (3,) receiver position in ECEF [m]
        p_satellites: (N, 3) ECEF positions of each satellite [m]
        clock_bias: receiver clock offset [m] (c * dt_r)
    
    Returns:
        pseudoranges: (N,) predicted pseudoranges [m]
        H: (N, 4) observation Jacobian (at the linearization point)
    """
    N = len(p_satellites)
    pseudoranges = np.zeros(N)
    H = np.zeros((N, 4))
    
    for i in range(N):
        diff = p_receiver - p_satellites[i]
        r = np.linalg.norm(diff)
        pseudoranges[i] = r + clock_bias
        
        # Jacobian: d(rho)/d(x,y,z,cb)
        e = diff / r  # unit vector (satellite-to-receiver direction; diff = p_receiver - p_satellite)
        H[i, :3] = e
        H[i, 3] = 1.0  # partial derivative with respect to clock bias
    
    return pseudoranges, H


def geodetic_to_enu(lat, lon, alt, lat0, lon0, alt0):
    """Convert WGS84 coordinates to local ENU coordinates.
    
    Args:
        lat, lon: latitude and longitude of the point to convert [rad]
        alt: ellipsoidal altitude [m]
        lat0, lon0, alt0: latitude, longitude, altitude of the origin [rad, rad, m]
    
    Returns:
        (3,) ENU coordinates [m]
    """
    # Simplified conversion (WGS84 Earth radius)
    a = 6378137.0  # WGS84 semi-major axis
    e2 = 0.00669437999014  # squared eccentricity
    
    sin_lat0 = np.sin(lat0)
    N0 = a / np.sqrt(1 - e2 * sin_lat0**2)
    
    dlat = lat - lat0
    dlon = lon - lon0
    dalt = alt - alt0
    
    east = (N0 + alt0) * np.cos(lat0) * dlon
    north = (N0 * (1 - e2) / (1 - e2 * sin_lat0**2) + alt0) * dlat  # meridional radius of curvature M
    up = dalt
    
    return np.array([east, north, up])
```

---

## 2.5 Radar Model

Unlike cameras and time-of-flight LiDAR—both of which suffer in heavy rain, fog, snow, or dust—radio-frequency radar penetrates adverse weather reliably. Equally important, radar measures **relative radial velocity directly** in a single shot via the Doppler effect, rather than differentiating noisy position snapshots across frames.

### 2.5.1 FMCW Radar Principles

Automotive and robotics radars predominantly use **FMCW (Frequency Modulated Continuous Wave)** signaling. Instead of timing discrete light pulses, an FMCW radar transmits a continuous radio wave whose frequency sweeps linearly over time (a *chirp*) and recovers range and velocity from the frequency and phase differences between the transmitted wave and its echo.

**Chirp signal.** During a single chirp of duration $T_c$, the transmitted frequency ramps linearly across bandwidth $B$ starting from $f_0$:

$$f_{\text{TX}}(t) = f_0 + \frac{B}{T_c} t$$

**Beat frequency and range.** An echo returning from a target at range $R$ arrives after a round-trip delay $\tau = 2R/c$ ($c$: speed of light). Mixing the delayed echo with the live transmitted chirp produces a constant difference tone—the **beat frequency**:

$$f_b = \frac{2BR}{cT_c}$$

Solving for range $R$ gives:

$$R = \frac{f_b \cdot c \cdot T_c}{2B}$$

**Range resolution.** Two targets can be separated in range only if their beat tones resolve distinctly over the chirp duration, setting a bandwidth-limited resolution floor:

$$\Delta R = \frac{c}{2B}$$

With a 77 GHz automotive radar sweeping $B = 4\,\text{GHz}$, for example, $\Delta R \approx 3.75\,\text{cm}$.

**Doppler velocity.** Transmitting a burst of consecutive chirps and tracking the subtle phase shift across echoes from the same target yields its radial (line-of-sight) velocity:

$$v_r = \frac{\lambda \cdot f_d}{2}$$

where $f_d$ is the Doppler shift and $\lambda$ is the carrier wavelength. At 77 GHz, $\lambda \approx 3.9\,\text{mm}$, making the radar sensitive to even small velocity changes.

**Range-Doppler map and angle of arrival.** Applying a 2D FFT across fast-time samples (within each chirp) and slow-time chirps (across a burst) produces a **Range-Doppler map**, where each peak pinpoints a reflector's range and radial velocity simultaneously. Finally, measuring phase differences across multiple receive antennas yields angle of arrival—horizontal arrays resolve azimuth, while vertical arrays resolve elevation.

### 2.5.2 Recent Trends in 4D Imaging Radar

Conventional automotive radar captures only three dimensions—range, Doppler velocity, and azimuth—leaving elevation virtually unresolved. **4D imaging radar** adds high-resolution **elevation**, outputting dense 3D point clouds annotated with per-point radial velocity.

To achieve finer angular resolution without physically massive dishes, 4D imaging radars synthesize large virtual antenna arrays via MIMO (Multiple Input Multiple Output): for instance, 12 transmit antennas paired with 16 receive antennas form $12 \times 16 = 192$ virtual channels, with horizontal and vertical angular resolution governed by the virtual array's aperture along each axis.

The table below contrasts 4D imaging radar with LiDAR across key perception properties:

| Property | 4D Imaging Radar | LiDAR |
|------|-----------------|-------|
| Adverse-weather operation | Excellent | Weak |
| Point density | Medium (thousands of points/frame) | High (hundreds of thousands of points/frame) |
| Velocity measurement | Direct (Doppler) | Not available (requires two-frame differencing; FMCW LiDAR is the exception) |
| Angular resolution | $\sim 1°$ | $\sim 0.1°$ |
| Cost | Low to medium | Medium to high |
| Static-object detection | Possible (depends on detection and clutter-removal settings) | Excellent |

**Leveraging radar inside sensor fusion.** Radar's direct Doppler measurement brings three distinct advantages to a multi-sensor stack:

1. **Instantaneous ego-velocity estimation**: Fitting observed radial velocities against static background reflectors recovers the platform's 3D linear velocity directly—without integrating noisy accelerometers and without temporal drift:

$$v_r^{(i)} = -\mathbf{e}^{(i)\top} \mathbf{v}_{\text{ego}}$$

where $\mathbf{e}^{(i)}$ is the unit bearing vector toward the $i$-th static reflector and $\mathbf{v}_{\text{ego}}$ is the radar's velocity. With three or more non-coplanar reflectors, least squares solves for $\mathbf{v}_{\text{ego}}$ in closed form.

2. **Dynamic-object segmentation**: Points whose measured Doppler velocities deviate sharply from the static-world prediction immediately stand out as moving vehicles or pedestrians.
3. **All-weather redundancy**: When fog, heavy rain, or dust blinds cameras and LiDAR, radar provides an independent exteroceptive anchor—though real-world range and clutter still depend on precipitation intensity, frequency band, antenna design, and signal-processing thresholds.

The Python functions below implement beat-frequency range conversion and least-squares ego-velocity estimation from static Doppler returns.

```python
import numpy as np

def fmcw_range_from_beat(f_beat, bandwidth, chirp_time, c=3e8):
    """Compute range from an FMCW radar beat frequency.
    
    Args:
        f_beat: beat frequency [Hz]
        bandwidth: frequency bandwidth [Hz]
        chirp_time: chirp duration [s]
        c: speed of light [m/s]
    
    Returns:
        range_m: range [m]
    """
    return f_beat * c * chirp_time / (2 * bandwidth)


def estimate_ego_velocity(bearings, doppler_velocities):
    """Estimate ego-velocity from Doppler observations of static reflectors.
    
    v_doppler_i = -e_i^T @ v_ego  (assumes a static environment)
    
    Args:
        bearings: (N, 3) unit vectors toward each reflector
        doppler_velocities: (N,) radial velocity of each reflector [m/s]
    
    Returns:
        v_ego: (3,) estimated ego-velocity [m/s]
    """
    # -E @ v_ego = v_doppler  ->  v_ego = -(E^T E)^{-1} E^T v_doppler
    E = bearings  # (N, 3)
    v_ego = -np.linalg.lstsq(E, doppler_velocities, rcond=None)[0]
    return v_ego
```

Because radar measures radial velocity directly—an observable that neither standard cameras nor time-of-flight LiDAR provide—it plays an increasingly central role in modern multi-sensor fusion architectures (→ Chapter 8).

---

## 2.6 Other Sensors

Beyond cameras, LiDAR, IMUs, GNSS, and radar, practical robots frequently incorporate auxiliary sensors. Though insufficient for full 3D navigation on their own, these lightweight modalities provide targeted constraints—such as wheel motion, vertical altitude, magnetic heading, or indoor range—that shore up primary sensors in degenerate regimes.

### 2.6.1 Wheel Odometry

Wheel encoders are among the simplest proprioceptive sensors, counting axle rotations to infer ground displacement.

**Observation model.** For a differential-drive mobile robot with wheel radius $r$ and track width $d$, incremental left and right wheel rotations $\Delta \theta_L$ and $\Delta \theta_R$ yield forward travel $\Delta s$ and heading change $\Delta \psi$:

$$\Delta s = \frac{r(\Delta \theta_L + \Delta \theta_R)}{2}, \quad \Delta \psi = \frac{r(\Delta \theta_R - \Delta \theta_L)}{d}$$

For passenger vehicles using **Ackermann steering**, the kinematic bicycle approximation expresses body motion in terms of rear-wheel speed $v$, steering angle $\delta$, and wheelbase $L$:

$$\dot{x} = v \cos\psi, \quad \dot{y} = v \sin\psi, \quad \dot{\psi} = \frac{v \tan\delta}{L}$$

**Slip model.** Real tires never roll without slip—especially on wet pavement, loose gravel, or during hard acceleration and braking. The longitudinal **slip ratio** quantifies this discrepancy:

$$s = \frac{v_{\text{wheel}} - v_{\text{actual}}}{v_{\text{actual}}}$$

Because wheel odometry accuracy collapses under slip, fusion filters employ an **adaptive observation covariance**: whenever slip is detected (for example, by comparing wheel speed against IMU or visual motion), wheel-odometry uncertainty is inflated so other sensors govern the estimate.

**Role in sensor fusion.** When tires grip well, wheel odometry delivers crisp short-term velocity and displacement updates—especially along straight paths—suppressing IMU drift inside featureless tunnels or long corridors. Several extensions of VINS-Mono, for example, add explicit wheel-odometry factors to prevent scale and velocity drift during constant-velocity driving.

### 2.6.2 Barometer

A barometer estimates **altitude** from ambient atmospheric pressure.

**Observation model.** Under the International Standard Atmosphere (ISA), pressure $P$ maps to altitude $h$ via:

$$h = \frac{T_0}{L}\left(1 - \left(\frac{P}{P_0}\right)^{\frac{RL}{g_0}}\right)$$

where:
- $P$: measured atmospheric pressure [Pa]
- $P_0 = 101325\,\text{Pa}$: standard sea-level pressure
- $T_0 = 288.15\,\text{K}$: standard sea-level temperature
- $L = 0.0065\,\text{K/m}$: temperature lapse rate
- $R = 287.053\,\text{J/(kg}\cdot\text{K)}$: specific gas constant for dry air
- $g_0 = 9.80665\,\text{m/s}^2$: standard gravitational acceleration

Near sea level, this nonlinear relation linearizes cleanly:

$$\Delta h \approx -\frac{\Delta P}{\rho g} \approx -\frac{\Delta P}{12.0}\,[\text{m}], \quad (\Delta P\text{ in Pa})$$

meaning roughly every $12.0\,\text{Pa}$ drop corresponds to $1\,\text{m}$ of climb.

**Noise characteristics:**
- Short-term precision: $\pm 0.1\text{–}0.5\,\text{m}$ (highly responsive to small vertical motions)
- Long-term precision: $\pm 1\text{–}10\,\text{m}$ (slow drift driven by shifting weather fronts)

**Role in sensor fusion.** Barometers provide a direct vertical constraint that arrests IMU vertical drift, making them indispensable for drone altitude hold. Because ambient barometric pressure drifts with changing weather, however, estimators generally fuse relative pressure differences rather than raw absolute altitude—and indoors, must guard against transient pressure spikes caused by HVAC systems or opening doors.

### 2.6.3 Magnetometer

A 3-axis magnetometer measures the local magnetic field vector. When roll and pitch are known, hard- and soft-iron distortions are calibrated out, and external magnetic interference is minimal, it supplies a heading relative to geomagnetic north (which converts to true north once local magnetic declination is added).

**Observation model.** In practice, nearby ferromagnetic structures and electronics distort Earth's field $\mathbf{m}_w$ (roughly $25\text{–}65\,\mu\text{T}$ depending on geographic location):

$$\tilde{\mathbf{m}} = \mathbf{R}_{bw} \mathbf{m}_w + \mathbf{b}_{\text{hard}} + \mathbf{S}_{\text{soft}} \mathbf{R}_{bw} \mathbf{m}_w + \mathbf{n}_m$$

where:
- $\mathbf{b}_{\text{hard}}$: **hard-iron bias**, a constant offset created by permanent magnets or magnetized components mounted near the sensor
- $\mathbf{S}_{\text{soft}}$: **soft-iron distortion**, a direction-dependent warping caused by nearby permeable ferromagnetic materials
- $\mathbf{n}_m$: broadband measurement noise

**Yaw extraction.** Given roll $\phi$ and pitch $\theta$ (typically from the accelerometer) alongside hard-iron-corrected readings $m_x, m_y, m_z$, tilt-compensated magnetic yaw follows:

$$\psi = \arctan2(m_z \sin\phi - m_y \cos\phi, \, m_x \cos\theta + m_y \sin\theta \sin\phi + m_z \sin\theta \cos\phi)$$

This expression assumes an NED world frame, an x-forward / y-right / z-down body frame, and Z-Y-X Euler angles, measuring $\psi$ clockwise from magnetic north.

**Limitations and cautions:**
- **Severe environmental distortion**: steel rebar in concrete buildings, vehicle chassis, and indoor infrastructure easily skew magnetic headings by tens of degrees.
- **Electromagnetic interference**: high-current power lines, ESCs, and motors introduce dynamic magnetic fields correlated with throttle.
- **Auxiliary weighting**: consequently, fusion filters treat magnetometers cautiously—accepting heading updates only when field-norm consistency checks pass, or down-weighting them adaptively.

### 2.6.4 UWB (Ultra-Wideband)

Ultra-Wideband (UWB) radios emit nanosecond-scale pulses across wide bandwidths (500 MHz or more) to measure node-to-node **range** with decimeter precision, making UWB a popular indoor alternative where GNSS signals fail.

**Observation model (Two-Way Ranging, TWR).** Measuring round-trip pulse time $t_{\text{round}}$ and subtracting the responder's turnaround delay $t_{\text{reply}}$ avoids requiring tight clock synchronization between nodes:

$$d = \frac{c \cdot (t_{\text{round}} - t_{\text{reply}})}{2} + n_d$$

where $n_d$ is range noise, typically $\sigma_d \approx 5\text{–}30\,\text{cm}$ under clear line-of-sight (LOS) conditions.

**Non-Line-of-Sight (NLOS) bias.** When walls, furniture, or human bodies block the direct path, the UWB signal arrives via slower refracted or reflected paths, systematically inflating measured ranges. Robust UWB localization therefore relies on NLOS detection and outlier rejection.

**Role in sensor fusion.** Fixed UWB anchors localize a mobile tag within the anchor network's coordinate frame (which becomes a global position only if those anchors are surveyed into a global frame). Fused with VIO, UWB ranges bound long-term trajectory drift whenever anchor geometry is well conditioned and NLOS bias is controlled.

**Trilateration equations.** With $N_a$ anchors at known coordinates $\mathbf{a}_i$, each range measurement constrains the unknown tag position $\mathbf{p}$ to a sphere:

$$d_i = \|\mathbf{p} - \mathbf{a}_i\| + n_{d,i}, \quad i = 1, \ldots, N_a$$

Three anchors suffice to fix a 2D position, whereas 3D localization requires at least four non-coplanar anchors—if all anchors lie at the same elevation, the linearized system below loses observability along $z$.

The Python function below solves 3D trilateration by subtracting anchor 0's squared range from the remaining anchors to form a linear least-squares system.

```python
import numpy as np

def uwb_trilateration(anchor_positions, ranges):
    """UWB trilateration (least squares).
    
    Args:
        anchor_positions: (N, 3) anchor positions [m]
        ranges: (N,) measured range to each anchor [m]
    
    Returns:
        position: (3,) estimated position [m]
    """
    N = len(ranges)
    # Linearization: difference relative to the first anchor
    # ||p - a_i||^2 - ||p - a_0||^2 = d_i^2 - d_0^2
    # 2(a_0 - a_i)^T p = d_i^2 - d_0^2 - ||a_i||^2 + ||a_0||^2
    
    A = np.zeros((N - 1, 3))
    b = np.zeros(N - 1)
    
    a0 = anchor_positions[0]
    d0 = ranges[0]
    
    for i in range(1, N):
        ai = anchor_positions[i]
        di = ranges[i]
        A[i - 1] = 2 * (a0 - ai)
        b[i - 1] = di**2 - d0**2 - np.dot(ai, ai) + np.dot(a0, a0)
    
    # Least-squares solution
    position = np.linalg.lstsq(A, b, rcond=None)[0]
    return position
```

---

## Summary

Every sensor discussed in this chapter reduces raw physical phenomena—photons, laser echoes, specific forces, radio chirps, pressure gradients, or magnetic fields—to a mathematical observation model paired with characteristic failure modes and noise statistics. The table below brings these models and validation criteria together as a quick reference.

| Sensor | Observation | Observation model | Main noise sources | Validation basis |
|------|--------|----------|-------------|------------------|
| Camera | 2D image coordinates | $\pi(\mathbf{T} \cdot \mathbf{P})$ (pinhole + distortion) | Detection noise, distortion residual | Target detector and reprojection residuals on held-out images |
| LiDAR | 3D point $(r, \alpha, \omega)$ | Range-bearing | Range noise, beam divergence, mixed pixel | Device datasheet + residual tests by range, surface, and incidence angle |
| IMU (gyro) | Angular velocity $\boldsymbol{\omega}$ | $\tilde{\boldsymbol{\omega}} = \boldsymbol{\omega} + \mathbf{b}_g + \mathbf{n}_g$ | Bias, random walk | Device datasheet + Allan, temperature, and vibration tests |
| IMU (accel) | Specific force $\mathbf{a}$ | $\tilde{\mathbf{a}} = \mathbf{R}_{bw}(\mathbf{a}_w-\mathbf{g}_w) + \mathbf{b}_a + \mathbf{n}_a$ | Bias, random walk | Device datasheet + Allan, temperature, and vibration tests |
| GNSS | Pseudorange $\rho$ | $\rho = r + c\delta t_r + I + T + \epsilon$ | Multipath, ionosphere, troposphere | Target-environment logs by receiver and correction status |
| Radar | Range, Doppler, bearing | FMCW beat frequency | Clutter, multipath | Device range/velocity/angular resolution + tests by target, RCS, and weather |
| Wheel odom. | Rotation count | $\Delta s = r \Delta\theta$ | Slip | Repeated trajectories by surface, load, and slip condition |
| Barometer | Pressure $P$ | $h = f(P)$ | Weather change | Repeated static, temperature, and weather-condition measurements |
| Magnetometer | Magnetic field $\mathbf{m}$ | $\tilde{\mathbf{m}} = \mathbf{R}_{bw}\mathbf{m}_w + \mathbf{b} + \mathbf{n}$ | Hard/soft iron | Calibration residual + magnetic-interference test at the operating site |
| UWB | Range $d$ | $d = \|\mathbf{p} - \mathbf{a}\| + n$ | NLOS | Per-anchor LOS/NLOS range residuals |

Yet even a flawless observation model fails if its parameters are inaccurate—or if two sensors disagree on where they sit on the robot and what time their clocks report. Turning these individual equations into a consistent multi-sensor pipeline requires **calibration** to pin down each sensor's intrinsics, the rigid transforms between sensors, and any clock offsets—the subject of Chapter 3.
