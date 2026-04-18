# Ch.2 — Sensor Modeling

In Ch.1 we examined the taxonomy of sensor fusion and its design principles. We now turn in earnest to defining, in mathematical terms, how each sensor "sees" the world. Sensor observation models are substituted directly into the measurement function $h(\mathbf{x})$ of the Kalman filter and factor graph formulations covered in Ch.4, so the equations of this chapter form the foundation for every algorithm that follows.

> If robotics-practice Ch.2 introduced sensors at an overview level, this chapter focuses on **noise models and mathematical observation models**. To design a sensor fusion algorithm, we must know precisely not only "what a sensor measures" but also "what mathematical relationship the measurement bears to the underlying physical quantity" and "how the error is distributed."

---

## 2.1 Camera Observation Model

A camera is a sensor that projects points of the 3D world onto a 2D image plane. Modeling this projection mathematically is the crux of the camera observation model.

### 2.1.1 Pinhole Camera Model

The pinhole camera model is the most basic mathematical model of a camera. It assumes that a 3D point is projected onto the image plane along a straight line passing through the optical center.

For a 3D point $\mathbf{P}_c = [X_c, Y_c, Z_c]^\top$ in the camera coordinate frame, its projected point $\mathbf{p} = [u, v]^\top$ on the image plane is computed as:

$$\begin{bmatrix} u \\ v \\ 1 \end{bmatrix} = \frac{1}{Z_c} \mathbf{K} \begin{bmatrix} X_c \\ Y_c \\ Z_c \end{bmatrix}$$

Here $\mathbf{K}$ is the camera intrinsic matrix:

$$\mathbf{K} = \begin{bmatrix} f_x & 0 & c_x \\ 0 & f_y & c_y \\ 0 & 0 & 1 \end{bmatrix}$$

Meaning of each parameter:
- $f_x, f_y$: focal length. The physical focal length $f$ divided by the pixel size $(\Delta x, \Delta y)$: $f_x = f / \Delta x$, $f_y = f / \Delta y$. Typically $f_x \approx f_y$, but the two values can differ for non-square pixels.
- $c_x, c_y$: principal point. The pixel coordinates at which the optical axis intersects the image plane. Ideally the image center, but manufacturing tolerances can shift it by several pixels.

In homogeneous coordinates:

$$s \begin{bmatrix} u \\ v \\ 1 \end{bmatrix} = \mathbf{K} [\mathbf{R} | \mathbf{t}] \begin{bmatrix} X_w \\ Y_w \\ Z_w \\ 1 \end{bmatrix}$$

Here $[\mathbf{R} | \mathbf{t}]$ is the extrinsic parameters mapping the world frame to the camera frame, and $s = Z_c$ is the depth scale factor.

Denoting the projection function as $\pi(\cdot)$:

$$\mathbf{p} = \pi(\mathbf{P}_c) = \begin{bmatrix} f_x \frac{X_c}{Z_c} + c_x \\ f_y \frac{Y_c}{Z_c} + c_y \end{bmatrix}$$

The Jacobian of this nonlinear projection function plays a central role in state estimation:

$$\frac{\partial \pi}{\partial \mathbf{P}_c} = \begin{bmatrix} \frac{f_x}{Z_c} & 0 & -\frac{f_x X_c}{Z_c^2} \\ 0 & \frac{f_y}{Z_c} & -\frac{f_y Y_c}{Z_c^2} \end{bmatrix}$$

This $2 \times 3$ Jacobian is used directly to linearize the observation model in EKF-based VIO, and it is also the key building block of the residual Jacobian in nonlinear optimization.

### 2.1.2 Lens Distortion Model

Real camera lenses introduce distortion that deviates from the ideal projection of the pinhole model. Neglecting distortion can inflate the reprojection error from a few pixels to tens of pixels, so it must be corrected for any application demanding accurate sensor fusion.

#### Radial-Tangential Distortion (Brown-Conrady Model)

This is the most widely used distortion model and is OpenCV's default.

For normalized image coordinates $\mathbf{p}_n = [x_n, y_n]^\top = [X_c/Z_c, \, Y_c/Z_c]^\top$:

$$r^2 = x_n^2 + y_n^2$$

**Radial distortion:**

$$x_r = x_n (1 + k_1 r^2 + k_2 r^4 + k_3 r^6)$$
$$y_r = y_n (1 + k_1 r^2 + k_2 r^4 + k_3 r^6)$$

Radial distortion arises from lens curvature and grows with the distance from the image center. If $k_1 < 0$ the result is barrel distortion (magnification toward the center); if $k_1 > 0$ the result is pincushion distortion (magnification toward the edges).

**Tangential distortion:**

$$x_t = 2p_1 x_n y_n + p_2 (r^2 + 2x_n^2)$$
$$y_t = p_1 (r^2 + 2y_n^2) + 2p_2 x_n y_n$$

Tangential distortion appears when the lens is not perfectly parallel to the image sensor (decentering). It is much smaller in magnitude than radial distortion, but cannot be ignored in precision applications.

**Final distorted coordinates:**

$$\begin{bmatrix} x_d \\ y_d \end{bmatrix} = \begin{bmatrix} x_r + x_t \\ y_r + y_t \end{bmatrix}$$

$$\begin{bmatrix} u \\ v \end{bmatrix} = \begin{bmatrix} f_x x_d + c_x \\ f_y y_d + c_y \end{bmatrix}$$

The distortion parameters $[k_1, k_2, p_1, p_2, k_3]$ are estimated using the calibration method of [Zhang (2000)](https://doi.org/10.1109/34.888718); we cover this in detail in Ch.3.

#### Fisheye (Equidistant) Model

For fisheye lenses with a wide field of view (FoV of 180° or more), the radial-tangential model is inadequate: $r$ becomes very large near the image edges and the polynomial approximation diverges.

The generic camera model of [Kannala & Brandt (2006)](https://doi.org/10.1109/TPAMI.2006.153) is defined as follows.

Let the incidence angle $\theta$ denote the angle of the 3D point from the optical axis:

$$\theta = \arctan\left(\frac{\sqrt{X_c^2 + Y_c^2}}{Z_c}\right)$$

The distorted radius $r_d$ is modeled as an odd polynomial in $\theta$:

$$r_d = k_1 \theta + k_2 \theta^3 + k_3 \theta^5 + k_4 \theta^7 + k_5 \theta^9$$

In a pure equidistant projection, $r_d = f \cdot \theta$, corresponding to $k_1 = f$, $k_2 = k_3 = \cdots = 0$.

Projected coordinates:

$$\begin{bmatrix} u \\ v \end{bmatrix} = \begin{bmatrix} f_x \cdot r_d \cdot \frac{x_n}{\sqrt{x_n^2 + y_n^2}} + c_x \\ f_y \cdot r_d \cdot \frac{y_n}{\sqrt{x_n^2 + y_n^2}} + c_y \end{bmatrix}$$

Fisheye lenses offer a wide FoV that benefits perception of the surrounding environment, and they are supported by VIO systems such as [VINS-Mono](https://arxiv.org/abs/1708.03852) and Basalt. For calibration we use OCamCalib of [Scaramuzza et al. (2006)](https://rpg.ifi.uzh.ch/docs/IROS06_scaramuzza.pdf) or Kalibr.

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

The reprojection error is the core cost function of nearly every algorithm that uses camera observations in sensor fusion.

For a 3D landmark $\mathbf{P}_w$, the reprojection error is the difference between the predicted image coordinate $\hat{\mathbf{p}}$ obtained by projecting $\mathbf{P}_w$ under the camera pose $\mathbf{T}_{cw} = [\mathbf{R}|\mathbf{t}]$, and the actually observed feature point $\mathbf{p}_{\text{obs}}$:

$$\mathbf{e}_{\text{reproj}} = \mathbf{p}_{\text{obs}} - \pi(\mathbf{T}_{cw} \cdot \mathbf{P}_w)$$

Here $\pi(\cdot)$ is the projection function defined above (including distortion).

In **bundle adjustment**, we minimize the sum of reprojection errors over all camera poses and landmarks:

$$\min_{\{\mathbf{T}_i\}, \{\mathbf{P}_j\}} \sum_{i,j} \rho\left(\| \mathbf{p}_{ij} - \pi(\mathbf{T}_i \cdot \mathbf{P}_j) \|^2_{\mathbf{\Sigma}_{ij}}\right)$$

where:
- $\mathbf{p}_{ij}$: image coordinates of landmark $j$ observed by camera $i$
- $\mathbf{\Sigma}_{ij}$: observation noise covariance (typically $\sigma^2 \mathbf{I}_2$, $\sigma \approx 1$ pixel)
- $\rho(\cdot)$: robust kernel (Huber, Cauchy, etc.) — suppresses the influence of outliers

The distribution of the reprojection error is typically modeled as Gaussian with $\sigma = 0.5$ to 2 pixels. This value depends on the precision of the feature detector; with sub-pixel corner detection it can drop to $\sigma \approx 0.5$ pixel. More recently, foundation models such as [Depth Anything V2 (Yang et al., 2024)](https://arxiv.org/abs/2406.09414) and [Metric3D v2 (Hu et al., 2024)](https://arxiv.org/abs/2404.15506) estimate dense depth from a single image, and are being leveraged to extend the camera observation model from 2D reprojection errors to 3D depth observations.

### 2.1.4 Rolling Shutter Model

Most low-cost cameras (smartphones, webcams) use CMOS image sensors and acquire images with a **rolling shutter**. Unlike a global shutter, which exposes all pixels simultaneously, a rolling shutter exposes rows sequentially. As a result, the top and bottom of the image are captured at different times.

The exposure time of row $k$ is:

$$t_k = t_0 + k \cdot t_r$$

where $t_0$ is the exposure time of the first row and $t_r$ is the row readout time. The total readout time across the whole image is $H \cdot t_r$ ($H$: image height), which ranges from a few to tens of milliseconds.

When a rolling-shutter image is captured while the camera moves, the following artifacts arise:
- **Geometric distortion**: vertical lines tilt, and moving objects deform as if made of jelly.
- **Feature position error**: each feature point is captured under a different camera pose, so a projection model that assumes a global shutter becomes inaccurate.

In a rolling-shutter-aware projection model, the camera pose at the time corresponding to each feature's row coordinate $v$ must be used:

$$\mathbf{p}_i = \pi\left(\mathbf{T}(t_{v_i}) \cdot \mathbf{P}_i\right)$$

where $\mathbf{T}(t_{v_i})$ is the camera pose at the time corresponding to row $v_i$ of the $i$-th feature. This pose is typically obtained by interpolation using IMU measurements:

$$\mathbf{T}(t_{v_i}) = \mathbf{T}(t_0) \cdot \text{Exp}\left(\frac{v_i}{H} \cdot \text{Log}(\mathbf{T}(t_0)^{-1} \mathbf{T}(t_0 + H \cdot t_r))\right)$$

Here $\text{Exp}$ and $\text{Log}$ are the exponential and logarithmic maps on the $SE(3)$ Lie group.

Rolling-shutter correction is optionally supported in VIO systems such as [VINS-Mono](https://arxiv.org/abs/1708.03852) and [ORB-SLAM3](https://arxiv.org/abs/2007.11898), and it is especially important for combinations of high-speed motion and low-cost sensors, such as smartphone or drone-mounted cameras.

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
    alpha = v_row / H  # interpolation ratio in [0, 1]
    
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

LiDAR is an active sensor that computes range by emitting laser pulses and measuring the time-of-flight or phase shift of the light reflected from objects. This section treats LiDAR's mathematical observation model and its error characteristics.

### 2.2.1 Range-Bearing Model

The basic LiDAR observation is a pair of **range** and **bearing** for each laser beam. In 3D LiDAR, each point observation is expressed in spherical coordinates $(r, \alpha, \omega)$:

- $r$: range
- $\alpha$: azimuth
- $\omega$: elevation

The observation model for a 3D point $\mathbf{P}_L = [x, y, z]^\top$ in the LiDAR frame is:

$$\begin{aligned}
r &= \sqrt{x^2 + y^2 + z^2} + n_r \\
\alpha &= \arctan2(y, x) + n_\alpha \\
\omega &= \arcsin\left(\frac{z}{\sqrt{x^2 + y^2 + z^2}}\right) + n_\omega
\end{aligned}$$

where $n_r, n_\alpha, n_\omega$ are the observation noise on range, azimuth, and elevation, respectively.

**Noise characteristics:**
- **Range noise** $n_r$: typically $\sigma_r \approx 1\text{–}3\,\text{cm}$ for mechanical LiDAR and $\sigma_r \approx 2\text{–}5\,\text{cm}$ for solid-state LiDAR. Noise grows with range because of beam divergence and decreasing received energy.
- **Angular noise** $n_\alpha, n_\omega$: determined by encoder precision. Typically $\sigma_\alpha, \sigma_\omega \approx 0.01°\text{–}0.1°$. These small values are amplified into position errors at long range — at 50 m, an angular error of $0.1°$ corresponds to roughly $8.7\,\text{cm}$ of lateral error.

Conversion from spherical to Cartesian coordinates:

$$\mathbf{P}_L = \begin{bmatrix} r \cos\omega \cos\alpha \\ r \cos\omega \sin\alpha \\ r \sin\omega \end{bmatrix}$$

**Beam divergence.** A LiDAR laser beam is not a perfect line but spreads as a narrow cone. The divergence angle is typically $0.1°\text{–}0.5°$; at long range this enlarges the beam footprint so that the measurement reports the average range over a region rather than a single reflection. When a beam partially straddles an object boundary and both the object and background contribute, the measured range falls between the two and a spurious point is produced — this is known as the **mixed pixel** effect.

### 2.2.2 Motion Distortion

A mechanical spinning LiDAR fires lasers while the sensor rotates through 360°. The Velodyne VLP-16, for example, takes roughly 100 ms per revolution. If the platform (vehicle, drone) moves during these 100 ms, the points within a single scan are measured in different coordinate frames. This is the **motion distortion** or **ego-motion compensation** problem.

This problem is essentially the same as rolling shutter for cameras. If the $i$-th point of a scan is measured at time $t_i$, we must transform it into the coordinate frame of the reference time $t_0$:

$$\mathbf{P}_L^{(t_0)} = \mathbf{T}(t_0)^{-1} \cdot \mathbf{T}(t_i) \cdot \mathbf{P}_L^{(t_i)}$$

where $\mathbf{T}(t_i)$ is the LiDAR pose at time $t_i$.

**Correction methods:**

1. **IMU-based interpolation**: interpolate the pose change over the scan duration using high-frequency IMU measurements. This is the most common approach and is used in LIO-SAM, FAST-LIO2, and others.
2. **Previous-frame odometry based**: apply a constant-velocity model with the velocity estimated in the immediately preceding frame.
3. **Continuous-time methods**: model the trajectory as a continuous function with, e.g., B-splines, and evaluate the pose at each point's time. [CT-ICP (Dellenbach et al., 2022)](https://arxiv.org/abs/2109.12979) is a representative example.

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
        R_i = slerp(timestamps[i]).as_matrix()
        t_i = (1 - alpha) * T_start[:3, 3] + alpha * T_end[:3, 3]
        
        # Frame at that time -> frame at start time
        p_world = R_i @ points[i] + t_i
        corrected[i] = T_start[:3, :3].T @ (p_world - T_start[:3, 3])
    
    return corrected
```

### 2.2.3 Impact of Spinning vs Solid-State LiDAR on Fusion

**Mechanical spinning LiDAR** (Velodyne, Ouster, Hesai) provides a 360° horizontal FoV, with each scan forming a complete annular point cloud. Algorithms in the LOAM family are designed around this property — extracting edge and planar features along horizontal scan lines and estimating 6-DoF pose from omnidirectional observations.

**Solid-state LiDAR** (Livox Mid-40/70, Avia, HAP, etc.) has no mechanical rotating part and uses a non-repetitive scan pattern within a restricted FoV (for example, roughly 70.4° circular for the Livox Mid-70). A characteristic feature is that coverage of the FoV grows gradually over time.

The impact of this difference on fusion algorithms:

| Property | Spinning LiDAR | Solid-State LiDAR |
|------|-------------|-------------------|
| FoV | 360° horizontal | Limited (40° to 120°) |
| Scan pattern | Repetitive (horizontal lines) | Non-repetitive (rosette, Lissajous, etc.) |
| Single-frame density | Uniform | Non-uniform (grows over time) |
| Feature extraction | Scan-line-based feasible | No scan-line structure |
| Suitable algorithms | LOAM, LeGO-LOAM | FAST-LIO/LIO2 (point-wise processing) |

FAST-LIO / [FAST-LIO2](https://arxiv.org/abs/2107.06829) are particularly strong on solid-state LiDAR because their iterated EKF structure does not depend on scan-line structure and instead **processes individual points sequentially**. In contrast, LOAM's edge/planar feature extraction presupposes scan-line structure and is hard to apply directly to solid-state sensors. More recently, [FAST-LIVO2 (Zheng et al., 2024)](https://arxiv.org/abs/2408.14035) extends this structure to sequentially fuse three sensors — LiDAR, inertial, and visual — within the same iterated EKF, using a direct method to process both LiDAR points and images without separate feature extraction.

---

## 2.3 IMU Model

An IMU (Inertial Measurement Unit) consists of a 3-axis accelerometer and a 3-axis gyroscope. In sensor fusion the IMU is the core sensor of virtually every system: it provides high-frequency observations (100 Hz to 1 kHz) that interpolate between camera or LiDAR frames, and it contributes to initialization and scale observability. This section treats IMU error models in detail.

### 2.3.1 Gyroscope Error Model

A gyroscope measures the 3-axis angular velocity $\boldsymbol{\omega}$. The actual measurement $\tilde{\boldsymbol{\omega}}$ is modeled as:

$$\tilde{\boldsymbol{\omega}} = \boldsymbol{\omega} + \mathbf{b}_g + \mathbf{n}_g$$

Meaning of each term:
- $\boldsymbol{\omega}$: true angular velocity (in the IMU body frame)
- $\mathbf{b}_g$: **bias** — a nearly constant offset that varies slowly in time
- $\mathbf{n}_g$: **measurement noise** — additive white Gaussian noise (AWGN)

**Bias dynamics.** The bias is not a constant but drifts slowly over time. We model it as a **random walk**:

$$\dot{\mathbf{b}}_g = \mathbf{n}_{bg}$$

where $\mathbf{n}_{bg} \sim \mathcal{N}(\mathbf{0}, \sigma_{bg}^2 \mathbf{I})$ is the driving noise for the bias. In discrete time:

$$\mathbf{b}_{g,k+1} = \mathbf{b}_{g,k} + \sigma_{bg} \sqrt{\Delta t} \cdot \mathbf{w}_k, \quad \mathbf{w}_k \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$$

Typical parameters for a MEMS-grade gyroscope:
- Measurement noise density: $\sigma_g \approx 0.004\,\text{rad/s}/\sqrt{\text{Hz}}$ (about $0.2\,°/\text{s}/\sqrt{\text{Hz}}$)
- In-run bias stability: $\sigma_{bg} \approx 10\text{–}100\,°/\text{hr}$
- Bias random walk: $\sigma_{bg} \approx 0.0002\,\text{rad/s}^2/\sqrt{\text{Hz}}$

### 2.3.2 Accelerometer Error Model

An accelerometer measures the 3-axis specific force $\mathbf{a}$. Specific force is the true acceleration minus gravity. The actual measurement $\tilde{\mathbf{a}}$ is:

$$\tilde{\mathbf{a}} = \mathbf{R}_{bw}(\mathbf{a}_w - \mathbf{g}_w) + \mathbf{b}_a + \mathbf{n}_a$$

Meaning of each term:
- $\mathbf{a}_w$: true acceleration in the world frame
- $\mathbf{g}_w = [0, 0, -g]^\top$: gravity vector ($g \approx 9.81\,\text{m/s}^2$)
- $\mathbf{R}_{bw}$: world-to-body rotation matrix
- $\mathbf{b}_a$: accelerometer bias
- $\mathbf{n}_a \sim \mathcal{N}(\mathbf{0}, \sigma_a^2 \mathbf{I})$: measurement noise

**Role of gravity.** The fact that the accelerometer "feels" gravity is of great importance in IMU-based fusion. Even at rest, the accelerometer measures $[0, 0, g]^\top$ (when z is up). From this gravity observation we can estimate roll and pitch. Yaw, however, is a rotation about the gravity vector and is therefore unobservable — this is why VIO/LIO systems require additional observations (e.g., motion of visual features) to estimate yaw during initialization.

**Bias dynamics.** As with the gyroscope, we model the bias as a random walk:

$$\dot{\mathbf{b}}_a = \mathbf{n}_{ba}, \quad \mathbf{n}_{ba} \sim \mathcal{N}(\mathbf{0}, \sigma_{ba}^2 \mathbf{I})$$

Typical parameters for a MEMS-grade accelerometer:
- Measurement noise density: $\sigma_a \approx 0.04\,\text{m/s}^2/\sqrt{\text{Hz}}$ (about $4\,\text{mg}/\sqrt{\text{Hz}}$)
- Bias stability: $\sigma_{ba} \approx 0.01\text{–}0.1\,\text{mg}$
- Bias random walk: $\sigma_{ba} \approx 0.001\,\text{m/s}^3/\sqrt{\text{Hz}}$

### 2.3.3 Allan Variance

Allan variance is the standard method for analyzing IMU noise characteristics. Data are collected at rest over a long period (several hours), and the variance is computed at various cluster times $\tau$.

Definition of the Allan variance $\sigma^2(\tau)$:

$$\sigma^2(\tau) = \frac{1}{2} \langle (\bar{y}_{k+1} - \bar{y}_k)^2 \rangle$$

where $\bar{y}_k$ is the mean output of the $k$-th interval of length $\tau$.

On a log-log plot, the slope of the Allan deviation $\sigma(\tau)$ identifies the noise type:

| Slope | Noise type | Physical meaning |
|--------|-----------|-----------|
| $-1/2$ | Angle/velocity random walk (ARW/VRW) | White noise $\mathbf{n}_g, \mathbf{n}_a$ |
| $0$ | Bias instability | Flicker noise; the minimum is the bias stability |
| $+1/2$ | Rate random walk (RRW) | Bias random walk $\mathbf{n}_{bg}, \mathbf{n}_{ba}$ |

**How to read a datasheet.** Extracting the key parameters needed for sensor fusion from an IMU datasheet:

1. **Angular random walk (ARW)**: units $°/\sqrt{\text{hr}}$ or $\text{rad/s}/\sqrt{\text{Hz}}$. Read the value at $\tau = 1\,\text{s}$ on the Allan deviation plot, or read from the slope-$-1/2$ region. This corresponds to $\sigma_g$.
2. **Velocity random walk (VRW)**: units $\text{m/s}/\sqrt{\text{hr}}$ or $\text{m/s}^2/\sqrt{\text{Hz}}$. The white-noise density of the accelerometer. This corresponds to $\sigma_a$.
3. **In-run bias stability**: the minimum of the Allan deviation plot. The theoretical lower bound on the bias estimate that the system can reach.
4. **Rate random walk**: the rate at which the bias changes over time. This corresponds to $\sigma_{bg}, \sigma_{ba}$.

```python
import numpy as np

def compute_allan_variance(data, dt, max_cluster_size=None):
    """Compute the Allan variance.
    
    Args:
        data: (N,) data for one IMU axis collected at rest
        dt: sampling period (seconds)
        max_cluster_size: maximum cluster size (defaults to N//2 when None)
    
    Returns:
        taus: array of cluster times
        adevs: array of Allan deviations
    """
    N = len(data)
    if max_cluster_size is None:
        max_cluster_size = N // 2
    
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

The equations that integrate IMU measurements into pose (position, velocity, attitude) are called the strapdown navigation equations. "Strapdown" refers to the sensor being rigidly fixed (strapped down) to the platform, so that coordinate transformations are carried out in software rather than by a mechanical gimbal.

Continuous-time dynamics of the state $[\mathbf{R}, \mathbf{v}, \mathbf{p}]$ in the world (or navigation) frame:

$$\dot{\mathbf{R}} = \mathbf{R} \cdot [\tilde{\boldsymbol{\omega}} - \mathbf{b}_g - \mathbf{n}_g]_\times$$

$$\dot{\mathbf{v}} = \mathbf{R} (\tilde{\mathbf{a}} - \mathbf{b}_a - \mathbf{n}_a) + \mathbf{g}$$

$$\dot{\mathbf{p}} = \mathbf{v}$$

where:
- $\mathbf{R} \in SO(3)$: body-to-world rotation matrix
- $\mathbf{v} \in \mathbb{R}^3$: velocity in the world frame
- $\mathbf{p} \in \mathbb{R}^3$: position in the world frame
- $[\cdot]_\times$: skew-symmetric matrix (expressing the vector cross product as a matrix multiplication)

$$[\boldsymbol{\omega}]_\times = \begin{bmatrix} 0 & -\omega_z & \omega_y \\ \omega_z & 0 & -\omega_x \\ -\omega_y & \omega_x & 0 \end{bmatrix}$$

**Discrete-time integration.** State propagation from time $t_k$ to $t_{k+1} = t_k + \Delta t$:

$$\mathbf{R}_{k+1} = \mathbf{R}_k \cdot \text{Exp}\left((\tilde{\boldsymbol{\omega}}_k - \mathbf{b}_{g,k}) \Delta t\right)$$

$$\mathbf{v}_{k+1} = \mathbf{v}_k + \mathbf{g} \Delta t + \mathbf{R}_k (\tilde{\mathbf{a}}_k - \mathbf{b}_{a,k}) \Delta t$$

$$\mathbf{p}_{k+1} = \mathbf{p}_k + \mathbf{v}_k \Delta t + \frac{1}{2} \mathbf{g} \Delta t^2 + \frac{1}{2} \mathbf{R}_k (\tilde{\mathbf{a}}_k - \mathbf{b}_{a,k}) \Delta t^2$$

Here $\text{Exp}(\boldsymbol{\phi})$ is the exponential map on $SO(3)$ that converts a rotation vector $\boldsymbol{\phi}$ into a rotation matrix. It can be computed via Rodrigues' formula:

$$\text{Exp}(\boldsymbol{\phi}) = \mathbf{I} + \frac{\sin\|\boldsymbol{\phi}\|}{\|\boldsymbol{\phi}\|} [\boldsymbol{\phi}]_\times + \frac{1 - \cos\|\boldsymbol{\phi}\|}{\|\boldsymbol{\phi}\|^2} [\boldsymbol{\phi}]_\times^2$$

**Midpoint integration.** The Euler integration above is first-order accurate. For second-order accuracy one can average two consecutive IMU measurements:

$$\bar{\boldsymbol{\omega}} = \frac{1}{2}(\tilde{\boldsymbol{\omega}}_k + \tilde{\boldsymbol{\omega}}_{k+1}) - \mathbf{b}_{g,k}$$

$$\mathbf{R}_{k+1} = \mathbf{R}_k \cdot \text{Exp}(\bar{\boldsymbol{\omega}} \Delta t)$$

$$\bar{\mathbf{a}} = \frac{1}{2}\left(\mathbf{R}_k(\tilde{\mathbf{a}}_k - \mathbf{b}_{a,k}) + \mathbf{R}_{k+1}(\tilde{\mathbf{a}}_{k+1} - \mathbf{b}_{a,k})\right)$$

$$\mathbf{v}_{k+1} = \mathbf{v}_k + (\bar{\mathbf{a}} + \mathbf{g}) \Delta t$$

$$\mathbf{p}_{k+1} = \mathbf{p}_k + \mathbf{v}_k \Delta t + \frac{1}{2}(\bar{\mathbf{a}} + \mathbf{g}) \Delta t^2$$

This midpoint integration is the default scheme used in VINS-Mono, FAST-LIO2, and similar systems. Higher-order fourth-order Runge-Kutta (RK4) integration is also possible, but midpoint integration is sufficient at typical IMU rates (200 to 400 Hz).

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

**Numerical meaning of drift.** Consider how quickly the strapdown integration above diverges when the bias is not corrected. For an accelerometer bias of $b_a = 0.01\,\text{m/s}^2$ (about $1\,\text{mg}$, typical for MEMS):

- Position error after 1 s: $\frac{1}{2} \times 0.01 \times 1^2 = 0.005\,\text{m}$ (5 mm)
- After 10 s: $\frac{1}{2} \times 0.01 \times 100 = 0.5\,\text{m}$
- After 60 s: $\frac{1}{2} \times 0.01 \times 3600 = 18\,\text{m}$

This is why standalone inertial navigation is infeasible, and why we must continually estimate and correct the bias through sensor fusion. In VIO/LIO systems the biases $\mathbf{b}_g, \mathbf{b}_a$ are **included as part of the state vector** and are updated continuously from the observations of other sensors. Research on deep-learning-based inertial odometry has also been active. [AirIO (Chen et al., 2025)](https://arxiv.org/abs/2501.15659) strengthens the observability of IMU features and reports an accuracy improvement of more than 50 % over prior learning-based inertial odometry in drone settings.

### 2.3.5 IMU Grade Classification

IMUs are broadly divided into three grades by performance. The grade chosen for a sensor fusion system determines the frequency and type of external-sensor aiding required.

| Grade | ARW | Bias stability (gyro) | Price | Standalone navigation time | Examples |
|------|-----|----------------------|-------|-------------|------|
| Navigation grade | $< 0.002°/\sqrt{\text{hr}}$ | $< 0.01°/\text{hr}$ | $>$ \$10k | hours | HG1700, LN-200 |
| Tactical grade | $0.05\text{–}0.5°/\sqrt{\text{hr}}$ | $0.1\text{–}10°/\text{hr}$ | \$1k–10k | minutes | STIM300, ADIS16490 |
| MEMS | $0.1\text{–}1°/\sqrt{\text{hr}}$ | $1\text{–}100°/\text{hr}$ | $<$ \$100 | seconds | BMI088, ICM-42688 |

Robotics and autonomous driving use MEMS-grade IMUs almost exclusively, which makes tight fusion with cameras, LiDAR, and other sensors essential.

---

## 2.4 GNSS Model

GNSS (Global Navigation Satellite System) is a general term for satellite navigation systems such as GPS, GLONASS, Galileo, and BeiDou. This section treats the mathematical models needed to use GNSS observations in sensor fusion.

### 2.4.1 Pseudorange Observation Model

A GNSS receiver measures a **pseudorange** $\rho$ from each satellite. The pseudorange is the actual geometric range to the satellite plus error terms such as the receiver clock offset:

$$\rho^s = r^s + c \cdot \delta t_r - c \cdot \delta t^s + I^s + T^s + \epsilon_\rho$$

Meaning of each term:
- $r^s = \|\mathbf{p}_r - \mathbf{p}^s\|$: geometric distance between the receiver position $\mathbf{p}_r$ and the satellite $s$ position $\mathbf{p}^s$
- $c \cdot \delta t_r$: receiver clock offset (unknown, estimated together with position)
- $c \cdot \delta t^s$: satellite clock offset (can be corrected from the navigation message)
- $I^s$: ionospheric delay — signal delay caused by free electrons in the ionosphere. Can be removed with a dual-frequency L1/L2 receiver.
- $T^s$: tropospheric delay — delay caused by water vapor and air in the troposphere. Corrected with the Saastamoinen model or similar.
- $\epsilon_\rho$: residual noise (thermal noise, multipath, etc.). Typically $\sigma_\rho \approx 1\text{–}5\,\text{m}$ for single-point positioning and $\sigma_\rho \approx 0.1\text{–}0.3\,\text{m}$ with dual frequency and corrections.

**Positioning principle.** With pseudoranges from four or more satellites we can solve for the four unknowns: the receiver position $(x, y, z)$ and the clock offset $\delta t_r$. The nonlinear equations are solved by least squares or a Kalman filter — this is the basis of standard Single Point Positioning (SPP).

**Multipath.** In urban environments, satellite signals reflected off buildings can arrive along paths other than the direct line of sight, adding several to tens of meters of error to the pseudorange. Multipath is hard to model and varies with the environment, so outlier handling is especially important when using GNSS in sensor fusion.

### 2.4.2 Carrier Phase Observation Model

Carrier phase observations are far more precise than pseudoranges (millimeter level). The L1 carrier (about 1575.42 MHz) has a wavelength of roughly 19 cm, and resolving the phase to only 1 % yields a range precision of roughly 2 mm.

$$\Phi^s = r^s + c \cdot \delta t_r - c \cdot \delta t^s + \lambda N^s - I^s + T^s + \epsilon_\Phi$$

where:
- $\lambda$: carrier wavelength
- $N^s$: **integer ambiguity** — the integer number of full wavelengths between receiver and satellite. An unknown integer whose accurate resolution is the crux of RTK/PPP.
- $\epsilon_\Phi \approx 1\text{–}5\,\text{mm}$: carrier-phase noise (about 1/100 of the pseudorange noise)

A notable point is that the ionospheric delay enters with the opposite sign from the pseudorange (group velocity vs phase velocity). This allows the ionospheric delay to be canceled with dual-frequency observations.

### 2.4.3 RTK (Real-Time Kinematic)

RTK is a technique that uses **differencing** observations against a nearby (within a few km) base station to eliminate common errors (satellite clock, ionosphere, troposphere) and resolve the carrier-phase integer ambiguities in real time, achieving **centimeter-level** positioning.

**Double differencing.** The double difference between rover and base for satellite $s$ and reference satellite $r$:

$$\nabla\Delta\Phi_{br}^{sr} = \nabla\Delta r_{br}^{sr} + \lambda \nabla\Delta N_{br}^{sr} + \epsilon$$

Double differencing removes the receiver and satellite clock offsets and nearly cancels ionospheric and tropospheric errors when the base station is close. The remaining unknowns are the geometric range (which depends on position) and the integer ambiguities, which are resolved with algorithms such as LAMBDA.

### 2.4.4 PPP (Precise Point Positioning)

PPP is a technique that achieves centimeter-level positioning with a single receiver, without a base station. Precise orbits and precise clock corrections received from an external service remove satellite-related errors, while ionospheric and tropospheric errors are included in the state vector and estimated.

**RTK vs PPP:**

| Property | RTK | PPP |
|------|-----|-----|
| Base station required | Yes (within a few km) | No |
| Convergence time | seconds to tens of seconds | tens of minutes (traditional), minutes (PPP-AR) |
| Precision (after convergence) | $\sim 2\,\text{cm}$ | $\sim 5\,\text{cm}$ |
| Coverage | Near the base station | Global |

**Using GNSS in sensor fusion.** Because GNSS provides absolute position, combining it with VIO/LIO can completely eliminate long-term drift. [LIO-SAM (Shan et al., 2020)](https://arxiv.org/abs/2007.00258) is a representative example that adds a GNSS factor directly to the factor graph. Key considerations when incorporating GNSS observations into fusion:

1. **Coordinate frame transformation**: GNSS is output in WGS84 (latitude, longitude, ellipsoidal height), while robotics systems use a local frame such as ENU (East-North-Up) or NED (North-East-Down). A transformation is required.
2. **Use of covariance**: The DOP (Dilution of Precision) values or position covariances output by the GNSS receiver should be used as the observation covariance in the fusion system.
3. **Outlier handling**: In multipath environments, GNSS positioning can have errors of tens of meters, so robust kernels or $\chi^2$ tests must be used to detect and reject anomalous observations.

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
        e = diff / r  # unit vector (receiver-to-satellite direction)
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
    north = (N0 * (1 - e2) + alt0) * dlat
    up = dalt
    
    return np.array([east, north, up])
```

---

## 2.5 Radar Model

Radar is an active sensor that uses radio waves. Its key advantage is reliable operation in **adverse weather (rain, fog, snow, dust)** where cameras and LiDAR degrade. It also has the unique property of measuring relative **velocity directly** via the Doppler effect. Radar is rapidly gaining importance in autonomous driving.

### 2.5.1 FMCW Radar Principles

Most radars used in automotive and robotic systems are **FMCW (Frequency Modulated Continuous Wave)**. They transmit a continuously frequency-modulated wave (chirp) and extract range and velocity from the frequency difference (beat frequency) between the transmitted and reflected waves.

**Chirp signal.** The instantaneous frequency of the transmitted signal increases linearly with time:

$$f_{\text{TX}}(t) = f_0 + \frac{B}{T_c} t$$

where $f_0$ is the starting frequency, $B$ is the frequency bandwidth, and $T_c$ is the chirp duration.

**Beat frequency.** A reflected wave from an object at range $R$ has a time delay $\tau = 2R/c$ ($c$: speed of light). Mixing the transmitted and received waves yields a beat frequency:

$$f_b = \frac{2BR}{cT_c}$$

So the range is:

$$R = \frac{f_b \cdot c \cdot T_c}{2B}$$

**Range resolution.** The minimum range difference at which two objects can be distinguished is determined by the bandwidth:

$$\Delta R = \frac{c}{2B}$$

For example, with a 77 GHz radar and $B = 4\,\text{GHz}$, $\Delta R \approx 3.75\,\text{cm}$.

**Doppler measurement.** By transmitting successive chirps and observing the phase change across reflections from the same object, the radial (line-of-sight) velocity can be measured:

$$v_r = \frac{\lambda \cdot f_d}{2}$$

where $f_d$ is the Doppler frequency and $\lambda$ is the carrier wavelength. For a 77 GHz radar, $\lambda \approx 3.9\,\text{mm}$, so even very small velocity changes can be detected.

**Range-Doppler map.** A 2D FFT over multiple chirps produces a 2D range-velocity map (Range-Doppler map). Each peak in this map corresponds to a reflector, and its position gives the range and radial velocity simultaneously.

**Azimuth measurement.** The angle of arrival is estimated from the phase differences across an antenna array. A horizontal array provides azimuth and a vertical array provides elevation.

### 2.5.2 Recent Trends in 4D Imaging Radar

Traditional automotive radar provides three-dimensional information — range, velocity, and azimuth — with very poor vertical resolution. **4D imaging radar** is a next-generation radar that provides four dimensions — range, Doppler, azimuth, and **elevation** — at high resolution.

The key to 4D imaging radar is implementing a large virtual antenna array with MIMO (Multiple Input Multiple Output) technology. For example, 12 transmit antennas × 16 receive antennas = 192 virtual antennas, which achieves sufficient angular resolution both horizontally and vertically.

**4D Radar vs LiDAR:**

| Property | 4D Imaging Radar | LiDAR |
|------|-----------------|-------|
| Adverse-weather operation | Excellent | Weak |
| Point density | Medium (thousands of points/frame) | High (hundreds of thousands of points/frame) |
| Velocity measurement | Direct (Doppler) | Not available (requires two-frame differencing) |
| Angular resolution | $\sim 1°$ | $\sim 0.1°$ |
| Cost | Low to medium | Medium to high |
| Static-object detection | Limited (Doppler = 0) | Excellent |

**Using radar in sensor fusion.** Radar's Doppler measurement provides unique value in sensor fusion:

1. **Ego-velocity estimation**: By fitting the radial velocities measured from static reflectors, we can estimate the sensor's own 3D velocity. This is more direct than integrating accelerometer readings and has no drift.

$$v_r^{(i)} = -\mathbf{e}^{(i)\top} \mathbf{v}_{\text{ego}}$$

where $\mathbf{e}^{(i)}$ is the unit vector toward the $i$-th reflector and $\mathbf{v}_{\text{ego}}$ is the ego-velocity. Observations from multiple reflectors allow $\mathbf{v}_{\text{ego}}$ to be estimated by least squares.

2. **Dynamic-object detection**: Moving objects are identified from the discrepancy between the predicted Doppler for the static environment and the actual observation.
3. **Adverse-weather backup**: When rain or fog causes cameras and LiDAR to fail, radar functions as the only exteroceptive sensor.

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

Radar's direct velocity measurement is information that other sensors cannot easily provide, and this is why radar is increasingly emerging as a core sensor in the multi-sensor fusion architectures covered in Ch.8.

---

## 2.6 Other Sensors

Beyond cameras, LiDAR, IMU, GNSS, and radar, sensor fusion systems make use of a variety of auxiliary sensors. These are not sufficient for navigation on their own, but they are useful for compensating for the limitations of the main sensors.

### 2.6.1 Wheel Odometry

A wheel encoder is the most basic proprioceptive sensor: it estimates travel distance by measuring wheel rotation.

**Observation model.** For a differential-drive robot, from the rotation angles of the left and right wheels $\Delta \theta_L, \Delta \theta_R$:

$$\Delta s = \frac{r(\Delta \theta_L + \Delta \theta_R)}{2}, \quad \Delta \psi = \frac{r(\Delta \theta_R - \Delta \theta_L)}{d}$$

where $r$ is the wheel radius, $d$ is the distance between the left and right wheels (tread), $\Delta s$ is the forward distance, and $\Delta \psi$ is the change in yaw angle.

**Ackermann steering model** (cars):

$$\dot{x} = v \cos\psi, \quad \dot{y} = v \sin\psi, \quad \dot{\psi} = \frac{v \tan\delta}{L}$$

where $v$ is the rear-wheel velocity, $\delta$ is the steering angle, and $L$ is the wheelbase.

**Slip model.** In reality the wheel slips. Slip is especially large on wet roads, off-road surfaces, and during rapid acceleration or braking. The slip ratio is:

$$s = \frac{v_{\text{wheel}} - v_{\text{actual}}}{v_{\text{actual}}}$$

Under heavy slip, the reliability of wheel odometry drops sharply. In sensor fusion we handle this with an **adaptive observation covariance** — when slip is detected, the uncertainty of wheel odometry is enlarged so that other sensors dominate.

**Role in sensor fusion.** Wheel odometry is used in VIO/LIO systems as an additional velocity/position observation. Its short-term accuracy is particularly high on straight-line motion, so it compensates for IMU drift in environments with few visual features (tunnels, long straight roads). There have been works that extend VINS-Mono by adding a wheel-odometry factor.

### 2.6.2 Barometer

A barometer measures atmospheric pressure to estimate **altitude**.

**Observation model.** The pressure-altitude relation under the International Standard Atmosphere (ISA):

$$h = \frac{T_0}{L}\left(1 - \left(\frac{P}{P_0}\right)^{\frac{RL}{g_0}}\right)$$

where:
- $P$: measured pressure [Pa]
- $P_0 = 101325\,\text{Pa}$: standard sea-level pressure
- $T_0 = 288.15\,\text{K}$: standard sea-level temperature
- $L = 0.0065\,\text{K/m}$: lapse rate
- $R = 287.053\,\text{J/(kg·K)}$: specific gas constant for air
- $g_0 = 9.80665\,\text{m/s}^2$: standard gravitational acceleration

Simplified approximation (low altitude):

$$\Delta h \approx -\frac{\Delta P}{\rho g} \approx -\frac{\Delta P}{12.0}\,[\text{m}], \quad (\Delta P\text{ in Pa})$$

Near sea level, a pressure change of about $8.5\,\text{Pa}$ corresponds to an altitude change of $1\,\text{m}$.

**Noise characteristics:**
- Short-term precision: $\pm 0.1\text{–}0.5\,\text{m}$ (excellent)
- Long-term precision: $\pm 1\text{–}10\,\text{m}$ (pressure drift from weather changes)

**Role in sensor fusion.** The barometer provides a vertical (altitude) observation. This suppresses the vertical drift of the IMU and is especially useful for holding vertical position during drone hover. However, long-term drift from weather changes means it is more appropriate to use relative altitude changes rather than absolute altitude. Indoors, one must be careful of pressure changes from air conditioning or opening and closing doors.

### 2.6.3 Magnetometer

A magnetometer measures the 3-axis magnetic field vector. From the Earth's magnetic field we can extract an absolute **yaw heading**.

**Observation model.** The magnetometer reading is:

$$\tilde{\mathbf{m}} = \mathbf{R}_{bw} \mathbf{m}_w + \mathbf{b}_{\text{hard}} + \mathbf{S}_{\text{soft}} \mathbf{R}_{bw} \mathbf{m}_w + \mathbf{n}_m$$

where:
- $\mathbf{m}_w$: Earth's magnetic field vector in the world frame (magnitude about $25\text{–}65\,\mu\text{T}$, varies with location)
- $\mathbf{b}_{\text{hard}}$: **hard-iron bias** — a constant magnetic field caused by nearby permanent magnets or metal
- $\mathbf{S}_{\text{soft}}$: **soft-iron distortion** — direction-dependent distortion of the magnetic field caused by surrounding ferromagnetic material
- $\mathbf{n}_m$: measurement noise

**Yaw extraction.** If roll and pitch are known (from the accelerometer), the yaw angle can be computed from the magnetometer reading:

$$\psi = \arctan2(m_y \cos\phi - m_z \sin\phi, \, m_x \cos\theta + m_y \sin\theta \sin\phi + m_z \sin\theta \cos\phi)$$

where $\phi$ is roll, $\theta$ is pitch, and $m_x, m_y, m_z$ are the hard-iron-corrected magnetometer readings.

**Limitations and cautions:**
- Magnetic distortion is very large indoors, inside vehicles, and near buildings. Near reinforced-concrete buildings, heading errors of tens of degrees can occur.
- Sensitive to electromagnetic interference from motors, wires, and other currents.
- For these reasons, the magnetometer plays only an auxiliary role in sensor fusion, and is used only when its reading is trusted or with adaptive weighting.

### 2.6.4 UWB (Ultra-Wideband)

UWB is a wireless technology that uses ultra-short pulses over a very wide bandwidth (500 MHz or more) to measure **range** between nodes with high precision.

**Observation model (TWR — Two-Way Ranging):**

$$d = \frac{c \cdot (t_{\text{round}} - t_{\text{reply}})}{2} + n_d$$

where:
- $t_{\text{round}}$: time from transmission of the request pulse to reception of the reply pulse
- $t_{\text{reply}}$: processing time at the responding node
- $n_d$: range noise, typically $\sigma_d \approx 5\text{–}30\,\text{cm}$ in LOS environments

**NLOS (Non-Line-of-Sight) problem.** UWB achieves high precision when a direct line of sight (LOS) is available, but when the signal passes through walls or obstacles (NLOS), it is delayed and reports a range larger than the true one. NLOS detection and mitigation is the central challenge of UWB-based positioning.

**Role in sensor fusion.** By pre-installing UWB anchors in the environment, one can estimate an absolute position by trilaterating the ranges to each anchor. This can substitute for GNSS indoors, and combined with VIO it corrects long-term drift.

**Observation equations (trilateration):**

$$d_i = \|\mathbf{p} - \mathbf{a}_i\| + n_{d,i}, \quad i = 1, \ldots, N_a$$

where $\mathbf{p}$ is the unknown tag position and $\mathbf{a}_i$ is the known position of the $i$-th anchor. Three or more anchors suffice to estimate a 2D position, and four or more for 3D.

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

## 2.7 Sensor Modeling Summary

We summarize the observation models and key characteristics of the sensors covered in this chapter.

| Sensor | Observation | Observation model | Main noise sources | Typical noise level |
|------|--------|----------|-------------|------------------|
| Camera | 2D image coordinates | $\pi(\mathbf{T} \cdot \mathbf{P})$ (pinhole + distortion) | Detection noise, distortion residual | $0.5\text{–}2$ pixels |
| LiDAR | 3D point $(r, \alpha, \omega)$ | Range-bearing | Range noise, beam divergence, mixed pixel | $1\text{–}5\,\text{cm}$ |
| IMU (gyro) | Angular velocity $\boldsymbol{\omega}$ | $\tilde{\boldsymbol{\omega}} = \boldsymbol{\omega} + \mathbf{b}_g + \mathbf{n}_g$ | Bias, random walk | ARW $\sim 0.1\text{–}1°/\sqrt{\text{hr}}$ |
| IMU (accel) | Specific force $\mathbf{a}$ | $\tilde{\mathbf{a}} = \mathbf{R}(\mathbf{a}-\mathbf{g}) + \mathbf{b}_a + \mathbf{n}_a$ | Bias, random walk | VRW $\sim 0.02\text{–}0.2\,\text{m/s}/\sqrt{\text{hr}}$ |
| GNSS | Pseudorange $\rho$ | $\rho = r + c\delta t_r + I + T + \epsilon$ | Multipath, ionosphere, troposphere | $1\text{–}5\,\text{m}$ (SPP) |
| Radar | Range, Doppler, bearing | FMCW beat frequency | Clutter, multipath | $\Delta R \sim 4\,\text{cm}$, $v \sim 0.1\,\text{m/s}$ |
| Wheel odom. | Rotation count | $\Delta s = r \Delta\theta$ | Slip | $1\text{–}5\%$ of travel distance |
| Barometer | Pressure $P$ | $h = f(P)$ | Weather change | $0.1\text{–}0.5\,\text{m}$ (short term) |
| Magnetometer | Magnetic field $\mathbf{m}$ | $\tilde{\mathbf{m}} = \mathbf{R}\mathbf{m}_w + \mathbf{b} + \mathbf{n}$ | Hard/soft iron | $1\text{–}5°$ (after calibration) |
| UWB | Range $d$ | $d = \|\mathbf{p} - \mathbf{a}\| + n$ | NLOS | $5\text{–}30\,\text{cm}$ (LOS) |

Accurately understanding each sensor's observation model is the first step of sensor fusion. In the next chapter we cover the prerequisite for actually using these sensors together — **calibration**, the process of precisely determining the geometric and temporal relationships between sensors. No matter how accurate the observation models, fusion performance degrades substantially if the relative sensor positions and time synchronization are inaccurate.
