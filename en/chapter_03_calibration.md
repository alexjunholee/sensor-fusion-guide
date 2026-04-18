# Ch.3 — Calibration Deep Dive

Ch.2 defined the observation model of each sensor mathematically. However, applying these models to real sensor data requires one prerequisite — the parameters of the model must be known precisely. The camera's focal length, the relative position between LiDAR and IMU, the time offset between sensors — the process that determines these values precisely is calibration.

> **Key message**: The accuracy of sensor fusion cannot exceed the accuracy of calibration. This chapter covers every aspect of calibration, from camera intrinsics to multi-sensor extrinsics and time synchronization.

Calibration is the first problem that must be solved in a sensor fusion pipeline. No matter how sophisticated the state estimation algorithm, if the internal model of a sensor is inaccurate or if the relative position/orientation between sensors is wrong, the fusion result diverges. In LiDAR-camera fusion in particular, an error of even 1 degree in the extrinsic parameters produces roughly 87 cm of registration error for an object 50 m away. In this chapter, we derive the mathematical foundation of each calibration problem and provide code and tools that can be used directly in practice.

---

## 3.1 Camera Intrinsic Calibration

Camera intrinsic calibration is the process of estimating the parameters that determine where a point in the 3D world is projected onto the 2D image. These parameters include the focal length, the principal point, and the lens distortion coefficients.

### 3.1.1 Pinhole Camera Model (Review)

A 3D world coordinate $\mathbf{P}_w = [X, Y, Z]^\top$ is transformed into the camera frame as $\mathbf{P}_c = [X_c, Y_c, Z_c]^\top$ and then projected onto the image plane as follows:

$$
s \begin{bmatrix} u \\ v \\ 1 \end{bmatrix}
= \mathbf{K} \begin{bmatrix} X_c \\ Y_c \\ Z_c \end{bmatrix}
= \begin{bmatrix} f_x & \gamma & c_x \\ 0 & f_y & c_y \\ 0 & 0 & 1 \end{bmatrix} \begin{bmatrix} X_c \\ Y_c \\ Z_c \end{bmatrix}
$$

Here:
- $f_x, f_y$: focal length in pixel units. $f_x = f / p_x$, where $f$ is the physical focal length (mm) and $p_x$ is the pixel size (mm/pixel).
- $(c_x, c_y)$: principal point. The intersection of the optical axis with the image sensor center.
- $\gamma$: skew coefficient. Nearly 0 on modern cameras.
- $s = Z_c$: scale factor (depth).

$\mathbf{K}$ is called the **camera intrinsic matrix** or **calibration matrix**. This matrix has 5 degrees of freedom ($f_x, f_y, c_x, c_y, \gamma$). In practice, $\gamma$ is often set to 0, reducing it to 4.

### 3.1.2 Lens Distortion Model

Real lenses do not follow the ideal straight-line projection of the pinhole model. Distortion is broadly divided into radial distortion and tangential distortion.

First, define **normalized coordinates**:

$$
x = X_c / Z_c, \quad y = Y_c / Z_c, \quad r^2 = x^2 + y^2
$$

**Radial distortion**:
$$
x_{\text{radial}} = x(1 + k_1 r^2 + k_2 r^4 + k_3 r^6)
$$
$$
y_{\text{radial}} = y(1 + k_1 r^2 + k_2 r^4 + k_3 r^6)
$$

Barrel distortion occurs when $k_1 < 0$ and pincushion distortion when $k_1 > 0$. Most lenses are adequately modeled using only $k_1$ and $k_2$.

**Tangential distortion**:
$$
x_{\text{tangential}} = 2p_1 xy + p_2(r^2 + 2x^2)
$$
$$
y_{\text{tangential}} = p_1(r^2 + 2y^2) + 2p_2 xy
$$

This arises when the lens elements are not perfectly parallel to the image sensor plane.

**Final distorted coordinates**:
$$
x_d = x_{\text{radial}} + x_{\text{tangential}}, \quad y_d = y_{\text{radial}} + y_{\text{tangential}}
$$

$$
u = f_x \cdot x_d + c_x, \quad v = f_y \cdot y_d + c_y
$$

The distortion parameter vector is expressed as $\mathbf{d} = [k_1, k_2, p_1, p_2, k_3]$, and OpenCV's `calibrateCamera()` follows this ordering.

### 3.1.3 Zhang's Method: Homography-Based Calibration

The method proposed by [Zhang (2000)](https://ieeexplore.ieee.org/document/888718) estimates camera parameters from images of a planar pattern (a checkerboard) captured in various poses. Since it requires no 3D calibration apparatus — only a printer-produced checkerboard — it is currently the most widely used calibration method.

**Key idea**: When the pattern lies on the $Z = 0$ plane, the 3D-to-2D projection simplifies to a homography.

#### Step 1: Homography Extraction

Since the checkerboard lies on the $Z = 0$ plane, the relation between world coordinate $\mathbf{M} = [X, Y, 0]^\top$ and image coordinate $\mathbf{m} = [u, v]^\top$ in homogeneous coordinates is:

$$
s \tilde{\mathbf{m}} = \mathbf{K} [\mathbf{r}_1 \quad \mathbf{r}_2 \quad \mathbf{r}_3 \quad \mathbf{t}] \begin{bmatrix} X \\ Y \\ 0 \\ 1 \end{bmatrix}
= \mathbf{K} [\mathbf{r}_1 \quad \mathbf{r}_2 \quad \mathbf{t}] \begin{bmatrix} X \\ Y \\ 1 \end{bmatrix}
$$

Since $Z = 0$, the $\mathbf{r}_3$ column vanishes. From this:

$$
s \tilde{\mathbf{m}} = \mathbf{H} \tilde{\mathbf{M}}, \quad \mathbf{H} = \mathbf{K} [\mathbf{r}_1 \quad \mathbf{r}_2 \quad \mathbf{t}]
$$

$\mathbf{H}$ is a $3 \times 3$ homography matrix. In each image, a minimum of 4 corresponding points allow $\mathbf{H}$ to be estimated via the DLT (Direct Linear Transform). In practice, since a checkerboard has dozens of corners, we solve the over-determined system using SVD and remove outliers with RANSAC.

**Homography estimation via DLT**: From the correspondence $(\mathbf{M}_j, \mathbf{m}_j)$, using $\tilde{\mathbf{m}}_j \times \mathbf{H} \tilde{\mathbf{M}}_j = \mathbf{0}$, we obtain the linear system:

$$
\begin{bmatrix}
\tilde{\mathbf{M}}_j^\top & \mathbf{0}^\top & -u_j \tilde{\mathbf{M}}_j^\top \\
\mathbf{0}^\top & \tilde{\mathbf{M}}_j^\top & -v_j \tilde{\mathbf{M}}_j^\top
\end{bmatrix}
\mathbf{h} = \mathbf{0}
$$

Here $\mathbf{h}$ is the vectorization of the 9 elements of $\mathbf{H}$. Given $n$ correspondences, we form the $2n \times 9$ matrix $\mathbf{A}$ and find $\mathbf{h}$ that minimizes $\|\mathbf{A}\mathbf{h}\|$ as the right singular vector corresponding to the smallest singular value in the SVD of $\mathbf{A}$.

#### Step 2: Deriving Constraints on the Intrinsic Parameters

Splitting $\mathbf{H} = [\mathbf{h}_1 \quad \mathbf{h}_2 \quad \mathbf{h}_3]$ by columns:

$$
[\mathbf{h}_1 \quad \mathbf{h}_2 \quad \mathbf{h}_3] = \lambda \mathbf{K} [\mathbf{r}_1 \quad \mathbf{r}_2 \quad \mathbf{t}]
$$

Here $\lambda$ is an arbitrary scale factor. From the orthogonality of the rotation matrix we obtain two constraints:

1. **Orthogonality condition**: $\mathbf{r}_1^\top \mathbf{r}_2 = 0$
   $$\mathbf{h}_1^\top \mathbf{K}^{-\top} \mathbf{K}^{-1} \mathbf{h}_2 = 0$$

2. **Equal-length condition**: $\|\mathbf{r}_1\| = \|\mathbf{r}_2\|$
   $$\mathbf{h}_1^\top \mathbf{K}^{-\top} \mathbf{K}^{-1} \mathbf{h}_1 = \mathbf{h}_2^\top \mathbf{K}^{-\top} \mathbf{K}^{-1} \mathbf{h}_2$$

Define $\mathbf{B} = \mathbf{K}^{-\top} \mathbf{K}^{-1}$. Since $\mathbf{B}$ is a symmetric positive-definite matrix, it has 6 independent elements:

$$
\mathbf{B} = \begin{bmatrix} B_{11} & B_{12} & B_{13} \\ B_{12} & B_{22} & B_{23} \\ B_{13} & B_{23} & B_{33} \end{bmatrix}
$$

Written as a vector, $\mathbf{b} = [B_{11}, B_{12}, B_{22}, B_{13}, B_{23}, B_{33}]^\top$.

#### Step 3: Constructing the Linear System

$\mathbf{h}_i^\top \mathbf{B} \mathbf{h}_j$ can be expressed as an inner product with respect to $\mathbf{b}$:

$$
\mathbf{h}_i^\top \mathbf{B} \mathbf{h}_j = \mathbf{v}_{ij}^\top \mathbf{b}
$$

Here:
$$
\mathbf{v}_{ij} = \begin{bmatrix} h_{1i}h_{1j} \\ h_{1i}h_{2j} + h_{2i}h_{1j} \\ h_{2i}h_{2j} \\ h_{3i}h_{1j} + h_{1i}h_{3j} \\ h_{3i}h_{2j} + h_{2i}h_{3j} \\ h_{3i}h_{3j} \end{bmatrix}
$$

From the two constraints on each image:

$$
\begin{bmatrix} \mathbf{v}_{12}^\top \\ (\mathbf{v}_{11} - \mathbf{v}_{22})^\top \end{bmatrix} \mathbf{b} = \mathbf{0}
$$

Given $n$ images, we obtain a $2n \times 6$ system.

**Minimum number of images**: Setting $\gamma = 0$ (adding the constraint $B_{12} = 0$) leaves 5 unknowns, so a minimum of 3 images suffices. The general 5-parameter model also requires a minimum of 3 images. In practice, 15-25 images are captured.

#### Step 4: Recovering K

Once $\mathbf{b}$ is solved, $\mathbf{B} = \mathbf{K}^{-\top} \mathbf{K}^{-1}$ can be recovered, and the Cholesky decomposition $\mathbf{B} = \mathbf{L}\mathbf{L}^\top$ yields $\mathbf{K}^{-1} = \mathbf{L}^\top$, from which $\mathbf{K}$ is obtained. Concretely:

$$
v_0 = (B_{12}B_{13} - B_{11}B_{23}) / (B_{11}B_{22} - B_{12}^2)
$$
$$
\lambda = B_{33} - [B_{13}^2 + v_0(B_{12}B_{13} - B_{11}B_{23})] / B_{11}
$$
$$
f_x = \sqrt{\lambda / B_{11}}
$$
$$
f_y = \sqrt{\lambda B_{11} / (B_{11}B_{22} - B_{12}^2)}
$$
$$
\gamma = -B_{12} f_x^2 f_y / \lambda
$$
$$
c_x = \gamma v_0 / f_y - B_{13} f_x^2 / \lambda
$$
$$
c_y = v_0
$$

#### Step 5: Computing Extrinsic Parameters

Once $\mathbf{K}$ is known, for each image $i$:

$$
\mathbf{r}_1 = \lambda \mathbf{K}^{-1} \mathbf{h}_1, \quad \mathbf{r}_2 = \lambda \mathbf{K}^{-1} \mathbf{h}_2, \quad \mathbf{t} = \lambda \mathbf{K}^{-1} \mathbf{h}_3
$$

where $\lambda = 1 / \|\mathbf{K}^{-1} \mathbf{h}_1\|$. We obtain $\mathbf{r}_3 = \mathbf{r}_1 \times \mathbf{r}_2$. Because the estimated $[\mathbf{r}_1, \mathbf{r}_2, \mathbf{r}_3]$ is generally not an exact rotation matrix (due to noise), it is projected onto the nearest rotation matrix using SVD: $\mathbf{R} = \mathbf{U}\mathbf{V}^\top$ (with $\det(\mathbf{R}) = 1$).

#### Step 6: Nonlinear Optimization (MLE)

The linear solution provides a good initial value but does not account for distortion parameters and is not optimal under noise. In the final stage, the Levenberg-Marquardt (LM) algorithm minimizes the **reprojection error**:

$$
\min_{\mathbf{K}, \mathbf{d}, \{\mathbf{R}_i, \mathbf{t}_i\}} \sum_{i=1}^{n}\sum_{j=1}^{m} \|\mathbf{m}_{ij} - \hat{\mathbf{m}}(\mathbf{K}, \mathbf{d}, \mathbf{R}_i, \mathbf{t}_i, \mathbf{M}_j)\|^2
$$

Here $n$ is the number of images, $m$ is the number of corners per image, and $\hat{\mathbf{m}}(\cdot)$ is the full projection function including distortion.

The Jacobian of this optimization consists of the partial derivatives of the projection function with respect to each parameter. OpenCV's `calibrateCamera()` implements this entire pipeline (DLT → linear K estimation → LM optimization).

### 3.1.4 Calibration Code with OpenCV

```python
import numpy as np
import cv2
import glob

# Checkerboard settings
CHECKERBOARD = (9, 6)  # Number of interior corners (columns, rows)
SQUARE_SIZE = 0.025    # 25mm square

# Generate 3D world coordinates (Z=0 plane)
objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
objp *= SQUARE_SIZE

obj_points = []  # 3D points (same in all images)
img_points = []  # 2D points (different per image)

images = sorted(glob.glob("calibration_images/*.jpg"))

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Corner detection
    ret, corners = cv2.findChessboardCorners(
        gray, CHECKERBOARD,
        cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE
    )
    
    if ret:
        # Refine corner locations to sub-pixel precision
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        
        obj_points.append(objp)
        img_points.append(corners_refined)

# Perform calibration (Zhang's method)
ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(
    obj_points, img_points, gray.shape[::-1], None, None
)

print(f"RMS reprojection error: {ret:.4f} pixels")
print(f"Camera matrix K:\n{K}")
print(f"Distortion coefficients: {dist.ravel()}")

# Reprojection error analysis
errors = []
for i in range(len(obj_points)):
    img_points_proj, _ = cv2.projectPoints(obj_points[i], rvecs[i], tvecs[i], K, dist)
    error = cv2.norm(img_points[i], img_points_proj, cv2.NORM_L2) / len(img_points_proj)
    errors.append(error)
    
print(f"Per-image mean error: {np.mean(errors):.4f} pixels")
print(f"Max error image: {np.argmax(errors)} ({max(errors):.4f} px)")
```

### 3.1.5 Practical Tips: Conditions for Good Calibration

The quality of calibration is determined during the data collection process. Below are key conditions that have been repeatedly confirmed in practice.

**Pose Diversity**: the most important factor. The checkerboard should be captured from diverse angles and positions. Concretely:
- Place the checkerboard in every region of the image — top, bottom, left, right, and center (essential for principal-point estimation)
- Tilt the board by more than 45 degrees when photographing (improves the accuracy of focal-length estimation)
- Place the board close to and far from the camera to obtain a variety of scales
- A minimum of 15-25 images, ideally more than 50

**Corner Accuracy**: always secure sub-pixel precision with `cv2.cornerSubPix()`. Exclude images where corner detection fails.

**Illumination Conditions**: uniform lighting is ideal, but slight shadows do not affect corner detection. Glare interferes with corner detection, so use a checkerboard printed on matte paper.

**Reprojection Error Criteria**:
- $< 0.3$ pixels: excellent
- $0.3 - 0.5$ pixels: good
- $0.5 - 1.0$ pixels: consider recollection
- $> 1.0$ pixels: problematic (insufficient pose diversity, corner-detection errors, etc.)

**Detecting Outlier Images**: compute the per-image reprojection error and remove any image with error 2-3 times larger than the mean, then recalibrate. The `errors` array above shows this.

**Warning — accuracy of the square size**: `SQUARE_SIZE` must exactly match the actual printed checkerboard square size. Printer scaling can cause the physical size to differ from the specified size, so always measure with a ruler. If this value is wrong, `tvecs` (the translation vector) will be incorrect, but `K` and the distortion coefficients are unaffected (since the distortion model is defined in normalized coordinates).

### 3.1.6 Fisheye / Omnidirectional Calibration

For fisheye lenses with a FoV exceeding 180 degrees, the standard radial-tangential distortion model fails. The distortion is so severe that the polynomial approximation does not converge.

**Equidistant Projection Model**:

The general camera model proposed by [Kannala & Brandt (2006)](https://ieeexplore.ieee.org/document/1642666) expresses the relation between the incidence angle $\theta$ and the image radius $r$ as a polynomial:

$$
r(\theta) = k_1 \theta + k_2 \theta^3 + k_3 \theta^5 + k_4 \theta^7 + k_5 \theta^9
$$

Here $\theta = \arctan\left(\sqrt{x^2 + y^2}\right)$ is the incidence angle and $x, y$ are normalized coordinates. The ideal equidistant projection is $r = f\theta$, and the higher-order polynomial terms correct for the actual lens deviation.

OpenCV implements a variant of this model in the `cv2.fisheye` module:

$$
\theta_d = \theta(1 + k_1\theta^2 + k_2\theta^4 + k_3\theta^6 + k_4\theta^8)
$$

```python
# Fisheye calibration (OpenCV fisheye module)
import cv2
import numpy as np

calibration_flags = (
    cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC
    + cv2.fisheye.CALIB_CHECK_COND
    + cv2.fisheye.CALIB_FIX_SKEW
)

K_fisheye = np.zeros((3, 3))
D_fisheye = np.zeros((4, 1))  # k1, k2, k3, k4

ret, K_fisheye, D_fisheye, rvecs, tvecs = cv2.fisheye.calibrate(
    obj_points,
    img_points,
    gray.shape[::-1],
    K_fisheye,
    D_fisheye,
    None, None,
    calibration_flags,
    (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6)
)

print(f"Fisheye RMS error: {ret:.4f}")
print(f"K:\n{K_fisheye}")
print(f"D: {D_fisheye.ravel()}")
```

**Scaramuzza OCamCalib (Omnidirectional Camera Calibration)**:

The OCamCalib of [Scaramuzza et al. (2006)](https://sites.google.com/site/scarabotix/ocamcalib-omnidirectional-camera-calibration-toolbox-for-matlab) is a unified calibration tool for catadioptric systems (mirror + lens) and ultra-wide-angle fisheye lenses. It models the projection function directly as a polynomial, independent of sensor type:

$$
\begin{bmatrix} u \\ v \end{bmatrix} = \begin{bmatrix} x_c \\ y_c \end{bmatrix} + \mathbf{A} \cdot \rho(\theta) \begin{bmatrix} \cos(\phi) \\ \sin(\phi) \end{bmatrix}
$$

Here $\mathbf{A}$ is an affine transformation matrix (for stretch and non-square pixel correction) and $\rho(\theta)$ is the image radius as a function of incidence angle.

### 3.1.7 Camera Calibration with Kalibr

[Kalibr](https://github.com/ethz-asl/kalibr), developed at ETH Zurich, is the de facto standard for camera-IMU calibration (Section 3.4). For camera intrinsic calibration as well, it often yields more refined results than OpenCV.

**Camera models supported by Kalibr**:

| Model | # Parameters | Suitable Lens |
|------|-----------|-----------|
| `pinhole-radtan` | 4 + 4 | Standard lens (same as OpenCV) |
| `pinhole-equi` | 4 + 4 | Fisheye lens |
| `omni-radtan` | 5 + 4 | Ultra-wide / catadioptric |
| `ds` (Double Sphere) | 6 | Wide-angle lens (modern model) |
| `eucm` (Extended UCM) | 6 | Wide-angle lens |

**AprilGrid vs Checkerboard**: Kalibr recommends the AprilGrid target. Each tag in an AprilGrid has a unique ID encoded, so corner detection works even under partial occlusion, and corner ordering is identified automatically.

**Kalibr run example** (camera intrinsic only):
```bash
# Generate the AprilGrid target file
kalibr_create_target_pdf --type apriltag \
    --nx 6 --ny 6 --tsize 0.024 --tspace 0.3

# Camera calibration
kalibr_calibrate_cameras \
    --target april_6x6_24x24mm.yaml \
    --bag camera_calibration.bag \
    --models pinhole-radtan \
    --topics /cam0/image_raw
```

Kalibr's internal operation proceeds as follows:
1. Detect AprilGrid corners in each frame
2. Estimate homographies from corner correspondences (similar to Step 1 of Zhang's method)
3. Perform batch optimization over the full parameter set of the camera model
4. Visualize the optimization result and the distribution of residuals

Kalibr's advantages over OpenCV's default calibration are:
- **B-spline trajectory representation**: a continuous-time model that naturally handles motion blur effects
- **Variety of camera models**: supports modern models such as DS and EUCM
- **Multi-camera**: can simultaneously estimate the relative poses of several cameras
- **IMU integration**: connects naturally to the camera-IMU calibration covered in Section 3.4

---

## 3.2 Camera-Camera (Stereo) Extrinsic

In a stereo camera system, calibrating the relative pose (extrinsic) between the two cameras is a prerequisite for depth estimation.

### 3.2.1 Stereo Calibration

Given two cameras $C_L$ (left) and $C_R$ (right), for the same 3D point $\mathbf{P}$:

$$
\mathbf{P}_{C_R} = \mathbf{R} \cdot \mathbf{P}_{C_L} + \mathbf{t}
$$

Here $(\mathbf{R}, \mathbf{t})$ is the transformation from the left camera frame to the right camera frame. To estimate it, we capture the same checkerboard simultaneously with both cameras.

OpenCV's `cv2.stereoCalibrate()` estimates $(\mathbf{R}, \mathbf{t})$ while either fixing the intrinsics of both cameras or refining them simultaneously:

```python
# Assume the intrinsics of both cameras are already calibrated
# K_L, dist_L, K_R, dist_R: intrinsic parameters of each camera

flags = cv2.CALIB_FIX_INTRINSIC  # Fix intrinsics

ret, K_L, dist_L, K_R, dist_R, R, t, E, F = cv2.stereoCalibrate(
    obj_points,
    img_points_left,
    img_points_right,
    K_L, dist_L,
    K_R, dist_R,
    gray.shape[::-1],
    flags=flags,
    criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6)
)

print(f"Stereo RMS error: {ret:.4f} pixels")
print(f"Baseline: {np.linalg.norm(t):.4f} m")
print(f"R:\n{R}")
print(f"t: {t.ravel()}")
```

### 3.2.2 Stereo Rectification

After calibration, the images from both cameras are transformed by **parallel rectification** so that the epipolar lines become horizontal. After rectification, the correspondence search is reduced to a 1D problem (search along the same row), greatly improving stereo matching efficiency.

**Epipolar Geometry Review**:

The plane formed by the two projection centers $O_L, O_R$ and a 3D point $\mathbf{P}$ is called the epipolar plane. The lines where this plane intersects each image plane are the epipolar lines. The point $\mathbf{m}_R$ in the right image corresponding to a point $\mathbf{m}_L$ in the left image must lie on the corresponding epipolar line.

Expressed as an equation:

$$
\tilde{\mathbf{m}}_R^\top \mathbf{F} \tilde{\mathbf{m}}_L = 0
$$

Here $\mathbf{F}$ is the **fundamental matrix**. For calibrated cameras, we use the **essential matrix** $\mathbf{E}$:

$$
\hat{\mathbf{m}}_R^\top \mathbf{E} \hat{\mathbf{m}}_L = 0, \quad \mathbf{E} = [\mathbf{t}]_\times \mathbf{R}
$$

Here $\hat{\mathbf{m}} = \mathbf{K}^{-1}\tilde{\mathbf{m}}$ is the normalized coordinate and $[\mathbf{t}]_\times$ is the skew-symmetric matrix of the translation vector.

**Rectification Procedure**:

Rectification is the process of finding homographies that transform both camera image planes onto a common virtual plane. This common plane is set to be parallel to the baseline vector of the two cameras.

OpenCV's `cv2.stereoRectify()` uses Bouguet's algorithm:

```python
# Compute the rectification mapping
R_L, R_R, P_L, P_R, Q, roi_L, roi_R = cv2.stereoRectify(
    K_L, dist_L, K_R, dist_R,
    gray.shape[::-1],
    R, t,
    alpha=0  # 0: keep only valid pixels, 1: keep all original pixels
)

# Create undistort + rectify maps
map_Lx, map_Ly = cv2.initUndistortRectifyMap(
    K_L, dist_L, R_L, P_L, gray.shape[::-1], cv2.CV_32FC1
)
map_Rx, map_Ry = cv2.initUndistortRectifyMap(
    K_R, dist_R, R_R, P_R, gray.shape[::-1], cv2.CV_32FC1
)

# Rectify images
img_L_rect = cv2.remap(img_L, map_Lx, map_Ly, cv2.INTER_LINEAR)
img_R_rect = cv2.remap(img_R, map_Rx, map_Ry, cv2.INTER_LINEAR)
```

**Verification via the epipolar constraint**: The most intuitive way to confirm that rectification is correct is to draw horizontal lines on the rectified left and right images. If corresponding points lie on the same horizontal line, the rectification is accurate.

```python
# Visualize epipolar-line verification
def draw_epilines(img_L_rect, img_R_rect, num_lines=20):
    h, w = img_L_rect.shape[:2]
    canvas = np.hstack([img_L_rect, img_R_rect])
    for y in np.linspace(0, h-1, num_lines, dtype=int):
        color = tuple(np.random.randint(0, 255, 3).tolist())
        cv2.line(canvas, (0, y), (2*w, y), color, 1)
    return canvas
```

**From disparity to depth**: In rectified stereo images, the horizontal displacement $d = u_L - u_R$ between corresponding left-right points is called the disparity. The depth is:

$$
Z = \frac{f \cdot B}{d}
$$

Here $f$ is the focal length (in pixel units) and $B = \|\mathbf{t}\|$ is the baseline length. Using the $Q$ matrix, 3D point clouds can be computed directly from the disparity map: $\mathbf{P}_{3D} = Q \cdot [u, v, d, 1]^\top$.

---

## 3.3 Camera-LiDAR Extrinsic Calibration

Estimating the extrinsic parameters $(\mathbf{R}, \mathbf{t})$ between a camera and a LiDAR is a core prerequisite for multi-modal sensor fusion. To project LiDAR point clouds onto images or to place image features in 3D space, this transformation must be accurate.

### 3.3.1 Target-based Calibration

The most traditional approach, in which a known geometric target (checkerboard, AprilTag, etc.) is observed simultaneously by the camera and the LiDAR to create correspondences.

**Principle**: The corners of a checkerboard are observed as 2D points in the camera image and as a 3D plane in the LiDAR point cloud. We extract the LiDAR points that fit the checkerboard plane and use the plane's normal and boundary to build 3D-2D correspondences.

**3D-2D correspondence-based method**:

1. Detect checkerboard corners in the camera image → 2D points $\{\mathbf{m}_j\}$
2. Fit the checkerboard plane in the LiDAR point cloud with RANSAC → plane equation $\mathbf{n}^\top \mathbf{p} + d = 0$
3. Extract the checkerboard boundary from the points on the plane to estimate the 3D coordinates of the corners $\{\mathbf{P}_j\}$
4. Estimate $(\mathbf{R}, \mathbf{t})$ from the 3D-2D correspondences $\{(\mathbf{P}_j, \mathbf{m}_j)\}$ via the PnP (Perspective-n-Point) algorithm

**Plane-constraint-based method**:

When the precise 3D positions of corners are hard to estimate, calibration is possible using only plane constraints. Back-project the corners detected by the camera to form 3D rays, and use as correspondences the points where those rays intersect the plane estimated by the LiDAR.

Plane constraints from $n$ checkerboard poses:

$$
\mathbf{n}_i^\top (\mathbf{R} \mathbf{p}_{L,i} + \mathbf{t}) + d_i = 0, \quad \forall i = 1, \ldots, n
$$

Here $\mathbf{n}_i$ is the plane normal in the camera frame and $\mathbf{p}_{L,i}$ is a point on the plane in the LiDAR frame.

```python
import numpy as np
import cv2

def calibrate_camera_lidar_target(
    img_corners_list,    # 2D checkerboard corners in each image [N_imgs x N_corners x 2]
    lidar_planes_list,   # LiDAR plane parameters per observation [N_imgs x (n, d)]
    K, dist,             # Camera intrinsics
    board_corners_3d     # 3D corners in the checkerboard frame [N_corners x 3]
):
    """Target-based Camera-LiDAR extrinsic calibration via PnP."""
    
    # Estimate camera-to-checkerboard transform for each image
    all_points_3d_lidar = []
    all_points_2d_camera = []
    
    for i, (corners_2d, (normal, d)) in enumerate(zip(img_corners_list, lidar_planes_list)):
        # Checkerboard pose as seen from the camera (PnP)
        ret, rvec, tvec = cv2.solvePnP(
            board_corners_3d, corners_2d, K, dist
        )
        R_cam_board, _ = cv2.Rodrigues(rvec)
        
        # Transform checkerboard corners into the camera frame
        corners_cam = (R_cam_board @ board_corners_3d.T + tvec).T
        
        # Collect points on the LiDAR plane
        # (in practice, use plane inliers from the LiDAR point cloud)
        all_points_2d_camera.append(corners_2d)
    
    # The final result is refined by nonlinear optimization
    return R_cam_lidar, t_cam_lidar
```

**Practical considerations**:
- The checkerboard must be large enough for a sufficient number of LiDAR points to fall on it. At minimum A2 size, ideally A0.
- 10-20 observations from various distances and angles are required.
- Enough LiDAR beams must strike the checkerboard plane. This is difficult with sparse LiDAR (16-channel).

### 3.3.2 Targetless Calibration

In practice, preparing and installing a calibration target is often cumbersome or impossible. Targetless calibration estimates the transformation between sensors using only information from natural scenes.

#### Mutual Information (MI) Based Method

**Intuition**: When a LiDAR point cloud is projected onto a camera image, the transformation for which the statistical dependency between the two data is maximized is the correct calibration.

**Definition of Mutual Information**:

The mutual information between two random variables $X$ (LiDAR reflectivity or depth) and $Y$ (image intensity):

$$
\text{MI}(X; Y) = \sum_{x}\sum_{y} p(x, y) \log \frac{p(x, y)}{p(x)p(y)}
$$

Or in entropy form:

$$
\text{MI}(X; Y) = H(X) + H(Y) - H(X, Y)
$$

If $X$ and $Y$ are independent, $\text{MI} = 0$; if perfectly dependent, $\text{MI} = H(X) = H(Y)$.

**Normalized Information Distance (NID)**:

[Koide et al. (2023)](https://arxiv.org/abs/2302.05094) use NID, a normalized MI, as the cost function:

$$
\text{NID}(X; Y) = 1 - \frac{\text{MI}(X; Y)}{H(X, Y)} = \frac{H(X, Y) - \text{MI}(X; Y)}{H(X, Y)}
$$

NID has a range of $[0, 1]$, and smaller values indicate better registration of the two datasets.

**Histogram-based MI computation**:

In practice, MI is estimated from the joint histogram:

```python
import numpy as np

def compute_nid(lidar_intensity, image_intensity, bins=64):
    """
    Compute NID (Normalized Information Distance).
    lidar_intensity: LiDAR reflectivity [N]
    image_intensity: image intensity at the projected location [N]
    """
    # Compute joint histogram
    hist_2d, _, _ = np.histogram2d(
        lidar_intensity, image_intensity, bins=bins,
        range=[[0, 255], [0, 255]]
    )
    
    # Normalize to a probability
    pxy = hist_2d / hist_2d.sum()
    px = pxy.sum(axis=1)  # marginal X
    py = pxy.sum(axis=0)  # marginal Y
    
    # Entropy computation (handle 0 log 0 = 0)
    eps = 1e-10
    H_xy = -np.sum(pxy[pxy > eps] * np.log(pxy[pxy > eps]))
    H_x = -np.sum(px[px > eps] * np.log(px[px > eps]))
    H_y = -np.sum(py[py > eps] * np.log(py[py > eps]))
    
    MI = H_x + H_y - H_xy
    NID = 1.0 - MI / (H_xy + eps)
    
    return NID, MI
```

The optimization of MI-based calibration has no explicit gradient, so we use gradient-free optimizers such as Nelder-Mead, or numerical gradients. [Pandey et al. (2015)](https://onlinelibrary.wiley.com/doi/abs/10.1002/rob.21542) were the first to apply this approach to LiDAR-camera calibration.

#### Edge Alignment Based Method

**Intuition**: Optimize the transformation so that depth-discontinuity edges extracted from the LiDAR point cloud register with edges extracted from the camera image.

Let $\mathbf{e}_L$ denote edges extracted from a LiDAR depth image, and $\mathbf{e}_C$ edges from the camera image:

$$
\min_{\mathbf{R}, \mathbf{t}} \sum_i \text{dist}(\pi(\mathbf{R}\mathbf{p}_{L,i} + \mathbf{t}), \mathbf{e}_C)
$$

Here $\pi(\cdot)$ is the camera projection function and $\text{dist}(\cdot, \mathbf{e}_C)$ is the distance to the nearest edge (using a distance transform).

#### Learning-based Methods

Deep learning approaches such as RegNet and CalibNet take the depth image of the LiDAR point cloud and the camera image as input and directly regress the 6-DoF transformation. These methods can perform calibration without initial values, but they currently fall short of traditional methods in accuracy and depend on the domain of the training data.

### 3.3.3 Koide et al. (2023) — State-of-the-Art Targetless Calibration

The `direct_visual_lidar_calibration` of [Koide et al. (2023)](https://arxiv.org/abs/2302.05094) is currently the most practical targetless LiDAR-camera calibration tool. Its core pipeline:

**Stage 1: LiDAR point cloud densification**

A single scan from a rotational LiDAR (Ouster, Velodyne, etc.) is too sparse for MI registration. The CT-ICP (Continuous-Time ICP) algorithm precisely accumulates several seconds of continuous scans to generate a dense point cloud. Solid-state LiDAR (Livox, etc.) densifies naturally thanks to its non-repetitive scan pattern.

**Stage 2: SuperGlue-based initial estimation**

The dense point cloud is rendered from a virtual camera viewpoint to produce a LiDAR intensity image. SuperGlue (learning-based matching) detects 2D-2D correspondences between this rendered image and the actual camera image. These are converted to 2D-3D correspondences, from which the initial transformation is estimated via RANSAC + PnP.

This stage is innovative because SuperGlue solves the cross-modal correspondence problem of matching images from different modalities (LiDAR intensity vs. camera RGB). The success rate of the initial estimation is over 80%.

**Stage 3: NID-based fine registration**

Starting from the initial estimate, a Nelder-Mead optimization minimizes the NID. View-based hidden point removal is used to discard LiDAR points not visible from the camera, improving registration quality.

**Result**: Mean translation error of 0.043 m and rotation error of 0.374 degrees. It operates across a wide variety of combinations — rotational/solid-state LiDAR, pinhole/fisheye/omnidirectional cameras.

### 3.3.4 Practical Tool Comparison

| Tool | Approach | Target needed | Accuracy | Automation level |
|------|------|----------|--------|-----------|
| Autoware Calibration Toolkit | Target-based | O | High | Semi-automatic |
| `direct_visual_lidar_calibration` (Koide) | Targetless (NID) | X | High | Automatic |
| ACSC (Automatic Calibration) | Target-based + auto corner | O | High | Automatic |
| LiveCalib | Online | X | Medium | Fully automatic |

Recently, [MFCalib (2024)](https://arxiv.org/abs/2409.00992) significantly improved the accuracy of single-shot targetless calibration by simultaneously leveraging depth-continuity/discontinuity edges and intensity-discontinuity edges. A distinguishing feature is that it resolves the edge inflation problem by modeling the physical measurement principle of the LiDAR beam.

In practice, a dual strategy is effective: first perform a precise target-based calibration, and then monitor calibration drift during operation with a targetless method.

---

## 3.4 Camera-IMU Extrinsic + Temporal Calibration

The key is to simultaneously estimate not only the spatial displacement (extrinsic) between the camera and the IMU but also the temporal offset. Modern VIO (Visual-Inertial Odometry) systems depend strongly on this calibration.

### 3.4.1 Why Time Offset Matters

The camera and the IMU generate data on different clocks. When a time offset $t_d$ exists between the two sensors, the IMU data corresponding to a camera timestamp $t_c$ is actually from time $t_c + t_d$.

A typical camera-IMU time offset is on the order of several to tens of milliseconds. Ignoring this offset dramatically increases the reprojection error under fast rotation. For example, if the time offset is 10 ms and the camera is rotating at 100 deg/s, a 1-degree rotation error results.

### 3.4.2 Kalibr: Continuous-Time B-Spline-Based Calibration

Kalibr, proposed by [Furgale et al. (2013)](https://ieeexplore.ieee.org/document/6696514), is the de facto standard for camera-IMU calibration.

**Key idea**: Represent the trajectory not as a sequence of discrete poses but as a continuous-time B-spline. This naturally handles sensors with different sampling rates (camera: 20-30 Hz, IMU: 200-1000 Hz).

**B-Spline trajectory representation**:

In a cubic B-spline, the pose $\mathbf{T}(t) \in SE(3)$ at time $t$ is expressed as a weighted combination of control points $\{\mathbf{T}_i\}$:

$$
\mathbf{T}(t) = \mathbf{T}_i \prod_{j=1}^{3} \text{Exp}(\mathbf{B}_j(u) \cdot \Omega_{i+j})
$$

Here:
- $u = (t - t_i) / \Delta t$ is the normalized time ($0 \leq u < 1$)
- $\mathbf{B}_j(u)$ are the cubic B-spline basis coefficients
- $\Omega_{i+j} = \text{Log}(\mathbf{T}_{i+j-1}^{-1} \mathbf{T}_{i+j})$ is the Lie algebra representation of the relative transform between adjacent control points
- $\text{Exp}, \text{Log}$ are the exponential/logarithm maps of $SE(3)$

Key advantages of the B-spline:
1. **Differentiable**: velocity and acceleration at any time can be computed analytically → direct connection to the IMU observation model
2. **Asynchronous sensor handling**: not constrained by each sensor's timestamp
3. **Locality**: each basis function affects only 4 control points → sparse optimization is possible

**Observation model**:

Camera observation: project a 3D landmark using the trajectory pose at time $t_c + t_d$ (time-offset-corrected):

$$
\mathbf{e}_{\text{cam},k} = \mathbf{m}_k - \pi\left(\mathbf{T}_{CB} \cdot \mathbf{T}(t_{c,k} + t_d) \cdot \mathbf{p}_w\right)
$$

Here $\mathbf{T}_{CB}$ is the camera-IMU extrinsic (the target of estimation).

IMU observation: predict acceleration and angular velocity from the derivatives of the trajectory at time $t_{\text{imu}}$:

$$
\mathbf{e}_{\text{accel},k} = \mathbf{a}_k - \left[\mathbf{R}(t_k)^\top(\ddot{\mathbf{p}}(t_k) - \mathbf{g}) + \mathbf{b}_a\right]
$$
$$
\mathbf{e}_{\text{gyro},k} = \boldsymbol{\omega}_k - \left[\boldsymbol{\omega}(t_k) + \mathbf{b}_g\right]
$$

**Optimization problem**:

$$
\min_{\mathbf{T}_{CB}, t_d, \mathbf{b}_a, \mathbf{b}_g, \{\mathbf{T}_i\}} \sum_k \|\mathbf{e}_{\text{cam},k}\|^2_{\Sigma_c} + \sum_k \left(\|\mathbf{e}_{\text{accel},k}\|^2_{\Sigma_a} + \|\mathbf{e}_{\text{gyro},k}\|^2_{\Sigma_g}\right)
$$

This is a nonlinear least-squares problem that can be solved with Gauss-Newton or LM. Thanks to the locality of the B-spline, the Jacobian is sparse, so even large-scale problems are handled efficiently.

### 3.4.3 Kalibr Execution Guide

```bash
# 1. Prepare the AprilGrid target
kalibr_create_target_pdf --type apriltag \
    --nx 6 --ny 6 --tsize 0.024 --tspace 0.3

# 2. Collect data (ROS bag)
#    - Place the target in the camera's view and move the sensor rig diversely
#    - Include both rotation and translation along all axes
#    - Minimum 60 seconds, ideally more than 2 minutes

# 3. Run the Camera-IMU calibration
kalibr_calibrate_imu_camera \
    --target april_6x6_24x24mm.yaml \
    --cam camchain.yaml \
    --imu imu.yaml \
    --bag calibration.bag \
    --bag-freq 20.0 \
    --timeoffset-padding 0.1
```

**IMU configuration file (imu.yaml) example**:
```yaml
# imu.yaml
rostopic: /imu0
update_rate: 200.0  # Hz

# IMU noise parameters (measured via Allan variance, see Section 3.4.4)
accelerometer_noise_density: 0.01    # m/s^2/sqrt(Hz)
accelerometer_random_walk: 0.0002   # m/s^3/sqrt(Hz)
gyroscope_noise_density: 0.005      # rad/s/sqrt(Hz)
gyroscope_random_walk: 4.0e-06      # rad/s^2/sqrt(Hz)
```

**Key tips for data collection**:
1. **Motion diversity**: excite all 6-DoF. Rotation about each axis is especially important.
2. **Motion speed**: too slow makes IMU bias hard to estimate, too fast blurs the images.
3. **Target visibility**: the target should be visible during more than 80% of the total collection time.
4. **Start and end**: begin and end at rest to facilitate IMU bias initialization.

**Interpreting the results**: Items to check in Kalibr's output:
- `T_cam_imu`: camera-IMU extrinsic (4x4 transformation matrix)
- `timeshift_cam_imu`: time offset $t_d$ (usually several ms)
- Reprojection-error distribution: a mean of 0.2-0.5 px is ideal
- Accelerometer/gyro residuals: must match the noise model

### 3.4.4 Allan Variance Measurement Hands-On

Accurately knowing the IMU noise parameters directly affects not only Kalibr calibration but the performance of every IMU-based system. Allan variance is a technique for analyzing the noise characteristics of a time series according to cluster time, originally developed to measure the stability of atomic clocks.

**Allan Variance Definition**:

For a time series $\{x_k\}$, the Allan variance at cluster time $\tau = n \cdot \tau_0$ ($\tau_0$: sampling period):

$$
\sigma^2(\tau) = \frac{1}{2(N-2n)} \sum_{k=1}^{N-2n} \left[\bar{x}_{k+n} - \bar{x}_k\right]^2
$$

Here $\bar{x}_k = \frac{1}{n}\sum_{j=0}^{n-1} x_{k+j}$ is the cluster mean.

**Noise identification on a log-log plot**:

Plotting the Allan deviation $\sigma(\tau)$ against $\tau$ on a log-log scale, each noise source appears as a line with a distinct slope:

| Noise type | Slope | $\sigma(\tau)$ |
|-----------|-------|---------------|
| Quantization noise | $-1$ | $\propto \tau^{-1}$ |
| **Angle/Velocity random walk** | $-1/2$ | $\propto \tau^{-1/2}$ |
| **Bias instability** | $0$ | constant (minimum) |
| Rate random walk | $+1/2$ | $\propto \tau^{+1/2}$ |
| Rate ramp | $+1$ | $\propto \tau$ |

**Hands-on code**:

```python
import numpy as np
import matplotlib.pyplot as plt

def compute_allan_variance(data, dt, max_clusters=100):
    """
    Compute the Allan variance.
    data: 1D time series (e.g., gyro x-axis raw data) [N]
    dt: sampling period [s]
    max_clusters: number of clusters distributed on a log scale
    """
    N = len(data)
    max_n = N // 2
    
    # Choose cluster sizes on a log scale
    n_values = np.unique(
        np.logspace(0, np.log10(max_n), max_clusters).astype(int)
    )
    
    taus = []
    allan_vars = []
    
    for n in n_values:
        tau = n * dt
        
        # Compute cluster means
        n_clusters = N // n
        trimmed = data[:n_clusters * n]
        clusters = trimmed.reshape(n_clusters, n)
        cluster_means = clusters.mean(axis=1)
        
        # Allan variance
        diffs = np.diff(cluster_means)
        avar = 0.5 * np.mean(diffs**2)
        
        taus.append(tau)
        allan_vars.append(avar)
    
    return np.array(taus), np.array(allan_vars)

def extract_imu_noise_params(taus, allan_vars, dt):
    """
    Extract IMU noise parameters from the Allan variance plot.
    """
    adev = np.sqrt(allan_vars)
    
    # 1. Random walk (slope -1/2): value at tau=1
    # N (noise density) = sigma(tau=1) 
    idx_1s = np.argmin(np.abs(taus - 1.0))
    noise_density = adev[idx_1s]
    
    # 2. Bias instability: minimum of the Allan deviation
    bias_instability = np.min(adev)
    tau_min = taus[np.argmin(adev)]
    
    # 3. Random walk (slope +1/2): extracted from the long-term slope
    # K (rate random walk) = sigma(tau=3) * sqrt(3) (approximation)
    # In practice, extract precisely via linear regression
    
    return {
        'noise_density': noise_density,
        'bias_instability': bias_instability,
        'tau_min': tau_min
    }

# Usage example
# 1. Record the IMU stationary for at least 2 hours
# 2. Analyze raw data from each gyro/accelerometer axis

# Example: 200 Hz IMU data, 2 hours long
dt = 1.0 / 200  # 5ms
# gyro_x = np.loadtxt("imu_static_gyro_x.txt")  # actual data

# Demo with simulated data
np.random.seed(42)
N = 200 * 3600 * 2  # 2 hours
noise_density = 0.005  # rad/s/sqrt(Hz)
bias = 0.0001  # rad/s (slow drift)
gyro_x = noise_density * np.sqrt(1/dt) * np.random.randn(N)
gyro_x += bias * np.cumsum(np.random.randn(N)) * np.sqrt(dt)

taus, avars = compute_allan_variance(gyro_x, dt)
params = extract_imu_noise_params(taus, avars, dt)

print(f"Gyroscope noise density: {params['noise_density']:.6f} rad/s/sqrt(Hz)")
print(f"Bias instability: {params['bias_instability']:.6f} rad/s")
print(f"Min at tau = {params['tau_min']:.1f} s")

# Allan deviation plot
plt.figure(figsize=(10, 6))
plt.loglog(taus, np.sqrt(avars), 'b-', linewidth=1.5)
plt.xlabel('Cluster time τ (s)')
plt.ylabel('Allan deviation σ(τ)')
plt.title('Gyroscope Allan Deviation')
plt.grid(True, which='both', alpha=0.3)

# Slope reference lines
tau_ref = np.array([0.01, 100])
plt.loglog(tau_ref, params['noise_density'] / np.sqrt(tau_ref), 
           'r--', label='Random walk (-1/2)')
plt.axhline(y=params['bias_instability'], color='g', linestyle='--', 
            label=f'Bias instability ({params["bias_instability"]:.1e})')
plt.legend()
plt.savefig('allan_deviation.png', dpi=150)
plt.show()
```

**Practical tips for Allan variance measurement**:
- **Stationary data collection**: place the IMU on a rigid surface free from vibration and record for at least 2 hours (ideally 6 hours). With short records, the minimum of the bias instability cannot be identified accurately.
- **Comparison with the data sheet**: compare the noise density value in the manufacturer's data sheet with the value extracted from the Allan variance to verify the sensor's health.
- **Connection to Kalibr**: the `accelerometer_noise_density` and `gyroscope_noise_density` entered into Kalibr's `imu.yaml` are exactly the Allan deviation at $\tau = 1$ second.
- **Temperature stabilization**: IMUs are sensitive to temperature, so warm up for 15-30 minutes after power-on before starting the recording.

---

## 3.5 LiDAR-IMU Extrinsic Calibration

The problem of estimating the extrinsic parameters between LiDAR and IMU reduces to the classical **hand-eye calibration** problem.

### 3.5.1 Hand-Eye Calibration (AX = XB)

The archetypal hand-eye calibration problem asks for the relation between the motion of a robot arm's end (hand), observed from the robot's base, and the motion of a target, observed from a camera mounted at the hand ([Tsai & Lenz, 1989](https://ieeexplore.ieee.org/document/34770)).

**Problem definition**:

Given sensors $A$ (e.g., LiDAR) and $B$ (e.g., IMU) rigidly mounted on a rigid body, let the relative motions observed by each sensor between two instants $i, j$ be $\mathbf{A}_{ij}$ and $\mathbf{B}_{ij}$:

$$
\mathbf{A}_{ij} = \mathbf{T}_A^{-1}(t_i) \cdot \mathbf{T}_A(t_j) \quad \text{(relative motion of sensor A)}
$$
$$
\mathbf{B}_{ij} = \mathbf{T}_B^{-1}(t_i) \cdot \mathbf{T}_B(t_j) \quad \text{(relative motion of sensor B)}
$$

The fixed transformation $\mathbf{X} = \mathbf{T}_{AB}$ between the two sensors satisfies:

$$
\mathbf{A}_{ij} \mathbf{X} = \mathbf{X} \mathbf{B}_{ij}
$$

This is the $\mathbf{AX} = \mathbf{XB}$ equation. We must solve for $\mathbf{X} \in SE(3)$.

**Separating rotation from translation**:

Separated into rotation and translation:

$$
\mathbf{R}_A \mathbf{R}_X = \mathbf{R}_X \mathbf{R}_B \quad \text{(rotation)}
$$
$$
\mathbf{R}_A \mathbf{t}_X + \mathbf{t}_A = \mathbf{R}_X \mathbf{t}_B + \mathbf{t}_X \quad \text{(translation)}
$$

The rotation equation is independent of $\mathbf{t}_X$, so we first solve for $\mathbf{R}_X$ and then solve for $\mathbf{t}_X$ from the translation equation.

**[Tsai & Lenz (1989)](https://ieeexplore.ieee.org/document/34770) solution**:

Convert the rotation equation into angle-axis representation. If the rotation axis of $\mathbf{R}_A$ is $\hat{\mathbf{a}}$ and the rotation angle is $\alpha$, then using modified Rodrigues parameters:

$$
\text{skew}(\hat{\mathbf{a}}_A + \hat{\mathbf{a}}_B) \cdot \mathbf{r}_X = \hat{\mathbf{a}}_A - \hat{\mathbf{a}}_B
$$

Here $\mathbf{r}_X$ is the modified Rodrigues vector of $\mathbf{R}_X$. Stacking this equation over many motion pairs yields a linear system $\mathbf{C} \mathbf{r}_X = \mathbf{d}$, solvable with a minimum of 2 motion pairs (with non-parallel rotation axes). The translation vector $\mathbf{t}_X$ is solved by a similar linear system.

**Leveraging more motion pairs**: in practice, dozens to hundreds of motion pairs are used; the resulting overdetermined system is solved by least squares and refined by LM optimization.

### 3.5.2 Motion-based Automatic Calibration

The key requirement of hand-eye calibration is that the motion estimates (odometry) for each sensor must already exist. Since LiDAR odometry and IMU integration operate independently, data can be collected by moving the sensor freely without any calibration target.

```python
import numpy as np
from scipy.spatial.transform import Rotation

def hand_eye_calibration_tsai(A_rotations, A_translations, 
                                B_rotations, B_translations):
    """
    Tsai-Lenz hand-eye calibration (AX = XB).
    
    A_rotations: [N x 3 x 3] relative rotation matrices of sensor A
    A_translations: [N x 3] relative translations of sensor A
    B_rotations: [N x 3 x 3] relative rotation matrices of sensor B
    B_translations: [N x 3] relative translations of sensor B
    """
    N = len(A_rotations)
    
    # Step 1: Estimate rotation RX
    C = []
    d = []
    for i in range(N):
        # Convert to angle-axis representation
        rA = Rotation.from_matrix(A_rotations[i]).as_rotvec()
        rB = Rotation.from_matrix(B_rotations[i]).as_rotvec()
        
        alpha = np.linalg.norm(rA)
        beta = np.linalg.norm(rB)
        
        if alpha < 1e-6 or beta < 1e-6:
            continue  # Ignore small rotations
        
        # Modified Rodrigues parameters
        a_prime = np.tan(alpha / 2) * rA / alpha
        b_prime = np.tan(beta / 2) * rB / beta
        
        # skew(a' + b') * rX = a' - b'
        skew_sum = np.array([
            [0, -(a_prime[2]+b_prime[2]), a_prime[1]+b_prime[1]],
            [a_prime[2]+b_prime[2], 0, -(a_prime[0]+b_prime[0])],
            [-(a_prime[1]+b_prime[1]), a_prime[0]+b_prime[0], 0]
        ])
        
        C.append(skew_sum)
        d.append(a_prime - b_prime)
    
    C = np.vstack(C)
    d = np.concatenate(d)
    
    # Least-squares solve
    rX, _, _, _ = np.linalg.lstsq(C, d, rcond=None)
    
    # Modified Rodrigues → rotation matrix
    angle = 2 * np.arctan(np.linalg.norm(rX))
    if angle > 1e-6:
        axis = rX / np.linalg.norm(rX)
        R_X = Rotation.from_rotvec(angle * axis).as_matrix()
    else:
        R_X = np.eye(3)
    
    # Step 2: Estimate translation tX
    C_t = []
    d_t = []
    for i in range(N):
        C_t.append(A_rotations[i] - np.eye(3))
        d_t.append(R_X @ B_translations[i] - A_translations[i])
    
    C_t = np.vstack(C_t)
    d_t = np.concatenate(d_t)
    
    t_X, _, _, _ = np.linalg.lstsq(C_t, d_t, rcond=None)
    
    return R_X, t_X
```

### 3.5.3 LI-Init (FAST-LIO Lineage)

Modern LIO systems such as FAST-LIO2 include a feature that automatically initializes the LiDAR-IMU extrinsic parameters online. The key idea of the **LI-Init** approach:

1. Estimate the attitude using only IMU data, and estimate the pose via LiDAR matching
2. Iteratively refine the relative transform from the difference of the two estimates
3. Include the LiDAR-IMU extrinsic in the state vector of the Error-State Iterated Kalman Filter (ESIKF) to estimate it online

This approach estimates the extrinsic parameters automatically at the start of the LIO system, with no separate calibration procedure. Convergence occurs after a few seconds of sufficiently diverse motion.

**Advantages**: no separate tools or procedures needed. Immediately usable in the field.
**Disadvantages**: may not converge or may be inaccurate if initial motion is insufficient. Accuracy can be lower than with target-based methods.

**GRIL-Calib**: When motion is confined to a plane, as with ground robots, existing methods suffer reduced accuracy because some axes are poorly observable. [GRIL-Calib (Kim et al., 2024)](https://arxiv.org/abs/2312.14035) leverages the ground-plane residual in LiDAR odometry and integrates a ground-plane motion (GPM) constraint into the optimization, enabling 6-DoF calibration parameters to be estimated from planar motion alone.

So far we have addressed calibration between a single LiDAR and a single IMU. However, in systems that use multiple LiDARs, such as autonomous vehicles, the relative poses between LiDARs must also be determined.

---

## 3.6 LiDAR-LiDAR Extrinsic Calibration

This is the problem of estimating the relative poses between each LiDAR in a multi-LiDAR rig. It is essential when LiDARs are placed at the front/side/rear of an autonomous vehicle.

### 3.6.1 Problem Setup

Given $n$ LiDARs $\{L_1, L_2, \ldots, L_n\}$ rigidly mounted on a vehicle body, estimate the relative transformations $\{\mathbf{T}_{L_1 L_i}\}_{i=2}^{n}$ of the other LiDARs with respect to the reference LiDAR $L_1$.

### 3.6.2 Target-based Methods

Use a large planar target (panel) so that several LiDARs can observe it simultaneously. Fit the target plane in each LiDAR with RANSAC, and estimate the relative transform from constraints on the plane parameters.

### 3.6.3 Targetless: ICP-Based

The most natural targetless method is to register the point clouds of two LiDARs with overlapping regions via ICP.

**When there is an overlapping region**:

If the FoVs of two LiDARs overlap, apply ICP directly to the point clouds in that region:

$$
\min_{\mathbf{R}, \mathbf{t}} \sum_{i} \|\mathbf{p}_{L_2,i} - (\mathbf{R} \mathbf{p}_{L_1,i'} + \mathbf{t})\|^2
$$

Point-to-plane ICP generally converges faster:

$$
\min_{\mathbf{R}, \mathbf{t}} \sum_{i} \left[(\mathbf{R} \mathbf{p}_{L_1,i'} + \mathbf{t} - \mathbf{p}_{L_2,i})^\top \mathbf{n}_{L_2,i}\right]^2
$$

Here $\mathbf{n}_{L_2,i}$ is the normal vector at the target point.

**When there is no overlapping region**:

If the FoVs do not overlap, direct registration is impossible. In this case:
1. Hand-eye calibration (Section 3.5.1): extract relative motions from each LiDAR's odometry to solve AX=XB.
2. SLAM-based: each LiDAR performs SLAM independently, and the relative transform is estimated from matching within a global map of the common environment.

### 3.6.4 Feature-based Methods

In structured environments (buildings, roads, etc.), geometric features such as planes, pillars, and edges are extracted and used for matching. This method is more robust to noise than point-wise matching:

- Extract the same large plane (wall, floor) from multiple LiDARs
- Compute the plane parameters $(n_i, d_i)$ in each LiDAR frame
- Estimate the relative transform from pairs of corresponding planes

A minimum of 3 non-coplanar plane correspondences is required.

The calibrations covered so far (3.1-3.6) concerned relations among cameras, LiDARs, and IMUs. In outdoor systems that exploit GNSS, the spatial relation between the GNSS antenna and the IMU must also be known precisely.

---

## 3.7 GNSS-IMU Lever Arm & Boresight

The spatial relation between the GNSS antenna and the IMU is called the **lever arm**. This is a core calibration parameter in GNSS/INS integrated navigation.

### 3.7.1 Lever Arm Vector

The 3D vector $\mathbf{l} = [l_x, l_y, l_z]^\top$ between the phase center of the GNSS antenna and the IMU's origin is defined in the IMU body frame.

GNSS measures the position of the antenna's phase center, but what we want is the position of the IMU (or vehicle reference point):

$$
\mathbf{p}_{\text{IMU}} = \mathbf{p}_{\text{GNSS}} - \mathbf{R}_{\text{body}}^{\text{nav}} \cdot \mathbf{l}
$$

Here $\mathbf{R}_{\text{body}}^{\text{nav}}$ is the rotation matrix from the body frame to the navigation frame. As the vehicle rotates, the GNSS antenna's position changes, so failing to correct for the lever arm produces position errors. If the lever arm is 1 m and the vehicle tilts by 10 deg, a position error of about 17 cm results.

### 3.7.2 Lever Arm Estimation Methods

**Method 1: Physical measurement**

The most intuitive method is to measure directly with a tape measure, laser rangefinder, or similar. The accuracy is on the order of several cm, which is sufficient for most applications.

**Method 2: Filter-based online estimation**

Include the lever arm $\mathbf{l}$ in the EKF state vector and estimate it online. For the lever arm to be observable in the GNSS observation model, sufficient rotational motion is needed. Straight-line motion alone makes it hard to estimate the forward component ($l_x$) of the lever arm.

**Method 3: Post-processing**

Process the collected GNSS/IMU data with a forward-backward smoother to optimize the lever arm. Primarily used in precision surveying.

### 3.7.3 GNSS Antenna Phase Center

The antenna phase center (APC) of a GNSS antenna does not coincide with the antenna's physical center and varies with the satellite's elevation angle and frequency. This variation is called the Phase Center Variation (PCV), and is on the order of mm to cm.

In precision positioning (RTK/PPP), antenna phase-center correction data (ANTEX files) must be applied. For robotics-grade applications this can usually be ignored, but when survey-grade accuracy is required it must be considered.

---

## 3.8 Online / Continuous Calibration

Calibration parameters may change during operation. Temperature variations, vibration, or shocks can slightly deform the sensor mount, causing the initial calibration to gradually become inaccurate. **Online calibration** is needed to correct for this.

### 3.8.1 Self-Calibration during SLAM

This approach includes calibration parameters in the SLAM system's state vector and estimates them during operation.

**EKF-based**: OpenVINS includes the camera intrinsics, camera-IMU extrinsic, and time offset in its state vector and estimates them online.

State vector extension:
$$
\mathbf{x} = \begin{bmatrix} \mathbf{x}_{\text{nav}} \\ \mathbf{x}_{\text{calib}} \end{bmatrix}
= \begin{bmatrix} \mathbf{q}, \mathbf{p}, \mathbf{v}, \mathbf{b}_g, \mathbf{b}_a \\ \mathbf{q}_{CI}, \mathbf{p}_{CI}, t_d, f_x, f_y, c_x, c_y, k_1, \ldots \end{bmatrix}
$$

The process model for the calibration parameters is typically a random walk:

$$
\dot{\mathbf{x}}_{\text{calib}} = \mathbf{w}_{\text{calib}}, \quad \mathbf{w}_{\text{calib}} \sim \mathcal{N}(\mathbf{0}, \mathbf{Q}_{\text{calib}})
$$

The magnitude of $\mathbf{Q}_{\text{calib}}$ reflects how quickly the calibration parameters can change. If too large, the parameters fluctuate unstably; if too small, real changes cannot be tracked.

**Factor graph-based**: LIO-SAM and VINS-Mono can also include the extrinsic parameters as optimization variables. The noise model of the between factor controls the stability of the calibration.

### 3.8.2 Correcting Extrinsic Drift

For long-duration systems (autonomous vehicles, robots), strategies for detecting and correcting drift in the extrinsic parameters:

1. **Reprojection-error monitoring**: continuously monitor the edge-registration quality when LiDAR point clouds are projected onto the image. When the registration quality drops, trigger a recalibration.

2. **Periodic recalibration**: periodically rerun targetless calibration using operational data.

3. **Online fine-tuning**: use the current calibration as the initial value and continuously optimize within a small range.

Recently, [CalibRefine (2025)](https://arxiv.org/abs/2502.17648) proposed a deep-learning-based framework that takes raw LiDAR point clouds and camera images directly as input, performs online targetless calibration, and improves accuracy through iterative post-refinement automatically.

### 3.8.3 OpenCalib: An Integrated Calibration Framework for Autonomous Driving

[OpenCalib (Yan et al., 2023)](https://arxiv.org/abs/2205.14087) is an open-source tool that integrates the various calibrations needed by autonomous-driving systems into a single framework.

**Calibration types supported**:

| Sensor pair | Method | Target |
|---------|------|------|
| Camera intrinsic | Zhang's method | Checkerboard/AprilTag |
| Camera-Camera | Stereo calibration | Shared target |
| Camera-LiDAR | PnP / Edge alignment | With/without |
| LiDAR-LiDAR | ICP / Feature matching | None |
| Camera-Ground | Vanishing point | None |
| Online correction | Monitoring + recalibration | None |

**OpenCalib's design philosophy**:
- **Modularity**: each calibration type is implemented as an independent module, so only what is needed can be used
- **Unified interface**: a ROS-based unified interface handles diverse sensor inputs
- **Visualization**: calibration results are visualized in real time so quality can be assessed intuitively

Calibration in an autonomous-driving system matters not only for a single sensor pair but also for the consistency of the **sensor chain**. For example, if a vehicle has 6 cameras and 1 LiDAR, even if the camera-LiDAR calibrations are performed independently, the relative poses among the cameras must remain consistent. OpenCalib integrates such consistency constraints into a global optimization.

---

## 3.9 Temporal Calibration

Time synchronization between sensors is as important as spatial calibration. On a fast-moving platform, even a few milliseconds of time error translate into centimeter-level position error.

### 3.9.1 Hardware Synchronization

Hardware synchronization is the most accurate and reliable method.

**Trigger-based synchronization**:

A single master timer sends hardware trigger signals to all sensors, causing them to capture data simultaneously.

- **Camera trigger**: the exposure start time is controlled via a GPIO pin connected to the external trigger input
- **LiDAR sync**: some LiDARs (Ouster, etc.) support a PPS (Pulse Per Second) input to synchronize the start time of the scan
- **IMU sync**: the IMU's sampling clock is locked to an external reference

**PPS (Pulse Per Second)**:

A GNSS receiver outputs a 1 Hz electrical pulse (PPS) synchronized with GPS time. Since the rising edge of this pulse corresponds exactly to a GPS-second boundary, it can be used as a reference to align every sensor's local timestamp to GPS time.

```
                   PPS Signal (from GNSS)
                   ───┐     ┌───┐     ┌───┐     ┌───
                      │     │   │     │   │     │
                      └─────┘   └─────┘   └─────┘
                   t=0     t=1       t=2       t=3   (GPS seconds)
                   
   Camera capture  ──X──X──X──X──X──X──X──X──X──X──  (30 Hz)
   LiDAR scan      ──X─────X─────X─────X─────X─────  (10 Hz)
   IMU sample      ──XXXX──XXXX──XXXX──XXXX──XXXX──  (200 Hz)
```

The essence of PPS synchronization is to measure the offset between each sensor's local timestamp and the PPS pulse, thereby aligning all data to a common time axis (GPS time).

### 3.9.2 PTP (Precision Time Protocol)

IEEE 1588 PTP is a protocol that provides microsecond-level time synchronization over Ethernet. It is far more precise than the millisecond-level accuracy of NTP (Network Time Protocol).

**How PTP works**:

1. **Master-slave structure**: one master clock (Grandmaster) in the network provides the reference time
2. **Sync message**: the master periodically broadcasts a Sync message
3. **Follow-up**: the precise transmission timestamp is sent in a separate message
4. **Delay Request/Response**: the slave measures the network delay
5. **Offset computation**: $\text{offset} = \frac{(t_2 - t_1) - (t_4 - t_3)}{2}$

```
   Master (Grandmaster)              Slave (Sensor)
         |                                |
         |------- Sync (t1) ------------->|  (t2: reception time)
         |                                |
         |------- Follow-up (t1) -------->|  (conveys the exact t1)
         |                                |
         |<------ Delay_Req (t3) ---------|  (t3: transmission time)
         |                                |
         |------- Delay_Resp (t4) ------->|  (conveys the reception time t4)
         |                                |
         
   offset = [(t2 - t1) - (t4 - t3)] / 2
   delay  = [(t2 - t1) + (t4 - t3)] / 2
```

Modern LiDARs (Ouster, Hesai, etc.) and industrial cameras (FLIR/Lucid, etc.) support PTP hardware timestamping, enabling microsecond-level synchronization without software overhead.

### 3.9.3 Software Synchronization

When hardware synchronization is not available, time alignment is done in software.

**Host clock based**: when all sensor data arrives at the host computer, the host's system clock timestamps them. Simple, but the jitter of USB/network delays introduces uncertainty on the order of milliseconds.

**ROS Time**: in ROS, `ros::Time::now()` records the message reception time. When the driver converts the sensor's local timestamp to ROS time, the accuracy of that conversion determines the overall synchronization precision.

### 3.9.4 Online Estimation of the Time Offset

[Li & Mourikis (2014)](https://journals.sagepub.com/doi/abs/10.1177/0278364913515286) proposed including the time offset $t_d$ in the state vector of VIO and estimating it online. The key idea:

**Reflecting the time offset in the observation model**:

The actual sensor pose at camera observation time $t_c$ is at $t_c + t_d$. Reflecting this in the IMU preintegration:

$$
\mathbf{z}(t_c) = \pi(\mathbf{T}(t_c + t_d) \cdot \mathbf{p}_w)
$$

Assuming $t_d$ is small, a first-order Taylor expansion gives:

$$
\mathbf{T}(t_c + t_d) \approx \mathbf{T}(t_c) \cdot \text{Exp}(\boldsymbol{\xi} \cdot t_d)
$$

Here $\boldsymbol{\xi}$ is the body velocity (angular + linear velocity) at $t_c$.

Through this approximation, the Jacobian with respect to $t_d$ can be computed analytically, so it can be estimated jointly with the other state variables in an EKF or optimization framework.

Recently, [iKalibr (Chen et al., 2024)](https://arxiv.org/abs/2407.11420) extended this idea to multiple sensors and proposed a unified calibration tool that estimates spatio-temporal parameters between heterogeneous sensors such as LiDAR, camera, IMU, and radar **all at once** in a B-spline continuous-time framework (IEEE T-RO 2025).

**Observability conditions**: For the time offset to be observable, the platform must undergo sufficient accelerated motion. With constant-velocity straight-line motion, the time offset cannot be estimated (temporal shift is indistinguishable from spatial shift).

Kalibr (Section 3.4.2), OpenVINS, VINS-Mono, and other modern VIO systems all implement variants of this method.

### 3.9.5 Practical Synchronization Strategy Guide

| Precision required | Recommended method | Cost |
|-----------|----------|------|
| $< 1\mu s$ | PPS + HW trigger | High (dedicated HW required) |
| $1\mu s - 100\mu s$ | PTP | Medium (PTP-capable equipment) |
| $100\mu s - 1ms$ | NTP + online estimation | Low |
| $> 1ms$ | Host clock + online estimation | None |

**Autonomous-driving grade**: PPS + PTP combination is standard. All sensors are synchronized to GPS time.

**Research/prototyping grade**: software synchronization + online time-offset estimation is often sufficient. Kalibr estimates the initial offset and the VIO system fine-tunes it online.

**Core principle**: if hardware synchronization is available, use it. Software synchronization complements, but does not replace, hardware synchronization.

---

## 3.10 Chapter Summary

We summarize the overall system of calibrations covered in this chapter.

| Calibration type | Parameters estimated | Key method | Minimum requirement |
|----------------|-------------|----------|-------------|
| Camera intrinsic | $\mathbf{K}, \mathbf{d}$ | Zhang's method | 15-25 checkerboard images |
| Stereo extrinsic | $\mathbf{R}, \mathbf{t}$ | Shared target + stereoCalibrate | 15-25 simultaneous observation pairs |
| Camera-LiDAR | $\mathbf{T}_{CL}$ | Target-based / NID | 10-20 target observations / natural scene |
| Camera-IMU | $\mathbf{T}_{CI}, t_d$ | Kalibr (B-spline) | 60+ seconds of diverse motion |
| LiDAR-IMU | $\mathbf{T}_{LI}$ | Hand-eye (AX=XB) / LI-Init | Diverse motion |
| LiDAR-LiDAR | $\mathbf{T}_{L_1 L_2}$ | ICP / Feature matching | Overlapping region or motion data |
| GNSS-IMU | $\mathbf{l}$ (lever arm) | Physical measurement / EKF estimation | Rotational motion |
| Temporal | $t_d$ | PPS/PTP + online estimation | Accelerated motion |

**Recommended calibration order**:

1. Intrinsic of each camera (performed independently)
2. Stereo extrinsic (if applicable)
3. Camera-IMU extrinsic + temporal (Kalibr)
4. LiDAR-IMU extrinsic (hand-eye or LI-Init)
5. Camera-LiDAR extrinsic (computed indirectly as the chain of camera-IMU and LiDAR-IMU, or calibrated directly)
6. GNSS lever arm
7. Set up online calibration (in-operation correction)

Example of indirect computation: the camera-LiDAR transform can be obtained as $\mathbf{T}_{CL} = \mathbf{T}_{CI} \cdot \mathbf{T}_{IL}$. However, errors accumulate, so verifying it via direct calibration is recommended.

**Key paper summary**:

- [Zhang (2000)](https://ieeexplore.ieee.org/document/888718): camera intrinsic — flexible homography-based calibration
- [Furgale et al. (2013)](https://ieeexplore.ieee.org/document/6696514): camera-IMU — simultaneous spatio-temporal calibration with B-spline continuous-time trajectories
- [Tsai & Lenz (1989)](https://ieeexplore.ieee.org/document/34770): hand-eye — the original AX=XB
- [Koide et al. (2023)](https://arxiv.org/abs/2302.05094): camera-LiDAR targetless — automatic calibration based on NID + SuperGlue
- [Li & Mourikis (2014)](https://journals.sagepub.com/doi/abs/10.1177/0278364913515286): temporal — online estimation of the time offset
- [OpenCalib (2023)](https://arxiv.org/abs/2205.14087): an integrated calibration framework for autonomous driving
- [GRIL-Calib (Kim et al., 2024)](https://arxiv.org/abs/2312.14035): targetless IMU-LiDAR calibration in ground-robot settings using planar-motion constraints. 6-DoF estimation is possible even from constrained motion.
- [MFCalib (2024)](https://arxiv.org/abs/2409.00992): single-shot targetless LiDAR-camera calibration that leverages multi-feature edges (depth continuity/discontinuity, intensity discontinuity). The edge inflation problem is solved with a LiDAR beam model.
- [iKalibr (Chen et al., 2024)](https://arxiv.org/abs/2407.11420): temporal — unified spatio-temporal calibration of heterogeneous multi-sensors (LiDAR, camera, IMU, radar) based on B-spline continuous time (IEEE T-RO 2025).
