# Chapter 3. Calibration

The observation models introduced in Chapter 2 (→ Chapter 2) describe *how* each sensor relates physical quantities to raw measurements, yet applying those equations to real hardware requires exact numerical parameters. Before a filter or optimizer can fuse incoming streams, it must know each camera's focal length and lens distortion, the rigid transformation separating a LiDAR from an IMU, and the clock offset between asynchronous timestamps. Calibration is the discipline of estimating these geometric and temporal parameters accurately.

> Calibration error introduces systematic bias into sensor fusion. Calibration spans camera intrinsics, multi-sensor extrinsics, and time synchronization.

Consequently, calibration is one of the first sanity checks in any sensor-fusion pipeline. Even small errors compound rapidly: an inaccurate sensor model or misaligned extrinsic pose injects persistent bias that can destabilize a filter or optimizer. In LiDAR-camera fusion, for instance, a single degree of angular misalignment produces a small-angle lateral offset of roughly 87 cm at 50 m—more than enough to project points from a pedestrian onto the background road. Preventing such failures requires both solid geometric foundations and disciplined practical workflows.

---

## 3.1 Camera Intrinsic Parameters

Camera intrinsic calibration estimates the internal optical parameters that govern how a 3D point in the camera's local frame maps onto a 2D pixel location—namely the focal length, the principal point, and the lens distortion coefficients.

### 3.1.1 Pinhole Camera Model (Review)

Before modeling the optical imperfections of real glass lenses, we establish the ideal straight-line projection baseline. A 3D world point $\mathbf{P}_w = [X, Y, Z]^\top$ is first transformed into the camera coordinate frame as $\mathbf{P}_c = [X_c, Y_c, Z_c]^\top$, and then projected onto the image plane by the pinhole model:

$$
s \begin{bmatrix} u \\ v \\ 1 \end{bmatrix}
= \mathbf{K} \begin{bmatrix} X_c \\ Y_c \\ Z_c \end{bmatrix}
= \begin{bmatrix} f_x & \gamma & c_x \\ 0 & f_y & c_y \\ 0 & 0 & 1 \end{bmatrix} \begin{bmatrix} X_c \\ Y_c \\ Z_c \end{bmatrix}
$$

In this projection equation, each entry carries a concrete physical meaning:
- $f_x, f_y$: focal length expressed in pixel units, where $f_x = f / p_x$ relates the physical focal length $f$ (mm) to the pixel pitch $p_x$ (mm/pixel)
- $(c_x, c_y)$: principal point where the optical axis intersects the image sensor plane
- $\gamma$: skew coefficient between the pixel axes, which is nearly 0 on modern digital sensors
- $s = Z_c$: projective scale factor equal to the point's depth along the optical axis

The upper-triangular matrix $\mathbf{K}$ is known as the **camera intrinsic matrix** or **calibration matrix**. While $\mathbf{K}$ theoretically has 5 degrees of freedom ($f_x, f_y, c_x, c_y, \gamma$), practitioners almost always fix $\gamma$ to 0 on square-pixel sensors, leaving 4 parameters to estimate.

### 3.1.2 Lens Distortion Model

Real camera lenses bend light rather than passing rays straight through an infinitesimal pinhole. Most standard-FoV lenses exhibit two dominant departures from ideal perspective projection: **radial distortion**, caused by lens curvature bending rays inward or outward toward the edges of the frame, and **tangential distortion**, caused when lens elements are not mounted perfectly parallel to the imaging sensor.

To model these optical effects independently of sensor resolution or focal length, we first strip away $\mathbf{K}$ and work in **normalized coordinates** on the unit depth plane ($Z_c = 1$):

$$
x = X_c / Z_c, \quad y = Y_c / Z_c, \quad r^2 = x^2 + y^2
$$

Because radial distortion scales symmetrically with distance $r$ from the optical center, it is modeled as an even-power polynomial multiplier:

**Radial distortion**:
$$
x_{\text{radial}} = x(1 + k_1 r^2 + k_2 r^4 + k_3 r^6)
$$
$$
y_{\text{radial}} = y(1 + k_1 r^2 + k_2 r^4 + k_3 r^6)
$$

A negative first coefficient ($k_1 < 0$) bends straight lines outward near the center—known as barrel distortion—whereas $k_1 > 0$ pinches them inward into pincushion distortion. For standard lenses, $k_1$ and $k_2$ alone capture almost all radial bending.

Slight decentering or tilt between the lens assembly and the image sensor introduces an asymmetric tangential shift as well:

**Tangential distortion**:
$$
x_{\text{tangential}} = 2p_1 xy + p_2(r^2 + 2x^2)
$$
$$
y_{\text{tangential}} = p_1(r^2 + 2y^2) + 2p_2 xy
$$

Adding both distortion components yields the distorted normalized coordinates, which are then scaled and shifted by the focal lengths and principal point to produce pixel coordinates:

**Final distorted coordinates**:
$$
x_d = x_{\text{radial}} + x_{\text{tangential}}, \quad y_d = y_{\text{radial}} + y_{\text{tangential}}
$$

$$
u = f_x \cdot x_d + c_x, \quad v = f_y \cdot y_d + c_y
$$

Collecting these five coefficients gives the standard distortion vector $\mathbf{d} = [k_1, k_2, p_1, p_2, k_3]$, which matches the parameter ordering expected by OpenCV's `calibrateCamera()`.

### 3.1.3 Zhang's Method: Homography-Based Calibration

Before [Zhang (2000)](https://ieeexplore.ieee.org/document/888718), accurate intrinsic calibration typically demanded a precisely machined 3D calibration rig. Zhang showed that observing a simple planar pattern (such as a printed checkerboard) across multiple poses suffices, because a plane collapses 3D-to-2D projection into a tractable 2D homography. OpenCV's `calibrateCamera()` follows this two-stage strategy: a closed-form linear estimate from planar homographies, followed by nonlinear bundle adjustment over the 2D–3D correspondences.

Defining the target plane as $Z = 0$ turns each image into a linear algebraic constraint on $\mathbf{K}$, unfolding over six steps.

#### Step 1: Homography Extraction

Because every checkerboard corner lies on the $Z = 0$ plane, its world coordinate $\mathbf{M} = [X, Y, 0]^\top$ projects to the image coordinate $\mathbf{m} = [u, v]^\top$ (in homogeneous form) as follows:

$$
s \tilde{\mathbf{m}} = \mathbf{K} [\mathbf{r}_1 \quad \mathbf{r}_2 \quad \mathbf{r}_3 \quad \mathbf{t}] \begin{bmatrix} X \\ Y \\ 0 \\ 1 \end{bmatrix}
= \mathbf{K} [\mathbf{r}_1 \quad \mathbf{r}_2 \quad \mathbf{t}] \begin{bmatrix} X \\ Y \\ 1 \end{bmatrix}
$$

Notice that $Z = 0$ causes the third rotation column $\mathbf{r}_3$ to drop out completely, reducing the projection to a $3 \times 3$ planar homography $\mathbf{H}$:

$$
s \tilde{\mathbf{m}} = \mathbf{H} \tilde{\mathbf{M}}, \quad \mathbf{H} = \mathbf{K} [\mathbf{r}_1 \quad \mathbf{r}_2 \quad \mathbf{t}]
$$

Up to scale, $\mathbf{H}$ has 8 degrees of freedom, so 4 non-collinear point correspondences per image are enough to solve for it via the Direct Linear Transform (DLT). In practice, a checkerboard provides dozens of corners, allowing us to solve an over-determined system with SVD while filtering out misdetections via RANSAC.

**Homography estimation via DLT**: Taking the cross product $\tilde{\mathbf{m}}_j \times \mathbf{H} \tilde{\mathbf{M}}_j = \mathbf{0}$ eliminates the unknown scale factor $s$ for each correspondence $(\mathbf{M}_j, \mathbf{m}_j)$ and yields two linear equations:

$$
\begin{bmatrix}
\tilde{\mathbf{M}}_j^\top & \mathbf{0}^\top & -u_j \tilde{\mathbf{M}}_j^\top \\
\mathbf{0}^\top & \tilde{\mathbf{M}}_j^\top & -v_j \tilde{\mathbf{M}}_j^\top
\end{bmatrix}
\mathbf{h} = \mathbf{0}
$$

Here $\mathbf{h}$ stacks the 9 entries of $\mathbf{H}$ into a vector. Combining $n$ corners produces a $2n \times 9$ matrix $\mathbf{A}$, and the vector $\mathbf{h}$ minimizing $\|\mathbf{A}\mathbf{h}\|$ is simply the right singular vector associated with the smallest singular value of $\mathbf{A}$.

#### Step 2: Deriving Constraints on the Intrinsic Parameters

Once $\mathbf{H}$ is known for a view, we can expose the intrinsic matrix $\mathbf{K}$ by splitting $\mathbf{H} = [\mathbf{h}_1 \quad \mathbf{h}_2 \quad \mathbf{h}_3]$ column by column:

$$
[\mathbf{h}_1 \quad \mathbf{h}_2 \quad \mathbf{h}_3] = \lambda \mathbf{K} [\mathbf{r}_1 \quad \mathbf{r}_2 \quad \mathbf{t}]
$$

Here $\lambda$ accounts for the arbitrary scale of the homography. Because $\mathbf{r}_1$ and $\mathbf{r}_2$ are the first two columns of an orthonormal rotation matrix, they must be perpendicular and share unit length—yielding two geometric constraints per image:

1. **Orthogonality condition**: $\mathbf{r}_1^\top \mathbf{r}_2 = 0$
   $$\mathbf{h}_1^\top \mathbf{K}^{-\top} \mathbf{K}^{-1} \mathbf{h}_2 = 0$$

2. **Equal-length condition**: $\|\mathbf{r}_1\| = \|\mathbf{r}_2\|$
   $$\mathbf{h}_1^\top \mathbf{K}^{-\top} \mathbf{K}^{-1} \mathbf{h}_1 = \mathbf{h}_2^\top \mathbf{K}^{-\top} \mathbf{K}^{-1} \mathbf{h}_2$$

Notice that the unknown matrix combination $\mathbf{B} = \mathbf{K}^{-\top} \mathbf{K}^{-1}$ appears identically in both constraints. Because $\mathbf{B}$ is symmetric and positive-definite, it is fully specified by just 6 independent elements:

$$
\mathbf{B} = \begin{bmatrix} B_{11} & B_{12} & B_{13} \\ B_{12} & B_{22} & B_{23} \\ B_{13} & B_{23} & B_{33} \end{bmatrix}
$$

Flattening those upper-triangular entries gives the 6-element unknown vector $\mathbf{b} = [B_{11}, B_{12}, B_{22}, B_{13}, B_{23}, B_{33}]^\top$.

#### Step 3: Constructing the Linear System

By expanding the quadratic form $\mathbf{h}_i^\top \mathbf{B} \mathbf{h}_j$, we can rewrite each constraint cleanly as a linear dot product with $\mathbf{b}$:

$$
\mathbf{h}_i^\top \mathbf{B} \mathbf{h}_j = \mathbf{v}_{ij}^\top \mathbf{b}
$$

Here the known coefficient vector $\mathbf{v}_{ij}$ depends solely on the entries of the estimated homography columns:
$$
\mathbf{v}_{ij} = \begin{bmatrix} h_{1i}h_{1j} \\ h_{1i}h_{2j} + h_{2i}h_{1j} \\ h_{2i}h_{2j} \\ h_{3i}h_{1j} + h_{1i}h_{3j} \\ h_{3i}h_{2j} + h_{2i}h_{3j} \\ h_{3i}h_{3j} \end{bmatrix}
$$

Substituting this form into the orthogonality and equal-length equations yields two linear equations in $\mathbf{b}$ for every checkerboard image:

$$
\begin{bmatrix} \mathbf{v}_{12}^\top \\ (\mathbf{v}_{11} - \mathbf{v}_{22})^\top \end{bmatrix} \mathbf{b} = \mathbf{0}
$$

Stacking $n$ images builds a $2n \times 6$ homogeneous linear system, which is again solved via SVD.

**Minimum number of images**: When skew is zero ($\gamma = 0$, adding the constraint $B_{12} = 0$), $\mathbf{b}$ drops to five entries with four degrees of freedom up to scale, so two nondegenerate views suffice algebraically. Estimating the full 5-parameter intrinsic matrix requires at least three distinct planar poses. In practice, however, algebraic minimums yield fragile estimates; you should capture a rich set of views spanning different positions, distances, and tilt angles so the system is well conditioned.

#### Step 4: Recovering K

From the solved vector $\mathbf{b}$, we reconstruct $\mathbf{B} = \mathbf{K}^{-\top} \mathbf{K}^{-1}$ and extract $\mathbf{K}$ in closed form (equivalent to taking the Cholesky factorization $\mathbf{B} = \mathbf{L}\mathbf{L}^\top$ with $\mathbf{K}^{-1} = \mathbf{L}^\top$). Expressed directly in terms of the elements of $\mathbf{B}$, the intrinsic parameters are:

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

With $\mathbf{K}$ recovered, the pose of the checkerboard in each image $i$ follows immediately by inverting $\mathbf{K}$ on the columns of that image's homography:

$$
\mathbf{r}_1 = \lambda \mathbf{K}^{-1} \mathbf{h}_1, \quad \mathbf{r}_2 = \lambda \mathbf{K}^{-1} \mathbf{h}_2, \quad \mathbf{t} = \lambda \mathbf{K}^{-1} \mathbf{h}_3
$$

Here $\lambda = 1 / \|\mathbf{K}^{-1} \mathbf{h}_1\|$ normalizes the first column to unit length, and the third axis is recovered via the cross product $\mathbf{r}_3 = \mathbf{r}_1 \times \mathbf{r}_2$. Because measurement noise prevents $[\mathbf{r}_1, \mathbf{r}_2, \mathbf{r}_3]$ from being strictly orthonormal, it is projected onto the closest valid $SO(3)$ rotation matrix via SVD: $\mathbf{R} = \mathbf{U}\mathbf{V}^\top$ (enforcing $\det(\mathbf{R}) = 1$).

#### Step 6: Nonlinear Optimization (MLE)

Although this closed-form solution is fast and gives a strong initialization, it ignores lens distortion and minimizes algebraic error rather than physical pixel error. To obtain a statistically optimal maximum-likelihood estimate, the final stage refines all parameters jointly—intrinsics, distortion coefficients, and per-image poses—by minimizing the sum of squared **reprojection errors** via the Levenberg-Marquardt (LM) algorithm:

$$
\min_{\mathbf{K}, \mathbf{d}, \{\mathbf{R}_i, \mathbf{t}_i\}} \sum_{i=1}^{n}\sum_{j=1}^{m} \|\mathbf{m}_{ij} - \hat{\mathbf{m}}(\mathbf{K}, \mathbf{d}, \mathbf{R}_i, \mathbf{t}_i, \mathbf{M}_j)\|^2
$$

Here $n$ is the number of images, $m$ is the number of corners per image, and $\hat{\mathbf{m}}(\cdot)$ denotes the full nonlinear projection including radial and tangential distortion. OpenCV's `calibrateCamera()` automates this entire pipeline—from DLT homographies and closed-form $\mathbf{K}$ recovery through LM nonlinear refinement—using analytical Jacobians of the projection function.

### 3.1.4 Calibration Code with OpenCV

The script below implements Zhang's workflow end to end in OpenCV: it detects checkerboard corners across a folder of images, refines them to sub-pixel accuracy, runs `cv2.calibrateCamera()`, and reports per-image reprojection residuals so you can spot problematic frames right away.

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
    residual = img_points[i].reshape(-1, 2) - img_points_proj.reshape(-1, 2)
    error = np.sqrt(np.mean(np.sum(residual ** 2, axis=1)))
    errors.append(error)

print(f"Per-image mean error: {np.mean(errors):.4f} pixels")
print(f"Max error image: {np.argmax(errors)} ({max(errors):.4f} px)")
```

Running the solver is only half the battle; the accuracy of the resulting intrinsics depends overwhelmingly on how well the input images constrain the optimization.

### 3.1.5 Practical Tips: Conditions for Good Calibration

Rather than chasing an arbitrary quota of image counts or board angles, focus on geometric observability when collecting calibration data:

**Pose Diversity**: Capture the checkerboard across a rich variety of positions and orientations so every intrinsic parameter becomes well constrained:
- Move the target into the top, bottom, left, right, and center of the frame—border coverage is especially critical for pinning down radial distortion
- Avoid repeating nearly fronto-parallel views; tilting the board about multiple axes breaks the correlation between focal length and camera-to-board depth
- Vary the distance between the board and the camera to span multiple scales
- Let pose diversity, numerical conditioning, and the spatial distribution of residuals tell you when you have enough data, rather than relying on a fixed image count

**Corner Accuracy**: Whenever a detector returns integer-pixel corners, refine them with a sub-pixel step such as `cv2.cornerSubPix()`. Visually inspect and discard frames with failed detections, motion blur, or specular glare across the target.

**Illumination Conditions**: Uniform, diffuse lighting keeps corner contrast consistent across the board. Harsh shadows or glossy reflections shift detected corners, so print the target on matte material and inspect exposure levels before capturing a full sequence.

**Interpreting Reprojection Error**: No single pixel threshold guarantees a good calibration, because expected error scales with image resolution, lens model, target size, and corner-detector noise. Evaluate the global RMS alongside per-image residuals, directional error patterns across the image, straightness of known physical lines after undistortion, and reprojection error on held-out validation frames.

**Detecting Outlier Images**: Use the `errors` array from the script above to spot high-residual frames, but resist discarding them blindly. Inspect each flagged image for misordered corners, motion blur, glare, or board warping, and verify whether removing it actually improves parameter stability and held-out error.

**Warning — accuracy of the square size**: `SQUARE_SIZE` must match the physical dimensions of the printed squares exactly. Office printers frequently apply silent scaling, so always verify the printed grid with a ruler or calipers. An inaccurate square size corrupts `tvecs` (the metric translation scale), even though `K` and the distortion coefficients remain unaffected because distortion operates in normalized coordinates.

### 3.1.6 Fisheye / Omnidirectional Calibration

Standard radial-tangential models break down on wide-angle and fisheye lenses. As the incidence angle approaches $90^\circ$, normalized perspective coordinates ($x = X_c / Z_c$) blow up toward infinity, and even high-order polynomials in $r^2$ struggle to capture extreme ray bending near the image perimeter. There is no rigid field-of-view cutoff for switching models; let the lens optics and the spatial pattern of your reprojection residuals guide the choice toward an equidistant or Kannala–Brandt formulation.

**Equidistant Projection Model**:

To keep wide-angle projection smooth and bounded even past a $180^\circ$ field of view, [Kannala & Brandt (2006)](https://ieeexplore.ieee.org/document/1642666) parameterize the image radius $r$ directly as an odd-power polynomial of the incidence angle $\theta$:

$$
r(\theta) = k_1 \theta + k_2 \theta^3 + k_3 \theta^5 + k_4 \theta^7 + k_5 \theta^9
$$

Here $\theta = \arctan\left(\sqrt{x^2 + y^2}\right)$ is the angle between the incoming ray and the optical axis, and $x, y$ are normalized coordinates. Notice that the first term $r = f\theta$ describes an ideal equidistant projection, while the higher-order terms absorb the specific lens's deviation from it.

OpenCV implements a four-parameter version of this model inside the `cv2.fisheye` module:

$$
\theta_d = \theta(1 + k_1\theta^2 + k_2\theta^4 + k_3\theta^6 + k_4\theta^8)
$$

Calling `cv2.fisheye.calibrate()` mirrors the standard pipeline above, though it expects slightly different array shapes and benefits from explicit conditioning checks:

```python
# Fisheye calibration (OpenCV fisheye module)
import cv2
import numpy as np

calibration_flags = (
    cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC
    + cv2.fisheye.CALIB_CHECK_COND
    + cv2.fisheye.CALIB_FIX_SKEW
)
# CALIB_CHECK_COND aborts with an exception if even one image is badly conditioned.
# In that case, drop the offending image or remove this flag.

K_fisheye = np.zeros((3, 3))
D_fisheye = np.zeros((4, 1))  # k1, k2, k3, k4

# The fisheye module requires object points as (N, 1, 3) and image points as (N, 1, 2) (convert the (N, 3) object points from 3.1.4)
obj_points_f = [o.reshape(-1, 1, 3).astype(np.float64) for o in obj_points]
img_points_f = [p.reshape(-1, 1, 2).astype(np.float64) for p in img_points]

ret, K_fisheye, D_fisheye, rvecs, tvecs = cv2.fisheye.calibrate(
    obj_points_f,
    img_points_f,
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

For catadioptric rigs (mirror plus lens) and ultra-wide fisheye optics exceeding a hemisphere, the [OCamCalib toolbox](https://sites.google.com/site/scarabotix/ocamcalib-omnidirectional-camera-calibration-toolbox-for-matlab) introduced by [Scaramuzza et al. (2006)](https://rpg.ifi.uzh.ch/docs/IROS06_scaramuzza.pdf) avoids assuming a specific physical lens geometry altogether. Instead, it fits the projection directly as a general polynomial mapping incidence angle to radial pixel offset:

$$
\begin{bmatrix} u \\ v \end{bmatrix} = \begin{bmatrix} x_c \\ y_c \end{bmatrix} + \mathbf{A} \cdot \rho(\theta) \begin{bmatrix} \cos(\phi) \\ \sin(\phi) \end{bmatrix}
$$

Here $\mathbf{A}$ is an affine matrix accounting for sensor stretch or non-square pixels, and $\rho(\theta)$ is the polynomial mapping incidence angle $\theta$ to radial distance on the sensor.

### 3.1.7 Camera Calibration with Kalibr

When a camera belongs to a larger multi-sensor rig—especially a visual-inertial system—calibrating its intrinsics inside the same toolbox used for extrinsics keeps conventions consistent. [Kalibr](https://github.com/ethz-asl/kalibr), developed at ETH Zurich, is a widely adopted open-source suite for camera-IMU calibration (Section 3.4) that supports multiple camera projection models under a unified batch-optimization framework. Whether Kalibr outperforms OpenCV on intrinsics alone depends on the chosen lens model, target coverage, initialization, and optimizer configuration.

The table below lists the primary projection-distortion combinations available in Kalibr:

| Model | # Parameters | Suitable Lens |
|------|-----------|-----------|
| `pinhole-radtan` | 4 + 4 | Standard lens (same as OpenCV) |
| `pinhole-equi` | 4 + 4 | Fisheye lens |
| `omni-radtan` | 5 + 4 | Ultra-wide / catadioptric |
| `ds` (Double Sphere) | 6 | Wide-angle lens (modern model) |
| `eucm` (Extended UCM) | 6 | Wide-angle lens |

**AprilGrid vs Checkerboard**: Rather than a plain checkerboard, Kalibr recommends an AprilGrid target. Because every AprilTag carries a unique binary ID, the detector can still recover corner identities and ordering automatically even when part of the board is occluded or clipped at the edge of the frame.

To calibrate a single camera's intrinsics from a recorded ROS bag, generate the target PDF and invoke `kalibr_calibrate_cameras`:

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

Under the hood, Kalibr runs a structured four-step pipeline:
1. Detect AprilGrid corners in every frame
2. Estimate initial per-view homographies from corner correspondences (mirroring Step 1 of Zhang's method)
3. Run nonlinear batch optimization over all camera and pose parameters jointly
4. Generate diagnostic plots summarizing the parameter estimates and residual distributions

Compared with OpenCV's standalone routine, Kalibr offers four notable strengths for robotics rigs:
- **B-spline trajectory representation**: models continuous-time motion so asynchronous camera and IMU streams align cleanly without resampling
- **Modern wide-angle models**: includes recent analytical models such as Double Sphere (`ds`) and Extended UCM (`eucm`)
- **Multi-camera rigs**: solves for the relative extrinsics of multiple cameras in a single optimization
- **Seamless IMU integration**: feeds calibrated camera chains directly into camera-IMU calibration (→ §3.4)

Once each camera's internal projection is accurately modeled, the next step for multi-camera systems is determining how those cameras sit relative to one another in 3D space.

---

## 3.2 Camera-Camera (Stereo) Extrinsic

Once each camera's internal optics are calibrated, a stereo rig requires one more geometric ingredient before it can triangulate depth: the rigid 3D transformation—rotation and baseline translation—linking the left and right camera frames.

### 3.2.1 Stereo Calibration

Given a left camera $C_L$ and a right camera $C_R$, any 3D point $\mathbf{P}$ observed by both views obeys the rigid transformation:

$$
\mathbf{P}_{C_R} = \mathbf{R} \cdot \mathbf{P}_{C_L} + \mathbf{t}
$$

Here $(\mathbf{R}, \mathbf{t})$ maps coordinates from the left camera frame into the right camera frame. Observing a shared checkerboard simultaneously with both cameras ties both sets of corner detections to the same physical 3D board, allowing the solver to lock down $(\mathbf{R}, \mathbf{t})$.

In OpenCV, `cv2.stereoCalibrate()` estimates this relative pose. When reliable single-camera intrinsics are already available, setting `cv2.CALIB_FIX_INTRINSIC` keeps them locked to prevent parameter drift during extrinsic optimization (though you can relax this flag for joint refinement if needed):

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

Knowing $(\mathbf{R}, \mathbf{t})$ is enough to triangulate a matched pixel pair, but finding matching pixels across unaligned 2D images is slow and error-prone. **Parallel rectification** solves this by warping both images onto a shared virtual plane so that all epipolar lines run strictly horizontally—collapsing stereo correspondence search into a fast 1D scan along the same pixel row.

**Epipolar Geometry Review**:

Any 3D point $\mathbf{P}$ together with the two camera projection centers $O_L, O_R$ defines an epipolar plane, and the intersections of that plane with the two image planes form the epipolar lines. Because $\mathbf{P}$ must lie on the ray through $\mathbf{m}_L$, its projection $\mathbf{m}_R$ in the right image is constrained to lie along the matching epipolar line.

Algebraically, this coplanarity constraint is captured by the **fundamental matrix** $\mathbf{F}$:

$$
\tilde{\mathbf{m}}_R^\top \mathbf{F} \tilde{\mathbf{m}}_L = 0
$$

Once the cameras are intrinsically calibrated, stripping away $\mathbf{K}$ via normalized coordinates $\hat{\mathbf{m}} = \mathbf{K}^{-1}\tilde{\mathbf{m}}$ expresses the same constraint through the **essential matrix** $\mathbf{E}$:

$$
\hat{\mathbf{m}}_R^\top \mathbf{E} \hat{\mathbf{m}}_L = 0, \quad \mathbf{E} = [\mathbf{t}]_\times \mathbf{R}
$$

Here $[\mathbf{t}]_\times$ denotes the skew-symmetric cross-product matrix of the translation vector.

**Rectification Procedure**:

Rectification computes virtual rotations and projection matrices that reproject both camera views onto a common image plane parallel to the stereo baseline vector, aligning their pixel rows. OpenCV's `cv2.stereoRectify()` implements Bouguet's algorithm and precomputes pixel lookup tables for fast remapping:

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

**Verification via the epipolar constraint**: Before running a stereo matcher, verify the rectification visually by drawing horizontal guide lines across the concatenated left and right rectified images; every physical feature in the scene should sit on the exact same row in both halves:

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

**From disparity to depth**: Once corresponding features are aligned horizontally, the horizontal pixel shift $d = u_L - u_R$—known as disparity—gives depth directly by similar triangles:

$$
Z = \frac{f \cdot B}{d}
$$

Here $f$ is the rectified focal length in pixels and $B = \|\mathbf{t}\|$ is the baseline distance. Using the reprojection matrix $Q$ returned by `cv2.stereoRectify()`, an entire disparity map converts directly into a 3D point cloud: first compute homogeneous coordinates $[X,Y,Z,W]^\top = Q \cdot [u,v,d,1]^\top$, then divide through by $W$ to get $\mathbf{P}_{3D} = [X/W,Y/W,Z/W]^\top$.

Stereo rigs align two sensors of the same modality; calibrating across different sensing modalities—such as projecting 3D LiDAR scans onto 2D camera images—introduces a fresh set of correspondence challenges.

---

## 3.3 Camera-LiDAR Extrinsic Calibration

Multi-modal fusion between a camera and a LiDAR depends critically on knowing the rigid transform $(\mathbf{R}, \mathbf{t})$ connecting their frames. Whether you are coloring a LiDAR point cloud from RGB imagery or assigning metric depth to visual features, even small extrinsic errors shift distant projections by tens of centimeters. Unlike stereo cameras, however, cameras capture dense 2D appearance without scale while LiDARs capture sparse 3D geometry without texture—so establishing correspondences requires either a shared physical target or statistical and geometric alignment in natural scenes.

### 3.3.1 Target-based Calibration

Target-based methods bridge the modality gap by placing a known geometric pattern—most often a planar checkerboard or AprilTag board—where both sensors can observe it simultaneously.

**Principle**: The camera resolves the board's printed corners as crisp 2D image points, while the LiDAR captures the flat board surface as a cluster of 3D points lying on a single plane. By extracting that plane and its boundary in the point cloud, we can align the two observations using either explicit 3D-2D point correspondences or geometric plane constraints.

**3D-2D correspondence-based method**:

1. Detect the checkerboard corners in the camera image to obtain 2D pixel points $\{\mathbf{m}_j\}$
2. Segment the checkerboard plane from the LiDAR point cloud via RANSAC to obtain the plane equation $\mathbf{n}^\top \mathbf{p} + d = 0$
3. Fit the rectangular board boundary to the inlier LiDAR points to infer the 3D corner coordinates $\{\mathbf{P}_j\}$
4. Solve for $(\mathbf{R}, \mathbf{t})$ from the resulting 3D-2D pairs $\{(\mathbf{P}_j, \mathbf{m}_j)\}$ using a Perspective-n-Point (PnP) solver

**Plane-constraint-based method**:

When LiDAR beams are too sparse to pinpoint board edges cleanly, you can calibrate using plane-to-plane geometry alone—no 3D corner reconstruction required. For each view, PnP recovers the target plane in camera coordinates from the 2D corners and known square size, and we constrain the LiDAR points on the board to fall onto that same plane once transformed by the unknown extrinsic pose.

Across $n$ distinct checkerboard poses, each observation contributes a linear plane constraint:

$$
\mathbf{n}_i^\top (\mathbf{R} \mathbf{p}_{L,i} + \mathbf{t}) + d_i = 0, \quad \forall i = 1, \ldots, n
$$

Here $\mathbf{n}_i$ is the board normal expressed in the camera frame and $\mathbf{p}_{L,i}$ is a LiDAR point lying on the board.

The function below implements this plane-constraint formulation in two closed-form linear stages: first aligning the paired plane normals via Kabsch/SVD to recover rotation, then solving a linear least-squares system over the plane offsets to recover translation:

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

    # Collect (camera-frame plane, LiDAR-frame plane) pairs from each observation. Planes are n·x + d = 0
    planes_cam, planes_lidar = [], []
    for corners_2d, (normal_l, d_l) in zip(img_corners_list, lidar_planes_list):
        # Checkerboard pose as seen from the camera (PnP)
        ret, rvec, tvec = cv2.solvePnP(board_corners_3d, corners_2d, K, dist)
        R_cam_board, _ = cv2.Rodrigues(rvec)
        # Checkerboard plane in the camera frame: normal is the board z-axis, distance from the board origin
        n_c = R_cam_board[:, 2]
        d_c = -n_c @ tvec.ravel()
        planes_cam.append((n_c, d_c))
        planes_lidar.append((normal_l, d_l))

    # Rotation: align normal pairs (Kabsch/SVD), n_c = R n_l
    N_c = np.stack([n for n, _ in planes_cam])
    N_l = np.stack([n for n, _ in planes_lidar])
    U, _, Vt = np.linalg.svd(N_c.T @ N_l)
    R_cam_lidar = U @ np.diag([1, 1, np.sign(np.linalg.det(U @ Vt))]) @ Vt

    # Translation: plane-distance constraint n_c · t = d_l - d_c by least squares (needs at least 3 planes)
    b = np.array([d_l - d_c for (_, d_c), (_, d_l) in zip(planes_cam, planes_lidar)])
    t_cam_lidar, *_ = np.linalg.lstsq(N_c, b, rcond=None)

    # The final result is refined by nonlinear optimization
    return R_cam_lidar, t_cam_lidar
```

**Practical considerations**:
- Size the checkerboard according to your working range, angular resolution, and scan pattern so every pose lands enough LiDAR returns for a stable plane fit
- Vary both distance and board orientation across views; judge dataset sufficiency by normal-vector diversity and matrix conditioning rather than a rigid frame quota
- With sparse multi-beam LiDARs, inspect actual ring hits on the board beforehand and adjust target size and placement so multiple scan lines intersect it cleanly

### 3.3.2 Targetless Calibration

Setting up and waving a large calibration board is often impractical—especially on deployed vehicles or large outdoor rigs. Targetless calibration avoids physical targets altogether by maximizing geometric or statistical consistency directly in natural scenes.

#### Mutual Information (MI) Based Method

**Intuition**: When LiDAR points are projected into a camera image using the true extrinsic transform, surface reflectivity and visual intensity align across object boundaries—maximizing the statistical dependence between the two signals. Any misalignment scrambles those pairings and reduces shared information.

**Definition of Mutual Information**:

Formally, the mutual information between two random variables $X$ (LiDAR reflectivity or depth) and $Y$ (image intensity) measures how much knowing one reduces uncertainty about the other:

$$
\text{MI}(X; Y) = \sum_{x}\sum_{y} p(x, y) \log \frac{p(x, y)}{p(x)p(y)}
$$

Written in terms of marginal and joint entropies, the same relationship becomes:

$$
\text{MI}(X; Y) = H(X) + H(Y) - H(X, Y)
$$

When $X$ and $Y$ are statistically independent, $\text{MI} = 0$. Conversely, when one variable is a deterministic function of the other, their mutual information is bounded by the smaller marginal entropy, reaching $\text{MI} = H(X) = H(Y)$ for an invertible one-to-one mapping.

**Normalized Information Distance (NID)**:

Because raw MI varies with scene entropy, [Koide et al. (2023)](https://arxiv.org/abs/2302.05094) minimize its normalized counterpart, NID, as the calibration objective:

$$
\text{NID}(X; Y) = 1 - \frac{\text{MI}(X; Y)}{H(X, Y)} = \frac{H(X, Y) - \text{MI}(X; Y)}{H(X, Y)}
$$

Bounded cleanly to $[0, 1]$, NID decreases toward zero as the LiDAR and camera observations register more tightly.

**Histogram-based MI computation**:

Given a candidate extrinsic projection, NID is evaluated empirically from the 2D joint intensity histogram of projected LiDAR returns and sampled image pixels:

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

Because discrete histogram binning lacks analytical gradients, MI-based optimization typically relies on derivative-free solvers such as Nelder-Mead or finite-difference gradients, as demonstrated for LiDAR-camera calibration by [Pandey et al. (2015)](https://arxiv.org/abs/1507.07595).

#### Edge Alignment Based Method

**Intuition**: Object silhouettes produce sharp depth discontinuities in LiDAR range images and strong intensity edges in camera images. Aligning those two edge maps directly guides $(\mathbf{R}, \mathbf{t})$ without relying on reflectivity correlation.

Let $\mathbf{e}_L$ denote edges extracted from a LiDAR depth image and $\mathbf{e}_C$ denote edges extracted from the camera image; calibration minimizes the projected distance onto the nearest visual edge:

$$
\min_{\mathbf{R}, \mathbf{t}} \sum_i \text{dist}(\pi(\mathbf{R}\mathbf{p}_{L,i} + \mathbf{t}), \mathbf{e}_C)
$$

Here $\pi(\cdot)$ is the camera projection function and $\text{dist}(\cdot, \mathbf{e}_C)$ is precomputed efficiently via an image distance transform.

#### Learning-based Methods

Deep-learning approaches such as RegNet and CalibNet take a LiDAR depth projection and an RGB image as input and directly regress a 6-DoF extrinsic correction. Because their valid operating envelope depends heavily on the sensor geometry, scene types, and perturbation ranges seen during training, treat them carefully on new hardware—validate them on representative data and, where possible, use their prediction to initialize geometric refinement rather than trusting it blindly.

### 3.3.3 Koide et al. (2023) — An Open Targetless Calibration Implementation

A prominent public implementation of targetless calibration is `direct_visual_lidar_calibration` by [Koide et al. (2023)](https://arxiv.org/abs/2302.05094). Rather than relying on a single brittle optimization step, it pairs dense point-cloud accumulation with learned cross-modal initialization and NID fine alignment across three stages:

**Stage 1: LiDAR point cloud densification**

A single sweep from a spinning LiDAR (such as Ouster or Velodyne) is too sparse for reliable MI registration. The pipeline therefore uses Continuous-Time ICP (CT-ICP) to register and accumulate several seconds of scans into a crisp, dense local point cloud—whereas non-repetitive solid-state LiDARs (such as Livox) naturally densify over short static intervals.

**Stage 2: SuperGlue-based initial estimation**

Next, the dense point cloud is rendered from virtual camera viewpoints as synthetic LiDAR intensity images. SuperGlue—a learned feature matcher—establishes 2D-2D correspondences between these rendered intensity views and the real camera frames, which lift immediately into 2D-3D pairs for an initial RANSAC + PnP pose estimate. Note that although the paper reports an initialization success rate above 80% on its benchmark datasets and criteria, cross-modal matching success varies with scene texture and intensity quality on new rigs.

**Stage 3: NID-based fine registration**

Initialized from Stage 2, Nelder-Mead optimization minimizes NID globally across the images, while view-based hidden point removal culls occluded background LiDAR returns that would otherwise pollute the joint histogram.

**Reported result**: Across their benchmark combinations of spinning and solid-state LiDARs paired with pinhole, fisheye, and omnidirectional cameras, the authors report a mean translation error of 0.043 m and a mean rotation error of 0.374 degrees.

### 3.3.4 Practical Tool Comparison

Choosing a calibration workflow involves trading off setup effort against automation and observability. The table below summarizes the main public tools and online strategies:

| Tool | Approach | Target needed | Operating mode | Check before use |
|------|----------|---------------|----------------|------------------|
| Autoware Calibration Toolkit | Target-based | Yes | Offline workflow with operator steps | Supported sensors and target; initialization |
| `direct_visual_lidar_calibration` (Koide) | Targetless (NID) | No | Automated offline pipeline | Scene overlap, intensity rendering, initialization success |
| ACSC (Automatic Calibration) | Target-based + automatic corner extraction | Yes | Automated target detection | Required target and supported sensor model |
| Online extrinsic estimation (VIO/LIO state augmentation) | Online | No | Estimation during operation | Motion observability and drift-monitoring criterion |

Recent work continues to sharpen edge-based targetless alignment: [MFCalib (2024)](https://arxiv.org/abs/2409.00992) combines depth-continuity, depth-discontinuity, and intensity-discontinuity edges while explicitly modeling LiDAR beam geometry to suppress edge inflation, reporting lower errors than baseline methods under its test conditions.

In long-lived deployments, target-based and targetless tools work best together. Koide et al. (2023) suggest establishing a high-confidence target-based baseline in the lab and using targetless NID during operation to watch for drift—triggering recalibration only once validated thresholds link NID degradation to downstream task error.

When we replace a LiDAR with a high-rate IMU, spatial calibration alone is no longer enough: because camera frames and inertial samples originate on separate clocks, time alignment enters the estimation problem directly.

---

## 3.4 Camera-IMU Extrinsic + Temporal Calibration

Pairing a camera with an IMU introduces a new wrinkle: you must estimate not only the rigid 6-DoF spatial transform between the two sensors, but also the constant time offset separating their clocks. While Visual-Inertial Odometry (VIO) frameworks such as VINS-Mono and OpenVINS can refine both parameters online during operation, offline tools like Kalibr establish the accurate initial estimates needed for reliable convergence.

### 3.4.1 Why Time Offset Matters

Cameras and IMUs normally run on independent hardware clocks. Whenever a constant offset $t_d$ separates them, the inertial measurements corresponding to a camera frame stamped at $t_c$ actually occurred at physical time $t_c + t_d$.

As shown in the experiments of Furgale et al. (2013), camera-IMU offsets commonly span several to tens of milliseconds. During aggressive motion, ignoring even a modest offset severely inflates reprojection errors—for instance, a 10 ms clock mismatch during a 100 deg/s rotation introduces a full one-degree orientation error between the image and the inertial trajectory.

### 3.4.2 Kalibr: Continuous-Time B-Spline-Based Calibration

Proposed by [Furgale et al. (2013)](https://ieeexplore.ieee.org/document/6696514), Kalibr has become the standard calibration tool across VIO pipelines such as VINS-Mono, OpenVINS, and MSCKF.

Rather than representing motion as a sequence of discrete poses—which struggles to align 20–30 Hz camera frames against 200–1000 Hz asynchronous IMU packets—Kalibr parameterizes the rig's trajectory as a continuous-time $SE(3)$ B-spline. That single representation lets us query pose at any arbitrary shifted timestamp and compute exact analytical derivatives for acceleration and angular velocity.

**B-Spline trajectory representation**:

In a cubic B-spline, the rig pose $\mathbf{T}(t) \in SE(3)$ at any continuous time $t$ is interpolated smoothly from four neighboring control points $\{\mathbf{T}_i\}$:

$$
\mathbf{T}(t) = \mathbf{T}_i \prod_{j=1}^{3} \text{Exp}(\mathbf{B}_j(u) \cdot \Omega_{i+j})
$$

Here each symbol plays a specific geometric role:
- $u = (t - t_i) / \Delta t$ is the normalized segment time ($0 \leq u < 1$)
- $\mathbf{B}_j(u)$ are the cubic B-spline basis weights
- $\Omega_{i+j} = \text{Log}(\mathbf{T}_{i+j-1}^{-1} \mathbf{T}_{i+j})$ is the Lie algebra increment between consecutive control points
- $\text{Exp}, \text{Log}$ are the exponential and logarithm maps on $SE(3)$

Representing the trajectory this way gives Kalibr three decisive advantages:
1. **Analytical differentiability**: velocity and acceleration follow directly in closed form at any instant, tying straight into the IMU measurement model
2. **Asynchronous sensor fusion**: every sensor reading is evaluated at its native timestamp—shifted by $t_d$—without interpolation or resampling hacks
3. **Local support**: each timestamp touches only four adjacent control points, keeping the optimization Jacobian sparse and computationally tractable

**Observation model**:

Because $\mathbf{T}(t)$ tracks the IMU body frame relative to the world, a camera landmark observation at timestamp $t_{c,k}$ simply evaluates the spline at the corrected time $t_c + t_d$ (specifically $t_{c,k} + t_d$) and applies the unknown camera-to-IMU extrinsic $\mathbf{T}_{CB}$:

$$
\mathbf{e}_{\text{cam},k} = \mathbf{m}_k - \pi\left(\mathbf{T}_{CB} \cdot \mathbf{T}(t_{c,k} + t_d)^{-1} \cdot \mathbf{p}_w\right)
$$

Here $\mathbf{T}(t)$ maps the IMU (body) frame to the world, and $\mathbf{T}_{CB}$ maps the IMU (body) frame to the camera—the spatial extrinsic we wish to recover.

Meanwhile, IMU residuals compare raw accelerometer and gyroscope readings against the spline's second and first time derivatives at $t_{\text{imu}}$, rotated into the body frame and offset by sensor biases:

$$
\mathbf{e}_{\text{accel},k} = \mathbf{a}_k - \left[\mathbf{R}(t_k)^\top(\ddot{\mathbf{p}}(t_k) - \mathbf{g}) + \mathbf{b}_a\right]
$$
$$
\mathbf{e}_{\text{gyro},k} = \boldsymbol{\omega}_k - \left[\boldsymbol{\omega}(t_k) + \mathbf{b}_g\right]
$$

**Optimization problem**:

Combining all visual and inertial residuals weighted by their respective noise covariances yields a joint nonlinear least-squares objective:

$$
\min_{\mathbf{T}_{CB}, t_d, \mathbf{b}_a, \mathbf{b}_g, \{\mathbf{T}_i\}} \sum_k \|\mathbf{e}_{\text{cam},k}\|^2_{\Sigma_c} + \sum_k \left(\|\mathbf{e}_{\text{accel},k}\|^2_{\Sigma_a} + \|\mathbf{e}_{\text{gyro},k}\|^2_{\Sigma_g}\right)
$$

Gauss-Newton or Levenberg-Marquardt solves this problem efficiently because the local support of the B-spline keeps the underlying Jacobian block-sparse even over long calibration sequences.

### 3.4.3 Kalibr Execution Guide

Running a full camera-IMU calibration in Kalibr takes three steps: printing an AprilGrid target, recording a dynamic ROS bag with rich 6-DoF motion, and invoking `kalibr_calibrate_imu_camera`:

```bash
# 1. Prepare the AprilGrid target
kalibr_create_target_pdf --type apriltag \
    --nx 6 --ny 6 --tsize 0.024 --tspace 0.3

# 2. Collect data (ROS bag)
#    - Place the target in the camera's view and move the sensor rig diversely
#    - Include both rotation and translation along all axes
#    - Record until all axes are excited, the target remains detectable, and residuals are well conditioned

# 3. Run the Camera-IMU calibration
kalibr_calibrate_imu_camera \
    --target april_6x6_24x24mm.yaml \
    --cam camchain.yaml \
    --imu imu.yaml \
    --bag calibration.bag \
    --bag-freq 20.0 \
    --timeoffset-padding 0.1
```

Kalibr also needs accurate noise parameters for the IMU in `imu.yaml` so it can weight visual and inertial residuals properly:

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

**Data collection**:
1. **Motion diversity**: Excite all 6-DoF motion axes thoroughly; rotations about all three axes are essential for making extrinsics observable
2. **Motion speed**: Move dynamically enough to reveal IMU scale and bias above noise, yet smoothly enough to avoid motion blur in the camera frames
3. **Target visibility**: Keep the AprilGrid consistently detectable while sweeping through multi-axis motions
4. **Start and end**: Begin and finish each sequence stationary so initial IMU biases settle cleanly

**Interpreting the results**: Always inspect four outputs from Kalibr before trusting the calibration:
- `T_cam_imu`: the estimated 4x4 camera-IMU extrinsic transformation matrix
- `timeshift_cam_imu`: the estimated time offset $t_d$ for this sequence—double-check its sign convention carefully before pasting it into a VIO config
- Reprojection-error distribution: check both temporal trends and spatial patterns across the image, not just the global mean
- Accelerometer/gyro residuals: verify that the inertial residuals stay consistent with the configured noise bounds

### 3.4.4 Allan Variance Measurement Hands-On

Whether you are configuring Kalibr or tuning a VIO filter, realistic IMU noise parameters make or break estimator weighting. Originally invented to characterize atomic-clock stability, **Allan variance** separates fast random sensor noise from slow bias wander by averaging a long stationary recording over clusters of progressively increasing duration $\tau$. Over short windows, high-frequency white noise dominates and averages down smoothly as $\tau^{-1/2}$; over longer windows, slow bias drift takes over and causes the curve to flatten and turn upward.

**Allan Variance Definition**:

For a stationary discrete time series $\{x_k\}$ sampled with period $\tau_0$, the Allan variance at cluster duration $\tau = n \cdot \tau_0$ measures half the mean-squared difference between adjacent cluster averages:

$$
\sigma^2(\tau) = \frac{1}{2(N-2n+1)} \sum_{k=1}^{N-2n+1} \left[\bar{x}_{k+n} - \bar{x}_k\right]^2
$$

Here $\bar{x}_k = \frac{1}{n}\sum_{j=0}^{n-1} x_{k+j}$ is the mean over a contiguous cluster of $n$ samples.

**Noise identification on a log-log plot**:

When you plot the Allan deviation $\sigma(\tau)$ against cluster time $\tau$ on a log-log graph, distinct physical noise mechanisms appear cleanly as straight-line segments with characteristic slopes:

| Noise type | Slope | $\sigma(\tau)$ |
|-----------|-------|---------------|
| Quantization noise | $-1$ | $\propto \tau^{-1}$ |
| **Angle/Velocity random walk** | $-1/2$ | $\propto \tau^{-1/2}$ |
| **Bias instability** | $0$ | constant (minimum) |
| Rate random walk | $+1/2$ | $\propto \tau^{+1/2}$ |
| Rate ramp | $+1$ | $\propto \tau$ |

**Hands-on code**:

The Python script below computes the Allan variance curve across log-spaced cluster sizes, automatically locates the $-1/2$ white-noise slope and flat bias-instability valley, and plots the classic log-log diagnostic curve:

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
    # Retain at least ten clusters at the largest tau.
    max_n = max(1, N // 10)

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

def extract_imu_noise_params(taus, allan_vars):
    """
    Extract IMU noise parameters from the Allan variance plot.
    """
    adev = np.sqrt(allan_vars)

    valid = (taus > 0) & (adev > 0) & np.isfinite(adev)
    taus, adev = taus[valid], adev[valid]
    slopes = np.gradient(np.log(adev), np.log(taus))

    # Fit the coefficient N of sigma(tau) = N / sqrt(tau) over a -1/2 region.
    white = np.abs(slopes + 0.5) < 0.1
    if not np.any(white):
        raise ValueError("No stable -1/2-slope region; collect or inspect more data")
    noise_coefficient = np.median(adev[white] * np.sqrt(taus[white]))

    # Report bias instability only if a near-flat region is present.
    flat = np.abs(slopes) < 0.1
    bias_instability = np.min(adev[flat]) / 0.664 if np.any(flat) else np.nan
    tau_min = taus[np.argmin(adev)]

    return {
        'noise_coefficient': noise_coefficient,
        'bias_instability': bias_instability,
        'tau_min': tau_min
    }

# Usage example
# 1. Record enough stationary data to retain many clusters at the largest tau of interest
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
params = extract_imu_noise_params(taus, avars)

print(f"White-noise coefficient: {params['noise_coefficient']:.6f}")
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
plt.loglog(tau_ref, params['noise_coefficient'] / np.sqrt(tau_ref),
           'r--', label='Random walk (-1/2)')
plt.axhline(y=params['bias_instability'], color='g', linestyle='--',
            label=f'Bias instability ({params["bias_instability"]:.1e})')
plt.legend()
plt.savefig('allan_deviation.png', dpi=150)
plt.show()
```

**Practical tips for Allan variance measurement**:
- **Stationary data collection**: Mount the IMU on a rigid, vibration-free surface, and record long enough that plenty of non-overlapping clusters remain at the largest $\tau$ of interest. If no bias-instability trough appears yet, extend the recording—though no fixed duration guarantees clean separation on every device.
- **Comparison with the data sheet**: Cross-check your extracted noise density against the manufacturer's datasheet specification to confirm healthy sensor behavior and proper driver scaling.
- **Connection to Kalibr**: Always fit the white-noise coefficient across the actual $-1/2$-slope region and verify that your Allan deviation convention and units match Kalibr's continuous-time noise-density format; blindly reading off the sample nearest $\tau=1$ second easily masks slope or unit mismatches.
- **Temperature stabilization**: Start logging only after the sensor's internal temperature and output mean have stabilized, and log temperature alongside the raw data. Required warm-up time varies widely across IMUs, enclosures, and ambient conditions.

Camera-IMU calibration relies on observing a visual calibration target while moving the rig. When we pair a LiDAR with an IMU instead, neither sensor sees visual AprilTags—so motion itself becomes the common calibration reference.

---

## 3.5 LiDAR-IMU Extrinsic Calibration

Without a shared visual pattern, calibrating the rigid transform between a LiDAR and an IMU reduces to matching the ego-motion each sensor measures on its own—a classic formulation known as **hand-eye calibration**.

### 3.5.1 Hand-Eye Calibration (AX = XB)

Originally formulated in robotics to find the transform between a robot gripper ("hand") and a wrist-mounted camera ("eye") ([Tsai & Lenz, 1989](https://ieeexplore.ieee.org/document/34770)), hand-eye calibration exploits one simple geometric fact: if two sensors are bolted rigidly together, any physical motion of the rig shifts both sensors simultaneously while keeping the rigid link between them constant.

**Problem definition**:

Suppose sensors $A$ (e.g., LiDAR) and $B$ (e.g., IMU) are fixed on a single rigid body. Between two instants $i, j$ (at times $t_i$ and $t_j$), each sensor independently tracks its own local relative displacement:

$$
\mathbf{A}_{ij} = \mathbf{T}_A^{-1}(t_i) \cdot \mathbf{T}_A(t_j) \quad \text{(relative motion of sensor A)}
$$
$$
\mathbf{B}_{ij} = \mathbf{T}_B^{-1}(t_i) \cdot \mathbf{T}_B(t_j) \quad \text{(relative motion of sensor B)}
$$

Consider walking around the coordinate loop from sensor $A$ at $t_i$ to sensor $B$ at $t_j$: whether you step along $\mathbf{A}_{ij}$ first and cross the rigid mount $\mathbf{X} = \mathbf{T}_{AB}$ afterward, or cross $\mathbf{X}$ first and step along $\mathbf{B}_{ij}$, you must arrive at the same pose:

$$
\mathbf{A}_{ij} \mathbf{X} = \mathbf{X} \mathbf{B}_{ij}
$$

This is the celebrated $\mathbf{AX} = \mathbf{XB}$ equation, which we solve for the rigid transformation $\mathbf{X} \in SE(3)$.

**Separating rotation from translation**:

Expanding the $4 \times 4$ matrix equation into its $3 \times 3$ rotation and $3 \times 1$ translation blocks cleanly decouples the two unknowns:

$$
\mathbf{R}_A \mathbf{R}_X = \mathbf{R}_X \mathbf{R}_B \quad \text{(rotation)}
$$
$$
\mathbf{R}_A \mathbf{t}_X + \mathbf{t}_A = \mathbf{R}_X \mathbf{t}_B + \mathbf{t}_X \quad \text{(translation)}
$$

Because the rotation equation does not depend on $\mathbf{t}_X$, we can solve for $\mathbf{R}_X$ first and then plug it into the linear translation system for $\mathbf{t}_X$.

**[Tsai & Lenz (1989)](https://ieeexplore.ieee.org/document/34770) solution**:

Express each relative rotation in axis-angle form—$\hat{\mathbf{a}}_A, \alpha$ for sensor $A$ and $\hat{\mathbf{a}}_B, \beta$ for sensor $B$—and define the scaled quaternion vector parts $\mathbf{p}_A = 2\sin(\alpha/2)\hat{\mathbf{a}}_A$ and $\mathbf{p}_B = 2\sin(\beta/2)\hat{\mathbf{a}}_B$. Rotational conjugation then turns into a purely linear vector equation:

$$
\text{skew}(\mathbf{p}_A + \mathbf{p}_B) \cdot \mathbf{r}_X = \mathbf{p}_B - \mathbf{p}_A
$$

Here $\mathbf{r}_X = \tan(\theta_X/2)\hat{\mathbf{a}}_X$ is the modified Rodrigues vector representing $\mathbf{R}_X$. Stacking these equations across multiple motion pairs produces a linear system $\mathbf{C} \mathbf{r}_X = \mathbf{d}$ that becomes uniquely solvable as soon as we have at least 2 motion pairs with non-parallel rotation axes; once $\mathbf{R}_X$ is recovered, $\mathbf{t}_X$ follows from an analogous linear system.

**Leveraging more motion pairs**: Real-world workflows gather dozens or hundreds of motion pairs, solve the overdetermined linear system via least squares for an initial estimate, and finish with nonlinear Levenberg-Marquardt refinement.

### 3.5.2 Motion-based Automatic Calibration

Because hand-eye calibration operates purely on each sensor's independent relative motion (odometry), you can run it anywhere simply by moving the rig freely through 3D space—no checkerboard or special calibration room required. The implementation below solves the Tsai-Lenz rotation and translation systems via least squares, skipping near-zero rotation intervals whose tiny angles would otherwise make the linear system ill-conditioned:

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

        # Scaled quaternion vector parts: 2 sin(angle / 2) * axis
        a_prime = 2 * np.sin(alpha / 2) * rA / alpha
        b_prime = 2 * np.sin(beta / 2) * rB / beta

        # skew(a' + b') * rX = b' - a'
        skew_sum = np.array([
            [0, -(a_prime[2]+b_prime[2]), a_prime[1]+b_prime[1]],
            [a_prime[2]+b_prime[2], 0, -(a_prime[0]+b_prime[0])],
            [-(a_prime[1]+b_prime[1]), a_prime[0]+b_prime[0], 0]
        ])

        C.append(skew_sum)
        d.append(b_prime - a_prime)

    C = np.vstack(C)
    d = np.concatenate(d)

    # Least-squares solve
    rX, _, _, _ = np.linalg.lstsq(C, d, rcond=None)

    # Rodrigues vector → rotation matrix
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

Although tightly coupled LIO filters such as FAST-LIO2 can refine LiDAR-IMU extrinsics online, they still need a reasonable initial guess and rich motion excitation to converge. [**LI-Init**](https://arxiv.org/abs/2202.11006) solves this cold-start problem with a dedicated initialization module that jointly recovers time offset, extrinsics, gravity direction, and IMU biases before launching FAST-LIO2:

1. Compute independent motion estimates: integrate IMU measurements for attitude and run LiDAR scan matching for 6-DoF pose
2. Align and iteratively refine the spatial transform and clock offset from the discrepancy between the two trajectories
3. Pass the initialized state to FAST-LIO2, whose Error-State Iterated Kalman Filter (ESIKF) keeps the LiDAR-IMU extrinsic inside the state vector for ongoing online refinement

Because this pipeline needs neither a calibration target nor a third sensor, you can initialize an LIO system directly in the field. Keep in mind, however, that convergence still hinges on exciting multiple rotational and translational axes—the public implementation recommends starting with a brief stationary segment followed by 15–30 seconds of online refinement after initialization.

**Advantages**: Completely targetless and self-contained, enabling convenient field initialization without extra hardware.
**Disadvantages**: Can stall or drift if initial excitation is weak, and typically trails dedicated target-based methods in ultimate geometric accuracy.

**GRIL-Calib**: Standard motion-based methods lose observability on wheeled ground robots whose motion stays mostly planar, leaving pitch, roll, or vertical translation poorly constrained. [GRIL-Calib (Kim et al., 2024)](https://arxiv.org/abs/2312.14035) overcomes this degeneracy by exploiting ground-plane residuals from LiDAR odometry alongside a ground-plane motion (GPM) constraint, enabling full 6-DoF LiDAR-IMU calibration even from planar driving alone.

Multi-LiDAR rigs—common on autonomous vehicles—add one more geometric layer: alongside the LiDAR-IMU transform, we must calibrate the relative poses linking every LiDAR unit to its peers.

---

## 3.6 LiDAR-LiDAR Extrinsic Calibration

Autonomous vehicles and large mobile robots routinely mount multiple LiDARs—at the front, sides, and rear—to eliminate blind spots. Merging their scans into one coherent 3D point cloud requires estimating the rigid relative pose of every secondary unit with respect to a chosen reference LiDAR.

### 3.6.1 Problem Setup

Given $n$ LiDARs $\{L_1, L_2, \ldots, L_n\}$ bolted rigidly to the same chassis, the goal is to recover the relative transformations $\{\mathbf{T}_{L_1 L_i}\}_{i=2}^{n}$ mapping each sensor $L_i$ into the coordinate frame of the primary reference LiDAR $L_1$.

### 3.6.2 Target-based Methods

A straightforward target-based workflow places a large planar calibration board where two or more LiDARs can scan it simultaneously. Fitting the board plane in each sensor's point cloud via RANSAC yields paired plane equations whose normals and offsets constrain the inter-LiDAR transform.

### 3.6.3 Targetless: ICP-Based

When field of view permits—or when carrying a large calibration board is impractical—scan registration offers a natural targetless alternative via Iterative Closest Point (ICP).

**When there is an overlapping region**:

If two LiDARs share an overlapping field of view, we can align their simultaneous point clouds directly by minimizing point-to-point Euclidean distances:

$$
\min_{\mathbf{R}, \mathbf{t}} \sum_{i} \|\mathbf{p}_{L_2,i} - (\mathbf{R} \mathbf{p}_{L_1,i'} + \mathbf{t})\|^2
$$

In structured scenes with planar walls or road surfaces, **point-to-plane ICP** typically converges faster and resists tangential sliding by penalizing displacement only along the local surface normal $\mathbf{n}_{L_2,i}$:

$$
\min_{\mathbf{R}, \mathbf{t}} \sum_{i} \left[(\mathbf{R} \mathbf{p}_{L_1,i'} + \mathbf{t} - \mathbf{p}_{L_2,i})^\top \mathbf{n}_{L_2,i}\right]^2
$$

Here $\mathbf{n}_{L_2,i}$ denotes the surface normal estimated at the target point in $L_2$.

**When there is no overlapping region**:

Side- or rear-facing sensors often share zero instantaneous overlap. Direct scan-to-scan ICP fails in that case, leaving two effective motion-based strategies:
1. **Hand-eye calibration (Section 3.5.1)**: run independent LiDAR odometry on each sensor while driving a figure-eight or multi-turn trajectory, then solve $\mathbf{A}\mathbf{X} = \mathbf{X}\mathbf{B}$ from the paired relative motions
2. **SLAM-based alignment**: let each LiDAR build a local map independently via SLAM, then register the two overlapping maps accumulated as the vehicle moves through the same environment

### 3.6.4 Feature-based Methods

Point-wise correspondence can struggle when two LiDARs differ in beam count or scanning angle. In structured urban or indoor environments, extracting macroscopic geometric primitives—building walls, ground planes, poles, and sharp architectural edges—provides far greater robustness to scan noise:

- Segment large shared planes (such as building facades or level floors) independently in each LiDAR
- Estimate the plane parameters $(n_i, d_i)$ in each sensor's local frame
- Solve for the relative rotation and translation from matched plane pairs

To constrain all 6 degrees of freedom uniquely, you need at least 3 plane correspondences whose surface normals span 3D space (i.e., are non-coplanar).

The calibration problems examined so far (3.1-3.6) align exteroceptive perception sensors and inertial units mounted locally on the platform. When an outdoor robot relies on GNSS for global positioning, we must also account for the physical offset between the GNSS antenna and the IMU.

---

## 3.7 GNSS-IMU Lever Arm & Boresight

In an integrated GNSS/INS navigation system, the 3D offset between the GNSS antenna and the IMU is called the **lever arm**. Because GNSS reports the position of an antenna often mounted high on a roof rack while the inertial filter tracks the IMU mounted inside the chassis, failing to model this offset injects attitude-dependent position errors into the fusion loop.

### 3.7.1 Lever Arm Vector

Formally, the lever arm $\mathbf{l} = [l_x, l_y, l_z]^\top$ is the fixed 3D vector pointing from the IMU origin to the electrical phase center of the GNSS antenna, expressed in the IMU body frame.

Think of the antenna as riding on the end of a rigid boom attached to the IMU: GNSS measures the tip of that boom in the navigation frame, whereas the estimator tracks the IMU at its base. Subtracting the body-to-navigation rotated lever arm recovers the true IMU position:

$$
\mathbf{p}_{\text{IMU}} = \mathbf{p}_{\text{GNSS}} - \mathbf{R}_{\text{body}}^{\text{nav}} \cdot \mathbf{l}
$$

Here $\mathbf{R}_{\text{body}}^{\text{nav}}$ rotates vectors from the IMU body frame into the navigation frame. Whenever the vehicle pitches, rolls, or turns, the antenna swings along an arc around the IMU, so omitting or miscalibrating $\mathbf{l}$ corrupts position updates proportionally to attitude change. For example, if a 1 m lever arm perpendicular to the rotation axis is accounted for only at a static reference attitude and a subsequent 10 deg rotation goes uncorrected, the resulting position discrepancy is roughly 17 cm.

### 3.7.2 Lever Arm Estimation Methods

Depending on the accuracy target and platform dynamics, the lever arm is determined in one of three ways:

**Method 1: Physical measurement**

The simplest approach measures the offset directly on the vehicle CAD model or with a tape measure or laser rangefinder. Whether centimeter-level tape accuracy suffices depends on your error budget—derive the required tolerance from the lever-arm length, expected pitch/roll/yaw angles, and maximum allowable position error, and escalate to surveying or filter estimation whenever tighter bounds are needed.

**Method 2: Filter-based online estimation**

Augmenting the EKF state vector with $\mathbf{l}$ lets the navigation filter refine the lever arm online, provided the platform undergoes rich rotational motion. During straight-line driving at constant attitude, $\mathbf{R}\mathbf{l}$ remains constant and becomes indistinguishable from a fixed bias, rendering all three components unobservable; similarly, pure yaw rotation about a vertical axis leaves the vertical component ($l_z$) along that axis unobservable.

**Method 3: Post-processing**

In precision mobile mapping and geodesy, recorded GNSS and IMU logs are processed backward and forward through a fixed-interval smoother to jointly optimize the trajectory and lever arm with maximum accuracy.

### 3.7.3 GNSS Antenna Phase Center

Careful GNSS integration requires one final distinction: the antenna phase center (APC)—the effective point where satellite radio signals are received—rarely coincides exactly with the geometric center of the plastic radome, and it even shifts by millimeters to centimeters depending on satellite elevation angle and carrier frequency. This effect is known as Phase Center Variation (PCV).

For centimeter-grade RTK or Precise Point Positioning (PPP), apply standardized ANTEX calibration tables whenever your error budget and processing pipeline call for them, weighing antenna design, base/rover differential cancellation, expected vehicle tilt, and target accuracy.

---

## 3.8 Online / Continuous Calibration

Even a flawless factory or lab calibration will not stay exact forever. Thermal expansion, mechanical vibration, or minor bumps slowly flex sensor mounts over days and months of deployment. **Online calibration** keeps the sensor model honest by tracking and correcting those slow drifts during live operation.

### 3.8.1 Self-Calibration during SLAM

A natural strategy is to promote key calibration parameters from fixed constants to time-varying state variables inside the SLAM or VIO estimator itself.

**EKF-based**: Frameworks like OpenVINS augment the navigation state with camera intrinsics, camera-IMU extrinsics, and the temporal offset so the filter updates them continuously whenever motion excites observability:

$$
\mathbf{x} = \begin{bmatrix} \mathbf{x}_{\text{nav}} \\ \mathbf{x}_{\text{calib}} \end{bmatrix}
= \begin{bmatrix} \mathbf{q}, \mathbf{p}, \mathbf{v}, \mathbf{b}_g, \mathbf{b}_a \\ \mathbf{q}_{CI}, \mathbf{p}_{CI}, t_d, f_x, f_y, c_x, c_y, k_1, \ldots \end{bmatrix}
$$

To allow slow physical drift without letting parameters jump wildly on noisy frames, the calibration sub-state typically evolves under a gentle random-walk process model:

$$
\dot{\mathbf{x}}_{\text{calib}} = \mathbf{w}_{\text{calib}}, \quad \mathbf{w}_{\text{calib}} \sim \mathcal{N}(\mathbf{0}, \mathbf{Q}_{\text{calib}})
$$

Tuning $\mathbf{Q}_{\text{calib}}$ balances responsiveness against stability: setting it too large makes estimates jitter under uninformative motion, while setting it too small locks out genuine mechanical shift.

**Factor graph-based**: Optimization backends such as LIO-SAM and VINS-Mono similarly insert extrinsic nodes into the factor graph, using the covariance of connecting prior or between factors to regulate how tightly parameters stay anchored.

### 3.8.2 Correcting Extrinsic Drift

Long-running autonomous fleets typically pair online estimators with explicit health-monitoring safeguards to detect and recover from mounting shifts:

1. **Reprojection-error monitoring**: Continuously evaluate edge alignment or reprojection residuals when projecting LiDAR points into camera images, triggering a recalibration alert whenever alignment degrades
2. **Periodic recalibration**: Schedule automated targetless calibration runs over logged operational segments at regular service intervals
3. **Online fine-tuning**: Warm-start optimization from the active calibration and adjust parameters continuously within tight sanity bounds

Continuing the trend toward automated maintenance, [CalibRefine (2025)](https://arxiv.org/abs/2502.17648) introduces a deep-learning framework that ingests raw LiDAR point clouds and camera images directly to perform online targetless calibration followed by iterative refinement.

### 3.8.3 OpenCalib: An Integrated Calibration Framework for Autonomous Driving

Managing a full autonomous-driving sensor suite often means juggling a patchwork of separate calibration scripts. [OpenCalib (Yan et al., 2022)](https://arxiv.org/abs/2205.14087) consolidates the major spatial calibrations for autonomous vehicles into one cohesive open-source toolkit.

The table below summarizes the sensor pairs and target requirements supported in OpenCalib:

| Sensor pair | Method | Target |
|---------|------|------|
| Camera intrinsic | Zhang's method | Checkerboard/AprilTag |
| Camera-Camera | Stereo calibration | Shared target |
| Camera-LiDAR | PnP / Edge alignment | With/without |
| LiDAR-LiDAR | ICP / Feature matching | None |
| Camera-Ground | Vanishing point | None |
| Online correction | Monitoring + recalibration | None |

**OpenCalib's design philosophy**:
- **Modularity**: each sensor-pair solver ships as an independent module so teams can adopt only the components their rig requires
- **Unified interface**: a consistent ROS-based workflow ingests heterogeneous sensor streams without custom glue code
- **Visualization**: real-time projection overlays let engineers verify calibration quality visually on the spot

One crucial systems caveat remains: calibration across a multi-sensor vehicle must form a **consistent sensor chain**. If a car carries six cameras and one LiDAR, calibrating each camera-to-LiDAR pair independently does not guarantee that composing two transforms agrees with the stereo extrinsic between adjacent cameras. Because OpenCalib operates pairwise, you should explicitly verify loop consistency across composed transforms whenever multiple overlapping sensors share the same chassis.

With spatial transforms and online drift tracking in place, one critical dimension remains: ensuring that every sensor's measurements align cleanly in **time**.

---

## 3.9 Temporal Calibration

Accurate spatial geometry fails as soon as sensor timestamps drift apart. On a vehicle moving at highway speed—or a drone pitching rapidly—even a few milliseconds of clock skew translate directly into centimeters of spatial error or degrees of angular mismatch.

### 3.9.1 Hardware Synchronization

Whenever your platform allows wiring a common trigger line and shared clock, hardware synchronization should be your first line of defense—though you still need to verify internal device exposure/scan latencies and confirm whether timestamps mark the start, midpoint, or end of an acquisition.

**Trigger-based synchronization**:

A central master timer broadcasts electrical trigger pulses so every sensor initiates capture in lockstep:

- **Camera trigger**: an external GPIO pulse triggers shutter exposure deterministically
- **LiDAR sync**: spinning LiDARs such as Ouster accept a Pulse Per Second (PPS) input to phase-lock their rotation and scan start times
- **IMU sync**: the inertial sampling clock locks to an external reference pulse

**PPS (Pulse Per Second)**:

GNSS receivers commonly output a crisp 1 Hz electrical pulse aligned to the exact GNSS second boundary. Provided the receiver is configured to align its rising edge with the integer second—and you verify cable delays, optocoupler latency, and each sensor's timestamp convention—that 1 Hz pulse serves as a universal metronome. As illustrated below, sensors running at wildly different rates (30 Hz cameras, 10 Hz LiDARs, 200 Hz IMUs) can measure their local clock offset against each rising PPS edge and map every packet onto a single GPS timeline:

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

By tracking the offset between each device's internal counter and the incoming PPS pulse, all heterogeneous streams align onto one common GPS time axis.

### 3.9.2 PTP (Precision Time Protocol)

When sensors connect over Ethernet without dedicated trigger wiring, IEEE 1588 PTP synchronizes device clocks directly over the network. Given hardware timestamping and a PTP-aware switch, synchronization can reach sub-microsecond precision, though real-world accuracy depends on switch symmetry and driver configuration.

**How PTP works**:

1. **Master-slave hierarchy**: a designated Grandmaster clock broadcasts the authoritative network time
2. **Sync message**: the Grandmaster periodically emits a `Sync` packet and notes its exact egress time $t_1$
3. **Follow-up**: because software stacks rarely inject $t_1$ on the fly, a `Follow-up` packet carries the precise hardware departure timestamp $t_1$
4. **Delay Request/Response**: the slave sends `Delay_Req` at $t_3$ and receives `Delay_Resp` reporting the master's arrival timestamp $t_4$
5. **Offset computation**: assuming symmetric uplink and downlink propagation delays, subtracting the round-trip transit time yields $\text{offset} = \frac{(t_2 - t_1) - (t_4 - t_3)}{2}$

The sequence diagram below traces this four-timestamp exchange between Grandmaster and sensor, showing how both clock offset and one-way network delay fall out of simple arithmetic:

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

Many modern Ethernet LiDARs and industrial GigE cameras support PTP hardware timestamping out of the box; check each datasheet carefully to confirm support and verify which physical instant (exposure start vs. mid-exposure, packet transmit vs. frame start) the timestamp represents.

### 3.9.3 Software Synchronization

When hardware triggers and PTP are unavailable—such as on low-cost USB webcams or consumer IMUs—software must bridge the clocks instead.

**Host-clock based**: Stamping incoming packets with the host CPU clock upon arrival is simple, but every timestamp silently absorbs USB polling intervals, network buffering, and OS scheduling jitter—latencies that vary with system load and must be benchmarked empirically.

**ROS Time**: Similarly, calling `ros::Time::now()` inside a driver captures message arrival rather than physical exposure. Whenever a driver translates an internal device counter into ROS time, the fidelity of that clock-skew model sets the ceiling on your overall synchronization accuracy.

### 3.9.4 Online Estimation of the Time Offset

Even without hardware synchronization, software estimators can recover an unknown inter-sensor time offset directly from dynamic motion. [Li & Mourikis (2014)](https://arxiv.org/abs/1310.7863) pioneered estimating the camera-IMU time offset $t_d$ inside a VIO state vector alongside pose and extrinsics.

**Reflecting the time offset in the observation model**:

Rather than querying the IMU-integrated trajectory at the raw camera stamp $t_c$, the observation model evaluates the platform pose at the shifted physical instant $t_c + t_d$:

$$
\mathbf{z}(t_c) = \pi(\mathbf{T}(t_c + t_d) \cdot \mathbf{p}_w)
$$

Because $t_d$ is unknown, how do we differentiate with respect to a time shift? Assuming $t_d$ is small, a first-order Taylor expansion turns that tiny temporal offset into an equivalent spatial perturbation along the platform's instantaneous body velocity (angular + linear twist $\boldsymbol{\xi}$):

$$
\mathbf{T}(t_c + t_d) \approx \mathbf{T}(t_c) \cdot \text{Exp}(\boldsymbol{\xi} \cdot t_d)
$$

Here $\boldsymbol{\xi}$ is the body twist (angular and linear velocity) at $t_c$.

This linearization exposes an analytical Jacobian with respect to $t_d$, letting an EKF or nonlinear least-squares solver refine the time offset right alongside spatial states.

Extending this concept beyond two sensors, [iKalibr (Chen et al., 2024)](https://arxiv.org/abs/2407.11420) formulates a unified continuous-time B-spline framework (IEEE T-RO 2025) capable of calibrating spatial extrinsics and temporal offsets across LiDAR, camera, IMU, and radar **simultaneously**.

**Observability conditions**: Time-offset estimation demands genuine acceleration and rotation. Under constant-velocity straight-line motion, shifting a timestamp produces the exact same visual displacement as shifting a sensor along its velocity vector—making temporal offset indistinguishable from spatial translation.

Modern VIO frameworks such as OpenVINS and VINS-Mono implement online variants of this formulation, while offline tools like Kalibr (Section 3.4.2) estimate the same offset globally via continuous-time B-spline batch optimization.

### 3.9.5 Practical Synchronization Strategy Guide

The decision table below matches common hardware capabilities to the most effective synchronization strategy and the checks needed to validate it:

| Configuration | First method to consider | What to verify |
|---------------|--------------------------|----------------|
| A shared trigger and clock can be wired | Hardware trigger + common clock/PPS | Timestamp reference within exposure, sample, or scan; fixed latency |
| Every device and switch supports PTP and hardware timestamps | PTP | Grandmaster selection, PTP profile, path asymmetry, lock state |
| Device clocks are readable but no shared signal exists | Clock conversion + offline/online offset estimation | Clock drift, motion observability, time variation of the offset |
| Only receive timestamps are available | Use host timestamps provisionally and measure jitter | USB/network queueing; change hardware if the error budget is not met |

**For systems requiring hardware synchronization**: Combining a GNSS-derived PPS pulse with Ethernet PTP lets every sensor—from cameras and LiDARs to IMUs—lock smoothly onto a single authoritative GNSS clock.

**Research/prototyping grade**: Software timestamping paired with online time-offset estimation often suffices: run Kalibr offline to pin down the initial offset, then let your VIO filter track minor thermal clock drift online.

In all cases, prefer hardware synchronization whenever the hardware supports it—and still measure fixed trigger latencies and timestamp conventions—treating software offset estimation as either a fine-tuning safety net or a fallback when shared trigger lines are absent.

---

## Summary

The table below consolidates the full calibration toolkit covered in this chapter, mapping each sensor pairing to its target parameters, primary solvers, and data prerequisites:

| Calibration type | Parameters estimated | Method | Minimum requirement |
|----------------|-------------|----------|-------------|
| Camera intrinsic | $\mathbf{K}, \mathbf{d}$ | Zhang's method | Diverse target poses covering the image |
| Stereo extrinsic | $\mathbf{R}, \mathbf{t}$ | Shared target + stereoCalibrate | Diverse poses detected simultaneously by both cameras |
| Camera-LiDAR | $\mathbf{T}_{CL}$ | Target-based / NID | Multiple target views / natural scene |
| Camera-IMU | $\mathbf{T}_{CI}, t_d$ | Kalibr (B-spline) | A sufficiently long sequence with diverse motion |
| LiDAR-IMU | $\mathbf{T}_{LI}$ | Hand-eye (AX=XB) / LI-Init | Diverse motion |
| LiDAR-LiDAR | $\mathbf{T}_{L_1 L_2}$ | ICP / Feature matching | Overlapping region or motion data |
| GNSS-IMU | $\mathbf{l}$ (lever arm) | Physical measurement / EKF estimation | Rotational motion |
| Temporal | $t_d$ | PPS/PTP + online estimation | Accelerated motion |

**Recommended calibration order**: Because later stages depend on accurate upstream intrinsics and reference frames, calibrate a multi-sensor rig in a disciplined sequence:

1. Intrinsics of each camera (calibrated independently first)
2. Stereo extrinsics (if using multi-camera pairs)
3. Camera-IMU extrinsics and time offset (Kalibr)
4. LiDAR-IMU extrinsics (hand-eye calibration or LI-Init)
5. Camera-LiDAR extrinsics (derived indirectly via the IMU chain or calibrated directly)
6. GNSS lever arm
7. Online calibration and drift monitoring (active during deployment)

While chaining transforms indirectly—for instance, computing camera-to-LiDAR as $\mathbf{T}_{CL} = \mathbf{T}_{CI} \cdot \mathbf{T}_{IL}$—is convenient, individual estimation errors compound across the chain, so always cross-check composed poses with a direct calibration or visual overlay.

With every sensor's internal optics, rigid inter-sensor extrinsics, and clock offsets calibrated, all incoming streams now share a consistent geometric and temporal reference frame—ready to be fused by the state estimators of Chapter 4 (→ Chapter 4).

**Related papers**:

- [Zhang (2000)](https://ieeexplore.ieee.org/document/888718): camera intrinsic — flexible homography-based calibration
- [Furgale et al. (2013)](https://ieeexplore.ieee.org/document/6696514): camera-IMU — simultaneous spatio-temporal calibration with B-spline continuous-time trajectories
- [Tsai & Lenz (1989)](https://ieeexplore.ieee.org/document/34770): hand-eye — the original AX=XB
- [Koide et al. (2023)](https://arxiv.org/abs/2302.05094): camera-LiDAR targetless — automatic calibration based on NID + SuperGlue
- [Li & Mourikis (2014)](https://arxiv.org/abs/1310.7863): temporal — online estimation of the time offset
- [OpenCalib (2022)](https://arxiv.org/abs/2205.14087): an integrated calibration framework for autonomous driving
- [GRIL-Calib (Kim et al., 2024)](https://arxiv.org/abs/2312.14035): targetless IMU-LiDAR calibration in ground-robot settings using planar-motion constraints, enabling 6-DoF estimation even from constrained planar motion
- [MFCalib (2024)](https://arxiv.org/abs/2409.00992): single-shot targetless LiDAR-camera calibration leveraging multi-feature edges (depth continuity/discontinuity and intensity discontinuity), resolving edge inflation through a LiDAR beam model
- [iKalibr (Chen et al., 2024)](https://arxiv.org/abs/2407.11420): temporal — unified spatio-temporal calibration of heterogeneous multi-sensors (LiDAR, camera, IMU, radar) based on B-spline continuous time (IEEE T-RO 2025)
