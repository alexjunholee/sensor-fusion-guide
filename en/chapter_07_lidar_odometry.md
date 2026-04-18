# Ch.7 — LiDAR Odometry & LiDAR-Inertial Odometry

In Ch.6 we covered camera(+IMU)-based Visual Odometry. In this chapter we approach the same ego-motion estimation problem from the LiDAR side.

LiDAR is a sensor that complements the camera. While the camera provides rich texture information but is sensitive to illumination and cannot recover absolute range, LiDAR delivers illumination-invariant, precise 3D range measurements. This chapter examines in depth the internal structure of LiDAR Odometry (LO), which estimates ego-motion from LiDAR alone, and of LiDAR-Inertial Odometry (LIO), which couples LiDAR with an IMU.

The central problem in LiDAR odometry is **point cloud registration** — finding the rigid-body transformation $\mathbf{T} \in SE(3)$ between two consecutive scans. Despite its seeming simplicity, this problem in practice involves a variety of challenges: data association (correspondence), noise modeling, computational efficiency, and motion distortion compensation.

---

## 7.1 Point Cloud Registration Fundamentals

Point cloud registration is the problem of finding the optimal rigid-body transformation between two point clouds $\mathcal{P} = \{\mathbf{p}_i\}$ and $\mathcal{Q} = \{\mathbf{q}_j\}$:

$$\mathbf{T}^* = \underset{\mathbf{T} \in SE(3)}{\arg\min} \sum_i d(\mathbf{T} \cdot \mathbf{p}_i, \mathcal{Q})$$

Here $d(\cdot, \cdot)$ is a distance metric between a transformed source point and the target point cloud. Different definitions of this distance give rise to different ICP variants.

### 7.1.1 ICP Variants

**Point-to-Point ICP ([Besl & McKay, 1992](https://doi.org/10.1109/34.121791))**

In its most basic form, it minimizes the Euclidean distance between a transformed source point and its closest target point:

$$\mathbf{T}^* = \underset{\mathbf{T}}{\arg\min} \sum_i \left\|\mathbf{T} \cdot \mathbf{p}_i - \mathbf{q}_{c(i)}\right\|^2$$

where $c(i) = \arg\min_j \|\mathbf{T} \cdot \mathbf{p}_i - \mathbf{q}_j\|$ is the closest-point correspondence. ICP iterates two steps:

1. **Correspondence search (closest point)**: Using the current transformation, transform the source points and then find the closest point in the target. Using a kd-tree this takes $O(N\log N)$.

2. **Transformation estimation**: Given the correspondence pairs, the optimal transformation can be obtained in closed form. Using SVD:

   Subtract the centroids of both point sets:
   $$\bar{\mathbf{p}} = \frac{1}{N}\sum_i \mathbf{p}_i, \quad \bar{\mathbf{q}} = \frac{1}{N}\sum_i \mathbf{q}_{c(i)}$$

   Compute the cross-covariance matrix:
   $$\mathbf{W} = \sum_i (\mathbf{p}_i - \bar{\mathbf{p}})(\mathbf{q}_{c(i)} - \bar{\mathbf{q}})^T$$

   SVD: $\mathbf{W} = \mathbf{U}\boldsymbol{\Sigma}\mathbf{V}^T$

   Optimal rotation and translation:
   $$\mathbf{R}^* = \mathbf{V} \text{diag}(1, 1, \det(\mathbf{V}\mathbf{U}^T)) \mathbf{U}^T, \quad \mathbf{t}^* = \bar{\mathbf{q}} - \mathbf{R}^*\bar{\mathbf{p}}$$

   When $\det(\mathbf{V}\mathbf{U}^T) = 1$, we have $\mathbf{R}^* = \mathbf{V}\mathbf{U}^T$; when $\det(\mathbf{V}\mathbf{U}^T) = -1$, the sign of the last column of $\mathbf{V}$ is flipped to prevent a reflection.

Limitations of point-to-point ICP:
- Sliding on planes — points on a plane can slide along the plane without changing the cost, leading to slow convergence.
- Initialization dependence — it easily falls into local minima.
- Inaccurate closest-point correspondences — if the two scans have different sampling patterns, the nearest neighbor may not be the true correspondence.

**Point-to-Plane ICP**

For points on a planar surface, the distance from a point to the plane is physically more meaningful than the distance between two points:

$$\mathbf{T}^* = \underset{\mathbf{T}}{\arg\min} \sum_i \left((\mathbf{T} \cdot \mathbf{p}_i - \mathbf{q}_{c(i)})^T \mathbf{n}_{c(i)}\right)^2$$

where $\mathbf{n}_{c(i)}$ is the surface normal at the target point $\mathbf{q}_{c(i)}$. Because only the normal-direction component of the distance is measured, sliding along the plane does not contribute to the cost.

Advantages: Convergence is much faster than point-to-point. It is particularly effective in indoor/urban environments rich in planar structures.

Disadvantages: There is no closed-form solution, so iterative optimization (e.g., Gauss-Newton) is required. Accuracy depends on the quality of normal estimation.

Normal estimation is performed by running PCA (principal component analysis) on the neighboring points of each point and taking the eigenvector corresponding to the smallest eigenvalue as the normal. The neighborhood covariance matrix is

$$\mathbf{C} = \frac{1}{k}\sum_{j \in \mathcal{N}(i)} (\mathbf{q}_j - \bar{\mathbf{q}})(\mathbf{q}_j - \bar{\mathbf{q}})^T$$

and from the eigendecomposition $\mathbf{C} = \mathbf{V}\boldsymbol{\Lambda}\mathbf{V}^T$ the eigenvector corresponding to $\lambda_{\min}$ is the normal direction.

### 7.1.2 GICP (Generalized ICP)

GICP ([Segal et al., 2009](https://doi.org/10.15607/RSS.2009.V.021)) unifies point-to-point, point-to-plane, and plane-to-plane ICP into a single probabilistic framework.

The key idea: each point is modeled as carrying a covariance $\mathbf{C}_i$ that reflects the uncertainty of the local surface. The cost function is

$$\mathbf{T}^* = \underset{\mathbf{T}}{\arg\min} \sum_i (\mathbf{T} \cdot \mathbf{p}_i - \mathbf{q}_{c(i)})^T (\mathbf{C}_i^{\mathcal{Q}} + \mathbf{R}\mathbf{C}_i^{\mathcal{P}}\mathbf{R}^T)^{-1} (\mathbf{T} \cdot \mathbf{p}_i - \mathbf{q}_{c(i)})$$

where $\mathbf{C}_i^{\mathcal{P}}, \mathbf{C}_i^{\mathcal{Q}}$ are the local surface covariances at the source and target points, respectively.

**Physical meaning of the covariance**:
- For a point on a plane: small variance along the normal, large variance along the tangent directions $\rightarrow \mathbf{C} = \mathbf{R}_s \text{diag}(\epsilon, 1, 1) \mathbf{R}_s^T$ ($\epsilon \ll 1$; $\mathbf{R}_s$ is the rotation aligning the normal to the first axis).
- In this case GICP automatically becomes plane-to-plane registration.
- When $\mathbf{C}^{\mathcal{P}} = \mathbf{0}$ it reduces to point-to-plane, and when $\mathbf{C}^{\mathcal{P}} = \mathbf{C}^{\mathcal{Q}} = \mathbf{I}$ it reduces to point-to-point.

GICP is theoretically the most general ICP framework, and in practice it delivers the most accurate results across a wide range of environments.

```python
# GICP core iteration pseudocode
def gicp(P, Q, T_init, max_iter=50, tol=1e-6):
    """
    P: source point cloud (N x 3)
    Q: target point cloud (M x 3)
    T_init: initial transformation (4 x 4)
    """
    T = T_init.copy()
    
    # Precompute local surface covariance for each point
    C_P = compute_local_covariances(P, k_neighbors=20)  # N x 3 x 3
    C_Q = compute_local_covariances(Q, k_neighbors=20)  # M x 3 x 3
    
    # Build target kd-tree
    tree = KDTree(Q)
    
    for iteration in range(max_iter):
        # 1. Transform source points
        P_transformed = apply_transform(T, P)
        
        # 2. Find closest-point correspondences
        distances, indices = tree.query(P_transformed)
        
        # 3. Gauss-Newton update
        H = np.zeros((6, 6))  # Hessian approximation
        b = np.zeros(6)       # gradient
        
        for i in range(len(P)):
            j = indices[i]
            residual = P_transformed[i] - Q[j]
            
            # Combined covariance (in the transformed frame)
            R = T[:3, :3]
            Sigma = C_Q[j] + R @ C_P[i] @ R.T
            Sigma_inv = np.linalg.inv(Sigma)
            
            # SE(3) Jacobian
            J = compute_se3_jacobian(T, P[i])  # 2D: 3 x 6
            
            # Accumulate
            H += J.T @ Sigma_inv @ J
            b += J.T @ Sigma_inv @ residual
        
        # 4. Compute and apply the increment
        xi = np.linalg.solve(H, -b)
        T = se3_exp(xi) @ T
        
        if np.linalg.norm(xi) < tol:
            break
    
    return T
```

### 7.1.3 NDT (Normal Distributions Transform)

[NDT (Biber & Strasser, 2003)](https://doi.org/10.1109/IROS.2003.1249285) does not use the point cloud directly; instead it partitions space into voxels and models the distribution of points within each voxel as a Gaussian.

**NDT procedure**:

1. **Build the NDT representation of the target point cloud**: Partition space into a 3D voxel grid and, from the points in each voxel $k$, compute a Gaussian $\mathcal{N}(\boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k)$:
   $$\boldsymbol{\mu}_k = \frac{1}{n_k}\sum_{i \in k} \mathbf{q}_i, \quad \boldsymbol{\Sigma}_k = \frac{1}{n_k-1}\sum_{i \in k} (\mathbf{q}_i - \boldsymbol{\mu}_k)(\mathbf{q}_i - \boldsymbol{\mu}_k)^T$$

2. **Optimize the transformation**: Optimize so that the transformed source points have high likelihood under the target NDT distribution:
   $$\mathbf{T}^* = \underset{\mathbf{T}}{\arg\min} \sum_i -\log p(\mathbf{T} \cdot \mathbf{p}_i \mid \boldsymbol{\mu}_{k(i)}, \boldsymbol{\Sigma}_{k(i)})$$
   
   Under the Gaussian assumption,
   $$\mathbf{T}^* = \underset{\mathbf{T}}{\arg\min} \sum_i (\mathbf{T} \cdot \mathbf{p}_i - \boldsymbol{\mu}_{k(i)})^T \boldsymbol{\Sigma}_{k(i)}^{-1} (\mathbf{T} \cdot \mathbf{p}_i - \boldsymbol{\mu}_{k(i)})$$

Advantages of NDT:
- No explicit correspondence search is required — we only need to determine which voxel a point belongs to. There is no kd-tree construction cost.
- The voxel size controls the trade-off between precision and the convergence basin — larger voxels give a wider convergence basin, smaller voxels give higher precision.
- The cost function is smooth, so the optimization is stable.

Disadvantages:
- Sensitive to the choice of voxel size.
- Covariance estimation is unstable in voxels with few points.
- 2D NDT is widely used in autonomous driving, but 3D NDT tends to be slightly less accurate than ICP/GICP.

### 7.1.4 Convergence and Initialization Dependence

All registration algorithms are local optimizers, so the initial guess matters. If the initial pose error is large, the optimization converges to the wrong local minimum.

Practical ways to supply an initial guess:
1. **Constant-velocity model**: Extrapolate the transformation from the previous two frames. The simplest option, but it fails under abrupt motion.
2. **IMU integration**: Integrating the IMU over the short scan interval provides an excellent initial guess. This is one reason LIO systems are more robust than LO.
3. **Multi-resolution**: A coarse-to-fine approach. Perform a rough registration first with large voxels or downsampling, then refine with a finer registration.
4. **RANSAC + feature matching**: Establish correspondences using 3D descriptors such as FPFH or SHOT, and estimate the initial transformation with RANSAC. Useful for general registration, but in odometry the constant-velocity model combined with IMU is more practical because of temporal continuity.

---

## 7.2 Feature-based LiDAR Odometry

### 7.2.1 LOAM (Lidar Odometry and Mapping in Real-time)

LOAM ([Zhang & Singh, 2014](https://doi.org/10.15607/RSS.2014.X.007)) is the reference point of LiDAR odometry. It held a top position on the KITTI odometry benchmark for a long time and became the foundation of many follow-up systems such as LeGO-LOAM, LIO-SAM, and A-LOAM.

**Key idea: feature extraction + two-stage architecture**

LOAM's core insights are twofold:
1. Extracting geometric features (edge, planar) from a LiDAR scan enables accurate registration using far fewer points than the full point cloud.
2. Separating fast odometry from slow mapping achieves real-time operation and high accuracy at the same time.

**Feature extraction**

On each scan line, the local curvature at each point is computed. The curvature at point $\mathbf{p}_i$ is

$$c_i = \frac{1}{|\mathcal{S}_i| \cdot \|\mathbf{p}_i\|} \left\| \sum_{j \in \mathcal{S}_i} (\mathbf{p}_j - \mathbf{p}_i) \right\|$$

where $\mathcal{S}_i$ is the set of left and right neighbors of $\mathbf{p}_i$ on the same scan line (typically five on each side), and $\|\mathbf{p}_i\|$ is the range from the sensor, used for normalization so that curvatures of near and far points can be compared.

- **Edge feature**: point of high curvature ($c_i > c_{\text{thresh}}^e$). Physically corresponds to corners, poles, and sharp boundaries.
- **Planar feature**: point of low curvature ($c_i < c_{\text{thresh}}^p$). Physically corresponds to walls, floors, and ceilings.

Additional rules when selecting feature points:
- Divide each scan line into four sectors to ensure a uniform distribution.
- If a neighbor has already been selected, exclude the current point (non-maximum suppression).
- Exclude points on near-horizontal surfaces or at occlusion boundaries because they are unstable.

**Odometry module (~10 Hz)**

A fast motion estimate is obtained via scan-to-scan matching. Feature points from the current scan are associated with feature points from the previous scan, but the distance metric depends on the feature type.

**Edge point-to-edge distance**: For an edge point $\mathbf{p}$ in the current scan, find the two closest edge points $\mathbf{a}, \mathbf{b}$ in the previous scan. The distance from $\mathbf{p}$ to the line $\overline{\mathbf{ab}}$ is

$$d_e = \frac{\|(\mathbf{p}-\mathbf{a}) \times (\mathbf{p}-\mathbf{b})\|}{\|\mathbf{a}-\mathbf{b}\|}$$

This is the magnitude of the cross product of the two vectors divided by the base length, which equals the height of the triangle (i.e., the distance from the point to the line).

**Planar point-to-plane distance**: For a planar point $\mathbf{p}$ in the current scan, find the three closest planar points $\mathbf{a}, \mathbf{b}, \mathbf{c}$ in the previous scan. The distance from $\mathbf{p}$ to the plane $\triangle\mathbf{abc}$ is

$$d_p = \frac{(\mathbf{p}-\mathbf{a})^T \left((\mathbf{a}-\mathbf{b}) \times (\mathbf{a}-\mathbf{c})\right)}{\|(\mathbf{a}-\mathbf{b}) \times (\mathbf{a}-\mathbf{c})\|}$$

The numerator is the scalar triple product, which equals the signed distance from the point to the plane multiplied by the magnitude of the normal.

Cost function:

$$\mathbf{T}^* = \underset{\mathbf{T}}{\arg\min} \sum_{\mathbf{p} \in \mathcal{E}} d_e(\mathbf{T}\cdot\mathbf{p})^2 + \sum_{\mathbf{p} \in \mathcal{P}} d_p(\mathbf{T}\cdot\mathbf{p})^2$$

It is optimized with Levenberg-Marquardt.

**Mapping module (~1 Hz)**

Scan-to-map registration provides precise pose refinement and map updates. The new scan is registered against the accumulated global map to correct drift.

The mapping module is slower than the odometry module but more accurate. The map is maintained as a cube centered on the current position, with voxel downsampling to control density.

**Motion Distortion Compensation**

A spinning LiDAR takes about 100 ms to complete a single scan. During that time the robot is moving, so the points in a scan are acquired at different instants. Without compensating for this motion distortion, registration accuracy suffers significantly.

Compensation method: If the pose change $\mathbf{T}_{s \to e}$ between the scan start $t_s$ and scan end $t_e$ is known, we estimate the intermediate pose at each point's timestamp $t_k$ by constant-velocity interpolation:

$$\mathbf{T}(t_k) = \text{Exp}\left(\frac{t_k - t_s}{t_e - t_s} \cdot \text{Log}(\mathbf{T}_{s \to e})\right)$$

Each point is then transformed to a reference time (typically the start of the scan):

$$\mathbf{p}_k^{\text{corrected}} = \mathbf{T}(t_k)^{-1} \cdot \mathbf{p}_k$$

When an IMU is available, IMU integration provides more accurate intermediate poses.

```python
# LOAM core pipeline pseudocode
class LOAM:
    def __init__(self):
        self.map_edge = VoxelMap(resolution=0.2)
        self.map_planar = VoxelMap(resolution=0.4)
        self.T_odom = np.eye(4)     # accumulated odometry transformation
        self.T_map = np.eye(4)      # mapping correction transformation
    
    def extract_features(self, scan):
        """Extract edge/planar features from a scan"""
        edge_points = []
        planar_points = []
        
        for scan_line in scan.lines:
            curvatures = []
            for i in range(5, len(scan_line) - 5):
                # Sum of vectors to 5 neighbors on each side, normalized by range
                diff = sum(scan_line[j] - scan_line[i] for j in range(i-5, i+6) if j != i)
                c = np.linalg.norm(diff) / (10 * np.linalg.norm(scan_line[i]))
                curvatures.append((c, i))
            
            # Split the scan line into 4 sectors for uniform extraction
            for sector in split_into_4(curvatures):
                sector.sort(reverse=True)  # sort by descending curvature
                
                n_edge, n_planar = 0, 0
                for c, idx in sector:
                    if c > EDGE_THRESH and n_edge < 2:
                        if not near_selected(idx, edge_points):
                            edge_points.append(scan_line[idx])
                            n_edge += 1
                
                sector.sort()  # sort by ascending curvature
                for c, idx in sector:
                    if c < PLANAR_THRESH and n_planar < 4:
                        if not near_selected(idx, planar_points):
                            planar_points.append(scan_line[idx])
                            n_planar += 1
        
        return edge_points, planar_points
    
    def odometry(self, edge_curr, planar_curr, edge_prev, planar_prev):
        """Scan-to-scan matching (10 Hz)"""
        T_relative = np.eye(4)
        tree_edge = KDTree(edge_prev)
        tree_planar = KDTree(planar_prev)
        
        for iter in range(25):
            residuals = []
            jacobians = []
            
            # Edge point-to-edge residuals
            for p in edge_curr:
                p_t = apply_transform(T_relative, p)
                _, idx = tree_edge.query(p_t, k=2)
                a, b = edge_prev[idx[0]], edge_prev[idx[1]]
                
                d_e = point_to_line_distance(p_t, a, b)
                J_e = point_to_line_jacobian(T_relative, p, a, b)
                residuals.append(d_e)
                jacobians.append(J_e)
            
            # Planar point-to-plane residuals
            for p in planar_curr:
                p_t = apply_transform(T_relative, p)
                _, idx = tree_planar.query(p_t, k=3)
                a, b, c = planar_prev[idx[0]], planar_prev[idx[1]], planar_prev[idx[2]]
                
                d_p = point_to_plane_distance(p_t, a, b, c)
                J_p = point_to_plane_jacobian(T_relative, p, a, b, c)
                residuals.append(d_p)
                jacobians.append(J_p)
            
            # LM update
            delta = levenberg_marquardt_step(residuals, jacobians)
            T_relative = se3_exp(delta) @ T_relative
            
            if np.linalg.norm(delta) < 1e-6:
                break
        
        self.T_odom = self.T_odom @ T_relative
        return self.T_odom
    
    def mapping(self, edge_curr, planar_curr):
        """Scan-to-map registration (1 Hz)"""
        # Extract the local map around the current position
        local_edge_map = self.map_edge.get_points_near(self.T_odom[:3, 3], radius=50.0)
        local_planar_map = self.map_planar.get_points_near(self.T_odom[:3, 3], radius=50.0)
        
        # Scan-to-map optimization (similar to odometry but against the map)
        T_correction = optimize_scan_to_map(edge_curr, planar_curr, 
                                            local_edge_map, local_planar_map)
        self.T_map = T_correction @ self.T_odom
        
        # Update the map
        self.map_edge.add_points(apply_transform(self.T_map, edge_curr))
        self.map_planar.add_points(apply_transform(self.T_map, planar_curr))
```

### 7.2.2 LeGO-LOAM

LeGO-LOAM ([Shan & Englot, 2018](https://doi.org/10.1109/IROS.2018.8594299)) adds ground segmentation to LOAM and trims the computation to achieve real-time operation even on embedded systems such as the Jetson TX2.

**Key additions in LeGO-LOAM**:

1. **Ground segmentation**: The point cloud is converted to a range image, and ground points are separated from non-ground points. A point is classified as ground if the slope between adjacent beams is below 10°. Ground points are used as planar features, and edge features are extracted from non-ground points.

2. **Point cloud segmentation**: On the non-ground points, range-image-based clustering is performed. Clusters below a certain size are discarded as noise. This preprocessing significantly improves feature quality compared to LOAM.

3. **Two-stage LM optimization**: Whereas LOAM optimizes all 6-DoF at once, LeGO-LOAM first estimates $[t_z, \theta_{\text{roll}}, \theta_{\text{pitch}}]$ from ground planar features and then $[t_x, t_y, \theta_{\text{yaw}}]$ from edge features. This decomposition improves convergence speed and stability.

4. **Pose graph optimization**: LeGO-LOAM adds loop closure and pose graph optimization, which were absent in LOAM, to correct global drift.

### 7.2.3 Why the LOAM Family Has Endured

Over ten years have passed since LOAM was published in 2014, yet the LOAM-family ideas remain the mainstream of LiDAR odometry. The reasons:

1. **Geometric clarity**: Edge and planar features correspond to physically meaningful geometric primitives. This structured-environment assumption fits most man-made environments well.

2. **Computational efficiency**: Using hundreds to thousands of feature points instead of the full point cloud (tens to hundreds of thousands of points) makes the system fast.

3. **Extensibility**: IMU (LIO-SAM), GPU (KISS-ICP), camera (LVI-SAM), and other modules can be plugged into the base framework.

4. **Robustness**: The edge/planar classification acts as a kind of outlier filter — points belonging to noise or dynamic objects do not show consistent curvature patterns and are naturally excluded.

That said, the limitations of the LOAM family are equally clear. Performance degrades in environments lacking geometric features (open fields, long tunnels), and the existing feature extraction is not well suited to the non-repetitive scan patterns of solid-state LiDARs. The direct approach of FAST-LIO2 addresses these limitations.

---

## 7.3 Tightly-Coupled LiDAR-Inertial Odometry

With LiDAR alone, motion distortion becomes severe under fast motion, and providing a good initial guess is difficult. A tightly coupled combination with an IMU overcomes these limitations.

### 7.3.1 LIO-SAM

LIO-SAM ([Shan et al., 2020](https://arxiv.org/abs/2007.00258)) is a LIO system that integrates LiDAR, IMU, GPS, and loop closure on top of a factor graph framework. It combines the LOAM-family feature-based approach with modern graph optimization.

**Factor-graph-based integration**

The core innovation of LIO-SAM is modeling the various sensor measurements as factors of a factor graph:

1. **IMU Preintegration Factor**: On-manifold preintegration from Forster et al. (2017) expresses the IMU constraint between consecutive keyframes. This factor provides constraints on the relative rotation, velocity, and position between two keyframes, along with bias estimation.

2. **LiDAR Odometry Factor**: LOAM-style scan-to-map matching estimates the relative pose. The result is inserted as a relative-pose factor between two keyframes.

3. **GPS Factor**: When GPS is available, the position measurement is added as a unary factor. In segments without GPS, this factor is absent, so the system naturally falls back to LiDAR+IMU only.

4. **Loop Closure Factor**: Place recognition (e.g., Scan Context) detects a loop, ICP estimates the relative pose, and the result is added as a binary factor.

All of these factors enter a single graph, and GTSAM's iSAM2 performs incremental optimization. The strength of the factor graph is its **modularity** — each sensor can independently add or remove factors, and adding a new sensor is straightforward.

**IMU-based de-skewing**

In LIO-SAM the IMU plays two roles:
1. **Motion distortion compensation**: IMU data during the LiDAR scan is used to precisely interpolate the per-timestamp pose of each point for de-skewing.
2. **Initial guess**: IMU preintegration predicts the pose of the next keyframe, which is used as the initial guess for scan-to-map registration.

This bidirectional coupling is the essence of the "tightly-coupled" nature of LIO-SAM: the IMU provides the LiDAR with the initial guess and de-skewing, while the LiDAR provides the IMU with pose correction and bias estimation.

**Keyframe-based efficiency**

Instead of a global map, scan matching is performed against a submap observed by the keyframes around the current position. This sliding-window approach significantly reduces computation compared to the global map.

```python
# LIO-SAM factor graph construction pseudocode
import gtsam

class LIOSAM:
    def __init__(self):
        self.graph = gtsam.NonlinearFactorGraph()
        self.values = gtsam.Values()
        self.isam = gtsam.ISAM2()
        self.key_idx = 0
    
    def add_keyframe(self, lidar_scan, imu_data, gps_data=None):
        # 1. IMU Preintegration Factor
        preint = gtsam.PreintegratedImuMeasurements(self.imu_params, self.current_bias)
        for imu in imu_data:
            preint.integrateMeasurement(imu.acc, imu.gyro, imu.dt)
        
        imu_factor = gtsam.ImuFactor(
            X(self.key_idx - 1), V(self.key_idx - 1),
            X(self.key_idx), V(self.key_idx),
            B(self.key_idx - 1), preint
        )
        self.graph.add(imu_factor)
        
        # 2. LiDAR Odometry Factor
        # De-skewing with IMU
        deskewed_scan = self.deskew(lidar_scan, imu_data)
        
        # Feature extraction (LOAM-style)
        edge_pts, planar_pts = extract_features(deskewed_scan)
        
        # Scan-to-submap matching
        T_lidar = scan_to_map_match(edge_pts, planar_pts, self.local_map)
        
        lidar_factor = gtsam.BetweenFactorPose3(
            X(self.key_idx - 1), X(self.key_idx),
            T_lidar, self.lidar_noise
        )
        self.graph.add(lidar_factor)
        
        # 3. GPS Factor (if available)
        if gps_data is not None:
            gps_factor = gtsam.GPSFactor(
                X(self.key_idx), gps_data.position, self.gps_noise
            )
            self.graph.add(gps_factor)
        
        # 4. Loop Closure Factor
        loop_candidate = self.detect_loop(deskewed_scan)
        if loop_candidate is not None:
            T_loop = icp_align(deskewed_scan, loop_candidate.scan)
            loop_factor = gtsam.BetweenFactorPose3(
                X(loop_candidate.key_idx), X(self.key_idx),
                T_loop, self.loop_noise
            )
            self.graph.add(loop_factor)
        
        # Initial value (from IMU prediction)
        T_predict = preint.predict(self.current_state, self.current_bias)
        self.values.insert(X(self.key_idx), T_predict.pose())
        self.values.insert(V(self.key_idx), T_predict.velocity())
        self.values.insert(B(self.key_idx), self.current_bias)
        
        # 5. iSAM2 incremental update
        result = self.isam.update(self.graph, self.values)
        self.graph.resize(0)
        self.values.clear()
        
        self.key_idx += 1
```

### 7.3.2 FAST-LIO / FAST-LIO2

FAST-LIO2 ([Xu et al., 2022](https://doi.org/10.1109/TRO.2022.3141855)) takes a completely different approach from the LOAM family. It eliminates feature extraction and is a direct LiDAR-inertial odometry that registers raw LiDAR points directly to the map.

**Three core innovations**:

**1. Direct point registration (no feature extraction)**

Unlike LOAM, which extracts edge/planar features, FAST-LIO2 uses every raw point directly. For each point $\mathbf{p}_k$ it finds the closest plane in the map and minimizes the point-to-plane distance:

$$d_k = \mathbf{n}_k^T (\mathbf{T} \cdot \mathbf{p}_k - \mathbf{q}_k)$$

where $\mathbf{n}_k$ is the normal of the closest plane in the map and $\mathbf{q}_k$ is the closest point.

Why eliminate feature extraction?
- Feature extraction is information loss — useful points can be discarded depending on the classification threshold.
- It applies generically across diverse LiDAR scan patterns (spinning, solid-state, non-repetitive). In particular, solid-state LiDARs such as Livox use non-repetitive scanning, which is poorly suited to the existing curvature-based feature extraction.
- With a sufficiently efficient map data structure (ikd-Tree), raw-point registration can be performed in real time.

**2. ikd-Tree (Incremental k-d Tree)**

The second key innovation of FAST-LIO2 is the map data structure. A conventional kd-tree is static and inefficient for point insertion/deletion. The ikd-Tree supports:

- **Point insertion**: insert a new point in $O(\log N)$ time.
- **Point deletion**: efficient removal of points outside the map region via lazy deletion.
- **Dynamic re-balancing**: when insertions/deletions unbalance the tree, partial rebuilding in a scapegoat-tree style restores balance.
- **Box-range deletion**: points in regions far from the current position are deleted in box units to manage map size.

Thanks to the ikd-Tree, FAST-LIO2 maintains the map in real time while also performing fast nearest-neighbor search.

**3. Iterated Extended Kalman Filter (IEKF)**

FAST-LIO2 uses a filter-based approach (IEKF), not an optimization-based one (as in LIO-SAM).

A standard EKF linearizes the observation model only once; when the LiDAR observation is highly nonlinear, this linearization is inaccurate. IEKF repeats the update several times to improve the linearization point:

$$\hat{\mathbf{x}}^{(k+1)} = \hat{\mathbf{x}}^{-} + \mathbf{K}^{(k)} (\mathbf{z} - h(\hat{\mathbf{x}}^{(k)}) - \mathbf{H}^{(k)}(\hat{\mathbf{x}}^{-} - \hat{\mathbf{x}}^{(k)}))$$

where $k$ is the iteration index, $\hat{\mathbf{x}}^{-}$ is the prediction, and $\mathbf{H}^{(k)}$ is the Jacobian at $\hat{\mathbf{x}}^{(k)}$.

Kalman gain:
$$\mathbf{K}^{(k)} = \mathbf{P}^{-} (\mathbf{H}^{(k)})^T (\mathbf{H}^{(k)} \mathbf{P}^{-} (\mathbf{H}^{(k)})^T + \mathbf{R})^{-1}$$

IEKF typically converges in 3-5 iterations. It achieves an effect similar to Gauss-Newton optimization while retaining the filter's advantage of naturally propagating uncertainty (covariance).

**State vector**:

$$\mathbf{x} = [{}^G\mathbf{R}_I, {}^G\mathbf{p}_I, {}^G\mathbf{v}_I, \mathbf{b}_g, \mathbf{b}_a, {}^I\mathbf{R}_L, {}^I\mathbf{p}_L, \mathbf{g}]$$

In addition to the rotation ${}^G\mathbf{R}_I$, position ${}^G\mathbf{p}_I$, velocity ${}^G\mathbf{v}_I$, gyro bias $\mathbf{b}_g$, and accelerometer bias $\mathbf{b}_a$, the state also includes the LiDAR-IMU extrinsics ${}^I\mathbf{R}_L, {}^I\mathbf{p}_L$ and the gravity vector $\mathbf{g}$. In other words, the extrinsic calibration and the gravity direction are also estimated online.

**Performance of FAST-LIO2**: Up to 100 Hz odometry + mapping is achieved in outdoor environments. It runs on multi-line spinning LiDARs, solid-state LiDARs (Livox), UAV/handheld platforms, and Intel/ARM processors alike.

```cpp
// FAST-LIO2 IEKF update pseudocode (C++)
struct State {
    Matrix3d R_GI;    // world-to-IMU rotation
    Vector3d p_GI;    // IMU position in world
    Vector3d v_GI;    // IMU velocity in world
    Vector3d bg, ba;  // gyro/accel bias
    Matrix3d R_IL;    // IMU-to-LiDAR rotation
    Vector3d p_IL;    // IMU-to-LiDAR translation
    Vector3d gravity; // gravity vector
};

void FASTLIO2::iterated_ekf_update(const PointCloud& scan, State& x, MatrixXd& P) {
    State x_predict = x;  // keep the prediction result
    MatrixXd K;  // reuse the Kalman gain of the final iteration outside the loop
    MatrixXd H;  // reuse the Jacobian of the final iteration outside the loop
    int n_valid = 0;
    
    for (int iter = 0; iter < MAX_ITER; iter++) {
        // 1. Transform points to the world frame using the current state
        PointCloud world_pts = transform_to_world(scan, x);
        
        // 2. For each point, find the nearest plane in the ikd-tree
        vector<Plane> planes = ikd_tree.find_nearest_planes(world_pts, k=5);
        
        // 3. Compute the observation Jacobians and residuals
        n_valid = 0;
        H.resize(scan.size(), STATE_DIM);
        VectorXd z(scan.size());
        
        for (int i = 0; i < scan.size(); i++) {
            if (!planes[i].valid) continue;
            
            Vector3d p_w = x.R_GI * (x.R_IL * scan[i] + x.p_IL) + x.p_GI;
            
            // Point-to-plane residual: the observation is 0 (point should lie on the plane)
            // z = 0 - h(x) = -d_k
            z(n_valid) = -planes[i].normal.dot(p_w - planes[i].center);
            
            // Jacobian: d(residual) / d(state_error)
            // ∂z/∂δθ_GI = n^T * [-(R_GI(R_IL*p + p_IL))×]
            // ∂z/∂δp_GI = n^T
            // ∂z/∂δθ_IL = n^T * R_GI * [-(R_IL*p)×]
            // ∂z/∂δp_IL = n^T * R_GI
            H.row(n_valid) = compute_jacobian(x, scan[i], planes[i]);
            n_valid++;
        }
        
        H.conservativeResize(n_valid, STATE_DIM);
        z.conservativeResize(n_valid);
        
        // 4. IEKF update
        MatrixXd S = H * P * H.transpose() + R_meas * MatrixXd::Identity(n_valid, n_valid);
        K = P * H.transpose() * S.inverse();
        
        VectorXd dx = K * (z - H * state_difference(x_predict, x));
        
        // 5. State correction (on-manifold): x^{(k+1)} = x^{-} ⊞ dx
        x = state_plus(x_predict, dx);
        
        // Check convergence
        if (dx.norm() < CONVERGENCE_THRESH) break;
    }
    
    // Covariance update
    MatrixXd I_KH = MatrixXd::Identity(STATE_DIM, STATE_DIM) - K * H;
    P = I_KH * P * I_KH.transpose() + K * R_meas * K.transpose();
    
    // Map update: insert the registered points into the ikd-tree
    PointCloud aligned = transform_to_world(scan, x);
    ikd_tree.insert(aligned);
}
```

### 7.3.3 Faster-LIO

Faster-LIO replaces the ikd-Tree of FAST-LIO2 with an incremental voxel structure to achieve even faster processing.

The key change: instead of a kd-tree, a hash-map-based voxel structure is used. A plane is maintained within each voxel, and the plane parameters are updated incrementally whenever a point is added. The $O(\log N)$ kd-tree search is replaced by $O(1)$ hash access, boosting speed.

### 7.3.4 Point-LIO

Point-LIO ([He et al., 2023](https://doi.org/10.1002/aisy.202200459)) is an extreme extension of the FAST-LIO series. It updates the state at the granularity of **individual points** rather than scans.

**Why per-point processing?**

Conventional LIO treats an entire scan (~100 ms) as a single observation. During that interval, motion distortion is corrected by constant-velocity interpolation, but under fast/high-angular-rate motion the constant-velocity assumption breaks down.

Point-LIO performs an EKF update as soon as each point arrives (on the order of $\sim$μs). Using the IMU's high-rate (~1 kHz) measurements together with the LiDAR point timestamps, it uses the IMU state that exactly corresponds to each point.

**Mathematical core**: Point-LIO's state propagation model, over the short interval between IMU measurements, discretizes

$$\frac{d}{dt}\mathbf{R} = \mathbf{R}[\boldsymbol{\omega}]_\times, \quad \frac{d}{dt}\mathbf{v} = \mathbf{R}\mathbf{a} + \mathbf{g}, \quad \frac{d}{dt}\mathbf{p} = \mathbf{v}$$

Whenever a point arrives, it performs state propagation followed by a single-point update, effectively approximating a continuous-time filter.

Advantages: Accurate odometry even under extremely fast motion (hundreds of degrees per second of rotation). Motion distortion compensation happens implicitly (each point is already processed with the correct per-timestamp state).

Disadvantages: The number of updates scales with the number of points, increasing the computational burden. It is about 2-3 times slower than FAST-LIO2.

### 7.3.5 COIN-LIO

[COIN-LIO (Pfreundschuh et al., 2024)](https://arxiv.org/abs/2310.01235) adds **camera intensity** information to a LiDAR-Inertial system. Coupling a camera into a traditional LIO typically requires a separate visual feature extraction/tracking stack, but COIN-LIO takes a simpler approach:

It records the brightness (intensity) of the camera pixel corresponding to each LiDAR point and assigns an intensity to each map point as well. During registration, the intensity difference is included in the cost together with the geometric distance (point-to-plane):

$$e_k = \alpha \cdot d_{\text{geom}}(\mathbf{p}_k) + (1-\alpha) \cdot |I_{\text{obs}}(\mathbf{p}_k) - I_{\text{map}}(\mathbf{p}_k)|$$

In geometrically degenerate environments — for example long tunnels or empty halls — the intensity information provides additional constraints and preserves accuracy. It is a pragmatic approach that exploits the camera's texture information while avoiding the complexity of a full VIO pipeline.

---

## 7.4 Continuous-Time LiDAR Odometry

Existing LiDAR odometry uses a discrete-time model. It assigns one pose per scan and approximates intra-scan motion by constant-velocity interpolation. The continuous-time approach overcomes this limitation by representing the trajectory as a continuous function.

### 7.4.1 CT-ICP

CT-ICP ([Dellenbach et al., 2022](https://arxiv.org/abs/2109.12979)) assigns not one but **two poses** per scan (at the start and end of the scan).

For each point's timestamp $t_k \in [t_s, t_e]$ within the scan, the pose is linearly interpolated:

$$\mathbf{T}(t_k) = \mathbf{T}_s \cdot \text{Exp}\left(\frac{t_k - t_s}{t_e - t_s} \cdot \text{Log}(\mathbf{T}_s^{-1}\mathbf{T}_e)\right)$$

The two poses $\mathbf{T}_s, \mathbf{T}_e$ are optimized jointly. Unlike constant-velocity interpolation, the intra-scan motion model is refined along with the optimization.

CT-ICP effectively compensates for motion distortion without an IMU, making it especially useful for IMU-less systems.

### 7.4.2 B-Spline-Based Trajectory Representation

A more general continuous-time approach represents the trajectory with a B-spline. A B-spline is a smooth curve defined by control points $\{\mathbf{T}_i\}$:

$$\mathbf{T}(t) = \prod_{i=0}^{k} \text{Exp}\left(B_i(t) \cdot \text{Log}(\mathbf{T}_{i-1}^{-1}\mathbf{T}_i)\right)$$

where $B_i(t)$ is a B-spline basis function. Cubic B-splines are commonly used and guarantee $C^2$ continuity.

Advantages of a B-spline trajectory:
1. **Query at arbitrary times**: At any time $t$, the pose, velocity, and acceleration can be obtained via differentiation. This enables natural handling of asynchronous sensor data.
2. **Smooth trajectory**: A physically plausible smooth trajectory is guaranteed.
3. **Local control**: Thanks to the locality of B-splines, modifying one control point does not affect the entire trajectory.

Disadvantages:
- The control-point (knot) spacing is the primary hyperparameter. Too dense causes overfitting; too sparse fails to represent fast motion.
- Computation increases compared to the discrete-time case.

The camera-IMU calibration in Kalibr (see Ch.3) also uses a B-spline trajectory representation.

---

## 7.5 Solid-State LiDAR Specifics

Solid-state LiDARs (e.g., the Livox series) have a fundamentally different scan pattern from spinning LiDARs.

**Spinning vs solid-state**:

| Property | Spinning (Velodyne, Ouster) | Solid-state (Livox) |
|------|--------------------------|---------------------|
| Scan pattern | Repetitive (same pattern every rotation) | Non-repetitive (petal / rose pattern) |
| FoV | 360° horizontal | Limited (70-77°) |
| Point density | Uniform | Accumulates over time, non-uniform |
| Price | High | Low |
| Size / weight | Large | Small |

**Impact of non-repetitive scanning on feature extraction**

LOAM-style curvature-based feature extraction uses neighbors on the same scan line. However, solid-state LiDARs have no defined scan line and their points are distributed irregularly. Consequently:

1. The existing line-based curvature computation does not apply.
2. Instead, one must use KNN (K-Nearest Neighbors)-based local curvature, or abandon feature extraction altogether and use raw points.

**Why FAST-LIO is strong on solid-state**

FAST-LIO/FAST-LIO2 work well on solid-state LiDARs for three reasons:

1. **No feature extraction needed**: Using raw points directly, the system is agnostic to the scan pattern.
2. **Leverages the non-repetitive scan**: A solid-state LiDAR gradually fills the FoV more densely over time. FAST-LIO2's ikd-Tree map naturally accommodates this progressive densification, so map quality improves over time.
3. **Compensates for the narrow FoV**: A narrow FoV means less information per scan, but tight coupling with the IMU compensates for this.

The Livox series is rapidly spreading in drones, handheld devices, and small robots thanks to its strong price/performance, and the FAST-LIO2 + Livox combination is currently one of the most popular LIO configurations.

---

## 7.6 Learning-Based LiDAR Odometry

### 7.6.1 The DeepLO Family

Learning-based LiDAR odometry trains a network that takes a pair of point clouds as input and predicts the relative pose.

Representative approaches:
- **LO-Net** (Li et al., 2019): Converts a LiDAR scan to a 2D range image and uses a CNN to extract features and predict the pose. Normal estimation and mask prediction are added as auxiliary tasks to encourage geometric understanding.
- **DeepLO** (Cho et al., 2020): Predicts the pose by processing 3D point clouds directly using a PointNet backbone.
- **PWCLO-Net** (Wang et al., 2021): Applies the Pyramid, Warping, and Cost-volume architecture to LiDAR odometry.

### 7.6.2 Current Limitations

Learning-based LiDAR odometry still lags significantly behind classical methods. The reasons:

1. **Characteristics of LiDAR data**: Unlike images, point clouds are unstructured, sparse, and unordered. CNNs cannot process them naturally.

2. **Geometry is already sufficient**: Geometric methods such as ICP/GICP/NDT are already very accurate. The reasons learning shines in the camera domain — illumination changes, lack of texture, and other issues that are hard to solve with geometry alone — do not apply to LiDAR.

3. **Lack of data**: Large-scale LiDAR odometry training data is much scarcer than image data.

4. **Generalization**: A model trained on a particular LiDAR/environment does not generalize well to other LiDARs/environments.

Today, learning is more effective as auxiliary components than as LiDAR odometry itself:
- **Loop closure detection**: Scan Context, PointNetVLAD, etc.
- **Point cloud registration initialization**: GeoTransformer (see Ch.5)
- **Semantic segmentation**: dynamic-object removal, road/building classification

---

## 7.7 Recent Trends (2023-2024)

Beyond the systems discussed above, several recent studies in LiDAR odometry are worth noting.

**[KISS-ICP (Vizzo et al., 2023)](https://arxiv.org/abs/2209.15397)**: Shows that point-to-point ICP, combined only with adaptive thresholding, a robust kernel, and simple motion compensation, can achieve SOTA-level performance. The key is generality — it operates without tuning, regardless of the sensor type (automotive, UAV, handheld). The work reaffirms the "power of simplicity" in LiDAR odometry.

**[MAD-ICP (Ferrari et al., 2024)](https://arxiv.org/abs/2405.05828)**: Uses a PCA-based kd-tree to extract the structural information of the point cloud and applies it to point-to-plane registration. It emphasizes the importance of the data-matching strategy itself and matches the performance of domain-specific methods across a variety of LiDAR sensors.

**[iG-LIO (Chen et al., 2024)](https://github.com/zijiechenrobotics/ig_lio)**: A system that integrates incremental GICP into a tightly coupled LIO. A voxel-based surface covariance estimator (VSCE) improves the efficiency of GICP's covariance computation, and an incremental voxel map reduces the nearest-neighbor search cost. It is more efficient than Faster-LIO while retaining SOTA-level accuracy.

---

## Chapter 7 Summary

| System | Approach | Estimation | Sensors | Key feature |
|--------|------|-----------|------|-----------|
| ICP/GICP/NDT | Registration | Iterative optimization | LiDAR only | Fundamental building block |
| LOAM | Feature-based | LM optimization | LiDAR only | Edge/planar feature, two-stage architecture |
| LeGO-LOAM | Feature-based | LM optimization | LiDAR only | Ground segmentation, lightweight |
| LIO-SAM | Feature-based | Factor graph (iSAM2) | LiDAR + IMU + GPS | Modular multi-sensor integration |
| FAST-LIO2 | Direct | IEKF | LiDAR + IMU | No feature extraction, ikd-Tree, 100 Hz |
| Point-LIO | Direct | Point-wise EKF | LiDAR + IMU | Per-point update, fast motion |
| COIN-LIO | Direct + intensity | IEKF | LiDAR + IMU + camera (intensity) | Intensity-based degeneration mitigation |
| CT-ICP | Direct | Optimization | LiDAR only | Continuous-time motion model, no IMU needed |
| KISS-ICP | Direct (P2P) | Iterative optimization | LiDAR only | Adaptive threshold, tuning-free, general-purpose |
| MAD-ICP | Direct (P2Plane) | Iterative optimization | LiDAR only | PCA-based structural extraction, data-matching focus |
| iG-LIO | Direct (GICP) | IEKF | LiDAR + IMU | Incremental GICP, voxel covariance estimation |

**The big picture of LiDAR odometry**:

The LOAM (2014) → LeGO-LOAM (2018) → LIO-SAM (2020) lineage evolved in the direction of **feature-based + factor graph**. The FAST-LIO (2021) → FAST-LIO2 (2022) → Point-LIO (2023) lineage evolved in the direction of **direct + Kalman filter**. The two lineages represent different design philosophies, yet their practical performance is converging to similar levels.

The feature-based approach is strong in structured environments (buildings, cities), and the direct approach is strong in unstructured environments (forests, caves) and on solid-state LiDARs. In the next chapter we take up multi-sensor fusion architectures that integrate these two sensor modalities (camera + LiDAR) together with an IMU.
