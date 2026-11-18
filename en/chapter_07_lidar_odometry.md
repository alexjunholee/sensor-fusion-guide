# Chapter 7. LiDAR Odometry and LIO

Swapping passive cameras for active LiDAR transforms the ego-motion problem from the ground up. Where the visual pipelines of Chapter 6 infer 3D structure indirectly through perspective projection, image gradients, and parallax—making monocular scale unobservable and leaving tracking exposed to lighting swings—LiDAR measures 3D surface geometry directly by firing laser pulses and timing their returns. Metric range arrives out of the box with far less sensitivity to ambient illumination, though rain, fog, extreme surface reflectivity, and multipath reflections still degrade returns.

Estimating motion from those 3D sweeps defines **LiDAR Odometry (LO)**, while tightly coupling them with an IMU forms **LiDAR-Inertial Odometry (LIO)**. At the heart of both lies **point cloud registration**: solving for the rigid-body transform $\mathbf{T} \in SE(3)$ that aligns one sweep onto another. Because LiDAR points are sparse, unordered, and acquired continuously while the sensor moves, reliable registration hinges on four coupled concerns: data association, local surface noise modeling, computational efficiency, and intra-scan motion distortion compensation.

---

## 7.1 Point Cloud Registration Fundamentals

Given a source point cloud $\mathcal{P} = \{\mathbf{p}_i\}$ and a target point cloud $\mathcal{Q} = \{\mathbf{q}_j\}$, point cloud registration seeks the rigid-body transformation $\mathbf{T} \in SE(3)$ that minimizes an aggregate alignment residual:

$$\mathbf{T}^* = \underset{\mathbf{T} \in SE(3)}{\arg\min} \sum_i d(\mathbf{T} \cdot \mathbf{p}_i, \mathcal{Q})$$

How we define the point-to-cloud distance $d(\cdot, \cdot)$—whether between isolated points, points and tangent planes, local surface ellipsoids, or voxel-wise Gaussian densities—gives rise to the classic registration family below.

### 7.1.1 ICP Variants

**Point-to-Point ICP ([Besl & McKay, 1992](https://doi.org/10.1109/34.121791))**

The foundational Iterative Closest Point (ICP) algorithm measures Euclidean distance between each transformed source point and its nearest neighbor in the target cloud:

$$\mathbf{T}^* = \underset{\mathbf{T}}{\arg\min} \sum_i \left\|\mathbf{T} \cdot \mathbf{p}_i - \mathbf{q}_{c(i)}\right\|^2$$

where $c(i) = \arg\min_j \|\mathbf{T} \cdot \mathbf{p}_i - \mathbf{q}_j\|$ assigns closest-point correspondences. Because true point correspondences are unknown beforehand, ICP alternates between two steps until convergence:

1. **Correspondence search**: Transform every source point using the current pose estimate and look up its nearest neighbor in $\mathcal{Q}$, typically in $O(N\log N)$ time via a kd-tree.
2. **Closed-form alignment via SVD**: Holding those correspondences fixed, the optimal rigid-body transform has an exact closed-form solution. Subtract the two point-set centroids,

   $$\bar{\mathbf{p}} = \frac{1}{N}\sum_i \mathbf{p}_i, \quad \bar{\mathbf{q}} = \frac{1}{N}\sum_i \mathbf{q}_{c(i)}$$

   assemble the cross-covariance matrix,

   $$\mathbf{W} = \sum_i (\mathbf{p}_i - \bar{\mathbf{p}})(\mathbf{q}_{c(i)} - \bar{\mathbf{q}})^T$$

   take its Singular Value Decomposition $\mathbf{W} = \mathbf{U}\boldsymbol{\Sigma}\mathbf{V}^T$, and read off the optimal rotation and translation:

   $$\mathbf{R}^* = \mathbf{V} \text{diag}(1, 1, \det(\mathbf{V}\mathbf{U}^T)) \mathbf{U}^T, \quad \mathbf{t}^* = \bar{\mathbf{q}} - \mathbf{R}^*\bar{\mathbf{p}}$$

   Here $\det(\mathbf{V}\mathbf{U}^T) = 1$ yields $\mathbf{R}^* = \mathbf{V}\mathbf{U}^T$ directly, whereas $\det(\mathbf{V}\mathbf{U}^T) = -1$ flips the last column of $\mathbf{V}$ to rule out an improper reflection.

Despite its elegance, point-to-point ICP runs into three practical pitfalls. First, on flat surfaces it exhibits **plane sliding**: while a tangential shift does raise the fixed-correspondence objective, the very next nearest-neighbor search re-associates each source point to a freshly adjacent target point, effectively flattening the cost landscape along the plane and slowing convergence. Second, its alternating search settles easily into local minima when initial alignment is poor. Third, discrete scans sampled from slightly different viewpoints rarely hit identical physical points, so the nearest neighbor is often only an approximate correspondence.

**Point-to-Plane ICP**

When scans capture walls, floors, or facades, penalizing distance along the local surface normal captures the underlying geometry far better than point-to-point proximity:

$$\mathbf{T}^* = \underset{\mathbf{T}}{\arg\min} \sum_i \left((\mathbf{T} \cdot \mathbf{p}_i - \mathbf{q}_{c(i)})^T \mathbf{n}_{c(i)}\right)^2$$

where $\mathbf{n}_{c(i)}$ is the surface normal at target point $\mathbf{q}_{c(i)}$. Projecting residuals onto $\mathbf{n}_{c(i)}$ ignores tangential offset along the surface so planes can glide smoothly into place, yielding noticeably faster convergence than point-to-point ICP in structured indoor and urban scenes. The trade-off is that point-to-plane ICP lacks a closed-form SVD step—requiring iterative nonlinear optimization such as Gauss-Newton—and its accuracy hinges on reliable normal estimates.

Normals are typically estimated via local Principal Component Analysis (PCA) over each point's $k$-nearest neighborhood $\mathcal{N}(i)$:

$$\mathbf{C} = \frac{1}{k}\sum_{j \in \mathcal{N}(i)} (\mathbf{q}_j - \bar{\mathbf{q}})(\mathbf{q}_j - \bar{\mathbf{q}})^T$$

In the eigendecomposition $\mathbf{C} = \mathbf{V}\boldsymbol{\Lambda}\mathbf{V}^T$, points spreading across a flat surface exhibit large variance in two tangent directions and minimal variance perpendicular to the surface, so the eigenvector corresponding to $\lambda_{\min}$ defines the local surface normal.

### 7.1.2 GICP (Generalized ICP)

GICP ([Segal et al., 2009](https://doi.org/10.15607/RSS.2009.V.021)) folds point-to-point, point-to-plane, and plane-to-plane registration into a single probabilistic model. Rather than treating scans as dimensionless points or infinitely flat planes, GICP models every point as drawn from a local Gaussian distribution whose covariance $\mathbf{C}_i$ reflects local surface geometry, minimizing the Mahalanobis distance across combined covariances:

$$\mathbf{T}^* = \underset{\mathbf{T}}{\arg\min} \sum_i (\mathbf{T} \cdot \mathbf{p}_i - \mathbf{q}_{c(i)})^T (\mathbf{C}_i^{\mathcal{Q}} + \mathbf{R}\mathbf{C}_i^{\mathcal{P}}\mathbf{R}^T)^{-1} (\mathbf{T} \cdot \mathbf{p}_i - \mathbf{q}_{c(i)})$$

where $\mathbf{C}_i^{\mathcal{P}}$ and $\mathbf{C}_i^{\mathcal{Q}}$ denote the local surface covariances at the source and target points.

Geometrically, shaping these covariances recovers each classic ICP variant as a special case:

- On a planar surface, uncertainty is tiny along the normal and broad across the tangent plane, giving $\mathbf{C} = \mathbf{R}_s \text{diag}(\epsilon, 1, 1) \mathbf{R}_s^T$ with $\epsilon \ll 1$, where the rotation matrix $\mathbf{R}_s$ aligns its first column with the normal ($\mathbf{R}_s \mathbf{e}_1 = \mathbf{n}$). Because the eigenvectors of $\mathbf{C}$ are the columns of $\mathbf{R}_s$, placing $\mathbf{n}$ in the first column assigns variance $\epsilon$ strictly along the normal, naturally yielding plane-to-plane registration.
- Setting $\mathbf{C}^{\mathcal{P}} = \mathbf{0}$ (with planar $\mathbf{C}^{\mathcal{Q}}$) collapses the metric to point-to-plane ICP, while setting $\mathbf{C}^{\mathcal{P}} = \mathbf{C}^{\mathcal{Q}} = \mathbf{I}$ recovers standard point-to-point ICP.

In its paper benchmarks, GICP demonstrates stronger accuracy and resilience to correspondence errors than both point-to-point and point-to-plane baselines—though real-world performance still depends on pose initialization, scan overlap, sampling density, and outlier filtering.

The Python sketch below walks through a single GICP loop—precomputing local surface covariances, querying a target kd-tree, and accumulating rotated covariance-weighted Gauss-Newton updates on $SE(3)$:

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

Rather than matching discrete points via a kd-tree, [NDT (Biber & Strasser, 2003)](https://doi.org/10.1109/IROS.2003.1249285) represents the target scan as a spatial field of smooth local Gaussians. First, the target space is partitioned into a 3D voxel grid, and the points inside each occupied voxel $k$ are summarized by a Gaussian $\mathcal{N}(\boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k)$:

$$\boldsymbol{\mu}_k = \frac{1}{n_k}\sum_{i \in k} \mathbf{q}_i, \quad \boldsymbol{\Sigma}_k = \frac{1}{n_k-1}\sum_{i \in k} (\mathbf{q}_i - \boldsymbol{\mu}_k)(\mathbf{q}_i - \boldsymbol{\mu}_k)^T$$

Registration then maximizes the summed Gaussian likelihood of the transformed source points landing inside their corresponding target voxels:

$$\mathbf{T}^* = \underset{\mathbf{T}}{\arg\max} \sum_i \exp\left(-\tfrac{1}{2}(\mathbf{T} \cdot \mathbf{p}_i - \boldsymbol{\mu}_{k(i)})^T \boldsymbol{\Sigma}_{k(i)}^{-1} (\mathbf{T} \cdot \mathbf{p}_i - \boldsymbol{\mu}_{k(i)})\right)$$

Notice a subtle algebraic property: because NDT sums the raw Gaussian exponentials *without* taking a logarithm, points with huge residuals contribute nearly zero gradient and naturally saturate. Switching to a sum of log-likelihoods—minimizing the quadratic Mahalanobis sum

$$\sum_i (\mathbf{T} \cdot \mathbf{p}_i - \boldsymbol{\mu}_{k(i)})^T \boldsymbol{\Sigma}_{k(i)}^{-1} (\mathbf{T} \cdot \mathbf{p}_i - \boldsymbol{\mu}_{k(i)})$$

produces a different algorithm (voxel-based Mahalanobis ICP) where outlier leverage grows unboundedly. Practical NDT implementations preserve bounded outlier influence by mixing a uniform background distribution into each voxel Gaussian.

By replacing explicit nearest-neighbor queries with constant-time voxel lookups, NDT eliminates kd-tree traversal and naturally suppresses distant outliers through its smooth saturating score, while letting voxel resolution trade convergence basin width against final precision. On the flip side, accuracy is sensitive to voxel size, sparse voxels yield ill-conditioned covariances, and while 3D NDT excels at prior-map localization in autonomous driving stacks such as Autoware's `ndt_scan_matcher`, scan-to-scan odometry studies often report slightly lower precision than ICP or GICP depending on the dataset.

### 7.1.4 Convergence and Initialization Dependence

Every ICP and NDT variant is a local optimizer: start outside its basin of convergence, and it will lock onto the wrong local minimum. Odometry pipelines therefore seed registration using one of four initialization strategies:

1. **Constant-velocity model**: extrapolate relative motion from the previous two scans—simple and sensor-free, though brittle under sudden accelerations or turns.
2. **IMU integration**: integrate high-rate inertial measurements across the inter-scan interval to deliver a tight initial pose prior, a primary reason LIO outperforms pure LO during aggressive maneuvers.
3. **Multi-resolution pyramids**: align coarse downsampled clouds or large voxels first to widen the convergence basin, then tighten resolution iteratively.
4. **RANSAC with 3D descriptors**: match geometric descriptors such as FPFH or SHOT inside RANSAC for global initialization—essential for loop closure or kidnapped-robot relocalization, though slower than temporal motion priors in frame-to-frame odometry.

---

## 7.2 Feature-based LiDAR Odometry

Registering tens of thousands of raw points per sweep at 10–20 Hz quickly saturates onboard compute—especially when most points simply lie redundantly on the same flat walls or road surfaces. Feature-based LiDAR odometry mirrors feature-based VO: extract only geometrically distinctive primitives, match them to lines and planes, and reserve dense map alignment for a slower background thread.

### 7.2.1 LOAM (Lidar Odometry and Mapping in Real-time)

[LOAM (Zhang & Singh, 2014)](https://doi.org/10.15607/RSS.2014.X.007) defined the modern blueprint for LiDAR odometry, topping the KITTI odometry benchmark for years and spawning derivatives including A-LOAM, LeGO-LOAM, and LIO-SAM.

Its enduring success rests on two architectural ideas: compressing each sweep into geometric **edge and planar features** so registration operates on a fraction of the points, and **decoupling fast odometry (~10 Hz) from slower mapping (~1 Hz)** so the system stays real-time without sacrificing global map accuracy.

**Feature extraction**

Rather than computing full 3D neighborhoods, LOAM exploits the ring structure of a spinning LiDAR and evaluates 1D local smoothness (curvature) along each horizontal scan line:

$$c_i = \frac{1}{|\mathcal{S}_i| \cdot \|\mathbf{p}_i\|} \left\| \sum_{j \in \mathcal{S}_i} (\mathbf{p}_j - \mathbf{p}_i) \right\|$$

Here $\mathcal{S}_i$ collects left and right neighbors of $\mathbf{p}_i$ on the same scan ring (typically five on each side), and normalizing by range $\|\mathbf{p}_i\|$ keeps curvature values comparable across near and distant surfaces. Geometrically, if $\mathbf{p}_i$ lies on a flat surface, neighbor vectors on both sides cancel out and $c_i$ is small; at a sharp corner or pole, the neighbor sum points strongly off the line and $c_i$ spikes:

- **Edge features** ($c_i > c_{\text{thresh}}^e$): sharp corners, building edges, poles, and jump boundaries.
- **Planar features** ($c_i < c_{\text{thresh}}^p$): smooth walls, facades, and ground planes.

To keep features well distributed and geometrically stable, LOAM splits each scan line into four sectors, applies non-maximum suppression around already-chosen points, and filters out two unstable cases: surfaces nearly parallel to the laser beam—since grazing incidence (measured relative to the laser ray, not world horizontality) yields noisy range returns—and points straddling occlusion boundaries that vanish under small viewpoint shifts.

**Odometry module (~10 Hz)**

The high-rate odometry thread estimates frame-to-frame motion via scan-to-scan matching, assigning each feature type its natural geometric residual.

For an **edge point** $\mathbf{p}$ in the current scan, LOAM locates the nearest edge point $\mathbf{a}$ in the previous scan and a second edge point $\mathbf{b}$ on an adjacent scan ring (two points from the same ring cannot span a vertical edge). Geometrically, the area of the parallelogram spanned by $(\mathbf{p}-\mathbf{a})$ and $(\mathbf{p}-\mathbf{b})$ divided by the base $\|\mathbf{a}-\mathbf{b}\|$ gives the exact perpendicular distance from $\mathbf{p}$ to line $\overline{\mathbf{ab}}$:

$$d_e = \frac{\|(\mathbf{p}-\mathbf{a}) \times (\mathbf{p}-\mathbf{b})\|}{\|\mathbf{a}-\mathbf{b}\|}$$

For a **planar point** $\mathbf{p}$, LOAM finds three nearby planar points $\mathbf{a}, \mathbf{b}, \mathbf{c}$ in the previous scan (two on the same ring and one on an adjacent ring) to span a local triangle $\triangle\mathbf{abc}$. Taking the scalar triple product against the triangle's cross-product normal and dividing by that normal's length yields the perpendicular point-to-plane distance:

$$d_p = \frac{(\mathbf{p}-\mathbf{a})^T \left((\mathbf{a}-\mathbf{b}) \times (\mathbf{a}-\mathbf{c})\right)}{\|(\mathbf{a}-\mathbf{b}) \times (\mathbf{a}-\mathbf{c})\|}$$

Summing squared edge and planar residuals over the scan yields a joint Levenberg-Marquardt objective for the inter-scan transform:

$$\mathbf{T}^* = \underset{\mathbf{T}}{\arg\min} \sum_{\mathbf{p} \in \mathcal{E}} d_e(\mathbf{T}\cdot\mathbf{p})^2 + \sum_{\mathbf{p} \in \mathcal{P}} d_p(\mathbf{T}\cdot\mathbf{p})^2$$

**Mapping module (~1 Hz)**

Scan-to-scan odometry runs fast, yet pairwise errors inevitably accumulate. At roughly 1 Hz, the mapping thread registers the undistorted sweep against a voxel-downsampled local map cube centered around the current pose, correcting odometry drift and inserting newly aligned features into the global map.

**Motion Distortion Compensation**

Unlike a global-shutter camera that freezes an entire frame at one instant, a spinning LiDAR sweeps continuously—taking about 100 ms per full rotation at 10 Hz. As the sensor translates and turns during that interval, points at the start and end of the same "scan" are observed from different poses, skewing straight walls into curves unless deskewed.

Given the pose change $\mathbf{T}_{s \to e}$ between scan start $t_s$ and scan end $t_e$, constant-velocity $SE(3)$ interpolation estimates the sensor pose at each point's timestamp $t_k$:

$$\mathbf{T}(t_k) = \text{Exp}\left(\frac{t_k - t_s}{t_e - t_s} \cdot \text{Log}(\mathbf{T}_{s \to e})\right)$$

Transforming each point back through $\mathbf{T}(t_k)^{-1}$ aligns the entire sweep to a common reference time (typically $t_s$):

$$\mathbf{p}_k^{\text{corrected}} = \mathbf{T}(t_k)^{-1} \cdot \mathbf{p}_k$$

When an IMU is present, high-rate inertial integration replaces constant-velocity interpolation to handle rapid turns and accelerations far more accurately.

The Python sketch below summarizes LOAM's complete workflow across ring-wise feature extraction, 10 Hz scan-to-scan odometry, and 1 Hz scan-to-map refinement:

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
                _, idx = tree_edge.query(p_t, k=2)  # The real LOAM picks the two points from different scan lines
                a, b = edge_prev[idx[0]], edge_prev[idx[1]]
                
                d_e = point_to_line_distance(p_t, a, b)
                J_e = point_to_line_jacobian(T_relative, p, a, b)
                residuals.append(d_e)
                jacobians.append(J_e)
            
            # Planar point-to-plane residuals
            for p in planar_curr:
                p_t = apply_transform(T_relative, p)
                _, idx = tree_planar.query(p_t, k=3)  # The real LOAM uses two points on the same line + one on an adjacent line
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

Tailored for ground vehicles on embedded compute such as the Jetson TX2, [LeGO-LOAM (Shan & Englot, 2018)](https://doi.org/10.1109/IROS.2018.8594299) refines LOAM with four targeted upgrades:

1. **Ground segmentation**: Projecting each sweep into a 2D range image separates ground points—identified where the elevation angle between adjacent laser rings falls below 10°—as planar features, isolating non-ground points for edge extraction.
2. **Point-cloud clustering**: Connected-component clustering on the non-ground range image discards small noisy clusters (leaves, clutter) before selecting stable structural features in ground-vehicle scenes.
3. **Two-stage LM optimization**: Rather than solving all 6 DoF simultaneously, LeGO-LOAM first estimates $[t_z, \theta_{\text{roll}}, \theta_{\text{pitch}}]$ from ground planes and then $[t_x, t_y, \theta_{\text{yaw}}]$ from edge features, accelerating convergence and stabilizing vertical/attitude estimation.
4. **Loop closure via pose-graph optimization**: Adds loop detection and pose-graph optimization—absent from original LOAM—to bound global drift over long trajectories.

### 7.2.3 Why the LOAM Family Has Endured

More than a decade after LOAM's debut in 2014, its core concepts continue to anchor LiDAR odometry for four reasons. First, edge and planar primitives map cleanly onto the corners, poles, walls, and roads dominating man-made spaces (**geometric clarity**). Second, distilling hundreds of thousands of raw returns down to a few hundred or thousand features keeps registration lightweight (**computational efficiency**). Third, the frontend plugs naturally into inertial and visual estimators such as LIO-SAM and LVI-SAM (**extensibility**)—note that KISS-ICP belongs to a distinct lineage as a deliberately simple LiDAR-only ICP pipeline rather than a LOAM extension. Fourth, curvature screening acts as a natural filter against isolated noise and grazing-angle artifacts (**robustness**). Note, however, that single-scan curvature is purely geometric and blind to motion: corners and panels on a passing vehicle still score as crisp features, so filtering dynamic objects requires a separate stage.

At the same time, LOAM carries two clear boundaries. Because curvature is computed along fixed concentric scan rings, it does not carry over directly to solid-state LiDARs with non-repetitive scan patterns—a gap FAST-LIO2 bridges by eliminating feature extraction altogether. And in geometrically degenerate spaces such as flat open fields or long uniform tunnels, accuracy drops steeply—not from how features are selected, but because the environment itself lacks observation constraints along the travel direction, a challenge shared by feature-based and direct registration alike.

---

## 7.3 Tightly-Coupled LiDAR-Inertial Odometry

Pure LiDAR odometry buckles under aggressive motion: scans warp faster than constant-velocity deskewing can track, and rapid rotations deprive ICP or LOAM of an accurate initial pose guess. Tightly coupling an IMU solves both halves of the problem in one closed loop—inertial measurements deskew each sweep and seed registration, while LiDAR alignment calibrates IMU biases and caps inertial drift.

### 7.3.1 LIO-SAM

[LIO-SAM (Shan et al., 2020)](https://arxiv.org/abs/2007.00258) bridges classic LOAM-style feature registration with modern incremental factor-graph optimization, fusing LiDAR, IMU, optional GPS, and loop closures inside GTSAM.

**Factor-graph-based integration**

Rather than locking sensors into a rigid filter pipeline, LIO-SAM inserts each sensor modality as a factor connecting keyframe states inside a global factor graph solved incrementally via GTSAM's iSAM2:

1. **IMU Preintegration Factor**: Following the on-manifold preintegration of Forster et al. (2017), inertial increments between consecutive keyframes constrain relative pose, velocity, and IMU bias evolution.
2. **LiDAR Odometry Factor**: LOAM-style edge and planar registration against a local keyframe submap supplies a relative `BetweenFactorPose3` linking consecutive keyframes.
3. **GPS Factor**: Whenever GNSS fixes are available, position measurements enter as unary factors; during GPS outages, the factor is simply omitted and the estimator falls back seamlessly to LiDAR+IMU.
4. **Loop Closure Factor**: Place recognition (such as Scan Context) detects revisited locations, and ICP alignment inserts a binary relative-pose factor to eliminate accumulated drift.

Alongside this keyframe graph, LIO-SAM maintains a lightweight auxiliary graph for high-rate IMU odometry—periodically reset from the global optimization—to deliver IMU-rate pose propagation and responsive bias estimation. The payoff of this factor-graph architecture is **modularity**: sensors can enter or drop out asynchronously without rewriting the core estimator.

**IMU-based de-skewing**

Information flows in both directions between the LiDAR and IMU. Before registration begins, IMU integration over the scan duration reconstructs per-point intermediate poses to **deskew** the sweep, while IMU preintegration supplies a tight **initial guess** for scan-to-submap matching. In return, the resulting LiDAR pose estimate anchors IMU velocity, position, and biases against drift.

**Keyframe-based efficiency**

To keep nearest-neighbor queries bounded as the trajectory grows, LIO-SAM matches each incoming sweep not against the entire global map, but against a sliding submap assembled from nearby keyframes.

The GTSAM sketch below outlines how LIO-SAM inserts IMU preintegration, deskewed LiDAR odometry, optional GPS, loop closures, and bias random-walk factors before triggering an incremental iSAM2 update:

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
        # Random-walk factor constraining the new bias variable. Without it the update still runs, but the bias loses its temporal constraint: each B(k) is fixed by a single IMU factor and the estimate jumps between keyframes
        self.graph.add(gtsam.BetweenFactorConstantBias(
            B(self.key_idx - 1), B(self.key_idx), gtsam.imuBias.ConstantBias(), self.bias_noise))
        
        # 5. iSAM2 incremental update
        result = self.isam.update(self.graph, self.values)
        self.graph.resize(0)
        self.values.clear()
        
        self.key_idx += 1
```

### 7.3.2 FAST-LIO / FAST-LIO2

Where LIO-SAM builds on LOAM feature extraction and factor graphs, [FAST-LIO2 (Xu et al., 2022)](https://doi.org/10.1109/TRO.2022.3141855) takes the opposite path on both counts: it abandons feature extraction altogether to register raw points directly onto the map, and replaces graph optimization with an Iterated Extended Kalman Filter (IEKF).

**1. Direct point registration (no feature extraction)**

Rather than hand-picking edge and planar corners along scan lines, FAST-LIO2 matches every incoming raw point $\mathbf{p}_k$ to the five nearest map points and fits a local tangent plane, minimizing the signed point-to-plane distance:

$$d_k = \mathbf{n}_k^T (\mathbf{T} \cdot \mathbf{p}_k - \mathbf{q}_k)$$

where $\mathbf{n}_k$ is the fitted plane normal and $\mathbf{q}_k$ is a point on that plane. Dropping feature extraction brings three immediate dividends: it preserves geometric information that rigid curvature thresholds throw away; it works agnostically across spinning LiDARs and non-repetitive solid-state scanners such as Livox (whose irregular scan patterns break ring-based curvature); and when paired with an incremental tree structure, raw-point alignment still runs well within real-time budgets.

**2. ikd-Tree (Incremental k-d Tree)**

Registering raw scans directly against a growing map would choke a conventional static kd-tree, which must rebuild from scratch whenever points are inserted or pruned. FAST-LIO2 solves this with the **ikd-Tree**, supporting $O(\log N)$ incremental point insertion, lazy deletion of outdated points, scapegoat-tree-style partial re-balancing when insertions skew tree depth, and fast box-range deletion to evict distant map regions outside the active sliding cube. As a result, map maintenance and nearest-plane lookup execute concurrently in real time.

**3. Iterated Extended Kalman Filter (IEKF)**

Because LiDAR point-to-plane observations are strongly nonlinear in rotation and translation, a single-shot EKF update linearizes at a rough prediction and leaves residual error behind. FAST-LIO2 instead uses an **Iterated EKF**, repeatedly re-evaluating the observation model and Jacobian $\mathbf{H}^{(k)}$ at each iterate $\hat{\mathbf{x}}^{(k)}$ while anchoring updates to the prior prediction $\hat{\mathbf{x}}^{-}$:

$$\hat{\mathbf{x}}^{(k+1)} = \hat{\mathbf{x}}^{-} + \mathbf{K}^{(k)} (\mathbf{z} - h(\hat{\mathbf{x}}^{(k)}) - \mathbf{H}^{(k)}(\hat{\mathbf{x}}^{-} - \hat{\mathbf{x}}^{(k)}))$$

with Kalman gain

$$\mathbf{K}^{(k)} = \mathbf{P}^{-} (\mathbf{H}^{(k)})^T (\mathbf{H}^{(k)} \mathbf{P}^{-} (\mathbf{H}^{(k)})^T + \mathbf{R})^{-1}$$

Iterations stop once state increments or residual reductions converge; FAST-LIO configurations typically require only a handful of passes, depending on motion severity and scene geometry. Algebraically, this iterated relinearization acts like a MAP Gauss-Newton step around the propagated prior while carrying forward full filter covariance.

**State vector**:

Alongside the IMU rotation ${}^G\mathbf{R}_I$, position ${}^G\mathbf{p}_I$, velocity ${}^G\mathbf{v}_I$, gyro bias $\mathbf{b}_g$, and accelerometer bias $\mathbf{b}_a$, FAST-LIO2 augments its state with the LiDAR–IMU extrinsics ${}^I\mathbf{R}_L, {}^I\mathbf{p}_L$ and world gravity vector $\mathbf{g}$, allowing calibration offsets and gravity alignment to refine online:

$$\mathbf{x} = [{}^G\mathbf{R}_I, {}^G\mathbf{p}_I, {}^G\mathbf{v}_I, \mathbf{b}_g, \mathbf{b}_a, {}^I\mathbf{R}_L, {}^I\mathbf{p}_L, \mathbf{g}]$$

**Reported FAST-LIO2 performance**: Under the authors' hardware and datasets—spanning multi-line spinning and solid-state LiDARs across multiple compute platforms—the paper reports odometry and mapping rates up to 100 Hz; in practice, always remeasure throughput for your own point budget, CPU, and local map size.

The C++ sketch below traces one IEKF scan update—re-projecting points, querying local planes in the `ikd_tree`, accumulating point-to-plane Jacobians across iterations, updating covariance on convergence, and inserting aligned points into the map:

```cpp
// FAST-LIO2 IEKF update pseudocode (C++)
struct State {
    Matrix3d R_GI;    // IMU -> world rotation (prefix notation G_R_I)
    Vector3d p_GI;    // IMU position in world
    Vector3d v_GI;    // IMU velocity in world
    Vector3d bg, ba;  // gyro/accel bias
    Matrix3d R_IL;    // LiDAR -> IMU rotation (prefix notation I_R_L)
    Vector3d p_IL;    // LiDAR origin expressed in the IMU frame
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

Faster-LIO speeds up FAST-LIO2's data structure bottleneck by swapping out the `ikd-Tree` for an incremental hashed voxel map. Each active voxel caches local plane parameters and updates them incrementally as new points arrive, trading $O(\log N)$ tree traversals for $O(1)$ hash-table lookups.

### 7.3.4 Point-LIO

[Point-LIO (He et al., 2023)](https://doi.org/10.1002/aisy.202200459) pushes the FAST-LIO lineage to its temporal limit by updating the filter state **point by point** rather than waiting for an entire ~100 ms sweep to finish.

Conventional scan-based LIO integrates the IMU to deskew a sweep (§7.3.1), yet still lumps the entire 100 ms interval into a single batch update—an assumption that breaks down when a drone or handheld sensor spins at hundreds of degrees per second or saturates short-interval deskewing. Point-LIO instead interleaves high-rate IMU propagation and single-point EKF updates in exact chronological order, discretizing the continuous kinematics between IMU samples:

$$\frac{d}{dt}\mathbf{R} = \mathbf{R}[\boldsymbol{\omega}]_\times, \quad \frac{d}{dt}\mathbf{v} = \mathbf{R}\mathbf{a} + \mathbf{g}, \quad \frac{d}{dt}\mathbf{p} = \mathbf{v}$$

Every individual LiDAR return immediately triggers a state propagation and measurement update at its true timestamp, approximating a continuous-time filter. Because each point is processed directly at its acquisition state, motion distortion disappears implicitly without a separate deskewing stage, yielding accurate tracking even during violent rotations. That fine-grained state estimate still remains subject to IMU noise, bias drift, and clock synchronization accuracy, and computational load scales directly with point rate—so any speedup or overhead ratio relative to FAST-LIO2 depends on the target point cloud density, CPU, and configuration.

### 7.3.5 COIN-LIO

Geometric registration fails whenever a scene lacks surface variation along an axis—such as a long featureless tunnel or an endless flat plain. [COIN-LIO (Pfreundschuh et al., 2024)](https://arxiv.org/abs/2310.01235) addresses geometric degeneracy by folding **LiDAR return intensity** directly into an iterated EKF. It projects laser intensity measurements into a virtual intensity image, filters the image to stabilize brightness across views, detects directions poorly constrained by 3D point-to-plane geometry, and selects complementary intensity patches whose photometric residuals fuse alongside IMU measurements and point-to-plane constraints. As long as surface reflectivity varies along the geometrically unobservable axis, intensity patches supply the missing constraint.

---

## 7.4 Continuous-Time LiDAR Odometry

Discrete-time registration assigns one pose per LiDAR sweep and falls back on open-loop interpolation to deskew intra-scan points. Continuous-time odometry drops that rigid frame boundary by parameterizing the sensor trajectory as a smooth function $\mathbf{T}(t)$ directly inside the optimization—letting every point's timestamp constrain the curve shape simultaneously.

### 7.4.1 CT-ICP

[CT-ICP (Dellenbach et al., 2022)](https://arxiv.org/abs/2109.12979) brings continuous-time refinement into pure LiDAR odometry by attaching **two poses**—start pose $\mathbf{T}_s$ and end pose $\mathbf{T}_e$—to every sweep. For any point timestamp $t_k \in [t_s, t_e]$, its sensor pose is interpolated along the $SE(3)$ geodesic:

$$\mathbf{T}(t_k) = \mathbf{T}_s \cdot \text{Exp}\left(\frac{t_k - t_s}{t_e - t_s} \cdot \text{Log}(\mathbf{T}_s^{-1}\mathbf{T}_e)\right)$$

By optimizing $\mathbf{T}_s$ and $\mathbf{T}_e$ jointly against the point cloud residuals (while enforcing continuity across consecutive sweeps), CT-ICP refines intra-scan velocity during registration itself rather than trusting a fixed open-loop deskew step—making it notably effective on IMU-less platforms.

### 7.4.2 B-Spline-Based Trajectory Representation

For agile motion or asynchronous multi-sensor rigs, a piecewise-linear scan parameterization can be generalized to a smooth B-spline curve anchored by control poses $\{\mathbf{T}_i\}$:

$$\mathbf{T}(t) = \mathbf{T}_0 \prod_{i=1}^{k} \text{Exp}\left(\tilde{B}_i(t) \cdot \text{Log}(\mathbf{T}_{i-1}^{-1}\mathbf{T}_i)\right)$$

where $\tilde{B}_i(t)$ is the cumulative B-spline basis function; a cubic B-spline maintains $C^2$ continuity across ordinary interior knots, though repeated knots or boundary conditions can lower continuity.

Representing $\mathbf{T}(t)$ as a B-spline brings three natural strengths: querying poses—and differentiating analytic velocities and accelerations—at arbitrary timestamps handles asynchronous sensors effortlessly; spline degree directly tunes smoothness (though geometric smoothness alone does not guarantee dynamic feasibility); and local support ensures adjusting one control pose only alters a bounded time window. The trade-offs are heavier optimization cost and sensitivity to knot spacing: knots spaced too densely overfit measurement noise, while overly sparse knots wash out rapid maneuvers. Kalibr uses this same B-spline parameterization for camera–IMU temporal and spatial calibration (→ Chapter 3).

---

## 7.5 Solid-State LiDAR Specifics

Unlike mechanical spinning LiDARs that sweep fixed concentric rings on every rotation, solid-state LiDARs (such as the Livox series) trace non-repetitive petal or rose scan patterns across a narrower field of view. The table below contrasts the operational differences between the two hardware classes:

| Property | Spinning (Velodyne, Ouster) | Solid-state (Livox) |
|------|--------------------------|---------------------|
| Scan pattern | Repetitive (same pattern every rotation) | Non-repetitive (petal / rose pattern) |
| FoV | 360° horizontal | Model-dependent (Mid-40 38.4° circular, Avia 70.4°×77.2°, Horizon 81.7°×25.1°) |
| Point density | Uniform | Accumulates over time, non-uniform |
| Price | High | Low |
| Size / weight | Large | Small |

**Impact of non-repetitive scanning on feature extraction**

Because LOAM-style curvature evaluates neighbors along a shared horizontal scan ring, solid-state sweeps—which lack fixed scan lines and scatter points irregularly across a rosette—break ring-based feature extraction immediately. Adapting to them requires either computing expensive $k$-nearest-neighbor (KNN) local curvatures or discarding feature extraction altogether in favor of raw points.

**Why FAST-LIO2 is strong on solid-state**

Because FAST-LIO2 registers raw points directly without feature extraction (unlike original FAST-LIO, which still relied on edge and planar features), it handles arbitrary scan geometries out of the box. Moreover, a non-repetitive sensor gradually fills in its field of view over successive sweeps, and the `ikd-Tree` naturally absorbs that progressive densification into an ever-sharper local surface map—while tight IMU coupling stabilizes motion estimation across a narrower single-scan FoV.

That combination of compact size, low cost, and direct registration compatibility has made Livox + FAST-LIO2 a staple pairing on drones, handheld scanners, and lightweight ground robots. Even so, sensor selection should weigh field of view, maximum range, hardware time synchronization, and point distribution alongside unit cost.

---

## 7.6 Learning-Based LiDAR Odometry

Mirroring DeepVO in vision, learning-based LiDAR odometry trains neural networks to regress relative 6-DoF motion directly from consecutive point cloud sweeps.

### 7.6.1 The DeepLO Family

Because raw 3D point clouds are irregular, early networks project sweeps into structured 2D range or surface representations before applying convolutions:

- **LO-Net** (Li et al., 2019): projects each LiDAR sweep onto a 2D range image for CNN pose regression, adding surface normal and dynamic-mask prediction as auxiliary losses to enforce geometric structure.
- **DeepLO** (Cho et al., 2020): feeds spherical vertex and normal maps into a CNN trained unsupervised via a point-to-plane ICP residual loss.
- **PWCLO-Net** (Wang et al., 2021): adapts the Pyramid, Warping, and Cost-volume architecture from optical flow to hierarchical 3D point cloud odometry.

### 7.6.2 Current Limitations

Despite active research, end-to-end learned LiDAR odometry still trails classical geometric registration for four structural reasons:

1. **Irregular data structure**: Unlike dense pixel grids, point clouds are sparse, unordered, and varying in density across range, resisting standard 2D CNN inductive biases.
2. **Formidable geometric baselines**: Unlike monocular images that lack metric depth, LiDAR sweeps already carry accurate 3D coordinates, so ICP, GICP, and NDT work reliably without training data whenever overlap and structure hold. Classical methods still struggle in repetitive geometry, heavy dynamics, precipitation or dust, and low-overlap sweeps—precisely where learning helps selectively via learned correspondence, dynamic-point masking, or uncertainty weighting.
3. **Scarcity of training data**: Large-scale, calibrated LiDAR odometry benchmarks remain orders of magnitude smaller than internet-scale image corpora.
4. **Cross-sensor brittleness**: Networks trained on one scanner's beam pattern and vertical FoV rarely transfer cleanly to another LiDAR geometry.

Consequently, deep learning currently delivers its highest payoff in **auxiliary modules** rather than replacing geometric optimization—powering place recognition (PointNetVLAD), global registration initialization (GeoTransformer, → Chapter 5), and semantic dynamic-object filtering. Note that Scan Context, widely deployed for LiDAR loop closure in the same ecosystem, is a handcrafted polar-grid descriptor rather than a learned model.

---

## 7.7 2023–2024 Research

Recent LiDAR odometry work has pushed in two complementary directions: stripping pipelines down for cross-platform generality, and accelerating covariance-aware registration.

**[KISS-ICP (Vizzo et al., 2023)](https://arxiv.org/abs/2209.15397)** strips LiDAR odometry back to basics—combining plain point-to-point ICP with adaptive correspondence thresholding, a robust kernel, and constant-velocity motion compensation. Under a single shared configuration across automotive, UAV, and handheld benchmarks, the paper reports competitive trajectory accuracy while avoiding per-dataset parameter tuning.

**[MAD-ICP (Ferrari et al., 2024)](https://arxiv.org/abs/2405.05828)** builds a PCA-based kd-tree to capture local planar geometry directly inside the spatial index and feed point-to-plane registration, emphasizing principled data association to match domain-tuned pipelines across diverse LiDAR models.

**[iG-LIO (Chen et al., 2024)](https://github.com/zijiechenrobotics/ig_lio)** brings full GICP covariance matching into a tightly-coupled IEKF LIO framework. By pairing a voxel-based surface covariance estimator (VSCE) with an incremental voxel map, it cuts both covariance estimation and nearest-neighbor lookup overhead, comparing compute and accuracy against Faster-LIO on its evaluated benchmarks.

---

## Summary

| System | Approach | Estimation | Sensors | Feature |
|--------|------|-----------|------|-----------|
| ICP/GICP/NDT | Registration | Iterative optimization | LiDAR only | Basic building block |
| LOAM | Feature-based | LM optimization | LiDAR only | Edge/planar feature, two-stage architecture |
| LeGO-LOAM | Feature-based | LM optimization | LiDAR only | Ground segmentation, lightweight |
| LIO-SAM | Feature-based | Factor graph (iSAM2) | LiDAR + IMU + GPS | Modular multi-sensor integration |
| FAST-LIO2 | Direct | IEKF | LiDAR + IMU | No feature extraction, ikd-Tree, paper reports up to 100 Hz |
| Point-LIO | Direct | Point-wise EKF | LiDAR + IMU | Per-point update, fast motion |
| COIN-LIO | Direct + intensity | IEKF | LiDAR + IMU | LiDAR intensity complements weak geometry |
| CT-ICP | Direct | Optimization | LiDAR only | Continuous-time motion model, no IMU needed |
| KISS-ICP | Direct (P2P) | Iterative optimization | LiDAR only | Adaptive threshold, tuning-free, general-purpose |
| MAD-ICP | Direct (P2Plane) | Iterative optimization | LiDAR only | PCA-based structural extraction, data-matching focus |
| iG-LIO | Direct (GICP) | IEKF | LiDAR + IMU | Incremental GICP, voxel covariance estimation |

Two distinct architectural lineages emerge from the table above: **feature-based + factor graph** pipelines exemplified by LOAM (2014) → LeGO-LOAM (2018) → LIO-SAM (2020), and **direct + iterated Kalman filter** pipelines driven by FAST-LIO (2021) → FAST-LIO2 (2022) → Point-LIO (2023). Neither dominates universally—accuracy and runtime depend tightly on scan pattern, motion aggressiveness, map scale, hardware, and benchmark protocol. Feature-based designs excel in structured urban canyons rich in clean corners and walls, whereas direct point-to-plane filters handle unstructured foliage and non-repetitive solid-state sweeps naturally.

Yet even tightly-coupled LIO hits hard physical boundaries—in featureless tunnels, dense fog, or structurally repetitive corridors where 3D geometry alone cannot disambiguate motion—just as Chapter 6's VIO falters in darkness or textureless stretches. Bringing cameras, LiDAR, and IMUs together under one estimator raises a deeper architectural question: at what level should heterogeneous sensors share measurements and states? Chapter 8 takes up that multi-sensor fusion design space.
