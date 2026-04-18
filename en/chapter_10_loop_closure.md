# Ch.10 — Loop Closure & Global Optimization

In Ch.9 we covered place recognition — recognizing previously visited locations. This chapter addresses how those recognition results are integrated into a SLAM system to actually correct the accumulated drift.

In a SLAM system, odometry inevitably accumulates drift. No matter how precise the sensors are, small errors in relative pose estimation build up over time and break global consistency. Loop closure is the mechanism that recognizes "we have revisited a previously visited place" and uses that information to correct the accumulated drift all at once.

In this chapter we walk through the full loop closure pipeline (detection → verification → correction), the mathematical foundations of pose graph optimization at the heart of the correction step, and we extend the discussion to global relocalization and multi-session SLAM.

---

## 10.1 Loop Closure Pipeline

Loop closure consists of three stages: **Detection** (candidate search), **Verification** (geometric verification), and **Correction** (graph correction). Each stage plays a distinct role, and failure in any one of them can break the consistency of the entire system.

### 10.1.1 Detection: Candidate Search

Loop closure detection asks: "Which past observation is similar to the current sensor observation?" This is essentially the same problem as place recognition covered in Ch.9.

In **visual loop closure detection**, the global descriptor of the current image is compared against a descriptor database of past keyframes:

1. **BoW-based (traditional)**: DBoW2 is used to compare the visual word histograms of ORB keypoints. ORB-SLAM3 uses this approach. Each keyframe is represented as a bag-of-words vector $\mathbf{v}_i$, and its similarity to the current frame $\mathbf{v}_q$ is computed as $s(\mathbf{v}_q, \mathbf{v}_i) = 1 - \frac{1}{2} \left| \frac{\mathbf{v}_q}{\|\mathbf{v}_q\|} - \frac{\mathbf{v}_i}{\|\mathbf{v}_i\|} \right|$ (L1-score).

2. **Learning-based (modern)**: Global descriptors such as [NetVLAD](https://arxiv.org/abs/1511.07247) and [AnyLoc](https://arxiv.org/abs/2308.00688) are used. AnyLoc aggregates the dense features of [DINOv2](https://arxiv.org/abs/2304.07193) with VLAD, providing general-purpose operation without environment-specific training. Candidates are ranked by cosine similarity:

$$s(\mathbf{d}_q, \mathbf{d}_i) = \frac{\mathbf{d}_q^\top \mathbf{d}_i}{\|\mathbf{d}_q\| \|\mathbf{d}_i\|}$$

In **LiDAR loop closure detection**, 3D point cloud descriptors are used:

- **[Scan Context](https://doi.org/10.1109/IROS.2018.8593953)**: A descriptor that directly preserves spatial structure by recording the maximum height per bin/sector in a sensor-centered polar coordinate system. Efficient candidate search is possible through a two-stage search using a ring key and a sector key, and it is robust even to reverse revisits.
- **[PointNetVLAD](https://arxiv.org/abs/1804.03492), [OverlapTransformer](https://arxiv.org/abs/2203.03397)**: Learning-based 3D place recognition methods that achieve higher recall than Scan Context in large-scale environments.

**Temporal filtering**: Matching against recent frames is not loop closure but simply continuous tracking. Therefore, only keyframes that are sufficiently separated in time (e.g., at least 30 seconds apart) are considered as candidates.

```python
import numpy as np

def detect_loop_candidates(query_descriptor, database_descriptors, 
                           timestamps, current_time, 
                           min_time_gap=30.0, top_k=5, threshold=0.7):
    """
    Loop closure candidate detection.
    
    Args:
        query_descriptor: global descriptor of the current frame (D,)
        database_descriptors: descriptors of past keyframes (N, D)
        timestamps: timestamp of each keyframe (N,)
        current_time: current time
        min_time_gap: minimum time gap (seconds)
        top_k: number of candidates to return
        threshold: similarity threshold
        
    Returns:
        candidates: list of [(index, similarity)]
    """
    # Temporal filtering: exclude recent frames
    time_mask = (current_time - timestamps) > min_time_gap
    
    if not np.any(time_mask):
        return []
    
    # Compute cosine similarity
    query_norm = query_descriptor / (np.linalg.norm(query_descriptor) + 1e-8)
    db_norms = database_descriptors / (
        np.linalg.norm(database_descriptors, axis=1, keepdims=True) + 1e-8
    )
    
    similarities = db_norms @ query_norm  # (N,)
    
    # Apply time filter
    similarities[~time_mask] = -1.0
    
    # Select top-k candidates
    top_indices = np.argsort(similarities)[::-1][:top_k]
    candidates = [
        (idx, similarities[idx]) 
        for idx in top_indices 
        if similarities[idx] > threshold
    ]
    
    return candidates
```

### 10.1.2 Verification: Geometric Verification

Candidates found in the detection stage are selected based on appearance similarity alone, so **false positives** (actually different places that merely look similar) can be included. Perceptual aliasing — visually similar but actually distinct places — is especially frequent in indoor environments (similar corridors, repetitive structures).

A false positive loop closure is **catastrophic**. A single incorrect loop closure can warp the entire map. Verification must therefore be performed conservatively — maximize precision even at some cost to recall.

**Geometric verification methods**:

1. **2D-2D: Essential matrix verification**: Feature point matching is performed between the current frame and the candidate keyframe, and the essential matrix $\mathbf{E}$ is estimated with RANSAC. If the number of inliers is sufficient (e.g., ≥ 20) and the inlier ratio is high (e.g., ≥ 50%), the loop closure is deemed valid.

$$\mathbf{p}_2^\top \mathbf{E} \mathbf{p}_1 = 0, \quad \mathbf{E} = [\mathbf{t}]_\times \mathbf{R}$$

2. **3D-3D: Point cloud registration**: In LiDAR-based systems, the relative transform $\mathbf{T}_{ij} \in SE(3)$ between two scans is estimated with ICP or GeoTransformer. It is verified by the fitness score (ratio of registered points) and RMSE.

3. **2D-3D: PnP**: The relative pose is estimated by solving a PnP problem between the current 2D feature points and the 3D map points of the candidate keyframe.

4. **Temporal consistency**: Rather than a single match, check whether matches against the same place appear consistently across several consecutive frames. ORB-SLAM3 accepts a loop closure only when the same place is detected three times in a row.

```python
import numpy as np

def verify_loop_closure(kp_current, kp_candidate, matches, K,
                        min_inliers=20, min_inlier_ratio=0.5):
    """
    Essential matrix-based geometric verification of loop closure.
    
    Args:
        kp_current: keypoint coordinates of the current frame (N, 2)
        kp_candidate: keypoint coordinates of the candidate frame (M, 2)
        matches: list of match index pairs [(i, j), ...]
        K: camera intrinsic matrix (3, 3)
        min_inliers: minimum number of inliers
        min_inlier_ratio: minimum inlier ratio
        
    Returns:
        is_valid: whether it is a valid loop closure
        T_relative: relative transform (4, 4) or None
    """
    if len(matches) < min_inliers:
        return False, None
    
    pts1 = np.array([kp_current[m[0]] for m in matches], dtype=np.float64)
    pts2 = np.array([kp_candidate[m[1]] for m in matches], dtype=np.float64)
    
    # Convert to normalized coordinates
    K_inv = np.linalg.inv(K)
    pts1_norm = (K_inv @ np.hstack([pts1, np.ones((len(pts1), 1))]).T).T[:, :2]
    pts2_norm = (K_inv @ np.hstack([pts2, np.ones((len(pts2), 1))]).T).T[:, :2]
    
    # Estimate essential matrix with RANSAC
    E, inlier_mask = estimate_essential_ransac(pts1_norm, pts2_norm, 
                                                threshold=1e-3, max_iter=1000)
    
    num_inliers = np.sum(inlier_mask)
    inlier_ratio = num_inliers / len(matches)
    
    if num_inliers < min_inliers or inlier_ratio < min_inlier_ratio:
        return False, None
    
    # Recover R, t from E
    R, t = decompose_essential(E, pts1_norm[inlier_mask], pts2_norm[inlier_mask])
    
    T_relative = np.eye(4)
    T_relative[:3, :3] = R
    T_relative[:3, 3] = t.flatten()
    
    return True, T_relative


def estimate_essential_ransac(pts1, pts2, threshold=1e-3, max_iter=1000):
    """Estimate essential matrix via the 5-point algorithm + RANSAC."""
    best_E = None
    best_inliers = np.zeros(len(pts1), dtype=bool)
    
    for _ in range(max_iter):
        # Randomly sample 8 points
        idx = np.random.choice(len(pts1), 8, replace=False)
        
        # Generate an E candidate with the 8-point algorithm (simplified 5-point variant)
        E_candidate = eight_point_essential(pts1[idx], pts2[idx])
        
        if E_candidate is None:
            continue
        
        # Inlier test via Sampson error
        errors = sampson_error(E_candidate, pts1, pts2)
        inliers = errors < threshold
        
        if np.sum(inliers) > np.sum(best_inliers):
            best_inliers = inliers
            best_E = E_candidate
    
    return best_E, best_inliers


def sampson_error(E, pts1, pts2):
    """Sampson error — first-order approximate distance of the epipolar constraint."""
    # Convert pts to homogeneous
    p1 = np.hstack([pts1, np.ones((len(pts1), 1))])  # (N, 3)
    p2 = np.hstack([pts2, np.ones((len(pts2), 1))])  # (N, 3)
    
    Ep1 = (E @ p1.T).T    # (N, 3)
    Etp2 = (E.T @ p2.T).T  # (N, 3)
    
    # p2^T E p1
    numerator = np.sum(p2 * Ep1, axis=1) ** 2
    denominator = Ep1[:, 0]**2 + Ep1[:, 1]**2 + Etp2[:, 0]**2 + Etp2[:, 1]**2
    
    return numerator / (denominator + 1e-10)
```

### 10.1.3 The Danger of False Positives and How to Prevent Them

Let us examine concretely why a false positive loop closure is so dangerous.

**Scenario**: Suppose the robot traverses two similar-looking corridors. If an incorrect loop closure is formed between keyframe $i$ in corridor A and keyframe $j$ in corridor B, the pose graph optimizer pulls these two poses close together. As a result, all poses between the two corridors are distorted, and the entire map folds or twists.

**Prevention strategies**:

1. **Multi-stage verification**: The candidate must sequentially pass appearance similarity → geometric consistency → temporal consistency.

2. **Use robust kernels** (detailed in §10.2): Make the optimizer itself less sensitive to outlier constraints.

3. **[Switchable constraints (Sünderhauf & Protzel, 2012)](https://doi.org/10.1109/IROS.2012.6385590)**: Add an on/off switch variable to each loop closure factor so that the optimizer automatically deactivates loop closures with poor consistency.

4. **[DCS (Dynamic Covariance Scaling) (Agarwal et al., 2013)](https://doi.org/10.1109/ICRA.2013.6630733)**: Dynamically adjust the covariance (uncertainty) of each loop closure so that the influence of outliers is attenuated automatically.

5. **[PCM (Pairwise Consistency Maximization) (Mangelson et al., 2018)](https://doi.org/10.1109/ICRA.2018.8460217)**: Check pairwise consistency among multiple loop closure candidates and accept only the largest consistent subset.

### 10.1.4 Correction: Graph Correction

A loop closure that passes verification is added to the pose graph as a new constraint. The loop closure edge encodes the relative transform $\mathbf{T}_{ij}$ between the two poses and its uncertainty $\boldsymbol{\Sigma}_{ij}$:

$$e_{ij} = \text{Log}(\mathbf{T}_{ij}^{-1} \cdot \mathbf{T}_i^{-1} \cdot \mathbf{T}_j)$$

Once this edge is added, the pose graph optimizer (§10.2) re-optimizes the entire graph and corrects the drift. In this process, not only the loop closure edge but also the odometry edges are adjusted, so that the error is distributed evenly over the full trajectory.

---

## 10.2 Pose Graph Optimization

Pose graph optimization is the core of the SLAM backend. It optimizes the global consistency of the entire pose trajectory while satisfying the relative constraints produced by the frontend (odometry, loop closure).

### 10.2.1 SE(3) Pose Graph

The pose graph is represented as a graph $\mathcal{G} = (\mathcal{V}, \mathcal{E})$:

- **Nodes** $\mathcal{V} = \{\mathbf{T}_1, \mathbf{T}_2, \ldots, \mathbf{T}_n\}$: the pose of each keyframe, $\mathbf{T}_i \in SE(3)$.
- **Edges** $\mathcal{E}$: relative constraints between two nodes, divided into odometry edges and loop closure edges.

Each edge $(i, j) \in \mathcal{E}$ has a measured relative transform $\tilde{\mathbf{T}}_{ij}$ and an information matrix $\boldsymbol{\Omega}_{ij}$.

**Error definition on SE(3)**: In a pose graph, the error is defined not on Euclidean space but on the Lie group SE(3):

$$\mathbf{e}_{ij} = \text{Log}(\tilde{\mathbf{T}}_{ij}^{-1} \cdot \mathbf{T}_i^{-1} \cdot \mathbf{T}_j) \in \mathbb{R}^6$$

Here, $\text{Log}: SE(3) \to \mathfrak{se}(3) \cong \mathbb{R}^6$ is the mapping to the Lie algebra via the matrix logarithm. This 6-dimensional vector encodes the error in $($ 3 rotation + 3 translation $)$.

**Optimization objective**: Minimize the weighted sum of squared edge errors:

$$\mathbf{T}^* = \arg\min_{\mathbf{T}_1, \ldots, \mathbf{T}_n} \sum_{(i,j) \in \mathcal{E}} \mathbf{e}_{ij}^\top \boldsymbol{\Omega}_{ij} \mathbf{e}_{ij}$$

This is a nonlinear least squares problem that can be solved with Gauss-Newton or Levenberg-Marquardt.

**Gauss-Newton on manifold**: Since SE(3) is not a Euclidean space, we cannot use direct addition (+). Instead, we compute an increment $\boldsymbol{\delta}$ on the Lie algebra and apply it on the manifold via the exponential map:

$$\mathbf{T}_i \leftarrow \mathbf{T}_i \cdot \text{Exp}(\boldsymbol{\delta}_i)$$

The update at a single iteration:

1. At the current estimate, compute the error $\mathbf{e}_{ij}$ and Jacobian $\mathbf{J}_{ij}$ for each edge.
2. Form the normal equations: $\mathbf{H} \boldsymbol{\delta} = -\mathbf{b}$, where

$$\mathbf{H} = \sum_{(i,j)} \mathbf{J}_{ij}^\top \boldsymbol{\Omega}_{ij} \mathbf{J}_{ij}, \quad \mathbf{b} = \sum_{(i,j)} \mathbf{J}_{ij}^\top \boldsymbol{\Omega}_{ij} \mathbf{e}_{ij}$$

3. Since $\mathbf{H}$ is sparse, it can be solved efficiently by sparse Cholesky decomposition.
4. Apply the increment: $\mathbf{T}_i \leftarrow \mathbf{T}_i \cdot \text{Exp}(\boldsymbol{\delta}_i)$.
5. Iterate until convergence.

**[g2o (General Graph Optimization)](https://doi.org/10.1109/ICRA.2011.5979949)**: A framework developed by Kümmerle et al. (2011) that implements the above procedure for general graph optimization problems. The user defines only the vertex (node) and edge (constraint) types, and the sparse optimization engine handles the rest automatically.

```python
import numpy as np
from scipy.spatial.transform import Rotation

def se3_log(T):
    """SE(3) matrix -> 6-dimensional vector (rotation + translation)."""
    R = T[:3, :3]
    t = T[:3, 3]
    
    # SO(3) logarithm
    rot = Rotation.from_matrix(R)
    omega = rot.as_rotvec()  # (3,)
    
    theta = np.linalg.norm(omega)
    
    if theta < 1e-10:
        V_inv = np.eye(3)
    else:
        omega_hat = skew(omega)
        V_inv = (np.eye(3) 
                 - 0.5 * omega_hat 
                 + (1.0/theta**2) * (1 - theta/(2*np.tan(theta/2))) * omega_hat @ omega_hat)
    
    rho = V_inv @ t
    return np.concatenate([rho, omega])  # (6,) — [translation; rotation]


def se3_exp(xi):
    """6-dimensional vector -> SE(3) matrix."""
    rho = xi[:3]   # translation part
    omega = xi[3:]  # rotation part
    
    theta = np.linalg.norm(omega)
    
    if theta < 1e-10:
        R = np.eye(3)
        V = np.eye(3)
    else:
        omega_hat = skew(omega)
        R = (np.eye(3) 
             + (np.sin(theta)/theta) * omega_hat 
             + ((1 - np.cos(theta))/theta**2) * omega_hat @ omega_hat)
        V = (np.eye(3) 
             + ((1 - np.cos(theta))/theta**2) * omega_hat 
             + ((theta - np.sin(theta))/theta**3) * omega_hat @ omega_hat)
    
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = V @ rho
    return T


def skew(v):
    """3-dimensional vector -> skew-symmetric matrix."""
    return np.array([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0]
    ])


def pose_graph_error(T_i, T_j, T_ij_measured):
    """Error vector (6-dimensional) between two poses."""
    T_ij_estimated = np.linalg.inv(T_i) @ T_j
    T_error = np.linalg.inv(T_ij_measured) @ T_ij_estimated
    return se3_log(T_error)


def pose_graph_cost(poses, edges, measurements, information_matrices):
    """
    Cost function of the full pose graph.
    
    Args:
        poses: list of SE(3) matrices [T_0, T_1, ..., T_n]
        edges: list of edge indices [(i, j), ...]
        measurements: list of measured relative transforms [T_ij, ...]
        information_matrices: list of information matrices [Omega_ij, ...]
        
    Returns:
        total_cost: scalar cost
    """
    total_cost = 0.0
    
    for (i, j), T_ij, Omega_ij in zip(edges, measurements, information_matrices):
        e = pose_graph_error(poses[i], poses[j], T_ij)
        total_cost += e @ Omega_ij @ e
    
    return total_cost
```

### 10.2.2 Robust Kernel

In practical SLAM systems, outlier measurements are unavoidable. Incorrect loop closures, sensor errors, dynamic objects, and more are the causes. The standard least squares cost function $\rho(x) = x^2$ is extremely sensitive to outliers — large errors dominate the cost and distort the whole solution.

A **robust kernel** (M-estimator) limits the influence of large residuals and enables optimization that is robust to outliers:

| Kernel | $\rho(s)$ ($s = e^2$) | Characteristics |
|--------|----------------------|------|
| Least Squares | $s$ | Vulnerable to outliers |
| Huber | $\begin{cases} s & \text{if } \sqrt{s} \leq k \\ 2k\sqrt{s} - k^2 & \text{otherwise} \end{cases}$ | Linear beyond threshold $k$ |
| Cauchy | $k^2 \log(1 + s/k^2)$ | Smooth attenuation |
| Geman-McClure | $\frac{s}{k^2 + s}$ | Strong outlier suppression |

Applying a robust kernel changes the cost function to:

$$\sum_{(i,j)} \rho\left(\mathbf{e}_{ij}^\top \boldsymbol{\Omega}_{ij} \mathbf{e}_{ij}\right)$$

This can be solved via **IRLS (Iteratively Reweighted Least Squares)**: at each iteration, the weights are recomputed based on the residual magnitudes and a weighted least squares problem is solved.

$$w_i = \rho'(s_i), \quad s_i = \mathbf{e}_i^\top \boldsymbol{\Omega}_i \mathbf{e}_i$$

Here $\rho'$ is the derivative with respect to $s$ of the $\rho(s)$ defined in the table above. Outlier edges are assigned small weights so that their influence is reduced automatically.

**[Switchable constraints](https://doi.org/10.1109/IROS.2012.6385590)**: Sünderhauf & Protzel (2012) introduced a binary switch variable $s_{ij} \in [0, 1]$ on each loop closure factor, allowing the optimizer to deactivate inconsistent loop closures ($s_{ij} \to 0$):

$$\rho_{\text{switch}}(\mathbf{e}_{ij}, s_{ij}) = s_{ij}^2 \mathbf{e}_{ij}^\top \boldsymbol{\Omega}_{ij} \mathbf{e}_{ij} + \lambda (1 - s_{ij})^2$$

The first term minimizes the error when the switch is on, and the second term is a penalty for turning the switch off. The optimizer automatically balances these two.

**[DCS (Dynamic Covariance Scaling)](https://doi.org/10.1109/ICRA.2013.6630733)** (Agarwal et al., 2013): Dynamically scales the covariance of each loop closure according to the residual magnitude. The covariance of an outlier automatically grows (= increased uncertainty), so its influence decreases.

```python
def huber_kernel(s, k=1.345):
    """Huber robust kernel."""
    sqrt_s = np.sqrt(s)
    if sqrt_s <= k:
        return s
    else:
        return 2 * k * sqrt_s - k**2

def cauchy_kernel(s, k=2.3849):
    """Cauchy robust kernel."""
    return k**2 * np.log(1 + s / k**2)

def huber_weight(s, k=1.345):
    """IRLS weight for the Huber kernel."""
    sqrt_s = np.sqrt(s)
    if sqrt_s <= k:
        return 1.0
    else:
        return k / sqrt_s

def robust_pose_graph_cost(poses, edges, measurements, info_matrices, 
                           kernel='huber', k=1.345):
    """Pose graph cost with a robust kernel applied."""
    total_cost = 0.0
    
    kernel_fn = {'huber': huber_kernel, 'cauchy': cauchy_kernel}[kernel]
    
    for (i, j), T_ij, Omega in zip(edges, measurements, info_matrices):
        e = pose_graph_error(poses[i], poses[j], T_ij)
        s = e @ Omega @ e  # Mahalanobis distance squared
        total_cost += kernel_fn(s, k)
    
    return total_cost
```

### 10.2.3 iSAM2: Incremental Smoothing and Mapping

Most SLAM systems, rather than re-optimizing the entire graph in a batch fashion whenever a new keyframe is added, use an incremental approach that selectively updates only the affected parts. [iSAM2 (Kaess et al., 2012)](https://doi.org/10.1177/0278364911430419) is the key algorithm for this incremental optimization.

**Core idea**: When a new variable or measurement is added, precisely identify the range over which its influence propagates, and recompute only that part.

**Bayes tree**: The core data structure of iSAM2. Variable elimination on a factor graph yields a clique tree, and the Bayes tree is this clique tree endowed with direction.

MAP estimation on a factor graph is decomposed as follows:

$$p(\mathbf{x} | \mathbf{z}) \propto \prod_k f_k(\mathbf{x}_k)$$

Here $f_k$ is each factor (odometry, loop closure, etc.) and $\mathbf{x}_k$ is the set of variables associated with that factor.

Performing variable elimination yields:

$$p(\mathbf{x} | \mathbf{z}) = \prod_i p(x_i | \text{Sep}(x_i))$$

Here $\text{Sep}(x_i)$ is the separator of $x_i$ in the clique tree — that is, the conditioning variables that remain when this variable is eliminated. This conditional distribution structure forms the Bayes tree.

**Incremental update procedure**:

1. When a new factor is added, identify only the affected cliques (typically a small number).
2. Redo the QR decomposition only for those cliques and their ancestors.
3. Keep the rest of the tree unchanged.

**Fluid relinearization**: In nonlinear optimization, only variables whose linearization point has drifted significantly from the current estimate are relinearized. This completely removes the periodic batch relinearization that was required in iSAM v1.

**Variable reordering**: When a new variable is added, the full elimination order is not recomputed; only the affected part is reordered incrementally.

**Practical impact**: iSAM2 is the core engine of the GTSAM library and is used in the backend of nearly every modern SLAM system, including LIO-SAM and ORB-SLAM3. Because it can complete optimization within bounded time even in large environments, it enables real-time SLAM.

```python
class SimpleIncrementalOptimizer:
    """
    Simplified incremental optimizer illustrating the core concepts of iSAM2.
    The real iSAM2 is Bayes-tree-based; here we implement only the
    affected-variable tracking concept for intuitive understanding.
    """
    
    def __init__(self):
        self.poses = {}          # {id: SE(3) matrix}
        self.edges = []          # [(i, j, T_ij, Omega_ij)]
        self.affected = set()    # variable IDs that need re-optimization
        
    def add_pose(self, pose_id, T_init):
        """Add a new pose node."""
        self.poses[pose_id] = T_init.copy()
        self.affected.add(pose_id)
        
    def add_edge(self, i, j, T_ij, Omega_ij):
        """Add a new edge (constraint). Mark the connected variables as affected."""
        self.edges.append((i, j, T_ij, Omega_ij))
        self.affected.add(i)
        self.affected.add(j)
        # In the real iSAM2, ancestor cliques are also marked as affected
        # by walking up the Bayes tree.
        
    def add_loop_closure(self, i, j, T_ij, Omega_ij):
        """
        Add a loop closure edge.
        Identical to a regular edge, but a loop closure connects a node from
        the distant past with the current node, so more variables are affected.
        """
        self.add_edge(i, j, T_ij, Omega_ij)
        # A loop closure influences every intermediate pose along the trajectory.
        # In the real iSAM2, due to the Bayes tree structure,
        # the influence propagates up to cliques near the root.
        for k in range(min(i, j) + 1, max(i, j)):
            if k in self.poses:
                self.affected.add(k)
    
    def optimize(self, max_iter=5):
        """
        Re-optimize only the affected variables.
        In the real iSAM2, this is performed via partial re-decomposition of the Bayes tree.
        """
        if not self.affected:
            return
            
        # For simplicity we run full Gauss-Newton here, but
        # update only the affected variables.
        for iteration in range(max_iter):
            # 1. Select only the relevant edges
            relevant_edges = [
                (i, j, T_ij, Omega) 
                for (i, j, T_ij, Omega) in self.edges
                if i in self.affected or j in self.affected
            ]
            
            # 2. Compute error and Jacobian for each edge
            # 3. Form and solve the normal equations
            # 4. Apply the increment only to the affected variables
            
            # (The actual implementation would require solving a sparse linear system; omitted)
            pass
        
        self.affected.clear()
```

---

## 10.3 Global Relocalization

Global relocalization is the problem of finding the robot's location on a pre-built map (prior map). If loop closure is about recognizing "a place I have visited before," relocalization asks, "Where am I within a map built by someone else?"

### 10.3.1 Map-Based Localization

When a pre-built map is available, the current pose is estimated by registering new sensor observations to this map.

**Visual relocalization pipeline**:

1. Extract feature points from the current image.
2. Find 2D-3D correspondences with the 3D points of the map (via visual words or direct matching).
3. Estimate the pose with PnP + RANSAC.
4. Resume tracking using the estimated pose as a starting point.

ORB-SLAM3's relocalization performs this procedure as follows:
- Retrieve candidate keyframes with DBoW2.
- Obtain 2D-3D correspondences via ORB matching.
- Estimate the pose with EPnP + RANSAC.
- Improve accuracy by finding additional matches via guided search.

**LiDAR relocalization**: Register the current LiDAR scan to a prior (point cloud) map.

1. Use a **global descriptor** (Scan Context, PointNetVLAD, etc.) to search for nearby regions in the map.
2. **Coarse registration**: Perform initial registration with FPFH + RANSAC or GeoTransformer.
3. **Fine registration**: Refine alignment with ICP/GICP.

### 10.3.2 Prior Map + Online Sensor

In autonomous driving, it is common to localize in real time with live sensor data against a pre-built HD map. The key challenges are:

- **Discrepancies between the map and the current environment**: Over time, buildings change, vehicles park, and foliage grows. We must handle differences between the prior map and current observations.
- **Cross-modal matching**: The HD map may have been built with LiDAR while the current sensor is only a camera. Registration across heterogeneous sensors is required.
- **No initial pose**: If the robot does not know where in the map it starts, place recognition must be performed against the entire map.

### 10.3.3 Monte Carlo Localization (MCL)

MCL is a particle-filter-based global localization method. It represents the robot's possible poses as particles and updates particle weights according to sensor observations.

**Algorithm**:

1. **Initialization**: Distribute particles uniformly over the entire map (global uncertainty).
2. **Prediction**: Move particles according to the robot's motion model:
$$x_t^{[k]} \sim p(x_t | u_t, x_{t-1}^{[k]})$$
3. **Update**: Compute the weight of each particle by comparing the current sensor observation with the expected observation from the map:
$$w_t^{[k]} = p(z_t | x_t^{[k]}, m)$$
4. **Resampling**: Resample particles in proportion to their weights. Particles with high weights (close to the true location) are duplicated, and low-weight particles are eliminated.

The key advantage of MCL is that it can represent multi-modal distributions. When the robot does not know which of several places it might be in, it can maintain several hypotheses simultaneously. As observations accumulate, the particles gradually converge to the correct location.

**Example of LiDAR-based MCL**:

```python
import numpy as np

class MonteCarloLocalization:
    """
    2D LiDAR-based Monte Carlo Localization.
    Prior map: occupancy grid.
    """
    
    def __init__(self, occupancy_map, num_particles=1000, 
                 map_resolution=0.05):
        """
        Args:
            occupancy_map: 2D numpy array, 0=free, 1=occupied
            num_particles: number of particles
            map_resolution: map resolution (m/pixel)
        """
        self.map = occupancy_map
        self.resolution = map_resolution
        self.num_particles = num_particles
        
        # Particle initialization: [x, y, theta, weight]
        self.particles = self._initialize_particles()
    
    def _initialize_particles(self):
        """Distribute particles uniformly over the free space of the map."""
        free_cells = np.argwhere(self.map == 0)
        
        if len(free_cells) == 0:
            raise ValueError("The map has no free space.")
        
        # Randomly select free cells
        indices = np.random.choice(len(free_cells), self.num_particles, 
                                   replace=True)
        selected = free_cells[indices]
        
        particles = np.zeros((self.num_particles, 4))
        particles[:, 0] = selected[:, 1] * self.resolution  # x
        particles[:, 1] = selected[:, 0] * self.resolution  # y
        particles[:, 2] = np.random.uniform(-np.pi, np.pi, 
                                             self.num_particles)  # theta
        particles[:, 3] = 1.0 / self.num_particles  # weight (uniform)
        
        return particles
    
    def predict(self, delta_x, delta_y, delta_theta, 
                noise_std=[0.1, 0.1, 0.05]):
        """Move particles with the motion model + noise."""
        noise = np.random.randn(self.num_particles, 3) * noise_std
        
        cos_theta = np.cos(self.particles[:, 2])
        sin_theta = np.sin(self.particles[:, 2])
        
        # Convert motion in the robot frame to the global frame
        self.particles[:, 0] += (delta_x * cos_theta 
                                  - delta_y * sin_theta + noise[:, 0])
        self.particles[:, 1] += (delta_x * sin_theta 
                                  + delta_y * cos_theta + noise[:, 1])
        self.particles[:, 2] += delta_theta + noise[:, 2]
        
        # Normalize angles
        self.particles[:, 2] = np.arctan2(
            np.sin(self.particles[:, 2]), 
            np.cos(self.particles[:, 2])
        )
    
    def update(self, scan_ranges, scan_angles, sigma_hit=0.2):
        """
        Update particle weights with sensor observations.
        
        Args:
            scan_ranges: actual LiDAR range measurements (N,)
            scan_angles: angle of each beam (N,)
            sigma_hit: sensor noise standard deviation
        """
        for k in range(self.num_particles):
            x, y, theta = self.particles[k, :3]
            
            log_weight = 0.0
            for r_measured, angle in zip(scan_ranges, scan_angles):
                # Compute expected range at this particle's location (ray casting)
                r_expected = self._ray_cast(x, y, theta + angle)
                
                # Gaussian sensor model
                diff = r_measured - r_expected
                log_weight += -0.5 * (diff / sigma_hit) ** 2
            
            self.particles[k, 3] = np.exp(log_weight)
        
        # Normalize weights
        total = np.sum(self.particles[:, 3])
        if total > 0:
            self.particles[:, 3] /= total
        else:
            self.particles[:, 3] = 1.0 / self.num_particles
    
    def resample(self):
        """Low-variance resampling."""
        weights = self.particles[:, 3]
        N = self.num_particles
        
        new_particles = np.zeros_like(self.particles)
        r = np.random.uniform(0, 1.0 / N)
        c = weights[0]
        i = 0
        
        for m in range(N):
            u = r + m / N
            while u > c:
                i += 1
                c += weights[i]
            new_particles[m] = self.particles[i].copy()
        
        new_particles[:, 3] = 1.0 / N
        self.particles = new_particles
    
    def get_estimate(self):
        """Estimate the current pose as a weighted average."""
        weights = self.particles[:, 3]
        x_est = np.average(self.particles[:, 0], weights=weights)
        y_est = np.average(self.particles[:, 1], weights=weights)
        
        # Weighted average of angles (circular mean)
        sin_avg = np.average(np.sin(self.particles[:, 2]), weights=weights)
        cos_avg = np.average(np.cos(self.particles[:, 2]), weights=weights)
        theta_est = np.arctan2(sin_avg, cos_avg)
        
        return x_est, y_est, theta_est
    
    def _ray_cast(self, x, y, angle, max_range=30.0):
        """Simple Bresenham-based ray casting."""
        dx = np.cos(angle) * self.resolution
        dy = np.sin(angle) * self.resolution
        
        cx, cy = x, y
        for step in range(int(max_range / self.resolution)):
            cx += dx
            cy += dy
            
            # Convert to map coordinates
            mx = int(cy / self.resolution)
            my = int(cx / self.resolution)
            
            if (mx < 0 or mx >= self.map.shape[0] or 
                my < 0 or my >= self.map.shape[1]):
                return max_range
            
            if self.map[mx, my] == 1:  # obstacle hit
                return step * self.resolution
        
        return max_range
```

---

## 10.4 Multi-Session SLAM & Map Merging

In practical environments, SLAM is often not completed in a single run. The same building may be mapped over several days, multiple robots may simultaneously explore different areas, or tracking may fail and then resume from a different point. Multi-session SLAM integrates the maps from such separate sessions into a single globally consistent map.

### 10.4.1 Map Anchoring

To integrate maps from multiple sessions, the coordinate frame of each map must be aligned. This is called **map anchoring**.

**Method 1: Based on shared landmarks**: If two sessions observe the same 3D points or feature points in an overlapping region, the relative transform ${}^{A}\mathbf{T}_{B}$ between the two maps is estimated from them.

$${}^{A}\mathbf{T}_{B} = \arg\min_{\mathbf{T}} \sum_k \| \mathbf{p}_k^A - \mathbf{T} \cdot \mathbf{p}_k^B \|^2$$

**Method 2: Based on place recognition**: Even when shared landmarks are not explicitly available, place recognition can find keyframe pairs from the two sessions that visited the same location, and alignment is performed from these pairs.

**Method 3: Based on GNSS**: If both sessions have GNSS information, it can be used as a shared coordinate frame for direct alignment.

### 10.4.2 Inter-Session Loop Closure

Once map anchoring provides an initial alignment, inter-session loop closure performs the precise registration. The principle is the same as standard loop closure, but two additional challenges arise:

1. **Appearance change**: Over time, lighting, season, and furniture arrangement change. Foundation-model-based descriptors such as AnyLoc are robust to this problem.

2. **Frame misalignment**: Because the initial alignment may be inaccurate, the tolerance of geometric verification must be increased.

### 10.4.3 ORB-SLAM3 Multi-Map System

The Atlas system of ORB-SLAM3 is a representative implementation of multi-session SLAM. The core mechanisms are as follows:

1. **Active map**: The map currently being tracked. When operating normally, keyframes and map points are added to this map.

2. **Map creation**: If tracking fails (e.g., due to occlusion or a textureless environment), a new map is created and set as the active map. The previous map is kept in an inactive state inside Atlas.

3. **Map merging**: When place recognition detects a common location between the current active map and an inactive map in Atlas, the two maps are merged:
   - Estimate the relative transform $\mathbf{T}_{merge}$ from the common keyframe pair.
   - Transform all keyframes and map points of the inactive map by $\mathbf{T}_{merge}$.
   - Fuse common map points.
   - Ensure consistency in the merged region via welding bundle adjustment.

4. **Leveraging multi-session**: A map from a previous session can be loaded, and when the current session visits the same location, it is automatically merged. This allows the map to be extended incrementally across multiple sessions.

```python
class MultiMapAtlas:
    """
    Core concepts of the ORB-SLAM3 Atlas system.
    """
    
    def __init__(self):
        self.maps = {}           # {map_id: MapData}
        self.active_map_id = None
        self.next_map_id = 0
    
    def create_new_map(self, initial_pose):
        """Create a new map and set it as active."""
        map_id = self.next_map_id
        self.next_map_id += 1
        
        self.maps[map_id] = {
            'keyframes': [initial_pose],
            'map_points': [],
            'is_active': True,
            'origin': initial_pose.copy()
        }
        
        # Deactivate the previous active map
        if self.active_map_id is not None:
            self.maps[self.active_map_id]['is_active'] = False
        
        self.active_map_id = map_id
        return map_id
    
    def on_tracking_lost(self, last_pose):
        """
        On tracking failure: preserve the current map and create a new one.
        """
        print(f"Tracking lost. Map {self.active_map_id} "
              f"saved with {len(self.maps[self.active_map_id]['keyframes'])} "
              f"keyframes.")
        self.create_new_map(last_pose)
    
    def try_merge_maps(self, current_kf_descriptor, current_kf_pose):
        """
        Search inactive maps for a match using the current keyframe's descriptor.
        If a match is found, merge the two maps.
        """
        for map_id, map_data in self.maps.items():
            if map_id == self.active_map_id:
                continue
            
            # Search inactive-map keyframes via place recognition
            match_idx = self._search_in_map(current_kf_descriptor, map_id)
            
            if match_idx is not None:
                # Geometric verification
                T_merge = self._compute_merge_transform(
                    current_kf_pose, 
                    map_data['keyframes'][match_idx]
                )
                
                if T_merge is not None:
                    self._merge_maps(self.active_map_id, map_id, T_merge)
                    return True
        
        return False
    
    def _merge_maps(self, active_id, merge_id, T_merge):
        """
        Merge the merge_id map into the active_id map.
        Transform all keyframes/map points of merge_id by T_merge.
        """
        merge_map = self.maps[merge_id]
        active_map = self.maps[active_id]
        
        # Transform the keyframes of the inactive map and add them to the active map
        for kf in merge_map['keyframes']:
            transformed_kf = T_merge @ kf
            active_map['keyframes'].append(transformed_kf)
        
        # Map points are also transformed and added
        for mp in merge_map['map_points']:
            transformed_mp = T_merge @ mp
            active_map['map_points'].append(transformed_mp)
        
        # Remove the merged map
        del self.maps[merge_id]
        
        print(f"Maps {active_id} and {merge_id} merged. "
              f"Total keyframes: {len(active_map['keyframes'])}")
    
    def _search_in_map(self, descriptor, map_id):
        """Perform place recognition in an inactive map (placeholder)."""
        return None  # Actual implementation would compare descriptors
    
    def _compute_merge_transform(self, pose_a, pose_b):
        """Compute the transform between two poses (placeholder)."""
        return None  # Actual implementation would include geometric verification
```

### 10.4.4 Multi-Robot Map Merging

When multiple robots explore simultaneously, each robot's map must be integrated in real time. The additional constraints are:

- **Communication bandwidth**: Full point clouds cannot be transmitted, so only compressed descriptors (Scan Context, compact visual descriptors) are exchanged.
- **Distributed optimization**: Robots must be able to merge maps autonomously without a central server. Kimera-Multi and Swarm-SLAM address this problem.
- **Relative pose uncertainty**: Because the initial relative pose between robots is unknown, alignment must be achieved via inter-robot loop closures.

The core idea of distributed pose graph optimization:

$$\mathbf{T}^* = \arg\min \sum_{\text{robot } r} \sum_{(i,j) \in \mathcal{E}_r} \rho(\mathbf{e}_{ij}) + \sum_{(i,j) \in \mathcal{E}_{\text{inter}}} \rho(\mathbf{e}_{ij})$$

Each robot performs the optimization over its own edges $\mathcal{E}_r$ locally, and exchanges information only for the inter-robot edges $\mathcal{E}_{\text{inter}}$. Distributed convergence can be achieved via ADMM (Alternating Direction Method of Multipliers) or Gauss-Seidel iteration.

---

## 10.5 Recent Research (2024-2025)

**[riSAM (McGann et al., 2023)](https://arxiv.org/abs/2209.14359)**: A robust backend that integrates **Graduated Non-Convexity (GNC)** into iSAM2 to remove outlier loop closures online in incremental SLAM. It operates robustly even with more than 90% outlier measurements and achieves, in real time, performance comparable to existing offline methods. [The theoretical foundation of GNC was presented by Yang et al. (2020)](https://arxiv.org/abs/1909.08605).

**[Kimera2 (Abate et al., 2024)](https://arxiv.org/abs/2401.06323)**: The next-generation version of the Kimera SLAM library, which replaces the backend's outlier rejection from PCM with GNC, significantly improving robustness. It has been validated on diverse platforms such as drones, quadruped robots, and autonomous vehicles, and includes comprehensive improvements for the practical deployment of metric-semantic SLAM.

**[Group-k Consistent Measurement Set Maximization (Forsgren & Kaess, 2022)](https://arxiv.org/abs/2209.02658)**: Extends PCM's pairwise consistency to group-k consistency, enabling stricter outlier detection. In multi-robot map merging, it further suppresses false positives relative to PCM.

---

Through loop closure and global optimization, a SLAM system produces a globally consistent trajectory and map. But what form does this "map" take? Is it a point cloud, a grid, or a neural network? In the next chapter, we examine the various forms of **spatial representation** — the final output of sensor fusion — along with their strengths and weaknesses.
