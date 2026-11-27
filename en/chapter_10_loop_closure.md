# Chapter 10. Loop Closure and Global Optimization

Place recognition (→ Chapter 9) answers whether the current observation matches a location seen earlier, yet retrieval alone does not fix accumulated drift. Before those candidate matches can bend a drifting trajectory back into global alignment, they must pass geometric scrutiny and enter the SLAM backend as constraints.

Every odometry or inertial-fusion frontend drifts over time. Even with tightly calibrated sensors, tiny frame-to-frame errors in rotation and translation compound with distance traveled and gradually warp the reconstructed environment. Loop closure breaks this open-loop error growth by recognizing a revisited place and tying the current pose back to an earlier point in the trajectory.

A complete loop-closure system runs in three stages—detection, verification, and correction—with pose graph optimization at the heart of the final step. The same machinery also underpins global relocalization against a prior map and multi-session map merging.

---

## 10.1 Loop Closure Pipeline

Closing a loop proceeds through three consecutive stages: **Detection** (finding candidate matches), **Verification** (testing geometric consistency), and **Correction** (optimizing the graph). The asymmetry between errors is stark: missing a valid loop merely leaves drift uncorrected for a little longer, whereas admitting a single false loop can fold the entire map onto itself.

### 10.1.1 Detection: Candidate Search

Loop closure detection asks which past observation resembles the current sensor frame—essentially the same retrieval problem addressed by place recognition (→ Chapter 9).

In **visual loop closure detection**, the system encodes the current image into a global descriptor and queries a database of past keyframes:

1. **Bag-of-Words (traditional)**: Systems such as ORB-SLAM3 use DBoW2 to compare histograms of quantized ORB descriptors ("visual words"). Each keyframe is represented as a bag-of-words vector $\mathbf{v}_i$, and its similarity to the query vector $\mathbf{v}_q$ is measured by the L1-score $s(\mathbf{v}_q, \mathbf{v}_i) = 1 - \frac{1}{2} \left| \frac{\mathbf{v}_q}{\|\mathbf{v}_q\|} - \frac{\mathbf{v}_i}{\|\mathbf{v}_i\|} \right|$.

2. **Learning-based descriptors**: Learned global representations such as [NetVLAD](https://arxiv.org/abs/1511.07247) and [AnyLoc](https://arxiv.org/abs/2308.00688) capture broader appearance context. AnyLoc aggregates dense [DINOv2](https://arxiv.org/abs/2304.07193) features with VLAD and transfers across diverse domains without place-specific labels, though recall should still be validated whenever deploying on a new sensor or environment. Candidates are ranked by cosine similarity:

$$s(\mathbf{d}_q, \mathbf{d}_i) = \frac{\mathbf{d}_q^\top \mathbf{d}_i}{\|\mathbf{d}_q\| \|\mathbf{d}_i\|}$$

In **LiDAR loop closure detection**, global descriptors summarize 3D scan geometry instead of image appearance:

- **[Scan Context](https://doi.org/10.1109/IROS.2018.8593953)**: Preserves spatial layout directly by recording the maximum point height in each bin and sector of a sensor-centered polar grid. A two-stage lookup—a fast ring-key search in a kd-tree followed by sector-shift alignment—keeps retrieval efficient and handles reverse revisits reliably.
- **[PointNetVLAD](https://arxiv.org/abs/1804.03492) and [OverlapTransformer](https://arxiv.org/abs/2203.03397)**: Learned 3D place-recognition networks whose recall relative to Scan Context varies with the dataset, scanning pattern, and training domain.

Before ranking descriptors, the detector must also apply **temporal filtering**. Adjacent keyframes naturally look alike because the robot has barely moved; matching against them reflects continuous tracking rather than a true loop revisit. The required time or travel-distance gap depends on platform speed, keyframe rate, and revisit patterns, so parameters such as the 30-second threshold in the example below should be tuned to the deployment. The function below illustrates this temporal mask combined with normalized cosine-similarity retrieval:

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

Because candidate retrieval relies on compressed global descriptors, it discards fine-grained 3D layout and inevitably admits **false positives**—genuinely distinct places that happen to look similar. This perceptual aliasing is especially common indoors, where corridors, stairwells, and office bays repeat the same architectural patterns.

In SLAM, a false positive loop closure is **catastrophic**: even one spurious edge can drag distant parts of the trajectory together and warp the entire map. Geometric verification therefore errs on the side of caution, trading some recall to keep precision as close to 100% as possible.

Depending on the sensor modality and available map structure, systems verify candidates through four complementary checks:

1. **2D–2D Essential-matrix verification**: Match local features between the query and candidate images and estimate the essential matrix $\mathbf{E}$ inside RANSAC under the epipolar constraint:

$$\mathbf{p}_2^\top \mathbf{E} \mathbf{p}_1 = 0, \quad \mathbf{E} = [\mathbf{t}]_\times \mathbf{R}$$

Rather than trusting a raw inlier count alone, inspect the inlier ratio, spatial distribution across the image, plausibility of the recovered pose, and whether neighboring frames agree. Thresholds such as 20 inliers and a 50% inlier ratio in the code below serve only as starting points for a given detector and scene.

2. **3D–3D Point cloud registration**: In LiDAR systems, estimate the relative transform $\mathbf{T}_{ij} \in SE(3)$ between two scans with ICP or GeoTransformer, then gate acceptance on the registration fitness score (fraction of aligned points) and residual RMSE.

3. **2D–3D PnP**: Solve a Perspective-n-Point problem between 2D features in the current frame and 3D map points associated with the candidate keyframe. Notice the coordinate convention: if those 3D points are expressed in the candidate keyframe's local frame, PnP directly yields the relative transform $\mathbf{T}_{ij}$ needed for the loop edge; if they live in the global map frame, PnP returns the query frame's absolute pose, which must then be composed with the candidate's pose.

4. **Temporal and covisibility consistency**: Single-frame checks are brittle, so systems require agreement across multiple views. ORB-SLAM2 demanded detections of the same place across three consecutive keyframes. ORB-SLAM3 replaces that sequential wait with geometric verification over a local covisibility window: a loop is accepted as soon as the query keyframe and two keyframes covisible with it (three in total) support the transform. Since those covisible neighbors already exist in the map, the system closes loops without waiting for new frames to arrive.

The implementation below demonstrates 2D–2D essential-matrix verification using Sampson error inside RANSAC (remember that monocular essential geometry leaves translation scale normalized unless paired with depth, stereo, or map points):

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
        T_relative: relative transform (4, 4) or None.
                    The t recovered from an essential matrix is a unit
                    direction with no determined magnitude, so using this
                    value as a pose graph loop edge requires obtaining the
                    metric scale separately from stereo, depth, or map points.
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
    """Estimate essential matrix via the 8-point linear solver + RANSAC.

    Nister's 5-point algorithm, which solves from a minimal sample of 5,
    is a separate solver; its minimal sample size differs, so the number of
    RANSAC iterations it requires differs as well.
    """
    best_E = None
    best_inliers = np.zeros(len(pts1), dtype=bool)
    
    for _ in range(max_iter):
        # Randomly sample 8 points
        idx = np.random.choice(len(pts1), 8, replace=False)
        
        # Generate an E candidate with the 8-point linear solver (a separate solver from 5-point, not a simplification of it)
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

To see how quickly a single bad match can unravel a map, consider a robot traversing two nearly identical corridors. If an erroneous loop closure links keyframe $i$ in corridor A to keyframe $j$ in corridor B, the optimizer pulls those two distant poses together, twisting every intermediate pose along the path and folding the map onto itself.

Because no single test catches every alias, practical SLAM systems stack defenses across both the front-end and back-end:

1. **Multi-stage verification**: Require every candidate to pass appearance similarity, geometric consistency, and temporal or covisibility consistency in sequence before creating a loop edge.

2. **Robust kernels** (detailed in §10.2): Reshape the optimization cost so that large outlier residuals exert bounded influence rather than dominating the solver.

3. **[Switchable constraints (Sünderhauf & Protzel, 2012)](https://doi.org/10.1109/IROS.2012.6385590)**: Attach an explicit continuous switch variable to each loop closure factor, letting the optimizer turn off edges that conflict with the rest of the graph.

4. **[DCS (Dynamic Covariance Scaling) (Agarwal et al., 2013)](https://doi.org/10.1109/ICRA.2013.6630733)**: Scale each loop edge's covariance dynamically according to its residual, automatically inflating the uncertainty of suspect matches.

5. **[PCM (Pairwise Consistency Maximization) (Mangelson et al., 2018)](https://doi.org/10.1109/ICRA.2018.8460217)**: Test pairwise geometric agreement across multiple loop candidates and commit only to the largest mutually consistent subset.

### 10.1.4 Correction: Graph Correction

Once a loop candidate survives verification, it enters the pose graph as a constraint encoding the measured relative transform $\mathbf{T}_{ij}$ and its uncertainty $\boldsymbol{\Sigma}_{ij}$, producing the residual error:

$$e_{ij} = \text{Log}(\mathbf{T}_{ij}^{-1} \cdot \mathbf{T}_i^{-1} \cdot \mathbf{T}_j)$$

With this new edge in place, the backend optimizer (§10.2) re-solves the graph to pull the drifting trajectory back into alignment. Notice that optimization adjusts the node poses rather than the edge measurements themselves, redistributing residuals across the entire loop. That redistribution is uneven by design: guided by graph topology and each edge's information matrix $\boldsymbol{\Omega}_{ij}$, high-confidence odometry segments deform very little while uncertain stretches absorb more of the correction.

---

## 10.2 Pose Graph Optimization

Pose graph optimization reconciles local incremental constraints from odometry with global loop-closure constraints, restoring global consistency across the full trajectory.

### 10.2.1 SE(3) Pose Graph

Formally, the trajectory is modeled as a graph $\mathcal{G} = (\mathcal{V}, \mathcal{E})$, where the nodes $\mathcal{V} = \{\mathbf{T}_1, \mathbf{T}_2, \ldots, \mathbf{T}_n\}$ represent keyframe poses $\mathbf{T}_i \in SE(3)$ and the edges $\mathcal{E}$ encode relative pose measurements—both sequential odometry edges and long-range loop edges. Each edge $(i, j) \in \mathcal{E}$ carries a measured relative transform $\tilde{\mathbf{T}}_{ij}$ together with an information matrix $\boldsymbol{\Omega}_{ij} = \boldsymbol{\Sigma}_{ij}^{-1}$ reflecting measurement confidence.

Because $SE(3)$ poses do not live in a flat vector space, edge residuals cannot be computed by simple subtraction. Instead, we compare the predicted relative pose $\mathbf{T}_i^{-1}\mathbf{T}_j$ against the measurement $\tilde{\mathbf{T}}_{ij}$ on the Lie group and map the discrepancy to the Lie algebra via the matrix logarithm:

$$\mathbf{e}_{ij} = \text{Log}(\tilde{\mathbf{T}}_{ij}^{-1} \cdot \mathbf{T}_i^{-1} \cdot \mathbf{T}_j) \in \mathbb{R}^6$$

Here $\text{Log}: SE(3) \to \mathfrak{se}(3) \cong \mathbb{R}^6$ produces a six-dimensional tangent vector packing three translational and three rotational error components. Summing the information-weighted squared errors across all edges gives the global optimization objective:

$$\mathbf{T}^* = \arg\min_{\mathbf{T}_1, \ldots, \mathbf{T}_n} \sum_{(i,j) \in \mathcal{E}} \mathbf{e}_{ij}^\top \boldsymbol{\Omega}_{ij} \mathbf{e}_{ij}$$

We solve this nonlinear least-squares problem iteratively with Gauss-Newton or Levenberg-Marquardt. Because adding a raw vector directly to a rotation matrix breaks orthogonality, each step computes a small tangent-space increment $\boldsymbol{\delta}_i \in \mathbb{R}^6$ and applies it back onto the manifold using the exponential map:

$$\mathbf{T}_i \leftarrow \mathbf{T}_i \cdot \text{Exp}(\boldsymbol{\delta}_i)$$

One Gauss-Newton iteration proceeds as follows:

1. Linearize around the current pose estimates by evaluating each edge's error $\mathbf{e}_{ij}$ and Jacobian $\mathbf{J}_{ij}$.
2. Assemble the sparse normal equations $\mathbf{H} \boldsymbol{\delta} = -\mathbf{b}$, where:

$$\mathbf{H} = \sum_{(i,j)} \mathbf{J}_{ij}^\top \boldsymbol{\Omega}_{ij} \mathbf{J}_{ij}, \quad \mathbf{b} = \sum_{(i,j)} \mathbf{J}_{ij}^\top \boldsymbol{\Omega}_{ij} \mathbf{e}_{ij}$$

3. Solve the linear system via sparse Cholesky decomposition—after addressing **gauge freedom**. Because every edge constrains only *relative* transforms between pairs of nodes, rigidly shifting and rotating the entire trajectory in $SE(3)$ leaves all residuals $\mathbf{e}_{ij}$ untouched. That global invariance gives $\mathbf{H}$ a six-dimensional null space, making it singular until one reference pose is held fixed or anchored with a prior factor so that $\mathbf{H}$ becomes positive definite.
4. Retract each pose onto $SE(3)$ via $\mathbf{T}_i \leftarrow \mathbf{T}_i \cdot \text{Exp}(\boldsymbol{\delta}_i)$.
5. Repeat until the cost change or step norm falls below a convergence tolerance.

In practice, frameworks such as **[g2o (General Graph Optimization)](https://doi.org/10.1109/ICRA.2011.5979949)** (Kümmerle et al., 2011) automate this pipeline: you specify vertex and edge types along with their error functions, and the solver handles sparse linearization, ordering, and Cholesky factorization. The self-contained Python functions below make the underlying $SE(3)$ Lie-algebra operations and cost evaluation explicit:

```python
import numpy as np
from scipy.spatial.transform import Rotation

def se3_log(T):
    """SE(3) matrix -> 6-dimensional vector [translation; rotation]."""
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

No front-end pipeline rejects 100% of bad measurements. Dynamic objects, registration slips, and perceptual aliasing inevitably leak occasional outliers into the pose graph. Under the standard quadratic penalty $\rho(s) = s$ (where $s = e^2$ is the squared Mahalanobis residual), the gradient grows linearly with the error—meaning a single grossly mismatched loop closure pulls the entire trajectory harder than dozens of valid odometry edges combined.

To prevent outliers from hijacking the solution, a **robust kernel** (or M-estimator) replaces the quadratic penalty with a sublinear or bounded function $\rho(s)$ that flattens as $s$ grows large:

| Kernel | $\rho(s)$ ($s = e^2$) | Characteristics |
|--------|----------------------|------|
| Least Squares | $s$ | Vulnerable to outliers |
| Huber | $\begin{cases} s & \text{if } \sqrt{s} \leq k \\ 2k\sqrt{s} - k^2 & \text{otherwise} \end{cases}$ | Linear beyond threshold $k$ |
| Cauchy | $k^2 \log(1 + s/k^2)$ | Smooth attenuation |
| Geman-McClure | $\frac{s}{k^2 + s}$ | Strong outlier suppression |

Substituting $\rho$ into the pose-graph objective bounds the cost of large edge discrepancies:

$$\sum_{(i,j)} \rho\left(\mathbf{e}_{ij}^\top \boldsymbol{\Omega}_{ij} \mathbf{e}_{ij}\right)$$

Rather than writing a bespoke solver for each robust function, we optimize this objective via **IRLS (Iteratively Reweighted Least Squares)**. Linearizing $\rho$ around the current squared residual converts each step back into a standard weighted least-squares problem whose edge weights depend on the residual magnitude:

$$w_i = \rho'(s_i), \quad s_i = \mathbf{e}_i^\top \boldsymbol{\Omega}_i \mathbf{e}_i$$

Here $\rho'$ denotes the derivative of $\rho(s)$ with respect to $s$ as defined in the table above. Because $\rho'(s_i)$ shrinks whenever $s_i$ becomes large, outlier edges automatically lose leverage in the normal equations while inlier edges keep weights near one.

Fixed robust kernels still have a blind spot: when an incorrect loop closure enters the graph early or carries strong local support, an IRLS solver can still get trapped in a distorted local minimum. Two dynamic mechanisms address this directly at the factor level:

- **[Switchable constraints](https://doi.org/10.1109/IROS.2012.6385590)**: Sünderhauf & Protzel (2012) augment each loop closure factor with a continuous switch variable $s_{ij} \in [0, 1]$:

$$\rho_{\text{switch}}(\mathbf{e}_{ij}, s_{ij}) = s_{ij}^2 \mathbf{e}_{ij}^\top \boldsymbol{\Omega}_{ij} \mathbf{e}_{ij} + \lambda (1 - s_{ij})^2$$

  The first term drives the residual down when the switch is active, while the prior penalty $\lambda (1 - s_{ij})^2$ resists disabling valid loops. Whenever a loop edge conflicts violently with the surrounding odometry, paying the fixed penalty $\lambda$ becomes cheaper than distorting the trajectory, so the optimizer drives $s_{ij} \to 0$ and cleanly switches the outlier off.

- **[DCS (Dynamic Covariance Scaling)](https://doi.org/10.1109/ICRA.2013.6630733)**: Agarwal et al. (2013) eliminate the extra switch variables by deriving a closed-form scaling factor from the residual magnitude, dynamically inflating the covariance of inconsistent loop edges so their weight fades without adding optimization state.

The Python functions below implement the Huber and Cauchy robust kernels, the Huber IRLS weight $\rho'(s)$, and the resulting robust pose-graph cost:

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

Re-solving the entire pose graph from scratch whenever a new keyframe arrives does not scale to long missions: almost all earlier variables and measurements remain unchanged between consecutive frames. Instead of running full batch optimization on every update, [iSAM2 (Kaess et al., 2012)](https://doi.org/10.1177/0278364911430419) maintains the matrix factorization incrementally, isolating and recomputing only the portion of the graph touched by new measurements.

The data structure that makes this selective update possible is the **Bayes tree**, which arises naturally by giving direction to the clique tree produced when eliminating variables in a factor graph. In MAP estimation, the joint posterior over all poses and landmarks factors into a product of local measurement factors $f_k(\mathbf{x}_k)$:

$$p(\mathbf{x} | \mathbf{z}) \propto \prod_k f_k(\mathbf{x}_k)$$

Eliminating variables one by one converts this undirected factor graph into a product of conditional densities:

$$p(\mathbf{x} | \mathbf{z}) = \prod_i p(x_i | \text{Sep}(x_i))$$

where $\text{Sep}(x_i)$ denotes the separator of $x_i$ in the clique tree—the remaining conditioning variables still coupled to $x_i$ at the moment it is eliminated. Grouping variables that share the same elimination structure forms cliques organized into a rooted tree, the Bayes tree.

Because dependencies flow upward toward the root, an incremental update only disturbs the subtree along the path from the newly constrained variables to the root:

1. **Affected clique detection**: Identify the cliques touched by the newly added factor—typically just a handful during normal odometry.
2. **Partial re-factorization**: Re-run QR decomposition only for those affected cliques and their ancestors up to the root.
3. **Tree reuse**: Leave all unaffected branches of the Bayes tree completely untouched.

Two additional mechanisms keep iSAM2 both accurate and fast over long runs. Through **fluid relinearization**, it selectively re-linearizes only those variables whose estimates have drifted beyond a threshold, avoiding iSAM v1's periodic full-batch relinearization (though a large loop closure can still propagate across much of the tree). Through **incremental variable reordering**, it updates the elimination ordering locally around newly added variables rather than recomputing a global fill-reducing ordering for the entire graph.

In practice, iSAM2 lives at the core of the GTSAM library and drives backends such as [LIO-SAM](https://arxiv.org/abs/2007.00258), whereas systems like ORB-SLAM3 rely on a customized g2o solver for bundle adjustment and pose-graph optimization. The simplified class below omits the Bayes tree's clique factorization to illustrate the core idea of tracking affected variables across odometry and loop-closure updates:

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

Whereas loop closure asks *"Have I been here before?"* during an active mapping run, **global relocalization** asks *"Where am I inside an already-built map?"* Whether a tracker has just lost lock or a vehicle is booting up in a mapped facility, the system must recover its 6-DoF or planar pose from scratch against the prior map.

### 10.3.1 Map-Based Localization

With a prior map in hand, relocalization registers live sensor observations back to that map:

- **Visual relocalization**: Extract local keypoints from the current camera image, retrieve candidate keyframes or map landmarks via visual words or descriptor matching, establish 2D–3D correspondences, and recover the pose with PnP inside RANSAC before resuming frame-to-frame tracking. ORB-SLAM3, for instance, retrieves candidate keyframes through DBoW2, matches ORB descriptors, solves pose with MLPnP + RANSAC (replacing the EPnP solver used through ORB-SLAM2 with a formulation independent of a specific camera model), and refines the estimate via guided local feature matching.
- **LiDAR relocalization**: Register the live scan against a prior point-cloud map in three steps: narrow the search space with a global descriptor such as Scan Context or PointNetVLAD, compute a coarse alignment via FPFH + RANSAC or GeoTransformer, and refine the pose to centimeter accuracy with ICP or GICP.

### 10.3.2 Prior Map + Online Sensor

Production autonomous vehicles routinely localize in real time against pre-built HD maps using live onboard sensors. Deploying that pairing in the real world surfaces three persistent challenges:

- **Map staleness and environmental change**: Construction, seasonal foliage, and parked vehicles alter scenes over time, so localizers must stay robust to discrepancies between the stored map and live observations.
- **Cross-modal mismatch**: An HD map may have been surveyed with LiDAR while a fleet vehicle carries only cameras, demanding cross-modal registration across fundamentally different sensing modalities.
- **Cold-start initialization**: When the vehicle wakes up with no prior pose estimate—for instance inside an underground garage where GNSS is unavailable—global place recognition over the entire map must bootstrap the initial pose.

### 10.3.3 Monte Carlo Localization (MCL)

When a robot has no initial pose guess, a single Gaussian state estimate cannot represent competing hypotheses at once. **Monte Carlo Localization (MCL)** solves global localization with a particle filter, approximating the posterior pose distribution using a swarm of weighted candidate poses (particles).

Once initialized, MCL cycles through prediction, measurement update, and resampling:

1. **Initialization**: Scatter particles uniformly across the map's free space to express complete global uncertainty.
2. **Prediction**: Propagate each particle forward using the probabilistic motion model:
$$x_t^{[k]} \sim p(x_t | u_t, x_{t-1}^{[k]})$$
3. **Update**: Score each particle by comparing the live sensor scan against the observation expected at that pose on the map:
$$w_t^{[k]} = p(z_t | x_t^{[k]}, m)$$
4. **Resampling**: Draw a new particle generation with probability proportional to particle weights, replicating hypotheses that match the measurements and culling inconsistent ones.

Because a particle set is non-parametric, MCL naturally maintains **multi-modal distributions** whenever multiple locations look plausible, collapsing onto a single pose only as distinct landmarks break the symmetry. In genuinely ambiguous layouts—such as long, symmetric corridors—multiple modes properly persist until a distinguishing junction appears; conversely, using too few particles or an overly peaked likelihood model can prematurely deplete the true mode and lock the filter onto the wrong location.

The 2D LiDAR implementation below illustrates this full predict–update–resample loop on an occupancy grid:

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
            
            self.particles[k, 3] = log_weight   # collect log-weights first
        
        # Normalize weights -- subtract the max log-weight before exp (the log-sum over hundreds of beams underflows if exponentiated directly)
        log_w = self.particles[:, 3]
        self.particles[:, 3] = np.exp(log_w - np.max(log_w))
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
        """Simple ray casting based on fixed-step sampling.

        Cells are looked up by adding a direction vector to real-valued
        coordinates, so cells can be skipped or queried more than once.
        When every cell the line passes through must be visited exactly
        once, use the Amanatides-Woo DDA. Bresenham avoids repeated lookups,
        but as an 8-connected rasterization it still skips some cells the
        line passes through.
        """
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
                return (step + 1) * self.resolution   # we advance before checking, so the checked point lies (step+1) cells away
        
        return max_range
```

---

## 10.4 Multi-Session SLAM & Map Merging

Real-world mapping rarely happens in one uninterrupted trajectory. A building may be surveyed over several days, a team of robots may explore different wings simultaneously, or tracking may drop out and restart elsewhere. **Multi-session SLAM** stitches those separate runs into a single, globally consistent coordinate frame.

### 10.4.1 Map Anchoring

Before separate sub-maps can be fused, their independent coordinate origins must be tied together—a step known as **map anchoring**. Depending on what overlaps or external signals exist, systems establish that relative transform in three ways:

1. **Shared landmarks**: When two sessions observe common 3D map points or landmarks in an overlapping region, the relative transform ${}^{A}\mathbf{T}_{B}$ follows directly by aligning corresponding points:

$${}^{A}\mathbf{T}_{B} = \arg\min_{\mathbf{T}} \sum_k \| \mathbf{p}_k^A - \mathbf{T} \cdot \mathbf{p}_k^B \|^2$$

2. **Place recognition**: Even without pre-associated landmarks, place recognition can match keyframes across sessions that revisited the same spot, providing anchor pairs for relative-pose estimation.

3. **GNSS**: Outdoors, when both sessions log GNSS measurements, satellite coordinates anchor each map directly into a shared global reference frame.

### 10.4.2 Inter-Session Loop Closure

Once anchoring yields a rough alignment, **inter-session loop closure** refines the registration between trajectories. The mechanics mirror single-session loop closure, except two hazards grow sharper across separate runs:

1. **Appearance change**: Hours, lighting, weather, and seasons shift between sessions. Foundation-model descriptors such as AnyLoc handle these visual shifts far better than brittle hand-crafted local descriptors.

2. **Frame misalignment**: Because initial map anchoring may be coarse, widen the candidate search rather than relaxing verification standards. Open up the pose prior's search gate, retrieve more candidates, run additional RANSAC iterations, and broaden the basin of initial guesses passed to registration—but never loosen geometric verification's residual acceptance criteria, or false inter-session loops will pass straight through and corrupt both maps.

### 10.4.3 ORB-SLAM3 Multi-Map System

ORB-SLAM3's **Atlas** architecture gives a concrete blueprint for managing and merging multiple maps automatically:

1. **Active map**: Normal tracking appends new keyframes and map points to the single currently active map.
2. **Map creation**: Whenever tracking fails (for example due to occlusion or a textureless wall), Atlas preserves the current map in an inactive pool and spawns a fresh active map from the new origin.
3. **Map merging**: Whenever place recognition links the active map to an inactive map in Atlas, the two maps fuse seamlessly:
   - Estimate the relative transform $\mathbf{T}_{merge}$ from the matched keyframe pair.
   - Transform all keyframes and map points of the inactive map by $\mathbf{T}_{merge}$.
   - Fuse duplicate map points across the overlap.
   - Run welding bundle adjustment to restore local and global consistency across the seam.
4. **Cross-session persistence**: A map saved from an earlier run can be reloaded as an inactive map in Atlas; as soon as the live session crosses an overlapping area, the old and new maps merge automatically.

The skeleton below captures Atlas's active/inactive map lifecycle and merge hook:

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

When a fleet of robots explores an environment in parallel, their individual maps must be reconciled in real time under three additional operational constraints:

- **Communication bandwidth**: Robots rarely have bandwidth to broadcast raw point clouds or dense images, so candidate search shares only compact global descriptors (such as Scan Context or compressed visual descriptors). Only after a candidate fires do robots exchange sparse local features or downsampled scans for geometric verification, and distributed optimization exchanges just the boundary poses and uncertainties adjacent to inter-robot links.
- **Distributed optimization**: Without relying on a central server, robots must solve a joint pose graph autonomously—the setting addressed by systems such as Kimera-Multi and Swarm-SLAM.
- **Unknown initial relative poses**: Because robots start in arbitrary, uncalibrated frames, global alignment must emerge directly from inter-robot loop closures.

Formally, the distributed pose-graph objective splits into each robot's internal edges $\mathcal{E}_r$ plus the cross-robot loop edges $\mathcal{E}_{\text{inter}}$:

$$\mathbf{T}^* = \arg\min \sum_{\text{robot } r} \sum_{(i,j) \in \mathcal{E}_r} \rho(\mathbf{e}_{ij}) + \sum_{(i,j) \in \mathcal{E}_{\text{inter}}} \rho(\mathbf{e}_{ij})$$

Each robot optimizes over its own local edges $\mathcal{E}_r$ while exchanging boundary-state messages across $\mathcal{E}_{\text{inter}}$, typically via distributed ADMM or Gauss-Seidel block coordinate descent. Care is needed with theoretical guarantees, however: because pose graphs involve 3D rotations, the objective is non-convex, so convex ADMM convergence theorems do not apply off the shelf. Even reaching a first-order critical point depends on verifying the algorithm's specific hypotheses—for instance, [non-convex ADMM convergence results](https://arxiv.org/abs/1410.1390) require suitable problem structure and a sufficiently large penalty parameter, and both initialization and communication topology affect the final solution. Rather than relying on ADMM or Gauss-Seidel alone, [certifiably correct methods such as DC2-PGO](https://arxiv.org/abs/1911.03721) solve the formulated relaxation under its required conditions and verify global optimality a posteriori.

---

## 10.5 Outlier-Robust Incremental Back-Ends (2022–2024)

Recent backend research pushes robust incremental optimization and multi-robot outlier rejection even further:

- **[riSAM (McGann et al., 2023)](https://arxiv.org/abs/2209.14359)** brings [Graduated Non-Convexity](https://arxiv.org/abs/1909.08605) (GNC) into an incremental robust backend. Under its synthetic and experimental benchmarks, the paper reports coping with outlier rates above 90% alongside competitive comparisons to offline baselines—though performance on arbitrary outlier structures or poor initializations still requires validation.
- **[Kimera2 (Abate et al., 2024)](https://arxiv.org/abs/2401.06323)** upgrades the Kimera SLAM library by swapping PCM outlier rejection in the backend for GNC, improving resilience across drones, autonomous vehicles, and broader metric-semantic SLAM deployments.
- **[Group-k Consistent Measurement Set Maximization (Forsgren & Kaess, 2022)](https://arxiv.org/abs/2209.02658)** strengthens PCM's pairwise consistency test to $k$-way group consistency, filtering out false positives more aggressively during multi-robot map merging.

---

Loop closure and global optimization turn drifting odometry into a globally consistent trajectory. Yet a trajectory alone is only the skeleton of a SLAM system—the robot still needs a lasting representation of the world around it. Should that map be a sparse voxel grid, a smooth triangle mesh, a neural radiance or Gaussian field, or a hierarchical scene graph? Answering that question brings us to **spatial representations** (→ Chapter 11).
