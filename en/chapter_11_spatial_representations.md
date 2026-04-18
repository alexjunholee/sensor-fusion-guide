# Ch.11 — Spatial Representations

In Ch.6-10 we covered the process of estimating the robot's trajectory from sensor data and securing global consistency through loop closure. This chapter treats the byproduct and ultimate goal of that process: the **map** — the form in which the robot remembers and uses the world.

The ultimate output of sensor fusion is a **map** — a spatial representation of the environment. What a SLAM system estimates is not only the robot's trajectory but also the structure of the environment observed along that trajectory. The way in which the environment is represented determines what the robot can do: path planning requires free/occupied information, visual rendering requires texture information, and human interaction requires semantic information.

In this chapter, starting from metric maps (quantitative geometric maps), we cover the full spectrum of spatial representations — mesh, neural representation, semantic map, and long-term maintenance.

---

## 11.1 Metric Maps

A metric map represents the geometric structure of the environment in quantitative coordinates. It is the most basic form and is still the most widely used.

### 11.1.1 Occupancy Grid (2D/3D)

An occupancy grid partitions space into a uniform grid and represents each cell probabilistically as occupied/free/unknown.

**2D occupancy grid**: used for planar environments (indoor, single-floor). For each cell $m_i$, the occupancy probability $p(m_i \mid z_{1:t})$ is maintained via Bayesian update.

$$\text{log-odds}(m_i \mid z_{1:t}) = \text{log-odds}(m_i \mid z_{1:t-1}) + \text{log-odds}(m_i \mid z_t) - \text{log-odds}(m_i)$$

Using the log-odds representation turns multiplication into addition, making computation efficient and numerically stable:

$$l(m_i) = \log\frac{p(m_i)}{1 - p(m_i)}$$

**Problem**: representing a 3D environment requires a 3D occupancy grid. Representing an $L \times W \times H$ space at resolution $r$ requires $(L/r)(W/r)(H/r)$ cells. For instance, representing a 100 m $\times$ 100 m $\times$ 10 m space at 5 cm resolution requires about $8 \times 10^9$ cells, roughly 32 GB of memory. This is impractical.

```python
import numpy as np

class OccupancyGrid2D:
    """2D Occupancy Grid — log-odds-based Bayesian update."""
    
    def __init__(self, width_m, height_m, resolution=0.05):
        """
        Args:
            width_m: map width (meters)
            height_m: map height (meters)
            resolution: cell size (meters)
        """
        self.resolution = resolution
        self.width = int(width_m / resolution)
        self.height = int(height_m / resolution)
        
        # Log-odds map: 0 = unknown (prior = 0.5)
        self.log_odds = np.zeros((self.height, self.width), dtype=np.float32)
        
        # Sensor model parameters
        self.l_occ = 0.85   # log-odds(occupied | hit)
        self.l_free = -0.40  # log-odds(free | pass-through)
        
        # Clipping range (prevents probability saturation)
        self.l_min = -5.0
        self.l_max = 5.0
    
    def update(self, robot_x, robot_y, scan_ranges, scan_angles):
        """
        Update the map with a LiDAR scan.
        
        Args:
            robot_x, robot_y: robot position (meters)
            scan_ranges: range of each beam (N,)
            scan_angles: angle of each beam (N,)
        """
        rx = int(robot_x / self.resolution)
        ry = int(robot_y / self.resolution)
        
        for r, angle in zip(scan_ranges, scan_angles):
            # Endpoint of the beam (occupied)
            ex = int((robot_x + r * np.cos(angle)) / self.resolution)
            ey = int((robot_y + r * np.sin(angle)) / self.resolution)
            
            # Bresenham line: robot -> endpoint = free
            cells = self._bresenham(rx, ry, ex, ey)
            for cx, cy in cells[:-1]:  # exclude last cell (that one is occupied)
                if 0 <= cx < self.width and 0 <= cy < self.height:
                    self.log_odds[cy, cx] = np.clip(
                        self.log_odds[cy, cx] + self.l_free,
                        self.l_min, self.l_max
                    )
            
            # Endpoint cell = occupied
            if 0 <= ex < self.width and 0 <= ey < self.height:
                self.log_odds[ey, ex] = np.clip(
                    self.log_odds[ey, ex] + self.l_occ,
                    self.l_min, self.l_max
                )
    
    def get_probability_map(self):
        """Log-odds -> probability conversion."""
        return 1.0 - 1.0 / (1.0 + np.exp(self.log_odds))
    
    def _bresenham(self, x0, y0, x1, y1):
        """Bresenham line algorithm — returns grid cells between two points."""
        cells = []
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy
        
        while True:
            cells.append((x0, y0))
            if x0 == x1 and y0 == y1:
                break
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x0 += sx
            if e2 < dx:
                err += dx
                y0 += sy
        
        return cells
```

### 11.1.2 Voxel Maps: OctoMap, VDB, ikd-tree

To address the memory problem of uniform grids, various adaptive data structures have been proposed.

**[OctoMap](https://doi.org/10.1007/s10514-012-9321-0)** (Hornung et al. 2013): an octree-based probabilistic 3D occupancy map. Space is recursively subdivided into octants, and regions that are entirely occupied or entirely free stop subdividing early (pruning). Through this, empty space is represented with large units and only regions near fine structure are represented with small units, saving memory.

Key properties:
- **Probabilistic update**: uses the same log-odds update as an occupancy grid.
- **Adaptive resolution**: automatically adjusts from the finest resolution (e.g., 2 cm, leaf node) to the coarsest resolution (e.g., several m, near the root).
- **Memory efficiency**: a 64 m$^3$ space can be represented at 1 cm resolution in about 60 MB (hundreds of times smaller than a uniform grid).
- **Limitations**: the cost of tree rebalancing on dynamic insertion/deletion, and slow nearest-neighbor (kNN) search.

```python
class SimpleOctreeNode:
    """Core idea of OctoMap — a recursively 8-way subdivided octree."""
    
    def __init__(self, center, size, depth=0, max_depth=16):
        self.center = np.array(center)  # node center coordinate
        self.size = size                 # edge length of the node
        self.depth = depth
        self.max_depth = max_depth
        self.children = [None] * 8       # 8 children
        self.log_odds = 0.0              # occupancy log-odds
        self.is_leaf = True
    
    def get_child_index(self, point):
        """Determine which child octant the point belongs to."""
        idx = 0
        if point[0] >= self.center[0]: idx |= 1
        if point[1] >= self.center[1]: idx |= 2
        if point[2] >= self.center[2]: idx |= 4
        return idx
    
    def get_child_center(self, child_idx):
        """Compute the center coordinate of a child octant."""
        offset = self.size / 4
        center = self.center.copy()
        center[0] += offset if (child_idx & 1) else -offset
        center[1] += offset if (child_idx & 2) else -offset
        center[2] += offset if (child_idx & 4) else -offset
        return center
    
    def update(self, point, is_occupied, l_occ=0.85, l_free=-0.40):
        """Update the octree with a point."""
        if self.depth >= self.max_depth:
            # Reached maximum depth: update log-odds at the leaf
            self.log_odds += l_occ if is_occupied else l_free
            self.log_odds = np.clip(self.log_odds, -5.0, 5.0)
            return
        
        # Decide the child octant
        child_idx = self.get_child_index(point)
        
        if self.children[child_idx] is None:
            self.children[child_idx] = SimpleOctreeNode(
                self.get_child_center(child_idx),
                self.size / 2,
                self.depth + 1,
                self.max_depth
            )
            self.is_leaf = False
        
        self.children[child_idx].update(point, is_occupied, l_occ, l_free)
    
    def prune(self):
        """
        Merge children if they are all in the same state (pruning).
        This is the core memory-saving technique of OctoMap.
        """
        if self.is_leaf:
            return
        
        # Check that all children exist, are leaves, and are in the same state
        all_same = True
        first_odds = None
        
        for child in self.children:
            if child is None or not child.is_leaf:
                return  # cannot prune
            
            if first_odds is None:
                first_odds = child.log_odds
            elif abs(child.log_odds - first_odds) > 0.01:
                all_same = False
                break
        
        if all_same and first_odds is not None:
            self.log_odds = first_odds
            self.children = [None] * 8
            self.is_leaf = True
```

**OpenVDB**: a sparse volumetric data structure originating from the film VFX industry. Based on a hash map, it stores only activated voxels, making it highly efficient when only a tiny fraction of voxels in a large space are occupied. It is faster to traverse than OctoMap, but the user must implement probabilistic updates directly.

**ikd-tree** (FAST-LIO2): an incremental k-d tree that performs point insertion and deletion in $O(\log n)$ and supports dynamic rebalancing. Used as the map data structure in FAST-LIO2, it adds LiDAR points to the map in real time while efficiently performing nearest-neighbor (kNN) search.

Core operations of ikd-tree:
- **Insertion**: insert a new point into the k-d tree. Partial rebalancing occurs when the imbalance exceeds a threshold.
- **Deletion**: remove old points outside a certain range via lazy deletion.
- **kNN search**: find the nearest map points to an observed point and use them for point-to-plane registration.

OctoMap vs. ikd-tree:

| Property | OctoMap | ikd-tree |
|------|---------|----------|
| Data structure | Octree | k-d tree |
| Probabilistic update | Built-in (log-odds) | None (only point storage) |
| kNN search | Slow | Fast |
| Dynamic insertion/deletion | Possible but slow | $O(\log n)$ |
| Use case | Path planning, exploration | LiDAR odometry map maintenance |

### 11.1.3 Surfel Maps

A surfel (surface element) is a disk-shaped surface primitive that augments a point with a normal vector and radius information. Each surfel is represented as $(\mathbf{p}, \mathbf{n}, r, c)$:

- $\mathbf{p} \in \mathbb{R}^3$: position
- $\mathbf{n} \in \mathbb{R}^3$: normal vector (unit)
- $r \in \mathbb{R}^+$: radius
- $c$: color/confidence

**[ElasticFusion](https://doi.org/10.15607/RSS.2015.XI.001)** (Whelan et al. 2015): a dense SLAM system that builds a surfel map in real time from an RGB-D sensor. The core ideas are:

1. Frame-to-model tracking: register the current frame against a rendering of the surfel map.
2. Map deformation: upon loop closure, non-rigidly deform the entire surfel map via an embedded deformation graph.
3. Surfel fusion: when existing surfels overlap with new observations, merge them via a weighted average.

The advantage of surfels is that they can represent a continuous surface without an explicit mesh. They carry richer surface information than a point cloud while being simpler to update than a mesh.

```python
class Surfel:
    """A single Surface Element."""
    
    def __init__(self, position, normal, radius, color, confidence=1.0):
        self.position = np.array(position, dtype=np.float64)  # (3,)
        self.normal = np.array(normal, dtype=np.float64)      # (3,)
        self.radius = float(radius)
        self.color = np.array(color, dtype=np.float64)        # (3,) RGB
        self.confidence = confidence
        self.update_count = 1
    
    def fuse(self, new_pos, new_normal, new_radius, new_color, 
             new_confidence=1.0):
        """Update the surfel with a new observation (weighted average)."""
        total_w = self.confidence + new_confidence
        
        self.position = (self.confidence * self.position 
                         + new_confidence * new_pos) / total_w
        
        # Normal: weighted average then normalize
        avg_normal = (self.confidence * self.normal 
                      + new_confidence * new_normal) / total_w
        norm = np.linalg.norm(avg_normal)
        if norm > 1e-6:
            self.normal = avg_normal / norm
        
        self.radius = (self.confidence * self.radius 
                       + new_confidence * new_radius) / total_w
        self.color = (self.confidence * self.color 
                      + new_confidence * new_color) / total_w
        
        self.confidence = min(total_w, 100.0)  # confidence upper bound
        self.update_count += 1


class SurfelMap:
    """A simple surfel map implementation."""
    
    def __init__(self, fusion_distance=0.02, fusion_normal_threshold=0.9):
        self.surfels = []
        self.fusion_dist = fusion_distance
        self.fusion_normal_thresh = fusion_normal_threshold
    
    def integrate(self, points, normals, radii, colors):
        """
        Integrate new observations into the map.
        Fuse if close to an existing surfel; otherwise add a new one.
        """
        for p, n, r, c in zip(points, normals, radii, colors):
            fused = False
            
            for surfel in self.surfels:
                dist = np.linalg.norm(surfel.position - p)
                normal_dot = abs(np.dot(surfel.normal, n))
                
                if (dist < self.fusion_dist and 
                    normal_dot > self.fusion_normal_thresh):
                    surfel.fuse(p, n, r, c)
                    fused = True
                    break
            
            if not fused:
                self.surfels.append(Surfel(p, n, r, c))
```

---

## 11.2 Mesh & CAD-level Maps

A mesh represents surfaces as a collection of triangle faces. It is visually more natural than voxels or point clouds and can be used directly in physical simulation (collision detection, etc.).

### 11.2.1 TSDF (Truncated Signed Distance Function)

A TSDF stores the signed distance from each voxel to the nearest surface:

- Positive: in front of the surface (free space)
- Negative: behind the surface (occupied)
- Zero crossing: the actual surface location

**TSDF update**: whenever a new depth observation arrives, the voxels along the corresponding ray are updated:

$$\text{TSDF}(\mathbf{v}) \leftarrow \frac{W(\mathbf{v}) \cdot \text{TSDF}(\mathbf{v}) + w_{\text{new}} \cdot d_{\text{new}}}{W(\mathbf{v}) + w_{\text{new}}}$$

$$W(\mathbf{v}) \leftarrow \min(W(\mathbf{v}) + w_{\text{new}}, W_{\max})$$

Here $d_{\text{new}}$ is the signed distance computed from the new observation, $w_{\text{new}}$ is the observation weight, and $W(\mathbf{v})$ is the accumulated weight. The signed distance is truncated within the truncation distance $\delta$:

$$d_{\text{new}} = \text{clip}(D(\mathbf{u}) - \|\mathbf{v} - \mathbf{c}\|, -\delta, \delta)$$

$D(\mathbf{u})$ is the depth value at pixel $\mathbf{u}$, $\mathbf{c}$ is the camera position, and $\mathbf{v}$ is the voxel center.

```python
class TSDFVolume:
    """
    TSDF volume — 3D reconstruction from RGB-D frames.
    """
    
    def __init__(self, volume_bounds, voxel_size=0.02):
        """
        Args:
            volume_bounds: [[x_min, x_max], [y_min, y_max], [z_min, z_max]]
            voxel_size: voxel edge length (meters)
        """
        self.voxel_size = voxel_size
        self.bounds = np.array(volume_bounds)
        
        self.dims = np.ceil(
            (self.bounds[:, 1] - self.bounds[:, 0]) / voxel_size
        ).astype(int)
        
        # TSDF values and weights
        self.tsdf = np.ones(self.dims) * 1.0  # initial value: truncation value
        self.weight = np.zeros(self.dims, dtype=np.float32)
        self.color = np.zeros((*self.dims, 3), dtype=np.float32)
        
        # Truncation distance
        self.trunc_dist = 3.0 * voxel_size
    
    def integrate(self, depth_image, color_image, K, T_camera_to_world):
        """
        Integrate a single RGB-D frame into the TSDF.
        
        Args:
            depth_image: (H, W) depth (meters)
            color_image: (H, W, 3) RGB (0-255)
            K: camera intrinsic parameters (3, 3)
            T_camera_to_world: camera pose (4, 4)
        """
        T_world_to_camera = np.linalg.inv(T_camera_to_world)
        cam_pos = T_camera_to_world[:3, 3]
        
        H, W = depth_image.shape
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]
        
        # For each voxel
        for ix in range(self.dims[0]):
            for iy in range(self.dims[1]):
                for iz in range(self.dims[2]):
                    # World coordinates of the voxel center
                    vx = self.bounds[0, 0] + (ix + 0.5) * self.voxel_size
                    vy = self.bounds[1, 0] + (iy + 0.5) * self.voxel_size
                    vz = self.bounds[2, 0] + (iz + 0.5) * self.voxel_size
                    
                    # Transform to camera coordinates
                    p_world = np.array([vx, vy, vz, 1.0])
                    p_cam = T_world_to_camera @ p_world
                    
                    if p_cam[2] <= 0:
                        continue
                    
                    # Project to image coordinates
                    u = int(fx * p_cam[0] / p_cam[2] + cx)
                    v = int(fy * p_cam[1] / p_cam[2] + cy)
                    
                    if u < 0 or u >= W or v < 0 or v >= H:
                        continue
                    
                    depth = depth_image[v, u]
                    if depth <= 0:
                        continue
                    
                    # Compute signed distance
                    sdf = depth - p_cam[2]
                    
                    if sdf < -self.trunc_dist:
                        continue
                    
                    tsdf_val = min(sdf / self.trunc_dist, 1.0)
                    
                    # Weighted-average update
                    w_old = self.weight[ix, iy, iz]
                    w_new = 1.0
                    
                    self.tsdf[ix, iy, iz] = (
                        (w_old * self.tsdf[ix, iy, iz] + w_new * tsdf_val) 
                        / (w_old + w_new)
                    )
                    self.color[ix, iy, iz] = (
                        (w_old * self.color[ix, iy, iz] 
                         + w_new * color_image[v, u].astype(float))
                        / (w_old + w_new)
                    )
                    self.weight[ix, iy, iz] = min(w_old + w_new, 100.0)
    
    def extract_mesh(self):
        """
        Extract a mesh from the TSDF via Marching Cubes.
        Find zero crossings and generate triangle faces.
        """
        # Marching Cubes algorithm:
        # check the TSDF sign at the 8 corners of each voxel cube and
        # interpolate vertices on edges where the sign changes to generate triangles.
        
        vertices = []
        faces = []
        
        for ix in range(self.dims[0] - 1):
            for iy in range(self.dims[1] - 1):
                for iz in range(self.dims[2] - 1):
                    # TSDF values at the 8 corners
                    cube = np.array([
                        self.tsdf[ix, iy, iz],
                        self.tsdf[ix+1, iy, iz],
                        self.tsdf[ix+1, iy+1, iz],
                        self.tsdf[ix, iy+1, iz],
                        self.tsdf[ix, iy, iz+1],
                        self.tsdf[ix+1, iy, iz+1],
                        self.tsdf[ix+1, iy+1, iz+1],
                        self.tsdf[ix, iy+1, iz+1],
                    ])
                    
                    # Process only if there is a sign change
                    if np.all(cube > 0) or np.all(cube < 0):
                        continue
                    
                    # The actual Marching Cubes uses a 256-case
                    # look-up table to generate triangles.
                    # Here we only show the concept.
                    pass
        
        return vertices, faces
```

### 11.2.2 Voxblox

**[Voxblox](https://arxiv.org/abs/1611.03631)** (Oleynikova et al. 2017) is a TSDF-based real-time 3D reconstruction system that efficiently computes the **ESDF (Euclidean Signed Distance Field)** essential for path planning.

Core pipeline:

1. **TSDF integration**: integrate RGB-D or depth point clouds into the TSDF in a projective manner.
2. **Mesh extraction**: extract a mesh from the TSDF incrementally via Marching Cubes. Reprocess only changed voxel blocks.
3. **ESDF computation**: compute the ESDF from the TSDF. The ESDF stores the Euclidean distance from each voxel to the nearest obstacle.

Why ESDF matters: in path planning the robot must maintain a safety distance from obstacles, and having an ESDF allows querying the distance from an arbitrary point to the nearest obstacle in $O(1)$. It also provides gradient information, so the direction to avoid obstacles is immediately available.

### 11.2.3 Poisson Surface Reconstruction

Poisson reconstruction is a method that generates a watertight mesh from an oriented point cloud (position + normals).

**Core idea**: interpret the normal vectors as a gradient field, take the divergence of this gradient, and solve the Poisson equation:

$$\nabla^2 \chi = \nabla \cdot \mathbf{V}$$

Here $\mathbf{V}$ is the normal vector field, and $\chi$ is the indicator function (1 inside the surface, 0 outside). This PDE is solved efficiently on an octree, and an iso-surface is extracted.

Advantages: robust to noise, produces a watertight mesh, and works well even when the input points have non-uniform density. Disadvantages: suited to offline processing; for real-time SLAM, TSDF-based methods are preferred.

### 11.2.4 Real-Time Mesh Generation

Modern approaches to real-time mesh generation in SLAM systems:

1. **Voxblox incremental meshing**: re-run Marching Cubes only on voxel blocks whose TSDF has been updated. Real time is achievable because the entire volume is not reprocessed.
2. **FAST-LIVO2 mesh**: attach camera colors to LiDAR points to generate a colored mesh in real time. Post-process from the unified voxel map via Poisson or Ball Pivoting.

---

## 11.3 Neural / Learned Representations

Traditional map representations (voxel, mesh, surfel) are explicit — they store 3D structure directly. Neural representations, in contrast, encode the 3D scene into the weights of a neural network in an **implicit** or **parametric** manner.

### 11.3.1 NeRF-SLAM: Neural Implicit + Odometry

**[NeRF](https://arxiv.org/abs/2003.08934) (Neural Radiance Field)** (Mildenhall et al. 2020) represents a scene with an MLP that takes a 3D coordinate and viewing direction as input and outputs color and density:

$$F_\theta: (\mathbf{x}, \mathbf{d}) \to (c, \sigma)$$

Here $\mathbf{x} \in \mathbb{R}^3$ is a 3D position, $\mathbf{d} \in \mathbb{S}^2$ is the viewing direction, $c$ is RGB color, and $\sigma$ is volume density.

**Volume rendering equation**: integrate along the camera ray $\mathbf{r}(t) = \mathbf{o} + t\mathbf{d}$ to synthesize pixel color:

$$\hat{C}(\mathbf{r}) = \int_{t_n}^{t_f} T(t) \sigma(\mathbf{r}(t)) c(\mathbf{r}(t), \mathbf{d}) \, dt$$

Here $T(t) = \exp\left(-\int_{t_n}^{t} \sigma(\mathbf{r}(s)) ds\right)$ is the accumulated transmittance.

**NeRF-SLAM pipeline**: whereas the original NeRF performed offline reconstruction, NeRF-SLAM couples with SLAM to operate online:

1. **Tracking**: optimize the camera pose of the current frame against the existing neural map (photometric loss + depth loss).
2. **Mapping**: at the estimated pose, update the neural network weights. While learning newly observed regions, the representation of previously observed regions must remain consistent.

**[iMAP](https://arxiv.org/abs/2103.12352)** (Sucar et al. 2021): the first neural implicit SLAM. It represents the scene with a single MLP. For real-time performance, it uses keyframe-based learning and an active sampling strategy.

Limitations:
- **Training speed**: MLP training is slow, constraining real-time mapping.
- **Forgetting**: learning new regions degrades the representation of previously seen regions (catastrophic forgetting).
- **Scalability**: the capacity limit of a single MLP in large-scale environments.

Directions of improvement:
- **Instant-NGP**: hash-grid-based feature encoding accelerates training by tens of times. NeRF-SLAM variants leveraging this have emerged.
- **Multi-resolution hash grid**: partition space into hash tables at multiple resolutions, storing and querying local features efficiently.

**[NICE-SLAM](https://arxiv.org/abs/2112.12130)** (Zhu et al. 2022): a dense SLAM system that addresses iMAP's scalability problem by introducing a hierarchical feature grid and a pre-trained geometry decoder. It operates stably even in large-scale indoor environments and was presented at CVPR 2022.

### 11.3.2 3DGS-SLAM: 3D Gaussian Splatting-Based SLAM

**[3D Gaussian Splatting (3DGS)](https://arxiv.org/abs/2308.04079)** (Kerbl et al. 2023) is a representation that emerged as an alternative to NeRF, representing a scene with millions of 3D Gaussians. Each Gaussian consists of:

- Position $\boldsymbol{\mu} \in \mathbb{R}^3$
- Covariance $\boldsymbol{\Sigma} \in \mathbb{R}^{3 \times 3}$ (parameterized by rotation + scale)
- Opacity $\alpha \in [0, 1]$
- Color (spherical harmonics coefficients)

Rendering is performed by splatting — projecting 3D Gaussians onto the image plane and alpha-blending them in depth order — which is tens of times faster than NeRF's ray marching.

**[3DGS-SLAM](https://arxiv.org/abs/2312.06741)** (Matsuki et al. 2024): uses 3DGS as the SLAM representation:

1. **Tracking**: optimize the camera pose using photometric + geometric loss between the predicted image rendered from the Gaussian map and the actual image.
2. **Mapping**: according to new observations, add (densification), split (splitting), and remove (pruning) Gaussians.
3. **Loop closure**: when correcting poses, the positions of the Gaussians must be deformed together.

**NeRF-SLAM vs. 3DGS-SLAM**:

| Property | NeRF-SLAM | 3DGS-SLAM |
|------|-----------|-----------|
| Representation | MLP weights | Explicit 3D Gaussian set |
| Rendering speed | Slow (ray marching) | Fast (rasterization) |
| Training speed | Slow | Fast |
| Editability | Hard | Easy (manipulate individual Gaussians) |
| Memory | Fixed (model size) | Variable (proportional to number of Gaussians) |
| Loop closure handling | Hard (weight deformation) | Relatively easy (Gaussian transformation) |

**Current limitations**: neither method yet surpasses traditional map representations (TSDF, surfel) in accuracy and real-time performance across all situations. Especially in large-scale environments, dynamic scenes, and long-term SLAM, traditional methods remain more stable. However, in rendering quality neural representations are overwhelmingly superior, and this gap is closing rapidly.

**Recent major advances (2024-2025)**:

- **[SplaTAM](https://arxiv.org/abs/2312.02126)** (Keetha et al. CVPR 2024): tracks and maps 3D Gaussians in real time from an RGB-D camera, and with silhouette-mask-based structured map expansion achieves more than 2x improvement over prior methods in camera pose estimation and novel-view synthesis.
- **[MonoGS](https://arxiv.org/abs/2312.06741)** (Matsuki et al. CVPR 2024 Highlight): the first monocular 3DGS SLAM, performing accurate tracking, mapping, and high-quality rendering in an integrated pipeline at 3fps with only a monocular camera. It resolves the ambiguity of monocular 3D reconstruction with geometric verification and regularization.
- **[MASt3R-SLAM](https://arxiv.org/abs/2412.12392)** (Murai et al. CVPR 2025): a system that integrates a 3D reconstruction foundation model (MASt3R) into SLAM, recovering globally-consistent dense geometry at 15fps without camera-model assumptions.

```python
import numpy as np

class Gaussian3D:
    """Single 3D Gaussian representation."""
    
    def __init__(self, mean, covariance, color, opacity):
        """
        Args:
            mean: 3D position (3,)
            covariance: 3x3 covariance matrix (3, 3)
            color: RGB color (3,)
            opacity: opacity (scalar)
        """
        self.mean = np.array(mean, dtype=np.float64)
        self.covariance = np.array(covariance, dtype=np.float64)
        self.color = np.array(color, dtype=np.float64)
        self.opacity = float(opacity)
    
    def project_to_2d(self, T_world_to_cam, K):
        """
        Project a 3D Gaussian onto the image plane.
        
        Returns:
            mean_2d: projected center (2,)
            cov_2d: projected 2D covariance (2, 2)
        """
        # World -> camera coordinate transform
        R = T_world_to_cam[:3, :3]
        t = T_world_to_cam[:3, 3]
        
        mean_cam = R @ self.mean + t
        
        if mean_cam[2] <= 0:
            return None, None
        
        # 3D covariance -> camera frame
        cov_cam = R @ self.covariance @ R.T
        
        # Projection Jacobian (pinhole model)
        fx, fy = K[0, 0], K[1, 1]
        z = mean_cam[2]
        
        J = np.array([
            [fx / z, 0, -fx * mean_cam[0] / z**2],
            [0, fy / z, -fy * mean_cam[1] / z**2]
        ])
        
        # 2D covariance (EWA splatting)
        cov_2d = J @ cov_cam @ J.T
        
        # 2D center
        mean_2d = np.array([
            fx * mean_cam[0] / z + K[0, 2],
            fy * mean_cam[1] / z + K[1, 2]
        ])
        
        return mean_2d, cov_2d


def render_gaussians(gaussians, T_world_to_cam, K, image_size):
    """
    3D Gaussian Splatting rendering (simplified version).
    
    Args:
        gaussians: list of Gaussian3D
        T_world_to_cam: camera pose (4, 4)
        K: camera intrinsic parameters (3, 3)
        image_size: (H, W)
        
    Returns:
        rendered_image: (H, W, 3)
    """
    H, W = image_size
    image = np.zeros((H, W, 3), dtype=np.float64)
    accumulated_alpha = np.zeros((H, W), dtype=np.float64)
    
    # Sort by depth
    R = T_world_to_cam[:3, :3]
    t = T_world_to_cam[:3, 3]
    
    depths = []
    for g in gaussians:
        mean_cam = R @ g.mean + t
        depths.append(mean_cam[2])
    
    sorted_indices = np.argsort(depths)
    
    for idx in sorted_indices:
        g = gaussians[idx]
        mean_2d, cov_2d = g.project_to_2d(T_world_to_cam, K)
        
        if mean_2d is None:
            continue
        
        # Influence range of the 2D Gaussian (3 sigma)
        eigenvalues = np.linalg.eigvalsh(cov_2d)
        radius = int(3 * np.sqrt(max(eigenvalues))) + 1
        
        u_min = max(0, int(mean_2d[0]) - radius)
        u_max = min(W, int(mean_2d[0]) + radius + 1)
        v_min = max(0, int(mean_2d[1]) - radius)
        v_max = min(H, int(mean_2d[1]) + radius + 1)
        
        cov_2d_inv = np.linalg.inv(cov_2d + 1e-6 * np.eye(2))
        
        for v in range(v_min, v_max):
            for u in range(u_min, u_max):
                d = np.array([u - mean_2d[0], v - mean_2d[1]])
                power = -0.5 * d @ cov_2d_inv @ d
                
                if power < -4.0:  # skip pixels that are too far
                    continue
                
                alpha = g.opacity * np.exp(power)
                
                # Front-to-back alpha blending
                remaining = 1.0 - accumulated_alpha[v, u]
                if remaining < 0.01:
                    continue
                
                weight = alpha * remaining
                image[v, u] += weight * g.color
                accumulated_alpha[v, u] += weight
    
    return np.clip(image, 0, 255).astype(np.uint8)
```

---

## 11.4 Semantic Maps

A geometric map answers only the "where" of "where is what." A semantic map also answers the "what" — whether this is a wall, a door, a chair, or a person. For the robot to achieve human-level environmental understanding, semantic information is essential.

### 11.4.1 Object-Level Maps

The most intuitive semantic map recognizes the objects in an environment and registers their positions and sizes in the map.

Pipeline:
1. **2D detection/segmentation**: detect objects in camera images (YOLO, Mask R-CNN, SAM, etc.).
2. **3D lifting**: back-project 2D detections into 3D space using depth information and the camera pose.
3. **Data association**: link the same object observed across multiple frames (tracking + re-identification).
4. **Map integration**: register the 3D bounding box or shape model of the object into the map.

```python
class ObjectMap:
    """3D object map — integrate 2D detections into 3D space."""
    
    def __init__(self, association_threshold=0.5):
        self.objects = []  # [{class, center_3d, bbox_3d, observations, id}]
        self.next_id = 0
        self.assoc_thresh = association_threshold
    
    def integrate_detection(self, class_label, mask_2d, depth_image, 
                            K, T_cam_to_world):
        """
        Integrate a 2D detection into the 3D map.
        
        Args:
            class_label: object class (str)
            mask_2d: 2D segmentation mask (H, W) bool
            depth_image: depth map (H, W)
            K: camera intrinsic parameters (3, 3)
            T_cam_to_world: camera -> world transform (4, 4)
        """
        # 1. Recover 3D points of the masked region
        points_3d = self._backproject(mask_2d, depth_image, K, T_cam_to_world)
        
        if len(points_3d) < 10:
            return
        
        center = np.mean(points_3d, axis=0)
        bbox_min = np.min(points_3d, axis=0)
        bbox_max = np.max(points_3d, axis=0)
        
        # 2. Data association: match against existing objects
        best_match = None
        best_dist = float('inf')
        
        for obj in self.objects:
            if obj['class'] != class_label:
                continue
            dist = np.linalg.norm(obj['center_3d'] - center)
            if dist < best_dist and dist < self.assoc_thresh:
                best_dist = dist
                best_match = obj
        
        if best_match is not None:
            # Update existing object (weighted average)
            n = best_match['observations']
            best_match['center_3d'] = (
                (n * best_match['center_3d'] + center) / (n + 1)
            )
            best_match['observations'] = n + 1
        else:
            # Register a new object
            self.objects.append({
                'class': class_label,
                'center_3d': center,
                'bbox_3d': (bbox_min, bbox_max),
                'observations': 1,
                'id': self.next_id
            })
            self.next_id += 1
    
    def _backproject(self, mask, depth, K, T):
        """Back-project 2D pixels of the masked region to 3D world coordinates."""
        ys, xs = np.where(mask & (depth > 0))
        
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]
        
        z = depth[ys, xs]
        x = (xs - cx) * z / fx
        y = (ys - cy) * z / fy
        
        points_cam = np.stack([x, y, z, np.ones_like(z)], axis=1)  # (N, 4)
        points_world = (T @ points_cam.T).T[:, :3]  # (N, 3)
        
        return points_world
```

### 11.4.2 3D Scene Graph: Hydra & S-Graphs

An object-level map only recognizes individual objects. But humans understand environments as hierarchical relations: "this room has a table, on top of which is a cup, and this room is a living room." A **3D Scene Graph** is such a hierarchical, relational environmental representation.

**[Hydra](https://arxiv.org/abs/2201.13360)** (Hughes et al. 2022) is the first system to build a 3D scene graph incrementally in real time. It consists of five layers:

| Layer | Contents | Construction method |
|-------|------|-----------|
| 1 | Metric-Semantic 3D Mesh | TSDF integration + semantic segmentation |
| 2 | Objects & Agents | 3D bounding box detection |
| 3 | Places (topological) | Extracted from GVD (Generalized Voronoi Diagram) |
| 4 | Rooms | Community detection on the place graph |
| 5 | Buildings | Higher-level grouping of rooms |

**Construction of the Places layer**: this layer is the most original part of Hydra.

1. Compute the ESDF from the TSDF.
2. Incrementally extract the GVD from the ESDF. Vertices of the GVD are points maximally far from obstacles — i.e., good points for the robot to pass through.
3. Configure GVD vertices as place nodes and GVD edges as connections between places.
4. This place graph is a topological map that can be used directly for path planning.

**Room detection**: a method for detecting rooms from the place graph:

1. Weight the edges between place nodes according to their proximity to obstacles.
2. At narrow passages such as doorways, weights become high (indicating difficulty of passage).
3. Group places into rooms using a community detection algorithm (e.g., dilation-based).

**Hierarchical loop closure**: Hydra leverages the hierarchical structure of the scene graph to improve the quality of loop closure:

- **Top-down**: first find candidates at higher layers (room, place).
- **Bottom-up**: perform geometric verification at lower layers (visual feature, object) (TEASER++ based).
- This hierarchical approach detects more and more accurate loop closures than a simple BoW approach.

**S-Graphs** (Situational Graphs): a hierarchical scene graph similar to Hydra, but directly incorporating hierarchical information into factor graph optimization. Structural elements such as rooms, walls, and floors are added as variables of the factor graph to improve SLAM accuracy.

### 11.4.3 Open-Vocabulary Semantic Mapping

Traditional semantic mapping operates only over a predefined class set (e.g., COCO's 80 classes). **Open-vocabulary semantic mapping** enables searching the map with arbitrary text queries.

Core techniques:

1. **CLIP/DINO feature embedding**: extract features from a vision-language model for each observation (image or image patch).
2. **Feature -> 3D map**: store the extracted features at the corresponding 3D location (attached to a point cloud or voxel).
3. **Text query**: when the user queries, say, "red fire extinguisher," encode the text with the CLIP text encoder, compute similarity against the map's visual features, and return the corresponding location.

Impact of this approach: the robot can recognize and locate objects it was not trained on in advance. It works even for "things never seen before." This is essential in unpredictable environments such as household robots and exploration robots.

```python
class OpenVocabSemanticMap:
    """
    Core idea of open-vocabulary semantic mapping.
    Attach CLIP features to each 3D point.
    """
    
    def __init__(self, feature_dim=512):
        self.points_3d = []     # (N, 3)
        self.features = []      # (N, D) CLIP visual features
        self.feature_dim = feature_dim
    
    def add_observation(self, points_3d, clip_features):
        """
        Add 3D points and corresponding CLIP features to the map.
        
        Args:
            points_3d: (M, 3) 3D point coordinates
            clip_features: (M, D) CLIP visual feature of each point
        """
        self.points_3d.extend(points_3d)
        self.features.extend(clip_features)
    
    def query(self, text_feature, top_k=10):
        """
        Search the map for relevant locations given a text query.
        
        Args:
            text_feature: (D,) CLIP text feature
            top_k: number of top results to return
            
        Returns:
            results: list of (point_3d, similarity)
        """
        if len(self.features) == 0:
            return []
        
        features_array = np.array(self.features)
        points_array = np.array(self.points_3d)
        
        # Cosine similarity
        text_norm = text_feature / (np.linalg.norm(text_feature) + 1e-8)
        feat_norms = features_array / (
            np.linalg.norm(features_array, axis=1, keepdims=True) + 1e-8
        )
        
        similarities = feat_norms @ text_norm
        
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        return [
            (points_array[idx], similarities[idx]) 
            for idx in top_indices
        ]
    
    def visualize_heatmap(self, text_feature):
        """Generate a relevance heatmap for a text query."""
        if len(self.features) == 0:
            return np.array([]), np.array([])
        
        features_array = np.array(self.features)
        points_array = np.array(self.points_3d)
        
        text_norm = text_feature / (np.linalg.norm(text_feature) + 1e-8)
        feat_norms = features_array / (
            np.linalg.norm(features_array, axis=1, keepdims=True) + 1e-8
        )
        
        similarities = feat_norms @ text_norm
        
        # Normalize to 0-1
        sim_min, sim_max = similarities.min(), similarities.max()
        if sim_max > sim_min:
            heatmap = (similarities - sim_min) / (sim_max - sim_min)
        else:
            heatmap = np.zeros_like(similarities)
        
        return points_array, heatmap
```

---

## 11.5 Long-Term & Dynamic Maps

Real environments are not static. Vehicles move, furniture is rearranged, and buildings are newly constructed. Robots operating over long periods must adapt to such changes.

### 11.5.1 Change Detection

Identify changes by comparing the map with current observations.

**Geometric change detection**: compare expected observations from the past map with current sensor observations.

1. **Free-to-occupied**: if a new obstacle is detected in a region marked free on the map -> a new object has been added.
2. **Occupied-to-free**: if a region marked occupied on the map is passed through in the current observation -> an existing object has been removed.

```python
def detect_changes(occupancy_map, current_scan, robot_pose, threshold=0.3):
    """
    Detect changes by comparing the existing map with the current scan.
    
    Returns:
        new_objects: locations of newly appeared obstacles
        removed_objects: locations of vanished obstacles
    """
    new_objects = []
    removed_objects = []
    
    for point in current_scan:
        # Map coordinate of the current scan point
        map_value = occupancy_map.get_probability(point)
        
        if map_value < threshold:
            # Map says free but an obstacle is currently detected -> new object
            new_objects.append(point)
    
    # Detect regions that were occupied on the map but are currently passed through
    for ray_endpoint in current_scan:
        ray_cells = trace_ray(robot_pose, ray_endpoint)
        for cell in ray_cells[:-1]:  # exclude endpoint (may still be occupied)
            map_value = occupancy_map.get_probability(cell)
            if map_value > (1.0 - threshold):
                # Map says occupied but current ray passes through -> object removed
                removed_objects.append(cell)
    
    return new_objects, removed_objects
```

**Semantic change detection**: detect not only geometric changes but also changes in an object's class or state. For example, "chair moved" or "door opened/closed."

### 11.5.2 Map Maintenance: Retention Strategies

In long-term operation, we cannot store all observations of the map indefinitely. Which information do we keep and which do we discard?

**Strategy 1: Recency weighting**: assign higher weights to recent observations and gradually decay the influence of older ones. A scheme that decays the TSDF weight over time.

**Strategy 2: Semi-static classification**: classify each element of the environment as static, semi-static, or dynamic:
- **Static**: walls, floors, buildings -> permanently preserved.
- **Semi-static**: furniture, parked vehicles -> periodically updated.
- **Dynamic**: pedestrians, moving vehicles -> removed from the map.

**Strategy 3: Multi-experience mapping**: store multiple "experiences" of the same place. Maintain several versions of the map under different lighting, season, and furniture arrangement, and select the experience that best matches the current observation.

### 11.5.3 Handling Dynamic Objects

In SLAM, dynamic objects (moving people, vehicles) cause two problems:

1. **Tracking errors**: if feature points of dynamic objects are mistakenly used as those of the static environment, pose estimation breaks.
2. **Map contamination**: if dynamic objects are recorded in the map, they become "ghost obstacles."

**Countermeasures**:

1. **Semantic filtering**: identify dynamic object classes (person, vehicle) via semantic classification and exclude those observations from the SLAM pipeline.

2. **Geometric consistency check**: classify as dynamic the observations that are inconsistent across multiple frames (points visible in one frame but disappearing in the next).

3. **Background subtraction**: classify as dynamic the voxels in the TSDF that show a free-to-occupied-to-free pattern.

4. **SLAM with dynamic object tracking**: instead of simply ignoring dynamic objects, track them with separate state variables. This allows estimating the trajectories of dynamic objects as well (SLAM + MOT).

```python
class DynamicMapManager:
    """
    Dynamic environment map manager.
    Classifies observations as static/semi-static/dynamic for differential handling.
    """
    
    def __init__(self):
        self.static_map = {}       # permanent storage
        self.semistatic_map = {}   # periodic update
        self.dynamic_objects = []  # dynamic objects being tracked
        
        # Dynamic class definitions
        self.dynamic_classes = {'person', 'car', 'bicycle', 'dog'}
        self.semistatic_classes = {'chair', 'box', 'parked_car'}
    
    def process_observation(self, point_3d, class_label, timestamp):
        """
        Classify a new observation and integrate it into the appropriate map.
        """
        if class_label in self.dynamic_classes:
            self._track_dynamic(point_3d, class_label, timestamp)
        elif class_label in self.semistatic_classes:
            self._update_semistatic(point_3d, class_label, timestamp)
        else:
            self._update_static(point_3d, class_label)
    
    def _track_dynamic(self, point, cls, timestamp):
        """Track a dynamic object. Do not add to the map."""
        # Match against existing dynamic objects via data association
        matched = False
        for obj in self.dynamic_objects:
            if (obj['class'] == cls and 
                np.linalg.norm(obj['last_position'] - point) < 2.0):
                obj['trajectory'].append((timestamp, point.copy()))
                obj['last_position'] = point.copy()
                matched = True
                break
        
        if not matched:
            self.dynamic_objects.append({
                'class': cls,
                'last_position': point.copy(),
                'trajectory': [(timestamp, point.copy())]
            })
    
    def _update_semistatic(self, point, cls, timestamp):
        """Update a semi-static object."""
        key = self._spatial_key(point)
        self.semistatic_map[key] = {
            'position': point.copy(),
            'class': cls,
            'last_seen': timestamp
        }
    
    def _update_static(self, point, cls):
        """Update the static map."""
        key = self._spatial_key(point)
        self.static_map[key] = {
            'position': point.copy(),
            'class': cls
        }
    
    def cleanup_old_semistatic(self, current_time, max_age=3600):
        """Remove old semi-static observations (e.g., 1 hour)."""
        keys_to_remove = [
            k for k, v in self.semistatic_map.items()
            if current_time - v['last_seen'] > max_age
        ]
        for k in keys_to_remove:
            del self.semistatic_map[k]
    
    def _spatial_key(self, point, resolution=0.1):
        """Convert a 3D point to a discrete grid key."""
        return (
            int(point[0] / resolution),
            int(point[1] / resolution),
            int(point[2] / resolution)
        )
```

---

In this chapter we surveyed the full spectrum of spatial representations, from point-based representations (OctoMap) through continuous surfaces (TSDF/Mesh), neural representations (NeRF/3DGS), and semantic hierarchies (Scene Graph). Each representation is suited to different downstream tasks, and modern systems trend toward combining them hierarchically. In the next chapter we examine how the algorithms and representations covered so far are integrated in **real platforms** (autonomous driving, drones, handheld) and how performance is evaluated.
