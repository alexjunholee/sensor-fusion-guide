# Chapter 11. Spatial Representations

Once loop closure and global optimization (→ Chapter 10) yield a consistent trajectory, the robot still has to record the world it observed in an enduring **map**. A map is the spatial memory through which a system reasons about its surroundings, and no single data structure serves every purpose equally well: path planners need explicit free and occupied space, view synthesis requires appearance and texture, and high-level task execution demands semantic categories and relational structure.

In this chapter, we move upward along that hierarchy of abstraction—from quantitative metric maps (occupancy grids, adaptive voxel trees, and surfels), to continuous surface meshes and distance fields (TSDF, ESDF, and Poisson reconstruction), to neural and Gaussian representations (NeRF-SLAM and 3DGS-SLAM), and finally to semantic 3D scene graphs and long-term maintenance in dynamic environments.

---

## 11.1 Metric Maps

Metric maps encode the geometry of an environment in quantitative coordinates, linking directly to localization and motion planning. Which specific metric structure fits best depends on the sensing modality, workspace scale, and query pattern.

### 11.1.1 Occupancy Grid (2D/3D)

An occupancy grid discretizes space into a regular lattice and tracks the probability that each cell is occupied, free, or unknown.

**2D occupancy grids** are standard for planar indoor navigation on a single floor. For each cell $m_i$, we update the occupancy posterior $p(m_i \mid z_{1:t})$ recursively as new range scans arrive. Rather than multiplying raw probabilities—which invites floating-point underflow and awkward normalization—we convert probabilities to **log-odds**, turning Bayesian filtering into simple, numerically stable addition:

$$\text{log-odds}(m_i \mid z_{1:t}) = \text{log-odds}(m_i \mid z_{1:t-1}) + \text{log-odds}(m_i \mid z_t) - \text{log-odds}(m_i)$$

where the log-odds mapping is defined as:

$$l(m_i) = \log\frac{p(m_i)}{1 - p(m_i)}$$

While uniform grids work cleanly in 2D, extending them naively to 3D runs into a severe memory bottleneck: covering an $L \times W \times H$ volume at resolution $r$ demands $(L/r)(W/r)(H/r)$ cells. At 5 cm resolution, even a modest 100 m $\times$ 100 m $\times$ 10 m space contains $2000 \times 2000 \times 200 = 8 \times 10^8$ voxels—roughly 3.2 GB just to store one float per cell, before accounting for weights, semantics, or container overhead.

The class below demonstrates log-odds Bayesian updates on a 2D grid using Bresenham ray tracing to mark free cells along each beam and occupied cells at its endpoint:

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
        self.l_free = -0.40  # log-odds(occupied | miss) — update for a pass-through observation
        
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

Because most of any 3D environment is empty air or unobserved interior, allocating a uniform dense grid wastes memory on regions that carry no geometric detail. Sparse and hierarchical structures solve this by allocating fine cells only where observations require them.

**[OctoMap](https://doi.org/10.1007/s10514-012-9321-0)** (Hornung et al. 2013) stores 3D probabilistic occupancy inside an octree. Space is subdivided recursively into eight octants, and whenever all eight children converge to the same state—entirely occupied or entirely free—they collapse back into their parent node (**pruning**), keeping large uniform regions coarse while preserving fine resolution along object boundaries.

In practice, OctoMap trades speed for probabilistic generality:
- **Probabilistic updates**: It updates leaf occupancy using the same additive log-odds formulation as a 2D occupancy grid.
- **Multi-resolution queries**: Queries can traverse down to fine leaf nodes (e.g., 2 cm) or stop early at coarser ancestors (several meters) depending on planning needs.
- **Memory savings**: In environments dominated by free or unknown volume, pruning dramatically reduces memory footprint relative to a dense grid, though the exact ratio depends on scene clutter and tree pointer overhead.
- **Limitations**: Pointer chasing adds traversal overhead, occupancy lookups do not replace fast k-nearest-neighbor (kNN) or surface-registration indices, and clearing dynamic objects still requires inverse sensor ray updates and a temporal policy.

The simplified node class below captures OctoMap's core 8-way recursive subdivision and leaf-pruning logic:

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

Two alternative sparse structures address very different bottlenecks:

- **OpenVDB**: Originating in film VFX, OpenVDB replaces deep binary/octary pointer trees with a shallow, fixed-depth wide hierarchy paired with tile and value compression that omits uniform background space. Because its tree layout and API differ fundamentally from OctoMap, traversal speed depends heavily on the workload, and probabilistic occupancy updates must be layered on at the application level.
- **ikd-tree** (FAST-LIO2): Designed specifically for real-time LiDAR odometry, the incremental k-d tree inserts individual points in amortized $O(\log n)$ time while maintaining fast spatial neighbor queries. Range-wise deletions are applied lazily and subtrees are rebalanced only when an imbalance criterion triggers, separating their cost from per-point $O(\log n)$ insertion. Rather than maintaining volumetric occupancy, ikd-tree supports three fast geometric primitives:
  - **Insertion**: Add new LiDAR points incrementally, triggering partial subtree rebalancing only past an imbalance threshold.
  - **Deletion**: Drop stale map points outside the active local window via lazy range deletion.
  - **kNN search**: Retrieve nearest map neighbors for incoming scan points to drive real-time point-to-plane registration.

The table below summarizes how OctoMap and ikd-tree divide along planning versus registration workloads:

| Property | OctoMap | ikd-tree |
|------|---------|----------|
| Data structure | Octree | k-d tree |
| Probabilistic update | Built-in (log-odds) | None (only point storage) |
| kNN search | Slow | Fast |
| Dynamic insertion/deletion | Possible but slow | Point insertion is amortized $O(\log n)$; range deletion and rebalancing carry separate costs |
| Use case | Path planning, exploration | LiDAR odometry map maintenance |

### 11.1.3 Surfel Maps

Raw point clouds lack local surface orientation, while full triangle meshes are awkward to deform or update incrementally. A **surfel** (surface element) splits the difference: it models each local patch of a surface as an oriented disk $(\mathbf{p}, \mathbf{n}, r, c, w)$ carrying position $\mathbf{p} \in \mathbb{R}^3$, unit normal $\mathbf{n} \in \mathbb{R}^3$, radius $r \in \mathbb{R}^+$, color $c$, and accumulated fusion confidence $w$.

**[ElasticFusion](https://doi.org/10.15607/RSS.2015.XI.001)** (Whelan et al. 2015) uses surfels as the backbone of a real-time dense RGB-D SLAM pipeline built around three operations:

1. **Frame-to-model tracking**: Align each incoming depth/RGB frame against a rendered prediction of the current surfel map.
2. **Non-rigid map deformation**: When a loop closure fires, warp the entire surfel collection coherently using an embedded deformation graph rather than rebuilding a voxel grid.
3. **Surfel fusion**: Merge newly observed surface samples into matching existing surfels via confidence-weighted averaging.

Because surfels require no explicit triangle connectivity, they represent continuous surfaces cleanly while remaining lightweight to fuse and warp. The minimal implementation below illustrates confidence-weighted position, normal, radius, and color fusion across overlapping observations:

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

While occupancy voxels and surfels serve basic navigation and tracking, applications such as physical collision simulation, architectural reconstruction, and manipulation require smooth, connected surfaces. Triangle meshes meet that need—and in online SLAM, they are almost always extracted from an implicit signed-distance field rather than stitched directly from noisy points.

### 11.2.1 TSDF (Truncated Signed Distance Function)

Updating a noisy triangle mesh frame by frame is notoriously brittle, whereas averaging scalar distance values across overlapping depth measurements is trivial. A **TSDF** exploits this insight by storing at every voxel the signed distance to the nearest observed surface along the sensor's line of sight: positive values sit in front of the surface (free space), negative values lie just behind it within a narrow truncation band, and the surface itself emerges cleanly as the zero-crossing iso-surface.

Note an important distinction from the omnidirectional ESDF introduced in §11.2.2: what online RGB-D fusion maintains is a *projective* TSDF measured along camera rays, truncated to $[-\delta, +\delta]$. In front of a surface, distances are clamped to $+\delta$ and averaged into free space; beyond $-\delta$ behind the surface, voxels are skipped during updates and retain their initial default value. Consequently, a negative TSDF value does not carry the same volumetric occupancy commitment as an occupied voxel in §11.1.

As each new depth frame arrives, voxels along the viewing rays fuse via a running weighted average:

$$\text{TSDF}(\mathbf{v}) \leftarrow \frac{W(\mathbf{v}) \cdot \text{TSDF}(\mathbf{v}) + w_{\text{new}} \cdot d_{\text{new}}}{W(\mathbf{v}) + w_{\text{new}}}$$

$$W(\mathbf{v}) \leftarrow \min(W(\mathbf{v}) + w_{\text{new}}, W_{\max})$$

where $W(\mathbf{v})$ is the accumulated voxel weight, $w_{\text{new}}$ is the weight of the incoming observation, and $d_{\text{new}}$ is the projective signed distance clipped to the truncation band $[-\delta, \delta]$:

$$d_{\text{new}} = \text{clip}\left(D(\mathbf{u}) - \lambda^{-1}\|\mathbf{v} - \mathbf{c}\|, -\delta, \delta\right), \quad \lambda = \|\mathbf{K}^{-1}\dot{\mathbf{u}}\|$$

Here $D(\mathbf{u})$ is the measured depth at pixel $\mathbf{u}$, $\mathbf{c}$ is the camera center, and $\mathbf{v}$ is the voxel center. Because $D(\mathbf{u})$ measures depth along the optical axis rather than Euclidean ray distance, dividing the ray length $\|\mathbf{v} - \mathbf{c}\|$ by $\lambda$ projects it onto that same optical-axis basis prior to subtraction.

The `TSDFVolume` class below implements projective RGB-D fusion into a dense voxel grid alongside the zero-crossing check used by Marching Cubes mesh extraction:

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

While a projective TSDF excels at incremental surface reconstruction, it only measures distance along sensor rays and truncates sharply at $\pm \delta$. Gradient-based and optimization-based trajectory planners need something richer: an **ESDF (Euclidean Signed Distance Field)** that reports the true omnidirectional distance to the nearest obstacle at any point in $O(1)$ time, together with smooth spatial gradients pointing directly away from collisions. (Note that sampling-based and search-based planners work directly on occupancy maps, so an ESDF is specifically needed for continuous optimization planners rather than planning in general.)

**[Voxblox](https://arxiv.org/abs/1611.03631)** (Oleynikova et al. 2017) bridges incremental mapping and trajectory optimization in three stages:

1. **TSDF integration**: Fuse incoming RGB-D or LiDAR depth scans projectively into hashed voxel blocks.
2. **Incremental meshing**: Run Marching Cubes exclusively on voxel blocks whose TSDF values changed, keeping mesh updates real time.
3. **ESDF propagation**: Propagate true Euclidean distances outward from the TSDF surface band via wavefront expansion, yielding omnidirectional obstacle clearance and avoidance gradients everywhere in the workspace.

### 11.2.3 Poisson Surface Reconstruction

When the goal is a clean, watertight offline mesh from an already-collected oriented point cloud (positions plus surface normals), **Poisson surface reconstruction** casts meshing as a global partial differential equation. Interpreting the oriented normals as samples of the gradient of an interior indicator function $\chi$ (equal to $1$ inside the object and $0$ outside), taking the divergence yields a Poisson equation:

$$\nabla^2 \chi = \nabla \cdot \mathbf{V}$$

where $\mathbf{V}$ is the smoothed normal vector field. Solving this PDE over an adaptive octree and extracting an iso-surface produces smooth, closed meshes. Because it solves a global implicit system, however, Poisson reconstruction depends heavily on accurate normal orientation, octree depth, and screening parameters, and can inadvertently seal doorways or openings that ought to stay open. Its computational cost suits offline post-processing, whereas TSDF meshing remains the natural choice for incremental online depth fusion.

### 11.2.4 Real-Time Mesh Generation

In practice, SLAM pipelines either extract meshes directly from an incremental TSDF or hand an exported point/surfel map to a downstream mesher:

1. **Voxblox incremental meshing**: Re-evaluating Marching Cubes only over recently modified voxel blocks keeps meshing overhead bounded and sustains real-time updates without reprocessing the entire map.
2. **FAST-LIVO2 colored point map + post-processing**: FAST-LIVO2 fuses camera intensities during visual updates and can export a dense colored point map, but its online output is a point map rather than a triangle mesh. When a surface mesh is required, verify normal consistency and point density on the exported point cloud before running a separate Poisson or Ball-Pivoting reconstruction pass.

---

## 11.3 Neural / Learned Representations

Explicit geometric maps—voxels, surfels, and meshes—store 3D structure directly, which makes geometric queries straightforward but often struggles to synthesize photorealistic views or compress fine appearance smoothly. Neural and learned representations span a distinct spectrum: at one extreme sit purely **implicit** coordinate networks that pack an entire scene into MLP weights; in the middle sit hybrid **feature grids** (multi-resolution hash or hierarchical grids) pairing explicit spatial storage with tiny neural decoders; and at the other extreme sit explicit differentiable primitives such as **3D Gaussians**.

### 11.3.1 NeRF-SLAM: Neural Implicit + Odometry

**[NeRF](https://arxiv.org/abs/2003.08934) (Neural Radiance Field)** (Mildenhall et al. 2020) models a continuous 3D scene as an MLP mapping spatial location $\mathbf{x} \in \mathbb{R}^3$ and viewing direction $\mathbf{d} \in \mathbb{S}^2$ to view-dependent RGB color $c$ and differential volume density $\sigma$:

$$F_\theta: (\mathbf{x}, \mathbf{d}) \to (c, \sigma)$$

Pixel colors emerge by integrating radiance along each camera ray $\mathbf{r}(t) = \mathbf{o} + t\mathbf{d}$ via classical volume rendering:

$$\hat{C}(\mathbf{r}) = \int_{t_n}^{t_f} T(t) \sigma(\mathbf{r}(t)) c(\mathbf{r}(t), \mathbf{d}) \, dt$$

where $T(t) = \exp\left(-\int_{t_n}^{t} \sigma(\mathbf{r}(s)) ds\right)$ tracks accumulated transmittance—the probability that the ray travels from $t_n$ to $t$ without hitting an opaque surface.

While the original NeRF optimized offline from known poses, **NeRF-SLAM** embeds this differentiable renderer inside an online tracking-and-mapping loop:

1. **Tracking**: Optimize the live camera pose by backpropagating photometric and depth rendering residuals through the frozen neural field.
2. **Mapping**: Update the network weights at the tracked poses using selected keyframes, balancing new scene coverage against replay of earlier keyframes so previously learned rooms do not degrade.

**[iMAP](https://arxiv.org/abs/2103.12352)** (Sucar et al. 2021) demonstrated this concept in real-time handheld RGB-D SLAM using a single MLP as the sole scene representation, reporting 10 Hz tracking and 2 Hz global-map updates via keyframe replay and information-guided pixel sampling. Yet relying on a single monolithic MLP exposes three structural bottlenecks: slow per-iteration optimization, **catastrophic forgetting** when new observations overwrite shared weights, and limited representational capacity in large scenes.

To break those bottlenecks, subsequent neural SLAM systems store local geometry in explicit spatial feature grids decoded by tiny MLPs:
- **Instant-NGP multi-resolution hash grids** partition space into stacked hash tables across coarse-to-fine scales, trading modest memory for orders-of-magnitude faster feature lookup and training.
- **[NICE-SLAM](https://arxiv.org/abs/2112.12130)** (Zhu et al. 2022, CVPR 2022) replaces iMAP's single MLP with a hierarchical spatial feature grid coupled to pre-trained geometry decoders, localizing updates so large-scale indoor mapping remains stable without forgetting earlier rooms.

### 11.3.2 3DGS-SLAM: 3D Gaussian Splatting-Based SLAM

Even with hash grids, NeRF variants still evaluate dozens of samples along every ray (ray marching), which limits rendering throughput. **[3D Gaussian Splatting (3DGS)](https://arxiv.org/abs/2308.04079)** (Kerbl et al. 2023) removes ray marching altogether by representing the scene explicitly as a cloud of anisotropic 3D Gaussians, each parameterized by:

- Position $\boldsymbol{\mu} \in \mathbb{R}^3$
- 3D covariance $\boldsymbol{\Sigma} \in \mathbb{R}^{3 \times 3}$ (factored into rotation and scale)
- Opacity $\alpha \in [0, 1]$
- View-dependent color (spherical harmonics coefficients)

Instead of marching rays through a volume, 3DGS projects all Gaussians onto the image plane and alpha-blends them in depth order via tile-based rasterization, enabling real-time rendering under the original paper's scenes, resolutions, and hardware setup.

**[Gaussian Splatting SLAM](https://arxiv.org/abs/2312.06741)** (Matsuki et al. CVPR 2024; publicly released as **MonoGS**) brings explicit Gaussians into online SLAM, centering on three coupled tasks:

1. **Tracking**: Optimize camera pose directly against photometric and geometric residuals between the splatted image and the incoming frame.
2. **Mapping**: Maintain map detail by adding new Gaussians in under-reconstructed areas (densification), splitting overly large Gaussians, and pruning transparent or redundant ones.
3. **Loop closure**: Rigidly or non-rigidly deforming millions of Gaussians to match a corrected pose graph remains an active challenge—and the MonoGS paper itself does not implement loop closure.

The table below contrasts NeRF-SLAM and 3DGS-SLAM across representation, speed, memory, and loop-closure behavior:

| Property | NeRF-SLAM | 3DGS-SLAM |
|------|-----------|-----------|
| Representation | MLP weights | Explicit 3D Gaussian set |
| Rendering speed | Slow (ray marching) | Fast (rasterization) |
| Training speed | Slow | Fast |
| Editability | Hard | Easy (manipulate individual Gaussians) |
| Memory | Fixed for a single MLP (model size); grows with scene scale for spatial feature-grid variants | Variable (proportional to number of Gaussians) |
| Loop closure handling | Hard (weight deformation) | Relatively easy at the representation level (Gaussian transformation); full-system implementations remain rare |

Choosing between learned neural/Gaussian maps and classical TSDF or surfel maps hinges on sensor suite, scene scale, dynamic clutter, and whether photorealistic novel-view synthesis is actually needed. While neural representations jointly optimize geometry and appearance, memory scaling and consistent global loop correction still demand careful evaluation in large-scale, long-running deployments.

Several recent systems illustrate how rapidly this space is moving (2024–2025):

- **[SplaTAM](https://arxiv.org/abs/2312.02126)** (Keetha et al. CVPR 2024): Tracks and maps 3D Gaussians online from RGB-D input, expanding the map selectively using a silhouette mask; depending on the metric and scene, the paper reports up to 2x improvements across camera pose accuracy, reconstruction, and novel-view synthesis.
- **[MonoGS](https://arxiv.org/abs/2312.06741)** (Matsuki et al. CVPR 2024 Highlight): Runs unified monocular tracking, mapping, and rendering at 3 fps using 3D Gaussians as the sole representation, relying on geometric verification and isotropic regularization to curb monocular depth ambiguities.
- **[MASt3R-SLAM](https://arxiv.org/abs/2412.12392)** (Murai et al. CVPR 2025): Integrates the MASt3R 3D reconstruction foundation model into real-time SLAM—representing the map directly as dense predicted pointmaps rather than Gaussians—and reports 15 fps dense monocular SLAM without a predefined camera model.

The Python snippet below walks through the core forward pass of 3DGS rendering: projecting each 3D Gaussian's mean and covariance onto the image plane via the pinhole Jacobian (EWA splatting) and compositing depth-sorted splats front-to-back:

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

Geometric and photorealistic maps answer *"where is the surface?"*—yet an autonomous agent also needs to know *"what is it?"* Without semantics, a chair, a closed door, and a load-bearing wall are all indistinguishable triangles. Adding semantic meaning lets a robot reason about object categories, affordances, and human instructions.

### 11.4.1 Object-Level Maps

An **object-level map** elevates raw geometry into persistent 3D instances, tracking each recognized object's class, position, and spatial extent over time. Typically, such maps are constructed in four stages:

1. **2D detection or segmentation**: Run an image-space detector or segmenter (such as YOLO, Mask R-CNN, or SAM) on incoming frames.
2. **3D lifting**: Back-project masked depth pixels into world coordinates using the camera intrinsics and estimated pose.
3. **Data association**: Match newly lifted detections against existing map objects via spatial proximity, class agreement, or appearance tracking.
4. **Map integration**: Merge matched observations into the existing object's centroid, bounding box, or shape model, or spawn a new instance when no match exists.

The simplified `ObjectMap` class below illustrates this back-projection and centroid-based object association loop:

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

A flat catalog of isolated objects still misses the structural relationships humans rely on effortlessly: *"the cup sits on a table inside the living room."* A **3D Scene Graph** captures this multi-level organization explicitly, linking low-level metric geometry upward through objects, topological places, rooms, and buildings.

**[Hydra](https://arxiv.org/abs/2201.13360)** (Hughes et al. 2022) was an early open-source system to construct a multi-layer 3D scene graph incrementally online and deform all layers consistently when a loop closure fires. Its hierarchy spans five layers:

| Layer | Contents | Construction method |
|-------|------|-----------|
| 1 | Metric-Semantic 3D Mesh | TSDF integration + semantic segmentation |
| 2 | Objects & Agents | 3D bounding box detection |
| 3 | Places (topological) | Extracted from GVD (Generalized Voronoi Diagram) |
| 4 | Rooms | Community detection on the place graph |
| 5 | Buildings | Higher-level grouping of rooms |

To make navigation and structural reasoning practical online, Hydra builds the topological **Places** and **Rooms** layers directly from obstacle geometry:

- **Building the Places layer**: Starting from the TSDF, Hydra computes an ESDF and incrementally extracts its **Generalized Voronoi Diagram (GVD)**—the locus of points equidistant from two or more nearest obstacles, where graph vertices sit equidistant from four or more obstacles (while points equidistant from three obstacles trace 3D GVD edges). Because those loci maximize local obstacle clearance, turning GVD vertices into place nodes and GVD edges into traversability links yields a sparse, collision-safe topological map ready for path planning.
- **Detecting rooms from clearances**: Because doorways and hallways pinch clearance down to narrow bottlenecks, each place node's distance to its nearest obstacle provides a natural geometric cue for room boundaries. Hydra sweeps a clearance threshold upward (dilation) to sever narrow doorway links, inspects the connected-component count of the surviving subgraph at each threshold, and takes the components of the largest subgraph whose component count equals the median across thresholds as room seeds; modularity-based community detection initialized from those seeds then assigns every remaining threshold-removed place node to a room.
- **Hierarchical loop closure**: Hydra also feeds the scene graph back into localization. Descriptors drawn from the place and object layers narrow down loop candidates, and TEASER++ verifies 3D registration across the paired subgraphs—yielding more detected loops and higher accuracy in Hydra's reported experiments than appearance-only BoW retrieval.

Whereas Hydra builds the scene graph atop SLAM and deforms it upon loop closure, **S-Graphs** (Situational Graphs) pushes hierarchy directly *inside* factor-graph optimization: structural elements such as rooms, walls, and floors become explicit factor-graph variables coupled to robot poses, improving both trajectory accuracy and structural consistency simultaneously.

### 11.4.3 Open-Vocabulary Semantic Mapping

Closed-set detectors lock a semantic map to a fixed taxonomy—such as COCO's 80 categories—making them blind to novel items. **Open-vocabulary semantic mapping** lifts that ceiling by indexing 3D space with vision-language embeddings so users can query arbitrary natural-language phrases at runtime.

During mapping, the system extracts CLIP image features for each observed frame or segmented region and attaches them to the corresponding 3D points. When a user later searches for *"red fire extinguisher,"* the CLIP text encoder embeds the query into the same shared space, and cosine similarity surfaces the highest-scoring 3D locations—even if the robot never trained on a dedicated extinguisher class. Keep in mind that only models trained with an aligned vision-language objective like CLIP share a text embedding space; self-supervised visual features such as DINO cannot answer text queries directly, though they often serve alongside CLIP as a separate channel for sharper segmentation or cross-view consistency.

The `OpenVocabSemanticMap` class below sketches how 3D points store normalized CLIP embeddings and answer free-text queries by cosine similarity:

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

Real environments never stay frozen: people walk through corridors, chairs get rearranged, and outdoor scenery shifts with the weather and seasons. Any robot operating over days or months needs mechanisms to detect change, prune stale beliefs, and keep transient motion out of its permanent map.

### 11.5.1 Change Detection

Geometric change detection compares live sensor scans against the stored map to flag two complementary transitions:

1. **Free-to-occupied**: When a range beam terminates in a cell previously marked free, a new obstacle has entered the space.
2. **Occupied-to-free**: When a beam passes unimpeded through cells previously marked occupied, an earlier obstacle has moved or been removed.

The function below implements both checks over a 2D occupancy map by inspecting beam endpoints for newly occupied cells and tracing interior ray cells for cleared obstacles:

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

Beyond raw geometry, **semantic change detection** tracks shifts in an object's state or placement—distinguishing, for example, a moved chair or an opened door from structural walls.

### 11.5.2 Map Maintenance: Retention Strategies

Because finite onboard memory cannot archive every scan forever, long-term mapping relies on explicit retention policies:

- **Recency weighting**: Weight recent observations more heavily than older ones—for instance, by exponentially decaying accumulated TSDF weights so obsolete geometry gradually gives way to fresh measurements.
- **Semi-static classification**: Partition map elements by expected permanence: **static** structure (walls, floors, buildings) stays permanent, **semi-static** objects (furniture, parked vehicles) refresh periodically, and **dynamic** actors (pedestrians, moving cars) never enter the persistent map.
- **Multi-experience mapping**: Store multiple visual or geometric "experiences" of the same place across varying lighting conditions, weather, or seasons, selecting the experience closest to current conditions during localization.

### 11.5.3 Handling Dynamic Objects

Moving agents such as pedestrians and vehicles damage SLAM in two distinct ways: tracking features on a moving person corrupts ego-motion estimation, while baking transient points into the map leaves behind "ghost obstacles" that block future planning. Systems counter both failure modes through four complementary techniques:

1. **Semantic filtering**: Mask out known dynamic classes (people, vehicles) prior to feature tracking and map integration.
2. **Geometric consistency checks**: Flag points as dynamic when they should remain visible under predicted camera motion yet violate epipolar or reprojection residuals. Crucially, disappearance alone does not prove motion—static landmarks also drop out when they exit the field of view, fall behind static occlusion, or fail descriptor matching—so the test must combine visibility reasoning with reprojection error.
3. **Background subtraction**: In volumetric TSDF maps, identify voxels that flip repeatedly along a free → occupied → free cycle as transient motion.
4. **Joint SLAM and multi-object tracking (SLAM + MOT)**: Rather than discarding moving objects entirely, track them with dedicated dynamic state variables so the system estimates both its own pose and surrounding object trajectories.

The `DynamicMapManager` class below shows a simple semantic routing scheme that separates static structure, time-stamped semi-static objects with expiration cleanup, and tracked dynamic trajectories:

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

No single representation wins across every robotics workload, which is why modern systems layer them hierarchically—planning collision-free paths over OctoMap or ESDFs, reconstructing detailed appearance with surfels or 3D Gaussians, and answering high-level queries through semantic scene graphs. With all the core algorithmic building blocks now in place—from sensor modeling and state estimation through loop closure and spatial mapping—we turn in Chapter 12 to how complete autonomous-driving, drone, and handheld platforms integrate and benchmark these pieces in the real world (→ Chapter 12).
