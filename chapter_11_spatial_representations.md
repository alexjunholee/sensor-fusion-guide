# Ch.11 — Spatial Representations

Ch.6-10에서 센서 데이터로부터 로봇의 궤적을 추정하고, loop closure로 전역 일관성을 확보하는 과정을 다루었다. 이 챕터에서는 그 과정의 부산물이자 궁극적 목적인 맵 — 로봇이 세상을 기억하고 활용하는 형태 — 을 다룬다.

센서 퓨전의 궁극적 결과물은 **맵(map)** — 환경의 공간적 표현 — 이다. SLAM 시스템이 추정하는 것은 로봇의 궤적뿐 아니라, 그 궤적을 통해 관측한 환경의 구조이다. 어떤 형태로 환경을 표현하느냐에 따라 로봇이 할 수 있는 일이 달라진다. 경로 계획은 free/occupied 정보를 요구하고, 시각적 렌더링은 텍스처를 요구하며, 인간과의 상호작용은 의미론적 레이블을 요구한다.

이 챕터에서는 metric map(정량적 기하 맵)에서 출발하여, mesh, neural representation, semantic map, 그리고 장기 유지(long-term maintenance)까지 공간 표현의 전체 스펙트럼을 다룬다.

---

## 11.1 Metric Maps

Metric map은 환경의 기하학적 구조를 정량적 좌표로 표현한다. 가장 기본적이면서도 여전히 가장 널리 사용되는 형태다.

### 11.1.1 Occupancy Grid (2D/3D)

Occupancy grid는 공간을 균일한 격자(grid)로 분할하고, 각 셀이 occupied/free/unknown 상태를 갖는 확률적 표현이다.

**2D occupancy grid**: 평면 환경(실내, 단층)에서 사용한다. 각 셀 $m_i$에 대해 occupancy 확률 $p(m_i \mid z_{1:t})$를 베이즈 갱신으로 유지한다.

$$\text{log-odds}(m_i \mid z_{1:t}) = \text{log-odds}(m_i \mid z_{1:t-1}) + \text{log-odds}(m_i \mid z_t) - \text{log-odds}(m_i)$$

여기서 log-odds 표현을 사용하면 곱셈이 덧셈이 되어 계산이 효율적이고, 수치적으로도 안정적이다:

$$l(m_i) = \log\frac{p(m_i)}{1 - p(m_i)}$$

**문제**: 3D 환경을 표현하려면 3D occupancy grid가 필요한데, 해상도 $r$로 $L \times W \times H$ 공간을 표현하면 $(L/r)(W/r)(H/r)$ 개의 셀이 필요하다. 예를 들어 100 m $\times$ 100 m $\times$ 10 m 공간을 5 cm 해상도로 표현하면 약 $8 \times 10^9$개의 셀, 약 32 GB 메모리가 필요하다. 이는 비현실적이다.

```python
import numpy as np

class OccupancyGrid2D:
    """2D Occupancy Grid — log-odds 기반 베이즈 갱신."""
    
    def __init__(self, width_m, height_m, resolution=0.05):
        """
        Args:
            width_m: 맵 너비 (미터)
            height_m: 맵 높이 (미터)
            resolution: 셀 크기 (미터)
        """
        self.resolution = resolution
        self.width = int(width_m / resolution)
        self.height = int(height_m / resolution)
        
        # Log-odds 맵: 0 = unknown (prior = 0.5)
        self.log_odds = np.zeros((self.height, self.width), dtype=np.float32)
        
        # 센서 모델 파라미터
        self.l_occ = 0.85   # log-odds(occupied | hit)
        self.l_free = -0.40  # log-odds(free | pass-through)
        
        # Clipping 범위 (확률 포화 방지)
        self.l_min = -5.0
        self.l_max = 5.0
    
    def update(self, robot_x, robot_y, scan_ranges, scan_angles):
        """
        LiDAR 스캔으로 맵 갱신.
        
        Args:
            robot_x, robot_y: 로봇 위치 (미터)
            scan_ranges: 각 빔의 거리 (N,)
            scan_angles: 각 빔의 각도 (N,)
        """
        rx = int(robot_x / self.resolution)
        ry = int(robot_y / self.resolution)
        
        for r, angle in zip(scan_ranges, scan_angles):
            # 빔의 끝점 (occupied)
            ex = int((robot_x + r * np.cos(angle)) / self.resolution)
            ey = int((robot_y + r * np.sin(angle)) / self.resolution)
            
            # Bresenham line: 로봇 → 끝점 사이 = free
            cells = self._bresenham(rx, ry, ex, ey)
            for cx, cy in cells[:-1]:  # 마지막 셀 제외 (그건 occupied)
                if 0 <= cx < self.width and 0 <= cy < self.height:
                    self.log_odds[cy, cx] = np.clip(
                        self.log_odds[cy, cx] + self.l_free,
                        self.l_min, self.l_max
                    )
            
            # 끝점 셀 = occupied
            if 0 <= ex < self.width and 0 <= ey < self.height:
                self.log_odds[ey, ex] = np.clip(
                    self.log_odds[ey, ex] + self.l_occ,
                    self.l_min, self.l_max
                )
    
    def get_probability_map(self):
        """Log-odds → 확률 변환."""
        return 1.0 - 1.0 / (1.0 + np.exp(self.log_odds))
    
    def _bresenham(self, x0, y0, x1, y1):
        """Bresenham 직선 알고리즘 — 두 점 사이의 격자 셀 반환."""
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

균일 격자의 메모리 문제를 해결하기 위해, 다양한 적응적(adaptive) 자료구조가 제안되었다.

**[OctoMap](https://doi.org/10.1007/s10514-012-9321-0)** (Hornung et al. 2013): octree 기반의 확률적 3D occupancy map. 공간을 재귀적으로 8등분하여, occupied 또는 free인 영역은 조기에 분할을 중단(pruning)한다. 이를 통해 빈 공간은 큰 단위로, 세밀한 구조 근처만 작은 단위로 표현하여 메모리를 절약한다.

핵심 특성:
- **확률적 갱신**: occupancy grid와 동일한 log-odds 갱신 사용
- **적응적 해상도**: 최고 해상도(예: 2 cm, leaf node)부터 최저 해상도(예: 수 m, root 근처)까지 자동 조절
- **메모리 효율**: 64 m$^3$ 공간을 1 cm 해상도로 약 60 MB로 표현 가능 (균일 격자 대비 수백 배 절약)
- **한계**: 동적 삽입/삭제 시 tree 재균형 비용, 최근접 이웃 검색(kNN)이 느림

```python
class SimpleOctreeNode:
    """OctoMap의 핵심 개념 — 재귀적 8분할 octree."""
    
    def __init__(self, center, size, depth=0, max_depth=16):
        self.center = np.array(center)  # 노드 중심 좌표
        self.size = size                 # 노드 한 변 길이
        self.depth = depth
        self.max_depth = max_depth
        self.children = [None] * 8       # 8개 자식
        self.log_odds = 0.0              # occupancy log-odds
        self.is_leaf = True
    
    def get_child_index(self, point):
        """포인트가 어느 자식 octant에 속하는지 결정."""
        idx = 0
        if point[0] >= self.center[0]: idx |= 1
        if point[1] >= self.center[1]: idx |= 2
        if point[2] >= self.center[2]: idx |= 4
        return idx
    
    def get_child_center(self, child_idx):
        """자식 octant의 중심 좌표 계산."""
        offset = self.size / 4
        center = self.center.copy()
        center[0] += offset if (child_idx & 1) else -offset
        center[1] += offset if (child_idx & 2) else -offset
        center[2] += offset if (child_idx & 4) else -offset
        return center
    
    def update(self, point, is_occupied, l_occ=0.85, l_free=-0.40):
        """포인트로 octree 갱신."""
        if self.depth >= self.max_depth:
            # 최대 깊이 도달: leaf에서 log-odds 갱신
            self.log_odds += l_occ if is_occupied else l_free
            self.log_odds = np.clip(self.log_odds, -5.0, 5.0)
            return
        
        # 자식 octant 결정
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
        모든 자식이 같은 상태면 병합 (pruning).
        이것이 OctoMap의 핵심 메모리 절약 기법.
        """
        if self.is_leaf:
            return
        
        # 모든 자식이 존재하고 leaf이며 같은 상태인지 확인
        all_same = True
        first_odds = None
        
        for child in self.children:
            if child is None or not child.is_leaf:
                return  # pruning 불가
            
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

**OpenVDB**: 영화 VFX 산업에서 유래한 sparse volumetric 자료구조. 해시 맵 기반으로 활성화된 voxel만 저장하여, 넓은 공간에서 극소수의 voxel만 occupied인 경우 매우 효율적이다. OctoMap보다 탐색이 빠르지만, 확률적 갱신 기능은 사용자가 직접 구현해야 한다.

**ikd-tree** (FAST-LIO2): incremental k-d tree로, 포인트 삽입과 삭제를 $O(\log n)$에 수행하며 동적 재균형을 지원한다. FAST-LIO2에서 맵 자료구조로 사용되어, LiDAR 포인트를 실시간으로 맵에 추가하면서 kNN을 효율적으로 수행한다.

ikd-tree의 핵심 연산:
- **삽입**: 새 포인트를 k-d tree에 삽입. 불균형이 임계값을 초과하면 부분적으로 재균형.
- **삭제**: 일정 범위 밖의 오래된 포인트를 lazy deletion으로 제거.
- **kNN 검색**: 관측 포인트의 최근접 맵 포인트를 찾아 point-to-plane 정합에 사용.

OctoMap vs ikd-tree 비교:

| 특성 | OctoMap | ikd-tree |
|------|---------|----------|
| 자료구조 | Octree | k-d tree |
| 확률적 갱신 | 내장 (log-odds) | 없음 (포인트 저장만) |
| kNN 검색 | 느림 | 빠름 |
| 동적 삽입/삭제 | 가능하지만 느림 | $O(\log n)$ |
| 용도 | 경로 계획, 탐사 | LiDAR odometry 맵 유지 |

### 11.1.3 Surfel Maps

Surfel(surface element)은 포인트에 법선 벡터와 반경 정보를 추가한 디스크(disk) 형태의 표면 원소다. 각 surfel은 $(\mathbf{p}, \mathbf{n}, r, c)$로 표현된다:

- $\mathbf{p} \in \mathbb{R}^3$: 위치
- $\mathbf{n} \in \mathbb{R}^3$: 법선 벡터 (단위)
- $r \in \mathbb{R}^+$: 반경
- $c$: 색상/신뢰도

**[ElasticFusion](https://doi.org/10.15607/RSS.2015.XI.001)** (Whelan et al. 2015): RGB-D 센서로부터 surfel map을 실시간 구축하는 dense SLAM 시스템. 핵심 아이디어는:

1. Frame-to-model tracking: 현재 프레임을 surfel map의 렌더링과 정합한다.
2. Map deformation: loop closure 시 embedded deformation graph로 surfel map 전체를 비강체적으로 변형한다.
3. Surfel fusion: 기존 surfel과 새 관측이 겹치면 가중 평균으로 병합한다.

Surfel은 명시적 mesh 없이도 연속적 표면을 표현할 수 있다. 포인트 클라우드보다 표면 정보가 풍부하면서도, mesh보다 갱신이 간단하다.

```python
class Surfel:
    """하나의 Surface Element."""
    
    def __init__(self, position, normal, radius, color, confidence=1.0):
        self.position = np.array(position, dtype=np.float64)  # (3,)
        self.normal = np.array(normal, dtype=np.float64)      # (3,)
        self.radius = float(radius)
        self.color = np.array(color, dtype=np.float64)        # (3,) RGB
        self.confidence = confidence
        self.update_count = 1
    
    def fuse(self, new_pos, new_normal, new_radius, new_color, 
             new_confidence=1.0):
        """새 관측으로 surfel 갱신 (가중 평균)."""
        total_w = self.confidence + new_confidence
        
        self.position = (self.confidence * self.position 
                         + new_confidence * new_pos) / total_w
        
        # 법선: 가중 평균 후 정규화
        avg_normal = (self.confidence * self.normal 
                      + new_confidence * new_normal) / total_w
        norm = np.linalg.norm(avg_normal)
        if norm > 1e-6:
            self.normal = avg_normal / norm
        
        self.radius = (self.confidence * self.radius 
                       + new_confidence * new_radius) / total_w
        self.color = (self.confidence * self.color 
                      + new_confidence * new_color) / total_w
        
        self.confidence = min(total_w, 100.0)  # confidence 상한
        self.update_count += 1


class SurfelMap:
    """간단한 Surfel Map 구현."""
    
    def __init__(self, fusion_distance=0.02, fusion_normal_threshold=0.9):
        self.surfels = []
        self.fusion_dist = fusion_distance
        self.fusion_normal_thresh = fusion_normal_threshold
    
    def integrate(self, points, normals, radii, colors):
        """
        새 관측을 맵에 통합.
        기존 surfel과 가까우면 fusion, 아니면 새로 추가.
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

Mesh는 삼각형(triangle) 면의 집합으로 표면을 표현한다. Voxel이나 포인트 클라우드보다 시각적으로 자연스럽고, 물리 시뮬레이션(충돌 감지 등)에도 직접 활용할 수 있다.

### 11.2.1 TSDF (Truncated Signed Distance Function)

TSDF는 각 voxel에 가장 가까운 표면까지의 부호 거리(signed distance)를 저장한다:

- 양수: 표면 앞 (free space)
- 음수: 표면 뒤 (occupied)
- zero crossing: 실제 표면 위치

**TSDF 갱신**: 새 depth 관측이 들어올 때마다 해당 시선(ray)을 따라 voxel을 갱신한다:

$$\text{TSDF}(\mathbf{v}) \leftarrow \frac{W(\mathbf{v}) \cdot \text{TSDF}(\mathbf{v}) + w_{\text{new}} \cdot d_{\text{new}}}{W(\mathbf{v}) + w_{\text{new}}}$$

$$W(\mathbf{v}) \leftarrow \min(W(\mathbf{v}) + w_{\text{new}}, W_{\max})$$

여기서 $d_{\text{new}}$는 새로운 관측에서 계산한 부호 거리, $w_{\text{new}}$는 관측 가중치, $W(\mathbf{v})$는 누적 가중치다. 부호 거리는 truncation distance $\delta$ 이내로 제한(truncate)된다:

$$d_{\text{new}} = \text{clip}(D(\mathbf{u}) - \|\mathbf{v} - \mathbf{c}\|, -\delta, \delta)$$

$D(\mathbf{u})$는 픽셀 $\mathbf{u}$의 depth 값, $\mathbf{c}$는 카메라 위치, $\mathbf{v}$는 voxel 중심이다.

```python
class TSDFVolume:
    """
    TSDF 볼륨 — RGB-D 프레임으로부터 3D 재구성.
    """
    
    def __init__(self, volume_bounds, voxel_size=0.02):
        """
        Args:
            volume_bounds: [[x_min, x_max], [y_min, y_max], [z_min, z_max]]
            voxel_size: voxel 한 변 크기 (미터)
        """
        self.voxel_size = voxel_size
        self.bounds = np.array(volume_bounds)
        
        self.dims = np.ceil(
            (self.bounds[:, 1] - self.bounds[:, 0]) / voxel_size
        ).astype(int)
        
        # TSDF 값과 가중치
        self.tsdf = np.ones(self.dims) * 1.0  # 초기값: truncation 값
        self.weight = np.zeros(self.dims, dtype=np.float32)
        self.color = np.zeros((*self.dims, 3), dtype=np.float32)
        
        # Truncation distance
        self.trunc_dist = 3.0 * voxel_size
    
    def integrate(self, depth_image, color_image, K, T_camera_to_world):
        """
        하나의 RGB-D 프레임을 TSDF에 통합.
        
        Args:
            depth_image: (H, W) depth (미터)
            color_image: (H, W, 3) RGB (0~255)
            K: 카메라 내부 파라미터 (3, 3)
            T_camera_to_world: 카메라 pose (4, 4)
        """
        T_world_to_camera = np.linalg.inv(T_camera_to_world)
        cam_pos = T_camera_to_world[:3, 3]
        
        H, W = depth_image.shape
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]
        
        # 각 voxel에 대해
        for ix in range(self.dims[0]):
            for iy in range(self.dims[1]):
                for iz in range(self.dims[2]):
                    # Voxel 중심의 월드 좌표
                    vx = self.bounds[0, 0] + (ix + 0.5) * self.voxel_size
                    vy = self.bounds[1, 0] + (iy + 0.5) * self.voxel_size
                    vz = self.bounds[2, 0] + (iz + 0.5) * self.voxel_size
                    
                    # 카메라 좌표로 변환
                    p_world = np.array([vx, vy, vz, 1.0])
                    p_cam = T_world_to_camera @ p_world
                    
                    if p_cam[2] <= 0:
                        continue
                    
                    # 이미지 좌표로 투영
                    u = int(fx * p_cam[0] / p_cam[2] + cx)
                    v = int(fy * p_cam[1] / p_cam[2] + cy)
                    
                    if u < 0 or u >= W or v < 0 or v >= H:
                        continue
                    
                    depth = depth_image[v, u]
                    if depth <= 0:
                        continue
                    
                    # 부호 거리 계산
                    sdf = depth - p_cam[2]
                    
                    if sdf < -self.trunc_dist:
                        continue
                    
                    tsdf_val = min(sdf / self.trunc_dist, 1.0)
                    
                    # 가중 평균 갱신
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
        Marching Cubes로 TSDF에서 mesh 추출.
        zero crossing을 찾아 삼각형 면을 생성.
        """
        # Marching Cubes 알고리즘:
        # 각 voxel cube의 8개 꼭짓점에서 TSDF 부호를 확인하고,
        # 부호가 바뀌는 edge 위에 vertex를 보간하여 삼각형을 생성.
        
        vertices = []
        faces = []
        
        for ix in range(self.dims[0] - 1):
            for iy in range(self.dims[1] - 1):
                for iz in range(self.dims[2] - 1):
                    # 8개 꼭짓점의 TSDF 값
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
                    
                    # 부호 변화가 있는 경우에만 처리
                    if np.all(cube > 0) or np.all(cube < 0):
                        continue
                    
                    # 실제 Marching Cubes는 256가지 경우의
                    # look-up table을 사용하여 삼각형을 생성.
                    # 여기서는 개념만 표시.
                    pass
        
        return vertices, faces
```

### 11.2.2 Voxblox

**[Voxblox](https://arxiv.org/abs/1611.03631)** (Oleynikova et al. 2017)는 TSDF 기반 실시간 3D 재구성 시스템으로, 경로 계획에 필수적인 **ESDF (Euclidean Signed Distance Field)**를 효율적으로 계산한다.

핵심 파이프라인:

1. **TSDF 통합**: RGB-D 또는 depth 포인트 클라우드를 TSDF에 projective 방식으로 통합.
2. **Mesh 추출**: TSDF에서 Marching Cubes로 mesh를 incremental하게 추출. 변경된 voxel 블록만 재처리.
3. **ESDF 계산**: TSDF에서 ESDF를 계산. ESDF는 각 voxel에서 가장 가까운 장애물까지의 유클리드 거리를 저장한다.

ESDF가 중요한 이유: 경로 계획에서 로봇이 장애물로부터 안전 거리를 유지해야 하는데, ESDF가 있으면 임의의 점에서 장애물까지의 거리를 $O(1)$에 조회할 수 있다. 그래디언트 정보도 함께 제공하여, 장애물을 피하는 방향을 즉시 알 수 있다.

### 11.2.3 Poisson Surface Reconstruction

Poisson reconstruction은 oriented point cloud(위치 + 법선)로부터 수밀(watertight) mesh를 생성하는 방법이다.

**핵심 아이디어**: 법선 벡터를 gradient field로 해석하고, 이 gradient의 divergence를 구하여 Poisson 방정식을 푼다:

$$\nabla^2 \chi = \nabla \cdot \mathbf{V}$$

여기서 $\mathbf{V}$는 법선 벡터 필드, $\chi$는 indicator function(표면 안쪽 = 1, 바깥쪽 = 0)이다. 이 PDE를 octree 위에서 효율적으로 풀어 iso-surface를 추출한다.

장점은 노이즈 강건성과 수밀 mesh 생성이다. 포인트 밀도가 불균일해도 잘 동작한다. 실시간 SLAM에는 적합하지 않아 TSDF 기반 방법이 선호된다.

### 11.2.4 실시간 Mesh 생성

SLAM 시스템에서 실시간으로 mesh를 생성하는 현대적 접근:

1. **Voxblox incremental meshing**: TSDF가 갱신된 voxel 블록에서만 Marching Cubes를 재실행. 전체 볼륨을 재처리하지 않으므로 실시간 가능.
2. **FAST-LIVO2 mesh**: LiDAR 포인트에 카메라 색상을 부착하여 colored mesh를 실시간 생성. 통합 voxel map에서 Poisson 또는 Ball Pivoting으로 후처리.

---

## 11.3 Neural / Learned Representations

앞서 다룬 voxel, mesh, surfel 표현은 모두 명시적(explicit)이다 — 3D 구조를 직접 저장한다. 반면, neural representation은 **암묵적(implicit)** 방식으로 3D 장면을 신경망의 가중치에 인코딩한다.

### 11.3.1 NeRF-SLAM: Neural Implicit + Odometry

**[NeRF](https://arxiv.org/abs/2003.08934) (Neural Radiance Field)** (Mildenhall et al. 2020)는 3D 좌표와 시선 방향을 입력받아 색상과 밀도를 출력하는 MLP로 장면을 표현한다:

$$F_\theta: (\mathbf{x}, \mathbf{d}) \to (c, \sigma)$$

여기서 $\mathbf{x} \in \mathbb{R}^3$은 3D 위치, $\mathbf{d} \in \mathbb{S}^2$은 시선 방향, $c$는 RGB 색상, $\sigma$는 체적 밀도(volume density)다.

**볼륨 렌더링 방정식**: 카메라 ray $\mathbf{r}(t) = \mathbf{o} + t\mathbf{d}$를 따라 적분하여 픽셀 색상을 합성한다:

$$\hat{C}(\mathbf{r}) = \int_{t_n}^{t_f} T(t) \sigma(\mathbf{r}(t)) c(\mathbf{r}(t), \mathbf{d}) \, dt$$

여기서 $T(t) = \exp\left(-\int_{t_n}^{t} \sigma(\mathbf{r}(s)) ds\right)$는 누적 투과율(accumulated transmittance)이다.

**NeRF-SLAM 파이프라인**: 기존 NeRF가 offline reconstruction이었다면, NeRF-SLAM은 SLAM과 결합하여 online으로 동작한다:

1. **Tracking**: 현재 프레임의 camera pose를 기존 neural map에 대해 최적화한다 (photometric loss + depth loss).
2. **Mapping**: 추정된 pose에서 neural network 가중치를 갱신한다. 새로운 관측 영역을 학습하면서 기존 영역의 일관성도 유지해야 한다.

**[iMAP](https://arxiv.org/abs/2103.12352)** (Sucar et al. 2021): 최초의 neural implicit SLAM. 단일 MLP로 장면을 표현. 실시간성을 위해 keyframe 기반 학습과 active sampling 전략을 사용.

한계는 두 가지다. MLP 학습 속도(training speed)가 느려 실시간 mapping에 제약이 있고, 새 영역을 학습하면 이전 영역의 표현이 퇴화하는 catastrophic forgetting이 발생한다. 대규모 환경에서는 단일 MLP의 용량 한계도 드러난다.

개선 방향으로는 Instant-NGP가 주목된다. hash grid 기반 feature encoding으로 학습 속도를 수십 배 향상시켰고, 이를 활용한 NeRF-SLAM 변형이 등장했다. Instant-NGP의 핵심은 공간을 여러 해상도의 해시 테이블로 나누는 multi-resolution hash grid다.

**[NICE-SLAM](https://arxiv.org/abs/2112.12130)** (Zhu et al. 2022): iMAP의 확장성 문제를 해결하기 위해 계층적 feature grid와 사전학습된 기하 디코더를 도입한 dense SLAM 시스템. 대규모 실내 환경에서도 안정적으로 동작하며, CVPR 2022에서 발표되었다.

### 11.3.2 3DGS-SLAM: 3D Gaussian Splatting 기반 SLAM

**[3D Gaussian Splatting (3DGS)](https://arxiv.org/abs/2308.04079)** (Kerbl et al. 2023)은 NeRF의 대안으로 등장한 표현으로, 장면을 수백만 개의 3D Gaussian으로 표현한다. 각 Gaussian은:

- 위치 $\boldsymbol{\mu} \in \mathbb{R}^3$
- 공분산 $\boldsymbol{\Sigma} \in \mathbb{R}^{3 \times 3}$ (회전 + 스케일로 파라미터화)
- 불투명도 $\alpha \in [0, 1]$
- 색상 (spherical harmonics 계수)

렌더링은 splatting — 3D Gaussian을 이미지 평면에 투영하고, 깊이 순서대로 alpha blending — 으로 수행하며, NeRF의 ray marching보다 수십 배 빠르다.

**[3DGS-SLAM](https://arxiv.org/abs/2312.06741)** (Matsuki et al. 2024): 3DGS를 SLAM 표현으로 사용:

1. **Tracking**: Gaussian map을 렌더링한 예측 이미지와 실제 이미지의 photometric + geometric loss로 camera pose를 최적화.
2. **Mapping**: 새 관측에 따라 Gaussian을 추가(densification), 분할(splitting), 제거(pruning)한다.
3. **Loop closure**: pose 보정 시 Gaussian들의 위치도 함께 변형해야 한다.

**NeRF-SLAM vs 3DGS-SLAM 비교**:

| 특성 | NeRF-SLAM | 3DGS-SLAM |
|------|-----------|-----------|
| 표현 | MLP 가중치 | 명시적 3D Gaussian 집합 |
| 렌더링 속도 | 느림 (ray marching) | 빠름 (rasterization) |
| 학습 속도 | 느림 | 빠름 |
| 편집 가능성 | 어려움 | 용이 (개별 Gaussian 조작) |
| 메모리 | 고정 (모델 크기) | 가변 (Gaussian 수에 비례) |
| Loop closure 대응 | 어려움 (가중치 변형) | 상대적 용이 (Gaussian 변환) |

두 방법 모두 아직 전통적 맵 표현의 정확도와 실시간성을 모든 상황에서 능가하지는 못한다. 대규모 환경이나 장시간 SLAM에서는 TSDF 기반 방법이 여전히 더 안정적이다. 렌더링 품질에서는 neural representation이 압도적이며, 이 격차는 빠르게 좁혀지고 있다.

**최근 주요 발전 (2024~2025)**:

- **[SplaTAM](https://arxiv.org/abs/2312.02126)** (Keetha et al. CVPR 2024): RGB-D 카메라로부터 3D Gaussian을 실시간으로 추적·매핑하며, silhouette mask 기반의 구조화된 맵 확장으로 기존 방법 대비 camera pose 추정과 novel-view synthesis에서 2배 이상 성능 향상을 달성했다.
- **[MonoGS](https://arxiv.org/abs/2312.06741)** (Matsuki et al. CVPR 2024 Highlight): 최초의 monocular 3DGS SLAM으로, 3fps에서 단안 카메라만으로 tracking·mapping·렌더링을 통합 수행한다. geometric verification과 regularization으로 monocular 3D 재구성의 depth 모호성을 억제했다.
- **[MASt3R-SLAM](https://arxiv.org/abs/2412.12392)** (Murai et al. CVPR 2025): 3D reconstruction foundation model(MASt3R)을 SLAM에 통합한 시스템으로, 카메라 모델 가정 없이 15fps로 globally-consistent한 dense geometry를 복원한다.

```python
import numpy as np

class Gaussian3D:
    """단일 3D Gaussian 표현."""
    
    def __init__(self, mean, covariance, color, opacity):
        """
        Args:
            mean: 3D 위치 (3,)
            covariance: 3x3 공분산 행렬 (3, 3)
            color: RGB 색상 (3,)
            opacity: 불투명도 (scalar)
        """
        self.mean = np.array(mean, dtype=np.float64)
        self.covariance = np.array(covariance, dtype=np.float64)
        self.color = np.array(color, dtype=np.float64)
        self.opacity = float(opacity)
    
    def project_to_2d(self, T_world_to_cam, K):
        """
        3D Gaussian을 이미지 평면에 투영.
        
        Returns:
            mean_2d: 투영된 중심 (2,)
            cov_2d: 투영된 2D 공분산 (2, 2)
        """
        # 월드 → 카메라 좌표 변환
        R = T_world_to_cam[:3, :3]
        t = T_world_to_cam[:3, 3]
        
        mean_cam = R @ self.mean + t
        
        if mean_cam[2] <= 0:
            return None, None
        
        # 3D 공분산 → 카메라 좌표계
        cov_cam = R @ self.covariance @ R.T
        
        # 투영 Jacobian (pinhole model)
        fx, fy = K[0, 0], K[1, 1]
        z = mean_cam[2]
        
        J = np.array([
            [fx / z, 0, -fx * mean_cam[0] / z**2],
            [0, fy / z, -fy * mean_cam[1] / z**2]
        ])
        
        # 2D 공분산 (EWA splatting)
        cov_2d = J @ cov_cam @ J.T
        
        # 2D 중심
        mean_2d = np.array([
            fx * mean_cam[0] / z + K[0, 2],
            fy * mean_cam[1] / z + K[1, 2]
        ])
        
        return mean_2d, cov_2d


def render_gaussians(gaussians, T_world_to_cam, K, image_size):
    """
    3D Gaussian Splatting 렌더링 (간소화 버전).
    
    Args:
        gaussians: Gaussian3D 리스트
        T_world_to_cam: 카메라 pose (4, 4)
        K: 카메라 내부 파라미터 (3, 3)
        image_size: (H, W)
        
    Returns:
        rendered_image: (H, W, 3)
    """
    H, W = image_size
    image = np.zeros((H, W, 3), dtype=np.float64)
    accumulated_alpha = np.zeros((H, W), dtype=np.float64)
    
    # 깊이 순서로 정렬
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
        
        # 2D Gaussian의 영향 범위 (3 sigma)
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
                
                if power < -4.0:  # 너무 먼 픽셀은 스킵
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

기하학적 맵은 "어디에 무엇이 있는가"의 "어디"만 답한다. **Semantic map(의미 맵)**은 "무엇"까지 답한다 — 이것이 벽인지, 문인지, 의자인지, 사람인지. 로봇이 인간 수준의 환경 이해를 하려면 이 레이블 정보가 필수적이다.

### 11.4.1 Object-Level Maps

가장 직관적인 semantic map은 환경의 객체(object)를 인식하고 그 위치와 크기를 맵에 등록하는 것이다.

파이프라인:
1. **2D detection/segmentation**: 카메라 이미지에서 객체를 검출 (YOLO, Mask R-CNN, SAM 등).
2. **3D lifting**: depth 정보와 camera pose를 이용하여 2D 검출을 3D 공간으로 역투영.
3. **Data association**: 여러 프레임에서 관측된 같은 객체를 연결 (tracking + re-identification).
4. **Map integration**: 객체의 3D 바운딩 박스 또는 형상 모델을 맵에 등록.

```python
class ObjectMap:
    """3D 객체 맵 — 2D 검출을 3D 공간에 통합."""
    
    def __init__(self, association_threshold=0.5):
        self.objects = []  # [{class, center_3d, bbox_3d, observations, id}]
        self.next_id = 0
        self.assoc_thresh = association_threshold
    
    def integrate_detection(self, class_label, mask_2d, depth_image, 
                            K, T_cam_to_world):
        """
        2D 검출을 3D 맵에 통합.
        
        Args:
            class_label: 객체 클래스 (str)
            mask_2d: 2D segmentation mask (H, W) bool
            depth_image: depth map (H, W)
            K: 카메라 내부 파라미터 (3, 3)
            T_cam_to_world: 카메라 → 월드 변환 (4, 4)
        """
        # 1. 마스크 영역의 3D 포인트 복원
        points_3d = self._backproject(mask_2d, depth_image, K, T_cam_to_world)
        
        if len(points_3d) < 10:
            return
        
        center = np.mean(points_3d, axis=0)
        bbox_min = np.min(points_3d, axis=0)
        bbox_max = np.max(points_3d, axis=0)
        
        # 2. Data association: 기존 객체와 매칭
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
            # 기존 객체 갱신 (가중 평균)
            n = best_match['observations']
            best_match['center_3d'] = (
                (n * best_match['center_3d'] + center) / (n + 1)
            )
            best_match['observations'] = n + 1
        else:
            # 새 객체 등록
            self.objects.append({
                'class': class_label,
                'center_3d': center,
                'bbox_3d': (bbox_min, bbox_max),
                'observations': 1,
                'id': self.next_id
            })
            self.next_id += 1
    
    def _backproject(self, mask, depth, K, T):
        """마스크 영역의 2D 픽셀을 3D 월드 좌표로 역투영."""
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

Object-level map은 개별 객체만 인식한다. 하지만 인간은 환경을 "이 방에 테이블이 있고, 그 위에 컵이 있고, 이 방은 거실이다"와 같은 계층적 관계로 이해한다. **3D Scene Graph**는 이러한 계층적, 관계적 환경 표현이다.

**[Hydra](https://arxiv.org/abs/2201.13360)** (Hughes et al. 2022)는 실시간으로 3D scene graph를 점진적으로 구축하는 최초의 시스템이다. 5개 계층으로 구성된다:

| Layer | 내용 | 구성 방법 |
|-------|------|-----------|
| 1 | Metric-Semantic 3D Mesh | TSDF 통합 + semantic segmentation |
| 2 | Objects & Agents | 3D bounding box detection |
| 3 | Places (topological) | GVD(Generalized Voronoi Diagram)에서 추출 |
| 4 | Rooms | 장소 그래프의 커뮤니티 검출 |
| 5 | Buildings | 방들의 상위 그룹핑 |

**Places layer의 구축**: 이 계층이 Hydra의 가장 독창적인 부분이다.

1. TSDF에서 ESDF를 계산한다.
2. ESDF에서 GVD를 점진적으로 추출한다. GVD의 꼭짓점은 장애물로부터 최대한 먼 지점 — 즉, 로봇이 통과하기 좋은 지점 — 이다.
3. GVD 꼭짓점들을 place 노드로, GVD 에지를 place 간 연결로 구성한다.
4. 이 place graph는 경로 계획에 직접 사용할 수 있는 topological map이다.

**Room detection**: place graph에서 방을 검출하는 방법:

1. Place 노드들의 에지를 장애물 근접도에 따라 가중치를 부여한다.
2. 문(doorway) 같은 좁은 통로에서 가중치가 높아진다 (통과하기 어렵다는 의미).
3. 커뮤니티 검출 알고리즘(예: dilation 기반)으로 place들을 방 단위로 그룹핑한다.

**계층적 loop closure**: Hydra는 scene graph의 계층 구조를 활용하여 loop closure의 품질을 높인다. 먼저 상위 계층(room, place)에서 후보를 좁히고, 하위 계층(visual feature, object)에서 TEASER++ 기반으로 기하학적 검증을 수행한다. 이 top-down/bottom-up 구조 덕분에 단순 BoW 방식보다 더 많은 loop closure를 더 정확하게 확보할 수 있다.

**S-Graphs** (Situational Graphs): Hydra와 유사한 계층적 scene graph이지만, factor graph 최적화에 계층 정보를 직접 포함시킨다. Room, wall, floor 같은 구조적 요소를 factor graph의 변수로 추가하여 SLAM 정확도를 향상시킨다.

### 11.4.3 Open-Vocabulary Semantic Mapping

전통적 semantic mapping은 사전에 정의된 클래스 집합(예: COCO의 80개 클래스)에서만 동작한다. **Open-vocabulary semantic mapping**은 임의의 텍스트 질의로 맵을 탐색할 수 있게 한다.

파이프라인은 단순하다. 각 관측(이미지 또는 패치)에서 CLIP/DINO feature를 추출하여 대응하는 3D 위치에 부착한다. 사용자가 "빨간 소화기"라고 질의하면, CLIP text encoder로 텍스트를 인코딩하고 맵의 visual feature와 코사인 유사도를 계산하여 해당 위치를 반환한다.

로봇이 사전에 학습하지 않은 객체에도 동작한다는 것이 핵심이다. 예측 불가능한 환경에서 운용되는 가정용 로봇이나 탐사 로봇에 특히 유용하다.

```python
class OpenVocabSemanticMap:
    """
    Open-vocabulary semantic mapping의 핵심 개념.
    각 3D 포인트에 CLIP feature를 부착.
    """
    
    def __init__(self, feature_dim=512):
        self.points_3d = []     # (N, 3)
        self.features = []      # (N, D) CLIP visual features
        self.feature_dim = feature_dim
    
    def add_observation(self, points_3d, clip_features):
        """
        3D 포인트와 대응하는 CLIP feature를 맵에 추가.
        
        Args:
            points_3d: (M, 3) 3D 포인트 좌표
            clip_features: (M, D) 각 포인트의 CLIP visual feature
        """
        self.points_3d.extend(points_3d)
        self.features.extend(clip_features)
    
    def query(self, text_feature, top_k=10):
        """
        텍스트 질의로 맵에서 관련 위치를 검색.
        
        Args:
            text_feature: (D,) CLIP text feature
            top_k: 반환할 상위 결과 수
            
        Returns:
            results: [(point_3d, similarity)] 리스트
        """
        if len(self.features) == 0:
            return []
        
        features_array = np.array(self.features)
        points_array = np.array(self.points_3d)
        
        # 코사인 유사도 계산
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
        """텍스트 질의에 대한 관련도 히트맵 생성."""
        if len(self.features) == 0:
            return np.array([]), np.array([])
        
        features_array = np.array(self.features)
        points_array = np.array(self.points_3d)
        
        text_norm = text_feature / (np.linalg.norm(text_feature) + 1e-8)
        feat_norms = features_array / (
            np.linalg.norm(features_array, axis=1, keepdims=True) + 1e-8
        )
        
        similarities = feat_norms @ text_norm
        
        # 0~1로 정규화
        sim_min, sim_max = similarities.min(), similarities.max()
        if sim_max > sim_min:
            heatmap = (similarities - sim_min) / (sim_max - sim_min)
        else:
            heatmap = np.zeros_like(similarities)
        
        return points_array, heatmap
```

---

## 11.5 Long-Term & Dynamic Maps

실제 환경은 정적이지 않다. 사람이 오가고, 가구가 옮겨지고, 계절에 따라 외관이 달라진다. 장기간 운용되는 로봇은 이러한 변화에 적응해야 한다.

### 11.5.1 Change Detection (변화 감지)

맵과 현재 관측을 비교하여 변화된 부분을 식별한다.

**Geometric change detection**: 과거 맵의 예상 관측과 현재 센서 관측을 비교한다.

1. **Free-to-occupied**: 맵에서 free인 영역에 새 장애물이 감지되면 → 새 객체가 나타난 것.
2. **Occupied-to-free**: 맵에서 occupied인 영역을 현재 관측이 관통(pass-through)하면 → 기존 객체가 사라진 것.

```python
def detect_changes(occupancy_map, current_scan, robot_pose, threshold=0.3):
    """
    기존 맵과 현재 스캔을 비교하여 변화 감지.
    
    Returns:
        new_objects: 새로 나타난 장애물 위치들
        removed_objects: 사라진 장애물 위치들
    """
    new_objects = []
    removed_objects = []
    
    for point in current_scan:
        # 현재 스캔 포인트의 맵 좌표
        map_value = occupancy_map.get_probability(point)
        
        if map_value < threshold:
            # 맵에서는 free인데 현재 장애물 감지 → 새 객체
            new_objects.append(point)
    
    # 맵에서 occupied인데 현재 관통되는 영역 탐지
    for ray_endpoint in current_scan:
        ray_cells = trace_ray(robot_pose, ray_endpoint)
        for cell in ray_cells[:-1]:  # 끝점 제외 (여전히 occupied일 수 있음)
            map_value = occupancy_map.get_probability(cell)
            if map_value > (1.0 - threshold):
                # 맵에서 occupied인데 현재 ray가 관통 → 객체 제거됨
                removed_objects.append(cell)
    
    return new_objects, removed_objects
```

**Semantic change detection**: 기하학적 변화뿐 아니라 객체의 클래스나 상태 변화도 잡는다. "의자가 이동됨", "문이 열림/닫힘" 같은 사례가 여기 해당한다.

### 11.5.2 Map Maintenance: 유지 전략

장기 운용에서 맵의 모든 관측을 무한히 저장할 수는 없다. 어떤 정보를 유지하고 어떤 정보를 삭제할 것인가?

**Recency weighting**: 최근 관측에 더 높은 가중치를 주고 오래된 관측의 영향을 줄여간다. TSDF 가중치를 시간에 따라 감쇠하는 것이 대표적인 구현이다.

**Semi-static classification**: 환경 요소를 세 범주로 나눈다. 벽·바닥·건물 같은 정적(static) 요소는 영구 보존하고, 가구나 주차된 차량 같은 준정적(semi-static) 요소는 주기적으로 갱신하며, 보행자나 이동 차량 같은 동적(dynamic) 요소는 맵에서 제외한다.

**Multi-experience mapping**: 같은 장소의 여러 "경험(experience)"을 저장한다. 조명 조건이나 계절이 다른 여러 버전의 맵을 두고, 현재 관측과 가장 일치하는 경험을 고른다.

### 11.5.3 Dynamic Object 처리

SLAM에서 동적 객체(움직이는 사람, 차량)는 두 가지 문제를 일으킨다:

1. **Tracking 오류**: 동적 객체의 특징점을 정적 환경의 것으로 잘못 사용하면 pose 추정이 틀어진다.
2. **맵 오염**: 동적 객체가 맵에 기록되면 "유령 장애물"이 된다.

**대응 방법**:

1. **Semantic filtering**: semantic segmentation으로 동적 객체(사람, 차량) 클래스를 구분하고, 해당 관측을 SLAM 파이프라인에서 뺀다.

2. **Geometric consistency check**: 여러 프레임에 걸쳐 일관되지 않은 관측 — 한 프레임에서만 보이고 다음 프레임에서 사라지는 포인트 — 을 동적으로 분류한다.

3. **Background subtraction**: TSDF에서 free-to-occupied-to-free 패턴을 보이는 voxel을 동적으로 판별한다.

4. **SLAM with dynamic object tracking**: 동적 객체를 무시하는 대신 별도의 상태 변수로 추적한다. 동적 객체의 궤적도 함께 추정할 수 있다 (SLAM + MOT).

```python
class DynamicMapManager:
    """
    동적 환경 맵 관리자.
    관측을 static/semi-static/dynamic으로 분류하여 차별 처리.
    """
    
    def __init__(self):
        self.static_map = {}       # 영구 저장
        self.semistatic_map = {}   # 주기적 갱신
        self.dynamic_objects = []  # 추적 중인 동적 객체
        
        # 동적 클래스 정의
        self.dynamic_classes = {'person', 'car', 'bicycle', 'dog'}
        self.semistatic_classes = {'chair', 'box', 'parked_car'}
    
    def process_observation(self, point_3d, class_label, timestamp):
        """
        새 관측을 분류하고 적절한 맵에 통합.
        """
        if class_label in self.dynamic_classes:
            self._track_dynamic(point_3d, class_label, timestamp)
        elif class_label in self.semistatic_classes:
            self._update_semistatic(point_3d, class_label, timestamp)
        else:
            self._update_static(point_3d, class_label)
    
    def _track_dynamic(self, point, cls, timestamp):
        """동적 객체 추적. 맵에는 추가하지 않음."""
        # Data association으로 기존 동적 객체와 매칭
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
        """준정적 객체 갱신."""
        key = self._spatial_key(point)
        self.semistatic_map[key] = {
            'position': point.copy(),
            'class': cls,
            'last_seen': timestamp
        }
    
    def _update_static(self, point, cls):
        """정적 맵 갱신."""
        key = self._spatial_key(point)
        self.static_map[key] = {
            'position': point.copy(),
            'class': cls
        }
    
    def cleanup_old_semistatic(self, current_time, max_age=3600):
        """오래된 준정적 관측 제거 (예: 1시간)."""
        keys_to_remove = [
            k for k, v in self.semistatic_map.items()
            if current_time - v['last_seen'] > max_age
        ]
        for k in keys_to_remove:
            del self.semistatic_map[k]
    
    def _spatial_key(self, point, resolution=0.1):
        """3D 포인트를 이산 격자 키로 변환."""
        return (
            int(point[0] / resolution),
            int(point[1] / resolution),
            int(point[2] / resolution)
        )
```

---

각 표현은 서로 다른 downstream task에 맞는다. 현대 시스템은 이들을 계층적으로 조합한다 — OctoMap으로 경로를 계획하고, surfel로 렌더링하고, scene graph로 의미 질의를 처리한다. 다음 챕터에서는 이 알고리즘과 표현들이 자율주행·드론·핸드헬드 플랫폼에서 실제로 어떻게 통합되는지, 그리고 성능을 어떻게 평가하는지를 본다.
