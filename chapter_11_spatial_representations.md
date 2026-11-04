# Ch.11 — Spatial Representations

로봇은 센서 데이터로 궤적을 추정하고 loop closure로 궤적의 어긋남을 바로잡은 뒤, 그 결과를 맵으로 남긴다.

**맵(map)**은 환경을 공간적으로 나타낸 것이다. SLAM 시스템은 로봇의 궤적과 함께 그 궤적에서 관측한 환경의 구조를 추정한다. 맵을 어떤 형태로 두느냐에 따라 로봇이 그 맵으로 할 수 있는 일이 정해진다. 경로 계획에는 어디가 비어 있고 어디가 막혀 있는지(free/occupied)가 필요하다. 시각적 렌더링에는 텍스처가 필요하고, 사람과 말로 주고받으려면 의미 레이블이 필요하다.

---

## 11.1 Metric Maps

Metric map은 환경의 기하 구조를 좌표로 나타낸다. localization과 planning에 바로 쓸 수 있다. 어떤 metric map이 맞는지는 작업과 센서에 따라 다르다.

### 11.1.1 Occupancy Grid (2D/3D)

Occupancy grid는 공간을 같은 크기의 격자(grid)로 나누고 칸마다 점유 확률을 둔다. 칸의 상태는 occupied(점유), free(빈 곳), unknown(미관측)으로 읽는다.

**2D occupancy grid**: 단층 실내처럼 평면으로 볼 수 있는 환경에서 쓴다. 칸 $m_i$마다 occupancy 확률 $p(m_i \mid z_{1:t})$를 베이즈 갱신으로 고쳐 나간다.

$$\text{log-odds}(m_i \mid z_{1:t}) = \text{log-odds}(m_i \mid z_{1:t-1}) + \text{log-odds}(m_i \mid z_t) - \text{log-odds}(m_i)$$

log-odds로 쓰면 확률의 곱셈이 덧셈이 되어 계산이 가볍고, 확률이 0이나 1에 가까워져도 수치가 안정적이다. log-odds의 정의는 다음과 같다:

$$l(m_i) = \log\frac{p(m_i)}{1 - p(m_i)}$$

**메모리**: 해상도 $r$로 $L \times W \times H$ 공간을 빈틈없는(dense) 3D grid로 나타내면 $(L/r)(W/r)(H/r)$개의 cell이 필요하다. 예를 들어 100 m $\times$ 100 m $\times$ 10 m를 5 cm로 나누면 $2000 \times 2000 \times 200 = 8 \times 10^8$개 cell이다. cell마다 4바이트 float 하나만 저장해도 약 3.2 GB이고, 가중치·semantic 정보·자료구조 overhead를 더하면 더 커진다.

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
        self.l_free = -0.40  # log-odds(occupied | miss), 통과 관측의 갱신값
        
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

균일 격자의 메모리 문제를 줄이려고 필요한 곳만 저장하는(sparse) 자료구조들이 나왔다.

**[OctoMap](https://doi.org/10.1007/s10514-012-9321-0)** (Hornung et al. 2013): octree로 만든 확률적 3D occupancy map이다. 공간을 재귀적으로 8등분한다. 한 노드의 자식 8개가 모두 같은 점유 값으로 굳으면(log-odds가 clamping 상한이나 하한에 닿으면) 자식을 지우고 부모 노드 하나로 대신한다(pruning). 나중에 어긋나는 관측이 들어오면 자식을 다시 만든다. 그래서 넓은 빈 공간은 큰 칸 하나로, 구조가 복잡한 곳만 작은 칸으로 저장해 메모리를 아낀다.

OctoMap의 특성:
- **확률적 갱신**: occupancy grid와 같은 log-odds 갱신을 쓴다
- **적응적 해상도**: leaf node의 가장 고운 해상도(예: 2 cm)부터 root 근처의 거친 해상도(예: 수 m)까지 쓴다
- **메모리 효율**: 관측한 경계 근처만 잘게 나누므로 free나 unknown 영역이 넓은 장면에서는 dense grid보다 메모리를 덜 쓴다. 얼마나 줄어드는지는 점유 구조와 tree overhead에 따라 다르다.
- **한계**: pointer와 tree overhead가 있다. 점유 여부 조회에는 맞지만, kNN 검색이나 표면 정합(surface registration)용 index로 쓰기에는 맞지 않는다. 움직이는 물체를 지우려면 inverse sensor update와 시간에 따른 갱신 규칙을 따로 설계해야 한다.

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

**OpenVDB**: 영화 VFX 업계에서 만든 sparse volumetric 자료구조다. 깊이가 고정된 넓고 얕은 계층 tree를 쓰고, 값이 같은 넓은 background 영역은 tile 하나로 저장한다. OctoMap과 구조와 쓰임이 달라 어느 쪽이 더 빠르다고 일반적으로 말하기 어렵다. occupancy 확률 갱신은 OpenVDB에 들어 있지 않으므로 응용에서 직접 정해야 한다.

**[ikd-tree](https://arxiv.org/abs/2102.10808)** (Cai et al. 2021): 점을 하나씩 넣고 빼는 incremental k-d tree로, FAST-LIO2가 맵 자료구조로 쓴다(§7.3.2). 큰 부분 트리는 별도 스레드에서 다시 만든다.

OctoMap과 ikd-tree 비교:

| 특성 | OctoMap | ikd-tree |
|------|---------|----------|
| 자료구조 | Octree | k-d tree |
| 확률적 갱신 | 내장 (log-odds) | 없음 (포인트 저장만) |
| kNN 검색 | 검색용 index로 설계되지 않음 | 빠름 |
| 동적 삽입/삭제 | 가능 (관측마다 log-odds 갱신·pruning) | 점 단위 $O(\log n)$, 부분 트리 재구성은 별도 비용 |
| 용도 | 경로 계획, 탐사 | LiDAR odometry 맵 유지 |

### 11.1.3 Surfel Maps

Surfel(surface element)은 점에 법선 벡터와 반경을 더한 작은 원판(disk) 모양의 표면 조각이다. surfel 하나를 $(\mathbf{p}, \mathbf{n}, r, c, w)$로 나타낸다:

- $\mathbf{p} \in \mathbb{R}^3$: 위치
- $\mathbf{n} \in \mathbb{R}^3$: 법선 벡터 (단위)
- $r \in \mathbb{R}^+$: 반경
- $c$: 색상 (관측된 값)
- $w$: 융합 신뢰도 (누적 가중치)

**[ElasticFusion](https://doi.org/10.15607/RSS.2015.XI.001)** (Whelan et al., RSS 2015)은 RGB-D 센서로 surfel map을 실시간으로 만드는 dense SLAM 시스템이다. 동작 과정:

1. Frame-to-model tracking: 현재 프레임을 surfel map의 렌더링과 정합한다.
2. Map deformation: loop closure가 들어오면 embedded deformation graph로 surfel map 전체를 비강체적으로 변형한다.
3. Surfel fusion: 기존 surfel과 새 관측이 겹치면 가중 평균으로 합친다.

Surfel은 mesh를 따로 만들지 않고도 이어진 표면을 나타낸다. 점마다 법선과 크기가 있어 포인트 클라우드보다 표면 정보가 많고, 삼각형 사이 연결 관계가 없어 mesh보다 갱신이 간단하다.

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

Mesh는 삼각형(triangle) 면을 이어 표면을 나타낸다. Voxel이나 포인트 클라우드보다 보기에 자연스럽고, 충돌 감지 같은 물리 시뮬레이션에 바로 넣을 수 있다.

### 11.2.1 TSDF (Truncated Signed Distance Function)

TSDF는 voxel마다 시선 방향으로 잰 표면까지의 부호 거리(signed distance)를 저장한다. 여기서 다루는 것은 depth 관측을 투영해 얻는 projective TSDF다. 방향과 상관없이 가장 가까운 표면까지의 거리를 담는 ESDF는 §11.2.2에서 다룬다. TSDF 값의 부호:

- 양수: 표면 앞 (free space)
- 음수: 관측된 표면 뒤 (truncation band 안)
- zero crossing: 실제 표면 위치

표면 앞쪽 값은 $+\delta$에서 자른다. 표면 뒤로 $-\delta$보다 깊은 물체 내부는 아래 코드처럼 갱신하지 않고 기본값으로 둔다. 그래서 음수 TSDF 값은 표면 바로 뒤의 얇은 띠에만 있다. §11.1의 occupied처럼 그 칸이 막혀 있다고 확정하는 값과는 뜻이 다르다.

**TSDF 갱신**: 새 depth 관측이 들어올 때마다 그 시선(ray)을 따라 voxel 값을 가중 평균으로 고친다:

$$\text{TSDF}(\mathbf{v}) \leftarrow \frac{W(\mathbf{v}) \cdot \text{TSDF}(\mathbf{v}) + w_{\text{new}} \cdot d_{\text{new}}}{W(\mathbf{v}) + w_{\text{new}}}$$

$$W(\mathbf{v}) \leftarrow \min(W(\mathbf{v}) + w_{\text{new}}, W_{\max})$$

여기서 $d_{\text{new}}$는 새 관측에서 계산한 부호 거리, $w_{\text{new}}$는 관측 가중치, $W(\mathbf{v})$는 누적 가중치다. 부호 거리는 truncation distance $\delta$ 안으로 자른다(truncate):

$$d_{\text{new}} = \text{clip}\left(D(\mathbf{u}) - \lambda^{-1}\|\mathbf{v} - \mathbf{c}\|, -\delta, \delta\right), \quad \lambda = \|\mathbf{K}^{-1}\dot{\mathbf{u}}\|$$

$D(\mathbf{u})$는 픽셀 $\mathbf{u}$의 depth 값, $\mathbf{c}$는 카메라 위치, $\mathbf{v}$는 voxel 중심이다. $D(\mathbf{u})$가 광축 방향 깊이이므로 시선 길이 $\|\mathbf{v} - \mathbf{c}\|$를 $\lambda$로 나눠 같은 기준으로 맞춘 뒤 뺀다.

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

**[Voxblox](https://arxiv.org/abs/1611.03631)** (Oleynikova et al., IROS 2017)는 TSDF로 ESDF(Euclidean Signed Distance Field)를 점진적으로 만드는 시스템이다. 논문은 MAV의 궤적 최적화 local planner가 장애물 거리 정보가 필요해서 이 방법을 만들었다고 밝히고, CPU 코어 하나로 실시간으로 돈다는 것을 보였다. 샘플링 기반이나 탐색 기반 planner는 점유 맵만으로도 돌아가므로 모든 경로 계획에 ESDF가 필요하지는 않다.

동작 단계:

1. **TSDF 통합**: RGB-D나 depth 포인트 클라우드를 projective 방식으로 TSDF에 넣는다.
2. **Mesh 추출**: Marching Cubes로 TSDF에서 mesh를 뽑는다. 바뀐 voxel 블록만 다시 처리한다.
3. **ESDF 계산**: TSDF로 ESDF를 계산한다. ESDF는 voxel마다 가장 가까운 장애물까지의 유클리드 거리를 저장한다.

경로 계획에서는 로봇이 장애물에서 일정 거리 이상 떨어져 있어야 한다. ESDF가 있으면 어느 voxel에서든 장애물까지의 거리를 $O(1)$에 읽는다. 이웃 voxel 값으로 그래디언트를 구하면 장애물에서 멀어지는 방향도 바로 나온다.

### 11.2.3 Poisson Surface Reconstruction

Poisson reconstruction은 위치와 법선이 있는 점들(oriented point cloud)로 빈틈없이 닫힌(watertight) mesh를 만든다. 물체 안쪽은 1, 바깥은 0인 indicator function을 생각하면, 그 gradient는 표면 근처에서만 0이 아니고 안쪽을 향하는 법선(inward normal) 방향을 가리킨다. 점의 법선으로 이 gradient를 근사하는 vector field를 만들고, 아래 Poisson 방정식을 풀어 indicator function $\chi$를 구한 뒤 등위면을 mesh로 뽑는다.

$$\nabla^2 \chi = \nabla \cdot \mathbf{V}$$

$\mathbf{V}$가 앞에서 말한 vector field다. 원 논문은 이 방정식을 octree 위에서 푼다.

Poisson reconstruction은 장면 전체에 대한 implicit 함수 하나를 풀어 이어진 표면을 만든다. 결과는 법선 방향의 품질과 octree depth·screening 설정에 크게 좌우된다. 관측하지 못한 부분까지 닫아 watertight 표면을 만들기 때문에 원하지 않는 막이 생기기도 한다. depth를 계속 넣으면서 점진적으로 갱신해야 할 때는 TSDF meshing을 쓴다.

### 11.2.4 실시간 Mesh 생성

SLAM 시스템에서 mesh를 얻는 방법으로는 TSDF를 점진적으로 meshing하는 방법과, online으로 만든 point/surfel map을 따로 surface reconstruction하는 방법이 있다.

1. **Voxblox incremental meshing**: §11.2.2에서 본 대로 바뀐 voxel 블록만 다시 처리하므로 실시간으로 돌아간다.
2. **FAST-LIVO2 colored point map + 후처리**: FAST-LIVO2는 카메라 밝기값으로 visual update를 하고 색이 입혀진 point map을 내놓는다. 이 online 출력은 mesh가 아니다. mesh가 필요하면 내보낸 point map의 법선과 밀도를 확인한 뒤 Poisson이나 Ball Pivoting으로 따로 재구성한다.

---

## 11.3 Neural / Learned Representations

Voxel, mesh, surfel은 3D 구조를 그대로 저장하는 명시적(explicit) 표현이다. 학습으로 만드는 표현에는 MLP 하나의 가중치에 장면을 담는 암묵적(implicit) 방식, hash-grid나 계층 grid에 공간 feature를 두고 작은 MLP로 읽어 내는 방식, 3D Gaussian처럼 명시적 primitive를 최적화하는 방식이 있다.

### 11.3.1 NeRF-SLAM: Neural Implicit + Odometry

**[NeRF](https://arxiv.org/abs/2003.08934) (Neural Radiance Field)** (Mildenhall et al., ECCV 2020)는 3D 좌표와 시선 방향을 넣으면 색과 밀도를 내는 MLP로 장면을 나타낸다:

$$F_\theta: (\mathbf{x}, \mathbf{d}) \to (c, \sigma)$$

여기서 $\mathbf{x} \in \mathbb{R}^3$은 3D 위치, $\mathbf{d} \in \mathbb{S}^2$은 시선 방향, $c$는 RGB 색상, $\sigma$는 체적 밀도(volume density)다.

**볼륨 렌더링 방정식**: 카메라 ray $\mathbf{r}(t) = \mathbf{o} + t\mathbf{d}$를 따라 적분해 픽셀 색을 만든다:

$$\hat{C}(\mathbf{r}) = \int_{t_n}^{t_f} T(t) \sigma(\mathbf{r}(t)) c(\mathbf{r}(t), \mathbf{d}) \, dt$$

여기서 $T(t) = \exp\left(-\int_{t_n}^{t} \sigma(\mathbf{r}(s)) ds\right)$는 누적 투과율(accumulated transmittance)이다.

**NeRF-SLAM 파이프라인**: NeRF는 포즈를 아는 이미지들로 장면을 오프라인으로 재구성한다. NeRF-SLAM은 이 표현을 SLAM에 넣어 온라인으로 돌린다:

1. **Tracking**: 현재 neural map을 고정하고 photometric loss와 depth loss로 현재 프레임의 카메라 포즈를 최적화한다.
2. **Mapping**: 추정한 포즈로 network 가중치를 갱신한다. 새로 본 영역을 학습하는 동안 이미 학습한 영역이 망가지지 않게 해야 한다.

**[iMAP](https://arxiv.org/abs/2103.12352)** (Sucar et al., ICCV 2021)은 손에 든 RGB-D 카메라로 실시간 SLAM을 하면서 MLP 하나만 장면 표현으로 쓴 첫 시스템이다. 키프레임 구조, 멀티프로세싱, 정보량에 따라 픽셀을 고르는 sampling으로 tracking 10 Hz, 전역 맵 갱신 2 Hz를 얻었다.

MLP 학습이 느려 mapping도 느리다. 새 영역을 학습하면 이전 영역의 표현이 망가지는 catastrophic forgetting도 생긴다. 환경이 커지면 MLP 하나에 다 담기 어렵다.

Instant-NGP는 여러 해상도의 격자 위치마다 해시 테이블에 local feature를 저장하고 조회한다(multi-resolution hash-grid encoding). 이 방법으로 신경 장면 표현의 학습과 렌더링이 빨라졌고, 이를 쓴 NeRF-SLAM 변형도 나왔다.

**[NICE-SLAM](https://arxiv.org/abs/2112.12130)** (Zhu et al., CVPR 2022): 계층적 feature grid에 장면을 담고, 미리 학습한 기하 decoder를 써서 최적화하는 dense SLAM 시스템이다. 논문은 MLP 하나만 쓰는 방식이 장면을 지나치게 매끈하게 재구성하고 큰 장면으로 넓히기 어렵다고 지적하고, 이 구조로 큰 실내 장면에서 세밀한 재구성을 얻었다고 보고한다.

### 11.3.2 3DGS-SLAM: 3D Gaussian Splatting 기반 SLAM

**[3D Gaussian Splatting (3DGS)](https://arxiv.org/abs/2308.04079)** (Kerbl et al. 2023)은 장면을 수많은 3D Gaussian으로 나타낸다. Gaussian 하나에는 다음 값이 있다:

- 위치 $\boldsymbol{\mu} \in \mathbb{R}^3$
- 공분산 $\boldsymbol{\Sigma} \in \mathbb{R}^{3 \times 3}$ (회전 + 스케일로 파라미터화)
- 불투명도 $\alpha \in [0, 1)$
- 색상 (spherical harmonics 계수)

렌더링은 splatting으로 한다. 3D Gaussian을 이미지 평면에 투영한 뒤 깊이 순서대로 alpha blending한다. 원 논문은 1080p 해상도에서 30 fps 이상의 실시간 novel-view 렌더링을 보고했다.

**[Gaussian Splatting SLAM](https://arxiv.org/abs/2312.06741)** (Matsuki et al. CVPR 2024, 공개 구현 이름은 MonoGS): 3DGS를 SLAM의 장면 표현으로 쓴다. 이런 시스템은 다음 일을 해야 한다.

1. **Tracking**: Gaussian map을 렌더링한 이미지와 실제 이미지 사이의 photometric·geometric loss로 카메라 포즈를 최적화한다.
2. **Mapping**: 새 관측에 따라 Gaussian을 추가(densification), 분할(splitting), 제거(pruning)한다.
3. **Loop closure**: 포즈를 보정하면 Gaussian 위치도 같이 옮겨야 한다. Gaussian Splatting SLAM 논문은 loop closure를 넣지 않았고, 앞으로 넣을 과제로 남겼다.

**NeRF-SLAM과 3DGS-SLAM 비교**:

| 특성 | NeRF-SLAM | 3DGS-SLAM |
|------|-----------|-----------|
| 표현 | MLP 가중치 | 명시적 3D Gaussian 집합 |
| 렌더링 속도 | 느림 (ray marching) | 빠름 (rasterization) |
| 학습 속도 | 느림 | 빠름 |
| 편집 가능성 | 어려움 | 용이 (개별 Gaussian 조작) |
| 메모리 | 단일 MLP는 고정 (모델 크기), 공간 feature grid 변형은 장면 규모에 따라 증가 | 가변 (Gaussian 수에 비례) |
| Loop closure 대응 | 어려움 (가중치 변형) | 표현상으로는 상대적 용이 (Gaussian 변환). 예: [LoopSplat (3DV 2025)](https://arxiv.org/abs/2408.10154)은 3DGS submap끼리 직접 정합해 loop 제약을 만든다 |

Neural map과 TSDF·surfel map 가운데 무엇이 나은지는 센서, 장면 규모, 움직이는 물체, 장시간 운용, 렌더링 필요 여부에 따라 다르다. Neural representation은 새 시점 렌더링(novel-view rendering)까지 함께 최적화한다. 넓은 장소를 오래 다닐 때는 메모리가 얼마나 느는지, loop closure 보정을 어떻게 반영하는지 따로 따져 봐야 한다.

**최근 시스템 (2024~2025)**:

- **[SplaTAM](https://arxiv.org/abs/2312.02126)** (Keetha et al. CVPR 2024): RGB-D 카메라 하나로 3D Gaussian 맵을 만들며 online tracking과 mapping을 한다. silhouette mask로 이미 맵이 있는 영역을 가려내고, 빈 곳에 Gaussian을 더해 맵을 넓힌다. 논문은 카메라 포즈 추정, 맵 구축, novel-view synthesis에서 기존 방법보다 최대 2배 좋았다고 보고한다.
- **[MonoGS](https://arxiv.org/abs/2312.06741)** (Matsuki et al. CVPR 2024 Highlight): 위에서 다룬 Gaussian Splatting SLAM이다. 단안 SLAM에서 3D 표현으로 Gaussian만 쓰고 3 fps로 돈다. 점진적 재구성에서 생기는 모호성은 geometric verification과 regularization으로 다룬다. 깊이 센서가 있으면 RGB-D SLAM으로도 쓴다.
- **[MASt3R-SLAM](https://arxiv.org/abs/2412.12392)** (Murai et al. CVPR 2025 Highlight): 두 이미지에서 3D 구조와 대응을 예측하는 MASt3R를 바탕으로 처음부터 설계한 실시간 단안 dense SLAM이다. 맵은 MASt3R가 예측한 dense pointmap으로 이루어진다. 카메라 중심이 하나라는 것 말고는 카메라 모델을 가정하지 않으며, loop closure와 2차 전역 최적화를 포함해 15 FPS로 돈다.

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

기하 맵에는 어디가 막혀 있는지만 있다. Semantic map(의미 맵)에는 그것이 벽인지, 문인지, 의자인지, 사람인지 같은 레이블이 더 붙는다. 로봇이 물체와 그 종류를 구분하려면 이 레이블이 필요하다.

### 11.4.1 Object-Level Maps

Object-level map은 환경의 물체(object)를 인식해 그 위치와 크기를 맵에 올린다. 만드는 순서:
1. **2D detection/segmentation**: YOLO, Mask R-CNN, SAM 같은 모델로 카메라 이미지에서 물체를 찾는다.
2. **3D lifting**: depth와 카메라 포즈로 2D 검출 결과를 3D 공간에 역투영한다.
3. **Data association**: 여러 프레임에서 본 같은 물체를 tracking과 re-identification으로 묶는다.
4. **Map integration**: 물체의 3D 바운딩 박스나 형상 모델을 맵에 올린다.

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

3D Scene Graph는 object-level map에 물체, 방, 건물 사이의 계층과 관계를 더한다. 예: "거실에 테이블이 있고 테이블 위에 컵이 있다."

**[Hydra](https://arxiv.org/abs/2201.13360)** (Hughes et al., RSS 2022)는 로봇이 탐사하는 동안 센서 데이터로 3D scene graph의 계층들을 실시간으로 쌓아 간다. Loop closure가 들어오면 embedded deformation graph로 scene graph 전체를 보정한다. 논문은 이를 loop closure에 맞춰 3D scene graph를 최적화하는 첫 알고리즘이라고 밝힌다. 계층 구성:

| Layer | 내용 | 구성 방법 |
|-------|------|-----------|
| 1 | Metric-Semantic 3D Mesh | TSDF 통합 + semantic segmentation |
| 2 | Objects & Agents | object: 의미 레이블별로 mesh 정점을 Euclidean clustering (중심·bounding box) / agent: 로봇 궤적의 pose graph |
| 3 | Places (topological) | GVD(Generalized Voronoi Diagram)에서 추출 |
| 4 | Rooms | place 그래프 dilation + 커뮤니티 검출 |
| 5 | Buildings | 건물 노드 하나를 모든 방에 연결 (논문은 건물이 하나라고 가정) |

**Places 계층 만들기**:

1. TSDF에서 ESDF를 계산한다.
2. ESDF를 갱신하면서 GVD를 함께 얻는다. GVD는 가장 가까운 장애물 두 개 이상에서 거리가 같은 voxel의 집합으로, 환경의 뼈대 모양을 이룬다. Hydra는 장애물 넷 이상에서 거리가 같은 GVD voxel과, 모서리 모양 템플릿에 맞는 voxel을 place 노드로 삼는다.
3. Flood-fill로 GVD voxel에 가장 가까운 노드를 표시해 이웃 노드끼리 엣지 후보를 만들고, 직선 엣지가 GVD에서 너무 벗어나면 그 자리에 노드를 더해 엣지를 나눈다.
4. Place 노드는 장애물이 없는 위치이고, 엣지는 두 place 사이를 직선으로 지나갈 수 있다는 뜻이다. 논문은 이 부분 그래프를 사실상 topological map이라고 설명한다.

**Room detection**: place graph에서 방을 찾는 순서:

1. place 노드마다 장애물까지의 거리(clearance)가 있다.
2. 문(doorway) 같은 좁은 통로에서는 이 값이 낮다.
3. 맵을 점점 더 두껍게 부풀리며(dilation, 논문 예: 0.45~1.2 m 사이를 같은 간격으로 나눈 10단계) clearance가 그보다 작은 place를 지워 좁은 연결을 끊는다. 단계마다 남은 그래프의 연결 성분 수를 세고, 그 중앙값만큼 연결 성분을 가진 그래프 가운데 가장 큰 것을 고른다. 그 연결 성분들이 방의 씨앗이다. 지워져서 빠진 place는 이 씨앗을 초기 커뮤니티로 둔 greedy modularity 커뮤니티 검출로 방에 배정한다.

**계층적 loop closure**: Hydra는 계층마다 디스크립터를 둔다. 맨 아래 agent 계층에는 DBoW2 외관 디스크립터를, 그 위에는 object와 place의 통계를 담은 디스크립터를 써서 후보를 찾는다. 후보가 나오면 아래 계층부터 정합한다. 먼저 시각 특징을 RANSAC으로 정합하고, 실패하면 물체들을 TEASER++로 정합한다. 둘 다 실패하면 버린다. 논문은 이 방식이 BoW나 시각 특징 매칭 방식보다 루프를 더 많이, 더 정확하게 잡았다고 보고한다.

**S-Graphs** (Situational Graphs): Hydra처럼 계층적 scene graph를 쓴다. 차이는 계층 정보를 factor graph 최적화에 직접 넣는 데 있다. room, wall, floor 같은 구조 요소를 factor graph의 변수로 두고 함께 최적화한다.

### 11.4.3 Open-Vocabulary Semantic Mapping

기존 semantic mapping은 COCO의 80개 클래스처럼 미리 정한 클래스만 다룬다. Open-vocabulary semantic mapping에서는 아무 텍스트로나 맵을 검색한다.

이미지나 패치마다 CLIP image feature를 뽑아 그에 해당하는 3D 위치에 붙인다. 사용자가 "빨간 소화기"를 찾으면 CLIP text encoder로 그 문장을 벡터로 바꾸고, 맵의 CLIP feature와 코사인 유사도를 계산해 가장 비슷한 위치를 돌려준다. 이 구성에서 텍스트와 같은 임베딩 공간을 쓰는 것은 CLIP image encoder다. DINO 특징을 함께 쓸 때는 분할이나 프레임 사이 대응 같은 다른 용도에 쓴다.

CLIP text encoder로 질의하므로 학습할 때 클래스로 정하지 않은 물체도 찾는다. 어떤 물체가 있을지 미리 알 수 없는 가정용 로봇이나 탐사 로봇에 쓴다.

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

실제 환경은 계속 바뀐다. 사람이 오가고, 가구 위치가 바뀌고, 계절에 따라 겉모습이 달라진다. 오래 운용하는 로봇의 맵은 이런 변화를 따라가야 한다.

### 11.5.1 Change Detection (변화 감지)

맵과 현재 관측을 비교해 바뀐 곳을 찾는다.

**Geometric change detection**: 과거 맵의 예상 관측과 현재 센서 관측을 비교한다.

1. **Free-to-occupied**: 맵에서 free였던 곳에 장애물이 보이면 새 물체가 나타난 것이다.
2. **Occupied-to-free**: 맵에서 occupied였던 곳을 지금 센서 빔이 통과하면(pass-through) 있던 물체가 사라진 것이다.

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

**Semantic change detection**: 모양 변화에 더해 물체의 종류나 상태가 바뀐 것도 잡는다. 의자가 옮겨졌거나 문이 열리고 닫힌 경우가 그렇다.

### 11.5.2 Map Maintenance: 유지 전략

오래 운용하면 관측을 모두 저장할 수 없으므로, 무엇을 남기고 무엇을 지울지 기준이 필요하다.

**Recency weighting**: 최근 관측에 가중치를 더 주고 오래된 관측의 영향은 줄여 간다. 예를 들어 TSDF 가중치를 시간에 따라 줄인다.

**Semi-static classification**: 벽·바닥·건물 같은 정적(static) 요소는 계속 남긴다. 가구나 주차된 차량 같은 준정적(semi-static) 요소는 주기적으로 갱신한다. 보행자나 이동 차량 같은 동적(dynamic) 요소는 맵에서 뺀다.

**Multi-experience mapping**: 같은 장소를 지난 여러 번의 기록(experience)을 따로 저장한다. 조명이나 계절이 다른 여러 맵을 두고, 지금 관측과 가장 잘 맞는 것을 골라 쓴다.

### 11.5.3 Dynamic Object 처리

움직이는 사람이나 차량 같은 동적 객체가 SLAM에 일으키는 문제:

1. **Tracking 오류**: 움직이는 물체의 특징점을 정적 환경의 것으로 알고 쓰면 포즈 추정이 틀어진다.
2. **맵 오염**: 지나간 물체가 맵에 남으면 실제로는 없는 장애물이 된다.

**대응 방법**:

1. **Semantic filtering**: semantic segmentation으로 사람·차량처럼 움직일 수 있는 클래스를 찾아 그 관측을 SLAM 입력에서 뺀다.

2. **Geometric consistency check**: 로봇 자신의 움직임만으로는 새 관측 위치가 설명되지 않는 점을 동적으로 분류한다. 점이 안 보이는 것은 시야를 벗어나거나, 정적 구조에 가려지거나, 대응을 못 찾아서일 수도 있다. 그래서 동적 여부는 그 점이 보여야 하는 자리인지 따지는 가시성 추론과 재투영 잔차로 판정한다.

3. **Background subtraction**: TSDF에서 free-to-occupied-to-free 패턴을 보이는 voxel을 동적으로 판별한다.

4. **SLAM with dynamic object tracking**: 움직이는 물체를 빼지 않고 상태 변수를 따로 두어 그 궤적도 함께 추정한다(SLAM + MOT, multi-object tracking).

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

한 시스템 안에서 여러 표현을 함께 쓰기도 한다. 예를 들어 OctoMap으로 경로를 계획하고, surfel로 렌더링하고, scene graph로 의미 질의를 처리한다.
