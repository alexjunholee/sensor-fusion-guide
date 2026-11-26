# 11장 공간 표현

> 일관된 포즈 궤적을 손에 쥐었다면, 이제 그 궤적 위에서 모은 센서 데이터를 로봇의 목적에 맞는 '지도'라는 그릇에 담아야 한다.

10장에서 다룬 루프 클로저와 전역 최적화는 로봇이 지나온 궤적의 뒤틀림을 바로잡는 과정이었다(→ 10장). 이렇게 반듯하게 정렬된 포즈 추정치와 각 시점의 센서 관측을 한데 엮어 환경의 물리적·의미적 구조를 기록한 결과물이 바로 **맵(map, 지도)**이다.

세상의 모든 목적을 단번에 만족시키는 단일한 만능 지도 표현은 존재하지 않는다. 공간을 어떤 자료구조로 빚어내느냐에 따라 로봇이 그 지도 위에서 수행할 수 있는 작업이 곧바로 갈린다. 예컨대 경로 계획(path planning) 모듈은 어느 곳이 뚫려 있고 어디가 막혀 있는지(free/occupied)를 빠르게 판별하는 기하 지도가 필요하다. 시뮬레이션이나 디지털 트윈을 위한 시각적 렌더링에는 매끈한 표면과 텍스처가 있어야 한다. 또 사람이 건네는 자연어 명령("거실 테이블 위의 컵을 가져와")을 실행하려면 사물의 종류와 공간적 관계를 담은 의미 계층이 뒷받침되어야 한다.

이 장에서는 가장 기초적인 이산 격자 기반의 메트릭 맵(occupancy grid, voxel, surfel)에서 출발해, 연속적인 표면을 복원하는 메쉬(TSDF, ESDF, Poisson)를 거쳐, 최근 급부상한 신경망·가우시안 기반 학습 표현(NeRF-SLAM, 3DGS-SLAM), 고차원 의미 추론을 돕는 3D Scene Graph와 open-vocabulary 맵, 그리고 시간이 흐르며 변하는 현실 세계를 따라가는 장기·동적 맵 관리 기법까지 차례로 살펴본다.

---

## 11.1 Metric Maps

**메트릭 맵(metric map)**은 환경의 기하학적 구조를 실제 미터 단위의 3D 좌표계에 정량적으로 기록한다. 로봇의 자기 위치 추정(localization)과 충돌 없는 경로 생성(planning)에 즉각 활용할 수 있는 가장 기본적인 공간 표현이다. 플랫폼에 탑재된 센서와 주된 임무에 따라 격자, 복셀, 서펠 등 세부 형태를 고른다.

### 11.1.1 Occupancy Grid (2D/3D)

가장 직관적인 메트릭 맵은 **점유 격자 지도(occupancy grid)**다. 연속적인 물리 공간을 일정한 크기의 정사각형(2D) 또는 정육면체(3D) 셀로 잘게 쪼개고, 각 셀이 장애물로 차 있는지 비어 있는지를 확률로 저장한다. 센서가 닿아 장애물이 확인된 셀은 **occupied(점유)**, 센서 빔이 아무것에도 걸리지 않고 통과한 셀은 **free(빈 공간)**, 아직 센서 시야가 미치지 않은 영역은 **unknown(미관측)** 상태로 구분한다.

**2D occupancy grid**: 창고나 실내 복도처럼 바닥 평면을 따라 움직이는 환경에서는 2D 격자만으로도 충분하다. 각 격자 셀 $m_i$가 점유되어 있을 사후 확률 $p(m_i \mid z_{1:t})$는 새로운 센서 관측 $z_t$가 들어올 때마다 베이즈 정리(Bayes' rule)로 갱신한다. 이때 확률 $p$를 그대로 곱하는 대신 **로그 오즈(log-odds)** 형태로 변환해 두면, 확률의 곱셈이 단순한 덧셈으로 바뀌어 연산이 가벼워질 뿐 아니라, 확률값이 0이나 1에 극단적으로 가까워져도 부동소수점 언더플로 없이 수치적으로 안정되게 갱신할 수 있다.

$$\text{log-odds}(m_i \mid z_{1:t}) = \text{log-odds}(m_i \mid z_{1:t-1}) + \text{log-odds}(m_i \mid z_t) - \text{log-odds}(m_i)$$

여기서 임의의 셀 $m_i$의 log-odds는 점유 확률과 비점유 확률의 비율에 로그를 취해 다음과 같이 정의한다(사전 확률을 $p(m_i)=0.5$로 두면 초기 log-odds $\text{log-odds}(m_i)=0$이 된다).

$$l(m_i) = \log\frac{p(m_i)}{1 - p(m_i)}$$

**메모리의 벽**: 2D에서 잘 작동하던 균일 격자를 3D 공간으로 그대로 확장하면 메모리 요구량이 세제곱으로 폭증한다. 가로·세로·높이가 $L \times W \times H$인 공간을 해상도 $r$의 촘촘한(dense) 3D 격자로 채우려면 총 $(L/r)(W/r)(H/r)$개의 셀을 미리 할당해야 한다. 예컨대 $100\text{ m} \times 100\text{ m} \times 10\text{ m}$ 크기의 건물 한 층을 $5\text{ cm}$ 해상도로 나누면 $2000 \times 2000 \times 200 = 8 \times 10^8$개의 셀이 생긴다. 셀 하나에 4바이트 float 값 딱 하나만 저장해도 약 $3.2\text{ GB}$가 필요하고, 여기에 신뢰도 가중치나 의미 레이블, 자료구조 오버헤드까지 얹히면 감당하기 어려워진다.

아래 `OccupancyGrid2D` 클래스는 log-odds 기반 베이즈 갱신의 기본 구현이다. 2D LiDAR 스캔을 받아 Bresenham 직선 알고리즘으로 빔이 통과한 셀에는 `l_free`를, 빔이 부딪힌 끝점 셀에는 `l_occ`를 누적한다. 확률이 0 또는 1에 포화되어 이후 환경 변화를 반영하지 못하는 현상을 막기 위해 `l_min`, `l_max` 범위로 값을 잘라 내는 클리핑(clipping)이 적용되어 있다.

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

앞에서 본 균일 3D 격자에서는 텅 빈 허공이나 아직 한 번도 쳐다보지 않은 공간까지 똑같은 메모리를 할당하므로 결정적인 낭비가 생긴다. 이 병목을 돌파하기 위해 실제로 관측된 영역이나 구조가 복잡한 경계 부근만 선택적으로 저장하는 **희소(sparse) 자료구조**가 개발되었다.

**[OctoMap](https://doi.org/10.1007/s10514-012-9321-0)**(Hornung et al. 2013)은 옥트리(octree) 구조를 활용해 확률적 3D 점유 지도를 구현한 고전적인 표준이다. 큰 정육면체 공간을 재귀적으로 8등분(octant)해 내려가되, 어느 부모 노드 아래에 달린 8개의 자식 노드가 모두 동일한 점유 상태(예: log-odds가 clamping 상한이나 하한에 도달)로 수렴하면 자식 노드들을 메모리에서 지우고 부모 노드 하나로 합쳐 버린다. 이를 **가지치기(pruning)**라 한다. 훗날 그 공간에 상반되는 센서 관측이 들어오면 다시 자식을 분할해 세밀함을 되살린다.

덕분에 OctoMap은 다음과 같은 실무적 성격을 띤다.
- **확률적 갱신**: 앞서 본 occupancy grid와 동일한 log-odds 베이즈 갱신 규칙을 내장해 센서 노이즈를 걸러 낸다.
- **적응적 해상도**: 트리의 깊이(depth)에 따라 가장 촘촘한 리프 노드의 세밀한 해상도(예: $2\text{ cm}$)부터 루트 근처의 거친 블록(예: 수 m)까지 한 트리에 공존한다.
- **메모리 효율**: 벽면이나 장애물 경계 주변만 잘게 쪼개고 넓은 빈 공간이나 미관측 영역은 큰 노드 하나로 뭉뚱그리므로, 3D 공간의 상당 부분이 비어 있는 일반 환경에서 dense 격자 대비 메모리를 크게 아낄 수 있다(실제 절감 비율은 공간의 점유 구조와 트리 포인터 오버헤드에 좌우된다).
- **구조적 한계**: 포인터를 따라 트리를 내려가야 하는 간접 참조 오버헤드가 있다. 특정 좌표가 비어 있는지 막혀 있는지를 조회하는 경로 계획 용도에는 잘 맞지만, 고속 최근접 이웃(kNN) 탐색이나 LiDAR 스캔 정합용 인덱스로 쓰기에는 적합하지 않다. 또한 움직이는 물체를 지워 내려면 역센서 모델(inverse sensor model) 기반 ray casting과 시간에 따른 갱신 규칙을 별도로 정교하게 설계해야 한다.

아래 `SimpleOctreeNode` 클래스에는 OctoMap의 핵심 아이디어가 담겨 있다. 여기서는 3D 좌표를 8개 octant 중 하나로 분기해 내려가며 log-odds를 갱신하고, 여덟 자식이 같은 상태로 수렴했을 때 병합해 메모리를 회수한다.

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

Octree 외에도 목적에 따라 두 가지 희소 볼륨·포인트 자료구조가 널리 쓰인다.

**OpenVDB**는 영화 시각효과(VFX) 산업에서 거대한 연기·유체 시뮬레이션을 다루기 위해 탄생한 희소 볼륨(sparse volumetric) 자료구조다. 깊이 제한 없이 2분할·8분할로 길게 내려가는 octree와 달리 깊이가 고정된 넓고 얕은(wide and shallow) 계층 트리를 사용하며, 값이 균일한 넓은 배경 영역은 별도의 타일(tile) 하나로 압축해 저장한다. 다만 OctoMap과는 트리 구조와 주된 쓰임새(로보틱스 점유 지도 vs. 범용 볼륨 그리드)가 달라 어느 한쪽이 일방적으로 빠르다고 단정하기 어려우며, occupancy 확률 갱신 로직은 OpenVDB 자체에 내장되어 있지 않으므로 응용 단에서 직접 구현해 얹어야 한다.

**[ikd-tree](https://arxiv.org/abs/2102.10808)**(Cai et al. 2021)는 점유 확률 대신 원시 3D 포인트 클라우드의 고속 탐색에 특화된 증분형(incremental) k-d 트리다. 로봇이 전진함에 따라 새로 들어온 LiDAR 점을 실시간으로 삽입하고 오래된 영역을 잘라 내되, 트리 균형이 무너지면 큰 부분 트리를 재구성하는 작업을 별도의 백그라운드 스레드에 넘겨 실시간 주기를 놓치지 않는다. 7장에서 살펴본 FAST-LIO2가 바로 이 ikd-tree를 핵심 맵 자료구조로 채택해 고속 LiDAR-관성 오도메트리를 구현했다(§7.3.2).

로보틱스 현장에서 자주 비교되는 OctoMap과 ikd-tree의 성격 차이는 다음 표로 요약된다.

| 특성 | OctoMap | ikd-tree |
|------|---------|----------|
| 자료구조 | Octree | k-d tree |
| 확률적 갱신 | 내장 (log-odds) | 없음 (포인트 저장만) |
| kNN 검색 | 검색용 index로 설계되지 않음 | 빠름 |
| 동적 삽입/삭제 | 가능 (관측마다 log-odds 갱신·pruning) | 점 단위 $O(\log n)$, 부분 트리 재구성은 별도 비용 |
| 용도 | 경로 계획, 탐사 | LiDAR odometry 맵 유지 |

이처럼 표에서 보듯 두 자료구조는 경쟁 관계라기보다 역할이 뚜렷이 나뉜다. 장애물 회피와 자율 탐사를 위해 빈 공간과 점유 공간을 확률적으로 가려내야 할 때는 OctoMap이 어울리고, 오도메트리 스캔 매칭을 위해 매 프레임 수천 개의 최근접 3D 점을 밀리초 단위로 찾아내야 할 때는 ikd-tree가 위력을 발휘한다.

### 11.1.3 Surfel Maps

격자나 복셀이 공간을 작은 상자들로 쪼개는 방식이라면, **서펠(surfel, surface element)**은 아예 물체의 표면 자체를 지향성을 띤 작은 원판(disk) 조각들의 모음으로 직접 표현한다. 하나의 surfel은 중심 위치, 표면 방향, 크기, 색상, 신뢰도를 담은 튜플 $(\mathbf{p}, \mathbf{n}, r, c, w)$로 정의된다.

- $\mathbf{p} \in \mathbb{R}^3$: 3D 공간상의 원판 중심 위치
- $\mathbf{n} \in \mathbb{R}^3$: 표면에 수직인 단위 법선(normal) 벡터
- $r \in \mathbb{R}^+$: 원판의 반경
- $c$: 관측을 융합해 얻은 RGB 색상
- $w$: 여러 프레임에 걸쳐 누적된 융합 신뢰도(가중치)

RGB-D 실시간 조밀 SLAM의 대표작인 **[ElasticFusion](https://doi.org/10.15607/RSS.2015.XI.001)**(Whelan et al., RSS 2015)은 바로 이 surfel map을 중심에 두고 다음과 같이 동작한다.

1. **Frame-to-model tracking**: 현재 카메라 프레임의 깊이·색상 영상을 기존 surfel map에서 투영해 렌더링한 모델 뷰와 직접 정합해 포즈를 추정한다.
2. **Map deformation**: 루프 클로저가 탐지되면 포즈 그래프만 고치는 대신, 공간 위에 성긴 변형 그래프(embedded deformation graph)를 얹어 surfel map 전체를 고무판처럼 비강체적으로 구부려 아귀를 맞춘다.
3. **Surfel fusion**: 새로 관측된 점과 위치·법선이 비슷한 기존 surfel을 찾으면, 관측 신뢰도를 가중치로 삼아 위치·법선·반경·색상을 가중 평균으로 부드럽게 융합한다.

Surfel 표현의 묘미는 무거운 삼각형 연결망(메쉬)을 따로 엮지 않고도 시각적으로 연속된 표면을 그럴듯하게 표현할 수 있다는 데 있다. 점마다 법선과 반경을 품고 있어 단순 포인트 클라우드보다 훨씬 풍부한 표면 기하 정보를 담는다. 그러면서도 정점(vertex)과 면(face) 사이의 위상(topology) 연결 관계를 유지할 필요가 없으므로, 새 관측을 융합하거나 루프 클로저 이후 지도를 변형하는 갱신 작업이 메쉬보다 훨씬 간결하다.

아래 `Surfel`과 `SurfelMap` 코드에는 기본 융합 로직이 담겨 있다. 위치 거리와 법선 내적 조건을 만족하는 기존 surfel을 찾으면 가중 평균으로 정보를 누적하고, 일치하는 요소가 없으면 새 surfel을 추가한다.

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

서펠이나 복셀은 3D 공간이나 표면을 이산적인 조각으로 표현하므로, 그래픽스 렌더링 파이프라인이나 정밀한 물리 시뮬레이션·충돌 감지 엔진에 그대로 넣기에는 빈틈이 보이기 마련이다. 반면 **메쉬(mesh)**는 정점들을 삼각형(triangle) 면으로 이어 연속적인 닫힌 표면을 구성하기 때문에, 시각적으로 자연스러울 뿐 아니라 로보틱스 시뮬레이터나 CAD 수준의 정밀 기하 처리에도 곧바로 투입할 수 있다.

### 11.2.1 TSDF (Truncated Signed Distance Function)

로봇이 RGB-D 카메라나 LiDAR로 얻은 노이즈 섞인 깊이 영상들을 매끈한 3D 메쉬로 빚어낼 때 가장 널리 쓰이는 중간 표현이 **TSDF(Truncated Signed Distance Function)**다. TSDF는 3D 복셀 격자마다 시선 방향(ray)을 따라 측정한 표면까지의 **부호 있는 거리(signed distance)**를 저장한다. (참고로 여기서는 깊이 영상을 카메라 시선으로 투영해 계산하는 projective TSDF를 다루며, 방향과 무관하게 가장 가까운 표면까지의 실제 유클리드 거리를 저장하는 ESDF는 §11.2.2에서 이어 다룬다.) TSDF 값의 부호를 보면 공간에서의 위치를 직관적으로 알 수 있다.

- **양수($+$)**: 관측된 표면보다 카메라 쪽에 가까운 앞쪽(빈 공간, free space)
- **음수($-$)**: 관측된 표면 너머 뒤쪽(물체 내부 방향의 얇은 truncation band 안)
- **Zero crossing ($0$)**: 양수와 음수가 교차하며 부호가 바뀌는 지점으로, 곧 실제 물체 표면의 위치

여기서 '절단(truncated)'이라는 이름이 붙은 이유가 핵심이다. 물체 표면 앞쪽의 빈 공간 값은 최대 $+\delta$에서 잘라 내고, 표면 뒤쪽으로는 $-\delta$보다 깊이 들어간 물체 내부 영역을 갱신하지 않고, 아래 코드처럼 기본값 상태로 남겨 둔다. 센서 빔은 물체 표면을 뚫고 내부를 직접 볼 수 없기 때문에, 우리가 정밀하게 알고 싶은 실제 표면 경계 근처의 얇은 띠($[-\delta, +\delta]$)에만 해상도와 가중치 갱신을 집중하려는 설계다. 따라서 음수 TSDF 값은 표면 바로 뒤의 얇은 띠에만 국한되며, §11.1의 occupancy grid에서처럼 그 셀이 확실히 막혀 있다고 단정 짓는 값과는 의미가 다르다.

**TSDF 갱신**: 새로운 깊이 프레임이 들어올 때마다 각 시선(ray)을 따라 복셀 값을 누적 가중 평균으로 부드럽게 다듬어 센서 노이즈를 상쇄한다.

$$\text{TSDF}(\mathbf{v}) \leftarrow \frac{W(\mathbf{v}) \cdot \text{TSDF}(\mathbf{v}) + w_{\text{new}} \cdot d_{\text{new}}}{W(\mathbf{v}) + w_{\text{new}}}$$

$$W(\mathbf{v}) \leftarrow \min(W(\mathbf{v}) + w_{\text{new}}, W_{\max})$$

여기서 $d_{\text{new}}$는 현재 프레임에서 계산한 부호 거리, $w_{\text{new}}$는 이번 관측의 가중치, $W(\mathbf{v})$는 지금까지 누적된 가중치다. 부호 거리는 truncation distance $\delta$ 안쪽으로 자르는데(clip), 수식으로 쓰면 다음과 같다.

$$d_{\text{new}} = \text{clip}\left(D(\mathbf{u}) - \lambda^{-1}\|\mathbf{v} - \mathbf{c}\|, -\delta, \delta\right), \quad \lambda = \|\mathbf{K}^{-1}\dot{\mathbf{u}}\|$$

이 식에서 $D(\mathbf{u})$는 픽셀 $\mathbf{u}$에서 읽은 깊이값, $\mathbf{c}$는 카메라 중심 위치, $\mathbf{v}$는 평가하려는 복셀의 중심 좌표다. 카메라에서 얻는 $D(\mathbf{u})$는 시선 광선 길이가 아니라 카메라 광축(z축)을 따라 잰 깊이이므로, 카메라-복셀 간 3D 유클리드 거리 $\|\mathbf{v} - \mathbf{c}\|$를 $\lambda = \|\mathbf{K}^{-1}\dot{\mathbf{u}}\|$로 나누어 광축 방향 깊이로 환산한 뒤 차이를 구한다. (아래 Python 구현에서는 좌표를 먼저 카메라 좌표계로 변환해 z성분 `p_cam[2]`를 직접 뽑아 쓰므로 이 투영 환산이 자연스럽게 반영되어 있다.)

아래 `TSDFVolume` 클래스는 RGB-D 프레임을 순회하며 복셀별 TSDF·색상·가중치를 융합하는 `integrate` 단계와, 8개 꼭짓점의 TSDF 부호 변화(zero crossing)를 탐지해 삼각형 메쉬를 추출하는 Marching Cubes의 골격을 보여 준다.

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

Projective TSDF는 표면을 매끈하게 복원하는 데 뛰어나지만, 장애물에서 멀리 떨어진 빈 공간에서는 거리가 $+\delta$로 잘려 버리기 때문에 경로 계획기가 "지금 로봇이 벽에서 정확히 몇 미터 떨어져 있는가?"를 알기 어렵다. **[Voxblox](https://arxiv.org/abs/1611.03631)**(Oleynikova et al., IROS 2017)는 바로 이 공백을 메우기 위해, TSDF 볼륨으로부터 **ESDF(Euclidean Signed Distance Field)**를 점진적으로 구축해 내는 시스템이다. 소형 비행 로봇(MAV)의 궤적 최적화(trajectory optimization) 기반 로컬 플래너에 장애물까지의 연속적인 거리와 기울기 정보가 필요했기 때문에, 저자들은 이 구조를 고안했다. 또한 단일 CPU 코어만으로도 실시간 동작이 가능함을 보였다. (물론 RRT나 A* 같은 샘플링·탐색 기반 플래너는 점유 여부만 알아도 동작하므로 모든 경로 계획에 ESDF가 필수적인 것은 아니다.)

Voxblox의 파이프라인은 다음 세 단계가 맞물려 돌아간다.

1. **TSDF 통합**: RGB-D 카메라나 깊이 포인트 클라우드를 투영 방식(projective)으로 TSDF 복셀 블록에 통합한다.
2. **Mesh 추출**: Marching Cubes를 적용해 TSDF의 zero crossing에서 삼각형 메쉬를 뽑아낸다. 이때 전체 볼륨을 다시 스캔하지 않고 이번 프레임에 값이 바뀐 복셀 블록만 선별해 갱신한다.
3. **ESDF 계산**: 갱신된 TSDF 표면 경계를 출발점으로 삼아 파면 전파(wavefront propagation) 방식으로 주변 빈 공간에 ESDF 값을 점진적으로 채워 나간다. ESDF는 truncation 한계 없이 각 복셀에서 가장 가까운 장애물 표면까지의 실제 유클리드 거리를 저장한다.

연속적인 궤적 최적화를 돌리는 플래너 입장에서 ESDF의 가치는 압도적이다. 로봇의 후보 경로 위에 놓인 임의의 좌표가 장애물과 얼마나 떨어져 있는지 $O(1)$ 상수 시간에 바로 조회할 수 있을 뿐 아니라, 인접 복셀 값의 차분으로 그래디언트(gradient)를 구하면 장애물로부터 가장 빠르게 멀어지는 밀어내기 방향까지 즉시 얻을 수 있기 때문이다.

### 11.2.3 Poisson Surface Reconstruction

TSDF가 센서 깊이 프레임을 격자에 쏘아 누적하는 온라인 방식이라면, **푸아송 표면 재구성(Poisson surface reconstruction)**은 이미 수집이 끝난 점군(위치와 법선 벡터가 붙은 oriented point cloud)에서 구멍이나 빈틈 없이 닫힌(watertight) 고품질 메쉬를 빚어내는 대표적인 기법이다.

수학적 직관은 아름답다. 어떤 3D 물체의 안쪽을 $1$, 바깥쪽을 $0$으로 정의하는 지시 함수(indicator function) $\chi$를 머릿속에 그려 보자. 이 함수는 물체 내부나 외부의 평탄한 영역에서는 변화가 없으므로 그래디언트가 $0$이고, 오직 물체 표면 경계에서만 바깥($0$)에서 안($1$)으로 급격히 뛰어오르며 안쪽을 향하는 법선(inward normal) 방향을 가리킨다. 따라서 입력 점군의 법선 벡터들로 이 그래디언트를 근사하는 벡터장 $\mathbf{V}$를 구성한 뒤, 양변에 발산(divergence)을 취한 아래 푸아송 방정식을 풀어 지시 함수 $\chi$를 복원하고 적절한 등위면(isosurface)을 추출하면 곧바로 매끈한 메쉬가 된다.

$$\nabla^2 \chi = \nabla \cdot \mathbf{V}$$

여기서 $\mathbf{V}$는 점의 법선으로부터 보간해 만든 벡터장이다. 원 논문에서는 계산 효율을 높이기 위해 이 편미분 방정식을 적응형 옥트리 위에서 푼다.

푸아송 재구성에서는 장면 전체에 걸쳐 하나의 전역적인 암묵적(implicit) 함수를 풀기 때문에 노이즈가 섞인 점군에서도 매우 매끈하고 이어진 표면을 얻는다. 다만 결과 품질이 입력 점의 법선 방향 일관성과 옥트리 깊이, 그리고 점 위치 제약을 얼마나 강하게 거는지를 결정하는 screening 파라미터에 크게 좌우된다. 특히 관측되지 않아 점이 비어 있는 영역까지 억지로 이어 닫아 watertight 표면을 만들려 하기 때문에, 문이 열린 빈 공간이나 천장 너머에 실제로는 없는 얇은 막(artifact membrane)이 생길 수 있다. 따라서 깊이 관측이 계속 들어오는 도중에 지도를 점진적으로 고쳐 나가야 하는 온라인 SLAM 환경에서는 푸아송 재구성 대신 TSDF 기반 메싱을 쓰는 것이 일반적이다.

### 11.2.4 실시간 Mesh 생성

정리하자면, SLAM 파이프라인에서 3D 메쉬를 얻는 실전 전략은 크게 두 갈래로 나뉜다.

1. **Voxblox incremental meshing**: §11.2.2에서 살펴본 대로 관측이 닿아 갱신된 복셀 블록에만 선택적으로 Marching Cubes를 다시 돌리므로, 탐사 도중에도 실시간으로 메쉬가 자라난다.
2. **FAST-LIVO2 colored point map + 후처리**: FAST-LIVO2는 LiDAR 기하와 카메라의 밝기(photometric) 정보로 상태를 갱신한 뒤 색상이 입혀진 정밀 point map을 실시간으로 출력한다. 이 온라인 출력 자체는 점군일 뿐 메쉬가 아니므로, CAD 분석이나 시각화용 메쉬가 필요하다면 주행을 마치고 내보낸 point map의 법선 일관성과 점 밀도를 점검한 뒤 푸아송 재구성이나 Ball Pivoting 알고리즘을 별도의 후처리 단계로 적용한다.

---

## 11.3 Neural / Learned Representations

지금까지 살펴본 복셀, 메쉬, 서펠은 3D 공간의 기하 구조를 이산적인 격자나 기하 요소에 직접 저장하는 **명시적(explicit)** 표현이었다. 반대로 최근에는 미분 가능한 렌더링(differentiable rendering)을 매개로 관측 영상으로부터 장면을 직접 학습해 압축하는 표현들이 주류로 떠올랐다. 이 가운데 순수 **암묵적(implicit)** 방식은 장면 전체를 하나의 신경망 MLP 가중치 속에 녹여 넣는다. **하이브리드** 방식은 다해상도 해시 그리드(hash-grid)나 계층적 격자에 국소 공간 특징(feature)을 심어 두고 가벼운 MLP로 읽어 낸다. 또 미분 가능한 3D Gaussian 같은 명시적 기하 원시 요소(primitive)를 경사 하강법으로 직접 조율하는 방식도 있다.

### 11.3.1 NeRF-SLAM: Neural Implicit + Odometry

이 흐름의 출발점인 **[NeRF](https://arxiv.org/abs/2003.08934)(Neural Radiance Field)**(Mildenhall et al., ECCV 2020)는 3D 공간 좌표와 시선 방향을 입력받아 해당 지점의 색상과 체적 밀도를 출력하는 다층 퍼셉트론(MLP)으로 장면 전체를 연속 함수로 부호화한다.

$$F_\theta: (\mathbf{x}, \mathbf{d}) \to (c, \sigma)$$

여기서 $\mathbf{x} \in \mathbb{R}^3$은 공간상의 3D 좌표, $\mathbf{d} \in \mathbb{S}^2$는 바라보는 시선 단위 벡터, $c$는 RGB 색상, $\sigma$는 광선이 그 지점을 통과할 때 빛이 얼마나 막히고 산란되는지를 뜻하는 체적 밀도(volume density)다.

**볼륨 렌더링 방정식**: 이렇게 신경망으로 표현한 장면에서 가상 카메라의 한 픽셀 색을 그려 내려면, 카메라 중심 $\mathbf{o}$에서 나가는 광선 $\mathbf{r}(t) = \mathbf{o} + t\mathbf{d}$를 따라 적분한다(실제 구현에서는 유한 개수의 점을 샘플링해 합산하는 ray marching을 쓴다).

$$\hat{C}(\mathbf{r}) = \int_{t_n}^{t_f} T(t) \sigma(\mathbf{r}(t)) c(\mathbf{r}(t), \mathbf{d}) \, dt$$

여기서 $T(t) = \exp\left(-\int_{t_n}^{t} \sigma(\mathbf{r}(s)) ds\right)$는 광선이 시작점 $t_n$에서 거리 $t$까지 도달하는 동안 앞선 입자에 가려지지 않고 살아남은 **누적 투과율(accumulated transmittance)**을 뜻한다. 즉, 픽셀 색상은 광선 경로상의 모든 지점이 내뿜는 색에 각 지점의 밀도와 그 앞까지 가려지지 않고 도달할 확률을 곱한 값을 적분해 얻는다. 모든 연산이 미분 가능하므로, 렌더링된 색 $\hat{C}(\mathbf{r})$와 실제 카메라 픽셀 색 사이의 오차를 역전파하면 네트워크 가중치 $\theta$는 물론 카메라 포즈까지 동시에 최적화할 수 있다.

**NeRF-SLAM 파이프라인**: 원래 NeRF는 이미 포즈를 정확히 아는 다량의 사진을 모아 두고 몇 시간씩 오프라인으로 학습하는 기법이었지만, **NeRF-SLAM** 계열 연구들은 이를 두 개의 비동기 루프로 쪼개 온라인 SLAM에 끌어들였다.

1. **Tracking**: 현재까지 학습된 신경망 맵의 가중치 $\theta$를 고정한 채, 가상 렌더링 영상과 실제 관측 영상 사이의 밝기 오차(photometric loss) 및 깊이 오차(depth loss)를 최소화하도록 현재 카메라 포즈를 역전파로 최적화한다.
2. **Mapping**: 추정한 키프레임 포즈를 바탕으로 신경망 가중치 $\theta$를 갱신해 새 공간을 지도 속에 녹여 넣는다. 이때 방금 본 새로운 방을 학습하느라 아까 지나온 거실의 기억(가중치)을 덮어써 망가뜨리지 않도록, 과거 키프레임들을 주기적으로 함께 꺼내 복습(replay)하는 전략이 필수적이다.

이 개념을 처음으로 실시간 SLAM에 구현한 시스템이 **[iMAP](https://arxiv.org/abs/2103.12352)**(Sucar et al., ICCV 2021)이다. iMAP은 핸드헬드 RGB-D 카메라만으로 실시간 SLAM을 돌리면서 장면 전체를 단 하나의 MLP로 표현한 최초의 시스템이다. 키프레임 관리와 멀티프로세싱, 그리고 정보량이 높은 픽셀을 우선 뽑는 능동 샘플링(active sampling)을 결합해 추적 10 Hz, 전역 맵 갱신 2 Hz의 속도를 달성했다.

그러나 단일 MLP 기반 방식은 근본적인 병목을 안고 있었다. 좌표마다 무거운 MLP를 반복 통과해야 해 학습과 매핑 속도가 느릴 뿐 아니라, 새로운 영역을 학습할 때 기존 영역의 재구성 품질이 떨어지는 **치명적 망각(catastrophic forgetting)** 위험도 크다. 또한 방 여러 개로 이루어진 넓은 환경을 작은 MLP 하나에 구겨 넣으면 세부 형상이 뭉개지는 한계가 드러났다.

이 한계를 돌파한 해법이 공간 특징 격자와 가벼운 디코더를 결합한 하이브리드 구조다. 대표적으로 **Instant-NGP**는 여러 해상도의 격자 꼭짓점에 지역 특징(local feature)을 매핑하고 이를 다해상도 해시 테이블(multi-resolution hash-grid encoding)에 저장해 조회함으로써 학습과 렌더링 속도를 수십 배 끌어올렸다. 이 기법을 백엔드로 삼은 고속 NeRF-SLAM 변형들도 뒤따라 등장했다. 같은 문제의식에서 출발한 **[NICE-SLAM](https://arxiv.org/abs/2112.12130)**(Zhu et al., CVPR 2022)은 단일 MLP 대신 계층적 특징 그리드(hierarchical feature grid)에 장면의 다중 스케일 기하 정보를 저장하고, 사전 학습된 기하 디코더로 이를 해석해 최적화한다. 저자들은 단일 MLP 방식이 장면을 지나치게 매끈하게 뭉개고 대규모 공간으로 확장하기 어렵다는 점을 짚어 내며, 계층적 격자 구조로 훨씬 넓고 복잡한 실내 장면에서도 세밀한 복원 품질을 얻었음을 보고했다.

### 11.3.2 3DGS-SLAM: 3D Gaussian Splatting 기반 SLAM

NeRF 계열이 공간을 연속 함수로 암묵화하면서 픽셀마다 광선을 따라 수십~수백 번씩 네트워크를 호출하는 ray marching 비용을 감수해야 했다면, **[3D Gaussian Splatting (3DGS)](https://arxiv.org/abs/2308.04079)**(Kerbl et al. 2023)는 장면을 다시 명시적인 기하 원시 요소들의 집합, 즉 수많은 이방성(anisotropic) **3D Gaussian 타원체**로 돌려놓음으로써 실시간 렌더링의 판도를 바꿨다. 각 3D Gaussian은 다음 네 가지 파라미터로 정의된다.

- 위치 $\boldsymbol{\mu} \in \mathbb{R}^3$: Gaussian의 3D 중심 좌표
- 공분산 $\boldsymbol{\Sigma} \in \mathbb{R}^{3 \times 3}$: 회전(quaternion)과 3축 스케일로 분해·파라미터화되어 타원체의 방향과 납작한 정도를 결정
- 불투명도 $\alpha \in [0, 1)$: 빛을 가리는 정도
- 색상: 시점에 따른 광택 변화까지 표현할 수 있는 구면 조화 함수(spherical harmonics) 계수

렌더링에는 광선을 쏘아 적분하는 ray marching 대신 그래픽스 파이프라인 친화적인 **스플래팅(splatting)**을 쓴다. 타일 기반 래스터화 방식에서는 3D Gaussian 타원체들을 카메라 이미지 평면으로 한꺼번에 투영한 뒤, 깊이 순서대로 정렬해 앞쪽에서 뒤쪽으로 알파 블렌딩(alpha blending)한다. 이 방식 덕분에 원 논문에서는 1080p 고해상도에서도 30 fps를 웃도는 실시간 신규 시점(novel-view) 렌더링 속도를 입증했다.

이 빠른 미분 가능 래스터화를 SLAM에 이식한 시스템이 **[Gaussian Splatting SLAM](https://arxiv.org/abs/2312.06741)**(Matsuki et al., CVPR 2024, 공개 구현명 MonoGS)을 비롯한 3DGS-SLAM 계열이다. 3DGS를 장면 표현으로 쓰는 SLAM 파이프라인은 세 가지 핵심 루프를 돌린다.

1. **Tracking**: 현재 3D Gaussian 맵에서 렌더링한 영상과 실제 카메라 영상 사이의 광도 오차(photometric loss) 및 깊이 기하 오차를 최소화하며 카메라 포즈를 직접 최적화한다.
2. **Mapping**: 새로운 시점에서 드러난 빈 공간이나 디테일이 부족한 영역에 맞춰 Gaussian을 새로 추가(densification)하고 쪼개며(splitting), 불필요한 투명 입자를 제거(pruning)하는 작업을 반복한다.
3. **Loop closure**: 궤적 드리프트가 보정되어 키프레임 포즈가 수정되면 그에 종속된 Gaussian들의 위치와 방향도 함께 조정해야 한다. (다만 초기 Gaussian Splatting SLAM 논문에서는 루프 클로저 모듈을 직접 구현하지 않고 향후 과제로 남겨 두었다.)

NeRF 기반 SLAM과 3DGS 기반 SLAM의 구조적 성격 차이는 아래 표처럼 대조된다.

| 특성 | NeRF-SLAM | 3DGS-SLAM |
|------|-----------|-----------|
| 표현 | MLP 가중치 | 명시적 3D Gaussian 집합 |
| 렌더링 속도 | 느림 (ray marching) | 빠름 (rasterization) |
| 학습 속도 | 느림 | 빠름 |
| 편집 가능성 | 어려움 | 쉬움 (개별 Gaussian 조작) |
| 메모리 | 단일 MLP는 고정 (모델 크기), 공간 feature grid 변형은 장면 규모에 따라 증가 | 가변 (Gaussian 수에 비례) |
| Loop closure 대응 | 어려움 (가중치 변형) | 표현상으로는 상대적으로 쉬움 (Gaussian 변환). 예: [LoopSplat (3DV 2025)](https://arxiv.org/abs/2408.10154)은 3DGS submap끼리 직접 정합해 loop 제약을 만든다 |

표에서 보듯 3DGS-SLAM은 래스터화 기반의 빠른 속도와 명시적 타원체 조작의 유연성 덕분에 루프 클로저 연계나 부분 지도 정합에서 유리한 고지를 점하고 있다. 그렇다고 학습 기반 신경 표현이 언제나 고전적인 TSDF나 서펠 맵을 대체하는 정답인 것은 아니다. Neural representation은 정밀한 기하 복원뿐 아니라 사실적인 신규 시점 렌더링(novel-view rendering)까지 함께 최적화하는 데 특화되어 있다. 하지만 동적 물체가 오가고 조명이 변하는 대규모 환경에서 장시간 운용할 때는 메모리 증가량과 루프 클로저 보정 비용은 물론, 단순 충돌 회피에 정말 포토리얼리스틱 렌더링이 필요한지까지 냉정하게 따져 보아야 한다.

최근 2024~2025년에는 3DGS와 기하 파운데이션 모델을 결합한 고성능 시스템들이 잇달아 발표되며 이 분야가 빠르게 성숙하고 있다.

- **[SplaTAM](https://arxiv.org/abs/2312.02126)**(Keetha et al., CVPR 2024): RGB-D 카메라 한 대로 3D Gaussian 맵을 구축하며 온라인 추적과 매핑을 동시에 수행한다. 현재 시점에서 이미 Gaussian이 채워진 영역을 실루엣 마스크(silhouette mask)로 가려낸 뒤, 아직 비어 있는 미관측 영역에만 새 Gaussian을 추가해 맵을 효율적으로 확장한다. 논문에서는 카메라 포즈 추정, 맵 재구성, 신규 시점 합성 전반에서 기존 신경 표현 방식 대비 최대 2배 향상된 성능을 보고했다.
- **[MonoGS](https://arxiv.org/abs/2312.06741)**(Matsuki et al., CVPR 2024 Highlight): 앞서 소개한 Gaussian Splatting SLAM의 공개 시스템으로, 깊이 센서 없는 순수 단안 카메라 환경에서도 3D Gaussian을 유일한 장면 표현으로 유지하며 3 fps로 동작한다. 단안 점진적 재구성 과정에서 피할 수 없는 깊이 모호성은 기하학적 일관성 검증(geometric verification)과 등방성 정규화(regularization)로 억제하며, 깊이 센서를 연결하면 RGB-D SLAM으로도 곧바로 동작한다.
- **[MASt3R-SLAM](https://arxiv.org/abs/2412.12392)**(Murai et al., CVPR 2025 Highlight): 두 영상 쌍만으로 3D 기하 구조와 조밀한 픽셀 대응을 직접 예측하는 파운데이션 모델 MASt3R를 뼈대로 삼아 처음부터 새롭게 설계한 실시간 단안 dense SLAM 시스템이다. 별도의 신경망 렌더링 최적화 대신 MASt3R가 내놓는 조밀한 포인트맵(dense pointmap)을 곧바로 지도로 삼는다. 카메라 중심이 하나라는 점 외에는 특정 카메라 투영 모델을 가정하지 않는 범용성을 갖추었으며, 루프 클로저 및 2차 전역 최적화까지 포함해 15 FPS의 실시간 속도를 달성했다.

다음 Python 코드는 3D Gaussian Splatting 렌더링의 핵심 원리를 압축해 보여 준다. `Gaussian3D.project_to_2d`는 EWA splatting 기법에 따라 3D 공분산을 핀홀 투영 야코비안 $\mathbf{J}$로 2D 이미지 평면의 타원으로 사상하고, `render_gaussians`는 깊이 순서로 정렬한 타원들을 앞쪽에서 뒤쪽으로 알파 블렌딩해 최종 픽셀 색상을 합성한다.

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

앞선 메트릭 맵, 메쉬, 신경망 표현은 모두 "공간 어디가 막혀 있고 표면이 어떻게 생겼으며 무슨 색을 띠는가"라는 기하와 외관 질문에 답한다. 그러나 로봇이 인간과 같은 공간에서 지능적으로 협력하려면 그 벽이 출입문인지 유리창인지, 앞에 놓인 덩어리가 의자인지 사람인지를 분별할 수 있어야 한다. **시맨틱 맵(semantic map, 의미 맵)**은 3D 기하 구조 위에 사물의 종류·속성·관계라는 의미 레이블을 결합해, 단순한 장애물 회피를 넘어 의미 기반 추론과 임무 계획을 가능하게 한다.

### 11.4.1 Object-Level Maps

**객체 수준 맵(object-level map)**은 공간을 무수한 점이나 복셀 단위로 쪼개 레이블만 붙이는 대신, 환경 속 개별 사물(object)을 하나의 독립된 3D 개체로 인식해 위치·크기·범주를 지도에 올리는 방식이다. 구축 파이프라인은 보통 네 단계를 거친다.

1. **2D detection/segmentation**: YOLO, Mask R-CNN, SAM 같은 2D 시각 인식 모델로 카메라 영상에서 개별 물체의 경계 상자나 픽셀 마스크를 뽑아낸다.
2. **3D lifting**: RGB-D 깊이 영상(또는 LiDAR 투영 깊이)과 카메라 포즈를 이용해 2D 마스크 안쪽의 픽셀들을 3D 월드 좌표계로 역투영(backprojection)한다.
3. **Data association**: 로봇이 이동하며 여러 프레임에 걸쳐 같은 물체를 다시 보았을 때, 기하학적 거리와 클래스 일치 여부를 따져 동일한 객체 ID로 묶어 낸다(tracking 및 re-identification).
4. **Map integration**: 여러 관측에서 모인 3D 점군을 통합해 물체의 중심 좌표, 3D 바운딩 박스, 혹은 정밀 형상 모델로 지도를 갱신한다.

아래 `ObjectMap` 클래스는 기본 데이터 연관 파이프라인을 구현한 예시다. 2D 세그멘테이션 마스크를 3D로 역투영해 중심점과 바운딩 박스를 계산한 뒤, 같은 클래스이면서 중심 거리가 가까운 기존 객체와 매칭해 위치를 점진적으로 보정한다.

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

개별 객체의 위치만 나열한 object-level map에서 한 걸음 더 나아가, 물체와 방, 복도, 건물 전체가 맺는 위계와 공간적 관계를 그래프 계층으로 엮어 낸 표현이 **3D Scene Graph**다. 이를 활용하면 로봇은 단순 좌표를 넘어 "거실에 테이블이 있고, 그 테이블 위에 컵이 놓여 있다"는 식의 구조화된 환경 이해를 바탕으로 고차원 경로 탐색이나 작업 계획을 세울 수 있다.

대표적인 실시간 시스템인 **[Hydra](https://arxiv.org/abs/2201.13360)**(Hughes et al., RSS 2022)는 로봇이 환경을 탐사하는 동안 센서 스트림으로부터 다섯 층의 계층적 3D scene graph를 실시간으로 쌓아 올린다. 특히 루프 클로저가 발생하면 embedded deformation graph로 최하단 메쉬부터 상위의 객체·방 노드까지 한꺼번에 구부려 정합을 맞춘다. 논문에서는 이를 루프 클로저에 맞춰 3D scene graph 전체를 일관되게 최적화하는 최초의 알고리즘으로 소개한다. Hydra가 구축하는 5단계 계층 구조는 다음 표와 같다.

| Layer | 내용 | 구성 방법 |
|-------|------|-----------|
| 1 | Metric-Semantic 3D Mesh | TSDF 통합 + semantic segmentation |
| 2 | Objects & Agents | object: 의미 레이블별 mesh 정점의 Euclidean clustering (중심·bounding box) / agent: 로봇 궤적의 pose graph |
| 3 | Places (topological) | GVD(Generalized Voronoi Diagram)에서 추출 |
| 4 | Rooms | place 그래프 dilation + 커뮤니티 검출 |
| 5 | Buildings | 건물 노드 하나를 모든 방에 연결 (논문에서는 건물이 하나라고 가정) |

이 가운데 위상 공간(Places)과 의미 공간(Rooms)을 기하 정보로부터 자동으로 뽑아내는 과정이 특히 정교하다.

**Places 계층 만들기**:
1. 먼저 TSDF 볼륨으로부터 전역 장애물 거리장인 ESDF를 계산한다.
2. ESDF를 갱신하는 과정에서 **GVD(Generalized Voronoi Diagram)**를 동시에 얻는다. GVD는 가장 가까운 장애물 두 개 이상으로부터 같은 거리에 놓인 복셀들의 집합으로, 장애물 사이 통로의 정중앙을 가로지르는 환경의 뼈대(skeleton)를 이룬다. Hydra는 네 개 이상의 장애물에서 같은 거리에 있는 GVD 복셀과 모서리(corner) 형상 템플릿에 부합하는 복셀을 추려 `place` 노드로 삼는다.
3. 이어서 flood-fill 탐색으로 각 GVD 복셀에 가장 가까운 place 노드를 표시해 인접한 노드 사이를 엣지 후보로 잇는다. 이때 직선 엣지가 GVD 뼈대에서 너무 멀어져 장애물에 닿을 위험이 있으면 그 중간 지점에 새 노드를 추가해 엣지를 분할한다.
4. 이렇게 만들어진 place 노드는 충돌 없는 안전한 빈 공간을 뜻하며, 엣지는 두 지점 사이를 직선으로 안전하게 건너갈 수 있음을 보장한다. 저자들은 이 부분 그래프가 사실상 주행 가능한 위상 지도(topological map) 역할을 한다고 설명한다.

**Room detection**: 흩어진 place 노드들을 의미 있는 '방(room)' 단위로 묶는 절차는 문(doorway)처럼 좁은 병목 지점을 찾아 끊어 내는 원리를 따른다.
1. 모든 place 노드는 ESDF에서 물려받은, 장애물까지의 여유 거리(clearance) 정보를 담고 있다.
2. 방과 방을 잇는 문지방이나 좁은 통로에서는 이 clearance 값이 눈에 띄게 작아진다.
3. 이를 이용해 장애물 경계를 단계적으로 부풀려 가며(dilation), 현재 팽창 반경보다 clearance가 작은 place 노드를 임시로 지워 좁은 연결 고리를 끊어 본다. 논문 예시에서는 $0.45\text{ m}$에서 $1.2\text{ m}$ 사이를 등간격 10단계로 나눈다. 각 단계마다 쪼개진 그래프의 연결 성분(connected component) 개수를 센 뒤, 연결 성분 수 분포에서 중앙값(median)에 해당하는 단계의 그래프 중 가장 큰 것을 선택하면 그 연결 성분들이 곧 각각의 '방 씨앗'이 된다. 이 씨앗들을 초기 커뮤니티로 삼아 탐욕적 모듈러리티(greedy modularity) 커뮤니티 검출 알고리즘을 돌려, 앞서 지워져 빠졌던 place 노드들을 가장 어울리는 방에 마저 배정한다.

**계층적 loop closure**: Hydra는 이 계층 구조를 루프 클로저 탐지와 검증에도 영리하게 재활용한다. 최하단의 agent 계층에는 DBoW2 외관 디스크립터를 두고, 그 위 계층에는 주변 object와 place의 분포 통계를 압축한 그래프 디스크립터를 붙여 후보를 검색한다. 후보를 찾으면 아래 계층부터 단계적으로 기하 정합을 시도한다. 먼저 시각 특징점을 RANSAC으로 맞춰 보고, 조명 변화 등으로 시각 정합이 실패하면 주변 물체 노드들의 배치를 TEASER++로 정합해 본다. 두 가지 모두 실패한 후보만 버린다. 논문에서는 이러한 계층적 전략이 기존의 BoW나 시각 특징 매칭 단독 방식보다 루프를 더 많이, 그리고 더 높은 정확도로 검출했다고 보고한다.

비슷한 계보의 **S-Graphs**(Situational Graphs) 역시 다층 계층적 scene graph를 구축하지만, 설계 철학에서 한 가지 결정적인 차이를 보인다. Hydra가 포즈 그래프와 변형 그래프를 거치며 계층을 맞추는 반면, S-Graphs는 방(room), 벽(wall), 바닥(floor) 같은 고수준 구조 요소들을 아예 **팩터 그래프 최적화의 직접적인 상태 변수**로 편입시켜 로봇 궤적과 구조 기하를 처음부터 함께(jointly) 최적화한다.

### 11.4.3 Open-Vocabulary Semantic Mapping

전통적인 semantic mapping은 학습 데이터셋에 미리 정의된 고정 어휘표(closed-set, 예: COCO의 80개 클래스) 밖의 사물을 전혀 알아보지 못한다. 반면 **Open-vocabulary semantic mapping**은 시각-언어 파운데이션 모델을 지도 표현에 접목해, 사전에 목록으로 등록해 두지 않은 임의의 자연어 텍스트로도 3D 공간을 검색할 수 있게 해 준다.

핵심 메커니즘은 간단하면서도 강력하다. 카메라 이미지 전체 혹은 객체 마스크·패치 영역에서 CLIP image encoder로 고차원 시각 임베딩 벡터를 추출한 뒤, 이를 깊이 정보와 포즈로 대응되는 3D 포인트나 복셀에 달아 놓는다. 나중에 사용자가 "빨간 소화기"나 "노트북 옆 커피잔" 같은 문장을 입력하면, CLIP text encoder가 그 문장을 동일한 임베딩 공간의 벡터로 변환한 뒤, 이 벡터를 지도에 저장된 모든 3D 특징 벡터와 코사인 유사도로 비교해 가장 잘 일치하는 3D 위치를 즉각 찾아낸다. (이 구조에서는 CLIP image encoder가 자연어 텍스트와 같은 임베딩 공간을 직접 공유하며, 만약 DINO 특징을 함께 활용한다면 이는 텍스트 검색용이 아니라 정밀한 객체 분할이나 프레임 간 특징점 대응을 돕는 보조 역할을 한다.)

미리 라벨링할 수 없는 온갖 물건이 널려 있는 가정 환경의 서비스 로봇이나, 예측 불가능한 재난 현장을 누비는 탐사 로봇은 이처럼 열린 어휘로 공간을 질의할 수 있는 능력 덕분에 결정적인 유연성을 얻는다.

아래 `OpenVocabSemanticMap` 클래스는 open-vocabulary 검색의 기본 원리를 구현한 코드다. 3D 좌표마다 CLIP 시각 특징을 누적 저장해 두었다가, 임의의 텍스트 임베딩 질의가 들어오면 L2 정규화된 코사인 유사도로 상위 일치 지점이나 공간 전체의 관련도 히트맵을 산출한다.

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

현실 세계의 공간은 한 번 지도를 그려 두고 끝낼 수 있는 정적인 박물관이 아니다. 복도에는 사람이 끊임없이 오가고, 회의실 의자와 상자의 배치는 매일 바뀌며, 야외 도로는 계절과 날씨에 따라 모습이 달라진다. 로봇이 하루 이틀이 아니라 몇 주·몇 달 이상 현장에 머물며 임무를 수행하려면, 시간이 지남에 따라 지도가 스스로 낡은 정보를 비우고 새로운 변화를 받아들이며 움직이는 방해물을 걸러 낼 줄 알아야 한다.

### 11.5.1 변화 감지 (Change Detection)

장기 맵 유지의 첫걸음은 과거에 저장해 둔 지도와 방금 들어온 실시간 센서 관측을 대조해 "어디가 어떻게 달라졌는가"를 포착하는 **변화 감지(change detection)**다.

**Geometric change detection**은 기존 지도에서 예측되는 기하 관측과 현재 스캔을 비교해 두 가지 상반된 이벤트를 찾아낸다.

1. **Free-to-occupied**: 기존 지도에서는 분명 비어 있던(free) 공간에 현재 센서 스캔이 닿아 장애물이 관측되었다면, 없던 물체가 새로 나타났거나 옮겨 왔음을 뜻한다.
2. **Occupied-to-free**: 기존 지도에서 장애물로 막혀 있다고(occupied) 기록된 영역을 현재 센서 광선이 막힘없이 통과(pass-through)해 더 먼 뒤쪽을 찍었다면, 그 자리에 있던 물체가 치워져 사라졌음을 뜻한다.

아래 `detect_changes` 함수에는 기하학적 변화 감지 로직이 구현되어 있다. 점유 확률 지도에서 현재 스캔의 끝점과 광선 경로상의 셀을 차례로 조회해 새로 등장한 장애물과 사라진 장애물을 분리해 낸다.

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

기하학적 유무 변화뿐 아니라 의미 변화를 함께 짚어 내는 **Semantic change detection**도 중요하다. 의미 변화 감지에서는 물체의 위치가 몇 센티미터 바뀌었거나, 닫혀 있던 문이 열렸거나, 의자가 다른 종류의 가구로 교체된 경우처럼 3D 형상 차이와 의미 레이블·상태 변화를 아울러 포착한다.

### 11.5.2 Map Maintenance: 유지 전략

수 주 이상 장기간 운용할 때 들어오는 모든 스캔과 관측을 무한정 쌓아 둘 수는 없다. 제한된 메모리와 연산량 안에서 지도를 최신 상태로 유지하려면 무엇을 남기고 무엇을 잊을지 정하는 정교한 생애주기 전략이 필요하다.

- **Recency weighting**: 가장 최근에 들어온 관측에 높은 가중치를 부여하고, 오래 확인되지 않은 과거 관측의 영향력은 시간에 따라 서서히 감쇠(decay)시킨다. 예를 들어 TSDF 복셀의 누적 가중치 $W(\mathbf{v})$에 시간 감쇠 계수를 곱해 두면, 환경이 바뀌었을 때 새 관측이 낡은 표면 기록을 자연스럽게 덮어쓸 수 있다.
- **Semi-static classification**: 모든 공간 요소를 똑같이 취급하지 않고 시간적 지속성에 따라 층위를 나눈다. 벽·바닥·기둥 같은 **정적(static)** 골격은 영구 보존하고, 의자·택배 상자·주차된 차량처럼 가끔 자리가 바뀌는 **준정적(semi-static)** 요소는 최근 관측 시각을 기록해 두었다가 주기적으로 갱신·정리하며, 보행자나 달리는 차량 같은 **동적(dynamic)** 객체는 정적 지도에서 아예 배제한다.
- **Multi-experience mapping**: 시간대별 조명 변화나 계절 변화처럼 한 번 바뀐 뒤 되돌아오기도 하는 환경에서는 하나의 지도에 모든 변화를 뒤섞어 넣기보다, 같은 장소를 지나며 기록한 여러 시점의 경험(experience)을 별도 계층이나 하위 지도로 보관해 둔다. 측위할 때는 현재 센서 관측과 가장 잘 들어맞는 경험을 골라 활용한다.

### 11.5.3 Dynamic Object 처리

보행자나 주행 중인 차량처럼 프레임 사이사이를 가로지르는 동적 물체는 SLAM 프런트엔드와 맵 양쪽에 심각한 독이 된다.

1. **Tracking 오류**: 움직이는 트럭이나 사람에서 뽑힌 특징점을 정지해 있는 배경의 특징점으로 착각하고 오도메트리를 풀면, 로봇 자신이 반대로 움직였다고 오판해 포즈 추정이 크게 틀어진다.
2. **맵 오염**: 방을 가로질러 걸어간 사람의 잔상이 지도에 그대로 굳어 남으면(ghosting effect), 실제로는 텅 빈 복도에 거대한 가짜 장애물 벽이 서 있는 것처럼 그려져 이후 경로 계획을 가로막는다.

이러한 동적 간섭을 걷어 내고 깨끗한 정적 지도를 유지하기 위해 다음 네 가지 기법이 상호 보완적으로 쓰인다.

1. **Semantic filtering**: 이미지·포인트 클라우드 세그멘테이션 모델로 '사람', '자동차', '자전거'처럼 본래 움직일 수 있는 클래스를 식별해, 그 영역에서 뽑힌 특징점이나 스캔을 SLAM 입력 단에서 아예 걸러 낸다. 다만 주차된 차량처럼 잠재적 동적 클래스라도 실제로는 멈춰 서 있어 측위에 도움이 되는 경우까지 모두 버리게 된다는 점은 감안해야 한다.
2. **Geometric consistency check**: 로봇 자신의 이동량(ego-motion)을 감안하더라도 새 관측 위치가 기하학적으로 설명되지 않는 점을 동적 이상치로 골라낸다. 다음 프레임에서 사라진 점이라고 해서 모두 움직인 것은 아니다. 시야 밖으로 나갔거나 정적 벽에 가려졌거나 매칭에 실패했을 수도 있기 때문이다. 따라서 가시성 추론(visibility reasoning)으로 해당 점이 현재 시야에 보여야 마땅한 위치에 있는지 따지고, 재투영 잔차도 함께 검사해 실제로 동적인지 판정한다.
3. **Background subtraction**: TSDF나 점유 격자에서 짧은 시간 안에 '비어 있음 → 점유됨 → 다시 비어 있음(free-to-occupied-to-free)' 상태 전이를 거치는 복셀을 찾아내 일시적으로 스쳐 지나간 동적 물체로 판별하고 제거한다.
4. **SLAM with dynamic object tracking**: 동적 물체를 지워 버리기만 하지 않고, 그 물체의 6자유도 포즈와 속도를 별도의 상태 변수로 두어 정적 환경 지도를 추정하는 동시에 움직이는 물체의 궤적까지 추적한다(SLAM + MOT, multi-object tracking).

아래 `DynamicMapManager` 클래스에는 관측된 객체를 의미 클래스에 따라 분기해 처리하는 구조가 간결하게 담겨 있다. 관리 방식은 영구 보존(`static_map`), 시간 만료 기반 주기적 갱신(`semistatic_map`), 지도 편입 배제 및 별도 궤적 추적(`dynamic_objects`)으로 나뉜다.

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

하나의 거대한 로봇 시스템에서는 지금까지 살펴본 여러 공간 표현을 목적에 따라 겹쳐 함께 운용하는 것이 일반적이다. 예를 들어 경로 계획 모듈에는 고속 충돌 검사를 지원하는 OctoMap이나 ESDF를 물려주고, 시각화·관제 화면에는 텍스처가 살아 있는 서펠이나 3D Gaussian 맵을 띄우며, 고수준 임무 명령을 해석할 때는 3D Scene Graph와 open-vocabulary 계층에 질의하는 식이다.

이제 센서 모델링과 캘리브레이션부터 상태 추정, 오도메트리, 장소 인식, 루프 클로저, 다층 공간 표현에 이르는 모든 핵심 부품을 손에 쥐었다. 다음 장에서는 이 조각들이 자율주행 자동차, 드론, 핸드헬드 스캐너 같은 실제 하드웨어 플랫폼에서 어떻게 하나의 거대한 실전 시스템으로 엮이고, 표준 벤치마크에서 어떤 지표로 검증되는지 살펴보자(→ 12장).
