# Ch.7 — LiDAR Odometry & LiDAR-Inertial Odometry

Ch.6에서 카메라(+IMU) 기반의 Visual Odometry를 다루었다. 이 챕터에서는 LiDAR 기반으로 동일한 자기 운동 추정 문제에 접근한다.

LiDAR는 카메라와 상보적인 센서다. 카메라가 풍부한 텍스처 정보를 제공하지만 조명에 민감하고 절대 거리를 알 수 없는 반면, LiDAR는 조명 불변의 정밀한 3D 거리 측정을 제공한다. 이 챕터에서는 LiDAR로부터 자기 운동을 추정하는 LiDAR Odometry(LO)와, IMU를 결합한 LiDAR-Inertial Odometry(LIO)의 내부 구조를 심도 있게 다룬다.

LiDAR 오도메트리의 핵심 문제는 **포인트 클라우드 정합(registration)**이다 — 연속된 두 스캔 사이의 강체 변환 $\mathbf{T} \in SE(3)$를 찾는 것이다. 이 단순해 보이는 문제가 실제로는 데이터 연관(correspondence), 노이즈 모델, 계산 효율, 모션 왜곡 보정 등 다양한 도전을 포함한다.

---

## 7.1 Point Cloud Registration 기초

포인트 클라우드 정합은 두 점군 $\mathcal{P} = \{\mathbf{p}_i\}$와 $\mathcal{Q} = \{\mathbf{q}_j\}$ 사이의 최적 강체 변환을 찾는 문제다:

$$\mathbf{T}^* = \underset{\mathbf{T} \in SE(3)}{\arg\min} \sum_i d(\mathbf{T} \cdot \mathbf{p}_i, \mathcal{Q})$$

여기서 $d(\cdot, \cdot)$는 변환된 소스 점과 타겟 점군 사이의 거리 메트릭이다. 이 거리의 정의에 따라 다양한 ICP 변종이 나뉜다.

### 7.1.1 ICP 변종들

**Point-to-Point ICP ([Besl & McKay, 1992](https://doi.org/10.1109/34.121791))**

가장 기본적인 형태로, 변환된 소스 점과 가장 가까운 타겟 점 사이의 유클리드 거리를 최소화한다:

$$\mathbf{T}^* = \underset{\mathbf{T}}{\arg\min} \sum_i \left\|\mathbf{T} \cdot \mathbf{p}_i - \mathbf{q}_{c(i)}\right\|^2$$

여기서 $c(i) = \arg\min_j \|\mathbf{T} \cdot \mathbf{p}_i - \mathbf{q}_j\|$는 최근접점(closest point) 대응이다. ICP는 두 단계를 반복한다:

1. **대응 찾기**: 현재 변환으로 소스 점을 변환한 뒤, 타겟에서 최근접점을 찾는다. kd-tree를 사용하면 $O(N\log N)$이다.

2. **변환 추정**: 대응 쌍이 주어지면, 최적 변환은 closed-form으로 구할 수 있다. SVD를 이용한 방법:
   
   양 점군의 중심을 뺀다:
   $$\bar{\mathbf{p}} = \frac{1}{N}\sum_i \mathbf{p}_i, \quad \bar{\mathbf{q}} = \frac{1}{N}\sum_i \mathbf{q}_{c(i)}$$
   
   교차 공분산 행렬을 계산한다:
   $$\mathbf{W} = \sum_i (\mathbf{p}_i - \bar{\mathbf{p}})(\mathbf{q}_{c(i)} - \bar{\mathbf{q}})^T$$
   
   SVD 분해: $\mathbf{W} = \mathbf{U}\boldsymbol{\Sigma}\mathbf{V}^T$
   
   최적 회전과 이동:
   $$\mathbf{R}^* = \mathbf{V} \text{diag}(1, 1, \det(\mathbf{V}\mathbf{U}^T)) \mathbf{U}^T, \quad \mathbf{t}^* = \bar{\mathbf{q}} - \mathbf{R}^*\bar{\mathbf{p}}$$
   
   $\det(\mathbf{V}\mathbf{U}^T) = 1$이면 $\mathbf{R}^* = \mathbf{V}\mathbf{U}^T$이고, $\det(\mathbf{V}\mathbf{U}^T) = -1$이면 반사(reflection)를 방지하기 위해 $\mathbf{V}$의 마지막 열 부호를 뒤집는다.

Point-to-point ICP의 한계:
- 평면에서의 슬라이딩 — 평면 위의 점들은 평면을 따라 미끄러져도 비용이 변하지 않아, 수렴이 느리다.
- 초기값 의존성 — 로컬 최소값에 빠지기 쉽다.
- 최근접점 대응의 부정확함 — 두 스캔의 샘플링 패턴이 다르면 진정한 대응이 아닐 수 있다.

**Point-to-Plane ICP**

평면 위의 점에 대해서는, 점 사이 거리보다 점에서 평면까지의 거리가 더 물리적으로 의미 있다:

$$\mathbf{T}^* = \underset{\mathbf{T}}{\arg\min} \sum_i \left((\mathbf{T} \cdot \mathbf{p}_i - \mathbf{q}_{c(i)})^T \mathbf{n}_{c(i)}\right)^2$$

여기서 $\mathbf{n}_{c(i)}$는 타겟 점 $\mathbf{q}_{c(i)}$에서의 표면 법선(surface normal)이다. 점과 점 사이 거리가 아니라 점에서 법선 방향으로의 거리만 측정하므로, 평면을 따른 슬라이딩은 비용에 기여하지 않는다.

장점: Point-to-point 대비 수렴 속도가 훨씬 빠르다. 특히 평면이 많은 실내/도시 환경에서 효과적이다.

단점: closed-form 해가 없어 반복 최적화(Gauss-Newton 등)가 필요하다. 법선 추정의 정확도에 의존한다.

법선 추정은 각 점의 이웃점들에 대해 PCA(주성분 분석)를 수행하여 가장 작은 고유값에 대응하는 고유벡터를 법선으로 취한다. 이웃의 공분산 행렬:

$$\mathbf{C} = \frac{1}{k}\sum_{j \in \mathcal{N}(i)} (\mathbf{q}_j - \bar{\mathbf{q}})(\mathbf{q}_j - \bar{\mathbf{q}})^T$$

고유값 분해 $\mathbf{C} = \mathbf{V}\boldsymbol{\Lambda}\mathbf{V}^T$에서 $\lambda_{\min}$에 대응하는 고유벡터가 법선 방향이다.

### 7.1.2 GICP (Generalized ICP)

GICP ([Segal et al., 2009](https://doi.org/10.15607/RSS.2009.V.021))는 point-to-point, point-to-plane, plane-to-plane ICP를 하나의 확률적 프레임워크로 통합한다.

핵심 아이디어: 각 점이 국소 표면의 불확실성을 반영하는 공분산 $\mathbf{C}_i$를 가진다고 모델링한다. 비용 함수는:

$$\mathbf{T}^* = \underset{\mathbf{T}}{\arg\min} \sum_i (\mathbf{T} \cdot \mathbf{p}_i - \mathbf{q}_{c(i)})^T (\mathbf{C}_i^{\mathcal{Q}} + \mathbf{R}\mathbf{C}_i^{\mathcal{P}}\mathbf{R}^T)^{-1} (\mathbf{T} \cdot \mathbf{p}_i - \mathbf{q}_{c(i)})$$

여기서 $\mathbf{C}_i^{\mathcal{P}}, \mathbf{C}_i^{\mathcal{Q}}$는 각각 소스와 타겟 점의 국소 표면 공분산이다.

**공분산의 물리적 의미**:
- 평면 위의 점: 법선 방향으로 작은 분산, 접선 방향으로 큰 분산 → $\mathbf{C} = \mathbf{R}_s \text{diag}(\epsilon, 1, 1) \mathbf{R}_s^T$ ($\epsilon \ll 1$, $\mathbf{R}_s$는 법선을 첫 축에 정렬하는 회전)
- 이 경우 GICP는 자동으로 plane-to-plane 정합이 된다.
- $\mathbf{C}^{\mathcal{P}} = \mathbf{0}$이면 point-to-plane, $\mathbf{C}^{\mathcal{P}} = \mathbf{C}^{\mathcal{Q}} = \mathbf{I}$이면 point-to-point가 된다.

GICP는 이론적으로 가장 일반적인 ICP 프레임워크이며, 실제로도 다양한 환경에서 가장 정확한 결과를 보인다.

```python
# GICP 핵심 반복 수도코드
def gicp(P, Q, T_init, max_iter=50, tol=1e-6):
    """
    P: source point cloud (N x 3)
    Q: target point cloud (M x 3)
    T_init: initial transformation (4 x 4)
    """
    T = T_init.copy()
    
    # 각 점의 국소 표면 공분산 사전 계산
    C_P = compute_local_covariances(P, k_neighbors=20)  # N x 3 x 3
    C_Q = compute_local_covariances(Q, k_neighbors=20)  # M x 3 x 3
    
    # Target kd-tree 구축
    tree = KDTree(Q)
    
    for iteration in range(max_iter):
        # 1. 소스 점 변환
        P_transformed = apply_transform(T, P)
        
        # 2. 최근접점 대응 찾기
        distances, indices = tree.query(P_transformed)
        
        # 3. Gauss-Newton 업데이트
        H = np.zeros((6, 6))  # Hessian approximation
        b = np.zeros(6)       # gradient
        
        for i in range(len(P)):
            j = indices[i]
            residual = P_transformed[i] - Q[j]
            
            # 결합 공분산 (변환된 좌표계)
            R = T[:3, :3]
            Sigma = C_Q[j] + R @ C_P[i] @ R.T
            Sigma_inv = np.linalg.inv(Sigma)
            
            # SE(3) 자코비안
            J = compute_se3_jacobian(T, P[i])  # 2D: 3 x 6
            
            # 누적
            H += J.T @ Sigma_inv @ J
            b += J.T @ Sigma_inv @ residual
        
        # 4. 증분 계산 및 적용
        xi = np.linalg.solve(H, -b)
        T = se3_exp(xi) @ T
        
        if np.linalg.norm(xi) < tol:
            break
    
    return T
```

### 7.1.3 NDT (Normal Distributions Transform)

[NDT (Biber & Strasser, 2003)](https://doi.org/10.1109/IROS.2003.1249285)는 점군을 직접 사용하는 대신, 공간을 복셀(voxel)로 나누고 각 복셀 내 점 분포를 가우시안으로 모델링한다.

**NDT 절차**:

1. **타겟 점군의 NDT 표현 구축**: 공간을 3D 복셀 격자로 나누고, 각 복셀 $k$ 내의 점들로부터 가우시안 $\mathcal{N}(\boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k)$를 계산한다:
   $$\boldsymbol{\mu}_k = \frac{1}{n_k}\sum_{i \in k} \mathbf{q}_i, \quad \boldsymbol{\Sigma}_k = \frac{1}{n_k-1}\sum_{i \in k} (\mathbf{q}_i - \boldsymbol{\mu}_k)(\mathbf{q}_i - \boldsymbol{\mu}_k)^T$$

2. **변환 최적화**: 변환된 소스 점이 타겟 NDT 분포에서 높은 가능도(likelihood)를 가지도록 최적화:
   $$\mathbf{T}^* = \underset{\mathbf{T}}{\arg\min} \sum_i -\log p(\mathbf{T} \cdot \mathbf{p}_i \mid \boldsymbol{\mu}_{k(i)}, \boldsymbol{\Sigma}_{k(i)})$$
   
   가우시안 가정 하에:
   $$\mathbf{T}^* = \underset{\mathbf{T}}{\arg\min} \sum_i (\mathbf{T} \cdot \mathbf{p}_i - \boldsymbol{\mu}_{k(i)})^T \boldsymbol{\Sigma}_{k(i)}^{-1} (\mathbf{T} \cdot \mathbf{p}_i - \boldsymbol{\mu}_{k(i)})$$

NDT의 장점:
- 명시적 대응 찾기가 불필요 — 점이 어느 복셀에 속하는지만 판단하면 된다. kd-tree 구축 비용이 없다.
- 복셀 크기로 정밀도와 수렴 영역을 조절할 수 있다 — 큰 복셀은 넓은 수렴 영역, 작은 복셀은 높은 정밀도.
- 비용 함수가 매끄러워 최적화가 안정적이다.

단점:
- 복셀 크기 선택에 민감하다.
- 점이 적은 복셀에서 공분산 추정이 불안정하다.
- 2D NDT는 자율주행에서 많이 쓰이지만, 3D NDT는 ICP/GICP 대비 정확도가 약간 떨어지는 경향이 있다.

### 7.1.4 수렴성과 초기값 의존성

모든 정합 알고리즘은 로컬 최적화이므로, 초기값이 중요하다. 초기 포즈 오차가 크면 잘못된 로컬 미니멈에 수렴한다.

실전에서의 초기값 제공 방법:
1. **등속 모델**: 이전 두 프레임의 변환을 외삽. 가장 간단하지만, 급격한 모션에서 실패.
2. **IMU 적분**: 짧은 시간 동안의 IMU 적분이 좋은 초기값을 제공. LIO 시스템이 LO보다 강건한 이유 중 하나.
3. **멀티 해상도 (Multi-resolution)**: coarse-to-fine 접근. 먼저 큰 복셀/다운샘플링으로 대략적 정합 후, 세밀한 정합으로 정제.
4. **RANSAC + 특징 매칭**: FPFH, SHOT 등의 3D 디스크립터로 대응을 찾고 RANSAC으로 초기 변환 추정. 일반적 정합에서는 유용하지만, odometry에서는 시간적 연속성으로 인해 등속 모델 + IMU가 더 실용적.

---

## 7.2 Feature-based LiDAR Odometry

### 7.2.1 LOAM (Lidar Odometry and Mapping in Real-time)

LOAM ([Zhang & Singh, 2014](https://doi.org/10.15607/RSS.2014.X.007))은 LiDAR 오도메트리의 기준점이다. KITTI 오도메트리 벤치마크에서 오랫동안 상위권을 유지했으며, 이후 LeGO-LOAM, LIO-SAM, A-LOAM 등 수많은 후속 시스템의 기반이 되었다.

LOAM의 핵심 통찰은 두 가지다:
1. LiDAR 스캔에서 기하학적 특징(edge, planar)을 추출하면, 전체 점군 대비 훨씬 적은 점으로 정확한 정합이 가능하다.
2. 빠른 odometry와 느린 mapping을 분리하면, 실시간성과 정확도를 동시에 달성할 수 있다.

**특징점 추출**

각 스캔 라인(scan line)에서 점의 국소 곡률(curvature)을 계산한다. 점 $\mathbf{p}_i$의 곡률:

$$c_i = \frac{1}{|\mathcal{S}_i| \cdot \|\mathbf{p}_i\|} \left\| \sum_{j \in \mathcal{S}_i} (\mathbf{p}_j - \mathbf{p}_i) \right\|$$

여기서 $\mathcal{S}_i$는 같은 스캔 라인에서 $\mathbf{p}_i$의 좌우 이웃점(보통 5개씩) 집합이고, $\|\mathbf{p}_i\|$는 센서로부터의 거리(range)로 정규화하여 가까운 점과 먼 점의 곡률을 비교 가능하게 한다.

- **Edge feature**: 곡률이 높은 점 ($c_i > c_{\text{thresh}}^e$). 물리적으로 모서리, 기둥 등 날카로운 경계에 해당.
- **Planar feature**: 곡률이 낮은 점 ($c_i < c_{\text{thresh}}^p$). 물리적으로 벽, 바닥 같은 평탄한 표면에 해당.

특징점 선택 시 추가 규칙:
- 각 스캔 라인을 4개 구간으로 나누어 균등 분포를 보장한다.
- 이웃 점에 이미 선택된 점이 있으면 제외(non-maximum suppression).
- 거의 수평인 표면이나 가림(occlusion) 경계의 점은 불안정하므로 제외한다.

**Odometry 모듈 (~10Hz)**

Scan-to-scan 매칭으로 빠른 모션 추정을 수행한다. 현재 스캔의 특징점을 이전 스캔의 특징점과 대응시키되, 거리 메트릭이 특징 유형에 따라 다르다:

**Edge point-to-edge distance**: 현재 스캔의 edge 점 $\mathbf{p}$에 대해, 이전 스캔에서 가장 가까운 edge 점 두 개 $\mathbf{a}, \mathbf{b}$를 찾는다. $\mathbf{p}$에서 직선 $\overline{\mathbf{ab}}$까지의 거리:

$$d_e = \frac{\|(\mathbf{p}-\mathbf{a}) \times (\mathbf{p}-\mathbf{b})\|}{\|\mathbf{a}-\mathbf{b}\|}$$

이것은 두 벡터의 외적 크기를 밑변 길이로 나눈 것으로, 삼각형의 높이(= 점에서 직선까지의 거리)와 같다.

**Planar point-to-plane distance**: 현재 스캔의 planar 점 $\mathbf{p}$에 대해, 이전 스캔에서 가장 가까운 planar 점 세 개 $\mathbf{a}, \mathbf{b}, \mathbf{c}$를 찾는다. $\mathbf{p}$에서 평면 $\triangle\mathbf{abc}$까지의 거리:

$$d_p = \frac{(\mathbf{p}-\mathbf{a})^T \left((\mathbf{a}-\mathbf{b}) \times (\mathbf{a}-\mathbf{c})\right)}{\|(\mathbf{a}-\mathbf{b}) \times (\mathbf{a}-\mathbf{c})\|}$$

분자는 혼합곱(scalar triple product)으로, 점에서 평면까지의 부호 있는 거리에 법선 크기를 곱한 것이다.

비용 함수:

$$\mathbf{T}^* = \underset{\mathbf{T}}{\arg\min} \sum_{\mathbf{p} \in \mathcal{E}} d_e(\mathbf{T}\cdot\mathbf{p})^2 + \sum_{\mathbf{p} \in \mathcal{P}} d_p(\mathbf{T}\cdot\mathbf{p})^2$$

Levenberg-Marquardt로 최적화한다.

**Mapping 모듈 (~1Hz)**

Scan-to-map 정합으로 정밀한 포즈 보정 및 맵 업데이트를 수행한다. 누적된 글로벌 맵에 대해 새 스캔을 정합하여 드리프트를 보정한다.

Mapping 모듈은 Odometry 모듈보다 느리지만 더 정확하다. 맵은 현재 위치 주변의 큐브 형태로 유지하며, voxel 다운샘플링으로 밀도를 관리한다.

**모션 왜곡 보정 (Motion Distortion Compensation)**

회전형(spinning) LiDAR는 한 스캔을 완성하는 데 약 100ms가 걸린다. 이 동안 로봇이 움직이므로, 스캔 내 각 점은 서로 다른 시점에서 획득된다. 이 모션 왜곡을 보정하지 않으면 정합 정확도가 크게 떨어진다.

보정 방법: 스캔 시작 시점 $t_s$와 끝 시점 $t_e$ 사이의 포즈 변화 $\mathbf{T}_{s \to e}$를 알면, 각 점의 타임스탬프 $t_k$에 대해 중간 포즈를 등속 보간으로 추정한다:

$$\mathbf{T}(t_k) = \text{Exp}\left(\frac{t_k - t_s}{t_e - t_s} \cdot \text{Log}(\mathbf{T}_{s \to e})\right)$$

그리고 각 점을 기준 시점(보통 스캔 시작)으로 변환한다:

$$\mathbf{p}_k^{\text{corrected}} = \mathbf{T}(t_k)^{-1} \cdot \mathbf{p}_k$$

IMU가 있으면 IMU 적분으로 더 정확한 중간 포즈를 구할 수 있다.

```python
# LOAM 핵심 파이프라인 수도코드
class LOAM:
    def __init__(self):
        self.map_edge = VoxelMap(resolution=0.2)
        self.map_planar = VoxelMap(resolution=0.4)
        self.T_odom = np.eye(4)     # odometry 누적 변환
        self.T_map = np.eye(4)      # mapping 보정 변환
    
    def extract_features(self, scan):
        """스캔에서 edge/planar feature 추출"""
        edge_points = []
        planar_points = []
        
        for scan_line in scan.lines:
            curvatures = []
            for i in range(5, len(scan_line) - 5):
                # 좌우 5개 이웃의 벡터 합, range로 정규화
                diff = sum(scan_line[j] - scan_line[i] for j in range(i-5, i+6) if j != i)
                c = np.linalg.norm(diff) / (10 * np.linalg.norm(scan_line[i]))
                curvatures.append((c, i))
            
            # 스캔 라인을 4구간으로 나눠 균등 추출
            for sector in split_into_4(curvatures):
                sector.sort(reverse=True)  # 곡률 내림차순
                
                n_edge, n_planar = 0, 0
                for c, idx in sector:
                    if c > EDGE_THRESH and n_edge < 2:
                        if not near_selected(idx, edge_points):
                            edge_points.append(scan_line[idx])
                            n_edge += 1
                
                sector.sort()  # 곡률 오름차순
                for c, idx in sector:
                    if c < PLANAR_THRESH and n_planar < 4:
                        if not near_selected(idx, planar_points):
                            planar_points.append(scan_line[idx])
                            n_planar += 1
        
        return edge_points, planar_points
    
    def odometry(self, edge_curr, planar_curr, edge_prev, planar_prev):
        """Scan-to-scan 매칭 (10Hz)"""
        T_relative = np.eye(4)
        tree_edge = KDTree(edge_prev)
        tree_planar = KDTree(planar_prev)
        
        for iter in range(25):
            residuals = []
            jacobians = []
            
            # Edge point-to-edge 잔차
            for p in edge_curr:
                p_t = apply_transform(T_relative, p)
                _, idx = tree_edge.query(p_t, k=2)
                a, b = edge_prev[idx[0]], edge_prev[idx[1]]
                
                d_e = point_to_line_distance(p_t, a, b)
                J_e = point_to_line_jacobian(T_relative, p, a, b)
                residuals.append(d_e)
                jacobians.append(J_e)
            
            # Planar point-to-plane 잔차
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
        """Scan-to-map 정합 (1Hz)"""
        # 현재 위치 주변의 맵 추출
        local_edge_map = self.map_edge.get_points_near(self.T_odom[:3, 3], radius=50.0)
        local_planar_map = self.map_planar.get_points_near(self.T_odom[:3, 3], radius=50.0)
        
        # Scan-to-map 최적화 (odometry와 유사하지만 맵에 대해)
        T_correction = optimize_scan_to_map(edge_curr, planar_curr, 
                                            local_edge_map, local_planar_map)
        self.T_map = T_correction @ self.T_odom
        
        # 맵 업데이트
        self.map_edge.add_points(apply_transform(self.T_map, edge_curr))
        self.map_planar.add_points(apply_transform(self.T_map, planar_curr))
```

### 7.2.2 LeGO-LOAM

LeGO-LOAM ([Shan & Englot, 2018](https://doi.org/10.1109/IROS.2018.8594299))은 LOAM에 ground segmentation을 추가하고, 계산을 경량화하여 임베디드 시스템(Jetson TX2 등)에서도 실시간 동작을 달성했다.

**LeGO-LOAM의 핵심 추가 사항**:

1. **Ground Segmentation**: 포인트 클라우드를 range image로 변환한 뒤, 지면(ground)과 비지면을 분리한다. 인접 빔 간 기울기가 10° 미만이면 지면으로 판정한다. 지면점은 planar feature로 사용하고, 비지면점에서 edge feature를 추출한다.

2. **Point Cloud Segmentation**: 비지면점에 대해 range image 기반 클러스터링을 수행한다. 일정 크기 미만의 클러스터는 노이즈로 제거한다. 이 전처리가 LOAM 대비 특징점 품질을 크게 향상시킨다.

3. **2단계 LM 최적화**: LOAM이 6-DoF를 한 번에 최적화하는 반면, LeGO-LOAM은 ground planar feature로 먼저 $[t_z, \theta_{\text{roll}}, \theta_{\text{pitch}}]$를, 그 다음 edge feature로 $[t_x, t_y, \theta_{\text{yaw}}]$를 최적화한다. 이 분리가 수렴 속도와 안정성을 높인다.

4. **포즈 그래프 최적화**: LOAM에 없던 루프 클로저 + 포즈 그래프 최적화를 추가하여 글로벌 드리프트를 보정한다.

### 7.2.3 왜 LOAM 계열이 오래 살아남았는가

LOAM이 2014년에 발표된 이후 10년이 넘었지만, LOAM 계열의 아이디어는 여전히 LiDAR 오도메트리의 주류다. 그 이유:

1. **기하학적 명확성**: edge/planar feature는 물리적으로 의미 있는 기하학적 원시체(geometric primitive)에 대응한다. 이 구조화된 환경 가정은 대부분의 인공 환경에서 잘 맞는다.

2. **계산 효율**: 전체 점군(수만~수십만 점) 대신 수백~수천 개의 특징점만 사용하므로 빠르다.

3. **확장성**: 기본 프레임워크에 IMU(LIO-SAM), GPU(KISS-ICP), 카메라(LVI-SAM) 등을 모듈식으로 추가할 수 있다.

4. **강건성**: edge/planar 분류가 일종의 아웃라이어 필터 역할을 한다 — 노이즈나 동적 물체에 속하는 점은 일관된 곡률 패턴을 보이지 않으므로 자연스럽게 제외된다.

다만, LOAM 계열의 한계도 명확하다. 기하학적 특징이 부족한 환경(넓은 들판, 긴 터널)에서는 성능이 저하되며, solid-state LiDAR의 비반복 스캔 패턴에는 기존 특징 추출이 적합하지 않다. 이 한계를 극복한 것이 FAST-LIO2의 direct 접근이다.

---

## 7.3 Tightly-Coupled LiDAR-Inertial Odometry

LiDAR만으로는 빠른 모션에서 모션 왜곡이 심해지고, 초기값 제공이 어렵다. IMU를 tightly coupled로 결합하면 이 한계를 극복할 수 있다.

### 7.3.1 LIO-SAM

LIO-SAM ([Shan et al., 2020](https://arxiv.org/abs/2007.00258))은 factor graph 프레임워크 위에 LiDAR, IMU, GPS, loop closure를 통합한 LIO 시스템이다. LOAM 계열의 feature 기반 접근과 현대적 그래프 최적화를 결합했다.

**Factor Graph 기반 통합**

LIO-SAM의 핵심 혁신은 다양한 센서 측정을 factor graph의 factor로 모델링한다는 것이다:

1. **IMU Preintegration Factor**: Forster et al. (2017)의 on-manifold preintegration으로 연속 키프레임 간 IMU 제약을 표현한다. 이 factor는 두 키프레임의 상대 회전, 속도, 위치에 대한 제약과 함께, 바이어스 추정도 포함한다.

2. **LiDAR Odometry Factor**: LOAM 스타일의 scan-to-map 매칭으로 상대 포즈를 추정한다. 이 결과를 두 키프레임 간 상대 포즈 factor로 삽입한다.

3. **GPS Factor**: GPS 수신이 가능할 때, 위치 측정을 단항(unary) factor로 추가한다. GPS가 없는 구간에서는 이 factor가 없으므로, 시스템이 자연스럽게 LiDAR+IMU만으로 동작한다.

4. **Loop Closure Factor**: 장소 인식(Scan Context 등)으로 루프를 검출하고, ICP로 상대 포즈를 추정하여 이진(binary) factor로 추가한다.

이 모든 factor가 하나의 그래프에 들어가고, GTSAM의 iSAM2로 incremental 최적화를 수행한다. Factor graph의 강점은 **모듈성**이다 — 각 센서는 독립적으로 factor를 추가/제거할 수 있으며, 새 센서를 추가하는 것이 간단하다.

**IMU 기반 De-skewing**

LIO-SAM에서 IMU는 두 가지 역할을 한다:
1. **모션 왜곡 보정**: LiDAR 스캔 동안의 IMU 데이터로 각 점의 시점별 포즈를 정밀하게 보간하여 de-skewing한다.
2. **초기값 제공**: IMU preintegration으로 다음 키프레임의 포즈를 예측하여 scan-to-map 정합의 초기값으로 사용한다.

이 양방향 결합이 LIO-SAM의 "tightly-coupled"의 핵심이다: IMU가 LiDAR에 초기값과 de-skewing을 제공하고, LiDAR가 IMU에 포즈 보정과 바이어스 추정을 제공한다.

**Keyframe 기반 효율화**

전역 맵 대신, 현재 위치 주변의 키프레임들이 관측한 서브맵에 대해 scan matching을 수행한다. 이 슬라이딩 윈도우 기반 접근이 전역 맵 대비 계산량을 크게 줄인다.

```python
# LIO-SAM Factor Graph 구성 수도코드
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

FAST-LIO2 ([Xu et al., 2022](https://doi.org/10.1109/TRO.2022.3141855))는 LOAM 계열과 완전히 다른 접근을 취한다. 특징 추출을 제거하고, raw LiDAR 점을 직접 맵에 정합하는 direct LiDAR-inertial odometry이다.

**1. Direct Point Registration (특징 추출 제거)**

LOAM이 edge/planar feature를 추출하는 것과 달리, FAST-LIO2는 모든 raw 점을 직접 사용한다. 각 점 $\mathbf{p}_k$에 대해 맵에서 최근접 평면을 찾고, point-to-plane 거리를 최소화한다:

$$d_k = \mathbf{n}_k^T (\mathbf{T} \cdot \mathbf{p}_k - \mathbf{q}_k)$$

여기서 $\mathbf{n}_k$는 맵 내 최근접 평면의 법선, $\mathbf{q}_k$는 최근접점이다.

왜 특징 추출을 제거하는가?
- 특징 추출은 정보 손실이다 — 분류 임계값에 따라 유용한 점이 버려질 수 있다.
- 다양한 LiDAR 스캔 패턴(spinning, solid-state, non-repetitive)에 범용으로 적용 가능하다. 특히 Livox 같은 solid-state LiDAR는 비반복 스캔이라 기존 곡률 기반 특징 추출이 적합하지 않다.
- 충분히 효율적인 맵 자료구조(ikd-Tree)가 있으면 raw 점 정합이 real-time 가능하다.

**2. ikd-Tree (Incremental k-d Tree)**

FAST-LIO2의 두 번째 핵심 혁신은 맵 자료구조다. 기존 kd-tree는 정적이라 점 삽입/삭제에 비효율적이다. ikd-Tree는:

- **점 삽입**: $O(\log N)$ 시간에 새 점을 삽입한다.
- **점 삭제**: 맵 영역 밖의 점을 lazy delete로 효율적 제거한다.
- **동적 re-balancing**: 삽입/삭제로 인해 트리가 불균형해지면 scapegoat tree 방식으로 부분 재구축한다.
- **Box 범위 삭제**: 현재 위치에서 먼 영역의 점을 박스 단위로 삭제하여 맵 크기를 관리한다.

ikd-Tree 덕분에 FAST-LIO2는 맵을 실시간으로 유지하면서 최근접점 검색도 빠르게 수행한다.

**3. Iterated Extended Kalman Filter (IEKF)**

FAST-LIO2는 최적화 기반(LIO-SAM)이 아닌 필터 기반(IEKF)을 사용한다.

표준 EKF는 관측 모델을 한 번만 선형화하는데, LiDAR 관측의 비선형성이 크면 이 선형화가 부정확하다. IEKF는 업데이트를 여러 번 반복하여 선형화 지점을 개선한다:

$$\hat{\mathbf{x}}^{(k+1)} = \hat{\mathbf{x}}^{-} + \mathbf{K}^{(k)} (\mathbf{z} - h(\hat{\mathbf{x}}^{(k)}) - \mathbf{H}^{(k)}(\hat{\mathbf{x}}^{-} - \hat{\mathbf{x}}^{(k)}))$$

여기서 $k$는 반복 인덱스, $\hat{\mathbf{x}}^{-}$는 prediction 결과, $\mathbf{H}^{(k)}$는 $\hat{\mathbf{x}}^{(k)}$에서의 자코비안이다.

칼만 게인:
$$\mathbf{K}^{(k)} = \mathbf{P}^{-} (\mathbf{H}^{(k)})^T (\mathbf{H}^{(k)} \mathbf{P}^{-} (\mathbf{H}^{(k)})^T + \mathbf{R})^{-1}$$

IEKF는 보통 3~5회 반복으로 수렴한다. Gauss-Newton 최적화와 유사한 효과를 내지만, 불확실성(공분산)을 자연스럽게 전파한다는 필터의 장점을 유지한다.

**상태 벡터**:

$$\mathbf{x} = [{}^G\mathbf{R}_I, {}^G\mathbf{p}_I, {}^G\mathbf{v}_I, \mathbf{b}_g, \mathbf{b}_a, {}^I\mathbf{R}_L, {}^I\mathbf{p}_L, \mathbf{g}]$$

회전 ${}^G\mathbf{R}_I$, 위치 ${}^G\mathbf{p}_I$, 속도 ${}^G\mathbf{v}_I$, 자이로 바이어스 $\mathbf{b}_g$, 가속도계 바이어스 $\mathbf{b}_a$ 외에, LiDAR-IMU extrinsic ${}^I\mathbf{R}_L, {}^I\mathbf{p}_L$과 중력 벡터 $\mathbf{g}$도 포함한다. 즉, 외부 캘리브레이션과 중력 방향까지 온라인으로 추정한다.

실외 환경에서 최대 100Hz odometry + mapping을 달성하며, multi-line spinning LiDAR, solid-state LiDAR(Livox), UAV/핸드헬드 플랫폼, Intel/ARM 프로세서에서 모두 동작한다.

```cpp
// FAST-LIO2 IEKF 업데이트 수도코드 (C++)
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
    State x_predict = x;  // prediction 결과 보관
    MatrixXd K;  // 최종 반복의 칼만 게인을 루프 밖에서 사용
    MatrixXd H;  // 최종 반복의 자코비안을 루프 밖에서 사용
    int n_valid = 0;
    
    for (int iter = 0; iter < MAX_ITER; iter++) {
        // 1. 현재 상태로 점을 world frame으로 변환
        PointCloud world_pts = transform_to_world(scan, x);
        
        // 2. ikd-tree에서 각 점의 최근접 평면 검색
        vector<Plane> planes = ikd_tree.find_nearest_planes(world_pts, k=5);
        
        // 3. 관측 자코비안 및 잔차 계산
        n_valid = 0;
        H.resize(scan.size(), STATE_DIM);
        VectorXd z(scan.size());
        
        for (int i = 0; i < scan.size(); i++) {
            if (!planes[i].valid) continue;
            
            Vector3d p_w = x.R_GI * (x.R_IL * scan[i] + x.p_IL) + x.p_GI;
            
            // Point-to-plane 잔차: 관측값은 0 (점이 평면 위에 있어야 함)
            // z = 0 - h(x) = -d_k
            z(n_valid) = -planes[i].normal.dot(p_w - planes[i].center);
            
            // 자코비안: d(residual) / d(state_error)
            // ∂z/∂δθ_GI = n^T * [-(R_GI(R_IL*p + p_IL))×]
            // ∂z/∂δp_GI = n^T
            // ∂z/∂δθ_IL = n^T * R_GI * [-(R_IL*p)×]
            // ∂z/∂δp_IL = n^T * R_GI
            H.row(n_valid) = compute_jacobian(x, scan[i], planes[i]);
            n_valid++;
        }
        
        H.conservativeResize(n_valid, STATE_DIM);
        z.conservativeResize(n_valid);
        
        // 4. IEKF 업데이트
        MatrixXd S = H * P * H.transpose() + R_meas * MatrixXd::Identity(n_valid, n_valid);
        K = P * H.transpose() * S.inverse();
        
        VectorXd dx = K * (z - H * state_difference(x_predict, x));
        
        // 5. 상태 보정 (on-manifold): x^{(k+1)} = x^{-} ⊞ dx
        x = state_plus(x_predict, dx);
        
        // 수렴 확인
        if (dx.norm() < CONVERGENCE_THRESH) break;
    }
    
    // 공분산 업데이트
    MatrixXd I_KH = MatrixXd::Identity(STATE_DIM, STATE_DIM) - K * H;
    P = I_KH * P * I_KH.transpose() + K * R_meas * K.transpose();
    
    // 맵 업데이트: 정합된 점을 ikd-tree에 삽입
    PointCloud aligned = transform_to_world(scan, x);
    ikd_tree.insert(aligned);
}
```

### 7.3.3 Faster-LIO

Faster-LIO는 FAST-LIO2의 ikd-Tree를 incremental voxel 구조로 대체하여 더 빠른 처리를 달성한다.

핵심 변경: kd-tree 대신 해시 맵 기반 voxel 구조를 사용한다. 각 voxel 내에서 평면을 유지하며, 점이 추가될 때마다 평면 파라미터를 incremental하게 업데이트한다. kd-tree의 $O(\log N)$ 검색 대신 해시 $O(1)$ 접근으로 속도를 높인다.

### 7.3.4 Point-LIO

Point-LIO ([He et al., 2023](https://doi.org/10.1002/aisy.202200459))는 FAST-LIO 시리즈의 극단적 확장이다. 스캔 단위가 아닌 **개별 점** 단위로 상태를 업데이트한다.

기존 LIO는 전체 스캔(~100ms)을 하나의 관측으로 처리한다. 이 동안 등속 보간으로 모션 왜곡을 보정하지만, 고속/고각속도 모션에서는 등속 가정이 깨진다.

Point-LIO는 각 점이 도착하는 즉시 ($\sim$μs 단위) EKF 업데이트를 수행한다. IMU의 고주파(~1kHz) 측정과 LiDAR 점의 타임스탬프를 이용해, 각 점에 대응하는 정확한 IMU 상태를 사용한다.

Point-LIO의 상태 전파 모델은 IMU 측정 사이의 짧은 시간 간격에서:

$$\frac{d}{dt}\mathbf{R} = \mathbf{R}[\boldsymbol{\omega}]_\times, \quad \frac{d}{dt}\mathbf{v} = \mathbf{R}\mathbf{a} + \mathbf{g}, \quad \frac{d}{dt}\mathbf{p} = \mathbf{v}$$

를 이산화한다. 점 하나가 올 때마다 state propagation → single-point update를 수행하므로, 사실상 연속 시간(continuous-time) 필터에 근접한다.

장점: 극단적으로 빠른 모션(초당 수백 도 회전)에서도 정확한 오도메트리. 모션 왜곡 보정이 암묵적으로 이루어진다(각 점이 이미 올바른 시점의 상태로 처리되므로).

단점: 점 수에 비례하는 업데이트 횟수로 계산 부담 증가. FAST-LIO2 대비 약 2~3배 느리다.

### 7.3.5 COIN-LIO

[COIN-LIO (Pfreundschuh et al., 2024)](https://arxiv.org/abs/2310.01235)는 LiDAR-Inertial 시스템에 **카메라 intensity** 정보를 추가한다. 전통적 LIO에서 카메라를 결합하려면 별도의 visual feature 추출/추적이 필요하지만, COIN-LIO는 더 간단한 접근을 취한다:

LiDAR 점에 해당하는 카메라 픽셀의 밝기(intensity)를 기록하고, 맵 포인트에도 intensity를 할당한다. 정합 시 기하학적 거리(point-to-plane)와 함께 intensity 차이도 비용에 포함한다:

$$e_k = \alpha \cdot d_{\text{geom}}(\mathbf{p}_k) + (1-\alpha) \cdot |I_{\text{obs}}(\mathbf{p}_k) - I_{\text{map}}(\mathbf{p}_k)|$$

기하학적으로 퇴화(degenerate)된 환경 — 예를 들어 긴 터널이나 빈 홀 — 에서 intensity 정보가 추가적인 구속을 제공하여 정확도를 유지한다. 카메라의 텍스처 정보를 활용하면서도 본격적인 VIO 파이프라인의 복잡성을 피하는 실용적 접근이다.

---

## 7.4 Continuous-Time LiDAR Odometry

기존 LiDAR 오도메트리는 이산 시간(discrete-time) 모델을 사용한다. 각 스캔에 하나의 포즈를 할당하고, 스캔 내 모션은 등속 보간으로 근사한다. Continuous-time 접근은 궤적을 연속 함수로 표현하여 이 한계를 극복한다.

### 7.4.1 CT-ICP

CT-ICP ([Dellenbach et al., 2022](https://arxiv.org/abs/2109.12979))는 각 스캔에 하나의 포즈가 아닌 **두 개의 포즈**(스캔 시작과 끝)를 할당한다.

스캔 내 각 점의 타임스탬프 $t_k \in [t_s, t_e]$에 대해, 포즈를 선형 보간한다:

$$\mathbf{T}(t_k) = \mathbf{T}_s \cdot \text{Exp}\left(\frac{t_k - t_s}{t_e - t_s} \cdot \text{Log}(\mathbf{T}_s^{-1}\mathbf{T}_e)\right)$$

이 두 포즈 $\mathbf{T}_s, \mathbf{T}_e$를 동시에 최적화한다. 기존 등속 보간과 달리, 최적화 과정에서 스캔 내 모션 모델이 함께 정제된다.

CT-ICP는 IMU 없이도 모션 왜곡을 효과적으로 보정할 수 있어, IMU가 없는 시스템에서 특히 유용하다.

### 7.4.2 B-Spline 기반 궤적 표현

더 일반적인 continuous-time 접근은 B-spline으로 궤적을 표현하는 것이다. B-spline은 제어점(control point) $\{\mathbf{T}_i\}$에 의해 정의되는 매끄러운 곡선이다:

$$\mathbf{T}(t) = \prod_{i=0}^{k} \text{Exp}\left(B_i(t) \cdot \text{Log}(\mathbf{T}_{i-1}^{-1}\mathbf{T}_i)\right)$$

여기서 $B_i(t)$는 B-spline 기저 함수(basis function)다. 3차(cubic) B-spline이 주로 사용되며, $C^2$ 연속성을 보장한다.

B-spline 궤적의 핵심 이점은 **임의 시점 질의**다. 어떤 시점 $t$에서든 포즈, 속도, 가속도를 미분으로 얻을 수 있어, 비동기(asynchronous) 센서 데이터를 자연스럽게 처리한다. 물리적으로 타당한 $C^2$ 연속 궤적을 보장하며, B-spline의 국소성 덕분에 한 제어점의 변경이 전체 궤적에 파급되지 않는다.

단점:
- 제어점 간격(knot spacing)이 주요 하이퍼파라미터다. 너무 조밀하면 과적합, 너무 듬성하면 고속 모션을 표현하지 못한다.
- 이산 시간 대비 계산량이 증가한다.

Kalibr(Ch.3 참조)의 camera-IMU 캘리브레이션도 B-spline 궤적 표현을 사용한다.

---

## 7.5 Solid-State LiDAR 특화

Solid-state LiDAR(Livox 시리즈 등)는 회전형 LiDAR와 전혀 다른 스캔 패턴을 가진다.

**회전형 vs Solid-state**:

| 특성 | 회전형 (Velodyne, Ouster) | Solid-state (Livox) |
|------|--------------------------|---------------------|
| 스캔 패턴 | 반복적 (매 회전 같은 패턴) | 비반복적 (꽃잎/로즈 패턴) |
| FoV | 360° 수평 | 제한적 (70~77°) |
| 점 밀도 | 균등 | 시간에 따라 누적, 불균등 |
| 가격 | 높음 | 낮음 |
| 크기/무게 | 큼 | 작음 |

**비반복 스캔이 특징 추출에 미치는 영향**

LOAM 스타일의 곡률 기반 특징 추출은 같은 스캔 라인의 이웃점을 이용한다. 그러나 solid-state LiDAR는 정의된 스캔 라인이 없고, 점들이 비규칙적으로 분포한다. 기존 라인 기반 곡률 계산은 쓸 수 없다. KNN(K-Nearest Neighbors) 기반 국소 곡률을 쓰거나, 아예 특징 추출을 포기하고 raw 점을 그대로 써야 한다.

**FAST-LIO가 solid-state에 강한 이유**

FAST-LIO/FAST-LIO2는 raw 점을 직접 사용하므로 스캔 패턴과 무관하게 동작한다. Solid-state LiDAR는 시간이 지남에 따라 FoV를 점점 더 조밀하게 채우는데, FAST-LIO2의 ikd-Tree 맵은 이 점진적 밀집화를 자연스럽게 수용하여 맵 품질이 시간이 지날수록 향상된다. FoV가 좁아 한 스캔의 정보가 제한적이지만, IMU와의 tight coupling이 이를 보상한다.

Livox 시리즈는 가격 대비 성능이 뛰어나 드론과 핸드헬드 플랫폼으로 빠르게 퍼졌으며, FAST-LIO2 + Livox 조합은 현재 가장 널리 쓰이는 LIO 구성 중 하나다.

---

## 7.6 학습 기반 LiDAR Odometry

### 7.6.1 DeepLO 계열

학습 기반 LiDAR 오도메트리는 포인트 클라우드 쌍을 입력으로 받아 상대 포즈를 예측하는 네트워크를 훈련한다.

대표적 접근:
- **LO-Net** (Li et al., 2019): LiDAR 스캔을 2D range image로 변환하고, CNN으로 특징을 추출하여 포즈를 예측한다. 법선 추정과 마스크 예측을 보조 작업으로 추가하여 기하학적 이해를 유도한다.
- **DeepLO** (Cho et al., 2020): PointNet 기반으로 3D 점군을 직접 처리하여 포즈를 예측한다.
- **PWCLO-Net** (Wang et al., 2021): Pyramid, Warping, Cost volume 구조를 LiDAR 오도메트리에 적용한다.

### 7.6.2 현재의 한계

학습 기반 LiDAR 오도메트리는 전통적 방법 대비 아직 큰 격차가 있다. 그 이유:

1. **LiDAR 데이터의 특성**: 포인트 클라우드는 이미지와 달리 비정형(unstructured)이고 순서가 없다. CNN이 자연스럽게 처리하기 어렵다.

2. **기하학의 충분함**: ICP/GICP/NDT 같은 기하학적 방법이 이미 매우 정확하다. 카메라 영역에서 학습이 빛나는 이유 — 조명 변화, 텍스처 부족 등 기하학만으로 해결하기 어려운 문제 — 가 LiDAR에는 적용되지 않는다.

3. **데이터 부족**: 대규모 LiDAR 오도메트리 학습 데이터가 이미지 데이터에 비해 훨씬 적다.

4. **일반화**: 특정 LiDAR/환경에서 학습한 모델이 다른 LiDAR/환경에 잘 일반화되지 않는다.

현재 학습은 LiDAR 오도메트리 자체보다 보조 컴포넌트에서 더 효과적이다. 루프 클로저 검출(Scan Context, PointNetVLAD), 정합 초기값 추정(GeoTransformer, Ch.5 참조), 동적 물체 제거를 위한 시맨틱 분할 등이 그 예다.

---

## 7.7 최근 동향 (2023-2024)

위에서 다룬 시스템들 외에, 최근 주목할 만한 LiDAR 오도메트리 연구들이 있다.

**[KISS-ICP (Vizzo et al., 2023)](https://arxiv.org/abs/2209.15397)**: Point-to-point ICP가 적응적 임계값(adaptive thresholding)과 강건 커널(robust kernel), 간단한 모션 보상만으로 SOTA 수준의 성능을 달성할 수 있음을 보였다. 자동차, UAV 등 센서 타입에 무관하게 튜닝 없이 동작하는 범용성이 핵심이다. LiDAR 오도메트리에서 "단순함의 힘"을 재확인시킨 연구다.

**[MAD-ICP (Ferrari et al., 2024)](https://arxiv.org/abs/2405.05828)**: PCA 기반 kd-tree를 활용하여 포인트 클라우드의 구조적 정보를 추출하고, point-to-plane 정합에 사용한다. 데이터 매칭 전략 자체의 중요성을 강조하며, 다양한 LiDAR 센서에서 도메인 특화 방법과 동등한 성능을 달성한다.

**[iG-LIO (Chen et al., 2024)](https://github.com/zijiechenrobotics/ig_lio)**: Incremental GICP를 tightly-coupled LIO에 통합한 시스템이다. Voxel 기반 표면 공분산 추정기(VSCE)로 GICP의 공분산 계산 효율을 높이고, incremental voxel map으로 최근접점 검색 비용을 줄였다. Faster-LIO보다 효율적이면서 SOTA 수준의 정확도를 유지한다.

---

## 7장 요약

| 시스템 | 접근 | 추정 방법 | 센서 | 핵심 특징 |
|--------|------|-----------|------|-----------|
| ICP/GICP/NDT | Registration | 반복 최적화 | LiDAR only | 기본 빌딩 블록 |
| LOAM | Feature-based | LM 최적화 | LiDAR only | Edge/planar feature, 2단계 아키텍처 |
| LeGO-LOAM | Feature-based | LM 최적화 | LiDAR only | Ground segmentation, 경량화 |
| LIO-SAM | Feature-based | Factor graph (iSAM2) | LiDAR + IMU + GPS | 모듈식 다중 센서 통합 |
| FAST-LIO2 | Direct | IEKF | LiDAR + IMU | 특징 추출 없음, ikd-Tree, 100Hz |
| Point-LIO | Direct | Point-wise EKF | LiDAR + IMU | 점 단위 업데이트, 고속 모션 |
| COIN-LIO | Direct + Intensity | IEKF | LiDAR + IMU + Camera(intensity) | Intensity 기반 degeneration 방지 |
| CT-ICP | Direct | 최적화 | LiDAR only | 연속 시간 모션 모델, IMU 불필요 |
| KISS-ICP | Direct (P2P) | 반복 최적화 | LiDAR only | 적응적 임계값, 튜닝 불필요, 범용 |
| MAD-ICP | Direct (P2Plane) | 반복 최적화 | LiDAR only | PCA 기반 구조 추출, 데이터 매칭 중심 |
| iG-LIO | Direct (GICP) | IEKF | LiDAR + IMU | Incremental GICP, voxel 공분산 추정 |

LOAM(2014) → LeGO-LOAM(2018) → LIO-SAM(2020) 계보는 **feature-based + factor graph** 방향으로 진화했다. FAST-LIO(2021) → FAST-LIO2(2022) → Point-LIO(2023) 계보는 **direct + Kalman filter** 방향으로 진화했다. 두 계보는 서로 다른 설계 철학을 대표하지만, 실전 성능은 비슷한 수준에 수렴하고 있다.

Feature-based 접근은 구조화된 환경(건물, 도시)에서 강하고, direct 접근은 비구조화된 환경(숲, 동굴)과 solid-state LiDAR에서 강하다. 다음 챕터에서는 이 두 센서 모달리티(카메라 + LiDAR)를 IMU와 함께 통합하는 multi-sensor fusion 아키텍처를 다룬다.
