# Ch.7 — LiDAR Odometry & LiDAR-Inertial Odometry

같은 자기 운동 추정 문제도 카메라(+IMU) 대신 LiDAR로 풀면 조건이 바뀐다.

LiDAR는 카메라와 상보적인 센서다. 카메라는 풍부한 텍스처를 제공하지만 가시광 조명에 민감하고 단안 기하만으로 절대 거리를 정할 수 없다. LiDAR는 가시광 조명 변화에 덜 민감한 3D 거리를 측정하지만, 비·안개·반사율·다중경로의 영향을 받을 수 있다. LiDAR Odometry(LO)는 LiDAR 점군을 기반으로 자체 움직임을 추정하는 기법이다. 여기에 IMU 관성 측정을 결합한 형태를 LiDAR-Inertial Odometry(LIO)라고 부른다.

LiDAR 오도메트리는 **포인트 클라우드 정합(registration)**으로 연속된 두 스캔 사이의 강체 변환 $\mathbf{T} \in SE(3)$를 찾는다. 이 과정에는 데이터 연관(correspondence), 노이즈 모델, 계산 효율, 모션 왜곡 보정이 얽힌다.

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
- 평면에서의 슬라이딩 — 평면 위에서는 접선 방향으로 조금 움직여도 매 반복의 재대응이 다시 가까운 점을 찾아 주므로 비용면이 그 방향으로 거의 평평해지고 수렴이 느리다. 고정된 대응에서는 접선 이동이 비용을 키우므로, 평평해지는 원인은 비용 함수가 아니라 재대응이다.
- 초기값 의존성 — 로컬 최소값에 빠지기 쉽다.
- 최근접점 대응의 부정확함 — 두 스캔의 샘플링 패턴이 다르면 진정한 대응이 아닐 수 있다.

**Point-to-Plane ICP**

평면 위의 점에 대해서는, 점 사이 거리보다 점에서 평면까지의 거리가 더 물리적으로 의미 있다:

$$\mathbf{T}^* = \underset{\mathbf{T}}{\arg\min} \sum_i \left((\mathbf{T} \cdot \mathbf{p}_i - \mathbf{q}_{c(i)})^T \mathbf{n}_{c(i)}\right)^2$$

여기서 $\mathbf{n}_{c(i)}$는 타겟 점 $\mathbf{q}_{c(i)}$에서의 표면 법선(surface normal)이다. 이는 점 사이의 3차원 유클리드 거리가 아닌 접평면 법선 방향 오차만을 투영해 측정한다. 따라서 평면을 따라 미끄러지는 움직임은 잔차 비용에 불필요한 페널티를 주지 않는다.

장점: Point-to-point 대비 수렴 속도가 훨씬 빠르다. 특히 평면이 많은 실내/도시 환경에서 효과적이다.

단점: closed-form 해가 없어 반복 최적화(Gauss-Newton 등)가 필요하다. 법선 추정의 정확도에 의존한다.

법선은 각 점의 이웃점들에 대해 PCA(주성분 분석)를 수행하여 가장 작은 고유값에 대응하는 고유벡터로 추정한다. 이웃의 공분산 행렬:

$$\mathbf{C} = \frac{1}{k}\sum_{j \in \mathcal{N}(i)} (\mathbf{q}_j - \bar{\mathbf{q}})(\mathbf{q}_j - \bar{\mathbf{q}})^T$$

고유값 분해 $\mathbf{C} = \mathbf{V}\boldsymbol{\Lambda}\mathbf{V}^T$에서 $\lambda_{\min}$에 대응하는 고유벡터가 법선 방향이다.

### 7.1.2 GICP (Generalized ICP)

GICP ([Segal et al., 2009](https://doi.org/10.15607/RSS.2009.V.021))는 point-to-point, point-to-plane, plane-to-plane ICP를 하나의 확률적 프레임워크로 통합한다.

각 점은 국소 표면의 불확실성을 반영하는 공분산 $\mathbf{C}_i$를 가진다고 모델링한다. 비용 함수는:

$$\mathbf{T}^* = \underset{\mathbf{T}}{\arg\min} \sum_i (\mathbf{T} \cdot \mathbf{p}_i - \mathbf{q}_{c(i)})^T (\mathbf{C}_i^{\mathcal{Q}} + \mathbf{R}\mathbf{C}_i^{\mathcal{P}}\mathbf{R}^T)^{-1} (\mathbf{T} \cdot \mathbf{p}_i - \mathbf{q}_{c(i)})$$

여기서 $\mathbf{C}_i^{\mathcal{P}}, \mathbf{C}_i^{\mathcal{Q}}$는 각각 소스와 타겟 점의 국소 표면 공분산이다.

**공분산의 물리적 의미**:
- 평면 위의 점: 법선 방향으로 작은 분산, 접선 방향으로 큰 분산 → $\mathbf{C} = \mathbf{R}_s \text{diag}(\epsilon, 1, 1) \mathbf{R}_s^T$ ($\epsilon \ll 1$, $\mathbf{R}_s$는 첫 열이 법선인 회전, 즉 $\mathbf{R}_s \mathbf{e}_1 = \mathbf{n}$). $\mathbf{C}$의 고유벡터가 $\mathbf{R}_s$의 열이므로 첫 열이 법선일 때 그 방향의 분산이 $\epsilon$이 된다.
- 이 경우 GICP는 자동으로 plane-to-plane 정합이 된다.
- $\mathbf{C}^{\mathcal{P}} = \mathbf{0}$이면 point-to-plane, $\mathbf{C}^{\mathcal{P}} = \mathbf{C}^{\mathcal{Q}} = \mathbf{I}$이면 point-to-point가 된다.

GICP는 point-to-point와 point-to-plane ICP를 covariance 기반의 확률적 framework로 묶는다. 원 논문 실험에서는 두 baseline보다 높은 정확도와 correspondence error에 대한 강건성을 보고했지만, 실제 결과는 초기값·overlap·sampling·outlier 처리에 따라 달라진다.

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
   $$\mathbf{T}^* = \underset{\mathbf{T}}{\arg\max} \sum_i \exp\left(-\tfrac{1}{2}(\mathbf{T} \cdot \mathbf{p}_i - \boldsymbol{\mu}_{k(i)})^T \boldsymbol{\Sigma}_{k(i)}^{-1} (\mathbf{T} \cdot \mathbf{p}_i - \boldsymbol{\mu}_{k(i)})\right)$$
   
   가우시안 값을 로그 없이 더하므로 잔차가 큰 점의 기여가 포화된다. 이를 로그가능도의 합, 즉 마할라노비스 제곱합
   $$\sum_i (\mathbf{T} \cdot \mathbf{p}_i - \boldsymbol{\mu}_{k(i)})^T \boldsymbol{\Sigma}_{k(i)}^{-1} (\mathbf{T} \cdot \mathbf{p}_i - \boldsymbol{\mu}_{k(i)})$$
   의 최소화로 바꾸면 outlier의 영향이 무한히 커지는 복셀 기반 마할라노비스 ICP가 되어 원 NDT와 다른 알고리즘이 된다. 실제 구현은 가우시안에 균등분포를 섞은 근사를 쓴다.

NDT의 장점:
- 명시적 대응 찾기가 불필요 — 점이 어느 복셀에 속하는지만 판단하면 된다. kd-tree 구축 비용이 없다.
- 복셀 크기로 정밀도와 수렴 영역을 조절할 수 있다 — 큰 복셀은 넓은 수렴 영역, 작은 복셀은 높은 정밀도.
- 점수 함수가 매끄럽고, 멀리 떨어진 점의 기여가 포화되어 outlier에 덜 끌린다.

단점:
- 복셀 크기 선택에 민감하다.
- 점이 적은 복셀에서 공분산 추정이 불안정하다.
- 3D NDT는 Autoware의 ndt_scan_matcher처럼 자율주행의 사전 지도 기반 localization에 널리 쓰인다. scan-to-scan 오도메트리에서는 ICP/GICP 대비 정확도가 약간 떨어지는 경향이 보고되지만 데이터셋에 따라 갈린다.

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

LOAM은 두 가지를 결합했다:
1. LiDAR 스캔에서 기하학적 특징(edge, planar)을 추출하면, 전체 점군 대비 훨씬 적은 점으로 정확한 정합이 가능하다.
2. 빠른 odometry와 느린 mapping을 분리하면, 실시간성과 정확도를 동시에 달성할 수 있다.

**특징점 추출**

각 스캔 라인(scan line)에서 점의 국소 곡률(curvature)을 계산한다. 점 $\mathbf{p}_i$의 곡률:

$$c_i = \frac{1}{|\mathcal{S}_i| \cdot \|\mathbf{p}_i\|} \left\| \sum_{j \in \mathcal{S}_i} (\mathbf{p}_j - \mathbf{p}_i) \right\|$$

여기서 $\mathcal{S}_i$는 같은 스캔 라인에서 $\mathbf{p}_i$의 좌우 이웃점(보통 5개씩) 집합이고, $\|\mathbf{p}_i\|$는 센서로부터의 거리(range)다. 이 거리로 곡률 점수를 정규화하여 가까운 점과 먼 점을 비교 가능하게 한다.

- **Edge feature**: 곡률이 높은 점 ($c_i > c_{\text{thresh}}^e$). 물리적으로 모서리, 기둥 등 날카로운 경계에 해당.
- **Planar feature**: 곡률이 낮은 점 ($c_i < c_{\text{thresh}}^p$). 물리적으로 벽, 바닥 같은 평탄한 표면에 해당.

특징점 선택 시 추가 규칙:
- 각 스캔 라인을 4개 구간으로 나누어 균등 분포를 보장한다.
- 이웃 점에 이미 선택된 점이 있으면 제외(non-maximum suppression).
- 레이저 빔과 거의 평행한 표면의 점이나 가림(occlusion) 경계의 점은 불안정하므로 제외한다. 기준은 세계 좌표계의 수평이 아니라 빔에 대한 입사 기하다. 빔을 스치는 표면은 거리 추정이 불안정하고, 가림 경계의 점은 시점이 조금 바뀌면 사라진다.

**Odometry 모듈 (~10Hz)**

Scan-to-scan 매칭으로 빠른 모션 추정을 수행한다. 현재 스캔의 특징점을 이전 스캔의 특징점과 대응시키되, 거리 메트릭이 특징 유형에 따라 다르다:

**Edge point-to-edge distance**: 현재 스캔의 edge 점 $\mathbf{p}$에 대해, 이전 스캔에서 최근접 edge 점 $\mathbf{a}$와, $\mathbf{a}$의 인접 스캔 라인에서 가장 가까운 edge 점 $\mathbf{b}$를 찾는다(같은 라인의 두 점을 쓰면 직선이 퇴화한다). $\mathbf{p}$에서 직선 $\overline{\mathbf{ab}}$까지의 거리:

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

회전형(spinning) LiDAR의 한 스캔 시간은 회전 주파수의 역수다. 예컨대 10Hz 회전 주파수라면 1회 스캔에 약 100ms가 소요된다. 이 시간 동안 이동체가 계속 주행하므로 스캔 내의 각 측정점은 서로 다른 위치와 시점에서 기록된다. 모션 왜곡의 크기는 스캔 시간과 운동에 비례하므로 실제 timestamp와 궤적으로 보정한다.

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
                _, idx = tree_edge.query(p_t, k=2)  # 실제 LOAM은 두 점을 서로 다른 스캔 라인에서 고른다
                a, b = edge_prev[idx[0]], edge_prev[idx[1]]
                
                d_e = point_to_line_distance(p_t, a, b)
                J_e = point_to_line_jacobian(T_relative, p, a, b)
                residuals.append(d_e)
                jacobians.append(J_e)
            
            # Planar point-to-plane 잔차
            for p in planar_curr:
                p_t = apply_transform(T_relative, p)
                _, idx = tree_planar.query(p_t, k=3)  # 실제 LOAM은 같은 라인 두 점 + 인접 라인 한 점
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

LeGO-LOAM은 네 요소를 추가했다.

1. **Ground Segmentation**: 포인트 클라우드를 range image로 변환한 뒤, 지면(ground)과 비지면을 분리한다. 인접 빔 간 기울기가 10° 미만이면 지면으로 판정한다. 지면점은 planar feature로 사용하고, 비지면점에서 edge feature를 추출한다.

2. **Point Cloud Segmentation**: 비지면점에 대해 range image 기반 클러스터링을 수행한다. 일정 크기 미만의 클러스터는 노이즈로 제거한다. LeGO-LOAM은 이 전처리로 지면 차량 환경의 비지면 특징을 선별한다.

3. **2단계 LM 최적화**: LOAM이 6-DoF를 한 번에 최적화하는 반면, LeGO-LOAM은 ground planar feature로 먼저 $[t_z, \theta_{\text{roll}}, \theta_{\text{pitch}}]$를, 그 다음 edge feature로 $[t_x, t_y, \theta_{\text{yaw}}]$를 최적화한다. 이 분리가 수렴 속도와 안정성을 높인다.

4. **포즈 그래프 최적화**: LOAM에 없던 루프 클로저 + 포즈 그래프 최적화를 추가하여 글로벌 드리프트를 보정한다.

### 7.2.3 왜 LOAM 계열이 오래 살아남았는가

LOAM이 2014년에 발표된 이후 10년이 넘었지만, LOAM 계열의 아이디어는 여전히 LiDAR 오도메트리의 주류다. 그 배경에는 다음 네 가지 특성이 있다.

1. **기하학적 명확성**: edge/planar feature는 물리적으로 의미 있는 기하학적 원시체(geometric primitive)에 대응한다. 이 구조화된 환경 가정은 대부분의 인공 환경에서 잘 맞는다.

2. **계산 효율**: 전체 점군(수만~수십만 점) 대신 수백~수천 개의 특징점만 사용하므로 빠르다.

3. **확장성**: LiDAR frontend를 IMU(LIO-SAM)나 카메라(LVI-SAM)와 결합하는 식으로 센서 구성을 확장할 수 있다. KISS-ICP는 이런 센서 추가형 확장이 아닌, 단순한 LiDAR-only ICP 설계를 택한 대안적인 접근 방식을 취한다.

4. **강건성**: edge/planar 분류가 잡음과 불안정한 입사 기하에 대한 필터 역할을 한다 — 곡률 패턴이 일관되지 않은 점은 선택되지 않는다. 다만 곡률은 한 스캔 라인의 이웃점에서 계산하는 기하량이라 운동을 판별하지 못한다. 움직이는 차량의 모서리와 평면도 좋은 특징으로 선택되므로, 동적 물체 제거는 별도 단계가 필요하다.

다만, LOAM 계열의 한계도 명확하다. 첫째, 비반복 주사 패턴을 사용하는 solid-state LiDAR에서는 링 기반 특징 추출 기법을 그대로 적용하기 어렵다. FAST-LIO2의 direct 접근이 특징 추출을 없애 이 쪽을 푼다. 둘째, 기하학적 특징이 희소한 개활지나 직선 터널에서는 오도메트리 정확도가 크게 떨어진다. 이것은 환경의 관측 기하가 퇴화한 문제이므로 특징을 추출하든 raw 점을 쓰든 남는다.

---

## 7.3 Tightly-Coupled LiDAR-Inertial Odometry

LiDAR만으로는 빠른 모션에서 모션 왜곡이 심해지고, 초기값 제공이 어렵다. IMU를 tightly coupled로 결합하면 이 한계를 극복할 수 있다.

### 7.3.1 LIO-SAM

LIO-SAM ([Shan et al., 2020](https://arxiv.org/abs/2007.00258))은 factor graph 프레임워크 위에 LiDAR, IMU, GPS, loop closure를 통합한 LIO 시스템이다. LOAM 계열의 feature 기반 접근과 현대적 그래프 최적화를 결합했다.

**Factor Graph 기반 통합**

LIO-SAM은 다양한 센서 측정을 factor graph의 factor로 모델링한다:

1. **IMU Preintegration Factor**: Forster et al. (2017)의 on-manifold preintegration으로 연속 키프레임 간 IMU 제약을 표현한다. 이 factor는 두 키프레임의 상대 회전, 속도, 위치에 대한 제약과 함께, 바이어스 추정도 포함한다.

2. **LiDAR Odometry Factor**: LOAM 스타일의 scan-to-map 매칭으로 상대 포즈를 추정한다. 이 결과를 두 키프레임 간 상대 포즈 factor로 삽입한다.

3. **GPS Factor**: GPS 수신이 가능할 때, 위치 측정을 단항(unary) factor로 추가한다. GPS가 없는 구간에서는 이 factor가 없으므로, 시스템이 자연스럽게 LiDAR+IMU만으로 동작한다.

4. **Loop Closure Factor**: 장소 인식(Scan Context 등)으로 루프를 검출하고, ICP로 상대 포즈를 추정하여 이진(binary) factor로 추가한다.

이 factor들은 GTSAM의 iSAM2로 incremental 최적화하는 전역 포즈 그래프에 들어간다. LIO-SAM은 여기에 더해 IMU odometry용 그래프를 따로 두고 주기적으로 초기화하여, IMU 주기의 상태 전파와 바이어스 추정을 담당하게 한다. Factor graph 구조의 핵심 강점은 뛰어난 **모듈성**에 있다. 각 센서 관측을 독립적인 factor 형태로 손쉽게 탈부착할 수 있어 새로운 센서 모달리티의 추가가 수월하다.

**IMU 기반 De-skewing**

LIO-SAM에서 IMU는 두 가지 역할을 한다:
1. **모션 왜곡 보정**: LiDAR 스캔 동안의 IMU 데이터로 각 점의 시점별 포즈를 정밀하게 보간하여 de-skewing한다.
2. **초기값 제공**: IMU preintegration으로 다음 키프레임의 포즈를 예측하여 scan-to-map 정합의 초기값으로 사용한다.

이 양방향 결합 구조에서 IMU는 스캔 왜곡 보정과 점군 정합 초기값을 제공한다. 반대로 LiDAR 정합 결과는 IMU의 적분 드리프트를 보정하고 바이어스를 실시간으로 갱신하는 기준이 된다.

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
        # 새 바이어스 변수를 구속하는 랜덤워크 factor. 없으면 미구속 변수로 iSAM2 update가 실패한다
        self.graph.add(gtsam.BetweenFactorConstantBias(
            B(self.key_idx - 1), B(self.key_idx), gtsam.imuBias.ConstantBias(), self.bias_noise))
        
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

FAST-LIO2가 두 번째로 바꾼 것은 맵 자료구조다. 기존 kd-tree는 정적이라 점 삽입/삭제에 비효율적이다. ikd-Tree는:

- **점 삽입**: $O(\log N)$ 시간에 새 점을 삽입한다.
- **점 삭제**: 맵 영역 밖의 점을 lazy delete로 효율적으로 제거한다.
- **동적 re-balancing**: 삽입/삭제로 인해 트리가 불균형해지면 scapegoat tree 방식으로 부분 재구축한다.
- **Box 범위 삭제**: 현재 위치에서 먼 영역의 점을 박스 단위로 삭제하여 맵 크기를 관리한다.

ikd-Tree 덕분에 FAST-LIO2는 맵을 실시간으로 유지하면서 최근접점 검색도 빠르게 수행한다.

**3. Iterated Extended Kalman Filter (IEKF)**

FAST-LIO2는 최적화 기반(LIO-SAM)이 아닌 필터 기반(IEKF)을 사용한다.

표준 EKF는 관측 모델을 한 번만 선형화하는데, LiDAR 관측의 비선형성이 크면 이 선형화가 부정확하다. IEKF는 업데이트를 여러 번 반복하여 선형화 지점을 개선한다:

$$\hat{\mathbf{x}}^{(k+1)} = \hat{\mathbf{x}}^{-} + \mathbf{K}^{(k)} (\mathbf{z} - h(\hat{\mathbf{x}}^{(k)}) - \mathbf{H}^{(k)}(\hat{\mathbf{x}}^{-} - \hat{\mathbf{x}}^{(k)}))$$

여기서 $k$는 반복 인덱스, $\hat{\mathbf{x}}^{-}$는 prediction 결과, $\mathbf{H}^{(k)}$는 $\hat{\mathbf{x}}^{(k)}$에서의 자코비안이다.

칼만 이득:
$$\mathbf{K}^{(k)} = \mathbf{P}^{-} (\mathbf{H}^{(k)})^T (\mathbf{H}^{(k)} \mathbf{P}^{-} (\mathbf{H}^{(k)})^T + \mathbf{R})^{-1}$$

IEKF 반복 횟수는 잔차 감소나 상태 증분 기준으로 정한다. FAST-LIO 계열의 설정에서는 소수 회 반복하는 경우가 많지만, 필요한 횟수는 초기값과 장면 기하에 따라 달라진다. 이 갱신은 Gauss-Newton과 관련된 반복 선형화를 사용하면서 공분산도 전파한다.

**상태 벡터**:

$$\mathbf{x} = [{}^G\mathbf{R}_I, {}^G\mathbf{p}_I, {}^G\mathbf{v}_I, \mathbf{b}_g, \mathbf{b}_a, {}^I\mathbf{R}_L, {}^I\mathbf{p}_L, \mathbf{g}]$$

회전 ${}^G\mathbf{R}_I$, 위치 ${}^G\mathbf{p}_I$, 속도 ${}^G\mathbf{v}_I$, 자이로 바이어스 $\mathbf{b}_g$, 가속도계 바이어스 $\mathbf{b}_a$ 외에, LiDAR-IMU extrinsic ${}^I\mathbf{R}_L, {}^I\mathbf{p}_L$과 중력 벡터 $\mathbf{g}$도 포함한다. 즉, 외부 캘리브레이션과 중력 방향까지 온라인으로 추정한다.

FAST-LIO2 논문은 저자들의 하드웨어와 데이터에서 최대 100Hz의 odometry·mapping 처리율을 보고하고, multi-line spinning 및 solid-state LiDAR와 여러 플랫폼·프로세서의 실험을 제시한다. 현재 시스템의 처리율은 점 수, CPU, map 크기로 다시 측정해야 한다.

```cpp
// FAST-LIO2 IEKF 업데이트 수도코드 (C++)
struct State {
    Matrix3d R_GI;    // IMU -> world rotation (앞첨자 표기 G_R_I)
    Vector3d p_GI;    // IMU position in world
    Vector3d v_GI;    // IMU velocity in world
    Vector3d bg, ba;  // gyro/accel bias
    Matrix3d R_IL;    // LiDAR -> IMU rotation (앞첨자 표기 I_R_L)
    Vector3d p_IL;    // LiDAR 원점의 IMU 좌표계 위치
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

Faster-LIO는 FAST-LIO2의 ikd-Tree를 incremental voxel 구조로 대체해 처리 속도를 높인다.

kd-tree 대신 해시 맵 기반 voxel 구조를 사용한다. 각 voxel 내에서 평면을 유지하며, 점이 추가될 때마다 평면 파라미터를 incremental하게 업데이트한다. kd-tree의 $O(\log N)$ 검색 대신 해시 $O(1)$ 접근으로 속도를 높인다.

### 7.3.4 Point-LIO

Point-LIO ([He et al., 2023](https://doi.org/10.1002/aisy.202200459))는 FAST-LIO 시리즈의 극단적 확장이다. 스캔 단위가 아닌 **개별 점** 단위로 상태를 업데이트한다.

기존 LIO는 전체 스캔(~100ms)을 하나의 관측으로 처리한다. 스캔 안의 왜곡은 IMU 적분으로 각 점 시각의 포즈를 구해 보정하지만(§7.3.1), 갱신 자체는 스캔 단위이므로 그 구간을 하나의 운동으로 묶는 가정이 남는다. 고속·고각속도 모션에서는 이 가정이 깨진다.

Point-LIO는 점 timestamp 순서로 상태를 전파하고 point-wise update를 수행한다. 고주파 IMU와 LiDAR 점의 timestamp로 각 관측 시각의 상태를 추정해 scan-level deskew의 단일 운동 가정보다 시간 해상도를 높이지만, 그 상태도 IMU noise·bias·동기화 오차를 포함한 추정치다.

Point-LIO의 상태 전파는 IMU 측정 사이의 짧은 시간 간격에서 다음 연속 모델을 이산화한다:

$$\frac{d}{dt}\mathbf{R} = \mathbf{R}[\boldsymbol{\omega}]_\times, \quad \frac{d}{dt}\mathbf{v} = \mathbf{R}\mathbf{a} + \mathbf{g}, \quad \frac{d}{dt}\mathbf{p} = \mathbf{v}$$

점 하나가 올 때마다 state propagation → single-point update를 수행하므로, 사실상 연속 시간(continuous-time) 필터에 근접한다.

장점: 극단적으로 빠른 모션(초당 수백 도 회전)에서도 정확한 오도메트리. 모션 왜곡 보정이 암묵적으로 이루어진다(각 점이 이미 올바른 시점의 상태로 처리되므로).

단점: 점 수에 따라 업데이트 부담이 늘어난다. Point-LIO 논문이 보고한 FAST-LIO2와의 처리시간 비율은 해당 데이터·하드웨어·설정에 한정되므로 현재 구현에서 profile해야 한다.

### 7.3.5 COIN-LIO

[COIN-LIO (Pfreundschuh et al., 2024)](https://arxiv.org/abs/2310.01235)는 LiDAR-Inertial 시스템에 **LiDAR 반사 강도(intensity)** 정보를 결합한다. LiDAR가 측정한 반사 강도를 intensity image로 투영하고, 영상 내부와 관측 사이의 밝기 일관성을 개선하는 필터링을 수행한다.

포인트 클라우드 정합에서 구속이 약한 방향을 찾고, 그 방향을 보완하는 intensity image 패치를 선택한다. 선택한 패치의 photometric residual은 IMU 측정 및 point-to-plane residual과 함께 iterated EKF에서 융합된다.

긴 터널이나 평탄한 개활지처럼 기하학적으로 퇴화한 환경에서는 LiDAR 반사 강도의 공간적 변화가 추가 구속을 제공할 수 있다. 이 구속의 효과는 해당 방향에서 관측할 수 있는 intensity 패턴에 달려 있다.

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

$$\mathbf{T}(t) = \mathbf{T}_0 \prod_{i=1}^{k} \text{Exp}\left(\tilde{B}_i(t) \cdot \text{Log}(\mathbf{T}_{i-1}^{-1}\mathbf{T}_i)\right)$$

여기서 $\tilde{B}_i(t)$는 누적(cumulative) B-spline 기저 함수(basis function)다. 3차(cubic) B-spline이 주로 사용되며, $C^2$ 연속성을 보장한다.

B-spline 궤적에서는 **임의 시점 질의**가 가능하다. 어떤 시점 $t$에서든 궤적을 평가해 포즈를 얻고, 미분으로 속도와 가속도를 구해 비동기 센서 데이터를 처리한다. 3차 spline은 매듭 설정이 적절할 때 $C^2$ 연속성을 제공하지만, 이 수학적 매끄러움만으로 동역학적 실행 가능성이 보장되지는 않는다. 국소 지지 덕분에 한 제어점의 변경은 인접 구간에 주로 영향을 준다.

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
| FoV | 360° 수평 | 모델별 (Mid-40 38.4° 원형, Avia 70.4° 원형, Horizon 81.7°×25.1°) |
| 점 밀도 | 균등 | 시간에 따라 누적, 불균등 |
| 가격 | 높음 | 낮음 |
| 크기/무게 | 큼 | 작음 |

**비반복 스캔이 특징 추출에 미치는 영향**

LOAM 스타일의 곡률 기반 특징 추출은 같은 스캔 라인의 이웃점을 이용한다. 그러나 solid-state LiDAR는 정의된 스캔 라인이 없고, 점들이 비규칙적으로 분포한다. 기존 라인 기반 곡률 계산은 쓸 수 없다. KNN(K-Nearest Neighbors) 기반 국소 곡률을 쓰거나, 아예 특징 추출을 포기하고 raw 점을 그대로 써야 한다.

**FAST-LIO2가 solid-state에 강한 이유**

FAST-LIO2는 raw 점을 직접 사용하므로 스캔 패턴과 무관하게 동작한다. FAST-LIO는 아직 edge/planar 특징을 추출하며, 특징 추출을 없앤 것은 FAST-LIO2의 기여다. Solid-state LiDAR는 시간이 지남에 따라 FoV를 점점 더 조밀하게 채우는데, FAST-LIO2의 ikd-Tree 맵은 이 점진적 밀집화를 자연스럽게 수용하여 맵 품질이 시간이 지날수록 향상된다. FoV가 좁아 한 스캔의 정보가 제한적이지만, IMU와의 tight coupling이 이를 보상한다.

Livox 계열은 비반복 스캔 패턴과 작은 폼팩터 때문에 드론·핸드헬드·소형 로봇의 공개 연구에서 자주 사용된다. FAST-LIO2가 이 스캔 패턴을 지원하므로 Livox와 조합한 공개 예제와 데이터셋도 쉽게 찾을 수 있다. 실제 선택에서는 가격뿐 아니라 FoV, 거리, 시간 동기화, 점 분포와 목표 플랫폼을 함께 비교한다.

---

## 7.6 학습 기반 LiDAR Odometry

### 7.6.1 DeepLO 계열

학습 기반 LiDAR 오도메트리는 포인트 클라우드 쌍을 입력으로 받아 상대 포즈를 예측하는 네트워크를 훈련한다.

대표적인 접근은 다음과 같다.
- **LO-Net** (Li et al., 2019): LiDAR 스캔을 2D range image로 변환하고, CNN으로 특징을 추출하여 포즈를 예측한다. 법선 추정과 마스크 예측을 보조 작업으로 추가하여 기하학적 이해를 유도한다.
- **DeepLO** (Cho et al., 2020): 구면 투영한 vertex map·normal map을 CNN에 넣고, point-to-plane ICP 잔차 형태의 기하 인지 손실로 비지도 학습한다.
- **PWCLO-Net** (Wang et al., 2021): Pyramid, Warping, Cost volume 구조를 LiDAR 오도메트리에 적용한다.

### 7.6.2 현재의 한계

학습 기반 LiDAR 오도메트리는 전통적 방법 대비 아직 큰 격차가 있다. 네 가지 이유가 있다.

1. **LiDAR 데이터의 특성**: 포인트 클라우드는 이미지와 달리 비정형(unstructured)이고 순서가 없다. CNN이 자연스럽게 처리하기 어렵다.

2. **강한 기하 기준선**: ICP/GICP/NDT는 충분한 중첩과 구조가 있을 때 학습 없이도 강한 기준선을 제공한다. 그러나 반복 구조, 동적 객체, 강수·먼지, 희소·저중첩 장면에서는 여전히 실패할 수 있어 학습은 대응점·동적점·불확실성 추정 등에 보조적으로 쓰인다.

3. **데이터 부족**: 대규모 LiDAR 오도메트리 학습 데이터가 이미지 데이터에 비해 훨씬 적다.

4. **일반화**: 특정 LiDAR/환경에서 학습한 모델이 다른 LiDAR/환경에 잘 일반화되지 않는다.

현재 학습은 LiDAR 오도메트리 자체보다 보조 컴포넌트에서 더 효과적이다. 루프 클로저 검출(PointNetVLAD), 정합 초기값 추정(GeoTransformer, Ch.5 참조), 동적 물체 제거를 위한 시맨틱 분할 등이 그 예다. 같은 자리에서 잘 쓰이는 Scan Context는 학습이 아니라 수작업 설계 디스크립터이므로 학습의 예가 아니라 비교 대상이다.

---

## 7.7 최근 동향 (2023-2024)

2023~2024년에는 간결성과 적응성을 앞세운 신규 LiDAR 오도메트리 파이프라인들이 주목받았다.

**[KISS-ICP (Vizzo et al., 2023)](https://arxiv.org/abs/2209.15397)**: Point-to-point ICP에 적응적 임계값, 강건 커널, 모션 보상을 결합한다. 원 논문은 자동차·UAV·handheld 데이터셋 전반에서 단일 파라미터 세트로 경쟁력 있는 궤적 추정 성능을 달성할 수 있음을 보고했다. 벤치마크마다 개별 파라미터를 미세 조정해야 하는 부담을 대폭 줄였다.

**[MAD-ICP (Ferrari et al., 2024)](https://arxiv.org/abs/2405.05828)**: PCA 기반 kd-tree를 활용하여 포인트 클라우드의 구조적 정보를 추출하고, point-to-plane 정합에 사용한다. 데이터 매칭 전략에 초점을 맞추며, 다양한 LiDAR 센서에서 도메인 특화 방법과 동등한 성능을 달성한다.

**[iG-LIO (Chen et al., 2024)](https://github.com/zijiechenrobotics/ig_lio)**: Incremental GICP를 tightly-coupled LIO에 통합한 시스템이다. Voxel 기반 표면 공분산 추정기(VSCE)로 GICP의 공분산 계산 효율을 높이고, incremental voxel map으로 최근접점 검색 비용을 줄였다. 공개 구현과 논문은 선택한 benchmark에서 Faster-LIO와 계산량·정확도를 비교한다.

---

## 7장 요약

| 시스템 | 접근 | 추정 방법 | 센서 | 특징 |
|--------|------|-----------|------|-----------|
| ICP/GICP/NDT | Registration | 반복 최적화 | LiDAR only | 기본 빌딩 블록 |
| LOAM | Feature-based | LM 최적화 | LiDAR only | Edge/planar feature, 2단계 아키텍처 |
| LeGO-LOAM | Feature-based | LM 최적화 | LiDAR only | Ground segmentation, 경량화 |
| LIO-SAM | Feature-based | Factor graph (iSAM2) | LiDAR + IMU + GPS | 모듈식 다중 센서 통합 |
| FAST-LIO2 | Direct | IEKF | LiDAR + IMU | 특징 추출 없음, ikd-Tree, 논문 설정에서 최대 100Hz 보고 |
| Point-LIO | Direct | Point-wise EKF | LiDAR + IMU | 점 단위 업데이트, 고속 모션 |
| COIN-LIO | Direct + Intensity | IEKF | LiDAR + IMU | LiDAR intensity로 기하학적 퇴화 보완 |
| CT-ICP | Direct | 최적화 | LiDAR only | 연속 시간 모션 모델, IMU 불필요 |
| KISS-ICP | Direct (P2P) | 반복 최적화 | LiDAR only | 적응적 임계값, 튜닝 불필요, 범용 |
| MAD-ICP | Direct (P2Plane) | 반복 최적화 | LiDAR only | PCA 기반 구조 추출, 데이터 매칭 중심 |
| iG-LIO | Direct (GICP) | IEKF | LiDAR + IMU | Incremental GICP, voxel 공분산 추정 |

LOAM(2014) → LeGO-LOAM(2018) → LIO-SAM(2020) 계보는 **feature-based + factor graph** 방향을 보여준다. FAST-LIO(2021) → FAST-LIO2(2022) → Point-LIO(2023) 계보는 **direct + Kalman filter** 방향을 보여준다. 어느 쪽이 더 정확하거나 빠른지는 sensor pattern, motion, map scale, 하드웨어, benchmark protocol에 따라 달라진다.

Feature-based 방식은 기하학적 평면과 모서리가 풍부한 도심 환경에서 높은 안정성을 보인다. 반면 direct 접근법은 형태가 비정형적인 산림이나 solid-state LiDAR 환경에서 더 유리하다. 카메라와 LiDAR를 IMU와 함께 묶으면 multi-sensor fusion 아키텍처 문제가 된다.
