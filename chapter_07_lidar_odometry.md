# Ch.7 — LiDAR Odometry & LiDAR-Inertial Odometry

같은 자기 운동 추정 문제도 카메라(+IMU) 대신 LiDAR로 풀면 조건이 바뀐다.

LiDAR와 카메라는 서로의 약점을 메운다. 카메라는 텍스처를 준다. 다만 가시광 조명에 민감하고, 단안 기하만으로는 절대 거리를 알 수 없다. LiDAR는 3D 거리를 직접 측정하며 가시광 조명 변화의 영향을 덜 받는다. 대신 비, 안개, 반사율, 다중경로의 영향을 받는다. LiDAR Odometry(LO)는 LiDAR 포인트 클라우드로 센서 자신의 움직임을 추정한다. 여기에 IMU 관성 측정을 더한 것을 LiDAR-Inertial Odometry(LIO)라고 부른다.

LiDAR 오도메트리는 포인트 클라우드 정합(registration)으로 연속된 두 스캔 사이의 강체 변환 $\mathbf{T} \in SE(3)$를 찾는다. 이 과정에는 데이터 연관(correspondence), 노이즈 모델, 계산 효율, 모션 왜곡 보정이 얽힌다.

---

## 7.1 Point Cloud Registration 기초

포인트 클라우드 정합은 두 포인트 클라우드 $\mathcal{P} = \{\mathbf{p}_i\}$와 $\mathcal{Q} = \{\mathbf{q}_j\}$를 가장 잘 겹치는 강체 변환을 찾는 문제다.

$$\mathbf{T}^* = \underset{\mathbf{T} \in SE(3)}{\arg\min} \sum_i d(\mathbf{T} \cdot \mathbf{p}_i, \mathcal{Q})$$

여기서 $d(\cdot, \cdot)$는 변환된 소스 점과 타겟 포인트 클라우드 사이의 거리 메트릭이다. 이 거리를 어떻게 정의하느냐로 ICP 변종이 나뉜다.

### 7.1.1 ICP 변종들

**Point-to-Point ICP ([Besl & McKay, 1992](https://doi.org/10.1109/34.121791))**

가장 기본 형태다. 변환한 소스 점과 가장 가까운 타겟 점 사이의 유클리드 거리를 최소화한다.

$$\mathbf{T}^* = \underset{\mathbf{T}}{\arg\min} \sum_i \left\|\mathbf{T} \cdot \mathbf{p}_i - \mathbf{q}_{c(i)}\right\|^2$$

여기서 $c(i) = \arg\min_j \|\mathbf{T} \cdot \mathbf{p}_i - \mathbf{q}_j\|$는 최근접점(closest point) 대응이다. ICP는 두 단계를 반복한다:

1. **대응 찾기**: 현재 변환으로 소스 점을 옮긴 뒤 타겟에서 최근접점을 찾는다. kd-tree를 쓰면 점 $N$개 전체에 $O(N\log N)$이 든다.

2. **변환 추정**: 대응 쌍을 알면 최적 변환을 closed-form으로 구한다. SVD로 푸는 방법은 다음과 같다.

   두 포인트 클라우드에서 각각 중심을 뺀다:
   $$\bar{\mathbf{p}} = \frac{1}{N}\sum_i \mathbf{p}_i, \quad \bar{\mathbf{q}} = \frac{1}{N}\sum_i \mathbf{q}_{c(i)}$$

   교차 공분산 행렬을 계산한다:
   $$\mathbf{W} = \sum_i (\mathbf{p}_i - \bar{\mathbf{p}})(\mathbf{q}_{c(i)} - \bar{\mathbf{q}})^T$$

   SVD 분해: $\mathbf{W} = \mathbf{U}\boldsymbol{\Sigma}\mathbf{V}^T$

   최적 회전과 이동:
   $$\mathbf{R}^* = \mathbf{V} \text{diag}(1, 1, \det(\mathbf{V}\mathbf{U}^T)) \mathbf{U}^T, \quad \mathbf{t}^* = \bar{\mathbf{q}} - \mathbf{R}^*\bar{\mathbf{p}}$$

   $\det(\mathbf{V}\mathbf{U}^T) = 1$이면 $\mathbf{R}^* = \mathbf{V}\mathbf{U}^T$이고, $\det(\mathbf{V}\mathbf{U}^T) = -1$이면 반사(reflection)를 막으려고 $\mathbf{V}$의 마지막 열 부호를 뒤집는다.

Point-to-point ICP의 한계:
- 평면 위 슬라이딩 — 평면 위에서는 접선 방향으로 조금 움직여도 매 반복의 재대응이 다시 가까운 점을 찾아 주므로 비용면이 그 방향으로 거의 평평해지고 수렴이 느리다. 대응을 고정하면 접선 이동이 비용을 키운다.
- 초기값 의존성 — 로컬 최소값에 빠지기 쉽다.
- 최근접점 대응이 부정확하다 — 두 스캔의 샘플링 패턴이 다르면 최근접점이 실제 같은 표면 점이 아닐 수 있다.

**Point-to-Plane ICP**

평면 위의 점이라면 점 사이 거리보다 점에서 평면까지의 거리를 오차로 삼는 편이 맞다.

$$\mathbf{T}^* = \underset{\mathbf{T}}{\arg\min} \sum_i \left((\mathbf{T} \cdot \mathbf{p}_i - \mathbf{q}_{c(i)})^T \mathbf{n}_{c(i)}\right)^2$$

여기서 $\mathbf{n}_{c(i)}$는 타겟 점 $\mathbf{q}_{c(i)}$의 표면 법선(surface normal)이다. 이 잔차는 오차를 접평면 법선 방향으로 투영한 성분만 잰다. 따라서 평면을 따라 미끄러지는 움직임에는 잔차가 0으로 유지된다.

장점: Point-to-point보다 훨씬 빨리 수렴한다. 평면이 많은 실내·도시 환경에서 특히 그렇다.

단점: closed-form 해가 없어 Gauss-Newton 같은 반복 최적화를 써야 한다. 결과가 법선 추정의 정확도에 따라 달라진다.

법선은 점마다 이웃점들에 PCA(주성분 분석)를 적용해, 가장 작은 고유값의 고유벡터로 추정한다. 이웃의 공분산 행렬은 다음과 같다.

$$\mathbf{C} = \frac{1}{k}\sum_{j \in \mathcal{N}(i)} (\mathbf{q}_j - \bar{\mathbf{q}})(\mathbf{q}_j - \bar{\mathbf{q}})^T$$

고유값 분해 $\mathbf{C} = \mathbf{V}\boldsymbol{\Lambda}\mathbf{V}^T$에서 $\lambda_{\min}$에 대응하는 고유벡터가 법선 방향이다.

### 7.1.2 GICP (Generalized ICP)

GICP ([Segal et al., 2009](https://doi.org/10.15607/RSS.2009.V.021))는 point-to-point, point-to-plane, plane-to-plane ICP를 확률 모델 하나로 묶는다.

점마다 국소 표면의 불확실성을 나타내는 공분산 $\mathbf{C}_i$를 둔다. 비용 함수는 다음과 같다.

$$\mathbf{T}^* = \underset{\mathbf{T}}{\arg\min} \sum_i (\mathbf{T} \cdot \mathbf{p}_i - \mathbf{q}_{c(i)})^T (\mathbf{C}_i^{\mathcal{Q}} + \mathbf{R}\mathbf{C}_i^{\mathcal{P}}\mathbf{R}^T)^{-1} (\mathbf{T} \cdot \mathbf{p}_i - \mathbf{q}_{c(i)})$$

여기서 $\mathbf{C}_i^{\mathcal{P}}, \mathbf{C}_i^{\mathcal{Q}}$는 각각 소스와 타겟 점의 국소 표면 공분산이다.

**공분산의 물리적 의미**:
- 평면 위의 점: 법선 방향으로 작은 분산, 접선 방향으로 큰 분산 → $\mathbf{C} = \mathbf{R}_s \text{diag}(\epsilon, 1, 1) \mathbf{R}_s^T$ ($\epsilon \ll 1$, $\mathbf{R}_s$는 첫 열이 법선인 회전, 즉 $\mathbf{R}_s \mathbf{e}_1 = \mathbf{n}$). $\mathbf{C}$의 고유벡터가 $\mathbf{R}_s$의 열이므로 첫 열이 법선일 때 그 방향의 분산이 $\epsilon$이 된다.
- 이 경우 GICP는 자동으로 plane-to-plane 정합이 된다.
- 원 논문 3절은 $\mathbf{C}^{\mathcal{P}} = \mathbf{0}$, $\mathbf{C}^{\mathcal{Q}} = \mathbf{I}$로 두면 point-to-point가 되고, $\mathbf{C}^{\mathcal{P}} = \mathbf{0}$으로 두고 $(\mathbf{C}^{\mathcal{Q}})^{-1}$을 법선 방향 사영 행렬 $\mathbf{P}_i$로 두면 point-to-plane이 된다고 보였다.

원 논문은 실험에서 GICP가 두 baseline보다 정확했고 잘못된 correspondence에도 더 강했다고 보고했다. 실제 결과는 초기값·overlap·sampling·outlier 처리에 따라 달라진다.

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
            J = compute_se3_jacobian(T, P[i])  # 3 x 6

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

[NDT (Biber & Strasser, 2003)](https://doi.org/10.1109/IROS.2003.1249285)는 점을 그대로 쓰지 않고, 공간을 복셀(voxel)로 나눈 뒤 복셀마다 점 분포를 가우시안으로 나타낸다.

**NDT 절차**:

1. **타겟 포인트 클라우드의 NDT 표현 만들기**: 공간을 3D 복셀 격자로 나누고, 복셀 $k$마다 그 안의 점들로 가우시안 $\mathcal{N}(\boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k)$를 구한다.
   $$\boldsymbol{\mu}_k = \frac{1}{n_k}\sum_{i \in k} \mathbf{q}_i, \quad \boldsymbol{\Sigma}_k = \frac{1}{n_k-1}\sum_{i \in k} (\mathbf{q}_i - \boldsymbol{\mu}_k)(\mathbf{q}_i - \boldsymbol{\mu}_k)^T$$

2. **변환 최적화**: 변환된 소스 점이 타겟 NDT 분포에서 높은 가능도(likelihood)를 얻도록 최적화:
   $$\mathbf{T}^* = \underset{\mathbf{T}}{\arg\max} \sum_i \exp\left(-\tfrac{1}{2}(\mathbf{T} \cdot \mathbf{p}_i - \boldsymbol{\mu}_{k(i)})^T \boldsymbol{\Sigma}_{k(i)}^{-1} (\mathbf{T} \cdot \mathbf{p}_i - \boldsymbol{\mu}_{k(i)})\right)$$

   가우시안 값을 로그 없이 더하므로 잔차가 큰 점의 기여가 포화된다. 이를 로그가능도의 합, 즉 마할라노비스 제곱합
   $$\sum_i (\mathbf{T} \cdot \mathbf{p}_i - \boldsymbol{\mu}_{k(i)})^T \boldsymbol{\Sigma}_{k(i)}^{-1} (\mathbf{T} \cdot \mathbf{p}_i - \boldsymbol{\mu}_{k(i)})$$
   의 최소화로 바꾸면 outlier의 영향이 제한 없이 커지는 복셀 기반 마할라노비스 ICP가 된다. 원 NDT와는 다른 알고리즘이다. 실제 구현은 가우시안에 균등분포를 섞은 근사를 쓴다.

NDT의 장점:
- 점끼리 대응을 찾지 않는다 — 점이 어느 복셀에 드는지만 보면 된다. kd-tree를 만드는 비용이 없다.
- 복셀 크기로 정밀도와 수렴 영역을 조절할 수 있다 — 큰 복셀은 넓은 수렴 영역, 작은 복셀은 높은 정밀도.
- 점수 함수가 매끄럽고, 멀리 떨어진 점의 기여가 포화되어 outlier에 덜 끌린다.

단점:
- 복셀 크기 선택에 민감하다.
- 점이 적은 복셀에서 공분산 추정이 불안정하다.
- 실패할 때는 크게 틀린다. Kvarntorp 광산 데이터로 두 방법을 비교한 [Magnusson et al. (2008)](http://ais.informatik.uni-freiburg.de/3dmapping_ws_iros08/abstracts/magnusson.pdf)에서 NDT는 ICP보다 넓은 초기 오차 범위에서 수렴했다. 그러나 NDT가 실패한 경우에는 이동과 회전이 함께 틀려 ICP보다 결과가 나쁜 때가 있었다.

3D NDT는 Autoware의 ndt_scan_matcher처럼 자율주행에서 사전 지도로 위치를 찾는 localization에 쓰인다.

### 7.1.4 수렴성과 초기값 의존성

위의 정합 알고리즘은 모두 국소 최적화라서 초기값을 잘 줘야 한다. 초기 포즈 오차가 크면 엉뚱한 로컬 미니멈에 수렴한다.

초기값은 보통 다음 방법으로 얻는다.
1. **등속 모델**: 직전 두 프레임의 변환을 외삽한다. 가장 간단하지만 모션이 급하게 바뀌면 틀린다.
2. **IMU 적분**: 짧은 시간의 IMU 적분이 좋은 초기값이 된다. LIO가 LO보다 강건한 이유 하나가 이것이다.
3. **멀티 해상도 (Multi-resolution)**: coarse-to-fine 방식이다. 큰 복셀이나 다운샘플한 포인트 클라우드로 대략 정합한 뒤 세밀하게 다듬는다.
4. **RANSAC + 특징 매칭**: FPFH, SHOT 같은 3D 디스크립터로 대응을 찾고 RANSAC으로 초기 변환을 추정한다. 일반 정합에는 쓸모 있지만, odometry에서는 연속한 스캔 사이 변화가 작아 등속 모델이나 IMU가 더 실용적이다.

---

## 7.2 Feature-based LiDAR Odometry

### 7.2.1 LOAM (Lidar Odometry and Mapping in Real-time)

LOAM ([Zhang & Singh, 2014](https://doi.org/10.15607/RSS.2014.X.007))은 뒤의 LeGO-LOAM, LIO-SAM, A-LOAM 같은 시스템이 바탕으로 삼은 LiDAR 오도메트리다.

LOAM의 착상은 다음과 같다.
1. LiDAR 스캔에서 기하 특징(edge, planar)만 뽑으면 전체 포인트 클라우드보다 훨씬 적은 점으로도 정확하게 정합한다.
2. 빠른 odometry와 느린 mapping을 나누면 실시간성과 정확도를 함께 얻는다.

**특징점 추출**

각 스캔 라인(scan line)에서 점의 국소 곡률(curvature)을 계산한다. 점 $\mathbf{p}_i$의 곡률:

$$c_i = \frac{1}{|\mathcal{S}_i| \cdot \|\mathbf{p}_i\|} \left\| \sum_{j \in \mathcal{S}_i} (\mathbf{p}_j - \mathbf{p}_i) \right\|$$

여기서 $\mathcal{S}_i$는 같은 스캔 라인에서 $\mathbf{p}_i$ 좌우의 이웃점(공개 구현은 5개씩) 집합이고, $\|\mathbf{p}_i\|$는 센서에서 점까지의 거리(range)다. 이 거리로 곡률 점수를 나눠 가까운 점과 먼 점을 견줄 수 있게 한다.

- **Edge feature**: 곡률이 큰 점 ($c_i > c_{\text{thresh}}^e$). 모서리, 기둥처럼 날카로운 경계다.
- **Planar feature**: 곡률이 작은 점 ($c_i < c_{\text{thresh}}^p$). 벽, 바닥 같은 평평한 표면이다.

특징점을 고를 때는 규칙을 더 둔다.
- 스캔 라인마다 4구간으로 나눠 특징점이 고르게 퍼지게 한다.
- 이웃에 이미 고른 점이 있으면 뺀다(non-maximum suppression).
- 레이저 빔과 거의 평행한 표면의 점이나 가림(occlusion) 경계의 점은 불안정하므로 제외한다. 판정 기준은 빔과 표면이 이루는 입사 기하다. 빔을 스치는 표면에서는 거리 추정이 불안정하다. 가림 경계의 점은 시점이 조금 바뀌면 사라진다.

**Odometry 모듈 (~10Hz)**

Scan-to-scan 매칭으로 모션을 빠르게 추정한다. 현재 스캔의 특징점을 이전 스캔의 특징점과 대응시키되, 거리 메트릭이 특징 유형에 따라 다르다:

**Edge point-to-edge distance**: 현재 스캔의 edge 점 $\mathbf{p}$에 대해, 이전 스캔에서 최근접 edge 점 $\mathbf{a}$와, $\mathbf{a}$의 인접 스캔 라인에서 가장 가까운 edge 점 $\mathbf{b}$를 찾는다(같은 라인의 두 점을 쓰면 직선이 퇴화한다). $\mathbf{p}$에서 직선 $\overline{\mathbf{ab}}$까지의 거리:

$$d_e = \frac{\|(\mathbf{p}-\mathbf{a}) \times (\mathbf{p}-\mathbf{b})\|}{\|\mathbf{a}-\mathbf{b}\|}$$

이것은 두 벡터의 외적 크기를 밑변 길이로 나눈 것으로, 삼각형의 높이(= 점에서 직선까지의 거리)와 같다.

**Planar point-to-plane distance**: 현재 스캔의 planar 점 $\mathbf{p}$마다 이전 스캔에서 평면점 셋을 찾는다. 최근접 planar 점 $\mathbf{a}$, $\mathbf{a}$와 같은 라인에서 가장 가까운 점 $\mathbf{b}$, 인접 라인에서 가장 가까운 점 $\mathbf{c}$다(세 점이 한 라인에 있으면 평면이 퇴화한다). $\mathbf{p}$에서 평면 $\triangle\mathbf{abc}$까지의 거리:

$$d_p = \frac{(\mathbf{p}-\mathbf{a})^T \left((\mathbf{a}-\mathbf{b}) \times (\mathbf{a}-\mathbf{c})\right)}{\|(\mathbf{a}-\mathbf{b}) \times (\mathbf{a}-\mathbf{c})\|}$$

분자는 혼합곱(scalar triple product)으로, 점에서 평면까지의 부호 있는 거리에 법선 크기를 곱한 값이다.

비용 함수:

$$\mathbf{T}^* = \underset{\mathbf{T}}{\arg\min} \sum_{\mathbf{p} \in \mathcal{E}} d_e(\mathbf{T}\cdot\mathbf{p})^2 + \sum_{\mathbf{p} \in \mathcal{P}} d_p(\mathbf{T}\cdot\mathbf{p})^2$$

Levenberg-Marquardt로 최적화한다.

**Mapping 모듈 (~1Hz)**

Scan-to-map 정합으로 포즈를 다듬고 맵을 갱신한다. 새 스캔을 누적 맵에 맞춰 드리프트를 줄인다.

Mapping 모듈은 Odometry 모듈보다 느리지만 더 정확하다. 맵은 현재 위치 주변의 큐브 영역만 두고, voxel 다운샘플링으로 밀도를 조절한다.

**모션 왜곡 보정 (Motion Distortion Compensation)**

회전형(spinning) LiDAR의 한 스캔 시간은 회전 주파수의 역수다. 예컨대 10Hz로 돌면 한 스캔에 약 100ms가 걸린다. 그동안 이동체가 계속 움직이므로 한 스캔 안의 측정점마다 찍힌 위치와 시점이 다르다. 모션 왜곡의 크기는 스캔 시간과 운동에 비례하므로 실제 timestamp와 궤적으로 보정한다.

보정 방법: LOAM은 한 스캔 동안 각속도와 선속도가 일정하다고 본다. 스캔 시작 시점 $t_s$부터 $t_e$까지의 LiDAR 포즈 변환 $\mathbf{T}_{s \to e}$를 알면, 각 점의 타임스탬프 $t_k$에 대해 $[t_s, t_k]$ 구간의 변환을 보간으로 구한다. LOAM 논문은 6자유도 벡터 $[t_x, t_y, t_z, \theta_x, \theta_y, \theta_z]$를 선형 보간하며, 아래 식은 같은 등속 가정을 Lie 군 위에서 쓴 형태다:

$$\mathbf{T}(t_k) = \text{Exp}\left(\frac{t_k - t_s}{t_e - t_s} \cdot \text{Log}(\mathbf{T}_{s \to e})\right)$$

LOAM의 식 (5)는 이 변환을 $\mathbf{p}_k = \mathbf{T}(t_k)\,\tilde{\mathbf{p}}_k$로 쓴다. 곧 스캔 시작 시점 좌표의 점 $\tilde{\mathbf{p}}_k$를 시각 $t_k$의 LiDAR 좌표로 옮기는 변환이다. 따라서 각 점을 스캔 시작 시점 좌표로 되돌리려면 역변환을 쓴다:

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

LeGO-LOAM ([Shan & Englot, 2018](https://doi.org/10.1109/IROS.2018.8594299))은 LOAM에 ground segmentation을 더하고 계산을 줄여 임베디드 보드(Jetson TX2 등)에서도 실시간으로 돈다.

LeGO-LOAM이 더한 것은 다음과 같다.

1. **Ground Segmentation**: 포인트 클라우드를 range image로 바꾼 뒤, 지면(ground)과 비지면을 나눈다. 공개 구현은 인접 빔 사이 기울기가 10° 이하면 지면으로 본다. 지면점은 planar feature로 쓰고, 비지면점에서 edge feature를 뽑는다.

2. **Point Cloud Segmentation**: 비지면점을 range image 위에서 클러스터링하고, 일정 크기보다 작은 클러스터는 노이즈로 보고 지운다. LeGO-LOAM은 이 전처리로 지면 차량 환경에서 쓸 비지면 특징을 고른다.

3. **2단계 LM 최적화**: LOAM은 6-DoF를 한 번에 최적화하지만, LeGO-LOAM은 ground planar feature로 먼저 $[t_z, \theta_{\text{roll}}, \theta_{\text{pitch}}]$를, 이어 edge feature로 $[t_x, t_y, \theta_{\text{yaw}}]$를 최적화한다. 원 논문은 이렇게 나누어 정확도는 비슷하게 두고 계산 시간을 약 35% 줄였다고 보고했다.

4. **포즈 그래프 최적화**: LOAM에 없던 loop closure와 포즈 그래프 최적화를 더해 전역 드리프트를 바로잡는다.

### 7.2.3 왜 LOAM 계열이 오래 살아남았는가

LOAM이 나온 지 10년이 넘었지만 edge/planar 특징 정합은 LIO-SAM(2020) 같은 후속 시스템에도 남아 있다. 이유는 다음과 같다.

1. **기하학적 명확성**: edge/planar feature는 실제 모서리와 평면이라는 기하 원시체(geometric primitive)에 대응한다. 이런 구조 가정은 건물과 도로가 많은 인공 환경에 잘 맞는다.

2. **계산 효율**: 전체 포인트 클라우드 대신 뽑아낸 특징점만 쓰므로 빠르다.

3. **확장성**: LiDAR frontend를 IMU(LIO-SAM)나 카메라(LVI-SAM)와 결합하는 식으로 센서 구성을 확장할 수 있다. KISS-ICP는 센서를 더하지 않고 LiDAR-only ICP를 단순하게 다듬는 쪽을 택했다.

4. **강건성**: edge/planar 분류가 잡음과 불안정한 입사 기하를 걸러 낸다. 곡률 패턴이 일관된 점만 특징으로 남는다. 곡률은 한 스캔 라인의 이웃점에서 계산하는 기하량이므로, 움직이는 차량의 모서리와 평면도 특징으로 뽑힌다. 동적 물체 제거는 별도 단계에서 한다.

한계도 있다. 비반복 주사 패턴을 쓰는 solid-state LiDAR에는 스캔 라인을 전제로 한 특징 추출을 그대로 쓰기 어렵다. FAST-LIO2는 특징 추출을 없애 이 문제를 피한다. 또 기하 특징이 드문 개활지나 곧은 터널에서는 오도메트리 정확도가 크게 떨어진다. 이것은 환경의 관측 기하가 퇴화한 탓이므로 특징을 뽑든 raw 점을 쓰든 남는다.

---

## 7.3 Tightly-Coupled LiDAR-Inertial Odometry

LiDAR만으로는 빠른 모션에서 모션 왜곡이 심해지고, 초기값을 얻기도 어렵다. IMU를 tightly coupled로 결합하면 이 문제가 줄어든다.

### 7.3.1 LIO-SAM

LIO-SAM ([Shan et al., 2020](https://arxiv.org/abs/2007.00258))은 LiDAR, IMU, GPS, loop closure를 한 factor graph에 넣은 LIO 시스템이다. LOAM 계열의 특징 정합을 그래프 최적화와 묶었다.

**Factor Graph 기반 통합**

LIO-SAM은 센서 측정마다 factor graph의 factor를 만든다.

1. **IMU Preintegration Factor**: Forster et al. (2017)의 on-manifold preintegration으로 연속 키프레임 간 IMU 제약을 표현한다. 이 factor는 두 키프레임의 상대 회전, 속도, 위치를 제약하고 바이어스도 함께 추정한다.

2. **LiDAR Odometry Factor**: LOAM 방식의 scan-to-map 매칭으로 상대 포즈를 추정하고, 두 키프레임 사이의 상대 포즈 factor로 넣는다.

3. **GPS Factor**: GPS가 잡히고, 추정한 위치 공분산이 GPS에서 받은 위치 공분산보다 크면 위치 측정을 단항(unary) factor로 넣는다. GPS가 끊긴 구간에서는 LiDAR+IMU factor만으로 돈다.

4. **Loop Closure Factor**: 원 논문은 거리 기반의 단순한 방법으로 루프 후보를 찾고(Scan Context(§9.3.1) 같은 장소 인식으로 바꿀 수 있다고 언급), ICP로 상대 포즈를 추정해 이진(binary) factor로 넣는다.

이 factor들은 GTSAM의 iSAM2로 증분 최적화하는 전역 포즈 그래프에 들어간다. LIO-SAM은 IMU odometry용 그래프를 따로 두고 주기적으로 다시 초기화해, IMU 주기의 상태 전파와 바이어스 추정을 맡긴다. 센서 관측이 저마다 factor이므로 새 센서를 더하기 쉽다.

**IMU 기반 De-skewing**

LIO-SAM에서 IMU의 역할:
1. **모션 왜곡 보정(de-skewing)**: 스캔 동안 들어온 IMU 데이터로 점마다 찍힌 시각의 포즈를 구해 왜곡을 편다.
2. **초기값 예측**: IMU preintegration으로 다음 키프레임 포즈를 예측해 scan-to-map 정합의 초기값으로 쓴다.

거꾸로 LiDAR 정합 결과는 IMU 적분 드리프트를 바로잡고 바이어스를 갱신하는 기준이 된다.

**Keyframe 기반 효율화**

전역 맵 전체 대신 현재 위치 근처와 최근의 키프레임들이 본 점으로 만든 서브맵에 스캔을 정합한다. 그래서 전역 맵에 맞출 때보다 계산이 크게 준다.

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
        # 새 바이어스 변수를 구속하는 랜덤워크 factor. 없으면 update는 오류 없이 돌지만 바이어스에 시간 연속 제약이 없어 B(k)가 IMU factor 하나로만 정해지고 추정이 키프레임마다 튄다
        self.graph.add(gtsam.BetweenFactorConstantBias(
            B(self.key_idx - 1), B(self.key_idx), gtsam.imuBias.ConstantBias(), self.bias_noise))

        # 5. iSAM2 incremental update
        result = self.isam.update(self.graph, self.values)
        self.graph.resize(0)
        self.values.clear()

        self.key_idx += 1
```

### 7.3.2 FAST-LIO / FAST-LIO2

FAST-LIO2 ([Xu et al., 2022](https://doi.org/10.1109/TRO.2022.3141876))는 특징을 뽑지 않고 raw LiDAR 점을 바로 맵에 정합하는 direct LiDAR-inertial odometry다.

**1. Direct Point Registration (특징 추출 제거)**

LOAM은 edge/planar feature를 뽑지만, FAST-LIO2는 (다운샘플한) raw 점을 그대로 쓴다. 점 $\mathbf{p}_k$마다 맵에서 최근접점들로 평면을 맞추고 point-to-plane 거리를 최소화한다.

$$d_k = \mathbf{n}_k^T (\mathbf{T} \cdot \mathbf{p}_k - \mathbf{q}_k)$$

여기서 $\mathbf{n}_k$는 맵 내 최근접 평면의 법선, $\mathbf{q}_k$는 최근접점이다.

특징 추출을 없앤 이유는 다음과 같다.
- 특징 추출은 정보를 버린다. 분류 임계값에 따라 쓸모 있는 점도 빠진다.
- 스캔 패턴(spinning, solid-state, non-repetitive)과 상관없이 쓸 수 있다. Livox 같은 solid-state LiDAR는 비반복 스캔이라 스캔 라인 곡률로 특징을 뽑기 어렵다.
- 맵 자료구조(ikd-Tree)가 충분히 빠르면 raw 점 정합도 실시간으로 돈다.

**2. ikd-Tree (Incremental k-d Tree)**

FAST-LIO2는 맵 자료구조도 바꿨다. 보통의 kd-tree는 정적이라 점을 넣고 빼려면 다시 지어야 한다. ikd-Tree는 다음을 지원한다.

- **점 삽입**: $O(\log N)$ 시간에 새 점을 삽입한다.
- **점 삭제**: lazy delete로 효율적으로 제거한다.
- **동적 re-balancing**: 삽입·삭제로 트리가 한쪽으로 기울면 scapegoat tree 방식으로 그 부분만 다시 짓는다.
- **Box 범위 삭제**: 현재 위치에서 먼 영역의 점을 박스 단위로 지워 맵 크기를 관리한다.

ikd-Tree 덕분에 FAST-LIO2는 맵을 실시간으로 유지하면서 최근접점도 빠르게 찾는다.

**3. Iterated Extended Kalman Filter (IEKF)**

FAST-LIO2는 factor graph 최적화를 쓰는 LIO-SAM과 달리 필터(IEKF)로 상태를 추정한다.

표준 EKF는 관측 모델을 한 번만 선형화하므로 LiDAR 관측의 비선형성이 크면 오차가 커진다. IEKF는 업데이트를 되풀이하며 선형화 지점을 새 추정치로 옮긴다.

$$\hat{\mathbf{x}}^{(k+1)} = \hat{\mathbf{x}}^{-} + \mathbf{K}^{(k)} (\mathbf{z} - h(\hat{\mathbf{x}}^{(k)}) - \mathbf{H}^{(k)}(\hat{\mathbf{x}}^{-} - \hat{\mathbf{x}}^{(k)}))$$

여기서 $k$는 반복 인덱스, $\hat{\mathbf{x}}^{-}$는 prediction 결과, $\mathbf{H}^{(k)}$는 $\hat{\mathbf{x}}^{(k)}$에서 구한 자코비안이다.

칼만 이득:
$$\mathbf{K}^{(k)} = \mathbf{P}^{-} (\mathbf{H}^{(k)})^T (\mathbf{H}^{(k)} \mathbf{P}^{-} (\mathbf{H}^{(k)})^T + \mathbf{R})^{-1}$$

반복은 잔차 감소나 상태 증분이 기준 아래로 떨어지면 멈춘다. FAST-LIO2 공개 구현은 최대 반복 횟수를 작게 두지만(코드 기본값 4회, 배포 launch 파일 3회), 필요한 횟수는 초기값과 장면 기하에 따라 다르다. 이 갱신은 Gauss-Newton과 같은 반복 선형화를 하면서 공분산도 함께 계산한다.

**상태 벡터**:

$$\mathbf{x} = [{}^G\mathbf{R}_I, {}^G\mathbf{p}_I, {}^G\mathbf{v}_I, \mathbf{b}_g, \mathbf{b}_a, {}^I\mathbf{R}_L, {}^I\mathbf{p}_L, \mathbf{g}]$$

표기는 FAST-LIO2 논문을 따른다. $G$는 첫 IMU 프레임으로 잡은 전역 좌표계, $I$는 IMU, $L$은 LiDAR 좌표계다. 왼쪽 위첨자는 기준 좌표계, 오른쪽 아래첨자는 대상 좌표계를 뜻한다(${}^G\mathbf{R}_I$는 $G$에서 본 IMU 자세). 회전 ${}^G\mathbf{R}_I$, 위치 ${}^G\mathbf{p}_I$, 속도 ${}^G\mathbf{v}_I$, 자이로 바이어스 $\mathbf{b}_g$, 가속도계 바이어스 $\mathbf{b}_a$에 더해 LiDAR-IMU extrinsic ${}^I\mathbf{R}_L, {}^I\mathbf{p}_L$과 중력 벡터 $\mathbf{g}$까지 상태에 넣어, 외부 캘리브레이션과 중력 방향도 온라인으로 추정한다.

FAST-LIO2 논문은 저자들의 하드웨어와 데이터에서 최대 100Hz의 odometry·mapping 처리율을 보고한다. 실험에는 multi-line spinning LiDAR와 solid-state LiDAR, 여러 플랫폼과 프로세서를 썼다. 현재 시스템의 처리율은 점 수, CPU, map 크기로 다시 측정해야 한다.

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

[Faster-LIO (Bai et al., 2022)](https://doi.org/10.1109/LRA.2022.3152830)는 FAST-LIO2의 ikd-Tree를 증분 voxel 구조(iVox)로 바꿔 처리 속도를 높인다.

점은 해시 맵으로 찾는 희소 voxel에 담는다. 질의점이 든 voxel과 이웃 voxel(6·18·26개)에서만 근사 k-NN을 찾고, 그 이웃점들로 평면을 맞춘다. voxel 찾기는 해시로 $O(1)$이라 트리 검색보다 빠르다.

### 7.3.4 Point-LIO

Point-LIO ([He et al., 2023](https://doi.org/10.1002/aisy.202200459))는 FAST-LIO 시리즈를 점 단위로 밀고 나가, 점이 하나 들어올 때마다 상태를 갱신한다.

앞에서 본 LIO들은 스캔 하나(~100ms)를 한 관측으로 처리한다. 스캔 안의 왜곡은 IMU 적분으로 각 점 시각의 포즈를 구해 보정하지만(§7.3.1), 갱신 자체는 스캔 단위이므로 그 구간을 하나의 운동으로 묶는 가정이 남는다. 고속·고각속도 모션에서는 이 가정이 깨진다.

Point-LIO는 점 timestamp 순서로 상태를 전파하고 point-wise update를 적용한다. 고주파 IMU와 LiDAR 점의 timestamp로 각 관측 시각의 상태를 추정해 scan-level deskew의 단일 운동 가정보다 시간 해상도가 높다. 다만 이 상태도 IMU noise·bias·동기화 오차를 안은 추정치다.

Point-LIO의 상태 전파는 IMU 측정 사이의 짧은 시간 간격에서 다음 연속 모델을 이산화한다:

$$\frac{d}{dt}\mathbf{R} = \mathbf{R}[\boldsymbol{\omega}]_\times, \quad \frac{d}{dt}\mathbf{v} = \mathbf{R}\mathbf{a} + \mathbf{g}, \quad \frac{d}{dt}\mathbf{p} = \mathbf{v}$$

점 하나가 올 때마다 state propagation → single-point update를 하므로 연속 시간(continuous-time) 필터에 가깝다.

장점: 아주 빠른 모션에서도 추정이 버틴다. 저자 실험은 각속도 75 rad/s(약 4300°/s)까지 다뤘다. 점마다 그 시각의 상태로 처리하므로 모션 왜곡을 따로 보정하지 않는다.

단점: 점 수에 따라 업데이트 부담이 늘어난다. 논문이 보고한 FAST-LIO2 대비 처리 시간은 그 데이터·하드웨어·설정에서 잰 값이므로, 쓰려는 구현에서 다시 재야 한다.

### 7.3.5 COIN-LIO

[COIN-LIO (Pfreundschuh et al., 2024)](https://arxiv.org/abs/2310.01235)는 LiDAR-Inertial 시스템에 LiDAR 반사 강도(intensity)를 더한다. 반사 강도를 intensity image로 투영하고, 영상 안과 스캔 사이의 밝기가 고르도록 필터링한다.

포인트 클라우드 정합만으로 제약이 약한 방향을 찾고, 그 방향을 메울 intensity image 패치를 고른다. 고른 패치의 photometric residual을 IMU 측정, point-to-plane residual과 함께 iterated EKF에 넣는다.

긴 터널이나 평탄한 개활지처럼 기하학적으로 퇴화한 환경에서는 LiDAR 반사 강도의 공간적 변화가 추가 제약이 될 수 있다. 이 제약의 효과는 해당 방향에서 관측할 수 있는 intensity 패턴에 달려 있다.

---

## 7.4 Continuous-Time LiDAR Odometry

앞의 LiDAR 오도메트리는 이산 시간(discrete-time) 모델을 쓴다. 스캔마다 포즈를 하나 두고, 스캔 안의 모션은 등속 보간으로 근사한다. Continuous-time 방법은 궤적을 연속 함수로 나타내, 스캔 안의 모션도 추정 대상으로 삼는다.

### 7.4.1 CT-ICP

CT-ICP ([Dellenbach et al., 2022](https://arxiv.org/abs/2109.12979))는 스캔마다 포즈 두 개(스캔 시작과 끝)를 둔다.

스캔 안 점의 타임스탬프 $t_k \in [t_s, t_e]$마다 $\alpha_k = \frac{t_k - t_s}{t_e - t_s}$로 회전은 구면 선형 보간(slerp), 위치는 선형 보간한다.

$$\mathbf{R}(t_k) = \text{slerp}(\mathbf{R}_s, \mathbf{R}_e, \alpha_k), \quad \mathbf{t}(t_k) = (1-\alpha_k)\mathbf{t}_s + \alpha_k\mathbf{t}_e$$

두 포즈 $\mathbf{T}_s, \mathbf{T}_e$를 함께 최적화하므로, 앞 절의 고정된 등속 보간과 달리 스캔 안의 모션도 최적화 중에 다듬어진다.

IMU 없이도 모션 왜곡을 보정하므로 IMU가 없는 시스템에 쓸모 있다.

### 7.4.2 B-Spline 기반 궤적 표현

더 일반적인 continuous-time 방법은 궤적을 B-spline으로 나타낸다. B-spline은 제어점(control point) $\{\mathbf{T}_i\}$가 정하는 매끄러운 곡선이다.

$$\mathbf{T}(t) = \mathbf{T}_0 \prod_{i=1}^{k} \text{Exp}\left(\tilde{B}_i(t) \cdot \text{Log}(\mathbf{T}_{i-1}^{-1}\mathbf{T}_i)\right)$$

여기서 $\tilde{B}_i(t)$는 누적(cumulative) B-spline 기저 함수(basis function)다. 3차(cubic) B-spline을 쓰면 $C^2$ 연속이 된다.

B-spline 궤적은 어떤 시점 $t$에서든 평가해 포즈를 얻을 수 있고, 미분하면 속도와 가속도가 나온다. 그래서 시각이 제각각인 비동기 센서 데이터를 그대로 넣을 수 있다. $C^2$ 연속이라고 해서 동역학적으로 실현 가능한 궤적이라는 보장은 없다. 국소 지지 덕분에 한 제어점을 바꾸면 그 제어점이 걸친 몇 구간만 바뀐다.

단점:
- 제어점 간격(knot spacing)이 주요 하이퍼파라미터다. 간격이 너무 조밀하면 과적합된다. 간격이 너무 넓으면 고속 모션이 뭉개진다.
- 이산 시간 모델보다 계산이 많다.

Kalibr(§3.4.2)의 camera-IMU 캘리브레이션도 B-spline 궤적을 쓴다.

---

## 7.5 Solid-State LiDAR 특화

Solid-state LiDAR(Livox 시리즈 등)는 회전형 LiDAR와 스캔 패턴이 다르다.

**회전형 vs Solid-state**:

| 특성 | 회전형 (Velodyne, Ouster) | Solid-state (Livox) |
|------|--------------------------|---------------------|
| 스캔 패턴 | 반복적 (매 회전 같은 패턴) | 비반복적 (꽃잎/로즈 패턴) |
| FoV | 360° 수평 | 모델별 (Mid-40 38.4° 원형, Avia 70.4°×77.2°, Horizon 81.7°×25.1°) |
| 점 밀도 | 균등 | 시간에 따라 누적, 불균등 |
| 가격 | 높음 | 낮음 |
| 크기/무게 | 큼 | 작음 |

**비반복 스캔이 특징 추출에 미치는 영향**

LOAM 방식의 곡률 기반 특징 추출은 같은 스캔 라인의 이웃점을 쓴다. solid-state LiDAR의 점은 스캔 라인 없이 비규칙적으로 분포한다. 그래서 KNN(K-Nearest Neighbors) 기반 국소 곡률을 쓰거나, raw 점을 그대로 정합에 쓴다.

**FAST-LIO2가 solid-state에 강한 이유**

FAST-LIO2는 raw 점을 직접 정합하므로 어떤 스캔 패턴에도 그대로 쓸 수 있다. 1세대 FAST-LIO는 edge/planar 특징을 추출했고, FAST-LIO2가 특징 추출 단계를 없앴다. 비반복 스캔 LiDAR는 적분 시간이 길수록 FoV를 더 조밀하게 채운다. FAST-LIO2의 ikd-Tree는 점을 증분 삽입하므로 이렇게 점차 조밀해지는 맵을 그대로 담는다. 좁은 FoV로 생기는 한 스캔의 정보 부족은 IMU와의 tight coupling으로 메운다.

Livox 계열은 비반복 스캔 패턴을 쓰고 폼팩터가 작다. 이 장에서 다룬 FAST-LIO2와 Point-LIO의 공개 저장소에는 Livox Avia용 설정 파일(avia.yaml)이 들어 있다. 실제 선택에서는 가격뿐 아니라 FoV, 거리, 시간 동기화, 점 분포와 목표 플랫폼을 함께 비교한다.

---

## 7.6 학습 기반 LiDAR Odometry

### 7.6.1 DeepLO 계열

학습 기반 LiDAR 오도메트리는 포인트 클라우드 쌍을 받아 상대 포즈를 내는 네트워크를 학습한다.

대표 방법은 다음과 같다.
- **LO-Net** (Li et al., 2019): LiDAR 스캔을 2D range image로 바꾸고 CNN이 뽑은 특징으로 포즈를 예측한다. 법선 추정과 마스크 예측을 보조 과제로 함께 학습한다.
- **DeepLO** (Cho et al., 2020): 구면 투영한 vertex map·normal map을 CNN에 넣고, point-to-plane ICP 잔차 형태의 기하 인지 손실로 비지도 학습한다.
- **PWCLO-Net** (Wang et al., 2021): Pyramid, Warping, Cost volume 구조를 LiDAR 오도메트리에 적용한다.

### 7.6.2 현재의 한계

학습 기반 LiDAR 오도메트리가 부딪히는 어려움은 다음과 같다.

1. **LiDAR 데이터의 특성**: 포인트 클라우드는 비정형(unstructured)이고 순서가 없는 집합이라, 격자 입력을 전제로 한 CNN에 바로 넣기 어렵다.

2. **강한 기하 기준선**: ICP/GICP/NDT는 충분한 중첩과 구조가 있을 때 학습 없이도 강한 기준선이 된다. 반복 구조, 동적 객체, 강수·먼지, 희소·저중첩 장면에서는 이들도 실패할 수 있어서, 학습은 대응점·동적점·불확실성 추정 같은 보조 역할에 쓰인다.

3. **데이터 부족**: LiDAR 오도메트리 학습 데이터는 이미지 데이터보다 훨씬 적다.

4. **일반화**: 한 LiDAR·환경에서 학습한 모델은 다른 LiDAR·환경에서 성능이 떨어진다.

학습은 오도메트리 밖의 보조 구성 요소에도 쓰인다. Loop closure 검출(PointNetVLAD), 정합 초기값 추정(GeoTransformer, §5.8.5), 동적 물체 제거용 시맨틱 분할이 그 예다. Loop closure 검출에 쓰는 Scan Context는 학습 없이 손으로 설계한 디스크립터다.

---

## 7.7 최근 동향 (2023-2024)

**[KISS-ICP (Vizzo et al., 2023)](https://arxiv.org/abs/2209.15397)**: Point-to-point ICP에 적응적 임계값, 강건 커널, 모션 보상을 더했다. 원 논문은 자동차·UAV·핸드헬드 데이터셋 전반에서 파라미터 세트 하나로 경쟁력 있는 궤적을 얻었다고 보고했다.

**[MAD-ICP (Ferrari et al., 2024)](https://arxiv.org/abs/2405.05828)**: PCA 기반 kd-tree로 포인트 클라우드의 구조 정보를 뽑아 point-to-plane 정합에 쓴다. 데이터 매칭 방식에 초점을 두며, 원 논문은 여러 작동 조건에서 특정 도메인용 방법과 비슷한 성능을 보고했다.

**[iG-LIO (Chen et al., 2024)](https://github.com/zijiechenrobotics/ig_lio)**: Incremental GICP를 tightly-coupled LIO에 넣은 시스템이다. Voxel 기반 표면 공분산 추정기(VSCE)로 GICP의 공분산 계산 효율을 높이고, incremental voxel map으로 최근접점 검색 비용을 줄였다. 논문은 기계식·고체 LiDAR 데이터셋에서 모든 데이터셋에 같은 파라미터를 쓰고도 Faster-LIO보다 효율이 높고 최신 LIO와 비슷한 정확도를 냈다고 보고했다.

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
| KISS-ICP | Direct (P2P) | 반복 최적화 | LiDAR only | 적응적 임계값, 단일 파라미터 세트 |
| MAD-ICP | Direct (P2Plane) | 반복 최적화 | LiDAR only | PCA 기반 구조 추출, 데이터 매칭 중심 |
| iG-LIO | Direct (GICP) | IEKF | LiDAR + IMU | Incremental GICP, voxel 공분산 추정 |

LOAM(2014) → LeGO-LOAM(2018) → LIO-SAM(2020)은 feature-based + factor graph 쪽으로, FAST-LIO(2021) → FAST-LIO2(2022) → Point-LIO(2023)은 direct + Kalman filter 쪽으로 발전했다. 어느 쪽이 더 정확하거나 빠른지는 sensor pattern, motion, map scale, 하드웨어, benchmark protocol에 따라 달라진다.
