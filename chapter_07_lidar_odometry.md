# 7장 LiDAR 오도메트리와 LIO

카메라 기반 VO와 VIO(→ 6장)는 풍부한 텍스처 위에서 정밀한 궤적을 그려 내지만, 가시광에 의존하는 수동 센서인 탓에 어두운 밤길이나 급격한 조도 변화 앞에서는 취약해지고 단안 기하만으로는 절대 거리를 직접 잴 수 없다. 같은 자기 운동 추정 문제라도 센서를 능동 3D 측량기인 LiDAR로 바꾸면 조건이 근본적으로 달라진다. LiDAR는 레이저 비행시간(ToF)으로 주변 3D 좌표를 직접 측정해 조명 변화에 흔들리지 않는 대신, 비·안개 같은 기상 조건이나 유리·금속의 반사율, 다중경로(multipath) 반사에 영향을 받는다. 연속된 3D 스캔만으로 센서 자신의 움직임을 추적하는 **LiDAR Odometry(LO)**와, 여기에 IMU 관성 측정을 결합해 모션 왜곡과 고속 기동을 보완하는 **LiDAR-Inertial Odometry(LIO)**가 바로 이 축을 이룬다.

LiDAR 오도메트리의 핵심은 연속된 두 포인트 클라우드 스캔을 가장 잘 겹치는 강체 변환 $\mathbf{T} \in SE(3)$을 구하는 포인트 클라우드 정합(registration)이다. 언뜻 2D 이미지 매칭보다 3D 기하가 단순해 보이지만, 실제로는 점과 표면 사이의 데이터 연관(correspondence), 센서 노이즈 모델링, 수만 개 점의 검색 속도, 그리고 스캔 도중 센서가 움직여 생기는 모션 왜곡 보정이 촘촘하게 얽혀 있다.

---

## 7.1 Point Cloud Registration 기초

모든 LiDAR 오도메트리 파이프라인의 바탕에는 두 포인트 클라우드 사이의 거리 총합을 최소화하는 최적화 문제가 있다. 하나는 움직여 맞출 소스 $\mathcal{P} = \{\mathbf{p}_i\}$이고, 다른 하나는 기준이 되는 타겟 $\mathcal{Q} = \{\mathbf{q}_j\}$다.

$$\mathbf{T}^* = \underset{\mathbf{T} \in SE(3)}{\arg\min} \sum_i d(\mathbf{T} \cdot \mathbf{p}_i, \mathcal{Q})$$

여기서 $d(\cdot, \cdot)$는 변환된 소스 점과 타겟 포인트 클라우드 사이의 어긋남을 재는 거리 메트릭이다. 점과 점 사이의 직선 거리를 쓸지, 국소 표면의 법선이나 확률 분포까지 반영할지에 따라 여러 ICP 변종과 확률적 정합 기법이 갈라진다.

### 7.1.1 ICP 변종

**Point-to-Point ICP ([Besl & McKay, 1992](https://doi.org/10.1109/34.121791))**

가장 직관적인 출발점은 변환한 소스 점 $\mathbf{T}\cdot\mathbf{p}_i$와 타겟 클라우드에서 가장 가까운 점 $\mathbf{q}_{c(i)}$ 사이의 유클리드 거리를 직접 줄이는 방식이다.

$$\mathbf{T}^* = \underset{\mathbf{T}}{\arg\min} \sum_i \left\|\mathbf{T} \cdot \mathbf{p}_i - \mathbf{q}_{c(i)}\right\|^2$$

여기서 $c(i) = \arg\min_j \|\mathbf{T} \cdot \mathbf{p}_i - \mathbf{q}_j\|$는 현재 변환을 기준으로 찾은 최근접점(closest point) 인덱스다. 처음에는 올바른 짝 $c(i)$조차 모르기 때문에, ICP는 대응 탐색과 변환 추정을 번갈아 반복한다.

1. **대응 찾기**: 현재 추정 변환 $\mathbf{T}$로 소스 점들을 옮긴 뒤 타겟에서 가장 가까운 점을 찾는다. 타겟 클라우드를 kd-tree로 색인해 두면 점 $N$개 전체의 대응을 $O(N\log N)$ 시간에 찾을 수 있다.
2. **변환 추정**: 대응 쌍이 정해지면 그 쌍들에 대해서는 반복 없이 closed-form으로 최적 강체 변환이 구해진다. SVD로 푸는 표준 절차는 다음과 같다.

   먼저 두 대응 점 집합에서 각각 무게중심을 빼 평행이동 성분을 분리한다.
   $$\bar{\mathbf{p}} = \frac{1}{N}\sum_i \mathbf{p}_i, \quad \bar{\mathbf{q}} = \frac{1}{N}\sum_i \mathbf{q}_{c(i)}$$

   중심을 맞춘 두 점군 사이의 교차 공분산 행렬을 만든다.
   $$\mathbf{W} = \sum_i (\mathbf{p}_i - \bar{\mathbf{p}})(\mathbf{q}_{c(i)} - \bar{\mathbf{q}})^T$$

   이 행렬을 $\mathbf{W} = \mathbf{U}\boldsymbol{\Sigma}\mathbf{V}^T$로 SVD 분해하면 최적 회전과 이동이 바로 나온다.
   $$\mathbf{R}^* = \mathbf{V} \text{diag}(1, 1, \det(\mathbf{V}\mathbf{U}^T)) \mathbf{U}^T, \quad \mathbf{t}^* = \bar{\mathbf{q}} - \mathbf{R}^*\bar{\mathbf{p}}$$

   여기서 $\det(\mathbf{V}\mathbf{U}^T) = 1$이면 $\mathbf{R}^* = \mathbf{V}\mathbf{U}^T$ 그대로 유효한 회전 행렬이고, 만약 노이즈나 대칭 탓에 $\det(\mathbf{V}\mathbf{U}^T) = -1$이 나오면 거울 반사(reflection)를 막기 위해 $\mathbf{V}$의 마지막 열 부호를 뒤집어 $SO(3)$ 위에 머물게 한다.

하지만 Point-to-Point ICP를 실제 LiDAR 스캔에 그대로 쓰면 세 가지 약점이 도드라진다.
- **평면 위 슬라이딩**: 벽이나 바닥처럼 넓은 평면 위에서는 두 스캔이 접선 방향으로 조금 미끄러져도, 매 반복 새로 찾는 최근접점이 옆 점으로 갈아타면서 비용 함수가 평면 방향으로는 거의 평평해지고 수렴이 극도로 느려진다(대응을 고정한 한 스텝 안에서는 접선 이동도 비용을 키우지만, 반복 전체로 보면 미끄러짐이 풀리지 않는다).
- **초기값 의존성**: 최근접점 가정이 국소적으로만 유효하므로 초기 포즈가 조금만 어긋나도 엉뚱한 로컬 최소값에 걸린다.
- **이산 샘플링 불일치**: 두 스캔의 레이저 빔이 정확히 같은 물리적 지점에 꽂히지는 않으므로, 실제로 같은 표면 위에 놓인 점이라도 최근접점 대응 자체에 샘플링 간격만큼의 오차가 낀다.

**Point-to-Plane ICP**

평면 위의 점을 맞출 때는 서로 어긋난 두 점 사이의 거리를 억지로 좁히기보다, 소스 점에서 타겟이 이루는 국소 접평면까지의 수직 거리만 0으로 만드는 편이 기하학적으로 훨씬 자연스럽다.

$$\mathbf{T}^* = \underset{\mathbf{T}}{\arg\min} \sum_i \left((\mathbf{T} \cdot \mathbf{p}_i - \mathbf{q}_{c(i)})^T \mathbf{n}_{c(i)}\right)^2$$

여기서 $\mathbf{n}_{c(i)}$는 타겟 점 $\mathbf{q}_{c(i)}$에서 추정한 표면 법선(surface normal)이다. 오차 벡터를 법선 방향으로 사영(projection)한 성분만 비용으로 계산하기 때문에, 평면 위를 따라 옆으로 미끄러져도 잔차가 늘지 않아 빔 샘플링 불일치의 영향을 받지 않는다. 덕분에 실내 복도나 도심 건물처럼 평면 구조가 많은 환경에서는 Point-to-Point보다 훨씬 적은 반복으로 빠르게 수렴한다.

대신 closed-form SVD 해가 사라지므로 매 반복 Gauss-Newton 같은 비선형 반복 최적화를 돌려야 하며, 최종 정합 품질 역시 타겟 점마다 추정한 표면 법선이 얼마나 정확한지에 좌우된다. 보통 법선은 점 주변의 $k$개 이웃점을 모아 PCA(주성분 분석)를 수행해 구한다. 이웃점들의 공분산 행렬을 세운 뒤,

$$\mathbf{C} = \frac{1}{k}\sum_{j \in \mathcal{N}(i)} (\mathbf{q}_j - \bar{\mathbf{q}})(\mathbf{q}_j - \bar{\mathbf{q}})^T$$

고유값 분해 $\mathbf{C} = \mathbf{V}\boldsymbol{\Lambda}\mathbf{V}^T$를 수행하면 점들이 가장 얇게 퍼져 있는 방향, 곧 최소 고유값 $\lambda_{\min}$에 대응하는 고유벡터가 표면 법선으로 얻어진다.

### 7.1.2 GICP (Generalized ICP)

Point-to-Point ICP가 모든 점의 오차를 방향성 없는 구형으로 보고 Point-to-Plane ICP가 타겟 쪽에만 무한 평면 제약을 건다면, [GICP (Segal et al., 2009)](https://doi.org/10.15607/RSS.2009.V.021)는 소스와 타겟 양쪽 점 모두에 국소 표면 공분산 $\mathbf{C}_i$를 부여해 두 변종과 plane-to-plane ICP를 하나의 확률적 프레임워크로 통합한다.

$$\mathbf{T}^* = \underset{\mathbf{T}}{\arg\min} \sum_i (\mathbf{T} \cdot \mathbf{p}_i - \mathbf{q}_{c(i)})^T (\mathbf{C}_i^{\mathcal{Q}} + \mathbf{R}\mathbf{C}_i^{\mathcal{P}}\mathbf{R}^T)^{-1} (\mathbf{T} \cdot \mathbf{p}_i - \mathbf{q}_{c(i)})$$

여기서 $\mathbf{C}_i^{\mathcal{P}}, \mathbf{C}_i^{\mathcal{Q}}$는 각각 소스와 타겟 점 주변의 국소 표면 형상을 담은 $3 \times 3$ 공분산 행렬이다.

이 공분산의 고유값 구조를 조절하면 물리적 의미가 선명하게 드러난다. 평면 위의 점은 법선 방향으로는 거의 두께가 없고 접선 두 방향으로는 넓게 퍼져 있으므로, 첫 번째 열이 법선($\mathbf{R}_s \mathbf{e}_1 = \mathbf{n}$)인 회전 행렬 $\mathbf{R}_s$를 써서 $\mathbf{C} = \mathbf{R}_s \text{diag}(\epsilon, 1, 1) \mathbf{R}_s^T$ ($\epsilon \ll 1$) 형태로 정규화한다. $\mathbf{C}$의 고유벡터가 $\mathbf{R}_s$의 열이므로 법선 방향 분산만 아주 작은 값 $\epsilon$으로 눌리고, 양쪽 점에 모두 이 공분산을 쓰면 GICP는 자연스럽게 plane-to-plane 정합으로 동작한다. 실제로 원 논문 3절은 $\mathbf{C}^{\mathcal{P}} = \mathbf{0}$, $\mathbf{C}^{\mathcal{Q}} = \mathbf{I}$를 대입하면 Point-to-Point가 되고, $\mathbf{C}^{\mathcal{P}} = \mathbf{0}$에 $(\mathbf{C}^{\mathcal{Q}})^{-1}$을 법선 방향 사영 행렬 $\mathbf{P}_i$로 두면 Point-to-Plane으로 환원됨을 보였다.

원 논문의 실험에서 GICP는 두 기준 기법보다 정확도가 더 높았고 잘못된 대응에도 덜 흔들렸다. 다만 실제 주행 성능은 초기 포즈 오차, 스캔 중첩률, 점 샘플링 밀도, 아웃라이어 제거 기준에 따라 달라진다.

아래는 GICP의 핵심 루프를 담은 Python 수도코드다. 사전 계산한 국소 공분산과 kd-tree 대응 탐색을 바탕으로 $SE(3)$ 위에서 Gauss-Newton을 반복한다.

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

점 하나하나마다 최근접점을 뒤지는 ICP 계열과 달리, [NDT (Biber & Strasser, 2003)](https://doi.org/10.1109/IROS.2003.1249285)는 3D 공간을 규칙적인 복셀(voxel) 격자로 쪼갠 뒤 각 복셀 안에 담긴 점들의 분포를 하나의 가우시안 확률 분포로 요약해 정합한다.

**NDT 절차**

1. **타겟 포인트 클라우드의 NDT 표현 만들기**: 공간을 일정 크기의 3D 복셀 격자로 분할하고, 점이 충분히 들어 있는 복셀 $k$마다 평균 $\boldsymbol{\mu}_k$와 공분산 $\boldsymbol{\Sigma}_k$로 가우시안 분포 $\mathcal{N}(\boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k)$를 구한다.
   $$\boldsymbol{\mu}_k = \frac{1}{n_k}\sum_{i \in k} \mathbf{q}_i, \quad \boldsymbol{\Sigma}_k = \frac{1}{n_k-1}\sum_{i \in k} (\mathbf{q}_i - \boldsymbol{\mu}_k)(\mathbf{q}_i - \boldsymbol{\mu}_k)^T$$
2. **변환 최적화**: 변환된 소스 점 $\mathbf{T}\cdot\mathbf{p}_i$가 자신이 떨어진 타겟 복셀 $k(i)$의 가우시안 분포 위에서 높은 가능도(likelihood)를 얻도록 변환 $\mathbf{T}$를 최적화한다.
   $$\mathbf{T}^* = \underset{\mathbf{T}}{\arg\max} \sum_i \exp\left(-\tfrac{1}{2}(\mathbf{T} \cdot \mathbf{p}_i - \boldsymbol{\mu}_{k(i)})^T \boldsymbol{\Sigma}_{k(i)}^{-1} (\mathbf{T} \cdot \mathbf{p}_i - \boldsymbol{\mu}_{k(i)})\right)$$

여기서 한 가지 중요한 수학적 미묘함이 있다. 위 식처럼 지수 함수 형태의 가우시안 값을 그대로 더하면, 분포 중심에서 멀리 벗어난 아웃라이어의 점수는 0으로 포화되어 기울기가 사라진다. 만약 이 식을 로그 가능도의 합, 즉 마할라노비스 거리 제곱합
$$\sum_i (\mathbf{T} \cdot \mathbf{p}_i - \boldsymbol{\mu}_{k(i)})^T \boldsymbol{\Sigma}_{k(i)}^{-1} (\mathbf{T} \cdot \mathbf{p}_i - \boldsymbol{\mu}_{k(i)})$$
의 최소화 문제로 바꾸면, 아웃라이어의 페널티가 끝없이 커져서 원래 NDT와는 전혀 다른 복셀 기반 마할라노비스 ICP가 된다. 실제 NDT 구현체는 가우시안 분포에 작은 균등 분포를 섞은 혼합 모델을 근사해 아웃라이어 내성과 수치 안정성을 함께 챙긴다.

NDT는 반복마다 kd-tree에서 점 대응을 찾지 않고, 변환된 점이 어느 복셀 칸에 떨어지는지만 조회하면 되므로 검색 비용이 가볍다. 복셀 크기를 조절해 큰 복셀로 넓은 수렴 영역을 확보한 뒤 작은 복셀로 정밀도를 높이는 전략을 쓰기도 좋다. 또한 점수 함수가 매끄럽고 멀리 떨어진 점의 기여가 포화되어 아웃라이어에도 덜 끌린다.

반면 복셀 크기 선택에 성능이 민감하고, 점이 몇 개 안 들어 있는 복셀에서는 공분산 행렬 $\boldsymbol{\Sigma}_k$가 특이 행렬에 가까워져 불안정해진다. 게다가 실패할 때의 파급력도 크다. Kvarntorp 광산 데이터에서 ICP와 NDT를 비교한 [Magnusson et al. (2008)](http://ais.informatik.uni-freiburg.de/3dmapping_ws_iros08/abstracts/magnusson.pdf)의 실험을 보면, NDT는 ICP보다 넓은 초기 오차 범위에서 수렴했지만 일단 정합에 실패했을 때는 이동과 회전이 한꺼번에 크게 틀어져 ICP보다 나쁜 결과를 낸 경우도 있었다.

오늘날 3D NDT는 Autoware의 `ndt_scan_matcher`처럼 미리 구축한 정밀 3D 지도 위에서 자율주행 차량이 현재 위치를 찾는 localization 모듈에서 널리 쓰인다.

### 7.1.4 수렴성과 초기값 의존성

지금까지 살펴본 ICP, GICP, NDT는 모두 현재 위치 주변에서 경사를 따라 내려가는 국소 최적화 알고리즘이다. 따라서 초기 변환 $\mathbf{T}_{\text{init}}$가 정답에서 크게 벗어나 있으면 엉뚱한 로컬 미니멈에 갇히고 만다. 실전 시스템에서는 주로 다음 네 가지 방법으로 안전한 초기값을 구한다.

1. **등속 모델**: 직전 두 스캔 사이의 상대 변환을 그대로 외삽한다. 가장 단순하고 추가 센서가 필요 없지만, 차량이 급제동하거나 급선회하는 순간에는 초기값이 틀어진다.
2. **IMU 적분**: 스캔 사이의 짧은 시간 동안 고주파 IMU 각속도와 가속도를 적분하면 급격한 회전에서도 정확한 초기 포즈를 얻을 수 있다. LIO가 순수 LO보다 훨씬 강건한 첫 번째 이유다.
3. **멀티 해상도 (Multi-resolution)**: 복셀 크기를 크게 잡거나 포인트 클라우드를 거칠게 다운샘플링해 먼저 대략적인 정렬을 마친 뒤, 해상도를 높여 세밀하게 다듬는 coarse-to-fine 전략이다.
4. **RANSAC + 특징 매칭**: FPFH나 SHOT 같은 3D 기하 디스크립터로 대응을 찾고 RANSAC으로 초기 변환을 구한다. 초기 포즈에 대한 사전 단서가 전혀 없는 전역 정합에서는 유용하지만, 스캔 사이 움직임이 작은 오도메트리에서는 등속 모델이나 IMU 적분이 훨씬 빠르고 실용적이다.

---

## 7.2 Feature-based LiDAR Odometry

ICP나 GICP는 전 스캔에 대해 매 반복 kd-tree로 최근접점을 뒤진다. 정확도는 높지만, LiDAR가 초당 수만에서 수십만 개의 점을 뱉기 때문에 연산 비용이 버겁고, 나뭇잎이나 거친 지면 같은 불안정한 점들이 정합을 흔들기도 한다. 영상 기반 VO가 코너점만 추려 연산과 노이즈를 함께 줄였던 것처럼, **LOAM**은 3D 포인트 클라우드에서 기하학적으로 의미가 뚜렷한 모서리와 평면 점만 추려 내는 해법을 제시했다.

### 7.2.1 LOAM (Lidar Odometry and Mapping in Real-time)

[LOAM (Zhang & Singh, 2014)](https://doi.org/10.15607/RSS.2014.X.007)은 이후 등장한 LeGO-LOAM, LIO-SAM, A-LOAM 같은 수많은 LiDAR 오도메트리 시스템의 원형이 되었다. LOAM이 실시간성과 높은 정확도를 동시에 거머쥔 비결은 두 가지 설계 결정에 있다.

1. 스캔 전체 대신 기하학적으로 구조가 선명한 **edge**와 **planar** 특징점만 추출해 정합 점 수를 대폭 줄인다.
2. 높은 빈도로 거칠게 모션을 쫓는 **odometry** 스레드와, 낮은 빈도로 누적 지도에 맞춰 드리프트를 깎아 내는 **mapping** 스레드로 작업을 둘로 가른다.

**특징점 추출**

회전형 다채널 LiDAR는 각 레이저 채널이 수평으로 한 바퀴를 돌며 점들을 순서대로 찍어 내므로, 같은 스캔 라인(scan line) 위에서 이웃한 점들 사이의 3차원 배치만 살펴봐도 표면이 날카로운지 평평한지 가려낼 수 있다. LOAM은 점 $\mathbf{p}_i$를 둘러싼 같은 스캔 라인의 좌우 이웃 집합 $\mathcal{S}_i$(공개 구현 기준 좌우 5개씩 총 10개)에 대해 국소 곡률(curvature)을 정의한다.

$$c_i = \frac{1}{|\mathcal{S}_i| \cdot \|\mathbf{p}_i\|} \left\| \sum_{j \in \mathcal{S}_i} (\mathbf{p}_j - \mathbf{p}_i) \right\|$$

좌우 이웃 점들이 한 평면 위에 나란히 놓여 있으면 대칭 위치의 차이 벡터 $(\mathbf{p}_j - \mathbf{p}_i)$들이 서로 상쇄되어 $c_i$가 0에 가까워지고, 반대로 건물 모서리처럼 급격히 꺾이는 곳에서는 벡터 합이 커져 $c_i$가 치솟는다. 분모에 센서 거리 $\|\mathbf{p}_i\|$를 둔 것은 거리가 멀수록 빔 간격이 벌어지는 기하 효과를 정규화해 가까운 점과 먼 점을 동일한 척도로 비교하기 위해서다.

- **Edge feature**: 곡률이 큰 점 ($c_i > c_{\text{thresh}}^e$). 건물 모서리나 기둥처럼 뾰족한 경계선 위에 놓인다.
- **Planar feature**: 곡률이 작은 점 ($c_i < c_{\text{thresh}}^p$). 벽면이나 아스팔트 바닥처럼 매끈한 평면을 이룬다.

추출한 점들이 한쪽 구조물에만 몰리지 않고 정합을 망가뜨리지 않도록 몇 가지 필터링 규칙을 덧붙인다. 스캔 라인을 4개 구간으로 나누어 구간마다 일정 개수씩 고르게 뽑고, 이미 뽑힌 특징점 바로 옆의 점은 non-maximum suppression으로 건너뛴다. 또한 레이저 빔이 거의 평행하게 스쳐 가는 표면에서는 거리 오차가 크고, 앞 물체에 가려져 생기는 가림(occlusion) 경계에서는 시점이 조금만 바뀌어도 한쪽 점이 사라진다. 이 때문에 빔과 표면이 이루는 입사 기하를 따져 이런 불안정한 점들도 미리 걸러 낸다.

**Odometry 모듈 (~10Hz)**

빠른 Odometry 스레드는 직전 스캔의 특징점과 현재 스캔의 특징점을 맞추는 scan-to-scan 정합을 수행한다. 이때 모든 점에 똑같은 유클리드 거리를 쓰지 않고, edge 점은 직선에, planar 점은 평면에 맞추도록 거리 메트릭을 나눈다.

**Edge point-to-edge distance**: 현재 스캔의 edge 점 $\mathbf{p}$마다 이전 스캔에서 가장 가까운 edge 점 $\mathbf{a}$를 찾고, 이어서 $\mathbf{a}$와 **인접한 스캔 라인**에서 가장 가까운 edge 점 $\mathbf{b}$를 하나 더 찾는다(같은 스캔 라인의 두 점을 고르면 수평면 위의 엉뚱한 직선이 되거나 직선이 퇴화할 수 있다). 이때 $\mathbf{p}$에서 공간 직선 $\overline{\mathbf{ab}}$까지의 수직 거리는 삼각형의 높이에 해당하므로, 두 차이 벡터의 외적 크기를 밑변 길이로 나누어 구한다.

$$d_e = \frac{\|(\mathbf{p}-\mathbf{a}) \times (\mathbf{p}-\mathbf{b})\|}{\|\mathbf{a}-\mathbf{b}\|}$$

**Planar point-to-plane distance**: 현재 스캔의 planar 점 $\mathbf{p}$마다 이전 스캔에서 평면을 결정할 점 세 개를 뽑는다. 가장 가까운 planar 점 $\mathbf{a}$, $\mathbf{a}$와 같은 스캔 라인의 최근접 점 $\mathbf{b}$, 그리고 인접 스캔 라인의 최근접 점 $\mathbf{c}$를 골라야 세 점이 일직선에 놓이는 퇴화를 피하고 온전한 평면 $\triangle\mathbf{abc}$를 세울 수 있다. $\mathbf{p}$에서 이 평면까지의 거리는 법선 벡터 $(\mathbf{a}-\mathbf{b}) \times (\mathbf{a}-\mathbf{c})$ 방향의 혼합곱(scalar triple product)을 법선 크기로 정규화해 계산한다.

$$d_p = \frac{(\mathbf{p}-\mathbf{a})^T \left((\mathbf{a}-\mathbf{b}) \times (\mathbf{a}-\mathbf{c})\right)}{\|(\mathbf{a}-\mathbf{b}) \times (\mathbf{a}-\mathbf{c})\|}$$

이렇게 모은 edge 잔차와 planar 잔차의 제곱합을 하나의 비용 함수로 묶어 Levenberg-Marquardt법으로 최적화하면 스캔 간 상대 변환 $\mathbf{T}$가 나온다.

$$\mathbf{T}^* = \underset{\mathbf{T}}{\arg\min} \sum_{\mathbf{p} \in \mathcal{E}} d_e(\mathbf{T}\cdot\mathbf{p})^2 + \sum_{\mathbf{p} \in \mathcal{P}} d_p(\mathbf{T}\cdot\mathbf{p})^2$$

**Mapping 모듈 (~1Hz)**

Scan-to-scan 오도메트리는 빠르지만 직전 스캔 한 장에만 맞추기 때문에 미세한 오차가 매 스캔 차곡차곡 쌓인다. 느린 Mapping 스레드는 오도메트리가 예측한 포즈를 초기값 삼아, 현재 스캔을 지금까지 누적된 로컬 맵 전체에 정합하는 scan-to-map 최적화를 1Hz 안팎의 주기로 돌려 누적 드리프트를 억제한다. 이때 맵이 무한정 커지는 것을 막으려고 현재 위치 주변의 정육면체(cube) 영역만 메모리에 남기고, 복셀 다운샘플링으로 점 밀도를 일정하게 유지한다.

**모션 왜곡 보정 (Motion Distortion Compensation)**

카메라가 글로벌 셔터로 한 순간을 통째로 찍는 것과 달리, 회전형(spinning) LiDAR는 레이저 헤드가 한 바퀴 도는 데 회전 주파수의 역수만큼 시간이 걸린다. 예컨대 10Hz로 회전한다면 한 스캔을 완성하는 데 약 100ms가 흐른다. 만약 그 100ms 동안 차량이 시속 36km(10m/s)로 달리고 있다면, 스캔 시작 시점의 점과 끝 시점의 점은 무려 1m나 떨어진 서로 다른 센서 좌표계에서 측정된 셈이다. 이 찌그러짐을 펴지 않고 강체 변환을 맞추면 지도와 포즈가 함께 틀어진다.

모션 왜곡의 크기는 스캔 소요 시간과 이동 속도에 비례하므로, 각 점의 타임스탬프와 그 사이의 센서 궤적을 이용해 모든 점을 한 기준 시각 좌표계로 되돌려야 한다. LOAM은 한 스캔 동안 각속도와 선속도가 일정하다고 가정하고, 스캔 시작 $t_s$부터 끝 $t_e$까지의 전체 상대 변환 $\mathbf{T}_{s \to e}$를 구한 뒤 각 점의 타임스탬프 $t_k$에 맞춰 중간 변환을 보간한다. 원 논문에서는 6자유도 벡터 $[t_x, t_y, t_z, \theta_x, \theta_y, \theta_z]$를 선형 보간해 썼고, 이를 Lie 군 $SE(3)$ 위에서 표현하면 다음과 같이 정리된다.

$$\mathbf{T}(t_k) = \text{Exp}\left(\frac{t_k - t_s}{t_e - t_s} \cdot \text{Log}(\mathbf{T}_{s \to e})\right)$$

원 논문의 식 (5)에서는 이 변환을 $\mathbf{p}_k = \mathbf{T}(t_k)\,\tilde{\mathbf{p}}_k$로 정의한다. 즉 $\mathbf{T}(t_k)$는 스캔 시작 시점 좌표계의 점 $\tilde{\mathbf{p}}_k$를 시각 $t_k$의 움직인 LiDAR 좌표계로 옮기는 변환이다. 따라서 실제로 시각 $t_k$에 측정된 점 $\mathbf{p}_k$를 스캔 시작(또는 끝) 기준 좌표계로 똑바로 되돌려 모션 왜곡을 펴려면 역변환 $\mathbf{T}(t_k)^{-1}$을 곱해야 한다.

$$\mathbf{p}_k^{\text{corrected}} = \mathbf{T}(t_k)^{-1} \cdot \mathbf{p}_k$$

IMU가 함께 탑재된 시스템이라면 등속 가정에 기댈 필요 없이 고주파 IMU 적분 궤적으로 각 타임스탬프의 훨씬 정확한 중간 포즈를 얻어 왜곡을 보정한다.

LOAM 파이프라인은 곡률 기반 특징 추출, scan-to-scan 오도메트리, scan-to-map 매핑 스레드로 이어진다. 아래 수도코드에는 이 파이프라인의 핵심 뼈대가 담겨 있다.

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

LOAM은 3D 매핑 정밀도 면에서는 뛰어났지만, 지상 무인 차량(UGV)에 싣고 Jetson TX2 같은 저전력 임베디드 보드에서 돌리기에는 연산량이 많았다. 또한 나뭇잎 같은 지저분한 특징점을 거르지 못해 장시간 주행할 때 드리프트가 누적되곤 했다. [LeGO-LOAM (Shan & Englot, 2018)](https://doi.org/10.1109/IROS.2018.8594299)은 지면 차량의 환경 특성을 파고들어 파이프라인을 경량화하고 안정성을 끌어올렸다.

1. **Ground Segmentation**: 3D 포인트 클라우드를 2D range image로 투영한 뒤, 인접 레이저 빔 사이의 수직 기울기 각도(공개 구현 기준 10° 이하)를 기준으로 지면(ground)과 비지면을 먼저 가른다. 넓게 깔린 지면점은 곧바로 planar feature로 삼고, 나머지 비지면점에서만 edge feature를 추려 낸다.
2. **Point Cloud Segmentation**: 비지면 점들을 range image 위에서 연결 성분 클러스터링으로 묶어, 점 개수가 기준에 못 미치는 작은 클러스터(나뭇잎이나 산발적 노이즈)는 과감히 버린다. 덕분에 지면 차량 주행에 방해가 되는 잡음 특징점이 대폭 줄어든다.
3. **2단계 LM 최적화**: LOAM이 6-DoF를 한 덩어리로 풀었던 것과 달리, LeGO-LOAM은 수직·기울기 자유도 $[t_z, \theta_{\text{roll}}, \theta_{\text{pitch}}]$를 주로 바닥 평면이 구속하고 수평 이동·방위각 $[t_x, t_y, \theta_{\text{yaw}}]$를 주로 수직 벽면과 모서리가 구속한다는 기하학적 직관을 활용한다. ground planar feature로 먼저 $[t_z, \theta_{\text{roll}}, \theta_{\text{pitch}}]$를 풀고, 이어서 edge feature로 $[t_x, t_y, \theta_{\text{yaw}}]$를 푸는 2단계 최적화로 정확도를 유지하면서도 원 논문 기준 계산 시간을 약 35% 줄였다.
4. **포즈 그래프 최적화**: 순수 오도메트리였던 LOAM에 loop closure 검출과 포즈 그래프 최적화를 결합해 장거리 주행에서 쌓이는 전역 드리프트를 되돌릴 수 있게 했다.

### 7.2.3 왜 LOAM 계열이 오래 살아남았는가

LOAM이 발표된 지 10년이 넘었지만, edge와 planar 특징점을 뽑아 맞추는 기하학적 골격은 LIO-SAM(2020)을 비롯한 현대 시스템에서도 여전히 현역으로 쓰인다. 이 설계가 오래 살아남은 데는 네 가지 이유가 맞물려 있다.

1. **기하학적 명확성**: edge와 planar 특징점은 인공 환경의 뼈대를 이루는 실제 모서리와 평면이라는 기하 원시체(geometric primitive)에 직관적으로 대응한다. 건물과 도로가 지배적인 도심·실내 환경에서는 이 구조적 가정이 놀라울 만큼 잘 들어맞는다.
2. **계산 효율**: 매 스캔 수만 개의 점 전체를 정합하는 대신 정보량이 응축된 수백·수천 개의 특징점만 다루므로, CPU 자원만으로도 실시간 처리가 가능하다.
3. **확장성**: 특징 추출과 정합을 담당하는 LiDAR 프런트엔드를 IMU(LIO-SAM)나 카메라(LVI-SAM)와 모듈처럼 엮어 다중 센서 시스템으로 확장하기 좋다(반면 KISS-ICP처럼 다른 센서를 더하지 않고 LiDAR 단독 ICP 자체를 단순화해 밀어붙인 갈래도 있다).
4. **강건성**: 곡률 기반 분류와 입사 기하 필터가 측정 노이즈와 불안정한 경계점을 사전에 걸러 내어, 곡률 패턴이 일관된 점들만 정합에 남긴다. 다만 곡률은 단일 스캔 라인의 이웃점 배치만으로 계산하는 순수 기하량이므로 움직이는 차량의 모서리와 평면도 똑같이 특징으로 잡히며, 동적 물체 제거는 별도 단계로 처리해야 한다.

물론 LOAM 방식에도 뚜렷한 사각지대가 있다. 비반복(non-repetitive) 주사 패턴을 쓰는 solid-state LiDAR에는 회전형 다채널 LiDAR처럼 가지런한 스캔 라인이 없어, 기존 곡률 추출 방식을 그대로 적용하기 어렵다(뒤의 FAST-LIO2는 아예 특징 추출을 없애 이 문제를 피해 간다). 나아가 기하학적 요철이 거의 없는 드넓은 개활지나 끝없이 뻗은 직선 터널처럼 환경 자체의 관측 기하가 퇴화한 곳에서는 특징점을 뽑든 raw 점을 쓰든 오도메트리 정확도가 무너진다.

## 7.3 Tightly-Coupled LiDAR-Inertial Odometry

지금까지의 순수 LiDAR 오도메트리는 차량이 급격히 회전하거나 가속할 때 두 가지 난관에 부딪힌다. 스캔이 진행되는 ~100ms 동안 움직임이 심해지면 모션 왜곡을 등속 모델만으로 펴기 어렵고, 스캔 간 변위가 정합 알고리즘의 수렴 반경을 벗어나면서 초기 포즈 예측도 흔들린다. 단안 VO가 IMU를 결합해 스케일 문제를 해결하고 빠른 회전을 견뎠듯(→ 6.3절), LiDAR 역시 고주파 IMU 측정과 원시 수준에서 결합하는 **tightly-coupled LIO**로 진화하면서 훨씬 거친 기동을 버텨 내게 되었다.

### 7.3.1 LIO-SAM

[LIO-SAM (Shan et al., 2020)](https://arxiv.org/abs/2007.00258)은 LOAM 계열의 edge·planar 특징 정합을 GTSAM 기반의 factor graph 최적화와 하나로 묶은 대표적인 LIO 시스템이다. LiDAR와 IMU를 합칠 뿐 아니라, GPS나 loop closure처럼 서로 다른 주기로 들어오는 제약까지 그래프 위에서 유연하게 통합할 수 있도록 설계되었다.

**Factor Graph 기반 통합**

LIO-SAM은 각 센서 관측을 factor graph의 노드를 잇는 factor로 변환한다.

1. **IMU Preintegration Factor**: Forster et al. (2017)의 on-manifold preintegration으로 연속 키프레임 사이에 쌓인 관성 측정을 하나의 제약으로 압축해, 두 키프레임의 상대 회전·속도·위치와 IMU 바이어스를 동시에 엮는다.
2. **LiDAR Odometry Factor**: LOAM 방식으로 왜곡을 편 뒤 추출한 특징점을 로컬 서브맵에 scan-to-map 정합하고, 그렇게 얻은 상대 포즈를 두 키프레임 사이의 상대 변환 factor로 넣는다.
3. **GPS Factor**: GPS 측정이 들어왔을 때 현재 추정한 위치 공분산이 GPS의 위치 공분산보다 크다면, 드리프트를 눌러 주기 위해 GPS 좌표를 단항(unary) 위치 factor로 그래프에 추가한다. 반대로 터널이나 실내처럼 GPS가 끊긴 구간에서는 LiDAR와 IMU factor만으로 궤적을 이어 간다.
4. **Loop Closure Factor**: 원 논문에서는 현재 포즈와 거리상 가까운 과거 키프레임을 찾는 단순한 방식으로 루프 후보를 고르고(Scan Context(→ §9.3.1) 같은 3D 장소 인식으로 대체할 수 있음을 함께 언급한다), ICP로 두 스캔 사이의 정밀 상대 변환을 계산해 이진(binary) factor로 묶는다.

이 네 종류의 factor는 GTSAM의 증분 최적화 솔버 iSAM2가 관리하는 전역 포즈 그래프에 들어가 실시간으로 풀린다. 여기에 LIO-SAM은 보조 IMU 오도메트리 그래프를 따로 두고 주기적으로 초기화하며 연동하는데, 이 보조 그래프가 LiDAR 키프레임 주기보다 훨씬 빠른 IMU 주기(수백 Hz)로 상태를 전파하고 바이어스를 추정한다. 이처럼 모든 센서 관측이 독립적인 factor 모듈로 표현되므로, 새로운 보조 센서를 얹거나 빼기가 한결 수월하다.

**IMU 기반 De-skewing**

LIO-SAM 안에서 IMU는 정합을 두 번 돕는다. 먼저 한 스캔이 찍히는 동안 들어온 IMU 적분 궤적으로 각 점의 타임스탬프에 해당하는 중간 포즈를 구해 스캔 왜곡을 똑바로 편다(de-skewing). 이어서 preintegration이 예측한 다음 키프레임 포즈를 scan-to-map 정합의 든든한 초기값으로 건넨다. 거꾸로 LiDAR가 정합으로 맞춰 낸 정확한 포즈는 IMU 적분에 쌓인 드리프트를 교정하고 가속도계·자이로 바이어스를 바로잡는 기준점이 된다.

**Keyframe 기반 효율화**

전역 맵 전체에 스캔을 매번 맞추려면 시간이 지날수록 검색 비용이 커지므로, LIO-SAM은 현재 위치 주변에 있거나 최근에 방문한 키프레임들의 점만 모아 만든 로컬 서브맵에 스캔을 정합해 연산량을 일정하게 유지한다.

아래 수도코드를 보면, 키프레임이 추가될 때마다 LIO-SAM이 IMU preintegration, scan-to-submap LiDAR 오도메트리, 선택적 GPS·루프 factor, 바이어스 랜덤워크 제약을 그래프에 넣고 iSAM2로 증분 갱신한다.

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

LIO-SAM이 여전히 LOAM식 스캔 라인 특징 추출과 factor graph 최적화에 기대고 있다면, [FAST-LIO2 (Xu et al., 2022)](https://doi.org/10.1109/TRO.2022.3141876)는 정반대의 철학을 밀어붙인다. 공간 자료구조만 충분히 빠르다면 애초에 특징점을 골라내느라 기하 정보를 버릴 이유가 없다고 판단했다. 이에 다운샘플링한 raw LiDAR 점을 곧바로 맵에 맞추고 반복 확장 칼만 필터(IEKF)로 풀어내는 **direct LiDAR-inertial odometry**를 완성했다.

**1. Direct Point Registration (특징 추출 제거)**

FAST-LIO2는 edge나 planar 특징을 가려내지 않고 다운샘플링한 raw 점 $\mathbf{p}_k$ 전부를 정합에 투입한다. 각 점을 현재 추정 포즈로 맵에 옮긴 뒤, 맵에서 찾은 최근접 점들로 국소 평면을 피팅하고 점과 평면 사이의 수직 거리만 최소화한다.

$$d_k = \mathbf{n}_k^T (\mathbf{T} \cdot \mathbf{p}_k - \mathbf{q}_k)$$

여기서 $\mathbf{n}_k$는 맵에서 추정한 최근접 평면의 법선 벡터, $\mathbf{q}_k$는 그 평면 위의 점이다.

굳이 특징 추출 단계를 걷어 낸 까닭은 세 가지다. 첫째, 임계값에 맞춰 특징을 분류하다 보면 정합에 쓸모 있는 점들까지 버려져 환경이 성길수록 불리해진다. 둘째, 스캔 라인 곡률에 의존하지 않기 때문에 회전형 LiDAR는 물론 Livox처럼 비반복(non-repetitive) 스캔 패턴을 그리는 solid-state LiDAR에도 별도 수정 없이 곧바로 적용된다. 셋째, 맵 조회와 갱신을 뒷받침하는 자료구조(`ikd-Tree`)가 충분히 빠르기 때문에 raw 점 전체를 쓰더라도 실시간성을 잃지 않는다.

**2. ikd-Tree (Incremental k-d Tree)**

기존의 정적 kd-tree에서는 새 스캔이 들어와 맵에 점이 추가되거나 오래된 구역이 지워질 때마다 트리 전체를 처음부터 다시 구축해야 했다. FAST-LIO2는 이 병목을 깨기 위해 동적 증분 트리인 `ikd-Tree`를 직접 설계해 넣었다.

- **점 삽입**: 기존 트리를 유지한 채 새 점을 $O(\log N)$ 시간에 증분 삽입한다.
- **점 삭제**: 즉각 노드를 뜯어내는 대신 삭제 표시만 남기는 lazy delete로 삭제 비용을 낮춘다.
- **동적 re-balancing**: 삽입과 삭제로 특정 서브트리가 한쪽으로 지나치게 기울어졌을 때만 scapegoat tree 기법으로 그 서브트리만 국소적으로 재구축한다.
- **Box 범위 삭제**: 현재 센서 위치에서 멀리 벗어난 구역의 점들을 바운딩 박스 단위로 일괄 제거해 맵 크기가 무한히 커지는 것을 막는다.

이 `ikd-Tree` 덕분에 FAST-LIO2는 스캔마다 맵을 실시간으로 갱신하면서도 k-NN 평면 검색을 빠르게 끝낸다.

**3. Iterated Extended Kalman Filter (IEKF)**

상태 추정 백엔드로 factor graph 대신 필터 계열인 **IEKF**를 선택한 점도 특징이다. 표준 EKF는 예측 상태에서 관측 모델을 단 한 번만 선형화하고 끝내므로, LiDAR의 point-to-plane 관측처럼 비선형성이 큰 문제에서는 잔여 오차가 크게 남는다. IEKF는 업데이트 수식을 여러 차례 반복하면서 선형화 지점을 직전 반복에서 얻은 새 추정치 $\hat{\mathbf{x}}^{(k)}$로 계속 옮겨 싣는다.

$$\hat{\mathbf{x}}^{(k+1)} = \hat{\mathbf{x}}^{-} + \mathbf{K}^{(k)} (\mathbf{z} - h(\hat{\mathbf{x}}^{(k)}) - \mathbf{H}^{(k)}(\hat{\mathbf{x}}^{-} - \hat{\mathbf{x}}^{(k)}))$$

여기서 $k$는 반복 인덱스, $\hat{\mathbf{x}}^{-}$는 IMU 전파로 얻은 예측 상태, $\mathbf{H}^{(k)}$는 현재 선형화 지점 $\hat{\mathbf{x}}^{(k)}$에서 다시 계산한 관측 자코비안이며, 칼만 이득 $\mathbf{K}^{(k)}$도 매 반복 함께 갱신된다.

$$\mathbf{K}^{(k)} = \mathbf{P}^{-} (\mathbf{H}^{(k)})^T (\mathbf{H}^{(k)} \mathbf{P}^{-} (\mathbf{H}^{(k)})^T + \mathbf{R})^{-1}$$

반복은 상태 증분이나 잔차 변화가 수렴 임계값 아래로 떨어지면 멈춘다. 공개된 FAST-LIO2 구현은 계산 속도를 위해 최대 반복 횟수를 작게 묶어 두지만(코드 기본값 4회, 배포 launch 파일 3회), 실제 수렴에 필요한 반복 수는 초기 포즈 오차와 주변 장면 기하에 따라 달라진다. 수학적으로 보면 이 IEKF 루프는 Gauss-Newton 최적화처럼 재선형화를 반복하면서도 필터 고유의 공분산 전파·갱신 구조를 그대로 유지한다(여기에 FAST-LIO는 점 수 $m$이 상태 차원 $n$보다 훨씬 크다는 데 착안해, 칼만 이득 항등식 변형으로 행렬 역연산 차원도 크게 낮춘다).

**상태 벡터**:

$$\mathbf{x} = [{}^G\mathbf{R}_I, {}^G\mathbf{p}_I, {}^G\mathbf{v}_I, \mathbf{b}_g, \mathbf{b}_a, {}^I\mathbf{R}_L, {}^I\mathbf{p}_L, \mathbf{g}]$$

표기는 FAST-LIO2 원 논문을 따른다. $G$는 첫 IMU 프레임으로 정한 전역 좌표계, $I$는 IMU, $L$은 LiDAR 좌표계이며 왼쪽 위첨자가 기준 좌표계, 오른쪽 아래첨자가 대상 좌표계다(${}^G\mathbf{R}_I$는 $G$에서 본 IMU 자세). 기본적인 회전·위치·속도·자이로 바이어스·가속도계 바이어스에 더해 LiDAR-IMU 외부 파라미터 ${}^I\mathbf{R}_L, {}^I\mathbf{p}_L$과 중력 벡터 $\mathbf{g}$까지 상태 벡터에 올려 두기 때문에, 주행 중 외부 캘리브레이션 오차와 중력 방향도 온라인으로 함께 보정된다.

FAST-LIO2 원 논문은 저자들의 하드웨어와 multi-line 회전형 및 solid-state LiDAR 데이터셋에서 최대 100Hz에 달하는 오도메트리·매핑 처리 속도를 보고했다. 다만 실제 현장 시스템에서의 처리율은 스캔 점 수, CPU 성능, 누적 맵 규모에 따라 크게 달라지므로 목표 플랫폼에서 다시 측정해야 한다.

아래 C++ 수도코드에서 FAST-LIO2 업데이트 루프는 예측 상태에서 출발해 `ikd-Tree` 평면 검색, point-to-plane 자코비안 계산, IEKF 반복 갱신과 공분산 업데이트, 맵 점 삽입으로 이어진다.

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

FAST-LIO2의 `ikd-Tree`도 매우 빠르지만, 트리 구조를 타고 내려가는 $O(\log N)$ 탐색과 재균형 비용은 여전히 남아 있다. [Faster-LIO (Bai et al., 2022)](https://doi.org/10.1109/LRA.2022.3152830)는 이 트리 자료구조를 해시 테이블 기반의 증분 복셀 구조인 **iVox**로 교체해 검색 속도를 한 단계 더 끌어올렸다. 공간 해시 맵으로 조회하는 희소 복셀에 점들을 담아 두고, 질의 점이 속한 복셀과 그 인접 복셀(6·18·26 이웃) 안에서만 근사 k-NN을 찾아 평면을 맞춘다. 해시 조회가 $O(1)$ 상수 시간에 끝나기 때문에 트리 검색보다 지연 시간이 줄어든다.

### 7.3.4 Point-LIO

LIO-SAM이나 FAST-LIO2를 포함한 대부분의 스캔 기반 LIO는 약 100ms 분량의 스캔 하나를 통째로 묶어 한 번에 업데이트한다. 스캔 도중 발생한 모션 왜곡은 IMU 적분으로 각 점의 타임스탬프에 맞춰 펴 주지만(→ §7.3.1), 필터 업데이트 자체는 스캔 단위로 일어나므로 그 100ms 구간 안에서의 왜곡 보정이 직전 스캔 끝 추정치와 단일 적분 궤적에 의존한다는 한계가 남는다. 드론이 격렬하게 뒤집히거나 강하게 충돌하는 고속·고각속도 기동에서는 이 가정이 금세 무너진다.

[Point-LIO (He et al., 2023)](https://doi.org/10.1002/aisy.202200459)는 이 스캔 단위 배칭(batching)을 아예 깨고, LiDAR 점이 하나 들어올 때마다 상태를 전파하고 즉시 갱신하는 극단적인 점 단위(point-wise) 업데이트로 밀어붙였다. 고주파 IMU 측정과 각 LiDAR 점의 타임스탬프를 시간 순서대로 엮어 각 점의 관측 시각마다 상태를 즉각 추정하므로, 스캔 단위 deskew보다 시간 해상도가 훨씬 높다(물론 이 순간 상태 추정치 역시 IMU 노이즈와 바이어스, 하드웨어 동기화 오차의 영향을 안고 있다).

IMU 측정 사이의 미소 시간 간격에서 Point-LIO는 다음 연속 시간 기구학 모델을 이산화해 상태를 전파한다.

$$\frac{d}{dt}\mathbf{R} = \mathbf{R}[\boldsymbol{\omega}]_\times, \quad \frac{d}{dt}\mathbf{v} = \mathbf{R}\mathbf{a} + \mathbf{g}, \quad \frac{d}{dt}\mathbf{p} = \mathbf{v}$$

점 하나가 도착할 때마다 state propagation과 single-point update가 곧바로 맞물려 돌아가므로, 사실상 연속 시간(continuous-time) 필터에 가깝게 동작한다. 점마다 그 순간의 포즈로 정합하기 때문에 별도의 스캔 왜곡 보정 단계가 필요 없고, 저자 실험에서는 각속도 75 rad/s(초당 약 4,300°)에 이르는 극한 기동까지 추적이 무너지지 않았다. 반대로 점 개수만큼 업데이트가 반복되므로 점 밀도가 높은 LiDAR에서는 연산 부담이 커지며, 논문에 보고된 FAST-LIO2 대비 처리 시간 역시 해당 데이터와 하드웨어 설정에 묶인 수치이므로 실제 적용 환경에서 직접 벤치마크해야 한다.

### 7.3.5 COIN-LIO

아무리 정교한 IEKF와 빠른 자료구조를 갖추더라도, 끝없이 곧게 뻗은 매끈한 콘크리트 터널이나 지형 기복이 없는 평탄한 개활지 앞에서는 LiDAR 정합 자체가 길을 잃는다. 모든 법선 벡터가 터널 벽 방향에만 쏠려 있어 진행 방향을 구속하는 평면이 하나도 없기 때문이다.

[COIN-LIO (Pfreundschuh et al., 2024)](https://arxiv.org/abs/2310.01235)는 기하 좌표만 보던 LIO에 LiDAR가 함께 반환하는 **반사 강도(intensity)** 정보를 끌어들여 이 기하학적 퇴화를 메운다. 반사 강도 데이터를 2D intensity image로 투영한 뒤 스캔 내부와 스캔 간 밝기가 일관되도록 필터링하고, 포인트 클라우드 정합만으로 구속력이 부족한 퇴화 방향을 진단해 그 방향을 채워 줄 수 있는 intensity image 패치를 골라낸다. 선택된 패치의 photometric residual을 IMU 측정 및 기존 point-to-plane 잔차와 함께 iterated EKF에 통합하면, 기하 형상은 밋밋하더라도 노면 표시나 벽면 재질 변화처럼 반사 강도 패턴이 살아 있는 구간에서는 추가적인 위치 제약을 얻을 수 있다(물론 이 보조 제약의 효과 역시 해당 퇴화 방향을 따라 유의미한 intensity 무늬가 관측되는지에 달려 있다).

---

## 7.4 Continuous-Time LiDAR Odometry

지금까지 다룬 오도메트리는 대부분 스캔마다 이산적인 포즈 하나를 세워 두고(discrete-time), 스캔 내부의 움직임은 등속 보간이나 IMU 사전 적분에 맡겼다. 하지만 IMU가 아예 없거나, 스캔 도중 속도가 매끄럽지 않게 변하는 상황이라면 스캔 내부의 궤적 형태 자체를 최적화 변수로 끌어올리는 편이 자연스럽다. **Continuous-time LiDAR 오도메트리**는 궤적을 시간 $t$의 연속 함수 $\mathbf{T}(t)$로 모델링해, 각 점의 측정 시각에 맞는 포즈를 정합 과정에서 함께 추정한다.

### 7.4.1 CT-ICP

[CT-ICP (Dellenbach et al., 2022)](https://arxiv.org/abs/2109.12979)는 이 연속 시간 아이디어를 스캔 단위로 가장 간결하게 구현한 알고리즘이다. 스캔 하나마다 시작 포즈 $\mathbf{T}_s$와 끝 포즈 $\mathbf{T}_e$ 두 개를 변수로 두고, 스캔 안의 점 타임스탬프 $t_k \in [t_s, t_e]$마다 정규화 시간 비율 $\alpha_k = \frac{t_k - t_s}{t_e - t_s}$로 회전은 구면 선형 보간(slerp), 위치는 선형 보간해 점의 위치를 계산한다.

$$\mathbf{R}(t_k) = \text{slerp}(\mathbf{R}_s, \mathbf{R}_e, \alpha_k), \quad \mathbf{t}(t_k) = (1-\alpha_k)\mathbf{t}_s + \alpha_k\mathbf{t}_e$$

스캔 밖에서 미리 등속 가정으로 점을 펴 놓고 정합하는 대신, 포인트 클라우드 정합 루프 안에서 두 포즈 $\mathbf{T}_s, \mathbf{T}_e$를 동시에 최적화하기 때문에 스캔 내부의 실제 궤적까지 함께 교정된다. 별도의 IMU 없이도 모션 왜곡과 정합을 한 번에 풀어내므로, LiDAR 단독으로 동작해야 하는 플랫폼에서 특히 유용하다.

### 7.4.2 B-Spline 기반 궤적 표현

선형 보간은 한 스캔 안에서의 움직임만 매끄럽게 이을 뿐 스캔 경계에서 속도가 꺾일 수 있다. 여러 스캔에 걸쳐 속도와 가속도까지 연속적인 궤적을 만들고 싶다면, 궤적 전체를 제어점(control point) $\{\mathbf{T}_i\}$가 정의하는 누적 **B-spline** 곡선으로 표현한다.

$$\mathbf{T}(t) = \mathbf{T}_0 \prod_{i=1}^{k} \text{Exp}\left(\tilde{B}_i(t) \cdot \text{Log}(\mathbf{T}_{i-1}^{-1}\mathbf{T}_i)\right)$$

여기서 $\tilde{B}_i(t)$는 누적(cumulative) B-spline 기저 함수(basis function)이며, 3차(cubic) B-spline을 고르면 궤적이 2차 미분까지 연속인 $C^2$ 곡선이 된다.

B-spline 궤적을 쓰면 임의의 시점 $t$의 포즈를 곧바로 계산할 수 있을 뿐 아니라, 이 궤적을 해석적으로 미분해 각속도와 가속도까지 뽑아낼 수 있다. 이 덕분에 100Hz IMU, 10Hz LiDAR, 30Hz 카메라처럼 타임스탬프가 제각각인 비동기 센서 측정값을 억지로 한 시각에 맞추지 않고 그대로 궤적 식에 꽂아 넣을 수 있다. 또한 국소 지지(local support) 성질이 있어 제어점 하나를 수정해도 그 제어점이 영향을 미치는 인접 몇 구간만 다시 계산하면 된다(다만 $C^2$ 수학적 연속성만으로는 궤적이 물리적으로 로봇 동역학을 만족하는 것까지 자동으로 보장되지는 않는다).

대신 제어점 간격(knot spacing)이라는 핵심 하이퍼파라미터를 신중히 잡아야 한다. 간격을 너무 촘촘히 두면 노이즈에 과적합되고, 너무 넓히면 민첩한 고속 움직임이 뭉개진다. 또한 제어점이 많아질수록 자코비안 계산량이 불어나므로 이산 시간 모델보다 연산 비용이 크다. 앞서 살펴본 Kalibr(→ §3.4.2)의 camera-IMU 캘리브레이션이 바로 이 연속 시간 B-spline 궤적의 대표적인 활용 사례다.

---

## 7.5 Solid-State LiDAR 특화

최근에는 모터로 거울이나 헤드를 360° 돌리는 기계식 회전형 LiDAR 대신, Livox 시리즈처럼 가격과 크기·무게를 크게 낮춘 **Solid-state LiDAR**의 채택이 빠르게 늘고 있다. 하드웨어 주사 방식이 바뀌면 소프트웨어 파이프라인의 전제도 함께 달라진다.

회전형 LiDAR와 대표적인 비반복 주사 방식 solid-state LiDAR의 물리적 특성 차이는 아래 표와 같다.

| 특성 | 회전형 (Velodyne, Ouster) | Solid-state (Livox) |
|------|--------------------------|---------------------|
| 스캔 패턴 | 반복적 (매 회전 같은 패턴) | 비반복적 (꽃잎/로즈 패턴) |
| FoV | 360° 수평 | 모델별 (Mid-40 38.4° 원형, Avia 70.4°×77.2°, Horizon 81.7°×25.1°) |
| 점 밀도 | 균등 | 시간에 따라 누적, 불균등 |
| 가격 | 높음 | 낮음 |
| 크기/무게 | 큼 | 작음 |

**비반복 스캔이 특징 추출에 미치는 영향**

LOAM 계열의 곡률 기반 특징 추출은 레이저 채널이 일정한 스캔 라인을 따라 점을 순서대로 찍는다는 점을 전제로 한다. 그러나 Livox 같은 solid-state LiDAR는 매 순간 꽃잎 모양의 비반복 궤적을 그리며 쏘기 때문에 가지런한 스캔 라인 자체가 존재하지 않고 점 분포도 불규칙하다. 이 때문에 기존 스캔 라인 곡률 공식을 그대로 쓰기 어렵고, 공간상에서 직접 $K$-최근접 이웃(KNN)을 모아 국소 곡률을 계산하거나, 아예 특징 추출을 건너뛰고 raw 점을 정합에 넣는 접근이 힘을 얻는다.

**FAST-LIO2가 solid-state에 강한 이유**

바로 이 지점에서 1세대 FAST-LIO와 구별되는 FAST-LIO2의 진가가 드러난다. edge·planar 특징을 뽑던 1세대와 달리, FAST-LIO2는 특징 추출 단계를 없애고 raw 점을 직접 정합하므로 스캔 패턴이 회전형이든 비반복형이든 가리지 않는다. 게다가 비반복 주사 LiDAR는 정지하거나 천천히 움직이면서 적분 시간을 늘릴수록 좁은 시야각(FoV) 안을 촘촘하게 메워 나가는데, `ikd-Tree`는 새 점을 버리지 않고 증분 삽입해 이렇게 갈수록 조밀해지는 환경 지도를 손실 없이 담아낸다. 한 스캔의 FoV가 좁아 순간 기하 제약이 부족해지는 약점 역시 tightly-coupled IMU가 메워 준다.

실제로 Livox 계열은 작은 폼팩터와 비반복 스캔의 특성 덕분에 소형 드론이나 핸드헬드 스캐너에서 널리 쓰이며, FAST-LIO2와 Point-LIO의 공식 저장소에도 Livox Avia용 설정 파일(`avia.yaml`)이 기본 포함되어 있다. 실제 센서를 고를 때는 단가뿐 아니라 FoV, 최대 측정 거리, 시간 동기화 인터페이스, 점 분포 특성을 목표 플랫폼의 운동 조건과 함께 따져 봐야 한다.

---

## 7.6 학습 기반 LiDAR Odometry

카메라 기반 VO에서 딥러닝이 미분 가능 BA(DROID-SLAM 등, → §6.5)로 존재감을 키운 것처럼, LiDAR 분야에서도 포인트 클라우드 쌍을 입력받아 상대 포즈를 추정하는 학습 기반 연구들이 이어져 왔다.

### 7.6.1 DeepLO 계열

초기 및 중기 학습 기반 LiDAR 오도메트리에서는 비정형 포인트 클라우드를 2D 구면 투영 영상(range image)으로 펴서 합성곱 신경망에 넣는 흐름이 주류를 이루었다.

- **LO-Net (Li et al., 2019)**: LiDAR 스캔을 2D range image로 투영한 뒤 CNN 특징으로 상대 포즈를 회귀하며, 법선 추정과 동적 객체 마스크 예측을 보조 과제로 함께 학습한다.
- **DeepLO (Cho et al., 2020)**: 구면 투영한 vertex map과 normal map을 CNN에 통과시키고, point-to-plane ICP 잔차와 닮은 기하 인지 손실(geometry-aware loss)을 설계해 정답 포즈 없이도 비지도 학습이 가능하게 했다.
- **PWCLO-Net (Wang et al., 2021)**: 영상 광학 흐름에서 검증된 피라미드(Pyramid), 워핑(Warping), 비용 볼륨(Cost volume) 구조를 3D 포인트 클라우드 계층에 이식해 LiDAR 오도메트리 정확도를 끌어올렸다.

### 7.6.2 현재의 한계

그럼에도 오늘날 실전 LiDAR 오도메트리 현장에서 순수 학습 기반 방법이 ICP나 FAST-LIO2 같은 기하 파이프라인을 대체하지 못한 데는 네 가지 현실적 장벽이 있다.

1. **LiDAR 데이터의 특성**: 3D 포인트 클라우드는 순서가 정해지지 않은 비정형 집합(unordered set)이어서, 규칙적 픽셀 격자를 전제로 설계된 CNN 구조를 그대로 적용하면 투영 손실이나 해상도 왜곡이 생긴다.
2. **강한 기하 기준선**: 스캔 간 충분한 중첩과 3D 구조가 갖춰진 환경에서는 고전 ICP·GICP·NDT가 학습 데이터 없이도 이미 매우 정밀한 기준선을 세운다. 물론 반복 구조, 동적 객체, 강수·먼지, 극단적 저중첩 구간에서는 고전 방법도 무너질 수 있으므로, 딥러닝은 파이프라인 전체를 대체하기보다 대응점 탐색이나 동적 점 분리, 불확실성 추정 같은 보조 모듈로 들어가는 편이 효과적이다.
3. **데이터 부족**: 대량의 주행 궤적 레이블을 갖춘 대규모 LiDAR 데이터셋은 카메라 영상 데이터셋에 비해 절대적으로 부족하다.
4. **일반화**: 특정 LiDAR의 채널 수(예: 64채널)와 주행 환경에 맞춰 학습된 모델은 센서 사양이나 장착 각도, 지형이 조금만 달라져도 성능이 크게 떨어진다.

이 때문에 현재 실무에서는 오도메트리 코어 자체는 기하 최적화나 IEKF에 맡기고, 학습 모델은 loop closure 장소 인식(PointNetVLAD), 넓은 초기 오차를 잡는 전역 정합(GeoTransformer, → §5.8.5), 움직이는 차량·보행자를 걸러 내는 시맨틱 분할 같은 주변 구성 요소에서 주로 활약한다(반면 loop closure용 Scan Context는 학습 없이 기하 분포만으로 설계한 수작업 디스크립터다).

---

## 7.7 2023–2024년 연구

최근의 LiDAR 오도메트리 연구는 복잡한 특징 추출이나 무거운 딥러닝 모델로 가기보다, 고전 정합 알고리즘의 핵심 병목을 정교하게 다듬어 단순성과 범용성을 극대화하는 방향으로 다시 모이고 있다.

**[KISS-ICP (Vizzo et al., 2023)](https://arxiv.org/abs/2209.15397)**는 화려한 특징점 분류나 다중 센서 결합 없이도 가장 단순한 point-to-point ICP만으로 어디까지 갈 수 있는지를 보여 주었다. 등속 모델 기반 모션 보상에 주행 역학에 따라 자동으로 조절되는 적응적 대응 임계값과 강건 커널을 결합했다. 원 논문은 그 결과 자동차·드론·핸드헬드 데이터셋 전반에서 단 하나의 파라미터 세트만으로 복잡한 파이프라인들과 대등한 궤적을 냈다고 보고했다.

**[MAD-ICP (Ferrari et al., 2024)](https://arxiv.org/abs/2405.05828)**는 정합의 성패가 결국 좋은 데이터 매칭에 달렸다는 점에 주목해, PCA 기반 kd-tree로 포인트 클라우드의 국소 평면 구조를 빠르게 추출한 뒤 point-to-plane 정합에 투입한다. 원 논문은 별도의 도메인 맞춤 튜닝 없이도 다양한 주행 조건에서 특화 시스템과 견줄 만한 성능을 보고했다.

**[iG-LIO (Chen et al., 2024)](https://github.com/zijiechenrobotics/ig_lio)**는 point-to-plane 정합보다 표면 형상을 더 온전히 반영하는 GICP를 tightly-coupled IEKF LIO 안에 매끄럽게 녹여 냈다. 복셀 기반 표면 공분산 추정기(VSCE)로 점마다 국소 공분산을 구하는 부담을 줄이고, 증분 복셀 맵(incremental voxel map)으로 최근접점 탐색 비용을 낮췄다. 원 논문은 그 덕분에 회전형·solid-state LiDAR 데이터셋 전반에서 동일 파라미터를 유지하면서도 Faster-LIO보다 높은 효율과 최신 LIO 수준의 정확도를 달성했다고 보고했다.

---

## 요약

지금까지 살펴본 주요 LiDAR 오도메트리와 LIO 시스템의 정합 방식, 추정기, 센서 구성과 핵심 특징은 다음 표와 같다.

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

큰 흐름을 짚어 보면, **LOAM(2014) → LeGO-LOAM(2018) → LIO-SAM(2020)**이 기하 특징 추출과 factor graph 최적화를 결합하는 방향으로 발전해 왔다면, **FAST-LIO(2021) → FAST-LIO2(2022) → Point-LIO(2023)**는 특징 추출을 버리고 raw 점을 고속 공간 자료구조와 반복 칼만 필터로 직접 밀어 넣는 방향으로 진화해 왔다. 물론 어느 계열이 더 빠르고 정확한지는 LiDAR 주사 패턴, 운동의 격렬함, 맵 규모, 하드웨어 자원과 벤치마크 기준에 따라 달라진다.

이로써 카메라+IMU 기반 VIO(→ 6장)와 LiDAR+IMU 기반 LIO라는 두 갈래의 핵심 오도메트리 엔진을 모두 갖추었다. 하지만 실제 자율주행 차량이나 로봇은 어느 한쪽 센서만 싣지 않는다. 터널이나 눈비, 역광처럼 각 센서가 무력화되는 순간을 서로 메우려면 카메라·LiDAR·IMU·GNSS를 하나의 아키텍처로 엮어야 한다. 이어지는 8장에서는 다중 센서 융합 구조를 본격적으로 다루며, 이 이질적인 센서들을 어느 깊이에서 결합할지(loosely, tightly, ultra-tightly), 그리고 비동기 지연과 센서 고장(degradation)에 어떻게 대응할지를 논의한다.
