# Ch.6 — Visual Odometry & Visual-Inertial Odometry

Ch.4에서 상태 추정 프레임워크를, Ch.5에서 특징점 매칭 기술을 다루었다. 이제 이 두 가지가 실제로 결합되는 첫 번째 시스템 — Visual Odometry(VO)와 Visual-Inertial Odometry(VIO)를 살펴본다.

Visual Odometry(VO)는 카메라 영상만으로 카메라의 자기 운동(ego-motion)을 추정하는 기술이고, Visual-Inertial Odometry(VIO)는 여기에 IMU를 결합하여 스케일 관측 가능성과 강건성을 확보한 기술이다. 이 챕터에서는 VO/VIO의 내부 구조와 설계 선택을 깊이 있게 다룬다.

VO의 기원은 [Nistér et al. (2004)](https://doi.org/10.1109/CVPR.2004.1315094)로 거슬러 올라간다. 이 논문은 "Visual Odometry"라는 용어를 최초로 정의하고, 스테레오/단안 카메라로 실시간 자기 운동 추정 시스템을 제시했다. 스테레오 접근에서는 좌우 카메라에서 3D 점을 삼각측량한 뒤 3-point 알고리즘으로 프레임 간 강체 변환을 추정했고, 단안 접근에서는 5-point 알고리즘으로 Essential Matrix를 추정했다. 이 기본 파이프라인 — 특징점 검출 → 매칭 → RANSAC → 모션 추정 — 은 20년이 지난 지금도 feature-based VO의 뼈대를 이루고 있다.

VO/VIO 시스템의 분류는 크게 세 축으로 나눌 수 있다:

1. **Feature-based vs Direct**: 기하학적 특징점(corner, edge)을 추출하여 매칭하는가, 아니면 픽셀 밝기 자체를 직접 사용하는가
2. **Filter vs Optimization**: 상태 추정에 칼만 필터 계열을 쓰는가, 비선형 최적화를 쓰는가
3. **Loosely coupled vs Tightly coupled**: IMU와 카메라를 독립적으로 처리한 뒤 결과를 합치는가, raw measurement를 하나의 최적화 문제에 넣는가

이 세 축의 조합이 다양한 시스템을 낳았다. 이 챕터는 대표적인 시스템들을 하나씩 해부하면서, 각 설계 선택의 이유와 결과를 분석한다.

---

## 6.1 Feature-based Visual Odometry

Feature-based VO는 영상에서 기하학적 특징점을 추출하고, 프레임 간 대응 관계를 찾아 카메라 모션을 추정하는 접근이다. 가장 오래되고 가장 잘 이해된 VO 패러다임이며, 현재까지도 ORB-SLAM3를 통해 가장 널리 사용된다.

### 6.1.1 Frontend: Detection, Tracking, Outlier Rejection

Feature-based VO의 프론트엔드는 세 가지 핵심 작업을 수행한다.

**특징점 검출 (Feature Detection)**

프레임에서 추적 가능한 점을 찾는 단계다. 이상적인 특징점은 반복 가능성(repeatability)이 높아야 한다 — 같은 3D 점이 다른 시점에서 촬영되어도 비슷한 위치에서 검출되어야 한다.

Harris corner detector는 이미지 패치의 자기상관 행렬(autocorrelation matrix, 또는 second moment matrix) $\mathbf{M}$을 기반으로 코너를 검출한다:

$$\mathbf{M} = \sum_{(x,y) \in W} w(x,y) \begin{bmatrix} I_x^2 & I_xI_y \\ I_xI_y & I_y^2 \end{bmatrix}$$

여기서 $I_x, I_y$는 이미지 그래디언트, $W$는 윈도우, $w$는 가중치 함수다. $\mathbf{M}$의 두 고유값이 모두 크면 코너로 판정한다. Harris 응답 함수는:

$$R = \det(\mathbf{M}) - k \cdot \text{tr}(\mathbf{M})^2 = \lambda_1\lambda_2 - k(\lambda_1 + \lambda_2)^2$$

FAST (Features from Accelerated Segment Test)는 속도에 최적화된 검출기다. 후보 픽셀 $p$ 주위 반지름 3의 원(Bresenham circle) 위 16개 점 중 $n$개(보통 $n=9$) 이상이 연속으로 $p$보다 밝거나 어두우면 코너로 판정한다. 사전 테스트로 1, 5, 9, 13번 점만 먼저 검사하여 비코너를 빠르게 제거한다. Harris 대비 수십 배 빠르지만, 노이즈에 약하고 방향/스케일 불변성이 없다.

ORB (Oriented FAST and Rotated BRIEF)는 FAST 검출에 방향 정보를 추가하고 BRIEF 디스크립터를 회전 보정하여 실시간 SLAM에 적합한 특징점을 제공한다. 방향은 이미지 패치의 intensity centroid로 계산한다:

$$\theta = \text{atan2}(m_{01}, m_{10}), \quad m_{pq} = \sum_{x,y} x^p y^q I(x,y)$$

ORB-SLAM 시리즈는 스케일 불변성을 위해 이미지 피라미드(보통 8레벨, 스케일 팩터 1.2)에서 ORB를 추출한다.

**특징점 추적 (Feature Tracking)**

프레임 간 대응을 찾는 방법은 두 가지다:

1. **디스크립터 매칭**: 각 프레임에서 독립적으로 특징점을 검출하고, 디스크립터 간 거리(Hamming distance for binary descriptors, L2 for float descriptors)로 매칭한다. ORB-SLAM3는 이 접근을 사용한다. DBoW2 어휘 트리를 이용해 매칭 후보를 제한하여 속도를 높인다.

2. **광학 흐름 추적 (Optical Flow Tracking)**: Lucas-Kanade (LK) 추적기로 이전 프레임의 특징점 위치가 현재 프레임에서 어디로 이동했는지 추적한다. VINS-Mono는 이 접근을 사용한다. 밝기 항상성 가정(brightness constancy assumption) 하에:

$$I(x + u, y + v, t + \Delta t) = I(x, y, t)$$

1차 Taylor 전개 후 윈도우 $W$ 내에서 최소제곱:

$$\begin{bmatrix} u \\ v \end{bmatrix} = \left(\sum_W \begin{bmatrix} I_x^2 & I_xI_y \\ I_xI_y & I_y^2 \end{bmatrix}\right)^{-1} \sum_W \begin{bmatrix} -I_xI_t \\ -I_yI_t \end{bmatrix}$$

이미지 피라미드 위에서 coarse-to-fine으로 수행하면 큰 변위도 추적 가능하다.

두 접근의 트레이드오프: 디스크립터 매칭은 넓은 baseline(시점 변화가 큰 경우)에서 강하고 루프 클로저에 재사용 가능하지만, 검출+기술+매칭에 시간이 든다. LK 추적은 빠르고 서브픽셀 정밀도를 제공하지만, 큰 시점 변화에 실패하기 쉽다.

**아웃라이어 제거 (Outlier Rejection)**

매칭 결과에는 반드시 오매칭(outlier)이 섞인다. 이를 제거하지 않으면 모션 추정이 크게 틀어진다.

기본적인 접근은 RANSAC (Random Sample Consensus)이다. 최소 샘플로 모델을 추정하고, 전체 데이터에 대한 인라이어 수를 세어 최적 모델을 선택한다. VO에서는:

- **2D-2D**: 5-point 알고리즘으로 Essential Matrix $\mathbf{E}$를 추정. $\mathbf{E} = [\mathbf{t}]_\times \mathbf{R}$로 분해하여 상대 포즈 $(R, t)$ 획득.
- **3D-2D**: PnP (Perspective-n-Point) 알고리즘 + RANSAC. 이미 삼각측량된 3D 점과 현재 2D 관측의 대응으로 절대 포즈를 추정.
- **3D-3D**: ICP 변종으로 강체 변환 추정.

Epipolar constraint를 이용한 아웃라이어 제거의 핵심 수식은:

$$\mathbf{p}_2^T \mathbf{E} \mathbf{p}_1 = 0$$

여기서 $\mathbf{p}_1, \mathbf{p}_2$는 정규화된 이미지 좌표다. 이 등식을 만족하지 않는 대응은 아웃라이어로 판정한다.

ORB-SLAM3는 Fundamental Matrix와 Homography를 동시에 추정하여, 장면 구조(평면 vs 비평면)에 따라 적절한 모델을 선택한다. 평면 장면에서는 Homography가 더 적은 자유도를 가지므로 안정적이다.

### 6.1.2 Backend: PnP, Motion-only BA, Local BA

프론트엔드가 대응 관계를 제공하면, 백엔드는 이를 기반으로 카메라 포즈와 3D 구조를 추정한다.

**PnP (Perspective-n-Point)**

이미 삼각측량된 3D 맵포인트 $\mathbf{P}_j \in \mathbb{R}^3$와 현재 프레임에서의 2D 관측 $\mathbf{u}_j \in \mathbb{R}^2$의 대응이 주어졌을 때, 카메라 포즈 $\mathbf{T} = (\mathbf{R}, \mathbf{t}) \in SE(3)$를 추정하는 문제다.

$$\mathbf{u}_j = \pi(\mathbf{R}\mathbf{P}_j + \mathbf{t})$$

여기서 $\pi: \mathbb{R}^3 \to \mathbb{R}^2$는 카메라 투영 함수다. 최소 해는 4점(P3P의 경우 3점 + 1점 검증)으로 얻으며, RANSAC과 결합하여 아웃라이어를 처리한다.

초기 해를 구한 뒤에는 모든 인라이어에 대해 비선형 최적화(refinement)를 수행한다.

**Motion-only Bundle Adjustment**

PnP로 초기 포즈를 얻은 뒤, 3D 점 위치는 고정한 채 카메라 포즈만 최적화하는 단계다:

$$\mathbf{T}^* = \underset{\mathbf{T}}{\arg\min} \sum_{j} \rho\left(\left\|\mathbf{u}_j - \pi(\mathbf{T} \cdot \mathbf{P}_j)\right\|^2_{\Sigma_j}\right)$$

여기서 $\rho(\cdot)$는 robust kernel (Huber 등), $\Sigma_j$는 관측 공분산이다. Huber 함수는:

$$\rho(s) = \begin{cases} s & \text{if } s \leq \delta^2 \\ 2\delta\sqrt{s} - \delta^2 & \text{if } s > \delta^2 \end{cases}$$

이 최적화는 6-DoF(SE(3) 위의 최적화)이므로, Lie algebra $\mathfrak{se}(3)$에서의 증분 $\boldsymbol{\xi} \in \mathbb{R}^6$을 통해 업데이트한다:

$$\mathbf{T} \leftarrow \exp(\boldsymbol{\xi}^{\wedge}) \cdot \mathbf{T}$$

**Local Bundle Adjustment**

최근 키프레임 $N$개와 이들이 관측하는 맵포인트를 함께 최적화한다:

$$\{\mathbf{T}_i^*, \mathbf{P}_j^*\} = \underset{\{\mathbf{T}_i, \mathbf{P}_j\}}{\arg\min} \sum_{i,j} \rho\left(\left\|\mathbf{u}_{ij} - \pi(\mathbf{T}_i \cdot \mathbf{P}_j)\right\|^2_{\Sigma_{ij}}\right)$$

이것이 Bundle Adjustment(BA)의 핵심 형태다. "Bundle"은 각 3D 점에서 카메라들로 향하는 광선(ray)의 다발을 의미한다. BA의 정규 방정식(normal equations)은 특수한 희소 구조(Schur complement structure)를 가진다:

$$\begin{bmatrix} \mathbf{H}_{cc} & \mathbf{H}_{cp} \\ \mathbf{H}_{pc} & \mathbf{H}_{pp} \end{bmatrix} \begin{bmatrix} \delta\boldsymbol{\xi} \\ \delta\mathbf{p} \end{bmatrix} = \begin{bmatrix} \mathbf{b}_c \\ \mathbf{b}_p \end{bmatrix}$$

$\mathbf{H}_{pp}$는 블록 대각(각 점은 독립)이므로, Schur complement로 점 변수를 소거하면:

$$(\mathbf{H}_{cc} - \mathbf{H}_{cp}\mathbf{H}_{pp}^{-1}\mathbf{H}_{pc})\delta\boldsymbol{\xi} = \mathbf{b}_c - \mathbf{H}_{cp}\mathbf{H}_{pp}^{-1}\mathbf{b}_p$$

이 축소된 시스템의 크기는 카메라 수에만 의존하므로(점 수와 무관), 효율적으로 풀 수 있다. 이것이 BA가 수만 개의 점을 다루면서도 실시간에 가까운 속도를 달성하는 핵심 비결이다.

### 6.1.3 ORB-SLAM3 아키텍처 상세 분석

[ORB-SLAM3 (Campos et al., 2021)](https://doi.org/10.1109/TRO.2021.3075644)는 feature-based visual(-inertial) SLAM의 현재 사실상 표준이다. 단일 프레임워크에서 monocular/stereo/RGB-D 카메라와 IMU를 모두 지원하며, pinhole과 fisheye 렌즈 모델을 수용한다.

**전체 아키텍처**

ORB-SLAM3는 세 개의 병렬 스레드로 구성된다:

1. **Tracking Thread**: 매 프레임에서 ORB 특징점을 추출하고, 기존 맵포인트와 매칭하여 현재 포즈를 추정한다. Motion-only BA로 포즈를 정제한다.
2. **Local Mapping Thread**: 새 키프레임이 삽입되면 새 맵포인트를 삼각측량하고, local BA를 수행한다. 중복 키프레임/맵포인트를 제거(culling)하여 맵을 컴팩트하게 유지한다.
3. **Loop Closing & Map Merging Thread**: DBoW2로 루프 후보를 검출하고, Sim(3) (또는 SE(3)) 정합으로 검증한다. 확인되면 pose graph optimization 후 full BA를 수행한다.

**Tracking Thread 상세**

1. 이전 프레임 기반 초기 포즈 예측: 등속 모델(constant velocity model) 또는 IMU preintegration으로 현재 포즈를 예측한다.
2. 예측 포즈를 이용해 맵포인트를 현재 프레임에 투영(projection), ORB 매칭으로 대응을 찾는다.
3. Motion-only BA로 포즈를 정제한다.
4. 키프레임 여부를 결정한다. 키프레임 조건: (a) 마지막 키프레임으로부터 일정 프레임 이상 경과, (b) 현재 프레임의 추적된 맵포인트 비율이 임계값 이하, (c) 현재 프레임에서 충분한 수의 맵포인트가 관측됨.

**Visual-Inertial 모드**

ORB-SLAM3의 visual-inertial 모드는 [Forster et al. (2017)](https://doi.org/10.1109/TRO.2016.2597321)의 IMU preintegration을 기반으로 한다. 핵심은 MAP(Maximum-a-Posteriori) 추정으로, 시각 잔차와 IMU 잔차를 하나의 비용 함수에서 동시 최적화한다:

$$\mathcal{C} = \sum_{i,j} \rho\left(\left\|\mathbf{e}_{ij}^{\text{vis}}\right\|^2_{\Sigma_{ij}}\right) + \sum_k \left\|\mathbf{e}_k^{\text{IMU}}\right\|^2_{\Sigma_k^{\text{IMU}}} + \left\|\mathbf{e}^{\text{prior}}\right\|^2_{\Sigma^{\text{prior}}}$$

IMU 잔차 $\mathbf{e}_k^{\text{IMU}}$는 키프레임 $k$와 $k+1$ 사이의 preintegrated IMU 측정과 상태 추정치의 차이로 정의된다. 상태 벡터는 각 키프레임마다:

$$\mathbf{x}_k = [{}^W\mathbf{R}_k, {}^W\mathbf{p}_k, {}^W\mathbf{v}_k, \mathbf{b}_k^g, \mathbf{b}_k^a]$$

을 포함한다. 이전 챕터(Ch.4)에서 유도한 preintegrated 측정:

$$\Delta\mathbf{R}_{k,k+1}, \quad \Delta\mathbf{v}_{k,k+1}, \quad \Delta\mathbf{p}_{k,k+1}$$

을 이용해 IMU 잔차를 다음과 같이 구성한다:

$$\mathbf{e}^{\text{IMU}}_{\Delta R} = \text{Log}\left(\Delta\hat{\mathbf{R}}_{k,k+1}^T \cdot \mathbf{R}_k^T \mathbf{R}_{k+1}\right)$$
$$\mathbf{e}^{\text{IMU}}_{\Delta v} = \mathbf{R}_k^T(\mathbf{v}_{k+1} - \mathbf{v}_k - \mathbf{g}\Delta t) - \Delta\hat{\mathbf{v}}_{k,k+1}$$
$$\mathbf{e}^{\text{IMU}}_{\Delta p} = \mathbf{R}_k^T(\mathbf{p}_{k+1} - \mathbf{p}_k - \mathbf{v}_k\Delta t - \frac{1}{2}\mathbf{g}\Delta t^2) - \Delta\hat{\mathbf{p}}_{k,k+1}$$

**Multi-Map System (Atlas)**

ORB-SLAM3의 가장 중요한 기여 중 하나는 Atlas 구조다. 시각 정보가 부족한 구간(빠른 회전, 가림 등)에서 트래킹이 실패하면, 기존 시스템은 재초기화 후 이전 맵과의 연결을 잃는다. ORB-SLAM3의 Atlas는:

1. 트래킹 실패 시 새로운 맵(sub-map)을 생성한다.
2. 각 sub-map은 독립적으로 유지된다.
3. 장소 인식(DBoW2)을 통해 이전에 방문한 sub-map을 감지하면, 두 맵을 자동으로 병합(merging)한다.
4. 병합 시 Sim(3)(단안의 경우) 또는 SE(3)(스테레오/VI의 경우) 정합을 수행한다.

이 구조 덕분에 ORB-SLAM3는 트래킹 실패로부터 우아하게 복구할 수 있으며, 과거 세션의 맵을 재사용하는 multi-session SLAM도 지원한다.

**ORB-SLAM3의 성능**: EuRoC MAV 데이터셋에서 stereo-inertial 구성이 3.6cm, TUM-VI에서 9mm 정확도를 달성한다. 이는 feature-based 접근의 강점 — 정밀한 기하학적 제약, 안정적인 루프 클로저 — 을 잘 보여준다.

```python
# ORB-SLAM3 Tracking Thread 수도코드
def track(frame, map, last_keyframe):
    # 1. ORB 특징점 추출 (이미지 피라미드 8레벨)
    keypoints, descriptors = extract_orb(frame, n_features=1000, n_levels=8, scale=1.2)
    
    # 2. 초기 포즈 예측
    if imu_available:
        T_predict = imu_preintegrate(last_frame.T, imu_measurements)
    else:
        T_predict = constant_velocity_model(last_frame.T, last_frame.velocity)
    
    # 3. 맵포인트를 현재 프레임에 투영 → 매칭
    projected_points = project_map_points(map.local_points, T_predict, frame.camera)
    matches = match_by_projection(keypoints, descriptors, projected_points, radius=15)
    
    # 4. Motion-only BA (포즈만 최적화, 맵포인트 고정)
    T_refined = motion_only_BA(
        pose_init=T_predict,
        observations=[(kp, mp) for kp, mp in matches],
        robust_kernel='huber',
        n_iterations=10
    )
    
    # 5. 키프레임 결정
    tracked_ratio = len(matches) / len(last_keyframe.observations)
    if tracked_ratio < 0.9 and len(matches) > 50:
        insert_keyframe(frame, T_refined, matches)
    
    return T_refined
```

---

## 6.2 Direct Visual Odometry

Direct 방법은 특징점 추출 없이 픽셀 밝기(intensity) 자체를 관측으로 사용한다. 기본 아이디어는: 3D 점 $\mathbf{P}$가 두 프레임에서 같은 밝기를 가져야 한다는 가정(brightness constancy)이다.

### 6.2.1 Photometric Error

Direct VO의 핵심 잔차는 photometric error다. 카메라 포즈 $\mathbf{T}_i, \mathbf{T}_j$와 3D 점 $\mathbf{P}$(또는 호스트 프레임의 픽셀 $\mathbf{u}$와 역깊이 $d^{-1}$로 파라미터화)에 대해:

$$e_{\text{photo}} = I_j\left(\pi(\mathbf{T}_j \mathbf{T}_i^{-1} \pi^{-1}(\mathbf{u}_i, d_i^{-1}))\right) - I_i(\mathbf{u}_i)$$

여기서:
- $\pi^{-1}(\mathbf{u}, d^{-1})$: 2D 점과 역깊이로부터 3D 점 복원 (unprojection)
- $\pi(\cdot)$: 3D→2D 투영
- $I_i, I_j$: 프레임 $i, j$의 밝기 이미지

이 잔차를 최소화하여 포즈와 깊이를 추정한다. Feature-based 방법의 reprojection error와의 핵심 차이는:

| | Reprojection error | Photometric error |
|---|---|---|
| 관측 | 특징점 좌표 (2D) | 픽셀 밝기 (1D 또는 패치) |
| 데이터 연관 | 명시적 (매칭 필요) | 암묵적 (와핑으로 계산) |
| 그래디언트 | 기하학적 | 이미지 그래디언트 |
| 텍스처 요구 | 코너/에지 필요 | 그래디언트만 있으면 됨 |
| 조명 변화 | 불변 (기하학적) | 민감 (보정 필요) |

Photometric error의 장점은 명시적 특징점 매칭이 불필요하다는 것이다. 이미지 전체에서 그래디언트가 있는 모든 영역을 활용할 수 있으므로, 텍스처가 부족하지만 약한 그래디언트가 존재하는 환경(하얀 벽의 미세한 질감 등)에서도 동작할 수 있다.

한계는 두 가지다:
1. **밝기 항상성 위반**: 조명 변화, 자동 노출, 렌즈 비네팅 등으로 같은 3D 점의 밝기가 프레임마다 달라진다. 이를 보정하지 않으면 정확도가 급격히 떨어진다.
2. **좁은 수렴 영역 (Basin of convergence)**: 이미지 그래디언트 기반 최적화이므로, 초기 포즈 추정이 나쁘면 로컬 미니멈에 빠진다. 보통 1~2 픽셀 이내의 초기 정합이 필요하다.

### 6.2.2 DSO (Direct Sparse Odometry) 아키텍처 상세 분석

[DSO (Engel et al., 2018)](https://doi.org/10.1109/TPAMI.2017.2658577)는 direct 방법과 sparse 표현을 결합한 VO 시스템이다. 기존에는 "direct = dense" ([LSD-SLAM, Engel et al., 2014](https://doi.org/10.1007/978-3-319-10605-2_54)), "sparse = indirect" (ORB-SLAM)이라는 암묵적 등식이 있었는데, DSO는 이 두 축을 새롭게 조합했다.

**DSO의 핵심 설계 원칙**

1. **Direct**: 특징점 없이 픽셀 밝기를 직접 사용
2. **Sparse**: 이미지 전체가 아니라, 그래디언트가 있는 영역에서 균등하게 점을 샘플링
3. **Joint Optimization**: 포즈, 역깊이, 카메라 내부 파라미터(affine brightness parameters)를 동시 최적화

**완전한 Photometric Calibration**

DSO의 가장 중요한 기여 중 하나는 photometric calibration의 체계적 처리다. 실제 카메라에서 관측되는 밝기 $I'$은 장면의 실제 복사 휘도(irradiance) $B$와 다음 관계를 갖는다:

$$I'(\mathbf{u}) = G(t \cdot V(\mathbf{u}) \cdot B(\mathbf{u}))$$

여기서:
- $G(\cdot)$: 카메라의 비선형 응답 함수 (response function)
- $t$: 노출 시간 (exposure time)
- $V(\mathbf{u})$: 렌즈 비네팅 (vignetting) — 이미지 중심에서 멀어질수록 밝기 감소

이 세 요소를 모두 보정하여 photometrically corrected 이미지를 얻는다:

$$I(\mathbf{u}) = t^{-1} \cdot G^{-1}(I'(\mathbf{u})) / V(\mathbf{u})$$

이 보정 없이는 같은 3D 점이라도 이미지 중심과 가장자리, 또는 노출이 다른 프레임에서 다른 밝기를 가지므로 photometric error가 부정확해진다.

**Affine Brightness Transfer Function**

완벽한 photometric calibration이 불가능한 경우를 대비해, DSO는 프레임 간 밝기 변화를 affine 모델로 추가 보상한다:

$$e_{\mathbf{p}j} = \sum_{\mathbf{p} \in \mathcal{N}_\mathbf{p}} w_\mathbf{p} \left\| (I_j[\mathbf{p}'] - b_j) - \frac{t_j e^{a_j}}{t_i e^{a_i}}(I_i[\mathbf{p}] - b_i) \right\|_\gamma$$

여기서 $a_i, b_i, a_j, b_j$는 프레임별 affine brightness 파라미터로 함께 최적화된다. $\|\cdot\|_\gamma$는 Huber norm이다.

**점 선택 전략**

DSO는 이미지를 격자로 나누고, 각 셀에서 그래디언트 크기가 가장 큰 점을 선택한다. 핵심은 "균등 분포"다 — 한 영역에 특징이 집중되는 것을 방지한다. 약 2000개의 점을 선택하며, 그래디언트 임계값을 적응적으로 조절하여 텍스처가 적은 영역에서도 점을 확보한다.

**Sliding Window Optimization**

DSO는 최근 5~7개의 키프레임과 이들에 속한 점들의 역깊이를 슬라이딩 윈도우에서 joint 최적화한다. 최적화 변수는:

$$\boldsymbol{\theta} = \{\mathbf{T}_1, \ldots, \mathbf{T}_n, d_1^{-1}, \ldots, d_m^{-1}, a_1, b_1, \ldots, a_n, b_n\}$$

즉 카메라 포즈(SE(3)), 역깊이(inverse depth), affine brightness 파라미터를 모두 포함한다.

윈도우에서 빠지는 프레임/점은 Schur complement로 마지널라이즈되어 prior로 남는다. 이 마지널라이제이션은 Ch.4.7에서 다룬 Schur complement 기반 마지널라이제이션과 동일한 원리이지만, 시각 잔차만 다룬다는 점이 다르다.

**DSO의 한계와 확장**

DSO의 본래 설계에는 루프 클로저가 없다. 이는 direct 방법의 근본적 한계가 아니라 설계 선택이다. LDSO (Loop-closing DSO)는 DBoW + direct 정합을 결합하여 이 한계를 해결했다. 또한 VI-DSO, BASALT 등은 DSO에 IMU를 결합한 VIO 변종이다.

```python
# DSO 핵심 흐름 수도코드
def dso_track(frame, window, camera):
    # 1. Photometric calibration 적용
    frame.I = apply_photometric_correction(frame.raw, camera.G_inv, camera.V, frame.exposure)
    
    # 2. 직접 이미지 정합으로 초기 포즈 추정 (coarse-to-fine)
    T_init = direct_alignment_pyramid(
        ref_frame=window.latest_keyframe,
        cur_frame=frame,
        initial_guess=constant_velocity_predict(),
        levels=[4, 3, 2, 1]  # 이미지 피라미드 레벨
    )
    
    # 3. 점 선택 (그래디언트 기반, 균등 분포)
    candidate_points = select_points_gradient_based(frame, n_blocks=32*32, n_per_block=1)
    
    # 4. 역깊이 초기화 (에피폴라 서치)
    for p in candidate_points:
        p.inv_depth = epipolar_search(window.keyframes, frame, p.u)
    
    # 5. 키프레임이면: 슬라이딩 윈도우에 삽입 후 joint optimization
    if is_keyframe(frame, window):
        window.add(frame, candidate_points)
        # 포즈 + 역깊이 + affine 파라미터 동시 최적화
        gauss_newton_optimize(
            residuals=photometric_residuals(window),
            variables=[T, inv_depth, a, b],
            max_iter=6
        )
        # 오래된 프레임 마지널라이즈
        if len(window) > 7:
            marginalize_oldest(window)
    
    return T_init
```

### 6.2.3 Semi-Direct: SVO

[SVO (Semi-direct Visual Odometry, Forster et al., 2017)](https://doi.org/10.1109/TRO.2016.2623335)는 feature-based와 direct의 장점을 결합한 하이브리드 접근이다. "Semi-direct"라는 이름은 추적(tracking)에는 direct 방법을, 매핑(mapping)에는 feature-based 방법을 사용하기 때문이다.

**SVO의 핵심 아이디어**

1. **Sparse Model-based Image Alignment**: 기존 3D 맵포인트를 현재 프레임에 투영하고, 각 투영점 주위의 패치에 대해 photometric error를 최소화하여 프레임 포즈를 추정한다. 이는 DSO처럼 이미지 그래디언트를 직접 사용하지만, 전체 이미지가 아닌 이미 알고 있는 맵포인트 주위만 사용하므로 매우 빠르다.

2. **Feature Alignment**: 포즈가 추정된 후, 각 맵포인트의 투영 위치를 서브픽셀 정밀도로 정제한다. 이때도 패치 기반 direct 정합을 사용한다.

3. **Structure & Motion Refinement**: 정제된 2D 위치를 "가상 특징점"으로 취급하여, BA(reprojection error 최소화)로 포즈와 3D 구조를 함께 최적화한다.

SVO는 이 3단계 분리 덕분에 매우 빠르다 — 고해상도 이미지에서도 200~400Hz를 달성하며, 이는 고속 드론 같은 agile 로봇에 적합하다. 반면 루프 클로저가 없고, 순수 rotation(제자리 회전)에 취약하다는 한계가 있다.

---

## 6.3 Tightly-Coupled Visual-Inertial Odometry

VO만으로는 두 가지 구조적 한계가 있다: (1) 단안 카메라의 스케일 모호성, (2) 빠른 모션이나 텍스처 부족 시 트래킹 실패. IMU를 결합하면 두 문제를 동시에 해결할 수 있다. Tightly-coupled VIO는 카메라와 IMU의 raw 측정을 하나의 추정 프레임워크에서 함께 처리한다.

### 6.3.1 VINS-Mono 아키텍처 상세

[VINS-Mono (Qin et al., 2018)](https://doi.org/10.1109/TRO.2018.2853729)는 단안 카메라 + 저가 IMU만으로 강건한 6-DoF 상태 추정을 달성하는 완전한 VIO 시스템이다. 초기화, 오도메트리, 루프 클로저, 맵 재사용까지 전체 파이프라인을 하나의 시스템으로 통합했다.

**시스템 아키텍처**

VINS-Mono는 크게 세 모듈로 구성된다:

1. **프론트엔드 (Measurement Preprocessing)**: KLT 기반 특징점 추적 + IMU preintegration
2. **백엔드 (Estimator)**: 비선형 최적화 기반 tightly-coupled VIO
3. **루프 클로저 (Relocalization)**: DBoW2 기반 장소 인식 + 재위치추정

**강건한 초기화 (Robust Initialization)**

단안 VIO의 최대 난제는 초기화다. 단안 카메라만으로는 메트릭 스케일을 관측할 수 없고, IMU 바이어스도 알 수 없다. VINS-Mono의 초기화는 loosely-coupled 접근으로 이 문제를 해결한다:

**Step 1: 순수 비전 SfM**

처음 몇 프레임에서 순수 비전만으로 Structure from Motion을 수행한다. 5-point 알고리즘으로 Essential Matrix를 추정하고, 삼각측량으로 3D 점과 카메라 포즈를 복원한다. 이때 스케일은 임의적이다.

**Step 2: Visual-Inertial Alignment**

SfM 결과와 IMU preintegration을 정렬(alignment)한다. 이 과정에서 다음을 추정한다:

(a) 자이로 바이어스 $\mathbf{b}_g$: 연속 키프레임 쌍의 SfM 회전 $\mathbf{R}_{c_k c_{k+1}}^{\text{sfm}}$과 IMU preintegrated 회전 $\Delta\hat{\mathbf{R}}_{k,k+1}$을 비교:

$$\min_{\mathbf{b}_g} \sum_k \left\| \text{Log}\left(\Delta\hat{\mathbf{R}}_{k,k+1}(\mathbf{b}_g)^T \cdot \mathbf{R}_{c_k}^{c_{k+1}} \right) \right\|^2$$

바이어스 변화에 대한 1차 근사(Ch.4.6의 preintegration 자코비안)를 이용하면 이 문제는 선형으로 풀린다.

(b) 중력 방향, 속도, 메트릭 스케일을 동시 추정. 이는 다음 선형 시스템으로 정리된다. 각 키프레임 쌍 $(k, k+1)$에 대해:

$$s\mathbf{R}_{c_0}^w \mathbf{p}_{c_{k+1}}^{c_0} - s\mathbf{R}_{c_0}^w \mathbf{p}_{c_k}^{c_0} - \mathbf{v}_k^w \Delta t_k + \frac{1}{2}\mathbf{g}^w\Delta t_k^2 = \mathbf{R}_k^w \Delta\hat{\mathbf{p}}_{k,k+1}$$

여기서 미지수는 $s$ (스케일), $\mathbf{g}^w$ (중력 벡터), $\{\mathbf{v}_k^w\}$ (속도)다. 중력 크기 $\|\mathbf{g}\| = 9.81$이라는 구속 조건을 추가하여 정확도를 높인다.

이 loosely-coupled 초기화가 수렴하면 tightly-coupled 최적화로 전환한다.

**Tightly-Coupled Nonlinear Optimization**

VINS-Mono의 백엔드는 슬라이딩 윈도우 내에서 세 종류의 잔차를 동시 최적화한다:

$$\min_{\mathcal{X}} \left\{ \left\|\mathbf{r}_p - \mathbf{H}_p \mathcal{X}\right\|^2 + \sum_{k \in \mathcal{B}} \left\|\mathbf{r}_{\mathcal{B}}(\hat{\mathbf{z}}_{b_k b_{k+1}}, \mathcal{X})\right\|^2_{\mathbf{P}_{b_k b_{k+1}}^{-1}} + \sum_{(l,j) \in \mathcal{C}} \left\|\mathbf{r}_{\mathcal{C}}(\hat{\mathbf{z}}_{l}^{c_j}, \mathcal{X})\right\|^2_{(\mathbf{P}_{l}^{c_j})^{-1}} \right\}$$

여기서:
- $\mathbf{r}_p$: 마지널라이제이션 사전(prior) 잔차
- $\mathbf{r}_{\mathcal{B}}$: IMU preintegration 잔차
- $\mathbf{r}_{\mathcal{C}}$: 시각 reprojection error 잔차
- $\mathcal{X}$: 상태 변수 (슬라이딩 윈도우 내 키프레임 포즈, 속도, IMU 바이어스, 특징점 역깊이)

**시각 잔차의 상세 정의**:

특징점 $l$이 키프레임 $i$에서 처음 관측되고 역깊이 $\lambda_l$로 파라미터화되었을 때, 키프레임 $j$에서의 reprojection error는:

$$\mathbf{r}_{\mathcal{C}} = \begin{bmatrix} \bar{u}_l^{c_j} - u_l^{c_j} / z_l^{c_j} \\ \bar{v}_l^{c_j} - v_l^{c_j} / z_l^{c_j} \end{bmatrix}$$

여기서 $\begin{bmatrix} u_l^{c_j} & v_l^{c_j} & z_l^{c_j} \end{bmatrix}^T = \mathbf{R}_{b_j}^{c} (\mathbf{R}_{w}^{b_j}(\mathbf{R}_{b_i}^{w}(\mathbf{R}_{c}^{b_i} \frac{1}{\lambda_l}\begin{bmatrix} \bar{u}_l^{c_i} \\ \bar{v}_l^{c_i} \\ 1 \end{bmatrix} + \mathbf{p}_c^b) + \mathbf{p}_{b_i}^w - \mathbf{p}_{b_j}^w) - \mathbf{p}_c^b)$

**슬라이딩 윈도우 관리와 마지널라이제이션**

VINS-Mono는 윈도우 크기를 고정(보통 10개 키프레임)하되, 키프레임 여부에 따라 두 가지 마지널라이제이션 전략을 적용한다:

1. **최신 프레임이 키프레임인 경우**: 가장 오래된 키프레임을 마지널라이즈한다. Schur complement로 해당 프레임에 연결된 모든 측정을 prior로 변환한다.

2. **최신 프레임이 키프레임이 아닌 경우**: 직전 프레임(second-newest)을 마지널라이즈한다. 이때 시각 측정만 버리고, IMU 측정은 인접 키프레임 사이의 preintegration에 포함되므로 정보가 보존된다.

이 두 전략의 핵심은 **정보 보존**이다. 마지널라이제이션은 변수를 제거하되 그 변수가 제공하던 정보를 prior 형태로 남긴다. Schur complement의 수학적 메커니즘은 Ch.4.7에서 상세히 다루었다.

**4-DoF 포즈 그래프 최적화**

루프 클로저가 검출되면, VINS-Mono는 6-DoF가 아닌 4-DoF (yaw + 3D translation)로 포즈 그래프를 최적화한다. 왜 4-DoF인가? IMU의 가속도계가 중력 방향을 관측하므로, roll과 pitch는 IMU에 의해 이미 충분히 관측 가능(observable)하다. 드리프트는 yaw와 위치에서만 누적된다. 따라서 루프 클로저 시 roll/pitch는 건드리지 않고 yaw와 위치만 보정하는 것이 물리적으로 타당하다.

```python
# VINS-Mono 슬라이딩 윈도우 최적화 수도코드
class VINSEstimator:
    def __init__(self, window_size=10):
        self.window_size = window_size
        self.states = []  # [pose, velocity, bias_g, bias_a] per keyframe
        self.features = {}  # feature_id -> inverse_depth
        self.prior = None  # marginalization prior
    
    def optimize(self):
        # 비용 함수 구성
        cost = 0.0
        
        # 1. 마지널라이제이션 prior
        if self.prior is not None:
            cost += self.prior.evaluate(self.states)
        
        # 2. IMU preintegration 잔차
        for k in range(len(self.states) - 1):
            preint = self.imu_preintegrations[k]
            r_imu = compute_imu_residual(
                self.states[k], self.states[k+1], preint
            )
            cost += r_imu.T @ preint.info_matrix @ r_imu
        
        # 3. 시각 reprojection error
        for feat_id, inv_depth in self.features.items():
            for obs in self.observations[feat_id]:
                r_vis = compute_reprojection_error(
                    self.states[obs.host_frame], self.states[obs.target_frame],
                    inv_depth, obs.uv, self.T_cam_imu
                )
                cost += huber(r_vis.T @ obs.info_matrix @ r_vis)
        
        # Gauss-Newton / Levenberg-Marquardt 최적화
        solve_nonlinear_least_squares(cost, max_iterations=8)
    
    def marginalize(self, is_keyframe):
        if is_keyframe:
            # 가장 오래된 프레임을 마지널라이즈
            self.prior = schur_complement_marginalize(
                self.states[0], connected_factors(self.states[0])
            )
            self.states.pop(0)
        else:
            # 직전 프레임을 마지널라이즈 (시각만, IMU는 보존)
            self.prior = schur_complement_marginalize(
                self.states[-2], visual_factors(self.states[-2])
            )
            self.states.pop(-2)
```

### 6.3.2 OKVIS

[OKVIS (Open Keyframe-based Visual-Inertial SLAM, Leutenegger et al., 2015)](https://doi.org/10.1177/0278364914554813)는 VINS-Mono보다 먼저 나온 tightly-coupled VIO로, 키프레임 기반 슬라이딩 윈도우 최적화의 초기 형태를 제시했다.

**OKVIS의 핵심 특징**:

1. **Harris corner + BRISK descriptor**: ORB 대신 Harris 코너와 BRISK 디스크립터를 사용한다.
2. **Keyframe-based marginalization**: VINS-Mono와 유사하게 슬라이딩 윈도우에서 마지널라이제이션을 수행한다.
3. **Speed error term**: IMU preintegration 대신, 짧은 시간 간격의 IMU 적분을 직접 수행하고 속도 제약으로 사용한다. 이후 OKVIS2에서 preintegration으로 전환했다.
4. **Ceres Solver 기반**: 최적화에 Ceres Solver를 사용한다.

OKVIS는 VINS-Mono 대비 초기화가 간단하지만(스테레오 카메라를 기본으로 가정), 단안 모드에서의 강건한 초기화는 VINS-Mono가 더 우수하다.

### 6.3.3 MSCKF: Multi-State Constraint Kalman Filter

[MSCKF (Mourikis & Roumeliotis, 2007)](https://doi.org/10.1109/ROBOT.2007.364024)는 필터 기반 VIO의 대표 알고리즘이다. 최적화 기반(VINS-Mono, ORB-SLAM3)과 구조적으로 다른 접근을 취하며, 현재까지도 특정 응용에서 활발히 사용된다.

**MSCKF의 핵심 아이디어: 랜드마크를 상태에 넣지 않는다**

EKF-SLAM은 랜드마크(3D 점)를 상태 벡터에 포함시킨다. $N$개의 랜드마크가 있으면 상태 벡터 크기가 $3N + 15$가 되고, 공분산 행렬의 크기가 $(3N+15)^2$이 되어 랜드마크 수에 대해 $O(N^2)$ 공간, $O(N^3)$ 시간이 필요하다. 이는 실시간 처리에 치명적이다.

**랜드마크를 상태 벡터에서 제외하면서도, 랜드마크가 제공하는 기하학적 구속 정보를 보존할 수 있다**는 것이 MSCKF의 출발점이다.

**상태 벡터 구조**

MSCKF의 상태 벡터는 IMU 오차 상태(error-state)와 슬라이딩 윈도우 내 $N$개 카메라 포즈로 구성된다:

$$\tilde{\mathbf{x}} = [\tilde{\mathbf{x}}_{IMU}^T, \tilde{\mathbf{x}}_{C_1}^T, \ldots, \tilde{\mathbf{x}}_{C_N}^T]^T$$

여기서 IMU 오차 상태는:

$$\tilde{\mathbf{x}}_{IMU} = [\delta\boldsymbol{\theta}^T, \tilde{\mathbf{b}}_g^T, {}^G\tilde{\mathbf{v}}_I^T, \tilde{\mathbf{b}}_a^T, {}^G\tilde{\mathbf{p}}_I^T]^T \in \mathbb{R}^{15}$$

각 카메라 포즈 오차 상태는:

$$\tilde{\mathbf{x}}_{C_k} = [\delta\boldsymbol{\theta}_{C_k}^T, {}^G\tilde{\mathbf{p}}_{C_k}^T]^T \in \mathbb{R}^6$$

상태 벡터의 전체 크기는 $15 + 6N$이다. 랜드마크 수와 무관하다.

**Null-Space 투영: 핵심 수학**

하나의 정적 특징점 $\mathbf{p}_f$가 $M$개의 카메라 포즈에서 관측되었다고 하자. 관측 방정식을 선형화하면:

$$\mathbf{r} = \mathbf{H}_X \tilde{\mathbf{x}} + \mathbf{H}_f \tilde{\mathbf{p}}_f + \mathbf{n}$$

여기서:
- $\mathbf{r} \in \mathbb{R}^{2M}$: 잔차 벡터 (관측 - 예측)
- $\mathbf{H}_X \in \mathbb{R}^{2M \times (15+6N)}$: 상태에 대한 자코비안
- $\mathbf{H}_f \in \mathbb{R}^{2M \times 3}$: 특징점 위치에 대한 자코비안
- $\tilde{\mathbf{p}}_f \in \mathbb{R}^3$: 특징점 위치 오차

$\mathbf{H}_f$를 QR 분해한다:

$$\mathbf{H}_f = \begin{bmatrix} \mathbf{Q}_1 & \mathbf{Q}_2 \end{bmatrix} \begin{bmatrix} \mathbf{R}_1 \\ \mathbf{0} \end{bmatrix}$$

$\mathbf{Q}_2$는 $\mathbf{H}_f$의 left null space를 구성한다 ($\mathbf{Q}_2^T \mathbf{H}_f = \mathbf{0}$). 양변에 $\mathbf{Q}_2^T$를 곱하면:

$$\mathbf{r}_o = \mathbf{Q}_2^T \mathbf{r} = \mathbf{Q}_2^T \mathbf{H}_X \tilde{\mathbf{x}} + \mathbf{Q}_2^T \mathbf{n} = \mathbf{H}_o \tilde{\mathbf{x}} + \mathbf{n}_o$$

특징점 위치 $\tilde{\mathbf{p}}_f$가 완전히 소거되었다. $\mathbf{r}_o$와 $\mathbf{H}_o$만으로 EKF 업데이트를 수행할 수 있다. 이것이 MSCKF의 "multi-state constraint"다 — 하나의 특징점이 여러 카메라 포즈에 걸쳐 만드는 기하학적 구속을 직접 이용하되, 특징점 자체는 상태에서 제외한다.

**계산 복잡도**: 상태 벡터 크기가 $15 + 6N$ (카메라 포즈 수 $N$)으로 랜드마크 수 $M$과 무관하다. EKF-SLAM은 $M$개 랜드마크를 상태에 포함하여 상태 크기가 $O(M)$이 되고, 공분산 업데이트에 $O(M^2)$이 필요하다. MSCKF는 랜드마크를 상태에서 제외하므로, 카메라 수 $N \ll M$에만 의존하는 것이 핵심 장점이다.

**MSCKF 업데이트 절차**

1. **IMU propagation**: 새 IMU 측정이 들어오면 상태를 전파하고 공분산을 업데이트한다:
   $$\hat{\mathbf{x}}_{k+1|k} = f(\hat{\mathbf{x}}_{k|k}, \mathbf{u}_k)$$
   $$\mathbf{P}_{k+1|k} = \boldsymbol{\Phi}_k \mathbf{P}_{k|k} \boldsymbol{\Phi}_k^T + \mathbf{G}_k \mathbf{Q} \mathbf{G}_k^T$$

2. **State augmentation**: 새 이미지가 들어오면 현재 IMU 포즈를 복사하여 카메라 포즈를 상태에 추가한다. 공분산 행렬도 확장한다.

3. **MSCKF update**: 추적이 끝난 특징점(더 이상 관측되지 않는 점)에 대해:
   - 삼각측량으로 $\hat{\mathbf{p}}_f$를 추정한다.
   - null-space 투영으로 $\mathbf{r}_o, \mathbf{H}_o$를 계산한다.
   - 표준 EKF 업데이트를 수행한다:
     $$\mathbf{K} = \mathbf{P} \mathbf{H}_o^T (\mathbf{H}_o \mathbf{P} \mathbf{H}_o^T + \sigma^2 \mathbf{I})^{-1}$$
     $$\hat{\mathbf{x}} \leftarrow \hat{\mathbf{x}} + \mathbf{K} \mathbf{r}_o$$
     $$\mathbf{P} \leftarrow (\mathbf{I} - \mathbf{K}\mathbf{H}_o)\mathbf{P}$$

4. **State pruning**: 슬라이딩 윈도우에서 오래된 카메라 포즈를 제거하고 공분산을 축소한다.

```cpp
// MSCKF 핵심 업데이트 수도코드 (C++)
void MSCKF::msckf_update(const Feature& feature) {
    // 1. 삼각측량으로 특징점 3D 위치 추정
    Vector3d p_f = triangulate(feature.observations, cam_states);
    
    // 2. 관측 자코비안 계산
    int N_obs = feature.observations.size();
    MatrixXd H_X(2*N_obs, state_dim);  // 상태에 대한 자코비안
    MatrixXd H_f(2*N_obs, 3);          // 특징점에 대한 자코비안
    VectorXd r(2*N_obs);               // 잔차
    
    for (int i = 0; i < N_obs; i++) {
        auto& obs = feature.observations[i];
        auto& cam = cam_states[obs.cam_id];
        
        Vector3d p_c = cam.R_w2c * (p_f - cam.p_w);
        double X = p_c(0), Y = p_c(1), Z = p_c(2);
        
        // 투영 자코비안 (pinhole)
        Matrix<double, 2, 3> J_proj;
        J_proj << 1.0/Z, 0, -X/(Z*Z),
                  0, 1.0/Z, -Y/(Z*Z);
        
        // H_f 계산
        H_f.block<2,3>(2*i, 0) = J_proj * cam.R_w2c;
        
        // H_X 계산 (카메라 포즈에 대한 부분)
        // ... (rotation, translation 자코비안)
        
        // 잔차
        r.segment<2>(2*i) = obs.uv - Vector2d(X/Z, Y/Z);
    }
    
    // 3. QR 분해로 null-space 투영
    // H_f = Q * [R1; 0] → Q2^T * H_f = 0
    HouseholderQR<MatrixXd> qr(H_f);
    MatrixXd Q = qr.householderQ();
    MatrixXd Q2 = Q.rightCols(2*N_obs - 3);
    
    MatrixXd H_o = Q2.transpose() * H_X;
    VectorXd r_o = Q2.transpose() * r;
    
    // 4. 표준 EKF 업데이트
    MatrixXd S = H_o * P * H_o.transpose() + sigma2 * MatrixXd::Identity(r_o.size(), r_o.size());
    MatrixXd K = P * H_o.transpose() * S.inverse();
    
    VectorXd dx = K * r_o;
    // 상태 업데이트 (on-manifold)
    apply_correction(dx);
    // 공분산 업데이트
    P = (MatrixXd::Identity(state_dim, state_dim) - K * H_o) * P;
}
```

### 6.3.4 OpenVINS

[OpenVINS (Geneva et al., 2020)](https://doi.org/10.1109/ICRA40945.2020.9196524)는 MSCKF 기반 VIO의 가장 완성된 오픈소스 구현이다. 단순한 구현을 넘어, 다양한 VIO 알고리즘 변형을 모듈식으로 비교/실험할 수 있는 연구 플랫폼을 지향한다.

**OpenVINS의 핵심 특징**:

1. **On-Manifold Sliding Window EKF**: MSCKF를 기반으로 한 sliding window 칼만 필터. SO(3) 매니폴드 위에서 회전을 처리한다.

2. **온라인 캘리브레이션**: 카메라 intrinsic, camera-IMU extrinsic, 시간 오프셋(temporal offset)을 런타임에 자동 추정한다. 시간 오프셋 추정은 특히 중요한데, 카메라와 IMU의 타임스탬프가 완벽히 동기화되지 않으면 성능이 크게 저하되기 때문이다.

3. **SLAM 랜드마크 지원**: 순수 MSCKF는 특징점을 상태에 포함하지 않지만, OpenVINS는 선택적으로 일부 랜드마크를 SLAM feature로 상태에 포함할 수 있다. SLAM feature는 오래 추적되는 점으로, anchored inverse depth로 파라미터화된다.

4. **First-Estimates Jacobian (FEJ)**: EKF의 일관성(consistency) 문제를 해결하기 위한 기법이다. 표준 EKF는 매 업데이트마다 최신 상태 추정치에서 자코비안을 재계산하는데, 이는 관측 가능성(observability) 속성을 위반하여 공분산을 과도하게 줄이는 문제를 일으킨다. FEJ는 자코비안을 최초 추정치(first estimate)에서만 계산하여 올바른 관측 가능성을 유지한다.

5. **시뮬레이터**: VIO 알고리즘을 테스트하기 위한 시뮬레이터를 내장하고 있다. 다양한 궤적과 환경 설정, IMU 노이즈 모델을 지원하며, ground truth와의 비교를 통해 정량적 평가가 가능하다.

### 6.3.5 Basalt

[Basalt (Usenko et al., 2020)](https://doi.org/10.1109/LRA.2019.2961227)는 VINS-Mono와 유사한 tightly-coupled VIO이지만, 몇 가지 설계 선택에서 차별화된다.

**Basalt의 핵심 특징**:

1. **Visual-only Frontend**: KLT 대신 패치 기반 direct 정합(SVO와 유사)으로 서브픽셀 정밀도의 특징점 추적을 수행한다.

2. **Non-linear Factor Recovery (NFR)**: 마지널라이제이션의 대안이다. 마지널라이제이션은 선형화 지점에 의존하는 prior를 남기는데, 이 선형화 지점이 나중에 크게 변하면 정보 왜곡이 발생한다. Basalt의 NFR은 마지널라이즈된 정보를 비선형 factor로 근사하여, 재선형화가 가능하게 한다.

3. **Efficient Implementation**: Basalt는 factor graph의 구조를 활용한 효율적 구현으로, VINS-Mono 대비 더 빠른 처리 속도를 달성한다.

4. **Stereo/Multi-camera 지원**: 여러 카메라의 시각 정보를 자연스럽게 통합한다.

---

## 6.4 VIO 설계 선택지

VIO 시스템을 설계할 때 몇 가지 핵심적인 설계 선택이 있다. 이 섹션에서는 각 선택지의 장단점과 적용 시나리오를 분석한다.

### 6.4.1 Filter vs Optimization

이것은 VIO 분야에서 가장 오래된 논쟁 중 하나다.

**Filter 기반 (MSCKF, OpenVINS)**

- 현재 상태만 유지하고, 새 측정이 올 때마다 순차적으로 업데이트
- 과거 상태는 현재 상태의 분포(mean + covariance)에 "흡수"됨
- 계산 복잡도: 업데이트당 $O(N^2)$ (N은 상태 차원)
- 장점: 일정한 계산 비용, 구현 간단
- 단점: 선형화 오차 누적 (한번 선형화하면 교정 불가), 일관성(consistency) 문제

**Optimization 기반 (VINS-Mono, ORB-SLAM3, Basalt)**

- 슬라이딩 윈도우 내 여러 상태를 동시에 유지하고, 모든 측정을 함께 최적화
- 반복 재선형화(iterative re-linearization)가 가능
- 계산 복잡도: 반복당 $O(N^3)$ (N은 윈도우 크기), 하지만 Schur complement로 효율화
- 장점: 더 정확 (재선형화로 선형화 오차 감소), 다양한 측정 통합 용이
- 단점: 계산 비용 높음, 윈도우 크기에 민감

**실험적 비교**: 같은 조건에서 최적화 기반이 필터 기반보다 일반적으로 더 정확하다. 그러나 MSCKF도 FEJ, 적절한 관측 가능성 분석 등을 적용하면 격차가 크게 줄어든다. 리소스 제약이 심한 환경(초저전력 MCU 등)에서는 필터 기반이 여전히 매력적이다.

### 6.4.2 Keyframe Selection 전략

키프레임 선택은 VIO 성능에 큰 영향을 미친다. 너무 자주 키프레임을 삽입하면 계산 부담이 커지고, 너무 드물게 삽입하면 정보 손실이 발생한다.

일반적인 키프레임 선택 기준:

1. **시간 기반**: 마지막 키프레임 이후 일정 시간 경과
2. **시차(parallax) 기반**: 현재 프레임과 마지막 키프레임 사이의 평균 특징점 시차가 임계값 초과. VINS-Mono는 이 기준을 사용한다.
3. **추적 품질 기반**: 추적된 특징점 수가 임계값 이하로 떨어지면 키프레임 삽입. ORB-SLAM3는 이 기준을 사용한다.
4. **Information gain 기반**: 새 키프레임이 제공할 정보량(정보 이득)을 추정하여 결정. 이론적으로 가장 합리적이지만 계산 비용이 높다.

키프레임 선택은 마지널라이제이션과 밀접하게 연결된다. VINS-Mono의 two-way marginalization 전략(6.3.1절 참조)은 키프레임 여부에 따라 마지널라이제이션 방향을 바꾸는 것으로, 이 연결을 잘 보여준다.

### 6.4.3 Feature Parameterization

3D 점을 어떻게 파라미터화하느냐도 중요한 설계 선택이다.

**XYZ (Euclidean 3D 좌표)**

가장 직관적이다. $\mathbf{P} = [X, Y, Z]^T \in \mathbb{R}^3$. 그러나 먼 점(far point)에서 불안정하다 — 작은 각도 변화에도 $Z$ 값이 크게 변한다.

**Inverse Depth**

점을 "호스트 프레임의 관측 방향 + 역깊이"로 표현한다:

$$\boldsymbol{\lambda} = [\theta, \phi, \rho]^T$$

여기서 $\theta, \phi$는 호스트 프레임에서의 방위각/고도각, $\rho = 1/d$는 역깊이다. 장점:

1. **원거리 점 처리**: $d \to \infty$일 때 $\rho \to 0$으로 수치적으로 안정적이다. XYZ에서는 $Z \to \infty$가 되어 불안정하다.
2. **선형성 개선**: 단안 초기화에서 깊이가 불확실할 때, 역깊이의 불확실성 분포가 가우시안에 더 가깝다.

VINS-Mono와 OpenVINS는 역깊이 파라미터화를 사용한다.

**Anchored Inverse Depth**

역깊이를 특정 "앵커" 키프레임에 고정(anchored)하여 표현한다:

$$\mathbf{P} = \mathbf{T}_{\text{anchor}} \cdot \frac{1}{\rho} [\bar{u}, \bar{v}, 1]^T$$

여기서 $(\bar{u}, \bar{v})$는 앵커 프레임에서의 정규화 좌표, $\rho$는 역깊이다. 이 파라미터화의 장점은 앵커 프레임의 포즈가 변해도 역깊이 자체는 변하지 않으므로, 부분적으로 선형화 오차를 줄인다. ORB-SLAM3와 OpenVINS에서 SLAM feature에 사용된다.

---

## 6.5 학습 기반 VO/VIO

전통적 VO/VIO는 "사람이 설계한 파이프라인"에 의존한다: 특징점 검출 → 매칭 → RANSAC → BA. 학습 기반 접근은 이 파이프라인의 일부 또는 전체를 신경망으로 대체하려 한다.

### 6.5.1 Supervised: DeepVO 계열

초기 학습 기반 VO ([DeepVO, Wang et al., 2017](https://doi.org/10.1109/ICRA.2017.7989236))는 연속 이미지 쌍을 입력으로 받아 상대 포즈를 직접 예측하는 end-to-end 네트워크를 훈련했다. CNN으로 시각 특징을 추출하고, LSTM으로 시간적 의존성을 모델링한다.

한계는 명백하다:
- 학습 데이터의 환경에 과적합 (generalization 부족)
- 기하학적 제약(에피폴라 기하 등)을 활용하지 않아 정확도가 전통 방법에 못 미침
- 스케일 드리프트가 심함

### 6.5.2 Self-supervised: 한계와 현재 위치

자기 지도 학습 VO (SfMLearner, Monodepth2 등)는 view synthesis를 통한 photometric loss로 깊이와 포즈를 동시에 학습한다. 레이블 없이 학습할 수 있다는 장점이 있지만, 움직이는 물체(moving objects), 텍스처 부족, occlusion 등에서 어려움을 겪는다.

현재 위치: 자기 지도 학습 VO는 단안 깊이 추정(monocular depth estimation)에서는 큰 성과를 거두었지만, 순수 VO/VIO 시스템으로서는 전통 방법에 크게 못 미친다. 특히 누적 드리프트 문제를 해결하지 못했다.

### 6.5.3 Hybrid: DROID-SLAM

[DROID-SLAM (Teed & Deng, 2021)](https://arxiv.org/abs/2108.10869)은 전통적 BA의 기하학적 엄밀성과 딥러닝의 강건한 매칭 능력을 하나의 미분 가능 파이프라인으로 통합한 시스템이다. 학습 기반 SLAM이 전통적 시스템을 모든 지표에서 능가할 수 있음을 최초로 입증했다.

**아키텍처**

DROID-SLAM은 두 컴포넌트로 작동한다:

1. **RAFT 기반 반복 업데이트 연산자 (Iterative Update Operator)**

RAFT (Recurrent All-Pairs Field Transforms, Teed & Deng, 2020)에서 영감을 받은 구조다. 프레임 쌍 $(i, j)$에 대해:
- 두 프레임의 feature map에서 4D correlation volume을 계산한다.
- 현재 포즈/깊이 추정으로부터 유도된 correspondence field를 기준으로, correlation volume에서 특징을 인덱싱한다.
- 3×3 Convolutional GRU가 correlation 특징, 현재 flow, context 특징을 입력받아 **flow revision** $\Delta\mathbf{f}_{ij}$와 **confidence weight** $\mathbf{w}_{ij}$를 출력한다.

$$(\Delta\mathbf{f}_{ij}, \mathbf{w}_{ij}) = \text{ConvGRU}(\text{corr}_{ij}, \mathbf{f}_{ij}^{\text{curr}}, \text{context}_i)$$

이 업데이트는 반복적으로(iteratively) 수행되어 correspondence를 점진적으로 정제한다.

2. **미분 가능 Dense Bundle Adjustment (DBA) 레이어**

GRU가 출력한 flow revision을 기하학적 업데이트로 변환하는 레이어다. 핵심은: 모든 픽셀에 대해 reprojection error를 정의하고, 이를 카메라 포즈 $\mathbf{T}_i \in SE(3)$와 역깊이 $d_i$에 대해 Gauss-Newton으로 최소화한다:

$$\sum_{(i,j)} \sum_{\mathbf{p}} \left\| \mathbf{w}_{ij}^{\mathbf{p}} \circ (\mathbf{p}^{*}_{ij} - \pi(\mathbf{T}_j \circ \mathbf{T}_i^{-1} \circ \pi^{-1}(\mathbf{p}, d_i^{\mathbf{p}}))) \right\|^2$$

여기서 $\mathbf{p}^{*}_{ij} = \mathbf{p} + \mathbf{f}_{ij}^{\text{curr}} + \Delta\mathbf{f}_{ij}$는 target correspondence, $\mathbf{w}_{ij}^{\mathbf{p}}$는 confidence weight이다.

Gauss-Newton 업데이트를 유도하면 정규 방정식이 나온다:

$$\begin{bmatrix} \mathbf{B} & \mathbf{E} \\ \mathbf{E}^T & \mathbf{C} \end{bmatrix} \begin{bmatrix} \boldsymbol{\xi} \\ \delta\mathbf{d} \end{bmatrix} = \begin{bmatrix} \mathbf{v} \\ \mathbf{w} \end{bmatrix}$$

여기서 $\boldsymbol{\xi}$는 포즈 업데이트(SE(3) tangent space), $\delta\mathbf{d}$는 역깊이 업데이트, $\mathbf{B}$는 포즈 블록, $\mathbf{C}$는 깊이 블록(대각), $\mathbf{E}$는 교차 블록이다.

전통 BA와 동일하게 Schur complement를 적용한다:

$$(\mathbf{B} - \mathbf{E}\mathbf{C}^{-1}\mathbf{E}^T)\boldsymbol{\xi} = \mathbf{v} - \mathbf{E}\mathbf{C}^{-1}\mathbf{w}$$

$\mathbf{C}$는 대각이므로 $\mathbf{C}^{-1}$은 trivial하다. 축소된 시스템은 카메라 수에만 의존하므로 효율적이다.

핵심 혁신은: 이 전체 Gauss-Newton 솔버가 **미분 가능**하다는 것이다. 역전파를 통해 GRU의 파라미터가 "좋은 correspondence"를 출력하도록 학습된다.

**프레임 그래프와 루프 클로저**

DROID-SLAM은 co-visibility 기반 프레임 그래프를 동적으로 구축한다. 새 프레임이 추가될 때 이웃 프레임과 엣지를 연결하고, 과거 프레임과의 co-visibility가 감지되면 장거리 엣지를 추가하여 루프 클로저를 수행한다. 백엔드에서는 전체 키프레임 히스토리에 대한 글로벌 BA를 수행한다.

**다중 모달리티 지원**

DROID-SLAM은 단안 영상만으로 학습했음에도 추론 시 스테레오와 RGB-D를 직접 활용할 수 있다:
- **스테레오**: 왼쪽/오른쪽 프레임을 별도 프레임으로 취급하되, 알려진 baseline으로 상대 포즈를 고정한다.
- **RGB-D**: 깊이 정보를 역깊이 초기값으로 사용하고, 깊이 관측을 추가 제약으로 반영한다.

**성능**

- TartanAir: 기존 최고 대비 오차 62% 감소
- EuRoC (monocular): 82% 감소
- ETH-3D: 32개 중 30개 시퀀스 성공 (기존 최고 19개)
- 합성 데이터(TartanAir)만으로 학습 후 4개 실제 데이터셋에서 모두 SOTA

**왜 중요한가**

DROID-SLAM은 기하학적 엄밀성(BA)과 데이터 기반 강건성(learned correspondence)을 결합한 최초의 실용적 시스템이다. 전통 방법이 실패하는 환경(반복 패턴, 텍스처 부족, 급격한 조명 변화)에서도 안정적으로 동작한다.

그러나 한계도 있다:
- **실시간성**: GPU가 필수이며, 현재 실시간보다 느리다.
- **IMU 미포함**: 순수 시각 시스템으로, IMU 결합은 아직 연구 과제다.
- **메모리**: 모든 프레임의 feature map과 correlation volume을 유지해야 하므로 메모리 사용량이 크다.

현재 후속 연구가 이 한계를 하나씩 해결하고 있으며, 학습 기반 VO/VIO는 빠르게 전통 방법을 추격하고 있다.

### 6.5.4 최근 동향 (2023-2025)

[DPVO (Teed & Deng, 2023)](https://arxiv.org/abs/2208.04726)는 DROID-SLAM의 dense flow를 sparse patch 기반 매칭으로 대체하여, 메모리를 1/3로 줄이고 속도를 3배 향상시키면서도 동등 이상의 정확도를 달성했다. 패치 단위 recurrent update operator와 미분 가능 BA를 결합하여 실시간에 가까운 학습 기반 VO를 실현했다.

[MAC-VO (Qu et al., 2024)](https://arxiv.org/abs/2409.09479)는 학습 기반 매칭 불확실성(metrics-aware covariance)을 스테레오 VO에 도입하여, 키포인트 선택과 포즈 그래프 최적화의 잔차 가중치를 불확실성으로 결정한다. 조명 변화와 텍스처 부족 환경에서 기존 VO/SLAM 시스템을 능가하며, ICRA 2025 Best Paper로 선정되었다.

---

## 6장 요약

| 시스템 | 유형 | 추정 방법 | 센서 | 핵심 특징 |
|--------|------|-----------|------|-----------|
| ORB-SLAM3 | Feature-based | Optimization (BA) | Mono/Stereo/RGBD + IMU | Multi-map Atlas, fisheye 지원 |
| DSO | Direct | Optimization (windowed) | Mono | Photometric calibration, sparse sampling |
| SVO | Semi-direct | Optimization (BA) | Mono/Stereo | 200-400Hz, 고속 드론 적합 |
| VINS-Mono | Feature-based | Optimization (sliding window) | Mono + IMU | 강건한 초기화, 4-DoF loop closure |
| MSCKF | Feature-based | EKF (sliding window) | Mono/Stereo + IMU | 랜드마크를 상태에서 제외, null-space 투영 |
| OpenVINS | Feature-based | EKF (MSCKF) | Mono/Stereo + IMU | 온라인 캘리브레이션, FEJ, 연구 플랫폼 |
| Basalt | Semi-direct | Optimization (sliding window) | Stereo + IMU | NFR, 효율적 구현 |
| DROID-SLAM | Learned | Differentiable BA | Mono/Stereo/RGBD | 미분 가능 BA, 합성 데이터 학습 |
| DPVO | Learned (sparse) | Differentiable BA | Mono | DROID 대비 3배 빠르고 메모리 1/3 |
| MAC-VO | Learned + Opt. | Pose graph opt. | Stereo | Metrics-aware covariance, ICRA 2025 Best Paper |

다음 챕터에서는 LiDAR 기반 오도메트리와 LiDAR-Inertial 결합 시스템을 다루며, 카메라 기반 시스템과의 상보적 관계를 분석한다.
