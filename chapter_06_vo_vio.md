# Ch.6 — Visual Odometry & Visual-Inertial Odometry

Visual Odometry(VO)와 Visual-Inertial Odometry(VIO)는 Ch.4의 상태 추정 프레임워크와 Ch.5의 특징점 매칭을 한 시스템으로 묶는다.

VO는 카메라 영상만으로 카메라의 자기 운동(ego-motion)을 추정한다. VIO는 여기에 IMU를 더해 스케일을 관측할 수 있게 하고 추정을 더 강건하게 만든다.

VO의 기원은 [Nistér et al. (2004)](https://doi.org/10.1109/CVPR.2004.1315094)로 거슬러 올라간다. 이 논문은 "Visual Odometry"라는 용어를 처음 정의하고, 스테레오·단안 카메라로 실시간 자기 운동 추정 시스템을 제시했다. 스테레오 접근에서는 좌우 카메라에서 삼각측량한 3D 점에 3-point 알고리즘을 적용해 프레임 간 강체 변환을 추정했고, 단안 접근에서는 5-point 알고리즘으로 Essential Matrix를 추정했다. 특징점 검출, 매칭, RANSAC, 모션 추정으로 이어지는 이 파이프라인은 20년이 지난 지금도 feature-based VO의 뼈대다.

VO·VIO 시스템은 다음 기준으로 나눈다.

1. **Feature-based vs Direct**: 기하학적 특징점(corner, edge)을 추출해 매칭하는 방식과 픽셀 밝기를 직접 쓰는 방식의 구분
2. **Filter vs Optimization**: 상태 추정에 칼만 필터 계열을 쓰는가, 비선형 최적화를 쓰는가
3. **Loosely coupled vs Tightly coupled**: IMU와 카메라를 독립적으로 처리한 뒤 결과를 합치는가, raw measurement를 하나의 추정 프레임워크에서 결합하는가

---

## 6.1 Feature-based Visual Odometry

Feature-based VO는 영상에서 기하학적 특징점을 뽑고, 프레임 사이의 대응으로 카메라 모션을 추정한다. 오래 연구된 방식이라 구성 요소와 실패 양상(failure mode)이 비교적 잘 알려져 있고, ORB-SLAM3 같은 공개 시스템도 이 방식을 쓴다.

### 6.1.1 Frontend: Detection, Tracking, Outlier Rejection

Feature-based VO의 프론트엔드는 다음 작업을 한다.

**특징점 검출 (Feature Detection)**

프레임에서 추적할 만한 점을 찾는 단계다. 좋은 특징점은 반복성(repeatability)이 높다. 같은 3D 점을 다른 시점에서 찍어도 영상마다 그 점의 투영 위치가 다시 검출된다.

Harris corner detector는 이미지 패치의 자기상관 행렬(autocorrelation matrix, 또는 second moment matrix) $\mathbf{M}$으로 코너를 찾는다.

$$\mathbf{M} = \sum_{(x,y) \in W} w(x,y) \begin{bmatrix} I_x^2 & I_xI_y \\ I_xI_y & I_y^2 \end{bmatrix}$$

여기서 $I_x, I_y$는 이미지 그래디언트, $W$는 윈도우, $w$는 가중치 함수다. $\mathbf{M}$의 두 고유값이 모두 크면 코너로 본다. Harris 응답 함수는 다음과 같다.

$$R = \det(\mathbf{M}) - k \cdot \text{tr}(\mathbf{M})^2 = \lambda_1\lambda_2 - k(\lambda_1 + \lambda_2)^2$$

FAST (Features from Accelerated Segment Test)는 속도를 우선한 검출기다. 후보 픽셀 $p$ 주위 반지름 3인 원(Bresenham circle) 위 점 16개 가운데 $n$개 이상이 연속으로 $p$보다 밝거나 어두우면 코너로 본다. 처음 제안된 검출기(Rosten & Drummond, ICCV 2005)는 $n=12$를 썼다. 결정 트리 학습을 더한 후속 연구(ECCV 2006, TPAMI 2010)에서는 $n=9$의 반복성이 가장 좋았고, OpenCV의 기본값도 $n=9$(`TYPE_9_16`)다. 1, 5, 9, 13번 점만 먼저 보는 4점 사전 검사는 $n=12$ 전용 단축법이다. $n=9$에서는 학습된 결정 트리가 검사 순서를 정한다. Harris와의 속도 차이는 구현과 하드웨어에 따라 달라진다. 방향·스케일 불변성은 ORB 같은 확장에서 더해진다.

ORB (Oriented FAST and Rotated BRIEF)는 FAST 검출에 방향을 붙이고 BRIEF 디스크립터를 그 방향으로 돌려, 실시간 SLAM에 맞는 특징점을 만든다. 방향은 이미지 패치의 intensity centroid로 계산한다.

$$\theta = \text{atan2}(m_{01}, m_{10}), \quad m_{pq} = \sum_{x,y} x^p y^q I(x,y)$$

ORB-SLAM 시리즈는 스케일 변화에 대응하려고 이미지 피라미드에서 ORB를 뽑는다. [ORB-SLAM 논문 (Mur-Artal et al., 2015)](https://arxiv.org/abs/1502.00956)의 설정은 8레벨, 스케일 팩터 1.2다.

**특징점 추적 (Feature Tracking)**

프레임 사이의 대응은 다음 방법으로 찾는다.

1. **디스크립터 매칭**: 프레임마다 따로 특징점을 검출하고, 디스크립터 사이의 거리(이진 디스크립터는 Hamming distance, float 디스크립터는 L2)로 매칭한다. ORB-SLAM3가 이 방식을 쓴다. DBoW2 어휘 트리(§9.2.1)로 매칭 후보를 좁혀 속도를 높인다.

2. **광학 흐름 추적 (Optical Flow Tracking)**: Lucas-Kanade (LK) 추적기로 이전 프레임의 특징점이 현재 프레임에서 어디로 옮겨 갔는지 따라간다. VINS-Mono가 이 방식을 쓴다. 밝기 항상성 가정(brightness constancy assumption)을 두면 다음 식이 선다.

$$I(x + u, y + v, t + \Delta t) = I(x, y, t)$$

1차 Taylor 전개를 하고 윈도우 $W$ 안에서 최소자승으로 풀면 다음과 같다.

$$\begin{bmatrix} u \\ v \end{bmatrix} = \left(\sum_W \begin{bmatrix} I_x^2 & I_xI_y \\ I_xI_y & I_y^2 \end{bmatrix}\right)^{-1} \sum_W \begin{bmatrix} -I_xI_t \\ -I_yI_t \end{bmatrix}$$

이미지 피라미드에서 coarse-to-fine으로 추적하면 변위가 커도 따라간다.

디스크립터 매칭은 baseline이 넓어도(시점 변화가 커도) 잘 버티고 디스크립터를 loop closure에 다시 쓸 수 있지만, 검출·기술·매칭에 시간이 든다. LK 추적은 빠르고 서브픽셀 정밀도를 내지만, 시점이 크게 바뀌면 놓치기 쉽다.

**아웃라이어 제거 (Outlier Rejection)**

매칭 결과에 오매칭(outlier)이 섞이면 모션 추정이 크게 틀어진다.

기본 도구는 RANSAC (Random Sample Consensus)이다. 최소 샘플로 모델을 추정하고, 전체 데이터에서 인라이어 수를 세어 가장 많은 모델을 고른다. VO에서는 대응 종류에 따라 다음을 쓴다.

- **2D-2D**: 5-point 알고리즘으로 Essential Matrix $\mathbf{E}$를 추정하고, $\mathbf{E} = [\mathbf{t}]_\times \mathbf{R}$로 분해해 상대 포즈 $(R, t)$를 얻는다.
- **3D-2D**: PnP (Perspective-n-Point) 알고리즘에 RANSAC을 붙인다. 이미 삼각측량한 3D 점과 현재 2D 관측의 대응으로 절대 포즈를 추정한다.
- **3D-3D**: ICP 변형으로 강체 변환을 추정한다.

Epipolar constraint로 아웃라이어를 거를 때는 다음 식을 쓴다.

$$\mathbf{p}_2^T \mathbf{E} \mathbf{p}_1 = 0$$

여기서 $\mathbf{p}_1, \mathbf{p}_2$는 정규화된 이미지 좌표다. 실제로는 노이즈 때문에 0이 정확히 나오지 않으므로, 이 식의 오차(예: Sampson 거리)가 임계값을 넘는 대응을 아웃라이어로 본다.

ORB-SLAM3는 Fundamental Matrix와 Homography를 동시에 추정하고 두 모델의 점수 비율로 장면 구조(평면 vs 비평면)에 맞는 쪽을 고른다. 평면에서 Homography를 쓰는 까닭은 퇴화에 있다. 평면 위의 대응만으로는 $\mathbf{F}$가 유일하게 정해지지 않는다. 자유도는 $\mathbf{F}$가 7, $\mathbf{H}$가 8로 $\mathbf{H}$ 쪽이 더 많다.

### 6.1.2 Backend: PnP, Motion-only BA, Local BA

프론트엔드가 대응을 넘기면 백엔드가 그것으로 카메라 포즈와 3D 구조를 추정한다.

**PnP (Perspective-n-Point)**

이미 삼각측량한 3D 맵포인트 $\mathbf{P}_j \in \mathbb{R}^3$와 현재 프레임의 2D 관측 $\mathbf{u}_j \in \mathbb{R}^2$의 대응을 알 때 카메라 포즈 $\mathbf{T} = (\mathbf{R}, \mathbf{t}) \in SE(3)$를 추정하는 문제다.

$$\mathbf{u}_j = \pi(\mathbf{R}\mathbf{P}_j + \mathbf{t})$$

여기서 $\pi: \mathbb{R}^3 \to \mathbb{R}^2$는 카메라 투영 함수다. P3P는 3점으로 해를 최대 4개 내고, 1점을 더 써서 하나를 고른다. RANSAC과 함께 써서 아웃라이어를 거른다.

초기 해를 구한 뒤에는 모든 인라이어에 대해 비선형 최적화(refinement)로 해를 다듬는다.

**Motion-only Bundle Adjustment**

PnP로 초기 포즈를 얻은 뒤 3D 점 위치는 고정하고 카메라 포즈만 최적화한다.

$$\mathbf{T}^* = \underset{\mathbf{T}}{\arg\min} \sum_{j} \rho\left(\left\|\mathbf{u}_j - \pi(\mathbf{T} \cdot \mathbf{P}_j)\right\|^2_{\Sigma_j}\right)$$

여기서 $\rho(\cdot)$는 robust kernel (Huber 등), $\Sigma_j$는 관측 공분산이다. Huber 함수는:

$$\rho(s) = \begin{cases} s & \text{if } s \leq \delta^2 \\ 2\delta\sqrt{s} - \delta^2 & \text{if } s > \delta^2 \end{cases}$$

변수는 SE(3) 위의 6-DoF 포즈이므로 Lie algebra $\mathfrak{se}(3)$의 증분 $\boldsymbol{\xi} \in \mathbb{R}^6$으로 갱신한다.

$$\mathbf{T} \leftarrow \exp(\boldsymbol{\xi}^{\wedge}) \cdot \mathbf{T}$$

**Local Bundle Adjustment**

최근 키프레임 $N$개와 이들이 관측하는 맵포인트를 함께 최적화한다:

$$\{\mathbf{T}_i^*, \mathbf{P}_j^*\} = \underset{\{\mathbf{T}_i, \mathbf{P}_j\}}{\arg\min} \sum_{i,j} \rho\left(\left\|\mathbf{u}_{ij} - \pi(\mathbf{T}_i \cdot \mathbf{P}_j)\right\|^2_{\Sigma_{ij}}\right)$$

Bundle Adjustment(BA)의 기본 형태다. "Bundle"은 3D 점마다 카메라들로 뻗는 광선(ray)의 다발을 가리킨다. BA의 정규 방정식(normal equations)은 Schur complement를 쓸 수 있는 희소 구조다.

$$\begin{bmatrix} \mathbf{H}_{cc} & \mathbf{H}_{cp} \\ \mathbf{H}_{pc} & \mathbf{H}_{pp} \end{bmatrix} \begin{bmatrix} \delta\boldsymbol{\xi} \\ \delta\mathbf{p} \end{bmatrix} = \begin{bmatrix} \mathbf{b}_c \\ \mathbf{b}_p \end{bmatrix}$$

$\mathbf{H}_{pp}$는 블록 대각(각 점은 독립)이므로, Schur complement로 점 변수를 소거하면:

$$(\mathbf{H}_{cc} - \mathbf{H}_{cp}\mathbf{H}_{pp}^{-1}\mathbf{H}_{pc})\delta\boldsymbol{\xi} = \mathbf{b}_c - \mathbf{H}_{cp}\mathbf{H}_{pp}^{-1}\mathbf{b}_p$$

줄인 시스템의 크기는 점 수와 상관없이 카메라 수로만 정해진다.

### 6.1.3 ORB-SLAM3 아키텍처 상세 분석

[ORB-SLAM3 (Campos et al., 2021)](https://doi.org/10.1109/TRO.2021.3075644)는 공개된 feature-based visual(-inertial) SLAM 시스템이다. 한 프레임워크에서 monocular/stereo/RGB-D 카메라와 IMU를 모두 지원하고, pinhole과 fisheye 렌즈 모델을 다룬다.

**전체 아키텍처**

ORB-SLAM3는 다음 스레드를 나란히 돌린다.

1. **Tracking Thread**: 프레임마다 ORB 특징점을 뽑고 기존 맵포인트와 매칭해 현재 포즈를 추정한다. 포즈는 Motion-only BA로 다듬는다.
2. **Local Mapping Thread**: 새 키프레임이 들어오면 새 맵포인트를 삼각측량하고 local BA를 돌린다. 중복된 키프레임과 맵포인트는 골라 지워(culling) 맵이 너무 커지지 않게 한다.
3. **Loop Closing & Map Merging Thread**: DBoW2로 루프 후보를 검출한 뒤 Sim(3) (또는 SE(3)) 정합으로 기하학적 유효성을 검증한다. 루프가 확인되면 pose graph optimization과 full BA를 차례로 실행한다.

**Tracking Thread 상세**

1. 이전 프레임 기반 초기 포즈 예측: 등속 모델(constant velocity model) 또는 IMU preintegration으로 현재 포즈를 예측한다.
2. 예측한 포즈로 맵포인트를 현재 프레임에 투영하고(projection), ORB 매칭으로 대응을 찾는다.
3. Motion-only BA로 포즈를 정제한다.
4. 키프레임 여부를 결정한다. 키프레임 조건: (a) 마지막 키프레임으로부터 일정 프레임 이상 경과, (b) 현재 프레임의 추적된 맵포인트 비율이 임계값 이하, (c) 현재 프레임에서 충분한 수의 맵포인트가 관측됨.

**Visual-Inertial 모드**

ORB-SLAM3의 visual-inertial 모드는 [Forster et al. (2017)](https://doi.org/10.1109/TRO.2016.2597321)의 IMU preintegration과 MAP(Maximum-a-Posteriori) 추정을 쓴다. 시각 잔차와 IMU 잔차를 한 비용 함수에 넣어 함께 최적화한다.

$$\mathcal{C} = \sum_{i,j} \rho\left(\left\|\mathbf{e}_{ij}^{\text{vis}}\right\|^2_{\Sigma_{ij}}\right) + \sum_k \left\|\mathbf{e}_k^{\text{IMU}}\right\|^2_{\Sigma_k^{\text{IMU}}} + \left\|\mathbf{e}^{\text{prior}}\right\|^2_{\Sigma^{\text{prior}}}$$

IMU 잔차 $\mathbf{e}_k^{\text{IMU}}$는 키프레임 $k$와 $k+1$ 사이의 preintegrated IMU 측정과 상태 추정치의 차이다. 키프레임마다 상태 벡터는 다음과 같다.

$$\mathbf{x}_k = [{}^W\mathbf{R}_k, {}^W\mathbf{p}_k, {}^W\mathbf{v}_k, \mathbf{b}_k^g, \mathbf{b}_k^a]$$

§4.6.2에서 유도한 preintegrated 측정

$$\Delta\mathbf{R}_{k,k+1}, \quad \Delta\mathbf{v}_{k,k+1}, \quad \Delta\mathbf{p}_{k,k+1}$$

으로 IMU 잔차를 다음처럼 쓴다.

$$\mathbf{e}^{\text{IMU}}_{\Delta R} = \text{Log}\left(\Delta\hat{\mathbf{R}}_{k,k+1}^T \cdot \mathbf{R}_k^T \mathbf{R}_{k+1}\right)$$
$$\mathbf{e}^{\text{IMU}}_{\Delta v} = \mathbf{R}_k^T(\mathbf{v}_{k+1} - \mathbf{v}_k - \mathbf{g}\Delta t) - \Delta\hat{\mathbf{v}}_{k,k+1}$$
$$\mathbf{e}^{\text{IMU}}_{\Delta p} = \mathbf{R}_k^T(\mathbf{p}_{k+1} - \mathbf{p}_k - \mathbf{v}_k\Delta t - \frac{1}{2}\mathbf{g}\Delta t^2) - \Delta\hat{\mathbf{p}}_{k,k+1}$$

**Multi-Map System (Atlas)**

시각 정보가 모자란 구간(빠른 회전, 가림 등)에서 트래킹이 끊기면, 그 전 시스템들은 다시 초기화하면서 이전 맵과의 연결을 잃었다. ORB-SLAM3의 Atlas는 다음 순서로 동작한다.

1. 트래킹이 끊기면 새 맵(sub-map)을 만든다.
2. 각 sub-map은 따로 유지한다.
3. 장소 인식(DBoW2)으로 이전에 방문한 sub-map을 감지하면, 두 맵을 자동으로 병합(merging)한다.
4. 병합할 때는 Sim(3)(단안의 경우) 또는 SE(3)(스테레오/VI의 경우)로 정합한다.

같은 방식으로 지난 세션의 맵을 다시 쓰는 multi-session SLAM도 지원한다.

**ORB-SLAM3의 평가 범위**: 원 논문은 EuRoC MAV와 TUM-VI에서 stereo-inertial 구성을 평가하고, 각 데이터셋의 프로토콜에 따른 궤적 오차를 보고한다. 수치를 비교하려면 ATE/RPE 중 어떤 지표인지, 대상 시퀀스와 집계 방식, 초기화 및 트래킹 실패 처리까지 함께 밝혀야 한다. 새 카메라·IMU 조합에서는 목표 시퀀스의 ATE/RPE, 실패율, 처리 시간과 메모리를 다시 측정한다.

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

Direct 방법은 특징점을 뽑지 않고 픽셀 밝기(intensity) 자체를 관측으로 쓴다. 바탕에는 3D 점 $\mathbf{P}$가 두 프레임에서 같은 밝기로 찍힌다는 가정(brightness constancy)이 있다.

### 6.2.1 Photometric Error

Direct VO는 photometric error를 잔차로 쓴다. 카메라 포즈 $\mathbf{T}_i, \mathbf{T}_j$와 3D 점 $\mathbf{P}$(또는 호스트 프레임의 픽셀 $\mathbf{u}$와 역깊이 $d^{-1}$로 파라미터화)에 대해:

$$e_{\text{photo}} = I_j\left(\pi(\mathbf{T}_j \mathbf{T}_i^{-1} \pi^{-1}(\mathbf{u}_i, d_i^{-1}))\right) - I_i(\mathbf{u}_i)$$

여기서:
- $\pi^{-1}(\mathbf{u}, d^{-1})$: 2D 점과 역깊이로부터 3D 점 복원 (unprojection)
- $\pi(\cdot)$: 3D→2D 투영
- $I_i, I_j$: 프레임 $i, j$의 밝기 이미지

이 잔차를 최소화해 포즈와 깊이를 추정한다. Feature-based 방법의 reprojection error와 비교하면 다음과 같다.

| | Reprojection error | Photometric error |
|---|---|---|
| 관측 | 특징점 좌표 (2D) | 픽셀 밝기 (1D 또는 패치) |
| 데이터 연관 | 명시적 (매칭 필요) | 암묵적 (와핑으로 계산) |
| 텍스처 요구 | 코너 같은 keypoint 필요 | 에지나 밝기 변화가 약한 영역도 씀 |
| 조명 변화 | detector·descriptor가 대부분 흡수 | 민감 (photometric calibration 필요) |

표의 판정은 [DSO 논문(arXiv 1607.02565)](https://arxiv.org/abs/1607.02565)의 1절 direct/indirect 구분과 2.1절 calibration 설명에 바탕을 둔다. 조명 변화 행은 "흔히 쓰는 feature extractor와 descriptor는 광도 변화에 불변이거나 매우 강건하다"는 2.1절의 문장에 근거한다.

Photometric error는 특징점을 따로 매칭하지 않고 계산한다. 그래디언트가 있는 영역이면 어디든 쓰므로, 하얀 벽의 옅은 결처럼 그래디언트가 약한 곳에서도 동작한다.

한계:
1. **밝기 항상성 위반**: 조명 변화, 자동 노출, 렌즈 비네팅 등으로 같은 3D 점의 밝기가 프레임마다 달라진다. DSO는 노출 시간, 비네팅, 응답 함수를 photometric calibration으로 보정한다.
2. **제한된 수렴 영역**: 이미지 그래디언트 기반 최적화는 초기 포즈가 나쁘면 다른 로컬 미니멈으로 갈 수 있다. 허용되는 초기 오차는 피라미드, 텍스처, 시점·밝기 변화에 따라 달라지므로 coarse-to-fine 정렬과 검증 잔차로 확인한다.

### 6.2.2 DSO (Direct Sparse Odometry) 아키텍처 상세 분석

[DSO (Engel et al., 2018)](https://doi.org/10.1109/TPAMI.2017.2658577)는 direct 방법과 sparse 표현을 결합한 VO 시스템이다. 기존에는 "direct = dense 또는 semi-dense", "sparse = indirect"라는 암묵적 등식이 있었다. Newcombe et al.의 DTAM(ICCV 2011)이 dense, [LSD-SLAM(Engel et al., 2014)](https://doi.org/10.1007/978-3-319-10605-2_54)이 semi-dense, ORB-SLAM이 sparse·indirect의 예다. DSO는 잔차 방식과 표현 밀도라는 두 축을 새롭게 조합했다.

**DSO의 설계**

1. **Direct**: 특징점 없이 픽셀 밝기를 직접 사용
2. **Sparse**: 이미지 전체가 아니라, 그래디언트가 있는 영역에서 균등하게 점을 샘플링
3. **Joint Optimization**: 포즈, 역깊이, 카메라 내부 파라미터와 affine brightness parameters를 동시 최적화

**완전한 Photometric Calibration**

DSO는 photometric calibration을 모델에 넣는다. 실제 카메라가 기록한 밝기 $I'$과 이미지 평면에 들어온 복사조도(irradiance) $B$의 관계는 다음과 같다.

$$I'(\mathbf{u}) = G(t \cdot V(\mathbf{u}) \cdot B(\mathbf{u}))$$

여기서:
- $G(\cdot)$: 카메라의 비선형 응답 함수 (response function)
- $t$: 노출 시간 (exposure time)
- $V(\mathbf{u})$: 렌즈 비네팅 (vignetting) — 이미지 중심에서 멀어질수록 밝기 감소

이 세 요소를 모두 보정하면 photometric 보정 이미지를 얻는다.

$$I(\mathbf{u}) = t^{-1} \cdot G^{-1}(I'(\mathbf{u})) / V(\mathbf{u})$$

보정하지 않으면 같은 3D 점도 이미지 중심과 가장자리에서, 또는 노출이 다른 프레임에서 밝기가 달라 photometric error가 틀어진다.

**Affine Brightness Transfer Function**

photometric calibration이 완벽하지 않을 때를 대비해, DSO는 프레임 사이의 밝기 변화를 affine 모델로 한 번 더 보정한다.

$$e_{\mathbf{p}j} = \sum_{\mathbf{p} \in \mathcal{N}_\mathbf{p}} w_\mathbf{p} \left\| (I_j[\mathbf{p}'] - b_j) - \frac{t_j e^{a_j}}{t_i e^{a_i}}(I_i[\mathbf{p}] - b_i) \right\|_\gamma$$

여기서 $a_i, b_i, a_j, b_j$는 프레임별 affine brightness 파라미터이고, 다른 변수와 함께 최적화한다. $\|\cdot\|_\gamma$는 Huber norm이다.

**점 선택 전략**

DSO는 이미지를 격자로 나누고 셀마다 그래디언트가 가장 큰 점을 골라, 점이 한 영역에 몰리지 않게 한다. 점은 2000개쯤 고르고, 그래디언트 임계값을 영역에 맞춰 바꿔 텍스처가 적은 곳에서도 점을 얻는다.

**Sliding Window Optimization**

DSO는 활성 키프레임 최대 7개($N_f = 7$)와 거기 속한 점들의 역깊이를 슬라이딩 윈도우 안에서 함께 최적화한다. 최적화 변수는 다음과 같다.

$$\boldsymbol{\theta} = \{\mathbf{T}_1, \ldots, \mathbf{T}_n, d_1^{-1}, \ldots, d_m^{-1}, \mathbf{c}, a_1, b_1, \ldots, a_n, b_n\}$$

카메라 포즈(SE(3)), 역깊이(inverse depth), 카메라 내부 파라미터 $\mathbf{c}$, affine brightness 파라미터가 모두 들어간다.

윈도우에서 빠지는 프레임과 점은 Schur complement로 마지널라이즈해 prior로 남긴다. 원리는 §4.7의 Schur complement 기반 마지널라이제이션과 같고, 대상은 시각 잔차다.

**DSO의 한계와 확장**

DSO는 설계상 odometry에 한정되며 loop closure는 범위 밖에 두었다. LDSO (Loop-closing DSO)는 DBoW와 direct 정합을 결합해 loop closure를 더했다. VI-DSO는 DSO에 IMU를 결합한 확장판이다. Basalt는 패치 기반 KLT 추적 frontend와 Non-linear Factor Recovery를 갖춘 별도 시스템이다(§6.3.5).

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

[SVO (Semi-direct Visual Odometry, Forster et al., 2017)](https://doi.org/10.1109/TRO.2016.2623335)는 feature-based 방법과 direct 방법을 섞었다. 추적(tracking)과 삼각측량에는 direct 방법을, 구조와 모션의 공동 최적화에는 feature-based 방법을 써서 "semi-direct"라는 이름이 붙었다.

**SVO의 처리 단계**

1. **Sparse Model-based Image Alignment**: 기존 3D 맵포인트를 현재 프레임에 투영하고, 투영점 주위 패치의 photometric error를 최소화해 프레임 포즈를 추정한다. DSO처럼 이미지 그래디언트를 직접 쓰지만, 이미지 전체가 아니라 아는 맵포인트 주위만 처리해 계산을 줄인다.

2. **Feature Alignment**: 포즈를 구한 뒤 맵포인트마다 투영 위치를 서브픽셀 정밀도로 다듬는다. 여기에도 패치 기반 direct 정합을 쓴다.

3. **Structure & Motion Refinement**: 다듬은 2D 위치를 "가상 특징점"으로 보고, BA(reprojection error 최소화)로 포즈와 3D 구조를 함께 최적화한다.

SVO 논문은 이 3단계 분리로 저자들의 해상도·하드웨어 설정에서 수백 Hz의 처리율을 보고했다. 실제 rate는 카메라 해상도, 맵 크기, CPU와 구현에 따라 달라지므로 target platform에서 측정해야 한다. 기본 구성은 odometry에 한정된다. depth update에 필요한 translation이 부족한 pure rotation 구간은 별도로 처리해야 한다.

---

## 6.3 Tightly-Coupled Visual-Inertial Odometry

카메라만 쓰는 VO는 단안이면 스케일을 알 수 없고, 빠르게 움직이거나 텍스처가 부족하면 트래킹이 끊긴다. IMU를 더하면 스케일을 관측할 수 있고, 영상이 나쁜 짧은 구간을 IMU 적분으로 넘긴다. Tightly-coupled VIO는 카메라와 IMU의 raw 측정을 한 추정 프레임워크에서 함께 처리한다.

### 6.3.1 VINS-Mono 아키텍처 상세

[VINS-Mono (Qin et al., 2018)](https://doi.org/10.1109/TRO.2018.2853729)는 단안 카메라 하나와 저가 IMU로 6-DoF 상태를 추정하는 VIO 시스템이다. 초기화, 오도메트리, loop closure, 맵 재사용까지 한 시스템에 들어 있다.

**시스템 아키텍처**

VINS-Mono의 모듈 구성:

1. **프론트엔드 (Measurement Preprocessing)**: KLT 기반 특징점 추적 + IMU preintegration
2. **백엔드 (Estimator)**: 비선형 최적화 기반 tightly-coupled VIO
3. **Loop closure (Relocalization)**: DBoW2 기반 장소 인식 + 재위치추정

**강건한 초기화 (Robust Initialization)**

단안 VIO는 초기화가 까다롭다. 시작할 때는 메트릭 스케일, 중력 방향, IMU 바이어스를 모두 모른다. VINS-Mono는 loosely-coupled 방식으로 이 값들을 먼저 구한다.

**Step 1: 순수 비전 SfM**

처음 몇 프레임은 영상만으로 Structure from Motion을 푼다. 5-point 알고리즘으로 Essential Matrix를 추정하고, 삼각측량으로 3D 점과 카메라 포즈를 복원한다. 이때 스케일은 임의로 정해진다.

**Step 2: Visual-Inertial Alignment**

SfM 결과와 IMU preintegration을 정렬(alignment)하면서 다음 값을 추정한다.

(a) 자이로 바이어스 $\mathbf{b}_g$: 연속 키프레임 쌍의 SfM 회전 $\mathbf{R}_{c_k c_{k+1}}^{\text{sfm}}$과 IMU preintegrated 회전 $\Delta\hat{\mathbf{R}}_{k,k+1}$을 비교:

$$\min_{\mathbf{b}_g} \sum_k \left\| \text{Log}\left(\Delta\hat{\mathbf{R}}_{k,k+1}(\mathbf{b}_g)^T \cdot \mathbf{R}_{c_k}^{c_{k+1}} \right) \right\|^2$$

바이어스 변화를 1차로 근사하면(§4.6.2의 preintegration 바이어스 자코비안) 이 문제는 선형 최소자승이 된다.

(b) 중력 방향, 속도, 메트릭 스케일을 함께 추정한다. 키프레임 쌍 $(k, k+1)$마다 다음 선형 식이 선다.

$$s\mathbf{R}_{c_0}^w \mathbf{p}_{c_{k+1}}^{c_0} - s\mathbf{R}_{c_0}^w \mathbf{p}_{c_k}^{c_0} - \mathbf{v}_k^w \Delta t_k - \frac{1}{2}\mathbf{g}^w\Delta t_k^2 = \mathbf{R}_k^w \Delta\hat{\mathbf{p}}_{k,k+1}$$

여기서 미지수는 $s$ (스케일), $\mathbf{g}^w$ (중력 벡터), $\{\mathbf{v}_k^w\}$ (속도)다. 중력 부호는 이 가이드의 규약($\mathbf{g}=[0,0,-9.81]^\top$, Ch.4)을 따른다. VINS-Mono 원 논문은 $\mathbf{g}^w$의 부호를 반대로 두므로 원문 식과 부호가 다르다. VINS-Mono는 여기에 중력 크기 $\|\mathbf{g}\| = 9.81$이라는 제약을 더해 중력 방향을 다시 다듬는다(gravity refinement).

이 loosely-coupled 초기화가 수렴하면 tightly-coupled 최적화로 전환한다.

**Tightly-Coupled Nonlinear Optimization**

VINS-Mono의 백엔드는 슬라이딩 윈도우 안에서 다음 잔차를 함께 최적화한다:

$$\min_{\mathcal{X}} \left\{ \left\|\mathbf{r}_p - \mathbf{H}_p \mathcal{X}\right\|^2 + \sum_{k \in \mathcal{B}} \left\|\mathbf{r}_{\mathcal{B}}(\hat{\mathbf{z}}_{b_k b_{k+1}}, \mathcal{X})\right\|^2_{\mathbf{P}_{b_k b_{k+1}}^{-1}} + \sum_{(l,j) \in \mathcal{C}} \left\|\mathbf{r}_{\mathcal{C}}(\hat{\mathbf{z}}_{l}^{c_j}, \mathcal{X})\right\|^2_{(\mathbf{P}_{l}^{c_j})^{-1}} \right\}$$

여기서:
- $\mathbf{r}_p$: 마지널라이제이션 사전(prior) 잔차
- $\mathbf{r}_{\mathcal{B}}$: IMU preintegration 잔차
- $\mathbf{r}_{\mathcal{C}}$: 시각 reprojection error 잔차
- $\mathcal{X}$: 상태 변수 (슬라이딩 윈도우 내 키프레임 포즈, 속도, IMU 바이어스, 특징점 역깊이)

**시각 잔차의 상세 정의**:

특징점 $l$을 키프레임 $i$에서 처음 관측하고 역깊이 $\lambda_l$로 나타냈을 때, 키프레임 $j$의 reprojection error는:

$$\mathbf{r}_{\mathcal{C}} = \begin{bmatrix} \bar{u}_l^{c_j} - u_l^{c_j} / z_l^{c_j} \\ \bar{v}_l^{c_j} - v_l^{c_j} / z_l^{c_j} \end{bmatrix}$$

여기서 $\begin{bmatrix} u_l^{c_j} & v_l^{c_j} & z_l^{c_j} \end{bmatrix}^T = \mathbf{R}_{b_j}^{c} (\mathbf{R}_{w}^{b_j}(\mathbf{R}_{b_i}^{w}(\mathbf{R}_{c}^{b_i} \frac{1}{\lambda_l}\begin{bmatrix} \bar{u}_l^{c_i} \\ \bar{v}_l^{c_i} \\ 1 \end{bmatrix} + \mathbf{p}_c^b) + \mathbf{p}_{b_i}^w - \mathbf{p}_{b_j}^w) - \mathbf{p}_c^b)$

**슬라이딩 윈도우 관리와 마지널라이제이션**

VINS-Mono의 원 논문과 공개 구성은 frame state를 10개 안팎으로 두는 고정 크기 윈도우를 쓰고, 뒤에서 두 번째 프레임(second latest frame)이 키프레임인지에 따라 마지널라이즈 방식을 바꾼다.

1. **뒤에서 두 번째 프레임이 키프레임인 경우**: 그 프레임은 윈도우에 남기고, 가장 오래된 프레임을 마지널라이즈한다. Schur complement로 해당 프레임에 연결된 모든 측정을 prior로 바꾼다.

2. **뒤에서 두 번째 프레임이 키프레임이 아닌 경우**: 그 프레임을 윈도우에서 뺀다. 이때 시각 측정만 버리고, IMU preintegration은 다음 프레임까지 이어서 계산하므로 IMU 정보는 남는다.

Schur complement 계산은 §4.7.2에 있다.

**4-DoF 포즈 그래프 최적화**

Loop closure가 검출되면 VINS-Mono는 4-DoF(yaw + 3D translation) 포즈 그래프를 최적화한다. 운동이 충분하고 중력·바이어스 모델이 맞으면 VIO는 roll과 pitch를 중력 방향에 맞출 수 있지만, 전역 yaw와 위치에는 gauge freedom이 남는다. 이 설계는 그 가정에 맞춰 roll/pitch를 local VIO에 맡기고 yaw·위치를 전역 보정한다.

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
            # 직전 프레임(비키프레임)을 제거: 시각 측정은 버리고 IMU는 preintegration으로 이어 붙인다
            drop_visual_factors(self.states[-2])
            merge_imu_preintegration(self.states[-3], self.states[-2], self.states[-1])
            self.states.pop(-2)
```

### 6.3.2 OKVIS

[OKVIS (Open Keyframe-based Visual-Inertial SLAM, Leutenegger et al., 2015)](https://doi.org/10.1177/0278364914554813)는 VINS-Mono보다 먼저 나온 tightly-coupled VIO로, 키프레임 기반 슬라이딩 윈도우 최적화를 일찍 보여 준 시스템이다.

**OKVIS의 설계**:

1. **Harris corner + BRISK descriptor**: ORB 대신 Harris 코너와 BRISK 디스크립터를 쓴다.
2. **Keyframe-based marginalization**: VINS-Mono처럼 슬라이딩 윈도우에서 오래된 상태를 마지널라이즈한다.
3. **IMU error term**: 두 프레임 사이의 IMU 운동학을 재전파(repropagation)해 자세·속도·위치·바이어스 전체의 15차원 잔차를 만든다. "speed and bias"는 상태 블록 $[\mathbf{v}, \mathbf{b}_g, \mathbf{b}_a]$의 이름이다. 이후 OKVIS2에서 preintegration으로 전환했다.
4. **Ceres Solver 기반**: 최적화에 Ceres Solver를 쓴다.

OKVIS는 스테레오 카메라를 기본으로 가정한다.

### 6.3.3 MSCKF: Multi-State Constraint Kalman Filter

[MSCKF (Mourikis & Roumeliotis, 2007)](https://doi.org/10.1109/ROBOT.2007.364024)는 필터 기반 VIO의 대표 알고리즘이다. 구조는 최적화 기반 시스템(VINS-Mono, ORB-SLAM3)과 다르다. OpenVINS(§6.3.4) 같은 구현이 지금도 쓰인다.

**MSCKF는 랜드마크를 상태에 넣지 않는다**

EKF-SLAM은 랜드마크(3D 점)를 상태 벡터에 넣는다. 랜드마크가 $L$개면 상태 벡터 크기가 $3L + 15$, 공분산 행렬 크기가 $(3L+15)^2$이므로 메모리도 갱신 한 번의 연산도 $O(L^2)$이다. 혁신 공분산은 측정 차원의 작은 행렬이라 큰 행렬을 분해할 일이 없어 비용은 제곱에서 멈춘다. 그래도 랜드마크가 늘면 실시간으로 돌리기 어렵다.

MSCKF는 랜드마크를 상태 벡터에서 빼면서도 랜드마크가 주는 기하 제약은 살린다.

**상태 벡터 구조**

MSCKF의 상태 벡터에는 IMU 오차 상태(error-state)와 슬라이딩 윈도우 안의 카메라 포즈 $N$개가 들어간다.

$$\tilde{\mathbf{x}} = [\tilde{\mathbf{x}}_{IMU}^T, \tilde{\mathbf{x}}_{C_1}^T, \ldots, \tilde{\mathbf{x}}_{C_N}^T]^T$$

여기서 IMU 오차 상태는:

$$\tilde{\mathbf{x}}_{IMU} = [\delta\boldsymbol{\theta}^T, \tilde{\mathbf{b}}_g^T, {}^G\tilde{\mathbf{v}}_I^T, \tilde{\mathbf{b}}_a^T, {}^G\tilde{\mathbf{p}}_I^T]^T \in \mathbb{R}^{15}$$

각 카메라 포즈 오차 상태는:

$$\tilde{\mathbf{x}}_{C_k} = [\delta\boldsymbol{\theta}_{C_k}^T, {}^G\tilde{\mathbf{p}}_{C_k}^T]^T \in \mathbb{R}^6$$

상태 벡터의 전체 크기는 $15 + 6N$이다. 랜드마크 수와 무관하다.

**Null-Space 투영**

정적 특징점 $\mathbf{p}_f$ 하나를 카메라 포즈 $M$개에서 관측했다고 하자. 관측 방정식을 선형화하면:

$$\mathbf{r} = \mathbf{H}_X \tilde{\mathbf{x}} + \mathbf{H}_f \tilde{\mathbf{p}}_f + \mathbf{n}$$

여기서:
- $\mathbf{r} \in \mathbb{R}^{2M}$: 잔차 벡터 (관측 - 예측)
- $\mathbf{H}_X \in \mathbb{R}^{2M \times (15+6N)}$: 상태 자코비안
- $\mathbf{H}_f \in \mathbb{R}^{2M \times 3}$: 특징점 위치 자코비안
- $\tilde{\mathbf{p}}_f \in \mathbb{R}^3$: 특징점 위치 오차

$\mathbf{H}_f$를 QR 분해한다:

$$\mathbf{H}_f = \begin{bmatrix} \mathbf{Q}_1 & \mathbf{Q}_2 \end{bmatrix} \begin{bmatrix} \mathbf{R}_1 \\ \mathbf{0} \end{bmatrix}$$

$\mathbf{Q}_2$는 $\mathbf{H}_f$의 left null space를 이룬다 ($\mathbf{Q}_2^T \mathbf{H}_f = \mathbf{0}$). 양변에 $\mathbf{Q}_2^T$를 곱하면:

$$\mathbf{r}_o = \mathbf{Q}_2^T \mathbf{r} = \mathbf{Q}_2^T \mathbf{H}_X \tilde{\mathbf{x}} + \mathbf{Q}_2^T \mathbf{n} = \mathbf{H}_o \tilde{\mathbf{x}} + \mathbf{n}_o$$

특징점 위치 $\tilde{\mathbf{p}}_f$가 식에서 빠졌다. $\mathbf{r}_o$와 $\mathbf{H}_o$만으로 EKF 업데이트를 한다. MSCKF의 "multi-state constraint"는 특징점 하나가 여러 카메라 포즈 사이에 거는 기하 제약을 쓰되 특징점 자체는 상태에 넣지 않는다는 뜻이다.

**계산 복잡도**: 보통 $N \ll L$이므로 EKF-SLAM의 $O(L^2)$ 갱신보다 훨씬 가볍다.

**MSCKF 업데이트 절차**

1. **IMU propagation**: 새 IMU 측정이 들어오면 상태를 전파하고 공분산을 업데이트한다:
   $$\hat{\mathbf{x}}_{k+1|k} = f(\hat{\mathbf{x}}_{k|k}, \mathbf{u}_k)$$
   $$\mathbf{P}_{k+1|k} = \boldsymbol{\Phi}_k \mathbf{P}_{k|k} \boldsymbol{\Phi}_k^T + \mathbf{G}_k \mathbf{Q} \mathbf{G}_k^T$$

2. **State augmentation**: 새 이미지가 들어오면 현재 IMU 포즈로 카메라 포즈를 만들어 상태에 붙이고, 공분산 행렬도 그만큼 늘린다.

3. **MSCKF update**: 추적이 끝난 특징점(더 이상 관측되지 않는 점)에 대해:
   - 삼각측량으로 $\hat{\mathbf{p}}_f$를 추정한다.
   - null-space 투영으로 $\mathbf{r}_o, \mathbf{H}_o$를 계산한다.
   - 표준 EKF 업데이트를 적용한다:
     $$\mathbf{K} = \mathbf{P} \mathbf{H}_o^T (\mathbf{H}_o \mathbf{P} \mathbf{H}_o^T + \sigma^2 \mathbf{I})^{-1}$$
     $$\hat{\mathbf{x}} \leftarrow \hat{\mathbf{x}} + \mathbf{K} \mathbf{r}_o$$
     $$\mathbf{P} \leftarrow (\mathbf{I} - \mathbf{K}\mathbf{H}_o)\mathbf{P}$$

4. **State pruning**: 슬라이딩 윈도우에서 오래된 카메라 포즈를 빼고 공분산 행렬도 그만큼 줄인다.

```cpp
// MSCKF 핵심 업데이트 수도코드 (C++)
void MSCKF::msckf_update(const Feature& feature) {
    // 1. 삼각측량으로 특징점 3D 위치 추정
    Vector3d p_f = triangulate(feature.observations, cam_states);

    // 2. 관측 자코비안 계산
    int N_obs = feature.observations.size();
    MatrixXd H_X(2*N_obs, state_dim);  // 상태 자코비안
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

[OpenVINS (Geneva et al., 2020)](https://doi.org/10.1109/ICRA40945.2020.9196524)는 MSCKF 기반 VIO를 오픈소스로 구현한 연구 플랫폼이다. 모듈을 바꿔 끼우며 여러 VIO 변형을 비교하고 실험하기 좋게 만들었다.

**OpenVINS의 구성**:

1. **On-Manifold Sliding Window EKF**: MSCKF를 기반으로 한 sliding window 칼만 필터. SO(3) 매니폴드 위에서 회전을 처리한다.

2. **온라인 캘리브레이션**: OpenVINS는 설정에 따라 카메라 intrinsic, camera-IMU extrinsic, 시간 오프셋을 상태에 넣을 수 있다. 이 값들은 충분한 운동과 장면 기하가 있어야 안정적으로 수렴하므로, 하드웨어 동기화와 오프라인 보정을 마친 시스템에서는 고정해 두기도 한다.

3. **SLAM 랜드마크 지원**: 순수 MSCKF는 특징점을 상태에 넣지 않지만, OpenVINS는 설정하면 일부 랜드마크를 SLAM feature로 상태에 넣는다. SLAM feature는 오래 추적되는 점이고, 여러 표현 가운데 anchored inverse depth도 고를 수 있다.

4. **First-Estimates Jacobian (FEJ)**: 자코비안의 일부를 first estimate에서 계산해, 선형화한 시스템에서 관측할 수 없는 방향이 잘못 줄어드는 현상을 덜어 준다. FEJ는 중요한 일관성 장치지만 모델 오차·noise tuning·모든 선형화 문제를 해결하지는 않는다.

5. **시뮬레이터**: VIO 알고리즘을 시험할 시뮬레이터가 들어 있다. 궤적, 환경, IMU 노이즈 모델을 바꿔 가며 ground truth와 비교해 정량 평가를 한다.

### 6.3.5 Basalt

[Basalt (Usenko et al., 2020)](https://doi.org/10.1109/LRA.2019.2961227)도 VINS-Mono처럼 tightly-coupled VIO지만 몇 가지 설계가 다르다.

**Basalt의 구성**:

1. **Visual-only Frontend**: 패치 단위 KLT 추적(inverse-compositional, SE(2) warp)으로 서브픽셀 정밀도의 특징점을 추적한다. VINS-Mono의 점 단위 KLT와 계열은 같고 패치·warp 설계가 다르다.

2. **Non-linear Factor Recovery (NFR)**: 마지널라이제이션이 남기는 prior는 그때의 선형화 지점에 묶여 있어서, 뒤에 그 지점이 크게 바뀌면 정보가 틀어진다. Basalt의 NFR은 VIO가 궤적에 대해 쌓은 정보를 가장 잘 근사하는 비선형 factor 집합을 복원한다. 이 factor는 비선형이므로 나중에 다시 선형화할 수 있다. Basalt는 이 factor를 loop closure 제약과 함께 bundle adjustment에 넣어 전역 맵을 만들고, 이 factor 덕분에 전역 맵의 roll과 pitch가 관측 가능해진다([Usenko et al., 2020](https://arxiv.org/abs/1904.06504)).

3. **Efficient Implementation**: Basalt는 factor graph의 희소 구조를 살려 구현을 빠르게 했다. 처리 속도를 다른 시스템과 비교하려면 센서 구성, 해상도, 하드웨어를 맞춰야 한다. 특히 Basalt는 stereo+IMU 시스템이라 mono+IMU 시스템과 바로 견주기 어렵다.

4. **Stereo/Multi-camera 지원**: 여러 카메라의 영상을 한 최적화에 넣는다.

---

## 6.4 VIO 설계 선택지

VIO 시스템의 설계 선택은 계산량, 정확도, 적용 조건을 바꾼다.

### 6.4.1 Filter vs Optimization

Filter와 optimization 가운데 무엇을 쓸지는 VIO에서 오래 다퉈 온 문제다.

**Filter 기반 (MSCKF, OpenVINS)**

- 현재 유지하는 상태(일부 과거 카메라 포즈 포함)를 새 측정이 올 때마다 순차적으로 업데이트
- 제거한 과거 상태의 정보는 유지하는 상태의 분포(mean + covariance)에 "흡수"됨
- 계산 복잡도: 업데이트당 $O(N^2)$ (N은 상태 차원)
- 장점: 일정한 계산 비용, 구현 간단
- 단점: 선형화 오차 누적 (한 번 선형화하면 교정 불가), 일관성(consistency) 문제

**Optimization 기반 (VINS-Mono, ORB-SLAM3, Basalt)**

- 슬라이딩 윈도우 안의 여러 상태를 동시에 유지하고, 모든 측정을 함께 최적화
- 반복 재선형화(iterative re-linearization)가 가능
- 계산 복잡도: 반복당 $O(N^3)$ (N은 윈도우 크기), Schur complement로 줄임
- 장점: 재선형화로 선형화 오차 감소, 여러 종류의 측정을 넣기 쉬움
- 단점: 계산 비용 높음, 윈도우 크기에 민감

**실험적 비교**: 어느 쪽이 더 정확한지는 데이터셋, 센서, 계산 예산을 맞춘 실험으로 가린다(§4.4.4). 필터도 FEJ나 관측 가능성 제약을 적용하면 일관성 문제를 크게 덜 수 있다. 계산 자원이 아주 적은 플랫폼에서는 필터의 일정한 계산 비용이 쓸모 있다.

### 6.4.2 Keyframe Selection 전략

키프레임을 너무 자주 넣으면 계산이 무거워지고, 너무 드물게 넣으면 정보를 잃는다. 키프레임 선택 기준은 보통 다음과 같다.

1. **시간 기반**: 마지막 키프레임 이후 일정 시간 경과
2. **시차(parallax) 기반**: 현재 프레임과 마지막 키프레임 사이의 평균 특징점 시차가 임계값 초과. VINS-Mono는 이 기준을 쓴다.
3. **추적 품질 기반**: 추적된 특징점 수가 임계값 이하로 떨어지면 키프레임 삽입. ORB-SLAM3는 이 기준을 쓴다.
4. **Information gain 기반**: 새 키프레임이 더할 정보량(정보 이득)을 추정해 정한다. 정보량을 직접 보지만 계산 비용이 크다.

키프레임 선택은 마지널라이제이션과 맞물린다. VINS-Mono의 two-way marginalization 전략(§6.3.1 참조)은 키프레임 여부에 따라 마지널라이제이션 방향을 바꾼다.

### 6.4.3 Feature Parameterization

3D 점을 어떤 변수로 나타낼지도 정해야 한다.

**XYZ (Euclidean 3D 좌표)**

$\mathbf{P} = [X, Y, Z]^T \in \mathbb{R}^3$로 바로 쓴다. 먼 점(far point)에서는 불안정하다. 시차가 조금만 바뀌어도 $Z$ 값이 크게 바뀐다.

**Inverse Depth**

점을 "호스트 프레임의 관측 방향 + 역깊이"로 표현한다:

$$\boldsymbol{\lambda} = [\theta, \phi, \rho]^T$$

여기서 $\theta, \phi$는 호스트 프레임 기준 방위각/고도각, $\rho = 1/d$는 역깊이다. 장점:

1. **원거리 점 처리**: $d \to \infty$일 때 $\rho \to 0$으로 수치적으로 안정적이다. XYZ에서는 $Z \to \infty$가 되어 불안정하다.
2. **선형성 개선**: 단안 초기화에서 깊이가 불확실할 때, 역깊이의 불확실성 분포가 가우시안에 더 가깝다.

VINS-Mono와 OpenVINS는 역깊이 파라미터화를 쓴다.

**Anchored Inverse Depth**

역깊이를 특정 "앵커" 키프레임에 고정(anchored)해 나타낸다:

$$\mathbf{P} = \mathbf{T}_{\text{anchor}} \cdot \frac{1}{\rho} [\bar{u}, \bar{v}, 1]^T$$

여기서 $(\bar{u}, \bar{v})$는 앵커 프레임 기준 정규화 좌표, $\rho$는 역깊이다. 앵커 프레임의 포즈가 바뀌어도 역깊이 값이 유지되므로 선형화 오차가 일부 줄어든다. OpenVINS는 SLAM feature를 이 방식으로 표현한다. ORB-SLAM3는 맵포인트를 세계 좌표계의 3D 위치로 저장·최적화한다.

---

## 6.5 학습 기반 VO/VIO

전통적 VO/VIO는 사람이 설계한 파이프라인으로 돈다. Feature-based 방식이라면 특징점 검출 → 매칭 → RANSAC → BA다. 학습 기반 방법은 이 파이프라인의 일부나 전체를 신경망으로 바꾼다.

### 6.5.1 Supervised: DeepVO 계열

초기 학습 기반 VO ([DeepVO, Wang et al., 2017](https://doi.org/10.1109/ICRA.2017.7989236))는 연속 이미지 쌍을 입력으로 받아 상대 포즈를 직접 예측하는 end-to-end 네트워크를 훈련했다. CNN이 영상 특징을 뽑고, LSTM이 시간 순서에 걸친 관계를 모델링한다.

남은 한계:
- 학습 데이터의 환경에 과적합 (generalization 부족)
- 기하학적 제약(에피폴라 기하 등)을 쓰지 않아 정확도가 전통 방법에 못 미침
- 스케일 드리프트가 심함

### 6.5.2 Self-supervised: 한계와 현재 위치

자기 지도 학습 VO (SfMLearner, Monodepth2 등)는 view synthesis로 만든 photometric loss로 깊이와 포즈를 동시에 학습한다. 레이블 없이 학습하지만, 움직이는 물체(moving objects), 텍스처 부족, 가림(occlusion)이 있으면 잘 안 된다.

자기 지도 학습은 단안 깊이 추정(monocular depth estimation)에 쓰인다.

### 6.5.3 Hybrid: DROID-SLAM

[DROID-SLAM (Teed & Deng, 2021)](https://arxiv.org/abs/2108.10869)은 BA의 기하 구조와 학습된 correspondence update를 미분 가능한 파이프라인 하나로 묶은 시스템이다. 원 논문은 TartanAir, EuRoC, TUM-RGBD, ETH-3D에서 기존 학습·기하 기반 방법보다 정확도가 크게 높고 catastrophic failure가 훨씬 적었다고 보고했다.

**아키텍처**

DROID-SLAM의 구성 요소:

1. **RAFT 기반 반복 업데이트 연산자 (Iterative Update Operator)**

RAFT (Recurrent All-Pairs Field Transforms, Teed & Deng, 2020)의 구조를 바탕으로 한다. 프레임 쌍 $(i, j)$에 대해:
- 두 프레임의 feature map에서 4D correlation volume을 계산한다.
- 현재 포즈·깊이 추정에서 구한 correspondence field가 가리키는 위치의 특징을 correlation volume에서 읽는다.
- 3×3 Convolutional GRU가 correlation 특징, 현재 flow, context 특징을 입력받아 flow revision $\Delta\mathbf{f}_{ij}$와 confidence weight $\mathbf{w}_{ij}$를 출력한다.

$$(\Delta\mathbf{f}_{ij}, \mathbf{w}_{ij}) = \text{ConvGRU}(\text{corr}_{ij}, \mathbf{f}_{ij}^{\text{curr}}, \text{context}_i)$$

이 업데이트를 여러 번(iteratively) 되풀이하며 correspondence를 조금씩 다듬는다.

2. **미분 가능 Dense Bundle Adjustment (DBA) 레이어**

GRU가 출력한 flow revision을 기하 업데이트로 바꾸는 레이어다. 모든 픽셀에 reprojection error를 정의하고, 이를 카메라 포즈 $\mathbf{T}_i \in SE(3)$와 역깊이 $d_i$에 대해 Gauss-Newton으로 최소화한다:

$$\sum_{(i,j)} \sum_{\mathbf{p}} \left\| \mathbf{w}_{ij}^{\mathbf{p}} \circ (\mathbf{p}^{*}_{ij} - \pi(\mathbf{T}_j \circ \mathbf{T}_i^{-1} \circ \pi^{-1}(\mathbf{p}, d_i^{\mathbf{p}}))) \right\|^2$$

여기서 $\mathbf{p}^{*}_{ij} = \mathbf{p} + \mathbf{f}_{ij}^{\text{curr}} + \Delta\mathbf{f}_{ij}$는 target correspondence, $\mathbf{w}_{ij}^{\mathbf{p}}$는 confidence weight이다.

Gauss-Newton 업데이트를 유도하면 정규 방정식이 나온다:

$$\begin{bmatrix} \mathbf{B} & \mathbf{E} \\ \mathbf{E}^T & \mathbf{C} \end{bmatrix} \begin{bmatrix} \boldsymbol{\xi} \\ \delta\mathbf{d} \end{bmatrix} = \begin{bmatrix} \mathbf{v} \\ \mathbf{w} \end{bmatrix}$$

여기서 $\boldsymbol{\xi}$는 포즈 업데이트(SE(3) tangent space), $\delta\mathbf{d}$는 역깊이 업데이트, $\mathbf{B}$는 포즈 블록, $\mathbf{C}$는 깊이 블록(대각), $\mathbf{E}$는 교차 블록이다.

전통 BA처럼 Schur complement를 쓴다:

$$(\mathbf{B} - \mathbf{E}\mathbf{C}^{-1}\mathbf{E}^T)\boldsymbol{\xi} = \mathbf{v} - \mathbf{E}\mathbf{C}^{-1}\mathbf{w}$$

$\mathbf{C}$는 대각이므로 $\mathbf{C}^{-1}$은 대각 원소의 역수로 바로 구한다. 줄인 시스템의 크기는 카메라 수로만 정해지므로 계산이 가볍다.

Gauss-Newton 솔버 전체가 미분 가능하므로, 역전파로 GRU 파라미터를 "좋은 correspondence"를 내도록 학습한다.

**프레임 그래프와 loop closure**

DROID-SLAM은 co-visibility 기반 프레임 그래프를 실행 중에 만들어 간다. 새 프레임이 들어오면 이웃 프레임과 엣지로 잇고, 과거 프레임과의 co-visibility가 확인되면 장거리 엣지를 더해 루프를 닫는다. 백엔드는 전체 키프레임 히스토리로 글로벌 BA를 푼다.

**다중 모달리티 지원**

DROID-SLAM은 단안 영상으로만 학습했는데도 추론할 때 스테레오와 RGB-D 입력을 그대로 쓴다.
- **스테레오**: 왼쪽/오른쪽 영상을 각각 프레임으로 다루되, 두 프레임 사이의 상대 포즈는 DBA 레이어에서 고정한다.
- **RGB-D**: 센서 깊이에는 노이즈와 빈 곳이 있어 깊이는 그대로 변수로 두고, 측정 깊이와 예측 깊이 사이 거리의 제곱을 벌점으로 주는 항을 목적 함수(논문 식 4)에 더한다.

**성능**

다음 수치는 원 논문이 보고한 결과다:

- TartanAir: 단안 트랙에서 이전 최고 결과보다 오차 62% 감소(스테레오 트랙 60%)
- EuRoC monocular: 추적 실패가 없는 방법들과 비교해 오차 82% 감소(ORB-SLAM3와는 ORB-SLAM3가 성공한 10개 시퀀스 기준 43% 감소)
- ETH-3D: RGB-D 시퀀스 32개 중 30개 추적 성공 (그다음으로 좋은 방법은 19개)

위 결과는 모두 합성 데이터셋 TartanAir의 단안 영상으로 한 번 학습한 모델 하나로 얻었다.

**한계**

DROID-SLAM의 한계는 다음과 같다:
- **연산량**: 공개 구현은 GPU를 전제로 한다. 프레임 속도는 해상도·키프레임 수·하드웨어와 설정에 따라 달라진다.
- **IMU 미포함**: 원 시스템은 영상만 쓴다.
- **메모리**: 모든 프레임의 feature map을 들고 있어야 해서 메모리를 많이 쓴다.

후속 연구로 sparse patch(DPVO), 메모리 축소, 관성 센서 결합이 나왔다.

### 6.5.4 최근 동향 (2023-2025)

[DPVO (Teed, Lipson & Deng, 2023)](https://arxiv.org/abs/2208.04726)는 DROID-SLAM의 dense flow를 sparse patch 기반 매칭으로 바꿨다. 원 논문은 자체 비교 설정에서 메모리 약 1/3, 속도 약 3배와 더 낮은 평균 오차를 보고한다. 패치 단위 recurrent update operator와 미분 가능 BA를 결합한 구조다.

[MAC-VO (Qiu et al., ICRA 2025)](https://arxiv.org/abs/2409.09479)는 학습 기반 매칭 불확실성(metrics-aware covariance)을 스테레오 VO에 넣고, 키포인트 선택과 포즈 그래프 최적화의 잔차 가중치를 이 불확실성으로 정한다. 원 논문은 조명 변화와 텍스처 부족을 포함한 공개 benchmark에서 비교 방법보다 낮은 오차를 보고했고, 이 논문은 ICRA 2025 Best Paper Award on Robot Perception을 받았다.

---

## 6장 요약

| 시스템 | 유형 | 추정 방법 | 센서 | 특징 |
|--------|------|-----------|------|-----------|
| ORB-SLAM3 | Feature-based | Optimization (BA) | Mono/Stereo/RGBD + IMU | Multi-map Atlas, fisheye 지원 |
| DSO | Direct | Optimization (windowed) | Mono | Photometric calibration, sparse sampling |
| SVO | Semi-direct | Optimization (BA) | Mono/Stereo | Semi-direct front end, sparse depth, 높은 처리율을 겨냥한 설계 |
| VINS-Mono | Feature-based | Optimization (sliding window) | Mono + IMU | 강건한 초기화, 4-DoF loop closure |
| MSCKF | Feature-based | EKF (sliding window) | Mono/Stereo + IMU | 랜드마크를 상태에서 제외, null-space 투영 |
| OpenVINS | Feature-based | EKF (MSCKF) | Mono/Stereo + IMU | 온라인 캘리브레이션, FEJ, 연구 플랫폼 |
| Basalt | Feature-based (KLT) | Optimization (sliding window) | Stereo + IMU | NFR, 효율적 구현 |
| DROID-SLAM | Learned | Differentiable BA | Mono/Stereo/RGBD | 미분 가능 BA, 합성 데이터 학습 |
| DPVO | Learned (sparse) | Differentiable BA | Mono | Sparse patch; 논문 설정에서 DROID-SLAM보다 낮은 연산·메모리 비용 보고 |
| MAC-VO | Learned + Opt. | Pose graph opt. | Stereo | Metrics-aware covariance, ICRA 2025 Best Paper on Robot Perception |
