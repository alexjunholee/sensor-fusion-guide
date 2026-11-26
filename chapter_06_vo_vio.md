# 6장 VO와 VIO

상태 추정 프레임워크(→ 4장)가 모션 모델과 관측 모델을 확률적으로 결합하는 수학적 뼈대이고, 특징점 매칭과 대응점 탐색(→ 5장)이 서로 다른 시점의 관측을 연결하는 감각 기관이라면, 이제 두 조각을 엮어 센서의 6자유도 궤적을 실시간으로 복원하는 실제 오도메트리 시스템을 조립할 차례다. Visual Odometry(VO)는 연속된 카메라 영상만으로 카메라의 자기 운동(ego-motion)을 추정하며, 여기에 IMU 관성 측정을 결합한 Visual-Inertial Odometry(VIO)는 단안 카메라가 놓치는 절대 스케일을 복원하고 빠른 회전이나 흐릿한 영상 구간에서도 추정을 지탱한다.

VO라는 이름은 [Nistér et al. (2004)](https://doi.org/10.1109/CVPR.2004.1315094)의 논문에서 처음 등장했다. 바퀴 회전수로 주행 거리를 재는 휠 오도메트리처럼 카메라 영상으로 이동 궤적을 잰다는 뜻에서 붙은 명칭이다. 이 연구에서는 스테레오와 단안 카메라 각각의 실시간 자기 운동 추정 파이프라인을 정립했다. 스테레오 구성에서는 좌우 카메라로 삼각측량한 3D 점에 3-point 알고리즘을 적용해 프레임 간 강체 변환을 구했고, 단안 구성에서는 5-point 알고리즘으로 Essential Matrix를 추정했다. 특징점 검출에서 매칭, RANSAC, 모션 추정으로 이어지는 이 골격은 20년이 지난 오늘날에도 feature-based VO의 기본 구조로 남아 있다.

오늘날의 VO·VIO 시스템은 크게 세 가지 설계 축으로 갈라진다. 어떤 관측을 쓰느냐에 따라 기하학적 특징점(corner, edge)을 뽑아 매칭하는 **Feature-based** 방식과 픽셀 밝기를 그대로 활용하는 **Direct** 방식으로 나뉘고, 상태를 어떻게 풀느냐에 따라 칼만 필터 계열의 **Filter** 방식과 비선형 **Optimization** 방식으로 구분된다. 여기에 IMU를 쓰는 VIO에서는 카메라와 IMU를 독립적으로 처리한 뒤 결과만 합치는 **Loosely coupled** 구조와 원시 측정값(raw measurement)을 한 추정 프레임워크 안에 직접 엮는 **Tightly coupled** 구조라는 선택지가 더해진다.

---

## 6.1 Feature-based Visual Odometry

세 가지 축 가운데 가장 오랫동안 검증된 조합은 영상에서 기하학적 특징점을 뽑아 프레임 간 대응을 맞추는 feature-based VO다. 특징점 추출과 모션 추정이 깔끔하게 분리되어 있어 각 단계의 동작 원리를 파악하고 실패 양상(failure mode)을 진단하기 쉽고, ORB-SLAM3 같은 대표적인 공개 시스템도 이 구조를 바탕으로 삼는다.

### 6.1.1 Frontend: Detection, Tracking, Outlier Rejection

초당 수십 장씩 들어오는 영상 프레임에서 유의미한 기하 정보를 건져 올리는 일이 프런트엔드의 몫이다. 프런트엔드는 각 프레임에서 추적할 점을 고르는 **검출(detection)**, 다음 프레임에서 그 점의 새 위치를 찾아 잇는 **추적(tracking)**, 잘못 붙은 대응을 걸러 내는 **아웃라이어 제거(outlier rejection)**라는 세 단계를 거친다.

**특징점 검출 (Feature Detection)**

프레임 간 모션을 정밀하게 추정하려면, 시점이 조금 바뀌어도 같은 3D 물리적 지점이 다음 영상에서 어김없이 다시 뽑혀야 한다. 이 성질이 바로 반복성(repeatability)이다.

평탄한 벽면 위에서는 작은 패치를 어느 방향으로 옮겨도 밝기가 변하지 않고, 직선 경계(edge) 위에서는 경계 방향을 따라 움직일 때 밝기 변화가 없다. 오직 두 경계가 만나는 **코너(corner)**에서만 패치를 어느 방향으로 조금만 움직여도 밝기가 뚜렷하게 달라진다. Harris corner detector는 이 직관을 이미지 패치의 자기상관 행렬(autocorrelation matrix, 또는 second moment matrix) $\mathbf{M}$으로 수식화한다.

$$\mathbf{M} = \sum_{(x,y) \in W} w(x,y) \begin{bmatrix} I_x^2 & I_xI_y \\ I_xI_y & I_y^2 \end{bmatrix}$$

여기서 $I_x, I_y$는 이미지 그래디언트, $W$는 패치 윈도우, $w$는 가중치 함수다. $\mathbf{M}$의 두 고유값 $\lambda_1, \lambda_2$는 서로 직교하는 두 주방향으로의 밝기 변화량을 뜻하므로, 두 고유값이 모두 크면 어느 방향으로 움직여도 밝기가 변하는 코너로 판정할 수 있다. 매번 고유값을 직접 구하는 대신 계산이 가벼운 Harris 응답 함수를 쓴다.

$$R = \det(\mathbf{M}) - k \cdot \text{tr}(\mathbf{M})^2 = \lambda_1\lambda_2 - k(\lambda_1 + \lambda_2)^2$$

하지만 실시간 SLAM처럼 매 프레임 수천 개의 특징점을 밀리초 단위로 뽑아야 하는 환경에서는 그래디언트 행렬을 계산하는 비용마저 부담스럽다. FAST (Features from Accelerated Segment Test)는 후보 픽셀 주위 원 위의 16개 점과 중심 픽셀의 밝기만 단순 비교해 연산 속도를 극단적으로 끌어올렸다(원 논문의 $n$ 값과 사전 검사는 §5.2.1). OpenCV의 기본값은 $n=9$(`TYPE_9_16`)이다. 순수 FAST에서 빠진 회전·스케일 불변성은 ORB 같은 확장 기법에서 보완한다.

ORB (Oriented FAST and Rotated BRIEF)는 실시간 SLAM의 요구에 맞춰 FAST 코너에 방향을 부여하고, 이진 디스크립터인 BRIEF를 그 방향에 맞춰 회전시킨 특징점이다. 패치의 밝기 중심(intensity centroid)으로 주방향을 계산하므로 회전이 섞인 카메라 움직임에도 디스크립터가 흔들리지 않는다.

$$\theta = \text{atan2}(m_{01}, m_{10}), \quad m_{pq} = \sum_{x,y} x^p y^q I(x,y)$$

여기에 카메라가 앞뒤로 이동할 때 생기는 스케일 변화까지 견디려면 여러 해상도의 이미지 피라미드에서 특징점을 추출해야 한다. [ORB-SLAM 논문 (Mur-Artal et al., 2015)](https://arxiv.org/abs/1502.00956)에서는 8레벨 피라미드에 스케일 팩터 1.2를 기본 설정으로 사용한다.

**특징점 추적 (Feature Tracking)**

검출한 특징점을 시간 축을 따라 다음 프레임과 연결하는 방법은 크게 두 갈래로 나뉜다.

1. **디스크립터 매칭**: 프레임마다 독립적으로 특징점을 검출한 뒤, 디스크립터 벡터 사이의 거리(이진 디스크립터는 Hamming distance, 부동소수점 디스크립터는 L2)를 비교해 짝을 찾는다. ORB-SLAM3가 채택한 방식으로, DBoW2 어휘 트리(→ §9.2.1)로 탐색 후보를 미리 좁혀 매칭 속도를 확보한다.
2. **광학 흐름 추적 (Optical Flow Tracking)**: 새 프레임에서 특징점을 다시 뽑지 않고, Lucas-Kanade (LK) 추적기로 픽셀 패치를 직접 쫓아가 이전 프레임의 점이 현재 프레임에서 어디로 이동했는지 찾아낸다. VINS-Mono가 이 방식을 쓴다. 광학 흐름 추적은 같은 3D 점이 짧은 시간 간격 동안 비슷한 밝기를 유지한다는 밝기 항상성 가정(brightness constancy assumption)에서 출발한다.

$$I(x + u, y + v, t + \Delta t) = I(x, y, t)$$

변위 $(u, v)$가 작다고 보고 1차 Taylor 전개를 적용한 뒤 윈도우 $W$ 안에서 최소자승으로 풀면, 패치 내 픽셀들의 이동 벡터를 한 번에 구할 수 있다.

$$\begin{bmatrix} u \\ v \end{bmatrix} = \left(\sum_W \begin{bmatrix} I_x^2 & I_xI_y \\ I_xI_y & I_y^2 \end{bmatrix}\right)^{-1} \sum_W \begin{bmatrix} -I_xI_t \\ -I_yI_t \end{bmatrix}$$

이미지 피라미드 위에서 거친 해상도부터 세밀한 해상도 순으로(coarse-to-fine) 추적을 이어 가면 변위가 큰 움직임도 놓치지 않고 따라간다. 두 방식의 트레이드오프는 분명하다. 디스크립터 매칭은 시점 변화(baseline)가 크더라도 잘 버티며 한 번 뽑아 둔 디스크립터를 loop closure에서 재사용할 수 있지만, 매 프레임 검출·기술·매칭에 연산을 써야 한다. 반면 LK 추적은 계산이 가볍고 서브픽셀 단위의 정밀한 좌표를 내지만, 급격한 시점 변화나 큰 조명 변동 앞에서는 추적이 끊기기 쉽다.

**아웃라이어 제거 (Outlier Rejection)**

최소자승 기반의 모션 추정은 단 몇 퍼센트의 오매칭(outlier)만 섞여도 해가 크게 망가진다. 이를 막는 표준 도구가 RANSAC (Random Sample Consensus)이다. RANSAC은 기하 모델을 결정하는 최소 개수의 샘플만 무작위로 뽑아 가설 모델을 세운 뒤, 전체 대응점에서 모델과 부합하는 인라이어 수를 세어 지지를 가장 많이 받는 모델을 택한다. VO에서는 확보한 대응의 형태에 따라 쓰는 최소 모델이 달라진다.

- **2D-2D**: 5-point 알고리즘으로 Essential Matrix $\mathbf{E}$를 추정하고, $\mathbf{E} = [\mathbf{t}]_\times \mathbf{R}$로 분해해 상대 포즈 $(R, t)$를 복원한다.
- **3D-2D**: 이미 삼각측량해 둔 3D 점과 현재 프레임의 2D 관측을 묶어 PnP (Perspective-n-Point) 알고리즘에 RANSAC을 결합해 절대 포즈를 추정한다.
- **3D-3D**: 스테레오나 RGB-D처럼 두 프레임 모두 3D 좌표를 알 때는 ICP 변형으로 강체 변환을 구한다.

2D-2D 대응에서 에피폴라 기하(epipolar constraint)로 아웃라이어를 걸러 낼 때는 다음 식이 기준이 된다.

$$\mathbf{p}_2^T \mathbf{E} \mathbf{p}_1 = 0$$

여기서 $\mathbf{p}_1, \mathbf{p}_2$는 정규화된 이미지 좌표다. 실제 영상에는 센서 노이즈가 섞여 우변이 정확히 0으로 떨어지지 않으므로, 이 식에서 유도되는 기하 오차(예: Sampson 거리)가 허용 임계값을 넘는 쌍을 아웃라이어로 배제한다.

단안 초기화처럼 2D-2D 대응만으로 초기 포즈를 잡을 때는 한 가지 기하학적 함정에 더 주의해야 한다. 벽면이나 바닥처럼 장면의 점들이 한 평면 위에 놓여 있거나 카메라가 순수 회전만 할 때, Fundamental Matrix $\mathbf{F}$는 자유도(7-DoF)가 실제 변환을 유일하게 결정하지 못하는 퇴화(degeneracy) 상태에 빠진다. ORB-SLAM3는 이 문제를 피하려고 Fundamental Matrix와 8-DoF 평면 호모그래피(Homography) $\mathbf{H}$를 동시에 추정한 뒤, 두 모델의 재투영 점수 비율을 비교해 평면 장면이면 $\mathbf{H}$를, 일반 3D 장면이면 $\mathbf{F}$를 선택한다.

### 6.1.2 Backend: PnP, Motion-only BA, Local BA

프런트엔드가 아웃라이어를 털어낸 깔끔한 대응을 넘기면, 이제 백엔드가 이를 메트릭 공간의 카메라 포즈와 정교한 3D 지도 구조로 다듬는다. 실시간성과 전역 일관성을 동시에 챙기기 위해 백엔드는 보통 빠른 포즈 추정(PnP → Motion-only BA)과 주기적인 구조·모션 공동 최적화(Local BA)를 계층적으로 돌린다.

**PnP (Perspective-n-Point)**

PnP는 이미 삼각측량해 둔 3D 맵포인트 $\mathbf{P}_j \in \mathbb{R}^3$와 현재 프레임에서 관측된 2D 좌표 $\mathbf{u}_j \in \mathbb{R}^2$의 짝을 알 때, 현재 카메라 포즈 $\mathbf{T} = (\mathbf{R}, \mathbf{t}) \in SE(3)$를 찾는 문제다.

$$\mathbf{u}_j = \pi(\mathbf{R}\mathbf{P}_j + \mathbf{t})$$

여기서 $\pi: \mathbb{R}^3 \to \mathbb{R}^2$는 카메라 투영 함수다. 최소 3개의 대응점을 쓰는 P3P 알고리즘은 최대 4개의 후보 해를 내므로, 4번째 점에 투영해 오차가 가장 작은 해 하나를 고른다. 실제 시스템에서는 RANSAC 루프 안에서 P3P를 돌려 아웃라이어를 먼저 걷어낸 뒤, 살아남은 모든 인라이어를 묶어 비선형 최적화로 해를 다시 정제한다.

**Motion-only Bundle Adjustment**

PnP로 초기 포즈를 잡았다면 이제 모든 인라이어 관측을 반영해 포즈를 더 정밀하게 다듬어야 한다. 프레임마다 수백·수천 개의 3D 맵포인트까지 한꺼번에 움직이면 연산량이 급증하므로, 추적 단계에서는 3D 점 위치를 고정한 채 카메라 포즈만 최적화하는 Motion-only BA를 수행한다.

$$\mathbf{T}^* = \underset{\mathbf{T}}{\arg\min} \sum_{j} \rho\left(\left\|\mathbf{u}_j - \pi(\mathbf{T} \cdot \mathbf{P}_j)\right\|^2_{\Sigma_j}\right)$$

여기서 $\Sigma_j$는 관측 공분산이고, $\rho(\cdot)$는 큰 잔차의 영향력을 눌러 주는 robust kernel이다. 대표적인 예인 Huber 함수는 오차가 임계값 $\delta$를 넘어가면 제곱 증가를 선형 증가로 꺾어, 남아 있을지 모를 아웃라이어가 전체 해를 끌어당기지 못하게 막는다.

$$\rho(s) = \begin{cases} s & \text{if } s \leq \delta^2 \\ 2\delta\sqrt{s} - \delta^2 & \text{if } s > \delta^2 \end{cases}$$

최적화 대상 $\mathbf{T}$는 유클리드 벡터가 아니라 $SE(3)$ 다양체 위의 6-DoF 포즈이므로, 반복 최적화의 각 스텝에서는 Lie algebra $\mathfrak{se}(3)$의 접공간 증분 $\boldsymbol{\xi} \in \mathbb{R}^6$을 구해 왼쪽 곱으로 포즈를 갱신한다.

$$\mathbf{T} \leftarrow \exp(\boldsymbol{\xi}^{\wedge}) \cdot \mathbf{T}$$

**Local Bundle Adjustment**

고정된 맵포인트에 카메라 포즈만 맞추다 보면 맵포인트 자체에 쌓인 초기 삼각측량 오차가 궤적 드리프트로 번진다. 이를 막기 위해 키프레임이 추가될 때마다 최근 키프레임 $N$개와 이들이 함께 바라보는 맵포인트 전체를 묶어 동시에 최적화하는 Local BA를 실행한다.

$$\{\mathbf{T}_i^*, \mathbf{P}_j^*\} = \underset{\{\mathbf{T}_i, \mathbf{P}_j\}}{\arg\min} \sum_{i,j} \rho\left(\left\|\mathbf{u}_{ij} - \pi(\mathbf{T}_i \cdot \mathbf{P}_j)\right\|^2_{\Sigma_{ij}}\right)$$

"Bundle"이라는 이름은 3D 공간의 각 점에서 여러 카메라 중심으로 뻗어 나가는 광선(ray) 다발에서 유래했다. 수천 개의 3D 점이 변수로 들어가면 정규 방정식(normal equations)이 거대해 보이지만, 각 3D 점은 오직 카메라 포즈들과만 연결될 뿐 다른 3D 점과는 직접 얽히지 않는다. 이 희소 구조 덕분에 점 블록 $\mathbf{H}_{pp}$는 깔끔한 블록 대각 행렬이 된다.

$$\begin{bmatrix} \mathbf{H}_{cc} & \mathbf{H}_{cp} \\ \mathbf{H}_{pc} & \mathbf{H}_{pp} \end{bmatrix} \begin{bmatrix} \delta\boldsymbol{\xi} \\ \delta\mathbf{p} \end{bmatrix} = \begin{bmatrix} \mathbf{b}_c \\ \mathbf{b}_p \end{bmatrix}$$

블록 대각 행렬 $\mathbf{H}_{pp}$의 역행렬은 각 $3 \times 3$ 점 블록을 개별적으로 뒤집기만 하면 적은 비용으로 구할 수 있으므로, Schur complement로 점 변수 $\delta\mathbf{p}$를 소거하면 선형 시스템이 카메라 변수 $\delta\boldsymbol{\xi}$만의 축소 방정식으로 압축된다.

$$(\mathbf{H}_{cc} - \mathbf{H}_{cp}\mathbf{H}_{pp}^{-1}\mathbf{H}_{pc})\delta\boldsymbol{\xi} = \mathbf{b}_c - \mathbf{H}_{cp}\mathbf{H}_{pp}^{-1}\mathbf{b}_p$$

결과적으로 풀어야 할 행렬의 크기는 맵포인트가 수만 개로 늘어나도 윈도우 안의 카메라 수에만 비례해 일정하게 유지된다.

### 6.1.3 ORB-SLAM3 아키텍처 상세 분석

매 프레임 포즈를 추적하면서 동시에 새 맵포인트를 삼각측량해 Local BA까지 돌리려면, 느린 최적화 작업이 프레임 속도의 추적을 가로막지 않도록 스레드를 분리해야 한다. [ORB-SLAM3 (Campos et al., 2021)](https://doi.org/10.1109/TRO.2021.3075644)는 이 설계를 완성한 대표적인 공개 feature-based visual(-inertial) SLAM 시스템으로, monocular·stereo·RGB-D 카메라와 IMU, 나아가 pinhole과 fisheye 렌즈 모델까지 단일 프레임워크로 아우른다.

**전체 아키텍처**

ORB-SLAM3는 서로 다른 시간 규모로 움직이는 세 스레드를 병렬로 구동한다.

1. **Tracking Thread**: 프레임마다 ORB 특징점을 추출하고 기존 맵포인트와 매칭해 현재 포즈를 추정한 뒤, Motion-only BA로 다듬는다.
2. **Local Mapping Thread**: Tracking Thread가 새 키프레임을 넘기면 새 맵포인트를 삼각측량하고 local BA를 돌린다. 동시에 중복된 키프레임과 품질이 떨어지는 맵포인트는 솎아 내어(culling) 맵 크기가 불어나지 않게 유지한다.
3. **Loop Closing & Map Merging Thread**: DBoW2로 재방문 후보를 감지한 뒤 Sim(3)(단안) 또는 SE(3)(스테레오·VI) 정합으로 기하학적 유효성을 검증한다. 루프가 확정되면 pose graph optimization과 full BA를 차례로 수행해 누적 드리프트를 해소한다.

**Tracking Thread 상세**

프레임 하나가 들어오면 Tracking Thread는 다음 네 단계를 거쳐 포즈를 확정한다.

1. 직전 프레임의 속도를 이어받는 등속 모델(constant velocity model)이나 IMU preintegration으로 현재 포즈를 먼저 예측한다.
2. 예측한 포즈로 로컬 맵의 3D 맵포인트를 현재 영상에 투영하고(projection), 투영 위치 주변의 작은 탐색 반경 안에서 ORB 디스크립터를 매칭한다.
3. 확보한 대응으로 Motion-only BA를 풀어 포즈를 정제한다.
4. 이 프레임을 새 키프레임으로 승격할지 판정한다. 마지막 키프레임 이후 일정 수 이상의 프레임이 지나거나 추적된 맵포인트 비율이 임계값 아래로 떨어졌을 때, 그리고 현재 프레임에서 충분한 맵포인트가 관측될 때 키프레임으로 삽입한다.

**Visual-Inertial 모드**

IMU를 함께 쓰는 visual-inertial 모드에서는 [Forster et al. (2017)](https://doi.org/10.1109/TRO.2016.2597321)의 IMU preintegration과 MAP(Maximum-a-Posteriori) 추정을 채택한다. 텍스처가 풍부한 구간에서는 시각 재투영 오차가 장기 드리프트를 잡아 주고, 빠른 회전이나 모션 블러 순간에는 IMU 적분 항이 포즈를 지탱하도록 두 잔차를 하나의 비용 함수에 묶어 함께 최적화한다.

$$\mathcal{C} = \sum_{i,j} \rho\left(\left\|\mathbf{e}_{ij}^{\text{vis}}\right\|^2_{\Sigma_{ij}}\right) + \sum_k \left\|\mathbf{e}_k^{\text{IMU}}\right\|^2_{\Sigma_k^{\text{IMU}}} + \left\|\mathbf{e}^{\text{prior}}\right\|^2_{\Sigma^{\text{prior}}}$$

이 식에서 IMU 잔차 $\mathbf{e}_k^{\text{IMU}}$는 키프레임 $k$와 $k+1$ 사이의 preintegrated IMU 측정값과 상태 추정치가 예측한 변화량 사이의 어긋남을 잰다. 키프레임마다 유지하는 상태 벡터에는 회전·위치·속도와 자이로·가속도계 바이어스가 모두 들어간다.

$$\mathbf{x}_k = [{}^W\mathbf{R}_k, {}^W\mathbf{p}_k, {}^W\mathbf{v}_k, \mathbf{b}_k^g, \mathbf{b}_k^a]$$

§4.6.2에서 유도한 preintegrated 측정

$$\Delta\mathbf{R}_{k,k+1}, \quad \Delta\mathbf{v}_{k,k+1}, \quad \Delta\mathbf{p}_{k,k+1}$$

을 대입하면 회전·속도·위치에 대한 세 가지 IMU 잔차가 다음 꼴로 정리된다.

$$\mathbf{e}^{\text{IMU}}_{\Delta R} = \text{Log}\left(\Delta\hat{\mathbf{R}}_{k,k+1}^T \cdot \mathbf{R}_k^T \mathbf{R}_{k+1}\right)$$
$$\mathbf{e}^{\text{IMU}}_{\Delta v} = \mathbf{R}_k^T(\mathbf{v}_{k+1} - \mathbf{v}_k - \mathbf{g}\Delta t) - \Delta\hat{\mathbf{v}}_{k,k+1}$$
$$\mathbf{e}^{\text{IMU}}_{\Delta p} = \mathbf{R}_k^T(\mathbf{p}_{k+1} - \mathbf{p}_k - \mathbf{v}_k\Delta t - \frac{1}{2}\mathbf{g}\Delta t^2) - \Delta\hat{\mathbf{p}}_{k,k+1}$$

**Multi-Map System (Atlas)**

빠른 회전이나 순간적인 가림 때문에 시각 추적이 끊기면, 이전 세대의 SLAM 시스템들은 초기화에 들어가면서 기존 맵과의 연결을 영영 잃어버리곤 했다. ORB-SLAM3는 Atlas라는 다중 맵 관리 구조로 이 취약점을 해결한다.

1. 추적이 끊기면 현재 맵을 폐기하는 대신 보관해 두고 새 sub-map을 즉시 시작한다.
2. 각 sub-map은 독립적으로 생성·유지된다.
3. 이후 장소 인식(DBoW2)으로 앞서 지나갔던 sub-map 재방문을 포착하면 두 sub-map을 자동으로 병합(merging)한다.
4. 병합 시에는 단안 구성이면 스케일 보정을 포함한 Sim(3)로, 스테레오나 VI 구성이면 강체 변환 SE(3)로 두 맵을 정합한다.

덕분에 단일 주행 중 추적 복구뿐 아니라, 지난 세션에 저장해 둔 지도를 불러와 이어 쓰는 multi-session SLAM도 같은 틀 안에서 동작한다.

**ORB-SLAM3의 평가 범위**: 원 논문은 EuRoC MAV와 TUM-VI 데이터셋에서 stereo-inertial 구성을 평가하고, 각 벤치마크의 프로토콜에 따른 궤적 오차를 보고한다. 다른 시스템과 수치를 견주려면 ATE와 RPE 중 어떤 지표를 썼는지, 대상 시퀀스와 집계 방식, 초기화·트래킹 실패 처리까지 나란히 맞춰야 한다. 새로운 카메라·IMU 조합에 투입할 때도 목표 시퀀스에서 ATE/RPE, 실패율, 처리 시간과 메모리 점유율을 직접 다시 측정하는 편이 안전하다.

아래 수도코드는 지금까지 짚은 Tracking Thread의 한 프레임 처리 흐름을 코드 구조로 옮긴 예시다.

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

Feature-based VO는 코너처럼 뚜렷한 점만 추려 내므로 영상 속 대부분의 픽셀을 버린다. 이 접근은 텍스처가 풍부한 실내외에서는 위력을 발휘하지만, 희미한 콘크리트 벽면이나 어두운 복도처럼 뚜렷한 코너가 드문 장면에서는 추적할 점 자체가 말라 버린다. **Direct VO**는 특징점을 뽑고 디스크립터를 매칭하는 중간 단계를 과감히 생략하고, 이미지 픽셀의 밝기(intensity) 그 자체를 관측값으로 삼는다.

### 6.2.1 Photometric Error

Direct VO의 출발점은 같은 3D 점 $\mathbf{P}$가 연속된 두 프레임에서 동일한 밝기로 찍힌다는 밝기 항상성 가정(brightness constancy)이다. 호스트 프레임 $i$의 픽셀 $\mathbf{u}_i$와 그 지점의 역깊이 $d_i^{-1}$를 알면 3D 공간 좌표를 복원한 뒤 타겟 프레임 $j$의 이미지 평면에 투영해 볼 수 있다. 포즈와 깊이가 정확하다면 타겟 프레임의 투영 위치에서 읽은 밝기는 원래 프레임 $i$의 밝기와 같아야 하므로, 두 밝기의 차이를 **photometric error**로 정의한다.

$$e_{\text{photo}} = I_j\left(\pi(\mathbf{T}_j \mathbf{T}_i^{-1} \pi^{-1}(\mathbf{u}_i, d_i^{-1}))\right) - I_i(\mathbf{u}_i)$$

여기서 각 기호는 다음 연산을 뜻한다.
- $\pi^{-1}(\mathbf{u}, d^{-1})$: 2D 픽셀 좌표와 역깊이를 3D 공간 좌표로 되돌리는 역투영(unprojection)
- $\pi(\cdot)$: 3D 점을 카메라 이미지 평면으로 내리는 3D→2D 투영
- $I_i, I_j$: 프레임 $i, j$의 밝기 이미지

결국 이 잔차를 최소화하는 방향으로 카메라 포즈와 역깊이를 함께 끌어당기는 셈이다. Feature-based VO의 핵심 잔차인 reprojection error와 나란히 놓으면 두 패러다임의 구조적 차이가 선명하게 드러난다.

| | Reprojection error | Photometric error |
|---|---|---|
| 관측 | 특징점 좌표 (2D) | 픽셀 밝기 (1D 또는 패치) |
| 데이터 연관 | 명시적 (매칭 필요) | 암묵적 (와핑으로 계산) |
| 텍스처 요구 | 코너 같은 keypoint 필요 | 에지나 밝기 변화가 약한 영역도 씀 |
| 조명 변화 | detector·descriptor가 대부분 흡수 | 민감 (photometric calibration 필요) |

이 표의 비교 기준은 [DSO 논문(arXiv 1607.02565)](https://arxiv.org/abs/1607.02565) 1절의 direct/indirect 구분과 2.1절 calibration 논의에 바탕을 둔다. 조명 변화 행은 흔히 쓰는 feature extractor와 descriptor가 광도 변화에 불변이거나 매우 강건하다는 2.1절의 기술에 근거한다.

Photometric error를 쓰면 명시적 매칭 없이도 이미지 와핑(warping)만으로 데이터 연관이 자동으로 이뤄지고, 코너가 아닌 완만한 에지나 희미한 밝기 그래디언트까지 모두 모션 추정에 동원할 수 있다. 이 두 특성이 가장 큰 강점이다. 반대로 두 가지 구조적 약점도 따라온다.

1. **밝기 항상성 위반**: 실제 카메라는 자동 노출로 셔터 속도를 바꾸고 렌즈 가장자리에는 비네팅(vignetting)이 생기므로, 조명이 바뀌거나 카메라 각도만 틀어져도 같은 3D 점의 픽셀 값이 달라진다. 이 왜곡을 보정하지 않으면 최적화 자체가 엉뚱한 방향으로 끌려가므로, DSO 같은 시스템은 노출 시간·비네팅·카메라 응답 함수를 정밀하게 모델링하는 photometric calibration을 필수 전처리로 둔다.
2. **좁은 수렴 영역 (Basin of Convergence)**: 픽셀 밝기 공간은 고주파 텍스처 때문에 굴곡이 심해, 초기 포즈가 정답에서 조금만 벗어나도 이미지 그래디언트가 엉뚱한 로컬 미니멈으로 빠진다. 허용되는 초기 오차 폭은 이미지 피라미드 깊이, 텍스처 패턴, 시점·밝기 변화에 따라 달라지므로, 거친 해상도에서 먼저 정렬한 뒤 세밀한 해상도로 내려오는 coarse-to-fine 전략과 검증 잔차 확인이 뒤따라야 한다.

### 6.2.2 DSO (Direct Sparse Odometry) 아키텍처 상세 분석

과거 컴퓨터 비전에서는 "direct 방식은 모든 픽셀을 다 쓰는 dense 또는 semi-dense 방식이고, sparse 방식은 특징점을 뽑는 indirect 방식"이라는 암묵적 등식이 통용됐다. Newcombe et al.의 DTAM(ICCV 2011)이 dense, [LSD-SLAM(Engel et al., 2014)](https://doi.org/10.1007/978-3-319-10605-2_54)이 semi-dense, ORB-SLAM이 sparse·indirect의 대표 주자였다. [DSO (Engel et al., 2018)](https://doi.org/10.1109/TPAMI.2017.2658577)는 잔차를 계산하는 방식(direct vs indirect)과 정보를 샘플링하는 밀도(dense vs sparse)가 서로 독립적인 축이라는 점에 주목해, 직접 밝기를 비교하되 정보량이 큰 픽셀만 고르게 추려 쓰는 새로운 조합을 제시했다.

**DSO의 설계 원칙**

1. **Direct**: 특징점과 디스크립터를 뽑지 않고 픽셀 밝기 잔차를 직접 최소화한다.
2. **Sparse**: 계산 효율을 확보하면서도 텍스처가 희박한 영역을 놓치지 않도록, 이미지 전역에서 그래디언트가 유의미한 점들을 균등하게 샘플링한다.
3. **Joint Optimization**: 카메라 포즈와 점의 역깊이뿐 아니라 카메라 내부 파라미터, 그리고 프레임 간 밝기 변화를 흡수하는 affine brightness parameters까지 한 윈도우 안에서 동시에 최적화한다.

**완전한 Photometric Calibration**

실제 카메라 센서가 기록하는 픽셀 값 $I'$은 이미지 평면에 도달한 실제 복사조도(irradiance) $B$ 그대로가 아니다. 빛이 렌즈를 통과하면서 중심에서 멀어질수록 어두워지는 비네팅 $V(\mathbf{u})$를 겪고, 노출 시간 $t$만큼 적분된 뒤, 카메라 내부의 비선형 응답 함수(gamma 보정 등) $G(\cdot)$를 거쳐 디지털 밝기로 기록된다.

$$I'(\mathbf{u}) = G(t \cdot V(\mathbf{u}) \cdot B(\mathbf{u}))$$

여기서 $G(\cdot)$는 카메라의 비선형 응답 함수, $t$는 노출 시간, $V(\mathbf{u})$는 렌즈 비네팅이다. 이 세 요소를 미리 캘리브레이션해 역연산을 적용하면 복사조도에 선형으로 비례하는 보정 이미지를 복원할 수 있다.

$$I(\mathbf{u}) = t^{-1} \cdot G^{-1}(I'(\mathbf{u})) / V(\mathbf{u})$$

이 과정을 거치지 않으면 같은 3D 표면이라도 화면 중심에서 찍힐 때와 가장자리에 걸릴 때, 또는 자동 노출이 바뀐 다음 프레임에서 밝기가 달라져 photometric error 전체가 왜곡된다.

**Affine Brightness Transfer Function**

사전 photometric calibration이 완벽하지 않거나 노출 시간이 메타데이터로 제공되지 않는 상황까지 대비해, DSO는 두 프레임 사이의 잔여 밝기 차이를 흡수하는 affine brightness 모델을 잔차 식 안에 직접 끼워 넣는다.

$$e_{\mathbf{p}j} = \sum_{\mathbf{p} \in \mathcal{N}_\mathbf{p}} w_\mathbf{p} \left\| (I_j[\mathbf{p}'] - b_j) - \frac{t_j e^{a_j}}{t_i e^{a_i}}(I_i[\mathbf{p}] - b_i) \right\|_\gamma$$

여기서 $a_i, b_i, a_j, b_j$는 각 프레임에 할당된 affine brightness 파라미터로 포즈·깊이와 함께 최적화되며, $\|\cdot\|_\gamma$는 아웃라이어의 영향력을 억누르는 Huber norm이다. 픽셀 하나만 비교하지 않고 주변 작은 패턴 $\mathcal{N}_\mathbf{p}$의 가중합을 쓰는 점도 노이즈에 대한 강건성을 높이는 장치다.

**점 선택 전략**

이미지를 그대로 쓰지 않고 균일한 격자(grid)로 쪼갠 뒤, 셀마다 밝기 그래디언트가 가장 큰 픽셀을 골라 점이 한쪽 텍스처 덩어리에 몰리지 않게 분산시킨다. 한 프레임당 2,000개 안팎의 점을 유지하되, 블록의 평균 그래디언트에 맞춰 임계값을 지역적으로 낮춰 주므로 벽면처럼 텍스처가 빈약한 영역에서도 기하 정보를 빠짐없이 건져 올린다.

**Sliding Window Optimization**

DSO는 최대 7개의 활성 키프레임($N_f = 7$)과 거기에 매달린 점들의 역깊이를 슬라이딩 윈도우 안에 두고 Gauss-Newton 최적화로 동시 정제한다. 전체 최적화 변수 벡터는 다음과 같이 구성된다.

$$\boldsymbol{\theta} = \{\mathbf{T}_1, \ldots, \mathbf{T}_n, d_1^{-1}, \ldots, d_m^{-1}, \mathbf{c}, a_1, b_1, \ldots, a_n, b_n\}$$

카메라 포즈($SE(3)$), 역깊이, 카메라 내부 파라미터 $\mathbf{c}$, affine brightness 파라미터가 한꺼번에 얽혀 풀리는 셈이다. 윈도우 크기를 7개로 제한하기 위해 오래된 프레임이나 시야에서 벗어난 점은 §4.7의 Schur complement 기반 마지널라이제이션과 같은 원리로 소거하되, 그 정보를 버리지 않고 남은 변수들에 대한 사전 확률(prior) 항으로 압축해 넘긴다.

**DSO의 한계와 확장**

순수 오도메트리로 설계된 DSO에는 전역 드리프트를 되감아 주는 loop closure가 빠져 있다. 이 한계를 보완하기 위해 LDSO (Loop-closing DSO)는 DBoW 어휘 트리와 direct 정합을 결합해 루프를 닫았고, VI-DSO는 여기에 IMU 측정을 엮어 빠른 회전에 대한 내성과 스케일을 확보했다. 한편 패치 기반 KLT 추적 프런트엔드와 Non-linear Factor Recovery를 갖춘 Basalt는 계열은 다르지만 효율적인 슬라이딩 윈도우 VIO로 발전했다(→ §6.3.5).

DSO에서 한 프레임을 처리하는 흐름은 photometric 보정부터 피라미드 정합, 점 샘플링, 슬라이딩 윈도우 최적화와 마지널라이제이션으로 이어진다. 이를 수도코드로 나타내면 다음과 같다.

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

Feature-based 방식은 매 프레임 특징점을 뽑고 디스크립터를 매칭하느라 연산을 적지 않게 쓰고, 순수 direct 방식은 photometric calibration과 촘촘한 역깊이 관리가 무겁다. [SVO (Semi-direct Visual Odometry, Forster et al., 2017)](https://doi.org/10.1109/TRO.2016.2623335)는 두 방식의 장점만 취해 추적과 깊이 업데이트에는 direct 방법을, 구조와 모션의 최종 최적화에는 feature-based 방법을 배치한 절충안이다.

**SVO의 처리 단계**

1. **Sparse Model-based Image Alignment**: 이미 알고 있는 3D 맵포인트를 현재 프레임에 투영한 뒤, 투영점 주위 패치의 photometric error만 최소화해 카메라 포즈를 단번에 맞춘다. 이미지 전체가 아니라 기존 맵포인트 주변 패치만 쓰므로 연산이 극도로 가볍다.
2. **Feature Alignment**: 대략적인 포즈를 잡은 뒤 각 맵포인트의 투영 위치를 패치 기반 direct 정합으로 한 번 더 미세 조정해 서브픽셀 정밀도의 2D 좌표를 얻는다.
3. **Structure & Motion Refinement**: 이렇게 다듬어진 2D 위치를 일종의 "가상 특징점"으로 간주하고, 마지막으로 reprojection error 기반 Bundle Adjustment를 풀어 포즈와 3D 점 좌표를 정제한다.

이 3단계 분업 덕분에 SVO 논문은 저자들이 시험한 해상도와 하드웨어 환경에서 수백 Hz에 달하는 높은 처리율을 달성했다. 물론 실제 프레임 속도는 카메라 해상도, 맵 규모, CPU 성능과 구현 세부 사항에 따라 크게 달라지므로 목표 플랫폼에서 직접 재측정해야 한다. 또한 기본 구성은 순수 오도메트리이므로, 깊이 필터가 수렴할 만한 병진 이동(translation)이 없는 제자리 회전 구간에서는 별도의 예외 처리가 뒤따라야 한다.

---

## 6.3 Tightly-Coupled Visual-Inertial Odometry

지금까지 살펴본 feature-based VO와 direct VO는 모두 카메라 한 대만 쓸 때 근본적인 한계에 부딪힌다. 단안 영상만으로는 장면을 몇 배 축소·확대해도 픽셀 투영이 똑같기 때문에 절대 메트릭 스케일을 결정할 수 없고, 빠르게 회전하거나 모션 블러가 생길 때, 또는 텍스처 없는 벽면을 지날 때는 시각 추적 자체가 끊긴다. 여기에 초당 수백에서 수천 번 가속도와 각속도를 내뱉는 IMU를 붙이면 이야기가 달라진다. 가속도계가 중력 방향과 미터 단위의 스케일을 되살려 주고, 자이로스코프가 영상이 흔들리는 짧은 공백 구간을 메워 준다. **Tightly-coupled VIO**는 카메라와 IMU를 따로 돌려 결과 포즈만 섞는 대신, 두 센서의 원시 측정값을 하나의 상태 추정 프레임워크 안에 직접 엮어 서로의 오차를 실시간으로 교정한다.

### 6.3.1 VINS-Mono 아키텍처 상세

[VINS-Mono (Qin et al., 2018)](https://doi.org/10.1109/TRO.2018.2853729)는 단안 카메라 한 대와 저가 IMU만으로 6-DoF 상태를 추정하는 대표적인 tightly-coupled VIO 시스템이다. 초기화부터 슬라이딩 윈도우 오도메트리, loop closure, 맵 재사용까지 실용적인 VIO 파이프라인 전체를 하나의 아키텍처로 구현했다.

**시스템 아키텍처**

전체 파이프라인은 역할이 뚜렷한 세 모듈로 나뉜다.

1. **프런트엔드 (Measurement Preprocessing)**: KLT 광학 흐름으로 특징점을 추적하는 동시에, 프레임 사이에 쏟아진 IMU 측정을 preintegration으로 압축한다.
2. **백엔드 (Estimator)**: 시각 잔차와 IMU preintegration 잔차, 마지널라이제이션 사전 정보를 한 비용 함수에 넣어 비선형 최적화로 푼다.
3. **Loop closure (Relocalization)**: DBoW2 어휘 트리로 재방문 장소를 포착해 전역 궤적을 재정렬한다.

**강건한 초기화 (Robust Initialization)**

단안 VIO에서 가장 까다로운 대목은 시스템이 처음 켜졌을 때의 초기화다. 출발 시점에는 장면의 절대 스케일, 중력 벡터가 기울어진 방향, 초기 속도, 그리고 IMU의 자이로·가속도계 바이어스가 모두 미지수인 데다 이 변수들이 비선형으로 얽혀 있다. 비선형 최적화를 아무 초기값에서나 바로 돌리면 로컬 미니멈에 빠지기 쉬우므로, VINS-Mono는 먼저 loosely-coupled 방식으로 초기값을 안정적으로 추려 낸 뒤 tightly-coupled 최적화로 넘어간다.

**Step 1: 순수 비전 SfM**

먼저 처음 몇 개 프레임에서는 IMU를 쓰지 않고 영상만으로 Structure from Motion(SfM)을 푼다. 5-point 알고리즘으로 Essential Matrix를 구하고 특징점을 삼각측량해 카메라 포즈와 3D 점 구조를 복원하는데, 이 시점의 궤적은 아직 임의의 상수배 스케일만 맞은 상태다.

**Step 2: Visual-Inertial Alignment**

이제 비전 SfM이 내놓은 상대 궤적을 IMU preintegration 결과와 나란히 맞추면서(alignment) 미지수들을 차례로 풀어낸다. 먼저 연속된 키프레임 쌍의 SfM 상대 회전 $\mathbf{R}_{c_k c_{k+1}}^{\text{sfm}}$과 IMU preintegrated 회전 $\Delta\hat{\mathbf{R}}_{k,k+1}$이 일치하도록 자이로 바이어스 $\mathbf{b}_g$부터 보정한다.

$$\min_{\mathbf{b}_g} \sum_k \left\| \text{Log}\left(\Delta\hat{\mathbf{R}}_{k,k+1}(\mathbf{b}_g)^T \cdot \mathbf{R}_{c_k}^{c_{k+1}} \right) \right\|^2$$

§4.6.2에서 본 preintegration 바이어스 자코비안으로 바이어스 변화를 1차 근사하면, 이 회전 정렬 문제는 별다른 반복 없이 선형 최소자승으로 깔끔하게 풀린다. 자이로 바이어스를 잡은 다음에는 각 키프레임 쌍 $(k, k+1)$ 사이의 위치·속도 관계를 엮어 메트릭 스케일 $s$, 중력 벡터 $\mathbf{g}^w$, 그리고 각 키프레임의 속도 $\{\mathbf{v}_k^w\}$를 한꺼번에 구한다.

$$s\mathbf{R}_{c_0}^w \mathbf{p}_{c_{k+1}}^{c_0} - s\mathbf{R}_{c_0}^w \mathbf{p}_{c_k}^{c_0} - \mathbf{v}_k^w \Delta t_k - \frac{1}{2}\mathbf{g}^w\Delta t_k^2 = \mathbf{R}_k^w \Delta\hat{\mathbf{p}}_{k,k+1}$$

여기서 중력 부호는 이 가이드의 좌표계 규약($\mathbf{g}=[0,0,-9.81]^\top$, → 4장)을 따랐다(VINS-Mono 원 논문은 $\mathbf{g}^w$의 부호 정의를 반대로 둔다). 이렇게 선형 해를 구한 뒤에는 지구 중력의 크기 $\|\mathbf{g}\| = 9.81$이 고정되어 있다는 물리 제약을 더해 중력 방향을 구면 접공간 위에서 한 번 더 다듬는다(gravity refinement). 이 loosely-coupled 초기화가 수렴해 스케일과 중력 방향이 자리 잡으면, 비로소 모든 측정을 함께 푸는 tightly-coupled 최적화 단계로 진입한다.

**Tightly-Coupled Nonlinear Optimization**

VINS-Mono의 백엔드는 슬라이딩 윈도우 안의 상태 $\mathcal{X}$(키프레임별 포즈·속도·IMU 바이어스와 특징점 역깊이)에 대해 다음 세 종류의 잔차를 동시에 최소화한다.

$$\min_{\mathcal{X}} \left\{ \left\|\mathbf{r}_p - \mathbf{H}_p \mathcal{X}\right\|^2 + \sum_{k \in \mathcal{B}} \left\|\mathbf{r}_{\mathcal{B}}(\hat{\mathbf{z}}_{b_k b_{k+1}}, \mathcal{X})\right\|^2_{\mathbf{P}_{b_k b_{k+1}}^{-1}} + \sum_{(l,j) \in \mathcal{C}} \left\|\mathbf{r}_{\mathcal{C}}(\hat{\mathbf{z}}_{l}^{c_j}, \mathcal{X})\right\|^2_{(\mathbf{P}_{l}^{c_j})^{-1}} \right\}$$

각 항의 역할은 다음과 같다.
- $\mathbf{r}_p$: 윈도우 밖으로 내보낸 과거 프레임들의 정보를 압축해 넘겨받은 마지널라이제이션 사전(prior) 잔차
- $\mathbf{r}_{\mathcal{B}}$: 연속 프레임 사이 관성 변화를 제약하는 IMU preintegration 잔차
- $\mathbf{r}_{\mathcal{C}}$: 특징점 관측과 3D 투영 위치를 맞추는 시각 reprojection error 잔차

**시각 잔차의 상세 정의**:

특징점 $l$을 호스트 키프레임 $i$에서 처음 관측해 역깊이 $\lambda_l$로 파라미터화했을 때, 다른 키프레임 $j$에서의 reprojection error는 정규화 이미지 평면 위에서 두 좌표의 차이로 정의된다.

$$\mathbf{r}_{\mathcal{C}} = \begin{bmatrix} \bar{u}_l^{c_j} - u_l^{c_j} / z_l^{c_j} \\ \bar{v}_l^{c_j} - v_l^{c_j} / z_l^{c_j} \end{bmatrix}$$

여기서 타겟 카메라 좌표계로 옮겨진 3D 점 좌표는 $\begin{bmatrix} u_l^{c_j} & v_l^{c_j} & z_l^{c_j} \end{bmatrix}^T = \mathbf{R}_{b_j}^{c} (\mathbf{R}_{w}^{b_j}(\mathbf{R}_{b_i}^{w}(\mathbf{R}_{c}^{b_i} \frac{1}{\lambda_l}\begin{bmatrix} \bar{u}_l^{c_i} \\ \bar{v}_l^{c_i} \\ 1 \end{bmatrix} + \mathbf{p}_c^b) + \mathbf{p}_{b_i}^w - \mathbf{p}_{b_j}^w) - \mathbf{p}_c^b)$이다.

**슬라이딩 윈도우 관리와 마지널라이제이션**

연산량을 일정하게 묶어 두려면 윈도우 크기를 10개 안팎으로 제한해야 한다(원 논문과 공개 구현의 기본값 기준). VINS-Mono는 새 프레임이 들어와 윈도우가 찰 때마다 뒤에서 두 번째 프레임(second latest frame)이 키프레임인지 아닌지에 따라 두 갈래로 상태를 정리한다.

1. **뒤에서 두 번째 프레임이 키프레임인 경우**: 시차가 충분한 정보성 프레임이므로 윈도우에 남기고, 대신 가장 오래된 첫 번째 프레임을 내보낸다. 이때 그 프레임에 걸려 있던 시각·IMU 측정을 그냥 버리지 않고 §4.7.2의 Schur complement 마지널라이제이션을 거쳐 prior 항에 녹여 넣는다.
2. **뒤에서 두 번째 프레임이 키프레임이 아닌 경우**: 새 정보가 적은 프레임이므로 그 프레임을 윈도우에서 바로 뺀다. 다만 시각 측정만 버리고, 그 프레임을 가로지르던 IMU preintegration 구간은 다음 프레임까지 그대로 이어 붙여 관성 정보의 연속성은 끊어지지 않게 보존한다.

**4-DoF 포즈 그래프 최적화**

장시간 주행 뒤 루프가 닫히면 VINS-Mono는 6-DoF가 아닌 4-DoF(yaw와 3D 위치) 포즈 그래프 최적화를 돌린다. 충분한 가속·회전 운동과 올바른 중력·바이어스 모델이 주어지면 VIO는 중력 벡터를 기준 삼아 roll과 pitch를 절대적으로 관측할 수 있지만, 수직축 회전인 전역 yaw와 3D 위치에는 여전히 관측 불가능한 자유도(gauge freedom)가 남기 때문이다. 따라서 roll과 pitch는 로컬 VIO 추정에 맡겨 두고, 전역 루프 보정에서는 드리프트가 쌓이는 yaw와 위치만 교정한다.

아래 수도코드에는 지금까지 설명한 VINS-Mono 백엔드의 비용 함수 구성과 두 갈래 마지널라이제이션 로직이 압축되어 있다.

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

VINS-Mono가 단안 VIO의 대중화를 이끌기에 앞서, 키프레임 기반 슬라이딩 윈도우 최적화를 VIO에 일찍 정착시킨 시스템이 [OKVIS (Open Keyframe-based Visual-Inertial SLAM, Leutenegger et al., 2015)](https://doi.org/10.1177/0278364914554813)다. OKVIS는 스테레오 카메라를 기본 센서로 가정하며, 다음과 같은 구성으로 설계되었다.

1. **Harris corner + BRISK descriptor**: KLT 추적 대신 Harris 코너와 이진 디스크립터인 BRISK를 조합해 매칭한다.
2. **Keyframe-based marginalization**: VINS-Mono와 마찬가지로 슬라이딩 윈도우에서 오래된 키프레임 상태를 마지널라이즈해 연산량을 일정하게 묶는다.
3. **IMU error term**: 초기 OKVIS는 두 프레임 사이의 IMU 운동학을 최적화 반복마다 재전파(repropagation)해 자세·속도·위치·바이어스 전체의 15차원 잔차를 세웠다(문헌에서 "speed and bias"는 상태 블록 $[\mathbf{v}, \mathbf{b}_g, \mathbf{b}_a]$를 가리킨다). 후속판인 OKVIS2에서는 연산 효율을 높이기 위해 preintegration으로 전환했다.
4. **Ceres Solver 기반**: 비선형 최적화 백엔드 전체를 Google Ceres Solver 위에서 구동한다.

### 6.3.3 MSCKF: Multi-State Constraint Kalman Filter

VINS-Mono와 OKVIS 같은 슬라이딩 윈도우 최적화 방식은 반복 재선형화로 높은 정확도를 내지만, 윈도우 크기가 커질수록 연산량이 불어나고 반복 수에 따라 지연 시간도 들쑥날쑥해진다. 자원이 빠듯한 임베디드 플랫폼이나 일정한 프레임당 연산 예산이 중요한 시스템에서는 필터 기반 VIO가 강력한 대안이 되며, 그 계보의 출발점이 바로 [MSCKF (Mourikis & Roumeliotis, 2007)](https://doi.org/10.1109/ROBOT.2007.364024)다. 오늘날 널리 쓰이는 OpenVINS(→ §6.3.4) 역시 이 알고리즘을 토대로 한다.

**MSCKF는 랜드마크를 상태에 넣지 않는다**

전통적인 EKF-SLAM은 관측한 3D 랜드마크 좌표를 모두 상태 벡터에 집어넣었다. 랜드마크가 $L$개라면 상태 벡터 차원은 $3L + 15$, 공분산 행렬 크기는 $(3L+15)^2$으로 불어나므로, EKF 갱신 한 번에 드는 메모리와 연산이 모두 $O(L^2)$로 치솟는다. 관측 차원의 혁신 공분산(innovation covariance) 자체가 작아 큰 역행렬 분해를 피할 수는 있지만, 수백·수천 개의 특징점이 쉴 새 없이 드나드는 시각 주행에서는 $O(L^2)$ 비용만으로도 실시간 구동이 버겁다.

MSCKF의 핵심 착상은 3D 랜드마크를 상태 벡터에서 아예 빼버리면서도, 그 랜드마크가 여러 카메라 시점 사이에 걸어 주는 다중 시점 기하 제약(multi-state constraint)만큼은 온전히 살려 내는 데 있다.

**상태 벡터 구조**

MSCKF가 상태 벡터에 유지하는 변수는 현재 시점의 IMU 오차 상태(error-state)와 최근 슬라이딩 윈도우에 속한 카메라 포즈 $N$개뿐이다.

$$\tilde{\mathbf{x}} = [\tilde{\mathbf{x}}_{IMU}^T, \tilde{\mathbf{x}}_{C_1}^T, \ldots, \tilde{\mathbf{x}}_{C_N}^T]^T$$

여기서 15차원 IMU 오차 상태와 각 6차원 카메라 포즈 오차 상태는 다음과 같이 정의된다.

$$\tilde{\mathbf{x}}_{IMU} = [\delta\boldsymbol{\theta}^T, \tilde{\mathbf{b}}_g^T, {}^G\tilde{\mathbf{v}}_I^T, \tilde{\mathbf{b}}_a^T, {}^G\tilde{\mathbf{p}}_I^T]^T \in \mathbb{R}^{15}$$

$$\tilde{\mathbf{x}}_{C_k} = [\delta\boldsymbol{\theta}_{C_k}^T, {}^G\tilde{\mathbf{p}}_{C_k}^T]^T \in \mathbb{R}^6$$

상태 벡터 전체 차원이 $15 + 6N$으로 고정되므로, 장면에 특징점이 몇 개가 뜨든 공분산 행렬 크기는 늘어나지 않는다.

**Null-Space 투영**

그렇다면 상태에 없는 랜드마크 $\mathbf{p}_f$의 오차를 어떻게 EKF 업데이트에서 지워 버릴까? 정적 특징점 $\mathbf{p}_f$ 하나를 슬라이딩 윈도우 내 $M$개의 카메라 포즈에서 관측했다고 하자. 먼저 다중 시점 삼각측량으로 점의 추정 위치를 구한 뒤 관측 방정식을 선형화하면, 잔차는 카메라 상태 오차 $\tilde{\mathbf{x}}$에 의존하는 항과 특징점 위치 오차 $\tilde{\mathbf{p}}_f$에 의존하는 항으로 갈라진다.

$$\mathbf{r} = \mathbf{H}_X \tilde{\mathbf{x}} + \mathbf{H}_f \tilde{\mathbf{p}}_f + \mathbf{n}$$

여기서 각 행렬의 크기는 다음과 같다.
- $\mathbf{r} \in \mathbb{R}^{2M}$: 관측과 예측의 차이로 만든 잔차 벡터
- $\mathbf{H}_X \in \mathbb{R}^{2M \times (15+6N)}$: 카메라 포즈 및 IMU 상태에 대한 자코비안
- $\mathbf{H}_f \in \mathbb{R}^{2M \times 3}$: 특징점 3D 위치에 대한 자코비안
- $\tilde{\mathbf{p}}_f \in \mathbb{R}^3$: 특징점 위치 추정 오차

특징점이 $M \ge 2$개의 시점에서 관측되면 $2M > 3$이므로, $\mathbf{H}_f$에 QR 분해를 적용했을 때 $\mathbf{H}_f$의 열공간과 수직을 이루는 $2M - 3$차원의 왼쪽 영공간(left null space) $\mathbf{Q}_2$가 반드시 존재한다.

$$\mathbf{H}_f = \begin{bmatrix} \mathbf{Q}_1 & \mathbf{Q}_2 \end{bmatrix} \begin{bmatrix} \mathbf{R}_1 \\ \mathbf{0} \end{bmatrix}$$

$\mathbf{Q}_2^T \mathbf{H}_f = \mathbf{0}$이 성립하므로 선형화된 관측 방정식 양변에 $\mathbf{Q}_2^T$를 곱하면, 놀랍게도 $\tilde{\mathbf{p}}_f$ 항이 통째로 사라진다.

$$\mathbf{r}_o = \mathbf{Q}_2^T \mathbf{r} = \mathbf{Q}_2^T \mathbf{H}_X \tilde{\mathbf{x}} + \mathbf{Q}_2^T \mathbf{n} = \mathbf{H}_o \tilde{\mathbf{x}} + \mathbf{n}_o$$

이렇게 랜드마크 변수를 깔끔히 소거한 $\mathbf{r}_o$와 $\mathbf{H}_o$만으로 표준 EKF 업데이트를 수행한다. "Multi-state constraint"라는 이름은 바로 이처럼 특징점 자체를 상태에 넣지 않고도 그 점이 여러 카메라 포즈 사이에 거는 기하학적 제약을 영공간 투영으로 추출해 쓰는 원리에서 왔다.

**계산 복잡도**: 일반적으로 유지하는 카메라 포즈 수 $N$은 랜드마크 수 $L$보다 훨씬 작으므로($N \ll L$), EKF-SLAM의 $O(L^2)$ 갱신 비용에 비해 계산 부담이 극적으로 줄어든다.

**MSCKF 업데이트 절차**

실제 파이프라인에서는 매 스텝 다음 네 단계가 순환한다.

1. **IMU propagation**: 새 IMU 측정이 들어올 때마다 비선형 관성 운동 모델로 상태를 전파하고 오차 상태 공분산 행렬을 갱신한다.
   $$\hat{\mathbf{x}}_{k+1|k} = f(\hat{\mathbf{x}}_{k|k}, \mathbf{u}_k)$$
   $$\mathbf{P}_{k+1|k} = \boldsymbol{\Phi}_k \mathbf{P}_{k|k} \boldsymbol{\Phi}_k^T + \mathbf{G}_k \mathbf{Q} \mathbf{G}_k^T$$
2. **State augmentation**: 새 카메라 프레임이 도착하면 현재 IMU 포즈와 외부 파라미터로 카메라 포즈를 복제해 상태 벡터 뒤에 붙이고, 공분산 행렬도 그 차원만큼 확장한다.
3. **MSCKF update**: 추적이 끝났거나(시야 밖 이탈) 윈도우 길이 한계에 도달한 특징점마다 관측을 모아 삼각측량으로 $\hat{\mathbf{p}}_f$를 구한 뒤, null-space 투영으로 $\mathbf{r}_o, \mathbf{H}_o$를 뽑아 표준 EKF 업데이트를 실행한다.
   $$\mathbf{K} = \mathbf{P} \mathbf{H}_o^T (\mathbf{H}_o \mathbf{P} \mathbf{H}_o^T + \sigma^2 \mathbf{I})^{-1}$$
   $$\hat{\mathbf{x}} \leftarrow \hat{\mathbf{x}} + \mathbf{K} \mathbf{r}_o$$
   $$\mathbf{P} \leftarrow (\mathbf{I} - \mathbf{K}\mathbf{H}_o)\mathbf{P}$$
4. **State pruning**: 슬라이딩 윈도우 한도를 넘겨 더는 쓰지 않는 가장 오래된 카메라 포즈를 상태 벡터와 공분산 행렬에서 마지널라이즈해 덜어 낸다.

아래 C++ 수도코드는 한 특징점의 관측 이력을 받아 삼각측량, QR 분해 기반 영공간 투영, 그리고 EKF 갱신까지 처리하는 핵심 함수를 정리한 예시다.

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

MSCKF의 아이디어를 현대적인 오픈소스 코드베이스로 정교하게 구현해 학계와 실무의 표준 연구 플랫폼으로 자리 잡은 시스템이 [OpenVINS (Geneva et al., 2020)](https://doi.org/10.1109/ICRA40945.2020.9196524)다. 모듈식 구조로 설계되어 다양한 특징 추적기, 센서 구성, 일관성 보정 기법을 갈아 끼우며 정량적으로 비교하기에 좋다.

**OpenVINS의 구성**

1. **On-Manifold Sliding Window EKF**: MSCKF를 $SO(3)$ 매니폴드 위의 오차 상태 칼만 필터로 구현해 회전 표현의 특이점과 선형화 왜곡을 줄인다.
2. **온라인 캘리브레이션**: 설정에 따라 카메라 내부 파라미터, camera-IMU 외부 파라미터, 센서 간 시간 오프셋까지 상태 벡터에 포함해 주행 중 함께 추정한다. 다만 이러한 보정 변수들은 충분한 6축 운동과 풍부한 장면 기하가 뒷받침되어야 수렴하므로, 하드웨어 동기화와 정밀 오프라인 캘리브레이션이 이미 끝난 플랫폼에서는 아예 고정해 두는 편이 안정적이다.
3. **SLAM 랜드마크 지원**: 순수 MSCKF는 모든 특징점을 상태에서 빼지만, 오래 살아남는 특징점까지 매번 윈도우 밖으로 흘려보내면 장기 정보 손실이 생긴다. OpenVINS는 설정에 따라 추적 수명이 긴 일부 특징점을 선별해 SLAM feature로 상태에 직접 올리는 하이브리드 방식을 지원하며, 표현법으로도 원거리 점에 유리한 anchored inverse depth 등을 선택하게 한다.
4. **First-Estimates Jacobian (FEJ)**: 표준 EKF는 스텝마다 선형화 지점이 조금씩 옮겨 가면서 원래 관측 불가능한 전역 yaw나 위치 방향에 가짜 정보가 새어 들어가 공분산이 비정상적으로 줄어드는(overconfident) 문제를 겪는다. OpenVINS는 자코비안 계산 시 최초 추정 지점을 고정하는 FEJ를 적용해 이 비일관성을 완화한다. 물론 FEJ는 일관성을 지켜 주는 중요한 장치일 뿐, 모델 오차나 노이즈 튜닝 오류, 모든 비선형 오차까지 만능으로 해결하지는 않는다.
5. **시뮬레이터**: 궤적, 3D 환경, IMU 노이즈 파라미터를 바꾸어 가며 ground truth와 궤적 오차를 즉시 대조할 수 있는 내장 시뮬레이터를 제공한다.

### 6.3.5 Basalt

지금까지 살펴본 최적화 기반 VIO(VINS-Mono, OKVIS)에도 한 가지 미묘한 제약이 숨어 있다. 슬라이딩 윈도우에서 오래된 프레임을 내보낼 때 Schur complement로 만든 마지널라이제이션 prior는 마지널라이즈 시점의 선형화 지점에 묶여 버린다. 나중에 loop closure가 걸려 전역 궤적이 크게 꺾이면, 이미 과거 선형화 지점에 얼어붙은 prior 정보는 어긋난 선형 근사 위에 서 있게 된다.

[Basalt (Usenko et al., 2020)](https://doi.org/10.1109/LRA.2019.2961227)는 이 문제를 해결한 스테레오 기반 tightly-coupled VIO 시스템으로, 다음과 같은 독자적 설계를 채택했다.

**Basalt의 구성**

1. **Visual-only Frontend**: VINS-Mono와 같은 광학 흐름 계열이지만, 점 하나 대신 패치 단위로 $SE(2)$ 워프를 적용하는 inverse-compositional KLT 추적을 써서 서브픽셀 정밀도와 회전 내성을 높였다.
2. **Non-linear Factor Recovery (NFR)**: Basalt의 가장 큰 차별점이다. 로컬 VIO가 주행 중 축적한 마지널라이제이션 정보 행렬을 그대로 얼려 두는 대신, 그 정보 분포를 가장 잘 근사하는 새로운 비선형 factor 집합(키프레임 간 상대 포즈 및 roll·pitch 기울기 제약)으로 복원한다. 복원된 factor는 비선형 함수 형태를 되찾았기 때문에 이후 전역 bundle adjustment 과정에서 얼마든지 다시 선형화될 수 있으며, 덕분에 loop closure와 결합한 전역 맵에서도 roll과 pitch의 관측 가능성이 살아남는다([Usenko et al., 2020](https://arxiv.org/abs/1904.06504)).
3. **Efficient Implementation**: Factor graph의 희소 블록 구조를 정교하게 활용해 빠른 실행 속도를 끌어냈다. 다만 시스템 간 속도를 비교할 때는 센서 구성(Basalt는 stereo+IMU 기반이므로 mono+IMU와 직접 견주기 어렵다), 해상도, 하드웨어 조건을 반드시 맞춰야 한다.
4. **Stereo/Multi-camera 지원**: 스테레오를 비롯한 다중 카메라 영상을 하나의 공동 최적화 프레임워크 안에 통합한다.

---

## 6.4 VIO 설계 선택지

앞에서 살펴본 시스템들은 저마다 다른 설계 축의 조합으로 구현되어 있다. 실제 로봇 플랫폼에 맞춰 VIO 파이프라인을 설계할 때는 추정기 구조, 키프레임 관리, 특징점 표현이라는 세 가지 핵심 선택지가 계산 비용과 정확도, 운용 한계를 가른다.

### 6.4.1 Filter vs Optimization

상태 추정에 필터를 쓸지 비선형 최적화를 쓸지는 VIO 아키텍처에서 가장 오래 논의되어 온 갈림길이다.

**Filter 기반 (MSCKF, OpenVINS)**은 현재 시점의 IMU 상태와 일부 과거 카메라 포즈만 유지한 채 새 측정이 들어올 때마다 순차적으로 상태를 갱신한다. 윈도우에서 내려보낸 과거 상태의 정보는 남은 상태들의 평균과 공분산 행렬 안에 녹아 들어가며, 업데이트 한 번의 연산 복잡도는 상태 차원 $N$에 대해 $O(N^2)$으로 일정하게 묶인다. 구현이 간결하고 프레임당 계산 비용이 예측 가능하다는 장점이 뚜렷하지만, 측정을 한 번 선형화해 공분산에 반영하고 나면 뒤늦게 선형화 지점을 되돌릴 수 없어 선형화 오차가 쌓이고 일관성(consistency)이 흔들리기 쉽다는 약점이 뒤따른다.

**Optimization 기반 (VINS-Mono, ORB-SLAM3, Basalt)**은 슬라이딩 윈도우 안의 여러 상태와 관측을 함께 쥐고 있다가 매 스텝 반복 재선형화(iterative re-linearization)를 거쳐 비선형 최적화를 푼다. 반복당 복잡도는 윈도우 크기 $N$에 대해 $O(N^3)$ 수준이지만 Schur complement로 랜드마크 변수를 소거해 계산을 줄인다. 여러 센서의 잔차를 자유롭게 엮어 넣기 좋고 재선형화로 선형화 오차를 크게 낮출 수 있는 반면, 연산 비용이 높고 윈도우 크기에 따라 지연시간이 민감하게 변한다.

**실험적 비교**: 어느 패러다임이 절대적으로 우위에 있는지는 데이터셋과 센서 구성, 계산 예산을 동일하게 맞춘 실험을 바탕으로 판단해야 한다(→ §4.4.4). 필터 계열 역시 FEJ나 관측 가능성 제약을 정교하게 걸어 주면 일관성 문제를 상당 부분 덜어 낼 수 있어, 연산 자원이 극도로 제한된 플랫폼에서는 필터가 지닌 일정한 연산 비용이 여전히 매력적인 무기가 된다.

### 6.4.2 Keyframe Selection 전략

슬라이딩 윈도우 최적화에서 키프레임을 너무 자주 넣으면 연산량만 키울 뿐이고, 시차가 좁아 삼각측량 정밀도도 떨어진다. 반대로 너무 드물게 넣으면 시각 제약이 성겨져 추적 공백이 생긴다. 실제 시스템은 보통 다음 기준 가운데 하나 이상을 조합해 키프레임 삽입 시점을 정한다.

1. **시간 기반**: 마지막 키프레임 이후 일정 시간이 지나거나 일정 프레임 수에 도달하면 삽입한다.
2. **시차(parallax) 기반**: 현재 프레임과 직전 키프레임 사이의 평균 특징점 시차가 임계값을 넘어설 때 새 키프레임을 세운다. VINS-Mono가 대표적으로 쓰는 기준으로, 카메라가 제자리에 서 있거나 회전만 할 때 불필요한 키프레임이 쌓이는 일을 막는다.
3. **추적 품질 기반**: 추적 중인 특징점 수나 비율이 기준 아래로 떨어지면 새 관측을 보충하기 위해 키프레임을 넣는다. ORB-SLAM3가 이 전략을 쓴다.
4. **Information gain 기반**: 새 키프레임이 상태 불확실성을 얼마나 줄여 주는지, 즉 정보 이득을 직접 추정해 결정한다. 이론적으로 가장 깔끔하지만 정보 행렬 계산 부담이 크다.

키프레임 선택 규칙은 곧바로 마지널라이제이션 정책과 맞물린다. 앞서 본 VINS-Mono의 two-way marginalization(→ §6.3.1)이 대표적이다. 이 방식에서는 새 프레임이 키프레임인지에 따라 가장 오래된 프레임을 마지널라이즈할지, 직전 비키프레임을 덜어 낼지가 갈린다.

### 6.4.3 Feature Parameterization

3D 랜드마크를 최적화 상태에 어떤 좌표로 표현하느냐에 따라 수치 안정성과 선형화 오차가 크게 달라진다.

**XYZ (Euclidean 3D 좌표)**는 점의 위치를 직교 좌표 $\mathbf{P} = [X, Y, Z]^T \in \mathbb{R}^3$로 그대로 둔다. 직관적이지만 카메라에서 멀리 떨어진 점(far point)에서는 다루기 까다롭다. 시차가 조금만 흔들려도 추정 깊이 $Z$가 크게 요동쳐 비선형성이 급격히 커지기 때문이다.

**Inverse Depth**는 점을 "호스트 프레임에서 바라본 단위 방향 + 역깊이"로 바꿔 표현한다.

$$\boldsymbol{\lambda} = [\theta, \phi, \rho]^T$$

여기서 $\theta, \phi$는 호스트 프레임 기준 방위각과 고도각, $\rho = 1/d$는 깊이의 역수다. 이 표현은 두 가지 면에서 수치적 이점을 준다.

1. **원거리 점 처리**: 실제 거리가 $d \to \infty$로 멀어져도 역깊이는 $\rho \to 0$으로 매끄럽게 수렴한다. $Z \to \infty$로 발산하는 XYZ 좌표와 달리 무한원 근처의 점도 무리 없이 상태에 담아 둘 수 있다.
2. **선형성 개선**: 단안 VIO 초기화처럼 깊이 불확실성이 큰 구간에서는 깊이 $d$ 자체보다 역깊이 $\rho$의 불확실성 분포가 가우시안에 훨씬 가깝기 때문에 선형화 근사가 잘 맞는다.

VINS-Mono와 OpenVINS가 이 역깊이 파라미터화를 기본으로 채택한다.

**Anchored Inverse Depth**는 한 걸음 더 나아가 역깊이를 처음 관측한 특정 "앵커" 키프레임의 좌표계에 고정(anchored)해 둔다.

$$\mathbf{P} = \mathbf{T}_{\text{anchor}} \cdot \frac{1}{\rho} [\bar{u}, \bar{v}, 1]^T$$

여기서 $(\bar{u}, \bar{v})$는 앵커 프레임 기준 정규화 이미지 좌표, $\rho$는 역깊이다. 궤적이 수정되어 앵커 프레임의 포즈가 움직이더라도 앵커 기준 역깊이 값 자체는 크게 흔들리지 않으므로 재선형화 오차를 줄이는 효과가 있다. OpenVINS는 장기 추적용 SLAM feature에 이 표현을 지원한다. 반대로 ORB-SLAM3는 스테레오·RGB-D처럼 초기 깊이를 직접 재는 구성까지 포괄하며, 맵포인트를 전역 세계 좌표계의 3D 위치로 저장·최적화한다.

---

## 6.5 학습 기반 VO/VIO

고전적 VO/VIO는 특징점 검출에서 매칭, RANSAC, Bundle Adjustment까지 사람이 설계한 기하 모듈을 차례로 이어 붙인다. 딥러닝 기반 연구는 이 파이프라인의 일부 혹은 전체를 신경망으로 대체하려 시도해 왔으며, 초기의 단순 포즈 회귀를 넘어 최근에는 신경망과 미분 가능한 기하 최적화를 결합하는 방향으로 수렴하고 있다.

### 6.5.1 Supervised: DeepVO 계열

초기 학습 기반 VO를 연 [DeepVO (Wang et al., 2017)](https://doi.org/10.1109/ICRA.2017.7989236)는 연속 이미지 쌍을 입력받아 상대 포즈를 곧장 예측하는 end-to-end 지도 학습 네트워크를 제안했다. CNN이 두 영상에서 모션 특징을 뽑고 LSTM이 시간 축에 걸친 운동 연속성을 학습하는 구조였다. 하지만 명시적인 에피폴라 기하나 3D 투영 제약을 적용하지 않고 순수 회귀에 의존한 탓에, 학습 데이터셋과 다른 환경에 투입하면 일반화 성능이 떨어지고 정확도 역시 전통 기하 파이프라인에 못 미쳤으며 스케일 드리프트도 심하게 남았다.

### 6.5.2 Self-supervised: 한계와 현재 위치

자기지도 학습 VO(SfMLearner, Monodepth2 등)는 정답 포즈 레이블을 구하기 어려운 한계를 넘기 위해 등장했다. 이 방식은 한 시점의 예측 깊이와 상대 포즈로 옆 시점의 영상을 합성(view synthesis)한 뒤, 실제 영상과의 photometric loss를 줄이는 방식으로 깊이와 포즈를 동시에 학습한다. 별도의 정답 궤적 없이도 학습할 수 있다는 점이 돋보였으나, 정적 장면 가정을 깨는 움직이는 물체나 텍스처가 빈약한 영역, 시점 이동에 따른 가림(occlusion) 앞에서는 오차가 커지는 한계를 넘지 못했다. 오늘날 이 계열의 방법들은 완전한 오도메트리 엔진보다는 단안 깊이 추정(monocular depth estimation) 영역에서 주로 쓰인다.

### 6.5.3 Hybrid: DROID-SLAM

순수 신경망 회귀가 기하 정합의 정밀도를 따라잡지 못했던 벽을 깬 시스템이 [DROID-SLAM (Teed & Deng, 2021)](https://arxiv.org/abs/2108.10869)이다. DROID-SLAM은 대응점을 찾아 다듬는 일은 딥러닝 네트워크에 맡기되, 포즈와 깊이를 푸는 단계에는 전통적인 Bundle Adjustment를 미분 가능한 레이어 형태로 그대로 끼워 넣었다. 원 논문에서는 DROID-SLAM이 TartanAir, EuRoC, TUM-RGBD, ETH-3D 벤치마크에서 기존 학습 기반 방법은 물론 고전 기하 기반 방법보다도 오차를 크게 낮추고, 추적이 완전히 무너지는 catastrophic failure를 현저히 줄였다고 보고했다.

**아키텍처**

DROID-SLAM의 미분 가능 파이프라인은 두 핵심 블록이 맞물려 돌아간다.

1. **RAFT 기반 반복 업데이트 연산자 (Iterative Update Operator)**

광학 흐름 모델인 RAFT (Recurrent All-Pairs Field Transforms, Teed & Deng, 2020)의 구조를 프레임 그래프 위로 옮겼다. 연결된 프레임 쌍 $(i, j)$마다 다음과 같이 correspondence를 점진적으로 교정한다.
- 두 프레임의 특징 맵에서 모든 픽셀 쌍의 유사도를 담은 4D correlation volume을 미리 계산해 둔다.
- 현재 추정 중인 포즈와 깊이로 correspondence field를 계산한 뒤, 이 필드가 가리키는 위치 주변의 상관 특징을 correlation volume에서 읽어 온다.
- 3×3 Convolutional GRU가 상관 특징, 현재 flow, 문맥 특징(context)을 입력받아 픽셀별 flow 수정량 $\Delta\mathbf{f}_{ij}$와 이 대응을 얼마나 믿을지 나타내는 신뢰도 가중치 $\mathbf{w}_{ij}$를 출력한다.

$$(\Delta\mathbf{f}_{ij}, \mathbf{w}_{ij}) = \text{ConvGRU}(\text{corr}_{ij}, \mathbf{f}_{ij}^{\text{curr}}, \text{context}_i)$$

이 연산을 여러 차례(iteratively) 반복하면서 correspondence field를 조금씩 정답 쪽으로 당겨 놓는다.

2. **미분 가능 Dense Bundle Adjustment (DBA) 레이어**

GRU가 수정한 flow를 실제 3D 기하 업데이트로 번역해 주는 단계다. 모든 픽셀에 대해 reprojection error를 세우고, 신뢰도 가중치 $\mathbf{w}_{ij}^{\mathbf{p}}$를 곱한 비용 함수를 카메라 포즈 $\mathbf{T}_i \in SE(3)$와 픽셀별 역깊이 $d_i$에 대해 Gauss-Newton법으로 최소화한다.

$$\sum_{(i,j)} \sum_{\mathbf{p}} \left\| \mathbf{w}_{ij}^{\mathbf{p}} \circ (\mathbf{p}^{*}_{ij} - \pi(\mathbf{T}_j \circ \mathbf{T}_i^{-1} \circ \pi^{-1}(\mathbf{p}, d_i^{\mathbf{p}}))) \right\|^2$$

여기서 $\mathbf{p}^{*}_{ij} = \mathbf{p} + \mathbf{f}_{ij}^{\text{curr}} + \Delta\mathbf{f}_{ij}$는 GRU가 제안한 목표 대응 좌표고, $\mathbf{w}_{ij}^{\mathbf{p}}$는 해당 픽셀 대응의 신뢰도 가중치다.

이 목적 함수에 Gauss-Newton 스텝을 적용하면 고전 BA와 정확히 같은 꼴의 정규 방정식이 얻어진다.

$$\begin{bmatrix} \mathbf{B} & \mathbf{E} \\ \mathbf{E}^T & \mathbf{C} \end{bmatrix} \begin{bmatrix} \boldsymbol{\xi} \\ \delta\mathbf{d} \end{bmatrix} = \begin{bmatrix} \mathbf{v} \\ \mathbf{w} \end{bmatrix}$$

여기서 $\boldsymbol{\xi}$는 $SE(3)$ 접공간의 포즈 증분, $\delta\mathbf{d}$는 역깊이 증분이며, $\mathbf{B}$는 포즈 블록, $\mathbf{C}$는 대각 행렬인 깊이 블록, $\mathbf{E}$는 포즈와 깊이 사이의 교차 블록이다.

고전 BA에서 3D 점을 소거했던 것처럼 Schur complement를 적용하면 역깊이 변수 $\delta\mathbf{d}$를 깨끗이 소거할 수 있다.

$$(\mathbf{B} - \mathbf{E}\mathbf{C}^{-1}\mathbf{E}^T)\boldsymbol{\xi} = \mathbf{v} - \mathbf{E}\mathbf{C}^{-1}\mathbf{w}$$

깊이 블록 $\mathbf{C}$가 대각 행렬이므로 역행렬 $\mathbf{C}^{-1}$은 대각 성분의 역수를 취하는 것만으로 즉시 계산된다. 덕분에 픽셀 수가 아무리 많아도 축소된 선형 시스템은 크기가 키프레임 수에만 비례해 빠르게 풀린다. 게다가 이 Gauss-Newton 솔버 전체가 미분 가능하기 때문에, 학습 과정에서는 최종 포즈·깊이 손실의 역전파를 솔버에 거꾸로 통과시켜 GRU가 "기하 최적화에 유리한 correspondence와 가중치"를 내뱉도록 종단 간 학습한다.

**프레임 그래프와 loop closure**

DROID-SLAM은 실행 중에 공시성(co-visibility) 기반의 프레임 그래프를 동적으로 가꾸어 나간다. 새 키프레임이 들어오면 인접 프레임과 엣지로 잇고, 과거 키프레임과 시야 겹침이 포착되면 장거리 엣지를 추가해 루프를 닫은 뒤 전체 키프레임 이력에 대해 전역 BA를 푼다.

**다중 모달리티 지원**

단안 영상만으로 학습한 모델이지만 추론 시 스테레오와 RGB-D 입력도 별도 재학습 없이 그대로 받아들인다.
- **스테레오**: 좌우 영상을 각각 별도 프레임 노드로 취급하되, 알려진 좌우 카메라 사이의 상대 포즈는 DBA 레이어 안에서 고정해 메트릭 스케일을 강제한다.
- **RGB-D**: 센서가 준 깊이 맵에는 노이즈와 결측 영역이 섞여 있으므로 깊이를 상수로 못 박는 대신 변수로 열어 두고, 목적 함수(원 논문 식 4)에 측정 깊이와 예측 깊이 차이의 제곱 페널티 항을 더해 보강한다.

**성능**

원 논문에서 보고한 정량 비교 결과는 다음과 같다.

- TartanAir: 단안 트랙에서 기존 최고 성능 대비 오차 62% 감소(스테레오 트랙에서는 60% 감소)
- EuRoC monocular: 추적 실패 없이 완주한 방법들과 비교해 오차 82% 감소(ORB-SLAM3와는 ORB-SLAM3가 추적에 성공한 10개 시퀀스 기준으로 43% 감소)
- ETH-3D: RGB-D 시퀀스 32개 중 30개 추적 성공(차순위 방법은 19개)

이 모든 수치는 합성 데이터셋인 TartanAir의 단안 영상만으로 한 번 학습한 단일 모델에서 얻은 결과다.

**한계**

물론 GPU 기반 고밀도 연산 구조에 따른 현실적인 제약도 뚜렷하다.
- **연산량**: 공개 구현은 GPU 가속을 전제로 하며, 실제 프레임 속도는 입력 해상도와 활성 키프레임 수, 하드웨어 사양에 따라 크게 달라진다.
- **IMU 미포함**: 기본 DROID-SLAM은 순수 영상 입력만 처리하므로 IMU 결합은 별도 확장을 거쳐야 한다.
- **메모리**: 활성 그래프에 걸린 모든 키프레임의 dense feature map을 메모리에 붙들고 있어야 해 메모리 사용량이 크다.

이 한계를 풀기 위해 고밀도 플로우를 희소 패치로 줄인 후속 연구(DPVO)와 메모리 경량화, 관성 센서 통합 연구들이 뒤를 이었다.

### 6.5.4 2023–2025년 연구

[DPVO (Teed, Lipson & Deng, 2023)](https://arxiv.org/abs/2208.04726)는 DROID-SLAM의 dense optical flow 계산을 과감하게 덜어 내고, 이미지에서 무작위로 뽑은 희소 패치(sparse patch) 단위 매칭으로 대체했다. 패치 단위 recurrent update 연산자와 미분 가능 BA를 결합한 이 구조는 원 논문의 비교 설정에서 DROID-SLAM 대비 메모리 사용량을 약 1/3로 줄이고 실행 속도는 약 3배 끌어올리면서도 더 낮은 평균 오차를 기록했다.

[MAC-VO (Qiu et al., ICRA 2025)](https://arxiv.org/abs/2409.09479)는 학습된 매칭 불확실성(metrics-aware covariance)을 스테레오 VO 파이프라인에 직접 끌어들였다. 이 방법은 신경망이 예측한 공분산으로 추적할 키포인트를 선별하고 포즈 그래프 최적화의 잔차 가중치까지 정교하게 조절했다. 원 논문에 따르면 그 결과 조명 급변과 텍스처 부족 구간을 포함한 공개 벤치마크에서 기존 방법보다 낮은 궤적 오차를 달성했다. 이 논문은 ICRA 2025 Best Paper Award on Robot Perception을 수상했다.

---

## 요약

지금까지 살펴본 주요 VO·VIO 시스템들의 관측 방식, 상태 추정 엔진, 지원 센서와 핵심 설계 특징은 아래 표로 한눈에 정리된다.

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

결국 카메라와 IMU를 결합하면 시각 텍스처가 살아 있는 환경에서 가볍고 정밀한 6-DoF 오도메트리가 가능하다. 하지만 칠흑 같은 어둠이나 강렬한 역광이 있는 환경, 혹은 수십 미터 너머의 3D 구조물을 즉각 측량해야 하는 실외 주행에서는 가시광 수동 센서의 물리적 한계가 드러난다. 센서를 능동적으로 3D 거리를 재는 LiDAR로 바꾸면 대응점 탐색과 오차 모델링, 심지어 스캔 도중 발생하는 모션 왜곡을 다루는 방식까지 근본적으로 달라진다. 이어지는 7장에서는 포인트 클라우드 정합에서 출발해 LOAM과 LIO-SAM, FAST-LIO 계열로 이어지는 LiDAR 오도메트리와 LIO 아키텍처를 살펴본다.
