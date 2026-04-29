# Ch.1 — Why Sensor Fusion?

자율주행 차량, 드론, 서비스 로봇은 모두 같은 한 가지 질문에서 출발한다. 어떤 센서로 세상을 인지할 것인가. 어떤 단일 센서도 답이 되지 못한다는 사실에서 센서 퓨전(Sensor Fusion)이라는 분야가 시작했다. 이 가이드는 그 분야의 이론과 실전을 다루며, 첫 챕터는 왜 단일 센서가 부족한지, 어떤 결합 방식이 있는지, 고전 방법과 딥러닝이 어디에서 서로를 보완하는지 살핀다.

---

## 1.1 단일 센서의 한계

각 센서는 물리적 원리에 기반하여 환경의 특정 측면을 관측한다. 그리고 바로 그 물리적 원리 때문에 본질적인 한계를 갖는다.

### 카메라의 한계

카메라는 풍부한 시각 정보를 제공하지만 한계가 분명하다.

**조명 의존성.** 카메라는 피사체로부터 반사된 빛을 감지하는 수동(passive) 센서이다. 따라서 야간, 터널, 역광 등 조명 조건이 열악한 환경에서 성능이 급격히 저하된다. 자동 노출(auto-exposure)로 일부 완화할 수 있으나, 센서 자체의 다이내믹 레인지(dynamic range)를 넘어서는 장면에서는 포화(saturation) 또는 언더익스포저(underexposure)가 불가피하다.

**스케일 모호성.** 단안(monocular) 카메라는 3D 세계를 2D 이미지로 투영하면서 깊이 정보를 잃는다. 2m 거리의 1m 물체와 20m 거리의 10m 물체는 이미지에서 동일한 크기로 나타날 수 있다. 단안 비주얼 오도메트리(monocular visual odometry)가 절대 스케일을 복원할 수 없는 것도 이 스케일 모호성(scale ambiguity) 때문이다. 스테레오 카메라나 다른 센서와의 융합 없이는 미터 단위의 정확한 거리 추정이 원천적으로 불가능하다.

**텍스처리스(textureless) 환경.** 흰 벽, 긴 복도, 넓은 포장도로처럼 시각적 특징이 부족한 환경에서는 특징점(feature point) 추출과 추적이 실패한다. Direct 방식의 비주얼 오도메트리도 포토메트릭 그래디언트(photometric gradient)가 부족하면 동일한 문제에 직면한다.

**모션 블러.** 고속 이동이나 급격한 회전 시 이미지가 흐려지며, 특징점 추출과 매칭 성능이 크게 저하된다.

### LiDAR의 한계

LiDAR(Light Detection And Ranging)는 레이저 펄스를 발사하고 반사파의 비행시간(Time-of-Flight)을 측정하여 정밀한 3D 거리 정보를 제공하는 능동(active) 센서이다. 하지만 다음과 같은 한계를 갖는다.

**텍스처 정보 부재.** LiDAR는 기하학적 구조(geometry)는 정밀하게 캡처하지만, 색상이나 텍스처는 거의 제공하지 않는다 (반사 강도(intensity)는 일부 가능하나 카메라 이미지에 비할 바는 아니다). 그래서 구조가 유사한 장소—예컨대 같은 모양의 건물이 반복되는 거리—에서 장소 인식(place recognition)이 어렵다.

**날씨 및 환경 민감성.** 비, 안개, 눈, 먼지에서 레이저 빔이 산란돼 허위 반사점(ghost points)이 쏟아지거나 탐지 거리가 짧아진다. 검은색 물체와 고반사 표면(유리, 금속)에서도 측정이 불안정하다.

**저해상도 및 비용.** 기계식 스피닝(spinning) LiDAR의 수직 해상도는 16~32채널 수준이다. 고해상도 모델은 수천에서 수만 달러를 호가한다. 최근 솔리드 스테이트(solid-state) LiDAR가 가격을 낮추고 있지만, 좁아진 시야각(FoV)이 트레이드오프다.

### IMU의 한계

관성 측정 장치(Inertial Measurement Unit, IMU)는 가속도와 각속도를 측정하는 센서로, 고주파(100Hz–1kHz)의 자기수용적(proprioceptive) 데이터를 제공한다. 외부 환경에 의존하지 않는다는 것이 최대 장점이나, 치명적인 한계가 있다.

**드리프트(drift).** IMU 측정을 적분하여 속도와 위치를 계산하면, 센서 바이어스(bias)와 노이즈가 시간에 따라 적분되어 오차가 누적된다. 가속도의 이중 적분으로 위치를 구하면 오차는 $t^2$에 비례하여 발산한다. 항법 등급(navigation-grade)의 고급 IMU조차 수 분 내에 상당한 위치 오차를 보이며, 로보틱스에 흔히 사용되는 MEMS급 IMU는 수 초 만에 미터 단위의 오차가 발생한다.

$$\delta \mathbf{p}(t) \approx \frac{1}{2} \mathbf{b}_a \, t^2 + \frac{1}{\sqrt{3}} \sigma_a \, t^{3/2}$$

여기서 $\mathbf{b}_a$는 가속도계 바이어스, $\sigma_a$는 가속도 노이즈 밀도이다. 바이어스가 $0.01\,\text{m/s}^2$만 되어도 10초 후 위치 오차는 0.5m에 달한다.

**절대 기준 부재.** IMU는 상대적 변화만 측정하며, 절대 위치나 절대 방위(heading)에 대한 정보를 제공하지 않는다. 중력 방향으로부터 롤(roll)과 피치(pitch)를 추출할 수 있으나, 요(yaw)는 자력계(magnetometer) 없이는 관측 불가능하다(observability 문제).

### GNSS의 한계

위성 항법 시스템(Global Navigation Satellite System, GNSS)은 전지구적 절대 위치를 제공하지만, 다음과 같은 한계가 있다.

**차폐 환경.** 실내, 터널, 도심의 고층 빌딩 사이(urban canyon)에서는 위성 신호가 차단되거나 다중경로(multipath) 반사로 수십 미터의 오차가 생긴다.

**업데이트 주기.** 수신기 출력 주기는 1–10Hz로, 빠른 동적 운동을 추적하기 어렵다.

**정밀도 한계.** 표준 단독 측위의 정밀도는 수 미터 수준이다. RTK(Real-Time Kinematic)를 사용하면 센티미터 급으로 향상되지만, 기준국(base station)이 필요하고, 초기 수렴(convergence)에 시간이 걸린다.

### 센서 한계의 상보성

위에서 본 각 센서의 한계를 표로 모아 보면, 한 센서의 약점이 다른 센서의 강점에서 메워지는 패턴이 보인다.

| 특성 | 카메라 | LiDAR | IMU | GNSS |
|------|--------|-------|-----|------|
| 절대 위치 | ✗ | ✗ | ✗ | ✓ |
| 상대 이동 (단기) | ✓ | ✓ | ✓ | ✗ |
| 상대 이동 (장기) | △ (드리프트) | △ (드리프트) | ✗ (발산) | ✓ |
| 고주파 모션 캡처 | ✗ | ✗ | ✓ | ✗ |
| 조명 독립 | ✗ | ✓ | ✓ | ✓ |
| 날씨 강건성 | △ | ✗ | ✓ | ✓ |
| 3D 기하정보 | △ (깊이 모호) | ✓ | ✗ | ✗ |
| 텍스처/의미 정보 | ✓ | ✗ | ✗ | ✗ |
| 실내 동작 | ✓ | ✓ | ✓ | ✗ |
| 비용 | 저 | 고 | 중~저 | 중 |

어떤 단일 센서도 모든 상황을 감당하지 못한다. 센서 퓨전(Sensor Fusion)은 이 한계를 체계적으로 메우는 방법이다.

---

## 1.2 센서 퓨전의 분류

센서 퓨전은 여러 센서의 정보를 결합해 개별 센서로는 닿지 못하는 정확도와 강건성을 끌어내는 기술이다. 결합 방식에 따라 세 가지 범주로 나뉜다.

### Complementary Fusion (상보적 융합)

서로 다른 물리량을 측정하는 센서들이 각자의 부족한 부분을 보완하는 형태이다. 각 센서는 전체 상태(state)의 서로 다른 부분집합을 관측하며, 이들을 결합하면 단일 센서로는 관측할 수 없는 완전한 상태를 추정할 수 있다.

**대표 예시: 카메라 + IMU (Visual-Inertial Odometry, VIO)**

- 카메라는 6-DoF 포즈의 상대 변화를 제공하지만 스케일이 모호하고 고속 모션에서 실패한다.
- IMU는 고주파의 가속도/각속도를 제공하여 카메라 프레임 사이의 빠른 모션을 보간하고, 중력 방향으로부터 스케일을 복원한다.
- 두 센서는 서로 보완한다. 카메라가 제공하지 못하는 스케일·고주파 모션을 IMU가, IMU가 제공하지 못하는 드리프트 보정을 카메라가 맡는다.

**대표 예시: GNSS + IMU**

- GNSS는 저주파(1–10Hz)의 절대 위치를 제공한다.
- IMU는 고주파(수백 Hz)의 상대 이동을 제공한다.
- GNSS가 끊기는 터널 내부에서는 IMU가 단기 항법을 이어가고, GNSS가 복귀하면 누적된 IMU 드리프트를 교정한다.

이 유형의 핵심은 **관측 가능성(observability)**의 확장이다. 다른 센서의 관측이, 단일 센서로는 관측 불가능한(unobservable) 상태 변수를 관측 가능하게 만든다.

### Competitive Fusion (경쟁적 융합)

동일한 물리량을 측정하는 센서를 중복(redundancy)으로 배치하여 신뢰성과 정확도를 높이는 형태이다.

**대표 예시: 다중 카메라 시스템**

- 동일 방향을 바라보는 두 카메라가 각각 독립적으로 특징점을 추적한다.
- 한 카메라가 오염(렌즈 오염, 고장)되어도 나머지 카메라로 시스템이 계속 동작한다.
- 두 추정치를 결합하면 개별 추정치보다 분산이 줄어든다.

**통계적 기초.** 두 독립 관측 $z_1 \sim \mathcal{N}(\mu, \sigma_1^2)$, $z_2 \sim \mathcal{N}(\mu, \sigma_2^2)$를 최적으로 결합하면:

$$\hat{\mu} = \frac{\sigma_2^2 z_1 + \sigma_1^2 z_2}{\sigma_1^2 + \sigma_2^2}, \quad \sigma_{\text{fused}}^2 = \frac{\sigma_1^2 \sigma_2^2}{\sigma_1^2 + \sigma_2^2}$$

결합된 추정치의 분산은 항상 개별 추정치의 분산보다 작다: $\sigma_{\text{fused}}^2 < \min(\sigma_1^2, \sigma_2^2)$. 이것은 칼만 필터(Kalman Filter)의 갱신 단계와 정확히 동일한 원리이다.

**대표 예시: 다중 LiDAR 시스템**

자율주행 차량에서 4–6개의 LiDAR를 차량 주위에 배치하여 360° 시야를 확보하는 동시에, 겹치는 영역에서 중복 관측으로 신뢰성을 높이는 것이 일반적이다.

### Cooperative Fusion (협력적 융합)

각 센서의 원시(raw) 데이터를 결합하여 어느 단일 센서로도 불가능한 **새로운 형태의 정보**를 생성하는 형태이다.

**대표 예시: 스테레오 비전 (Stereo Vision)**

- 좌우 카메라의 이미지를 결합하여 시차(disparity)를 계산한다.
- 시차로부터 3D 깊이 정보를 복원한다.
- 단일 카메라로는 깊이를 알 수 없지만, 두 카메라의 협력으로 새로운 물리량(깊이)이 생성된다.

**대표 예시: 카메라 + LiDAR → 컬러 포인트 클라우드**

- LiDAR 포인트를 카메라 이미지에 투영하여 각 3D 점에 색상을 부여한다.
- 결과물인 컬러 포인트 클라우드(colored point cloud)는 LiDAR의 정밀 기하정보와 카메라의 풍부한 텍스처를 동시에 가지며, 이는 개별 센서로는 생성 불가능하다.
- [R3LIVE (Lin et al., 2022)](https://arxiv.org/abs/2109.07982) 같은 시스템이 이를 실시간으로 수행한다.

**대표 예시: Radar + Camera → 악천후 인지**

- 레이더의 도플러(Doppler) 측정과 카메라의 시각 정보를 결합하여, 안개나 비 속에서도 이동 물체의 속도와 클래스를 동시에 인식한다.

세 범주는 상호 배타적이지 않다. 실제 시스템은 종종 이 세 가지를 동시에 활용한다. 예를 들어, 자율주행 차량의 센서 퓨전 시스템은 카메라+LiDAR의 cooperative 융합(컬러 포인트 클라우드), 카메라+IMU의 complementary 융합(VIO), 다중 LiDAR의 competitive 융합(중복 커버리지)을 모두 포함한다.

---

## 1.3 결합 수준에 따른 분류: Loosely vs Tightly vs Ultra-tightly Coupled

센서 퓨전의 또 다른 중요한 분류 축은 **센서 데이터가 결합되는 수준**이다. 이 분류는 시스템의 정확도, 복잡도, 강건성에 직접적인 영향을 미친다.

### Loosely Coupled (느슨한 결합)

각 센서 서브시스템이 독립적으로 자신의 추정치를 산출하고, 이 결과(output)들을 상위 레벨에서 결합하는 방식이다.

**구조:**
```
센서A → [서브시스템A] → 추정치A ─┐
                                   ├─→ [Fusion] → 최종 추정
센서B → [서브시스템B] → 추정치B ─┘
```

**대표 예시: 독립 VO + 독립 LiDAR Odometry → EKF Fusion**

- Visual Odometry가 독립적으로 포즈를 출력한다.
- LiDAR Odometry가 독립적으로 포즈를 출력한다.
- 상위의 EKF가 두 포즈를 결합하여 최종 추정치를 생성한다.

**장점:**
- 모듈성: 각 서브시스템을 독립적으로 개발, 테스트, 교체할 수 있다.
- 단순성: 퓨전 레이어의 설계가 상대적으로 간단하다.
- 부분 실패 대응: 한 서브시스템이 실패해도 다른 서브시스템의 출력으로 계속 동작 가능하다.

**단점:**
- 정보 손실: 각 서브시스템이 내부적으로 관측 정보를 요약(compress)하여 출력하므로, 원시 관측의 세부 정보(예: 각 특징점의 개별 불확실성)가 손실된다.
- 상관관계 무시: 독립 서브시스템들이 공통 관측(예: 동일 IMU 데이터)을 사용할 경우, 두 추정치가 상관관계를 가지게 되지만 이를 무시하면 과신(overconfidence)이 발생한다. 이는 "이중 카운팅(double counting)" 문제로 알려져 있다.
- 최적성 상실: 정보가 요약되므로 전체 시스템이 정보 이론적으로 최적(optimal)이 되지 못한다.

### Tightly Coupled (긴밀한 결합)

모든 센서의 **원시 관측(raw measurement)**을 단일 추정 프레임워크(single estimator)에 직접 투입하는 방식이다.

**구조:**
```
센서A → raw 관측A ─┐
                     ├─→ [Single Estimator] → 최종 추정
센서B → raw 관측B ─┘
```

**대표 예시: [VINS-Mono (Qin et al., 2018)](https://arxiv.org/abs/1708.03852)**

- 카메라의 원시 특징점 관측과 IMU의 원시 가속도/각속도 측정을 하나의 비선형 최적화(sliding-window optimization)에 함께 넣는다.
- 최적화의 비용 함수는 재투영 오차(reprojection error)와 IMU 사전적분 잔차(preintegration residual)를 동시에 최소화한다.

$$\min_{\mathcal{X}} \left\{ \sum_{(i,j) \in \mathcal{B}} \| \mathbf{r}_{\text{IMU}}(\mathbf{x}_i, \mathbf{x}_j) \|^2_{\mathbf{P}_{ij}} + \sum_{(i,l) \in \mathcal{C}} \| \mathbf{r}_{\text{cam}}(\mathbf{x}_i, \mathbf{f}_l) \|^2_{\mathbf{\Sigma}_l} \right\}$$

여기서 $\mathbf{r}_{\text{IMU}}$는 IMU 사전적분 잔차, $\mathbf{r}_{\text{cam}}$은 시각 재투영 잔차, $\mathcal{X}$는 전체 상태(포즈, 속도, 바이어스, 랜드마크)이다.

**대표 예시: [FAST-LIO2 (Xu et al., 2022)](https://arxiv.org/abs/2107.06829)**

- LiDAR의 개별 포인트와 IMU의 원시 측정을 하나의 반복 에러스테이트 칼만 필터(Iterated Error-State Kalman Filter)에 직접 투입한다.

**대표 예시: [LIO-SAM (Shan et al., 2020)](https://arxiv.org/abs/2007.00258)**

- LiDAR 특징점, IMU 사전적분, GNSS 위치 관측을 하나의 팩터 그래프(factor graph)에서 통합 최적화한다.

**장점:**
- 정보 최대 활용: 원시 관측의 모든 정보를 활용하므로 정보 이론적으로 더 나은 추정이 가능하다.
- 교차 보정: 센서 간 상호 보정(cross-calibration)이 자연스럽게 이루어진다. 예를 들어, 카메라 관측이 IMU 바이어스 추정에 기여하고, IMU 데이터가 카메라 특징점 추적을 안정화한다.
- 열화 대응(graceful degradation): 한 센서의 관측 수가 줄어들어도(예: 특징점이 적은 환경) 나머지 센서의 관측이 추정을 지탱한다.

**단점:**
- 복잡도: 단일 추정기에 모든 센서의 관측 모델을 구현해야 하므로 시스템 설계와 디버깅이 복잡하다.
- 계산 비용: 상태 벡터가 커지고 관측 수가 많아져 계산량이 증가한다.
- 센서 의존성: 특정 센서를 제거하거나 교체하려면 전체 추정기를 수정해야 한다.

### Ultra-tightly Coupled (초긴밀 결합)

센서의 **신호 수준(signal level)**에서 융합이 이루어지는 방식이다. 이 용어는 주로 GNSS/INS 통합에서 사용된다.

**대표 예시: GNSS/INS Ultra-tight Integration**

- 일반적인 tightly coupled에서는 GNSS 수신기가 의사거리(pseudorange)를 출력하고 이를 항법 필터에 입력한다.
- Ultra-tight에서는 INS의 예측 속도를 GNSS 수신기 내부의 코드/반송파 추적 루프(tracking loop)에 피드백한다.
- 이를 통해 수신기의 추적 루프 대역폭을 줄여 노이즈에 대한 내성을 높이고, 심한 간섭이나 약신호 환경에서도 위성 추적을 유지할 수 있다.

**비전 분야에서의 유사 개념:**

비전-관성 시스템에서 ultra-tight coupling에 해당하는 것은 IMU 예측을 이용하여 카메라의 특징점 탐색 영역을 제한하거나, 이미지 왜곡(motion blur) 보정에 IMU 데이터를 직접 사용하는 것이다. VINS-Mono에서 IMU 예측으로 특징점 추적의 초기값을 설정하는 것이 이에 가깝다.

### 결합 수준 비교

| 특성 | Loosely Coupled | Tightly Coupled | Ultra-tightly Coupled |
|------|----------------|-----------------|----------------------|
| 융합 레벨 | 결과(output) | 관측(measurement) | 신호(signal) |
| 정보 활용도 | 낮음 | 높음 | 최고 |
| 구현 복잡도 | 낮음 | 중간~높음 | 매우 높음 |
| 모듈성 | 높음 | 낮음 | 매우 낮음 |
| 부분 실패 대응 | 용이 | 설계 필요 | 어려움 |
| 대표 시스템 | 독립 VO + LO → EKF | VINS-Mono, LIO-SAM, FAST-LIO2, ORB-SLAM3 | GNSS/INS deep integration |

현대 로보틱스에서는 **tightly coupled** 방식이 주류이다. Loosely coupled는 구현이 간단하지만 정보 손실로 인해 정확도가 떨어지고, ultra-tightly coupled는 특수한 하드웨어 접근이 필요하여 적용 범위가 제한된다. VINS-Mono, FAST-LIO2, LIO-SAM 등 현재 가장 널리 사용되는 오픈소스 시스템들은 모두 tightly coupled 아키텍처를 채택하고 있다. 최근에는 LiDAR-관성-비전 세 센서를 단일 프레임워크에서 tightly coupled로 융합하는 [FAST-LIVO2 (Zheng et al., 2024)](https://arxiv.org/abs/2408.14035)가 정확도와 실시간 성능 모두에서 기존 시스템을 크게 능가하는 결과를 보여주고 있다.

---

## 1.4 Classical vs Learning-based: 딥러닝이 바꾼 것과 바꾸지 못한 것

센서 퓨전 분야는 수십 년간 확률론적 추정 이론(Kalman Filter, Factor Graph)과 기하학적 방법(epipolar geometry, ICP)에 기반한 **고전적(classical)** 접근법이 지배해왔다. 2010년대 중반 이후 딥러닝이 컴퓨터 비전의 거의 모든 영역을 혁신하면서, 센서 퓨전에도 학습 기반(learning-based) 방법이 빠르게 침투하고 있다. 그러나 그 침투의 양상은 영역에 따라 크게 다르다.

### 딥러닝이 바꾼 것

**특징점 추출과 매칭.** 전통적으로는 SIFT, ORB 같은 수작업 설계(handcrafted) 특징 기술자를 썼다. [SuperPoint (DeTone et al., 2018)](https://arxiv.org/abs/1712.07629)는 자기 지도 학습(self-supervised learning)으로 키포인트 검출과 기술을 동시에 수행하며, 조명과 시점 변화에 대한 강건성을 크게 높였다. [SuperGlue (Sarlin et al., 2020)](https://arxiv.org/abs/1911.11763)는 그래프 뉴럴 네트워크(GNN)와 어텐션 메커니즘으로 특징점 매칭에 적용하여, 수작업 기술자 기반의 최근접 이웃 매칭보다 낮은 오매칭률을 기록했다. 가장 최근에는 [LoFTR (Sun et al., 2021)](https://arxiv.org/abs/2104.00680), [RoMa (Edstedt et al., 2024)](https://arxiv.org/abs/2305.15404) 같은 **detector-free** 방법이 키포인트 없이 직접 밀집 대응(dense correspondence)을 찾아, 텍스처가 부족한 환경에서도 매칭에 성공하고 있다.

이 영역에서 학습 기반 방법은 전통 방법을 명확히 능가한다. **패러다임 전환**이 진행 중이다.

**장소 인식(Place Recognition).** Bag of Words (DBoW2)에서 [NetVLAD (Arandjelović et al., 2016)](https://arxiv.org/abs/1511.07247)로의 전환은 성능 격차가 컸다. CNN 기반의 전역 기술자(global descriptor)는 조명, 계절 변화가 있는 환경에서 DBoW2보다 재인식률이 크게 높았다. 최근 [AnyLoc (Keetha et al., 2023)](https://arxiv.org/abs/2308.00688)은 DINOv2 같은 Foundation Model의 특징을 활용하여 별도의 학습 없이도 다양한 환경에서 범용적으로 동작하는 장소 인식을 보여주었다.

**단안 깊이 추정.** 단일 이미지로부터 깊이를 추정하는 것은 고전적 방법으로는 불가능한 작업이다(기하학적 단서가 부족). [Depth Anything (Yang et al., 2024)](https://arxiv.org/abs/2401.10891)은 대규모 데이터에서 학습하여 KITTI에서 metric 오차를 0.1m 이하로 끌어내렸다. 후속작인 [Depth Anything V2 (Yang et al., 2024)](https://arxiv.org/abs/2406.09414)는 합성 데이터 학습과 대규모 pseudo-labeling을 통해 정밀도를 더 끌어올렸고, [Metric3D v2 (Hu et al., 2024)](https://arxiv.org/abs/2404.15506)는 zero-shot으로 절대 스케일의 깊이 추정까지 가능해, LiDAR 없이도 메트릭 깊이 정보를 센서 퓨전에 가져올 수 있는 길을 열었다.

**맵 표현.** NeRF와 3D Gaussian Splatting은 장면을 신경망으로 표현하는 새로운 패러다임을 열었다. NeRF-SLAM, Gaussian Splatting SLAM 등은 전통적인 점 지도(point map)나 복셀 격자(voxel grid)를 넘어서는 포토리얼리스틱 맵 표현을 제공한다.

**이벤트 카메라(Event Camera).** 뉴로모픽 비전 센서로 불리는 이벤트 카메라는 각 픽셀이 밝기 변화를 비동기적으로 감지하여, 극도로 높은 시간 분해능(마이크로초 수준)과 넓은 다이나믹 레인지를 제공한다. 최근 [이벤트 카메라 서베이 (Huang et al., 2024)](https://arxiv.org/abs/2408.13627)가 정리하듯, 이벤트 기반 VIO와 SLAM 연구가 빠르게 발전하고 있으며, 기존 프레임 기반 카메라와의 퓨전을 통해 고속 모션과 저조도 환경에서 새로운 가능성을 열고 있다.

### 딥러닝이 바꾸지 못한 것

**상태 추정(State Estimation) 백엔드.** 칼만 필터와 팩터 그래프 최적화 같은 확률론적 추정 프레임워크는 딥러닝이 대체하지 못했다. 네 가지 측면에서 그렇다.

1. **불확실성의 엄밀한 전파**: 칼만 필터와 팩터 그래프는 관측의 불확실성을 수학적으로 엄밀하게 추적하고 전파한다. 딥러닝 모델이 유사한 수준의 calibrated uncertainty를 제공하기 어렵다.
2. **물리 법칙의 보장**: 상태 전이 모델(동역학, 기구학)에 물리 법칙을 직접 인코딩하여 물리적으로 불가능한 추정을 방지한다. 학습 기반 방법은 이런 하드 제약(hard constraint)을 보장하지 못한다.
3. **데이터 효율성**: 확률론적 프레임워크는 센서 노이즈 모델과 시스템 모델만 있으면 데이터 없이도 동작한다. 학습 기반 방법은 대규모 학습 데이터가 필요하다.
4. **일반화**: 학습 기반 오도메트리(예: DeepVO)는 학습 데이터와 다른 환경에서 성능이 크게 떨어진다. 기하학적 방법은 환경에 구애받지 않는다.

**LiDAR 오도메트리.** [LOAM (Zhang & Singh, 2014)](https://frc.ri.cmu.edu/~zhangji/publications/RSS_2014.pdf) 이후의 LiDAR 오도메트리는 여전히 전통 방법이 주류다. ICP, GICP, NDT 같은 점군 정합 방법은 수학적으로 잘 이해되어 있고, 실시간 성능을 내며, 새로운 환경에 곧장 적용된다. 학습 기반 LiDAR 오도메트리(DeepLO 계열)는 아직 전통 방법의 정확도와 일반화 성능에 미치지 못한다.

**캘리브레이션.** 카메라 내부 파라미터 캘리브레이션은 [Zhang (2000)](https://doi.org/10.1109/34.888718)의 체커보드 방법이 여전히 표준이다. 타겟리스(targetless) 캘리브레이션에서 학습 기반 방법이 연구되고 있지만, 정밀도 면에서 타겟 기반 방법을 넘어서지 못하고 있다.

### Hybrid 접근: 현재의 주류

현재 가장 성공적인 시스템들은 학습 기반 프런트엔드(frontend)와 고전적 백엔드(backend)를 결합하는 **하이브리드** 구조를 취한다.

```
[학습 기반 프런트엔드]           [고전적 백엔드]
 SuperPoint/SuperGlue          Factor Graph / EKF
 Mono Depth Estimation    →    Nonlinear Optimization
 Semantic Segmentation         Kalman Filtering
 Place Recognition              Pose Graph SLAM
```

- 프런트엔드에서 딥러닝이 원시 센서 데이터로부터 고수준 특징(feature point, depth, semantic label)을 추출한다.
- 백엔드에서 기하학적/확률론적 프레임워크가 이 특징들을 시간적으로 일관된 상태 추정으로 통합한다.

[DROID-SLAM (Teed & Deng, 2021)](https://arxiv.org/abs/2108.10869)은 이러한 하이브리드 접근의 좋은 예이다. 학습된 특징 추출과 대응 찾기를 사용하면서, 최종 포즈 추정은 미분 가능한(differentiable) 번들 조정(Bundle Adjustment)으로 수행한다.

### 기술 계보 요약

이 가이드는 각 주제에서 **전통 방법 → 딥러닝이 가능하게 한 것 → 아직 전통이 필요한 부분**의 흐름을 일관되게 보여줄 것이다. 아래 표는 이 가이드 전체를 관통하는 기술 계보를 요약한다.

| 영역 | Classical | Learning-based | 현재 주류 |
|------|-----------|---------------|----------|
| Feature matching | SIFT/ORB + BF/FLANN | SuperPoint+SuperGlue → LoFTR → RoMa | Hybrid |
| Visual odometry | Feature-based (ORB) / Direct (DSO) | DROID-SLAM, DPV-SLAM | 전통 우세, 학습 추격 |
| LiDAR odometry | ICP/LOAM | DeepLO 계열 | 전통 압도적 우세 |
| Place recognition | BoW/VLAD | NetVLAD → AnyLoc | 학습 우세 |
| Depth estimation | Stereo matching | Mono depth (Depth Anything) | 학습 우세 (mono) |
| Calibration | Target-based | Targetless + learning | 전통 우세 |
| Map representation | Occupancy/TSDF | NeRF / 3DGS | 공존 |
| State estimation backend | KF / Factor Graph | End-to-end 시도 | 전통 압도적 우세 |

표에서 눈에 띄는 패턴이 있다. **지각(perception)에 가까울수록 학습이 강하고, 추론(inference/estimation)에 가까울수록 전통이 강하다.** 센서 퓨전 시스템을 설계할 때 어디에 딥러닝을 투입하고 어디에 전통 방법을 유지할지, 이 패턴이 기준이 된다.

---

## 1.5 이 가이드의 범위와 구성

### robotics-practice와의 관계

이 가이드는 robotics-practice 가이드의 심화 편이다. robotics-practice가 Spatial AI 전반을 넓게 조망하는 입문서라면, 이 가이드는 **센서 퓨전, 로컬라이제이션, 리트리벌**에 집중한 심화 레퍼런스이다.

- robotics-practice에서 EKF/PF를 1–2페이지로 소개했다면, 이 가이드에서는 칼만 필터의 유도 과정부터 ESKF, IMU 사전적분, 팩터 그래프 최적화까지 챕터 단위로 깊이 있게 다룬다.
- robotics-practice에서 센서 소개를 개략적으로 했다면, 이 가이드에서는 각 센서의 **노이즈 모델과 관측 방정식**을 수식으로 유도한다.
- 겹치는 기초 내용은 robotics-practice 참조로 대체하고, 이 가이드에서는 추가적인 깊이만 다룬다.

### 가이드 구성

이 가이드는 다음과 같은 구성을 따른다:

1. **Ch.1 — Introduction** (이 챕터): 센서 퓨전의 동기와 분류
2. **Ch.2 — Sensor Modeling**: 각 센서의 관측 모델과 노이즈 특성 (수식 중심)
3. **Ch.3 — Calibration**: 다양한 센서 조합의 캘리브레이션 이론과 실전
4. **Ch.4 — State Estimation 이론**: 베이지안 필터링, 칼만 필터 계열, 파티클 필터, 팩터 그래프
5. **Ch.5 — Feature Matching & Correspondence**: SIFT에서 RoMa까지의 기술 계보
6. **Ch.6 — Visual Odometry & VIO**: VO/VIO 시스템의 내부 구조
7. **Ch.7 — LiDAR Odometry & LIO**: LiDAR 기반 오도메트리와 LiDAR-관성 퓨전
8. **Ch.8 — Multi-Sensor Fusion 아키텍처**: 다중 센서 통합 설계론
9. **Ch.9 — Place Recognition & Retrieval**: BoW에서 Foundation Model까지
10. **Ch.10 — Loop Closure & Global Optimization**: 루프 클로저와 전역 최적화
11. **Ch.11 — Spatial Representations**: 점 지도에서 Neural Map까지
12. **Ch.12 — 실전 시스템 & 벤치마크**: 실제 응용과 평가
13. **Ch.13 — Frontiers**: 최신 동향과 열린 문제

### 대상 독자

이 가이드의 대상 독자는 다음과 같은 배경을 가진 로보틱스 입문자이다:

- **선형대수**: 행렬 연산, 고유값 분해, SVD에 대한 이해
- **확률론**: 확률 분포, 조건부 확률, 베이즈 정리, 가우시안 분포에 대한 이해
- **기초 최적화**: 최소자승법, 그래디언트 디센트의 개념
- **Python**: numpy, scipy 기반 코드를 읽고 실행할 수 있는 수준

이 가이드의 각 챕터는 **직관 → 수식 → 코드/예제**의 흐름을 따른다. 먼저 개념의 직관적 이해를 제공하고, 수학적으로 엄밀하게 유도한 뒤, Python 코드로 구현하여 확인한다.

---

## 1.6 이 가이드의 관통 테마

이 가이드 전체를 읽으면서 독자가 반복적으로 만나게 될 핵심 질문들이 있다:

1. **"왜 이 전통 방법이 중요했는가?"** — 각 전통 방법이 해결한 근본적 문제와 그 해법의 우아함을 이해한다.
2. **"딥러닝이 뭘 바꿨는가?"** — 학습 기반 방법이 전통 방법의 어떤 한계를 극복했는지 구체적으로 본다.
3. **"아직 전통이 필요한 부분은 어디인가?"** — 딥러닝이 대체하지 못한 영역과 그 이유를 분석한다.
4. **"이론과 실전의 간극은 어디에 있는가?"** — 논문과 실제 시스템 사이의 차이, 실전에서 마주치는 엔지니어링 문제를 다룬다.

이 질문들을 염두에 두고 각 챕터를 읽으면, 개별 알고리즘의 이해를 넘어서 센서 퓨전이라는 분야의 전체 그림을 조망할 수 있을 것이다.

다음 챕터에서는 각 센서의 관측 모델을 수식으로 유도한다. 센서가 세상을 어떻게 "보는지"가 모든 퓨전 알고리즘의 출발점이다.
