# Ch.1 — Why Sensor Fusion?

자율주행 차량, 드론, 서비스 로봇은 주변을 인지할 센서를 골라야 한다. 센서마다 측정하는 물리량과 약점이 달라서, 여러 센서를 묶어 쓰는 센서 퓨전(Sensor Fusion)이 필요해졌다.

---

## 1.1 단일 센서의 한계

센서의 한계는 그 센서가 재는 물리 원리에서 나온다.

### 카메라의 한계

**조명 의존성.** 카메라는 피사체에서 반사된 빛을 감지하는 수동(passive) 센서다. 그래서 야간, 터널, 역광처럼 빛이 모자라거나 고르지 않은 곳에서 성능이 크게 떨어진다. 자동 노출(auto-exposure)로 어느 정도 버틸 수 있지만, 밝기 차가 센서의 다이내믹 레인지(dynamic range)를 넘는 장면에서는 밝은 곳이 하얗게 날아가거나(saturation) 어두운 곳이 뭉개진다(underexposure).

**스케일 모호성.** 단안(monocular) 카메라는 3D 세계를 2D 이미지로 투영하면서 깊이 정보를 잃는다. 2m 거리의 1m 물체와 20m 거리의 10m 물체는 이미지에서 같은 크기로 찍힌다. 이 스케일 모호성(scale ambiguity) 때문에 단안 비주얼 오도메트리(monocular visual odometry)는 궤적을 임의의 배율로만 복원한다. 미터 단위 거리를 정하려면 스테레오 기선, 다른 센서의 측정, 또는 물체 크기의 사전지식이 필요하다.

**텍스처리스(textureless) 환경.** 흰 벽, 긴 복도, 넓은 포장도로처럼 시각적 특징이 부족한 환경에서는 특징점(feature point)을 뽑지도 추적하지도 못한다. 픽셀 밝기를 직접 쓰는 direct 방식 비주얼 오도메트리도 밝기 기울기(photometric gradient)가 부족하면 같은 문제를 겪는다.

**모션 블러.** 빠르게 움직이거나 급하게 돌면 노출 시간 동안 상이 번져 특징점을 뽑고 짝짓기가 어려워진다.

### LiDAR의 한계

LiDAR(Light Detection And Ranging)는 레이저 펄스를 발사하고 반사파의 비행시간(Time-of-Flight)을 재서 3D 거리를 정밀하게 얻는 능동(active) 센서다.

**텍스처 정보 부재.** LiDAR는 형상(geometry)을 정밀하게 재지만 색과 텍스처는 거의 담지 못한다. 반사 강도(intensity)를 함께 기록하는 기종이 많아도 카메라 영상만큼의 정보는 없다. 그래서 같은 모양의 건물이 이어지는 거리처럼 구조가 비슷한 곳에서는 장소 인식(place recognition)이 어렵다.

**날씨 및 환경 민감성.** 비, 안개, 눈, 먼지에서 레이저 빔이 흩어져 가짜 반사점(ghost points)이 생기거나 탐지 거리가 짧아진다. 빛을 흡수하는 검은 물체나 유리·금속처럼 거울같이 반사하는 표면에서도 측정이 불안정하다.

**저해상도 및 비용.** 기계식 스피닝(spinning) LiDAR의 수직 채널 수는 저가 모델에서 16~32개, KITTI에 쓰인 HDL-64E에서 64개, 상위 모델에서 128개까지 이른다. 채널이 많은 모델일수록 값이 비싸다. 회전부가 없는 솔리드 스테이트(solid-state) LiDAR가 나오면서 값은 내려갔지만 시야각(FoV)이 좁다.

### IMU의 한계

관성 측정 장치(Inertial Measurement Unit, IMU)는 자신이 달린 물체의 가속도와 각속도를 100Hz–1kHz로 재는 고유수용(proprioceptive) 센서다. 바깥 환경에 기대지 않는 대신, 측정 오차가 적분되면서 쌓인다.

**드리프트(drift).** IMU 측정을 적분해 속도와 위치를 구하면 센서 바이어스(bias)와 노이즈도 함께 적분돼 오차가 쌓인다. 일정한 가속도계 바이어스가 만드는 위치 오차 항은 $t^2$에 비례한다. 실제 드리프트가 얼마나 빠른지는 센서 등급, 온도 보정, 초기화, 운동, 외부 보정 관측에 따라 크게 달라진다. IMU 등급만 보고 몇 초나 몇 분 뒤의 위치 오차를 정해 말하기는 어렵다.

$$\delta \mathbf{p}(t) \approx \frac{1}{2} \mathbf{b}_a \, t^2 + \frac{1}{\sqrt{3}} \sigma_a \, t^{3/2}$$

여기서 $\mathbf{b}_a$는 가속도계 바이어스, $\sigma_a$는 가속도 노이즈 밀도다. 바이어스가 $0.01\,\text{m/s}^2$만 되어도 10초 후 위치 오차는 0.5m에 달한다.

**절대 기준 부재.** IMU는 각속도와 비력(specific force, 중력을 뺀 가속도)을 잰다. 둘 다 움직임의 변화율이라 적분해도 출발 시점을 기준으로 한 상대 위치와 자세만 나온다. 절대 위치와 절대 방위(heading)는 바깥 기준에서 얻어야 한다. 멈춰 있을 때처럼 가속도계가 중력 방향을 안정적으로 가리키면 롤(roll)과 피치(pitch)는 맞출 수 있다. 요(yaw)는 IMU만으로 알 수 없어서, 자력계나 GNSS 진행 방향(course), 카메라·LiDAR 지도 정합 같은 외부 관측으로 heading 기준을 잡는다.

### GNSS의 한계

위성 항법 시스템(Global Navigation Satellite System, GNSS)은 지구 어디서나 절대 위치를 알려 준다.

**차폐 환경.** 실내, 터널, 도심의 고층 빌딩 사이(urban canyon)에서는 위성 신호가 막히거나, 건물에 반사된 신호가 섞이는 다중경로(multipath) 때문에 수십 미터 오차가 생긴다.

**업데이트 주기.** 수신기는 보통 1–10Hz로 위치를 내보내서 빠른 움직임을 따라가지 못한다.

**정밀도 한계.** 수신기 하나로 하는 단독 측위는 오차가 수 미터다. RTK(Real-Time Kinematic)를 쓰면 센티미터급까지 좋아지는데, 기준국(base station)이 있어야 하고 처음 해가 수렴(convergence)하기까지 시간이 걸린다.

### 센서 한계의 상보성

네 센서의 성질을 표로 모으면 한 센서가 못 하는 일을 다른 센서가 하는 경우가 많다.

| 특성 | 카메라 | LiDAR | IMU | GNSS |
|------|--------|-------|-----|------|
| 절대 위치 | ✗ | ✗ | ✗ | ✓ |
| 상대 이동 (단기) | ✓ | ✓ | ✓ | ✗ |
| 상대 이동 (장기) | △ (드리프트) | △ (드리프트) | ✗ (발산) | ✓ |
| 고주파 모션 캡처 | ✗ | ✗ | ✓ | ✗ |
| 조명 독립 | ✗ | ✓ | ✓ | ✓ |
| 날씨 강건성 | △ | △ | ✓ | ✓ |
| 3D 기하정보 | △ (깊이 모호) | ✓ | ✗ | ✗ |
| 텍스처/의미 정보 | ✓ | ✗ | ✗ | ✗ |
| 실내 동작 | ✓ | ✓ | ✓ | ✗ |
| 비용 | 저 | 고 | 중~저 | 중 |


---

## 1.2 센서 퓨전의 분류

센서들이 서로 어떤 관계로 정보를 보태는지에 따라 complementary, competitive, cooperative로 나눈다.

### Complementary Fusion (상보적 융합)

서로 다른 물리량을 재는 센서가 각자 전체 상태(state)의 다른 부분을 관측한다. 합치면 어느 한 센서로는 관측할 수 없던(unobservable) 상태 성분까지 추정한다.

**대표 예시: 카메라 + IMU (Visual-Inertial Odometry, VIO)**

- 카메라로는 6-DoF 포즈의 상대 변화를 얻지만, 단안이면 스케일을 모르고 빠르게 움직이면 추적이 끊긴다.
- IMU는 가속도와 각속도를 고주파로 재서 카메라 프레임 사이의 빠른 움직임을 메운다. 기기를 충분히 흔들어 가속도가 여러 방향으로 걸리면(motion excitation) 미터 단위로 잰 가속도와 영상 제약을 함께 써서 스케일을 구할 수 있다. 가속도계는 중력 방향도 알려 준다.
- 거꾸로 카메라 관측은 IMU 적분에서 쌓인 드리프트를 바로잡는다.

**대표 예시: GNSS + IMU**

- GNSS는 1–10Hz로 절대 위치를 준다.
- IMU는 수백 Hz로 상대 이동을 잰다.
- 터널에서 GNSS가 끊기면 IMU 적분으로 잠시 항법을 이어 가고, GNSS가 다시 잡히면 그동안 쌓인 IMU 드리프트를 바로잡는다.

### Competitive Fusion (경쟁적 융합)

같은 물리량을 재는 센서를 여러 개 겹쳐(redundancy) 달아서 신뢰성과 정확도를 높인다.

**대표 예시: 다중 카메라 시스템**

- 같은 방향을 보는 두 카메라가 따로 특징점을 추적한다.
- 한 카메라 렌즈가 더러워지거나 고장 나도 시스템은 나머지 카메라로 계속 동작한다.
- 두 추정치를 합치면 하나만 쓸 때보다 분산이 줄어든다.

**통계적 근거.** 독립인 두 관측 $z_1 \sim \mathcal{N}(\mu, \sigma_1^2)$, $z_2 \sim \mathcal{N}(\mu, \sigma_2^2)$를 최적으로 합치면 다음과 같다.

$$\hat{\mu} = \frac{\sigma_2^2 z_1 + \sigma_1^2 z_2}{\sigma_1^2 + \sigma_2^2}, \quad \sigma_{\text{fused}}^2 = \frac{\sigma_1^2 \sigma_2^2}{\sigma_1^2 + \sigma_2^2}$$

두 관측이 편향 없고(unbiased) 서로 독립인 Gaussian이며 분산을 정확히 안다면 $\sigma_{\text{fused}}^2 < \min(\sigma_1^2, \sigma_2^2)$다. 실제로는 두 관측이 상관돼 있거나 분산을 잘못 잡으면 이 부등식이 성립하지 않는다. 이 식은 스칼라 관측 하나로 하는 Kalman update와 같다.

**대표 예시: 다중 LiDAR 시스템**

차량 둘레에 LiDAR를 여러 대 달면 시야(field of view)가 넓어지고 겹치는 구역은 두 번 관측된다. 몇 대를 어디에 달지는 필요한 측정 거리, 가림(occlusion), 비용, 고장 대비에 따라 정한다.

### Cooperative Fusion (협력적 융합)

센서들의 원시(raw) 데이터를 합쳐 어느 센서 하나로도 얻지 못하는 정보를 만든다.

**대표 예시: 스테레오 비전 (Stereo Vision)**

- 좌우 카메라 영상에서 같은 점이 옮겨 간 거리, 곧 시차(disparity)를 계산한다.
- 시차와 두 카메라 사이 거리(기선)로 깊이를 구한다.
- 카메라 한 대로는 알 수 없던 깊이를 두 대가 함께 재는 셈이다.

**대표 예시: 카메라 + LiDAR → 컬러 포인트 클라우드**

- LiDAR 점을 카메라 영상에 투영해 3D 점마다 색을 입힌다.
- 그렇게 만든 컬러 포인트 클라우드(colored point cloud)에는 LiDAR가 잰 형상과 카메라가 찍은 텍스처가 함께 담긴다.
- [R3LIVE (Lin et al., 2022)](https://arxiv.org/abs/2109.07982) 같은 시스템은 이 계산을 실시간으로 한다.

**대표 예시: Radar + Camera → 악천후 인지**

- 레이더가 도플러(Doppler) 효과로 잰 상대 속도에 카메라 영상을 더해, 안개나 비 속에서도 움직이는 물체의 속도와 종류를 함께 알아낸다.

한 시스템에 세 범주가 다 들어가기도 한다. 자율주행 차량이라면 카메라와 LiDAR로 컬러 포인트 클라우드를 만들고(cooperative), 카메라와 IMU로 VIO를 돌리고(complementary), LiDAR 여러 대로 같은 구역을 겹쳐 본다(competitive).

---

## 1.3 결합 수준에 따른 분류: Loosely vs Tightly vs Ultra-tightly Coupled

센서 데이터를 어느 단계에서 합치느냐로도 나눈다. 합치는 단계에 따라 정확도, 구현 난도, 강건성이 달라진다.

### Loosely Coupled (느슨한 결합)

센서마다 서브시스템이 따로 추정치를 내고, 그 결과(output)를 위 단계에서 합친다.

**구조:**
```
센서A → [서브시스템A] → 추정치A ─┐
                                   ├─→ [Fusion] → 최종 추정
센서B → [서브시스템B] → 추정치B ─┘
```

**대표 예시: 독립 VO + 독립 LiDAR Odometry → EKF Fusion**

- Visual Odometry가 독립적으로 포즈를 출력한다.
- LiDAR Odometry가 독립적으로 포즈를 출력한다.
- 그 위의 EKF가 두 포즈를 합쳐 최종 추정치를 낸다.

**장점:**
- 모듈성: 각 서브시스템을 독립적으로 개발, 테스트, 교체할 수 있다.
- 단순성: 퓨전 단계 설계가 비교적 간단하다.
- 부분 실패 대응: 한 서브시스템이 멈춰도 퓨전 단계는 다른 서브시스템의 출력으로 계속 동작한다.

**단점:**
- 정보 손실: 서브시스템이 관측을 요약(compress)해 내보내므로 특징점별 불확실성 같은 원시 관측의 세부가 사라진다.
- 상관관계 무시: 두 서브시스템이 같은 IMU 데이터처럼 공통 관측을 쓰면 두 추정치는 서로 상관된다. 이를 무시하고 합치면 같은 정보를 두 번 세는 셈이라(double counting) 추정기가 자기 추정을 과신(overconfidence)한다.
- 최적성 저하: 중간 출력이 충분통계량이 아니거나 교차 상관(cross-correlation)을 함께 넘기지 않으면, 원시 관측을 한 추정기에서 합칠 때보다 정보를 잃는다.

### Tightly Coupled (긴밀한 결합)

모든 센서의 원시 관측(raw measurement)을 추정기 하나(single estimator)에 바로 넣는다.

**구조:**
```
센서A → raw 관측A ─┐
                     ├─→ [Single Estimator] → 최종 추정
센서B → raw 관측B ─┘
```

**대표 예시: [VINS-Mono (Qin et al., 2018)](https://arxiv.org/abs/1708.03852)**

- 카메라의 특징점 관측과 IMU의 가속도·각속도 측정을 sliding window 비선형 최적화 하나에 함께 넣는다.
- 비용 함수에는 재투영 오차(reprojection error)와 IMU 사전적분 잔차(preintegration residual)가 함께 들어가고, 최적화는 그 합을 줄인다.

$$\min_{\mathcal{X}} \left\{ \sum_{(i,j) \in \mathcal{B}} \| \mathbf{r}_{\text{IMU}}(\mathbf{x}_i, \mathbf{x}_j) \|^2_{\mathbf{P}_{ij}} + \sum_{(i,l) \in \mathcal{C}} \| \mathbf{r}_{\text{cam}}(\mathbf{x}_i, \mathbf{f}_l) \|^2_{\mathbf{\Sigma}_l} \right\}$$

여기서 $\mathbf{r}_{\text{IMU}}$는 IMU 사전적분 잔차, $\mathbf{r}_{\text{cam}}$은 시각 재투영 잔차, $\mathcal{X}$는 포즈, 속도, 바이어스, 랜드마크를 모은 전체 상태다.

**대표 예시: [FAST-LIO2 (Xu et al., 2022)](https://arxiv.org/abs/2107.06829)**

- LiDAR 점 하나하나와 IMU 측정을 반복 오차 상태 칼만 필터(Iterated Error-State Kalman Filter) 하나에 바로 넣는다. 점을 특징으로 추리지 않고 직접 정합한다.

**대표 예시: [LIO-SAM (Shan et al., 2020)](https://arxiv.org/abs/2007.00258)**

- LiDAR 특징점, IMU 사전적분, GNSS 위치 관측을 factor graph 하나에 넣어 함께 최적화한다.

**장점:**
- 정보 보존: 관측별 잔차(residual)와 공분산을 그대로 쓰므로 중간 포즈만 합칠 때보다 정보를 덜 잃는다. 실제 품질은 모델, 선형화, outlier 처리에 달렸다.
- 교차 추정: 외부 파라미터(extrinsic), 시간 오프셋, 바이어스를 상태에 넣고 운동이 충분히 다양하면, 카메라 관측으로 IMU 바이어스를 추정하는 식으로 센서끼리 서로 보정한다.
- 열화 대응: 한 센서의 관측이 줄어도 다른 센서가 그 상태를 충분히 보면 추정을 이어 간다. 그러려면 고장 감지(failure detection)와 관측 가능성 분석이 따로 있어야 한다.

**단점:**
- 복잡도: 추정기 하나에 모든 센서의 관측 모델을 넣어야 해서 설계와 디버깅이 어렵다.
- 계산 비용: 상태 벡터가 커지고 관측이 많아져 계산량이 늘어난다.
- 센서 의존성: 센서 하나를 빼거나 바꾸려면 추정기 전체를 고쳐야 한다.

### Ultra-tightly Coupled (초긴밀 결합)

센서 내부 신호(signal level)에서 합친다. 이 용어는 주로 GNSS/INS 통합에서 쓴다.

**대표 예시: GNSS/INS Ultra-tight Integration**

- 보통의 tightly coupled에서는 GNSS 수신기가 내놓은 의사거리(pseudorange)를 항법 필터에 넣는다.
- Ultra-tight에서는 INS가 예측한 속도를 GNSS 수신기 안의 코드·반송파 추적 루프(tracking loop)로 되돌려 넣는다.
- 차량 운동을 INS가 미리 알려 주므로 추적 루프의 대역폭을 좁힐 수 있다. 대역폭이 좁으면 노이즈가 덜 들어와, 간섭이 심하거나 신호가 약해도 위성을 놓치지 않는다.

**비전에서 비슷한 예**

비전-관성 시스템에서 ultra-tight coupling에 가까운 것은 IMU 예측으로 특징점을 찾을 영역을 좁히거나 IMU 데이터로 모션 블러(motion blur)를 되돌리는 프런트엔드다. VINS-Mono는 KLT optical flow로 특징점을 추적하고, IMU 측정은 사전적분해 백엔드와 초기화에만 쓴다. 그래서 VINS-Mono는 이 범주에 넣지 않는다.

### 결합 수준 비교

| 특성 | Loosely Coupled | Tightly Coupled | Ultra-tightly Coupled |
|------|----------------|-----------------|----------------------|
| 융합 레벨 | 결과(output) | 관측(measurement) | 신호(signal) |
| 정보 활용도 | 낮음 | 높음 | 최고 |
| 구현 복잡도 | 낮음 | 중간~높음 | 매우 높음 |
| 모듈성 | 높음 | 낮음 | 매우 낮음 |
| 부분 실패 대응 | 용이 | 설계 필요 | 어려움 |
| 대표 시스템 | 독립 VO + LO → EKF | VINS-Mono, LIO-SAM, FAST-LIO2, ORB-SLAM3 | GNSS/INS deep integration |

VINS-Mono(카메라·IMU), FAST-LIO2(LiDAR·IMU), LIO-SAM(LiDAR·IMU·GNSS)처럼 이 가이드에서 다루는 공개 VIO·LIO 시스템은 tightly coupled로 짠다. Ultra-tightly coupled는 센서 내부 신호나 추적 루프에 손댈 수 있어야 해서 쓰는 곳이 좁다. [FAST-LIVO2 (Zheng et al., 2025)](https://arxiv.org/abs/2408.14035)는 LiDAR, IMU, 카메라를 한 추정기에 묶었고, 논문의 데이터셋과 비교 조건에서 정확도와 처리 속도가 나아졌다고 보고한다.

---

## 1.4 Classical vs Learning-based: 딥러닝이 바꾼 것과 바꾸지 못한 것

센서 퓨전은 수십 년 동안 확률 추정 이론(Kalman Filter, Factor Graph)과 기하 방법(epipolar geometry, ICP)으로 풀어 왔다. 2010년대 중반 딥러닝이 컴퓨터 비전 전반에 퍼지면서 센서 퓨전에도 학습 기반(learning-based) 방법이 들어왔다. 얼마나 쓰이고 얼마나 잘 되는지는 영역마다 다르다.

### 딥러닝이 바꾼 것

**특징점 추출과 매칭.** 예전에는 사람이 규칙을 정한(handcrafted) SIFT, ORB 같은 기술자(descriptor)를 썼다. [SuperPoint (DeTone et al., 2018)](https://arxiv.org/abs/1712.07629)는 자기 지도 학습(self-supervised learning)으로 키포인트 검출과 기술자 계산을 한 네트워크에서 하고, 조명과 시점이 바뀌어도 잘 맞는다. [SuperGlue (Sarlin et al., 2020)](https://arxiv.org/abs/1911.11763)는 그래프 뉴럴 네트워크(GNN)와 어텐션으로 두 영상의 특징점을 짝지어, 기존 기술자의 최근접 이웃 매칭보다 잘못 짝짓는 비율이 낮았다. [LoFTR (Sun et al., 2021)](https://arxiv.org/abs/2104.00680)와 [RoMa (Edstedt et al., 2024)](https://arxiv.org/abs/2305.15404) 같은 detector-free 방법은 키포인트를 뽑지 않고 픽셀 단위로 촘촘하게 대응(dense correspondence)을 찾으므로 텍스처가 적은 곳에서도 짝을 찾는다.

학습 기반 matcher는 여러 공개 benchmark에서 handcrafted 방법보다 점수가 높다. 실제로 쓸 만한지는 지연 시간, 메모리, 학습 데이터와 다른 환경(domain shift)을 함께 따져야 하고, 올릴 임베디드 하드웨어와 운용 환경에서 직접 재 봐야 안다.

**장소 인식(Place Recognition).** [NetVLAD (Arandjelović et al., 2016)](https://arxiv.org/abs/1511.07247)는 CNN feature를 학습 가능한 VLAD 층으로 모아 장소 기술자(place descriptor)를 학습했고, 논문의 benchmark에서 기존 영상 검색 방법보다 recall이 높았다. [AnyLoc (Keetha et al., RA-L 2024)](https://arxiv.org/abs/2308.00688)은 따로 학습하지 않은 DINOv2 feature를 비지도 VLAD로 모아 실내, 항공, 수중 같은 여러 환경에서 평가했다. 이런 방법은 DBoW2와 쓰는 feature도 계산량도 달라서, 같은 데이터셋과 같은 검증 절차에서 견줘야 공정하다.

**단안 깊이 추정.** 영상 한 장의 투영 기하만으로는 배율을 모르는 상대 깊이밖에 얻지 못한다. [Depth Anything (Yang et al., 2024)](https://arxiv.org/abs/2401.10891)은 라벨 없는 영상까지 대량으로 써서 학습한 단안 상대 깊이 모델이다. 후속작 [Depth Anything V2 (Yang et al., 2024)](https://arxiv.org/abs/2406.09414)는 합성 영상으로 교사 모델을 학습하고 실제 영상에 가짜 라벨(pseudo-label)을 대량으로 달아 더 세밀한 깊이를 낸다. [Metric3D v2 (Hu et al., 2024)](https://arxiv.org/abs/2404.15506)는 처음 보는 카메라에서도 추가 학습 없이(zero-shot) 미터 단위 깊이를 추정한다. 센서 퓨전에 쓰려면 실제 카메라와 환경에서 스케일이 치우치지 않는지, 불확실성이 얼마인지 따로 재 봐야 한다.

**맵 표현.** NeRF는 장면을 신경망 하나에 담는다. 3D Gaussian Splatting은 장면을 수많은 3D 가우시안으로 나타내고 그 위치와 모양을 최적화한다. NeRF-SLAM이나 Gaussian Splatting SLAM으로 만든 지도는 점 지도(point map)나 복셀 격자(voxel grid)와 달리 사진처럼 다시 렌더링할 수 있다.

**이벤트 카메라(Event Camera).** 딥러닝과는 별개로 센서도 새로 나왔다. 뉴로모픽 비전 센서라고도 부르는 이벤트 카메라는 픽셀마다 밝기가 바뀌는 순간을 따로따로 기록한다. 시간 분해능이 마이크로초 단위이고 다이내믹 레인지가 넓다. [이벤트 카메라 서베이 (Chakravarthi et al., 2025)](https://arxiv.org/abs/2408.13627)에 정리된 응용에는 이벤트 기반 VO·SLAM과 이벤트를 프레임 카메라 영상과 합치는 융합이 있으며, 주로 빠른 움직임과 어두운 환경을 겨냥한다.

### 딥러닝이 바꾸지 못한 것

**상태 추정(State Estimation) 백엔드.** 칼만 필터와 factor graph 최적화 같은 확률 추정기는 지금도 이 가이드에서 다루는 시스템 대부분의 백엔드다.

1. **불확실성 계산**: 칼만 필터와 factor graph는 가정한 모델과 공분산에서 불확실성을 직접 계산한다. 모델이 틀리거나 선형화·상관관계를 잘못 다루면 이 공분산도 믿을 수 없게 되고, 학습한 관측을 넣을 때는 그 불확실성을 따로 보정해야 한다.
2. **물리 모델과 제약**: 상태 전이와 잔차에 동역학과 기구학을 넣는다. 물리적으로 가능한 추정만 남기려면 하드 제약을 최적화나 매개변수화에 따로 적어야 한다.
3. **학습 데이터**: 모델 기반 추정기는 대규모 지도 학습 없이 돌아가지만, 센서 캘리브레이션, 노이즈 튜닝, 검증 데이터는 여전히 있어야 한다. 학습한 부품은 학습 데이터 분포와 라벨 조건까지 관리해야 한다.
4. **일반화**: 학습 기반 오도메트리는 학습 때와 다른 환경에서 성능이 떨어지기 쉽다. 기하 방법도 텍스처, 조명, 움직이는 물체, 센서 노이즈에 영향을 받지만 재학습 없이 새 환경에 쓴다.

**LiDAR 오도메트리.** [LOAM (Zhang & Singh, 2014)](https://frc.ri.cmu.edu/~zhangji/publications/RSS_2014.pdf) 이후로도 LiDAR 오도메트리는 주로 기하 방법을 쓴다. ICP, GICP, NDT 같은 포인트 클라우드 정합은 언제 실패하는지가 비교적 잘 밝혀져 있고, 재학습 없이 새 환경에 쓴다. 학습 기반 LiDAR 오도메트리도 특정 데이터셋에서는 좋은 결과를 내지만, 환경이나 센서가 바뀌어도 통하는지는 따로 확인해야 한다.

**캘리브레이션.** 카메라 내부 파라미터는 지금도 [Zhang (2000)](https://doi.org/10.1109/34.888718)의 평면 체커보드 방법으로 구한다. OpenCV의 `calibrateCamera`도 이 방법을 바탕으로 한다. 타깃 없이(targetless) 하거나 학습으로 하는 방법도 연구되지만, 정밀도는 센서 구성과 데이터를 모은 조건에서 타깃 기반 방법과 나란히 재 봐야 판단할 수 있다.

### Hybrid 접근

프런트엔드(frontend)는 학습 기반으로, 백엔드(backend)는 고전 추정기로 짜는 하이브리드 구조도 있다.

```
[학습 기반 프런트엔드]           [고전적 백엔드]
 SuperPoint/SuperGlue          Factor Graph / EKF
 Mono Depth Estimation    →    Nonlinear Optimization
 Semantic Segmentation         Kalman Filtering
 Place Recognition              Pose Graph SLAM
```

- 프런트엔드에서는 신경망이 원시 센서 데이터에서 특징점, 깊이, 의미 라벨(semantic label)을 뽑는다.
- 백엔드에서는 기하·확률 추정기가 이 결과를 받아 시간에 걸쳐 앞뒤가 맞는 상태를 추정한다.

[DROID-SLAM (Teed & Deng, 2021)](https://arxiv.org/abs/2108.10869)은 학습한 네트워크로 특징을 뽑고 대응을 찾은 뒤, 포즈는 미분 가능한(differentiable) 번들 조정(Bundle Adjustment) 층에서 푼다.

### 영역별 고전 방법과 학습 기반 방법



| 영역 | Classical | Learning-based | 실무에서 쓰는 방식 |
|------|-----------|---------------|----------|
| Feature matching | SIFT/ORB + BF/FLANN | SuperPoint+SuperGlue → LoFTR → RoMa | Hybrid |
| Visual odometry | Feature-based (ORB) / Direct (DSO) | DROID-SLAM, DPV-SLAM | 정확도·compute·domain 조건별 선택 |
| LiDAR odometry | ICP/LOAM | DeepLO 계열 | Geometry 중심, learned component 병용 |
| Place recognition | BoW/VLAD | NetVLAD → AnyLoc | 후보 검색은 혼용, geometry로 검증 |
| Depth estimation | Stereo matching | Mono depth (Depth Anything) | Metric stereo와 learned prior의 역할 구분 |
| Calibration | Target-based | Targetless + learning | Target·targetless를 요구조건별 선택 |
| Map representation | Occupancy/TSDF | NeRF / 3DGS | 공존 |
| State estimation backend | KF / Factor Graph | End-to-end 시도 | Model-based estimator와 learned observation 결합 |

학습한 feature는 주로 인식 모듈에서 쓰고, 기하와 불확실성을 합치는 부분은 주로 모델 기반 추정기가 맡는다. 모듈마다 입력, 실패 양상, 검증 지표를 정할 때도 이 구분을 따른다.

---

## 1.5 범위와 구성

### robotics-practice와의 관계

robotics-practice는 Spatial AI 전반을 넓게 다루는 입문서다. 이 가이드는 그중 센서 퓨전, 위치 추정(localization), 장소 검색(retrieval)을 깊게 다룬다.

- robotics-practice에서 소개한 EKF와 PF를, 여기서는 칼만 필터 유도부터 ESKF, IMU 사전적분, factor graph 최적화까지 장을 따로 잡아 설명한다.
- 센서마다 노이즈 모델과 관측 방정식을 수식으로 유도한다.
- 겹치는 기초 내용은 robotics-practice를 참조한다.

### 가이드 구성

1. **Ch.1 — Introduction**: 센서 퓨전의 동기와 분류
2. **Ch.2 — Sensor Modeling**: 각 센서의 관측 모델과 노이즈 특성 (수식 중심)
3. **Ch.3 — Calibration**: 다양한 센서 조합의 캘리브레이션 이론과 실전
4. **Ch.4 — State Estimation 이론**: 베이지안 필터링, 칼만 필터 계열, 파티클 필터, factor graph
5. **Ch.5 — Feature Matching & Correspondence**: Harris·SIFT에서 RoMa·VGGT까지 특징 매칭과 대응점 탐색 방법의 변화
6. **Ch.6 — Visual Odometry & VIO**: VO/VIO 시스템의 내부 구조
7. **Ch.7 — LiDAR Odometry & LIO**: LiDAR 기반 오도메트리와 LiDAR-관성 퓨전
8. **Ch.8 — Multi-Sensor Fusion 아키텍처**: 여러 센서를 한 시스템에 묶는 설계
9. **Ch.9 — Place Recognition & Retrieval**: BoW에서 Foundation Model까지
10. **Ch.10 — Loop Closure & Global Optimization**: loop closure와 전역 최적화
11. **Ch.11 — Spatial Representations**: 점 지도에서 Neural Map과 semantic map까지
12. **Ch.12 — 실전 시스템 & 벤치마크**: 실제 응용과 평가
13. **Ch.13 — Frontiers**: 최신 동향과 풀리지 않은 문제

### 대상 독자

이 가이드는 다음 배경을 갖춘 로보틱스 입문자를 위해 썼다.

- **선형대수**: 행렬 연산, 고유값 분해, SVD
- **확률론**: 확률 분포, 조건부 확률, 베이즈 정리, 가우시안 분포
- **기초 최적화**: 최소자승법, 그래디언트 디센트의 개념
- **Python**: numpy, scipy 기반 코드를 읽고 실행할 수 있는 수준
