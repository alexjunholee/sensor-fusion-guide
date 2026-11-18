# 4장 상태 추정

관측 모델로 센서 측정의 성질을 파악하고 캘리브레이션으로 센서 사이의 기하·시간 어긋남을 맞췄다면(→ 2장, 3장), 이제 그 데이터를 하나로 엮어 로봇의 위치·자세·속도·바이어스 같은 숨은 상태를 추정할 차례다. 실제 주행 환경에서는 어떤 센서도 참값을 그대로 건네주지 않는다. IMU는 높은 주기로 움직임을 포착하지만 적분할수록 오차가 쌓이고, 카메라나 LiDAR는 기하학적 단서를 주지만 주기가 느리거나 환경에 따라 관측이 끊긴다. 상태 추정(state estimation)은 이처럼 특성과 주기가 다른 불완전한 관측을 운동 모델과 확률적으로 결합해, 가장 그럴듯한 상태와 그 불확실성까지 함께 추적하는 과정이다. Odometry와 다중 센서 융합 시스템은 이 문제를 칼만 필터 계열의 재귀 필터링, 파티클 필터, 또는 factor graph와 IMU 사전적분(preintegration)을 결합한 비선형 최적화로 풀어낸다.

---

## 4.1 Bayesian Filtering Framework

칼만 필터부터 파티클 필터까지 모든 재귀적 상태 추정기는 공통된 확률론적 뼈대 위에서 움직인다. 핵심 발상은 단순하다. 지금까지 모은 불완전한 단서들을 바탕으로 현재 상태가 어디쯤 있을지 확률 분포로 나타내고, 새 움직임이나 관측이 생길 때마다 그 분포를 조금씩 다듬어 나간다.

### 4.1.1 상태 추정 문제의 정의

확률론적 관점에서 상태 추정은 조건부 확률을 구하는 문제로 정리된다. 즉, 시각 $1$부터 $k$까지 들어온 모든 관측 $\mathbf{z}_{1:k}$와 제어 입력 $\mathbf{u}_{1:k}$를 알고 있을 때, 현재 상태 $\mathbf{x}_k$의 사후 확률(posterior) 분포를 구하고자 한다.

$$p(\mathbf{x}_k \mid \mathbf{z}_{1:k}, \mathbf{u}_{1:k})$$

이 식에서 각 변수가 가리키는 대상은 다음과 같다.
- $\mathbf{x}_k \in \mathbb{R}^n$: 시각 $k$의 상태 벡터 (위치, 속도, 자세, 바이어스 등)
- $\mathbf{z}_k \in \mathbb{R}^m$: 시각 $k$의 관측 벡터 (카메라, LiDAR, IMU 등)
- $\mathbf{u}_k \in \mathbb{R}^l$: 시각 $k$의 제어 입력

시간의 흐름에 따라 이 사후 확률을 계속 추적하려면 두 가지 확률 모델이 뒷받침되어야 한다.

**운동 모델(Motion Model, Process Model)**은 제어 입력 $\mathbf{u}_k$가 주어졌을 때 직전 상태 $\mathbf{x}_{k-1}$이 현재 상태 $\mathbf{x}_k$로 어떻게 옮겨가는지를 기술한다.

$$\mathbf{x}_k = f(\mathbf{x}_{k-1}, \mathbf{u}_k) + \mathbf{w}_k, \quad \mathbf{w}_k \sim \mathcal{N}(\mathbf{0}, \mathbf{Q}_k)$$

$$p(\mathbf{x}_k \mid \mathbf{x}_{k-1}, \mathbf{u}_k)$$

여기서 프로세스 노이즈 $\mathbf{w}_k$는 단순화한 운동 모델이 실제 물리 움직임을 온전히 담아내지 못해 생기는 불확실성을 뜻한다.

**관측 모델(Observation Model, Measurement Model)**은 반대로 현재 상태 $\mathbf{x}_k$를 안다고 가정할 때 센서에서 어떤 관측값 $\mathbf{z}_k$가 나올지, 곧 우도(likelihood)를 정의한다.

$$\mathbf{z}_k = h(\mathbf{x}_k) + \mathbf{v}_k, \quad \mathbf{v}_k \sim \mathcal{N}(\mathbf{0}, \mathbf{R}_k)$$

$$p(\mathbf{z}_k \mid \mathbf{x}_k)$$

관측 노이즈 $\mathbf{v}_k$는 센서 측정 자체에 섞여 들어오는 오차를 반영한다. 필터는 이 운동 모델과 관측 모델을 번갈아 적용하며 상태에 대한 확률 분포를 꾸준히 갱신해 나가는데, 이렇게 유지하는 확률 분포를 믿음(belief)이라고도 부른다.

### 4.1.2 Markov 가정과 Recursive Estimation

시간이 지날수록 관측과 제어 입력의 이력은 끝없이 불어난다. 매 시각 처음부터 전체 이력 $\mathbf{z}_{1:k}, \mathbf{u}_{1:k}$를 다시 훑어 계산하면 곧 메모리와 연산량이 한계에 부딪힌다. 과거의 모든 정보를 직전 상태 하나에 압축해 두고 바로 다음 단계로 넘어가려면 두 가지 Markov 가정이 필요하다.

1. **1차 Markov 프로세스**: 직전 상태 $\mathbf{x}_{k-1}$과 현재 제어 $\mathbf{u}_k$만 알면 현재 상태 $\mathbf{x}_k$의 분포가 정해지며, 그보다 앞선 상태나 과거 관측은 추가로 영향을 주지 않는다.
$$p(\mathbf{x}_k \mid \mathbf{x}_{0:k-1}, \mathbf{u}_{1:k}, \mathbf{z}_{1:k-1}) = p(\mathbf{x}_k \mid \mathbf{x}_{k-1}, \mathbf{u}_k)$$

2. **조건부 관측 독립**: 현재 상태 $\mathbf{x}_k$가 주어지면 현재 관측 $\mathbf{z}_k$의 분포는 과거의 다른 상태나 관측과 독립이다.
$$p(\mathbf{z}_k \mid \mathbf{x}_{0:k}, \mathbf{u}_{1:k}, \mathbf{z}_{1:k-1}) = p(\mathbf{z}_k \mid \mathbf{x}_k)$$

이 두 가정이 성립하면 지나간 관측 이력을 통째로 들고 있을 필요 없이, 직전 시점의 추정 분포와 방금 들어온 입력·관측만으로 현재 분포를 바로 계산할 수 있다. 이 구조를 재귀적 추정(recursive estimation)이라 부른다.

### 4.1.3 Prediction-Update Cycle

운동 모델과 관측 모델, 그리고 Markov 가정이 갖춰지면 Bayesian filter는 예측(prediction)과 갱신(update) 두 단계를 번갈아 도는 순환 구조로 작동한다.

**예측 단계(Prediction Step)**에서는 이전 시각의 사후 확률에 운동 모델을 적용해, 새 관측을 보기 직전의 사전 확률(prior)을 구한다. 움직이는 동안에는 프로세스 노이즈가 더해지므로 분포가 이전보다 넓게 퍼진다.

$$\boxed{p(\mathbf{x}_k \mid \mathbf{z}_{1:k-1}, \mathbf{u}_{1:k}) = \int p(\mathbf{x}_k \mid \mathbf{x}_{k-1}, \mathbf{u}_k) \, p(\mathbf{x}_{k-1} \mid \mathbf{z}_{1:k-1}, \mathbf{u}_{1:k-1}) \, d\mathbf{x}_{k-1}}$$

이 식이 바로 Chapman-Kolmogorov 방정식이다. 직전 상태 $\mathbf{x}_{k-1}$이 가질 수 있는 모든 값에 대해 상태 전이 확률을 가중 평균한 것으로, 각 항의 역할은 다음과 같다.
- $p(\mathbf{x}_{k-1} \mid \mathbf{z}_{1:k-1})$: 시각 $k-1$의 사후 확률 (이전 단계에서 얻은 믿음)
- $p(\mathbf{x}_k \mid \mathbf{x}_{k-1}, \mathbf{u}_k)$: 상태 전이 확률 (운동 모델)
- 적분 결과는 이전 믿음을 운동 모델로 밀어낸 현재 시각의 예측 분포가 된다

**갱신 단계(Update Step)**에서는 새 관측 $\mathbf{z}_k$가 도착했을 때 Bayes 정리를 적용해 예측 분포를 사후 확률로 좁힌다.

$$\boxed{p(\mathbf{x}_k \mid \mathbf{z}_{1:k}, \mathbf{u}_{1:k}) = \frac{p(\mathbf{z}_k \mid \mathbf{x}_k) \, p(\mathbf{x}_k \mid \mathbf{z}_{1:k-1}, \mathbf{u}_{1:k})}{p(\mathbf{z}_k \mid \mathbf{z}_{1:k-1})}}$$

분자와 분모의 각 요소는 다음을 뜻한다.
- 분자의 $p(\mathbf{z}_k \mid \mathbf{x}_k)$: 관측 모델이 주는 likelihood
- 분자의 $p(\mathbf{x}_k \mid \mathbf{z}_{1:k-1})$: 예측 단계에서 구한 prior
- 분모의 $p(\mathbf{z}_k \mid \mathbf{z}_{1:k-1}) = \int p(\mathbf{z}_k \mid \mathbf{x}_k) p(\mathbf{x}_k \mid \mathbf{z}_{1:k-1}) d\mathbf{x}_k$: 전체 적분값이 1이 되도록 맞추는 정규화 상수 (evidence)

직관적으로 보면, 예측 단계가 "지금까지 정보로 보아 로봇이 대략 이 부근으로 움직였을 것"이라며 후보 영역을 펼치는 과정이라면, 갱신 단계는 "방금 들어온 센서 측정과 잘 맞아떨어지는 후보만 남기자"며 믿음을 다시 다듬는 과정이다.

### 4.1.4 왜 Closed-Form이 안 되는가

이론적으로는 위의 예측과 갱신 두 방정식만으로 상태 추정 문제가 완결되지만, 실제 로봇 시스템에서 Chapman-Kolmogorov 적분 $\int p(\mathbf{x}_k \mid \mathbf{x}_{k-1}) p(\mathbf{x}_{k-1} \mid \mathbf{z}_{1:k-1}) d\mathbf{x}_{k-1}$이 닫힌 식(closed-form)으로 풀리는 일은 거의 없다. 현실의 문제가 막히는 지점은 크게 세 가지다.

1. **비선형 운동·관측 모델**: $f(\cdot)$이나 $h(\cdot)$이 비선형이면 처음에 가우시안 분포에서 출발하더라도 변환을 거친 뒤에는 더 이상 가우시안 형태를 유지하지 못하고, 깔끔한 수식 하나로 적기 어려운 모양으로 찌그러진다.

2. **고차원 상태 공간**: 해석적 적분 대신 상태 공간을 격자로 쪼개 수치 적분을 시도하면, 격자 점의 수와 계산량이 상태 차원에 따라 지수적으로 불어난다(차원의 저주).

3. **다봉(multimodal) 분포**: 생김새가 똑같은 복도가 여러 개 놓인 건물에서 로봇이 자신의 초기 위치를 찾는 상황(global localization)처럼 가능한 위치 후보가 여러 군데로 갈리면, 사후 분포 역시 봉우리가 여럿 솟은 복잡한 형태를 띤다.

이 때문에 실전 상태 추정기는 확률 분포나 비선형 모델을 저마다의 방식으로 근사(approximation)한다. 어떤 가정을 두고 어떻게 근사하는지에 따라 아래 표처럼 서로 다른 필터 계열로 갈라진다.

| 근사 방법 | 필터 | 분포 표현 | 적용 조건 |
|----------|------|---------|----------|
| 선형-가우시안 가정 | Kalman Filter | $\mathcal{N}(\hat{\mathbf{x}}, \mathbf{P})$ | 선형 시스템 |
| 1차 선형화 + 가우시안 | EKF, ESKF | $\mathcal{N}(\hat{\mathbf{x}}, \mathbf{P})$ | 약한 비선형 |
| Sigma point 변환 | UKF | $\mathcal{N}(\hat{\mathbf{x}}, \mathbf{P})$ | 중간 비선형 |
| 샘플 기반 | Particle Filter | $\{(\mathbf{x}^{(i)}, w^{(i)})\}_{i=1}^N$ | 강한 비선형, 다봉 |
| 반복 선형화 | IEKF | $\mathcal{N}(\hat{\mathbf{x}}, \mathbf{P})$ | 강한 비선형 |

가장 먼저 살펴볼 것은 이 근사 방법들 가운데 선형-가우시안 가정에서 정확한 닫힌 해를 내놓는 Kalman Filter와, 이를 비선형·회전 시스템으로 넓힌 변형들이다.

---

## 4.2 Kalman Filter 계열

상태 공간이 연속이고 분포를 하나의 가우시안 $\mathcal{N}(\hat{\mathbf{x}}, \mathbf{P})$로 근사할 수 있다면, 방대한 적분 대신 평균 벡터와 공분산 행렬을 갱신하는 행렬 연산만으로 Bayesian filter를 굴릴 수 있다. 이 절에서는 선형 시스템에서 정확한 해를 내놓는 기본 Kalman Filter에서 출발해, 비선형성과 3D 회전을 다루기 위해 갈라져 나온 EKF, ESKF, UKF, IEKF를 차례로 살펴본다.

### 4.2.1 Kalman Filter: 유도와 최적성

운동 모델과 관측 모델이 모두 선형이고, 프로세스 노이즈·관측 노이즈·초기 상태 분포가 전부 가우시안이라면 Bayesian filter의 적분은 근사 없이 닫힌 식으로 풀린다. 그 정확한 해가 바로 Kalman Filter(KF)다. Rudolf Kalman이 1960년 논문 ["A New Approach to Linear Filtering and Prediction Problems"](https://doi.org/10.1115/1.3662552)에서 이 정식화를 발표한 뒤, NASA Ames의 Stanley Schmidt 팀이 Apollo 우주선의 궤도 추정에 적용하면서 실제 항법 시스템의 핵심 도구로 자리 잡았다.

#### 선형 시스템 모델

선형 시스템에서는 상태 전이와 관측이 모두 행렬 곱과 가우시안 잡음의 합으로 표현된다.

$$\mathbf{x}_k = \mathbf{F}_k \mathbf{x}_{k-1} + \mathbf{B}_k \mathbf{u}_k + \mathbf{w}_k, \quad \mathbf{w}_k \sim \mathcal{N}(\mathbf{0}, \mathbf{Q}_k)$$
$$\mathbf{z}_k = \mathbf{H}_k \mathbf{x}_k + \mathbf{v}_k, \quad \mathbf{v}_k \sim \mathcal{N}(\mathbf{0}, \mathbf{R}_k)$$

각 행렬과 벡터의 역할은 다음과 같다.
- $\mathbf{F}_k \in \mathbb{R}^{n \times n}$: 상태 전이 행렬 (state transition matrix)
- $\mathbf{B}_k \in \mathbb{R}^{n \times l}$: 제어 입력 행렬
- $\mathbf{H}_k \in \mathbb{R}^{m \times n}$: 관측 행렬
- $\mathbf{Q}_k \in \mathbb{R}^{n \times n}$: 프로세스 노이즈 공분산 (대칭 양의 준정부호 행렬)
- $\mathbf{R}_k \in \mathbb{R}^{m \times m}$: 관측 노이즈 공분산 (대칭 양의 정부호 행렬)

가우시안 확률 변수에 선형 변환을 가하고 독립인 가우시안을 더하면 결과 역시 언제나 가우시안이 된다. 덕분에 사후 분포의 모양이 흐트러지지 않으며, 필터는 매 시각 평균과 공분산 두 통계량만 추적하면 충분하다.

#### MMSE 유도 — 왜 Kalman Filter가 최적인가

Kalman Filter는 평균 제곱 오차 최소(Minimum Mean Square Error, MMSE) 기준에서 최적 추정량이다. 참 상태와 추정치 사이의 오차 제곱 기댓값을 가장 작게 만드는 값 $\hat{\mathbf{x}}_k$를 찾는다는 뜻이다.

$$\hat{\mathbf{x}}_k = \arg\min_{\hat{\mathbf{x}}} \mathbb{E}[\|\mathbf{x}_k - \hat{\mathbf{x}}\|^2 \mid \mathbf{z}_{1:k}]$$

이 목적 함수를 최소화하는 추정량은 수학적으로 조건부 기댓값 $\hat{\mathbf{x}}_k = \mathbb{E}[\mathbf{x}_k \mid \mathbf{z}_{1:k}]$과 같으며, 선형-가우시안 시스템에서는 이 조건부 기댓값이 곧바로 Kalman Filter의 상태 갱신식으로 귀결된다.

**Bayesian 관점의 유도**: 시각 $k-1$까지의 관측을 반영한 사후 분포가 가우시안이라고 놓고 예측과 갱신 두 단계를 따라가 보자.

$$p(\mathbf{x}_{k-1} \mid \mathbf{z}_{1:k-1}) = \mathcal{N}(\hat{\mathbf{x}}_{k-1|k-1}, \mathbf{P}_{k-1|k-1})$$

**Step 1: Prediction — Chapman-Kolmogorov 적분 계산**

선형 운동 모델은 직전 상태와 제어 입력, 노이즈의 선형 결합이다.
$$\mathbf{x}_k = \mathbf{F}_k \mathbf{x}_{k-1} + \mathbf{B}_k \mathbf{u}_k + \mathbf{w}_k$$

$\mathbf{x}_{k-1}$이 가우시안이고 $\mathbf{w}_k$가 이와 독립인 가우시안이므로 적분을 직접 풀지 않아도 $\mathbf{x}_k$의 분포가 가우시안임을 바로 알 수 있다. 기댓값을 취하면 예측 평균을 얻는다.

$$\hat{\mathbf{x}}_{k|k-1} = \mathbb{E}[\mathbf{x}_k \mid \mathbf{z}_{1:k-1}] = \mathbf{F}_k \hat{\mathbf{x}}_{k-1|k-1} + \mathbf{B}_k \mathbf{u}_k$$

예측 오차를 $\tilde{\mathbf{x}}_{k|k-1} = \mathbf{x}_k - \hat{\mathbf{x}}_{k|k-1} = \mathbf{F}_k \tilde{\mathbf{x}}_{k-1|k-1} + \mathbf{w}_k$로 적으면 $\mathbf{w}_k$와 이전 오차가 독립이므로 예측 공분산은 다음과 같이 전파된다.

$$\mathbf{P}_{k|k-1} = \mathbb{E}[\tilde{\mathbf{x}}_{k|k-1} \tilde{\mathbf{x}}_{k|k-1}^\top] = \mathbf{F}_k \mathbf{P}_{k-1|k-1} \mathbf{F}_k^\top + \mathbf{Q}_k$$

이로써 예측 단계가 내놓는 사전 분포는 $p(\mathbf{x}_k \mid \mathbf{z}_{1:k-1}) = \mathcal{N}(\hat{\mathbf{x}}_{k|k-1}, \mathbf{P}_{k|k-1})$로 확정된다.

**Step 2: Update — Bayes 정리 적용**

관측 모델 $\mathbf{z}_k = \mathbf{H}_k \mathbf{x}_k + \mathbf{v}_k$ 역시 선형이므로, 예측 상태 $\mathbf{x}_k$와 아직 관측 전인 $\mathbf{z}_k$를 나란히 묶은 결합 분포(joint distribution) 역시 가우시안이 된다.

$$\begin{bmatrix} \mathbf{x}_k \\ \mathbf{z}_k \end{bmatrix} \sim \mathcal{N}\left( \begin{bmatrix} \hat{\mathbf{x}}_{k|k-1} \\ \mathbf{H}_k \hat{\mathbf{x}}_{k|k-1} \end{bmatrix}, \begin{bmatrix} \mathbf{P}_{k|k-1} & \mathbf{P}_{k|k-1} \mathbf{H}_k^\top \\ \mathbf{H}_k \mathbf{P}_{k|k-1} & \mathbf{S}_k \end{bmatrix} \right)$$

여기서 우하단 블록 $\mathbf{S}_k = \mathbf{H}_k \mathbf{P}_{k|k-1} \mathbf{H}_k^\top + \mathbf{R}_k \in \mathbb{R}^{m \times m}$은 예측된 관측과 실제 관측 사이의 불확실성을 합친 혁신 공분산(innovation covariance)이다.

다변량 가우시안의 성질에 따라 결합 분포가 가우시안이면 조건부 분포 $p(\mathbf{x}_k \mid \mathbf{z}_k)$도 닫힌 식의 가우시안으로 떨어진다. 조건부 평균은 사전 평균 $\hat{\mathbf{x}}_{k|k-1}$에, 관측 오차와 상태 사이의 상관도만큼 보정량을 더한 형태가 된다.

$$\hat{\mathbf{x}}_{k|k} = \hat{\mathbf{x}}_{k|k-1} + \mathbf{P}_{k|k-1} \mathbf{H}_k^\top \mathbf{S}_k^{-1} (\mathbf{z}_k - \mathbf{H}_k \hat{\mathbf{x}}_{k|k-1})$$

이 식에서 관측 잔차에 곱해지는 행렬을 **칼만 이득(Kalman gain)** $\mathbf{K}_k \in \mathbb{R}^{n \times m}$으로 정의한다.

$$\boxed{\mathbf{K}_k = \mathbf{P}_{k|k-1} \mathbf{H}_k^\top \mathbf{S}_k^{-1} = \mathbf{P}_{k|k-1} \mathbf{H}_k^\top (\mathbf{H}_k \mathbf{P}_{k|k-1} \mathbf{H}_k^\top + \mathbf{R}_k)^{-1}}$$

칼만 이득은 예측의 불확실성과 관측의 불확실성을 견주어 어느 쪽에 무게를 실을지 자동으로 결정한다. 극단적인 두 경우를 보면 그 역할이 한눈에 들어온다.
- $\mathbf{R}_k \to \mathbf{0}$ (센서 관측이 매우 정확함): 관측 가능한 부분공간에서는 갱신된 상태가 측정값을 거의 그대로 따르며, 특히 $\mathbf{H}_k$가 정사각·가역 행렬이면 $\mathbf{K}_k \to \mathbf{H}_k^{-1}$에 수렴한다.
- $\mathbf{P}_{k|k-1} \to \mathbf{0}$ (예측 자체의 확신이 매우 큼): $\mathbf{K}_k \to \mathbf{0}$이 되어 새 관측을 거의 무시하고 예측치를 유지한다.
- 선형-가우시안 모델과 지정한 잡음 공분산이 실제 시스템과 일치한다면, 이 이득으로 계산한 갱신값이 정확한 조건부 평균이 된다.

**혁신(Innovation)**은 실제 관측값에서 현재 예측 상태로 예상한 관측값을 뺀 차이다.
$$\tilde{\mathbf{y}}_k = \mathbf{z}_k - \mathbf{H}_k \hat{\mathbf{x}}_{k|k-1} \in \mathbb{R}^m$$

이 잔차 벡터가 클수록 예측과 측정 사이에 어긋남이 크다는 뜻이지만, 반대로 혁신이 0이라고 해서 관측되지 않는 상태 성분이나 모델 오차까지 없다는 보장은 없다.

칼만 이득과 혁신을 쓰면 **상태 갱신**과 **공분산 갱신**은 간결한 두 식으로 정리된다.

$$\boxed{\hat{\mathbf{x}}_{k|k} = \hat{\mathbf{x}}_{k|k-1} + \mathbf{K}_k \tilde{\mathbf{y}}_k}$$

$$\boxed{\mathbf{P}_{k|k} = (\mathbf{I} - \mathbf{K}_k \mathbf{H}_k) \mathbf{P}_{k|k-1}}$$

수치 구현에서는 공분산 갱신식을 단순 형태 대신 Joseph form $\mathbf{P}_{k|k} = (\mathbf{I} - \mathbf{K}_k \mathbf{H}_k) \mathbf{P}_{k|k-1} (\mathbf{I} - \mathbf{K}_k \mathbf{H}_k)^\top + \mathbf{K}_k \mathbf{R}_k \mathbf{K}_k^\top$으로 쓰는 편이 안전하다. 대칭 양의 준정부호 행렬 둘을 더하는 구조 덕분에, $\mathbf{P}_{k|k-1}$과 $\mathbf{R}_k$가 양의 준정부호인 한 수치 오차로 공분산이 음의 고윳값을 갖게 되는 일을 크게 줄여 준다. 물론 부동소수점 반올림이 오래 쌓이면 미세한 비대칭이 남을 수 있으므로, 실무 코드에서는 공분산을 주기적으로 대칭화하거나 $LDL^\top$ 분해 같은 제곱근·분해 형태를 함께 활용한다.

#### KF 최적성 정리

**정리**: 선형-가우시안 시스템에서 Kalman Filter는 다음 세 기준을 동시에 만족하는 최적 추정량이다.
1. **MMSE (Minimum Mean Square Error) 추정량**: 평균 제곱 오차를 최소화한다.
2. **MAP (Maximum A Posteriori) 추정량**: 가우시안 분포에서는 평균과 최빈값이 일치하므로 MAP 추정과 MMSE 추정이 같아진다.
3. **BLUE (Best Linear Unbiased Estimator)**: 노이즈가 가우시안이 아니더라도 평균·공분산만 맞으면 모든 선형 비편향 추정량 가운데 최소 분산을 달성한다.

이처럼 Kalman의 1960년 원논문은 기존 Wiener 필터의 주파수 영역 해법을 상태 공간(state-space) 시간 영역으로 옮겨 오면서, 시변(time-varying) 시스템과 다변량 시스템까지 일관되게 다룰 수 있는 길을 열었다.

#### Python 구현

아래 코드는 방금 유도한 예측·갱신 식을 1차원 등속 운동 모델에 적용한 선형 Kalman Filter 구현이다. 위치만 측정하더라도 상태 전이 행렬 $\mathbf{F}$에 담긴 위치-속도 상관관계 덕분에 속도 추정치와 공분산이 함께 수렴하는 모습을 볼 수 있다.

```python
import numpy as np

class KalmanFilter:
    """선형 Kalman Filter 구현.
    
    상태 공간 모델:
        x_k = F @ x_{k-1} + B @ u_k + w_k,  w_k ~ N(0, Q)
        z_k = H @ x_k + v_k,                 v_k ~ N(0, R)
    """
    def __init__(self, F, H, Q, R, B=None):
        """
        Parameters
        ----------
        F : ndarray, shape (n, n) — 상태 전이 행렬
        H : ndarray, shape (m, n) — 관측 행렬
        Q : ndarray, shape (n, n) — 프로세스 노이즈 공분산
        R : ndarray, shape (m, m) — 관측 노이즈 공분산
        B : ndarray, shape (n, l) — 제어 입력 행렬 (optional)
        """
        self.F = F
        self.H = H
        self.Q = Q
        self.R = R
        self.B = B
        self.n = F.shape[0]
        
    def predict(self, x, P, u=None):
        """예측 단계.
        
        Parameters
        ----------
        x : ndarray, shape (n,) — 이전 상태 추정
        P : ndarray, shape (n, n) — 이전 공분산
        u : ndarray, shape (l,) — 제어 입력 (optional)
        
        Returns
        -------
        x_pred : ndarray, shape (n,) — 예측 상태
        P_pred : ndarray, shape (n, n) — 예측 공분산
        """
        x_pred = self.F @ x
        if self.B is not None and u is not None:
            x_pred += self.B @ u
        P_pred = self.F @ P @ self.F.T + self.Q
        return x_pred, P_pred
    
    def update(self, x_pred, P_pred, z):
        """갱신 단계.
        
        Parameters
        ----------
        x_pred : ndarray, shape (n,) — 예측 상태
        P_pred : ndarray, shape (n, n) — 예측 공분산
        z : ndarray, shape (m,) — 관측값
        
        Returns
        -------
        x_upd : ndarray, shape (n,) — 갱신 상태
        P_upd : ndarray, shape (n, n) — 갱신 공분산
        """
        # 혁신 (innovation)
        y = z - self.H @ x_pred                          # (m,)
        # 혁신 공분산
        S = self.H @ P_pred @ self.H.T + self.R           # (m, m)
        # 칼만 이득
        K = P_pred @ self.H.T @ np.linalg.inv(S)          # (n, m)
        # 상태 갱신
        x_upd = x_pred + K @ y                            # (n,)
        # 공분산 갱신 (Joseph form for numerical stability)
        I_KH = np.eye(self.n) - K @ self.H                # (n, n)
        P_upd = I_KH @ P_pred @ I_KH.T + K @ self.R @ K.T  # (n, n)
        return x_upd, P_upd

# 예제: 1D 등속 운동 모델로 위치 추정
# 상태: [position, velocity]^T
dt = 0.1  # 시간 간격
F = np.array([[1, dt],
              [0, 1]])      # (2, 2) 등속 운동 전이 행렬
H = np.array([[1, 0]])       # (1, 2) 위치만 관측
Q = np.array([[dt**3/3, dt**2/2],
              [dt**2/2, dt]]) * 0.1  # (2, 2) 프로세스 노이즈 (등속 모델 + 연속 시간 백색잡음 가속도)
R = np.array([[1.0]])         # (1, 1) 관측 노이즈 분산

kf = KalmanFilter(F, H, Q, R)

# 초기 상태
x = np.array([0.0, 1.0])  # 위치=0, 속도=1
P = np.eye(2) * 10.0       # 초기 불확실성 큼

# 시뮬레이션
np.random.seed(42)
true_positions = []
estimated_positions = []

for k in range(100):
    # Ground truth
    true_pos = 0.0 + 1.0 * k * dt  # 등속 운동
    true_positions.append(true_pos)
    
    # 예측
    x, P = kf.predict(x, P)
    
    # 노이즈가 섞인 관측
    z = np.array([true_pos + np.random.randn() * 1.0])
    
    # 갱신
    x, P = kf.update(x, P, z)
    estimated_positions.append(x[0])

print(f"최종 위치 추정: {x[0]:.3f}, 실제: {true_positions[-1]:.3f}")
print(f"최종 위치 불확실성 (1σ): {np.sqrt(P[0,0]):.3f}")
```

선형 모델에서는 이처럼 Kalman Filter가 손실 없이 작동하지만, 실제 센서 융합 시스템에서는 상태 전이와 관측에 곧바로 비선형성이 들어온다.

### 4.2.2 Extended Kalman Filter (EKF)

실제 로봇 시스템은 거의 항상 비선형이다. 카메라의 3D→2D 투영, IMU의 쿼터니언 기반 회전, LiDAR의 scan matching, GPS의 좌표 변환 모두 비선형 함수다. EKF는 이 비선형 함수를 현재 추정치 주변의 접평면으로 1차 근사한 뒤, 앞서 본 Kalman Filter의 공식을 그대로 빌려 쓰는 가장 직접적인 확장이다.

#### 1차 테일러 전개 (선형화)

비선형 운동 모델과 관측 모델은 일반적인 함수 $f, h$로 적는다.
$$\mathbf{x}_k = f(\mathbf{x}_{k-1}, \mathbf{u}_k) + \mathbf{w}_k$$
$$\mathbf{z}_k = h(\mathbf{x}_k) + \mathbf{v}_k$$

가우시안 분포가 비선형 함수를 지나면 출력 분포는 더 이상 가우시안이 아니다. EKF는 평균 추정치 주변에서 함수를 1차 테일러 전개로 선형화해 가우시안 근사를 유지한다.

먼저 운동 모델은 직전 사후 추정치 $\hat{\mathbf{x}}_{k-1|k-1}$ 근방에서 편미분 자코비안 $\mathbf{F}_k$로 선형화한다.

$$f(\mathbf{x}_{k-1}, \mathbf{u}_k) \approx f(\hat{\mathbf{x}}_{k-1|k-1}, \mathbf{u}_k) + \mathbf{F}_k (\mathbf{x}_{k-1} - \hat{\mathbf{x}}_{k-1|k-1})$$

$$\mathbf{F}_k = \left.\frac{\partial f}{\partial \mathbf{x}}\right|_{\hat{\mathbf{x}}_{k-1|k-1}, \mathbf{u}_k} \in \mathbb{R}^{n \times n}$$

관측 모델은 예측 단계를 방금 마친 사전 추정치 $\hat{\mathbf{x}}_{k|k-1}$ 근방에서 자코비안 $\mathbf{H}_k$로 선형화한다.

$$h(\mathbf{x}_k) \approx h(\hat{\mathbf{x}}_{k|k-1}) + \mathbf{H}_k (\mathbf{x}_k - \hat{\mathbf{x}}_{k|k-1})$$

$$\mathbf{H}_k = \left.\frac{\partial h}{\partial \mathbf{x}}\right|_{\hat{\mathbf{x}}_{k|k-1}} \in \mathbb{R}^{m \times n}$$

#### EKF 알고리즘

선형화한 자코비안 $\mathbf{F}_k$와 $\mathbf{H}_k$를 대입하면 EKF의 예측과 갱신 절차는 다음과 같이 정리된다.

**예측 단계**:
$$\hat{\mathbf{x}}_{k|k-1} = f(\hat{\mathbf{x}}_{k-1|k-1}, \mathbf{u}_k)$$
$$\mathbf{P}_{k|k-1} = \mathbf{F}_k \mathbf{P}_{k-1|k-1} \mathbf{F}_k^\top + \mathbf{Q}_k$$

평균 상태는 비선형 함수 $f$를 직접 통과시켜 정확히 전파하고, 1차 자코비안 $\mathbf{F}_k$는 불확실성(공분산)을 옮길 때만 쓴다.

**갱신 단계**:
$$\tilde{\mathbf{y}}_k = \mathbf{z}_k - h(\hat{\mathbf{x}}_{k|k-1})$$
$$\mathbf{S}_k = \mathbf{H}_k \mathbf{P}_{k|k-1} \mathbf{H}_k^\top + \mathbf{R}_k$$
$$\mathbf{K}_k = \mathbf{P}_{k|k-1} \mathbf{H}_k^\top \mathbf{S}_k^{-1}$$
$$\hat{\mathbf{x}}_{k|k} = \hat{\mathbf{x}}_{k|k-1} + \mathbf{K}_k \tilde{\mathbf{y}}_k$$
$$\mathbf{P}_{k|k} = (\mathbf{I} - \mathbf{K}_k \mathbf{H}_k) \mathbf{P}_{k|k-1}$$

혁신 $\tilde{\mathbf{y}}_k$를 계산할 때도 선형 근사 대신 비선형 관측 함수 $h$를 직접 평가한다.

#### EKF의 한계

EKF는 계산량이 적고 구현이 직관적이지만, 세 가지 구조적 약점을 안고 있다.

1. **선형화 오차**: 비선형성이 강하거나 초기 추정 오차가 클수록 1차 테일러 근사가 실제 곡률을 따라가지 못한다. 이때 실제 추정 오차가 필터가 스스로 보고하는 공분산보다 커지는 현상이 벌어지는데, 이를 필터가 일관성(consistency)을 잃었다고 한다.

2. **자코비안 계산 부담**: 매 스텝 현재 추정치에서 $\mathbf{F}_k$와 $\mathbf{H}_k$를 새로 계산해야 한다. 상태 차원이 커지고 관측 모델이 복잡해질수록 손으로 유도한 자코비안에 부호 실수나 누락이 섞이기 쉽다.

3. **단봉(unimodal) 가정**: 분포를 단일 가우시안으로만 표현하므로 여러 위치 후보가 공존하는 다봉 사후 분포를 다루려면 가우시안 혼합이나 입자 기반 표현이 필요하다.

특히 3차원 공간에서 로봇의 자세(rotation)를 함께 추정하려 들면, EKF의 기본 덧셈 갱신 $\hat{\mathbf{x}} + \mathbf{K}\tilde{\mathbf{y}}$가 회전의 기하적 제약과 정면으로 충돌한다. 이 문제를 우회하려고 상태를 명목 궤적과 작은 오차로 나누어 다루는 방식이 다음 절의 ESKF다.

### 4.2.3 Error-State Kalman Filter (ESKF)

3차원 공간에서 IMU를 결합한 항법·상태 추정에는 오차 상태를 추적하는 ESKF(Error-State Kalman Filter)가 널리 쓰인다. 필터 기반 VIO인 MSCKF·OpenVINS와 LiDAR-관성 오도메트리인 FAST-LIO 계열이 모두 이 구조를 채택하고 있으며, 반면 [VINS-Mono](https://doi.org/10.1109/TRO.2018.2853729)처럼 필터 대신 sliding-window 비선형 최적화를 쓰는 갈래도 있다.

#### 왜 EKF 대신 ESKF를 쓰는가

3D 자세(orientation)가 포함된 상태를 일반 EKF에 그대로 넣고 돌리면 세 가지 난점에 부딪힌다.

**문제 1: 회전의 비유클리드 특성**

3차원 회전은 평평한 벡터 공간 $\mathbb{R}^n$이 아니라 곡면 매니폴드인 $SO(3)$ 위에 놓인다. 자세를 단위 쿼터니언 $\mathbf{q} \in \mathbb{H}$로 적으면 $\|\mathbf{q}\| = 1$ 제약이 따르고, 회전 행렬 $\mathbf{R} \in SO(3)$로 적으면 $\mathbf{R}^\top \mathbf{R} = \mathbf{I}$와 $\det(\mathbf{R}) = 1$ 제약을 지켜야 한다.

문제는 기본 EKF의 갱신식 $\hat{\mathbf{x}} \leftarrow \hat{\mathbf{x}} + \mathbf{K} \tilde{\mathbf{y}}$에 쓰이는 "+"가 일반 유클리드 덧셈이라는 점이다. 쿼터니언 네 성분에 보정 벡터를 그대로 더하면 노름이 1에서 벗어나 회전으로서의 성질이 깨진다. 그렇다고 갱신 뒤에 값만 억지로 다시 정규화하면, 공분산 행렬에는 여전히 제약 방향(반지름 방향)의 불확실성이 잉여 성분으로 남아 통계적 일관성이 어긋난다. 쿼터니언을 상태 벡터에 직접 두고 EKF를 설계하려면 정규화와 함께 공분산 투영까지 따로 처리해야 하는 번거로움이 따른다.

**문제 2: 오차 상태를 0 근처에 묶어 두기**

ESKF는 추정할 대상을 전체 상태가 아니라 명목 상태와 참 상태 사이의 작은 차이, 곧 오차 상태(error state) $\delta\mathbf{x} = \mathbf{x} \boxminus \hat{\mathbf{x}}$로 바꾼다. 관측 갱신이 끝날 때마다 추정한 오차를 명목 상태에 더한 뒤 오차 상태는 0으로 되돌린다. 명목 궤적이 실제 움직임과 크게 어긋나지 않는 한 오차는 늘 0 근처의 작은 값으로 유지되므로, 0 근방의 접선 공간에서 선형화해도 1차 근사가 잘 들어맞는다. 물론 오차 상태를 0으로 리셋한다고 해서 이미 틀어진 명목 궤적이 저절로 참값에 붙는 것은 아니므로, 초기화와 관측 가능성, 모델 오차는 별도로 점검해야 한다.

**문제 3: 명목 상태와 오차 상태의 서로 다른 동역학**

로봇이 빠르게 회전하고 가속하는 큰 움직임은 고주파 IMU 적분으로 명목 상태가 떠맡고, 오차 상태는 그 주변을 맴도는 비교적 완만한 오차 성분만 추적한다. 덕분에 평소에는 IMU만으로 명목 상태를 촘촘히 적분하다가, 그보다 낮은 주기로 카메라나 LiDAR 관측이 들어올 때 오차 상태를 갱신해 전체 추정을 바로잡는 다중 주기(multi-rate) 융합 구조가 깔끔하게 맞아떨어진다.

#### ESKF 구조

이 아이디어에 따라 ESKF는 전체 상태를 성격이 다른 두 부분으로 나누어 관리한다.

1. **명목 상태(Nominal State)** $\hat{\mathbf{x}}$: 노이즈 항을 뺀 결정론적 비선형 운동 모델로 IMU 측정을 그대로 적분하는 상태다. 불확실성은 여기 담지 않고 오차 상태 쪽에 맡긴다.
2. **오차 상태(Error State)** $\delta\mathbf{x}$: 명목 상태와 실제 상태 사이의 작은 차이로, 칼만 필터가 평균과 공분산을 추적하는 대상이다. 명목 상태가 참값 부근을 따라가는 동안에는 접선 공간의 선형 근사가 유효하다. 오차를 명목 상태에 주입한 직후에는 reset Jacobian을 곱해 오차 공분산의 기준 좌표를 새 명목 상태에 맞게 옮겨 준다.

실제 상태는 명목 상태와 오차 상태를 매니폴드 연산 $\boxplus$로 합성해 복원한다.

$$\mathbf{x}_{\text{true}} = \hat{\mathbf{x}} \boxplus \delta\mathbf{x}$$

위치·속도·바이어스 같은 유클리드 성분에서 $\boxplus$는 보통 벡터 덧셈이지만, 회전 성분에서는 회전 행렬의 지수 사상(exponential map)을 오른쪽에 곱하는 형태로 정의한다.

$$\mathbf{R}_{\text{true}} = \hat{\mathbf{R}} \cdot \text{Exp}(\delta\boldsymbol{\theta})$$

같은 연산을 단위 쿼터니언으로 쓰면 다음과 같이 소각(small-angle) 근사를 얻을 수 있다.

$$\mathbf{q}_{\text{true}} = \hat{\mathbf{q}} \otimes \delta\mathbf{q} \approx \hat{\mathbf{q}} \otimes \begin{bmatrix} 1 \\ \frac{1}{2}\delta\boldsymbol{\theta} \end{bmatrix}$$

여기서 $\delta\boldsymbol{\theta} \in \mathbb{R}^3$는 회전 오차를 3차원 각-축(angle-axis) 벡터로 나타낸 최소 표현이다.

#### IMU 기반 ESKF의 상태 벡터

IMU에 카메라나 LiDAR를 결합하는 표준 3차원 항법 시스템에서는 명목 상태와 오차 상태를 다음과 같이 구성한다.

**명목 상태** (쿼터니언 사용 시 16차원):
$$\hat{\mathbf{x}} = \begin{bmatrix} {}^W\hat{\mathbf{p}} \\ {}^W\hat{\mathbf{v}} \\ \hat{\mathbf{q}}_{WB} \\ \hat{\mathbf{b}}_a \\ \hat{\mathbf{b}}_g \end{bmatrix} \in \mathbb{R}^{3} \times \mathbb{R}^{3} \times \mathbb{S}^3 \times \mathbb{R}^{3} \times \mathbb{R}^{3}$$

**오차 상태** (회전의 최소 파라미터화 덕분에 15차원):
$$\delta\mathbf{x} = \begin{bmatrix} \delta\mathbf{p} \\ \delta\mathbf{v} \\ \delta\boldsymbol{\theta} \\ \delta\mathbf{b}_a \\ \delta\mathbf{b}_g \end{bmatrix} \in \mathbb{R}^{15}$$

단위 쿼터니언 $\hat{\mathbf{q}}_{WB}$는 성분이 네 개지만 $\|\hat{\mathbf{q}}_{WB}\|=1$ 제약 때문에 실제 자유도는 셋뿐이다. ESKF는 오차 상태의 회전을 3차원 접선 벡터 $\delta\boldsymbol{\theta}$로 둠으로써 공분산 행렬을 잉여 자유도 없는 $15 \times 15$ 정칙(non-singular) 행렬로 유지한다.

#### ESKF 알고리즘

ESKF의 한 주기는 명목 상태 전파, 오차 상태 공분산 전파, 관측 갱신, 그리고 오차 주입과 리셋 네 단계로 돌아간다.

**1단계: 명목 상태 전파 (IMU mechanization)**

IMU 측정 $(\tilde{\boldsymbol{\omega}}_k, \tilde{\mathbf{a}}_k)$가 들어오면 현재 추정 바이어스를 빼고, 랜덤 노이즈 항은 0으로 놓은 채 명목 상태를 적분한다.

$$\hat{\mathbf{q}}_{k+1} = \hat{\mathbf{q}}_k \otimes \mathbf{q}\{(\tilde{\boldsymbol{\omega}}_k - \hat{\mathbf{b}}_{g,k}) \Delta t\}$$
$$\hat{\mathbf{v}}_{k+1} = \hat{\mathbf{v}}_k + (\hat{\mathbf{R}}_k (\tilde{\mathbf{a}}_k - \hat{\mathbf{b}}_{a,k}) + \mathbf{g}) \Delta t$$
$$\hat{\mathbf{p}}_{k+1} = \hat{\mathbf{p}}_k + \hat{\mathbf{v}}_k \Delta t + \frac{1}{2}(\hat{\mathbf{R}}_k (\tilde{\mathbf{a}}_k - \hat{\mathbf{b}}_{a,k}) + \mathbf{g}) \Delta t^2$$
$$\hat{\mathbf{b}}_{a,k+1} = \hat{\mathbf{b}}_{a,k}$$
$$\hat{\mathbf{b}}_{g,k+1} = \hat{\mathbf{b}}_{g,k}$$

여기서 $\hat{\mathbf{R}}_k = \mathbf{R}(\hat{\mathbf{q}}_k) \in SO(3)$는 명목 회전 행렬이고, $\mathbf{g} = [0, 0, -9.81]^\top \, \text{m/s}^2$는 중력 벡터다.

**2단계: 오차 상태 전파 (prediction)**

다음으로 명목 상태에서 벗어난 오차가 시간에 따라 어떻게 자라는지 연속 시간 동역학을 세운다. 실제 IMU 측정 모델 $\tilde{\boldsymbol{\omega}} = \boldsymbol{\omega} + \mathbf{b}_g + \mathbf{n}_g$와 $\tilde{\mathbf{a}} = \mathbf{R}^\top(\mathbf{a}_W - \mathbf{g}) + \mathbf{b}_a + \mathbf{n}_a$를 참 상태 방정식에 대입한 뒤 명목 상태 방정식을 빼고 2차 이상의 미소항을 버리면 다음 선형 미분방정식을 얻는다.

$$\delta\dot{\boldsymbol{\theta}} = -[\tilde{\boldsymbol{\omega}} - \hat{\mathbf{b}}_g]_\times \delta\boldsymbol{\theta} - \delta\mathbf{b}_g - \mathbf{n}_g$$
$$\delta\dot{\mathbf{v}} = -\hat{\mathbf{R}}[\tilde{\mathbf{a}} - \hat{\mathbf{b}}_a]_\times \delta\boldsymbol{\theta} - \hat{\mathbf{R}} \delta\mathbf{b}_a - \hat{\mathbf{R}} \mathbf{n}_a$$
$$\delta\dot{\mathbf{p}} = \delta\mathbf{v}$$
$$\delta\dot{\mathbf{b}}_a = \mathbf{n}_{ba}$$
$$\delta\dot{\mathbf{b}}_g = \mathbf{n}_{bg}$$

여기서 $[\mathbf{a}]_\times$는 3차원 벡터 $\mathbf{a}$의 외적을 행렬 곱으로 바꾸는 반대칭 행렬(skew-symmetric matrix)이다.

$$[\mathbf{a}]_\times = \begin{bmatrix} 0 & -a_3 & a_2 \\ a_3 & 0 & -a_1 \\ -a_2 & a_1 & 0 \end{bmatrix} \in \mathbb{R}^{3 \times 3}$$

다섯 줄의 미분방정식을 블록 행렬로 모으면 깔끔한 선형 상태 공간 모델이 된다.

$$\delta\dot{\mathbf{x}} = \mathbf{F}_c \delta\mathbf{x} + \mathbf{G}_c \mathbf{n}$$

$$\mathbf{F}_c = \begin{bmatrix}
\mathbf{0}_3 & \mathbf{I}_3 & \mathbf{0}_3 & \mathbf{0}_3 & \mathbf{0}_3 \\
\mathbf{0}_3 & \mathbf{0}_3 & -\hat{\mathbf{R}}[\tilde{\mathbf{a}} - \hat{\mathbf{b}}_a]_\times & -\hat{\mathbf{R}} & \mathbf{0}_3 \\
\mathbf{0}_3 & \mathbf{0}_3 & -[\tilde{\boldsymbol{\omega}} - \hat{\mathbf{b}}_g]_\times & \mathbf{0}_3 & -\mathbf{I}_3 \\
\mathbf{0}_3 & \mathbf{0}_3 & \mathbf{0}_3 & \mathbf{0}_3 & \mathbf{0}_3 \\
\mathbf{0}_3 & \mathbf{0}_3 & \mathbf{0}_3 & \mathbf{0}_3 & \mathbf{0}_3
\end{bmatrix} \in \mathbb{R}^{15 \times 15}$$

이 연속 시간 모델을 샘플링 간격 $\Delta t$로 1차 근사 이산화($\mathbf{F}_d \approx \mathbf{I} + \mathbf{F}_c \Delta t$)하면, 오차 공분산의 전파는 표준 칼만 예측과 같은 꼴로 계산된다. (직전 갱신 직후 오차 상태 평균이 0으로 리셋되므로 평균은 전파하지 않아도 계속 0으로 남는다.)

$$\mathbf{P}_{k+1|k} = \mathbf{F}_d \mathbf{P}_{k|k} \mathbf{F}_d^\top + \mathbf{G}_d \mathbf{Q}_d \mathbf{G}_d^\top$$

**3단계: 관측 갱신**

카메라나 LiDAR 관측 $\mathbf{z}$가 들어오면, 먼저 관측 모델 $\mathbf{z} = h(\mathbf{x}_{\text{true}})$를 현재 명목 상태 $\hat{\mathbf{x}}$ 근방에서 오차 상태 $\delta\mathbf{x}$에 대해 선형화한다.

$$\mathbf{z} - h(\hat{\mathbf{x}}) \approx \mathbf{H} \delta\mathbf{x} + \mathbf{v}$$

$$\mathbf{H} = \frac{\partial h}{\partial \delta\mathbf{x}}\bigg|_{\hat{\mathbf{x}}} \in \mathbb{R}^{m \times 15}$$

이 자코비안 $\mathbf{H}$는 연쇄 법칙을 써서 두 부분으로 쪼개 구하면 편리하다. 앞쪽 항은 일반 센서 관측 모델의 유클리드 미분이고, 뒤쪽 항은 15차원 접선 공간 오차가 16차원 명목 상태에 미치는 미분이다.

$$\mathbf{H} = \frac{\partial h}{\partial \mathbf{x}_{\text{true}}} \cdot \frac{\partial \mathbf{x}_{\text{true}}}{\partial \delta\mathbf{x}}\bigg|_{\delta\mathbf{x}=\mathbf{0}}$$

자코비안 $\mathbf{H}$가 구해지면 칼만 이득, 오차 상태 추정치, 공분산 갱신은 표준 칼만 필터와 동일하다.
$$\mathbf{K} = \mathbf{P}_{k|k-1} \mathbf{H}^\top (\mathbf{H} \mathbf{P}_{k|k-1} \mathbf{H}^\top + \mathbf{R})^{-1}$$
$$\delta\hat{\mathbf{x}} = \mathbf{K}(\mathbf{z} - h(\hat{\mathbf{x}}))$$
$$\mathbf{P}_{k|k} = (\mathbf{I} - \mathbf{K}\mathbf{H})\mathbf{P}_{k|k-1}$$

**4단계: 오차 상태 주입과 리셋**

마지막으로 방금 추정한 오차 $\delta\hat{\mathbf{x}}$를 명목 상태에 합쳐(inject) 명목 궤적을 보정한다.

$$\hat{\mathbf{p}} \leftarrow \hat{\mathbf{p}} + \delta\hat{\mathbf{p}}$$
$$\hat{\mathbf{v}} \leftarrow \hat{\mathbf{v}} + \delta\hat{\mathbf{v}}$$
$$\hat{\mathbf{q}} \leftarrow \hat{\mathbf{q}} \otimes \mathbf{q}\{\delta\hat{\boldsymbol{\theta}}\}$$
$$\hat{\mathbf{b}}_a \leftarrow \hat{\mathbf{b}}_a + \delta\hat{\mathbf{b}}_a$$
$$\hat{\mathbf{b}}_g \leftarrow \hat{\mathbf{b}}_g + \delta\hat{\mathbf{b}}_g$$

명목 상태가 갱신되었으므로 오차 상태는 $\delta\hat{\mathbf{x}} \leftarrow \mathbf{0}$으로 비우고, 회전 기준점이 바뀐 만큼 공분산도 리셋 자코비안 $\mathbf{G}$로 옮겨 준다.

$$\mathbf{P} \leftarrow \mathbf{G} \mathbf{P} \mathbf{G}^\top$$

여기서 $\mathbf{G} = \frac{\partial (\delta\mathbf{x} \boxminus \delta\hat{\mathbf{x}})}{\partial \delta\mathbf{x}}\big|_{\delta\hat{\mathbf{x}}}$이며, 회전 보정량 $\delta\hat{\boldsymbol{\theta}}$가 매우 작으면 $\mathbf{G} \approx \mathbf{I}$로 근사하고 넘어가는 구현도 적지 않다.

이처럼 EKF와 ESKF는 모두 1차 미분(자코비안)으로 비선형성을 다룬다. 반면 관측 함수가 지나치게 복잡해 자코비안을 손으로 유도하기 어렵거나 1차 근사보다 더 높은 곡률 정확도가 필요할 때는, 미분 대신 대표 샘플을 통과시키는 UKF가 대안이 된다.

### 4.2.4 Unscented Kalman Filter (UKF)

UKF(Unscented Kalman Filter)는 비선형 함수를 자코비안으로 잘라 근사하는 대신, 함수 자체는 손대지 않고 확률 분포를 소수의 대표점(sigma point)으로 뽑아 그대로 함수에 통과시키는 필터다. Julier와 Uhlmann은 임의의 비선형 함수 전체를 근사하기보다 확률 분포의 평균과 공분산을 근사하는 편이 훨씬 쉽다는 통찰에서 이 기법을 제안했다.

#### Unscented Transform

핵심 도구인 Unscented Transform(UT)은 $n$차원 가우시안 $\mathbf{x} \sim \mathcal{N}(\hat{\mathbf{x}}, \mathbf{P})$의 평균과 공분산을 정확히 재현하도록 $2n+1$개의 sigma point와 가중치를 배치하는 방법이다.

$$\boldsymbol{\chi}_0 = \hat{\mathbf{x}}, \quad w_0 = \frac{\lambda}{n + \lambda}$$
$$\boldsymbol{\chi}_i = \hat{\mathbf{x}} + \left(\sqrt{(n+\lambda)\mathbf{P}}\right)_i, \quad w_i = \frac{1}{2(n+\lambda)}, \quad i = 1, \ldots, n$$
$$\boldsymbol{\chi}_{n+i} = \hat{\mathbf{x}} - \left(\sqrt{(n+\lambda)\mathbf{P}}\right)_i, \quad w_{n+i} = \frac{1}{2(n+\lambda)}, \quad i = 1, \ldots, n$$

각 기호가 뜻하는 바는 다음과 같다.
- $\lambda = \alpha^2(n + \kappa) - n$: 스케일링 파라미터 ($\alpha$는 중심 주변으로 sigma point가 퍼지는 정도를 조절하고, $\kappa$는 보조 파라미터로 보통 $\kappa = 0$이나 $3-n$을 둔다)
- $\left(\sqrt{(n+\lambda)\mathbf{P}}\right)_i$: 행렬 $(n+\lambda)\mathbf{P}$에 Cholesky 분해를 적용했을 때 얻어지는 제곱근 행렬의 $i$번째 열

생성한 sigma point를 하나씩 비선형 함수 $f$에 그대로 통과시킨 뒤,

$$\boldsymbol{\gamma}_i = f(\boldsymbol{\chi}_i)$$

변환된 점들의 가중 평균과 가중 공분산을 다시 모으면 출력 분포의 통계량을 얻는다.

$$\hat{\mathbf{y}} = \sum_{i=0}^{2n} w_i^{(m)} \boldsymbol{\gamma}_i$$
$$\mathbf{P}_y = \sum_{i=0}^{2n} w_i^{(c)} (\boldsymbol{\gamma}_i - \hat{\mathbf{y}})(\boldsymbol{\gamma}_i - \hat{\mathbf{y}})^\top$$

여기서 $w_i^{(m)}$과 $w_i^{(c)}$는 각각 평균과 공분산 복원에 쓰는 가중치로, $i \geq 1$에서는 두 값이 같고 중심점만 $w_0^{(c)} = w_0^{(m)} + (1 - \alpha^2 + \beta)$처럼 달라진다($\beta = 2$가 가우시안 분포에서 최적값이다).

#### UKF 알고리즘

이 Unscented Transform을 예측과 갱신 단계에 차례로 적용하면 미분 없는 칼만 필터가 완성된다.

**예측 단계**:
1. 현재 상태 $(\hat{\mathbf{x}}_{k-1|k-1}, \mathbf{P}_{k-1|k-1})$에서 $2n+1$개의 sigma point 생성
2. 각 sigma point를 비선형 운동 모델에 통과: $\boldsymbol{\chi}_{k|k-1}^{(i)} = f(\boldsymbol{\chi}_{k-1|k-1}^{(i)}, \mathbf{u}_k)$
3. 변환된 점들의 가중 평균으로 예측 상태 계산: $\hat{\mathbf{x}}_{k|k-1} = \sum w_i^{(m)} \boldsymbol{\chi}_{k|k-1}^{(i)}$
4. 가중 공분산에 프로세스 노이즈 공분산 합산: $\mathbf{P}_{k|k-1} = \sum w_i^{(c)} (\boldsymbol{\chi}_{k|k-1}^{(i)} - \hat{\mathbf{x}}_{k|k-1})(\cdots)^\top + \mathbf{Q}_k$

**갱신 단계**:
1. 프로세스 잡음 $\mathbf{Q}_k$가 반영된 예측 평균과 공분산에서 sigma point 재생성
2. 각 점을 비선형 관측 모델에 통과: $\boldsymbol{\zeta}_k^{(i)} = h(\boldsymbol{\chi}_{k|k-1}^{(i)})$
3. 예측 관측 평균 계산: $\hat{\mathbf{z}}_k = \sum w_i^{(m)} \boldsymbol{\zeta}_k^{(i)}$
4. 관측 공분산 계산: $\mathbf{P}_{zz} = \sum w_i^{(c)} (\boldsymbol{\zeta}_k^{(i)} - \hat{\mathbf{z}}_k)(\cdots)^\top + \mathbf{R}_k$
5. 상태-관측 간 교차 공분산 계산: $\mathbf{P}_{xz} = \sum w_i^{(c)} (\boldsymbol{\chi}_{k|k-1}^{(i)} - \hat{\mathbf{x}}_{k|k-1})(\boldsymbol{\zeta}_k^{(i)} - \hat{\mathbf{z}}_k)^\top$
6. 칼만 이득 산출: $\mathbf{K}_k = \mathbf{P}_{xz} \mathbf{P}_{zz}^{-1}$
7. 상태와 공분산 갱신: $\hat{\mathbf{x}}_{k|k} = \hat{\mathbf{x}}_{k|k-1} + \mathbf{K}_k (\mathbf{z}_k - \hat{\mathbf{z}}_k)$, $\mathbf{P}_{k|k} = \mathbf{P}_{k|k-1} - \mathbf{K}_k \mathbf{P}_{zz} \mathbf{K}_k^\top$

#### UKF의 장단점

**장점**:
- 해석적 자코비안을 구할 필요가 없다. 렌즈 왜곡 모델이 얽힌 카메라 투영처럼 편미분이 까다로운 관측 모델에서도 함수를 그대로 호출하기만 하면 된다.
- 함수에 직접 통과시킨 sigma point의 퍼짐을 모으므로, 1차 접평면만 보는 EKF보다 비선형 함수의 곡률 효과를 더 잘 반영한다. 다만 실제 정확도 차수는 입력 분포와 함수의 성질, sigma point 파라미터 설정에 따라 달라진다.

**단점**:
- 매 스텝 $2n+1$개의 sigma point를 전부 비선형 함수에 통과시켜야 하므로, 상태 차원 $n$이 커질수록 연산량이 빠르게 늘어난다.
- $SO(3)$ 같은 매니폴드 위의 상태를 다루려면 sigma point 생성과 가중 평균 계산까지 모두 매니폴드 연산으로 다시 정의해야 해 구현이 복잡해진다.
- 실제 IMU 기반 상태 추정에서 UKF보다 ESKF가 더 널리 쓰이는 이유도 여기에 있다. ESKF는 접선 공간 오차로 매니폴드를 자연스럽게 처리할 뿐 아니라, sigma point 여러 개를 반복 전파할 필요가 없어 계산 부담이 가볍기 때문이다. 물론 두 필터 가운데 어느 쪽이 더 높은 정확도를 내는지는 구체적인 모델, 초기 오차 크기, 파라미터 조정에 달려 있다.

한편 관측 모델의 자코비안을 구할 수 있더라도, 예측과 실제 관측 사이의 거리가 멀면 한 번의 선형화만으로는 오차가 남는다. 갱신 단계에서 선형화 지점을 반복해서 옮겨 잡는 방식이 다음의 IEKF다.

### 4.2.5 Iterated Extended Kalman Filter (IEKF)

IEKF(Iterated Extended Kalman Filter)는 EKF의 갱신 단계를 한 번의 선형화로 끝내지 않고, 갱신된 추정치 위에서 다시 선형화하기를 수렴할 때까지 되풀이해 비선형 관측 모델의 정확도를 끌어올리는 필터다.

#### 동기: EKF 갱신의 선형화 오차

표준 EKF는 관측 자코비안 $\mathbf{H}_k$를 예측 추정치 $\hat{\mathbf{x}}_{k|k-1}$ 자리에서 계산한다. 그런데 새 관측의 정보가 강해서 갱신 후 추정치 $\hat{\mathbf{x}}_{k|k}$가 예측 $\hat{\mathbf{x}}_{k|k-1}$에서 꽤 멀리 이동한다면, 결국 자코비안을 엉뚱한 자리에서 평가해 갱신한 셈이 된다. IEKF는 방금 얻은 갱신 추정치로 선형화 기준점을 옮겨 자코비안과 칼만 이득을 다시 계산하는 반복 루프를 돈다.

#### IEKF 알고리즘

예측 단계는 일반 EKF와 완전히 같고, 갱신 단계만 초기값 $\hat{\mathbf{x}}^{(0)} = \hat{\mathbf{x}}_{k|k-1}$에서 출발해 반복 루프로 바뀐다.

반복 횟수 $j = 0, 1, 2, \ldots$를 늘려 가며 변화량이 기준 아래로 떨어질 때까지 다음 세 식을 되풀이한다.

$$\mathbf{H}^{(j)} = \left.\frac{\partial h}{\partial \mathbf{x}}\right|_{\hat{\mathbf{x}}^{(j)}}$$

$$\mathbf{K}^{(j)} = \mathbf{P}_{k|k-1} \mathbf{H}^{(j)\top} (\mathbf{H}^{(j)} \mathbf{P}_{k|k-1} \mathbf{H}^{(j)\top} + \mathbf{R}_k)^{-1}$$

$$\hat{\mathbf{x}}^{(j+1)} = \hat{\mathbf{x}}_{k|k-1} + \mathbf{K}^{(j)} \left[\mathbf{z}_k - h(\hat{\mathbf{x}}^{(j)}) - \mathbf{H}^{(j)}(\hat{\mathbf{x}}_{k|k-1} - \hat{\mathbf{x}}^{(j)})\right]$$

여기서 대괄호 안의 추가 항 $-\mathbf{H}^{(j)}(\hat{\mathbf{x}}_{k|k-1} - \hat{\mathbf{x}}^{(j)})$는 사전 평균 $\hat{\mathbf{x}}_{k|k-1}$에 대한 MAP 비용과 관측 잔차의 균형을 올바른 기준점에서 맞추기 위한 보정이다. 수렴한 뒤에는 마지막 반복 결과를 상태와 공분산에 반영한다: $\hat{\mathbf{x}}_{k|k} = \hat{\mathbf{x}}^{(j+1)}$, $\mathbf{P}_{k|k} = (\mathbf{I} - \mathbf{K}^{(j)} \mathbf{H}^{(j)}) \mathbf{P}_{k|k-1}$.

수학적으로 보면 IEKF의 반복 갱신은 단일 시점의 관측 갱신 문제를 Gauss-Newton 최적화로 푸는 것과 동치다(Bell & Cathey, 1993). 뒤에서 다룰 §4.5의 factor graph 최적화는 이 같은 Gauss-Newton 원리를 한 시점이 아닌 전체 궤적 위로 확장한 것으로 볼 수 있다. 실전 LiDAR-관성 오도메트리인 FAST-LIO2 역시 수백~수천 개의 LiDAR point-to-plane 잔차를 상태 갱신에 집어넣고 반복 재선형화하는 매니폴드 위 IEKF를 사용한다. 한 번만 선형화할 때보다 관측 잔차의 곡률을 훨씬 잘 따라가지만, 실제로 정확도가 얼마나 올라가고 몇 회 만에 수렴하는지는 초기 오차 크기, 주변 환경의 기하 구조, 수렴 판정 기준에 따라 달라진다.

지금까지 살펴본 KF, EKF, ESKF, UKF, IEKF는 모두 사후 분포를 단 하나의 가우시안 $\mathcal{N}(\hat{\mathbf{x}}, \mathbf{P})$로 요약한다는 공통점이 있다. 하지만 로봇의 위치 후보가 여러 곳에 동시에 떠 있는 다봉 분포나, 가우시안 하나로 덮기 힘든 극심한 비선형 시스템 앞에서는 이 가정이 무너진다. 이때 분포 자체를 수식 대신 수많은 입자 샘플의 집합으로 직접 표현하는 방법이 Particle Filter다.

---

## 4.3 Particle Filter

앞선 칼만 필터 계열은 사후 분포가 하나의 봉우리만 갖는 가우시안이라는 전제 위에서 빠르게 동작한다. 하지만 로봇이 대칭적인 복도 어디쯤에 있는지 여러 가능성을 동시에 품어야 하는 전역 위치 추정(global localization)이나, 가우시안 하나로 감당하기 어려운 비선형 동역학을 만났을 때는 모수적 가정 자체를 내려놓아야 한다. Particle Filter는 확률 분포의 형태를 미리 가정하지 않고 수많은 표본(particle)의 집합으로 분포를 직접 표현한다.

### 4.3.1 Sequential Monte Carlo (SMC) 개요

Particle Filter(PF), 곧 Sequential Monte Carlo(SMC) 방법은 사후 분포를 가중치가 붙은 샘플 집합으로 근사한다. 상태 공간 곳곳에 입자를 뿌려 두고 관측과 잘 맞는 입자에 높은 가중치를 주는 방식이므로, 가우시안 가정이 필요 없고 봉우리가 여러 개인 다봉(multimodal) 분포나 비선형성이 강한 시스템도 자연스럽게 다룬다.

사후 분포에 대한 입자 근사는 디랙 델타 함수의 가중 합으로 적는다.

$$p(\mathbf{x}_k \mid \mathbf{z}_{1:k}) \approx \sum_{i=1}^{N} w_k^{(i)} \delta(\mathbf{x}_k - \mathbf{x}_k^{(i)})$$

이 식의 각 요소는 다음을 뜻한다.
- $N$: 전체 입자 수
- $\mathbf{x}_k^{(i)}$: $i$번째 입자가 가리키는 상태
- $w_k^{(i)}$: $i$번째 입자의 정규화된 가중치 ($\sum_{i=1}^N w_k^{(i)} = 1$)
- $\delta(\cdot)$: 디랙 델타 함수

입자 수 $N \to \infty$에 가까워질수록 이 경험적 분포는 실제 사후 분포에 수렴한다. 실제 구현에서는 연산 예산 안에서 유한 개의 입자만 유지하며, 필요한 입자 수는 상태 공간의 차원과 분포의 복잡도를 보고 정한다.

### 4.3.2 Importance Sampling

이론적으로는 실제 사후 분포 $p(\mathbf{x}_k \mid \mathbf{z}_{1:k})$에서 곧바로 샘플을 뽑고 싶지만, 그 분포 자체를 모르기 때문에 직접 샘플링할 수 없다. 대신 샘플을 뽑기 쉬운 제안 분포(proposal distribution) $q(\mathbf{x}_k \mid \mathbf{x}_{k-1}, \mathbf{z}_k)$에서 입자를 생성한 뒤, 실제 분포와 제안 분포 사이의 비율을 중요도 가중치(importance weight)로 곱해 분포의 편향을 보정한다.

입자의 가중치를 매 스텝 재귀적으로 갱신하는 규칙은 전체 궤적의 사후 확률에 Bayes 정리를 적용해 얻을 수 있다.

$$p(\mathbf{x}_{0:k} \mid \mathbf{z}_{1:k}) = \frac{p(\mathbf{z}_k \mid \mathbf{x}_k) \, p(\mathbf{x}_k \mid \mathbf{x}_{k-1}) \, p(\mathbf{x}_{0:k-1} \mid \mathbf{z}_{1:k-1})}{p(\mathbf{z}_k \mid \mathbf{z}_{1:k-1})}$$

양변을 제안 분포의 인수분해로 나누어 이전 단계와의 중요도 비율을 정리하면 다음 재귀 갱신식이 나온다.

$$w_k^{(i)} \propto w_{k-1}^{(i)} \cdot \frac{p(\mathbf{z}_k \mid \mathbf{x}_k^{(i)}) \, p(\mathbf{x}_k^{(i)} \mid \mathbf{x}_{k-1}^{(i)})}{q(\mathbf{x}_k^{(i)} \mid \mathbf{x}_{k-1}^{(i)}, \mathbf{z}_k)}$$

**가장 간단한 제안 분포**: 실무에서 가장 널리 쓰이는 선택은 운동 모델의 전이 확률(transition prior) 자체를 제안 분포로 쓰는 방식, 곧 $q(\mathbf{x}_k \mid \mathbf{x}_{k-1}, \mathbf{z}_k) = p(\mathbf{x}_k \mid \mathbf{x}_{k-1})$로 두는 방법이다. 이렇게 하면 입자마다 운동 모델에 프로세스 노이즈를 더해 한 스텝 전파시키기만 하면 되고, 분자·분모의 전이 확률이 서로 약분되어 가중치 갱신식이 극도로 단순해진다.

$$w_k^{(i)} \propto w_{k-1}^{(i)} \cdot p(\mathbf{z}_k \mid \mathbf{x}_k^{(i)})$$

결국 각 입자는 이전 가중치에 자신의 위치에서 평가한 관측 우도(likelihood)를 곱해 갱신한다. 이전 가중치가 같다면 방금 들어온 센서 측정과 잘 맞아떨어지는 입자일수록 더 큰 가중치를 받는다.

이론상 가중치 분산을 가장 작게 만드는 최적의 제안 분포는 $q^*(\mathbf{x}_k \mid \mathbf{x}_{k-1}^{(i)}, \mathbf{z}_k) = p(\mathbf{x}_k \mid \mathbf{x}_{k-1}^{(i)}, \mathbf{z}_k)$이지만, 실제 비선형 관측 모델에서는 이를 닫힌 식으로 구해 샘플링하기가 쉽지 않다.

### 4.3.3 Resampling

전파와 가중치 곱셈만 몇 단계 반복하면, 금세 가중치 대부분이 극소수 입자에게만 몰리고 나머지 입자들의 가중치는 0에 가까워지는 **가중치 퇴화(weight degeneracy)** 현상이 나타난다. 쓸모 있는 입자가 사실상 몇 개 남지 않게 되면 사후 분포 근사의 정확도가 급격히 무너진다.

퇴화가 얼마나 진행되었는지는 유효 입자 수(effective sample size)로 진단한다.

$$N_{\text{eff}} = \frac{1}{\sum_{i=1}^N (w_k^{(i)})^2}$$

모든 입자의 가중치가 $1/N$으로 고르면 $N_{\text{eff}} = N$이 되고, 가중치가 한 입자에 쏠리면 $N_{\text{eff}} \approx 1$로 떨어진다. 보통 $N_{\text{eff}}$가 임계값(흔히 $N/2$) 아래로 내려가면 리샘플링(resampling)을 수행한다.

**리샘플링**은 가중치가 큰 입자는 여러 개로 복제하고 가중치가 미미한 입자는 걸러낸 뒤, 살아남은 $N$개 입자의 가중치를 다시 $1/N$로 균등하게 초기화하는 단계다. 대표적인 리샘플링 기법 세 가지는 난수를 추출하는 방식에 따라 분산 특성이 갈린다.

- **Multinomial Resampling**: 현재 가중치를 이산 확률 분포로 보고 $N$번 독립적으로 복원 추출한다. 개념이 가장 단순하지만 추첨 편차(분산)가 크다.
- **Systematic Resampling**: 균등 난수 단 하나 $U_0 \sim \text{Uniform}(0, 1/N)$을 뽑은 뒤, $U_i = U_0 + (i-1)/N$ 간격으로 누적 분포(CDF)를 훑어 입자를 선택한다. 구현이 간단하고 계산량도 적으며 대개 multinomial 방식보다 분산이 작다. 다만 가중치 배열 형태에 따라서는 다른 기법보다 분산이 더 크게 나올 수도 있다(Douc, Cappé & Moulines, 2005).
- **Stratified Resampling**: Systematic처럼 $[0,1)$ 구간을 $N$등분하되, 각 소구간마다 독립적인 균등 난수를 하나씩 뽑아 입자를 고른다.

아래 파이썬 코드는 `systematic_resampling`과 이를 결합한 가장 기본적인 Bootstrap Particle Filter(SIR: Sampling Importance Resampling)를 구현한 예제다. 운동 모델의 전이 분포에서 입자를 전파하고, 가우시안 관측 우도로 가중치를 갱신한 뒤 $N_{\text{eff}} < N/2$일 때만 리샘플링을 발동하는 전형적인 흐름을 확인할 수 있다.

```python
import numpy as np

def systematic_resampling(weights, N):
    """체계적 리샘플링 (Systematic Resampling).
    
    Parameters
    ----------
    weights : ndarray, shape (N,) — 정규화된 가중치
    N : int — 리샘플링할 입자 수
    
    Returns
    -------
    indices : ndarray, shape (N,) — 선택된 입자의 인덱스
    """
    cumsum = np.cumsum(weights)
    u0 = np.random.uniform(0, 1.0 / N)
    u = u0 + np.arange(N) / N
    indices = np.searchsorted(cumsum, u)
    return indices

def bootstrap_particle_filter(f, h, Q, R, z_seq, N=1000, x0_sampler=None):
    """Bootstrap Particle Filter (SIR: Sampling Importance Resampling).
    
    제안 분포 = 전이 사전 (가장 기본적인 PF)
    
    Parameters
    ----------
    f : callable — 상태 전이 함수 f(x, noise) → x_next
    h : callable — 관측 함수 h(x) → z_predicted
    Q : ndarray — 프로세스 노이즈 공분산
    R : ndarray — 관측 노이즈 공분산
    z_seq : list of ndarray — 관측 시퀀스
    N : int — 입자 수
    x0_sampler : callable — 초기 입자 샘플러 (없으면 N(0,I))
    
    Returns
    -------
    x_est : list of ndarray — 각 시각의 가중 평균 추정치
    """
    n = Q.shape[0]
    m = R.shape[0]
    T = len(z_seq)
    
    # 초기화
    if x0_sampler:
        particles = np.array([x0_sampler() for _ in range(N)])  # (N, n)
    else:
        particles = np.random.randn(N, n)
    weights = np.ones(N) / N
    
    x_est = []
    L_Q = np.linalg.cholesky(Q)
    
    for k in range(T):
        # 1. 예측: 전이 모델로 입자 전파
        noise = (L_Q @ np.random.randn(n, N)).T  # (N, n)
        particles = np.array([f(particles[i], noise[i]) for i in range(N)])
        
        # 2. 가중치 갱신: 관측 likelihood
        for i in range(N):
            z_pred = h(particles[i])
            innovation = z_seq[k] - z_pred
            # 가우시안 likelihood
            log_w = -0.5 * innovation @ np.linalg.solve(R, innovation)
            weights[i] *= np.exp(log_w)
        
        # 정규화
        weights /= np.sum(weights)
        
        # 가중 평균 추정치
        x_est.append(np.average(particles, weights=weights, axis=0))
        
        # 3. 리샘플링 (유효 입자 수가 임계값 미만이면)
        N_eff = 1.0 / np.sum(weights ** 2)
        if N_eff < N / 2:
            indices = systematic_resampling(weights, N)
            particles = particles[indices]
            weights = np.ones(N) / N
    
    return x_est
```

순수 Bootstrap PF는 어떤 비선형 모델에도 곧바로 적용할 수 있지만, 상태 차원이 조금만 높아져도 필요한 입자 수가 폭증하는 약점이 있다. 이 부담을 덜기 위해 상태 가운데 해석적으로 다룰 수 있는 부분을 칼만 필터로 떼어내는 기법이 다음의 RBPF다.

### 4.3.4 Rao-Blackwellized Particle Filter (RBPF)

RBPF(Rao-Blackwellized Particle Filter)는 전체 상태를 둘로 쪼개어, 비선형적이거나 다봉인 핵심 부분만 입자로 샘플링하고 나머지 조건부 선형·가우시안 부분은 칼만 필터 같은 해석적 추정기에 맡기는 혼합 기법이다. 입자가 덮어야 할 차원이 줄어드니 훨씬 적은 입자만으로도 쓸 만한 근사를 얻는다.

상태를 $\mathbf{x} = [\mathbf{x}_1, \mathbf{x}_2]$로 나눴을 때, $\mathbf{x}_1$의 값이 고정되면 나머지 $\mathbf{x}_2$의 조건부 분포가 가우시안처럼 닫힌 식으로 추적된다고 하자. 그러면 결합 사후 분포를 다음과 같이 두 인수의 곱으로 인수분해할 수 있다.

$$p(\mathbf{x}_1, \mathbf{x}_2 \mid \mathbf{z}_{1:k}) = p(\mathbf{x}_2 \mid \mathbf{x}_1, \mathbf{z}_{1:k}) \cdot p(\mathbf{x}_1 \mid \mathbf{z}_{1:k})$$

- $p(\mathbf{x}_1 \mid \mathbf{z}_{1:k})$: 입자 필터로 비모수 근사
- $p(\mathbf{x}_2 \mid \mathbf{x}_1, \mathbf{z}_{1:k})$: 각 입자마다 딸려 있는 독립적인 칼만 필터로 해석적 추적

**Rao-Blackwell 정리**는 이렇게 조건부 해석 추정을 결합해 얻은 추정량의 분산이, 같은 수의 입자로 상태 전체를 샘플링하는 순수 Particle Filter의 분산보다 결코 크지 않음을 보장한다.

$$\text{Var}[\hat{\mathbf{x}}_{\text{RBPF}}] \leq \text{Var}[\hat{\mathbf{x}}_{\text{PF}}]$$

**FastSLAM**: FastSLAM(Montemerlo et al., 2002)은 이 아이디어를 SLAM에 적용한 대표적인 알고리즘이다. 로봇의 궤적(trajectory)만 입자로 샘플링하고, 각 입자 안에서는 랜드마크 위치들을 작은 EKF 여러 개로 나누어 추적한다.

- $\mathbf{x}_1 = \mathbf{x}_{0:k}^{\text{robot}}$ (로봇 경로) → 입자 필터
- $\mathbf{x}_2 = \{\mathbf{m}_1, \ldots, \mathbf{m}_M\}$ (랜드마크 집합) → 각 입자에 매달린 $M$개의 독립 2D EKF

로봇 경로가 주어지면 서로 다른 랜드마크들의 위치 추정이 조건부 독립이 된다는 성질을 이용한 설계다. 어떤 관측이 어느 랜드마크에 해당하는지 데이터 연관을 이미 안다고 놓았을 때, 맵을 균형 이진 트리로 관리하는 FastSLAM은 입자가 $N$개, 랜드마크가 $M$개일 때 한 번의 갱신 비용이 $O(N \log M)$에 그친다. 거대한 단일 공분산 행렬을 갱신해야 하는 EKF-SLAM의 $O(M^2)$과 비교하면 랜드마크가 많을수록 격차가 크게 벌어진다.

### 4.3.5 Particle Filter의 한계와 현재 위치

Particle Filter의 가장 근본적인 한계는 **차원의 저주(curse of dimensionality)**다. 상태 차원이 높아질수록 공간을 빈틈없이 메우는 데 필요한 입자 수가 기하급수적으로 불어난다. 3차원 VIO나 LIO의 상태 벡터만 해도 자세·위치·속도·자이로 바이어스·가속도계 바이어스까지 최소 15차원에 달한다. 정교한 제안 분포나 Rao-Blackwellization 없이 이 15차원 상태 전체를 순수 입자로 굴리면 실시간 처리가 사실상 불가능하다.

이런 특성 때문에 오늘날 로보틱스 현장에서 Particle Filter는 모든 추정을 도맡기보다 자신의 강점이 뚜렷한 세 영역에 주로 배치된다.

- **2D SLAM (RBPF 기반)**: GMapping처럼 로봇의 3-DoF 평면 자세만 입자로 두고, 각 입자가 저마다 2D 점유 격자 지도를 유지하는 구조.
- **Global Localization (MCL)**: 이미 지도를 갖고 있는 상태에서 로봇이 어디서 출발했는지 모를 때 유용하다. 닮은 복도가 여럿 있어 사후 분포에 봉우리가 여럿 솟아도 입자들이 자연스럽게 여러 가설을 동시에 유지하기 때문이다. 운용 도중 누군가 로봇을 들어 다른 곳으로 옮기는 상황(kidnapped robot problem)에 대비해 입자를 다시 퍼뜨리는 복구 로직을 함께 넣기도 한다.
- **저차원 비선형 추정**: 상태 차원은 낮지만 비선형성이 극심하거나 노이즈가 비가우시안인 특수 문제.

반면 3차원 공간의 고차원 상태 추정과 정밀한 궤적 복원에는 주로 앞서 본 Kalman 필터 계열(특히 ESKF)이나, 이제부터 살펴볼 비선형 최적화 기반 스무딩이 주력으로 쓰인다.

---

## 4.4 필터링과 스무딩

앞에서 다룬 칼만 필터와 파티클 필터는 모두 현재 시각의 상태만 남기고 과거 상태를 흘려보내는 재귀 필터다. 그런데 로봇이 한 번 지나간 뒤에 들어온 관측은 과거의 추정치까지 더 정확히 고쳐 쓸 수 있는 귀중한 단서다. 현대의 고정밀 VIO와 SLAM이 왜 단일 시점 필터링에서 여러 시점을 함께 푸는 스무딩(smoothing)과 비선형 최적화로 중심을 옮겼는지, 두 패러다임의 구조적 차이를 짚어 본다.

### 4.4.1 Filtering과 Smoothing의 차이

두 접근법의 차이는 조건부 확률에 어떤 시점까지의 관측을 넣는지에서 바로 드러난다.

**Filtering**은 현재 시각 $k$까지 들어온 관측만을 써서 지금 이 순간의 상태를 추정한다.
$$p(\mathbf{x}_k \mid \mathbf{z}_{1:k})$$

**Smoothing**은 시각 $k$ 이후에 들어온 관측까지 모두 동원해 과거 시점의 상태까지 함께 되돌아보며 추정한다.
$$p(\mathbf{x}_k \mid \mathbf{z}_{1:T}), \quad k < T$$

확률 모델이 정확하고 동일한 손실 함수를 쓴다면, 미래 관측까지 추가 정보로 조건지은 smoother의 평균 Bayes risk는 filter보다 클 수 없다. 물론 개별 궤적 하나의 실제 오차나 근사 최적화 결과에서는 smoother 쪽 오차가 더 크게 나오는 경우도 있다. 실시간 시스템에서는 용도와 지연 허용 범위에 따라 한 스텝씩 갱신하는 필터와 함께 incremental smoother나 fixed-lag smoother를 널리 쓰며, 전체 주행 데이터를 다 모은 뒤 최고 정밀도가 필요한 후처리에서는 batch smoother를 돌린다.

### 4.4.2 Fixed-Lag Smoother

실시간 주행에서는 주행이 다 끝날 때까지 기다릴 수 없다. Fixed-lag smoother는 현재 시각 $k$까지의 관측을 모두 반영하되, $L$단계 앞선 시각 $k-L$까지의 상태나 최근 $L$개 구간을 추정하는 타협점을 택한다.

$$p(\mathbf{x}_{k-L} \mid \mathbf{z}_{1:k})$$

결과 확정에 $L$단계만큼의 지연(latency)을 감수하거나 최근 창 안의 상태만 유지하는 대신, 그사이에 새로 들어온 관측으로 직전 상태들의 선형화 오차를 바로잡을 수 있어 순수 filtering과 full smoothing의 중간 역할을 맡는다.

VINS-Mono나 ORB-SLAM3 같은 슬라이딩 윈도우 기반 최적화 시스템도 이 fixed-lag smoothing 철학과 궤를 같이한다. 정해진 윈도우 안에 살아 있는 최근 키프레임들의 포즈와 특징점을 함께 다시 선형화하며 최적화하기 때문이다. 다만 필터와 슬라이딩 윈도우 최적화 가운데 어느 쪽이 더 정확한지를 공정하게 따지려면 계산 예산, 관측 모델, 평가 데이터셋을 동일하게 맞춘 비교가 필요하다.

### 4.4.3 Full Smoothing (Batch Optimization)

Full smoothing(batch optimization)은 전체 주행이 끝났을 때, 혹은 충분한 구간이 쌓였을 때 모든 관측 $\mathbf{z}_{1:T}$를 한꺼번에 조건으로 걸어 전체 궤적 $\mathbf{x}_{0:T}$의 결합 사후 분포를 추정하는 방식이다.

$$p(\mathbf{x}_{0:T} \mid \mathbf{z}_{1:T})$$

이 사후 확률을 최대화하는 단일 최적 궤적을 구하면 MAP(Maximum A Posteriori) 추정 문제가 된다.

$$\mathbf{x}_{0:T}^* = \arg\max_{\mathbf{x}_{0:T}} p(\mathbf{x}_{0:T} \mid \mathbf{z}_{1:T})$$

여기에 관측 노이즈와 운동 노이즈가 가우시안이라는 가정을 얹고 음의 로그를 취하면, 확률 최대화 문제가 곧바로 가중 잔차 제곱합을 줄이는 비선형 최소자승(Nonlinear Least Squares, NLS) 문제로 바뀐다. 이것이 §4.5에서 다룰 factor graph 기반 최적화의 출발점이다.

### 4.4.4 왜 현대 SLAM은 Filtering에서 Optimization으로 갔는가

2000년대 초반까지만 해도 SLAM 연구의 주류는 EKF-SLAM이었으나, 이후 점차 graph-based optimization(batch·incremental smoothing) 쪽으로 무게 중심이 이동했다. 그 배경에는 네 가지 결정적인 이유가 있다.

**1. 선형화 지점의 문제 (Linearization Point)**

기본 EKF는 시각 $k$의 관측이 들어왔을 때 그 시점의 추정치에서 자코비안을 딱 한 번 계산해 상태를 갱신한 뒤, 그 결정을 영구히 굳혀 버린다. 나중에 훨씬 정확한 관측이 들어와 궤적 추정치가 개선되더라도 이미 지나간 시점의 자코비안을 소급해 고치지 않는다. 반면 batch optimization은 반복 단계마다 최신 추정치 전체에서 자코비안을 다시 계산(relinearize)해 1차 근사 오차를 꾸준히 깎아 나간다.

실제로 [Strasdat et al. (2012) "Visual SLAM: Why Filter?"](https://doi.org/10.1016/j.imavis.2012.02.009)는 monocular와 stereo SLAM 실험에서 같은 계산 시간 예산이 주어졌을 때 프레임 수를 늘리기보다 한 프레임당 추적하는 특징점 수를 늘리는 편이 정확도 향상에 훨씬 효과적임을 보였다. 그런데 특징점 수를 $M$이라 할 때 bundle adjustment(최적화)의 계산 비용은 특징점 수에 선형으로 늘어나는 반면, 공분산 행렬 전체를 들고 다니는 EKF-SLAM은 3제곱에 비례해 폭증한다. 따라서 특징점을 많이 쓸수록 최적화 쪽이 압도적으로 유리해진다.

**2. 일관성(Consistency) 문제**

GPS 같은 절대 외부 기준이 없는 SLAM이나 VIO에서는 추정된 궤적과 지도를 전역 좌표계 채로 통째로 평행 이동하거나 회전시켜도 센서 관측값 자체는 전혀 변하지 않는다. 이렇게 관측만으로 결정되지 않는 전역 자유도를 **gauge freedom**이라 부르며, 이 방향은 시스템의 관측 불가능 방향(unobservable direction)이 된다. 차원은 문제마다 달라서 2D SLAM은 3개, 3D SLAM은 6개, 중력 방향이 관측되는 VIO는 전역 위치 3축과 중력축 둘레의 yaw 회전까지 모두 4개의 gauge freedom을 갖는다. 실제 구현에서는 첫 번째 포즈를 고정하는 식으로 이 자유도를 묶어 둔다.

문제는 EKF가 매 스텝 서로 다른 추정치에서 선형화를 반복할 때 불거진다. 선형화 지점이 달라지면 원래는 관측 불가능해야 할 gauge 방향으로 가짜 정보가 스며들어 필터가 실제보다 불확실성을 과소평가하게 되는데, 이를 필터가 일관성(consistency)을 잃었다고 한다. 이를 막으려면 선형화를 수행한 뒤에도 관측 불가능 방향이 그대로 영공간(nullspace)에 남도록 강제하는 FEJ(First-Estimate Jacobian)나 observability-constrained 기법을 써야 한다.

**3. 스케일러빌리티(Scalability)**

EKF-SLAM은 랜드마크가 $M$개일 때 랜드마크 사이의 상호 상관관계를 모두 저장해야 하므로 공분산 행렬이 빽빽해(dense)지고, 원소 수와 한 번 갱신하는 비용이 모두 $O((n + 3M)^2)$로 커진다. 반대로 graph-based SLAM이 다루는 정보 행렬(information matrix, 근사 Hessian)은 본질적으로 희소(sparse)하다. 각 포즈나 랜드마크가 실제로 관측이나 odometry로 직접 이어진 이웃 변수하고만 엮여 있어 행렬의 원소 대부분이 0이기 때문이다. 이 희소 구조를 살려 풀면 dense 행렬 연산보다 계산량을 극적으로 줄일 수 있다. 다만 실제로 얼마나 빨라지는지는 그래프의 연결 구조, 변수 소거(elimination) 순서, 그리고 변수 제거 과정에서 새로 생기는 비영(non-zero) 항인 fill-in이 얼마나 억제되는지에 따라 달라진다.

**4. Loop closure 처리**

로봇이 한참을 돌다가 예전에 지나왔던 장소로 되돌아왔을 때(loop closure), 필터로 과거 궤적 전체를 고치려면 지나온 모든 상태와 공분산 상관을 계속 유지하고 있어야 해 계산 부담이 크게 불어난다. 반면 그래프 기반 시스템에서는 두 시점의 포즈를 잇는 loop closure factor(제약 조건) 하나만 그래프에 꽂아 넣고 다시 최적화하면 오차가 전체 궤적에 자연스럽게 분배된다.

두 패러다임의 핵심 차이를 정리하면 아래 표와 같다.

| 관점 | Filtering (EKF) | Optimization (Graph) |
|------|----------------|---------------------|
| 선형화 | 기본 EKF는 갱신 시 한 번 | batch/smoother는 반복 재선형화 가능 |
| 과거 상태 | 기본 필터는 제거하지만 상태 증강 가능 | full smoothing은 유지, fixed-lag는 마지널라이즈 |
| Loop closure | 과거 상태·상관을 유지해야 함 | 새 factor를 추가한 뒤 영향 구간 갱신 |
| 행렬 구조 | 공분산은 대체로 dense | factor/Hessian의 희소성을 활용 가능 |
| 계산량 | 상태와 관측 구조에 의존 | 그래프 구조·소거 순서·fill-in에 의존 |
| 일관성 | 선형화·관측 가능성 처리가 필요 | 재선형화만으로 보장되지 않으며 gauge·마지널라이제이션 처리가 필요 |

그렇다고 필터가 완전히 자취를 감춘 것은 아니다. [MSCKF (Mourikis & Roumeliotis, 2007)](https://ieeexplore.ieee.org/document/4209642)와 OpenVINS는 슬라이딩 윈도우 안의 포즈만 유지하며 연산 자원이 빠듯한 플랫폼에서도 맹활약하고 있고, FAST-LIO2는 매니폴드 IEKF에 증분식 k-d 트리(ikd-tree)를 결합해 고속 LiDAR-관성 오도메트리를 구현했다. 필터와 최적화 가운데 어느 쪽이 더 우월한지는 단정할 수 없으며, 실제 비교는 동일한 데이터셋과 센서 구성, 계산 예산을 맞춘 실험으로 가려야 한다.

이제 관점을 최적화 쪽으로 돌려, 서로 다른 센서 측정을 그래프 위의 제약 조건으로 엮어 풀고 3차원 회전을 매니폴드 위에서 다루는 Factor Graph 기법을 자세히 살펴보자.

---

## 4.5 Factor Graph & Optimization

앞에서 최적화 기반 스무딩이 왜 유리한지 살펴보았다면, 이제 그 최적화 문제를 실제로 어떻게 표현하고 푸는지 들여다볼 차례다. Factor graph는 서로 다른 시각에 들어온 카메라·LiDAR·IMU·GPS 측정을 하나의 그래프로 엮어 희소 비선형 최소자승 문제로 바꾸어 주는 표준 프레임워크다.

### 4.5.1 Factor Graph 표현

Factor graph는 확률적 그래피컬 모델(probabilistic graphical model)의 한 갈래로, 추정하려는 **변수(variable) 노드**와 그 변수들 사이의 제약을 나타내는 **인자(factor) 노드**를 선으로 잇는 이분 그래프(bipartite graph)다. 전체 사후 확률은 그래프 안의 모든 factor를 곱한 형태에 비례한다.

$$p(\mathbf{X} \mid \mathbf{Z}) \propto \prod_{i} f_i(\mathbf{X}_i)$$

이 식의 각 요소는 다음을 나타낸다.
- $\mathbf{X} = \{\mathbf{x}_0, \mathbf{x}_1, \ldots, \mathbf{x}_T, \mathbf{l}_1, \ldots, \mathbf{l}_M\}$: 추정해야 할 변수 노드들의 집합 (로봇 포즈, 랜드마크 위치, 센서 바이어스 등)
- $f_i(\mathbf{X}_i)$: $i$번째 factor. 자신에게 연결된 변수 부분집합 $\mathbf{X}_i$를 받아, 그 값들이 관측 데이터나 사전 정보와 얼마나 잘 들어맞는지 음이 아닌 수로 돌려주는 포텐셜 함수(potential function)다
- $\mathbf{Z}$: 지금까지 수집한 모든 관측

요컨대 factor 하나가 곧 센서 관측 하나 또는 사전 제약 하나에 대응한다. 센서나 제약의 종류에 따라 그래프에 올라가는 factor의 연결 구조와 의미도 달라진다.

| Factor 유형 | 연결하는 변수 | 의미 |
|------------|-------------|------|
| Prior factor | $\mathbf{x}_0$ | 초기 상태 사전 분포 |
| Odometry factor | $\mathbf{x}_{k-1}, \mathbf{x}_k$ | 연속 포즈 간 상대 운동 |
| IMU preintegration factor | $\mathbf{x}_{i}, \mathbf{x}_{j}, \mathbf{v}_i, \mathbf{v}_j, \mathbf{b}_i$ | 키프레임 간 IMU 적분 |
| Vision factor | $\mathbf{x}_k, \mathbf{l}_m$ | 카메라에서 랜드마크 관측 |
| LiDAR factor | $\mathbf{x}_k$ | 포인트-투-플레인/에지 정합 |
| GPS factor | $\mathbf{x}_k$ | 절대 위치 관측 |
| Loop closure factor | $\mathbf{x}_i, \mathbf{x}_j$ | loop closure 상대 포즈 |

이 구조 덕분에 새로운 센서를 시스템에 추가할 때 기존 코드를 뜯어고칠 필요가 없다. 새 센서의 관측 모델에 맞는 factor만 하나 정의해 그래프에 꽂아 넣으면 나머지 factor들은 그대로 둔 채 통합 최적화가 돌아간다. 이러한 설계 특성을 factor graph의 **모듈성(modularity)**이라 한다.

### 4.5.2 MAP Inference = Nonlinear Least Squares

노이즈가 가우시안 분포를 따른다고 놓으면, 확률 추론 문제가 놀랄 만큼 깔끔한 최적화 문제로 바뀐다. 가우시안 오차 아래에서 각 factor는 다음과 같은 지수 함수 꼴을 띤다.

$$f_i(\mathbf{X}_i) \propto \exp\left(-\frac{1}{2} \|\mathbf{r}_i(\mathbf{X}_i)\|^2_{\boldsymbol{\Sigma}_i}\right)$$

여기서 $\mathbf{r}_i(\mathbf{X}_i)$는 예측과 측정의 차이를 나타내는 잔차(residual) 벡터이고, $\|\mathbf{r}\|^2_{\boldsymbol{\Sigma}} = \mathbf{r}^\top \boldsymbol{\Sigma}^{-1} \mathbf{r}$은 불확실성 공분산으로 가중치를 준 Mahalanobis 거리의 제곱이다. 예컨대 일반적인 센서 관측 factor라면 잔차와 공분산은 다음과 같이 잡는다.

$$\mathbf{r}_i = \mathbf{z}_i - h_i(\mathbf{X}_i), \quad \boldsymbol{\Sigma}_i = \mathbf{R}_i \text{ (관측 노이즈 공분산)}$$

이제 전체 사후 분포를 최대화하는 MAP 추정을 구하면 factor들의 곱을 최대화하는 문제와 같아진다.

$$\mathbf{X}^* = \arg\max_\mathbf{X} p(\mathbf{X} \mid \mathbf{Z}) = \arg\max_\mathbf{X} \prod_i f_i(\mathbf{X}_i)$$

곱셈을 덧셈으로 바꾸기 위해 양변에 자연로그를 취하고 부호를 뒤집어 최소화 문제로 바꾸면, 지수 함수가 벗겨지면서 다음과 같은 목적 함수를 얻는다.

$$\boxed{\mathbf{X}^* = \arg\min_\mathbf{X} \sum_i \|\mathbf{r}_i(\mathbf{X}_i)\|^2_{\boldsymbol{\Sigma}_i}}$$

이로써 확률적 MAP 추론이 각 관측 잔차의 Mahalanobis 제곱합을 가장 작게 만드는 **비선형 최소자승(Nonlinear Least Squares, NLS)** 문제로 변환되었다. 이 유도 과정과 그래프 기반 추론의 전체 체계는 [Dellaert & Kaess (2017)의 "Factor Graphs for Robot Perception"](https://doi.org/10.1561/2300000043)에 상세히 정리되어 있다.

### 4.5.3 Gauss-Newton Method

앞에서 얻은 비선형 최소자승 문제는 닫힌 해가 없으므로, 현재 추정치 주변에서 국소 선형화를 반복하며 해를 찾아가는 **Gauss-Newton(GN)** 방법으로 푼다. 먼저 잔차 함수 $\mathbf{r}_i$를 현재 추정치 $\mathbf{X}^{(0)}$ 근방에서 1차 테일러 전개한다.

$$\mathbf{r}_i(\mathbf{X}^{(0)} \boxplus \Delta\mathbf{X}) \approx \mathbf{r}_i(\mathbf{X}^{(0)}) + \mathbf{J}_i \Delta\mathbf{X}$$

여기서 $\mathbf{J}_i = \frac{\partial \mathbf{r}_i}{\partial \mathbf{X}}\big|_{\mathbf{X}^{(0)}}$는 잔차의 자코비안이고, $\boxplus$는 3차원 회전 같은 매니폴드 변수를 포함할 수 있는 증분 연산자다.

선형화한 잔차를 비용 함수에 대입하면 증분 $\Delta\mathbf{X}$에 대한 2차 근사 식을 얻는다.

$$\sum_i \|\mathbf{r}_i + \mathbf{J}_i \Delta\mathbf{X}\|^2_{\boldsymbol{\Sigma}_i}$$

여기에 공분산 역행렬의 제곱근을 곱해 $\mathbf{r}_i' = \boldsymbol{\Sigma}_i^{-1/2} \mathbf{r}_i$, $\mathbf{J}_i' = \boldsymbol{\Sigma}_i^{-1/2} \mathbf{J}_i$로 백색화(whitening)하면 가중치가 사라진 표준 최소자승 형태로 정리된다.

$$\sum_i \|\mathbf{r}_i' + \mathbf{J}_i' \Delta\mathbf{X}\|^2$$

이 2차 함수를 $\Delta\mathbf{X}$로 편미분해 0으로 놓으면 선형 연립방정식인 **정규 방정식(normal equation)**이 도출된다.

$$\underbrace{\left(\sum_i \mathbf{J}_i'^\top \mathbf{J}_i'\right)}_{\mathbf{H}} \Delta\mathbf{X} = -\underbrace{\sum_i \mathbf{J}_i'^\top \mathbf{r}_i'}_{\mathbf{b}}$$

$$\boxed{\mathbf{H} \Delta\mathbf{X} = -\mathbf{b}}$$

여기서 $\mathbf{H} = \mathbf{J}^\top \boldsymbol{\Sigma}^{-1} \mathbf{J} \in \mathbb{R}^{N \times N}$은 근사 Hessian(정보 행렬)이고, $\mathbf{b} = \mathbf{J}^\top \boldsymbol{\Sigma}^{-1} \mathbf{r}$은 비용 함수의 gradient에 해당한다.

이 식에서 factor graph가 왜 빠른지 결정적인 성질이 드러난다. factor $i$의 자코비안 $\mathbf{J}_i$에서 0이 아닌 열은 오직 그 factor에 실제로 연결된 소수 변수의 열뿐이다. 따라서 $\mathbf{J}_i'^\top \mathbf{J}_i'$를 합쳐 만든 $\mathbf{H}$에서도 0이 아닌 블록은 **같은 factor를 공유하는 변수 쌍**의 교차점에만 생긴다. SLAM의 factor 대부분은 두세 개 변수만 연결하므로, 행렬 $\mathbf{H}$는 거대하더라도 대부분이 0으로 채워진 희소 행렬이 된다.

Gauss-Newton은 정규 방정식을 풀어 얻은 보정량 $\Delta\mathbf{X}^{(k)}$를 현재 추정치에 반영하기를 수렴할 때까지 되풀이한다.

$$\mathbf{X}^{(k+1)} = \mathbf{X}^{(k)} \boxplus \Delta\mathbf{X}^{(k)}$$

각 반복에서 정규 방정식을 풀 때는 보통 $\mathbf{H} = \mathbf{L}\mathbf{L}^\top$처럼 희소 Cholesky 분해를 적용한 뒤 전방·후방 대입으로 해를 구한다. 이때 변수를 어떤 순서로 배열하느냐(variable ordering)에 따라 분해 인자 $\mathbf{L}$ 안에 원래 0이던 자리가 채워지는 fill-in의 양이 크게 달라지므로, 실전 솔버는 COLAMD 같은 근사 최소 차수 순서(approximate minimum degree ordering) 알고리즘으로 변수를 미리 재배치한다.

### 4.5.4 Levenberg-Marquardt Method

Gauss-Newton은 2차 미분 항을 버리고 $\mathbf{J}^\top\mathbf{J}$만으로 Hessian을 근사하므로, 초기 추정치가 참값에서 멀리 떨어져 있거나 잔차의 비선형성이 강하면 스텝이 너무 크게 튀어 오히려 비용이 늘거나 발산할 수 있다. **Levenberg-Marquardt(LM)** 방법은 정규 방정식의 대각 성분에 감쇠 항(damping term) $\lambda \mathbf{I}$를 더해 Gauss-Newton과 경사 하강법(gradient descent) 사이를 부드럽게 오간다.

$$(\mathbf{H} + \lambda \mathbf{I}) \Delta\mathbf{X} = -\mathbf{b}$$

감쇠 계수 $\lambda$의 크기에 따라 거동이 달라진다.
- $\lambda$가 작으면 → $\mathbf{H}\Delta\mathbf{X} \approx -\mathbf{b}$가 되어 Gauss-Newton에 가깝게 동작한다(해 근처의 조건이 좋으면 2차 수렴에 가깝게 빠르게 다가간다).
- $\lambda$가 크면 → $\Delta\mathbf{X} \approx -\frac{1}{\lambda}\mathbf{b}$가 되어 gradient 반대 방향으로 조심스럽게 한 걸음씩 내딛는 경사 하강법에 가까워진다.

매 반복마다 $\lambda$는 갱신 결과의 성패를 보고 동적으로 조절한다. 시험 삼아 계산한 스텝으로 실제 비용이 줄어들면 근사가 잘 맞았다는 뜻이므로 $\lambda$를 낮춰 Gauss-Newton 쪽으로 속도를 높이고, 반대로 비용이 늘어나면 그 스텝을 기각한 뒤 $\lambda$를 키워 보폭을 줄이고 다시 시도한다.

### 4.5.5 매니폴드 위의 최적화 (Optimization on Manifolds)

로봇의 3D 포즈 $\mathbf{T} \in SE(3)$는 평평한 유클리드 공간이 아니라 Lie 군 매니폴드의 원소다. 만약 정규 방정식에서 나온 증분을 $4 \times 4$ 변환 행렬에 그대로 더하면 회전 부분의 직교성($\mathbf{R}^\top \mathbf{R} = \mathbf{I}$)이 즉시 깨진다. 따라서 매니폴드 위에서는 덧셈 대신 현재 추정치 $\mathbf{T}^{(k)}$의 평평한 접선 공간(tangent space)에서 6차원 증분 $\boldsymbol{\xi} \in \mathbb{R}^6$을 구한 뒤, retraction이나 exponential map으로 다시 매니폴드 위로 되돌려 놓는 방식을 쓴다.

접선 공간의 증분을 오른쪽에 곱해 갱신하는 우측 섭동(right perturbation) 규약은 다음과 같다.

$$\mathbf{T}^{(k+1)} = \mathbf{T}^{(k)} \cdot \text{Exp}(\boldsymbol{\xi})$$

반대로 왼쪽에 곱하는 좌측 섭동(left perturbation) 규약도 쓸 수 있다.

$$\mathbf{T}^{(k+1)} = \text{Exp}(\boldsymbol{\xi}) \cdot \mathbf{T}^{(k)}$$

어느 쪽에 곱할지는 일관성만 유지하면 되며, 사용하는 라이브러리나 수식 유도에서 정한 규약을 따르면 된다.

여기서 $\text{Exp}: \mathbb{R}^6 \to SE(3)$는 Lie 대수 벡터를 Lie 군 행렬로 보내는 exponential map이다. $\boldsymbol{\xi} = [\boldsymbol{\rho}^\top, \boldsymbol{\phi}^\top]^\top$ 가운데 $\boldsymbol{\rho} \in \mathbb{R}^3$는 병진 이동 성분, $\boldsymbol{\phi} \in \mathbb{R}^3$는 회전의 각-축(angle-axis) 벡터다. 특히 $SO(3)$의 exponential map은 다음과 같이 Rodrigues 공식으로 닫힌 형태로 계산된다.

$$\text{Exp}(\boldsymbol{\phi}) = \mathbf{I} + \frac{\sin\theta}{\theta}[\boldsymbol{\phi}]_\times + \frac{1 - \cos\theta}{\theta^2}[\boldsymbol{\phi}]_\times^2 \in \mathbb{R}^{3 \times 3}$$

여기서 $\theta = \|\boldsymbol{\phi}\|$는 회전 각도(라디안)다.

그 역연산인 $\text{Log}: SE(3) \to \mathbb{R}^6$은 매니폴드 원소 사이의 차이를 다시 6차원 접선 공간 벡터로 끌어내리는 logarithmic map이다. 정리하면, Gauss-Newton이나 LM의 선형 방정식은 언제나 제약 없는 접선 공간의 최소 자유도 벡터 $\boldsymbol{\xi}$로 풀고, 구해진 해만 $\text{Exp}$를 거쳐 $\mathbf{T}$에 반영한다. 이에 맞춰 잔차의 자코비안 역시 접선 공간 증분 $\boldsymbol{\xi} = \mathbf{0}$에서 평가한다.

$$\mathbf{J}_i = \frac{\partial \mathbf{r}_i}{\partial \boldsymbol{\xi}}\bigg|_{\boldsymbol{\xi}=\mathbf{0}}$$

### 4.5.6 iSAM2: Incremental Smoothing

실제 주행 중에는 새 키프레임과 관측이 계속 들어온다. 키프레임이 추가될 때마다 처음부터 끝까지 batch 최적화를 매번 새로 돌리면 궤적이 길어질수록 연산 시간이 감당할 수 없이 늘어난다. **iSAM2**([Kaess et al., 2012](https://doi.org/10.1177/0278364911430419))는 factor graph를 매번 통째로 다시 푸는 대신, **Bayes tree**라는 자료구조를 도입해 새 관측의 영향을 받는 부분만 증분식으로 갱신한다.

iSAM2를 떠받치는 네 가지 핵심 장치는 다음과 같다.

1. **Bayes Tree**: Factor graph에 변수 소거(variable elimination)를 수행한 결과를 방향성 트리로 옮겨 담은 자료구조다. 트리의 각 노드는 함께 묶인 변수 집합(clique)의 조건부 확률 밀도를 저장한다.
2. **Incremental Update**: 새 factor가 들어오면 그래프 전체를 다시 분해하지 않고, 새 factor가 건드린 clique에서 트리의 루트까지 올라가는 경로만 다시 계산하고 나머지 트리는 그대로 재사용한다.
3. **Fluid Relinearization**: 모든 변수를 매번 재선형화하는 대신, 현재 추정치가 직전 선형화 지점에서 미리 정한 임계값 이상 멀어진 변수만 골라내 선형화를 갱신한다. 덕분에 주기적인 전체 batch 재계산 없이도 비선형 오차를 제어한다.
4. **변수 순서 재정렬(Variable Reordering)**: 새 변수나 factor가 추가될 때 영향받는 트리 부분의 소거 순서만 국소적으로 다시 정리해 fill-in을 억제한다.

GTSAM 라이브러리에 iSAM2가 기본 탑재되어 있으며, 대표적으로 [LIO-SAM](https://arxiv.org/abs/2007.00258)이 이를 백엔드 추정기로 사용한다. 반면 VINS-Mono는 전체 궤적의 Bayes tree 대신 고정 길이 슬라이딩 윈도우 안의 키프레임을 Ceres로 푸는 방식을 택한다.

> **Continuous-Time Factor Graph**: 키프레임이 찍힌 이산 시각에만 상태 변수를 놓는 대신, 로봇의 궤적 자체를 시간에 따른 연속 함수로 모델링하는 factor graph도 연구되고 있다. 예를 들어 [Burnett et al. (2025)](https://arxiv.org/abs/2402.06174)는 Gaussian Process motion prior를 결합한 연속 시간 추정 프레임워크로 radar-inertial과 LiDAR-inertial odometry를 구현했다.

### 4.5.7 GTSAM / Ceres / g2o 비교

실제 프로젝트에서 factor graph 최적화를 구현할 때는 직접 솔버를 짜기보다 검증된 C++ 최적화 라이브러리를 쓰는 것이 일반적이다. 로보틱스에서 가장 널리 쓰이는 세 라이브러리—GTSAM, Ceres Solver, g2o—의 설계 철학과 지원 기능을 정리하면 다음과 같다.

| 특성 | GTSAM | Ceres Solver | g2o |
|------|-------|-------------|-----|
| 개발 | Georgia Tech ([Dellaert](https://gtsam.org/)) | Google ([Ceres](http://ceres-solver.org/)) | [Kümmerle et al.](https://doi.org/10.1109/ICRA.2011.5979949) |
| 설계 | Factor graph + Bayes tree | 범용 NLS solver | Graph optimization |
| Incremental | iSAM2 (native) | 없음 (batch) | 예제(g2o_incremental)로 제공 |
| 매니폴드 | 내장 (Rot2, Rot3, Pose2, Pose3, ...) | `Manifold` (2.1에서 도입, LocalParameterization은 2.2에서 제거) | 내장 |
| IMU Preintegration | 내장 (`PreintegratedImuMeasurements`) | 사용자 정의 | 사용자 정의 |
| 자동 미분 | Expression 기반 지원 (수치 미분도 가능) | 자동 미분 (ceres::AutoDiffCostFunction) | 기본은 수치 미분, 자동 미분도 지원 |
| 언어 | C++ (Python 바인딩) | C++ | C++ |
| 대표 사용처 | LIO-SAM | Cartographer, VINS-Mono | ORB-SLAM3, 많은 SLAM 시스템의 포즈 그래프 |
| 학습 곡선 | Factor 정의만 하면 됨 | Cost function 정의 | Vertex/Edge 정의 |

세 라이브러리 가운데 GTSAM은 Python 바인딩을 공식 지원해 프로토타이핑에 편리하다. 아래 예제는 한 변이 2m인 정사각형 경로를 돌며 odometry factor를 쌓은 뒤, 출발점 `x0`과 마지막 포즈 `x4`를 잇는 loop closure factor를 추가해 `iSAM2`로 오차를 보정하는 간단한 2D pose graph 최적화 코드다.

```python
# GTSAM으로 푸는 간단한 Pose Graph Optimization 예제
import gtsam
import numpy as np

# 1. Factor graph 생성
graph = gtsam.NonlinearFactorGraph()

# 2. 초기 추정치
initial = gtsam.Values()

# 3. Prior factor on first pose
prior_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.3, 0.3, 0.1]))  # (x, y, theta)
graph.add(gtsam.PriorFactorPose2(0, gtsam.Pose2(0.0, 0.0, 0.0), prior_noise))
initial.insert(0, gtsam.Pose2(0.0, 0.0, 0.0))

# 4. Odometry factors (연속 포즈 간)
odom_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.2, 0.2, 0.1]))

# 사각형 경로: 4개 포즈 (각 변에서 전진 2m + 좌회전 90도)
odometry = [
    gtsam.Pose2(2.0, 0.0, np.pi / 2),   # x0 → x1: 전진 2m + 좌회전 90도
    gtsam.Pose2(2.0, 0.0, np.pi / 2),   # x1 → x2: 전진 2m + 좌회전 90도
    gtsam.Pose2(2.0, 0.0, np.pi / 2),   # x2 → x3: 전진 2m + 좌회전 90도
    gtsam.Pose2(2.0, 0.0, np.pi / 2),   # x3 → x4: 전진 2m + 좌회전 90도
]

# Odometry factor 추가 및 초기값 설정
pose = gtsam.Pose2(0.0, 0.0, 0.0)
for i, odom in enumerate(odometry):
    graph.add(gtsam.BetweenFactorPose2(i, i + 1, odom, odom_noise))
    pose = pose.compose(odom)
    # odometry 누적값을 초기값으로 사용
    initial.insert(i + 1, pose)

# 5. Loop closure factor: x4와 x0이 같은 위치 (사각형 경로가 닫힘)
loop_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.1, 0.1, 0.05]))
graph.add(gtsam.BetweenFactorPose2(4, 0, gtsam.Pose2(0.0, 0.0, 0.0), loop_noise))

# 6. iSAM2로 최적화
params = gtsam.ISAM2Params()
isam = gtsam.ISAM2(params)
isam.update(graph, initial)
result = isam.calculateEstimate()

# 결과 출력
for i in range(5):
    pose_i = result.atPose2(i)
    print(f"Pose {i}: x={pose_i.x():.3f}, y={pose_i.y():.3f}, theta={pose_i.theta():.3f}")
```

이 예제처럼 두 포즈 사이를 잇는 odometry factor는 간단히 그래프에 올릴 수 있다. 그런데 카메라나 LiDAR와 함께 200~1000Hz로 쏟아지는 고주파 IMU 측정을 factor graph에 집어넣으려면 문제가 달라진다. IMU 샘플마다 포즈 노드를 만들면 그래프가 순식간에 비대해지고, 그렇다고 두 키프레임 사이를 매번 처음부터 다시 적분하자니 최적화 속도가 떨어지기 때문이다. 이 병목을 푸는 핵심 기법이 다음 절의 IMU Preintegration이다.

---

## 4.6 IMU Preintegration

IMU를 카메라나 LiDAR와 함께 factor graph에 얹으려면 두 센서의 극단적인 속도 차이를 해결해야 한다. 그 해법이 키프레임 사이에 쏟아지는 수많은 IMU 측정을 하나의 상대 운동 제약으로 미리 묶어 두는 IMU preintegration이다.

### 4.6.1 왜 Preintegration이 필요한가

예컨대 IMU가 200~1000Hz로 각속도와 가속도를 측정하는 동안 카메라나 LiDAR 키프레임은 10~30Hz로 들어온다면, 두 키프레임 $i$와 $j$ 사이에는 대략 7개에서 100개에 달하는 IMU 측정이 끼어 있게 된다.

**Naive 접근: 직접 적분**

두 키프레임 사이의 IMU 측정을 전역 좌표계에서 그대로 적분해 시각 $j$의 자세·속도·위치를 쓰면 다음과 같다.

$$\mathbf{R}_j = \mathbf{R}_i \prod_{k=i}^{j-1} \text{Exp}((\tilde{\boldsymbol{\omega}}_k - \mathbf{b}_g) \Delta t)$$
$$\mathbf{v}_j = \mathbf{v}_i + \mathbf{g} \Delta t_{ij} + \sum_{k=i}^{j-1} \mathbf{R}_k (\tilde{\mathbf{a}}_k - \mathbf{b}_a) \Delta t$$
$$\mathbf{p}_j = \mathbf{p}_i + \sum_{k=i}^{j-1}\left[\mathbf{v}_k \Delta t + \frac{1}{2}\left(\mathbf{R}_k(\tilde{\mathbf{a}}_k - \mathbf{b}_a) + \mathbf{g}\right)\Delta t^2\right]$$

문제는 이 적분 결과가 시작 키프레임 $i$의 전역 상태 $(\mathbf{R}_i, \mathbf{v}_i, \mathbf{p}_i)$와 바이어스 $(\mathbf{b}_g, \mathbf{b}_a)$에 직접 매여 있다는 점이다. 비선형 최적화는 반복 단계마다 키프레임 상태와 바이어스 추정치를 조금씩 고치는데, 그럴 때마다 중간 구간의 모든 적분과 exponential map을 수백 번씩 처음부터 다시 계산해야 한다면 실시간 처리가 어려워진다.

**Preintegration의 해법**: 적분 식을 전역 좌표계 대신 키프레임 $i$의 body frame 기준 상대 변화량으로 다시 쓰면, 적분 결과 자체가 키프레임 $i$의 전역 포즈·속도와 완전히 독립이 된다. 그리고 최적화 중에 바이어스 추정치가 바뀌더라도, 미리 구해 둔 1차 자코비안으로 측정값만 보정해 재적분을 피해 간다.

### 4.6.2 On-Manifold Preintegration 유도

이제 [Forster et al. (2017)](https://doi.org/10.1109/TRO.2016.2597321)가 정립한 on-manifold preintegration을 다섯 단계로 나누어 유도한다. 논문 전문은 [arXiv:1512.02363](https://arxiv.org/abs/1512.02363)에서도 확인할 수 있다.

#### Step 1: 상대 변화량 정의

앞선 전역 프레임 적분식에서 키프레임 $i$의 상태 $\mathbf{R}_i, \mathbf{v}_i, \mathbf{p}_i$와 중력 항을 모두 좌변으로 넘기고 양변에 $\mathbf{R}_i^\top$를 곱해 정리하면, 키프레임 $i$의 body frame에서 바라본 상대 회전·상대 속도·상대 위치 변화량 세 가지를 얻는다.

$$\Delta\mathbf{R}_{ij} \triangleq \mathbf{R}_i^\top \mathbf{R}_j = \prod_{k=i}^{j-1} \text{Exp}((\tilde{\boldsymbol{\omega}}_k - \mathbf{b}_g^i)\Delta t) \in SO(3)$$

$$\Delta\mathbf{v}_{ij} \triangleq \mathbf{R}_i^\top (\mathbf{v}_j - \mathbf{v}_i - \mathbf{g}\Delta t_{ij}) = \sum_{k=i}^{j-1} \Delta\mathbf{R}_{ik}(\tilde{\mathbf{a}}_k - \mathbf{b}_a^i)\Delta t \in \mathbb{R}^3$$

$$\Delta\mathbf{p}_{ij} \triangleq \mathbf{R}_i^\top (\mathbf{p}_j - \mathbf{p}_i - \mathbf{v}_i \Delta t_{ij} - \frac{1}{2}\mathbf{g}\Delta t_{ij}^2) = \sum_{k=i}^{j-1}\left[\Delta\mathbf{v}_{ik}\Delta t + \frac{1}{2}\Delta\mathbf{R}_{ik}(\tilde{\mathbf{a}}_k - \mathbf{b}_a^i)\Delta t^2\right] \in \mathbb{R}^3$$

여기서 결정적인 점을 눈여겨보자. **우변에는 오직 IMU 원시 측정값과 바이어스 추정치만 들어 있을 뿐**, 키프레임 $i$의 전역 포즈나 속도 $(\mathbf{R}_i, \mathbf{v}_i, \mathbf{p}_i)$가 전혀 등장하지 않는다. 따라서 최적화 과정에서 포즈와 속도가 아무리 바뀌어도 우변의 적분값을 다시 계산할 필요가 없다.

#### Step 2: 재귀적 계산 (On-Manifold)

실시간 파이프라인에서는 두 키프레임 사이의 IMU 측정을 한꺼번에 모아서 적분하지 않고, IMU 샘플이 도착할 때마다 재귀적으로 누적해 둔다.

$$\Delta\mathbf{R}_{i,k+1} = \Delta\mathbf{R}_{ik} \cdot \text{Exp}((\tilde{\boldsymbol{\omega}}_k - \mathbf{b}_g)\Delta t) \in SO(3)$$
$$\Delta\mathbf{v}_{i,k+1} = \Delta\mathbf{v}_{ik} + \Delta\mathbf{R}_{ik}(\tilde{\mathbf{a}}_k - \mathbf{b}_a)\Delta t \in \mathbb{R}^3$$
$$\Delta\mathbf{p}_{i,k+1} = \Delta\mathbf{p}_{ik} + \Delta\mathbf{v}_{ik}\Delta t + \frac{1}{2}\Delta\mathbf{R}_{ik}(\tilde{\mathbf{a}}_k - \mathbf{b}_a)\Delta t^2 \in \mathbb{R}^3$$

적분을 시작하는 시각 $k = i$에서는 항등 원소와 영벡터($\Delta\mathbf{R}_{ii} = \mathbf{I}_{3\times 3}$, $\Delta\mathbf{v}_{ii} = \mathbf{0}$, $\Delta\mathbf{p}_{ii} = \mathbf{0}$)로 초기화한다. "On-manifold"라는 이름은 회전 누적값 $\Delta\mathbf{R}_{ij}$를 오일러 각으로 풀지 않고 $SO(3)$ 매니폴드 위에서 직접 행렬 곱(또는 단위 쿼터니언 곱)으로 쌓아 올린다는 뜻이다.

#### Step 3: 바이어스 변화의 1차 보정

최적화가 진행되면서 바이어스 추정치가 $\mathbf{b} \to \mathbf{b} + \delta\mathbf{b}$만큼 수정되면, 원래 바이어스로 적분해 둔 preintegrated 측정값도 미세하게 달라진다. 이때도 구간 전체를 재적분하는 대신, 미리 계산해 둔 바이어스 자코비안을 곱하는 1차 테일러 근사로 즉시 보정한다.

**회전 보정**:
$$\Delta\hat{\mathbf{R}}_{ij}(\mathbf{b}_g + \delta\mathbf{b}_g) \approx \Delta\hat{\mathbf{R}}_{ij}(\mathbf{b}_g) \cdot \text{Exp}\left(\frac{\partial \Delta\bar{\mathbf{R}}_{ij}}{\partial \mathbf{b}_g} \delta\mathbf{b}_g\right)$$

**속도 보정**:
$$\Delta\hat{\mathbf{v}}_{ij}(\mathbf{b} + \delta\mathbf{b}) \approx \Delta\hat{\mathbf{v}}_{ij}(\mathbf{b}) + \frac{\partial \Delta\bar{\mathbf{v}}_{ij}}{\partial \mathbf{b}_g} \delta\mathbf{b}_g + \frac{\partial \Delta\bar{\mathbf{v}}_{ij}}{\partial \mathbf{b}_a} \delta\mathbf{b}_a$$

**위치 보정**:
$$\Delta\hat{\mathbf{p}}_{ij}(\mathbf{b} + \delta\mathbf{b}) \approx \Delta\hat{\mathbf{p}}_{ij}(\mathbf{b}) + \frac{\partial \Delta\bar{\mathbf{p}}_{ij}}{\partial \mathbf{b}_g} \delta\mathbf{b}_g + \frac{\partial \Delta\bar{\mathbf{p}}_{ij}}{\partial \mathbf{b}_a} \delta\mathbf{b}_a$$

이 보정에 쓰이는 $\frac{\partial \Delta\bar{\mathbf{R}}_{ij}}{\partial \mathbf{b}_g}$ 같은 자코비안 역시 Step 2의 측정 누적과 동시에 재귀적으로 갱신한다. 그중 회전의 자이로 바이어스 자코비안은 다음과 같이 전파된다.

$$\frac{\partial \Delta\bar{\mathbf{R}}_{i,k+1}}{\partial \mathbf{b}_g} = \Delta\bar{\mathbf{R}}_{k,k+1}^\top \frac{\partial \Delta\bar{\mathbf{R}}_{ik}}{\partial \mathbf{b}_g} - \text{Jr}((\tilde{\boldsymbol{\omega}}_k - \mathbf{b}_g)\Delta t) \Delta t$$

여기서 $\text{Jr}(\boldsymbol{\phi})$는 $\text{Exp}$ 사상을 섭동했을 때 나타나는 $SO(3)$의 right Jacobian이다.

$$\text{Jr}(\boldsymbol{\phi}) = \mathbf{I} - \frac{1 - \cos\theta}{\theta^2}[\boldsymbol{\phi}]_\times + \frac{\theta - \sin\theta}{\theta^3}[\boldsymbol{\phi}]_\times^2, \quad \theta = \|\boldsymbol{\phi}\|$$

바이어스 수정량 $\|\delta\mathbf{b}\|$가 작을 때는 이 1차 선형 보정만으로 충분히 정확하다. 다만 바이어스 변화가 구현에서 정한 임계값을 넘어설 만큼 커지면, 저장해 두었던 원시 IMU 측정을 꺼내 preintegration을 새로 계산한다.

#### Step 4: 공분산 전파

각 IMU 샘플에 섞여 있는 측정 노이즈 $\mathbf{n}_g, \mathbf{n}_a \sim \mathcal{N}(\mathbf{0}, \boldsymbol{\Sigma})$ 역시 적분을 거치며 누적 측정값의 불확실성으로 번진다. $9 \times 9$ 공분산 행렬 $\boldsymbol{\Sigma}_k$도 측정과 함께 매 샘플 재귀적으로 키운다.

$$\boldsymbol{\Sigma}_{k+1} = \mathbf{A}_k \boldsymbol{\Sigma}_k \mathbf{A}_k^\top + \mathbf{B}_k \boldsymbol{\Sigma}_\eta \mathbf{B}_k^\top$$

여기서 $\mathbf{A}_k$는 이전 시점 오차의 전파 자코비안, $\mathbf{B}_k$는 현재 샘플 노이즈의 입력 자코비안, $\boldsymbol{\Sigma}_\eta$는 IMU 센서 노이즈 공분산이다. 최종 구간 끝에서 얻은 $\boldsymbol{\Sigma}_{ij}$의 역행렬이 곧 factor graph 안 IMU factor의 정보 행렬이 된다.

#### Step 5: Factor Graph의 IMU Factor

이렇게 적분과 공분산 전파를 마친 preintegrated 측정은 두 키프레임 $i, j$의 포즈·속도와 시작점 바이어스를 하나로 묶는 IMU factor로 올라간다. 이 factor가 평가하는 9차원 잔차는 다음과 같다.

$$\mathbf{r}_{\Delta\mathbf{R}_{ij}} = \text{Log}\left(\Delta\hat{\mathbf{R}}_{ij}^\top \mathbf{R}_i^\top \mathbf{R}_j\right) \in \mathbb{R}^3$$

$$\mathbf{r}_{\Delta\mathbf{v}_{ij}} = \mathbf{R}_i^\top(\mathbf{v}_j - \mathbf{v}_i - \mathbf{g}\Delta t_{ij}) - \Delta\hat{\mathbf{v}}_{ij} \in \mathbb{R}^3$$

$$\mathbf{r}_{\Delta\mathbf{p}_{ij}} = \mathbf{R}_i^\top(\mathbf{p}_j - \mathbf{p}_i - \mathbf{v}_i\Delta t_{ij} - \frac{1}{2}\mathbf{g}\Delta t_{ij}^2) - \Delta\hat{\mathbf{p}}_{ij} \in \mathbb{R}^3$$

최종 최적화 목적 함수에는 이 9차원 잔차 벡터 $\mathbf{r} = [\mathbf{r}_{\Delta\mathbf{R}_{ij}}^\top, \mathbf{r}_{\Delta\mathbf{v}_{ij}}^\top, \mathbf{r}_{\Delta\mathbf{p}_{ij}}^\top]^\top$의 Mahalanobis 거리 제곱 $\|\mathbf{r}\|^2_{\boldsymbol{\Sigma}_{ij}}$이 더해진다.

### 4.6.3 코드로 보는 Preintegration

아래 파이썬 코드는 방금 유도한 on-manifold preintegration의 전체 계산 흐름을 그대로 옮긴 구현이다. 먼저 `skew`, `exp_so3`, `log_so3`, `right_jacobian_so3`로 $SO(3)$ 리 군 연산을 준비하고, `IMUPreintegration` 클래스 안에서 Step 2~5의 상태 누적·바이어스 자코비안 갱신·공분산 전파·잔차 계산을 차례로 수행한다. 코드 하단의 `create_imu_factor_gtsam`은 동일한 과정을 GTSAM의 내장 API로 작성했을 때 어떤 인터페이스로 호출하는지 대조해 보여 준다.

```python
import numpy as np
from scipy.spatial.transform import Rotation

def skew(v):
    """3D 벡터 → 반대칭 행렬 (skew-symmetric matrix).
    
    Parameters
    ----------
    v : ndarray, shape (3,)
    
    Returns
    -------
    S : ndarray, shape (3, 3)
    """
    return np.array([[0, -v[2], v[1]],
                     [v[2], 0, -v[0]],
                     [-v[1], v[0], 0]])

def exp_so3(phi):
    """so(3) → SO(3) exponential map (Rodrigues' formula).
    
    Parameters
    ----------
    phi : ndarray, shape (3,) — 각축(angle-axis) 벡터
    
    Returns
    -------
    R : ndarray, shape (3, 3) — 회전 행렬
    """
    theta = np.linalg.norm(phi)
    if theta < 1e-10:
        return np.eye(3) + skew(phi)
    axis = phi / theta
    K = skew(axis)
    return np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * K @ K

def log_so3(R):
    """SO(3) → so(3) logarithmic map.
    
    Parameters
    ----------
    R : ndarray, shape (3, 3) — 회전 행렬
    
    Returns
    -------
    phi : ndarray, shape (3,) — 각축(angle-axis) 벡터
    """
    cos_theta = (np.trace(R) - 1) / 2
    cos_theta = np.clip(cos_theta, -1, 1)
    theta = np.arccos(cos_theta)
    if theta < 1e-10:
        return np.array([R[2,1] - R[1,2], R[0,2] - R[2,0], R[1,0] - R[0,1]]) / 2
    return theta / (2 * np.sin(theta)) * np.array([R[2,1] - R[1,2],
                                                    R[0,2] - R[2,0],
                                                    R[1,0] - R[0,1]])

def right_jacobian_so3(phi):
    """SO(3)의 right Jacobian.
    
    Parameters
    ----------
    phi : ndarray, shape (3,)
    
    Returns
    -------
    Jr : ndarray, shape (3, 3)
    """
    theta = np.linalg.norm(phi)
    if theta < 1e-10:
        return np.eye(3) - 0.5 * skew(phi)
    K = skew(phi)
    return (np.eye(3) 
            - (1 - np.cos(theta)) / theta**2 * K 
            + (theta - np.sin(theta)) / theta**3 * K @ K)

class IMUPreintegration:
    """On-manifold IMU preintegration (Forster et al., 2017).
    
    두 키프레임 사이의 IMU 측정을 preintegrate하여 
    factor graph에서 IMU factor로 사용할 수 있는 형태로 만든다.
    """
    def __init__(self, bias_gyro, bias_acc, 
                 gyro_noise_density, acc_noise_density,
                 gyro_random_walk, acc_random_walk):
        """
        Parameters
        ----------
        bias_gyro : ndarray, shape (3,) — 자이로 바이어스 초기 추정치 [rad/s]
        bias_acc : ndarray, shape (3,) — 가속도 바이어스 초기 추정치 [m/s^2]
        gyro_noise_density : float — 자이로 노이즈 밀도 [rad/s/sqrt(Hz)]
        acc_noise_density : float — 가속도 노이즈 밀도 [m/s^2/sqrt(Hz)]
        gyro_random_walk : float — 자이로 바이어스 random walk [rad/s^2/sqrt(Hz)]
        acc_random_walk : float — 가속도 바이어스 random walk [m/s^3/sqrt(Hz)]
        """
        self.bg = bias_gyro.copy()
        self.ba = bias_acc.copy()
        
        # Preintegrated 측정 (초기값)
        self.delta_R = np.eye(3)     # SO(3) 상의 상대 회전
        self.delta_v = np.zeros(3)   # 상대 속도 변화
        self.delta_p = np.zeros(3)   # 상대 위치 변화
        self.dt_sum = 0.0            # 총 적분 시간
        
        # 공분산 (9x9: rotation, velocity, position)
        self.cov = np.zeros((9, 9))
        
        # 바이어스 자코비안 (바이어스 보정용)
        self.d_R_d_bg = np.zeros((3, 3))
        self.d_v_d_bg = np.zeros((3, 3))
        self.d_v_d_ba = np.zeros((3, 3))
        self.d_p_d_bg = np.zeros((3, 3))
        self.d_p_d_ba = np.zeros((3, 3))
        
        # 노이즈 파라미터
        self.sigma_g = gyro_noise_density
        self.sigma_a = acc_noise_density
        self.sigma_bg = gyro_random_walk
        self.sigma_ba = acc_random_walk
        
    def integrate(self, gyro_meas, acc_meas, dt):
        """하나의 IMU 측정을 preintegration에 추가한다.
        
        Parameters
        ----------
        gyro_meas : ndarray, shape (3,) — 자이로 측정 [rad/s]
        acc_meas : ndarray, shape (3,) — 가속도 측정 [m/s^2]
        dt : float — 시간 간격 [s]
        """
        # 바이어스 보정된 측정
        omega = gyro_meas - self.bg   # (3,)
        acc = acc_meas - self.ba       # (3,)
        
        # 중간 변수
        dR = exp_so3(omega * dt)       # (3, 3)  이 dt 동안의 회전
        Jr = right_jacobian_so3(omega * dt)  # (3, 3)
        
        # --- 바이어스 자코비안 재귀 갱신 (Step 3) ---
        # 위치 자코비안 (delta_p에 대한 바이어스 자코비안)
        self.d_p_d_bg += self.d_v_d_bg * dt - 0.5 * self.delta_R @ skew(acc) @ self.d_R_d_bg * dt**2
        self.d_p_d_ba += self.d_v_d_ba * dt - 0.5 * self.delta_R * dt**2
        # 속도 자코비안
        self.d_v_d_bg += -self.delta_R @ skew(acc) @ self.d_R_d_bg * dt
        self.d_v_d_ba += -self.delta_R * dt
        # 회전 자코비안
        self.d_R_d_bg = dR.T @ self.d_R_d_bg - Jr * dt   # 새 항에는 dR.T가 곱해지지 않는다
        
        # --- 공분산 전파 (Step 4) ---
        A = np.eye(9)
        A[0:3, 0:3] = dR.T
        A[3:6, 0:3] = -self.delta_R @ skew(acc) * dt
        A[6:9, 0:3] = -0.5 * self.delta_R @ skew(acc) * dt**2
        A[6:9, 3:6] = np.eye(3) * dt
        
        B = np.zeros((9, 6))
        B[0:3, 0:3] = Jr * dt
        B[3:6, 3:6] = self.delta_R * dt
        B[6:9, 3:6] = 0.5 * self.delta_R * dt**2
        
        Sigma_eta = np.zeros((6, 6))
        # 연속 시간 노이즈 밀도 → 이산 시간 공분산: 밀도²/dt (B에 dt가 이미 곱해져 있다)
        Sigma_eta[0:3, 0:3] = np.eye(3) * self.sigma_g**2 / dt
        Sigma_eta[3:6, 3:6] = np.eye(3) * self.sigma_a**2 / dt
        
        self.cov = A @ self.cov @ A.T + B @ Sigma_eta @ B.T
        
        # --- Preintegrated 측정 재귀 갱신 (Step 2) ---
        # 순서 중요: p를 먼저, 그 다음 v, 마지막 R
        self.delta_p += self.delta_v * dt + 0.5 * self.delta_R @ acc * dt**2
        self.delta_v += self.delta_R @ acc * dt
        self.delta_R = self.delta_R @ dR
        self.dt_sum += dt
    
    def predict(self, R_i, v_i, p_i, gravity):
        """Preintegrated 측정으로 키프레임 j의 상태를 예측한다.
        
        Parameters
        ----------
        R_i : ndarray, shape (3, 3) — 키프레임 i의 회전
        v_i : ndarray, shape (3,) — 키프레임 i의 속도
        p_i : ndarray, shape (3,) — 키프레임 i의 위치
        gravity : ndarray, shape (3,) — 중력 벡터 (e.g., [0, 0, -9.81])
        
        Returns
        -------
        R_j, v_j, p_j : 예측된 키프레임 j의 상태
        """
        dt = self.dt_sum
        R_j = R_i @ self.delta_R
        v_j = v_i + gravity * dt + R_i @ self.delta_v
        p_j = p_i + v_i * dt + 0.5 * gravity * dt**2 + R_i @ self.delta_p
        return R_j, v_j, p_j
    
    def compute_residual(self, R_i, v_i, p_i, R_j, v_j, p_j, gravity):
        """IMU factor의 잔차를 계산한다.
        
        Returns
        -------
        residual : ndarray, shape (9,) — [r_R(3), r_v(3), r_p(3)]
        """
        dt = self.dt_sum
        
        # 회전 잔차
        r_R = log_so3(self.delta_R.T @ R_i.T @ R_j)
        
        # 속도 잔차
        r_v = R_i.T @ (v_j - v_i - gravity * dt) - self.delta_v
        
        # 위치 잔차
        r_p = R_i.T @ (p_j - p_i - v_i * dt - 0.5 * gravity * dt**2) - self.delta_p
        
        return np.concatenate([r_R, r_v, r_p])

# GTSAM 사용 예시
import gtsam

def create_imu_factor_gtsam():
    """GTSAM의 내장 IMU preintegration을 사용하는 예제."""
    # IMU 파라미터 설정
    imu_params = gtsam.PreintegrationParams.MakeSharedU(9.81)  # Z-up 프레임용. 내부 중력 벡터는 (0, 0, -9.81)
    imu_params.setAccelerometerCovariance(np.eye(3) * 0.01**2)
    imu_params.setGyroscopeCovariance(np.eye(3) * 0.001**2)
    imu_params.setIntegrationCovariance(np.eye(3) * 1e-8)
    
    # 초기 바이어스
    bias = gtsam.imuBias.ConstantBias(
        np.array([0.1, -0.05, 0.02]),   # accelerometer bias
        np.array([0.001, -0.002, 0.003]) # gyroscope bias
    )
    
    # Preintegration 객체 생성
    pim = gtsam.PreintegratedImuMeasurements(imu_params, bias)
    
    # IMU 측정 적분 (200Hz 가정, 0.1초 = 20개 측정)
    dt = 0.005  # 200Hz
    for k in range(20):
        acc_meas = np.array([0.0, 0.0, 9.81])  # 정지 상태 (중력만)
        gyro_meas = np.array([0.0, 0.0, 0.0])
        pim.integrateMeasurement(acc_meas, gyro_meas, dt)
    
    # Factor 생성
    # ImuFactor는 i, j의 pose·velocity와 i의 bias를 연결
    imu_factor = gtsam.ImuFactor(
        gtsam.symbol('x', 0),  # pose_i
        gtsam.symbol('v', 0),  # vel_i
        gtsam.symbol('x', 1),  # pose_j
        gtsam.symbol('v', 1),  # vel_j
        gtsam.symbol('b', 0),  # bias
        pim
    )
    
    return imu_factor
```

구현에서 주목할 부분은 `integrate` 메서드 안의 갱신 순서다. `delta_p`와 `delta_v`는 직전 스텝까지의 회전 `delta_R`을 써야 하므로 위치·속도를 먼저 갱신한 뒤에 `delta_R`을 업데이트해야 한다.

---

## 4.7 Marginalization & Sliding Window

IMU preintegration으로 키프레임 사이의 고주파 데이터를 하나로 압축하더라도, 로봇이 오래 주행할수록 그래프에 쌓이는 키프레임 변수 자체는 계속 늘어난다. 제한된 시간 안에 일관된 계산량을 유지하려면 오래된 변수를 그래프에서 내보내되 그 변수가 담고 있던 관측 정보는 잃지 않는 메커니즘이 필요하다.

### 4.7.1 왜 Marginalization이 필요한가

Factor graph 기반 SLAM이나 VIO에서는 새 키프레임이 들어올 때마다 상태 변수와 factor가 하나둘 보태진다. 지나온 키프레임을 전부 메모리에 붙들고 있으면 언젠가는 최적화 계산 시간이 실시간 주기를 넘어선다.

그래서 실시간 VIO·LIO는 최근 $N$개의 키프레임만 활성 창에 남기고 가장 오래된 키프레임을 비우는 **슬라이딩 윈도우(sliding window)** 방식을 쓴다. 그런데 오래된 키프레임과 거기 매달린 관측 factor를 무작정 지워 버리면 과거에 쌓아 둔 기하 정보까지 함께 증발해 궤적이 쉽게 표류한다. **Marginalization**은 제거하려는 변수를 확률적으로 적분해 내보내면서, 그 변수들이 남는 변수들에게 걸어 주던 제약과 상관관계를 하나의 사전 분포(prior factor)로 압축해 전달하는 연산이다.

### 4.7.2 Schur Complement

실제 최적화 코드에서 marginalization은 정보 행렬에 **Schur complement**를 적용하는 선형대수 연산으로 구현된다. 먼저 정보 행렬 $\mathbf{H}$와 우변 벡터를 마지널라이즈해 뺄 변수 $\mathbf{x}_m$과 윈도우에 남길 변수 $\mathbf{x}_r$의 블록으로 나눈다. 표기 편의를 위해 여기서는 §4.5.3의 $-\mathbf{b}$를 새로 $\mathbf{b}$라 부른다($\mathbf{b} \triangleq -\mathbf{J}^\top \boldsymbol{\Sigma}^{-1} \mathbf{r}$).

$$\mathbf{H} \Delta\mathbf{x} = \mathbf{b}$$

$$\begin{bmatrix} \mathbf{H}_{mm} & \mathbf{H}_{mr} \\ \mathbf{H}_{rm} & \mathbf{H}_{rr} \end{bmatrix} \begin{bmatrix} \Delta\mathbf{x}_m \\ \Delta\mathbf{x}_r \end{bmatrix} = \begin{bmatrix} \mathbf{b}_m \\ \mathbf{b}_r \end{bmatrix}$$

각 블록이 가리키는 대상은 다음과 같다.
- $\mathbf{x}_m$: 마지널라이즈해 제거할 변수
- $\mathbf{x}_r$: 윈도우 안에 유지(retain)할 변수
- $\mathbf{H}_{mm} \in \mathbb{R}^{n_m \times n_m}$: 제거할 변수들끼리의 정보 블록
- $\mathbf{H}_{mr} \in \mathbb{R}^{n_m \times n_r}$: 제거 변수와 유지 변수 사이의 교차 정보 블록
- $\mathbf{H}_{rr} \in \mathbb{R}^{n_r \times n_r}$: 유지할 변수들끼리의 정보 블록

첫 번째 블록 방정식에서 $\Delta\mathbf{x}_m = \mathbf{H}_{mm}^{-1}(\mathbf{b}_m - \mathbf{H}_{mr}\Delta\mathbf{x}_r)$를 구해 두 번째 방정식에 대입하면 $\Delta\mathbf{x}_m$이 깔끔히 소거된다.

$$\underbrace{(\mathbf{H}_{rr} - \mathbf{H}_{rm} \mathbf{H}_{mm}^{-1} \mathbf{H}_{mr})}_{\mathbf{H}^*} \Delta\mathbf{x}_r = \underbrace{\mathbf{b}_r - \mathbf{H}_{rm} \mathbf{H}_{mm}^{-1} \mathbf{b}_m}_{\mathbf{b}^*}$$

이때 좌변의 $\mathbf{H}^* = \mathbf{H}_{rr} - \mathbf{H}_{rm} \mathbf{H}_{mm}^{-1} \mathbf{H}_{mr}$을 **Schur complement**라고 부른다. 이 $\mathbf{H}^*$와 $\mathbf{b}^*$가 곧 윈도우에 남은 변수 $\mathbf{x}_r$에 걸리는 새로운 prior factor의 정보 행렬과 gradient 항이 된다.

**직관적 의미**: $\mathbf{x}_m$을 매개로 삼아서만 간접적으로 연결되어 있던 $\mathbf{x}_r$ 내부의 변수들이, $\mathbf{x}_m$이 사라지는 대신 서로 직접 연결된다(fill-in). 이 때문에 $\mathbf{H}^*$는 원래의 $\mathbf{H}_{rr}$보다 0이 아닌 항이 더 빽빽해질 수 있다.

아래 코드는 블록 행렬 분할과 Schur complement 계산을 직접 구현해, 연쇄적으로 연결된 3개 2D 포즈 `[x0, x1, x2]` 가운데 가장 오래된 `x0`을 마지널라이즈할 때 정보 행렬의 희소 패턴이 어떻게 바뀌는지 확인하는 예제다.

```python
import numpy as np

def marginalize(H, b, indices_to_marginalize, indices_to_keep):
    """Schur complement를 이용한 변수 마지널라이제이션.
    
    Parameters
    ----------
    H : ndarray, shape (N, N) — 정보 행렬 (Hessian)
    b : ndarray, shape (N,) — 음의 gradient 벡터
    indices_to_marginalize : list of int — 제거할 변수의 인덱스
    indices_to_keep : list of int — 유지할 변수의 인덱스
    
    Returns
    -------
    H_star : ndarray — 마지널라이즈 후 정보 행렬
    b_star : ndarray — 마지널라이즈 후 음의 gradient
    """
    m = indices_to_marginalize
    r = indices_to_keep
    
    H_mm = H[np.ix_(m, m)]
    H_mr = H[np.ix_(m, r)]
    H_rm = H[np.ix_(r, m)]
    H_rr = H[np.ix_(r, r)]
    
    b_m = b[m]
    b_r = b[r]
    
    # Schur complement
    H_mm_inv = np.linalg.inv(H_mm)
    H_star = H_rr - H_rm @ H_mm_inv @ H_mr
    b_star = b_r - H_rm @ H_mm_inv @ b_m
    
    return H_star, b_star

# 예시: 3개 포즈 (각 2D, 3-DoF) 중 첫 번째를 마지널라이즈
# 상태: [x0(3), x1(3), x2(3)] = 9차원
np.random.seed(42)

# 희소 정보 행렬 (x0-x1, x1-x2 연결)
H = np.zeros((9, 9))
# x0 자체 (prior + odom factor)
H[0:3, 0:3] = np.diag([10, 10, 5])
# x0-x1 odometry factor
odom_info = np.diag([25, 25, 50])
H[0:3, 0:3] += odom_info
H[3:6, 3:6] += odom_info
H[0:3, 3:6] = -odom_info
H[3:6, 0:3] = -odom_info
# x1-x2 odometry factor
H[3:6, 3:6] += odom_info
H[6:9, 6:9] += odom_info
H[3:6, 6:9] = -odom_info
H[6:9, 3:6] = -odom_info

b = np.random.randn(9) * 0.1

print("마지널라이즈 전 H의 비영 패턴:")
print((np.abs(H) > 1e-10).astype(int))

# x0 (인덱스 0,1,2)를 마지널라이즈
H_star, b_star = marginalize(H, b, [0, 1, 2], [3, 4, 5, 6, 7, 8])

print("\n마지널라이즈 후 H*의 비영 패턴:")
print((np.abs(H_star) > 1e-10).astype(int))
print("→ x0은 x1에만 연결돼 있어 새 fill-in은 생기지 않는다")
```

이 예제에서는 `x0`이 `x1` 하나에만 붙어 있어 `H*`에 새로운 비영 블록이 생기지 않았지만, 실제 VIO처럼 제거할 포즈가 여러 랜드마크나 다수의 포즈와 동시에 연결되어 있을 때는 그 변수들을 한꺼번에 엮는 dense한 prior 블록이 새로 만들어진다. 게다가 이렇게 생성된 prior factor는 또 다른 미묘한 일관성 문제를 불러온다.

### 4.7.3 First-Estimate Jacobian (FEJ)

Schur complement로 만든 marginalization prior는 그 마지널라이즈 시점의 선형화 지점에서 계산된 값으로 굳어진다. 그런데 이후 최적화 반복에서 윈도우에 남은 변수의 추정치가 업데이트되면, **똑같은 변수인데도 marginalization prior 안에서는 예전 선형화 지점으로 평가되고, 새로 들어온 factor 안에서는 방금 바뀐 새 선형화 지점으로 평가되는 불일치**가 생긴다.

이처럼 한 변수를 두고 factor마다 서로 다른 지점에서 자코비안을 계산하면, §4.4.4에서 짚었던 관측 불가능한 gauge 방향으로 실제로는 존재하지 않는 가짜 정보가 유입되어 추정기가 과신하게 되고 일관성(consistency)을 잃는다.

**FEJ(First-Estimate Jacobian)**는 이 충돌을 막기 위해, 한 번 prior에 얽힌 변수의 자코비안을 계산할 때는 언제나 그 변수가 **처음 추정되었을 때의 지점(first estimate)**을 고정해 쓰는 규칙이다.

$$\mathbf{J}_{\text{FEJ}} = \left.\frac{\partial \mathbf{r}}{\partial \mathbf{x}}\right|_{\mathbf{x}^{(0)}}$$

여기서 $\mathbf{x}^{(0)}$는 그 변수가 상태 추정기에 처음 들어와 추정된 시점의 값이다.

FEJ를 적용하면 marginalization prior와 일반 관측 factor가 같은 선형화 기준점을 공유하므로 관측 불가능 방향에 허위 정보가 쌓이지 않는다. MSCKF 계열과 OpenVINS가 이 방식으로 필터의 일관성을 유지한다([Li & Mourikis, 2013](https://doi.org/10.1177/0278364913481251)). 다만 첫 추정치 $\mathbf{x}^{(0)}$ 자체가 참값에서 크게 벗어나 있었다면 고정된 자코비안도 부정확해진다는 한계가 있고, 변수마다 최초 추정치를 따로 보관하고 관리해야 하므로 구현 복잡도가 올라간다.

### 4.7.4 Sliding Window 구현의 실전 이슈

실제 슬라이딩 윈도우 최적화 백엔드를 바닥부터 구현하거나 튜닝할 때는 수식 밖의 네 가지 실전 문제를 함께 다루어야 한다.

#### 이슈 1: 뺄 키프레임 고르기

무조건 가장 오래된 키프레임만 빼면 로봇이 정지해 있거나 천천히 움직일 때 윈도우 안의 프레임들이 한 지점에 몰려 시차(parallax)가 사라진다. VINS-Mono는 뒤에서 두 번째 프레임(second latest frame)이 충분한 시차를 갖춘 키프레임인지 판별한 뒤, 키프레임이면 가장 오래된 프레임을 마지널라이즈하고, 그렇지 않으면 뒤에서 두 번째 프레임 자체를 버린다(§6.3.1). 덕분에 윈도우 안에는 공간적으로 고르게 퍼진 키프레임이 유지된다.

#### 이슈 2: Fill-in과 행렬 조밀화

마지널라이즈를 수십·수백 번 거듭하다 보면 prior factor가 연결하는 변수 범위가 점점 넓어져, 원래 희소하던 정보 행렬이 빽빽해지면서(dense) 최적화 속도가 눈에 띄게 떨어진다. 실무에서는 다음 방법으로 fill-in 확산을 억제한다.
- Prior factor에 포함할 변수 수에 상한을 둔다.
- 마지널라이즈할 변수의 순서를 신중히 고른다.
- 정보를 일부 잃는 것을 감수하고, 중요도가 낮은 일부 factor는 Schur complement에 넣지 않고 그냥 버린다(sparsification).

#### 이슈 3: 바이어스와 마지널라이제이션

IMU 바이어스는 키프레임 전체에 걸쳐 천천히 변하는 물리량이다. 오래된 키프레임을 뺄 때 그 시점의 바이어스까지 함께 마지널라이즈해 버리면 당시의 선형화 지점이 prior에 못 박혀, 이후 주행 중에 바이어스가 서서히 드리프트하더라도 prior가 이를 억제하는 부작용이 생길 수 있다. VINS-Mono 역시 가장 오래된 키프레임을 제거할 때 그 프레임의 속도·바이어스를 함께 마지널라이즈하지만, 윈도우에 남은 키프레임들의 바이어스 변수는 계속 최적화 대상으로 유지하며 prior와 연결해 둔다.

#### 이슈 4: 수치 안정성

Schur complement를 구할 때 $\mathbf{H}_{mm}^{-1}$을 계산해야 하는데, 관측이 부족한 변수가 끼어 있어 $\mathbf{H}_{mm}$의 조건수(condition number)가 나쁘면 역행렬 계산에서 수치 오차가 크게 튄다. 이를 막기 위해 실제 구현에서는 다음과 같은 안전장치를 둔다.
- $\mathbf{H}_{mm}$ 대각 성분에 작은 정규화 항을 더해 $(\mathbf{H}_{mm} + \epsilon \mathbf{I})^{-1}$로 푼다.
- 역행렬을 직접 구하지 않고 $LDL^\top$ 분해 같은 수치적으로 안정한 선형 솔버를 쓴다.
- 마지널라이즈 후 얻은 $\mathbf{H}^*$가 양의 준정부호(positive semi-definite, PSD) 성질을 유지하는지 확인하고, 부동소수점 오차로 음의 고윳값이 끼어들면 0으로 잘라 가장 가까운 PSD 행렬로 투영한다.

---

## 참고 자료

이 장에서는 단일 시점의 확률 분포를 재귀적으로 갱신하는 Bayesian filter(KF, EKF, ESKF, UKF, IEKF, Particle Filter)에서 출발해, 여러 시점을 묶어 비선형 최소자승 문제로 푸는 factor graph, 고주파 IMU 측정을 압축하는 on-manifold preintegration, 그리고 실시간 윈도우 크기를 묶어 두는 Schur complement와 FEJ까지 상태 추정의 핵심 도구를 살펴보았다.

이 밖에도 최근 상태 추정 연구에서는 기하학적 대칭과 학습 기반 접근으로 필터의 한계를 넓히려는 시도가 이어지고 있다. Invariant EKF와 Equivariant Filter는 Lie 군의 대칭 구조를 활용해, 조건이 충족되는 시스템에서 오차 동역학이 현재 추정치에 덜 의존하도록 설계한다. 또한 [AI-Aided Kalman Filters (Shlezinger et al., 2025)](https://arxiv.org/abs/2410.12289)는 칼만 이득이나 불완전한 시스템 모델 일부를 신경망으로 학습해 보완하는 흐름을 정리했다. 다만 학습 기반 요소가 들어간 추정기는 학습 분포 밖의 환경에서도 일관성과 안전성을 지키는지 별도로 검증해야 한다.

한편 지금까지 다룬 칼만 필터와 factor graph는 모두 "지금 들어온 관측이 지도나 이전 프레임의 어느 점·랜드마크에 해당하는지 이미 안다"는 전제 위에서 작동한다. 실제 센서 데이터에서 이 대응 관계를 찾아내는 특징 매칭과 데이터 연관이 다음 장의 주제다(→ 5장).
