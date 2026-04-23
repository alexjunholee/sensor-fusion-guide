# Ch.4 — State Estimation 이론

Ch.2에서 센서의 관측 모델을, Ch.3에서 센서 간 캘리브레이션을 다루었다. 이제 이 관측 데이터로부터 로봇의 상태(위치, 자세, 속도 등)를 추정하는 **알고리즘**의 차례다. 이 챕터에서 다루는 칼만 필터, 팩터 그래프, IMU 사전적분은 Ch.6-8에서 살펴볼 모든 odometry/fusion 시스템의 수학적 엔진이다.

> **목표**: 센서 퓨전의 수학적 기반인 상태 추정(state estimation) 이론을 체계적으로 다룬다. Bayesian filtering에서 출발하여 Kalman 필터 계열, Particle Filter를 거쳐, 현대 SLAM의 핵심인 factor graph 기반 최적화까지 이어지는 기술 계보를 따라간다. 각 방법의 유도 과정을 상세히 보여주고, 왜 현대 로봇 시스템이 filtering에서 optimization으로 이동했는지를 설명한다.

---

## 4.1 Bayesian Filtering Framework

### 4.1.1 상태 추정 문제의 정의

로봇의 상태 추정 문제는 본질적으로 **조건부 확률 추론** 문제다. 우리가 알고 싶은 것은 현재까지의 모든 관측 $\mathbf{z}_{1:k}$와 제어 입력 $\mathbf{u}_{1:k}$가 주어졌을 때 상태 $\mathbf{x}_k$의 사후 확률(posterior)이다:

$$p(\mathbf{x}_k \mid \mathbf{z}_{1:k}, \mathbf{u}_{1:k})$$

여기서:
- $\mathbf{x}_k \in \mathbb{R}^n$: 시각 $k$에서의 상태 벡터 (위치, 속도, 자세, 바이어스 등)
- $\mathbf{z}_k \in \mathbb{R}^m$: 시각 $k$에서의 관측 벡터 (카메라, LiDAR, IMU 등)
- $\mathbf{u}_k \in \mathbb{R}^l$: 시각 $k$에서의 제어 입력

이 문제를 풀기 위해 두 가지 모델을 정의한다:

**운동 모델(Motion Model, Process Model)**:
$$\mathbf{x}_k = f(\mathbf{x}_{k-1}, \mathbf{u}_k) + \mathbf{w}_k, \quad \mathbf{w}_k \sim \mathcal{N}(\mathbf{0}, \mathbf{Q}_k)$$

$$p(\mathbf{x}_k \mid \mathbf{x}_{k-1}, \mathbf{u}_k)$$

이것은 제어 입력 $\mathbf{u}_k$가 주어졌을 때, 이전 상태에서 현재 상태로의 전이 확률을 나타낸다. 프로세스 노이즈 $\mathbf{w}_k$는 모델의 불확실성을 반영한다.

**관측 모델(Observation Model, Measurement Model)**:
$$\mathbf{z}_k = h(\mathbf{x}_k) + \mathbf{v}_k, \quad \mathbf{v}_k \sim \mathcal{N}(\mathbf{0}, \mathbf{R}_k)$$

$$p(\mathbf{z}_k \mid \mathbf{x}_k)$$

현재 상태가 주어졌을 때, 관측값이 나올 likelihood다. 관측 노이즈 $\mathbf{v}_k$는 센서의 불확실성을 반영한다.

직관적으로 말하면, 로봇은 "내가 어디에 있는지" 정확히 알 수 없지만, "어떻게 움직였는지"(운동 모델)와 "무엇을 보는지"(관측 모델)를 결합하여 자신의 위치에 대한 **믿음(belief)**을 점진적으로 개선해 나간다.

### 4.1.2 Markov 가정과 Recursive Estimation

실용적인 필터링을 위해 **Markov 가정**을 도입한다:

1. **1차 Markov 프로세스**: 현재 상태 $\mathbf{x}_k$는 직전 상태 $\mathbf{x}_{k-1}$과 제어 $\mathbf{u}_k$에만 의존하며, 그 이전의 상태나 관측에는 조건부 독립이다:
$$p(\mathbf{x}_k \mid \mathbf{x}_{0:k-1}, \mathbf{u}_{1:k}, \mathbf{z}_{1:k-1}) = p(\mathbf{x}_k \mid \mathbf{x}_{k-1}, \mathbf{u}_k)$$

2. **조건부 관측 독립**: 관측 $\mathbf{z}_k$는 현재 상태 $\mathbf{x}_k$가 주어지면 다른 모든 것에 독립이다:
$$p(\mathbf{z}_k \mid \mathbf{x}_{0:k}, \mathbf{u}_{1:k}, \mathbf{z}_{1:k-1}) = p(\mathbf{z}_k \mid \mathbf{x}_k)$$

이 두 가정 덕분에, 전체 관측 이력 $\mathbf{z}_{1:k}$를 저장할 필요 없이 직전 추정치만으로 현재 추정치를 갱신할 수 있다. 이것이 **재귀적 추정(recursive estimation)**의 핵심이다.

### 4.1.3 Prediction-Update Cycle

Bayesian filter는 두 단계를 반복한다:

**예측 단계(Prediction Step)**: 운동 모델을 사용하여 이전 사후 확률로부터 현재 사전 확률(prior)을 예측한다.

$$\boxed{p(\mathbf{x}_k \mid \mathbf{z}_{1:k-1}, \mathbf{u}_{1:k}) = \int p(\mathbf{x}_k \mid \mathbf{x}_{k-1}, \mathbf{u}_k) \, p(\mathbf{x}_{k-1} \mid \mathbf{z}_{1:k-1}, \mathbf{u}_{1:k-1}) \, d\mathbf{x}_{k-1}}$$

이 적분이 바로 **Chapman-Kolmogorov 방정식**이다. 물리적 의미는 다음과 같다:
- $p(\mathbf{x}_{k-1} \mid \mathbf{z}_{1:k-1})$: 시각 $k-1$에서의 사후 확률 (이전 단계의 결과)
- $p(\mathbf{x}_k \mid \mathbf{x}_{k-1}, \mathbf{u}_k)$: 상태 전이 확률 (운동 모델)
- 이전 상태의 가능한 모든 값에 대해 전이 확률을 가중 평균하여 예측 분포를 구한다

**갱신 단계(Update Step)**: 새로운 관측 $\mathbf{z}_k$가 들어오면 Bayes 정리로 사후 확률을 갱신한다.

$$\boxed{p(\mathbf{x}_k \mid \mathbf{z}_{1:k}, \mathbf{u}_{1:k}) = \frac{p(\mathbf{z}_k \mid \mathbf{x}_k) \, p(\mathbf{x}_k \mid \mathbf{z}_{1:k-1}, \mathbf{u}_{1:k})}{p(\mathbf{z}_k \mid \mathbf{z}_{1:k-1})}}$$

여기서:
- 분자의 $p(\mathbf{z}_k \mid \mathbf{x}_k)$: likelihood (관측 모델)
- 분자의 $p(\mathbf{x}_k \mid \mathbf{z}_{1:k-1})$: 예측 단계에서 구한 prior
- 분모의 $p(\mathbf{z}_k \mid \mathbf{z}_{1:k-1}) = \int p(\mathbf{z}_k \mid \mathbf{x}_k) p(\mathbf{x}_k \mid \mathbf{z}_{1:k-1}) d\mathbf{x}_k$: 정규화 상수 (evidence)

직관적으로, 관측이 들어올 때마다 "이 관측은 내 예측과 얼마나 일치하는가?"를 평가하여 믿음을 수정하는 것이다.

### 4.1.4 왜 Closed-Form이 안 되는가

위의 prediction-update가 이론적으로는 완벽하지만, 실전에서는 Chapman-Kolmogorov 적분 $\int p(\mathbf{x}_k \mid \mathbf{x}_{k-1}) p(\mathbf{x}_{k-1} \mid \mathbf{z}_{1:k-1}) d\mathbf{x}_{k-1}$을 해석적으로 풀 수 없는 경우가 대부분이다. 그 이유는:

1. **비선형 운동/관측 모델**: $f(\cdot)$과 $h(\cdot)$가 비선형이면 가우시안 분포를 전파했을 때 가우시안이 유지되지 않는다. 사후 분포가 임의의 복잡한 형태를 가질 수 있다.

2. **고차원 상태 공간**: 상태 벡터의 차원이 커지면 적분의 계산량이 지수적으로 증가한다 (차원의 저주).

3. **다봉(multimodal) 분포**: 로봇의 위치가 여러 곳일 가능성이 있을 때 (예: 대칭적 환경에서의 global localization) 사후 분포가 다봉이 된다.

따라서 실전에서는 근사(approximation)가 필수적이며, 근사 방법에 따라 다양한 필터가 존재한다:

| 근사 방법 | 필터 | 분포 표현 | 적용 조건 |
|----------|------|---------|----------|
| 선형-가우시안 가정 | Kalman Filter | $\mathcal{N}(\hat{\mathbf{x}}, \mathbf{P})$ | 선형 시스템 |
| 1차 선형화 + 가우시안 | EKF, ESKF | $\mathcal{N}(\hat{\mathbf{x}}, \mathbf{P})$ | 약한 비선형 |
| Sigma point 변환 | UKF | $\mathcal{N}(\hat{\mathbf{x}}, \mathbf{P})$ | 중간 비선형 |
| 샘플 기반 | Particle Filter | $\{(\mathbf{x}^{(i)}, w^{(i)})\}_{i=1}^N$ | 강한 비선형, 다봉 |
| 반복 선형화 | IEKF | $\mathcal{N}(\hat{\mathbf{x}}, \mathbf{P})$ | 강한 비선형 |

이후 각 방법을 상세히 유도한다.

---

## 4.2 Kalman Filter 계열

### 4.2.1 Kalman Filter: 유도와 최적성

Kalman Filter(KF)는 **선형 시스템 + 가우시안 노이즈** 가정 하에서 Bayesian filter의 정확한(exact) 해다. Rudolf Kalman이 1960년 논문 ["A New Approach to Linear Filtering and Prediction Problems"](https://doi.org/10.1115/1.3662552)에서 제시했으며, 이후 Apollo 프로그램의 궤도 추정에 적용되면서 실용성이 입증되었다.

#### 선형 시스템 모델

상태 전이와 관측이 모두 선형인 시스템을 가정한다:

$$\mathbf{x}_k = \mathbf{F}_k \mathbf{x}_{k-1} + \mathbf{B}_k \mathbf{u}_k + \mathbf{w}_k, \quad \mathbf{w}_k \sim \mathcal{N}(\mathbf{0}, \mathbf{Q}_k)$$
$$\mathbf{z}_k = \mathbf{H}_k \mathbf{x}_k + \mathbf{v}_k, \quad \mathbf{v}_k \sim \mathcal{N}(\mathbf{0}, \mathbf{R}_k)$$

여기서:
- $\mathbf{F}_k \in \mathbb{R}^{n \times n}$: 상태 전이 행렬 (state transition matrix)
- $\mathbf{B}_k \in \mathbb{R}^{n \times l}$: 제어 입력 행렬
- $\mathbf{H}_k \in \mathbb{R}^{m \times n}$: 관측 행렬
- $\mathbf{Q}_k \in \mathbb{R}^{n \times n}$: 프로세스 노이즈 공분산 (양의 반정치, symmetric)
- $\mathbf{R}_k \in \mathbb{R}^{m \times m}$: 관측 노이즈 공분산 (양의 정치, symmetric)

핵심 성질: 가우시안 분포에 선형 변환을 적용하면 결과도 가우시안이다. 따라서 사후 분포가 항상 가우시안으로 유지되며, 평균과 공분산만으로 완전히 기술할 수 있다.

#### MMSE 유도 — 왜 Kalman Filter가 최적인가

Kalman Filter의 최적성을 Minimum Mean Square Error(MMSE) 관점에서 유도한다. 우리가 원하는 것은 평균 제곱 오차를 최소화하는 추정량이다:

$$\hat{\mathbf{x}}_k = \arg\min_{\hat{\mathbf{x}}} \mathbb{E}[\|\mathbf{x}_k - \hat{\mathbf{x}}\|^2 \mid \mathbf{z}_{1:k}]$$

MMSE 추정량은 조건부 기댓값 $\hat{\mathbf{x}}_k = \mathbb{E}[\mathbf{x}_k \mid \mathbf{z}_{1:k}]$이다. 선형-가우시안 시스템에서 이것이 정확히 Kalman Filter의 상태 갱신 방정식과 일치함을 보인다.

**Bayesian 관점의 유도**: 시각 $k-1$에서의 사후 분포가 가우시안이라고 가정한다:

$$p(\mathbf{x}_{k-1} \mid \mathbf{z}_{1:k-1}) = \mathcal{N}(\hat{\mathbf{x}}_{k-1|k-1}, \mathbf{P}_{k-1|k-1})$$

**Step 1: Prediction — Chapman-Kolmogorov 적분 수행**

선형 운동 모델에서:
$$\mathbf{x}_k = \mathbf{F}_k \mathbf{x}_{k-1} + \mathbf{B}_k \mathbf{u}_k + \mathbf{w}_k$$

$\mathbf{x}_{k-1}$이 가우시안이고 $\mathbf{w}_k$가 독립 가우시안이므로, $\mathbf{x}_k$도 가우시안이다. 평균과 공분산을 계산하면:

$$\hat{\mathbf{x}}_{k|k-1} = \mathbb{E}[\mathbf{x}_k \mid \mathbf{z}_{1:k-1}] = \mathbf{F}_k \hat{\mathbf{x}}_{k-1|k-1} + \mathbf{B}_k \mathbf{u}_k$$

공분산은 $\tilde{\mathbf{x}}_{k|k-1} = \mathbf{x}_k - \hat{\mathbf{x}}_{k|k-1} = \mathbf{F}_k \tilde{\mathbf{x}}_{k-1|k-1} + \mathbf{w}_k$이므로:

$$\mathbf{P}_{k|k-1} = \mathbb{E}[\tilde{\mathbf{x}}_{k|k-1} \tilde{\mathbf{x}}_{k|k-1}^\top] = \mathbf{F}_k \mathbf{P}_{k-1|k-1} \mathbf{F}_k^\top + \mathbf{Q}_k$$

예측된 분포: $p(\mathbf{x}_k \mid \mathbf{z}_{1:k-1}) = \mathcal{N}(\hat{\mathbf{x}}_{k|k-1}, \mathbf{P}_{k|k-1})$

**Step 2: Update — Bayes 정리 적용**

관측 모델 $\mathbf{z}_k = \mathbf{H}_k \mathbf{x}_k + \mathbf{v}_k$에서, 예측 상태와 관측의 결합 분포(joint distribution)를 구성한다:

$$\begin{bmatrix} \mathbf{x}_k \\ \mathbf{z}_k \end{bmatrix} \sim \mathcal{N}\left( \begin{bmatrix} \hat{\mathbf{x}}_{k|k-1} \\ \mathbf{H}_k \hat{\mathbf{x}}_{k|k-1} \end{bmatrix}, \begin{bmatrix} \mathbf{P}_{k|k-1} & \mathbf{P}_{k|k-1} \mathbf{H}_k^\top \\ \mathbf{H}_k \mathbf{P}_{k|k-1} & \mathbf{S}_k \end{bmatrix} \right)$$

여기서 $\mathbf{S}_k = \mathbf{H}_k \mathbf{P}_{k|k-1} \mathbf{H}_k^\top + \mathbf{R}_k \in \mathbb{R}^{m \times m}$는 혁신 공분산(innovation covariance)이다.

가우시안 결합 분포의 조건부 분포 공식을 적용하면 (가우시안의 핵심 성질: 결합이 가우시안이면 조건부도 가우시안이고, 조건부 평균은 원래 평균 + 관측과의 상관에 비례하는 보정):

$$\hat{\mathbf{x}}_{k|k} = \hat{\mathbf{x}}_{k|k-1} + \mathbf{P}_{k|k-1} \mathbf{H}_k^\top \mathbf{S}_k^{-1} (\mathbf{z}_k - \mathbf{H}_k \hat{\mathbf{x}}_{k|k-1})$$

**칼만 이득(Kalman gain)** $\mathbf{K}_k \in \mathbb{R}^{n \times m}$을 정의하면:

$$\boxed{\mathbf{K}_k = \mathbf{P}_{k|k-1} \mathbf{H}_k^\top \mathbf{S}_k^{-1} = \mathbf{P}_{k|k-1} \mathbf{H}_k^\top (\mathbf{H}_k \mathbf{P}_{k|k-1} \mathbf{H}_k^\top + \mathbf{R}_k)^{-1}}$$

칼만 이득의 직관적 의미:
- $\mathbf{R}_k \to \mathbf{0}$ (관측이 매우 정확): $\mathbf{K}_k \to \mathbf{H}_k^{-1}$ → 관측을 거의 그대로 믿는다
- $\mathbf{P}_{k|k-1} \to \mathbf{0}$ (예측이 매우 정확): $\mathbf{K}_k \to \mathbf{0}$ → 관측을 무시하고 예측을 믿는다
- 칼만 이득은 예측의 불확실성과 관측의 불확실성 사이의 **최적 가중치**를 자동으로 결정한다

**혁신(Innovation)**:
$$\tilde{\mathbf{y}}_k = \mathbf{z}_k - \mathbf{H}_k \hat{\mathbf{x}}_{k|k-1} \in \mathbb{R}^m$$

예측한 관측과 실제 관측의 차이. 이것이 0이면 예측이 완벽했다는 뜻이다.

**상태 갱신**:
$$\boxed{\hat{\mathbf{x}}_{k|k} = \hat{\mathbf{x}}_{k|k-1} + \mathbf{K}_k \tilde{\mathbf{y}}_k}$$

**공분산 갱신**:
$$\boxed{\mathbf{P}_{k|k} = (\mathbf{I} - \mathbf{K}_k \mathbf{H}_k) \mathbf{P}_{k|k-1}}$$

이 공분산 갱신 공식은 Joseph form $\mathbf{P}_{k|k} = (\mathbf{I} - \mathbf{K}_k \mathbf{H}_k) \mathbf{P}_{k|k-1} (\mathbf{I} - \mathbf{K}_k \mathbf{H}_k)^\top + \mathbf{K}_k \mathbf{R}_k \mathbf{K}_k^\top$으로 쓰면 수치적으로 더 안정적이다. Joseph form은 $\mathbf{K}_k$에 수치 오차가 있어도 $\mathbf{P}_{k|k}$의 대칭성과 양의 반정치성을 보장한다.

#### KF 최적성 정리

**정리**: 선형-가우시안 시스템에서 Kalman Filter는 다음 세 가지 의미에서 최적이다:
1. **MMSE (Minimum Mean Square Error) 추정량**: 평균 제곱 오차를 최소화
2. **MAP (Maximum A Posteriori) 추정량**: 가우시안 분포에서 평균 = 최빈값이므로 MAP과 MMSE가 일치
3. **BLUE (Best Linear Unbiased Estimator)**: 가우시안 가정 없이도, 선형 비편향 추정량 중에서 최소 분산

Kalman의 1960년 원논문은 Wiener 필터의 주파수 도메인 접근법을 상태공간(state-space) 시간 도메인 표현으로 대체함으로써, 시변(time-varying) 시스템과 다변량 시스템에 자연스럽게 확장할 수 있게 만든 혁신이었다.

#### Python 구현

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


# 예제: 1D 등속 운동 모델에서의 위치 추정
# 상태: [position, velocity]^T
dt = 0.1  # 시간 간격
F = np.array([[1, dt],
              [0, 1]])      # (2, 2) 등속 운동 전이 행렬
H = np.array([[1, 0]])       # (1, 2) 위치만 관측
Q = np.array([[dt**3/3, dt**2/2],
              [dt**2/2, dt]]) * 0.1  # (2, 2) 프로세스 노이즈 (등가속도 모델)
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

### 4.2.2 Extended Kalman Filter (EKF)

실제 로봇 시스템은 거의 항상 비선형이다. 카메라의 3D→2D 투영, IMU의 쿼터니언 기반 회전, LiDAR의 scan matching, GPS의 좌표 변환 모두 비선형 함수다. EKF는 Kalman Filter를 비선형 시스템으로 확장하는 가장 직접적인 방법이다.

#### 핵심 아이디어: 1차 테일러 전개 (선형화)

비선형 시스템 모델:
$$\mathbf{x}_k = f(\mathbf{x}_{k-1}, \mathbf{u}_k) + \mathbf{w}_k$$
$$\mathbf{z}_k = h(\mathbf{x}_k) + \mathbf{v}_k$$

이 시스템에서는 가우시안을 비선형 함수에 통과시키면 결과가 더 이상 가우시안이 아니다. EKF의 핵심 근사는 **현재 추정치 근방에서 함수를 1차 테일러 전개하여 선형화**하는 것이다.

운동 모델의 선형화 (추정치 $\hat{\mathbf{x}}_{k-1|k-1}$ 근방에서):

$$f(\mathbf{x}_{k-1}, \mathbf{u}_k) \approx f(\hat{\mathbf{x}}_{k-1|k-1}, \mathbf{u}_k) + \mathbf{F}_k (\mathbf{x}_{k-1} - \hat{\mathbf{x}}_{k-1|k-1})$$

$$\mathbf{F}_k = \left.\frac{\partial f}{\partial \mathbf{x}}\right|_{\hat{\mathbf{x}}_{k-1|k-1}, \mathbf{u}_k} \in \mathbb{R}^{n \times n}$$

관측 모델의 선형화 (예측 추정치 $\hat{\mathbf{x}}_{k|k-1}$ 근방에서):

$$h(\mathbf{x}_k) \approx h(\hat{\mathbf{x}}_{k|k-1}) + \mathbf{H}_k (\mathbf{x}_k - \hat{\mathbf{x}}_{k|k-1})$$

$$\mathbf{H}_k = \left.\frac{\partial h}{\partial \mathbf{x}}\right|_{\hat{\mathbf{x}}_{k|k-1}} \in \mathbb{R}^{m \times n}$$

#### EKF 알고리즘

**예측 단계**:
$$\hat{\mathbf{x}}_{k|k-1} = f(\hat{\mathbf{x}}_{k-1|k-1}, \mathbf{u}_k)$$
$$\mathbf{P}_{k|k-1} = \mathbf{F}_k \mathbf{P}_{k-1|k-1} \mathbf{F}_k^\top + \mathbf{Q}_k$$

주의: 상태 예측에는 비선형 함수 $f$를 그대로 사용하지만, 공분산 전파에는 자코비안 $\mathbf{F}_k$를 사용한다.

**갱신 단계**:
$$\tilde{\mathbf{y}}_k = \mathbf{z}_k - h(\hat{\mathbf{x}}_{k|k-1})$$
$$\mathbf{S}_k = \mathbf{H}_k \mathbf{P}_{k|k-1} \mathbf{H}_k^\top + \mathbf{R}_k$$
$$\mathbf{K}_k = \mathbf{P}_{k|k-1} \mathbf{H}_k^\top \mathbf{S}_k^{-1}$$
$$\hat{\mathbf{x}}_{k|k} = \hat{\mathbf{x}}_{k|k-1} + \mathbf{K}_k \tilde{\mathbf{y}}_k$$
$$\mathbf{P}_{k|k} = (\mathbf{I} - \mathbf{K}_k \mathbf{H}_k) \mathbf{P}_{k|k-1}$$

혁신 $\tilde{\mathbf{y}}_k$에서도 비선형 함수 $h$를 직접 사용한다는 점에 주의한다.

#### EKF의 한계

1. **선형화 오차**: 비선형성이 심할수록 1차 근사의 오차가 커진다. 이는 필터의 일관성(consistency)을 해칠 수 있다. 실제 오차가 필터가 추정한 불확실성보다 크게 벌어지는 것이다.

2. **자코비안 계산 부담**: 모든 시간 단계에서 $\mathbf{F}_k$와 $\mathbf{H}_k$의 해석적 미분을 구해야 한다. 시스템이 복잡하면 자코비안 유도가 매우 번거롭고 오류가 발생하기 쉽다.

3. **단봉(unimodal) 가정**: 가우시안은 항상 단봉이므로, 다봉 사후 분포를 표현할 수 없다.

### 4.2.3 Error-State Kalman Filter (ESKF)

ESKF(Error-State Kalman Filter)는 현대 로봇 센서 퓨전 시스템에서 가장 널리 사용되는 필터 형태다. MSCKF, VINS-Mono, OpenVINS, FAST-LIO 등 거의 모든 주요 VIO/LIO 시스템이 ESKF를 채택하고 있다.

#### 왜 EKF 대신 ESKF를 쓰는가

3D 자세(orientation)를 포함하는 로봇 상태를 추정할 때 EKF를 직접 쓰면 여러 문제가 발생한다:

**문제 1: 회전의 비유클리드 특성**

3D 회전은 SO(3) 매니폴드 위에 존재하며, $\mathbb{R}^n$이 아니다. 쿼터니언 $\mathbf{q} \in \mathbb{H}$로 표현하면 $\|\mathbf{q}\| = 1$ 제약이 있고, 회전 행렬 $\mathbf{R} \in SO(3)$은 $\mathbf{R}^\top \mathbf{R} = \mathbf{I}$, $\det(\mathbf{R}) = 1$ 제약이 있다.

EKF의 상태 갱신 $\hat{\mathbf{x}} \leftarrow \hat{\mathbf{x}} + \mathbf{K} \tilde{\mathbf{y}}$에서 "+" 연산은 유클리드 공간의 덧셈이다. 쿼터니언에 벡터를 더하면 단위 쿼터니언이 아니게 된다. 갱신 후 재정규화하는 임시방편은 이론적으로 올바르지 않으며, 일관성 문제를 야기한다.

**문제 2: 오차 상태는 "거의 0"**

오차 상태(error state) $\delta\mathbf{x} = \mathbf{x} - \hat{\mathbf{x}}$는 정의상 0 근처에 머문다 (갱신 때마다 리셋되므로). 따라서 1차 선형화의 정확도가 매우 높다. 반면 원래 상태에 대한 선형화는 운동이 큰 경우 정확도가 떨어진다.

**문제 3: 느린 변화(slow-varying) 상태와 빠른 변화(fast-varying) 상태의 분리**

IMU 바이어스처럼 느리게 변하는 상태와 속도/자세처럼 빠르게 변하는 상태를 분리하여 처리하면, 각각에 적합한 업데이트 전략을 적용할 수 있다.

#### ESKF 구조

ESKF는 두 개의 상태를 동시에 관리한다:

1. **명목 상태(Nominal State)** $\hat{\mathbf{x}}$: 비선형 운동 모델을 따라 적분되며, 노이즈 항을 포함하지 않는다. 불확실성을 추적하지 않는다.

2. **오차 상태(Error State)** $\delta\mathbf{x}$: 명목 상태와 실제 상태의 차이. 칼만 필터로 추정한다. 오차 상태는 정의상 "작은 값"이므로 선형화 오차가 최소화된다.

실제 상태는 두 상태의 합성(composition)으로 복원된다:

$$\mathbf{x}_{\text{true}} = \hat{\mathbf{x}} \boxplus \delta\mathbf{x}$$

여기서 $\boxplus$는 매니폴드 위의 합성 연산이다. 유클리드 성분에서는 일반 덧셈이고, 회전 성분에서는:

$$\mathbf{R}_{\text{true}} = \hat{\mathbf{R}} \cdot \text{Exp}(\delta\boldsymbol{\theta})$$

또는 쿼터니언으로:

$$\mathbf{q}_{\text{true}} = \hat{\mathbf{q}} \otimes \begin{bmatrix} 1 \\ \frac{1}{2}\delta\boldsymbol{\theta} \end{bmatrix} \approx \hat{\mathbf{q}} \otimes \delta\mathbf{q}$$

여기서 $\delta\boldsymbol{\theta} \in \mathbb{R}^3$는 회전 오차의 각축(angle-axis) 표현이다.

#### IMU 기반 ESKF의 상태 벡터

전형적인 IMU-camera/LiDAR 퓨전 시스템에서의 상태 벡터:

**명목 상태** (16차원, 쿼터니언 사용시):
$$\hat{\mathbf{x}} = \begin{bmatrix} {}^W\hat{\mathbf{p}} \\ {}^W\hat{\mathbf{v}} \\ \hat{\mathbf{q}}_{WB} \\ \hat{\mathbf{b}}_a \\ \hat{\mathbf{b}}_g \end{bmatrix} \in \mathbb{R}^{3} \times \mathbb{R}^{3} \times \mathbb{S}^3 \times \mathbb{R}^{3} \times \mathbb{R}^{3}$$

**오차 상태** (15차원 — 회전의 최소 파라미터화):
$$\delta\mathbf{x} = \begin{bmatrix} \delta\mathbf{p} \\ \delta\mathbf{v} \\ \delta\boldsymbol{\theta} \\ \delta\mathbf{b}_a \\ \delta\mathbf{b}_g \end{bmatrix} \in \mathbb{R}^{15}$$

핵심: 쿼터니언(4차원)의 오차 표현이 3차원 벡터 $\delta\boldsymbol{\theta}$다. 단위 쿼터니언 제약 때문에 실제 자유도는 3이고, ESKF는 이 최소 파라미터화를 자연스럽게 사용한다. EKF에서 쿼터니언을 직접 상태에 넣으면 4차원 표현의 1개 자유도가 과잉이 되어 공분산 행렬이 singular해지는 문제가 있다.

#### ESKF 알고리즘

**1단계: 명목 상태 전파 (IMU mechanization)**

IMU 측정 $(\tilde{\boldsymbol{\omega}}_k, \tilde{\mathbf{a}}_k)$로부터 명목 상태를 적분한다. 바이어스를 빼고 노이즈 항은 무시한다:

$$\hat{\mathbf{q}}_{k+1} = \hat{\mathbf{q}}_k \otimes \mathbf{q}\{(\tilde{\boldsymbol{\omega}}_k - \hat{\mathbf{b}}_{g,k}) \Delta t\}$$
$$\hat{\mathbf{v}}_{k+1} = \hat{\mathbf{v}}_k + (\hat{\mathbf{R}}_k (\tilde{\mathbf{a}}_k - \hat{\mathbf{b}}_{a,k}) + \mathbf{g}) \Delta t$$
$$\hat{\mathbf{p}}_{k+1} = \hat{\mathbf{p}}_k + \hat{\mathbf{v}}_k \Delta t + \frac{1}{2}(\hat{\mathbf{R}}_k (\tilde{\mathbf{a}}_k - \hat{\mathbf{b}}_{a,k}) + \mathbf{g}) \Delta t^2$$
$$\hat{\mathbf{b}}_{a,k+1} = \hat{\mathbf{b}}_{a,k}$$
$$\hat{\mathbf{b}}_{g,k+1} = \hat{\mathbf{b}}_{g,k}$$

여기서 $\hat{\mathbf{R}}_k = \mathbf{R}(\hat{\mathbf{q}}_k) \in SO(3)$이고, $\mathbf{g} = [0, 0, -9.81]^\top \, \text{m/s}^2$는 중력 벡터이다.

**2단계: 오차 상태 전파 (prediction)**

오차 상태의 연속 시간 동역학을 유도한다. 실제 IMU 측정 $\tilde{\boldsymbol{\omega}} = \boldsymbol{\omega} + \mathbf{b}_g + \mathbf{n}_g$, $\tilde{\mathbf{a}} = \mathbf{R}^\top(\mathbf{a}_W - \mathbf{g}) + \mathbf{b}_a + \mathbf{n}_a$를 대입하고 명목 상태와의 차이를 취하면:

$$\delta\dot{\boldsymbol{\theta}} = -[\tilde{\boldsymbol{\omega}} - \hat{\mathbf{b}}_g]_\times \delta\boldsymbol{\theta} - \delta\mathbf{b}_g - \mathbf{n}_g$$
$$\delta\dot{\mathbf{v}} = -\hat{\mathbf{R}}[\tilde{\mathbf{a}} - \hat{\mathbf{b}}_a]_\times \delta\boldsymbol{\theta} - \hat{\mathbf{R}} \delta\mathbf{b}_a - \hat{\mathbf{R}} \mathbf{n}_a$$
$$\delta\dot{\mathbf{p}} = \delta\mathbf{v}$$
$$\delta\dot{\mathbf{b}}_a = \mathbf{n}_{ba}$$
$$\delta\dot{\mathbf{b}}_g = \mathbf{n}_{bg}$$

여기서 $[\mathbf{a}]_\times$는 벡터 $\mathbf{a}$의 반대칭 행렬(skew-symmetric matrix):

$$[\mathbf{a}]_\times = \begin{bmatrix} 0 & -a_3 & a_2 \\ a_3 & 0 & -a_1 \\ -a_2 & a_1 & 0 \end{bmatrix} \in \mathbb{R}^{3 \times 3}$$

이를 행렬 형태로 쓰면:

$$\delta\dot{\mathbf{x}} = \mathbf{F}_c \delta\mathbf{x} + \mathbf{G}_c \mathbf{n}$$

$$\mathbf{F}_c = \begin{bmatrix}
\mathbf{0}_3 & \mathbf{I}_3 & \mathbf{0}_3 & \mathbf{0}_3 & \mathbf{0}_3 \\
\mathbf{0}_3 & \mathbf{0}_3 & -\hat{\mathbf{R}}[\tilde{\mathbf{a}} - \hat{\mathbf{b}}_a]_\times & -\hat{\mathbf{R}} & \mathbf{0}_3 \\
\mathbf{0}_3 & \mathbf{0}_3 & -[\tilde{\boldsymbol{\omega}} - \hat{\mathbf{b}}_g]_\times & \mathbf{0}_3 & -\mathbf{I}_3 \\
\mathbf{0}_3 & \mathbf{0}_3 & \mathbf{0}_3 & \mathbf{0}_3 & \mathbf{0}_3 \\
\mathbf{0}_3 & \mathbf{0}_3 & \mathbf{0}_3 & \mathbf{0}_3 & \mathbf{0}_3
\end{bmatrix} \in \mathbb{R}^{15 \times 15}$$

이산화 (1차 근사): $\mathbf{F}_d \approx \mathbf{I} + \mathbf{F}_c \Delta t$

$$\mathbf{P}_{k+1|k} = \mathbf{F}_d \mathbf{P}_{k|k} \mathbf{F}_d^\top + \mathbf{G}_d \mathbf{Q}_d \mathbf{G}_d^\top$$

**3단계: 관측 업데이트**

카메라/LiDAR 관측이 들어오면 표준 EKF 업데이트를 오차 상태에 대해 수행한다. 관측 모델 $\mathbf{z} = h(\mathbf{x}_{\text{true}})$를 오차 상태에 대해 선형화:

$$\mathbf{z} - h(\hat{\mathbf{x}}) \approx \mathbf{H} \delta\mathbf{x} + \mathbf{v}$$

$$\mathbf{H} = \frac{\partial h}{\partial \delta\mathbf{x}}\bigg|_{\hat{\mathbf{x}}} \in \mathbb{R}^{m \times 15}$$

이 자코비안은 체인 룰로 계산한다:

$$\mathbf{H} = \frac{\partial h}{\partial \mathbf{x}_{\text{true}}} \cdot \frac{\partial \mathbf{x}_{\text{true}}}{\partial \delta\mathbf{x}}\bigg|_{\delta\mathbf{x}=\mathbf{0}}$$

표준 칼만 갱신:
$$\mathbf{K} = \mathbf{P}_{k|k-1} \mathbf{H}^\top (\mathbf{H} \mathbf{P}_{k|k-1} \mathbf{H}^\top + \mathbf{R})^{-1}$$
$$\delta\hat{\mathbf{x}} = \mathbf{K}(\mathbf{z} - h(\hat{\mathbf{x}}))$$
$$\mathbf{P}_{k|k} = (\mathbf{I} - \mathbf{K}\mathbf{H})\mathbf{P}_{k|k-1}$$

**4단계: 오차 상태 주입 및 리셋**

오차 상태 추정치 $\delta\hat{\mathbf{x}}$를 명목 상태에 주입(inject)한다:

$$\hat{\mathbf{p}} \leftarrow \hat{\mathbf{p}} + \delta\hat{\mathbf{p}}$$
$$\hat{\mathbf{v}} \leftarrow \hat{\mathbf{v}} + \delta\hat{\mathbf{v}}$$
$$\hat{\mathbf{q}} \leftarrow \hat{\mathbf{q}} \otimes \mathbf{q}\{\delta\hat{\boldsymbol{\theta}}\}$$
$$\hat{\mathbf{b}}_a \leftarrow \hat{\mathbf{b}}_a + \delta\hat{\mathbf{b}}_a$$
$$\hat{\mathbf{b}}_g \leftarrow \hat{\mathbf{b}}_g + \delta\hat{\mathbf{b}}_g$$

주입 후, 오차 상태를 $\delta\hat{\mathbf{x}} \leftarrow \mathbf{0}$으로 리셋한다. 공분산도 리셋 자코비안으로 변환해야 한다:

$$\mathbf{P} \leftarrow \mathbf{G} \mathbf{P} \mathbf{G}^\top$$

여기서 $\mathbf{G} = \frac{\partial (\delta\mathbf{x} \boxminus \delta\hat{\mathbf{x}})}{\partial \delta\mathbf{x}}\big|_{\delta\hat{\mathbf{x}}}$이다. 실전에서 $\delta\hat{\boldsymbol{\theta}}$가 작으면 $\mathbf{G} \approx \mathbf{I}$로 근사하는 경우가 많다.

### 4.2.4 Unscented Kalman Filter (UKF)

UKF는 "분포를 근사하는 것이 비선형 함수를 근사하는 것보다 쉽다"는 통찰에 기반한다. EKF는 비선형 함수를 선형화하여 근사하지만, UKF는 비선형 함수는 그대로 두고 분포를 유한 개의 **sigma point**로 근사한다.

#### Unscented Transform

$n$차원 가우시안 확률 변수 $\mathbf{x} \sim \mathcal{N}(\hat{\mathbf{x}}, \mathbf{P})$에 대해, $2n+1$개의 sigma point와 가중치를 생성한다:

$$\boldsymbol{\chi}_0 = \hat{\mathbf{x}}, \quad w_0 = \frac{\lambda}{n + \lambda}$$
$$\boldsymbol{\chi}_i = \hat{\mathbf{x}} + \left(\sqrt{(n+\lambda)\mathbf{P}}\right)_i, \quad w_i = \frac{1}{2(n+\lambda)}, \quad i = 1, \ldots, n$$
$$\boldsymbol{\chi}_{n+i} = \hat{\mathbf{x}} - \left(\sqrt{(n+\lambda)\mathbf{P}}\right)_i, \quad w_{n+i} = \frac{1}{2(n+\lambda)}, \quad i = 1, \ldots, n$$

여기서:
- $\lambda = \alpha^2(n + \kappa) - n$는 스케일링 파라미터 ($\alpha$: sigma point 분산 조절, $\kappa$: 보조 파라미터, 보통 $\kappa = 0$ 또는 $3-n$)
- $\left(\sqrt{(n+\lambda)\mathbf{P}}\right)_i$는 행렬 $(n+\lambda)\mathbf{P}$의 Cholesky 분해의 $i$번째 열

각 sigma point를 비선형 함수에 통과시킨다:

$$\boldsymbol{\gamma}_i = f(\boldsymbol{\chi}_i)$$

변환된 sigma point들로부터 평균과 공분산을 복원한다:

$$\hat{\mathbf{y}} = \sum_{i=0}^{2n} w_i^{(m)} \boldsymbol{\gamma}_i$$
$$\mathbf{P}_y = \sum_{i=0}^{2n} w_i^{(c)} (\boldsymbol{\gamma}_i - \hat{\mathbf{y}})(\boldsymbol{\gamma}_i - \hat{\mathbf{y}})^\top$$

$w_i^{(m)}$과 $w_i^{(c)}$는 평균과 공분산에 각각 사용하는 가중치로, $w_0^{(c)} = w_0^{(m)} + (1 - \alpha^2 + \beta)$ ($\beta = 2$는 가우시안에 최적)이고, 나머지는 동일하다.

#### UKF 알고리즘

**예측 단계**:
1. 현재 상태 $(\hat{\mathbf{x}}_{k-1|k-1}, \mathbf{P}_{k-1|k-1})$에서 sigma point 생성
2. 각 sigma point를 운동 모델에 통과: $\boldsymbol{\chi}_{k|k-1}^{(i)} = f(\boldsymbol{\chi}_{k-1|k-1}^{(i)}, \mathbf{u}_k)$
3. 예측 평균과 공분산 계산: $\hat{\mathbf{x}}_{k|k-1} = \sum w_i^{(m)} \boldsymbol{\chi}_{k|k-1}^{(i)}$
4. $\mathbf{P}_{k|k-1} = \sum w_i^{(c)} (\boldsymbol{\chi}_{k|k-1}^{(i)} - \hat{\mathbf{x}}_{k|k-1})(\cdots)^\top + \mathbf{Q}_k$

**갱신 단계**:
1. 예측 상태에서 sigma point 재생성 (또는 예측 단계의 sigma point를 재사용)
2. 관측 모델에 통과: $\boldsymbol{\zeta}_k^{(i)} = h(\boldsymbol{\chi}_{k|k-1}^{(i)})$
3. 예측 관측 평균: $\hat{\mathbf{z}}_k = \sum w_i^{(m)} \boldsymbol{\zeta}_k^{(i)}$
4. 관측 공분산: $\mathbf{P}_{zz} = \sum w_i^{(c)} (\boldsymbol{\zeta}_k^{(i)} - \hat{\mathbf{z}}_k)(\cdots)^\top + \mathbf{R}_k$
5. 교차 공분산: $\mathbf{P}_{xz} = \sum w_i^{(c)} (\boldsymbol{\chi}_{k|k-1}^{(i)} - \hat{\mathbf{x}}_{k|k-1})(\boldsymbol{\zeta}_k^{(i)} - \hat{\mathbf{z}}_k)^\top$
6. 칼만 이득: $\mathbf{K}_k = \mathbf{P}_{xz} \mathbf{P}_{zz}^{-1}$
7. 갱신: $\hat{\mathbf{x}}_{k|k} = \hat{\mathbf{x}}_{k|k-1} + \mathbf{K}_k (\mathbf{z}_k - \hat{\mathbf{z}}_k)$
8. $\mathbf{P}_{k|k} = \mathbf{P}_{k|k-1} - \mathbf{K}_k \mathbf{P}_{zz} \mathbf{K}_k^\top$

#### UKF의 장단점

**장점**:
- 자코비안 계산이 불필요하다. 복잡한 관측 모델(예: 카메라 투영 + 왜곡)에서 큰 이점.
- 2차 비선형성까지 정확히 포착한다 (EKF는 1차까지만).
- 구현이 EKF보다 간단할 수 있다 (자코비안 유도 대신 함수 호출만 필요).

**단점**:
- $2n+1$개 sigma point 각각을 비선형 함수에 통과시켜야 하므로, 상태 차원 $n$이 클 때 연산량이 증가한다.
- 매니폴드 위의 상태(SO(3) 등)를 다루려면 sigma point의 생성과 통계 계산을 매니폴드 연산으로 대체해야 하며, 이것이 깔끔하지 않다.
- 실전에서 ESKF가 UKF보다 선호되는 이유: ESKF는 이미 오차 상태(작은 값)에서 동작하므로 1차 선형화의 정확도가 충분하고, 매니폴드 처리가 자연스러우며, 계산량이 적다.

### 4.2.5 Iterated Extended Kalman Filter (IEKF)

IEKF는 EKF의 갱신 단계에서 선형화를 한 번만 하는 대신, 반복적으로 재선형화하여 비선형 관측 모델의 처리 정확도를 높이는 방법이다.

#### 동기: EKF 갱신의 선형화 오차

EKF에서 관측 모델의 자코비안 $\mathbf{H}_k$는 예측 추정치 $\hat{\mathbf{x}}_{k|k-1}$에서 계산된다. 그런데 갱신 후의 추정치 $\hat{\mathbf{x}}_{k|k}$가 $\hat{\mathbf{x}}_{k|k-1}$과 크게 다르면, 선형화 지점이 최적이 아니게 된다. IEKF는 갱신 후의 추정치에서 다시 선형화하고 갱신을 반복함으로써 이 문제를 완화한다.

#### IEKF 알고리즘

예측 단계는 EKF와 동일하다. 갱신 단계에서 반복을 수행한다:

초기화: $\hat{\mathbf{x}}^{(0)} = \hat{\mathbf{x}}_{k|k-1}$

$j = 0, 1, 2, \ldots$ 에 대해 수렴할 때까지:

$$\mathbf{H}^{(j)} = \left.\frac{\partial h}{\partial \mathbf{x}}\right|_{\hat{\mathbf{x}}^{(j)}}$$

$$\mathbf{K}^{(j)} = \mathbf{P}_{k|k-1} \mathbf{H}^{(j)\top} (\mathbf{H}^{(j)} \mathbf{P}_{k|k-1} \mathbf{H}^{(j)\top} + \mathbf{R}_k)^{-1}$$

$$\hat{\mathbf{x}}^{(j+1)} = \hat{\mathbf{x}}_{k|k-1} + \mathbf{K}^{(j)} \left[\mathbf{z}_k - h(\hat{\mathbf{x}}^{(j)}) - \mathbf{H}^{(j)}(\hat{\mathbf{x}}_{k|k-1} - \hat{\mathbf{x}}^{(j)})\right]$$

수렴 후: $\hat{\mathbf{x}}_{k|k} = \hat{\mathbf{x}}^{(j+1)}$, $\mathbf{P}_{k|k} = (\mathbf{I} - \mathbf{K}^{(j)} \mathbf{H}^{(j)}) \mathbf{P}_{k|k-1}$

IEKF는 사실상 관측 업데이트 단계에서 **Gauss-Newton 최적화**를 수행하는 것과 동치이다. 이 관점은 나중에 factor graph 기반 최적화와의 연결을 이해하는 데 중요하다.

FAST-LIO2가 IEKF를 채택한 이유: LiDAR point-to-plane/point-to-edge 관측 모델은 비선형성이 상당하며, 수백~수천 개의 점을 한 번에 갱신해야 한다. IEKF의 반복 선형화는 이 상황에서 한 번의 EKF 업데이트보다 상당히 정확한 결과를 준다.

---

## 4.3 Particle Filter

### 4.3.1 Sequential Monte Carlo (SMC) 개요

Particle Filter(PF), 또는 Sequential Monte Carlo(SMC) 방법은 사후 분포를 가중치가 부여된 샘플(particle)들의 집합으로 표현한다. 가우시안 가정이 필요 없으므로, 다봉(multimodal) 분포나 강한 비선형 시스템을 다룰 수 있다.

사후 분포의 입자 근사:

$$p(\mathbf{x}_k \mid \mathbf{z}_{1:k}) \approx \sum_{i=1}^{N} w_k^{(i)} \delta(\mathbf{x}_k - \mathbf{x}_k^{(i)})$$

여기서:
- $N$: 입자 수
- $\mathbf{x}_k^{(i)}$: $i$번째 입자의 상태
- $w_k^{(i)}$: $i$번째 입자의 가중치 ($\sum_{i=1}^N w_k^{(i)} = 1$)
- $\delta(\cdot)$: 디랙 델타 함수

$N \to \infty$이면 입자 분포는 실제 사후 분포에 수렴한다. 실전에서는 유한 개의 입자로 근사하며, 입자 수는 문제의 복잡도와 상태 공간의 차원에 따라 결정한다.

### 4.3.2 Importance Sampling

사후 분포 $p(\mathbf{x}_k \mid \mathbf{z}_{1:k})$에서 직접 샘플링하는 것은 일반적으로 불가능하다. 대신 **제안 분포(proposal distribution)** $q(\mathbf{x}_k \mid \mathbf{x}_{k-1}, \mathbf{z}_k)$에서 샘플을 뽑고, 중요도 가중치(importance weight)로 보정한다.

재귀적 가중치 갱신을 유도한다. Bayes 정리에서:

$$p(\mathbf{x}_{0:k} \mid \mathbf{z}_{1:k}) = \frac{p(\mathbf{z}_k \mid \mathbf{x}_k) \, p(\mathbf{x}_k \mid \mathbf{x}_{k-1}) \, p(\mathbf{x}_{0:k-1} \mid \mathbf{z}_{1:k-1})}{p(\mathbf{z}_k \mid \mathbf{z}_{1:k-1})}$$

제안 분포로 나누어 중요도 비율을 구하면:

$$w_k^{(i)} \propto w_{k-1}^{(i)} \cdot \frac{p(\mathbf{z}_k \mid \mathbf{x}_k^{(i)}) \, p(\mathbf{x}_k^{(i)} \mid \mathbf{x}_{k-1}^{(i)})}{q(\mathbf{x}_k^{(i)} \mid \mathbf{x}_{k-1}^{(i)}, \mathbf{z}_k)}$$

**가장 간단한 제안 분포**: 전이 사전(transition prior)을 제안 분포로 사용, 즉 $q(\mathbf{x}_k \mid \mathbf{x}_{k-1}, \mathbf{z}_k) = p(\mathbf{x}_k \mid \mathbf{x}_{k-1})$. 이 경우 가중치가 간단해진다:

$$w_k^{(i)} \propto w_{k-1}^{(i)} \cdot p(\mathbf{z}_k \mid \mathbf{x}_k^{(i)})$$

각 입자의 가중치는 해당 입자 위치에서의 관측 likelihood에 비례한다. 직관적으로, 관측과 일치하는 입자는 높은 가중치를, 일치하지 않는 입자는 낮은 가중치를 받는다.

최적 제안 분포는 $q^*(\mathbf{x}_k \mid \mathbf{x}_{k-1}^{(i)}, \mathbf{z}_k) = p(\mathbf{x}_k \mid \mathbf{x}_{k-1}^{(i)}, \mathbf{z}_k)$이지만, 대부분의 경우 이를 구할 수 없다.

### 4.3.3 Resampling

제안 분포로 입자를 전파하다 보면 **가중치 퇴화(weight degeneracy)** 문제가 발생한다: 몇 개의 입자에 가중치가 집중되고 나머지 입자들은 무시해도 될 만큼 작은 가중치를 갖게 된다. 이러면 사실상 소수의 입자만 유효하므로 근사 품질이 급격히 저하된다.

유효 입자 수(effective sample size)로 퇴화를 진단한다:

$$N_{\text{eff}} = \frac{1}{\sum_{i=1}^N (w_k^{(i)})^2}$$

$N_{\text{eff}} < N_{\text{threshold}}$ (보통 $N/2$)이면 리샘플링을 수행한다.

**리샘플링**: 가중치가 높은 입자를 복제하고 낮은 입자를 제거하여 가중치를 균등하게 만드는 과정.

주요 리샘플링 전략:

**Multinomial Resampling**: 가중치를 확률로 사용하여 $N$번 독립 추출. 가장 직관적이지만 분산이 크다.

**Systematic Resampling**: 하나의 균등 난수 $U_0 \sim \text{Uniform}(0, 1/N)$을 생성하고, $U_i = U_0 + (i-1)/N$으로 CDF를 타서 리샘플링. 분산이 가장 작아 실전에서 가장 많이 사용된다.

**Stratified Resampling**: 각 층에서 독립 균등 난수를 사용. Systematic과 multinomial의 중간.

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
        # 1. 예측: 전이 모델을 통해 입자 전파
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

### 4.3.4 Rao-Blackwellized Particle Filter (RBPF)

RBPF는 상태 공간을 두 부분으로 분할하여 일부는 입자로, 나머지는 해석적(예: 칼만 필터)으로 추정하는 방법이다. 이렇게 하면 입자 필터가 담당하는 차원이 줄어들어 훨씬 적은 수의 입자로도 좋은 근사가 가능하다.

상태를 $\mathbf{x} = [\mathbf{x}_1, \mathbf{x}_2]$로 분할한다고 하자. $\mathbf{x}_1$이 주어지면 $\mathbf{x}_2$의 조건부 분포가 해석적으로 (예: 가우시안으로) 추적 가능하다면:

$$p(\mathbf{x}_1, \mathbf{x}_2 \mid \mathbf{z}_{1:k}) = p(\mathbf{x}_2 \mid \mathbf{x}_1, \mathbf{z}_{1:k}) \cdot p(\mathbf{x}_1 \mid \mathbf{z}_{1:k})$$

- $p(\mathbf{x}_1 \mid \mathbf{z}_{1:k})$: 입자 필터로 근사
- $p(\mathbf{x}_2 \mid \mathbf{x}_1, \mathbf{z}_{1:k})$: 각 입자에 부착된 칼만 필터로 추적

**Rao-Blackwell 정리**: 이 분할에 의한 추정량의 분산은 순수 입자 필터의 분산보다 항상 작거나 같다.

$$\text{Var}[\hat{\mathbf{x}}_{\text{RBPF}}] \leq \text{Var}[\hat{\mathbf{x}}_{\text{PF}}]$$

**FastSLAM과의 연결**: FastSLAM은 RBPF의 대표적 응용이다. 로봇 경로(trajectory)를 입자로, 랜드마크 위치를 각 입자에 부착된 개별 EKF로 추적한다.

- $\mathbf{x}_1 = \mathbf{x}_{0:k}^{\text{robot}}$ (로봇 경로) → 입자 필터
- $\mathbf{x}_2 = \{\mathbf{m}_1, \ldots, \mathbf{m}_M\}$ (랜드마크) → 각 입자마다 $M$개의 독립 2D EKF

로봇 경로가 주어지면 각 랜드마크의 관측들은 서로 독립이 되므로(조건부 독립), 하나의 거대한 EKF 대신 $M$개의 소형 EKF를 독립적으로 운영할 수 있다. 이것이 EKF-SLAM의 $O(M^2)$ 복잡도를 FastSLAM의 $O(M \log M)$으로 낮추는 핵심이다.

### 4.3.5 Particle Filter의 한계와 현재 위치

PF의 가장 큰 한계는 **차원의 저주(curse of dimensionality)**다. 상태 공간의 차원이 높아지면 의미 있는 근사를 위해 필요한 입자 수가 지수적으로 증가한다. 전형적인 VIO/LIO 시스템의 상태 벡터는 15차원 이상이므로, 순수 PF는 실용적이지 않다.

따라서 현대 로봇 시스템에서 PF의 역할은 제한적이다:

- **2D SLAM (RBPF 기반)**: GMapping 같은 2D 점유 격자 SLAM에서 여전히 사용. 로봇 자세(3-DoF)만 입자로, 맵은 각 입자에 부착된 격자로 관리.
- **Global Localization (MCL)**: 이미 만들어진 맵에서 로봇의 초기 위치를 모를 때 (kidnapped robot problem). 다봉 분포를 자연스럽게 표현할 수 있으므로 적합하다.
- **저차원 비선형 추정**: 상태 차원이 낮고 비선형성이 심한 특수 문제.

고차원 상태 추정은 Kalman 필터 계열 (특히 ESKF) 또는 factor graph 기반 최적화가 지배하고 있다.

---

## 4.4 Smoothing vs Filtering

### 4.4.1 Filtering과 Smoothing의 차이

**Filtering**: 현재까지의 관측을 사용하여 현재 상태를 추정한다.
$$p(\mathbf{x}_k \mid \mathbf{z}_{1:k})$$

**Smoothing**: 모든 관측 (미래 포함)을 사용하여 과거 상태를 추정한다.
$$p(\mathbf{x}_k \mid \mathbf{z}_{1:T}), \quad k < T$$

Smoother는 "미래의 관측"을 활용하므로, 같은 시각의 추정치가 filter보다 항상 같거나 더 정확하다. 단, 실시간 추정에는 filter가 필요하고, smoother는 후처리(batch) 또는 지연(fixed-lag) 형태로 사용된다.

### 4.4.2 Fixed-Lag Smoother

Fixed-lag smoother는 현재 시각 $k$에서 $L$단계 이전까지의 관측을 활용하여 시각 $k-L$의 상태를 추정한다:

$$p(\mathbf{x}_{k-L} \mid \mathbf{z}_{1:k})$$

이것은 filtering과 full smoothing의 절충이다. $L$만큼의 지연(latency)을 허용하면 더 나은 추정을 얻을 수 있다.

VINS-Mono, ORB-SLAM3 등의 슬라이딩 윈도우 최적화 시스템은 사실상 fixed-lag smoother다. 윈도우 내의 키프레임들을 동시에 최적화하므로, 단순 필터링보다 정확하다.

### 4.4.3 Full Smoothing (Batch Optimization)

전체 궤적의 모든 상태를 모든 관측을 사용하여 동시에 추정한다:

$$p(\mathbf{x}_{0:T} \mid \mathbf{z}_{1:T})$$

이것을 MAP(Maximum A Posteriori) 추정으로 풀면:

$$\mathbf{x}_{0:T}^* = \arg\max_{\mathbf{x}_{0:T}} p(\mathbf{x}_{0:T} \mid \mathbf{z}_{1:T})$$

가우시안 노이즈 가정 하에서 MAP는 비선형 최소자승(Nonlinear Least Squares, NLS) 문제가 된다. 이것이 factor graph 기반 최적화의 출발점이다.

### 4.4.4 왜 현대 SLAM은 Filtering에서 Optimization으로 갔는가

2000년대 초반까지 SLAM은 EKF-SLAM이 주류였다. 그러나 점차 graph-based optimization(= batch smoothing)으로 이동했다. 그 이유:

**1. 선형화 지점의 문제 (Linearization Point)**

EKF는 "한 번 선형화하면 끝"이다. 시각 $k$에서의 자코비안은 시각 $k$의 추정치에서 계산되고, 이후에 더 나은 추정치를 얻어도 과거의 자코비안을 수정하지 않는다. 반면 batch optimization은 전체 궤적에 대해 자코비안을 현재 추정치에서 반복적으로 재계산(relinearize)할 수 있다.

[Strasdat et al. (2012) "Visual SLAM: Why Filter?"](https://doi.org/10.1016/j.imavis.2012.02.009)가 이 논증을 체계적으로 제시했다: 같은 계산량이 주어지면, optimization에 더 많은 키프레임을 넣는 것이 filtering에 더 많은 관측을 넣는 것보다 정확도가 높다.

**2. 일관성(Consistency) 문제**

EKF-SLAM은 관측 가능성(observability) 조건을 위반하는 경향이 있다. 특히 SLAM에서 첫 번째 포즈가 고정되어야(unobservable) 하는데, EKF의 선형화가 이를 깨뜨려 4개의 unobservable direction이 observable해지는 inconsistency가 발생한다. 이 문제는 FEJ(First-Estimate Jacobian)로 완화할 수 있지만 완전히 해결되지는 않는다.

**3. 스케일러빌리티(Scalability)**

EKF-SLAM에서 $M$개 랜드마크의 공분산 행렬 크기는 $O((n + 3M)^2)$이고, 각 갱신의 계산량은 $O((n+3M)^2)$이다. 맵이 커지면 감당할 수 없다.

Graph-based SLAM에서는 정보 행렬(information matrix, Hessian)이 **희소(sparse)**하다. 각 변수(포즈, 랜드마크)는 직접 관측한 다른 변수들과만 연결되므로, 정보 행렬의 대부분이 0이다. 이 희소성을 활용하면 변수 수에 거의 선형인 시간에 최적화할 수 있다.

**4. 루프 클로저(Loop Closure) 처리의 자연스러움**

필터 기반 시스템에서 루프 클로저를 처리하려면 과거 상태의 공분산 정보를 유지해야 하며, 이것이 계산량을 크게 증가시킨다. Graph-based 시스템에서는 루프 클로저를 단순히 새로운 factor(constraint)로 추가하고 전체 그래프를 재최적화하면 된다.

| 관점 | Filtering (EKF) | Optimization (Graph) |
|------|----------------|---------------------|
| 선형화 | 한 번, 고정 | 반복 재선형화 가능 |
| 과거 상태 | 마지널라이즈되어 소실 | 전부 유지 |
| 루프 클로저 | 어렵고 비쌈 | factor 추가로 자연스럽게 |
| 정보 행렬 구조 | Dense | Sparse |
| 계산량 (M 랜드마크) | $O(M^2)$ per step | $O(M)$ (sparse 활용) |
| 일관성 | FEJ 등 추가 조치 필요 | 재선형화로 자연 완화 |

그러나 필터 기반이 완전히 사라진 것은 아니다. [MSCKF (Mourikis & Roumeliotis, 2007)](https://ieeexplore.ieee.org/document/4209642)와 OpenVINS는 EKF 기반이면서도 경쟁력 있는 성능을 보여주며, 특히 계산 자원이 극히 제한된 환경(마이크로 UAV 등)에서 여전히 유용하다. FAST-LIO2의 IEKF도 필터 기반이지만, ikd-tree와 결합하여 최적화 기반 시스템과 대등한 정확도를 달성한다.

---

## 4.5 Factor Graph & Optimization

### 4.5.1 Factor Graph 표현

Factor graph는 확률적 그래피컬 모델(probabilistic graphical model)의 한 종류로, 변수(variable)와 인자(factor)로 구성된 이분 그래프(bipartite graph)다.

$$p(\mathbf{X} \mid \mathbf{Z}) \propto \prod_{i} f_i(\mathbf{X}_i)$$

여기서:
- $\mathbf{X} = \{\mathbf{x}_0, \mathbf{x}_1, \ldots, \mathbf{x}_T, \mathbf{l}_1, \ldots, \mathbf{l}_M\}$: 변수 노드 (포즈, 랜드마크, 바이어스 등)
- $f_i(\mathbf{X}_i)$: $i$번째 factor. 변수의 부분집합 $\mathbf{X}_i$에 대한 "에너지 함수" 또는 "확률적 구속 조건"
- $\mathbf{Z}$: 모든 관측

각 factor는 특정 관측이나 사전 정보에 대응한다:

| Factor 유형 | 연결하는 변수 | 의미 |
|------------|-------------|------|
| Prior factor | $\mathbf{x}_0$ | 초기 상태 사전 분포 |
| Odometry factor | $\mathbf{x}_{k-1}, \mathbf{x}_k$ | 연속 포즈 간 상대 운동 |
| IMU preintegration factor | $\mathbf{x}_{i}, \mathbf{x}_{j}, \mathbf{v}_i, \mathbf{v}_j, \mathbf{b}_i$ | 키프레임 간 IMU 적분 |
| Vision factor | $\mathbf{x}_k, \mathbf{l}_m$ | 카메라에서 랜드마크 관측 |
| LiDAR factor | $\mathbf{x}_k$ | 포인트-투-플레인/엣지 정합 |
| GPS factor | $\mathbf{x}_k$ | 절대 위치 관측 |
| Loop closure factor | $\mathbf{x}_i, \mathbf{x}_j$ | 루프 클로저 상대 포즈 |

factor graph의 핵심 강점: **모듈성(modularity)**. 새로운 센서를 추가하려면 해당 센서에 대응하는 factor를 정의하고 그래프에 추가하기만 하면 된다. 기존 factor들은 수정할 필요가 없다.

### 4.5.2 MAP Inference = Nonlinear Least Squares

가우시안 노이즈 모델을 가정하면, 각 factor는 다음 형태가 된다:

$$f_i(\mathbf{X}_i) \propto \exp\left(-\frac{1}{2} \|\mathbf{r}_i(\mathbf{X}_i)\|^2_{\boldsymbol{\Sigma}_i}\right)$$

여기서 $\mathbf{r}_i(\mathbf{X}_i)$는 잔차(residual)이고, $\|\mathbf{r}\|^2_{\boldsymbol{\Sigma}} = \mathbf{r}^\top \boldsymbol{\Sigma}^{-1} \mathbf{r}$은 Mahalanobis 거리의 제곱이다.

예를 들어, 관측 factor에서:
$$\mathbf{r}_i = \mathbf{z}_i - h_i(\mathbf{X}_i), \quad \boldsymbol{\Sigma}_i = \mathbf{R}_i \text{ (관측 노이즈 공분산)}$$

전체 사후 분포의 MAP 추정:

$$\mathbf{X}^* = \arg\max_\mathbf{X} p(\mathbf{X} \mid \mathbf{Z}) = \arg\max_\mathbf{X} \prod_i f_i(\mathbf{X}_i)$$

로그를 취하고 부호를 뒤집으면:

$$\boxed{\mathbf{X}^* = \arg\min_\mathbf{X} \sum_i \|\mathbf{r}_i(\mathbf{X}_i)\|^2_{\boldsymbol{\Sigma}_i}}$$

이것이 **NLS** 문제다. 확률적 추론 문제가 최적화 문제로 바뀐다. [Dellaert & Kaess (2017)의 "Factor Graphs for Robot Perception"](https://doi.org/10.1561/2300000043) 튜토리얼이 이 과정을 139페이지에 걸쳐 풀어낸다.

### 4.5.3 Gauss-Newton Method

NLS 문제를 풀기 위해 Gauss-Newton(GN) 방법을 사용한다. 잔차를 현재 추정치 $\mathbf{X}^{(0)}$ 근방에서 1차 테일러 전개한다:

$$\mathbf{r}_i(\mathbf{X}^{(0)} \boxplus \Delta\mathbf{X}) \approx \mathbf{r}_i(\mathbf{X}^{(0)}) + \mathbf{J}_i \Delta\mathbf{X}$$

여기서 $\mathbf{J}_i = \frac{\partial \mathbf{r}_i}{\partial \mathbf{X}}\big|_{\mathbf{X}^{(0)}}$는 잔차의 자코비안이고, $\boxplus$는 매니폴드 위의 증분 연산이다.

대입하면:

$$\sum_i \|\mathbf{r}_i + \mathbf{J}_i \Delta\mathbf{X}\|^2_{\boldsymbol{\Sigma}_i}$$

$\mathbf{r}_i' = \boldsymbol{\Sigma}_i^{-1/2} \mathbf{r}_i$, $\mathbf{J}_i' = \boldsymbol{\Sigma}_i^{-1/2} \mathbf{J}_i$로 화이트닝하면, 표준 최소자승 문제가 된다:

$$\sum_i \|\mathbf{r}_i' + \mathbf{J}_i' \Delta\mathbf{X}\|^2$$

$\Delta\mathbf{X}$에 대해 미분하여 0으로 놓으면 **정규 방정식(normal equation)**을 얻는다:

$$\underbrace{\left(\sum_i \mathbf{J}_i'^\top \mathbf{J}_i'\right)}_{\mathbf{H}} \Delta\mathbf{X} = -\underbrace{\sum_i \mathbf{J}_i'^\top \mathbf{r}_i'}_{\mathbf{b}}$$

$$\boxed{\mathbf{H} \Delta\mathbf{X} = -\mathbf{b}}$$

여기서 $\mathbf{H} = \mathbf{J}^\top \boldsymbol{\Sigma}^{-1} \mathbf{J} \in \mathbb{R}^{N \times N}$는 근사 Hessian (정보 행렬)이고, $\mathbf{b} = \mathbf{J}^\top \boldsymbol{\Sigma}^{-1} \mathbf{r}$은 gradient이다.

SLAM 문제에서 $\mathbf{H}$는 **희소**하다. 각 factor의 자코비안 $\mathbf{J}_i$는 해당 factor에 연결된 변수에 대한 열만 비영이고 나머지는 0이다. 따라서 $\mathbf{H}$의 비영 원소는 factor graph의 간선에 대응하며, 그래프가 희소하면 $\mathbf{H}$도 희소하다.

Gauss-Newton 반복:

$$\mathbf{X}^{(k+1)} = \mathbf{X}^{(k)} \boxplus \Delta\mathbf{X}^{(k)}$$

각 반복에서 정규 방정식을 풀어야 한다. 희소 선형 시스템의 풀이에는 **희소 Cholesky 분해** ($\mathbf{H} = \mathbf{L}\mathbf{L}^\top$, 전방/후방 대입)를 사용하며, 변수 순서(variable ordering)에 따라 $\mathbf{L}$의 fill-in이 달라지므로 COLAMD 등의 근사 최적 순서(approximate minimum degree ordering)를 사용한다.

### 4.5.4 Levenberg-Marquardt Method

Gauss-Newton은 순수한 근사 2차 방법이지만, 초기값이 나쁘거나 비선형성이 심하면 발산할 수 있다. **Levenberg-Marquardt(LM)** 방법은 GN과 경사 하강법(gradient descent)의 절충으로, 정규화 항을 추가한다:

$$(\mathbf{H} + \lambda \mathbf{I}) \Delta\mathbf{X} = -\mathbf{b}$$

- $\lambda$가 작으면 → GN에 가까움 (빠른 수렴, 2차 수렴)
- $\lambda$가 크면 → gradient descent에 가까움 (작은 스텝, 안전)

$\lambda$의 조절 전략: 갱신이 비용 함수를 감소시키면 $\lambda$를 줄이고 (GN 모드), 증가시키면 $\lambda$를 키운다 (보수적 모드).

### 4.5.5 매니폴드 위의 최적화 (Optimization on Manifolds)

3D 자세 $\mathbf{T} \in SE(3)$를 최적화할 때, $SE(3)$는 유클리드 공간이 아닌 매니폴드이므로 일반 덧셈을 쓸 수 없다. 표준 해법은 **retraction** (또는 **exponential map**)이다.

현재 추정치 $\mathbf{T}^{(k)}$ 근방에서 접선 공간(tangent space) $\boldsymbol{\xi} \in \mathbb{R}^6$의 증분을 정의하고:

$$\mathbf{T}^{(k+1)} = \mathbf{T}^{(k)} \cdot \text{Exp}(\boldsymbol{\xi})$$

또는:

$$\mathbf{T}^{(k+1)} = \text{Exp}(\boldsymbol{\xi}) \cdot \mathbf{T}^{(k)}$$

(왼쪽/오른쪽 증분의 선택은 convention에 따름)

여기서 $\text{Exp}: \mathbb{R}^6 \to SE(3)$는 Lie group의 exponential map이다. $\boldsymbol{\xi} = [\boldsymbol{\rho}^\top, \boldsymbol{\phi}^\top]^\top$에서 $\boldsymbol{\rho} \in \mathbb{R}^3$는 이동, $\boldsymbol{\phi} \in \mathbb{R}^3$는 회전(각축 표현)이다.

SO(3)에서의 exponential map (Rodrigues' formula):

$$\text{Exp}(\boldsymbol{\phi}) = \mathbf{I} + \frac{\sin\theta}{\theta}[\boldsymbol{\phi}]_\times + \frac{1 - \cos\theta}{\theta^2}[\boldsymbol{\phi}]_\times^2 \in \mathbb{R}^{3 \times 3}$$

여기서 $\theta = \|\boldsymbol{\phi}\|$이다.

반대로 $\text{Log}: SE(3) \to \mathbb{R}^6$은 logarithmic map이다.

Gauss-Newton/LM의 정규 방정식은 접선 공간에서의 증분 $\boldsymbol{\xi}$에 대해 풀고, exponential map으로 매니폴드 위의 상태를 갱신한다. 자코비안도 접선 공간에 대해 계산한다:

$$\mathbf{J}_i = \frac{\partial \mathbf{r}_i}{\partial \boldsymbol{\xi}}\bigg|_{\boldsymbol{\xi}=\mathbf{0}}$$

### 4.5.6 iSAM2: Incremental Smoothing

Batch 최적화를 매 키프레임마다 처음부터 다시 수행하면 시간이 비현실적으로 오래 걸린다. **iSAM2** ([Kaess et al., 2012](https://doi.org/10.1177/0278364911430419))는 Bayes tree 자료구조를 사용하여 incremental하게 최적화를 수행한다.

핵심 아이디어:

1. **Bayes Tree**: factor graph의 elimination 결과를 방향성 트리로 표현. 각 노드는 clique(변수의 부분집합)에 대한 조건부 확률 밀도를 저장.

2. **Incremental Update**: 새 factor가 추가되면, 영향을 받는 clique만 재계산. 트리의 대부분은 변경되지 않음.

3. **Fluid Relinearization**: 선형화 지점이 현재 추정치와 크게 달라진 변수만 선택적으로 재선형화. 주기적 batch 처리 불필요.

4. **변수 순서 재정렬(Variable Reordering)**: 새 변수/factor 추가 시 전체 재정렬 없이 영향받는 부분만 지역적으로 재정렬.

iSAM2는 GTSAM의 핵심 알고리즘으로, LIO-SAM·VINS-Mono 등 현대 SLAM 시스템 다수가 이를 백엔드로 쓴다.

> **최근 동향 — Continuous-Time Factor Graph**: 이산 키프레임 기반 factor graph를 **연속 시간(continuous-time)**으로 확장하는 연구가 활발하다. [Wong et al. (2024)](https://arxiv.org/abs/2402.06174)는 Gaussian Process motion prior를 사용하여 radar-inertial 및 LiDAR-inertial odometry를 연속 시간 factor graph로 통합하고, 비동기 센서 측정을 자연스럽게 처리할 수 있음을 보였다.

### 4.5.7 GTSAM / Ceres / g2o 비교

| 특성 | GTSAM | Ceres Solver | g2o |
|------|-------|-------------|-----|
| 개발 | Georgia Tech ([Dellaert](https://gtsam.org/)) | Google ([Ceres](http://ceres-solver.org/)) | [Kümmerle et al.](https://doi.org/10.1109/ICRA.2011.5979949) |
| 핵심 철학 | Factor graph + Bayes tree | 범용 NLS solver | Graph optimization |
| Incremental | iSAM2 (native) | 없음 (batch) | 없음 (batch) |
| 매니폴드 | 내장 (Rot2, Rot3, Pose2, Pose3, ...) | Local parameterization | 내장 |
| IMU Preintegration | 내장 (`PreintegratedImuMeasurements`) | 사용자 정의 | 사용자 정의 |
| 자동 미분 | 수치 미분 가능 | 자동 미분 (ceres::AutoDiffCostFunction) | 없음 |
| 언어 | C++ (Python 바인딩) | C++ | C++ |
| 대표 사용처 | LIO-SAM, VINS-Mono | Cartographer, ORB-SLAM3 BA | 많은 SLAM 시스템의 포즈 그래프 |
| 학습 곡선 | Factor 정의만 하면 됨 | Cost function 정의 | Vertex/Edge 정의 |

```python
# GTSAM을 이용한 간단한 Pose Graph Optimization 예제
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
    # 초기값에 노이즈를 약간 추가 (실제로는 odometry 누적값)
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

---

## 4.6 IMU Preintegration

### 4.6.1 왜 Preintegration이 필요한가

IMU는 보통 200~1000Hz로 가속도와 각속도를 측정하지만, 카메라/LiDAR 키프레임은 10~30Hz 간격이다. 두 키프레임 $i, j$ 사이에 수백 개의 IMU 측정이 존재한다.

**Naive 접근: 직접 적분**

두 키프레임 사이의 IMU 측정을 적분하여 상대 포즈를 구하면:

$$\mathbf{R}_j = \mathbf{R}_i \prod_{k=i}^{j-1} \text{Exp}((\tilde{\boldsymbol{\omega}}_k - \mathbf{b}_g) \Delta t)$$
$$\mathbf{v}_j = \mathbf{v}_i + \mathbf{g} \Delta t_{ij} + \sum_{k=i}^{j-1} \mathbf{R}_k (\tilde{\mathbf{a}}_k - \mathbf{b}_a) \Delta t$$
$$\mathbf{p}_j = \mathbf{p}_i + \mathbf{v}_i \Delta t_{ij} + \frac{1}{2}\mathbf{g}\Delta t_{ij}^2 + \sum_{k=i}^{j-1}\left[\mathbf{v}_k \Delta t + \frac{1}{2}\mathbf{R}_k(\tilde{\mathbf{a}}_k - \mathbf{b}_a)\Delta t^2\right]$$

문제: 이 적분은 키프레임 $i$의 상태 $(\mathbf{R}_i, \mathbf{v}_i, \mathbf{p}_i)$와 바이어스 $(\mathbf{b}_g, \mathbf{b}_a)$에 의존한다. 최적화 반복에서 $\mathbf{R}_i, \mathbf{v}_i, \mathbf{p}_i$의 추정치가 바뀌면, 모든 중간 상태를 재적분해야 한다. 바이어스 추정치가 바뀔 때도 마찬가지다. 이것은 수백 번의 exponential map 계산이 매 최적화 반복마다 필요함을 의미한다.

**Preintegration의 해법**: 키프레임 $i$의 body frame 기준의 상대 변화량을 정의하여, 이 값이 키프레임 $i$의 글로벌 포즈와 무관하게 만든다. 바이어스에 대해서는 1차 자코비안 보정으로 재적분을 회피한다.

### 4.6.2 On-Manifold Preintegration 유도

[Forster et al. (2017)](https://doi.org/10.1109/TRO.2016.2597321)의 on-manifold preintegration을 단계별로 유도한다. 원논문은 [arXiv:1512.02363](https://arxiv.org/abs/1512.02363)에서도 확인할 수 있다.

#### Step 1: 상대 변화량 정의

글로벌 프레임에서의 적분 방정식을 키프레임 $i$의 body frame 기준으로 재배열한다. 글로벌 변수($\mathbf{R}_i, \mathbf{v}_i, \mathbf{p}_i$)를 좌변으로 이동:

$$\Delta\mathbf{R}_{ij} \triangleq \mathbf{R}_i^\top \mathbf{R}_j = \prod_{k=i}^{j-1} \text{Exp}((\tilde{\boldsymbol{\omega}}_k - \mathbf{b}_g^i)\Delta t) \in SO(3)$$

$$\Delta\mathbf{v}_{ij} \triangleq \mathbf{R}_i^\top (\mathbf{v}_j - \mathbf{v}_i - \mathbf{g}\Delta t_{ij}) = \sum_{k=i}^{j-1} \Delta\mathbf{R}_{ik}(\tilde{\mathbf{a}}_k - \mathbf{b}_a^i)\Delta t \in \mathbb{R}^3$$

$$\Delta\mathbf{p}_{ij} \triangleq \mathbf{R}_i^\top (\mathbf{p}_j - \mathbf{p}_i - \mathbf{v}_i \Delta t_{ij} - \frac{1}{2}\mathbf{g}\Delta t_{ij}^2) = \sum_{k=i}^{j-1}\left[\Delta\mathbf{v}_{ik}\Delta t + \frac{1}{2}\Delta\mathbf{R}_{ik}(\tilde{\mathbf{a}}_k - \mathbf{b}_a^i)\Delta t^2\right] \in \mathbb{R}^3$$

핵심 관찰: **우변은 IMU 측정값과 바이어스 추정치에만 의존하고, 키프레임 $i$의 글로벌 포즈 $(\mathbf{R}_i, \mathbf{v}_i, \mathbf{p}_i)$와 무관하다.** 따라서 키프레임 포즈가 최적화로 바뀌어도 우변을 재계산할 필요가 없다.

#### Step 2: 재귀적 계산 (On-Manifold)

Preintegrated 측정은 재귀적으로 누적 계산할 수 있다:

$$\Delta\mathbf{R}_{i,k+1} = \Delta\mathbf{R}_{ik} \cdot \text{Exp}((\tilde{\boldsymbol{\omega}}_k - \mathbf{b}_g)\Delta t) \in SO(3)$$
$$\Delta\mathbf{v}_{i,k+1} = \Delta\mathbf{v}_{ik} + \Delta\mathbf{R}_{ik}(\tilde{\mathbf{a}}_k - \mathbf{b}_a)\Delta t \in \mathbb{R}^3$$
$$\Delta\mathbf{p}_{i,k+1} = \Delta\mathbf{p}_{ik} + \Delta\mathbf{v}_{ik}\Delta t + \frac{1}{2}\Delta\mathbf{R}_{ik}(\tilde{\mathbf{a}}_k - \mathbf{b}_a)\Delta t^2 \in \mathbb{R}^3$$

초기값: $\Delta\mathbf{R}_{ii} = \mathbf{I}_{3\times 3}$, $\Delta\mathbf{v}_{ii} = \mathbf{0}$, $\Delta\mathbf{p}_{ii} = \mathbf{0}$.

"On-manifold"의 의미: 회전 $\Delta\mathbf{R}_{ij}$를 직접 $SO(3)$ 위에서 누적한다. 오일러 각이나 쿼터니언 정규화 같은 임시방편이 필요 없다.

#### Step 3: 바이어스 변화에 대한 1차 보정

최적화 과정에서 바이어스 추정치가 $\mathbf{b} \to \mathbf{b} + \delta\mathbf{b}$로 변하면, preintegrated 측정도 변한다. 그러나 전체 재적분을 피하기 위해 1차 테일러 전개로 보정한다:

**회전 보정**:
$$\Delta\hat{\mathbf{R}}_{ij}(\mathbf{b}_g + \delta\mathbf{b}_g) \approx \Delta\hat{\mathbf{R}}_{ij}(\mathbf{b}_g) \cdot \text{Exp}\left(\frac{\partial \Delta\bar{\mathbf{R}}_{ij}}{\partial \mathbf{b}_g} \delta\mathbf{b}_g\right)$$

**속도 보정**:
$$\Delta\hat{\mathbf{v}}_{ij}(\mathbf{b} + \delta\mathbf{b}) \approx \Delta\hat{\mathbf{v}}_{ij}(\mathbf{b}) + \frac{\partial \Delta\bar{\mathbf{v}}_{ij}}{\partial \mathbf{b}_g} \delta\mathbf{b}_g + \frac{\partial \Delta\bar{\mathbf{v}}_{ij}}{\partial \mathbf{b}_a} \delta\mathbf{b}_a$$

**위치 보정**:
$$\Delta\hat{\mathbf{p}}_{ij}(\mathbf{b} + \delta\mathbf{b}) \approx \Delta\hat{\mathbf{p}}_{ij}(\mathbf{b}) + \frac{\partial \Delta\bar{\mathbf{p}}_{ij}}{\partial \mathbf{b}_g} \delta\mathbf{b}_g + \frac{\partial \Delta\bar{\mathbf{p}}_{ij}}{\partial \mathbf{b}_a} \delta\mathbf{b}_a$$

자코비안 $\frac{\partial \Delta\bar{\mathbf{R}}_{ij}}{\partial \mathbf{b}_g}$ 등은 preintegration 과정에서 재귀적으로 누적 계산한다. 예를 들어 회전의 바이어스 자코비안:

$$\frac{\partial \Delta\bar{\mathbf{R}}_{i,k+1}}{\partial \mathbf{b}_g} = -\Delta\bar{\mathbf{R}}_{k,k+1}^\top \text{Jr}((\tilde{\boldsymbol{\omega}}_k - \mathbf{b}_g)\Delta t) \Delta t + \Delta\bar{\mathbf{R}}_{k,k+1}^\top \frac{\partial \Delta\bar{\mathbf{R}}_{ik}}{\partial \mathbf{b}_g}$$

여기서 $\text{Jr}(\boldsymbol{\phi})$는 SO(3)의 right Jacobian:

$$\text{Jr}(\boldsymbol{\phi}) = \mathbf{I} - \frac{1 - \cos\theta}{\theta^2}[\boldsymbol{\phi}]_\times + \frac{\theta - \sin\theta}{\theta^3}[\boldsymbol{\phi}]_\times^2, \quad \theta = \|\boldsymbol{\phi}\|$$

바이어스 변화가 작으면 ($\|\delta\mathbf{b}\|$가 작으면) 이 1차 보정은 충분히 정확하다. 바이어스가 크게 변하면 preintegration을 처음부터 다시 계산하지만, 실전에서는 거의 발생하지 않는다.

#### Step 4: 공분산 전파

IMU 측정 노이즈 $\mathbf{n}_g, \mathbf{n}_a \sim \mathcal{N}(\mathbf{0}, \boldsymbol{\Sigma})$가 preintegration을 통해 전파된다. 공분산도 재귀적으로 계산:

$$\boldsymbol{\Sigma}_{k+1} = \mathbf{A}_k \boldsymbol{\Sigma}_k \mathbf{A}_k^\top + \mathbf{B}_k \boldsymbol{\Sigma}_\eta \mathbf{B}_k^\top$$

여기서 $\mathbf{A}_k, \mathbf{B}_k$는 재귀적 전파의 자코비안이고, $\boldsymbol{\Sigma}_\eta$는 IMU 노이즈 공분산이다. 이 공분산 $\boldsymbol{\Sigma}_{ij}$는 factor graph에서 IMU factor의 정보 행렬 $\boldsymbol{\Sigma}_{ij}^{-1}$로 사용된다.

#### Step 5: Factor Graph에서의 IMU Factor

최종적으로 preintegrated 측정은 두 키프레임 $i, j$ 사이의 factor로 삽입된다. 잔차:

$$\mathbf{r}_{\Delta\mathbf{R}_{ij}} = \text{Log}\left(\Delta\hat{\mathbf{R}}_{ij}^\top \mathbf{R}_i^\top \mathbf{R}_j\right) \in \mathbb{R}^3$$

$$\mathbf{r}_{\Delta\mathbf{v}_{ij}} = \mathbf{R}_i^\top(\mathbf{v}_j - \mathbf{v}_i - \mathbf{g}\Delta t_{ij}) - \Delta\hat{\mathbf{v}}_{ij} \in \mathbb{R}^3$$

$$\mathbf{r}_{\Delta\mathbf{p}_{ij}} = \mathbf{R}_i^\top(\mathbf{p}_j - \mathbf{p}_i - \mathbf{v}_i\Delta t_{ij} - \frac{1}{2}\mathbf{g}\Delta t_{ij}^2) - \Delta\hat{\mathbf{p}}_{ij} \in \mathbb{R}^3$$

이 9차원 잔차가 IMU preintegration factor의 잔차이며, Mahalanobis 거리의 제곱 $\|\mathbf{r}\|^2_{\boldsymbol{\Sigma}_{ij}}$이 비용 함수에 추가된다.

### 4.6.3 코드로 보는 Preintegration

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
        self.d_R_d_bg = dR.T @ (self.d_R_d_bg - Jr * dt)
        
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
        Sigma_eta[0:3, 0:3] = np.eye(3) * self.sigma_g**2
        Sigma_eta[3:6, 3:6] = np.eye(3) * self.sigma_a**2
        
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


# GTSAM에서의 사용 예시
import gtsam

def create_imu_factor_gtsam():
    """GTSAM의 내장 IMU preintegration을 사용하는 예제."""
    # IMU 파라미터 설정
    imu_params = gtsam.PreintegrationParams.MakeSharedU(9.81)  # 중력 방향: +z
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
    # CombinedImuFactor는 키프레임 i, j의 pose, velocity, bias를 연결
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

---

## 4.7 Marginalization & Sliding Window

### 4.7.1 왜 Marginalization이 필요한가

Factor graph 기반 SLAM/VIO 시스템은 새로운 키프레임이 들어올 때마다 변수와 factor가 추가된다. 무한정 키프레임을 유지하면 최적화의 계산량이 끝없이 증가한다.

슬라이딩 윈도우(sliding window) 방식은 최근 $N$개의 키프레임만 유지하고 오래된 키프레임을 제거한다. 그런데 단순히 삭제하면 그 키프레임과 연결된 관측 정보가 모두 사라져 정확도가 저하된다.

**Marginalization**은 오래된 변수를 그래프에서 제거하면서도, 그 변수를 통해 연결된 변수들 간의 정보를 보존하는 방법이다. 제거되는 변수의 정보가 나머지 변수들에 대한 **사전 분포**로 변환된다.

### 4.7.2 Schur Complement

Marginalization의 수학적 핵심은 **Schur complement**다. 정보 행렬 $\mathbf{H}$를 마지널라이즈할 변수($\mathbf{x}_m$)와 유지할 변수($\mathbf{x}_r$)로 분할한다. 정규 방정식을 $\mathbf{H}\Delta\mathbf{x} = \mathbf{b}$로 쓴다 (§4.5.3의 $-\mathbf{b}$를 $\mathbf{b}$로 재정의, 즉 $\mathbf{b} \triangleq -\mathbf{J}^\top \boldsymbol{\Sigma}^{-1} \mathbf{r}$):

$$\mathbf{H} \Delta\mathbf{x} = \mathbf{b}$$

$$\begin{bmatrix} \mathbf{H}_{mm} & \mathbf{H}_{mr} \\ \mathbf{H}_{rm} & \mathbf{H}_{rr} \end{bmatrix} \begin{bmatrix} \Delta\mathbf{x}_m \\ \Delta\mathbf{x}_r \end{bmatrix} = \begin{bmatrix} \mathbf{b}_m \\ \mathbf{b}_r \end{bmatrix}$$

여기서:
- $\mathbf{x}_m$: 마지널라이즈(제거)할 변수
- $\mathbf{x}_r$: 유지(retain)할 변수
- $\mathbf{H}_{mm} \in \mathbb{R}^{n_m \times n_m}$: 제거할 변수들끼리의 정보 블록
- $\mathbf{H}_{mr} \in \mathbb{R}^{n_m \times n_r}$: 교차 정보 블록
- $\mathbf{H}_{rr} \in \mathbb{R}^{n_r \times n_r}$: 유지할 변수들끼리의 정보 블록

위 블록 행렬에서 $\Delta\mathbf{x}_m$을 소거하면:

$$\underbrace{(\mathbf{H}_{rr} - \mathbf{H}_{rm} \mathbf{H}_{mm}^{-1} \mathbf{H}_{mr})}_{\mathbf{H}^*} \Delta\mathbf{x}_r = \underbrace{\mathbf{b}_r - \mathbf{H}_{rm} \mathbf{H}_{mm}^{-1} \mathbf{b}_m}_{\mathbf{b}^*}$$

$\mathbf{H}^* = \mathbf{H}_{rr} - \mathbf{H}_{rm} \mathbf{H}_{mm}^{-1} \mathbf{H}_{mr}$이 **Schur complement**이며, 이것이 마지널라이즈 후 유지되는 변수에 대한 사전 factor(prior factor)의 정보 행렬이 된다.

**직관적 의미**: $\mathbf{x}_m$을 통해 간접적으로 연결되어 있던 $\mathbf{x}_r$의 변수들 사이에 직접 연결(fill-in)이 생긴다. $\mathbf{H}^*$는 $\mathbf{H}_{rr}$보다 dense하며, 이것은 마지널라이즈 전에는 없던 변수 간 상관 관계가 명시적으로 등장했음을 뜻한다.

```python
import numpy as np

def marginalize(H, b, indices_to_marginalize, indices_to_keep):
    """Schur complement를 이용한 변수 마지널라이제이션.
    
    Parameters
    ----------
    H : ndarray, shape (N, N) — 정보 행렬 (Hessian)
    b : ndarray, shape (N,) — gradient 벡터
    indices_to_marginalize : list of int — 제거할 변수의 인덱스
    indices_to_keep : list of int — 유지할 변수의 인덱스
    
    Returns
    -------
    H_star : ndarray — 마지널라이즈 후 정보 행렬
    b_star : ndarray — 마지널라이즈 후 gradient
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
print("→ x1과 x2 사이에 fill-in이 발생할 수 있음")
```

### 4.7.3 First-Estimate Jacobian (FEJ)

Marginalization에는 중요한 주의점이 있다. 마지널라이즈된 factor(prior)는 **고정된 선형화 지점**에서 계산된다. 이후 최적화 반복에서 유지된 변수의 추정치가 바뀌면, 마지널라이즈된 prior의 선형화 지점과 불일치가 생긴다.

이 불일치는 필터/스무더의 **일관성**을 해칠 수 있다. 구체적으로, 시스템의 관측 가능성(observability) 특성이 바뀌어, 실제로는 관측 불가능한 방향이 관측 가능한 것으로 잘못 처리될 수 있다.

**FEJ (First-Estimate Jacobian)** 전략: 마지널라이즈에 관련된 변수의 자코비안은 항상 **최초 추정치(first estimate)**에서 계산하고, 이후 추정치가 바뀌어도 자코비안을 갱신하지 않는다.

$$\mathbf{J}_{\text{FEJ}} = \left.\frac{\partial \mathbf{r}}{\partial \mathbf{x}}\right|_{\mathbf{x}^{(0)}}$$

여기서 $\mathbf{x}^{(0)}$는 해당 변수가 처음 추정된 시점의 값이다.

FEJ의 장점:
- 마지널라이즈된 prior와 현재 factor들이 같은 선형화 지점에서의 정보를 사용하므로 일관성이 유지된다
- MSCKF/OpenVINS에서 핵심적으로 사용 ([Li & Mourikis, 2013](https://doi.org/10.1177/0278364913481251))

FEJ의 단점:
- 최초 추정치가 부정확하면 수렴 속도가 느려질 수 있다
- 구현이 복잡해진다 (각 변수마다 "최초 추정치"를 별도 저장해야 함)

### 4.7.4 Sliding Window 구현의 실전 이슈

#### 이슈 1: 어떤 키프레임을 마지널라이즈할 것인가

VINS-Mono의 두 가지 전략:
- **최신 프레임이 키프레임이면**: 가장 오래된 키프레임을 마지널라이즈. 윈도우가 공간적으로 넓게 유지됨.
- **최신 프레임이 키프레임이 아니면**: 직전 non-keyframe을 마지널라이즈. 시각 정보만 버리고 IMU 정보는 인접 키프레임으로 전달.

#### 이슈 2: Fill-in에 의한 dense화

마지널라이즈가 반복되면 prior factor가 점점 dense해져서, 원래 희소했던 정보 행렬이 dense해질 수 있다. 이것은 계산 효율을 크게 저하시킨다.

대응 방법:
- Prior factor의 크기를 제한 (연결된 변수 수를 제한)
- 마지널라이즈 순서를 신중히 선택
- 정보 손실을 감수하고 일부 factor를 단순 삭제 (FAST-LIO2는 마지널라이즈 대신 오래된 점을 맵에서 삭제)

#### 이슈 3: 바이어스와 마지널라이제이션

IMU 바이어스는 모든 키프레임에 걸쳐 천천히 변하는 상태다. 키프레임을 마지널라이즈할 때 바이어스도 함께 마지널라이즈하면, 바이어스 정보가 prior에 고정되어 이후 바이어스 추정의 유연성이 줄어든다.

VINS-Mono의 접근: 바이어스를 마지널라이즈하지 않고 윈도우 내에서 계속 유지. 마지널라이즈 prior는 바이어스를 조건으로 하는 형태로 만든다.

#### 이슈 4: 수치 안정성

Schur complement 계산에서 $\mathbf{H}_{mm}^{-1}$의 역행렬이 필요하다. $\mathbf{H}_{mm}$이 나쁜 조건수(condition number)를 가지면 수치 불안정이 발생할 수 있다.

대응:
- $\mathbf{H}_{mm}$에 작은 정규화 항 추가: $(\mathbf{H}_{mm} + \epsilon \mathbf{I})^{-1}$
- LDL 분해를 사용하여 수치 안정성 확보
- 마지널라이즈 후 $\mathbf{H}^*$가 양의 반정치(positive semi-definite)인지 확인하고, 그렇지 않으면 가장 가까운 PSD 행렬로 보정

---

## 4장 요약

이 장에서는 센서 퓨전의 수학적 기반인 상태 추정 이론을 다루었다. 핵심 메시지를 정리한다:

1. **Bayesian Filtering Framework**는 prediction-update 재귀 구조로, 모든 상태 추정 방법의 공통 뼈대다. Chapman-Kolmogorov 방정식과 Bayes 정리가 이론적 근거이지만, 비선형 시스템에서는 근사가 필수적이다.

2. **Kalman Filter 계열**은 가우시안 근사를 통해 사후 분포를 평균과 공분산으로 추적한다. EKF는 1차 선형화, ESKF는 오차 상태에서의 선형화로 매니폴드 문제를 자연스럽게 처리하며, UKF는 sigma point 변환, IEKF는 반복 선형화로 비선형성에 대응한다. 현대 로봇 시스템에서는 ESKF가 사실상 표준이다.

3. **Particle Filter**는 다봉 분포와 강한 비선형을 다룰 수 있지만 차원의 저주로 고차원 문제에 부적합하다. RBPF(FastSLAM)로 일부 완화 가능하며, 2D SLAM과 global localization에서 여전히 활용된다.

4. **Filtering에서 Optimization으로의 전환**은 재선형화, 희소성 활용, 루프 클로저 처리 등 여러 이점 때문에 현대 SLAM의 주류가 되었다. 다만 필터 기반(MSCKF, FAST-LIO2)도 특정 조건에서 경쟁력을 유지한다.

5. **Factor Graph**는 확률적 추론을 모듈적으로 구성하고 MAP = NLS로 환원하는 강력한 프레임워크다. Gauss-Newton/LM으로 매니폴드 위에서 풀며, iSAM2의 incremental smoothing이 실시간 처리를 가능하게 한다.

6. **IMU Preintegration**은 고속 IMU 측정을 키프레임 간 factor로 압축하는 핵심 기술이다. On-manifold 유도와 바이어스 1차 보정이 재적분 없이 factor graph에 통합할 수 있게 한다.

7. **Marginalization**은 슬라이딩 윈도우의 정보 보존 메커니즘이다. Schur complement가 핵심 연산이며, FEJ가 일관성 유지의 열쇠다.

이 장의 이론은 이후의 VIO(Ch.6), LIO(Ch.7), 멀티센서 퓨전(Ch.8) 챕터에서 실제 시스템의 설계와 구현을 이해하는 데 필수적인 기반이 된다.

> **2024-2025 연구 방향**: 두 흐름이 특히 주목받는다. **대칭 기반 필터**: Equivariant Filter(EqF)와 Invariant EKF가 Lie 군의 대칭 구조를 활용하여 일관성과 수렴성을 구조적으로 보장한다. **연속 시간 최적화**: Gaussian Process motion prior를 사용한 continuous-time factor graph가 비동기 멀티센서 퓨전의 새 패러다임으로 자리잡고 있다. 한편 [AI-Aided Kalman Filters (Revach et al., 2024)](https://arxiv.org/abs/2410.12289)처럼 RNN/Transformer로 칼만 이득이나 프로세스 모델을 학습하는 접근도 활발하나, 안전성 보장이 과제로 남아 있다.
