# Ch.4 — State Estimation 이론

Ch.2의 관측 모델과 Ch.3의 캘리브레이션 값은 로봇의 상태(위치, 자세, 속도 등)를 추정하는 알고리즘에 들어간다. Ch.6~8의 odometry와 퓨전 시스템은 이 입력을 칼만 필터, factor graph, IMU 사전적분(preintegration)으로 처리한다.

---

## 4.1 Bayesian Filtering Framework

### 4.1.1 상태 추정 문제의 정의

상태 추정(state estimation)은 조건부 확률을 구하는 문제다. 지금까지의 관측 $\mathbf{z}_{1:k}$와 제어 입력 $\mathbf{u}_{1:k}$를 알 때 상태 $\mathbf{x}_k$의 사후 확률(posterior)을 구한다.

$$p(\mathbf{x}_k \mid \mathbf{z}_{1:k}, \mathbf{u}_{1:k})$$

여기서:
- $\mathbf{x}_k \in \mathbb{R}^n$: 시각 $k$의 상태 벡터 (위치, 속도, 자세, 바이어스 등)
- $\mathbf{z}_k \in \mathbb{R}^m$: 시각 $k$의 관측 벡터 (카메라, LiDAR, IMU 등)
- $\mathbf{u}_k \in \mathbb{R}^l$: 시각 $k$의 제어 입력

이 문제를 풀려면 모델이 둘 있어야 한다.

**운동 모델(Motion Model, Process Model)**:
$$\mathbf{x}_k = f(\mathbf{x}_{k-1}, \mathbf{u}_k) + \mathbf{w}_k, \quad \mathbf{w}_k \sim \mathcal{N}(\mathbf{0}, \mathbf{Q}_k)$$

$$p(\mathbf{x}_k \mid \mathbf{x}_{k-1}, \mathbf{u}_k)$$

제어 입력 $\mathbf{u}_k$를 알 때 이전 상태가 현재 상태로 넘어갈 확률이다. 프로세스 노이즈 $\mathbf{w}_k$는 모델이 실제 운동을 다 설명하지 못하는 부분을 나타낸다.

**관측 모델(Observation Model, Measurement Model)**:
$$\mathbf{z}_k = h(\mathbf{x}_k) + \mathbf{v}_k, \quad \mathbf{v}_k \sim \mathcal{N}(\mathbf{0}, \mathbf{R}_k)$$

$$p(\mathbf{z}_k \mid \mathbf{x}_k)$$

현재 상태를 알 때 그 관측값이 나올 우도(likelihood)다. 관측 노이즈 $\mathbf{v}_k$는 센서 측정의 오차를 나타낸다.

필터는 운동 모델과 관측 모델을 번갈아 써서 상태의 확률 분포를 갱신한다. 이 분포를 믿음(belief)이라고도 부른다.

### 4.1.2 Markov 가정과 Recursive Estimation

필터를 실제로 계산하려면 Markov 가정 두 가지를 둔다.

1. **1차 Markov 프로세스**: 직전 상태 $\mathbf{x}_{k-1}$과 제어 $\mathbf{u}_k$를 알면 현재 상태 $\mathbf{x}_k$는 그보다 앞선 상태나 관측과 무관하다.
$$p(\mathbf{x}_k \mid \mathbf{x}_{0:k-1}, \mathbf{u}_{1:k}, \mathbf{z}_{1:k-1}) = p(\mathbf{x}_k \mid \mathbf{x}_{k-1}, \mathbf{u}_k)$$

2. **조건부 관측 독립**: 현재 상태 $\mathbf{x}_k$를 알면 관측 $\mathbf{z}_k$는 다른 모든 변수와 무관하다.
$$p(\mathbf{z}_k \mid \mathbf{x}_{0:k}, \mathbf{u}_{1:k}, \mathbf{z}_{1:k-1}) = p(\mathbf{z}_k \mid \mathbf{x}_k)$$

이 가정을 두면 관측 이력 $\mathbf{z}_{1:k}$를 모두 저장하지 않고 직전 추정치에서 현재 추정치를 바로 계산한다. 이를 재귀적 추정(recursive estimation)이라 한다.

### 4.1.3 Prediction-Update Cycle

Bayesian filter는 예측과 갱신을 번갈아 한다.

**예측 단계(Prediction Step)**: 이전 시각의 사후 확률에 운동 모델을 적용해 현재 시각의 사전 확률(prior)을 구한다.

$$\boxed{p(\mathbf{x}_k \mid \mathbf{z}_{1:k-1}, \mathbf{u}_{1:k}) = \int p(\mathbf{x}_k \mid \mathbf{x}_{k-1}, \mathbf{u}_k) \, p(\mathbf{x}_{k-1} \mid \mathbf{z}_{1:k-1}, \mathbf{u}_{1:k-1}) \, d\mathbf{x}_{k-1}}$$

이 적분이 Chapman-Kolmogorov 방정식이다. 각 항의 뜻은 다음과 같다.
- $p(\mathbf{x}_{k-1} \mid \mathbf{z}_{1:k-1})$: 시각 $k-1$의 사후 확률 (이전 단계의 결과)
- $p(\mathbf{x}_k \mid \mathbf{x}_{k-1}, \mathbf{u}_k)$: 상태 전이 확률 (운동 모델)
- 이전 상태가 가질 수 있는 모든 값에 걸쳐 전이 확률을 가중 평균해 예측 분포를 구한다

**갱신 단계(Update Step)**: 새 관측 $\mathbf{z}_k$가 들어오면 Bayes 정리로 사후 확률을 구한다.

$$\boxed{p(\mathbf{x}_k \mid \mathbf{z}_{1:k}, \mathbf{u}_{1:k}) = \frac{p(\mathbf{z}_k \mid \mathbf{x}_k) \, p(\mathbf{x}_k \mid \mathbf{z}_{1:k-1}, \mathbf{u}_{1:k})}{p(\mathbf{z}_k \mid \mathbf{z}_{1:k-1})}}$$

여기서:
- 분자의 $p(\mathbf{z}_k \mid \mathbf{x}_k)$: likelihood (관측 모델)
- 분자의 $p(\mathbf{x}_k \mid \mathbf{z}_{1:k-1})$: 예측 단계에서 구한 prior
- 분모의 $p(\mathbf{z}_k \mid \mathbf{z}_{1:k-1}) = \int p(\mathbf{z}_k \mid \mathbf{x}_k) p(\mathbf{x}_k \mid \mathbf{z}_{1:k-1}) d\mathbf{x}_k$: 정규화 상수 (evidence)

필터는 관측이 들어올 때마다 관측이 예측과 얼마나 맞는지 따져 믿음을 고친다.

### 4.1.4 왜 Closed-Form이 안 되는가

Chapman-Kolmogorov 적분 $\int p(\mathbf{x}_k \mid \mathbf{x}_{k-1}) p(\mathbf{x}_{k-1} \mid \mathbf{z}_{1:k-1}) d\mathbf{x}_{k-1}$은 대부분의 실제 문제에서 해석적으로 풀리지 않는다. 이유는 다음과 같다.

1. **비선형 운동·관측 모델**: $f(\cdot)$과 $h(\cdot)$가 비선형이면 가우시안 분포를 통과시킨 결과가 더는 가우시안이 아니고, 식 하나로 쓸 수 없는 모양이 된다.

2. **고차원 상태 공간**: 적분을 격자로 계산하면 계산량이 상태 차원에 대해 지수적으로 늘어난다(차원의 저주).

3. **다봉(multimodal) 분포**: 생김새가 같은 복도가 여럿인 건물에서 처음 위치를 찾을 때(global localization)처럼 로봇이 있을 만한 곳이 여러 군데면 사후 분포의 봉우리도 여럿이 된다.

그래서 실제로는 근사(approximation)를 쓰고, 근사하는 방법에 따라 필터가 나뉜다.

| 근사 방법 | 필터 | 분포 표현 | 적용 조건 |
|----------|------|---------|----------|
| 선형-가우시안 가정 | Kalman Filter | $\mathcal{N}(\hat{\mathbf{x}}, \mathbf{P})$ | 선형 시스템 |
| 1차 선형화 + 가우시안 | EKF, ESKF | $\mathcal{N}(\hat{\mathbf{x}}, \mathbf{P})$ | 약한 비선형 |
| Sigma point 변환 | UKF | $\mathcal{N}(\hat{\mathbf{x}}, \mathbf{P})$ | 중간 비선형 |
| 샘플 기반 | Particle Filter | $\{(\mathbf{x}^{(i)}, w^{(i)})\}_{i=1}^N$ | 강한 비선형, 다봉 |
| 반복 선형화 | IEKF | $\mathcal{N}(\hat{\mathbf{x}}, \mathbf{P})$ | 강한 비선형 |

---

## 4.2 Kalman Filter 계열

### 4.2.1 Kalman Filter: 유도와 최적성

시스템이 선형이고, 프로세스·관측 노이즈와 초기 상태 분포가 가우시안이면 Bayesian filter의 해는 정확히(exact) Kalman Filter(KF)가 된다. Rudolf Kalman이 1960년 논문 ["A New Approach to Linear Filtering and Prediction Problems"](https://doi.org/10.1115/1.3662552)에서 발표했다. NASA Ames의 Stanley Schmidt 팀이 이를 Apollo 우주선의 궤도 추정에 쓰면서 실제 항법에 들어갔다.

#### 선형 시스템 모델

상태 전이와 관측이 모두 선형인 시스템을 생각한다.

$$\mathbf{x}_k = \mathbf{F}_k \mathbf{x}_{k-1} + \mathbf{B}_k \mathbf{u}_k + \mathbf{w}_k, \quad \mathbf{w}_k \sim \mathcal{N}(\mathbf{0}, \mathbf{Q}_k)$$
$$\mathbf{z}_k = \mathbf{H}_k \mathbf{x}_k + \mathbf{v}_k, \quad \mathbf{v}_k \sim \mathcal{N}(\mathbf{0}, \mathbf{R}_k)$$

여기서:
- $\mathbf{F}_k \in \mathbb{R}^{n \times n}$: 상태 전이 행렬 (state transition matrix)
- $\mathbf{B}_k \in \mathbb{R}^{n \times l}$: 제어 입력 행렬
- $\mathbf{H}_k \in \mathbb{R}^{m \times n}$: 관측 행렬
- $\mathbf{Q}_k \in \mathbb{R}^{n \times n}$: 프로세스 노이즈 공분산 (양의 준정부호, symmetric)
- $\mathbf{R}_k \in \mathbb{R}^{m \times m}$: 관측 노이즈 공분산 (양의 정부호, symmetric)

가우시안 분포에 선형 변환을 적용하면 결과도 가우시안이다. 그래서 사후 분포는 계속 가우시안이고, 평균과 공분산만 알면 된다.

#### MMSE 유도 — 왜 Kalman Filter가 최적인가

여기서는 Kalman Filter가 왜 최적인지 평균 제곱 오차 최소(Minimum Mean Square Error, MMSE) 기준으로 보인다. 찾는 추정량은 평균 제곱 오차를 가장 작게 만드는 값이다.

$$\hat{\mathbf{x}}_k = \arg\min_{\hat{\mathbf{x}}} \mathbb{E}[\|\mathbf{x}_k - \hat{\mathbf{x}}\|^2 \mid \mathbf{z}_{1:k}]$$

MMSE 추정량은 조건부 기댓값 $\hat{\mathbf{x}}_k = \mathbb{E}[\mathbf{x}_k \mid \mathbf{z}_{1:k}]$이다. 선형-가우시안 시스템에서는 이 기댓값이 Kalman Filter의 상태 갱신식과 같아진다.

**Bayesian 관점의 유도**: 시각 $k-1$의 사후 분포가 가우시안이라고 하자.

$$p(\mathbf{x}_{k-1} \mid \mathbf{z}_{1:k-1}) = \mathcal{N}(\hat{\mathbf{x}}_{k-1|k-1}, \mathbf{P}_{k-1|k-1})$$

**Step 1: Prediction — Chapman-Kolmogorov 적분 계산**

선형 운동 모델은 다음과 같다.
$$\mathbf{x}_k = \mathbf{F}_k \mathbf{x}_{k-1} + \mathbf{B}_k \mathbf{u}_k + \mathbf{w}_k$$

$\mathbf{x}_{k-1}$이 가우시안이고 $\mathbf{w}_k$가 이와 독립인 가우시안이므로 $\mathbf{x}_k$도 가우시안이다. 평균은 다음과 같다.

$$\hat{\mathbf{x}}_{k|k-1} = \mathbb{E}[\mathbf{x}_k \mid \mathbf{z}_{1:k-1}] = \mathbf{F}_k \hat{\mathbf{x}}_{k-1|k-1} + \mathbf{B}_k \mathbf{u}_k$$

예측 오차가 $\tilde{\mathbf{x}}_{k|k-1} = \mathbf{x}_k - \hat{\mathbf{x}}_{k|k-1} = \mathbf{F}_k \tilde{\mathbf{x}}_{k-1|k-1} + \mathbf{w}_k$이므로 공분산은 다음과 같다.

$$\mathbf{P}_{k|k-1} = \mathbb{E}[\tilde{\mathbf{x}}_{k|k-1} \tilde{\mathbf{x}}_{k|k-1}^\top] = \mathbf{F}_k \mathbf{P}_{k-1|k-1} \mathbf{F}_k^\top + \mathbf{Q}_k$$

예측된 분포: $p(\mathbf{x}_k \mid \mathbf{z}_{1:k-1}) = \mathcal{N}(\hat{\mathbf{x}}_{k|k-1}, \mathbf{P}_{k|k-1})$

**Step 2: Update — Bayes 정리 적용**

관측 모델 $\mathbf{z}_k = \mathbf{H}_k \mathbf{x}_k + \mathbf{v}_k$로 예측 상태와 관측의 결합 분포(joint distribution)를 쓴다.

$$\begin{bmatrix} \mathbf{x}_k \\ \mathbf{z}_k \end{bmatrix} \sim \mathcal{N}\left( \begin{bmatrix} \hat{\mathbf{x}}_{k|k-1} \\ \mathbf{H}_k \hat{\mathbf{x}}_{k|k-1} \end{bmatrix}, \begin{bmatrix} \mathbf{P}_{k|k-1} & \mathbf{P}_{k|k-1} \mathbf{H}_k^\top \\ \mathbf{H}_k \mathbf{P}_{k|k-1} & \mathbf{S}_k \end{bmatrix} \right)$$

여기서 $\mathbf{S}_k = \mathbf{H}_k \mathbf{P}_{k|k-1} \mathbf{H}_k^\top + \mathbf{R}_k \in \mathbb{R}^{m \times m}$는 혁신 공분산(innovation covariance)이다.

결합 분포가 가우시안이면 조건부 분포도 가우시안이다. 조건부 평균은 원래 평균에, 관측과 상관된 정도만큼의 보정을 더한 값이다.

$$\hat{\mathbf{x}}_{k|k} = \hat{\mathbf{x}}_{k|k-1} + \mathbf{P}_{k|k-1} \mathbf{H}_k^\top \mathbf{S}_k^{-1} (\mathbf{z}_k - \mathbf{H}_k \hat{\mathbf{x}}_{k|k-1})$$

**칼만 이득(Kalman gain)** $\mathbf{K}_k \in \mathbb{R}^{n \times m}$을 다음과 같이 둔다.

$$\boxed{\mathbf{K}_k = \mathbf{P}_{k|k-1} \mathbf{H}_k^\top \mathbf{S}_k^{-1} = \mathbf{P}_{k|k-1} \mathbf{H}_k^\top (\mathbf{H}_k \mathbf{P}_{k|k-1} \mathbf{H}_k^\top + \mathbf{R}_k)^{-1}}$$

칼만 이득의 뜻은 아래와 같다.
- $\mathbf{R}_k \to \mathbf{0}$ (관측 잡음이 작음): 관측 가능한 부분공간에서 갱신이 측정값을 거의 그대로 따른다. $\mathbf{H}_k$가 정사각·가역인 특수한 경우에는 $\mathbf{K}_k \to \mathbf{H}_k^{-1}$이다
- $\mathbf{P}_{k|k-1} \to \mathbf{0}$ (예측이 매우 정확): $\mathbf{K}_k \to \mathbf{0}$이 되어 관측은 거의 반영되지 않고 예측값이 그대로 남는다
- 선형 가우시안 모델과 지정한 잡음 공분산이 맞다면 이 이득으로 갱신한 값이 조건부 평균이 된다

**혁신(Innovation)**:
$$\tilde{\mathbf{y}}_k = \mathbf{z}_k - \mathbf{H}_k \hat{\mathbf{x}}_{k|k-1} \in \mathbb{R}^m$$

예측한 관측과 실제 관측의 차이다. 혁신이 0이라고 해서 관측되지 않는 상태나 모델 오차까지 없다는 뜻은 아니다.

**상태 갱신**:
$$\boxed{\hat{\mathbf{x}}_{k|k} = \hat{\mathbf{x}}_{k|k-1} + \mathbf{K}_k \tilde{\mathbf{y}}_k}$$

**공분산 갱신**:
$$\boxed{\mathbf{P}_{k|k} = (\mathbf{I} - \mathbf{K}_k \mathbf{H}_k) \mathbf{P}_{k|k-1}}$$

같은 갱신을 Joseph form $\mathbf{P}_{k|k} = (\mathbf{I} - \mathbf{K}_k \mathbf{H}_k) \mathbf{P}_{k|k-1} (\mathbf{I} - \mathbf{K}_k \mathbf{H}_k)^\top + \mathbf{K}_k \mathbf{R}_k \mathbf{K}_k^\top$으로 쓰면 수치적으로 더 안정하다. 이 식은 대칭 양의 준정부호 행렬의 합이라서, $\mathbf{P}_{k|k-1}$과 $\mathbf{R}_k$가 양의 준정부호이면 결과도 대칭 양의 준정부호가 된다. 그래도 부동소수점 반올림이 쌓이므로 구현에서는 공분산을 주기적으로 대칭화하거나 $LDL^\top$ 분해로 푼다.

#### KF 최적성 정리

**정리**: 선형-가우시안 시스템에서 Kalman Filter는 다음 기준 모두에서 최적이다.
1. **MMSE (Minimum Mean Square Error) 추정량**: 평균 제곱 오차를 최소화
2. **MAP (Maximum A Posteriori) 추정량**: 가우시안 분포에서 평균 = 최빈값이므로 MAP과 MMSE가 일치
3. **BLUE (Best Linear Unbiased Estimator)**: 가우시안 가정 없이도, 선형 비편향 추정량 중에서 최소 분산

Kalman의 1960년 원논문은 Wiener 필터의 주파수 도메인 접근법을 상태공간(state-space) 시간 도메인 표현으로 바꾸어 시변(time-varying) 시스템과 다변량 시스템으로 확장했다.

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

### 4.2.2 Extended Kalman Filter (EKF)

실제 로봇 시스템은 거의 항상 비선형이다. 카메라의 3D→2D 투영, IMU의 쿼터니언 기반 회전, LiDAR의 scan matching, GPS의 좌표 변환 모두 비선형 함수다. EKF는 이 함수들을 현재 추정치 근처에서 선형으로 근사하고 Kalman Filter 식을 그대로 쓴다.

#### 1차 테일러 전개 (선형화)

비선형 시스템 모델은 다음과 같다.
$$\mathbf{x}_k = f(\mathbf{x}_{k-1}, \mathbf{u}_k) + \mathbf{w}_k$$
$$\mathbf{z}_k = h(\mathbf{x}_k) + \mathbf{v}_k$$

이 시스템에서는 가우시안을 비선형 함수에 통과시키면 결과가 더 이상 가우시안이 아니다. EKF는 현재 추정치 근방에서 함수를 1차 테일러 전개하여 선형화한다.

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

상태 예측에는 비선형 함수 $f$를 그대로 쓰고, 공분산 전파에만 자코비안 $\mathbf{F}_k$를 쓴다.

**갱신 단계**:
$$\tilde{\mathbf{y}}_k = \mathbf{z}_k - h(\hat{\mathbf{x}}_{k|k-1})$$
$$\mathbf{S}_k = \mathbf{H}_k \mathbf{P}_{k|k-1} \mathbf{H}_k^\top + \mathbf{R}_k$$
$$\mathbf{K}_k = \mathbf{P}_{k|k-1} \mathbf{H}_k^\top \mathbf{S}_k^{-1}$$
$$\hat{\mathbf{x}}_{k|k} = \hat{\mathbf{x}}_{k|k-1} + \mathbf{K}_k \tilde{\mathbf{y}}_k$$
$$\mathbf{P}_{k|k} = (\mathbf{I} - \mathbf{K}_k \mathbf{H}_k) \mathbf{P}_{k|k-1}$$

혁신 $\tilde{\mathbf{y}}_k$도 비선형 함수 $h$로 직접 계산한다.

#### EKF의 한계

1. **선형화 오차**: 비선형성이 심할수록 1차 근사의 오차가 커진다. 그러면 실제 오차가 필터가 계산한 공분산보다 커지는데, 이를 필터가 일관성(consistency)을 잃었다고 한다.

2. **자코비안 계산 부담**: 시간 단계마다 자코비안 $\mathbf{F}_k$와 $\mathbf{H}_k$를 새로 계산해야 한다. 시스템이 복잡하면 자코비안을 손으로 유도하기 번거롭고 틀리기 쉽다.

3. **단봉(unimodal) 가정**: 가우시안은 항상 단봉이므로 다봉 사후 분포를 표현하려면 가우시안 혼합이나 입자 표현이 필요하다.

### 4.2.3 Error-State Kalman Filter (ESKF)

ESKF(Error-State Kalman Filter)는 IMU를 쓰는 상태 추정에서 널리 쓰인다. MSCKF·OpenVINS 같은 필터 기반 VIO와 FAST-LIO 계열이 오차 상태를 추정하는 필터를 쓴다. [VINS-Mono](https://doi.org/10.1109/TRO.2018.2853729)는 필터 대신 sliding-window 비선형 최적화를 쓴다.

#### 왜 EKF 대신 ESKF를 쓰는가

3D 자세(orientation)가 들어간 상태를 EKF로 바로 추정하면 다음 문제가 생긴다.

**문제 1: 회전의 비유클리드 특성**

3D 회전은 벡터 공간 $\mathbb{R}^n$이 아니라 SO(3) 매니폴드에 속한다. 쿼터니언 $\mathbf{q} \in \mathbb{H}$로 표현하면 $\|\mathbf{q}\| = 1$ 제약이 있고, 회전 행렬 $\mathbf{R} \in SO(3)$은 $\mathbf{R}^\top \mathbf{R} = \mathbf{I}$, $\det(\mathbf{R}) = 1$ 제약이 있다.

EKF의 상태 갱신 $\hat{\mathbf{x}} \leftarrow \hat{\mathbf{x}} + \mathbf{K} \tilde{\mathbf{y}}$에서 "+"는 유클리드 덧셈이다. 쿼터니언 네 성분에 증분을 그대로 더하면 노름이 1에서 벗어난다. 갱신 뒤 값만 다시 정규화하면 공분산에는 제약 방향의 불확실성이 그대로 남는다. 쿼터니언을 직접 쓰는 EKF를 만들려면 정규화와 공분산 투영을 함께 설계해야 한다.

**문제 2: 오차 상태를 0 근처에 묶어 두기**

오차 상태(error state) $\delta\mathbf{x} = \mathbf{x} \boxminus \hat{\mathbf{x}}$는 갱신 뒤 명목 상태에 더해지고 0으로 돌아간다. 명목 궤적이 실제 상태에 가까우면 오차가 작으므로 0 근처의 접선 공간에서 선형화해도 오차가 적다. 리셋한다고 명목 궤적이 실제에 가까워지지는 않으므로 초기화, 관측 가능성, 모델 오차를 함께 확인한다.

**문제 3: 명목 상태와 오차 상태의 서로 다른 동역학**

명목 상태는 IMU로 큰 운동을 적분하고, 오차 상태는 그 주변의 작은 오차만 추적한다. 그래서 IMU로 계속 예측하다가 그보다 드물게 오는 카메라나 LiDAR 관측이 도착하면 그때 오차를 고친다.

#### ESKF 구조

ESKF는 상태를 둘로 나누어 관리한다.

1. **명목 상태(Nominal State)** $\hat{\mathbf{x}}$: 노이즈 항을 뺀 비선형 운동 모델로 적분하는 결정론적 상태. 불확실성은 오차 상태가 맡는다.

2. **오차 상태(Error State)** $\delta\mathbf{x}$: 명목 상태와 실제 상태 사이의 작은 차이로, 칼만 필터로 추정한다. 명목 상태가 참값 가까이 있는 동안에는 접선 공간에서 1차 근사가 맞는다. 오차를 명목 상태에 더한 직후에는 reset Jacobian으로 오차 공분산을 새 명목 상태 기준으로 옮긴다.

실제 상태는 두 상태를 합성(composition)해서 얻는다.

$$\mathbf{x}_{\text{true}} = \hat{\mathbf{x}} \boxplus \delta\mathbf{x}$$

여기서 $\boxplus$는 매니폴드 위의 합성 연산이다. 유클리드 성분에서는 보통 덧셈이고, 회전 성분에서는 다음과 같다.

$$\mathbf{R}_{\text{true}} = \hat{\mathbf{R}} \cdot \text{Exp}(\delta\boldsymbol{\theta})$$

쿼터니언으로 쓰면 이렇다.

$$\mathbf{q}_{\text{true}} = \hat{\mathbf{q}} \otimes \delta\mathbf{q} \approx \hat{\mathbf{q}} \otimes \begin{bmatrix} 1 \\ \frac{1}{2}\delta\boldsymbol{\theta} \end{bmatrix}$$

여기서 $\delta\boldsymbol{\theta} \in \mathbb{R}^3$는 회전 오차의 각-축(angle-axis) 표현이다.

#### IMU 기반 ESKF의 상태 벡터

IMU에 카메라나 LiDAR를 더한 퓨전 시스템은 보통 다음 상태 벡터를 쓴다.

**명목 상태** (16차원, 쿼터니언 사용 시):
$$\hat{\mathbf{x}} = \begin{bmatrix} {}^W\hat{\mathbf{p}} \\ {}^W\hat{\mathbf{v}} \\ \hat{\mathbf{q}}_{WB} \\ \hat{\mathbf{b}}_a \\ \hat{\mathbf{b}}_g \end{bmatrix} \in \mathbb{R}^{3} \times \mathbb{R}^{3} \times \mathbb{S}^3 \times \mathbb{R}^{3} \times \mathbb{R}^{3}$$

**오차 상태** (15차원 — 회전의 최소 파라미터화):
$$\delta\mathbf{x} = \begin{bmatrix} \delta\mathbf{p} \\ \delta\mathbf{v} \\ \delta\boldsymbol{\theta} \\ \delta\mathbf{b}_a \\ \delta\mathbf{b}_g \end{bmatrix} \in \mathbb{R}^{15}$$

쿼터니언은 성분이 넷이지만 노름이 1이라는 제약 때문에 자유도는 셋이다. 그래서 ESKF는 회전 오차를 3차원 벡터 $\delta\boldsymbol{\theta}$로 둔다.

#### ESKF 알고리즘

**1단계: 명목 상태 전파 (IMU mechanization)**

IMU 측정 $(\tilde{\boldsymbol{\omega}}_k, \tilde{\mathbf{a}}_k)$에서 바이어스를 빼고, 노이즈 항은 0으로 두고 명목 상태를 적분한다.

$$\hat{\mathbf{q}}_{k+1} = \hat{\mathbf{q}}_k \otimes \mathbf{q}\{(\tilde{\boldsymbol{\omega}}_k - \hat{\mathbf{b}}_{g,k}) \Delta t\}$$
$$\hat{\mathbf{v}}_{k+1} = \hat{\mathbf{v}}_k + (\hat{\mathbf{R}}_k (\tilde{\mathbf{a}}_k - \hat{\mathbf{b}}_{a,k}) + \mathbf{g}) \Delta t$$
$$\hat{\mathbf{p}}_{k+1} = \hat{\mathbf{p}}_k + \hat{\mathbf{v}}_k \Delta t + \frac{1}{2}(\hat{\mathbf{R}}_k (\tilde{\mathbf{a}}_k - \hat{\mathbf{b}}_{a,k}) + \mathbf{g}) \Delta t^2$$
$$\hat{\mathbf{b}}_{a,k+1} = \hat{\mathbf{b}}_{a,k}$$
$$\hat{\mathbf{b}}_{g,k+1} = \hat{\mathbf{b}}_{g,k}$$

여기서 $\hat{\mathbf{R}}_k = \mathbf{R}(\hat{\mathbf{q}}_k) \in SO(3)$이고, $\mathbf{g} = [0, 0, -9.81]^\top \, \text{m/s}^2$는 중력 벡터이다.

**2단계: 오차 상태 전파 (prediction)**

오차 상태의 연속 시간 동역학을 유도한다. 실제 IMU 측정 $\tilde{\boldsymbol{\omega}} = \boldsymbol{\omega} + \mathbf{b}_g + \mathbf{n}_g$, $\tilde{\mathbf{a}} = \mathbf{R}^\top(\mathbf{a}_W - \mathbf{g}) + \mathbf{b}_a + \mathbf{n}_a$를 대입하고 명목 상태를 빼면 다음과 같다.

$$\delta\dot{\boldsymbol{\theta}} = -[\tilde{\boldsymbol{\omega}} - \hat{\mathbf{b}}_g]_\times \delta\boldsymbol{\theta} - \delta\mathbf{b}_g - \mathbf{n}_g$$
$$\delta\dot{\mathbf{v}} = -\hat{\mathbf{R}}[\tilde{\mathbf{a}} - \hat{\mathbf{b}}_a]_\times \delta\boldsymbol{\theta} - \hat{\mathbf{R}} \delta\mathbf{b}_a - \hat{\mathbf{R}} \mathbf{n}_a$$
$$\delta\dot{\mathbf{p}} = \delta\mathbf{v}$$
$$\delta\dot{\mathbf{b}}_a = \mathbf{n}_{ba}$$
$$\delta\dot{\mathbf{b}}_g = \mathbf{n}_{bg}$$

여기서 $[\mathbf{a}]_\times$는 벡터 $\mathbf{a}$의 반대칭 행렬(skew-symmetric matrix)이다.

$$[\mathbf{a}]_\times = \begin{bmatrix} 0 & -a_3 & a_2 \\ a_3 & 0 & -a_1 \\ -a_2 & a_1 & 0 \end{bmatrix} \in \mathbb{R}^{3 \times 3}$$

행렬로 모아 쓰면 다음과 같다.

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

카메라나 LiDAR 관측이 들어오면 오차 상태에 보통의 EKF 갱신을 적용한다. 먼저 관측 모델 $\mathbf{z} = h(\mathbf{x}_{\text{true}})$를 오차 상태에 대해 선형화한다.

$$\mathbf{z} - h(\hat{\mathbf{x}}) \approx \mathbf{H} \delta\mathbf{x} + \mathbf{v}$$

$$\mathbf{H} = \frac{\partial h}{\partial \delta\mathbf{x}}\bigg|_{\hat{\mathbf{x}}} \in \mathbb{R}^{m \times 15}$$

이 자코비안은 연쇄 법칙으로 계산한다.

$$\mathbf{H} = \frac{\partial h}{\partial \mathbf{x}_{\text{true}}} \cdot \frac{\partial \mathbf{x}_{\text{true}}}{\partial \delta\mathbf{x}}\bigg|_{\delta\mathbf{x}=\mathbf{0}}$$

나머지는 보통의 칼만 갱신이다.
$$\mathbf{K} = \mathbf{P}_{k|k-1} \mathbf{H}^\top (\mathbf{H} \mathbf{P}_{k|k-1} \mathbf{H}^\top + \mathbf{R})^{-1}$$
$$\delta\hat{\mathbf{x}} = \mathbf{K}(\mathbf{z} - h(\hat{\mathbf{x}}))$$
$$\mathbf{P}_{k|k} = (\mathbf{I} - \mathbf{K}\mathbf{H})\mathbf{P}_{k|k-1}$$

**4단계: 오차 상태 주입 및 리셋**

오차 상태 추정치 $\delta\hat{\mathbf{x}}$를 명목 상태에 더한다(inject).

$$\hat{\mathbf{p}} \leftarrow \hat{\mathbf{p}} + \delta\hat{\mathbf{p}}$$
$$\hat{\mathbf{v}} \leftarrow \hat{\mathbf{v}} + \delta\hat{\mathbf{v}}$$
$$\hat{\mathbf{q}} \leftarrow \hat{\mathbf{q}} \otimes \mathbf{q}\{\delta\hat{\boldsymbol{\theta}}\}$$
$$\hat{\mathbf{b}}_a \leftarrow \hat{\mathbf{b}}_a + \delta\hat{\mathbf{b}}_a$$
$$\hat{\mathbf{b}}_g \leftarrow \hat{\mathbf{b}}_g + \delta\hat{\mathbf{b}}_g$$

그다음 오차 상태를 $\delta\hat{\mathbf{x}} \leftarrow \mathbf{0}$으로 되돌리고, 공분산도 리셋 자코비안으로 옮긴다.

$$\mathbf{P} \leftarrow \mathbf{G} \mathbf{P} \mathbf{G}^\top$$

여기서 $\mathbf{G} = \frac{\partial (\delta\mathbf{x} \boxminus \delta\hat{\mathbf{x}})}{\partial \delta\mathbf{x}}\big|_{\delta\hat{\mathbf{x}}}$이다. $\delta\hat{\boldsymbol{\theta}}$가 작으면 $\mathbf{G} \approx \mathbf{I}$로 두는 구현이 많다.

### 4.2.4 Unscented Kalman Filter (UKF)

UKF는 비선형 함수를 선형화하지 않고 그대로 둔다. 대신 분포를 몇 개의 대표점(sigma point)으로 바꾸어 함수에 통과시킨다. Julier와 Uhlmann은 비선형 함수보다 확률 분포를 근사하기가 쉽다는 생각에서 이 방법을 냈다.

#### Unscented Transform

$n$차원 가우시안 확률 변수 $\mathbf{x} \sim \mathcal{N}(\hat{\mathbf{x}}, \mathbf{P})$에서 sigma point $2n+1$개와 가중치를 만든다.

$$\boldsymbol{\chi}_0 = \hat{\mathbf{x}}, \quad w_0 = \frac{\lambda}{n + \lambda}$$
$$\boldsymbol{\chi}_i = \hat{\mathbf{x}} + \left(\sqrt{(n+\lambda)\mathbf{P}}\right)_i, \quad w_i = \frac{1}{2(n+\lambda)}, \quad i = 1, \ldots, n$$
$$\boldsymbol{\chi}_{n+i} = \hat{\mathbf{x}} - \left(\sqrt{(n+\lambda)\mathbf{P}}\right)_i, \quad w_{n+i} = \frac{1}{2(n+\lambda)}, \quad i = 1, \ldots, n$$

여기서:
- $\lambda = \alpha^2(n + \kappa) - n$는 스케일링 파라미터 ($\alpha$: sigma point 분산 조절, $\kappa$: 보조 파라미터, 보통 $\kappa = 0$ 또는 $3-n$)
- $\left(\sqrt{(n+\lambda)\mathbf{P}}\right)_i$는 행렬 $(n+\lambda)\mathbf{P}$의 Cholesky 분해의 $i$번째 열

sigma point를 하나씩 비선형 함수에 넣는다.

$$\boldsymbol{\gamma}_i = f(\boldsymbol{\chi}_i)$$

함수를 통과한 sigma point로 평균과 공분산을 다시 계산한다.

$$\hat{\mathbf{y}} = \sum_{i=0}^{2n} w_i^{(m)} \boldsymbol{\gamma}_i$$
$$\mathbf{P}_y = \sum_{i=0}^{2n} w_i^{(c)} (\boldsymbol{\gamma}_i - \hat{\mathbf{y}})(\boldsymbol{\gamma}_i - \hat{\mathbf{y}})^\top$$

$w_i^{(m)}$과 $w_i^{(c)}$는 평균과 공분산에 각각 쓰는 가중치다. $w_0^{(c)} = w_0^{(m)} + (1 - \alpha^2 + \beta)$ ($\beta = 2$는 가우시안에 최적)이고, $i \geq 1$에서 두 가중치가 같다.

#### UKF 알고리즘

**예측 단계**:
1. 현재 상태 $(\hat{\mathbf{x}}_{k-1|k-1}, \mathbf{P}_{k-1|k-1})$에서 sigma point 생성
2. 각 sigma point를 운동 모델에 통과: $\boldsymbol{\chi}_{k|k-1}^{(i)} = f(\boldsymbol{\chi}_{k-1|k-1}^{(i)}, \mathbf{u}_k)$
3. 예측 평균과 공분산 계산: $\hat{\mathbf{x}}_{k|k-1} = \sum w_i^{(m)} \boldsymbol{\chi}_{k|k-1}^{(i)}$
4. $\mathbf{P}_{k|k-1} = \sum w_i^{(c)} (\boldsymbol{\chi}_{k|k-1}^{(i)} - \hat{\mathbf{x}}_{k|k-1})(\cdots)^\top + \mathbf{Q}_k$

**갱신 단계**:
1. 프로세스 잡음 $\mathbf{Q}_k$가 포함된 예측 평균과 공분산에서 sigma point 재생성
2. 관측 모델에 통과: $\boldsymbol{\zeta}_k^{(i)} = h(\boldsymbol{\chi}_{k|k-1}^{(i)})$
3. 예측 관측 평균: $\hat{\mathbf{z}}_k = \sum w_i^{(m)} \boldsymbol{\zeta}_k^{(i)}$
4. 관측 공분산: $\mathbf{P}_{zz} = \sum w_i^{(c)} (\boldsymbol{\zeta}_k^{(i)} - \hat{\mathbf{z}}_k)(\cdots)^\top + \mathbf{R}_k$
5. 교차 공분산: $\mathbf{P}_{xz} = \sum w_i^{(c)} (\boldsymbol{\chi}_{k|k-1}^{(i)} - \hat{\mathbf{x}}_{k|k-1})(\boldsymbol{\zeta}_k^{(i)} - \hat{\mathbf{z}}_k)^\top$
6. 칼만 이득: $\mathbf{K}_k = \mathbf{P}_{xz} \mathbf{P}_{zz}^{-1}$
7. 갱신: $\hat{\mathbf{x}}_{k|k} = \hat{\mathbf{x}}_{k|k-1} + \mathbf{K}_k (\mathbf{z}_k - \hat{\mathbf{z}}_k)$
8. $\mathbf{P}_{k|k} = \mathbf{P}_{k|k-1} - \mathbf{K}_k \mathbf{P}_{zz} \mathbf{K}_k^\top$

#### UKF의 장단점

**장점**:
- 자코비안이 필요 없다. 카메라 투영에 왜곡까지 들어간 관측 모델처럼 자코비안 유도가 번거로운 경우에 편하다.
- 비선형 함수를 통과시킨 sigma point로 평균과 공분산의 곡률 효과를 근사한다. 정확도 차수는 분포, 함수, sigma-point 파라미터에 따라 달라진다.

**단점**:
- sigma point $2n+1$개를 하나하나 비선형 함수에 통과시키므로 상태 차원 $n$이 크면 계산이 많아진다.
- 매니폴드 위의 상태(SO(3) 등)를 다루려면 sigma point의 생성과 통계 계산을 매니폴드 연산으로 대체해야 하므로 구현이 복잡해진다.
- IMU 상태 추정에 ESKF가 더 자주 쓰이는 까닭은 접선 공간 오차로 매니폴드를 다루고, sigma point를 여러 개 전파하지 않아 계산이 적기 때문이다. 어느 필터가 더 정확한지는 모델, 초기 오차, 조정값에 따라 달라진다.

### 4.2.5 Iterated Extended Kalman Filter (IEKF)

IEKF는 EKF의 갱신 단계에서 선형화를 한 번으로 끝내지 않고 여러 번 다시 해서, 비선형 관측 모델을 더 정확히 다룬다.

#### 동기: EKF 갱신의 선형화 오차

EKF는 관측 모델의 자코비안 $\mathbf{H}_k$를 예측 추정치 $\hat{\mathbf{x}}_{k|k-1}$에서 계산한다. 갱신한 추정치 $\hat{\mathbf{x}}_{k|k}$가 예측과 많이 다르면 자코비안을 엉뚱한 자리에서 계산한 셈이 된다. IEKF는 갱신한 추정치에서 다시 선형화하고 갱신하기를 되풀이한다.

#### IEKF 알고리즘

예측 단계는 EKF와 같고, 갱신 단계를 반복한다.

초기화: $\hat{\mathbf{x}}^{(0)} = \hat{\mathbf{x}}_{k|k-1}$

$j = 0, 1, 2, \ldots$ 에 대해 수렴할 때까지:

$$\mathbf{H}^{(j)} = \left.\frac{\partial h}{\partial \mathbf{x}}\right|_{\hat{\mathbf{x}}^{(j)}}$$

$$\mathbf{K}^{(j)} = \mathbf{P}_{k|k-1} \mathbf{H}^{(j)\top} (\mathbf{H}^{(j)} \mathbf{P}_{k|k-1} \mathbf{H}^{(j)\top} + \mathbf{R}_k)^{-1}$$

$$\hat{\mathbf{x}}^{(j+1)} = \hat{\mathbf{x}}_{k|k-1} + \mathbf{K}^{(j)} \left[\mathbf{z}_k - h(\hat{\mathbf{x}}^{(j)}) - \mathbf{H}^{(j)}(\hat{\mathbf{x}}_{k|k-1} - \hat{\mathbf{x}}^{(j)})\right]$$

수렴 후: $\hat{\mathbf{x}}_{k|k} = \hat{\mathbf{x}}^{(j+1)}$, $\mathbf{P}_{k|k} = (\mathbf{I} - \mathbf{K}^{(j)} \mathbf{H}^{(j)}) \mathbf{P}_{k|k-1}$

IEKF의 반복 갱신은 관측 업데이트 한 번을 Gauss-Newton으로 푸는 것과 같다(Bell & Cathey, 1993). §4.5의 factor graph 최적화는 같은 Gauss-Newton을 궤적 전체에 쓴다.

FAST-LIO2는 LiDAR point-to-plane 잔차 여러 개를 상태 갱신에 넣고 반복해서 다시 선형화하는 매니폴드 IEKF를 쓴다. 선형화를 한 번만 할 때보다 잔차의 비선형성을 더 잘 따라가지만, 정확도가 얼마나 오르고 몇 번 만에 수렴하는지는 초기 추정 오차, 주변의 기하 구조, 수렴 판정 기준에 달렸다.

---

## 4.3 Particle Filter

### 4.3.1 Sequential Monte Carlo (SMC) 개요

Particle Filter(PF), 곧 Sequential Monte Carlo(SMC) 방법은 사후 분포를 가중치 붙은 샘플(particle) 묶음으로 나타낸다. 가우시안을 가정하지 않으므로 다봉(multimodal) 분포나 비선형성이 강한 시스템도 다룬다.

사후 분포의 입자 근사:

$$p(\mathbf{x}_k \mid \mathbf{z}_{1:k}) \approx \sum_{i=1}^{N} w_k^{(i)} \delta(\mathbf{x}_k - \mathbf{x}_k^{(i)})$$

여기서:
- $N$: 입자 수
- $\mathbf{x}_k^{(i)}$: $i$번째 입자의 상태
- $w_k^{(i)}$: $i$번째 입자의 가중치 ($\sum_{i=1}^N w_k^{(i)} = 1$)
- $\delta(\cdot)$: 디랙 델타 함수

$N \to \infty$이면 입자 분포는 실제 사후 분포에 수렴한다. 실제로는 입자를 유한 개만 쓰고, 그 수는 문제의 복잡도와 상태 차원을 보고 정한다.

### 4.3.2 Importance Sampling

사후 분포 $p(\mathbf{x}_k \mid \mathbf{z}_{1:k})$에서는 보통 직접 샘플을 뽑지 못한다. 그래서 뽑기 쉬운 제안 분포(proposal distribution) $q(\mathbf{x}_k \mid \mathbf{x}_{k-1}, \mathbf{z}_k)$에서 샘플을 뽑고, 중요도 가중치(importance weight)로 보정한다.

가중치를 재귀적으로 갱신하는 식을 유도한다. Bayes 정리에서 출발한다.

$$p(\mathbf{x}_{0:k} \mid \mathbf{z}_{1:k}) = \frac{p(\mathbf{z}_k \mid \mathbf{x}_k) \, p(\mathbf{x}_k \mid \mathbf{x}_{k-1}) \, p(\mathbf{x}_{0:k-1} \mid \mathbf{z}_{1:k-1})}{p(\mathbf{z}_k \mid \mathbf{z}_{1:k-1})}$$

제안 분포로 나누어 중요도 비율을 구하면 다음과 같다.

$$w_k^{(i)} \propto w_{k-1}^{(i)} \cdot \frac{p(\mathbf{z}_k \mid \mathbf{x}_k^{(i)}) \, p(\mathbf{x}_k^{(i)} \mid \mathbf{x}_{k-1}^{(i)})}{q(\mathbf{x}_k^{(i)} \mid \mathbf{x}_{k-1}^{(i)}, \mathbf{z}_k)}$$

**가장 간단한 제안 분포**: 운동 모델의 전이 확률(transition prior)을 그대로 제안 분포로 쓴다. 곧 $q(\mathbf{x}_k \mid \mathbf{x}_{k-1}, \mathbf{z}_k) = p(\mathbf{x}_k \mid \mathbf{x}_{k-1})$이다. 그러면 가중치에 관측 우도만 곱하면 된다.

$$w_k^{(i)} \propto w_{k-1}^{(i)} \cdot p(\mathbf{z}_k \mid \mathbf{x}_k^{(i)})$$

각 입자의 이전 가중치에 해당 입자 위치의 관측 우도(likelihood)를 곱해 갱신한다. 이전 가중치가 같다면 관측과 일치하는 입자가 더 높은 가중치를 받는다.

가중치 분산을 가장 작게 만드는 제안 분포는 $q^*(\mathbf{x}_k \mid \mathbf{x}_{k-1}^{(i)}, \mathbf{z}_k) = p(\mathbf{x}_k \mid \mathbf{x}_{k-1}^{(i)}, \mathbf{z}_k)$인데, 대개 닫힌 식으로 구하기 어렵다.

### 4.3.3 Resampling

입자를 몇 단계 전파하면 가중치가 입자 몇 개에 몰리고 나머지는 0에 가까워진다. 이를 가중치 퇴화(weight degeneracy)라 한다. 쓸모 있는 입자가 몇 개만 남으므로 근사가 크게 나빠진다.

퇴화 정도는 유효 입자 수(effective sample size)로 잰다.

$$N_{\text{eff}} = \frac{1}{\sum_{i=1}^N (w_k^{(i)})^2}$$

$N_{\text{eff}}$가 기준값(보통 $N/2$)보다 작아지면 리샘플링한다.

**리샘플링**: 가중치가 높은 입자는 여러 개로 복제하고 낮은 입자는 지워서 모든 입자의 가중치를 같게 만든다.

리샘플링 방법은 다음과 같다.

**Multinomial Resampling**: 가중치를 확률로 보고 $N$번 따로 뽑는다. 가장 단순하고, 분산이 크다.

**Systematic Resampling**: 하나의 균등 난수 $U_0 \sim \text{Uniform}(0, 1/N)$을 생성하고, $U_i = U_0 + (i-1)/N$ 위치에서 누적 분포(CDF)를 읽어 입자를 고른다. 구현이 싸고 multinomial보다 분산이 작은 경우가 많다. 가중치 배열에 따라서는 다른 방법보다 분산이 클 수도 있다(Douc, Cappé & Moulines, 2005).

**Stratified Resampling**: Systematic처럼 $[0,1)$을 $N$개 구간으로 나누되, 구간마다 균등 난수를 따로 뽑는다.

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

### 4.3.4 Rao-Blackwellized Particle Filter (RBPF)

RBPF는 상태를 둘로 나누어 한쪽은 입자로, 다른 쪽은 칼만 필터 같은 해석적 방법으로 추정한다. 입자로 다룰 차원이 줄어드니 입자 수가 훨씬 적어도 된다.

상태를 $\mathbf{x} = [\mathbf{x}_1, \mathbf{x}_2]$로 나누고, $\mathbf{x}_1$을 알면 $\mathbf{x}_2$의 조건부 분포를 가우시안 같은 닫힌 식으로 추적할 수 있다고 하자. 그러면 사후 분포를 다음처럼 쪼갠다.

$$p(\mathbf{x}_1, \mathbf{x}_2 \mid \mathbf{z}_{1:k}) = p(\mathbf{x}_2 \mid \mathbf{x}_1, \mathbf{z}_{1:k}) \cdot p(\mathbf{x}_1 \mid \mathbf{z}_{1:k})$$

- $p(\mathbf{x}_1 \mid \mathbf{z}_{1:k})$: 입자 필터로 근사
- $p(\mathbf{x}_2 \mid \mathbf{x}_1, \mathbf{z}_{1:k})$: 입자마다 딸린 칼만 필터로 추적

**Rao-Blackwell 정리**: 이렇게 나누어 얻은 추정량의 분산은 같은 입자 수를 쓰는 순수 입자 필터의 분산보다 크지 않다.

$$\text{Var}[\hat{\mathbf{x}}_{\text{RBPF}}] \leq \text{Var}[\hat{\mathbf{x}}_{\text{PF}}]$$

**FastSLAM**: FastSLAM(Montemerlo et al., 2002)은 RBPF를 SLAM에 쓴 대표 사례다. 로봇 경로(trajectory)는 입자로, 랜드마크 위치는 입자마다 딸린 EKF로 추적한다.

- $\mathbf{x}_1 = \mathbf{x}_{0:k}^{\text{robot}}$ (로봇 경로) → 입자 필터
- $\mathbf{x}_2 = \{\mathbf{m}_1, \ldots, \mathbf{m}_M\}$ (랜드마크) → 각 입자마다 $M$개의 독립 2D EKF

로봇 경로를 알면 랜드마크들의 추정이 서로 독립이 되므로, 큰 EKF 하나 대신 작은 EKF $M$개를 따로 돌린다. 관측 한 건의 랜드마크 대응을 안다고 할 때, 맵을 균형 트리로 관리하는 FastSLAM의 갱신 비용은 입자 $N$개, 랜드마크 $M$개에 대해 $O(N \log M)$이다. 같은 갱신이 EKF-SLAM에서는 $O(M^2)$이다.

### 4.3.5 Particle Filter의 한계와 현재 위치

PF의 큰 약점은 차원의 저주(curse of dimensionality)다. 상태 차원이 커지면 쓸 만한 근사에 필요한 입자 수가 매우 빠르게 늘어난다. VIO·LIO의 상태에는 자세, 속도, 바이어스가 함께 들어간다. 좋은 제안 분포나 Rao–Blackwellization 없이 이 상태 전체를 입자로 표현하면 실시간으로 돌리기 어렵다.

그래서 지금 로봇 시스템에서 PF는 주로 다음 자리에 쓰인다.

- **2D SLAM (RBPF 기반)**: GMapping 같은 2D 점유 격자 SLAM. 로봇 자세(3-DoF)만 입자로 두고, 격자 지도는 입자마다 하나씩 둔다.
- **Global Localization (MCL)**: 지도는 있는데 로봇이 어디서 출발하는지 모를 때 쓴다. 입자가 봉우리가 여럿인 분포를 그대로 표현하기 때문이다. 운용 중 누가 로봇을 들어 옮긴 경우(kidnapped robot problem)에는 입자를 다시 퍼뜨리는 복구 방법도 함께 쓴다.
- **저차원 비선형 추정**: 상태 차원이 낮고 비선형성이 심한 문제.

고차원 상태 추정에는 주로 Kalman 필터 계열(특히 ESKF)이나 factor graph 최적화를 쓴다.

---

## 4.4 Smoothing vs Filtering

### 4.4.1 Filtering과 Smoothing의 차이

**Filtering**: 지금까지의 관측으로 현재 상태를 추정한다.
$$p(\mathbf{x}_k \mid \mathbf{z}_{1:k})$$

**Smoothing**: 그 시각 뒤의 관측까지 모두 써서 과거 상태를 추정한다.
$$p(\mathbf{x}_k \mid \mathbf{z}_{1:T}), \quad k < T$$

확률 모델이 정확하고 손실 함수가 같다면, 뒤의 관측까지 조건에 넣은 smoother의 평균 Bayes risk는 filter보다 크지 않다. 궤적 하나하나의 실제 오차나 근사 최적화 결과에서는 smoother가 더 클 때도 있다. 실시간 추정에는 filter와 함께 incremental smoother, fixed-lag smoother도 쓰고, batch smoother는 후처리에 쓴다.

### 4.4.2 Fixed-Lag Smoother

Fixed-lag smoother는 현재 시각 $k$까지의 관측으로 $L$단계 앞 시각 $k-L$의 상태를 추정한다.

$$p(\mathbf{x}_{k-L} \mid \mathbf{z}_{1:k})$$

$L$단계만큼 늦게(latency) 결과를 내는 대신 그동안 들어온 관측으로 과거 상태를 다시 추정한다. filtering과 full smoothing의 중간이다.

VINS-Mono, ORB-SLAM3 같은 슬라이딩 윈도우 최적화 시스템은 fixed-lag smoothing에 가깝다. 윈도우 안의 키프레임을 함께 다시 선형화할 수 있다. 정확도를 필터와 비교하려면 계산 예산, 모델, 데이터셋을 맞춰야 한다.

### 4.4.3 Full Smoothing (Batch Optimization)

모든 관측으로 궤적 전체의 상태를 한꺼번에 추정한다.

$$p(\mathbf{x}_{0:T} \mid \mathbf{z}_{1:T})$$

MAP(Maximum A Posteriori) 추정으로 쓰면 다음과 같다.

$$\mathbf{x}_{0:T}^* = \arg\max_{\mathbf{x}_{0:T}} p(\mathbf{x}_{0:T} \mid \mathbf{z}_{1:T})$$

가우시안 노이즈를 가정하면 MAP는 비선형 최소자승(Nonlinear Least Squares, NLS) 문제가 된다. 이것이 factor graph 기반 최적화의 출발점이다.

### 4.4.4 왜 현대 SLAM은 Filtering에서 Optimization으로 갔는가

2000년대 초반까지 SLAM 연구는 주로 EKF-SLAM을 썼고, 그 뒤 graph-based optimization(batch smoothing)으로 옮겨 갔다. 이유는 다음과 같다.

**1. 선형화 지점의 문제 (Linearization Point)**

EKF는 시각 $k$의 자코비안을 그 시각의 추정치에서 한 번 계산하고, 나중에 더 나은 추정치를 얻어도 다시 계산하지 않는다. Batch optimization에서는 반복할 때마다 궤적 전체의 자코비안을 최신 추정치에서 다시 계산(relinearize)한다.

[Strasdat et al. (2012) "Visual SLAM: Why Filter?"](https://doi.org/10.1016/j.imavis.2012.02.009)는 monocular와 stereo SLAM 실험에서 같은 계산 시간이라면 프레임 수보다 특징점 수를 늘려야 정확도가 더 오른다는 결과를 냈다. Bundle adjustment의 비용은 특징점 수에 선형이고 filtering은 3제곱이어서, 특징점을 늘리기에는 bundle adjustment가 유리하다.

**2. 일관성(Consistency) 문제**

외부 기준이 없으면 SLAM 결과는 전역 좌표계를 통째로 옮기거나 돌려도 관측이 똑같다. 이런 자유도를 gauge freedom이라 하고, 이 방향은 관측으로 정해지지 않는다(unobservable direction). 그 수는 2D SLAM, 3D SLAM, VIO마다 다르고, VIO에서는 전역 위치 3축과 중력축 둘레의 yaw로 넷이다. 구현에서는 첫 포즈 같은 일부 자유도를 고정해 gauge를 정한다. EKF가 시점마다 다른 추정치에서 선형화하면 이 방향에 없는 정보가 생겨 필터가 일관성을 잃는다(inconsistency). FEJ(First-Estimate Jacobian)와 observability-constrained 방법은 선형화한 뒤에도 이 방향이 관측되지 않은 채로 남도록 한다.

**3. 스케일러빌리티(Scalability)**

EKF-SLAM에서 랜드마크가 $M$개면 공분산 행렬의 원소 수와 갱신 한 번의 계산량이 모두 $O((n + 3M)^2)$이다.

Graph-based SLAM의 정보 행렬(information matrix, Hessian)은 희소(sparse)하다. 포즈나 랜드마크는 관측이나 odometry로 직접 엮인 변수하고만 연결되므로 정보 행렬의 항 대부분이 0이다. 이 구조를 쓰면 dense 행렬로 풀 때보다 계산이 준다. 얼마나 주는지는 그래프의 연결 구조, 소거 순서, loop closure가 만드는 fill-in에 따라 다르다.

**4. Loop closure 처리**

필터로 loop closure를 처리하려면 과거 상태와 그 상관을 계속 들고 있어야 해서 계산량이 크게 는다. Graph-based 시스템에서는 loop closure를 새 factor(constraint)로 더하고 그래프를 다시 최적화한다.

| 관점 | Filtering (EKF) | Optimization (Graph) |
|------|----------------|---------------------|
| 선형화 | 기본 EKF는 갱신 시 한 번 | batch/smoother는 반복 재선형화 가능 |
| 과거 상태 | 기본 필터는 제거하지만 상태 증강 가능 | full smoothing은 유지, fixed-lag는 마지널라이즈 |
| Loop closure | 과거 상태·상관을 유지해야 함 | 새 factor를 추가한 뒤 영향 구간 갱신 |
| 행렬 구조 | 공분산은 대체로 dense | factor/Hessian의 희소성을 활용 가능 |
| 계산량 | 상태와 관측 구조에 의존 | 그래프 구조·소거 순서·fill-in에 의존 |
| 일관성 | 선형화·관측 가능성 처리가 필요 | 재선형화만으로 보장되지 않으며 gauge·마지널라이제이션 처리가 필요 |

필터도 여전히 쓰인다. [MSCKF (Mourikis & Roumeliotis, 2007)](https://ieeexplore.ieee.org/document/4209642)와 OpenVINS는 계산 자원이 적은 플랫폼에 쓰이고, FAST-LIO2는 IEKF에 ikd-tree를 붙였다. 필터와 최적화 가운데 어느 쪽이 더 정확한지는 데이터셋, 센서, 계산 예산을 맞춘 실험으로 가린다.

---

## 4.5 Factor Graph & Optimization

### 4.5.1 Factor Graph 표현

Factor graph는 확률적 그래피컬 모델(probabilistic graphical model)의 한 종류다. 변수(variable) 노드와 인자(factor) 노드를 잇는 이분 그래프(bipartite graph)다.

$$p(\mathbf{X} \mid \mathbf{Z}) \propto \prod_{i} f_i(\mathbf{X}_i)$$

여기서:
- $\mathbf{X} = \{\mathbf{x}_0, \mathbf{x}_1, \ldots, \mathbf{x}_T, \mathbf{l}_1, \ldots, \mathbf{l}_M\}$: 변수 노드 (포즈, 랜드마크, 바이어스 등)
- $f_i(\mathbf{X}_i)$: $i$번째 factor. 변수 일부 $\mathbf{X}_i$를 받아, 그 값이 관측이나 사전 정보와 얼마나 맞는지를 0 이상의 수로 돌려주는 함수(potential)
- $\mathbf{Z}$: 모든 관측

factor 하나는 관측 하나나 사전 정보 하나에 대응한다.

| Factor 유형 | 연결하는 변수 | 의미 |
|------------|-------------|------|
| Prior factor | $\mathbf{x}_0$ | 초기 상태 사전 분포 |
| Odometry factor | $\mathbf{x}_{k-1}, \mathbf{x}_k$ | 연속 포즈 간 상대 운동 |
| IMU preintegration factor | $\mathbf{x}_{i}, \mathbf{x}_{j}, \mathbf{v}_i, \mathbf{v}_j, \mathbf{b}_i$ | 키프레임 간 IMU 적분 |
| Vision factor | $\mathbf{x}_k, \mathbf{l}_m$ | 카메라에서 랜드마크 관측 |
| LiDAR factor | $\mathbf{x}_k$ | 포인트-투-플레인/에지 정합 |
| GPS factor | $\mathbf{x}_k$ | 절대 위치 관측 |
| Loop closure factor | $\mathbf{x}_i, \mathbf{x}_j$ | loop closure 상대 포즈 |

새 센서를 붙이려면 그 센서의 factor를 정의해 그래프에 더하면 되고, 기존 factor는 고칠 필요가 없다. 이를 factor graph의 모듈성(modularity)이라 한다.

### 4.5.2 MAP Inference = Nonlinear Least Squares

노이즈가 가우시안이면 factor는 다음 꼴이 된다.

$$f_i(\mathbf{X}_i) \propto \exp\left(-\frac{1}{2} \|\mathbf{r}_i(\mathbf{X}_i)\|^2_{\boldsymbol{\Sigma}_i}\right)$$

여기서 $\mathbf{r}_i(\mathbf{X}_i)$는 잔차(residual)이고, $\|\mathbf{r}\|^2_{\boldsymbol{\Sigma}} = \mathbf{r}^\top \boldsymbol{\Sigma}^{-1} \mathbf{r}$은 Mahalanobis 거리의 제곱이다.

관측 factor라면 잔차는 다음과 같다.
$$\mathbf{r}_i = \mathbf{z}_i - h_i(\mathbf{X}_i), \quad \boldsymbol{\Sigma}_i = \mathbf{R}_i \text{ (관측 노이즈 공분산)}$$

전체 사후 분포의 MAP 추정은 다음과 같다.

$$\mathbf{X}^* = \arg\max_\mathbf{X} p(\mathbf{X} \mid \mathbf{Z}) = \arg\max_\mathbf{X} \prod_i f_i(\mathbf{X}_i)$$

로그를 씌우고 부호를 바꾸면 다음과 같다.

$$\boxed{\mathbf{X}^* = \arg\min_\mathbf{X} \sum_i \|\mathbf{r}_i(\mathbf{X}_i)\|^2_{\boldsymbol{\Sigma}_i}}$$

확률 추론 문제가 비선형 최소자승(Nonlinear Least Squares, NLS) 문제로 바뀌었다. 이 과정은 [Dellaert & Kaess (2017)의 "Factor Graphs for Robot Perception"](https://doi.org/10.1561/2300000043)에 자세히 나온다.

### 4.5.3 Gauss-Newton Method

NLS 문제는 Gauss-Newton(GN) 방법으로 푼다. 먼저 잔차를 현재 추정치 $\mathbf{X}^{(0)}$ 근처에서 1차 테일러 전개한다.

$$\mathbf{r}_i(\mathbf{X}^{(0)} \boxplus \Delta\mathbf{X}) \approx \mathbf{r}_i(\mathbf{X}^{(0)}) + \mathbf{J}_i \Delta\mathbf{X}$$

여기서 $\mathbf{J}_i = \frac{\partial \mathbf{r}_i}{\partial \mathbf{X}}\big|_{\mathbf{X}^{(0)}}$는 잔차의 자코비안이고, $\boxplus$는 매니폴드 위의 증분 연산이다.

이를 비용 함수에 대입하면 다음과 같다.

$$\sum_i \|\mathbf{r}_i + \mathbf{J}_i \Delta\mathbf{X}\|^2_{\boldsymbol{\Sigma}_i}$$

$\mathbf{r}_i' = \boldsymbol{\Sigma}_i^{-1/2} \mathbf{r}_i$, $\mathbf{J}_i' = \boldsymbol{\Sigma}_i^{-1/2} \mathbf{J}_i$로 공분산을 단위 행렬로 맞추면(whitening) 보통의 최소자승 문제가 된다.

$$\sum_i \|\mathbf{r}_i' + \mathbf{J}_i' \Delta\mathbf{X}\|^2$$

$\Delta\mathbf{X}$로 미분해 0으로 두면 정규 방정식(normal equation)이 나온다.

$$\underbrace{\left(\sum_i \mathbf{J}_i'^\top \mathbf{J}_i'\right)}_{\mathbf{H}} \Delta\mathbf{X} = -\underbrace{\sum_i \mathbf{J}_i'^\top \mathbf{r}_i'}_{\mathbf{b}}$$

$$\boxed{\mathbf{H} \Delta\mathbf{X} = -\mathbf{b}}$$

여기서 $\mathbf{H} = \mathbf{J}^\top \boldsymbol{\Sigma}^{-1} \mathbf{J} \in \mathbb{R}^{N \times N}$는 근사 Hessian (정보 행렬)이고, $\mathbf{b} = \mathbf{J}^\top \boldsymbol{\Sigma}^{-1} \mathbf{r}$은 gradient이다.

factor $i$의 자코비안 $\mathbf{J}_i$에서 0이 아닌 열은 그 factor에 연결된 변수의 열뿐이다. 그래서 $\mathbf{H}$에서 0이 아닌 블록은 같은 factor에 함께 걸린 변수 쌍에만 생긴다. SLAM의 factor는 대부분 변수 몇 개만 잇기 때문에 $\mathbf{H}$는 희소하다.

Gauss-Newton은 다음을 반복한다.

$$\mathbf{X}^{(k+1)} = \mathbf{X}^{(k)} \boxplus \Delta\mathbf{X}^{(k)}$$

반복마다 정규 방정식을 푼다. 보통 희소 Cholesky 분해 $\mathbf{H} = \mathbf{L}\mathbf{L}^\top$ 뒤 전방·후방 대입으로 푼다. 변수 순서(variable ordering)에 따라 $\mathbf{L}$에 새로 생기는 0 아닌 항(fill-in)의 수가 달라지므로, COLAMD 같은 근사 최소 차수 순서(approximate minimum degree ordering)로 변수를 정렬한다.

### 4.5.4 Levenberg-Marquardt Method

Gauss-Newton은 잔차의 2차 미분 항을 버리고 $\mathbf{J}^\top\mathbf{J}$로 Hessian을 근사하므로, 초기값이 나쁘거나 비선형성이 심하면 발산하기도 한다. Levenberg-Marquardt(LM) 방법은 감쇠 항을 더해 GN과 경사 하강법 사이를 오간다.

$$(\mathbf{H} + \lambda \mathbf{I}) \Delta\mathbf{X} = -\mathbf{b}$$

- $\lambda$가 작으면 → GN에 가까움 (해 근처의 조건이 좋으면 빠르게 수렴)
- $\lambda$가 크면 → gradient descent에 가까움 (작은 스텝, 안전)

$\lambda$ 조절: 갱신으로 비용이 줄면 $\lambda$를 줄여 GN 쪽으로 가고, 비용이 늘면 그 갱신을 버리고 $\lambda$를 키워 보폭을 줄인다.

### 4.5.5 매니폴드 위의 최적화 (Optimization on Manifolds)

3D 포즈 $\mathbf{T} \in SE(3)$는 매니폴드의 원소이므로, 최적화에서 갱신할 때 덧셈 대신 retraction(또는 exponential map)을 쓴다.

현재 추정치 $\mathbf{T}^{(k)}$ 의 접선 공간(tangent space)에서 증분 $\boldsymbol{\xi} \in \mathbb{R}^6$을 정의하고 다음과 같이 갱신한다.

$$\mathbf{T}^{(k+1)} = \mathbf{T}^{(k)} \cdot \text{Exp}(\boldsymbol{\xi})$$

증분을 왼쪽에 곱하면 다음과 같다.

$$\mathbf{T}^{(k+1)} = \text{Exp}(\boldsymbol{\xi}) \cdot \mathbf{T}^{(k)}$$

어느 쪽에 곱할지는 라이브러리마다 정한 규약을 따른다.

여기서 $\text{Exp}: \mathbb{R}^6 \to SE(3)$는 Lie group의 exponential map이다. $\boldsymbol{\xi} = [\boldsymbol{\rho}^\top, \boldsymbol{\phi}^\top]^\top$에서 $\boldsymbol{\rho} \in \mathbb{R}^3$는 이동, $\boldsymbol{\phi} \in \mathbb{R}^3$는 회전(각-축 표현)이다.

SO(3)의 exponential map은 Rodrigues 공식으로 계산한다.

$$\text{Exp}(\boldsymbol{\phi}) = \mathbf{I} + \frac{\sin\theta}{\theta}[\boldsymbol{\phi}]_\times + \frac{1 - \cos\theta}{\theta^2}[\boldsymbol{\phi}]_\times^2 \in \mathbb{R}^{3 \times 3}$$

여기서 $\theta = \|\boldsymbol{\phi}\|$이다.

반대로 $\text{Log}: SE(3) \to \mathbb{R}^6$은 logarithmic map이다.

Gauss-Newton이나 LM의 정규 방정식은 접선 공간의 증분 $\boldsymbol{\xi}$로 풀고, 그 해를 exponential map으로 상태에 반영한다. 자코비안도 접선 공간의 증분에 대해 계산한다.

$$\mathbf{J}_i = \frac{\partial \mathbf{r}_i}{\partial \boldsymbol{\xi}}\bigg|_{\boldsymbol{\xi}=\mathbf{0}}$$

### 4.5.6 iSAM2: Incremental Smoothing

키프레임이 들어올 때마다 batch 최적화를 처음부터 다시 하면 궤적이 길어질수록 시간이 감당할 수 없게 늘어난다. iSAM2([Kaess et al., 2012](https://doi.org/10.1177/0278364911430419))는 Bayes tree 자료구조로 바뀐 부분만 다시 계산한다.

iSAM2의 구성 요소는 다음과 같다.

1. **Bayes Tree**: factor graph를 변수 소거(elimination)한 결과를 방향 있는 트리로 나타낸다. 노드마다 clique(변수 묶음)의 조건부 확률 밀도를 저장한다.

2. **Incremental Update**: 새 factor가 들어오면 그 factor가 걸린 clique에서 루트까지의 경로만 다시 계산하고 나머지 트리는 그대로 둔다.

3. **Fluid Relinearization**: 선형화 지점이 현재 추정치에서 기준 이상 멀어진 변수만 골라 다시 선형화한다. 그래서 주기적으로 batch 처리를 돌릴 필요가 없다.

4. **변수 순서 재정렬(Variable Reordering)**: 새 변수나 factor가 들어오면 영향받는 부분의 순서만 다시 정한다.

GTSAM 라이브러리에 iSAM2가 들어 있고, [LIO-SAM](https://arxiv.org/abs/2007.00258)이 이를 백엔드로 쓴다. VINS-Mono의 백엔드는 Ceres로 푸는 sliding-window 최적화다.

> Continuous-Time Factor Graph: 키프레임 시각에만 변수를 두는 대신 궤적을 연속 시간(continuous-time) 함수로 두는 factor graph도 있다. [Burnett et al. (2025)](https://arxiv.org/abs/2402.06174)는 Gaussian Process motion prior를 쓴 연속 시간 추정으로 radar-inertial과 LiDAR-inertial odometry를 구현했다.

### 4.5.7 GTSAM / Ceres / g2o 비교

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

---

## 4.6 IMU Preintegration

### 4.6.1 왜 Preintegration이 필요한가

예를 들어 IMU가 200~1000Hz로 가속도와 각속도를 재고 카메라나 LiDAR 키프레임이 10~30Hz로 들어온다면, 두 키프레임 $i, j$ 사이에 IMU 측정이 대략 7~100개 있다.

**Naive 접근: 직접 적분**

두 키프레임 사이의 IMU 측정을 적분해 상대 포즈를 구하면 다음과 같다.

$$\mathbf{R}_j = \mathbf{R}_i \prod_{k=i}^{j-1} \text{Exp}((\tilde{\boldsymbol{\omega}}_k - \mathbf{b}_g) \Delta t)$$
$$\mathbf{v}_j = \mathbf{v}_i + \mathbf{g} \Delta t_{ij} + \sum_{k=i}^{j-1} \mathbf{R}_k (\tilde{\mathbf{a}}_k - \mathbf{b}_a) \Delta t$$
$$\mathbf{p}_j = \mathbf{p}_i + \sum_{k=i}^{j-1}\left[\mathbf{v}_k \Delta t + \frac{1}{2}\left(\mathbf{R}_k(\tilde{\mathbf{a}}_k - \mathbf{b}_a) + \mathbf{g}\right)\Delta t^2\right]$$

이 적분 결과는 키프레임 $i$의 상태 $(\mathbf{R}_i, \mathbf{v}_i, \mathbf{p}_i)$와 바이어스 $(\mathbf{b}_g, \mathbf{b}_a)$에 따라 달라진다. 최적화 반복에서 이 상태나 바이어스 추정치가 바뀔 때마다 중간 상태를 모두 다시 적분해야 하고, 그러면 반복마다 exponential map을 수백 번 계산하게 된다.

**Preintegration의 해법**: 키프레임 $i$의 body frame에서 본 상대 변화량을 정의하면 이 값은 키프레임 $i$의 전역 포즈와 무관해진다. 바이어스가 바뀌면 다시 적분하지 않고 1차 자코비안으로 보정한다.

### 4.6.2 On-Manifold Preintegration 유도

[Forster et al. (2017)](https://doi.org/10.1109/TRO.2016.2597321)의 on-manifold preintegration을 단계별로 유도한다. 논문 원고는 [arXiv:1512.02363](https://arxiv.org/abs/1512.02363)에서 볼 수 있다.

#### Step 1: 상대 변화량 정의

전역 프레임으로 쓴 적분식을 키프레임 $i$의 body frame 기준으로 다시 정리한다. 전역 변수 $\mathbf{R}_i, \mathbf{v}_i, \mathbf{p}_i$를 좌변으로 옮기면 다음과 같다.

$$\Delta\mathbf{R}_{ij} \triangleq \mathbf{R}_i^\top \mathbf{R}_j = \prod_{k=i}^{j-1} \text{Exp}((\tilde{\boldsymbol{\omega}}_k - \mathbf{b}_g^i)\Delta t) \in SO(3)$$

$$\Delta\mathbf{v}_{ij} \triangleq \mathbf{R}_i^\top (\mathbf{v}_j - \mathbf{v}_i - \mathbf{g}\Delta t_{ij}) = \sum_{k=i}^{j-1} \Delta\mathbf{R}_{ik}(\tilde{\mathbf{a}}_k - \mathbf{b}_a^i)\Delta t \in \mathbb{R}^3$$

$$\Delta\mathbf{p}_{ij} \triangleq \mathbf{R}_i^\top (\mathbf{p}_j - \mathbf{p}_i - \mathbf{v}_i \Delta t_{ij} - \frac{1}{2}\mathbf{g}\Delta t_{ij}^2) = \sum_{k=i}^{j-1}\left[\Delta\mathbf{v}_{ik}\Delta t + \frac{1}{2}\Delta\mathbf{R}_{ik}(\tilde{\mathbf{a}}_k - \mathbf{b}_a^i)\Delta t^2\right] \in \mathbb{R}^3$$

**우변에는 IMU 측정값과 바이어스 추정치만 들어 있다.** 키프레임 $i$의 상태 $(\mathbf{R}_i, \mathbf{v}_i, \mathbf{p}_i)$가 최적화로 바뀌어도 우변은 다시 계산하지 않는다.

#### Step 2: 재귀적 계산 (On-Manifold)

Preintegrated 측정은 IMU 측정이 들어올 때마다 재귀적으로 쌓는다.

$$\Delta\mathbf{R}_{i,k+1} = \Delta\mathbf{R}_{ik} \cdot \text{Exp}((\tilde{\boldsymbol{\omega}}_k - \mathbf{b}_g)\Delta t) \in SO(3)$$
$$\Delta\mathbf{v}_{i,k+1} = \Delta\mathbf{v}_{ik} + \Delta\mathbf{R}_{ik}(\tilde{\mathbf{a}}_k - \mathbf{b}_a)\Delta t \in \mathbb{R}^3$$
$$\Delta\mathbf{p}_{i,k+1} = \Delta\mathbf{p}_{ik} + \Delta\mathbf{v}_{ik}\Delta t + \frac{1}{2}\Delta\mathbf{R}_{ik}(\tilde{\mathbf{a}}_k - \mathbf{b}_a)\Delta t^2 \in \mathbb{R}^3$$

초기값: $\Delta\mathbf{R}_{ii} = \mathbf{I}_{3\times 3}$, $\Delta\mathbf{v}_{ii} = \mathbf{0}$, $\Delta\mathbf{p}_{ii} = \mathbf{0}$.

"On-manifold"는 회전 $\Delta\mathbf{R}_{ij}$를 $SO(3)$ 안에서 곱해 쌓는다는 뜻이다. 구현에는 회전 행렬이나 단위 쿼터니언을 쓴다.

#### Step 3: 바이어스 변화의 1차 보정

최적화 과정에서 바이어스 추정치가 $\mathbf{b} \to \mathbf{b} + \delta\mathbf{b}$로 바뀌면 preintegrated 측정도 바뀐다. 처음부터 다시 적분하는 대신 1차 테일러 전개로 보정한다.

**회전 보정**:
$$\Delta\hat{\mathbf{R}}_{ij}(\mathbf{b}_g + \delta\mathbf{b}_g) \approx \Delta\hat{\mathbf{R}}_{ij}(\mathbf{b}_g) \cdot \text{Exp}\left(\frac{\partial \Delta\bar{\mathbf{R}}_{ij}}{\partial \mathbf{b}_g} \delta\mathbf{b}_g\right)$$

**속도 보정**:
$$\Delta\hat{\mathbf{v}}_{ij}(\mathbf{b} + \delta\mathbf{b}) \approx \Delta\hat{\mathbf{v}}_{ij}(\mathbf{b}) + \frac{\partial \Delta\bar{\mathbf{v}}_{ij}}{\partial \mathbf{b}_g} \delta\mathbf{b}_g + \frac{\partial \Delta\bar{\mathbf{v}}_{ij}}{\partial \mathbf{b}_a} \delta\mathbf{b}_a$$

**위치 보정**:
$$\Delta\hat{\mathbf{p}}_{ij}(\mathbf{b} + \delta\mathbf{b}) \approx \Delta\hat{\mathbf{p}}_{ij}(\mathbf{b}) + \frac{\partial \Delta\bar{\mathbf{p}}_{ij}}{\partial \mathbf{b}_g} \delta\mathbf{b}_g + \frac{\partial \Delta\bar{\mathbf{p}}_{ij}}{\partial \mathbf{b}_a} \delta\mathbf{b}_a$$

$\frac{\partial \Delta\bar{\mathbf{R}}_{ij}}{\partial \mathbf{b}_g}$ 같은 자코비안도 preintegration 중에 재귀적으로 쌓는다. 회전의 바이어스 자코비안은 다음과 같다.

$$\frac{\partial \Delta\bar{\mathbf{R}}_{i,k+1}}{\partial \mathbf{b}_g} = \Delta\bar{\mathbf{R}}_{k,k+1}^\top \frac{\partial \Delta\bar{\mathbf{R}}_{ik}}{\partial \mathbf{b}_g} - \text{Jr}((\tilde{\boldsymbol{\omega}}_k - \mathbf{b}_g)\Delta t) \Delta t$$

여기서 $\text{Jr}(\boldsymbol{\phi})$는 SO(3)의 right Jacobian이다.

$$\text{Jr}(\boldsymbol{\phi}) = \mathbf{I} - \frac{1 - \cos\theta}{\theta^2}[\boldsymbol{\phi}]_\times + \frac{\theta - \sin\theta}{\theta^3}[\boldsymbol{\phi}]_\times^2, \quad \theta = \|\boldsymbol{\phi}\|$$

$\|\delta\mathbf{b}\|$가 작으면 1차 보정으로 충분하다. 바이어스 변화가 구현에서 정한 기준을 넘으면 저장해 둔 IMU 측정으로 preintegration을 다시 계산한다.

#### Step 4: 공분산 전파

IMU 측정 노이즈 $\mathbf{n}_g, \mathbf{n}_a \sim \mathcal{N}(\mathbf{0}, \boldsymbol{\Sigma})$는 preintegration 값에도 퍼진다. 그 공분산도 재귀적으로 계산한다.

$$\boldsymbol{\Sigma}_{k+1} = \mathbf{A}_k \boldsymbol{\Sigma}_k \mathbf{A}_k^\top + \mathbf{B}_k \boldsymbol{\Sigma}_\eta \mathbf{B}_k^\top$$

여기서 $\mathbf{A}_k, \mathbf{B}_k$는 한 단계 전파의 자코비안이고, $\boldsymbol{\Sigma}_\eta$는 IMU 노이즈 공분산이다. Factor graph에서는 $\boldsymbol{\Sigma}_{ij}$의 역행렬을 IMU factor의 정보 행렬로 쓴다.

#### Step 5: Factor Graph의 IMU Factor

preintegrated 측정은 두 키프레임 $i, j$를 잇는 factor가 된다. 잔차는 다음과 같다.

$$\mathbf{r}_{\Delta\mathbf{R}_{ij}} = \text{Log}\left(\Delta\hat{\mathbf{R}}_{ij}^\top \mathbf{R}_i^\top \mathbf{R}_j\right) \in \mathbb{R}^3$$

$$\mathbf{r}_{\Delta\mathbf{v}_{ij}} = \mathbf{R}_i^\top(\mathbf{v}_j - \mathbf{v}_i - \mathbf{g}\Delta t_{ij}) - \Delta\hat{\mathbf{v}}_{ij} \in \mathbb{R}^3$$

$$\mathbf{r}_{\Delta\mathbf{p}_{ij}} = \mathbf{R}_i^\top(\mathbf{p}_j - \mathbf{p}_i - \mathbf{v}_i\Delta t_{ij} - \frac{1}{2}\mathbf{g}\Delta t_{ij}^2) - \Delta\hat{\mathbf{p}}_{ij} \in \mathbb{R}^3$$

이 9차원 잔차의 Mahalanobis 거리 제곱 $\|\mathbf{r}\|^2_{\boldsymbol{\Sigma}_{ij}}$을 비용 함수에 더한다.

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

---

## 4.7 Marginalization & Sliding Window

### 4.7.1 왜 Marginalization이 필요한가

Factor graph 기반 SLAM이나 VIO에서는 키프레임이 들어올 때마다 변수와 factor가 는다. 키프레임을 모두 남겨 두면 최적화 계산량이 끝없이 커진다.

슬라이딩 윈도우(sliding window) 방식은 최근 $N$개의 키프레임만 유지하고 오래된 키프레임을 제거한다. 그냥 지우면 그 키프레임에 걸린 관측 정보가 모두 사라져 정확도가 떨어진다.

Marginalization은 오래된 변수를 그래프에서 빼면서, 그 변수에 걸려 있던 정보를 남은 변수들의 사전 분포(prior)로 옮긴다.

### 4.7.2 Schur Complement

Marginalization에는 Schur complement를 쓴다. 정보 행렬 $\mathbf{H}$를 뺄 변수 $\mathbf{x}_m$과 남길 변수 $\mathbf{x}_r$의 블록으로 나누고, 정규 방정식을 $\mathbf{H}\Delta\mathbf{x} = \mathbf{b}$로 쓴다. 여기서는 §4.5.3의 $-\mathbf{b}$를 새로 $\mathbf{b}$라 부른다($\mathbf{b} \triangleq -\mathbf{J}^\top \boldsymbol{\Sigma}^{-1} \mathbf{r}$).

$$\mathbf{H} \Delta\mathbf{x} = \mathbf{b}$$

$$\begin{bmatrix} \mathbf{H}_{mm} & \mathbf{H}_{mr} \\ \mathbf{H}_{rm} & \mathbf{H}_{rr} \end{bmatrix} \begin{bmatrix} \Delta\mathbf{x}_m \\ \Delta\mathbf{x}_r \end{bmatrix} = \begin{bmatrix} \mathbf{b}_m \\ \mathbf{b}_r \end{bmatrix}$$

여기서:
- $\mathbf{x}_m$: 마지널라이즈(제거)할 변수
- $\mathbf{x}_r$: 유지(retain)할 변수
- $\mathbf{H}_{mm} \in \mathbb{R}^{n_m \times n_m}$: 제거할 변수들끼리의 정보 블록
- $\mathbf{H}_{mr} \in \mathbb{R}^{n_m \times n_r}$: 교차 정보 블록
- $\mathbf{H}_{rr} \in \mathbb{R}^{n_r \times n_r}$: 유지할 변수들끼리의 정보 블록

위 블록 행렬에서 $\Delta\mathbf{x}_m$을 소거하면 다음과 같다.

$$\underbrace{(\mathbf{H}_{rr} - \mathbf{H}_{rm} \mathbf{H}_{mm}^{-1} \mathbf{H}_{mr})}_{\mathbf{H}^*} \Delta\mathbf{x}_r = \underbrace{\mathbf{b}_r - \mathbf{H}_{rm} \mathbf{H}_{mm}^{-1} \mathbf{b}_m}_{\mathbf{b}^*}$$

$\mathbf{H}^* = \mathbf{H}_{rr} - \mathbf{H}_{rm} \mathbf{H}_{mm}^{-1} \mathbf{H}_{mr}$을 Schur complement라 한다. 이 행렬이 남은 변수에 걸리는 prior factor의 정보 행렬이 된다.

**직관적 의미**: $\mathbf{x}_m$을 거쳐서만 이어져 있던 $\mathbf{x}_r$의 변수들이 직접 연결된다(fill-in). 그래서 $\mathbf{H}^*$는 $\mathbf{H}_{rr}$보다 0 아닌 항이 많아질 수 있다.

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

### 4.7.3 First-Estimate Jacobian (FEJ)

Marginalization으로 만든 prior는 그 순간의 선형화 지점에서 계산한 값으로 고정된다. 그 뒤 최적화에서 남은 변수의 추정치가 바뀌면, 같은 변수가 prior 안에서는 옛 지점에서, 다른 factor에서는 새 지점에서 선형화된다.

그러면 실제로는 관측되지 않는 방향(§4.4.4의 gauge 방향)에 정보가 있는 것처럼 계산되어 추정기가 일관성을 잃는다.

**FEJ**: prior에 걸린 변수의 자코비안은 늘 그 변수의 첫 추정치(first estimate)에서 계산하고 뒤에도 그 값을 쓴다.

$$\mathbf{J}_{\text{FEJ}} = \left.\frac{\partial \mathbf{r}}{\partial \mathbf{x}}\right|_{\mathbf{x}^{(0)}}$$

여기서 $\mathbf{x}^{(0)}$는 해당 변수가 처음 추정된 시점의 값이다.

FEJ를 쓰면 prior와 다른 factor가 같은 지점에서 선형화되므로 관측되지 않는 방향에 없는 정보가 쌓이지 않는다. MSCKF 계열과 OpenVINS가 이 방법을 쓴다([Li & Mourikis, 2013](https://doi.org/10.1177/0278364913481251)). 첫 추정치가 많이 틀렸으면 그 자리에서 계산한 자코비안도 부정확하다. 변수마다 첫 추정치를 따로 저장해야 해서 구현도 번거롭다.

### 4.7.4 Sliding Window 구현의 실전 이슈

#### 이슈 1: 뺄 키프레임 고르기

VINS-Mono는 뒤에서 두 번째 프레임(second latest frame)이 키프레임인지에 따라 뺄 프레임을 다르게 고른다.
- **그 프레임이 키프레임이면**: 그 프레임은 윈도우에 남기고, 가장 오래된 프레임을 마지널라이즈한다. 윈도우에는 공간적으로 떨어진 키프레임들이 남는다.
- **그 프레임이 키프레임이 아니면**: 그 프레임을 빼되, 시각 관측만 버리고 IMU preintegration은 다음 프레임으로 이어 붙인다.

#### 이슈 2: Fill-in과 행렬 조밀화

마지널라이즈를 거듭하면 prior factor가 점점 많은 변수를 잇게 되어 희소하던 정보 행렬이 조밀해진다. 그만큼 계산이 느려진다.

다음 방법으로 막는다.
- Prior factor에 연결되는 변수 수를 제한한다
- 마지널라이즈할 변수의 순서를 따져 고른다
- 정보를 일부 잃더라도 일부 factor는 마지널라이즈하지 않고 그냥 지운다

#### 이슈 3: 바이어스와 마지널라이제이션

IMU 바이어스는 키프레임 전체에 걸쳐 천천히 변한다. 키프레임을 뺄 때 그 바이어스도 함께 마지널라이즈하면 그때의 선형화 지점이 prior에 고정되어, 뒤에 바이어스가 바뀌어도 prior가 따라가지 못한다.

VINS-Mono도 가장 오래된 키프레임을 제거할 때 그 프레임의 속도·바이어스를 함께 마지널라이즈한다. 다만 남은 키프레임의 바이어스는 윈도우 안에서 계속 추정하며 prior에 연결된다.

#### 이슈 4: 수치 안정성

Schur complement를 계산하려면 역행렬 $\mathbf{H}_{mm}^{-1}$이 필요하다. $\mathbf{H}_{mm}$의 조건수(condition number)가 크면 역행렬 계산이 불안정해진다.

다음과 같이 대응한다.
- $\mathbf{H}_{mm}$에 작은 정규화 항을 더한다: $(\mathbf{H}_{mm} + \epsilon \mathbf{I})^{-1}$
- LDL 분해로 풀어 수치 안정성을 높인다
- 마지널라이즈 후 $\mathbf{H}^*$가 양의 준정부호(positive semi-definite, PSD)인지 확인하고, 음의 고윳값이 생기면 가장 가까운 PSD 행렬로 바꾼다

---

## 더 읽을거리

Invariant EKF와 Equivariant Filter는 Lie 군의 대칭을 이용해, 조건이 맞는 시스템에서 오차 동역학이 추정치에 덜 의존하도록 만든다. [AI-Aided Kalman Filters (Shlezinger et al., 2025)](https://arxiv.org/abs/2410.12289)는 칼만 이득이나 모델 일부를 학습하는 방법들을 정리했다. 이런 방법이 학습 데이터와 다른 조건에서도 제대로 동작하는지, 안전한지는 따로 검증해야 한다.
