# Ch.4 — State Estimation Theory

Ch.2 covered sensor observation models, and Ch.3 addressed inter-sensor calibration. It is now time for the **algorithms** that estimate the robot's state (position, attitude, velocity, etc.) from this observation data. The Kalman filter, factor graph, and IMU preintegration covered in this chapter are the mathematical engines behind every odometry and fusion system we examine in Ch.6–8.

> **Goal**: We systematically treat state estimation theory, the mathematical foundation of sensor fusion. Starting from Bayesian filtering, we follow the technical lineage through the Kalman filter family and the particle filter, up to factor-graph-based optimization, which is at the heart of modern SLAM. We show each method's derivation in detail and explain why modern robotics systems moved from filtering to optimization.

---

## 4.1 Bayesian Filtering Framework

### 4.1.1 Definition of the State Estimation Problem

The robot state estimation problem is fundamentally a **conditional probabilistic inference** problem. What we wish to know is the posterior over the state $\mathbf{x}_k$ given all observations up to the present $\mathbf{z}_{1:k}$ and control inputs $\mathbf{u}_{1:k}$:

$$p(\mathbf{x}_k \mid \mathbf{z}_{1:k}, \mathbf{u}_{1:k})$$

Here:
- $\mathbf{x}_k \in \mathbb{R}^n$: state vector at time $k$ (position, velocity, attitude, bias, etc.)
- $\mathbf{z}_k \in \mathbb{R}^m$: observation vector at time $k$ (camera, LiDAR, IMU, etc.)
- $\mathbf{u}_k \in \mathbb{R}^l$: control input at time $k$

To solve this problem we define two models:

**Motion Model (Process Model)**:
$$\mathbf{x}_k = f(\mathbf{x}_{k-1}, \mathbf{u}_k) + \mathbf{w}_k, \quad \mathbf{w}_k \sim \mathcal{N}(\mathbf{0}, \mathbf{Q}_k)$$

$$p(\mathbf{x}_k \mid \mathbf{x}_{k-1}, \mathbf{u}_k)$$

This represents the transition probability from the previous state to the current state given the control input $\mathbf{u}_k$. The process noise $\mathbf{w}_k$ reflects the model's uncertainty.

**Observation Model (Measurement Model)**:
$$\mathbf{z}_k = h(\mathbf{x}_k) + \mathbf{v}_k, \quad \mathbf{v}_k \sim \mathcal{N}(\mathbf{0}, \mathbf{R}_k)$$

$$p(\mathbf{z}_k \mid \mathbf{x}_k)$$

This is the likelihood of the observation given the current state. The measurement noise $\mathbf{v}_k$ reflects the sensor's uncertainty.

Intuitively, the robot cannot know exactly "where it is," but by combining "how it moved" (motion model) and "what it sees" (observation model), it progressively refines a **belief** about its own position.

### 4.1.2 Markov Assumptions and Recursive Estimation

For practical filtering, we introduce the **Markov assumptions**:

1. **First-order Markov process**: The current state $\mathbf{x}_k$ depends only on the immediately previous state $\mathbf{x}_{k-1}$ and the control $\mathbf{u}_k$; it is conditionally independent of earlier states and observations:
$$p(\mathbf{x}_k \mid \mathbf{x}_{0:k-1}, \mathbf{u}_{1:k}, \mathbf{z}_{1:k-1}) = p(\mathbf{x}_k \mid \mathbf{x}_{k-1}, \mathbf{u}_k)$$

2. **Conditional observation independence**: Given the current state $\mathbf{x}_k$, the observation $\mathbf{z}_k$ is independent of everything else:
$$p(\mathbf{z}_k \mid \mathbf{x}_{0:k}, \mathbf{u}_{1:k}, \mathbf{z}_{1:k-1}) = p(\mathbf{z}_k \mid \mathbf{x}_k)$$

Thanks to these two assumptions, we can update the current estimate using only the previous estimate, without storing the entire observation history $\mathbf{z}_{1:k}$. This is the crux of **recursive estimation**.

### 4.1.3 Prediction-Update Cycle

The Bayesian filter iterates two steps:

**Prediction Step**: Using the motion model, predict the current prior from the previous posterior.

$$\boxed{p(\mathbf{x}_k \mid \mathbf{z}_{1:k-1}, \mathbf{u}_{1:k}) = \int p(\mathbf{x}_k \mid \mathbf{x}_{k-1}, \mathbf{u}_k) \, p(\mathbf{x}_{k-1} \mid \mathbf{z}_{1:k-1}, \mathbf{u}_{1:k-1}) \, d\mathbf{x}_{k-1}}$$

This integral is the **Chapman-Kolmogorov equation**. Its physical meaning is:
- $p(\mathbf{x}_{k-1} \mid \mathbf{z}_{1:k-1})$: posterior at time $k-1$ (result of the previous step)
- $p(\mathbf{x}_k \mid \mathbf{x}_{k-1}, \mathbf{u}_k)$: state transition probability (motion model)
- The predicted distribution is obtained by a weighted average of the transition probability over all possible values of the previous state.

**Update Step**: When a new observation $\mathbf{z}_k$ arrives, update the posterior via Bayes' rule.

$$\boxed{p(\mathbf{x}_k \mid \mathbf{z}_{1:k}, \mathbf{u}_{1:k}) = \frac{p(\mathbf{z}_k \mid \mathbf{x}_k) \, p(\mathbf{x}_k \mid \mathbf{z}_{1:k-1}, \mathbf{u}_{1:k})}{p(\mathbf{z}_k \mid \mathbf{z}_{1:k-1})}}$$

Here:
- Numerator $p(\mathbf{z}_k \mid \mathbf{x}_k)$: likelihood (observation model)
- Numerator $p(\mathbf{x}_k \mid \mathbf{z}_{1:k-1})$: prior obtained in the prediction step
- Denominator $p(\mathbf{z}_k \mid \mathbf{z}_{1:k-1}) = \int p(\mathbf{z}_k \mid \mathbf{x}_k) p(\mathbf{x}_k \mid \mathbf{z}_{1:k-1}) d\mathbf{x}_k$: normalization constant (evidence)

Intuitively, each time an observation arrives, we ask "how consistent is this observation with my prediction?" and revise our belief accordingly.

### 4.1.4 Why Closed-Form Solutions Fail

Although the prediction-update formulation above is theoretically perfect, in practice the Chapman-Kolmogorov integral $\int p(\mathbf{x}_k \mid \mathbf{x}_{k-1}) p(\mathbf{x}_{k-1} \mid \mathbf{z}_{1:k-1}) d\mathbf{x}_{k-1}$ cannot be solved analytically in most cases. The reasons are:

1. **Nonlinear motion/observation models**: If $f(\cdot)$ and $h(\cdot)$ are nonlinear, propagating a Gaussian distribution does not preserve Gaussianity. The posterior can take arbitrarily complex shapes.

2. **High-dimensional state spaces**: As the dimensionality of the state vector grows, the computational cost of the integral grows exponentially (curse of dimensionality).

3. **Multimodal distributions**: When the robot could be at multiple locations (e.g., global localization in a symmetric environment), the posterior becomes multimodal.

Therefore, approximations are essential in practice, and different filters arise depending on the approximation:

| Approximation | Filter | Distribution representation | Regime |
|----------|------|---------|----------|
| Linear-Gaussian assumption | Kalman Filter | $\mathcal{N}(\hat{\mathbf{x}}, \mathbf{P})$ | Linear systems |
| First-order linearization + Gaussian | EKF, ESKF | $\mathcal{N}(\hat{\mathbf{x}}, \mathbf{P})$ | Mildly nonlinear |
| Sigma point transform | UKF | $\mathcal{N}(\hat{\mathbf{x}}, \mathbf{P})$ | Moderately nonlinear |
| Sample-based | Particle Filter | $\{(\mathbf{x}^{(i)}, w^{(i)})\}_{i=1}^N$ | Strongly nonlinear, multimodal |
| Iterated linearization | IEKF | $\mathcal{N}(\hat{\mathbf{x}}, \mathbf{P})$ | Strongly nonlinear |

We derive each method in detail in what follows.

---

## 4.2 Kalman Filter Family

### 4.2.1 Kalman Filter: Derivation and Optimality

The Kalman Filter (KF) is the exact solution of the Bayesian filter under the assumptions of a **linear system + Gaussian noise**. Rudolf Kalman introduced it in his 1960 paper ["A New Approach to Linear Filtering and Prediction Problems"](https://doi.org/10.1115/1.3662552), and its practicality was established when it was applied to orbit estimation in the Apollo program.

#### Linear System Model

We assume a system with linear state transition and observation:

$$\mathbf{x}_k = \mathbf{F}_k \mathbf{x}_{k-1} + \mathbf{B}_k \mathbf{u}_k + \mathbf{w}_k, \quad \mathbf{w}_k \sim \mathcal{N}(\mathbf{0}, \mathbf{Q}_k)$$
$$\mathbf{z}_k = \mathbf{H}_k \mathbf{x}_k + \mathbf{v}_k, \quad \mathbf{v}_k \sim \mathcal{N}(\mathbf{0}, \mathbf{R}_k)$$

Here:
- $\mathbf{F}_k \in \mathbb{R}^{n \times n}$: state transition matrix
- $\mathbf{B}_k \in \mathbb{R}^{n \times l}$: control input matrix
- $\mathbf{H}_k \in \mathbb{R}^{m \times n}$: observation matrix
- $\mathbf{Q}_k \in \mathbb{R}^{n \times n}$: process noise covariance (positive semi-definite, symmetric)
- $\mathbf{R}_k \in \mathbb{R}^{m \times m}$: measurement noise covariance (positive definite, symmetric)

Key property: applying a linear transformation to a Gaussian yields another Gaussian. Thus the posterior remains Gaussian at all times and is fully described by its mean and covariance.

#### MMSE Derivation — Why the Kalman Filter Is Optimal

We derive the Kalman filter's optimality from the Minimum Mean Square Error (MMSE) perspective. What we want is the estimator that minimizes the mean squared error:

$$\hat{\mathbf{x}}_k = \arg\min_{\hat{\mathbf{x}}} \mathbb{E}[\|\mathbf{x}_k - \hat{\mathbf{x}}\|^2 \mid \mathbf{z}_{1:k}]$$

The MMSE estimator is the conditional expectation $\hat{\mathbf{x}}_k = \mathbb{E}[\mathbf{x}_k \mid \mathbf{z}_{1:k}]$. We show that in a linear-Gaussian system this coincides exactly with the Kalman filter's state update equations.

**Bayesian derivation**: Assume the posterior at time $k-1$ is Gaussian:

$$p(\mathbf{x}_{k-1} \mid \mathbf{z}_{1:k-1}) = \mathcal{N}(\hat{\mathbf{x}}_{k-1|k-1}, \mathbf{P}_{k-1|k-1})$$

**Step 1: Prediction — evaluate the Chapman-Kolmogorov integral**

Under the linear motion model:
$$\mathbf{x}_k = \mathbf{F}_k \mathbf{x}_{k-1} + \mathbf{B}_k \mathbf{u}_k + \mathbf{w}_k$$

Since $\mathbf{x}_{k-1}$ is Gaussian and $\mathbf{w}_k$ is independent Gaussian, $\mathbf{x}_k$ is also Gaussian. Computing its mean and covariance:

$$\hat{\mathbf{x}}_{k|k-1} = \mathbb{E}[\mathbf{x}_k \mid \mathbf{z}_{1:k-1}] = \mathbf{F}_k \hat{\mathbf{x}}_{k-1|k-1} + \mathbf{B}_k \mathbf{u}_k$$

The covariance follows from $\tilde{\mathbf{x}}_{k|k-1} = \mathbf{x}_k - \hat{\mathbf{x}}_{k|k-1} = \mathbf{F}_k \tilde{\mathbf{x}}_{k-1|k-1} + \mathbf{w}_k$:

$$\mathbf{P}_{k|k-1} = \mathbb{E}[\tilde{\mathbf{x}}_{k|k-1} \tilde{\mathbf{x}}_{k|k-1}^\top] = \mathbf{F}_k \mathbf{P}_{k-1|k-1} \mathbf{F}_k^\top + \mathbf{Q}_k$$

Predicted distribution: $p(\mathbf{x}_k \mid \mathbf{z}_{1:k-1}) = \mathcal{N}(\hat{\mathbf{x}}_{k|k-1}, \mathbf{P}_{k|k-1})$.

**Step 2: Update — apply Bayes' rule**

From the observation model $\mathbf{z}_k = \mathbf{H}_k \mathbf{x}_k + \mathbf{v}_k$, construct the joint distribution of the predicted state and the observation:

$$\begin{bmatrix} \mathbf{x}_k \\ \mathbf{z}_k \end{bmatrix} \sim \mathcal{N}\left( \begin{bmatrix} \hat{\mathbf{x}}_{k|k-1} \\ \mathbf{H}_k \hat{\mathbf{x}}_{k|k-1} \end{bmatrix}, \begin{bmatrix} \mathbf{P}_{k|k-1} & \mathbf{P}_{k|k-1} \mathbf{H}_k^\top \\ \mathbf{H}_k \mathbf{P}_{k|k-1} & \mathbf{S}_k \end{bmatrix} \right)$$

Here $\mathbf{S}_k = \mathbf{H}_k \mathbf{P}_{k|k-1} \mathbf{H}_k^\top + \mathbf{R}_k \in \mathbb{R}^{m \times m}$ is the innovation covariance.

Applying the formula for the conditional of a Gaussian joint (the key Gaussian property: if the joint is Gaussian, the conditional is also Gaussian, and the conditional mean equals the original mean plus a correction proportional to the correlation with the observation):

$$\hat{\mathbf{x}}_{k|k} = \hat{\mathbf{x}}_{k|k-1} + \mathbf{P}_{k|k-1} \mathbf{H}_k^\top \mathbf{S}_k^{-1} (\mathbf{z}_k - \mathbf{H}_k \hat{\mathbf{x}}_{k|k-1})$$

Defining the **Kalman gain** $\mathbf{K}_k \in \mathbb{R}^{n \times m}$:

$$\boxed{\mathbf{K}_k = \mathbf{P}_{k|k-1} \mathbf{H}_k^\top \mathbf{S}_k^{-1} = \mathbf{P}_{k|k-1} \mathbf{H}_k^\top (\mathbf{H}_k \mathbf{P}_{k|k-1} \mathbf{H}_k^\top + \mathbf{R}_k)^{-1}}$$

Intuitive meaning of the Kalman gain:
- $\mathbf{R}_k \to \mathbf{0}$ (observation is very accurate): $\mathbf{K}_k \to \mathbf{H}_k^{-1}$ → trust the observation almost entirely
- $\mathbf{P}_{k|k-1} \to \mathbf{0}$ (prediction is very accurate): $\mathbf{K}_k \to \mathbf{0}$ → ignore the observation and trust the prediction
- The Kalman gain automatically determines the **optimal weighting** between the uncertainty of prediction and that of observation.

**Innovation**:
$$\tilde{\mathbf{y}}_k = \mathbf{z}_k - \mathbf{H}_k \hat{\mathbf{x}}_{k|k-1} \in \mathbb{R}^m$$

The difference between the predicted observation and the actual one. If it is zero, the prediction was perfect.

**State update**:
$$\boxed{\hat{\mathbf{x}}_{k|k} = \hat{\mathbf{x}}_{k|k-1} + \mathbf{K}_k \tilde{\mathbf{y}}_k}$$

**Covariance update**:
$$\boxed{\mathbf{P}_{k|k} = (\mathbf{I} - \mathbf{K}_k \mathbf{H}_k) \mathbf{P}_{k|k-1}}$$

This covariance update is numerically more stable when written in the Joseph form $\mathbf{P}_{k|k} = (\mathbf{I} - \mathbf{K}_k \mathbf{H}_k) \mathbf{P}_{k|k-1} (\mathbf{I} - \mathbf{K}_k \mathbf{H}_k)^\top + \mathbf{K}_k \mathbf{R}_k \mathbf{K}_k^\top$. Even if $\mathbf{K}_k$ has numerical errors, the Joseph form guarantees that $\mathbf{P}_{k|k}$ remains symmetric and positive semi-definite.

#### KF Optimality Theorem

**Theorem**: In a linear-Gaussian system, the Kalman filter is optimal in the following three senses:
1. **MMSE (Minimum Mean Square Error) estimator**: minimizes the mean squared error
2. **MAP (Maximum A Posteriori) estimator**: in a Gaussian, mean equals mode, so MAP and MMSE coincide
3. **BLUE (Best Linear Unbiased Estimator)**: even without the Gaussian assumption, it is the linear unbiased estimator with minimum variance

Kalman's 1960 paper replaced Wiener's frequency-domain approach with a state-space, time-domain formulation, enabling natural extensions to time-varying and multivariate systems — an innovation of far-reaching consequences.

#### Python Implementation

```python
import numpy as np

class KalmanFilter:
    """Linear Kalman Filter implementation.
    
    State-space model:
        x_k = F @ x_{k-1} + B @ u_k + w_k,  w_k ~ N(0, Q)
        z_k = H @ x_k + v_k,                 v_k ~ N(0, R)
    """
    def __init__(self, F, H, Q, R, B=None):
        """
        Parameters
        ----------
        F : ndarray, shape (n, n) — state transition matrix
        H : ndarray, shape (m, n) — observation matrix
        Q : ndarray, shape (n, n) — process noise covariance
        R : ndarray, shape (m, m) — measurement noise covariance
        B : ndarray, shape (n, l) — control input matrix (optional)
        """
        self.F = F
        self.H = H
        self.Q = Q
        self.R = R
        self.B = B
        self.n = F.shape[0]
        
    def predict(self, x, P, u=None):
        """Prediction step.
        
        Parameters
        ----------
        x : ndarray, shape (n,) — previous state estimate
        P : ndarray, shape (n, n) — previous covariance
        u : ndarray, shape (l,) — control input (optional)
        
        Returns
        -------
        x_pred : ndarray, shape (n,) — predicted state
        P_pred : ndarray, shape (n, n) — predicted covariance
        """
        x_pred = self.F @ x
        if self.B is not None and u is not None:
            x_pred += self.B @ u
        P_pred = self.F @ P @ self.F.T + self.Q
        return x_pred, P_pred
    
    def update(self, x_pred, P_pred, z):
        """Update step.
        
        Parameters
        ----------
        x_pred : ndarray, shape (n,) — predicted state
        P_pred : ndarray, shape (n, n) — predicted covariance
        z : ndarray, shape (m,) — observation
        
        Returns
        -------
        x_upd : ndarray, shape (n,) — updated state
        P_upd : ndarray, shape (n, n) — updated covariance
        """
        # innovation
        y = z - self.H @ x_pred                          # (m,)
        # innovation covariance
        S = self.H @ P_pred @ self.H.T + self.R           # (m, m)
        # Kalman gain
        K = P_pred @ self.H.T @ np.linalg.inv(S)          # (n, m)
        # state update
        x_upd = x_pred + K @ y                            # (n,)
        # covariance update (Joseph form for numerical stability)
        I_KH = np.eye(self.n) - K @ self.H                # (n, n)
        P_upd = I_KH @ P_pred @ I_KH.T + K @ self.R @ K.T  # (n, n)
        return x_upd, P_upd


# Example: position estimation under a 1D constant-velocity model
# state: [position, velocity]^T
dt = 0.1  # time step
F = np.array([[1, dt],
              [0, 1]])      # (2, 2) constant-velocity transition matrix
H = np.array([[1, 0]])       # (1, 2) observe position only
Q = np.array([[dt**3/3, dt**2/2],
              [dt**2/2, dt]]) * 0.1  # (2, 2) process noise (constant-acceleration model)
R = np.array([[1.0]])         # (1, 1) observation noise variance

kf = KalmanFilter(F, H, Q, R)

# initial state
x = np.array([0.0, 1.0])  # position=0, velocity=1
P = np.eye(2) * 10.0       # large initial uncertainty

# simulation
np.random.seed(42)
true_positions = []
estimated_positions = []

for k in range(100):
    # ground truth
    true_pos = 0.0 + 1.0 * k * dt  # constant-velocity motion
    true_positions.append(true_pos)
    
    # predict
    x, P = kf.predict(x, P)
    
    # noisy observation
    z = np.array([true_pos + np.random.randn() * 1.0])
    
    # update
    x, P = kf.update(x, P, z)
    estimated_positions.append(x[0])

print(f"final position estimate: {x[0]:.3f}, true: {true_positions[-1]:.3f}")
print(f"final position uncertainty (1 sigma): {np.sqrt(P[0,0]):.3f}")
```

### 4.2.2 Extended Kalman Filter (EKF)

Real robotic systems are almost always nonlinear. The 3D→2D projection of a camera, the quaternion-based rotation of an IMU, LiDAR scan matching — all are nonlinear functions. The EKF is the most direct way to extend the Kalman filter to nonlinear systems.

#### Core Idea: First-Order Taylor Expansion (Linearization)

Nonlinear system model:
$$\mathbf{x}_k = f(\mathbf{x}_{k-1}, \mathbf{u}_k) + \mathbf{w}_k$$
$$\mathbf{z}_k = h(\mathbf{x}_k) + \mathbf{v}_k$$

In this system, passing a Gaussian through a nonlinear function no longer yields a Gaussian. The EKF's key approximation is to **linearize the function about the current estimate via a first-order Taylor expansion**.

Linearization of the motion model (about the estimate $\hat{\mathbf{x}}_{k-1|k-1}$):

$$f(\mathbf{x}_{k-1}, \mathbf{u}_k) \approx f(\hat{\mathbf{x}}_{k-1|k-1}, \mathbf{u}_k) + \mathbf{F}_k (\mathbf{x}_{k-1} - \hat{\mathbf{x}}_{k-1|k-1})$$

$$\mathbf{F}_k = \left.\frac{\partial f}{\partial \mathbf{x}}\right|_{\hat{\mathbf{x}}_{k-1|k-1}, \mathbf{u}_k} \in \mathbb{R}^{n \times n}$$

Linearization of the observation model (about the predicted estimate $\hat{\mathbf{x}}_{k|k-1}$):

$$h(\mathbf{x}_k) \approx h(\hat{\mathbf{x}}_{k|k-1}) + \mathbf{H}_k (\mathbf{x}_k - \hat{\mathbf{x}}_{k|k-1})$$

$$\mathbf{H}_k = \left.\frac{\partial h}{\partial \mathbf{x}}\right|_{\hat{\mathbf{x}}_{k|k-1}} \in \mathbb{R}^{m \times n}$$

#### EKF Algorithm

**Prediction step**:
$$\hat{\mathbf{x}}_{k|k-1} = f(\hat{\mathbf{x}}_{k-1|k-1}, \mathbf{u}_k)$$
$$\mathbf{P}_{k|k-1} = \mathbf{F}_k \mathbf{P}_{k-1|k-1} \mathbf{F}_k^\top + \mathbf{Q}_k$$

Note: the state prediction uses the nonlinear function $f$ directly, while the covariance propagation uses the Jacobian $\mathbf{F}_k$.

**Update step**:
$$\tilde{\mathbf{y}}_k = \mathbf{z}_k - h(\hat{\mathbf{x}}_{k|k-1})$$
$$\mathbf{S}_k = \mathbf{H}_k \mathbf{P}_{k|k-1} \mathbf{H}_k^\top + \mathbf{R}_k$$
$$\mathbf{K}_k = \mathbf{P}_{k|k-1} \mathbf{H}_k^\top \mathbf{S}_k^{-1}$$
$$\hat{\mathbf{x}}_{k|k} = \hat{\mathbf{x}}_{k|k-1} + \mathbf{K}_k \tilde{\mathbf{y}}_k$$
$$\mathbf{P}_{k|k} = (\mathbf{I} - \mathbf{K}_k \mathbf{H}_k) \mathbf{P}_{k|k-1}$$

Observe that the innovation $\tilde{\mathbf{y}}_k$ also uses the nonlinear function $h$ directly.

#### Limitations of the EKF

1. **Linearization error**: The stronger the nonlinearity, the larger the error of the first-order approximation. This can hurt the filter's consistency — the actual error may be larger than the uncertainty the filter reports.

2. **Jacobian computation burden**: We must compute the analytical derivatives $\mathbf{F}_k$ and $\mathbf{H}_k$ at every time step. When the system is complex, the Jacobian derivation becomes cumbersome and error-prone.

3. **Unimodal assumption**: Gaussians are always unimodal, so multimodal posteriors cannot be represented.

### 4.2.3 Error-State Kalman Filter (ESKF)

The ESKF (Error-State Kalman Filter) is the most widely used filter form in modern robotic sensor fusion systems. Nearly all major VIO/LIO systems — MSCKF, VINS-Mono, OpenVINS, FAST-LIO, and others — adopt the ESKF.

#### Why ESKF Instead of EKF

When estimating a robot state — especially one that includes 3D orientation — using the EKF directly causes several problems:

**Problem 1: Non-Euclidean nature of rotation**

3D rotation lives on the SO(3) manifold, not $\mathbb{R}^n$. Representing it with a quaternion $\mathbf{q} \in \mathbb{H}$ imposes $\|\mathbf{q}\| = 1$; representing it with a rotation matrix $\mathbf{R} \in SO(3)$ imposes $\mathbf{R}^\top \mathbf{R} = \mathbf{I}$, $\det(\mathbf{R}) = 1$.

In the EKF state update $\hat{\mathbf{x}} \leftarrow \hat{\mathbf{x}} + \mathbf{K} \tilde{\mathbf{y}}$, the "+" is Euclidean addition. Adding a vector to a quaternion breaks the unit norm. Re-normalizing after the update is a makeshift fix that is not theoretically correct and leads to consistency problems.

**Problem 2: The error state is "almost zero"**

The error state $\delta\mathbf{x} = \mathbf{x} - \hat{\mathbf{x}}$ stays near zero by construction (it is reset every update). Therefore the first-order linearization is extremely accurate. In contrast, linearizing about the original state becomes inaccurate when motion is large.

**Problem 3: Separation of slow-varying and fast-varying states**

Separating slow-varying states such as IMU biases from fast-varying states such as velocity and attitude allows update strategies tailored to each.

#### ESKF Structure

The ESKF maintains two states simultaneously:

1. **Nominal state** $\hat{\mathbf{x}}$: integrated through the nonlinear motion model without noise terms. It does not track uncertainty.

2. **Error state** $\delta\mathbf{x}$: the difference between the nominal state and the true state. It is estimated with a Kalman filter. Since the error state is a "small value" by construction, linearization error is minimized.

The true state is recovered by composition of the two:

$$\mathbf{x}_{\text{true}} = \hat{\mathbf{x}} \boxplus \delta\mathbf{x}$$

Here $\boxplus$ is the composition operation on the manifold. For Euclidean components it is ordinary addition; for rotation components it is:

$$\mathbf{R}_{\text{true}} = \hat{\mathbf{R}} \cdot \text{Exp}(\delta\boldsymbol{\theta})$$

or, in quaternion form:

$$\mathbf{q}_{\text{true}} = \hat{\mathbf{q}} \otimes \begin{bmatrix} 1 \\ \frac{1}{2}\delta\boldsymbol{\theta} \end{bmatrix} \approx \hat{\mathbf{q}} \otimes \delta\mathbf{q}$$

Here $\delta\boldsymbol{\theta} \in \mathbb{R}^3$ is the angle-axis representation of the rotational error.

#### State Vector of an IMU-Based ESKF

Typical state vector in an IMU-camera/LiDAR fusion system:

**Nominal state** (16-dimensional, with a quaternion):
$$\hat{\mathbf{x}} = \begin{bmatrix} {}^W\hat{\mathbf{p}} \\ {}^W\hat{\mathbf{v}} \\ \hat{\mathbf{q}}_{WB} \\ \hat{\mathbf{b}}_a \\ \hat{\mathbf{b}}_g \end{bmatrix} \in \mathbb{R}^{3} \times \mathbb{R}^{3} \times \mathbb{S}^3 \times \mathbb{R}^{3} \times \mathbb{R}^{3}$$

**Error state** (15-dimensional — minimal parameterization of rotation):
$$\delta\mathbf{x} = \begin{bmatrix} \delta\mathbf{p} \\ \delta\mathbf{v} \\ \delta\boldsymbol{\theta} \\ \delta\mathbf{b}_a \\ \delta\mathbf{b}_g \end{bmatrix} \in \mathbb{R}^{15}$$

The key point: the error representation of the quaternion (4D) is the 3D vector $\delta\boldsymbol{\theta}$. Because of the unit-quaternion constraint the actual DoF is 3, and the ESKF naturally uses this minimal parameterization. Putting the quaternion directly into the state of an EKF introduces one redundant DoF in its 4D representation, which makes the covariance matrix singular.

#### ESKF Algorithm

**Step 1: Nominal state propagation (IMU mechanization)**

Integrate the nominal state from the IMU measurements $(\tilde{\boldsymbol{\omega}}_k, \tilde{\mathbf{a}}_k)$. Subtract the biases and ignore the noise terms:

$$\hat{\mathbf{q}}_{k+1} = \hat{\mathbf{q}}_k \otimes \mathbf{q}\{(\tilde{\boldsymbol{\omega}}_k - \hat{\mathbf{b}}_{g,k}) \Delta t\}$$
$$\hat{\mathbf{v}}_{k+1} = \hat{\mathbf{v}}_k + (\hat{\mathbf{R}}_k (\tilde{\mathbf{a}}_k - \hat{\mathbf{b}}_{a,k}) + \mathbf{g}) \Delta t$$
$$\hat{\mathbf{p}}_{k+1} = \hat{\mathbf{p}}_k + \hat{\mathbf{v}}_k \Delta t + \frac{1}{2}(\hat{\mathbf{R}}_k (\tilde{\mathbf{a}}_k - \hat{\mathbf{b}}_{a,k}) + \mathbf{g}) \Delta t^2$$
$$\hat{\mathbf{b}}_{a,k+1} = \hat{\mathbf{b}}_{a,k}$$
$$\hat{\mathbf{b}}_{g,k+1} = \hat{\mathbf{b}}_{g,k}$$

Here $\hat{\mathbf{R}}_k = \mathbf{R}(\hat{\mathbf{q}}_k) \in SO(3)$, and $\mathbf{g} = [0, 0, -9.81]^\top \, \text{m/s}^2$ is the gravity vector.

**Step 2: Error-state propagation (prediction)**

We derive the continuous-time dynamics of the error state. Substituting the true IMU measurements $\tilde{\boldsymbol{\omega}} = \boldsymbol{\omega} + \mathbf{b}_g + \mathbf{n}_g$, $\tilde{\mathbf{a}} = \mathbf{R}^\top(\mathbf{a}_W - \mathbf{g}) + \mathbf{b}_a + \mathbf{n}_a$, and subtracting the nominal state:

$$\delta\dot{\boldsymbol{\theta}} = -[\tilde{\boldsymbol{\omega}} - \hat{\mathbf{b}}_g]_\times \delta\boldsymbol{\theta} - \delta\mathbf{b}_g - \mathbf{n}_g$$
$$\delta\dot{\mathbf{v}} = -\hat{\mathbf{R}}[\tilde{\mathbf{a}} - \hat{\mathbf{b}}_a]_\times \delta\boldsymbol{\theta} - \hat{\mathbf{R}} \delta\mathbf{b}_a - \hat{\mathbf{R}} \mathbf{n}_a$$
$$\delta\dot{\mathbf{p}} = \delta\mathbf{v}$$
$$\delta\dot{\mathbf{b}}_a = \mathbf{n}_{ba}$$
$$\delta\dot{\mathbf{b}}_g = \mathbf{n}_{bg}$$

Here $[\mathbf{a}]_\times$ is the skew-symmetric matrix of the vector $\mathbf{a}$:

$$[\mathbf{a}]_\times = \begin{bmatrix} 0 & -a_3 & a_2 \\ a_3 & 0 & -a_1 \\ -a_2 & a_1 & 0 \end{bmatrix} \in \mathbb{R}^{3 \times 3}$$

In matrix form:

$$\delta\dot{\mathbf{x}} = \mathbf{F}_c \delta\mathbf{x} + \mathbf{G}_c \mathbf{n}$$

$$\mathbf{F}_c = \begin{bmatrix}
\mathbf{0}_3 & \mathbf{I}_3 & \mathbf{0}_3 & \mathbf{0}_3 & \mathbf{0}_3 \\
\mathbf{0}_3 & \mathbf{0}_3 & -\hat{\mathbf{R}}[\tilde{\mathbf{a}} - \hat{\mathbf{b}}_a]_\times & -\hat{\mathbf{R}} & \mathbf{0}_3 \\
\mathbf{0}_3 & \mathbf{0}_3 & -[\tilde{\boldsymbol{\omega}} - \hat{\mathbf{b}}_g]_\times & \mathbf{0}_3 & -\mathbf{I}_3 \\
\mathbf{0}_3 & \mathbf{0}_3 & \mathbf{0}_3 & \mathbf{0}_3 & \mathbf{0}_3 \\
\mathbf{0}_3 & \mathbf{0}_3 & \mathbf{0}_3 & \mathbf{0}_3 & \mathbf{0}_3
\end{bmatrix} \in \mathbb{R}^{15 \times 15}$$

Discretization (first-order approximation): $\mathbf{F}_d \approx \mathbf{I} + \mathbf{F}_c \Delta t$.

$$\mathbf{P}_{k+1|k} = \mathbf{F}_d \mathbf{P}_{k|k} \mathbf{F}_d^\top + \mathbf{G}_d \mathbf{Q}_d \mathbf{G}_d^\top$$

**Step 3: Observation update**

When a camera/LiDAR observation arrives, perform a standard EKF update on the error state. Linearize the observation model $\mathbf{z} = h(\mathbf{x}_{\text{true}})$ with respect to the error state:

$$\mathbf{z} - h(\hat{\mathbf{x}}) \approx \mathbf{H} \delta\mathbf{x} + \mathbf{v}$$

$$\mathbf{H} = \frac{\partial h}{\partial \delta\mathbf{x}}\bigg|_{\hat{\mathbf{x}}} \in \mathbb{R}^{m \times 15}$$

This Jacobian is computed by the chain rule:

$$\mathbf{H} = \frac{\partial h}{\partial \mathbf{x}_{\text{true}}} \cdot \frac{\partial \mathbf{x}_{\text{true}}}{\partial \delta\mathbf{x}}\bigg|_{\delta\mathbf{x}=\mathbf{0}}$$

Standard Kalman update:
$$\mathbf{K} = \mathbf{P}_{k|k-1} \mathbf{H}^\top (\mathbf{H} \mathbf{P}_{k|k-1} \mathbf{H}^\top + \mathbf{R})^{-1}$$
$$\delta\hat{\mathbf{x}} = \mathbf{K}(\mathbf{z} - h(\hat{\mathbf{x}}))$$
$$\mathbf{P}_{k|k} = (\mathbf{I} - \mathbf{K}\mathbf{H})\mathbf{P}_{k|k-1}$$

**Step 4: Injection of the error state and reset**

Inject the error-state estimate $\delta\hat{\mathbf{x}}$ into the nominal state:

$$\hat{\mathbf{p}} \leftarrow \hat{\mathbf{p}} + \delta\hat{\mathbf{p}}$$
$$\hat{\mathbf{v}} \leftarrow \hat{\mathbf{v}} + \delta\hat{\mathbf{v}}$$
$$\hat{\mathbf{q}} \leftarrow \hat{\mathbf{q}} \otimes \mathbf{q}\{\delta\hat{\boldsymbol{\theta}}\}$$
$$\hat{\mathbf{b}}_a \leftarrow \hat{\mathbf{b}}_a + \delta\hat{\mathbf{b}}_a$$
$$\hat{\mathbf{b}}_g \leftarrow \hat{\mathbf{b}}_g + \delta\hat{\mathbf{b}}_g$$

After injection, reset the error state to $\delta\hat{\mathbf{x}} \leftarrow \mathbf{0}$. The covariance must also be transformed by the reset Jacobian:

$$\mathbf{P} \leftarrow \mathbf{G} \mathbf{P} \mathbf{G}^\top$$

Here $\mathbf{G} = \frac{\partial (\delta\mathbf{x} \boxminus \delta\hat{\mathbf{x}})}{\partial \delta\mathbf{x}}\big|_{\delta\hat{\mathbf{x}}}$. In practice, when $\delta\hat{\boldsymbol{\theta}}$ is small, we often approximate $\mathbf{G} \approx \mathbf{I}$.

### 4.2.4 Unscented Kalman Filter (UKF)

The UKF rests on the insight that "approximating a distribution is easier than approximating a nonlinear function." The EKF approximates nonlinear functions by linearizing them, whereas the UKF leaves the nonlinear function intact and approximates the distribution with a finite set of **sigma points**.

#### Unscented Transform

For an $n$-dimensional Gaussian random variable $\mathbf{x} \sim \mathcal{N}(\hat{\mathbf{x}}, \mathbf{P})$, we generate $2n+1$ sigma points and weights:

$$\boldsymbol{\chi}_0 = \hat{\mathbf{x}}, \quad w_0 = \frac{\lambda}{n + \lambda}$$
$$\boldsymbol{\chi}_i = \hat{\mathbf{x}} + \left(\sqrt{(n+\lambda)\mathbf{P}}\right)_i, \quad w_i = \frac{1}{2(n+\lambda)}, \quad i = 1, \ldots, n$$
$$\boldsymbol{\chi}_{n+i} = \hat{\mathbf{x}} - \left(\sqrt{(n+\lambda)\mathbf{P}}\right)_i, \quad w_{n+i} = \frac{1}{2(n+\lambda)}, \quad i = 1, \ldots, n$$

Here:
- $\lambda = \alpha^2(n + \kappa) - n$ is a scaling parameter ($\alpha$: controls the sigma-point spread; $\kappa$: auxiliary parameter, typically $\kappa = 0$ or $3-n$)
- $\left(\sqrt{(n+\lambda)\mathbf{P}}\right)_i$ is the $i$-th column of the Cholesky factor of $(n+\lambda)\mathbf{P}$

Pass each sigma point through the nonlinear function:

$$\boldsymbol{\gamma}_i = f(\boldsymbol{\chi}_i)$$

Recover the mean and covariance from the transformed sigma points:

$$\hat{\mathbf{y}} = \sum_{i=0}^{2n} w_i^{(m)} \boldsymbol{\gamma}_i$$
$$\mathbf{P}_y = \sum_{i=0}^{2n} w_i^{(c)} (\boldsymbol{\gamma}_i - \hat{\mathbf{y}})(\boldsymbol{\gamma}_i - \hat{\mathbf{y}})^\top$$

The weights $w_i^{(m)}$ and $w_i^{(c)}$ are used for the mean and covariance respectively; $w_0^{(c)} = w_0^{(m)} + (1 - \alpha^2 + \beta)$ ($\beta = 2$ is optimal for Gaussians), and the remaining weights are identical.

#### UKF Algorithm

**Prediction step**:
1. Generate sigma points from the current state $(\hat{\mathbf{x}}_{k-1|k-1}, \mathbf{P}_{k-1|k-1})$
2. Pass each sigma point through the motion model: $\boldsymbol{\chi}_{k|k-1}^{(i)} = f(\boldsymbol{\chi}_{k-1|k-1}^{(i)}, \mathbf{u}_k)$
3. Compute the predicted mean and covariance: $\hat{\mathbf{x}}_{k|k-1} = \sum w_i^{(m)} \boldsymbol{\chi}_{k|k-1}^{(i)}$
4. $\mathbf{P}_{k|k-1} = \sum w_i^{(c)} (\boldsymbol{\chi}_{k|k-1}^{(i)} - \hat{\mathbf{x}}_{k|k-1})(\cdots)^\top + \mathbf{Q}_k$

**Update step**:
1. Regenerate sigma points from the predicted state (or reuse those from the prediction step)
2. Pass them through the observation model: $\boldsymbol{\zeta}_k^{(i)} = h(\boldsymbol{\chi}_{k|k-1}^{(i)})$
3. Predicted observation mean: $\hat{\mathbf{z}}_k = \sum w_i^{(m)} \boldsymbol{\zeta}_k^{(i)}$
4. Observation covariance: $\mathbf{P}_{zz} = \sum w_i^{(c)} (\boldsymbol{\zeta}_k^{(i)} - \hat{\mathbf{z}}_k)(\cdots)^\top + \mathbf{R}_k$
5. Cross-covariance: $\mathbf{P}_{xz} = \sum w_i^{(c)} (\boldsymbol{\chi}_{k|k-1}^{(i)} - \hat{\mathbf{x}}_{k|k-1})(\boldsymbol{\zeta}_k^{(i)} - \hat{\mathbf{z}}_k)^\top$
6. Kalman gain: $\mathbf{K}_k = \mathbf{P}_{xz} \mathbf{P}_{zz}^{-1}$
7. Update: $\hat{\mathbf{x}}_{k|k} = \hat{\mathbf{x}}_{k|k-1} + \mathbf{K}_k (\mathbf{z}_k - \hat{\mathbf{z}}_k)$
8. $\mathbf{P}_{k|k} = \mathbf{P}_{k|k-1} - \mathbf{K}_k \mathbf{P}_{zz} \mathbf{K}_k^\top$

#### Pros and Cons of the UKF

**Pros**:
- No Jacobian computation. Big advantage for complex observation models (e.g., camera projection with distortion).
- Captures nonlinearity up to second order exactly (EKF only up to first order).
- Can be simpler to implement than the EKF (function calls replace Jacobian derivation).

**Cons**:
- Each of the $2n+1$ sigma points must be pushed through the nonlinear function, so the computational cost grows with the state dimension $n$.
- Handling manifold states (e.g., SO(3)) requires replacing sigma-point generation and statistical aggregation with manifold operations, which is not clean.
- Why the ESKF tends to be preferred over the UKF in practice: the ESKF already operates on the error state (a small value), so first-order linearization is accurate enough, manifolds are handled naturally, and the cost is lower.

### 4.2.5 Iterated Extended Kalman Filter (IEKF)

The IEKF improves the accuracy of nonlinear observation handling by relinearizing iteratively in the update step, instead of linearizing only once.

#### Motivation: Linearization Error in the EKF Update

In the EKF, the Jacobian $\mathbf{H}_k$ of the observation model is computed at the predicted estimate $\hat{\mathbf{x}}_{k|k-1}$. However, if the post-update estimate $\hat{\mathbf{x}}_{k|k}$ differs significantly from $\hat{\mathbf{x}}_{k|k-1}$, the linearization point is no longer optimal. The IEKF mitigates this by relinearizing at the post-update estimate and repeating the update.

#### IEKF Algorithm

The prediction step is identical to the EKF. In the update step we iterate:

Initialization: $\hat{\mathbf{x}}^{(0)} = \hat{\mathbf{x}}_{k|k-1}$

For $j = 0, 1, 2, \ldots$ until convergence:

$$\mathbf{H}^{(j)} = \left.\frac{\partial h}{\partial \mathbf{x}}\right|_{\hat{\mathbf{x}}^{(j)}}$$

$$\mathbf{K}^{(j)} = \mathbf{P}_{k|k-1} \mathbf{H}^{(j)\top} (\mathbf{H}^{(j)} \mathbf{P}_{k|k-1} \mathbf{H}^{(j)\top} + \mathbf{R}_k)^{-1}$$

$$\hat{\mathbf{x}}^{(j+1)} = \hat{\mathbf{x}}_{k|k-1} + \mathbf{K}^{(j)} \left[\mathbf{z}_k - h(\hat{\mathbf{x}}^{(j)}) - \mathbf{H}^{(j)}(\hat{\mathbf{x}}_{k|k-1} - \hat{\mathbf{x}}^{(j)})\right]$$

After convergence: $\hat{\mathbf{x}}_{k|k} = \hat{\mathbf{x}}^{(j+1)}$, $\mathbf{P}_{k|k} = (\mathbf{I} - \mathbf{K}^{(j)} \mathbf{H}^{(j)}) \mathbf{P}_{k|k-1}$.

The IEKF is effectively equivalent to performing **Gauss-Newton optimization** in the observation update step. This perspective is important for understanding the connection to factor-graph-based optimization that we establish later.

Why FAST-LIO2 adopts the IEKF: the LiDAR point-to-plane/point-to-edge observation model is strongly nonlinear, and hundreds to thousands of points must be updated together. In such settings the IEKF's iterated linearization yields a considerably more accurate result than a single EKF update.

---

## 4.3 Particle Filter

### 4.3.1 Overview of Sequential Monte Carlo (SMC)

The particle filter (PF), or Sequential Monte Carlo (SMC) method, represents the posterior as a set of weighted samples (particles). Because it does not require the Gaussian assumption, it can handle multimodal distributions and strongly nonlinear systems.

Particle approximation of the posterior:

$$p(\mathbf{x}_k \mid \mathbf{z}_{1:k}) \approx \sum_{i=1}^{N} w_k^{(i)} \delta(\mathbf{x}_k - \mathbf{x}_k^{(i)})$$

Here:
- $N$: number of particles
- $\mathbf{x}_k^{(i)}$: state of the $i$-th particle
- $w_k^{(i)}$: weight of the $i$-th particle ($\sum_{i=1}^N w_k^{(i)} = 1$)
- $\delta(\cdot)$: Dirac delta function

As $N \to \infty$, the particle distribution converges to the true posterior. In practice we approximate with a finite number of particles; the number is chosen based on the complexity of the problem and the dimensionality of the state space.

### 4.3.2 Importance Sampling

Sampling directly from the posterior $p(\mathbf{x}_k \mid \mathbf{z}_{1:k})$ is generally impossible. Instead, we draw samples from a **proposal distribution** $q(\mathbf{x}_k \mid \mathbf{x}_{k-1}, \mathbf{z}_k)$ and correct them with importance weights.

We derive the recursive weight update. From Bayes' rule:

$$p(\mathbf{x}_{0:k} \mid \mathbf{z}_{1:k}) = \frac{p(\mathbf{z}_k \mid \mathbf{x}_k) \, p(\mathbf{x}_k \mid \mathbf{x}_{k-1}) \, p(\mathbf{x}_{0:k-1} \mid \mathbf{z}_{1:k-1})}{p(\mathbf{z}_k \mid \mathbf{z}_{1:k-1})}$$

Dividing by the proposal gives the importance ratio:

$$w_k^{(i)} \propto w_{k-1}^{(i)} \cdot \frac{p(\mathbf{z}_k \mid \mathbf{x}_k^{(i)}) \, p(\mathbf{x}_k^{(i)} \mid \mathbf{x}_{k-1}^{(i)})}{q(\mathbf{x}_k^{(i)} \mid \mathbf{x}_{k-1}^{(i)}, \mathbf{z}_k)}$$

**Simplest proposal**: use the transition prior as the proposal, i.e., $q(\mathbf{x}_k \mid \mathbf{x}_{k-1}, \mathbf{z}_k) = p(\mathbf{x}_k \mid \mathbf{x}_{k-1})$. The weights then simplify:

$$w_k^{(i)} \propto w_{k-1}^{(i)} \cdot p(\mathbf{z}_k \mid \mathbf{x}_k^{(i)})$$

Each particle's weight is proportional to the observation likelihood at that particle's location. Intuitively, particles consistent with the observation receive high weights, while inconsistent particles receive low weights.

The optimal proposal is $q^*(\mathbf{x}_k \mid \mathbf{x}_{k-1}^{(i)}, \mathbf{z}_k) = p(\mathbf{x}_k \mid \mathbf{x}_{k-1}^{(i)}, \mathbf{z}_k)$, but in most cases this cannot be obtained.

### 4.3.3 Resampling

Propagating particles through the proposal leads to a **weight degeneracy** problem: a few particles accumulate most of the weight while the others become negligibly small. In effect, only a small subset of particles contribute, and the quality of the approximation deteriorates rapidly.

We diagnose degeneracy using the effective sample size:

$$N_{\text{eff}} = \frac{1}{\sum_{i=1}^N (w_k^{(i)})^2}$$

If $N_{\text{eff}} < N_{\text{threshold}}$ (typically $N/2$), we perform resampling.

**Resampling**: the process of duplicating high-weight particles and removing low-weight ones to make the weights uniform.

Main resampling strategies:

**Multinomial resampling**: draw $N$ independent samples using the weights as probabilities. The most intuitive but has high variance.

**Systematic resampling**: generate a single uniform random number $U_0 \sim \text{Uniform}(0, 1/N)$, then traverse the CDF with $U_i = U_0 + (i-1)/N$ to resample. Has the smallest variance and is the most used in practice.

**Stratified resampling**: use independent uniform random numbers within each stratum. Intermediate between systematic and multinomial.

```python
import numpy as np

def systematic_resampling(weights, N):
    """Systematic resampling.
    
    Parameters
    ----------
    weights : ndarray, shape (N,) — normalized weights
    N : int — number of particles to resample
    
    Returns
    -------
    indices : ndarray, shape (N,) — indices of the selected particles
    """
    cumsum = np.cumsum(weights)
    u0 = np.random.uniform(0, 1.0 / N)
    u = u0 + np.arange(N) / N
    indices = np.searchsorted(cumsum, u)
    return indices


def bootstrap_particle_filter(f, h, Q, R, z_seq, N=1000, x0_sampler=None):
    """Bootstrap Particle Filter (SIR: Sampling Importance Resampling).
    
    proposal = transition prior (the most basic PF)
    
    Parameters
    ----------
    f : callable — state transition function f(x, noise) -> x_next
    h : callable — observation function h(x) -> z_predicted
    Q : ndarray — process noise covariance
    R : ndarray — observation noise covariance
    z_seq : list of ndarray — observation sequence
    N : int — number of particles
    x0_sampler : callable — initial particle sampler (defaults to N(0, I))
    
    Returns
    -------
    x_est : list of ndarray — weighted mean estimate at each time step
    """
    n = Q.shape[0]
    m = R.shape[0]
    T = len(z_seq)
    
    # initialization
    if x0_sampler:
        particles = np.array([x0_sampler() for _ in range(N)])  # (N, n)
    else:
        particles = np.random.randn(N, n)
    weights = np.ones(N) / N
    
    x_est = []
    L_Q = np.linalg.cholesky(Q)
    
    for k in range(T):
        # 1. prediction: propagate particles through the transition model
        noise = (L_Q @ np.random.randn(n, N)).T  # (N, n)
        particles = np.array([f(particles[i], noise[i]) for i in range(N)])
        
        # 2. weight update: observation likelihood
        for i in range(N):
            z_pred = h(particles[i])
            innovation = z_seq[k] - z_pred
            # Gaussian likelihood
            log_w = -0.5 * innovation @ np.linalg.solve(R, innovation)
            weights[i] *= np.exp(log_w)
        
        # normalize
        weights /= np.sum(weights)
        
        # weighted-mean estimate
        x_est.append(np.average(particles, weights=weights, axis=0))
        
        # 3. resampling (if the effective sample size is below the threshold)
        N_eff = 1.0 / np.sum(weights ** 2)
        if N_eff < N / 2:
            indices = systematic_resampling(weights, N)
            particles = particles[indices]
            weights = np.ones(N) / N
    
    return x_est
```

### 4.3.4 Rao-Blackwellized Particle Filter (RBPF)

The RBPF splits the state space into two parts, estimating one with particles and the other analytically (e.g., with a Kalman filter). This reduces the dimensionality the particle filter must cover, yielding a good approximation with far fewer particles.

Let the state be partitioned as $\mathbf{x} = [\mathbf{x}_1, \mathbf{x}_2]$. If, given $\mathbf{x}_1$, the conditional distribution of $\mathbf{x}_2$ is analytically tractable (e.g., Gaussian):

$$p(\mathbf{x}_1, \mathbf{x}_2 \mid \mathbf{z}_{1:k}) = p(\mathbf{x}_2 \mid \mathbf{x}_1, \mathbf{z}_{1:k}) \cdot p(\mathbf{x}_1 \mid \mathbf{z}_{1:k})$$

- $p(\mathbf{x}_1 \mid \mathbf{z}_{1:k})$: approximated by a particle filter
- $p(\mathbf{x}_2 \mid \mathbf{x}_1, \mathbf{z}_{1:k})$: tracked by a Kalman filter attached to each particle

**Rao-Blackwell theorem**: the variance of the estimator under this partition is always less than or equal to the variance of the pure particle filter.

$$\text{Var}[\hat{\mathbf{x}}_{\text{RBPF}}] \leq \text{Var}[\hat{\mathbf{x}}_{\text{PF}}]$$

**Connection to FastSLAM**: FastSLAM is a representative application of the RBPF. It represents the robot trajectory with particles and tracks landmark positions with individual EKFs attached to each particle.

- $\mathbf{x}_1 = \mathbf{x}_{0:k}^{\text{robot}}$ (robot trajectory) → particle filter
- $\mathbf{x}_2 = \{\mathbf{m}_1, \ldots, \mathbf{m}_M\}$ (landmarks) → $M$ independent 2D EKFs per particle

Given the robot trajectory, observations of each landmark become mutually independent (conditional independence), so $M$ small EKFs can be run independently instead of one giant EKF. This is the key insight that reduces the $O(M^2)$ complexity of EKF-SLAM to the $O(M \log M)$ of FastSLAM.

### 4.3.5 Limitations of the Particle Filter and Its Current Role

The greatest limitation of the PF is the **curse of dimensionality**. As the state-space dimension grows, the number of particles needed for a meaningful approximation grows exponentially. A typical VIO/LIO state vector is 15-dimensional or more, so a pure PF is impractical.

For this reason, the PF plays a limited role in modern robotic systems:

- **2D SLAM (RBPF-based)**: still used in 2D occupancy-grid SLAM such as GMapping. Only the robot pose (3-DoF) is represented by particles, with the map represented by a grid attached to each particle.
- **Global localization (MCL)**: when the robot's initial position is unknown in an existing map (the kidnapped-robot problem). The PF's ability to represent multimodal distributions naturally makes it a good fit.
- **Low-dimensional nonlinear estimation**: specialized problems with a low-dimensional state and strong nonlinearity.

High-dimensional state estimation is dominated by the Kalman filter family (especially the ESKF) and factor-graph-based optimization.

---

## 4.4 Smoothing vs Filtering

### 4.4.1 The Difference Between Filtering and Smoothing

**Filtering**: uses observations up to the present to estimate the current state.
$$p(\mathbf{x}_k \mid \mathbf{z}_{1:k})$$

**Smoothing**: uses all observations (including future ones) to estimate past states.
$$p(\mathbf{x}_k \mid \mathbf{z}_{1:T}), \quad k < T$$

A smoother leverages "future observations," so its estimate at the same time is always at least as accurate as the filter's. However, real-time estimation requires a filter; smoothers are used in batch (post-processing) or fixed-lag form.

### 4.4.2 Fixed-Lag Smoother

A fixed-lag smoother uses observations up to the current time $k$ to estimate the state at $k-L$, $L$ steps in the past:

$$p(\mathbf{x}_{k-L} \mid \mathbf{z}_{1:k})$$

This is a compromise between filtering and full smoothing. Allowing a latency of $L$ yields a better estimate.

The sliding-window optimization systems of VINS-Mono, ORB-SLAM3, and so on are effectively fixed-lag smoothers. Keyframes within the window are optimized jointly, which is more accurate than simple filtering.

### 4.4.3 Full Smoothing (Batch Optimization)

Estimate all states of the entire trajectory simultaneously using all observations:

$$p(\mathbf{x}_{0:T} \mid \mathbf{z}_{1:T})$$

Solving this as MAP (Maximum A Posteriori) estimation gives:

$$\mathbf{x}_{0:T}^* = \arg\max_{\mathbf{x}_{0:T}} p(\mathbf{x}_{0:T} \mid \mathbf{z}_{1:T})$$

Under Gaussian noise assumptions, MAP becomes a Nonlinear Least Squares (NLS) problem. This is the starting point of factor-graph-based optimization.

### 4.4.4 Why Modern SLAM Moved From Filtering To Optimization

Until the early 2000s, EKF-SLAM was the mainstream of SLAM. Gradually, the field shifted to graph-based optimization (= batch smoothing). The reasons:

**1. The linearization-point problem**

The EKF is "linearize once, and you're done." The Jacobian at time $k$ is computed at the time-$k$ estimate, and if a better estimate is later obtained, the past Jacobians are not revised. In contrast, batch optimization can iteratively relinearize the Jacobians of the entire trajectory at the current estimate.

[Strasdat et al. (2012) "Visual SLAM: Why Filter?"](https://doi.org/10.1016/j.imavis.2012.02.009) presented this argument systematically: given the same computational budget, adding more keyframes to optimization yields higher accuracy than adding more observations to filtering.

**2. Consistency issues**

EKF-SLAM tends to violate observability conditions. In SLAM, the first pose should be unobservable (and therefore fixed), but the EKF's linearization breaks this, making four unobservable directions observable and introducing inconsistency. The First-Estimate Jacobian (FEJ) alleviates this problem but does not fully resolve it.

**3. Scalability**

In EKF-SLAM with $M$ landmarks, the covariance matrix has size $O((n + 3M)^2)$ and each update has cost $O((n+3M)^2)$. This becomes intractable as the map grows.

In graph-based SLAM the information matrix (Hessian) is **sparse**. Each variable (pose, landmark) is connected only to the variables it directly observed, so most of the information matrix is zero. Exploiting this sparsity allows optimization in time near-linear in the number of variables.

**4. Natural handling of loop closures**

In filter-based systems, handling a loop closure requires retaining the covariance information for past states, which dramatically increases the cost. In graph-based systems, a loop closure is simply a new factor (constraint) to add, and the whole graph is re-optimized.

| Aspect | Filtering (EKF) | Optimization (Graph) |
|------|----------------|---------------------|
| Linearization | Once, fixed | Can be repeatedly relinearized |
| Past states | Marginalized out | All retained |
| Loop closure | Difficult and costly | Natural via adding a factor |
| Information-matrix structure | Dense | Sparse |
| Cost per step ($M$ landmarks) | $O(M^2)$ | $O(M)$ (exploiting sparsity) |
| Consistency | Requires FEJ and the like | Naturally mitigated by relinearization |

That said, filter-based approaches have not disappeared entirely. [MSCKF (Mourikis & Roumeliotis, 2007)](https://ieeexplore.ieee.org/document/4209642) and OpenVINS are EKF-based yet exhibit competitive performance, and remain useful in environments with extremely limited compute (e.g., micro UAVs). FAST-LIO2's IEKF is also filter-based, yet combined with ikd-tree it achieves accuracy on par with optimization-based systems.

---

## 4.5 Factor Graph & Optimization

### 4.5.1 Factor Graph Representation

A factor graph is a kind of probabilistic graphical model — a bipartite graph consisting of variables and factors.

$$p(\mathbf{X} \mid \mathbf{Z}) \propto \prod_{i} f_i(\mathbf{X}_i)$$

Here:
- $\mathbf{X} = \{\mathbf{x}_0, \mathbf{x}_1, \ldots, \mathbf{x}_T, \mathbf{l}_1, \ldots, \mathbf{l}_M\}$: variable nodes (poses, landmarks, biases, etc.)
- $f_i(\mathbf{X}_i)$: the $i$-th factor. An "energy function" or "probabilistic constraint" on the subset $\mathbf{X}_i$ of variables
- $\mathbf{Z}$: all observations

Each factor corresponds to a specific observation or piece of prior information:

| Factor type | Connected variables | Meaning |
|------------|-------------|------|
| Prior factor | $\mathbf{x}_0$ | Prior on the initial state |
| Odometry factor | $\mathbf{x}_{k-1}, \mathbf{x}_k$ | Relative motion between consecutive poses |
| IMU preintegration factor | $\mathbf{x}_{i}, \mathbf{x}_{j}, \mathbf{v}_i, \mathbf{v}_j, \mathbf{b}_i$ | IMU integration between keyframes |
| Vision factor | $\mathbf{x}_k, \mathbf{l}_m$ | Camera observation of a landmark |
| LiDAR factor | $\mathbf{x}_k$ | Point-to-plane/point-to-edge registration |
| GPS factor | $\mathbf{x}_k$ | Absolute-position observation |
| Loop closure factor | $\mathbf{x}_i, \mathbf{x}_j$ | Loop-closure relative pose |

The key strength of factor graphs is **modularity**. To add a new sensor, one defines the corresponding factor and adds it to the graph — existing factors need not change.

### 4.5.2 MAP Inference = Nonlinear Least Squares

Under a Gaussian noise model, each factor takes the form:

$$f_i(\mathbf{X}_i) \propto \exp\left(-\frac{1}{2} \|\mathbf{r}_i(\mathbf{X}_i)\|^2_{\boldsymbol{\Sigma}_i}\right)$$

Here $\mathbf{r}_i(\mathbf{X}_i)$ is the residual and $\|\mathbf{r}\|^2_{\boldsymbol{\Sigma}} = \mathbf{r}^\top \boldsymbol{\Sigma}^{-1} \mathbf{r}$ is the squared Mahalanobis distance.

For an observation factor, for example:
$$\mathbf{r}_i = \mathbf{z}_i - h_i(\mathbf{X}_i), \quad \boldsymbol{\Sigma}_i = \mathbf{R}_i \text{ (observation noise covariance)}$$

MAP estimate of the full posterior:

$$\mathbf{X}^* = \arg\max_\mathbf{X} p(\mathbf{X} \mid \mathbf{Z}) = \arg\max_\mathbf{X} \prod_i f_i(\mathbf{X}_i)$$

Taking logarithms and flipping signs:

$$\boxed{\mathbf{X}^* = \arg\min_\mathbf{X} \sum_i \|\mathbf{r}_i(\mathbf{X}_i)\|^2_{\boldsymbol{\Sigma}_i}}$$

This is the **Nonlinear Least Squares (NLS)** problem. The probabilistic inference problem has been converted into an optimization problem. The 139-page tutorial [Dellaert & Kaess (2017) "Factor Graphs for Robot Perception"](https://doi.org/10.1561/2300000043) systematically explains this process.

### 4.5.3 Gauss-Newton Method

To solve the NLS problem we use the Gauss-Newton (GN) method. Expand the residuals to first order about the current estimate $\mathbf{X}^{(0)}$:

$$\mathbf{r}_i(\mathbf{X}^{(0)} \boxplus \Delta\mathbf{X}) \approx \mathbf{r}_i(\mathbf{X}^{(0)}) + \mathbf{J}_i \Delta\mathbf{X}$$

Here $\mathbf{J}_i = \frac{\partial \mathbf{r}_i}{\partial \mathbf{X}}\big|_{\mathbf{X}^{(0)}}$ is the residual Jacobian, and $\boxplus$ is the increment operation on the manifold.

Substituting:

$$\sum_i \|\mathbf{r}_i + \mathbf{J}_i \Delta\mathbf{X}\|^2_{\boldsymbol{\Sigma}_i}$$

Whitening via $\mathbf{r}_i' = \boldsymbol{\Sigma}_i^{-1/2} \mathbf{r}_i$, $\mathbf{J}_i' = \boldsymbol{\Sigma}_i^{-1/2} \mathbf{J}_i$ yields a standard least-squares problem:

$$\sum_i \|\mathbf{r}_i' + \mathbf{J}_i' \Delta\mathbf{X}\|^2$$

Differentiating with respect to $\Delta\mathbf{X}$ and setting to zero yields the **normal equation**:

$$\underbrace{\left(\sum_i \mathbf{J}_i'^\top \mathbf{J}_i'\right)}_{\mathbf{H}} \Delta\mathbf{X} = -\underbrace{\sum_i \mathbf{J}_i'^\top \mathbf{r}_i'}_{\mathbf{b}}$$

$$\boxed{\mathbf{H} \Delta\mathbf{X} = -\mathbf{b}}$$

Here $\mathbf{H} = \mathbf{J}^\top \boldsymbol{\Sigma}^{-1} \mathbf{J} \in \mathbb{R}^{N \times N}$ is the approximate Hessian (information matrix), and $\mathbf{b} = \mathbf{J}^\top \boldsymbol{\Sigma}^{-1} \mathbf{r}$ is the gradient.

In SLAM problems $\mathbf{H}$ is **sparse**. Each factor's Jacobian $\mathbf{J}_i$ has nonzero entries only in the columns corresponding to the variables it connects, and zero elsewhere. Thus the nonzero entries of $\mathbf{H}$ correspond to edges in the factor graph: when the graph is sparse, $\mathbf{H}$ is sparse.

Gauss-Newton iteration:

$$\mathbf{X}^{(k+1)} = \mathbf{X}^{(k)} \boxplus \Delta\mathbf{X}^{(k)}$$

Each iteration solves a normal equation. Sparse linear systems are solved with **sparse Cholesky decomposition** ($\mathbf{H} = \mathbf{L}\mathbf{L}^\top$, forward/back substitution); since the fill-in of $\mathbf{L}$ depends on the variable ordering, an approximate minimum-degree ordering such as COLAMD is used.

### 4.5.4 Levenberg-Marquardt Method

Gauss-Newton is a purely approximate second-order method, but with a bad initial estimate or strong nonlinearity it may diverge. The **Levenberg-Marquardt (LM)** method is a compromise between GN and gradient descent, adding a regularization term:

$$(\mathbf{H} + \lambda \mathbf{I}) \Delta\mathbf{X} = -\mathbf{b}$$

- Small $\lambda$ → closer to GN (fast, quadratic convergence)
- Large $\lambda$ → closer to gradient descent (small step, safe)

Strategy for $\lambda$: if the update decreases the cost, decrease $\lambda$ (GN mode); if it increases the cost, increase $\lambda$ (conservative mode).

### 4.5.5 Optimization on Manifolds

When optimizing a 3D pose $\mathbf{T} \in SE(3)$, $SE(3)$ is a manifold, not Euclidean space, so we cannot use ordinary addition. The standard way to handle this is to use a **retraction** (or **exponential map**).

Near the current estimate $\mathbf{T}^{(k)}$, define the increment $\boldsymbol{\xi} \in \mathbb{R}^6$ in the tangent space and set:

$$\mathbf{T}^{(k+1)} = \mathbf{T}^{(k)} \cdot \text{Exp}(\boldsymbol{\xi})$$

or:

$$\mathbf{T}^{(k+1)} = \text{Exp}(\boldsymbol{\xi}) \cdot \mathbf{T}^{(k)}$$

(the choice of left/right increment depends on convention)

Here $\text{Exp}: \mathbb{R}^6 \to SE(3)$ is the Lie group exponential map. In $\boldsymbol{\xi} = [\boldsymbol{\rho}^\top, \boldsymbol{\phi}^\top]^\top$, $\boldsymbol{\rho} \in \mathbb{R}^3$ is translation and $\boldsymbol{\phi} \in \mathbb{R}^3$ is rotation (angle-axis).

Exponential map on SO(3) (Rodrigues' formula):

$$\text{Exp}(\boldsymbol{\phi}) = \mathbf{I} + \frac{\sin\theta}{\theta}[\boldsymbol{\phi}]_\times + \frac{1 - \cos\theta}{\theta^2}[\boldsymbol{\phi}]_\times^2 \in \mathbb{R}^{3 \times 3}$$

where $\theta = \|\boldsymbol{\phi}\|$.

Conversely, $\text{Log}: SE(3) \to \mathbb{R}^6$ is the logarithmic map.

The Gauss-Newton/LM normal equation is solved for the tangent-space increment $\boldsymbol{\xi}$, and the manifold state is updated via the exponential map. Jacobians are computed with respect to the tangent space:

$$\mathbf{J}_i = \frac{\partial \mathbf{r}_i}{\partial \boldsymbol{\xi}}\bigg|_{\boldsymbol{\xi}=\mathbf{0}}$$

### 4.5.6 iSAM2: Incremental Smoothing

Running batch optimization from scratch at every keyframe takes unrealistically long. **iSAM2** ([Kaess et al., 2012](https://doi.org/10.1177/0278364911430419)) uses the Bayes tree data structure to perform optimization incrementally.

Core ideas:

1. **Bayes tree**: a directed-tree representation of the elimination result of a factor graph. Each node stores the conditional density over a clique (subset of variables).

2. **Incremental update**: when a new factor is added, only the affected cliques are recomputed. Most of the tree is unchanged.

3. **Fluid relinearization**: only variables whose linearization point has drifted significantly from the current estimate are selectively relinearized. No periodic batch pass is needed.

4. **Variable reordering**: when new variables/factors are added, no full reordering is performed; only the affected portion is locally reordered.

iSAM2 is the core algorithm of the GTSAM library, serving as the backend of many modern SLAM systems such as LIO-SAM and VINS-Mono.

> **Recent trends — continuous-time factor graph**: there is active research extending discrete keyframe-based factor graphs to **continuous time**. [Wong et al. (2024)](https://arxiv.org/abs/2402.06174) use a Gaussian Process motion prior to unify radar-inertial and LiDAR-inertial odometry in a continuous-time factor graph, and show that asynchronous sensor measurements can be handled naturally.

### 4.5.7 GTSAM / Ceres / g2o Comparison

| Aspect | GTSAM | Ceres Solver | g2o |
|------|-------|-------------|-----|
| Developer | Georgia Tech ([Dellaert](https://gtsam.org/)) | Google ([Ceres](http://ceres-solver.org/)) | [Kümmerle et al.](https://doi.org/10.1109/ICRA.2011.5979949) |
| Core philosophy | Factor graph + Bayes tree | General-purpose NLS solver | Graph optimization |
| Incremental | iSAM2 (native) | None (batch) | None (batch) |
| Manifolds | Built-in (Rot2, Rot3, Pose2, Pose3, ...) | Local parameterization | Built-in |
| IMU preintegration | Built-in (`PreintegratedImuMeasurements`) | User-defined | User-defined |
| Automatic differentiation | Numerical differentiation available | Auto-diff (ceres::AutoDiffCostFunction) | None |
| Language | C++ (Python bindings) | C++ | C++ |
| Representative users | LIO-SAM, VINS-Mono | Cartographer, ORB-SLAM3 BA | Pose graphs in many SLAM systems |
| Learning curve | Just define factors | Define cost functions | Define vertices/edges |

```python
# Simple pose graph optimization example using GTSAM
import gtsam
import numpy as np

# 1. create factor graph
graph = gtsam.NonlinearFactorGraph()

# 2. initial estimates
initial = gtsam.Values()

# 3. prior factor on the first pose
prior_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.3, 0.3, 0.1]))  # (x, y, theta)
graph.add(gtsam.PriorFactorPose2(0, gtsam.Pose2(0.0, 0.0, 0.0), prior_noise))
initial.insert(0, gtsam.Pose2(0.0, 0.0, 0.0))

# 4. odometry factors (between consecutive poses)
odom_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.2, 0.2, 0.1]))

# square trajectory: 4 poses (forward 2 m + left turn 90 deg on each side)
odometry = [
    gtsam.Pose2(2.0, 0.0, np.pi / 2),   # x0 -> x1: forward 2 m + left turn 90 deg
    gtsam.Pose2(2.0, 0.0, np.pi / 2),   # x1 -> x2: forward 2 m + left turn 90 deg
    gtsam.Pose2(2.0, 0.0, np.pi / 2),   # x2 -> x3: forward 2 m + left turn 90 deg
    gtsam.Pose2(2.0, 0.0, np.pi / 2),   # x3 -> x4: forward 2 m + left turn 90 deg
]

# add odometry factors and set initial values
pose = gtsam.Pose2(0.0, 0.0, 0.0)
for i, odom in enumerate(odometry):
    graph.add(gtsam.BetweenFactorPose2(i, i + 1, odom, odom_noise))
    pose = pose.compose(odom)
    # add a little noise to the initial guess (in reality this is the accumulated odometry)
    initial.insert(i + 1, pose)

# 5. loop-closure factor: x4 and x0 are at the same position (the square trajectory closes)
loop_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.1, 0.1, 0.05]))
graph.add(gtsam.BetweenFactorPose2(4, 0, gtsam.Pose2(0.0, 0.0, 0.0), loop_noise))

# 6. optimize with iSAM2
params = gtsam.ISAM2Params()
isam = gtsam.ISAM2(params)
isam.update(graph, initial)
result = isam.calculateEstimate()

# print results
for i in range(5):
    pose_i = result.atPose2(i)
    print(f"Pose {i}: x={pose_i.x():.3f}, y={pose_i.y():.3f}, theta={pose_i.theta():.3f}")
```

---

## 4.6 IMU Preintegration

### 4.6.1 Why Preintegration Is Needed

An IMU usually measures acceleration and angular velocity at 200 Hz to 1000 Hz, while camera/LiDAR keyframes arrive at 10 Hz to 30 Hz intervals. Hundreds of IMU measurements occur between two keyframes $i, j$.

**Naive approach: direct integration**

Integrating the IMU measurements between two keyframes to obtain the relative pose gives:

$$\mathbf{R}_j = \mathbf{R}_i \prod_{k=i}^{j-1} \text{Exp}((\tilde{\boldsymbol{\omega}}_k - \mathbf{b}_g) \Delta t)$$
$$\mathbf{v}_j = \mathbf{v}_i + \mathbf{g} \Delta t_{ij} + \sum_{k=i}^{j-1} \mathbf{R}_k (\tilde{\mathbf{a}}_k - \mathbf{b}_a) \Delta t$$
$$\mathbf{p}_j = \mathbf{p}_i + \mathbf{v}_i \Delta t_{ij} + \frac{1}{2}\mathbf{g}\Delta t_{ij}^2 + \sum_{k=i}^{j-1}\left[\mathbf{v}_k \Delta t + \frac{1}{2}\mathbf{R}_k(\tilde{\mathbf{a}}_k - \mathbf{b}_a)\Delta t^2\right]$$

The problem: this integration depends on the state $(\mathbf{R}_i, \mathbf{v}_i, \mathbf{p}_i)$ at keyframe $i$ and on the biases $(\mathbf{b}_g, \mathbf{b}_a)$. In the optimization loop, whenever the estimates of $\mathbf{R}_i, \mathbf{v}_i, \mathbf{p}_i$ change, all intermediate states must be re-integrated. The same is true when the bias estimates change. This means hundreds of exponential-map evaluations per optimization iteration.

**Preintegration's remedy**: define the relative change in the body frame of keyframe $i$, so that this quantity is independent of keyframe $i$'s global pose. For biases, first-order Jacobian correction avoids re-integration.

### 4.6.2 On-Manifold Preintegration Derivation

We derive the on-manifold preintegration of [Forster et al. (2017)](https://doi.org/10.1109/TRO.2016.2597321) step by step. The original paper is also available on [arXiv:1512.02363](https://arxiv.org/abs/1512.02363).

#### Step 1: Define relative quantities

Rearrange the integration equations in the global frame so they are expressed in the body frame of keyframe $i$. Move the global variables ($\mathbf{R}_i, \mathbf{v}_i, \mathbf{p}_i$) to the left-hand side:

$$\Delta\mathbf{R}_{ij} \triangleq \mathbf{R}_i^\top \mathbf{R}_j = \prod_{k=i}^{j-1} \text{Exp}((\tilde{\boldsymbol{\omega}}_k - \mathbf{b}_g^i)\Delta t) \in SO(3)$$

$$\Delta\mathbf{v}_{ij} \triangleq \mathbf{R}_i^\top (\mathbf{v}_j - \mathbf{v}_i - \mathbf{g}\Delta t_{ij}) = \sum_{k=i}^{j-1} \Delta\mathbf{R}_{ik}(\tilde{\mathbf{a}}_k - \mathbf{b}_a^i)\Delta t \in \mathbb{R}^3$$

$$\Delta\mathbf{p}_{ij} \triangleq \mathbf{R}_i^\top (\mathbf{p}_j - \mathbf{p}_i - \mathbf{v}_i \Delta t_{ij} - \frac{1}{2}\mathbf{g}\Delta t_{ij}^2) = \sum_{k=i}^{j-1}\left[\Delta\mathbf{v}_{ik}\Delta t + \frac{1}{2}\Delta\mathbf{R}_{ik}(\tilde{\mathbf{a}}_k - \mathbf{b}_a^i)\Delta t^2\right] \in \mathbb{R}^3$$

Key observation: **the right-hand sides depend only on the IMU measurements and the bias estimates, and are independent of the global pose $(\mathbf{R}_i, \mathbf{v}_i, \mathbf{p}_i)$ at keyframe $i$.** Therefore, when the keyframe pose changes during optimization, the right-hand sides do not need to be recomputed.

#### Step 2: Recursive computation (on-manifold)

The preintegrated measurements can be accumulated recursively:

$$\Delta\mathbf{R}_{i,k+1} = \Delta\mathbf{R}_{ik} \cdot \text{Exp}((\tilde{\boldsymbol{\omega}}_k - \mathbf{b}_g)\Delta t) \in SO(3)$$
$$\Delta\mathbf{v}_{i,k+1} = \Delta\mathbf{v}_{ik} + \Delta\mathbf{R}_{ik}(\tilde{\mathbf{a}}_k - \mathbf{b}_a)\Delta t \in \mathbb{R}^3$$
$$\Delta\mathbf{p}_{i,k+1} = \Delta\mathbf{p}_{ik} + \Delta\mathbf{v}_{ik}\Delta t + \frac{1}{2}\Delta\mathbf{R}_{ik}(\tilde{\mathbf{a}}_k - \mathbf{b}_a)\Delta t^2 \in \mathbb{R}^3$$

Initial values: $\Delta\mathbf{R}_{ii} = \mathbf{I}_{3\times 3}$, $\Delta\mathbf{v}_{ii} = \mathbf{0}$, $\Delta\mathbf{p}_{ii} = \mathbf{0}$.

The meaning of "on-manifold": the rotation $\Delta\mathbf{R}_{ij}$ is accumulated directly on $SO(3)$. Makeshift workarounds such as Euler angles or quaternion re-normalization are not needed.

#### Step 3: First-order correction for bias change

During optimization, the bias estimate changes as $\mathbf{b} \to \mathbf{b} + \delta\mathbf{b}$, and the preintegrated measurements change accordingly. To avoid full re-integration, we apply a first-order Taylor correction:

**Rotation correction**:
$$\Delta\hat{\mathbf{R}}_{ij}(\mathbf{b}_g + \delta\mathbf{b}_g) \approx \Delta\hat{\mathbf{R}}_{ij}(\mathbf{b}_g) \cdot \text{Exp}\left(\frac{\partial \Delta\bar{\mathbf{R}}_{ij}}{\partial \mathbf{b}_g} \delta\mathbf{b}_g\right)$$

**Velocity correction**:
$$\Delta\hat{\mathbf{v}}_{ij}(\mathbf{b} + \delta\mathbf{b}) \approx \Delta\hat{\mathbf{v}}_{ij}(\mathbf{b}) + \frac{\partial \Delta\bar{\mathbf{v}}_{ij}}{\partial \mathbf{b}_g} \delta\mathbf{b}_g + \frac{\partial \Delta\bar{\mathbf{v}}_{ij}}{\partial \mathbf{b}_a} \delta\mathbf{b}_a$$

**Position correction**:
$$\Delta\hat{\mathbf{p}}_{ij}(\mathbf{b} + \delta\mathbf{b}) \approx \Delta\hat{\mathbf{p}}_{ij}(\mathbf{b}) + \frac{\partial \Delta\bar{\mathbf{p}}_{ij}}{\partial \mathbf{b}_g} \delta\mathbf{b}_g + \frac{\partial \Delta\bar{\mathbf{p}}_{ij}}{\partial \mathbf{b}_a} \delta\mathbf{b}_a$$

The Jacobians $\frac{\partial \Delta\bar{\mathbf{R}}_{ij}}{\partial \mathbf{b}_g}$ and the like are accumulated recursively during preintegration. For example, the bias Jacobian of the rotation:

$$\frac{\partial \Delta\bar{\mathbf{R}}_{i,k+1}}{\partial \mathbf{b}_g} = -\Delta\bar{\mathbf{R}}_{k,k+1}^\top \text{Jr}((\tilde{\boldsymbol{\omega}}_k - \mathbf{b}_g)\Delta t) \Delta t + \Delta\bar{\mathbf{R}}_{k,k+1}^\top \frac{\partial \Delta\bar{\mathbf{R}}_{ik}}{\partial \mathbf{b}_g}$$

where $\text{Jr}(\boldsymbol{\phi})$ is the right Jacobian of SO(3):

$$\text{Jr}(\boldsymbol{\phi}) = \mathbf{I} - \frac{1 - \cos\theta}{\theta^2}[\boldsymbol{\phi}]_\times + \frac{\theta - \sin\theta}{\theta^3}[\boldsymbol{\phi}]_\times^2, \quad \theta = \|\boldsymbol{\phi}\|$$

When the bias change is small ($\|\delta\mathbf{b}\|$ is small), this first-order correction is sufficiently accurate. When the bias change is large, the preintegration is recomputed from scratch, but this rarely occurs in practice.

#### Step 4: Covariance propagation

The IMU measurement noise $\mathbf{n}_g, \mathbf{n}_a \sim \mathcal{N}(\mathbf{0}, \boldsymbol{\Sigma})$ propagates through the preintegration. The covariance is also computed recursively:

$$\boldsymbol{\Sigma}_{k+1} = \mathbf{A}_k \boldsymbol{\Sigma}_k \mathbf{A}_k^\top + \mathbf{B}_k \boldsymbol{\Sigma}_\eta \mathbf{B}_k^\top$$

Here $\mathbf{A}_k, \mathbf{B}_k$ are the Jacobians of the recursive propagation and $\boldsymbol{\Sigma}_\eta$ is the IMU noise covariance. The covariance $\boldsymbol{\Sigma}_{ij}$ is used as the information matrix $\boldsymbol{\Sigma}_{ij}^{-1}$ of the IMU factor in the factor graph.

#### Step 5: IMU Factor in the Factor Graph

Finally, the preintegrated measurement is inserted as a factor between two keyframes $i, j$. The residual:

$$\mathbf{r}_{\Delta\mathbf{R}_{ij}} = \text{Log}\left(\Delta\hat{\mathbf{R}}_{ij}^\top \mathbf{R}_i^\top \mathbf{R}_j\right) \in \mathbb{R}^3$$

$$\mathbf{r}_{\Delta\mathbf{v}_{ij}} = \mathbf{R}_i^\top(\mathbf{v}_j - \mathbf{v}_i - \mathbf{g}\Delta t_{ij}) - \Delta\hat{\mathbf{v}}_{ij} \in \mathbb{R}^3$$

$$\mathbf{r}_{\Delta\mathbf{p}_{ij}} = \mathbf{R}_i^\top(\mathbf{p}_j - \mathbf{p}_i - \mathbf{v}_i\Delta t_{ij} - \frac{1}{2}\mathbf{g}\Delta t_{ij}^2) - \Delta\hat{\mathbf{p}}_{ij} \in \mathbb{R}^3$$

This 9-dimensional residual is the IMU preintegration factor's residual, and its squared Mahalanobis distance $\|\mathbf{r}\|^2_{\boldsymbol{\Sigma}_{ij}}$ is added to the cost function.

### 4.6.3 Preintegration in Code

```python
import numpy as np
from scipy.spatial.transform import Rotation

def skew(v):
    """3D vector -> skew-symmetric matrix.
    
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
    """so(3) -> SO(3) exponential map (Rodrigues' formula).
    
    Parameters
    ----------
    phi : ndarray, shape (3,) — angle-axis vector
    
    Returns
    -------
    R : ndarray, shape (3, 3) — rotation matrix
    """
    theta = np.linalg.norm(phi)
    if theta < 1e-10:
        return np.eye(3) + skew(phi)
    axis = phi / theta
    K = skew(axis)
    return np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * K @ K


def log_so3(R):
    """SO(3) -> so(3) logarithmic map.
    
    Parameters
    ----------
    R : ndarray, shape (3, 3) — rotation matrix
    
    Returns
    -------
    phi : ndarray, shape (3,) — angle-axis vector
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
    """Right Jacobian of SO(3).
    
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
    
    Preintegrates IMU measurements between two keyframes 
    into a form usable as an IMU factor in a factor graph.
    """
    def __init__(self, bias_gyro, bias_acc, 
                 gyro_noise_density, acc_noise_density,
                 gyro_random_walk, acc_random_walk):
        """
        Parameters
        ----------
        bias_gyro : ndarray, shape (3,) — initial gyro bias estimate [rad/s]
        bias_acc : ndarray, shape (3,) — initial accel bias estimate [m/s^2]
        gyro_noise_density : float — gyro noise density [rad/s/sqrt(Hz)]
        acc_noise_density : float — accel noise density [m/s^2/sqrt(Hz)]
        gyro_random_walk : float — gyro bias random walk [rad/s^2/sqrt(Hz)]
        acc_random_walk : float — accel bias random walk [m/s^3/sqrt(Hz)]
        """
        self.bg = bias_gyro.copy()
        self.ba = bias_acc.copy()
        
        # preintegrated measurements (initial values)
        self.delta_R = np.eye(3)     # relative rotation on SO(3)
        self.delta_v = np.zeros(3)   # relative velocity change
        self.delta_p = np.zeros(3)   # relative position change
        self.dt_sum = 0.0            # total integration time
        
        # covariance (9x9: rotation, velocity, position)
        self.cov = np.zeros((9, 9))
        
        # bias Jacobians (for bias correction)
        self.d_R_d_bg = np.zeros((3, 3))
        self.d_v_d_bg = np.zeros((3, 3))
        self.d_v_d_ba = np.zeros((3, 3))
        self.d_p_d_bg = np.zeros((3, 3))
        self.d_p_d_ba = np.zeros((3, 3))
        
        # noise parameters
        self.sigma_g = gyro_noise_density
        self.sigma_a = acc_noise_density
        self.sigma_bg = gyro_random_walk
        self.sigma_ba = acc_random_walk
        
    def integrate(self, gyro_meas, acc_meas, dt):
        """Add a single IMU measurement to the preintegration.
        
        Parameters
        ----------
        gyro_meas : ndarray, shape (3,) — gyro measurement [rad/s]
        acc_meas : ndarray, shape (3,) — accelerometer measurement [m/s^2]
        dt : float — time step [s]
        """
        # bias-corrected measurements
        omega = gyro_meas - self.bg   # (3,)
        acc = acc_meas - self.ba       # (3,)
        
        # intermediate quantities
        dR = exp_so3(omega * dt)       # (3, 3)  rotation over this dt
        Jr = right_jacobian_so3(omega * dt)  # (3, 3)
        
        # --- recursive update of bias Jacobians (Step 3) ---
        # position Jacobian (bias Jacobian of delta_p)
        self.d_p_d_bg += self.d_v_d_bg * dt - 0.5 * self.delta_R @ skew(acc) @ self.d_R_d_bg * dt**2
        self.d_p_d_ba += self.d_v_d_ba * dt - 0.5 * self.delta_R * dt**2
        # velocity Jacobian
        self.d_v_d_bg += -self.delta_R @ skew(acc) @ self.d_R_d_bg * dt
        self.d_v_d_ba += -self.delta_R * dt
        # rotation Jacobian
        self.d_R_d_bg = dR.T @ (self.d_R_d_bg - Jr * dt)
        
        # --- covariance propagation (Step 4) ---
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
        
        # --- recursive update of preintegrated measurements (Step 2) ---
        # order matters: first p, then v, finally R
        self.delta_p += self.delta_v * dt + 0.5 * self.delta_R @ acc * dt**2
        self.delta_v += self.delta_R @ acc * dt
        self.delta_R = self.delta_R @ dR
        self.dt_sum += dt
    
    def predict(self, R_i, v_i, p_i, gravity):
        """Predict the state at keyframe j from the preintegrated measurements.
        
        Parameters
        ----------
        R_i : ndarray, shape (3, 3) — rotation at keyframe i
        v_i : ndarray, shape (3,) — velocity at keyframe i
        p_i : ndarray, shape (3,) — position at keyframe i
        gravity : ndarray, shape (3,) — gravity vector (e.g., [0, 0, -9.81])
        
        Returns
        -------
        R_j, v_j, p_j : predicted state at keyframe j
        """
        dt = self.dt_sum
        R_j = R_i @ self.delta_R
        v_j = v_i + gravity * dt + R_i @ self.delta_v
        p_j = p_i + v_i * dt + 0.5 * gravity * dt**2 + R_i @ self.delta_p
        return R_j, v_j, p_j
    
    def compute_residual(self, R_i, v_i, p_i, R_j, v_j, p_j, gravity):
        """Compute the IMU factor residual.
        
        Returns
        -------
        residual : ndarray, shape (9,) — [r_R(3), r_v(3), r_p(3)]
        """
        dt = self.dt_sum
        
        # rotation residual
        r_R = log_so3(self.delta_R.T @ R_i.T @ R_j)
        
        # velocity residual
        r_v = R_i.T @ (v_j - v_i - gravity * dt) - self.delta_v
        
        # position residual
        r_p = R_i.T @ (p_j - p_i - v_i * dt - 0.5 * gravity * dt**2) - self.delta_p
        
        return np.concatenate([r_R, r_v, r_p])


# usage example with GTSAM
import gtsam

def create_imu_factor_gtsam():
    """Example using GTSAM's built-in IMU preintegration."""
    # IMU parameters
    imu_params = gtsam.PreintegrationParams.MakeSharedU(9.81)  # gravity direction: +z
    imu_params.setAccelerometerCovariance(np.eye(3) * 0.01**2)
    imu_params.setGyroscopeCovariance(np.eye(3) * 0.001**2)
    imu_params.setIntegrationCovariance(np.eye(3) * 1e-8)
    
    # initial bias
    bias = gtsam.imuBias.ConstantBias(
        np.array([0.1, -0.05, 0.02]),   # accelerometer bias
        np.array([0.001, -0.002, 0.003]) # gyroscope bias
    )
    
    # create preintegration object
    pim = gtsam.PreintegratedImuMeasurements(imu_params, bias)
    
    # integrate IMU measurements (assume 200 Hz, 0.1 s = 20 measurements)
    dt = 0.005  # 200 Hz
    for k in range(20):
        acc_meas = np.array([0.0, 0.0, 9.81])  # static (gravity only)
        gyro_meas = np.array([0.0, 0.0, 0.0])
        pim.integrateMeasurement(acc_meas, gyro_meas, dt)
    
    # create factor
    # CombinedImuFactor connects pose, velocity, and bias at keyframes i, j
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

### 4.7.1 Why Marginalization Is Needed

Factor-graph-based SLAM/VIO systems add variables and factors whenever a new keyframe arrives. Keeping keyframes indefinitely causes the optimization cost to grow without bound.

The sliding-window approach retains only the most recent $N$ keyframes and removes older ones. However, simple deletion loses all the observation information associated with those keyframes and degrades accuracy.

**Marginalization** is the method for removing old variables from the graph while preserving the information they carried between the variables they connected. The information of the eliminated variable is converted into a **prior** on the remaining variables.

### 4.7.2 Schur Complement

The mathematical core of marginalization is the **Schur complement**. Partition the information matrix (Hessian) $\mathbf{H}$ into variables to marginalize ($\mathbf{x}_m$) and variables to retain ($\mathbf{x}_r$). Write the normal equation as $\mathbf{H}\Delta\mathbf{x} = \mathbf{b}$ (redefining the $-\mathbf{b}$ of §4.5.3 as $\mathbf{b}$, i.e., $\mathbf{b} \triangleq -\mathbf{J}^\top \boldsymbol{\Sigma}^{-1} \mathbf{r}$):

$$\mathbf{H} \Delta\mathbf{x} = \mathbf{b}$$

$$\begin{bmatrix} \mathbf{H}_{mm} & \mathbf{H}_{mr} \\ \mathbf{H}_{rm} & \mathbf{H}_{rr} \end{bmatrix} \begin{bmatrix} \Delta\mathbf{x}_m \\ \Delta\mathbf{x}_r \end{bmatrix} = \begin{bmatrix} \mathbf{b}_m \\ \mathbf{b}_r \end{bmatrix}$$

Here:
- $\mathbf{x}_m$: variables to marginalize (remove)
- $\mathbf{x}_r$: variables to retain
- $\mathbf{H}_{mm} \in \mathbb{R}^{n_m \times n_m}$: information block among the variables to remove
- $\mathbf{H}_{mr} \in \mathbb{R}^{n_m \times n_r}$: cross-information block
- $\mathbf{H}_{rr} \in \mathbb{R}^{n_r \times n_r}$: information block among the variables to retain

Eliminating $\Delta\mathbf{x}_m$ from the block matrix above yields:

$$\underbrace{(\mathbf{H}_{rr} - \mathbf{H}_{rm} \mathbf{H}_{mm}^{-1} \mathbf{H}_{mr})}_{\mathbf{H}^*} \Delta\mathbf{x}_r = \underbrace{\mathbf{b}_r - \mathbf{H}_{rm} \mathbf{H}_{mm}^{-1} \mathbf{b}_m}_{\mathbf{b}^*}$$

$\mathbf{H}^* = \mathbf{H}_{rr} - \mathbf{H}_{rm} \mathbf{H}_{mm}^{-1} \mathbf{H}_{mr}$ is the **Schur complement**, and it becomes the information matrix of the prior factor on the retained variables after marginalization.

**Intuitive meaning**: variables in $\mathbf{x}_r$ that were indirectly connected through $\mathbf{x}_m$ become directly connected (fill-in). $\mathbf{H}^*$ is denser than $\mathbf{H}_{rr}$, which reflects that variable-to-variable correlations hidden before marginalization now appear explicitly.

```python
import numpy as np

def marginalize(H, b, indices_to_marginalize, indices_to_keep):
    """Variable marginalization via the Schur complement.
    
    Parameters
    ----------
    H : ndarray, shape (N, N) — information matrix (Hessian)
    b : ndarray, shape (N,) — gradient vector
    indices_to_marginalize : list of int — indices of variables to remove
    indices_to_keep : list of int — indices of variables to retain
    
    Returns
    -------
    H_star : ndarray — information matrix after marginalization
    b_star : ndarray — gradient after marginalization
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


# example: 3 poses (each 2D, 3-DoF), marginalize the first one
# state: [x0(3), x1(3), x2(3)] = 9D
np.random.seed(42)

# sparse information matrix (x0-x1, x1-x2 connections)
H = np.zeros((9, 9))
# x0 itself (prior + odom factor)
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

print("nonzero pattern of H before marginalization:")
print((np.abs(H) > 1e-10).astype(int))

# marginalize x0 (indices 0, 1, 2)
H_star, b_star = marginalize(H, b, [0, 1, 2], [3, 4, 5, 6, 7, 8])

print("\nnonzero pattern of H* after marginalization:")
print((np.abs(H_star) > 1e-10).astype(int))
print("-> fill-in may appear between x1 and x2")
```

### 4.7.3 First-Estimate Jacobian (FEJ)

There is an important caveat in marginalization. The marginalized factor (prior) is computed at a **fixed linearization point**. In subsequent optimization iterations, if the retained variables' estimates change, a mismatch arises between the marginalized prior's linearization point and the current one.

This mismatch can hurt the **consistency** of the filter/smoother. Specifically, the system's observability properties can change, so directions that are actually unobservable may be incorrectly treated as observable.

The **FEJ (First-Estimate Jacobian)** strategy: Jacobians related to marginalization are always computed at the **first estimate** of the relevant variable, and are not updated even when the estimate later changes.

$$\mathbf{J}_{\text{FEJ}} = \left.\frac{\partial \mathbf{r}}{\partial \mathbf{x}}\right|_{\mathbf{x}^{(0)}}$$

Here $\mathbf{x}^{(0)}$ is the value of the variable when it was first estimated.

Advantages of FEJ:
- The marginalized prior and the current factors use information from the same linearization point, preserving consistency.
- Used centrally in MSCKF/OpenVINS ([Li & Mourikis, 2013](https://doi.org/10.1177/0278364913481251)).

Disadvantages of FEJ:
- If the first estimate is inaccurate, convergence may slow down.
- Implementation becomes more complex (the "first estimate" of each variable must be stored separately).

### 4.7.4 Practical Issues of Sliding-Window Implementation

#### Issue 1: Which keyframe to marginalize

The two strategies of VINS-Mono:
- **If the latest frame is a keyframe**: marginalize the oldest keyframe. The window remains spatially wide.
- **If the latest frame is not a keyframe**: marginalize the immediately previous non-keyframe. Only the visual information is discarded, while the IMU information is forwarded to the neighboring keyframe.

#### Issue 2: Densification due to fill-in

As marginalization is repeated, the prior factor becomes progressively denser, and what was originally a sparse information matrix can become dense. This severely degrades computational efficiency.

Mitigations:
- Limit the size of the prior factor (limit the number of connected variables)
- Choose the marginalization order carefully
- Accept some loss of information and simply delete certain factors (FAST-LIO2 removes old points from the map rather than marginalizing)

#### Issue 3: Biases and marginalization

The IMU bias is a slowly varying state that spans all keyframes. Marginalizing the bias together with a keyframe fixes the bias information in the prior, reducing the flexibility of subsequent bias estimation.

VINS-Mono's approach: the bias is not marginalized, but kept within the window. The marginalization prior is formed conditional on the bias.

#### Issue 4: Numerical stability

The Schur complement computation requires the inverse $\mathbf{H}_{mm}^{-1}$. If $\mathbf{H}_{mm}$ has a bad condition number, numerical instability can arise.

Mitigations:
- Add a small regularization term to $\mathbf{H}_{mm}$: $(\mathbf{H}_{mm} + \epsilon \mathbf{I})^{-1}$
- Use LDL decomposition for numerical stability
- After marginalization, check that $\mathbf{H}^*$ is positive semi-definite and, if not, project it onto the nearest PSD matrix

---

## Chapter 4 Summary

In this chapter we covered state estimation theory, the mathematical foundation of sensor fusion. The key messages:

1. The **Bayesian Filtering Framework** is a prediction-update recursion that forms the common skeleton of every state estimation method. The Chapman-Kolmogorov equation and Bayes' rule are the theoretical underpinnings, but in nonlinear systems approximation is essential.

2. The **Kalman Filter family** tracks the posterior via Gaussian approximation in terms of mean and covariance. The EKF linearizes to first order; the ESKF linearizes in the error state, handling manifold issues naturally; the UKF uses the sigma-point transform; and the IEKF uses iterated linearization to cope with strong nonlinearity. In modern robotics systems, the ESKF is effectively the standard.

3. The **particle filter** can handle multimodal distributions and strong nonlinearity, but the curse of dimensionality makes it unsuitable for high-dimensional problems. The RBPF (FastSLAM) partially alleviates this, and the PF is still used in 2D SLAM and global localization.

4. The **shift from filtering to optimization** has become the mainstream of modern SLAM thanks to benefits such as relinearization, exploitation of sparsity, and natural handling of loop closures. That said, filter-based approaches (MSCKF, FAST-LIO2) remain competitive under specific conditions.

5. The **factor graph** is a powerful framework that modularly composes probabilistic inference and reduces MAP to NLS. It is solved on the manifold with Gauss-Newton/LM, and iSAM2's incremental smoothing makes real-time processing possible.

6. **IMU preintegration** is the key technique for compressing high-rate IMU measurements into a factor between keyframes. The on-manifold derivation and first-order bias correction allow integration into the factor graph without re-integration.

7. **Marginalization** is the information-preservation mechanism of the sliding window. The Schur complement is the core operation, and FEJ is the key to maintaining consistency.

This chapter's theory is an indispensable foundation for understanding the design and implementation of real systems in the subsequent VIO (Ch.6), LIO (Ch.7), and multi-sensor fusion (Ch.8) chapters.

> **2024-2025 research directions**: state estimation is evolving along three major axes. (1) **Symmetry-based filters**: the Equivariant Filter (EqF) and the Invariant EKF exploit the symmetry structure of Lie groups to structurally guarantee consistency and convergence. (2) **Continuous-time optimization**: continuous-time factor graphs with Gaussian Process motion priors are becoming a new paradigm for asynchronous multi-sensor fusion. (3) **Learning-based hybrids**: approaches such as [AI-Aided Kalman Filters (Revach et al., 2024)](https://arxiv.org/abs/2410.12289) that learn the Kalman gain or process model with RNNs/Transformers are active, though providing safety guarantees remains a challenge.
