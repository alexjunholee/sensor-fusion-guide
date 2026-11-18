# Chapter 4. State Estimation

Chapters 2 and 3 established how individual sensors measure the physical world and how their measurements are spatially aligned and temporally synchronized. Once those calibrated streams arrive, the next challenge is fusing them into a coherent estimate of the robot's evolving state—its position, velocity, orientation, and internal sensor biases. That is the role of the **state estimation** algorithms that power odometry and multi-sensor fusion pipelines (→ Chapters 6–8).

> State estimation traces a continuous arc from recursive Bayesian filtering—through the Kalman filter family and particle filters—to factor-graph optimization. Following the derivations in sequence reveals not just how each estimator works, but why modern robotics largely moved from single-step filtering to graph-based smoothing.

---

## 4.1 Bayesian Filtering Framework

### 4.1.1 Definition of the State Estimation Problem

Because real sensors are noisy and motion commands never execute perfectly, a robot cannot observe its true state directly. Instead, state estimation is formulated as a **conditional probabilistic inference** problem: given every observation up to the present, $\mathbf{z}_{1:k}$, and every control input, $\mathbf{u}_{1:k}$, we seek the posterior distribution over the current state $\mathbf{x}_k$:

$$p(\mathbf{x}_k \mid \mathbf{z}_{1:k}, \mathbf{u}_{1:k})$$

In this notation, $\mathbf{x}_k \in \mathbb{R}^n$ denotes the state vector at time $k$ (typically position, velocity, attitude, and sensor biases), $\mathbf{z}_k \in \mathbb{R}^m$ denotes the observation vector from sensors such as cameras, LiDAR, or IMUs, and $\mathbf{u}_k \in \mathbb{R}^l$ denotes the control input applied at step $k$.

To relate these quantities probabilistically, we specify two generative models that describe how the robot moves and how its sensors perceive the world:

**Motion Model (Process Model)**:
$$\mathbf{x}_k = f(\mathbf{x}_{k-1}, \mathbf{u}_k) + \mathbf{w}_k, \quad \mathbf{w}_k \sim \mathcal{N}(\mathbf{0}, \mathbf{Q}_k)$$

$$p(\mathbf{x}_k \mid \mathbf{x}_{k-1}, \mathbf{u}_k)$$

This defines the transition probability from the previous state to the current state under control input $\mathbf{u}_k$, with process noise $\mathbf{w}_k$ capturing unmodeled dynamics and actuation uncertainty.

**Observation Model (Measurement Model)**:
$$\mathbf{z}_k = h(\mathbf{x}_k) + \mathbf{v}_k, \quad \mathbf{v}_k \sim \mathcal{N}(\mathbf{0}, \mathbf{R}_k)$$

$$p(\mathbf{z}_k \mid \mathbf{x}_k)$$

This specifies the likelihood of receiving observation $\mathbf{z}_k$ given the current state, with measurement noise $\mathbf{v}_k$ capturing sensor noise.

Taken together, the motion model predicts how the robot's belief spreads as it moves, while the observation model anchors that belief against incoming sensor readings.

### 4.1.2 Markov Assumptions and Recursive Estimation

If computing $p(\mathbf{x}_k \mid \mathbf{z}_{1:k}, \mathbf{u}_{1:k})$ required re-evaluating the entire measurement history from time $1$ to $k$ at every step, memory and runtime would grow without bound. To make real-time filtering tractable, we introduce two **Markov assumptions**:

1. **First-order Markov process**: Given the immediately preceding state $\mathbf{x}_{k-1}$ and the current control $\mathbf{u}_k$, the current state $\mathbf{x}_k$ is conditionally independent of all earlier states, controls, and observations:
$$p(\mathbf{x}_k \mid \mathbf{x}_{0:k-1}, \mathbf{u}_{1:k}, \mathbf{z}_{1:k-1}) = p(\mathbf{x}_k \mid \mathbf{x}_{k-1}, \mathbf{u}_k)$$

2. **Conditional observation independence**: Given the current state $\mathbf{x}_k$, the current observation $\mathbf{z}_k$ is conditionally independent of past states, controls, and earlier observations:
$$p(\mathbf{z}_k \mid \mathbf{x}_{0:k}, \mathbf{u}_{1:k}, \mathbf{z}_{1:k-1}) = p(\mathbf{z}_k \mid \mathbf{x}_k)$$

Together, these assumptions allow us to fold all past history into the previous posterior and update it incrementally as each new measurement arrives—a paradigm known as **recursive estimation**.

### 4.1.3 Prediction-Update Cycle

Under the Markov assumptions, Bayesian filtering alternates between predicting the state forward through motion and correcting that prediction with incoming measurements:

**Prediction Step**: Starting from the previous posterior $p(\mathbf{x}_{k-1} \mid \mathbf{z}_{1:k-1}, \mathbf{u}_{1:k-1})$, we propagate our belief forward using the motion model:

$$\boxed{p(\mathbf{x}_k \mid \mathbf{z}_{1:k-1}, \mathbf{u}_{1:k}) = \int p(\mathbf{x}_k \mid \mathbf{x}_{k-1}, \mathbf{u}_k) \, p(\mathbf{x}_{k-1} \mid \mathbf{z}_{1:k-1}, \mathbf{u}_{1:k-1}) \, d\mathbf{x}_{k-1}}$$

This integral is the **Chapman-Kolmogorov equation**. Because we do not know $\mathbf{x}_{k-1}$ with certainty, we average the transition probability $p(\mathbf{x}_k \mid \mathbf{x}_{k-1}, \mathbf{u}_k)$ over every plausible previous state, weighted by the previous posterior $p(\mathbf{x}_{k-1} \mid \mathbf{z}_{1:k-1})$. The result is our predicted prior before seeing $\mathbf{z}_k$.

**Update Step**: When a new observation $\mathbf{z}_k$ arrives, we incorporate it via Bayes' rule:

$$\boxed{p(\mathbf{x}_k \mid \mathbf{z}_{1:k}, \mathbf{u}_{1:k}) = \frac{p(\mathbf{z}_k \mid \mathbf{x}_k) \, p(\mathbf{x}_k \mid \mathbf{z}_{1:k-1}, \mathbf{u}_{1:k})}{p(\mathbf{z}_k \mid \mathbf{z}_{1:k-1})}}$$

Here the numerator multiplies the observation likelihood $p(\mathbf{z}_k \mid \mathbf{x}_k)$ by the predicted prior $p(\mathbf{x}_k \mid \mathbf{z}_{1:k-1})$, while the denominator $p(\mathbf{z}_k \mid \mathbf{z}_{1:k-1}) = \int p(\mathbf{z}_k \mid \mathbf{x}_k) p(\mathbf{x}_k \mid \mathbf{z}_{1:k-1}) d\mathbf{x}_k$ acts as the normalizing evidence constant. Intuitively, each update scores candidate states by how well they explain the new sensor measurement and shifts probability mass toward the most consistent hypotheses.

### 4.1.4 Why Closed-Form Solutions Fail

While the prediction-update recursion is exact in theory, the Chapman-Kolmogorov integral $\int p(\mathbf{x}_k \mid \mathbf{x}_{k-1}) p(\mathbf{x}_{k-1} \mid \mathbf{z}_{1:k-1}) d\mathbf{x}_{k-1}$ rarely admits a closed-form solution in real robotic systems for three reasons:

1. **Nonlinear motion and observation models**: When $f(\cdot)$ or $h(\cdot)$ is nonlinear, passing a Gaussian distribution through the model does not preserve Gaussianity, and the posterior can take arbitrarily skewed or curved shapes.
2. **High-dimensional state spaces**: Numerical integration on a grid scales exponentially with state dimension, quickly running into the curse of dimensionality.
3. **Multimodal distributions**: In problems such as global localization in symmetric environments, several distinct poses can explain the observations equally well, producing a multimodal posterior that a single-peak density cannot capture.

Practical filters therefore differ primarily in how they approximate this intractable integral and represent the resulting belief:

| Approximation | Filter | Distribution representation | Regime |
|----------|------|---------|----------|
| Linear-Gaussian assumption | Kalman Filter | $\mathcal{N}(\hat{\mathbf{x}}, \mathbf{P})$ | Linear systems |
| First-order linearization + Gaussian | EKF, ESKF | $\mathcal{N}(\hat{\mathbf{x}}, \mathbf{P})$ | Mildly nonlinear |
| Sigma point transform | UKF | $\mathcal{N}(\hat{\mathbf{x}}, \mathbf{P})$ | Moderately nonlinear |
| Sample-based | Particle Filter | $\{(\mathbf{x}^{(i)}, w^{(i)})\}_{i=1}^N$ | Strongly nonlinear, multimodal |
| Iterated linearization | IEKF | $\mathcal{N}(\hat{\mathbf{x}}, \mathbf{P})$ | Strongly nonlinear |

We begin with the one setting where no approximation is needed—the linear-Gaussian Kalman filter—and then examine how each extension relaxes those assumptions.

---

## 4.2 Kalman Filter Family

### 4.2.1 Kalman Filter: Derivation and Optimality

When both the dynamics and the measurements are linear and all uncertainties are Gaussian, the intractable Bayesian recursion collapses into closed-form linear algebra. Under these assumptions, Gaussianity is preserved across every prediction and update step, so the entire posterior can be tracked exactly through just two objects: a mean vector and a covariance matrix. That exact solution is the **Kalman Filter (KF)**, introduced by Rudolf Kalman in his 1960 paper ["A New Approach to Linear Filtering and Prediction Problems"](https://doi.org/10.1115/1.3662552) and proven in practice on Apollo-era orbit estimation.

#### Linear System Model

To see why Gaussianity closes under linear dynamics, we write the state transition and observation models in matrix form:

$$\mathbf{x}_k = \mathbf{F}_k \mathbf{x}_{k-1} + \mathbf{B}_k \mathbf{u}_k + \mathbf{w}_k, \quad \mathbf{w}_k \sim \mathcal{N}(\mathbf{0}, \mathbf{Q}_k)$$
$$\mathbf{z}_k = \mathbf{H}_k \mathbf{x}_k + \mathbf{v}_k, \quad \mathbf{v}_k \sim \mathcal{N}(\mathbf{0}, \mathbf{R}_k)$$

Here $\mathbf{F}_k \in \mathbb{R}^{n \times n}$ is the state transition matrix, $\mathbf{B}_k \in \mathbb{R}^{n \times l}$ is the control input matrix, and $\mathbf{H}_k \in \mathbb{R}^{m \times n}$ maps the state into measurement space. Uncertainty enters through the symmetric positive semi-definite process noise covariance $\mathbf{Q}_k \in \mathbb{R}^{n \times n}$ and the symmetric positive definite measurement noise covariance $\mathbf{R}_k \in \mathbb{R}^{m \times m}$. Because any affine transformation of a Gaussian random vector produces another Gaussian, a Gaussian initial belief stays Gaussian forever and is completely specified by its mean and covariance.

#### MMSE Derivation — Why the Kalman Filter Is Optimal

We can derive the filter directly from the Minimum Mean Square Error (MMSE) criterion, seeking the estimator that minimizes the expected squared Euclidean error given all measurements so far:

$$\hat{\mathbf{x}}_k = \arg\min_{\hat{\mathbf{x}}} \mathbb{E}[\|\mathbf{x}_k - \hat{\mathbf{x}}\|^2 \mid \mathbf{z}_{1:k}]$$

For any distribution, the MMSE estimator is the conditional expectation $\hat{\mathbf{x}}_k = \mathbb{E}[\mathbf{x}_k \mid \mathbf{z}_{1:k}]$. Evaluating the Bayesian prediction-update recursion for a linear-Gaussian system shows that this conditional expectation coincides exactly with the Kalman filter equations.

Assume that after incorporating measurements up to step $k-1$, the posterior is Gaussian:

$$p(\mathbf{x}_{k-1} \mid \mathbf{z}_{1:k-1}) = \mathcal{N}(\hat{\mathbf{x}}_{k-1|k-1}, \mathbf{P}_{k-1|k-1})$$

**Step 1: Prediction — evaluate the Chapman-Kolmogorov integral**

Under the linear motion model,
$$\mathbf{x}_k = \mathbf{F}_k \mathbf{x}_{k-1} + \mathbf{B}_k \mathbf{u}_k + \mathbf{w}_k$$

the predicted state combines two independent Gaussian terms—the previous state and the process noise $\mathbf{w}_k$—so $\mathbf{x}_k$ remains Gaussian. Taking expectations gives the predicted mean:

$$\hat{\mathbf{x}}_{k|k-1} = \mathbb{E}[\mathbf{x}_k \mid \mathbf{z}_{1:k-1}] = \mathbf{F}_k \hat{\mathbf{x}}_{k-1|k-1} + \mathbf{B}_k \mathbf{u}_k$$

Subtracting this mean from $\mathbf{x}_k$ isolates the prediction error $\tilde{\mathbf{x}}_{k|k-1} = \mathbf{x}_k - \hat{\mathbf{x}}_{k|k-1} = \mathbf{F}_k \tilde{\mathbf{x}}_{k-1|k-1} + \mathbf{w}_k$, whose outer-product expectation gives the predicted covariance:

$$\mathbf{P}_{k|k-1} = \mathbb{E}[\tilde{\mathbf{x}}_{k|k-1} \tilde{\mathbf{x}}_{k|k-1}^\top] = \mathbf{F}_k \mathbf{P}_{k-1|k-1} \mathbf{F}_k^\top + \mathbf{Q}_k$$

Thus the Chapman-Kolmogorov integral evaluates cleanly to $p(\mathbf{x}_k \mid \mathbf{z}_{1:k-1}) = \mathcal{N}(\hat{\mathbf{x}}_{k|k-1}, \mathbf{P}_{k|k-1})$.

**Step 2: Update — apply Bayes' rule**

When the measurement $\mathbf{z}_k = \mathbf{H}_k \mathbf{x}_k + \mathbf{v}_k$ arrives, the predicted state and the predicted observation are jointly Gaussian:

$$\begin{bmatrix} \mathbf{x}_k \\ \mathbf{z}_k \end{bmatrix} \sim \mathcal{N}\left( \begin{bmatrix} \hat{\mathbf{x}}_{k|k-1} \\ \mathbf{H}_k \hat{\mathbf{x}}_{k|k-1} \end{bmatrix}, \begin{bmatrix} \mathbf{P}_{k|k-1} & \mathbf{P}_{k|k-1} \mathbf{H}_k^\top \\ \mathbf{H}_k \mathbf{P}_{k|k-1} & \mathbf{S}_k \end{bmatrix} \right)$$

where $\mathbf{S}_k = \mathbf{H}_k \mathbf{P}_{k|k-1} \mathbf{H}_k^\top + \mathbf{R}_k \in \mathbb{R}^{m \times m}$ is the innovation covariance. Conditioning one component of a jointly Gaussian vector on another yields a Gaussian whose mean shifts from the prior mean in proportion to the cross-covariance and the measurement discrepancy:

$$\hat{\mathbf{x}}_{k|k} = \hat{\mathbf{x}}_{k|k-1} + \mathbf{P}_{k|k-1} \mathbf{H}_k^\top \mathbf{S}_k^{-1} (\mathbf{z}_k - \mathbf{H}_k \hat{\mathbf{x}}_{k|k-1})$$

Factoring out the matrix multiplier defines the **Kalman gain** $\mathbf{K}_k \in \mathbb{R}^{n \times m}$:

$$\boxed{\mathbf{K}_k = \mathbf{P}_{k|k-1} \mathbf{H}_k^\top \mathbf{S}_k^{-1} = \mathbf{P}_{k|k-1} \mathbf{H}_k^\top (\mathbf{H}_k \mathbf{P}_{k|k-1} \mathbf{H}_k^\top + \mathbf{R}_k)^{-1}}$$

The Kalman gain acts as an optimal weighting factor balancing prior confidence against sensor precision. As measurement noise vanishes ($\mathbf{R}_k \to \mathbf{0}$), the update tracks the measurement constraint tightly in the observable subspace (noting that writing $\mathbf{H}_k^{-1}$ applies only when $\mathbf{H}_k$ is square and invertible). Conversely, when the prediction is already highly certain ($\mathbf{P}_{k|k-1} \to \mathbf{0}$), $\mathbf{K}_k \to \mathbf{0}$ and the filter relies on the prior rather than noisy observations. Whenever the linear model and noise covariances are accurate, this gain achieves the exact conditional-mean weighting.

The discrepancy driving the correction is the **innovation**—the difference between what actually arrived and what the filter expected to see:
$$\tilde{\mathbf{y}}_k = \mathbf{z}_k - \mathbf{H}_k \hat{\mathbf{x}}_{k|k-1} \in \mathbb{R}^m$$

Note that a zero innovation only indicates that this particular measurement residual vanishes, not that unobservable state components or modeling errors are absent. Multiplying the innovation by the Kalman gain completes the state update, and conditioning shrinks the prior covariance accordingly:

$$\boxed{\hat{\mathbf{x}}_{k|k} = \hat{\mathbf{x}}_{k|k-1} + \mathbf{K}_k \tilde{\mathbf{y}}_k}$$

$$\boxed{\mathbf{P}_{k|k} = (\mathbf{I} - \mathbf{K}_k \mathbf{H}_k) \mathbf{P}_{k|k-1}}$$

In numerical implementations, the covariance update is more stable in **Joseph form**: $\mathbf{P}_{k|k} = (\mathbf{I} - \mathbf{K}_k \mathbf{H}_k) \mathbf{P}_{k|k-1} (\mathbf{I} - \mathbf{K}_k \mathbf{H}_k)^\top + \mathbf{K}_k \mathbf{R}_k \mathbf{K}_k^\top$. Unlike the simpler subtraction above, which can lose symmetry or positive semidefiniteness under floating-point roundoff, the Joseph form preserves that algebraic structure whenever $\mathbf{P}_{k|k-1}$ and $\mathbf{R}_k$ are positive semidefinite; practical code often adds explicit symmetrization or uses factorized solvers as well.

#### KF Optimality Theorem

**Theorem**: In a linear-Gaussian system, the Kalman filter is simultaneously optimal in three distinct senses:
1. **MMSE (Minimum Mean Square Error) estimator**: It minimizes the expected squared estimation error $\mathbb{E}[\|\mathbf{x}_k - \hat{\mathbf{x}}\|^2 \mid \mathbf{z}_{1:k}]$.
2. **MAP (Maximum A Posteriori) estimator**: Because a Gaussian distribution's mean and mode coincide, the MMSE and MAP estimates are identical.
3. **BLUE (Best Linear Unbiased Estimator)**: Even if the noise is non-Gaussian, so long as its first two moments match $\mathbf{Q}_k$ and $\mathbf{R}_k$, the KF remains the minimum-variance estimator among all linear unbiased estimators.

Historically, Kalman's 1960 formulation replaced Wiener's stationary frequency-domain filter with a recursive state-space representation in the time domain, extending naturally to time-varying, multi-input multi-output systems.

#### Python Implementation

To make the prediction-update cycle concrete, the implementation below sets up a minimal `KalmanFilter` class using the Joseph-form covariance update and runs it on a 1D constant-velocity target tracked solely through noisy position measurements.

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
              [dt**2/2, dt]]) * 0.1  # (2, 2) process noise (constant-velocity model + continuous-time white-noise acceleration)
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

Unfortunately, real robotic sensors and kinematics rarely obey linear equations. Perspective camera projection maps 3D points onto 2D pixels nonlinearly, IMU integration rotates gravity and acceleration through 3D orientation, LiDAR scan matching couples pose nonlinearly with scene geometry, and GNSS coordinate conversions bend across a curved Earth. The **Extended Kalman Filter (EKF)** is the most direct way to preserve the Kalman recursion in these nonlinear regimes: keep the Gaussian parameterization, and linearize the models locally at each step.

#### First-Order Taylor Expansion (Linearization)

Consider a general nonlinear system with additive Gaussian noise:
$$\mathbf{x}_k = f(\mathbf{x}_{k-1}, \mathbf{u}_k) + \mathbf{w}_k$$
$$\mathbf{z}_k = h(\mathbf{x}_k) + \mathbf{v}_k$$

Because $f$ and $h$ are nonlinear, passing a Gaussian belief through them yields a non-Gaussian distribution. The EKF restores tractability by replacing each nonlinear function with its **first-order Taylor expansion** around the current best estimate. For the motion model, we linearize around the previous posterior mean $\hat{\mathbf{x}}_{k-1|k-1}$:

$$f(\mathbf{x}_{k-1}, \mathbf{u}_k) \approx f(\hat{\mathbf{x}}_{k-1|k-1}, \mathbf{u}_k) + \mathbf{F}_k (\mathbf{x}_{k-1} - \hat{\mathbf{x}}_{k-1|k-1})$$

$$\mathbf{F}_k = \left.\frac{\partial f}{\partial \mathbf{x}}\right|_{\hat{\mathbf{x}}_{k-1|k-1}, \mathbf{u}_k} \in \mathbb{R}^{n \times n}$$

Similarly, for the observation model, we linearize around the predicted state $\hat{\mathbf{x}}_{k|k-1}$:

$$h(\mathbf{x}_k) \approx h(\hat{\mathbf{x}}_{k|k-1}) + \mathbf{H}_k (\mathbf{x}_k - \hat{\mathbf{x}}_{k|k-1})$$

$$\mathbf{H}_k = \left.\frac{\partial h}{\partial \mathbf{x}}\right|_{\hat{\mathbf{x}}_{k|k-1}} \in \mathbb{R}^{m \times n}$$

#### EKF Algorithm

Once the Jacobians $\mathbf{F}_k$ and $\mathbf{H}_k$ are evaluated, the EKF mimics the linear KF structure while still pushing the mean and innovation through the true nonlinear models. In the **prediction step**, the state advances via $f$ directly while uncertainty propagates through $\mathbf{F}_k$:
$$\hat{\mathbf{x}}_{k|k-1} = f(\hat{\mathbf{x}}_{k-1|k-1}, \mathbf{u}_k)$$
$$\mathbf{P}_{k|k-1} = \mathbf{F}_k \mathbf{P}_{k-1|k-1} \mathbf{F}_k^\top + \mathbf{Q}_k$$

In the **update step**, the innovation compares the actual measurement against the nonlinear prediction $h(\hat{\mathbf{x}}_{k|k-1})$, and the linear Jacobian $\mathbf{H}_k$ governs the gain and covariance shrinkage:
$$\tilde{\mathbf{y}}_k = \mathbf{z}_k - h(\hat{\mathbf{x}}_{k|k-1})$$
$$\mathbf{S}_k = \mathbf{H}_k \mathbf{P}_{k|k-1} \mathbf{H}_k^\top + \mathbf{R}_k$$
$$\mathbf{K}_k = \mathbf{P}_{k|k-1} \mathbf{H}_k^\top \mathbf{S}_k^{-1}$$
$$\hat{\mathbf{x}}_{k|k} = \hat{\mathbf{x}}_{k|k-1} + \mathbf{K}_k \tilde{\mathbf{y}}_k$$
$$\mathbf{P}_{k|k} = (\mathbf{I} - \mathbf{K}_k \mathbf{H}_k) \mathbf{P}_{k|k-1}$$

#### Limitations of the EKF

This local linearization makes the EKF fast and simple, yet it introduces three structural weaknesses that motivate the variants in the rest of this section:

1. **Linearization error and inconsistency**: When curvature is strong or uncertainty is wide, a tangent-plane approximation diverges from the true nonlinear transformation. Beyond degrading accuracy, this underestimates covariance and undermines filter consistency—the actual estimation error outgrows the uncertainty the filter claims.
2. **Jacobian maintenance**: Analytical Jacobians $\mathbf{F}_k$ and $\mathbf{H}_k$ must be derived and evaluated at every step. For tightly coupled multi-sensor models, deriving these partial derivatives by hand is tedious and bug-prone.
3. **Unimodal Gaussian constraint**: Because the posterior is forced into a single Gaussian, an EKF cannot represent multi-hypothesis beliefs when a robot faces ambiguous symmetry or global localization.

Even more pressing for 3D robotics is a fourth issue hidden inside the update equation $\hat{\mathbf{x}}_{k|k} = \hat{\mathbf{x}}_{k|k-1} + \mathbf{K}_k \tilde{\mathbf{y}}_k$: ordinary vector addition breaks down as soon as the state includes 3D rotation. That limitation leads directly to the Error-State Kalman Filter.

### 4.2.3 Error-State Kalman Filter (ESKF)

When an IMU is part of the sensor suite, the **Error-State Kalman Filter (ESKF)** is the standard way to formulate a Kalman filter in 3D. Filter-based VIO systems such as MSCKF and OpenVINS, along with the FAST-LIO family, rely on error-state or closely related manifold-filter formulations—whereas [VINS-Mono](https://doi.org/10.1109/TRO.2018.2853729) opts for sliding-window nonlinear optimization instead.

#### Why ESKF Instead of EKF

Applying a standard EKF directly to a 3D navigation state—one that couples position, velocity, orientation, and IMU biases—runs into three structural difficulties:

**Problem 1: Non-Euclidean nature of rotation**

3D orientation lives on the $SO(3)$ manifold rather than in $\mathbb{R}^n$. Representing it with a unit quaternion $\mathbf{q} \in \mathbb{H}$ imposes $\|\mathbf{q}\| = 1$, while a rotation matrix $\mathbf{R} \in SO(3)$ must satisfy $\mathbf{R}^\top \mathbf{R} = \mathbf{I}$ and $\det(\mathbf{R}) = 1$. In a naive EKF update $\hat{\mathbf{x}} \leftarrow \hat{\mathbf{x}} + \mathbf{K} \tilde{\mathbf{y}}$, the "+" denotes ordinary Euclidean addition, so adding an update vector to all four quaternion entries violates the unit-norm constraint. Although a direct quaternion EKF can be engineered with consistent normalization and covariance projection, merely normalizing the quaternion after an additive update mishandles uncertainty along the constrained radial direction.

**Property 2: The error state is "almost zero"**

After each measurement update, the estimated error state $\delta\mathbf{x} = \mathbf{x} \boxminus \hat{\mathbf{x}}$ is injected into the nominal state and reset to zero. As long as the nominal trajectory tracks the true state closely, the error stays within a small tangent-space neighborhood where first-order linearization holds well. Resetting alone does not guarantee small linearization errors—good initialization, observability, and accurate modeling remain essential—but separating the large trajectory from small perturbations keeps the filter's linearization well-behaved.

**Problem 3: Different dynamics of the nominal and error states**

Large, rapid motion is integrated cleanly inside the nominal state, while small deviations around that trajectory evolve inside the error state. That separation maps directly onto multi-rate sensor fusion: high-rate IMU readings drive nominal propagation, and lower-rate exteroceptive measurements correct the accumulated error.

#### ESKF Structure

To exploit this separation, the ESKF maintains two coupled representations in parallel:

1. **Nominal state** $\hat{\mathbf{x}}$: integrated through the nonlinear motion model without noise terms, serving as the noise-free reference trajectory without tracking uncertainty.
2. **Error state** $\delta\mathbf{x}$: the small local discrepancy between the true state and the nominal state, estimated by a linear/extended Kalman filter in the tangent space. While the nominal trajectory remains close to the truth, a first-order tangent-space approximation is effective; once an update is injected into $\hat{\mathbf{x}}$, a reset Jacobian consistently transforms the error covariance into the new tangent coordinates.

The true state is recovered by composing the nominal estimate with the error state via the manifold operator $\boxplus$:

$$\mathbf{x}_{\text{true}} = \hat{\mathbf{x}} \boxplus \delta\mathbf{x}$$

For Euclidean components such as position, velocity, and biases, $\boxplus$ reduces to ordinary vector addition; for 3D rotation, it applies a small multiplicative perturbation in the tangent space:

$$\mathbf{R}_{\text{true}} = \hat{\mathbf{R}} \cdot \text{Exp}(\delta\boldsymbol{\theta})$$

In unit-quaternion form, the same small-angle perturbation writes as:

$$\mathbf{q}_{\text{true}} = \hat{\mathbf{q}} \otimes \begin{bmatrix} 1 \\ \frac{1}{2}\delta\boldsymbol{\theta} \end{bmatrix} \approx \hat{\mathbf{q}} \otimes \delta\mathbf{q}$$

where $\delta\boldsymbol{\theta} \in \mathbb{R}^3$ is the angle-axis error vector.

#### State Vector of an IMU-Based ESKF

In a typical IMU–camera or IMU–LiDAR fusion pipeline, the **nominal state** is 16-dimensional when parameterized with a quaternion:
$$\hat{\mathbf{x}} = \begin{bmatrix} {}^W\hat{\mathbf{p}} \\ {}^W\hat{\mathbf{v}} \\ \hat{\mathbf{q}}_{WB} \\ \hat{\mathbf{b}}_a \\ \hat{\mathbf{b}}_g \end{bmatrix} \in \mathbb{R}^{3} \times \mathbb{R}^{3} \times \mathbb{S}^3 \times \mathbb{R}^{3} \times \mathbb{R}^{3}$$

The corresponding **error state**, by contrast, is 15-dimensional because it uses a minimal 3-DoF tangent-space parameterization for orientation:
$$\delta\mathbf{x} = \begin{bmatrix} \delta\mathbf{p} \\ \delta\mathbf{v} \\ \delta\boldsymbol{\theta} \\ \delta\mathbf{b}_a \\ \delta\mathbf{b}_g \end{bmatrix} \in \mathbb{R}^{15}$$

Representing orientation errors with $\delta\boldsymbol{\theta} \in \mathbb{R}^3$ rather than four quaternion components matches the three true degrees of freedom of $SO(3)$ and avoids redundant singular directions in the covariance matrix. (A direct quaternion EKF can retain four components, provided it applies consistent normalization and covariance projection.)

#### ESKF Algorithm

Putting the nominal and error states together gives a four-step cycle across IMU propagation and external sensor updates.

**Step 1: Nominal state propagation (IMU mechanization)**

Each incoming IMU sample $(\tilde{\boldsymbol{\omega}}_k, \tilde{\mathbf{a}}_k)$ drives the nominal state forward once estimated biases are subtracted and noise terms are omitted:

$$\hat{\mathbf{q}}_{k+1} = \hat{\mathbf{q}}_k \otimes \mathbf{q}\{(\tilde{\boldsymbol{\omega}}_k - \hat{\mathbf{b}}_{g,k}) \Delta t\}$$
$$\hat{\mathbf{v}}_{k+1} = \hat{\mathbf{v}}_k + (\hat{\mathbf{R}}_k (\tilde{\mathbf{a}}_k - \hat{\mathbf{b}}_{a,k}) + \mathbf{g}) \Delta t$$
$$\hat{\mathbf{p}}_{k+1} = \hat{\mathbf{p}}_k + \hat{\mathbf{v}}_k \Delta t + \frac{1}{2}(\hat{\mathbf{R}}_k (\tilde{\mathbf{a}}_k - \hat{\mathbf{b}}_{a,k}) + \mathbf{g}) \Delta t^2$$
$$\hat{\mathbf{b}}_{a,k+1} = \hat{\mathbf{b}}_{a,k}$$
$$\hat{\mathbf{b}}_{g,k+1} = \hat{\mathbf{b}}_{g,k}$$

Here $\hat{\mathbf{R}}_k = \mathbf{R}(\hat{\mathbf{q}}_k) \in SO(3)$ is the nominal body-to-world rotation matrix, and $\mathbf{g} = [0, 0, -9.81]^\top \, \text{m/s}^2$ is the gravity vector in a Z-up world frame.

**Step 2: Error-state propagation (prediction)**

To propagate uncertainty, we derive the continuous-time error dynamics by substituting the IMU measurement models $\tilde{\boldsymbol{\omega}} = \boldsymbol{\omega} + \mathbf{b}_g + \mathbf{n}_g$ and $\tilde{\mathbf{a}} = \mathbf{R}^\top(\mathbf{a}_W - \mathbf{g}) + \mathbf{b}_a + \mathbf{n}_a$ into the true kinematics and subtracting the nominal equations:

$$\delta\dot{\boldsymbol{\theta}} = -[\tilde{\boldsymbol{\omega}} - \hat{\mathbf{b}}_g]_\times \delta\boldsymbol{\theta} - \delta\mathbf{b}_g - \mathbf{n}_g$$
$$\delta\dot{\mathbf{v}} = -\hat{\mathbf{R}}[\tilde{\mathbf{a}} - \hat{\mathbf{b}}_a]_\times \delta\boldsymbol{\theta} - \hat{\mathbf{R}} \delta\mathbf{b}_a - \hat{\mathbf{R}} \mathbf{n}_a$$
$$\delta\dot{\mathbf{p}} = \delta\mathbf{v}$$
$$\delta\dot{\mathbf{b}}_a = \mathbf{n}_{ba}$$
$$\delta\dot{\mathbf{b}}_g = \mathbf{n}_{bg}$$

Here $[\mathbf{a}]_\times$ denotes the skew-symmetric matrix associated with the cross product $\mathbf{a} \times (\cdot)$:

$$[\mathbf{a}]_\times = \begin{bmatrix} 0 & -a_3 & a_2 \\ a_3 & 0 & -a_1 \\ -a_2 & a_1 & 0 \end{bmatrix} \in \mathbb{R}^{3 \times 3}$$

Collecting the five equations into linear state-space form yields:

$$\delta\dot{\mathbf{x}} = \mathbf{F}_c \delta\mathbf{x} + \mathbf{G}_c \mathbf{n}$$

$$\mathbf{F}_c = \begin{bmatrix}
\mathbf{0}_3 & \mathbf{I}_3 & \mathbf{0}_3 & \mathbf{0}_3 & \mathbf{0}_3 \\
\mathbf{0}_3 & \mathbf{0}_3 & -\hat{\mathbf{R}}[\tilde{\mathbf{a}} - \hat{\mathbf{b}}_a]_\times & -\hat{\mathbf{R}} & \mathbf{0}_3 \\
\mathbf{0}_3 & \mathbf{0}_3 & -[\tilde{\boldsymbol{\omega}} - \hat{\mathbf{b}}_g]_\times & \mathbf{0}_3 & -\mathbf{I}_3 \\
\mathbf{0}_3 & \mathbf{0}_3 & \mathbf{0}_3 & \mathbf{0}_3 & \mathbf{0}_3 \\
\mathbf{0}_3 & \mathbf{0}_3 & \mathbf{0}_3 & \mathbf{0}_3 & \mathbf{0}_3
\end{bmatrix} \in \mathbb{R}^{15 \times 15}$$

Applying a first-order discretization $\mathbf{F}_d \approx \mathbf{I} + \mathbf{F}_c \Delta t$ then propagates the error covariance across the IMU interval:

$$\mathbf{P}_{k+1|k} = \mathbf{F}_d \mathbf{P}_{k|k} \mathbf{F}_d^\top + \mathbf{G}_d \mathbf{Q}_d \mathbf{G}_d^\top$$

**Step 3: Observation update**

Whenever an external measurement from a camera or LiDAR arrives, the filter runs a standard EKF update on the 15-dimensional error state. Linearizing the measurement model $\mathbf{z} = h(\mathbf{x}_{\text{true}})$ around the nominal state gives:

$$\mathbf{z} - h(\hat{\mathbf{x}}) \approx \mathbf{H} \delta\mathbf{x} + \mathbf{v}$$

$$\mathbf{H} = \frac{\partial h}{\partial \delta\mathbf{x}}\bigg|_{\hat{\mathbf{x}}} \in \mathbb{R}^{m \times 15}$$

This error-state Jacobian factors via the chain rule into the derivative with respect to the full state and the manifold perturbation Jacobian at zero error:

$$\mathbf{H} = \frac{\partial h}{\partial \mathbf{x}_{\text{true}}} \cdot \frac{\partial \mathbf{x}_{\text{true}}}{\partial \delta\mathbf{x}}\bigg|_{\delta\mathbf{x}=\mathbf{0}}$$

With $\mathbf{H}$ in hand, the Kalman gain, error-state estimate, and updated covariance follow immediately:
$$\mathbf{K} = \mathbf{P}_{k|k-1} \mathbf{H}^\top (\mathbf{H} \mathbf{P}_{k|k-1} \mathbf{H}^\top + \mathbf{R})^{-1}$$
$$\delta\hat{\mathbf{x}} = \mathbf{K}(\mathbf{z} - h(\hat{\mathbf{x}}))$$
$$\mathbf{P}_{k|k} = (\mathbf{I} - \mathbf{K}\mathbf{H})\mathbf{P}_{k|k-1}$$

**Step 4: Injection of the error state and reset**

Finally, we fold the estimated correction $\delta\hat{\mathbf{x}}$ back into the nominal state—additively for Euclidean states and multiplicatively on $SO(3)$ for the quaternion:

$$\hat{\mathbf{p}} \leftarrow \hat{\mathbf{p}} + \delta\hat{\mathbf{p}}$$
$$\hat{\mathbf{v}} \leftarrow \hat{\mathbf{v}} + \delta\hat{\mathbf{v}}$$
$$\hat{\mathbf{q}} \leftarrow \hat{\mathbf{q}} \otimes \mathbf{q}\{\delta\hat{\boldsymbol{\theta}}\}$$
$$\hat{\mathbf{b}}_a \leftarrow \hat{\mathbf{b}}_a + \delta\hat{\mathbf{b}}_a$$
$$\hat{\mathbf{b}}_g \leftarrow \hat{\mathbf{b}}_g + \delta\hat{\mathbf{b}}_g$$

Once injected, the error mean is reset to $\delta\hat{\mathbf{x}} \leftarrow \mathbf{0}$, and the covariance is mapped into the new nominal tangent frame through the reset Jacobian:

$$\mathbf{P} \leftarrow \mathbf{G} \mathbf{P} \mathbf{G}^\top$$

where $\mathbf{G} = \frac{\partial (\delta\mathbf{x} \boxminus \delta\hat{\mathbf{x}})}{\partial \delta\mathbf{x}}\big|_{\delta\hat{\mathbf{x}}}$. When the rotational correction $\delta\hat{\boldsymbol{\theta}}$ is small, implementations often approximate $\mathbf{G} \approx \mathbf{I}$.

### 4.2.4 Unscented Kalman Filter (UKF)

Both the EKF and ESKF work by linearizing the system equations and propagating a Gaussian through the resulting tangent approximation. The **Unscented Kalman Filter (UKF)** reverses that philosophy: rather than approximating a nonlinear function with a Taylor series, it is often easier and more accurate to approximate the probability distribution itself. The UKF leaves $f$ and $h$ untouched and captures the mean and covariance using a small, deterministic ensemble of **sigma points**.

#### Unscented Transform

Given an $n$-dimensional Gaussian random variable $\mathbf{x} \sim \mathcal{N}(\hat{\mathbf{x}}, \mathbf{P})$, the Unscented Transform places $2n+1$ sigma points symmetrically around the mean along the principal axes of the covariance:

$$\boldsymbol{\chi}_0 = \hat{\mathbf{x}}, \quad w_0 = \frac{\lambda}{n + \lambda}$$
$$\boldsymbol{\chi}_i = \hat{\mathbf{x}} + \left(\sqrt{(n+\lambda)\mathbf{P}}\right)_i, \quad w_i = \frac{1}{2(n+\lambda)}, \quad i = 1, \ldots, n$$
$$\boldsymbol{\chi}_{n+i} = \hat{\mathbf{x}} - \left(\sqrt{(n+\lambda)\mathbf{P}}\right)_i, \quad w_{n+i} = \frac{1}{2(n+\lambda)}, \quad i = 1, \ldots, n$$

Here $\lambda = \alpha^2(n + \kappa) - n$ sets the spread of the points ($\alpha$ controls dispersion around the mean, while $\kappa$ is a secondary scaling parameter typically chosen as $0$ or $3-n$), and $\left(\sqrt{(n+\lambda)\mathbf{P}}\right)_i$ denotes the $i$-th column of the Cholesky factor of $(n+\lambda)\mathbf{P}$.

Each sigma point is then pushed directly through the true nonlinear mapping:

$$\boldsymbol{\gamma}_i = f(\boldsymbol{\chi}_i)$$

Recombining the transformed points via weighted sample statistics yields the propagated mean and covariance:

$$\hat{\mathbf{y}} = \sum_{i=0}^{2n} w_i^{(m)} \boldsymbol{\gamma}_i$$
$$\mathbf{P}_y = \sum_{i=0}^{2n} w_i^{(c)} (\boldsymbol{\gamma}_i - \hat{\mathbf{y}})(\boldsymbol{\gamma}_i - \hat{\mathbf{y}})^\top$$

Here $w_i^{(m)}$ and $w_i^{(c)}$ denote the mean and covariance weights; only the center weight differs, with $w_0^{(c)} = w_0^{(m)} + (1 - \alpha^2 + \beta)$ (where $\beta = 2$ is optimal for Gaussian priors) and all remaining weights matching $w_i$.

#### UKF Algorithm

Embedded inside the Bayesian recursion, the Unscented Transform replaces Jacobian propagation in both phases:

**Prediction step**:
1. Generate sigma points from the current posterior $(\hat{\mathbf{x}}_{k-1|k-1}, \mathbf{P}_{k-1|k-1})$.
2. Propagate each sigma point through the nonlinear motion model: $\boldsymbol{\chi}_{k|k-1}^{(i)} = f(\boldsymbol{\chi}_{k-1|k-1}^{(i)}, \mathbf{u}_k)$.
3. Recover the predicted mean: $\hat{\mathbf{x}}_{k|k-1} = \sum w_i^{(m)} \boldsymbol{\chi}_{k|k-1}^{(i)}$.
4. Recover the predicted covariance and add process noise: $\mathbf{P}_{k|k-1} = \sum w_i^{(c)} (\boldsymbol{\chi}_{k|k-1}^{(i)} - \hat{\mathbf{x}}_{k|k-1})(\cdots)^\top + \mathbf{Q}_k$.

**Update step**:
1. Regenerate sigma points from $(\hat{\mathbf{x}}_{k|k-1}, \mathbf{P}_{k|k-1})$ (already incorporating $\mathbf{Q}_k$).
2. Project each point through the observation model: $\boldsymbol{\zeta}_k^{(i)} = h(\boldsymbol{\chi}_{k|k-1}^{(i)})$.
3. Compute the predicted observation mean: $\hat{\mathbf{z}}_k = \sum w_i^{(m)} \boldsymbol{\zeta}_k^{(i)}$.
4. Compute the innovation covariance: $\mathbf{P}_{zz} = \sum w_i^{(c)} (\boldsymbol{\zeta}_k^{(i)} - \hat{\mathbf{z}}_k)(\cdots)^\top + \mathbf{R}_k$.
5. Compute the state–observation cross-covariance: $\mathbf{P}_{xz} = \sum w_i^{(c)} (\boldsymbol{\chi}_{k|k-1}^{(i)} - \hat{\mathbf{x}}_{k|k-1})(\boldsymbol{\zeta}_k^{(i)} - \hat{\mathbf{z}}_k)^\top$.
6. Form the Kalman gain: $\mathbf{K}_k = \mathbf{P}_{xz} \mathbf{P}_{zz}^{-1}$.
7. Update the state mean: $\hat{\mathbf{x}}_{k|k} = \hat{\mathbf{x}}_{k|k-1} + \mathbf{K}_k (\mathbf{z}_k - \hat{\mathbf{z}}_k)$.
8. Update the state covariance: $\mathbf{P}_{k|k} = \mathbf{P}_{k|k-1} - \mathbf{K}_k \mathbf{P}_{zz} \mathbf{K}_k^\top$.

#### Pros and Cons of the UKF

**Pros**:
- **Jacobian-free implementation**: Skipping analytical derivatives is a major practical advantage for intricate observation models such as camera projection with nonlinear lens distortion, making the filter cleaner to code and maintain.
- **Higher-order moment capture**: Because deterministic sigma points sample the spread of the distribution directly through $f$ and $h$, they capture nonlinear curvature in the transformed mean and covariance better than a single tangent plane (though the formal order of accuracy depends on the distribution, nonlinearity, and sigma-point tuning).

**Cons**:
- **Scaling with state dimension**: Every one of the $2n+1$ sigma points must be evaluated through the nonlinear models, so compute grows linearly with $n$—expensive when the state grows to include many landmarks or calibration parameters.
- **Awkward manifold handling**: Extending sigma-point generation and weighted averaging to Lie groups such as $SO(3)$ requires replacing Euclidean differences and sums with manifold retractions and intrinsic means, which is neither trivial nor cheap.
- **Less common in IMU fusion**: In IMU-centric estimation, ESKF formulations are far more widespread because they handle local manifold errors cleanly without propagating $2n+1$ trajectories at IMU rate; which filter achieves lower error in a given system still depends on the model, initial error, and tuning.

### 4.2.5 Iterated Extended Kalman Filter (IEKF)

Instead of discarding Jacobians altogether as the UKF does, another way to curb linearization error is to keep the EKF structure and **relinearize iteratively inside the update step**. That idea yields the **Iterated Extended Kalman Filter (IEKF)**.

#### Motivation: Linearization Error in the EKF Update

A standard EKF evaluates the measurement Jacobian $\mathbf{H}_k$ once at the predicted state $\hat{\mathbf{x}}_{k|k-1}$. When an observation is highly informative or strongly nonlinear, the posterior state $\hat{\mathbf{x}}_{k|k}$ can shift far enough from $\hat{\mathbf{x}}_{k|k-1}$ that the original tangent plane is a poor fit at the new estimate. The IEKF resolves this by treating the update as an iterative refinement: recompute $\mathbf{H}$ at the latest iterate and re-evaluate the correction until the estimate settles.

#### IEKF Algorithm

Prediction proceeds identically to the standard EKF. During the update step, we initialize $\hat{\mathbf{x}}^{(0)} = \hat{\mathbf{x}}_{k|k-1}$ and iterate for $j = 0, 1, 2, \ldots$ until convergence:

$$\mathbf{H}^{(j)} = \left.\frac{\partial h}{\partial \mathbf{x}}\right|_{\hat{\mathbf{x}}^{(j)}}$$

$$\mathbf{K}^{(j)} = \mathbf{P}_{k|k-1} \mathbf{H}^{(j)\top} (\mathbf{H}^{(j)} \mathbf{P}_{k|k-1} \mathbf{H}^{(j)\top} + \mathbf{R}_k)^{-1}$$

$$\hat{\mathbf{x}}^{(j+1)} = \hat{\mathbf{x}}_{k|k-1} + \mathbf{K}^{(j)} \left[\mathbf{z}_k - h(\hat{\mathbf{x}}^{(j)}) - \mathbf{H}^{(j)}(\hat{\mathbf{x}}_{k|k-1} - \hat{\mathbf{x}}^{(j)})\right]$$

Notice the extra term $-\mathbf{H}^{(j)}(\hat{\mathbf{x}}_{k|k-1} - \hat{\mathbf{x}}^{(j)})$: it keeps the prior anchor centered at $\hat{\mathbf{x}}_{k|k-1}$ while linearizing the measurement at $\hat{\mathbf{x}}^{(j)}$. Once the iterates converge, we set $\hat{\mathbf{x}}_{k|k} = \hat{\mathbf{x}}^{(j+1)}$ and compute the posterior covariance $\mathbf{P}_{k|k} = (\mathbf{I} - \mathbf{K}^{(j)} \mathbf{H}^{(j)}) \mathbf{P}_{k|k-1}$.

Mathematically, this loop is equivalent to solving a single-step **Gauss-Newton optimization** problem balancing the prior against the measurement likelihood—a deep structural link to the factor-graph optimizers we examine in Section 4.5. In practice, FAST-LIO2 builds directly on this insight by running a manifold IEKF that relinearizes thousands of LiDAR point-to-plane residuals across iterations during each scan update; doing so accounts for residual nonlinearity beyond a single linearization, with the actual accuracy gain and iteration count depending on initial error, scene geometry, and the stopping criterion.

Yet every filter in the Kalman family shares one fundamental boundary: each represents the posterior as a single unimodal Gaussian. When a robot faces global localization ambiguity or severe non-Gaussianity where multiple distinct hypotheses must survive simultaneously, Gaussian filtering breaks down and we turn to sample-based particle methods.

---

## 4.3 Particle Filter

### 4.3.1 Overview of Sequential Monte Carlo (SMC)

The **particle filter (PF)**—known more generally as the **Sequential Monte Carlo (SMC)** method—discards the Gaussian assumption altogether. Rather than summarizing the posterior with a single mean and covariance, it represents the distribution empirically as a cloud of weighted hypotheses called **particles**. Because no parametric shape is imposed, a particle cloud can track skewed, non-Gaussian beliefs as well as multimodal posteriors with several competing peaks.

Mathematically, the posterior at time $k$ is approximated by a weighted sum of Dirac delta functions:

$$p(\mathbf{x}_k \mid \mathbf{z}_{1:k}) \approx \sum_{i=1}^{N} w_k^{(i)} \delta(\mathbf{x}_k - \mathbf{x}_k^{(i)})$$

where $N$ is the particle count, $\mathbf{x}_k^{(i)}$ is the state hypothesis of the $i$-th particle, $w_k^{(i)}$ is its normalized importance weight ($\sum_{i=1}^N w_k^{(i)} = 1$), and $\delta(\cdot)$ denotes the Dirac delta distribution. In the asymptotic limit $N \to \infty$, this empirical measure converges to the true posterior; in practice, $N$ is chosen to balance approximation fidelity against real-time compute, taking into account both the complexity of the posterior and the dimension of the state space.

### 4.3.2 Importance Sampling

Drawing samples directly from the true posterior $p(\mathbf{x}_k \mid \mathbf{z}_{1:k})$ is typically impossible because its normalizing constant and closed form are unknown. **Importance sampling** sidesteps this barrier by drawing particles from an easier **proposal distribution** $q(\mathbf{x}_k \mid \mathbf{x}_{k-1}, \mathbf{z}_k)$ and compensating for the mismatch with importance weights.

To derive the recursive weight update, apply Bayes' rule to the full trajectory posterior:

$$p(\mathbf{x}_{0:k} \mid \mathbf{z}_{1:k}) = \frac{p(\mathbf{z}_k \mid \mathbf{x}_k) \, p(\mathbf{x}_k \mid \mathbf{x}_{k-1}) \, p(\mathbf{x}_{0:k-1} \mid \mathbf{z}_{1:k-1})}{p(\mathbf{z}_k \mid \mathbf{z}_{1:k-1})}$$

Dividing by the factorized proposal distribution yields the recursive importance weight update:

$$w_k^{(i)} \propto w_{k-1}^{(i)} \cdot \frac{p(\mathbf{z}_k \mid \mathbf{x}_k^{(i)}) \, p(\mathbf{x}_k^{(i)} \mid \mathbf{x}_{k-1}^{(i)})}{q(\mathbf{x}_k^{(i)} \mid \mathbf{x}_{k-1}^{(i)}, \mathbf{z}_k)}$$

The simplest and most widely taught choice—known as the **bootstrap filter**—uses the motion transition prior itself as the proposal, setting $q(\mathbf{x}_k \mid \mathbf{x}_{k-1}, \mathbf{z}_k) = p(\mathbf{x}_k \mid \mathbf{x}_{k-1})$. The transition terms cancel, leaving an intuitive update rule:

$$w_k^{(i)} \propto w_{k-1}^{(i)} \cdot p(\mathbf{z}_k \mid \mathbf{x}_k^{(i)})$$

Each particle simply moves according to the motion model and then scales its previous weight by the likelihood of the new sensor observation at its predicted pose: particles that explain $\mathbf{z}_k$ well gain weight, while inconsistent hypotheses fade. In theory, the minimum-variance proposal is $q^*(\mathbf{x}_k \mid \mathbf{x}_{k-1}^{(i)}, \mathbf{z}_k) = p(\mathbf{x}_k \mid \mathbf{x}_{k-1}^{(i)}, \mathbf{z}_k)$, which already incorporates the latest measurement, though in most nonlinear systems it cannot be evaluated or sampled directly.

### 4.3.3 Resampling

Repeating importance updates over many time steps introduces **weight degeneracy**: after several iterations, one or two particles accumulate virtually all the probability mass while the remaining particles carry near-zero weights and contribute nothing to the estimate. We monitor this collapse via the **effective sample size**:

$$N_{\text{eff}} = \frac{1}{\sum_{i=1}^N (w_k^{(i)})^2}$$

When all weights are uniform, $N_{\text{eff}} = N$; when a single particle dominates, $N_{\text{eff}} \to 1$. Whenever $N_{\text{eff}}$ drops below a chosen threshold $N_{\text{threshold}}$ (commonly $N/2$), the filter triggers **resampling**—duplicating high-weight particles, pruning negligible ones, and resetting all weights to $1/N$.

Three resampling schemes are standard in practice:
- **Multinomial resampling**: draws $N$ independent indices with replacement from the discrete distribution defined by the weights. While conceptually simple, independent draws introduce high sampling variance.
- **Systematic resampling**: draws a single uniform offset $U_0 \sim \text{Uniform}(0, 1/N)$ and steps across the cumulative weight distribution at regular intervals $U_i = U_0 + (i-1)/N$. It runs in linear time and typically achieves lower variance than multinomial resampling, though it does not guarantee minimum variance across every possible weight configuration.
- **Stratified resampling**: partitions $[0, 1)$ into $N$ equal strata like systematic resampling, but draws an independent uniform sample within each stratum rather than sharing one global offset.

The Python implementation below puts these pieces together into a **Sampling Importance Resampling (SIR)** bootstrap particle filter using systematic resampling whenever $N_{\text{eff}}$ falls below $N/2$.

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

Pure particle filters struggle as state dimension grows, because covering a high-dimensional space with random samples quickly becomes prohibitive. The **Rao-Blackwellized Particle Filter (RBPF)** mitigates this by partitioning the state: sample only the low-dimensional, strongly nonlinear variables with particles, and track the remaining conditionally Gaussian variables analytically with Kalman filters.

Suppose the state splits into $\mathbf{x} = [\mathbf{x}_1, \mathbf{x}_2]$ such that, conditioned on $\mathbf{x}_1$, the posterior over $\mathbf{x}_2$ is analytically tractable. By the chain rule of probability:

$$p(\mathbf{x}_1, \mathbf{x}_2 \mid \mathbf{z}_{1:k}) = p(\mathbf{x}_2 \mid \mathbf{x}_1, \mathbf{z}_{1:k}) \cdot p(\mathbf{x}_1 \mid \mathbf{z}_{1:k})$$

Here $p(\mathbf{x}_1 \mid \mathbf{z}_{1:k})$ is approximated by the particle cloud, while $p(\mathbf{x}_2 \mid \mathbf{x}_1, \mathbf{z}_{1:k})$ is maintained analytically by a bank of Kalman filters attached to each particle. By the **Rao-Blackwell theorem**, marginalizing out the tractable subspace analytically is guaranteed never to increase estimator variance relative to sampling the joint space:

$$\text{Var}[\hat{\mathbf{x}}_{\text{RBPF}}] \leq \text{Var}[\hat{\mathbf{x}}_{\text{PF}}]$$

**Connection to FastSLAM**: FastSLAM is the classic robotic application of an RBPF. It uses particles to represent the robot's trajectory $\mathbf{x}_1 = \mathbf{x}_{0:k}^{\text{robot}}$ and tracks the map of $M$ landmarks $\mathbf{x}_2 = \{\mathbf{m}_1, \ldots, \mathbf{m}_M\}$ with $M$ independent 2D EKFs per particle. Why does the map factor cleanly? Conditioned on a specific robot trajectory, observations of distinct landmarks become conditionally independent, eliminating the cross-correlations that force a single EKF-SLAM filter to maintain one giant dense covariance matrix. With balanced-tree map storage and known data association, FastSLAM updates a single landmark observation in $O(N \log M)$ time for $N$ particles and $M$ landmarks, compared with $O(M^2)$ for a standard EKF-SLAM update.

### 4.3.5 Limitations of the Particle Filter and Its Current Role

Even with Rao-Blackwellization, the particle filter remains constrained by the **curse of dimensionality**: in general high-dimensional state spaces, the number of particles needed to avoid sample impoverishment can grow explosively. Because visual-inertial and LiDAR-inertial odometry states couple 3D pose, linear velocity, and accelerometer/gyroscope biases—often 15 dimensions or more before adding landmarks—running a full-state particle filter in real time is rarely practical without an unusually tight proposal or structural factorization.

Consequently, modern robotics deploys particle filters selectively where their nonparametric flexibility shines:
- **2D RBPF SLAM**: Occupancy-grid mapping pipelines such as GMapping track only the 3-DoF planar pose with particles while attaching a 2D grid map to each particle.
- **Global localization (Monte Carlo Localization, MCL)**: When a robot boots up inside a known map without an initial pose estimate—or must recover from sudden relocation in the kidnapped-robot problem (paired with recovery heuristics such as injecting fresh particles)—a particle cloud handles multimodal ambiguity naturally.
- **Low-dimensional nonlinear tracking**: Specialized tasks with small state vectors and severe non-Gaussianity or multimodality.

High-dimensional 3D navigation, by contrast, is dominated by the Kalman filter family (especially the ESKF) and factor-graph optimization.

Yet every sequential filter—whether Kalman-based or particle-based—shares a structural habit: as time marches forward, past states are either marginalized out immediately or frozen at the linearization points chosen when their measurements first arrived. To understand why modern SLAM largely moved beyond pure filtering, we need to contrast **filtering** with **smoothing**.

---

## 4.4 Filtering and Smoothing

### 4.4.1 The Difference Between Filtering and Smoothing

The distinction between filtering and smoothing comes down to which measurements are allowed to inform an estimate:

- **Filtering** estimates only the *current* state given observations received up to the present:
$$p(\mathbf{x}_k \mid \mathbf{z}_{1:k})$$

- **Smoothing** estimates *past* (or all) states given the full measurement sequence—including future observations collected after time $k$:
$$p(\mathbf{x}_k \mid \mathbf{z}_{1:T}), \quad k < T$$

Under a well-specified probabilistic model and identical loss function, conditioning on future measurements can reduce a smoother's expected Bayes risk, though this theoretical advantage does not guarantee lower realized error on every individual trajectory or for every approximate solver. In practice, real-time systems rely on sequential filters alongside incremental and fixed-lag smoothers, while full batch smoothers remain standard for offline mapping and post-processing.

### 4.4.2 Fixed-Lag Smoother

Between single-step filtering and full-trajectory smoothing lies the **fixed-lag smoother**, which estimates the state $L$ steps behind the current time using all observations up to time $k$:

$$p(\mathbf{x}_{k-L} \mid \mathbf{z}_{1:k})$$

By tolerating a bounded lag $L$, the estimator can revisit recent poses as new measurements tighten geometric constraints before older states leave the window. Sliding-window nonlinear optimizers such as VINS-Mono and ORB-SLAM3 operate in much the same spirit, jointly relinearizing keyframes inside an active window; as with any architectural choice, rigorous accuracy comparisons against filters require holding the compute budget, sensor model, and dataset fixed.

### 4.4.3 Full Smoothing (Batch Optimization)

Pushing the window to the entire mission yields **full smoothing (batch optimization)**, which solves for the entire state history simultaneously given all recorded measurements:

$$p(\mathbf{x}_{0:T} \mid \mathbf{z}_{1:T})$$

Framing this inference as Maximum A Posteriori (MAP) estimation seeks the trajectory that maximizes the joint posterior:

$$\mathbf{x}_{0:T}^* = \arg\max_{\mathbf{x}_{0:T}} p(\mathbf{x}_{0:T} \mid \mathbf{z}_{1:T})$$

When sensor noise is modeled as Gaussian, maximizing this joint posterior turns into a **Nonlinear Least Squares (NLS)** problem—the exact mathematical starting point for factor-graph optimization.

### 4.4.4 Why Modern SLAM Moved From Filtering To Optimization

Through the early 2000s, EKF-SLAM was the dominant paradigm in mobile robotics. Over the following decade, however, the field shifted decisively toward graph-based optimization (smoothing) for four fundamental reasons:

**1. The linearization-point problem**

A standard EKF linearizes once and never looks back: the Jacobian at time $k$ is evaluated at the time-$k$ estimate and baked into the covariance forever, even after later observations reveal a much better trajectory estimate. By contrast, batch and sliding-window optimizers can repeatedly relinearize every factor along the trajectory around the latest iterate. In their influential analysis ["Visual SLAM: Why Filter?" (2012)](https://doi.org/10.1016/j.imavis.2012.02.009), Strasdat et al. showed that under an equal computational budget, tracking more visual features improves accuracy more effectively than keeping more frames—and bundle adjustment, whose cost scales linearly in the number of features, exploits that tradeoff far more efficiently than EKF-SLAM, whose dense update scales cubically.

**2. Consistency issues**

Without an external absolute reference, SLAM exhibits **gauge freedoms**: shifting or rotating the global reference frame leaves all relative observations unchanged. The exact null-space dimension depends on the problem—2D SLAM, 3D SLAM, and visual-inertial odometry differ, with VIO typically having four unobservable directions (three global translations and yaw about gravity). While implementations choose a gauge by anchoring the first pose or selected degrees of freedom, an EKF that relinearizes at shifting estimates along the trajectory can inadvertently inject spurious information into these unobservable directions, causing overconfident covariances and filter inconsistency. Techniques such as First-Estimate Jacobian (FEJ) and observability-constrained filtering were specifically invented to preserve that null space more faithfully.

**3. Scalability**

In EKF-SLAM with $M$ landmarks, maintaining all pairwise state–landmark correlations produces a dense covariance matrix of size $O((n + 3M)^2)$, making each update cost $O((n+3M)^2)$—rapidly intractable as a map grows. In graph-based SLAM, by contrast, the information matrix (Hessian) is **sparse**, because variables only couple when they share a factor. Exploiting that sparsity often slashes runtime compared to a dense filter update, though actual computational cost still depends on graph topology, variable elimination ordering, and the fill-in created by loop closures.

**4. Natural handling of loop closures**

In a filter, closing a loop back to a long-past pose requires retaining correlations with historical states or performing awkward state augmentation. In a factor graph, a loop closure is nothing more than adding a new relative-pose constraint connecting two existing nodes and re-solving the graph.

| Aspect | Filtering (EKF) | Optimization (Graph) |
|------|----------------|---------------------|
| Linearization | A basic EKF linearizes once per update | Batch/smoothing methods can relinearize repeatedly |
| Past states | A basic filter removes them, but state augmentation is possible | Full smoothing retains them; fixed-lag smoothing marginalizes them |
| Loop closure | Requires retaining past state and correlations | Add a factor and update the affected region |
| Matrix structure | Covariance is usually dense | Factor/Hessian sparsity can be exploited |
| Cost | Depends on state and observation structure | Depends on graph topology, ordering, and fill-in |
| Consistency | Requires care with linearization and observability | Relinearization alone is insufficient; gauge and marginalization still matter |

That is not to say filtering has vanished. Bounded-compute filters such as [MSCKF (Mourikis & Roumeliotis, 2007)](https://ieeexplore.ieee.org/document/4209642) and OpenVINS remain workhorses on resource-constrained aerial platforms, and FAST-LIO2 achieves remarkable throughput by pairing a manifold IEKF with an incremental `ikd-tree`. Any fair accuracy comparison between filtering and optimization must therefore keep the dataset, sensor models, and compute budget matched.

To turn the smoothing formulation into practical real-time software, however, we need a clean representation that exposes factor sparsity and supports Lie-group manifolds directly. That representation is the **factor graph**.

---

## 4.5 Factor Graph & Optimization

### 4.5.1 Factor Graph Representation

To turn full or fixed-lag smoothing into a clean, modular solver, we encode the joint probability distribution as a **factor graph**—a bipartite graph linking *variable nodes* (unknown states to be estimated) with *factor nodes* (probabilistic constraints imposed by sensor measurements or priors). Under conditional independence across measurements, the joint posterior factors into a product of local potentials:

$$p(\mathbf{X} \mid \mathbf{Z}) \propto \prod_{i} f_i(\mathbf{X}_i)$$

Here $\mathbf{X} = \{\mathbf{x}_0, \mathbf{x}_1, \ldots, \mathbf{x}_T, \mathbf{l}_1, \ldots, \mathbf{l}_M\}$ collects all state variables—robot poses, velocities, biases, and landmarks—$\mathbf{Z}$ denotes the full measurement set, and each factor $f_i(\mathbf{X}_i)$ expresses a local probabilistic constraint over the small subset of variables $\mathbf{X}_i$ involved in that observation.

Different sensors and priors map directly onto distinct factor types:

| Factor type | Connected variables | Meaning |
|------------|-------------|------|
| Prior factor | $\mathbf{x}_0$ | Prior on the initial state |
| Odometry factor | $\mathbf{x}_{k-1}, \mathbf{x}_k$ | Relative motion between consecutive poses |
| IMU preintegration factor | $\mathbf{x}_{i}, \mathbf{x}_{j}, \mathbf{v}_i, \mathbf{v}_j, \mathbf{b}_i$ | IMU integration between keyframes |
| Vision factor | $\mathbf{x}_k, \mathbf{l}_m$ | Camera observation of a landmark |
| LiDAR factor | $\mathbf{x}_k$ | Point-to-plane/point-to-edge registration |
| GPS factor | $\mathbf{x}_k$ | Absolute-position observation |
| Loop closure factor | $\mathbf{x}_i, \mathbf{x}_j$ | Loop-closure relative pose |

Because every sensor enters strictly as an independent factor node, factor graphs are inherently **modular**: fusing an additional modality requires defining a new factor type and wiring it into the relevant variables, without altering the rest of the graph.

### 4.5.2 MAP Inference = Nonlinear Least Squares

When measurement noise is Gaussian, each factor takes an exponential quadratic form:

$$f_i(\mathbf{X}_i) \propto \exp\left(-\frac{1}{2} \|\mathbf{r}_i(\mathbf{X}_i)\|^2_{\boldsymbol{\Sigma}_i}\right)$$

where $\mathbf{r}_i(\mathbf{X}_i)$ is the factor's residual vector and $\|\mathbf{r}\|^2_{\boldsymbol{\Sigma}} = \mathbf{r}^\top \boldsymbol{\Sigma}^{-1} \mathbf{r}$ denotes the squared Mahalanobis distance weighted by sensor precision. For a typical observation factor, the residual compares the measurement against the predicted observation:
$$\mathbf{r}_i = \mathbf{z}_i - h_i(\mathbf{X}_i), \quad \boldsymbol{\Sigma}_i = \mathbf{R}_i \text{ (observation noise covariance)}$$

Seeking the Maximum A Posteriori (MAP) trajectory means maximizing the product of all factors:

$$\mathbf{X}^* = \arg\max_\mathbf{X} p(\mathbf{X} \mid \mathbf{Z}) = \arg\max_\mathbf{X} \prod_i f_i(\mathbf{X}_i)$$

Taking the negative logarithm turns that product of Gaussians into a sum of squared residuals:

$$\boxed{\mathbf{X}^* = \arg\min_\mathbf{X} \sum_i \|\mathbf{r}_i(\mathbf{X}_i)\|^2_{\boldsymbol{\Sigma}_i}}$$

Probabilistic MAP inference has therefore been converted into a **Nonlinear Least Squares (NLS)** optimization problem. For a comprehensive 139-page treatment of this duality and its linear-algebraic machinery, see [Dellaert & Kaess (2017), "Factor Graphs for Robot Perception"](https://doi.org/10.1561/2300000043).

### 4.5.3 Gauss-Newton Method

Because the residuals $\mathbf{r}_i$ are generally nonlinear, we solve the NLS objective iteratively using the **Gauss-Newton (GN)** method. At each iteration, we linearize every residual to first order around the current estimate $\mathbf{X}^{(0)}$:

$$\mathbf{r}_i(\mathbf{X}^{(0)} \boxplus \Delta\mathbf{X}) \approx \mathbf{r}_i(\mathbf{X}^{(0)}) + \mathbf{J}_i \Delta\mathbf{X}$$

where $\mathbf{J}_i = \frac{\partial \mathbf{r}_i}{\partial \mathbf{X}}\big|_{\mathbf{X}^{(0)}}$ is the residual Jacobian and $\boxplus$ denotes the manifold update operator. Substituting this linear approximation back into the cost function gives a quadratic subproblem in $\Delta\mathbf{X}$:

$$\sum_i \|\mathbf{r}_i + \mathbf{J}_i \Delta\mathbf{X}\|^2_{\boldsymbol{\Sigma}_i}$$

Whitening each residual and Jacobian by its inverse square-root covariance—$\mathbf{r}_i' = \boldsymbol{\Sigma}_i^{-1/2} \mathbf{r}_i$ and $\mathbf{J}_i' = \boldsymbol{\Sigma}_i^{-1/2} \mathbf{J}_i$—reduces the weighted objective to an unweighted least-squares problem:

$$\sum_i \|\mathbf{r}_i' + \mathbf{J}_i' \Delta\mathbf{X}\|^2$$

Differentiating with respect to $\Delta\mathbf{X}$ and setting the gradient to zero yields the **normal equation**:

$$\underbrace{\left(\sum_i \mathbf{J}_i'^\top \mathbf{J}_i'\right)}_{\mathbf{H}} \Delta\mathbf{X} = -\underbrace{\sum_i \mathbf{J}_i'^\top \mathbf{r}_i'}_{\mathbf{b}}$$

$$\boxed{\mathbf{H} \Delta\mathbf{X} = -\mathbf{b}}$$

Here $\mathbf{H} = \mathbf{J}^\top \boldsymbol{\Sigma}^{-1} \mathbf{J} \in \mathbb{R}^{N \times N}$ is the Gauss-Newton approximate Hessian (information matrix) and $\mathbf{b} = \mathbf{J}^\top \boldsymbol{\Sigma}^{-1} \mathbf{r}$ is the gradient vector.

Crucially, in SLAM factor graphs, $\mathbf{H}$ is **sparse**. Each factor's Jacobian $\mathbf{J}_i$ is nonzero only in the columns belonging to the variables $\mathbf{X}_i$ that factor touches, so an off-diagonal block $(a, b)$ in $\mathbf{H}$ is nonzero only if variables $a$ and $b$ co-occur in at least one factor. Since each observation connects only a handful of poses or landmarks, almost every block in $\mathbf{H}$ is zero.

Solving the normal equation yields the update $\mathbf{X}^{(k+1)} = \mathbf{X}^{(k)} \boxplus \Delta\mathbf{X}^{(k)}$, after which we relinearize and repeat until convergence:

$$\mathbf{X}^{(k+1)} = \mathbf{X}^{(k)} \boxplus \Delta\mathbf{X}^{(k)}$$

To solve $\mathbf{H}\Delta\mathbf{X} = -\mathbf{b}$ without destroying sparsity, solvers apply **sparse Cholesky decomposition** ($\mathbf{H} = \mathbf{L}\mathbf{L}^\top$ followed by forward and back substitution). Because the amount of nonzero fill-in inside $\mathbf{L}$ depends heavily on elimination order, solvers precede factorization with a fill-reducing variable permutation such as approximate minimum-degree ordering (COLAMD).

### 4.5.4 Levenberg-Marquardt Method

Because Gauss-Newton discards second-order residual derivatives and approximates the Hessian solely with $\mathbf{J}^\top \boldsymbol{\Sigma}^{-1} \mathbf{J}$, it can overshoot or diverge when initialized far from the solution or under strong nonlinearity. The **Levenberg-Marquardt (LM)** algorithm stabilizes optimization by adding an adaptive damping term to the normal equation:

$$(\mathbf{H} + \lambda \mathbf{I}) \Delta\mathbf{X} = -\mathbf{b}$$

Tuning $\lambda$ smoothly interpolates between two regimes: small $\lambda$ recovers fast, quadratic-like Gauss-Newton steps near the minimum, whereas large $\lambda$ shrinks the step and aligns it with steepest gradient descent for safety. At each step, if a candidate update lowers the cost, the solver accepts it and reduces $\lambda$ toward Gauss-Newton mode; if the cost rises, the solver rejects the step, increases $\lambda$, and tries a more conservative step.

### 4.5.5 Optimization on Manifolds

Just as in the ESKF, robot poses $\mathbf{T} \in SE(3)$ do not live in a flat Euclidean vector space—they evolve on a curved Lie-group manifold where component-wise matrix addition breaks orthogonality and determinant constraints. Instead of adding updates directly to rotation matrices or quaternions, manifold optimization parameterizes local perturbations $\boldsymbol{\xi} \in \mathbb{R}^6$ in the flat **tangent space** (Lie algebra) around the current estimate $\mathbf{T}^{(k)}$ and maps them back onto $SE(3)$ through a **retraction** or the **exponential map**:

$$\mathbf{T}^{(k+1)} = \mathbf{T}^{(k)} \cdot \text{Exp}(\boldsymbol{\xi})$$

or equivalently using a left perturbation,

$$\mathbf{T}^{(k+1)} = \text{Exp}(\boldsymbol{\xi}) \cdot \mathbf{T}^{(k)}$$

depending on whether the solver adopts a right- or left-increment convention. Here $\text{Exp}: \mathbb{R}^6 \to SE(3)$ maps a 6-DoF twist $\boldsymbol{\xi} = [\boldsymbol{\rho}^\top, \boldsymbol{\phi}^\top]^\top$—comprising translation $\boldsymbol{\rho} \in \mathbb{R}^3$ and angle-axis rotation $\boldsymbol{\phi} \in \mathbb{R}^3$—onto a rigid transform, while its inverse $\text{Log}: SE(3) \to \mathbb{R}^6$ maps relative transforms back to the tangent space. On $SO(3)$, the exponential map is given in closed form by **Rodrigues' formula**:

$$\text{Exp}(\boldsymbol{\phi}) = \mathbf{I} + \frac{\sin\theta}{\theta}[\boldsymbol{\phi}]_\times + \frac{1 - \cos\theta}{\theta^2}[\boldsymbol{\phi}]_\times^2 \in \mathbb{R}^{3 \times 3}$$

where $\theta = \|\boldsymbol{\phi}\|$. At each Gauss-Newton or LM iteration, we compute residual Jacobians directly with respect to the minimal tangent perturbation at zero:

$$\mathbf{J}_i = \frac{\partial \mathbf{r}_i}{\partial \boldsymbol{\xi}}\bigg|_{\boldsymbol{\xi}=\mathbf{0}}$$

solve the sparse linear system for the 6-DoF tangent step $\boldsymbol{\xi}$, and retract onto the manifold via $\text{Exp}(\boldsymbol{\xi})$.

### 4.5.6 iSAM2: Incremental Smoothing

Solving the full sparse normal equation from scratch whenever a new keyframe arrives becomes prohibitively slow over long missions. **iSAM2** ([Kaess et al., 2012](https://doi.org/10.1177/0278364911430419)) avoids repeated batch solves by translating the factor graph into a **Bayes tree** and updating only the variables touched by new constraints.

Specifically, iSAM2 combines four mechanisms:
1. **Bayes tree**: Converts the variable-elimination ordering of a factor graph into a directed tree whose nodes store conditional densities over variable cliques.
2. **Incremental update**: When new factors arrive, only the affected cliques near the top of the tree are recomputed, leaving the rest of the solved factorization untouched.
3. **Fluid relinearization**: Instead of running periodic full-graph relinearization passes, iSAM2 tracks linearization drift per variable and relinearizes only those variables whose estimates have moved noticeably.
4. **Local variable reordering**: New variables and factors trigger local reordering confined to the modified subtree rather than a global COLAMD pass over the entire graph.

GTSAM provides the reference implementation of iSAM2, which powers backends such as [LIO-SAM](https://arxiv.org/abs/2007.00258), whereas sliding-window systems like VINS-Mono use Ceres Solver to optimize a bounded local window.

> **Recent trends — continuous-time factor graph**: Active research extends discrete keyframe factor graphs to **continuous-time** trajectories. For example, [Burnett et al. (2024)](https://arxiv.org/abs/2402.06174) employ a Gaussian Process motion prior to fuse asynchronous radar-inertial and LiDAR-inertial measurements inside a single continuous-time factor graph without forcing observations onto discrete keyframe timestamps.

### 4.5.7 GTSAM / Ceres / g2o Comparison

In practice, three C++ optimization libraries underpin most modern SLAM backends, each reflecting a different trade-off between general-purpose NLS solving and incremental probabilistic inference:

| Aspect | GTSAM | Ceres Solver | g2o |
|------|-------|-------------|-----|
| Developer | Georgia Tech ([Dellaert](https://gtsam.org/)) | Google ([Ceres](http://ceres-solver.org/)) | [Kümmerle et al.](https://doi.org/10.1109/ICRA.2011.5979949) |
| Design | Factor graph + Bayes tree | General-purpose NLS solver | Graph optimization |
| Incremental | iSAM2 (native) | None (batch) | None (batch) |
| Manifolds | Built-in (Rot2, Rot3, Pose2, Pose3, ...) | Local parameterization | Built-in |
| IMU preintegration | Built-in (`PreintegratedImuMeasurements`) | User-defined | User-defined |
| Automatic differentiation | Expression-based support (numerical differentiation also available) | Auto-diff (ceres::AutoDiffCostFunction) | AutoDifferentiation support |
| Language | C++ (Python bindings) | C++ | C++ |
| Representative users | LIO-SAM | Cartographer, VINS-Mono | ORB-SLAM3 and pose graphs in many SLAM systems |
| Learning curve | Just define factors | Define cost functions | Define vertices/edges |

To illustrate how a factor-graph backend works in code, the GTSAM example below builds a simple 2D square trajectory with four noisy odometry edges, closes the loop between the final pose and the origin, and solves incrementally with `ISAM2`.

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
    # use accumulated odometry as the initial guess
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

In a real visual-inertial or LiDAR-inertial factor graph, however, odometry factors rarely arrive neatly at keyframe rate. An IMU streams measurements an order of magnitude faster than camera frames or LiDAR scans—raising the question of how to summarize dozens of high-rate inertial samples into a single factor without bogging down the optimizer.

---

## 4.6 IMU Preintegration

### 4.6.1 Why Preintegration Is Needed

An IMU typically outputs linear acceleration and angular velocity at 200 Hz to 1000 Hz, whereas camera and LiDAR keyframes arrive at only 10 Hz to 30 Hz. Between any two consecutive keyframes $i$ and $j$, roughly 7 to 100 raw inertial measurements accumulate.

**Naive approach: direct integration**

Integrating those IMU readings directly in the world frame propagates the navigation state from keyframe $i$ to keyframe $j$:

$$\mathbf{R}_j = \mathbf{R}_i \prod_{k=i}^{j-1} \text{Exp}((\tilde{\boldsymbol{\omega}}_k - \mathbf{b}_g) \Delta t)$$
$$\mathbf{v}_j = \mathbf{v}_i + \mathbf{g} \Delta t_{ij} + \sum_{k=i}^{j-1} \mathbf{R}_k (\tilde{\mathbf{a}}_k - \mathbf{b}_a) \Delta t$$
$$\mathbf{p}_j = \mathbf{p}_i + \sum_{k=i}^{j-1}\left[\mathbf{v}_k \Delta t + \frac{1}{2}\left(\mathbf{R}_k(\tilde{\mathbf{a}}_k - \mathbf{b}_a) + \mathbf{g}\right)\Delta t^2\right]$$

Notice the computational trap hidden inside these equations: every term in the velocity and position sums depends explicitly on the intermediate orientations $\mathbf{R}_k$ and velocities $\mathbf{v}_k$, which in turn depend on the initial navigation state $(\mathbf{R}_i, \mathbf{v}_i, \mathbf{p}_i)$ and biases $(\mathbf{b}_g, \mathbf{b}_a)$. Inside a nonlinear optimizer, $(\mathbf{R}_i, \mathbf{v}_i, \mathbf{p}_i)$ and the biases shift at every Gauss-Newton iteration, forcing the solver to re-integrate all 7–100 IMU steps—and evaluate hundreds of exponential maps—between every pair of keyframes on every single iteration.

**Preintegration's remedy**: Express the integrated motion increment in the *body frame of keyframe $i$* instead of the world frame, cleanly decoupling the integral from keyframe $i$'s global pose and velocity; then handle small updates to the biases through a first-order Jacobian correction so re-integration is rarely needed.

### 4.6.2 On-Manifold Preintegration Derivation

Following [Forster et al. (2017)](https://doi.org/10.1109/TRO.2016.2597321) (also available as [arXiv:1512.02363](https://arxiv.org/abs/1512.02363)), we derive on-manifold IMU preintegration in five logical steps.

#### Step 1: Define relative quantities

To isolate the inertial measurements from the global navigation state, multiply the global propagation equations on the left by $\mathbf{R}_i^\top$ and collect all terms depending on $(\mathbf{R}_i, \mathbf{v}_i, \mathbf{p}_i)$ on the left-hand side:

$$\Delta\mathbf{R}_{ij} \triangleq \mathbf{R}_i^\top \mathbf{R}_j = \prod_{k=i}^{j-1} \text{Exp}((\tilde{\boldsymbol{\omega}}_k - \mathbf{b}_g^i)\Delta t) \in SO(3)$$

$$\Delta\mathbf{v}_{ij} \triangleq \mathbf{R}_i^\top (\mathbf{v}_j - \mathbf{v}_i - \mathbf{g}\Delta t_{ij}) = \sum_{k=i}^{j-1} \Delta\mathbf{R}_{ik}(\tilde{\mathbf{a}}_k - \mathbf{b}_a^i)\Delta t \in \mathbb{R}^3$$

$$\Delta\mathbf{p}_{ij} \triangleq \mathbf{R}_i^\top (\mathbf{p}_j - \mathbf{p}_i - \mathbf{v}_i \Delta t_{ij} - \frac{1}{2}\mathbf{g}\Delta t_{ij}^2) = \sum_{k=i}^{j-1}\left[\Delta\mathbf{v}_{ik}\Delta t + \frac{1}{2}\Delta\mathbf{R}_{ik}(\tilde{\mathbf{a}}_k - \mathbf{b}_a^i)\Delta t^2\right] \in \mathbb{R}^3$$

**The right-hand sides depend solely on the raw IMU measurements and the bias estimates at keyframe $i$, completely independent of the global navigation state $(\mathbf{R}_i, \mathbf{v}_i, \mathbf{p}_i)$.** Whenever the optimizer adjusts keyframe poses or velocities, the preintegrated sums stay untouched.

#### Step 2: Recursive computation (on-manifold)

Because each new IMU measurement arrives sequentially between keyframes, the three relative increments can be accumulated incrementally as samples stream in:

$$\Delta\mathbf{R}_{i,k+1} = \Delta\mathbf{R}_{ik} \cdot \text{Exp}((\tilde{\boldsymbol{\omega}}_k - \mathbf{b}_g)\Delta t) \in SO(3)$$
$$\Delta\mathbf{v}_{i,k+1} = \Delta\mathbf{v}_{ik} + \Delta\mathbf{R}_{ik}(\tilde{\mathbf{a}}_k - \mathbf{b}_a)\Delta t \in \mathbb{R}^3$$
$$\Delta\mathbf{p}_{i,k+1} = \Delta\mathbf{p}_{ik} + \Delta\mathbf{v}_{ik}\Delta t + \frac{1}{2}\Delta\mathbf{R}_{ik}(\tilde{\mathbf{a}}_k - \mathbf{b}_a)\Delta t^2 \in \mathbb{R}^3$$

starting from $\Delta\mathbf{R}_{ii} = \mathbf{I}_{3\times 3}$, $\Delta\mathbf{v}_{ii} = \mathbf{0}$, and $\Delta\mathbf{p}_{ii} = \mathbf{0}$. Saying the scheme is "on-manifold" means $\Delta\mathbf{R}_{ij}$ evolves directly on $SO(3)$ via Lie-group composition (implemented with rotation matrices or unit quaternions) rather than Euler angles or unnormalized matrices.

#### Step 3: First-order correction for bias change

Even though global poses no longer trigger re-integration, the optimizer still refines the bias estimates by $\mathbf{b} \to \mathbf{b} + \delta\mathbf{b}$. Rather than re-summing every sample each time a bias shifts slightly, we propagate a first-order Taylor expansion alongside the measurements:

**Rotation correction**:
$$\Delta\hat{\mathbf{R}}_{ij}(\mathbf{b}_g + \delta\mathbf{b}_g) \approx \Delta\hat{\mathbf{R}}_{ij}(\mathbf{b}_g) \cdot \text{Exp}\left(\frac{\partial \Delta\bar{\mathbf{R}}_{ij}}{\partial \mathbf{b}_g} \delta\mathbf{b}_g\right)$$

**Velocity correction**:
$$\Delta\hat{\mathbf{v}}_{ij}(\mathbf{b} + \delta\mathbf{b}) \approx \Delta\hat{\mathbf{v}}_{ij}(\mathbf{b}) + \frac{\partial \Delta\bar{\mathbf{v}}_{ij}}{\partial \mathbf{b}_g} \delta\mathbf{b}_g + \frac{\partial \Delta\bar{\mathbf{v}}_{ij}}{\partial \mathbf{b}_a} \delta\mathbf{b}_a$$

**Position correction**:
$$\Delta\hat{\mathbf{p}}_{ij}(\mathbf{b} + \delta\mathbf{b}) \approx \Delta\hat{\mathbf{p}}_{ij}(\mathbf{b}) + \frac{\partial \Delta\bar{\mathbf{p}}_{ij}}{\partial \mathbf{b}_g} \delta\mathbf{b}_g + \frac{\partial \Delta\bar{\mathbf{p}}_{ij}}{\partial \mathbf{b}_a} \delta\mathbf{b}_a$$

These five bias Jacobians are updated recursively in the exact same loop as Step 2. For instance, the gyroscope bias Jacobian of the preintegrated rotation evolves via:

$$\frac{\partial \Delta\bar{\mathbf{R}}_{i,k+1}}{\partial \mathbf{b}_g} = \Delta\bar{\mathbf{R}}_{k,k+1}^\top \frac{\partial \Delta\bar{\mathbf{R}}_{ik}}{\partial \mathbf{b}_g} - \text{Jr}((\tilde{\boldsymbol{\omega}}_k - \mathbf{b}_g)\Delta t) \Delta t$$

where $\text{Jr}(\boldsymbol{\phi})$ is the right Jacobian of $SO(3)$:

$$\text{Jr}(\boldsymbol{\phi}) = \mathbf{I} - \frac{1 - \cos\theta}{\theta^2}[\boldsymbol{\phi}]_\times + \frac{\theta - \sin\theta}{\theta^3}[\boldsymbol{\phi}]_\times^2, \quad \theta = \|\boldsymbol{\phi}\|$$

So long as $\|\delta\mathbf{b}\|$ remains small, this linear update stays accurate; if the bias shift or corrected residual crosses an implementation's validity threshold, the estimator falls back to re-integrating the stored raw IMU samples around the new bias estimate.

#### Step 4: Covariance propagation

Raw gyroscope and accelerometer white noise $\mathbf{n}_g, \mathbf{n}_a \sim \mathcal{N}(\mathbf{0}, \boldsymbol{\Sigma})$ also accumulates across the interval. Tracking its $9 \times 9$ uncertainty recursively via

$$\boldsymbol{\Sigma}_{k+1} = \mathbf{A}_k \boldsymbol{\Sigma}_k \mathbf{A}_k^\top + \mathbf{B}_k \boldsymbol{\Sigma}_\eta \mathbf{B}_k^\top$$

where $\mathbf{A}_k$ and $\mathbf{B}_k$ are the linearized error-propagation Jacobians and $\boldsymbol{\Sigma}_\eta$ is the discrete-time IMU noise covariance, yields the preintegrated covariance $\boldsymbol{\Sigma}_{ij}$ whose inverse weights the factor inside the graph.

#### Step 5: IMU Factor in the Factor Graph

With relative increments, bias Jacobians, and covariance in place, the entire inertial interval between keyframes $i$ and $j$ collapses into a single factor comparing predicted and preintegrated increments:

$$\mathbf{r}_{\Delta\mathbf{R}_{ij}} = \text{Log}\left(\Delta\hat{\mathbf{R}}_{ij}^\top \mathbf{R}_i^\top \mathbf{R}_j\right) \in \mathbb{R}^3$$

$$\mathbf{r}_{\Delta\mathbf{v}_{ij}} = \mathbf{R}_i^\top(\mathbf{v}_j - \mathbf{v}_i - \mathbf{g}\Delta t_{ij}) - \Delta\hat{\mathbf{v}}_{ij} \in \mathbb{R}^3$$

$$\mathbf{r}_{\Delta\mathbf{p}_{ij}} = \mathbf{R}_i^\top(\mathbf{p}_j - \mathbf{p}_i - \mathbf{v}_i\Delta t_{ij} - \frac{1}{2}\mathbf{g}\Delta t_{ij}^2) - \Delta\hat{\mathbf{p}}_{ij} \in \mathbb{R}^3$$

Concatenating these three vectors forms a 9-dimensional residual whose squared Mahalanobis norm $\|\mathbf{r}\|^2_{\boldsymbol{\Sigma}_{ij}}$ enters the factor-graph objective directly.

### 4.6.3 Preintegration in Code

The code below translates the full Forster et al. (2017) pipeline into Python: Lie-group primitives (`skew`, `exp_so3`, `log_so3`, `right_jacobian_so3`), an `IMUPreintegration` class that recursively maintains relative motion, bias Jacobians, and the $9 \times 9$ covariance, and a brief GTSAM snippet showing how `PreintegratedImuMeasurements` and `ImuFactor` wire into a factor graph.

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
        self.d_R_d_bg = dR.T @ self.d_R_d_bg - Jr * dt   # the new term is not multiplied by dR.T
        
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
        # continuous-time noise density -> discrete-time covariance: density^2/dt (B already carries a factor of dt)
        Sigma_eta[0:3, 0:3] = np.eye(3) * self.sigma_g**2 / dt
        Sigma_eta[3:6, 3:6] = np.eye(3) * self.sigma_a**2 / dt
        
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
    imu_params = gtsam.PreintegrationParams.MakeSharedU(9.81)  # for a Z-up frame; the internal gravity vector is (0, 0, -9.81)
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
    # ImuFactor connects pose and velocity at i, j and the bias at i
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

With IMU preintegration summarizing dozens of inertial readings into a compact keyframe-to-keyframe factor, our factor graph stays lean over short horizons. Over a long mission, however, inserting new keyframes forever still pushes optimization cost upward—requiring a principled way to retire old states without forgetting what they taught us.

---

## 4.7 Marginalization & Sliding Window

### 4.7.1 Why Marginalization Is Needed

Every new keyframe in a factor-graph SLAM or VIO pipeline adds new poses, velocities, biases, and landmark factors. Left unchecked, the graph expands indefinitely until real-time optimization misses its frame budget. A **sliding window** caps compute by keeping only the latest $N$ keyframes and retiring older ones. Yet simply deleting an old keyframe—along with every factor attached to it—throws away hard-won information about scale, velocity, and landmark geometry, degrading accuracy.

**Marginalization** solves this cleanly: it eliminates old variables algebraically while summarizing all constraints they carried into an equivalent **prior factor** over the remaining variables in the window.

### 4.7.2 Schur Complement

Algebraically, marginalizing Gaussian variables out of a linearized factor graph reduces to taking the **Schur complement** of the information matrix. Partition the state increment into variables to marginalize ($\mathbf{x}_m$) and variables to retain ($\mathbf{x}_r$), writing the normal equation as $\mathbf{H}\Delta\mathbf{x} = \mathbf{b}$ (redefining $-\mathbf{b}$ from §4.5.3 as $\mathbf{b} \triangleq -\mathbf{J}^\top \boldsymbol{\Sigma}^{-1} \mathbf{r}$):

$$\mathbf{H} \Delta\mathbf{x} = \mathbf{b}$$

$$\begin{bmatrix} \mathbf{H}_{mm} & \mathbf{H}_{mr} \\ \mathbf{H}_{rm} & \mathbf{H}_{rr} \end{bmatrix} \begin{bmatrix} \Delta\mathbf{x}_m \\ \Delta\mathbf{x}_r \end{bmatrix} = \begin{bmatrix} \mathbf{b}_m \\ \mathbf{b}_r \end{bmatrix}$$

Here $\mathbf{H}_{mm} \in \mathbb{R}^{n_m \times n_m}$ captures information among the variables being removed, $\mathbf{H}_{rr} \in \mathbb{R}^{n_r \times n_r}$ captures direct information among the retained variables, and $\mathbf{H}_{mr} \in \mathbb{R}^{n_m \times n_r}$ encodes their cross-couplings. Solving the top block row for $\Delta\mathbf{x}_m$ and substituting into the bottom row eliminates $\Delta\mathbf{x}_m$ cleanly:

$$\underbrace{(\mathbf{H}_{rr} - \mathbf{H}_{rm} \mathbf{H}_{mm}^{-1} \mathbf{H}_{mr})}_{\mathbf{H}^*} \Delta\mathbf{x}_r = \underbrace{\mathbf{b}_r - \mathbf{H}_{rm} \mathbf{H}_{mm}^{-1} \mathbf{b}_m}_{\mathbf{b}^*}$$

The matrix $\mathbf{H}^* = \mathbf{H}_{rr} - \mathbf{H}_{rm} \mathbf{H}_{mm}^{-1} \mathbf{H}_{mr}$ is the **Schur complement**, which serves as the information matrix of the new prior factor tying together the retained variables.

Notice what happens structurally: any two retained variables in $\mathbf{x}_r$ that previously interacted only indirectly through $\mathbf{x}_m$ now become directly coupled in $\mathbf{H}^*$. This **fill-in** makes $\mathbf{H}^*$ denser than $\mathbf{H}_{rr}$, converting indirect chains into explicit pairwise links. The short script below demonstrates this effect directly on a 3-pose chain where marginalizing `x0` propagates information into the surviving states.

```python
import numpy as np

def marginalize(H, b, indices_to_marginalize, indices_to_keep):
    """Variable marginalization via the Schur complement.
    
    Parameters
    ----------
    H : ndarray, shape (N, N) — information matrix (Hessian)
    b : ndarray, shape (N,) — negative gradient vector
    indices_to_marginalize : list of int — indices of variables to remove
    indices_to_keep : list of int — indices of variables to retain
    
    Returns
    -------
    H_star : ndarray — information matrix after marginalization
    b_star : ndarray — negative gradient after marginalization
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

Marginalization carries a subtle consistency trap. Because the Schur complement prior $(\mathbf{H}^*, \mathbf{b}^*)$ is baked from linearizations evaluated at the moment $\mathbf{x}_m$ is eliminated, its linearization point for the remaining variables is **frozen** from that step onward. On subsequent iterations, however, new measurements pull the retained variables to updated estimates, creating a mismatch between the linearization point locked inside the marginalization prior and the one used by active factors in the window.

That mismatch can silently break **consistency**: the null spaces of the old and new Jacobians no longer align, causing the estimator to hallucinate information along genuinely unobservable gauge directions (such as global position and yaw about gravity in VIO) and report overconfident covariances.

The **First-Estimate Jacobian (FEJ)** strategy closes this gap by forcing every Jacobian linked to marginalization to be evaluated at each variable's **initial estimate** $\mathbf{x}^{(0)}$—the value when that variable was first introduced—and holding that linearization point fixed even as the state estimate improves:

$$\mathbf{J}_{\text{FEJ}} = \left.\frac{\partial \mathbf{r}}{\partial \mathbf{x}}\right|_{\mathbf{x}^{(0)}}$$

By keeping linearization points consistent across the frozen prior and current factors, FEJ suppresses spurious information gain along unobservable directions and plays a central role in MSCKF and OpenVINS ([Li & Mourikis, 2013](https://doi.org/10.1177/0278364913481251)). The trade-off is practical: if the initial estimate $\mathbf{x}^{(0)}$ is poor, locking the Jacobian there can slow convergence, and storing a separate first estimate alongside the current iterate adds bookkeeping complexity.

### 4.7.4 Practical Issues of Sliding-Window Implementation

Turning Schur-complement marginalization into a robust sliding-window estimator requires navigating four implementation details:

#### Issue 1: Which keyframe to marginalize

Dropping the wrong frame can starve triangulation of parallax. VINS-Mono therefore switches between two rules depending on motion: if the latest frame qualifies as a keyframe, it marginalizes the oldest keyframe so the window retains a wide spatial baseline; if the latest frame is not a keyframe, it marginalizes the immediately preceding non-keyframe instead, discarding redundant visual observations while forwarding the IMU preintegration factor onto the neighboring keyframe.

#### Issue 2: Densification due to fill-in

Repeatedly taking Schur complements links ever more retained variables together inside the prior factor, turning a sparse information matrix into a dense block that slows down every solve. Practical systems counter this by capping the size of the prior (limiting how many variables remain connected), ordering marginalization to minimize fill-in, or selectively dropping weak factors—accepting minor information loss for bounded runtime. (By contrast, FAST-LIO2's local-map point deletion operates on the `ikd-tree` map rather than a Hessian prior.)

#### Issue 3: Biases and marginalization

Because IMU biases evolve slowly across the entire trajectory, marginalizing a bias alongside an outgoing keyframe bakes information about that bias into the prior and can stiffen later bias adaptation. In practice, systems like VINS-Mono marginalize the oldest keyframe's pose, velocity, and biases together when sliding the window, while keeping the remaining keyframes' biases active and linked through the prior.

#### Issue 4: Numerical stability

Computing the Schur complement requires inverting $\mathbf{H}_{mm}$, which can become ill-conditioned when some eliminated variables (such as newly initialized or poorly constrained landmarks) carry near-zero information. To keep the solve well-behaved, implementations add a small diagonal regularizer $(\mathbf{H}_{mm} + \epsilon \mathbf{I})^{-1}$, prefer LDL decomposition over explicit matrix inversion, and verify after marginalization that $\mathbf{H}^*$ remains positive semi-definite—projecting onto the nearest PSD matrix if floating-point roundoff introduces tiny negative eigenvalues.

---

## Summary

Looking back across the chapter, the major state-estimation paradigms fit together as progressive trade-offs between expressiveness, accuracy, and compute:

1. **Bayesian Filtering Framework**: Sequential filters alternate prediction via the Chapman-Kolmogorov integral with measurement updates via Bayes' rule, relying on Markov assumptions to keep history bounded; outside linear-Gaussian models, exact integration is intractable and requires approximation. Batch smoothing and factor graphs address related probabilistic models using a different, multi-step computational structure.
2. **Kalman Filter family**: Gaussian filters track beliefs via mean and covariance—an exact MMSE/MAP/BLUE solution in linear-Gaussian systems. The EKF handles mild nonlinearity via first-order Taylor expansion; the ESKF moves linearization into a minimal tangent-space error state that respects $SO(3)$ manifolds cleanly for IMU fusion; the UKF propagates deterministic sigma points instead of deriving Jacobians; and the IEKF relinearizes iteratively inside the update step, bridging directly toward Gauss-Newton optimization.
3. **Particle Filter**: Sequential Monte Carlo methods drop parametric Gaussian assumptions to capture strong nonlinearities and multimodal distributions, yet the curse of dimensionality limits full-state particle filtering in high-dimensional 3D navigation. Rao-Blackwellized formulations (such as FastSLAM) ease this scaling by analytically marginalizing conditionally Gaussian landmarks, leaving particle filters well suited to 2D occupancy-grid SLAM, global localization (MCL), and low-dimensional multimodal tasks.
4. **Shift from Filtering to Optimization**: Modern SLAM largely moved from single-step EKF-SLAM to graph-based smoothing to gain iterative relinearization, Hessian sparsity, and seamless loop-closure constraints. Even so, bounded-compute filters such as MSCKF, OpenVINS, and FAST-LIO2 remain competitive when compute budgets and latency constraints demand them.
5. **Factor Graphs on Manifolds**: Factor graphs express MAP inference as a modular Nonlinear Least Squares problem solved via Gauss-Newton or Levenberg-Marquardt updates in Lie-group tangent spaces, while iSAM2 exploits the Bayes tree for real-time incremental smoothing.
6. **IMU Preintegration**: Transforming high-rate inertial integration into the starting keyframe's body frame decouples preintegrated factors from global poses during optimization, using first-order bias Jacobians for small bias adjustments and re-integrating raw samples only when shifts exceed a threshold.
7. **Marginalization & FEJ**: Sliding-window estimators bound compute by condensing outgoing keyframes into a Schur-complement prior, while First-Estimate Jacobians keep linearization points aligned across the prior and active window to preserve unobservable gauge directions.

This estimation machinery drives the VIO, LIO, and multi-sensor fusion architectures developed later in the book (→ Chapters 6–8). Before any filter or factor graph can evaluate a measurement residual $\mathbf{z}_k - h(\mathbf{x}_k)$, however, the front-end must decide *which* observation corresponds to *which* physical point or landmark in the scene—the correspondence and data-association problem we examine next in Chapter 5.

> **2024–2025 research directions**: Equivariant Filters and Invariant EKFs exploit Lie-group symmetries to tighten error dynamics, consistency, and convergence guarantees for specific system classes. Continuous-time factor graphs backed by Gaussian-process motion priors provide a natural framework for fusing asynchronous multi-sensor streams. Meanwhile, learned estimators such as [AI-Aided Kalman Filters (Shlezinger et al., 2024)](https://arxiv.org/abs/2410.12289) replace or augment gains and system models with neural components, though generalization under distribution shift and safety validation remain open challenges.
