# Ch.2 — Sensor Modeling

센서가 세상을 어떻게 "보는지"는 수학적 관측 모델로 정의한다. 이 모델은 Ch.4의 칼만 필터와 팩터 그래프에서 관측 함수 $h(\mathbf{x})$로 직접 쓰인다.

> robotics-practice Ch.2의 개략적 소개에서 한 단계 더 들어가 **노이즈 모델과 수학적 관측 모델**을 본다. 퓨전 알고리즘을 설계하려면 측정값과 실제 물리량의 수학적 관계, 오차 분포를 알아야 한다.

---

## 2.1 카메라 관측 모델

카메라는 3D 세계의 점을 2D 이미지 평면에 투영하는 센서이다. 이 투영 과정을 수학으로 나타낸 것이 카메라 관측 모델이다.

### 2.1.1 핀홀 카메라 모델 (Pinhole Camera Model)

핀홀 카메라 모델은 카메라의 가장 기본적인 수학적 모델이다. 광학 중심(optical center)을 통과하는 직선으로 3D 점이 이미지 평면에 투영된다고 가정한다.

카메라 좌표계에서 3D 점 $\mathbf{P}_c = [X_c, Y_c, Z_c]^\top$의 이미지 평면 위의 투영점 $\mathbf{p} = [u, v]^\top$은 다음과 같이 계산된다:

$$\begin{bmatrix} u \\ v \\ 1 \end{bmatrix} = \frac{1}{Z_c} \mathbf{K} \begin{bmatrix} X_c \\ Y_c \\ Z_c \end{bmatrix}$$

여기서 $\mathbf{K}$는 카메라 내부 파라미터 행렬(intrinsic matrix)이다:

$$\mathbf{K} = \begin{bmatrix} f_x & 0 & c_x \\ 0 & f_y & c_y \\ 0 & 0 & 1 \end{bmatrix}$$

각 파라미터의 의미:
- $f_x, f_y$: 초점 거리(focal length). 물리적 초점 거리 $f$를 픽셀 크기 $(\Delta x, \Delta y)$로 나눈 값이다. $f_x = f / \Delta x$, $f_y = f / \Delta y$. 정사각형 픽셀($\Delta x = \Delta y$)에서는 $f_x = f_y$이지만, 직사각형 픽셀에서는 두 값이 다르다.
- $c_x, c_y$: 주점(principal point). 광축(optical axis)이 이미지 평면과 만나는 점의 픽셀 좌표. 이상적으로는 이미지 중심이지만, 제조 공차로 인해 수 픽셀 벗어날 수 있다.

동차 좌표(homogeneous coordinates)로 표현하면:

$$s \begin{bmatrix} u \\ v \\ 1 \end{bmatrix} = \mathbf{K} [\mathbf{R} | \mathbf{t}] \begin{bmatrix} X_w \\ Y_w \\ Z_w \\ 1 \end{bmatrix}$$

여기서 $[\mathbf{R} | \mathbf{t}]$는 월드 좌표계에서 카메라 좌표계로의 외부 파라미터(extrinsic parameters)이고, $s = Z_c$는 깊이 스케일 팩터이다.

투영 함수를 $\pi(\cdot)$로 표기하면:

$$\mathbf{p} = \pi(\mathbf{P}_c) = \begin{bmatrix} f_x \frac{X_c}{Z_c} + c_x \\ f_y \frac{Y_c}{Z_c} + c_y \end{bmatrix}$$

이 비선형 투영 함수의 자코비안(Jacobian)은 상태 추정에서 직접 사용된다:

$$\frac{\partial \pi}{\partial \mathbf{P}_c} = \begin{bmatrix} \frac{f_x}{Z_c} & 0 & -\frac{f_x X_c}{Z_c^2} \\ 0 & \frac{f_y}{Z_c} & -\frac{f_y Y_c}{Z_c^2} \end{bmatrix}$$

이 $2 \times 3$ 자코비안은 EKF 기반 VIO에서 관측 모델을 선형화하고, 비선형 최적화에서 잔차의 자코비안을 구성한다.

### 2.1.2 렌즈 왜곡 모델 (Lens Distortion Model)

실제 카메라의 렌즈는 핀홀 모델의 이상적 투영에서 벗어나는 왜곡(distortion)을 도입한다. 왜곡을 무시하면 재투영 오차(reprojection error)가 수 픽셀에서 수십 픽셀까지 증가하므로, 정밀한 센서 퓨전을 위해서는 반드시 보정해야 한다.

#### Radial-Tangential 왜곡 (Brown-Conrady 모델)

OpenCV가 지원하는 대표적인 왜곡 모델이며, pinhole camera를 쓰는 SLAM·VIO에서 흔히 선택된다. Wide-FoV lens에는 fisheye나 omnidirectional model이 더 적합할 수 있다.

정규화된 이미지 좌표 $\mathbf{p}_n = [x_n, y_n]^\top = [X_c/Z_c, \, Y_c/Z_c]^\top$에 대해:

$$r^2 = x_n^2 + y_n^2$$

**방사 왜곡(Radial Distortion):**

$$x_r = x_n (1 + k_1 r^2 + k_2 r^4 + k_3 r^6)$$
$$y_r = y_n (1 + k_1 r^2 + k_2 r^4 + k_3 r^6)$$

방사 왜곡은 렌즈의 곡률에서 비롯되며, 이미지 중심으로부터의 거리에 따라 증가한다. $k_1 < 0$이면 배럴 왜곡(barrel distortion, 중심부 확대), $k_1 > 0$이면 핀쿠션 왜곡(pincushion distortion, 가장자리 확대)이 발생한다.

**접선 왜곡(Tangential Distortion):**

$$x_t = 2p_1 x_n y_n + p_2 (r^2 + 2x_n^2)$$
$$y_t = p_1 (r^2 + 2y_n^2) + 2p_2 x_n y_n$$

접선 왜곡은 렌즈가 이미지 센서와 완벽히 평행하지 않을 때(decentering) 발생한다. 방사 왜곡보다 크기가 훨씬 작지만, 정밀 응용에서는 무시할 수 없다.

**왜곡된 좌표의 최종 계산:**

$$\begin{bmatrix} x_d \\ y_d \end{bmatrix} = \begin{bmatrix} x_r + x_t \\ y_r + y_t \end{bmatrix}$$

$$\begin{bmatrix} u \\ v \end{bmatrix} = \begin{bmatrix} f_x x_d + c_x \\ f_y y_d + c_y \end{bmatrix}$$

왜곡 파라미터 $[k_1, k_2, p_1, p_2, k_3]$는 [Zhang (2000)](https://doi.org/10.1109/34.888718)의 캘리브레이션 방법으로 추정되며, 이는 Ch.3에서 상세히 다룬다.

#### 어안 (Fisheye / Equidistant) 모델

시야각(FoV)이 넓은 어안 렌즈(180° 이상)에서는 radial-tangential 모델이 적합하지 않다. 이미지 가장자리에서 $r$이 매우 크므로 다항식 근사가 발산하기 때문이다.

[Kannala & Brandt (2006)](https://doi.org/10.1109/TPAMI.2006.153)의 generic camera model은 다음과 같이 정의된다:

입사각 $\theta$를 3D 점의 광축으로부터의 각도로 정의한다:

$$\theta = \arctan\left(\frac{\sqrt{X_c^2 + Y_c^2}}{Z_c}\right)$$

왜곡된 반경 $r_d$를 $\theta$의 홀수 다항식으로 모델링한다:

$$r_d = k_1 \theta + k_2 \theta^3 + k_3 \theta^5 + k_4 \theta^7 + k_5 \theta^9$$

순수 등거리(equidistant) 투영에서는 $r_d = f \cdot \theta$이며, 이는 $k_1 = f$, $k_2 = k_3 = \cdots = 0$에 해당한다.

투영 좌표:

$$\begin{bmatrix} u \\ v \end{bmatrix} = \begin{bmatrix} f_x \cdot r_d \cdot \frac{x_n}{\sqrt{x_n^2 + y_n^2}} + c_x \\ f_y \cdot r_d \cdot \frac{y_n}{\sqrt{x_n^2 + y_n^2}} + c_y \end{bmatrix}$$

어안 렌즈는 넓은 FoV로 인해 주변 환경 인식에 유리하며, [VINS-Mono](https://arxiv.org/abs/1708.03852), Basalt 등의 VIO 시스템에서 지원된다. 캘리브레이션에는 [Scaramuzza et al. (2006)](https://rpg.ifi.uzh.ch/docs/IROS06_scaramuzza.pdf)의 OCamCalib이나 Kalibr를 사용한다.

```python
import numpy as np

def project_pinhole(P_c, K, dist_coeffs=None):
    """핀홀 카메라 모델에 의한 3D→2D 투영.
    
    Args:
        P_c: (3,) 카메라 좌표계의 3D 점 [X, Y, Z]
        K: (3,3) 내부 파라미터 행렬
        dist_coeffs: (5,) 왜곡 계수 [k1, k2, p1, p2, k3] 또는 None
    
    Returns:
        (2,) 이미지 좌표 [u, v]
    """
    X, Y, Z = P_c
    # 정규화 좌표
    x_n = X / Z
    y_n = Y / Z
    
    if dist_coeffs is not None:
        k1, k2, p1, p2, k3 = dist_coeffs
        r2 = x_n**2 + y_n**2
        r4 = r2**2
        r6 = r2 * r4
        
        # 방사 왜곡
        radial = 1 + k1 * r2 + k2 * r4 + k3 * r6
        x_d = x_n * radial + 2 * p1 * x_n * y_n + p2 * (r2 + 2 * x_n**2)
        y_d = y_n * radial + p1 * (r2 + 2 * y_n**2) + 2 * p2 * x_n * y_n
    else:
        x_d = x_n
        y_d = y_n
    
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    
    u = fx * x_d + cx
    v = fy * y_d + cy
    return np.array([u, v])


def project_fisheye(P_c, K, fisheye_coeffs):
    """어안 (equidistant) 카메라 모델에 의한 3D→2D 투영.
    
    OpenCV fisheye 모듈의 모델을 사용한다:
      theta_d = theta * (1 + k1*theta^2 + k2*theta^4 + k3*theta^6 + k4*theta^8)
    이는 위 본문의 Kannala-Brandt 5-파라미터 모델과 파라미터화가 다르다.
    
    Args:
        P_c: (3,) 카메라 좌표계의 3D 점 [X, Y, Z]
        K: (3,3) 내부 파라미터 행렬
        fisheye_coeffs: (4,) 왜곡 계수 [k1, k2, k3, k4] (OpenCV 컨벤션)
    
    Returns:
        (2,) 이미지 좌표 [u, v]
    """
    X, Y, Z = P_c
    r_xyz = np.sqrt(X**2 + Y**2)
    theta = np.arctan2(r_xyz, Z)
    
    k1, k2, k3, k4 = fisheye_coeffs
    theta2 = theta**2
    r_d = theta * (1 + k1 * theta2 + k2 * theta2**2 + k3 * theta2**3 + k4 * theta2**4)
    
    if r_xyz < 1e-10:
        return np.array([K[0, 2], K[1, 2]])
    
    u = K[0, 0] * r_d * (X / r_xyz) + K[0, 2]
    v = K[1, 1] * r_d * (Y / r_xyz) + K[1, 2]
    return np.array([u, v])
```

### 2.1.3 재투영 오차 (Reprojection Error)

재투영 오차(reprojection error)는 센서 퓨전에서 카메라 관측을 활용하는 거의 모든 알고리즘의 비용 함수이다.

재투영 오차는 카메라 포즈 $\mathbf{T}_{cw} = [\mathbf{R}|\mathbf{t}]$로 3D 랜드마크 $\mathbf{P}_w$를 이미지에 투영한 예측 좌표 $\hat{\mathbf{p}}$와 실제 관측된 특징점 좌표 $\mathbf{p}_{\text{obs}}$ 사이의 차이이다:

$$\mathbf{e}_{\text{reproj}} = \mathbf{p}_{\text{obs}} - \pi(\mathbf{T}_{cw} \cdot \mathbf{P}_w)$$

여기서 $\pi(\cdot)$는 위에서 정의한 투영 함수(왜곡 포함)이다.

**번들 조정(Bundle Adjustment)**에서는 모든 카메라 포즈와 모든 랜드마크에 대한 재투영 오차의 합을 최소화한다:

$$\min_{\{\mathbf{T}_i\}, \{\mathbf{P}_j\}} \sum_{i,j} \rho\left(\| \mathbf{p}_{ij} - \pi(\mathbf{T}_i \cdot \mathbf{P}_j) \|^2_{\mathbf{\Sigma}_{ij}}\right)$$

여기서:
- $\mathbf{p}_{ij}$: 카메라 $i$에서 관측된 랜드마크 $j$의 이미지 좌표
- $\mathbf{\Sigma}_{ij}$: 관측 노이즈 공분산 (등방성 모델에서 $\sigma^2 \mathbf{I}_2$, $\sigma \approx 1$ 픽셀)
- $\rho(\cdot)$: 로버스트 커널 (Huber, Cauchy 등) — 아웃라이어의 영향을 억제

재투영 오차의 분포는 $\sigma = 0.5 \sim 2$ 픽셀의 가우시안으로 모델링된다. 이 값은 특징점 검출기의 정밀도에 의존하며, 서브픽셀 정밀도의 코너 검출이 가능한 경우 $\sigma \approx 0.5$ 픽셀까지 줄어든다. 최근에는 [Depth Anything V2 (Yang et al., 2024)](https://arxiv.org/abs/2406.09414)와 [Metric3D v2 (Hu et al., 2024)](https://arxiv.org/abs/2404.15506) 같은 foundation model이 단일 이미지로부터 밀집 깊이를 추정하여, 카메라의 관측 모델을 2D 재투영 오차에서 3D 깊이 관측으로 확장하는 데 활용되고 있다.

### 2.1.4 롤링 셔터 모델 (Rolling Shutter Model)

대부분의 저가 카메라(스마트폰, 웹캠)는 CMOS 이미지 센서를 사용하며, **롤링 셔터(rolling shutter)** 방식으로 이미지를 취득한다. 글로벌 셔터(global shutter)가 모든 픽셀을 동시에 노출하는 것과 달리, 롤링 셔터는 행(row) 단위로 순차적으로 노출한다. 이미지의 상단과 하단은 서로 다른 시점에 캡처된다.

$k$번째 행의 노출 시각은:

$$t_k = t_0 + k \cdot t_r$$

여기서 $t_0$은 첫 번째 행의 노출 시각, $t_r$은 행 간 시간 간격(row readout time)이다. 전체 이미지 노출에 걸리는 시간은 $H \cdot t_r$ ($H$: 이미지 높이)이며, 이는 수 밀리초에서 수십 밀리초에 달한다.

카메라가 움직이는 동안 롤링 셔터로 이미지를 취득하면 다음과 같은 아티팩트가 발생한다:
- **기하학적 왜곡**: 수직선이 기울어지거나, 움직이는 물체가 젤리처럼 변형된다.
- **특징점 위치 오차**: 각 특징점이 서로 다른 카메라 포즈에서 촬영되므로, 글로벌 셔터를 가정한 투영 모델이 부정확해진다.

롤링 셔터를 고려한 투영 모델에서는, 각 특징점의 행 좌표 $v$에 해당하는 시각의 카메라 포즈를 사용해야 한다:

$$\mathbf{p}_i = \pi\left(\mathbf{T}(t_{v_i}) \cdot \mathbf{P}_i\right)$$

여기서 $\mathbf{T}(t_{v_i})$는 $i$번째 특징점의 행 $v_i$에 대응하는 시각의 카메라 포즈이다. 이 포즈는 IMU 측정을 이용한 보간(interpolation)으로 구한다:

$$\mathbf{T}(t_{v_i}) = \mathbf{T}(t_0) \cdot \text{Exp}\left(\frac{v_i}{H} \cdot \text{Log}(\mathbf{T}(t_0)^{-1} \mathbf{T}(t_0 + H \cdot t_r))\right)$$

여기서 $\text{Exp}$와 $\text{Log}$는 $SE(3)$ 리 군 위의 지수/로그 맵이다.

롤링 셔터 보정은 VIO 시스템([VINS-Mono](https://arxiv.org/abs/1708.03852), [ORB-SLAM3](https://arxiv.org/abs/2007.11898))에서 선택적으로 지원되며, 특히 스마트폰이나 드론 탑재 카메라처럼 고속 모션과 저가 센서의 조합에서 중요하다.

```python
import numpy as np
from scipy.spatial.transform import Rotation, Slerp

def rolling_shutter_project(P_w, T_start, T_end, K, H, v_row):
    """롤링 셔터 카메라 모델의 투영.
    
    이미지 첫 행(T_start)과 마지막 행(T_end)의 포즈를 보간하여
    해당 행의 포즈에서 3D 점을 투영한다.
    
    Args:
        P_w: (3,) 월드 좌표 3D 점
        T_start: (4,4) 첫 행 노출 시 카메라→월드 변환
        T_end: (4,4) 마지막 행 노출 시 카메라→월드 변환
        K: (3,3) 내부 파라미터 행렬
        H: 이미지 높이 (행 수)
        v_row: 투영할 특징점의 행 좌표
    
    Returns:
        (2,) 이미지 좌표 [u, v]
    """
    alpha = v_row / H  # 보간 비율 [0, 1]
    
    # 회전 보간 (SLERP)
    R_start = Rotation.from_matrix(T_start[:3, :3])
    R_end = Rotation.from_matrix(T_end[:3, :3])
    slerp = Slerp([0, 1], Rotation.concatenate([R_start, R_end]))
    R_interp = slerp(alpha).as_matrix()
    
    # 이동 보간 (선형)
    t_interp = (1 - alpha) * T_start[:3, 3] + alpha * T_end[:3, 3]
    
    # 월드→카메라 변환 후 투영
    P_c = R_interp.T @ (P_w - t_interp)
    
    u = K[0, 0] * P_c[0] / P_c[2] + K[0, 2]
    v = K[1, 1] * P_c[1] / P_c[2] + K[1, 2]
    return np.array([u, v])
```

---

## 2.2 LiDAR 관측 모델

LiDAR는 레이저 펄스를 발사하여 반사광의 비행시간(Time-of-Flight)이나 위상차로 거리를 계산하는 능동 센서다. 관측 모델과 오차 특성을 차례로 본다.

### 2.2.1 Range-Bearing 모델

LiDAR의 기본 관측은 각 레이저 빔에 대한 **거리(range)**와 **방위(bearing)** 쌍이다. 3D LiDAR에서 각 포인트의 관측은 구면 좌표 $(r, \alpha, \omega)$로 표현된다:

- $r$: 거리 (range)
- $\alpha$: 수평 각도 (azimuth)
- $\omega$: 수직 각도 (elevation)

관측 모델은 LiDAR 좌표계의 3D 점 $\mathbf{P}_L = [x, y, z]^\top$에 대해:

$$\begin{aligned}
r &= \sqrt{x^2 + y^2 + z^2} + n_r \\
\alpha &= \arctan2(y, x) + n_\alpha \\
\omega &= \arcsin\left(\frac{z}{\sqrt{x^2 + y^2 + z^2}}\right) + n_\omega
\end{aligned}$$

여기서 $n_r, n_\alpha, n_\omega$는 각각 거리, 수평각, 수직각의 관측 노이즈이다.

**노이즈 특성:**
- **거리 노이즈** $n_r$: $\sigma_r \approx 1\text{–}3\,\text{cm}$ (기계식 LiDAR), $\sigma_r \approx 2\text{–}5\,\text{cm}$ (솔리드 스테이트). 거리가 증가하면 빔 확산(beam divergence)과 수신 에너지 감소로 노이즈가 커진다.
- **각도 노이즈** $n_\alpha, n_\omega$: 인코더 정밀도에 의존. $\sigma_\alpha, \sigma_\omega \approx 0.01°\text{–}0.1°$. 작은 값이지만 원거리에서는 위치 오차로 증폭된다 — 50m 거리에서 $0.1°$의 각도 오차는 약 $8.7\,\text{cm}$의 횡방향 오차에 해당한다.

구면 좌표에서 직교 좌표로의 변환:

$$\mathbf{P}_L = \begin{bmatrix} r \cos\omega \cos\alpha \\ r \cos\omega \sin\alpha \\ r \sin\omega \end{bmatrix}$$

**빔 확산(Beam Divergence).** LiDAR의 레이저 빔은 완벽한 직선이 아니라 미세한 원뿔형으로 확산된다. 빔 확산각은 $0.1°\text{–}0.5°$이며, 이는 원거리에서 빔의 풋프린트(footprint)가 커져 하나의 반사점이 아닌 영역의 평균 거리를 측정하게 만든다. 물체의 경계에서 빔이 부분적으로 물체와 배경 모두에 걸치면 두 거리의 중간값이 측정되어 허위 점이 생성된다 — 이를 **혼합 픽셀(mixed pixel)** 효과라고 한다.

### 2.2.2 Motion Distortion (모션 왜곡)

기계식 스피닝(spinning) LiDAR는 센서가 360° 회전하면서 레이저를 발사한다. Velodyne VLP-16의 경우 1회전에 약 100ms가 소요된다. 이 100ms 동안 플랫폼(차량, 드론)이 이동하면, 한 스캔 내의 포인트들이 서로 다른 좌표계에서 측정된 것이 된다. 이것이 **모션 왜곡(motion distortion)** 또는 **ego-motion compensation** 문제이다.

이 문제는 카메라의 롤링 셔터와 구조가 같다. 스캔의 $i$번째 포인트가 시각 $t_i$에 측정되었다면, 이 포인트를 기준 시각 $t_0$의 좌표계로 변환해야 한다:

$$\mathbf{P}_L^{(t_0)} = \mathbf{T}(t_0)^{-1} \cdot \mathbf{T}(t_i) \cdot \mathbf{P}_L^{(t_i)}$$

여기서 $\mathbf{T}(t_i)$는 시각 $t_i$에서의 LiDAR 포즈이다.

**보정 방법:**

1. **IMU 기반 보간**: IMU의 고주파 측정으로 스캔 기간 동안의 포즈 변화를 보간한다. LIO-SAM과 FAST-LIO2를 비롯한 LIO 시스템에서 쓰이는 방식이다.
2. **이전 프레임 오도메트리 기반**: 직전 프레임의 추정 속도로 등속(constant velocity) 모델을 적용한다.
3. **연속 시간(Continuous-time) 방법**: B-스플라인 등으로 궤적을 연속 함수로 모델링하고, 각 포인트의 시각에서의 포즈를 평가한다. [CT-ICP (Dellenbach et al., 2022)](https://arxiv.org/abs/2109.12979)가 대표적이다.

```python
import numpy as np
from scipy.spatial.transform import Rotation, Slerp

def undistort_scan(points, timestamps, T_start, T_end, t_start, t_end):
    """IMU 기반 모션 왜곡 보정.
    
    스캔 시작과 끝의 포즈를 보간하여 각 포인트를 시작 시각 좌표계로 변환.
    
    Args:
        points: (N, 3) LiDAR 포인트 클라우드
        timestamps: (N,) 각 포인트의 타임스탬프
        T_start: (4,4) 스캔 시작 시 LiDAR 포즈 (lidar→world)
        T_end: (4,4) 스캔 끝 시 LiDAR 포즈 (lidar→world)
        t_start, t_end: 스캔 시작/끝 시각
    
    Returns:
        (N, 3) 보정된 포인트 클라우드
    """
    R_start = Rotation.from_matrix(T_start[:3, :3])
    R_end = Rotation.from_matrix(T_end[:3, :3])
    slerp = Slerp([t_start, t_end], Rotation.concatenate([R_start, R_end]))
    
    corrected = np.zeros_like(points)
    for i in range(len(points)):
        alpha = (timestamps[i] - t_start) / (t_end - t_start)
        alpha = np.clip(alpha, 0, 1)
        
        # 해당 시각의 포즈 보간
        R_i = slerp(timestamps[i]).as_matrix()
        t_i = (1 - alpha) * T_start[:3, 3] + alpha * T_end[:3, 3]
        
        # 해당 시각 좌표계 → 시작 시각 좌표계
        p_world = R_i @ points[i] + t_i
        corrected[i] = T_start[:3, :3].T @ (p_world - T_start[:3, 3])
    
    return corrected
```

### 2.2.3 Spinning vs Solid-State LiDAR가 퓨전에 미치는 영향

**기계식 스피닝 LiDAR** (Velodyne, Ouster, Hesai)는 360° 수평 FoV를 제공하며, 한 스캔이 완전한 환형 포인트 클라우드를 구성한다. LOAM 계열 알고리즘은 이 특성을 전제로 만들어졌다 — edge/planar feature를 수평 스캔 라인에서 추출하고, 전방위 관측으로 6-DoF 포즈를 추정한다.

**솔리드 스테이트 LiDAR** (Livox Mid-40/70, Avia, HAP 등)는 기계적 회전부가 없으며, 제한된 FoV(예: Livox Mid-70은 약 70.4° 원형) 내에서 비반복(non-repetitive) 스캔 패턴을 사용한다. 시간이 지남에 따라 FoV 내의 커버리지가 점진적으로 증가하는 특성이 있다.

이 차이가 퓨전 알고리즘에 미치는 영향:

| 특성 | 스피닝 LiDAR | 솔리드 스테이트 LiDAR |
|------|-------------|-------------------|
| FoV | 360° 수평 | 제한적 (40°~120°) |
| 스캔 패턴 | 반복적 (수평 라인) | 비반복적 (로제트, 리사주 등) |
| 단일 프레임 밀도 | 균일 | 불균일 (시간에 따라 증가) |
| Feature 추출 | 스캔 라인 기반 가능 | 스캔 라인 구조 없음 |
| 적합한 알고리즘 | LOAM, LeGO-LOAM | FAST-LIO/LIO2 (점 단위 처리) |

FAST-LIO / [FAST-LIO2](https://arxiv.org/abs/2107.06829)는 스캔 라인 구조에 의존하지 않고 **개별 포인트를 순차적으로 처리**하는 iterated EKF 구조를 사용하므로 솔리드 스테이트 LiDAR에 잘 맞는다. 반면 LOAM의 edge/planar feature 추출은 스캔 라인 구조를 전제하므로 솔리드 스테이트에 직접 적용하기 어렵다. 최근 [FAST-LIVO2 (Zheng et al., 2024)](https://arxiv.org/abs/2408.14035)는 이 구조를 확장하여 LiDAR-관성-비전 세 센서를 동일한 iterated EKF 내에서 순차적으로 융합하며, direct 방법으로 별도의 특징점 추출 없이 LiDAR 포인트와 이미지 모두를 처리한다.

---

## 2.3 IMU 모델

IMU(Inertial Measurement Unit)는 3축 가속도계(accelerometer)와 3축 자이로스코프(gyroscope)로 구성된다. 100Hz–1kHz의 고주파 관측은 카메라·LiDAR 프레임 사이를 보간하며 초기화와 스케일 관측성을 제공한다. 퓨전에는 이 측정의 오차 모델이 필요하다.

### 2.3.1 자이로스코프 오차 모델

자이로스코프는 3축 각속도 $\boldsymbol{\omega}$를 측정한다. 실제 측정값 $\tilde{\boldsymbol{\omega}}$는 다음과 같이 모델링된다:

$$\tilde{\boldsymbol{\omega}} = \boldsymbol{\omega} + \mathbf{b}_g + \mathbf{n}_g$$

각 항의 의미:
- $\boldsymbol{\omega}$: 참 각속도 (IMU 바디 프레임에서)
- $\mathbf{b}_g$: **바이어스(bias)** — 시간에 따라 천천히 변하는 상수적 오프셋
- $\mathbf{n}_g$: **측정 노이즈** — 백색 가우시안 노이즈(Additive White Gaussian Noise, AWGN)

**바이어스의 동역학.** 바이어스의 느린 시간 변화는 **랜덤 워크(random walk)**로 모델링한다:

$$\dot{\mathbf{b}}_g = \mathbf{n}_{bg}$$

여기서 $\mathbf{n}_{bg} \sim \mathcal{N}(\mathbf{0}, \sigma_{bg}^2 \mathbf{I})$는 바이어스 변화의 구동 노이즈이다. 이산 시간에서:

$$\mathbf{b}_{g,k+1} = \mathbf{b}_{g,k} + \sigma_{bg} \sqrt{\Delta t} \cdot \mathbf{w}_k, \quad \mathbf{w}_k \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$$

자이로스코프의 노이즈 파라미터는 장치와 운용 조건에 맞춰 정한다.

- **백색 노이즈 밀도**: 이름이 명시된 장치의 데이터시트와 Allan deviation의 $-1/2$ 기울기 구간에서 추정한다. 단위와 연속 시간 노이즈 밀도 정의가 구현의 이산 공분산 변환과 일치하는지 확인한다.
- **In-run bias stability**: 충분히 긴 정지 데이터의 Allan deviation에서 기울기가 거의 0인 구간을 확인한다. 온도, 장착 응력, 진동, 측정 시간에 따라 달라지므로 데이터시트의 한 수치를 그대로 쓰지 않는다.
- **바이어스 랜덤 워크 구동 노이즈**: 장시간 구간의 양의 기울기에서 추정한다. 측정 노이즈 $\sigma_g$와 바이어스 구동 노이즈 $\sigma_{bg}$를 구분하고, 연속 시간 밀도와 이산 시간 공분산의 변환 규약을 기록한다.

### 2.3.2 가속도계 오차 모델

가속도계는 3축 비력(specific force) $\mathbf{a}$를 측정한다. 비력은 진가속도에서 중력을 뺀 값이다. 실제 측정값 $\tilde{\mathbf{a}}$는:

$$\tilde{\mathbf{a}} = \mathbf{R}_{bw}(\mathbf{a}_w - \mathbf{g}_w) + \mathbf{b}_a + \mathbf{n}_a$$

각 항의 의미:
- $\mathbf{a}_w$: 월드 프레임에서의 진가속도
- $\mathbf{g}_w = [0, 0, -g]^\top$: 중력 벡터 ($g \approx 9.81\,\text{m/s}^2$)
- $\mathbf{R}_{bw}$: 월드→바디 회전 행렬
- $\mathbf{b}_a$: 가속도계 바이어스
- $\mathbf{n}_a \sim \mathcal{N}(\mathbf{0}, \sigma_a^2 \mathbf{I})$: 측정 노이즈

**중력의 역할.** 가속도계가 중력을 "느끼는" 것은 IMU 기반 퓨전에서 매우 중요하다. 정지 상태에서도 가속도계는 $[0, 0, g]^\top$ (위 방향을 z로 놓은 경우)을 측정한다. 이 중력 관측으로부터 롤(roll)과 피치(pitch)를 추정할 수 있다. 그러나 요(yaw)는 중력 벡터에 대한 회전이므로 관측 불가능(unobservable)하다. 이 때문에 VIO/LIO 초기화에서 요 각도를 추정하려면 시각적 특징점의 이동 등 추가 관측이 필요하다.

**바이어스 동역학.** 자이로스코프와 동일하게 랜덤 워크로 모델링한다:

$$\dot{\mathbf{b}}_a = \mathbf{n}_{ba}, \quad \mathbf{n}_{ba} \sim \mathcal{N}(\mathbf{0}, \sigma_{ba}^2 \mathbf{I})$$

가속도계도 같은 절차로 파라미터를 정한다.

- **백색 노이즈 밀도**: 장치 데이터시트와 Allan deviation의 $-1/2$ 기울기 구간을 함께 사용하고, $\text{m/s}^2/\sqrt{\text{Hz}}$, $\text{mg}/\sqrt{\text{Hz}}$, VRW 사이의 단위 변환을 명시한다.
- **In-run bias stability**: 정지·온도 시험에서 거의 평평한 Allan 구간과 반복 실험의 분산을 확인한다. 중력 투영 오차와 장착 자세 변화가 바이어스로 섞이지 않게 한다.
- **바이어스 랜덤 워크 구동 노이즈**: 장시간 데이터의 해당 기울기에서 $\sigma_{ba}$를 추정하고, 필터가 사용하는 샘플 주기에 맞춰 이산 공분산으로 변환한다.

### 2.3.3 Allan Variance

Allan Variance는 IMU의 노이즈 특성을 분석하는 표준적인 방법이다. 정지 상태에서 장시간(수 시간) 데이터를 수집하여 다양한 평균 시간(cluster time) $\tau$에서의 분산을 계산한다.

Allan Variance $\sigma^2(\tau)$의 정의:

$$\sigma^2(\tau) = \frac{1}{2} \langle (\bar{y}_{k+1} - \bar{y}_k)^2 \rangle$$

여기서 $\bar{y}_k$는 $k$번째 구간(길이 $\tau$)의 평균 출력이다.

log-log 플롯에서 Allan Deviation $\sigma(\tau)$의 기울기로 노이즈 종류를 식별한다:

| 기울기 | 노이즈 종류 | 물리적 의미 |
|--------|-----------|-----------|
| $-1/2$ | 각도/속도 랜덤 워크 (ARW/VRW) | 백색 노이즈 $\mathbf{n}_g, \mathbf{n}_a$ |
| $0$ | 바이어스 불안정성 (Bias Instability) | 플리커 노이즈, 최소값이 바이어스 안정성 |
| $+1/2$ | 레이트 랜덤 워크 (RRW) | 바이어스 랜덤 워크 $\mathbf{n}_{bg}, \mathbf{n}_{ba}$ |

**데이터시트 읽는 법.** IMU 데이터시트에서는 센서 퓨전에 필요한 파라미터를 다음과 같이 추출한다.

1. **각도 랜덤 워크(Angular Random Walk, ARW)**: 단위 $°/\sqrt{\text{hr}}$ 또는 $\text{rad/s}/\sqrt{\text{Hz}}$. Allan Deviation 플롯에서 $\tau = 1\,\text{s}$일 때의 값, 또는 기울기 $-1/2$ 구간에서 읽는다. 이것이 $\sigma_g$에 해당한다.
2. **속도 랜덤 워크(Velocity Random Walk, VRW)**: 단위 $\text{m/s}/\sqrt{\text{hr}}$ 또는 $\text{m/s}^2/\sqrt{\text{Hz}}$. 가속도계의 백색 노이즈 밀도. 이것이 $\sigma_a$에 해당한다.
3. **바이어스 안정성(In-run Bias Stability)**: Allan Deviation 플롯의 최솟값. 시스템이 도달할 수 있는 바이어스 추정의 이론적 하한이다.
4. **레이트 랜덤 워크(Rate Random Walk)**: 바이어스가 시간에 따라 변하는 속도. 이것이 $\sigma_{bg}, \sigma_{ba}$에 해당한다.

```python
import numpy as np

def compute_allan_variance(data, dt, max_cluster_size=None):
    """Allan Variance 계산.
    
    Args:
        data: (N,) 정지 상태에서 수집한 IMU 한 축의 데이터
        dt: 샘플링 주기 (초)
        max_cluster_size: 최대 클러스터 크기 (None이면 N//2)
    
    Returns:
        taus: 클러스터 시간 배열
        adevs: Allan Deviation 배열
    """
    N = len(data)
    if max_cluster_size is None:
        max_cluster_size = N // 2
    
    # 로그 스케일로 클러스터 크기 생성
    cluster_sizes = np.unique(np.logspace(
        0, np.log10(max_cluster_size), num=100
    ).astype(int))
    cluster_sizes = cluster_sizes[cluster_sizes > 0]
    
    taus = []
    adevs = []
    
    for m in cluster_sizes:
        tau = m * dt
        # 비겹침 평균 계산
        n_clusters = N // m
        if n_clusters < 2:
            break
        
        # 각 클러스터의 평균
        truncated = data[:n_clusters * m].reshape(n_clusters, m)
        cluster_means = truncated.mean(axis=1)
        
        # Allan Variance
        diff = np.diff(cluster_means)
        avar = 0.5 * np.mean(diff**2)
        
        taus.append(tau)
        adevs.append(np.sqrt(avar))
    
    return np.array(taus), np.array(adevs)


def extract_imu_params(taus, adevs):
    """Allan Deviation 플롯에서 IMU 노이즈 파라미터 추출.
    
    Args:
        taus: 클러스터 시간 배열
        adevs: Allan Deviation 배열
    
    Returns:
        dict: {
            'white_noise': tau=1에서의 값 (ARW 또는 VRW),
            'bias_instability': 최솟값,
            'bias_instability_tau': 최솟값에 해당하는 tau
        }
    """
    # 백색 노이즈: tau=1에서의 Allan Deviation (기울기 -1/2 구간)
    idx_1s = np.argmin(np.abs(taus - 1.0))
    white_noise = adevs[idx_1s]
    
    # 바이어스 안정성: Allan Deviation의 최솟값
    idx_min = np.argmin(adevs)
    bias_instability = adevs[idx_min]
    bias_instability_tau = taus[idx_min]
    
    return {
        'white_noise': white_noise,
        'bias_instability': bias_instability,
        'bias_instability_tau': bias_instability_tau
    }
```

### 2.3.4 Strapdown Navigation Equation

IMU 측정으로부터 포즈(위치, 속도, 자세)를 적분하는 방정식을 스트랩다운 관성 항법 방정식(Strapdown Navigation Equation)이라 한다. "스트랩다운(strapdown)"이란 센서가 플랫폼에 직접 고정(strapped down)되어 있어, 기계식 짐벌 없이 소프트웨어로 좌표 변환을 수행한다는 의미이다.

월드 프레임(또는 항법 프레임)에서의 상태 $[\mathbf{R}, \mathbf{v}, \mathbf{p}]$의 연속 시간 동역학:

$$\dot{\mathbf{R}} = \mathbf{R} \cdot [\tilde{\boldsymbol{\omega}} - \mathbf{b}_g - \mathbf{n}_g]_\times$$

$$\dot{\mathbf{v}} = \mathbf{R} (\tilde{\mathbf{a}} - \mathbf{b}_a - \mathbf{n}_a) + \mathbf{g}$$

$$\dot{\mathbf{p}} = \mathbf{v}$$

여기서:
- $\mathbf{R} \in SO(3)$: 바디→월드 회전 행렬
- $\mathbf{v} \in \mathbb{R}^3$: 월드 프레임에서의 속도
- $\mathbf{p} \in \mathbb{R}^3$: 월드 프레임에서의 위치
- $[\cdot]_\times$: 반대칭(skew-symmetric) 행렬 (벡터의 외적을 행렬 곱으로 표현)

$$[\boldsymbol{\omega}]_\times = \begin{bmatrix} 0 & -\omega_z & \omega_y \\ \omega_z & 0 & -\omega_x \\ -\omega_y & \omega_x & 0 \end{bmatrix}$$

**이산 시간 적분.** 시각 $t_k$에서 $t_{k+1} = t_k + \Delta t$로의 상태 전파:

$$\mathbf{R}_{k+1} = \mathbf{R}_k \cdot \text{Exp}\left((\tilde{\boldsymbol{\omega}}_k - \mathbf{b}_{g,k}) \Delta t\right)$$

$$\mathbf{v}_{k+1} = \mathbf{v}_k + \mathbf{g} \Delta t + \mathbf{R}_k (\tilde{\mathbf{a}}_k - \mathbf{b}_{a,k}) \Delta t$$

$$\mathbf{p}_{k+1} = \mathbf{p}_k + \mathbf{v}_k \Delta t + \frac{1}{2} \mathbf{g} \Delta t^2 + \frac{1}{2} \mathbf{R}_k (\tilde{\mathbf{a}}_k - \mathbf{b}_{a,k}) \Delta t^2$$

여기서 $\text{Exp}(\boldsymbol{\phi})$는 $SO(3)$ 위의 지수 맵으로, 회전 벡터 $\boldsymbol{\phi}$를 회전 행렬로 변환한다. Rodrigues' 공식으로 계산할 수 있다:

$$\text{Exp}(\boldsymbol{\phi}) = \mathbf{I} + \frac{\sin\|\boldsymbol{\phi}\|}{\|\boldsymbol{\phi}\|} [\boldsymbol{\phi}]_\times + \frac{1 - \cos\|\boldsymbol{\phi}\|}{\|\boldsymbol{\phi}\|^2} [\boldsymbol{\phi}]_\times^2$$

**중간점(midpoint) 적분.** 위의 오일러 적분은 1차 정밀도이다. 2차 정밀도를 위해 연속된 두 IMU 측정의 중간값을 사용하는 방법이 있다:

$$\bar{\boldsymbol{\omega}} = \frac{1}{2}(\tilde{\boldsymbol{\omega}}_k + \tilde{\boldsymbol{\omega}}_{k+1}) - \mathbf{b}_{g,k}$$

$$\mathbf{R}_{k+1} = \mathbf{R}_k \cdot \text{Exp}(\bar{\boldsymbol{\omega}} \Delta t)$$

$$\bar{\mathbf{a}} = \frac{1}{2}\left(\mathbf{R}_k(\tilde{\mathbf{a}}_k - \mathbf{b}_{a,k}) + \mathbf{R}_{k+1}(\tilde{\mathbf{a}}_{k+1} - \mathbf{b}_{a,k})\right)$$

$$\mathbf{v}_{k+1} = \mathbf{v}_k + (\bar{\mathbf{a}} + \mathbf{g}) \Delta t$$

$$\mathbf{p}_{k+1} = \mathbf{p}_k + \mathbf{v}_k \Delta t + \frac{1}{2}(\bar{\mathbf{a}} + \mathbf{g}) \Delta t^2$$

중간점 적분은 VINS-Mono 같은 VIO 구현에서 흔히 쓰인다. FAST-LIO2를 포함한 각 system의 propagation 식과 discretization은 구현별로 확인해야 한다. IMU rate가 높더라도 angular acceleration과 sampling jitter가 크면 적분 오차가 남으므로, 중간점·RK4·preintegration 선택은 목표 motion과 error budget으로 검증한다.

```python
import numpy as np
from scipy.spatial.transform import Rotation

def skew(v):
    """3D 벡터의 skew-symmetric 행렬."""
    return np.array([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0]
    ])

def exp_so3(phi):
    """SO(3) 지수 맵: 회전 벡터 → 회전 행렬 (Rodrigues' formula)."""
    angle = np.linalg.norm(phi)
    if angle < 1e-10:
        return np.eye(3) + skew(phi)
    
    axis = phi / angle
    K = skew(axis)
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * K @ K

def imu_strapdown(gyro_data, accel_data, dt, R0, v0, p0, bg, ba, gravity):
    """스트랩다운 관성 항법 (중간점 적분).
    
    Args:
        gyro_data: (N, 3) 자이로스코프 측정 [rad/s]
        accel_data: (N, 3) 가속도계 측정 [m/s^2]
        dt: 샘플링 주기 [s]
        R0: (3,3) 초기 회전 행렬 (바디→월드)
        v0: (3,) 초기 속도 [m/s] (월드 프레임)
        p0: (3,) 초기 위치 [m] (월드 프레임)
        bg: (3,) 자이로 바이어스
        ba: (3,) 가속도 바이어스
        gravity: (3,) 중력 벡터 (예: [0, 0, -9.81])
    
    Returns:
        Rs: (N+1, 3, 3) 회전 이력
        vs: (N+1, 3) 속도 이력
        ps: (N+1, 3) 위치 이력
    """
    N = len(gyro_data)
    Rs = np.zeros((N + 1, 3, 3))
    vs = np.zeros((N + 1, 3))
    ps = np.zeros((N + 1, 3))
    
    Rs[0] = R0
    vs[0] = v0
    ps[0] = p0
    
    for k in range(N - 1):
        # 중간점 자이로
        omega_k = gyro_data[k] - bg
        omega_k1 = gyro_data[k + 1] - bg
        omega_mid = 0.5 * (omega_k + omega_k1)
        
        # 회전 업데이트
        Rs[k + 1] = Rs[k] @ exp_so3(omega_mid * dt)
        
        # 중간점 가속도 (월드 프레임)
        a_k = Rs[k] @ (accel_data[k] - ba)
        a_k1 = Rs[k + 1] @ (accel_data[k + 1] - ba)
        a_mid = 0.5 * (a_k + a_k1)
        
        # 속도/위치 업데이트
        vs[k + 1] = vs[k] + (a_mid + gravity) * dt
        ps[k + 1] = ps[k] + vs[k] * dt + 0.5 * (a_mid + gravity) * dt**2
    
    # 마지막 스텝 (단순 오일러)
    k = N - 1
    omega_k = gyro_data[k] - bg
    Rs[k + 1] = Rs[k] @ exp_so3(omega_k * dt)
    a_k = Rs[k] @ (accel_data[k] - ba)
    vs[k + 1] = vs[k] + (a_k + gravity) * dt
    ps[k + 1] = ps[k] + vs[k] * dt + 0.5 * (a_k + gravity) * dt**2
    
    return Rs, vs, ps
```

**드리프트의 수치적 의미.** 위의 스트랩다운 적분을 바이어스 보정 없이 수행하면 어떻게 오차가 누적되는지 단순한 가정으로 계산해보자. 설명을 위해 일정한 가속도계 바이어스를 $b_a = 0.01\,\text{m/s}^2$ (약 $1\,\text{mg}$)로 둔다.

- 1초 후 위치 오차: $\frac{1}{2} \times 0.01 \times 1^2 = 0.005\,\text{m}$ (5mm)
- 10초 후: $\frac{1}{2} \times 0.01 \times 100 = 0.5\,\text{m}$
- 60초 후: $\frac{1}{2} \times 0.01 \times 3600 = 18\,\text{m}$

이 예는 보조 센서가 없는 저가형 MEMS 관성 추정의 위치 오차가 짧은 임무에서도 빠르게 허용 범위를 넘을 수 있음을 보여준다. 실제 지속 시간은 IMU 성능, 온도 환경, 운동, 초기화, 임무의 오차 예산에 따라 달라진다. 더 높은 성능의 INS는 보조 없이 더 오래 운용할 수 있지만 드리프트 자체는 계속 누적된다. VIO/LIO 시스템에서는 바이어스 $\mathbf{b}_g, \mathbf{b}_a$를 **상태 벡터의 일부로 포함**하고 다른 센서의 관측으로 갱신한다. 한편, 최근 딥러닝 기반 관성 오도메트리 연구도 활발하다. [AirIO (Chen et al., 2025)](https://arxiv.org/abs/2501.15659)는 IMU 특징의 관측 가능성을 강화하여 드론 환경에서 기존 학습 기반 관성 오도메트리 대비 50% 이상의 정확도 향상을 보고하고 있다.

### 2.3.5 IMU 등급 분류

항법등급, 전술등급, 산업용, 소비자용 같은 명칭은 제조사와 분야마다 경계가 다르다. 또한 MEMS는 제작 기술을 가리키므로 하나의 성능 등급과 같은 뜻이 아니다. 따라서 등급명이나 고정 가격표보다 이름이 명시된 장치의 사양과 목표 플랫폼 시험으로 선택한다.

| 선택 기준 | 확인할 항목 | 검증 자료 | 설계 영향 |
|----------|------------|----------|----------|
| 백색 노이즈 | 자이로·가속도계 노이즈 밀도, ARW/VRW 단위 | 장치 데이터시트 + Allan 시험 | 단기 상태 전파 오차 |
| 바이어스 | In-run stability, 랜덤 워크, 온도 계수 | 정지·온도 반복 시험 + Allan 시험 | 무보조 드리프트와 외부 관측 주기 |
| 동적 범위 | 측정 범위, 대역폭, 포화·클리핑 | 목표 운동·진동 프로파일 시험 | 급격한 운동에서의 관측 손실 |
| 시간 특성 | 출력률, 타임스탬프, clock/trigger 지원 | 지연·지터 측정 | 시간 보정과 퓨전 일관성 |
| 환경 내성 | 온도, 진동, 충격, 장착 응력 | 제조사 qualification + 목표 플랫폼 시험 | 반복성과 장기 안정성 |

외부 센서의 종류와 보조 주기는 이 측정값을 임무의 오차 예산에 넣어 정한다. 가격은 시점과 수량에 따라 달라지므로 현재 견적과 필요한 캘리브레이션·동기화 비용까지 함께 비교한다.

---

## 2.4 GNSS 모델

GNSS(Global Navigation Satellite System)는 GPS, GLONASS, Galileo, BeiDou 등 위성 항법 시스템의 총칭이다. 센서 퓨전에서 GNSS 관측을 활용하기 위한 수학적 모델을 본다.

### 2.4.1 의사거리 관측 모델 (Pseudorange)

GNSS 수신기는 각 위성으로부터의 **의사거리(pseudorange)** $\rho$를 측정한다. 의사거리는 위성까지의 실제 거리에 수신기 시계 오차 등의 오차 항이 더해진 것이다:

$$\rho^s = r^s + c \cdot \delta t_r - c \cdot \delta t^s + I^s + T^s + \epsilon_\rho$$

각 항의 의미:
- $r^s = \|\mathbf{p}_r - \mathbf{p}^s\|$: 수신기 위치 $\mathbf{p}_r$와 위성 $s$ 위치 $\mathbf{p}^s$ 사이의 기하학적 거리
- $c \cdot \delta t_r$: 수신기 시계 오차 (미지수, 위치와 함께 추정)
- $c \cdot \delta t^s$: 위성 시계 오차 (항법 메시지에서 보정 가능)
- $I^s$: 전리층(ionosphere) 지연 — 전리층의 자유 전자에 의한 신호 지연. L1/L2 이중 주파수 수신기로 제거 가능.
- $T^s$: 대류권(troposphere) 지연 — 대기의 수증기와 공기에 의한 지연. Saastamoinen 모델 등으로 보정.
- $\epsilon_\rho$: 잔여 노이즈 (열잡음, 다중경로 등). 일반적으로 $\sigma_\rho \approx 1\text{–}5\,\text{m}$ (단독 측위), $\sigma_\rho \approx 0.1\text{–}0.3\,\text{m}$ (이중 주파수 + 보정).

**측위(positioning) 원리.** 4개 이상의 위성으로부터 의사거리를 관측하면, 수신기 위치 $(x, y, z)$와 시계 오차 $\delta t_r$의 4개 미지수를 풀 수 있다. 비선형 방정식을 최소자승법 또는 칼만 필터로 풀며, 이것이 표준 단독 측위(Single Point Positioning, SPP)의 기본이다.

**다중경로(Multipath).** 도심 환경에서 위성 신호가 건물에 반사되어 직접 경로 외의 경로로 도달하면, 의사거리에 수 미터에서 수십 미터의 오차가 추가된다. 이는 모델링이 어렵고 환경에 따라 변하므로, 센서 퓨전에서 GNSS를 사용할 때 아웃라이어 처리가 특히 중요하다.

### 2.4.2 반송파 위상 관측 모델 (Carrier Phase)

반송파 위상의 fractional-phase 관측 잡음은 의사거리보다 훨씬 작아 millimeter 수준일 수 있다. 그러나 이것이 곧바로 millimeter positioning accuracy를 뜻하지는 않는다. L1 반송파(약 1575.42 MHz)의 파장은 약 19 cm이며, 정수 모호성·cycle slip·multipath·대기 오차를 함께 풀어야 한다.

$$\Phi^s = r^s + c \cdot \delta t_r - c \cdot \delta t^s + \lambda N^s - I^s + T^s + \epsilon_\Phi$$

여기서:
- $\lambda$: 반송파 파장
- $N^s$: **정수 모호성(integer ambiguity)** — 수신기와 위성 사이의 전체 파장 수. RTK/PPP는 이 미지의 정수값을 정확히 결정해야 한다.
- $\epsilon_\Phi \approx 1\text{–}5\,\text{mm}$: 반송파 위상 노이즈 (의사거리 노이즈의 약 1/100)

전리층 지연의 부호는 의사거리와 반대이다(군속도 vs 위상속도). 이중 주파수 조합으로 1차 전리층 항을 추정·완화할 수 있지만, higher-order term과 다른 오차원은 남는다.

### 2.4.3 RTK (Real-Time Kinematic)

RTK는 기준국 또는 network correction과의 **차분(differencing)** 관측으로 공통 오차를 줄이고 carrier-phase integer ambiguity를 실시간으로 해결하는 기법이다. Baseline이 길어질수록 대기 오차의 상관이 약해지며, ambiguity가 fixed되고 신호·correction 품질이 좋은 조건에서 centimeter-class 결과를 낼 수 있다.

**이중 차분(Double Difference).** 위성 $s$와 기준 위성 $r$에 대한 기준국-이동국 간 이중 차분:

$$\nabla\Delta\Phi_{br}^{sr} = \nabla\Delta r_{br}^{sr} + \lambda \nabla\Delta N_{br}^{sr} + \epsilon$$

이중 차분을 통해 수신기/위성 시계 오차가 제거되고, 전리층/대류권 오차가 (기준국이 가까울 때) 거의 제거된다. 남은 미지수는 기하학적 거리(위치에 의존)와 정수 모호성이며, LAMBDA 알고리즘 등으로 정수 모호성을 해결한다.

### 2.4.4 PPP (Precise Point Positioning)

PPP는 기준국 없이 단일 수신기로 센티미터 급 측위를 달성하는 기법이다. 정밀 궤도력(precise orbit)과 정밀 시계 보정(precise clock)을 외부 서비스에서 수신하여 위성 관련 오차를 제거하고, 전리층/대류권 오차를 상태 벡터에 포함하여 추정한다.

**RTK vs PPP:**

| 특성 | RTK | PPP |
|------|-----|-----|
| 기준국 필요 | 예 (수 km 이내) | 아니오 |
| 수렴 시간 | 수 초~수십 초 | 수십 분 (전통), 수 분 (PPP-AR) |
| 정밀도 (수렴 후) | $\sim 2\,\text{cm}$ | $\sim 5\,\text{cm}$ |
| 커버리지 | 기준국 근처 | 전지구 |

**센서 퓨전에서의 GNSS 활용.** GNSS의 global position observation은 VIO/LIO의 장기 drift를 제한하거나 보정할 수 있다. 다만 outage·multipath·frame alignment·lever arm·time offset이 남으면 drift와 bias도 남는다. [LIO-SAM (Shan et al., 2020)](https://arxiv.org/abs/2007.00258)은 조건을 만족하는 GNSS measurement를 factor graph에 추가할 수 있는 대표적 예다. GNSS 관측을 fusion에 포함할 때는 세 가지를 확인해야 한다:

1. **좌표계 변환**: GNSS는 WGS84(위도, 경도, 타원체고)로 출력되며, 로보틱스 시스템은 ENU(East-North-Up) 또는 NED(North-East-Down) 로컬 프레임을 사용한다. 변환이 필요하다.
2. **공분산 활용**: GNSS 수신기가 출력하는 DOP(Dilution of Precision) 값이나 위치 공분산을 퓨전 시스템의 관측 공분산으로 활용한다.
3. **아웃라이어 처리**: 다중경로 환경에서 GNSS 측위 결과가 수십 미터 오차를 가질 수 있으므로, 로버스트 커널이나 $\chi^2$ 테스트로 이상 관측을 탐지/제거해야 한다.

```python
import numpy as np

def pseudorange_model(p_receiver, p_satellites, clock_bias):
    """의사거리 관측 모델 (단순화).
    
    Args:
        p_receiver: (3,) ECEF 좌표에서의 수신기 위치 [m]
        p_satellites: (N, 3) 각 위성의 ECEF 위치 [m]
        clock_bias: 수신기 시계 오차 [m] (c * dt_r)
    
    Returns:
        pseudoranges: (N,) 예측 의사거리 [m]
        H: (N, 4) 관측 자코비안 (선형화 기준점에서)
    """
    N = len(p_satellites)
    pseudoranges = np.zeros(N)
    H = np.zeros((N, 4))
    
    for i in range(N):
        diff = p_receiver - p_satellites[i]
        r = np.linalg.norm(diff)
        pseudoranges[i] = r + clock_bias
        
        # 자코비안: d(rho)/d(x,y,z,cb) 
        e = diff / r  # 단위 벡터 (수신기→위성 방향)
        H[i, :3] = e
        H[i, 3] = 1.0  # clock bias에 대한 편미분
    
    return pseudoranges, H


def geodetic_to_enu(lat, lon, alt, lat0, lon0, alt0):
    """WGS84 좌표를 로컬 ENU 좌표로 변환.
    
    Args:
        lat, lon: 변환할 점의 위도, 경도 [rad]
        alt: 타원체 고도 [m]
        lat0, lon0, alt0: 원점의 위도, 경도, 고도 [rad, rad, m]
    
    Returns:
        (3,) ENU 좌표 [m]
    """
    # 간이 변환 (WGS84 지구 반경 사용)
    a = 6378137.0  # WGS84 장반경
    e2 = 0.00669437999014  # 이심률 제곱
    
    sin_lat0 = np.sin(lat0)
    N0 = a / np.sqrt(1 - e2 * sin_lat0**2)
    
    dlat = lat - lat0
    dlon = lon - lon0
    dalt = alt - alt0
    
    east = (N0 + alt0) * np.cos(lat0) * dlon
    north = (N0 * (1 - e2) + alt0) * dlat
    up = dalt
    
    return np.array([east, north, up])
```

---

## 2.5 Radar 모델

레이더(Radar)는 전파(radio wave)를 사용하는 능동 센서로, 카메라와 LiDAR가 성능이 저하되는 **악천후(비, 안개, 눈, 먼지)** 환경에서도 안정적으로 동작한다. 도플러 효과를 이용하여 상대 **속도를 직접 측정**할 수 있다는 점도 다른 센서에는 없는 특성이다.

### 2.5.1 FMCW Radar 원리

자동차/로보틱스에 사용되는 레이더의 대부분은 **FMCW(Frequency Modulated Continuous Wave)** 방식이다. 연속적으로 주파수가 변하는 전파(chirp)를 발사하고, 반사파와의 주파수 차이(beat frequency)로부터 거리와 속도를 추출한다.

**Chirp 신호.** 송신 신호의 순시 주파수가 시간에 따라 선형적으로 증가한다:

$$f_{\text{TX}}(t) = f_0 + \frac{B}{T_c} t$$

여기서 $f_0$은 시작 주파수, $B$는 주파수 대역폭, $T_c$는 chirp 기간이다.

**Beat 주파수.** 거리 $R$에 있는 물체로부터의 반사파는 시간 지연 $\tau = 2R/c$ ($c$: 광속)을 가진다. 송신파와 수신파를 혼합(mixing)하면 beat 주파수가 발생한다:

$$f_b = \frac{2BR}{cT_c}$$

따라서 거리는:

$$R = \frac{f_b \cdot c \cdot T_c}{2B}$$

**거리 분해능.** 두 물체를 구별할 수 있는 최소 거리 차이는 대역폭이 결정한다:

$$\Delta R = \frac{c}{2B}$$

예를 들어, 77GHz 레이더에서 $B = 4\,\text{GHz}$이면 $\Delta R \approx 3.75\,\text{cm}$이다.

**도플러 측정.** 여러 chirp를 연속 발사하여 같은 물체로부터의 위상 변화를 관측하면 시선 방향(radial) 속도를 측정할 수 있다:

$$v_r = \frac{\lambda \cdot f_d}{2}$$

여기서 $f_d$는 도플러 주파수, $\lambda$는 반송파 파장이다. 77GHz 레이더의 $\lambda \approx 3.9\,\text{mm}$이므로, 매우 미세한 속도 변화도 감지 가능하다.

**Range-Doppler Map.** 여러 chirp의 2D FFT를 통해 거리-속도 2D 맵(Range-Doppler Map)을 생성한다. 이 맵의 각 피크가 하나의 반사체에 해당하며, 그 위치에서 거리와 시선 방향 속도를 동시에 읽을 수 있다.

**방위각 측정.** 다수의 수신 안테나(antenna array)로부터 수신 신호의 위상 차이를 이용하여 각도(angle of arrival)를 추정한다. 수평 배열은 수평 각도, 수직 배열은 수직 각도를 제공한다.

### 2.5.2 4D Imaging Radar 최신 동향

전통적인 자동차 레이더는 거리, 속도, 수평 각도의 3차원 정보를 제공하며, 수직 방향 분해능은 매우 낮았다. **4D 이미징 레이더(4D Imaging Radar)**는 거리, 속도(도플러), 수평 각도, **수직 각도**의 4차원 정보를 높은 분해능으로 제공하는 차세대 레이더이다.

4D 이미징 레이더는 MIMO(Multiple Input Multiple Output) 기술로 대규모 가상 안테나 어레이(virtual antenna array)를 구현한다. 예를 들어, 12개 송신 안테나 × 16개 수신 안테나 = 192개 가상 안테나로, 수평/수직 모두에서 충분한 각도 분해능을 달성한다.

**4D Radar vs LiDAR:**

| 특성 | 4D Imaging Radar | LiDAR |
|------|-----------------|-------|
| 악천후 동작 | 우수 | 취약 |
| 포인트 밀도 | 중간 (수천 점/프레임) | 높음 (수십만 점/프레임) |
| 속도 측정 | 직접 측정 (도플러) | 불가 (두 프레임 차분 필요) |
| 각도 분해능 | $\sim 1°$ | $\sim 0.1°$ |
| 비용 | 저~중 | 중~고 |
| 정적 물체 감지 | 제한적 (도플러 = 0) | 우수 |

**센서 퓨전에서의 레이더 활용.** 레이더의 도플러 측정은 센서 퓨전에서 고유한 가치를 제공한다:

1. **자기 속도 추정(Ego-velocity Estimation)**: 정적 환경의 반사체들로부터 측정된 시선 방향 속도를 피팅하여 자체 3D 속도를 추정할 수 있다. 이는 IMU의 가속도 적분보다 직접적이고 드리프트가 없다.

$$v_r^{(i)} = -\mathbf{e}^{(i)\top} \mathbf{v}_{\text{ego}}$$

여기서 $\mathbf{e}^{(i)}$는 $i$번째 반사체 방향의 단위 벡터, $\mathbf{v}_{\text{ego}}$는 자기 속도. 여러 반사체로부터의 관측으로 $\mathbf{v}_{\text{ego}}$를 최소자승법으로 추정한다.

2. **동적 물체 탐지**: 정적 환경에 대한 예측 도플러와 실제 관측의 차이로 이동 물체를 식별한다.
3. **악천후 보완**: 비나 안개로 카메라와 LiDAR 관측이 약해질 때 레이더가 독립적인 외향(exteroceptive) 관측을 제공한다. 성능은 강수량, 주파수 대역, 안테나와 신호 처리 설정에 따라 달라진다.

```python
import numpy as np

def fmcw_range_from_beat(f_beat, bandwidth, chirp_time, c=3e8):
    """FMCW 레이더의 beat 주파수로부터 거리 계산.
    
    Args:
        f_beat: beat 주파수 [Hz]
        bandwidth: 주파수 대역폭 [Hz]
        chirp_time: chirp 기간 [s]
        c: 광속 [m/s]
    
    Returns:
        range_m: 거리 [m]
    """
    return f_beat * c * chirp_time / (2 * bandwidth)


def estimate_ego_velocity(bearings, doppler_velocities):
    """정적 반사체의 도플러 관측으로부터 자기 속도 추정.
    
    v_doppler_i = -e_i^T @ v_ego  (정적 환경 가정)
    
    Args:
        bearings: (N, 3) 각 반사체 방향 단위 벡터
        doppler_velocities: (N,) 각 반사체의 시선 방향 속도 [m/s]
    
    Returns:
        v_ego: (3,) 추정 자기 속도 [m/s]
    """
    # -E @ v_ego = v_doppler  →  v_ego = -(E^T E)^{-1} E^T v_doppler
    E = bearings  # (N, 3)
    v_ego = -np.linalg.lstsq(E, doppler_velocities, rcond=None)[0]
    return v_ego
```

레이더는 다른 센서가 쉽게 줄 수 없는 속도 정보를 직접 측정한다. 이 특성 때문에 Ch.8의 멀티 센서 퓨전 아키텍처에서 레이더 활용이 늘고 있다.

---

## 2.6 기타 센서

센서 퓨전 시스템에서는 카메라, LiDAR, IMU, GNSS, Radar 외에도 다양한 보조 센서가 활용된다. 이들은 단독으로는 충분한 항법을 제공하지 못하지만, 메인 센서의 한계를 보완하는 데 유용하다.

### 2.6.1 Wheel Odometry (차륜 오도메트리)

차륜 인코더(wheel encoder)는 바퀴의 회전수를 측정하여 이동 거리를 추정하는 가장 기본적인 자기수용(proprioceptive) 센서이다.

**관측 모델.** 차동 구동(differential drive) 로봇의 경우, 좌우 바퀴의 회전 각도 $\Delta \theta_L, \Delta \theta_R$로부터:

$$\Delta s = \frac{r(\Delta \theta_L + \Delta \theta_R)}{2}, \quad \Delta \psi = \frac{r(\Delta \theta_R - \Delta \theta_L)}{d}$$

여기서 $r$은 바퀴 반경, $d$는 좌우 바퀴 간 거리(트레드), $\Delta s$는 전진 거리, $\Delta \psi$는 요 각도 변화이다.

**Ackermann 조향 모델** (자동차):

$$\dot{x} = v \cos\psi, \quad \dot{y} = v \sin\psi, \quad \dot{\psi} = \frac{v \tan\delta}{L}$$

여기서 $v$는 후륜 속도, $\delta$는 조향 각도, $L$은 휠베이스이다.

**슬립(Slip) 모델.** 실제 환경에서는 바퀴가 미끄러진다. 특히 젖은 노면, 비포장, 급가속/급감속 시 슬립이 크다. 슬립 비율(slip ratio)은:

$$s = \frac{v_{\text{wheel}} - v_{\text{actual}}}{v_{\text{actual}}}$$

슬립이 큰 환경에서 차륜 오도메트리의 신뢰도는 급격히 떨어진다. 센서 퓨전에서는 이를 **적응적 관측 공분산(adaptive observation covariance)**으로 처리한다 — 슬립이 감지되면 차륜 오도메트리의 불확실성을 크게 설정하여 다른 센서의 관측이 주도하도록 한다.

**센서 퓨전에서의 역할.** 차륜 오도메트리는 VIO/LIO 시스템에서 추가적인 속도/위치 관측으로 활용된다. 특히 직선 주행 시 단기 정밀도가 높아, 시각 특징점이 부족한 환경(터널, 긴 직선 도로)에서 IMU 드리프트를 보완한다. VINS-Mono의 확장판에서 차륜 오도메트리 팩터를 추가한 연구들이 있다.

### 2.6.2 기압계 (Barometer)

기압계는 대기압을 측정하여 **고도**를 추정하는 센서이다.

**관측 모델.** 국제 표준 대기(ISA)에 따른 기압-고도 관계:

$$h = \frac{T_0}{L}\left(1 - \left(\frac{P}{P_0}\right)^{\frac{RL}{g_0}}\right)$$

여기서:
- $P$: 측정 기압 [Pa]
- $P_0 = 101325\,\text{Pa}$: 해수면 표준 기압
- $T_0 = 288.15\,\text{K}$: 해수면 표준 온도
- $L = 0.0065\,\text{K/m}$: 온도 감률(lapse rate)
- $R = 287.053\,\text{J/(kg·K)}$: 공기의 기체 상수
- $g_0 = 9.80665\,\text{m/s}^2$: 표준 중력 가속도

간략화된 근사식 (저고도):

$$\Delta h \approx -\frac{\Delta P}{\rho g} \approx -\frac{\Delta P}{12.0}\,[\text{m}], \quad (\Delta P\text{는 Pa 단위})$$

해수면 근처에서 약 $8.5\,\text{Pa}$의 기압 변화가 $1\,\text{m}$의 고도 변화에 해당한다.

**노이즈 특성:**
- 단기 정밀도: $\pm 0.1\text{–}0.5\,\text{m}$ (매우 우수)
- 장기 정밀도: $\pm 1\text{–}10\,\text{m}$ (기상 변화에 의한 기압 변동)

**센서 퓨전에서의 역할.** 기압계는 수직 방향(고도)의 관측을 제공한다. 이는 IMU의 수직 드리프트를 억제하고, 특히 드론의 호버링 시 수직 위치 유지에 유용하다. 단, 기상 변화에 의한 장기 드리프트가 있으므로 절대 고도보다는 상대 고도 변화에 활용하는 것이 적절하다. 실내에서는 에어컨이나 문의 개폐에 의한 기압 변화에 주의해야 한다.

### 2.6.3 자력계 (Magnetometer)

자력계는 3축 자기장 벡터를 측정한다. Roll·pitch 보정과 hard/soft-iron calibration이 유효하고 주변 자기 교란이 작다면, 지구 자기장에 대한 heading을 얻을 수 있다. True north가 필요하면 지역 magnetic declination도 보정해야 한다.

**관측 모델.** 자력계 측정값은:

$$\tilde{\mathbf{m}} = \mathbf{R}_{bw} \mathbf{m}_w + \mathbf{b}_{\text{hard}} + \mathbf{S}_{\text{soft}} \mathbf{R}_{bw} \mathbf{m}_w + \mathbf{n}_m$$

여기서:
- $\mathbf{m}_w$: 월드 프레임에서의 지구 자기장 벡터 (크기 약 $25\text{–}65\,\mu\text{T}$, 위치에 따라 다름)
- $\mathbf{b}_{\text{hard}}$: **하드아이언(hard-iron) 바이어스** — 근처의 영구 자석이나 금속에 의한 상수 자기장
- $\mathbf{S}_{\text{soft}}$: **소프트아이언(soft-iron) 왜곡** — 주변 강자성 물질에 의한 자기장의 방향 의존적 왜곡
- $\mathbf{n}_m$: 측정 노이즈

**요 각도 추출.** 롤과 피치가 알려져 있으면 (가속도계로부터), 자력계 측정에서 요 각도를 계산할 수 있다:

$$\psi = \arctan2(m_y \cos\phi - m_z \sin\phi, \, m_x \cos\theta + m_y \sin\theta \sin\phi + m_z \sin\theta \cos\phi)$$

여기서 $\phi$는 롤, $\theta$는 피치, $m_x, m_y, m_z$는 하드아이언 보정된 자력계 측정이다.

**한계와 주의사항:**
- 실내, 차량 내, 건물 근처에서 자기장 왜곡이 매우 크다. 철근 콘크리트 건물 근처에서는 수십 도의 방위 오차가 발생할 수 있다.
- 모터, 전선 등의 전류에 의한 전자기 간섭에 민감하다.
- 센서 퓨전에서 자력계는 보조적 역할에 그치며, 신뢰도가 높을 때만 사용하거나 적응적 가중치를 적용한다.

### 2.6.4 UWB (Ultra-Wideband)

UWB는 매우 넓은 대역폭(500MHz 이상)의 극초단 펄스를 사용하여 노드 간 **거리**를 정밀하게 측정하는 무선 기술이다.

**관측 모델 (TWR — Two-Way Ranging):**

$$d = \frac{c \cdot (t_{\text{round}} - t_{\text{reply}})}{2} + n_d$$

여기서:
- $t_{\text{round}}$: 요청 펄스 전송에서 응답 펄스 수신까지의 시간
- $t_{\text{reply}}$: 응답 노드의 처리 시간
- $n_d$: 거리 노이즈, 일반적으로 $\sigma_d \approx 5\text{–}30\,\text{cm}$ (LOS 환경)

**NLOS(Non-Line-of-Sight) 문제.** UWB는 직접 가시선(LOS)이 확보된 환경에서는 높은 정밀도를 보이지만, 벽이나 장애물을 통과하면(NLOS) 신호가 지연되어 실제보다 먼 거리를 보고한다. UWB 기반 측위에서는 NLOS를 탐지하고 그 영향을 줄여야 한다.

**센서 퓨전에서의 역할.** UWB anchor를 환경에 미리 설치하면, 각 anchor까지의 range로 anchor frame 안의 position을 추정할 수 있다. Anchor 좌표를 survey해 global frame에 묶은 경우에만 global absolute position이 된다. VIO와 결합하면 geometry와 NLOS 조건이 충분한 구간에서 drift를 제한할 수 있다.

**관측 방정식 (삼변측량):**

$$d_i = \|\mathbf{p} - \mathbf{a}_i\| + n_{d,i}, \quad i = 1, \ldots, N_a$$

여기서 $\mathbf{p}$는 미지의 태그 위치, $\mathbf{a}_i$는 $i$번째 앵커의 알려진 위치이다. 3개 이상의 앵커로 2D 위치를, 4개 이상으로 3D 위치를 추정할 수 있다.

```python
import numpy as np

def uwb_trilateration(anchor_positions, ranges):
    """UWB 삼변측량 (최소자승법).
    
    Args:
        anchor_positions: (N, 3) 앵커 위치 [m]
        ranges: (N,) 각 앵커까지의 측정 거리 [m]
    
    Returns:
        position: (3,) 추정 위치 [m]
    """
    N = len(ranges)
    # 선형화: 첫 앵커를 기준으로 차분
    # ||p - a_i||^2 - ||p - a_0||^2 = d_i^2 - d_0^2
    # 2(a_0 - a_i)^T p = d_i^2 - d_0^2 - ||a_i||^2 + ||a_0||^2
    
    A = np.zeros((N - 1, 3))
    b = np.zeros(N - 1)
    
    a0 = anchor_positions[0]
    d0 = ranges[0]
    
    for i in range(1, N):
        ai = anchor_positions[i]
        di = ranges[i]
        A[i - 1] = 2 * (a0 - ai)
        b[i - 1] = di**2 - d0**2 - np.dot(ai, ai) + np.dot(a0, a0)
    
    # 최소자승 해
    position = np.linalg.lstsq(A, b, rcond=None)[0]
    return position
```

---

## 2.7 센서 모델링 요약

표는 각 센서의 관측 모델과 주요 특성을 정리한다.

| 센서 | 관측량 | 관측 모델 | 주요 노이즈원 | 검증 기준 |
|------|--------|----------|-------------|------------------|
| 카메라 | 2D 이미지 좌표 | $\pi(\mathbf{T} \cdot \mathbf{P})$ (핀홀 + 왜곡) | 검출 노이즈, 왜곡 잔차 | 목표 검출기와 별도 검증 영상의 재투영 잔차 |
| LiDAR | 3D 점 $(r, \alpha, \omega)$ | Range-bearing | 거리 노이즈, 빔 확산, 혼합 픽셀 | 장치 데이터시트 + 거리·표면·입사각별 잔차 시험 |
| IMU (자이로) | 각속도 $\boldsymbol{\omega}$ | $\tilde{\boldsymbol{\omega}} = \boldsymbol{\omega} + \mathbf{b}_g + \mathbf{n}_g$ | 바이어스, 랜덤 워크 | 장치 데이터시트 + Allan·온도·진동 시험 |
| IMU (가속도) | 비력 $\mathbf{a}$ | $\tilde{\mathbf{a}} = \mathbf{R}(\mathbf{a}-\mathbf{g}) + \mathbf{b}_a + \mathbf{n}_a$ | 바이어스, 랜덤 워크 | 장치 데이터시트 + Allan·온도·진동 시험 |
| GNSS | 의사거리 $\rho$ | $\rho = r + c\delta t_r + I + T + \epsilon$ | 다중경로, 전리층, 대류권 | 수신기·보정 상태별 목표 환경 주행 로그 |
| Radar | Range, Doppler, 방위 | FMCW beat frequency | 클러터, 다중경로 | 장치의 거리·속도·각도 해상도 + 타겟·RCS·기상별 시험 |
| Wheel Odom. | 회전수 | $\Delta s = r \Delta\theta$ | 슬립 | 노면·하중·슬립 조건별 반복 궤적 |
| 기압계 | 기압 $P$ | $h = f(P)$ | 기상 변화 | 정지·온도·기상 조건별 반복 측정 |
| 자력계 | 자기장 $\mathbf{m}$ | $\tilde{\mathbf{m}} = \mathbf{R}\mathbf{m}_w + \mathbf{b} + \mathbf{n}$ | 하드/소프트 아이언 | 캘리브레이션 잔차 + 운용 장소의 자기 간섭 시험 |
| UWB | 거리 $d$ | $d = \|\mathbf{p} - \mathbf{a}\| + n$ | NLOS | 앵커별 LOS/NLOS 거리 잔차 |

관측 모델이 정확해도 센서 간 상대 위치와 시간 동기가 어긋나면 퓨전 성능이 무너진다. 센서를 함께 쓰려면 센서 간 기하·시간 관계를 정하는 **캘리브레이션(Calibration)**이 필요하다.
