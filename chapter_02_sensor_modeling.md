# Ch.2 — Sensor Modeling

관측 모델은 센서가 내놓는 측정값을 상태 $\mathbf{x}$의 함수로 적은 식이다. Ch.4의 칼만 필터와 factor graph는 이 식을 관측 함수 $h(\mathbf{x})$로 그대로 쓴다.

> 센서의 개요는 robotics-practice Ch.2에 있다. 퓨전 알고리즘을 설계하려면 측정값이 실제 물리량과 어떤 식으로 이어지는지, 오차가 어떻게 분포하는지까지 알아야 한다.

---

## 2.1 카메라 관측 모델

카메라는 3D 세계의 점을 2D 이미지 평면에 투영하는 센서이다. 카메라 관측 모델은 이 투영 과정을 수학으로 나타낸다.

### 2.1.1 핀홀 카메라 모델 (Pinhole Camera Model)

핀홀 카메라 모델은 3D 점이 광학 중심(optical center)을 지나는 직선을 따라 이미지 평면에 투영된다고 가정한다.

카메라 좌표계에서 3D 점 $\mathbf{P}_c = [X_c, Y_c, Z_c]^\top$를 이미지 평면에 투영한 점 $\mathbf{p} = [u, v]^\top$은 다음과 같이 계산한다.

$$\begin{bmatrix} u \\ v \\ 1 \end{bmatrix} = \frac{1}{Z_c} \mathbf{K} \begin{bmatrix} X_c \\ Y_c \\ Z_c \end{bmatrix}$$

여기서 $\mathbf{K}$는 카메라 내부 파라미터 행렬(intrinsic matrix)이다.

$$\mathbf{K} = \begin{bmatrix} f_x & 0 & c_x \\ 0 & f_y & c_y \\ 0 & 0 & 1 \end{bmatrix}$$

파라미터의 뜻은 다음과 같다.
- $f_x, f_y$: 초점 거리(focal length). 물리적 초점 거리 $f$를 픽셀 크기 $(\Delta x, \Delta y)$로 나눈 값이다. $f_x = f / \Delta x$, $f_y = f / \Delta y$. 픽셀이 정사각형($\Delta x = \Delta y$)이면 $f_x = f_y$이고, 직사각형이면 두 값이 다르다.
- $c_x, c_y$: 주점(principal point). 광축(optical axis)이 이미지 평면과 만나는 점의 픽셀 좌표다. 설계상으로는 이미지 중심이지만 제조 공차 때문에 몇 픽셀 벗어나곤 한다.

동차 좌표(homogeneous coordinates)로 쓰면 다음과 같다.

$$s \begin{bmatrix} u \\ v \\ 1 \end{bmatrix} = \mathbf{K} [\mathbf{R} | \mathbf{t}] \begin{bmatrix} X_w \\ Y_w \\ Z_w \\ 1 \end{bmatrix}$$

여기서 $[\mathbf{R} | \mathbf{t}]$는 월드 좌표를 카메라 좌표로 바꾸는 외부 파라미터(extrinsic parameters)이고, $s = Z_c$는 깊이 배율이다.

투영 함수를 $\pi(\cdot)$로 쓰면 다음과 같다.

$$\mathbf{p} = \pi(\mathbf{P}_c) = \begin{bmatrix} f_x \frac{X_c}{Z_c} + c_x \\ f_y \frac{Y_c}{Z_c} + c_y \end{bmatrix}$$

상태 추정에서는 이 비선형 투영 함수의 자코비안(Jacobian)을 그대로 쓴다.

$$\frac{\partial \pi}{\partial \mathbf{P}_c} = \begin{bmatrix} \frac{f_x}{Z_c} & 0 & -\frac{f_x X_c}{Z_c^2} \\ 0 & \frac{f_y}{Z_c} & -\frac{f_y Y_c}{Z_c^2} \end{bmatrix}$$

EKF 기반 VIO는 이 $2 \times 3$ 자코비안으로 관측 모델을 선형화하고, 비선형 최적화는 이것으로 잔차의 자코비안을 만든다.

### 2.1.2 렌즈 왜곡 모델 (Lens Distortion Model)

실제 카메라의 렌즈는 핀홀 모델의 이상적 투영에서 벗어나는 왜곡(distortion)을 일으킨다. 왜곡을 무시하면 재투영 오차(reprojection error)가 수 픽셀에서 수십 픽셀까지 커지므로, 정밀한 센서 퓨전에서는 반드시 보정해야 한다.

#### Radial-Tangential 왜곡 (Brown-Conrady 모델)

OpenCV의 기본 왜곡 모델이고, 핀홀 카메라를 가정하는 SLAM·VIO에서 널리 쓴다. 시야각이 넓은 렌즈에는 fisheye나 omnidirectional 모델이 더 잘 맞는다.

정규화 이미지 좌표 $\mathbf{p}_n = [x_n, y_n]^\top = [X_c/Z_c, \, Y_c/Z_c]^\top$에서 왜곡은 다음과 같다.

$$r^2 = x_n^2 + y_n^2$$

**방사 왜곡(Radial Distortion):**

$$x_r = x_n (1 + k_1 r^2 + k_2 r^4 + k_3 r^6)$$
$$y_r = y_n (1 + k_1 r^2 + k_2 r^4 + k_3 r^6)$$

방사 왜곡은 렌즈 곡률 때문에 생기고, 이미지 중심에서 멀수록 커진다. $k_1 < 0$이면 중심에서 멀수록 배율이 줄어 중심을 지나지 않는 직선이 바깥으로 불룩하게 휘는 배럴 왜곡(barrel distortion)이 생긴다. $k_1 > 0$이면 중심에서 멀수록 배율이 커져 그런 직선이 중심 쪽으로 오목하게 휘는 핀쿠션 왜곡(pincushion distortion)이 생긴다.

**접선 왜곡(Tangential Distortion):**

$$x_t = 2p_1 x_n y_n + p_2 (r^2 + 2x_n^2)$$
$$y_t = p_1 (r^2 + 2y_n^2) + 2p_2 x_n y_n$$

접선 왜곡은 렌즈가 이미지 센서와 완벽히 평행하지 않을 때(decentering) 생긴다. 크기는 방사 왜곡보다 훨씬 작으나, 정밀 응용에서는 모델에 넣는다.

**왜곡 좌표 계산**

$$\begin{bmatrix} x_d \\ y_d \end{bmatrix} = \begin{bmatrix} x_r + x_t \\ y_r + y_t \end{bmatrix}$$

$$\begin{bmatrix} u \\ v \end{bmatrix} = \begin{bmatrix} f_x x_d + c_x \\ f_y y_d + c_y \end{bmatrix}$$

왜곡 파라미터 $[k_1, k_2, p_1, p_2, k_3]$는 [Zhang (2000)](https://doi.org/10.1109/34.888718)의 방법을 따르는 OpenCV `calibrateCamera()`로 추정한다(§3.1.3).

#### 어안 (Fisheye / Equidistant) 모델

시야각(FoV)이 180°에 이르거나 넘는 어안 렌즈에는 radial-tangential 모델이 맞지 않는다. 이미지 가장자리에서 $r$이 매우 커져 다항식 근사가 발산하기 때문이다.

[Kannala & Brandt (2006)](https://doi.org/10.1109/TPAMI.2006.153)의 generic camera model은 다음과 같이 정의한다.

입사각 $\theta$는 3D 점이 광축과 이루는 각이다.

$$\theta = \operatorname{atan2}\left(\sqrt{X_c^2 + Y_c^2},\; Z_c\right)$$

$Z_c < 0$인 뒤쪽 반구(180° 이상)까지 다루려면 인자가 둘인 atan2를 써야 하고, 방향 단위벡터는 $X_c/\sqrt{X_c^2+Y_c^2}$ 꼴로 구한다. 아래 파이썬 구현도 그렇게 한다.

왜곡된 반경 $r_d$는 $\theta$의 홀수 차 다항식으로 둔다.

$$r_d = k_1 \theta + k_2 \theta^3 + k_3 \theta^5 + k_4 \theta^7 + k_5 \theta^9$$

여기서 $r_d$는 무차원 정규화 반경이다. 순수 등거리(equidistant) 투영이면 $r_d = \theta$, 곧 $k_1 = 1$, $k_2 = k_3 = \cdots = 0$이다. 픽셀 단위 초점 거리는 아래 투영식의 $f_x$, $f_y$에서 곱한다.

투영 좌표는 다음과 같다.

$$\begin{bmatrix} u \\ v \end{bmatrix} = \begin{bmatrix} f_x \cdot r_d \cdot \frac{X_c}{\sqrt{X_c^2 + Y_c^2}} + c_x \\ f_y \cdot r_d \cdot \frac{Y_c}{\sqrt{X_c^2 + Y_c^2}} + c_y \end{bmatrix}$$

어안 렌즈는 한 번에 넓게 보므로 주변을 파악하기 좋고, [VINS-Mono](https://arxiv.org/abs/1708.03852)와 Basalt 같은 VIO 시스템이 지원한다. 캘리브레이션에는 [Scaramuzza et al. (2006)](https://rpg.ifi.uzh.ch/docs/IROS06_scaramuzza.pdf)의 OCamCalib이나 Kalibr를 쓴다.

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

재투영 오차(reprojection error)는 특징점 기반 VO·VIO와 번들 조정이 카메라 관측을 비용 함수에 넣는 방식이다.

재투영 오차는 카메라 포즈 $\mathbf{T}_{cw} = [\mathbf{R}|\mathbf{t}]$로 3D 랜드마크 $\mathbf{P}_w$를 이미지에 투영한 예측 좌표 $\hat{\mathbf{p}}$와 실제로 관측한 특징점 좌표 $\mathbf{p}_{\text{obs}}$의 차이다.

$$\mathbf{e}_{\text{reproj}} = \mathbf{p}_{\text{obs}} - \pi(\mathbf{T}_{cw} \cdot \mathbf{P}_w)$$

여기서 $\pi(\cdot)$는 위에서 정의한 투영 함수(왜곡 포함)이다.

**번들 조정(Bundle Adjustment)**은 모든 카메라 포즈와 랜드마크에 대해 재투영 오차의 합을 최소화한다.

$$\min_{\{\mathbf{T}_i\}, \{\mathbf{P}_j\}} \sum_{i,j} \rho\left(\| \mathbf{p}_{ij} - \pi(\mathbf{T}_i \cdot \mathbf{P}_j) \|^2_{\mathbf{\Sigma}_{ij}}\right)$$

여기서:
- $\mathbf{p}_{ij}$: 카메라 $i$가 관측한 랜드마크 $j$의 이미지 좌표
- $\mathbf{\Sigma}_{ij}$: 관측 노이즈 공분산 (등방성 모델에서 $\sigma^2 \mathbf{I}_2$, $\sigma \approx 1$ 픽셀)
- $\rho(\cdot)$: 로버스트 커널(Huber, Cauchy 등). 아웃라이어의 영향을 줄인다.

재투영 오차는 보통 $\sigma = 0.5 \sim 2$ 픽셀인 가우시안으로 둔다. 이 값은 특징점 검출기가 얼마나 정밀한지에 달렸고, 코너를 서브픽셀 단위로 찾으면 $\sigma \approx 0.5$ 픽셀까지 내려간다.

### 2.1.4 롤링 셔터 모델 (Rolling Shutter Model)

대부분의 저가 카메라(스마트폰, 웹캠)는 CMOS 이미지 센서를 쓰고, 롤링 셔터(rolling shutter) 방식으로 영상을 얻는다. 글로벌 셔터(global shutter)가 모든 픽셀을 동시에 노출하는 것과 달리, 롤링 셔터는 행(row) 단위로 순차적으로 노출한다. 그래서 이미지 위쪽과 아래쪽은 서로 다른 순간에 찍힌다.

$k$번째 행의 노출 시각은 다음과 같다.

$$t_k = t_0 + k \cdot t_r$$

여기서 $t_0$은 첫 번째 행의 노출 시각, $t_r$은 한 행을 읽는 데 걸리는 시간(row readout time)이다. 이미지 높이가 $H$이면 첫 행과 마지막 행의 노출 시작 시각은 $(H-1) \cdot t_r$만큼 차이 나고, 각 행이 빛을 받는 노출 시간은 이와 따로 정해진다. 전체를 읽는 데는 대략 $H \cdot t_r$, 곧 수 밀리초에서 수십 밀리초가 걸린다.

카메라가 움직이는 동안 롤링 셔터로 찍으면 다음 문제가 생긴다.
- **기하 왜곡**: 수직선이 기울거나 움직이는 물체가 젤리처럼 휘어 보인다.
- **특징점 위치 오차**: 특징점마다 찍힌 순간의 카메라 포즈가 달라서, 모든 행을 한 순간에 찍었다고 보는 글로벌 셔터 투영 모델이 틀어진다.

롤링 셔터 투영 모델은 특징점마다 그 특징점이 놓인 행 $v$가 노출된 시각의 카메라 포즈로 투영한다.

$$\mathbf{p}_i = \pi\left(\mathbf{T}(t_{v_i}) \cdot \mathbf{P}_i\right)$$

여기서 $\mathbf{T}(t_{v_i})$는 $i$번째 특징점의 행 $v_i$에 대응하는 시각의 월드→카메라 변환이다. 첫 행과 마지막 행의 포즈를 알면, 그 사이 속도(twist)가 일정하다고 보고 다음처럼 보간한다($H>1$).

$$\mathbf{T}(t_{v_i}) = \mathbf{T}(t_0) \cdot \text{Exp}\left(\frac{v_i}{H-1} \cdot \text{Log}(\mathbf{T}(t_0)^{-1} \mathbf{T}(t_0 + (H-1) \cdot t_r))\right)$$

여기서 $\text{Exp}$와 $\text{Log}$는 $SE(3)$ 리 군 위의 지수/로그 맵이다. 아래 예제는 카메라→월드 변환을 받아 회전은 SLERP로, 이동은 선형으로 따로 보간하는 근사다.

롤링 셔터 보정은 일부 VIO 시스템이 선택 사항으로 둔다. [VINS-Mono](https://arxiv.org/abs/1708.03852)에는 readout time 설정이 있고, ORB-SLAM3는 글로벌 셔터를 가정한다. 스마트폰이나 드론 카메라처럼 저가 센서로 빠르게 움직일 때 특히 필요하다.

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
    alpha = v_row / (H - 1) if H > 1 else 0.0  # 첫 행 0, 마지막 행 1
    
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

LiDAR는 레이저 펄스를 쏘고 반사광의 비행시간(Time-of-Flight)이나 위상차로 거리를 계산하는 능동 센서다.

### 2.2.1 Range-Bearing 모델

LiDAR는 레이저 빔마다 거리(range)와 방향(bearing)을 한 쌍으로 잰다. 3D LiDAR의 점 하나는 구면 좌표 $(r, \alpha, \omega)$로 나타낸다.

- $r$: 거리 (range)
- $\alpha$: 수평 각도 (azimuth)
- $\omega$: 수직 각도 (elevation)

LiDAR 좌표계의 3D 점 $\mathbf{P}_L = [x, y, z]^\top$를 관측하는 모델은 다음과 같다.

$$\begin{aligned}
r &= \sqrt{x^2 + y^2 + z^2} + n_r \\
\alpha &= \arctan2(y, x) + n_\alpha \\
\omega &= \arcsin\left(\frac{z}{\sqrt{x^2 + y^2 + z^2}}\right) + n_\omega
\end{aligned}$$

여기서 $n_r, n_\alpha, n_\omega$는 각각 거리, 수평각, 수직각의 관측 노이즈이다.

**노이즈 특성**
- **거리 노이즈** $n_r$: 데이터시트 값으로 보면 기계식 [Velodyne VLP-16](https://www.mapix.com/wp-content/uploads/2018/07/63-9229_Rev-H_Puck-_Datasheet_Web.pdf)의 거리 정확도는 최대 $\pm 3\,\text{cm}$(typical, 대부분 채널의 벽면 시험 기준), [Livox Avia](https://www.livoxtech.com/avia/specs)의 거리 정밀도는 반사율 80% 표적 20 m에서 $2\,\text{cm}$(1σ)다. 멀어질수록 빔이 퍼지고(beam divergence) 돌아오는 에너지가 줄어 노이즈가 커진다.
- **각도 노이즈** $n_\alpha, n_\omega$: [Livox Avia](https://www.livoxtech.com/avia/specs) 데이터시트의 각도 정밀도는 $0.05°$ 미만(1σ)이다. 작은 각이라도 멀리서는 큰 위치 오차가 된다. 50m 거리에서 $0.1°$ 틀리면 옆으로 약 $8.7\,\text{cm}$ 어긋난다.

구면 좌표를 직교 좌표로 바꾸면 다음과 같다.

$$\mathbf{P}_L = \begin{bmatrix} r \cos\omega \cos\alpha \\ r \cos\omega \sin\alpha \\ r \sin\omega \end{bmatrix}$$

**빔 확산(Beam Divergence).** LiDAR 레이저 빔은 가는 원뿔 모양으로 퍼진다. 퍼지는 각은 데이터시트 기준으로 VLP-16이 수평 $0.18°$·수직 $0.07°$([Rev-F 사양서](https://www.hypertech.co.il/wp-content/uploads/2015/12/63-9229_Rev-F_Puck__Spec_Sheet_Web.pdf)), Livox Avia가 수평 $0.03°$·수직 $0.28°$다. 멀리 갈수록 빔이 닿는 면적(footprint)이 넓어져, 측정값은 그 면적 안의 평균 거리가 된다. 물체 경계에서 빔이 물체와 배경에 함께 걸리면 두 거리 사이 어딘가에 가짜 점이 찍히는데, 이를 혼합 픽셀(mixed pixel) 효과라고 한다.

### 2.2.2 Motion Distortion (모션 왜곡)

기계식 스피닝(spinning) LiDAR는 센서가 360° 회전하면서 레이저를 발사한다. Velodyne VLP-16은 기본 설정(10Hz)에서 한 바퀴에 약 100ms가 걸린다. 그동안 차량이나 드론이 움직이면 한 스캔 안의 점들이 서로 다른 위치에서 찍힌다. 이것을 모션 왜곡(motion distortion)이라 하고, 이를 바로잡는 일을 ego-motion compensation이라 한다.

카메라의 롤링 셔터와 같은 구조의 문제다. 스캔의 $i$번째 점을 시각 $t_i$에 쟀다면, 이 점을 기준 시각 $t_0$의 좌표계로 옮긴다.

$$\mathbf{P}_L^{(t_0)} = \mathbf{T}(t_0)^{-1} \cdot \mathbf{T}(t_i) \cdot \mathbf{P}_L^{(t_i)}$$

여기서 $\mathbf{T}(t_i)$는 시각 $t_i$의 LiDAR 포즈(LiDAR→월드 변환)다.

**보정 방법**

1. **IMU로 보간**: 고주파 IMU 측정으로 스캔 동안의 포즈 변화를 보간한다. LIO-SAM, FAST-LIO2 같은 LIO 시스템이 이렇게 한다.
2. **직전 프레임 속도로 외삽**: 직전 프레임에서 추정한 속도가 스캔 동안 그대로라고 본다(constant velocity).
3. **연속 시간(continuous-time) 방법**: 궤적을 연속 함수로 두고 점마다 찍힌 시각의 포즈를 계산한다. B-스플라인을 흔히 쓰고, [CT-ICP (Dellenbach et al., 2022)](https://arxiv.org/abs/2109.12979)는 스캔 처음과 끝 두 포즈 사이를 보간하는 방식이다.

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
        R_i = slerp(np.clip(timestamps[i], t_start, t_end)).as_matrix()
        t_i = (1 - alpha) * T_start[:3, 3] + alpha * T_end[:3, 3]
        
        # 해당 시각 좌표계 → 시작 시각 좌표계
        p_world = R_i @ points[i] + t_i
        corrected[i] = T_start[:3, :3].T @ (p_world - T_start[:3, 3])
    
    return corrected
```

### 2.2.3 Spinning vs Solid-State LiDAR가 퓨전에 미치는 영향

**기계식 스피닝 LiDAR**(Velodyne, Ouster, Hesai)는 수평으로 360°를 보고, 한 스캔이 빙 둘러싼 고리 모양의 포인트 클라우드가 된다. LOAM과 그 후속 알고리즘은 이 구조를 전제로 만들었다. 수평 스캔 라인을 따라 edge·planar feature를 뽑고, 사방의 관측으로 6-DoF 포즈를 추정한다.

**시야가 좁은 Livox LiDAR**(Mid-40/70, Avia, HAP 등)에는 같은 궤적을 되풀이하지 않는 비반복(non-repetitive) 스캔 모드가 있다. Mid-40은 안에서 프리즘을 돌려 빔 방향을 바꾸므로 기계적 회전부가 아예 없지는 않다. Avia에서는 반복 스캔 모드도 고를 수 있다. 비반복 모드에서는 좁은 시야(Mid-70은 지름 약 70.4°의 원) 안이 시간이 지날수록 촘촘히 채워진다.



| 특성 | 스피닝 LiDAR | 제한된 FoV의 비반복 스캔 모드 |
|------|-------------|-------------------|
| FoV | 360° 수평 | 제한적 (40°~120°) |
| 스캔 패턴 | 반복적 (수평 라인) | 비반복적 (로제트, 리사주 등) |
| 단일 프레임 밀도 | 균일 | 불균일 (시간에 따라 증가) |
| Feature 추출 | 스캔 라인 기반 가능 | 스캔 라인 구조 없음 |
| 적합한 알고리즘 | LOAM, LeGO-LOAM | FAST-LIO/LIO2 (점 단위 처리) |

LOAM의 edge·planar feature 추출은 스캔 라인이 있어야 하므로 비반복 스캔에 그대로 쓰기 어렵다. FAST-LIO와 [FAST-LIO2](https://arxiv.org/abs/2107.06829)는 한 스캔의 점별 잔차를 모아 iterated EKF로 갱신한다. 특히 FAST-LIO2는 원시 점을 지도에 바로 정합하므로 스캔 라인 구조가 필요 없고, 비반복 스캔 LiDAR에도 잘 맞는다. 점이 들어올 때마다 상태를 갱신하는 Point-LIO는 §7.3.4에서 다룬다. [FAST-LIVO2 (Zheng et al., 2025)](https://arxiv.org/abs/2408.14035)는 같은 틀에 카메라를 더해, LiDAR·IMU·카메라 관측을 iterated EKF 하나에서 차례로 갱신하고 LiDAR 점과 영상 픽셀을 특징 추출 없이 direct 방식으로 쓴다.

---

## 2.3 IMU 모델

IMU(Inertial Measurement Unit)는 3축 가속도계(accelerometer)와 3축 자이로스코프(gyroscope)로 이루어진다. 100Hz–1kHz로 재므로 카메라·LiDAR 프레임 사이를 메우고, 초기화와 스케일 추정에도 쓴다. 이 측정을 퓨전에 넣으려면 오차 모델이 있어야 한다.

### 2.3.1 자이로스코프 오차 모델

자이로스코프는 3축 각속도 $\boldsymbol{\omega}$를 측정한다. 측정값 $\tilde{\boldsymbol{\omega}}$는 다음과 같이 둔다.

$$\tilde{\boldsymbol{\omega}} = \boldsymbol{\omega} + \mathbf{b}_g + \mathbf{n}_g$$

- $\boldsymbol{\omega}$: IMU 바디 프레임에서 본 실제 각속도
- $\mathbf{b}_g$: 바이어스(bias). 거의 일정하지만 시간에 따라 천천히 변하는 오프셋
- $\mathbf{n}_g$: 측정 노이즈. 백색 가우시안 노이즈(Additive White Gaussian Noise, AWGN)

**바이어스의 동역학.** 바이어스가 천천히 변하는 모습은 랜덤 워크(random walk)로 둔다.

$$\dot{\mathbf{b}}_g = \mathbf{n}_{bg}$$

여기서 $\mathbf{n}_{bg} \sim \mathcal{N}(\mathbf{0}, \sigma_{bg}^2 \mathbf{I})$는 바이어스를 움직이는 구동 노이즈다. 이산 시간으로 쓰면 다음과 같다.

$$\mathbf{b}_{g,k+1} = \mathbf{b}_{g,k} + \sigma_{bg} \sqrt{\Delta t} \cdot \mathbf{w}_k, \quad \mathbf{w}_k \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$$

자이로스코프의 노이즈 파라미터는 장치와 운용 조건에 맞춰 정한다.

- **백색 노이즈 밀도**: 쓰는 장치의 데이터시트와 Allan deviation에서 기울기가 $-1/2$인 구간으로 추정한다. 단위와 연속 시간 노이즈 밀도의 정의가 코드에서 이산 공분산으로 바꾸는 방식과 맞는지 확인한다.
- **In-run bias stability**: 충분히 길게 정지해 모은 데이터의 Allan deviation에서 기울기가 거의 0인 구간을 본다. 온도, 장착 응력, 진동, 측정 시간에 따라 달라지므로 실제로 장착한 상태에서 잰 값을 쓴다.
- **바이어스 랜덤 워크 구동 노이즈**: 긴 τ 쪽에서 기울기가 양수인 구간으로 추정한다. 측정 노이즈 $\sigma_g$와 바이어스 구동 노이즈 $\sigma_{bg}$를 헷갈리지 말고, 연속 시간 밀도를 이산 공분산으로 어떻게 바꿨는지 적어 둔다.

### 2.3.2 가속도계 오차 모델

가속도계는 3축 비력(specific force) $\mathbf{a}$를 측정한다. 비력은 실제 가속도에서 중력 가속도를 뺀 값이다. 측정값 $\tilde{\mathbf{a}}$는 다음과 같다.

$$\tilde{\mathbf{a}} = \mathbf{R}_{bw}(\mathbf{a}_w - \mathbf{g}_w) + \mathbf{b}_a + \mathbf{n}_a$$

- $\mathbf{a}_w$: 월드 프레임에서 본 실제 가속도
- $\mathbf{g}_w = [0, 0, -g]^\top$: 중력 벡터 ($g \approx 9.81\,\text{m/s}^2$)
- $\mathbf{R}_{bw}$: 월드→바디 회전 행렬
- $\mathbf{b}_a$: 가속도계 바이어스
- $\mathbf{n}_a \sim \mathcal{N}(\mathbf{0}, \sigma_a^2 \mathbf{I})$: 측정 노이즈. $\sigma_a$는 연속 시간 노이즈 밀도($\text{m/s}^2/\sqrt{\text{Hz}}$)이므로, 주기 $\Delta t$로 샘플한 이산 노이즈의 표준편차는 $\sigma_a/\sqrt{\Delta t}$다. 바이어스 랜덤 워크는 거꾸로 $\sqrt{\Delta t}$를 곱한다.

**중력.** 가속도계는 중력도 잰다. 위를 z로 두면 가만히 있어도 $[0, 0, g]^\top$이 나온다. 이 값으로 롤(roll)과 피치(pitch)를 추정한다. 요(yaw)는 중력 축을 도는 회전이라 중력으로는 알 수 없다(unobservable). 그래서 VIO/LIO는 처음 요를 임의로 정하고 그로부터 상대 회전을 추정한다. 전역 요가 필요하면 자력계나 이미 만든 지도처럼 방향 기준을 주는 외부 관측을 더한다.

**바이어스 동역학.** 자이로스코프와 마찬가지로 랜덤 워크로 둔다.

$$\dot{\mathbf{b}}_a = \mathbf{n}_{ba}, \quad \mathbf{n}_{ba} \sim \mathcal{N}(\mathbf{0}, \sigma_{ba}^2 \mathbf{I})$$

가속도계도 같은 절차로 파라미터를 정한다.

- **백색 노이즈 밀도**: 장치 데이터시트와 Allan deviation의 $-1/2$ 기울기 구간을 함께 보고, $\text{m/s}^2/\sqrt{\text{Hz}}$, $\text{mg}/\sqrt{\text{Hz}}$, VRW 사이를 어떻게 환산했는지 적어 둔다.
- **In-run bias stability**: 정지·온도 시험에서 Allan 곡선이 거의 평평한 구간과 반복 실험 사이의 편차를 본다. 중력이 축에 비스듬히 실리는 투영 오차나 장착 자세 변화가 바이어스로 섞이지 않게 한다.
- **바이어스 랜덤 워크 구동 노이즈**: 장시간 데이터의 해당 기울기에서 $\sigma_{ba}$를 추정하고, 필터가 쓰는 샘플 주기에 맞춰 이산 공분산으로 바꾼다.

### 2.3.3 Allan Variance

Allan Variance는 IMU 노이즈를 분석하는 표준 방법이다. 가만히 둔 IMU에서 몇 시간 동안 데이터를 모으고, 평균 시간(cluster time) $\tau$를 바꿔 가며 분산을 계산한다.

Allan Variance $\sigma^2(\tau)$는 다음과 같이 정의한다.

$$\sigma^2(\tau) = \frac{1}{2} \langle (\bar{y}_{k+1} - \bar{y}_k)^2 \rangle$$

여기서 $\bar{y}_k$는 $k$번째 구간(길이 $\tau$)의 평균 출력이다.

Allan Deviation $\sigma(\tau)$를 log-log 그래프로 그리면 기울기로 노이즈 종류를 가린다.

| 기울기 | 노이즈 종류 | 물리적 의미 |
|--------|-----------|-----------|
| $-1/2$ | 각도/속도 랜덤 워크 (ARW/VRW) | 백색 노이즈 $\mathbf{n}_g, \mathbf{n}_a$ |
| $0$ | 바이어스 불안정성 (Bias Instability) | 플리커 노이즈. 이 구간의 최솟값에서 bias instability를 읽는다 |
| $+1/2$ | 레이트 랜덤 워크 (RRW) | 바이어스 랜덤 워크 $\mathbf{n}_{bg}, \mathbf{n}_{ba}$ |

**데이터시트 읽는 법.** IMU 데이터시트에서는 센서 퓨전에 필요한 파라미터를 아래 순서로 뽑는다.

1. **각도 랜덤 워크(Angular Random Walk, ARW)**: 단위 $°/\sqrt{\text{hr}}$ 또는 $\text{rad/s}/\sqrt{\text{Hz}}$. Allan Deviation 그래프에서 기울기 $-1/2$ 구간을 $\tau = 1\,\text{s}$까지 이어 읽은 값이다. 이 값이 $\sigma_g$다.
2. **속도 랜덤 워크(Velocity Random Walk, VRW)**: 단위 $\text{m/s}/\sqrt{\text{hr}}$ 또는 $\text{m/s}^2/\sqrt{\text{Hz}}$. 가속도계의 백색 노이즈 밀도이고, 이 값이 $\sigma_a$다.
3. **바이어스 안정성(In-run Bias Stability)**: 데이터시트는 흔히 Allan Deviation 곡선의 최솟값을 적는다. IEEE Std 952의 bias instability 계수 $B$는 평탄 구간에서 $\sigma(\tau) \approx 0.664B$이므로, 최솟값을 0.664로 나누면 $B$가 나온다. §3.4.4의 Allan 코드가 돌려주는 `bias_instability`가 이 $B$다. 정지 상태에서 출력을 시간 평균했을 때 얼마나 안정한지를 나타낼 뿐, 융합 시스템이 바이어스를 얼마나 정확히 추정할 수 있는지와는 다른 지표다.
4. **레이트 랜덤 워크(Rate Random Walk)**: 바이어스가 시간에 따라 떠도는 정도이고, $\sigma_{bg}, \sigma_{ba}$에 해당한다.

```python
import numpy as np

def compute_allan_variance(data, dt, max_cluster_size=None):
    """Allan Variance 계산.
    
    Args:
        data: (N,) 정지 상태에서 수집한 IMU 한 축의 데이터
        dt: 샘플링 주기 (초)
        max_cluster_size: 최대 클러스터 크기 (None이면 N//10: 가장 긴 tau에서도 클러스터 10개 이상)
    
    Returns:
        taus: 클러스터 시간 배열
        adevs: Allan Deviation 배열
    """
    N = len(data)
    if max_cluster_size is None:
        max_cluster_size = N // 10  # 클러스터가 2~3개뿐인 긴 tau는 추정 분산이 커서 최솟값을 과소 추정한다
    
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

IMU 측정으로 항법 상태(위치, 속도, 자세)를 적분하는 방정식을 스트랩다운 관성 항법 방정식(Strapdown Navigation Equation)이라 한다. "스트랩다운(strapdown)"은 센서를 기체에 바로 묶어(strapped down) 기계식 짐벌 없이 소프트웨어로 좌표를 변환한다는 뜻이다.

지구 자전을 무시하고 월드 프레임을 관성 프레임으로 보는 VIO/LIO의 통상 가정에서, 월드 프레임 상태 $[\mathbf{R}, \mathbf{v}, \mathbf{p}]$의 연속 시간 동역학은 아래와 같다. 지구에 붙은 항법 프레임(NED/ENU)을 쓰면 지구 자전율과 전달률(transport rate) 항이 자세·속도 식에 더 붙는다.

$$\dot{\mathbf{R}} = \mathbf{R} \cdot [\tilde{\boldsymbol{\omega}} - \mathbf{b}_g - \mathbf{n}_g]_\times$$

$$\dot{\mathbf{v}} = \mathbf{R} (\tilde{\mathbf{a}} - \mathbf{b}_a - \mathbf{n}_a) + \mathbf{g}$$

$$\dot{\mathbf{p}} = \mathbf{v}$$

- $\mathbf{R} \in SO(3)$: 바디→월드 회전 행렬
- $\mathbf{v} \in \mathbb{R}^3$: 월드 프레임 속도
- $\mathbf{p} \in \mathbb{R}^3$: 월드 프레임 위치
- $[\cdot]_\times$: 반대칭(skew-symmetric) 행렬. 벡터 외적을 행렬 곱으로 쓴 것

$$[\boldsymbol{\omega}]_\times = \begin{bmatrix} 0 & -\omega_z & \omega_y \\ \omega_z & 0 & -\omega_x \\ -\omega_y & \omega_x & 0 \end{bmatrix}$$

**이산 시간 적분.** 시각 $t_k$에서 $t_{k+1} = t_k + \Delta t$까지 상태를 전파하면 다음과 같다.

$$\mathbf{R}_{k+1} = \mathbf{R}_k \cdot \text{Exp}\left((\tilde{\boldsymbol{\omega}}_k - \mathbf{b}_{g,k}) \Delta t\right)$$

$$\mathbf{v}_{k+1} = \mathbf{v}_k + \mathbf{g} \Delta t + \mathbf{R}_k (\tilde{\mathbf{a}}_k - \mathbf{b}_{a,k}) \Delta t$$

$$\mathbf{p}_{k+1} = \mathbf{p}_k + \mathbf{v}_k \Delta t + \frac{1}{2} \mathbf{g} \Delta t^2 + \frac{1}{2} \mathbf{R}_k (\tilde{\mathbf{a}}_k - \mathbf{b}_{a,k}) \Delta t^2$$

여기서 $\text{Exp}(\boldsymbol{\phi})$는 회전 벡터 $\boldsymbol{\phi}$를 회전 행렬로 바꾸는 $SO(3)$ 지수 맵이고, Rodrigues 공식으로 계산한다.

$$\text{Exp}(\boldsymbol{\phi}) = \mathbf{I} + \frac{\sin\|\boldsymbol{\phi}\|}{\|\boldsymbol{\phi}\|} [\boldsymbol{\phi}]_\times + \frac{1 - \cos\|\boldsymbol{\phi}\|}{\|\boldsymbol{\phi}\|^2} [\boldsymbol{\phi}]_\times^2$$

**중간점(midpoint) 적분.** 위의 오일러 적분은 1차 정확도다. 연이은 두 IMU 측정의 평균을 쓰면 2차 정확도가 된다.

$$\bar{\boldsymbol{\omega}} = \frac{1}{2}(\tilde{\boldsymbol{\omega}}_k + \tilde{\boldsymbol{\omega}}_{k+1}) - \mathbf{b}_{g,k}$$

$$\mathbf{R}_{k+1} = \mathbf{R}_k \cdot \text{Exp}(\bar{\boldsymbol{\omega}} \Delta t)$$

$$\bar{\mathbf{a}} = \frac{1}{2}\left(\mathbf{R}_k(\tilde{\mathbf{a}}_k - \mathbf{b}_{a,k}) + \mathbf{R}_{k+1}(\tilde{\mathbf{a}}_{k+1} - \mathbf{b}_{a,k})\right)$$

$$\mathbf{v}_{k+1} = \mathbf{v}_k + (\bar{\mathbf{a}} + \mathbf{g}) \Delta t$$

$$\mathbf{p}_{k+1} = \mathbf{p}_k + \mathbf{v}_k \Delta t + \frac{1}{2}(\bar{\mathbf{a}} + \mathbf{g}) \Delta t^2$$

VINS-Mono는 중간점 적분을 쓴다. FAST-LIO2를 비롯한 다른 시스템의 전파 식과 이산화 방식은 코드마다 다르니 직접 확인해야 한다. IMU 출력률이 높아도 각가속도와 샘플링 지터가 크면 적분 오차가 남는다. 중간점, RK4, 사전적분 가운데 무엇을 쓸지는 목표 운동과 허용 오차에 맞춰 시험해 정한다.

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

**드리프트 크기.** 바이어스를 보정하지 않고 스트랩다운 적분을 하면 오차가 얼마나 쌓이는지 계산해 보자. 가속도계 바이어스를 $b_a = 0.01\,\text{m/s}^2$(약 $1\,\text{mg}$)로 일정하게 둔다.

- 1초 후 위치 오차: $\frac{1}{2} \times 0.01 \times 1^2 = 0.005\,\text{m}$ (5mm)
- 10초 후: $\frac{1}{2} \times 0.01 \times 100 = 0.5\,\text{m}$
- 60초 후: $\frac{1}{2} \times 0.01 \times 3600 = 18\,\text{m}$

다른 센서 없이 저가 MEMS IMU만 쓰면 짧은 임무에서도 금방 허용 오차를 넘는다. 얼마나 버티는지는 IMU 등급, 주변 온도, 운동, 허용 오차에 따라 다르다. 고성능 INS는 더 오래 버티지만 오차는 계속 쌓인다. 그래서 VIO/LIO 시스템은 바이어스 $\mathbf{b}_g, \mathbf{b}_a$를 상태 벡터에 넣고 다른 센서 관측으로 계속 갱신한다.

학습으로 관성 오도메트리를 하는 연구도 있다. [AirIO (Qiu et al., RA-L 2025)](https://arxiv.org/abs/2501.15659)는 IMU 데이터를 전역 좌표로 바꾸지 않고 바디 프레임 그대로 네트워크에 넣었다. 논문은 입력 표현만 이렇게 바꿔도 드론 데이터셋 세 개(시뮬레이션 하나 포함)에서 정확도가 평균 66.7% 나아졌다고 보고했다.

### 2.3.5 IMU 등급 분류

항법급, 전술급, 산업용, 소비자용 같은 등급 이름은 제조사와 분야마다 경계가 다르다. MEMS는 만드는 기술을 가리키는 말이라 MEMS IMU 안에도 성능이 여러 층이다. 그러니 등급 이름이나 가격표보다는 실제로 쓸 장치의 사양을 보고, 목표 플랫폼에서 시험해 고른다.

| 선택 기준 | 확인할 항목 | 검증 자료 | 설계 영향 |
|----------|------------|----------|----------|
| 백색 노이즈 | 자이로·가속도계 노이즈 밀도, ARW/VRW 단위 | 장치 데이터시트 + Allan 시험 | 단기 상태 전파 오차 |
| 바이어스 | In-run stability, 랜덤 워크, 온도 계수 | 정지·온도 반복 시험 + Allan 시험 | 무보조 드리프트와 외부 관측 주기 |
| 동적 범위 | 측정 범위, 대역폭, 포화·클리핑 | 목표 운동·진동 프로파일 시험 | 급격한 운동 중 관측 손실 |
| 시간 특성 | 출력률, 타임스탬프, clock/trigger 지원 | 지연·지터 측정 | 시간 보정과 퓨전 일관성 |
| 환경 내성 | 온도, 진동, 충격, 장착 응력 | 제조사 qualification + 목표 플랫폼 시험 | 반복성과 장기 안정성 |

어떤 외부 센서를 얼마나 자주 넣을지는 이렇게 잰 값을 임무의 허용 오차와 맞대어 정한다. 가격은 시기와 수량에 따라 달라지므로 그때의 견적에 캘리브레이션·동기화 비용까지 더해 비교한다.

---

## 2.4 GNSS 모델

GNSS(Global Navigation Satellite System)는 GPS, GLONASS, Galileo, BeiDou 등 위성 항법 시스템을 함께 이르는 말이다.

### 2.4.1 의사거리 관측 모델 (Pseudorange)

GNSS 수신기는 위성마다 의사거리(pseudorange) $\rho$를 잰다. 의사거리는 위성까지의 실제 거리에 수신기 시계 오차를 비롯한 여러 오차가 더해진 값이다.

$$\rho^s = r^s + c \cdot \delta t_r - c \cdot \delta t^s + I^s + T^s + \epsilon_\rho$$

- $r^s = \|\mathbf{p}_r - \mathbf{p}^s\|$: 수신기 위치 $\mathbf{p}_r$와 위성 $s$ 위치 $\mathbf{p}^s$ 사이의 기하학적 거리
- $c \cdot \delta t_r$: 수신기 시계 오차 (미지수, 위치와 함께 추정)
- $c \cdot \delta t^s$: 위성 시계 오차 (항법 메시지에서 보정 가능)
- $I^s$: 전리층(ionosphere) 지연. 전리층의 자유 전자 때문에 신호가 늦어진다. L1/L2 이중 주파수 수신기로 1차 항을 없앤다.
- $T^s$: 대류권(troposphere) 지연. 대기의 공기와 수증기 때문에 늦어진다. Saastamoinen 모델 같은 것으로 보정한다.
- $\epsilon_\rho$: 열잡음, 다중경로 같은 남은 노이즈. [ESA Navipedia](https://gssc.esa.int/navipedia/index.php/Receiver_noise)에 따르면 수신기 잡음은 코드 칩 길이의 약 1% 이하로, GPS C/A 코드에서 최대 3 m, P 코드에서 약 30 cm다.

**측위(positioning) 원리.** 위성 넷 이상의 의사거리를 재면 수신기 위치 $(x, y, z)$와 시계 오차 $\delta t_r$, 미지수 넷을 푼다. 이 비선형 방정식을 최소자승법이나 칼만 필터로 푸는 것이 표준 단독 측위(Single Point Positioning, SPP)다.

**다중경로(Multipath).** 도심에서 위성 신호가 건물에 반사된 경로로 수신기에 들어오면 의사거리에 수 미터에서 수십 미터 오차가 더해진다. 이 오차는 모델로 나타내기 어렵고 장소마다 달라서, GNSS를 퓨전에 넣을 때는 아웃라이어 처리를 특히 신경 써야 한다.

### 2.4.2 반송파 위상 관측 모델 (Carrier Phase)

반송파 위상 가운데 한 파장보다 작은 부분(fractional phase)은 잡음이 의사거리보다 훨씬 작아 밀리미터 단위까지 내려간다. 위치를 그만큼 정확히 얻으려면 정수 모호성, 위상이 끊겨 건너뛰는 cycle slip, 다중경로, 대기 오차를 함께 풀어야 한다. L1 반송파(1575.42 MHz)의 파장은 약 19 cm다.

$$\Phi^s = r^s + c \cdot \delta t_r - c \cdot \delta t^s + \lambda N^s - I^s + T^s + \epsilon_\Phi$$

여기서:
- $\lambda$: 반송파 파장
- $N^s$: 정수 모호성(integer ambiguity) — 수신기와 위성 사이에 든 온전한 파장의 개수다. RTK의 fixed 해와 PPP-AR는 이 값을 정수로 확정하고, 기존 PPP는 실수(float)로 추정한다.
- $\epsilon_\Phi$: 반송파 위상 노이즈. 반송파 파장의 약 1%로 수 mm 수준이다([ESA Navipedia](https://gssc.esa.int/navipedia/index.php/Receiver_noise)).

전리층은 코드(군속도)는 늦추고 반송파 위상(위상속도)은 앞당기므로, 위상 식에서는 전리층 항의 부호가 의사거리와 반대다. 이중 주파수 조합으로 1차 전리층 항은 없애지만 고차 항(higher-order term)과 다른 오차는 남는다.

### 2.4.3 RTK (Real-Time Kinematic)

RTK는 기준국이나 네트워크 보정 정보와 관측을 차분(differencing)해 공통 오차를 줄이고, 반송파 위상의 정수 모호성을 실시간으로 푼다. 기준국과 멀어질수록(baseline이 길수록) 두 곳의 대기 오차가 덜 닮아 차분 효과가 줄어든다. 정수 모호성이 고정(fix)되고 신호와 보정 정보가 좋으면 센티미터급 정확도가 나온다.

**이중 차분(Double Difference).** 위성 $s$와 기준 위성 $r$에 대해 기준국과 이동국의 관측을 이중으로 차분하면 다음과 같다.

$$\nabla\Delta\Phi_{br}^{sr} = \nabla\Delta r_{br}^{sr} + \lambda \nabla\Delta N_{br}^{sr} + \epsilon$$

이중 차분하면 수신기와 위성의 시계 오차가 지워지고, 기준국이 가까우면 전리층·대류권 오차도 거의 사라진다. 남는 미지수는 위치에 따라 정해지는 기하 거리와 정수 모호성이고, 정수 모호성은 LAMBDA 알고리즘 같은 방법으로 푼다.

### 2.4.4 PPP (Precise Point Positioning)

PPP는 기준국 없이 수신기 하나로 센티미터급 위치를 얻는다. 정밀 궤도(precise orbit)와 정밀 시계(precise clock) 보정을 외부 서비스에서 받아 위성 쪽 오차를 없애고, 전리층·대류권 오차는 상태 벡터에 넣어 추정한다.

**RTK vs PPP:**

| 특성 | RTK | PPP |
|------|-----|-----|
| 기준국 필요 | 예 (단일 기준국은 보통 10–20 km 이내) | 아니오 |
| 수렴 시간 | 수 초~수십 초 | 수십 분 (전통), 수 분 (PPP-AR) |
| 정밀도 (수렴 후) | $\sim 2\,\text{cm}$ | $\sim 5\,\text{cm}$ |
| 커버리지 | 기준국 근처 | 전지구 |

**센서 퓨전에 GNSS 넣기.** GNSS가 주는 전역 위치로 VIO/LIO의 장기 드리프트를 묶어 둘 수 있다. 다만 신호 끊김, 다중경로, 좌표계 정렬, 안테나와 IMU 사이 거리(lever arm), 시간 오프셋이 해결되지 않으면 드리프트와 바이어스가 남는다. [LIO-SAM (Shan et al., 2020)](https://arxiv.org/abs/2007.00258)은 추정한 위치 공분산이 GNSS에서 받은 위치 공분산보다 클 때만 GNSS factor를 더한다. GNSS 관측을 넣을 때는 다음을 확인한다.

1. **좌표계 변환**: GNSS는 WGS84(위도, 경도, 타원체고)로 위치를 내고, 로봇은 대개 ENU(East-North-Up)나 NED(North-East-Down) 지역 좌표계를 쓰므로 변환해야 한다.
2. **공분산**: 수신기가 내는 위치 공분산을 관측 좌표계로 바꿔 퓨전에 넣는다. DOP(Dilution of Precision)는 위성 배치만으로 정해지는 무차원 지표이고, 위치 불확실성은 DOP에 관측 오차 크기를 곱해 어림한다.
3. **아웃라이어**: 다중경로가 심하면 위치가 수십 미터 틀리기도 하므로, 로버스트 커널이나 $\chi^2$ 검정으로 이상 관측을 걸러 낸다.

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
        e = diff / r  # 단위 벡터 (위성→수신기 방향; diff = p_receiver - p_satellite)
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
    north = (N0 * (1 - e2) / (1 - e2 * sin_lat0**2) + alt0) * dlat  # 자오선 곡률반경 M
    up = dalt
    
    return np.array([east, north, up])
```

---

## 2.5 Radar 모델

레이더(Radar)는 전파(radio wave)를 쏘는 능동 센서다. 비, 안개, 눈, 먼지 때문에 카메라와 LiDAR 성능이 떨어지는 날씨에도 비교적 안정적으로 동작한다. 도플러 효과로 상대 속도를 바로 재는데, 카메라와 ToF 방식 LiDAR는 이렇게 재지 못한다.

### 2.5.1 FMCW Radar 원리

자동차와 로봇에 쓰는 밀리미터파 레이더는 흔히 FMCW(Frequency Modulated Continuous Wave) 방식이다. [Han et al.의 4D radar 서베이](https://arxiv.org/abs/2306.04242) 2절은 지금 나와 있는 4D radar 대부분이 이 방식을 쓴다고 정리한다. 주파수가 계속 바뀌는 전파(chirp)를 쏘고, 쏜 전파와 돌아온 전파의 주파수 차이(beat frequency)로 거리와 속도를 구한다.

**Chirp 신호.** 송신 신호의 순간 주파수는 시간에 따라 선형으로 올라간다.

$$f_{\text{TX}}(t) = f_0 + \frac{B}{T_c} t$$

여기서 $f_0$은 시작 주파수, $B$는 주파수 대역폭, $T_c$는 chirp 기간이다.

**Beat 주파수.** 거리 $R$에 있는 물체에서 돌아온 전파는 $\tau = 2R/c$($c$: 광속)만큼 늦게 온다. 송신파와 수신파를 섞으면(mixing) 두 주파수의 차인 beat 주파수가 나온다.

$$f_b = \frac{2BR}{cT_c}$$

따라서 거리는 다음과 같다.

$$R = \frac{f_b \cdot c \cdot T_c}{2B}$$

**거리 분해능.** 두 물체를 따로 구별할 수 있는 최소 거리 차이는 대역폭으로 정해진다.

$$\Delta R = \frac{c}{2B}$$

예를 들어 77GHz 레이더에서 $B = 4\,\text{GHz}$이면 $\Delta R \approx 3.75\,\text{cm}$다.

**도플러 측정.** chirp를 연달아 쏘아 같은 물체에서 돌아온 신호의 위상 변화를 보면 시선 방향(radial) 속도를 잴 수 있다.

$$v_r = \frac{\lambda \cdot f_d}{2}$$

여기서 $f_d$는 도플러 주파수, $\lambda$는 반송파 파장이다. 77GHz 레이더는 $\lambda \approx 3.9\,\text{mm}$로 파장이 짧아 작은 속도 차이도 잡아낸다.

**Range-Doppler Map.** 여러 chirp를 모아 2D FFT를 하면 거리-속도 2D 맵(Range-Doppler Map)이 나온다. 맵의 피크 하나가 반사체 하나에 해당하고, 피크 위치에서 거리와 시선 방향 속도를 함께 읽는다.

**방위각 측정.** 여러 수신 안테나(antenna array)에 들어온 신호의 위상 차이로 도래각(angle of arrival)을 추정한다. 안테나를 수평으로 늘어놓으면 수평 각도를, 수직으로 늘어놓으면 수직 각도를 잰다.

### 2.5.2 4D Imaging Radar 최신 동향

기존 자동차 레이더는 거리, 속도, 수평 각도를 쟀고 수직 방향은 거의 가려내지 못했다. 4D 이미징 레이더(4D Imaging Radar)는 여기에 수직 각도를 더해 네 가지를 높은 분해능으로 잰다.

4D 이미징 레이더는 MIMO(Multiple Input Multiple Output)로 큰 가상 안테나 배열(virtual antenna array)을 만든다. 송신 안테나 12개와 수신 안테나 16개를 쓰면 가상 채널이 12×16=192개다. 수평·수직 각도 분해능은 가상 배열의 각 방향 개구와 배치에 따라 달라진다.

**4D Radar와 LiDAR 비교**

| 특성 | 4D Imaging Radar | LiDAR |
|------|-----------------|-------|
| 악천후 동작 | 상대적으로 강함 (강수·radome 영향은 받음) | 저하됨 |
| 포인트 밀도 | LiDAR보다 성김 (저채널 LiDAR에 견줄 수준) | 높음 |
| 속도 측정 | 직접 측정 (도플러) | 불가 (두 프레임 차분 필요, FMCW LiDAR는 예외) |
| 수평 각도 분해능 | 약 0.5°–1.5° | 0.1°–0.4° (VLP-16) |
| 비용 | 저~중 | 중~고 |
| 정적 물체 감지 | 가능 (검출·클러터 제거 설정에 영향받음) | 우수 |

포인트 밀도 행의 출처는 [Han et al. 서베이](https://arxiv.org/abs/2306.04242)의 1절과 4절이다. 4D radar의 수평 각도 분해능은 같은 서베이 Table III에 실린 4D radar 데이터셋 장치 대부분(VoD, TJ4DRadSet, K-Radar, NTU4DRadLM 등)의 값이고, LiDAR 값은 [VLP-16 데이터시트](https://www.mapix.com/wp-content/uploads/2018/07/63-9229_Rev-H_Puck-_Datasheet_Web.pdf)의 회전 속도별 범위다.

**센서 퓨전에 레이더 넣기.** 도플러 측정은 다른 센서로는 바로 얻지 못하는 정보를 준다.

1. **자기 속도 추정(Ego-velocity Estimation)**: 멈춰 있는 반사체들의 시선 방향 속도를 모아 맞추면 자기 3D 속도를 추정한다. IMU 가속도를 적분하지 않고 속도를 바로 재므로 드리프트가 쌓이지 않는다.

$$v_r^{(i)} = -\mathbf{e}^{(i)\top} \mathbf{v}_{\text{ego}}$$

여기서 $\mathbf{e}^{(i)}$는 $i$번째 반사체 쪽 단위 벡터, $\mathbf{v}_{\text{ego}}$는 자기 속도다. 반사체 여럿의 관측으로 $\mathbf{v}_{\text{ego}}$를 최소자승 추정한다. 움직이는 물체에서 온 점은 RANSAC 같은 방법으로 먼저 걸러 낸다.

2. **움직이는 물체 찾기**: 주변이 멈춰 있다고 보고 예측한 도플러와 실제 값이 다르면 그 점은 움직이는 물체에서 온 것이다.
3. **악천후**: 비나 안개로 카메라와 LiDAR가 약해져도 레이더는 바깥을 계속 잰다(exteroceptive). 얼마나 버티는지는 강수량, 주파수 대역, 안테나와 신호 처리 설정에 따라 다르다.

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

---

## 2.6 기타 센서

센서 퓨전 시스템에서는 카메라, LiDAR, IMU, GNSS, Radar 외에도 여러 보조 센서를 함께 쓴다. 메인 센서의 한계를 보완하는 용도다.

### 2.6.1 Wheel Odometry (차륜 오도메트리)

차륜 인코더(wheel encoder)는 바퀴가 돈 양을 재서 이동 거리를 추정하는 고유수용(proprioceptive) 센서다.

**관측 모델.** 차동 구동(differential drive) 로봇은 좌우 바퀴가 돈 각도 $\Delta \theta_L, \Delta \theta_R$로 이동을 구한다.

$$\Delta s = \frac{r(\Delta \theta_L + \Delta \theta_R)}{2}, \quad \Delta \psi = \frac{r(\Delta \theta_R - \Delta \theta_L)}{d}$$

여기서 $r$은 바퀴 반경, $d$는 좌우 바퀴 간 거리(트레드), $\Delta s$는 전진 거리, $\Delta \psi$는 요 각도 변화이다.

**Ackermann 조향 모델**(자동차)

$$\dot{x} = v \cos\psi, \quad \dot{y} = v \sin\psi, \quad \dot{\psi} = \frac{v \tan\delta}{L}$$

여기서 $v$는 후륜 속도, $\delta$는 조향 각도, $L$은 휠베이스이다.

**슬립(Slip) 모델.** 실제로는 바퀴가 미끄러진다. 젖은 노면이나 비포장길, 급가속·급감속 때 특히 많이 미끄러진다. 슬립 비율(slip ratio)은 다음과 같다.

$$s = \frac{v_{\text{wheel}} - v_{\text{actual}}}{v_{\text{actual}}}$$

슬립이 크면 차륜 오도메트리를 믿기 어렵다. 그래서 퓨전에서는 관측 공분산을 상황에 맞게 바꾼다(adaptive observation covariance). 슬립을 감지하면 차륜 오도메트리의 불확실성을 크게 잡아 다른 센서 관측의 비중을 높인다.

**센서 퓨전에 넣기.** VIO/LIO는 차륜 오도메트리를 속도나 위치 관측으로 더한다. 곧게 달릴 때 단기 정밀도가 높아서, 터널이나 긴 직선 도로처럼 시각 특징점이 모자란 곳에서 IMU 드리프트를 줄인다. [Liu et al. (IROS 2019)](https://doi.org/10.1109/IROS40897.2019.8967607)은 IMU와 차륜 인코더 측정을 함께 사전적분(IMU-odometer pre-integration)해 VINS-Mono 방식의 sliding window 최적화에 넣었다.

### 2.6.2 기압계 (Barometer)

기압계는 대기압을 재서 고도를 추정한다.

**관측 모델.** 국제 표준 대기(ISA)에서 기압과 고도의 관계는 다음과 같다.

$$h = \frac{T_0}{L}\left(1 - \left(\frac{P}{P_0}\right)^{\frac{RL}{g_0}}\right)$$

여기서:
- $P$: 측정 기압 [Pa]
- $P_0 = 101325\,\text{Pa}$: 해수면 표준 기압
- $T_0 = 288.15\,\text{K}$: 해수면 표준 온도
- $L = 0.0065\,\text{K/m}$: 온도 감률(lapse rate)
- $R = 287.053\,\text{J/(kg}\cdot\text{K)}$: 공기의 기체 상수
- $g_0 = 9.80665\,\text{m/s}^2$: 표준 중력 가속도

고도가 낮으면 다음처럼 근사한다.

$$\Delta h \approx -\frac{\Delta P}{\rho g} \approx -\frac{\Delta P}{12.0}\,[\text{m}], \quad (\Delta P\text{는 Pa 단위})$$

해수면 근처에서는 고도 $1\,\text{m}$에 기압이 약 $12.0\,\text{Pa}$ 바뀐다.

**노이즈 특성**
- 단기 정밀도: $\pm 0.1\text{–}0.5\,\text{m}$
- 장기 정밀도: 날씨에 따라 기압이 바뀌어 $\pm 1\text{–}10\,\text{m}$

**센서 퓨전에 넣기.** 기압계는 IMU의 수직 드리프트를 누르는 데 쓰고, 드론이 제자리 비행할 때 높이를 지키는 데 특히 쓸모 있다. 날씨 때문에 길게 보면 값이 떠돌므로, 절대 고도보다 상대 고도 변화를 보는 데 쓴다. 실내에서는 냉난방이나 문 여닫기로도 기압이 바뀐다.

### 2.6.3 자력계 (Magnetometer)

자력계는 자기장 벡터를 3축으로 잰다. 롤·피치를 알고 하드·소프트 아이언 보정(hard/soft-iron calibration)을 마쳤으며 주변 자기 교란이 작으면, 자북 기준 방위를 얻는다. 진북(true north)이 필요하면 그 지역의 자기 편차(magnetic declination)도 보정한다.

**관측 모델.** 자력계 측정값은 다음과 같다.

$$\tilde{\mathbf{m}} = \mathbf{R}_{bw} \mathbf{m}_w + \mathbf{b}_{\text{hard}} + \mathbf{S}_{\text{soft}} \mathbf{R}_{bw} \mathbf{m}_w + \mathbf{n}_m$$

여기서:
- $\mathbf{m}_w$: 월드 프레임에서 본 지구 자기장 벡터. 크기는 장소에 따라 약 $25\text{–}65\,\mu\text{T}$
- $\mathbf{b}_{\text{hard}}$: 하드 아이언(hard-iron) 바이어스. 기체에 붙은 영구 자석이나 자화된 금속이 만드는 일정한 자기장
- $\mathbf{S}_{\text{soft}}$: 소프트 아이언(soft-iron) 왜곡. 주변 강자성체가 자기장을 방향에 따라 휘게 하는 효과
- $\mathbf{n}_m$: 측정 노이즈

**요 각도 구하기.** 가속도계로 롤과 피치를 알면 자력계 측정으로 요 각도를 계산한다.

$$\psi = \arctan2(m_z \sin\phi - m_y \cos\phi, \, m_x \cos\theta + m_y \sin\theta \sin\phi + m_z \sin\theta \cos\phi)$$

여기서 $\phi$는 롤, $\theta$는 피치, $m_x, m_y, m_z$는 하드 아이언을 보정한 자력계 측정이다. 이 식은 NED 월드 좌표계, x 전방·y 우측·z 하방 기체축, Z-Y-X 오일러각을 전제하며, $\psi$는 자북에서 시계 방향으로 잰 각이다.

**한계**
- 실내, 차 안, 건물 근처에서는 자기장이 크게 휜다. 철골·콘크리트 건물 안에서는 방위가 수십 도 틀리기도 한다.
- 모터나 전선에 흐르는 전류가 만드는 자기장에도 흔들린다.
- 그래서 퓨전에서는 자력계를 보조로만 쓰고, 믿을 만할 때만 넣거나 가중치를 상황에 맞게 바꾼다.

### 2.6.4 UWB (Ultra-Wideband)

UWB는 매우 넓은 대역폭(500MHz 이상)의 극초단 펄스를 써서 노드 사이 거리를 정밀하게 재는 무선 기술이다.

**관측 모델(TWR, Two-Way Ranging)**

$$d = \frac{c \cdot (t_{\text{round}} - t_{\text{reply}})}{2} + n_d$$

여기서:
- $t_{\text{round}}$: 요청 펄스를 보내고 응답 펄스를 받기까지 걸린 시간
- $t_{\text{reply}}$: 응답 노드가 처리하는 데 쓴 시간
- $n_d$: 거리 노이즈. 가시선(LOS)이 막히지 않으면 대개 $\sigma_d \approx 5\text{–}30\,\text{cm}$

**NLOS(Non-Line-of-Sight) 문제.** 신호가 벽이나 장애물을 지나면(NLOS) 늦게 도착해 실제보다 먼 거리가 나온다. UWB 기반 측위에서는 NLOS를 탐지하고 그 영향을 줄여야 한다.

**센서 퓨전에 넣기.** UWB 앵커(anchor)를 미리 설치해 두면, 앵커마다 잰 거리(range)로 앵커 좌표계(anchor frame) 안의 위치를 추정한다. 앵커 좌표를 측량해 전역 좌표계(global frame)에 묶어 두어야 전역 절대 위치(global absolute position)가 된다. VIO와 함께 쓰면 앵커 배치가 좋고 가시선이 막히지 않은(NLOS가 적은) 구간에서 드리프트를 묶어 둔다.

**관측 방정식(삼변측량)**

$$d_i = \|\mathbf{p} - \mathbf{a}_i\| + n_{d,i}, \quad i = 1, \ldots, N_a$$

여기서 $\mathbf{p}$는 미지의 태그 위치, $\mathbf{a}_i$는 $i$번째 앵커의 알려진 위치이다. 앵커가 셋 이상이면 2D 위치를, 한 평면에 놓이지 않은 앵커가 넷 이상이면 3D 위치를 추정한다. 앵커가 모두 같은 높이에 있으면 아래 선형화에서 $z$를 알 수 없다.

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

센서별 관측 모델과 주요 특성은 다음과 같다.

| 센서 | 관측량 | 관측 모델 | 주요 노이즈원 | 검증 기준 |
|------|--------|----------|-------------|------------------|
| 카메라 | 2D 이미지 좌표 | $\pi(\mathbf{T} \cdot \mathbf{P})$ (핀홀 + 왜곡) | 검출 노이즈, 왜곡 잔차 | 목표 검출기와 별도 검증 영상의 재투영 잔차 |
| LiDAR | 3D 점 $(r, \alpha, \omega)$ | Range-bearing | 거리 노이즈, 빔 확산, 혼합 픽셀 | 장치 데이터시트 + 거리·표면·입사각별 잔차 시험 |
| IMU (자이로) | 각속도 $\boldsymbol{\omega}$ | $\tilde{\boldsymbol{\omega}} = \boldsymbol{\omega} + \mathbf{b}_g + \mathbf{n}_g$ | 바이어스, 랜덤 워크 | 장치 데이터시트 + Allan·온도·진동 시험 |
| IMU (가속도) | 비력 $\mathbf{a}$ | $\tilde{\mathbf{a}} = \mathbf{R}_{bw}(\mathbf{a}_w-\mathbf{g}_w) + \mathbf{b}_a + \mathbf{n}_a$ | 바이어스, 랜덤 워크 | 장치 데이터시트 + Allan·온도·진동 시험 |
| GNSS | 의사거리 $\rho$, 반송파 위상 $\Phi$ | $\rho = r + c\delta t_r + I + T + \epsilon_\rho$, $\Phi = r + c\delta t_r + \lambda N - I + T + \epsilon_\Phi$ | 다중경로, 전리층, 대류권, 정수 모호성, cycle slip | 수신기·보정 상태별 목표 환경 주행 로그 |
| Radar | Range, Doppler, 방위 | FMCW beat frequency | 클러터, 다중경로 | 장치의 거리·속도·각도 해상도 + 타겟·RCS·기상별 시험 |
| Wheel Odom. | 회전수 | $\Delta s = r(\Delta\theta_L + \Delta\theta_R)/2$ | 슬립 | 노면·하중·슬립 조건별 반복 궤적 |
| 기압계 | 기압 $P$ | $h = f(P)$ | 기상 변화 | 정지·온도·기상 조건별 반복 측정 |
| 자력계 | 자기장 $\mathbf{m}$ | $\tilde{\mathbf{m}} = (\mathbf{I}+\mathbf{S})\mathbf{R}_{bw}\mathbf{m}_w + \mathbf{b} + \mathbf{n}$ | 하드/소프트 아이언 | 캘리브레이션 잔차 + 운용 장소의 자기 간섭 시험 |
| UWB | 거리 $d$ | $d = \|\mathbf{p} - \mathbf{a}\| + n$ | NLOS | 앵커별 LOS/NLOS 거리 잔차 |

관측 모델이 정확해도 센서 사이의 상대 위치나 시간이 어긋나면 퓨전 결과가 틀어진다. 센서 사이의 기하·시간 관계를 정하는 일이 캘리브레이션(Calibration)이고, Ch.3에서 다룬다.
