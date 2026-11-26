# 2장 센서 모델링

센서 융합 알고리즘은 각 센서의 측정값을 곧바로 믿지 않는다. 대신 현재 추정한 상태 $\mathbf{x}$(포즈, 속도, 바이어스 등)가 맞다면 센서가 어떤 값을 내놓아야 하는지 예측하고, 실제 측정값과의 차이(잔차)로 상태를 보정한다. 이때 상태 $\mathbf{x}$를 센서의 측정값으로 연결하는 함수가 관측 모델 $h(\mathbf{x})$이며, 칼만 필터와 factor graph 최적화는 이 함수와 그 오차 분포를 핵심 입력으로 사용한다(→ 4장).

> 융합을 제대로 설계하려면 센서의 측정값이 실제 물리량과 어떤 방정식으로 이어지는지, 그리고 노이즈와 왜곡이 어떤 분포를 따르는지까지 파악해야 한다(센서 개요 → robotics-practice 2장).

---

## 2.1 카메라 관측 모델

카메라는 3차원 공간의 점을 2차원 이미지 평면으로 투영하는 센서다. 한 번의 촬영으로 광선 방향의 깊이 정보를 잃는 대신, 이미지 평면 위의 픽셀 좌표를 매우 조밀하게 얻는다. 따라서 시각 기반 추정기에서는 후보 포즈와 3D 랜드마크 좌표를 투영 함수에 넣어 예측 픽셀 위치를 계산하고, 실제 검출된 특징점 위치와 비교해 오차를 줄여 나간다.

### 2.1.1 핀홀 카메라 모델 (Pinhole Camera Model)

가장 기본이 되는 핀홀(pinhole) 모델은 3D 점이 렌즈의 광학 중심(optical center)을 지나는 직선을 따라 이미지 평면에 곧바로 맺힌다고 이상화한다. 카메라 좌표계의 3D 점 $\mathbf{P}_c = [X_c, Y_c, Z_c]^\top$를 이미지 평면 좌표 $\mathbf{p} = [u, v]^\top$로 옮기는 식은 다음과 같다.

$$\begin{bmatrix} u \\ v \\ 1 \end{bmatrix} = \frac{1}{Z_c} \mathbf{K} \begin{bmatrix} X_c \\ Y_c \\ Z_c \end{bmatrix}$$

여기서 행렬 $\mathbf{K}$는 카메라 고유의 기하 성질을 담은 내부 파라미터 행렬(intrinsic matrix)이다.

$$\mathbf{K} = \begin{bmatrix} f_x & 0 & c_x \\ 0 & f_y & c_y \\ 0 & 0 & 1 \end{bmatrix}$$

행렬을 이루는 네 파라미터의 물리적 의미는 이렇다.
- $f_x, f_y$: 픽셀 단위로 환산한 초점 거리(focal length)다. 물리적 초점 거리 $f$를 이미지 센서의 픽셀 크기 $(\Delta x, \Delta y)$로 나눈 값($f_x = f / \Delta x$, $f_y = f / \Delta y$)이며, 픽셀이 정사각형($\Delta x = \Delta y$)이면 $f_x = f_y$, 직사각형이면 두 값이 서로 다르다.
- $c_x, c_y$: 주점(principal point)으로, 광축(optical axis)이 이미지 평면과 수직으로 만나는 지점의 픽셀 좌표다. 설계상으로는 이미지 한가운데를 겨냥하지만 실제 조립 공차 때문에 중심에서 몇 픽셀씩 어긋나기 마련이다.

실제 추정 문제에서는 랜드마크 좌표가 월드 좌표계 $[X_w, Y_w, Z_w]^\top$로 주어지므로, 먼저 외부 파라미터(extrinsic parameters) $[\mathbf{R} | \mathbf{t}]$를 곱해 카메라 좌표계로 옮긴 뒤 투영한다. 이를 동차 좌표(homogeneous coordinates)로 한 줄에 적으면 아래 식이 된다.

$$s \begin{bmatrix} u \\ v \\ 1 \end{bmatrix} = \mathbf{K} [\mathbf{R} | \mathbf{t}] \begin{bmatrix} X_w \\ Y_w \\ Z_w \\ 1 \end{bmatrix}$$

이때 앞의 배율 $s = Z_c$가 바로 카메라 좌표계에서의 깊이다. 이 과정을 비선형 투영 함수 $\pi(\cdot)$로 추려 쓰면 다음과 같이 정리된다.

$$\mathbf{p} = \pi(\mathbf{P}_c) = \begin{bmatrix} f_x \frac{X_c}{Z_c} + c_x \\ f_y \frac{Y_c}{Z_c} + c_y \end{bmatrix}$$

실제 상태 추정에서는 이 비선형 함수를 그대로 다루기 어렵기 때문에, 현재 추정점 근방에서 선형화하기 위한 자코비안(Jacobian)이 반드시 필요하다.

$$\frac{\partial \pi}{\partial \mathbf{P}_c} = \begin{bmatrix} \frac{f_x}{Z_c} & 0 & -\frac{f_x X_c}{Z_c^2} \\ 0 & \frac{f_y}{Z_c} & -\frac{f_y Y_c}{Z_c^2} \end{bmatrix}$$

EKF 기반 VIO에서는 이 $2 \times 3$ 자코비안을 관측 행렬 구성에 직접 쓰고, 비선형 최적화 기반 SLAM에서는 잔차의 기울기를 구할 때 사용한다. 다만 이 깔끔한 핀홀 식은 렌즈가 빛을 전혀 굴절 왜곡하지 않는 이상적인 경우에만 성립한다.

### 2.1.2 렌즈 왜곡 모델 (Lens Distortion Model)

현실의 카메라 렌즈에서는 빛을 모으는 곡률과 조립 오차 탓에 핀홀 모델의 직선 투영에서 벗어난 왜곡(distortion)이 생긴다. 이 왜곡을 보정하지 않고 그대로 두면 재투영 오차(reprojection error)가 수 픽셀에서 수십 픽셀까지 벌어지므로, 정밀한 센서 융합에서는 렌즈 왜곡을 관측 모델에 반드시 포함해야 한다.

#### Radial-Tangential 왜곡 (Brown-Conrady 모델)

가장 흔히 쓰이는 Brown-Conrady 모델은 OpenCV의 기본 왜곡 모델이자, 핀홀 가정이 유효한 일반 화각 SLAM·VIO의 표준이다(시야각이 매우 넓은 렌즈에는 뒤에서 볼 fisheye나 omnidirectional 모델이 더 알맞다).

왜곡은 픽셀 좌표가 아니라 깊이 $Z_c$로 나눈 정규화 이미지 좌표 $\mathbf{p}_n = [x_n, y_n]^\top = [X_c/Z_c, \, Y_c/Z_c]^\top$에서 계산한다. 먼저 광축 중심까지의 거리를 제곱해 $r^2$을 구한다.

$$r^2 = x_n^2 + y_n^2$$

렌즈의 둥근 곡률 때문에 생기는 **방사 왜곡(Radial Distortion)**은 중심에서 멀어질수록 $r^2$의 다항식으로 커진다.

$$x_r = x_n (1 + k_1 r^2 + k_2 r^4 + k_3 r^6)$$
$$y_r = y_n (1 + k_1 r^2 + k_2 r^4 + k_3 r^6)$$

부호에 따라 상이 휘는 방향이 정반대로 갈린다. $k_1 < 0$이면 가장자리로 갈수록 배율이 줄어 중심을 지나지 않는 직선이 바깥으로 불룩하게 휘는 배럴 왜곡(barrel distortion)이 나타나고, 반대로 $k_1 > 0$이면 가장자리 배율이 커져 직선이 중심 쪽으로 오목하게 파고드는 핀쿠션 왜곡(pincushion distortion)이 생긴다.

한편 **접선 왜곡(Tangential Distortion)**은 조립 과정에서 렌즈와 이미지 센서가 완벽히 평행하지 않을 때(decentering) 발생한다. 크기는 방사 왜곡보다 훨씬 작지만 정밀도를 요구하는 시스템에서는 함께 모델링한다.

$$x_t = 2p_1 x_n y_n + p_2 (r^2 + 2x_n^2)$$
$$y_t = p_1 (r^2 + 2y_n^2) + 2p_2 x_n y_n$$

두 왜곡 성분을 정규화 좌표에 더한 뒤 내부 파라미터 $\mathbf{K}$를 적용하면 최종 픽셀 좌표 $(u, v)$가 나온다.

$$\begin{bmatrix} x_d \\ y_d \end{bmatrix} = \begin{bmatrix} x_r + x_t \\ y_r + y_t \end{bmatrix}$$

$$\begin{bmatrix} u \\ v \end{bmatrix} = \begin{bmatrix} f_x x_d + c_x \\ f_y y_d + c_y \end{bmatrix}$$

이 다섯 개의 왜곡 계수 $[k_1, k_2, p_1, p_2, k_3]$는 [Zhang (2000)](https://doi.org/10.1109/34.888718)의 체커보드 기법을 구현한 OpenCV `calibrateCamera()`로 추정한다(§3.1.3).

#### 어안 (Fisheye / Equidistant) 모델

시야각(FoV)이 180°에 육박하거나 이를 넘어서는 어안(fisheye) 렌즈에서는 사정이 달라진다. 광선 입사각이 90°에 가까워지면 $Z_c \to 0$이 되어 정규화 반경 $r$이 급격히 치솟고, $r^2$에 기반한 Brown-Conrady 다항식 근사가 가장자리에서 발산해 버린다.

이를 해결하기 위해 [Kannala & Brandt (2006)](https://doi.org/10.1109/TPAMI.2006.153)의 generic camera model에서는 정규화 평면에서의 거리 $r$ 대신, 3D 점의 광선이 광축과 이루는 입사각 $\theta$를 기준으로 왜곡을 정의한다.

$$\theta = \operatorname{atan2}\left(\sqrt{X_c^2 + Y_c^2},\; Z_c\right)$$

$Z_c < 0$인 뒤쪽 반구(시야각 180° 이상)까지 부호 모호성 없이 다루려면 인자가 둘인 `atan2`를 써야 하며, 이미지 평면 위의 방향 단위 벡터 역시 $X_c/\sqrt{X_c^2+Y_c^2}$ 꼴로 구한다(아래 파이썬 구현도 같은 방식을 따른다).

왜곡된 무차원 정규화 반경 $r_d$는 입사각 $\theta$의 홀수 차 다항식으로 둔다.

$$r_d = k_1 \theta + k_2 \theta^3 + k_3 \theta^5 + k_4 \theta^7 + k_5 \theta^9$$

왜곡이 전혀 없는 순수 등거리(equidistant) 투영이라면 $r_d = \theta$, 즉 $k_1 = 1$이고 나머지 계수는 모두 0이다. 이 무차원 반경에 방향 벡터를 곱하고 마지막으로 픽셀 단위 초점 거리 $f_x, f_y$와 주점 $c_x, c_y$를 적용하면 이미지 좌표를 얻는다.

$$\begin{bmatrix} u \\ v \end{bmatrix} = \begin{bmatrix} f_x \cdot r_d \cdot \frac{X_c}{\sqrt{X_c^2 + Y_c^2}} + c_x \\ f_y \cdot r_d \cdot \frac{Y_c}{\sqrt{X_c^2 + Y_c^2}} + c_y \end{bmatrix}$$

어안 렌즈는 한 프레임으로 드넓은 주변을 훑을 수 있어 특징점 유지에 유리하므로 [VINS-Mono](https://arxiv.org/abs/1708.03852)나 Basalt 같은 VIO 시스템이 이를 기본 지원하며, 캘리브레이션에는 [Scaramuzza et al. (2006)](https://rpg.ifi.uzh.ch/docs/IROS06_scaramuzza.pdf)의 OCamCalib이나 Kalibr가 널리 쓰인다.

두 투영 모델이 실제 코드에서 어떻게 구현되는지 비교해 보자. 아래에서 `project_pinhole`은 5-파라미터 Brown-Conrady 왜곡을 구현하고, `project_fisheye`는 OpenCV `fisheye` 모듈의 4-파라미터 컨벤션(주석 참고)을 따른다.

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

이처럼 왜곡까지 반영한 투영 함수 $\pi(\cdot)$가 준비되면, 이제 추정기가 실제로 줄이려는 목적 함수, 즉 관측과 예측 사이의 잔차를 정의할 수 있다.

### 2.1.3 재투영 오차 (Reprojection Error)

특징점 기반 VO·VIO와 번들 조정(Bundle Adjustment)은 카메라 관측을 비용 함수에 넣을 때 재투영 오차(reprojection error)를 표준 잔차로 쓴다. 이 오차는 현재 추정한 카메라 포즈 $\mathbf{T}_{cw} = [\mathbf{R}|\mathbf{t}]$로 3D 랜드마크 $\mathbf{P}_w$를 이미지 평면에 투영한 예측 좌표 $\hat{\mathbf{p}}$와, 영상에서 실제로 검출한 특징점 좌표 $\mathbf{p}_{\text{obs}}$ 사이의 픽셀 차이다.

$$\mathbf{e}_{\text{reproj}} = \mathbf{p}_{\text{obs}} - \pi(\mathbf{T}_{cw} \cdot \mathbf{P}_w)$$

여기서 $\pi(\cdot)$는 앞서 정의한 투영 함수로, 렌즈 왜곡까지 포함한다.

**번들 조정(Bundle Adjustment)**은 관측된 모든 카메라 포즈와 랜드마크 위치에 대해 이 재투영 오차의 가중 제곱합을 한꺼번에 최소화한다.

$$\min_{\{\mathbf{T}_i\}, \{\mathbf{P}_j\}} \sum_{i,j} \rho\left(\| \mathbf{p}_{ij} - \pi(\mathbf{T}_i \cdot \mathbf{P}_j) \|^2_{\mathbf{\Sigma}_{ij}}\right)$$

각 항의 역할은 다음과 같다.
- $\mathbf{p}_{ij}$: 카메라 $i$가 실제로 관측한 랜드마크 $j$의 2D 이미지 좌표다.
- $\mathbf{\Sigma}_{ij}$: 관측 노이즈 공분산으로, 등방성 모델에서는 보통 $\sigma^2 \mathbf{I}_2$($\sigma \approx 1$ 픽셀)로 둔다.
- $\rho(\cdot)$: Huber나 Cauchy 같은 로버스트 커널(robust kernel)로, 잘못 매칭된 아웃라이어가 전체 해를 망가뜨리지 않도록 큰 오차의 비중을 눌러 준다.

재투영 오차는 보통 $\sigma = 0.5 \sim 2$ 픽셀인 가우시안 분포로 모델링한다. 구체적인 $\sigma$ 값은 특징점 검출기의 정밀도에 좌우되며, 코너를 서브픽셀(sub-pixel) 수준까지 정교하게 찾으면 $\sigma \approx 0.5$ 픽셀까지 낮아진다.

지금까지의 투영 모델에서는 이미지 한 장 안의 모든 픽셀이 완전히 똑같은 시각에 찍혔다고 가정했다. 그러나 실제 하드웨어, 특히 스마트폰이나 보급형 카메라에서는 한 프레임 안에서도 위치마다 촬영 시각이 조금씩 어긋난다.

### 2.1.4 롤링 셔터 모델 (Rolling Shutter Model)

저가 카메라(스마트폰, 웹캠 등)가 주로 쓰는 CMOS 이미지 센서는 롤링 셔터(rolling shutter) 방식을 사용한다. 롤링 셔터에서는 모든 픽셀을 동시에 노출하는 글로벌 셔터(global shutter)와 달리, 맨 위 행(row)부터 아래 행까지 차례대로 읽어 내린다. 그 결과 이미지 위쪽과 아래쪽에는 물리적으로 서로 다른 순간에 찍힌 풍경이 담긴다.

첫 번째 행의 노출 시각을 $t_0$, 한 행을 읽는 데 걸리는 시간(row readout time)을 $t_r$이라 하면 $k$번째 행의 노출 시각은 다음과 같이 밀린다.

$$t_k = t_0 + k \cdot t_r$$

이미지 높이가 $H$일 때 첫 행과 마지막 행의 노출 시작 시각 차이는 $(H-1) \cdot t_r$에 이르며(각 행이 빛을 받는 노출 시간은 이와 별도로 정해진다), 프레임 전체를 읽어 내리는 데는 대략 $H \cdot t_r$, 즉 수 밀리초에서 수십 밀리초가 걸린다.

카메라가 정지해 있을 때는 문제가 없지만, 빠르게 움직이는 동안 롤링 셔터로 촬영하면 두 가지 왜곡이 나타난다.
- **기하 왜곡**: 수직으로 서 있는 기둥이 비스듬히 기울어지거나 움직이는 물체가 젤리처럼 휘어 보인다.
- **특징점 위치 오차**: 특징점이 어느 행에 놓였느냐에 따라 찍힌 순간의 카메라 포즈가 달라지므로, 모든 행을 한 순간에 찍었다고 보는 글로벌 셔터 투영 모델과 어긋난다.

이를 바로잡기 위해 롤링 셔터 투영 모델에서는 특징점 $i$를 투영할 때 고정된 프레임 포즈가 아니라 그 특징점이 위치한 행 $v_i$의 노출 시각에 해당하는 포즈 $\mathbf{T}(t_{v_i})$를 쓴다.

$$\mathbf{p}_i = \pi\left(\mathbf{T}(t_{v_i}) \cdot \mathbf{P}_i\right)$$

여기서 $\mathbf{T}(t_{v_i})$는 시각 $t_{v_i}$에서의 월드→카메라 변환이다. 프레임의 첫 행 포즈와 마지막 행 포즈를 알고 있다면, 그 짧은 readout 구간 동안 카메라의 속도(twist)가 일정하다고 가정해 리 군 $SE(3)$ 위에서 다음처럼 보간할 수 있다($H>1$).

$$\mathbf{T}(t_{v_i}) = \mathbf{T}(t_0) \cdot \text{Exp}\left(\frac{v_i}{H-1} \cdot \text{Log}(\mathbf{T}(t_0)^{-1} \mathbf{T}(t_0 + (H-1) \cdot t_r))\right)$$

식에서 $\text{Exp}$와 $\text{Log}$는 $SE(3)$의 지수·로그 맵이다. 실무에서 롤링 셔터 보정은 시스템마다 선택적으로 켜고 끈다. 예를 들어 [VINS-Mono](https://arxiv.org/abs/1708.03852)는 readout time 설정을 지원하지만 ORB-SLAM3는 글로벌 셔터를 기본으로 가정한다. 스마트폰이나 드론처럼 저가 센서를 달고 빠르게 기동할 때는 이 보정의 효과가 두드러진다.

아래 파이썬 함수는 `SE(3)` 지수 맵 대신 카메라→월드 변환의 회전을 SLERP로, 이동을 선형으로 각각 보간해 롤링 셔터 투영을 근사 구현한 예제다.

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

이처럼 카메라에서는 투영 기하와 렌즈 왜곡, 그리고 센서의 행별 시간차까지 고려해 관측 모델을 세운다. 흥미롭게도 3차원 거리를 직접 재는 LiDAR에서도 광선 방향과 거리 오차 모델은 물론, 스캔하는 동안 센서가 움직이면서 생기는 시간차 왜곡을 똑같이 다뤄야 한다.

---

## 2.2 LiDAR 관측 모델

LiDAR는 레이저 펄스를 발사한 뒤 반사광이 되돌아오는 비행시간(Time-of-Flight)이나 위상차를 측정해 3차원 거리를 얻는 능동 센서다. 포인트 클라우드를 직교 좌표 $(x, y, z)$ 묶음으로 다루기 쉽지만, 실제 센서의 물리적 오차는 구면 좌표계의 거리와 각도 축을 따라 발생하고, 센서가 회전하며 스캔하는 동안에는 시간차 왜곡까지 겹친다.

### 2.2.1 Range-Bearing 모델

LiDAR가 원시 단계에서 재는 값은 레이저 빔마다 얻는 거리(range)와 조향 방향(bearing)이다. 3D LiDAR의 측정점 하나는 구면 좌표 $(r, \alpha, \omega)$로 표현된다.

- $r$: 거리 (range)
- $\alpha$: 수평 각도 (azimuth)
- $\omega$: 수직 각도 (elevation)

LiDAR 좌표계의 실제 3D 점 $\mathbf{P}_L = [x, y, z]^\top$을 센서가 관측할 때, 각 성분에는 센서 고유의 관측 노이즈가 더해진다.

$$\begin{aligned}
r &= \sqrt{x^2 + y^2 + z^2} + n_r \\
\alpha &= \arctan2(y, x) + n_\alpha \\
\omega &= \arcsin\left(\frac{z}{\sqrt{x^2 + y^2 + z^2}}\right) + n_\omega
\end{aligned}$$

식에서 $n_r, n_\alpha, n_\omega$는 각각 거리, 수평각, 수직각의 관측 노이즈를 뜻한다.

**노이즈 특성**
- **거리 노이즈** $n_r$: 데이터시트 기준으로 기계식 [Velodyne VLP-16](https://www.mapix.com/wp-content/uploads/2018/07/63-9229_Rev-H_Puck-_Datasheet_Web.pdf)의 거리 정확도는 최대 $\pm 3\,\text{cm}$(typical, 대부분 채널의 벽면 시험 기준)이고, [Livox Avia](https://www.livoxtech.com/avia/specs)의 거리 정밀도는 반사율 80% 표적 20 m에서 $2\,\text{cm}$(1σ)다. 표적이 멀어질수록 레이저 빔이 퍼지고(beam divergence) 되돌아오는 에너지가 줄어 거리 노이즈가 커진다.
- **각도 노이즈** $n_\alpha, n_\omega$: [Livox Avia](https://www.livoxtech.com/avia/specs) 데이터시트의 각도 정밀도는 $0.05°$ 미만(1σ)이다. 얼핏 아주 작아 보이지만, 각도 오차는 거리에 비례해 횡방향 위치 오차로 증폭된다. 예컨대 50m 거리에서 각도가 $0.1°$만 틀려도 점의 위치가 옆으로 약 $8.7\,\text{cm}$ 어긋난다.

정합 알고리즘에서 쓸 직교 좌표계 점 $\mathbf{P}_L$은 측정된 구면 좌표를 역변환해 구한다.

$$\mathbf{P}_L = \begin{bmatrix} r \cos\omega \cos\alpha \\ r \cos\omega \sin\alpha \\ r \sin\omega \end{bmatrix}$$

**빔 확산(Beam Divergence).** 레이저 빔은 두께가 없는 이상적인 선분이 아니라, 가느다란 원뿔 모양으로 거리가 멀어질수록 퍼져 나간다. 데이터시트 기준으로 퍼지는 각도는 VLP-16이 수평 $0.18°$·수직 $0.07°$([Rev-F 사양서](https://www.hypertech.co.il/wp-content/uploads/2015/12/63-9229_Rev-F_Puck__Spec_Sheet_Web.pdf)), Livox Avia가 수평 $0.03°$·수직 $0.28°$다. 거리가 멀어져 빔이 닿는 면적(footprint)이 넓어지면 센서는 그 면적 안의 평균적인 거리를 기록한다. 특히 전경 물체의 모서리에서 빔의 일부는 물체에 맞고 나머지는 뒤쪽 배경에 맞으면 두 거리 사이의 허공에 가짜 점이 찍히는데, 이를 혼합 픽셀(mixed pixel) 효과라 부른다.

포인트 클라우드에는 이처럼 공간적 빔 확산에서 오는 오차뿐 아니라, 스캔을 끝마치는 동안 센서 자체가 움직여서 생기는 시간축 왜곡도 함께 들어간다.

### 2.2.2 모션 왜곡 (Motion Distortion)

기계식 스피닝(spinning) LiDAR는 내부 레이저 모듈이 360° 회전하면서 순차적으로 점들을 찍는다. 예컨대 Velodyne VLP-16은 기본 설정(10Hz)에서 한 바퀴를 도는 데 약 100ms가 걸린다. 그 짧은 시간 동안 차량이나 드론이 이동하거나 회전하면, 한 스캔 안에 담긴 점들이 서로 다른 순간과 위치에서 관측된다. 이를 모션 왜곡(motion distortion)이라 부르며, 모든 점을 하나의 기준 시각 좌표계로 되돌리는 보정 절차를 ego-motion compensation이라 한다.

문제의 구조는 앞서 본 카메라의 롤링 셔터와 꼭 닮았다. 스캔 안의 $i$번째 점이 시각 $t_i$에 측정되었다면, 그 순간의 센서 포즈 $\mathbf{T}(t_i)$를 이용해 점을 월드 좌표로 보낸 뒤 스캔 기준 시각 $t_0$의 LiDAR 좌표계로 다시 옮긴다.

$$\mathbf{P}_L^{(t_0)} = \mathbf{T}(t_0)^{-1} \cdot \mathbf{T}(t_i) \cdot \mathbf{P}_L^{(t_i)}$$

여기서 $\mathbf{T}(t_i)$는 시각 $t_i$에서의 LiDAR→월드 변환 행렬이다.

**보정 방법**

스캔 사이의 궤적 $\mathbf{T}(t)$를 어떻게 복원하느냐에 따라 크게 세 가지 방식이 쓰인다.

1. **IMU로 보간**: 수백 Hz로 들어오는 IMU 측정값을 적분해 스캔 구간 안의 미세한 포즈 변화를 보간한다. LIO-SAM, FAST-LIO2 같은 LIO 시스템이 택하는 표준 방식이다.
2. **직전 프레임 속도로 외삽**: IMU가 없거나 느슨할 때는 직전 프레임에서 추정한 속도가 현재 스캔 동안에도 일정하게 유지된다고 가정(constant velocity)해 각 시각의 포즈를 외삽한다.
3. **연속 시간(continuous-time) 방법**: 궤적을 B-스플라인 같은 연속 함수로 매개변수화해 각 점이 찍힌 시각의 포즈를 직접 계산한다. [CT-ICP (Dellenbach et al., 2022)](https://arxiv.org/abs/2109.12979)처럼 스캔 시작 포즈와 끝 포즈를 함께 최적화하면서 그 사이를 연속 보간하는 방식도 여기에 속한다.

아래 `undistort_scan` 함수는 전형적인 보정 코드다. 스캔 시작과 끝 포즈가 주어지면 회전은 SLERP로, 이동은 선형 보간으로 각 타임스탬프의 포즈를 계산해 모든 점을 시작 시각 좌표계로 모은다.

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

모션 왜곡 보정을 거치고 나면 한 스캔의 점들이 동일한 기준 시각 $t_{\text{start}}$에 찍힌 강체 스냅샷처럼 정렬되므로, 이후 포인트 클라우드 정합 단계에서 일관된 기하 제약을 세울 수 있다. 다만 센서가 점을 뿌리는 스캔 패턴 자체가 기계식인지 비반복식인지에 따라 특징 추출 전략은 크게 달라진다.

### 2.2.3 Spinning vs Solid-State LiDAR가 융합에 미치는 영향

**기계식 스피닝 LiDAR**(Velodyne, Ouster, Hesai 등)는 수평 360°를 회전하며 훑기 때문에 한 스캔이 둥근 고리 모양의 정렬된 링(ring) 구조로 들어온다. 고전적인 LOAM과 그 후속 계열은 바로 이 수평 스캔 라인 구조를 전제로 설계되어, 같은 링 안의 이웃 점들과 곡률을 비교해 edge·planar feature를 뽑고 사방에서 들어온 관측으로 6-DoF 포즈를 안정적으로 맞춘다.

반면 **시야가 좁은 Livox LiDAR**(Mid-40/70, Avia, HAP 등)에는 같은 궤적을 되풀이하지 않는 비반복(non-repetitive) 스캔 모드가 있다(참고로 Mid-40은 내부에서 프리즘을 회전시켜 빔 방향을 꺾으므로 기계적 회전부가 전혀 없는 것은 아니며, Avia에서는 반복 스캔 모드도 선택할 수 있다). 비반복 모드에서는 좁은 시야(Mid-70 기준 지름 약 70.4°의 원형) 안을 꽃잎이나 리사주 곡선 모양으로 지나가며 시간이 누적될수록 빈틈이 촘촘히 메워진다.

두 방식의 구조적 차이에 따라 특징 추출과 추정기 설계가 어떻게 달라지는지 아래 표로 비교할 수 있다.

| 특성 | 스피닝 LiDAR | 제한된 FoV의 비반복 스캔 모드 |
|------|-------------|-------------------|
| FoV | 360° 수평 | 제한적 (40°~120°) |
| 스캔 패턴 | 반복적 (수평 라인) | 비반복적 (로제트, 리사주 등) |
| 단일 프레임 밀도 | 균일 | 불균일 (시간에 따라 증가) |
| Feature 추출 | 스캔 라인 기반 가능 | 스캔 라인 구조 없음 |
| 적합한 알고리즘 | LOAM, LeGO-LOAM | FAST-LIO/LIO2 (점 단위 처리) |

수평 스캔 라인이라는 전제가 사라지면 LOAM 방식의 edge·planar feature 추출을 그대로 적용하기 어렵다. 이 한계를 우회하기 위해 FAST-LIO와 [FAST-LIO2](https://arxiv.org/abs/2107.06829)는 한 스캔의 점별 평면 잔차를 모아 iterated EKF로 상태를 갱신한다. 특히 FAST-LIO2는 별도의 특징 추출 없이 원시 점을 ikd-Tree 지도에 직접 정합하므로 스캔 라인 구조에 매이지 않고 비반복 스캔 LiDAR에도 자연스럽게 동작한다. 더 나아가 점이 들어올 때마다 상태를 갱신하는 Point-LIO는 §7.3.4에서 다루며, [FAST-LIVO2 (Zheng et al., 2025)](https://arxiv.org/abs/2408.14035)는 같은 iterated EKF 틀 안에 카메라까지 통합해 LiDAR 점과 영상 픽셀을 특징 추출 없이 direct 방식으로 순차 갱신한다.

카메라와 LiDAR는 모두 외부 장면을 관측하지만, 프레임이나 스캔 사이에 수십 밀리초의 공백이 생기고 센서 자체가 움직이는 동안 왜곡이 끼어든다. 바로 그 빈틈을 IMU가 100Hz–1kHz로 촘촘히 채워 주고 모션 왜곡 보정의 뼈대 역할까지 한다.

---

## 2.3 IMU 모델

IMU(Inertial Measurement Unit)는 3축 자이로스코프(gyroscope)와 3축 가속도계(accelerometer)가 한 몸체에 들어 있는 관성 센서다. 100Hz–1kHz의 빠른 속도로 각속도와 가속도를 뱉어 내므로 카메라나 LiDAR 프레임 사이의 공백을 메우고, 초기 자세 정렬과 단안 스케일 추정에도 핵심 역할을 한다. 하지만 IMU 출력은 적분을 거쳐 자세·속도·위치로 바뀌기 때문에, 원시 신호에 섞인 작은 바이어스와 노이즈라도 정확히 모델링해 추적하지 않으면 추정 궤적이 순식간에 발산하고 만다.

### 2.3.1 자이로스코프 오차 모델

자이로스코프는 센서 바디 프레임 기준의 3축 각속도 $\boldsymbol{\omega}$를 잰다. 실제 출력값 $\tilde{\boldsymbol{\omega}}$에는 천천히 변하는 오프셋과 무작위 잡음이 참값에 함께 실린다.

$$\tilde{\boldsymbol{\omega}} = \boldsymbol{\omega} + \mathbf{b}_g + \mathbf{n}_g$$

- $\boldsymbol{\omega}$: IMU 바디 프레임에서 본 실제 각속도다.
- $\mathbf{b}_g$: 바이어스(bias)로, 0점 근처에서 거의 일정해 보이지만 시간과 온도에 따라 서서히 떠도는 오프셋이다.
- $\mathbf{n}_g$: 백색 가우시안 노이즈(Additive White Gaussian Noise, AWGN) 형태의 측정 노이즈다.

**바이어스의 동역학.** 바이어스가 시간에 따라 천천히 변하는 성질은 연속 시간 랜덤 워크(random walk)로 모델링한다.

$$\dot{\mathbf{b}}_g = \mathbf{n}_{bg}$$

여기서 $\mathbf{n}_{bg} \sim \mathcal{N}(\mathbf{0}, \sigma_{bg}^2 \mathbf{I})$는 바이어스를 조금씩 흔드는 구동 노이즈다. 실제 필터나 최적화기에서 주기 $\Delta t$로 이산화해 전파할 때는 아래처럼 쓴다.

$$\mathbf{b}_{g,k+1} = \mathbf{b}_{g,k} + \sigma_{bg} \sqrt{\Delta t} \cdot \mathbf{w}_k, \quad \mathbf{w}_k \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$$

자이로스코프의 노이즈 파라미터를 세팅할 때는 데이터시트 수치만 그대로 적기보다 실제 장착 장치와 운용 조건에 맞춰 다음 세 가지를 확인해야 한다.

- **백색 노이즈 밀도**: 장치 데이터시트와 정지 데이터 Allan deviation 그래프에서 기울기가 $-1/2$인 구간을 대조해 읽는다. 이때 연속 시간 노이즈 밀도의 단위가 코드에서 이산 공분산으로 환산하는 공식에 정확히 맞는지 꼭 검산해야 한다.
- **In-run bias stability**: 센서를 충분히 길게 정지시켜 모은 데이터의 Allan deviation에서 기울기가 거의 0으로 평평해지는 구간을 확인한다. 이 값은 주변 온도나 장착 응력, 기체 진동, 측정 시간에 따라 달라지므로 가능하면 실제 플랫폼에 장착한 상태에서 측정한 값을 참고한다.
- **바이어스 랜덤 워크 구동 노이즈**: 긴 평균 시간 $\tau$ 영역 중 기울기가 양수로 올라가는 구간에서 추정한다. 실무에서 측정 노이즈 $\sigma_g$와 바이어스 구동 노이즈 $\sigma_{bg}$를 혼동하기 쉬우므로, 연속 시간 밀도를 이산 공분산으로 어떻게 변환했는지 문서화해 두는 편이 안전하다.

### 2.3.2 가속도계 오차 모델

가속도계는 3축 비력(specific force) $\mathbf{a}$를 측정한다. 비력이란 물체의 실제 선가속도에서 중력 가속도를 뺀 값, 즉 센서가 중력을 버티며 받는 지지력을 뜻한다. 따라서 위쪽이 $+z$인 좌표계에서 책상 위에 가만히 놓여 있으면 실제 가속도 $\mathbf{a}_w = \mathbf{0}$이고 중력 $\mathbf{g}_w = [0, 0, -g]^\top$이므로, 가속도계는 $[0, 0, +g]^\top$을 출력한다. 이를 식으로 쓰면 다음과 같다.

$$\tilde{\mathbf{a}} = \mathbf{R}_{bw}(\mathbf{a}_w - \mathbf{g}_w) + \mathbf{b}_a + \mathbf{n}_a$$

- $\mathbf{a}_w$: 월드 프레임에서 본 실제 가속도다.
- $\mathbf{g}_w = [0, 0, -g]^\top$: 중력 벡터($g \approx 9.81\,\text{m/s}^2$)다.
- $\mathbf{R}_{bw}$: 월드 좌표를 바디 좌표로 돌리는 회전 행렬이다.
- $\mathbf{b}_a$: 가속도계 바이어스다.
- $\mathbf{n}_a \sim \mathcal{N}(\mathbf{0}, \sigma_a^2 \mathbf{I})$: 백색 측정 노이즈다. 연속 시간 노이즈 밀도 $\sigma_a$의 단위가 $\text{m/s}^2/\sqrt{\text{Hz}}$이므로, 샘플링 주기 $\Delta t$에서의 이산 노이즈 표준편차는 $\sigma_a/\sqrt{\Delta t}$가 된다. 반대로 바이어스 랜덤 워크의 이산 변화량에는 $\sqrt{\Delta t}$를 곱한다.

**중력과 관측 가능성.** 가속도계가 항상 중력 성분을 함께 재는 덕분에, 정지 상태나 등속 직진 구간에서 중력 방향을 읽어 롤(roll)과 피치(pitch)를 바로 잡아낼 수 있다. 반면 요(yaw)는 중력 축을 중심으로 도는 회전이라 중력 벡터의 방향이 전혀 변하지 않으므로 IMU만으로는 관측할 수 없다(unobservable). 그래서 VIO나 LIO는 출발 시점의 요 각도를 임의로 0이라 두고 상대 회전만 추적하며, 전역 방위가 꼭 필요할 때는 자력계나 사전 지도 정합처럼 방향 기준을 주는 외부 관측을 곁들인다.

**바이어스 동역학.** 가속도계 바이어스 역시 자이로스코프와 똑같이 랜덤 워크로 모델링한다.

$$\dot{\mathbf{b}}_a = \mathbf{n}_{ba}, \quad \mathbf{n}_{ba} \sim \mathcal{N}(\mathbf{0}, \sigma_{ba}^2 \mathbf{I})$$

가속도계 파라미터를 캘리브레이션할 때도 자이로스코프와 같은 흐름을 밟되 단위 환산과 중력 성분에 특히 주의한다.

- **백색 노이즈 밀도**: 데이터시트 값과 Allan deviation에서 기울기가 $-1/2$인 구간을 함께 살피고, $\text{m/s}^2/\sqrt{\text{Hz}}$, $\text{mg}/\sqrt{\text{Hz}}$, VRW 사이의 단위 환산 근거를 명확히 남긴다.
- **In-run bias stability**: 정지·온도 시험에서 Allan 곡선이 가장 평평한 구간과 반복 실험 간의 편차를 살핀다. 이때 센서가 기울어져 중력이 다른 축으로 비스듬히 실리는 투영 오차나 장착 자세 변화가 바이어스로 잘못 섞여 들지 않도록 주의한다.
- **바이어스 랜덤 워크 구동 노이즈**: 장시간 정지 데이터 중 기울기가 양수인 구간에서 $\sigma_{ba}$를 구한 뒤, 추정기가 동작하는 샘플 주기에 맞춰 이산 공분산으로 변환한다.

그렇다면 실제 손에 쥔 IMU에서 백색 노이즈와 바이어스 안정성, 랜덤 워크 수치를 어떻게 가려낼까? 정지 데이터를 시간 구간별로 쪼개어 분산을 계산하는 Allan Variance가 바로 그 표준 도구다.

### 2.3.3 Allan Variance

Allan Variance는 본래 원자시계의 주파수 안정도를 재기 위해 고안되었으나, 오늘날에는 IMU 노이즈 성분을 분리해 내는 표준 분석법으로 자리 잡았다. 센서를 진동 없는 평평한 곳에 가만히 두고 몇 시간 동안 데이터를 수집한 뒤, 데이터를 묶는 평균 시간(cluster time) $\tau$를 짧게도 길게도 바꿔 가며 이웃한 두 구간 평균의 차이 분산을 계산한다.

$$\sigma^2(\tau) = \frac{1}{2} \langle (\bar{y}_{k+1} - \bar{y}_k)^2 \rangle$$

여기서 $\bar{y}_k$는 길이가 $\tau$인 $k$번째 구간의 평균 측정값이다. 제곱근인 Allan Deviation $\sigma(\tau)$를 양대수(log-log) 그래프로 그려 보면, 시간 규모 $\tau$에 따라 지배적인 노이즈 성분이 서로 다른 기울기의 직선으로 나타난다.

| 기울기 | 노이즈 종류 | 물리적 의미 |
|--------|-----------|-----------|
| $-1/2$ | 각도/속도 랜덤 워크 (ARW/VRW) | 백색 노이즈 $\mathbf{n}_g, \mathbf{n}_a$ |
| $0$ | 바이어스 불안정성 (Bias Instability) | 플리커 노이즈. 이 구간의 최솟값에서 bias instability를 읽는다 |
| $+1/2$ | 레이트 랜덤 워크 (RRW) | 바이어스 랜덤 워크 $\mathbf{n}_{bg}, \mathbf{n}_{ba}$ |

**데이터시트 읽는 법.** IMU 데이터시트와 Allan Deviation 곡선에서 융합 필터에 필요한 네 가지 수치를 추출하는 순서는 다음과 같다.

1. **각도 랜덤 워크(Angular Random Walk, ARW)**: 단위는 보통 $°/\sqrt{\text{hr}}$ 또는 $\text{rad/s}/\sqrt{\text{Hz}}$로 적힌다. Allan Deviation 그래프에서 기울기가 $-1/2$인 직선을 $\tau = 1\,\text{s}$까지 연장해 읽은 값이 곧 자이로 백색 노이즈 밀도 $\sigma_g$다.
2. **속도 랜덤 워크(Velocity Random Walk, VRW)**: 단위는 $\text{m/s}/\sqrt{\text{hr}}$ 또는 $\text{m/s}^2/\sqrt{\text{Hz}}$를 쓰며, 같은 방식으로 읽은 값이 가속도계 백색 노이즈 밀도 $\sigma_a$다.
3. **바이어스 안정성(In-run Bias Stability)**: 데이터시트에는 대개 Allan Deviation 곡선이 가장 낮게 내려앉는 최솟값이 적혀 있다. IEEE Std 952 정의상 bias instability 계수 $B$는 평탄 구간에서 $\sigma(\tau) \approx 0.664B$ 관계를 가지므로, 곡선 최솟값을 0.664로 나누면 $B$를 얻는다(§3.4.4의 Allan 코드가 돌려주는 `bias_instability`가 이 값에 해당한다). 다만 이는 정지 상태에서 출력을 시간 평균했을 때 얼마나 잔잔한지를 보여 줄 뿐, 실제 융합 시스템이 동적 주행 중에 바이어스를 얼마나 정확히 추정할 수 있는지는 별개의 문제다.
4. **레이트 랜덤 워크(Rate Random Walk)**: 장시간 구간($+1/2$ 기울기)에서 읽는 값으로, 바이어스가 시간에 따라 떠도는 정도인 $\sigma_{bg}, \sigma_{ba}$에 대응한다.

아래 파이썬 코드에서는 정지 상태로 수집한 단일 축 IMU 데이터에서 겹치지 않는 구간 평균으로 Allan Deviation 곡선을 구하고(`compute_allan_variance`), 이 곡선에서 백색 노이즈와 바이어스 불안정성 지표를 뽑아낸다(`extract_imu_params`).

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

이처럼 IMU 측정에 어떤 바이어스와 노이즈가 섞이는지 수식으로 잡았으니, 이제 실제 측정값 $\tilde{\boldsymbol{\omega}}, \tilde{\mathbf{a}}$를 적분해 자세·속도·위치라는 항법 상태로 전파하는 운동 방정식을 세울 차례다.

### 2.3.4 Strapdown Navigation Equation

IMU 측정으로부터 자세, 속도, 위치를 적분해 나가는 수식을 스트랩다운 관성 항법 방정식(Strapdown Navigation Equation)이라 부른다. "스트랩다운(strapdown)"이라는 이름은 센서를 기계식 짐벌 위에 띄워 수평을 유지하던 옛 방식과 달리, 센서를 기체에 단단히 고정(strapped down)한 채 소프트웨어 좌표 변환으로 자세를 추적한다는 뜻에서 왔다.

일반적인 VIO/LIO에서는 지구 자전 효과를 무시하고 월드 프레임을 관성 프레임으로 근사한다. 이렇게 근사하면 월드 프레임 상태 $[\mathbf{R}, \mathbf{v}, \mathbf{p}]$의 연속 시간 동역학은 세 개의 미분 방정식으로 표현된다(지구에 고정된 NED/ENU 항법 프레임을 엄밀히 쓸 때는 지구 자전율과 전달률(transport rate) 항이 자세·속도 식에 추가된다).

$$\dot{\mathbf{R}} = \mathbf{R} \cdot [\tilde{\boldsymbol{\omega}} - \mathbf{b}_g - \mathbf{n}_g]_\times$$

$$\dot{\mathbf{v}} = \mathbf{R} (\tilde{\mathbf{a}} - \mathbf{b}_a - \mathbf{n}_a) + \mathbf{g}$$

$$\dot{\mathbf{p}} = \mathbf{v}$$

- $\mathbf{R} \in SO(3)$: 바디 프레임의 벡터를 월드 프레임으로 돌리는 회전 행렬이다.
- $\mathbf{v} \in \mathbb{R}^3$: 월드 프레임 기준 속도다.
- $\mathbf{p} \in \mathbb{R}^3$: 월드 프레임 기준 위치다.
- $[\cdot]_\times$: 3차원 벡터의 외적을 행렬 곱으로 바꾸는 반대칭(skew-symmetric) 행렬이다.

$$[\boldsymbol{\omega}]_\times = \begin{bmatrix} 0 & -\omega_z & \omega_y \\ \omega_z & 0 & -\omega_x \\ -\omega_y & \omega_x & 0 \end{bmatrix}$$

**이산 시간 적분.** 실제 임베디드 시스템에서는 센서가 주기 $\Delta t$마다 샘플을 내보내므로 연속 방정식을 이산화해야 한다. 가장 단순한 1차 오일러 전파에서는 시각 $t_k$에서 $t_{k+1} = t_k + \Delta t$까지 바이어스를 뺀 각속도와 가속도가 일정하다고 보고 상태를 갱신한다.

$$\mathbf{R}_{k+1} = \mathbf{R}_k \cdot \text{Exp}\left((\tilde{\boldsymbol{\omega}}_k - \mathbf{b}_{g,k}) \Delta t\right)$$

$$\mathbf{v}_{k+1} = \mathbf{v}_k + \mathbf{g} \Delta t + \mathbf{R}_k (\tilde{\mathbf{a}}_k - \mathbf{b}_{a,k}) \Delta t$$

$$\mathbf{p}_{k+1} = \mathbf{p}_k + \mathbf{v}_k \Delta t + \frac{1}{2} \mathbf{g} \Delta t^2 + \frac{1}{2} \mathbf{R}_k (\tilde{\mathbf{a}}_k - \mathbf{b}_{a,k}) \Delta t^2$$

여기서 $\text{Exp}(\boldsymbol{\phi})$는 회전 벡터 $\boldsymbol{\phi}$를 $SO(3)$ 회전 행렬로 매핑하는 지수 맵으로, Rodrigues 공식을 써서 닫힌 형태로 계산한다.

$$\text{Exp}(\boldsymbol{\phi}) = \mathbf{I} + \frac{\sin\|\boldsymbol{\phi}\|}{\|\boldsymbol{\phi}\|} [\boldsymbol{\phi}]_\times + \frac{1 - \cos\|\boldsymbol{\phi}\|}{\|\boldsymbol{\phi}\|^2} [\boldsymbol{\phi}]_\times^2$$

**중간점(midpoint) 적분.** 오일러 적분은 1차 정확도에 머물러 회전이 빠를 때 오차가 커진다. 한 구간 $\Delta t$의 시작과 끝에서 얻은 두 IMU 측정값의 평균을 쓰면 별다른 연산 부담 없이 2차 정확도로 올라간다.

$$\bar{\boldsymbol{\omega}} = \frac{1}{2}(\tilde{\boldsymbol{\omega}}_k + \tilde{\boldsymbol{\omega}}_{k+1}) - \mathbf{b}_{g,k}$$

$$\mathbf{R}_{k+1} = \mathbf{R}_k \cdot \text{Exp}(\bar{\boldsymbol{\omega}} \Delta t)$$

$$\bar{\mathbf{a}} = \frac{1}{2}\left(\mathbf{R}_k(\tilde{\mathbf{a}}_k - \mathbf{b}_{a,k}) + \mathbf{R}_{k+1}(\tilde{\mathbf{a}}_{k+1} - \mathbf{b}_{a,k})\right)$$

$$\mathbf{v}_{k+1} = \mathbf{v}_k + (\bar{\mathbf{a}} + \mathbf{g}) \Delta t$$

$$\mathbf{p}_{k+1} = \mathbf{p}_k + \mathbf{v}_k \Delta t + \frac{1}{2}(\bar{\mathbf{a}} + \mathbf{g}) \Delta t^2$$

VINS-Mono도 바로 이 중간점 적분을 채택하며, FAST-LIO2를 비롯한 다른 시스템의 전파 식과 이산화 방식은 구현마다 조금씩 다르므로 코드에서 직접 확인해야 한다. 설령 IMU 출력률이 높더라도 기체의 각가속도나 샘플링 지터가 크면 적분 오차가 남기 때문에, 중간점 적분·RK4·사전적분 가운데 어느 쪽을 택할지는 목표 운동과 허용 오차에 맞춰 시험으로 정한다.

아래 코드는 Rodrigues 공식을 이용한 $SO(3)$ 지수 맵과 중간점 적분 기반 스트랩다운 관성 항법을 처음부터 구현한 예제다.

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

**드리프트 크기.** 위 적분 식에서 바이어스를 보정하지 않은 채 그대로 흘려보내면 위치 오차가 얼마나 빠르게 불어나는지 계산해 보자. 가속도계 바이어스 $b_a = 0.01\,\text{m/s}^2$(약 $1\,\text{mg}$)를 예로 들면 다음과 같다.

- 1초 후 위치 오차: $\frac{1}{2} \times 0.01 \times 1^2 = 0.005\,\text{m}$ (5mm)
- 10초 후: $\frac{1}{2} \times 0.01 \times 100 = 0.5\,\text{m}$
- 60초 후: $\frac{1}{2} \times 0.01 \times 3600 = 18\,\text{m}$

보조 센서 없이 저가 MEMS IMU 단독 적분만으로는 짧은 임무에서조차 금세 허용 오차를 넘어선다. 물론 실제로 버티는 시간은 IMU 등급과 온도, 운동 패턴, 요구 정밀도에 따라 다르며, 고성능 관성 항법 장치(INS)라도 오차가 쌓이는 구조 자체는 피할 수 없다. 바로 이 때문에 VIO/LIO 시스템은 바이어스 $\mathbf{b}_g, \mathbf{b}_a$를 상태 벡터에 포함해 카메라나 LiDAR 관측이 들어올 때마다 함께 추정·갱신한다.

한편 최근에는 이 적분 과정에 학습을 결합하는 연구도 나오고 있다. [AirIO (Qiu et al., RA-L 2025)](https://arxiv.org/abs/2501.15659)는 IMU 데이터를 전역 좌표로 돌리지 않고 바디 프레임 표현 그대로 신경망에 입력했다. 이 논문에서는 입력 표현만 이렇게 바꿔도 드론 데이터셋 세 개(시뮬레이션 하나 포함)에서 평균 정확도가 66.7% 향상되었다고 보고했다.

하드웨어 선정 단계로 내려오면, 수식 모델뿐 아니라 실제 센서의 스펙 시트와 운용 조건까지 함께 따져 보아야 한다.

### 2.3.5 IMU 등급 분류

현장에서 흔히 부르는 항법급, 전술급, 산업용, 소비자용 같은 등급 구분은 제조사와 분야마다 경계가 제각각이다. MEMS 역시 제조 공정을 뜻하는 말일 뿐 그 안에서도 성능 스펙트럼이 넓게 펼쳐진다. 따라서 등급 명칭이나 가격표만 보고 고르기보다, 실제 후보 장치의 데이터시트 수치를 확인하고 목표 플랫폼 위에서 직접 시험해 선정해야 한다.

| 선택 기준 | 확인할 항목 | 검증 자료 | 설계 영향 |
|----------|------------|----------|----------|
| 백색 노이즈 | 자이로·가속도계 노이즈 밀도, ARW/VRW 단위 | 장치 데이터시트 + Allan 시험 | 단기 상태 전파 오차 |
| 바이어스 | In-run stability, 랜덤 워크, 온도 계수 | 정지·온도 반복 시험 + Allan 시험 | 무보조 드리프트와 외부 관측 주기 |
| 동적 범위 | 측정 범위, 대역폭, 포화·클리핑 | 목표 운동·진동 프로파일 시험 | 급격한 운동 중 관측 손실 |
| 시간 특성 | 출력률, 타임스탬프, clock/trigger 지원 | 지연·지터 측정 | 시간 보정과 융합 일관성 |
| 환경 내성 | 온도, 진동, 충격, 장착 응력 | 제조사 qualification + 목표 플랫폼 시험 | 반복성과 장기 안정성 |

이 기준들로 잰 노이즈 특성과 임무의 허용 오차를 맞대어 보면, 외부 보정 센서를 얼마나 자주 결합해야 하는지가 결정된다. 가격은 발주 시기와 수량에 따라 달라지므로 당시 견적에 캘리브레이션·동기화 비용까지 더해 비교하는 것이 현실적이다.

IMU가 짧은 순간의 움직임을 촘촘하게 이어 주는 대신 시간이 갈수록 전역 위치가 떠돈다면, 거꾸로 주기적이면서도 장기 드리프트 없이 전역 좌표를 쥐여 주는 대표 센서는 GNSS다.

---

## 2.4 GNSS 모델

GNSS(Global Navigation Satellite System)는 GPS, GLONASS, Galileo, BeiDou 같은 위성 항법 시스템을 통틀어 일컫는 말이다. VIO나 LIO는 출발점을 기준으로 상대 궤적을 쌓아 올리면서 서서히 틀어질 수밖에 없지만, GNSS는 지구 고정 좌표계에서 직접 절대 위치를 제공해 장기 드리프트를 잡아 준다. 다만 위성에서 쏜 전파가 대기와 도심 빌딩을 통과해 수신기에 닿기까지 여러 시계 오차와 지연, 반사 간섭이 끼어들기 때문에, 이를 융합에 쓰려면 각 오차 항의 출처를 모델로 분리해 다뤄야 한다.

### 2.4.1 의사거리 관측 모델 (Pseudorange)

GNSS 수신기는 각 위성이 보낸 코드 신호가 도착하기까지 걸린 시간에 광속 $c$를 곱해 거리 추정치인 의사거리(pseudorange) $\rho$를 얻는다. 여기서 "의사(pseudo)"라는 말이 붙은 이유는 순수한 기하학적 거리 외에도 수신기·위성의 시계 오차와 대기 지연이 함께 섞여 있기 때문이다.

$$\rho^s = r^s + c \cdot \delta t_r - c \cdot \delta t^s + I^s + T^s + \epsilon_\rho$$

- $r^s = \|\mathbf{p}_r - \mathbf{p}^s\|$: 수신기 위치 $\mathbf{p}_r$와 위성 $s$ 위치 $\mathbf{p}^s$ 사이의 실제 기하학적 거리다.
- $c \cdot \delta t_r$: 수신기 시계 오차(clock bias)로, 저가 수정 발진기를 쓰는 수신기 시각의 어긋남이다. 미지수이므로 위치와 함께 추정한다.
- $c \cdot \delta t^s$: 위성 시계 오차로, 위성이 방송하는 항법 메시지의 파라미터로 보정할 수 있다.
- $I^s$: 전리층(ionosphere) 지연이다. 고층 대기의 자유 전자가 신호를 늦추며, L1/L2 같은 이중 주파수 수신기를 쓰면 분산 성질을 이용해 1차 항을 제거할 수 있다.
- $T^s$: 대류권(troposphere) 지연으로, 하층 대기의 건조 공기와 수증기 때문에 발생하며 Saastamoinen 모델 같은 경험식으로 보정한다.
- $\epsilon_\rho$: 열잡음이나 다중경로처럼 모델링되지 않고 남는 노이즈다. [ESA Navipedia](https://gssc.esa.int/navipedia/index.php/Receiver_noise)에 따르면 수신기 내부 잡음은 코드 칩 길이의 약 1% 이하 수준으로, GPS C/A 코드에서 최대 3 m, P 코드에서 약 30 cm에 이른다.

**측위(positioning) 원리.** 위성 네 기 이상에서 의사거리를 동시에 재면 미지수인 수신기 3D 위치 $(x, y, z)$와 수신기 시계 오차 $\delta t_r$까지 네 개의 방정식을 얻어 푼다. 가장 기본인 표준 단독 측위(Single Point Positioning, SPP)는 이 비선형 방정식을 최소자승법이나 칼만 필터로 선형화해 해를 구한다.

**다중경로(Multipath).** 도심 협곡에서 위성 신호가 직접 들어오지 못하고 건물 벽에 반사돼 우회 경로로 들어오면 의사거리에 수 미터에서 수십 미터의 양(+)의 오차가 덧씌워진다. 이 오차는 주변 건물 배치와 차량 자세에 따라 시시각각 변해 모델로 깔끔히 잡기 어려우므로, GNSS를 센서 융합에 넣을 때는 아웃라이어 검출과 제거가 성패를 가른다.

### 2.4.2 반송파 위상 관측 모델 (Carrier Phase)

의사거리가 코드 신호의 정렬로 거리를 재어 오차가 미터급이라면, 신호를 실어 나르는 반송파 자체의 위상을 측정하는 반송파 위상(carrier phase) 관측은 정밀도가 한 차원 다르다. 한 파장보다 작은 소수 부분(fractional phase)의 측정 잡음이 밀리미터 단위까지 내려가기 때문이다. 예컨대 L1 반송파(1575.42 MHz)의 파장은 약 19 cm에 불과하다.

$$\Phi^s = r^s + c \cdot \delta t_r - c \cdot \delta t^s + \lambda N^s - I^s + T^s + \epsilon_\Phi$$

식에서 추가된 항의 의미는 다음과 같다.
- $\lambda$: 반송파의 파장이다.
- $N^s$: 정수 모호성(integer ambiguity)으로, 수신기가 위성을 처음 추적하기 시작한 순간 안테나와 위성 사이에 이미 걸쳐 있던 온전한 파장의 정수 개수를 뜻한다. RTK의 fixed 해와 PPP-AR은 이 $N^s$를 정수로 확정해 밀리미터·센티미터급 정밀도를 끌어내며, 기존 PPP는 이를 실수(float) 상태로 추정한다.
- $\epsilon_\Phi$: 반송파 위상 노이즈로, 반송파 파장의 약 1%인 수 mm 수준에 그친다([ESA Navipedia](https://gssc.esa.int/navipedia/index.php/Receiver_noise)).

식에서 눈여겨볼 부분은 전리층 항 $I^s$의 부호다. 전리층은 변조 코드(군속도)는 늦추는 반면 반송파 위상(위상속도)은 오히려 앞당기므로, 반송파 위상 방정식에서는 $-I^s$로 부호가 반대가 된다. 이중 주파수 조합으로 1차 전리층 오차를 없애더라도 고차 항(higher-order term)과 잔여 오차는 남는다. 실전에서 반송파 위상의 정밀도를 온전히 누리려면 정수 모호성을 풀 뿐 아니라 cycle slip(위상 추적이 잠깐 끊겨 정수가 튀는 현상), 다중경로, 대기 오차도 함께 해결해야 한다.

### 2.4.3 RTK (Real-Time Kinematic)

RTK는 정확한 좌표를 이미 아는 기준국(또는 네트워크 보정 서비스)의 관측과 이동국(로버)의 관측을 서로 차분(differencing)해 공통 오차를 소거하고, 반송파 위상의 정수 모호성을 실시간으로 푸는 기법이다. 기준국과 로버 사이 거리(baseline)가 가까울수록 두 지점이 겪는 전리층·대류권 지연이 거의 같아 차분 효과가 크며, 거리가 멀어질수록 대기 상관성이 떨어져 잔여 오차가 늘어난다. 정수 모호성이 고정(fix)되고 신호 및 보정 품질이 받쳐 주면 센티미터급 정확도를 얻는다.

**이중 차분(Double Difference).** 기준국 $b$와 이동국 $r$이 공통으로 보는 두 위성(대상 위성 $s$와 기준 위성 $r$)의 관측을 위성 간에 한 번, 수신기 간에 한 번 이중으로 빼면 식이 극적으로 단순해진다.

$$\nabla\Delta\Phi_{br}^{sr} = \nabla\Delta r_{br}^{sr} + \lambda \nabla\Delta N_{br}^{sr} + \epsilon$$

이중 차분을 취하면 미지수였던 수신기 시계 오차와 위성 시계 오차가 대수학적으로 완전히 지워지고, 기준국이 가까이 있다면 대기 지연 역시 거의 상쇄된다. 결국 남는 것은 위치에 종속된 기하학적 거리 항과 정수 모호성뿐이며, 이 정수 모호성은 LAMBDA 알고리즘 같은 기법으로 빠르게 탐색·고정한다.

### 2.4.4 PPP (Precise Point Positioning)

기준국 인프라에 기대기 어려운 환경에서는 PPP가 대안이 된다. PPP는 기준국 없이 수신기 한 대만으로 센티미터급 절대 위치를 구하는 방식이다. 외부 보정 서비스에서 제공하는 정밀 위성 궤도(precise orbit)와 정밀 시계(precise clock) 정보를 받아 위성 측 오차를 제거하고, 전리층·대류권 지연은 상태 벡터에 넣어 함께 추정한다.

두 고정밀 측위 기법의 운용 조건과 특성을 표로 견주어 보자.

| 특성 | RTK | PPP |
|------|-----|-----|
| 기준국 필요 | 예 (단일 기준국은 보통 10–20 km 이내) | 아니요 |
| 수렴 시간 | 수 초~수십 초 | 수십 분 (전통), 수 분 (PPP-AR) |
| 정밀도 (수렴 후) | $\sim 2\,\text{cm}$ | $\sim 5\,\text{cm}$ |
| 커버리지 | 기준국 근처 | 전 지구 |

**센서 융합에 GNSS 넣기.** GNSS가 주는 전역 좌표로 VIO/LIO의 장기 드리프트를 단단히 묶어 둘 수 있지만, 신호 단절, 다중경로, 좌표계 정렬, 안테나와 IMU 사이의 물리적 거리(lever arm), 시간 오프셋 문제를 해결하지 않으면 오히려 추정 궤적에 드리프트와 편향이 끼어든다. 예컨대 [LIO-SAM (Shan et al., 2020)](https://arxiv.org/abs/2007.00258)은 현재 필터가 추정한 위치 공분산이 GNSS가 출력한 위치 공분산보다 커졌을 때만 GNSS factor를 그래프에 추가한다. 실제로 GNSS 관측을 융합 파이프라인에 넣을 때는 세 가지를 반드시 챙겨야 한다.

1. **좌표계 변환**: GNSS는 지구 타원체 기준의 WGS84(위도, 경도, 타원체고) 좌표를 내보내는 반면, 로봇과 자율주행 플래너는 대개 출발점 기준의 ENU(East-North-Up)나 NED(North-East-Down) 지역 직교 좌표계를 쓰므로 변환이 필요하다.
2. **공분산**: 수신기가 출력하는 위치 공분산을 로컬 관측 좌표계로 회전시켜 추정기에 전달한다. DOP(Dilution of Precision)에 관측 오차 크기를 곱해 위치 불확실성을 어림하기도 한다. DOP는 위성 배치가 기하학적으로 얼마나 좋은지를 나타내는 무차원 지표다.
3. **아웃라이어**: 도심 다중경로가 심할 때는 측정값이 수십 미터씩 튀므로 로버스트 커널이나 마할라노비스 거리 기반 $\chi^2$ 검정으로 이상 관측을 걸러 내야 한다.

아래 코드에서 `pseudorange_model`은 ECEF 좌표계 위성 위치를 받아 예측 의사거리와 선형화 자코비안 $\mathbf{H}$를 구하는 기본 관측 모델을, `geodetic_to_enu`는 위도·경도·고도(WGS84)를 기준점 중심의 로컬 ENU 좌표로 옮기는 간이 변환을 보여 준다.

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

이처럼 위성 신호로 전역 위치의 닻을 내릴 수 있지만, 터널이나 짙은 안개·폭우 속으로 들어가면 상황은 달라진다. 빛 기반 센서인 카메라·LiDAR마저 악천후에 약해질 때, 전파를 쏘아 주변 물체의 거리와 상대 속도를 곧바로 낚아채는 센서가 레이더다.

---

## 2.5 Radar 모델

레이더(Radar)는 밀리미터파 대역의 전파(radio wave)를 쏘아 반사 신호를 받는 능동 센서다. 파장이 빛보다 훨씬 길어 비, 안개, 눈, 먼지 때문에 카메라와 LiDAR가 시야를 잃는 악천후에서도 비교적 안정적으로 주변을 감지한다. 게다가 반사파의 도플러 주파수 편이에서 물체의 시선 방향 상대 속도를 한 번에 재는데, 이는 카메라나 일반적인 비행시간(ToF) LiDAR로는 직접 얻을 수 없는 고유한 관측량이다.

### 2.5.1 FMCW Radar 원리

오늘날 자율주행 차량과 로보틱스에 널리 쓰이는 밀리미터파 레이더는 대부분 FMCW(Frequency Modulated Continuous Wave) 방식을 채택하고 있다([Han et al.의 4D radar 서베이](https://arxiv.org/abs/2306.04242) 2절에서는 현재 나와 있는 4D radar 대부분이 이 방식이라고 정리한다). FMCW 방식은 펄스의 왕복 시간만 재는 대신, 시간에 따라 주파수가 선형으로 변하는 전파(chirp)를 쏜 뒤 되돌아온 전파와 섞고 주파수 차이(beat frequency)를 읽어 거리와 속도를 동시에 구한다.

**Chirp 신호.** 송신기에서 쏘는 전파의 순간 주파수는 chirp 지속 시간 $T_c$ 동안 시작 주파수 $f_0$에서 대역폭 $B$만큼 일정한 기울기로 올라간다.

$$f_{\text{TX}}(t) = f_0 + \frac{B}{T_c} t$$

**Beat 주파수.** 거리 $R$에 놓인 표적에서 반사된 전파는 왕복 시간 $\tau = 2R/c$($c$: 광속)만큼 늦게 수신기에 닿는다. 바로 그 순간 송신기는 이미 더 높은 주파수를 내보내고 있으므로, 송신파와 수신파를 믹서(mixer)로 섞으면 두 주파수의 차이인 일정한 beat 주파수 $f_b$가 솟아오른다.

$$f_b = \frac{2BR}{cT_c}$$

따라서 beat 주파수만 측정하면 반사체까지의 거리로 곧바로 환산할 수 있다.

$$R = \frac{f_b \cdot c \cdot T_c}{2B}$$

**거리 분해능.** 서로 다른 두 물체를 별개의 피크로 갈라 볼 수 있는 최소 거리 간격은 오로지 주파수 대역폭 $B$에 반비례해 정해진다.

$$\Delta R = \frac{c}{2B}$$

예컨대 77GHz 차량용 레이더가 대역폭 $B = 4\,\text{GHz}$를 쓰면 거리 분해능은 $\Delta R \approx 3.75\,\text{cm}$가 된다.

**도플러 측정.** 단일 chirp로는 거리만 알 수 있지만, 짧은 간격으로 chirp를 연달아 쏘아 같은 물체에서 돌아온 신호의 위상 변화를 추적하면 시선 방향(radial) 상대 속도를 잴 수 있다.

$$v_r = \frac{\lambda \cdot f_d}{2}$$

여기서 $f_d$는 도플러 주파수, $\lambda$는 반송파의 파장이다. 77GHz 레이더는 $\lambda \approx 3.9\,\text{mm}$로 파장이 몹시 짧아, 미세한 속도 변화도 뚜렷한 위상 회전으로 잡힌다.

**Range-Doppler Map.** 연속된 여러 chirp 신호를 행렬로 쌓아 2D FFT를 돌리면 가로축이 거리, 세로축이 속도인 Range-Doppler Map을 얻는다. 이 맵에 솟은 피크 하나가 반사체 하나에 대응하며, 피크의 좌표에서 거리와 시선 방향 속도를 한꺼번에 읽어 낸다.

**방위각 측정.** 마지막으로 물체가 어느 방향에 있는지 알아내려면 여러 개의 수신 안테나를 배열(antenna array)로 늘어놓는다. 안테나 소자마다 전파가 도착하는 거리가 조금씩 달라 위상 차이가 생기므로, 이 위상차로 도래각(angle of arrival)을 추정한다. 안테나를 수평으로 배열하면 수평 각도(azimuth)를, 수직으로 배열하면 수직 각도(elevation)를 잰다.

### 2.5.2 4D Imaging Radar

기존 3D 자동차 레이더는 거리, 속도, 수평 각도까지만 재고 수직 방향은 거의 구분하지 못해 고가도로 표지판과 전방 정지 차량을 가려내지 못하는 한계가 있었다. 이를 넘어서기 위해 수직 각도까지 높은 분해능으로 재는 센서가 4D 이미징 레이더(4D Imaging Radar)다.

핵심은 MIMO(Multiple Input Multiple Output) 안테나 기술에 있다. 송신 안테나 12개와 수신 안테나 16개를 조합하면 물리적 안테나 수보다 훨씬 많은 $12 \times 16 = 192$개의 가상 안테나 채널(virtual antenna array)을 만들어 낼 수 있다. 수평·수직 각도 분해능은 이 가상 배열이 각 축으로 얼마나 넓게 펼쳐져 있는지에 따라 정해진다.

실제 주행 인지에서 4D 이미징 레이더와 LiDAR의 성질을 나란히 비교하면 다음과 같다.

| 특성 | 4D Imaging Radar | LiDAR |
|------|-----------------|-------|
| 악천후 동작 | 상대적으로 강함 (강수·radome 영향은 받음) | 저하됨 |
| 포인트 밀도 | LiDAR보다 성김 (저채널 LiDAR에 견줄 수준) | 높음 |
| 속도 측정 | 직접 측정 (도플러) | 불가 (두 프레임 차분 필요, FMCW LiDAR는 예외) |
| 수평 각도 분해능 | 약 0.5°–1.5° | 0.1°–0.4° (VLP-16) |
| 비용 | 저~중 | 중~고 |
| 정적 물체 감지 | 가능 (검출·클러터 제거 설정에 영향받음) | 우수 |

포인트 밀도 행의 출처는 [Han et al. 서베이](https://arxiv.org/abs/2306.04242)의 1절과 4절이다. 4D radar의 수평 각도 분해능은 같은 서베이 Table III에 실린 4D radar 데이터셋 장치(VoD, TJ4DRadSet, K-Radar, NTU4DRadLM 등) 대부분의 값이고, LiDAR 값은 [VLP-16 데이터시트](https://www.mapix.com/wp-content/uploads/2018/07/63-9229_Rev-H_Puck-_Datasheet_Web.pdf)의 회전 속도별 범위다.

**센서 융합에 레이더 넣기.** 포인트 클라우드가 LiDAR보다 성기긴 하지만, 레이더의 도플러 속도는 융합 파이프라인에 독보적인 이점을 보탠다.

1. **자기 속도 추정(Ego-velocity Estimation)**: 멈춰 있는 주변 반사체 여러 개에서 잰 시선 방향 속도로 연립방정식을 풀면 센서 자체의 3D 이동 속도 $\mathbf{v}_{\text{ego}}$를 바로 구할 수 있다. IMU 가속도를 적분할 때와 달리 속도를 직접 관측하므로 시간 적분에 따른 드리프트가 쌓이지 않는다.

$$v_r^{(i)} = -\mathbf{e}^{(i)\top} \mathbf{v}_{\text{ego}}$$

여기서 $\mathbf{e}^{(i)}$는 $i$번째 반사체를 향한 단위 방향 벡터다. 먼저 RANSAC 같은 이상치 제거 기법으로 움직이는 차량·보행자의 점을 걸러 낸 뒤, 남은 정적 반사체들로 최소자승법을 풀어 $\mathbf{v}_{\text{ego}}$를 구한다.

2. **움직이는 물체 찾기**: 추정한 자기 속도 $\mathbf{v}_{\text{ego}}$를 바탕으로 정적 환경일 때 예상되는 도플러 속도를 계산하면, 실제 측정한 도플러 속도가 이 예상치와 크게 다른 점은 움직이는 물체에서 반사된 것으로 쉽게 판별할 수 있다.
3. **악천후 대응**: 짙은 안개나 비로 카메라와 LiDAR 관측이 흐려지더라도 레이더는 외부 환경을 뚫고 관측을 계속 이어 간다(물론 실제로 얼마나 버티는지는 강수량, 주파수 대역, 안테나와 신호 처리 설정에 따라 달라진다).

아래 두 함수는 앞에서 유도한 FMCW 거리 환산 식(`fmcw_range_from_beat`)과, 정적 반사체들의 시선 방향 속도에서 최소자승법으로 자기 3D 속도를 구하는 선형 해법(`estimate_ego_velocity`)을 보여 준다.

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

실제 현장에서는 정적 반사체만 골라내는 RANSAC 루프 위에 이 `estimate_ego_velocity` 함수를 올려, 카메라가 눈이 멀거나 LiDAR가 텍스처 없는 긴 터널에 갇혀도 튼튼한 속도 제약을 얻는다.

다섯 가지 주력 센서(카메라, LiDAR, IMU, GNSS, Radar) 외에도, 실제 로봇과 차량에는 플랫폼의 움직임이나 운용 공간에 맞춰 여러 보조 센서를 함께 얹는다.

---

## 2.6 기타 센서

카메라·LiDAR·IMU·GNSS·Radar 같은 주력 센서만으로는 해결하기 까다로운 약점(예컨대 터널 속 속도 유지, 드론의 수직 고도 드리프트, 실내 절대 위치 기준)이 남는다. 실전 센서 융합 시스템에서는 플랫폼의 기구학과 운용 공간에 맞춰 보조 센서를 곁들여 그 빈틈을 메운다.

### 2.6.1 차륜 오도메트리 (Wheel Odometry)

차륜 인코더(wheel encoder)는 바퀴 축의 회전수를 세어 이동 거리와 회전량을 추정하는 고유수용(proprioceptive) 센서다. 지상 로봇이나 차량처럼 바퀴가 지면에 닿아 굴러가는 플랫폼에서는 가장 값싸고 안정적인 단기 속도·이동량 기준이 된다.

**관측 모델.** 좌우 두 바퀴를 독립 구동하는 차동 구동(differential drive) 로봇은 두 바퀴의 회전 각도 변화량 $\Delta \theta_L, \Delta \theta_R$로부터 전진 거리와 요 변화량을 곧장 계산한다.

$$\Delta s = \frac{r(\Delta \theta_L + \Delta \theta_R)}{2}, \quad \Delta \psi = \frac{r(\Delta \theta_R - \Delta \theta_L)}{d}$$

여기서 $r$은 바퀴 반경, $d$는 좌우 바퀴 사이 거리(트레드), $\Delta s$는 전진 거리, $\Delta \psi$는 요 각도 변화다.

**Ackermann 조향 모델**(자동차)에서는 앞바퀴 조향 각도 $\delta$와 휠베이스 $L$, 후륜 속도 $v$가 기구학 방정식을 이룬다.

$$\dot{x} = v \cos\psi, \quad \dot{y} = v \sin\psi, \quad \dot{\psi} = \frac{v \tan\delta}{L}$$

**슬립(Slip) 모델.** 이 기구학 모델들은 바퀴가 미끄러지지 않고 구른다는 가정을 깔고 있다. 하지만 젖은 노면이나 비포장길, 급가속·급감속 구간에서는 바퀴가 헛돌거나 밀리면서 슬립이 발생한다. 슬립 비율(slip ratio)은 바퀴 회전에서 추정한 속도 $v_{\text{wheel}}$과 실제 대지 속도 $v_{\text{actual}}$의 상대 차이로 정의한다.

$$s = \frac{v_{\text{wheel}} - v_{\text{actual}}}{v_{\text{actual}}}$$

슬립이 커지면 차륜 오도메트리를 곧이곧대로 믿었을 때 궤적이 크게 틀어진다. 그래서 융합 필터는 IMU나 시각 관측과 비교해 슬립 징후가 포착되면 차륜 관측의 공분산을 동적으로 키워(adaptive observation covariance) 다른 센서의 비중을 높인다.

**센서 융합에 넣기.** VIO나 LIO에 차륜 오도메트리를 속도 또는 상대 이동량 관측으로 결합한다. 특히 직선 주행에서 단기 정밀도가 높기 때문에, 터널이나 특징점이 메마른 직선 도로에서 IMU의 위치·속도 드리프트를 단단히 붙들어 준다. 실제로 [Liu et al. (IROS 2019)](https://doi.org/10.1109/IROS40897.2019.8967607)은 IMU와 차륜 인코더 측정값을 함께 사전적분(IMU-odometer pre-integration)해 VINS-Mono 방식의 sliding window 최적화에 통합하는 기법을 보였다.

### 2.6.2 기압계 (Barometer)

가속도계를 두 번 적분해 수직 고도를 유지하려 하면 바이어스 탓에 금세 위아래로 뜬다. 이때 주변 대기압을 재서 고도 변화를 잡아 주는 소형 센서가 기압계(barometer)다.

**관측 모델.** 국제 표준 대기(ISA) 모델에서는 측정 기압 $P$와 고도 $h$가 다음 지수 관계로 묶인다.

$$h = \frac{T_0}{L}\left(1 - \left(\frac{P}{P_0}\right)^{\frac{RL}{g_0}}\right)$$

각 상수의 값과 단위는 다음과 같다.
- $P$: 측정 기압 [Pa]
- $P_0 = 101325\,\text{Pa}$: 해수면 표준 기압
- $T_0 = 288.15\,\text{K}$: 해수면 표준 온도
- $L = 0.0065\,\text{K/m}$: 온도 감률(lapse rate)
- $R = 287.053\,\text{J/(kg}\cdot\text{K)}$: 공기의 기체 상수
- $g_0 = 9.80665\,\text{m/s}^2$: 표준 중력 가속도

저고도 비행이나 지상 주행처럼 고도 변화가 크지 않을 때는 이를 선형으로 근사해 간편하게 쓴다.

$$\Delta h \approx -\frac{\Delta P}{\rho g} \approx -\frac{\Delta P}{12.0}\,[\text{m}], \quad (\Delta P\text{는 Pa 단위})$$

즉 해수면 부근에서는 고도가 $1\,\text{m}$ 올라갈 때마다 기압이 약 $12.0\,\text{Pa}$씩 내려간다.

**노이즈 특성**
- 단기 정밀도: $\pm 0.1\text{–}0.5\,\text{m}$
- 장기 정밀도: 기상 변화에 따른 기압 변동 탓에 $\pm 1\text{–}10\,\text{m}$

**센서 융합에 넣기.** 기압계는 단기 분해능이 좋으므로 IMU의 수직 방향 드리프트를 누르는 데 쓰이며, 특히 드론이 제자리 비행(hovering)할 때 고도를 일정하게 붙드는 데 효과적이다. 반대로 날씨에 따라 기준 기압 자체가 서서히 변하므로, 절대 고도 기준으로 삼기보다는 짧은 구간의 상대 고도 변화를 재는 용도에 맞다. 실내에서는 냉난방 공조나 출입문 개폐만으로도 기압이 출렁인다는 점을 염두에 두어야 한다.

### 2.6.3 자력계 (Magnetometer)

앞서 보았듯 가속도계는 중력 벡터를 읽어 롤과 피치는 정하지만, 중력 축을 도는 요(yaw) 방향은 가려내지 못한다. 자력계(magnetometer)는 지구 자기장 벡터를 3축으로 측정해 빠진 방위 기준을 보태는 센서다. 롤·피치를 알고 있고, 하드·소프트 아이언 보정(hard/soft-iron calibration)을 마쳤으며, 주변 자기 교란이 작다면 자북(magnetic north) 기준 방위를 계산할 수 있다. 진북(true north) 기준이 필요하면 해당 지역의 자기 편차(magnetic declination)를 추가로 보정한다.

**관측 모델.** 센서가 뱉어 내는 자기장 측정값 $\tilde{\mathbf{m}}$에는 지구 자기장 외에 기체 자체의 자성체와 강자성 구조물이 만드는 왜곡이 함께 얹힌다.

$$\tilde{\mathbf{m}} = \mathbf{R}_{bw} \mathbf{m}_w + \mathbf{b}_{\text{hard}} + \mathbf{S}_{\text{soft}} \mathbf{R}_{bw} \mathbf{m}_w + \mathbf{n}_m$$

- $\mathbf{m}_w$: 월드 프레임에서 본 지구 자기장 벡터로, 지역에 따라 크기가 약 $25\text{–}65\,\mu\text{T}$다.
- $\mathbf{b}_{\text{hard}}$: 하드 아이언(hard-iron) 바이어스로, 기체에 붙은 영구 자석이나 자화된 금속이 센서 좌표계에 만드는 일정한 오프셋이다.
- $\mathbf{S}_{\text{soft}}$: 소프트 아이언(soft-iron) 왜곡으로, 주변 강자성체가 지구 자기장의 방향과 크기를 비틀어 놓는 효과를 나타내는 행렬 항이다.
- $\mathbf{n}_m$: 측정 노이즈다.

**요 각도 구하기.** 가속도계로 롤 $\phi$와 피치 $\theta$를 구하고, 하드 아이언 바이어스를 뺀 자기장 측정값 $(m_x, m_y, m_z)$를 수평면으로 투영하면 요 각도 $\psi$를 얻는다.

$$\psi = \arctan2(m_z \sin\phi - m_y \cos\phi, \, m_x \cos\theta + m_y \sin\theta \sin\phi + m_z \sin\theta \cos\phi)$$

이 식은 NED 월드 좌표계, x 전방·y 우측·z 하방 기체축, Z-Y-X 오일러각 규약을 전제하며, 결과 $\psi$는 자북에서 시계 방향으로 잰 방위각이다.

**한계**
- 실내나 차량 내부, 철골·콘크리트 건물 근처에서는 주변 금속 구조물이 지구 자기장을 크게 휘게 해 방위가 수십 도씩 틀어지곤 한다.
- 드론 모터나 배선에 흐르는 큰 전류가 만드는 자기장 역시 시간에 따라 측정값을 흔든다.
- 이 때문에 실전 센서 융합에서는 자력계를 주 센서로 앞세우기보다 보조 관측으로 두고, 자기장 크기의 변화를 감시해 교란이 없을 때만 반영하거나 가중치를 낮춰 조절한다.

### 2.6.4 UWB (Ultra-Wideband)

GNSS 신호가 전혀 닿지 않는 실내 창고나 공장, 지하 공간에서 절대 위치가 필요할 때 대안이 되는 무선 센서가 UWB(Ultra-Wideband)다. 대역폭이 500MHz 이상으로 매우 넓은 극초단 펄스를 주고받아 앵커와 태그 사이의 거리를 수십 센티미터 수준으로 잰다.

**관측 모델(TWR, Two-Way Ranging).** 양방향 거리 측정(TWR)에서는 요청 펄스를 보내고 응답 펄스가 돌아오기까지 걸린 전체 왕복 시간 $t_{\text{round}}$에서 상대 노드가 응답을 준비하는 데 쓴 처리 시간 $t_{\text{reply}}$를 빼서 거리를 구한다.

$$d = \frac{c \cdot (t_{\text{round}} - t_{\text{reply}})}{2} + n_d$$

- $t_{\text{round}}$: 요청 펄스를 송신한 뒤 응답 펄스를 수신할 때까지의 총 왕복 시간이다.
- $t_{\text{reply}}$: 응답 노드가 신호를 받고 되돌려 보내기까지 걸린 내부 처리 시간이다.
- $n_d$: 거리 측정 노이즈로, 가시선(LOS)이 트여 있을 때는 대개 $\sigma_d \approx 5\text{–}30\,\text{cm}$ 수준이다.

**NLOS(Non-Line-of-Sight) 문제.** 전파가 벽이나 철제 선반 같은 장애물을 통과하거나 돌아서 들어오면(NLOS) 전파 지연이 길어져 실제보다 먼 거리가 측정된다. 따라서 UWB 기반 측위에서는 채널 임펄스 응답이나 잔차 검정으로 NLOS 측정값을 탐지해 아웃라이어로 걸러 내거나 공분산을 크게 잡아야 한다.

**센서 융합에 넣기.** 작업 공간에 UWB 앵커(anchor)를 미리 설치해 두면, 각 앵커까지 잰 거리(range)로 앵커 좌표계(anchor frame)에서의 위치를 추정할 수 있다. 앵커 좌표를 사전 측량해 전역 좌표계(global frame)에 묶어 두어야 전역 절대 위치가 된다. UWB를 VIO와 엮어 쓰면 앵커 배치가 좋고 가시선이 확보된 구간에서 VIO의 장기 드리프트를 주기적으로 눌러 준다.

**관측 방정식(삼변측량).** 알려진 위치 $\mathbf{a}_i$에 놓인 앵커 $N_a$개까지의 거리를 재면 미지의 태그 위치 $\mathbf{p}$에 대한 비선형 거리 방정식을 세운다.

$$d_i = \|\mathbf{p} - \mathbf{a}_i\| + n_{d,i}, \quad i = 1, \ldots, N_a$$

2D 평면 위치를 구하려면 앵커가 셋 이상, 3D 공간 위치를 구하려면 한 평면에 몰려 있지 않은 앵커가 넷 이상 있어야 한다. 모든 앵커를 같은 높이에 달아 버리면 아래 코드의 선형화 행렬 $\mathbf{A}$에서 $z$축 성분이 사라져 높이를 관측할 수 없다.

아래 `uwb_trilateration` 함수는 각 앵커 거리 방정식을 제곱한 뒤 첫 번째 앵커 식과 차분해 미지수 $\|\mathbf{p}\|^2$ 항을 소거하고, 선형 시스템 $\mathbf{A}\mathbf{p} = \mathbf{b}$를 최소자승법으로 풀어 3D 위치를 추정한다.

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

실제 융합 필터에서는 이 대수적 삼변측량 해를 초기 위치로 잡은 뒤, 각 앵커의 원시 거리 잔차 $\| \mathbf{p} - \mathbf{a}_i \| - d_i$를 비선형 관측 모델 $h(\mathbf{x})$로 직접 넣어 NLOS 아웃라이어를 개별적으로 걸러 내며 최적화한다.

---

## 요약

지금까지 살펴본 열 가지 센서의 관측 모델, 지배적인 노이즈원, 현장 검증 기준을 한 표에 모아 정리했다.

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

각 센서의 물리 관측 모델이 아무리 정교해도, 센서 내부 파라미터가 부정확하거나 서로 다른 센서 사이의 상대 위치·자세와 시각 동기가 어긋나 있으면 추정 결과는 한쪽으로 치우친다. 센서 고유의 파라미터와 센서 간 기하·시간 관계를 정확히 재어 맞추는 캘리브레이션(Calibration)을 이어서 다룬다(→ 3장).
