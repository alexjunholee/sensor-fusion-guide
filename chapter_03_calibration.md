# Ch.3 — Calibration Deep Dive

Ch.2에서 각 센서의 관측 모델을 수학적으로 정의했다. 그런데 이 모델들을 실제 센서 데이터에 적용하려면 한 가지 전제가 필요하다 — 모델의 파라미터가 정확히 알려져 있어야 한다는 것이다. 카메라의 초점 거리, LiDAR와 IMU 사이의 상대 위치, 센서 간 시간 오프셋 — 이 값들을 정밀하게 결정하는 과정이 캘리브레이션이다.

> **핵심 메시지**: 센서 퓨전의 정확도는 캘리브레이션의 정확도를 넘을 수 없다. 이 챕터는 카메라 내부 파라미터부터 다중 센서 간 외부 파라미터, 시간 동기화까지 캘리브레이션의 모든 측면을 다룬다.

캘리브레이션(calibration)은 센서 퓨전 파이프라인에서 가장 먼저 해결해야 하는 문제다. 아무리 정교한 상태 추정 알고리즘을 사용하더라도, 센서의 내부 모델이 부정확하거나 센서 간 상대 위치/자세가 잘못되어 있으면 퓨전 결과는 발산한다. 특히 LiDAR-카메라 퓨전에서 외부 파라미터가 1도만 틀어져도, 50m 거리의 물체에서 약 87cm의 정합 오차가 발생한다. 이 챕터에서는 각 캘리브레이션 문제의 수학적 기초를 유도하고, 실전에서 바로 사용할 수 있는 코드와 도구를 제공한다.

---

## 3.1 Camera Intrinsic Calibration

카메라 내부 캘리브레이션(intrinsic calibration)은 3D 세계의 점이 2D 이미지 위 어디에 투영되는지를 결정하는 파라미터를 추정하는 과정이다. 이 파라미터에는 초점 거리(focal length), 주점(principal point), 그리고 렌즈 왜곡 계수(distortion coefficients)가 포함된다.

### 3.1.1 핀홀 카메라 모델 복습

3D 세계 좌표 $\mathbf{P}_w = [X, Y, Z]^\top$이 카메라 좌표계에서 $\mathbf{P}_c = [X_c, Y_c, Z_c]^\top$로 변환된 후, 이미지 평면에 투영되는 과정은 다음과 같다:

$$
s \begin{bmatrix} u \\ v \\ 1 \end{bmatrix}
= \mathbf{K} \begin{bmatrix} X_c \\ Y_c \\ Z_c \end{bmatrix}
= \begin{bmatrix} f_x & \gamma & c_x \\ 0 & f_y & c_y \\ 0 & 0 & 1 \end{bmatrix} \begin{bmatrix} X_c \\ Y_c \\ Z_c \end{bmatrix}
$$

여기서:
- $f_x, f_y$: 픽셀 단위의 초점 거리 (focal length). $f_x = f / p_x$이며 $f$는 물리적 초점 거리(mm), $p_x$는 픽셀 크기(mm/pixel).
- $(c_x, c_y)$: 주점. 이미지 센서의 중심과 광축(optical axis)의 교점.
- $\gamma$: 비대칭 계수(skew coefficient). 현대 카메라에서는 거의 0.
- $s = Z_c$: 스케일 팩터 (깊이값).

$\mathbf{K}$를 **카메라 내부 행렬(intrinsic matrix)** 또는 **캘리브레이션 행렬**이라 부른다. 이 행렬은 5개의 자유도를 가진다($f_x, f_y, c_x, c_y, \gamma$). 실전에서는 $\gamma = 0$으로 두어 4개로 줄이는 경우가 많다.

### 3.1.2 렌즈 왜곡 모델

실제 렌즈는 핀홀 모델의 이상적인 직선 투영을 따르지 않는다. 왜곡은 크게 방사 왜곡(radial distortion)과 접선 왜곡(tangential distortion)으로 나뉜다.

**정규화된 좌표**(normalized coordinates)를 먼저 정의한다:

$$
x = X_c / Z_c, \quad y = Y_c / Z_c, \quad r^2 = x^2 + y^2
$$

**방사 왜곡** (radial distortion):
$$
x_{\text{radial}} = x(1 + k_1 r^2 + k_2 r^4 + k_3 r^6)
$$
$$
y_{\text{radial}} = y(1 + k_1 r^2 + k_2 r^4 + k_3 r^6)
$$

배럴(barrel) 왜곡은 $k_1 < 0$, 핀쿠션(pincushion) 왜곡은 $k_1 > 0$일 때 나타난다. 대부분의 렌즈는 $k_1, k_2$만으로 충분히 모델링된다.

**접선 왜곡** (tangential distortion):
$$
x_{\text{tangential}} = 2p_1 xy + p_2(r^2 + 2x^2)
$$
$$
y_{\text{tangential}} = p_1(r^2 + 2y^2) + 2p_2 xy
$$

렌즈 요소가 이미지 센서 평면과 완벽히 평행하지 않을 때 발생한다.

**왜곡이 적용된 최종 좌표**:
$$
x_d = x_{\text{radial}} + x_{\text{tangential}}, \quad y_d = y_{\text{radial}} + y_{\text{tangential}}
$$

$$
u = f_x \cdot x_d + c_x, \quad v = f_y \cdot y_d + c_y
$$

왜곡 파라미터 벡터는 $\mathbf{d} = [k_1, k_2, p_1, p_2, k_3]$로 표현하며, OpenCV의 `calibrateCamera()`가 이 순서를 따른다.

### 3.1.3 Zhang's Method: 호모그래피 기반 캘리브레이션

[Zhang (2000)](https://ieeexplore.ieee.org/document/888718)이 제안한 방법은 평면 패턴(체커보드)을 다양한 자세로 촬영한 이미지들로부터 카메라 파라미터를 추정한다. 3D 캘리브레이션 장비 없이 프린터로 출력한 체커보드만 있으면 되므로, 현재 가장 널리 사용되는 캘리브레이션 방법이다.

**핵심 아이디어**: 패턴이 $Z = 0$ 평면에 놓이면, 3D-2D 투영이 호모그래피로 단순화된다.

#### Step 1: 호모그래피 추출

체커보드가 $Z = 0$ 평면에 있으므로, 세계 좌표 $\mathbf{M} = [X, Y, 0]^\top$과 이미지 좌표 $\mathbf{m} = [u, v]^\top$의 관계를 동차(homogeneous) 좌표로 쓰면:

$$
s \tilde{\mathbf{m}} = \mathbf{K} [\mathbf{r}_1 \quad \mathbf{r}_2 \quad \mathbf{r}_3 \quad \mathbf{t}] \begin{bmatrix} X \\ Y \\ 0 \\ 1 \end{bmatrix}
= \mathbf{K} [\mathbf{r}_1 \quad \mathbf{r}_2 \quad \mathbf{t}] \begin{bmatrix} X \\ Y \\ 1 \end{bmatrix}
$$

$Z = 0$이므로 $\mathbf{r}_3$ 열이 사라진다. 이로부터:

$$
s \tilde{\mathbf{m}} = \mathbf{H} \tilde{\mathbf{M}}, \quad \mathbf{H} = \mathbf{K} [\mathbf{r}_1 \quad \mathbf{r}_2 \quad \mathbf{t}]
$$

$\mathbf{H}$는 $3 \times 3$ 호모그래피 행렬이다. 각 이미지에서 최소 4개의 대응점으로 $\mathbf{H}$를 DLT(Direct Linear Transform)로 추정할 수 있다. 실전에서는 체커보드 코너가 수십 개이므로 과결정(over-determined) 시스템을 SVD로 풀고, RANSAC으로 아웃라이어를 제거한다.

**DLT를 이용한 호모그래피 추정**: 대응점 $(\mathbf{M}_j, \mathbf{m}_j)$에서 $\tilde{\mathbf{m}}_j \times \mathbf{H} \tilde{\mathbf{M}}_j = \mathbf{0}$으로부터 다음 선형 시스템을 얻는다:

$$
\begin{bmatrix}
\tilde{\mathbf{M}}_j^\top & \mathbf{0}^\top & -u_j \tilde{\mathbf{M}}_j^\top \\
\mathbf{0}^\top & \tilde{\mathbf{M}}_j^\top & -v_j \tilde{\mathbf{M}}_j^\top
\end{bmatrix}
\mathbf{h} = \mathbf{0}
$$

여기서 $\mathbf{h}$는 $\mathbf{H}$의 9개 원소를 벡터로 편 것이다. $n$개 대응점에서 $2n \times 9$ 행렬 $\mathbf{A}$를 구성하고, $\|\mathbf{A}\mathbf{h}\|$를 최소화하는 $\mathbf{h}$를 $\mathbf{A}$의 SVD에서 가장 작은 singular value에 대응하는 right singular vector로 구한다.

#### Step 2: 내부 파라미터의 제약 조건 도출

$\mathbf{H} = [\mathbf{h}_1 \quad \mathbf{h}_2 \quad \mathbf{h}_3]$로 열을 나누면:

$$
[\mathbf{h}_1 \quad \mathbf{h}_2 \quad \mathbf{h}_3] = \lambda \mathbf{K} [\mathbf{r}_1 \quad \mathbf{r}_2 \quad \mathbf{t}]
$$

여기서 $\lambda$는 임의의 스케일 팩터이다. 회전 행렬의 직교성에서 두 가지 제약 조건을 얻는다:

1. **직교 조건**: $\mathbf{r}_1^\top \mathbf{r}_2 = 0$
   $$\mathbf{h}_1^\top \mathbf{K}^{-\top} \mathbf{K}^{-1} \mathbf{h}_2 = 0$$

2. **등장 조건**: $\|\mathbf{r}_1\| = \|\mathbf{r}_2\|$
   $$\mathbf{h}_1^\top \mathbf{K}^{-\top} \mathbf{K}^{-1} \mathbf{h}_1 = \mathbf{h}_2^\top \mathbf{K}^{-\top} \mathbf{K}^{-1} \mathbf{h}_2$$

$\mathbf{B} = \mathbf{K}^{-\top} \mathbf{K}^{-1}$로 정의하자. $\mathbf{B}$는 대칭 양의 정치(positive definite) 행렬이므로 6개의 독립 원소를 가진다:

$$
\mathbf{B} = \begin{bmatrix} B_{11} & B_{12} & B_{13} \\ B_{12} & B_{22} & B_{23} \\ B_{13} & B_{23} & B_{33} \end{bmatrix}
$$

이를 벡터로 쓰면 $\mathbf{b} = [B_{11}, B_{12}, B_{22}, B_{13}, B_{23}, B_{33}]^\top$이다.

#### Step 3: 선형 시스템 구성

$\mathbf{h}_i^\top \mathbf{B} \mathbf{h}_j$를 $\mathbf{b}$에 대한 내적으로 표현할 수 있다:

$$
\mathbf{h}_i^\top \mathbf{B} \mathbf{h}_j = \mathbf{v}_{ij}^\top \mathbf{b}
$$

여기서:
$$
\mathbf{v}_{ij} = \begin{bmatrix} h_{1i}h_{1j} \\ h_{1i}h_{2j} + h_{2i}h_{1j} \\ h_{2i}h_{2j} \\ h_{3i}h_{1j} + h_{1i}h_{3j} \\ h_{3i}h_{2j} + h_{2i}h_{3j} \\ h_{3i}h_{3j} \end{bmatrix}
$$

각 이미지에서 두 가지 제약으로부터:

$$
\begin{bmatrix} \mathbf{v}_{12}^\top \\ (\mathbf{v}_{11} - \mathbf{v}_{22})^\top \end{bmatrix} \mathbf{b} = \mathbf{0}
$$

$n$장의 이미지가 있으면 $2n \times 6$ 시스템을 얻는다.

**최소 이미지 수**: $\gamma = 0$으로 두면 ($B_{12} = 0$ 제약 추가) 5개 미지수에 대해 최소 3장. 일반적인 5-파라미터 모델은 최소 3장이 필요하다. 실전에서는 15~25장을 촬영한다.

#### Step 4: K 복원

$\mathbf{b}$를 풀면 $\mathbf{B} = \mathbf{K}^{-\top} \mathbf{K}^{-1}$을 복원할 수 있고, Cholesky 분해 $\mathbf{B} = \mathbf{L}\mathbf{L}^\top$에서 $\mathbf{K}^{-1} = \mathbf{L}^\top$로 $\mathbf{K}$를 구한다. 구체적으로:

$$
v_0 = (B_{12}B_{13} - B_{11}B_{23}) / (B_{11}B_{22} - B_{12}^2)
$$
$$
\lambda = B_{33} - [B_{13}^2 + v_0(B_{12}B_{13} - B_{11}B_{23})] / B_{11}
$$
$$
f_x = \sqrt{\lambda / B_{11}}
$$
$$
f_y = \sqrt{\lambda B_{11} / (B_{11}B_{22} - B_{12}^2)}
$$
$$
\gamma = -B_{12} f_x^2 f_y / \lambda
$$
$$
c_x = \gamma v_0 / f_y - B_{13} f_x^2 / \lambda
$$
$$
c_y = v_0
$$

#### Step 5: 외부 파라미터 계산

$\mathbf{K}$를 알면 각 이미지 $i$에 대해:

$$
\mathbf{r}_1 = \lambda \mathbf{K}^{-1} \mathbf{h}_1, \quad \mathbf{r}_2 = \lambda \mathbf{K}^{-1} \mathbf{h}_2, \quad \mathbf{t} = \lambda \mathbf{K}^{-1} \mathbf{h}_3
$$

여기서 $\lambda = 1 / \|\mathbf{K}^{-1} \mathbf{h}_1\|$이다. $\mathbf{r}_3 = \mathbf{r}_1 \times \mathbf{r}_2$로 구한다. 추정된 $[\mathbf{r}_1, \mathbf{r}_2, \mathbf{r}_3]$는 일반적으로 완벽한 회전 행렬이 아니므로 (노이즈 때문에), SVD를 이용하여 가장 가까운 회전 행렬로 투영한다: $\mathbf{R} = \mathbf{U}\mathbf{V}^\top$ (단, $\det(\mathbf{R}) = 1$).

#### Step 6: 비선형 최적화 (MLE)

선형 해는 좋은 초기값을 제공하지만, 왜곡 파라미터를 고려하지 않으며 노이즈에 최적이지 않다. 최종 단계에서 Levenberg-Marquardt (LM) 알고리즘으로 **재투영 오차(reprojection error)**를 최소화한다:

$$
\min_{\mathbf{K}, \mathbf{d}, \{\mathbf{R}_i, \mathbf{t}_i\}} \sum_{i=1}^{n}\sum_{j=1}^{m} \|\mathbf{m}_{ij} - \hat{\mathbf{m}}(\mathbf{K}, \mathbf{d}, \mathbf{R}_i, \mathbf{t}_i, \mathbf{M}_j)\|^2
$$

여기서 $n$은 이미지 수, $m$은 각 이미지의 코너 수, $\hat{\mathbf{m}}(\cdot)$은 왜곡을 포함한 전체 투영 함수이다.

이 최적화의 자코비안은 각 파라미터에 대한 투영 함수의 편미분으로 구성된다. OpenCV의 `calibrateCamera()`는 이 전체 파이프라인(DLT → 선형 K 추정 → LM 최적화)을 구현한다.

### 3.1.4 OpenCV를 이용한 캘리브레이션 코드

```python
import numpy as np
import cv2
import glob

# 체커보드 설정
CHECKERBOARD = (9, 6)  # 내부 코너 수 (columns, rows)
SQUARE_SIZE = 0.025    # 25mm 정사각형

# 3D 월드 좌표 생성 (Z=0 평면)
objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
objp *= SQUARE_SIZE

obj_points = []  # 3D 점 (모든 이미지에서 동일)
img_points = []  # 2D 점 (이미지마다 다름)

images = sorted(glob.glob("calibration_images/*.jpg"))

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 코너 검출
    ret, corners = cv2.findChessboardCorners(
        gray, CHECKERBOARD,
        cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE
    )
    
    if ret:
        # 서브픽셀 정밀도로 코너 위치 정제
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        
        obj_points.append(objp)
        img_points.append(corners_refined)

# 캘리브레이션 수행 (Zhang's method)
ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(
    obj_points, img_points, gray.shape[::-1], None, None
)

print(f"RMS reprojection error: {ret:.4f} pixels")
print(f"Camera matrix K:\n{K}")
print(f"Distortion coefficients: {dist.ravel()}")

# 재투영 오차 분석
errors = []
for i in range(len(obj_points)):
    img_points_proj, _ = cv2.projectPoints(obj_points[i], rvecs[i], tvecs[i], K, dist)
    error = cv2.norm(img_points[i], img_points_proj, cv2.NORM_L2) / len(img_points_proj)
    errors.append(error)
    
print(f"Per-image mean error: {np.mean(errors):.4f} pixels")
print(f"Max error image: {np.argmax(errors)} ({max(errors):.4f} px)")
```

### 3.1.5 실전 팁: 좋은 캘리브레이션을 위한 조건

캘리브레이션의 품질은 데이터 수집 과정에서 결정된다. 다음은 실전에서 반복적으로 확인된 핵심 조건들이다.

**포즈 다양성 (Pose Diversity)**: 가장 중요한 요소. 체커보드를 다양한 각도와 위치에서 촬영해야 한다. 구체적으로:
- 체커보드를 이미지의 상하좌우, 중앙 모든 영역에 배치 (주점 추정에 필수)
- 보드를 45도 이상 기울여서 촬영 (초점 거리 추정의 정밀도 향상)
- 보드를 카메라에 가깝게/멀리 배치하여 다양한 스케일 확보
- 최소 15~25장, 이상적으로는 50장 이상

**코너 정확도**: `cv2.cornerSubPix()`로 서브픽셀 정밀도를 반드시 확보한다. 코너 검출 실패 이미지는 과감히 제외한다.

**조명 조건**: 균일한 조명이 이상적이지만, 약간의 그림자는 코너 검출에 영향을 주지 않는다. 반사(glare)는 코너 검출을 방해하므로 무광(matte) 종이에 인쇄한 체커보드를 사용한다.

**재투영 오차 기준**: 
- $< 0.3$ pixels: 우수
- $0.3 - 0.5$ pixels: 양호
- $0.5 - 1.0$ pixels: 재수집 고려
- $> 1.0$ pixels: 문제 있음 (포즈 다양성 부족, 코너 검출 오류 등)

**이상치 이미지 감지**: per-image 재투영 오차를 계산하여 평균보다 2~3배 큰 이미지는 제거 후 재캘리브레이션한다. 위 코드의 `errors` 배열로 확인할 수 있다.

**경고 — 정사각형 크기의 정확성**: `SQUARE_SIZE`는 실제 인쇄된 체커보드의 정사각형 크기와 정확히 일치해야 한다. 프린터 스케일링으로 인해 실제 크기가 지정 크기와 다를 수 있으므로, 반드시 자로 측정한다. 이 값이 틀리면 `tvecs` (이동 벡터)가 잘못되지만, `K`와 왜곡 계수에는 영향을 주지 않는다 (왜곡 모델은 정규화 좌표에서 정의되므로).

### 3.1.6 Fisheye / Omnidirectional 캘리브레이션

FOV(Field of View)가 180도 이상인 어안 렌즈(fisheye lens)에서는 표준 radial-tangential 왜곡 모델이 작동하지 않는다. 왜곡이 너무 극심하여 다항식 근사가 수렴하지 않기 때문이다.

**등거리 투영 모델 (Equidistant Projection Model)**:

[Kannala & Brandt (2006)](https://ieeexplore.ieee.org/document/1642666)가 제안한 일반 카메라 모델은 입사각(incidence angle) $\theta$와 이미지 거리 $r$의 관계를 다항식으로 표현한다:

$$
r(\theta) = k_1 \theta + k_2 \theta^3 + k_3 \theta^5 + k_4 \theta^7 + k_5 \theta^9
$$

여기서 $\theta = \arctan\left(\sqrt{x^2 + y^2}\right)$는 입사각, $x, y$는 정규화 좌표이다. 이상적인 등거리 투영은 $r = f\theta$이며, 다항식의 추가 항이 실제 렌즈의 편차를 보정한다.

OpenCV는 `cv2.fisheye` 모듈에서 이 모델의 변형을 구현한다:

$$
\theta_d = \theta(1 + k_1\theta^2 + k_2\theta^4 + k_3\theta^6 + k_4\theta^8)
$$

```python
# Fisheye 캘리브레이션 (OpenCV fisheye 모듈)
import cv2
import numpy as np

calibration_flags = (
    cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC
    + cv2.fisheye.CALIB_CHECK_COND
    + cv2.fisheye.CALIB_FIX_SKEW
)

K_fisheye = np.zeros((3, 3))
D_fisheye = np.zeros((4, 1))  # k1, k2, k3, k4

ret, K_fisheye, D_fisheye, rvecs, tvecs = cv2.fisheye.calibrate(
    obj_points,
    img_points,
    gray.shape[::-1],
    K_fisheye,
    D_fisheye,
    None, None,
    calibration_flags,
    (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6)
)

print(f"Fisheye RMS error: {ret:.4f}")
print(f"K:\n{K_fisheye}")
print(f"D: {D_fisheye.ravel()}")
```

**Scaramuzza OCamCalib (Omnidirectional Camera Calibration)**:

[Scaramuzza et al. (2006)](https://sites.google.com/site/scarabotix/ocamcalib-omnidirectional-camera-calibration-toolbox-for-matlab)의 OCamCalib은 catadioptric 시스템(거울 + 렌즈)과 초광각 어안 렌즈를 위한 통합 캘리브레이션 도구이다. 투영 함수를 다항식으로 직접 모델링하여 센서 타입에 독립적이다:

$$
\begin{bmatrix} u \\ v \end{bmatrix} = \begin{bmatrix} x_c \\ y_c \end{bmatrix} + \mathbf{A} \cdot \rho(\theta) \begin{bmatrix} \cos(\phi) \\ \sin(\phi) \end{bmatrix}
$$

여기서 $\mathbf{A}$는 affine 변환 행렬 (stretch와 non-square pixel 보정), $\rho(\theta)$는 입사각에 따른 이미지 반경이다.

### 3.1.7 Kalibr를 이용한 카메라 캘리브레이션

[Kalibr](https://github.com/ethz-asl/kalibr)는 ETH Zurich에서 개발한 캘리브레이션 도구로, 카메라-IMU 캘리브레이션(3.4절)에서 사실상 표준이다. 카메라 intrinsic 캘리브레이션에서도 OpenCV보다 더 정교한 결과를 얻을 수 있는 경우가 많다.

**Kalibr의 카메라 모델 지원**:

| 모델 | 파라미터 수 | 적합한 렌즈 |
|------|-----------|-----------|
| `pinhole-radtan` | 4 + 4 | 일반 렌즈 (OpenCV와 동일) |
| `pinhole-equi` | 4 + 4 | 어안 렌즈 |
| `omni-radtan` | 5 + 4 | 초광각/catadioptric |
| `ds` (Double Sphere) | 6 | 광각 렌즈 (최신 모델) |
| `eucm` (Extended UCM) | 6 | 광각 렌즈 |

**AprilGrid vs Checkerboard**: Kalibr는 AprilGrid 타겟을 권장한다. AprilGrid는 각 태그에 고유 ID가 인코딩되어 있어 부분 가려짐(partial occlusion)에서도 코너 검출이 가능하고, 코너 순서를 자동으로 식별한다.

**Kalibr 실행 예시** (카메라 intrinsic만):
```bash
# AprilGrid 타겟 파일 생성
kalibr_create_target_pdf --type apriltag \
    --nx 6 --ny 6 --tsize 0.024 --tspace 0.3

# 카메라 캘리브레이션
kalibr_calibrate_cameras \
    --target april_6x6_24x24mm.yaml \
    --bag camera_calibration.bag \
    --models pinhole-radtan \
    --topics /cam0/image_raw
```

Kalibr의 내부 동작은 다음과 같다:
1. AprilGrid의 코너를 각 프레임에서 검출
2. 코너 대응으로부터 호모그래피 추정 (Zhang's method의 Step 1과 유사)
3. 카메라 모델의 전체 파라미터에 대해 batch 최적화 수행
4. 최적화 결과와 잔차(residuals) 분포를 시각화

Kalibr가 OpenCV 기본 캘리브레이션보다 유리한 점은:
- **B-spline 궤적 표현**: 연속 시간 모델로 모션 블러 효과를 자연스럽게 처리
- **다양한 카메라 모델**: DS, EUCM 등 최신 모델 지원
- **멀티카메라**: 여러 카메라의 상대 포즈를 동시에 추정 가능
- **IMU 연동**: 3.4절에서 다룰 camera-IMU 캘리브레이션과 자연스럽게 연결

---

## 3.2 Camera-Camera (Stereo) Extrinsic

스테레오 카메라 시스템에서 두 카메라 간의 상대 포즈(extrinsic)를 캘리브레이션하는 것은 깊이 추정의 전제 조건이다.

### 3.2.1 스테레오 캘리브레이션

두 카메라 $C_L$(왼쪽)과 $C_R$(오른쪽)이 있을 때, 같은 3D 점 $\mathbf{P}$에 대해:

$$
\mathbf{P}_{C_R} = \mathbf{R} \cdot \mathbf{P}_{C_L} + \mathbf{t}
$$

여기서 $(\mathbf{R}, \mathbf{t})$는 왼쪽 카메라 좌표계에서 오른쪽 카메라 좌표계로의 변환이다. 이를 추정하기 위해 두 카메라로 동일한 체커보드를 동시에 촬영한다.

OpenCV의 `cv2.stereoCalibrate()`는 두 카메라의 intrinsic을 고정하거나 동시에 정제하면서 $(\mathbf{R}, \mathbf{t})$를 추정한다:

```python
# 두 카메라의 intrinsic은 이미 캘리브레이션 되었다고 가정
# K_L, dist_L, K_R, dist_R: 각 카메라의 내부 파라미터

flags = cv2.CALIB_FIX_INTRINSIC  # intrinsic 고정

ret, K_L, dist_L, K_R, dist_R, R, t, E, F = cv2.stereoCalibrate(
    obj_points,
    img_points_left,
    img_points_right,
    K_L, dist_L,
    K_R, dist_R,
    gray.shape[::-1],
    flags=flags,
    criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6)
)

print(f"Stereo RMS error: {ret:.4f} pixels")
print(f"Baseline: {np.linalg.norm(t):.4f} m")
print(f"R:\n{R}")
print(f"t: {t.ravel()}")
```

### 3.2.2 Stereo Rectification

캘리브레이션 후, 두 카메라의 이미지를 **평행 정렬(rectification)**하여 에피폴라 라인이 수평이 되도록 변환한다. 정렬 후에는 대응점 탐색이 1D 문제(같은 행에서 탐색)로 축소되어 스테레오 매칭이 크게 효율화된다.

**에피폴라 기하학(Epipolar Geometry) 복습**:

두 카메라의 투영 중심 $O_L, O_R$과 3D 점 $\mathbf{P}$가 이루는 평면을 에피폴라 평면(epipolar plane)이라 한다. 이 평면이 각 이미지 평면과 만나는 선이 에피폴라 라인(epipolar line)이다. 왼쪽 이미지의 점 $\mathbf{m}_L$에 대응하는 오른쪽 이미지의 점 $\mathbf{m}_R$은 반드시 대응하는 에피폴라 라인 위에 존재한다.

이 관계를 수식으로 표현하면:

$$
\tilde{\mathbf{m}}_R^\top \mathbf{F} \tilde{\mathbf{m}}_L = 0
$$

여기서 $\mathbf{F}$는 **기본 행렬(Fundamental matrix)**이다. 캘리브레이션된 카메라에서는 **본질 행렬(Essential matrix)** $\mathbf{E}$를 사용한다:

$$
\hat{\mathbf{m}}_R^\top \mathbf{E} \hat{\mathbf{m}}_L = 0, \quad \mathbf{E} = [\mathbf{t}]_\times \mathbf{R}
$$

여기서 $\hat{\mathbf{m}} = \mathbf{K}^{-1}\tilde{\mathbf{m}}$는 정규화 좌표이고, $[\mathbf{t}]_\times$는 이동 벡터의 반대칭 행렬(skew-symmetric matrix)이다.

**Rectification 과정**:

Rectification은 두 카메라의 이미지 평면을 가상의 공통 평면으로 변환하는 호모그래피를 찾는 것이다. 이 공통 평면은 두 카메라의 기선(baseline) 벡터와 평행하도록 설정한다.

OpenCV의 `cv2.stereoRectify()`는 Bouguet의 알고리즘을 사용한다:

```python
# Rectification 매핑 계산
R_L, R_R, P_L, P_R, Q, roi_L, roi_R = cv2.stereoRectify(
    K_L, dist_L, K_R, dist_R,
    gray.shape[::-1],
    R, t,
    alpha=0  # 0: 유효 픽셀만 보존, 1: 모든 원본 픽셀 보존
)

# Undistort + Rectify 매핑 생성
map_Lx, map_Ly = cv2.initUndistortRectifyMap(
    K_L, dist_L, R_L, P_L, gray.shape[::-1], cv2.CV_32FC1
)
map_Rx, map_Ry = cv2.initUndistortRectifyMap(
    K_R, dist_R, R_R, P_R, gray.shape[::-1], cv2.CV_32FC1
)

# 이미지 정렬
img_L_rect = cv2.remap(img_L, map_Lx, map_Ly, cv2.INTER_LINEAR)
img_R_rect = cv2.remap(img_R, map_Rx, map_Ry, cv2.INTER_LINEAR)
```

**에피폴라 제약 기반 검증**: 정렬이 올바른지 확인하는 가장 직관적인 방법은 정렬된 좌우 이미지에 수평선을 그어보는 것이다. 대응점이 같은 수평선 위에 있으면 정렬이 정확한 것이다.

```python
# 에피폴라 라인 검증 시각화
def draw_epilines(img_L_rect, img_R_rect, num_lines=20):
    h, w = img_L_rect.shape[:2]
    canvas = np.hstack([img_L_rect, img_R_rect])
    for y in np.linspace(0, h-1, num_lines, dtype=int):
        color = tuple(np.random.randint(0, 255, 3).tolist())
        cv2.line(canvas, (0, y), (2*w, y), color, 1)
    return canvas
```

**디스패리티에서 깊이로**: 정렬된 스테레오 이미지에서 좌우 대응점의 수평 변위 $d = u_L - u_R$를 디스패리티(disparity)라 한다. 깊이는:

$$
Z = \frac{f \cdot B}{d}
$$

여기서 $f$는 초점 거리(픽셀 단위), $B = \|\mathbf{t}\|$는 기선 길이(baseline)이다. $Q$ 행렬을 이용하면 디스패리티 맵에서 직접 3D 점군을 계산할 수 있다: $\mathbf{P}_{3D} = Q \cdot [u, v, d, 1]^\top$.

---

## 3.3 Camera-LiDAR Extrinsic Calibration

카메라와 LiDAR 간의 외부 파라미터 $(\mathbf{R}, \mathbf{t})$를 추정하는 것은 멀티모달 센서 퓨전의 핵심 전제 조건이다. LiDAR 점군을 이미지에 투영하거나, 이미지 특징을 3D 공간에 배치하려면 이 변환이 정확해야 한다.

### 3.3.1 Target-based 캘리브레이션

가장 전통적인 접근법으로, 알려진 기하학적 타겟(체커보드, AprilTag 등)을 카메라와 LiDAR가 동시에 관측하여 대응점을 만든다.

**원리**: 체커보드의 코너는 카메라 이미지에서 2D 점으로, LiDAR 점군에서 3D 평면으로 관측된다. 체커보드 평면에 맞는 LiDAR 점들을 추출하고, 평면의 법선과 경계를 이용하여 3D-2D 대응을 구축한다.

**3D-2D 대응 기반 방법**:

1. 카메라 이미지에서 체커보드 코너 검출 → 2D 점 $\{\mathbf{m}_j\}$
2. LiDAR 점군에서 체커보드 평면을 RANSAC으로 피팅 → 평면 방정식 $\mathbf{n}^\top \mathbf{p} + d = 0$
3. 평면 위의 점들에서 체커보드 경계를 추출하여 코너의 3D 좌표 $\{\mathbf{P}_j\}$ 추정
4. 3D-2D 대응 $\{(\mathbf{P}_j, \mathbf{m}_j)\}$으로부터 PnP(Perspective-n-Point) 알고리즘으로 $(\mathbf{R}, \mathbf{t})$ 추정

**평면 제약 기반 방법**:

코너의 정확한 3D 위치를 추정하기 어려운 경우, 평면 제약만으로도 캘리브레이션이 가능하다. 카메라에서 검출한 코너를 역투영(back-project)하여 3D 광선을 만들고, 이 광선이 LiDAR에서 추정한 평면과 만나는 점을 대응점으로 사용한다.

$n$개의 체커보드 포즈에서 평면 제약:

$$
\mathbf{n}_i^\top (\mathbf{R} \mathbf{p}_{L,i} + \mathbf{t}) + d_i = 0, \quad \forall i = 1, \ldots, n
$$

여기서 $\mathbf{n}_i$는 카메라 좌표계에서의 평면 법선, $\mathbf{p}_{L,i}$는 LiDAR 좌표계에서의 평면 위 점이다.

```python
import numpy as np
import cv2

def calibrate_camera_lidar_target(
    img_corners_list,    # 각 이미지의 체커보드 2D 코너 [N_imgs x N_corners x 2]
    lidar_planes_list,   # 각 관측의 LiDAR 평면 파라미터 [N_imgs x (n, d)]
    K, dist,             # 카메라 intrinsic
    board_corners_3d     # 체커보드 좌표계에서의 3D 코너 [N_corners x 3]
):
    """Target-based Camera-LiDAR extrinsic calibration via PnP."""
    
    # 각 이미지에서 카메라->체커보드 변환 추정
    all_points_3d_lidar = []
    all_points_2d_camera = []
    
    for i, (corners_2d, (normal, d)) in enumerate(zip(img_corners_list, lidar_planes_list)):
        # 카메라에서 본 체커보드 pose (PnP)
        ret, rvec, tvec = cv2.solvePnP(
            board_corners_3d, corners_2d, K, dist
        )
        R_cam_board, _ = cv2.Rodrigues(rvec)
        
        # 체커보드 코너를 카메라 좌표계로 변환
        corners_cam = (R_cam_board @ board_corners_3d.T + tvec).T
        
        # LiDAR 평면 위의 점들 수집
        # (실제로는 LiDAR 점군에서 평면 inlier를 사용)
        all_points_2d_camera.append(corners_2d)
    
    # 최종 결과는 비선형 최적화로 정제
    return R_cam_lidar, t_cam_lidar
```

**실전 주의사항**:
- 체커보드의 크기가 충분히 커야 LiDAR 점이 충분히 들어온다. 최소 A2 이상, 이상적으로 A0 크기.
- 다양한 거리와 각도에서 10~20회 관측 필요.
- LiDAR 빔이 체커보드 평면에 충분한 수의 점을 찍어야 한다. 희소한 LiDAR(16채널)에서는 어렵다.

### 3.3.2 Targetless 캘리브레이션

실전에서는 캘리브레이션 타겟을 준비하고 설치하는 것이 번거롭거나 불가능한 경우가 많다. Targetless 캘리브레이션은 자연 장면의 정보만으로 센서 간 변환을 추정한다.

#### Mutual Information (MI) 기반 방법

**직관**: LiDAR 점군을 카메라 이미지에 투영했을 때, 두 데이터의 통계적 의존성(statistical dependency)이 최대가 되는 변환이 정확한 캘리브레이션이다.

**Mutual Information 정의**:

두 확률 변수 $X$(LiDAR 반사 강도 또는 깊이)와 $Y$(이미지 밝기)의 상호 정보량:

$$
\text{MI}(X; Y) = \sum_{x}\sum_{y} p(x, y) \log \frac{p(x, y)}{p(x)p(y)}
$$

또는 엔트로피로 표현하면:

$$
\text{MI}(X; Y) = H(X) + H(Y) - H(X, Y)
$$

$X$와 $Y$가 독립이면 $\text{MI} = 0$, 완벽히 의존적이면 $\text{MI} = H(X) = H(Y)$.

**Normalized Information Distance (NID)**:

[Koide et al. (2023)](https://arxiv.org/abs/2302.05094)은 MI를 정규화한 NID를 비용 함수로 사용한다:

$$
\text{NID}(X; Y) = 1 - \frac{\text{MI}(X; Y)}{H(X, Y)} = \frac{H(X, Y) - \text{MI}(X; Y)}{H(X, Y)}
$$

NID는 $[0, 1]$ 범위를 가지며, 값이 작을수록 두 데이터의 정합이 좋다.

**히스토그램 기반 MI 계산**:

실전에서 MI는 결합 히스토그램(joint histogram)으로 추정한다:

```python
import numpy as np

def compute_nid(lidar_intensity, image_intensity, bins=64):
    """
    NID (Normalized Information Distance) 계산.
    lidar_intensity: LiDAR 반사 강도 [N]
    image_intensity: 투영된 위치의 이미지 밝기 [N]
    """
    # 결합 히스토그램 (joint histogram) 계산
    hist_2d, _, _ = np.histogram2d(
        lidar_intensity, image_intensity, bins=bins,
        range=[[0, 255], [0, 255]]
    )
    
    # 확률로 정규화
    pxy = hist_2d / hist_2d.sum()
    px = pxy.sum(axis=1)  # marginal X
    py = pxy.sum(axis=0)  # marginal Y
    
    # 엔트로피 계산 (0 log 0 = 0 처리)
    eps = 1e-10
    H_xy = -np.sum(pxy[pxy > eps] * np.log(pxy[pxy > eps]))
    H_x = -np.sum(px[px > eps] * np.log(px[px > eps]))
    H_y = -np.sum(py[py > eps] * np.log(py[py > eps]))
    
    MI = H_x + H_y - H_xy
    NID = 1.0 - MI / (H_xy + eps)
    
    return NID, MI
```

MI 기반 캘리브레이션의 최적화는 기울기가 명시적이지 않으므로 Nelder-Mead 같은 기울기 없는(gradient-free) 최적화를 사용하거나, 수치 기울기를 사용한다. [Pandey et al. (2015)](https://onlinelibrary.wiley.com/doi/abs/10.1002/rob.21542)이 이 접근법을 LiDAR-카메라 캘리브레이션에 처음 적용했다.

#### Edge Alignment 기반 방법

**직관**: LiDAR 점군에서 추출한 깊이 불연속(depth discontinuity) 에지와 이미지에서 추출한 에지가 정합되도록 변환을 최적화한다.

LiDAR 깊이 이미지에서 에지를 추출하고 $\mathbf{e}_L$, 카메라 이미지의 에지를 $\mathbf{e}_C$라 하면:

$$
\min_{\mathbf{R}, \mathbf{t}} \sum_i \text{dist}(\pi(\mathbf{R}\mathbf{p}_{L,i} + \mathbf{t}), \mathbf{e}_C)
$$

여기서 $\pi(\cdot)$는 카메라 투영 함수, $\text{dist}(\cdot, \mathbf{e}_C)$는 가장 가까운 에지까지의 거리(distance transform 사용)이다.

#### Learning-based 방법

RegNet, CalibNet 등의 딥러닝 기반 방법은 LiDAR 점군의 깊이 이미지와 카메라 이미지를 입력으로 받아 6-DoF 변환을 직접 회귀(regression)한다. 이 방법들은 초기값 없이 캘리브레이션을 수행할 수 있지만, 현재는 정밀도에서 전통 방법에 미치지 못하며 학습 데이터의 도메인에 의존적이다.

### 3.3.3 Koide et al. (2023) — 최신 Targetless 캘리브레이션

[Koide et al. (2023)](https://arxiv.org/abs/2302.05094)의 `direct_visual_lidar_calibration`은 현재 가장 실용적인 targetless LiDAR-카메라 캘리브레이션 도구이다. 핵심 파이프라인:

**1단계: LiDAR 포인트 클라우드 밀집화**

회전식 LiDAR(Ouster, Velodyne 등)의 단일 스캔은 희소하여 MI 정합에 불충분하다. CT-ICP(Continuous-Time ICP) 알고리즘으로 수 초간의 연속 스캔을 정밀하게 누적하여 밀집 점군을 생성한다. Solid-state LiDAR(Livox 등)는 비반복 스캔 패턴 덕분에 자연스럽게 밀집화된다.

**2단계: SuperGlue 기반 초기 추정**

밀집 점군을 가상 카메라 시점에서 렌더링하여 LiDAR 강도 이미지를 생성한다. 이 렌더링 이미지와 실제 카메라 이미지 사이에서 SuperGlue(학습 기반 매칭)로 2D-2D 대응점을 검출한다. 이를 2D-3D 대응으로 변환한 뒤 RANSAC + PnP로 초기 변환을 추정한다.

이 단계가 혁신적인 이유는, 서로 다른 모달리티(LiDAR 강도 vs 카메라 RGB)의 이미지를 매칭하는 cross-modal correspondence 문제를 SuperGlue가 해결한다는 점이다. 초기 추정 성공률은 80% 이상이다.

**3단계: NID 기반 정밀 정합**

초기 추정을 시작점으로 NID를 최소화하는 Nelder-Mead 최적화를 수행한다. 이 때 뷰 기반 은닉점 제거(hidden point removal)로 카메라에서 보이지 않는 LiDAR 점을 제거하여 정합 품질을 높인다.

**결과**: 평균 이동 오차 0.043m, 회전 오차 0.374도. 회전식/솔리드스테이트 LiDAR, 핀홀/어안/전방향 카메라 등 다양한 조합에서 동작한다.

### 3.3.4 실전 도구 비교

| 도구 | 방식 | 타겟 필요 | 정밀도 | 자동화 수준 |
|------|------|----------|--------|-----------|
| Autoware Calibration Toolkit | Target-based | O | 높음 | 반자동 |
| `direct_visual_lidar_calibration` (Koide) | Targetless (NID) | X | 높음 | 자동 |
| ACSC (Automatic Calibration) | Target-based + auto corner | O | 높음 | 자동 |
| LiveCalib | Online | X | 중간 | 완전 자동 |

최근 [MFCalib (2024)](https://arxiv.org/abs/2409.00992)은 깊이 연속/불연속 에지와 강도 불연속 에지를 동시에 활용하여 single-shot targetless 캘리브레이션의 정밀도를 크게 향상시켰다. LiDAR 빔의 물리적 측정 원리를 모델링하여 에지 팽창(edge inflation) 문제를 해결한 점이 특징이다.

실전에서는 먼저 target-based로 정밀 캘리브레이션을 수행하고, 운용 중에 targetless 방법으로 캘리브레이션 드리프트를 모니터링하는 이중 전략이 효과적이다.

---

## 3.4 Camera-IMU Extrinsic + Temporal Calibration

카메라와 IMU 사이의 공간적 변위(extrinsic)뿐 아니라 시간적 오프셋(temporal offset)을 동시에 추정하는 것이 핵심이다. 현대의 VIO(Visual-Inertial Odometry) 시스템은 이 캘리브레이션에 강하게 의존한다.

### 3.4.1 왜 시간 오프셋이 중요한가

카메라와 IMU는 서로 다른 클록으로 데이터를 생성한다. 두 센서 간 시간 오프셋 $t_d$가 존재하면, 카메라 타임스탬프 $t_c$에 대응하는 IMU 데이터는 실제로 $t_c + t_d$ 시점의 것이다.

일반적인 카메라-IMU 시간 오프셋은 수 밀리초에서 수십 밀리초이다. 이 오프셋을 무시하면, 빠른 회전 시 재투영 오차가 크게 증가한다. 예를 들어, 시간 오프셋이 10ms이고 카메라가 100 deg/s로 회전 중이라면, 1도의 회전 오차가 발생한다.

### 3.4.2 Kalibr: Continuous-Time B-Spline 기반 캘리브레이션

[Furgale et al. (2013)](https://ieeexplore.ieee.org/document/6696514)이 제안한 Kalibr는 camera-IMU 캘리브레이션의 사실상 표준이다.

**핵심 아이디어**: 궤적(trajectory)을 이산적인 포즈의 시퀀스가 아닌, 연속 시간 B-spline으로 표현한다. 이렇게 하면 서로 다른 샘플링 레이트의 센서(카메라: 20~30Hz, IMU: 200~1000Hz)를 자연스럽게 처리할 수 있다.

**B-Spline 궤적 표현**:

3차(cubic) B-spline에서 시간 $t$에서의 포즈 $\mathbf{T}(t) \in SE(3)$는 제어점(control points) $\{\mathbf{T}_i\}$의 가중 조합으로 표현된다:

$$
\mathbf{T}(t) = \mathbf{T}_i \prod_{j=1}^{3} \text{Exp}(\mathbf{B}_j(u) \cdot \Omega_{i+j})
$$

여기서:
- $u = (t - t_i) / \Delta t$는 정규화된 시간 ($0 \leq u < 1$)
- $\mathbf{B}_j(u)$는 3차 B-spline 기저 함수의 계수
- $\Omega_{i+j} = \text{Log}(\mathbf{T}_{i+j-1}^{-1} \mathbf{T}_{i+j})$는 인접 제어점 간 상대 변환의 리 대수(Lie algebra) 표현
- $\text{Exp}, \text{Log}$는 $SE(3)$의 지수/로그 사상

B-spline의 핵심 장점:
1. **미분 가능**: 임의 시간에서 속도, 가속도를 해석적으로 계산 가능 → IMU 관측 모델과 직접 연결
2. **비동기 센서 처리**: 센서별 타임스탬프에 구속받지 않음
3. **국소성(locality)**: 각 기저 함수는 4개의 제어점에만 영향 → 희소(sparse) 최적화 가능

**관측 모델**:

카메라 관측: 시간 $t_c + t_d$ (시간 오프셋 보정)에서의 궤적 포즈로 3D 랜드마크를 투영:

$$
\mathbf{e}_{\text{cam},k} = \mathbf{m}_k - \pi\left(\mathbf{T}_{CB} \cdot \mathbf{T}(t_{c,k} + t_d) \cdot \mathbf{p}_w\right)
$$

여기서 $\mathbf{T}_{CB}$는 카메라-IMU 외부 파라미터 (추정 대상).

IMU 관측: 시간 $t_{\text{imu}}$에서의 궤적 미분으로 가속도와 각속도를 예측:

$$
\mathbf{e}_{\text{accel},k} = \mathbf{a}_k - \left[\mathbf{R}(t_k)^\top(\ddot{\mathbf{p}}(t_k) - \mathbf{g}) + \mathbf{b}_a\right]
$$
$$
\mathbf{e}_{\text{gyro},k} = \boldsymbol{\omega}_k - \left[\boldsymbol{\omega}(t_k) + \mathbf{b}_g\right]
$$

**최적화 문제**:

$$
\min_{\mathbf{T}_{CB}, t_d, \mathbf{b}_a, \mathbf{b}_g, \{\mathbf{T}_i\}} \sum_k \|\mathbf{e}_{\text{cam},k}\|^2_{\Sigma_c} + \sum_k \left(\|\mathbf{e}_{\text{accel},k}\|^2_{\Sigma_a} + \|\mathbf{e}_{\text{gyro},k}\|^2_{\Sigma_g}\right)
$$

이 최적화는 비선형 최소자승 문제로, Gauss-Newton 또는 LM 알고리즘으로 풀 수 있다. B-spline의 국소성 덕분에 자코비안이 희소하여 대규모 문제도 효율적으로 풀린다.

### 3.4.3 Kalibr 실행 가이드

```bash
# 1. AprilGrid 타겟 준비
kalibr_create_target_pdf --type apriltag \
    --nx 6 --ny 6 --tsize 0.024 --tspace 0.3

# 2. 데이터 수집 (ROS bag)
#    - 타겟을 카메라 시야에 놓고, 센서 리그를 다양하게 움직임
#    - 모든 축에서 회전 + 병진 운동을 포함
#    - 최소 60초, 이상적으로 2분 이상 수집

# 3. Camera-IMU 캘리브레이션 실행
kalibr_calibrate_imu_camera \
    --target april_6x6_24x24mm.yaml \
    --cam camchain.yaml \
    --imu imu.yaml \
    --bag calibration.bag \
    --bag-freq 20.0 \
    --timeoffset-padding 0.1
```

**IMU 설정 파일 (imu.yaml) 예시**:
```yaml
# imu.yaml
rostopic: /imu0
update_rate: 200.0  # Hz

# IMU 노이즈 파라미터 (Allan variance에서 측정, 3.4.4절 참조)
accelerometer_noise_density: 0.01    # m/s^2/sqrt(Hz)
accelerometer_random_walk: 0.0002   # m/s^3/sqrt(Hz)
gyroscope_noise_density: 0.005      # rad/s/sqrt(Hz)
gyroscope_random_walk: 4.0e-06      # rad/s^2/sqrt(Hz)
```

**데이터 수집 핵심 팁**:
1. **운동의 다양성**: 모든 6-DoF를 흥분(excite)해야 한다. 특히 각 축의 회전이 중요.
2. **운동 속도**: 너무 느리면 IMU bias 추정이 어렵고, 너무 빠르면 이미지가 흐려진다.
3. **타겟 가시성**: 전체 수집 시간의 80% 이상에서 타겟이 보여야 한다.
4. **시작과 끝**: 정지 상태에서 시작/끝나야 IMU bias 초기화가 용이하다.

**결과 해석**: Kalibr 출력에서 확인해야 할 항목:
- `T_cam_imu`: 카메라-IMU 외부 파라미터 (4x4 변환 행렬)
- `timeshift_cam_imu`: 시간 오프셋 $t_d$ (보통 수 ms)
- 재투영 오차 분포: 평균 0.2~0.5 px이 이상적
- 가속도계/자이로 잔차: 노이즈 모델과 일치해야 함

### 3.4.4 Allan Variance 측정 실습

IMU 노이즈 파라미터를 정확히 아는 것은 Kalibr 캘리브레이션뿐 아니라 모든 IMU 기반 시스템의 성능에 직접적으로 영향을 미친다. Allan variance는 시계열 데이터의 노이즈 특성을 클러스터 시간(cluster time)에 따라 분석하는 기법으로, 원래 원자 시계의 안정성을 측정하기 위해 개발되었다.

**Allan Variance 정의**:

시계열 데이터 $\{x_k\}$에서 클러스터 시간 $\tau = n \cdot \tau_0$ ($\tau_0$: 샘플링 주기)에 대한 Allan variance:

$$
\sigma^2(\tau) = \frac{1}{2(N-2n)} \sum_{k=1}^{N-2n} \left[\bar{x}_{k+n} - \bar{x}_k\right]^2
$$

여기서 $\bar{x}_k = \frac{1}{n}\sum_{j=0}^{n-1} x_{k+j}$는 클러스터 평균이다.

**로그-로그 플롯에서의 노이즈 식별**:

Allan deviation $\sigma(\tau)$를 $\tau$에 대해 로그-로그 스케일로 그리면, 각 노이즈 원인이 서로 다른 기울기의 직선으로 나타난다:

| 노이즈 유형 | 기울기 | $\sigma(\tau)$ |
|-----------|-------|---------------|
| Quantization noise | $-1$ | $\propto \tau^{-1}$ |
| **Angle/Velocity random walk** | $-1/2$ | $\propto \tau^{-1/2}$ |
| **Bias instability** | $0$ | 일정 (최솟값) |
| Rate random walk | $+1/2$ | $\propto \tau^{+1/2}$ |
| Rate ramp | $+1$ | $\propto \tau$ |

**실습 코드**:

```python
import numpy as np
import matplotlib.pyplot as plt

def compute_allan_variance(data, dt, max_clusters=100):
    """
    Allan variance 계산.
    data: 1D 시계열 (예: 자이로 x축 raw 데이터) [N]
    dt: 샘플링 주기 [s]
    max_clusters: 로그 스케일로 분배할 클러스터 수
    """
    N = len(data)
    max_n = N // 2
    
    # 로그 스케일로 cluster size 선택
    n_values = np.unique(
        np.logspace(0, np.log10(max_n), max_clusters).astype(int)
    )
    
    taus = []
    allan_vars = []
    
    for n in n_values:
        tau = n * dt
        
        # 클러스터 평균 계산
        n_clusters = N // n
        trimmed = data[:n_clusters * n]
        clusters = trimmed.reshape(n_clusters, n)
        cluster_means = clusters.mean(axis=1)
        
        # Allan variance
        diffs = np.diff(cluster_means)
        avar = 0.5 * np.mean(diffs**2)
        
        taus.append(tau)
        allan_vars.append(avar)
    
    return np.array(taus), np.array(allan_vars)

def extract_imu_noise_params(taus, allan_vars, dt):
    """
    Allan variance 플롯에서 IMU 노이즈 파라미터를 추출.
    """
    adev = np.sqrt(allan_vars)
    
    # 1. Random walk (기울기 -1/2): tau=1에서의 값
    # N (noise density) = sigma(tau=1) 
    idx_1s = np.argmin(np.abs(taus - 1.0))
    noise_density = adev[idx_1s]
    
    # 2. Bias instability: Allan deviation의 최솟값
    bias_instability = np.min(adev)
    tau_min = taus[np.argmin(adev)]
    
    # 3. Random walk (기울기 +1/2): 장기 기울기에서 추출
    # K (rate random walk) = sigma(tau=3) * sqrt(3) (근사)
    # 실전에서는 선형 회귀로 정밀 추출
    
    return {
        'noise_density': noise_density,
        'bias_instability': bias_instability,
        'tau_min': tau_min
    }

# 사용 예시
# 1. IMU를 정지 상태로 최소 2시간 이상 녹화
# 2. 자이로/가속도 각 축의 raw 데이터를 분석

# 예시: 200Hz IMU 데이터, 2시간 분량
dt = 1.0 / 200  # 5ms
# gyro_x = np.loadtxt("imu_static_gyro_x.txt")  # 실제 데이터

# 시뮬레이션 데이터로 시연
np.random.seed(42)
N = 200 * 3600 * 2  # 2시간
noise_density = 0.005  # rad/s/sqrt(Hz)
bias = 0.0001  # rad/s (천천히 드리프트)
gyro_x = noise_density * np.sqrt(1/dt) * np.random.randn(N)
gyro_x += bias * np.cumsum(np.random.randn(N)) * np.sqrt(dt)

taus, avars = compute_allan_variance(gyro_x, dt)
params = extract_imu_noise_params(taus, avars, dt)

print(f"Gyroscope noise density: {params['noise_density']:.6f} rad/s/sqrt(Hz)")
print(f"Bias instability: {params['bias_instability']:.6f} rad/s")
print(f"Min at tau = {params['tau_min']:.1f} s")

# Allan deviation 플롯
plt.figure(figsize=(10, 6))
plt.loglog(taus, np.sqrt(avars), 'b-', linewidth=1.5)
plt.xlabel('Cluster time τ (s)')
plt.ylabel('Allan deviation σ(τ)')
plt.title('Gyroscope Allan Deviation')
plt.grid(True, which='both', alpha=0.3)

# 기울기 참조선
tau_ref = np.array([0.01, 100])
plt.loglog(tau_ref, params['noise_density'] / np.sqrt(tau_ref), 
           'r--', label='Random walk (-1/2)')
plt.axhline(y=params['bias_instability'], color='g', linestyle='--', 
            label=f'Bias instability ({params["bias_instability"]:.1e})')
plt.legend()
plt.savefig('allan_deviation.png', dpi=150)
plt.show()
```

**Allan Variance 측정 실전 팁**:
- **정지 데이터 수집**: IMU를 진동이 없는 단단한 표면 위에 놓고, 최소 2시간 (이상적으로 6시간) 녹화한다. 짧은 데이터로는 bias instability의 최솟값을 정확히 식별할 수 없다.
- **데이터시트와의 비교**: 제조사 데이터시트의 noise density 값과 Allan variance에서 추출한 값을 비교하여 센서 상태를 검증한다.
- **Kalibr와의 연결**: Kalibr의 `imu.yaml`에 입력하는 `accelerometer_noise_density`와 `gyroscope_noise_density`가 정확히 Allan variance에서 $\tau = 1$초에서의 Allan deviation 값이다.
- **온도 안정화**: IMU는 온도에 민감하므로, 전원 투입 후 15~30분 동안 워밍업 후 녹화를 시작한다.

---

## 3.5 LiDAR-IMU Extrinsic Calibration

LiDAR와 IMU 간의 외부 파라미터를 추정하는 문제는 고전적인 **hand-eye calibration** 문제로 환원된다.

### 3.5.1 Hand-Eye Calibration (AX = XB)

로봇 팔의 끝(hand)에 카메라가 달려 있고, 로봇의 기저(base)에서 본 팔 끝의 움직임과 카메라에서 본 타겟의 움직임 사이의 관계를 구하는 문제가 hand-eye calibration의 원형이다 ([Tsai & Lenz, 1989](https://ieeexplore.ieee.org/document/34770)).

**문제 정의**:

센서 $A$(예: LiDAR)와 센서 $B$(예: IMU)가 강체(rigid body)에 고정되어 있을 때, 두 시점 $i, j$에서 각 센서가 관측한 상대 운동이 $\mathbf{A}_{ij}$와 $\mathbf{B}_{ij}$라 하자:

$$
\mathbf{A}_{ij} = \mathbf{T}_A^{-1}(t_i) \cdot \mathbf{T}_A(t_j) \quad \text{(센서 A의 상대 운동)}
$$
$$
\mathbf{B}_{ij} = \mathbf{T}_B^{-1}(t_i) \cdot \mathbf{T}_B(t_j) \quad \text{(센서 B의 상대 운동)}
$$

두 센서 간 고정된 변환 $\mathbf{X} = \mathbf{T}_{AB}$는 다음을 만족한다:

$$
\mathbf{A}_{ij} \mathbf{X} = \mathbf{X} \mathbf{B}_{ij}
$$

이것이 $\mathbf{AX} = \mathbf{XB}$ 방정식이다. $\mathbf{X} \in SE(3)$를 풀어야 한다.

**회전과 이동의 분리**:

회전과 이동으로 분리하면:

$$
\mathbf{R}_A \mathbf{R}_X = \mathbf{R}_X \mathbf{R}_B \quad \text{(회전)}
$$
$$
\mathbf{R}_A \mathbf{t}_X + \mathbf{t}_A = \mathbf{R}_X \mathbf{t}_B + \mathbf{t}_X \quad \text{(이동)}
$$

회전 방정식은 $\mathbf{t}_X$에 독립이므로, 먼저 $\mathbf{R}_X$를 풀고, 그 다음 이동 방정식에서 $\mathbf{t}_X$를 풀 수 있다.

**[Tsai & Lenz (1989)](https://ieeexplore.ieee.org/document/34770)의 해법**:

회전 방정식을 축-각(axis-angle) 표현으로 변환한다. $\mathbf{R}_A$의 회전축이 $\hat{\mathbf{a}}$이고 회전각이 $\alpha$이면, modified Rodrigues 파라미터를 사용하여:

$$
\text{skew}(\hat{\mathbf{a}}_A + \hat{\mathbf{a}}_B) \cdot \mathbf{r}_X = \hat{\mathbf{a}}_A - \hat{\mathbf{a}}_B
$$

여기서 $\mathbf{r}_X$는 $\mathbf{R}_X$의 수정 로드리게스 벡터이다. 여러 운동 쌍에서 이 방정식을 쌓으면 선형 시스템 $\mathbf{C} \mathbf{r}_X = \mathbf{d}$를 얻고, 최소 2개의 (비평행 회전축을 가진) 운동 쌍이 있으면 풀 수 있다. 이동 벡터 $\mathbf{t}_X$도 유사한 선형 시스템으로 풀린다.

**더 많은 운동 쌍의 활용**: 실전에서는 수십에서 수백 개의 운동 쌍을 사용하여 overdetermined system을 최소자승으로 풀고, LM 최적화로 정제한다.

### 3.5.2 Motion-based 자동 캘리브레이션

Hand-eye calibration의 핵심 요구사항은 두 센서 각각의 모션 추정(odometry)이 이미 존재해야 한다는 것이다. LiDAR odometry와 IMU integration이 독립적으로 동작하므로, 별도의 캘리브레이션 타겟 없이 센서를 자유롭게 움직이면서 데이터를 수집할 수 있다.

```python
import numpy as np
from scipy.spatial.transform import Rotation

def hand_eye_calibration_tsai(A_rotations, A_translations, 
                                B_rotations, B_translations):
    """
    Tsai-Lenz hand-eye calibration (AX = XB).
    
    A_rotations: [N x 3 x 3] 센서 A의 상대 회전 행렬
    A_translations: [N x 3] 센서 A의 상대 이동
    B_rotations: [N x 3 x 3] 센서 B의 상대 회전 행렬
    B_translations: [N x 3] 센서 B의 상대 이동
    """
    N = len(A_rotations)
    
    # Step 1: 회전 RX 추정
    C = []
    d = []
    for i in range(N):
        # 축-각 표현으로 변환
        rA = Rotation.from_matrix(A_rotations[i]).as_rotvec()
        rB = Rotation.from_matrix(B_rotations[i]).as_rotvec()
        
        alpha = np.linalg.norm(rA)
        beta = np.linalg.norm(rB)
        
        if alpha < 1e-6 or beta < 1e-6:
            continue  # 작은 회전은 무시
        
        # Modified Rodrigues parameters
        a_prime = np.tan(alpha / 2) * rA / alpha
        b_prime = np.tan(beta / 2) * rB / beta
        
        # skew(a' + b') * rX = a' - b'
        skew_sum = np.array([
            [0, -(a_prime[2]+b_prime[2]), a_prime[1]+b_prime[1]],
            [a_prime[2]+b_prime[2], 0, -(a_prime[0]+b_prime[0])],
            [-(a_prime[1]+b_prime[1]), a_prime[0]+b_prime[0], 0]
        ])
        
        C.append(skew_sum)
        d.append(a_prime - b_prime)
    
    C = np.vstack(C)
    d = np.concatenate(d)
    
    # 최소자승 풀이
    rX, _, _, _ = np.linalg.lstsq(C, d, rcond=None)
    
    # Modified Rodrigues → 회전 행렬
    angle = 2 * np.arctan(np.linalg.norm(rX))
    if angle > 1e-6:
        axis = rX / np.linalg.norm(rX)
        R_X = Rotation.from_rotvec(angle * axis).as_matrix()
    else:
        R_X = np.eye(3)
    
    # Step 2: 이동 tX 추정
    C_t = []
    d_t = []
    for i in range(N):
        C_t.append(A_rotations[i] - np.eye(3))
        d_t.append(R_X @ B_translations[i] - A_translations[i])
    
    C_t = np.vstack(C_t)
    d_t = np.concatenate(d_t)
    
    t_X, _, _, _ = np.linalg.lstsq(C_t, d_t, rcond=None)
    
    return R_X, t_X
```

### 3.5.3 LI-Init (FAST-LIO 계열)

FAST-LIO2와 같은 최신 LIO 시스템은 LiDAR-IMU 외부 파라미터를 온라인으로 자동 초기화하는 기능을 포함한다. **LI-Init** 방식의 핵심 아이디어:

1. IMU 데이터만으로 자세(attitude)를 추정하고, LiDAR 매칭으로 포즈를 추정
2. 두 추정의 차이로부터 상대 변환을 반복적으로 정제
3. Error-State Iterated Kalman Filter (ESIKF)의 상태 벡터에 LiDAR-IMU extrinsic을 포함하여 온라인 추정

이 접근법은 별도의 캘리브레이션 과정 없이, LIO 시스템 시작 시 자동으로 외부 파라미터를 추정한다. 초기 수 초 동안 충분히 다양한 운동이 있으면 수렴한다.

**장점**: 별도 도구나 절차가 필요 없음. 현장에서 바로 사용 가능.
**단점**: 초기 운동이 불충분하면 수렴하지 않거나 부정확할 수 있음. Target-based 방법보다 정밀도가 낮을 수 있음.

**GRIL-Calib**: 지상 로봇처럼 운동이 평면에 제한되는 경우, 기존 방법은 일부 축의 관측 가능성이 떨어져 정밀도가 저하된다. [GRIL-Calib (Kim et al., 2024)](https://arxiv.org/abs/2312.14035)은 지면 평면 잔차(ground plane residual)를 LiDAR odometry에 활용하고, 지면 평면 운동(GPM) 제약을 최적화에 통합하여 평면 운동만으로도 6-DoF 캘리브레이션 파라미터를 추정할 수 있게 했다.

지금까지 단일 LiDAR와 단일 IMU 사이의 캘리브레이션을 다루었다. 하지만 자율주행 차량처럼 여러 대의 LiDAR를 사용하는 시스템에서는 LiDAR 간의 상대 포즈도 결정해야 한다.

---

## 3.6 LiDAR-LiDAR Extrinsic Calibration

다중 LiDAR 시스템(multi-LiDAR rig)에서 각 LiDAR 간의 상대 포즈를 추정하는 문제이다. 자율주행 차량에서 전방/측방/후방에 LiDAR를 배치하는 경우 필수적이다.

### 3.6.1 문제 설정

$n$개의 LiDAR $\{L_1, L_2, \ldots, L_n\}$가 차체에 고정되어 있을 때, 기준 LiDAR $L_1$에 대한 나머지 LiDAR의 상대 변환 $\{\mathbf{T}_{L_1 L_i}\}_{i=2}^{n}$를 추정한다.

### 3.6.2 Target-based 방법

대형 평면 타겟(패널)을 사용하여 여러 LiDAR가 동시에 관측할 수 있도록 한다. 각 LiDAR에서 타겟 평면을 RANSAC으로 피팅하고, 평면 파라미터의 제약 조건으로부터 상대 변환을 추정한다.

### 3.6.3 Targetless: ICP 기반

가장 자연스러운 targetless 방법은 중첩 영역(overlapping region)이 있는 두 LiDAR의 점군을 ICP로 정합하는 것이다.

**중첩 영역이 있는 경우**:

두 LiDAR의 FOV가 겹치는 영역이 있으면, 해당 영역의 점군에 대해 직접 ICP를 적용한다:

$$
\min_{\mathbf{R}, \mathbf{t}} \sum_{i} \|\mathbf{p}_{L_2,i} - (\mathbf{R} \mathbf{p}_{L_1,i'} + \mathbf{t})\|^2
$$

Point-to-plane ICP가 일반적으로 더 빠르게 수렴한다:

$$
\min_{\mathbf{R}, \mathbf{t}} \sum_{i} \left[(\mathbf{R} \mathbf{p}_{L_1,i'} + \mathbf{t} - \mathbf{p}_{L_2,i})^\top \mathbf{n}_{L_2,i}\right]^2
$$

여기서 $\mathbf{n}_{L_2,i}$는 target 점의 법선 벡터이다.

**중첩 영역이 없는 경우**:

FOV가 겹치지 않으면 직접 정합이 불가능하다. 이 경우:
1. Hand-eye calibration(3.5.1절): 각 LiDAR의 odometry로부터 상대 운동을 추출하여 AX=XB를 풀 수 있다.
2. SLAM 기반: 각 LiDAR가 독립적으로 SLAM을 수행하고, 공통 환경의 전역 맵에서 매칭하여 상대 변환을 추정한다.

### 3.6.4 Feature-based 방법

구조화된 환경(건물, 도로 등)에서는 평면, 기둥, 에지 같은 기하학적 특징을 추출하여 매칭에 사용한다. 이 방법은 점 단위 매칭보다 노이즈에 강건하다:

- 여러 LiDAR에서 동일한 대형 평면(벽, 바닥)을 추출
- 평면 파라미터 $(n_i, d_i)$를 각 LiDAR 좌표계에서 계산
- 대응 평면 쌍으로부터 상대 변환 추정

최소 3개의 비공선(non-coplanar) 평면 대응이 필요하다.

지금까지 다룬 캘리브레이션(3.1~3.6)은 카메라, LiDAR, IMU 사이의 관계였다. 야외 환경에서 GNSS를 활용하는 시스템에서는 GNSS 안테나와 IMU 사이의 공간적 관계도 정밀하게 알아야 한다.

---

## 3.7 GNSS-IMU Lever Arm & Boresight

GNSS 안테나와 IMU 사이의 공간적 관계를 **레버 암(lever arm)**이라 한다. 이는 GNSS/INS 통합 항법에서 핵심적인 캘리브레이션 파라미터이다.

### 3.7.1 Lever Arm Vector

GNSS 안테나의 위상 중심(phase center)과 IMU의 원점 사이의 3D 벡터 $\mathbf{l} = [l_x, l_y, l_z]^\top$를 IMU body frame에서 정의한다.

GNSS가 측정하는 위치는 안테나의 위상 중심이지만, 우리가 원하는 것은 IMU(또는 차량 기준점)의 위치이다:

$$
\mathbf{p}_{\text{IMU}} = \mathbf{p}_{\text{GNSS}} - \mathbf{R}_{\text{body}}^{\text{nav}} \cdot \mathbf{l}
$$

여기서 $\mathbf{R}_{\text{body}}^{\text{nav}}$는 body frame에서 navigation frame으로의 회전 행렬이다. 차량이 회전하면 GNSS 안테나 위치가 변하므로, lever arm을 보정하지 않으면 위치 오차가 발생한다. lever arm이 1m이고 차량이 10 deg 기울면, 약 17cm의 위치 오차가 생긴다.

### 3.7.2 Lever Arm 추정 방법

**방법 1: 물리적 측정**

가장 직관적인 방법은 줄자, 레이저 거리 측정기 등으로 직접 측정하는 것이다. 정밀도는 수 cm 수준이며, 대부분의 응용에서 충분하다.

**방법 2: 필터 기반 온라인 추정**

EKF 상태 벡터에 lever arm $\mathbf{l}$을 포함하여 온라인으로 추정한다. GNSS 관측 모델에서 lever arm이 관측 가능(observable)하려면 충분한 회전 운동이 필요하다. 직선 주행만으로는 lever arm의 전방 성분($l_x$)을 추정하기 어렵다.

**방법 3: 사후 처리(post-processing)**

수집된 GNSS/IMU 데이터를 전후방(forward-backward) 스무더로 처리하면서 lever arm을 최적화한다. 정밀 측량 분야에서 주로 사용한다.

### 3.7.3 GNSS 안테나 위상 중심

GNSS 안테나의 위상 중심(Antenna Phase Center, APC)은 안테나의 물리적 중심과 일치하지 않으며, 위성의 고도각과 주파수에 따라 달라진다. 이 변동을 Phase Center Variation (PCV)이라 하며, mm~cm 수준이다.

정밀 측위(RTK/PPP)에서는 안테나 위상 중심 보정 데이터(ANTEX 파일)를 적용해야 한다. 로봇 수준의 응용에서는 보통 무시해도 되지만, 측량 수준의 정밀도가 필요하면 반드시 고려한다.

---

## 3.8 Online / Continuous Calibration

운용 중에 캘리브레이션 파라미터가 변할 수 있다. 온도 변화, 진동, 충격 등으로 센서 마운트가 미세하게 변형되면 초기 캘리브레이션이 점차 부정확해진다. 이를 보정하기 위해 **온라인 캘리브레이션(online calibration)**이 필요하다.

### 3.8.1 Self-Calibration during SLAM

SLAM 시스템의 상태 벡터에 캘리브레이션 파라미터를 포함하여 운용 중에 추정하는 방법이다.

**EKF 기반**: OpenVINS는 카메라 intrinsic, camera-IMU extrinsic, 시간 오프셋을 상태 벡터에 포함하여 온라인으로 추정한다.

상태 벡터 확장:
$$
\mathbf{x} = \begin{bmatrix} \mathbf{x}_{\text{nav}} \\ \mathbf{x}_{\text{calib}} \end{bmatrix}
= \begin{bmatrix} \mathbf{q}, \mathbf{p}, \mathbf{v}, \mathbf{b}_g, \mathbf{b}_a \\ \mathbf{q}_{CI}, \mathbf{p}_{CI}, t_d, f_x, f_y, c_x, c_y, k_1, \ldots \end{bmatrix}
$$

캘리브레이션 파라미터의 프로세스 모델은 일반적으로 random walk:

$$
\dot{\mathbf{x}}_{\text{calib}} = \mathbf{w}_{\text{calib}}, \quad \mathbf{w}_{\text{calib}} \sim \mathcal{N}(\mathbf{0}, \mathbf{Q}_{\text{calib}})
$$

$\mathbf{Q}_{\text{calib}}$의 크기는 캘리브레이션 파라미터가 얼마나 빨리 변할 수 있는지를 반영한다. 너무 크면 파라미터가 불안정하게 변동하고, 너무 작으면 실제 변화를 추적하지 못한다.

**Factor graph 기반**: LIO-SAM이나 VINS-Mono에서도 외부 파라미터를 최적화 변수로 포함할 수 있다. Between factor의 노이즈 모델로 캘리브레이션의 안정성을 제어한다.

### 3.8.2 Extrinsic Drift 보정

장기간 운용하는 시스템(자율주행 차량, 로봇)에서 외부 파라미터의 드리프트를 감지하고 보정하는 전략:

1. **재투영 오차 모니터링**: LiDAR 점군을 이미지에 투영했을 때 에지 정합 품질을 지속적으로 모니터링. 정합 품질이 떨어지면 캘리브레이션 재수행을 트리거.

2. **주기적 재캘리브레이션**: 운용 데이터를 이용하여 주기적으로 targetless 캘리브레이션을 재수행.

3. **온라인 미세 조정**: 현재 캘리브레이션을 초기값으로 하여 작은 범위 내에서 연속적으로 최적화.

최근 [CalibRefine (2025)](https://arxiv.org/abs/2502.17648)은 딥러닝 기반으로 raw LiDAR 점군과 카메라 이미지를 직접 입력받아 온라인 targetless 캘리브레이션을 수행하며, 반복적 후처리 정제(iterative post-refinement)로 정밀도를 높인 자동 프레임워크를 제안했다.

### 3.8.3 OpenCalib: 자율주행 통합 캘리브레이션 프레임워크

[OpenCalib (Yan et al., 2023)](https://arxiv.org/abs/2205.14087)은 자율주행 시스템에 필요한 다양한 캘리브레이션을 하나의 프레임워크로 통합한 오픈소스 도구이다.

**지원하는 캘리브레이션 유형**:

| 센서 쌍 | 방법 | 타겟 |
|---------|------|------|
| Camera intrinsic | Zhang's method | 체커보드/AprilTag |
| Camera-Camera | Stereo calibration | 공유 타겟 |
| Camera-LiDAR | PnP / Edge alignment | 있음/없음 |
| LiDAR-LiDAR | ICP / Feature matching | 없음 |
| Camera-Ground | Vanishing point | 없음 |
| 온라인 보정 | 모니터링 + 재캘리브레이션 | 없음 |

**OpenCalib의 설계 철학**:
- **모듈화**: 각 캘리브레이션 유형이 독립 모듈로 구현되어, 필요한 것만 사용 가능
- **통합 인터페이스**: ROS 기반 통합 인터페이스로 다양한 센서 입력을 처리
- **시각화**: 캘리브레이션 결과를 실시간으로 시각화하여 품질을 직관적으로 확인

자율주행 시스템에서의 캘리브레이션은 단일 센서 쌍이 아닌 **센서 체인(sensor chain)**의 일관성이 중요하다. 예를 들어, 차량에 6대의 카메라와 1대의 LiDAR가 있으면, 각 카메라-LiDAR 캘리브레이션이 독립적으로 수행되더라도 카메라 간 상대 포즈가 일관되어야 한다. OpenCalib은 이런 일관성 제약을 전역 최적화에 통합한다.

---

## 3.9 Temporal Calibration

센서 간 시간 동기화는 공간 캘리브레이션만큼 중요하다. 빠르게 움직이는 플랫폼에서 수 밀리초의 시간 오차도 센티미터 수준의 위치 오차로 이어진다.

### 3.9.1 Hardware Synchronization

하드웨어 동기화는 가장 정확하고 신뢰할 수 있는 방법이다.

**Trigger 기반 동기화**:

하나의 마스터 타이머가 모든 센서에 하드웨어 트리거 신호를 보내 동시에 데이터를 캡처하게 한다.

- **카메라 트리거**: 외부 트리거 입력에 연결된 GPIO 핀으로 노출 시작 시점을 제어
- **LiDAR 동기**: 일부 LiDAR(Ouster 등)는 PPS(Pulse Per Second) 입력을 지원하여 스캔 시작 시점을 동기화
- **IMU 동기**: IMU의 샘플링 클록을 외부 기준에 고정

**PPS (Pulse Per Second)**:

GNSS 수신기는 GPS 시간에 동기된 1Hz의 전기 펄스(PPS)를 출력한다. 이 펄스의 상승 에지(rising edge)가 정확히 GPS 초의 경계에 대응하므로, 모든 센서의 로컬 타임스탬프를 GPS 시간에 정렬하는 기준으로 사용할 수 있다.

```
                   PPS Signal (from GNSS)
                   ───┐     ┌───┐     ┌───┐     ┌───
                      │     │   │     │   │     │
                      └─────┘   └─────┘   └─────┘
                   t=0     t=1       t=2       t=3   (GPS seconds)
                   
   Camera capture  ──X──X──X──X──X──X──X──X──X──X──  (30 Hz)
   LiDAR scan      ──X─────X─────X─────X─────X─────  (10 Hz)
   IMU sample      ──XXXX──XXXX──XXXX──XXXX──XXXX──  (200 Hz)
```

PPS 동기화의 핵심은 각 센서의 로컬 타임스탬프와 PPS 펄스 사이의 오프셋을 측정하여, 모든 데이터를 공통 시간축(GPS time)에 정렬하는 것이다.

### 3.9.2 PTP (Precision Time Protocol)

IEEE 1588 PTP는 이더넷 네트워크를 통해 마이크로초 수준의 시간 동기화를 제공하는 프로토콜이다. NTP(Network Time Protocol)의 밀리초 수준 정확도보다 훨씬 정밀하다.

**PTP 동작 원리**:

1. **마스터-슬레이브 구조**: 네트워크에서 하나의 마스터 클록(Grandmaster)이 기준 시간을 제공
2. **Sync 메시지**: 마스터가 주기적으로 Sync 메시지를 브로드캐스트
3. **Follow-up**: 정확한 송신 타임스탬프를 별도 메시지로 전송
4. **Delay Request/Response**: 슬레이브가 네트워크 지연을 측정
5. **오프셋 계산**: $\text{offset} = \frac{(t_2 - t_1) - (t_4 - t_3)}{2}$

```
   Master (Grandmaster)              Slave (Sensor)
         |                                |
         |------- Sync (t1) ------------->|  (t2: 수신 시각)
         |                                |
         |------- Follow-up (t1) -------->|  (정확한 t1 전달)
         |                                |
         |<------ Delay_Req (t3) ---------|  (t3: 송신 시각)
         |                                |
         |------- Delay_Resp (t4) ------->|  (t4: 수신 시각 전달)
         |                                |
         
   offset = [(t2 - t1) - (t4 - t3)] / 2
   delay  = [(t2 - t1) + (t4 - t3)] / 2
```

최신 LiDAR(Ouster, Hesai 등)와 산업용 카메라(FLIR/Lucid 등)는 PTP 하드웨어 타임스탬핑을 지원하여, 소프트웨어 오버헤드 없이 마이크로초 수준의 동기화가 가능하다.

### 3.9.3 Software Synchronization

하드웨어 동기화가 불가능한 경우, 소프트웨어로 시간 정렬을 수행한다.

**Host Clock 기반**: 모든 센서 데이터가 호스트 컴퓨터에 도착하면 호스트의 시스템 시계로 타임스탬프를 찍는다. 단순하지만 USB/네트워크 지연의 불확실성(jitter)이 밀리초 수준으로 존재한다.

**ROS Time**: ROS(Robot Operating System)에서는 `ros::Time::now()`로 메시지 수신 시각을 기록한다. 드라이버에서 센서의 로컬 타임스탬프를 ROS 시간으로 변환하는 경우, 변환의 정확도가 전체 동기화 정밀도를 결정한다.

### 3.9.4 Time Offset 온라인 추정

[Li & Mourikis (2014)](https://journals.sagepub.com/doi/abs/10.1177/0278364913515286)는 시간 오프셋 $t_d$를 VIO의 상태 벡터에 포함하여 온라인으로 추정하는 방법을 제안했다. 핵심 아이디어:

**관측 모델에서의 시간 오프셋 반영**:

카메라 관측 시각 $t_c$에서의 실제 센서 포즈는 $t_c + t_d$이다. IMU preintegration에서 이를 반영:

$$
\mathbf{z}(t_c) = \pi(\mathbf{T}(t_c + t_d) \cdot \mathbf{p}_w)
$$

$t_d$가 작다고 가정하면, 1차 테일러 전개:

$$
\mathbf{T}(t_c + t_d) \approx \mathbf{T}(t_c) \cdot \text{Exp}(\boldsymbol{\xi} \cdot t_d)
$$

여기서 $\boldsymbol{\xi}$는 $t_c$에서의 body 속도 (angular + linear velocity)이다.

이 근사를 통해 $t_d$에 대한 자코비안을 해석적으로 계산할 수 있고, EKF 또는 최적화 프레임워크에서 다른 상태 변수와 함께 추정할 수 있다.

최근 [iKalibr (Chen et al., 2024)](https://arxiv.org/abs/2407.11420)는 이 아이디어를 다중 센서로 확장하여, LiDAR·카메라·IMU·radar 등 이종 센서 간의 시공간 파라미터를 B-spline 연속 시간 프레임워크에서 **한 번에** 추정하는 통합 캘리브레이션 도구를 제안했다 (IEEE T-RO 2025).

**관측 가능성(Observability) 조건**: 시간 오프셋이 관측 가능하려면 플랫폼이 충분한 가속 운동을 해야 한다. 등속 직선 운동에서는 시간 오프셋을 추정할 수 없다 (시간 이동이 공간 이동과 구별 불가).

Kalibr(3.4.2절), OpenVINS, VINS-Mono 등 현대의 VIO 시스템은 모두 이 방법의 변형을 구현하고 있다.

### 3.9.5 실전 동기화 전략 가이드

| 정밀도 요구 | 추천 방법 | 비용 |
|-----------|----------|------|
| $< 1\mu s$ | PPS + HW trigger | 높음 (전용 HW 필요) |
| $1\mu s - 100\mu s$ | PTP | 중간 (PTP 지원 장비) |
| $100\mu s - 1ms$ | NTP + 온라인 추정 | 낮음 |
| $> 1ms$ | Host clock + 온라인 추정 | 없음 |

**자율주행 수준**: PPS + PTP 조합이 표준. 모든 센서를 GPS 시간에 동기화.

**연구/프로토타이핑 수준**: 소프트웨어 동기화 + 온라인 시간 오프셋 추정으로 충분한 경우가 많다. Kalibr로 초기 오프셋을 추정하고, VIO 시스템에서 온라인으로 미세 조정한다.

**핵심 원칙**: 하드웨어 동기화가 가능하면 반드시 사용하라. 소프트웨어 동기화는 하드웨어 동기화의 보완이지 대체가 아니다.

---

## 3.10 챕터 요약

이 챕터에서 다룬 캘리브레이션의 전체 체계를 정리한다.

| 캘리브레이션 유형 | 추정 파라미터 | 핵심 방법 | 최소 요구 조건 |
|----------------|-------------|----------|-------------|
| Camera intrinsic | $\mathbf{K}, \mathbf{d}$ | Zhang's method | 체커보드 15~25장 |
| Stereo extrinsic | $\mathbf{R}, \mathbf{t}$ | 공유 타겟 + stereoCalibrate | 동시 관측 15~25쌍 |
| Camera-LiDAR | $\mathbf{T}_{CL}$ | Target-based / NID | 타겟 10~20회 / 자연 장면 |
| Camera-IMU | $\mathbf{T}_{CI}, t_d$ | Kalibr (B-spline) | 60초 이상 다양한 운동 |
| LiDAR-IMU | $\mathbf{T}_{LI}$ | Hand-eye (AX=XB) / LI-Init | 다양한 운동 |
| LiDAR-LiDAR | $\mathbf{T}_{L_1 L_2}$ | ICP / Feature matching | 중첩 영역 또는 운동 데이터 |
| GNSS-IMU | $\mathbf{l}$ (lever arm) | 물리적 측정 / EKF 추정 | 회전 운동 |
| Temporal | $t_d$ | PPS/PTP + 온라인 추정 | 가속 운동 |

**캘리브레이션 순서 권장**:

1. 각 카메라의 intrinsic (독립적으로 수행)
2. 스테레오 extrinsic (있는 경우)
3. Camera-IMU extrinsic + temporal (Kalibr)
4. LiDAR-IMU extrinsic (hand-eye 또는 LI-Init)
5. Camera-LiDAR extrinsic (camera-IMU와 LiDAR-IMU의 체인으로 간접 계산하거나 직접 캘리브레이션)
6. GNSS lever arm
7. 온라인 캘리브레이션 설정 (운용 중 보정)

간접 계산의 예: Camera-LiDAR 변환은 $\mathbf{T}_{CL} = \mathbf{T}_{CI} \cdot \mathbf{T}_{IL}$로 구할 수 있다. 단, 오차가 누적되므로 직접 캘리브레이션으로 검증하는 것이 좋다.

**핵심 논문 정리**:

- [Zhang (2000)](https://ieeexplore.ieee.org/document/888718): 카메라 intrinsic — 호모그래피 기반의 유연한 캘리브레이션
- [Furgale et al. (2013)](https://ieeexplore.ieee.org/document/6696514): Camera-IMU — B-spline 연속 시간 궤적으로 시공간 동시 캘리브레이션
- [Tsai & Lenz (1989)](https://ieeexplore.ieee.org/document/34770): Hand-eye — AX=XB의 원조
- [Koide et al. (2023)](https://arxiv.org/abs/2302.05094): Camera-LiDAR targetless — NID + SuperGlue 기반 자동 캘리브레이션
- [Li & Mourikis (2014)](https://journals.sagepub.com/doi/abs/10.1177/0278364913515286): Temporal — 시간 오프셋 온라인 추정
- [OpenCalib (2023)](https://arxiv.org/abs/2205.14087): 자율주행 통합 캘리브레이션 프레임워크
- [GRIL-Calib (Kim et al., 2024)](https://arxiv.org/abs/2312.14035): 지상 로봇 환경에서 평면 운동 제약을 활용한 targetless IMU-LiDAR 캘리브레이션. 제한된 운동에서도 6-DoF 추정 가능.
- [MFCalib (2024)](https://arxiv.org/abs/2409.00992): 다중 특징 에지(깊이 연속/불연속, 강도 불연속)를 활용한 single-shot targetless LiDAR-카메라 캘리브레이션. LiDAR 빔 모델로 에지 팽창 문제를 해결.
- [iKalibr (Chen et al., 2024)](https://arxiv.org/abs/2407.11420): Temporal — B-spline 연속 시간 기반 다중 이종 센서(LiDAR·카메라·IMU·radar) 통합 시공간 캘리브레이션 (IEEE T-RO 2025).

센서 모델(Ch.2)과 캘리브레이션 파라미터(Ch.3)가 갖추어졌으면, 이제 센서 데이터로부터 로봇의 상태를 추정하는 알고리즘을 다룰 차례다. Ch.4에서는 칼만 필터에서 팩터 그래프까지, 센서 퓨전의 수학적 엔진인 **상태 추정 이론**을 체계적으로 유도한다.
