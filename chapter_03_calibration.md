# Ch.3 — Calibration Deep Dive

Ch.2의 관측 모델을 실제 센서 데이터에 쓰려면 모델 안의 파라미터 값을 알아야 한다. 카메라의 초점 거리, LiDAR와 IMU 사이의 상대 위치, 센서 사이의 시간 오프셋이 그런 값이다. 이 값을 재서 정하는 일을 캘리브레이션(calibration)이라 한다.

퓨전 결과가 이상하면 캘리브레이션부터 확인한다. 센서의 내부 모델이나 센서 사이의 상대 위치·자세가 틀리면 추정값에 일정한 방향의 오차(편향)가 끼고, 조건에 따라 필터나 최적화가 발산하기도 한다. LiDAR-카메라 퓨전에서 각도가 1도 틀리면 50m 앞 물체는 옆으로 약 87cm 어긋난다($50 \tan 1° \approx 0.87$m).

---

## 3.1 Camera Intrinsic Calibration

카메라 내부 캘리브레이션(intrinsic calibration)에서는 3D 점이 2D 이미지 어디에 찍힐지 정하는 파라미터를 추정한다. 초점 거리(focal length), 주점(principal point), 렌즈 왜곡 계수(distortion coefficients)가 그런 파라미터다.

### 3.1.1 핀홀 카메라 모델 복습

3D 세계 좌표 $\mathbf{P}_w = [X, Y, Z]^\top$를 카메라 좌표계의 $\mathbf{P}_c = [X_c, Y_c, Z_c]^\top$로 옮긴 뒤 이미지 평면에 투영하는 과정은 아래 식으로 쓴다:

$$
s \begin{bmatrix} u \\ v \\ 1 \end{bmatrix}
= \mathbf{K} \begin{bmatrix} X_c \\ Y_c \\ Z_c \end{bmatrix}
= \begin{bmatrix} f_x & \gamma & c_x \\ 0 & f_y & c_y \\ 0 & 0 & 1 \end{bmatrix} \begin{bmatrix} X_c \\ Y_c \\ Z_c \end{bmatrix}
$$

여기서:
- $f_x, f_y$: 픽셀 단위의 초점 거리 (focal length). $f_x = f / p_x$이며 $f$는 물리적 초점 거리(mm), $p_x$는 픽셀 크기(mm/pixel).
- $(c_x, c_y)$: 주점. 이미지 센서 평면과 광축(optical axis)의 교점.
- $\gamma$: 비대칭 계수(skew coefficient). 현대 카메라에서는 거의 0.
- $s = Z_c$: 스케일 팩터 (깊이값).

$\mathbf{K}$를 카메라 내부 행렬(intrinsic matrix) 또는 캘리브레이션 행렬이라 부른다. 자유도는 $f_x, f_y, c_x, c_y, \gamma$의 다섯이다. 실제로는 $\gamma = 0$으로 두고 넷만 추정하는 경우가 많다.

### 3.1.2 렌즈 왜곡 모델

실제 렌즈는 핀홀 모델의 직선 투영에서 벗어난다. 왜곡은 크게 방사 왜곡(radial distortion)과 접선 왜곡(tangential distortion)으로 나뉜다.

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

$k_1 < 0$이면 중심에서 멀수록 배율이 줄어 직선이 바깥으로 불룩하게 휘는 배럴(barrel) 왜곡이 생긴다. $k_1 > 0$이면 배율이 커져 직선이 중심 쪽으로 오목하게 휘는 핀쿠션(pincushion) 왜곡이 생긴다. 보통의 렌즈는 $k_1, k_2$만 써도 왜곡이 대부분 설명된다.

**접선 왜곡** (tangential distortion):
$$
x_{\text{tangential}} = 2p_1 xy + p_2(r^2 + 2x^2)
$$
$$
y_{\text{tangential}} = p_1(r^2 + 2y^2) + 2p_2 xy
$$

접선 왜곡은 렌즈 요소가 이미지 센서 평면과 완벽히 평행하지 않을 때 생긴다.

**왜곡을 적용한 최종 좌표**:
$$
x_d = x_{\text{radial}} + x_{\text{tangential}}, \quad y_d = y_{\text{radial}} + y_{\text{tangential}}
$$

$$
u = f_x \cdot x_d + c_x, \quad v = f_y \cdot y_d + c_y
$$

왜곡 파라미터는 $\mathbf{d} = [k_1, k_2, p_1, p_2, k_3]$ 순서로 묶는다. OpenCV의 `calibrateCamera()`도 이 순서로 돌려준다.

### 3.1.3 Zhang's Method: 호모그래피 기반 캘리브레이션

[Zhang (2000)](https://ieeexplore.ieee.org/document/888718)의 방법은 평면 패턴(체커보드)을 여러 자세로 찍은 이미지에서 카메라 파라미터를 추정한다. 입체 캘리브레이션 장치(3D calibration rig) 없이 인쇄한 평면 하나로 끝나서 널리 쓰인다. OpenCV의 `calibrateCamera()`가 이 방법을 따른다.

패턴이 $Z = 0$ 평면에 놓이면 3D-2D 투영이 호모그래피로 단순화된다.

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

$\mathbf{H}$는 $3 \times 3$ 호모그래피 행렬이다. 이미지 한 장에서 대응점이 넷만 있으면 DLT(Direct Linear Transform)로 $\mathbf{H}$를 구한다. 체커보드 코너는 보통 수십 개이므로 식이 미지수보다 많은 과결정(over-determined) 시스템을 SVD로 푼다.

**DLT로 호모그래피 추정**: 대응점 $(\mathbf{M}_j, \mathbf{m}_j)$에서 $\tilde{\mathbf{m}}_j \times \mathbf{H} \tilde{\mathbf{M}}_j = \mathbf{0}$으로부터 다음 선형 시스템을 얻는다:

$$
\begin{bmatrix}
\tilde{\mathbf{M}}_j^\top & \mathbf{0}^\top & -u_j \tilde{\mathbf{M}}_j^\top \\
\mathbf{0}^\top & \tilde{\mathbf{M}}_j^\top & -v_j \tilde{\mathbf{M}}_j^\top
\end{bmatrix}
\mathbf{h} = \mathbf{0}
$$

여기서 $\mathbf{h}$는 $\mathbf{H}$의 9개 원소를 펼친 벡터다. $n$개 대응점에서 $2n \times 9$ 행렬 $\mathbf{A}$를 만들고, $\|\mathbf{A}\mathbf{h}\|$를 최소화하는 $\mathbf{h}$를 $\mathbf{A}$의 SVD에서 가장 작은 singular value에 대응하는 right singular vector로 구한다.

#### Step 2: 내부 파라미터의 제약 조건 도출

$\mathbf{H} = [\mathbf{h}_1 \quad \mathbf{h}_2 \quad \mathbf{h}_3]$로 열을 나누면:

$$
[\mathbf{h}_1 \quad \mathbf{h}_2 \quad \mathbf{h}_3] = \lambda \mathbf{K} [\mathbf{r}_1 \quad \mathbf{r}_2 \quad \mathbf{t}]
$$

여기서 $\lambda$는 임의의 스케일 팩터다. $\mathbf{r}_1, \mathbf{r}_2$는 회전 행렬의 열이므로 다음 두 제약이 생긴다.

1. **직교 조건**: $\mathbf{r}_1^\top \mathbf{r}_2 = 0$
   $$\mathbf{h}_1^\top \mathbf{K}^{-\top} \mathbf{K}^{-1} \mathbf{h}_2 = 0$$

2. **동일 노름 조건**: $\|\mathbf{r}_1\| = \|\mathbf{r}_2\|$
   $$\mathbf{h}_1^\top \mathbf{K}^{-\top} \mathbf{K}^{-1} \mathbf{h}_1 = \mathbf{h}_2^\top \mathbf{K}^{-\top} \mathbf{K}^{-1} \mathbf{h}_2$$

$\mathbf{B} = \mathbf{K}^{-\top} \mathbf{K}^{-1}$로 두자. $\mathbf{B}$는 대칭 양의 정부호(positive definite) 행렬이라 독립 원소가 여섯이다.

$$
\mathbf{B} = \begin{bmatrix} B_{11} & B_{12} & B_{13} \\ B_{12} & B_{22} & B_{23} \\ B_{13} & B_{23} & B_{33} \end{bmatrix}
$$

이를 벡터로 쓰면 $\mathbf{b} = [B_{11}, B_{12}, B_{22}, B_{13}, B_{23}, B_{33}]^\top$이다.

#### Step 3: 선형 시스템 구성

$\mathbf{h}_i^\top \mathbf{B} \mathbf{h}_j$는 $\mathbf{b}$와 어떤 벡터의 내적으로 쓸 수 있다.

$$
\mathbf{h}_i^\top \mathbf{B} \mathbf{h}_j = \mathbf{v}_{ij}^\top \mathbf{b}
$$

여기서:
$$
\mathbf{v}_{ij} = \begin{bmatrix} h_{1i}h_{1j} \\ h_{1i}h_{2j} + h_{2i}h_{1j} \\ h_{2i}h_{2j} \\ h_{3i}h_{1j} + h_{1i}h_{3j} \\ h_{3i}h_{2j} + h_{2i}h_{3j} \\ h_{3i}h_{3j} \end{bmatrix}
$$

이미지 한 장에서 나오는 두 제약은 다음과 같다.

$$
\begin{bmatrix} \mathbf{v}_{12}^\top \\ (\mathbf{v}_{11} - \mathbf{v}_{22})^\top \end{bmatrix} \mathbf{b} = \mathbf{0}
$$

$n$장의 이미지가 있으면 $2n \times 6$ 시스템을 얻는다.

**최소 이미지 수**: $\gamma = 0$으로 두면 $B_{12} = 0$ 제약이 더해진다. 남은 동차 성분 다섯은 스케일을 빼면 자유도가 넷이므로, 퇴화하지 않은 자세라면 이미지 두 장으로 풀린다. 다섯 파라미터를 모두 두면 서로 다른 평면 자세가 셋 있어야 선형 해가 나온다. 이 수는 식을 풀기 위한 최솟값이다. 실제로 몇 장을 찍을지는 보드가 시야 곳곳에 여러 거리와 기울기로 찍혔는지, 추정 문제의 조건수가 괜찮은지를 보고 정한다.

#### Step 4: K 복원

$\mathbf{b}$를 풀면 $\mathbf{B}$가 나오고, Cholesky 분해 $\mathbf{B} = \mathbf{L}\mathbf{L}^\top$에서 $\mathbf{K}^{-1} = \mathbf{L}^\top$이므로 $\mathbf{K}$를 얻는다. 닫힌 식은 다음과 같다.

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

여기서 $\lambda = 1 / \|\mathbf{K}^{-1} \mathbf{h}_1\|$이고, $\mathbf{r}_3 = \mathbf{r}_1 \times \mathbf{r}_2$다. 노이즈 때문에 이렇게 얻은 $[\mathbf{r}_1, \mathbf{r}_2, \mathbf{r}_3]$는 정확한 회전 행렬이 되지 못한다. 그래서 SVD로 가장 가까운 회전 행렬 $\mathbf{R} = \mathbf{U}\mathbf{V}^\top$를 찾는다($\det(\mathbf{R}) = 1$이 되도록).

#### Step 6: 비선형 최적화 (MLE)

선형 해는 왜곡을 무시하고 대수 오차를 줄인 값이어서 초기값으로만 쓴다. 마지막으로 Levenberg-Marquardt(LM) 알고리즘으로 재투영 오차(reprojection error)를 줄인다.

$$
\min_{\mathbf{K}, \mathbf{d}, \{\mathbf{R}_i, \mathbf{t}_i\}} \sum_{i=1}^{n}\sum_{j=1}^{m} \|\mathbf{m}_{ij} - \hat{\mathbf{m}}(\mathbf{K}, \mathbf{d}, \mathbf{R}_i, \mathbf{t}_i, \mathbf{M}_j)\|^2
$$

여기서 $n$은 이미지 수, $m$은 각 이미지의 코너 수, $\hat{\mathbf{m}}(\cdot)$은 왜곡을 포함한 전체 투영 함수이다.

자코비안은 투영 함수를 파라미터마다 편미분해서 얻는다. OpenCV의 `calibrateCamera()`는 DLT, 선형 K 추정, LM 최적화를 한 함수로 묶어 둔다.

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

# 캘리브레이션 실행 (Zhang's method)
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
    residual = img_points[i].reshape(-1, 2) - img_points_proj.reshape(-1, 2)
    error = np.sqrt(np.mean(np.sum(residual ** 2, axis=1)))
    errors.append(error)

print(f"Per-image mean error: {np.mean(errors):.4f} pixels")
print(f"Max error image: {np.argmax(errors)} ({max(errors):.4f} px)")
```

### 3.1.5 실전 팁: 좋은 캘리브레이션을 위한 조건

캘리브레이션 품질은 데이터를 어떻게 모으느냐에 크게 달렸다. 아래 조건은 파라미터가 서로 구분되도록 관측을 구성하는 원칙이다.

**자세 다양성 (Pose Diversity)**: 체커보드를 여러 각도와 위치에서 찍는다.
- 체커보드를 이미지의 상하좌우와 중앙에 배치해 시야 전역을 관측
- 정면에 가까운 포즈만 반복하지 말고 여러 축으로 충분히 기울여 초점 거리와 외부 파라미터를 구분
- 보드를 가까이도, 멀리도 두어 크기가 다르게 찍히도록 한다
- 검출에 성공한 장수보다 포즈 다양성과 재투영 잔차의 공간적 패턴을 기준으로 추가 촬영 여부를 결정

**코너 정확도**: 검출기가 서브픽셀 정제를 하지 않으면 `cv2.cornerSubPix()` 같은 함수로 코너 위치를 다듬는다. 코너 검출이 실패했거나 보드가 흐리고 반사된 이미지는 제외한다.

**조명 조건**: 조명이 고르면 코너 대비가 일정하다. 강한 그림자와 반사(glare)가 있으면 검출 위치가 틀어지므로 무광 타겟을 쓰고, 찍힌 이미지를 눈으로 확인한다.

**재투영 오차 해석**: 적정 오차는 해상도, 렌즈 모델, 타겟 크기, 검출기 정확도에 따라 달라서 기준값 하나로 판정하기 어렵다. 전체 RMS 외에 이미지별 잔차, 영상 위치별로 잔차가 한쪽으로 쏠리는지, 보정한 영상에서 직선이 곧게 펴졌는지, 따로 찍은 검증 영상의 오차를 본다.

**이상치 이미지 찾기**: 위 코드의 `errors`로 이미지마다 재투영 오차를 비교한다. 오차가 큰 이미지가 나오면 검출 실패, 모션 블러, 반사, 휘어진 타겟이 원인인지 먼저 본다. 그 이미지를 빼기 전과 뺀 뒤에 파라미터가 얼마나 바뀌는지, 검증 영상 오차가 어떻게 되는지 비교해서 뺄지 정한다.

**경고 — 정사각형 크기**: `SQUARE_SIZE`는 인쇄한 체커보드 칸의 실제 크기와 같아야 한다. 프린터가 크기를 줄이거나 늘려 찍는 일이 있으니 반드시 자로 잰다. 이 값이 틀리면 `tvecs`(이동 벡터)의 크기가 그만큼 틀린다. 왜곡 계수는 정규화 좌표에서, `K`는 픽셀 단위로 정의되므로 영향을 받지 않는다.

### 3.1.6 Fisheye / Omnidirectional 캘리브레이션

광각·어안 렌즈는 시야 가장자리가 표준 radial-tangential 모델로 잘 맞지 않는다. 시야각(FoV) 몇 도부터 다른 모델을 써야 한다는 경계는 없다. 렌즈의 투영 방식과 잔차가 쏠리는 모양을 보고 equidistant나 Kannala–Brandt 같은 모델을 고른다.

**등거리 투영 모델 (Equidistant Projection Model)**:

[Kannala & Brandt (2006)](https://ieeexplore.ieee.org/document/1642666)의 일반 카메라 모델은 입사각(incidence angle) $\theta$와 이미지 중심에서 투영점까지의 거리 $r$을 다항식으로 잇는다.

$$
r(\theta) = k_1 \theta + k_2 \theta^3 + k_3 \theta^5 + k_4 \theta^7 + k_5 \theta^9
$$

여기서 $\theta = \arctan\left(\sqrt{x^2 + y^2}\right)$는 입사각, $x, y$는 정규화 좌표이다. 이상적인 등거리 투영은 $r = f\theta$이며, 다항식의 추가 항이 실제 렌즈의 편차를 보정한다.

OpenCV의 `cv2.fisheye` 모듈이 이 모델을 조금 바꿔 구현해 두었다.

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
# CALIB_CHECK_COND는 조건수가 나쁜 이미지가 하나라도 있으면 예외로 중단한다.
# 그때는 해당 이미지를 제외하거나 이 플래그를 뺀다.

K_fisheye = np.zeros((3, 3))
D_fisheye = np.zeros((4, 1))  # k1, k2, k3, k4

# fisheye 모듈은 객체점을 (N, 1, 3), 이미지점을 (N, 1, 2)로 요구한다 (3.1.4의 (N, 3) 객체점을 변환)
obj_points_f = [o.reshape(-1, 1, 3).astype(np.float64) for o in obj_points]
img_points_f = [p.reshape(-1, 1, 2).astype(np.float64) for p in img_points]

ret, K_fisheye, D_fisheye, rvecs, tvecs = cv2.fisheye.calibrate(
    obj_points_f,
    img_points_f,
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

[Scaramuzza et al. (2006)](https://rpg.ifi.uzh.ch/docs/IROS06_scaramuzza.pdf)의 [OCamCalib toolbox](https://sites.google.com/site/scarabotix/ocamcalib-omnidirectional-camera-calibration-toolbox-for-matlab)는 거울과 렌즈를 함께 쓰는 catadioptric 카메라와 초광각 어안 렌즈를 한 도구로 캘리브레이션한다. 투영 함수 자체를 다항식으로 두기 때문에 센서 종류가 달라도 같은 식을 쓴다.

$$
\begin{bmatrix} u \\ v \end{bmatrix} = \begin{bmatrix} x_c \\ y_c \end{bmatrix} + \mathbf{A} \cdot \rho(\theta) \begin{bmatrix} \cos(\phi) \\ \sin(\phi) \end{bmatrix}
$$

여기서 $\mathbf{A}$는 affine 변환 행렬(stretch와 non-square pixel 보정), $\rho(\theta)$는 입사각에 따른 이미지 반경이다.

### 3.1.7 Kalibr를 이용한 카메라 캘리브레이션

[Kalibr](https://github.com/ethz-asl/kalibr)는 ETH Zurich ASL이 공개한 캘리브레이션 도구로, 카메라-IMU 캘리브레이션(§3.4)에 널리 쓰인다. 여러 카메라 모델과 batch 최적화를 지원해서 카메라 intrinsic도 이 도구로 함께 추정한다. OpenCV보다 정밀한지는 같은 모델, 같은 타겟 관측, 같은 초기값으로 돌려 비교해 봐야 안다.

**Kalibr의 카메라 모델 지원**:

| 모델 | 파라미터 수 | 적합한 렌즈 |
|------|-----------|-----------|
| `pinhole-radtan` | 4 + 4 | 일반 렌즈 (OpenCV와 동일) |
| `pinhole-equi` | 4 + 4 | 어안 렌즈 |
| `omni-radtan` | 5 + 4 | 초광각/catadioptric |
| `ds` (Double Sphere) | 6 | 광각 렌즈 |
| `eucm` (Extended UCM) | 6 | 광각 렌즈 |

**AprilGrid vs Checkerboard**: Kalibr는 AprilGrid 타겟을 권한다. AprilGrid는 태그마다 고유 ID가 들어 있어서 보드 일부가 가려져도(partial occlusion) 코너를 찾고, 어느 코너가 몇 번인지도 저절로 안다.

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

Kalibr의 내부 동작:
1. AprilGrid의 코너를 각 프레임에서 검출
2. 코너 대응으로부터 호모그래피 추정 (Zhang's method의 Step 1과 유사)
3. 카메라 모델의 모든 파라미터를 batch 최적화로 추정
4. 최적화 결과와 잔차(residuals) 분포를 시각화

OpenCV 기본 캘리브레이션에 없는 Kalibr의 기능은 다음과 같다.
- **B-spline 궤적 표현**: camera-IMU 캘리브레이션에서 연속 시간 궤적으로 비동기 관측을 처리
- **카메라 모델**: double sphere(DS), EUCM 같은 광각용 모델 지원
- **멀티카메라**: 여러 카메라의 상대 포즈를 동시에 추정 가능
- **IMU 연동**: camera-IMU 캘리브레이션(§3.4)과 연결

---

## 3.2 Camera-Camera (Stereo) Extrinsic

스테레오 카메라로 깊이를 추정하려면 먼저 두 카메라 사이의 상대 포즈(extrinsic)를 캘리브레이션해야 한다.

### 3.2.1 스테레오 캘리브레이션

두 카메라 $C_L$(왼쪽)과 $C_R$(오른쪽)이 있을 때, 같은 3D 점 $\mathbf{P}$에 대해:

$$
\mathbf{P}_{C_R} = \mathbf{R} \cdot \mathbf{P}_{C_L} + \mathbf{t}
$$

여기서 $(\mathbf{R}, \mathbf{t})$는 왼쪽 카메라 좌표를 오른쪽 카메라 좌표로 옮기는 변환이다. 이 값을 구하려면 두 카메라로 같은 체커보드를 동시에 찍는다.

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

캘리브레이션을 마치면 두 카메라 이미지를 평행 정렬(rectification)해 epipolar line이 수평이 되게 바꾼다. 정렬한 뒤에는 대응점을 같은 행에서만 찾으면 되므로(1D 탐색) 스테레오 매칭이 훨씬 빨라진다.

**Epipolar Geometry 복습**:

두 카메라의 투영 중심 $O_L, O_R$과 3D 점 $\mathbf{P}$가 이루는 평면을 epipolar plane이라 한다. 이 평면이 각 이미지 평면과 만나는 선이 epipolar line이다. 왼쪽 이미지의 점 $\mathbf{m}_L$과 짝을 이루는 오른쪽 점 $\mathbf{m}_R$은 반드시 $\mathbf{m}_L$이 정하는 epipolar line 위에 있다.

식으로 쓰면 다음과 같다.

$$
\tilde{\mathbf{m}}_R^\top \mathbf{F} \tilde{\mathbf{m}}_L = 0
$$

여기서 $\mathbf{F}$는 기본 행렬(Fundamental matrix)이다. 내부 파라미터를 아는 카메라에서는 본질 행렬(Essential matrix) $\mathbf{E}$를 쓴다.

$$
\hat{\mathbf{m}}_R^\top \mathbf{E} \hat{\mathbf{m}}_L = 0, \quad \mathbf{E} = [\mathbf{t}]_\times \mathbf{R}
$$

여기서 $\hat{\mathbf{m}} = \mathbf{K}^{-1}\tilde{\mathbf{m}}$는 정규화 좌표이고, $[\mathbf{t}]_\times$는 이동 벡터의 반대칭 행렬(skew-symmetric matrix)이다.

**Rectification 과정**:

Rectification은 두 카메라의 이미지를 같은 가상 평면으로 옮기는 호모그래피를 찾는 일이다. 이 가상 평면은 두 카메라를 잇는 기선(baseline)과 나란하게 잡는다. 그러면 epipolar line이 모두 수평선이 된다.

OpenCV의 `cv2.stereoRectify()`는 Bouguet의 알고리즘을 쓴다.

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

**Epipolar 제약으로 검증하기**: 정렬한 좌우 이미지에 수평선을 그어 본다. 짝이 되는 점이 같은 수평선에 놓이면 정렬이 맞다.

```python
# epipolar line 검증 시각화
def draw_epilines(img_L_rect, img_R_rect, num_lines=20):
    h, w = img_L_rect.shape[:2]
    canvas = np.hstack([img_L_rect, img_R_rect])
    for y in np.linspace(0, h-1, num_lines, dtype=int):
        color = tuple(np.random.randint(0, 255, 3).tolist())
        cv2.line(canvas, (0, y), (2*w, y), color, 1)
    return canvas
```

**디스패리티에서 깊이로**: 정렬한 스테레오 이미지에서 좌우 대응점이 수평으로 떨어진 거리 $d = u_L - u_R$를 디스패리티(disparity)라 한다. 깊이는 다음과 같다.

$$
Z = \frac{f \cdot B}{d}
$$

여기서 $f$는 픽셀 단위 초점 거리, $B = \|\mathbf{t}\|$는 기선 길이다. `stereoRectify()`가 돌려주는 $Q$ 행렬로 디스패리티 맵에서 바로 3D 포인트 클라우드를 얻는다. $[X,Y,Z,W]^\top = Q \cdot [u,v,d,1]^\top$로 동차 좌표를 구한 뒤 $W$로 나누어 $\mathbf{P}_{3D} = [X/W,Y/W,Z/W]^\top$를 얻는다.

---

## 3.3 Camera-LiDAR Extrinsic Calibration

LiDAR 포인트 클라우드를 이미지에 투영하거나 이미지 특징을 3D 공간에 놓으려면 카메라와 LiDAR 사이의 외부 파라미터 $(\mathbf{R}, \mathbf{t})$를 정확히 알아야 한다.

### 3.3.1 Target-based 캘리브레이션

Target-based 캘리브레이션은 알려진 기하학적 타겟(체커보드, AprilTag 등)을 카메라와 LiDAR가 동시에 관측하여 대응점을 만든다.

**원리**: 카메라 이미지에서는 체커보드 코너를 2D 점으로 찾고, LiDAR 포인트 클라우드에서는 보드 표면을 3D 평면으로 찾는다. 보드 평면에 들어맞는 LiDAR 점을 골라내고, 평면의 법선과 경계로 3D-2D 대응을 만든다.

**3D-2D 대응 기반 방법**:

1. 카메라 이미지에서 체커보드 코너 검출 → 2D 점 $\{\mathbf{m}_j\}$
2. LiDAR 포인트 클라우드에서 체커보드 평면을 RANSAC으로 피팅 → 평면 방정식 $\mathbf{n}^\top \mathbf{p} + d = 0$
3. 평면 위의 점들에서 체커보드 경계를 추출하여 코너의 3D 좌표 $\{\mathbf{P}_j\}$ 추정
4. 3D-2D 대응 $\{(\mathbf{P}_j, \mathbf{m}_j)\}$에 PnP(Perspective-n-Point) 알고리즘을 적용해 $(\mathbf{R}, \mathbf{t})$ 추정

**평면 제약 기반 방법**:

코너의 3D 위치를 정확히 추정하기 어려우면 평면 제약만으로도 캘리브레이션한다. 카메라에서 검출한 코너와 알려진 체커보드 형상으로 카메라 좌표계의 평면을 추정한다. LiDAR의 체커보드 점을 미지의 외부 변환으로 카메라 좌표계에 옮겼을 때 이 평면 위에 놓이도록 제약한다.

$n$개의 체커보드 포즈에서 평면 제약:

$$
\mathbf{n}_i^\top (\mathbf{R} \mathbf{p}_{L,i} + \mathbf{t}) + d_i = 0, \quad \forall i = 1, \ldots, n
$$

여기서 $\mathbf{n}_i$는 카메라 좌표계로 쓴 평면 법선, $\mathbf{p}_{L,i}$는 LiDAR 좌표계로 쓴 평면 위의 점이다.

```python
import numpy as np
import cv2

def calibrate_camera_lidar_target(
    img_corners_list,    # 각 이미지의 체커보드 2D 코너 [N_imgs x N_corners x 2]
    lidar_planes_list,   # 각 관측의 LiDAR 평면 파라미터 [N_imgs x (n, d)]
    K, dist,             # 카메라 intrinsic
    board_corners_3d     # 체커보드 좌표계 기준 3D 코너 [N_corners x 3]
):
    """Target-based Camera-LiDAR extrinsic calibration via PnP."""

    # 각 이미지에서 카메라->체커보드 변환 추정
    all_points_3d_lidar = []
    all_points_2d_camera = []

    # 각 관측에서 (카메라 좌표계 평면, LiDAR 좌표계 평면) 쌍을 모은다. 평면은 n·x + d = 0
    planes_cam, planes_lidar = [], []
    for corners_2d, (normal_l, d_l) in zip(img_corners_list, lidar_planes_list):
        # 카메라에서 본 체커보드 pose (PnP)
        ret, rvec, tvec = cv2.solvePnP(board_corners_3d, corners_2d, K, dist)
        R_cam_board, _ = cv2.Rodrigues(rvec)
        # 체커보드 평면을 카메라 좌표계로: 법선은 보드 z축, 거리는 보드 원점에서
        n_c = R_cam_board[:, 2]
        d_c = -n_c @ tvec.ravel()
        planes_cam.append((n_c, d_c))
        planes_lidar.append((normal_l, d_l))

    # 회전: 법선 쌍 정렬 (Kabsch/SVD), n_c = R n_l
    N_c = np.stack([n for n, _ in planes_cam])
    N_l = np.stack([n for n, _ in planes_lidar])
    U, _, Vt = np.linalg.svd(N_c.T @ N_l)
    R_cam_lidar = U @ np.diag([1, 1, np.sign(np.linalg.det(U @ Vt))]) @ Vt

    # 이동: 평면 거리 제약 n_c · t = d_l - d_c 를 최소자승으로 (평면 3개 이상 필요)
    b = np.array([d_l - d_c for (_, d_c), (_, d_l) in zip(planes_cam, planes_lidar)])
    t_cam_lidar, *_ = np.linalg.lstsq(N_c, b, rcond=None)

    # 최종 결과는 비선형 최적화로 정제
    return R_cam_lidar, t_cam_lidar
```

**실전 주의사항**:
- 체커보드 크기는 거리, 각도 해상도, 빔 패턴에 맞춰 한 관측에서 평면을 안정적으로 추정할 만큼 점이 들어오도록 정한다.
- 거리와 입사각을 바꾼 여러 관측을 모으고, 관측 수보다 평면의 방향·위치 다양성과 최적화의 조건수를 확인한다.
- 희소한 LiDAR에서는 타겟에 닿는 점이 적을 수 있으므로 실제 점 분포를 확인한 뒤 타겟 크기와 거리를 조정한다.

### 3.3.2 Targetless 캘리브레이션

캘리브레이션 타겟을 준비해 설치하기가 번거롭거나 아예 못 할 때가 있다. Targetless 캘리브레이션은 자연 장면만 보고 센서 사이 변환을 추정한다.

#### Mutual Information (MI) 기반 방법

**직관**: 변환이 맞으면 LiDAR 점을 이미지에 투영했을 때 점의 반사 강도와 그 자리의 이미지 밝기가 가장 강하게 함께 움직인다. 이 통계적 의존성(statistical dependency)이 가장 커지는 변환을 찾는다.

**Mutual Information 정의**:

두 확률 변수 $X$(LiDAR 반사 강도나 깊이)와 $Y$(이미지 밝기)의 상호 정보량은 다음과 같다.

$$
\text{MI}(X; Y) = \sum_{x}\sum_{y} p(x, y) \log \frac{p(x, y)}{p(x)p(y)}
$$

엔트로피로 쓰면 이렇다.

$$
\text{MI}(X; Y) = H(X) + H(Y) - H(X, Y)
$$

$X$와 $Y$가 독립이면 $\text{MI} = 0$이다. 상호 정보량은 두 변수 가운데 작은 쪽 엔트로피를 넘지 못하고, 두 변수가 일대일로 대응하면 $\text{MI} = H(X) = H(Y)$가 된다.

**Normalized Information Distance (NID)**:

[Koide et al. (2023)](https://arxiv.org/abs/2302.05094)은 MI를 정규화한 NID(normalized information distance)를 비용 함수로 쓴다.

$$
\text{NID}(X; Y) = 1 - \frac{\text{MI}(X; Y)}{H(X, Y)} = \frac{H(X, Y) - \text{MI}(X; Y)}{H(X, Y)}
$$

NID는 0과 1 사이 값이고, 작을수록 두 데이터가 잘 맞는다.

**히스토그램 기반 MI 계산**:

MI는 보통 결합 히스토그램(joint histogram)으로 추정한다.

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

히스토그램으로 센 MI는 해석적인 기울기가 없어서, Nelder-Mead 같은 기울기 없는(gradient-free) 최적화나 수치 기울기를 쓴다. [Pandey et al. (2015)](https://doi.org/10.1002/rob.21542)는 MI를 최대화하는 방식으로 LiDAR-카메라 외부 파라미터를 추정했다.

#### Edge Alignment 기반 방법

**직관**: LiDAR 포인트 클라우드에서 깊이가 갑자기 바뀌는 곳(depth discontinuity)을 에지로 뽑고, 이 에지가 이미지 에지에 겹치도록 변환을 조정한다.

LiDAR 깊이 이미지에서 추출한 에지를 $\mathbf{e}_L$, 카메라 이미지의 에지를 $\mathbf{e}_C$라 하면:

$$
\min_{\mathbf{R}, \mathbf{t}} \sum_i \text{dist}(\pi(\mathbf{R}\mathbf{p}_{L,i} + \mathbf{t}), \mathbf{e}_C)
$$

여기서 $\pi(\cdot)$는 카메라 투영 함수, $\text{dist}(\cdot, \mathbf{e}_C)$는 가장 가까운 에지까지의 거리(distance transform 사용)이다.

#### Learning-based 방법

RegNet, CalibNet 같은 딥러닝 방법은 LiDAR 깊이 이미지와 카메라 이미지를 받아 6-DoF 보정량을 바로 출력한다. 학습 때 본 센서, 장면, 어긋남 크기를 벗어나면 결과를 믿기 어렵다. 새 장비에 쓸 때는 가진 데이터로 먼저 검증하고, 필요하면 결과를 기하 최적화의 초기값으로만 쓴다.

### 3.3.3 Koide et al. (2023) — 공개 Targetless Calibration 구현

[Koide et al. (2023)](https://arxiv.org/abs/2302.05094)의 `direct_visual_lidar_calibration`은 타겟 없이 LiDAR-카메라 캘리브레이션을 하는 공개 도구다(ICRA 2023).

**1단계: LiDAR 포인트 클라우드 밀집화**

회전식 LiDAR(Ouster, Velodyne 등)는 스캔 한 장의 점이 성겨서 NID 정합에 쓰기 어렵다. 그래서 CT-ICP(Continuous-Time ICP)로 연속된 여러 스캔을 맞춰 겹쳐 쌓아 촘촘한 포인트 클라우드를 만든다. Livox Avia처럼 비반복 패턴으로 스캔하는 LiDAR는 점을 그냥 쌓기만 해도 촘촘해진다.

**2단계: SuperGlue 기반 초기 추정**

촘촘한 포인트 클라우드를 가상 카메라 시점에서 렌더링해 LiDAR 강도 이미지를 만든다. 이 이미지와 실제 카메라 이미지 사이의 2D-2D 대응점을 학습 기반 매처 SuperGlue로 찾는다. 렌더링 이미지의 픽셀마다 3D 점을 알고 있으므로 이 대응은 곧 2D-3D 대응이 되고, RANSAC과 재투영 오차 최소화로 초기 변환을 구한다.

논문은 이동 오차 0.5m, 회전 오차 1도 안쪽이면 초기 추정이 성공했다고 보았고, Ouster LiDAR와 핀홀 카메라 조합의 15개 데이터 가운데 12개에서 성공했다. 센서와 장면이 바뀌면 성공률도 달라진다.

**3단계: NID 기반 정밀 정합**

초기 추정에서 출발해 Nelder-Mead로 NID를 최소화한다. 그 전에 카메라 시점에서 가려지는 LiDAR 점을 view-based hidden point removal로 지운다. 변환이 바뀌면 이 제거와 NID 최소화를 다시 반복한다.

**논문 결과**: Ouster OS1-64와 핀홀 카메라 조합에서 평균 이동 오차는 0.043m, 회전 오차는 0.374도였다. 기준값은 고반사 구 타겟으로 구한 변환이고, 논문은 이를 유사 정답(pseudo ground truth)이라 부른다. 저자들은 Livox Avia, 전방향 카메라 조합도 실험했다.

### 3.3.4 실전 도구 비교

| 도구 | 방식 | 타겟 필요 | 실행 형태 | 적용 전 확인 |
|------|------|-----------|-----------|--------------|
| Autoware Calibration Toolkit | Target-based | O | 사용자 절차를 포함한 오프라인 도구 | 지원 센서와 타겟, 초기값 |
| `direct_visual_lidar_calibration` (Koide) | Targetless (NID) | X | 오프라인 자동화 파이프라인 | 장면 중첩, 강도 영상, 초기화 성공 여부 |
| ACSC (Automatic Calibration) | Target-based + auto corner | O | 타겟 검출 자동화 | 구현이 요구하는 타겟과 센서 모델 |
| 온라인 extrinsic 추정 (VIO/LIO 상태 증강) | Online | X | 주행 중 추정 | 운동의 관측 가능성과 드리프트 감시 기준 |

[MFCalib (2024)](https://arxiv.org/abs/2409.00992)은 깊이가 이어지는 에지, 깊이가 끊기는 에지, 평면 위의 강도 불연속 에지를 함께 쓴다. LiDAR 빔 폭 때문에 에지가 실제보다 두껍게 잡히는 현상(edge inflation)은 빔 모델로 보정하고, 자기 데이터셋에서 비교 방법보다 오차가 작았다고 보고했다.

---

## 3.4 Camera-IMU Extrinsic + Temporal Calibration

카메라와 IMU 사이의 공간적 변위(extrinsic)와 시간적 오프셋(temporal offset)을 함께 추정한다. VINS-Mono, OpenVINS 같은 VIO(Visual-Inertial Odometry) 시스템은 이 두 파라미터를 초기값으로 받아 운용 중 정제한다. Kalibr 같은 캘리브레이션 도구는 두 값을 오프라인으로 추정한다.

### 3.4.1 왜 시간 오프셋이 중요한가

카메라와 IMU는 각자의 클록으로 타임스탬프를 찍는다. 두 클록 사이에 시간 오프셋 $t_d$가 있으면, 카메라 타임스탬프 $t_c$에 찍힌 영상은 IMU 시간으로 $t_c + t_d$ 순간의 장면이다.

오프셋을 무시하면 빠르게 회전할 때 재투영 오차가 커진다. 100 deg/s로 회전하는 중에 오프셋이 10ms면 카메라 방향이 1도 어긋난다. Furgale et al. (2013)은 노출 시간을 바꿔 가며 오프셋을 추정했다. 추정값에 맞춘 직선의 기울기는 0.498로 이론값 0.5에 가까웠고, 모든 추정값이 이 직선에서 ±0.2ms 안에 들었다. 0.2ms는 IMU 샘플링 주기 5ms보다 훨씬 작다.

### 3.4.2 Kalibr: Continuous-Time B-Spline 기반 캘리브레이션

Kalibr는 [Furgale et al. (2013)](https://ieeexplore.ieee.org/document/6696514)의 방법을 구현한 카메라-IMU 캘리브레이션 도구다. VINS-Mono, OpenVINS 같은 VIO 시스템에 넣을 외부 파라미터와 시간 오프셋을 미리 구할 때 쓴다.

Kalibr는 궤적(trajectory)을 연속 시간 B-spline으로 표현한다. 그러면 카메라(20~30Hz)와 IMU(200~1000Hz)처럼 주기가 다른 센서의 관측을 같은 궤적의 각 시각에서 비교할 수 있다.

**B-Spline 궤적 표현**:

Furgale et al. (2013)은 IMU 포즈를 6차원 B-spline으로 표현한다. 이동 3성분과 회전 3성분(axis/angle 표현의 회전 벡터)을 각각 6차 B-spline(조각마다 5차 다항식)으로 두므로, 가속도가 3차 다항식으로 표현된다. 가속도계와 자이로 bias는 3차 B-spline으로 둔다. 아래 식은 Lovegrove et al. (2013)이 제안한 누적 3차(cubic) B-spline이다. 제어점(control points) $\{\mathbf{T}_i\}$ 사이의 상대 변환을 $SE(3)$ 위에서 누적해, 시각 $t$의 포즈 $\mathbf{T}(t) \in SE(3)$를 Lie 군 위에서 바로 보간한다.

$$
\mathbf{T}(t) = \mathbf{T}_i \prod_{j=1}^{3} \text{Exp}(\mathbf{B}_j(u) \cdot \Omega_{i+j})
$$

여기서:
- $u = (t - t_i) / \Delta t$는 정규화된 시간 ($0 \leq u < 1$)
- $\mathbf{B}_j(u)$는 3차 B-spline 기저 함수의 누적(cumulative) 계수
- $\Omega_{i+j} = \text{Log}(\mathbf{T}_{i+j-1}^{-1} \mathbf{T}_{i+j})$는 인접 제어점 간 상대 변환의 리 대수(Lie algebra) 표현
- $\text{Exp}, \text{Log}$는 $SE(3)$의 지수/로그 사상

B-spline을 쓰면 다음이 쉬워진다.
1. **미분 가능**: 임의 시간에서 속도, 가속도를 해석적으로 계산 가능 → IMU 관측 모델과 직접 연결
2. **비동기 센서 처리**: 각 센서의 타임스탬프에서 연속 궤적을 직접 평가
3. **국소성(locality)**: 각 시각의 궤적은 인접한 몇 개의 제어점(조각이 3차 다항식이면 4개, Furgale et al.처럼 5차 다항식이면 6개)에만 의존 → 희소(sparse) 최적화 가능

**관측 모델**:

카메라 관측: 오프셋을 보정한 시각 $t_c + t_d$의 궤적 포즈로 3D 랜드마크를 투영한다.

$$
\mathbf{e}_{\text{cam},k} = \mathbf{m}_k - \pi\left(\mathbf{T}_{CB} \cdot \mathbf{T}(t_{c,k} + t_d)^{-1} \cdot \mathbf{p}_w\right)
$$

여기서 $\mathbf{T}(t)$는 IMU(body)→월드 변환이며, $\mathbf{T}_{CB}$는 IMU(body)→카메라 외부 변환 (추정 대상)이다. §3.10 표의 $\mathbf{T}_{CI}$와 같은 변환이다.

IMU 관측: IMU 샘플 시각 $t_k$에서 궤적을 미분해 가속도와 각속도를 예측한다.

$$
\mathbf{e}_{\text{accel},k} = \mathbf{a}_k - \left[\mathbf{R}(t_k)^\top(\ddot{\mathbf{p}}(t_k) - \mathbf{g}) + \mathbf{b}_a\right]
$$
$$
\mathbf{e}_{\text{gyro},k} = \boldsymbol{\omega}_k - \left[\mathbf{R}(t_k)^\top\boldsymbol{\omega}_w(t_k) + \mathbf{b}_g\right]
$$

$\mathbf{R}(t)$와 $\mathbf{p}(t)$는 $\mathbf{T}(t)$의 회전과 이동이고, $\boldsymbol{\omega}_w(t)$는 월드 좌표로 나타낸 각속도다. 가속도와 각속도는 월드 좌표로 예측한 뒤 $\mathbf{R}^\top$으로 IMU 좌표에 옮겨 측정값과 비교한다.

**최적화 문제**:

$$
\min_{\mathbf{T}_{CB}, t_d, \mathbf{g}, \mathbf{b}_a(\cdot), \mathbf{b}_g(\cdot), \{\mathbf{T}_i\}} \sum_k \|\mathbf{e}_{\text{cam},k}\|^2_{\Sigma_c} + \sum_k \left(\|\mathbf{e}_{\text{accel},k}\|^2_{\Sigma_a} + \|\mathbf{e}_{\text{gyro},k}\|^2_{\Sigma_g}\right) + \int \left(\|\dot{\mathbf{b}}_a(\tau)\|^2_{Q_a} + \|\dot{\mathbf{b}}_g(\tau)\|^2_{Q_g}\right) d\tau
$$

마지막 적분 항은 bias를 random walk로 모델링한 데서 나오며, bias가 빠르게 변하지 못하게 묶는다. 중력 방향 $\mathbf{g}$도 함께 추정한다. Furgale et al. (2013)은 이 비선형 최소자승 문제를 Levenberg-Marquardt(LM)로 푼다. B-spline은 각 시각이 인접한 제어점 몇 개에만 걸리므로 정보 행렬이 희소하고, 희소 Cholesky 분해(CHOLMOD)로 매 반복의 선형 시스템을 푼다. 논문의 약 80초 데이터셋에서 반복 한 번에 선형 시스템을 만드는 데 약 18초, 푸는 데 약 0.2초가 걸렸다.

### 3.4.3 Kalibr 실행 가이드

```bash
# 1. AprilGrid 타겟 준비
kalibr_create_target_pdf --type apriltag \
    --nx 6 --ny 6 --tsize 0.024 --tspace 0.3

# 2. 데이터 수집 (ROS bag)
#    - 타겟을 카메라 시야에 놓고, 센서 리그를 다양하게 움직임
#    - 모든 축에서 회전 + 병진 운동을 포함
#    - 고정된 시간보다 모든 축의 운동, 타겟 검출, 잔차의 조건을 충족할 만큼 수집

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

# IMU 노이즈 파라미터 (Allan variance에서 측정, §3.4.4 참조)
accelerometer_noise_density: 0.01    # m/s^2/sqrt(Hz)
accelerometer_random_walk: 0.0002   # m/s^3/sqrt(Hz)
gyroscope_noise_density: 0.005      # rad/s/sqrt(Hz)
gyroscope_random_walk: 4.0e-06      # rad/s^2/sqrt(Hz)
```

**데이터 수집**:
1. **운동의 다양성**: 세 축 이동과 세 축 회전을 모두 충분히 일으켜야(excite) 한다. 특히 축마다 회전을 꼭 넣는다.
2. **운동 속도**: 너무 느리게 움직이면 IMU bias를 추정하기 어렵다. 너무 빠르게 움직이면 이미지가 흐려진다.
3. **타겟 가시성**: 카메라가 타겟을 지속적으로 검출하면서 각 축의 운동을 충분히 포함하도록 궤적을 설계한다.
4. **시작과 끝**: 정지한 상태로 시작하고 끝내면 IMU bias를 초기화하기 쉽다.

**결과 해석**: Kalibr 출력에서 다음을 본다.
- `T_cam_imu`: 카메라-IMU 외부 파라미터 (4x4 변환 행렬)
- `timeshift_cam_imu`: 이 데이터에서 추정된 시간 오프셋 $t_d$와 부호 규약
- 재투영 오차 분포: 전체 평균과 함께, 시간과 영상 위치에 따라 잔차가 쏠리는지, 이상치가 있는지 본다
- 가속도계·자이로 잔차: 설정한 노이즈 모델 크기와 맞는지 본다

### 3.4.4 Allan Variance 측정 실습

IMU 노이즈 파라미터가 틀리면 Kalibr 캘리브레이션뿐 아니라 IMU를 쓰는 시스템 전체의 성능이 떨어진다. Allan variance는 클러스터 시간(cluster time)을 바꿔 가며 시계열의 노이즈를 나누어 보는 방법이다. David Allan이 1966년에 원자 주파수 표준의 안정도를 재려고 만들었다.

**Allan Variance 정의**:

시계열 $\{x_k\}$를 길이 $\tau = n \cdot \tau_0$($\tau_0$는 샘플링 주기)의 구간으로 나누었을 때 Allan variance는 다음과 같다.

$$
\sigma^2(\tau) = \frac{1}{2(N-2n+1)} \sum_{k=1}^{N-2n+1} \left[\bar{x}_{k+n} - \bar{x}_k\right]^2
$$

여기서 $\bar{x}_k = \frac{1}{n}\sum_{j=0}^{n-1} x_{k+j}$는 클러스터 평균이다. 이 식은 클러스터를 한 샘플씩 밀며 겹쳐 잡는 overlapping Allan variance다. 아래 코드는 클러스터를 겹치지 않게 나눠 계산한다(non-overlapping Allan variance).

**로그-로그 그래프로 노이즈 구분하기**:

Allan deviation $\sigma(\tau)$를 $\tau$에 대해 로그-로그 축에 그리면 노이즈 원인마다 기울기가 다른 직선 구간이 나타난다.

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
    # 큰 tau에서도 통계량이 남도록 최소 10개 cluster를 확보한다.
    max_n = max(1, N // 10)

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

def extract_imu_noise_params(taus, allan_vars):
    """
    Allan variance 플롯에서 IMU 노이즈 파라미터를 추출.
    """
    adev = np.sqrt(allan_vars)

    valid = (taus > 0) & (adev > 0) & np.isfinite(adev)
    taus, adev = taus[valid], adev[valid]
    slopes = np.gradient(np.log(adev), np.log(taus))

    # -1/2 구간 전체에서 sigma(tau) = N / sqrt(tau)의 계수 N을 적합한다.
    white = np.abs(slopes + 0.5) < 0.1
    if not np.any(white):
        raise ValueError("No stable -1/2-slope region; collect or inspect more data")
    noise_coefficient = np.median(adev[white] * np.sqrt(taus[white]))

    # 0 기울기 구간이 있을 때만 bias-instability 계수를 보고한다.
    flat = np.abs(slopes) < 0.1
    bias_instability = np.min(adev[flat]) / 0.664 if np.any(flat) else np.nan
    tau_min = taus[np.argmin(adev)]

    return {
        'noise_coefficient': noise_coefficient,
        'bias_instability': bias_instability,
        'tau_min': tau_min
    }

# 사용 예시
# 1. 분석하려는 최대 cluster time에서 충분한 cluster가 남도록 정지 데이터를 녹화
# 2. 자이로/가속도 각 축의 raw 데이터를 분석

# 예시: 200Hz IMU 데이터, 2시간 분량
dt = 1.0 / 200  # 5ms
# gyro_x = np.loadtxt("imu_static_gyro_x.txt")  # 실제 데이터

# 시뮬레이션 데이터로 시연
np.random.seed(42)
N = 200 * 3600 * 2  # 2시간
noise_density = 0.005  # rad/s/sqrt(Hz)
bias = 0.0001  # bias random walk 세기 [rad/s/sqrt(s)]. 아래 누적합으로 bias가 천천히 떠돈다
gyro_x = noise_density * np.sqrt(1/dt) * np.random.randn(N)
gyro_x += bias * np.cumsum(np.random.randn(N)) * np.sqrt(dt)

taus, avars = compute_allan_variance(gyro_x, dt)
params = extract_imu_noise_params(taus, avars)

print(f"White-noise coefficient: {params['noise_coefficient']:.6f}")
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
plt.loglog(tau_ref, params['noise_coefficient'] / np.sqrt(tau_ref),
           'r--', label='Random walk (-1/2)')
plt.axhline(y=params['bias_instability'], color='g', linestyle='--',
            label=f'Bias instability ({params["bias_instability"]:.1e})')
plt.legend()
plt.savefig('allan_deviation.png', dpi=150)
plt.show()
```

**Allan Variance 측정 실전 팁**:
- **정지 데이터 수집**: IMU를 진동이 없는 단단한 표면 위에 놓는다. 기록 길이는 분석하려는 가장 긴 $\tau$에서 서로 겹치지 않는 cluster가 충분히 남도록 정한다. bias-instability 평탄부가 나타날 때까지 기록 시간을 늘린다.
- **데이터시트와 비교**: Allan variance에서 읽은 noise density를 제조사 데이터시트 값과 비교해 센서 상태를 확인한다.
- **Kalibr에 넣을 값**: 기울기 $-1/2$ 구간 전체에 직선을 맞춰 white-noise 계수를 구하고, 그 단위가 Kalibr가 받는 연속 시간 noise density 단위와 같은지 확인한다. $\tau=1$초에 가장 가까운 점 하나만 읽어 옮기면 기울기가 $-1/2$에서 벗어난 구간이나 단위 차이를 놓친다.
- **온도 안정화**: 전원을 켠 뒤 센서 온도와 출력 평균이 자리를 잡은 다음부터 기록하고, 온도도 함께 남긴다. 온도가 안정되기까지 걸리는 시간은 IMU 케이스와 바깥 공기에 얼마나 노출됐는지에 따라 다르다.

---

## 3.5 LiDAR-IMU Extrinsic Calibration

LiDAR와 IMU 사이의 외부 파라미터는 고전적인 hand-eye calibration 문제로 푼다.

### 3.5.1 Hand-Eye Calibration (AX = XB)

Hand-eye calibration은 로봇 팔 끝(hand)에 단 카메라(eye)와 팔 끝 사이의 변환을 구하는 문제에서 나왔다. 로봇 기저(base)에서 잰 팔 끝의 움직임과 카메라로 본 타겟의 움직임을 맞대어 이 변환을 푼다([Tsai & Lenz, 1989](https://ieeexplore.ieee.org/document/34770)).

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

이 식이 $\mathbf{AX} = \mathbf{XB}$이고, 미지수는 $\mathbf{X} \in SE(3)$다.

**회전과 이동의 분리**:

회전과 이동으로 나누어 쓰면 다음과 같다.

$$
\mathbf{R}_A \mathbf{R}_X = \mathbf{R}_X \mathbf{R}_B \quad \text{(회전)}
$$
$$
\mathbf{R}_A \mathbf{t}_X + \mathbf{t}_A = \mathbf{R}_X \mathbf{t}_B + \mathbf{t}_X \quad \text{(이동)}
$$

회전 식에는 $\mathbf{t}_X$가 없으므로 $\mathbf{R}_X$를 먼저 풀고, 그 값을 이동 식에 넣어 $\mathbf{t}_X$를 푼다.

**[Tsai & Lenz (1989)](https://ieeexplore.ieee.org/document/34770)의 해법**:

회전축과 회전각을 각각 $\hat{\mathbf{a}}_A, \alpha$와 $\hat{\mathbf{a}}_B, \beta$로 두고, $\mathbf{p}_A = 2\sin(\alpha/2)\hat{\mathbf{a}}_A$, $\mathbf{p}_B = 2\sin(\beta/2)\hat{\mathbf{a}}_B$를 사용하면:

$$
\text{skew}(\mathbf{p}_A + \mathbf{p}_B) \cdot \mathbf{r}_X = \mathbf{p}_B - \mathbf{p}_A
$$

여기서 $\mathbf{r}_X = \tan(\theta_X/2)\hat{\mathbf{a}}_X$는 $\mathbf{R}_X$의 로드리게스 벡터다. 운동 쌍마다 이 식을 쌓으면 선형 시스템 $\mathbf{C} \mathbf{r}_X = \mathbf{d}$가 되고, 회전축이 서로 나란하지 않은 운동 쌍이 둘 있으면 풀린다. 이동 벡터 $\mathbf{t}_X$도 비슷한 선형 시스템으로 푼다.

**운동 쌍이 많을 때**: 운동 쌍을 둘보다 많이 모으면 과결정 시스템을 최소자승으로 풀고, LM 최적화로 다듬는다.

### 3.5.2 Motion-based 자동 캘리브레이션

Hand-eye calibration을 쓰려면 두 센서가 각자 자기 움직임을 추정(odometry)할 수 있어야 한다. LiDAR odometry와 IMU 적분은 서로 따로 돌아가므로, 타겟 없이 센서를 이리저리 움직이며 데이터를 모으면 된다.

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

        # Scaled quaternion vector parts: 2 sin(angle / 2) * axis
        a_prime = 2 * np.sin(alpha / 2) * rA / alpha
        b_prime = 2 * np.sin(beta / 2) * rB / beta

        # skew(a' + b') * rX = b' - a'
        skew_sum = np.array([
            [0, -(a_prime[2]+b_prime[2]), a_prime[1]+b_prime[1]],
            [a_prime[2]+b_prime[2], 0, -(a_prime[0]+b_prime[0])],
            [-(a_prime[1]+b_prime[1]), a_prime[0]+b_prime[0], 0]
        ])

        C.append(skew_sum)
        d.append(b_prime - a_prime)

    C = np.vstack(C)
    d = np.concatenate(d)

    # 최소자승 풀이
    rX, _, _, _ = np.linalg.lstsq(C, d, rcond=None)

    # Rodrigues 벡터 → 회전 행렬
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

FAST-LIO2는 주행하면서 LiDAR-IMU extrinsic을 추정하는데, 좋은 초기값과 충분한 운동이 있어야 한다. [LI-Init](https://arxiv.org/abs/2202.11006)은 FAST-LIO2를 시작하기 전에 시간 오프셋, extrinsic, 중력 벡터, IMU bias를 실시간으로 추정해 넘겨주는 초기화 모듈이다.

1. LiDAR 측정만으로 상태를 추정하고, 이를 IMU 측정과 비교
2. 두 쪽이 맞도록 시간 오프셋과 외부 파라미터를 반복해서 다듬음
3. 결과를 FAST-LIO2에 넘김. 이후에는 FAST-LIO2의 IEKF(§4.2.5)가 상태 벡터에 든 LiDAR-IMU extrinsic을 갱신하며 온라인으로 다듬음

LIO를 켤 때 타겟이나 추가 센서 없이 외부 파라미터가 정해진다. 운동은 충분히 다양해야 한다. 공개 구현의 README는 실행 직후 5초 넘게 멈춰 있다가 움직이라고 하고, 초기화 뒤 FAST-LIO2로 15~30초 동안 extrinsic을 다듬으라고 권한다.

초기 운동이 부족하면 수렴하지 않거나 값이 부정확하게 나오므로, 정밀도가 중요하면 target-based 결과와 비교해 본다.

**GRIL-Calib**: 지상 로봇은 평면에서만 움직이므로, 모든 축의 운동을 가정한 기존 방법으로는 일부 축이 잘 추정되지 않는다. [GRIL-Calib (Kim et al., 2024)](https://arxiv.org/abs/2312.14035)은 LiDAR odometry에 지면 평면 잔차(ground plane residual)를 넣고, 지면 평면 운동(GPM) 제약을 최적화에 더해 평면 운동만으로 6-DoF 외부 파라미터를 추정한다(IEEE RA-L 2024).

---

## 3.6 LiDAR-LiDAR Extrinsic Calibration

LiDAR를 여러 대 단 시스템(multi-LiDAR rig)에서는 LiDAR 사이의 상대 포즈를 알아야 한다. 차량 앞, 옆, 뒤에 LiDAR를 나눠 단 자율주행 차량이 그런 경우다.

### 3.6.1 문제 설정

LiDAR $n$대 $\{L_1, L_2, \ldots, L_n\}$이 차체에 고정되어 있을 때, $L_1$을 기준으로 나머지 LiDAR의 상대 변환 $\{\mathbf{T}_{L_1 L_i}\}_{i=2}^{n}$를 추정한다.

### 3.6.2 Target-based 방법

여러 LiDAR가 한꺼번에 볼 수 있는 큰 평면 패널을 타겟으로 쓴다. LiDAR마다 타겟 평면을 RANSAC으로 맞추고, 같은 평면이 두 좌표계에서 어떻게 보이는지를 제약으로 삼아 상대 변환을 구한다.

### 3.6.3 ICP 기반 Targetless 방법

중첩 영역(overlapping region)이 있는 두 LiDAR는 포인트 클라우드를 ICP로 정합하는 targetless 방법을 쓴다.

**중첩 영역이 있는 경우**:

두 LiDAR의 FoV가 겹치면 그 영역의 포인트 클라우드를 ICP로 바로 맞춘다.

$$
\min_{\mathbf{R}, \mathbf{t}} \sum_{i} \|\mathbf{p}_{L_2,i} - (\mathbf{R} \mathbf{p}_{L_1,i'} + \mathbf{t})\|^2
$$

Point-to-plane ICP를 쓰면 보통 더 빨리 수렴한다.

$$
\min_{\mathbf{R}, \mathbf{t}} \sum_{i} \left[(\mathbf{R} \mathbf{p}_{L_1,i'} + \mathbf{t} - \mathbf{p}_{L_2,i})^\top \mathbf{n}_{L_2,i}\right]^2
$$

여기서 $\mathbf{n}_{L_2,i}$는 target 점의 법선 벡터이다.

**중첩 영역이 없는 경우**:

FoV가 겹치지 않으면 다음 방법을 쓴다.
1. Hand-eye calibration(§3.5.1): LiDAR마다 odometry로 상대 운동을 구해 AX=XB를 푼다.
2. SLAM 기반: LiDAR마다 따로 SLAM을 돌려 같은 장소의 지도를 만들고, 두 지도를 맞춰 상대 변환을 구한다.

### 3.6.4 Feature-based 방법

건물이나 도로처럼 평면과 기둥이 많은 곳에서는 평면, 기둥, 에지 같은 기하 특징을 뽑아 매칭한다. 평면 하나에 점 수백 개가 모이므로 점 하나하나의 노이즈가 평균되어 줄어든다.

- 여러 LiDAR에서 동일한 대형 평면(벽, 바닥)을 추출
- 평면 파라미터 $(n_i, d_i)$를 각 LiDAR 좌표계에서 계산
- 대응 평면 쌍으로부터 상대 변환 추정

법선 벡터가 비공면(non-coplanar)인 평면 대응이 최소 3개 필요하다.

---

## 3.7 GNSS-IMU Lever Arm & Boresight

GNSS/INS 통합 항법을 하려면 GNSS 안테나가 IMU에서 얼마나 떨어져 있는지 알아야 한다. 이 거리 벡터를 레버 암(lever arm)이라 한다.

### 3.7.1 Lever Arm Vector

GNSS 안테나의 위상 중심(phase center)과 IMU의 원점 사이의 3D 벡터 $\mathbf{l} = [l_x, l_y, l_z]^\top$를 IMU body frame에서 정의한다.

GNSS가 재는 위치는 안테나 위상 중심의 위치다. 필요한 것은 IMU나 차량 기준점의 위치이므로 다음 식으로 옮긴다.

$$
\mathbf{p}_{\text{IMU}} = \mathbf{p}_{\text{GNSS}} - \mathbf{R}_{\text{body}}^{\text{nav}} \cdot \mathbf{l}
$$

여기서 $\mathbf{R}_{\text{body}}^{\text{nav}}$는 body frame을 navigation frame으로 돌리는 회전 행렬이다. 차량이 돌면 안테나도 IMU 둘레로 함께 돌므로, lever arm은 매 순간의 자세로 회전시켜 더해야 한다. 회전축에 수직인 1m lever arm을 처음 자세 그대로 둔 채 차량이 10도 돌면 위치가 약 17cm($2\sin 5°$ m) 틀린다.

### 3.7.2 Lever Arm 추정 방법

**방법 1: 물리적 측정**

줄자나 레이저 거리 측정기로 직접 잴 수 있다. 필요한 측정 정밀도는 안테나와 IMU 사이 거리, 플랫폼의 회전 범위, 허용 위치 오차로 계산한다. 수 cm의 물리 측정 오차가 허용 오차 예산을 넘을 때는 정밀 측량이나 온라인 추정을 쓴다.

**방법 2: 필터 기반 온라인 추정**

EKF 상태 벡터에 lever arm $\mathbf{l}$을 넣어 주행 중에 추정한다. 이때 lever arm을 관측할 수 있으려면(observable) 차량이 충분히 회전해야 한다. 자세가 바뀌지 않는 직선 주행에서는 $\mathbf{R}\mathbf{l}$이 일정한 위치 편향과 구별되지 않아 세 성분 모두 추정되지 않는다. yaw 회전만 있으면 회전축과 나란한 수직 성분 $l_z$가 끝까지 추정되지 않는다.

**방법 3: 사후 처리(post-processing)**

모아 둔 GNSS/IMU 데이터를 앞뒤 양방향(forward-backward) 스무더로 처리하면서 lever arm을 함께 추정한다. 정밀 측량에서 주로 쓴다.

### 3.7.3 GNSS 안테나 위상 중심

GNSS 안테나의 위상 중심(Antenna Phase Center, APC)은 안테나의 기계적 중심에서 조금 떨어져 있고, 위성의 고도각과 주파수에 따라 움직인다. 이 변동을 Phase Center Variation(PCV)이라 하며, 크기는 mm에서 cm 수준이다.

RTK나 PPP 같은 정밀 측위에서는 안테나 위상 중심 보정 데이터(ANTEX 파일)를 적용한다. 보정할지는 안테나 모델, 기준국·이동국 처리 방식, 자세 변화, 전체 오차 예산을 보고 정한다.

---

## 3.8 Online / Continuous Calibration

온도 변화, 진동, 충격으로 센서 마운트가 조금씩 휘면 처음 잡은 캘리브레이션 값이 점점 틀어진다. 운용 중에 이 값을 다시 추정하는 일을 온라인 캘리브레이션(online calibration)이라 한다.

### 3.8.1 Self-Calibration during SLAM

SLAM 시스템의 상태 벡터에 캘리브레이션 파라미터를 넣어 운용 중에 함께 추정한다.

**EKF 기반**: OpenVINS는 카메라 intrinsic, camera-IMU extrinsic, 시간 오프셋을 상태 벡터에 넣어 주행 중에 추정한다.

상태 벡터 확장:
$$
\mathbf{x} = \begin{bmatrix} \mathbf{x}_{\text{nav}} \\ \mathbf{x}_{\text{calib}} \end{bmatrix}
= \begin{bmatrix} \mathbf{q}, \mathbf{p}, \mathbf{v}, \mathbf{b}_g, \mathbf{b}_a \\ \mathbf{q}_{CI}, \mathbf{p}_{CI}, t_d, f_x, f_y, c_x, c_y, k_1, \ldots \end{bmatrix}
$$

캘리브레이션 파라미터는 보통 random walk로 변한다고 가정한다.

$$
\dot{\mathbf{x}}_{\text{calib}} = \mathbf{w}_{\text{calib}}, \quad \mathbf{w}_{\text{calib}} \sim \mathcal{N}(\mathbf{0}, \mathbf{Q}_{\text{calib}})
$$

$\mathbf{Q}_{\text{calib}}$는 캘리브레이션 파라미터가 얼마나 빨리 변한다고 볼지를 정한다. 너무 크게 잡으면 추정값이 흔들리고, 너무 작게 잡으면 실제로 값이 바뀌었을 때 늦게 따라간다.

**Factor graph 기반**: VINS-Mono는 camera-IMU 외부 파라미터를 sliding window 최적화의 변수 하나로 두고 다른 상태와 함께 추정한다. 외부 파라미터가 시간에 따라 바뀐다고 보는 factor graph에서는 시각마다 값을 따로 두고 이웃한 값을 between factor로 잇는다. 이 factor의 노이즈를 작게 잡을수록 추정값이 천천히 변한다.

### 3.8.2 Extrinsic Drift 보정

자율주행 차량이나 로봇을 오래 운용하면 진동과 충격으로 외부 파라미터가 조금씩 틀어진다. 이를 찾아 고치는 방법은 다음과 같다.

1. **투영 정합 감시**: LiDAR 포인트 클라우드를 이미지에 투영해 에지가 얼마나 잘 겹치는지 계속 본다. 겹침이 나빠지면 캘리브레이션을 다시 한다.

2. **주기적 재캘리브레이션**: 운용 중에 모은 데이터로 정기적으로 targetless 캘리브레이션을 다시 한다.

3. **온라인 미세 조정**: 지금 값을 초기값으로 두고 좁은 범위 안에서 계속 최적화한다.

[CalibRefine (2025)](https://arxiv.org/abs/2502.17648)은 가공하지 않은 LiDAR 포인트 클라우드와 카메라 이미지를 바로 받아 딥러닝으로 타겟 없이 온라인 캘리브레이션을 한다. 처음 얻은 값은 반복 후처리(iterative post-refinement)로 다시 다듬는다.

### 3.8.3 OpenCalib: 자율주행 통합 캘리브레이션 프레임워크

[OpenCalib (Yan et al., 2022)](https://arxiv.org/abs/2205.14087)은 자율주행 차량에 필요한 여러 캘리브레이션을 한데 모은 오픈소스 도구다.

**지원하는 캘리브레이션 유형** (논문 3절 기준):

| 센서 쌍 | 방법 | 타겟 |
|---------|------|------|
| Camera intrinsic | Zhang's method, calibration harp로 왜곡 평가 | 체커보드 |
| Camera-LiDAR | 수동 정렬 / 원형 구멍 보드로 intrinsic·extrinsic 동시 추정 / 차선·기둥 선 특징 자동 정렬 | 없음 / 있음 / 없음 |
| LiDAR-LiDAR | 수동 정렬 / 지면 정렬 + NICP + octree 최적화 | 없음 |
| LiDAR-IMU | BALM 기반 local map 최적화 | 없음 |
| Radar-Camera, Radar-LiDAR | 수동 정렬 | 없음 |
| Camera-Ground | vanishing point와 지면 homography | 공장 보드 또는 차선 |
| 온라인 보정 | Camera-IMU, LiDAR-IMU (hand-eye 방식), Radar yaw | 없음 |

캘리브레이션 종류마다 따로 모듈로 되어 있어서 필요한 것만 골라 쓴다.

차량에 카메라 6대와 LiDAR 1대가 있으면 카메라마다 LiDAR와 따로 캘리브레이션한다. 이렇게 얻은 결과를 이어 붙여 계산한 카메라 사이의 상대 포즈가 카메라끼리 직접 잰 값과 맞아야 한다. OpenCalib은 센서 하나나 한 쌍씩 다루므로, 이 일관성은 결과를 합성해 따로 확인한다.

---

## 3.9 Temporal Calibration

빠르게 움직이는 플랫폼에서는 수 밀리초의 시간 오차도 센티미터 단위 위치 오차가 된다. 100km/h로 달리는 차에서 1ms는 약 2.8cm다.

### 3.9.1 Hardware Synchronization

센서들이 같은 clock과 trigger를 받도록 배선할 수 있으면 하드웨어 동기화를 먼저 쓴다. 이때도 장치마다 latency가 얼마인지, timestamp가 노출 시작·끝·중간 가운데 언제를 가리키는지 따로 확인한다.

**Trigger 기반 동기화**:

마스터 타이머 하나가 모든 센서에 트리거 신호를 보내 같은 순간에 측정하게 한다.

- **카메라 트리거**: 외부 트리거 입력(GPIO 핀)으로 노출 시작 시각을 맞춘다
- **LiDAR 동기**: Ouster 같은 일부 LiDAR는 PPS(Pulse Per Second) 입력을 받아 내부 시계와 스캔 시각을 맞춘다
- **IMU 동기**: IMU 샘플링 클록을 외부 기준 신호에 묶는다

**PPS (Pulse Per Second)**:

GNSS 수신기는 GNSS 시간에 맞춘 1Hz 전기 펄스(PPS)를 내보낸다. 펄스의 상승 에지가 GNSS 시간의 정각 초에 오도록 수신기를 설정하면, 이 에지를 모든 장치의 공통 기준으로 쓴다. 수신기 설정, 케이블과 입력 회로의 지연, 장치마다 timestamp가 가리키는 순간은 따로 확인한다.

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

PPS 동기화에서는 센서마다 로컬 타임스탬프와 PPS 펄스 사이의 오프셋을 재서 모든 데이터를 공통 시간축(GPS time)에 맞춘다.

### 3.9.2 PTP (Precision Time Protocol)

IEEE 1588 PTP는 이더넷으로 장치들의 클록을 맞추는 프로토콜이다. 장치가 하드웨어 타임스탬프를 찍고 스위치도 PTP를 지원하면 오차를 1마이크로초 아래로 줄일 수 있다. 실제 정확도는 장비와 네트워크 구성에 따라 다르다.

**PTP 동작 원리**:

1. **마스터-슬레이브 구조**: 네트워크의 마스터 클록 하나(Grandmaster)가 기준 시간을 내보낸다
2. **Sync 메시지**: 마스터가 Sync 메시지를 주기적으로 보낸다
3. **Follow-up**: Sync를 실제로 보낸 정확한 시각을 다음 메시지로 알린다
4. **Delay Request/Response**: 슬레이브가 요청을 보내고 응답을 받아 네트워크 지연을 잰다
5. **오프셋 계산**: 경로 지연이 양방향으로 같다고 보고 $\text{offset} = \frac{(t_2 - t_1) - (t_4 - t_3)}{2}$로 계산한다

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

일부 LiDAR와 산업용 카메라는 PTP와 하드웨어 타임스탬핑을 지원한다. 지원 여부와 timestamp 기준점은 모델별 데이터시트에서 확인해야 한다.

### 3.9.3 Software Synchronization

하드웨어 동기화를 할 수 없으면 소프트웨어로 시간을 맞춘다.

**Host Clock 기반**: 센서 데이터가 호스트 컴퓨터에 도착한 순간 시스템 시계로 타임스탬프를 찍는다. 이 값에는 USB, 네트워크, 드라이버 queue에서 생긴 지연과 그 흔들림(jitter)이 섞인다. 크기는 장치와 부하에 따라 다르므로 직접 재 봐야 한다.

**ROS Time**: ROS(Robot Operating System)에서는 `ros::Time::now()`로 메시지를 받은 시각을 기록한다. 드라이버가 센서 자체 타임스탬프를 ROS 시간으로 바꿔 넣는다면, 이 변환이 얼마나 정확한지가 곧 동기화 정밀도다.

### 3.9.4 Time Offset 온라인 추정

[Li & Mourikis (2014)](https://doi.org/10.1177/0278364913515286)는 시간 오프셋 $t_d$를 VIO 상태 벡터에 넣어 주행 중에 추정했다.

**관측 모델에 시간 오프셋 넣기**:

카메라 타임스탬프 $t_c$에 찍힌 영상의 포즈는 보정한 시각 $t_c + t_d$에서 구한다. IMU preintegration에도 이 시각을 쓴다.

$$
\mathbf{z}(t_c) = \pi(\mathbf{T}(t_c + t_d) \cdot \mathbf{p}_w)
$$

$t_d$가 작다고 보고 1차 테일러 전개하면 다음과 같다.

$$
\mathbf{T}(t_c + t_d) \approx \mathbf{T}(t_c) \cdot \text{Exp}(\boldsymbol{\xi} \cdot t_d)
$$

여기서 $\boldsymbol{\xi}$는 시각 $t_c$의 body 속도(각속도와 선속도)다.

이 근사 덕분에 $t_d$에 대한 자코비안이 닫힌 식으로 나오고, EKF나 최적화에서 다른 상태 변수와 함께 $t_d$를 추정한다.

[iKalibr (Chen et al., IEEE T-RO 2025)](https://arxiv.org/abs/2407.11420)는 IMU에 LiDAR, 카메라, radar를 여러 대 묶은 시스템의 공간·시간 파라미터를 타겟 없이 B-spline 연속 시간 궤적으로 한 번에 추정한다.

**관측 가능성(Observability) 조건**: 시간 오프셋을 추정하려면 플랫폼이 충분히 가속하고 회전해야 한다. 등속 직선 운동에서는 시간을 조금 미는 것과 위치를 조금 옮기는 것이 같은 결과를 내서 시간 오프셋을 구할 수 없다.

OpenVINS, VINS-Mono 같은 VIO 시스템이 이 방법의 변형을 구현한다. 오프라인 도구인 Kalibr(§3.4.2)는 연속시간 B-spline 배치 최적화로 같은 값을 따로 추정한다.

### 3.9.5 실전 동기화 전략 가이드

| 구성 조건 | 우선 검토할 방법 | 반드시 확인할 항목 |
|-----------|------------------|---------------------|
| 공통 트리거 입력과 clock을 배선할 수 있음 | 하드웨어 트리거 + 공통 clock/PPS | 노출·샘플·스캔 중 timestamp 기준 시점, 고정 latency |
| 모든 장치와 스위치가 PTP 및 하드웨어 timestamp를 지원 | PTP | grandmaster 선택, PTP profile, 경로 비대칭, lock 상태 |
| 장치 clock은 읽을 수 있으나 공통 신호가 없음 | clock 변환 + 오프라인/온라인 offset 추정 | clock drift, 운동의 관측 가능성, offset의 시간 변화 |
| 수신 시각만 사용할 수 있음 | host timestamp를 임시 사용하고 jitter 측정 | USB·네트워크 queue 지연; 요구 오차를 만족하지 못하면 장비 구성 변경 |

**PPS와 PTP 함께 쓰기**: GNSS 수신기의 PPS로 PTP grandmaster를 GNSS 시간에 맞추고, 나머지 장치는 PTP로 grandmaster를 따르게 한다.

**연구용 시제품**: 요구 오차를 만족한다면 소프트웨어 동기화에 온라인 시간 오프셋 추정을 더하는 것으로 충분하다. Kalibr로 초기 오프셋을 구하고, VIO가 주행 중에 조금씩 고친다.

---

## 3.10 챕터 요약

| 캘리브레이션 유형 | 추정 파라미터 | 방법 | 최소 요구 조건 |
|----------------|-------------|----------|-------------|
| Camera intrinsic | $\mathbf{K}, \mathbf{d}$ | Zhang's method | 시야 전역을 포괄하는 다양한 타겟 포즈 |
| Stereo extrinsic | $\mathbf{R}, \mathbf{t}$ | 공유 타겟 + stereoCalibrate | 두 카메라가 동시에 검출한 다양한 포즈 |
| Camera-LiDAR | $\mathbf{T}_{CL}$ | Target-based / NID | 여러 시점의 타겟 관측 / 자연 장면 |
| Camera-IMU | $\mathbf{T}_{CI}, t_d$ | Kalibr (B-spline) | 충분한 시간의 다양한 운동 |
| LiDAR-IMU | $\mathbf{T}_{LI}$ | Hand-eye (AX=XB) / LI-Init | 다양한 운동 |
| LiDAR-LiDAR | $\mathbf{T}_{L_1 L_2}$ | ICP / Feature matching | 중첩 영역 또는 운동 데이터 |
| GNSS-IMU | $\mathbf{l}$ (lever arm) | 물리적 측정 / EKF 추정 | 여러 축의 회전 운동 |
| Temporal | $t_d$ | PPS/PTP + 온라인 추정 | 가속·회전 운동 |

**캘리브레이션 순서 권장**:

1. 각 카메라의 intrinsic (카메라마다 따로 구함)
2. 스테레오 extrinsic (있는 경우)
3. Camera-IMU extrinsic + temporal (Kalibr)
4. LiDAR-IMU extrinsic (hand-eye 또는 LI-Init)
5. Camera-LiDAR extrinsic (camera-IMU와 LiDAR-IMU의 체인으로 간접 계산하거나 직접 캘리브레이션)
6. GNSS lever arm
7. 온라인 캘리브레이션 설정 (운용 중 보정)

5번의 간접 계산은 $\mathbf{T}_{CL} = \mathbf{T}_{CI} \cdot \mathbf{T}_{IL}$이다. 두 단계의 오차가 쌓이므로 직접 캘리브레이션한 값과 비교해 본다.

**관련 논문**:

- [Zhang (2000)](https://ieeexplore.ieee.org/document/888718): 카메라 intrinsic — 호모그래피 기반의 유연한 캘리브레이션
- [Furgale et al. (2013)](https://ieeexplore.ieee.org/document/6696514): Camera-IMU — B-spline 연속 시간 궤적으로 시공간 동시 캘리브레이션
- [Tsai & Lenz (1989)](https://ieeexplore.ieee.org/document/34770): Hand-eye — AX=XB의 초기 해법
- [Koide et al. (2023)](https://arxiv.org/abs/2302.05094): Camera-LiDAR targetless — NID + SuperGlue 기반 자동 캘리브레이션
- [Li & Mourikis (2014)](https://doi.org/10.1177/0278364913515286): Temporal — 시간 오프셋 온라인 추정
- [OpenCalib (2022)](https://arxiv.org/abs/2205.14087): 자율주행 통합 캘리브레이션 프레임워크
- [GRIL-Calib (Kim et al., 2024)](https://arxiv.org/abs/2312.14035): 지면 평면 운동 제약을 쓰는 지상 로봇용 targetless IMU-LiDAR 캘리브레이션. 평면 운동만으로 6-DoF 추정.
- [MFCalib (2024)](https://arxiv.org/abs/2409.00992): 깊이 연속·불연속 에지와 강도 불연속 에지를 함께 쓰는 single-shot targetless LiDAR-카메라 캘리브레이션. LiDAR 빔 모델로 edge inflation을 보정.
- [iKalibr (Chen et al., IEEE T-RO 2025)](https://arxiv.org/abs/2407.11420): Temporal — B-spline 연속 시간 기반 다중 이종 센서(LiDAR·카메라·IMU·radar) 통합 시공간 캘리브레이션.
