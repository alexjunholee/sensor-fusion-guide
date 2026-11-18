# 3장 캘리브레이션

관측 모델을 실제 센서 데이터에 적용하려면 모델 속 미지 파라미터부터 확정해야 한다(→ 2장). 카메라의 초점 거리와 왜곡 계수 같은 센서 고유 값뿐 아니라, LiDAR와 IMU 사이의 상대 위치·자세, 그리고 서로 다른 센서 시계 사이의 시간 오프셋까지 실측으로 구해 내는 과정이 캘리브레이션(calibration)이다.

융합 결과가 어딘가 어긋날 때 가장 먼저 점검해야 할 대상도 캘리브레이션이다. 센서 내부 모델이나 센서 간 변환이 조금만 틀어져도 추정값 전체에 일정한 방향의 편향(bias)이 깔리고, 조건에 따라서는 필터나 최적화가 아예 발산하기도 한다. 가령 LiDAR-카메라 융합에서 장착 각도가 단 1도만 틀어져도 50m 앞 물체는 투영 위치가 옆으로 약 87cm나 밀려난다($50 \tan 1° \approx 0.87$m). 개별 센서의 내부 모델이 흔들리면 센서 쌍을 아무리 정밀하게 맞춰도 오차가 남으므로, 캘리브레이션은 단일 카메라의 내부 파라미터를 잡는 일에서 출발해 센서 간 외부 파라미터와 시간 동기화로 넓혀 가는 순서가 자연스럽다.

---

## 3.1 카메라 내부 파라미터

카메라 내부 캘리브레이션(intrinsic calibration)은 3D 공간의 점이 2D 이미지의 어느 픽셀에 찍힐지 결정하는 센서 고유 파라미터를 추정하는 작업이다. 초점 거리(focal length), 주점(principal point), 렌즈 왜곡 계수(distortion coefficients)를 정확히 알아야 이후의 스테레오 깊이 복원이나 LiDAR·IMU와의 외부 정합도 제대로 성립한다.

### 3.1.1 핀홀 카메라 모델 복습

3D 점은 먼저 카메라 좌표계 기준의 3D 위치로 옮겨진 뒤, 투영과 픽셀 스케일링을 거쳐 2D 이미지 평면 위에 자리 잡는다. 세계 좌표 $\mathbf{P}_w = [X, Y, Z]^\top$를 카메라 좌표계의 $\mathbf{P}_c = [X_c, Y_c, Z_c]^\top$로 변환했다고 할 때, 핀홀 모델의 이미지 투영 관계는 다음 행렬식으로 나타낸다.

$$
s \begin{bmatrix} u \\ v \\ 1 \end{bmatrix}
= \mathbf{K} \begin{bmatrix} X_c \\ Y_c \\ Z_c \end{bmatrix}
= \begin{bmatrix} f_x & \gamma & c_x \\ 0 & f_y & c_y \\ 0 & 0 & 1 \end{bmatrix} \begin{bmatrix} X_c \\ Y_c \\ Z_c \end{bmatrix}
$$

각 기호의 물리적 의미는 다음과 같다.
- $f_x, f_y$: 픽셀 단위의 초점 거리(focal length)로, 물리적 초점 거리 $f$(mm)를 센서 픽셀 크기 $p_x$(mm/pixel)로 나눈 값이다($f_x = f / p_x$).
- $(c_x, c_y)$: 광축(optical axis)이 이미지 센서 평면과 만나는 주점(principal point)의 픽셀 좌표다.
- $\gamma$: 픽셀 격자의 기울어짐을 나타내는 비대칭 계수(skew coefficient)로, 현대 디지털 카메라에서는 사실상 0에 가깝다.
- $s = Z_c$: 동차 좌표를 2D 픽셀 좌표로 되돌릴 때 나누어 주는 스케일 팩터, 즉 점의 깊이값이다.

행렬 $\mathbf{K}$를 카메라 내부 행렬(intrinsic matrix) 또는 캘리브레이션 행렬이라 부른다. 이론상 자유도는 $f_x, f_y, c_x, c_y, \gamma$의 다섯이지만, 실무에서는 센서 픽셀이 직교 격자라고 보고 $\gamma = 0$으로 고정한 채 나머지 네 파라미터만 추정하는 것이 보통이다.

### 3.1.2 렌즈 왜곡 모델

실제 카메라 렌즈는 이상적인 핀홀 모델처럼 직선을 곧게 투영하지 못한다. 렌즈 왜곡은 픽셀 단위 격자가 아니라 광축을 기준으로 한 입사각과 중심에서의 반경에 따라 물리적으로 발생하므로, 먼저 깊이 $Z_c$로 나누어 초점 거리와 주점의 영향을 덜어 낸 **정규화된 좌표**(normalized coordinates)부터 정의한다.

$$
x = X_c / Z_c, \quad y = Y_c / Z_c, \quad r^2 = x^2 + y^2
$$

왜곡은 크게 광축 중심에서 멀어질수록 배율이 변하는 **방사 왜곡**(radial distortion)과 렌즈 조립 오차에서 비롯되는 **접선 왜곡**(tangential distortion)으로 나뉜다. 먼저 방사 왜곡은 반경 $r$의 짝수 거듭제곱 다항식으로 모델링한다.

$$
x_{\text{radial}} = x(1 + k_1 r^2 + k_2 r^4 + k_3 r^6)
$$
$$
y_{\text{radial}} = y(1 + k_1 r^2 + k_2 r^4 + k_3 r^6)
$$

계수 $k_1 < 0$이면 이미지 중심에서 멀어질수록 배율이 줄어 직선이 바깥으로 불룩하게 휘는 배럴(barrel) 왜곡이 나타나고, 반대로 $k_1 > 0$이면 배율이 커져 직선이 중심 쪽으로 오목하게 휘는 핀쿠션(pincushion) 왜곡이 생긴다. 일반적인 표준 화각 렌즈에서는 4차 항까지의 계수 $k_1, k_2$만 써도 방사 왜곡의 대부분이 충분히 설명된다.

한편 접선 왜곡은 렌즈 요소가 이미지 센서 평면과 완벽히 평행하게 조립되지 않았을 때 발생하며, 다음 식으로 보정한다.

$$
x_{\text{tangential}} = 2p_1 xy + p_2(r^2 + 2x^2)
$$
$$
y_{\text{tangential}} = p_1(r^2 + 2y^2) + 2p_2 xy
$$

두 성분을 합쳐 왜곡이 반영된 정규화 좌표를 만든 뒤, 마지막으로 내부 행렬의 초점 거리와 주점을 적용해야 최종 왜곡 픽셀 좌표 $(u, v)$가 나온다.

$$
x_d = x_{\text{radial}} + x_{\text{tangential}}, \quad y_d = y_{\text{radial}} + y_{\text{tangential}}
$$

$$
u = f_x \cdot x_d + c_x, \quad v = f_y \cdot y_d + c_y
$$

이 왜곡 계수들은 관례상 $\mathbf{d} = [k_1, k_2, p_1, p_2, k_3]$ 순서의 벡터로 묶어 다루며, OpenCV의 `calibrateCamera()` 역시 같은 순서로 결과를 반환한다.

### 3.1.3 Zhang's Method: 호모그래피 기반 캘리브레이션

과거에는 3차원 위치가 정밀하게 가공된 입체 캘리브레이션 장치(3D calibration rig)가 있어야 카메라 내부 파라미터를 풀 수 있었지만, [Zhang (2000)](https://ieeexplore.ieee.org/document/888718)이 제안한 방법은 종이에 인쇄한 평면 패턴(체커보드) 하나를 여러 각도와 거리에서 찍는 것만으로 충분하다. 제작과 촬영이 간편해 오늘날 사실상의 표준으로 자리 잡았으며, OpenCV의 `calibrateCamera()`도 바로 이 알고리즘을 따른다.

핵심 아이디어는 타겟을 평면 하나에 가두는 데 있다. 패턴이 놓인 평면을 세계 좌표계의 $Z = 0$ 평면으로 정하면 3D-2D 투영식이 평면 간 변환인 호모그래피(homography) 하나로 단순해지고, 여러 장의 호모그래피가 공유하는 회전 행렬의 성질을 모으면 외부 포즈를 소거한 채 $\mathbf{K}$만 따로 풀어낼 수 있다.

#### Step 1: 호모그래피 추출

체커보드가 $Z = 0$ 평면에 놓여 있으므로, 보드 위의 점 $\mathbf{M} = [X, Y, 0]^\top$과 이미지 좌표 $\mathbf{m} = [u, v]^\top$의 관계를 동차(homogeneous) 좌표로 적으면 $Z = 0$과 곱해지는 $\mathbf{r}_3$ 열이 자연스럽게 사라진다.

$$
s \tilde{\mathbf{m}} = \mathbf{K} [\mathbf{r}_1 \quad \mathbf{r}_2 \quad \mathbf{r}_3 \quad \mathbf{t}] \begin{bmatrix} X \\ Y \\ 0 \\ 1 \end{bmatrix}
= \mathbf{K} [\mathbf{r}_1 \quad \mathbf{r}_2 \quad \mathbf{t}] \begin{bmatrix} X \\ Y \\ 1 \end{bmatrix}
$$

따라서 보드 평면과 이미지 평면 사이는 $3 \times 3$ 호모그래피 행렬 $\mathbf{H}$ 하나로 연결된다.

$$
s \tilde{\mathbf{m}} = \mathbf{H} \tilde{\mathbf{M}}, \quad \mathbf{H} = \mathbf{K} [\mathbf{r}_1 \quad \mathbf{r}_2 \quad \mathbf{t}]
$$

$\mathbf{H}$는 스케일을 제외하면 8자유도를 가지므로 이론상 대응점 4쌍만 있어도 DLT(Direct Linear Transform)로 구할 수 있고, 실제로는 체커보드 코너 수십 개가 한꺼번에 검출되므로 식이 미지수보다 훨씬 많은 과결정(over-determined) 시스템을 SVD로 풀게 된다.

**DLT로 호모그래피 추정**: 대응점 $(\mathbf{M}_j, \mathbf{m}_j)$에서는 투영 벡터 $\mathbf{H}\tilde{\mathbf{M}}_j$와 이미지 동차 벡터 $\tilde{\mathbf{m}}_j$가 같은 방향을 가리키므로 외적 $\tilde{\mathbf{m}}_j \times \mathbf{H} \tilde{\mathbf{M}}_j = \mathbf{0}$이 성립한다. 이를 선형 방정식으로 정리하면 다음 형태가 된다.

$$
\begin{bmatrix}
\tilde{\mathbf{M}}_j^\top & \mathbf{0}^\top & -u_j \tilde{\mathbf{M}}_j^\top \\
\mathbf{0}^\top & \tilde{\mathbf{M}}_j^\top & -v_j \tilde{\mathbf{M}}_j^\top
\end{bmatrix}
\mathbf{h} = \mathbf{0}
$$

여기서 $\mathbf{h}$는 $\mathbf{H}$의 9개 원소를 펼친 벡터다. $n$개의 대응점을 쌓아 $2n \times 9$ 행렬 $\mathbf{A}$를 만든 뒤, $\|\mathbf{A}\mathbf{h}\|$를 최소화하는 단위 벡터 $\mathbf{h}$를 $\mathbf{A}$의 SVD에서 가장 작은 특잇값(singular value)에 대응하는 오른쪽 특이 벡터(right singular vector)로 구한다.

#### Step 2: 내부 파라미터의 제약 조건 도출

추정한 호모그래피 한 장에는 내부 행렬 $\mathbf{K}$와 그 시점의 외부 자세 $\mathbf{r}_1, \mathbf{r}_2, \mathbf{t}$가 뒤섞여 있다. $\mathbf{H} = [\mathbf{h}_1 \quad \mathbf{h}_2 \quad \mathbf{h}_3]$로 열을 나누어 적으면 관계가 드러난다.

$$
[\mathbf{h}_1 \quad \mathbf{h}_2 \quad \mathbf{h}_3] = \lambda \mathbf{K} [\mathbf{r}_1 \quad \mathbf{r}_2 \quad \mathbf{t}]
$$

여기서 $\lambda$는 임의의 스케일 팩터다. $\mathbf{r}_1, \mathbf{r}_2$는 회전 행렬의 처음 두 열이므로 서로 직교하고 길이가 같아야 하며, 이 두 기하학적 성질이 $\mathbf{K}$에 대한 제약을 만든다.

1. **직교 조건**: $\mathbf{r}_1^\top \mathbf{r}_2 = 0$으로부터
   $$\mathbf{h}_1^\top \mathbf{K}^{-\top} \mathbf{K}^{-1} \mathbf{h}_2 = 0$$

2. **동일 노름 조건**: $\|\mathbf{r}_1\| = \|\mathbf{r}_2\|$로부터
   $$\mathbf{h}_1^\top \mathbf{K}^{-\top} \mathbf{K}^{-1} \mathbf{h}_1 = \mathbf{h}_2^\top \mathbf{K}^{-\top} \mathbf{K}^{-1} \mathbf{h}_2$$

두 식 모두 $\mathbf{B} = \mathbf{K}^{-\top} \mathbf{K}^{-1}$라는 행렬을 가운데에 품고 있다. $\mathbf{B}$는 대칭 양의 정부호(positive definite) 행렬이므로 독립 원소가 여섯 개뿐이며, 다음과 같이 쓸 수 있다.

$$
\mathbf{B} = \begin{bmatrix} B_{11} & B_{12} & B_{13} \\ B_{12} & B_{22} & B_{23} \\ B_{13} & B_{23} & B_{33} \end{bmatrix}
$$

이를 6차원 벡터 $\mathbf{b} = [B_{11}, B_{12}, B_{22}, B_{13}, B_{23}, B_{33}]^\top$로 묶으면 $\mathbf{b}$에 대한 선형 식을 세울 수 있다.

#### Step 3: 선형 시스템 구성

행렬 곱 $\mathbf{h}_i^\top \mathbf{B} \mathbf{h}_j$를 전개하면 $\mathbf{b}$와 호모그래피 성분으로 만든 벡터 $\mathbf{v}_{ij}$의 내적으로 깔끔하게 분리된다.

$$
\mathbf{h}_i^\top \mathbf{B} \mathbf{h}_j = \mathbf{v}_{ij}^\top \mathbf{b}
$$

여기서 벡터 $\mathbf{v}_{ij}$는 다음과 같이 정의한다.
$$
\mathbf{v}_{ij} = \begin{bmatrix} h_{1i}h_{1j} \\ h_{1i}h_{2j} + h_{2i}h_{1j} \\ h_{2i}h_{2j} \\ h_{3i}h_{1j} + h_{1i}h_{3j} \\ h_{3i}h_{2j} + h_{2i}h_{3j} \\ h_{3i}h_{3j} \end{bmatrix}
$$

직교 조건과 동일 노름 조건을 이 표기로 옮기면 이미지 한 장당 두 개의 선형 방정식이 나온다.

$$
\begin{bmatrix} \mathbf{v}_{12}^\top \\ (\mathbf{v}_{11} - \mathbf{v}_{22})^\top \end{bmatrix} \mathbf{b} = \mathbf{0}
$$

$n$장의 이미지를 촬영했다면 이 행을 쌓아 $2n \times 6$ 동차 선형 시스템을 얻고, 마찬가지로 SVD로 $\mathbf{b}$를 구한다.

**최소 이미지 수**: $\gamma = 0$으로 고정하면 $B_{12} = 0$이라는 제약이 추가된다. 그러면 남은 동차 성분 다섯 개는 스케일 자유도를 빼고 4자유도를 가지므로, 서로 퇴화하지 않은 자세의 이미지 두 장만 있어도 식이 풀린다. 반면 다섯 파라미터를 모두 추정하려면 서로 다른 평면 자세가 최소 세 장 필요하다. 물론 이 숫자는 선형 방정식이 풀리는 이론상 최솟값일 뿐이므로, 실제 촬영에서는 보드가 시야 곳곳에 다양한 거리와 기울기로 충분히 잡혔는지, 그리고 추정 문제의 조건수(condition number)가 안정적인지를 보고 장수를 결정한다.

#### Step 4: K 복원

$\mathbf{b}$를 구해 $\mathbf{B}$ 행렬을 조립했다면, Cholesky 분해 $\mathbf{B} = \mathbf{L}\mathbf{L}^\top$를 이용해 $\mathbf{K}^{-1} = \mathbf{L}^\top$로부터 $\mathbf{K}$를 얻거나 아래의 닫힌 식(closed-form)으로 각 성분을 곧장 계산한다.

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

$\mathbf{K}$가 정해지면 앞서 분리해 두었던 호모그래피 열로부터 각 이미지 $i$의 외부 자세를 바로 되살릴 수 있다.

$$
\mathbf{r}_1 = \lambda \mathbf{K}^{-1} \mathbf{h}_1, \quad \mathbf{r}_2 = \lambda \mathbf{K}^{-1} \mathbf{h}_2, \quad \mathbf{t} = \lambda \mathbf{K}^{-1} \mathbf{h}_3
$$

여기서 스케일은 $\lambda = 1 / \|\mathbf{K}^{-1} \mathbf{h}_1\|$이고, 세 번째 열은 $\mathbf{r}_3 = \mathbf{r}_1 \times \mathbf{r}_2$로 놓는다. 다만 측정 노이즈 때문에 이렇게 구한 $[\mathbf{r}_1, \mathbf{r}_2, \mathbf{r}_3]$는 엄밀한 직교 회전 행렬 조건에서 조금 벗어나게 되므로, SVD를 써서 가장 가까운 회전 행렬 $\mathbf{R} = \mathbf{U}\mathbf{V}^\top$로 투영한다($\det(\mathbf{R}) = 1$이 되도록 보정).

#### Step 6: 비선형 최적화 (MLE)

지금까지의 선형 해는 렌즈 왜곡을 무시하고 대수적 오차(algebraic error)만 줄인 근사치이므로 비선형 최적화의 초기값으로만 쓴다. 마지막 단계에서는 왜곡 계수 $\mathbf{d}$까지 모두 포함해, 실제 검출된 코너 위치와 모델이 예측한 투영 위치 사이의 기하학적 거리인 **재투영 오차**(reprojection error)를 Levenberg-Marquardt(LM) 알고리즘으로 최소화한다.

$$
\min_{\mathbf{K}, \mathbf{d}, \{\mathbf{R}_i, \mathbf{t}_i\}} \sum_{i=1}^{n}\sum_{j=1}^{m} \|\mathbf{m}_{ij} - \hat{\mathbf{m}}(\mathbf{K}, \mathbf{d}, \mathbf{R}_i, \mathbf{t}_i, \mathbf{M}_j)\|^2
$$

여기서 $n$은 이미지 수, $m$은 각 이미지의 코너 수, $\hat{\mathbf{m}}(\cdot)$은 왜곡을 포함한 전체 투영 함수이며, 최적화에 필요한 자코비안은 투영 함수를 각 파라미터로 편미분해 계산한다. OpenCV의 `calibrateCamera()`는 DLT 호모그래피 추정, 선형 $\mathbf{K}$ 복원, 그리고 LM 비선형 최적화를 한 번의 호출 안에 묶어 제공한다.

### 3.1.4 OpenCV를 이용한 캘리브레이션 코드

Zhang's method의 전체 파이프라인을 파이썬과 OpenCV로 실행하면 어떤 흐름으로 코드가 이어지는지 살펴보자. 아래 스크립트는 체커보드 3D 기준 좌표를 만든 뒤, 이미지마다 코너를 검출해 서브픽셀 단위로 정제하고, `cv2.calibrateCamera()`로 내부 행렬과 왜곡 계수를 추정한 다음 이미지별 재투영 오차를 계산해 품질을 점검한다.

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

코드를 돌려 보면 전체 RMS 오차가 작더라도 특정 이미지 몇 장에서 오차가 유독 크게 튀는 일을 흔히 겪는다. 캘리브레이션의 성패는 알고리즘 호출 자체보다 어떤 조건에서 데이터를 모으고 걸러 냈느냐에 훨씬 크게 좌우된다.

### 3.1.5 실전 팁: 좋은 캘리브레이션을 얻는 조건

아래 다섯 가지 원칙은 내부 파라미터와 외부 자세, 왜곡 계수가 서로 뒤섞이지 않고 제각각 안정적으로 분리되도록 관측을 설계하고 결과를 검증하는 실전 기준이다.

**자세 다양성 (Pose Diversity)**: 체커보드를 다양한 각도와 위치에서 촬영해야 한다.
- 이미지의 중앙뿐 아니라 상하좌우와 네 모서리까지 보드를 배치해 시야 전역의 렌즈 왜곡을 관측한다.
- 정면에 가까운 자세만 반복하면 초점 거리와 보드까지의 거리가 서로 상쇄되므로, 여러 축으로 보드를 충분히 기울여 초점 거리와 외부 파라미터가 분리되게 만든다.
- 보드를 가까이도, 멀리도 두어 화면에 잡히는 크기를 다양하게 바꾼다.
- 단순히 검출에 성공한 장수만 늘리기보다, 포즈가 얼마나 다양하게 퍼져 있는지와 재투영 잔차가 영상 위치에 따라 치우치지 않는지를 보고 추가 촬영 여부를 판단한다.

**코너 정확도**: 사용 중인 검출기가 서브픽셀 정제를 자체적으로 수행하지 않는다면 `cv2.cornerSubPix()` 같은 함수로 코너 좌표를 다듬어야 한다. 코너 검출이 어긋났거나 모션 블러·반사로 보드가 흐려진 이미지는 미련 없이 버린다.

**조명 조건**: 조명이 고르게 퍼져야 코너 경계의 대비가 일정하게 잡힌다. 강한 그림자나 빛 반사(glare)가 끼면 코너 위치가 한쪽으로 밀리므로 무광 재질의 타겟을 사용하고, 촬영 직후 이미지를 눈으로 확인하는 편이 안전하다.

**재투영 오차 해석**: 적정 오차의 기준은 이미지 해상도, 렌즈 모델, 타겟 크기, 검출기 정밀도에 따라 모두 달라서 단일 수치 하나로 합격·불합격을 나눌 수 없다. 전체 RMS 오차만 보지 말고 이미지별 잔차, 영상 위치별 잔차의 편향 여부, 왜곡을 편 영상에서 실제 직선이 곧게 펴졌는지, 그리고 학습에 쓰지 않은 별도 검증 영상의 재투영 오차까지 함께 확인해야 한다.

**이상치 이미지 찾기**: 위 코드에서 계산한 `errors` 리스트를 비교하면 오차가 유난히 큰 이미지를 바로 집어낼 수 있다. 먼저 검출 실패, 모션 블러, 빛 반사, 휘어진 타겟이 원인인지 확인한 뒤, 그 이미지를 제외하기 전후의 파라미터 변화량과 검증 영상 오차를 비교해 최종 제거 여부를 결정한다.

**경고 — 정사각형 크기**: 코드의 `SQUARE_SIZE`는 인쇄된 체커보드 한 칸의 실제 길이와 정확히 일치해야 한다. 프린터 설정에 따라 도면이 미세하게 축소·확대 출력되는 일이 흔하므로 인쇄 후 반드시 자로 실측한다. 이 값이 틀리면 `tvecs`(이동 벡터)의 미터 단위 스케일이 그 비율만큼 통째로 어긋난다. 반면 왜곡 계수는 정규화 좌표에서, `K`는 픽셀 단위로 정의되므로 영향을 받지 않는다.

지금까지 살펴본 핀홀과 표준 radial-tangential 모델은 일반 화각 렌즈에서는 잘 맞지만, 화각이 크게 넓어지면 투영 기하 자체가 달라져 다른 모델이 필요해진다.

### 3.1.6 Fisheye / Omnidirectional 캘리브레이션

광각·어안(fisheye) 렌즈는 입사각 $\theta$가 $90^\circ$에 가까워질수록 정규화 좌표 $r = \tan\theta$가 급격히 커지므로, 표준 핀홀 + radial-tangential 다항식으로는 시야 가장자리의 왜곡을 감당하기 어렵다. 시야각(FoV) 몇 도부터 모델을 바꿔야 한다는 절대적인 경계는 없으나, 사용하는 렌즈의 투영 특성과 시야 외곽에서 잔차가 한 방향으로 벌어지는 패턴을 보고 equidistant나 Kannala–Brandt 같은 광각 모델을 선택한다.

**등거리 투영 모델 (Equidistant Projection Model)**:

[Kannala & Brandt (2006)](https://ieeexplore.ieee.org/document/1642666)가 제안한 일반 카메라 모델은 $\tan\theta$ 대신 입사각(incidence angle) $\theta$ 자체와 이미지 중심에서 투영점까지의 거리 $r$을 홀수 차수 다항식으로 연결한다.

$$
r(\theta) = k_1 \theta + k_2 \theta^3 + k_3 \theta^5 + k_4 \theta^7 + k_5 \theta^9
$$

여기서 $\theta = \arctan\left(\sqrt{x^2 + y^2}\right)$는 입사각이고 $x, y$는 정규화 좌표다. 이상적인 등거리 투영이 $r = f\theta$를 따르는 데 비해, 다항식의 고차 항이 실제 어안 렌즈에서 나타나는 편차를 흡수한다.

OpenCV의 `cv2.fisheye` 모듈은 이 식을 아래와 같이 조금 변형해 구현해 두었다.

$$
\theta_d = \theta(1 + k_1\theta^2 + k_2\theta^4 + k_3\theta^6 + k_4\theta^8)
$$

실제 코드로 돌릴 때는 일반 `calibrateCamera()`와 달리 입력 점 배열을 `(N, 1, 3)` 및 `(N, 1, 2)` 형상의 `float64`로 맞추어야 하며, `CALIB_CHECK_COND` 플래그가 조건수 불량을 엄격하게 검사해 예외를 발생시킬 수 있다는 점에 유의한다.

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

초광각 어안 렌즈뿐만 아니라 거울과 렌즈를 결합한 반사굴절(catadioptric) 카메라처럼 시야각이 $180^\circ$를 넘나드는 전방향 카메라에는 [Scaramuzza et al. (2006)](https://rpg.ifi.uzh.ch/docs/IROS06_scaramuzza.pdf)의 [OCamCalib toolbox](https://sites.google.com/site/scarabotix/ocamcalib-omnidirectional-camera-calibration-toolbox-for-matlab)가 널리 쓰인다. 투영 함수 자체를 입사각의 다항식 $\rho(\theta)$로 두기 때문에 광학계 구조가 달라도 하나의 수식으로 다룰 수 있다.

$$
\begin{bmatrix} u \\ v \end{bmatrix} = \begin{bmatrix} x_c \\ y_c \end{bmatrix} + \mathbf{A} \cdot \rho(\theta) \begin{bmatrix} \cos(\phi) \\ \sin(\phi) \end{bmatrix}
$$

여기서 $\mathbf{A}$는 센서 축의 늘어남과 비정방형 픽셀(non-square pixel)을 보정하는 affine 변환 행렬이고, $\rho(\theta)$는 입사각에 따른 이미지 반경이다.

OpenCV나 OCamCalib이 주로 단일 카메라의 내부 모델에 집중한다면, 여러 카메라를 한꺼번에 묶거나 뒤이어 다룰 IMU와 연결할 때는 Kalibr가 실무의 표준 도구로 쓰인다.

### 3.1.7 Kalibr를 이용한 카메라 캘리브레이션

[Kalibr](https://github.com/ethz-asl/kalibr)는 ETH Zurich의 ASL(Autonomous Systems Lab)이 공개한 캘리브레이션 도구로, 카메라-IMU 캘리브레이션(§3.4)을 비롯한 다중 센서 일괄 최적화에 널리 쓰인다. 여러 광각 카메라 모델과 batch 최적화를 지원하므로 카메라 내부 파라미터 역시 이 도구로 함께 추정할 수 있다. 다만 OpenCV보다 정밀한지 여부는 동일한 카메라 모델, 동일한 타겟 관측 데이터, 동일한 초기값을 넣고 직접 돌려 비교해 보아야 가릴 수 있다.

먼저 Kalibr가 지원하는 주요 카메라 모델과 파라미터 구성을 정리하면 다음과 같다.

| 모델 | 파라미터 수 | 적합한 렌즈 |
|------|-----------|-----------|
| `pinhole-radtan` | 4 + 4 | 일반 렌즈 (OpenCV와 동일) |
| `pinhole-equi` | 4 + 4 | 어안 렌즈 |
| `omni-radtan` | 5 + 4 | 초광각/catadioptric |
| `ds` (Double Sphere) | 6 | 광각 렌즈 |
| `eucm` (Extended UCM) | 6 | 광각 렌즈 |

**AprilGrid vs Checkerboard**: Kalibr는 일반 체커보드보다 AprilGrid 타겟을 권장한다. AprilGrid는 격자 속 태그마다 고유 ID가 새겨져 있어 보드 일부가 화면 밖으로 잘리거나 가려져도(partial occlusion) 남은 코너를 검출할 수 있고, 각 코너의 보드 내 인덱스도 자동으로 식별된다.

실제 작업은 타겟 PDF를 생성해 촬영한 ROS bag 파일을 입력으로 넘기는 두 단계의 명령으로 실행한다.

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

명령을 실행하면 Kalibr는 내부적으로 네 단계를 거쳐 해를 구한다.
1. 각 프레임에서 AprilGrid 코너를 검출한다.
2. 코너 대응으로부터 호모그래피를 추정해 초기값을 잡는다(Zhang's method의 Step 1과 유사).
3. 선택한 카메라 모델의 모든 파라미터를 batch 최적화로 동시에 추정한다.
4. 최적화 결과와 잔차(residuals) 분포를 보고서와 그래프로 시각화한다.

OpenCV의 기본 카메라 캘리브레이션과 비교했을 때 Kalibr가 지닌 강점은 다음과 같다.
- **B-spline 궤적 표현**: camera-IMU 캘리브레이션에서 연속 시간 궤적으로 비동기 센서 관측을 자연스럽게 처리한다.
- **확장된 카메라 모델**: double sphere(DS), EUCM처럼 광각 렌즈에 유리한 모델을 기본 제공한다.
- **멀티카메라 지원**: 여러 대의 카메라 내부 파라미터와 상대 포즈를 한 번에 추정한다.
- **IMU 연동**: camera-IMU 시공간 캘리브레이션(§3.4)과 같은 포맷으로 바로 이어진다.

카메라 한 대의 내부 모델을 확정했다면, 이제 두 카메라를 나란히 짝지어 3차원 깊이를 복원하기 위한 스테레오 외부 캘리브레이션으로 넘어갈 차례다.

---

## 3.2 Camera-Camera (Stereo) Extrinsic

두 대의 카메라로 장면을 찍어 3차원 깊이를 복원하려면, 각 카메라의 내부 파라미터를 따로 구해 두는 것만으로는 부족하다. 삼각측량이 성립하려면 왼쪽 카메라와 오른쪽 카메라가 공간상에서 서로 어떤 각도와 간격으로 떨어져 있는지, 즉 두 카메라 사이의 상대 포즈(extrinsic)를 정확히 알아야 한다.

### 3.2.1 스테레오 캘리브레이션

왼쪽 카메라 $C_L$과 오른쪽 카메라 $C_R$이 같은 3D 점 $\mathbf{P}$를 바라볼 때, 두 카메라 좌표계에서의 3D 좌표는 강체 변환 $(\mathbf{R}, \mathbf{t})$로 연결된다.

$$
\mathbf{P}_{C_R} = \mathbf{R} \cdot \mathbf{P}_{C_L} + \mathbf{t}
$$

여기서 $(\mathbf{R}, \mathbf{t})$는 왼쪽 카메라 기준 좌표를 오른쪽 카메라 기준 좌표로 옮기는 회전과 이동이다. 이 값을 구하려면 두 카메라가 같은 체커보드를 동시에 바라보도록 동기화해 여러 자세로 촬영한다.

OpenCV의 `cv2.stereoCalibrate()`는 이 공동 관측 데이터로부터 $(\mathbf{R}, \mathbf{t})$를 추정하며, 필요에 따라 두 카메라의 내부 파라미터를 고정하거나 함께 미세 조정한다. 이미 3.1절의 단일 카메라 캘리브레이션으로 내부 파라미터를 충분히 많은 자세에서 정밀하게 구해 두었다면, 아래 코드처럼 `CALIB_FIX_INTRINSIC` 플래그를 켜서 공유 프레임 수가 상대적으로 적은 스테레오 단계에서 초점 거리나 왜곡 계수가 헛되이 흔들리지 않도록 묶어 두는 편이 안정적이다.

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

외부 파라미터 $(\mathbf{R}, \mathbf{t})$를 구했다면, 실제 깊이 계산에 앞서 두 카메라 영상을 나란히 펴는 평행 정렬(rectification)을 거친다. 두 카메라가 미세하게 틀어져 있으면 대응점이 이미지 안에서 비스듬한 직선을 따라 놓이지만, 정렬을 마쳐 epipolar line이 모두 수평선과 일치하게 만들면 같은 행($y$ 좌표) 위에서만 짝을 찾는 1차원 탐색으로 스테레오 매칭 속도와 안정성이 크게 올라간다.

**Epipolar Geometry 복습**:

두 카메라의 광학 중심 $O_L, O_R$과 3D 점 $\mathbf{P}$가 만드는 평면을 epipolar plane이라 부르며, 이 평면이 왼쪽·오른쪽 이미지 평면과 만나 그리는 직선이 epipolar line이다. 따라서 왼쪽 이미지의 점 $\mathbf{m}_L$에 대응하는 오른쪽 이미지의 점 $\mathbf{m}_R$은 반드시 $\mathbf{m}_L$이 정해 주는 epipolar line 위에 놓이게 되며, 이 기하학적 제약은 기본 행렬(Fundamental matrix) $\mathbf{F}$로 표현된다.

$$
\tilde{\mathbf{m}}_R^\top \mathbf{F} \tilde{\mathbf{m}}_L = 0
$$

내부 파라미터 $\mathbf{K}$를 이미 알고 있을 때는 픽셀 좌표에서 내부 행렬을 걷어 낸 정규화 좌표 $\hat{\mathbf{m}} = \mathbf{K}^{-1}\tilde{\mathbf{m}}$를 써서, 외부 파라미터만으로 깔끔하게 분해되는 본질 행렬(Essential matrix) $\mathbf{E}$로 적을 수 있다.

$$
\hat{\mathbf{m}}_R^\top \mathbf{E} \hat{\mathbf{m}}_L = 0, \quad \mathbf{E} = [\mathbf{t}]_\times \mathbf{R}
$$

여기서 $[\mathbf{t}]_\times$는 이동 벡터 $\mathbf{t}$의 외적을 행렬 곱으로 나타낸 반대칭 행렬(skew-symmetric matrix)이다.

**Rectification 과정**:

Rectification은 두 카메라의 이미지 평면을 두 광학 중심을 잇는 기선(baseline)과 나란한 하나의 가상 평면 위로 회전·재투영하는 호모그래피를 구하는 과정이다. 이렇게 두 영상을 같은 평면에 나란히 올려놓으면 모든 epipolar line이 이미지의 수평선과 평행해진다. OpenCV의 `cv2.stereoRectify()`는 Bouguet 알고리즘을 써서 좌우 회전량을 절반씩 나누어 왜곡을 줄이면서 이 변환 행렬과 투영 행렬을 계산하며, 이어서 `initUndistortRectifyMap()`과 `remap()`으로 왜곡 보정과 평행 정렬을 한 번의 리샘플링으로 처리한다.

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

**Epipolar 제약으로 검증하기**: 정렬이 제대로 되었는지는 수치만 보는 것보다 두 정렬 영상을 가로로 이어 붙이고 수평선을 여러 줄 그어 눈으로 확인하면 확실히 드러난다. 왼쪽과 오른쪽에서 같은 물체의 모서리나 코너가 정확히 같은 색 수평선 위에 걸린다면 rectification이 제대로 들어맞았다고 볼 수 있다.

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

**디스패리티에서 깊이로**: 정렬을 마친 스테레오 이미지에서는 좌우 대응점이 같은 수평선에 놓이므로, 두 점의 가로 좌표 차이인 디스패리티(disparity) $d = u_L - u_R$만으로 3차원 깊이 $Z$를 곧바로 계산한다.

$$
Z = \frac{f \cdot B}{d}
$$

여기서 $f$는 픽셀 단위 초점 거리, $B = \|\mathbf{t}\|$는 기선 길이(baseline)다. 실무에서는 `stereoRectify()`가 함께 반환하는 $4 \times 4$ 재투영 행렬 $Q$를 이용해, $[X,Y,Z,W]^\top = Q \cdot [u,v,d,1]^\top$로 동차 좌표를 계산한 뒤 $W$로 나누어 픽셀별 3D 포인트 $\mathbf{P}_{3D} = [X/W,Y/W,Z/W]^\top$를 일괄 복원한다.

스테레오처럼 같은 종류의 카메라 두 대를 맞출 때는 양쪽 영상에서 동일한 시각 코너를 바로 짝지을 수 있다. 반면 카메라의 2D 영상 정보와 LiDAR의 3D 점군을 직접 겹치려면 전혀 다른 형태의 관측치를 공통 기하 공간에 묶어야 한다.

---

## 3.3 Camera-LiDAR Extrinsic Calibration

LiDAR 포인트 클라우드를 카메라 이미지에 투영해 색을 입히거나 이미지 속 객체의 3차원 거리를 읽어 내려면, 두 센서 좌표계를 잇는 외부 파라미터 $(\mathbf{R}, \mathbf{t})$를 정밀하게 구해야 한다. 두 센서는 데이터를 기록하는 방식이 2D 픽셀과 3D 점군으로 완전히 다르기 때문에, 양쪽이 동시에 뚜렷하게 포착할 수 있는 인공 타겟을 세우거나 자연 장면 속 통계적·기하학적 단서를 활용하는 전략으로 나뉜다.

### 3.3.1 Target-based 캘리브레이션

Target-based 방식은 카메라와 LiDAR가 체커보드나 AprilTag처럼 크기와 형태를 미리 아는 기하학적 타겟을 동시에 관측해 대응 관계를 만드는 가장 확실한 방법이다.

**원리**: 카메라는 타겟에서 체커보드 코너의 2D 픽셀 좌표를 추출하고, LiDAR는 타겟 표면에 반사된 3D 점군으로부터 보드가 놓인 3D 평면을 구한다. 보드 평면에 속한 점들을 분리해 낸 뒤, 평면의 법선과 경계 정보를 어떻게 활용하느냐에 따라 두 가지 풀이가 쓰인다.

**3D-2D 대응 기반 방법**: 보드 경계가 LiDAR 점군에서도 선명하게 보일 때는 3D-2D 코너 쌍을 직접 만들어 푼다.
1. 카메라 이미지에서 체커보드 코너를 검출해 2D 점 $\{\mathbf{m}_j\}$를 얻는다.
2. LiDAR 포인트 클라우드에서 RANSAC으로 체커보드 평면을 피팅해 평면 방정식 $\mathbf{n}^\top \mathbf{p} + d = 0$을 구한다.
3. 피팅된 평면 위의 점군에서 보드 외곽 경계를 찾아내 코너의 3D 좌표 $\{\mathbf{P}_j\}$를 추정한다.
4. 확보한 3D-2D 대응 $\{(\mathbf{P}_j, \mathbf{m}_j)\}$에 PnP(Perspective-n-Point) 알고리즘을 적용해 $(\mathbf{R}, \mathbf{t})$를 구한다.

**평면 제약 기반 방법**: 채널 수가 적은 희소한 LiDAR에서는 보드 모서리나 코너의 3D 좌표를 정밀하게 짚기 어렵다. 이때는 코너 위치 대신 보드 평면 자체를 제약 조건으로 삼아도 캘리브레이션이 가능하다. 카메라에서 검출한 코너와 이미 아는 체커보드 크기로부터 PnP를 풀어 카메라 좌표계 기준의 보드 평면을 구해 두고, LiDAR 점군을 미지의 외부 변환 $(\mathbf{R}, \mathbf{t})$로 카메라 좌표계에 옮겼을 때 그 평면 위에 올라앉도록 방정식을 세우는 방식이다.

$n$개의 서로 다른 체커보드 자세에서 얻은 평면 제약은 다음과 같다.

$$
\mathbf{n}_i^\top (\mathbf{R} \mathbf{p}_{L,i} + \mathbf{t}) + d_i = 0, \quad \forall i = 1, \ldots, n
$$

여기서 $\mathbf{n}_i$와 $d_i$는 카메라 좌표계로 나타낸 $i$번째 보드 평면의 법선과 원점 거리이며, $\mathbf{p}_{L,i}$는 LiDAR 좌표계에서 그 보드 평면 위에 놓인 점이다.

아래 함수는 바로 이 평면 제약 기반 풀이를 구현한 예시다. 각 포즈에서 `solvePnP`로 카메라 기준 보드 평면 $(\mathbf{n}_c, d_c)$를 얻고, LiDAR 평면 법선과 짝지어 Kabsch/SVD로 회전 $\mathbf{R}$을 먼저 구한 다음, 평면 간 거리 차이 방정식을 최소자승으로 풀어 이동 벡터 $\mathbf{t}$의 초기해를 얻는다.

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
- 체커보드 크기는 측정 거리와 LiDAR의 각도 해상도, 빔 패턴을 고려해 한 장의 스캔에서도 평면을 안정적으로 피팅할 만큼 충분한 점이 떨어지도록 정한다.
- 거리와 입사각을 달리한 관측을 모으되, 단순 관측 횟수보다 법선 벡터가 세 방향으로 고르게 퍼져 최적화의 조건수가 나쁘지 않은지를 우선 점검한다.
- 빔이 희소한 LiDAR에서는 보드에 맞는 점 개수가 예상보다 적을 수 있으므로 실제 점군 분포를 확인해 보며 타겟 크기와 거리를 조절한다.

### 3.3.2 Targetless 캘리브레이션

타겟 기반 캘리브레이션은 기준이 분명해 정밀도가 높지만, 운용 현장에 나갈 때마다 커다란 체커보드를 들고 다니며 여러 각도로 세워 찍기는 번거롭고 때로는 불가능하기도 하다. Targetless 캘리브레이션은 별도의 인공 타겟 없이 자연스러운 주행 장면만으로 센서 사이의 변환을 추정한다.

#### Mutual Information (MI) 기반 방법

**직관**: 외부 변환이 정확하게 들어맞으면, LiDAR 점을 카메라 이미지에 투영했을 때 그 점의 반사 강도(intensity)와 이미지 해당 위치의 밝기가 가장 일관된 통계적 관계를 맺는다. 가령 같은 아스팔트 노면이나 건물 외벽 위에서는 두 값이 나란히 변하고, 변환이 어긋나면 전혀 다른 물체끼리 겹치면서 관계가 흐려진다. 결국 두 확률 변수 사이의 통계적 의존성(statistical dependency)이 최대가 되는 변환을 찾는 문제로 바뀐다.

**Mutual Information 정의**:

두 확률 변수 $X$(LiDAR 반사 강도나 깊이)와 $Y$(투영 위치의 이미지 밝기)의 상호 정보량(Mutual Information)은 결합 분포와 주변 분포의 차이로 정의한다.

$$
\text{MI}(X; Y) = \sum_{x}\sum_{y} p(x, y) \log \frac{p(x, y)}{p(x)p(y)}
$$

이를 엔트로피로 다시 쓰면 직관적인 의미가 더 잘 드러난다.

$$
\text{MI}(X; Y) = H(X) + H(Y) - H(X, Y)
$$

$X$와 $Y$가 아무 상관 없이 독립이면 $\text{MI} = 0$이 된다. 반대로 상호 정보량은 두 변수 가운데 작은 쪽 엔트로피를 넘지 못하며, 두 변수가 일대일로 대응할 때 $\text{MI} = H(X) = H(Y)$로 최댓값에 이른다.

**Normalized Information Distance (NID)**:

원래의 MI 값은 장면에 따라 상한과 범위가 달라지므로, [Koide et al. (2023)](https://arxiv.org/abs/2302.05094)은 결합 엔트로피로 정규화한 NID(normalized information distance)를 비용 함수로 채택한다.

$$
\text{NID}(X; Y) = 1 - \frac{\text{MI}(X; Y)}{H(X, Y)} = \frac{H(X, Y) - \text{MI}(X; Y)}{H(X, Y)}
$$

NID는 항상 0과 1 사이의 값을 가지며, 두 센서의 정합이 정확해질수록 0에 가까워진다.

**히스토그램 기반 MI 계산**:

실제 구현에서는 연속 확률 분포 대신 이산적인 결합 히스토그램(joint histogram)을 쌓아 MI와 NID를 근사한다. 아래 함수는 투영된 LiDAR 반사 강도와 그 자리의 이미지 밝기를 2차원 히스토그램으로 집계해 엔트로피와 NID를 계산하는 기본 절차를 보여 준다.

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

이처럼 히스토그램으로 구간을 나누어 센 MI는 변환 파라미터에 대한 해석적인 기울기(gradient)가 존재하지 않는다. 따라서 최적화를 수행할 때는 Nelder-Mead처럼 기울기를 쓰지 않는(gradient-free) 탐색 기법이나 수치 기울기를 활용한다. [Pandey et al. (2015)](https://doi.org/10.1002/rob.21542)은 바로 이 MI 최대화 원리로 LiDAR-카메라 외부 파라미터를 추정하는 체계를 제시했다.

#### Edge Alignment 기반 방법

**직관**: 반사 강도 대신 기하학적 윤곽선을 맞추는 방법도 널리 쓰인다. LiDAR 포인트 클라우드에서 전경 물체와 배경 사이처럼 깊이가 급격히 바뀌는 경계(depth discontinuity)를 에지로 뽑아낸 뒤, 이 점들을 이미지에 투영했을 때 카메라 이미지의 밝기 에지와 나란히 포개지도록 변환을 조정한다.

LiDAR 깊이 이미지에서 추출한 에지를 $\mathbf{e}_L$, 카메라 이미지의 에지를 $\mathbf{e}_C$라 하면 목적 함수는 다음과 같다.

$$
\min_{\mathbf{R}, \mathbf{t}} \sum_i \text{dist}(\pi(\mathbf{R}\mathbf{p}_{L,i} + \mathbf{t}), \mathbf{e}_C)
$$

여기서 $\pi(\cdot)$는 카메라 투영 함수이고, $\text{dist}(\cdot, \mathbf{e}_C)$는 투영된 픽셀에서 가장 가까운 이미지 에지까지의 거리(미리 계산한 distance transform 맵 참조)를 뜻한다.

#### Learning-based 방법

RegNet, CalibNet 같은 딥러닝 기반 방법은 LiDAR 깊이 이미지와 카메라 이미지를 입력받아 6-DoF 보정량을 한 번에 회귀해 낸다. 다만 학습 데이터에 포함된 센서 모델, 주행 환경, 초기 어긋남 범위를 벗어나면 예측 결과를 그대로 신뢰하기 어렵다. 새로운 장비 조합에 적용할 때는 실제 보유 데이터로 성능을 먼저 검증하고, 필요한 경우 네트워크의 출력을 기하 최적화의 초기값으로만 활용하는 편이 안전하다.

### 3.3.3 Koide et al. (2023) — 공개 Targetless Calibration 구현

실무에서 바로 가져다 쓰기 좋은 공개 구현으로 [Koide et al. (2023)](https://arxiv.org/abs/2302.05094)이 ICRA 2023에서 발표한 `direct_visual_lidar_calibration`이 있다. 타겟 없이 LiDAR와 카메라의 외부 파라미터를 자동으로 구하는 이 파이프라인은 세 단계로 작동한다.

**1단계: LiDAR 포인트 클라우드 밀집화**

Ouster나 Velodyne 같은 회전식 LiDAR는 스캔 한 장만으로는 점이 성겨서 NID 정합에 쓰기 어렵다. 이를 해결하기 위해 CT-ICP(Continuous-Time ICP)로 연속된 여러 스캔을 정합해 겹쳐 쌓음으로써 조밀한 포인트 클라우드를 먼저 만든다. 반면 Livox Avia처럼 비반복(non-repetitive) 패턴으로 스캔하는 LiDAR는 정지 상태에서 점을 누적하기만 해도 자연스럽게 촘촘해진다.

**2단계: SuperGlue 기반 초기 추정**

앞서 만든 조밀한 포인트 클라우드를 가상의 카메라 시점에서 렌더링해 LiDAR 강도(intensity) 이미지를 생성한다. 그런 다음 이 가상 이미지와 실제 카메라 이미지 사이의 2D-2D 대응점을 학습 기반 특징 매처인 SuperGlue로 찾는다. 렌더링 이미지의 픽셀마다 원래의 3D 좌표가 매달려 있으므로, 찾은 2D-2D 대응은 곧바로 2D-3D 대응으로 바뀌고, 여기에 RANSAC과 재투영 오차 최소화를 적용해 초기 변환을 계산한다.

논문에서는 이동 오차 0.5m, 회전 오차 1도 이내로 들어오면 초기 추정이 성공한 것으로 판정했으며, Ouster LiDAR와 핀홀 카메라 조합의 15개 데이터셋 중 12개에서 초기화에 성공했다. 물론 센서 모델이나 장면 텍스처가 달라지면 초기 성공률도 달라진다.

**3단계: NID 기반 정밀 정합**

초기 추정에서 출발해 Nelder-Mead 알고리즘으로 NID 비용 함수를 최소화한다. 이때 카메라 시점에서 앞쪽 물체에 가려 실제로는 보이지 않아야 할 LiDAR 점을 view-based hidden point removal로 걸러 내어 엉뚱한 픽셀과 짝지어지는 일을 막으며, 변환이 갱신될 때마다 가려진 점 제거와 NID 최소화를 반복한다.

**논문 결과**: Ouster OS1-64와 핀홀 카메라 조합 실험에서 평균 이동 오차는 0.043m, 회전 오차는 0.374도로 보고되었다. 평가 기준값으로는 고반사 구 타겟을 써서 구한 변환을 사용했으며, 저자들은 이를 유사 정답(pseudo ground truth)이라 불렀다. 논문에는 Livox Avia와 전방향 카메라 조합에 대한 실험도 함께 수록되어 있다.

### 3.3.4 실전 도구 비교

실제 프로젝트에서 선택할 수 있는 대표적인 Camera-LiDAR 캘리브레이션 도구와 접근 방식을 실행 형태와 사전 확인 사항 중심으로 비교하면 아래 표와 같다.

| 도구 | 방식 | 타겟 필요 | 실행 형태 | 적용 전 확인 |
|------|------|-----------|-----------|--------------|
| Autoware Calibration Toolkit | Target-based | O | 사용자 절차를 포함한 오프라인 도구 | 지원 센서와 타겟, 초기값 |
| `direct_visual_lidar_calibration` (Koide) | Targetless (NID) | X | 오프라인 자동화 파이프라인 | 장면 중첩, 강도 영상, 초기화 성공 여부 |
| ACSC (Automatic Calibration) | Target-based + auto corner | O | 타겟 검출 자동화 | 구현이 요구하는 타겟과 센서 모델 |
| 온라인 extrinsic 추정 (VIO/LIO 상태 증강) | Online | X | 주행 중 추정 | 운동의 관측 가능성과 드리프트 감시 기준 |

최근에는 단일 프레임 기반의 정합 정밀도를 끌어올리려는 시도도 이어지고 있다. 예컨대 [MFCalib (2024)](https://arxiv.org/abs/2409.00992)은 깊이가 연속적으로 꺾이는 에지, 깊이가 끊기는 에지, 그리고 평면 위의 반사 강도 불연속 에지를 함께 추출해 짝짓는다. 또한 LiDAR 빔의 유한한 폭 때문에 물체 경계가 실제보다 두껍게 번져 보이는 현상(edge inflation)을 빔 모델로 보정해, 자체 데이터셋에서 기존 비교 방법보다 작은 정합 오차를 얻었다고 보고했다.

지금까지 살펴본 스테레오와 Camera-LiDAR 캘리브레이션은 두 센서가 같은 순간을 포착한다는 전제 아래 정적인 공간 변환만 맞추면 충분했다. 그러나 카메라와 IMU처럼 샘플링 주기가 수십 배 차이 나고 센서 자체 시계가 어긋난 상태에서 고속 움직임을 함께 적분해야 할 때는, 공간 변환뿐만 아니라 두 센서 사이의 시간 오프셋까지 동시에 풀어야 한다.

---

## 3.4 Camera-IMU Extrinsic + Temporal Calibration

카메라와 IMU를 함께 쓰는 시스템에서는 두 센서 사이의 공간적 상대 변환(extrinsic)뿐 아니라 시간축 어긋남(temporal offset)까지 한꺼번에 추정해야 한다. VINS-Mono나 OpenVINS 같은 VIO(Visual-Inertial Odometry) 시스템도 주행 중에 두 값을 계속 다듬긴 하지만, 처음부터 터무니없는 초기값을 주면 수렴하기 어렵기 때문에 Kalibr 같은 오프라인 도구로 미리 정밀하게 잡아 두고 출발한다.

### 3.4.1 왜 시간 오프셋이 중요한가

카메라와 IMU는 대개 각자 독립된 내부 클록을 기준으로 타임스탬프를 기록하며, 전송 지연이나 노출 기준 시점의 차이도 제각각이다. 두 시계 사이에 시간 오프셋 $t_d$가 존재하면, 카메라가 타임스탬프 $t_c$로 기록한 영상은 사실 IMU 시간축 기준으로 $t_c + t_d$ 순간의 장면을 담은 셈이 된다.

정지해 있거나 느리게 움직일 때는 이 차이가 눈에 띄지 않지만, 빠르게 회전하는 순간에는 곧바로 큰 재투영 오차로 번진다. 예를 들어 초당 100도(100 deg/s)로 회전하는 도중 시간 오프셋이 10ms만 어긋나 있어도 카메라가 바라보는 방향은 1도나 틀어진다. [Furgale et al. (2013)](https://ieeexplore.ieee.org/document/6696514)의 검증 실험에서도 이 정밀도가 잘 드러난다. 저자들은 카메라 노출 시간을 바꾸어 가며 시간 오프셋을 추정했는데, 추정값들에 맞춘 직선의 기울기가 0.498로 이론값인 0.5(노출 중간 시점 이동)에 바짝 다가섰고, 모든 추정값이 직선에서 ±0.2ms 이내에 모였다. 이는 200Hz IMU 샘플링 주기인 5ms보다 훨씬 정밀한 수준이다.

### 3.4.2 Kalibr: Continuous-Time B-Spline 기반 캘리브레이션

이산 시간(discrete-time) 상태 추정에서는 센서 프레임마다 상태 변수를 따로 두어야 하므로, 20~30Hz로 들어오는 카메라 영상과 200~1000Hz로 쏟아지는 IMU 데이터를 시간 오프셋 $t_d$까지 움직여 가며 직접 짝짓기가 까다롭다. Kalibr는 [Furgale et al. (2013)](https://ieeexplore.ieee.org/document/6696514)의 방법론을 구현해 이 문제를 궤적의 연속 시간(continuous-time) B-spline 표현으로 해결한다. 연속 함수로 궤적을 세워 두면 샘플링 주기가 전혀 다른 두 센서의 관측을 임의의 시각 $t$에서 곧바로 꺼내 비교할 수 있기 때문이다.

**B-Spline 궤적 표현**:

원 논문인 Furgale et al. (2013)은 IMU의 6-DoF 포즈를 이동 3성분과 회전 3성분(axis/angle 회전 벡터)으로 나누어 각각 6차 B-spline(조각마다 5차 다항식)으로 표현함으로써 위치의 2계 미분인 가속도가 부드러운 3차 다항식이 되도록 설계했고, 가속도계·자이로 bias는 3차 B-spline으로 두었다. 이후 Lovegrove et al. (2013)은 회전 벡터 보간의 특이점을 피하기 위해 아래 식과 같은 누적 3차(cubic) B-spline 형태를 제안했다. 이 방식은 인접한 제어점(control points) $\{\mathbf{T}_i\}$ 사이의 상대 변환을 $SE(3)$ 다양체 위에서 누적 곱함으로써, 임의 시각 $t$의 포즈 $\mathbf{T}(t) \in SE(3)$를 Lie 군 위에서 직접 보간한다.

$$
\mathbf{T}(t) = \mathbf{T}_i \prod_{j=1}^{3} \text{Exp}(\mathbf{B}_j(u) \cdot \Omega_{i+j})
$$

여기서 각 기호는 다음을 뜻한다.
- $u = (t - t_i) / \Delta t$: 제어점 구간 $[t_i, t_{i+1})$ 안에서의 정규화된 시간 ($0 \leq u < 1$)
- $\mathbf{B}_j(u)$: 3차 B-spline 기저 함수의 누적(cumulative) 계수
- $\Omega_{i+j} = \text{Log}(\mathbf{T}_{i+j-1}^{-1} \mathbf{T}_{i+j})$: 인접 제어점 사이 상대 변환을 리 대수(Lie algebra)로 옮긴 증분 벡터
- $\text{Exp}, \text{Log}$: $SE(3)$의 지수 및 로그 사상

궤적을 B-spline으로 잡으면 연속 시간 캘리브레이션에 필요한 세 가지 이점이 한 번에 따라온다.
1. **해석적 미분 가능**: 임의의 시각에서 속도와 가속도를 다항식 미분만으로 정확히 구해 IMU 관측 모델과 곧장 연결할 수 있다.
2. **비동기 센서 처리**: 센서 데이터를 억지로 보간하거나 리샘플링하지 않고 각 센서의 고유 타임스탬프에서 연속 궤적을 그대로 평가한다.
3. **국소성(locality)**: 특정 시각의 궤적 값은 주위의 소수 제어점(조각이 3차 다항식이면 4개, Furgale et al.처럼 5차 다항식이면 6개)에만 걸리므로, 전체 최적화 행렬이 희소(sparse)해져 대규모 데이터도 빠르게 풀린다.

**관측 모델**:

이제 동일한 연속 궤적 $\mathbf{T}(t)$ 위에서 카메라와 IMU의 예측 오차를 각각 세운다. 먼저 카메라는 시간 오프셋을 보정한 시각 $t_{c,k} + t_d$에서의 IMU 포즈를 꺼내 3D 랜드마크를 투영하고 실제 관측 픽셀과 비교한다.

$$
\mathbf{e}_{\text{cam},k} = \mathbf{m}_k - \pi\left(\mathbf{T}_{CB} \cdot \mathbf{T}(t_{c,k} + t_d)^{-1} \cdot \mathbf{p}_w\right)
$$

여기서 $\mathbf{T}(t)$는 IMU(body)에서 월드 좌표로 가는 변환이며, $\mathbf{T}_{CB}$는 추정 대상인 IMU(body)→카메라 외부 변환이다(아래 「요약」 표의 $\mathbf{T}_{CI}$와 동일한 변환을 가리킨다).

한편 IMU는 각 샘플 시각 $t_k$에서 연속 궤적을 시간 미분해 예측한 가속도와 각속도를 실제 측정치와 맞댄다.

$$
\mathbf{e}_{\text{accel},k} = \mathbf{a}_k - \left[\mathbf{R}(t_k)^\top(\ddot{\mathbf{p}}(t_k) - \mathbf{g}) + \mathbf{b}_a\right]
$$
$$
\mathbf{e}_{\text{gyro},k} = \boldsymbol{\omega}_k - \left[\mathbf{R}(t_k)^\top\boldsymbol{\omega}_w(t_k) + \mathbf{b}_g\right]
$$

여기서 $\mathbf{R}(t)$와 $\mathbf{p}(t)$는 $\mathbf{T}(t)$의 회전과 위치, $\boldsymbol{\omega}_w(t)$는 월드 좌표 기준 각속도다. 궤적에서 구한 가속도와 각속도는 월드 좌표로 표현되어 있으므로, $\mathbf{R}(t_k)^\top$를 곱해 IMU 로컬 좌표로 옮겨야 센서 측정치와 직접 뺄 수 있다.

**최적화 문제**:

모든 카메라 잔차, IMU 잔차, 그리고 시간에 따른 bias 변화를 하나의 가중 최소자승 목적 함수로 묶어 동시에 푼다.

$$
\min_{\mathbf{T}_{CB}, t_d, \mathbf{g}, \mathbf{b}_a(\cdot), \mathbf{b}_g(\cdot), \{\mathbf{T}_i\}} \sum_k \|\mathbf{e}_{\text{cam},k}\|^2_{\Sigma_c} + \sum_k \left(\|\mathbf{e}_{\text{accel},k}\|^2_{\Sigma_a} + \|\mathbf{e}_{\text{gyro},k}\|^2_{\Sigma_g}\right) + \int \left(\|\dot{\mathbf{b}}_a(\tau)\|^2_{Q_a} + \|\dot{\mathbf{b}}_g(\tau)\|^2_{Q_g}\right) d\tau
$$

마지막 적분 항은 bias를 random walk 과정으로 모델링한 데서 비롯된 정규화 항으로, bias가 짧은 순간에 비현실적으로 튀지 않도록 억제한다. 이때 중력 벡터 $\mathbf{g}$의 방향도 미지수로 함께 추정한다. Furgale et al. (2013)은 이 비선형 최소자승 문제를 Levenberg-Marquardt(LM) 알고리즘으로 풀었으며, B-spline의 국소성 덕분에 정보 행렬이 희소하게 유지되므로 희소 Cholesky 분해(CHOLMOD)로 선형 시스템을 빠르게 해독할 수 있다. 실제로 논문에서 보고한 약 80초 분량의 데이터셋에서는 반복 한 번마다 선형 시스템을 조립하는 데 약 18초, 푸는 데 약 0.2초가 소요되었다.

### 3.4.3 Kalibr 실행 가이드

Kalibr로 Camera-IMU 캘리브레이션을 실행하려면 AprilGrid 타겟을 인쇄하고, 타겟을 시야에 담은 채 센서 리그를 충분히 움직여 ROS bag을 녹화한 다음 `kalibr_calibrate_imu_camera`를 호출한다.

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

여기서 입력으로 넘기는 `imu.yaml` 파일에는 IMU 토픽 이름과 샘플링 속도, 그리고 관측 항의 가중치($\Sigma_a, \Sigma_g, Q_a, Q_g$)를 결정하는 노이즈 파라미터를 명시해야 한다.

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

데이터 수집 품질과 결과 검증에서는 아래 항목을 빠짐없이 확인한다.

**데이터 수집**:
1. **운동의 다양성**: 세 축 방향의 병진 이동과 세 축 둘레의 회전을 모두 고르게 일으켜(excite) 관측 가능성을 확보한다. 특히 세 축 모두에서 충분한 회전을 넣어야 시간 오프셋과 회전 외부 파라미터가 분리된다.
2. **운동 속도**: 너무 느리게 움직이면 가속도와 각속도 신호가 노이즈에 묻혀 IMU bias를 가려내기 어렵고, 반대로 너무 격렬하게 흔들면 영상에 모션 블러가 생겨 코너 검출 정밀도가 떨어진다.
3. **타겟 가시성**: 카메라가 AprilGrid 타겟을 안정적으로 계속 검출하면서도 각 축의 운동을 충분히 밟아 나가도록 이동 궤적을 설계한다.
4. **시작과 끝**: 녹화 시작과 끝 무렵에 잠시 정지 구간을 두면 초기 IMU bias와 중력 방향을 잡기가 한결 수월하다.

**결과 해석**: Kalibr가 생성한 출력 보고서에서는 다음 네 가지를 중점적으로 살핀다.
- `T_cam_imu`: 추정된 카메라-IMU 외부 파라미터(4×4 변환 행렬)가 실제 하드웨어 도면이나 줄자 실측치와 방향·거리 면에서 상식적으로 부합하는지 본다.
- `timeshift_cam_imu`: 추정된 시간 오프셋 $t_d$의 크기와 부호 규약을 확인한다.
- 재투영 오차 분포: 전체 평균 오차뿐만 아니라 시간 흐름이나 이미지 영역에 따라 잔차가 한쪽으로 쏠리지 않는지, 튄 이상치가 없는지 살핀다.
- 가속도계·자이로 잔차: 측정 잔차의 크기가 `imu.yaml`에 넣어 준 노이즈 모델 범위 안에 들어오는지 점검한다.

이때 `imu.yaml`에 넣는 노이즈 파라미터가 부정확하면 카메라와 IMU 잔차 사이의 균형이 무너져 캘리브레이션 품질이 떨어진다. 다음 절에서는 이 값을 직접 실측해 내는 표준 방법인 Allan variance를 다룬다.

### 3.4.4 Allan Variance 측정 실습

IMU의 노이즈 파라미터가 실제 센서 특성과 어긋나면 Kalibr의 캘리브레이션 가중치가 틀어질 뿐 아니라, 이후 VIO나 LIO 필터 전체의 공분산 추정과 성능까지 함께 무너진다. **Allan variance**는 David Allan이 1966년 원자 주파수 표준의 안정도를 재기 위해 고안한 통계량으로, 데이터를 묶는 클러스터 시간(cluster time) $\tau$를 바꾸어 가며 시계열 속에 섞인 백색 잡음과 저주파 bias 드리프트를 깔끔하게 분리해 낸다.

**Allan Variance 정의**:

샘플링 주기 $\tau_0$로 모은 길이 $N$의 정지 시계열 $\{x_k\}$를 길이 $\tau = n \cdot \tau_0$의 구간(클러스터)으로 나누었을 때, 인접한 클러스터 평균들의 차이로 정의한 Allan variance는 다음과 같다.

$$
\sigma^2(\tau) = \frac{1}{2(N-2n+1)} \sum_{k=1}^{N-2n+1} \left[\bar{x}_{k+n} - \bar{x}_k\right]^2
$$

여기서 $\bar{x}_k = \frac{1}{n}\sum_{j=0}^{n-1} x_{k+j}$는 $k$번째 위치부터 $n$개 샘플을 평균한 값이다. 위 수식은 클러스터를 한 샘플씩 옆으로 밀며 겹쳐 잡는 overlapping Allan variance를 나타내고, 아래 실습 코드는 구현이 직관적이고 계산이 빠른 non-overlapping 방식(클러스터를 겹치지 않게 분할)으로 계산한다.

**로그-로그 그래프로 노이즈 구분하기**:

Allan deviation $\sigma(\tau) = \sqrt{\sigma^2(\tau)}$를 클러스터 시간 $\tau$에 대해 로그-로그(log-log) 평면에 그리면, 구간별 기울기만 보고도 지배적인 노이즈 성분을 한눈에 읽어 낼 수 있다.

| 노이즈 유형 | 기울기 | $\sigma(\tau)$ |
|-----------|-------|---------------|
| Quantization noise | $-1$ | $\propto \tau^{-1}$ |
| **Angle/Velocity random walk** | $-1/2$ | $\propto \tau^{-1/2}$ |
| **Bias instability** | $0$ | 일정 (최솟값) |
| Rate random walk | $+1/2$ | $\propto \tau^{+1/2}$ |
| Rate ramp | $+1$ | $\propto \tau$ |

표에서 특히 중요한 두 구간이 기울기 $-1/2$ 영역(고주파 백색 잡음에서 비롯되는 angle/velocity random walk)과 곡선이 가장 바닥을 찍는 기울기 $0$ 영역(장시간 드리프트를 나타내는 bias instability)이다. 아래 파이썬 스크립트는 정지 상태의 자이로 시계열(시뮬레이션 데이터)로부터 non-overlapping Allan variance를 계산한 뒤, 로그-로그 기울기를 따라 백색 잡음 계수와 bias instability를 자동으로 추출해 참조선과 함께 플롯한다.

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
- **정지 데이터 수집**: IMU를 바닥 진동이 전해지지 않는 단단한 표면에 고정해 둔다. 녹화 길이는 분석하려는 최대 클러스터 시간 $\tau$에서도 서로 겹치지 않는 클러스터가 여러 개 남도록 길게 잡아야 하며, 최소한 bias-instability 평탄부(기울기 0 구간)가 뚜렷이 보일 때까지 수집 시간을 늘린다.
- **데이터시트와 비교**: 그래프에서 구한 noise density를 센서 제조사의 데이터시트 사양과 대조해 센서에 이상이 없는지 점검한다.
- **Kalibr에 넣을 값**: 기울기 $-1/2$ 구간 전체에 직선을 적합해 white-noise 계수를 뽑고, 그 단위가 Kalibr 설정이 요구하는 연속 시간 noise density 단위와 일치하는지 확인한다. 흔히 하듯 $\tau = 1$초 근처의 점 하나만 대충 읽어 옮기면 기울기가 $-1/2$에서 이미 벗어난 구간이나 단위 변환 실수를 놓치기 쉽다.
- **온도 안정화**: 전원을 켠 직후에는 칩 자체 발열로 bias가 크게 요동치므로, 센서 온도와 출력 평균이 충분히 자리를 잡은 뒤부터 본 녹화를 시작하고 당시 온도도 함께 기록한다. 열적 안정에 걸리는 시간은 IMU 하우징 구조와 외부 공기 노출 정도에 따라 크게 달라진다.

카메라와 IMU를 맞췄다면, 다음은 같은 관성 센서를 LiDAR와 결합할 차례다. 카메라처럼 평면 타겟의 2D 코너를 곧바로 투영할 수 없는 LiDAR-IMU 조합에서는 두 센서가 각자 계산한 3차원 움직임 자체를 맞대는 고전적인 접근이 힘을 발휘한다.

---

## 3.5 LiDAR-IMU Extrinsic Calibration

LiDAR와 IMU는 공통의 2D 시각 코너를 공유하지 않지만, 두 센서 모두 시간에 따른 자신의 3차원 상대 운동을 독립적으로 추정할 수 있다. 이 성질 덕분에 LiDAR와 IMU 사이의 외부 파라미터는 고전적인 **hand-eye calibration** 문제로 자연스럽게 정식화된다.

### 3.5.1 Hand-Eye Calibration (AX = XB)

Hand-eye calibration은 원래 로봇 팔 끝(hand)에 장착한 카메라(eye)와 로봇 손목 사이의 상대 변환을 구하기 위해 정립된 기법이다. 로봇 기저부(base)에서 계산한 팔 끝의 상대 운동과 카메라 영상으로 추정한 타겟의 상대 운동을 서로 맞대어 미지의 고정 변환을 풀어낸다([Tsai & Lenz, 1989](https://ieeexplore.ieee.org/document/34770)).

**문제 정의**:

센서 $A$(예: LiDAR)와 센서 $B$(예: IMU)가 하나의 강체(rigid body) 플랫폼에 단단히 고정되어 있다고 하자. 두 시각 $t_i, t_j$ 사이에 각 센서가 자신의 좌표계에서 관측한 상대 운동이 $\mathbf{A}_{ij}$와 $\mathbf{B}_{ij}$라면 다음과 같이 쓸 수 있다.

$$
\mathbf{A}_{ij} = \mathbf{T}_A^{-1}(t_i) \cdot \mathbf{T}_A(t_j) \quad \text{(센서 A의 상대 운동)}
$$
$$
\mathbf{B}_{ij} = \mathbf{T}_B^{-1}(t_i) \cdot \mathbf{T}_B(t_j) \quad \text{(센서 B의 상대 운동)}
$$

두 센서 사이의 고정 변환을 $\mathbf{X} = \mathbf{T}_{AB}$라 두면, 강체 운동의 일관성 때문에 센서 A의 좌표계에서 먼저 운동 $\mathbf{A}_{ij}$를 밟고 $\mathbf{X}$로 센서 B로 건너가는 변환과, 먼저 $\mathbf{X}$로 센서 B로 건너간 뒤 센서 B의 운동 $\mathbf{B}_{ij}$를 밟는 변환이 정확히 같아야 한다.

$$
\mathbf{A}_{ij} \mathbf{X} = \mathbf{X} \mathbf{B}_{ij}
$$

이것이 유명한 $\mathbf{AX} = \mathbf{XB}$ 방정식이며, 우리가 구해야 할 미지수는 $\mathbf{X} \in SE(3)$다.

**회전과 이동의 분리**:

위 $4 \times 4$ 동차 행렬 방정식을 회전 파트와 이동 파트로 쪼개면 해법의 실마리가 보인다.

$$
\mathbf{R}_A \mathbf{R}_X = \mathbf{R}_X \mathbf{R}_B \quad \text{(회전)}
$$
$$
\mathbf{R}_A \mathbf{t}_X + \mathbf{t}_A = \mathbf{R}_X \mathbf{t}_B + \mathbf{t}_X \quad \text{(이동)}
$$

회전 방정식에는 이동 벡터 $\mathbf{t}_X$가 전혀 들어가지 않으므로, $\mathbf{R}_X$를 먼저 독립적으로 푼 뒤 그 결과를 이동 방정식에 대입해 $\mathbf{t}_X$에 관한 선형 시스템을 푸는 2단계 절차가 성립한다.

**[Tsai & Lenz (1989)](https://ieeexplore.ieee.org/document/34770)의 해법**:

두 센서의 상대 회전축과 회전각을 각각 $\hat{\mathbf{a}}_A, \alpha$와 $\hat{\mathbf{a}}_B, \beta$라 두고, 스케일링된 회전 벡터 $\mathbf{p}_A = 2\sin(\alpha/2)\hat{\mathbf{a}}_A$와 $\mathbf{p}_B = 2\sin(\beta/2)\hat{\mathbf{a}}_B$를 정의하면 비선형 회전 식이 아래의 선형 방정식으로 바뀐다.

$$
\text{skew}(\mathbf{p}_A + \mathbf{p}_B) \cdot \mathbf{r}_X = \mathbf{p}_B - \mathbf{p}_A
$$

여기서 $\mathbf{r}_X = \tan(\theta_X/2)\hat{\mathbf{a}}_X$는 미지의 회전 $\mathbf{R}_X$를 나타내는 수정 로드리게스 벡터(modified Rodrigues vector)다. 운동 쌍마다 이 식을 세로로 쌓으면 선형 시스템 $\mathbf{C} \mathbf{r}_X = \mathbf{d}$가 만들어지며, 회전축이 서로 나란하지 않은 운동 쌍이 최소 둘만 있어도 유일한 해가 나온다. 구한 $\mathbf{R}_X$를 이동 식에 넣으면 $(\mathbf{R}_A - \mathbf{I})\mathbf{t}_X = \mathbf{R}_X\mathbf{t}_B - \mathbf{t}_A$꼴의 선형 시스템이 되므로 $\mathbf{t}_X$도 곧바로 풀린다.

**운동 쌍이 많을 때**: 실제 캘리브레이션에서는 노이즈를 줄이기 위해 둘보다 훨씬 많은 운동 쌍을 모아 과결정 선형 시스템을 최소자승으로 풀고, 필요하다면 그 해를 초기값 삼아 LM 비선형 최적화로 한 번 더 다듬는다.

### 3.5.2 Motion-based 자동 캘리브레이션

Hand-eye calibration을 실제 장비에 적용하려면 두 센서가 각자 자기 상대 운동을 추정할 수 있어야 한다. 다행히 LiDAR odometry(scan-to-scan ICP 등)와 IMU 사전적분은 서로의 정보 없이도 따로 돌아가므로, 별도의 캘리브레이션 타겟을 세우지 않고 센서를 여러 방향으로 움직여 모은 데이터만으로 아래 Tsai-Lenz 알고리즘을 돌려 외부 파라미터를 자동으로 얻어 낼 수 있다. 코드는 1단계에서 작은 회전(`angle < 1e-6`)을 걸러 내며 $\mathbf{R}_X$를 먼저 최소자승으로 푼 뒤, 2단계에서 $\mathbf{t}_X$를 구한다.

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

FAST-LIO2처럼 상태 벡터 안에 LiDAR-IMU 외부 파라미터를 넣고 주행 중에 함께 다듬는 시스템이라도, 처음 출발할 때의 초기값이 너무 멀리 떨어져 있거나 초기 기동에 회전·가속이 부족하면 제대로 수렴하지 못한다. [LI-Init](https://arxiv.org/abs/2202.11006)은 바로 이 지점을 메워, FAST-LIO2를 본격 가동하기에 앞서 시간 오프셋, 외부 파라미터, 중력 벡터, IMU bias를 실시간으로 추정해 넘겨주는 전용 초기화 모듈이다.

LI-Init은 세 단계를 거쳐 초기 파라미터를 잡아낸다.
1. 먼저 LiDAR 측정치만으로 궤적 상태를 추정한 뒤, 이를 IMU의 각속도·가속도 측정치와 비교한다.
2. 두 센서의 운동 예측이 맞물리도록 시간 오프셋과 외부 파라미터를 번갈아 반복 정제한다.
3. 구한 값을 FAST-LIO2로 넘겨주며, 이후에는 FAST-LIO2의 IEKF(§4.2.5)가 상태 벡터에 포함된 LiDAR-IMU extrinsic을 주행 중에 온라인으로 조금씩 더 다듬는다.

이 구성을 쓰면 LIO를 켜는 순간 별도 타겟이나 추가 센서 없이도 외부 파라미터를 잡을 수 있지만, 그만큼 충분하고 다양한 초기 운동이 뒤따라야 한다. 공개 구현의 README에서도 실행 직후 5초 이상 정지 상태를 유지한 뒤 센서 리그를 움직이기 시작하고, 초기화가 끝난 다음에도 FAST-LIO2 상태에서 15~30초 동안 움직이며 extrinsic을 안정화하라고 안내한다. 초기 운동이 단조로우면 값이 부정확하게 굳거나 수렴에 실패하므로, 높은 정밀도가 요구되는 시스템이라면 target-based 캘리브레이션 결과와 한 번 교차 검증해 보는 편이 안전하다.

**GRIL-Calib**: 손으로 들고 흔들 수 있는 소형 장비와 달리, 바퀴로 달리는 지상 로봇은 거의 평면 위에서만 움직이기 때문에 z축 외의 회전이 부족해 기존 6-DoF motion-based 방법으로는 일부 외부 파라미터 축이 관측되지 않는다. [GRIL-Calib (Kim et al., 2024)](https://arxiv.org/abs/2312.14035)은 이 제약을 정면으로 다루어, LiDAR odometry에 지면 평면 잔차(ground plane residual)를 추가하고 지면 평면 운동(GPM) 제약을 최적화에 직접 결합함으로써 평면 주행만으로도 6-DoF 외부 파라미터를 모두 추정해 낸다(IEEE RA-L 2024).

지금까지는 카메라, LiDAR, IMU처럼 서로 다른 종류의 센서를 짝짓는 방법을 보았다. 반면 자율주행 차량처럼 360도 주변 사각지대를 없애기 위해 LiDAR를 여러 대 장착한 시스템에서는 같은 종류의 LiDAR끼리 하나의 기준 좌표계로 모으는 절차가 필요하다.

---

## 3.6 LiDAR-LiDAR Extrinsic Calibration

자율주행 차량의 지붕이나 범퍼처럼 여러 위치에 LiDAR를 나누어 장착한 다중 LiDAR 리그(multi-LiDAR rig)에서는 각 LiDAR의 점군을 하나의 통일된 좌표계로 모아야 한다. 이를 위해 센서들 사이의 상대 포즈를 정밀하게 캘리브레이션한다.

### 3.6.1 문제 설정

차체에 $n$대의 LiDAR $\{L_1, L_2, \ldots, L_n\}$이 고정되어 있을 때, 기준이 되는 주 LiDAR $L_1$의 좌표계로 나머지 센서들의 점군을 옮겨 오기 위한 상대 변환 $\{\mathbf{T}_{L_1 L_i}\}_{i=2}^{n}$를 추정하는 것이 목표다.

### 3.6.2 Target-based 방법

가장 직관적인 방법은 여러 LiDAR의 시야가 동시에 닿는 자리에 커다란 평면 패널을 세워 두는 방식이다. 각 LiDAR의 스캔에서 타겟 평면을 RANSAC으로 피팅한 뒤, 물리적으로 동일한 하나의 평면이 두 센서 좌표계에서 각각 어떤 법선과 거리로 관측되는지를 제약 조건으로 삼아 상대 변환을 풀어낸다.

### 3.6.3 ICP 기반 Targetless 방법

별도의 패널을 세우기 어렵거나 운용 중 자연 장면을 그대로 쓰고 싶을 때는 두 센서의 시야각(FoV)이 겹치는지 여부에 따라 전략을 달리한다.

**중첩 영역이 있는 경우**:

두 LiDAR의 시야가 일정 부분 겹친다면, 겹치는 구역의 포인트 클라우드를 ICP(Iterative Closest Point)로 직접 정합해 상대 변환을 구할 수 있다. 가장 기본적인 point-to-point ICP는 대응점 사이의 유클리드 거리 합을 최소화한다.

$$
\min_{\mathbf{R}, \mathbf{t}} \sum_{i} \|\mathbf{p}_{L_2,i} - (\mathbf{R} \mathbf{p}_{L_1,i'} + \mathbf{t})\|^2
$$

반면 구조물이 평면 위주인 환경에서는 목표 점의 접평면 방향으로만 거리를 재는 point-to-plane ICP를 쓰는 편이 보통 더 빠르게 수렴한다.

$$
\min_{\mathbf{R}, \mathbf{t}} \sum_{i} \left[(\mathbf{R} \mathbf{p}_{L_1,i'} + \mathbf{t} - \mathbf{p}_{L_2,i})^\top \mathbf{n}_{L_2,i}\right]^2
$$

여기서 $\mathbf{n}_{L_2,i}$는 기준 포인트 클라우드 $L_2$의 점 $\mathbf{p}_{L_2,i}$에서 추정한 표면 법선 벡터다.

**중첩 영역이 없는 경우**:

차량 앞뒤나 좌우에 달려 시야가 전혀 겹치지 않는 LiDAR 쌍은 한 프레임 안에서 직접 정합할 수 없으므로, 플랫폼을 움직여 얻은 궤적이나 누적 지도를 매개로 삼는다.
1. **Hand-eye calibration (§3.5.1)**: 각 LiDAR가 독립적으로 추정한 odometry의 상대 운동을 모아 $\mathbf{AX}=\mathbf{XB}$ 방정식을 푼다.
2. **SLAM 기반**: 각 LiDAR로 따로 SLAM을 돌려 동일한 주행 구간의 3D 지도를 각각 만든 뒤, 두 지도가 겹치도록 정합해 상대 변환을 구한다.

### 3.6.4 Feature-based 방법

건물 벽면이나 도로의 기둥·경계석처럼 뚜렷한 기하 구조가 많은 환경에서는 전체 점군을 무작정 맞추는 대신 평면, 기둥, 에지 같은 기하 특징을 뽑아 대응시킨다. 평면 하나에도 수백 개의 점이 실리므로 개별 점에 섞인 거리 노이즈가 자연스럽게 평균되어 상쇄된다.

절차는 간단하다.
- 여러 LiDAR에서 공통으로 관측된 대형 평면(벽면, 바닥 등)을 각각 추출한다.
- 각 LiDAR 좌표계에서 평면 파라미터 $(n_i, d_i)$를 계산한다.
- 짝지어진 대응 평면 쌍의 법선과 거리 조건으로부터 상대 변환을 추정한다.

이때 3자유도 이동과 3자유도 회전을 빠짐없이 묶으려면, 법선 벡터가 한 평면 위에만 누워 있지 않은(non-coplanar) 평면 대응이 최소 3개 이상 확보되어야 한다.

지금까지 카메라와 LiDAR 같은 외수용(exteroceptive) 센서들 사이의 연결을 정리했다면, 이제 절대 좌표를 제공하는 GNSS 수신기와 IMU 사이의 물리적 간격, 즉 레버 암(lever arm)을 다룰 차례다.

---

## 3.7 GNSS-IMU Lever Arm & Boresight

GNSS/INS 통합 항법에서는 GNSS 안테나와 IMU가 차량의 서로 다른 위치에 달릴 수밖에 없으므로, 두 기준점 사이의 정확한 공간 오프셋을 알고 있어야 한다. 이 3차원 거리 벡터를 레버 암(lever arm)이라 부른다.

### 3.7.1 Lever Arm Vector

레버 암 $\mathbf{l} = [l_x, l_y, l_z]^\top$는 IMU의 관성 중심에서 GNSS 안테나의 위상 중심(phase center)을 가리키는 3D 벡터로, 플랫폼에 고정된 IMU body frame에서 정의한다. GNSS 수신기가 계산해 내놓는 위치는 차량 중심이나 IMU 위치가 아니라 안테나 위상 중심의 위치이므로, 이를 항법 기준점인 IMU 위치로 옮기려면 매 순간의 자세 회전을 반영해야 한다.

$$
\mathbf{p}_{\text{IMU}} = \mathbf{p}_{\text{GNSS}} - \mathbf{R}_{\text{body}}^{\text{nav}} \cdot \mathbf{l}
$$

여기서 $\mathbf{R}_{\text{body}}^{\text{nav}}$는 body frame을 navigation frame으로 돌리는 회전 행렬이다. 차량이 선회하면 안테나도 IMU 둘레를 따라 원호를 그리며 움직이므로, 레버 암을 단순히 상수 오프셋으로 취급해선 안 되고 반드시 현재 자세로 회전시킨 뒤 빼 주어야 한다. 가령 회전축에 수직인 1m 길이의 레버 암을 처음 자세 그대로 둔 채 차량이 10도만 돌아도 계산된 위치는 약 17cm($2\sin 5°$ m)나 어긋난다.

### 3.7.2 Lever Arm 추정 방법

필요한 정밀도와 장비 운용 조건에 따라 레버 암을 구하는 방법은 세 가지로 나뉜다.

**방법 1: 물리적 측정**

줄자나 레이저 거리 측정기, CAD 도면으로 직접 잴 수 있다. 요구되는 측정 정밀도는 안테나와 IMU 사이 거리, 플랫폼의 회전 각도 범위, 그리고 시스템의 허용 위치 오차를 함께 따져 가늠한다. 수 cm 수준의 손 측정 오차마저 전체 오차 예산을 넘어서는 정밀 항법에서는 아래의 정밀 측량이나 필터 추정을 병행한다.

**방법 2: 필터 기반 온라인 추정**

EKF 상태 벡터 안에 레버 암 벡터 $\mathbf{l}$을 추가해 주행하면서 함께 추정하는 방식이다. 다만 레버 암 성분이 필터에서 관측 가능해지려면(observable) 차량이 충분히 회전해야 한다. 자세 변화 없이 직선으로만 달리면 $\mathbf{R}\mathbf{l}$ 항이 일정한 위치 편향과 구별되지 않아 세 성분 중 어느 것도 분리해 낼 수 없으며, 평지에서 yaw 회전만 반복하더라도 회전축과 나란한 수직 성분 $l_z$는 끝까지 관측되지 않는다.

**방법 3: 사후 처리(post-processing)**

주행 중에 기록해 둔 GNSS/IMU 원시 데이터를 앞뒤 양방향(forward-backward) 스무더로 일괄 처리하면서 레버 암을 정밀하게 역산하는 방법이다. 실시간성보다 최고 정밀도가 중요한 정밀 측량(mobile mapping) 분야에서 주로 활용한다.

### 3.7.3 GNSS 안테나 위상 중심

물리적 측정에서 주의할 점이 하나 더 있다. GNSS 수신기가 실제 신호를 수신해 위치를 산출하는 기준점은 안테나의 기하학적 중심이 아니라 전기적인 **위상 중심**(Antenna Phase Center, APC)이다. APC는 기계적 외형 중심에서 조금 벗어나 있을 뿐 아니라, 위성의 고도각과 신호 주파수에 따라서도 미세하게 위치가 바뀐다. 이 변동을 Phase Center Variation(PCV)이라 하며, 크기는 수 mm에서 수 cm 수준에 이른다.

따라서 RTK나 PPP처럼 센티미터급 정밀 측위가 목표일 때는 안테나 모델별 위상 중심 보정 데이터(ANTEX 파일)를 적용해 주어야 한다. ANTEX 보정을 넣을지는 사용하는 안테나 등급, 기준국과 이동국의 처리 방식, 플랫폼의 자세 변화 폭, 그리고 전체 오차 예산을 종합해 결정한다.

지금까지 살펴본 오프라인 캘리브레이션은 실험실이나 출고 시점에 초기 파라미터를 정밀하게 잡아 준다. 하지만 실제 차량이나 로봇이 현장에 투입되어 장기간 운용되면 온도 변화와 노면 진동, 크고 작은 충격으로 센서 마운트가 조금씩 틀어지기 마련이다. 이 미세한 변동을 운용 중에 스스로 추적해 바로잡는 기법이 온라인 캘리브레이션이다.

---

## 3.8 Online / Continuous Calibration

초기 캘리브레이션을 마치고 현장에 나간 뒤에도 열팽창이나 진동, 충격으로 브래킷이 미세하게 휘면 고정해 두었던 변환 파라미터가 서서히 어긋난다. 주행이나 작업 도중에 센서 관측치를 이용해 이 파라미터를 지속적으로 다시 추정하는 기술을 온라인 캘리브레이션(online calibration)이라 한다.

### 3.8.1 Self-Calibration during SLAM

가장 널리 쓰이는 접근은 SLAM이나 VIO 추정기의 상태 벡터 안에 캘리브레이션 파라미터를 함께 편입시켜 항법 상태와 동시에 추정하는 전략이다.

**EKF 기반**: 예를 들어 OpenVINS는 카메라 내부 파라미터, camera-IMU 외부 파라미터, 시간 오프셋까지 상태 벡터 뒤에 붙여 주행 중에 실시간으로 갱신한다.

확장된 상태 벡터는 다음과 같은 모양이 된다.
$$
\mathbf{x} = \begin{bmatrix} \mathbf{x}_{\text{nav}} \\ \mathbf{x}_{\text{calib}} \end{bmatrix}
= \begin{bmatrix} \mathbf{q}, \mathbf{p}, \mathbf{v}, \mathbf{b}_g, \mathbf{b}_a \\ \mathbf{q}_{CI}, \mathbf{p}_{CI}, t_d, f_x, f_y, c_x, c_y, k_1, \ldots \end{bmatrix}
$$

보통 캘리브레이션 파라미터의 시간 변화는 느린 random walk 과정으로 모델링한다.

$$
\dot{\mathbf{x}}_{\text{calib}} = \mathbf{w}_{\text{calib}}, \quad \mathbf{w}_{\text{calib}} \sim \mathcal{N}(\mathbf{0}, \mathbf{Q}_{\text{calib}})
$$

여기서 공분산 $\mathbf{Q}_{\text{calib}}$는 캘리브레이션 값이 시간에 따라 얼마나 빨리 변할 수 있다고 허용할지를 정하는 튜닝 노브다. 값을 너무 크게 잡으면 순간적인 관측 노이즈를 파라미터 변화로 오해해 추정치가 출렁이고, 반대로 너무 작게 잡으면 실제 충격으로 마운트가 틀어졌을 때 새 값으로 따라가는 속도가 턱없이 느려진다.

**Factor graph 기반**: 최적화 기반 시스템인 VINS-Mono 역시 camera-IMU 외부 파라미터를 sliding window 최적화의 변수 노드로 넣어 다른 항법 상태와 함께 풀어낸다. 만약 외부 파라미터가 시간에 따라 서서히 변한다고 가정하는 factor graph 구조라면, 시각마다 별도의 파라미터 노드를 두고 이웃한 두 시각 사이를 between factor로 연결한다. 이때 between factor에 부여하는 노이즈를 작게 줄수록 파라미터가 시간 축에서 매끄럽고 천천히 변하도록 묶인다.

### 3.8.2 Extrinsic Drift 보정

장기 운용 중에 누적되는 외부 파라미터 드리프트를 감지하고 바로잡기 위해 실전에서는 다음 세 가지 전략을 상황에 맞게 조합한다.

1. **투영 정합 감시**: LiDAR 포인트 클라우드를 카메라 이미지에 주기적으로 투영해 깊이 에지와 이미지 에지가 얼마나 잘 겹치는지 감시하다가, 정합 점수가 기준 아래로 떨어지면 재캘리브레이션 경고를 띄우거나 보정 루틴을 가동한다.
2. **주기적 재캘리브레이션**: 주행 중에 자연스럽게 모인 로그 데이터를 백그라운드에서 돌려 정기적으로 targetless 캘리브레이션을 새로 수행한다.
3. **온라인 미세 조정**: 현재 운용 중인 값을 초기값으로 삼아 좁은 탐색 범위 안에서 지속적으로 국소 최적화를 돌린다.

최근에는 딥러닝을 접목해 온라인 보정의 속도와 편의성을 높이려는 연구도 활발하다. [CalibRefine (2025)](https://arxiv.org/abs/2502.17648)은 전처리 없는 원시 LiDAR 포인트 클라우드와 카메라 이미지를 곧바로 입력받아 타겟 없이 온라인으로 외부 파라미터를 예측한 뒤, 처음 얻은 추정치를 반복 후처리(iterative post-refinement) 단계로 다시 다듬어 정밀도를 높인다.

### 3.8.3 OpenCalib: 자율주행 통합 캘리브레이션 프레임워크

자율주행 차량처럼 카메라, LiDAR, IMU, Radar가 뒤섞인 풀셋 센서 구성을 다룰 때는 센서 쌍마다 제각각 흩어진 도구를 끌어모으기 번거롭다. [OpenCalib (Yan et al., 2022)](https://arxiv.org/abs/2205.14087)은 자율주행 차량에 필요한 주요 캘리브레이션 파이프라인을 하나의 오픈소스 프레임워크로 묶어 제공한다.

논문 3절 기준으로 OpenCalib이 지원하는 센서 쌍별 방식과 타겟 유무를 정리하면 다음과 같다.

| 센서 쌍 | 방법 | 타겟 |
|---------|------|------|
| Camera intrinsic | Zhang's method, calibration harp로 왜곡 평가 | 체커보드 |
| Camera-LiDAR | 수동 정렬 / 원형 구멍 보드로 intrinsic·extrinsic 동시 추정 / 차선·기둥 선 특징 자동 정렬 | 없음 / 있음 / 없음 |
| LiDAR-LiDAR | 수동 정렬 / 지면 정렬 + NICP + octree 최적화 | 없음 |
| LiDAR-IMU | BALM 기반 local map 최적화 | 없음 |
| Radar-Camera, Radar-LiDAR | 수동 정렬 | 없음 |
| Camera-Ground | vanishing point와 지면 homography | 공장 보드 또는 차선 |
| 온라인 보정 | Camera-IMU, LiDAR-IMU (hand-eye 방식), Radar yaw | 없음 |

각 캘리브레이션이 독립된 모듈로 구현되어 있어 차량의 센서 구성에 맞춰 필요한 모듈만 골라 실행할 수 있다. 다만 주의할 점도 있다. 예컨대 차량에 카메라 6대와 LiDAR 1대가 달려 있을 때 각 카메라와 LiDAR를 따로따로 캘리브레이션했다면, 이렇게 얻은 변환들을 이어 붙여 간접 계산한 카메라 간 상대 포즈가 실제 카메라끼리 직접 맞춘 값과 모순 없이 맞아야 한다. OpenCalib은 기본적으로 개별 센서나 한 쌍 단위로 문제를 풀기 때문에, 다중 센서 루프 전체의 기하학적 일관성은 결과를 합성해 별도로 검증해야 한다.

공간적인 외부 파라미터를 오프라인과 온라인으로 맞추는 흐름을 짚어 보았다. 이제 멀티센서 융합의 또 다른 핵심 축인 시간 동기화와 시간 오프셋 캘리브레이션(temporal calibration)을 체계적으로 정리한다.

---

## 3.9 Temporal Calibration

센서 간의 기하학적 위치와 각도를 아무리 정밀하게 맞춰 놓아도, 두 센서가 가리키는 시각이 다르면 융합의 전제가 흔들린다. 빠르게 움직이는 플랫폼에서는 불과 수 밀리초의 시간 오차도 센티미터 단위의 위치 어긋남으로 이어진다(구체적인 계산 예는 §8.6.2 참조).

### 3.9.1 Hardware Synchronization

센서 하드웨어가 공통의 클록이나 트리거 입력을 받을 수 있도록 배선할 수 있다면, 소프트웨어 보정에 기대기 전에 하드웨어 동기화를 최우선으로 구축하는 것이 정석이다. 다만 하드웨어 신호를 물렸더라도 장치 내부의 처리 지연(latency)이 얼마인지, 그리고 기록되는 타임스탬프가 카메라 노출의 시작·중간·끝 중 어느 시점을 가리키는지는 장치마다 따로 확인해 두어야 한다.

**Trigger 기반 동기화**:

하나의 마스터 타이머가 모든 센서에 동일한 전기적 트리거 펄스를 뿌려 같은 물리적 순간에 측정을 시작하게 만드는 방식이다.
- **카메라 트리거**: 외부 트리거 입력(GPIO 핀)에 펄스를 넣어 셔터 노출이 시작되는 시각을 정확히 맞춘다.
- **LiDAR 동기**: Ouster처럼 PPS(Pulse Per Second) 입력을 지원하는 LiDAR는 외부 펄스를 받아 내부 시계와 회전 스캔 위상을 동기화한다.
- **IMU 동기**: IMU의 내부 샘플링 클록을 외부 기준 신호에 잠가(lock) 드리프트를 막는다.

**PPS (Pulse Per Second)**:

GNSS 수신기는 위성 시간(GPS time)의 매 정각 초에 맞춰 1Hz 전기 펄스(PPS)를 출력한다. 펄스의 상승 에지(rising edge)가 정각 초에 딱 떨어지도록 수신기를 설정해 두면, 이 신호를 시스템 전체의 공통 시간 기준으로 쓸 수 있다. 아래 그림은 GPS 정각 초마다 들어오는 PPS 펄스를 기준으로 주기가 서로 다른 카메라(30Hz), LiDAR(10Hz), IMU(200Hz) 샘플이 어떻게 하나의 공통 시간축 위에 정렬되는지 보여 준다.

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

PPS 기반 동기화에서는 각 센서가 자신의 로컬 타임스탬프와 가장 가까운 PPS 펄스 사이의 오프셋을 측정해 모든 센서 데이터를 공통의 GPS 시간축으로 변환한다. 물론 이때도 수신기 설정, 케이블과 입력 회로의 전파 지연, 그리고 각 장치가 찍는 타임스탬프의 기준 시점은 별도로 검증해야 한다.

### 3.9.2 PTP (Precision Time Protocol)

별도의 트리거 배선을 깔기 어려운 이더넷 기반 센서 네트워크에서는 **IEEE 1588 PTP**(Precision Time Protocol)가 표준으로 쓰인다. 센서와 네트워크 스위치가 모두 하드웨어 타임스탬핑과 PTP를 지원한다면 클록 동기화 오차를 1마이크로초 미만으로 묶을 수 있으며, 실제 달성 정밀도는 네트워크 토폴로지와 장비 사양에 따라 달라진다.

**PTP 동작 원리**:

PTP는 마스터(Grandmaster)와 슬레이브가 타임스탬프가 담긴 패킷을 주고받으며 시계 오프셋과 네트워크 왕복 지연을 동시에 계산한다. 아래 시퀀스 다이어그램은 네 번의 시각 측정($t_1 \sim t_4$)으로 오프셋과 지연을 구하는 메시지 교환 과정이다.

1. **마스터-슬레이브 구조**: 네트워크에서 가장 정확한 기준 시계 하나를 Grandmaster로 지정해 기준 시간을 전파한다.
2. **Sync 메시지**: 마스터가 슬레이브로 Sync 메시지를 주기적으로 송신한다($t_1$에 출발해 $t_2$에 도착).
3. **Follow-up**: 마스터가 앞서 보낸 Sync의 정확한 송신 시각 $t_1$을 Follow-up 메시지에 실어 알려 준다.
4. **Delay Request/Response**: 슬레이브가 $t_3$에 Delay_Req를 보내면 마스터가 $t_4$에 수신한 뒤 그 시각을 Delay_Resp로 되돌려주어 네트워크 지연을 잰다.
5. **오프셋 계산**: 마스터→슬레이브와 슬레이브→마스터 경로 지연이 대칭으로 같다고 가정하고, $\text{offset} = \frac{(t_2 - t_1) - (t_4 - t_3)}{2}$로 시계 차이를 보정한다.

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

최근 출시되는 많은 이더넷 LiDAR와 산업용 카메라가 PTP와 하드웨어 타임스탬핑을 내장하고 있으므로, 실제 적용 시 지원 여부와 타임스탬프가 찍히는 기준 시점을 모델별 데이터시트에서 먼저 확인한다.

### 3.9.3 Software Synchronization

센서가 외부 트리거나 PTP를 전혀 지원하지 않아 하드웨어 동기화를 꾸릴 수 없을 때는 소프트웨어 방식으로 시간을 맞출 수밖에 없다.

**Host Clock 기반**: 센서 패킷이 호스트 컴퓨터에 도착한 순간 운영체제의 시스템 시계로 타임스탬프를 찍는 가장 단순한 방식이다. 다만 이 시각에는 USB 버스나 네트워크 스택, 드라이버 대기열(queue)을 거치며 생기는 전송 지연과 그 불규칙한 흔들림(jitter)이 고스란히 섞여 들어간다. 지연과 jitter의 크기는 센서 드라이버와 시스템 부하에 따라 크게 달라지므로 실측으로 확인해야 한다.

**ROS Time**: ROS(Robot Operating System) 환경에서는 메시지를 수신한 시점에 `ros::Time::now()`를 찍어 기록하기도 한다. 만약 센서 드라이버가 센서 내부 클록의 타임스탬프를 ROS 시간축으로 변환해 넣어 주는 구조라면, 그 변환 로직이 클록 드리프트와 전송 지연을 얼마나 정확하게 걸러 내는지가 곧 전체 동기화 정밀도를 좌우한다.

### 3.9.4 Time Offset 온라인 추정

하드웨어 동기화가 불완전하거나 고정된 지연이 남아 있을 때는 상태 추정기 안에서 시간 오프셋 $t_d$ 자체를 미지수로 놓고 온라인으로 추정할 수 있다. [Li & Mourikis (2014)](https://doi.org/10.1177/0278364913515286)는 이 시간 오프셋을 VIO 상태 벡터에 포함시켜 주행 중에 실시간으로 추정하는 방법을 정립했다.

**관측 모델에 시간 오프셋 넣기**:

카메라가 타임스탬프 $t_c$를 달고 내보낸 이미지의 실제 촬영 포즈는 IMU 시간축으로 보정한 시각 $t_c + t_d$에서 꺼내 써야 하며, IMU 사전적분 구간 역시 이 보정 시각에 맞춰 끊는다.

$$
\mathbf{z}(t_c) = \pi(\mathbf{T}(t_c + t_d) \cdot \mathbf{p}_w)
$$

시간 오프셋 $t_d$가 작다고 가정하고 $t_c$ 주위에서 1차 테일러 전개를 적용하면, 시간 축의 이동을 포즈 위의 속도 섭동으로 깔끔하게 끌어낼 수 있다.

$$
\mathbf{T}(t_c + t_d) \approx \mathbf{T}(t_c) \cdot \text{Exp}(\boldsymbol{\xi} \cdot t_d)
$$

여기서 $\boldsymbol{\xi}$는 시각 $t_c$에서의 body 속도(각속도와 선속도를 묶은 twist)다. 이 1차 근사 덕분에 시간 오프셋 $t_d$에 대한 자코비안이 닫힌 식으로 떨어지므로, EKF나 비선형 최적화 안에서 다른 항법 상태와 나란히 $t_d$를 추정할 수 있다.

이 연속 시간 아이디어를 더 넓은 이종 센서 조합으로 확장한 연구도 있다. [iKalibr (Chen et al., IEEE T-RO 2025)](https://arxiv.org/abs/2407.11420)은 IMU를 중심으로 여러 대의 LiDAR, 카메라, radar를 묶은 다중 센서 시스템의 공간 변환과 시간 오프셋을 타겟 없이 B-spline 연속 시간 궤적으로 한 번에 일괄 추정한다.

**관측 가능성(Observability) 조건**: 시간 오프셋 $t_d$를 안정적으로 구별해 내려면 플랫폼이 충분히 가속하고 회전해야 한다. 만약 등속 직선 운동만 계속하면 시간을 조금 앞당겨 잡는 것과 위치를 조금 옮겨 잡는 것이 센서 관측에서 완전히 같은 결과를 내기 때문에 시간 오프셋이 관측 불가능해진다.

실제로 OpenVINS나 VINS-Mono 같은 주요 VIO 프레임워크들이 Li & Mourikis 방식의 변형을 기본 탑재하고 있으며, 오프라인 캘리브레이션 도구인 Kalibr(§3.4.2)는 연속 시간 B-spline 배치 최적화로 같은 $t_d$를 사전에 정밀 추정해 준다.

### 3.9.5 실전 동기화 전략 가이드

하드웨어 지원 여부와 네트워크 구성에 따라 선택할 수 있는 현실적인 동기화 전략과 사전 점검 항목을 정리하면 아래 표와 같다.

| 구성 조건 | 우선 검토할 방법 | 반드시 확인할 항목 |
|-----------|------------------|---------------------|
| 공통 트리거 입력과 clock을 배선할 수 있음 | 하드웨어 트리거 + 공통 clock/PPS | 노출·샘플·스캔 중 timestamp 기준 시점, 고정 latency |
| 모든 장치와 스위치가 PTP 및 하드웨어 timestamp를 지원 | PTP | grandmaster 선택, PTP profile, 경로 비대칭, lock 상태 |
| 장치 clock은 읽을 수 있으나 공통 신호가 없음 | clock 변환 + 오프라인/온라인 offset 추정 | clock drift, 운동의 관측 가능성, offset의 시간 변화 |
| 수신 시각만 사용할 수 있음 | host timestamp를 임시 사용하고 jitter 측정 | USB·네트워크 queue 지연; 요구 오차를 만족하지 못하면 장비 구성 변경 |

**PPS와 PTP 함께 쓰기**: 이더넷 센서와 GNSS를 함께 쓰는 차량에서는 GNSS 수신기의 1Hz PPS 신호로 PTP grandmaster의 시각을 GPS 시간에 맞춰 두고, 나머지 LiDAR와 카메라는 PTP로 grandmaster를 따르게 구성하는 조합이 깔끔하다.

**연구용 시제품**: 시스템의 허용 시간 오차 범위를 넘지 않는 수준이라면, 소프트웨어 타임스탬프 동기화에 온라인 시간 오프셋 추정을 얹는 것만으로도 충분히 안정적인 결과를 얻는다. 예를 들어 Kalibr로 초기 오프셋을 구해 넣어 두고 VIO가 주행 중에 미세 변동을 따라가게 하면 하드웨어 개조 부담을 크게 덜 수 있다.

---

## 요약

지금까지 다룬 여덟 가지 캘리브레이션 유형의 추정 대상과 대표 기법, 그리고 올바른 해를 얻기 위한 최소 요구 조건을 한눈에 비교하면 다음과 같다.

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

여러 센서가 물린 플랫폼을 처음 셋업할 때는 앞 단계의 결과가 다음 단계의 입력이나 제약으로 이어지므로 아래 순서를 따르는 것이 안전하다.

1. 각 카메라의 intrinsic (카메라마다 따로 구함)
2. 스테레오 extrinsic (있는 경우)
3. Camera-IMU extrinsic + temporal (Kalibr)
4. LiDAR-IMU extrinsic (hand-eye 또는 LI-Init)
5. Camera-LiDAR extrinsic (camera-IMU와 LiDAR-IMU의 체인으로 간접 계산하거나 직접 캘리브레이션)
6. GNSS lever arm
7. 온라인 캘리브레이션 설정 (운용 중 보정)

5번 단계의 간접 계산은 $\mathbf{T}_{CL} = \mathbf{T}_{CI} \cdot \mathbf{T}_{IL}$의 변환 체인을 이용하는 방식인데, 두 번의 캘리브레이션 오차가 그대로 겹쳐 쌓이므로 가능하다면 Camera-LiDAR 직접 캘리브레이션 결과와 대조해 오차를 확인해 두어야 한다.

이렇게 모든 센서의 내부 모델과 센서 사이의 공간 변환 및 시간 오프셋을 확정했다면, 이제 캘리브레이션된 센서 관측치들을 한데 모아 로봇의 위치와 자세, 속도를 실시간으로 계산하는 상태 추정 알고리즘으로 넘어갈 차례다(→ 4장).

**관련 논문**:

- [Zhang (2000)](https://ieeexplore.ieee.org/document/888718): 카메라 intrinsic — 호모그래피 기반의 유연한 캘리브레이션
- [Furgale et al. (2013)](https://ieeexplore.ieee.org/document/6696514): Camera-IMU — B-spline 연속 시간 궤적으로 시공간 동시 캘리브레이션
- [Tsai & Lenz (1989)](https://ieeexplore.ieee.org/document/34770): Hand-eye — AX=XB의 초기 해법
- [Koide et al. (2023)](https://arxiv.org/abs/2302.05094): Camera-LiDAR targetless — NID + SuperGlue 기반 자동 캘리브레이션
- [Li & Mourikis (2014)](https://doi.org/10.1177/0278364913515286): Temporal — 시간 오프셋 온라인 추정
- [OpenCalib (2022)](https://arxiv.org/abs/2205.14087): 자율주행 통합 캘리브레이션 프레임워크
- [GRIL-Calib (Kim et al., 2024)](https://arxiv.org/abs/2312.14035): 지면 평면 운동 제약을 쓰는 지상 로봇용 targetless IMU-LiDAR 캘리브레이션. 평면 운동만으로 6-DoF 추정.
- [MFCalib (2024)](https://arxiv.org/abs/2409.00992): 깊이 연속·불연속 에지와 강도 불연속 에지를 함께 쓰는 single-shot targetless LiDAR-카메라 캘리브레이션. LiDAR 빔 모델로 edge inflation을 보정.
- [iKalibr (Chen et al., IEEE T-RO 2025)](https://arxiv.org/abs/2407.11420): Temporal — B-spline 연속 시간 기반 다중 이종 센서(LiDAR·카메라·IMU·radar) 통합 시공간 캘리브레이션
