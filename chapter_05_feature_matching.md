# Ch.5 — Feature Matching & Correspondence: 기술 계보

Ch.4의 칼만 필터와 factor graph는 어떤 관측이 어떤 랜드마크에서 왔는지, 곧 데이터 연관(data association)을 이미 안다고 놓고 계산한다. 이 대응은 특징점 매칭(feature matching)과 대응점 탐색(correspondence search)으로 구한다.

---

## 5.1 Correspondence 문제란

### 5.1.1 "같은 것"을 찾는 문제

Correspondence 문제는 둘 이상의 관측(observation)에서 물리적으로 같은 점, 영역, 구조물을 찾아내는 문제다. 센서 퓨전 파이프라인은 거의 모든 단계에서 대응점을 쓴다.

- **Visual Odometry**: 연속된 프레임에서 같은 3D 점의 2D 투영을 찾아야 카메라 모션을 추정할 수 있다.
- **Calibration**: 카메라-LiDAR 외부 파라미터를 추정하려면 두 센서 관측 사이의 대응을 찾아야 한다. target을 쓰는 방법은 같은 물리적 점을 대응으로 삼고, targetless MI 기반 방법(§5.4)은 겹치는 영역의 통계적 의존성을 대응으로 삼는다.
- **Loop Closure**: 이전에 방문한 장소를 재인식하려면 현재 관측과 과거 관측 사이의 대응을 확인해야 한다.
- **Point Cloud Registration**: 두 스캔을 맞추려면(alignment) 대응점 쌍으로 강체 변환을 추정한다.

### 5.1.2 Correspondence의 유형

#### 2D-2D Correspondence

두 이미지에서 같은 3D 점이 투영된 위치를 찾는 문제다. Visual Odometry, Stereo Matching, Image Stitching이 이 대응을 쓴다.

이미지 $I_1$의 점 $\mathbf{p}_1 = (u_1, v_1)$과 이미지 $I_2$의 점 $\mathbf{p}_2 = (u_2, v_2)$가 같은 3D 점 $\mathbf{X}$의 투영일 때, 이 쌍 $(\mathbf{p}_1, \mathbf{p}_2)$를 대응점(correspondence)이라 한다. $3 \times 3$ 행렬을 곱하고 $\mathbf{K}^{-1}$을 적용하려면 점이 동차 좌표여야 하므로, 아래 식에서는 점을 $(u, v, 1)^\top$로 쓴다.

두 이미지 사이의 기하 관계는 epipolar constraint로 쓴다.

$$\mathbf{p}_2^\top \mathbf{F} \mathbf{p}_1 = 0$$

여기서 $\mathbf{F}$는 Fundamental matrix (3×3, rank 2)이다. 내부 파라미터를 알면 Essential matrix $\mathbf{E} = \mathbf{K}_2^\top \mathbf{F} \mathbf{K}_1$를 쓴다.

$$\hat{\mathbf{p}}_2^\top \mathbf{E} \hat{\mathbf{p}}_1 = 0$$

여기서 $\hat{\mathbf{p}} = \mathbf{K}^{-1} \mathbf{p}$는 정규화된 이미지 좌표다.

#### 2D-3D Correspondence

이미지의 2D 점과 3D 맵 포인트 사이의 대응이다. PnP(Perspective-n-Point) 문제의 입력이며, Visual Localization과 SLAM의 map-based tracking에 쓰인다.

3D 점 $\mathbf{X} = (X, Y, Z)^\top$과 그 2D 투영 $\mathbf{p} = (u, v)^\top$의 관계는 다음과 같다.

$$s \begin{bmatrix} u \\ v \\ 1 \end{bmatrix} = \mathbf{K} [\mathbf{R} | \mathbf{t}] \begin{bmatrix} X \\ Y \\ Z \\ 1 \end{bmatrix}$$

카메라 포즈 $(\mathbf{R}, \mathbf{t})$는 DLT로는 대응점 6개, P3P로는 3개(해를 하나로 고르려면 1개 더)로 추정한다.

#### 3D-3D Correspondence

두 포인트 클라우드(point cloud)에서 같은 물리적 점을 찾는 문제다. LiDAR scan-to-scan 정합, multi-LiDAR calibration, loop closure의 point cloud registration에 쓰인다.

두 포인트 클라우드 $\mathcal{P} = \{\mathbf{p}_i\}$와 $\mathcal{Q} = \{\mathbf{q}_i\}$ 사이의 강체 변환 $(\mathbf{R}, \mathbf{t})$를 다음 식으로 추정한다.

$$\min_{\mathbf{R}, \mathbf{t}} \sum_{i} \| \mathbf{q}_i - (\mathbf{R} \mathbf{p}_i + \mathbf{t}) \|^2$$

대응을 알면 이 문제의 closed-form 해를 SVD로 구한다(ICP, Ch.7).

### 5.1.3 센서 퓨전과 correspondence

잘못된 대응점(outlier)이 섞이면 포즈 추정 오차가 크게 늘어난다. 텍스처가 없는 환경에서는 쓸 만한 대응점이 모자라 포즈를 아예 계산하지 못하기도 한다.

---

## 5.2 전통적 Feature Detection & Description

전통적 correspondence 파이프라인은 검출(detect), 기술(describe), 매칭(match) 세 단계를 차례로 거친다. 검출은 특징점을 어디서 찾을지, 기술은 그 점을 어떤 숫자로 나타낼지 정한다.

### 5.2.1 Corner Detection: Harris → FAST → ORB

#### Harris Corner Detector (1988)

Harris corner detector는 이미지 패치를 어느 방향으로 옮겨도 밝기가 크게 변하는 점을 코너로 본다.

이미지 $I$에서 점 $(x, y)$ 주변의 윈도우를 $(\Delta u, \Delta v)$만큼 옮길 때의 밝기 변화는 다음과 같다.

$$E(\Delta u, \Delta v) = \sum_{x, y} w(x, y) [I(x + \Delta u, y + \Delta v) - I(x, y)]^2$$

$w(x, y)$는 가우시안 윈도우다. 1차 Taylor 전개를 적용하면 다음과 같다.

$$E(\Delta u, \Delta v) \approx \begin{bmatrix} \Delta u & \Delta v \end{bmatrix} \mathbf{M} \begin{bmatrix} \Delta u \\ \Delta v \end{bmatrix}$$

여기서 $\mathbf{M}$은 structure tensor(또는 second moment matrix)다.

$$\mathbf{M} = \sum_{x, y} w(x, y) \begin{bmatrix} I_x^2 & I_x I_y \\ I_x I_y & I_y^2 \end{bmatrix}$$

$I_x, I_y$는 이미지의 x, y 방향 그래디언트다. $\mathbf{M}$의 고유값 $(\lambda_1, \lambda_2)$로 점을 다음처럼 나눈다.

- $\lambda_1 \approx 0, \lambda_2 \approx 0$: 평탄한 영역 (밝기 변화 없음)
- $\lambda_1 \gg \lambda_2 \approx 0$: 에지 (한 방향으로만 변화)
- $\lambda_1, \lambda_2$ 모두 큼: 코너 (모든 방향으로 변화)

Harris는 고유값을 직접 계산하지 않고 corner response function을 쓴다.

$$R = \det(\mathbf{M}) - k \cdot \text{tr}(\mathbf{M})^2 = \lambda_1 \lambda_2 - k(\lambda_1 + \lambda_2)^2$$

$k$는 보통 0.04~0.06이다. $R$이 임계값보다 큰 점을 코너로 고르고 non-maximum suppression을 적용한다.

**Harris의 한계**: 회전에는 불변(rotation invariant)이고 스케일에는 불변이 아니다. 카메라가 다가가면 같은 코너가 윈도우 안에서 에지처럼 보일 수 있다. SIFT는 scale-space로 이 문제를 다룬다.

```python
import cv2
import numpy as np

# Harris Corner Detection
img = cv2.imread('scene.jpg', cv2.IMREAD_GRAYSCALE)
img_float = np.float32(img)

# blockSize: 이웃 크기, ksize: Sobel 커널, k: Harris 파라미터
harris_response = cv2.cornerHarris(img_float, blockSize=2, ksize=3, k=0.04)

# 임계값 적용 — cornerHarris는 응답 맵만 돌려주므로 NMS는 별도 단계가 필요하다
corners = harris_response > 0.01 * harris_response.max()
```

#### FAST (Features from Accelerated Segment Test, 2006)

FAST는 속도를 우선한 코너 검출기다. Rosten과 Drummond가 ICCV 2005에서 segment test 검출기를 처음 내놓았고, ECCV 2006에서 여기에 결정 트리 학습을 더했다. 확장판인 [Rosten, Porter & Drummond (TPAMI 2010)](https://arxiv.org/abs/0810.2434)에 따르면 PAL 실시간 영상을 처리하는 데 가용 처리 시간의 5% 미만이 들었고, 같은 조건에서 Harris는 115%, SIFT는 195%였다.

알고리즘은 다음과 같다.

1. 후보 픽셀 $p$를 중심으로 반지름 3인 원(Bresenham circle) 위의 픽셀 16개를 본다.
2. 원 위에서 연속한 $N$개 픽셀(2005년 원래 검출기는 $N=12$, OpenCV 기본값은 $N=9$)이 모두 $p$보다 밝거나 모두 어두우면 $p$를 코너로 본다.
3. 빠른 기각(high-speed test): 1, 5, 9, 13번 픽셀 넷만 먼저 본다. 이 중 셋 이상이 모두 밝거나 모두 어두울 때만 전체 검사로 넘어간다. 이 단축 검사는 $N=12$ 전용이므로, $N=9$ 같은 변형에는 아래의 결정 트리 학습을 쓴다.

$$\text{FAST condition: } \exists \text{ contiguous arc of } N \text{ pixels on circle, all } > I_p + t \text{ or all } < I_p - t$$

여기서 $t$는 밝기 임계값이다.

**Decision tree 학습**: FAST는 ID3 decision tree로 픽셀 검사 순서를 학습한다. 어느 픽셀부터 봐야 코너가 아닌 점을 가장 빨리 기각하는지를 데이터에서 찾는다.

**FAST의 한계**: FAST는 코너 위치만 준다. 방향(orientation), 스케일(scale), 디스크립터는 뒤에 ORB가 덧붙였다.

```python
# FAST Corner Detection
fast = cv2.FastFeatureDetector_create(threshold=20, nonmaxSuppression=True)
keypoints = fast.detect(img, None)
print(f"Detected {len(keypoints)} keypoints")
```

#### ORB (Oriented FAST and Rotated BRIEF, 2011)

ORB는 [Rublee et al. (2011)](https://ieeexplore.ieee.org/document/6126544)이 특허 제약이 없고 SIFT보다 빠른 대안으로 제안했다. FAST 검출기에 방향을 붙이고, BRIEF 디스크립터를 그 방향으로 회전시킨다.

**oFAST (Oriented FAST)**:
- FAST로 찾은 키포인트에 intensity centroid 방법으로 방향을 정한다.
- 키포인트 주변 패치의 이미지 모멘트를 계산한다.

$$m_{pq} = \sum_{x, y} x^p y^q I(x, y)$$

- Centroid: $\mathbf{C} = (m_{10}/m_{00}, m_{01}/m_{00})$
- 방향: $\theta = \text{atan2}(m_{01}, m_{10})$

**rBRIEF (Rotated BRIEF)**:
- BRIEF는 패치 안의 무작위 점 쌍 $(x_i, y_i)$의 밝기를 비교해 이진 디스크립터를 만든다.

$$\tau(\mathbf{p}; x_i, y_i) = \begin{cases} 1 & \text{if } I(\mathbf{p}, x_i) < I(\mathbf{p}, y_i) \\ 0 & \text{otherwise} \end{cases}$$

- 비교 256번으로 256비트 디스크립터를 얻는다.
- ORB는 키포인트 방향 $\theta$만큼 비교 점 쌍을 돌려 회전 불변성을 얻는다.
- 또 서로 상관이 낮은 점 쌍을 탐욕적(greedy)으로 골라 분별력(discriminability)을 높인다.

**멀티 스케일**: 이미지 피라미드(OpenCV 구현의 기본값은 8단계)의 각 레벨에서 FAST를 돌려 스케일 변화에 대응한다.

ORB-SLAM 시리즈는 ORB 특징점을 쓴다. 이진 디스크립터라서 Hamming distance로 빠르게 매칭한다.

```python
# ORB Detection & Description
orb = cv2.ORB_create(nfeatures=1000)
keypoints, descriptors = orb.detectAndCompute(img, None)
# descriptors.shape: (N, 32) — 256-bit = 32 bytes
```

### 5.2.2 Blob Detection: SIFT → SURF

#### SIFT (Scale-Invariant Feature Transform, 2004)

SIFT는 [Lowe (2004)](https://link.springer.com/article/10.1023/B:VISI.0000029664.99615.94)가 제안한 특징점 검출·기술 알고리즘이다. 스케일과 평면 내 회전에 불변이고, 그래디언트를 정규화한 디스크립터라서 어느 정도의 조명 변화와 affine 왜곡에도 견딘다. 오랫동안 특징점 매칭의 기준선으로 쓰였고, 미국 특허(US 6,711,293)는 2020년 3월에 만료됐다.

**Stage 1 — Scale-Space Extrema Detection (DoG)**:

이미지를 여러 스케일의 가우시안으로 흐리게 해 scale-space를 만든다.

$$L(x, y, \sigma) = G(x, y, \sigma) * I(x, y)$$

여기서 $G(x, y, \sigma) = \frac{1}{2\pi\sigma^2} \exp\left(-\frac{x^2 + y^2}{2\sigma^2}\right)$.

Laplacian of Gaussian(LoG)을 빠르게 근사하려고 Difference of Gaussians(DoG)를 쓴다.

$$D(x, y, \sigma) = L(x, y, k\sigma) - L(x, y, \sigma) \approx (k-1)\sigma^2 \nabla^2 G$$

DoG 이미지에서 같은 스케일의 이웃 8개와 위아래 스케일의 9개씩, 모두 26개와 비교해 극값(extrema)을 찾는다.

**Stage 2 — Keypoint Localization**:

scale-space에서 Taylor 전개로 극값 위치를 픽셀·스케일 간격보다 정밀하게 구한다.

$$D(\mathbf{x}) = D + \frac{\partial D}{\partial \mathbf{x}}^\top \mathbf{x} + \frac{1}{2} \mathbf{x}^\top \frac{\partial^2 D}{\partial \mathbf{x}^2} \mathbf{x}$$

극값의 정밀 위치: $\hat{\mathbf{x}} = -\frac{\partial^2 D}{\partial \mathbf{x}^2}^{-1} \frac{\partial D}{\partial \mathbf{x}}$

대비가 낮은 키포인트는 버린다: $|D(\hat{\mathbf{x}})| < 0.03$ (픽셀 값을 [0,1]로 둘 때).
에지 위의 불안정한 극값은 Hessian 고유값의 비로 가려내 버린다.

**Stage 3 — Orientation Assignment**:

키포인트 주변의 그래디언트 크기와 방향으로 36-bin 방향 히스토그램을 만든다(가우시안 가중, $\sigma = 1.5 \times$ 키포인트 스케일). 가장 높은 봉우리를 키포인트 방향으로 삼고, 그 80% 이상인 봉우리가 또 있으면 그 방향으로도 키포인트를 하나 더 만든다.

**Stage 4 — Keypoint Descriptor**:

키포인트 주변 $16 \times 16$ 영역을 $4 \times 4$ 블록 16개로 나누고, 블록마다 8-bin 방향 히스토그램을 만든다.

$$\text{Descriptor} = 4 \times 4 \times 8 = 128\text{-dimensional vector}$$

L2 정규화한 뒤 0.2를 넘는 값을 0.2로 자르고 다시 정규화한다. 그러면 비선형 조명 변화로 일부 그래디언트만 커져도 영향이 줄어든다.

```python
# SIFT Detection & Description
sift = cv2.SIFT_create(nfeatures=2000)
keypoints, descriptors = sift.detectAndCompute(img, None)
# descriptors.shape: (N, 128) — 128-dimensional float32
```

#### SURF (Speeded-Up Robust Features, 2006)

SURF는 [Bay et al. (2006)](https://link.springer.com/chapter/10.1007/11744023_32)이 SIFT보다 빠른 검출기·디스크립터로 제안했다. 계산량은 다음 방법으로 줄인다.

- **Integral image**로 박스 필터를 계산해 가우시안 2차 미분을 근사한다. 박스 크기와 상관없이 필터 한 번에 상수 시간이 든다.
- DoG 대신 **Hessian determinant**로 특징점을 고른다.

$$\det(\mathbf{H}) = D_{xx} D_{yy} - (0.9 \cdot D_{xy})^2$$

- 디스크립터는 Haar wavelet 응답의 합과 절댓값 합으로 만든 64차원으로, SIFT(128차원)의 절반이다.

SIFT와의 속도·정확도 차이는 구현, 영상 크기, 하드웨어에 따라 다르다.

SURF는 특허 때문에 OpenCV에서 non-free(`xfeatures2d`) 모듈로 분리되어 있다. 이 가이드에서 다루는 시스템 가운데 ORB-SLAM 계열은 ORB를, COLMAP은 SIFT를 기본 특징점으로 쓴다.

### 5.2.3 Binary Descriptors: BRIEF, ORB, BRISK

이진 디스크립터는 패치 안 점 쌍의 밝기를 비교해 0/1 비트열을 만든다. Hamming distance는 XOR와 popcount로 계산하므로 부동소수점 거리보다 대개 싸다. 실제로 몇 배 빠른지는 디스크립터 길이, 라이브러리, 하드웨어에 따라 다르다.

| 디스크립터 | 비트 수 | 특징 |
|-----------|--------|------|
| BRIEF | 128/256/512 | 랜덤 점 쌍, 회전 불변 아님 |
| ORB | 256 | 학습된 점 쌍, 회전 불변 |
| BRISK | 512 | 동심원 샘플링, 스케일 불변 |

BRIEF의 비교 연산은 다음과 같다.

$$b_i = \begin{cases} 1 & \text{if } I(\mathbf{p}_i) < I(\mathbf{q}_i) \\ 0 & \text{otherwise} \end{cases}, \quad \text{descriptor} = \sum_{1 \le i \le n} 2^{i-1} b_i$$

popcount는 워드 단위 명령이라, 256비트 디스크립터는 64비트 워드 4개를 XOR·popcount해 더하고 512비트는 그 두 배를 한다.

$$d_H(\mathbf{a}, \mathbf{b}) = \text{popcount}(\mathbf{a} \oplus \mathbf{b})$$

### 5.2.4 전통적 특징점의 계보

아래는 이 절의 방법을 발표 순서대로 늘어놓은 것이다.

```
Harris (1988)     — 코너 검출의 수학적 정의
    ↓ 스케일 불변성 필요
SIFT (2004)       — scale-space + 128D float descriptor (정확하지만 느림)
    ↓ 속도 개선
SURF (2006)       — integral image + 64D (속도 지향, float)
    ↓ 실시간 요구
FAST (2006)       — 극한 속도 검출 (descriptor 없음)
    ↓ 검출+기술 통합
ORB (2011)        — oFAST + rBRIEF, 256-bit binary (Hamming 매칭)
```

---

## 5.3 전통적 Matching & Outlier Rejection

디스크립터를 구한 뒤에는 어느 특징점 쌍이 같은 3D 점에서 왔는지 정해야 한다.

### 5.3.1 Brute-Force Matching

가장 단순한 방법이다. 이미지 A의 디스크립터 하나하나를 이미지 B의 모든 디스크립터와 비교해 가장 가까운 것과 짝짓는다.

- Float 디스크립터 (SIFT): L2 거리
- Binary 디스크립터 (ORB): Hamming 거리
- 시간 복잡도: $O(N \cdot M \cdot D)$ — $N, M$은 각 이미지의 특징점 수, $D$는 디스크립터 차원

```python
# Brute-Force Matching
bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)  # SIFT
# bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)  # ORB
matches = bf.match(desc1, desc2)
matches = sorted(matches, key=lambda x: x.distance)
```

`crossCheck=True`로 두면 A→B 최근접과 B→A 최근접이 서로 같을 때만 매칭으로 남긴다(mutual nearest neighbor).

### 5.3.2 FLANN (Fast Library for Approximate Nearest Neighbors)

디스크립터가 많으면 brute-force는 너무 느려서 근사 최근접 이웃 탐색(approximate nearest neighbor, ANN)을 쓴다. FLANN은 randomized kd-tree, hierarchical k-means tree 같은 알고리즘 가운데 데이터에 맞는 것과 파라미터를 자동으로 고를 수 있다.

```python
# FLANN Matching for SIFT
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)  # 탐색 노드 수 제한

flann = cv2.FlannBasedMatcher(index_params, search_params)
matches = flann.knnMatch(desc1, desc2, k=2)  # k=2 for ratio test
```

### 5.3.3 Lowe's Ratio Test

Lowe(2004)가 SIFT 논문에서 제안한 걸러내기 방법이다. 가장 가까운 디스크립터까지의 거리와 두 번째로 가까운 것까지의 거리 비가 임계값 이하일 때만 매칭으로 남긴다.

$$\frac{d(\mathbf{f}, \mathbf{f}_1)}{d(\mathbf{f}, \mathbf{f}_2)} < \tau$$

$\mathbf{f}_1, \mathbf{f}_2$는 각각 가장 가까운 디스크립터와 두 번째로 가까운 디스크립터다. $\tau$는 보통 0.7~0.8을 쓴다(Lowe는 0.8).

올바른 매칭이면 가장 가까운 후보가 두 번째 후보보다 확연히 가까워 $d_1 \ll d_2$가 된다. 잘못된 매칭이면 비슷한 후보가 여럿이라 $d_1 \approx d_2$가 되기 쉽다.

```python
# Ratio test (Lowe's)
good_matches = []
for m, n in matches:  # m: best, n: second best
    if m.distance < 0.75 * n.distance:
        good_matches.append(m)
```

### 5.3.4 RANSAC 계열: RANSAC → PROSAC → MAGSAC++

매칭을 걸러도 아웃라이어(잘못된 매칭)는 남는다. RANSAC 계열은 Fundamental matrix나 Essential matrix 같은 기하 모델을 추정하면서 아웃라이어를 함께 걸러낸다.

#### RANSAC (Random Sample Consensus, 1981)

[Fischler & Bolles (1981)](https://dl.acm.org/doi/10.1145/358669.358692)의 RANSAC은 다음 절차로 아웃라이어에 강한 추정을 한다.

1. 매칭 가운데 모델에 필요한 최소 개수 $n$개를 무작위로 뽑는다(Fundamental matrix는 8점 또는 7점, Essential matrix는 5점)
2. 뽑은 점으로 모델을 추정한다
3. 모델과의 오차가 임계값 $t$ 이내인 매칭(인라이어)을 모아 consensus set을 만든다
4. 1~3을 반복해 consensus set이 가장 큰 모델을 고른다
5. 그 인라이어 전체로 모델을 다시 추정한다

**필요한 반복 횟수**:

$$k = \frac{\log(1 - p)}{\log(1 - w^n)}$$

- $p$: 적어도 한 번은 아웃라이어 없는 표본을 뽑을 확률 (보통 0.99)
- $w$: 인라이어 비율
- $n$: 모델에 필요한 최소 점 수

인라이어가 50%이고 8점 모델에 $p=0.99$이면 $k \approx 1177$번 반복한다.

```python
# RANSAC으로 Fundamental matrix 추정
pts1 = np.float32([kp1[m.queryIdx].pt for m in good_matches])
pts2 = np.float32([kp2[m.trainIdx].pt for m in good_matches])

F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC,
                                  ransacReprojThreshold=3.0,
                                  confidence=0.99)
inlier_matches = [m for m, flag in zip(good_matches, mask.ravel()) if flag]
```

#### PROSAC (Progressive Sample Consensus, 2005)

Chum과 Matas가 CVPR 2005에서 제안했다. PROSAC은 매칭을 품질(디스크립터 거리 등) 순으로 정렬하고, 상위 매칭에서 시작해 표본을 뽑는 범위를 점점 넓힌다.

품질 순위가 높은 매칭일수록 실제로 인라이어일 확률이 높다면, 맞는 모델을 RANSAC보다 일찍 찾는다. 순위가 인라이어 여부와 무관하면 이 이점도 사라진다.

#### MAGSAC / MAGSAC++ (2019/2020)

Barath 등이 제안했다(MAGSAC은 CVPR 2019, MAGSAC++는 CVPR 2020). RANSAC은 인라이어 임계값 $t$를 사람이 정해야 하는데, MAGSAC은 이 값을 하나로 고르지 않는다.

- 임계값 $\sigma$를 $[0, \sigma_{\max}]$ 구간에서 marginalize해 모델의 품질을 구한다.

$$Q(\theta) = \int_0^{\sigma_{\max}} q(\theta, \sigma) f(\sigma) d\sigma$$

$q(\theta, \sigma)$는 임계값이 $\sigma$일 때 모델 $\theta$의 품질이고, $f(\sigma)$는 임계값의 사전 분포다.

- MAGSAC++는 같은 생각을 더 빠르게 계산하는 가중치 함수를 쓰고, 이 가중치로 반복 가중 최소자승(IRLS) 피팅을 한다.
- 인라이어 판정 임계값을 고르지 않으므로 그 값에 덜 민감하다. 적분 상한 $\sigma_{\max}$는 여전히 정해야 한다. OpenCV의 `USAC_MAGSAC`은 이 상한을 `ransacReprojThreshold`와 모델별 내부 기본값(fundamental matrix는 2.5) 중 큰 값으로 잡고, `ransacReprojThreshold` 자체는 종료 판정과 반환 `mask`의 인라이어 판정에 쓴다.

```python
# OpenCV의 USAC (MAGSAC++ 포함)
F, mask = cv2.findFundamentalMat(
    pts1, pts2,
    cv2.USAC_MAGSAC,           # MAGSAC++ 사용
    ransacReprojThreshold=1.0,  # MAGSAC++는 이 값에 덜 민감하다
    confidence=0.999,
    maxIters=10000
)
```

### 5.3.5 Fundamental / Essential Matrix Estimation

2D-2D 매칭으로 두 카메라 사이의 기하 관계를 추정한다.

**Fundamental Matrix** ($\mathbf{F}$, 7 DOF):
- 내부 파라미터를 모를 때 쓴다
- 8-point algorithm: 대응점 8개 이상으로 선형 시스템을 풀고 rank-2 제약을 건다
- 7-point algorithm: 대응점 7개로 3차 다항식을 풀어 해를 최대 3개 얻는다

**Essential Matrix** ($\mathbf{E}$, 5 DOF):
- 내부 파라미터를 알 때 쓴다
- $\mathbf{E} = [\mathbf{t}]_\times \mathbf{R}$, $\mathbf{E} = \mathbf{K}_2^\top \mathbf{F} \mathbf{K}_1$
- 5-point algorithm (Nistér, 2004): 대응점 5개로 10차 다항식을 풀어 해를 최대 10개 얻는다

실제로는 RANSAC과 함께 쓴다.

```python
# Essential matrix 추정 (calibrated camera)
E, mask = cv2.findEssentialMat(pts1, pts2, cameraMatrix=K,
                                method=cv2.RANSAC,
                                prob=0.999, threshold=1.0)

# Essential matrix에서 R, t 복원
_, R, t, mask_pose = cv2.recoverPose(E, pts1, pts2, cameraMatrix=K)
```

### 5.3.6 로버스트 추정의 계보

```
Least Squares (아웃라이어에 취약)
    ↓ 아웃라이어 대응
RANSAC (1981)   — 무작위 표본 기반 robust estimation의 대표적 출발점
    ↓ 사전 정보 활용
PROSAC (2005)   — 매칭 품질 기반 진행적 샘플링
    ↓ 임계값 의존 줄이기
MAGSAC++ (2020) — 임계값 하나를 고르지 않고 여러 σ에 걸쳐 가중하는 로버스트 추정
    ↓ 학습 기반 제거
GeoTransformer (2022) — RANSAC 없이 직접 변환 추정
```

---

## 5.4 Mutual Information & Intensity-Based Registration

### 5.4.1 MI의 정의와 직관

**Mutual Information(MI, 상호 정보량)**은 한 확률 변수를 알면 다른 변수의 불확실성이 얼마나 줄어드는지 재는 정보 이론의 척도다.

두 확률 변수 $X, Y$의 mutual information은 다음과 같다.

$$I(X; Y) = \sum_{x \in X} \sum_{y \in Y} p(x, y) \log \frac{p(x, y)}{p(x) p(y)}$$

연속 변수라면 다음과 같다.

$$I(X; Y) = \int \int p(x, y) \log \frac{p(x, y)}{p(x) p(y)} \, dx \, dy$$

엔트로피(entropy)로 쓰면 다음과 같다.

$$I(X; Y) = H(X) + H(Y) - H(X, Y)$$

여기서:
- $H(X) = -\sum_x p(x) \log p(x)$: $X$의 엔트로피
- $H(X, Y) = -\sum_{x,y} p(x, y) \log p(x, y)$: 결합 엔트로피

**직관**: 두 이미지가 제대로 정렬(aligned)되어 있으면 한 이미지의 픽셀 값으로 같은 위치의 다른 이미지 픽셀 값을 더 잘 예측할 수 있고, 이때 $I(X; Y)$가 커진다. 정렬이 어긋나면 이 관계가 흐려져 $I(X; Y)$가 줄어든다.

MI는 비선형 의존성까지 재므로 선형 관계만 보는 상관계수(correlation coefficient)가 놓치는 관계도 잡는다.

### 5.4.2 MI 기반 다중 모달리티 정합

MI로 종류가 다른(multimodal) 센서 데이터를 정합할 수 있다. 1995년 무렵 Viola & Wells와 Collignon·Maes 등이 의료 영상의 CT-MRI 정합에 처음 썼다.

MI가 서로 다른 모달리티에서도 통하는 이유는 다음과 같다.

- 같은 물체를 찍어도 CT와 MRI는 밝기 분포가 전혀 다르다. 뼈는 CT에서 밝고 MRI에서는 어두울 수 있다.
- 밝기 차이(SSD)와 상관관계(NCC)는 선형 밝기 관계를 가정한다.
- MI는 밝기 값 사이의 통계적 의존성만 보므로, 두 밝기의 관계가 단조(monotonic)든 아니든 쓸 수 있다.

로보틱스에서는 카메라-LiDAR 정합에 쓴다. LiDAR intensity와 카메라 밝기는 서로 다른 물리량이지만, 반사가 강한 표지판이나 차선처럼 같은 표면에서 함께 밝아지는 경우가 많아 정렬이 맞으면 MI가 커진다.

### 5.4.3 MI와 NMI의 실용적 계산

이미지 정합에서는 확률 분포 $p(x), p(y), p(x, y)$를 결합 히스토그램(joint histogram)으로 추정한다.

이미지 $A$와 $B$를 변환 $T$로 맞춘 상태에서 다음 순서로 계산한다.
1. 겹치는 위치 $(u, v)$마다 $A(u, v)$와 $B(T(u, v))$의 밝기 쌍을 모은다.
2. 2D 히스토그램으로 결합 분포 $p(a, b)$를 추정한다.
3. 주변 분포 $p(a), p(b)$는 히스토그램의 행 합과 열 합이다.
4. MI를 계산한다.

**Normalized Mutual Information(NMI)**은 MI를 정규화해 겹치는 영역 크기의 영향을 줄인다.

$$NMI(A, B) = \frac{H(A) + H(B)}{H(A, B)}$$

다른 정의는 다음과 같다.

$$NMI(A, B) = \frac{2 \cdot I(A; B)}{H(A) + H(B)}$$

겹치는 영역이 바뀌는 정합에서는 MI보다 NMI를 많이 쓴다(Studholme et al., 1999).

### 5.4.4 MI Gradient 계산

MI를 목적 함수로 그래디언트 기반 최적화를 하려면 변환 파라미터에 대한 MI의 그래디언트가 필요하다.

변환 $T_\xi$ (파라미터 $\xi$)에 대한 MI의 그래디언트는 다음과 같다.

$$\frac{\partial I}{\partial \xi} = \sum_{a, b} \frac{\partial p(a, b)}{\partial \xi} \left(1 + \log \frac{p(a, b)}{p(a) p(b)}\right)$$

보통의 히스토그램은 계단 함수라 미분할 수 없으므로, Parzen 윈도우(커널 밀도 추정)나 B-spline으로 미분 가능한 히스토그램을 만든다.

그래디언트 없이 푸는(gradient-free) Nelder-Mead simplex도 쓴다. Koide et al.(2023)의 캘리브레이션 툴이 NID(§5.4.5)를 Nelder-Mead로 최소화한다.

### 5.4.5 왜 Calibration에서 MI가 쓰이는가

Ch.3의 targetless 카메라-LiDAR 캘리브레이션은 MI나 NMI, NID를 비용 함수로 쓴다. 절차는 다음과 같다.

1. LiDAR 포인트 클라우드를 현재 외부 파라미터 추정치로 카메라 이미지에 투영한다
2. 투영한 LiDAR intensity와 카메라 픽셀 밝기의 MI를 계산한다
3. MI가 커지도록 외부 파라미터를 고친다

**Normalized Information Distance(NID)**는 MI로 정의한 거리 척도다.

$$NID(A, B) = 1 - \frac{I(A; B)}{H(A, B)} = \frac{H(A|B) + H(B|A)}{H(A, B)}$$

NID는 0(완전 의존)에서 1(완전 독립) 사이 값이고, 거리 함수(metric)의 조건을 만족한다.

```python
import numpy as np
from sklearn.metrics import mutual_info_score

def compute_mi(img_a, img_b, bins=256):
    """두 이미지의 Mutual Information 계산."""
    # Joint histogram
    hist_2d, _, _ = np.histogram2d(
        img_a.ravel(), img_b.ravel(), bins=bins
    )
    # Joint probability
    pxy = hist_2d / hist_2d.sum()
    px = pxy.sum(axis=1)
    py = pxy.sum(axis=0)

    # MI = H(X) + H(Y) - H(X,Y)
    hx = -np.sum(px[px > 0] * np.log(px[px > 0]))
    hy = -np.sum(py[py > 0] * np.log(py[py > 0]))
    hxy = -np.sum(pxy[pxy > 0] * np.log(pxy[pxy > 0]))

    return hx + hy - hxy

def compute_nid(img_a, img_b, bins=256):
    """Normalized Information Distance 계산."""
    mi = compute_mi(img_a, img_b, bins)
    hist_2d, _, _ = np.histogram2d(
        img_a.ravel(), img_b.ravel(), bins=bins
    )
    pxy = hist_2d / hist_2d.sum()
    hxy = -np.sum(pxy[pxy > 0] * np.log(pxy[pxy > 0]))

    return 1.0 - mi / hxy if hxy > 0 else 1.0
```

---

## 5.5 학습 기반 Feature Detection & Description

전통적 특징점은 밝기 그래디언트, 코너, 블롭 같은 저수준 단서를 쓰므로 조명, 시점, 날씨가 크게 바뀌면 잘 맞지 않는다. 학습 기반 방법은 많은 데이터에서 이런 변화에 덜 흔들리는 특징을 학습한다.

### 5.5.1 SuperPoint (2018): 검출과 기술을 함께 자기지도 학습

[DeTone et al. (2018)](https://arxiv.org/abs/1712.07629)의 SuperPoint는 키포인트 검출과 디스크립터 추출을 단일 fully convolutional network에서 함께 계산하고, Homographic Adaptation으로 자기지도 학습한다.

#### Homographic Adaptation 학습 전략

SuperPoint는 사람이 단 라벨 없이 키포인트 검출기를 학습한다. 순서는 다음과 같다.

1. 이미지 하나에 무작위 호모그래피를 여러 개(논문은 100개) 적용한다
2. 변환한 이미지마다 키포인트를 검출한다
3. 검출 결과를 원래 좌표로 되돌려 모은다(aggregate)
4. 모은 결과에서 반응이 강한 점을 pseudo ground-truth로 삼는다

#### 2단계 학습: MagicPoint → SuperPoint

- **1단계**: 삼각형, 사각형, 선분 같은 합성 도형으로 만든 Synthetic Shapes 데이터셋에서 코너·접합점 검출기 MagicPoint를 먼저 학습한다.
- **2단계**: MS-COCO 실제 이미지에 MagicPoint와 Homographic Adaptation을 적용해 pseudo ground-truth를 만들고, 이것으로 SuperPoint를 학습한다.

#### 아키텍처

VGG 스타일 인코더 하나를 공유하고, 그 뒤에 디코더 헤드가 둘 붙는다.

**Interest Point Decoder**:
- 입력 이미지를 8×8 픽셀 셀로 나눈다
- 셀마다 65채널(셀 안의 위치 64개와 '키포인트 없음' 1개)에 softmax를 적용한다
- 결과를 펼쳐 픽셀 단위 키포인트 히트맵을 만든다

**Descriptor Decoder**:
- 공유 인코더의 특징 맵에서 256차원 디스크립터 맵(1/8 해상도)을 낸다
- 키포인트 위치의 값을 bicubic interpolation으로 뽑는다
- L2 정규화한다

#### 학습 손실

- 키포인트 검출: cross-entropy loss
- 디스크립터: 호모그래피로 대응을 알므로 positive/negative 쌍에 hinge loss를 건다.

$$L_{desc} = \sum_{(i,j) \in \text{pos}} \max(0, m_p - \mathbf{d}_i^\top \mathbf{d}_j) + \sum_{(i,j) \in \text{neg}} \max(0, \mathbf{d}_i^\top \mathbf{d}_j - m_n)$$

$m_p, m_n$은 각각 positive, negative margin이다.

**논문 보고 속도**: 한 번의 forward pass로 검출과 기술을 함께 계산한다. 원 논문은 Titan X GPU에서 640×480 영상을 약 70 FPS로 처리했다고 보고했다. 실제 처리량은 쓰는 하드웨어와 키포인트 수로 직접 측정해야 한다.

```python
import torch
# SuperPoint 사용 예제 (cvg/LightGlue 패키지. kornia.feature에는 SuperPoint 클래스가 없다)
from lightglue import SuperPoint

# 모델 로드
sp = SuperPoint(max_num_keypoints=2048)
sp = sp.eval()

# 추론
with torch.no_grad():
    img_tensor = torch.from_numpy(img).float().unsqueeze(0).unsqueeze(0) / 255.0
    pred = sp({'image': img_tensor})   # 반환 키: keypoints, keypoint_scores, descriptors
    keypoints = pred['keypoints']        # (1, N, 2)
    descriptors = pred['descriptors']    # (1, N, 256)
    scores = pred['keypoint_scores']              # (1, N)
```

### 5.5.2 D2-Net (2019): Detect-and-Describe Jointly

[D2-Net (Dusmanu et al., 2019)](https://arxiv.org/abs/1905.03561)은 CNN 하나가 디스크립터 맵이자 검출기 역할을 한다. SuperPoint는 검출 헤드와 기술 헤드를 따로 두었는데, D2-Net은 같은 특징 맵에서 키포인트도 고르고 디스크립터도 뽑는다.

VGG16의 중간 특징 맵 $\mathbf{F} \in \mathbb{R}^{H \times W \times C}$를 다음처럼 쓴다.
- **Detection**: 위치마다 값이 가장 큰 채널을 고르고, 그 채널에서 그 위치의 값이 주변보다 크면(공간 NMS) 키포인트로 고른다
- **Description**: 같은 위치의 $C$차원 벡터를 디스크립터로 쓴다

깊은 층의 특징을 쓰므로 주야간 같은 큰 외관 변화에서도 매칭이 된다. 원 논문은 Aachen Day-Night와 InLoc localization 벤치마크에서 당시 최고 성능을 보고했다. 특징 맵이 입력의 1/4 해상도라 키포인트 위치 정밀도가 그 간격에 묶이고, 논문은 국소 정제로 서브픽셀 위치를 보완한다.

### 5.5.3 R2D2 (2019): Reliable and Repeatable Detector-Descriptor

[R2D2 (Revaud et al., 2019)](https://arxiv.org/abs/1906.06195)는 반복성(repeatability)과 신뢰성(reliability)을 따로 학습한다.

- **Repeatability**: 시점이 달라도 같은 점이 검출되는 정도
- **Reliability**: 그 점의 디스크립터로 매칭을 구별해 낼 수 있는 정도. 하늘처럼 텍스처가 없는 곳은 반복적으로 검출돼도 디스크립터로 구별이 안 된다

R2D2는 두 값을 각각 confidence map으로 예측하고, 두 맵을 곱한 값으로 키포인트를 고른다.

### 5.5.4 DISK (2020)

[DISK (Tyszkiewicz et al., 2020)](https://arxiv.org/abs/2006.13566)는 강화학습(reinforcement learning)의 policy gradient로 특징을 학습한다. 매칭이 맞으면 보상(reward), 틀리면 벌점(penalty)을 준다.

이렇게 맞는 매칭 수를 기준으로 검출과 기술을 end-to-end 학습한다.

### 5.5.5 학습 기반 특징점이 나은 경우

학습 기반 특징점이 전통 방법보다 나은 경우는 다음과 같다. 표의 근거는 D2-Net 논문의 Aachen Day-Night 결과(§5.5.2)다. 시점 변화, 반복 패턴, 모션 블러에서 어느 쪽이 나은지는 이 장에서 인용한 논문들이 직접 비교하지 않았다.

| 시나리오 | SIFT/ORB 한계 | SuperPoint/D2-Net 개선 |
|---------|-------------|---------------------|
| 극한 조명 변화 (주야간) | DoG/그래디언트 기반 검출 실패 | 학습된 특징이 고수준 구조를 잡음 |

학습 기반 방법은 학습 데이터의 분포에 영향을 받는다. 수중처럼 학습 데이터에 없던 환경에서는 성능이 떨어질 수 있다.

---

## 5.6 학습 기반 Feature Matching

### 5.6.1 SuperGlue (2020): Attention 기반 매칭

[Sarlin et al. (2020)](https://arxiv.org/abs/1911.11763)의 SuperGlue는 그래프 신경망(GNN), 어텐션, optimal transport로 키포인트를 짝짓고 짝 없는 점을 걸러내는 학습 기반 matcher다.

#### 문제 정의: 부분 할당(Partial Assignment)

두 이미지에서 뽑은 키포인트 집합 $\mathcal{A} = \{(\mathbf{p}_i, \mathbf{d}_i)\}_{i=1}^{N}$과 $\mathcal{B} = \{(\mathbf{p}_j, \mathbf{d}_j)\}_{j=1}^{M}$ 사이의 대응을 찾는다. 짝이 없는 키포인트는 '매칭 없음'을 뜻하는 가상 노드(dustbin)에 보낸다.

#### Keypoint Encoder

키포인트 위치 $(x, y)$와 검출 confidence $c$를 MLP로 임베딩해 디스크립터 벡터에 더한다.

$$\mathbf{f}_i^{(0)} = \mathbf{d}_i + \text{MLP}_{\text{enc}}([\mathbf{p}_i, c_i])$$

그러면 특징 벡터에 위치 정보가 들어간다.

#### Attentional Graph Neural Network

키포인트를 그래프의 노드로 두고 Self-Attention과 Cross-Attention을 번갈아 적용한다.

**Self-Attention (intra-image)**: 같은 이미지 안의 키포인트끼리 정보를 주고받는다.

$$\text{message}_i^{\text{self}} = \sum_{j \in \mathcal{A}} \text{softmax}\left(\frac{\mathbf{q}_i^\top \mathbf{k}_j}{\sqrt{d}}\right) \mathbf{v}_j$$

**Cross-Attention (inter-image)**: 상대 이미지의 키포인트에서 정보를 받아, 어느 키포인트와 비슷한지를 반영한다.

$$\text{message}_i^{\text{cross}} = \sum_{j \in \mathcal{B}} \text{softmax}\left(\frac{\mathbf{q}_i^\top \mathbf{k}_j}{\sqrt{d}}\right) \mathbf{v}_j$$

이 쌍을 $L$번(논문은 9번) 되풀이한다.

#### Optimal Transport를 이용한 매칭

GNN 출력끼리의 내적으로 매칭 점수 행렬을 만들고 Sinkhorn 알고리즘을 적용한다. Sinkhorn은 Hungarian algorithm이 푸는 할당 문제를 미분 가능하게 근사한다.

dustbin까지 넣은 점수 행렬 $\mathbf{S} \in \mathbb{R}^{(N+1) \times (M+1)}$은 다음과 같다.

$$S_{ij} = \langle \mathbf{f}_i^{(L)}, \mathbf{f}_j^{(L)} \rangle, \quad S_{i,M+1} = S_{N+1,j} = z$$

$z$는 학습하는 dustbin 점수다.

Sinkhorn 정규화를 (논문은 100번) 반복한다.

$$\mathbf{S} \leftarrow \text{row-normalize}(\mathbf{S}), \quad \mathbf{S} \leftarrow \text{col-normalize}(\mathbf{S})$$

그 결과인 soft assignment 행렬에서 임계값을 넘고 서로 최댓값인 항을 매칭으로 고른다.

#### 학습

호모그래피나 상대 포즈·깊이 맵으로 만든 정답 대응의 negative log-likelihood를 최소화해 end-to-end로 학습한다.

$$L = -\sum_{(i,j) \in \mathcal{M}} \log \hat{P}_{ij} - \sum_{i \in \mathcal{U}_A} \log \hat{P}_{i, M+1} - \sum_{j \in \mathcal{U}_B} \log \hat{P}_{N+1, j}$$

$\mathcal{M}$은 정답 매칭 쌍, $\mathcal{U}_A, \mathcal{U}_B$는 짝이 없는 키포인트다.

```python
# SuperGlue 사용 예제 (hloc)
from hloc import match_features, extract_features
from hloc.utils.io import list_h5_names

# SuperPoint 특징 추출
feature_conf = extract_features.confs['superpoint_aachen']
features_path = extract_features.main(feature_conf, images_dir)

# SuperGlue 매칭
match_conf = match_features.confs['superglue']
matches_path = match_features.main(match_conf, pairs_path, feature_conf['output'], features_path)
```

#### 계보상의 위치

SuperPoint로 검출과 기술을, SuperGlue로 매칭을 학습하면 파이프라인의 세 단계가 모두 학습 기반이 된다. 검출, 기술, 매칭을 차례로 거치는 구조는 그대로 남는다. LoFTR(§5.7.2)는 검출 단계를 없앴다.

### 5.6.2 LightGlue (2023): SuperGlue의 효율화

[Lindenberger et al. (2023)](https://arxiv.org/abs/2306.13643)의 LightGlue는 SuperGlue의 설계를 다시 살펴 고친 matcher로, 입력의 난이도에 따라 계산량을 바꾼다(adaptive computation).

#### SuperGlue의 문제점 진단

- 공개된 SuperGlue는 GNN 층 수와 Sinkhorn 반복 수가 고정이라, 쉬운 입력에도 계산량이 같다.
- 층마다 키포인트 수 $N$에 대해 $O(N^2)$인 attention을 계산하므로 키포인트가 많으면 빠르게 느려진다.

#### 핵심 개선: Adaptive Depth & Width

**Adaptive Depth (레이어 조기 종료)**:
- 층마다 가벼운 classifier(MLP)가 각 점의 예측이 확정됐는지를 판단한다.
- 확정된 점의 비율이 기준을 넘으면 거기서 멈춘다.
- 논문 그림 2의 예에서는 쉬운 쌍이 3층, 어려운 쌍이 8층에서 멈췄다(전체 9층).

**Adaptive Width (키포인트 pruning)**:
- 짝이 없다고 판단한 키포인트는 중간 층에서 뺀다.
- 그만큼 attention에 들어가는 점이 줄어 뒤쪽 층의 계산이 가벼워진다.

**Sinkhorn 제거**:
Optimal Transport 대신 매칭 가능성 점수와 dual-softmax로 할당 행렬을 만들어 Sinkhorn 반복을 없앴다.

$$P_{ij} = \text{softmax}_j(S_{ij}) \cdot \text{softmax}_i(S_{ij})$$

논문의 식에는 여기에 두 점 각각의 매칭 가능성 $\sigma_i \sigma_j$가 곱해진다.

#### 학습 전략

- 학습할 때는 모든 층을 통과시키고 층마다 출력에 손실을 건다(deep supervision).
- 조기 종료는 추론할 때만 쓴다.
- SuperPoint, DISK, ALIKED 같은 여러 로컬 특징과 함께 쓸 수 있다.

**성능**: LightGlue 논문은 SuperGlue와 비슷한 정확도에서 처리량이 2.5배, 효율적인 self-attention 구현을 쓴 변형은 4배라고 보고했다. 쉬운 영상 쌍일수록 더 빨라진다. 실제 배율은 키포인트 수와 하드웨어에 따라 다르다.

```python
# LightGlue 사용 예제 (kornia / hloc)
from kornia.feature import LightGlue as KorniaLightGlue

# SuperPoint + LightGlue 조합
lg = KorniaLightGlue(features='superpoint')
lg = lg.eval()

with torch.no_grad():
    # pred0, pred1: SuperPoint 출력
    matches = lg({'image0': pred0, 'image1': pred1})
    # matches['matches']: (K, 2) — 매칭된 키포인트 인덱스 쌍
    # matches['scores']: (K,) — 매칭 신뢰도
```

### 5.6.3 Detect → Describe → Match 파이프라인과 학습 기반 방법

```
[전통적 파이프라인]
SIFT detect → SIFT descriptor → BF/FLANN + ratio test
                                        ↓
[학습 기반 대체]
SuperPoint detect+describe → SuperGlue attention matching → LightGlue 효율화
    (2018)                       (2020)                        (2023)
```

단계를 나누면 모듈마다 따로 분석하고 바꿔 끼우기 쉽다. 첫 단계에서 키포인트를 못 찾으면 기술과 매칭도 할 수 없다. §5.7의 detector-free 방법은 검출 단계를 없앤다.

---

## 5.7 Detector-Free Matching

### 5.7.1 왜 Detector-Free가 필요한가

Detect-then-match 파이프라인은 검출기가 찾은 키포인트끼리만 매칭한다. 다음 환경에서 이 제약이 문제가 된다.

- **텍스처 없는 영역**: 흰 벽, 콘크리트 바닥, 하늘은 그래디언트가 약해 코너나 블롭이 검출되지 않는다
- **반복 패턴**: 타일이나 창문 격자는 검출은 되는데 디스크립터가 비슷해 매칭이 모호하다
- **넓은 시점 변화**: 시점 차이가 크면 같은 곳의 로컬 패치도 전혀 다르게 보인다

Detector-free 방법은 검출 단계 없이 이미지의 모든 위치를 매칭 후보로 둔다.

### 5.7.2 LoFTR (2021): Coarse-to-Fine Transformer Matching

[Sun et al. (2021)](https://arxiv.org/abs/2104.00680)의 LoFTR는 검출기 없이 두 이미지를 촘촘하게 매칭하는 트랜스포머 기반 방법이다.

#### 아키텍처

**1. Local Feature CNN**: ResNet-18 기반 FPN(Feature Pyramid Network)으로 두 이미지에서 coarse(1/8 해상도)와 fine(1/2 해상도) 특징 맵을 뽑는다.

**2. Coarse-Level Matching (Transformer)**:

1/8 특징 맵을 1차원 시퀀스로 펴고(flatten), Self-Attention과 Cross-Attention을 번갈아 $N$번(논문은 4번) 적용한다.

위치 정보는 2D sinusoidal positional encoding으로 넣는다.

출력 특징끼리의 내적으로 score matrix를 만들고, dual-softmax로 confidence matrix를 구한다.

$$P_{ij} = \text{softmax}_j(\mathbf{f}_i^A \cdot \mathbf{f}_j^B) \cdot \text{softmax}_i(\mathbf{f}_i^A \cdot \mathbf{f}_j^B)$$

confidence가 임계값을 넘고 mutual nearest neighbor인 쌍을 coarse 매칭으로 고른다.

**3. Fine-Level Refinement**:

coarse 매칭마다 1/2 해상도 특징 맵에서 그 위치 주변 $w \times w$ 윈도우(논문은 5×5)를 잘라 낸다. 윈도우 안에서 다시 상관을 계산해 서브픽셀 매칭 위치를 회귀(regression)한다.

#### Linear Transformer

계산을 줄이려고 standard softmax attention 대신 kernel 기반 linear attention을 쓴다.

Standard attention: $O(N^2)$, $\text{Attn}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{softmax}(\mathbf{Q}\mathbf{K}^\top / \sqrt{d}) \mathbf{V}$

Linear attention: $O(N)$, $\text{Attn}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \phi(\mathbf{Q})(\phi(\mathbf{K})^\top \mathbf{V})$

$\phi(\cdot)$는 커널 함수 $\text{elu}(\cdot)+1$이다. 행렬 곱 순서를 바꾸면 복잡도가 $O(N)$이 된다.

linear attention은 softmax attention을 근사하므로 표현력이 떨어질 수 있다.

#### 학습

정답 포즈와 깊이 맵으로 만든 대응점으로 학습한다.
- Coarse level: cross-entropy loss
- Fine level: L2 regression loss
- 데이터셋: ScanNet(실내), MegaDepth(실외)

```python
# LoFTR 사용 예제 (kornia)
from kornia.feature import LoFTR as KorniaLoFTR

loftr = KorniaLoFTR(pretrained='outdoor')
loftr = loftr.eval()

with torch.no_grad():
    input_dict = {
        'image0': img0_tensor,  # (1, 1, H, W) grayscale
        'image1': img1_tensor,
    }
    result = loftr(input_dict)

    mkpts0 = result['keypoints0']     # (K, 2) — 이미지 0의 매칭 좌표
    mkpts1 = result['keypoints1']     # (K, 2) — 이미지 1의 매칭 좌표
    confidence = result['confidence'] # (K,) — 매칭 신뢰도
```

#### 계보상의 위치

LoFTR는 detect-then-match 파이프라인에서 검출기를 뺐다. coarse 단계에서 한 이미지의 모든 위치가 cross-attention으로 상대 이미지의 모든 위치를 참조하므로, 검출 단계 없이 매칭한다.

### 5.7.3 QuadTree Attention: LoFTR의 효율화

LoFTR의 coarse 트랜스포머는 이미지의 모든 위치를 시퀀스로 펴므로, 해상도가 높아지면 시퀀스가 빠르게 길어진다. QuadTree Attention(Tang et al., ICLR 2022)은 attention을 계층적으로 계산한다.

순서는 다음과 같다.

1. 가장 거친(coarsest) 해상도에서 전체 attention을 계산한다
2. attention 점수가 높은 영역 몇 개만 고른다
3. 고른 영역 안에서만 다음 해상도의 attention을 계산한다
4. 이를 가장 세밀한 해상도까지 반복한다

그러면 standard attention의 $O(N^2)$ 비용이 $O(N)$으로 준다. LoFTR의 linear attention도 이미 $O(N)$이므로, 원 논문은 비슷한 연산량에서 정확도가 더 높다는 점을 내세운다.

### 5.7.4 ASpanFormer (2022): Adaptive Span Attention

ASpanFormer(Chen et al., ECCV 2022)는 위치마다 attention 범위(span)를 다르게 정한다.

- 텍스처가 많은 영역: 좁은 범위에서 정밀하게 매칭한다
- 텍스처가 적은 영역: 넓은 범위의 주변 정보를 쓴다

논문은 범위를 위치마다 추정한 flow와 그 불확실성으로 정한다.

### 5.7.5 RoMa (2024): DINOv2 + Dense Matching

[Edstedt et al. (2024)](https://arxiv.org/abs/2305.15404)의 RoMa는 DINOv2 특징으로 dense matching을 한다.

#### Foundation Model 활용

RoMa는 사전학습한 DINOv2 ViT-Large를 가중치를 고정(frozen)한 채 coarse 특징 추출기로 쓴다. LoFTR는 특징 추출기를 처음부터(scratch) 학습했다.

DINOv2는 대규모 이미지로 자기지도 학습한 범용 시각 특징이다. RoMa 논문은 이 특징이 처음부터 학습한 로컬 특징보다 훨씬 강건하지만 해상도가 거칠다고 보고, 세밀한 위치는 별도의 CNN 특징으로 잡는다.

#### 아키텍처

**1. Frozen DINOv2 Backbone**: 고정한 DINOv2 ViT-Large에서 1/14 해상도의 patch 특징을 뽑는다.

**2. Coarse Matching (Warp Estimation)**:
- DINOv2 특징에 transformer match decoder를 적용한다.
- 이미지 A의 위치마다 이미지 B의 대응 위치를 anchor 위치들에 대한 확률로 예측한다.

$$p(\mathbf{x}_B | \mathbf{x}_A) = \sum_{k} w_k \mathcal{N}(\mathbf{x}_B; \mu_k, \Sigma_k)$$

점 하나 대신 분포를 예측하므로 대응 후보가 여럿인 경우(multimodality)도 나타낼 수 있다.

**3. Fine Matching (Iterative Refinement)**:
- coarse warp에서 출발해, CNN(VGG19)으로 뽑은 fine 특징으로 warp를 여러 번 다듬는다.
- 단계마다 해상도를 높이고, 이전 warp 주변에서 local correlation을 계산해 잔차를 예측한다.

**4. Certainty Estimation**: 매칭마다 신뢰도(certainty)도 함께 예측하므로, 뒤 단계에서 신뢰도가 높은 매칭만 골라 쓸 수 있다.

#### 손실 함수: regression-by-classification과 robust regression

RoMa는 매칭 위치를 L2 loss로 바로 회귀하지 않는다. coarse 단계에서는 정답 위치가 속한 anchor를 맞히는 분류 손실을 쓰고(regression-by-classification), fine 단계에서는 큰 오차에 덜 민감한 robust regression 손실을 쓴다. coarse 단계의 분류 손실은 다음 꼴이다.

$$L = -\sum_{(\mathbf{x}_A, \mathbf{x}_B^*)} \log p(\mathbf{x}_B^* | \mathbf{x}_A)$$

#### 성능

RoMa 논문은 MegaDepth와 ScanNet의 relative pose 실험에서 비교한 LoFTR·ASpanFormer 계열보다 높은 AUC를 보고했다. 시점과 외관 변화가 큰 WxBS 벤치마크에서는 36% 향상을 보고했다.

```python
# RoMa 사용 예제
from romatch import roma_outdoor

# 모델 로드 (DINOv2 backbone 포함)
roma_model = roma_outdoor(device='cuda')
roma_model.eval()

# 매칭
warp, certainty = roma_model.match(img0_path, img1_path)

# 고신뢰 매칭만 추출
matches, certainty_scores = roma_model.sample(
    warp, certainty,
    num=5000  # 최대 매칭 수
)
# matches: (K, 4) — [x0, y0, x1, y1] 정규화 좌표
```

### 5.7.6 3D-Aware Dense Matching

다음 방법들은 3D 기하를 직접 예측하고, 그 결과에서 대응점을 얻는다.

**DUSt3R (Wang et al., 2024)**: [DUSt3R](https://arxiv.org/abs/2312.14132)는 캘리브레이션이나 포즈 정보 없이 이미지 쌍에서 바로 3D pointmap을 회귀한다. 2D 매칭 뒤 3D를 복원하는 대신, 3D 형상을 먼저 회귀하고 3D 공간에서 가까운 점끼리 대응으로 삼는다.

**MASt3R (Leroy et al., 2024)**: [MASt3R](https://arxiv.org/abs/2406.09756)는 DUSt3R에 dense local feature head를 더했다. 저자들은 Map-free localization 벤치마크에서 VCRE AUC가 기존 최고 방법보다 30%(절댓값) 높았다고 보고했다.

**VGGT (Wang et al., 2025)**: [VGGT](https://arxiv.org/abs/2503.11651) (Visual Geometry Grounded Transformer)는 CVPR 2025 Best Paper를 받았다. 이미지 한 장에서 수백 장까지 받아 camera parameter, pointmap, depth map, 3D point track을 한 번의 feed-forward pass로 추론한다. 원 논문은 재구성을 1초 안에 끝내면서 후처리 최적화를 쓰는 방법보다 여러 3D 과제에서 성능이 높았다고 보고했다.

---

## 5.8 3D-3D Correspondence

3D 포인트 클라우드 사이의 대응점은 LiDAR scan-to-scan 정합, multi-session map merging, loop closure에 쓰인다.

### 5.8.1 FPFH (Fast Point Feature Histograms)

[FPFH (Rusu et al., 2009)](https://ieeexplore.ieee.org/document/5152473)는 점 주변의 기하 형태를 히스토그램으로 나타내는 수작업 디스크립터다.

#### SPFH (Simple Point Feature Histogram)

쿼리 점 $\mathbf{p}$와 이웃 점 $\mathbf{p}_k$마다 로컬 좌표계를 세우고 각도 특성 셋을 계산한다.

법선 벡터 $\mathbf{n}_p, \mathbf{n}_k$와 방향 벡터 $\mathbf{d} = \mathbf{p}_k - \mathbf{p}$로 좌표축을 만든다.

$$\mathbf{u} = \mathbf{n}_p$$
$$\mathbf{v} = \mathbf{u} \times \frac{\mathbf{d}}{\|\mathbf{d}\|}$$
$$\mathbf{w} = \mathbf{u} \times \mathbf{v}$$

각도 특성은 다음 셋이다.
$$\alpha = \mathbf{v} \cdot \mathbf{n}_k, \quad \phi = \mathbf{u} \cdot \frac{\mathbf{d}}{\|\mathbf{d}\|}, \quad \theta = \text{atan2}(\mathbf{w} \cdot \mathbf{n}_k, \mathbf{u} \cdot \mathbf{n}_k)$$

각 특성을 $B$-bin 히스토그램으로 양자화한다.

#### FPFH: PFH를 빠르게 근사

PFH는 반경 $r$ 안의 이웃 쌍 전부에서 특성을 계산하므로 점마다 $O(k^2)$이 든다. FPFH는 쿼리 점과 이웃 사이의 특성만으로 SPFH를 구하고, 이웃들의 SPFH를 가중합해 점마다 $O(k)$로 근사한다.

$$\text{FPFH}(\mathbf{p}) = \text{SPFH}(\mathbf{p}) + \frac{1}{k} \sum_{i=1}^{k} \frac{1}{w_i} \text{SPFH}(\mathbf{p}_i)$$

$w_i = \|\mathbf{p}_i - \mathbf{p}\|$이다. 결과는 각도마다 11-bin씩, 33차원 히스토그램이다.

```python
import open3d as o3d

# FPFH 계산
pcd = o3d.io.read_point_cloud("scan.ply")
pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

fpfh = o3d.pipelines.registration.compute_fpfh_feature(
    pcd,
    o3d.geometry.KDTreeSearchParamHybrid(radius=0.25, max_nn=100)
)
# fpfh.data.shape: (33, N)
```

### 5.8.2 3DMatch (2017): 초기 학습 기반 3D 디스크립터

[Zeng et al. (2017)](https://arxiv.org/abs/1603.08182)의 3DMatch는 기존 RGB-D 재구성 데이터에서 자동으로 얻은 대응으로 local volumetric descriptor를 학습한다.

- **학습 데이터**: 실내 RGB-D 재구성 62개 장면에서 자동 생성한 대응점 쌍
- **구조**: 3D TDF(Truncated Distance Function) 볼륨을 입력받는 3D CNN
- **출력**: 512차원 로컬 디스크립터

논문과 함께 공개한 3DMatch Benchmark는 FCGF, Predator, GeoTransformer 등 뒤 연구들이 평가에 쓴다.

### 5.8.3 FCGF (Fully Convolutional Geometric Features, 2019)

[Choy et al. (2019)](https://doi.org/10.1109/ICCV.2019.00905)의 FCGF는 sparse convolution으로 포인트 클라우드 전체에서 한 번의 forward pass로 모든 점의 디스크립터를 뽑는다.

3DMatch는 키포인트마다 주변 볼륨을 따로 처리했다. FCGF 논문은 32차원 디스크립터로 3DMatch보다 feature-match recall이 높고 특징 추출은 약 169배 빨랐다고 보고했다. 비교한 3DMatch의 시간은 FCGF와 다른 CPU에서 잰 값이다.

### 5.8.4 Predator (2021): Overlap-Aware 3D Matching

[Huang et al. (2021)](https://arxiv.org/abs/2011.13005)의 Predator는 두 포인트 클라우드가 겹치는 영역을 따로 예측한다.

실제 두 스캔은 일부만 겹치는데, 그 전 방법들은 겹치지 않는 영역에서도 똑같이 디스크립터를 뽑았다. Predator의 구성은 다음과 같다.

1. **Overlap attention**: cross-attention으로 각 점이 상대 포인트 클라우드와 겹칠 확률을 예측한다
2. **Matchability score**: 겹치는 영역 안에서도 매칭에 쓸 만한 점(형태가 독특한 점)에 점수를 따로 매긴다

논문은 겹침이 10~30%인 3DLoMatch에서 성능이 크게 올랐다고 보고했다.

### 5.8.5 GeoTransformer (2022): 기하학적 트랜스포머

[Qin et al. (2022)](https://arxiv.org/abs/2202.06688)의 GeoTransformer는 강체 변환에 불변인 기하 특징을 학습하고, RANSAC 없이 변환을 추정한다.

#### 키포인트-프리 슈퍼포인트 매칭

키포인트를 검출하지 않고, 다운샘플링한 슈퍼포인트(superpoint)끼리 대응을 찾은 뒤 이를 촘촘한 점들로 넓힌다(propagation).

#### 기하학적 트랜스포머 아키텍처

강체 변환에 불변인 특징을 학습하려고 다음 기하 인코딩을 쓴다.

**1. 쌍별 거리 인코딩(Pairwise Distance)**:

슈퍼포인트 $\mathbf{p}_i, \mathbf{p}_j$ 사이의 유클리드 거리 $d_{ij} = \|\mathbf{p}_i - \mathbf{p}_j\|$는 강체 변환에 불변이다. 이 거리를 positional encoding으로 쓴다.

$$\text{PE}(d_{ij}) = [\sin(\omega_1 d_{ij}), \cos(\omega_1 d_{ij}), \ldots, \sin(\omega_K d_{ij}), \cos(\omega_K d_{ij})]$$

**2. 삼중 각도 인코딩(Triplet Angle)**:

세 점 $\mathbf{p}_i, \mathbf{p}_j, \mathbf{p}_k$가 이루는 각도도 강체 변환에 불변이라 함께 인코딩한다.

이 인코딩을 attention 점수 계산에 더한다.

$$\text{Attn}_{ij} = \frac{\mathbf{q}_i^\top \mathbf{k}_j}{\sqrt{d}} + b(\text{PE}(d_{ij}))$$

#### RANSAC-Free 변환 추정

GeoTransformer는 슈퍼포인트 대응마다 변환 가설을 만들고 local-to-global registration으로 가장 나은 것을 고르므로 RANSAC을 따로 돌리지 않는다. 저자들은 RANSAC보다 100배 빠르다고 보고했다. 아웃라이어가 더 많은 데이터에서도 같은 결과가 나오는지는 따로 확인해야 한다.

#### 성능

논문은 겹침이 적은 3DLoMatch에서 인라이어 비율이 17~30%p, registration recall이 7%p 넘게 올랐다고 보고했다.

```python
# GeoTransformer 사용 예제
from geotransformer.utils.pointcloud import apply_transform
import torch

# 두 포인트 클라우드 로드 (N, 3)
src_points = torch.from_numpy(src_pcd.points).float().cuda()
ref_points = torch.from_numpy(ref_pcd.points).float().cuda()

# 다운샘플링 + 슈퍼포인트 추출
# ... (voxel downsampling)

# GeoTransformer 추론
output = model(src_points, ref_points, src_feats, ref_feats)
# output['estimated_transform']: (4, 4) 변환 행렬
# output['src_corr_points'], output['ref_corr_points']: 대응점
```

### 5.8.6 3D-3D Correspondence 기술 계보

```
[Handcrafted 3D Descriptors]
FPFH (2009)          — 기하학적 각도 히스토그램
SHOT (2010)          — 방향 히스토그램
    ↓ 학습 기반
[Learned 3D Descriptors]
3DMatch (2017)       — 초기 learned 3D descriptor + benchmark
    ↓ 효율화
FCGF (2019)          — sparse conv, 전체 포인트 클라우드 한 번에 처리
    ↓ 오버랩 인식
Predator (2021)      — overlap-aware cross-attention
    ↓ RANSAC 제거
GeoTransformer (2022) — 기하학적 불변 트랜스포머, RANSAC-free
```

---

## 5.9 Cross-Modal Correspondence

### 5.9.1 2D-3D Matching: 카메라-LiDAR 간

이 절에서 다루는 cross-modal correspondence는 2D 이미지와 3D 포인트 클라우드의 매칭이다. 쓰이는 곳은 다음과 같다.

- **카메라-LiDAR 외부 캘리브레이션**: 두 센서가 본 같은 물리적 점을 찾아 외부 파라미터를 추정한다
- **LiDAR 맵 기반 Visual Localization**: LiDAR로 만든 지도 안에서 카메라 위치를 찾는다
- **Loop Closure**: 카메라와 LiDAR 관측으로 서로의 루프 후보를 확인한다

### 5.9.2 Image-to-Point Cloud 매칭 접근법

#### 투영 기반 접근 (Projection-based)

3D 포인트 클라우드를 2D 이미지 평면에 투영하면 2D-2D 매칭 문제가 된다.

1. LiDAR 포인트 클라우드를 가상 카메라 시점으로 렌더링해 depth 또는 intensity 이미지를 만든다
2. 이 이미지와 카메라 이미지를 SuperGlue 같은 방법으로 매칭한다
3. 매칭 결과를 렌더링 때의 깊이로 3D 좌표에 되돌린다

Koide et al.(2023)의 캘리브레이션 툴이 이 방식이다. 밀집한 LiDAR 포인트 클라우드를 가상 카메라로 렌더링하고, SuperGlue로 카메라 이미지와의 2D-3D 대응을 찾는다(초기값 추정 단계).

#### 학습 기반 직접 매칭

LCD(Learned Cross-Domain Descriptors, Pham et al., AAAI 2020)는 2D 이미지 패치와 3D 포인트 클라우드 패치를 같은 임베딩 공간에 놓도록 학습한다.

P2-Net(Wang et al., ICCV 2021)은 이미지 픽셀과 3D 점 사이의 대응을 직접 학습한다.

### 5.9.3 Cross-Modal 매칭이 어려운 이유

2D-3D cross-modal 매칭을 어렵게 하는 차이는 다음과 같다.

1. **표현의 이질성(Representation Heterogeneity)**: 2D 이미지는 규칙적인 격자 위의 밝기·색이고, 3D 포인트 클라우드는 불규칙하게 흩어진 좌표와 강도다.

2. **정보 비대칭(Information Asymmetry)**: 이미지에는 텍스처가 많고 깊이가 없다. 포인트 클라우드에는 정확한 기하 정보가 있고 텍스처는 거의 없다.

3. **밀도 차이(Density Gap)**: LiDAR 스캔 한 장의 점 수는 카메라 영상 한 장의 픽셀 수보다 훨씬 적다. LiDAR 포인트 클라우드는 거리가 멀수록 성겨진다.

4. **외관 도메인 갭(Appearance Domain Gap)**: 같은 물체를 두 센서가 서로 다른 양으로 잰다. 카메라는 조명과 표면 반사 특성이 함께 결정하는 밝기를 기록한다. LiDAR는 레이저 파장에서 되돌아온 광량(intensity)을 측정한다. 표면 고유의 반사율(albedo)은 두 측정값에 간접적으로만 섞여 있다.

MI 기반 방법(§5.4)은 이 차이를 통계적 의존성으로 건너뛰고, 투영 기반 방법은 두 관측을 같은 2D 영상 형식으로 바꾼다.

---

## 5.10 Dense Matching & Optical Flow

Dense matching은 이미지의 모든 픽셀에서 대응을 찾는 문제다.

### 5.10.1 Classical Optical Flow: Lucas-Kanade, Horn-Schunck

#### Optical Flow의 정의

Optical flow는 이미지 시퀀스에서 픽셀마다 보이는 움직임(apparent motion)을 2D 벡터 필드로 나타낸 것이다.

$$\mathbf{u}(x, y) = (u(x, y), v(x, y))$$

**밝기 항상성 가정(brightness constancy assumption)**:

$$I(x, y, t) = I(x + u, y + v, t + 1)$$

1차 Taylor 전개로 optical flow constraint equation을 얻는다.

$$I_x u + I_y v + I_t = 0$$

여기서 $I_x = \frac{\partial I}{\partial x}$, $I_y = \frac{\partial I}{\partial y}$, $I_t = \frac{\partial I}{\partial t}$.

식은 하나인데 미지수는 $(u, v)$ 둘이라 제약이 하나 더 있어야 한다(aperture problem).

#### Lucas-Kanade (1981)

**로컬 일관성 가정**: 작은 윈도우 $\Omega$ 안에서는 flow가 같다고 둔다. 윈도우 안의 모든 픽셀에 대해 다음이 성립한다.

$$\begin{bmatrix} I_{x_1} & I_{y_1} \\ I_{x_2} & I_{y_2} \\ \vdots & \vdots \\ I_{x_n} & I_{y_n} \end{bmatrix} \begin{bmatrix} u \\ v \end{bmatrix} = -\begin{bmatrix} I_{t_1} \\ I_{t_2} \\ \vdots \\ I_{t_n} \end{bmatrix}$$

곧 $\mathbf{A} \mathbf{u} = -\mathbf{b}$이다. 식이 미지수보다 많으므로(overdetermined) 최소자승으로 푼다.

$$\begin{bmatrix} u \\ v \end{bmatrix} = (\mathbf{A}^\top \mathbf{A})^{-1} \mathbf{A}^\top (-\mathbf{b})$$

여기서 $\mathbf{A}^\top \mathbf{A} = \begin{bmatrix} \sum I_x^2 & \sum I_x I_y \\ \sum I_x I_y & \sum I_y^2 \end{bmatrix}$는 Harris corner detector의 structure tensor $\mathbf{M}$과 같다.

**한계**: 변위가 크면(large displacement) 로컬 선형 근사가 맞지 않는다. 그래서 이미지 피라미드에서 거친 해상도부터 차례로 적용한다(Pyramidal LK, Bouguet 2000).

```python
# Pyramidal Lucas-Kanade (OpenCV)
lk_params = dict(
    winSize=(21, 21),
    maxLevel=3,       # 피라미드 레벨
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
)

# 추적할 점 선택 (보통 FAST/Shi-Tomasi corners)
p0 = cv2.goodFeaturesToTrack(prev_gray, maxCorners=200, qualityLevel=0.3, minDistance=7)

# Lucas-Kanade 추적
p1, status, err = cv2.calcOpticalFlowPyrLK(prev_gray, next_gray, p0, None, **lk_params)
# status: 추적 성공 여부 (1/0)
```

#### Horn-Schunck (1981)

**전역 평활성 가정**: flow가 이미지 전체에서 부드럽게 변한다고 두고 다음 에너지 함수를 최소화한다.

$$E = \iint \left[ (I_x u + I_y v + I_t)^2 + \lambda^2 (\|\nabla u\|^2 + \|\nabla v\|^2) \right] dx \, dy$$

첫째 항은 데이터 항(brightness constancy), 둘째 항은 평활성 정규화 항이다. $\lambda$가 클수록 flow가 부드러워진다.

Euler-Lagrange 방정식에서 반복 갱신식을 얻는다. 모든 픽셀의 flow가 나오고, 물체 경계처럼 flow가 끊기는 곳에서는 flow가 번진다.

### 5.10.2 Learning-Based Optical Flow: FlowNet → RAFT → FlowFormer → UniMatch

#### FlowNet / FlowNet 2.0 (2015/2017)

[Dosovitskiy et al. (2015)](https://arxiv.org/abs/1504.06852)의 FlowNet은 두 이미지를 받아 optical flow를 바로 출력하도록 지도 학습한 초기 CNN이다. 구조는 encoder-decoder다.

[FlowNet 2.0 (Ilg et al., 2017)](https://arxiv.org/abs/1612.01925)은 FlowNet 여러 개를 쌓아 정확도를 높였다.

#### RAFT (2020)

[Teed & Deng (2020)](https://arxiv.org/abs/2003.12039)의 RAFT는 4D correlation volume과 GRU 반복 갱신으로 flow를 구하며, ECCV 2020 Best Paper를 받았다.

**구조**:

**1. Feature Encoder**: 입력 두 이미지 각각에 CNN(ResNet 변형)을 적용해 1/8 해상도의 특징 맵 $\mathbf{g}_1, \mathbf{g}_2 \in \mathbb{R}^{H/8 \times W/8 \times D}$를 뽑는다. 별도의 Context Encoder가 첫 번째 이미지에서 GRU의 초기 hidden state와 context feature를 뽑는다.

**2. Correlation Volume**: 두 특징 맵의 모든 픽셀 쌍의 내적을 계산한다.

$$C_{ijkl} = \sum_d g_1(i, j, d) \cdot g_2(k, l, d)$$

4D correlation volume $\mathbf{C} \in \mathbb{R}^{H/8 \times W/8 \times H/8 \times W/8}$ 이 생긴다. 축은 모두 특징 맵 해상도 기준이다. 뒤의 두 차원을 average pooling해 4단계 correlation pyramid(스케일 1, 2, 4, 8)를 만든다.

flow는 한 해상도에서만 갱신하고, 상관값은 여러 스케일의 pyramid에서 찾아본다.

**3. Iterative Update (GRU)**: ConvGRU가 flow 추정치를 반복해서 고친다.

각 반복 $k$에서:
1. 현재 flow $\mathbf{f}^k$가 가리키는 대응 위치 주변의 작은 윈도우에서 correlation pyramid 값을 읽는다
2. 읽은 상관값, 현재 flow, context feature를 합친다
3. ConvGRU가 hidden state $\mathbf{h}^k$를 갱신한다
4. hidden state에서 flow 잔차(residual) $\Delta \mathbf{f}$를 예측한다
5. $\mathbf{f}^{k+1} = \mathbf{f}^k + \Delta \mathbf{f}$로 갱신한다

원 논문은 학습 때 12번 갱신하고, 평가 때는 Sintel에서 32번, KITTI에서 24번 갱신한다. 반복을 늘리면 논문 실험에서 오차가 더 줄기도 했다. 실제 횟수는 지연시간과 검증 오차를 보고 정한다.

**All-Pairs와 Coarse-to-Fine**: coarse-to-fine 방법은 거친 단계에서 틀린 대응을 뒤 단계에서 바로잡기 어렵다. RAFT는 모든 특징 쌍의 상관을 미리 만들어 두고 갱신 때마다 여러 스케일에서 찾아보므로, 작은 물체의 큰 움직임 같은 경우도 후보에 들어간다. RAFT 논문은 이 점을 coarse-to-fine 대비 장점으로 든다(Teed & Deng 2020 §1).

**학습**: 반복 단계마다 정답 flow와의 L1 loss를 구하고, 뒤 반복일수록 가중치를 지수적으로 키운다.

$$L = \sum_{k=1}^{K} \gamma^{K-k} \| \mathbf{f}^k - \mathbf{f}^{gt} \|_1$$

여기서 $\gamma = 0.8$이다.

```python
# RAFT 사용 예제 (torchvision)
from torchvision.models.optical_flow import raft_large, Raft_Large_Weights

model = raft_large(weights=Raft_Large_Weights.DEFAULT)
model = model.eval().cuda()

with torch.no_grad():
    # img1, img2: (1, 3, H, W) float tensors, range [-1, 1]
    flow_predictions = model(img1, img2)
    # flow_predictions[-1]: 최종 flow (1, 2, H, W)
    flow = flow_predictions[-1]  # (u, v) per pixel
```

#### FlowFormer (2022)

[FlowFormer (Huang et al., 2022)](https://arxiv.org/abs/2203.16194)는 cost volume을 토큰으로 바꾸고 트랜스포머로 cost memory를 만든다. recurrent transformer decoder가 이 memory를 참조하며 flow를 반복 갱신한다. 논문은 Sintel 벤치마크에서 RAFT보다 낮은 오차를 보고했다.

#### UniMatch (2023)

[Xu et al. (2023)](https://arxiv.org/abs/2211.05783)의 UniMatch는 optical flow, stereo matching, depth estimation을 모두 두 시점 사이의 dense correspondence 문제로 보고 한 모델로 푼다.

### 5.10.3 Dense Stereo: SGM → RAFT-Stereo → UniMatch

Stereo matching은 정렬(rectify)된 좌우 이미지에서 수평 방향 시차(disparity)를 추정하는 문제다. 수직 flow가 0인 optical flow로 볼 수 있다.

#### SGM (Semi-Global Matching, Hirschmuller 2005)

SGM은 전통적인 dense stereo 방법으로, 다음 순서로 시차를 구한다.
- 픽셀마다 모든 시차 후보의 매칭 비용(cost)을 계산한다
- 8개(또는 16개) 방향을 따라 1차원 동적 계획법으로 비용을 모은다(aggregation)
- 모은 비용이 가장 작은 시차를 고른다

$$L_r(p, d) = C(p, d) + \min\begin{cases} L_r(p-r, d) \\ L_r(p-r, d-1) + P_1 \\ L_r(p-r, d+1) + P_1 \\ \min_i L_r(p-r, i) + P_2 \end{cases}$$

$P_1, P_2$는 평활성 페널티다.

#### RAFT-Stereo (2021)

RAFT 구조를 stereo matching에 맞춘 것이다(Lipson et al., 3DV 2021). 같은 행의 픽셀끼리만 상관을 계산해 3D correlation volume을 만들고, GRU로 시차를 반복 갱신한다.

#### UniMatch의 통합

UniMatch는 같은 모델에서 과제에 따라 매칭 범위만 바꾼다.

- **Stereo**: 수평 방향 1D correlation
- **Flow**: 2D all-pairs correlation
- **Depth**: 자세를 아는 두 시점 사이에서 깊이 후보를 따라 1D 매칭

### 5.10.4 Sparse Feature에서 Dense Correspondence까지

```
[Sparse Correspondence]
Harris → SIFT → SuperPoint → SuperGlue → LightGlue
    ↓ 밀집화
[Dense Correspondence]
Lucas-Kanade (1981)  — 로컬 window, sparse-to-dense
Horn-Schunck (1981)  — 글로벌 최적화, 전 픽셀 flow
    ↓ 딥러닝
FlowNet (2015)       — CNN 직접 예측
[PWC-Net](https://arxiv.org/abs/1709.02371) (2018)       — cost volume + coarse-to-fine
    ↓ 전역 상관·반복 갱신
RAFT (2020)          — all-pairs correlation + iterative GRU
    ↓ 트랜스포머
FlowFormer (2022)    — cost volume 토큰화 + transformer
    ↓ 태스크 통합
UniMatch (2023)      — flow + stereo + depth 통합
```

DROID-SLAM(Teed & Deng, 2021)은 RAFT의 correlation volume과 반복 갱신 구조를 SLAM에 가져왔다.

---

## 기술 계보 요약

이 장의 방법을 한 도식에 모았다.

```
═══════════════════════════════════════════════════════════════════════════════
                    FEATURE MATCHING & CORRESPONDENCE 기술 계보
═══════════════════════════════════════════════════════════════════════════════

[2D Detection & Description]

Harris (1988) ─────→ SIFT (2004) ───→ FAST (2006) ─→ ORB (2011)
  코너 검출           scale-space       극한 속도       FAST+BRIEF
  structure tensor    DoG + 128D        검출만          binary descriptor
                      float descriptor
         │                │
         │   정확도 vs 속도 트레이드오프의 역사
         ▼                ▼
[학습 기반 Detection & Description]

    SuperPoint (2018) ──→ D2-Net (2019) ──→ R2D2 (2019) ──→ DISK (2020)
      self-supervised      detect=describe    reliability      RL 기반
      homographic adapt    joint feature map  + repeatability
         │
         │
         ▼
[학습 기반 Matching — 파이프라인 유지]

    SuperGlue (2020) ──────────────→ LightGlue (2023)
      GNN + cross-attention           adaptive depth/width
      Sinkhorn optimal transport      dual-softmax (Sinkhorn 제거)
      O(N²) × 9 layers               조기 종료, 2.5-4× 빠름

═══════════════════════ 검출기 없는 정합으로 전환 ═══════════════════════

[Detector-Free Matching — 파이프라인 해체]

    LoFTR (2021) ──→ QuadTree (2022) ──→ ASpanFormer (2022) ──→ RoMa (2024)
      transformer         O(N)        adaptive span          DINOv2
      coarse-to-fine       효율화           텍스처 적응             확률적 매칭
      검출기 완전 제거                                              foundation
                                                                    model 활용
                                                                        │
═══════════════════════ 3D-Aware 전환 ═══════════════════════════════════
                                                                        │
[3D-Aware Dense Matching — 2D 매칭을 넘어]                              ▼

    DUSt3R (2024) ──→ MASt3R (2024) ──→ VGGT (2025, CVPR Best Paper)
      3D pointmap         dense local        feed-forward 3D 추론
      직접 회귀            feature 추가       camera+depth+pointmap
      매칭 = 3D 부산물     매칭 성능 강화     단일 패스 통합 추론
═══════════════════════════════════════════════════════════════════════════════

[Dense Matching & Optical Flow — 병렬 진화]

    LK (1981) ──→ Horn-Schunck (1981) ──→ FlowNet (2015)
      로컬 window    글로벌 smoothness      CNN 직접 예측
         │                                      │
         ▼                                      ▼
    PWC-Net (2018) ──→ RAFT (2020) ──→ FlowFormer (2022) ──→ UniMatch (2023)
      cost volume       4D all-pairs      transformer           flow+stereo
      coarse-to-fine    iterative GRU     cost tokenization     +depth 통합
                            │
                            │ RAFT의 아이디어 전파
                            └──→ DROID-SLAM (correlation + DBA)

═══════════════════════════════════════════════════════════════════════════════

[3D-3D Correspondence — 독립 진화]

    FPFH (2009) ──→ SHOT (2010) ──→ 3DMatch (2017) ──→ FCGF (2019)
      기하학 히스토그램   방향 히스토그램    학습 3D descriptor   sparse conv
                                                │
                                                ▼
                                         Predator (2021)
                                           overlap-aware
                                                │
                                                ▼
                                        GeoTransformer (2022)
                                           기하학적 transformer
                                           RANSAC-free

═══════════════════════════════════════════════════════════════════════════════

[Cross-Modal — 2D-3D 대응]

    MI (정보 이론) ──→ NMI ──→ NID (Koide et al. 2023)
      다중 모달리티      정규화     LiDAR-Camera calibration
      통계적 의존성

    투영 기반: LiDAR 포인트 클라우드 렌더링 + SuperGlue (Koide et al. 2023 초기값 추정)
    학습 기반: LCD (2020), P2-Net (2021)

═══════════════════════════════════════════════════════════════════════════════
```

