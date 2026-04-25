# Ch.5 — Feature Matching & Correspondence: 기술 계보

Ch.4에서 상태 추정의 수학적 프레임워크를 세웠다. 하지만 칼만 필터든 팩터 그래프든, "어떤 관측이 어떤 랜드마크에 대응하는가"라는 **데이터 연관(data association)** 문제가 먼저 풀려야 한다. 이 챕터는 그 핵심인 특징점 매칭(feature matching)과 대응점 탐색(correspondence search)의 기술 계보를 추적한다.

> **이 챕터의 목적**: 센서 퓨전의 거의 모든 컴포넌트 — Visual Odometry, calibration, loop closure, point cloud registration — 가 **correspondence**(대응점)에 의존한다. 이 챕터는 mutual information에서 출발하여 RoMa까지 이어지는 기술적 흐름을 추적하며, 각 방법이 이전 세대의 어떤 한계를 해결했는지 살핀다.

---

## 5.1 Correspondence 문제란

### 5.1.1 "같은 것"을 찾는 문제

Correspondence 문제는 두 개 이상의 관측(observation)에서 **물리적으로 동일한 점, 영역, 또는 구조물**을 식별하는 문제다. 센서 퓨전 파이프라인의 거의 모든 단계가 correspondence를 전제로 한다.

- **Visual Odometry**: 연속된 프레임에서 같은 3D 점의 2D 투영을 찾아야 카메라 모션을 추정할 수 있다.
- **Calibration**: 카메라-LiDAR 외부 파라미터를 추정하려면 두 센서가 관측한 같은 물리적 점을 식별해야 한다.
- **Loop Closure**: 이전에 방문한 장소를 재인식하려면 현재 관측과 과거 관측 사이의 대응을 확인해야 한다.
- **Point Cloud Registration**: 두 스캔의 정합(alignment)은 대응점 쌍을 기반으로 강체 변환을 추정하는 과정이다.

### 5.1.2 Correspondence의 세 가지 유형

#### 2D-2D Correspondence

두 이미지 사이에서 같은 3D 점의 투영을 찾는 문제다. Visual Odometry, Stereo Matching, Image Stitching의 기초가 된다.

이미지 $I_1$의 점 $\mathbf{p}_1 = (u_1, v_1)$과 이미지 $I_2$의 점 $\mathbf{p}_2 = (u_2, v_2)$가 같은 3D 점 $\mathbf{X}$의 투영일 때, 이 쌍 $(\mathbf{p}_1, \mathbf{p}_2)$를 대응점(correspondence)이라 한다.

두 이미지 사이의 기하학적 관계는 **epipolar constraint**로 표현된다:

$$\mathbf{p}_2^\top \mathbf{F} \mathbf{p}_1 = 0$$

여기서 $\mathbf{F}$는 Fundamental matrix (3×3, rank 2)이다. 내부 파라미터를 알면 Essential matrix $\mathbf{E} = \mathbf{K}_2^\top \mathbf{F} \mathbf{K}_1$를 사용한다:

$$\hat{\mathbf{p}}_2^\top \mathbf{E} \hat{\mathbf{p}}_1 = 0$$

여기서 $\hat{\mathbf{p}} = \mathbf{K}^{-1} \mathbf{p}$는 정규화된 이미지 좌표다.

#### 2D-3D Correspondence

이미지의 2D 점과 3D 맵 포인트 사이의 대응. **PnP(Perspective-n-Point)** 문제의 기초이며, Visual Localization과 SLAM의 map-based tracking에서 핵심적이다.

3D 점 $\mathbf{X} = (X, Y, Z)^\top$과 그 2D 투영 $\mathbf{p} = (u, v)^\top$의 관계:

$$s \begin{bmatrix} u \\ v \\ 1 \end{bmatrix} = \mathbf{K} [\mathbf{R} | \mathbf{t}] \begin{bmatrix} X \\ Y \\ Z \\ 1 \end{bmatrix}$$

최소 6개(DLT), 4개(P3P+1), 또는 3개(P3P) 대응점으로 카메라 포즈 $(\mathbf{R}, \mathbf{t})$를 추정한다.

#### 3D-3D Correspondence

두 점군(point cloud) 사이에서 같은 물리적 점을 찾는 문제. LiDAR scan-to-scan 정합, multi-LiDAR calibration, loop closure 시 point cloud registration에 쓰인다.

두 점군 $\mathcal{P} = \{\mathbf{p}_i\}$와 $\mathcal{Q} = \{\mathbf{q}_i\}$ 사이의 강체 변환 $(\mathbf{R}, \mathbf{t})$를 추정:

$$\min_{\mathbf{R}, \mathbf{t}} \sum_{i} \| \mathbf{q}_i - (\mathbf{R} \mathbf{p}_i + \mathbf{t}) \|^2$$

이 최적화 문제의 closed-form 해는 SVD를 이용하여 구할 수 있다 (ICP, Ch.7 참조).

### 5.1.3 왜 센서 퓨전의 핵심인가

센서 퓨전에서 correspondence는 단순한 전처리가 아니라 **시스템 전체의 정확도를 결정하는 병목**이다. 잘못된 대응점(outlier) 하나가 포즈 추정을 완전히 망가뜨릴 수 있고, 대응점을 찾지 못하면 (텍스처 없는 환경) 시스템 자체가 동작하지 않는다. 이 챕터의 나머지 부분에서는 이 문제를 어떻게 해결해 왔는지의 기술적 계보를 추적한다.

---

## 5.2 전통적 Feature Detection & Description

전통적 correspondence 파이프라인은 **detect → describe → match**의 3단계로 구성된다. 이 절에서는 처음 두 단계 — 어디에서 특징점을 찾고(detection), 그 특징점을 어떻게 표현하는가(description) — 를 본다.

### 5.2.1 Corner Detection: Harris → FAST → ORB

#### Harris Corner Detector (1988)

Harris corner detector는 **이미지 패치를 이동시켰을 때 밝기 변화가 모든 방향으로 크게 일어나는 점**을 코너로 정의한다.

이미지 $I$에서 점 $(x, y)$ 주변의 윈도우를 $(\Delta u, \Delta v)$만큼 이동시킬 때의 밝기 변화:

$$E(\Delta u, \Delta v) = \sum_{x, y} w(x, y) [I(x + \Delta u, y + \Delta v) - I(x, y)]^2$$

$w(x, y)$는 가우시안 윈도우. Taylor 1차 전개를 적용하면:

$$E(\Delta u, \Delta v) \approx \begin{bmatrix} \Delta u & \Delta v \end{bmatrix} \mathbf{M} \begin{bmatrix} \Delta u \\ \Delta v \end{bmatrix}$$

여기서 **structure tensor**(또는 second moment matrix) $\mathbf{M}$은:

$$\mathbf{M} = \sum_{x, y} w(x, y) \begin{bmatrix} I_x^2 & I_x I_y \\ I_x I_y & I_y^2 \end{bmatrix}$$

$I_x, I_y$는 이미지의 x, y 방향 그래디언트. $\mathbf{M}$의 고유값 $(\lambda_1, \lambda_2)$에 따라:

- $\lambda_1 \approx 0, \lambda_2 \approx 0$: 평탄한 영역 (밝기 변화 없음)
- $\lambda_1 \gg \lambda_2 \approx 0$: 엣지 (한 방향으로만 변화)
- $\lambda_1, \lambda_2$ 모두 큼: 코너 (모든 방향으로 변화)

Harris는 고유값을 직접 계산하는 대신 **corner response function**을 정의:

$$R = \det(\mathbf{M}) - k \cdot \text{tr}(\mathbf{M})^2 = \lambda_1 \lambda_2 - k(\lambda_1 + \lambda_2)^2$$

여기서 $k$는 보통 0.04~0.06. $R > \text{threshold}$인 점을 코너로 선택하고, non-maximum suppression을 적용한다.

**Harris의 한계**: 회전 불변(rotation invariant)이지만 **스케일 불변이 아니다**. 카메라가 가까이 다가가면 코너가 엣지로 보일 수 있다. 이 한계가 SIFT의 scale-space 접근법의 동기가 되었다.

```python
import cv2
import numpy as np

# Harris Corner Detection
img = cv2.imread('scene.jpg', cv2.IMREAD_GRAYSCALE)
img_float = np.float32(img)

# blockSize: 이웃 크기, ksize: Sobel 커널, k: Harris 파라미터
harris_response = cv2.cornerHarris(img_float, blockSize=2, ksize=3, k=0.04)

# Non-maximum suppression & 임계값
corners = harris_response > 0.01 * harris_response.max()
```

#### FAST (Features from Accelerated Segment Test, 2006)

FAST는 Harris의 정확성보다 **속도를 극한으로 추구**한 검출기다. 로봇 비전에서 수십 FPS로 특징점을 검출해야 하는 실시간 요구를 해결하기 위해 [Rosten & Drummond (2006)](https://arxiv.org/abs/0810.2434)가 제안했다.

알고리즘은 단순하다:

1. 후보 픽셀 $p$를 중심으로 반지름 3의 원 위에 16개 픽셀을 배치 (Bresenham circle).
2. 원 위의 연속된 $N$개 픽셀(보통 $N=12$)이 모두 $p$보다 밝거나 모두 어두우면 $p$는 코너.
3. 고속 reject: 1, 5, 9, 13번 위치의 4개 픽셀만 먼저 확인 — 이 중 3개 이상이 조건을 만족하지 않으면 즉시 기각.

$$\text{FAST condition: } \exists \text{ contiguous arc of } N \text{ pixels on circle, all } > I_p + t \text{ or all } < I_p - t$$

여기서 $t$는 밝기 임계값.

**Decision tree 학습**: FAST는 추가로 머신러닝(ID3 decision tree)을 활용하여 검사 순서를 최적화한다. 어떤 픽셀을 먼저 검사해야 가장 빠르게 비-코너를 기각할 수 있는지를 학습한다.

**FAST의 한계**: 방향(orientation)과 스케일(scale) 정보가 없고, 디스크립터를 생성하지 않는다. 이를 보완한 것이 ORB다.

```python
# FAST Corner Detection
fast = cv2.FastFeatureDetector_create(threshold=20, nonmaxSuppression=True)
keypoints = fast.detect(img, None)
print(f"Detected {len(keypoints)} keypoints")
```

#### ORB (Oriented FAST and Rotated BRIEF, 2011)

ORB는 [Rublee et al. (2011)](https://ieeexplore.ieee.org/document/6126544)이 SIFT/SURF의 특허 문제와 속도 문제를 동시에 해결하기 위해 제안한 방법이다. **FAST 검출기에 방향성을 부여하고, BRIEF 디스크립터에 회전 불변성을 추가**한 조합이다.

**oFAST (Oriented FAST)**:
- FAST로 검출한 키포인트에 **intensity centroid** 방법으로 방향을 할당.
- 키포인트 주변 패치의 이미지 모멘트를 계산:

$$m_{pq} = \sum_{x, y} x^p y^q I(x, y)$$

- Centroid: $\mathbf{C} = (m_{10}/m_{00}, m_{01}/m_{00})$
- 방향: $\theta = \text{atan2}(m_{01}, m_{10})$

**rBRIEF (Rotated BRIEF)**:
- BRIEF는 패치 내의 랜덤 점 쌍 $(x_i, y_i)$의 밝기를 비교하여 이진 디스크립터를 생성:

$$\tau(\mathbf{p}; x_i, y_i) = \begin{cases} 1 & \text{if } I(\mathbf{p}, x_i) < I(\mathbf{p}, y_i) \\ 0 & \text{otherwise} \end{cases}$$

- 256개 비교 → 256-bit 이진 디스크립터.
- ORB는 키포인트 방향 $\theta$에 따라 비교 점 쌍을 회전시켜 **회전 불변성**을 확보.
- 추가로, 비교 점 쌍의 상관관계를 최소화하도록 그리디하게 선택하여 분별력(discriminability)을 높인다.

**멀티 스케일**: 이미지 피라미드(보통 8단계)를 구성하고 각 레벨에서 FAST를 수행하여 스케일 불변성을 근사한다.

ORB는 **ORB-SLAM 시리즈의 핵심 특징점**이며, 이진 디스크립터 덕분에 매칭이 Hamming distance로 수행되어 매우 빠르다.

```python
# ORB Detection & Description
orb = cv2.ORB_create(nfeatures=1000)
keypoints, descriptors = orb.detectAndCompute(img, None)
# descriptors.shape: (N, 32) — 256-bit = 32 bytes
```

### 5.2.2 Blob Detection: SIFT → SURF

#### SIFT (Scale-Invariant Feature Transform, 2004)

SIFT는 [Lowe (2004)](https://link.springer.com/article/10.1023/B:VISI.0000029664.99615.94)가 제안한, **스케일, 회전, 조명 변화에 불변한** 특징점 검출 및 기술 알고리즘이다. 20년간 특징점 매칭의 사실상 표준이었으며, 2020년 특허 만료 이후 자유롭게 사용 가능하다.

**Stage 1 — Scale-Space Extrema Detection (DoG)**:

이미지를 다양한 스케일의 가우시안으로 블러링한 scale-space를 구성한다:

$$L(x, y, \sigma) = G(x, y, \sigma) * I(x, y)$$

여기서 $G(x, y, \sigma) = \frac{1}{2\pi\sigma^2} \exp\left(-\frac{x^2 + y^2}{2\sigma^2}\right)$.

Laplacian of Gaussian (LoG)의 효율적 근사로 **Difference of Gaussians (DoG)**를 사용한다:

$$D(x, y, \sigma) = L(x, y, k\sigma) - L(x, y, \sigma) \approx (k-1)\sigma^2 \nabla^2 G$$

DoG 이미지에서 공간적(8개 이웃) + 스케일(상하 각 9개) = 26개 이웃과 비교하여 극값(extrema)을 찾는다.

**Stage 2 — Keypoint Localization**:

스케일-공간에서 Taylor 전개를 이용한 서브픽셀/서브스케일 위치 정밀화:

$$D(\mathbf{x}) = D + \frac{\partial D}{\partial \mathbf{x}}^\top \mathbf{x} + \frac{1}{2} \mathbf{x}^\top \frac{\partial^2 D}{\partial \mathbf{x}^2} \mathbf{x}$$

극값의 정밀 위치: $\hat{\mathbf{x}} = -\frac{\partial^2 D}{\partial \mathbf{x}^2}^{-1} \frac{\partial D}{\partial \mathbf{x}}$

저대비 키포인트 제거: $|D(\hat{\mathbf{x}})| < 0.03$이면 제거.
엣지 응답 제거: Hessian 행렬의 고유값 비를 이용하여 엣지 위의 불안정한 극값을 제거.

**Stage 3 — Orientation Assignment**:

키포인트 주변의 그래디언트 크기와 방향을 계산하고, 36-bin 방향 히스토그램을 생성한다 (가우시안 가중, $\sigma = 1.5 \times$ 키포인트 스케일). 최대 피크의 80% 이상인 피크가 있으면 해당 방향에도 별도 키포인트를 생성하여 **회전 불변성**을 확보한다.

**Stage 4 — Keypoint Descriptor**:

키포인트 주변 $16 \times 16$ 영역을 $4 \times 4$ 블록 16개로 분할. 각 블록에서 8-bin 방향 히스토그램을 생성한다:

$$\text{Descriptor} = 4 \times 4 \times 8 = 128\text{-dimensional vector}$$

L2 정규화 후 0.2 이상의 값을 클리핑하고 재정규화하여 비선형 조명 변화에 강건하게 만든다.

```python
# SIFT Detection & Description
sift = cv2.SIFT_create(nfeatures=2000)
keypoints, descriptors = sift.detectAndCompute(img, None)
# descriptors.shape: (N, 128) — 128-dimensional float32
```

#### SURF (Speeded-Up Robust Features, 2006)

SURF는 [Bay et al. (2006)](https://link.springer.com/chapter/10.1007/11744023_32)이 SIFT의 속도 문제를 해결하기 위해 제안했다. 핵심 아이디어:

- **Integral image**를 이용한 박스 필터로 LoG를 근사. 어떤 크기의 박스 필터든 O(1)에 계산.
- **Hessian determinant**를 검출 기준으로 사용 (DoG 대신):

$$\det(\mathbf{H}) = D_{xx} D_{yy} - (0.9 \cdot D_{xy})^2$$

- 64차원 디스크립터 (SIFT의 128차원 대비 절반): Haar wavelet 응답의 합과 절댓값 합.
- SIFT 대비 3~7배 빠르면서 유사한 정확도.

SURF는 특허 문제로 최근에는 잘 사용되지 않으며, 실시간 응용에서는 ORB가, 정확도가 중요한 응용에서는 SIFT나 학습 기반 방법이 선호된다.

### 5.2.3 Binary Descriptors: BRIEF, ORB, BRISK

이진 디스크립터는 패치 내 점 쌍의 밝기를 비교하여 0/1 비트열을 생성하는 디스크립터다. **Hamming distance**로 매칭하므로 float 디스크립터 대비 수십 배 빠르다.

| 디스크립터 | 비트 수 | 특징 |
|-----------|--------|------|
| BRIEF | 128/256/512 | 랜덤 점 쌍, 회전 불변 아님 |
| ORB | 256 | 학습된 점 쌍, 회전 불변 |
| BRISK | 512 | 동심원 샘플링, 스케일 불변 |

BRIEF의 비교 연산:

$$b_i = \begin{cases} 1 & \text{if } I(\mathbf{p}_i) < I(\mathbf{q}_i) \\ 0 & \text{otherwise} \end{cases}, \quad \text{descriptor} = \sum_{1 \le i \le n} 2^{i-1} b_i$$

Hamming distance는 XOR + popcount로 CPU에서 단일 명령어로 계산된다:

$$d_H(\mathbf{a}, \mathbf{b}) = \text{popcount}(\mathbf{a} \oplus \mathbf{b})$$

### 5.2.4 기술 계보: 정확도 vs 속도 트레이드오프의 역사

전통적 특징점의 역사는 **정확도와 속도의 트레이드오프**를 최적화하려는 시도의 연속이었다:

```
Harris (1988)     — 코너 검출의 수학적 정의
    ↓ 스케일 불변성 필요
SIFT (2004)       — scale-space + 128D float descriptor (정확하지만 느림)
    ↓ 속도 개선
SURF (2006)       — integral image + 64D (3~7× 빠름, 여전히 float)
    ↓ 실시간 요구
FAST (2006)       — 극한 속도 검출 (descriptor 없음)
    ↓ 검출+기술 통합
ORB (2011)        — oFAST + rBRIEF, 256-bit binary (Hamming 매칭)
```

이 트레이드오프는 딥러닝 시대에도 계속되며, SuperPoint는 SIFT급 정확도를 ORB급 속도로 달성하는 것을 목표로 했다.

---

## 5.3 전통적 Matching & Outlier Rejection

특징점을 검출하고 디스크립터를 추출한 뒤, **어떤 특징점 쌍이 실제로 같은 3D 점에 해당하는지**를 결정하는 매칭 단계가 필요하다.

### 5.3.1 Brute-Force Matching

가장 단순한 방법. 이미지 A의 모든 디스크립터를 이미지 B의 모든 디스크립터와 비교하여 가장 가까운 것을 매칭한다.

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

`crossCheck=True`는 **mutual nearest neighbor** 조건을 적용한다: A에서 B로의 최근접과 B에서 A로의 최근접이 일치할 때만 매칭을 수용.

### 5.3.2 FLANN (Fast Library for Approximate Nearest Neighbors)

대규모 디스크립터 집합에서는 brute-force가 비실용적이므로, **근사 최근접 이웃 탐색(ANN)**을 사용한다. FLANN은 kd-tree, randomized kd-tree, hierarchical k-means 등을 자동 선택한다.

```python
# FLANN Matching for SIFT
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)  # 탐색 노드 수 제한

flann = cv2.FlannBasedMatcher(index_params, search_params)
matches = flann.knnMatch(desc1, desc2, k=2)  # k=2 for ratio test
```

### 5.3.3 Lowe's Ratio Test

Lowe (2004)가 SIFT 논문에서 제안한 매칭 필터링 기법. **최근접 거리와 차근접 거리의 비율**이 임계값 이하일 때만 매칭을 수용한다:

$$\frac{d(\mathbf{f}, \mathbf{f}_1)}{d(\mathbf{f}, \mathbf{f}_2)} < \tau$$

여기서 $\mathbf{f}_1, \mathbf{f}_2$는 각각 최근접, 차근접 디스크립터, $\tau$는 보통 0.7~0.8.

직관: 올바른 매칭은 최근접이 확실히 가까우므로 $d_1 \ll d_2$. 잘못된 매칭은 여러 유사한 후보가 있으므로 $d_1 \approx d_2$.

```python
# Ratio test (Lowe's)
good_matches = []
for m, n in matches:  # m: best, n: second best
    if m.distance < 0.75 * n.distance:
        good_matches.append(m)
```

### 5.3.4 RANSAC 계열: RANSAC → PROSAC → MAGSAC++

매칭 단계 이후에도 **아웃라이어(잘못된 매칭)**가 존재한다. 기하학적 모델(Fundamental/Essential matrix)을 추정하면서 동시에 아웃라이어를 제거하는 것이 RANSAC 계열의 역할이다.

#### RANSAC (Random Sample Consensus, 1981)

[Fischler & Bolles (1981)](https://dl.acm.org/doi/10.1145/358669.358692)가 제안한 로버스트 추정 패러다임:

1. 전체 매칭 중 모델에 필요한 최소 $n$개를 무작위 추출 (예: Fundamental matrix는 8점, 7점, 또는 5점)
2. 추출한 점으로 모델 추정
3. 전체 매칭에서 모델과의 오차가 임계값 $t$ 이내인 점(인라이어)의 consensus set 구성
4. 가장 큰 consensus set을 가진 모델을 최종 선택
5. 인라이어 전체로 모델 재추정

**필요한 반복 횟수**:

$$k = \frac{\log(1 - p)}{\log(1 - w^n)}$$

- $p$: 원하는 성공 확률 (보통 0.99)
- $w$: 인라이어 비율
- $n$: 모델에 필요한 최소 점 수

예: 인라이어 50%, 8점 모델, 99% 신뢰 → $k \approx 1177$ 반복.

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

Chum & Matas가 제안했다. RANSAC이 균등 무작위 샘플링을 하는 반면, PROSAC은 **매칭 품질(예: 디스크립터 거리)** 순으로 정렬한 뒤 상위 매칭부터 점진적으로 샘플링한다.

직관: 좋은 매칭일수록 인라이어일 확률이 높으므로, 상위 매칭에서 먼저 모델을 시도하면 더 빨리 좋은 모델을 찾는다. RANSAC 대비 수~수십 배 빠른 수렴.

#### MAGSAC / MAGSAC++ (2019/2020)

Barath et al.이 제안. RANSAC의 핵심 문제 중 하나는 **임계값 $t$의 수동 설정**이다. MAGSAC은 이를 자동화한다:

- 모든 가능한 임계값 $\sigma$에 대해 모델의 품질을 marginalize:

$$Q(\theta) = \int_0^{\sigma_{\max}} q(\theta, \sigma) f(\sigma) d\sigma$$

여기서 $q(\theta, \sigma)$는 임계값 $\sigma$에서의 모델 $\theta$의 품질, $f(\sigma)$는 임계값의 사전 분포.

- MAGSAC++는 이를 더 효율적으로 구현하고, $\sigma$-consensus 기반 가중 최소자승 피팅을 추가.
- 임계값 선택에 대한 민감도가 크게 줄어든다.

```python
# OpenCV의 USAC (MAGSAC++ 포함)
F, mask = cv2.findFundamentalMat(
    pts1, pts2,
    cv2.USAC_MAGSAC,           # MAGSAC++ 사용
    ransacReprojThreshold=1.0,  # 덜 민감
    confidence=0.999,
    maxIters=10000
)
```

### 5.3.5 Fundamental / Essential Matrix Estimation

2D-2D 매칭으로부터 두 카메라 사이의 기하학적 관계를 추정하는 과정:

**Fundamental Matrix** ($\mathbf{F}$, 7 DOF):
- 내부 파라미터를 모를 때 사용
- 8-point algorithm: 최소 8개 대응점으로 선형 시스템을 풀고, rank-2 제약 적용
- 7-point algorithm: 최소 7개 대응점, 3차 다항식의 근으로 최대 3개 해

**Essential Matrix** ($\mathbf{E}$, 5 DOF):
- 내부 파라미터를 알 때 사용
- $\mathbf{E} = [\mathbf{t}]_\times \mathbf{R}$, $\mathbf{E} = \mathbf{K}_2^\top \mathbf{F} \mathbf{K}_1$
- 5-point algorithm (Nistér, 2004): 최소 5개 대응점, 10차 다항식으로 최대 10개 해

RANSAC과 결합하여 사용:

```python
# Essential matrix 추정 (calibrated camera)
E, mask = cv2.findEssentialMat(pts1, pts2, cameraMatrix=K,
                                method=cv2.RANSAC,
                                prob=0.999, threshold=1.0)

# Essential matrix에서 R, t 복원
_, R, t, mask_pose = cv2.recoverPose(E, pts1, pts2, cameraMatrix=K)
```

### 5.3.6 기술 계보: 로버스트 추정의 진화

```
Least Squares (아웃라이어에 취약)
    ↓ 아웃라이어 대응
RANSAC (1981)   — 최초의 로버스트 추정 패러다임
    ↓ 사전 정보 활용
PROSAC (2005)   — 매칭 품질 기반 진행적 샘플링
    ↓ 임계값 자동화
MAGSAC++ (2020) — 임계값-free 로버스트 추정
    ↓ 학습 기반 제거
GeoTransformer (2022) — RANSAC 없이 직접 변환 추정
```

---

## 5.4 Mutual Information & Intensity-Based Registration

### 5.4.1 MI의 정의와 직관

**Mutual Information (MI, 상호 정보량)**은 두 확률 변수가 서로에 대해 얼마나 많은 정보를 공유하는지를 측정하는 정보 이론적 척도다.

두 확률 변수 $X, Y$의 mutual information:

$$I(X; Y) = \sum_{x \in X} \sum_{y \in Y} p(x, y) \log \frac{p(x, y)}{p(x) p(y)}$$

연속 변수의 경우:

$$I(X; Y) = \int \int p(x, y) \log \frac{p(x, y)}{p(x) p(y)} \, dx \, dy$$

엔트로피(entropy)를 이용한 동치 표현:

$$I(X; Y) = H(X) + H(Y) - H(X, Y)$$

여기서:
- $H(X) = -\sum_x p(x) \log p(x)$: $X$의 엔트로피
- $H(X, Y) = -\sum_{x,y} p(x, y) \log p(x, y)$: 결합 엔트로피

**직관**: 두 이미지가 올바르게 정렬(aligned)되었을 때, 한 이미지의 픽셀 값을 알면 같은 위치의 다른 이미지 픽셀 값을 더 잘 예측할 수 있다. 즉 $I(X; Y)$가 최대가 된다. 정렬이 어긋나면 두 이미지의 관계가 약해져서 $I(X; Y)$가 감소한다.

핵심 성질: MI는 두 변수 사이의 **비선형적 통계적 의존성**을 측정한다. 단순한 상관계수(correlation coefficient)가 포착하지 못하는 관계도 잡아낸다.

### 5.4.2 MI 기반 다중 모달리티 정합

MI의 진정한 가치는 **서로 다른 모달리티의 센서 데이터를 정합**할 수 있다는 점이다. 원래 의료 영상(medical imaging)에서 CT-MRI 정합을 위해 개발된 방법이다.

왜 MI가 다중 모달리티에서 작동하는가?

- 같은 물체를 촬영한 CT와 MRI는 밝기 값의 분포가 완전히 다르다 (뼈가 CT에서 밝고, MRI에서 어두울 수 있다).
- 단순한 밝기 차이(SSD)나 상관관계(NCC)는 이러한 비선형 관계를 모델링하지 못한다.
- MI는 밝기 값 사이의 **통계적 의존성**만 측정하므로, 밝기 값의 단조(monotonic) 또는 비단조 관계에도 작동한다.

로보틱스에서의 응용: **카메라-LiDAR 정합**. LiDAR intensity 이미지와 카메라 이미지는 완전히 다른 물리적 양을 측정하지만, 같은 장면의 같은 구조물을 반영하므로 MI가 높다.

### 5.4.3 MI와 NMI의 실용적 계산

MI를 이미지 정합에 적용할 때, 확률 분포 $p(x), p(y), p(x, y)$는 **결합 히스토그램(joint histogram)**에서 추정한다.

이미지 $A$와 $B$가 변환 $T$로 정렬되어 있을 때:
1. 각 공통 위치 $(u, v)$에서 $A(u, v)$와 $B(T(u, v))$의 밝기 값 쌍을 수집.
2. 2D 히스토그램으로 결합 분포 $p(a, b)$를 추정.
3. 주변 분포 $p(a), p(b)$는 히스토그램의 행/열 합.
4. MI를 계산.

**Normalized Mutual Information (NMI)**는 MI를 정규화하여 오버랩 영역 크기에 대한 민감도를 줄인다:

$$NMI(A, B) = \frac{H(A) + H(B)}{H(A, B)}$$

또는:

$$NMI(A, B) = \frac{2 \cdot I(A; B)}{H(A) + H(B)}$$

NMI는 오버랩 영역이 변할 때도 안정적이므로 실용적으로 MI보다 선호된다.

### 5.4.4 MI Gradient 계산

MI를 정합의 목적 함수로 사용하려면 변환 파라미터에 대한 그래디언트를 계산해야 한다.

변환 $T_\xi$ (파라미터 $\xi$)에 의한 MI의 그래디언트:

$$\frac{\partial I}{\partial \xi} = \sum_{a, b} \frac{\partial p(a, b)}{\partial \xi} \left(1 + \log \frac{p(a, b)}{p(a) p(b)}\right)$$

결합 히스토그램이 이산적이면 그래디언트가 존재하지 않으므로, **Parzen 윈도우(커널 밀도 추정)** 또는 **B-spline** 기반의 미분 가능한 히스토그램 추정 방법을 사용한다.

실용적으로는 그래디언트 기반 최적화보다 **Nelder-Mead simplex** 같은 미분-free 최적화가 자주 사용되기도 한다 (Koide et al., 2023의 캘리브레이션 툴에서 사용).

### 5.4.5 왜 Calibration에서 MI가 쓰이는가

Ch.3의 targetless 카메라-LiDAR 캘리브레이션에서 MI(또는 NMI, NID)가 핵심 비용 함수로 사용된다:

1. LiDAR 점군을 현재 외부 파라미터 추정치로 카메라 이미지에 투영
2. 투영된 LiDAR intensity와 카메라 픽셀 밝기의 MI를 계산
3. 외부 파라미터를 조정하여 MI를 최대화

**Normalized Information Distance (NID)**는 Koide et al. (2023)의 범용 캘리브레이션 툴에서 사용된 MI 기반 거리 척도:

$$NID(A, B) = 1 - \frac{I(A; B)}{H(A, B)} = \frac{H(A|B) + H(B|A)}{H(A, B)}$$

NID는 0(완전 의존)에서 1(완전 독립)까지의 값을 가지며, 메트릭 공간의 성질을 만족한다.

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

전통적 특징점은 밝기 그래디언트, 코너, 블롭 같은 **저수준 시각적 단서**에 의존한다. 이 때문에 조명 변화, 시점 변화, 날씨 변화에 취약하다. 딥러닝은 대량의 데이터로부터 더 강건한 특징 표현을 학습하여 이 한계를 극복했다.

### 5.5.1 SuperPoint (2018): 자기지도학습 기반 검출+기술의 통합

[DeTone et al. (2018)](https://arxiv.org/abs/1712.07629)의 SuperPoint는 **키포인트 검출과 디스크립터 추출을 단일 네트워크로 통합**한 최초의 실용적 딥러닝 파이프라인이다.

#### Homographic Adaptation: 핵심 학습 전략

SuperPoint의 가장 중요한 기술적 기여는 **라벨 없이 반복성 높은 키포인트 검출기를 학습하는 방법**이다.

1. 하나의 이미지에 무작위 호모그래피를 100회 이상 적용
2. 각 변환된 이미지에서 키포인트를 검출
3. 검출 결과를 원래 좌표계로 역변환하여 집계(aggregate)
4. 여러 변환에서 **일관되게 검출되는 점**만 pseudo ground-truth로 채택

수작업 라벨 없이도 반복성(repeatability) 높은 키포인트 검출기를 학습하는 방식이다.

#### 2단계 학습: MagicPoint → SuperPoint

- **1단계**: 합성 기하 도형(삼각형, 사각형, 선분 등)으로 구성된 Synthetic Shapes 데이터셋에서 코너/접합점 검출기(**MagicPoint**)를 사전 학습.
- **2단계**: MagicPoint를 MS-COCO 등 실제 이미지에 Homographic Adaptation과 함께 적용하여, 실제 장면에서의 pseudo ground-truth를 생성하고 SuperPoint를 학습.

#### 아키텍처

VGG 스타일 인코더(공유 백본) → 두 개의 디코더 헤드로 분기:

**Interest Point Decoder**: 
- 입력 이미지를 8×8 셀 그리드로 나눔
- 각 셀에서 65채널 (64개 위치 + 1 "no interest point") softmax 수행
- 셀 내 위치를 직접 예측하는 방식으로 픽셀 단위 키포인트 히트맵 생성

**Descriptor Decoder**: 
- 공유 백본의 feature map에서 256차원 디스크립터 맵을 출력
- 검출된 키포인트 위치에서 bi-cubic interpolation으로 샘플링
- L2 정규화 적용

#### 학습 손실

- 키포인트 검출: cross-entropy loss
- 디스크립터: 호모그래피로 대응점을 알고 있으므로 positive/negative pair에 대한 hinge loss:

$$L_{desc} = \sum_{(i,j) \in \text{pos}} \max(0, m_p - \mathbf{d}_i^\top \mathbf{d}_j) + \sum_{(i,j) \in \text{neg}} \max(0, \mathbf{d}_i^\top \mathbf{d}_j - m_n)$$

여기서 $m_p, m_n$은 positive/negative margin.

**성능**: 단일 포워드 패스로 검출+기술을 동시 수행. 640×480 이미지에서 약 70 FPS (GPU 기준).

```python
import torch
# SuperPoint 사용 예제 (hloc / kornia)
from kornia.feature import SuperPoint as KorniaSuperPoint

# 모델 로드
sp = KorniaSuperPoint(max_num_keypoints=2048)
sp = sp.eval()

# 추론
with torch.no_grad():
    img_tensor = torch.from_numpy(img).float().unsqueeze(0).unsqueeze(0) / 255.0
    pred = sp(img_tensor)
    keypoints = pred['keypoints']        # (1, N, 2)
    descriptors = pred['descriptors']    # (1, 256, N)
    scores = pred['scores']              # (1, N)
```

### 5.5.2 D2-Net (2019): Detect-and-Describe Jointly

[D2-Net (Dusmanu et al., 2019)](https://arxiv.org/abs/1905.03561)은 검출과 기술을 더 극단적으로 통합한 방법이다. SuperPoint가 여전히 검출 헤드와 기술 헤드를 분리한 반면, D2-Net은 **같은 특징 맵에서 검출과 기술을 동시에 수행**한다.

핵심 아이디어: VGG16의 중간 특징 맵 $\mathbf{F} \in \mathbb{R}^{H \times W \times C}$를 사용하여:
- **Detection**: 각 위치에서 채널 축 최대값을 취한 뒤, 공간적 NMS를 적용하여 키포인트 선택
- **Description**: 같은 위치의 $C$-차원 벡터를 디스크립터로 사용

장점: 높은 수준의 의미적(semantic) 특징을 사용하므로 큰 외관 변화에 강건.
단점: 검출 반복성이 SuperPoint보다 낮을 수 있고, 입력 해상도의 1/4까지만 로컬라이제이션 가능.

### 5.5.3 R2D2 (2019): Reliable and Repeatable Detector-Descriptor

[R2D2 (Revaud et al., 2019)](https://arxiv.org/abs/1906.06195)는 SuperPoint와 D2-Net의 한계를 분석하고, **반복성(repeatability)과 신뢰성(reliability)을 명시적으로 학습**하는 방법을 제안했다.

- **Repeatability**: 다양한 시점에서 같은 점이 검출되는가?
- **Reliability**: 검출된 점의 디스크립터가 매칭에 유용한가? (텍스처 없는 영역의 키포인트는 반복적이어도 매칭에 쓸모없다)

R2D2는 두 가지를 별도의 confidence map으로 예측하고, 이들의 곱으로 최종 키포인트 스코어를 결정한다.

### 5.5.4 DISK (2020)

[DISK (Tyszkiewicz et al., 2020)](https://arxiv.org/abs/2006.13566)는 **reinforcement learning** 관점에서 키포인트 검출을 학습한다. 매칭에 성공하면 보상(reward), 실패하면 벌칙(penalty)을 주어 검출기를 학습한다.

핵심 차별점: 매칭 정확도를 직접 최적화하는 방식으로, 검출과 매칭의 end-to-end 최적화에 한 발 더 다가갔다.

### 5.5.5 전통 대비 장점: Illumination/Viewpoint Invariance 향상

학습 기반 특징점이 전통적 방법 대비 우위를 가지는 구체적 시나리오:

| 시나리오 | SIFT/ORB 한계 | SuperPoint/D2-Net 개선 |
|---------|-------------|---------------------|
| 극한 조명 변화 (주야간) | DoG/그래디언트 기반 검출 실패 | 학습된 특징은 고수준 구조 포착 |
| 넓은 시점 변화 | affine 근사의 한계 | 대량의 훈련 데이터로 시점 불변성 학습 |
| 반복 패턴 | 디스크립터가 유사하여 모호 매칭 | 컨텍스트 정보를 포착하여 분별 |
| 모션 블러 | 그래디언트 약화 → 검출 실패 | CNN이 블러에 대한 강건성 학습 |

그러나 학습 기반 방법의 한계도 존재한다: 학습 데이터의 도메인에 의존적이며, 완전히 새로운 환경(예: 수중, 화성)에서는 성능이 저하될 수 있다.

---

## 5.6 학습 기반 Feature Matching

### 5.6.1 SuperGlue (2020): Attention 기반 매칭

[Sarlin et al. (2020)](https://arxiv.org/abs/1911.11763)의 SuperGlue는 키포인트 매칭을 **그래프 신경망(GNN)과 어텐션 메커니즘**으로 학습 가능한 문제로 재정의했다. 전통적인 nearest-neighbor 매칭을 학습 기반 매칭으로 대체한 최초의 상용급 시스템이다.

#### 문제 정의: 부분 할당(Partial Assignment)

두 이미지에서 추출된 키포인트 집합 $\mathcal{A} = \{(\mathbf{p}_i, \mathbf{d}_i)\}_{i=1}^{N}$과 $\mathcal{B} = \{(\mathbf{p}_j, \mathbf{d}_j)\}_{j=1}^{M}$ 사이의 대응을 찾되, **모든 키포인트가 대응점을 갖는 것은 아니다**. 이를 위해 "매칭 없음(dustbin)"이라는 가상 노드를 추가하여 unmatchable 포인트를 명시적으로 처리한다.

#### Keypoint Encoder

키포인트 위치 $(x, y)$와 검출 confidence $c$를 MLP로 임베딩하여 디스크립터 벡터에 더한다:

$$\mathbf{f}_i^{(0)} = \mathbf{d}_i + \text{MLP}_{\text{enc}}([\mathbf{p}_i, c_i])$$

이로써 기하학적 정보가 특징에 내재된다.

#### Attentional Graph Neural Network

키포인트를 그래프의 노드로 표현하고, Self-Attention과 Cross-Attention을 교대로 적용한다:

**Self-Attention (intra-image)**: 같은 이미지 내 키포인트들 사이의 관계를 학습. 예를 들어, 건물의 코너들이 직선 위에 정렬되어 있다는 구조적 정보를 포착한다.

$$\text{message}_i^{\text{self}} = \sum_{j \in \mathcal{A}} \text{softmax}\left(\frac{\mathbf{q}_i^\top \mathbf{k}_j}{\sqrt{d}}\right) \mathbf{v}_j$$

**Cross-Attention (inter-image)**: 두 이미지 간 키포인트 사이의 관계를 학습. "이 키포인트는 상대 이미지의 어떤 키포인트와 유사한가"를 어텐션으로 추론한다.

$$\text{message}_i^{\text{cross}} = \sum_{j \in \mathcal{B}} \text{softmax}\left(\frac{\mathbf{q}_i^\top \mathbf{k}_j}{\sqrt{d}}\right) \mathbf{v}_j$$

이를 $L$번(논문에서는 9회) 교대로 반복하여 점진적으로 매칭 정보를 정제한다.

#### Optimal Transport를 이용한 매칭

최종 매칭 점수 행렬을 GNN 출력의 내적으로 계산한 뒤, **Sinkhorn 알고리즘**(soft한 형태의 Hungarian algorithm)을 적용한다.

점수 행렬 $\mathbf{S} \in \mathbb{R}^{(N+1) \times (M+1)}$ (dustbin 포함):

$$S_{ij} = \langle \mathbf{f}_i^{(L)}, \mathbf{f}_j^{(L)} \rangle, \quad S_{i,M+1} = S_{N+1,j} = z$$

여기서 $z$는 학습 가능한 dustbin score.

Sinkhorn 정규화를 반복 적용(약 100회):

$$\mathbf{S} \leftarrow \text{row-normalize}(\mathbf{S}), \quad \mathbf{S} \leftarrow \text{col-normalize}(\mathbf{S})$$

수렴 후 soft assignment matrix를 threshold하여 최종 매칭을 결정한다.

#### 학습

Ground-truth 대응점(호모그래피 또는 상대 포즈 + 깊이 맵에서 생성)에 대한 negative log-likelihood 최대화로 end-to-end 학습:

$$L = -\sum_{(i,j) \in \mathcal{M}} \log \hat{P}_{ij} - \sum_{i \in \mathcal{U}_A} \log \hat{P}_{i, M+1} - \sum_{j \in \mathcal{U}_B} \log \hat{P}_{N+1, j}$$

여기서 $\mathcal{M}$은 매칭된 쌍, $\mathcal{U}_A, \mathcal{U}_B$는 매칭되지 않는 키포인트.

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

#### 기술 계보에서의 위치

SuperPoint가 검출+기술을 학습화했다면, SuperGlue는 **매칭 단계를 학습화**했다. 이로써 detect-then-describe-then-match 파이프라인의 **세 단계 모두가 딥러닝으로 대체**되었다. 그러나 여전히 파이프라인의 직렬적 3단계 구조 자체는 유지된다. 이 구조적 한계를 깨뜨린 것이 LoFTR이다.

### 5.6.2 LightGlue (2023): SuperGlue의 효율화

[Lindenberger et al. (2023)](https://arxiv.org/abs/2306.13643)의 LightGlue는 SuperGlue의 정확도를 유지하면서 **적응적 연산량 조절(adaptive computation)**로 속도를 3-5배 끌어올렸다.

#### SuperGlue의 문제점 진단

- 항상 고정된 9개 GNN 레이어와 100회 Sinkhorn 반복을 수행 → 쉬운 매칭에도 불필요하게 많은 연산.
- 키포인트 수 $N$에 대해 $O(N^2)$ attention이 반복되므로, 키포인트가 많을수록 급격히 느려짐.

#### 핵심 개선: Adaptive Depth & Width

**Adaptive Depth (레이어 조기 종료)**:
- 각 레이어 이후에 경량 classifier(MLP)가 매칭 확신도를 예측.
- 확신도가 충분히 높으면 네트워크를 조기 종료.
- 쉬운 이미지 쌍은 2-3 레이어만으로 처리, 어려운 쌍은 전체 9 레이어 사용.

**Adaptive Width (키포인트 pruning)**:
- 매칭 불가능하다고 판단된 키포인트를 중간 레이어에서 제거.
- Attention의 시퀀스 길이가 점진적으로 줄어들어 후반 레이어의 계산이 가벼워짐.

**Sinkhorn 제거**: 
Optimal Transport 대신 단순한 **dual-softmax + mutual nearest neighbor**로 매칭. Sinkhorn의 반복 비용을 완전히 제거하면서도 성능 저하가 거의 없다.

$$P_{ij} = \text{softmax}_j(S_{ij}) \cdot \text{softmax}_i(S_{ij})$$

#### 학습 전략

- 학습 시에는 적응적 종료를 사용하지 않고 전체 레이어를 통과시키되, 각 레이어의 출력에 supervision을 적용 (**deep supervision**).
- 추론 시에만 적응적 종료를 활성화.
- SuperPoint, DISK, ALIKED 등 다양한 로컬 특징 검출기와 호환되도록 범용 설계.

**성능**: SuperGlue 대비 동등한 정확도에서 **3-5배 빠름**. 쉬운 쌍에서는 최대 10배 이상 속도 향상.

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

### 5.6.3 기술 흐름: Detect → Describe → Match 분리 파이프라인의 딥러닝화

```
[전통적 파이프라인]
SIFT detect → SIFT descriptor → BF/FLANN + ratio test
                                        ↓
[학습 기반 대체]
SuperPoint detect+describe → SuperGlue attention matching → LightGlue 효율화
    (2018)                       (2020)                        (2023)
```

이 진화 경로의 핵심 서사: **파이프라인의 각 단계를 하나씩 딥러닝으로 대체하되, 3단계 직렬 구조 자체는 유지**. 이 구조의 장점은 모듈성과 해석 가능성이며, 단점은 검출 단계의 실패가 전체 파이프라인의 실패로 이어진다는 것이다. 이 구조적 한계를 깨뜨린 것이 다음 절의 Detector-Free 패러다임이다.

---

## 5.7 Detector-Free Matching (패러다임 전환)

### 5.7.1 왜 Detector-Free가 필요한가

Detect-then-match 파이프라인은 **검출기가 키포인트를 찾지 못하면 매칭 자체가 불가능**하다는 근본적 한계가 있다. 이는 다음과 같은 실전 환경에서 치명적이다:

- **텍스처 없는 영역**: 흰 벽, 콘크리트 바닥, 하늘 — 그래디언트가 약해 코너/블롭 검출 실패
- **반복 패턴**: 타일, 창문 격자 — 검출은 되지만 디스크립터가 유사하여 모호 매칭
- **넓은 시점 변화**: 극단적 시점 차이에서 로컬 패치의 외관이 완전히 달라짐

Detector-free 접근법은 검출 단계를 없애고, **이미지의 모든 위치를 잠재적 매칭 후보로 취급**하여 이 한계를 극복한다.

### 5.7.2 LoFTR (2021): Coarse-to-Fine Transformer Matching

[Sun et al. (2021)](https://arxiv.org/abs/2104.00680)의 LoFTR는 **검출기 없이 두 이미지 간 밀집 매칭을 수행하는 트랜스포머 기반 아키텍처**로, detector-free matching이라는 새로운 패러다임을 열었다.

#### 아키텍처

**1. Local Feature CNN**: ResNet-18 기반 FPN(Feature Pyramid Network)으로 두 이미지에서 coarse (1/8 해상도) 및 fine (1/2 해상도) 특징 맵을 추출한다.

**2. Coarse-Level Matching (Transformer)**:

1/8 특징 맵을 1D 시퀀스로 평탄화(flatten)하고, Self-Attention + Cross-Attention을 교대로 $N$번(보통 4회) 반복하는 트랜스포머 모듈을 적용한다.

위치 인코딩: sinusoidal positional encoding으로 공간 정보를 보존.

출력 특징들 간의 내적으로 score matrix를 계산하고, **dual-softmax**로 confidence matrix를 생성:

$$P_{ij} = \text{softmax}_j(\mathbf{f}_i^A \cdot \mathbf{f}_j^B) \cdot \text{softmax}_i(\mathbf{f}_i^A \cdot \mathbf{f}_j^B)$$

Threshold + mutual nearest neighbor 조건으로 coarse 매칭을 추출한다.

**3. Fine-Level Refinement**:

Coarse 매칭 각각에 대해, 1/2 해상도 특징 맵에서 해당 위치 주변 $w \times w$ 윈도우(보통 5×5)를 crop한다.
윈도우 내에서 다시 cross-attention 기반 correlation을 수행하여 **sub-pixel 정밀도**의 매칭 위치를 회귀(regression)한다.

#### Linear Transformer

계산량 절감을 위해 standard softmax attention 대신 **kernel-based linear attention**을 사용:

Standard attention: $O(N^2)$, $\text{Attn}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{softmax}(\mathbf{Q}\mathbf{K}^\top / \sqrt{d}) \mathbf{V}$

Linear attention: $O(N)$, $\text{Attn}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \phi(\mathbf{Q})(\phi(\mathbf{K})^\top \mathbf{V})$

여기서 $\phi$는 ELU 기반 커널 함수. 행렬 곱의 결합 순서를 바꿔 $O(N)$ 복잡도를 달성한다.

단, 후속 연구에서 standard attention이 정확도에서 우위라는 점이 밝혀졌다.

#### 학습

Ground-truth 포즈 + 깊이 맵에서 생성한 대응점을 supervision으로 사용:
- Coarse level: cross-entropy loss
- Fine level: L2 regression loss
- ScanNet(실내) 및 MegaDepth(실외) 데이터셋에서 학습

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

#### 기술 계보에서의 위치

LoFTR는 **패러다임 전환점**이다. SuperPoint→SuperGlue로 이어진 detect-then-match 파이프라인과 결별하고, 검출기를 완전히 제거했다. 트랜스포머의 어텐션이 검출과 매칭을 동시에 수행한다. 이미지의 모든 위치가 상대 이미지의 모든 위치와 직접 소통(cross-attention)하므로, 별도의 검출 단계 없이도 매칭이 가능해졌다.

### 5.7.3 QuadTree Attention: LoFTR의 효율화

LoFTR의 coarse-level transformer는 이미지의 모든 위치를 시퀀스로 평탄화하므로, 해상도가 높아지면 시퀀스 길이가 급격히 증가한다. QuadTree Attention (Tang et al., 2022)은 이를 해결한다.

핵심 아이디어: 어텐션을 계층적으로 수행한다.

1. 가장 거친(coarsest) 해상도에서 전체 어텐션을 수행
2. 높은 어텐션 점수를 가진 영역만 선택
3. 선택된 영역에서만 다음 해상도의 어텐션을 수행
4. 이를 반복하여 관련 영역에만 집중하는 계층적 어텐션 실현

이를 통해 $O(N^2)$에서 $O(N \log N)$으로 복잡도를 줄이면서도 LoFTR 수준의 정확도를 유지한다.

### 5.7.4 ASpanFormer (2022): Adaptive Span Attention

ASpanFormer (Chen et al., 2022)는 LoFTR의 또 다른 한계를 해결한다: **모든 위치가 같은 크기의 어텐션 범위를 가져야 하는가?**

핵심 아이디어: 각 위치마다 **적응적으로 어텐션 범위(span)**를 조절한다.

- 텍스처가 풍부한 영역: 좁은 범위로 정밀한 매칭
- 텍스처가 부족한 영역: 넓은 범위로 컨텍스트를 활용한 매칭

이를 통해 LoFTR이 텍스처 없는 영역에서 보이던 매칭 정확도 저하를 완화한다.

### 5.7.5 RoMa (2024): DINOv2 + Dense Matching의 성숙

[Edstedt et al. (2024)](https://arxiv.org/abs/2305.15404)의 RoMa는 detector-free 매칭의 **성숙 단계**를 대표한다. 두 가지 핵심 진화를 통해 LoFTR 계열을 큰 폭으로 상회한다.

#### Foundation Model 활용

LoFTR가 처음부터(scratch) 학습한 것과 달리, RoMa는 **사전학습된 DINOv2 ViT-Large를 특징 추출기로 사용**한다 (가중치 동결).

DINOv2는 대규모 자기지도학습으로 학습된 범용 시각 특징으로, 이미 풍부한 의미적(semantic) 정보를 내포한다. 이는 "특징 학습은 범용 모델에 맡기고, 매칭 로직만 학습하면 된다"는 철학적 전환이다.

#### 아키텍처

**1. Frozen DINOv2 Backbone**: 사전학습된 DINOv2 ViT-Large에서 1/14 해상도의 patch 특징을 추출한다.

**2. Coarse Matching (Warp Estimation)**:
- DINOv2 특징에 cross-attention (transformer decoder)을 적용.
- 이미지 A의 각 위치에 대해, 이미지 B에서의 대응 위치를 **확률 분포**로 예측:

$$p(\mathbf{x}_B | \mathbf{x}_A) = \sum_{k} w_k \mathcal{N}(\mathbf{x}_B; \mu_k, \Sigma_k)$$

단일 점이 아닌 확률 분포로 예측함으로써, 모호한 매칭의 불확실성을 명시적으로 표현한다.

**3. Fine Matching (Iterative Refinement)**:
- Coarse warp를 초기값으로, CNN 기반의 fine-level 특징을 사용하여 반복적으로 정밀화.
- 각 정밀화 단계에서 해상도를 높이며, 이전 단계의 warp를 기반으로 local correlation을 계산하고 잔차를 예측.
- RAFT와 유사한 iterative refinement 철학이지만, optical flow가 아닌 sparse-to-dense warp에 적용.

**4. Certainty Estimation**: 각 매칭에 대해 신뢰도(certainty)를 함께 예측하여, 후처리 시 고신뢰 매칭만 선택적으로 사용 가능.

#### Robust Regression

매칭 위치를 단순히 L2 loss로 회귀하는 대신, 예측된 확률 분포와 ground-truth 간의 **negative log-likelihood**를 최적화한다:

$$L = -\sum_{(\mathbf{x}_A, \mathbf{x}_B^*)} \log p(\mathbf{x}_B^* | \mathbf{x}_A)$$

이 접근법은 아웃라이어에 강건하다: 잘못된 ground-truth 대응점이 있더라도 분포의 꼬리(tail)로 흡수되어 학습이 안정적이다.

#### 성능

MegaDepth 및 ScanNet 벤치마크에서 LoFTR, ASpanFormer 등을 큰 폭으로 상회. 특히 **넓은 baseline(큰 시점 변화)**에서의 성능 향상이 두드러진다.

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

#### 기술 계보에서의 위치

RoMa는 두 가지 핵심 전환을 체현한다:

1. **Foundation Model 활용**: 처음부터 학습 → 대규모 사전학습 모델의 특징을 기반으로 매칭 로직만 학습
2. **확률적 매칭**: 결정론적 점 예측 → 확률 분포 예측, 불확실한 매칭을 명시적으로 다룸

RoMa는 RAFT의 iterative refinement 아이디어와 LoFTR의 detector-free 사고방식을 결합하면서, DINOv2의 사전학습 특징을 도입한 종합적 발전이다.

### 5.7.6 최신 동향: 3D-Aware Dense Matching (2024-2025)

2024-2025년에는 2D 매칭을 넘어 **3D 기하학을 직접 예측하면서 매칭을 수행**하는 패러다임이 등장했다.

**DUSt3R (Leroy et al., 2024)**: [DUSt3R](https://arxiv.org/abs/2312.14132)는 캘리브레이션이나 포즈 정보 없이 임의의 이미지 쌍에서 직접 3D pointmap을 회귀하는 방법이다. 기존 매칭 파이프라인이 "2D 매칭 → 3D 복원"의 순서를 따른 반면, DUSt3R는 이를 뒤집어 **3D 구조 자체를 직접 예측하고, 대응점은 3D 공간에서 자연스럽게 얻어지는 부산물**로 다룬다.

**MASt3R (Leroy et al., 2024)**: [MASt3R](https://arxiv.org/abs/2406.09756)는 DUSt3R에 dense local feature head를 추가하여 밀집 매칭을 강화했다. Map-free localization 벤치마크에서 기존 최고 방법 대비 30%p(절대) VCRE AUC 향상을 달성했다.

**VGGT (Wang et al., 2025)**: [VGGT](https://arxiv.org/abs/2503.11651) (Visual Geometry Grounded Transformer)는 CVPR 2025 Best Paper로, 하나 이상의 이미지로부터 카메라 파라미터, pointmap, depth map, 3D point track을 **단일 feed-forward 패스로 동시에 추론**한다. 1초 이내의 추론 시간으로, 후처리 최적화가 필요한 기존 방법들을 능가하는 정확도를 보인다.

이 흐름은 "매칭은 2D 문제"라는 오랜 가정을 해체하고, **3D 기하학을 직접 추론하는 것이 결국 더 강건한 매칭을 낳는다**는 새로운 패러다임을 형성하고 있다.

---

## 5.8 3D-3D Correspondence

2D 이미지 매칭의 기술 계보와 병렬로, **3D 점군(point cloud) 사이의 대응점을 찾는 연구**도 진화해왔다. LiDAR scan-to-scan 정합, multi-session map merging, loop closure에서 핵심적이다.

### 5.8.1 FPFH (Fast Point Feature Histograms)

[FPFH (Rusu et al., 2009)](https://ieeexplore.ieee.org/document/5152473)는 3D 점군의 **기하학적 특징을 히스토그램으로 인코딩**하는 수작업 디스크립터다.

#### SPFH (Simple Point Feature Histogram)

쿼리 점 $\mathbf{p}$와 그 이웃 점 $\mathbf{p}_k$ 사이에서 로컬 좌표계를 설정하고, 세 가지 각도 특성을 계산한다:

법선 벡터 $\mathbf{n}_p, \mathbf{n}_k$와 방향 벡터 $\mathbf{d} = \mathbf{p}_k - \mathbf{p}$로부터:

$$\mathbf{u} = \mathbf{n}_p$$
$$\mathbf{v} = \mathbf{u} \times \frac{\mathbf{d}}{\|\mathbf{d}\|}$$
$$\mathbf{w} = \mathbf{u} \times \mathbf{v}$$

세 가지 각도 특성:
$$\alpha = \mathbf{v} \cdot \mathbf{n}_k, \quad \phi = \mathbf{u} \cdot \frac{\mathbf{d}}{\|\mathbf{d}\|}, \quad \theta = \text{atan2}(\mathbf{w} \cdot \mathbf{n}_k, \mathbf{u} \cdot \mathbf{n}_k)$$

각 특성을 $B$-bin 히스토그램으로 양자화.

#### FPFH: SPFH의 가속 버전

SPFH는 반경 $r$ 내의 모든 이웃 쌍의 특성을 계산하므로 $O(k^2)$. FPFH는 이를 근사하여 $O(k)$로 줄인다:

$$\text{FPFH}(\mathbf{p}) = \text{SPFH}(\mathbf{p}) + \frac{1}{k} \sum_{i=1}^{k} \frac{1}{w_i} \text{SPFH}(\mathbf{p}_i)$$

여기서 $w_i = \|\mathbf{p}_i - \mathbf{p}\|$. 33차원 히스토그램 (각 각도 11-bin × 3).

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

### 5.8.2 3DMatch (2017): Learned 3D Descriptors의 시작

[Zeng et al. (2017)](https://arxiv.org/abs/1603.08182)의 3DMatch는 RGB-D 데이터에서 **학습을 통해 3D 매칭 디스크립터를 추출**하는 최초의 시스템이다.

- **학습 데이터**: 62개 실내 장면의 RGB-D 재구성에서 자동 생성한 대응점 쌍
- **아키텍처**: 3D TDF(Truncated Distance Function) 볼륨을 입력으로 하는 3D CNN
- **출력**: 512차원 로컬 디스크립터

3DMatch는 학습 기반 3D 디스크립터의 출발점이며, 동시에 **3DMatch Benchmark**라는 표준 벤치마크를 제공하여 후속 연구의 평가 기준이 되었다.

### 5.8.3 FCGF (Fully Convolutional Geometric Features, 2019)

[Choy et al. (2019)](https://arxiv.org/abs/1904.09793)의 FCGF는 **sparse convolution**을 이용하여 전체 점군에서 한 번의 포워드 패스로 모든 점의 디스크립터를 추출한다.

3DMatch가 각 키포인트 주변의 로컬 볼륨을 개별적으로 처리하는 반면, FCGF는 전체 점군을 한꺼번에 처리하므로 **수십~수백 배 빠르다**. 32차원 디스크립터로 3DMatch의 512차원보다 간결하면서도 더 높은 정확도를 달성했다.

### 5.8.4 Predator (2021): Overlap-Aware 3D Matching

[Huang et al. (2021)](https://arxiv.org/abs/2011.13005)의 Predator는 두 점군의 **오버랩 영역을 명시적으로 예측**하는 접근법이다.

기존 방법들은 두 점군의 모든 영역에서 동일하게 디스크립터를 추출하지만, 실제로는 일부 영역만 겹친다. Predator는:

1. **Overlap attention**: cross-attention을 통해 각 점이 상대 점군과 겹치는 정도를 예측
2. **Matchability score**: 오버랩 영역 내에서 매칭에 유용한 점(독특한 기하학적 구조)을 별도로 스코어링
3. 오버랩 비율이 낮은(10-30%) 어려운 시나리오에서 큰 성능 향상

### 5.8.5 GeoTransformer (2022): 기하학적 트랜스포머

[Qin et al. (2022)](https://arxiv.org/abs/2202.06688)의 GeoTransformer는 3D point cloud registration에서 **기하학적 불변 특징 학습과 RANSAC 제거**를 동시에 달성한다.

#### 키포인트-프리 슈퍼포인트 매칭

반복 가능한 키포인트 검출 대신, 다운샘플링된 **슈퍼포인트(superpoint)**에서 대응점을 찾고 이를 밀집 포인트로 전파(propagation)한다.

#### 기하학적 트랜스포머 아키텍처

핵심은 **rigid transformation에 불변한 기하학적 특징**을 학습하는 것이다. 두 가지 기하학적 인코딩을 사용한다:

**1. 쌍별 거리 인코딩(Pairwise Distance)**:

슈퍼포인트 $\mathbf{p}_i, \mathbf{p}_j$ 사이의 유클리드 거리 $d_{ij} = \|\mathbf{p}_i - \mathbf{p}_j\|$는 강체 변환에 불변이다. 이를 positional encoding으로 사용:

$$\text{PE}(d_{ij}) = [\sin(\omega_1 d_{ij}), \cos(\omega_1 d_{ij}), \ldots, \sin(\omega_K d_{ij}), \cos(\omega_K d_{ij})]$$

**2. 삼중 각도 인코딩(Triplet Angle)**:

세 점 $\mathbf{p}_i, \mathbf{p}_j, \mathbf{p}_k$가 이루는 각도도 강체 변환에 불변이다. 이 각도 정보를 추가 인코딩으로 사용하여 기하학적 구조를 더 풍부하게 포착한다.

이 기하학적 인코딩은 트랜스포머의 attention bias로 주입된다:

$$\text{Attn}_{ij} = \frac{\mathbf{q}_i^\top \mathbf{k}_j}{\sqrt{d}} + b(\text{PE}(d_{ij}))$$

#### RANSAC-Free 변환 추정

슈퍼포인트 수준의 강건한 대응점이 높은 인라이어 비율을 달성하므로, **RANSAC 없이 직접 변환을 추정**할 수 있다. 이로 인해 **100배의 속도 향상**을 달성한다.

#### 성능

3DLoMatch 벤치마크에서 인라이어 비율 17-30%p 향상, 정합 리콜 7%p 이상 향상. 특히 **저오버랩(10-30%)** 시나리오에서 기존 방법 대비 큰 폭의 개선.

```python
# GeoTransformer 사용 예제
from geotransformer.utils.pointcloud import apply_transform
import torch

# 두 점군 로드 (N, 3)
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
3DMatch (2017)       — 최초의 학습 기반 3D 디스크립터 + 벤치마크
    ↓ 효율화
FCGF (2019)          — sparse conv, 전체 점군 한 번에 처리
    ↓ 오버랩 인식
Predator (2021)      — overlap-aware cross-attention
    ↓ RANSAC 제거
GeoTransformer (2022) — 기하학적 불변 트랜스포머, RANSAC-free
```

---

## 5.9 Cross-Modal Correspondence

### 5.9.1 2D-3D Matching: 카메라-LiDAR 간

가장 빈번한 cross-modal correspondence 문제는 **2D 이미지와 3D 점군 사이의 매칭**이다. 응용 시나리오:

- **카메라-LiDAR 외부 캘리브레이션**: 두 센서가 관측한 같은 물리적 점을 찾아 외부 파라미터 추정
- **Visual Localization against LiDAR Map**: 카메라 이미지를 LiDAR 맵에 대해 위치 추정
- **Loop Closure**: 카메라와 LiDAR 관측을 교차 검증

### 5.9.2 Image-to-Point Cloud 매칭 접근법

#### 투영 기반 접근 (Projection-based)

가장 직접적인 방법은 3D 점군을 2D 이미지 평면에 투영하여 2D-2D 매칭 문제로 환원하는 것이다:

1. LiDAR 점군을 가상 카메라 뷰로 렌더링하여 depth/intensity 이미지 생성
2. 생성된 2D 이미지와 카메라 이미지 사이에서 2D 매칭 (SuperGlue 등) 수행
3. 2D 매칭 결과를 3D 좌표로 역투영

Koide et al. (2023)의 캘리브레이션 툴이 이 접근법을 사용한다: LiDAR 밀집 점군을 가상 카메라로 렌더링하고, SuperGlue로 카메라 이미지와 교차 모달 2D-3D 대응점을 검출한다.

#### 학습 기반 직접 매칭

LCD (LiDAR-Camera Descriptor): 2D 이미지 패치와 3D 점군 패치의 공통 임베딩 공간을 학습한다.

P2-Net (Yu et al., 2021): patch-to-point 매칭을 학습하여 2D 이미지 패치와 3D 점의 직접 대응을 추론한다.

### 5.9.3 왜 Cross-Modal이 어려운가: Representation Gap

2D-3D cross-modal 매칭이 근본적으로 어려운 이유:

1. **표현의 이질성(Representation Heterogeneity)**: 2D 이미지는 정규 격자 위의 밝기/색상, 3D 점군은 불규칙하게 분포된 좌표+강도. 데이터 구조 자체가 다르다.

2. **정보 비대칭(Information Asymmetry)**: 이미지는 풍부한 텍스처 정보를 제공하지만 깊이가 없고, 점군은 정확한 기하학 정보를 제공하지만 텍스처가 희소하다.

3. **밀도 차이(Density Gap)**: 카메라 이미지의 해상도(수백만 픽셀)와 LiDAR 점군의 밀도(수만~수십만 점)가 크게 다르며, LiDAR 점군은 거리에 따라 밀도가 급격히 변한다.

4. **외관 도메인 갭(Appearance Domain Gap)**: 같은 물체라도 카메라의 반사율(albedo)과 LiDAR의 반사 강도(intensity)는 다른 물리적 양을 측정한다.

이러한 어려움 때문에, cross-modal correspondence는 아직 unimodal (2D-2D 또는 3D-3D) 매칭에 비해 성숙도가 낮은 연구 영역이다. MI 기반 접근법 (5.4절)은 이 도메인 갭을 통계적으로 우회하는 전략이며, 투영 기반 접근법은 문제를 같은 모달리티로 환원하는 전략이다.

---

## 5.10 Dense Matching & Optical Flow

지금까지 다룬 방법들은 **sparse correspondence** (희소 대응점)에 집중했다. 이 절에서는 이미지의 **모든 픽셀에 대해 대응을 찾는 dense matching**을 다룬다.

### 5.10.1 Classical Optical Flow: Lucas-Kanade, Horn-Schunck

#### Optical Flow의 정의

Optical flow는 이미지 시퀀스에서 **각 픽셀의 겉보기 운동(apparent motion)**을 2D 벡터 필드로 표현한 것이다:

$$\mathbf{u}(x, y) = (u(x, y), v(x, y))$$

**밝기 항상성 가정(brightness constancy assumption)**:

$$I(x, y, t) = I(x + u, y + v, t + 1)$$

Taylor 1차 전개로 **optical flow constraint equation**을 얻는다:

$$I_x u + I_y v + I_t = 0$$

여기서 $I_x = \frac{\partial I}{\partial x}$, $I_y = \frac{\partial I}{\partial y}$, $I_t = \frac{\partial I}{\partial t}$.

이 하나의 방정식에 두 개의 미지수 $(u, v)$가 있으므로, 추가 제약이 필요하다 (aperture problem).

#### Lucas-Kanade (1981)

**로컬 일관성 가정**: 작은 윈도우 $\Omega$ 내에서 optical flow가 일정하다고 가정한다. 윈도우 내의 모든 픽셀에 대해:

$$\begin{bmatrix} I_{x_1} & I_{y_1} \\ I_{x_2} & I_{y_2} \\ \vdots & \vdots \\ I_{x_n} & I_{y_n} \end{bmatrix} \begin{bmatrix} u \\ v \end{bmatrix} = -\begin{bmatrix} I_{t_1} \\ I_{t_2} \\ \vdots \\ I_{t_n} \end{bmatrix}$$

즉 $\mathbf{A} \mathbf{u} = -\mathbf{b}$. Overdetermined system이므로 최소자승법으로:

$$\begin{bmatrix} u \\ v \end{bmatrix} = (\mathbf{A}^\top \mathbf{A})^{-1} \mathbf{A}^\top (-\mathbf{b})$$

여기서 $\mathbf{A}^\top \mathbf{A} = \begin{bmatrix} \sum I_x^2 & \sum I_x I_y \\ \sum I_x I_y & \sum I_y^2 \end{bmatrix}$는 Harris corner detector의 structure tensor $\mathbf{M}$과 동일하다.

**한계**: 큰 변위(large displacement)에서는 로컬 선형 근사가 실패. 이를 해결하기 위해 **이미지 피라미드**에서 coarse-to-fine으로 적용한다 (Pyramidal LK).

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

**글로벌 평활성 가정**: optical flow가 이미지 전체에서 부드럽게 변한다고 가정하고, 다음 에너지 함수를 최소화:

$$E = \iint \left[ (I_x u + I_y v + I_t)^2 + \lambda^2 (\|\nabla u\|^2 + \|\nabla v\|^2) \right] dx \, dy$$

첫 번째 항은 데이터 항(brightness constancy), 두 번째 항은 평활성 정규화. $\lambda$가 클수록 더 부드러운 flow.

Euler-Lagrange 방정식을 풀면 반복적 갱신 수식을 얻는다. Dense flow를 생성하지만, 경계에서의 불연속을 잘 처리하지 못한다는 한계가 있다.

### 5.10.2 Learning-Based Optical Flow: FlowNet → RAFT → FlowFormer → UniMatch

#### FlowNet / FlowNet 2.0 (2015/2017)

[Dosovitskiy et al. (2015)](https://arxiv.org/abs/1504.06852)의 FlowNet은 **optical flow를 CNN으로 직접 예측**한 최초의 딥러닝 방법이다. 두 이미지를 입력으로 받아 flow 필드를 출력하는 encoder-decoder 구조.

[FlowNet 2.0 (Ilg et al., 2017)](https://arxiv.org/abs/1612.01925)은 여러 FlowNet을 스택하여 정확도를 끌어올렸다.

#### RAFT (2020): Optical Flow의 새로운 표준

[Teed & Deng (2020)](https://arxiv.org/abs/2003.12039)의 RAFT는 **4D correlation volume + 반복적 GRU 업데이트**라는 새로운 아키텍처 패러다임을 제시하여, ECCV 2020 Best Paper를 수상했다.

**아키텍처 3단계**:

**1. Feature Encoder**: 입력 두 이미지 각각에 CNN(ResNet 변형)을 적용하여 1/8 해상도의 특징 맵 $\mathbf{g}_1, \mathbf{g}_2 \in \mathbb{R}^{H/8 \times W/8 \times D}$를 추출. 별도의 Context Encoder가 첫 번째 이미지에서 GRU의 초기 hidden state와 context feature를 추출.

**2. Correlation Volume 구성**: 두 특징 맵의 모든 픽셀 쌍에 대해 내적을 계산:

$$C_{ijkl} = \sum_d g_1(i, j, d) \cdot g_2(k, l, d)$$

4D correlation volume $\mathbf{C} \in \mathbb{R}^{H \times W \times H \times W}$ 생성. 이를 후반 두 차원에 대해 average pooling하여 4단계 **correlation pyramid**을 구축 (스케일 1, 2, 4, 8).

**coarse-to-fine이 아닌, single resolution에서 multi-scale lookup**을 수행한다는 점이 이전 방법과의 차이다.

**3. Iterative Update (GRU)**: ConvGRU가 현재 flow 추정치를 반복적으로 정제한다:

각 반복 $k$에서:
1. 현재 flow 추정 $\mathbf{f}^k$에 따라 correlation pyramid에서 값을 lookup (현재 대응 위치 주변의 local window 참조)
2. Correlation feature, 현재 flow, context feature를 결합
3. ConvGRU가 hidden state $\mathbf{h}^k$를 업데이트
4. Hidden state에서 flow 잔차(residual) $\Delta \mathbf{f}$ 예측
5. Flow 업데이트: $\mathbf{f}^{k+1} = \mathbf{f}^k + \Delta \mathbf{f}$

학습 시 12회, 추론 시 12~32회 반복. **반복 횟수를 늘리면 정확도가 향상**되는 특성(test-time adaptability).

**All-Pairs vs. Coarse-to-Fine**: 기존 PWC-Net, FlowNet 등은 피라미드에서 coarse flow를 먼저 추정하고 점진적으로 정밀화하는데, 큰 변위가 coarse 레벨에서 놓치면 복구 불가. RAFT는 **전체 해상도에서 모든 상관관계를 한 번에 보유**하므로 큰 변위도 놓치지 않는다.

**학습**: 모든 반복 단계의 예측에 대해 ground-truth flow와의 L1 loss를 적용하되, 후반 반복에 exponentially increasing weights:

$$L = \sum_{k=1}^{K} \gamma^{K-k} \| \mathbf{f}^k - \mathbf{f}^{gt} \|_1$$

여기서 $\gamma = 0.8$.

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

[FlowFormer (Huang et al., 2022)](https://arxiv.org/abs/2203.16194)는 RAFT의 GRU 업데이트를 **트랜스포머로 대체**한 방법이다. Cost volume을 토큰화하고, 트랜스포머의 자기주의(self-attention)로 글로벌 컨텍스트를 포착하여 RAFT를 상회하는 정확도를 달성했다.

#### UniMatch (2023)

[Xu et al. (2023)](https://arxiv.org/abs/2211.05783)의 UniMatch는 **optical flow, stereo matching, depth estimation을 하나의 통합 프레임워크**로 처리한다. 핵심 아이디어는 세 태스크가 모두 "두 관측 사이의 dense correspondence"라는 공통 문제로 귀결된다는 것이다.

### 5.10.3 Dense Stereo: SGM → RAFT-Stereo → UniMatch

Stereo matching은 좌우 카메라 이미지에서 **수평 방향의 시차(disparity)**를 추정하는 문제다. Optical flow의 특수한 경우 (수직 방향 flow = 0)로 볼 수 있다.

#### SGM (Semi-Global Matching, Hirschmuller 2005)

전통적 dense stereo의 대표. 핵심 아이디어:
- 각 픽셀에서 모든 시차 후보에 대한 매칭 비용(cost) 계산
- 8(또는 16) 방향에서 1D 동적 프로그래밍으로 비용을 집계(aggregation)
- 집계된 비용에서 최소값을 선택하여 시차 결정

$$L_r(p, d) = C(p, d) + \min\begin{cases} L_r(p-r, d) \\ L_r(p-r, d-1) + P_1 \\ L_r(p-r, d+1) + P_1 \\ \min_i L_r(p-r, i) + P_2 \end{cases}$$

여기서 $P_1, P_2$는 평활성 페널티.

#### RAFT-Stereo (2021)

RAFT의 아키텍처를 stereo matching에 적용. 4D correlation volume을 3D (H×W×D)로 축소하고, 반복적 GRU로 시차를 정제한다.

#### UniMatch의 통합

UniMatch는 flow, stereo, depth를 통합하여, 하나의 모델이 태스크에 따라 cross-attention의 방향과 범위를 조절한다:

- **Stereo**: 수평 방향 1D correlation
- **Flow**: 2D all-pairs correlation
- **Depth**: monocular 특징에서 depth regression

### 5.10.4 기술 흐름: Sparse Feature → Dense Correspondence

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
    ↓ 패러다임 전환
RAFT (2020)          — all-pairs correlation + iterative GRU
    ↓ 트랜스포머
FlowFormer (2022)    — cost volume 토큰화 + transformer
    ↓ 태스크 통합
UniMatch (2023)      — flow + stereo + depth 통합
```

RAFT의 all-pairs correlation과 iterative refinement 아이디어는 dense matching뿐 아니라 **detector-free feature matching** (LoFTR, RoMa)과 **learned SLAM** (DROID-SLAM)에도 직접적으로 영향을 미쳤다.

---

## 기술 계보 요약

이 챕터에서 다룬 모든 기술의 흐름을 하나의 다이어그램으로 정리한다:

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
      O(N²) × 9 layers               조기 종료, 3-5× 빠름                    
                                                                              
═══════════════════════ 패러다임 전환 ═══════════════════════════════════       
                                                                              
[Detector-Free Matching — 파이프라인 해체]                                    
                                                                              
    LoFTR (2021) ──→ QuadTree (2022) ──→ ASpanFormer (2022) ──→ RoMa (2024) 
      transformer         O(N log N)        adaptive span          DINOv2     
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
                            ├──→ LoFTR (all-pairs attention)                  
                            ├──→ RoMa (iterative refinement)                  
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

[Cross-Modal — MI 기반 우회 전략]                                             
                                                                              
    MI (정보 이론) ──→ NMI ──→ NID (Koide et al. 2023)                       
      다중 모달리티      정규화     LiDAR-Camera calibration                    
      통계적 의존성                                                            
                                                                              
═══════════════════════════════════════════════════════════════════════════════

핵심 서사:
  1. 전통적 3단계 파이프라인(detect → describe → match)을 딥러닝으로
     각각 대체하는 흐름: SIFT → SuperPoint → SuperGlue → LightGlue

  2. 파이프라인 자체를 해체하고 end-to-end 밀집 매칭으로 전환하는 흐름:
     RAFT → LoFTR → RoMa

  3. Foundation model 활용의 부상: DINOv2 특징을 매칭의 기반으로 활용하여,
     "특징 학습은 범용 모델에 맡기고 매칭 로직만 학습"하는 패러다임 전환.

  4. 3D-Aware 매칭의 등장 (2024-2025): DUSt3R → MASt3R → VGGT로 이어지는
     흐름에서, "2D 매칭 후 3D 복원"이 아닌 "3D를 직접 추론하면 매칭은
     자연스럽게 따라온다"는 역발상이 큰 성공을 거두고 있다.

  후자(2, 3, 4)가 점점 우세해지는 추세이나, 전자(1)의 효율성과 해석 가능성은
  여전히 실용적 가치가 있다.
```

이 챕터에서 다룬 매칭 기술들은 다음 챕터부터 본격적으로 활용된다. Ch.6에서는 이 기술들이 Visual Odometry의 frontend에서 어떻게 배치되는지, 그리고 Ch.4에서 다룬 상태 추정 방법이 backend에서 어떻게 결합되는지를 구체적인 시스템들을 통해 살펴본다.
