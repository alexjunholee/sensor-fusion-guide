# 5장 특징 매칭과 대응점

앞선 4장에서 살펴본 칼만 필터와 factor graph는 매우 중요한 전제 위에서 동작한다. 지금 들어온 관측값이 지도 위의 어느 랜드마크에서 비롯됐는지, 즉 **데이터 연관(data association)**이 이미 풀려 있다고 가정하기 때문이다. 하지만 실제 카메라 영상이나 LiDAR 스캔에는 랜드마크 번호표가 붙어 있지 않다. 두 장의 사진이나 두 번의 스캔, 혹은 서로 다른 센서 데이터 사이에서 "물리적으로 같은 지점"을 스스로 찾아 짝짓는 과정이 먼저 필요하며, 바로 이 역할을 **특징점 매칭(feature matching)**과 **대응점 탐색(correspondence search)**이 맡는다. 이 장에서는 고전적인 기하 기반 검출기·디스크립터부터 정보 이론 기반 정합, 딥러닝 기반 매처와 3D·dense 대응 기법까지 이어지는 기술의 흐름을 따라간다.

---

## 5.1 Correspondence 문제란

### 5.1.1 "같은 것"을 찾는 문제

Correspondence 문제는 둘 이상의 관측(observation)에서 물리적으로 같은 점, 영역, 혹은 구조물을 찾아내 서로 잇는 문제다. 표현이 단순해 보이지만, 센서 융합 파이프라인의 거의 모든 핵심 단계가 이 대응점 위에서 돌아간다.

- **Visual Odometry**: 연속된 카메라 프레임에서 같은 3D 점이 투영된 2D 위치를 찾아야 프레임 사이의 카메라 움직임을 계산할 수 있다.
- **Calibration**: 카메라와 LiDAR의 외부 파라미터를 추정하려면 두 센서가 바라본 공통 대상을 연결해야 한다. 체커보드 같은 target을 쓰는 방법은 같은 물리적 모서리나 평면을 대응으로 삼고, targetless 상호 정보량(MI) 기반 방법(§5.4)은 두 센서 데이터가 겹치는 영역의 통계적 의존성을 대응의 단서로 활용한다.
- **Loop Closure**: 로봇이 이전에 지나온 장소를 다시 만났을 때 누적 오차를 지우려면, 현재 관측과 과거 관측 사이에 충분한 대응이 성립하는지 확인해야 한다.
- **Point Cloud Registration**: 두 3D 스캔을 하나의 좌표계로 맞추려면(alignment) 양쪽 스캔에서 짝이 맞는 대응점 쌍을 찾아 강체 변환을 추정해야 한다.

### 5.1.2 Correspondence의 유형

센서 조합과 문제 설정에 따라 대응점의 차원이 달라지며, 그에 맞춰 동원되는 기하학적 제약식도 바뀐다. Correspondence는 크게 2D-2D, 2D-3D, 3D-3D의 세 가지 유형으로 나뉜다.

#### 2D-2D Correspondence

두 이미지에서 같은 3D 점이 투영된 픽셀 위치를 짝짓는 문제로, Visual Odometry, Stereo Matching, Image Stitching의 출발점이 된다.

이미지 $I_1$의 점 $\mathbf{p}_1 = (u_1, v_1)$과 이미지 $I_2$의 점 $\mathbf{p}_2 = (u_2, v_2)$가 공간상의 같은 3D 점 $\mathbf{X}$를 찍은 것이라면, 이 쌍 $(\mathbf{p}_1, \mathbf{p}_2)$를 대응점(correspondence)이라 부른다. $3 \times 3$ 행렬을 곱하고 $\mathbf{K}^{-1}$을 적용하려면 2차원 좌표를 동차 좌표로 올려야 하므로, 아래 식에서는 점을 $(u, v, 1)^\top$로 표기한다.

두 카메라의 중심과 3D 점 $\mathbf{X}$는 한 평면(epipolar plane)을 이루기 때문에, 대응하는 두 투영 광선은 아무 위치에나 놓일 수 없다. 이 기하 조건을 대수적으로 묶은 식이 epipolar constraint다.

$$\mathbf{p}_2^\top \mathbf{F} \mathbf{p}_1 = 0$$

여기서 $\mathbf{F}$는 3×3 크기, rank 2인 Fundamental matrix다. 카메라 내부 파라미터 $\mathbf{K}_1, \mathbf{K}_2$를 이미 알고 있다면 픽셀 좌표를 정규화해 더 간결한 Essential matrix $\mathbf{E} = \mathbf{K}_2^\top \mathbf{F} \mathbf{K}_1$를 쓸 수 있다.

$$\hat{\mathbf{p}}_2^\top \mathbf{E} \hat{\mathbf{p}}_1 = 0$$

이때 $\hat{\mathbf{p}} = \mathbf{K}^{-1} \mathbf{p}$는 카메라 내부 파라미터의 영향을 걷어 낸 정규화된 이미지 좌표다.

#### 2D-3D Correspondence

이미지의 2D 픽셀과 이미 알고 있는 3D 맵 포인트 사이의 대응을 찾는 문제다. PnP(Perspective-n-Point) 문제의 직접적인 입력이 되며, Visual Localization이나 SLAM의 map-based tracking처럼 기존 지도를 기준으로 현재 카메라 위치를 잡을 때 쓰인다.

월드 좌표계의 3D 점 $\mathbf{X} = (X, Y, Z)^\top$과 이미지에 투영된 2D 점 $\mathbf{p} = (u, v)^\top$ 사이에는 핀홀 투영 모델에 따라 다음 관계가 성립한다.

$$s \begin{bmatrix} u \\ v \\ 1 \end{bmatrix} = \mathbf{K} [\mathbf{R} | \mathbf{t}] \begin{bmatrix} X \\ Y \\ Z \\ 1 \end{bmatrix}$$

이미 3차원 좌표를 알고 있으므로 2D-2D와 달리 스케일 모호성이 남지 않는다. 카메라 포즈 $(\mathbf{R}, \mathbf{t})$를 구하려면 선형 해법인 DLT로는 최소 6개의 대응점이 필요하고, 비선형 최소 해법인 P3P를 쓰면 3개(나오는 여러 해 가운데 진짜 하나를 고르려면 1개 더)만으로 추정할 수 있다.

#### 3D-3D Correspondence

두 포인트 클라우드(point cloud)에서 같은 물리적 지점을 찾는 문제다. LiDAR scan-to-scan 정합, 다중 LiDAR 캘리브레이션, loop closure 단계의 point cloud registration에서 핵심 역할을 한다.

양쪽 점군 모두 3차원 좌표를 온전히 갖고 있으므로, 두 포인트 클라우드 $\mathcal{P} = \{\mathbf{p}_i\}$와 $\mathcal{Q} = \{\mathbf{q}_i\}$를 일치시키는 강체 변환 $(\mathbf{R}, \mathbf{t})$를 구하는 일은 거리 오차 제곱합을 최소화하는 문제로 정식화된다.

$$\min_{\mathbf{R}, \mathbf{t}} \sum_{i} \| \mathbf{q}_i - (\mathbf{R} \mathbf{p}_i + \mathbf{t}) \|^2$$

누가 누구와 짝인지 대응만 알고 있다면, 이 문제에서는 SVD(특잇값 분해)를 이용해 반복 계산 없이 단번에 closed-form 해를 구할 수 있다(ICP, 7장).

### 5.1.3 센서 융합과 correspondence

앞으로 보게 될 상태 추정기(4장)와 VO/VIO(6장)의 정확도는 프런트엔드가 넘겨주는 대응점의 품질에 좌우된다. 최소자승이나 factor graph 최적화는 기본적으로 모든 잔차가 정상 분포를 따른다고 보기 때문에, 엉뚱한 점을 연결한 오매칭(outlier)이 몇 개만 섞여도 포즈 추정 오차가 순식간에 폭증한다. 반대로 흰 벽이나 안개 낀 도로처럼 텍스처가 빈약한 환경에서는 쓸 만한 대응점 자체가 충분히 나오지 않아, 방정식의 자유도를 채우지 못하고 포즈를 아예 계산하지 못하는 상황에 빠지기도 한다.

결국 "어디에서 특징을 잘 뽑고, 어떻게 구별되게 표현하며, 틀린 짝을 어떻게 걸러낼 것인가"가 전체 시스템의 강건함을 결정한다. 이제 영상에서 이 문제를 풀어 온 전통적인 접근법부터 차례대로 살펴보자.

---

## 5.2 전통적 Feature Detection & Description

전통적인 correspondence 파이프라인은 **검출(detect) → 기술(describe) → 매칭(match)**의 세 단계로 나뉜다. 먼저 영상에서 다른 시점에도 다시 잡힐 만한 안정적인 특징점(반복성)을 어디서 찾을지 정하고(검출), 그 점 주변의 국소 패턴을 다른 점과 헷갈리지 않는 숫자 벡터로 요약한 뒤(기술), 두 영상의 벡터를 견주어 짝을 짓는다(매칭).

### 5.2.1 Corner Detection: Harris → FAST → ORB

#### Harris Corner Detector (1988)

평탄한 벽면은 창(window)을 조금 옮겨도 화면이 그대로이고, 직선 경계(edge)는 선을 따라 이동하면 변화가 없다. 반면 모서리(corner)는 창을 어느 방향으로 밀어도 밝기가 크게 바뀐다. Harris corner detector는 바로 이 직관을 수학적으로 옮긴 검출기다. 이미지 $I$에서 점 $(x, y)$ 주변의 윈도우를 $(\Delta u, \Delta v)$만큼 옮겼을 때의 가중 밝기 변화량은 다음과 같다.

$$E(\Delta u, \Delta v) = \sum_{x, y} w(x, y) [I(x + \Delta u, y + \Delta v) - I(x, y)]^2$$

여기서 $w(x, y)$는 중심에 더 큰 무게를 두는 가우시안 윈도우다. 작은 변위 $(\Delta u, \Delta v)$에 대해 1차 Taylor 전개를 적용하면 이 식은 2차 형식으로 깔끔하게 정리된다.

$$E(\Delta u, \Delta v) \approx \begin{bmatrix} \Delta u & \Delta v \end{bmatrix} \mathbf{M} \begin{bmatrix} \Delta u \\ \Delta v \end{bmatrix}$$

이때 등장하는 $2 \times 2$ 행렬 $\mathbf{M}$이 **structure tensor**(또는 second moment matrix)다.

$$\mathbf{M} = \sum_{x, y} w(x, y) \begin{bmatrix} I_x^2 & I_x I_y \\ I_x I_y & I_y^2 \end{bmatrix}$$

$I_x, I_y$는 이미지의 $x, y$ 방향 그래디언트다. $\mathbf{M}$의 두 고유값 $(\lambda_1, \lambda_2)$는 직교하는 주방향으로 창을 밀었을 때 밝기가 얼마나 변하는지를 나타내므로, 그 크기에 따라 픽셀의 기하적 성격을 곧바로 판별할 수 있다.

- $\lambda_1 \approx 0, \lambda_2 \approx 0$: 어느 쪽으로 움직여도 변화가 없는 평탄한 영역
- $\lambda_1 \gg \lambda_2 \approx 0$: 한 방향으로만 밝기가 바뀌는 에지
- $\lambda_1, \lambda_2$ 모두 큼: 모든 방향으로 밝기가 변하는 코너

다만 모든 픽셀에서 고유값을 직접 구하면 계산량이 부담스럽다. Harris는 행렬식과 대각합이 각각 두 고유값의 곱($\lambda_1 \lambda_2$)과 합($\lambda_1 + \lambda_2$)이라는 성질을 이용해, 고유값 분해 없이 코너 여부를 가늠하는 corner response function을 정의했다.

$$R = \det(\mathbf{M}) - k \cdot \text{tr}(\mathbf{M})^2 = \lambda_1 \lambda_2 - k(\lambda_1 + \lambda_2)^2$$

경험적 상수 $k$로는 보통 0.04~0.06을 쓴다. 응답값 $R$이 임계값을 넘는 점을 추린 뒤, 주변 최댓값만 남기는 non-maximum suppression(NMS)을 거치면 최종 코너가 된다.

**Harris의 한계**: $\mathbf{M}$의 고유값은 좌표계 회전에 영향을 받지 않으므로 회전에는 불변(rotation invariant)이다. 그러나 윈도우 크기가 고정되어 있어 스케일 변화에는 약하다. 카메라가 물체에 가까이 다가가면, 예전에는 윈도우 안에 쏙 들어오던 코너가 확대되면서 완만한 에지처럼 변해 코너로 잡히지 않을 수 있다. 뒤 절에서 볼 SIFT는 scale-space를 도입해 이 한계를 정면으로 해결한다.

OpenCV에서는 `cornerHarris` 함수로 각 픽셀의 응답 $R$을 바로 계산할 수 있다.

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

Harris는 수학적으로 우아하지만, 실시간 SLAM이나 임베디드 보드에서 매 프레임 그래디언트와 윈도우 합산을 돌리기에는 여전히 무겁다. FAST는 미분 연산을 아예 버리고 단순 밝기 비교만으로 속도를 극한까지 끌어올린 코너 검출기다. Rosten과 Drummond가 ICCV 2005에서 segment test 검출기를 처음 내놓았고, ECCV 2006에서 여기에 결정 트리 학습을 결합해 속도를 한층 더 높였다. 확장판인 [Rosten, Porter & Drummond (TPAMI 2010)](https://arxiv.org/abs/0810.2434)의 측정에 따르면 PAL 실시간 영상을 처리할 때 FAST는 가용 처리 시간의 5% 미만만 소비했는데, 같은 조건에서 Harris는 115%, SIFT는 195%를 넘겨 실시간을 따라가지 못했다.

FAST가 코너를 판별하는 절차는 직관적이다.

1. 후보 픽셀 $p$를 중심으로 반지름 3인 원(Bresenham circle) 위의 주변 픽셀 16개를 살펴본다.
2. 원 위에서 연속으로 이어진 $N$개 픽셀(2005년 원래 검출기는 $N=12$, 반복성이 가장 좋았던 값이자 OpenCV 기본값은 $N=9$)이 모두 중심 픽셀 $p$보다 일정 수준 이상 밝거나 모두 어두우면 $p$를 코너로 판정한다.
3. **빠른 기각(high-speed test)**: 매번 16개 픽셀을 다 확인하지 않고, 1, 5, 9, 13번(상·하·좌·우) 픽셀 넷만 먼저 훑는다. $N=12$ 조건이 성립하려면 이 네 픽셀 중 적어도 셋이 모두 밝거나 모두 어두워야 하므로, 이 기준을 통과한 점만 전체 검사로 넘긴다. 다만 이 단축 검사는 $N=12$ 전용 논리여서, $N=9$ 같은 변형에서는 아래의 결정 트리 학습이 그 역할을 대신한다.

$$\text{FAST condition: } \exists \text{ contiguous arc of } N \text{ pixels on circle, all } > I_p + t \text{ or all } < I_p - t$$

여기서 $t$는 노이즈에 휘둘리지 않도록 두는 밝기 임계값이다.

**Decision tree 학습**: 16개 픽셀을 어떤 순서로 검사해야 코너가 아닌 대다수의 픽셀을 가장 적은 비교 횟수로 탈락시킬 수 있을까? FAST는 학습 영상에서 ID3 decision tree를 훈련해 최적의 픽셀 검사 순서를 자동으로 찾아낸다.

**FAST의 한계**: FAST는 극도로 빠르게 "코너 위치"만 짚어 줄 뿐이다. 영상이 돌아갔을 때 기준을 잡아 줄 방향(orientation), 크기가 달라졌을 때 맞출 스케일(scale), 그리고 매칭에 쓸 디스크립터는 제공하지 않는다. 이 빈자리를 채워 실전 SLAM의 표준으로 자리 잡은 검출기·디스크립터가 바로 ORB다.

OpenCV에서는 `FastFeatureDetector_create`로 손쉽게 키포인트를 뽑아낼 수 있다.

```python
# FAST Corner Detection
fast = cv2.FastFeatureDetector_create(threshold=20, nonmaxSuppression=True)
keypoints = fast.detect(img, None)
print(f"Detected {len(keypoints)} keypoints")
```

#### ORB (Oriented FAST and Rotated BRIEF, 2011)

2000년대 후반까지 성능의 기준선이던 SIFT와 SURF는 특허로 묶여 상용 제품이나 오픈소스 배포에 제약이 컸고, 모바일·로봇의 CPU만으로 실시간 처리하기에도 무거웠다. ORB는 [Rublee et al. (2011)](https://ieeexplore.ieee.org/document/6126544)이 이러한 특허 제약과 속도 문제를 함께 풀기 위해 제안한 알고리즘이다. 핵심은 FAST 검출기에 방향성을 부여하고(**oFAST**), 빠른 이진 디스크립터인 BRIEF를 그 방향에 맞춰 회전시켜 쓰는 데 있다(**rBRIEF**).

**oFAST (Oriented FAST)**:
FAST로 찾은 키포인트에 방향을 매기려면 복잡한 그래디언트 히스토그램 대신 패치의 밝기 무게중심을 재는 **intensity centroid** 방법을 쓴다. 키포인트 주변 패치의 이미지 모멘트를 먼저 구한 뒤,

$$m_{pq} = \sum_{x, y} x^p y^q I(x, y)$$

패치의 기하 중심에서 밝기 무게중심 $\mathbf{C} = (m_{10}/m_{00}, m_{01}/m_{00})$을 향하는 각도 $\theta = \text{atan2}(m_{01}, m_{10})$를 키포인트의 대표 방향으로 삼는다.

**rBRIEF (Rotated BRIEF)**:
BRIEF는 패치 안에서 미리 정한 점 쌍 $(x_i, y_i)$의 밝기를 단순 비교해 0 또는 1의 이진 비트를 만든다.

$$\tau(\mathbf{p}; x_i, y_i) = \begin{cases} 1 & \text{if } I(\mathbf{p}, x_i) < I(\mathbf{p}, y_i) \\ 0 & \text{otherwise} \end{cases}$$

이 비교를 256번 수행해 묶으면 256비트 이진 디스크립터가 된다. 원래 BRIEF는 카메라가 기울어지면 점 쌍의 상대 위치가 틀어져 매칭이 무너지지만, ORB는 oFAST에서 구한 방향 $\theta$만큼 점 쌍 좌표를 회전(steer)시켜 회전 불변성을 확보한다. 게다가 무작위로 점 쌍을 뽑으면 비슷한 정보만 중복 측정하기 쉽다. 그래서 ORB는 오프라인 학습 데이터에서 분산이 크고 서로 상관관계가 낮은 점 쌍 256개를 탐욕적(greedy) 방식으로 선별해 분별력(discriminability)을 높였다.

**멀티 스케일**: 스케일 변화에 대응하기 위해 이미지 피라미드(OpenCV 구현의 기본값은 8단계)를 쌓고, 해상도 레벨마다 FAST를 돌려 특징점을 모은다.

이 조합 덕분에 ORB-SLAM 시리즈를 비롯한 수많은 실시간 SLAM 시스템이 ORB를 채택했다. 256비트 이진 벡터는 CPU의 비트 연산과 Hamming distance만으로 순식간에 매칭할 수 있기 때문이다.

```python
# ORB Detection & Description
orb = cv2.ORB_create(nfeatures=1000)
keypoints, descriptors = orb.detectAndCompute(img, None)
# descriptors.shape: (N, 32) — 256-bit = 32 bytes
```

### 5.2.2 Blob Detection: SIFT → SURF

코너 검출기가 날카로운 모서리 한 점에 주목한다면, 블롭(blob) 검출기는 주변과 대비를 이루는 덩어리 영역을 다양한 크기(스케일)에 걸쳐 탐색한다. 그 정점에 있는 알고리즘이 SIFT다.

#### SIFT (Scale-Invariant Feature Transform, 2004)

[Lowe (2004)](https://link.springer.com/article/10.1023/B:VISI.0000029664.99615.94)가 제안한 SIFT는 스케일 변화와 평면 내 회전에 완전히 불변일 뿐 아니라, 국소 그래디언트 방향을 정규화해 담는 디스크립터 덕분에 조명 변화와 약한 affine 왜곡에도 놀라울 만큼 잘 버틴다. SIFT는 발표 이후 20년 가까이 특징점 매칭의 절대적 기준선(baseline)으로 군림했으며, 미국 특허(US 6,711,293)는 2020년 3월에 만료되어 지금은 제약 없이 쓸 수 있다.

SIFT는 네 단계의 정교한 파이프라인을 거친다.

**Stage 1 — Scale-Space Extrema Detection (DoG)**:

물체와의 거리가 변해도 같은 특징점을 잡으려면 이미지를 한 가지 크기로만 보지 말고 여러 스케일로 흐리게(blur) 만들어 입체적으로 탐색해야 한다. 이 다중 스케일 공간을 scale-space라 부르며, 원 영상 $I(x, y)$에 스케일 $\sigma$의 가우시안 커널을 합성곱해 만든다.

$$L(x, y, \sigma) = G(x, y, \sigma) * I(x, y)$$

여기서 $G(x, y, \sigma) = \frac{1}{2\pi\sigma^2} \exp\left(-\frac{x^2 + y^2}{2\sigma^2}\right)$다. 이론적으로 스케일 불변 블롭 검출에 가장 이상적인 연산자는 정규화된 Laplacian of Gaussian(LoG)이지만, 스케일마다 2차 미분을 계산하는 비용이 만만치 않다. Lowe는 인접한 두 가우시안 블러 영상의 단순 뺄셈인 **Difference of Gaussians(DoG)**로 LoG를 훌륭하게 근사했다.

$$D(x, y, \sigma) = L(x, y, k\sigma) - L(x, y, \sigma) \approx (k-1)\sigma^2 \nabla^2 G$$

이렇게 만든 DoG 피라미드에서 한 픽셀을 총 26개 이웃(같은 스케일의 8개와 위아래 인접 스케일의 9개씩)과 비교해 가장 크거나 가장 작은 극값(extrema)을 후보 키포인트로 뽑는다.

**Stage 2 — Keypoint Localization**:

이산적인 픽셀 격자와 스케일 간격에서 뽑은 극값은 아직 거칠다. SIFT는 scale-space에서 2차 Taylor 전개를 적용해 극값의 정밀 위치를 서브픽셀·서브스케일 수준으로 보정한다.

$$D(\mathbf{x}) = D + \frac{\partial D}{\partial \mathbf{x}}^\top \mathbf{x} + \frac{1}{2} \mathbf{x}^\top \frac{\partial^2 D}{\partial \mathbf{x}^2} \mathbf{x}$$

도함수가 0이 되는 극값의 오프셋은 $\hat{\mathbf{x}} = -\frac{\partial^2 D}{\partial \mathbf{x}^2}^{-1} \frac{\partial D}{\partial \mathbf{x}}$로 구한다. 그런 다음 실제 환경에서 흔들리기 쉬운 두 가지 후보를 쳐낸다. 첫째, 픽셀 값을 $[0, 1]$로 정규화했을 때 $|D(\hat{\mathbf{x}})| < 0.03$인 저대비(low-contrast) 점은 노이즈일 가능성이 크므로 버린다. 둘째, DoG는 직선 에지 위에서도 강한 응답을 내기 쉬우므로, 2×2 Hessian 행렬의 두 고유값 비율을 확인해 한쪽으로만 곡률이 큰 에지 위의 점들을 걸러낸다.

**Stage 3 — Orientation Assignment**:

이제 키포인트의 대표 방향을 정해 회전 불변성을 부여할 차례다. 표준편차가 키포인트 스케일의 1.5배($\sigma = 1.5 \times \text{scale}$)인 가우시안 가중치를 적용해 주변 그래디언트 크기와 방향으로 36-bin(10도 간격) 방향 히스토그램을 만든다. 가장 높은 봉우리를 키포인트의 주 방향으로 삼되, 그 최댓값의 80%를 넘는 보조 봉우리가 또 있다면 그 방향의 키포인트를 하나 더 복제해 둔다. 덕분에 방향이 모호한 코너에서도 매칭 성공률이 크게 올라간다.

**Stage 4 — Keypoint Descriptor**:

마지막으로 키포인트의 주 방향에 맞춰 좌표축을 돌린 뒤, 주변 $16 \times 16$ 영역을 $4 \times 4$ 크기의 서브 블록 16개로 쪼갠다. 블록마다 8방향 그래디언트 히스토그램을 누적해 이어 붙이면 SIFT 고유의 128차원 벡터가 완성된다.

$$\text{Descriptor} = 4 \times 4 \times 8 = 128\text{-dimensional vector}$$

이 벡터에 L2 정규화를 적용하면 전체적인 밝기 스케일 변화(대비 변화)가 상쇄된다. 한 걸음 더 나아가, 정규화 후 값이 0.2를 넘는 원소를 0.2로 잘라 내고(clipping) 다시 한번 L2 정규화한다. 정반사(specular highlight)처럼 비선형 조명 변화로 특정 방향의 그래디언트만 유독 치솟더라도 나머지 성분이 묻히지 않도록 보호하는 영리한 장치다.

OpenCV에서는 다음과 같이 SIFT 키포인트와 128차원 부동소수점 디스크립터를 추출한다.

```python
# SIFT Detection & Description
sift = cv2.SIFT_create(nfeatures=2000)
keypoints, descriptors = sift.detectAndCompute(img, None)
# descriptors.shape: (N, 128) — 128-dimensional float32
```

#### SURF (Speeded-Up Robust Features, 2006)

SIFT의 정확도는 뛰어났지만, 가우시안 피라미드와 128차원 히스토그램 연산은 당시 CPU로 실시간 처리하기 어려웠다. SURF는 [Bay et al. (2006)](https://link.springer.com/chapter/10.1007/11744023_32)이 SIFT의 강건함을 최대한 유지하면서 속도를 끌어올리기 위해 내놓은 알고리즘이다. 계산량을 줄이는 핵심 기법은 세 가지다.

- **Integral image 활용**: 가우시안 2차 미분 필터를 박스(box) 형태의 근사 필터로 바꾸고, 미리 누적합 이미지(integral image)를 계산해 둔다. 덕분에 필터 크기가 아무리 커져도 덧셈·뺄셈 몇 번이면 상수 시간 안에 필터 응답을 얻는다.
- **Hessian determinant 기반 검출**: DoG 대신 **Hessian 행렬식**으로 특징점을 고른다.

$$\det(\mathbf{H}) = D_{xx} D_{yy} - (0.9 \cdot D_{xy})^2$$

- **64차원 Haar wavelet 디스크립터**: 서브 영역마다 x·y 방향 Haar wavelet 응답의 합과 절댓값 합 4가지를 모아 $4 \times 4 \times 4 = 64$차원 벡터를 만든다. 차원이 SIFT(128차원)의 절반이라 기술과 매칭이 모두 가볍다.

실제로 SIFT 대비 얼마나 빠르고 정확한지는 구현 최적화, 영상 크기, 하드웨어 환경에 따라 엇갈린다. 게다가 SURF는 여전히 특허 제약 때문에 OpenCV에서도 기본 패키지가 아닌 `xfeatures2d`(non-free) 모듈로 분리되어 있어 사용이 번거롭다. 이런 현실적인 이유로, 오늘날 실시간 SLAM(ORB-SLAM 계열)은 ORB를, 오프라인 3D 복원(COLMAP 등)처럼 극한의 정밀도가 필요한 곳은 특허가 만료된 SIFT를 주로 선택한다.

### 5.2.3 Binary Descriptors: BRIEF, ORB, BRISK

SIFT(128개 `float32` = 512바이트)나 SURF(64개 `float32` = 256바이트) 같은 부동소수점 디스크립터는 메모리도 많이 차지하고, 매칭할 때마다 L2 거리(제곱과 덧셈)를 구해야 해서 부담이 크다. 이를 해결하기 위해 등장한 이진 디스크립터는 패치 안 점 쌍의 밝기 대소를 비교해 0과 1의 비트열로 압축한다. 두 이진 벡터의 거리는 비트 XOR와 `popcount`(1의 개수를 세는 명령)로 구한다. 이 계산은 CPU가 단 몇 사이클 만에 처리하므로 부동소수점 거리 계산보다 대개 훨씬 가볍다. 실제로 몇 배나 빨라지는지는 디스크립터 길이, 라이브러리, 하드웨어 아키텍처에 따라 달라진다.

대표적인 이진 디스크립터의 특징을 정리하면 다음과 같다.

| 디스크립터 | 비트 수 | 특징 |
|-----------|--------|------|
| BRIEF | 128/256/512 | 랜덤 점 쌍, 회전 불변 아님 |
| ORB | 256 | 학습된 점 쌍, 회전 불변 |
| BRISK | 512 | 동심원 샘플링, 스케일 불변 |

가장 원형에 가까운 BRIEF의 비트 생성 규칙은 패치 안 점 쌍 $(\mathbf{p}_i, \mathbf{q}_i)$의 밝기 비교를 그대로 정수로 묶는 형태다.

$$b_i = \begin{cases} 1 & \text{if } I(\mathbf{p}_i) < I(\mathbf{q}_i) \\ 0 & \text{otherwise} \end{cases}, \quad \text{descriptor} = \sum_{1 \le i \le n} 2^{i-1} b_i$$

이렇게 만들어진 두 비트열 $\mathbf{a}, \mathbf{b}$의 Hamming distance는 XOR와 `popcount`의 조합으로 한 줄에 계산된다. 64비트 워드 단위 CPU 명령을 쓰므로, 256비트 디스크립터라면 워드 4개만 XOR·popcount해 더하면 끝이고, 512비트는 그 두 배면 충분하다.

$$d_H(\mathbf{a}, \mathbf{b}) = \text{popcount}(\mathbf{a} \oplus \mathbf{b})$$

### 5.2.4 전통적 특징점의 계보

지금까지 살펴본 전통적 특징점 알고리즘들은 "스케일 불변성"과 "실시간 속도"라는 두 가지 요구 사이에서 발전해 왔다. 발표 순서대로 그 계보를 짚어 보면 흐름이 한눈에 들어온다.

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

이제 각 영상에서 특징점과 디스크립터를 뽑아냈으니, 다음 과제는 양쪽 디스크립터를 실제로 짝짓고 그 속에 숨어 든 오매칭을 솎아 내는 일이다.

---

## 5.3 전통적 Matching & Outlier Rejection

두 영상에서 디스크립터를 모두 뽑았다면, 이제 같은 3D 점에서 유래한 특징점끼리 짝지어야 한다. 하지만 단순히 생김새가 가장 비슷한 점끼리만 연결하면 반복 패턴이나 가려짐 때문에 엉뚱한 오매칭이 수두룩하게 섞인다. 따라서 전통적 매칭 단계에서는 최근접 이웃을 빠르게 찾는 탐색과, 기하학적 일관성으로 아웃라이어를 도려내는 검증이 짝을 이뤄 동작한다.

### 5.3.1 Brute-Force Matching

가장 곧이곧대로 접근하는 방법이 전수 탐색(brute-force matching)이다. 이미지 A의 디스크립터 하나하나를 이미지 B의 모든 디스크립터와 빠짐없이 비교해 거리가 가장 가까운 상대를 고른다. 디스크립터의 성격에 따라 거리 척도를 달리 쓰며, 계산 비용은 특징점 수의 곱에 비례한다.

- **Float 디스크립터 (SIFT)**: L2 거리(유클리드 거리)
- **Binary 디스크립터 (ORB)**: Hamming 거리
- **시간 복잡도**: $O(N \cdot M \cdot D)$ — $N, M$은 각 이미지의 특징점 수, $D$는 디스크립터 차원

A에서 B로 짝을 찾았더라도 B에서 다시 A를 바라보았을 때 다른 점이 더 가깝다면 그 매칭은 미심쩍다. 아래 코드처럼 `crossCheck=True`를 켜 두면, A→B 최근접과 B→A 최근접이 서로를 지목하는 쌍(mutual nearest neighbor)만 살려 일방적인 오매칭을 꽤 줄일 수 있다.

```python
# Brute-Force Matching
bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)  # SIFT
# bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)  # ORB
matches = bf.match(desc1, desc2)
matches = sorted(matches, key=lambda x: x.distance)
```

### 5.3.2 FLANN (Fast Library for Approximate Nearest Neighbors)

이미지 한 장에 특징점이 수천 개씩 쌓이면 $O(N \cdot M \cdot D)$ 비용의 전수 탐색은 금세 병목이 된다. 이때 가장 가까운 이웃을 정확히 찾는 대신 "거의 가장 가까운 이웃"을 재빨리 찾아주는 근사 최근접 이웃 탐색(approximate nearest neighbor, ANN)이 현실적인 대안이다. FLANN은 randomized kd-tree나 hierarchical k-means tree 같은 인덱스 구조를 활용해 탐색 공간을 좁히며, 데이터 분포에 맞는 알고리즘과 파라미터를 자동으로 골라 쓰는 기능도 갖추고 있다.

```python
# FLANN Matching for SIFT
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)  # 탐색 노드 수 제한

flann = cv2.FlannBasedMatcher(index_params, search_params)
matches = flann.knnMatch(desc1, desc2, k=2)  # k=2 for ratio test
```

### 5.3.3 Lowe's Ratio Test

단순히 "거리가 일정 기준보다 가까우면 통과"라는 절대 임계값을 쓰면 낭패를 보기 쉽다. 벽돌 벽이나 창문 격자처럼 비슷한 패턴이 줄지어 있는 곳에서는, 틀린 후보끼리도 디스크립터 거리가 매우 가깝게 나오기 때문이다. Lowe(2004)가 SIFT 논문에서 제안한 ratio test는 절대 거리가 아니라 **1등과 2등 후보 사이의 격차**를 본다. 가장 가까운 디스크립터까지의 거리와 두 번째로 가까운 것까지의 거리 비가 임계값 $\tau$ 미만일 때만 고유한 매칭으로 인정한다.

$$\frac{d(\mathbf{f}, \mathbf{f}_1)}{d(\mathbf{f}, \mathbf{f}_2)} < \tau$$

여기서 $\mathbf{f}_1, \mathbf{f}_2$는 각각 가장 가까운 디스크립터와 두 번째로 가까운 디스크립터며, 임계값 $\tau$는 보통 0.7~0.8(Lowe의 원 논문은 0.8)을 쓴다. 올바른 매칭이라면 진짜 상대인 1등이 2등보다 압도적으로 가까워 $d_1 \ll d_2$가 되지만, 모호한 반복 패턴이나 잘못된 매칭에서는 비슷한 후보가 여럿 몰려 있어 $d_1 \approx d_2$가 되기 쉽다.

앞선 `knnMatch(..., k=2)`에서 뽑아 둔 상위 두 후보에 ratio test를 거는 코드는 다음과 같다.

```python
# Ratio test (Lowe's)
good_matches = []
for m, n in matches:  # m: best, n: second best
    if m.distance < 0.75 * n.distance:
        good_matches.append(m)
```

### 5.3.4 RANSAC 계열: RANSAC → PROSAC → MAGSAC++

Ratio test와 cross-check를 통과하더라도 겉모습이 우연히 닮은 아웃라이어(outlier)는 여전히 남는다. 국소 패치만 쳐다보는 디스크립터 단계에서는 각 매칭이 진짜인지 가짜인지 가려낼 방법이 없다. 결국 두 영상 전체가 공유하는 카메라 움직임, 즉 Fundamental matrix나 Essential matrix 같은 **기하 모델**을 맞추면서 그 모델에 어긋나는 아웃라이어를 통째로 걸러내야 한다.

#### RANSAC (Random Sample Consensus, 1981)

일반적인 최소자승법은 아웃라이어 단 몇 개만 섞여도 추정 모델이 망가진다. [Fischler & Bolles (1981)](https://dl.acm.org/doi/10.1145/358669.358692)이 제안한 **RANSAC**은 아예 전체 데이터를 한꺼번에 맞추려 하지 않고, 모델을 결정할 최소한의 표본만 무작위로 뽑아 반복 검증하는 발상으로 이 문제를 돌파했다.

1. 매칭 가운데 기하 모델을 세우는 데 필요한 최소 개수 $n$개를 무작위로 뽑는다(Fundamental matrix는 8점 또는 7점, Essential matrix는 5점).
2. 뽑은 최소 표본만으로 후보 모델을 추정한다.
3. 전체 매칭을 이 모델에 대입해 재투영 오차나 epipolar 오차가 임계값 $t$ 이내인 점(인라이어)을 모아 consensus set을 만든다.
4. 1~3단계를 충분히 반복해 consensus set이 가장 큰 최선의 모델을 고른다.
5. 마지막으로 그 인라이어 집합 전체를 한데 모아 모델을 정밀하게 다시 추정한다.

그렇다면 "적어도 한 번은 아웃라이어가 하나도 섞이지 않은 깨끗한 표본"을 뽑으려면 몇 번이나 반복해야 할까? 인라이어 비율을 $w$, 최소 표본 수를 $n$, 목표 성공 확률을 $p$(보통 0.99)라 두면 필요한 반복 횟수 $k$는 다음 식으로 계산된다.

$$k = \frac{\log(1 - p)}{\log(1 - w^n)}$$

가령 인라이어가 절반($w = 0.5$)뿐인 상황에서 8점 모델($n = 8$)을 $p=0.99$ 확률로 성공시키려면 $k \approx 1177$번의 시도가 필요하다. 인라이어 비율이 조금만 떨어져도 $w^n$이 급격히 작아지면서 반복 횟수가 폭발하는 셈이다.

OpenCV에서는 `findFundamentalMat`에 `FM_RANSAC` 플래그를 넘겨 모델 추정과 인라이어 마스크 추출을 한 번에 처리한다.

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

RANSAC은 모든 매칭이 인라이어일 가능성을 똑같이 보고 아무 쌍이나 뽑는다. 하지만 디스크립터 거리나 Lowe's ratio 같은 매칭 품질 점수는 이미 주어져 있다. Chum과 Matas가 CVPR 2005에서 제안한 **PROSAC**은 이 점수를 버리지 않고 활용한다. 매칭을 품질 순으로 정렬해 둔 뒤, 가장 믿을 만한 최상위 소수 후보에서 표본을 뽑기 시작해 점차 탐색 범위를 전체로 넓혀 나간다. 품질 순위가 높을수록 실제로 인라이어일 확률이 크다면, 무작위 탐색보다 훨씬 적은 반복 횟수로 정답 모델에 도달한다. 반대로 매칭 품질 순위가 실제 인라이어 여부와 별 상관이 없는 데이터라면, 탐색이 뒤로 밀리면서 일반 RANSAC과 같은 속도로 수렴한다.

#### MAGSAC / MAGSAC++ (2019/2020)

RANSAC 계열을 실전에 적용할 때 늘 따라다니는 고민은 인라이어를 가르는 거리 임계값 $t$를 사람이 직접 정해야 한다는 점이다. 너무 좁게 잡으면 노이즈가 조금만 끼어도 정상 대응점을 버리고, 너무 넓게 잡으면 아웃라이어가 모델에 스며든다. Barath 등이 제안한 MAGSAC(CVPR 2019)과 그 개량판 MAGSAC++(CVPR 2020)는 임계값 하나를 딱 잘라 고르지 않는 발상으로 임계값 설정에 따른 민감도를 낮췄다.

- 인라이어 임계값 $\sigma$를 하나의 고정값이 아닌 $[0, \sigma_{\max}]$ 구간의 확률 변수로 보고, 가능한 모든 임계값에 걸쳐 모델 품질을 적분(marginalize)해 평가한다.

$$Q(\theta) = \int_0^{\sigma_{\max}} q(\theta, \sigma) f(\sigma) d\sigma$$

여기서 $q(\theta, \sigma)$는 임계값이 $\sigma$일 때 모델 $\theta$의 품질이고, $f(\sigma)$는 임계값의 사전 분포다.

- 후속작인 **MAGSAC++**는 이 적분 아이디어를 훨씬 빠르게 계산할 수 있는 가중치 함수로 다시 빚어내고, 각 점이 모델에 기여하는 정도에 따라 가중치를 주는 반복 가중 최소자승(IRLS)으로 모델을 다듬는다.
- 특정 임계값 하나에 의존하지 않으므로 임계값 설정에 훨씬 덜 흔들린다. 물론 적분 상한 $\sigma_{\max}$는 여전히 정해 주어야 한다. 참고로 OpenCV의 `USAC_MAGSAC` 구현에서는 이 상한을 사용자가 넘긴 `ransacReprojThreshold`와 모델별 내부 기본값(fundamental matrix의 경우 2.5) 중 큰 쪽으로 잡으며, `ransacReprojThreshold` 자체는 알고리즘 종료 판정과 최종 반환 `mask`의 인라이어 구분에 활용한다.

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

앞서 5.1.2에서 정의한 두 카메라 사이의 기하 관계는 실제 파이프라인에서 RANSAC과 결합해 다음처럼 풀린다. 카메라 내부 파라미터를 모른다면 **Fundamental Matrix($\mathbf{F}$, 7 DOF)**를 추정한다. 대응점 8개 이상이 있으면 8-point algorithm을 써서 선형 방정식으로 $\mathbf{F}$를 푼 뒤 SVD로 rank-2 제약을 건다. 한편 최소 개수인 7개만 뽑는 7-point algorithm에서는 rank-2 조건인 $\det(\mathbf{F})=0$에서 나오는 3차 다항식을 풀어 최대 3개의 후보 해를 얻는다.

반면 카메라 캘리브레이션이 끝나 내부 파라미터 $\mathbf{K}$를 안다면 자유도가 5로 줄어든 **Essential Matrix($\mathbf{E} = [\mathbf{t}]_\times \mathbf{R}$, 5 DOF)**를 직접 추정할 수 있어 한결 안정적이다. Nistér(2004)의 5-point algorithm은 최소 5개의 대응점에서 10차 다항식을 풀어 최대 10개의 후보 해를 구한다. 이 알고리즘을 RANSAC 루프 안에서 돌려 아웃라이어를 거르고 나면 `recoverPose`로 실제 회전 $\mathbf{R}$과 스케일이 정규화된 병진 $\mathbf{t}$를 복원할 수 있다.

```python
# Essential matrix 추정 (calibrated camera)
E, mask = cv2.findEssentialMat(pts1, pts2, cameraMatrix=K,
                                method=cv2.RANSAC,
                                prob=0.999, threshold=1.0)

# Essential matrix에서 R, t 복원
_, R, t, mask_pose = cv2.recoverPose(E, pts1, pts2, cameraMatrix=K)
```

### 5.3.6 로버스트 추정의 계보

아웃라이어가 판치는 현실 데이터에서 기하 모델을 지키는 로버스트 추정 기법은 무작위 샘플링에서 출발해 품질 순위 활용과 임계값 적분을 거쳐, 최근에는 학습 기반으로 RANSAC을 제거하는 방식으로 이어졌다.

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

지금까지 다룬 특징점 기반 매칭은 같은 종류의 센서, 그중에서도 시각적 코너나 블롭이 선명하게 드러나는 영상에서 위력을 발휘한다. 그렇다면 서로 물리적 측정 원리가 달라 밝기나 질감 표현이 전혀 다른 센서끼리(예: 카메라 영상과 LiDAR 반사 강도) 정합해야 할 때는 어떻게 대응을 찾을까? 이제 국소 특징점을 뽑지 않고 신호 전체의 통계적 의존성을 겨냥하는 정보 이론 기반 정합으로 시선을 돌려보자.

---

## 5.4 Mutual Information & Intensity-Based Registration

### 5.4.1 MI의 정의와 직관

**Mutual Information(MI, 상호 정보량)**은 한 확률 변수의 값을 알았을 때 다른 확률 변수의 불확실성이 얼마나 줄어드는지를 재는 정보 이론의 척도다. 두 이산 확률 변수 $X, Y$ 사이의 상호 정보량은 결합 분포 $p(x, y)$와 주변 분포의 곱 $p(x)p(y)$가 얼마나 다른지(KL divergence)로 정의된다.

$$I(X; Y) = \sum_{x \in X} \sum_{y \in Y} p(x, y) \log \frac{p(x, y)}{p(x) p(y)}$$

연속 확률 변수라면 합 대신 적분을 취한다.

$$I(X; Y) = \int \int p(x, y) \log \frac{p(x, y)}{p(x) p(y)} \, dx \, dy$$

엔트로피(entropy) 관점으로 풀어서 적으면 MI의 뜻이 더욱 선명해진다.

$$I(X; Y) = H(X) + H(Y) - H(X, Y)$$

여기서 각 항의 의미는 다음과 같다.
- $H(X) = -\sum_x p(x) \log p(x)$: 변수 $X$ 하나가 품은 불확실성(엔트로피)
- $H(X, Y) = -\sum_{x,y} p(x, y) \log p(x, y)$: 두 변수 $X, Y$를 함께 놓고 보았을 때의 결합 엔트로피

**직관**: 두 이미지가 공간적으로 딱 맞게 정렬(aligned)되어 있다면, 한 이미지의 어느 픽셀 밝기를 보았을 때 다른 이미지의 같은 자리 밝기가 어떤 값일지 훨씬 좁은 범위로 예측할 수 있다. 두 이미지가 따로 놀지 않고 강하게 얽히면서 결합 엔트로피 $H(X, Y)$가 줄어들고, 그 결과 $I(X; Y)$가 커진다. 반대로 정렬이 조금만 어긋나도 엉뚱한 픽셀끼리 짝지어지며 통계적 질서가 흐려져 $I(X; Y)$가 뚝 떨어진다. 게다가 MI는 두 변수 사이의 비선형적인 의존성까지 포착하므로, 단순 선형 관계만 따지는 상관계수(correlation coefficient)가 놓치는 복잡한 관계도 잡아낸다.

### 5.4.2 MI 기반 다중 모달리티 정합

MI가 정합 분야에서 각광받는 가장 큰 이유는 종류가 다른(multimodal) 센서 데이터를 별다른 특징점 변환 없이 바로 맞출 수 있기 때문이다. 1995년 무렵 Viola & Wells, 그리고 Collignon·Maes 등이 의료 영상에서 CT와 MRI를 정합하는 데 처음 도입하면서 그 위력이 널리 알려졌다.

서로 다른 모달리티 사이에서도 MI가 잘 통하는 까닭은 밝기의 "절댓값"이나 "비례식"을 묻지 않기 때문이다. 같은 인체를 찍어도 뼈는 CT에서 하얗게 나오고 MRI에서는 까맣게 나올 수 있다. 두 영상의 밝기 차이 제곱합(SSD)이나 정규화 상관계수(NCC) 같은 전통적 척도는 두 밝기가 같거나 선형적으로 비례한다는 가정을 깔고 있어 이런 상황에서 무력하다. 반면 MI는 오직 두 밝기 분포 사이의 **통계적 의존성**이 얼마나 강한지만 측정하므로, 두 센서의 밝기 매핑이 음의 상관관계든, 비선형이든, 심지어 단조(monotonic) 관계가 아니든 거뜬히 동작한다.

로보틱스에서는 바로 이 성질 덕분에 카메라와 LiDAR의 정합(→ 3장)에 MI를 즐겨 쓴다. LiDAR가 재는 레이저 반사 강도(intensity)와 카메라가 찍는 RGB 밝기는 물리적 단위가 전혀 다르지만, 흰 차선이나 재귀반사 표지판처럼 두 센서 모두에서 유독 뚜렷하게 구분되는 표면이 많다. 따라서 두 데이터가 올바른 외부 파라미터로 겹치면 통계적 의존성이 극대화되어 MI가 정점을 찍는다.

### 5.4.3 MI와 NMI의 실용적 계산

실제 영상 정합에서는 확률 분포 $p(x), p(y), p(x, y)$의 해석적 식을 모르므로 **결합 히스토그램(joint histogram)**으로 분포를 근사한다. 이미지 $A$와 $B$를 현재 변환 $T$로 겹쳐 놓은 상태에서 다음 순서로 값을 얻는다.

1. 두 영상이 겹치는 모든 좌표 $(u, v)$에서 $A(u, v)$와 $B(T(u, v))$의 밝기 쌍을 수집한다.
2. 수집한 밝기 쌍을 2차원 히스토그램에 쌓은 뒤 전체 픽셀 수로 나누어 결합 분포 $p(a, b)$를 추정한다.
3. 결합 히스토그램을 행과 열 방향으로 각각 더해 주변 분포 $p(a), p(b)$를 구한다.
4. 앞의 식에 대입해 엔트로피와 MI를 계산한다.

다만 일반 MI에는 실전 정합에서 까다로운 함정이 하나 있다. 변환 $T$를 움직이다 보면 두 이미지가 겹치는 영역의 크기 자체가 달라지는데, 겹침이 줄어들면서 배경만 남는 바람에 오히려 MI가 올라가는 왜곡이 생길 수 있다. **Normalized Mutual Information(NMI)**은 주변 엔트로피나 결합 엔트로피로 값을 정규화해 겹치는 영역 크기 변화에 덜 휘둘리도록 만든 척도다(Studholme et al., 1999).

$$NMI(A, B) = \frac{H(A) + H(B)}{H(A, B)}$$

문헌에 따라 다음과 같은 형태로 정규화하기도 한다.

$$NMI(A, B) = \frac{2 \cdot I(A; B)}{H(A) + H(B)}$$

겹치는 영역이 계속 변하는 정합 문제에서는 순수 MI보다 NMI를 쓰는 편이 훨씬 안정적이다.

### 5.4.4 MI Gradient 계산

MI를 목적 함수로 삼아 경사 하강법 같은 연속 최적화를 돌리려면, 변환 파라미터 $\xi$가 변할 때 MI가 어느 방향으로 얼마나 바뀌는지 그래디언트 $\frac{\partial I}{\partial \xi}$가 필요하다. 변환 $T_\xi$에 대한 MI의 편미분은 다음 형태를 띤다.

$$\frac{\partial I}{\partial \xi} = \sum_{a, b} \frac{\partial p(a, b)}{\partial \xi} \left(1 + \log \frac{p(a, b)}{p(a) p(b)}\right)$$

문제는 픽셀 값을 정해진 구간(bin)에 딱 잘라 넣는 일반 히스토그램이 계단 함수여서 미분이 불가능하다는 점이다. 이를 우회하기 위해 Parzen 윈도우(커널 밀도 추정)나 B-spline 기저 함수로 히스토그램 경계를 부드럽게 뭉개 미분 가능한 결합 분포를 구성한다. 한편 미분 계산의 번거로움과 국소 최솟값을 피하려고 그래디언트 없이 푸는(gradient-free) Nelder-Mead simplex 최적화도 널리 쓰인다. 예를 들어 Koide et al.(2023)의 카메라-LiDAR 캘리브레이션 도구에서는 NID(§5.4.5)를 Nelder-Mead 방식으로 최소화한다.

### 5.4.5 왜 Calibration에서 MI가 쓰이는가

체커보드 같은 인공 타깃 없이 자연 환경에서 곧바로 외부 파라미터를 맞추는 targetless 카메라-LiDAR 캘리브레이션(→ 3장)에서는 주로 MI, NMI, 혹은 NID를 비용 함수로 사용한다. 최적화 루프는 세 단계를 되풀이한다.

1. 현재 추정 중인 외부 파라미터로 LiDAR 포인트 클라우드를 카메라 이미지 평면에 투영한다.
2. 같은 픽셀에 투영된 LiDAR 반사 강도(intensity)와 해당 픽셀의 카메라 영상 밝기 값 사이의 MI를 계산한다.
3. MI가 더 커지는(또는 NID가 작아지는) 방향으로 외부 파라미터를 조금씩 수정한다.

이때 최적화 목표로 자주 등장하는 **Normalized Information Distance(NID)**는 MI를 뒤집어 거리 척도로 다듬은 지표다.

$$NID(A, B) = 1 - \frac{I(A; B)}{H(A, B)} = \frac{H(A|B) + H(B|A)}{H(A, B)}$$

NID는 0(두 변수가 완전히 종속)에서 1(완전히 독립) 사이의 깔끔한 범위를 가지며, 삼각 부등식을 포함한 수학적 거리 함수(metric)의 조건을 만족한다.

실제 두 영상 배열에서 2D 히스토그램을 만들어 MI와 NID를 계산하는 기본 구현은 다음과 같다.

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

이상으로 고전적인 기하·통계 기반의 검출·매칭·정합 기법들을 살펴보았다. 이제 시야를 넓혀, 사람이 수작업으로 짠 그래디언트 규칙 대신 대규모 데이터에서 직접 불변 특징을 배우는 딥러닝 기반 방법들이 어떻게 고전 기법의 한계를 돌파했는지 살펴보자.

---

## 5.5 학습 기반 Feature Detection & Description

SIFT나 ORB 같은 전통적 특징점은 밝기 그래디언트, 코너, 블롭 같은 저수준 단서에 기대기 때문에 낮과 밤의 극단적인 조명 변화나 큰 시점 차이, 비·눈 같은 악천후 앞에서는 속수무책으로 무너지곤 한다. 딥러닝 기반 방법은 방대한 영상 데이터에서 더 넓고 의미 있는 문맥을 보며, 이러한 환경 변화에도 흔들리지 않는 특징점과 디스크립터를 스스로 학습한다.

### 5.5.1 SuperPoint (2018): 검출과 기술을 함께 자기지도 학습

딥러닝으로 특징점 검출기를 학습할 때 가장 먼저 부딪히는 난관은 정답 라벨이다. 영상에서 "좋은 키포인트가 어디인가"는 사물 인식처럼 사람이 일일이 점 찍어 정의하기가 애매하다. [DeTone et al. (2018)](https://arxiv.org/abs/1712.07629)의 **SuperPoint**는 단일 fully convolutional network에서 키포인트를 검출하는 동시에 디스크립터를 추출하되, **Homographic Adaptation**이라는 영리한 자기지도(self-supervised) 학습 전략으로 정답 라벨 문제를 풀었다.

#### Homographic Adaptation 학습 전략

핵심 발상은 "이미지를 어떤 시점으로 찌그러뜨려도 같은 물리적 코너는 계속 잡혀야 한다"는 기하학적 일관성에 있다. SuperPoint는 사람이 붙인 라벨 없이 다음 절차로 학습용 의사 정답(pseudo ground-truth)을 만든다.

1. 입력 이미지 한 장에 무작위 호모그래피(원근 변환)를 여러 개(논문에서는 100개) 적용해 다양한 시점의 변형 이미지를 만든다.
2. 변형한 이미지마다 기존 검출기로 키포인트를 뽑아낸다.
3. 검출된 좌표에 역호모그래피를 걸어 원래 이미지 좌표로 되돌려 한데 모은다(aggregate).
4. 여러 시점에 걸쳐 반복적으로 강하게 반응한 점들만 걸러내 신뢰할 수 있는 pseudo ground-truth로 삼는다.

#### 2단계 학습: MagicPoint → SuperPoint

이를 실제 이미지에 안착시키기 위해 2단계 학습 과정을 거친다.

- **1단계(기초 검출기 훈련)**: 삼각형, 사각형, 선분처럼 코너와 교차점의 정답 위치가 수학적으로 명확한 합성 도형 데이터셋(Synthetic Shapes)에서 코너·접합점 검출기 **MagicPoint**를 먼저 훈련한다.
- **2단계(실제 도메인 확장)**: MS-COCO 같은 실제 사진에 MagicPoint와 Homographic Adaptation을 결합해 키포인트 pseudo ground-truth를 대량 생성한 뒤, 이를 바탕으로 실제 영상에서 키포인트 검출과 디스크립터 추출을 함께 학습해 **SuperPoint**를 완성한다.

#### 아키텍처

SuperPoint는 가벼운 VGG 스타일 인코더 하나를 공유해 연산을 아낀 뒤, 두 갈래의 디코더 헤드로 결과를 내보낸다.

- **Interest Point Decoder**: 입력 이미지를 8×8 픽셀 크기의 셀로 쪼갠 뒤, 셀마다 65개 채널(셀 내부의 픽셀 위치 64개와 '키포인트 없음'을 뜻하는 dustbin 1개)에 걸쳐 softmax를 계산한다. 마지막으로 dustbin 채널을 버리고 64채널을 다시 8×8 공간으로 펼치면(reshape) 입력과 같은 해상도의 키포인트 히트맵이 완성된다.
- **Descriptor Decoder**: 공유 인코더의 특징 맵에서 1/8 해상도의 256차원 디스크립터 맵을 뽑아낸다. 검출 헤드가 고른 키포인트 좌표에 맞춰 bicubic interpolation으로 해당 위치의 256차원 벡터를 읽어 오고, 마지막에 L2 정규화로 크기를 맞춘다.

#### 학습 손실

두 헤드는 각자의 목표에 맞는 손실 함수로 동시에 학습된다. 키포인트 검출 헤드에는 cross-entropy loss를 건다. 두 이미지 쌍을 잇는 호모그래피로 정답 대응을 이미 알고 있으므로, 디스크립터 헤드에서는 이를 이용해 positive/negative 쌍에 hinge loss를 적용한다.

$$L_{desc} = \sum_{(i,j) \in \text{pos}} \max(0, m_p - \mathbf{d}_i^\top \mathbf{d}_j) + \sum_{(i,j) \in \text{neg}} \max(0, \mathbf{d}_i^\top \mathbf{d}_j - m_n)$$

여기서 $m_p, m_n$은 각각 positive 쌍이 가까워지도록, negative 쌍이 멀어지도록 강제하는 margin 값이다.

**논문 보고 속도**: 단 한 번의 forward pass로 검출과 기술을 동시에 끝내기 때문에 속도도 빠르다. 원 논문에서는 Titan X GPU 환경에서 640×480 영상을 약 70 FPS로 처리했다고 보고했다. 물론 실제 처리량은 사용 중인 GPU와 추출하는 키포인트 수에 따라 달라지므로 목표 하드웨어에서 직접 측정해 확인해야 한다.

실전에서는 `cvg/LightGlue` 패키지에서 제공하는 구현을 불러와 다음처럼 간단히 키포인트와 디스크립터를 뽑을 수 있다.

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

SuperPoint가 공유 인코더 뒤에 검출 헤드와 기술 헤드를 따로 두었다면, [D2-Net (Dusmanu et al., 2019)](https://arxiv.org/abs/1905.03561)은 아예 둘의 경계를 허물어 **단 하나의 CNN 특징 맵**이 디스크립터이자 검출기 역할을 동시에 맡도록 설계했다. 검출기가 먼저 점을 찍고 디스크립터가 그 주위를 요약하는 순차 구조 대신, 두 과정을 하나로 결합해 푼 셈이다.

구체적으로는 VGG16의 중간 특징 맵 $\mathbf{F} \in \mathbb{R}^{H \times W \times C}$를 꺼내 다음 두 가지 용도로 함께 쓴다.

- **Detection**: 특징 맵의 각 공간 위치마다 $C$개 채널 중 응답값이 가장 큰 채널을 고르고, 그 채널 안에서 주변 공간 이웃보다 값이 도드라지면(공간 NMS) 키포인트로 채택한다.
- **Description**: 그렇게 선택된 좌표의 $C$차원 채널 벡터를 별도 연산 없이 그대로 디스크립터로 사용한다.

깊은 CNN 층의 고수준 의미 특징을 직접 쓰기 때문에 낮과 밤처럼 외관이 극단적으로 뒤바뀌는 조건에서도 끈질기게 매칭을 이어 간다. 원 논문에서는 Aachen Day-Night와 InLoc localization 벤치마크에서 당시 최고 성능을 달성했다고 보고했다. 다만 특징 맵이 입력 영상의 1/4 해상도여서 키포인트 위치 정밀도가 격자 간격에 묶이는 태생적 약점이 있으며, 논문에서는 국소 정제(local refinement)로 서브픽셀 좌표를 보완해 이를 해결한다.

### 5.5.3 R2D2 (2019): Reliable and Repeatable Detector-Descriptor

D2-Net처럼 반응이 큰 점을 무조건 뽑다 보면 예상치 못한 함정에 빠진다. 예컨대 맑은 하늘이나 반복적인 창문 모서리, 나뭇잎처럼 어떤 시점에서도 눈에 잘 띄어 반복적으로 검출은 되지만, 막상 다른 점들과 구별할 특색이 없어 매칭에는 전혀 쓸모없는 지점이 섞이는 문제다. [R2D2 (Revaud et al., 2019)](https://arxiv.org/abs/1906.06195)는 좋은 특징점이 갖춰야 할 두 가지 조건을 명시적으로 분리해 학습하는 방식으로 이 문제를 해결했다.

- **Repeatability**: 카메라 시점이나 조명이 달라져도 같은 물리적 위치에서 꾸준히 다시 검출되는 정도
- **Reliability**: 그 지점에서 뽑은 디스크립터가 다른 수많은 점들 사이에서 헷갈리지 않고 짝을 구별해 낼 수 있는 정도

R2D2 네트워크는 반복성 맵과 신뢰성 맵을 각각 독립된 confidence map으로 예측한 뒤, 두 맵의 곱이 높은 지점만 골라 최종 키포인트로 삼는다. 덕분에 하늘이나 단조로운 벽면처럼 매칭 변별력이 없는 구역은 자연스럽게 걸러진다.

### 5.5.4 DISK (2020)

지금까지의 학습 기반 방법들은 "검출기가 잘 뽑고 디스크립터가 잘 구분하도록" 각자 대리 손실 함수를 최적화했지만, 우리가 궁극적으로 원하는 목표는 단 하나, "최종 단계에서 올바른 매칭 쌍을 최대한 많이 건져 내는 것"이다. [DISK (Tyszkiewicz et al., 2020)](https://arxiv.org/abs/2006.13566)는 특징점 샘플링과 매칭 과정을 확률적 정책으로 모델링하고, **강화학습(reinforcement learning)의 policy gradient**로 검출과 기술을 end-to-end로 묶어 학습한다. 두 이미지에서 뽑은 특징점을 매칭해 기하학적 정답과 맞으면 보상(reward)을, 틀리면 벌점(penalty)을 줌으로써 최종 매칭 성능 자체를 직접 끌어올렸다.

### 5.5.5 학습 기반 특징점이 나은 경우

딥러닝 기반 특징점이 고전 기법 대비 강점을 보이는 대표 사례를 요약하면 다음과 같다. 아래 표의 근거는 D2-Net 논문의 Aachen Day-Night 실험 결과(§5.5.2)다. 해당 논문에서는 시점 변화, 반복 패턴, 모션 블러 조건에서 두 진영 중 어느 쪽이 더 나은지 직접 비교하는 실험을 하지 않았다.

| 시나리오 | SIFT/ORB 한계 | SuperPoint/D2-Net 개선 |
|---------|-------------|---------------------|
| 극한 조명 변화 (주야간) | DoG/그래디언트 기반 검출 실패 | 학습된 특징이 고수준 구조를 잡음 |

다만 학습 기반 방법은 훈련 데이터 분포에 강하게 의존한다는 점을 염두에 두어야 한다. 도심 도로와 실내 사진 위주로 학습된 모델은 수중 영상이나 적외선, 극단적 항공 뷰처럼 학습 분포에서 크게 벗어난 환경에 투입되면 성능이 급락할 수 있다.

여기까지 특징점 검출과 디스크립터 추출을 신경망에 맡기는 흐름을 살펴보았다. 그렇다면 자연스럽게 다음 질문이 떠오른다. "특징점과 디스크립터를 딥러닝으로 뽑았다면, 그 다음 매칭 단계도 단순한 최근접 이웃 탐색과 ratio test 대신 신경망으로 해결할 수 없을까?"

---

## 5.6 학습 기반 Feature Matching

전통적인 Brute-Force나 FLANN 매칭은 각 키포인트의 디스크립터를 홀로 떼어 놓고 거리만 잰다. 다른 키포인트들이 화면 어디에 어떻게 배치되어 있는지 주변 맥락은 전혀 돌아보지 않는다. 5.6절에서는 이미지 전체의 공간적·시각적 문맥을 읽고 짝을 짓는 학습 기반 매처(matcher)의 등장을 살펴본다.

### 5.6.1 SuperGlue (2020): Attention 기반 매칭

[Sarlin et al. (2020)](https://arxiv.org/abs/1911.11763)의 **SuperGlue**는 두 이미지의 키포인트 집합을 그래프 노드로 보는 학습 기반 matcher다. 그래프 신경망(GNN), 어텐션(attention), optimal transport를 결합해 키포인트를 짝짓고 짝이 없는 점을 걸러낸다. 사람이 짠 ratio test와 mutual check를 하나의 미분 가능한 신경망 블록으로 대체한 기념비적 연구다.

#### 문제 정의: 부분 할당(Partial Assignment)

두 이미지에서 뽑은 키포인트 집합을 각각 $\mathcal{A} = \{(\mathbf{p}_i, \mathbf{d}_i)\}_{i=1}^{N}$과 $\mathcal{B} = \{(\mathbf{p}_j, \mathbf{d}_j)\}_{j=1}^{M}$이라 하자. 현실에서는 시야 밖으로 벗어나거나 물체에 가려져 상대 이미지에 대응점이 아예 없는 키포인트가 수두룩하다. SuperGlue는 모든 점을 억지로 연결하지 않고, 짝 없는 키포인트를 '매칭 없음'을 뜻하는 가상의 **dustbin** 노드로 보내는 **부분 할당(partial assignment)** 문제로 매칭을 정식화한다.

#### Keypoint Encoder

매칭에서는 생김새(디스크립터 $\mathbf{d}_i$)만큼이나 위치 정보(좌표 $\mathbf{p}_i = (x, y)$와 검출 confidence $c_i$)가 중요하다. 가령 똑같이 생긴 창문 모서리라도 위층과 아래층이라는 상대 위치 맥락이 주어지면 구별할 수 있다. SuperGlue는 위치와 신뢰도를 MLP로 임베딩한 뒤 디스크립터 벡터에 더해 특징 표현에 공간 정보를 녹여 넣는다.

$$\mathbf{f}_i^{(0)} = \mathbf{d}_i + \text{MLP}_{\text{enc}}([\mathbf{p}_i, c_i])$$

#### Attentional Graph Neural Network

이제 두 이미지의 키포인트를 모두 노드로 삼는 그래프 위에서 **Self-Attention**과 **Cross-Attention**을 번갈아 적용하며 특징을 계속 다듬어 간다.

- **Self-Attention (intra-image)**: 같은 이미지 안의 키포인트끼리 메시지를 주고받는다. "내 옆에 어떤 구조물이 있고 내가 전체 화면에서 어떤 기하적 배치에 놓여 있는가"라는 내부 문맥을 파악하는 단계다.

$$\text{message}_i^{\text{self}} = \sum_{j \in \mathcal{A}} \text{softmax}\left(\frac{\mathbf{q}_i^\top \mathbf{k}_j}{\sqrt{d}}\right) \mathbf{v}_j$$

- **Cross-Attention (inter-image)**: 상대 이미지의 모든 키포인트로부터 정보를 끌어와, 저쪽 이미지의 어느 후보들이 나와 생김새·주변 문맥에서 맞아떨어지는지 견주어 본다.

$$\text{message}_i^{\text{cross}} = \sum_{j \in \mathcal{B}} \text{softmax}\left(\frac{\mathbf{q}_i^\top \mathbf{k}_j}{\sqrt{d}}\right) \mathbf{v}_j$$

이 Self-Attention과 Cross-Attention 한 쌍을 $L$번(원 논문은 9번) 되풀이하는 동안 후보들의 모호성이 점차 해소되고, 특징 벡터가 확실한 짝을 향해 수렴한다.

#### Optimal Transport를 이용한 매칭

GNN 층을 모두 통과한 최종 특징끼리 내적해 유사도 점수 행렬을 만든 뒤, 양쪽 이미지가 일대일(혹은 dustbin)로 매칭되도록 **Sinkhorn 알고리즘**을 적용한다. Sinkhorn은 조합 최적화의 고전인 Hungarian algorithm(선형 할당 문제)을 미분 가능한 형태로 근사해 역전파로 학습할 수 있게 한다.

구체적으로 dustbin 행과 열까지 덧붙인 점수 행렬 $\mathbf{S} \in \mathbb{R}^{(N+1) \times (M+1)}$을 다음처럼 구성하고($z$는 학습 가능한 dustbin 점수),

$$S_{ij} = \langle \mathbf{f}_i^{(L)}, \mathbf{f}_j^{(L)} \rangle, \quad S_{i,M+1} = S_{N+1,j} = z$$

행 정규화와 열 정규화를 번갈아(원 논문은 100번) 반복해 행 합과 열 합이 1에 수렴하는 이중 확률 행렬(doubly stochastic matrix)을 얻는다.

$$\mathbf{S} \leftarrow \text{row-normalize}(\mathbf{S}), \quad \mathbf{S} \leftarrow \text{col-normalize}(\mathbf{S})$$

이렇게 구한 soft assignment 행렬에서 신뢰도 임계값을 넘고 행·열 양쪽에서 서로 최댓값을 가리키는 쌍만 최종 매칭으로 뽑아낸다.

#### 학습

학습 단계에서는 호모그래피나 알려진 카메라 상대 포즈·깊이 맵으로 만든 정답 대응 쌍 $\mathcal{M}$과 짝 없는 점 집합 $\mathcal{U}_A, \mathcal{U}_B$에 대해 negative log-likelihood를 최소화하며 end-to-end로 가중치를 업데이트한다.

$$L = -\sum_{(i,j) \in \mathcal{M}} \log \hat{P}_{ij} - \sum_{i \in \mathcal{U}_A} \log \hat{P}_{i, M+1} - \sum_{j \in \mathcal{U}_B} \log \hat{P}_{N+1, j}$$

실제 파이프라인에서는 3D 복원 도구인 `hloc`을 써서 SuperPoint 특징 추출과 SuperGlue 매칭을 다음처럼 엮어 돌릴 수 있다.

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

SuperPoint에 검출·기술을, SuperGlue에 매칭을 맡기면 파이프라인의 세 단계가 모두 학습 기반으로 바뀐다. 하지만 "검출 → 기술 → 매칭"을 차례로 밟는 뼈대 자체는 고전 파이프라인과 여전히 같다. 이 검출 단계마저 아예 들어내는 다음 세대 흐름은 §5.7의 LoFTR에서 본격적으로 이어진다.

### 5.6.2 LightGlue (2023): SuperGlue의 효율화

SuperGlue는 압도적인 매칭 성능을 보여 주었지만 계산 비용이 커서 실시간 VO/SLAM에 그대로 집어넣기에는 버거웠다. [Lindenberger et al. (2023)](https://arxiv.org/abs/2306.13643)은 SuperGlue의 구조를 밑바닥부터 재검토해 **LightGlue**를 선보였다. LightGlue는 입력 영상 쌍의 난이도에 따라 연산량을 유동적으로 줄이는 **adaptive computation** 매처다.

#### SuperGlue의 문제점 진단

LightGlue 저자들이 짚어 낸 SuperGlue의 비효율은 크게 두 가지였다. 첫째, 공개된 SuperGlue는 GNN 층 수(9층)와 Sinkhorn 반복 수(100회)가 딱 고정되어 있어, 눈으로 봐도 뻔한 쉬운 영상 쌍이 들어와도 가장 어려운 쌍과 똑같은 연산을 끝까지 돌린다. 둘째, 층마다 키포인트 수 $N$에 대해 복잡도가 $O(N^2)$인 attention을 계산하므로, 키포인트 수가 늘어날수록 추론 속도가 급격히 무너진다.

#### 핵심 개선: Adaptive Depth & Width

LightGlue는 계산량을 "깊이(depth)"와 "너비(width)" 양쪽에서 동시에 깎아 냈다.

- **Adaptive Depth (레이어 조기 종료)**: 각 층의 출력마다 가벼운 MLP 분류기를 달아 "각 점의 매칭 예측이 이미 충분히 확신에 찼는가"를 매 층 점검한다. 예측이 확정된 점의 비율이 기준선을 넘으면 나머지 층을 계산하지 않고 즉시 추론을 멈춘다. 실제로 논문 그림 2의 예시에서는 비슷한 시점의 쉬운 이미지 쌍이 3층 만에 종료된 반면, 난이도가 높은 쌍은 8층까지 계산을 이어 갔다(전체 9층).
- **Adaptive Width (키포인트 pruning)**: 중간 층을 지나는 도중에 "이 점은 상대 이미지에 짝이 없을 확률이 압도적이다"라고 판명된 키포인트는 과감히 연산 대상에서 솎아 낸다. 남은 점의 수가 줄어드니 뒤쪽 층으로 갈수록 attention 행렬 크기가 작아져 계산이 몰라보게 가벼워진다.
- **Sinkhorn 제거**: 행·열 정규화를 100번이나 반복해 돌리던 Sinkhorn Optimal Transport를 버리고, 각 점의 매칭 가능성 점수(matchability)와 행·열 방향 **dual-softmax**의 곱으로 할당 행렬을 단번에 계산해 병목을 없앴다.

$$P_{ij} = \text{softmax}_j(S_{ij}) \cdot \text{softmax}_i(S_{ij})$$

(논문의 식에서는 여기에 두 점 각각의 매칭 가능성 점수 $\sigma_i \sigma_j$를 곱해 짝 없는 점을 억누른다.)

#### 학습 전략

훈련할 때는 조기 종료 없이 9개 층을 모두 끝까지 통과시키되, 매 층의 중간 출력마다 정답 손실을 함께 거는 deep supervision으로 각 층이 독립적으로도 완성도 높은 예측을 내놓도록 길들인다. 조기 종료와 pruning은 추론 단계에서만 활성화한다. 또한 LightGlue는 SuperPoint 전용이 아니라 DISK, ALIKED 같은 다양한 로컬 특징과도 짝을 맞춰 쓸 수 있다.

**성능**: LightGlue 논문에서는 SuperGlue와 대등한 정확도를 유지하면서도 기본 설정에서 처리량이 2.5배, FlashAttention 등 효율적인 self-attention 구현을 결합한 변형에서는 4배까지 빨라졌다고 보고했다. 특히 시점 변화가 적은 쉬운 영상 쌍일수록 조기 종료가 일찍 걸려 체감 속도가 훨씬 빨라지며, 실제 속도 배율은 키포인트 수와 GPU 사양에 따라 달라진다.

Kornia에서는 `features='superpoint'` 옵션으로 SuperPoint 출력과 바로 연결해 쓸 수 있다.

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

여기까지 짚어 온 학습 기반 특징점과 매처의 진화 과정을 한 줄로 엮으면 다음과 같다.

```
[전통적 파이프라인]
SIFT detect → SIFT descriptor → BF/FLANN + ratio test
                                        ↓
[학습 기반 대체]
SuperPoint detect+describe → SuperGlue attention matching → LightGlue 효율화
    (2018)                       (2020)                        (2023)
```

이처럼 검출·기술·매칭을 분리해 두면 각 모듈의 성능을 따로 뜯어보고 더 나은 모델로 손쉽게 갈아 끼울 수 있다는 장점이 있다. 그러나 근본적인 구조적 약점 하나는 여전히 남아 있다. 첫 단계의 **검출기가 키포인트를 하나도 잡지 못하면**, 뒤따르는 디스크립터와 매처가 아무리 뛰어나도 아예 손쓸 여지가 없다. 바로 이 한계를 넘기 위해 §5.7의 detector-free 방법들이 검출기 자체를 파이프라인에서 지워 버린다.

---

## 5.7 Detector-Free Matching

### 5.7.1 왜 Detector-Free가 필요한가

앞선 Detect-then-match 파이프라인은 아무리 매처가 똑똑해도 검출기가 처음에 찍어 준 키포인트 목록 안에서만 짝을 찾을 수 있다. 이 구조적 제약 때문에 세 가지 까다로운 환경에서는 금세 한계가 드러난다.

- **텍스처 없는 영역**: 흰 벽면, 매끈한 콘크리트 바닥, 탁 트인 하늘처럼 밝기 그래디언트가 거의 없는 곳에서는 코너나 블롭 자체가 아예 검출되지 않는다.
- **반복 패턴**: 똑같은 타일이나 고층 건물의 창문 격자는 키포인트가 잔뜩 잡히지만 국소 패치가 판박이라 디스크립터만으로는 구별이 모호하다.
- **넓은 시점 변화**: 카메라 각도가 크게 틀어지면 같은 물체라도 국소 패치의 겉모습이 딴판으로 변해, 양쪽 이미지에서 동일한 지점이 동시에 검출될 확률 자체가 뚝 떨어진다.

Detector-free 방법은 "미리 점 몇 개를 골라 놓고 매칭한다"는 발상 자체를 뒤집는다. 검출 단계를 통째로 걷어 내고 이미지의 모든 격자 위치를 잠재적 매칭 후보로 올려놓은 뒤, 트랜스포머의 전역 문맥 파악 능력을 빌려 픽셀 단위의 대응을 직접 추론한다.

### 5.7.2 LoFTR (2021): Coarse-to-Fine Transformer Matching

[Sun et al. (2021)](https://arxiv.org/abs/2104.00680)의 **LoFTR**는 검출기 없이도 두 이미지 사이의 촘촘한 대응점을 뽑아내는 대표적인 트랜스포머 기반 방법이다. 모든 픽셀 쌍을 처음부터 끝까지 비교하면 계산량이 감당할 수 없이 불어나므로, 성긴 격자에서 대략적인 짝을 먼저 맞추고(coarse) 그 주변에서 정밀 좌표를 가다듬는(fine) **coarse-to-fine** 전략을 쓴다.

#### 아키텍처

1. **Local Feature CNN**: 먼저 ResNet-18 기반의 FPN(Feature Pyramid Network)으로 두 이미지 각각에서 성긴 특징 맵(원본의 1/8 해상도)과 세밀한 특징 맵(1/2 해상도)을 뽑는다.
2. **Coarse-Level Matching (Transformer)**: 1/8 해상도의 특징 맵을 1차원 토큰 시퀀스로 펼친 뒤(flatten), 2D 사인·코사인 위치 인코딩(sinusoidal positional encoding)을 더해 각 토큰이 화면 어느 위치에서 왔는지 기억하게 한다. 그다음 Self-Attention과 Cross-Attention을 번갈아 $N$번(원 논문은 4번) 통과시켜 이미지 내·외부의 전역 문맥을 풍부하게 반영한다. 변환된 특징 벡터끼리 내적해 유사도 행렬을 구한 다음, 양방향 **dual-softmax**를 적용해 신뢰도 행렬을 만든다.

$$P_{ij} = \text{softmax}_j(\mathbf{f}_i^A \cdot \mathbf{f}_j^B) \cdot \text{softmax}_i(\mathbf{f}_i^A \cdot \mathbf{f}_j^B)$$

여기서 신뢰도가 임계값을 넘으면서 동시에 상호 최근접 이웃(mutual nearest neighbor) 조건을 만족하는 쌍을 coarse 매칭으로 뽑아낸다.
3. **Fine-Level Refinement**: 1/8 해상도에서 찾은 대응점은 아직 최대 8픽셀가량의 거친 오차를 안고 있다. LoFTR는 각 coarse 매칭마다 1/2 해상도 특징 맵에서 대응 위치 주변의 작은 $w \times w$ 윈도우(논문은 5×5)를 잘라 낸 뒤, 윈도우 안에서 국소 상관을 다시 계산해 기대 위치로 서브픽셀 좌표를 회귀(regression)한다.

#### Linear Transformer

1/8 해상도라 해도 영상이 조금만 커지면 토큰 수 $N$이 수천 개에 달해 표준 softmax attention의 $O(N^2)$ 복잡도가 발목을 잡는다. LoFTR는 이를 우회하기 위해 커널 기반 **linear attention**을 도입했다.

- **Standard attention** ($O(N^2)$): $\text{Attn}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{softmax}(\mathbf{Q}\mathbf{K}^\top / \sqrt{d}) \mathbf{V}$
- **Linear attention** ($O(N)$): $\text{Attn}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \phi(\mathbf{Q})(\phi(\mathbf{K})^\top \mathbf{V})$

커널 함수 $\phi(\cdot) = \text{elu}(\cdot)+1$로 비선형성을 치환한 뒤 행렬 곱의 결합 순서만 바꾸면, $N \times N$ 어텐션 행렬을 만들지 않고 $D \times D$ 행렬부터 곱해 복잡도가 $O(N)$으로 뚝 떨어진다. 다만 softmax attention을 선형 커널로 근사하는 구조여서 날카로운 집중력(표현력)은 다소 무뎌질 수 있다는 트레이드오프가 따른다.

#### 학습

정답 카메라 포즈와 깊이 맵으로 짝지은 픽셀 대응을 정답지로 삼아 coarse 단계에는 cross-entropy loss를, fine 단계에는 L2 regression loss를 걸어 학습한다. 실내 장면은 ScanNet, 실외 장면은 MegaDepth 데이터셋을 주로 사용한다.

Kornia에서는 사전 학습된 LoFTR 모델을 불러와 grayscale 텐서 쌍만으로 바로 대응점을 추출할 수 있다.

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

LoFTR는 수십 년간 당연시되던 detect-then-match 파이프라인에서 **검출기를 완전히 제거한 분기점**이다. coarse 단계에서 한 이미지의 모든 위치가 cross-attention으로 상대 이미지의 모든 위치를 두루 훑으므로, LoFTR는 코너가 전혀 없는 흰 벽이나 반복 패턴 위에서도 전역 문맥을 단서로 삼아 대응점을 이끌어 낸다.

### 5.7.3 QuadTree Attention: LoFTR의 효율화

LoFTR의 coarse 트랜스포머는 이미지의 모든 위치를 평평한 시퀀스로 펼쳐 계산하기 때문에, 더 높은 해상도에서 촘촘히 매칭하려고 하면 토큰 길이가 걷잡을 수 없이 불어난다. **QuadTree Attention**(Tang et al., ICLR 2022)은 어텐션을 한꺼번에 계산하지 않고 피라미드처럼 계층적으로 좁혀 들어간다.

1. 가장 거친(coarsest) 해상도에서 먼저 전체 영역의 어텐션을 계산한다.
2. 그중 어텐션 점수가 높아 유망해 보이는 영역 몇 곳만 추려 낸다.
3. 한 단계 더 세밀한 해상도로 내려가, 고른 영역 내부에서만 어텐션을 계산한다.
4. 이 과정을 가장 세밀한 해상도까지 반복해 가지를 친다.

이 계층적 가지치기 덕분에 표준 어텐션의 $O(N^2)$ 비용이 $O(N)$ 수준으로 내려간다. 원 논문은 이미 $O(N)$인 LoFTR의 linear attention과 견주더라도, 비슷한 연산량에서 근사 없는 softmax attention을 유지해 정확도가 더 높다는 점을 강점으로 내세운다.

### 5.7.4 ASpanFormer (2022): Adaptive Span Attention

한편 **ASpanFormer**(Chen et al., ECCV 2022)는 화면의 모든 위치가 똑같은 범위의 주변 정보를 볼 필요는 없다는 점에 주목했다. 텍스처가 풍부하고 고유한 곳은 굳이 멀리까지 훑을 필요 없이 좁은 범위만 정밀하게 들여다보면 충분하지만, 특징이 밋밋하고 텍스처가 부족한 영역은 주변의 넓은 문맥을 끌어와야 비로소 위치를 가늠할 수 있다. ASpanFormer는 위치마다 추정한 초기 flow와 그 불확실성을 바탕으로 attention 범위(span)를 위치별로 다르게 조절해 정확도와 효율을 동시에 잡았다.

### 5.7.5 RoMa (2024): DINOv2 + Dense Matching

LoFTR 계열은 특징 추출 CNN부터 매칭 트랜스포머까지 모두 매칭 데이터셋만으로 처음부터(scratch) 학습했다. 반면 [Edstedt et al. (2024)](https://arxiv.org/abs/2305.15404)의 **RoMa**는 대규모 영상으로 사전 학습된 비전 파운데이션 모델 **DINOv2**를 도입해 dense matching의 수준을 한 단계 끌어올렸다.

#### Foundation Model 활용

RoMa는 사전 학습된 DINOv2 ViT-Large의 가중치를 동결(frozen)한 채 coarse 단계의 특징 추출기로 활용한다. 대규모 자기지도 학습을 거친 DINOv2의 범용 시각 표현은, 제한된 매칭 데이터셋에서 처음부터 학습한 로컬 CNN 특징보다 시점·조명·외관 변화에 훨씬 강건하다. 다만 DINOv2 특징 맵은 해상도가 거칠다는 한계가 있으므로, RoMa는 큰 그림을 잡는 몫은 DINOv2에 맡기고 세밀한 픽셀 위치를 다듬는 몫은 별도의 CNN 특징에 넘기는 영리한 분업 구조를 택했다.

#### 아키텍처

1. **Frozen DINOv2 Backbone**: 가중치를 고정한 DINOv2 ViT-Large에서 입력의 1/14 해상도로 패치 특징을 뽑아낸다.
2. **Coarse Matching (Warp Estimation)**: 추출한 DINOv2 특징 위에 transformer match decoder를 올려, 이미지 A의 각 위치에 대응하는 이미지 B의 위치를 단 하나의 좌표 대신 앵커(anchor) 위치들에 대한 확률 분포(혼합 가우시안 형태)로 예측한다.

$$p(\mathbf{x}_B | \mathbf{x}_A) = \sum_{k} w_k \mathcal{N}(\mathbf{x}_B; \mu_k, \Sigma_k)$$

단일 점만 회귀하지 않고 확률 분포 전체를 예측하므로, 반복 패턴처럼 헷갈리는 대응 후보가 여럿 있는 경우(multimodality)에도 모호성을 뭉개지 않고 온전히 표현할 수 있다.
3. **Fine Matching (Iterative Refinement)**: 이렇게 구한 거친 대응장(coarse warp)에서 출발해, VGG19 CNN으로 뽑은 고해상도 fine 특징을 바탕으로 대응 위치를 여러 단계에 걸쳐 정밀하게 다듬는다. 해상도를 한 단계씩 높일 때마다 직전 warp 위치 주변의 국소 상관을 계산해 잔차를 보정하는 식이다.
4. **Certainty Estimation**: 각 픽셀의 매칭 신뢰도(certainty)도 함께 내놓기 때문에, 뒤따르는 포즈 추정 단계에서는 고신뢰 픽셀만 선별해 쓸 수 있다.

#### 손실 함수: regression-by-classification과 robust regression

RoMa는 좌표를 단순 L2 loss로 곧바로 회귀하지 않는다. coarse 단계에서는 정답 위치가 속한 앵커 클래스를 맞히는 분류 손실(regression-by-classification)을 쓰고, fine 단계에서는 간혹 섞이는 큰 오차에 모델이 흔들리지 않도록 robust regression 손실을 적용한다. coarse 단계의 분류 손실은 다음 형태를 띤다.

$$L = -\sum_{(\mathbf{x}_A, \mathbf{x}_B^*)} \log p(\mathbf{x}_B^* | \mathbf{x}_A)$$

#### 성능

RoMa 논문은 MegaDepth와 ScanNet의 상대 포즈 추정(relative pose) 실험에서 함께 비교한 LoFTR·ASpanFormer 계열을 웃도는 AUC를 기록했다. 특히 시점과 외관 변화가 극심한 WxBS 벤치마크에서는 기존 대비 36%에 달하는 성능 향상을 보고했다.

실제 사용 시에는 `romatch` 패키지로 전 픽셀에 걸친 `warp`와 `certainty` 맵을 얻은 뒤, 신뢰도가 높은 위치 위주로 원하는 개수만큼 매칭 쌍을 샘플링한다.

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

최근에는 "2D 이미지 위에서 먼저 짝을 찾은 뒤 나중에 삼각측량이나 에피폴라 기하로 3D를 복원한다"는 오랜 순서를 아예 뒤집는 연구들이 쏟아지고 있다. 이미지에서 곧바로 3D 기하 구조를 예측하고, 그 결과에서 대응점을 읽어 내는 방식이다.

- **DUSt3R (Wang et al., 2024)**: [DUSt3R](https://arxiv.org/abs/2312.14132)은 카메라 내부 파라미터나 포즈 정보 없이 이미지 두 장만 입력받아 동일한 좌표계 위의 밀집 3D pointmap을 곧장 회귀한다. 2D 매칭을 거쳐 3D를 계산하는 대신 3D 형상을 먼저 만들어 낸 다음, 3D 공간에서 가장 가까운 점 쌍을 대응점으로 읽어 내는 역발상을 보여 주었다.
- **MASt3R (Leroy et al., 2024)**: [MASt3R](https://arxiv.org/abs/2406.09756)은 DUSt3R의 3D pointmap 회귀에 촘촘한 로컬 디스크립터 헤드를 덧붙여 매칭 정밀도를 한층 끌어올렸다. 저자들은 지도 없는 위치 추정(Map-free localization) 벤치마크에서 VCRE AUC가 당시 기존 최고 방법보다 30%(절댓값) 높았다고 보고했다.
- **VGGT (Wang et al., 2025)**: [VGGT](https://arxiv.org/abs/2503.11651)(Visual Geometry Grounded Transformer)는 CVPR 2025 Best Paper를 수상한 모델로, 이미지 한 장부터 수백 장의 시퀀스까지 한꺼번에 입력받아 카메라 파라미터, pointmap, depth map, 3D point track을 단 한 번의 feed-forward pass로 동시에 추론한다. 원 논문은 VGGT가 전체 재구성을 1초 안에 끝내면서도 별도의 후처리 최적화를 거치는 기존 기법들보다 여러 3D 과제에서 더 높은 성능을 냈다고 보고했다.

지금까지는 카메라가 찍은 2차원 영상 위에서 대응점을 찾는 방법들을 훑어보았다. 이제 무대를 3차원 공간으로 옮겨, LiDAR나 RGB-D 센서가 내놓는 두 개의 3D 포인트 클라우드 사이에서 직접 3D-3D 대응점을 찾아 정합하는 기법들을 살펴보자.

---

## 5.8 3D-3D Correspondence

3D 포인트 클라우드 사이의 대응점은 LiDAR scan-to-scan 정합, 여러 시점의 지도를 합치는 multi-session map merging, 그리고 3D loop closure의 첫 단추로 쓰인다. 하지만 2D 이미지와 달리 규칙적인 픽셀 격자나 밝기 그래디언트가 없기 때문에, 불규칙하게 흩어진 점들의 3차원 기하 형태만으로 국소 구조를 요약하고 짝을 찾아야 한다.

### 5.8.1 FPFH (Fast Point Feature Histograms)

[FPFH (Rusu et al., 2009)](https://ieeexplore.ieee.org/document/5152473)는 딥러닝 이전 시대를 대표하는 수작업 3D 기하 디스크립터로, 점 주변 표면의 법선 벡터들이 서로 어떤 각도를 이루는지 통계 히스토그램으로 담아낸다.

#### SPFH (Simple Point Feature Histogram)

강체 변환(회전과 이동)에 흔들리지 않는 특징을 만들려면, 절대 좌표 대신 두 점 사이의 **상대적 기하 관계**만 써야 한다. 쿼리 점 $\mathbf{p}$와 이웃 점 $\mathbf{p}_k$마다 두 점의 법선 벡터 $\mathbf{n}_p, \mathbf{n}_k$와 연결 벡터 $\mathbf{d} = \mathbf{p}_k - \mathbf{p}$를 이용해 다음과 같이 강체 불변 로컬 좌표계(Darboux frame)를 세운다.

$$\mathbf{u} = \mathbf{n}_p$$
$$\mathbf{v} = \mathbf{u} \times \frac{\mathbf{d}}{\|\mathbf{d}\|}$$
$$\mathbf{w} = \mathbf{u} \times \mathbf{v}$$

이 좌표축 위에서 두 법선이 이루는 상대적인 각도 특성 세 가지를 뽑아낸다.

$$\alpha = \mathbf{v} \cdot \mathbf{n}_k, \quad \phi = \mathbf{u} \cdot \frac{\mathbf{d}}{\|\mathbf{d}\|}, \quad \theta = \text{atan2}(\mathbf{w} \cdot \mathbf{n}_k, \mathbf{u} \cdot \mathbf{n}_k)$$

세 각도 특성은 점군이 통째로 회전하거나 이동해도 값이 변하지 않으며, 이를 각각 $B$-bin 히스토그램으로 양자화해 **SPFH**를 만든다.

#### FPFH: PFH를 빠르게 근사

원래의 PFH는 반경 $r$ 안에 들어온 모든 이웃 점 쌍 $(i, j)$마다 각도를 재느라 점 하나에 $O(k^2)$ 연산이 들었다. **FPFH**는 계산량을 줄이기 위해 쿼리 점 $\mathbf{p}$와 직접 연결된 이웃들 사이에서만 SPFH를 구한 뒤($O(k)$), 이웃 점마다 미리 계산해 둔 SPFH를 거리 역수 가중치($w_i = \|\mathbf{p}_i - \mathbf{p}\|$)로 섞어 넣어 주변 $2r$ 반경의 기하 문맥까지 간접 반영한다.

$$\text{FPFH}(\mathbf{p}) = \text{SPFH}(\mathbf{p}) + \frac{1}{k} \sum_{i=1}^{k} \frac{1}{w_i} \text{SPFH}(\mathbf{p}_i)$$

구현에서는 각도 특성 하나당 11-bin씩 묶어 최종 33차원 히스토그램을 만든다. Open3D에서는 법선을 먼저 추정한 뒤 `compute_fpfh_feature`를 호출해 바로 계산할 수 있다.

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

법선 추정에 민감하고 노이즈에 흔들리는 수작업 3D 히스토그램을 넘어, 3차원 영역에서도 딥러닝으로 로컬 특징을 배우려는 시도가 이어졌다. [Zeng et al. (2017)](https://arxiv.org/abs/1603.08182)의 **3DMatch**는 기존 RGB-D 3D 재구성 데이터셋에서 자동으로 추출한 정답 대응 쌍을 활용해 3D CNN 기반 디스크립터를 학습했다.

- **학습 데이터**: 실내 RGB-D 재구성 62개 장면에서 자동으로 생성한 대응점 쌍
- **구조**: 각 점 주변 공간을 3D TDF(Truncated Distance Function) 볼륨 격자로 복셀화해 입력받는 3D CNN
- **출력**: 512차원 로컬 디스크립터

논문과 함께 공개된 **3DMatch Benchmark**는 이후 FCGF, Predator, GeoTransformer 등 3D 정합 연구들이 성능을 겨루는 표준 무대로 자리 잡았다.

### 5.8.3 FCGF (Fully Convolutional Geometric Features, 2019)

3DMatch는 성능의 물꼬를 텄지만, 키포인트마다 주변 3D 패치 볼륨을 하나씩 잘라 CNN을 따로 돌려야 해서 속도가 몹시 느렸다. [Choy et al. (2019)](https://doi.org/10.1109/ICCV.2019.00905)의 **FCGF**는 3D 공간에 희소 합성곱(sparse convolution)을 도입해, 잘게 쪼개지 않고 **포인트 클라우드 전체에서 단 한 번의 forward pass로 모든 점의 32차원 디스크립터를 한꺼번에 뽑아냈다.** FCGF 논문에서는 16배나 작은 32차원 디스크립터만으로도 3DMatch보다 더 높은 feature-match recall을 달성했고 특징 추출 속도는 약 169배 빨랐다고 보고했다(단, 비교 대상인 3DMatch의 실행 시간은 FCGF와 다른 CPU 환경에서 측정됐다는 점을 감안해야 한다).

### 5.8.4 Predator (2021): Overlap-Aware 3D Matching

실제 로봇이나 자율주행차가 마주하는 두 3D 스캔은 전체가 아니라 일부 구간만 서로 겹친다. 그런데 FCGF까지의 방법들은 두 스캔이 겹치는지 아닌지 살피지 않고 모든 곳에서 똑같이 디스크립터를 뽑았기 때문에, 겹침 비율이 10~30% 수준으로 떨어지면 매칭 성공률이 급락했다. [Huang et al. (2021)](https://arxiv.org/abs/2011.13005)의 **Predator**는 2D의 SuperGlue처럼 cross-attention을 3D에 끌어들여, 정합 전에 두 스캔이 겹치는 영역부터 미리 가려내는 전략을 썼다.

1. **Overlap attention**: 두 포인트 클라우드 사이에 cross-attention을 걸어, 각 점이 상대 스캔과 겹치는 영역에 속할 확률을 예측한다.
2. **Matchability score**: 겹치는 영역 안에서도 평평한 바닥처럼 구분이 안 되는 점 대신, 모서리나 기둥처럼 매칭에 실제로 쓸 만한 독특한 점에 높은 점수를 매긴다.

이 두 가지 점수를 바탕으로 후보를 추린 결과, 원 논문에서는 겹침이 극히 적은(10~30%) 3DLoMatch 벤치마크에서 기존 기법 대비 성능이 크게 올랐다고 보고했다.

### 5.8.5 GeoTransformer (2022): 기하학적 트랜스포머

[Qin et al. (2022)](https://arxiv.org/abs/2202.06688)의 **GeoTransformer**는 한 걸음 더 나아가, 강체 변환에 완전히 불변인 기하 구조를 트랜스포머 내부에 직접 심고 계산 비용이 큰 RANSAC 루프까지 없애 버렸다.

#### 키포인트-프리 슈퍼포인트 매칭

별도의 3D 키포인트 검출기를 돌리지 않고, 포인트 클라우드를 성기게 다운샘플링해 얻은 **슈퍼포인트(superpoint)** 단위로 먼저 굵직한 대응을 찾은 뒤, 각 슈퍼포인트가 품은 촘촘한 점들로 대응을 전파(propagation)한다.

#### 기하학적 트랜스포머 아키텍처

일반적인 좌표 인코딩은 점군이 회전·이동하면 값이 바뀌어 버린다. GeoTransformer는 강체 변환($SE(3)$)을 가해도 절대 변하지 않는 두 가지 상대 기하량으로 위치 인코딩을 설계했다.

1. **쌍별 거리 인코딩(Pairwise Distance)**: 두 슈퍼포인트 $\mathbf{p}_i, \mathbf{p}_j$ 사이의 유클리드 거리 $d_{ij} = \|\mathbf{p}_i - \mathbf{p}_j\|$는 점군 전체를 어떻게 돌리고 옮겨도 변하지 않는다. 이 거리를 사인·코사인 주파수 인코딩으로 변환한다.

$$\text{PE}(d_{ij}) = [\sin(\omega_1 d_{ij}), \cos(\omega_1 d_{ij}), \ldots, \sin(\omega_K d_{ij}), \cos(\omega_K d_{ij})]$$

2. **삼중 각도 인코딩(Triplet Angle)**: 거리만으로는 좌우 대칭을 구분하기 어려우므로, 세 점 $\mathbf{p}_i, \mathbf{p}_j, \mathbf{p}_k$가 이루는 각도를 함께 인코딩해 attention 점수 계산에 보탠다. 이 각도 역시 강체 변환에 불변이다.

$$\text{Attn}_{ij} = \frac{\mathbf{q}_i^\top \mathbf{k}_j}{\sqrt{d}} + b(\text{PE}(d_{ij}))$$

#### RANSAC-Free 변환 추정

강체 변환 불변 특징 덕분에 슈퍼포인트 대응의 정확도가 크게 올라가자, GeoTransformer는 무작위 샘플링(RANSAC)을 생략하고 **local-to-global registration**으로 단번에 최적 변환을 결정한다. 먼저 각 슈퍼포인트 이웃 패치의 대응마다 1차 변환 가설을 하나씩 세운 뒤, 이를 전체 점군에 대입해 인라이어가 가장 많이 들어맞는 가설을 고른다. 저자들은 이 방식이 기존 RANSAC 기반 추정보다 100배 빨랐다고 보고했다(다만 아웃라이어 비율이 훨씬 더 극심한 현장 데이터에서도 RANSAC 없이 같은 강건성이 유지되는지는 별도 검증이 필요하다).

#### 성능

원 논문에서는 겹침이 적은 3DLoMatch 벤치마크에서 인라이어 비율이 17~30%p, registration recall이 7%p 넘게 상승했다고 보고했다.

실전에서는 두 포인트 클라우드를 다운샘플링한 뒤 모델에 통과시켜 변환 행렬 `(4, 4)`와 대응점 쌍을 한 번에 얻는다.

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

수작업 히스토그램에서 희소 합성곱, 겹침 인식, 그리고 기하 불변 트랜스포머로 이어지는 3D 정합의 흐름은 다음과 같이 요약된다.

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

지금까지는 2D는 2D끼리, 3D는 3D끼리 같은 차원의 데이터를 맞췄다. 그렇다면 센서 융합에서 빠질 수 없는 문제, 즉 차원과 물리적 성질이 완전히 다른 **2D 카메라 이미지와 3D LiDAR 포인트 클라우드** 사이에서는 어떻게 대응점을 찾을 수 있을까?

---

## 5.9 Cross-Modal Correspondence

### 5.9.1 2D-3D Matching: 카메라-LiDAR 간

지금까지 다룬 매칭은 주로 같은 차원의 관측(2D-2D 혹은 3D-3D)을 짝지었다면, cross-modal correspondence는 **2D 카메라 이미지와 3D 포인트 클라우드**라는 이질적인 두 센서를 직접 잇는 문제다. 이 대응은 센서 융합 시스템에서 크게 세 곳에 꼭 필요하다.

- **카메라-LiDAR 외부 캘리브레이션**: 두 센서가 동시에 바라본 같은 물리적 지점을 연결해 센서 사이의 상대 회전·이동을 구한다.
- **LiDAR 맵 기반 Visual Localization**: 정밀 LiDAR로 미리 구축해 둔 3D 지도 안에서 카메라 사진 한 장만으로 현재 위치를 찾아낸다.
- **Loop Closure**: 카메라와 LiDAR 관측을 서로 교차 대조해 이전에 방문했던 장소인지 검증한다.

### 5.9.2 Image-to-Point Cloud 매칭 접근법

성질이 전혀 다른 2D 영상과 3D 점군을 맞추기 위해 크게 두 가지 전략이 쓰인다.

#### 투영 기반 접근 (Projection-based)

가장 실용적인 우회로는 3D 포인트 클라우드를 2D 이미지 평면으로 끌어내려 익숙한 2D-2D 매칭 문제로 바꾸어 푸는 전략이다.

1. 밀집된 LiDAR 포인트 클라우드를 가상의 카메라 시점으로 렌더링해 depth 이미지나 intensity 이미지를 만든다.
2. 렌더링된 가상 영상과 실제 카메라 영상을 SuperGlue 같은 2D 학습 기반 매처로 짝짓는다.
3. 렌더링할 때 알고 있던 깊이 값을 이용해, 앞서 찾은 2D 매칭 좌표를 다시 3D 좌표로 되돌려 2D-3D 대응점을 얻는다.

실제로 Koide et al.(2023)의 targetless 캘리브레이션 툴이 바로 이 전략으로 초기값을 잡는다. 밀집 LiDAR 점군을 여러 가상 시점으로 투영한 뒤 SuperGlue로 카메라 이미지와 맞춰 2D-3D 대응을 구하고, 이후 §5.4의 NID 최적화로 정밀하게 다듬는다.

#### 학습 기반 직접 매칭

가상 렌더링이라는 중간 다리를 거치지 않고, 신경망으로 2D 패치와 3D 점군을 직접 연결하려는 연구도 있다. **LCD**(Learned Cross-Domain Descriptors, Pham et al., AAAI 2020)는 2D 이미지 패치와 3D 포인트 클라우드 패치가 같은 물리적 대상에 해당하면 공통 임베딩 공간의 가까운 위치에 모이도록 학습한다. 한 걸음 더 나아간 **P2-Net**(Wang et al., ICCV 2021)은 패치를 따로 잘라내지 않고 이미지 픽셀과 3D 점 사이의 직접적인 대응을 한 번에 학습한다.

### 5.9.3 Cross-Modal 매칭이 어려운 이유

그렇다면 2D-3D cross-modal 매칭은 왜 동종 센서 간 매칭보다 훨씬 까다로울까? 두 센서 사이에는 네 가지 근본적인 간극이 가로놓여 있다.

1. **표현의 이질성(Representation Heterogeneity)**: 2D 이미지는 일정한 픽셀 격자 위에 촘촘히 놓인 밝기와 색상 데이터지만, 3D 포인트 클라우드는 공간에 불규칙하게 흩어진 점들의 좌표와 반사 강도다. 2D CNN과 3D 점군 네트워크의 구조 자체가 달라 정보를 나란히 견주기가 쉽지 않다.
2. **정보 비대칭(Information Asymmetry)**: 카메라 영상에는 글자, 색감, 질감 같은 시각 텍스처가 넘쳐나지만 깊이 정보가 빠져 있다. 반대로 LiDAR 포인트 클라우드에는 정확한 3차원 기하 형상이 담겨 있지만 색상이나 세밀한 텍스처가 거의 없다.
3. **밀도 차이(Density Gap)**: 수백만 픽셀이 꽉 들어찬 카메라 영상 한 장에 비해, LiDAR 스캔 한 장의 점 수는 훨씬 적다. 더구나 LiDAR 점군은 센서에서 멀어질수록 간격이 성겨져 카메라 픽셀과 일대일로 맞추기가 까다롭다.
4. **외관 도메인 갭(Appearance Domain Gap)**: 같은 물체를 찍더라도 두 센서가 측정하는 물리량 자체가 다르다. 카메라는 주변 조명 조건과 표면 반사 특성이 뒤섞인 가시광 밝기를 기록하는 반면, LiDAR는 스스로 쏜 특정 파장 레이저의 반사 광량(intensity)을 잰다. 표면 고유의 반사율(albedo)은 두 센서의 측정값 속에 간접적으로만 녹아 있어 밝기 값을 직접 비교할 수 없다.

바로 이 네 가지 장벽 때문에 §5.4의 MI 기반 방법은 두 값을 직접 비교하는 대신 통계적 의존성으로 간극을 건너뛰고, 투영 기반 방법은 아예 3D 점군을 2D 영상 형식으로 바꾸어 2D 매처의 힘을 빌린다.

지금까지 살펴본 기법들은 주로 영상이나 점군에서 대표 지점들을 골라 짝짓는 희소(sparse) 대응이나 이질 센서 간 정합에 초점을 맞췄다. 마지막으로 5.10절에서는 연속된 영상 프레임이나 스테레오 카메라에서 **화면의 모든 픽셀**을 따라 움직임과 시차를 촘촘히 계산하는 **dense matching과 optical flow**로 넘어가 보자.

---

## 5.10 Dense Matching & Optical Flow

실시간 VO/VIO에서는 매 프레임 디스크립터를 새로 뽑아 짝짓는 대신, 직전 프레임의 픽셀이 다음 프레임에서 어디로 옮겨 갔는지 곧바로 추적하거나(optical flow), 좌우 스테레오 영상의 모든 픽셀에 대해 깊이(시차)를 구하는 일이 잦다. Dense matching은 이처럼 이미지의 전 픽셀 수준에서 빈틈없이 대응을 찾는 문제다.

### 5.10.1 Classical Optical Flow: Lucas-Kanade, Horn-Schunck

#### Optical Flow의 정의

**Optical flow**는 연속된 이미지 시퀀스에서 각 픽셀이 화면 위에서 얼마나 움직였는지(겉보기 운동, apparent motion)를 2차원 벡터 필드로 나타낸 값이다.

$$\mathbf{u}(x, y) = (u(x, y), v(x, y))$$

고전적인 optical flow 방정식은 한 가지 핵심 가정에서 출발한다. 물체가 살짝 움직여 다음 프레임의 $(x+u, y+v)$ 위치로 이동하더라도 그 픽셀의 밝기 값 자체는 그대로 유지된다는 **밝기 항상성 가정(brightness constancy assumption)**이다.

$$I(x, y, t) = I(x + u, y + v, t + 1)$$

움직임 $(u, v)$가 작다고 보고 우변에 1차 Taylor 전개를 적용해 양변의 $I(x, y, t)$를 소거하면, 그 유명한 optical flow constraint equation을 얻는다.

$$I_x u + I_y v + I_t = 0$$

여기서 $I_x = \frac{\partial I}{\partial x}$, $I_y = \frac{\partial I}{\partial y}$는 공간 그래디언트고, $I_t = \frac{\partial I}{\partial t}$는 시간에 따른 밝기 변화량이다. 그런데 이 식에는 치명적인 문제가 있다. 픽셀 하나당 식은 단 하나뿐인데 풀어야 할 미지수는 $(u, v)$ 두 개라는 점이다. 즉 좁은 구멍(aperture) 너머로 움직이는 직선을 바라보면 선을 따라 움직이는 성분을 알 수 없는 **aperture problem**이 생긴다. 해를 하나로 결정하려면 추가 제약 조건이 반드시 필요하며, 1981년 나란히 발표된 두 고전 알고리즘이 서로 다른 해법을 제시했다.

#### Lucas-Kanade (1981)

Lucas-Kanade는 **로컬 일관성 가정**으로 미지수 문제를 푼다. 한 픽셀만 보지 않고 그 주변의 작은 윈도우 $\Omega$ 안에 있는 픽셀들은 모두 같은 속도로 움직인다고 가정한다. 윈도우 안에 픽셀이 $n$개 있다면 미지수 $(u, v)$ 둘에 대해 식이 $n$개나 모이므로(overdetermined) 다음과 같은 연립방정식이 만들어진다.

$$\begin{bmatrix} I_{x_1} & I_{y_1} \\ I_{x_2} & I_{y_2} \\ \vdots & \vdots \\ I_{x_n} & I_{y_n} \end{bmatrix} \begin{bmatrix} u \\ v \end{bmatrix} = -\begin{bmatrix} I_{t_1} \\ I_{t_2} \\ \vdots \\ I_{t_n} \end{bmatrix}$$

이를 $\mathbf{A} \mathbf{u} = -\mathbf{b}$로 줄여 적고 최소자승 해를 구하면 다음과 같다.

$$\begin{bmatrix} u \\ v \end{bmatrix} = (\mathbf{A}^\top \mathbf{A})^{-1} \mathbf{A}^\top (-\mathbf{b})$$

흥미롭게도 여기서 역행렬을 취하는 $2 \times 2$ 행렬 $\mathbf{A}^\top \mathbf{A} = \begin{bmatrix} \sum I_x^2 & \sum I_x I_y \\ \sum I_x I_y & \sum I_y^2 \end{bmatrix}$는 §5.2.1에서 보았던 Harris corner detector의 **structure tensor $\mathbf{M}$과 정확히 같다.** 평탄한 영역이나 직선 에지 위에서는 $\mathbf{M}$의 역행렬을 안정적으로 구할 수 없지만, 두 고유값이 모두 큰 코너 주변에서는 $\mathbf{A}^\top \mathbf{A}$가 잘 조건화(well-conditioned)되어 흐름을 깔끔하게 풀 수 있다. 바로 이 때문에 VINS-Mono 같은 시스템은 추적할 점으로 Shi-Tomasi나 Harris 코너를 뽑은 뒤 Lucas-Kanade로 추적한다.

**한계**: 밝기 항상성 식에 1차 Taylor 전개를 썼기 때문에, 프레임 사이 움직임이 수 픽셀 이상으로 커지면(large displacement) 선형 근사가 깨진다. 실전에서는 **Pyramidal LK**(Bouguet 2000)로 이 한계를 넘는다. 이 방식은 해상도를 낮춘 이미지 피라미드의 꼭대기(거친 해상도)에서 대략적인 이동량을 먼저 잡고, 아래층으로 내려오며 오차를 다듬는다.

OpenCV에서는 `goodFeaturesToTrack`으로 추적하기 좋은 코너를 뽑은 뒤 `calcOpticalFlowPyrLK`로 다음 프레임에서의 위치를 추적한다.

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

Lucas-Kanade가 코너 주변의 희소(sparse) 추적으로 발전했다면, Horn-Schunck는 이미지의 모든 픽셀에서 촘촘한(dense) flow를 구하기 위해 **전역 평활성 가정**을 택했다. 이웃한 픽셀끼리는 흐름 벡터 $(u, v)$가 부드럽게 변한다는 제약을 더해, 이미지 전체에 걸친 에너지 함수를 최소화한다.

$$E = \iint \left[ (I_x u + I_y v + I_t)^2 + \lambda^2 (\|\nabla u\|^2 + \|\nabla v\|^2) \right] dx \, dy$$

첫째 항은 밝기 항상성 오차를 줄이는 데이터 항이고, 둘째 항은 flow 변화가 급격해지지 않도록 누르는 평활성 정규화 항이다. 가중치 $\lambda$를 키울수록 flow 장이 매끈해진다. 변분법의 Euler-Lagrange 방정식을 풀어 반복 갱신식을 얻으면 텍스처가 없는 평탄한 영역까지 전 픽셀의 flow가 채워지지만, 움직이는 물체의 경계처럼 flow가 뚝 끊겨야 하는 곳까지 뭉개져 번지는 약점이 있다.

### 5.10.2 Learning-Based Optical Flow: FlowNet → RAFT → FlowFormer → UniMatch

고전 변분법의 밝기 항상성 가정은 조명이 살짝 바뀌거나 물체가 빠르게 지나갈 때 쉽게 무너진다. 딥러닝 시대에 접어들며 대규모 합성·실제 데이터에서 직접 optical flow를 예측하는 모델들이 비약적으로 발전했다.

#### FlowNet / FlowNet 2.0 (2015/2017)

[Dosovitskiy et al. (2015)](https://arxiv.org/abs/1504.06852)의 **FlowNet**은 두 이미지를 이어 붙이거나 특징 맵 상관을 구한 뒤 encoder-decoder CNN을 통과시켜 optical flow를 직접 회귀하도록 지도 학습한 선구적 연구다. 뒤이어 나온 [FlowNet 2.0 (Ilg et al., 2017)](https://arxiv.org/abs/1612.01925)은 여러 개의 FlowNet 블록을 직렬로 쌓아 이전 단계의 추정치로 영상을 와핑(warping)해 가며 잔차를 보정하는 방식으로 정확도를 고전 기법 수준 이상으로 끌어올렸다.

#### RAFT (2020)

[Teed & Deng (2020)](https://arxiv.org/abs/2003.12039)의 **RAFT**(Recurrent All-Pairs Field Transforms)는 ECCV 2020 Best Paper를 수상하며 학습 기반 optical flow의 새로운 표준을 세웠다. RAFT에서는 모든 픽셀 쌍의 상관을 한 번에 계산해 둔 **4D correlation volume**과, 이를 참조하며 흐름을 조금씩 고쳐 나가는 **GRU 기반 반복 갱신**이 핵심이다.

1. **Feature Encoder & Context Encoder**: 두 입력 이미지를 공유 CNN(ResNet 변형)에 통과시켜 1/8 해상도의 특징 맵 $\mathbf{g}_1, \mathbf{g}_2 \in \mathbb{R}^{H/8 \times W/8 \times D}$를 뽑는다. 이와 별개로 Context Encoder가 첫 번째 이미지에서 배경·물체 경계 정보를 담은 context feature와 GRU의 초기 hidden state를 만든다.
2. **Correlation Volume**: 첫 번째 이미지의 모든 픽셀과 두 번째 이미지의 모든 픽셀 쌍에 대해 내적을 계산한다.

$$C_{ijkl} = \sum_d g_1(i, j, d) \cdot g_2(k, l, d)$$

이 연산으로 4D correlation volume $\mathbf{C} \in \mathbb{R}^{H/8 \times W/8 \times H/8 \times W/8}$이 만들어진다(네 축 모두 특징 맵 해상도 기준). 여기서 뒤쪽 두 차원을 average pooling으로 줄여 4단계의 correlation pyramid(스케일 1, 2, 4, 8)를 쌓는다. flow 추정치 자체는 1/8 단일 해상도에서만 계속 갱신하되, 상관값은 여러 스케일의 피라미드에서 동시에 훔쳐보는 영리한 구조다.
3. **Iterative Update (GRU)**: ConvGRU 블록이 초기값 0에서 출발해 flow 추정치를 여러 차례 고쳐 나간다. 매 반복 $k$마다 다음 다섯 단계가 이어진다.
   1. 현재 추정 flow $\mathbf{f}^k$가 가리키는 두 번째 이미지의 대응 위치 주변 작은 윈도우에서 correlation pyramid 값을 조회한다.
   2. 조회한 상관값, 현재 flow $\mathbf{f}^k$, context feature를 한데 묶는다.
   3. ConvGRU가 hidden state $\mathbf{h}^k$를 갱신한다.
   4. 갱신된 hidden state에서 flow의 수정량(residual) $\Delta \mathbf{f}$를 예측한다.
   5. $\mathbf{f}^{k+1} = \mathbf{f}^k + \Delta \mathbf{f}$로 flow를 한 걸음 보정한다.

원 논문에서는 훈련할 때 12번 반복 갱신하고, 평가할 때는 Sintel에서 32번, KITTI에서 24번 갱신했다. 논문 실험을 보면 반복 횟수를 늘릴수록 오차가 더 줄어드는 경향이 나타났으며, 실전에서는 허용 지연 시간과 검증 오차를 저울질해 적정 횟수를 정한다.

**All-Pairs와 Coarse-to-Fine**: 기존 coarse-to-fine 모델들은 거친 해상도에서 작은 물체의 큰 움직임을 놓치면 뒤쪽 세밀한 단계에서도 복구하기 어려웠다. 반면 RAFT는 처음부터 모든 특징 쌍의 상관을 다 계산해 두고 반복마다 여러 스케일에서 찾아보기 때문에, 작고 빠르게 움직이는 물체라도 후보 목록에서 사라지지 않는다(Teed & Deng 2020 §1).

**학습**: 매 반복 단계 $k$가 내놓는 중간 flow 예측마다 정답 $\mathbf{f}^{gt}$와의 L1 오차를 재되, 뒤쪽 반복일수록 지수적으로 높은 가중치를 주는 손실로 학습한다($\gamma = 0.8$).

$$L = \sum_{k=1}^{K} \gamma^{K-k} \| \mathbf{f}^k - \mathbf{f}^{gt} \|_1$$

`torchvision`에는 사전 학습된 RAFT 모델이 내장되어 있어 몇 줄의 코드만으로 바로 optical flow를 추론할 수 있다.

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

[FlowFormer (Huang et al., 2022)](https://arxiv.org/abs/2203.16194)는 RAFT가 만든 4D cost volume을 그대로 두지 않고 토큰화한 뒤 트랜스포머 인코더에서 전역 문맥이 녹아든 **cost memory**로 압축한다. 이어서 recurrent transformer decoder가 이 memory를 참조하며 flow를 반복 갱신한다. 논문에서는 그 결과 Sintel 벤치마크에서 RAFT를 앞서는 낮은 오차를 기록했다고 보고했다.

#### UniMatch (2023)

[Xu et al. (2023)](https://arxiv.org/abs/2211.05783)의 **UniMatch**는 그동안 따로따로 연구되던 optical flow, stereo matching, 깊이 추정을 모두 "두 시점 사이의 dense correspondence"라는 하나의 문제로 통합해 단일 아키텍처로 풀어냈다.

### 5.10.3 Dense Stereo: SGM → RAFT-Stereo → UniMatch

좌우 두 카메라가 수평으로 정렬(rectified)된 스테레오 환경에서는 에피폴라 선이 모두 수평 주사선과 일치한다. 따라서 대응점이 위아래로 움직일 여지가 사라지고, stereo matching은 수직 성분이 0인 순수 수평 방향 **시차(disparity)** 추정 문제로 단순화된다.

#### SGM (Semi-Global Matching, Hirschmuller 2005)

**SGM**은 오랫동안 자율주행과 로보틱스 현장을 지켜 온 전통적 dense stereo의 표준이다. 픽셀마다 한 행 위의 모든 시차 후보 $d$에 대해 매칭 비용 $C(p, d)$를 계산한 뒤, 노이즈에 흔들리지 않도록 8개(또는 16개) 방사 방향을 따라 1차원 동적 계획법(DP)으로 비용을 긁어모아(aggregation), 최종 비용이 가장 작은 시차를 고른다.

$$L_r(p, d) = C(p, d) + \min\begin{cases} L_r(p-r, d) \\ L_r(p-r, d-1) + P_1 \\ L_r(p-r, d+1) + P_1 \\ \min_i L_r(p-r, i) + P_2 \end{cases}$$

여기서 $P_1$은 경사면처럼 시차가 1만큼 부드럽게 변할 때 주는 작은 페널티고, $P_2$는 물체 경계에서 시차가 확 뛸 때 적용하는 큰 페널티다.

#### RAFT-Stereo (2021)

**RAFT-Stereo**(Lipson et al., 3DV 2021)는 RAFT의 성공 공식을 스테레오 매칭에 그대로 이식했다. 같은 수평선 위의 픽셀끼리만 내적해 4D 대신 가벼운 **3D correlation volume**을 만들고, 다중 스케일 ConvGRU로 시차 맵을 반복 갱신해 정밀한 깊이 정보를 뽑아낸다.

#### UniMatch의 통합

앞서 소개한 UniMatch는 같은 백본과 매칭 구조 위에서 과제에 따라 **탐색 범위(매칭 차원)**만 살짝 바꿔 세 가지 과제를 모두 해결한다.

- **Stereo**: 같은 수평선 위를 훑는 1D correlation
- **Flow**: 이미지 평면 전체를 훑는 2D all-pairs correlation
- **Depth**: 카메라 자세를 알 때 에피폴라 선을 따라 깊이 후보만 훑는 1D 매칭

### 5.10.4 Sparse Feature에서 Dense Correspondence까지

희소한 코너 추적에서 출발해 전 픽셀의 밀집 흐름과 깊이 통합 모델까지 이어진 발전 궤적을 정리하면 다음과 같다.

```
[Sparse Correspondence]
Harris → SIFT → SuperPoint → SuperGlue → LightGlue
    ↓ 밀집화
[Dense Correspondence]
Lucas-Kanade (1981)  — 로컬 window, sparse-to-dense
Horn-Schunck (1981)  — 글로벌 smoothness, 전 픽셀 flow
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

이러한 dense correspondence 아이디어는 SLAM에도 큰 파장을 일으켰다. 대표적으로 **DROID-SLAM**(Teed & Deng, 2021)은 RAFT의 4D correlation volume과 반복 GRU 갱신 구조를 SLAM 프런트엔드·백엔드 결합 루프(Dense Bundle Adjustment)로 가져와 딥러닝 SLAM의 성능을 한 차원 끌어올렸다.

---

## 요약

5장에서는 "서로 다른 관측에서 물리적으로 같은 대상을 어떻게 짝지을 것인가"라는 대응점 탐색의 문제를 고전 기법부터 최신 파운데이션 모델과 3D·dense 기법까지 폭넓게 짚어 보았다. 2D 특징점 검출과 기술은 Harris·SIFT·ORB의 속도-정확도 균형에서 SuperPoint·SuperGlue·LightGlue의 학습 기반 파이프라인으로, 다시 검출기마저 없애 버린 LoFTR·RoMa와 직접 3D를 예측하는 DUSt3R·VGGT로 진화했다. 같은 시기에 optical flow와 3D-3D 점군 정합, 그리고 이질 센서를 잇는 MI·cross-modal 매칭도 각자의 제약을 넘으며 발전해 왔다. 아래 전체 기술 계보도로 각 갈래의 연결 고리를 한눈에 정리해 보자.

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

이렇게 5장에서 확보한 2D-2D, 2D-3D 대응점과 optical flow는 그 자체로 끝이 아니라, 비로소 상태 추정에 투입할 수 있는 신뢰할 만한 관측 데이터다. **Visual Odometry(VO)**와 **Visual-Inertial Odometry(VIO)** 시스템은 이 대응점 프런트엔드를 4장의 칼만 필터·비선형 최적화 백엔드와 하나로 결합해, 카메라와 IMU만으로 실시간 이동 궤적을 복원한다. 이제 6장에서는 이 VO·VIO 시스템이 어떻게 설계되고 동작하는지 살펴본다.

