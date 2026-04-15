# Tier 1A — 핵심 논문 요약 (Landmark Papers)

> **목적**: sensor-fusion-guide의 깊은 기술 콘텐츠 작성을 위한 핵심 논문 요약
> **작성일**: 2026-04-15
> **논문 수**: 6편 (필터링, 로버스트 추정, 정합, 캘리브레이션, 특징 추출, 검색)

---

## 1. A New Approach to Linear Filtering and Prediction Problems

- **저자/연도**: Rudolf E. Kalman, 1960
- **발표**: Transactions of the ASME — Journal of Basic Engineering, Vol. 82, Series D, pp. 35–45
- **핵심 기여**: 이산 시간 선형 시스템의 최적 상태 추정을 **재귀적(recursive)**으로 풀 수 있는 수학적 프레임워크를 제시. Wiener 필터의 주파수 도메인 접근법을 상태공간(state-space) 표현으로 대체하여, 시변(time-varying) 시스템과 다변량 시스템에 자연스럽게 확장 가능하게 만들었다.

- **주요 내용**:
  - **상태공간 모델**: 시스템을 상태 전이 방정식과 관측 방정식으로 표현
    - 상태 전이: $\hat{\mathbf{x}}_{k|k-1} = \mathbf{F}_k \hat{\mathbf{x}}_{k-1|k-1} + \mathbf{B}_k \mathbf{u}_k$
    - 관측 모델: $\mathbf{z}_k = \mathbf{H}_k \mathbf{x}_k + \mathbf{v}_k$
  - **Predict-Update 2단계 구조**:
    - **예측(Predict)**:
      - 상태 예측: $\hat{\mathbf{x}}_{k|k-1} = \mathbf{F}_k \hat{\mathbf{x}}_{k-1|k-1} + \mathbf{B}_k \mathbf{u}_k$
      - 공분산 예측: $\mathbf{P}_{k|k-1} = \mathbf{F}_k \mathbf{P}_{k-1|k-1} \mathbf{F}_k^\top + \mathbf{Q}_k$
    - **갱신(Update)**:
      - 혁신(innovation): $\tilde{\mathbf{y}}_k = \mathbf{z}_k - \mathbf{H}_k \hat{\mathbf{x}}_{k|k-1}$
      - 혁신 공분산: $\mathbf{S}_k = \mathbf{H}_k \mathbf{P}_{k|k-1} \mathbf{H}_k^\top + \mathbf{R}_k$
      - 칼만 이득: $\mathbf{K}_k = \mathbf{P}_{k|k-1} \mathbf{H}_k^\top \mathbf{S}_k^{-1}$
      - 상태 갱신: $\hat{\mathbf{x}}_{k|k} = \hat{\mathbf{x}}_{k|k-1} + \mathbf{K}_k \tilde{\mathbf{y}}_k$
      - 공분산 갱신: $\mathbf{P}_{k|k} = (\mathbf{I} - \mathbf{K}_k \mathbf{H}_k) \mathbf{P}_{k|k-1}$
  - **최적성**: 선형-가우시안 시스템에서 MMSE(Minimum Mean Square Error) 추정량. 전체 관측 이력이 아닌 직전 추정치와 새 관측만으로 최적 추정 가능(재귀적 구조).
  - **Wiener 필터와의 차이**: Wiener 필터는 정상(stationary) 과정, 주파수 도메인, 스칼라에 적합. Kalman 필터는 비정상, 시간 도메인, 다변량으로 일반화.
  - **실용적 돌파**: Stanley Schmidt가 Apollo 프로그램 궤도 추정에 최초 실용화. 이후 항법, 제어, 신호처리 전 분야의 표준이 됨.

- **이 가이드에서의 위치**:
  - **Ch.5 (State Estimation & Filtering)**: 가이드의 핵심 챕터. Kalman 필터 유도를 Bayesian 관점에서 상세히 다루고, predict-update 구조를 센서 퓨전의 기본 프레임워크로 설명.
  - **Ch.6 (LiDAR-Inertial / Visual-Inertial Odometry)**: EKF/ESKF 기반 퓨전 시스템의 이론적 기초로 참조.
  - **Ch.1 (Introduction)**: 센서 퓨전의 역사에서 Kalman 필터의 위치를 설명할 때 인용.

- **후속 영향**:
  - **EKF (Extended Kalman Filter)**: 비선형 시스템에 1차 테일러 전개로 확장. SLAM, INS/GNSS 퓨전의 표준.
  - **UKF (Unscented Kalman Filter)**: Sigma point를 사용한 비선형 확장. 자코비안 계산 불필요.
  - **ESKF (Error-State Kalman Filter)**: VIO/LIO 시스템(MSCKF, VINS-Mono 등)에서 표준적으로 사용.
  - **Factor Graph / GTSAM**: Kalman 필터의 배치(batch) 확장으로 볼 수 있는 그래프 최적화 접근.
  - **Kalman-Bucy Filter (1961)**: 연속 시간 확장.

---

## 2. Random Sample Consensus: A Paradigm for Model Fitting with Applications to Image Analysis and Automated Cartography

- **저자/연도**: Martin A. Fischler, Robert C. Bolles, 1981
- **발표**: Communications of the ACM, Vol. 24, No. 6, pp. 381–395
- **핵심 기여**: 아웃라이어가 포함된 데이터에서 모델 파라미터를 로버스트하게 추정하는 **RANSAC 패러다임**을 제안. 기존의 최소자승법(Least Squares)이 아웃라이어에 취약한 근본적 한계를 해결.

- **주요 내용**:
  - **기존 방법의 문제**: 전통적 smoothing 기법은 모든 데이터를 사용하여 초기 해를 구한 뒤 이상치를 제거하려 시도 → 아웃라이어가 초기 해 자체를 오염시킴.
  - **RANSAC의 역발상**: 최소한의 데이터 부분집합으로 모델을 적합하고, 그 모델에 부합하는 데이터(consensus set)를 확장하는 방식.
  - **알고리즘 단계**:
    1. 데이터에서 모델 적합에 필요한 최소 $n$개 점을 무작위 추출
    2. 추출한 점으로 모델 파라미터 추정
    3. 전체 데이터에서 모델과의 오차가 임계값 $t$ 이내인 점(인라이어)을 찾아 consensus set 구성
    4. Consensus set 크기가 충분하면, 인라이어 전체로 모델을 재추정
    5. 1–4를 반복, 가장 큰 consensus set을 가진 모델을 최종 해로 선택
  - **반복 횟수 결정 공식**:
    $$k = \frac{\log(1-p)}{\log(1-w^n)}$$
    여기서 $p$ = 성공 확률(보통 0.99), $w$ = 인라이어 비율, $n$ = 모델에 필요한 최소 점 수, $k$ = 필요한 반복 횟수.
  - **적용 사례**: 논문에서는 Location Determination Problem(LDP)에 적용 — 랜드마크가 보이는 이미지로부터 카메라 위치를 결정하는 문제 (PnP 문제의 초기 형태).

- **이 가이드에서의 위치**:
  - **Ch.4 (Feature Matching & Geometric Verification)**: 특징점 매칭 후 기하학적 검증(Essential/Fundamental matrix 추정)에서 RANSAC을 핵심 기법으로 다룸.
  - **Ch.3 (Calibration)**: 캘리브레이션 파이프라인의 로버스트 추정 단계에서 언급.
  - **Ch.6 (Visual Odometry)**: 프레임 간 모션 추정에서 RANSAC 기반 아웃라이어 제거.

- **후속 영향**:
  - **MLESAC (Torr & Zisserman, 2000)**: Consensus set의 카디널리티 대신 likelihood로 평가.
  - **PROSAC (Chum & Matas, 2005)**: 사전 정보(매칭 품질 점수)를 활용한 progressive sampling.
  - **LO-RANSAC**: 로컬 최적화를 추가하여 정밀도 향상.
  - **USAC (Raguram et al., 2013)**: RANSAC 변형들을 통합한 프레임워크.
  - **MAGSAC / MAGSAC++ (Barath et al.)**: 임계값 $t$ 선택 문제를 자동화.
  - **GC-RANSAC**: Graph-Cut 기반 로컬 최적화.
  - 현대 Visual SLAM/VO (ORB-SLAM, OpenCV)에서 여전히 핵심 구성요소.

---

## 3. A Method for Registration of 3-D Shapes

- **저자/연도**: Paul J. Besl, Neil D. McKay, 1992
- **발표**: IEEE Transactions on Pattern Analysis and Machine Intelligence, Vol. 14, No. 2, pp. 239–256
- **핵심 기여**: 3D 형상의 정합(registration)을 위한 **ICP(Iterative Closest Point)** 알고리즘을 제안. 표현 방식(점군, 곡면 등)에 독립적이며, 6-DOF 강체 변환을 반복적으로 최적화.

- **주요 내용**:
  - **문제 정의**: 두 3D 형상 $P$(source)와 $X$(target) 사이의 강체 변환 $(\mathbf{R}, \mathbf{t})$를 찾아 정합하는 문제.
  - **ICP 알고리즘 단계**:
    1. **대응점 탐색(Closest Point Matching)**: Source 점군의 각 점 $\mathbf{p}_i$에 대해 Target에서 가장 가까운 점 $\mathbf{x}_i$를 찾음 (유클리드 거리 기준)
    2. **변환 추정(Transformation Estimation)**: 대응 쌍 $\{(\mathbf{p}_i, \mathbf{x}_i)\}$으로부터 MSE를 최소화하는 $(\mathbf{R}, \mathbf{t})$ 계산
       $$\min_{\mathbf{R}, \mathbf{t}} \sum_{i=1}^{N} \|\mathbf{x}_i - (\mathbf{R}\mathbf{p}_i + \mathbf{t})\|^2$$
    3. **변환 적용**: Source 점군에 추정된 변환 적용
    4. **수렴 판정**: MSE 변화량이 임계값 이하가 될 때까지 1–3 반복
  - **최적 변환 계산**: 대응 쌍이 주어지면, 중심점을 원점으로 이동 후 SVD를 이용하여 회전 행렬 $\mathbf{R}$을 계산하고, 이동 벡터 $\mathbf{t}$를 복원하는 closed-form 해법 사용.
  - **수렴 보장**: ICP는 mean-square distance 메트릭의 가장 가까운 **지역 최솟값(local minimum)**으로 항상 단조 수렴함을 증명. 초기 몇 반복에서 수렴 속도가 빠름.
  - **표현 독립성**: 점 집합(point set), 선분(line segment), 암묵적 곡면(implicit curve), 파라메트릭 곡면, 삼각형 집합 등 다양한 기하학적 표현에 적용 가능.
  - **전역 최적화**: 충분한 초기 회전/이동 조합을 시도하면 전역 최솟값에 도달할 수 있다고 제안 (실용적으로는 좋은 초기값이 필요).

- **이 가이드에서의 위치**:
  - **Ch.6 (LiDAR Odometry)**: Scan-to-scan, scan-to-map 정합의 핵심 알고리즘. LOAM, LeGO-LOAM 등의 기초.
  - **Ch.3 (Calibration)**: LiDAR-LiDAR, LiDAR-Camera 외부 캘리브레이션의 정합 단계에서 사용.
  - **Ch.8 (Map Representation)**: 다중 스캔 정합을 통한 3D 맵 구축.

- **후속 영향**:
  - **Point-to-Plane ICP (Chen & Medioni, 1992)**: 동시에 독립적으로 제안. 수렴 속도가 point-to-point보다 빠름.
  - **Generalized ICP (Segal et al., 2009)**: Point-to-point와 point-to-plane을 확률적으로 통합.
  - **NDT (Biber & Straßer, 2003)**: Normal Distribution Transform — 점군을 가우시안 셀로 모델링.
  - **LOAM (Zhang & Singh, 2014)**: Edge/planar feature 기반 ICP 변형으로 LiDAR odometry의 표준.
  - **KISS-ICP (Vizzo et al., 2023)**: 단순하지만 강력한 현대적 point-to-point ICP 변형.
  - **비강체(Non-rigid) ICP**: 변형 가능한 물체 정합으로 확장 (의료 영상 등).

---

## 4. A Flexible New Technique for Camera Calibration

- **저자/연도**: Zhengyou Zhang, 2000 (Microsoft Research 테크리포트는 1998)
- **발표**: IEEE Transactions on Pattern Analysis and Machine Intelligence, Vol. 22, No. 11, pp. 1330–1334
- **핵심 기여**: 평면 패턴(체커보드)만으로 카메라의 내부 파라미터와 렌즈 왜곡을 캘리브레이션하는 유연하고 실용적인 방법을 제안. 3D 캘리브레이션 장비 없이도 고정밀 캘리브레이션이 가능하게 함.

- **주요 내용**:
  - **핵심 아이디어**: 평면 패턴(z=0)을 여러 방향에서 촬영하면, 3D-2D 투영 관계가 호모그래피(homography)로 표현됨. 이 호모그래피들로부터 카메라 내부 파라미터를 추출.
  - **호모그래피 기반 모델**: 평면 패턴의 점 $\mathbf{M} = [X, Y, 0]^\top$과 이미지 점 $\mathbf{m} = [u, v]^\top$의 관계:
    $$s \begin{bmatrix} u \\ v \\ 1 \end{bmatrix} = \mathbf{K} [\mathbf{r}_1 \quad \mathbf{r}_2 \quad \mathbf{t}] \begin{bmatrix} X \\ Y \\ 1 \end{bmatrix} = \mathbf{H} \begin{bmatrix} X \\ Y \\ 1 \end{bmatrix}$$
    여기서 $\mathbf{H} = \mathbf{K}[\mathbf{r}_1 \quad \mathbf{r}_2 \quad \mathbf{t}]$ 는 $3 \times 3$ 호모그래피 행렬.
  - **내부 파라미터의 선형 해(Closed-form Solution)**:
    - 회전 벡터의 직교 조건 $\mathbf{r}_1^\top \mathbf{r}_2 = 0$, $\|\mathbf{r}_1\| = \|\mathbf{r}_2\|$에서 $\mathbf{K}$에 대한 2개의 제약 조건 도출.
    - $\mathbf{B} = \mathbf{K}^{-\top}\mathbf{K}^{-1}$ (대칭 양정치 행렬, 6개 미지수)을 정의하고, 각 이미지마다 2개 방정식을 얻음.
    - 최소 3장의 이미지(일반적인 5-파라미터 모델)로 $\mathbf{B}$의 선형 시스템을 풀고, Cholesky 분해로 $\mathbf{K}$ 복원.
  - **외부 파라미터 계산**: $\mathbf{K}$를 알면 호모그래피의 각 열에서 $\mathbf{r}_1, \mathbf{r}_2, \mathbf{t}$를 직접 계산. $\mathbf{r}_3 = \mathbf{r}_1 \times \mathbf{r}_2$.
  - **비선형 정제(MLE)**: Levenberg-Marquardt 최적화로 reprojection error 최소화. 렌즈 왜곡(radial distortion) 파라미터도 이 단계에서 동시 추정.
    $$\min \sum_{i=1}^{n}\sum_{j=1}^{m} \|\mathbf{m}_{ij} - \hat{\mathbf{m}}(\mathbf{K}, k_1, k_2, \mathbf{R}_i, \mathbf{t}_i, \mathbf{M}_j)\|^2$$
  - **실용성**: 프린터로 출력한 체커보드만 있으면 됨. 3D 정밀 장비(calibration rig) 불필요.

- **이 가이드에서의 위치**:
  - **Ch.3 (Calibration — Deep Dive)**: 이 챕터의 핵심 논문. 호모그래피 기반 캘리브레이션을 단계별로 유도하며, OpenCV `calibrateCamera()`의 이론적 근거로 상세 설명.
  - **Ch.2 (Sensor Modeling)**: 카메라 핀홀 모델, 왜곡 모델의 수학적 기반으로 참조.

- **후속 영향**:
  - **OpenCV `calibrateCamera()`**: Zhang's method의 사실상 표준 구현. 로보틱스/CV 분야에서 가장 널리 사용.
  - **Kalibr (Furgale et al.)**: 카메라-IMU 공동 캘리브레이션 도구. Zhang's method를 기초로 확장.
  - **Targetless Calibration**: 체커보드 없이 자연 장면의 특징점으로 캘리브레이션하는 연구들.
  - **Multi-camera / Multi-sensor Calibration**: 다중 센서 시스템의 외부 파라미터 캘리브레이션으로 확장.
  - **Self-calibration (auto-calibration)**: 어떤 패턴도 없이 이미지 시퀀스만으로 캘리브레이션.

---

## 5. Distinctive Image Features from Scale-Invariant Keypoints

- **저자/연도**: David G. Lowe, 2004
- **발표**: International Journal of Computer Vision, Vol. 60, No. 2, pp. 91–110
- **핵심 기여**: 스케일, 회전, 조명 변화에 불변하고, 시점 변화에도 강건한 **SIFT(Scale-Invariant Feature Transform)** 특징 추출 및 기술 알고리즘을 제안. 20년간 특징점 매칭의 사실상 표준.

- **주요 내용**:
  - **4단계 파이프라인**:

  - **Stage 1 — Scale-Space Extrema Detection (DoG)**:
    - 이미지를 다양한 스케일의 가우시안으로 블러링:
      $$L(x, y, \sigma) = G(x, y, \sigma) * I(x, y)$$
    - 인접 스케일 간 차이(Difference of Gaussians)를 계산하여 LoG(Laplacian of Gaussian)를 근사:
      $$D(x, y, \sigma) = L(x, y, k_i\sigma) - L(x, y, k_j\sigma)$$
    - DoG 이미지에서 공간적 (8 이웃) + 스케일 (상하 각 9) = 26개 이웃과 비교하여 극값(extrema) 탐색.

  - **Stage 2 — Keypoint Localization**:
    - 스케일-공간에서 Taylor 전개를 이용한 서브픽셀/서브스케일 위치 정밀화:
      $$D(\mathbf{x}) = D + \frac{\partial D}{\partial \mathbf{x}}^\top \mathbf{x} + \frac{1}{2} \mathbf{x}^\top \frac{\partial^2 D}{\partial \mathbf{x}^2} \mathbf{x}$$
    - 저대비(low-contrast) 키포인트 제거: $|D(\hat{\mathbf{x}})| < 0.03$ 이면 제거.
    - 엣지 응답 제거: Hessian 행렬의 고유값 비를 이용 (Harris corner 기준과 유사).

  - **Stage 3 — Orientation Assignment**:
    - 키포인트 주변의 그래디언트 크기와 방향을 계산.
    - 36-bin 방향 히스토그램 생성 (가우시안 가중, $\sigma = 1.5 \times$ 키포인트 스케일).
    - 최대 피크의 80% 이상인 피크가 있으면 해당 방향에도 별도 키포인트 생성 → 회전 불변성.

  - **Stage 4 — Keypoint Descriptor**:
    - 키포인트 주변 $16 \times 16$ 영역을 $4 \times 4$ 블록 16개로 분할.
    - 각 블록에서 8-bin 방향 히스토그램 생성 → $4 \times 4 \times 8 = 128$차원 기술자.
    - 정규화(L2 norm) 후 0.2 이상 값을 클리핑하고 재정규화 → 비선형 조명 변화에 강건.

  - **매칭**: Nearest-neighbor distance ratio (NNDR) 테스트 — 최근접/차근접 거리 비가 0.8 이하일 때만 매칭 수용.

- **이 가이드에서의 위치**:
  - **Ch.4 (Feature Matching & Geometric Verification)**: 특징점 기반 매칭의 원형(archetype)으로서 SIFT를 상세 설명. DoG → descriptor까지의 전체 파이프라인을 다루고, 후속 방법(SURF, ORB, SuperPoint)과의 비교 기준점.
  - **Ch.7 (Place Recognition & Loop Closure)**: SIFT 기술자가 BoW/VLAD 기반 검색의 입력으로 사용되는 맥락 설명.

- **후속 영향**:
  - **SURF (Bay et al., 2006)**: Integral image와 Hessian determinant로 속도 개선. 64차원 기술자.
  - **ORB (Rublee et al., 2011)**: FAST 검출 + BRIEF 기술, 특허 없고 실시간 가능. ORB-SLAM의 핵심.
  - **SuperPoint (DeTone et al., 2018)**: 딥러닝 기반 키포인트 검출 + 기술. self-supervised 학습.
  - **SuperGlue (Sarlin et al., 2020)**: Graph neural network 기반 특징 매칭. attention으로 대응 추론.
  - **LoFTR (Sun et al., 2021)**: Detector-free, Transformer 기반 dense matching.
  - **RoMa (Edstedt et al., 2024)**: 최신 robust dense matching.
  - SIFT 특허는 2020년 만료되어 현재 자유롭게 사용 가능.

---

## 6. Video Google: A Text Retrieval Approach to Object Matching in Videos

- **저자/연도**: Josef Sivic, Andrew Zisserman, 2003
- **발표**: Proceedings of the IEEE International Conference on Computer Vision (ICCV), Vol. 2, pp. 1470–1477
- **핵심 기여**: 텍스트 검색(text retrieval)의 아이디어를 시각적 객체 검색에 적용하여, **visual words**와 **inverted file index** 기반의 대규모 시각 검색 프레임워크를 최초로 제시. Bag of Visual Words (BoVW) 모델의 기원.

- **주요 내용**:
  - **핵심 아이디어**: 이미지의 로컬 기술자(affine covariant regions)를 벡터 양자화하여 "시각 단어(visual word)"로 변환하고, 텍스트 문서처럼 처리하여 검색.
  - **파이프라인**:
    1. **특징 추출**: 시점 불변 영역 기술자(affine covariant region descriptors) 추출.
    2. **시각 어휘 구축(Visual Vocabulary)**: 기술자를 k-means 클러스터링으로 양자화 → 각 클러스터 중심이 하나의 visual word.
    3. **역색인(Inverted File) 구축**: 각 visual word가 등장하는 이미지/프레임 목록을 역색인 구조로 저장.
    4. **TF-IDF 가중**: 텍스트 검색의 term frequency–inverse document frequency를 적용:
       - **TF (Term Frequency)**: 해당 이미지에서 특정 visual word가 차지하는 비율
       - **IDF (Inverse Document Frequency)**: 전체 데이터베이스에서 해당 word가 희귀할수록 높은 가중치
       - 문서 벡터: $\mathbf{v}_d = (w_1, w_2, \ldots, w_V)$, $w_i = \text{tf}_{i,d} \cdot \log(N/n_i)$
    5. **유사도 기반 랭킹**: 쿼리 이미지와 데이터베이스 이미지 간 벡터 유사도(코사인 유사도 등)로 랭킹.
  - **비디오 활용**: 비디오의 시간적 연속성을 활용하여 불안정한 영역 기술자를 추적(tracking)하고 제거 → 노이즈 감소.
  - **응용**: 두 편의 장편 영화에서 사용자가 지정한 객체의 모든 등장을 즉시 검색하는 시스템 시연.

- **이 가이드에서의 위치**:
  - **Ch.7 (Place Recognition & Loop Closure)**: 이 챕터의 기초 논문. BoW 기반 장소 인식(DBoW2 등)의 이론적 뿌리로서 상세히 다룸.
  - **Ch.4 (Feature Matching)**: Visual words 개념의 기원으로 참조. 기술자 양자화와 매칭 효율성 논의.

- **후속 영향**:
  - **DBoW / DBoW2 (Galvez-Lopez & Tardos)**: BoVW를 SLAM 루프 클로저에 적용한 실용적 구현. ORB-SLAM에서 사용.
  - **VLAD (Jegou et al., 2010)**: BoW를 확장하여 잔차(residual)를 집계. 더 정밀한 이미지 표현.
  - **Fisher Vectors (Perronnin et al.)**: GMM 기반의 더 풍부한 집계 방식.
  - **NetVLAD (Arandjelovic et al., 2016)**: VLAD를 CNN으로 end-to-end 학습. 장소 인식의 딥러닝 전환점.
  - **AnyLoc (Keetha et al., 2023)**: Foundation model (DINOv2) 기반의 범용 장소 인식.
  - **Bag of Words → Vector of Locally Aggregated Descriptors → Deep retrieval**: 시각 검색의 전체 진화 계보가 이 논문에서 출발.

---

## 논문 간 연결 관계 (Cross-Paper Connections)

```
Kalman (1960)                    Fischler & Bolles (1981)
    │                                   │
    ├─ 상태 추정 이론의 기초              ├─ 아웃라이어 제거의 표준
    │  → EKF/UKF/ESKF                   │  → 모든 기하학적 추정에 RANSAC 적용
    │  → VIO/LIO 시스템                  │
    │                                   │
    ▼                                   ▼
  Ch.5–6                          Ch.3–4, Ch.6
  (Filtering, Odometry)          (Calibration, Matching, VO)

Besl & McKay (1992)              Zhang (2000)
    │                                   │
    ├─ 3D 정합의 원형                    ├─ 카메라 캘리브레이션의 표준
    │  → LiDAR odometry               │  → OpenCV calibrateCamera
    │  → Scan matching                  │  → 멀티센서 캘리브레이션 기초
    │                                   │
    ▼                                   ▼
  Ch.6, Ch.8                      Ch.3
  (LiDAR Odom, Mapping)          (Calibration)

Lowe (2004)                      Sivic & Zisserman (2003)
    │                                   │
    ├─ 특징 추출/기술의 표준              ├─ 시각 검색의 기초
    │  → SURF, ORB, SuperPoint          │  → BoW, VLAD, NetVLAD
    │                                   │
    ▼                                   ▼
  Ch.4                            Ch.7
  (Feature Matching)              (Place Recognition)

연결:
  SIFT(Lowe) ──features──▶ BoW(Sivic) ──retrieval──▶ Loop Closure
  RANSAC(Fischler) ──robust──▶ ICP(Besl), Calibration(Zhang), VO
  Kalman ──filtering──▶ IMU fusion, VIO/LIO 전체
```
