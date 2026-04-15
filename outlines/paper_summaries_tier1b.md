# Tier 1b 논문 요약 — 센서 퓨전 가이드 핵심 논문

> **대상 논문**: Tier 1 목록 중 #7–#12 (VO, MSCKF, LOAM, NetVLAD, IMU Preintegration, VINS-Mono)
> **작성일**: 2026-04-15
> **목적**: 각 논문의 핵심 기여와 방법론을 정리하고, 가이드 내 위치를 매핑

---

## 1. Visual Odometry (Nistér et al., 2004)

- **저자/연도**: David Nistér, Oleg Naroditsky, James Bergen / 2004
- **발표**: IEEE Conference on Computer Vision and Pattern Recognition (CVPR 2004), pp. 652–659
- **핵심 기여**: "Visual Odometry"라는 용어를 처음 정의하고, 스테레오 및 단안 카메라로 실시간 자기 운동 추정(ego-motion estimation) 시스템을 제시한 원조 논문.

- **주요 내용**:
  - **스테레오 접근**: 좌우 카메라에서 3D 점을 삼각측량한 뒤, **3-point 알고리즘**으로 프레임 간 강체 변환(rigid body transformation)을 추정. 알려진 베이스라인(baseline)으로부터 절대 스케일 복원 가능.
  - **단안 접근**: 연속 프레임 간 2D-2D 대응점으로부터 **5-point 알고리즘**을 사용해 Essential Matrix를 추정. 스케일 모호성(scale ambiguity)이 존재하며, 연속 삼각측량으로 부분적 해결.
  - **특징점 추출 및 추적**: Harris corner detector로 특징점 검출, 11×11 윈도우의 normalized correlation으로 매칭. 부분 픽셀(subpixel) 정밀도 없이 작동하여 속도 최적화.
  - **Preemptive RANSAC**: Nistér가 2003년에 제안한 빠른 RANSAC 변종으로, 가설 평가를 선제적으로 중단하여 실시간 처리 달성. 점수가 낮은 가설을 빠르게 탈락시키는 토너먼트 방식.
  - **실시간 성능**: Pentium III 1GHz에서 360×240 해상도, 약 **13Hz** 처리 달성. 50° FOV, 28cm 베이스라인 스테레오 카메라 사용.

- **이 가이드에서의 위치**: **Ch.6 (Visual Odometry & VIO)** 도입부에서 VO의 기원으로 소개. 6.1절 Feature-based VO의 역사적 출발점으로, 스테레오/단안 VO 파이프라인의 기본 구조(특징점 검출 → 매칭 → RANSAC → 모션 추정)를 이 논문 기반으로 설명.

- **후속 영향**:
  - Scaramuzza et al. (2009) 1-point RANSAC VO — 단안 VO 효율화
  - ORB-SLAM 시리즈 — feature-based VO/SLAM의 사실상 표준
  - DSO (Engel et al., 2018) — direct 방법으로의 패러다임 전환
  - Nistér 본인의 확장판: "Visual odometry for ground vehicle applications" (Journal of Field Robotics, 2006)

---

## 2. A Multi-State Constraint Kalman Filter for Vision-aided Inertial Navigation (Mourikis & Roumeliotis, 2007) — MSCKF

- **저자/연도**: Anastasios I. Mourikis, Stergios I. Roumeliotis / 2007
- **발표**: IEEE International Conference on Robotics and Automation (ICRA 2007), pp. 3565–3572
- **핵심 기여**: 시각 특징점의 3D 위치를 상태 벡터에 포함하지 않으면서도 최적(선형화 오차까지)인 시각-관성 항법 필터를 제안. EKF-SLAM 대비 계산 복잡도를 $O(N^3)$에서 $O(N)$으로 낮춤.

- **주요 내용**:
  - **상태 벡터 구조**: IMU 오차 상태(error-state) + 슬라이딩 윈도우 내 $N$개 카메라 포즈로 구성. 랜드마크 위치를 상태에 포함하지 않는 것이 핵심 차별점.
    $$\mathbf{x} = [\mathbf{x}_{IMU}^T, \mathbf{x}_{C_1}^T, \ldots, \mathbf{x}_{C_N}^T]^T$$
    여기서 $\mathbf{x}_{IMU} = [{}^I_G\bar{q}^T, \mathbf{b}_g^T, {}^G\mathbf{v}_I^T, \mathbf{b}_a^T, {}^G\mathbf{p}_I^T]^T$
  - **다중 상태 구속 조건(Multi-State Constraint)**: 하나의 정적 특징점이 여러 카메라 포즈에서 관측될 때, 관측 방정식은 상태 $\mathbf{x}$와 특징점 위치 $\mathbf{p}_f$ 모두에 의존:
    $$\mathbf{z}_j^{(i)} = \mathbf{h}(\mathbf{x}, \mathbf{p}_f) + \mathbf{n}$$
  - **Null-space 투영**: 관측 야코비안을 특징점 위치에 대한 부분 $\mathbf{H}_f$와 상태에 대한 부분 $\mathbf{H}_X$로 분리한 뒤, $\mathbf{H}_f$의 left null space $\mathbf{A}$를 곱해 특징점 위치를 소거:
    $$\mathbf{r}_o = \mathbf{A}^T\mathbf{H}_X \delta\mathbf{x} + \mathbf{A}^T\mathbf{n}$$
    이로써 특징점 없이도 카메라 포즈 간 기하학적 구속을 EKF 업데이트에 반영.
  - **QR 분해**: $\mathbf{H}_f$의 null space를 효율적으로 계산하기 위해 QR 분해 사용.
  - **계산 복잡도**: 특징점 수 $M$에 대해 $O(M)$ — EKF-SLAM의 $O(M^3)$과 대비되는 핵심 장점.

- **이 가이드에서의 위치**: **Ch.6.3 (Tightly-coupled VIO)** 에서 필터 기반 VIO의 대표로 상세 분석. **Ch.4.2 (Kalman Filter 계열)** 에서 Error-State KF 예시로 참조. "왜 EKF 기반이 아직 살아있는가"의 핵심 근거 — 최적화 기반 대비 계산 효율과 일관성(consistency) 트레이드오프.

- **후속 영향**:
  - Li & Mourikis (2013) "High-Precision, Consistent EKF-based VIO" — MSCKF 2.0, 관측 가능성 분석 강화, FEJ (First-Estimate Jacobian) 도입
  - S-MSCKF — 스테레오 확장
  - **OpenVINS** (Geneva et al., 2020) — MSCKF의 가장 완성된 오픈소스 구현, 모듈형 설계
  - ROVIO — 직접법(direct method) 기반 EKF VIO
  - VINS-Mono/Fusion — 최적화 기반으로 갔지만 MSCKF의 슬라이딩 윈도우 아이디어 계승

---

## 3. LOAM: Lidar Odometry and Mapping in Real-time (Zhang & Singh, 2014)

- **저자/연도**: Ji Zhang, Sanjiv Singh / 2014
- **발표**: Robotics: Science and Systems (RSS 2014)
- **핵심 기여**: 고정밀 거리 센서나 IMU 없이도 저드리프트 실시간 LiDAR 오도메트리를 달성하는 시스템. KITTI 오도메트리 벤치마크에서 오랫동안 상위권을 유지한 LiDAR SLAM의 기준점.

- **주요 내용**:
  - **특징점 추출**: 포인트 클라우드의 국소 곡률(curvature)을 계산하여 두 종류의 특징점을 추출:
    - **Edge feature** (높은 곡률): 모서리, 날카로운 경계
    - **Planar feature** (낮은 곡률): 평면 표면
  - **2단계 아키텍처**: 계산 부하를 분리하여 실시간성 확보:
    - **Odometry 모듈 (~10Hz)**: scan-to-scan 매칭으로 빠른 모션 추정. edge 점은 **point-to-edge 거리**(점에서 직선까지의 수직 거리), planar 점은 **point-to-plane 거리**(점에서 평면까지의 수직 거리)를 최소화.
    - **Mapping 모듈 (~1Hz)**: scan-to-map 정합으로 정밀한 포즈 보정 및 맵 업데이트. 누적된 맵에 대해 새 스캔을 정합하여 드리프트 보정.
  - **Point-to-edge 거리**: edge 점 $\mathbf{p}$와 맵의 두 edge 점 $\mathbf{a}, \mathbf{b}$에 대해:
    $$d_e = \frac{|(\mathbf{p}-\mathbf{a}) \times (\mathbf{p}-\mathbf{b})|}{|\mathbf{a}-\mathbf{b}|}$$
  - **Point-to-plane 거리**: planar 점 $\mathbf{p}$와 맵의 세 planar 점으로 정의되는 평면에 대한 거리.
  - **모션 왜곡 보정(Motion Distortion Compensation)**: 회전형 LiDAR는 한 스캔 동안 로봇이 움직이므로 점들이 왜곡됨. IMU가 있으면 IMU 데이터로, 없으면 등속 운동 모델(constant velocity model)로 중간 포즈를 보간하여 보정.
  - **KITTI 성능**: 발표 당시 KITTI 오도메트리 벤치마크에서 LiDAR 전용 방법 중 최고 수준의 정확도 달성. 이후에도 오랫동안 상위권 유지.

- **이 가이드에서의 위치**: **Ch.7.2 (Feature-based LiDAR Odometry)** 의 핵심 논문. LOAM의 edge/planar 특징점 추출과 2단계 아키텍처를 상세히 분석. **Ch.7.1 (Point Cloud Registration)** 에서 point-to-edge/plane 거리 메트릭의 원천으로 참조. "왜 LOAM 계열이 오래 살아남았는가" 논의의 중심.

- **후속 영향**:
  - **A-LOAM** — LOAM의 Ceres 기반 리팩토링 (HKUST)
  - **LeGO-LOAM** (Shan & Englot, 2018) — ground segmentation 추가, 경량화, 임베디드 시스템 대응
  - **LIO-SAM** (Shan et al., 2020) — LeGO-LOAM에 IMU preintegration + GPS factor를 factor graph로 통합
  - **FAST-LIO / FAST-LIO2** (Xu et al., 2021/2022) — LOAM의 특징점 추출 대신 raw 점을 iterated EKF + ikd-tree로 직접 처리, 솔리드 스테이트 LiDAR 지원
  - **CT-ICP** (Dellenbach et al., 2022) — continuous-time LiDAR 오도메트리
  - **Point-LIO** (He et al., 2023) — 점 단위 처리로 고속 모션 대응

---

## 4. NetVLAD: CNN Architecture for Weakly Supervised Place Recognition (Arandjelovic et al., 2016)

- **저자/연도**: Relja Arandjelovic, Petr Gronat, Akihiko Torii, Tomas Pajdla, Josef Sivic / 2016
- **발표**: IEEE Conference on Computer Vision and Pattern Recognition (CVPR 2016)
- **핵심 기여**: 전통적인 VLAD 표현을 미분 가능한(differentiable) CNN 레이어로 재구성하여, 장소 인식(place recognition)을 위한 end-to-end 학습을 가능하게 한 논문. 학습 기반 Visual Place Recognition의 기준점.

- **주요 내용**:
  - **전통 VLAD vs NetVLAD**: VLAD는 로컬 디스크립터를 $K$개 클러스터에 hard assignment하여 잔차(residual)를 누적하지만, NetVLAD는 **soft assignment**로 이를 미분 가능하게 변환.
  - **Soft assignment**: softmax 기반의 연속적 가중치:
    $$\bar{a}_k(\mathbf{x}_i) = \frac{e^{-\alpha\|\mathbf{x}_i - \mathbf{c}_k\|^2}}{\sum_{k'} e^{-\alpha\|\mathbf{x}_i - \mathbf{c}_{k'}\|^2}}$$
    이를 통해 역전파로 클러스터 중심 $\mathbf{c}_k$와 디스크립터를 동시에 학습.
  - **NetVLAD 레이어 출력**: 각 클러스터 $k$에 대해 soft-weighted 잔차 합산:
    $$\mathbf{V}(j,k) = \sum_{i=1}^{N} \bar{a}_k(\mathbf{x}_i)(x_i(j) - c_k(j))$$
    결과적으로 $D \times K$ 차원의 행렬을 L2 정규화 후 벡터로 평탄화.
  - **CNN 백본**: VGG-16의 마지막 분류 레이어를 제거하고 NetVLAD 레이어를 연결. 이 레이어는 임의의 CNN 아키텍처에 플러그인 가능.
  - **약한 지도 학습(Weakly Supervised Training)**: Google Street View Time Machine 데이터를 활용. 동일 GPS 위치의 다른 시간대 이미지를 positive pair, 다른 위치를 negative로 사용. 정밀한 수동 어노테이션 불필요.
  - **Triplet ranking loss**: $\mathcal{L} = \max(0, m + d(\mathbf{q}, \mathbf{p}^+) - d(\mathbf{q}, \mathbf{p}^-))$ 형태의 랭킹 손실. Hard negative mining으로 학습 효율 향상.
  - **PCA 화이트닝**: 최종 디스크립터에 PCA 화이트닝을 적용하여 성능 추가 향상.
  - **벤치마크 성능**: Pitts250k Recall@1 약 84.3%, Tokyo 24/7 Recall@1 약 72.7%, Pitts30k Recall@1 약 86.3%로, 당시 hand-crafted 및 비학습 CNN 대비 대폭 향상.

- **이 가이드에서의 위치**: **Ch.9.2 (Visual Place Recognition)** 에서 학습 기반 VPR의 기준점으로 상세 분석. CNN 기반 retrieval의 시작점이자 후속 연구(CosPlace, EigenPlaces, AnyLoc)의 비교 기준. **Ch.10 (Loop Closure)** 에서 loop closure detection 백엔드로 활용되는 사례 소개.

- **후속 영향**:
  - **AP-GeM** (Revaud et al., 2019) — generalized mean pooling 기반 대안
  - **CosPlace** (Berton et al., 2022) — 분류 기반 학습으로 triplet loss 대체
  - **EigenPlaces** (Berton et al., 2023) — PCA 기반 차원 축소 통합
  - **AnyLoc** (Keetha et al., 2023) — DINOv2 foundation model 기반 VPR, NetVLAD 구조를 foundation model feature에 적용
  - **Patch-NetVLAD** (Hausler et al., 2021) — 패치 레벨 매칭으로 공간 정보 보존
  - **MixVPR** (Ali-bey et al., 2023) — feature mixing 기반 대안
  - 3D로의 확장: **PointNetVLAD** (Uy & Lee, 2018) — NetVLAD 아이디어를 3D 포인트 클라우드 retrieval에 적용

---

## 5. On-Manifold Preintegration for Real-Time Visual-Inertial Odometry (Forster et al., 2017)

- **저자/연도**: Christian Forster, Luca Carlone, Frank Dellaert, Davide Scaramuzza / 2017
- **발표**: IEEE Transactions on Robotics (TRO), Vol. 33, No. 1, pp. 1–21
- **핵심 기여**: IMU 측정값의 preintegration을 SO(3) 매니폴드 위에서 수행하는 이론을 제시. 바이어스 추정치가 변할 때 전체 재적분 없이 1차 야코비안으로 보정 가능하게 하여, factor graph 기반 VIO의 실시간 처리를 가능하게 함.

- **주요 내용**:
  - **문제**: 두 키프레임 $i, j$ 사이의 IMU 측정값을 적분하여 상대 모션 구속(relative motion constraint)을 만들고 싶다. 그런데 naive 적분은 글로벌 좌표계 기준이라, 키프레임 $i$의 포즈가 최적화로 바뀌면 **전체 재적분**이 필요. 바이어스 추정치가 바뀔 때도 마찬가지.
  - **Preintegrated 측정**: 글로벌 프레임 대신 키프레임 $i$의 body frame 기준으로 상대적 변화량을 누적:
    $$\Delta\mathbf{R}_{ij} = \prod_{k=i}^{j-1} \text{Exp}((\boldsymbol{\omega}_k - \mathbf{b}_g^i)\Delta t)$$
    $$\Delta\mathbf{v}_{ij} = \sum_{k=i}^{j-1} \Delta\mathbf{R}_{ik}(\mathbf{a}_k - \mathbf{b}_a^i)\Delta t$$
    $$\Delta\mathbf{p}_{ij} = \sum_{k=i}^{j-1} \Delta\mathbf{v}_{ik}\Delta t + \frac{1}{2}\Delta\mathbf{R}_{ik}(\mathbf{a}_k - \mathbf{b}_a^i)\Delta t^2$$
    이 값들은 키프레임 $i$의 포즈와 무관하므로, 포즈가 바뀌어도 재적분 불필요.
  - **On-manifold**: 회전 $\Delta\mathbf{R}_{ij} \in SO(3)$을 직접 매니폴드 위에서 처리. 오일러 각이나 쿼터니언 정규화 문제 회피.
  - **1차 바이어스 보정**: 바이어스 추정치가 $\delta\mathbf{b}$만큼 변할 때, preintegrated 측정을 완전히 재계산하지 않고 야코비안으로 1차 보정:
    $$\Delta\hat{\mathbf{R}}_{ij}(\mathbf{b}_g + \delta\mathbf{b}_g) \approx \Delta\hat{\mathbf{R}}_{ij}(\mathbf{b}_g) \cdot \text{Exp}\left(\frac{\partial \Delta\bar{\mathbf{R}}_{ij}}{\partial \mathbf{b}_g}\delta\mathbf{b}_g\right)$$
    이 야코비안은 preintegration 과정에서 재귀적으로 누적 계산.
  - **공분산 전파(Covariance Propagation)**: 자이로/가속도계 노이즈가 preintegration을 통해 전파되며, 결과 공분산이 factor graph에서 정보 행렬(information matrix)로 사용.
  - **Factor Graph 통합**: preintegrated IMU 측정이 연속 키프레임 간 **IMU factor**로 삽입. 시각 측정은 **structureless vision factor**(특징점 3D 위치를 명시적으로 추정하지 않는 형태)로 결합. GTSAM의 incremental smoothing (iSAM2) 위에서 실시간 동작.
  - **EuRoC MAV 결과**: preintegration + 바이어스 야코비안이 naive 재적분과 동등한 정확도를 보이면서 계산 비용을 크게 절감.

- **이 가이드에서의 위치**: **Ch.4.6 (IMU Preintegration)** 의 핵심 논문. preintegration 이론의 유도 과정을 단계별로 설명하는 데 사용. **Ch.4.5 (Factor Graph & Optimization)** 에서 factor graph에 IMU factor를 삽입하는 구체적 예시로 활용. Ch.6 (VIO)와 Ch.7 (LIO)의 거의 모든 시스템이 이 논문의 preintegration을 사용하므로, 반복 참조되는 기반 이론.

- **후속 영향**:
  - **GTSAM** 라이브러리에 `PreintegratedImuMeasurements` 클래스로 공식 구현
  - **VINS-Mono** (Qin et al., 2018) — 이 preintegration 이론을 tightly-coupled VIO에 적용
  - **LIO-SAM** (Shan et al., 2020) — LiDAR-inertial 시스템에서 IMU preintegration factor 활용
  - **ORB-SLAM3** (Campos et al., 2021) — visual-inertial mode에서 preintegration 사용
  - Eckenhoff et al. (2019) — continuous-time preintegration으로 확장
  - Brossard et al. (2021) — Lie group 기반 preintegration (Invariant EKF와의 결합)

---

## 6. VINS-Mono: A Robust and Versatile Monocular Visual-Inertial State Estimator (Qin et al., 2018)

- **저자/연도**: Tong Qin, Peiliang Li, Shaojie Shen / 2018
- **발표**: IEEE Transactions on Robotics (TRO), Vol. 34, No. 4, pp. 1004–1020
- **핵심 기여**: 단안 카메라 + 저가 IMU만으로 강건한 6-DoF 상태 추정을 달성하는 완전한 VIO 시스템. 초기화, 오도메트리, 루프 클로저, 맵 재사용까지 전체 파이프라인을 하나의 시스템으로 통합.

- **주요 내용**:
  - **시스템 아키텍처**: 크게 세 모듈로 구성:
    1. **프론트엔드**: KLT 기반 특징점 추적 + IMU preintegration
    2. **백엔드**: 비선형 최적화 기반 tightly-coupled VIO
    3. **루프 클로저**: DBoW2 기반 장소 인식 + 재위치추정(relocalization)
  - **강건한 초기화(Robust Initialization)**:
    - 단안 VIO의 난제인 스케일 모호성 해결을 위한 다단계 초기화:
    - (1) 순수 비전으로 SfM (Structure from Motion) 수행 → 초기 포즈 + 3D 점 획득
    - (2) SfM 결과를 IMU preintegration과 정렬(alignment): 자이로 바이어스 추정, 중력 방향 정제, **메트릭 스케일** 복원, 속도 추정
    - 이 loosely-coupled 초기화가 수렴하면 tightly-coupled 최적화로 전환
  - **Tightly-coupled 비선형 최적화**: 슬라이딩 윈도우 내에서 다음 잔차(residual)들을 동시 최적화:
    - **IMU 잔차**: Forster et al. (2017)의 on-manifold preintegration 사용
    - **시각 잔차**: 특징점 재투영 오차(reprojection error)
    - **마지날라이제이션 사전(marginalization prior)**: Schur complement로 슬라이딩 윈도우 밖으로 나간 변수를 사전 분포(prior)로 변환하여 정보 보존
  - **슬라이딩 윈도우 관리**: 윈도우 크기를 고정하되, 키프레임 여부에 따라 두 가지 마지날라이제이션 전략 적용:
    - 최신 프레임이 키프레임이면: 가장 오래된 프레임을 마지날라이즈
    - 키프레임이 아니면: 직전 프레임을 마지날라이즈 (시각 정보만 버리고 IMU 정보 보존)
  - **루프 클로저 & 재위치추정**: DBoW2로 루프 후보 검출 → 특징점 매칭 → 상대 포즈 추정 → tightly-coupled 최적화에 반영
  - **4-DoF 포즈 그래프 최적화**: 글로벌 일관성을 위해 루프 클로저 후 포즈 그래프 최적화 수행. 6-DoF가 아닌 **4-DoF**(yaw + 3D translation)인 이유: IMU가 중력 방향(roll, pitch)을 관측 가능하게 하므로, 이 두 각도는 이미 정확. 드리프트는 yaw와 위치에서만 발생.
  - **맵 재사용**: 최적화된 포즈와 특징점을 저장/로드하여 다음 세션에서 기존 맵 기반 위치추정 가능.
  - **실험 결과**: EuRoC MAV 데이터셋에서 당시 최고 수준의 정확도. MAV 플랫폼에서 실시간 자율비행 클로즈드루프 검증. iOS 모바일 디바이스에서도 실시간 구동 시연.

- **이 가이드에서의 위치**: **Ch.6.3 (Tightly-coupled VIO)** 에서 최적화 기반 VIO의 대표 시스템으로 상세 아키텍처 분석. **Ch.4.7 (Marginalization & Sliding Window)** 에서 Schur complement 기반 마지날라이제이션의 실전 예시로 활용. **Ch.10 (Loop Closure)** 에서 VIO + 루프 클로저 통합 설계의 사례로 참조. MSCKF(필터 기반)와의 비교를 통해 "Filter vs Optimization" 논의(Ch.6.4)의 핵심 사례.

- **후속 영향**:
  - **VINS-Fusion** (Qin et al., 2019) — 스테레오/스테레오+IMU 지원, GPS 융합 추가
  - **ORB-SLAM3** (Campos et al., 2021) — VINS-Mono와 유사한 VIO 파이프라인을 ORB 기반 SLAM에 통합
  - **Basalt** (Usenko et al., 2020) — 유사한 tightly-coupled VIO이지만 다른 최적화 전략
  - 수많은 산업 응용: 드론 자율비행, AR/VR 트래킹, 모바일 로봇 내비게이션의 기반 시스템으로 널리 채택
  - 오픈소스 생태계: GitHub에서 가장 많이 포크/참조된 VIO 시스템 중 하나

---

## 논문 간 관계도

```
Nistér (2004) Visual Odometry
    │  [VO 파이프라인의 기본 구조 정립]
    ├──→ Mourikis (2007) MSCKF
    │        │  [VO + IMU를 EKF로 결합, 필터 기반 VIO 시작]
    │        └──→ OpenVINS (2020)
    │
    ├──→ Forster (2017) IMU Preintegration
    │        │  [IMU 적분의 이론적 기반, factor graph에 IMU를 넣는 방법]
    │        └──→ Qin (2018) VINS-Mono
    │                 │  [Preintegration + 비선형 최적화 = 완전한 VIO 시스템]
    │                 └──→ VINS-Fusion, ORB-SLAM3
    │
    Zhang (2014) LOAM
    │  [LiDAR odometry의 기준, edge/planar feature 추출]
    ├──→ LeGO-LOAM → LIO-SAM (+ IMU Preintegration)
    └──→ FAST-LIO / FAST-LIO2
    
    Arandjelovic (2016) NetVLAD
    │  [학습 기반 장소 인식의 시작]
    ├──→ PointNetVLAD (3D 확장)
    └──→ AnyLoc (Foundation Model 기반)
```
