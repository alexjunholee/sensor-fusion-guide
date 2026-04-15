# Sensor Fusion Guide — 아웃라인 (Stage 1)

> **주제**: 멀티센서 퓨전 심화 가이드
> **대상**: 로보틱스 입문자 (선형대수, 확률론 기초 가정)
> **톤**: 직관 → 수식 → 코드/예제. robotics-practice보다 깊고 실전적.
> **데모**: robotics-practice와 겹치지 않는 선에서 추후 검토
> **미출판 연구 제외**: Rosetta/‖Ω‖ 등 미발표 내용은 다루지 않음

---

## 가이드 전체 관통 테마: Classical → Learning 기술 계보

이 가이드는 각 주제마다 **전통 방법 → 딥러닝으로 가능해진 방법**의 흐름을 명시적으로 보여준다.
독자가 "왜 이 전통 방법이 중요했는가 → 딥러닝이 뭘 바꿨는가 → 아직 전통이 필요한 부분은 어디인가"를 이해할 수 있도록 구성.

| 영역 | Classical | Learning-based | 현재 주류 |
|------|-----------|---------------|----------|
| Feature matching | SIFT/ORB + BF/FLANN | SuperPoint+SuperGlue → LoFTR → RoMa | Hybrid |
| Visual odometry | Feature-based (ORB) / Direct (DSO) | DROID-SLAM, DPV-SLAM | 전통 우세, 학습 추격 |
| LiDAR odometry | ICP/LOAM | DeepLO 계열 | 전통 압도적 우세 |
| Place recognition | BoW/VLAD | NetVLAD → AnyLoc | 학습 우세 |
| Depth estimation | Stereo matching | Mono depth (Depth Anything) | 학습 우세 (mono) |
| Calibration | Target-based | Targetless + learning | 전통 우세 |
| Map representation | Occupancy/TSDF | NeRF / 3DGS | 공존 |

---

## Ch.1 — Introduction: 왜 센서 퓨전인가

- 단일 센서의 한계 (카메라: 조명, LiDAR: 텍스처, IMU: 드리프트)
- 센서 퓨전의 분류: complementary / competitive / cooperative
- Loosely coupled vs Tightly coupled vs Ultra-tightly coupled
- Classical vs Learning-based: 센서 퓨전에서 딥러닝이 바꾼 것과 바꾸지 못한 것
- 이 가이드의 범위와 robotics-practice와의 관계

---

## Ch.2 — 센서 모델링 (Sensor Modeling)

> robotics-practice Ch.2는 센서 소개 수준. 여기서는 **노이즈 모델과 수학적 관측 모델**에 집중.

### 2.1 카메라 관측 모델
- Pinhole + distortion (radial-tangential, fisheye/equidistant)
- 관측 방정식: 3D→2D projection, reprojection error
- Rolling shutter 모델

### 2.2 LiDAR 관측 모델
- Range-bearing 모델, beam divergence
- Motion distortion (ego-motion compensation)
- Spinning vs solid-state 차이가 퓨전에 미치는 영향

### 2.3 IMU 모델
- Gyroscope / Accelerometer 오차 모델 (bias, random walk, Allan variance)
- IMU noise parameters: 데이터시트 읽는 법
- Strapdown navigation equation

### 2.4 GNSS 모델
- Pseudorange / carrier phase 관측 모델
- Multipath, ionosphere/troposphere 오차
- RTK / PPP 기초

### 2.5 Radar 모델
- FMCW radar 원리, range-Doppler map
- 4D imaging radar 최신 동향

### 2.6 기타 센서
- Wheel odometry (slip 모델)
- Barometer, magnetometer
- UWB (ranging 모델)

**핵심 논문:**
- [IMU] Woodman (2007) "An introduction to inertial navigation" — IMU 오차 모델 교과서급
- [Camera] Kannala & Brandt (2006) — fisheye 카메라 모델
- [LiDAR] 센서 모델 관련은 각 SLAM 논문에서 implicit하게 다룸

---

## Ch.3 — Calibration (Deep Dive)

> robotics-practice에서 체커보드 한두 페이지 수준. 여기서는 **챕터급으로 상세히**.

### 3.1 Camera Intrinsic Calibration
- Zhang's method (체커보드)
- 실전 팁: 포즈 다양성, 이미지 수, corner 정확도
- Fisheye / omnidirectional 캘리브레이션 (Scaramuzza OCamCalib)
- Kalibr 사용법

### 3.2 Camera-Camera (Stereo) Extrinsic
- Stereo rectification
- Epipolar constraint 기반 검증

### 3.3 Camera-LiDAR Extrinsic
- Target-based: 체커보드/AprilTag + 3D-2D correspondences
- Targetless: mutual information, edge alignment, learning-based
- 실전에서 많이 쓰는 도구: autoware calibration toolkit, direct_visual_lidar_calibration

### 3.4 Camera-IMU Extrinsic + Temporal
- Kalibr (continuous-time B-spline)
- Temporal offset 추정의 중요성
- Allan variance 측정 실습

### 3.5 LiDAR-IMU Extrinsic
- Hand-eye calibration (AX=XB)
- Motion-based 자동 캘리브레이션
- LI-Init (FAST-LIO 계열)

### 3.6 LiDAR-LiDAR Extrinsic
- Multi-LiDAR rig calibration
- Targetless: ICP 기반, feature-based

### 3.7 GNSS-IMU Lever Arm & Boresight
- Lever arm vector 추정
- GNSS antenna phase center

### 3.8 Online / Continuous Calibration
- Self-calibration during SLAM
- Extrinsic drift 보정
- OpenCalib (자율주행 통합 캘리브레이션 프레임워크)

### 3.9 Temporal Calibration
- Hardware sync vs software sync
- PTP, PPS, trigger-based 동기화
- Time offset 온라인 추정 (Li & Mourikis 2014)

**핵심 논문:**
- Zhang (2000) "A Flexible New Technique for Camera Calibration" — 장의 방법
- Furgale et al. (2013) "Kalibr" — camera-IMU 캘리브레이션의 사실상 표준
- Scaramuzza et al. (2006) "OCamCalib" — omnidirectional 캘리브레이션
- Pandey et al. (2015) "Automatic extrinsic calibration of LiDAR and camera" — targetless 캘리브레이션
- Tsai & Lenz (1989) "A new technique for fully autonomous and efficient 3D robotics hand/eye calibration" — hand-eye calibration 원조
- Koide et al. (2023) "General, target-free, and LiDAR-camera extrinsic calibration" — 최신 targetless
- OpenCalib (2023) — 자율주행 캘리브레이션 통합

---

## Ch.4 — State Estimation 이론

> robotics-practice에서 EKF/PF를 소개 수준으로 다룸. 여기서는 **유도 과정과 센서 퓨전 관점**에서 심도 있게.

### 4.1 Bayesian Filtering Framework
- Prediction-update cycle
- Chapman-Kolmogorov equation
- 왜 closed-form이 안 되는가 → 근사 필요

### 4.2 Kalman Filter 계열
- KF: 유도, 최적성 증명 (MMSE)
- EKF: 선형화, Jacobian 계산
- Error-State KF (ESKF): 왜 로봇에서 EKF보다 ESKF를 쓰는가
- UKF: sigma point 변환, 장단점
- IEKF (Iterated EKF): 비선형성이 심할 때

### 4.3 Particle Filter
- Sequential Monte Carlo
- Importance sampling, resampling 전략
- Rao-Blackwellized PF (RBPF) — FastSLAM 연결

### 4.4 Smoothing vs Filtering
- Fixed-lag smoother
- Full smoothing (batch optimization)
- 왜 현대 SLAM은 filtering에서 optimization으로 갔는가

### 4.5 Factor Graph & Optimization
- Factor graph 표현
- MAP inference = nonlinear least squares
- Gauss-Newton, LM on manifold
- GTSAM / Ceres / g2o 비교

### 4.6 IMU Preintegration
- 왜 preintegration이 필요한가 (bias 재선형화 문제)
- Forster et al. (2017) on-manifold preintegration 유도
- 코드로 보는 preintegration

### 4.7 Marginalization & Sliding Window
- Schur complement
- First-Estimate Jacobian (FEJ)
- Sliding window 구현의 실전 이슈

**핵심 논문:**
- Kalman (1960) "A New Approach to Linear Filtering and Prediction Problems" — 칼만 필터 원논문
- Thrun et al. (2005) "Probabilistic Robotics" — 교과서 (Ch.2-4)
- Forster et al. (2017) "On-Manifold Preintegration for Real-Time VIO" — IMU preintegration
- Dellaert & Kaess (2017) "Factor Graphs for Robot Perception" — factor graph 튜토리얼
- Kaess et al. (2012) "iSAM2" — incremental smoothing
- Barfoot (2017) "State Estimation for Robotics" — 교과서

---

## Ch.5 — Feature Matching & Correspondence (기술 계보)

> **이 챕터의 목적**: mutual information에서 출발해서 LoFTR, RoMa까지 이어지는 일련의 기술적 흐름을 보여준다.
> 센서 퓨전의 거의 모든 컴포넌트(VO, calibration, loop closure, registration)가 correspondence에 의존한다.

### 5.1 Correspondence 문제란
- 두 관측에서 "같은 것"을 찾는 문제
- 2D-2D, 2D-3D, 3D-3D correspondence
- 왜 센서 퓨전의 핵심인가

### 5.2 전통적 Feature Detection & Description
- Corner detection: Harris → FAST → ORB
- Blob detection: SIFT → SURF
- Binary descriptors: BRIEF, ORB, BRISK
- 기술 계보: 정확도 vs 속도 트레이드오프의 역사

### 5.3 전통적 Matching & Outlier Rejection
- Brute-force, FLANN, ratio test (Lowe's ratio)
- RANSAC 계열: RANSAC → PROSAC → MAGSAC → MAGSAC++
- Fundamental/Essential matrix estimation

### 5.4 Mutual Information & Intensity-based Registration
- MI 정의와 직관
- MI 기반 다중 모달리티 정합 (의료영상에서 로보틱스로)
- NMI, MI gradient 계산
- 왜 calibration (Ch.3 targetless)에서 MI가 쓰이는가

### 5.5 학습 기반 Feature Detection & Description
- SuperPoint: self-supervised keypoint + descriptor
- D2-Net: detect-and-describe jointly
- R2D2: reliable and repeatable detector
- DISK
- 전통 대비 장점: illumination/viewpoint invariance 향상

### 5.6 학습 기반 Feature Matching
- SuperGlue: attention 기반 matching (GNN)
- LightGlue: SuperGlue 경량화
- 기술 흐름: detect → describe → match 분리 파이프라인

### 5.7 Detector-Free Matching (혁신)
- LoFTR: coarse-to-fine transformer matching (keypoint 없이 직접 매칭)
- QuadTree Attention: LoFTR의 효율화
- ASpanFormer: adaptive span
- RoMa: robust dense matching, DINOv2 backbone 활용
- 왜 detector-free가 중요한가: textureless, repetitive pattern 환경

### 5.8 3D-3D Correspondence
- FPFH, SHOT: handcrafted 3D descriptors
- 3DMatch, FCGF: learned 3D descriptors
- Predator: overlap-aware 3D matching
- GeoTransformer: geometric transformer for 3D registration

### 5.9 Cross-Modal Correspondence
- 2D-3D matching: 카메라-LiDAR 간
- Image-to-point cloud: LCD, P2-Net
- 왜 cross-modal이 어려운가: representation gap

### 5.10 Dense Matching & Optical Flow
- Classical: Lucas-Kanade, Horn-Schunck
- Learning: FlowNet → RAFT → FlowFormer → UniMatch
- Dense stereo: SGM → RAFT-Stereo → UniMatch
- 기술 흐름: sparse feature → dense correspondence

**기술 계보 요약:**
```
Harris (1988) → SIFT (2004) → FAST/ORB (2006/2011)
      ↓ [detection & description]
SuperPoint (2018) → D2-Net (2019) → DISK (2020)
      ↓ [matching]
BF + ratio test → SuperGlue (2020) → LightGlue (2023)
      ↓ [detector-free paradigm shift]
LoFTR (2021) → ASpanFormer (2022) → RoMa (2024)
      ↓ [dense]
Lucas-Kanade → RAFT (2020) → UniMatch (2023)
      ↓ [3D]
FPFH → 3DMatch (2017) → Predator (2021) → GeoTransformer (2022)
```

**핵심 논문:**
- Lowe (2004) "SIFT" — 특징점의 원점
- Rublee et al. (2011) "ORB" — 실시간 대안
- DeTone et al. (2018) "SuperPoint" — 학습 기반 특징점의 시작
- Sarlin et al. (2020) "SuperGlue" — attention matching
- Sun et al. (2021) "LoFTR" — detector-free matching 패러다임
- Edstedt et al. (2024) "RoMa" — DINOv2 + dense matching
- Teed & Deng (2020) "RAFT" — optical flow의 기준점
- Qin et al. (2022) "GeoTransformer" — 3D registration transformer
- Lindenberger et al. (2023) "LightGlue"
- Fischler & Bolles (1981) "RANSAC" — 원논문

---

## Ch.6 — Visual Odometry & Visual-Inertial Odometry

> robotics-practice Ch.14에서 VO/VIO를 소개. 여기서는 **내부 구조와 설계 선택**을 깊이 다룸.

### 6.1 Feature-based VO
- Frontend: detection, tracking, outlier rejection
- Backend: PnP, motion-only BA, local BA
- ORB-SLAM3 아키텍처 상세 분석

### 6.2 Direct VO
- Photometric error, 장점과 한계
- DSO 아키텍처 상세 분석
- Semi-direct: SVO

### 6.3 Tightly-coupled VIO
- VINS-Mono 아키텍처 상세
- OKVIS
- MSCKF: multi-state constraint, 왜 EKF 기반이 아직 살아있는가
- Basalt

### 6.4 VIO 설계 선택지
- Filter vs Optimization
- Keyframe selection 전략
- Feature parameterization (inverse depth, anchored)

### 6.5 학습 기반 VO/VIO
- Supervised: DeepVO 계열
- Self-supervised: 한계와 현재 위치
- Hybrid: 전통+학습 조합 (DROID-SLAM)

**핵심 논문:**
- Nistér et al. (2004) "Visual Odometry" — VO 원조
- Mourikis & Roumeliotis (2007) "MSCKF" — EKF 기반 VIO의 기준점
- Qin et al. (2018) "VINS-Mono" — tightly-coupled VIO 대표
- Campos et al. (2021) "ORB-SLAM3" — multi-map, multi-sensor SLAM
- Engel et al. (2018) "DSO" — direct sparse odometry
- Forster et al. (2017) "SVO" — semi-direct
- Teed & Deng (2021) "DROID-SLAM" — 학습 기반 SLAM
- Geneva et al. (2020) "OpenVINS" — MSCKF 오픈소스 구현

---

## Ch.7 — LiDAR Odometry & LiDAR-Inertial Odometry

### 7.1 Point Cloud Registration 기초
- ICP 변종들: point-to-point, point-to-plane, GICP
- NDT (Normal Distributions Transform)
- 수렴성, 초기값 의존성

### 7.2 Feature-based LiDAR Odometry
- LOAM: edge/planar feature 추출, sweep 분리
- LeGO-LOAM: ground segmentation 추가
- 왜 LOAM 계열이 오래 살아남았는가

### 7.3 Tightly-coupled LIO
- LIO-SAM: factor graph + IMU preintegration + GPS
- FAST-LIO / FAST-LIO2: iterated EKF + ikd-tree
- Faster-LIO: incremental voxel
- Point-LIO: point-by-point 처리 (고속 모션)
- COIN-LIO: 카메라 intensity 활용 LIO

### 7.4 Continuous-Time LiDAR Odometry
- Elastic LiDAR fusion
- CT-ICP
- B-spline 기반 trajectory 표현

### 7.5 Solid-State LiDAR 특화
- Livox 시리즈 대응 (비반복 스캔)
- FAST-LIO가 solid-state에 강한 이유

### 7.6 학습 기반 LiDAR Odometry
- DeepLO 계열
- 현재의 한계

**핵심 논문:**
- Besl & McKay (1992) "ICP" — 원논문
- Segal et al. (2009) "GICP" — generalized ICP
- Zhang & Singh (2014) "LOAM" — LiDAR odometry의 기준
- Shan & Englot (2018) "LeGO-LOAM"
- Shan et al. (2020) "LIO-SAM" — factor graph 기반 LIO
- Xu et al. (2022) "FAST-LIO2" — iterated EKF + ikd-tree
- He et al. (2023) "Point-LIO"
- Dellenbach et al. (2022) "CT-ICP" — continuous-time

---

## Ch.8 — Multi-Sensor Fusion 아키텍처

> 개별 odometry를 넘어 **여러 센서를 어떻게 통합하는가**의 설계론.

### 7.1 Fusion 아키텍처 분류
- Loosely coupled: 각 센서 독립 처리 → 결과 결합
- Tightly coupled: raw measurement를 하나의 optimizer에
- Ultra-tightly coupled: signal level fusion

### 7.2 Camera + LiDAR + IMU 융합
- R3LIVE / R3LIVE++: VIO + LIO + photometric refinement
- LVI-SAM: VIO ↔ LIO 양방향
- FAST-LIVO: direct visual-LiDAR-inertial
- Multimodal factor graph 설계

### 7.3 GNSS 통합
- GNSS factor in factor graph (LIO-SAM)
- Loose: EKF fusion / Tight: pseudorange factor
- GNSS-denied → GNSS-available 전환 처리

### 7.4 Radar 퓨전
- 4D radar + camera fusion
- Radar odometry (근래 급부상)
- Radar의 장점: 악천후, Doppler 직접 측정

### 7.5 Multi-Robot / Decentralized Fusion
- Communication-constrained fusion
- Distributed pose graph optimization
- Kimera-Multi, Swarm-SLAM

### 7.6 시스템 설계 실전
- Sensor suite 선정 가이드
- Timing architecture (HW sync, SW sync)
- Failure mode와 degradation handling

**핵심 논문:**
- Lin et al. (2022) "R3LIVE" — visual-LiDAR-inertial 실시간 융합
- Shan et al. (2021) "LVI-SAM"
- Zheng et al. (2024) "FAST-LIVO2"
- Burnett et al. (2024) "Boreas" — 다중 센서 + radar 데이터셋 & 벤치마크
- Cioffi et al. (2022) "Continuous-time vs discrete-time VIO"
- Lajoie et al. (2024) "Swarm-SLAM"
- Rosinol et al. (2021) "Kimera-Multi"

---

## Ch.9 — Place Recognition & Retrieval

> robotics-practice Ch.14.14에서 PR을 간략 소개. 여기서는 **방법론 전체를 체계적으로**.

### 8.1 문제 정의
- Place recognition vs loop closure detection
- Retrieval pipeline: query → database → candidate → geometric verification
- Precision-recall, recall@N 평가

### 8.2 Visual Place Recognition (VPR)
- 전통: BoW (DBoW2), VLAD, Fisher Vector
- CNN 기반: NetVLAD, AP-GeM, CosPlace
- Transformer 기반: AnyLoc, EigenPlaces
- Foundation model 활용: DINOv2 feature aggregation
- Sequence matching: SeqSLAM, SeqNet

### 8.3 LiDAR Place Recognition
- Handcrafted: Scan Context, M2DP, ESF
- Learning: PointNetVLAD, MinkLoc3D, LoGG3D-Net
- BEV 기반: OverlapNet, BEVPlace
- Range image 기반: OverlapTransformer

### 8.4 Cross-Modal Place Recognition
- LiDAR ↔ Camera: (LC)² (Lee et al. 2023), ModaLink
- 왜 cross-modal PR이 어려운가: domain gap
- Modality-agnostic descriptor 접근

### 8.5 Multi-Session & Long-Term PR
- 계절/시간/날씨 변화 대응
- Map update 전략
- Lifelong PR

### 8.6 Geometric Verification & Re-ranking
- PnP + RANSAC
- 3D-3D registration
- Spatial re-ranking

### 8.7 최신 동향
- Foundation model 기반 PR의 부상
- Semantic place recognition
- 4D radar PR

**핵심 논문:**
- Sivic & Zisserman (2003) "Video Google" — BoW 기반 retrieval 원조
- Arandjelović et al. (2016) "NetVLAD" — 학습 기반 VPR의 기준점
- Kim & Kim (2018) "Scan Context" — LiDAR PR 대표
- Uy & Lee (2018) "PointNetVLAD" — 3D point cloud retrieval
- Keetha et al. (2023) "AnyLoc" — foundation model 기반 VPR
- Berton et al. (2023) "EigenPlaces"
- Lee et al. (2023) "(LC)²" — cross-modal LiDAR-Camera PR
- Cattaneo et al. (2022) "LCDNet" — LiDAR loop closure + relative pose
- Ma et al. (2022) "OverlapTransformer" — range image 기반 LiDAR PR

---

## Ch.10 — Loop Closure & Global Optimization

### 9.1 Loop Closure Pipeline
- Detection → Verification → Correction
- False positive의 위험과 방지

### 9.2 Pose Graph Optimization
- SE(3) pose graph
- Robust kernel (Huber, Cauchy, DCS)
- Incremental: iSAM2 동작 원리

### 9.3 Global Relocalization
- 지도에 대한 localization (map-based)
- Prior map + online sensor
- Monte Carlo Localization (MCL)

### 9.4 Multi-Session SLAM & Map Merging
- Map anchoring
- Inter-session loop closure
- ORB-SLAM3 multi-map system

**핵심 논문:**
- Kümmerle et al. (2011) "g2o" — graph optimization 프레임워크
- Kaess et al. (2012) "iSAM2"
- Sünderhauf & Protzel (2012) "Switchable constraints" — robust loop closure
- Latif et al. (2013) "Robust loop closing"
- Agarwal et al. (2013) "Robust map optimization using DCS"

---

## Ch.11 — Spatial Representations

> 센서 퓨전의 **결과물**: 어떤 형태로 세상을 표현하는가.

### 10.1 Metric Maps
- Occupancy grid (2D/3D)
- Voxel maps: OctoMap, VDB, ikd-tree
- Surfel maps: ElasticFusion, SurfelMeshing

### 10.2 Mesh & CAD-level Maps
- Poisson reconstruction
- Voxblox → TSDF → mesh
- 실시간 mesh 생성

### 10.3 Neural / Learned Representations
- NeRF-SLAM: neural implicit + odometry
- 3DGS-SLAM: Gaussian splatting 기반
- 장단점: 렌더링 품질 vs computational cost

### 10.4 Semantic Maps
- Object-level maps
- 3D Scene Graph (Hydra, S-Graphs)
- Open-vocabulary semantic mapping

### 10.5 Long-Term & Dynamic Maps
- 변화 감지 (change detection)
- Map maintenance: 어떤 정보를 유지/삭제할 것인가
- Dynamic object 처리

**핵심 논문:**
- Hornung et al. (2013) "OctoMap" — 3D occupancy mapping
- Whelan et al. (2015) "ElasticFusion" — surfel 기반 dense SLAM
- Oleynikova et al. (2017) "Voxblox" — TSDF 기반 매핑
- Hughes et al. (2022) "Hydra" — 3D scene graph
- Sucar et al. (2021) "iMAP" — 최초 neural implicit SLAM
- Matsuki et al. (2024) "Gaussian Splatting SLAM"
- Rosinol et al. (2020) "Kimera" — metric-semantic SLAM

---

## Ch.12 — 실전 시스템 & 벤치마크

### 11.1 자율주행 Perception Stack
- Sensor suite 구성 사례 (Waymo, nuScenes)
- Production-level fusion pipeline

### 11.2 드론/UAV
- Visual-inertial 중심 시스템
- GPS-denied navigation
- 실시간 제약

### 11.3 핸드헬드/백팩 매핑
- SLAM as a service
- Survey-grade mapping

### 11.4 벤치마크 & 평가
- 데이터셋: KITTI, EuRoC, TUM, Hilti, HeLiPR, Newer College
- 메트릭: ATE, RPE, recall@N
- 공정한 비교의 어려움

### 11.5 오픈소스 도구 가이드
- GTSAM, Ceres, g2o
- Kalibr, OpenCalib
- evo (evaluation)

---

## Ch.13 — Frontiers & Emerging Directions

### 12.1 Foundation Models for Spatial AI
- DINOv2/CLIP feature를 SLAM에 활용
- Open-vocabulary 3D understanding
- FM이 전통 파이프라인을 얼마나 대체할 수 있는가

### 12.2 End-to-End Learned Odometry/SLAM
- 현재의 한계와 가능성
- Differentiable SLAM components

### 12.3 Spatial Memory & Scene Graphs
- Persistent spatial memory
- Scene Graph 기반 환경 이해
- 시계열 공간 기억 관리

### 12.4 Cross-Modal Representation
- 이종 센서 간 표현 정렬 문제
- Contrastive learning, knowledge distillation
- 아직 열린 질문들

### 12.5 Event Camera 기반 퓨전
- 고속, HDR 환경에서의 장점
- Event + frame 퓨전, event + IMU
- EVO, Ultimate SLAM

### 12.6 4D Radar 퓨전
- 악천후 robustness
- Radar odometry의 최근 발전

---

## 논문 목록 (기념비적 / 반드시 읽어야 할 논문)

챕터별로 fetch하여 읽고 정리할 핵심 논문 목록:

### Tier 1 — 분야를 정의한 논문 (반드시 fetch & 정리)
1. Kalman (1960) — 칼만 필터 원논문
2. Fischler & Bolles (1981) — RANSAC
3. Besl & McKay (1992) — ICP
4. Zhang (2000) — Camera calibration (Zhang's method)
5. Lowe (2004) — SIFT
6. Sivic & Zisserman (2003) — Video Google (BoW retrieval)
7. Nistér et al. (2004) — Visual Odometry
8. Mourikis & Roumeliotis (2007) — MSCKF
9. Zhang & Singh (2014) — LOAM
10. Arandjelović et al. (2016) — NetVLAD
11. Forster et al. (2017) — IMU Preintegration (on-manifold)
12. Qin et al. (2018) — VINS-Mono

### Tier 2 — 현재 가장 많이 쓰이는 시스템
13. Campos et al. (2021) — ORB-SLAM3
14. Xu et al. (2022) — FAST-LIO2
15. Shan et al. (2020) — LIO-SAM
16. Lin et al. (2022) — R3LIVE
17. Engel et al. (2018) — DSO
18. Geneva et al. (2020) — OpenVINS
19. Kim & Kim (2018) — Scan Context
20. Kaess et al. (2012) — iSAM2
21. Dellaert & Kaess (2017) — Factor Graphs tutorial
22. Furgale et al. (2013) — Kalibr

### Tier 3 — 딥러닝 패러다임 전환 논문 (Feature Matching 계보)
23. DeTone et al. (2018) — SuperPoint
24. Sarlin et al. (2020) — SuperGlue
25. Sun et al. (2021) — LoFTR
26. Teed & Deng (2020) — RAFT (optical flow)
27. Edstedt et al. (2024) — RoMa
28. Lindenberger et al. (2023) — LightGlue

### Tier 4 — 최신 중요 논문
29. Teed & Deng (2021) — DROID-SLAM
30. Keetha et al. (2023) — AnyLoc
31. Hughes et al. (2022) — Hydra (3D Scene Graph)
32. Zheng et al. (2024) — FAST-LIVO2
33. Koide et al. (2023) — targetless LiDAR-camera calibration
34. Qin et al. (2022) — GeoTransformer (3D registration)
