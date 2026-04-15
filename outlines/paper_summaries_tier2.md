# Tier 2 논문 요약 — 현재 가장 많이 쓰이는 시스템

> 센서 퓨전 가이드 핵심 논문 정리 (Tier 2: 현재 널리 사용되는 시스템/프레임워크)

---

## 1. ORB-SLAM3: An Accurate Open-Source Library for Visual, Visual-Inertial and Multi-Map SLAM

- **저자/연도**: Carlos Campos, Richard Elvira, Juan J. Gomez Rodriguez, Jose M. M. Montiel, Juan D. Tardos (2021)
- **발표**: IEEE Transactions on Robotics (T-RO), 2021
- **핵심 기여**: 단일 프레임워크에서 monocular/stereo/RGB-D + IMU를 모두 지원하는 최초의 통합 visual(-inertial) SLAM 시스템으로, pinhole과 fisheye 렌즈 모델을 모두 수용한다.
- **주요 내용**:
  - **Tightly-coupled Visual-Inertial SLAM**: MAP(Maximum-a-Posteriori) 추정 기반으로 IMU 초기화부터 최적화까지 일관된 framework 제공. 기존 대비 2~5배 정확도 향상.
  - **Multi-Map System**: 시각 정보가 부족한 구간에서 새 맵을 생성하고, 재방문 시 기존 맵과 자동 병합하는 Atlas 구조 도입.
  - **정보 재사용**: VO와 달리 과거 키프레임 정보를 전면 활용 — co-visible keyframe, parallax가 큰 관측, 이전 세션 데이터까지 포함.
  - **실험 결과**: EuRoC에서 stereo-inertial 구성이 3.6cm, TUM-VI에서 9mm 정확도 달성.
  - **오픈소스**: 완전한 코드 공개로 연구/교육에 광범위하게 활용됨.
- **이 가이드에서의 위치**: Ch.6 (Visual Odometry & VIO) — feature-based VO의 대표 시스템으로 아키텍처 상세 분석. Ch.10 (Loop Closure & Global Optimization) — multi-map/multi-session SLAM 사례로 재등장.
- **전작 대비 개선점**: ORB-SLAM2 대비 IMU tight integration 추가, multi-map atlas 도입으로 tracking loss 후 복구 능력 획득, fisheye 렌즈 지원 추가.

---

## 2. FAST-LIO2: Fast Direct LiDAR-Inertial Odometry

- **저자/연도**: Wei Xu, Yixi Cai, Dongjiao He, Jiarong Lin, Fu Zhang (2022)
- **발표**: IEEE Transactions on Robotics (T-RO), 2022
- **핵심 기여**: 특징 추출 없이 raw point를 직접 맵에 정합하는 direct LiDAR-inertial odometry로, 새로 제안한 ikd-Tree 자료구조를 통해 최대 100Hz 실시간 처리를 달성했다.
- **주요 내용**:
  - **Direct Point Registration**: hand-crafted feature extraction을 제거하고 raw LiDAR point를 직접 맵에 정합. 미세한 환경 특징도 감지 가능하며, 다양한 LiDAR 스캔 패턴에 범용 적용.
  - **ikd-Tree (Incremental k-d Tree)**: point 삽입/삭제/동적 re-balancing을 지원하는 새로운 맵 자료구조. 기존 octree, R*-tree보다 효율적.
  - **Iterated Extended Kalman Filter (IEKF)**: 비선형성이 큰 LiDAR 관측에 대해 반복 선형화로 정확도 향상.
  - **하드웨어 범용성**: multi-line spinning LiDAR, solid-state LiDAR(Livox), UAV/핸드헬드 플랫폼, Intel/ARM 프로세서에서 모두 동작.
  - **계산 효율**: 실외 환경에서 100Hz odometry + mapping 달성, 기존 시스템 대비 높은 정확도와 낮은 연산량을 동시 충족.
- **이 가이드에서의 위치**: Ch.7 (LiDAR Odometry & LIO) — tightly-coupled LIO의 핵심 시스템으로 상세 분석. Ch.11 (Spatial Representations) — ikd-Tree를 voxel map 대안으로 소개.
- **전작 대비 개선점**: FAST-LIO(v1) 대비 feature extraction 단계를 완전히 제거하여 정보 손실 방지 및 연산량 절감. ikd-Tree 도입으로 맵 유지관리 효율 대폭 향상. solid-state LiDAR 지원 강화.

---

## 3. LIO-SAM: Tightly-coupled Lidar Inertial Odometry via Smoothing and Mapping

- **저자/연도**: Tixiao Shan, Brendan Englot, Drew Meyers, Wei Wang, Carlo Ratti, Daniela Rus (2020)
- **발표**: IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), 2020
- **핵심 기여**: Factor graph 프레임워크 위에 LiDAR, IMU, GPS, loop closure를 통합한 LIO 시스템으로, LOAM 계열의 feature 기반 접근과 현대적 그래프 최적화를 결합했다.
- **주요 내용**:
  - **Factor Graph 기반 통합**: LiDAR odometry, IMU preintegration, GPS, loop closure를 각각 factor로 모델링하여 하나의 그래프에서 joint optimization.
  - **IMU Preintegration**: IMU 데이터로 포인트 클라우드의 motion distortion을 보정(de-skewing)하고, LiDAR odometry가 IMU bias를 추정하는 양방향 구조.
  - **Keyframe 기반 효율화**: 전역 맵 대신 sliding window 내 sub-keyframe 집합에 대해 scan matching을 수행하여 계산 효율 확보.
  - **Marginalization**: 오래된 LiDAR scan을 marginalize하여 그래프 크기를 관리 가능한 수준으로 유지.
  - **다양한 플랫폼 검증**: 핸드헬드, 차량, 보트 등 다양한 규모와 환경에서 평가 완료.
- **이 가이드에서의 위치**: Ch.7 (LiDAR Odometry & LIO) — factor graph 기반 LIO의 대표 사례. Ch.4 (State Estimation) — IMU preintegration 실전 적용 예시. Ch.8 (Multi-Sensor Fusion) — GNSS 통합 사례.
- **전작 대비 개선점**: LeGO-LOAM 대비 factor graph 기반으로 전환하여 다중 센서(GPS 등) 통합이 자연스러워짐. IMU tight coupling으로 고속 모션 대응력 향상. Loop closure를 factor로 통합하여 전역 일관성 확보.

---

## 4. R3LIVE: A Robust, Real-time, RGB-colored, LiDAR-Inertial-Visual Tightly-coupled State Estimation and Mapping Package

- **저자/연도**: Jiarong Lin, Fu Zhang (2022)
- **발표**: IEEE Robotics and Automation Letters (RA-L), 2022; ICRA 2022 발표
- **핵심 기여**: LiDAR, IMU, 카메라 세 센서를 tightly-coupled로 융합하여 실시간으로 RGB-colored dense 3D 맵을 생성하는 최초의 통합 시스템이다.
- **주요 내용**:
  - **이중 서브시스템 아키텍처**: LiDAR-Inertial Odometry(LIO) 서브시스템이 기하학적 구조를, Visual-Inertial Odometry(VIO) 서브시스템이 텍스처 정보를 각각 담당하되 상태 추정에서 tightly coupled.
  - **Photometric Error 기반 시각 융합**: frame-to-map visual registration을 통해 맵 상의 각 포인트에 RGB 색상을 실시간으로 부여.
  - **Robustness**: LiDAR 또는 카메라 중 하나가 일시적으로 실패해도 나머지 센서로 지속 동작.
  - **Dense 3D 재구성**: SLAM과 동시에 surveying/mapping 수준의 컬러 3D 맵 생성.
  - **오픈소스**: 전체 코드, 유틸리티(mesh 재구성), 센서 장치 도면까지 공개.
- **이 가이드에서의 위치**: Ch.8 (Multi-Sensor Fusion Architecture) — camera + LiDAR + IMU 융합의 핵심 사례로 아키텍처 상세 분석.
- **전작 대비 개선점**: LiDAR-only 시스템(FAST-LIO2 등) 대비 시각 정보를 추가하여 텍스처 복원 및 degenerate geometry 환경(긴 복도 등)에서의 robustness 확보. 기존 visual-LiDAR 융합 시스템 대비 실시간 dense colored map 생성 능력이 핵심 차별점.

---

## 5. Direct Sparse Odometry (DSO)

- **저자/연도**: Jakob Engel, Vladlen Koltun, Daniel Cremers (2018)
- **발표**: IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI), Vol. 40(3), pp. 611-625, March 2018
- **핵심 기여**: 직접법(direct method)과 희소 표현(sparse representation)을 결합한 새로운 visual odometry로, photometric error를 최소화하면서 모든 모델 파라미터를 동시에 최적화한다.
- **주요 내용**:
  - **Direct + Sparse 결합**: 기존의 direct method는 dense였고 sparse method는 indirect(feature-based)였는데, DSO는 이 두 축을 새롭게 조합. 키포인트 없이 gradient가 있는 모든 영역에서 픽셀을 샘플링.
  - **완전한 Photometric Calibration**: 노출 시간(exposure time), 렌즈 비네팅(vignetting), 비선형 응답 함수(response function)를 모델에 통합하여 정확한 photometric error 계산.
  - **Joint Optimization**: 기하(inverse depth)와 카메라 모션을 하나의 확률적 프레임워크에서 동시 최적화.
  - **Textureless/Featureless 환경**: 키포인트에 의존하지 않으므로 텍스처가 없는 벽면, 반복 패턴 등에서도 동작 가능.
  - **성능**: 실제 다양한 환경에서 기존 direct/indirect 방법 대비 tracking 정확도와 robustness 모두 유의미하게 우수.
- **이 가이드에서의 위치**: Ch.6 (Visual Odometry & VIO) — Direct VO의 대표 시스템으로 아키텍처 상세 분석. Feature-based(ORB-SLAM3)와 대비하여 설계 철학 비교.
- **전작 대비 개선점**: LSD-SLAM(같은 그룹의 전작) 대비 dense regularization을 제거하고 sparse 샘플링으로 전환하여 연산 효율 확보. 완전한 photometric calibration 도입으로 정확도 향상. Windowed optimization으로 consistency 개선.

---

## 6. OpenVINS: A Research Platform for Visual-Inertial State Estimation

- **저자/연도**: Patrick Geneva, Kevin Eckenhoff, Woosik Lee, Yulin Yang, Guoquan Huang (2020)
- **발표**: IEEE International Conference on Robotics and Automation (ICRA), 2020, pp. 4666-4672
- **핵심 기여**: MSCKF 기반 visual-inertial 추정의 완전한 오픈소스 연구 플랫폼으로, 다양한 VIO 알고리즘 변형을 모듈식으로 비교/실험할 수 있는 환경을 제공한다.
- **주요 내용**:
  - **On-Manifold Sliding Window EKF**: MSCKF를 기반으로 한 sliding window 칼만 필터. 최적화 기반이 아닌 필터 기반 VIO의 대표 구현.
  - **온라인 캘리브레이션**: 카메라 intrinsic, extrinsic, camera-IMU 시간 오프셋을 런타임에 자동 추정.
  - **SLAM 랜드마크 지원**: 다양한 랜드마크 표현(anchored inverse depth 등)과 First-Estimates Jacobian(FEJ) 처리를 포함.
  - **모듈형 타입 시스템**: 상태 벡터를 유연하게 확장할 수 있는 타입 시스템 제공.
  - **시뮬레이터 & 평가 도구**: visual-inertial 시뮬레이터와 포괄적 알고리즘 평가 도구 내장.
- **이 가이드에서의 위치**: Ch.6 (Visual Odometry & VIO) — MSCKF 계열 VIO의 오픈소스 구현으로 소개. Filter vs Optimization 비교에서 필터 측 대표.
- **전작 대비 개선점**: 기존 MSCKF 구현들이 비공개이거나 제한적이었던 반면, OpenVINS는 완전한 모듈형 오픈소스로 다양한 설정(카메라 모델, 랜드마크 표현, 캘리브레이션 모드)을 실험적으로 비교 가능하게 만듦. 온라인 temporal calibration 포함.

---

## 7. Scan Context: Egocentric Spatial Descriptor for Place Recognition within 3D Point Cloud Map

- **저자/연도**: Giseop Kim, Ayoung Kim (2018)
- **발표**: IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), 2018, pp. 4802-4809
- **핵심 기여**: 3D LiDAR 스캔으로부터 histogram 없이 직접 공간 구조를 인코딩하는 전역 디스크립터로, 역방향 재방문과 코너 재방문에서도 장소 인식이 가능하다.
- **주요 내용**:
  - **Non-Histogram 기반 디스크립터**: 기존 방법들이 통계적 histogram으로 구조를 요약한 반면, Scan Context는 3D 공간을 bin/sector로 나누어 최대 높이값을 직접 기록. 공간 구조 자체를 보존.
  - **Egocentric 표현**: 센서 중심 좌표계에서 표현하여 viewpoint 변화에 invariant. 특히 역방향(reverse) 재방문에서 강점.
  - **효율적 검색**: ring key와 sector key를 이용한 2단계 검색으로 대규모 맵에서도 빠른 후보 탐색.
  - **학습 불필요**: 사전 학습 없이 순수 기하학적 접근으로 동작. 새로운 환경에 즉시 적용 가능.
  - **LiDAR SLAM 통합 용이**: LIO-SAM 등 주요 LiDAR SLAM 시스템에 loop closure 모듈로 널리 채택됨.
- **이 가이드에서의 위치**: Ch.9 (Place Recognition & Retrieval) — LiDAR Place Recognition의 handcrafted 대표로 상세 설명. 학습 기반(PointNetVLAD 등)과 비교.
- **전작 대비 개선점**: 기존 LiDAR PR 방법(M2DP, ESF 등)이 histogram 기반으로 공간 배치 정보를 소실한 반면, Scan Context는 공간 구조를 직접 보존. 역방향 방문 인식 능력이 핵심 차별점. 후속작 Scan Context++(2021)에서 semantic 정보 추가.

---

## 8. iSAM2: Incremental Smoothing and Mapping Using the Bayes Tree

- **저자/연도**: Michael Kaess, Hordur Johannsson, Richard Roberts, Viorela Ila, John J. Leonard, Frank Dellaert (2012)
- **발표**: International Journal of Robotics Research (IJRR), Vol. 31, No. 2, pp. 216-235, February 2012
- **핵심 기여**: Bayes tree라는 새로운 자료구조를 도입하여, 주기적 batch 처리 없이 incremental하게 비선형 최적화를 수행하는 SLAM 백엔드 알고리즘이다.
- **주요 내용**:
  - **Bayes Tree**: 기존 clique tree와 유사하지만 방향성을 가진 그래프 구조. 조건부 확률 밀도를 인코딩하며, SLAM의 square root information matrix에 자연스럽게 대응.
  - **Incremental Variable Reordering**: 새 변수/측정 추가 시 전체 재정렬 없이 영향받는 부분만 재구성. 탐색 중 대부분의 tree는 변경 불필요.
  - **Fluid Relinearization**: 선형화 지점이 크게 변한 변수만 선택적으로 재선형화. 주기적 batch step 필요성을 완전히 제거.
  - **그래프 편집의 직관적 해석**: 추상적인 행렬 분해 업데이트를 Bayes tree 상의 직관적 편집 연산으로 이해할 수 있음.
  - **실시간 SLAM 백엔드**: 대규모 환경에서도 일정한 시간 내에 최적화 완료. GTSAM 라이브러리의 핵심 알고리즘.
- **이 가이드에서의 위치**: Ch.4 (State Estimation) — factor graph & optimization 섹션에서 incremental smoothing의 핵심 알고리즘으로 상세 설명. Ch.10 (Loop Closure & Global Optimization) — pose graph optimization의 백엔드로 재등장.
- **전작 대비 개선점**: iSAM(v1) 대비 주기적 batch re-ordering/relinearization 필요성 완전 제거. 변수 추가/제거 시 영향 범위를 Bayes tree 구조로 정확히 파악하여 불필요한 연산 최소화. SLAM뿐 아니라 일반적인 incremental nonlinear least squares 문제에 적용 가능.

---

## 9. Factor Graphs for Robot Perception

- **저자/연도**: Frank Dellaert, Michael Kaess (2017)
- **발표**: Foundations and Trends in Robotics, Vol. 6, No. 1-2, pp. 1-139, August 2017
- **핵심 기여**: Factor graph를 이용한 로봇 인식 문제의 모델링과 해법을 139페이지에 걸쳐 체계적으로 정리한 튜토리얼/서베이로, SLAM 및 센서 퓨전 분야의 이론적 기초를 제공한다.
- **주요 내용**:
  - **Factor Graph 이론**: 확률적 그래피컬 모델의 한 종류인 factor graph가 왜 로봇 인식 문제에 적합한지 설명. 변수(노드)와 factor(제약 조건)의 분리를 통해 모듈적 문제 정의 가능.
  - **비선형 최적화**: factor graph 위의 MAP 추정을 nonlinear least squares로 환원하고, sparse linear system 풀이로 효율적 해법 제시.
  - **Incremental 방법론**: iSAM/iSAM2의 핵심 아이디어를 그래피컬 모델 관점에서 재해석. 행렬 분해의 incremental update를 Bayes tree 연산으로 설명.
  - **Manifold 위의 최적화**: 3D 회전(SO(3))과 pose(SE(3)) 같은 비선형 매니폴드에서의 최적화 방법 설명.
  - **SLAM 응용**: SLAM, SfM, 캘리브레이션 등 구체적 응용에서의 factor graph 구성 사례 다수 포함.
- **이 가이드에서의 위치**: Ch.4 (State Estimation) — Factor Graph & Optimization 섹션의 핵심 참고 문헌. 이 가이드 전체의 이론적 뼈대를 제공하는 논문.
- **전작 대비 개선점**: 개별 알고리즘 논문(iSAM, iSAM2 등)에서 단편적으로 설명된 이론을 하나의 통합된 프레임워크로 정리. 비전공자도 접근 가능한 수준에서 시작하여 최신 알고리즘까지 체계적으로 연결. GTSAM 라이브러리의 설계 철학과 직접 대응.

---

## 10. Unified Temporal and Spatial Calibration for Multi-Sensor Systems (Kalibr)

- **저자/연도**: Paul Furgale, Joern Rehder, Roland Siegwart (2013)
- **발표**: IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), 2013, pp. 1280-1286
- **핵심 기여**: 서로 다른 센서 간의 공간적 변위(extrinsic)와 시간적 오프셋(temporal)을 동시에 추정하는 통합 캘리브레이션 프레임워크를 제안했다.
- **주요 내용**:
  - **시공간 동시 캘리브레이션**: 기존에는 공간(extrinsic)과 시간(temporal offset) 캘리브레이션을 별도로 수행했으나, 이 논문은 두 문제를 하나의 최적화 문제로 통합 정의.
  - **Continuous-Time B-Spline Trajectory**: 연속 시간 B-spline으로 궤적을 표현하여 서로 다른 샘플링 레이트의 센서를 자연스럽게 처리. 이산 타임스탬프에 구속받지 않음.
  - **Camera-IMU 캘리브레이션**: 카메라와 IMU 간의 상대 pose 및 시간 오프셋을 calibration target(AprilGrid 등) 기반으로 정밀 추정.
  - **확장 가능한 프레임워크**: 카메라-카메라, 카메라-IMU, multi-IMU 등 다양한 센서 조합에 적용 가능.
  - **Kalibr 도구**: 이 논문의 방법론을 구현한 오픈소스 도구 Kalibr(ethz-asl/kalibr)가 camera-IMU 캘리브레이션의 사실상 표준으로 자리잡음.
- **이 가이드에서의 위치**: Ch.3 (Calibration Deep Dive) — Camera-IMU Extrinsic + Temporal 캘리브레이션의 핵심 방법으로 상세 분석. Temporal Calibration 섹션에서도 재등장.
- **전작 대비 개선점**: 기존 hand-eye calibration(AX=XB) 방식이 공간만 다루고 시간 동기화는 하드웨어에 의존한 반면, Kalibr는 소프트웨어적으로 시간 오프셋까지 추정. B-spline 연속 궤적 표현으로 비동기 센서 데이터를 우아하게 처리.

---

## 논문 간 관계도

```
[이론적 기초]
  Dellaert & Kaess (2017) Factor Graphs
        │
        ├── Kaess et al. (2012) iSAM2 ──── GTSAM 라이브러리
        │         │
        │         ├── Shan et al. (2020) LIO-SAM (factor graph 기반 LIO)
        │         └── Campos et al. (2021) ORB-SLAM3 (graph optimization)
        │
  Furgale et al. (2013) Kalibr ──── 모든 시스템의 사전 캘리브레이션

[Visual Odometry 계보]
  Engel et al. (2018) DSO ─── direct method
  Campos et al. (2021) ORB-SLAM3 ─── feature-based method
  Geneva et al. (2020) OpenVINS ─── filter-based VIO

[LiDAR-Inertial 계보]
  Shan et al. (2020) LIO-SAM ─── factor graph + LOAM features
  Xu et al. (2022) FAST-LIO2 ─── IEKF + direct registration
        │
        └── Lin et al. (2022) R3LIVE ─── LIO + VIO + colored mapping

[Place Recognition]
  Kim & Kim (2018) Scan Context ──── LIO-SAM 등에 loop closure로 채택
```
