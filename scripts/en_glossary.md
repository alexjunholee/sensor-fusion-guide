# EN Translation Glossary

This glossary enforces consistent English terminology across the 13-chapter
sensor fusion / SLAM / robotics guide. Translators working in parallel must
use the canonical English form from the tables below. When a Korean term has
an inline English gloss in the source (e.g., `상태 추정(state estimation)`),
drop the Korean + parentheses and keep only the canonical English form unless
the gloss itself is what the sentence is defining.

## Style rules

- **US English.** Use "color," "modeling," "optimization," "behavior,"
  "neighbor," "center," etc. Prefer Oxford-free commas except in lists where
  ambiguity would result.
- **Math, code, LaTeX: verbatim.** Do not translate, paraphrase, or reformat
  equations, code comments that are code-identifiers, variable names,
  inline `$...$`, display `$$...$$`, or fenced code blocks. Korean inline
  comments inside code blocks *should* be translated.
- **Acronyms stay as-is.** SLAM, IMU, LiDAR, GNSS, GPS, INS, VO, VIO, LIO,
  LVI, IEKF, EKF, UKF, ESKF, MSCKF, PF, RANSAC, ICP, NDT, GICP, BA, PGO,
  BoW, VLAD, NetVLAD, VPR, MI, NMI, DoF, FoV, IoU, ATE, RPE, MLE, MAP,
  RTK, PPP, PPS, PTP, ROS, TSDF, NeRF, 3DGS, BEV.
- **Preserve identifiers verbatim:** OpenCV, Ceres, GTSAM, g2o, Kalibr,
  hloc, kornia, PyTorch, NumPy, SciPy, OctoMap, COLMAP, VINS-Mono,
  ORB-SLAM2/3, DSO, LSD-SLAM, LOAM, LeGO-LOAM, LIO-SAM, LVI-SAM,
  FAST-LIO, FAST-LIO2, Point-LIO, R3LIVE, SuperPoint, SuperGlue,
  LightGlue, LoFTR, RoMa, DROID-SLAM, DPVO, LO-Net, DeepLO, DeepVO,
  TartanVO, AnyLoc, Scan Context, OverlapTransformer, NetVLAD, DBoW2,
  Patchwork, Patchwork++, Removert, LT-mapper.
- **Datasets / benchmarks kept verbatim:** KITTI, KITTI-360, TUM-VI,
  TUM-RGBD, EuRoC, NCLT, Oxford RobotCar, Boreas, nuScenes, Waymo Open,
  Argoverse, Newer College, MulRan, M2DP, Hilti, SubT, TartanAir.
- **Paper titles / author citations:** keep the original English title and
  author casing; do not italicize unless already italicized. "Koide et al.
  (2023)" stays as "Koide et al. (2023)."
- **Chapter heading format:** `# Ch.N — Title` with an em dash (U+2014) and
  a single space on each side. Preserve this exactly; do not substitute a
  hyphen or en dash.
- **Section numbering (`## 2.1`, `### 2.1.1`) is preserved verbatim.** Only
  translate the heading text that follows the number.
- **Tone:** graduate-level technical textbook. Direct, precise, third-person
  or inclusive "we." Translate Korean instructor phrases as follows:
  - "살펴보자 / 알아보자" → "we examine," "we review," "let us consider"
  - "핵심은 …이다" → "The key point is …" / "The crux is …"
  - "…에 주목하자" → "note that …" / "observe that …"
  - "…해야 한다" → "must …" / "should …" (choose by strength of claim)
  - "…임을 알 수 있다" → "we see that …" / "it follows that …"
  - Korean imperative ("구하라," "풀어보자") → "we solve," "compute," or
    imperative "Solve …" in worked-example contexts.
- **Sentence-final politeness endings** (`-다`, `-이다`, `-한다`) translate
  to simple declarative present tense. Do not insert hedges ("perhaps,"
  "arguably") that are absent in the Korean.
- **Quotes:** Korean "겹낫표" / `『 』` / `「 」` → double quotes `"…"` for
  English. Keep emphasis that uses `**bold**` or `*italic*` as-is.
- **Punctuation:** Korean full-width punctuation (`，`, `。`, `：`) → ASCII
  equivalents. Em dashes `—` remain em dashes.
- **Numbers and units:** keep numerals; use non-breaking space between
  number and unit in prose ("10 Hz," "5 m"). Do not translate unit symbols.
- **Parenthetical English glosses:** if the Korean source writes
  `자코비안(Jacobian)`, the English translation is simply "Jacobian" (no
  redundant gloss). If the Korean writes English first, e.g.,
  `Rolling Shutter(롤링 셔터)`, the translation is "rolling shutter."

## Terms (Korean → English)

| Korean | English |
|---|---|
| 가능도 | likelihood |
| 가려짐 / 가림 | occlusion |
| 가속도계 | accelerometer |
| 가시선 | line of sight (LOS) |
| 각속도 | angular velocity |
| 각축 표현 | angle-axis representation |
| 갱신 단계 | update step |
| 거리 | range (LiDAR/Radar) / distance (generic) |
| 검출 | detection |
| 격자 (지도) | grid (map) |
| 결합 수준 | coupling level |
| 경쟁적 융합 | competitive fusion |
| 계절 변화 | seasonal change |
| 고도각 | elevation angle |
| 곡률 | curvature |
| 공간 표현 | spatial representation |
| 공분산 | covariance |
| 공분산 전파 | covariance propagation |
| 관측 모델 | observation model / measurement model |
| 관측 가능성 | observability |
| 관측 불가능 | unobservable |
| 광류 / 광학 흐름 | optical flow |
| 광축 | optical axis |
| 궤적 | trajectory |
| 그래프 신경망 | graph neural network (GNN) |
| 극좌표 | polar coordinates |
| 기본 행렬 | fundamental matrix |
| 기선 | baseline |
| 기압계 | barometer |
| 기하학적 검증 | geometric verification |
| 깊이 | depth |
| 내부 파라미터 | intrinsic parameters |
| 내부 파라미터 행렬 | intrinsic matrix |
| 느슨한 결합 | loose coupling / loosely coupled |
| 다중경로 | multipath |
| 대응 / 대응점 | correspondence |
| 도플러 | Doppler |
| 동차 좌표 | homogeneous coordinates |
| 디스크립터 / 기술자 | descriptor |
| 딥러닝 | deep learning |
| 렌즈 왜곡 | lens distortion |
| 로버스트 추정 | robust estimation |
| 루프 클로저 | loop closure |
| 리 대수 | Lie algebra |
| 리 군 | Lie group |
| 마지널라이제이션 | marginalization |
| 매니폴드 | manifold |
| 매니폴드 위의 최적화 | optimization on manifolds |
| 매칭 | matching |
| 매핑 | mapping |
| 명목 상태 | nominal state |
| 모션 왜곡 | motion distortion |
| 밀집 대응 | dense correspondence |
| 바이어스 | bias |
| 바이어스 불안정성 | bias instability |
| 반사율 | albedo / reflectivity |
| 반송파 위상 | carrier phase |
| 방사 왜곡 | radial distortion |
| 방위각 | azimuth |
| 방정식 (정규) | normal equations |
| 백엔드 | back end / backend (use consistently as one word: backend) |
| 번들 조정 | bundle adjustment |
| 변환 행렬 | transformation matrix |
| 병진 | translation |
| 보정 | correction |
| 본질 행렬 | essential matrix |
| 분산 최적화 | distributed optimization |
| 불확실성 | uncertainty |
| 비선형 최적화 | nonlinear optimization |
| 비행시간 | time-of-flight |
| 사전 분포 | prior (distribution) |
| 사후 분포 | posterior (distribution) |
| 상대 포즈 | relative pose |
| 상보적 융합 | complementary fusion |
| 상태 | state |
| 상태 추정 | state estimation |
| 센서 동기화 | sensor synchronization |
| 센서 리그 | sensor rig |
| 센서 퓨전 | sensor fusion |
| 속도 (선형) | velocity |
| 손실 함수 | loss function |
| 스테레오 | stereo |
| 스트랩다운 관성 항법 방정식 | strapdown navigation equations |
| 슬라이딩 윈도우 | sliding window |
| 시야각 | field of view (FoV) |
| 시차 | parallax / disparity (context-dependent) |
| 아웃라이어 | outlier |
| 아웃라이어 제거 | outlier rejection |
| 어안 렌즈 | fisheye lens |
| 에피폴라 기하학 | epipolar geometry |
| 에피폴라 라인 | epipolar line |
| 연속 시간 | continuous time |
| 오도메트리 | odometry |
| 오차 상태 | error state |
| 외부 파라미터 | extrinsic parameters |
| 원시 측정 | raw measurements |
| 월드 좌표계 | world frame |
| 음속 | speed of sound |
| 의사거리 | pseudorange |
| 이동 객체 | dynamic object / moving object |
| 이벤트 카메라 | event camera |
| 이상치 | outlier |
| 일관성 | consistency |
| 일차 근사 | first-order approximation |
| 자기 속도 | ego-velocity |
| 자력계 | magnetometer |
| 자세 | attitude |
| 자유도 | degree of freedom (DoF) |
| 자이로스코프 | gyroscope |
| 자코비안 | Jacobian |
| 잔차 | residual |
| 장소 인식 | place recognition |
| 재귀적 추정 | recursive estimation |
| 재투영 오차 | reprojection error |
| 전방 센서 | forward-facing sensor |
| 점군 | point cloud |
| 접선 왜곡 | tangential distortion |
| 정규 방정식 | normal equations |
| 정렬 | alignment / registration |
| 정보 행렬 | information matrix / Hessian |
| 정합 | registration |
| 제어점 | control point |
| 좌표계 | coordinate frame |
| 주점 | principal point |
| 지도 | map |
| 지연 (시간) | latency / time delay |
| 직교 | orthogonal |
| 차량 좌표계 | vehicle frame / body frame |
| 차륜 오도메트리 | wheel odometry |
| 초점 거리 | focal length |
| 최소 제곱 | least squares |
| 추적 | tracking |
| 측정 잡음 | measurement noise |
| 측위 | localization |
| 칼만 이득 | Kalman gain |
| 캘리브레이션 | calibration |
| 크로스 모달 | cross-modal |
| 키프레임 | keyframe |
| 특징점 | feature point / keypoint |
| 파라미터 | parameter |
| 파티클 필터 | particle filter |
| 팩터 그래프 | factor graph |
| 퍼텐셜 장 | potential field |
| 포즈 | pose |
| 포즈 그래프 | pose graph |
| 포즈 그래프 최적화 | pose graph optimization (PGO) |
| 표면 법선 | surface normal |
| 프론트엔드 | front end / frontend (use consistently as one word: frontend) |
| 플리커 | flicker |
| 핀홀 카메라 모델 | pinhole camera model |
| 필터링 | filtering |
| 학습 기반 | learning-based |
| 합성 데이터 | synthetic data |
| 해상도 | resolution |
| 해석적 자코비안 | analytical Jacobian |
| 혁신 | innovation |
| 확장 칼만 필터 | extended Kalman filter (EKF) |
| 회전 | rotation |
| 회전 보간 | rotation interpolation (SLERP) |
| 흐릿함 (모션) | motion blur |
| 희소 | sparse |
| 희소 구조 | sparse structure |
| 히스토그램 | histogram |

## Section heading patterns

| Korean | English |
|---|---|
| … 요약 | … Summary |
| …의 한계 | Limitations of … / … Limitations |
| …의 분류 | Taxonomy of … / Classification of … |
| 문제 정의 | Problem Definition |
| 문제 설정 | Problem Setup |
| 결합 수준 비교 | Comparison of Coupling Levels |
| 기술 계보 요약 | Technical Lineage Summary |
| 최근 동향 (YYYY-YYYY) | Recent Trends (YYYY-YYYY) |
| 최신 동향 | Recent Developments |
| 실전 팁 | Practical Tips |
| 실전 도구 비교 | Practical Tool Comparison |
| 실전 이슈 | Practical Issues |
| 실행 가이드 | Execution Guide / Usage Guide |
| 대상 독자 | Target Audience |
| 이 가이드의 범위와 구성 | Scope and Organization of This Guide |
| 이 가이드의 관통 테마 | Cross-Cutting Themes of This Guide |
| 가이드 구성 | Guide Organization |
| 카메라의 한계 | Camera Limitations |
| LiDAR의 한계 | LiDAR Limitations |
| IMU의 한계 | IMU Limitations |
| GNSS의 한계 | GNSS Limitations |
| 센서 한계의 상보성 | Complementarity of Sensor Limitations |
| 센서 퓨전의 분류 | Taxonomy of Sensor Fusion |
| 핀홀 카메라 모델 (복습) | Pinhole Camera Model (Review) |
| 렌즈 왜곡 모델 | Lens Distortion Model |
| 재투영 오차 | Reprojection Error |
| 롤링 셔터 모델 | Rolling Shutter Model |
| 자이로스코프 오차 모델 | Gyroscope Error Model |
| 가속도계 오차 모델 | Accelerometer Error Model |
| IMU 등급 분류 | IMU Grade Classification |
| 의사거리 관측 모델 | Pseudorange Observation Model |
| 반송파 위상 관측 모델 | Carrier Phase Observation Model |
| 기타 센서 | Other Sensors |
| 센서 모델링 요약 | Sensor Modeling Summary |
| 왜 …가 필요한가 | Why … Is Needed |
| 왜 현대 SLAM은 … | Why Modern SLAM Moved From … To … |
| X장 요약 | Chapter X Summary |
| 사례 / 구성 사례 | Case Study / Configuration Examples |
| 아키텍처 상세 분석 | Architecture Deep Dive |
| 설계 선택지 | Design Choices |
| 설계 체크리스트 | Design Checklist |
| 문제 / 과제 | Challenges |
| 현재의 한계 | Current Limitations |
| 현재의 주류 | Current Mainstream |
| 평가 메트릭 | Evaluation Metrics |
| 벤치마크 | Benchmarks |
| 코드로 보는 … | … in Code |
| 개요 | Overview |
| 유도 | Derivation |
| 정의와 직관 | Definition and Intuition |
| 실습 | Hands-On |

## Do-not-translate list

- **Acronyms** (keep uppercase, no translation): SLAM, IMU, LiDAR, GNSS,
  GPS, INS, VO, VIO, LIO, LVIO, SfM, ICP, NDT, GICP, EKF, IEKF, ESKF,
  UKF, MSCKF, PF, BA, PGO, BoW, VPR, MI, NMI, DoF, FoV, IoU, ATE, RPE,
  MLE, MAP, RTK, PPP, PPS, PTP, ROS, TSDF, NeRF, 3DGS, BEV, FMCW,
  NLOS, LOS, ARW, VRW, RRW, HW, AoA, ToF, AR/VR, MAV, HRI, SMC, ANN,
  kNN, NCC, SSD, SIFT, SURF, ORB, FAST, BRIEF, AKAZE, FPFH.
- **Library / framework / tool names:** OpenCV, ROS, ROS 2, Ceres,
  GTSAM, g2o, Kalibr, hloc, kornia, COLMAP, Open3D, PCL, PyTorch,
  TensorFlow, NumPy, SciPy, Matplotlib, OctoMap, Voxblox, nanoflann,
  Eigen, Sophus.
- **System / algorithm names:** ORB-SLAM2, ORB-SLAM3, VINS-Mono,
  VINS-Fusion, DSO, LDSO, LSD-SLAM, SVO, LOAM, LeGO-LOAM, LIO-SAM,
  LVI-SAM, FAST-LIO, FAST-LIO2, Point-LIO, R3LIVE, DROID-SLAM, DPVO,
  DeepVO, TartanVO, LO-Net, DeepLO, SuperPoint, SuperGlue, LightGlue,
  LoFTR, QuadTree LoFTR, RoMa, DISK, GeoTransformer, 3DMatch,
  NetVLAD, MixVPR, AnyLoc, Patch-NetVLAD, SeqSLAM, DBoW2, DBoW3,
  Scan Context, Scan Context++, M2DP, ESF, OverlapTransformer,
  OverlapNet, Patchwork, Patchwork++, Removert, LT-mapper, OpenCalib,
  LI-Init, Zhang's method, AprilGrid.
- **Datasets / benchmarks:** KITTI, KITTI-360, KITTI-CARLA, TUM-VI,
  TUM-RGBD, EuRoC MAV, NCLT, Oxford RobotCar, Boreas, nuScenes,
  Waymo Open Dataset, Argoverse / Argoverse 2, Newer College,
  MulRan, Hilti SLAM Challenge, SubT, TartanAir, VIODE, UrbanNav.
- **Paper citations:** author-year strings ("Koide et al. (2023),"
  "Mur-Artal & Tardós (2017)") remain in Latin script, including the
  accent on "Tardós." Paper titles retain their published capitalization.
- **Code identifiers:** any token inside `` ` `` backticks, including
  variable names, function names, file paths, ROS topic / message
  names (`/imu/data`, `sensor_msgs/Imu`), configuration keys.
- **Math symbols:** all LaTeX delimited by `$…$` or `$$…$$`, including
  symbol choice (`\mathbf{K}`, `\xi`, `\boxplus`).
- **URLs and arXiv IDs.**
- **Foreign mathematical / personal names** in existing English form:
  Markov, Bayes, Gauss, Jacobi, Hessian, Kalman, Joseph (form),
  Schur (complement), Cholesky, Mahalanobis, Sampson, Taylor, Horn,
  Rodrigues, Huber, Tukey, Cauchy, Lorentzian, von Mises-Fisher.
