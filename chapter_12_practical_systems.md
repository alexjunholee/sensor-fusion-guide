# Ch.12 — 실전 시스템 & 벤치마크

Ch.2-11에서 센서 모델링부터 상태 추정, odometry, place recognition, 공간 표현까지 개별 알고리즘을 다루었다. 이 챕터에서는 시선을 알고리즘에서 시스템으로 넓혀, 이 기술들이 실제 플랫폼에서 어떻게 조합되는지를 살펴본다.

자율주행, 드론, 핸드헬드 매핑이라는 세 대표 플랫폼에서 센서 퓨전 아키텍처를 분석하고, 시스템을 평가하기 위한 벤치마크와 도구를 소개한다.

---

## 12.1 자율주행 Perception Stack

자율주행은 센서 퓨전이 가장 공격적으로 적용되는 분야다. 생사가 걸린 안전 요구사항 때문에, 단일 센서 실패에도 시스템이 동작해야 한다(redundancy). 여러 센서를 조합하는 것은 선택이 아니라 필수다.

### 12.1.1 Sensor Suite 구성 사례

**Waymo** (5th generation):
- 1 × 장거리 LiDAR (360°, 최대 300m)
- 4 × 단거리 LiDAR (근거리 사각지대 커버)
- 29 × 카메라 (360° 커버, 다양한 화각)
- 6 × radar (장거리 속도 측정)
- IMU, GNSS, wheel encoder

**[nuScenes](https://arxiv.org/abs/1903.11027) 데이터셋 기준 일반적 구성**:
- 1 × spinning LiDAR (32 또는 64 채널)
- 6 × 서라운드 카메라 (360° 커버)
- 5 × radar
- IMU, GNSS

**Tesla (Vision-only approach)**:
- 8 × 카메라만으로 360° 커버 (LiDAR/radar 제거)
- 뉴럴넷 기반 depth estimation으로 3D 인식
- 극도로 공격적인 접근이며, 업계에서 논쟁이 계속됨

### 12.1.2 Production-Level Fusion Pipeline

프로덕션 자율주행 시스템의 전형적인 센서 퓨전 파이프라인:

```
센서 동기화 (HW trigger + PTP)
    ↓
[LiDAR 처리]          [카메라 처리]         [Radar 처리]
- Motion compensation  - Object detection    - Doppler 속도
- Ground segmentation  - Semantic seg        - 장거리 탐지
- 3D object detection  - Lane detection
    ↓                      ↓                     ↓
         Late Fusion / Deep Fusion
    ↓
[Tracking & Prediction]
- Multi-object tracking (Kalman/JPDA)
- Trajectory prediction
    ↓
[Localization]
- HD map matching
- GNSS/IMU/LiDAR 통합
    ↓
[Planning & Control]
```

Late Fusion과 Deep Fusion은 서로 다른 설계 철학을 대표한다.

**Late fusion (전통)** 방식은 각 센서에서 독립적으로 3D bounding box를 검출하고 NMS(Non-Maximum Suppression)로 결합한다. 모듈화와 디버깅이 쉬운 반면, 센서 간 상보성을 충분히 활용하기 어렵다.

**Deep fusion (현대)** 방식은 BEV(Bird's Eye View) 공간에서 여러 센서의 feature를 직접 결합한다. [BEVFusion](https://arxiv.org/abs/2205.13542) (MIT/Nvidia), TransFusion 등이 대표적이며, 센서 간 상보적 정보를 네트워크가 학습으로 활용한다. 단, end-to-end 학습에 대규모 레이블 데이터가 필요하다는 부담이 있다.

```python
# BEV Fusion 개념도 (pseudo-code)

def bev_fusion_pipeline(lidar_points, camera_images, calibrations):
    """
    BEV 공간에서 LiDAR와 카메라 feature를 결합하는 fusion pipeline.
    """
    # 1. LiDAR → BEV feature
    # LiDAR 포인트를 voxelize하고 3D backbone으로 처리한 후
    # z축을 collapse하여 BEV feature map 생성
    lidar_voxels = voxelize(lidar_points, voxel_size=0.1)
    lidar_3d_features = sparse_3d_cnn(lidar_voxels)  # (X, Y, Z, C)
    lidar_bev = lidar_3d_features.max(dim='z')        # (X, Y, C)
    
    # 2. Camera → BEV feature
    # 각 카메라 이미지에서 feature를 추출하고,
    # depth estimation을 거쳐 BEV 공간으로 변환
    camera_features = []
    for img, calib in zip(camera_images, calibrations):
        feat_2d = image_backbone(img)  # (H', W', C)
        depth_dist = depth_net(feat_2d)  # (H', W', D) — depth 확률 분포
        
        # Lift: 2D feature를 3D로 (LSS 방식)
        feat_3d = outer_product(feat_2d, depth_dist)  # (H', W', D, C)
        
        # Splat: 3D feature를 BEV pillar로 합산
        feat_bev = splat_to_bev(feat_3d, calib)  # (X, Y, C)
        camera_features.append(feat_bev)
    
    camera_bev = sum(camera_features)  # 모든 카메라의 BEV feature 합산
    
    # 3. Fusion: BEV 공간에서 concatenate 또는 attention
    fused_bev = concat_and_conv(lidar_bev, camera_bev)  # (X, Y, C')
    
    # 4. Detection head
    detections = detection_head(fused_bev)  # 3D bounding boxes
    
    return detections
```

### 12.1.3 Localization Stack

자율주행 localization은 일반적으로 두 단계로 구성된다:

1. **Global localization**: GNSS (RTK 또는 PPP)로 초기 위치를 맵 위에 배치한다. 도심에서는 multipath 문제로 수 미터 오차가 발생할 수 있으므로, 이것만으로는 부족하다.

2. **Map-relative localization**: HD map(사전 구축 LiDAR 포인트 클라우드 맵)에 현재 LiDAR 스캔을 NDT/ICP로 정합하여 cm 수준 정확도를 달성한다. GNSS 신호가 차단되는 터널 등에서도 동작한다.

Factor graph 기반 통합:
$$\mathbf{x}^* = \arg\min \underbrace{f_{\text{IMU}}}_{\text{예측}} + \underbrace{f_{\text{LiDAR}}}_{\text{맵 정합}} + \underbrace{f_{\text{GNSS}}}_{\text{전역 앵커}} + \underbrace{f_{\text{wheel}}}_{\text{속도}}$$

---

## 12.2 드론/UAV

드론 환경의 센서 퓨전은 자율주행과는 상당히 다른 제약 아래에서 동작한다.

### 12.2.1 Visual-Inertial 중심 시스템

드론에서 가장 많이 사용되는 센서 조합은 카메라 + IMU다. 소형 드론은 무게와 전력 제약 때문에 LiDAR를 탑재하기 어렵고, 카메라와 IMU는 두 제약을 모두 만족한다(Livox Mid-360 같은 소형 solid-state LiDAR가 등장하면서 변화 중이긴 하다). 별도의 과제는 진동이다. 드론의 프로펠러 진동이 IMU 데이터에 노이즈를 추가하므로, 방진 마운트와 소프트웨어 필터링이 함께 필요하다.

대표적 VIO 시스템 for 드론:
- **VINS-Mono/Fusion**: tightly-coupled optimization 기반. PX4와 통합 가능.
- **MSCKF/OpenVINS**: filter 기반. 연산량이 적어 embedded 보드에 적합.
- **Basalt**: visual-inertial mapping with non-linear factor recovery.

### 12.2.2 GPS-Denied Navigation

드론의 핵심 도전은 GPS 신호가 없는 환경 — 실내, 터널, 숲 캐노피 아래, 전자전 환경 — 에서의 자율 비행이다.

접근 방식은 사전 인프라 여부에 따라 갈린다.

1. **VIO 단독**: 단기적으로 안정적이지만 드리프트가 누적된다. 수 분 이내의 단거리 미션에 적합하다.
2. **VIO + 지형 매칭**: 사전 구축된 지형·건물 맵과 현재 카메라 관측을 매칭해 드리프트를 억제한다. prior map이 필요하다.
3. **VIO + UWB**: 환경에 UWB 앵커를 설치하고 ranging 측정으로 드리프트를 보정한다. 인프라 사전 구축이 전제다.
4. **VIO + barometer**: 기압계를 보조 센서로 추가해 z축 드리프트를 잡는다. 별도 인프라 없이 적용 가능하다.

### 12.2.3 실시간 제약

드론은 고속 비행(5~15 m/s)과 급격한 자세 변화(flip, 급선회)를 수행한다. IMU rate는 200~1000 Hz가 필요하며, 고속 모션의 pose 변화를 충분히 캡처해야 한다. 카메라는 모션 블러를 줄이기 위해 짧은 노출 시간이 필요하지만, 이는 저조도에서의 노이즈 증가와 트레이드오프다. state estimation 결과는 30 ms 이내에 제어기에 전달되어야 하며, 지연이 길면 제어 불안정으로 이어진다.

**Point-LIO**는 포인트 단위로 처리하여 스캔 완료를 기다리지 않는 초저지연 LIO다. 드론의 고속 기동에 특히 유리하다.

```python
class DroneVIOConfig:
    """드론용 VIO 시스템 구성 예시."""
    
    # 센서 구성
    camera_fps = 30
    camera_resolution = (640, 480)  # 저해상도로 연산량 절감
    imu_rate = 400  # Hz
    
    # VIO 파라미터
    max_features = 150  # 특징점 수 제한 (연산량)
    keyframe_interval = 5  # 매 5프레임마다 키프레임
    sliding_window_size = 10  # 최적화 윈도우 크기
    
    # 드론 특화 설정
    gravity_magnitude = 9.81
    max_angular_velocity = 10.0  # rad/s — 급격한 회전 대응
    motion_blur_threshold = 0.3  # 블러 심한 프레임 제외
    
    # IMU 노이즈 (드론은 진동이 심하므로 값이 큼)
    gyro_noise_density = 0.004  # rad/s/sqrt(Hz)
    accel_noise_density = 0.05  # m/s^2/sqrt(Hz)
    gyro_random_walk = 0.0002   # rad/s^2/sqrt(Hz)
    accel_random_walk = 0.003   # m/s^3/sqrt(Hz)
    
    # 안전
    max_allowed_drift_m = 0.5   # 이 이상 드리프트 감지 시 경고
    min_tracked_features = 20   # 이하면 tracking quality 경고


def check_image_quality(image, angular_velocity, exposure_time):
    """
    드론 카메라 이미지 품질 검사.
    모션 블러가 심한 프레임은 VIO에서 제외.
    """
    # 모션 블러 추정: 각속도 × 노출 시간 × 초점 거리
    blur_pixels = abs(angular_velocity) * exposure_time * 300  # 대략적 focal length
    
    if blur_pixels > 5.0:  # 5 픽셀 이상 블러
        return False, f"Excessive motion blur: {blur_pixels:.1f} pixels"
    
    # 밝기 검사
    mean_brightness = image.mean()
    if mean_brightness < 20:
        return False, f"Too dark: mean={mean_brightness:.0f}"
    if mean_brightness > 240:
        return False, f"Too bright: mean={mean_brightness:.0f}"
    
    return True, "OK"
```

---

## 12.3 핸드헬드/백팩 매핑

핸드헬드 또는 백팩에 장착된 센서로 환경을 매핑하는 것은 측량(surveying), BIM(Building Information Modeling), 디지털 트윈 구축 등에 널리 사용된다.

### 12.3.1 SLAM as a Service

상용 핸드헬드 매핑 장비의 예:

- **Leica BLK2GO**: 핸드헬드 LiDAR 스캐너. LiDAR + IMU + 카메라 융합으로 실시간 SLAM 수행. 측량 등급(survey-grade) 정확도.
- **NavVis VLX**: 백팩 장착. 4개의 카메라 + LiDAR. 실내 매핑에 특화.
- **GeoSLAM ZEB**: 핸드헬드 모바일 매핑. 2D LiDAR를 수동으로 회전시키며 3D 스캔.

이러한 장비의 공통 파이프라인:

```
LiDAR + IMU → LIO (FAST-LIO2 또는 유사)
     ↓
Loop closure (Scan Context 등)
     ↓
Global optimization (iSAM2)
     ↓
Dense colorized point cloud (카메라 색상 부착)
     ↓
Post-processing (클라우드 정리, mesh 생성)
```

### 12.3.2 Survey-Grade Mapping

측량 등급 매핑에서의 핵심 요구사항:

- **절대 정확도**: GNSS 기준으로 수 cm 이내. 이를 위해 GCP(Ground Control Point)를 배치하고 후처리에서 맞춘다.
- **상대 정확도**: 맵 내부의 일관성. Loop closure와 global optimization이 핵심.
- **포인트 밀도**: 벽면 1 cm 간격 이상의 밀도. 후처리에서 다운샘플링.
- **색상 품질**: 정확한 HDR 색상 매핑. 카메라-LiDAR 시간 동기화와 extrinsic 캘리브레이션이 정밀해야 한다.

센서 퓨전에서의 실전 이슈:

1. **Degenerate environments**: 긴 복도, 빈 방 등 기하학적 특징이 부족한 환경. LiDAR-only에서 발생하는 drift를 카메라 또는 IMU가 보완. R3LIVE, FAST-LIVO2 같은 multi-modal 시스템이 효과적.

2. **다층 건물**: 엘리베이터/계단을 통한 층간 이동 시 loop closure가 필수. GNSS가 없으므로 z축 드리프트가 특히 문제. 기압계가 보조 센서로 유용.

3. **유리/거울**: LiDAR 빔이 투과하거나 반사. 카메라로 보완하거나, 반사 포인트 필터링.

---

## 12.4 벤치마크 & 평가

센서 퓨전 시스템을 공정하게 비교하려면 표준화된 데이터셋과 평가 메트릭이 필요하다.

### 12.4.1 주요 데이터셋

| 데이터셋 | 연도 | 환경 | 센서 | 특징 |
|----------|------|------|------|------|
| **[KITTI](https://doi.org/10.1177/0278364913491297)** | 2012 | 실외 (자율주행) | Stereo, LiDAR, GPS/IMU | SLAM/VO 벤치마크의 시초. 11개 training + 11개 test 시퀀스 |
| **[EuRoC](https://doi.org/10.1177/0278364915620033)** | 2016 | 실내 (MAV) | Stereo, IMU | VIO 벤치마크의 표준. Machine Hall + Vicon Room |
| **[TUM-RGBD](https://doi.org/10.1109/IROS.2012.6385773)** | 2012 | 실내 | RGB-D | Visual SLAM의 기본 벤치마크. Kinect v1 |
| **TUM-VI** | 2018 | 실내+실외 | Stereo, IMU | VIO 벤치마크. 다양한 모션 패턴 |
| **[Hilti](https://arxiv.org/abs/2109.11316)** | 2021~ | 건설 현장 | LiDAR, Camera, IMU | 산업 환경 특화. 도전적 조건 |
| **[HeLiPR](https://arxiv.org/abs/2309.14590)** | 2023 | 실외 (도심) | Heterogeneous LiDAR, Camera, IMU, GNSS | 이종 LiDAR 퓨전 연구용. Ouster+Velodyne+Livox+Aeva |
| **[nuScenes](https://arxiv.org/abs/1903.11027)** | 2020 | 실외 (자율주행) | Camera, LiDAR, Radar, GPS/IMU | 1000개 씬, 23 클래스 3D 어노테이션, 360° 서라운드 센서 |
| **Newer College** | 2020 | 실외+실내 | LiDAR, Camera, IMU | 옥스퍼드 대학 캠퍼스. Multi-session |

각 데이터셋의 특성과 용도:

**KITTI** — 2012년 공개로 역사가 깊지만 여전히 자율주행 SLAM 벤치마크의 표준이다. Velodyne 64채널 LiDAR, 스테레오 카메라, GPS/IMU를 제공한다. 한계도 뚜렷하다. 센서가 오래되었고 시퀀스가 짧다. ground truth는 GPS/INS 기반이라 cm 수준 정확도를 보장하지 않는다.

**EuRoC** — 드론(MAV)에 장착된 스테레오 카메라 + IMU 데이터로, VIO 시스템의 사실상 표준 벤치마크다. Ground truth는 Vicon 모션 캡처(sub-mm 정확도) 또는 Leica 레이저 트래커(mm 정확도)로 제공된다. 11개 시퀀스가 easy → medium → difficult로 분류되어 있다.

**Hilti** — 건설 현장이라는 도전적 환경(먼지, 진동, 반복 구조물)에서의 SLAM을 평가한다. 2021년부터 매년 SLAM 대회를 개최하여 최신 시스템의 한계를 드러내고 있다.

**HeLiPR** — 2023년 공개된 최신 데이터셋으로, 서로 다른 종류의 LiDAR(spinning, solid-state, FMCW)를 동시에 탑재한 것이 핵심 특징이다. 이종 LiDAR 퓨전이라는 새로운 연구 방향을 지원한다.

**Newer College** — 옥스퍼드 대학 캠퍼스를 여러 번 방문하며 수집한 데이터로, multi-session SLAM과 long-term mapping 연구에 적합하다. 핸드헬드 LiDAR로 수집되어 도전적인 모션 패턴을 포함한다.

2022년 이후 새로운 벤치마크가 빠르게 추가되고 있다.

[Hilti-Oxford](https://arxiv.org/abs/2208.09825) (2022)는 mm 수준 ground truth를 제공하는 건설 환경 SLAM 벤치마크로, 매년 SLAM challenge를 개최하며 최신 시스템의 한계를 드러낸다. [Boreas](https://arxiv.org/abs/2203.10168) (Burnett et al. 2023)는 동일 경로를 1년간 반복 주행하여 수집한 자율주행 데이터셋이다. LiDAR·radar·카메라를 포함하며 사계절 조건을 모두 담는다. [Snail-Radar](https://arxiv.org/abs/2407.11705) (Huai et al., IJRR 2025)는 4D radar SLAM 평가를 위한 대규모 벤치마크로, 다양한 환경과 플랫폼에서 4D radar odometry/SLAM을 체계적으로 비교한다.

### 12.4.2 평가 메트릭

**ATE (Absolute Trajectory Error)**: 추정 궤적과 ground truth 궤적의 전역적 차이를 측정한다.

$$\text{ATE} = \sqrt{\frac{1}{N} \sum_{i=1}^{N} \| \text{trans}(\mathbf{T}_{\text{gt},i}^{-1} \cdot \mathbf{T}_{\text{est},i}) \|^2}$$

평가 전에 두 궤적을 Sim(3) 또는 SE(3) 정렬(alignment)해야 한다. Monocular VO는 스케일이 모호하므로 Sim(3), stereo/LiDAR는 SE(3)를 사용한다.

$$\mathbf{S}^* = \arg\min_{\mathbf{S} \in \text{Sim}(3)} \sum_i \| \mathbf{p}_{\text{gt},i} - \mathbf{S} \cdot \mathbf{p}_{\text{est},i} \|^2$$

이 정렬은 Umeyama algorithm으로 closed-form 해를 구할 수 있다.

**RPE (Relative Pose Error)**: 짧은 구간에서의 상대적 정확도를 측정한다. 드리프트 성향을 나타낸다.

$$\text{RPE}(\Delta) = \sqrt{\frac{1}{M} \sum_{i=1}^{M} \| \text{trans}((\mathbf{T}_{\text{gt},i}^{-1} \mathbf{T}_{\text{gt},i+\Delta})^{-1} (\mathbf{T}_{\text{est},i}^{-1} \mathbf{T}_{\text{est},i+\Delta})) \|^2}$$

$\Delta$는 평가 구간 (프레임 수 또는 거리). 짧은 $\Delta$에서의 RPE는 odometry 정확도를, 긴 $\Delta$에서의 RPE는 드리프트를 반영한다.

Place recognition에는 별도의 메트릭이 사용된다. **Recall@N**은 상위 N개 후보 중 올바른 장소가 포함된 비율이며, Recall@1이 가장 엄격한 기준이다. **Precision-Recall curve**는 threshold에 따른 precision·recall 트레이드오프를 보여주고, **AUC**는 그 커브 아래 면적으로 전체 성능을 단일 수치로 요약한다.

```python
import numpy as np
from scipy.spatial.transform import Rotation

def compute_ate(poses_gt, poses_est, align='se3'):
    """
    Absolute Trajectory Error 계산.
    
    Args:
        poses_gt: ground truth poses [(4, 4), ...] 리스트
        poses_est: estimated poses [(4, 4), ...] 리스트
        align: 'se3' 또는 'sim3'
        
    Returns:
        ate_rmse: ATE RMSE (미터)
        ate_per_frame: 각 프레임의 ATE (N,)
    """
    positions_gt = np.array([T[:3, 3] for T in poses_gt])  # (N, 3)
    positions_est = np.array([T[:3, 3] for T in poses_est])  # (N, 3)
    
    # Umeyama alignment
    if align == 'sim3':
        S, R, t = umeyama_alignment(positions_est, positions_gt, 
                                     with_scale=True)
        positions_aligned = S * (R @ positions_est.T).T + t
    else:  # se3
        _, R, t = umeyama_alignment(positions_est, positions_gt, 
                                     with_scale=False)
        positions_aligned = (R @ positions_est.T).T + t
    
    # 프레임별 오차
    errors = np.linalg.norm(positions_gt - positions_aligned, axis=1)
    
    ate_rmse = np.sqrt(np.mean(errors ** 2))
    
    return ate_rmse, errors


def compute_rpe(poses_gt, poses_est, delta=10):
    """
    Relative Pose Error 계산.
    
    Args:
        poses_gt: ground truth poses 리스트
        poses_est: estimated poses 리스트
        delta: 평가 구간 (프레임 수)
        
    Returns:
        rpe_trans: RPE 병진 RMSE (미터)
        rpe_rot: RPE 회전 RMSE (도)
    """
    trans_errors = []
    rot_errors = []
    
    for i in range(len(poses_gt) - delta):
        # Ground truth 상대 변환
        T_gt_rel = np.linalg.inv(poses_gt[i]) @ poses_gt[i + delta]
        
        # Estimated 상대 변환
        T_est_rel = np.linalg.inv(poses_est[i]) @ poses_est[i + delta]
        
        # 오차
        T_error = np.linalg.inv(T_gt_rel) @ T_est_rel
        
        # 병진 오차
        trans_err = np.linalg.norm(T_error[:3, 3])
        trans_errors.append(trans_err)
        
        # 회전 오차
        rot = Rotation.from_matrix(T_error[:3, :3])
        rot_err = np.linalg.norm(rot.as_rotvec()) * 180 / np.pi  # 도 단위
        rot_errors.append(rot_err)
    
    rpe_trans = np.sqrt(np.mean(np.array(trans_errors) ** 2))
    rpe_rot = np.sqrt(np.mean(np.array(rot_errors) ** 2))
    
    return rpe_trans, rpe_rot


def umeyama_alignment(source, target, with_scale=True):
    """
    Umeyama alignment: source를 target에 정렬하는 
    최적 similarity/rigid 변환 계산.
    
    Args:
        source: (N, 3) 원본 포인트
        target: (N, 3) 대상 포인트
        with_scale: True면 Sim(3), False면 SE(3)
        
    Returns:
        scale: 스케일 (with_scale=False면 1.0)
        rotation: (3, 3) 회전 행렬
        translation: (3,) 병진 벡터
    """
    n = source.shape[0]
    
    # 중심 이동
    mu_source = np.mean(source, axis=0)
    mu_target = np.mean(target, axis=0)
    
    source_centered = source - mu_source
    target_centered = target - mu_target
    
    # 공분산 행렬
    sigma_source = np.sum(source_centered ** 2) / n
    H = (target_centered.T @ source_centered) / n
    
    # SVD
    U, D, Vt = np.linalg.svd(H)
    
    # 반사 보정
    d = np.linalg.det(U) * np.linalg.det(Vt)
    S = np.diag([1, 1, np.sign(d)])
    
    rotation = U @ S @ Vt
    
    if with_scale:
        scale = np.trace(np.diag(D) @ S) / sigma_source
    else:
        scale = 1.0
    
    translation = mu_target - scale * rotation @ mu_source
    
    return scale, rotation, translation
```

### 12.4.3 공정한 비교의 어려움

벤치마크 결과를 해석할 때 주의할 점:

1. **파라미터 튜닝**: 같은 알고리즘도 파라미터에 따라 성능이 크게 달라진다. 특정 데이터셋에 맞춰 튜닝하면 범용성이 떨어진다.

2. **하드웨어 의존성**: 실시간 성능은 하드웨어에 크게 좌우된다. "실시간"의 정의가 논문마다 다르다 (데스크톱 GPU vs 임베디드 ARM).

3. **Completeness**: 일부 시스템은 어려운 시퀀스에서 tracking loss가 발생하는데, 성공한 구간만으로 ATE를 계산하면 실패 빈도가 반영되지 않는다. **Completeness** (= 성공한 시퀀스 비율)도 함께 보고해야 한다.

4. **초기화 차이**: VIO 시스템의 초기화 방법과 시간이 다르면, 같은 시퀀스에서도 결과가 달라진다.

5. **Loop closure 포함 여부**: VO (loop closure 없음) vs SLAM (loop closure 있음)을 구분해야 한다. Loop closure가 있으면 ATE가 극적으로 좋아질 수 있다.

---

## 12.5 오픈소스 도구 가이드

센서 퓨전 연구와 실무에서 자주 쓰는 오픈소스 도구들을 묶었다.

### 12.5.1 최적화 라이브러리

**GTSAM** (Georgia Tech Smoothing and Mapping):
- Factor graph 기반 최적화 라이브러리
- iSAM2 구현 포함
- C++ with Python bindings (gtsam)
- LIO-SAM 등 많은 SLAM 시스템의 백엔드
- 장점: factor graph를 직관적으로 구성 가능, 다양한 factor 타입 내장

```python
# GTSAM Python 사용 예시 (개념 코드)

import gtsam
import numpy as np

def simple_pose_graph_gtsam():
    """GTSAM으로 간단한 pose graph optimization."""
    
    # Factor graph 생성
    graph = gtsam.NonlinearFactorGraph()
    
    # 초기값
    initial = gtsam.Values()
    
    # Prior factor: 첫 번째 pose 고정
    prior_noise = gtsam.noiseModel.Diagonal.Sigmas(
        np.array([0.01, 0.01, 0.01, 0.01, 0.01, 0.01])  # (rx, ry, rz, tx, ty, tz)
    )
    graph.add(gtsam.PriorFactorPose3(
        0, gtsam.Pose3(), prior_noise
    ))
    initial.insert(0, gtsam.Pose3())
    
    # Odometry factors
    odom_noise = gtsam.noiseModel.Diagonal.Sigmas(
        np.array([0.05, 0.05, 0.05, 0.1, 0.1, 0.1])
    )
    
    # Pose 1: 1m 전진
    T_01 = gtsam.Pose3(gtsam.Rot3(), gtsam.Point3(1.0, 0.0, 0.0))
    graph.add(gtsam.BetweenFactorPose3(0, 1, T_01, odom_noise))
    initial.insert(1, gtsam.Pose3(gtsam.Rot3(), gtsam.Point3(1.0, 0.1, 0.0)))
    
    # Pose 2: 90도 좌회전 + 1m 전진
    T_12 = gtsam.Pose3(
        gtsam.Rot3.Rz(np.pi / 2), 
        gtsam.Point3(1.0, 0.0, 0.0)
    )
    graph.add(gtsam.BetweenFactorPose3(1, 2, T_12, odom_noise))
    initial.insert(2, gtsam.Pose3(
        gtsam.Rot3.Rz(np.pi / 2), 
        gtsam.Point3(1.0, 1.1, 0.0)
    ))
    
    # Loop closure: pose 2 → pose 0
    loop_noise = gtsam.noiseModel.Diagonal.Sigmas(
        np.array([0.1, 0.1, 0.1, 0.2, 0.2, 0.2])
    )
    T_20 = gtsam.Pose3(
        gtsam.Rot3.Rz(np.pi / 2),
        gtsam.Point3(0.0, -1.0, 0.0)
    )
    graph.add(gtsam.BetweenFactorPose3(2, 0, T_20, loop_noise))
    
    # iSAM2로 최적화
    params = gtsam.ISAM2Params()
    isam = gtsam.ISAM2(params)
    isam.update(graph, initial)
    result = isam.calculateEstimate()
    
    # 결과 출력
    for i in range(3):
        pose = result.atPose3(i)
        print(f"Pose {i}: x={pose.x():.3f}, y={pose.y():.3f}, "
              f"z={pose.z():.3f}")
    
    return result
```

**Ceres Solver**:
- Google이 개발한 nonlinear least squares 최적화 라이브러리
- C++ 전용 (Python 바인딩은 제한적)
- 자동 미분(automatic differentiation) 지원이 핵심 장점
- VINS-Mono, ORB-SLAM 등에서 사용
- Factor graph 추상화 없이 순수 최적화 문제를 직접 정의

**g2o** (General Graph Optimization):
- Kümmerle et al. (2011) 개발
- C++ 라이브러리, 그래프 최적화 특화
- 다양한 vertex/edge 타입 사전 정의 (SE2, SE3, Sim3 등)
- GTSAM보다 가볍고 빠르지만, 유연성은 떨어짐

세 라이브러리 비교:

| 특성 | GTSAM | Ceres | g2o |
|------|-------|-------|-----|
| 추상화 수준 | Factor graph | Cost function | Graph vertex/edge |
| 자동 미분 | 부분 지원 | 완전 지원 | 미지원 |
| Incremental | iSAM2 | 미지원 | 미지원 |
| Python 지원 | 양호 | 제한적 | 제한적 |
| 대표 사용처 | LIO-SAM | VINS-Mono | ORB-SLAM |

### 12.5.2 캘리브레이션 도구

**Kalibr** (ethz-asl):
- Camera-IMU, Camera-Camera, multi-IMU 캘리브레이션
- Continuous-time B-spline trajectory 기반
- AprilGrid 타겟 사용
- 사실상 표준이지만 설치가 까다로움 (ROS 의존)

**OpenCalib** (2023):
- 자율주행 전체 센서 스택의 통합 캘리브레이션
- Camera, LiDAR, Radar, IMU 간 모든 조합 지원
- Target-based + Targetless 모두 포함

**[direct_visual_lidar_calibration](https://arxiv.org/abs/2302.05094)** (Koide et al. 2023):
- NID 기반 targetless LiDAR-카메라 캘리브레이션
- SuperGlue로 초기 추정, NID로 정밀 정합
- 단일 촬영만으로 동작

### 12.5.3 평가 도구

**evo** (MH Grupp):
- Python 기반 궤적 평가 도구
- ATE, RPE 계산 및 시각화
- TUM, KITTI, EuRoC 등 다양한 형식 지원
- 명령행 도구와 Python API 모두 제공

```bash
# evo 사용 예시
# ATE 계산
evo_ape tum groundtruth.txt estimated.txt -va --plot --plot_mode xz

# RPE 계산
evo_rpe tum groundtruth.txt estimated.txt -va --delta 100 --delta_unit f

# 두 시스템 비교
evo_traj tum system_a.txt system_b.txt --ref groundtruth.txt -p --plot_mode xz
```

```python
# evo Python API 사용 예시

from evo.core import metrics, sync
from evo.core.trajectory import PosePath3D, PoseTrajectory3D
from evo.tools import file_interface
import numpy as np

def evaluate_trajectory(gt_file, est_file, align=True):
    """
    evo를 이용한 궤적 평가.
    
    Args:
        gt_file: ground truth 파일 경로 (TUM 형식)
        est_file: estimated 궤적 파일 경로
        align: SE(3) 정렬 수행 여부
    """
    # 궤적 로드
    traj_gt = file_interface.read_tum_trajectory_file(gt_file)
    traj_est = file_interface.read_tum_trajectory_file(est_file)
    
    # 타임스탬프 동기화
    traj_gt, traj_est = sync.associate_trajectories(traj_gt, traj_est)
    
    # ATE 계산
    ate_metric = metrics.APE(metrics.PoseRelation.translation_part)
    
    if align:
        # Umeyama alignment
        traj_est_aligned = traj_est.align(traj_gt, correct_scale=False)
        ate_metric.process_data((traj_gt, traj_est_aligned))
    else:
        ate_metric.process_data((traj_gt, traj_est))
    
    stats = ate_metric.get_all_statistics()
    
    print(f"ATE RMSE: {stats['rmse']:.4f} m")
    print(f"ATE Mean: {stats['mean']:.4f} m")
    print(f"ATE Median: {stats['median']:.4f} m")
    print(f"ATE Max: {stats['max']:.4f} m")
    
    return stats


def compare_systems(gt_file, system_files, system_names):
    """여러 시스템의 ATE를 비교."""
    traj_gt = file_interface.read_tum_trajectory_file(gt_file)
    
    results = {}
    for name, est_file in zip(system_names, system_files):
        traj_est = file_interface.read_tum_trajectory_file(est_file)
        traj_gt_sync, traj_est_sync = sync.associate_trajectories(
            traj_gt, traj_est
        )
        
        traj_est_aligned = traj_est_sync.align(
            traj_gt_sync, correct_scale=False
        )
        
        ate = metrics.APE(metrics.PoseRelation.translation_part)
        ate.process_data((traj_gt_sync, traj_est_aligned))
        
        results[name] = ate.get_all_statistics()
    
    # 비교 테이블 출력
    print(f"{'System':<20} {'RMSE (m)':<12} {'Mean (m)':<12} {'Max (m)':<12}")
    print("-" * 56)
    for name, stats in results.items():
        print(f"{name:<20} {stats['rmse']:<12.4f} "
              f"{stats['mean']:<12.4f} {stats['max']:<12.4f}")
    
    return results
```

### 12.5.4 기타 필수 도구

**ROS 2** (Robot Operating System):
- 센서 퓨전 시스템의 통합 프레임워크
- 센서 드라이버, 시간 동기화, 메시지 전달 인프라 제공
- 대부분의 오픈소스 SLAM 시스템이 ROS 패키지로 나옴

**Open3D**:
- 3D 데이터 처리 Python 라이브러리
- Point cloud, mesh, TSDF 처리
- ICP, RANSAC, FPFH 등 기하학 알고리즘 내장
- 시각화 기능 우수

**CloudCompare**:
- 포인트 클라우드 비교/편집 GUI 도구
- 두 포인트 클라우드의 거리 비교(C2C, C2M)
- 포인트 클라우드 정합, 필터링, 다운샘플링

**COLMAP**:
- Structure from Motion (SfM) + Multi-View Stereo (MVS) 파이프라인
- 이미지 집합으로 3D 재구성
- 센서 퓨전에서 camera intrinsic 추정이나 ground truth 맵 구축에 쓴다

---

이 챕터에서 살펴본 실전 시스템과 벤치마크는 Ch.2-11의 이론이 실제 제품과 연구에서 어떻게 작동하는지를 보여준다. 마지막 챕터에서는 아직 성숙하지 않았지만 분야의 방향을 바꿀 수 있는 **연구 프런티어**(Foundation model, event camera, 4D radar, end-to-end SLAM)를 본다.
