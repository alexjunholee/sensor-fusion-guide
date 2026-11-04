# Ch.12 — 실전 시스템 & 벤치마크

센서 모델링, 상태 추정, odometry, place recognition, 공간 표현은 실제 플랫폼에서 한 시스템으로 묶인다. 센서 퓨전을 많이 쓰는 자율주행차, 드론, 핸드헬드 매핑 장비가 그런 플랫폼이고, 이런 시스템의 성능은 표준 벤치마크와 평가 도구로 잰다.

---

## 12.1 플랫폼별 시스템 설계

### 12.1.1 자율주행 (Autonomous Driving)

자율주행차는 센서 하나가 고장 나도 그 고장을 알아채고, 잘못된 값이 뒤 단계로 퍼지지 않게 막아야 한다. 고장이 났을 때 계속 달릴지, 최소 위험 상태(minimal risk condition)로 넘어가 멈출지는 운행 설계 영역(ODD, Operational Design Domain)과 안전 논증(safety case)에 따라 정한다. 종류가 다른 센서를 함께 달면 한 센서가 고장 나도 다른 센서가 그 역할을 이어받고, 센서끼리 값을 맞춰 보며 서로 검증할 수 있다.

#### Sensor Suite 구성 사례

**Waymo** (5th generation):
- 1 × 장거리 LiDAR (360°, 300m 넘는 거리까지)
- 4 × 단거리 perimeter LiDAR (차체 네 곳, 가까운 사각지대)
- 29 × 카메라 (360°, 화각이 겹치게 배치)
- radar (속도 측정. Waymo 발표문은 개수를 밝히지 않음)
- IMU, GNSS, wheel encoder

**[nuScenes](https://arxiv.org/abs/1903.11027) 수집 차량**:
- 1 × 32-beam spinning LiDAR
- 6 × 서라운드 카메라 (360° 커버)
- 5 × radar
- IMU, GNSS

**Camera-centric production example**:
- 여러 대의 surround camera와 신경망 인식을 중심에 둔다.
- 어떤 센서를 달고 radar를 쓰는지는 차량 세대, 판매 시장, 시기마다 다르다. 특정 회사 구성을 적을 때는 세대와 시점을 함께 밝힌다.

### 12.1.2 Production-Level Fusion Pipeline

양산 자율주행 시스템의 센서 퓨전 파이프라인은 대체로 아래와 같다. 화살표는 처리 순서를 대략 보여 줄 뿐이고, 실제 모듈은 이보다 더 얽혀 있다. 예를 들어 위치 추정은 GNSS·IMU·LiDAR·wheel 관측으로 인식과 동시에 돌아가고, 그 결과를 인식·예측·계획이 모두 가져다 쓴다.

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

**Late fusion**은 센서마다 따로 3D bounding box를 검출한 뒤 NMS(Non-Maximum Suppression, 겹치는 박스 중 점수가 가장 높은 것만 남기는 방법) 같은 방법으로 합친다. 모듈을 나누기 쉽고 디버깅도 쉽다. 그러나 각 센서가 이미 판단을 끝낸 결과만 합치므로, 한 센서의 약점을 다른 센서의 원시 정보로 메우기 어렵다.

**Deep fusion**은 센서의 feature를 네트워크 안에서 합치고, 합치는 방법을 학습한다. 어디서 합치는지는 방법마다 다르다. MIT의 [BEVFusion](https://arxiv.org/abs/2205.13542)은 카메라와 LiDAR feature를 같은 BEV(bird's-eye view, 위에서 내려다본 평면) 격자로 옮겨 그 위에서 합친다. TransFusion은 LiDAR BEV feature로 만든 object query가 transformer decoder에서 이미지 feature에 cross-attention을 건다. 이렇게 하면 한 센서의 약점을 다른 센서의 정보로 메우는 방법을 네트워크가 스스로 배운다. 대신 end-to-end로 학습하려면 레이블을 붙인 데이터가 많이 필요하다.

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

자율주행 localization은 보통 두 단계로 나뉜다.

1. **Global localization**: GNSS(RTK나 PPP)로 차량이 지도 위 어디쯤 있는지 처음 위치를 잡는다. 도심에서는 위성 신호가 건물에 반사되는 multipath 때문에 오차가 몇 미터까지 커질 수 있어, 이것만으로는 모자란다.

2. **Map-relative localization**: 지금 들어온 LiDAR scan을 미리 만든 HD point-cloud map에 NDT나 ICP로 맞춘다. 지도가 정확하고, scan과 지도가 충분히 겹치고, 주변 형상이 특징적이고, 초기값이 가까우면 GNSS가 약한 구간에서도 위치를 잡아 준다. cm 단위 정확도가 나오는지는 그 지도와 평가 조건에서 직접 확인해야 한다.

이 관측들을 factor graph 하나로 묶으면 다음과 같다.
$$\mathbf{x}^* = \arg\min \underbrace{f_{\text{IMU}}}_{\text{예측}} + \underbrace{f_{\text{LiDAR}}}_{\text{맵 정합}} + \underbrace{f_{\text{GNSS}}}_{\text{전역 앵커}} + \underbrace{f_{\text{wheel}}}_{\text{속도}}$$

---

## 12.2 드론/UAV

### 12.2.1 Visual-Inertial 중심 시스템

드론에는 카메라와 IMU를 함께 다는 경우가 많다. 소형 드론은 싣는 무게와 쓰는 전력이 빠듯해서 LiDAR를 달기 어렵고, 카메라와 IMU는 가볍고 전력도 적게 먹는다. Livox Mid-360처럼 작은 LiDAR를 다는 드론도 있다. 드론에서는 진동도 문제다. 프로펠러 진동이 IMU 측정에 그대로 실리므로 방진 마운트를 달고 소프트웨어 필터도 함께 쓴다.

드론에 쓰는 오픈소스 VIO의 예는 다음과 같다.
- **VINS-Mono/Fusion**: tightly-coupled 최적화 방식. PX4와 연동해 쓸 수 있다.
- **MSCKF/OpenVINS**: 필터 방식. 연산량이 적어 임베디드 보드에 맞다.
- **Basalt**: 논문 제목은 "Visual-Inertial Mapping with Non-Linear Factor Recovery"(Usenko et al., RA-L 2020)이다. VIO에서 얻은 정보를 비선형 factor로 요약해 전역 지도 최적화에 다시 쓴다.

### 12.2.2 GPS-Denied Navigation

실내, 터널, 우거진 숲 아래, 전파 교란을 받는 곳에서는 GPS 신호를 쓸 수 없다. 이런 곳에서 드론을 자율 비행시키는 방법은 미리 깔아 둔 지도나 장비가 있는지에 따라 달라진다.

1. **VIO 단독**: 드리프트가 쌓이므로 얼마나 오래 날 수 있는지는 움직임, 주변 텍스처, 캘리브레이션 품질, loop closure 유무, 허용 오차에 따라 달라진다.
2. **VIO + 지형 매칭**: 지금 카메라에 보이는 장면을 미리 만든 지형·건물 지도(prior map)와 맞춰 드리프트를 누른다.
3. **VIO + UWB**: 비행 구역에 UWB 앵커를 미리 설치하고, 앵커까지 잰 거리로 드리프트를 바로잡는다.
4. **VIO + 기압계**: 기압 변화로 상대 고도를 잰다. 날씨, 온도, 프로펠러 바람(prop wash) 때문에 바이어스가 생기므로 고도 드리프트를 확실히 줄이려면 다른 고도 기준과 함께 쓴다.

### 12.2.3 실시간 제약

센서를 몇 Hz로 읽고 지연을 얼마까지 허용할지는 기체의 최대 각가속도, 제어 대역폭, 카메라 노출 시간, 통신 지연을 함께 보고 정한다. IMU는 에일리어싱이 생기지 않고 적분 오차가 덜 쌓이도록 충분히 빠르게 샘플링해야 한다. 카메라 노출 시간은 길면 모션 블러가 생기고 짧으면 어두운 곳에서 노이즈가 늘어서, 그 사이에서 고른다. 센서 입력에서 상태 추정값이 나오기까지의 지연을 얼마나 허용할지도 폐루프 제어 안정성을 분석하고 비행 시험을 해 보며 정한다.

**Point-LIO**(He et al., Advanced Intelligent Systems 2023)는 스캔 한 장이 다 모이기를 기다리지 않고 점이 들어올 때마다 상태를 갱신하는 LIO다. 이렇게 하면 스캔 안의 모션 왜곡이 생기지 않는다. 논문 실험에서는 odometry를 4~8 kHz로 냈고, IMU 측정 범위를 넘는 75 rad/s 회전에서도 추정이 버텼다. 논문은 레이싱 드론에 적용한 예도 실었다.

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

손에 들거나 등에 메는 센서로 주변을 매핑하는 방법은 측량, BIM(Building Information Modeling, 건물 정보 모델), 디지털 트윈 제작에 쓰인다.

### 12.3.1 SLAM as a Service

상용 핸드헬드 매핑 장비의 예:

- **Leica BLK2GO**: 손에 드는 LiDAR 스캐너. LiDAR, IMU, 카메라를 합쳐 실시간으로 SLAM을 돌린다. 제조사가 밝힌 정확도는 스캔 길이, 루프를 닫았는지, 어떻게 검증했는지 같은 조건에서 나온 값이다. 실제 발주 규격은 §12.3.2의 기준으로 따로 확인한다.
- **NavVis VLX**: 몸 앞과 머리 위에 센서가 오도록 걸치는 웨어러블 장비. VLX 3 데이터시트 기준으로 32채널 LiDAR 2대와 카메라 4대를 단다. 실내와 실외에서 모두 쓴다.
- **GeoSLAM ZEB**: 손에 드는 모바일 매핑 장비. 초기 모델(ZEB1)은 2D LiDAR를 스프링에 달아 걸을 때 흔들리게 해서 3D를 훑었다. ZEB-REVO부터는 모터로 헤드를 돌린다.

세 제품 모두 내부 알고리즘을 공개하지 않았다. 아래 흐름은 이런 장비가 하는 일을 단계별로 나눈 것이고, 단계마다 적은 방법은 공개된 방법 가운데 고른 예다.

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

측량용 매핑에서는 발주 규격과 검증 절차부터 정한다.

- **절대 정확도**: 수평·수직 오차 한계와 신뢰수준을 정하고, 계산에 쓰지 않은 별도 check point로 검증한다. 기준점은 현장에 맞춰 GNSS, total station, GCP(지상 기준점) 가운데서 고른다.
- **상대 정확도**: 지도 안의 점들이 서로 얼마나 잘 맞는지 본다. Loop closure와 전역 최적화가 이를 맞춘다.
- **포인트 밀도**: 대상 크기와 납품물에 필요한 점 간격과 빠짐없는 정도를 정한다. 1 cm 같은 값을 요구하는 현장도 있지만 어디서나 통하는 기준은 아니다.
- **색상 품질**: 점에 HDR 색을 정확히 입히려면 카메라와 LiDAR의 시간 동기화와 extrinsic 캘리브레이션이 정밀해야 한다.

이런 장비로 매핑할 때 센서 퓨전이 부딪히는 문제는 다음과 같다.

1. **Degenerate environment(퇴화 환경)**: 긴 복도나 빈 방처럼 형상 특징이 부족한 곳. LiDAR만 쓰면 이런 곳에서 드리프트가 생기는데, 카메라나 IMU가 이를 메운다. R3LIVE, FAST-LIVO2 같은 LiDAR-카메라-IMU 시스템이 이런 경우를 겨냥해 만들어졌다.

2. **다층 건물**: 실내라 GNSS가 없어서 층을 오르내릴 때 높이 방향 드리프트가 특히 크다. 같은 층을 다시 지나가는 경로라면 loop closure로 이 오차를 크게 줄인다. 한 번 지나가고 마는 경로에는 닫을 루프가 없으므로 기압계, 층고·도면 정보, 기준점 같은 다른 관측으로 높이를 붙잡는다.

3. **유리와 거울**: LiDAR 빔이 유리를 뚫고 지나가거나 거울에 반사돼 엉뚱한 점이 생긴다. 카메라로 보완하거나 반사된 점을 걸러 낸다.

---

## 12.4 벤치마크 & 평가

### 12.4.1 주요 데이터셋

| 데이터셋 | 연도 | 환경 | 센서 | 특징 |
|----------|------|------|------|------|
| **[KITTI](https://doi.org/10.1177/0278364913491297)** | 2012 | 실외 (자율주행) | Stereo, LiDAR, GPS/IMU | Odometry 평가용 11개 training + 11개 test 시퀀스 |
| **[EuRoC](https://doi.org/10.1177/0278364915620033)** | 2016 | 실내 (MAV) | Stereo, IMU | Machine Hall + Vicon Room의 VIO 시퀀스 |
| **[TUM-RGBD](https://doi.org/10.1109/IROS.2012.6385773)** | 2012 | 실내 | RGB-D | Kinect v1 기반 Visual SLAM 시퀀스와 평가 도구 |
| **TUM-VI** | 2018 | 실내+실외 | Stereo, IMU | VIO 벤치마크. 다양한 모션 패턴 |
| **[Hilti](https://arxiv.org/abs/2109.11316)** | 2021–2023 challenge editions | 건설 현장 | LiDAR, Camera, IMU | 산업 환경 특화. 도전적 조건 |
| **[HeLiPR](https://arxiv.org/abs/2309.14590)** | 2023 | 실외 (도심) | 이종 LiDAR 4종, IMU, GNSS/INS | 서로 다른 LiDAR 사이의 장소 인식 연구용. Ouster OS2-128, Velodyne VLP-16, Livox Avia, Aeva Aeries II. 논문의 센서 목록에 카메라는 없다 |
| **[nuScenes](https://arxiv.org/abs/1903.11027)** | 2020 | 실외 (자율주행) | Camera, LiDAR, Radar, GPS/IMU | 1000개 씬, 23 클래스 3D 어노테이션, 360° 서라운드 센서 |
| **Newer College** | 2020 | 실외 (캠퍼스) | LiDAR, Camera, IMU | 옥스퍼드 New College 구내. 핸드헬드로 수집, 루프가 많다 |

각 데이터셋의 특성과 용도:

**KITTI** — 2012년에 공개된 자율주행 odometry·SLAM 데이터셋이다. Velodyne 64채널 LiDAR, 스테레오 카메라, GPS/IMU 데이터가 들어 있다. 2011년에 녹화한 데이터라 센서가 오래됐다. ground truth는 GPS/INS(OXTS RT3003)로 만들어서, 위성 신호가 나쁜 구간에서는 오차가 cm 수준을 넘을 수 있다.

**EuRoC** — 드론(MAV)에 단 스테레오 카메라와 IMU로 모은 데이터셋으로, ORB-SLAM3와 DROID-SLAM이 이 데이터셋으로 결과를 보고했다(Ch.6). Ground truth는 Vicon Room에서는 Vicon motion capture로, Machine Hall에서는 Leica laser tracker로 잡았다. 시퀀스 11개가 easy, medium, difficult로 나뉜다.

**Hilti** — 건설 현장에서 SLAM이 얼마나 버티는지 겨루는 챌린지다. 2021년, 2022년, 2023년에 열렸고, 해마다 센서 구성과 평가 규칙이 달라서 세부 사항은 해당 연도 논문을 봐야 한다.

**HeLiPR** — 2023년에 공개됐다. 회전식 LiDAR 2종, 비반복 스캔 LiDAR(Livox Avia), FMCW LiDAR(Aeva Aeries II)를 한 차량에 함께 달았다. 서로 다른 LiDAR로 찍은 scan끼리 같은 장소를 알아보는 inter-LiDAR place recognition 연구에 쓴다. 같은 장소를 약 2주 간격으로 다시 달렸고, MulRan과 겹치는 경로도 있다.

**Newer College** — 옥스퍼드 New College 구내를 손에 든 장비로 걸으며 모았다. 같은 구역을 여러 번 지나가서 loop closure를 평가하기 좋고, 사람이 들고 걸었기 때문에 흔들림이 큰 움직임도 들어 있다. 계절이 바뀌는 장기 데이터가 필요하면 Boreas 같은 데이터셋을 쓴다.

2022년 이후에도 새 벤치마크가 계속 나왔다. [Hilti-Oxford](https://arxiv.org/abs/2208.09825)(2022)는 건설 현장 SLAM 데이터셋으로, ground truth 정확도가 mm 수준이다. 2022년 Hilti 챌린지에 쓰였다. [Boreas](https://arxiv.org/abs/2203.10168)(Burnett et al., IJRR 2023)는 같은 경로를 1년 동안 되풀이해 달리며 모은 자율주행 데이터셋이다. LiDAR, radar, 카메라 데이터가 들어 있고 계절 변화와 함께 비·눈 같은 날씨, 밤 주행까지 담았다. [SNAIL Radar](https://arxiv.org/abs/2407.11705)(Huai et al., IJRR 2025)는 4D radar SLAM을 평가하려고 만든 데이터셋이다. 손에 든 장비, 전기자전거, SUV 세 가지로 우한대학 캠퍼스와 터널이 있는 고속도로를 달리며 모았고, 기존 radar odometry 3종과 place recognition 3종을 이 데이터로 비교한 결과도 실었다.

### 12.4.2 평가 메트릭

**ATE (Absolute Trajectory Error)**: 추정한 궤적이 ground truth 궤적과 전체적으로 얼마나 떨어져 있는지 잰다.

$$\text{ATE} = \sqrt{\frac{1}{N} \sum_{i=1}^{N} \| \text{trans}(\mathbf{T}_{\text{gt},i}^{-1} \cdot \mathbf{T}_{\text{est},i}) \|^2}$$

ATE를 계산하기 전에는 보통 두 궤적을 정렬(alignment)한다. 정렬에서 몇 자유도를 맞출지는 그 시스템이 관측할 수 없는 자유도와 벤치마크 규약에 따라 정한다. 스케일을 알 수 없는 단안 카메라만 쓰면 Sim(3)로, 스케일을 아는 stereo·LiDAR·단안+IMU면 SE(3)로 정렬한다. KITTI odometry는 공식 지표로 ATE 대신 구간별 상대 오차를 쓴다. §12.3.2의 측량처럼 정렬하지 않고 절대 좌표 오차를 그대로 재는 경우도 있다.

$$\mathbf{S}^* = \arg\min_{\mathbf{S} \in \text{Sim}(3)} \sum_i \| \mathbf{p}_{\text{gt},i} - \mathbf{S} \cdot \mathbf{p}_{\text{est},i} \|^2$$

이 정렬 문제는 Umeyama 알고리즘(1991)으로 닫힌 해를 바로 구한다.

**RPE (Relative Pose Error)**: 일정 간격만큼 떨어진 두 시점 사이의 상대 움직임이 얼마나 정확한지 잰다. 시간이 갈수록 오차가 얼마나 쌓이는지 보여 준다.

$$\text{RPE}(\Delta) = \sqrt{\frac{1}{M} \sum_{i=1}^{M} \| \text{trans}((\mathbf{T}_{\text{gt},i}^{-1} \mathbf{T}_{\text{gt},i+\Delta})^{-1} (\mathbf{T}_{\text{est},i}^{-1} \mathbf{T}_{\text{est},i+\Delta})) \|^2}$$

위 식의 $\Delta$는 프레임 번호 간격이다. $\Delta$가 짧으면 RPE는 odometry의 국소 정확도를 보여 주고, 길면 그동안 쌓인 드리프트를 보여 준다. KITTI처럼 이동 거리 구간(100~800 m)으로 평가하려면, 각 시작 프레임에서 이동 거리가 목표값에 이르는 프레임을 먼저 찾아 짝지어야 한다.

Place recognition은 다른 지표로 잰다. Recall@N은 상위 N개 후보 안에 맞는 장소가 들어 있는 질의의 비율이다. N이 1이면 첫 후보가 맞아야 하므로 가장 엄격하다. Precision-Recall curve는 판정 threshold를 바꿔 가며 precision과 recall이 어떻게 맞바뀌는지 보여 준다. AUC는 이 곡선 아래 넓이로, 성능을 숫자 하나로 요약한다.

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

벤치마크 숫자를 읽을 때는 다음을 확인한다.

1. **파라미터 튜닝**: 같은 알고리즘도 파라미터에 따라 성능이 크게 바뀐다. 특정 데이터셋에 맞춰 튜닝한 수치는 그 설정에서만 나온 결과다. 논문이 튜닝 조건을 밝혔는지 본다.

2. **하드웨어**: 실시간으로 도는지는 하드웨어에 크게 달려 있다. 데스크톱 GPU에서 실시간인 것과 임베디드 ARM 보드에서 실시간인 것은 다르다.

3. **Completeness**: 어떤 시스템은 어려운 시퀀스에서 추적을 놓친다. 성공한 구간만으로 ATE를 계산하면 얼마나 자주 실패했는지가 빠지므로, 성공한 시퀀스 비율(completeness)도 함께 보고한다.

4. **초기화**: VIO는 초기화 방법과 걸리는 시간이 시스템마다 달라서, 같은 시퀀스에서도 결과가 달라진다.

5. **Loop closure를 켰는가**: Loop closure를 켜면 ATE가 크게 떨어지므로 비교하는 시스템들의 설정을 맞춘다. DSO는 loop closure가 없는 odometry이고, LDSO는 여기에 loop closure를 더한 것이다. ORB-SLAM3는 loop closure를 끄고도 돌릴 수 있다. 시스템 이름만 보지 말고 실제 설정을 확인한다.

---

## 12.5 오픈소스 도구 가이드

### 12.5.1 최적화 라이브러리

**GTSAM** (Georgia Tech Smoothing and Mapping):
- Factor graph 기반 최적화 라이브러리
- iSAM2 구현이 들어 있다
- C++ 라이브러리이고 Python 바인딩(gtsam)도 있다
- LIO-SAM이 백엔드로 쓴다
- Factor graph를 문제 구조 그대로 짜기 쉽고, 자주 쓰는 factor가 미리 들어 있다

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
        gtsam.Point3(0.0, 1.0, 0.0)   # 병진은 pose1 좌표계에서 해석되므로, 회전 후 전진은 +y
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
        gtsam.Rot3.Rz(-np.pi / 2),
        gtsam.Point3(-1.0, 1.0, 0.0)   # pose2.between(pose0): pose2 = (Rz(90°), (1,1,0))에서 원점으로
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
- Google이 만든 비선형 최소자승 최적화 라이브러리
- C++ 라이브러리이고 Python에서 쓰기는 불편하다
- 자동 미분(automatic differentiation)을 지원한다
- VINS-Mono 등이 쓴다
- Factor graph 개념 없이 cost function을 직접 더해 문제를 만든다

**g2o** (General Graph Optimization):
- Kümmerle et al.(ICRA 2011)이 발표했다
- 그래프 최적화용 C++ 라이브러리
- SE2, SE3, Sim3 같은 vertex/edge 타입이 미리 들어 있다
- 문제를 vertex와 (hyper-)edge로 이루어진 그래프로 정의한다. GTSAM과 어느 쪽이 빠른지는 문제 크기, 변수 순서, 선형 solver, 빌드 설정에 따라 뒤바뀌므로 자기 문제로 직접 재 본다

세 라이브러리 비교:

| 특성 | GTSAM | Ceres | g2o |
|------|-------|-------|-----|
| 추상화 수준 | Factor graph | Cost function | Graph vertex/edge |
| 자동 미분 | 부분 지원 (Expression 기반) | 완전 지원 | 기본은 수치 미분, 버전에 따라 자동 미분 제공 |
| Incremental | iSAM2 | 미지원 | 예제(g2o_incremental)로 제공 |
| Python 지원 | 양호 | 제한적 | 제한적 |
| 대표 사용처 | LIO-SAM | VINS-Mono | ORB-SLAM |

### 12.5.2 캘리브레이션 도구

**Kalibr** (ethz-asl):
- 카메라-IMU, 카메라-카메라, 여러 IMU 사이의 캘리브레이션
- 궤적을 연속 시간 B-spline으로 표현해 추정한다
- AprilGrid 타겟을 쓴다
- ROS에 의존해서 설치가 까다롭다

**OpenCalib** (2022):
- 자율주행차에 달린 센서 전체를 캘리브레이션하는 도구 모음(Shanghai AI Lab, arXiv 2205.14087)
- 카메라 내부 파라미터, LiDAR-카메라, LiDAR-IMU, radar-카메라, LiDAR-LiDAR처럼 센서 쌍마다 도구가 따로 있다. 지원하는 조합은 저장소에서 확인한다
- 타겟을 쓰는 방법과 쓰지 않는 방법이 모두 들어 있다

**[direct_visual_lidar_calibration](https://arxiv.org/abs/2302.05094)** (Koide et al. 2023):
- 타겟 없이 NID(normalized information distance)로 LiDAR와 카메라를 맞추는 캘리브레이션
- SuperGlue 대응점으로 초기값을 잡고, NID를 최소화해 정밀하게 맞춘다
- 이미지 한 장과 LiDAR 포인트 클라우드 한 묶음만으로 돈다(single-shot). 회전형 LiDAR는 점이 성기므로 여러 scan을 쌓은 포인트 클라우드를 쓰기도 한다

### 12.5.3 평가 도구

**evo** (MH Grupp):
- Python으로 만든 궤적 평가 도구
- ATE, RPE를 계산하고 그림으로 그린다
- TUM, KITTI, EuRoC 형식을 읽는다
- 명령행 도구와 Python API가 모두 있다

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
import copy

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
        traj_est_aligned = copy.deepcopy(traj_est)
        traj_est_aligned.align(traj_gt, correct_scale=False)   # align은 제자리 변환, 반환값은 (R, t, s)
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
        
        traj_est_aligned = copy.deepcopy(traj_est_sync)
        traj_est_aligned.align(traj_gt_sync, correct_scale=False)
        
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
- 센서 퓨전 시스템을 묶는 미들웨어
- 센서 드라이버, 타임스탬프와 메시지 동기화, 프로세스 사이의 메시지 전달을 맡는다
- 이 가이드에서 다룬 LIO-SAM, FAST-LIO2, VINS-Mono 같은 오픈소스 SLAM 시스템이 ROS 패키지로 나온다

**Open3D**:
- 3D 데이터를 다루는 라이브러리(C++ 코어, Python API)
- point cloud, mesh, TSDF를 다룬다
- ICP, RANSAC, FPFH 같은 기하 알고리즘이 들어 있다
- 시각화 기능이 있다

**CloudCompare**:
- 포인트 클라우드를 비교하고 편집하는 GUI 도구
- 점과 점(C2C), 점과 mesh(C2M) 사이 거리를 잰다
- 정합, 필터링, 다운샘플링도 한다

**COLMAP**:
- Structure from Motion(SfM)과 Multi-View Stereo(MVS) 파이프라인
- 사진 여러 장으로 3D를 재구성한다
- 센서 퓨전에서는 camera intrinsic을 추정하거나 기준 지도를 만들 때 쓴다

