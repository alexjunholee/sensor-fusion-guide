# 12장 실전 시스템과 벤치마크

앞선 장들에서 센서 모델링과 상태 추정, odometry, place recognition, 공간 표현 같은 핵심 구성 요소를 하나씩 떼어 살펴봤다. 하지만 자율주행차나 드론, 핸드헬드 매핑 장비 같은 실제 플랫폼에서는 이 모듈들이 따로 놀지 않는다. 제한된 연산 자원과 전력, 엄격한 지연 한계, 안전 요구 사항 아래 하나의 파이프라인으로 맞물려 돌아간다. 플랫폼마다 물리적 제약과 운용 목표가 다르기에 센서 구성부터 융합 구조, 최적화 전략까지 트레이드오프가 뚜렷하게 갈리며, 이렇게 통합한 시스템이 현장에서 제 몫을 하는지는 표준 벤치마크와 평가 지표로 확인해야 한다.

---

## 12.1 플랫폼별 시스템 설계

### 12.1.1 자율주행 (Autonomous Driving)

자율주행차의 센서 시스템은 단일 센서의 결함이 곧바로 사고로 이어지지 않도록 설계해야 한다. 센서 하나가 고장 나더라도 이상을 즉시 감지해 잘못된 값이 후속 모듈로 전파되지 않게 차단해야 하며, 고장 이후 주행을 이어 갈지 최소 위험 상태(minimal risk condition)로 진입해 정차할지는 운행 설계 영역(ODD, Operational Design Domain)과 안전 논증(safety case)에 따라 결정한다. 서로 다른 물리 원리의 센서를 함께 배치하면 한쪽이 제 기능을 못 할 때 다른 센서가 공백을 메우고, 평소에도 센서 간 출력값을 대조해 상호 검증을 수행할 수 있다.

실제 차량에 어떤 센서를 몇 대나 싣는지는 중복성(redundancy)과 비용, 인식 아키텍처 전략에 따라 크게 달라진다.

#### Sensor Suite 구성 사례

- **Waymo** (5th generation): 360°를 덮고 300 m 넘는 거리를 보는 장거리 LiDAR 1대, 차체 네 모서리의 가까운 사각지대를 메우는 단거리 perimeter LiDAR 4대, 화각이 겹치도록 360°로 배치한 카메라 29대, 속도 측정을 위한 radar(Waymo 발표문은 정확한 개수를 밝히지 않음), 그리고 IMU, GNSS, wheel encoder로 구성된다.
- **[nuScenes](https://arxiv.org/abs/1903.11027) 수집 차량**: 32-beam spinning LiDAR 1대, 360° 커버용 서라운드 카메라 6대, radar 5대, IMU, GNSS를 탑재했다.
- **Camera-centric production example**: 여러 대의 surround camera와 신경망 인식을 파이프라인의 중심에 두는 구성이다. 다만 어떤 센서를 조합하고 radar를 탑재할지는 차량 세대와 판매 시장, 출시 시기에 따라 달라지므로, 특정 제조사의 구성을 인용하거나 비교할 때는 반드시 세대와 시점을 함께 명시해야 한다.

### 12.1.2 Production-Level Fusion Pipeline

양산 자율주행 시스템의 전체 센서 융합 흐름을 개략적으로 그리면 다음과 같다. 다이어그램의 화살표는 주된 데이터 전달 순서를 보여 줄 뿐, 실제 차량 소프트웨어에서는 각 모듈이 이보다 훨씬 촘촘하게 얽혀 돌아간다. 예를 들어 위치 추정은 GNSS·IMU·LiDAR·wheel 관측을 받아 인식 모듈과 병렬로 실행되며, 여기서 나온 자차 위치와 자세는 인식·예측·계획 모듈 모두가 동시에 참조한다.

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

여러 센서의 출력을 합치는 지점은 크게 두 갈래로 나뉜다.

- **Late fusion**은 각 센서가 독립적으로 3D bounding box를 검출한 뒤, NMS(Non-Maximum Suppression, 겹치는 박스 중 신뢰도 점수가 가장 높은 것만 남기는 기법) 등으로 최종 결과를 추린다. 모듈 경계가 뚜렷해 개발과 디버깅이 수월하지만, 이미 각 센서가 독자적 판단을 끝낸 결과물만 합치기 때문에 한 센서가 놓친 신호를 다른 센서의 원시 정보로 보완하기 어렵다.
- **Deep fusion**은 센서의 중간 feature를 신경망 내부에서 결합해, 융합 방식 자체를 데이터로부터 학습한다. feature를 어디서 어떻게 맞물리는지는 모델마다 다른데, MIT의 [BEVFusion](https://arxiv.org/abs/2205.13542)은 카메라와 LiDAR feature를 동일한 BEV(bird's-eye view, 위에서 내려다본 평면) 격자 위로 투영해 결합하고, TransFusion은 LiDAR BEV feature에서 만든 object query가 transformer decoder 안에서 이미지 feature에 cross-attention을 걸도록 설계했다. 덕분에 한 센서의 약점을 다른 센서의 단서로 메우는 전략을 네트워크가 스스로 익히지만, end-to-end 학습을 위해 대규모 레이블 데이터가 필요하다는 부담이 따른다.

아래 pseudo-code는 LSS(Lift-Splat-Shoot) 계열 방식으로 카메라와 LiDAR feature를 공통 BEV 평면에 정렬해 합치는 deep fusion의 전형적인 계산 흐름을 보여 준다.

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

자율주행의 위치 추정은 전역 초기화와 정밀 정합이라는 서로 다른 요구를 맞추기 위해 보통 두 층위로 구성된다.

1. **Global localization**: RTK나 PPP 같은 보정 GNSS로 차량이 지도 위 어디쯤 있는지 대략적인 초기 위치를 잡는다. 다만 도심지에서는 고층 건물에 위성 신호가 반사되는 multipath 현상 탓에 오차가 수 미터까지 벌어질 수 있어, GNSS만으로는 주행에 필요한 정밀도를 보장하기 어렵다.
2. **Map-relative localization**: 현재 들어온 LiDAR scan을 사전에 구축한 HD point-cloud map에 NDT나 ICP로 정합한다. 지도가 정확하고 scan과 지도 사이에 충분한 겹침이 있으며, 주변 기하 형상이 뚜렷하고 초기 위치 오차가 허용 범위 안에 있다면 GNSS가 흔들리는 구간에서도 정밀하게 위치를 붙잡아 준다. 다만 cm 단위의 정확도가 실제로 확보되는지는 해당 지도 품질과 운용 조건 아래에서 별도로 검증해야 한다.

실제 시스템에서는 두 단계를 분리해 두지 않고, IMU의 단기 예측과 LiDAR 맵 정합, GNSS 전역 앵커, 휠 속도 관측을 하나의 factor graph로 묶어 최적의 상태 $\mathbf{x}^*$를 동시에 추정한다.

$$\mathbf{x}^* = \arg\min \underbrace{f_{\text{IMU}}}_{\text{예측}} + \underbrace{f_{\text{LiDAR}}}_{\text{맵 정합}} + \underbrace{f_{\text{GNSS}}}_{\text{전역 앵커}} + \underbrace{f_{\text{wheel}}}_{\text{속도}}$$

---

## 12.2 드론/UAV

### 12.2.1 Visual-Inertial 중심 시스템

자율주행차와 달리 소형 드론은 탑재 중량과 배터리 전력(SWaP)이 극도로 빠듯하다. 무겁고 전력 소모가 큰 LiDAR를 싣기 어려운 대신, 가볍고 저전력으로 동작하는 카메라와 IMU 조합이 표준처럼 자리 잡았다(물론 Livox Mid-360처럼 소형·경량 LiDAR를 탑재하는 드론 플랫폼도 존재한다). 여기에 회전익 기체 특유의 기계적 진동이라는 난제가 겹친다. 프로펠러가 만들어 내는 고주파 진동이 IMU 측정치에 그대로 실려 적분 오차를 키우기 때문에, 하드웨어 방진 마운트와 소프트웨어 필터링을 반드시 함께 적용한다.

드론용 오픈소스 VIO는 연산 예산과 전역 지도 재활용 여부에 따라 최적화 기반, 필터 기반, factor 복원 기반 가운데 골라 쓴다.

- **VINS-Mono/Fusion**: tightly-coupled 비선형 최적화 방식으로, PX4 비행 제어기와 연동해 널리 쓰인다.
- **MSCKF/OpenVINS**: 필터 기반 구조라 연산량이 가벼워 연산 자원이 제한된 임베디드 보드에 유리하다.
- **Basalt**: 비선형 factor 복원(non-linear factor recovery) 기법을 제안한 시스템("Visual-Inertial Mapping with Non-Linear Factor Recovery", Usenko et al., RA-L 2020)으로, VIO 과정에서 추출한 정보를 비선형 factor 형태로 압축·요약해 전역 지도 최적화에 다시 활용한다.

### 12.2.2 GPS-Denied Navigation

실내나 터널, 울창한 숲 아래, 혹은 전파 교란(jamming) 환경에서는 GPS 신호에 기댈 수 없다. 이런 GPS-denied 환경에서 드론을 자율 비행시키는 전략은 비행 공간에 사전 지도나 외부 인프라를 준비할 수 있는지에 따라 달라진다.

1. **VIO 단독**: 외부 도움 없이 자체 센서만으로 항법을 수행한다. 시간이 지날수록 드리프트가 쌓이므로, 얼마나 오래 비행할 수 있는지는 기동 패턴, 주변 텍스처, 캘리브레이션 정밀도, loop closure 유무, 시스템이 허용하는 오차 한계에 좌우된다.
2. **VIO + 지형 매칭**: 카메라가 현재 포착한 장면을 미리 구축한 지형·건물 지도(prior map)와 정합해 누적 드리프트를 억제한다.
3. **VIO + UWB**: 비행 구역에 UWB 앵커를 미리 배치해 두고, 각 앵커까지의 거리 측정값으로 드리프트를 주기적으로 보정한다.
4. **VIO + 기압계**: 기압 변화로 상대 고도를 추정해 수직 드리프트를 보완한다. 다만 기상 변화나 온도, 프로펠러 하강 기류(prop wash)가 바이어스를 유발하므로 고도를 안정적으로 붙잡으려면 다른 고도 기준과 함께 융합해야 한다.

### 12.2.3 실시간 제약

드론 항법에서 센서를 몇 Hz로 샘플링하고 종단 간(end-to-end) 지연을 어디까지 허용할지는 기체의 최대 각가속도, 제어 루프 대역폭, 카메라 노출 시간, 통신 지연을 한꺼번에 저울질해 정한다. IMU는 에일리어싱을 피하고 적분 오차가 급격히 불어나지 않도록 충분히 높은 주기로 읽어야 하며, 카메라 노출 시간은 길어질수록 모션 블러가 심해지고 짧아질수록 저조도 노이즈가 늘어나므로 운용 조도와 기동 속도에 맞춰 타협점을 찾는다. 센서 입력이 들어와 상태 추정값이 나오기까지의 허용 지연 한계 역시 폐루프 제어 안정성 해석과 실제 비행 시험을 거쳐 확정한다.

고속 기동에 대응하기 위해 아예 스캔 개념을 버린 접근도 있다. **Point-LIO**(He et al., Advanced Intelligent Systems 2023)는 LiDAR 스캔 한 프레임이 모두 모이길 기다리지 않고, 개별 포인트가 수신되는 즉시 상태를 갱신하는 LIO다. 덕분에 스캔 도중 기체가 움직여 생기는 모션 왜곡 자체가 발생하지 않는다. 논문 실험에서는 4~8 kHz에 이르는 초고속 odometry 출력을 달성했으며, IMU 측정 한계를 넘어서는 75 rad/s의 급격한 회전 속에서도 추적을 유지했고, 레이싱 드론에 탑재한 비행 시연까지 선보였다.

아래 코드는 드론의 빠듯한 연산 예산과 진동·노출·모션 블러 제약을 실제 VIO 설정 파라미터와 프레임 품질 필터링 로직으로 구체화한 예시다.

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

드론이 비행 중량과 전력 제약을 맞추는 데 집중한다면, 손에 들거나 등에 메는 모바일 매핑 장비는 사람이 직접 걸어 다니며 건축 측량이나 BIM(Building Information Modeling, 건물 정보 모델), 디지털 트윈용 고밀도 3D 지도를 구축하는 데 초점을 둔다.

### 12.3.1 SLAM as a Service

현장에서 널리 쓰이는 상용 핸드헬드·웨어러블 매핑 장비의 특징은 다음과 같다.

- **Leica BLK2GO**: 한 손에 들고 걸으며 스캔하는 LiDAR 스캐너로, LiDAR·IMU·카메라를 결합해 실시간 SLAM을 수행한다. 다만 제조사가 제시하는 정확도 수치는 특정 스캔 길이와 루프 폐쇄 여부, 특정 검증 절차 아래에서 측정된 값이므로, 실제 발주 규격 충족 여부는 §12.3.2의 기준에 따라 별도로 확인해야 한다.
- **NavVis VLX**: 상체 앞쪽과 머리 위에 센서를 배치해 메는 웨어러블 장비다. VLX 3 데이터시트 기준으로 32채널 LiDAR 2대와 카메라 4대를 탑재해 실내·실외 현장을 모두 아우른다.
- **GeoSLAM ZEB**: 대표적인 핸드헬드 모바일 매핑 장비로, 초기 모델(ZEB1)은 2D LiDAR를 스프링에 달아 보행 시의 흔들림으로 3D 공간을 훑었고, ZEB-REVO부터는 모터로 스캐너 헤드를 직접 회전시키는 방식으로 발전했다.

세 제품 모두 내부 SLAM 알고리즘을 비공개로 유지한다. 다만 이런 장비들이 원시 센서 데이터에서 최종 3D 산출물을 뽑아내는 전체 공정은 대체로 아래 5단계 흐름을 따르며, 각 블록 아래에는 그 역할을 수행하는 대표적인 공개 알고리즘을 병기했다.

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

상업 측량이나 시설물 납품용 매핑에서는 스캔에 앞서 발주 규격과 검증 절차를 못 박고 시작한다.

- **절대 정확도**: 수평·수직 오차 한계와 신뢰수준을 미리 정하고, 최적화 계산에 넣지 않은 독립 check point를 현장에 두어 결과를 검증한다. 기준점은 현장 여건에 맞춰 GNSS, total station, 지상 기준점(GCP) 가운데 고른다.
- **상대 정확도**: 지도 내부의 점과 점 사이 거리나 형상이 얼마나 어긋나지 않는지를 평가하며, loop closure와 전역 최적화가 이 정합도를 책임진다.
- **포인트 밀도**: 대상 구조물의 크기와 최종 납품물 목적에 맞춰 점 간격과 누락 허용 범위를 정한다. 1 cm 수준을 요구하는 현장도 흔하지만, 모든 프로젝트에 일률적으로 적용되는 절대 기준은 아니다.
- **색상 품질**: 포인트 클라우드에 HDR 색상을 어긋남 없이 입히려면 카메라와 LiDAR 간 정밀한 시간 동기화와 extrinsic 캘리브레이션이 뒷받침돼야 한다.

현장에서 이런 장비를 들고 걸을 때 센서 융합 파이프라인이 가장 빈번하게 부딪히는 물리적 난관은 세 가지다.

1. **Degenerate environment(퇴화 환경)**: 끝이 보이지 않는 긴 복도나 텅 빈 실내처럼 기하학적 특징이 부족한 공간에서는 LiDAR scan matching만으로 움직임을 구속할 수 없어 드리프트가 급격히 커진다. 이를 피하려고 R3LIVE나 FAST-LIVO2 같은 LiDAR-카메라-IMU 결합 시스템은 시각 텍스처와 관성 정보를 함께 끌어와 빈 축을 메운다.
2. **다층 건물**: 실내에서는 GNSS를 쓸 수 없어 계단을 오르내릴 때 수직(높이) 방향 드리프트가 유독 크게 쌓인다. 같은 층을 다시 지나가는 동선이라면 loop closure로 오차를 상당 부분 되돌릴 수 있지만, 한 번 지나가고 마는 편도 경로에서는 루프를 닫을 수 없으므로 기압계나 층고·도면 정보, 측량 기준점 같은 보조 관측으로 고도를 붙잡아야 한다.
3. **유리와 거울**: LiDAR 레이저 빔이 유리창을 그대로 통과해 버리거나 거울 표면에 반사되어 실제 존재하지 않는 허공의 점(ghost point)을 찍어 낸다. 따라서 카메라 정보로 투명·반사 표면을 보완하거나 반사 아티팩트를 후처리·필터링 단계에서 걸러 내야 한다.

---

## 12.4 벤치마크 & 평가

플랫폼에 맞춰 센서 융합 파이프라인을 구축했다면, 다음 과제는 그 성능을 객관적으로 검증하는 일이다. 궤적이 얼마나 정확하고 장기 주행에서 드리프트가 얼마나 쌓이며 가혹한 환경에서 얼마나 버티는지를 정량화하려면, 공통 데이터셋과 명확하게 정의된 평가 지표가 뒷받침돼야 한다.

### 12.4.1 주요 데이터셋

센서 융합과 SLAM 연구에서 표준 비교 기준 구실을 하는 대표 데이터셋은 환경과 플랫폼, 탑재 센서 조합에 따라 다음과 같이 나뉜다.

| 데이터셋 | 연도 | 환경 | 센서 | 특징 |
|----------|------|------|------|------|
| **[KITTI](https://doi.org/10.1177/0278364913491297)** | 2012 | 실외 (자율주행) | Stereo, LiDAR, GPS/IMU | Odometry 평가용 11개 training + 11개 test 시퀀스 |
| **[EuRoC](https://doi.org/10.1177/0278364915620033)** | 2016 | 실내 (MAV) | Stereo, IMU | Machine Hall + Vicon Room의 VIO 시퀀스 |
| **[TUM-RGBD](https://doi.org/10.1109/IROS.2012.6385773)** | 2012 | 실내 | RGB-D | Kinect v1 기반 Visual SLAM 시퀀스와 평가 도구 |
| **TUM-VI** | 2018 | 실내+실외 | Stereo, IMU | VIO 벤치마크. 다양한 모션 패턴 |
| **[Hilti](https://arxiv.org/abs/2109.11316)** | 2021–2023 challenge editions | 건설 현장 | LiDAR, Camera, IMU | 산업 환경 특화. 어려운 조건 |
| **[HeLiPR](https://arxiv.org/abs/2309.14590)** | 2023 | 실외 (도심) | 이종 LiDAR 4종, IMU, GNSS/INS | 서로 다른 LiDAR 사이의 장소 인식 연구용. Ouster OS2-128, Velodyne VLP-16, Livox Avia, Aeva Aeries II. 논문의 센서 목록에 카메라는 없음 |
| **[nuScenes](https://arxiv.org/abs/1903.11027)** | 2020 | 실외 (자율주행) | Camera, LiDAR, Radar, GPS/IMU | 1000개 씬, 23 클래스 3D 어노테이션, 360° 서라운드 센서 |
| **Newer College** | 2020 | 실외 (캠퍼스) | LiDAR, Camera, IMU | 옥스퍼드 New College 구내. 핸드헬드로 수집, 루프가 많음 |

표에 묶은 데이터셋은 수집 시기와 목표 플랫폼에 따라 고유한 강점과 주의할 점을 지닌다.

- **KITTI**: 2012년 공개된 자율주행 odometry·SLAM의 원조격 벤치마크로, Velodyne 64채널 LiDAR와 스테레오 카메라, GPS/IMU 데이터를 제공한다. 다만 2011년에 녹화한 자료라 센서 사양이 오래됐고, ground truth 역시 GPS/INS(OXTS RT3003)에 의존하므로 위성 수신이 불량한 구간에서는 기준 궤적 자체의 오차가 cm 수준을 넘어설 수 있다.
- **EuRoC**: 소형 무인기(MAV)에 스테레오 카메라와 IMU를 달아 수집한 VIO 벤치마크로, 6장에서 다룬 ORB-SLAM3와 DROID-SLAM도 이 데이터셋으로 성능을 검증했다. Vicon Room에서는 Vicon 모션 캡처를, Machine Hall에서는 Leica 레이저 트래커를 써서 정밀한 ground truth를 확보했으며, 11개 시퀀스를 easy, medium, difficult 난이도로 구분해 놓았다.
- **Hilti**: 먼지와 저조도, 구조적 퇴화가 잦은 건설 현장에서 SLAM이 얼마나 버티는지 겨루는 챌린지 데이터셋이다. 2021년부터 2023년까지 해마다 센서 구성과 평가 규칙을 바꿔 가며 열렸으므로, 수치를 인용할 때는 해당 연도의 논문과 규칙을 함께 확인해야 한다.
- **HeLiPR**: 2023년에 공개된 도심 데이터셋으로, 회전식 LiDAR 2종과 비반복 스캔 LiDAR(Livox Avia), FMCW LiDAR(Aeva Aeries II)를 한 차량에 모두 실었다. 덕분에 서로 다른 스캔 패턴을 가진 이종 LiDAR 사이에서 같은 장소를 찾아내는 inter-LiDAR place recognition 연구에 알맞다. 같은 구간을 약 2주 간격으로 반복 주행했고, 기존 MulRan 데이터셋과 겹치는 경로도 포함한다.
- **Newer College**: 옥스퍼드 New College 구내를 사람이 매핑 장비를 손에 들고 걸으며 모았다. 동일 구역을 여러 차례 순회해 loop closure를 시험하기 좋고, 보행 특유의 큰 흔들림이 그대로 담겨 있다. 계절 변화 같은 장기 주행 데이터가 필요하다면 아래에서 소개할 Boreas 같은 데이터셋이 더 적합하다.

2022년 이후에는 측정 정밀도를 한층 끌어올리거나 악천후·신형 센서를 겨냥한 후속 벤치마크들이 잇따라 등장했다. [Hilti-Oxford](https://arxiv.org/abs/2208.09825)(2022)는 2022년 Hilti 챌린지에 쓰인 건설 현장 데이터셋으로 mm 단위의 정밀한 ground truth를 제공한다. [Boreas](https://arxiv.org/abs/2203.10168)(Burnett et al., IJRR 2023)는 LiDAR·radar·카메라를 실은 차량으로 같은 경로를 1년 내내 반복 주행하며 계절 변화와 비·눈, 야간 주행까지 담아냈다. 한편 [SNAIL Radar](https://arxiv.org/abs/2407.11705)(Huai et al., IJRR 2025)는 4D radar SLAM을 겨냥해 핸드헬드, 전기자전거, SUV 세 플랫폼으로 맑은 날과 밤, 폭우 속에서 우한대학 캠퍼스와 터널 포함 고속도로를 주행하며 44개 시퀀스를 구축했고, 기존 radar odometry 3종과 place recognition 3종의 비교 결과까지 함께 실었다.

### 12.4.2 평가 메트릭

궤적 추정 정확도를 재는 표준 지표는 전역 일관성을 보는 ATE와 국소 드리프트 누적량을 보는 RPE 두 가지로 나뉜다.

**ATE (Absolute Trajectory Error)**는 추정한 전체 궤적이 ground truth와 평균적으로 얼마나 떨어져 있는지를 RMSE로 측정한다.

$$\text{ATE} = \sqrt{\frac{1}{N} \sum_{i=1}^{N} \| \text{trans}(\mathbf{T}_{\text{gt},i}^{-1} \cdot \mathbf{T}_{\text{est},i}) \|^2}$$

추정 궤적과 기준 궤적은 출발 좌표계나 스케일이 서로 다를 수 있으므로, ATE를 계산하기 전 먼저 두 궤적을 정렬(alignment)하는 것이 보통이다. 이때 몇 자유도를 맞춰 줄지는 시스템이 관측 가능한 물리량과 벤치마크 규약에 따른다. 예를 들어 절대 스케일을 알 수 없는 단안 카메라는 $\text{Sim}(3)$로 정렬해 스케일 배율까지 맞추는 반면, 실제 미터 단위를 복원하는 stereo·LiDAR·단안+IMU 시스템은 강체 변환인 $\text{SE}(3)$로만 정렬한다(반면 KITTI odometry는 공식 지표로 ATE 대신 이동 구간별 상대 오차를 쓰며, §12.3.2의 측량처럼 좌표 정렬 자체를 생략하고 절대 오차를 곧바로 재는 사례도 있다). 정렬 변환 $\mathbf{S}^*$를 찾는 아래 최적화 문제는 Umeyama 알고리즘(1991)을 이용해 반복 계산 없이 닫힌 해(closed-form solution)로 구할 수 있다.

$$\mathbf{S}^* = \arg\min_{\mathbf{S} \in \text{Sim}(3)} \sum_i \| \mathbf{p}_{\text{gt},i} - \mathbf{S} \cdot \mathbf{p}_{\text{est},i} \|^2$$

**RPE (Relative Pose Error)**는 전체 지도의 전역 정합도 대신 일정 간격 $\Delta$만큼 떨어진 두 시점 사이의 상대 움직임이 얼마나 정확한지 잰다. 루프 폐쇄 없이 궤적을 이어 갈 때 오차가 어느 속도로 쌓이는지 가늠하는 지표다.

$$\text{RPE}(\Delta) = \sqrt{\frac{1}{M} \sum_{i=1}^{M} \| \text{trans}((\mathbf{T}_{\text{gt},i}^{-1} \mathbf{T}_{\text{gt},i+\Delta})^{-1} (\mathbf{T}_{\text{est},i}^{-1} \mathbf{T}_{\text{est},i+\Delta})) \|^2}$$

위 식에서 $\Delta$는 프레임 번호 간격으로, $\Delta$가 짧으면 프레임 간 odometry의 국소 정밀도를, 길면 그 구간 동안 누적된 드리프트를 드러낸다. KITTI 벤치마크처럼 프레임 수가 아니라 이동 거리 구간(100~800 m)으로 평가하려면, 각 시작 프레임에서 실제 누적 주행 거리가 목표 길이에 도달하는 종료 프레임을 먼저 찾아 짝지어 줘야 한다.

한편 궤적 추정과 별개로 **place recognition** 성능은 검색 품질 지표로 평가한다. Recall@N은 데이터베이스에서 뽑은 상위 N개 후보 안에 실제 정답 장소가 들어 있는 질의 비율을 뜻하며, N=1일 때가 첫 번째 후보만으로 정답을 맞혀야 해 가장 엄격하다. 판정 threshold를 연속적으로 바꾸며 precision과 recall의 트레이드오프를 그린 곡선이 precision-recall curve이고, 이 곡선 아래 면적인 AUC는 전체 검색 성능을 하나의 스칼라 숫자로 요약해 준다.

아래 Python 코드는 앞서 수식으로 정리한 Umeyama 정렬 기반 ATE 계산과 프레임 간격 $\Delta$ 기준 RPE 계산을 그대로 구현한 예시다.

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

논문이나 리더보드에 적힌 ATE·RPE 수치만 나란히 놓고 두 알고리즘의 우열을 바로 가리면 오판하기 쉽다. 벤치마크 숫자를 제대로 읽으려면 아래 다섯 가지 실험 조건이 맞물려 있는지 반드시 따져 봐야 한다.

1. **파라미터 튜닝**: 같은 알고리즘이라도 특징점 수, 키프레임 간격, 노이즈 공분산 설정에 따라 수치가 크게 출렁인다. 특정 데이터셋에 과적합시켜 얻은 점수인지, 모든 시퀀스에 고정 파라미터를 썼는지 논문에서 튜닝 조건을 확인해야 한다.
2. **하드웨어**: "실시간 동작"이라는 표현은 연산 플랫폼을 떼어 놓고는 성립하지 않는다. 데스크톱 GPU에서 30 FPS로 도는 알고리즘과 저전력 임베디드 ARM 보드에서 실시간을 맞추는 알고리즘은 전혀 다른 문제다.
3. **Completeness**: 어려운 시퀀스 중간에 추적을 놓쳐 버린 뒤, 성공한 구간만 잘라 ATE를 계산하면 오차가 인위적으로 작아 보인다. 따라서 오차 평균과 함께 전체 시퀀스 중 추적을 끝까지 유지한 비율(completeness)을 반드시 나란히 보고해야 한다.
4. **초기화**: VIO 시스템마다 초기화 방식과 수렴에 걸리는 시간이 달라, 초반 움직임이 적은 시퀀스에서는 초기화 지연이나 실패 자체가 전체 결과를 좌우한다.
5. **Loop closure 활성화 여부**: Loop closure를 켜고 전역 최적화를 돌리면 ATE가 극적으로 줄어들기 때문에, 순수 odometry와 SLAM 시스템을 동일 선상에 두고 비교해서는 안 된다. 예컨대 DSO는 loop closure가 없는 순수 odometry인 반면 LDSO는 여기에 loop closure를 붙인 버전이고, ORB-SLAM3는 설정에 따라 loop closure를 끄고 돌릴 수도 있다. 시스템 이름만 보지 말고 실제 구동 설정을 맞춰 비교해야 한다.

---

## 12.5 오픈소스 도구 가이드

실전 센서 융합 파이프라인을 밑바닥부터 혼자 짜는 경우는 드물다. 백엔드 최적화부터 다중 센서 캘리브레이션, 궤적 평가, 미들웨어 연동까지 검증된 오픈소스 도구를 적재적소에 조합해 쓰는 편이 안정성과 개발 속도 면에서 훨씬 유리하다.

### 12.5.1 최적화 라이브러리

- **GTSAM** (Georgia Tech Smoothing and Mapping): factor graph 기반 비선형 최적화 라이브러리로, 증분 최적화 알고리즘인 iSAM2 구현을 기본 탑재하고 있다. C++ 코어와 함께 공식 Python 바인딩(`gtsam`)도 제공하며, LIO-SAM을 비롯한 많은 SLAM 시스템이 백엔드로 채택했다. 문제의 확률적 구조를 factor graph 그대로 코드에 옮기기 직관적이고, IMU preintegration이나 `BetweenFactorPose3` 같은 핵심 factor가 미리 구현되어 있어 개발 생산성이 높다.

아래 Python 예시는 GTSAM으로 prior factor와 odometry factor, 그리고 루프 폐쇄 factor를 하나의 그래프에 추가해 iSAM2로 증분 최적화하는 전형적인 흐름을 보여 준다.

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

- **Ceres Solver**: Google이 개발한 범용 비선형 최소자승 최적화 라이브러리로, VINS-Mono 같은 시스템의 백엔드를 맡는다. C++ 기반으로 설계돼 Python에서 직접 쓰기는 번거롭지만, 자동 미분(automatic differentiation)을 강력하게 지원하며 factor graph 추상화 없이 residual(cost function) 블록을 직접 더해 문제를 조립할 수 있다.
- **g2o** (General Graph Optimization): Kümmerle et al.(ICRA 2011)이 발표한 그래프 최적화용 C++ 라이브러리다. `SE2`, `SE3`, `Sim3` 같은 vertex와 edge 타입이 내장돼 있어 문제를 vertex와 (hyper-)edge 그래프 형태로 곧바로 정의할 수 있다. GTSAM과 어느 쪽이 더 빠른지는 문제 규모, 변수 정렬 순서, 선형 solver, 빌드 옵션에 따라 엇갈리므로 실제 대상 문제로 직접 벤치마킹해 고르는 것이 정확하다.

세 라이브러리의 구조와 특징을 나란히 비교한 표는 §4.5.7에 정리돼 있다.

### 12.5.2 캘리브레이션 도구

- **Kalibr** (ethz-asl): 카메라-IMU, 다중 카메라, 다중 IMU 간 시공간 캘리브레이션의 사실상 표준 도구다. AprilGrid 타겟을 보며 센서 궤적을 연속 시간 B-spline으로 모델링해 extrinsic 파라미터와 시간 오프셋을 함께 추정한다. 다만 ROS 의존성이 있어 최신 환경에서 설치·빌드가 까다롭다는 점이 단점이다.
- **OpenCalib** (2022): Shanghai AI Lab(arXiv 2205.14087)이 공개한 자율주행차용 종합 캘리브레이션 도구 모음이다. 카메라 내부 파라미터부터 LiDAR-카메라, LiDAR-IMU, radar-카메라, LiDAR-LiDAR까지 센서 쌍별로 전용 도구를 제공하며(지원 조합은 저장소 최신 목록을 확인), 캘리브레이션 타겟을 쓰는 방식과 타겟 없이 자연 환경에서 맞추는 방식을 모두 포함한다.
- **[direct_visual_lidar_calibration](https://arxiv.org/abs/2302.05094)** (Koide et al. 2023): 별도 타겟 없이 정보 이론 지표인 NID(normalized information distance)를 최소화해 LiDAR와 카메라의 외부 파라미터를 정밀하게 맞추는 도구다. SuperGlue 대응점으로 초기 정렬을 잡은 뒤 NID 정합으로 다듬으며, 단 한 장의 이미지와 LiDAR 포인트 클라우드 한 묶음만으로(single-shot) 동작한다(점이 성긴 회전형 LiDAR에서는 여러 스캔을 누적한 포인트 클라우드를 입력으로 쓰기도 한다).

### 12.5.3 평가 도구

- **evo** (MH Grupp): Python 기반의 궤적 평가 도구로, ATE와 RPE를 손쉽게 계산하고 궤적 비교 플롯까지 그려 준다. TUM, KITTI, EuRoC 형식을 기본으로 읽으며, 터미널에서 바로 쓰는 CLI와 스크립트 연동용 Python API를 모두 제공한다.

터미널에서 `evo_ape`, `evo_rpe`, `evo_traj` 명령을 실행해 단일 시스템의 오차를 확인하거나 두 시스템을 나란히 비교하는 기본 사용법은 다음과 같다.

```bash
# evo 사용 예시
# ATE 계산
evo_ape tum groundtruth.txt estimated.txt -va --plot --plot_mode xz

# RPE 계산
evo_rpe tum groundtruth.txt estimated.txt -va --delta 100 --delta_unit f

# 두 시스템 비교
evo_traj tum system_a.txt system_b.txt --ref groundtruth.txt -p --plot_mode xz
```

대량 시퀀스를 자동 평가 파이프라인에 물려 돌릴 때는 아래처럼 `evo`의 Python API를 호출해 타임스탬프 동기화부터 정렬, 통계 요약까지 코드 안에서 한 번에 처리할 수 있다.

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

최적화·캘리브레이션·평가 외에 센서 데이터를 실제로 수집하고 3D 결과를 가공·검증할 때 빠지지 않는 기반 도구들은 다음과 같다.

- **ROS 2** (Robot Operating System): 센서 융합 시스템을 하나의 실행 환경으로 묶는 사실상의 표준 미들웨어다. 센서 드라이버 연동과 타임스탬프·메시지 동기화, 노드 간 통신을 맡으며, LIO-SAM·FAST-LIO2·VINS-Mono 등 주요 오픈소스 SLAM 시스템 대부분이 ROS 패키지로 배포된다.
- **Open3D**: C++ 코어와 Python API를 갖춘 3D 데이터 처리 라이브러리다. point cloud, mesh, TSDF 자료구조와 ICP, RANSAC, FPFH 같은 핵심 기하 알고리즘, 그리고 경량 시각화 기능을 함께 제공한다.
- **CloudCompare**: 대용량 포인트 클라우드를 눈으로 대조하고 편집하는 GUI 도구다. 점과 점(C2C), 점과 mesh(C2M) 사이의 오차 거리를 계산할 수 있어 정밀 매핑 결과 검사에 유용하며, 정합·필터링·다운샘플링 작업도 지원한다.
- **COLMAP**: Structure from Motion(SfM)과 Multi-View Stereo(MVS) 파이프라인을 구현한 오픈소스 재구성 도구다. 여러 시점의 사진만으로 정밀한 3D 구조를 복원하므로, 센서 융합 분야에서는 카메라 내부 파라미터를 정밀 추정하거나 비교용 기준 지도를 만들 때 널리 활용된다.

---

지금까지 살펴본 실전 아키텍처와 벤치마크, 오픈소스 생태계는 모두 정교한 기하 모델과 확률적 추정에 뿌리를 두고 있다. 그러나 변화가 극심한 야외 환경에서 장기간 임무를 수행하거나, 사람의 자연어 명령을 이해해 행동하고, 프레임 카메라나 LiDAR가 무력해지는 고속 기동·악천후 구간까지 넘어서려면 기존 파이프라인만으로는 풀기 어려운 과제가 남는다. 다음 13장에서는 foundation model과 end-to-end 학습, 시계열 공간 기억, 그리고 event camera·4D radar 같은 신형 센서가 이 한계를 어디까지 넓히고 있는지 살펴본다.

