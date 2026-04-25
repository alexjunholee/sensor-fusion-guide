# Ch.8 — Multi-Sensor Fusion 아키텍처

> 개별 odometry를 넘어 **여러 센서를 어떻게 통합하는가**의 설계론.
> 앞 챕터에서 Visual Odometry, LiDAR Odometry를 각각 다뤘다면, 이 챕터에서는 이들을 하나의 시스템으로 엮는 아키텍처를 심층적으로 분석한다.

---

## 8.1 Fusion 아키텍처 분류

멀티센서 퓨전 시스템을 설계할 때 가장 먼저 결정해야 하는 것은 **센서 데이터를 어느 수준에서 결합할 것인가**이다. 이 결합의 깊이에 따라 시스템의 복잡도, 성능, 그리고 실패 모드가 근본적으로 달라진다.

### 8.1.1 Loosely Coupled (느슨한 결합)

각 센서를 독립적인 "전문가"로 본다. 각 전문가가 자기 데이터로 독립적으로 결론을 내린 뒤, 상위 레벨에서 이 결론들을 종합한다.

구체적으로, LiDAR odometry 모듈이 LiDAR 스캔으로부터 $\mathbf{T}_{L}$을, Visual odometry 모듈이 이미지로부터 $\mathbf{T}_{V}$를 각각 독립적으로 추정하고, 상위의 fusion 모듈이 이 두 추정치를 결합한다.

$$
\hat{\mathbf{x}} = \arg\min_{\mathbf{x}} \left\| \mathbf{x} - \mathbf{x}_{LiDAR} \right\|^2_{\mathbf{P}_{L}^{-1}} + \left\| \mathbf{x} - \mathbf{x}_{Visual} \right\|^2_{\mathbf{P}_{V}^{-1}}
$$

여기서 $\mathbf{P}_{L}$, $\mathbf{P}_{V}$는 각 서브시스템이 보고하는 공분산이다.

장점은 모듈성이다. 각 센서 모듈을 독립적으로 교체하거나 업그레이드할 수 있고, 문제가 어느 모듈에서 왔는지 추적하기 용이하다. 한 센서가 실패해도 나머지가 계속 동작한다.

단점은 정보 손실이다. 각 서브시스템이 이미 정보를 압축한 상태에서 결합하므로, 센서 간 상호 보완의 이점을 최대한 활용하지 못한다. 예를 들어 LiDAR의 정밀한 기하 정보가 카메라의 스케일 모호성을 해결할 수 있지만, loosely coupled에서는 이 상호작용이 제한적이다. 각 서브시스템이 보고하는 공분산의 일관성(consistency)도 보장되지 않아, 낙관적 공분산을 보고하면 fusion 결과가 왜곡된다.

### 8.1.2 Tightly Coupled (긴밀한 결합)

모든 센서의 **원시 측정(raw measurement)**을 하나의 추정기(estimator)에 직접 넣는다. "전문가"를 두지 않고, 하나의 추정기가 모든 원본 데이터를 직접 본다.

Factor graph 관점에서, 각 센서의 raw measurement가 독립적인 factor로 삽입된다:

$$
\hat{\mathbf{x}} = \arg\min_{\mathbf{x}} \sum_{i} \left\| \mathbf{r}^{\text{IMU}}_i(\mathbf{x}) \right\|^2_{\boldsymbol{\Sigma}^{-1}_{\text{IMU}}} + \sum_{j} \left\| \mathbf{r}^{\text{LiDAR}}_j(\mathbf{x}) \right\|^2_{\boldsymbol{\Sigma}^{-1}_{\text{LiDAR}}} + \sum_{k} \left\| \mathbf{r}^{\text{cam}}_k(\mathbf{x}) \right\|^2_{\boldsymbol{\Sigma}^{-1}_{\text{cam}}}
$$

여기서 $\mathbf{r}^{\text{IMU}}_i$는 IMU preintegration 잔차, $\mathbf{r}^{\text{LiDAR}}_j$는 point-to-plane 잔차, $\mathbf{r}^{\text{cam}}_k$는 reprojection error이다.

센서 간 상호작용을 최대한 활용한다는 것이 핵심 장점이다. IMU가 LiDAR의 motion distortion을 보정하고, LiDAR가 VIO의 스케일을 잡아준다. 정보 이론적으로 최적에 가까운 융합이 가능하다.

대신 시스템 복잡도가 높다. 모든 센서의 관측 모델, 노이즈 모델, 시간 동기화를 하나의 프레임워크에서 관리해야 하고, 한 센서의 이상 데이터가 전체 추정을 오염시킬 수 있어 outlier 처리가 필수다. 실시간성 확보도 어렵다.

대표 시스템으로는 [LIO-SAM (Shan et al. 2020)](https://arxiv.org/abs/2007.00258) (LiDAR+IMU+GPS), VINS-Mono (Camera+IMU), [R3LIVE (Lin et al. 2022)](https://arxiv.org/abs/2109.07982) (Camera+LiDAR+IMU)이 있다.

### 8.1.3 Ultra-Tightly Coupled (신호 수준 결합)

센서의 측정값이 아니라 **신호 자체**를 결합한다. 가장 극단적인 통합이다.

대표적으로 GNSS-INS ultra-tight coupling이 있다. 일반적인 GNSS 수신기는 위성 신호에서 의사거리(pseudorange)를 추출한 뒤 이를 INS와 결합하지만, ultra-tight에서는 INS가 추정한 위치/속도로 GNSS 수신기의 코드/반송파 추적 루프(tracking loop)를 직접 보조한다. 이렇게 하면 약한 신호 환경(도심 캐니언, 실내 진입 직후)에서도 GNSS 신호를 더 오래 추적할 수 있다.

$$
\text{NCO frequency} = f_{\text{nominal}} + \Delta f_{\text{INS-aided}}
$$

여기서 NCO (Numerically Controlled Oscillator)의 주파수를 INS가 예측한 도플러 이동으로 보정하여, 수신기의 추적 범위를 넓힌다.

**현실**: ultra-tight coupling은 GNSS 수신기의 하드웨어/펌웨어 수준 접근이 필요하므로, 군사/항공 분야 외에는 보기 어렵다. 대부분의 로보틱스 시스템은 tightly coupled까지가 실용적 한계이다.

### 8.1.4 세 가지 수준의 비교

```
측정 흐름:

[Loosely]    센서A → 서브시스템A → 포즈A ─┐
                                          ├→ Fusion → 최종 포즈
             센서B → 서브시스템B → 포즈B ─┘

[Tightly]    센서A → raw 측정A ──┐
                                 ├→ 단일 Optimizer → 최종 포즈
             센서B → raw 측정B ──┘

[Ultra-Tight] 센서A 신호 ←→ 센서B 추정 (양방향 신호 수준 결합)
```

```python
import numpy as np
from scipy.linalg import inv

def loosely_coupled_fusion(x_lidar, P_lidar, x_visual, P_visual):
    """
    Loosely coupled fusion: 두 서브시스템의 독립적 추정치를 공분산 가중 평균으로 결합.
    
    Parameters:
        x_lidar: LiDAR 서브시스템의 상태 추정 (n,)
        P_lidar: LiDAR 추정의 공분산 (n, n)
        x_visual: Visual 서브시스템의 상태 추정 (n,)
        P_visual: Visual 추정의 공분산 (n, n)
    
    Returns:
        x_fused: 융합된 상태 추정 (n,)
        P_fused: 융합된 공분산 (n, n)
    """
    # 정보 형식(information form)으로 변환
    I_lidar = inv(P_lidar)
    I_visual = inv(P_visual)
    
    # 정보 행렬의 합산
    I_fused = I_lidar + I_visual
    P_fused = inv(I_fused)
    
    # 정보 가중 평균
    x_fused = P_fused @ (I_lidar @ x_lidar + I_visual @ x_visual)
    
    return x_fused, P_fused


# 예시: 2D 위치 추정
x_lidar = np.array([10.1, 5.2])      # LiDAR가 추정한 위치
P_lidar = np.diag([0.01, 0.01])       # LiDAR는 정밀하지만 균일
x_visual = np.array([10.0, 5.0])      # Visual이 추정한 위치
P_visual = np.diag([0.1, 0.05])       # Visual은 수직 방향이 덜 정밀

x_fused, P_fused = loosely_coupled_fusion(x_lidar, P_lidar, x_visual, P_visual)
print(f"LiDAR:  {x_lidar}, P_diag: {np.diag(P_lidar)}")
print(f"Visual: {x_visual}, P_diag: {np.diag(P_visual)}")
print(f"Fused:  {x_fused}, P_diag: {np.diag(P_fused)}")
# Fused 결과는 LiDAR 쪽에 더 가까움 (공분산이 더 작으므로)
```

<!-- DEMO: fusion_architecture_comparison.html -->

---

## 8.2 Camera + LiDAR + IMU 융합

카메라, LiDAR, IMU 세 센서의 조합은 현재 자율주행과 로보틱스에서 가장 풍부한 정보를 제공하는 센서 스위트다. 카메라는 텍스처와 색상 정보를, LiDAR는 정밀한 3D 기하 정보를, IMU는 고속 관성 측정을 제공하며, 이 세 센서는 서로의 약점을 보완한다:

| 상황 | 카메라 | LiDAR | IMU |
|------|--------|-------|-----|
| 어두운 환경 | ✗ | ✓ | ✓ |
| 텍스처 없는 벽 | ✗ | ✓ | ✓ |
| 기하적 퇴화 (긴 복도) | ✓ | ✗ | ✓ |
| 고속 회전 | ✗ | ✗ | ✓ |
| 스케일 관측 | ✗ (단안) | ✓ | ✗ |
| 색상/시맨틱 | ✓ | ✗ | ✗ |

이 세 센서를 통합하는 최신 시스템들을 분석한다.

### 8.2.1 R3LIVE / R3LIVE++

R3LIVE (Lin et al., 2022)는 LiDAR-Inertial Odometry(LIO)와 Visual-Inertial Odometry(VIO) 두 서브시스템을 tightly coupled로 결합한 시스템이다.

R3LIVE는 **이중 서브시스템(dual-subsystem)** 아키텍처를 채택한다. LIO 서브시스템이 기하학적 구조(geometry)를, VIO 서브시스템이 텍스처(photometric) 정보를 담당하되, 두 시스템이 **상태를 공유**하여 tightly coupled된다.

```
LiDAR scan ──→ [LIO 서브시스템] ──→ 상태 업데이트 (기하)
                     ↓                       ↓
                 IMU data ───────────→ 공유 상태 벡터
                     ↑                       ↑
Camera image ──→ [VIO 서브시스템] ──→ 상태 업데이트 (광도)
```

LIO 서브시스템은 FAST-LIO2와 동일한 방식으로, raw LiDAR point를 ikd-Tree 기반 맵에 직접 point-to-plane 정합한다. Iterated EKF로 상태를 업데이트한다.

VIO 서브시스템이 R3LIVE의 차별점이다. 일반적인 VIO는 특징점(feature point)의 reprojection error를 최소화하지만, R3LIVE는 **photometric error**(광도 오차)를 사용한다. LIO가 구축한 3D 맵의 각 포인트에 RGB 색상을 부여하고, 새 카메라 이미지가 들어올 때 이 맵 포인트들을 이미지에 투영하여 관측된 색상과 맵에 저장된 색상의 차이를 최소화한다:

$$
\mathbf{r}^{\text{photo}}_i = \mathbf{I}(\pi(\mathbf{T}_{CW} \mathbf{p}^W_i)) - \mathbf{c}_i^{\text{map}}
$$

여기서 $\mathbf{I}(\cdot)$는 이미지의 픽셀 강도, $\pi(\cdot)$는 3D→2D 투영 함수, $\mathbf{T}_{CW}$는 월드에서 카메라로의 변환, $\mathbf{p}^W_i$는 맵 포인트의 3D 좌표, $\mathbf{c}_i^{\text{map}}$는 맵에 저장된 해당 포인트의 색상이다.

LiDAR 또는 카메라 중 하나가 일시적으로 실패하더라도 나머지 센서로 계속 동작한다. LiDAR가 가려지면 VIO+IMU로, 카메라가 어두우면 LIO+IMU로 동작한다. 그 결과 SLAM과 동시에 컬러 3D 맵을 실시간으로 생성한다.

### 8.2.2 LVI-SAM

[LVI-SAM](https://arxiv.org/abs/2104.10831) (Shan et al., 2021)은 LIO-SAM의 확장으로, Visual-Inertial 서브시스템과 LiDAR-Inertial 서브시스템을 **양방향(bidirectional)**으로 결합한다.

**양방향 결합의 핵심**:

- **VIS → LIS 방향**: Visual-Inertial 서브시스템이 추정한 포즈를 LiDAR 스캔 매칭의 초기값으로 사용. 특히 LiDAR만으로는 초기값이 부정확한 경우(고속 회전, featureless 환경)에 VIS가 초기값을 제공하여 LiDAR 정합의 수렴을 돕는다.

- **LIS → VIS 방향**: LiDAR가 추정한 깊이 정보를 Visual 특징점에 부여하여, Visual 서브시스템의 깊이 초기화를 가속한다. 단안 VIO에서는 특징점의 깊이를 삼각측량으로 추정하는데, 충분한 시차(parallax)가 쌓이기 전에는 깊이가 부정확하다. LiDAR가 이 깊이를 직접 제공함으로써 즉시 초기화가 가능해진다.

```
         ┌─── VIS 초기 포즈 ───→ LIS 초기값
         │                            │
  [Visual-Inertial]            [LiDAR-Inertial]
         │                            │
         └←── LiDAR 깊이 ────────────┘
         
              ↓ 양쪽 factor 모두 ↓
           [Factor Graph (GTSAM/iSAM2)]
                     ↓
              최종 최적화된 포즈
```

**Factor Graph 설계**: LVI-SAM의 factor graph에는 다음 factor들이 삽입된다:
- IMU preintegration factor (연속 키프레임 사이)
- LiDAR odometry factor (scan matching 결과)
- Visual odometry factor (feature tracking 결과)
- GPS factor (가용 시)
- Loop closure factor (재방문 탐지 시)

### 8.2.3 FAST-LIVO / FAST-LIVO2

[FAST-LIVO2](https://arxiv.org/abs/2408.14035) (Zheng et al., 2024)는 FAST-LIO2 팀(HKU MARS Lab)이 개발한 Camera+LiDAR+IMU 직접(direct) 융합 시스템이다. "직접"이란 특징점 추출 없이 raw 데이터를 직접 사용한다는 뜻이다.

**핵심 혁신 1 — 순차적 업데이트(Sequential Update)**:

이종 센서의 측정은 차원이 다르다. LiDAR는 3D point-to-plane 잔차를, 카메라는 2D photometric 잔차를 제공한다. 이들을 하나의 큰 잔차 벡터로 쌓아서 동시에 최적화하면 Jacobian 행렬의 구조가 복잡해지고 수치적으로 불안정해질 수 있다.

FAST-LIVO2는 이 문제를 **순차적 베이지안 업데이트**로 해결한다:

1. IMU로 상태를 예측 (prediction)
2. LiDAR 측정으로 상태를 업데이트 (1차 업데이트)
3. 카메라 측정으로 상태를 다시 업데이트 (2차 업데이트)

이론적으로, 측정이 독립이라면 순차적 업데이트는 동시 업데이트와 동일한 결과를 준다:

$$
p(\mathbf{x} | \mathbf{z}_L, \mathbf{z}_C) = p(\mathbf{x} | \mathbf{z}_C, \mathbf{z}_L) \propto p(\mathbf{z}_C | \mathbf{x}) \cdot p(\mathbf{z}_L | \mathbf{x}) \cdot p(\mathbf{x})
$$

순차적으로 하면:
$$
\underbrace{p(\mathbf{x} | \mathbf{z}_L)}_{\text{LiDAR 업데이트 후}} \propto p(\mathbf{z}_L | \mathbf{x}) \cdot p(\mathbf{x})
$$
$$
\underbrace{p(\mathbf{x} | \mathbf{z}_L, \mathbf{z}_C)}_{\text{카메라 업데이트 후}} \propto p(\mathbf{z}_C | \mathbf{x}) \cdot p(\mathbf{x} | \mathbf{z}_L)
$$

두 번째 식에서 $p(\mathbf{x} | \mathbf{z}_L)$이 prior 역할을 하며, 최종 결과는 동시 업데이트와 수학적으로 동등하다.

**핵심 혁신 2 — 통합 적응형 복셀 맵**:

FAST-LIVO2는 해시 테이블 + 옥트리 기반의 단일 복셀 맵을 사용한다. LiDAR 모듈이 기하학적 구조(3D 좌표, 법선 벡터)를 구축하면, Visual 모듈이 같은 맵 포인트에 이미지 패치를 부착한다. 이렇게 하면 기하와 텍스처가 하나의 맵에서 일관되게 관리된다.

**핵심 혁신 3 — LiDAR 법선 활용 어파인 워핑**:

카메라의 direct method에서 이미지 패치를 비교할 때, 표면의 기울기를 고려한 어파인 워핑이 정확도를 높인다. FAST-LIVO2는 LiDAR에서 추출한 평면 법선 벡터를 활용하여, 별도의 법선 추정 없이 정확한 어파인 워핑을 수행한다. 이것이 LiDAR-카메라 상호 보완의 구체적 예이다.

**핵심 혁신 4 — 실시간 노출 보정**:

조명이 급격히 변하는 환경(터널 진입/탈출)에서, FAST-LIVO2는 노출 시간(exposure time)을 온라인으로 추정하여 photometric error를 보정한다.

```python
import numpy as np

def sequential_ekf_update(x_pred, P_pred, z_lidar, H_lidar, R_lidar, z_cam, H_cam, R_cam):
    """
    순차적 EKF 업데이트: LiDAR → Camera 순서.
    동시 업데이트와 수학적으로 동등하지만, 차원 불일치 문제를 회피.
    
    Parameters:
        x_pred: 예측 상태 (n,)
        P_pred: 예측 공분산 (n, n)
        z_lidar: LiDAR 측정 (m_L,)
        H_lidar: LiDAR 관측 자코비안 (m_L, n)
        R_lidar: LiDAR 측정 노이즈 (m_L, m_L)
        z_cam: 카메라 측정 (m_C,)
        H_cam: 카메라 관측 자코비안 (m_C, n)
        R_cam: 카메라 측정 노이즈 (m_C, m_C)
    
    Returns:
        x_updated: 최종 업데이트된 상태
        P_updated: 최종 업데이트된 공분산
    """
    # Step 1: LiDAR 업데이트
    S_L = H_lidar @ P_pred @ H_lidar.T + R_lidar
    K_L = P_pred @ H_lidar.T @ np.linalg.inv(S_L)
    y_L = z_lidar - H_lidar @ x_pred  # innovation
    x_after_lidar = x_pred + K_L @ y_L
    P_after_lidar = (np.eye(len(x_pred)) - K_L @ H_lidar) @ P_pred
    
    # Step 2: Camera 업데이트 (LiDAR 업데이트 결과를 prior로 사용)
    S_C = H_cam @ P_after_lidar @ H_cam.T + R_cam
    K_C = P_after_lidar @ H_cam.T @ np.linalg.inv(S_C)
    y_C = z_cam - H_cam @ x_after_lidar  # innovation
    x_updated = x_after_lidar + K_C @ y_C
    P_updated = (np.eye(len(x_pred)) - K_C @ H_cam) @ P_after_lidar
    
    return x_updated, P_updated
```

### 8.2.4 Multimodal Factor Graph 설계 비교

세 시스템의 설계를 factor graph 관점에서 비교한다:

| 측면 | R3LIVE | LVI-SAM | FAST-LIVO2 |
|------|--------|---------|------------|
| 백엔드 | IEKF (dual subsystem) | iSAM2 (factor graph) | IEKF (sequential) |
| LiDAR 처리 | Direct (point-to-plane) | Feature-based (LOAM) | Direct (point-to-plane) |
| 카메라 처리 | Direct (photometric) | Feature-based (ORB) | Direct (photometric) |
| 맵 표현 | ikd-Tree + RGB | Voxel map | Hash+Octree voxel map |
| 특징 추출 | 불필요 | 필요 (edge/planar, ORB) | 불필요 |
| GPS 통합 | 없음 | Factor로 통합 | 없음 |
| Loop closure | 없음 | Factor로 통합 | 없음 |
| 임베디드 지원 | 제한적 | 제한적 | ARM 실시간 가능 |

**선택 기준**:
- Loop closure와 GPS가 필요하면: LVI-SAM
- 최고 정밀도의 colored map이 필요하면: R3LIVE
- 임베디드 플랫폼에서 실시간이 필요하면: FAST-LIVO2
- 특징점이 부족한 환경(텍스처 없는 벽, 구조물 내부)에서: direct 방식 (R3LIVE, FAST-LIVO2)

---

## 8.3 GNSS 통합

GNSS (Global Navigation Satellite System)는 전역적 위치 참조(global position reference)를 제공하는 유일한 센서이다. IMU+LiDAR+카메라가 아무리 정밀해도, 이들은 모두 **상대적(relative)** 측정을 제공할 뿐이므로, 장시간 주행하면 드리프트가 누적된다. GNSS는 이 드리프트를 교정하는 앵커 역할을 한다.

### 8.3.1 GNSS Factor in Factor Graph (LIO-SAM 방식)

LIO-SAM (Shan et al. 2020)은 GNSS를 factor graph에 통합하는 방법을 보여준다. GNSS 수신기가 위치를 보고하면, 이를 **unary factor**로 포즈 노드에 연결한다:

$$
\mathbf{r}^{\text{GPS}}_i = \mathbf{T}^{-1}_{\text{ENU→map}} \cdot \mathbf{p}^{\text{ENU}}_{\text{GPS}} - \mathbf{p}^{\text{map}}_i - \mathbf{R}^{\text{map}}_i \cdot \mathbf{l}_{\text{antenna}}
$$

여기서:
- $\mathbf{p}^{\text{ENU}}_{\text{GPS}}$는 GNSS가 보고한 ENU 좌표
- $\mathbf{T}_{\text{ENU→map}}$은 ENU 좌표계에서 SLAM 맵 좌표계로의 변환
- $\mathbf{p}^{\text{map}}_i$는 SLAM이 추정한 로봇 위치
- $\mathbf{l}_{\text{antenna}}$는 GNSS 안테나와 로봇 body frame 사이의 lever arm 벡터
- $\mathbf{R}^{\text{map}}_i$는 로봇의 회전

**좌표계 정렬 문제**: SLAM의 로컬 맵 좌표계와 GNSS의 글로벌 좌표계(WGS84/ENU)는 다르다. 첫 번째 GNSS 수신 시점에서 ENU 원점을 설정하고, 초기 포즈들을 이용하여 맵↔ENU 변환을 추정해야 한다. 이 변환은 6-DoF (3 translation + 3 rotation)이지만, IMU가 중력 방향을 제공하므로 실제로는 4-DoF (yaw + 3 translation)만 추정하면 된다.

### 8.3.2 Loosely vs Tightly Coupled GNSS

**Loosely Coupled GNSS-INS**:
GNSS 수신기가 이미 계산한 위치/속도 해(PVT solution)를 EKF로 IMU 추정치와 결합한다. 대부분의 상용 시스템이 이 방식이다.

```python
def gnss_loose_coupling_ekf_update(x_ins, P_ins, gnss_position, R_gnss):
    """
    Loosely coupled GNSS-INS: GNSS PVT 솔루션으로 INS 상태 보정.
    
    x_ins: INS 상태 [position(3), velocity(3), attitude(3), biases(6)] = 15차원
    gnss_position: GNSS가 계산한 위치 (3,)
    R_gnss: GNSS 위치의 공분산 (3, 3) — 보통 HDOP * sigma_uere
    """
    n = len(x_ins)
    # 관측 행렬: GNSS는 위치만 관측
    H = np.zeros((3, n))
    H[0:3, 0:3] = np.eye(3)  # position 부분만 관측
    
    # Innovation
    y = gnss_position - H @ x_ins
    
    # Kalman gain
    S = H @ P_ins @ H.T + R_gnss
    K = P_ins @ H.T @ np.linalg.inv(S)
    
    # Update
    x_updated = x_ins + K @ y
    P_updated = (np.eye(n) - K @ H) @ P_ins
    
    return x_updated, P_updated
```

**Tightly Coupled GNSS-INS**:
GNSS 수신기의 PVT 솔루션이 아니라, 원시 의사거리(pseudorange)와 도플러(Doppler) 측정을 직접 사용한다. 각 위성에 대한 의사거리를 개별 factor로 삽입한다:

$$
\rho_i = \| \mathbf{p}_{\text{sat},i} - \mathbf{p}_{\text{rx}} \| + c \cdot \delta t_{\text{rx}} + I_i + T_i + \epsilon_i
$$

여기서 $\rho_i$는 위성 $i$에 대한 의사거리, $c \cdot \delta t_{\text{rx}}$는 수신기 시계 바이어스, $I_i$와 $T_i$는 전리층/대류층 지연이다.

Tightly coupled의 장점은, 위성이 4개 미만이어서 GNSS 자체적으로는 해를 구할 수 없는 상황에서도, 가용한 위성의 의사거리를 여전히 활용할 수 있다는 점이다. 도심 환경에서 건물에 의해 위성이 가려지는 경우가 빈번하므로, 이 장점은 실질적으로 매우 크다.

### 8.3.3 GNSS-Denied → GNSS-Available 전환 처리

실제 로봇 운행에서는 GNSS 신호가 수시로 끊기고 복원된다 (터널, 지하주차장, 고가도로 아래). 이 전환을 안정적으로 처리하는 것이 시스템 설계의 핵심 과제이다.

**전환 시 주의점**:
1. **좌표계 점프(coordinate jump) 방지**: GNSS 복원 직후 GNSS 위치와 IMU/LiDAR 추정 위치 사이에 큰 차이가 있을 수 있다. 이를 갑자기 교정하면 맵에 불연속이 생긴다. 해결책은 GNSS 불확실성을 초기에 크게 설정하고 점진적으로 줄이는 것이다.

2. **GNSS 품질 검증**: GNSS 복원 후 초기 몇 초의 측정은 multipath 등으로 정확도가 떨어질 수 있다. PDOP/HDOP, 위성 수, carrier phase 상태 등을 확인하여 충분히 신뢰할 수 있을 때만 factor에 포함한다.

3. **맵 좌표계 보정**: GNSS 장기 부재 후 드리프트가 누적되었다면, 복원 시 맵 좌표계 자체를 보정해야 할 수 있다. 이는 loop closure와 유사한 포즈 그래프 최적화로 처리한다.

---

## 8.4 Radar 퓨전

### 8.4.1 Radar의 재조명

전통적으로 자동차 레이더는 해상도가 낮아 SLAM/odometry 용으로는 적합하지 않다고 여겨졌다. 그러나 **4D imaging radar**의 등장으로 그 평가가 달라지고 있다.

**4D Radar란**: 기존 자동차 레이더가 거리(range), 속도(Doppler), 방위각(azimuth) 3가지를 측정했다면, 4D imaging radar는 여기에 **고도각(elevation)**을 추가하여 3D 포인트 클라우드를 생성한다. 해상도는 LiDAR에 비할 바가 아니지만(수백~수천 점 vs 수십만 점), 독보적인 장점들이 있다.

**Radar의 고유한 장점**:

1. **악천후 관통**: 비, 눈, 안개, 먼지를 관통한다. LiDAR(905nm/1550nm 레이저)는 이런 조건에서 성능이 크게 저하되지만, 레이더(mm-wave)는 영향을 거의 받지 않는다. 자율주행 안전성 관점에서 이 차이는 중요하다.

2. **직접 속도 측정**: FMCW (Frequency-Modulated Continuous Wave) 레이더는 도플러 효과를 이용하여 물체의 **상대 속도를 직접 측정**한다. 카메라나 LiDAR는 연속 프레임 비교로 속도를 간접 추정해야 하지만, 레이더는 단일 측정에서 속도를 얻는다.

3. **저렴한 비용**: 자동차 레이더 칩셋은 대량 생산으로 LiDAR 대비 한 자릿수 이상 저렴하다.

### 8.4.2 Radar Odometry

4D radar를 이용한 odometry는 2022년 이후 논문 수가 빠르게 늘고 있는 분야이다. 핵심 아이디어는 radar의 도플러 측정을 ego-motion 추정에 직접 활용하는 것이다.

FMCW radar의 각 측정점은 $(r, \theta, \phi, v_d)$ — 거리, 방위각, 고도각, 도플러 속도 — 를 제공한다. 로봇의 선속도 $\mathbf{v}$와 각속도 $\boldsymbol{\omega}$가 주어지면, 특정 방향 $\mathbf{d}_i = [\cos\phi_i \cos\theta_i, \cos\phi_i \sin\theta_i, \sin\phi_i]^T$의 점에서 관측되는 도플러 속도는:

$$
v_{d,i} = -\mathbf{d}_i^T (\mathbf{v} + \boldsymbol{\omega} \times \mathbf{p}_i) + n_i
$$

여기서 $\mathbf{p}_i = r_i \mathbf{d}_i$는 점의 3D 위치이다. 정적 점들만 사용하면 (움직이는 물체 제거 후), 이 방정식들의 집합으로부터 $(\mathbf{v}, \boldsymbol{\omega})$를 추정할 수 있다.

```python
import numpy as np

def radar_ego_velocity(radar_points, doppler_velocities):
    """
    Radar 도플러 측정으로부터 ego-velocity 추정.
    
    radar_points: (N, 3) — 각 점의 3D 좌표 (r*d_i)
    doppler_velocities: (N,) — 각 점의 관측 도플러 속도
    
    Returns:
        v_ego: (3,) — ego linear velocity
    """
    # 각 점의 방향 벡터 (단위 벡터)
    norms = np.linalg.norm(radar_points, axis=1, keepdims=True)
    directions = radar_points / (norms + 1e-8)  # (N, 3)
    
    # v_d = -d^T @ v_ego  (간단한 경우: 각속도 무시)
    # => A @ v_ego = b, 여기서 A = -directions, b = doppler_velocities
    A = -directions
    b = doppler_velocities
    
    # RANSAC으로 동적 물체 제거 후 최소자승
    # 간단한 버전: 전체 데이터로 최소자승
    v_ego, residuals, _, _ = np.linalg.lstsq(A, b, rcond=None)
    
    return v_ego
```

### 8.4.3 4D Radar + Camera Fusion

4D radar와 카메라의 조합은 "LiDAR-free" 자율주행의 유력한 대안으로 주목받고 있다. 두 센서의 상보성은 다음과 같다:

| 특성 | 카메라 | 4D Radar |
|------|--------|----------|
| 해상도 | 매우 높음 | 낮음 |
| 악천후 | 취약 | 강건 |
| 직접 깊이 측정 | ✗ | ✓ |
| 직접 속도 측정 | ✗ | ✓ |
| 시맨틱 이해 | 강함 | 약함 |
| 비용 | 매우 저렴 | 저렴 |

Fusion 접근법:
- **Early fusion**: Radar 포인트를 이미지에 투영하여 sparse depth cue로 활용. Mono depth estimation의 스케일 앵커로 사용.
- **Mid-level fusion**: 카메라 특징과 radar 특징을 네트워크 내부에서 결합. BEV (Bird's Eye View) 공간에서의 융합이 일반적.
- **Late fusion**: 각 센서로 독립적으로 물체 검출 후 결과를 결합.

### 8.4.4 Boreas 벤치마크

[Boreas](https://arxiv.org/abs/2203.10168) (Burnett et al., 2023)는 다양한 기상 조건(맑음, 비, 눈)에서 수집된 다중 센서 데이터셋으로, 특히 radar odometry의 벤치마킹에 중요하다. 카메라, LiDAR(Velodyne Alpha Prime), 4D radar(Navtech CIR304-H)를 동시에 탑재하고, 같은 경로를 다른 시간/계절에 반복 주행하여 long-term localization 연구에도 활용된다.

---

## 8.5 Multi-Robot / Decentralized Fusion

단일 로봇의 퓨전을 넘어, **여러 로봇이 협력적으로 환경을 인식**하는 문제는 난이도가 한 단계 더 높다. 통신 제약, 상대적 참조 프레임의 부재, 데이터 연관(data association)의 어려움이 추가되기 때문이다.

### 8.5.1 Multi-Robot SLAM의 핵심 과제

1. **상대 포즈 추정(Inter-Robot Relative Pose)**: 각 로봇은 자기만의 로컬 좌표계에서 SLAM을 수행한다. 두 로봇의 맵을 병합하려면 먼저 이들의 상대 좌표 변환을 알아야 한다. 이는 cross-robot place recognition + geometric verification으로 해결한다.

2. **통신 제약(Communication Constraint)**: 전체 맵이나 raw 센서 데이터를 전송하는 것은 대역폭 문제로 불가능한 경우가 많다. **어떤 정보를 압축하여 공유할 것인가**가 핵심 설계 결정이다.

3. **분산 최적화(Distributed Optimization)**: 중앙 서버에 모든 데이터를 모아 최적화하면 통신 병목과 단일 실패점(single point of failure) 문제가 생긴다. 각 로봇이 로컬 최적화를 하면서 이웃 로봇과 제한된 정보만 교환하는 분산 방식이 낫다.

### 8.5.2 Kimera-Multi

[Kimera-Multi](https://arxiv.org/abs/2106.14386) (Rosinol et al., 2021)는 MIT의 SPARK Lab에서 개발한 분산 멀티로봇 SLAM 시스템이다.

**아키텍처**:
- 각 로봇이 Kimera를 실행하여 로컬 metric-semantic SLAM 수행
- 로봇 간 조우(rendezvous) 시, **DBoW2** 기반 inter-robot place recognition으로 공통 장소를 탐지
- 탐지된 inter-robot loop closure를 분산 포즈 그래프 최적화에 반영
- **GNC (Graduated Non-Convexity)** 솔버로 이상치 loop closure를 걸러낸다

**분산 최적화**: 각 로봇이 자신의 포즈 그래프를 유지하면서 이웃 로봇과 inter-robot factor만 교환한다. Riemannian block-coordinate descent 등의 분산 최적화 알고리즘으로 수렴한다.

### 8.5.3 Swarm-SLAM

[Swarm-SLAM](https://arxiv.org/abs/2301.06230) (Lajoie et al., 2024)은 대규모 로봇 군집(swarm)을 위한 분산 SLAM으로, 통신 효율에 집중한다.

**핵심 설계**:
- **Place Recognition Descriptor 교환**: 전체 맵이 아니라 장소 인식 디스크립터(NetVLAD, Scan Context 등)만 교환하여 대역폭을 최소화
- **Inter-Robot Loop Closure**: 디스크립터 매칭으로 후보를 찾고, 최소한의 기하학적 정보(특징점 or 포인트 클라우드)만 교환하여 검증
- **인접 로봇 간 피어투피어 통신**: 중앙 서버 없이 인접 로봇 간 직접 통신
- **LiDAR/Visual/Multimodal 지원**: 카메라만, LiDAR만, 또는 혼합 센서 구성의 로봇들이 동시에 참여 가능

```python
# 분산 포즈 그래프 최적화의 개념적 구현
import numpy as np

class DistributedPoseGraphNode:
    """
    분산 포즈 그래프의 단일 로봇 노드.
    각 로봇은 자신의 로컬 그래프를 유지하고,
    이웃 로봇과 inter-robot factor만 교환한다.
    """
    def __init__(self, robot_id):
        self.robot_id = robot_id
        self.local_poses = []           # 자체 포즈 (로컬 좌표계)
        self.local_factors = []          # 로컬 odometry factor
        self.inter_robot_factors = []    # 다른 로봇과의 loop closure factor
        self.neighbor_info = {}          # 이웃 로봇으로부터 받은 경계 정보
    
    def add_odometry(self, delta_pose, covariance):
        """로컬 odometry factor 추가"""
        self.local_factors.append({
            'type': 'odom',
            'from': len(self.local_poses) - 1,
            'to': len(self.local_poses),
            'measurement': delta_pose,
            'covariance': covariance
        })
        self.local_poses.append(self.local_poses[-1] @ delta_pose)
    
    def add_inter_robot_factor(self, other_robot_id, other_pose_idx, 
                                relative_pose, covariance):
        """다른 로봇과의 loop closure factor 추가"""
        self.inter_robot_factors.append({
            'type': 'inter_robot',
            'robot': other_robot_id,
            'local_idx': len(self.local_poses) - 1,
            'remote_idx': other_pose_idx,
            'measurement': relative_pose,
            'covariance': covariance
        })
    
    def exchange_boundary_info(self, neighbor_node):
        """
        이웃 노드와 경계 정보(boundary variable의 추정치와 공분산)를 교환.
        전체 맵이 아니라 inter-robot factor에 관련된 변수만 교환.
        """
        boundary_poses = []
        for factor in self.inter_robot_factors:
            if factor['robot'] == neighbor_node.robot_id:
                idx = factor['local_idx']
                boundary_poses.append({
                    'idx': idx,
                    'pose': self.local_poses[idx]
                })
        neighbor_node.neighbor_info[self.robot_id] = boundary_poses
```

---

## 8.6 시스템 설계 실전

이론과 알고리즘을 넘어, 실제 멀티센서 퓨전 시스템을 설계하고 배포할 때 직면하는 실전적 문제들을 다룬다.

### 8.6.1 Sensor Suite 선정 가이드

센서 스위트의 선정은 **운용 환경(operational environment)**에 의해 결정된다:

| 환경 | 권장 최소 구성 | 선택 추가 센서 |
|------|---------------|---------------|
| 실내 (사무실/창고) | Camera + IMU | LiDAR (정밀 매핑 시) |
| 도심 자율주행 | Camera + LiDAR + IMU + GNSS | 4D Radar, Wheel Odom |
| 비포장/야외 | LiDAR + IMU + GNSS | Camera (시맨틱), Radar |
| 지하/터널 | LiDAR + IMU | Camera, UWB |
| 수중 | IMU + DVL (Doppler Velocity Log) | Sonar, Pressure |
| 항공/드론 | Camera + IMU + GNSS | LiDAR (매핑 시) |
| 악천후 (비/눈) | Radar + IMU | Camera, LiDAR |

**예산별 구성 예시**:
- **$500 이하**: Stereo Camera + IMU (Intel RealSense D435i)
- **$2,000 이하**: + 2D LiDAR (RPLidar)
- **$10,000 이하**: + 3D LiDAR (Livox Mid-360) + GNSS RTK
- **$30,000+**: Multi-LiDAR + Multi-Camera + 4D Radar + GNSS RTK

### 8.6.2 Timing Architecture (시간 동기화 설계)

멀티센서 시스템에서 **시간 동기화**는 정확도를 좌우하는 결정적 요소이다. 100km/h로 이동하는 차량에서 1ms의 시간 오차는 약 2.8cm의 위치 오차에 해당한다.

**하드웨어 동기화 (Hardware Sync)**:

가장 정밀한 방법은 하드웨어 트리거를 사용하는 것이다:

- **PPS (Pulse Per Second)**: GNSS 수신기가 1초마다 정밀한 펄스를 출력. 이 펄스를 다른 센서의 동기화 입력에 연결. 정밀도: ~50ns.
- **PTP (Precision Time Protocol, IEEE 1588)**: 이더넷 기반 시간 동기화. LiDAR(Velodyne, Ouster 등)가 지원. 정밀도: ~μs.
- **외부 트리거**: 마이크로컨트롤러가 카메라 셔터 트리거와 IMU 타임스탬프 캡처를 동시에 수행.

**소프트웨어 동기화 (Software Sync)**:

하드웨어 동기화가 불가능한 경우, 소프트웨어적으로 시간 오프셋을 추정한다:

- **Kalibr 방식**: B-spline 궤적으로 연속 시간 궤적을 표현하고, 센서 간 시간 오프셋을 최적화 변수로 포함하여 동시 추정.
- **상관 기반**: 두 센서의 운동 추정 결과 사이의 상호상관(cross-correlation)을 계산하여 시간 지연을 추정.

$$
\hat{\tau} = \arg\max_{\tau} \int \mathbf{a}_{\text{IMU}}(t) \cdot \dot{\mathbf{v}}_{\text{camera}}(t + \tau) \, dt
$$

```python
import numpy as np
from scipy.signal import correlate

def estimate_time_offset(timestamps_a, signal_a, timestamps_b, signal_b, max_offset_ms=100):
    """
    두 센서 신호의 상호상관으로 시간 오프셋 추정.
    
    예: IMU 각속도 vs 카메라 프레임 간 회전률
    """
    # 공통 타임라인에 리샘플링 (1kHz)
    dt = 0.001  # 1ms
    t_common = np.arange(
        max(timestamps_a[0], timestamps_b[0]),
        min(timestamps_a[-1], timestamps_b[-1]),
        dt
    )
    sig_a = np.interp(t_common, timestamps_a, signal_a)
    sig_b = np.interp(t_common, timestamps_b, signal_b)
    
    # 평균 제거
    sig_a -= np.mean(sig_a)
    sig_b -= np.mean(sig_b)
    
    # 상호상관
    correlation = correlate(sig_a, sig_b, mode='full')
    lags = np.arange(-len(sig_b) + 1, len(sig_a)) * dt
    
    # max_offset 범위 내에서 최대 상관 찾기
    mask = np.abs(lags) <= max_offset_ms / 1000
    valid_corr = correlation[mask]
    valid_lags = lags[mask]
    
    best_idx = np.argmax(valid_corr)
    estimated_offset = valid_lags[best_idx]
    
    return estimated_offset  # 초 단위

# 예시: IMU gyro Z축 vs Camera rotation rate
# offset = estimate_time_offset(imu_times, gyro_z, cam_times, cam_rotation_rate)
```

### 8.6.3 Failure Mode와 Degradation Handling

실제 시스템에서 센서는 반드시 실패한다. 강건한 시스템은 **graceful degradation** — 즉, 한 센서가 실패해도 성능이 다소 저하되면서 나머지 센서로 계속 동작하는 것 — 을 달성해야 한다.

**주요 실패 모드와 대응**:

| 실패 모드 | 증상 | 탐지 방법 | 대응 |
|-----------|------|-----------|------|
| 카메라 과노출/저노출 | 이미지 전체가 밝거나 어두움 | 히스토그램 분석 | 카메라 factor 비활성화, LIO만으로 동작 |
| LiDAR 기하 퇴화 (degenerate) | 긴 복도, 넓은 평지 | 정보 행렬의 고유값 분석 | 해당 DoF의 LiDAR 구속 완화, VIO로 보완 |
| IMU 포화 | 고속 충격 시 측정 범위 초과 | ADC 최대값 탐지 | 해당 시간대 IMU preintegration 불확실성 증가 |
| GNSS multipath | 건물 반사로 인한 큰 오차 | RAIM, 잔차 검사 | 해당 GNSS factor의 공분산 증가 또는 제거 |
| 센서 완전 단절 | 데이터 수신 없음 | Watchdog timer | 해당 센서의 모든 factor 비활성화 |

**LiDAR 기하 퇴화 탐지**:

LiDAR scan matching에서 정보 행렬(Hessian) $\mathbf{H} = \mathbf{J}^T \mathbf{J}$의 고유값 분석으로 기하 퇴화를 탐지할 수 있다. 만약 한 방향의 고유값이 다른 방향들에 비해 현저히 작으면, 그 방향으로의 구속이 약하다는 뜻이다.

$$
\mathbf{H} = \mathbf{U} \boldsymbol{\Lambda} \mathbf{U}^T, \quad \lambda_{\min} / \lambda_{\max} < \epsilon \Rightarrow \text{degenerate}
$$

예를 들어, 긴 복도에서 복도 축 방향의 구속이 약해지므로, 그 방향의 LiDAR 구속을 완화하고 카메라의 optical flow로 보완한다.

```python
import numpy as np

def check_lidar_degeneracy(jacobian, threshold=0.01):
    """
    LiDAR scan matching의 Hessian 고유값으로 기하 퇴화 검사.
    
    jacobian: (m, 6) — m개 point-to-plane 잔차의 6-DoF 자코비안
    threshold: 최소/최대 고유값 비의 임계값
    
    Returns:
        is_degenerate: bool
        degenerate_directions: (k, 6) — 퇴화된 방향의 고유벡터
        eigenvalues: (6,) — 정보 행렬의 고유값
    """
    # 정보 행렬 (근사 Hessian)
    H = jacobian.T @ jacobian
    
    # 고유값 분해
    eigenvalues, eigenvectors = np.linalg.eigh(H)
    
    # 고유값 비율 검사
    ratio = eigenvalues / (eigenvalues.max() + 1e-10)
    degenerate_mask = ratio < threshold
    
    is_degenerate = np.any(degenerate_mask)
    degenerate_directions = eigenvectors[:, degenerate_mask].T
    
    if is_degenerate:
        print(f"[경고] 기하 퇴화 감지!")
        print(f"  고유값: {eigenvalues}")
        print(f"  퇴화 방향 수: {degenerate_mask.sum()}")
    
    return is_degenerate, degenerate_directions, eigenvalues


def adaptive_fusion_weight(lidar_eigenvalues, camera_track_quality, 
                            lidar_min_eig_threshold=100.0):
    """
    LiDAR 퇴화 정도에 따라 카메라 가중치를 적응적으로 조절.
    """
    min_eig = lidar_eigenvalues.min()
    
    if min_eig > lidar_min_eig_threshold:
        # LiDAR 충분히 구속됨 → 기본 가중치
        lidar_weight = 1.0
        camera_weight = 0.3
    else:
        # LiDAR 퇴화 → 카메라 가중치 증가
        decay = min_eig / lidar_min_eig_threshold
        lidar_weight = decay
        camera_weight = 1.0
    
    return lidar_weight, camera_weight
```

### 8.6.5 최근 주목할 연구 (2024-2025)

- **[Gaussian-LIC (Lang et al., ICRA 2025)](https://arxiv.org/abs/2404.06926)**: 3D Gaussian Splatting을 LiDAR-Inertial-Camera tightly-coupled SLAM에 통합한 시스템. LiDAR의 정밀한 기하 정보와 카메라의 텍스처를 Gaussian 표현으로 융합하여, SLAM과 동시에 photo-realistic한 장면 복원을 달성한다.
- **[Snail-Radar (Huai et al., IJRR 2025)](https://arxiv.org/abs/2407.11705)**: 4D radar SLAM 평가를 위한 대규모 다양성 벤치마크. 다양한 환경(실내·실외, 도심·교외)과 플랫폼에서 4D radar 기반 odometry/SLAM 알고리즘을 체계적으로 비교한다.

### 8.6.4 시스템 설계 체크리스트

실제 멀티센서 퓨전 시스템을 설계할 때 반드시 확인해야 할 항목들:

**캘리브레이션**:
- [ ] 모든 센서 쌍의 extrinsic calibration 완료
- [ ] 시간 동기화 오프셋 측정/추정 완료
- [ ] 캘리브레이션 결과의 재현성 검증 (3회 이상 반복)
- [ ] 온라인 캘리브레이션 드리프트 보정 메커니즘 존재

**데이터 흐름**:
- [ ] 각 센서의 데이터 레이트와 시스템의 처리 레이트 매칭
- [ ] 센서 간 데이터 정렬 (temporal alignment) 방법 확정
- [ ] 버퍼 크기와 지연시간(latency) 분석

**Robustness**:
- [ ] 각 센서의 실패 모드 식별
- [ ] Degradation handling 전략 수립
- [ ] Outlier rejection 메커니즘 (robust kernel, chi-square test)
- [ ] 극단 환경 테스트 (어둠, 비, 진동, 기하 퇴화)

**성능**:
- [ ] 목표 정확도 (ATE/RPE) 정의
- [ ] 실시간 제약 충족 여부 (worst-case latency)
- [ ] 메모리 사용량 (장시간 운행 시 누적)
- [ ] CPU/GPU 점유율

---

## 8장 요약

멀티센서 퓨전의 아키텍처는 크게 loosely/tightly/ultra-tightly coupled로 분류되며, 현대 로보틱스에서는 **tightly coupled**가 주류이다. Camera+LiDAR+IMU 삼중 융합은 R3LIVE, LVI-SAM, FAST-LIVO2 등의 시스템으로 성숙 단계에 접어들었고, 각각 dual subsystem, factor graph, sequential update라는 서로 다른 설계 철학을 보여준다.

GNSS 통합은 전역 좌표 앵커를 제공하여 장기 드리프트를 해결하며, 4D radar는 악천후 robustness와 직접 속도 측정이라는 고유한 장점으로 급부상하고 있다. Multi-robot 퓨전은 통신 제약 하에서의 분산 최적화와 cross-robot place recognition이 핵심 도전이며, Kimera-Multi와 Swarm-SLAM이 이 분야를 선도한다.

마지막으로, 실전 시스템 설계에서는 센서 선정, 시간 동기화, failure mode 대응이 알고리즘만큼이나 중요하며, 이를 체계적으로 다루는 엔지니어링 역량이 성공적인 배포를 좌우한다.

Ch.6-8에서 다룬 odometry/fusion 시스템은 로컬 정밀도는 높지만, 장시간 운행하면 드리프트가 누적된다. 이 드리프트를 교정하려면 "과거에 방문했던 장소를 다시 인식하는" 능력이 필요하다. 다음 챕터에서는 이 핵심 컴포넌트인 **Place Recognition**을 다룬다.
