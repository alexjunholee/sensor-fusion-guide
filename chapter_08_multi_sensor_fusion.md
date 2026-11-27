# 8장 다중 센서 융합 구조

> Visual Odometry와 LiDAR Odometry를 한 시스템으로 엮으려면 결합 수준과 추정 구조를 정해야 한다.

카메라 기반 오도메트리(→ 6장)와 LiDAR 기반 오도메트리(→ 7장)는 각기 IMU와 짝을 이뤄 준수한 궤적을 만들어 내지만, 단일 모달리티에 기댄 이상 환경이 나빠지면 약점이 그대로 드러난다. 카메라는 어둡거나 텍스처가 없는 벽 앞에서 추적을 놓치기 쉽고, LiDAR는 기하학적 제약이 부족한 긴 터널이나 안개·강수 환경에서 정합이 흔들린다. 서로 다른 센서를 한 플랫폼에 얹어 빈틈을 메우려면, 데이터를 어느 깊이에서 합칠지(결합 수준)와 상태를 어떤 추정기로 풀지(추정 구조)부터 설계해야 한다.

---

## 8.1 Fusion 아키텍처 분류

멀티센서 융합 아키텍처를 설계할 때는 센서 데이터를 파이프라인의 어느 단계에서 합칠지부터 결정해야 한다. 이미 요약된 포즈 추정치만 주고받을지, 원시 측정을 하나의 추정기로 끌어올지, 아니면 센서 내부 신호 처리 루프까지 맞물리게 할지에 따라 시스템의 복잡도와 성능, 그리고 한 센서가 무너졌을 때의 실패 양상이 달라진다.

### 8.1.1 느슨한 결합 (Loosely Coupled)

느슨한 결합에서는 각 센서 서브시스템이 독립적으로 상태를 추정한 뒤, 상위 융합 단계에서 그 결과만 모아 합친다. 예를 들어 LiDAR 오도메트리 모듈이 스캔 매칭으로 $\mathbf{T}_{L}$을 계산하고 Visual 오도메트리 모듈이 영상으로 $\mathbf{T}_{V}$를 구해 넘기면, 융합 모듈은 두 포즈 추정치를 각각의 불확실성으로 가중 평균한다. 이를 최적화 문제로 적으면 두 추정치로부터의 마할라노비스 거리를 최소화하는 형태가 된다.

$$
\hat{\mathbf{x}} = \arg\min_{\mathbf{x}} \left\| \mathbf{x} - \mathbf{x}_{LiDAR} \right\|^2_{\mathbf{P}_{L}^{-1}} + \left\| \mathbf{x} - \mathbf{x}_{Visual} \right\|^2_{\mathbf{P}_{V}^{-1}}
$$

여기서 $\mathbf{P}_{L}$과 $\mathbf{P}_{V}$는 각 서브시스템이 보고하는 공분산이며, 그 역행렬이 곧 각 추정치에 부여되는 신뢰도(정보 행렬)다.

이 구조의 강점은 명확한 모듈성에 있다. 센서 모듈을 독립적으로 교체하거나 업그레이드하기 쉽고, 오류가 발생했을 때 어느 서브시스템에서 비롯됐는지 분리하기도 수월하다. 한쪽 센서가 고장 나더라도 나머지 모듈은 그대로 추정치를 내보낸다.

반면 서브시스템마다 내부 정보를 포즈와 공분산으로 압축해 버리므로 정보 손실이 뒤따른다. 센서끼리 원시 수준에서 서로의 약점을 메워 주지 못한다는 뜻이다. 가령 LiDAR의 3D 기하는 단안 카메라의 스케일 모호성을 직접 풀어 줄 수 있지만, loosely coupled 구조에서는 카메라 모듈이 LiDAR의 깊이 정보를 들여다보지 못한다. 게다가 서브시스템이 보고하는 공분산의 일관성(consistency)을 별도로 검증해야 하는데, 한 모듈이 자기 공분산을 지나치게 낙관적으로 잡으면 융합 결과 전체가 그쪽으로 왜곡되기 때문이다.

### 8.1.2 긴밀한 결합 (Tightly Coupled)

느슨한 결합이 요약된 포즈만 주고받는다면, 긴밀한 결합에서는 모든 센서의 원시 측정(raw measurement)이 하나의 공유 상태를 직접 제약한다. 아키텍처를 분류하는 기준은 최적화기 개수가 아니라 측정이 들어가는 수준이다. 따라서 [R3LIVE (Lin et al. 2022)](https://arxiv.org/abs/2109.07982)처럼 두 하위 시스템이 공유 상태 하나를 함께 갱신하는 구성이나 FAST-LIVO2처럼 순차 갱신을 쓰는 구성은 tightly coupled에 해당한다. LVI-SAM처럼 두 하위 시스템이 각자 그래프를 유지하더라도, 서로의 원시 측정(LiDAR 깊이)과 상태(초기값)를 직접 주고받는다면 역시 tightly coupled로 분류한다.

Factor graph 관점에서는 각 센서의 원시 측정이 저마다 잔차 항(factor)으로 들어가 공통 상태 $\mathbf{x}$를 함께 조인다.

$$
\hat{\mathbf{x}} = \arg\min_{\mathbf{x}} \sum_{i} \left\| \mathbf{r}^{\text{IMU}}_i(\mathbf{x}) \right\|^2_{\boldsymbol{\Sigma}^{-1}_{\text{IMU}}} + \sum_{j} \left\| \mathbf{r}^{\text{LiDAR}}_j(\mathbf{x}) \right\|^2_{\boldsymbol{\Sigma}^{-1}_{\text{LiDAR}}} + \sum_{k} \left\| \mathbf{r}^{\text{cam}}_k(\mathbf{x}) \right\|^2_{\boldsymbol{\Sigma}^{-1}_{\text{cam}}}
$$

여기서 $\mathbf{r}^{\text{IMU}}_i$는 IMU preintegration 잔차, $\mathbf{r}^{\text{LiDAR}}_j$는 point-to-plane 잔차, $\mathbf{r}^{\text{cam}}_k$는 reprojection error에 해당한다.

원시 측정이 한 상태 공간에 모이면 센서 간 상호 보완이 직접 일어난다. IMU의 고주파 관성 측정이 LiDAR 스캔의 모션 왜곡을 펴 주고, LiDAR의 거리 측정이 VIO의 미터 단위 스케일을 곧바로 잡아 준다. 관측 모델과 잡음 모델이 정확하다면 중간 요약을 거치지 않으므로 loosely coupled보다 정보 손실이 적다.

그 대가로 설계와 운영의 복잡도가 크게 올라간다. 모든 센서의 관측 모델, 노이즈 공분산, 시간 동기화를 한 프레임워크 안에서 일관되게 다뤄야 하며, 한 센서에서 들어온 이상치(outlier)가 공유 상태를 타고 전체 추정을 흔들 수 있어 강건한 이상치 제거가 필수적이다. 연산량이 늘어 실시간 주기를 맞추는 일도 한층 까다로워진다. 이 계보의 대표 시스템으로는 [LIO-SAM (Shan et al. 2020)](https://arxiv.org/abs/2007.00258) (LiDAR+IMU+GPS), VINS-Mono (Camera+IMU), R3LIVE (Camera+LiDAR+IMU)가 있다.

### 8.1.3 신호 수준 결합 (Ultra-Tightly Coupled)

한 걸음 더 들어가 센서 내부의 신호 처리 단계에서 결합하는 방식도 있다. 대표적인 사례가 GNSS-INS ultra-tight coupling이다. Tightly coupled GNSS-INS는 수신기가 위성 신호에서 복조해 낸 의사거리(pseudorange)를 INS와 합치지만, ultra-tight에서는 INS가 추정한 위치·속도를 GNSS 수신기의 코드·반송파 추적 루프(tracking loop)에 직접 입력한다. INS가 예측한 도플러 이동량으로 수신기 내부 수치 제어 발진기(NCO, Numerically Controlled Oscillator)의 주파수를 보정해 주는 식이다.

$$
\text{NCO frequency} = f_{\text{nominal}} + \Delta f_{\text{INS-aided}}
$$

이 보정 덕분에 수신기의 주파수 추적 범위가 넓어지므로, 도심 캐니언이나 실내 진입 직후처럼 신호가 급격히 약해지는 구간에서도 위성 신호를 훨씬 오래 붙잡고 갈 수 있다.

**구현 경계**: Ultra-tight coupling은 GNSS 수신기의 correlator나 tracking loop에 직접 접근해야 하므로, 수신기 SDK·FPGA·펌웨어까지 제어할 수 있는 통합 GNSS/INS 제품이나 전용 연구 플랫폼에서 주로 구현한다. 일반 로보틱스 개발 환경에서는 상용 수신기가 내보내는 원시 의사거리·반송파·도플러 측정을 활용하는 tightly coupled 구성이 훨씬 현실적이다.

### 8.1.4 결합 수준 비교

세 결합 수준의 차이는 센서 데이터가 상태 추정기까지 흘러 들어가는 경로를 나란히 놓아 보면 한눈에 들어온다. Loosely coupled는 서브시스템의 요약 출력을 뒤단에서 합치고, tightly coupled는 원시 측정을 공유 추정기에 직접 넣으며, ultra-tight는 추정 결과가 다시 센서 내부 신호 루프로 되먹임된다.

```
측정 흐름:

[Loosely]    센서A → 서브시스템A → 포즈A ─┐
                                          ├→ Fusion → 최종 포즈
             센서B → 서브시스템B → 포즈B ─┘

[Tightly]    센서A → raw 측정A ──┐
                                 ├→ 공유 상태 추정기 → 최종 포즈
             센서B → raw 측정B ──┘   (단일 optimizer, 순차 갱신, 공유 상태를 쓰는
                                      두 하위 시스템 모두 이 수준에 해당)

[Ultra-Tight] 센서A 신호 ←→ 센서B 추정 (양방향 신호 수준 결합)
```

아래 코드를 보면 이 가운데 가장 단순한 loosely coupled 융합이 정보 형식(information form)에서 어떻게 동작하는지 알 수 있다. 두 서브시스템의 공분산을 역행렬로 바꿔 정보 행렬로 더하면, 공분산이 작은(신뢰도가 높은) 센서 쪽으로 융합 결과가 자연스럽게 끌려간다.

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
P_visual = np.diag([0.1, 0.05])       # Visual은 축마다 정밀도가 다름 (x 방향이 덜 정밀)

x_fused, P_fused = loosely_coupled_fusion(x_lidar, P_lidar, x_visual, P_visual)
print(f"LiDAR:  {x_lidar}, P_diag: {np.diag(P_lidar)}")
print(f"Visual: {x_visual}, P_diag: {np.diag(P_visual)}")
print(f"Fused:  {x_fused}, P_diag: {np.diag(P_fused)}")
# Fused 결과는 LiDAR 쪽에 더 가까움 (공분산이 더 작으므로)
```

<!-- DEMO: fusion_architecture_comparison.html -->

이어지는 절에서는 이러한 결합 원칙 중 현재 주류로 자리 잡은 tightly coupled 구조가 실제 Camera+LiDAR+IMU 시스템에서 어떻게 구현되는지 살펴본다.

---

## 8.2 Camera + LiDAR + IMU 융합

카메라, LiDAR, IMU를 한데 묶으면 텍스처와 색상, 3D 거리, 고주파 관성 측정을 모두 손에 넣을 수 있다. 그렇다고 세 센서를 달기만 해서 시스템이 저절로 강건해지지는 않는다. 비용·전력·기상 조건·유효 거리뿐 아니라, 세 센서가 같은 환경에서 동시에 실패하지 않는지부터 따져 보아야 한다. 아래 표에 세 센서가 서로 다른 물리적 한계 앞에서 어떻게 갈리는지 비교해 정리했다.

| 상황 | 카메라 | LiDAR | IMU |
|------|--------|-------|-----|
| 조명 변화 | 영향을 받음 | 자체 레이저를 쓰므로 조명과 무관 | 주변을 관측하지 않음 |
| 텍스처 없는 환경 | 데이터 연관이 어려움 | 텍스처와 무관 (기하 구조가 있어야 함) | 주변을 관측하지 않음 |
| 기하 제약이 모자란 환경 (좁은 터널, 길게 이어진 벽 하나) | 기하 구조와 무관 (텍스처가 있어야 함) | 성능이 떨어짐 | 주변을 관측하지 않음 |
| 깊이·스케일 | 직접 측정하지 못함 (단안) | 직접 측정 | 가속도가 metric 단위 (운동 여기 필요) |
| 색상 | 있음 | 없음 | 없음 |

이 표에서 카메라와 LiDAR의 한계는 [FAST-LIVO2 논문](https://arxiv.org/abs/2408.14035) 1절에서 정리한 내용이고, IMU 항목은 외부 환경을 관측하지 않는 관성 센서의 측정 원리에서 비롯된다(→ 2장). IMU는 조명이나 텍스처, 주변 기하 구조에 흔들리지 않지만 단독으로 쓰면 적분 오차가 빠르게 쌓인다. 결국 카메라는 기하가 단조로운 구간에서 LiDAR를 받쳐 주고, LiDAR는 어둡거나 텍스처가 없는 구간에서 카메라를 대신하며, IMU는 두 센서 프레임 사이의 빠른 움직임을 이어 주는 역할을 맡는다.

### 8.2.1 R3LIVE / R3LIVE++

R3LIVE (Lin et al., 2022)에서는 이러한 상보성을 살리기 위해 LiDAR-Inertial Odometry(LIO)와 Visual-Inertial Odometry(VIO) 두 서브시스템(dual-subsystem)을 tightly coupled 방식으로 묶었다. LIO가 3D 기하 구조(geometry)를 맡고 VIO가 텍스처(photometric) 정보를 맡되, 두 서브시스템이 각자 따로 상태를 추정하지 않고 하나의 공유 상태 벡터를 번갈아 갱신하는 구조다.

```
LiDAR scan ──→ [LIO 서브시스템] ──→ 상태 업데이트 (기하)
                     ↓                       ↓
                 IMU data ───────────→ 공유 상태 벡터
                     ↑                       ↑
Camera image ──→ [VIO 서브시스템] ──→ 상태 업데이트 (광도)
```

구체적으로 LIO 서브시스템은 FAST-LIO2의 골격을 따라 원시 LiDAR 점을 ikd-Tree 맵에 바로 point-to-plane 정합한 뒤 Iterated EKF로 공유 상태를 갱신한다. 뒤이어 VIO 서브시스템이 두 단계로 상태를 한 번 더 조인다. 먼저 맵 점을 optical flow로 추적해 frame-to-frame PnP reprojection error를 줄이고, 이어서 frame-to-map photometric error(광도 오차)를 최소화한다. 두 번째 단계가 성립하려면 3D 맵 점에 색상 정보가 있어야 하는데, R3LIVE는 LIO가 생성한 3D 맵 점마다 카메라 영상에서 뽑은 RGB 색상을 입혀 둔다. 새 영상이 들어오면 맵 점을 현재 카메라 시점에 투영했을 때 관측되는 픽셀 강도와 맵에 저장된 색상의 차이가 줄어들도록 포즈를 맞춘다.

$$
\mathbf{r}^{\text{photo}}_i = \mathbf{I}(\pi(\mathbf{T}_{CW} \mathbf{p}^W_i)) - \mathbf{c}_i^{\text{map}}
$$

여기서 $\mathbf{I}(\cdot)$는 영상의 픽셀 강도, $\pi(\cdot)$는 3D→2D 투영 함수, $\mathbf{T}_{CW}$는 월드 좌표를 카메라 좌표로 옮기는 변환, $\mathbf{p}^W_i$는 맵 점의 3D 좌표, $\mathbf{c}_i^{\text{map}}$는 맵에 저장해 둔 그 점의 색상이다.

두 서브시스템이 하나의 상태를 공유하는 덕분에 한쪽 센서 입력이 끊기더라도 다른 쪽이 상태 갱신을 이어 간다. 다만 LiDAR가 가려졌을 때 visual update가 얼마나 버티는지는 이미 색을 입혀 둔 맵 점이 현재 영상 시야에 얼마나 들어오는지에 달려 있고, 영상이 어두워졌을 때 LIO가 얼마나 버티는지는 LiDAR 기하 제약과 IMU 품질이 결정한다. 논문에서는 이 구조를 바탕으로 실시간 online colored 3D mapping을 구현해 보여 준다.

### 8.2.2 LVI-SAM

R3LIVE에서는 필터 기반의 공유 상태 하나에 두 서브시스템을 얹었다면, [LVI-SAM](https://arxiv.org/abs/2104.10831) (Shan et al., 2021)에서는 LIO-SAM을 확장해 Visual-Inertial 서브시스템(VIS)과 LiDAR-Inertial 서브시스템(LIS)이 서로 부족한 정보를 양방향(bidirectional)으로 건네주도록 설계했다.

양쪽의 결합은 다음과 같이 맞물린다.

- **VIS → LIS 방향**: Visual-Inertial 서브시스템이 추정한 포즈를 LiDAR 스캔 매칭의 초기값으로 넘긴다. 빠른 회전이나 기하 특징이 부족한 환경에서는 LiDAR 스캔 매칭만으로는 초기값이 빗나가 수렴하지 못하기 쉬운데, 시각 추정치가 이 출발점을 잡아 준다.
- **LIS → VIS 방향**: LiDAR 깊이를 시간과 위치가 일치하는 시각 특징점에 붙여 깊이 초기화를 돕는다. 외부 파라미터와 시간 동기화, 가시성, occlusion 검사를 모두 통과한 특징점에는 삼각측량에 필요한 시차가 쌓이기 전부터 metric depth prior를 곧바로 줄 수 있다.

```
         ┌─── VIS 초기 포즈 ───→ LIS 초기값
         │                            │
  [Visual-Inertial]            [LiDAR-Inertial]
         │                            │
         └←── LiDAR 깊이 ────────────┘

              ↓ LiDAR·IMU factor ↓
           [Factor Graph (GTSAM/iSAM2)]
                     ↓
              최종 최적화된 포즈
```

이 그림의 아래쪽에서 보듯 전역 최적화를 담당하는 factor graph는 LIO-SAM에서 물려받은 LIS가 유지한다. 그래프에는 연속 키프레임 사이의 IMU preintegration factor, 스캔 매칭에서 나온 LiDAR odometry factor, 그리고 재방문을 탐지했을 때 추가되는 loop closure factor가 들어간다. 흥미로운 점은 VIS의 결합 방식이다. 논문 본문에서는 visual odometry constraint도 이 그래프에 넣는다고 설명하지만, 공개 구현에서는 그 factor를 꺼 둔 채 VIS가 LIS에 초기 포즈를 건네고 LIS에서 LiDAR 깊이를 받아 자신의 스케일을 잡는 형태로만 연동한다.

### 8.2.3 FAST-LIVO / FAST-LIVO2

R3LIVE와 LVI-SAM 이후 등장한 [FAST-LIVO2](https://arxiv.org/abs/2408.14035) (Zheng et al., 2025)는 FAST-LIO2를 개발한 HKU MARS Lab이 내놓은 Camera+LiDAR+IMU 직접(direct) 융합 시스템이다. 여기서 "직접"이라는 말은 코너나 평면 같은 특징점을 따로 뽑지 않고 원시 LiDAR 점과 영상 픽셀 패치를 그대로 오차 항으로 쓴다는 뜻이다.

FAST-LIVO2가 이종 센서를 한 프레임워크에 밀어 넣기 위해 택한 첫 번째 설계는 **순차적 업데이트(Sequential Update)**다. LiDAR는 점마다 point-to-plane 잔차 하나를 내고, 카메라는 패치 픽셀마다 photometric 잔차 하나를 낸다. 두 측정 집합은 개수와 물리 단위, 잡음 규모가 크게 달라 한 번에 묶으면 차원 불균형(dimension mismatch)을 다루기 까다로워진다. 원 논문에서는 이 문제를 피하기 위해 한 주기를 세 단계로 쪼개 순차 갱신한다. 먼저 IMU 측정으로 상태를 예측한 뒤, LiDAR 측정으로 1차 업데이트를 하고, 그 결과를 사전 분포 삼아 카메라 측정으로 2차 업데이트를 한다.

상태 $\mathbf{x}$가 주어졌을 때 LiDAR 측정 $\mathbf{z}_L$과 카메라 측정 $\mathbf{z}_C$가 조건부 독립이라면, 베이즈 정리 관점에서 순차 갱신과 동시 갱신의 사후 분포는 이론적으로 같다.

$$
p(\mathbf{x} | \mathbf{z}_L, \mathbf{z}_C) = p(\mathbf{x} | \mathbf{z}_C, \mathbf{z}_L) \propto p(\mathbf{z}_C | \mathbf{x}) \cdot p(\mathbf{z}_L | \mathbf{x}) \cdot p(\mathbf{x})
$$

이를 두 단계로 나누어 적으면 1차 LiDAR 업데이트의 사후 분포가 2차 카메라 업데이트의 사전 분포(prior) 자리를 이어받는다.

$$
\underbrace{p(\mathbf{x} | \mathbf{z}_L)}_{\text{LiDAR 업데이트 후}} \propto p(\mathbf{z}_L | \mathbf{x}) \cdot p(\mathbf{x})
$$
$$
\underbrace{p(\mathbf{x} | \mathbf{z}_L, \mathbf{z}_C)}_{\text{카메라 업데이트 후}} \propto p(\mathbf{z}_C | \mathbf{x}) \cdot p(\mathbf{x} | \mathbf{z}_L)
$$

선형·가우시안 모델에서는 두 방식의 해가 정확히 일치하지만, 반복 선형화를 거치는 IEKF에서는 1차 업데이트 결과에 따라 2차 업데이트의 선형화 지점이 옮겨 가므로 동시 갱신과 근사적으로만 같아진다.

나머지 세 가지 설계는 이 순차 갱신이 빠르고 정확하게 돌도록 맵과 영상 처리가 맞물리게 하는 장치들이다.

- **통합 적응형 복셀 맵**: 해시 테이블과 옥트리를 결합한 단일 복셀 맵에 기하와 텍스처를 함께 담는다. LiDAR 모듈이 3D 좌표와 평면 법선 벡터를 구해 두면, Visual 모듈이 바로 그 맵 점에 이미지 패치를 붙여 관리한다.
- **LiDAR 법선 활용 어파인 워핑**: Direct method로 이미지 패치를 비교할 때는 시점이 바뀜에 따라 표면 기울기가 달라 보이므로 패치를 어파인 워핑(affine warping)해야 정합이 어긋나지 않는다. FAST-LIVO2는 영상에서 법선을 따로 추정하는 대신 LiDAR 맵에 이미 들어 있는 평면 법선을 그대로 가져와 워핑에 쓴다.
- **실시간 노출 보정**: 터널을 드나들 때처럼 조명이 급변하면 동일한 표면이라도 픽셀 밝기가 달라진다. FAST-LIVO2는 카메라의 노출 시간(exposure time)을 온라인 상태로 함께 추정해 photometric error에 반영함으로써 밝기 변화를 흡수한다.

아래 함수에서는 이 가운데 순차적 EKF 업데이트가 선형 관측 모델에서 어떻게 2단계로 나뉘어 계산되는지 볼 수 있다. `x_after_lidar`와 `P_after_lidar`가 곧바로 카메라 업데이트의 입력(`prior`)으로 넘어가는 지점을 눈여겨보면 앞선 베이즈 식과 코드가 정확히 포개진다.

```python
import numpy as np

def sequential_ekf_update(x_pred, P_pred, z_lidar, H_lidar, R_lidar, z_cam, H_cam, R_cam):
    """
    순차적 EKF 업데이트: LiDAR → Camera 순서.
    선형·가우시안 모델이고 두 측정이 상태에 대해 조건부 독립이면 동시 업데이트와 결과가 같다.
    이종 측정의 dimension mismatch를 한 번에 다루지 않아도 된다.

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

지금까지 살펴본 R3LIVE, LVI-SAM, FAST-LIVO2는 같은 세 센서를 다루면서도 백엔드 추정기(IEKF vs. factor graph)와 front-end 처리 방식(direct vs. feature-based), 맵 자료구조에서 뚜렷하게 갈린다. 세 시스템의 설계 선택을 나란히 견주어 보면 다음과 같다.

| 항목 | R3LIVE | LVI-SAM | FAST-LIVO2 |
|------|--------|---------|------------|
| 백엔드 | IEKF (dual subsystem) | iSAM2 (factor graph) | IEKF (sequential) |
| LiDAR 처리 | Direct (point-to-plane) | Feature-based (LOAM) | Direct (point-to-plane) |
| 카메라 처리 | PnP reprojection + photometric | Feature-based (Shi-Tomasi + KLT) | Direct (photometric) |
| 맵 표현 | ikd-Tree + RGB | 키프레임 포인트 클라우드 submap (LIO-SAM 방식) | Hash+Octree voxel map |
| 특징 추출 | 불필요 | 필요 (LiDAR edge/planar, 시각 코너) | 불필요 |
| GPS 통합 | 없음 | 없음 (LIO-SAM에서 물려받은 `addGPSFactor()`가 있지만 호출 줄이 주석 처리돼 있음) | 없음 |
| Loop closure | 없음 | Factor로 통합 | 없음 |
| 임베디드 검증 | 목표 하드웨어 벤치마크 필요 | 목표 하드웨어 벤치마크 필요 | ARM 구현 보고; 목표 설정에서 재검증 |

실제 프로젝트에서 적용 후보를 좁힐 때는 요구 기능과 환경 조건을 기준으로 출발점을 고른다. Loop closure 통합이 꼭 필요하다면 그래프 기반인 LVI-SAM을 먼저 검토하고, 센서 구성과 ROS 의존성, 목표 데이터에서의 재방문 성능을 함께 점검한다. 반대로 direct photometric update와 실시간 colored 3D mapping이 목적이라면 R3LIVE를 올려 목표 장면에서 색상·궤적·맵 품질 지표를 비교해 본다. ARM 기반 임베디드 보드 배포가 목표라면 ARM 구현 사례가 보고된 FAST-LIVO2를 후보로 두되 목표 센서 해상도와 설정에서 처리율, 최악 지연시간, 메모리 사용량을 직접 재측정해야 한다. 마지막으로 시각 코너나 LOAM 특징이 잘 잡히지 않는 단조로운 환경에서는 R3LIVE와 FAST-LIVO2 같은 direct 계열을 우선 검토하되, photometric calibration 요구사항과 동적 객체에 대한 민감도를 반드시 함께 검증해야 한다.

---

## 8.3 GNSS 통합

카메라와 LiDAR, IMU를 빈틈없이 엮더라도 오도메트리의 본질적 한계까지 사라지지는 않는다. 세 센서 모두 프레임 사이의 상대(relative) 운동을 이어 붙이므로 주행 거리가 길어질수록 드리프트가 필연적으로 누적된다. 이 오차의 사슬을 전역 좌표계의 기준점에 못 박아 두는 역할은 GNSS(Global Navigation Satellite System, 위성항법시스템)가 맡는다. 측량 기준점이나 UWB 비콘, 모션 캡처 역시 운용 환경에 따라 같은 절대 위치 기준점으로 기능한다.

### 8.3.1 GNSS Factor in Factor Graph

[LIO-SAM (Shan et al. 2020)](https://arxiv.org/abs/2007.00258)이 factor graph에 GPS 측정을 올리는 방식은 §7.3.1에서 미리 짚었다. 이를 일반화된 GNSS factor의 잔차로 적으면 다음과 같다.

$$
\mathbf{r}^{\text{GPS}}_i = \mathbf{T}_{\text{ENU→map}} \cdot \mathbf{p}^{\text{ENU}}_{\text{GPS}} - \mathbf{p}^{\text{map}}_i - \mathbf{R}^{\text{map}}_i \cdot \mathbf{l}_{\text{antenna}}
$$

여기서 $\mathbf{p}^{\text{ENU}}_{\text{GPS}}$는 GNSS가 보고한 ENU 좌표, $\mathbf{T}_{\text{ENU→map}}$은 ENU 좌표계에서 SLAM 맵 좌표계로 옮기는 변환, $\mathbf{p}^{\text{map}}_i$와 $\mathbf{R}^{\text{map}}_i$는 SLAM이 추정한 로봇의 위치와 회전이다. 여기에 $\mathbf{l}_{\text{antenna}}$가 들어가는 이유는 GNSS 안테나가 로봇의 body 중심이 아니라 지붕 등에 떨어져 장착되기 때문이며, 이 lever arm 오프셋을 회전 $\mathbf{R}^{\text{map}}_i$로 돌려 빼 주어야 안테나 위치와 로봇 기준점이 어긋나지 않는다.

이 식을 쓰려면 먼저 두 좌표계를 정렬해야 한다. SLAM의 로컬 맵 좌표계는 출발 지점을 기준으로 임의로 세워지는 반면, GNSS는 WGS84/ENU 전역 좌표계를 쓴다. 보통 첫 GNSS 수신 시점을 ENU 원점으로 삼고 초기 주행 구간의 포즈 짝으로 맵↔ENU 변환을 추정한다. 원칙적으로 이 변환은 6-DoF(이동 3 + 회전 3)지만, 맵의 수직 축이 이미 IMU의 중력 방향에 맞춰져 있다면 roll과 pitch는 맞출 필요가 없으므로 yaw와 3축 이동을 합친 4-DoF만 추정하면 된다.

### 8.3.2 Loosely vs Tightly Coupled GNSS

GNSS 역시 §8.1에서 나눈 결합 수준에 따라 INS와 합치는 방식이 갈린다.

**Loosely Coupled GNSS-INS**에서는 GNSS 수신기가 내부 연산을 거쳐 내보낸 최종 위치·속도 해(PVT solution)를 EKF 관측치로 받아 INS 상태를 보정한다. 아래 코드는 15차원 INS 상태에서 위치 성분을 뽑아 GNSS 위치와 합치는 기본 EKF 갱신이다. 관측 잡음 행렬 `R_gnss`의 대각 성분은 수평과 수직 정밀도 지표(HDOP, VDOP)에 사용자 등가 거리 오차($\sigma_{\text{UERE}}$)를 곱한 표준편차의 제곱으로 둔다.

```python
def gnss_loose_coupling_ekf_update(x_ins, P_ins, gnss_position, R_gnss):
    """
    Loosely coupled GNSS-INS: GNSS PVT 솔루션으로 INS 상태 보정.

    x_ins: INS 상태 [position(3), velocity(3), attitude(3), biases(6)] = 15차원
    gnss_position: GNSS가 계산한 위치 (3,)
    R_gnss: GNSS 위치의 공분산 (3, 3) — 표준편차의 제곱을 대각에 넣는다.
            수평은 sigma_h = HDOP * sigma_uere, 수직은 VDOP를 쓴다.
            즉 diag(sigma_E**2, sigma_N**2, sigma_U**2) 형태다.
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

반면 **Tightly Coupled GNSS-INS**는 수신기가 계산해 둔 PVT 해 대신 위성별 원시 의사거리(pseudorange)와 도플러(Doppler) 측정을 직접 끌어온다. 보이는 위성 하나마다 의사거리 관측 방정식을 factor로 세우는 방식이다.

$$
\rho_i = \| \mathbf{p}_{\text{sat},i} - \mathbf{p}_{\text{rx}} \| + c \cdot \delta t_{\text{rx}} + I_i + T_i + \epsilon_i
$$

여기서 $\rho_i$는 위성 $i$의 의사거리, $c \cdot \delta t_{\text{rx}}$는 수신기 시계 바이어스, $I_i$와 $T_i$는 각각 전리층과 대류권 지연이다.

이 차이는 고층 빌딩이 시야를 가리는 도심 구간에서 빛을 발한다. 3D 위치와 수신기 시계 바이어스까지 4개 미지수를 풀어야 하는 GNSS 단독 해는 위성 수가 4개 미만으로 떨어지는 순간 곧바로 끊긴다. 하지만 tightly coupled 구조에서는 위성이 2~3개만 잡히더라도 남아 있는 위성의 의사거리가 IMU·오도메트리 추정을 계속 붙잡아 줄 수 있다.

### 8.3.3 GNSS-Denied → GNSS-Available 전환 처리

실제 주행에서는 터널이나 지하주차장, 고가도로 아래를 지나며 GNSS 신호가 끊겼다가 야외로 나오면서 다시 살아나는 전환이 잦다. 이때 복구된 GNSS 측정을 곧바로 신뢰하면 맵이 찢어지는 사고가 나므로 세 가지 안전장치를 둔다.

1. **좌표계 점프(coordinate jump) 방지**: 음영 구간을 지나는 동안 쌓인 오도메트리 드리프트 때문에, 신호가 돌아온 직후의 GNSS 위치와 IMU/LiDAR 추정 위치 사이에는 큰 간극이 생겨 있기 마련이다. 이 차이를 단번에 끌어당기면 궤적과 맵이 한순간에 튀어 끊어진다. 따라서 복구 직후에는 GNSS 공분산을 의도적으로 크게 잡아 부드럽게 당기다가 점차 정상 값으로 낮춘다.
2. **GNSS 품질 검증**: 터널 출구처럼 신호가 막 잡히기 시작한 직후 몇 초는 주변 구조물 반사로 인한 multipath 오차가 심해 정확도가 떨어지기 쉽다. PDOP/HDOP 수치와 가시 위성 수, carrier phase 잠금 상태를 먼저 확인해 신뢰 기준을 넘긴 측정만 factor로 넣는다.
3. **맵 좌표계 보정**: 음영 구간이 길어 드리프트가 크게 누적된 상태라면 단순 필터 보정만으로는 맵의 뒤틀림이 남는다. 이때는 loop closure와 마찬가지로 포즈 그래프 전체를 다시 최적화해 구간 전체의 궤적과 맵을 함께 펴 준다.

---

## 8.4 Radar 융합

GNSS가 실외 전역 좌표를 책임진다면, 악천후 속에서 로컬 인식을 지탱하는 센서는 밀리미터파 레이더(mm-wave radar)다. 안개가 끼거나 폭우가 내리거나 눈발이 날리면 가시광 카메라는 시야를 잃고 LiDAR는 공기 중 입자 반사로 잡음이 폭증하지만, 파장이 긴 레이더 전파는 이런 악천후를 뚫고 거리를 잰다. 여기에 최근 고도각 정보까지 복원하는 4D imaging radar가 보급되면서 주행 보조를 넘어 오도메트리와 맵핑에도 레이더를 적극 엮기 시작했다.

### 8.4.1 Radar와 4D Imaging

기존 자동차용 레이더는 수평 방위각 해상도에 치우쳐 고도 방향 구분이 어려웠고, 그 때문에 높이가 다른 물체(예: 노면 위 가드레일과 공중의 표지판)가 한 시야 안에서 겹쳐 잡혔다([SNAIL Radar 논문](https://arxiv.org/abs/2407.11705) 1절).

**4D imaging radar**는 기존 레이더가 재던 거리(range), 속도(Doppler), 방위각(azimuth)에 고도각(elevation)을 더해 3D 포인트 클라우드를 출력한다. [Han et al. 서베이](https://arxiv.org/abs/2306.04242)가 정리한 대로 점 밀도는 저채널 LiDAR에 견줄 수준까지 올라왔지만, LiDAR에 비하면 여전히 성기고 잡음이 많다. 실무에서 4D radar를 검토할 때는 다음 세 가지 특성을 함께 봐야 한다.

1. **악천후 내성**: mm-wave radar는 가시광 카메라나 일부 LiDAR에 비해 안개·비·눈의 영향을 상대적으로 덜 받는다. 그렇다고 악천후에서 오차가 사라지는 것은 아니며, 강한 비에 따른 신호 감쇠(attenuation)와 물방울·젖은 노면이 만드는 multipath·clutter는 여전히 남으므로 운용 조건별로 검증해야 한다.
2. **직접 radial velocity 측정**: FMCW 레이더의 도플러 측정은 한 프레임 안의 여러 chirp에 걸친 위상 변화에서 시선 방향(line-of-sight) 상대 속도 성분을 직접 뽑아낸다. 다만 점 하나에서 얻는 값은 시선 방향 성분뿐이므로, 한 물체의 완전한 3D 속도 벡터를 복원하려면 여러 각도와 시간에 걸친 측정·추적을 모으거나 다른 센서 정보를 보태야 한다.
3. **다른 비용 구조**: 대량 생산용 일반 레이더 칩셋은 단가가 낮지만, 고해상도 4D imaging radar와 LiDAR는 채널 수, 안테나 배열, 신호 처리 장치, 생산 규모에 따라 모듈 가격대가 겹친다. 어느 쪽이 경제적인지는 현재 견적과 요구 성능을 놓고 직접 비교해야 한다.

### 8.4.2 Radar Odometry

4D radar 오도메트리는 각 점이 품고 있는 도플러 속도를 자차 운동(ego-motion) 추정에 곧바로 활용하는 데서 출발한다. FMCW 레이더가 내보내는 측정점은 거리 $r$, 방위각 $\theta$, 고도각 $\phi$, 그리고 도플러 속도 $v_d$를 담고 있다. 측정점이 정지한 물체에서 반사되었고 레이더 위상 중심의 선속도가 $\mathbf{v}_S$라면, 단위 방향 벡터 $\mathbf{d}_i = [\cos\phi_i \cos\theta_i, \cos\phi_i \sin\theta_i, \sin\phi_i]^T$를 따라 관측되는 시선 방향 도플러 속도는 이 방향 벡터와 선속도 벡터 내적의 부호를 뒤집은 값과 같다.

$$
v_{d,i} = -\mathbf{d}_i^T \mathbf{v}_S + n_i
$$

정적 반사점이 여러 방위각과 고도각에 걸쳐 충분히 퍼져 있다면 이 선형 방정식들을 모아 최소자승법으로 3축 선속도 $\mathbf{v}_S$를 바로 풀 수 있다. 다만 단일 monostatic radar의 점별 도플러 측정만으로는 센서 원점에서의 각속도를 독립적으로 관측할 수 없다. 구한 센서 속도를 로봇의 body 원점으로 옮기려면 이미 알고 있는 lever arm에 더해 IMU 자이로나 스캔 정합(scan registration)에서 얻은 각속도를 반드시 보태야 한다.

아래 코드는 각 점의 3D 좌표에서 단위 방향 벡터를 뽑아 선속도 $\mathbf{v}_S$를 푸는 기본 최소자승 구현이다. 실제 주행 데이터에는 움직이는 차량이나 보행자처럼 정적 가정을 깨는 점이 섞여 들어오므로, 실전 파이프라인에서는 이 `lstsq` 앞단에 RANSAC을 붙여 동적 물체를 걸러 낸다.

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

4D radar와 카메라를 짝지으면 고가의 LiDAR 없이도 깊이와 속도, 시각 의미를 아우르는 자율주행 센서 구성을 꾸릴 수 있다. 두 센서는 아래 표처럼 장단점이 서로 정반대다.

| 특성 | 카메라 | 4D Radar |
|------|--------|----------|
| 해상도 | 매우 높음 | 낮음 |
| 악천후 | 저하됨 | 상대적으로 강함 (강수 영향은 받음) |
| 직접 깊이 측정 | ✗ | ✓ |
| radial velocity 측정 | ✗ | ✓ |
| 시맨틱 이해 | 강함 | 약함 |
| 비용 | 렌즈·카메라 수·연산에 의존 | 안테나·채널·연산에 의존 |

두 센서를 엮는 방식은 정보가 만나는 지점에 따라 세 갈래로 나뉜다.
- **Early fusion**: Radar 점을 카메라 영상 평면에 투영해 희소한 깊이 단서로 쓰거나, 단안 깊이 추정 네트워크의 미터 단위 스케일 기준으로 삼는다.
- **Mid-level fusion**: 각 모달리티에서 뽑은 중간 특징 맵을 신경망 내부에서 합친다. 두 센서의 특징을 위에서 내려다본 BEV(Bird's Eye View, 조감도) 격자 공간으로 끌어와 결합하는 방식이 대표적이다.
- **Late fusion**: 카메라와 radar가 각자 독립적으로 물체 검출이나 트래킹을 마친 뒤, 최종 결과 목록 단계에서 매칭하고 합친다.

### 8.4.4 Boreas 벤치마크

다중 센서와 레이더 연구를 비교할 때 흔히 참고하는 [Boreas](https://arxiv.org/abs/2203.10168) (Burnett et al., 2023)는 맑은 날과 비·눈, 계절 변화를 아울러 반복 주행하며 수집한 공개 데이터셋이다. 카메라와 Velodyne Alpha Prime LiDAR, 그리고 Navtech CIR304-H 레이더를 동기화해 제공한다. 다만 여기에 실린 Navtech 장비는 고도각 정보를 주지 않는 평면(2D) 기계식 scanning radar이므로, Boreas 벤치마크의 레이더 성능을 고도각까지 재는 자동차용 4D imaging radar의 성능으로 곧바로 치환해 해석해서는 안 된다.

---

## 8.5 Multi-Robot / Decentralized Fusion

지금까지는 로봇 한 대에 여러 센서를 얹는 문제를 다루었다. 하지만 여러 대의 로봇이 같은 공간을 함께 탐색하며 지도를 합치려면 융합의 경계가 플랫폼 밖으로 넓어진다. 이때는 좁은 통신 대역폭, 서로 다른 출발 원점(공통 참조 프레임의 부재), 그리고 중앙 서버 없이 데이터 연관을 푸는 문제까지 함께 떠안아야 한다.

### 8.5.1 Multi-Robot SLAM의 과제

다중 로봇 SLAM이 단일 로봇 SLAM과 갈라지는 지점은 크게 세 가지다.

1. **상대 포즈 추정(Inter-Robot Relative Pose)**: 로봇마다 각자의 로컬 좌표계에서 출발해 지도를 그리기 때문에 처음에는 서로의 좌표를 맞출 기준이 없다. 두 지도를 하나로 합치려면 로봇 사이의 장소 인식(cross-robot place recognition)으로 공통 지점을 찾고 기하 검증을 거쳐 두 좌표계 사이의 상대 변환부터 구해야 한다.
2. **통신 제약(Communication Constraint)**: 무선 통신 대역폭이 좁은 현장에서는 전체 지도나 원시 센서 데이터를 그대로 주고받기 어렵다. 장소 인식용 압축 디스크립터만 먼저 보낼지, 검증된 핵심 키프레임만 교환할지 설계 단계에서 전송 단위를 정해 두어야 한다.
3. **분산 최적화(Distributed Optimization)**: 중앙 서버에 모든 데이터를 끌어모아 최적화하면 통신 병목이 생기고 서버 하나가 죽었을 때 전체가 멈추는 단일 실패점(single point of failure)을 안게 된다. 반대로 각 로봇이 자기 그래프를 유지한 채 이웃과 일부 정보만 주고받는 분산 방식은 이 병목을 피할 수 있지만, 수렴 속도가 느려지고 동기화·상태 관리 구현이 한층 까다로워진다.

### 8.5.2 Kimera-Multi

[Kimera-Multi](https://doi.org/10.1109/TRO.2021.3137751) (Tian et al., T-RO 2022; [arXiv 2106.14386](https://arxiv.org/abs/2106.14386))는 MIT SPARK Lab이 제안한 분산 멀티로봇 SLAM 시스템으로, 이 세 과제를 분산 포즈 그래프 최적화와 이상치에 강건한 솔버로 풀어낸다.

각 로봇은 내부적으로 Kimera를 돌려 로컬 metric-semantic SLAM을 수행하다가, 다른 로봇과 통신 반경 안에서 만나면(rendezvous) DBoW2 기반 inter-robot place recognition으로 공통 방문 지점을 찾는다. 검증된 inter-robot loop closure는 각 로봇의 분산 포즈 그래프에 추가되며, 잘못된 루프가 전체 지도를 뒤틀지 않도록 **GNC(Graduated Non-Convexity)** 솔버로 이상치 loop closure를 걸러 낸다.

분산 최적화 과정은 통신 성격이 다른 두 단계로 갈린다. 로봇들은 먼저 루프가 검출된 시점에 inter-robot factor를 한 번 공유한 뒤, Riemannian block-coordinate descent 같은 분산 최적화 루프에 들어가 반복마다 이웃 그래프와 맞닿은 경계(separator) 포즈의 현재 추정치를 주고받으며 해에 수렴한다. 제약을 한 번 등록하는 통신과 최적화 변수를 반복 교환하는 통신을 구분해 전체 맵 전송을 피하는 셈이다.

### 8.5.3 Swarm-SLAM

[Swarm-SLAM](https://arxiv.org/abs/2301.06230) (Lajoie et al., 2024)은 대규모 로봇 군집(swarm)을 겨냥해 통신 효율을 한층 끌어올린 분산 SLAM 프레임워크다.

대역폭을 아끼기 위해 로봇들은 처음부터 포인트 클라우드나 이미지를 주고받지 않고 NetVLAD나 Scan Context 같은 가벼운 **장소 인식 디스크립터만 먼저 교환**한다. 디스크립터 매칭으로 실제로 같은 장소일 가능성이 큰 후보를 추린 뒤에야, 기하 검증에 꼭 필요한 특징점이나 포인트 클라우드 조각만 선택적으로 주고받는다. Swarm-SLAM은 중앙 서버를 거치지 않고 인접 로봇끼리 직접 통신하는 피어투피어(P2P) 구조를 따르며, 카메라만 단 로봇과 LiDAR만 단 로봇, 혹은 두 센서를 함께 단 로봇이 한 군집 안에서 섞여 참여할 수 있도록 설계됐다.

아래 코드에서는 이처럼 각 로봇이 로컬 포즈 그래프를 유지한 채 이웃과 경계 변수만 주고받는 분산 포즈 그래프 노드의 개념 구조를 확인할 수 있다. `exchange_boundary_info` 메서드는 로컬 맵 전체가 아니라 `inter_robot_factors`에 걸려 있는 경계 포즈(`local_idx`)만 골라 전달한다. 이 동작을 보면 분산 최적화의 통신 단위가 분명해진다.

```python
# 분산 포즈 그래프 최적화의 개념적 구현
import numpy as np

class DistributedPoseGraphNode:
    """
    분산 포즈 그래프의 단일 로봇 노드.
    각 로봇은 자신의 로컬 그래프를 유지하고,
    이웃 로봇과 inter-robot factor와 경계 포즈 추정치를 교환한다.
    """
    def __init__(self, robot_id):
        self.robot_id = robot_id
        self.local_poses = [np.eye(4)]  # 자체 포즈 (로컬 좌표계). 원점 초기 포즈를 넣어 둔다
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

## 8.6 센서 선정·시간 동기화·고장 대응 실전 설계

알고리즘과 결합 구조를 정했더라도 실제 하드웨어 위에 올려 안정적으로 굴리려면 센서 선정, 시간 동기화, 고장 대응(degradation handling)이라는 실전 과제가 남는다.

### 8.6.1 Sensor Suite 선정 가이드

센서 구성을 잡을 때는 아래 표를 출발점으로 삼되, 최종 구성은 운용 환경과 안전 요구사항, 유효 거리, 전력·질량 예산, 그리고 예상 실패 시나리오를 따져 결정한다.

| 환경 | 권장 최소 구성 | 선택 추가 센서 |
|------|---------------|---------------|
| 실내 (사무실/창고) | Camera + IMU | LiDAR (정밀 매핑 시) |
| 도심 자율주행 | Camera + LiDAR + IMU + GNSS | 4D Radar, Wheel Odom |
| 비포장/야외 | LiDAR + IMU + GNSS | Camera (시맨틱), Radar |
| 지하/터널 | LiDAR + IMU | Camera, UWB |
| 수중 | IMU + DVL (Doppler Velocity Log) | Sonar, Pressure |
| 항공/드론 | Camera + IMU + GNSS | LiDAR (매핑 시) |
| 악천후 (비/눈) | Radar + IMU | Camera, LiDAR |

비용을 따질 때는 센서 본체 가격표만 보지 않는 것이 중요하다. 센서 단가는 시점과 지역, 구매 수량에 따라 크게 달라진다. 따라서 먼저 필요한 측정량과 정확도·출력 주기(rate)·시야각(FoV)을 확정한 뒤, 현재 견적으로 총비용을 산출해야 한다. 총비용을 낼 때는 센서 본체에 케이블·동기화 장치·GNSS 보정 서비스·연산 보드·마운트·예비 부품·통합 공수를 더해야 한다. 저비용 프로토타입이라면 Camera+IMU에서 출발하고, 검증 과정에서 드러난 실패 모드와 오차 예산에 맞춰 2D/3D LiDAR, RTK GNSS, radar, 그리고 이중화(redundancy) 센서를 얹어 나간다.

### 8.6.2 시간 동기화 설계 (Timing Architecture)

멀티센서 시스템에서는 시간 동기화 오차가 곧바로 위치 오차로 번진다. 100km/h로 달리는 차량에서 1ms의 시간 오차는 약 2.8cm의 위치 오차에 해당한다(27.8 m/s × 0.001 s). 센서가 아무리 정밀해도 타임스탬프가 몇 밀리초만 어긋나면 고속 구간이나 급회전 구간에서 융합 필터의 잔차가 튄다.

가장 깔끔한 해법은 **하드웨어 동기화(Hardware Sync)**다. 공유 클록이나 물리적 트리거 선을 연결할 수 있으면 소프트웨어 수신 시각에 기댈 때보다 지연 경로가 투명해진다. 다만 펄스 신호의 정확도와 실제 데이터가 찍히는 측정 타임스탬프의 정확도는 다를 수 있으므로 반드시 end-to-end 지연까지 검증해야 한다.
- **PPS (Pulse Per Second)**: GNSS 수신기가 내보내는 초 경계 펄스를 여러 장치 클록의 공통 기준으로 삼는다. 이때 수신기 time-pulse 사양과 케이블·입력 지연, 타임스탬프 기준점이 맞는지 확인한다.
- **PTP (Precision Time Protocol, IEEE 1588)**: 이더넷 네트워크상의 장치 클록을 동기화한다. 실제 동기화 오차는 하드웨어 타임스탬프 지원 여부, 네트워크 스위치, PTP 프로파일, 송수신 경로 비대칭과 lock 상태에 좌우된다.
- **외부 트리거**: 마이크로컨트롤러가 카메라 셔터 트리거와 IMU 타임스탬프 캡처를 동시에 발생시켜 두 센서의 기준 시각을 맞춘다.

하드웨어 동기화를 구성할 수 없다면 **소프트웨어 동기화(Software Sync)**로 시간 오프셋 $\tau$를 추정해 보정한다. 대표적으로 **Kalibr 방식**에서는 궤적을 B-spline 연속 시간 곡선으로 표현한 뒤 센서 간 시간 오프셋을 캘리브레이션 최적화 변수에 넣어 함께 추정한다(→ 3장).

간단히 오프셋만 빠르게 가늠할 때는 두 센서가 같은 운동을 관측한 시계열의 **상호상관(cross-correlation)**을 이용한다. 이때 선속도나 가속도를 쓰려면 IMU 측정값에서 중력을 빼고 좌표계를 맞춰야 하며, 단안 카메라에서는 속도의 스케일조차 알 수 없다. 반면 각속도의 크기(노름)를 비교하면 중력 제거와 스케일 문제를 피할 수 있을 뿐 아니라, 노름이 회전에 불변이라 카메라-IMU 회전 외부 파라미터로 한쪽 벡터를 돌려 맞출 필요도 없어진다.

$$
\hat{\tau} = \arg\max_{\tau} \int \left\| \boldsymbol{\omega}_{\text{IMU}}(t) \right\| \, \left\| \boldsymbol{\omega}_{\text{camera}}(t + \tau) \right\| \, dt
$$

아래 함수는 이 아이디어를 이산 시계열로 옮긴 구현이다. 두 센서의 타임스탬프를 1kHz 공통 타임라인에 보간해 맞춘 뒤, 허용 범위(`max_offset_ms`) 안에서 상호상관이 최대가 되는 지연 시간 $\hat{\tau}$를 찾아낸다.

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

센서는 장시간 운용 환경에서 언제든 일시적 혹은 영구적으로 고장 난다. 다중 센서 융합의 진가는 평상시 정확도뿐 아니라, 한 센서가 무너졌을 때 시스템 전체가 멈추지 않고 성능이 조금 낮아진 상태로 버티는 **점진적 성능 저하(graceful degradation)**에 달려 있다. 아래 표에 센서별로 자주 마주치는 실패 모드와 탐지·대응 전략을 정리했다.

| 실패 모드 | 증상 | 탐지 방법 | 대응 |
|-----------|------|-----------|------|
| 카메라 과노출/저노출 | 이미지 전체가 밝거나 어두움 | 히스토그램 분석 | 카메라 factor 비활성화, LIO만으로 동작 |
| LiDAR 기하 퇴화 (degenerate) | 긴 복도, 넓은 평지 | 정보 행렬의 고유값 분석 | 해당 DoF의 LiDAR 제약 완화, VIO로 보완 |
| IMU 포화 | 고속 충격 시 측정 범위 초과 | ADC 최댓값 탐지 | 해당 시간대 IMU preintegration 불확실성 증가 |
| GNSS multipath | 건물 반사로 인한 큰 오차 | RAIM, 잔차 검사 | 해당 GNSS factor의 공분산 증가 또는 제거 |
| 센서 완전 단절 | 데이터 수신 없음 | Watchdog timer | 새 factor 추가 중단. 실패 시각이 불확실하면 의심 구간 이후만 가중치 하향 또는 제거 |

이 가운데 **LiDAR 기하 퇴화(degeneracy)**는 터널이나 긴 복도에서 자주 나타나는 전형적 실패 모드다. 이를 정량적으로 잡아내려면 LiDAR 스캔 매칭의 정보 행렬(근사 Hessian) $\mathbf{H} = \mathbf{J}^T \mathbf{J}$를 고유값 분해해 본다. 특정 방향의 고유값이 다른 방향에 비해 현저히 작다면, 그 고유벡터 방향으로는 스캔 정합이 위치를 붙잡아 주지 못한다는 뜻이다.

$$
\mathbf{H} = \mathbf{U} \boldsymbol{\Lambda} \mathbf{U}^T, \quad \lambda_{\min} / \lambda_{\max} < \epsilon \Rightarrow \text{degenerate}
$$

예를 들어 긴 복도에서는 복도 진행 축 방향의 고유값이 급감하므로, 그 방향의 LiDAR 제약만 약하게 풀고 카메라의 optical flow 등 다른 센서 제약으로 빈틈을 메운다. 아래 두 함수는 Hessian의 고유값 비율로 퇴화 방향을 찾아내고, 최소 고유값 크기에 따라 LiDAR와 카메라의 융합 가중치를 조절하는 기본 로직을 보여 준다.

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

### 8.6.4 최근 시스템과 연구 (2025)

최근에는 전통적 희소 특징이나 복셀 맵을 넘어 고품질 장면 복원 표현을 융합 상태에 직접 통합하는 시도도 이어지고 있다.

- **[Gaussian-LIC (Lang et al., ICRA 2025)](https://arxiv.org/abs/2404.06926)**: 3D Gaussian Splatting을 LiDAR-Inertial-Camera tightly-coupled SLAM에 통합한 시스템이다. LiDAR의 정밀한 기하 정보와 카메라의 텍스처를 3D Gaussian 표현으로 함께 묶어, 실시간 자기 위치 추정과 photo-realistic 장면 복원을 동시에 수행한다.

### 8.6.5 시스템 설계 체크리스트

실제 멀티센서 융합 시스템을 설계하고 배포하기 전에는 다음 항목들을 순서대로 점검한다.

**캘리브레이션**:
- [ ] 모든 센서 쌍의 extrinsic calibration 완료
- [ ] 시간 동기화 오프셋 측정/추정 완료
- [ ] 캘리브레이션 결과의 재현성 검증 (3회 이상 반복)
- [ ] 캘리브레이션 변화 감시와 대응 절차 구비 (온라인 추정 또는 주기적 재교정). 온라인 extrinsic 추정은 관측 가능성에 충분한 운동 여기가 필요하므로 모든 시스템의 필수 조건은 아니다.

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

다중 센서 융합으로 국소 오도메트리의 정밀도와 강건성을 끌어올리더라도, GNSS 같은 절대 위치 기준이 닿지 않는 실내나 도심 음영 구간을 오래 달리다 보면 미세한 오차가 결국 궤적 드리프트로 쌓인다. 이 누적 오차를 전역적으로 바로잡으려면 로봇 스스로 예전에 지나온 장소를 다시 알아보는 능력이 필요하다. 재방문 후보를 찾아내는 장소 인식(Place Recognition, → 9장)과, 검증된 재방문 제약을 지도에 반영해 오차를 펴는 루프 클로저(Loop Closure, → 10장)가 다음 과제다.

