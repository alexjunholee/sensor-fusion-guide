# Ch.10 — Loop Closure & Global Optimization

Ch.9에서 Place Recognition — 과거 방문 장소의 인식 — 을 다루었다. 이 챕터에서는 그 인식 결과를 SLAM 시스템에 통합하여 누적 드리프트를 실제로 교정하는 과정을 다룬다.

SLAM 시스템에서 odometry는 필연적으로 드리프트(drift)를 누적한다. 아무리 정밀한 센서를 쓰더라도, 상대적 pose 추정의 작은 오차들이 시간에 따라 쌓여 전역적 일관성을 무너뜨린다. Loop closure는 "과거에 방문했던 장소를 재방문했음"을 인식하고, 그 정보를 활용하여 누적된 드리프트를 한꺼번에 보정하는 메커니즘이다.

이 챕터에서는 loop closure의 전체 파이프라인(detection → verification → correction)을 살펴보고, 보정의 핵심인 pose graph optimization의 수학적 기초, 그리고 global relocalization과 multi-session SLAM까지 확장한다.

---

## 10.1 Loop Closure Pipeline

Loop closure는 세 단계로 구성된다: **Detection**(후보 탐지), **Verification**(기하학적 검증), **Correction**(그래프 보정). 각 단계는 서로 다른 역할을 하며, 어느 하나라도 실패하면 전체 시스템의 일관성이 깨질 수 있다.

### 10.1.1 Detection: 후보 탐지

Loop closure detection은 "현재 센서 관측이 과거의 어떤 관측과 유사한가?"를 묻는 문제다. 이것은 본질적으로 Ch.9에서 다룬 place recognition 문제와 동일하다.

**Visual loop closure detection**에서는 현재 이미지의 글로벌 디스크립터를 과거 키프레임들의 디스크립터 데이터베이스와 비교한다:

1. **BoW 기반 (전통)**: DBoW2를 사용하여 ORB 특징점의 visual word histogram을 비교한다. ORB-SLAM3는 이 방식을 사용한다. 각 키프레임을 bag-of-words 벡터 $\mathbf{v}_i$로 표현하고, 현재 프레임 $\mathbf{v}_q$와의 유사도를 $s(\mathbf{v}_q, \mathbf{v}_i) = 1 - \frac{1}{2} \left| \frac{\mathbf{v}_q}{\|\mathbf{v}_q\|} - \frac{\mathbf{v}_i}{\|\mathbf{v}_i\|} \right|$ (L1-score)로 계산한다.

2. **학습 기반 (현대)**: [NetVLAD](https://arxiv.org/abs/1511.07247), [AnyLoc](https://arxiv.org/abs/2308.00688) 등의 글로벌 디스크립터를 사용한다. AnyLoc은 [DINOv2](https://arxiv.org/abs/2304.07193)의 dense feature를 VLAD로 집계하여 환경 특화 학습 없이도 범용적으로 동작한다. 코사인 유사도로 후보를 랭킹한다:

$$s(\mathbf{d}_q, \mathbf{d}_i) = \frac{\mathbf{d}_q^\top \mathbf{d}_i}{\|\mathbf{d}_q\| \|\mathbf{d}_i\|}$$

**LiDAR loop closure detection**에서는 3D 포인트 클라우드 기반 디스크립터를 사용한다:

- **[Scan Context](https://doi.org/10.1109/IROS.2018.8593953)**: 센서 중심의 극좌표계에서 bin/sector별 최대 높이를 기록하여 공간 구조를 직접 보존하는 디스크립터다. ring key와 sector key를 이용한 2단계 검색으로 효율적 후보 탐색이 가능하며, 역방향 재방문에도 강건하다.
- **[PointNetVLAD](https://arxiv.org/abs/1804.03492), [OverlapTransformer](https://arxiv.org/abs/2203.03397)**: 학습 기반 3D place recognition으로, 대규모 환경에서 Scan Context보다 높은 recall을 보인다.

**시간적 필터링**: 최근 프레임과의 매칭은 loop closure가 아니라 단순히 연속 tracking이다. 따라서 시간적으로 충분히 떨어진 키프레임(예: 최소 30초 이상 경과)만 후보로 고려한다.

```python
import numpy as np

def detect_loop_candidates(query_descriptor, database_descriptors, 
                           timestamps, current_time, 
                           min_time_gap=30.0, top_k=5, threshold=0.7):
    """
    Loop closure 후보 탐지.
    
    Args:
        query_descriptor: 현재 프레임의 글로벌 디스크립터 (D,)
        database_descriptors: 과거 키프레임 디스크립터들 (N, D)
        timestamps: 각 키프레임의 타임스탬프 (N,)
        current_time: 현재 시간
        min_time_gap: 최소 시간 간격 (초)
        top_k: 반환할 후보 수
        threshold: 유사도 임계값
        
    Returns:
        candidates: [(index, similarity)] 리스트
    """
    # 시간적 필터링: 최근 프레임 제외
    time_mask = (current_time - timestamps) > min_time_gap
    
    if not np.any(time_mask):
        return []
    
    # 코사인 유사도 계산
    query_norm = query_descriptor / (np.linalg.norm(query_descriptor) + 1e-8)
    db_norms = database_descriptors / (
        np.linalg.norm(database_descriptors, axis=1, keepdims=True) + 1e-8
    )
    
    similarities = db_norms @ query_norm  # (N,)
    
    # 시간 필터 적용
    similarities[~time_mask] = -1.0
    
    # 상위 k개 후보 선택
    top_indices = np.argsort(similarities)[::-1][:top_k]
    candidates = [
        (idx, similarities[idx]) 
        for idx in top_indices 
        if similarities[idx] > threshold
    ]
    
    return candidates
```

### 10.1.2 Verification: 기하학적 검증

Detection 단계에서 찾은 후보는 appearance similarity만으로 선별되었기 때문에, **false positive**(실제로는 다른 장소인데 비슷하게 보이는 경우)가 포함될 수 있다. Perceptual aliasing — 시각적으로 유사하지만 실제로 다른 장소 — 은 특히 실내 환경(비슷한 복도, 반복적 구조)에서 빈번하다.

False positive loop closure는 **치명적**이다. 잘못된 loop closure 하나가 전체 맵을 뒤틀어버릴 수 있다. 따라서 verification은 보수적으로 수행해야 한다 — recall을 조금 희생하더라도 precision을 극대화한다.

**기하학적 검증 방법**:

1. **2D-2D: Essential matrix 검증**: 현재 프레임과 후보 키프레임 사이에서 특징점 매칭을 수행하고, RANSAC으로 essential matrix $\mathbf{E}$를 추정한다. 인라이어 수가 충분하고(예: ≥ 20), 인라이어 비율이 높으면(예: ≥ 50%) 유효한 loop closure로 판단한다.

$$\mathbf{p}_2^\top \mathbf{E} \mathbf{p}_1 = 0, \quad \mathbf{E} = [\mathbf{t}]_\times \mathbf{R}$$

2. **3D-3D: Point cloud registration**: LiDAR 기반 시스템에서는 ICP나 GeoTransformer로 두 스캔 간의 상대 변환 $\mathbf{T}_{ij} \in SE(3)$를 추정한다. Fitness score(정합된 포인트 비율)와 RMSE로 검증한다.

3. **2D-3D: PnP**: 현재 2D 특징점과 후보 키프레임의 3D 맵 포인트 사이의 PnP 문제를 풀어 상대 pose를 추정한다.

4. **Temporal consistency**: 단일 매칭이 아니라, 연속된 여러 프레임에서 동일 장소와의 매칭이 일관되게 나타나는지 확인한다. ORB-SLAM3는 세 번 연속 동일 장소가 검출되어야 loop closure를 수용한다.

```python
import numpy as np

def verify_loop_closure(kp_current, kp_candidate, matches, K,
                        min_inliers=20, min_inlier_ratio=0.5):
    """
    Essential matrix 기반 loop closure 기하학적 검증.
    
    Args:
        kp_current: 현재 프레임 키포인트 좌표 (N, 2)
        kp_candidate: 후보 프레임 키포인트 좌표 (M, 2)
        matches: 매칭 인덱스 쌍 리스트 [(i, j), ...]
        K: 카메라 내부 파라미터 행렬 (3, 3)
        min_inliers: 최소 인라이어 수
        min_inlier_ratio: 최소 인라이어 비율
        
    Returns:
        is_valid: 유효한 loop closure인지 여부
        T_relative: 상대 변환 (4, 4) 또는 None
    """
    if len(matches) < min_inliers:
        return False, None
    
    pts1 = np.array([kp_current[m[0]] for m in matches], dtype=np.float64)
    pts2 = np.array([kp_candidate[m[1]] for m in matches], dtype=np.float64)
    
    # 정규화 좌표로 변환
    K_inv = np.linalg.inv(K)
    pts1_norm = (K_inv @ np.hstack([pts1, np.ones((len(pts1), 1))]).T).T[:, :2]
    pts2_norm = (K_inv @ np.hstack([pts2, np.ones((len(pts2), 1))]).T).T[:, :2]
    
    # RANSAC으로 Essential matrix 추정
    E, inlier_mask = estimate_essential_ransac(pts1_norm, pts2_norm, 
                                                threshold=1e-3, max_iter=1000)
    
    num_inliers = np.sum(inlier_mask)
    inlier_ratio = num_inliers / len(matches)
    
    if num_inliers < min_inliers or inlier_ratio < min_inlier_ratio:
        return False, None
    
    # E에서 R, t 복원
    R, t = decompose_essential(E, pts1_norm[inlier_mask], pts2_norm[inlier_mask])
    
    T_relative = np.eye(4)
    T_relative[:3, :3] = R
    T_relative[:3, 3] = t.flatten()
    
    return True, T_relative


def estimate_essential_ransac(pts1, pts2, threshold=1e-3, max_iter=1000):
    """5-point 알고리즘 + RANSAC으로 Essential matrix 추정."""
    best_E = None
    best_inliers = np.zeros(len(pts1), dtype=bool)
    
    for _ in range(max_iter):
        # 8개 점 랜덤 샘플링
        idx = np.random.choice(len(pts1), 8, replace=False)
        
        # 8-point 알고리즘으로 E 후보 생성 (5-point의 간략화 버전)
        E_candidate = eight_point_essential(pts1[idx], pts2[idx])
        
        if E_candidate is None:
            continue
        
        # Sampson error로 인라이어 판별
        errors = sampson_error(E_candidate, pts1, pts2)
        inliers = errors < threshold
        
        if np.sum(inliers) > np.sum(best_inliers):
            best_inliers = inliers
            best_E = E_candidate
    
    return best_E, best_inliers


def sampson_error(E, pts1, pts2):
    """Sampson error 계산 — epipolar constraint의 1차 근사 거리."""
    # pts를 homogeneous로 변환
    p1 = np.hstack([pts1, np.ones((len(pts1), 1))])  # (N, 3)
    p2 = np.hstack([pts2, np.ones((len(pts2), 1))])  # (N, 3)
    
    Ep1 = (E @ p1.T).T    # (N, 3)
    Etp2 = (E.T @ p2.T).T  # (N, 3)
    
    # p2^T E p1
    numerator = np.sum(p2 * Ep1, axis=1) ** 2
    denominator = Ep1[:, 0]**2 + Ep1[:, 1]**2 + Etp2[:, 0]**2 + Etp2[:, 1]**2
    
    return numerator / (denominator + 1e-10)
```

### 10.1.3 False Positive의 위험과 방지

False positive loop closure가 왜 그토록 위험한가를 구체적으로 이해해보자.

**시나리오**: 로봇이 두 개의 비슷하게 생긴 복도를 지나간다고 하자. 복도 A의 키프레임 $i$와 복도 B의 키프레임 $j$ 사이에 잘못된 loop closure가 발생하면, pose graph optimizer는 이 두 pose를 가깝게 끌어당긴다. 그 결과 두 복도 사이의 모든 pose가 왜곡되어, 맵 전체가 접히거나 뒤틀린다.

**방지 전략**:

1. **다단계 검증**: appearance similarity → geometric consistency → temporal consistency를 순차적으로 통과해야 한다.

2. **Robust kernel 사용** (§10.2에서 상세): optimizer 자체가 이상치 제약에 덜 민감하게 만든다.

3. **[Switchable constraints (Sünderhauf & Protzel, 2012)](https://doi.org/10.1109/IROS.2012.6385590)**: 각 loop closure factor에 on/off 스위치 변수를 추가하여, optimizer가 일관성이 떨어지는 loop closure를 자동으로 비활성화한다.

4. **[DCS (Dynamic Covariance Scaling) (Agarwal et al., 2013)](https://doi.org/10.1109/ICRA.2013.6630733)**: loop closure의 공분산(uncertainty)을 동적으로 조절하여, 이상치의 영향을 자동 감쇠시킨다.

5. **[PCM (Pairwise Consistency Maximization) (Mangelson et al., 2018)](https://doi.org/10.1109/ICRA.2018.8460217)**: 여러 loop closure 후보들 간의 pairwise consistency를 검사하여, 일관된 최대 집합만 수용한다.

### 10.1.4 Correction: 그래프 보정

검증을 통과한 loop closure는 pose graph에 새로운 제약으로 추가된다. Loop closure edge는 두 pose 사이의 상대 변환 $\mathbf{T}_{ij}$와 그 불확실성 $\boldsymbol{\Sigma}_{ij}$를 인코딩한다:

$$e_{ij} = \text{Log}(\mathbf{T}_{ij}^{-1} \cdot \mathbf{T}_i^{-1} \cdot \mathbf{T}_j)$$

이 edge가 추가되면 pose graph optimizer(§10.2)가 전체 그래프를 재최적화하여 드리프트를 보정한다. 이 과정에서 loop closure edge 뿐 아니라 odometry edge들도 함께 조정되어, 오차가 경로 전체에 균등하게 분배된다.

---

## 10.2 Pose Graph Optimization

Pose graph optimization은 SLAM 백엔드의 핵심이다. 프론트엔드(odometry, loop closure)가 생성한 상대적 제약들을 만족시키면서 전체 pose trajectory의 전역적 일관성을 최적화한다.

### 10.2.1 SE(3) Pose Graph

Pose graph는 그래프 $\mathcal{G} = (\mathcal{V}, \mathcal{E})$로 표현된다:

- **노드** $\mathcal{V} = \{\mathbf{T}_1, \mathbf{T}_2, \ldots, \mathbf{T}_n\}$: 각 키프레임의 pose, $\mathbf{T}_i \in SE(3)$.
- **에지** $\mathcal{E}$: 두 노드 사이의 상대적 제약. odometry edge와 loop closure edge로 나뉜다.

각 에지 $(i, j) \in \mathcal{E}$는 측정된 상대 변환 $\tilde{\mathbf{T}}_{ij}$와 정보 행렬(information matrix) $\boldsymbol{\Omega}_{ij}$를 갖는다.

**SE(3)에서의 오차 정의**: pose graph에서의 오차는 유클리드 공간이 아니라 Lie group SE(3) 위에서 정의된다:

$$\mathbf{e}_{ij} = \text{Log}(\tilde{\mathbf{T}}_{ij}^{-1} \cdot \mathbf{T}_i^{-1} \cdot \mathbf{T}_j) \in \mathbb{R}^6$$

여기서 $\text{Log}: SE(3) \to \mathfrak{se}(3) \cong \mathbb{R}^6$은 행렬 로그(matrix logarithm)를 통해 Lie algebra로의 매핑이다. 이 6차원 벡터는 $($ 회전 3 + 병진 3 $)$의 오차를 인코딩한다.

**최적화 목표**: 모든 에지 오차의 가중 제곱합을 최소화한다:

$$\mathbf{T}^* = \arg\min_{\mathbf{T}_1, \ldots, \mathbf{T}_n} \sum_{(i,j) \in \mathcal{E}} \mathbf{e}_{ij}^\top \boldsymbol{\Omega}_{ij} \mathbf{e}_{ij}$$

이것은 nonlinear least squares 문제이며, Gauss-Newton 또는 Levenberg-Marquardt로 풀 수 있다.

**Gauss-Newton on manifold**: SE(3)는 유클리드 공간이 아니므로, 직접 더하기(+) 연산을 사용할 수 없다. 대신 Lie algebra에서의 증분(increment) $\boldsymbol{\delta}$를 계산하고, exponential map으로 manifold 위에 적용한다:

$$\mathbf{T}_i \leftarrow \mathbf{T}_i \cdot \text{Exp}(\boldsymbol{\delta}_i)$$

한 반복(iteration)에서의 업데이트:

1. 현재 추정치에서 각 에지의 오차 $\mathbf{e}_{ij}$와 Jacobian $\mathbf{J}_{ij}$를 계산한다.
2. Normal equation을 구성한다: $\mathbf{H} \boldsymbol{\delta} = -\mathbf{b}$, 여기서

$$\mathbf{H} = \sum_{(i,j)} \mathbf{J}_{ij}^\top \boldsymbol{\Omega}_{ij} \mathbf{J}_{ij}, \quad \mathbf{b} = \sum_{(i,j)} \mathbf{J}_{ij}^\top \boldsymbol{\Omega}_{ij} \mathbf{e}_{ij}$$

3. $\mathbf{H}$는 sparse하므로 sparse Cholesky 분해로 효율적으로 풀 수 있다.
4. 증분을 적용한다: $\mathbf{T}_i \leftarrow \mathbf{T}_i \cdot \text{Exp}(\boldsymbol{\delta}_i)$.
5. 수렴할 때까지 반복한다.

**[g2o (General Graph Optimization)](https://doi.org/10.1109/ICRA.2011.5979949)**: Kümmerle et al. (2011)이 개발한 프레임워크로, 위 과정을 일반적인 그래프 최적화 문제에 대해 구현한다. 사용자는 vertex(노드) 타입과 edge(제약) 타입만 정의하면, sparse 최적화 엔진이 자동으로 작동한다.

```python
import numpy as np
from scipy.spatial.transform import Rotation

def se3_log(T):
    """SE(3) 행렬 → 6차원 벡터 (회전 + 병진)."""
    R = T[:3, :3]
    t = T[:3, 3]
    
    # SO(3) logarithm
    rot = Rotation.from_matrix(R)
    omega = rot.as_rotvec()  # (3,)
    
    theta = np.linalg.norm(omega)
    
    if theta < 1e-10:
        V_inv = np.eye(3)
    else:
        omega_hat = skew(omega)
        V_inv = (np.eye(3) 
                 - 0.5 * omega_hat 
                 + (1.0/theta**2) * (1 - theta/(2*np.tan(theta/2))) * omega_hat @ omega_hat)
    
    rho = V_inv @ t
    return np.concatenate([rho, omega])  # (6,) — [translation; rotation]


def se3_exp(xi):
    """6차원 벡터 → SE(3) 행렬."""
    rho = xi[:3]   # 병진 부분
    omega = xi[3:]  # 회전 부분
    
    theta = np.linalg.norm(omega)
    
    if theta < 1e-10:
        R = np.eye(3)
        V = np.eye(3)
    else:
        omega_hat = skew(omega)
        R = (np.eye(3) 
             + (np.sin(theta)/theta) * omega_hat 
             + ((1 - np.cos(theta))/theta**2) * omega_hat @ omega_hat)
        V = (np.eye(3) 
             + ((1 - np.cos(theta))/theta**2) * omega_hat 
             + ((theta - np.sin(theta))/theta**3) * omega_hat @ omega_hat)
    
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = V @ rho
    return T


def skew(v):
    """3차원 벡터 → 반대칭 행렬."""
    return np.array([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0]
    ])


def pose_graph_error(T_i, T_j, T_ij_measured):
    """두 pose 사이의 오차 벡터 (6차원)."""
    T_ij_estimated = np.linalg.inv(T_i) @ T_j
    T_error = np.linalg.inv(T_ij_measured) @ T_ij_estimated
    return se3_log(T_error)


def pose_graph_cost(poses, edges, measurements, information_matrices):
    """
    전체 pose graph의 비용 함수.
    
    Args:
        poses: [T_0, T_1, ..., T_n] SE(3) 행렬 리스트
        edges: [(i, j), ...] 에지 인덱스 리스트
        measurements: [T_ij, ...] 측정된 상대 변환 리스트
        information_matrices: [Omega_ij, ...] 정보 행렬 리스트
        
    Returns:
        total_cost: 스칼라 비용
    """
    total_cost = 0.0
    
    for (i, j), T_ij, Omega_ij in zip(edges, measurements, information_matrices):
        e = pose_graph_error(poses[i], poses[j], T_ij)
        total_cost += e @ Omega_ij @ e
    
    return total_cost
```

### 10.2.2 Robust Kernel

실제 SLAM 시스템에서는 이상치(outlier) 측정이 불가피하다. 잘못된 loop closure, 센서 오류, 동적 객체 등이 원인이 된다. 표준 least squares 비용 함수 $\rho(x) = x^2$는 이상치에 극도로 민감하다 — 큰 오차가 비용을 지배하여 전체 해를 왜곡시킨다.

**Robust kernel** (M-estimator)은 큰 잔차의 영향을 제한하여 이상치에 강건한 최적화를 가능하게 한다:

| Kernel | $\rho(s)$ ($s = e^2$) | 특성 |
|--------|----------------------|------|
| Least Squares | $s$ | 이상치에 취약 |
| Huber | $\begin{cases} s & \text{if } \sqrt{s} \leq k \\ 2k\sqrt{s} - k^2 & \text{otherwise} \end{cases}$ | 임계값 $k$ 이상에서 선형 |
| Cauchy | $k^2 \log(1 + s/k^2)$ | 부드러운 감쇠 |
| Geman-McClure | $\frac{s}{k^2 + s}$ | 강한 이상치 억제 |

Robust kernel을 적용하면 비용 함수가 다음과 같이 변경된다:

$$\sum_{(i,j)} \rho\left(\mathbf{e}_{ij}^\top \boldsymbol{\Omega}_{ij} \mathbf{e}_{ij}\right)$$

**IRLS (Iteratively Reweighted Least Squares)** 방식으로 풀 수 있다: 각 반복에서 잔차 크기에 따라 가중치를 재계산하고, 가중 least squares를 풀면 된다.

$$w_i = \rho'(s_i), \quad s_i = \mathbf{e}_i^\top \boldsymbol{\Omega}_i \mathbf{e}_i$$

여기서 $\rho'$은 위 테이블에 정의된 $\rho(s)$의 $s$에 대한 도함수이다. 이상치 에지에는 작은 가중치가 부여되어 그 영향이 자동으로 감소한다.

**[Switchable constraints](https://doi.org/10.1109/IROS.2012.6385590)**: Sünderhauf & Protzel (2012)은 각 loop closure factor에 이진 스위치 변수 $s_{ij} \in [0, 1]$을 도입하여, optimizer가 일관성이 없는 loop closure를 비활성화($s_{ij} \to 0$)할 수 있게 했다:

$$\rho_{\text{switch}}(\mathbf{e}_{ij}, s_{ij}) = s_{ij}^2 \mathbf{e}_{ij}^\top \boldsymbol{\Omega}_{ij} \mathbf{e}_{ij} + \lambda (1 - s_{ij})^2$$

첫째 항은 스위치가 켜져 있을 때 오차를 최소화하고, 둘째 항은 스위치를 끄는 것에 대한 페널티다. Optimizer가 이 두 가지를 자동으로 균형 잡는다.

**[DCS (Dynamic Covariance Scaling)](https://doi.org/10.1109/ICRA.2013.6630733)** (Agarwal et al., 2013): 각 loop closure의 공분산을 잔차 크기에 따라 동적으로 스케일링한다. 이상치의 공분산이 자동으로 커져서(= 불확실성 증가) 그 영향이 줄어든다.

```python
def huber_kernel(s, k=1.345):
    """Huber robust kernel."""
    sqrt_s = np.sqrt(s)
    if sqrt_s <= k:
        return s
    else:
        return 2 * k * sqrt_s - k**2

def cauchy_kernel(s, k=2.3849):
    """Cauchy robust kernel."""
    return k**2 * np.log(1 + s / k**2)

def huber_weight(s, k=1.345):
    """Huber kernel에 대한 IRLS 가중치."""
    sqrt_s = np.sqrt(s)
    if sqrt_s <= k:
        return 1.0
    else:
        return k / sqrt_s

def robust_pose_graph_cost(poses, edges, measurements, info_matrices, 
                           kernel='huber', k=1.345):
    """Robust kernel이 적용된 pose graph 비용."""
    total_cost = 0.0
    
    kernel_fn = {'huber': huber_kernel, 'cauchy': cauchy_kernel}[kernel]
    
    for (i, j), T_ij, Omega in zip(edges, measurements, info_matrices):
        e = pose_graph_error(poses[i], poses[j], T_ij)
        s = e @ Omega @ e  # Mahalanobis distance squared
        total_cost += kernel_fn(s, k)
    
    return total_cost
```

### 10.2.3 iSAM2: Incremental Smoothing and Mapping

대부분의 SLAM 시스템은 새 키프레임이 추가될 때마다 전체 그래프를 재최적화하는 batch 방식이 아니라, 영향받는 부분만 선택적으로 업데이트하는 incremental 방식을 사용한다. [iSAM2 (Kaess et al., 2012)](https://doi.org/10.1177/0278364911430419)는 이 incremental 최적화의 핵심 알고리즘이다.

**핵심 아이디어**: 새 변수나 측정이 추가되었을 때, 그 영향이 파급되는 범위를 정확히 파악하고, 그 부분만 재계산한다.

**Bayes tree**: iSAM2의 핵심 자료구조다. Factor graph를 variable elimination하면 clique tree가 되는데, Bayes tree는 이 clique tree에 방향성을 부여한 것이다.

Factor graph에서의 MAP 추정은 다음과 같이 분해된다:

$$p(\mathbf{x} | \mathbf{z}) \propto \prod_k f_k(\mathbf{x}_k)$$

여기서 $f_k$는 각 factor(odometry, loop closure 등)이고, $\mathbf{x}_k$는 해당 factor에 관련된 변수들이다.

Variable elimination을 수행하면:

$$p(\mathbf{x} | \mathbf{z}) = \prod_i p(x_i | \text{Sep}(x_i))$$

여기서 $\text{Sep}(x_i)$는 clique tree에서 $x_i$의 separator — 즉, 이 변수를 제거할 때 남는 조건부 변수들이다. 이 조건부 분포 구조가 Bayes tree를 형성한다.

**Incremental update 과정**:

1. 새 factor가 추가되면, 영향받는 clique만 식별한다 (대부분 소수).
2. 해당 clique와 그 조상(ancestor)만 QR 분해를 재수행한다.
3. 나머지 tree는 그대로 유지한다.

**Fluid relinearization**: 비선형 최적화에서 선형화 지점이 현재 추정치와 크게 달라진 변수만 재선형화한다. 이를 통해 iSAM의 v1에서 필요했던 주기적 batch 재선형화를 완전히 제거했다.

**변수 재정렬(variable reordering)**: 새 변수 추가 시 전체 elimination order를 재계산하지 않고, 영향받는 부분만 incremental하게 재정렬한다.

**실전 임팩트**: iSAM2는 GTSAM 라이브러리의 핵심 엔진이며, LIO-SAM, ORB-SLAM3 등 거의 모든 현대 SLAM 시스템의 백엔드에서 사용된다. 대규모 환경에서도 일정 시간 내에 최적화를 완료할 수 있어 실시간 SLAM을 가능하게 한다.

```python
class SimpleIncrementalOptimizer:
    """
    iSAM2의 핵심 개념을 보여주는 간소화된 incremental optimizer.
    실제 iSAM2는 Bayes tree 기반이지만, 여기서는 직관적 이해를 위해
    affected-variable tracking 개념만 구현한다.
    """
    
    def __init__(self):
        self.poses = {}          # {id: SE(3) matrix}
        self.edges = []          # [(i, j, T_ij, Omega_ij)]
        self.affected = set()    # 재최적화 필요한 변수 ID
        
    def add_pose(self, pose_id, T_init):
        """새 pose 노드 추가."""
        self.poses[pose_id] = T_init.copy()
        self.affected.add(pose_id)
        
    def add_edge(self, i, j, T_ij, Omega_ij):
        """새 에지(제약) 추가. 연결된 변수를 affected로 표시."""
        self.edges.append((i, j, T_ij, Omega_ij))
        self.affected.add(i)
        self.affected.add(j)
        # 실제 iSAM2에서는 Bayes tree를 타고 올라가며
        # ancestor clique들도 affected로 표시한다.
        
    def add_loop_closure(self, i, j, T_ij, Omega_ij):
        """
        Loop closure 에지 추가.
        일반 에지와 동일하지만, loop closure는 먼 과거의 노드와
        현재 노드를 연결하므로 더 많은 변수가 affected된다.
        """
        self.add_edge(i, j, T_ij, Omega_ij)
        # Loop closure는 경로상의 모든 중간 pose에 영향을 미친다.
        # 실제 iSAM2에서는 Bayes tree 구조상
        # root에 가까운 clique까지 영향이 전파된다.
        for k in range(min(i, j) + 1, max(i, j)):
            if k in self.poses:
                self.affected.add(k)
    
    def optimize(self, max_iter=5):
        """
        Affected 변수들만 재최적화.
        실제 iSAM2에서는 Bayes tree의 부분적 재분해로 수행.
        """
        if not self.affected:
            return
            
        # 여기서는 간략화를 위해 전체 Gauss-Newton을 수행하되,
        # affected 변수만 업데이트한다.
        for iteration in range(max_iter):
            # 1. 관련 에지만 선별
            relevant_edges = [
                (i, j, T_ij, Omega) 
                for (i, j, T_ij, Omega) in self.edges
                if i in self.affected or j in self.affected
            ]
            
            # 2. 각 에지에 대해 오차와 Jacobian 계산
            # 3. Normal equation 구성 및 풀기
            # 4. Affected 변수에만 increment 적용
            
            # (실제 구현은 sparse linear system을 풀어야 하므로 생략)
            pass
        
        self.affected.clear()
```

---

## 10.3 Global Relocalization

Global relocalization은 로봇이 사전에 구축된 맵(prior map) 위에서 자신의 위치를 찾는 문제다. Loop closure가 "이전에 내가 방문했던 곳"을 인식하는 것이라면, relocalization은 "다른 사람이 만든 맵에서 나는 어디인가"를 묻는 것이다.

### 10.3.1 Map-Based Localization

이미 구축된 맵이 있을 때, 새로운 센서 관측을 이 맵에 정합하여 현재 pose를 추정한다.

**Visual relocalization 파이프라인**:

1. 현재 이미지에서 특징점을 추출한다.
2. 맵의 3D 포인트들과의 2D-3D 대응을 찾는다 (visual word 기반 또는 직접 매칭).
3. PnP + RANSAC으로 pose를 추정한다.
4. 추정된 pose를 시작점으로 tracking을 재개한다.

ORB-SLAM3의 relocalization은 이 과정을 다음과 같이 수행한다:
- DBoW2로 후보 키프레임을 검색한다.
- ORB 매칭으로 2D-3D 대응을 확보한다.
- EPnP + RANSAC으로 pose를 추정한다.
- Guided search로 추가 매칭을 찾아 정확도를 높인다.

**LiDAR relocalization**: 현재 LiDAR 스캔을 사전 맵(point cloud map)에 정합한다.

1. **Global descriptor** (Scan Context, PointNetVLAD 등)로 맵 내 근접 영역을 탐색한다.
2. **Coarse registration**: FPFH + RANSAC 또는 GeoTransformer로 초기 정합을 수행한다.
3. **Fine registration**: ICP/GICP로 정밀 정합한다.

### 10.3.2 Prior Map + Online Sensor

자율주행에서는 사전에 구축한 HD map(고정밀 지도) 위에서 실시간 센서 데이터로 localization하는 것이 일반적이다. 이때의 핵심 과제는:

- **맵과 현재 환경의 불일치**: 시간이 지나면 건물이 바뀌고, 차량이 주차되어 있고, 나무잎이 자란다. Prior map과 현재 관측 사이의 차이를 처리해야 한다.
- **Cross-modal matching**: HD map이 LiDAR로 만들어졌는데 현재 센서는 카메라뿐일 수 있다. 이종 센서 간 정합이 필요하다.
- **Initial pose 없음**: 로봇이 맵의 어디서 시작하는지 모르는 경우, 전체 맵에 대해 place recognition을 수행해야 한다.

### 10.3.3 Monte Carlo Localization (MCL)

MCL은 particle filter 기반의 global localization 방법이다. 로봇의 가능한 pose를 particle들로 표현하고, 센서 관측에 따라 particle 가중치를 갱신한다.

**알고리즘**:

1. **초기화**: 맵 전체에 particle을 균일하게 분포시킨다 (global uncertainty).
2. **Prediction**: 로봇 모션 모델에 따라 particle들을 이동시킨다:
$$x_t^{[k]} \sim p(x_t | u_t, x_{t-1}^{[k]})$$
3. **Update**: 현재 센서 관측과 맵에서의 예상 관측을 비교하여 각 particle의 가중치를 계산한다:
$$w_t^{[k]} = p(z_t | x_t^{[k]}, m)$$
4. **Resampling**: 가중치에 비례하여 particle을 재샘플링한다. 가중치가 높은 particle(실제 위치에 가까운)은 복제되고, 낮은 particle은 제거된다.

MCL은 multi-modal 분포를 표현할 수 있다는 것이 핵심 장점이다. 로봇이 여러 장소 중 어디에 있을지 모를 때, 여러 가설을 동시에 유지할 수 있다. 관측이 누적됨에 따라 particle들이 점차 올바른 위치로 수렴한다.

**LiDAR 기반 MCL 예시**:

```python
import numpy as np

class MonteCarloLocalization:
    """
    2D LiDAR 기반 Monte Carlo Localization.
    사전 맵: occupancy grid.
    """
    
    def __init__(self, occupancy_map, num_particles=1000, 
                 map_resolution=0.05):
        """
        Args:
            occupancy_map: 2D numpy array, 0=free, 1=occupied
            num_particles: particle 수
            map_resolution: 맵 해상도 (m/pixel)
        """
        self.map = occupancy_map
        self.resolution = map_resolution
        self.num_particles = num_particles
        
        # Particle 초기화: [x, y, theta, weight]
        self.particles = self._initialize_particles()
    
    def _initialize_particles(self):
        """맵의 free space에 균일하게 particle을 분포."""
        free_cells = np.argwhere(self.map == 0)
        
        if len(free_cells) == 0:
            raise ValueError("맵에 free space가 없습니다.")
        
        # 랜덤 free cell 선택
        indices = np.random.choice(len(free_cells), self.num_particles, 
                                   replace=True)
        selected = free_cells[indices]
        
        particles = np.zeros((self.num_particles, 4))
        particles[:, 0] = selected[:, 1] * self.resolution  # x
        particles[:, 1] = selected[:, 0] * self.resolution  # y
        particles[:, 2] = np.random.uniform(-np.pi, np.pi, 
                                             self.num_particles)  # theta
        particles[:, 3] = 1.0 / self.num_particles  # weight (균일)
        
        return particles
    
    def predict(self, delta_x, delta_y, delta_theta, 
                noise_std=[0.1, 0.1, 0.05]):
        """모션 모델로 particle 이동 + 노이즈."""
        noise = np.random.randn(self.num_particles, 3) * noise_std
        
        cos_theta = np.cos(self.particles[:, 2])
        sin_theta = np.sin(self.particles[:, 2])
        
        # 로봇 좌표계에서의 이동을 전역 좌표계로 변환
        self.particles[:, 0] += (delta_x * cos_theta 
                                  - delta_y * sin_theta + noise[:, 0])
        self.particles[:, 1] += (delta_x * sin_theta 
                                  + delta_y * cos_theta + noise[:, 1])
        self.particles[:, 2] += delta_theta + noise[:, 2]
        
        # 각도 정규화
        self.particles[:, 2] = np.arctan2(
            np.sin(self.particles[:, 2]), 
            np.cos(self.particles[:, 2])
        )
    
    def update(self, scan_ranges, scan_angles, sigma_hit=0.2):
        """
        센서 관측으로 particle 가중치 갱신.
        
        Args:
            scan_ranges: 실제 LiDAR 거리 측정 (N,)
            scan_angles: 각 빔의 각도 (N,)
            sigma_hit: 센서 노이즈 표준편차
        """
        for k in range(self.num_particles):
            x, y, theta = self.particles[k, :3]
            
            log_weight = 0.0
            for r_measured, angle in zip(scan_ranges, scan_angles):
                # 이 particle 위치에서의 예상 거리 계산 (ray casting)
                r_expected = self._ray_cast(x, y, theta + angle)
                
                # 가우시안 센서 모델
                diff = r_measured - r_expected
                log_weight += -0.5 * (diff / sigma_hit) ** 2
            
            self.particles[k, 3] = np.exp(log_weight)
        
        # 가중치 정규화
        total = np.sum(self.particles[:, 3])
        if total > 0:
            self.particles[:, 3] /= total
        else:
            self.particles[:, 3] = 1.0 / self.num_particles
    
    def resample(self):
        """Low-variance resampling."""
        weights = self.particles[:, 3]
        N = self.num_particles
        
        new_particles = np.zeros_like(self.particles)
        r = np.random.uniform(0, 1.0 / N)
        c = weights[0]
        i = 0
        
        for m in range(N):
            u = r + m / N
            while u > c:
                i += 1
                c += weights[i]
            new_particles[m] = self.particles[i].copy()
        
        new_particles[:, 3] = 1.0 / N
        self.particles = new_particles
    
    def get_estimate(self):
        """가중 평균으로 현재 pose 추정."""
        weights = self.particles[:, 3]
        x_est = np.average(self.particles[:, 0], weights=weights)
        y_est = np.average(self.particles[:, 1], weights=weights)
        
        # 각도의 가중 평균 (circular mean)
        sin_avg = np.average(np.sin(self.particles[:, 2]), weights=weights)
        cos_avg = np.average(np.cos(self.particles[:, 2]), weights=weights)
        theta_est = np.arctan2(sin_avg, cos_avg)
        
        return x_est, y_est, theta_est
    
    def _ray_cast(self, x, y, angle, max_range=30.0):
        """Bresenham 기반 간단한 ray casting."""
        dx = np.cos(angle) * self.resolution
        dy = np.sin(angle) * self.resolution
        
        cx, cy = x, y
        for step in range(int(max_range / self.resolution)):
            cx += dx
            cy += dy
            
            # 맵 좌표로 변환
            mx = int(cy / self.resolution)
            my = int(cx / self.resolution)
            
            if (mx < 0 or mx >= self.map.shape[0] or 
                my < 0 or my >= self.map.shape[1]):
                return max_range
            
            if self.map[mx, my] == 1:  # 장애물 히트
                return step * self.resolution
        
        return max_range
```

---

## 10.4 Multi-Session SLAM & Map Merging

실제 환경에서 SLAM은 한 번에 완료되지 않는 경우가 많다. 같은 건물을 여러 날에 걸쳐 매핑하거나, 여러 로봇이 동시에 다른 영역을 탐사하거나, tracking이 실패한 후 다른 지점에서 재시작하는 경우가 있다. Multi-session SLAM은 이러한 개별 세션의 맵들을 하나의 일관된 전역 맵으로 통합한다.

### 10.4.1 Map Anchoring

여러 세션의 맵을 통합하려면, 각 맵의 좌표계(coordinate frame)를 정렬해야 한다. 이를 **map anchoring**이라 한다.

**방법 1: 공유 랜드마크 기반**: 두 세션이 겹치는 영역에서 같은 3D 포인트나 특징점을 관측했다면, 이를 기반으로 두 맵 사이의 상대 변환 ${}^{A}\mathbf{T}_{B}$를 추정한다.

$${}^{A}\mathbf{T}_{B} = \arg\min_{\mathbf{T}} \sum_k \| \mathbf{p}_k^A - \mathbf{T} \cdot \mathbf{p}_k^B \|^2$$

**방법 2: Place recognition 기반**: 공유 랜드마크가 명시적으로 없더라도, place recognition으로 두 세션에서 같은 장소를 방문한 키프레임 쌍을 찾고, 이를 기반으로 정렬한다.

**방법 3: GNSS 기반**: 두 세션 모두 GNSS 정보가 있다면, 이를 공유 좌표계로 사용하여 직접 정렬한다.

### 10.4.2 Inter-Session Loop Closure

Map anchoring이 초기 정렬을 제공하면, inter-session loop closure가 정밀한 정합을 수행한다. 이것은 일반 loop closure와 원리가 같지만, 두 가지 추가적 도전이 있다:

1. **외관 변화**: 시간이 지나면 조명, 계절, 가구 배치 등이 변한다. AnyLoc 같은 foundation model 기반 디스크립터가 이 문제에 강건하다.

2. **좌표계 불일치**: 초기 정렬이 부정확할 수 있으므로, geometric verification의 tolerance를 높여야 한다.

### 10.4.3 ORB-SLAM3 Multi-Map System

ORB-SLAM3의 Atlas 시스템은 multi-session SLAM의 대표적 구현이다. 핵심 메커니즘은 다음과 같다:

1. **Active map**: 현재 tracking 중인 맵. 정상적으로 동작할 때는 이 맵에서 키프레임과 맵 포인트를 추가한다.

2. **Map creation**: tracking이 실패하면(예: 시야가 가려지거나, textureless 환경), 새로운 맵을 생성하고 이를 active map으로 설정한다. 이전 맵은 Atlas에 비활성 상태로 보관된다.

3. **Map merging**: place recognition으로 현재 active map과 Atlas의 비활성 맵 사이에서 공통 장소가 검출되면, 두 맵을 병합한다:
   - 공통 키프레임 쌍에서 상대 변환 $\mathbf{T}_{merge}$를 추정한다.
   - 비활성 맵의 모든 키프레임과 맵 포인트를 $\mathbf{T}_{merge}$로 변환한다.
   - 공통 맵 포인트를 fusion한다.
   - Welding bundle adjustment로 병합 영역의 일관성을 확보한다.

4. **Multi-session 활용**: 이전 세션의 맵을 로드하고, 현재 세션에서 같은 장소를 방문하면 자동으로 병합된다. 이를 통해 여러 세션에 걸쳐 점진적으로 맵을 확장할 수 있다.

```python
class MultiMapAtlas:
    """
    ORB-SLAM3 Atlas 시스템의 핵심 개념.
    """
    
    def __init__(self):
        self.maps = {}           # {map_id: MapData}
        self.active_map_id = None
        self.next_map_id = 0
    
    def create_new_map(self, initial_pose):
        """새 맵을 생성하고 active로 설정."""
        map_id = self.next_map_id
        self.next_map_id += 1
        
        self.maps[map_id] = {
            'keyframes': [initial_pose],
            'map_points': [],
            'is_active': True,
            'origin': initial_pose.copy()
        }
        
        # 이전 active map 비활성화
        if self.active_map_id is not None:
            self.maps[self.active_map_id]['is_active'] = False
        
        self.active_map_id = map_id
        return map_id
    
    def on_tracking_lost(self, last_pose):
        """
        Tracking 실패 시: 현재 맵을 보관하고 새 맵 생성.
        """
        print(f"Tracking lost. Map {self.active_map_id} "
              f"saved with {len(self.maps[self.active_map_id]['keyframes'])} "
              f"keyframes.")
        self.create_new_map(last_pose)
    
    def try_merge_maps(self, current_kf_descriptor, current_kf_pose):
        """
        현재 키프레임의 디스크립터로 비활성 맵에서 매칭을 탐색.
        매칭이 발견되면 두 맵을 병합.
        """
        for map_id, map_data in self.maps.items():
            if map_id == self.active_map_id:
                continue
            
            # Place recognition으로 비활성 맵의 키프레임과 매칭 탐색
            match_idx = self._search_in_map(current_kf_descriptor, map_id)
            
            if match_idx is not None:
                # 기하학적 검증
                T_merge = self._compute_merge_transform(
                    current_kf_pose, 
                    map_data['keyframes'][match_idx]
                )
                
                if T_merge is not None:
                    self._merge_maps(self.active_map_id, map_id, T_merge)
                    return True
        
        return False
    
    def _merge_maps(self, active_id, merge_id, T_merge):
        """
        merge_id 맵을 active_id 맵으로 병합.
        merge_id의 모든 키프레임/맵포인트를 T_merge로 변환.
        """
        merge_map = self.maps[merge_id]
        active_map = self.maps[active_id]
        
        # 비활성 맵의 키프레임을 변환하여 active 맵에 추가
        for kf in merge_map['keyframes']:
            transformed_kf = T_merge @ kf
            active_map['keyframes'].append(transformed_kf)
        
        # 맵 포인트도 변환하여 추가
        for mp in merge_map['map_points']:
            transformed_mp = T_merge @ mp
            active_map['map_points'].append(transformed_mp)
        
        # 병합된 맵 제거
        del self.maps[merge_id]
        
        print(f"Maps {active_id} and {merge_id} merged. "
              f"Total keyframes: {len(active_map['keyframes'])}")
    
    def _search_in_map(self, descriptor, map_id):
        """비활성 맵에서 place recognition 수행 (placeholder)."""
        return None  # 실제 구현 시 디스크립터 비교
    
    def _compute_merge_transform(self, pose_a, pose_b):
        """두 pose 사이의 변환 계산 (placeholder)."""
        return None  # 실제 구현 시 기하학적 검증 포함
```

### 10.4.4 Multi-Robot Map Merging

여러 로봇이 동시에 탐사하는 경우, 각 로봇의 맵을 실시간으로 통합해야 한다. 이때 추가적 제약은:

- **통신 대역폭**: 전체 포인트 클라우드를 전송할 수 없으므로, 압축된 디스크립터(Scan Context, compact visual descriptor)만 교환한다.
- **분산 최적화**: 중앙 서버 없이도 로봇들이 자율적으로 맵을 병합할 수 있어야 한다. Kimera-Multi, Swarm-SLAM이 이 문제를 다룬다.
- **Relative pose 불확실성**: 로봇 간 초기 상대 pose가 알려져 있지 않으므로, inter-robot loop closure로 정렬해야 한다.

분산 pose graph optimization의 핵심 아이디어:

$$\mathbf{T}^* = \arg\min \sum_{\text{robot } r} \sum_{(i,j) \in \mathcal{E}_r} \rho(\mathbf{e}_{ij}) + \sum_{(i,j) \in \mathcal{E}_{\text{inter}}} \rho(\mathbf{e}_{ij})$$

각 로봇은 자신의 에지 $\mathcal{E}_r$에 대한 최적화를 로컬에서 수행하고, inter-robot 에지 $\mathcal{E}_{\text{inter}}$에 대해서만 정보를 교환한다. ADMM (Alternating Direction Method of Multipliers)이나 Gauss-Seidel iteration으로 분산적으로 수렴할 수 있다.

---

## 10.5 최근 연구 (2024-2025)

**[riSAM (McGann et al., 2023)](https://arxiv.org/abs/2209.14359)**: iSAM2에 **Graduated Non-Convexity(GNC)**를 통합하여, incremental SLAM에서 온라인으로 이상치 loop closure를 제거하는 robust backend이다. 90% 이상의 outlier 측정에서도 강건하게 동작하며, 기존 offline 방법과 동등한 성능을 실시간으로 달성한다. [GNC의 이론적 토대는 Yang et al. (2020)](https://arxiv.org/abs/1909.08605)이 제시하였다.

**[Kimera2 (Abate et al., 2024)](https://arxiv.org/abs/2401.06323)**: Kimera SLAM 라이브러리의 차세대 버전으로, 백엔드의 outlier rejection을 기존 PCM에서 GNC로 교체하여 강건성을 크게 향상시켰다. 드론, 사족보행 로봇, 자율주행 차량 등 다양한 플랫폼에서 실증되었으며, metric-semantic SLAM의 실전 배포를 위한 종합적인 개선을 포함한다.

**[Group-k Consistent Measurement Set Maximization (Forsgren & Kaess, 2022)](https://arxiv.org/abs/2209.02658)**: PCM의 pairwise consistency를 group-k consistency로 확장하여, 더 엄격한 이상치 탐지를 가능하게 한다. Multi-robot map merging에서 PCM 대비 false positive를 추가로 억제한다.

---

Loop closure와 전역 최적화를 통해 SLAM 시스템은 전역적으로 일관된 궤적과 맵을 생성한다. 그런데 이 "맵"은 어떤 형태를 가지는가? 점군인가, 그리드인가, 아니면 신경망인가? 다음 챕터에서는 센서 퓨전의 최종 결과물인 **공간 표현(spatial representation)**의 다양한 형태와 그 장단점을 살펴본다.
