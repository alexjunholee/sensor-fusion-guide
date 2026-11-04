# Ch.10 — Loop Closure & Global Optimization

Ch.9의 Place Recognition은 로봇이 전에 지나간 장소를 알아본다. 그 결과를 SLAM 시스템에 넣어야 쌓인 드리프트가 실제로 줄어든다.

Odometry는 상대 포즈를 이어 붙여 추정하므로, 센서가 아무리 정밀해도 작은 오차가 시간에 따라 쌓인다. 쌓인 오차 때문에 같은 장소가 지도 위 서로 다른 곳에 그려진다. Loop closure는 로봇이 전에 지나간 곳에 다시 왔음을 알아내고, 두 시점 사이의 기하 제약을 더해 이 드리프트를 줄인다.

---

## 10.1 Loop Closure Pipeline

Loop closure는 Detection(후보 탐지), Verification(기하학적 검증), Correction(그래프 보정) 세 단계를 차례로 거친다. 보정의 중심은 pose graph optimization이다. 후보를 놓치면 드리프트를 줄일 기회를 잃는다. 잘못된 루프를 받아들이면 지도 전체가 뒤틀린다.

### 10.1.1 Detection: 후보 탐지

Loop closure detection은 "현재 관측이 과거의 어떤 관측과 비슷한가?"를 묻는다. Ch.9의 place recognition과 같은 문제다.

**Visual loop closure detection**은 현재 이미지의 디스크립터를 과거 키프레임 디스크립터 데이터베이스와 비교한다:

1. **BoW(Bag of Words) 기반**: DBoW2로 ORB 특징점의 visual word histogram을 비교한다. ORB-SLAM3가 이 방식이다. 키프레임마다 BoW 벡터 $\mathbf{v}_i$를 만들고, 현재 프레임 $\mathbf{v}_q$와의 유사도를 $s(\mathbf{v}_q, \mathbf{v}_i) = 1 - \frac{1}{2} \left| \frac{\mathbf{v}_q}{\|\mathbf{v}_q\|} - \frac{\mathbf{v}_i}{\|\mathbf{v}_i\|} \right|$ (L1-score, 노름은 L1)로 계산한다.

2. **학습 기반**: [NetVLAD](https://arxiv.org/abs/1511.07247), [AnyLoc](https://arxiv.org/abs/2308.00688) 같은 global descriptor를 쓴다. AnyLoc은 [DINOv2](https://arxiv.org/abs/2304.07193) dense feature를 VLAD로 모아 장소 label로 학습하지 않고도 여러 domain에 쓴다. 그래도 새 센서나 환경에서는 recall을 직접 확인해야 한다. 후보 순위는 코사인 유사도로 매긴다:

$$s(\mathbf{d}_q, \mathbf{d}_i) = \frac{\mathbf{d}_q^\top \mathbf{d}_i}{\|\mathbf{d}_q\| \|\mathbf{d}_i\|}$$

**LiDAR loop closure detection**은 3D 포인트 클라우드 디스크립터를 쓴다:

- **[Scan Context](https://doi.org/10.1109/IROS.2018.8593953)**: 센서 중심 극좌표 격자의 칸마다 최대 높이를 기록해 공간 배치를 그대로 남긴다. ring key로 kd-tree 검색해 후보를 좁히고, 섹터를 돌려 맞춰 순위를 다시 매긴다. 섹터를 돌려 맞추므로 yaw가 달라도 비교할 수 있다. 반대 방향 재방문에서 함께 생기는 횡방향 이동은 원 논문이 root shifting으로 따로 다룬다(§9.3.1).
- **[PointNetVLAD](https://arxiv.org/abs/1804.03492), [OverlapTransformer](https://arxiv.org/abs/2203.03397)**: 학습 기반 3D place recognition이다. Scan Context보다 recall이 높은지는 데이터셋, 센서 스캔 패턴, 학습 domain에 따라 다르다.

**시간적 필터링**: 방금 지나온 frame과 맞는 것은 연속 tracking의 결과이므로, 시간상 가까운 키프레임은 후보에서 뺀다. 얼마나 떨어져야 후보로 볼지는 플랫폼 속도, 키프레임 생성 빈도, 재방문 형태에 맞춰 정한다. 아래 코드의 30초는 예시값이다.

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

Detection은 겉모습이 비슷한지만 보고 후보를 고르므로, 실제로는 다른 장소인데 비슷해 보이는 false positive가 섞인다. 서로 다른 장소가 비슷하게 보이는 현상을 perceptual aliasing이라 한다. 비슷한 복도와 반복 구조가 있는 실내가 그 예다.

verification은 보수적으로 한다. recall을 좀 잃더라도 precision을 최대한 높인다.

1. **2D-2D: Essential matrix 검증**: 현재 frame과 후보 키프레임의 특징점을 맞추고 RANSAC으로 essential matrix를 추정한다. 인라이어의 수와 비율, 인라이어가 이미지에 고르게 퍼져 있는지, 추정한 포즈가 그럴듯한지, 같은 결과가 반복해서 나오는지를 본다. 아래 코드의 20개·50%는 예시값이고, detector와 장면에 맞춰 정해야 한다.

$$\mathbf{p}_2^\top \mathbf{E} \mathbf{p}_1 = 0, \quad \mathbf{E} = [\mathbf{t}]_\times \mathbf{R}$$

2. **3D-3D: Point cloud registration**: LiDAR 시스템은 ICP나 GeoTransformer로 두 스캔 사이의 상대 변환 $\tilde{\mathbf{T}}_{ij} \in SE(3)$를 추정한다. fitness score(정합된 점의 비율)와 RMSE로 판정한다.

3. **2D-3D: PnP**: 현재 2D 특징점과 후보 키프레임에 딸린 3D 맵 포인트로 PnP 문제를 푼다. 3D 점을 후보 키프레임 좌표계로 두면 결과가 곧 loop edge에 필요한 상대 변환 $\tilde{\mathbf{T}}_{ij}$이고, 맵 좌표계로 두면 질의 프레임의 절대 포즈가 나오므로 후보의 포즈와 합성해야 한다.

4. **Temporal consistency**: 이어지는 여러 프레임에서 같은 장소가 계속 검출되는지 본다. ORB-SLAM2까지는 DBoW2가 연속한 키프레임 세 개에서 같은 장소를 검출해야 루프로 받아들였다. ORB-SLAM3는 이 조건을 맵 안의 기하 검증으로 바꿨다. 쿼리 키프레임과 covisible한(같은 맵 포인트를 함께 보는) 키프레임 두 개를 맵에서 찾아 셋 모두에서 변환이 맞으면 받아들인다. 검증에 필요한 정보가 대개 맵에 이미 있어서 새 키프레임을 기다릴 필요가 없고, 그만큼 루프를 일찍 잡는다. 맵에서 찾지 못하면 새로 들어오는 키프레임으로 검증을 이어 간다.

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
        T_relative: 상대 변환 (4, 4) 또는 None.
                    Essential 분해의 t는 크기가 정해지지 않은 단위 방향이므로
                    이 값을 pose graph의 loop edge로 쓰려면 스테레오·깊이·맵
                    포인트에서 metric scale을 따로 얻어야 한다.
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
    """8-point 선형 해법 + RANSAC으로 Essential matrix 추정.

    최소 표본 5개로 푸는 Nister의 5-point 알고리즘은 별도 해법이며
    최소 표본 크기가 다르므로 필요한 RANSAC 반복 수도 다르다.
    """
    best_E = None
    best_inliers = np.zeros(len(pts1), dtype=bool)
    
    for _ in range(max_iter):
        # 8개 점 랜덤 샘플링
        idx = np.random.choice(len(pts1), 8, replace=False)
        
        # 8-point 선형 해법으로 E 후보 생성 (5-point와는 별개 해법)
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

로봇이 비슷하게 생긴 복도 두 개를 지나간다고 하자. 복도 A의 키프레임 $i$와 복도 B의 키프레임 $j$ 사이에 잘못된 loop closure가 들어가면, pose graph optimizer는 이 두 포즈를 가깝게 끌어당긴다. 그 사이의 포즈가 모두 끌려가 맵이 접히거나 뒤틀린다.

이를 막는 방법은 다음과 같다.

1. **다단계 검증**: 겉모습 유사도, 기하 일관성, 시간 일관성 검사를 차례로 모두 통과해야 받아들인다.

2. **Robust kernel 사용** (§10.2에서 상세): 잘못된 제약이 들어와도 optimizer가 덜 흔들리게 한다.

3. **[Switchable constraints (Sünderhauf & Protzel, 2012)](https://doi.org/10.1109/IROS.2012.6385590)** (§10.2에서 상세): 다른 제약과 맞지 않는 loop closure를 optimizer가 스스로 끄게 한다.

4. **[DCS (Dynamic Covariance Scaling) (Agarwal et al., 2013)](https://doi.org/10.1109/ICRA.2013.6630557)** (§10.2에서 상세): 잔차가 큰 loop closure의 영향을 줄인다.

5. **[PCM (Pairwise Consistency Maximization) (Mangelson et al., 2018)](https://doi.org/10.1109/ICRA.2018.8460217)**: loop closure 후보를 둘씩 짝지어 서로 맞는지(pairwise consistency) 검사하고, 서로 모두 맞는 가장 큰 집합만 받아들인다.

### 10.1.4 Correction: 그래프 보정

검증을 통과한 loop closure는 pose graph에 새 제약으로 들어간다. Loop closure edge에는 두 포즈 사이의 측정된 상대 변환 $\tilde{\mathbf{T}}_{ij}$와 그 불확실성 $\boldsymbol{\Sigma}_{ij}$가 담긴다:

$$\mathbf{e}_{ij} = \text{Log}(\tilde{\mathbf{T}}_{ij}^{-1} \cdot \mathbf{T}_i^{-1} \cdot \mathbf{T}_j)$$

이 엣지가 들어오면 pose graph optimizer(§10.2)가 그래프 전체를 다시 최적화해 드리프트를 줄인다. 최적화가 포즈를 옮기면 오차가 여러 엣지에 나뉘어 실린다. 어느 엣지가 오차를 얼마나 떠안을지는 엣지별 정보 행렬 $\boldsymbol{\Omega}_{ij} = \boldsymbol{\Sigma}_{ij}^{-1}$와 그래프 연결 구조가 정한다. 정보 행렬이 큰(더 믿을 만한) odometry 구간은 덜 움직인다.

---

## 10.2 Pose Graph Optimization

Pose graph optimization은 프론트엔드(odometry, loop closure)가 만든 상대 제약을 최대한 함께 만족하도록 전체 포즈 궤적을 조정한다.

### 10.2.1 SE(3) Pose Graph

Pose graph는 그래프 $\mathcal{G} = (\mathcal{V}, \mathcal{E})$다:

- **노드** $\mathcal{V} = \{\mathbf{T}_1, \mathbf{T}_2, \ldots, \mathbf{T}_n\}$: 각 키프레임의 포즈, $\mathbf{T}_i \in SE(3)$.
- **엣지** $\mathcal{E}$: 두 노드 사이의 상대 제약. odometry edge와 loop closure edge가 있다.

각 엣지 $(i, j) \in \mathcal{E}$에는 측정된 상대 변환 $\tilde{\mathbf{T}}_{ij}$와 정보 행렬(information matrix) $\boldsymbol{\Omega}_{ij}$가 딸린다.

**SE(3) 위의 오차**: 포즈는 유클리드 벡터가 아니므로, 오차를 Lie group SE(3) 위에서 정의한다.

$$\mathbf{e}_{ij} = \text{Log}(\tilde{\mathbf{T}}_{ij}^{-1} \cdot \mathbf{T}_i^{-1} \cdot \mathbf{T}_j) \in \mathbb{R}^6$$

여기서 $\text{Log}: SE(3) \to \mathfrak{se}(3) \cong \mathbb{R}^6$은 SE(3)의 원소를 Lie algebra로 옮기는 행렬 로그(matrix logarithm)다. 결과인 6차원 벡터에는 회전 오차 3개와 병진 오차 3개가 들어 있다.

**최적화 목표**: 모든 엣지 오차의 가중 제곱합을 최소화한다:

$$\mathbf{T}^* = \arg\min_{\mathbf{T}_1, \ldots, \mathbf{T}_n} \sum_{(i,j) \in \mathcal{E}} \mathbf{e}_{ij}^\top \boldsymbol{\Omega}_{ij} \mathbf{e}_{ij}$$

이 nonlinear least squares 문제는 Gauss-Newton이나 Levenberg-Marquardt로 푼다.

**Gauss-Newton on manifold**: SE(3)는 매니폴드라 덧셈으로 갱신할 수 없다. Lie algebra에서 증분(increment) $\boldsymbol{\delta}$를 계산하고 exponential map으로 포즈에 곱한다:

$$\mathbf{T}_i \leftarrow \mathbf{T}_i \cdot \text{Exp}(\boldsymbol{\delta}_i)$$

한 번 반복(iteration)할 때의 순서:

1. 현재 추정치에서 각 엣지의 오차 $\mathbf{e}_{ij}$와 Jacobian $\mathbf{J}_{ij}$를 계산한다.
2. Normal equation을 세운다: $\mathbf{H} \boldsymbol{\delta} = -\mathbf{b}$, 여기서

$$\mathbf{H} = \sum_{(i,j)} \mathbf{J}_{ij}^\top \boldsymbol{\Omega}_{ij} \mathbf{J}_{ij}, \quad \mathbf{b} = \sum_{(i,j)} \mathbf{J}_{ij}^\top \boldsymbol{\Omega}_{ij} \mathbf{e}_{ij}$$

3. $\mathbf{H}$는 sparse하므로 sparse Cholesky 분해로 빨리 푼다. 단, 상대 제약만 있으면 그래프 전체를 같은 SE(3) 변환으로 옮겨도 목적식이 그대로라서 $\mathbf{H}$에 6차원 영공간이 생긴다. 기준 포즈 하나를 고정하거나 prior factor를 넣어야 $\mathbf{H}$가 양의 정부호가 되어 Cholesky 분해를 할 수 있다.
4. 증분을 적용한다: $\mathbf{T}_i \leftarrow \mathbf{T}_i \cdot \text{Exp}(\boldsymbol{\delta}_i)$.
5. 수렴할 때까지 반복한다.

**[g2o (General Graph Optimization)](https://doi.org/10.1109/ICRA.2011.5979949)**: Kümmerle et al.(ICRA 2011)이 만든 프레임워크로, 위 과정을 일반 그래프 최적화 문제에 쓸 수 있게 했다. 사용자가 vertex(노드) 타입과 edge(제약) 타입을 정의하면 나머지 sparse 최적화는 라이브러리가 한다. ORB-SLAM3도 g2o로 비선형 최적화를 한다.

```python
import numpy as np
from scipy.spatial.transform import Rotation

def se3_log(T):
    """SE(3) 행렬 → 6차원 벡터 [병진; 회전]."""
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
        edges: [(i, j), ...] 엣지 인덱스 리스트
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

실제 SLAM 시스템에는 잘못된 loop closure, 센서 오류, 움직이는 물체 때문에 이상치(outlier) 측정이 들어온다. 표준 least squares 비용 $\rho(s) = s$에서는 큰 오차 하나가 비용을 거의 다 차지해 해 전체를 끌고 간다.

**Robust kernel**(M-estimator)은 큰 잔차의 영향을 제한해 이상치에 덜 흔들리게 한다.

| Kernel | $\rho(s)$ ($s = e^2$) | 특성 |
|--------|----------------------|------|
| Least Squares | $s$ | 이상치에 취약 |
| Huber | $\begin{cases} s & \text{if } \sqrt{s} \leq k \\ 2k\sqrt{s} - k^2 & \text{otherwise} \end{cases}$ | 임계값 $k$ 이상에서 선형 |
| Cauchy | $k^2 \log(1 + s/k^2)$ | 부드러운 감쇠 |
| Geman-McClure | $\frac{s}{k^2 + s}$ | 강한 이상치 억제 |

Robust kernel을 적용한 비용 함수:

$$\sum_{(i,j)} \rho\left(\mathbf{e}_{ij}^\top \boldsymbol{\Omega}_{ij} \mathbf{e}_{ij}\right)$$

이 비용은 **IRLS(Iteratively Reweighted Least Squares)**로 푼다. 반복할 때마다 잔차 크기로 가중치를 다시 계산하고 가중 least squares를 푼다.

$$w_i = \rho'(s_i), \quad s_i = \mathbf{e}_i^\top \boldsymbol{\Omega}_i \mathbf{e}_i$$

여기서 $\rho'$은 위 표의 $\rho(s)$를 $s$로 미분한 값이다. 잔차가 큰 엣지는 가중치가 작아져 영향이 줄어든다.

**[Switchable constraints](https://doi.org/10.1109/IROS.2012.6385590)**: Sünderhauf & Protzel(IROS 2012)은 loop closure factor마다 연속 스위치 변수 $s_{ij} \in [0, 1]$을 두어, 다른 제약과 맞지 않는 loop closure를 optimizer가 끌 수 있게($s_{ij} \to 0$) 했다.

$$\rho_{\text{switch}}(\mathbf{e}_{ij}, s_{ij}) = s_{ij}^2 \mathbf{e}_{ij}^\top \boldsymbol{\Omega}_{ij} \mathbf{e}_{ij} + \lambda (1 - s_{ij})^2$$

첫째 항은 스위치가 켜진 만큼 루프 오차를 비용에 넣고, 둘째 항은 스위치를 끌수록 벌점을 준다. 그래서 optimizer는 오차가 아주 큰 루프만 끈다.

**[DCS](https://doi.org/10.1109/ICRA.2013.6630557)** (Agarwal et al., ICRA 2013): loop closure마다 잔차 크기에 따라 공분산을 조정한다. 잔차가 큰 루프는 공분산이 커져 영향이 줄어든다.

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

새 키프레임이 들어올 때마다 그래프 전체를 다시 최적화하는 batch 방식은 그래프가 커질수록 느려진다. [iSAM2 (Kaess et al., 2012)](https://doi.org/10.1177/0278364911430419)는 새 변수나 측정이 들어오면 영향을 받는 부분만 찾아 다시 계산한다(§4.5.6).

**Bayes tree**: iSAM2가 쓰는 자료구조로, clique tree와 비슷하지만 방향이 있다.

MAP 추정에 쓰는 사후분포는 factor graph에서 이렇게 인수분해된다:

$$p(\mathbf{x} | \mathbf{z}) \propto \prod_k f_k(\mathbf{x}_k)$$

여기서 $f_k$는 각 factor(odometry, loop closure 등)이고, $\mathbf{x}_k$는 해당 factor에 관련된 변수들이다.

변수를 차례로 소거하면 사후분포는 조건부 분포의 곱(Bayes net)이 된다:

$$p(\mathbf{x} | \mathbf{z}) = \prod_i p(x_i | \text{Sep}(x_i))$$

여기서 $\text{Sep}(x_i)$는 $x_i$의 separator로, $x_i$를 소거하는 시점에 $x_i$와 factor로 바로 이어져 있던 변수들이다. Bayes tree는 이 조건부 분포들을 clique 단위로 묶은 것이다. Clique $C_k$마다 frontal 변수 $F_k$와 separator $S_k$(부모 clique와 겹치는 변수)가 있고, 전체 분포는 clique별 조건부의 곱이 된다([Kaess et al., 2012](https://doi.org/10.1177/0278364911430419), 식 (13)):

$$p(\mathbf{x} | \mathbf{z}) = \prod_k p(F_k | S_k)$$

**Incremental update**: 새 factor가 들어오면 영향받는 clique에서 root까지만 다시 분해하고 나머지 tree는 그대로 둔다. 소거 순서도 다시 분해하는 부분만 새로 정한다. 선형화 지점이 현재 추정치에서 기준보다 멀어진 변수만 골라 다시 선형화한다(fluid relinearization). 다만 큰 loop closure가 들어오면 다시 선형화할 범위도 넓어진다.

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
        """새 엣지(제약) 추가. 연결된 변수를 affected로 표시."""
        self.edges.append((i, j, T_ij, Omega_ij))
        self.affected.add(i)
        self.affected.add(j)
        # 실제 iSAM2에서는 Bayes tree를 타고 올라가며
        # ancestor clique들도 affected로 표시한다.
        
    def add_loop_closure(self, i, j, T_ij, Omega_ij):
        """
        Loop closure 엣지 추가.
        일반 엣지와 동일하지만, loop closure는 먼 과거의 노드와
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
            # 1. 관련 엣지만 선별
            relevant_edges = [
                (i, j, T_ij, Omega) 
                for (i, j, T_ij, Omega) in self.edges
                if i in self.affected or j in self.affected
            ]
            
            # 2. 각 엣지에 대해 오차와 Jacobian 계산
            # 3. Normal equation 구성 및 풀기
            # 4. Affected 변수에만 increment 적용
            
            # (실제 구현은 sparse linear system을 풀어야 하므로 생략)
            pass
        
        self.affected.clear()
```

---

## 10.3 Global Relocalization

Global relocalization은 로봇이 미리 만든 맵(prior map)에서 자기 위치를 찾는 문제다. 로봇이 tracking을 놓쳤거나 어디서 출발했는지 모를 때 쓴다.

### 10.3.1 Map-Based Localization

이미 만든 맵에 새 센서 관측을 맞춰 현재 포즈를 추정한다.

**Visual relocalization**: 현재 이미지에서 특징점을 뽑고, visual word나 직접 매칭으로 맵의 3D 포인트와 2D-3D 대응을 찾은 뒤, PnP + RANSAC으로 포즈를 추정한다. 그 포즈에서 tracking을 다시 시작한다.

ORB-SLAM3는 DBoW2로 후보 키프레임을 찾고, ORB 매칭으로 2D-3D 대응을 얻은 뒤, MLPnP + RANSAC으로 포즈를 추정한다. ORB-SLAM2까지는 EPnP를 썼는데, EPnP는 calibrated pinhole 카메라를 가정하므로 ORB-SLAM3는 카메라 모델과 무관한 MLPnP로 바꿨다. 그다음 guided search로 매칭을 더 찾아 정확도를 높인다.

**LiDAR relocalization**: 현재 LiDAR 스캔을 미리 만든 point cloud map에 맞춘다. Scan Context나 PointNetVLAD 같은 global descriptor로 가까운 영역을 먼저 찾고, FPFH + RANSAC이나 GeoTransformer로 대략 맞춘 뒤(coarse registration), ICP/GICP로 정밀하게 맞춘다.

### 10.3.2 Prior Map + Online Sensor

자율주행차는 미리 만든 HD map(고정밀 지도)에 실시간 센서 데이터를 맞춰 위치를 찾기도 한다. 이때 다음 문제를 풀어야 한다.

- **맵과 환경의 불일치**: 시간이 지나면 건물이 바뀌고 나뭇잎이 자란다. prior map과 지금 관측이 다른 부분을 감당해야 한다.
- **Cross-modal matching**: HD map은 LiDAR로 만들었는데 지금 차에는 카메라만 있을 수도 있다. 그러면 서로 다른 센서의 데이터를 맞춰야 한다.
- **초기 포즈 없음**: 로봇이 맵의 어디서 출발하는지 모르면, 맵 전체를 대상으로 place recognition을 돌려 시작 위치를 찾는다.

### 10.3.3 Monte Carlo Localization (MCL)

MCL은 particle filter로 하는 global localization이다. 로봇이 있을 법한 포즈를 particle(표본) 여러 개로 나타내고, 센서 관측에 따라 particle마다 가중치를 고친다. 초기화한 뒤에는 Prediction, Update, Resampling을 되풀이한다.

1. **초기화**: 위치를 전혀 모르므로 맵 전체에 particle을 고르게 뿌린다.
2. **Prediction**: 로봇 모션 모델에 따라 particle을 옮긴다:
$$x_t^{[k]} \sim p(x_t | u_t, x_{t-1}^{[k]})$$
3. **Update**: 실제 센서 관측과, 그 particle 위치에서 맵으로 예측한 관측을 비교해 가중치를 계산한다:
$$w_t^{[k]} = p(z_t | x_t^{[k]}, m)$$
4. **Resampling**: 가중치에 비례하게 particle을 다시 뽑는다. 가중치가 큰 particle은 여러 개로 복제되고 작은 particle은 사라진다.

MCL은 봉우리가 여러 개인(multi-modal) 분포도 나타낸다. 로봇이 여러 장소 중 어디에 있는지 모르면 여러 가설을 함께 들고 간다. 관측이 장소를 구별해 주고 정답 근방의 particle이 남아 있으면 가설이 하나로 줄어든다. 대칭 복도처럼 관측이 비슷한 장소에서는 다중 모드가 계속 남는다. 또 particle이 너무 적거나 가능도가 지나치게 뾰족하면 정답 근처 particle이 모두 사라져 잘못된 위치로 굳는다.

LiDAR 기반 MCL 예시:

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
            
            self.particles[k, 3] = log_weight   # 로그 가중치를 먼저 모은다
        
        # 가중치 정규화 — 최대 로그가중치를 뺀 뒤 exp (빔 수백 개의 로그합은 그대로 exp하면 언더플로)
        log_w = self.particles[:, 3]
        self.particles[:, 3] = np.exp(log_w - np.max(log_w))
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
        """고정 간격 샘플링 기반 간단한 ray casting.

        실수 좌표에 방향 벡터를 더하며 셀을 조회하므로 셀을 건너뛰거나
        중복 조회할 수 있다. 직선이 지나는 셀을 빠짐없이 순회해야 하면
        Amanatides-Woo DDA를 쓴다. Bresenham은 셀을 한 번씩만 조회하지만
        8-연결 래스터화라 직선이 지나는 셀 일부를 여전히 빠뜨린다.
        """
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
                return (step + 1) * self.resolution   # 전진 후 검사하므로 검사한 점의 거리는 (step+1)칸
        
        return max_range
```

---

## 10.4 Multi-Session SLAM & Map Merging

SLAM을 한 번에 끝내지 못할 때도 있다. 같은 건물을 며칠에 나눠 매핑하기도 하고, 로봇 여러 대가 서로 다른 영역을 동시에 탐사하기도 하고, tracking을 놓친 뒤 다른 곳에서 다시 시작하기도 한다. Multi-session SLAM은 이렇게 따로 만든 맵들을 하나의 맵으로 합친다.

### 10.4.1 Map Anchoring

여러 세션의 맵을 합치려면 먼저 맵마다 다른 좌표계(coordinate frame)를 맞춰야 한다. 이를 map anchoring이라 한다.

**방법 1 — 공유 랜드마크 기반**: 두 세션이 겹치는 영역에서 같은 3D 포인트나 특징점을 보았다면, 그 점들로 두 맵 사이의 상대 변환 ${}^{A}\mathbf{T}_{B}$를 추정한다.

$${}^{A}\mathbf{T}_{B} = \arg\min_{\mathbf{T}} \sum_k \| \mathbf{p}_k^A - \mathbf{T} \cdot \mathbf{p}_k^B \|^2$$

**방법 2 — Place recognition 기반**: 공유 랜드마크가 없으면 place recognition으로 두 세션에서 같은 장소를 지난 키프레임 쌍을 찾아 맞춘다.

**방법 3 — GNSS 기반**: 두 세션 모두 GNSS 위치가 있으면 GNSS 좌표계를 기준으로 바로 맞춘다.

### 10.4.2 Inter-Session Loop Closure

map anchoring으로 대략 맞춘 다음 inter-session loop closure로 정밀하게 맞춘다. 원리는 일반 loop closure와 같지만 어려운 점이 더 있다.

1. **외관 변화**: 세션 사이에 조명과 계절이 바뀐다. AnyLoc 같은 foundation model 기반 디스크립터가 이런 변화에서 기존 방법보다 recall이 높았다(§9.2.4).

2. **좌표계 불일치**: 처음 맞춘 결과가 부정확할 수 있으므로 후보를 넓게 찾는다. pose prior로 거르는 범위를 넓히고, 후보 수와 RANSAC 반복 횟수를 늘리고, 정합 초기값으로 받아들이는 범위도 넓힌다. 기하 검증에서 잔차를 받아들이는 기준은 그대로 둔다. 이 기준까지 느슨하게 하면 잘못된 루프가 들어와 맵이 깨진다.

### 10.4.3 ORB-SLAM3 Multi-Map System

ORB-SLAM3의 Atlas는 여러 맵을 한꺼번에 관리하는 구조로, 다음처럼 동작한다.

1. **Active map**: 지금 tracking에 쓰는 맵이다. 평소에는 이 맵에 키프레임과 맵 포인트를 더한다.

2. **Map creation**: 시야가 가려져 tracking을 놓치면 새 맵을 만들어 active map으로 삼는다. 이전 맵은 Atlas에 비활성 맵으로 남는다.

3. **Map merging**: place recognition이 active map과 Atlas의 비활성 맵에서 같은 장소를 찾으면 두 맵을 합친다:
   - 같은 장소를 본 키프레임 쌍에서 두 맵 사이의 변환 $\mathbf{T}_{merge}$를 추정한다.
   - 두 키프레임과 각각의 covisible 키프레임, 이들이 본 맵 포인트로 welding window를 만든다. 이때 active map 쪽 요소를 $\mathbf{T}_{merge}$로 변환해 비활성 맵 좌표계에 맞춘다.
   - 두 맵을 하나로 합쳐 새 active map으로 삼고, 중복된 맵 포인트를 합친다.
   - welding window 안에서 welding bundle adjustment를 하고, 그 보정을 pose graph optimization으로 맵의 나머지 부분에 퍼뜨린다.

4. **Multi-session 활용**: 이전 세션의 맵을 불러온 상태에서 같은 장소를 다시 지나가면 맵이 자동으로 합쳐진다. 이렇게 세션을 거듭하며 맵을 넓혀 간다.

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
        (단순화: ORB-SLAM3는 반대로 active map 쪽을 비활성 맵 좌표계로 옮긴다. §10.4.3)
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

로봇 여러 대가 동시에 탐사하면 각 로봇의 맵을 실시간으로 합쳐야 한다. 이때는 제약이 더 붙는다:

- **통신 대역폭**: 포인트 클라우드를 통째로 보낼 수 없으므로 후보를 찾을 때는 Scan Context나 compact visual descriptor 같은 압축 디스크립터만 주고받는다. 기하 검증에는 희소 특징이나 일부만 뽑은 포인트 클라우드를 보내고, 분산 최적화에는 다른 로봇과 이어진 포즈의 추정치와 불확실성을 보낸다.
- **분산 최적화**: 중앙 서버 없이 로봇끼리 맵을 합쳐야 할 때가 있다. Kimera-Multi와 Swarm-SLAM이 이 문제를 다룬다.
- **상대 포즈 불확실성**: 로봇 사이의 처음 상대 포즈를 모르므로 inter-robot loop closure로 맞춰야 한다.

분산 pose graph optimization은 아래 식으로 쓸 수 있다.

$$\mathbf{T}^* = \arg\min \sum_{\text{robot } r} \sum_{(i,j) \in \mathcal{E}_r} \rho(\mathbf{e}_{ij}) + \sum_{(i,j) \in \mathcal{E}_{\text{inter}}} \rho(\mathbf{e}_{ij})$$

로봇마다 자기 엣지 $\mathcal{E}_r$로 최적화를 돌리고, inter-robot 엣지 $\mathcal{E}_{\text{inter}}$에 걸린 포즈의 정보만 주고받는다. 분산 계산에는 ADMM이나 Gauss-Seidel 방식의 블록 좌표 하강을 쓴다. 회전이 들어간 pose graph 목적식은 비볼록이므로 볼록 문제에 대한 ADMM 수렴 보장은 여기에 그대로 적용되지 않는다. 비볼록 ADMM이 정류점으로 수렴한다는 [Hong et al.의 결과](https://arxiv.org/abs/1410.1390)도 특정 consensus·sharing 문제 구조와 충분히 큰 페널티 계수를 조건으로 한다. 초기값과 통신 그래프에 따라서도 수렴 과정과 얻는 해가 달라진다. [DC2-PGO](https://arxiv.org/abs/1911.03721)는 semidefinite relaxation으로 분산 PGO를 풀고, 측정 잡음이 적당한 범위에 있으면 얻은 해가 전역 최적인지 분산 방식으로 검증한다(T-RO).

---

## 10.5 최근 연구

**[riSAM (McGann et al., 2023)](https://arxiv.org/abs/2209.14359)**: iSAM2에 [Graduated Non-Convexity](https://arxiv.org/abs/1909.08605)(GNC)를 합친 incremental robust backend다(ICRA 2023). 논문은 MIT CSAIL과 Intel Research Lab 데이터셋에 전체 loop closure의 10%에서 90%까지 잘못된 loop closure를 섞어 실험했고, 온라인 방법 가운데 가장 좋았으며 offline 방법과 비슷하거나 더 나았다고 보고한다. outlier 구조나 초기값이 다른 문제에서도 그렇다는 보장은 없다.

**[Kimera2 (Abate et al., 2024)](https://arxiv.org/abs/2401.06323)**: Kimera를 개선한 버전이다(ISER 2023). pose graph 백엔드 Kimera-RPGO는 그동안 PCM으로 잘못된 loop closure를 걸렀는데, 여기에 GNC를 선택지로 더했다. Kimera-VIO의 feature tracking과 키프레임 선택도 고쳤다. 드론, 사족 보행 로봇, 바퀴 로봇, 시뮬레이션 자율주행차에서 평가했다.

**[Group-k Consistent Measurement Set Maximization (Forsgren et al., 2022)](https://arxiv.org/abs/2209.02658)**: 측정 둘만으로는 서로 맞는지 판단하기 어려운 경우가 있어서, PCM의 pairwise consistency를 측정 k개 묶음의 일관성(group-k consistency)으로 넓혔다. 저자들은 시뮬레이션 데이터로 PCM과 비교했다.

