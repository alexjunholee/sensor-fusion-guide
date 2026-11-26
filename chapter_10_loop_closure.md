# 10장 루프 클로저와 전역 최적화

> 장소 인식(place recognition)이 "여기가 아까 지나온 곳인가?"를 알아맞히는 문제라면, 루프 클로저와 전역 최적화는 그 재방문 정보를 기하학적 제약으로 바꾸어 궤적과 지도의 뒤틀림을 한꺼번에 바로잡는 과정이다.

9장에서 다룬 place recognition은 로봇이 과거에 방문했던 장소를 다시 알아보는 능력을 제공한다(→ 9장). 그러나 "이곳을 전에 본 적이 있다"는 사실을 알아내는 것만으로는 부족하다. 오도메트리(odometry)는 매 순간의 상대 이동량을 적분해 나가므로, 센서가 아무리 정밀하더라도 미세한 오차가 시간에 따라 누적되기 마련이다. 이 누적 드리프트가 방치되면 로봇이 한 바퀴를 돌아 출발점으로 되돌아왔을 때, 실제로는 같은 복도나 교차로가 지도 위에서는 서로 다른 위치에 이중으로 그려지고 만다. **루프 클로저(loop closure)**는 재방문한 두 시점 사이에 정밀한 상대 포즈 제약을 걸어, 그동안 쌓인 드리프트를 전체 궤적에 걸쳐 역으로 해소하는 메커니즘이다.

이 장에서는 먼저 후보 탐지·기하학적 검증·그래프 보정으로 이어지는 루프 클로저 파이프라인을 살펴본 뒤, 그 핵심 백엔드인 $SE(3)$ 포즈 그래프 최적화(pose graph optimization)와 이상치 대응 기법, 그리고 실시간 증분 최적화(iSAM2)를 다룬다. 이어서 추적이 완전히 끊겼거나 사전 지도가 주어진 상황에서 출발 위치를 찾는 전역 재측위(global relocalization)와, 여러 세션·여러 로봇이 만든 지도를 하나로 합치는 멀티세션 SLAM까지 확장해 나간다.

---

## 10.1 Loop Closure Pipeline

루프 클로저 파이프라인은 크게 **후보 탐지(Detection)**, **기하학적 검증(Verification)**, **그래프 보정(Correction)**이라는 세 단계를 순차적으로 밟는다. 이 중 보정 단계의 핵심은 뒤에서 자세히 다룰 포즈 그래프 최적화다. 이 파이프라인에는 뚜렷한 비대칭적 위험이 도사리고 있다. 진짜 루프 후보를 몇 번 놓치더라도 누적된 드리프트를 바로잡을 기회를 미룰 뿐이지만, 전혀 다른 두 장소를 같은 곳으로 착각한 오검출(false positive)을 단 하나라도 그래프에 받아들이면 지도 전체가 회복 불가능할 정도로 접히고 뒤틀린다.

### 10.1.1 Detection: 후보 탐지

후보 탐지 단계는 "현재 관측이 과거에 저장해 둔 키프레임 중 어느 것과 닮았는가?"라는 질문에 답한다. 문제 정의 자체는 9장에서 살펴본 place recognition과 맞닿아 있으며(→ 9장), 사용하는 센서에 따라 접근 방식이 갈린다.

**Visual loop closure detection**은 현재 카메라 영상에서 추출한 디스크립터를 과거 키프레임 데이터베이스와 대조한다.

1. **BoW(Bag of Words) 기반**: ORB-SLAM3 등에서 채택한 DBoW2 방식은 영상 내 ORB 특징점들을 사전 학습된 어휘 트리에 투영해 visual word 히스토그램으로 요약한다. 키프레임마다 BoW 벡터 $\mathbf{v}_i$를 만들어 두었다가, 현재 프레임의 벡터 $\mathbf{v}_q$와의 유사도를 L1-score(노름은 L1) $s(\mathbf{v}_q, \mathbf{v}_i) = 1 - \frac{1}{2} \left| \frac{\mathbf{v}_q}{\|\mathbf{v}_q\|} - \frac{\mathbf{v}_i}{\|\mathbf{v}_i\|} \right|$로 계산한다.

2. **학습 기반**: [NetVLAD](https://arxiv.org/abs/1511.07247)나 [AnyLoc](https://arxiv.org/abs/2308.00688)처럼 영상 전체를 하나의 고정 길이 global descriptor로 압축해 비교하는 방식이다. 특히 AnyLoc은 [DINOv2](https://arxiv.org/abs/2304.07193)의 조밀한(dense) 패치 특징을 VLAD로 집계해, 특정 장소 데이터셋으로 별도 학습을 거치지 않고도 다양한 도메인에 폭넓게 적용할 수 있다. 다만 사전 학습 모델이라도 새로운 센서 화각이나 낯선 조명 환경에서는 재현율(recall)이 기대에 미치지 못할 수 있으므로, 실전 투입 전에 목표 환경에서 검증해야 한다. 후보 간 순위는 보통 아래 코사인 유사도로 매긴다.

$$s(\mathbf{d}_q, \mathbf{d}_i) = \frac{\mathbf{d}_q^\top \mathbf{d}_i}{\|\mathbf{d}_q\| \|\mathbf{d}_i\|}$$

**LiDAR loop closure detection**은 조명에 영향을 받지 않는 3D 포인트 클라우드 디스크립터를 활용한다.

- **[Scan Context](https://doi.org/10.1109/IROS.2018.8593953)**: 센서를 중심으로 한 극좌표 격자(ring × sector)의 각 칸에 최대 높이 값을 기록해 주변 공간의 3D 배치를 2D 행렬로 압축한다. 먼저 회전 불변 특성을 지닌 ring key로 kd-tree를 검색해 후보군을 빠르게 추린 뒤, 섹터 열(column)을 한 칸씩 순환 이동시켜 맞춰 보며 순위를 다시 매긴다. 덕분에 로봇이 이전과 다른 방향(yaw)으로 진입하더라도 같은 장소를 알아볼 수 있다. 한편 반대 방향으로 재방문할 때 함께 동반되는 횡방향 이동(lateral offset)은 원 논문에서 root shifting 기법으로 별도 처리한다(§9.3.1).
- **[PointNetVLAD](https://arxiv.org/abs/1804.03492), [OverlapTransformer](https://arxiv.org/abs/2203.03397)**: 3D 포인트 클라우드나 거리 영상(range image)에서 직접 전역 특징을 학습하는 딥러닝 기반 방법이다. 기하학적 수작업 디스크립터인 Scan Context보다 실제로 더 높은 재현율을 낼지는 데이터셋의 특성, LiDAR의 스캔 패턴(채널 수와 빔 분포), 그리고 학습 도메인과의 일치 여부에 따라 달라진다.

**시간적 필터링**: 어떤 디스크립터를 쓰든, 방금 몇 초 전에 지나온 직전 프레임은 현재 프레임과 시야가 크게 겹치므로 유사도가 매우 높게 나온다. 하지만 이는 연속적인 추적이 잘 이어지고 있다는 뜻일 뿐, 드리프트를 보정할 수 있는 진짜 루프가 아니다. 따라서 현재 시점과 시간 간격이 가까운 키프레임은 후보군에서 반드시 제외해야 한다. 구체적인 최소 시간 간격은 로봇의 이동 속도, 키프레임 생성 빈도, 예상되는 재방문 경로의 길이에 맞춰 조율한다.

아래 `detect_loop_candidates` 함수는 후보 탐지 단계의 전형적인 구현이다. 시간적 필터링으로 최근 프레임을 걸러 낸 뒤, L2 정규화된 디스크립터 사이의 코사인 유사도를 구해 임계값을 넘는 상위 $k$개 후보를 뽑아낸다(코드의 `min_time_gap=30.0`은 설명을 위한 예시값이다).

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

후보 탐지 단계는 외관이나 전역 디스크립터의 유사도만 비교해 후보를 건져 올리므로, 실제로는 전혀 다른 장소인데 겉모습이 닮아 후보로 올라오는 오검출(false positive)이 섞여 들어오기 마련이다. 이처럼 서로 다른 공간이 센서 관측상 비슷하게 보이는 현상을 **지각적 에일리어싱(perceptual aliasing)**이라 부른다. 사무실 건물의 판박이 같은 복도나 반복 구조가 이어지는 아파트 단지, 나무와 가로수가 줄지어 선 공원 산책로가 전형적인 사례다.

따라서 기하학적 검증 단계에서는 재현율(recall)을 일부 포기하더라도 정밀도(precision)를 극대화하는 보수적 기준을 적용한다. 탐지된 두 관측 사이에 물리적으로 일관된 3D 강체 변환이 성립하는지, 그리고 그 변환이 시간 흐름과 주변 지도 구조에도 어긋나지 않는지를 아래의 기법들로 교차 확인한다.

1. **2D-2D: Essential matrix 검증**: 현재 프레임과 후보 키프레임 사이에서 지역 특징점을 매칭한 뒤, RANSAC으로 에피폴라 구속 조건(epipolar constraint)을 만족하는 essential matrix $\mathbf{E}$를 추정한다.

$$\mathbf{p}_2^\top \mathbf{E} \mathbf{p}_1 = 0, \quad \mathbf{E} = [\mathbf{t}]_\times \mathbf{R}$$

이때 인라이어(inlier) 개수만 세는 데 그치지 않고, 전체 매칭 대비 인라이어 비율과 함께 인라이어 특징점이 이미지 전역에 고르게 퍼져 있는지(공간 분포), 분해된 상대 포즈가 물리적으로 타당한지, 그리고 다른 가설로 반복 검산해도 일관된 결과가 나오는지를 두루 살핀다. 아래 코드에 등장하는 `min_inliers=20`과 `min_inlier_ratio=0.5`는 이해를 돕기 위한 예시값일 뿐, 실제 시스템에서는 사용하는 특징점 검출기와 환경 특성에 맞춰 튜닝해야 한다.

2. **3D-3D: Point cloud registration**: LiDAR 기반 시스템에서는 두 스캔 사이의 정합을 직접 풀어 3D 상대 변환 $\tilde{\mathbf{T}}_{ij} \in SE(3)$를 추정한다. ICP나 학습 기반 정합기인 GeoTransformer를 적용한 뒤, 두 스캔이 맞물린 비율을 뜻하는 fitness score와 정합 잔차(RMSE)가 기준치를 넘는지 확인한다.

3. **2D-3D: PnP**: 현재 이미지의 2D 특징점과 후보 키프레임이 관측하고 있던 3D 맵 포인트를 대응시켜 PnP 문제를 푼다. 여기서 3D 점의 좌표를 어떤 좌표계에 두느냐에 따라 결과의 의미가 달라진다는 점에 유의해야 한다. 3D 점을 후보 키프레임의 지역 좌표계로 표현해 풀면 곧바로 루프 엣지에 필요한 상대 변환 $\tilde{\mathbf{T}}_{ij}$가 나오지만, 전역 맵 좌표계로 두면 질의 프레임의 절대 포즈가 계산되므로 후보 키프레임의 포즈와 합성해 상대 변환을 구해야 한다.

4. **Temporal consistency**: 단일 프레임 쌍이 우연히 기하 검증을 통과하는 경우를 막기 위해, 연속된 여러 프레임에 걸쳐 같은 장소가 일관되게 검출되는지 살핀다. ORB-SLAM2까지는 DBoW2가 연속된 세 개의 키프레임에서 같은 장소 후보를 계속 지목해야 비로소 루프로 승인했다. 반면 ORB-SLAM3는 대기 지연을 줄이기 위해 이 조건을 맵 내부의 기하 검증으로 대체했다. 질의 키프레임과 시야를 공유하는(covisible, 같은 맵 포인트를 함께 관측하는) 이웃 키프레임 두 개를 기존 맵에서 찾아, 세 키프레임 모두에서 추정된 변환이 서로 맞아떨어지면 즉시 루프로 받아들이는 방식이다. 검증에 필요한 정보가 이미 맵에 축적되어 있어 새 키프레임이 들어올 때까지 대기할 필요가 없으므로 루프를 훨씬 일찍 닫을 수 있으며, 맵에서 covisible 키프레임을 충분히 찾지 못한 경우에만 후속 키프레임이 들어올 때까지 검증을 이어 간다.

다음 코드는 2D-2D 방식의 기하 검증 과정을 구현한 예시다. 정규화 이미지 좌표에서 8-point RANSAC과 Sampson error로 essential matrix를 추정한 뒤, 상대 회전 $\mathbf{R}$과 방향 벡터 $\mathbf{t}$를 복원한다. (단, 단안 카메라의 essential matrix 분해에서 얻는 $\mathbf{t}$는 크기가 1인 방향 벡터일 뿐이므로, 이를 실제 포즈 그래프의 루프 엣지로 쓰려면 스테레오·깊이 센서나 기존 맵 포인트로부터 미터 단위의 스케일을 별도로 복원해야 한다.)

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

기하학적 검증을 거치더라도 교묘한 오검출이 빠져나갈 위험은 여전히 남는다. 로봇이 생김새가 똑같은 두 복도 A와 B를 차례로 지나가는 상황을 떠올려 보자. 만약 복도 A의 키프레임 $i$와 복도 B의 키프레임 $j$ 사이에 잘못된 루프 클로저 제약이 추가되면, 포즈 그래프 최적화기는 두 위치가 실제로 가깝다고 믿고 키프레임 $i$와 $j$를 한 지점으로 강하게 끌어당긴다. 그 여파로 두 시점 사이에 놓여 있던 수백 개의 포즈가 덩달아 끌려가면서, 멀쩡하던 지도 전체가 반으로 접히거나 심하게 뒤틀려 버린다.

이러한 치명적 오류를 막기 위해 실제 SLAM 시스템은 프런트엔드와 백엔드 양쪽에서 다층적인 방어선을 구축한다.

1. **다단계 검증**: 외관 유사도, 기하 일관성, 시간(또는 공가시성) 일관성 검사를 차례로 모두 통과한 후보만 백엔드로 넘긴다.
2. **Robust kernel 사용** (§10.2에서 상세 설명): 오차가 비정상적으로 큰 제약이 들어와도 최적화기의 비용 함수가 가파르게 치솟지 않도록 억제해, 잘못된 루프 하나가 전체 궤적을 끌고 가는 사태를 완화한다.
3. **[Switchable constraints (Sünderhauf & Protzel, 2012)](https://doi.org/10.1109/IROS.2012.6385590)** (§10.2에서 상세 설명): 각 루프 클로저 제약에 연속적인 스위치 변수를 달아, 주변의 다른 제약들과 도저히 어우러지지 않는 루프는 최적화기 스스로 연결을 꺼 버리도록 만든다.
4. **[DCS (Dynamic Covariance Scaling) (Agarwal et al., 2013)](https://doi.org/10.1109/ICRA.2013.6630557)** (§10.2에서 상세 설명): 별도의 스위치 변수를 추가하는 대신, 현재 잔차가 큰 루프 제약의 공분산을 동적으로 키워 영향력을 실시간으로 낮춘다.
5. **[PCM (Pairwise Consistency Maximization) (Mangelson et al., 2018)](https://doi.org/10.1109/ICRA.2018.8460217)**: 여러 루프 클로저 후보를 둘씩 짝지어 기하학적으로 서로 모순이 없는지(pairwise consistency) 검사한 뒤, 서로 모두 양립하는 가장 큰 부분집합(maximum clique)만 그래프에 반영한다.

### 10.1.4 Correction: 그래프 보정

이처럼 엄격한 검증과 이상치 방어선을 통과한 루프 클로저는 비로소 포즈 그래프에 새로운 제약으로 등록된다. 이때 루프 클로저 엣지에는 검증 단계에서 추정한 두 포즈 사이의 상대 변환 측정치 $\tilde{\mathbf{T}}_{ij}$와 그 추정의 불확실성을 담은 공분산 $\boldsymbol{\Sigma}_{ij}$가 함께 기록된다. 현재 그래프상의 두 포즈 추정치 $\mathbf{T}_i, \mathbf{T}_j$와 측정치 $\tilde{\mathbf{T}}_{ij}$ 사이의 불일치는 다음 6차원 오차 벡터로 정량화된다.

$$\mathbf{e}_{ij} = \text{Log}(\tilde{\mathbf{T}}_{ij}^{-1} \cdot \mathbf{T}_i^{-1} \cdot \mathbf{T}_j)$$

새 루프 엣지가 추가되면 포즈 그래프 최적화기(§10.2)가 그래프 전체를 재최적화해 누적된 드리프트를 궤적 전반에 걸쳐 분산·해소한다. 최적화 과정에서 포즈들이 조금씩 자리를 옮길 때, 루프가 닫히면서 드러난 누적 오차는 궤적을 이루는 수많은 엣지에 나뉘어 실린다. 이때 어느 구간이 오차를 더 많이 떠안고 어느 구간이 형태를 지킬지는 각 엣지의 정보 행렬 $\boldsymbol{\Omega}_{ij} = \boldsymbol{\Sigma}_{ij}^{-1}$과 그래프의 연결 구조가 결정한다. 정보 행렬이 큰(즉, 센서의 신뢰도가 높은) 오도메트리 구간은 거의 변형되지 않고 버티는 반면, 불확실성이 큰 구간이 오차를 더 많이 흡수하며 자연스럽게 전체 궤적이 맞춰진다.

---

## 10.2 Pose Graph Optimization

**포즈 그래프 최적화(pose graph optimization, PGO)**는 앞 절의 그래프 보정 단계에서 핵심 엔진 역할을 하며, 프런트엔드(오도메트리와 루프 클로저)가 건네준 모든 상대 제약을 가장 고르게 만족하는 전역 포즈 궤적을 찾아낸다. 수많은 3D 랜드마크와 원시 특징점 관측까지 모두 함께 푸는 번들 조정(bundle adjustment)과 달리, 포즈 사이의 상대 변환 제약만으로 그래프를 추상화하므로 대규모 궤적에서도 계산 효율을 유지할 수 있다.

### 10.2.1 SE(3) Pose Graph

포즈 그래프는 수학적으로 노드와 엣지로 이루어진 그래프 $\mathcal{G} = (\mathcal{V}, \mathcal{E})$로 표현된다.

- **노드** $\mathcal{V} = \{\mathbf{T}_1, \mathbf{T}_2, \ldots, \mathbf{T}_n\}$: 각 키프레임 시점의 6자유도 포즈로, 3D 강체 변환 군인 $\mathbf{T}_i \in SE(3)$의 원소다.
- **엣지** $\mathcal{E}$: 두 노드 사이에 걸린 상대 포즈 제약이다. 시간에 따라 인접한 키프레임을 잇는 **오도메트리 엣지**와, 떨어져 있던 두 시점을 재방문으로 잇는 **루프 클로저 엣지**로 구성된다.

각 엣지 $(i, j) \in \mathcal{E}$에는 센서 정합이나 오도메트리로 측정한 상대 변환 $\tilde{\mathbf{T}}_{ij}$와 그 측정의 정밀도를 나타내는 정보 행렬(information matrix) $\boldsymbol{\Omega}_{ij}$가 부여된다.

**$SE(3)$ 위의 오차 정의**: 3D 포즈는 평평한 유클리드 벡터 공간이 아니라 회전이 포함된 곡면 공간(매니폴드) 위에 놓여 있으므로, 두 변환의 차이를 단순 뺄셈으로 구할 수 없다. 대신 Lie group $SE(3)$ 위에서 오차를 정의한 뒤 접공간(tangent space)으로 펼쳐 6차원 벡터로 다룬다.

$$\mathbf{e}_{ij} = \text{Log}(\tilde{\mathbf{T}}_{ij}^{-1} \cdot \mathbf{T}_i^{-1} \cdot \mathbf{T}_j) \in \mathbb{R}^6$$

여기서 $\text{Log}: SE(3) \to \mathfrak{se}(3) \cong \mathbb{R}^6$은 $SE(3)$ 행렬을 Lie algebra 공간으로 사상하는 행렬 로그(matrix logarithm) 연산이다. 추정된 상대 변환 $\mathbf{T}_i^{-1}\mathbf{T}_j$와 측정된 상대 변환 $\tilde{\mathbf{T}}_{ij}$가 완벽히 일치한다면 괄호 안은 항등 행렬 $\mathbf{I}_4$가 되어 오차 $\mathbf{e}_{ij} = \mathbf{0}$이 된다. 결과로 나오는 6차원 벡터에는 3개의 병진 오차와 3개의 회전 오차가 담겨 있다.

**최적화 목표**: 그래프 안의 모든 엣지 오차에 정보 행렬로 가중치를 매긴 제곱합(Mahalanobis 거리 제곱합)을 최소화하는 포즈 집합을 찾는다.

$$\mathbf{T}^* = \arg\min_{\mathbf{T}_1, \ldots, \mathbf{T}_n} \sum_{(i,j) \in \mathcal{E}} \mathbf{e}_{ij}^\top \boldsymbol{\Omega}_{ij} \mathbf{e}_{ij}$$

비선형 변환과 로그 연산이 얽혀 있어 닫힌 해(closed-form solution)가 존재하지 않으므로, 이 비선형 최소자승 문제는 **Gauss-Newton**이나 **Levenberg-Marquardt** 같은 반복 최적화 기법으로 풀어 나간다.

**매니폴드 위의 Gauss-Newton**: $SE(3)$는 행렬 덧셈에 닫혀 있지 않기 때문에, 일반적인 벡터 공간에서처럼 $\mathbf{T}_i + \boldsymbol{\delta}_i$로 상태를 갱신하면 회전 행렬의 직교성($\mathbf{R}^\top \mathbf{R} = \mathbf{I}$)이 깨진다. 대신 Lie algebra 위에서 6차원 미소 증분(increment) $\boldsymbol{\delta}_i \in \mathbb{R}^6$을 계산한 다음, 지수 사상(exponential map) $\text{Exp}$로 다시 $SE(3)$ 원소로 올려 기존 포즈에 곱해 준다.

$$\mathbf{T}_i \leftarrow \mathbf{T}_i \cdot \text{Exp}(\boldsymbol{\delta}_i)$$

한 번의 Gauss-Newton 반복(iteration)은 다음 다섯 단계로 진행된다.

1. 현재 포즈 추정치에서 모든 엣지의 오차 벡터 $\mathbf{e}_{ij}$와 각 노드의 증분에 대한 야코비안(Jacobian) $\mathbf{J}_{ij}$를 계산한다.
2. 선형화된 정규 방정식(normal equation) $\mathbf{H} \boldsymbol{\delta} = -\mathbf{b}$를 조립한다. 이때 헤시안 근사 행렬 $\mathbf{H}$와 우변 벡터 $\mathbf{b}$는 다음과 같이 모든 엣지의 기여를 합산해 만든다.

$$\mathbf{H} = \sum_{(i,j)} \mathbf{J}_{ij}^\top \boldsymbol{\Omega}_{ij} \mathbf{J}_{ij}, \quad \mathbf{b} = \sum_{(i,j)} \mathbf{J}_{ij}^\top \boldsymbol{\Omega}_{ij} \mathbf{e}_{ij}$$

3. 그래프의 각 포즈는 보통 극소수의 이웃 포즈와만 엣지로 연결되므로 $\mathbf{H}$는 대부분의 원소가 0인 희소(sparse) 행렬이 되며, 이를 sparse Cholesky 분해로 빠르게 풀 수 있다. 다만 여기서 한 가지 중요한 수치적 조건을 챙겨야 한다. 그래프가 순수하게 상대 제약만으로 이루어져 있으면, 전체 포즈를 통째로 같은 $SE(3)$ 강체 변환으로 평행이동하거나 회전시켜도 모든 상대 오차가 변하지 않는다. 이 게이지 자유도(gauge freedom) 때문에 $\mathbf{H}$에는 6차원 영공간(null space)이 생겨 역행렬이 존재하지 않는다. 따라서 첫 번째 기준 포즈를 고정하거나 절대 좌표에 대한 prior factor를 하나 추가해야 $\mathbf{H}$가 양의 정부호(positive definite)가 되어 Cholesky 분해를 안정적으로 수행할 수 있다.
4. 풀어서 얻은 증분 벡터를 각 노드에 적용한다: $\mathbf{T}_i \leftarrow \mathbf{T}_i \cdot \text{Exp}(\boldsymbol{\delta}_i)$.
5. 오차 감소량이나 증분 크기가 수렴 기준 아래로 떨어질 때까지 1~4단계를 반복한다.

실제 구현에서는 매번 희소 행렬 구조와 매니폴드 미분을 밑바닥부터 코딩하는 대신 범용 그래프 최적화 라이브러리를 활용한다. 대표적인 프레임워크인 **[g2o (General Graph Optimization)](https://doi.org/10.1109/ICRA.2011.5979949)**(Kümmerle et al., ICRA 2011)는 사용자가 상태 변수에 해당하는 vertex(노드) 타입과 오차 모델인 edge(제약) 타입만 정의하면, 희소 정규 방정식의 조립과 Cholesky·Levenberg-Marquardt 풀이를 자동으로 처리해 준다. ORB-SLAM3 역시 로컬 번들 조정과 전역 포즈 그래프 최적화를 모두 g2o 위에서 수행한다.

아래 코드는 포즈 그래프 최적화의 수학적 핵심인 $SE(3)$ 행렬의 $\text{Log}$·$\text{Exp}$ 사상과 두 포즈 사이의 6차원 오차, 그리고 전체 그래프의 비용 함수를 순수 NumPy와 SciPy만으로 직접 구현한 예시다.

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

실제 SLAM 시스템에는 프런트엔드의 검증을 뚫고 들어온 오검출 루프나 센서 오작동, 시야를 스쳐 간 동적 물체 때문에 이상치(outlier) 측정이 섞여 들 수 있다. 일반적인 최소자승 비용 함수 $\rho(s) = s$는 오차의 제곱에 비례해 페널티가 끝없이 커지므로, 단 하나의 큰 이상치만 섞여도 그 오차가 전체 비용을 독식하며 멀쩡한 궤적까지 자기 쪽으로 끌어당긴다.

이 문제를 풀기 위해 사용하는 **Robust kernel**(M-estimator)은 잔차가 일정 수준을 넘어가면 비용의 증가 폭을 둔화시키거나 아예 포화시켜, 이상치가 최적화 해를 흔들지 못하도록 영향력을 제한한다. 널리 쓰이는 대표적인 커널의 수식과 성질은 다음 표와 같다.

| Kernel | $\rho(s)$ ($s = e^2$) | 특성 |
|--------|----------------------|------|
| Least Squares | $s$ | 이상치에 취약 |
| Huber | $\begin{cases} s & \text{if } \sqrt{s} \leq k \\ 2k\sqrt{s} - k^2 & \text{otherwise} \end{cases}$ | 임계값 $k$ 이상에서 선형 |
| Cauchy | $k^2 \log(1 + s/k^2)$ | 부드러운 감쇠 |
| Geman-McClure | $\frac{s}{k^2 + s}$ | 강한 이상치 억제 |

표에서 보듯 Huber 커널은 임계값 $k$ 이하의 정상 오차에는 제곱 비용을 유지하면서도 그 바깥에서는 선형으로만 커져 기울기(영향력)를 상수로 제한하고, Cauchy와 Geman-McClure 커널은 잔차가 커질수록 기울기 자체를 0에 가깝게 떨어뜨려 극단적인 이상치를 사실상 무시한다. 이러한 robust kernel을 포즈 그래프 목적식에 씌우면 비용 함수는 다음과 같이 바뀐다.

$$\sum_{(i,j)} \rho\left(\mathbf{e}_{ij}^\top \boldsymbol{\Omega}_{ij} \mathbf{e}_{ij}\right)$$

비선형 커널 $\rho$가 들어간 이 목적식은 **IRLS(Iteratively Reweighted Least Squares)** 기법으로 푼다. 핵심 아이디어는 단순하다. 수렴할 때까지 반복(iteration)마다 현재 잔차의 크기 $s_i = \mathbf{e}_i^\top \boldsymbol{\Omega}_i \mathbf{e}_i$를 평가해 커널의 도함수로 엣지별 가중치 $w_i$를 다시 계산한 뒤, 이 가중치를 적용한 표준 가중 최소자승 문제를 한 번씩 푼다.

$$w_i = \rho'(s_i), \quad s_i = \mathbf{e}_i^\top \boldsymbol{\Omega}_i \mathbf{e}_i$$

여기서 $\rho'$은 위 표에 제시된 $\rho(s)$를 $s$로 미분한 함수다. 결과적으로 정상적인 오차 범위 안에 있는 엣지는 1에 가까운 가중치를 유지하는 반면, 잔차가 큰 의심 엣지는 가중치가 급격히 줄어들어 최적화에서 자연스럽게 밀려난다.

한 걸음 더 나아가, 루프 클로저 엣지 자체의 채택 여부를 최적화 변수로 편입하는 방법들도 개발되었다.

**[Switchable constraints](https://doi.org/10.1109/IROS.2012.6385590)**(Sünderhauf & Protzel, IROS 2012)에서는 각 루프 클로저 팩터마다 연속적인 스위치 변수 $s_{ij} \in [0, 1]$을 도입해, 주변의 다른 제약들과 모순되는 루프는 최적화기가 스스로 연결을 끊도록($s_{ij} \to 0$) 설계했다.

$$\rho_{\text{switch}}(\mathbf{e}_{ij}, s_{ij}) = s_{ij}^2 \mathbf{e}_{ij}^\top \boldsymbol{\Omega}_{ij} \mathbf{e}_{ij} + \lambda (1 - s_{ij})^2$$

첫째 항은 스위치가 켜진 정도에 비례해 루프 오차를 비용에 반영하고, 둘째 항은 스위치를 끌수록 벌점 $\lambda(1 - s_{ij})^2$을 부과한다. 덕분에 최적화기는 오차가 벌점보다도 훨씬 커서 다른 제약과 도저히 양립할 수 없는 오검출 루프만 선택적으로 꺼 버린다.

**[DCS](https://doi.org/10.1109/ICRA.2013.6630557)**(Dynamic Covariance Scaling, Agarwal et al., ICRA 2013)는 여기서 한발 더 나아가, 스위치 변수를 별도로 추가해 상태 공간을 키우는 대신 스위치의 최적값을 닫힌 형태로 유도해 각 루프의 공분산을 잔차 크기에 따라 동적으로 키워 준다. 잔차가 큰 루프일수록 공분산이 커지고(정보 행렬이 작아지고) 영향력이 줄어드는 효과는 같으면서도, 기존 포즈 그래프 변수 차원을 그대로 유지해 계산량을 절약한다.

아래 코드는 Huber와 Cauchy 커널의 비용 계산 및 IRLS 가중치 산출, 그리고 이를 적용한 robust 포즈 그래프 비용 함수를 간단히 구현한 예시다.

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

지금까지 살펴본 batch 방식의 포즈 그래프 최적화는 새 키프레임이 들어올 때마다 그래프 전체의 정규 방정식을 처음부터 다시 세우고 풀기 때문에, 주행 시간이 길어져 그래프가 커질수록 연산량이 불어나 실시간성을 지키기 어렵다. [iSAM2 (Kaess et al., 2012)](https://doi.org/10.1177/0278364911430419)는 이러한 병목을 해결하기 위한 알고리즘으로, 새로운 변수나 측정 팩터가 추가될 때 그래프에서 실제로 영향을 받는 일부분만 추려 증분적으로(incrementally) 다시 계산한다(§4.5.6).

이 증분 갱신을 뒷받침하는 핵심 자료구조가 바로 **Bayes tree**다. 이는 그래프 이론의 clique tree와 닮았지만 부모-자식 간 방향성을 지닌 트리 구조다. 먼저 MAP 추정에 사용하는 전체 상태의 사후 확률 분포는 팩터 그래프(factor graph) 위에서 개별 측정 팩터들의 곱으로 인수분해된다.

$$p(\mathbf{x} | \mathbf{z}) \propto \prod_k f_k(\mathbf{x}_k)$$

여기서 $f_k$는 오도메트리나 루프 클로저 같은 개별 팩터이고, $\mathbf{x}_k$는 해당 팩터가 연결하는 포즈 변수들의 부분집합이다. 선형 대수에서 가우스 소거법이나 QR 분해를 하듯 이 그래프에서 변수를 정해진 순서대로 하나씩 소거해 나가면, 사후 분포는 방향성이 있는 조건부 분포들의 곱, 즉 **Bayes net** 형태로 변환된다.

$$p(\mathbf{x} | \mathbf{z}) = \prod_i p(x_i | \text{Sep}(x_i))$$

이 식에서 $\text{Sep}(x_i)$는 변수 $x_i$의 분리자(separator)로, $x_i$를 소거하는 바로 그 순간에 팩터로 직접 연결되어 있던 나머지 변수들을 가리킨다. Bayes tree는 여기서 더 나아가 조밀하게 얽힌 조건부 분포들을 클리크(clique) 단위의 트리로 묶어 낸다. 각 클리크 $C_k$는 자기 자신이 담당하는 frontal 변수 집합 $F_k$와 부모 클리크와 공유하는 separator 집합 $S_k$로 나뉘며, 전체 사후 분포는 결국 클리크별 조건부 확률의 곱으로 깔끔하게 정리된다([Kaess et al., 2012](https://doi.org/10.1177/0278364911430419), 식 (13)).

$$p(\mathbf{x} | \mathbf{z}) = \prod_k p(F_k | S_k)$$

**증분 갱신(Incremental update)**의 효율성은 바로 이 트리 구조에서 나온다. 새로운 팩터가 들어오면 Bayes tree 전체를 갈아엎지 않는다. 대신 해당 팩터의 변수들이 속한 클리크부터 트리의 루트(root)까지 올라가는 경로에 있는 클리크들만 떼어 내 부분적으로 재분해하고, 나머지 트리 가지는 기존 계산 결과를 그대로 보존한다. 변수 소거 순서 역시 다시 분해해야 하는 윗부분에 한정해 새로 잡는다. 또한 비선형 문제의 선형화 오차를 관리하기 위해 **fluid relinearization**을 적용하는데, 현재 추정치가 기존 선형화 지점에서 미리 정한 임계값 이상으로 벗어난 변수만 선별해 다시 선형화한다. 다만 먼 과거와 현재를 단숨에 잇는 커다란 루프 클로저가 발생하면, 그 영향이 트리의 루트 근처와 중간 경로 전반에 걸쳐 퍼지므로 재선형화·재분해되는 범위도 자연스레 넓어진다.

다음 `SimpleIncrementalOptimizer` 클래스는 실제 Bayes tree 구현을 단순화해, 새 엣지나 루프 클로저가 추가될 때 영향을 받는 변수(`affected`)를 추적하고 관련 영역만 선별해 재최적화하는 iSAM2의 핵심 개념을 직관적으로 보여 준다.

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

루프 클로저가 현재 이어지고 있는 궤적의 끝을 과거 궤적과 연결해 드리프트를 줄이는 작업이라면, **전역 재측위(global relocalization)**는 로봇이 미리 만들어진 지도(prior map) 위에서 사전 정보 없이 자신의 절대 위치를 단번에 찾아내는 문제다. 가려짐이나 급격한 움직임으로 오도메트리 추적(tracking)을 완전히 잃어버렸을 때, 혹은 전원을 켜고 임의의 지점에서 출발할 때(일명 '납치된 로봇 문제', kidnapped robot problem) 반드시 넘어야 하는 관문이다.

### 10.3.1 Map-Based Localization

사전에 구축해 둔 정밀 지도가 있다면, 현재 들어오는 센서 관측을 그 지도 데이터와 직접 정합해 현재 포즈를 복원할 수 있다. 센서의 특성에 따라 시각 기반과 LiDAR 기반 파이프라인이 다르게 구성된다.

**Visual relocalization**: 현재 카메라 이미지에서 특징점을 추출한 뒤, visual word 색인이나 디스크립터 직접 매칭으로 지도에 저장된 3D 맵 포인트와의 2D-3D 대응쌍을 확보한다. 그런 다음 PnP와 RANSAC을 결합해 이상치를 걸러 내며 카메라의 6자유도 포즈를 추정하고, 복원된 포즈를 초기값 삼아 정상적인 추적을 재개한다.

예컨대 ORB-SLAM3는 DBoW2로 유사한 후보 키프레임을 검색하고 ORB 특징점 매칭으로 2D-3D 대응을 만든 뒤, MLPnP + RANSAC으로 초기 포즈를 구한다. ORB-SLAM2 시절까지는 보정된 핀홀(calibrated pinhole) 카메라 모델에 의존하는 EPnP를 썼으나, ORB-SLAM3에서는 어안(fisheye) 렌즈 등 다양한 카메라 모델에 두루 적용할 수 있는 MLPnP로 교체했다. 초기 포즈를 얻은 뒤에는 그 위치에서 지도 포인트를 다시 투영해 주변 대응점을 추가로 긁어모으는 guided search로 정밀도를 끌어올린다.

**LiDAR relocalization**: 현재 LiDAR 스캔을 미리 구축한 3D 포인트 클라우드 맵에 정합한다. 처음부터 거대한 지도 전체에 ICP를 돌리면 지역 최솟값에 빠지기 쉬우므로, 먼저 Scan Context나 PointNetVLAD 같은 전역 디스크립터로 현재 스캔과 닮은 지도 영역으로 범위를 좁혀 놓는다. 이어서 FPFH + RANSAC이나 GeoTransformer로 대략적인 초기 정합(coarse registration)을 거친 뒤, 마지막으로 ICP 또는 GICP로 밀리미터·센티미터 수준까지 정밀하게 맞춘다(fine registration).

### 10.3.2 Prior Map + Online Sensor

자율주행 자동차는 미리 정교하게 제작해 둔 HD map(고정밀 지도) 위에 실시간 센서 관측을 올려 자기 위치를 정밀하게 측위하는 방식을 널리 활용한다. 그러나 실험실의 정적인 사전 지도와 달리, 현실 도로에서 사전 지도를 활용하는 데는 세 가지 까다로운 난제가 뒤따른다.

- **맵과 환경의 불일치**: 지도를 제작한 뒤 시간이 흐르면 공사 구간이 생기고, 건물 외관이 바뀌며, 가로수의 나뭇잎이 무성해지거나 떨어지는 등 환경이 달라진다. 실시간 관측이 사전 지도와 어긋나더라도 이를 고장이나 추론 실패로 오해하지 않고 변화한 부분을 견뎌 내야 한다.
- **Cross-modal matching**: 고정밀 HD map은 고가의 측량용 LiDAR로 구축해 두었는데, 정작 양산 차량에는 단가 문제로 카메라만 탑재되어 있을 수 있다. 이처럼 지도를 만든 센서와 현재 위치를 찾는 센서가 다를 때는 각 모달리티의 기하·외관 정보를 서로 연결해야 한다.
- **초기 포즈 부재**: 도심 지하 주차장에서 막 빠져나왔거나 GNSS 신호를 받지 못해 지도상 어디에서 출발하는지 전혀 모르는 상태라면, 좁은 탐색 창(search window)에 기댈 수 없으므로 지도 전체를 대상으로 place recognition을 돌려 시작 위치 가설부터 세워야 한다.

### 10.3.3 Monte Carlo Localization (MCL)

로봇의 위치가 어딘지 하나로 단정하기 어려울 때는 여러 가능성을 확률 분포로 동시에 추적하는 것이 안전하다. **MCL(Monte Carlo Localization)**은 파티클 필터(particle filter)를 이용한 전역 측위 기법으로, 로봇이 있을 법한 후보 포즈들을 수많은 **파티클(particle, 표본)**로 뿌려 두고, 이동 모델과 센서 관측에 따라 각 파티클의 가중치를 갱신하며 정답 위치를 좁혀 나간다. 초기 위치를 뿌린 뒤에는 **예측(Prediction)–갱신(Update)–재표본추출(Resampling)**의 세 단계를 주기적으로 순환한다.

1. **초기화**: 초기 위치 정보가 전혀 없으므로, 지도 내 주행 가능한 빈 공간(free space) 전역에 파티클들을 고르게 흩뿌린다.
2. **Prediction**: 바퀴 오도메트리나 IMU에서 얻은 제어·모션 입력 $u_t$와 노이즈 모델을 적용해 모든 파티클을 다음 시점의 포즈로 이동시킨다.
$$x_t^{[k]} \sim p(x_t | u_t, x_{t-1}^{[k]})$$
3. **Update**: 실제 LiDAR 등에서 들어온 센서 관측 $z_t$와, 각 파티클 $x_t^{[k]}$의 위치에서 지도 $m$을 바탕으로 예측한 관측(예: ray casting 거리)을 비교해 관측 가능도(likelihood)로 가중치를 갱신한다.
$$w_t^{[k]} = p(z_t | x_t^{[k]}, m)$$
4. **Resampling**: 갱신된 가중치에 비례하는 확률로 파티클들을 다시 뽑는다. 실제 관측과 잘 맞아떨어진 파티클은 여러 개로 복제되어 살아남고, 엉뚱한 위치에 있던 파티클은 자연스럽게 도태된다.

MCL의 가장 큰 장점은 봉우리가 여러 개인 **다봉(multi-modal) 확률 분포**를 자연스럽게 표현할 수 있다는 점이다. 로봇이 처음 출발해 여러 장소 중 어디에 있는지 확신할 수 없을 때도 유력한 가설들을 모두 쥐고 가다가, 특징적인 코너나 장애물을 관측해 구별이 가능해지면 정답 주변의 파티클만 남기고 하나의 모드로 수렴한다. 반대로 좌우가 대칭인 긴 복도처럼 어느 쪽에서 보나 센서 관측이 똑같은 환경에서는 다중 모드가 해소되지 않고 여러 후보 군집이 계속 유지된다. 실전 구현에서는 파티클 개수가 너무 적거나 센서 관측 모델의 가능도가 지나치게 뾰족하면, 초기 단계에서 우연히 가중치가 낮게 잡힌 정답 근처 파티클들이 몽땅 제거되면서 엉뚱한 위치로 수렴해 버린다. 이런 **파티클 고갈(particle deprivation)** 현상을 각별히 경계해야 한다.

다음 `MonteCarloLocalization` 클래스는 2D 점유 격자 지도(occupancy grid) 위에서 ray casting 관측 모델, 로그 가중치 정규화, 저분산 재표본추출(low-variance resampling)을 이용해 2D LiDAR 기반 MCL을 구현한 예시다.

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

현장의 매핑 작업이 항상 단일 주행 한 번으로 깔끔하게 끝나지는 않는다. 넓은 건물 전체를 며칠에 걸쳐 구역별로 나누어 스캔하기도 하고, 여러 대의 로봇이 각자 다른 구역을 맡아 동시에 탐사하기도 하며, 주행 도중 추적을 놓친 뒤 다른 위치에서 새 하위 지도를 이어 가야 할 때도 있다. **Multi-session SLAM**은 이처럼 서로 다른 시간·장치·좌표계에서 따로 축적된 부분 지도들을 일관된 하나의 전역 지도로 통합하는 기술이다.

### 10.4.1 Map Anchoring

각 세션이나 로봇은 저마다 자신의 출발점을 원점 $(0, 0, 0)$으로 삼아 지도를 그려 나가므로, 서로 다른 두 지도를 합치려면 두 좌표계(coordinate frame)를 하나의 기준틀에 맞추는 **map anchoring** 과정을 먼저 거쳐야 한다.

**방법 1 — 공유 랜드마크 기반**: 두 세션이 겹치는 구역을 지나며 동일한 3D 랜드마크나 특징점을 공통으로 관측했다면, 대응하는 3D 점 쌍 $(\mathbf{p}_k^A, \mathbf{p}_k^B)$의 유클리드 거리 오차를 최소화해 두 지도 사이의 상대 강체 변환 ${}^{A}\mathbf{T}_{B}$를 직접 추정할 수 있다.

$${}^{A}\mathbf{T}_{B} = \arg\min_{\mathbf{T}} \sum_k \| \mathbf{p}_k^A - \mathbf{T} \cdot \mathbf{p}_k^B \|^2$$

**방법 2 — Place recognition 기반**: 미리 대응 지어진 공유 랜드마크가 없다면, 두 세션의 키프레임 데이터베이스 사이에서 place recognition을 돌려 같은 장소를 찍은 키프레임 쌍을 찾은 뒤, 기하학적 정합으로 초기 좌표계 변환을 구한다.

**방법 3 — GNSS 기반**: 야외 주행처럼 두 세션 모두 위성 항법(GNSS) 위치 측정을 함께 기록했다면, 공통 지구 좌표계를 매개로 두 지도의 원점과 방향을 곧바로 정렬할 수 있다.

### 10.4.2 Inter-Session Loop Closure

Map anchoring으로 두 지도의 대략적인 위치를 맞춰 놓은 다음에는, 서로 다른 세션의 궤적을 촘촘히 엮어 정밀도를 끌어올리는 **inter-session loop closure**가 뒤따른다. 기본 원리는 단일 세션 내의 루프 클로저와 동일하지만, 시간 간격과 좌표계 차이에서 비롯되는 두 가지 난관을 추가로 넘어야 한다.

1. **외관 변화**: 어제 낮과 오늘 밤, 혹은 여름과 겨울처럼 세션 사이에 조명·날씨·계절 변화가 끼어든다. 이런 교차 시점 재인식에서는 고전적인 지역 특징점보다 AnyLoc 같은 파운데이션 모델 기반 디스크립터 쪽이 재현율이 훨씬 높다(§9.2.4).
2. **좌표계 불일치**: 초기 map anchoring의 오차나 각 세션에 쌓인 드리프트 때문에, 같은 장소인데도 두 지도상의 예상 좌표가 꽤 어긋나 있을 수 있다. 따라서 후보 탐색 단계에서는 포즈 사전 정보(pose prior)로 거르는 반경을 넓히고, 검사할 후보 개수와 RANSAC 반복 횟수, 정합 초기값의 허용 편차를 넉넉하게 잡아야 한다. 단, 한 가지 선은 분명히 지켜야 한다. 기하학적 검증에서 인라이어로 받아들이는 **최종 잔차 임계값만큼은 절대 느슨하게 잡아서는 안 된다.** 이 기준까지 완화하면 엉뚱한 오검출 루프가 쏟아져 들어와 합쳐 놓은 지도 전체를 망가뜨리기 때문이다.

### 10.4.3 ORB-SLAM3 Multi-Map System

서로 다른 세션을 잇고 추적 실패에 유연하게 대처하는 대표적 실전 구조가 ORB-SLAM3의 **Atlas** 멀티맵 시스템이다. Atlas는 여러 개의 비활성(inactive) 지도와 현재 추적 중인 활성(active) 지도 하나를 함께 품고 있으면서 다음과 같은 생애주기로 동작한다.

1. **Active map**: 현재 카메라가 실시간으로 위치를 추적하며 새로운 키프레임과 3D 맵 포인트를 계속 추가해 나가는 작업용 지도다.
2. **Map creation**: 시야가 가려지거나 빠른 움직임으로 추적이 끊겨 재측위조차 실패하면, 현재까지 만든 지도를 버리지 않고 Atlas 보관함에 비활성 지도로 저장한 뒤 그 시점부터 새로운 active map을 생성해 추적을 이어 간다.
3. **Map merging**: 탐사를 계속하다가 place recognition 모듈이 현재 active map의 키프레임과 같은 장소를 Atlas에 보관된 비활성 지도 중 하나에서 포착하면 두 지도의 병합 절차에 들어간다.
   - 먼저, 매칭된 키프레임 쌍으로부터 두 지도 사이의 상대 변환 $\mathbf{T}_{merge}$를 정밀하게 추정한다.
   - 두 키프레임과 각각의 공가시성(covisible) 이웃 키프레임들, 그리고 이들이 함께 바라보는 맵 포인트들을 모아 **welding window**를 구성한다. 이때 active map 쪽의 키프레임과 맵 포인트들을 $\mathbf{T}_{merge}$로 변환해 비활성 지도의 좌표계로 정렬한다.
   - 정렬된 두 지도를 새로운 active map으로 합치면서 겹치는 맵 포인트를 하나로 통합한다.
   - 마지막으로 경계 부근의 welding window 안에서 정밀한 국소 번들 조정(welding bundle adjustment)을 수행한 뒤, 그 경계 보정 결과를 포즈 그래프 최적화로 나머지 지도 전체에 퍼뜨린다.
4. **Multi-session 활용**: 이전 주행에서 저장해 둔 지도 파일을 불러와 Atlas에 비활성 지도로 올려 둔 채 새 주행을 시작하면, 로봇이 예전 경로의 어느 한 지점과 마주치는 순간 자동으로 두 지도가 하나로 합쳐진다. 이 과정을 세션을 거듭하며 반복하면 방대한 공간을 점진적으로 이어 붙일 수 있다.

아래 `MultiMapAtlas` 클래스는 ORB-SLAM3 Atlas의 상태 전이 로직을 간략히 모사한 코드다. 추적이 끊기면 새 지도를 만들어 기존 지도를 보관했다가, 훗날 같은 장소를 재방문해 기하 검증을 통과하면 두 지도를 하나의 좌표계로 통합한다. (참고로 코드 내 `_merge_maps`는 설명의 편의를 위해 비활성 지도를 active 지도 쪽으로 옮기도록 단순화했으나, 실제 ORB-SLAM3는 반대로 active 지도 쪽을 비활성 지도 좌표계로 옮겨 붙인다.)

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

단일 로봇이 여러 날에 걸쳐 지도를 이어 붙이는 multi-session과 달리, 여러 대의 로봇이 동시에 흩어져 공간을 탐사하는 **multi-robot map merging** 환경에서는 통신과 분산 처리라는 현실적 제약이 한층 무겁게 작용한다.

- **통신 대역폭**: 무선 통신 대역폭이 빠듯한 현장에서는 고해상도 이미지나 3D 포인트 클라우드를 통째로 주고받을 수 없다. 따라서 루프 후보를 찾을 때는 Scan Context나 압축된 시각 디스크립터처럼 용량이 작은 요약 정보만 교환하고, 기하학적 검증 단계에서도 희소 특징점이나 다운샘플링된 포인트 클라우드 일부만 전송한다. 분산 최적화 단계에 들어가서는 로봇 간 연결에 걸린 경계 포즈의 추정치와 불확실성 정보만 주고받으며 트래픽을 최소화한다.
- **분산 최적화**: 모든 데이터를 중앙 서버로 모아 풀면 단일 장애점과 통신 병목이 생기므로, Kimera-Multi나 Swarm-SLAM 같은 시스템은 중앙 조정자 없이 로봇들끼리 P2P 통신으로 합의를 이뤄 지도를 합친다.
- **상대 포즈 불확실성**: 로봇들이 각기 다른 곳에서 통신 없이 출발했다면 초기 상대 포즈를 전혀 모르는 상태이므로, 현장에서 서로의 경로가 겹치거나 마주칠 때 발생하는 inter-robot loop closure로 비로소 좌표계를 맞춰 나가야 한다.

이러한 분산 포즈 그래프 최적화 문제는 각 로봇 내부의 자기 엣지 $\mathcal{E}_r$와 로봇 사이를 잇는 교차 엣지 $\mathcal{E}_{\text{inter}}$로 비용을 나누어 다음과 같이 정식화할 수 있다.

$$\mathbf{T}^* = \arg\min \sum_{\text{robot } r} \sum_{(i,j) \in \mathcal{E}_r} \rho(\mathbf{e}_{ij}) + \sum_{(i,j) \in \mathcal{E}_{\text{inter}}} \rho(\mathbf{e}_{ij})$$

각 로봇은 자신이 소유한 엣지 집합 $\mathcal{E}_r$을 중심으로 로컬 최적화를 풀면서, 다른 로봇과 얽힌 엣지 $\mathcal{E}_{\text{inter}}$의 양 끝단 포즈 정보만 이웃 로봇과 주고받는다. 이를 분산 환경에서 풀 때는 주로 ADMM이나 Gauss-Seidel 방식의 블록 좌표 하강(block coordinate descent)을 활용한다. 다만 3D 회전이 포함된 포즈 그래프 목적식은 본질적으로 비볼록(non-convex)이기 때문에, 볼록 최적화 문제에서 성립하는 표준 ADMM의 수렴 보장이 여기에 그대로 적용되지는 않는다. [Hong et al.의 연구](https://arxiv.org/abs/1410.1390)에서 보인 비볼록 ADMM의 정류점(stationary point) 수렴 역시 특정 consensus·sharing 문제 구조와 충분히 큰 페널티 계수라는 조건에서 성립하며, 실제 수렴 궤적과 최종 해의 품질은 초기값과 로봇 간 통신 그래프 연결 상태에 따라 민감하게 달라진다. 이에 대한 대안으로 [DC2-PGO](https://arxiv.org/abs/1911.03721)는 semidefinite relaxation을 이용해 분산 PGO를 볼록 완화 문제로 푼 뒤, 측정 잡음이 적정한 수준 이내라면 구한 해가 실제로 전역 최적해인지까지 분산 방식으로 사후 검증하는 기법을 제시했다(T-RO).

---

## 10.5 2022–2024년 연구

루프 클로저 백엔드의 최근 흐름은 대규모 오검출이 쏟아지는 극한 환경에서도 실시간 증분 최적화를 무너지지 않게 지탱하는 방향으로 발전해 왔다.

**[riSAM (McGann et al., 2023)](https://arxiv.org/abs/2209.14359)**은 증분 최적화기인 iSAM2에 점진적 비볼록 완화 기법인 [Graduated Non-Convexity](https://arxiv.org/abs/1909.08605)(GNC)를 결합한 강건 증분 백엔드다(ICRA 2023). 연구진은 MIT CSAIL과 Intel Research Lab 데이터셋에 전체 루프 클로저의 10%에서 무려 90%까지 오검출을 섞어 넣는 가혹한 실험을 했으며, 이 조건에서 riSAM이 온라인 방식 중 가장 뛰어난 정확도를 기록했고 오프라인 배치 최적화와 대등하거나 더 나은 결과를 냈다고 보고했다. 다만 실험에 쓰이지 않은 다른 이상치 분포나 더 나쁜 초기값 조건에서도 동일한 우위가 유지된다고 일반화하기는 어렵다.

**[Kimera2 (Abate et al., 2024)](https://arxiv.org/abs/2401.06323)**는 기존 Kimera 파이프라인을 전면 개선한 시스템이다(ISER 2023). 포즈 그래프 백엔드인 Kimera-RPGO는 원래 PCM으로 잘못된 루프를 걸러 내는 구조였으나, 여기에 GNC를 선택지로 추가해 이상치 대응력을 높였고, 프런트엔드인 Kimera-VIO의 특징점 추적과 키프레임 선정 전략도 손봤다. 드론, 사족 보행 로봇, 바퀴형 로봇, 시뮬레이션 자율주행차까지 다양한 플랫폼에서 실전 검증을 거쳤다.

**[Group-k Consistent Measurement Set Maximization (Forsgren et al., 2022)](https://arxiv.org/abs/2209.02658)**은 PCM의 근본적인 한계를 짚고 넘어간다. 두 개의 측정쌍만 비교해서는 서로 기하학적으로 모순되는지 판별하기 모호한 경우가 적지 않기 때문에, 저자들은 PCM의 쌍별 일관성(pairwise consistency) 검사를 $k$개 측정 묶음 단위의 일관성(group-$k$ consistency)으로 일반화하고 시뮬레이션 데이터에서 기존 PCM과 비교해 강점을 분석했다.

---

지금까지 살펴본 루프 클로저와 전역 최적화는 "로봇이 지나온 궤적을 어떻게 뒤틀림 없이 하나로 맞물리게 할 것인가"에 대한 해답이었다. 이제 일관되게 정렬된 포즈 궤적을 얻었으니, 다음 장에서는 그 궤적을 따라 수집한 센서 관측을 어떤 3D 공간 표현(spatial representation)으로 만들어 경로 계획·렌더링·의미 추론에 활용할 것인지 살펴보자(→ 11장).

