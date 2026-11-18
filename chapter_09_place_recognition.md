# 9장 장소 인식

카메라·LiDAR·IMU를 긴밀히 엮은 융합 오도메트리라도(→ 8장) GNSS 같은 절대 위치 기준이 닿지 않는 구간을 오래 달리면 작은 추정 오차가 차곡차곡 드리프트로 쌓인다. 이 누적 오차를 바로잡으려면 로봇이 전에 지나간 장소를 스스로 다시 알아내야 한다. Place Recognition(장소 인식)은 바로 그 재방문 후보를 찾아내는 단계이며, 여기서 건진 후보는 기하학적 검증을 거쳐 Loop Closure(→ 10장)의 제약 조건으로 이어진다.

> 로봇이 "이 장소를 전에 본 적이 있는가?"를 판단하는 문제다.
> Loop closure 후보를 만들고, multi-session SLAM(여러 번 나눠 만든 지도를 잇는 SLAM)과 재위치 추정(relocalization)에도 쓴다.

---

## 9.1 문제 정의

### 9.1.1 Place Recognition vs Loop Closure Detection

**Place Recognition(PR)**은 "현재 센서 관측이 데이터베이스에 저장된 어떤 장소와 일치하는가?"를 묻는 검색(retrieval) 문제다. SLAM 그래프나 궤적 추정이 없더라도 독립된 문제로 정의하고 풀 수 있다.

반면 **Loop Closure Detection**은 로봇이 이전에 지나갔던 곳에 실제로 되돌아왔는지를 SLAM 파이프라인 안에서 판별하는 과정이다. Place recognition이 외관이나 기하 분포가 닮은 후보 프레임을 추려 주면, loop closure 파이프라인은 이 후보들을 상대로 기하학적 검증(geometric verification)을 통과한 매칭만 골라 포즈 그래프에 얹는다.

```
Loop Closure Detection Pipeline:
┌─────────────┐    ┌────────────────┐    ┌───────────────────┐    ┌─────────────┐
│  현재 관측   │───→│ Place          │───→│ Geometric         │───→│ Pose Graph  │
│  (query)     │    │ Recognition    │    │ Verification      │    │ Update      │
│              │    │ (후보 검색)     │    │ (기하학적 검증)    │    │ (최적화)     │
└─────────────┘    └────────────────┘    └───────────────────┘    └─────────────┘
```

이때 탐색 비용도 함께 따져야 한다. 현재 query의 global descriptor를 과거 프레임 $N$개와 일일이 비교(exact search)하면 query 한 번에 $O(N)$, 전체 시퀀스를 모두 대조하면 $O(N^2)$ 연산이 든다. 실시간 SLAM에서 궤적이 길어져도 멈추지 않으려면 inverted index나 tree, 혹은 approximate-nearest-neighbor(ANN) 인덱스를 써서 평균 query 비용을 낮춰야 한다. 실제로 속도가 얼마나 줄어드는지는 디스크립터 분포와 시스템이 감수할 수 있는 recall 손실 폭에 달려 있다.

### 9.1.2 Retrieval Pipeline

대부분의 장소 인식 시스템은 정보 검색(Information Retrieval)의 표준 흐름을 그대로 따른다. 먼저 고차원의 센서 관측을 고정 길이 벡터 하나로 압축한 뒤 색인에 얹고, 질의가 들어오면 거리가 가장 가까운 후보를 꺼내 기하 검증으로 넘기는 네 단계다.

1. **Encoding**: 각 관측(이미지, 포인트 클라우드, 또는 둘 다)을 관측 전체를 요약하는 고정 길이 글로벌 디스크립터(global descriptor)로 변환한다.
2. **Indexing**: 데이터베이스에 쌓인 모든 디스크립터를 kd-tree나 FAISS 같은 빠른 검색 인덱스 구조에 등록한다.
3. **Retrieval**: 쿼리 디스크립터와 거리상 가장 가까운 데이터베이스 디스크립터를 상위 $k$개까지 찾는다.
4. **Re-ranking & Verification**: 추려진 후보에 기하학적 검증을 적용해 우연히 겉모습만 닮은 오탐을 걸러 내고 최종 매칭을 확정한다.

검색 단계의 핵심 연산은 데이터베이스 $\mathcal{D}$ 안에서 쿼리와 가장 가까운 디스크립터를 고르는 최단 거리 탐색이다.

$$
q^* = \arg\min_{q \in \mathcal{D}} d(\mathbf{f}(\text{query}), \mathbf{f}(q))
$$

여기서 $\mathbf{f}(\cdot)$는 센서 관측을 글로벌 디스크립터 벡터로 매핑하는 인코더 함수, $d(\cdot, \cdot)$는 벡터 사이의 거리 함수(주로 L2 거리 또는 코사인 거리)를 뜻한다.

### 9.1.3 평가 메트릭

장소 인식 알고리즘의 우열을 가릴 때는 검색 정확도와 오탐 비용을 함께 재야 하므로 다음 세 가지 기준을 주로 본다.

**Recall@N**: 디스크립터 거리가 가까운 상위 $N$개 후보 안에 실제 같은 장소(정답)가 하나라도 들어 있는 쿼리의 비율이다. NetVLAD, AnyLoc, SALAD 같은 VPR 논문들이 공통으로 보고하는 대표 지표다.

$$
\text{Recall@N} = \frac{|\{q : \text{top-}N \text{ 후보 중 정답이 있는 쿼리 } q\}|}{|\text{전체 쿼리}|}
$$

이 중 **Recall@1**은 첫 번째로 꺼낸 후보가 곧바로 정답인 비율이다. 다만 실제 SLAM에서는 지연시간과 오탐(false positive) 처리 비용이 감당 가능한 선에서 top-$k$ 후보를 한꺼번에 기하 검증으로 넘기므로, Recall@1 하나만 보지 않고 Recall@$k$, precision-recall 곡선, 그리고 query latency를 나란히 확인해야 한다.

**Precision-Recall Curve**: 디스크립터 유사도 임계값을 바꿔 가며 precision과 recall의 트레이드오프를 그린 곡선이다. 일반 이미지 검색과 달리 SLAM에서는 잘못 받아들인 loop closure 단 하나가 포즈 그래프 전체를 크게 뒤틀어 버리므로, recall을 다소 양보하더라도 높은 precision을 유지하는 임계점 선택이 중요하다.

**정답(Ground Truth)의 정의**: 두 프레임이 "같은 장소"인지 가르는 기준 역시 고정돼 있지 않고, 데이터셋의 annotation protocol이 위치 거리·시야 중첩·자세 차이 등을 바탕으로 정한다. 일부 야외 VPR 벤치마크가 GPS 반경 25m를 정답 임계값으로 쓰지만, 실내나 좁은 골목처럼 스케일이 다른 데이터셋에 이 값을 그대로 가져다 쓰면 판정이 어긋난다. 아래 코드에 적은 `threshold_m=25.0` 역시 동작을 보이기 위한 예시값이다.

```python
import numpy as np
from scipy.spatial.distance import cdist

def compute_recall_at_n(query_descriptors, db_descriptors, 
                         query_positions, db_positions,
                         n_values=[1, 5, 10], threshold_m=25.0):
    """
    Recall@N 계산.
    
    query_descriptors: (Q, D) — 쿼리 디스크립터
    db_descriptors: (M, D) — 데이터베이스 디스크립터
    query_positions: (Q, 2 or 3) — 쿼리의 GPS/GT 위치
    db_positions: (M, 2 or 3) — 데이터베이스의 GPS/GT 위치
    n_values: Recall@N에서 N 값들
    threshold_m: 같은 장소로 판정하는 거리 임계값 (미터)
    """
    # 디스크립터 유사도 행렬 (L2 거리)
    desc_dists = cdist(query_descriptors, db_descriptors, metric='euclidean')
    
    # 실제 거리 행렬
    geo_dists = cdist(query_positions, db_positions, metric='euclidean')
    
    recalls = {}
    for n in n_values:
        correct = 0
        for q in range(len(query_descriptors)):
            # 디스크립터 거리 기준 top-N 인덱스
            top_n_indices = np.argsort(desc_dists[q])[:n]
            
            # top-N 중 하나라도 실제 거리가 threshold 이내이면 정답
            min_geo_dist = geo_dists[q, top_n_indices].min()
            if min_geo_dist < threshold_m:
                correct += 1
        
        recalls[f"Recall@{n}"] = correct / len(query_descriptors)
    
    return recalls
```

위 함수를 돌리면 각 $N$ 값에 대한 Recall@N이 딕셔너리로 계산돼 나오며, $N$을 키울수록 상위 후보 폭이 넓어져 값이 올라가는 양상을 바로 확인할 수 있다. 실제 시스템에서는 이 검색 파이프라인을 카메라 영상과 LiDAR 스캔의 물리적 특성에 맞게 구현한다.

---

## 9.2 Visual Place Recognition (VPR)

Visual Place Recognition(VPR)은 카메라 이미지 한 장 또는 연속된 이미지 시퀀스만으로 장소를 알아보는 문제다. 초기의 지역 특징 빈도 히스토그램(BoW)에서 출발해, 클러스터 잔차를 모으는 VLAD와 미분 가능한 NetVLAD를 거쳐, 최근에는 대규모 사전학습 vision foundation model을 활용하는 흐름으로 발전해 왔다.

### 9.2.1 전통적 방법: Bag of Words (BoW)

초기 VPR의 돌파구는 텍스트 검색(text retrieval) 기법을 이미지 세계로 옮겨 온 **[Video Google (Sivic & Zisserman, 2003)](https://www.robots.ox.ac.uk/~vgg/publications/2003/Sivic03/)**의 Bag of Visual Words 모델이었다. 문서를 단어 등장 빈도로 요약하듯, 이미지를 "시각 단어"의 빈도 벡터로 바꾸어 빠르게 비교하겠다는 발상이다. 전체 파이프라인은 다섯 단계로 돌아간다.

1. **시각 어휘 구축(Visual Vocabulary Construction)**: 대량의 이미지에서 로컬 특징(SIFT, ORB 등)을 뽑은 뒤, 디스크립터 공간에서 k-means 클러스터링으로 $K$개 군집을 만든다. 각 클러스터 중심이 하나의 "시각 단어(visual word)" 역할을 한다.
2. **이미지 표현**: 새로 들어온 이미지의 로컬 특징마다 가장 가까운 시각 단어 하나를 배정(hard assignment)하고, 단어별 등장 횟수를 세어 $K$차원 히스토그램으로 요약한다.
3. **TF-IDF 가중**: 단순 빈도만 쓰면 어느 풍경에나 흔히 나타나는 나뭇잎이나 아스팔트 질감에 점수가 쏠린다. 이를 막기 위해 텍스트 검색의 TF-IDF(Term Frequency – Inverse Document Frequency) 가중치를 곱한다.

$$
w_{i,d} = \underbrace{\frac{n_{i,d}}{n_d}}_{\text{TF}} \cdot \underbrace{\log \frac{N}{N_i}}_{\text{IDF}}
$$

여기서 $n_{i,d}$는 이미지 $d$에서 시각 단어 $i$가 등장한 횟수, $n_d$는 이미지 $d$의 전체 시각 단어 수, $N$은 데이터베이스의 전체 이미지 수, $N_i$는 시각 단어 $i$를 포함하는 이미지 수다. TF 항은 현재 이미지 안에서 자주 등장하는 단어에 힘을 실어 주고, IDF 항은 데이터베이스 전체에서 드문 단어일수록 장소를 가려내는 변별력이 높다고 보고 가중치를 높인다.

4. **역색인(Inverted Index)**: 시각 단어마다 그 단어가 등장하는 이미지 ID 목록을 미리 정리해 둔다. 질의 이미지가 들어오면 데이터베이스 전체를 훑지 않고, 질의에 포함된 시각 단어를 하나라도 공유하는 후보 이미지만 골라 살핀다.
5. **유사도 랭킹**: 추려낸 후보들과 질의 이미지의 TF-IDF 벡터 사이 코사인 유사도를 계산해 순위를 매긴다.

$$
\text{sim}(\mathbf{v}_q, \mathbf{v}_d) = \frac{\mathbf{v}_q \cdot \mathbf{v}_d}{\|\mathbf{v}_q\| \cdot \|\mathbf{v}_d\|}
$$

실제 SLAM 시스템에서 널리 쓰이는 구현이 **DBoW2**다. DBoW2는 큰 vocabulary를 빠르게 탐색하기 위해 k-means를 계층적으로 쌓은 tree(hierarchical k-means tree)를 사용하며, ORB-SLAM 시리즈와 VINS-Mono 모두 loop 후보를 찾는 데 이 모듈을 채택했다. 아래 `SimpleBoW` 클래스는 이 가운데 flat vocabulary와 역색인, 동적 IDF 갱신이 어떻게 맞물려 동작하는지 핵심만 간추려 보여 준다.

```python
import numpy as np
from collections import defaultdict

class SimpleBoW:
    """
    Bag of Visual Words의 간소화 구현.
    """
    def __init__(self, vocabulary):
        """
        vocabulary: (K, D) — K개 시각 단어의 기술자 (k-means 중심)
        """
        self.vocabulary = vocabulary  # (K, D)
        self.K = len(vocabulary)
        self.inverted_index = defaultdict(list)  # word_id -> [(img_id, tf)]
        self.idf = np.ones(self.K)
        self.N = 0  # 데이터베이스 이미지 수
        self.db_words = {}  # img_id -> word_ids. TF-IDF 벡터는 질의 시점의 IDF로 다시 계산한다
    
    def quantize(self, descriptors):
        """로컬 기술자를 가장 가까운 시각 단어로 양자화."""
        # (N_desc, D) vs (K, D) → (N_desc, K) 거리 행렬
        dists = np.linalg.norm(
            descriptors[:, None, :] - self.vocabulary[None, :, :], axis=2
        )
        word_ids = np.argmin(dists, axis=1)
        return word_ids
    
    def compute_bow_vector(self, descriptors):
        """이미지의 BoW 벡터 (TF-IDF 가중) 계산."""
        return self.bow_from_words(self.quantize(descriptors))
    
    def bow_from_words(self, word_ids):
        """양자화된 단어 목록에서 현재 IDF로 TF-IDF 벡터 계산."""
        
        # Term Frequency
        tf = np.zeros(self.K)
        for w in word_ids:
            tf[w] += 1
        tf /= len(word_ids)  # 정규화
        
        # TF-IDF
        tfidf = tf * self.idf
        
        # L2 정규화
        norm = np.linalg.norm(tfidf)
        if norm > 0:
            tfidf /= norm
        
        return tfidf
    
    def add_to_database(self, img_id, descriptors):
        """이미지를 데이터베이스에 추가."""
        word_ids = self.quantize(descriptors)
        self.db_words[img_id] = word_ids   # 벡터는 저장하지 않고 단어만 보관한다
        self.N += 1
        
        # 역색인 업데이트
        unique_words = np.unique(word_ids)
        for w in unique_words:
            self.inverted_index[w].append(img_id)
        
        # IDF 재계산
        for w in range(self.K):
            n_w = len(set(self.inverted_index[w]))  # 해당 단어가 등장하는 이미지 수
            self.idf[w] = np.log(self.N / (n_w + 1e-6))
    
    def query(self, descriptors, top_k=5):
        """쿼리 이미지와 가장 유사한 데이터베이스 이미지 검색."""
        q_vector = self.compute_bow_vector(descriptors)
        
        # 역색인으로 후보를 좁힌다: 쿼리 단어를 하나라도 공유하는 이미지만
        candidates = set()
        for w in np.unique(self.quantize(descriptors)):
            candidates.update(self.inverted_index[w])
        
        scores = {}
        for img_id in candidates:
            db_vector = self.bow_from_words(self.db_words[img_id])   # 현재 IDF로 계산
            scores[img_id] = np.dot(q_vector, db_vector)
        
        # 유사도 순으로 정렬
        sorted_results = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_results[:top_k]
```

BoW는 속도가 빠르고 구현이 간결하지만, 비슷한 특징들을 하나의 정수 단어 ID로 묶어 버리는 양자화(quantization) 과정에서 원래 디스크립터가 지닌 세밀한 정보가 크게 날아간다. 또한 조명이나 시점 변화에 얼마나 버틸지는 밑단에 쓴 로컬 특징(SIFT, ORB)의 한계에 매이며, vocabulary 크기 $K$ 선택에 따라서도 검색 품질이 민감하게 흔들린다.

### 9.2.2 VLAD와 Fisher Vector

양자화로 잃어버리는 디스크립터 정보를 되살리기 위해, 각 시각 단어가 몇 번 나왔는지만 세는 대신 클러스터 중심과의 **잔차(residual)**를 누적하는 집계 기법이 등장했다.

대표 방식인 **[VLAD (Vector of Locally Aggregated Descriptors, Jégou et al., 2010)](https://doi.org/10.1109/CVPR.2010.5540039)**는 클러스터 $k$에 배정된 특징점들이 중심 $\mathbf{c}_k$로부터 어느 방향으로 얼마나 벗어나 있는지를 모두 더해 기록한다.

$$
\mathbf{V}_k = \sum_{i: \text{NN}(\mathbf{x}_i) = k} (\mathbf{x}_i - \mathbf{c}_k)
$$

이렇게 클러스터마다 모은 $D$차원 잔차 벡터 $\mathbf{V}_k$를 전부 이어 붙이면 최종 VLAD 디스크립터 $\mathbf{V} = [\mathbf{V}_1^T, \mathbf{V}_2^T, \ldots, \mathbf{V}_K^T]^T$가 되며, 총 차원은 $K \times D$다. 단어 빈도가 같더라도 특징점들의 분포 방향이 다르면 잔차 합이 달라지므로 BoW보다 훨씬 풍부한 단서를 보존한다.

이와 짝을 이루는 **Fisher Vector**는 시각 단어를 고정 중심 대신 GMM(Gaussian Mixture Model)으로 모델링한 뒤, 각 가우시안 컴포넌트에 대한 1차·2차 통계량을 Fisher Information Matrix의 제곱근으로 정규화해 담는다. 차원은 $2KD$로 VLAD의 두 배에 이른다. 다만 두 기법 중 어느 쪽 검색 성능이 앞서는지는 평가 지표와 데이터셋, 정규화 설정에 따라 갈린다. 실제로 [Jégou et al. (2010)](https://inria.hal.science/inria-00548637/document)은 VLAD를 Fisher Vector의 단순화 버전으로 해석하면서 Table 1에 평균 성분만 쓴 Fisher Vector와 VLAD를 동일 차원에서 비교한 결과를 실었는데, Holidays mAP 기준 2,048차원에서는 각각 0.497과 0.496으로 사실상 같았고 8,192차원에서는 VLAD(0.526)가 Fisher Vector(0.495)를 앞섰다.

### 9.2.3 NetVLAD: 학습 기반 VPR의 기준점

VLAD가 잔차를 잘 모으기는 했지만, 여전히 사전에 따로 학습한 k-means 중심에 기댄 수작업 파이프라인이었다. **[NetVLAD (Arandjelović et al., 2016)](https://arxiv.org/abs/1511.07247)**는 바로 이 VLAD 집계를 미분 가능한(differentiable) CNN 레이어로 재설계해, 특징 추출부터 집계까지 네트워크 전체를 장소 인식 목적에 맞춰 한 번에(end-to-end) 학습할 수 있게 만들었다.

걸림돌은 기존 VLAD의 hard assignment($\text{NN}(\mathbf{x}_i) = k$이면 1, 아니면 0)가 계단 함수라 미분이 불가능하다는 점이었다. NetVLAD는 이를 부드러운 softmax 기반 **Soft Assignment**로 바꿔 역전파의 물꼬를 텄다.

$$
\bar{a}_k(\mathbf{x}_i) = \frac{e^{\mathbf{w}_k^T \mathbf{x}_i + b_k}}{\sum_{k'} e^{\mathbf{w}_{k'}^T \mathbf{x}_i + b_{k'}}}
$$

원래 VLAD의 유클리드 거리 기반 가중치 $e^{-\alpha\|\mathbf{x}_i - \mathbf{c}_k\|^2}$을 전개하고 공통 항을 약분하면 $\mathbf{w}_k = 2\alpha\mathbf{c}_k$, $b_k = -\alpha\|\mathbf{c}_k\|^2$ 형태의 선형식과 같아진다. NetVLAD는 여기서 $\mathbf{w}_k$, $b_k$와 클러스터 중심 $\mathbf{c}_k$를 독립 파라미터로 풀어 함께 학습함으로써 클러스터 배치가 장소 판별에 유리하도록 스스로 움직이게 만든다.

소프트 가중치가 정해지면 각 클러스터의 잔차 합은 다음과 같이 미분 가능한 행렬 출력으로 계산된다.

$$
\mathbf{V}(j, k) = \sum_{i=1}^{N} \bar{a}_k(\mathbf{x}_i) (x_i(j) - c_k(j))
$$

이 $D \times K$ 행렬을 열마다 정규화한 뒤 1차원 벡터로 펼치고 다시 L2 정규화를 거치면 최종 글로벌 디스크립터가 나온다.

학습 데이터로는 같은 거리를 여러 해에 걸쳐 찍은 Google Street View Time Machine 영상을 썼고, GPS 좌표만 주어지는 현실을 반영해 약한 지도 학습(weakly supervised learning)을 택했다. GPS 위치가 가깝더라도 카메라가 반대편을 보고 있어 전혀 다른 장면이 찍혔을 수 있기 때문이다. 그래서 지리적으로 가까운 영상들을 우선 후보(potential positive)로 묶어 두고 현재 네트워크가 특징 공간에서 가장 가깝다고 판단한 영상을 실제 positive($p^+$)로 삼는 반면, 멀리 떨어진 영상 가운데 겉모습이 헷갈려 오탐을 일으키기 쉬운 hard negative($p^-$)를 뽑아 triplet ranking loss로 거리를 벌린다.

$$
\mathcal{L} = \sum_{(q, p^+, p^-)} \max\left(0, m + d(\mathbf{f}(q), \mathbf{f}(p^+)) - d(\mathbf{f}(q), \mathbf{f}(p^-))\right)
$$

여기서 $m$은 positive와 negative 쌍 사이에 확보할 최소 거리 마진이다. 논문 본문에 따르면 이렇게 학습한 AlexNet+NetVLAD는 Pitts250k-test에서 recall@1 81.0%를 기록해, 학습 없이 붙인 AlexNet+VLAD(55.0%) 대비 상대적으로 47% 높은 정확도를 얻었다(가장 좋은 VGG-16 결과는 그래프로만 보고했다).

### 9.2.4 AnyLoc: Foundation Model 기반 범용 VPR

NetVLAD 이후 VPR 연구는 더 좋은 손실 함수와 백본을 찾아 전용 데이터셋에 맞춰 학습하는 방향으로 흘렀다. 반면 **[AnyLoc (Keetha et al., RA-L 2024)](https://arxiv.org/abs/2308.00688)**은 정반대의 질문을 던졌다. 장소 인식 전용 학습을 전혀 거치지 않고, 대규모 범용 사전학습 모델(foundation model)인 DINOv2의 내부 표현만 꺼내 써도 온갖 환경에서 장소를 알아볼 수 있지 않을까 하는 시도다.

AnyLoc의 절차는 세 단계로 요약된다.

1. **DINOv2 특징 추출**: DINOv2 ViT-G/14 모델의 31번째 레이어에서 attention 모듈의 value facet을 뽑아 밀집 패치 특징(dense patch feature)으로 쓴다. 이미지 전체를 한 벡터로 요약한 CLS 토큰 대신 패치마다 살아 있는 공간 특징을 꺼낸 것이 핵심인데, 저자들은 자신들이 비교한 데이터셋에서 이 dense feature가 평균적으로 더 나았다고 보고했다(어느 레이어와 facet이 최적인지와 성능 격차는 backbone과 데이터셋에 따라 달라진다).
2. **VLAD 집계**: 뽑아낸 패치 특징들을 k-means로 묶어 시각 어휘를 세운 뒤, 별도 학습 없이 hard-assignment VLAD로 글로벌 디스크립터를 모은다.
3. **도메인별 어휘**: 저자들이 패치 특징으로 만든 디스크립터를 PCA로 투영해 보니 Urban, Indoor, Aerial, SubT, Degraded, Underwater처럼 환경 유형별로 또렷한 군집이 갈렸다. 이에 따라 AnyLoc은 단일 어휘 대신 도메인마다 전용 시각 어휘를 따로 구축해 쓴다. 따라서 완전히 새로운 도메인에 투입할 때는 어떤 어휘가 맞는지 다시 확인해야 한다.

기반 모델인 [DINOv2 (Oquab et al., 2023)](https://arxiv.org/abs/2304.07193)는 큐레이션한 이미지 1억 4,200만 장(LVD-142M)으로 자기지도 학습한 비전 인코더다. 장소 인식 라벨을 본 적 없는 이 특징이 다양한 벤치마크에서 놀라울 만큼 잘 버티기는 했지만, 똑같은 건물이 반복되는 구조나 극단적 시점 차이, 학습 분포를 벗어난 특수 센서 도메인에서는 여전히 장소를 헷갈릴 수 있다.

AnyLoc 논문이 자체 선정 벤치마크에서 전용 VPR 모델인 MixVPR / CosPlace와 견주어 보고한 Recall@1 향상 폭은 다음과 같다.
- 주야간 변화: 5% / 11%
- 계절 변화: 9% / 8%
- 오랜 시간을 두고 다시 찍은 영상(논문 예: 1년, 5년 간격): 21% / 28%
- 방향이 달라진 시점: 21% / 30%
- 90°·180°의 극단적 시점 변화: 39% / 49%
- 비정형 환경을 포함한 전체: 기존 방법의 최대 4배

고차원 VLAD 벡터를 실전에 쓰기 좋게 줄일 때도 강건함이 유지돼, 논문은 PCA-Whitening으로 49K 차원을 512차원으로 압축해도 비슷한 수준을 지켰다고 보고했다. 아래 코드는 이 전체 흐름을 개념적으로 옮긴 구현이다. 참고로 코드 안의 `get_intermediate_layers`는 블록의 출력 토큰을 꺼내는 간략화 버전이며, 논문에서 강조한 특정 레이어의 value facet을 직접 추출하려면 PyTorch attention 모듈에 forward hook을 걸어야 한다.

```python
import numpy as np

def anyloc_pipeline(images, dino_model, n_clusters=64, desc_dim=512):
    """
    AnyLoc 파이프라인의 개념적 구현.
    
    1. DINOv2에서 밀집 특징 추출
    2. k-means로 시각 어휘 구축
    3. VLAD 집계
    4. PCA 차원 축소
    """
    # Step 1: 밀집 특징 추출 (ViT 패치 토큰)
    all_patch_features = []
    image_patch_features = []
    
    for img in images:
        # DINOv2 forward: (1, N_patches, D_feat) 형태의 패치 토큰
        patch_tokens = dino_model.get_intermediate_layers(img, n=1)[0]
        # 아래는 마지막 블록의 출력 토큰이다. AnyLoc의 핵심 발견인
        # 특정 레이어의 value facet을 쓰려면 attention 모듈에 hook을 걸어야 한다.
        image_patch_features.append(patch_tokens)
        all_patch_features.append(patch_tokens)
    
    # Step 2: k-means 시각 어휘 구축
    all_features = np.concatenate([p.reshape(-1, p.shape[-1]) for p in all_patch_features], axis=0)  # (N_total_patches, D_feat)
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(all_features)
    centers = kmeans.cluster_centers_  # (K, D_feat)
    
    # Step 3: 각 이미지에 대해 VLAD 디스크립터 계산
    vlad_descriptors = []
    D_feat = centers.shape[1]
    
    for patches in image_patch_features:
        # Hard assignment
        patches = patches.reshape(-1, patches.shape[-1])   # (1, N, D) → (N, D)
        assignments = kmeans.predict(patches)  # (N_patches,)
        
        # VLAD: 각 클러스터에 대한 잔차 합산
        vlad = np.zeros((n_clusters, D_feat))
        for i, patch_feat in enumerate(patches):
            k = assignments[i]
            vlad[k] += patch_feat - centers[k]
        
        # Intra-normalization (각 클러스터 벡터를 개별 정규화)
        for k in range(n_clusters):
            norm = np.linalg.norm(vlad[k])
            if norm > 0:
                vlad[k] /= norm
        
        # Flatten + L2 normalization
        vlad_flat = vlad.flatten()
        vlad_flat /= (np.linalg.norm(vlad_flat) + 1e-10)
        
        vlad_descriptors.append(vlad_flat)
    
    vlad_descriptors = np.array(vlad_descriptors)
    
    # Step 4: PCA 차원 축소
    from sklearn.decomposition import PCA
    pca = PCA(n_components=desc_dim, whiten=True)
    reduced_descriptors = pca.fit_transform(vlad_descriptors)
    
    # 최종 L2 정규화
    norms = np.linalg.norm(reduced_descriptors, axis=1, keepdims=True)
    reduced_descriptors /= (norms + 1e-10)
    
    return reduced_descriptors  # (N_images, desc_dim)
```

### 9.2.5 EigenPlaces

학습 기반 VPR의 또 다른 갈래는 **[CosPlace (Berton et al., 2022)](https://arxiv.org/abs/2204.02287)**와 그 후속작 **[EigenPlaces (Berton et al., 2023)](https://arxiv.org/abs/2308.10832)**다. CosPlace는 대규모 데이터셋에서 비용이 많이 드는 triplet loss와 hard negative mining 대신 장소를 격자 클래스로 묶는 분류(classification) 문제로 학습 방식을 단순화했다. EigenPlaces는 이를 한 단계 더 밀고 나가 시점 변화에 강한 디스크립터를 얻기 위해 학습 클래스를 구성하는 방법 자체를 바꿨다. 클래스마다 카메라 위치 분포에 PCA를 적용해 대략 도로 진행 축에 해당하는 제1 주성분을 찾은 뒤, 그와 수직인 방향(도로 옆 건물 정면 쪽)에 초점(focal point)을 잡아 여러 위치에서 그 점을 함께 바라보는 이미지들을 하나의 학습 클래스로 묶었다. 전방 주행 카메라 데이터셋에 대응하기 위해 제1 주성분 방향을 정면으로 바라보는 frontal 클래스도 함께 둔다.

### 9.2.6 Foundation Model 활용의 확장

AnyLoc 이후에는 DINOv2뿐만 아니라 다양한 비전·멀티모달 foundation model이 VPR의 재료로 쓰이고 있다.

- **CLIP**: 텍스트와 이미지를 같은 임베딩 공간에 짝지어 학습한 모델이라 "도시 거리", "숲 속 길"처럼 자연어 의미 수준에서 장소를 검색하는 시도에 활용할 수 있다. 다만 AnyLoc이 자체 설정과 데이터셋에서 여러 foundation model을 나란히 비교했을 때는 DINOv2 특징이 장소를 더 날카롭게 구별해 냈다.
- **SAM (Segment Anything)**: [SegVLAD (Garg et al., ECCV 2024)](https://arxiv.org/abs/2409.18049)는 SAM으로 영상을 의미 덩어리(segment)로 쪼갠 뒤 이웃 segment를 묶은 SuperSegment마다 DINOv2 특징과 VLAD를 적용해, 이미지 전체가 아니라 영역 단위로 장소를 대조한다.
- **DINOv2 + 학습된 집계**: AnyLoc처럼 사전학습 특징을 학습 없는 VLAD에 그대로 태우는 대신, 특징을 모으는 집계기만 VPR 데이터에 맞게 가볍게 학습하는 갈래도 빠르게 커졌다. 뒤에서 다룰 SALAD(§9.7.4)는 NetVLAD의 할당 문제를 최적 수송(optimal transport)으로 재정식화했고, SelaVPR(Lu et al., ICLR 2024)은 사전학습 모델을 VPR 태스크에 맞게 적응시켜 성능을 끌어올렸다.

### 9.2.7 SeqSLAM과 시퀀스 매칭

아무리 정교한 디스크립터를 쓰더라도 이미지 한 장만 놓고 보면 구별이 불가능한 상황이 남는다. 똑같은 평면이 늘어선 주택가나 복도가 대표적이며, 낮과 밤이 뒤바뀌어 단일 프레임 매칭이 흔들릴 때도 마찬가지다. **[SeqSLAM (Milford & Wyeth, 2012)](https://doi.org/10.1109/ICRA.2012.6224623)**은 이 한계를 한 장의 이미지가 아닌 **연속된 이미지 시퀀스**의 순서로 돌파한다.

한 장면만 보면 헷갈려도 "좌회전 → 공원 → 건널목"처럼 이어지는 순서는 고유하다. SeqSLAM은 먼저 각 이미지를 아주 단순한 저차원 표현(저해상도 패치 정규화 벡터 등)으로 줄인 뒤, 쿼리 시퀀스와 데이터베이스 전체 사이의 쌍별 거리 행렬을 만든다. 그다음 로봇의 주행 속도 변화를 고려한 허용 기울기 범위 안에서 이 거리 행렬을 가로지르는 대각선 경로를 탐색해 누적 거리가 가장 작은 시퀀스 쌍을 골라낸다.

$$
S_{\text{seq}}(q_s, d_s) = \min_{\text{path}} \sum_{i=0}^{L-1} d(\mathbf{f}(q_{s+i}), \mathbf{f}(d_{s+\delta(i)}))
$$

여기서 $L$은 비교할 시퀀스 길이, $\delta(i)$는 두 주행 사이의 속도 차이를 흡수하는 경로 함수다. 낮과 밤, 여름과 겨울이 교차해 개별 프레임의 겉모습이 크게 달라져도 장소들이 스쳐 지나가는 순서 자체는 바뀌지 않기 때문에 강한 내성을 보인다. 이후 SeqNet(Garg & Milford, RA-L 2021)은 이 아이디어를 확장해 시퀀스 자체를 하나의 학습된 디스크립터로 인코딩하는 방식을 제안했다.

---

## 9.3 LiDAR Place Recognition

카메라 기반 장소 인식은 외관 단서가 풍부하지만 빛이 사라지거나 역광이 심하면 금세 흔들린다. 반면 3D LiDAR는 스스로 레이저를 쏘아 주변의 3차원 기하 구조를 직접 재기 때문에 조명 변화에 거의 영향을 받지 않는다. 대신 계절이 바뀌면서 나무와 풀의 부피가 달라지거나, 주행 시점·가림(occlusion) 양상이 크게 바뀔 때 성능이 떨어지는 과제를 안고 있다.

### 9.3.1 Handcrafted 방법: Scan Context

학습 기반 모델 없이 실전 LiDAR SLAM에서 가장 널리 쓰이는 대표 디스크립터는 **[Scan Context (Kim & Kim, 2018)](https://doi.org/10.1109/IROS.2018.8593953)**다. 복잡한 3D 포인트 클라우드를 공간 배치 정보가 살아 있는 2D 행렬 한 장으로 압축해 글로벌 디스크립터로 쓰며, 별도 학습이 필요 없어 여러 LiDAR SLAM 파이프라인에 곧바로 붙는다(참고로 LIO-SAM 공개 구현의 기본 루프 탐지는 반경 검색과 ICP에 기대며, Scan Context를 결합한 버전은 SC-LIO-SAM이라는 별도 저장소로 배포된다).

Scan Context 행렬을 만드는 과정은 다음과 같이 세 단계로 이어진다.

1. **극좌표 분할**: 센서를 원점으로 둔 2D 평면을 방위각(azimuth) 방향 $N_s$개 섹터, 거리(range) 방향 $N_r$개 링으로 쪼개 $N_s \times N_r$ 극좌표 격자를 세운다. 원 논문은 $60 \times 20$ 격자를 썼다.
2. **최대 높이 인코딩**: 각 격자 칸(bin)에 떨어진 3D 점 가운데 가장 높은 점의 $z$ 좌표를 대표값으로 기록한다. 건물 외벽이나 가로수, 기둥처럼 높게 솟은 영구 구조물의 윤곽을 살리기 위해서다.

$$
\mathbf{SC}(i, j) = \max_{p \in \text{bin}(i,j)} z(p)
$$

3. **센서 중심(egocentric) 표현**: 격자를 센서 좌표계 기준으로 자르기 때문에, 같은 위치라도 로봇의 heading(yaw)만 달라지면 행렬의 섹터 축이 위아래로 순환 이동(circular shift)한 모양이 된다. 따라서 두 스캔을 견줄 때는 섹터 축을 한 칸씩 밀어 가며 가장 잘 맞는 회전 정렬을 찾는다. 다만 반대 방향으로 되돌아올 때는 yaw 회전뿐 아니라 차선 폭만큼의 횡방향 이동과 가림 변화가 겹치므로, 원 논문은 이를 흡수하기 위한 root shifting 증강을 별도로 둔다.

데이터베이스가 커졌을 때 모든 Scan Context 행렬을 일일이 돌려 가며 비교하면 탐색이 느려진다. 이를 피하기 위해 검색은 두 단계(Ring Key → Sector Key)로 나뉜다. 먼저 링(열)마다 섹터 방향 값을 평균해 $N_r$차원 벡터 $\mathbf{k}_r = [\bar{h}_1, \bar{h}_2, \ldots, \bar{h}_{N_r}]$(**Ring Key**)를 뽑아 둔다. 섹터 방향을 평균했으므로 로봇의 yaw가 바뀌어도 값이 변하지 않으며, 이 짧은 벡터로 kd-tree를 훑어 유력한 후보만 빠르게 건져 올린다. 다음으로 섹터(행)마다 링 축 값을 모은 **Sector Key**끼리 맞춰 보면 두 스캔 사이의 대략적인 회전 이동량을 가늠할 수 있고, 그 주변 몇 칸만 돌려 보며 아래의 섹터별 코사인 거리를 계산해 최종 순위를 매긴다(아래 식과 예제 코드는 원리를 분명히 보이기 위해 전체 섹터를 순회해 최솟값을 찾는다).

$$
d(\mathbf{SC}_q, \mathbf{SC}_d) = \min_{s \in [0, N_s)} \frac{1}{N_s}\sum_{i=1}^{N_s}\left(1 - \frac{\mathbf{c}^q_i \cdot \mathbf{c}^d_{i+s}}{\|\mathbf{c}^q_i\|\,\|\mathbf{c}^d_{i+s}\|}\right)
$$

```python
import numpy as np

class ScanContext:
    """
    Scan Context 디스크립터 생성 및 매칭.
    """
    def __init__(self, n_sectors=60, n_rings=20, max_range=80.0):
        self.n_sectors = n_sectors
        self.n_rings = n_rings
        self.max_range = max_range
        self.database = []
        self.ring_keys = []
    
    def make_scan_context(self, point_cloud):
        """
        3D 포인트 클라우드 → Scan Context 행렬.
        
        point_cloud: (N, 3) — x, y, z 좌표
        """
        sc = np.zeros((self.n_sectors, self.n_rings))
        
        # 극좌표 변환
        x, y, z = point_cloud[:, 0], point_cloud[:, 1], point_cloud[:, 2]
        ranges = np.sqrt(x**2 + y**2)
        angles = np.arctan2(y, x)  # [-pi, pi]
        angles = (angles + np.pi) / (2 * np.pi)  # [0, 1]로 정규화
        
        # 범위 밖 점 제거
        valid = ranges < self.max_range
        ranges, angles, z = ranges[valid], angles[valid], z[valid]
        
        # 빈 인덱스 계산
        sector_idx = np.clip((angles * self.n_sectors).astype(int), 0, self.n_sectors - 1)
        ring_idx = np.clip((ranges / self.max_range * self.n_rings).astype(int), 0, self.n_rings - 1)
        
        # 각 빈의 최대 높이
        for i in range(len(z)):
            si, ri = sector_idx[i], ring_idx[i]
            sc[si, ri] = max(sc[si, ri], z[i])
        
        return sc
    
    def make_ring_key(self, sc):
        """Ring key: 각 링의 평균 높이."""
        return np.mean(sc, axis=0)  # (n_rings,)
    
    def add_to_database(self, point_cloud):
        """스캔을 데이터베이스에 추가."""
        sc = self.make_scan_context(point_cloud)
        rk = self.make_ring_key(sc)
        self.database.append(sc)
        self.ring_keys.append(rk)
    
    def query(self, point_cloud, top_k=5, n_candidates=20):
        """
        쿼리 스캔과 가장 유사한 데이터베이스 스캔 검색.
        """
        sc_query = self.make_scan_context(point_cloud)
        rk_query = self.make_ring_key(sc_query)
        
        # Stage 1: Ring key로 후보 선정
        rk_dists = [np.linalg.norm(rk_query - rk) for rk in self.ring_keys]
        candidate_indices = np.argsort(rk_dists)[:n_candidates]
        
        # Stage 2: Scan Context 행 이동 매칭
        scores = []
        for idx in candidate_indices:
            sc_db = self.database[idx]
            min_dist = float('inf')
            for shift in range(self.n_sectors):
                sc_shifted = np.roll(sc_db, shift, axis=0)
                # 섹터(행)별 코사인 거리의 평균 — 원 논문의 거리 정의
                num = np.sum(sc_query * sc_shifted, axis=1)
                den = np.linalg.norm(sc_query, axis=1) * np.linalg.norm(sc_shifted, axis=1) + 1e-9
                dist = np.mean(1.0 - num / den)
                min_dist = min(min_dist, dist)
            scores.append((idx, min_dist))
        
        scores.sort(key=lambda x: x[1])
        return scores[:top_k]
```

후속 연구인 **[Scan Context++](https://arxiv.org/abs/2109.13494)**는 디스크립터 표현을 넓혀, roll과 pitch 변화가 크지 않은 조건에서 주행 방향 회전은 물론 옆 차선으로 비켜난 횡방향 이동까지 강건하게 잡아내고, 장소 매칭과 함께 1-DOF 상대 정렬까지 추정해 낸다.

### 9.3.2 M2DP와 ESF

Scan Context 이전에도 수작업 3D 전역 디스크립터가 여러 갈래로 제안됐다.
- **M2DP (He et al., 2016)**는 포인트 클라우드를 여러 방향의 2D 평면에 투영해 점 밀도 서명 행렬을 만든 뒤, 특잇값 분해(SVD)로 압축해 192차원 벡터를 얻는다.
- **ESF (Ensemble of Shape Functions)**는 임의로 뽑은 점들 사이의 거리·각도·면적 비율 분포를 히스토그램으로 엮어 형상을 요약한다.

두 방법 모두 Scan Context에 비해 공간적 상대 배치 정보는 적게 남기는 대신, 디스크립터 자체에 회전 불변성(rotation invariance)이 있고 벡터 길이가 짧아 비교 부담이 작다.

### 9.3.3 Learning-based: PointNetVLAD

카메라 영역의 NetVLAD가 성공을 거두자, 3D 포인트 클라우드에도 end-to-end 학습을 도입하려는 시도가 이어졌다. **[PointNetVLAD (Uy & Lee, 2018)](https://arxiv.org/abs/1804.03492)**는 점 집합을 직접 받는 PointNet 백본으로 점별 특징을 뽑고, 그 위에 NetVLAD 레이어를 얹어 글로벌 디스크립터를 학습한 초기 대표작이다. 학습에는 가장 어려운 positive($p^+$)와 가장 헷갈리는 negative($p^-$)의 경계를 직접 조이는 lazy triplet·lazy quadruplet loss를 썼다.

$$
\mathcal{L} = \max(0, m + \max_{p^+} d(q, p^+) - \min_{p^-} d(q, p^-))
$$

다만 PointNet이 각 점을 독립적으로 처리한 뒤 전역 풀링으로 합치는 구조이다 보니, 점들 사이의 국소적인 기하 구조를 세밀하게 포착하는 데는 한계가 남았다.

### 9.3.4 MinkLoc3D

PointNet의 국소 구조 부재를 풀기 위해 **[MinkLoc3D (Komorowski, 2021)](https://arxiv.org/abs/2011.04530)** ([code](https://github.com/jac99/MinkLoc3D))는 포인트 클라우드를 희소 복셀(sparse voxel) 격자로 옮긴 뒤 Minkowski Engine의 희소 3D 합성곱(sparse 3D convolution)을 적용했다. 덕분에 인접한 3D 영역의 기하 문맥을 촘촘히 쌓아 올릴 수 있었고, 마지막 집계 단계에는 VLAD 대신 파라미터가 간결한 GeM(Generalized Mean) 풀링을 붙여 글로벌 디스크립터를 뽑았다.

### 9.3.5 OverlapTransformer: Range Image 기반

3D 희소 합성곱 대신 LiDAR 스캔의 물리적 획득 구조 자체를 2D 이미지로 펼쳐 속도를 끌어올린 갈래도 있다. **[OverlapTransformer (Ma et al., 2022)](https://arxiv.org/abs/2203.03397)**는 회전형 LiDAR 스캔을 $(h, w)$ 크기의 range image(빔 수 $h$ × 수평 해상도 $w$ 격자에 거리 값을 채운 2D 영상)로 바꾼 뒤, 가벼운 CNN으로 로컬 특징 맵을 뽑고 Transformer 인코더로 방위각 방향의 전역 문맥을 엮은 다음 NetVLAD 레이어로 글로벌 디스크립터를 만든다. 원 논문(RA-L/IROS 2022)은 이 구조로 yaw 회전에 불변한 디스크립터를 프레임당 2ms 안쪽에서 추출한다고 보고했다. 같은 연구실의 앞선 작업인 **[OverlapNet](https://github.com/PRBonn/OverlapNet)** 역시 range image를 입력으로 받아 두 스캔이 겹치는 비율(overlap)과 상대 yaw 각도를 함께 예측했다.

### 9.3.6 BEV 기반 방법

Range image가 센서의 원통형 투영을 따른다면, 포인트 클라우드를 위에서 수직으로 내려다본 2D 조감도(Bird's Eye View, BEV) 영상으로 투영한 뒤 2D 이미지 검색 기법을 그대로 얹는 접근도 쓰인다. 예를 들어 BEVPlace는 점군을 BEV 이미지로 만든 다음 NetVLAD 계열 집계로 글로벌 디스크립터를 추출해 도로와 건물 배치를 직관적으로 비교한다.

---

## 9.4 Cross-Modal Place Recognition

앞선 §9.2와 §9.3에서는 지도를 만들 때 쓴 센서와 위치를 찾을 때 쓰는 센서가 같다는 전제를 두었다. 하지만 실제 로봇 운용에서는 두 시점의 센서 구성이 달라지는 일이 잦다.

### 9.4.1 왜 Cross-Modal PR이 필요한가

다음과 같은 장면에서는 단일 모달리티 안에서의 검색만으로는 풀리지 않는다.
- 고성능 LiDAR로 미리 구축해 둔 정밀 지도 위에서, 저가형 카메라만 단 로봇이나 차량이 자기 위치를 찾아야 할 때
- LiDAR+Camera를 갖춘 자율주행 차량과 카메라만 실린 배달 로봇이 서로 공통 장소를 확인해야 할 때
- 공중의 카메라 드론이 지상 차량의 LiDAR 지도 안에서 재위치 추정을 해야 할 때

이처럼 서로 다른 센서의 관측을 한 장소로 엮어 주는 문제가 **cross-modal place recognition**이다.

### 9.4.2 Cross-Modal PR의 Domain Gap

어려움의 근원은 LiDAR 포인트 클라우드와 카메라 이미지가 물리적으로 담는 정보와 데이터 구조 자체가 판이하다는 데 있다.

| 특성 | LiDAR 포인트 클라우드 | 카메라 이미지 |
|------|---------------------|-------------|
| 데이터 구조 | 비정형 3D 점 집합 | 정형 2D 그리드 |
| 정보 | 기하(geometry) | 외관(appearance) |
| 조명 의존 | 낮음 | 매우 높음 |
| 텍스처 | 없음 | 풍부 |
| 밀도 | 점 간격이 거리에 비례 (표면 면적당 밀도는 거리 제곱에 반비례, 입사각에도 의존) | 이미지 평면에서 균일 |

이처럼 표현 형식과 관측 특성이 어긋나는 현상을 **domain gap**이라 부르며, 이 간극 때문에 같은 장소를 찍었더라도 LiDAR 인코더와 카메라 인코더가 뱉는 디스크립터 사이 거리가 멀어지기 쉽다.

### 9.4.3 (LC)²: LiDAR-Camera Cross-Modal PR

**[(LC)² (Lee et al., RA-L 2023)](https://arxiv.org/abs/2304.08660)**는 성질이 전혀 다른 2D RGB 이미지와 3D 포인트 클라우드를 곧장 한 공간에 밀어 넣지 않고, 두 센서를 **2.5D 깊이(depth) 이미지**라는 중간 다리로 먼저 끌어당긴다. LiDAR 스캔은 range image로, 스테레오/단안 카메라 영상은 disparity 이미지로 변환해 둘 다 2.5D 기하 표현으로 맞춘 뒤 각자의 인코더에 통과시키는 전략이다.

학습은 두 단계로 나뉜다. 1단계에서는 두 스캔·영상의 중첩 정도로 가중치를 둔 contrastive loss로 두 인코더를 처음부터 학습시켜, 같은 장소에서 온 LiDAR 깊이와 카메라 시차 표현이 임베딩 공간에서 모이게 한다. 2단계에서는 풀링 층을 NetVLAD로 교체하고 triplet margin loss로 다시 미세 조정해 검색 판별력을 높이며, 찾은 매칭은 포즈 그래프의 loop factor로 곧장 연결한다. 아래 식은 1단계 contrastive loss에서 겹침 가중치 항을 덜어 내고 기본 구조만 남긴 형태다.

$$
\mathcal{L}_{\text{contrastive}} = \sum_{(l, c) \in \mathcal{P}^+} \| \mathbf{f}_L(l) - \mathbf{f}_C(c) \|^2 + \sum_{(l, c) \in \mathcal{P}^-} \max(0, m - \| \mathbf{f}_L(l) - \mathbf{f}_C(c) \|)^2
$$

여기서 $\mathbf{f}_L$은 LiDAR 인코더, $\mathbf{f}_C$는 Camera 인코더, $\mathcal{P}^+$와 $\mathcal{P}^-$는 각각 같은 장소와 다른 장소에서 뽑은 positive·negative 쌍이다.

### 9.4.4 ModaLink

이와 달리 중간 깊이 영상 변환 대신 카메라 이미지로 대규모 LiDAR 포인트 클라우드 데이터베이스를 직접 빠르고 가볍게 검색하는 데 초점을 맞춘 방법으로 **[ModaLink (Xie et al., IROS 2024)](https://arxiv.org/abs/2403.18762)**가 제안되었다.

### 9.4.5 Modality-Agnostic Descriptor 접근

더 일반화된 목표는 센서 종류와 무관하게 같은 장소라면 언제나 같은 벡터가 나오도록 **모달리티에 무관한(modality-agnostic) 디스크립터**를 만드는 데 있다. 연구 흐름은 크게 세 방향으로 모인다.

- **Knowledge Distillation**: 정보가 풍부한 LiDAR+Camera 멀티모달 디스크립터를 teacher로 두고, 카메라나 LiDAR 하나만 쓰는 가벼운 student 네트워크가 그 임베딩 공간을 따라 배우도록 증류한다.
- **Canonical Representation**: 어느 센서에서든 끌어낼 수 있는 BEV 격자나 semantic layout 같은 공통 표준 표현으로 변환한 뒤 그 위에서 비교한다.
- **Foundation Model 기반**: [DXPR (Han et al., arXiv 2026)](https://arxiv.org/abs/2609.09005)은 카메라 이미지와 LiDAR 스캔을 모두 depth 이미지로 바꾼 다음, 단일 vision foundation model 백본을 공유해 두 센서 모두에서 통하는 공통 global descriptor를 학습한다.

---

## 9.5 Multi-Session & Long-Term Place Recognition

센서 모달리티를 맞추더라도 시간이라는 변수가 남는다. 하루가 지나거나 계절이 바뀌고 나면 같은 장소라도 전혀 다른 인상을 주기 때문이다.

### 9.5.1 Long-Term VPR의 어려움

장기간 운용하는 로봇 앞에서는 한 장소의 겉모습이 끊임없이 달라진다. 낮과 밤이 바뀌면 그림자와 인공 조명이 장면의 대비를 뒤집고, 여름의 푸른 가로수가 겨울에 앙상한 가지나 눈 더미로 덮이면 윤곽마저 변한다. 여기에 맑은 날과 안개 낀 날의 가시거리 차이, 그리고 건물의 신축·철거나 도로 공사 같은 영구적인 구조 변화까지 겹치면 같은 장소에서 뽑은 디스크립터라도 서로 멀어져 검색 성능이 크게 떨어진다.

### 9.5.2 계절/시간/날씨 변화 대응 전략

이처럼 시간과 환경에 따라 흔들리는 외관 변화를 견뎌 내기 위해 네 가지 전략이 주로 쓰인다.

- **학습 데이터에 여러 조건 넣기**: 같은 장소를 다양한 시간대와 날씨에 찍은 이미지를 함께 학습시킨다. 앞서 본 NetVLAD가 여러 해에 걸쳐 수집한 Google Street View Time Machine 영상을 쓴 것이 대표적이다.
- **Domain Invariant Feature**: 조명이 바뀌어도 잘 변하지 않는 구조적 특징을 뽑아 쓴다. 건물·도로·하늘의 배치를 담은 시맨틱 세그멘테이션 결과는 조도 변화에 비교적 둔감하지만, 한밤중이나 역광처럼 세그멘테이션 모델 자체가 흔들리는 조건에서는 레이블 맵도 함께 무너질 수 있다.
- **Foundation model 특징 쓰기**: AnyLoc 실험에서 보았듯 대규모 자기지도 학습을 거친 DINOv2 특징은 주야간이나 계절 변화 아래에서도 기존 VPR 전용 모델보다 높은 recall을 유지했다(§9.2.4).
- **시퀀스 매칭**: SeqSLAM처럼 단일 프레임의 외관 유사도가 떨어지더라도 앞뒤로 이어지는 장면들의 순서를 통째로 맞춰 변동을 흡수한다.

### 9.5.3 Map Update 전략

아무리 강건한 디스크립터를 쓰더라도 환경 자체가 변하는 이상 지도를 한 번 만들고 끝낼 수는 없다. 오래 운용하는 시스템에서는 재방문 관측이 들어올 때마다 장소 데이터베이스를 함께 갱신해 나간다.

- **경험 기반 맵(Experience-Based Map)**: 같은 장소를 서로 다른 시간대와 계절에 본 관측들을 버리지 않고 모두 쌓아 두었다가, 현재 조건과 가장 닮은 과거 경험과 대조한다. 조건 대응력은 뛰어나지만 운용 기간이 길어질수록 지도 용량과 탐색 대상이 커진다.
- **적응형 디스크립터**: 새 관측이 검증을 통과할 때마다 기존 디스크립터를 지수 이동 평균(exponential moving average) 등으로 조금씩 섞어 최신 모습을 반영한다.
- **Change Detection**: 건물이 철거되거나 새 구조물이 들어선 것처럼 영구적 구조 변화를 감지하면 해당 구역의 오래된 맵 데이터를 폐기하고 새로 구축한다.

### 9.5.4 Lifelong Place Recognition

로봇이 수명 내내 한 도시나 캠퍼스를 누비며 장소를 기억하려면, 메모리와 검색 시간이 끝없이 불어나지 않도록 정보를 압축하고 선별해야 한다. 오래되고 낡은 관측은 조금씩 비워 내되(graceful forgetting) 새로운 환경 변화를 받아들이면서 이전에 익힌 핵심 지식을 한꺼번에 망각하지 않는(catastrophic forgetting 없는) 균형이 요구된다. 이 문제는 연속 학습(continual learning)과 점진적 학습(incremental learning) 연구로 곧바로 이어진다.

---

## 9.6 Geometric Verification & Re-ranking

글로벌 디스크립터 검색은 어디까지나 "비슷해 보이는 후보"를 빠르게 추려 내는 1차 거름망이다. 똑같은 가로수가 늘어선 도로나 비슷한 구조의 사무실 복도에서는 실제로 다른 장소인데도 디스크립터 거리가 가깝게 나오는 perceptual aliasing이 일어나기 쉽다. 따라서 포즈 그래프에 루프 제약을 넣기 전에는 반드시 로컬 특징이나 3D 점군을 직접 정합해 두 관측이 기하학적으로도 일치하는지 확인해야 한다.

### 9.6.1 PnP + RANSAC (Visual)

카메라 기반 시스템이 뱉은 후보는 세 단계로 기하 일관성을 검증한다. 먼저 쿼리 이미지와 후보 이미지 사이에서 SuperPoint+LightGlue나 ORB+BF 같은 로컬 특징 매칭을 수행한다. 다음으로 후보 이미지 쪽 특징점에 연결된 3D 맵 포인트를 끌어와 쿼리의 2D 키포인트와 짝지음으로써 2D-3D 대응 쌍을 만든 뒤, **PnP(Perspective-n-Point)** 문제를 풀어 카메라 포즈를 추정한다. 이때 3D 점이 전역 맵 좌표계에 놓여 있다면 추정 결과는 곧바로 쿼리 카메라의 절대 포즈가 되므로, 두 프레임 사이의 상대 포즈를 얻으려면 후보 키프레임의 알려진 포즈와 합성해 주어야 한다. 마지막으로 이 추정 루프를 **RANSAC** 안에서 돌려 잘못 매칭된 아웃라이어 대응점을 걸러 내고, 남은 인라이어 수가 임계값을 넘을 때만 참 루프로 인정한다.

```python
import numpy as np

def geometric_verification_visual(query_keypoints, db_keypoints_3d, 
                                    K, ransac_threshold=5.0,
                                    min_inliers=30):
    """
    Visual geometric verification: PnP + RANSAC.
    
    query_keypoints: (N, 2) — 쿼리 이미지의 2D 키포인트
    db_keypoints_3d: (N, 3) — 매칭된 데이터베이스의 3D 포인트
    K: (3, 3) — 카메라 내부 파라미터
    """
    import cv2
    
    # PnP + RANSAC으로 카메라 포즈 추정
    success, rvec, tvec, inliers = cv2.solvePnPRansac(
        db_keypoints_3d.astype(np.float32),
        query_keypoints.astype(np.float32),
        K.astype(np.float32),
        distCoeffs=None,
        reprojectionError=ransac_threshold,
        confidence=0.99,
        iterationsCount=1000
    )
    
    if not success or inliers is None:
        return False, None, 0
    
    n_inliers = len(inliers)
    is_verified = n_inliers >= min_inliers
    
    # solvePnP의 (rvec, tvec)는 3D 점 좌표계 → 카메라 변환이다.
    # 3D 점이 맵 좌표계면 T는 맵→질의 카메라 변환이고, 후보와의 상대 포즈는 후보 포즈와 합성해야 한다.
    R, _ = cv2.Rodrigues(rvec)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = tvec.flatten()
    
    return is_verified, T, n_inliers
```

### 9.6.2 3D-3D Registration (LiDAR)

LiDAR 후보 검증은 2D 투영 대신 3차원 공간에서 직접 강체 변환을 찾아 정합 품질을 잰다. 먼저 FPFH나 FCGF, GeoTransformer 같은 기법으로 두 포인트 클라우드 사이의 3D-3D 대응점을 찾은 뒤, **RANSAC**이나 GeoTransformer처럼 RANSAC 없이 대응점 신뢰도만으로 변환을 푸는 방식으로 초기 상대 변환을 추정한다. 특히 GeoTransformer(§5.8.5)는 두 스캔이 겹치는 영역이 좁아도 정확한 대응을 잡아내 RANSAC 반복 없이 초기 정합을 내놓는다. 이어서 이 초기 변환을 출발점으로 **ICP**를 돌려(coarse-to-fine) 정밀하게 맞춘 뒤, 일정 거리 안에 들어온 점의 비율(`fitness`)을 보고 검증 통과 여부를 가린다.

```python
def geometric_verification_lidar(query_cloud, db_cloud, 
                                   voxel_size=0.5, 
                                   distance_threshold=0.5,
                                   fitness_threshold=0.3):
    """
    LiDAR geometric verification: FPFH + RANSAC + ICP.
    """
    import open3d as o3d
    
    # 다운샘플링
    q_down = query_cloud.voxel_down_sample(voxel_size)
    d_down = db_cloud.voxel_down_sample(voxel_size)
    
    # 법선 추정
    q_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size*2, max_nn=30))
    d_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size*2, max_nn=30))
    
    # FPFH 특징 추출
    q_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        q_down, o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size*5, max_nn=100))
    d_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        d_down, o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size*5, max_nn=100))
    
    # RANSAC 기반 정합
    result_ransac = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        q_down, d_down, q_fpfh, d_fpfh,
        mutual_filter=True,
        max_correspondence_distance=distance_threshold * 2,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        ransac_n=3,
        criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999)
    )
    
    # ICP 정밀 정합
    result_icp = o3d.pipelines.registration.registration_icp(
        q_down, d_down,
        distance_threshold,
        result_ransac.transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPlane()
    )
    
    is_verified = result_icp.fitness > fitness_threshold
    
    return is_verified, result_icp.transformation, result_icp.fitness
```

### 9.6.3 Spatial Re-ranking

기하 검증을 모든 후보에 무작정 돌리면 계산량이 커지므로, 디스크립터 유사도만으로 세운 후보 순위를 공간·기하 단서로 다시 매기는(re-ranking) 단계를 두기도 한다.

- **공간적 근접성 prior**: SLAM이 현재 추정하고 있는 포즈 반경 가까이에 있는 후보에 가산점을 준다. 다만 오도메트리 드리프트가 아직 작을 때만 유효하다.
- **Inlier 수 기반 재랭킹**: 1차 특징 매칭이나 간이 기하 모델 추정 뒤 살아남은 인라이어 개수가 많은 후보를 앞세운다.
- **포즈 일관성 검사**: 기하 정합으로 구한 상대 포즈가 오도메트리로 적분해 온 경로와 얼마나 어긋나는지 확인한다. 이때 유클리드 거리 대신 오도메트리 공분산으로 정규화한 Mahalanobis 거리를 써야 한다. 루프가 길어질수록 참 재방문이라도 그동안 쌓인 드리프트만큼 좌표상 어긋나 보이기 때문에, 불확실성 규모를 반영하지 않으면 정작 가장 필요한 긴 루프 후보를 버리게 된다.

---

## 9.7 파운데이션 모델·시맨틱·4D 레이더 장소 인식

최근 장소 인식 연구는 단순한 픽셀·포인트 기술자를 넘어 범용 파운데이션 모델의 표현력을 끌어오거나, 의미 단위 구조와 전천후 레이더 센서로 영역을 넓히고 있다.

### 9.7.1 파운데이션 모델 기반 장소 인식

현재 VPR 연구는 특정 도메인에 맞춰 모델을 파인튜닝하는 흐름과, AnyLoc처럼 VPR 전용 학습 없이 foundation model을 zero-shot으로 가져다 쓰는 흐름이 나란히 달리고 있다. 다만 AnyLoc이 기반으로 삼은 DINOv2 ViT-g는 파라미터 수가 11억 개에 달해 로봇의 임베디드 보드에서 실시간으로 돌리기 무겁다. 이 때문에 실전 지향 연구에서는 경량 백본과 부분 미세 조정을 택한다. 예컨대 SALAD(§9.7.4)는 훨씬 작은 DINOv2 ViT-B의 마지막 네 블록만 파인튜닝해 성능을 끌어올렸고, EffoVPR(§9.7.4)은 추가 학습 없이 foundation model 내부 self-attention 특징을 re-ranking 단계에 활용하는 길을 열었다.

### 9.7.2 Semantic Place Recognition

픽셀 밝기나 국소 점군 분포 대신 "빨간 건물 옆의 큰 나무"처럼 사물과 그 공간 배치(spatial layout)로 장소를 기술하는 방식도 활발하다. 시맨틱 세그멘테이션 결과의 공간 배치를 비교하거나, 객체 검출 결과를 scene graph로 엮고, CLIP으로 자연어 묘사에 맞는 장소를 찾는 식이다. 나뭇잎 색깔이 바뀌어도 "나무"라는 의미 레이블은 그대로 남으므로 계절과 조명 변화에 덜 흔들린다. 반대로 세그멘테이션 자체가 오작동하면 장소 표현도 함께 틀어지며, 똑같은 집과 가로수가 줄지어 선 주택가처럼 사물 배치가 닮은 장소들을 가려내는 데는 취약하다.

### 9.7.3 4D Radar Place Recognition

고도각(elevation)까지 분해하는 4D imaging radar가 보급되면서 레이더를 장소 인식에 투입하는 연구도 자리 잡기 시작했다. 4D radar 포인트 클라우드를 Scan Context 변형 디스크립터로 인코딩하거나, range-Doppler 이미지를 CNN에 넣어 글로벌 디스크립터를 뽑거나, 레이더와 카메라를 짝짓는 cross-modal PR을 수행한다. 비·눈·안개 속에서 카메라와 LiDAR PR이 동시에 무너질 때 대체 후보로 쓸 수 있다는 점이 강점이지만, 점군이 성기고 각도 해상도가 낮아 세밀한 기하 변별력에는 한계가 따른다.

### 9.7.4 2024–2025년 연구

- **[SALAD (Izquierdo & Civera, 2024)](https://arxiv.org/abs/2311.15937)**: NetVLAD의 soft assignment를 최적 수송(optimal transport) 문제로 다시 세우고, 하늘이나 도로처럼 장소 변별력이 없는 특징을 버리는 'dustbin' 클러스터를 도입했다. 백본으로는 DINOv2를 파인튜닝해 쓰며, 논문(CVPR 2024)은 이 단일 단계 모델이 공개 VPR 벤치마크에서 기존 단일 단계 방법뿐 아니라 별도 re-ranking을 붙인 2단계 방법보다도 높은 성능을 냈다고 보고한다.
- **[EffoVPR (Tzachor et al., ICLR 2025)](https://arxiv.org/abs/2405.18065)**: DINOv2 같은 foundation model의 self-attention 레이어 내부 특징을 추가 학습 없이 re-ranking에 그대로 활용한다. 논문에 따르면 ViT 내부 레이어에서 풀링한 글로벌 특징은 128차원까지 압축해도 높은 검색 성능을 유지했다.

### 9.7.5 기술 계보 요약

지금까지 살펴본 시각, LiDAR, 그리고 크로스모달 장소 인식 기법들이 어떤 한계를 넘으며 이어져 왔는지 흐름을 한눈에 정리하면 다음과 같다.

```
Visual Place Recognition:

Sivic (2003) Video Google [BoW]
    ↓ 양자화 → 잔차로
Jégou (2010) VLAD
    ↓ hand-crafted → CNN end-to-end
Arandjelović (2016) NetVLAD [triplet, soft-assignment]
    ↓ triplet → classification
Berton (2022) CosPlace, (2023) EigenPlaces
    ↓ VPR-specific → Foundation Model
Keetha (2024) AnyLoc [DINOv2 + VLAD, zero-shot]
    ↓ assignment 최적화 + FM fine-tuning
Izquierdo (2024) SALAD [optimal transport + DINOv2]

LiDAR Place Recognition:

Kim (2018) Scan Context [handcrafted, spatial]
    ↓ handcrafted → learned
Uy (2018) PointNetVLAD [PointNet + NetVLAD]
    ↓ PointNet → sparse convolution
Komorowski (2021) MinkLoc3D [sparse conv + GeM]
    ↓ 3D point → 2D range image
Ma (2022) OverlapTransformer [range image + transformer]

Cross-Modal:
Lee (2023) (LC)² [LiDAR range image ↔ Camera disparity]
```

장소 인식이 글로벌 디스크립터 검색과 기하학적 검증을 거쳐 "현재 로봇이 과거의 특정 키프레임을 다시 방문했다"는 상대 변환 추정치를 내놓으면, 그 제약을 SLAM 그래프에 밀어 넣어 그동안 쌓인 궤적과 지도의 오차를 실제로 펴 주어야 한다. 이 루프 제약 등록과 오탐 방어, 그리고 전역 포즈 그래프 최적화를 다음 장(→ 10장)에서 다룬다.

