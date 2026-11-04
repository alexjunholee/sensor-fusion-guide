# Ch.9 — Place Recognition & Retrieval

Odometry나 fusion 시스템의 추정에는 시간이 지날수록 드리프트가 쌓인다. Place Recognition(장소 인식)은 로봇이 전에 지나간 장소를 다시 찾아내 이 드리프트를 바로잡을 후보를 내놓는다. 그 결과는 Ch.10의 Loop Closure로 바로 넘어간다.

> 로봇이 "이 장소를 전에 본 적이 있는가?"를 판단하는 문제다.
> Loop closure 후보를 만들고, multi-session SLAM(여러 번 나눠 만든 지도를 잇는 SLAM)과 재위치추정(relocalization)에도 쓴다.

---

## 9.1 문제 정의

### 9.1.1 Place Recognition vs Loop Closure Detection

**Place Recognition (PR)**은 "현재 관측이 데이터베이스의 어떤 장소와 일치하는가?"를 판단하는 retrieval(검색) 문제다. SLAM이 없어도 따로 정의할 수 있다.

**Loop Closure Detection**은 로봇이 이전에 방문했던 장소에 돌아왔는지를 SLAM 시스템 안에서 탐지한다. Place recognition이 후보를 만들면 loop closure 파이프라인이 geometric verification(기하학적 검증)을 거쳐 확정한다.

```
Loop Closure Detection Pipeline:
┌─────────────┐    ┌────────────────┐    ┌───────────────────┐    ┌─────────────┐
│  현재 관측   │───→│ Place          │───→│ Geometric         │───→│ Pose Graph  │
│  (query)     │    │ Recognition    │    │ Verification      │    │ Update      │
│              │    │ (후보 검색)     │    │ (기하학적 검증)    │    │ (최적화)     │
└─────────────┘    └────────────────┘    └───────────────────┘    └─────────────┘
```

현재 query의 global descriptor를 $N$개 과거 frame과 하나하나 다 비교(exact search)하면 query당 $O(N)$, 전체 sequence의 all-pairs 비교에 $O(N^2)$가 든다. Inverted index나 tree, approximate-nearest-neighbor index를 쓰면 평균 query 비용이 줄어든다. 얼마나 줄어드는지는 데이터 분포와 감수할 recall 손실에 달렸다.

### 9.1.2 Retrieval Pipeline

Place recognition은 정보 검색(Information Retrieval)과 같은 순서로 처리한다:

1. **Encoding**: 각 관측(이미지, 포인트 클라우드, 또는 둘 다)을 고정 길이의 글로벌 디스크립터(global descriptor, 관측 하나를 요약한 벡터)로 바꾼다.
2. **Indexing**: 데이터베이스의 모든 디스크립터를 검색용 인덱스 구조(예: kd-tree, FAISS)에 넣어 둔다.
3. **Retrieval**: 쿼리 디스크립터와 가장 비슷한 데이터베이스 디스크립터를 찾는다.
4. **Re-ranking & Verification**: 후보를 기하학적으로 검증해 최종 매칭을 정한다.

$$
q^* = \arg\min_{q \in \mathcal{D}} d(\mathbf{f}(\text{query}), \mathbf{f}(q))
$$

여기서 $\mathbf{f}(\cdot)$는 관측을 글로벌 디스크립터로 바꾸는 함수, $d(\cdot, \cdot)$는 거리 함수(보통 L2 또는 코사인 거리), $\mathcal{D}$는 데이터베이스이다.

### 9.1.3 평가 메트릭

PR 시스템의 성능은 다음 메트릭으로 평가한다:

**Recall@N**: 상위 $N$개 후보 안에 올바른 매칭이 들어 있는 쿼리의 비율이다. 이 장에서 인용하는 VPR 논문(NetVLAD, AnyLoc, SALAD 등)은 모두 이 지표로 결과를 보고한다.

$$
\text{Recall@N} = \frac{|\{q : \text{top-}N \text{ 후보 중 정답이 있는 쿼리 } q\}|}{|\text{전체 쿼리}|}
$$

**Recall@1**은 첫 번째 검색 결과가 맞은 비율이다. 실제 SLAM은 지연시간과 false-positive 비용에 따라 top-$k$ 후보를 기하 검증하므로 Recall@$k$, precision-recall, query latency도 함께 봐야 한다.

**Precision-Recall Curve**: 디스크립터 유사도 임계값을 바꿔 가며 precision과 recall을 함께 그린다. SLAM에서는 precision이 특히 중요하다. 잘못 받아들인 loop closure(false positive) 하나가 맵을 크게 비튼다.

**"정답"의 정의**: 무엇을 같은 장소로 볼지는 데이터셋의 annotation protocol이 위치 거리, 시야 중첩, 자세 차이 같은 기준으로 정한다. 일부 VPR benchmark는 GPS 반경 25m를 쓰지만, 이 값을 다른 데이터셋에 그대로 가져다 쓰면 안 된다. 아래 코드의 25m도 예시값이다.

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

---

## 9.2 Visual Place Recognition (VPR)

Visual Place Recognition(VPR)은 이미지만으로 장소를 알아보는 문제다.

### 9.2.1 전통적 방법: Bag of Words (BoW)

**[Video Google (Sivic & Zisserman, 2003)](https://www.robots.ox.ac.uk/~vgg/publications/2003/Sivic03/)**이 제안한 Bag of Visual Words 모델은 텍스트 검색(text retrieval) 방법을 이미지 검색에 옮겨 왔다.

파이프라인:

1. **시각 어휘 구축(Visual Vocabulary Construction)**: 대량의 이미지에서 로컬 특징(SIFT, ORB 등)을 추출하고, 이 특징 디스크립터를 k-means로 $K$개 클러스터로 묶는다. 각 클러스터 중심이 하나의 "시각 단어(visual word)"가 된다.

2. **이미지 표현**: 각 이미지에서 추출한 특징들을 가장 가까운 시각 단어 하나에만 배정(hard assignment)한다. 시각 단어마다 등장 횟수를 세면 이미지 하나가 $K$차원 히스토그램이 된다.

3. **TF-IDF 가중**: 텍스트 검색에서 쓰던 TF-IDF(Term Frequency – Inverse Document Frequency) 가중치를 붙인다:

$$
w_{i,d} = \underbrace{\frac{n_{i,d}}{n_d}}_{\text{TF}} \cdot \underbrace{\log \frac{N}{N_i}}_{\text{IDF}}
$$

여기서 $n_{i,d}$는 이미지 $d$에서 시각 단어 $i$의 등장 횟수, $n_d$는 이미지 $d$의 전체 시각 단어 수, $N$은 데이터베이스의 전체 이미지 수, $N_i$는 시각 단어 $i$가 등장하는 이미지 수이다.

- **TF**: 그 이미지에 자주 나오는 단어일수록 가중치가 크다.
- **IDF**: 데이터베이스 전체에서 드문 단어일수록 이미지를 잘 구별해 주므로 가중치가 크다.

4. **역색인(Inverted Index)**: 시각 단어마다 그 단어가 나오는 이미지 목록을 미리 만들어 둔다. 쿼리가 오면 쿼리의 단어가 들어 있는 이미지만 살펴본다.

5. **유사도 랭킹**: 쿼리 이미지와 데이터베이스 이미지의 TF-IDF 벡터 사이 코사인 유사도로 순위를 매긴다:

$$
\text{sim}(\mathbf{v}_q, \mathbf{v}_d) = \frac{\mathbf{v}_q \cdot \mathbf{v}_d}{\|\mathbf{v}_q\| \cdot \|\mathbf{v}_d\|}
$$

**DBoW2**: ORB-SLAM 시리즈가 쓰는 BoW 구현이다. 계층형 k-means tree(hierarchical k-means tree)로 큰 vocabulary를 빠르게 검색한다. VINS-Mono도 loop 후보를 찾을 때 DBoW2를 쓴다.

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

BoW는 양자화하면서 정보를 많이 잃는다. 비슷한 특징이 모두 같은 단어로 묶이기 때문이다. 조명·시점 변화에 얼마나 견디는지는 바탕이 되는 특징(SIFT, ORB)이 정한다. vocabulary 크기 $K$에 따라서도 성능이 크게 달라진다.

### 9.2.2 VLAD와 Fisher Vector

단순 빈도 히스토그램 대신 잔차(residual)를 남기는 집계 방법이 나왔다.

**[VLAD (Vector of Locally Aggregated Descriptors, Jégou et al., 2010)](https://doi.org/10.1109/CVPR.2010.5540039)**:

VLAD는 각 시각 단어와 거기 배정된 실제 디스크립터 사이의 잔차가 어느 방향으로 얼마나 큰지를 기록한다.

클러스터 $k$마다, 그 클러스터에 배정된 디스크립터에서 클러스터 중심을 뺀 잔차를 모두 더한다:

$$
\mathbf{V}_k = \sum_{i: \text{NN}(\mathbf{x}_i) = k} (\mathbf{x}_i - \mathbf{c}_k)
$$

최종 VLAD 디스크립터는 모든 $\mathbf{V}_k$를 이어 붙인(concatenate) 벡터다: $\mathbf{V} = [\mathbf{V}_1^T, \mathbf{V}_2^T, \ldots, \mathbf{V}_K^T]^T$. 차원은 $K \times D$이다.

**Fisher Vector**: 시각 단어를 GMM(Gaussian Mixture Model)으로 모델링하고, 가우시안 컴포넌트마다 1차·2차 통계량을 Fisher Information Matrix의 제곱근으로 정규화한다. 차원은 $2KD$로 VLAD의 두 배다. 어느 쪽 검색 성능이 나은지는 지표·데이터·정규화 설정에 따라 달라진다. [Jégou et al. (2010)](https://inria.hal.science/inria-00548637/document)은 VLAD를 Fisher Vector를 단순화한 것으로 보았고, 이 논문의 Table 1은 평균 성분만 쓴 Fisher Vector와 VLAD를 같은 차원에서 비교했다. Holidays mAP는 2,048차원에서 0.497과 0.496으로 거의 같았고, 8,192차원에서는 VLAD가 0.526으로 Fisher Vector(0.495)보다 높았다.

### 9.2.3 NetVLAD: 학습 기반 VPR의 기준점

**[NetVLAD (Arandjelović et al., 2016)](https://arxiv.org/abs/1511.07247)**은 VLAD를 미분 가능한(differentiable) CNN 레이어로 다시 짜서, 네트워크 전체를 장소 인식 목적에 맞춰 한 번에(end-to-end) 학습할 수 있게 했다.

기존 VLAD처럼 디스크립터를 가장 가까운 클러스터 하나에 배정(hard assignment)하면 미분할 수 없어 역전파가 안 된다.

**해결: Soft Assignment**:

Hard assignment($\text{NN}(\mathbf{x}_i) = k$이면 1, 아니면 0)를 softmax로 바꾼다:

$$
\bar{a}_k(\mathbf{x}_i) = \frac{e^{\mathbf{w}_k^T \mathbf{x}_i + b_k}}{\sum_{k'} e^{\mathbf{w}_{k'}^T \mathbf{x}_i + b_{k'}}}
$$

원래 VLAD의 유클리드 거리 기반 할당 $e^{-\alpha\|\mathbf{x}_i - \mathbf{c}_k\|^2}$을 전개하면, $\mathbf{w}_k = 2\alpha\mathbf{c}_k$, $b_k = -\alpha\|\mathbf{c}_k\|^2$로 해석할 수 있다. 이 파라미터를 학습하면 클러스터 중심이 데이터에 맞게 움직인다.

**NetVLAD 레이어 출력**:

$$
\mathbf{V}(j, k) = \sum_{i=1}^{N} \bar{a}_k(\mathbf{x}_i) (x_i(j) - c_k(j))
$$

이 $D \times K$ 행렬을 L2 정규화하고 벡터로 펼치면 최종 글로벌 디스크립터가 된다.

학습 데이터는 같은 장소를 여러 시기에 찍은 Google Street View Time Machine 이미지이고, 학습 방식은 약한 지도 학습(weakly supervised learning)이다. GPS가 가깝다고 두 이미지가 같은 장면을 담았다는 보장은 없어서 지도 신호가 약하다. 그래서 지리적으로 가까운 이미지를 potential positive로 두고, 그중 특징 공간에서 가장 가까운 하나를 positive로 쓴다. negative는 멀리 떨어진 이미지 가운데 헷갈리기 쉬운 것(hard negative)을 골라 쓴다. 손실은 triplet ranking loss다:

$$
\mathcal{L} = \sum_{(q, p^+, p^-)} \max\left(0, m + d(\mathbf{f}(q), \mathbf{f}(p^+)) - d(\mathbf{f}(q), \mathbf{f}(p^-))\right)
$$

여기서 $m$은 마진, $p^+$는 positive 이미지, $p^-$는 hard negative 이미지이다.

논문 본문에 따르면 학습한 AlexNet+NetVLAD의 Pitts250k-test recall@1은 81.0%로, 학습하지 않은 AlexNet+VLAD(55.0%)보다 상대적으로 47% 높았다. 가장 좋은 VGG-16 결과는 그림으로만 제시했다.

### 9.2.4 AnyLoc: Foundation Model 기반 범용 VPR

**[AnyLoc (Keetha et al., RA-L 2024)](https://arxiv.org/abs/2308.00688)**은 VPR용으로 따로 학습하지 않은 모델로 여러 환경의 장소를 알아보려 했다. 범용 사전학습 모델(foundation model)인 DINOv2의 특징을 그대로 가져다 쓴다.

1. **DINOv2 특징 추출**: DINOv2 ViT-G/14의 31번째 layer에서 value facet의 dense patch feature를 뽑는다. 이미지 전체를 요약한 CLS token 대신 패치마다 나오는 특징을 쓴다. 저자들은 자신들이 비교한 데이터셋에서 dense feature가 평균적으로 더 나았다고 보고했다. 어느 layer가 좋은지와 그 차이는 backbone과 데이터셋에 따라 달라진다.

2. **VLAD 집계**: dense feature를 k-means로 묶어 시각 어휘를 만들고, hard-assignment VLAD로 글로벌 디스크립터를 만든다. 이 단계에는 학습이 없다.

3. **도메인별 어휘**: 논문은 dense feature를 이미지 단위로 모은 디스크립터를 PCA로 투영해 보면 Urban, Indoor, Aerial, SubT, Degraded, Underwater 같은 무리로 나뉜다는 것을 보이고, 이 domain마다 따로 vocabulary를 만든다. 새 domain에서는 어떤 vocabulary가 맞는지 다시 확인해야 한다.

[DINOv2 (Oquab et al., 2023)](https://arxiv.org/abs/2304.07193)는 선별한 이미지 1억 4,200만 장(LVD-142M)으로 자기지도 학습한 모델이다. 장소 인식 label로 파인튜닝하지 않은 feature도 여러 benchmark에서 쓸 만했다. 그래도 반복 구조, 극단적인 시점 차이, 학습 데이터에 없던 센서 domain에서 장소를 구별해 낸다는 보장은 없다.

AnyLoc 논문이 MixVPR / CosPlace와 비교해 보고한 Recall@1 차이는 다음과 같다(논문이 고른 benchmark 기준):
- 주야간 변화: 5% / 11%
- 계절 변화: 9% / 8%
- 오랜 시간을 두고 다시 찍은 영상(논문 예: 1년, 5년 간격): 21% / 28%
- 방향이 달라진 시점: 21% / 30%
- 90°·180°의 극단적 시점 변화: 39% / 49%
- 비정형 환경을 포함한 전체: 기존 방법의 최대 4배

논문은 PCA-Whitening으로 49K 차원을 512차원으로 줄여도 비슷한 성능을 냈다고 보고했다.

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

**[EigenPlaces (Berton et al., 2023)](https://arxiv.org/abs/2308.10832)**는 같은 저자의 [CosPlace (Berton et al., 2022)](https://arxiv.org/abs/2204.02287)를 이어, 시점이 바뀌어도 잘 맞는 디스크립터를 학습한다. CosPlace는 triplet loss와 hard negative mining 대신 분류(classification) 문제로 학습했다. EigenPlaces는 클래스마다 카메라 위치 분포에 PCA를 적용해 대략 도로 방향인 제1 주성분을 찾는다. 그 수직 방향(도로 옆 건물 정면 쪽)에 focal point를 두고, 여러 위치에서 그 점을 바라본 이미지로 학습 클래스를 만든다. 전방 카메라 데이터셋을 위해 제1 주성분 방향을 바라보는 frontal 클래스도 둔다.

### 9.2.6 Foundation Model 활용의 확장

DINOv2 말고도 여러 foundation model을 VPR에 쓴다:

- **CLIP**: 텍스트와 이미지를 짝지어 학습한 모델이라 "도시 거리", "숲 속 길"처럼 의미 수준으로 장소를 찾는 데 쓸 수 있다. AnyLoc이 자기 설정과 데이터셋에서 여러 foundation model을 비교했을 때는 DINOv2 특징이 장소를 더 잘 구별했다.
- **SAM (Segment Anything)**: [SegVLAD (Garg et al., ECCV 2024)](https://arxiv.org/abs/2409.18049)는 SAM으로 이미지를 segment로 나누고, 이웃 segment와 묶은 SuperSegment를 DINOv2 특징과 VLAD로 인코딩해 이미지 전체 대신 segment 단위로 검색한다.
- **DINOv2 + 학습된 집계**: AnyLoc처럼 학습 없이 VLAD로 모으는 대신, DINOv2 특징을 집계하는 단계를 학습하는 연구가 뒤따랐다. SALAD(§9.7.4)는 NetVLAD의 할당을 최적 수송(optimal transport) 문제로 바꾸었고, SelaVPR(Lu et al., ICLR 2024)은 사전학습 모델을 VPR에 맞게 조정한다.

### 9.2.7 SeqSLAM과 시퀀스 매칭

**[SeqSLAM (Milford & Wyeth, 2012)](https://doi.org/10.1109/ICRA.2012.6224623)**은 이미지 한 장씩 맞추지 않고 이어지는 이미지 묶음(시퀀스)을 통째로 맞춘다.

비슷하게 생긴 주택가처럼 한 장으로는 구별하기 어려운 장소라도, 좌회전 → 공원 → 건널목처럼 이어지는 순서로는 구별할 수 있다.

먼저 각 이미지를 아주 작은 표현(저해상도 패치나 단순한 디스크립터)으로 줄인다. 다음으로 쿼리 이미지와 데이터베이스 이미지를 모두 짝지어 유사도 행렬을 만든다. 이 행렬에서 정해 둔 속도 범위 안의 대각선 경로를 찾아 가장 잘 맞는 시퀀스를 고른다.

$$
S_{\text{seq}}(q_s, d_s) = \min_{\text{path}} \sum_{i=0}^{L-1} d(\mathbf{f}(q_{s+i}), \mathbf{f}(d_{s+\delta(i)}))
$$

여기서 $L$은 시퀀스 길이, $\delta(i)$는 속도 변화를 허용하는 경로 함수이다.

낮과 밤, 여름과 겨울처럼 겉모습이 달라져도 순서는 그대로 남아서 매칭이 잘 버틴다. SeqNet(Garg & Milford, RA-L 2021)은 시퀀스 디스크립터를 학습해 이 방식을 이었다.

---

## 9.3 LiDAR Place Recognition

3D LiDAR 포인트 클라우드로 장소를 알아보는 문제다. LiDAR는 스스로 빛을 쏘므로 조명 변화의 영향을 카메라보다 훨씬 덜 받는다. 대신 계절에 따라 나무와 풀이 바뀌거나 시점이 달라지면 성능이 떨어진다.

### 9.3.1 Handcrafted 방법: Scan Context

**[Scan Context (Kim & Kim, 2018)](https://doi.org/10.1109/IROS.2018.8593953)**은 3D LiDAR 스캔을 공간 배치가 남아 있는 2D 행렬로 바꿔 글로벌 디스크립터로 쓴다. 학습이 필요 없어 여러 LiDAR SLAM 시스템이 loop closure 모듈로 쓴다. LIO-SAM의 공개 구현은 기본으로 반경 검색과 ICP로 루프를 찾고, Scan Context를 붙인 버전은 SC-LIO-SAM이라는 별도 저장소에 있다.

디스크립터 생성 과정:

1. **극좌표 분할**: 센서를 중심으로 한 2D 평면을 방위각(azimuth)으로 $N_s$개 섹터, 거리(range)로 $N_r$개 링으로 나눠 $N_s \times N_r$ 그리드를 만든다. 원 논문은 60×20을 썼다.

2. **최대 높이 인코딩**: 각 칸(bin)에 들어간 3D 점 가운데 가장 높은 점의 높이를 기록한다. 이것이 Scan Context 행렬 $\mathbf{SC} \in \mathbb{R}^{N_s \times N_r}$이다.

$$
\mathbf{SC}(i, j) = \max_{p \in \text{bin}(i,j)} z(p)
$$

최대 높이를 쓰면 건물, 나무, 기둥처럼 솟은 구조물이 잘 드러난다.

3. **센서 중심(egocentric) 표현**: 센서 좌표계에서 만들기 때문에, 센서 위치가 같고 yaw만 다르면 섹터 축이 순환 이동(circular shift)한 행렬이 나온다. 매칭할 때는 섹터 축을 한 칸씩 돌려 가며 거리를 잰다. 실제 반대 방향 재방문에는 횡방향 이동과 가림 변화가 함께 와서, 원 논문은 root shifting 증강을 따로 둔다.

**검색 전략 — Ring Key & Sector Key**:

데이터베이스 전체와 Scan Context 행렬을 하나하나 비교하면 느리다. 그래서 두 단계로 찾는다:

1. **Ring Key**: 링(열)마다 섹터 방향으로 값을 평균해 벡터 $\mathbf{k}_r = [\bar{h}_1, \bar{h}_2, \ldots, \bar{h}_{N_r}]$를 만든다. yaw가 바뀌어도 이 값은 그대로다. 이 벡터로 kd-tree를 검색해 후보를 좁힌다.

2. **Sector Key**: 섹터마다 링 축 방향으로 값을 모은 키로, yaw에 따라 순서가 바뀐다. 두 스캔의 sector key를 맞춰 보면 이동량을 대략 알 수 있고, 그 근처 몇 칸만 돌려 가며 아래 거리를 계산한다. 아래 식과 코드는 간단히 하려고 모든 이동을 다 해 본다:

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

**후속작 — [Scan Context++](https://arxiv.org/abs/2109.13494)**: 디스크립터를 확장해 roll·pitch 변화가 크지 않으면 heading 회전과 옆으로 비켜난 위치 변화에도 잘 맞는다. 장소를 찾은 뒤에는 1-DOF 상대 정렬도 추정한다.

### 9.3.2 M2DP와 ESF

Scan Context 외의 handcrafted LiDAR 디스크립터:

- **M2DP (He et al., 2016)**: 포인트 클라우드를 여러 2D 평면에 투영하고, 투영마다 나온 점 분포를 한 행렬로 모아 SVD로 압축해 192차원 디스크립터를 만든다. 회전에 불변(rotation invariant)이다.
- **ESF (Ensemble of Shape Functions)**: 점 쌍 사이 거리, 각도, 면적 비율의 히스토그램을 묶은 디스크립터다.

둘 다 Scan Context보다 공간 배치 정보는 덜 남기는 대신 회전에 불변이고 디스크립터가 짧다.

### 9.3.3 Learning-based: PointNetVLAD

**[PointNetVLAD (Uy & Lee, 2018)](https://arxiv.org/abs/1804.03492)**는 PointNet과 NetVLAD를 합쳐 포인트 클라우드 장소 검색을 end-to-end로 학습한 초기 연구다. PointNet backbone으로 특징을 뽑고 NetVLAD layer로 global descriptor를 모은다. 손실로는 저자들이 제안한 lazy triplet·lazy quadruplet loss를 쓴다.

$$
\mathcal{L} = \max(0, m + \max_{p^+} d(q, p^+) - \min_{p^-} d(q, p^-))
$$

PointNet은 점을 하나씩 따로 처리한 뒤 풀링하기 때문에 주변 점들과의 국소 구조를 잘 담지 못한다.

### 9.3.4 MinkLoc3D

**[MinkLoc3D (Komorowski, 2021)](https://arxiv.org/abs/2011.04530)** ([code](https://github.com/jac99/MinkLoc3D))는 포인트 클라우드를 희소 voxel로 바꾸고 Minkowski Engine의 희소 3D 합성곱(sparse 3D convolution)으로 국소 구조를 잡는다. 글로벌 디스크립터는 GeM(Generalized Mean) 풀링으로 만든다.

### 9.3.5 OverlapTransformer: Range Image 기반

**[OverlapTransformer (Ma et al., 2022)](https://arxiv.org/abs/2203.03397)**는 LiDAR 포인트 클라우드를 range image로 바꿔 2D 이미지처럼 처리한다. 프레임당 2ms 안쪽으로 돌아가고 yaw 회전에 불변이다(RA-L/IROS 2022).

**Range Image**: 회전형 LiDAR 스캔을 $(h, w)$ 크기의 2D 이미지로 펼친다. 픽셀 값은 그 방향으로 잰 거리(range)다. $h$는 레이저 빔 수, $w$는 수평 해상도에 대응한다.

range image를 가벼운 CNN에 넣어 로컬 feature map을 뽑고, Transformer encoder로 특징 사이 문맥을 반영한 뒤, NetVLAD 레이어로 글로벌 디스크립터를 모은다.

**관련 방법 — [OverlapNet](https://github.com/PRBonn/OverlapNet)**: range image로 두 스캔이 얼마나 겹치는지(overlap)와 상대 yaw를 예측한다.

### 9.3.6 BEV 기반 방법

포인트 클라우드를 위에서 내려다본 2D 영상(Bird's Eye View, BEV)으로 투영하고 이미지 검색 기법을 쓰는 방법도 있다. BEVPlace가 그 예로, BEV 이미지에서 NetVLAD 디스크립터를 뽑는다.

---

## 9.4 Cross-Modal Place Recognition

### 9.4.1 왜 Cross-Modal PR이 필요한가

지도를 만들 때 쓴 센서와 그 지도에서 위치를 찾을 때 쓰는 센서가 다를 때가 있다. 예를 들면:
- LiDAR로 만든 맵에서 카메라만으로 위치를 찾아야 하는 경우
- 자율주행 차량(LiDAR+Camera)과 배달 로봇(Camera만)이 서로 장소를 알아봐야 하는 경우
- 드론(Camera)이 차량(LiDAR)이 만든 맵에서 자기 위치를 찾는 경우

이때는 LiDAR 관측과 카메라 관측을 서로 맞추는 장소 인식(cross-modal PR)이 필요하다.

### 9.4.2 Cross-Modal PR의 Domain Gap

LiDAR 포인트 클라우드와 카메라 이미지는 데이터 형태부터 다르다:

| 특성 | LiDAR 포인트 클라우드 | 카메라 이미지 |
|------|---------------------|-------------|
| 데이터 구조 | 비정형 3D 점 집합 | 정형 2D 그리드 |
| 정보 | 기하(geometry) | 외관(appearance) |
| 조명 의존 | 낮음 | 매우 높음 |
| 텍스처 | 없음 | 풍부 |
| 밀도 | 점 간격이 거리에 비례 (표면 면적당 밀도는 거리 제곱에 반비례, 입사각에도 의존) | 이미지 평면에서 균일 |

이 차이를 domain gap이라 한다. 이 때문에 같은 장소를 다른 센서로 보면 디스크립터 사이 거리가 멀어진다.

### 9.4.3 (LC)²: LiDAR-Camera Cross-Modal PR

**[(LC)² (Lee et al., RA-L 2023)](https://arxiv.org/abs/2304.08660)**는 2D 이미지와 3D 포인트 클라우드 사이에 곧바로 공통 임베딩을 학습하지 않고, 두 데이터를 모두 2.5D depth 이미지로 바꿔서 맞춘다. LiDAR 스캔은 range image로, 카메라 영상은 disparity 이미지로 바꾼다.

두 이미지는 각자의 인코더를 거친다. 학습은 두 단계다. 1단계에서는 겹치는 정도로 가중한 contrastive loss로 두 인코더를 처음부터 학습해 같은 장소의 LiDAR·카메라 이미지가 가까워지게 한다. 2단계에서는 pooling을 NetVLAD로 바꾸고 triplet margin loss로 다시 학습한다. 찾은 매칭은 pose graph의 loop factor로 넣는다. 아래 식은 1단계 손실에서 겹침 가중치를 빼고 단순화한 형태다.

$$
\mathcal{L}_{\text{contrastive}} = \sum_{(l, c) \in \mathcal{P}^+} \| \mathbf{f}_L(l) - \mathbf{f}_C(c) \|^2 + \sum_{(l, c) \in \mathcal{P}^-} \max(0, m - \| \mathbf{f}_L(l) - \mathbf{f}_C(c) \|)^2
$$

여기서 $\mathbf{f}_L$은 LiDAR 인코더, $\mathbf{f}_C$는 Camera 인코더, $\mathcal{P}^+$/$\mathcal{P}^-$는 positive/negative 쌍이다.

### 9.4.4 ModaLink

**[ModaLink (Xie et al., IROS 2024)](https://arxiv.org/abs/2403.18762)**는 카메라 이미지로 LiDAR 포인트 클라우드 데이터베이스를 검색하는 장소 인식을 빠르게 하려는 방법이다.

### 9.4.5 Modality-Agnostic Descriptor 접근

**모달리티에 무관한(modality-agnostic) 디스크립터**는 어떤 센서로 보든 같은 장소에서 같은 디스크립터가 나오게 하려는 것이다. 다음과 같은 방법이 있다:

- **Knowledge Distillation**: 정보가 많은 쪽(LiDAR+Camera)의 디스크립터를 teacher로 두고, 센서 하나만 쓰는 student 네트워크가 이를 따라 배우게 한다
- **Canonical Representation**: BEV나 semantic layout처럼 어느 센서에서나 만들 수 있는 표현으로 바꾼 뒤 비교한다
- **Foundation Model 기반**: [DXPR (Han et al., arXiv 2026)](https://arxiv.org/abs/2609.09005)은 카메라 이미지와 LiDAR 스캔을 모두 depth 이미지로 바꾼 뒤, vision foundation model backbone 하나로 두 센서에 공통인 global descriptor를 학습한다

---

## 9.5 Multi-Session & Long-Term Place Recognition

### 9.5.1 Long-Term VPR의 도전

같은 장소도 시간이 지나면 겉모습이 크게 바뀐다. 낮과 밤의 조명, 푸른 나무와 눈 덮인 풍경 같은 계절, 맑은 날과 안개 낀 날 같은 날씨, 건물 신축·철거나 도로 공사 같은 구조 변화가 모두 같은 장소의 디스크립터를 바꿔 PR 성능을 떨어뜨린다.

### 9.5.2 계절/시간/날씨 변화 대응 전략

**학습 데이터에 여러 조건 넣기**: 학습할 때 여러 조건에서 찍은 이미지를 넣는다. NetVLAD가 같은 장소를 여러 시기에 찍은 Google Street View Time Machine 이미지로 학습한 것이 그 예다.

**Domain Invariant Feature**: 겉모습이 바뀌어도 변하지 않는 특징을 학습한다. 건물, 도로, 하늘의 배치 같은 시맨틱 세그멘테이션 결과는 조명이 바뀌어도 비교적 덜 흔들린다. 다만 야간이나 역광처럼 세그멘테이션 자체가 부정확해지는 조건에서는 레이블 맵도 함께 흔들린다.

**Foundation model 특징 쓰기**: AnyLoc 실험에서 DINOv2 특징은 주야간·계절 변화에서도 기존 VPR 전용 모델보다 recall이 높았다(§9.2.4).

**시퀀스 매칭**: SeqSLAM처럼 이미지 한 장의 겉모습이 달라져도 이어지는 순서로 맞춘다.

### 9.5.3 Map Update 전략

오래 운용하는 시스템에서는 맵 자체도 고쳐 나가야 한다:

- **경험 기반 맵(Experience-Based Map)**: 같은 장소를 여러 시간대에 본 관측을 모두 저장해 두고, 지금 조건과 가장 비슷한 관측과 맞춘다. 대신 저장한 관측이 늘수록 맵이 커진다.

- **적응형 디스크립터**: 새 관측이 들어올 때마다 그 장소의 디스크립터를 조금씩 고친다. exponential moving average가 한 예다.

- **Change Detection**: 건물 철거 같은 구조 변화를 찾아내 그 영역의 맵을 버리고 다시 만든다.

### 9.5.4 Lifelong Place Recognition

로봇을 수명이 다할 때까지 운용하면서 장소 인식을 유지하려면, 맵이 끝없이 커지지 않게 정보를 압축해야 하고, 오래된 관측은 조금씩 잊으면서(graceful forgetting) 새 환경은 이전 지식을 한꺼번에 잃지 않고(catastrophic forgetting 없이) 배워야 한다. continual learning, incremental learning 연구와 맞닿은 문제다.

---

## 9.6 Geometric Verification & Re-ranking

Place recognition이 찾은 후보가 정말 같은 장소인지는 기하학적으로 확인한다.

### 9.6.1 PnP + RANSAC (Visual)

카메라로 찾은 후보는 이렇게 확인한다:

1. 쿼리 이미지와 후보 이미지 사이의 로컬 특징 매칭 (SuperPoint+LightGlue, ORB+BF 등)
2. **PnP (Perspective-n-Point)**: 1단계 매칭에서 후보 이미지 쪽 특징점을 그에 연결된 3D 맵 포인트로 바꿔 2D-3D 대응을 만들고, 이 대응으로 카메라 포즈를 추정한다. 3D 점이 맵 좌표계에 있으면 결과는 쿼리 카메라의 절대 포즈다. 후보와의 상대 포즈는 후보의 알려진 포즈와 합성해 얻는다.
3. **RANSAC**: 아웃라이어를 걸러 내면서 포즈를 추정한다.

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

LiDAR로 찾은 후보는 이렇게 확인한다:

1. 두 포인트 클라우드 사이 3D-3D 대응점을 찾는다(FPFH, FCGF, GeoTransformer 등).
2. **RANSAC**이나 RANSAC이 필요 없는 GeoTransformer 방식으로 강체 변환을 추정한다.
3. **ICP**로 정밀하게 맞춘다(coarse-to-fine).

GeoTransformer(Qin et al., CVPR 2022)는 점 쌍 사이 거리와 세 점이 이루는 각을 인코딩해, 겹치는 부분이 적은 경우에도 대응점을 정확히 찾는다. 대응이 충분히 정확해서 변환 추정에 RANSAC을 쓰지 않으며, 논문은 이 덕분에 100배 빨라졌다고 보고했다. 3DLoMatch benchmark에서는 inlier 비율이 17~30%p, registration recall이 7%p 넘게 올랐다. 100배는 논문의 구현과 하드웨어에서 잰 값이다.

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

디스크립터 유사도로만 정한 후보 순위를 기하 정보로 다시 매기는 방법들이다.

**공간적 근접성 prior**: SLAM이 추정한 현재 포즈 근처에 있는 후보에 점수를 더 준다. 드리프트가 작을 때만 쓸 수 있다.

**Inlier 수 기반 재랭킹**: 특징을 매칭한 뒤 인라이어가 많은 후보를 위로 올린다.

**포즈 일관성 검사**: 추정한 상대 포즈가 odometry로 쌓은 포즈와 맞는지 본다. 판정에는 odometry 공분산으로 정규화한 Mahalanobis 거리를 쓴다. 긴 루프에서는 올바른 검출도 쌓인 드리프트만큼 어긋나 보이므로, 공분산으로 정규화하지 않으면 쓸모 있는 루프까지 버린다.

---

## 9.7 최신 동향

### 9.7.1 Foundation Model 기반 PR의 부상

VPR 연구는 환경마다 따로 학습하는 방법과, foundation model을 추가 학습 없이(zero-shot) 쓰는 방법을 함께 다룬다. AnyLoc은 VPR 전용 학습 없이 논문이 고른 여러 환경에서 기존 방법과 겨룰 만한 결과를 냈다.

DINOv2의 ViT-g는 파라미터가 11억 개라 임베디드 기기에 올리기 어렵다. SALAD(§9.7.4)는 이보다 작은 DINOv2 ViT-B를 backbone으로 쓰고 마지막 네 블록만 파인튜닝한다. foundation model의 패치 단위 특징을 re-ranking에 쓰기도 한다. EffoVPR(§9.7.4)이 self-attention 특징을 re-ranking에 쓴 것이 그 예다.

### 9.7.2 Semantic Place Recognition

장소를 "빨간 건물 옆의 큰 나무"처럼 사물과 그 배치(spatial layout)로 나타내는 방법이다. 시맨틱 세그멘테이션 결과의 배치를 비교하거나, object detection으로 scene graph를 만들거나, CLIP으로 자연어 설명을 검색하는 방식이 있다. 잎 색깔이 바뀌어도 "나무"라는 레이블은 그대로이므로 조명·계절 변화에 덜 흔들린다. 대신 세그멘테이션이 틀리면 같이 틀리고, 비슷한 구조의 주택가처럼 사물 배치가 닮은 장소는 구별하기 어렵다.

### 9.7.3 4D Radar Place Recognition

높이(elevation)까지 재는 4D imaging radar가 나오면서 radar로 장소를 인식하는 연구도 생겼다. radar 포인트 클라우드를 Scan Context를 변형한 디스크립터로 만들거나, range-Doppler 이미지를 CNN에 넣어 글로벌 디스크립터를 만들거나, radar와 카메라 사이의 cross-modal PR을 하는 식이다.

radar는 비·눈·안개 속에서도 측정이 되므로, 이런 날씨에 LiDAR나 카메라 PR이 실패할 때 대신 쓸 후보가 된다. 다만 점이 성기고 해상도가 낮다.

### 9.7.4 최근 연구 (2024-2025)

**[SALAD (Izquierdo & Civera, 2024)](https://arxiv.org/abs/2311.15937)**: NetVLAD의 soft assignment를 optimal transport 문제로 다시 세우고, 정보가 적은 특징을 버리는 'dustbin' 클러스터를 둔다. backbone은 DINOv2를 파인튜닝해서 쓴다. 논문은 이 단일 단계 방법이 공개 VPR 데이터셋에서 다른 단일 단계 방법뿐 아니라 re-ranking을 더한 2단계 방법보다도 나았다고 보고한다(CVPR 2024).

**[EffoVPR (Tzachor et al., ICLR 2025)](https://arxiv.org/abs/2405.18065)**: DINOv2 같은 foundation model의 self-attention layer 특징을 추가 학습 없이 re-ranking에 쓴다. ViT 내부 layer로 pooling한 글로벌 특징은 128차원까지 줄여도 높은 성능을 냈다고 보고한다.

### 9.7.5 기술 계보 요약

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

