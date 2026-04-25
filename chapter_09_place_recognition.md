# Ch.9 — Place Recognition & Retrieval

Ch.6-8에서 다룬 odometry/fusion 시스템은 시간이 지나면 드리프트가 누적된다. 이 드리프트를 교정하려면 로봇이 과거에 방문한 장소를 다시 인식할 수 있어야 한다 — 이것이 Place Recognition의 역할이다. 이 챕터에서 다루는 기술들은 Ch.10의 Loop Closure에서 직접 활용된다.

> 로봇이 "이 장소를 전에 본 적이 있는가?"를 판단하는 문제.
> Loop closure의 핵심 컴포넌트이자, multi-session SLAM과 재위치추정(relocalization)의 기반이다.

---

## 9.1 문제 정의

### 9.1.1 Place Recognition vs Loop Closure Detection

**Place Recognition (PR)**은 "현재 관측이 데이터베이스의 어떤 장소와 일치하는가?"를 판단하는 retrieval 문제이다. 이것은 standalone 문제로, SLAM 없이도 정의된다.

**Loop Closure Detection**은 SLAM 시스템 내에서 "로봇이 이전에 방문했던 장소를 다시 방문했는가?"를 탐지하는 것이다. Place recognition은 loop closure detection의 핵심 컴포넌트이지만, loop closure는 PR 이후에 **geometric verification**(기하학적 검증)까지 포함하는 더 넓은 파이프라인이다.

```
Loop Closure Detection Pipeline:
┌─────────────┐    ┌────────────────┐    ┌───────────────────┐    ┌─────────────┐
│  현재 관측   │───→│ Place          │───→│ Geometric         │───→│ Pose Graph  │
│  (query)     │    │ Recognition    │    │ Verification      │    │ Update      │
│              │    │ (후보 검색)     │    │ (기하학적 검증)    │    │ (최적화)     │
└─────────────┘    └────────────────┘    └───────────────────┘    └─────────────┘
```

SLAM에서 loop closure가 없으면 드리프트가 계속 누적된다. 그런데 brute-force로 현재 프레임을 모든 과거 프레임과 비교하는 것은 $O(N^2)$으로 불가능하다. Place recognition은 이 비교를 sub-linear (보통 $O(\log N)$ 또는 $O(1)$)로 줄여주는 핵심 기술이다.

### 9.1.2 Retrieval Pipeline

Place recognition의 일반적인 파이프라인은 정보 검색(Information Retrieval)의 패러다임을 따른다:

1. **Encoding**: 각 관측(이미지, 포인트 클라우드, 또는 둘 다)을 고정 길이의 **글로벌 디스크립터(global descriptor)**로 인코딩.
2. **Indexing**: 데이터베이스의 모든 디스크립터를 검색 가능한 인덱스 구조(예: kd-tree, FAISS)에 저장.
3. **Retrieval**: 쿼리 디스크립터와 가장 유사한 데이터베이스 디스크립터를 검색.
4. **Re-ranking & Verification**: 후보들을 기하학적으로 검증하여 최종 매칭을 확정.

$$
q^* = \arg\min_{q \in \mathcal{D}} d(\mathbf{f}(\text{query}), \mathbf{f}(q))
$$

여기서 $\mathbf{f}(\cdot)$는 관측을 글로벌 디스크립터로 변환하는 함수, $d(\cdot, \cdot)$는 거리 함수(보통 L2 또는 코사인 거리), $\mathcal{D}$는 데이터베이스이다.

### 9.1.3 평가 메트릭

PR 시스템의 성능은 다음 메트릭으로 평가한다:

**Recall@N**: 상위 $N$개 후보 중 올바른 매칭이 포함되는 비율. 가장 보편적인 메트릭이다.

$$
\text{Recall@N} = \frac{|\{q : \text{top-}N \text{ 후보 중 정답이 있는 쿼리 } q\}|}{|\text{전체 쿼리}|}
$$

**Recall@1**이 특히 중요한 이유: 실시간 SLAM에서는 보통 가장 유사한 1개의 후보만 검증할 여유가 있기 때문이다.

**Precision-Recall Curve**: 디스크립터 유사도에 임계값을 변화시키며 precision과 recall의 관계를 그린다. 높은 precision (false positive 최소화)이 SLAM에서 특히 중요하다 — false positive loop closure는 맵을 파괴적으로 왜곡시킨다.

**"정답"의 정의**: 보통 GPS 거리 기준 25m 이내를 같은 장소로 정의한다. 데이터셋에 따라 다르다.

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

Visual Place Recognition은 이미지만으로 장소를 인식하는 문제이다. 가장 오래되고 가장 활발히 연구되는 PR 분야이다.

### 9.2.1 전통적 방법: Bag of Words (BoW)

**[Video Google (Sivic & Zisserman, 2003)](https://www.robots.ox.ac.uk/~vgg/publications/2003/Sivic03/)**이 제안한 **Bag of Visual Words** 모델은 VPR의 원점이다. 핵심 아이디어는 텍스트 검색(text retrieval)의 방법론을 시각 검색에 그대로 적용하는 것이다.

파이프라인:

1. **시각 어휘 구축(Visual Vocabulary Construction)**: 대량의 이미지에서 로컬 특징(SIFT, ORB 등)을 추출하고, 이 특징 기술자들을 k-means로 $K$개 클러스터로 그룹화한다. 각 클러스터 중심이 하나의 "시각 단어(visual word)"가 된다.

2. **이미지 표현**: 각 이미지에서 추출한 특징들을 가장 가까운 시각 단어에 할당(hard assignment)하고, 각 시각 단어의 등장 빈도를 세어 이미지를 $K$차원 히스토그램으로 표현한다.

3. **TF-IDF 가중**: 텍스트 검색에서 차용한 TF-IDF(Term Frequency – Inverse Document Frequency) 가중을 적용한다:

$$
w_{i,d} = \underbrace{\frac{n_{i,d}}{n_d}}_{\text{TF}} \cdot \underbrace{\log \frac{N}{N_i}}_{\text{IDF}}
$$

여기서 $n_{i,d}$는 이미지 $d$에서 시각 단어 $i$의 등장 횟수, $n_d$는 이미지 $d$의 전체 시각 단어 수, $N$은 데이터베이스의 전체 이미지 수, $N_i$는 시각 단어 $i$가 등장하는 이미지 수이다.

- **TF**: 해당 이미지에서 자주 등장하는 단어에 높은 가중치 → 이 이미지의 특징적 요소
- **IDF**: 전체 데이터베이스에서 희귀한 단어에 높은 가중치 → 변별력이 높은 요소

4. **역색인(Inverted Index)**: 각 시각 단어가 등장하는 이미지 목록을 미리 구축하여, 쿼리 시 전체 데이터베이스를 스캔하지 않고 해당 단어가 포함된 이미지만 빠르게 검색한다.

5. **유사도 랭킹**: 쿼리 이미지와 데이터베이스 이미지의 TF-IDF 벡터 간 코사인 유사도로 랭킹:

$$
\text{sim}(\mathbf{v}_q, \mathbf{v}_d) = \frac{\mathbf{v}_q \cdot \mathbf{v}_d}{\|\mathbf{v}_q\| \cdot \|\mathbf{v}_d\|}
$$

**DBoW2**: ORB-SLAM 시리즈에서 사용되는 BoW 구현이다. Hierarchical k-means tree를 사용하여 vocabulary 크기를 크게 늘리면서도 양자화 속도를 유지한다. 실시간 SLAM에서 loop closure 탐지의 사실상 표준이었다.

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
        self.db_vectors = {}  # img_id -> tf-idf vector
    
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
        word_ids = self.quantize(descriptors)
        
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
        bow_vector = self.compute_bow_vector(descriptors)
        self.db_vectors[img_id] = bow_vector
        self.N += 1
        
        # 역색인 업데이트
        word_ids = self.quantize(descriptors)
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
        
        scores = {}
        for img_id, db_vector in self.db_vectors.items():
            scores[img_id] = np.dot(q_vector, db_vector)
        
        # 유사도 순으로 정렬
        sorted_results = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_results[:top_k]
```

BoW의 한계는 세 가지다. 첫째, 양자화 과정에서 정보 손실이 크다(비슷한 특징이 같은 단어로 뭉침). 둘째, 조명·시점 변화에 대한 불변성이 기저 특징(SIFT, ORB)에 의존한다. 셋째, vocabulary 크기 $K$ 선택이 성능에 큰 영향을 미친다.

### 9.2.2 VLAD와 Fisher Vector

BoW의 한계를 극복하기 위해, 단순 빈도 히스토그램 대신 **잔차(residual) 정보**를 보존하는 집계 방법들이 등장했다.

**[VLAD (Vector of Locally Aggregated Descriptors, Jégou et al., 2010)](https://doi.org/10.1109/CVPR.2010.5540039)**:

BoW가 "어떤 시각 단어가 몇 번 등장했는가"만 기록하는 반면, VLAD는 "각 시각 단어와 실제 기술자 사이의 잔차가 어떤 방향으로 얼마나 큰가"를 기록한다.

각 클러스터 $k$에 대해, 해당 클러스터에 할당된 기술자들과 클러스터 중심의 잔차를 합산한다:

$$
\mathbf{V}_k = \sum_{i: \text{NN}(\mathbf{x}_i) = k} (\mathbf{x}_i - \mathbf{c}_k)
$$

최종 VLAD 디스크립터는 모든 $\mathbf{V}_k$를 연결(concatenate)한 벡터이다: $\mathbf{V} = [\mathbf{V}_1^T, \mathbf{V}_2^T, \ldots, \mathbf{V}_K^T]^T$. 차원은 $K \times D$이다.

**Fisher Vector**: VLAD보다 더 풍부한 표현. 시각 단어를 GMM (Gaussian Mixture Model)로 모델링하고, 각 가우시안 컴포넌트에 대한 1차/2차 통계량을 Fisher Information Matrix의 제곱근으로 정규화한다. 차원이 $2KD$로 VLAD의 2배지만, 일반적으로 더 높은 성능을 보인다.

### 9.2.3 NetVLAD: 학습 기반 VPR의 기준점

**[NetVLAD (Arandjelović et al., 2016)](https://arxiv.org/abs/1511.07247)**은 VLAD를 미분 가능한(differentiable) CNN 레이어로 재구성하여, end-to-end 학습을 가능하게 했다.

전통적 VLAD에서 각 기술자를 가장 가까운 클러스터에 hard assignment하는 것은 미분 불가능하다. 역전파를 위해서는 이 과정이 미분 가능해야 한다.

**해결: Soft Assignment**:

Hard assignment($\text{NN}(\mathbf{x}_i) = k$이면 1, 아니면 0)를 softmax로 대체한다:

$$
\bar{a}_k(\mathbf{x}_i) = \frac{e^{\mathbf{w}_k^T \mathbf{x}_i + b_k}}{\sum_{k'} e^{\mathbf{w}_{k'}^T \mathbf{x}_i + b_{k'}}}
$$

원래 VLAD의 유클리드 거리 기반 할당 $e^{-\alpha\|\mathbf{x}_i - \mathbf{c}_k\|^2}$을 전개하면, $\mathbf{w}_k = 2\alpha\mathbf{c}_k$, $b_k = -\alpha\|\mathbf{c}_k\|^2$로 해석할 수 있다. 이 파라미터들을 학습 가능하게 하면 클러스터 중심이 데이터에 맞게 조정된다.

**NetVLAD 레이어 출력**:

$$
\mathbf{V}(j, k) = \sum_{i=1}^{N} \bar{a}_k(\mathbf{x}_i) (x_i(j) - c_k(j))
$$

이 $D \times K$ 행렬을 L2 정규화하고 벡터로 펼치면 최종 글로벌 디스크립터가 된다.

Google Street View Time Machine 데이터를 활용한 약한 지도 학습(weakly supervised learning)으로 훈련된다. 같은 GPS 좌표의 다른 시간대 이미지를 positive pair, 먼 GPS 좌표의 이미지를 negative로 사용한다. Triplet ranking loss:

$$
\mathcal{L} = \sum_{(q, p^+, p^-)} \max\left(0, m + d(\mathbf{f}(q), \mathbf{f}(p^+)) - d(\mathbf{f}(q), \mathbf{f}(p^-))\right)
$$

여기서 $m$은 마진, $p^+$는 positive 이미지, $p^-$는 hard negative 이미지이다.

Pitts250k Recall@1 약 84.3%, Pitts30k Recall@1 약 86.3%로, 당시 hand-crafted 방법 대비 향상. VGG-16 백본 사용.

### 9.2.4 AnyLoc: Foundation Model 기반 범용 VPR

**[AnyLoc (Keetha et al., 2023)](https://arxiv.org/abs/2308.00688)**은 VPR 전용 학습 없이 범용적으로 작동하는 장소 인식이 가능한가를 묻는다. 대답은 DINOv2 같은 Foundation Model의 특징을 사용하면 가능하다는 것이다.

1. **DINOv2 특징 추출**: DINOv2 ViT-G14의 중간 레이어(31번째 레이어)에서 밀집(dense) 특징을 추출한다. CLS 토큰(이미지 전체를 하나의 벡터로 요약)이 아니라 모든 패치의 특징을 사용한다. CLS 토큰은 이미지 전체의 semantic을 포착하지만 장소를 구분하는 세밀한 구조적 차이를 놓칠 수 있다. 밀집 특징은 각 패치의 로컬 정보를 보존하므로 더 정밀한 장소 구분이 가능하다 (평균 23% 성능 향상).

2. **VLAD 집계**: 밀집 특징을 k-means로 클러스터링하여 시각 어휘를 구축하고, hard-assignment VLAD로 글로벌 디스크립터를 생성한다. NetVLAD처럼 학습하지 않고 비지도(unsupervised) VLAD를 사용한다.

3. **도메인별 어휘**: PCA 투영으로 Urban, Indoor, Aerial, SubT(지하), Degraded(악조건), Underwater(수중) 6개 도메인을 비지도적으로 발견하고, 도메인별 어휘를 구축하면 성능이 최대 19% 더 향상된다.

Foundation Model이 작동하는 이유:

[DINOv2 (Oquab et al., 2023)](https://arxiv.org/abs/2304.07193)는 1.42억 장의 이미지에서 자기지도 학습(self-supervised learning)으로 학습한 모델이다. 훈련 과정에서 장면의 구조적·시맨틱 특징을 범용적으로 익힌다. 특정 장소 인식 태스크에 학습하지 않았음에도 이 범용 특징이 장소를 구분하는 데 충분하다.

벤치마크 성능:
- 주야간 변화: 기존 SOTA(MixVPR, CosPlace) 대비 5-21% Recall@1 향상
- 계절 변화: 8-9% 향상
- 반대 시점(180도): 39-49% 향상
- 비정형 환경(수중, 지하): 기존 대비 최대 4배
- PCA-Whitening으로 49K 차원 → 512 차원 (100배 압축)하면서 SOTA 성능 유지

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
        # layer 31의 value facet 사용 (AnyLoc의 핵심 발견)
        image_patch_features.append(patch_tokens)
        all_patch_features.append(patch_tokens)
    
    # Step 2: k-means 시각 어휘 구축
    all_features = np.vstack(all_patch_features)  # (N_total_patches, D_feat)
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(all_features)
    centers = kmeans.cluster_centers_  # (K, D_feat)
    
    # Step 3: 각 이미지에 대해 VLAD 디스크립터 계산
    vlad_descriptors = []
    D_feat = centers.shape[1]
    
    for patches in image_patch_features:
        # Hard assignment
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

**[EigenPlaces (Berton et al., 2023)](https://arxiv.org/abs/2308.10832)**는 [CosPlace (Berton et al., 2022)](https://arxiv.org/abs/2204.02287)의 후속작으로, PCA 기반 차원 축소를 학습 과정에 통합한 방법이다. CosPlace가 분류(classification) 기반 학습으로 triplet loss의 번거로운 hard negative mining을 대체한 것에서 더 나아가, 특징 공간의 구조를 PCA 관점에서 최적화한다.

### 9.2.6 Foundation Model 활용의 확장

DINOv2 외에도 다양한 Foundation Model이 VPR에 활용되고 있다:

- **CLIP**: 텍스트-이미지 대응 학습으로, "도시 거리", "숲 속 길" 같은 시맨틱 수준의 장소 인식에 활용 가능. 그러나 세밀한 구조적 차이 구분에서는 DINOv2에 뒤진다.
- **SAM (Segment Anything)**: 세그멘테이션 마스크를 장소의 구조적 표현으로 활용하는 연구가 진행 중.
- **DINOv2 + NetVLAD**: AnyLoc의 비지도 VLAD 대신, DINOv2 특징에 학습된 NetVLAD 레이어를 붙이면 성능이 더 향상된다는 후속 연구.

### 9.2.7 SeqSLAM과 시퀀스 매칭

**[SeqSLAM (Milford & Wyeth, 2012)](https://doi.org/10.1109/ICRA.2012.6224623)**은 개별 이미지 매칭 대신 **이미지 시퀀스 전체의 매칭**을 택했다.

단일 이미지로는 구분이 어려운 장소(예: 비슷하게 생긴 주택가)도, 연속된 이미지들의 패턴(좌회전 → 공원 → 건널목 순서)은 고유할 수 있다.

방법은 세 단계다. 첫째, 각 이미지를 극도로 간소화된 표현(저해상도 패치 또는 단순 디스크립터)으로 변환한다. 둘째, 쿼리 시퀀스와 데이터베이스 시퀀스의 **시퀀스 유사도 행렬(sequence similarity matrix)**을 구성한다. 셋째, 그 위에서 일정 속도 범위 내의 대각선 경로를 탐색하여 최적 시퀀스 매칭을 찾는다.

$$
S_{\text{seq}}(q_s, d_s) = \min_{\text{path}} \sum_{i=0}^{L-1} d(\mathbf{f}(q_{s+i}), \mathbf{f}(d_{s+\delta(i)}))
$$

여기서 $L$은 시퀀스 길이, $\delta(i)$는 속도 변화를 허용하는 경로 함수이다.

주간 → 야간, 여름 → 겨울 같은 외관 변화에서도 시퀀스 패턴이 보존되므로 강건하다. SeqNet (Garg et al., 2021)은 이 시퀀스 매칭을 학습 기반으로 발전시켰다.

---

## 9.3 LiDAR Place Recognition

3D LiDAR 포인트 클라우드로 장소를 인식하는 문제이다. 카메라와 달리 조명 변화에 영향을 받지 않지만, 계절에 따른 식생 변화나 시점(viewpoint) 변화에 취약하다.

### 9.3.1 Handcrafted 방법: Scan Context

**[Scan Context (Kim & Kim, 2018)](https://doi.org/10.1109/IROS.2018.8593953)**은 3D LiDAR 스캔을 histogram 없이 **공간 구조를 직접 보존하는 글로벌 디스크립터**로 변환한다. 학습이 불필요하고, LIO-SAM 등 주요 시스템에 loop closure 모듈로 널리 채택되어 있다.

디스크립터 생성 과정:

1. **극좌표 분할**: 센서 중심에서 바라보는 2D 평면을 방위각(azimuth) $N_s$개 섹터와 거리(range) $N_r$개 링으로 분할하여 $N_s \times N_r$ 그리드를 만든다 (보통 60×20).

2. **최대 높이 인코딩**: 각 빈(bin)에 속하는 3D 점들 중 최대 높이(max height)를 기록한다. 이것이 Scan Context 행렬 $\mathbf{SC} \in \mathbb{R}^{N_s \times N_r}$이다.

$$
\mathbf{SC}(i, j) = \max_{p \in \text{bin}(i,j)} z(p)
$$

평균 높이보다 최대 높이가 건물, 나무, 기둥 같은 돌출 구조물을 더 잘 포착한다.

3. **Egocentric 표현의 장점**: 센서 중심 좌표계에서 표현하므로, 같은 장소를 **반대 방향**에서 재방문해도 디스크립터의 행(row, 즉 섹터 축)이 순환 이동(circular shift)된 것에 불과하다. 이를 행 이동 매칭으로 처리한다.

**검색 전략 — Ring Key & Sector Key**:

전체 데이터베이스에 대해 Scan Context 행렬을 직접 비교하는 것은 비효율적이다. 2단계 검색으로 효율화한다:

1. **Ring Key**: SC 행렬의 각 링(열)에 대해 섹터 방향으로 평균한 값을 벡터로 추출 — $\mathbf{k}_r = [\bar{h}_1, \bar{h}_2, \ldots, \bar{h}_{N_r}]$. 이 벡터로 kd-tree 검색하여 후보를 빠르게 좁힌다.

2. **Sector Key**: 후보들에 대해, SC 행렬의 행(섹터 축) 이동을 시도하며 최적 매칭을 찾는다:

$$
d(\mathbf{SC}_q, \mathbf{SC}_d) = \min_{s \in [0, N_s)} \left\| \mathbf{SC}_q - \text{shift}(\mathbf{SC}_d, s) \right\|_F
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
        
        # Stage 2: Scan Context 열 이동 매칭
        scores = []
        for idx in candidate_indices:
            sc_db = self.database[idx]
            min_dist = float('inf')
            for shift in range(self.n_sectors):
                sc_shifted = np.roll(sc_db, shift, axis=0)
                dist = np.linalg.norm(sc_query - sc_shifted)
                min_dist = min(min_dist, dist)
            scores.append((idx, min_dist))
        
        scores.sort(key=lambda x: x[1])
        return scores[:top_k]
```

**후속작 — Scan Context++**: 시맨틱 세그멘테이션을 추가하여, 높이 대신 시맨틱 레이블(건물, 도로, 식생 등)을 인코딩. 시맨틱 정보는 계절 변화에 더 강건하다.

### 9.3.2 M2DP와 ESF

Scan Context 외의 handcrafted LiDAR 디스크립터:

- **M2DP (He et al., 2016)**: 포인트 클라우드를 여러 2D 평면에 투영하고, 각 투영의 밀도 분포를 SVD로 압축하여 192차원 디스크립터를 생성. 방향 불변(rotation invariant).
- **ESF (Ensemble of Shape Functions)**: 점쌍 간 거리, 각도, 면적 비율의 히스토그램을 조합한 디스크립터.

이들은 Scan Context에 비해 공간 구조 보존이 약하지만, 회전 불변성이나 간결함에서 장점이 있다.

### 9.3.3 Learning-based: PointNetVLAD

**[PointNetVLAD (Uy & Lee, 2018)](https://arxiv.org/abs/1804.03492)**는 NetVLAD의 아이디어를 3D 포인트 클라우드에 직접 적용한 최초의 연구이다. PointNet 백본으로 로컬 특징을 추출하고, NetVLAD 레이어로 글로벌 디스크립터로 집계하며, lazy triplet loss로 학습한다.

$$
\mathcal{L} = \max(0, m + \max_{p^+} d(q, p^+) - \min_{p^-} d(q, p^-))
$$

PointNet은 포인트 간 상호작용을 충분히 모델링하지 못하며, 대규모 포인트 클라우드 처리에 비효율적이다.

### 9.3.4 MinkLoc3D

**[MinkLoc3D (Komorowski, 2021)](https://arxiv.org/abs/2011.04530)**는 PointNetVLAD의 한계를 극복하기 위해 Minkowski Convolutional Neural Network을 백본으로 사용한다. 희소 3D 합성곱(sparse 3D convolution)을 통해 포인트 클라우드의 로컬 구조를 효과적으로 포착하며, GeM (Generalized Mean) 풀링으로 글로벌 디스크립터를 생성한다.

### 9.3.5 OverlapTransformer: Range Image 기반

**[OverlapTransformer (Ma et al., 2022)](https://arxiv.org/abs/2203.03397)**는 LiDAR 포인트 클라우드를 **range image**로 변환하여, 2D 이미지 처리 파이프라인을 활용하는 접근법이다.

**Range Image**: 회전형 LiDAR 스캔을 $(h, w)$ 크기의 2D 이미지로 변환한다. 각 픽셀 값은 해당 방향의 거리(range)이다. $h$는 레이저 빔 수, $w$는 수평 해상도에 대응한다.

아키텍처는 3단계로 구성된다. 첫째, range image를 lightweight CNN으로 처리하여 feature map을 추출한다. 둘째, NetVLAD 레이어로 글로벌 디스크립터를 생성한다. 셋째, Transformer encoder로 전체적인 context를 반영한다. 3D 포인트 클라우드 처리보다 2D CNN 처리가 빠르고, 기존 이미지 네트워크 아키텍처를 그대로 활용할 수 있다는 점이 이 방식의 실용적 장점이다.

### 9.3.6 BEV 기반 방법

Bird's Eye View(BEV)로 포인트 클라우드를 투영하여 2D 맵으로 변환한 뒤, 2D 이미지 기반 검색을 수행하는 방법들:

- **OverlapNet**: BEV 투영 이미지의 오버랩 정도를 직접 예측
- **BEVPlace**: BEV 이미지에서 NetVLAD 디스크립터를 추출

---

## 9.4 Cross-Modal Place Recognition

### 9.4.1 왜 Cross-Modal PR이 필요한가

실제 로봇 시스템에서는 **매핑 시의 센서와 로컬라이제이션 시의 센서가 다를 수 있다**. 예:
- LiDAR로 만든 맵에서 카메라만으로 위치를 찾아야 하는 경우
- 자율주행 차량(LiDAR+Camera)과 배달 로봇(Camera만) 간의 장소 인식
- 드론(Camera)이 차량(LiDAR)이 만든 맵에서 자신의 위치를 파악

이런 시나리오에서는 **LiDAR 관측과 카메라 관측 사이의 장소 인식**, 즉 cross-modal PR이 필요하다.

### 9.4.2 Cross-Modal PR의 근본적 어려움: Domain Gap

LiDAR 포인트 클라우드와 카메라 이미지는 근본적으로 다른 표현(representation)이다:

| 특성 | LiDAR 포인트 클라우드 | 카메라 이미지 |
|------|---------------------|-------------|
| 데이터 구조 | 비정형 3D 점 집합 | 정형 2D 그리드 |
| 정보 | 기하(geometry) | 외관(appearance) |
| 조명 의존 | 없음 | 매우 높음 |
| 텍스처 | 없음 | 풍부 |
| 밀도 | 거리에 반비례 | 균일 |

이 근본적 차이를 **domain gap**이라 하며, 같은 장소를 다른 모달리티로 관측했을 때 디스크립터 공간에서의 거리가 멀어지는 원인이 된다.

### 9.4.3 (LC)²: LiDAR-Camera Cross-Modal PR

**[(LC)² (Lee et al., 2023)](https://arxiv.org/abs/2304.08660)**는 LiDAR 포인트 클라우드와 카메라 이미지를 **공통 디스크립터 공간(shared embedding space)**에 매핑하는 방법을 내놓았다.

LiDAR 포인트 클라우드를 range image/BEV image로 변환하여 2D 표현으로 통일하고, 카메라 이미지와 LiDAR 투영 이미지를 각각 CNN으로 처리한다. **Contrastive learning**으로 같은 장소의 LiDAR-Camera 쌍을 임베딩 공간에서 가깝게, 다른 장소의 쌍을 멀게 학습한다.

$$
\mathcal{L}_{\text{contrastive}} = \sum_{(l, c) \in \mathcal{P}^+} \| \mathbf{f}_L(l) - \mathbf{f}_C(c) \|^2 + \sum_{(l, c) \in \mathcal{P}^-} \max(0, m - \| \mathbf{f}_L(l) - \mathbf{f}_C(c) \|)^2
$$

여기서 $\mathbf{f}_L$은 LiDAR 인코더, $\mathbf{f}_C$는 Camera 인코더, $\mathcal{P}^+$/$\mathcal{P}^-$는 positive/negative 쌍이다.

### 9.4.4 ModaLink

**ModaLink**는 (LC)²보다 더 범용적인 cross-modal 프레임워크를 지향하며, LiDAR, Camera, Radar 등 다양한 모달리티 조합에 대응한다.

### 9.4.5 Modality-Agnostic Descriptor 접근

궁극적 목표는 **모달리티에 무관한(modality-agnostic) 디스크립터**이다. 어떤 센서로 관측하든 같은 장소는 같은 디스크립터를 생성하는 것이다. 이를 위한 접근법:

- **Knowledge Distillation**: 정보가 풍부한 모달리티(LiDAR+Camera)의 디스크립터를 teacher로, 단일 모달리티를 student로 학습
- **Canonical Representation**: BEV 또는 semantic layout 같은 모달리티 중립적 표현으로 변환 후 비교
- **Foundation Model 기반**: DINOv2 같은 FM이 이미지에서 추출하는 특징이 LiDAR를 이미지로 렌더링한 것에서 추출한 특징과 비슷한 공간에 놓이는지 탐구하는 연구

---

## 9.5 Multi-Session & Long-Term Place Recognition

### 9.5.1 Long-Term VPR의 도전

동일한 장소도 시간에 따라 외관이 크게 변한다. 조명(낮 vs 밤), 계절(초록 나무 vs 눈 덮인 풍경), 날씨(맑음 vs 안개), 구조적 변화(건물 신축·철거, 도로 공사) 모두 같은 장소의 디스크립터를 변동시켜 PR 성능을 저하시킨다.

### 9.5.2 계절/시간/날씨 변화 대응 전략

네 가지 접근이 병행된다.

**Data Augmentation**: 학습 시 다양한 조건의 이미지를 포함한다. NetVLAD가 Google Street View Time Machine을 활용한 것이 대표적이다.

**Domain Invariant Feature**: 외관 변화에 불변하는 특징을 학습한다. 시맨틱 세그멘테이션 결과(건물, 도로, 하늘의 배치)는 조명에 불변적이다.

**Foundation Model 활용**: AnyLoc에서 보여주었듯이, DINOv2의 특징은 조명·계절 변화에 강건하다. 자기지도 학습 과정에서 다양한 augmentation에 불변하는 특징을 학습하기 때문이다.

**시퀀스 매칭**: SeqSLAM 방식이다. 개별 이미지의 외관이 달라도 시퀀스 패턴은 유지되는 경향이 있다.

### 9.5.3 Map Update 전략

장기 운용 시스템에서는 맵 자체를 업데이트해야 한다:

- **경험 기반 맵(Experience-Based Map)**: 같은 장소의 여러 시간대 관측을 모두 저장하여, 현재 조건과 가장 유사한 경험에서 매칭. 맵 크기가 시간에 비례하여 증가하는 문제.

- **적응형 디스크립터**: 새로운 관측으로 기존 장소의 디스크립터를 점진적으로 업데이트. Exponential moving average 등.

- **Change Detection**: 구조적 변화(건물 철거 등)를 탐지하여 해당 맵 영역을 무효화하고 재구축.

### 9.5.4 Lifelong Place Recognition

평생 동안 운용되는 로봇의 장소 인식은 아직 열린 연구 문제다. 맵이 무한히 커지지 않도록 정보를 압축·관리해야 하고, 오래된 관측을 점진적으로 망각(graceful forgetting)하면서 새로운 환경을 catastrophic forgetting 없이 학습해야 한다. continual learning/incremental learning과 밀접하게 연관된다.

---

## 9.6 Geometric Verification & Re-ranking

Place recognition이 후보를 찾았다면, 그것이 **진짜 같은 장소인지 기하학적으로 검증**해야 한다. 이 단계가 없으면 false positive loop closure가 발생하여 맵이 파괴될 수 있다.

### 9.6.1 PnP + RANSAC (Visual)

카메라 기반 PR 후보에 대해:

1. 쿼리 이미지와 후보 이미지 사이의 로컬 특징 매칭 (SuperPoint+LightGlue, ORB+BF 등)
2. **PnP (Perspective-n-Point)**: 2D-3D 대응점으로부터 카메라의 상대 포즈를 추정
3. **RANSAC**: 아웃라이어를 제거하며 포즈 추정

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
    
    # 상대 포즈
    R, _ = cv2.Rodrigues(rvec)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = tvec.flatten()
    
    return is_verified, T, n_inliers
```

### 9.6.2 3D-3D Registration (LiDAR)

LiDAR 기반 PR 후보에 대해:

1. 두 포인트 클라우드 사이의 **3D-3D 대응점** 찾기 (FPFH, FCGF, GeoTransformer 등)
2. **RANSAC** 또는 **GeoTransformer의 RANSAC-free** 방식으로 강체 변환 추정
3. **ICP로 정밀 정합** (coarse-to-fine)

GeoTransformer (Qin et al., 2022)를 사용하면 RANSAC 없이도 강건한 정합이 가능하다. GeoTransformer는 쌍별 거리(pairwise distance)와 삼중 각도(triplet angle)를 인코딩하는 기하학적 트랜스포머로, rigid transformation에 불변한 특징을 학습하여 저오버랩 시나리오에서도 강건하다. 슈퍼포인트 수준의 대응에서 직접 변환을 추정하므로 RANSAC 대비 100배 빠르다.

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

디스크립터 유사도만으로 랭킹한 후보 목록을 기하학적 정보로 재랭킹하는 전략이 세 가지 있다.

**공간적 근접성 prior**: SLAM에서 현재 포즈 추정치 근처의 후보에 가산점을 부여한다. 드리프트가 작을 때만 유효하다.

**Inlier 수 기반 재랭킹**: 특징 매칭 후 인라이어 수가 많은 후보를 상위로 올린다.

**포즈 일관성 검사**: 추정된 상대 포즈가 odometry의 누적 포즈와 일관성이 있는지 확인한다. 큰 불일치가 있으면 false positive로 거부한다.

---

## 9.7 최신 동향

### 9.7.1 Foundation Model 기반 PR의 부상

VPR은 "환경별 전용 학습"에서 "범용 zero-shot 인식"으로 이동하고 있다. AnyLoc이 보여주었듯이, Foundation Model의 범용 특징이 VPR 전용 학습 없이도 대부분의 환경에서 SOTA를 달성한다.

향후 방향은 크게 두 가지다. 첫째, ViT-G14는 1B+ 파라미터로 임베디드 배포에 제약이 있어 경량 FM(ViT-S/B) + 도메인 적응의 조합이 연구되고 있다. 둘째, FM이 제공하는 패치 수준의 dense correspondence를 re-ranking이나 상대 포즈 추정에 직접 활용하는 연구가 진행 중이다.

### 9.7.2 Semantic Place Recognition

장소를 **시맨틱 구조**로 인식하는 접근:

"빨간 건물 옆의 큰 나무" → 건물(빨간)과 나무(큰)의 공간적 배치(spatial layout)로 장소를 표현

시맨틱 세그멘테이션으로 레이아웃을 비교하거나, object detection으로 scene graph를 구성하거나, CLIP으로 자연어 설명 기반 검색을 수행하는 방식이 있다. 시맨틱 구조는 조명·계절 변화에 강건하다. 나무의 잎 색깔이 변해도 "나무"라는 레이블은 유지된다. 다만 시맨틱 세그멘테이션의 정확도에 의존하며, 시맨틱 구조가 유사한 장소(비슷한 구조의 주택가)를 구분하기 어렵다.

### 9.7.3 4D Radar Place Recognition

4D imaging radar의 등장으로, radar 기반 PR도 연구되기 시작했다:

- Radar 포인트 클라우드를 Scan Context 변형으로 인코딩
- Range-Doppler image를 CNN으로 처리하여 글로벌 디스크립터 생성
- Radar-Camera cross-modal PR

비·눈·안개 환경에서 LiDAR PR과 Visual PR이 실패할 때 radar PR이 백업 역할을 할 수 있다. 다만 아직 초기 단계이며, radar의 낮은 해상도 때문에 장소 구별 능력(discriminability)이 LiDAR나 Visual에 비해 떨어진다. 4D radar의 해상도가 개선되고 있어 향후 주목할 분야다.

### 9.7.4 최근 연구 (2024-2025)

**[SALAD (Izquierdo & Civera, 2024)](https://arxiv.org/abs/2311.15937)**: NetVLAD의 feature-to-cluster 할당을 **optimal transport** 문제로 재정의하고, DINOv2를 백본으로 fine-tuning한다. Sinkhorn 알고리즘으로 soft assignment를 최적화하여 NetVLAD/CosPlace 대비 다수 벤치마크에서 SOTA를 달성한다.

**[EffoVPR (Taha et al., 2024)](https://arxiv.org/abs/2405.18065)**: DINOv2 등 Foundation Model의 특징을 효율적으로 활용하는 프레임워크. 128차원까지 압축된 디스크립터로도 SOTA 성능을 유지하며, 임베디드 배포에 유리한 경량 VPR을 보여준다.

### 9.7.5 기술 계보 요약

```
Visual Place Recognition 계보:

Sivic (2003) Video Google [BoW]
    ↓ 양자화 → 잔차로
Jégou (2010) VLAD
    ↓ hand-crafted → CNN end-to-end
Arandjelović (2016) NetVLAD [triplet, soft-assignment]
    ↓ triplet → classification
Berton (2022) CosPlace, (2023) EigenPlaces
    ↓ VPR-specific → Foundation Model
Keetha (2023) AnyLoc [DINOv2 + VLAD, zero-shot]
    ↓ assignment 최적화 + FM fine-tuning
Izquierdo (2024) SALAD [optimal transport + DINOv2]

LiDAR Place Recognition 계보:

Kim (2018) Scan Context [handcrafted, spatial]
    ↓ handcrafted → learned
Uy (2018) PointNetVLAD [PointNet + NetVLAD]
    ↓ PointNet → sparse convolution
Komorowski (2021) MinkLoc3D [sparse conv + GeM]
    ↓ 3D point → 2D range image
Ma (2022) OverlapTransformer [range image + transformer]

Cross-Modal:
Lee (2023) (LC)² [LiDAR ↔ Camera shared embedding]
```

---

## 9장 요약

Place Recognition은 SLAM 시스템에서 드리프트를 교정하는 loop closure의 핵심 컴포넌트이다. Visual PR은 BoW(Video Google) → VLAD → NetVLAD → AnyLoc으로 진화하며, Foundation Model(DINOv2) 기반의 범용 zero-shot 인식이 최근 패러다임이 되었다. LiDAR PR은 Scan Context(handcrafted) → PointNetVLAD → MinkLoc3D → OverlapTransformer로 발전하며, range image 기반 방법이 효율성 면에서 주목할 만하다.

Cross-modal PR은 domain gap이라는 근본적 어려움이 있으며, 공통 임베딩 공간 학습과 modality-agnostic 디스크립터가 연구되고 있다. Long-term PR은 계절/조명/구조적 변화에 대응해야 하며, Foundation Model의 강건한 특징이 이 문제에 유망하다.

Geometric verification은 PR 후보의 최종 검증 단계로, false positive를 방지하여 SLAM의 무결성을 보호한다. PnP+RANSAC(visual), ICP/GeoTransformer(LiDAR)가 표준적 방법이다.

최신 동향으로는 Foundation Model 기반 PR의 경량화, semantic PR, 4D radar PR이 활발히 연구되고 있다.

Place Recognition은 "이 장소를 본 적 있는가?"라는 질문에 답하지만, 그 답을 SLAM 시스템의 전역 일관성으로 전환하는 과정이 남아 있다. 다음 챕터에서는 PR 결과를 포즈 그래프에 통합하여 드리프트를 교정하는 **Loop Closure와 전역 최적화**를 다룬다.
