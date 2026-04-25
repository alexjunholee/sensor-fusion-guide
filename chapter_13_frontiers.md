# Ch.13 — Frontiers & Emerging Directions

Ch.2-12에서 센서 퓨전의 확립된 이론과 실전 시스템을 체계적으로 다루었다. 이 마지막 챕터에서는 시선을 미래로 돌린다.

센서 퓨전 분야는 빠르게 진화하고 있다. 이 챕터에서는 아직 완전히 성숙하지 않았지만, 향후 몇 년간 분야의 방향을 바꿀 수 있는 연구 프런티어를 다룬다. Foundation model의 공간 지능(spatial AI)으로의 확장, end-to-end 학습 SLAM, scene graph 기반의 환경 이해, cross-modal representation, 그리고 event camera와 4D radar라는 새로운 센서 모달리티의 퓨전이 그 주제다.

---

## 13.1 Foundation Models for Spatial AI

Foundation model — 대규모 데이터로 사전학습된 범용 모델 (DINOv2, CLIP, SAM, GPT-4V 등) — 이 센서 퓨전과 SLAM 파이프라인에 빠르게 유입되고 있다. 이 모델들은 특정 태스크를 위해 학습되지 않았음에도, 풍부한 시각적/의미론적 표현을 제공하여 전통 파이프라인의 여러 모듈을 대체하거나 강화한다.

### 13.1.1 DINOv2/CLIP Feature를 SLAM에 활용

**DINOv2의 시각 특징**: [DINOv2](https://arxiv.org/abs/2304.07193) (Oquab et al. 2023)는 자기지도학습(self-supervised learning)으로 훈련된 ViT로, 픽셀 수준의 dense feature를 제공한다. 이 feature는:

- 조명·계절이 달라도 (주간/야간, 여름/겨울) 같은 장소에서 유사한 feature를 생성한다.
- 같은 종류의 객체(예: 모든 "의자")에 유사한 feature를 부여하는 의미론적 인식 능력이 있다.
- 기하학적 구조(모서리, 평면 등)도 feature에 반영된다.

**AnyLoc의 접근**: [AnyLoc](https://arxiv.org/abs/2308.00688) (Keetha et al. 2023)은 DINOv2의 dense feature를 VLAD로 집계하여 글로벌 장소 디스크립터를 생성한다. 이 디스크립터는:

- 도심, 실내, 항공, 수중, 지하 등 다양한 환경에서 VPR 전용 학습 없이 동작한다.
- 기존 학습 기반 VPR (NetVLAD, CosPlace 등)을 다양한 도메인에서 능가한다.
- DINOv2의 31번째 레이어 value facet의 dense feature가 CLS 토큰보다 23% 더 좋은 성능을 보인다.

```python
import numpy as np

class FoundationModelFeatureExtractor:
    """
    DINOv2 기반 dense feature 추출의 개념적 구현.
    실제로는 torch + DINOv2 모델을 사용한다.
    """
    
    def __init__(self, model_name='dinov2_vitg14', layer=31, facet='value'):
        """
        Args:
            model_name: DINOv2 모델 이름
            layer: 추출할 레이어 (AnyLoc에서는 31)
            facet: 'key', 'query', 'value' 중 선택 (AnyLoc에서는 'value')
        """
        self.model_name = model_name
        self.layer = layer
        self.facet = facet
        # 실제로는 여기서 torch 모델을 로드
        # self.model = torch.hub.load('facebookresearch/dinov2', model_name)
        
    def extract_dense_features(self, image):
        """
        이미지에서 픽셀 수준 dense feature를 추출.
        
        Args:
            image: (H, W, 3) RGB 이미지
            
        Returns:
            features: (H', W', D) dense feature map
                      H' = H/14, W' = W/14 (patch size = 14)
                      D = 1536 (ViT-G14의 feature 차원)
        """
        # 1. 이미지를 14x14 패치로 분할
        # 2. ViT를 통과시켜 지정 레이어의 feature 추출
        # 3. (num_patches, D) 형태로 반환
        
        # placeholder
        H, W = image.shape[:2]
        h, w = H // 14, W // 14
        D = 1536
        features = np.random.randn(h, w, D).astype(np.float32)
        return features
    
    def extract_global_descriptor(self, image, num_clusters=64):
        """
        AnyLoc 스타일: dense feature → VLAD → global descriptor.
        
        Args:
            image: (H, W, 3) RGB 이미지
            num_clusters: VLAD 클러스터 수
            
        Returns:
            descriptor: (num_clusters * D,) global descriptor
        """
        dense_features = self.extract_dense_features(image)
        
        # Flatten spatial dimensions
        h, w, D = dense_features.shape
        features_flat = dense_features.reshape(-1, D)  # (N, D)
        
        # VLAD aggregation (simplified)
        # 실제로는 k-means로 클러스터 중심을 사전 계산
        cluster_centers = np.random.randn(num_clusters, D)  # placeholder
        
        vlad = np.zeros((num_clusters, D))
        for feat in features_flat:
            # Hard assignment
            dists = np.linalg.norm(cluster_centers - feat, axis=1)
            closest = np.argmin(dists)
            vlad[closest] += feat - cluster_centers[closest]
        
        # L2 정규화 (intra + inter)
        for i in range(num_clusters):
            norm = np.linalg.norm(vlad[i])
            if norm > 1e-8:
                vlad[i] /= norm
        
        descriptor = vlad.flatten()
        descriptor /= (np.linalg.norm(descriptor) + 1e-8)
        
        return descriptor
```

**SLAM 파이프라인에서의 활용 지점**:

| 파이프라인 모듈 | 전통 방법 | FM 대체/강화 |
|---------------|-----------|-------------|
| Feature detection | FAST, ORB | SuperPoint + DINOv2 hybrid |
| Feature matching | BF + ratio test | SuperGlue/LightGlue, LoFTR |
| Place recognition | DBoW2, Scan Context | AnyLoc (DINOv2 + VLAD) |
| Semantic segmentation | 전용 모델 학습 | [SAM](https://arxiv.org/abs/2304.02643), open-vocab segmentation |
| Depth estimation | Stereo matching | [Depth Anything](https://arxiv.org/abs/2401.10891) (monocular) |
| Loop closure verification | Geometric only | FM descriptor consistency |

### 13.1.2 Open-Vocabulary 3D Understanding

CLIP의 vision-language alignment을 3D 맵에 확장하면, 로봇이 자연어로 환경을 이해하고 탐색할 수 있다.

**작동 방식**:

1. SLAM으로 3D 맵(point cloud, mesh, 3DGS)을 구축한다.
2. 각 관측 이미지의 각 영역에 대해 CLIP visual feature를 추출한다.
3. 2D feature를 3D 맵의 대응 위치에 역투영하여 부착한다.
4. 사용자가 "소화기를 찾아"라고 하면, CLIP text encoder로 텍스트를 인코딩하고, 3D 맵에서 가장 높은 유사도를 가진 위치를 반환한다.

**ConceptFusion, LERF, OpenScene**: 이러한 접근의 대표적 시스템들. 사전 정의된 클래스 집합 없이, 임의의 텍스트 질의로 3D 공간을 탐색할 수 있다.

**현재의 한계**:
- CLIP feature의 공간적 해상도가 낮다 (패치 단위). 작은 객체의 정확한 위치 파악이 어렵다.
- 3D 일관성 보장이 어렵다 — 같은 객체가 다른 시점에서 다른 feature를 가질 수 있다.
- Computational cost: 모든 이미지에서 FM feature를 추출하는 것은 비용이 크다.

### 13.1.3 FM이 전통 파이프라인을 얼마나 대체할 수 있는가

현재 상황(2025~2026년)에서의 솔직한 평가:

**이미 대체가 진행 중인 영역**:
- **Visual place recognition**: AnyLoc이 DBoW2를 대부분의 환경에서 능가. 특히 조건 변화(주야간, 계절)가 있는 경우 격차가 크다.
- **Feature matching**: LoFTR, RoMa가 전통적 detect-describe-match를 대체하는 추세. Textureless 환경에서 특히 강점.
- **Monocular depth**: [Depth Anything](https://arxiv.org/abs/2401.10891)이 단안 카메라의 metric depth를 합리적 수준으로 추정. 보조 센서로 활용 가능.

**아직 대체가 어려운 영역**:
- **LiDAR odometry**: 전통 방법(ICP, LOAM, FAST-LIO2)이 여전히 압도적. 학습 기반 LiDAR odometry는 일반화와 정확도 모두에서 뒤처진다.
- **IMU integration**: 물리 모델 기반 preintegration이 학습으로 대체할 수 없는 정확도와 이론적 보장을 제공한다.
- **Backend optimization**: factor graph, iSAM2 같은 최적화 프레임워크는 FM으로 대체할 대상이 아니다. 오히려 FM의 출력을 factor로 통합하는 것이 올바른 방향이다.

**하이브리드 접근이 가장 유망**: 전통 파이프라인의 구조적 엄밀함을 유지하되, FM이 제공하는 강건한 feature/semantic 정보를 모듈별로 주입하는 것이 현재 가장 실용적인 방향이다.

**최근 주요 발전 (2024~2025)**:

- **[MASt3R-SLAM](https://arxiv.org/abs/2412.12392)** (Murai et al. CVPR 2025): 3D reconstruction foundation model(MASt3R)로부터 학습된 기하학적 prior를 SLAM에 직접 통합하여, 카메라 모델 가정 없이 15fps에서 globally-consistent dense SLAM을 달성했다.
- **[Depth Anything V2](https://arxiv.org/abs/2406.09414)** (Yang et al. NeurIPS 2024): 합성 데이터로 teacher를 학습하고 대규모 pseudo-label로 student를 훈련하는 전략으로, monocular depth estimation의 정확도와 강건성을 크게 향상시켰다. 센서 퓨전에서 depth prior로 활용 가능하다.

---

## 13.2 End-to-End Learned SLAM

전통 SLAM은 모듈형 파이프라인 (feature extraction → matching → motion estimation → mapping → loop closure → optimization)으로 구성된다. End-to-end 학습은 이 전체 파이프라인을 하나의 미분 가능한 시스템으로 만들어, 입력(이미지/센서)에서 출력(pose, map)까지 직접 학습하는 것을 목표로 한다.

### 13.2.1 현재의 대표 시스템

**[DROID-SLAM](https://arxiv.org/abs/2108.10869)** (Teed & Deng 2021): 현재 가장 성공적인 학습 기반 SLAM 시스템.

핵심 아키텍처:

1. **RAFT 기반 반복 업데이트 연산자**: Convolutional GRU가 correlation volume에서 추출한 특징으로 optical flow를 반복적으로 보정한다. 이 flow 보정이 correspondence를 정제하는 역할을 한다.

2. **미분 가능 Dense Bundle Adjustment (DBA)**: flow 보정값을 카메라 pose (SE(3))와 픽셀 단위 inverse depth 업데이트로 변환한다. Gauss-Newton을 Schur complement로 효율적으로 풀되, 전체가 미분 가능하여 역전파로 학습 가능하다.

$$\begin{bmatrix} \mathbf{H}_{pp} & \mathbf{H}_{pd} \\ \mathbf{H}_{dp} & \mathbf{H}_{dd} \end{bmatrix} \begin{bmatrix} \Delta \boldsymbol{\xi} \\ \Delta \mathbf{d} \end{bmatrix} = \begin{bmatrix} \mathbf{b}_p \\ \mathbf{b}_d \end{bmatrix}$$

Schur complement로 pose만 먼저 풀 수 있다:

$$(\mathbf{H}_{pp} - \mathbf{H}_{pd} \mathbf{H}_{dd}^{-1} \mathbf{H}_{dp}) \Delta \boldsymbol{\xi} = \mathbf{b}_p - \mathbf{H}_{pd} \mathbf{H}_{dd}^{-1} \mathbf{b}_d$$

$\mathbf{H}_{dd}$는 대각 행렬이므로(각 depth는 독립) 역행렬이 $O(1)$이다. 이 구조가 전통 BA와 동일한 효율성을 학습 시스템 안에서 달성하는 방식이다.

3. **프레임 그래프 기반 루프 클로저**: co-visibility 기반으로 프레임 그래프를 동적 구축. 재방문 시 장거리 에지를 추가하여 implicit loop closure 수행.

4. **단일 모델로 monocular/stereo/RGB-D 지원**: 합성 데이터(TartanAir)만으로 학습 후, 4개 벤치마크에서 SOTA.

**DROID-SLAM의 성과와 의미**:
- TartanAir에서 이전 최고 대비 오차 62% 감소
- EuRoC monocular에서 82% 감소
- 학습 기반 SLAM이 처음으로 전통 시스템을 체계적으로 능가

### 13.2.2 Differentiable SLAM Components

완전한 end-to-end가 아니더라도, SLAM 파이프라인의 개별 컴포넌트를 미분 가능하게 만드는 연구가 활발하다:

**미분 가능 렌더링**: NeRF, 3DGS 자체가 미분 가능 렌더링 시스템이다. SLAM에서 pose estimation을 photometric loss의 역전파로 수행할 수 있다.

$$\hat{\mathbf{T}}^* = \arg\min_{\hat{\mathbf{T}}} \| I_{\text{real}} - \text{Render}(\text{Map}, \hat{\mathbf{T}}) \|^2$$

이때 $\text{Render}$ 함수가 미분 가능하므로, $\hat{\mathbf{T}}$에 대한 그래디언트를 직접 계산할 수 있다.

**미분 가능 ICP**: 전통 ICP의 nearest neighbor search와 SVD를 미분 가능하게 만들어, 포인트 클라우드 정합을 학습 루프에 포함시킬 수 있다.

**미분 가능 pose graph optimization**: iSAM2 같은 최적화를 미분 가능하게 만들면, 프론트엔드(feature extraction, matching)를 백엔드 오차 신호로 학습시킬 수 있다. "최적화 결과가 나쁘면 → feature extractor를 개선하라"는 end-to-end 학습 신호.

### 13.2.3 현재의 한계와 가능성

**한계**:
- **일반화**: 학습 데이터에 없는 환경에서의 성능 저하. DROID-SLAM은 합성 데이터로 학습하여 어느 정도 일반화하지만, LiDAR가 주도하는 대규모 실외 환경에서는 아직 전통 시스템에 미치지 못한다.
- **이론적 보장 부재**: 전통 최적화는 수렴성, 일관성 등의 이론적 보장이 있다. 학습 기반 시스템은 이러한 보장이 없어 안전 중요(safety-critical) 응용에 적용하기 어렵다.
- **Computational cost**: 학습 기반 시스템은 대부분 GPU가 필요하다. 임베디드 환경에서의 실시간 동작이 도전적.
- **Interpretability**: 실패 시 원인 분석이 어렵다. 전통 시스템은 "어느 모듈에서 실패했는가"를 추적할 수 있지만, end-to-end 시스템은 블랙박스에 가깝다.

**가능성**:
- FM의 발전으로 feature extraction과 matching의 품질이 계속 향상.
- 미분 가능 최적화 기법의 성숙으로, 전통 구조를 유지하면서 학습의 이점을 취하는 하이브리드 접근이 현실적.
- Multi-task learning: pose estimation, depth estimation, semantic segmentation을 동시에 학습하여 상호 보완.

---

## 13.3 Spatial Memory & Scene Graphs

로봇이 "공간을 기억하고 이해한다"는 것은 단순히 포인트 클라우드를 저장하는 것 이상이다. 인간은 "주방에 냉장고가 있고, 그 안에 우유가 있었다"는 식의 계층적, 관계적, 시간적 공간 기억을 가진다. 이 섹션은 이러한 고수준 공간 기억 시스템의 연구 프런티어를 다룬다.

### 13.3.1 Persistent Spatial Memory

전통 SLAM 맵은 "현재 이 순간의 환경 상태"를 반영한다. Persistent spatial memory는 시간에 따른 환경의 변화 이력까지 포함하는 장기 공간 기억이다.

**핵심 과제**:

1. **Episodic spatial memory**: "지난주 화요일에 여기에 상자가 있었다"와 같은 시간-장소-사건 연결.
2. **Semantic persistence**: 영구적 요소(벽, 건물)와 일시적 요소(사람, 차량)를 구분하여 장기 맵의 안정성 유지.
3. **Incremental forgetting**: 오래된 관측의 세부사항은 서서히 잊되, 핵심 구조는 유지. 인간의 기억 방식과 유사.

```python
class PersistentSpatialMemory:
    """
    시간적 공간 기억 — 환경 상태의 시계열 기록.
    """
    
    def __init__(self, decay_rate=0.01):
        self.memories = {}  # {location_key: [MemoryEntry, ...]}
        self.decay_rate = decay_rate
    
    def record(self, location, observation, timestamp, semantic_class=None):
        """새 관측을 공간 기억에 기록."""
        key = self._spatial_key(location)
        
        entry = {
            'timestamp': timestamp,
            'observation': observation,
            'semantic_class': semantic_class,
            'confidence': 1.0,
            'access_count': 0
        }
        
        if key not in self.memories:
            self.memories[key] = []
        self.memories[key].append(entry)
    
    def recall(self, location, time_query=None, semantic_query=None):
        """
        공간 기억에서 관련 정보를 회상.
        
        Args:
            location: 질의 위치
            time_query: 특정 시점 질의 (예: "3일 전")
            semantic_query: 의미론적 질의 (예: "의자")
        """
        key = self._spatial_key(location)
        if key not in self.memories:
            return []
        
        results = []
        for entry in self.memories[key]:
            relevance = entry['confidence']
            
            if time_query is not None:
                time_diff = abs(entry['timestamp'] - time_query)
                relevance *= np.exp(-self.decay_rate * time_diff)
            
            if semantic_query is not None:
                if entry['semantic_class'] != semantic_query:
                    continue
            
            if relevance > 0.1:
                results.append((entry, relevance))
                entry['access_count'] += 1
        
        return sorted(results, key=lambda x: x[1], reverse=True)
    
    def detect_changes(self, location, current_observation):
        """현재 관측과 기억을 비교하여 변화 감지."""
        key = self._spatial_key(location)
        if key not in self.memories:
            return 'new_location'
        
        latest = self.memories[key][-1]
        
        # 관측 비교 (placeholder — 실제로는 feature 비교)
        similarity = self._compare_observations(
            latest['observation'], current_observation
        )
        
        if similarity < 0.5:
            return 'significant_change'
        elif similarity < 0.8:
            return 'minor_change'
        else:
            return 'no_change'
    
    def consolidate(self, max_entries_per_location=10):
        """
        기억 정리: 오래되고 접근 빈도가 낮은 기억을 제거.
        핵심 구조적 정보는 보존.
        """
        for key in self.memories:
            entries = self.memories[key]
            
            if len(entries) <= max_entries_per_location:
                continue
            
            # 우선순위: 최근성 + 접근 빈도 + confidence
            scored = []
            for entry in entries:
                score = (entry['confidence'] * 
                         (1 + entry['access_count']) * 
                         (1.0 / (1 + self.decay_rate * 
                                 (entries[-1]['timestamp'] - entry['timestamp']))))
                scored.append((entry, score))
            
            scored.sort(key=lambda x: x[1], reverse=True)
            self.memories[key] = [
                e for e, _ in scored[:max_entries_per_location]
            ]
    
    def _spatial_key(self, location, resolution=0.5):
        return tuple((np.array(location) / resolution).astype(int))
    
    def _compare_observations(self, obs1, obs2):
        return 0.5  # placeholder
```

### 13.3.2 Scene Graph 기반 환경 이해

Ch.11에서 [Hydra](https://arxiv.org/abs/2201.13360)의 3D Scene Graph를 다루었다. 여기서는 scene graph가 열어주는 미래 방향을 탐색한다.

**Scene Graph + Language**: Scene graph에 자연어 인터페이스를 결합하면, 로봇에게 "거실 소파 옆의 테이블 위에 있는 리모콘을 가져와"라는 자연어 명령을 이해시킬 수 있다. 이 명령은 scene graph의 계층적 탐색으로 변환된다:

1. "거실" → Room 노드 탐색
2. "소파 옆의 테이블" → Room 내 Object 노드의 관계 탐색
3. "리모콘" → 해당 Table 근처의 Object 탐색
4. 경로 계획 및 manipulation

**Scene Graph + LLM**: GPT-4 같은 LLM이 scene graph를 입력으로 받아 고수준 추론을 수행한다. "이 방에 사람이 넘어지면 가장 가까운 전화기는 어디에 있는가?" 같은 질의에 답할 수 있다.

**동적 Scene Graph**: Hydra의 현재 구현은 정적 환경을 가정한다. 동적 scene graph는 사람, 차량 등 움직이는 에이전트를 노드로 포함하고, 그들의 관계를 실시간으로 갱신한다. 이는 사회적 내비게이션(social navigation), 인간-로봇 상호작용(HRI)의 핵심이다.

### 13.3.3 시계열 공간 기억 관리

장기 운용 로봇은 공간 기억의 **생성, 유지, 삭제** 전략이 필요하다.

**Hierarchical forgetting**: 세부 정보(정확한 텍스처, 개별 포인트)는 시간에 따라 해상도를 낮추고, 구조적 정보(방의 배치, 통로 연결)는 영구 유지한다. 이는 인간의 공간 기억과 유사한 전략이다.

**Event-triggered update**: 전체 맵을 주기적으로 갱신하는 대신, 변화가 감지된 영역만 선택적으로 업데이트한다.

**Compression**: 시간이 지남에 따라 맵을 점진적으로 압축한다. 예를 들어, dense point cloud → sparse landmarks → topological graph → semantic description.

---

## 13.4 Cross-Modal Representation

센서 퓨전의 핵심 도전 중 하나는 이종 센서의 관측을 **공통 표현 공간**에서 비교하는 것이다. LiDAR 포인트 클라우드와 카메라 이미지는 완전히 다른 데이터 형태이지만, 같은 물리적 환경을 관측한다. Cross-modal representation은 이 "표현 격차(representation gap)"를 해소하는 연구 방향이다.

### 13.4.1 이종 센서 간 표현 정렬 문제

**왜 어려운가**:

- **차원 불일치**: LiDAR는 3D 포인트, 카메라는 2D 이미지, radar는 range-Doppler map. 데이터 형태가 본질적으로 다르다.
- **정보 비대칭**: LiDAR는 정확한 거리 정보를 제공하지만 텍스처가 없다. 카메라는 풍부한 텍스처를 제공하지만 절대 거리 정보가 없다.
- **센서 특유의 아티팩트**: LiDAR의 motion distortion, 카메라의 rolling shutter, radar의 speckle noise 등 각 센서 고유의 노이즈 패턴이 다르다.

### 13.4.2 Contrastive Learning for Cross-Modal Alignment

Contrastive learning은 같은 장소/객체의 다른 모달리티 관측을 가까이, 다른 장소/객체의 관측을 멀리 배치하는 표현을 학습한다.

$$\mathcal{L}_{\text{contrastive}} = -\log \frac{\exp(\text{sim}(f_L(\mathbf{x}_L), f_C(\mathbf{x}_C)) / \tau)}{\sum_{j} \exp(\text{sim}(f_L(\mathbf{x}_L), f_C(\mathbf{x}_C^j)) / \tau)}$$

여기서 $f_L$은 LiDAR encoder, $f_C$는 카메라 encoder, $\tau$는 temperature, $(\mathbf{x}_L, \mathbf{x}_C)$는 같은 장소의 LiDAR-카메라 쌍, $\mathbf{x}_C^j$는 negative sample이다.

**Cross-modal place recognition**에서의 응용: LiDAR로 만든 맵에서 카메라만으로 localization하는 시나리오. LiDAR descriptor와 camera descriptor가 같은 공간에 있으면, 카메라 query로 LiDAR 맵을 검색할 수 있다.

**LC$^2$** (Lee et al. 2023): LiDAR-Camera cross-modal place recognition. LiDAR BEV 이미지와 카메라 이미지의 feature를 공통 공간으로 정렬한다.

### 13.4.3 Knowledge Distillation

한 모달리티(teacher)의 풍부한 정보를 다른 모달리티(student)로 전달하는 방법.

**LiDAR → Camera distillation**: LiDAR의 정확한 3D 정보로 학습된 모델의 지식을, 카메라만 사용하는 모델로 전달한다. 이를 통해 배포 시에는 카메라만으로 LiDAR 수준의 3D 이해를 근사할 수 있다.

**Camera → LiDAR distillation**: 카메라의 풍부한 의미론적 정보를 LiDAR 처리 모델에 전달한다. 예를 들어, CLIP의 의미론적 feature를 LiDAR 포인트에 부여하여, 텍스트 질의로 LiDAR 맵을 검색할 수 있게 한다.

### 13.4.4 아직 열린 질문들

1. **모달리티 독립적(modality-agnostic) 표현이 가능한가?** LiDAR, 카메라, radar, event camera 등 어떤 센서 입력이든 같은 표현 공간으로 매핑할 수 있는 범용 encoder가 가능할까?

2. **Temporal alignment**: 다른 모달리티의 관측은 시간적으로 완벽히 동기화되지 않는다. 비동기 관측을 어떻게 공통 표현으로 융합할 것인가?

3. **Partial observation**: 하나의 센서가 일시적으로 실패(LiDAR가 비에 영향, 카메라가 어둠에 영향)할 때, 사용 가능한 모달리티만으로 일관된 표현을 유지하는 방법.

---

## 13.5 Event Camera 기반 퓨전

이벤트 카메라(Event Camera, Dynamic Vision Sensor, DVS)는 전통적 프레임 기반 카메라와 근본적으로 다른 센서다. 각 픽셀이 독립적으로 밝기 변화를 감지하여, 변화가 일어난 시점에만 **이벤트**를 비동기적으로 출력한다.

### 13.5.1 Event Camera의 원리와 장점

각 이벤트는 $(x, y, t, p)$로 표현된다:

- $(x, y)$: 픽셀 좌표
- $t$: 마이크로초 단위 타임스탬프
- $p \in \{+1, -1\}$: 극성 (밝아짐 / 어두워짐)

이벤트가 발생하는 조건:

$$|\log I(x, y, t) - \log I(x, y, t_{\text{last}})| \geq C$$

이때 극성(polarity)은 $p = \text{sign}(\log I(x, y, t) - \log I(x, y, t_{\text{last}}))$로 결정된다. 여기서 $I$는 밝기, $C$는 대비 임계값(contrast threshold), $t_{\text{last}}$는 해당 픽셀에서 마지막으로 이벤트가 발생한 시점이다.

**장점**:

| 특성 | 프레임 카메라 | Event 카메라 |
|------|------------|-------------|
| 시간 해상도 | 30~120 fps | 마이크로초 단위 |
| 동적 범위 | ~60 dB | >120 dB |
| 모션 블러 | 있음 | 거의 없음 |
| 데이터 출력 | 균일 프레임 | 비동기 이벤트 |
| 정적 장면 | 정보 제공 | 이벤트 없음 (정보 없음) |
| 전력 소비 | 높음 | 매우 낮음 |

**왜 센서 퓨전에서 중요한가**: Event camera는 전통 카메라가 실패하는 극한 조건 — 고속 회전, 급격한 조명 변화(터널 진입/출구), 저조도 환경 — 에서 강건하다. 다른 센서의 취약점을 보완하는 역할을 한다.

### 13.5.2 Event + Frame 퓨전

Event camera와 전통 프레임 카메라를 결합하는 접근:

**Event-enhanced frame tracking**: 프레임 간의 고속 모션을 이벤트로 추적하여, 프레임 기반 VO의 프레임 간격 사이를 채운다. 빠른 카메라 모션에서도 tracking이 끊기지 않는다.

**Event-aided HDR**: 이벤트의 높은 동적 범위를 활용하여, 프레임 이미지의 under/over-exposed 영역의 정보를 보완한다.

### 13.5.3 Event + IMU 퓨전

**[EVO](https://doi.org/10.1109/LRA.2016.2645143)** (Rebecq et al. 2017): Event-based Visual Odometry. 이벤트만으로 camera pose를 추정한다. IMU와 결합하면 스케일을 복원하고 정확도를 높일 수 있다.

**[Ultimate SLAM](https://arxiv.org/abs/1709.06310)** (Vidal et al. 2018): Event camera + 프레임 카메라 + IMU를 결합한 시스템. 세 센서의 상보성을 활용:
- 프레임 카메라: 정적 장면에서의 풍부한 텍스처
- Event camera: 고속 모션에서의 연속적 추적
- IMU: 스케일 복원과 빠른 모션 예측

**현재 과제**:
- Event camera의 데이터 형식(비동기 이벤트 스트림)이 전통적 컴퓨터 비전 파이프라인(프레임 기반)과 호환되지 않는다. 이벤트를 프레임으로 변환(event frame)하면 장점을 잃는다.
- 상용 event camera가 아직 고가이며, 해상도가 낮다 (최신 모델도 1280 × 720 수준).
- 학습 데이터가 부족하다. 대부분의 데이터셋은 프레임 카메라용이다.

**최근 주요 발전 (2024~2025)**:

- **[EvenNICER-SLAM](https://arxiv.org/abs/2410.03812)** (2024): Event camera를 neural implicit SLAM에 통합한 시스템으로, 이벤트의 높은 시간 해상도를 활용하여 고속 모션에서의 tracking 강건성을 향상시켰다.
- **Event-based 3D reconstruction survey** ([arxiv:2505.08438](https://arxiv.org/abs/2505.08438), 2025): Event-driven 3D reconstruction 분야 서베이로, NeRF·3DGS 기반 이벤트 재구성, depth estimation, optical flow 등 최신 연구를 분류·정리했다.

```python
class EventProcessor:
    """
    Event camera 데이터 처리 유틸리티.
    """
    
    def __init__(self, width, height, time_window_us=33000):
        """
        Args:
            width, height: 센서 해상도
            time_window_us: 이벤트 누적 윈도우 (마이크로초)
        """
        self.width = width
        self.height = height
        self.time_window = time_window_us
    
    def events_to_frame(self, events, method='histogram'):
        """
        이벤트 스트림을 프레임으로 변환.
        
        Args:
            events: [(x, y, t, p), ...] 이벤트 리스트
            method: 'histogram' 또는 'time_surface'
            
        Returns:
            frame: (H, W) 또는 (H, W, 2) 이벤트 프레임
        """
        if method == 'histogram':
            return self._event_histogram(events)
        elif method == 'time_surface':
            return self._time_surface(events)
    
    def _event_histogram(self, events):
        """
        이벤트 히스토그램: 양/음 이벤트를 별도 채널로 누적.
        가장 단순한 변환이지만 시간 정보를 잃는다.
        """
        frame = np.zeros((self.height, self.width, 2), dtype=np.float32)
        
        for x, y, t, p in events:
            if 0 <= x < self.width and 0 <= y < self.height:
                channel = 0 if p > 0 else 1
                frame[int(y), int(x), channel] += 1
        
        return frame
    
    def _time_surface(self, events):
        """
        Time Surface: 각 픽셀의 가장 최근 이벤트 시간을 기록.
        시간 정보를 보존하면서 프레임 형태로 변환.
        """
        time_surface = np.zeros((self.height, self.width, 2), 
                                 dtype=np.float64)
        
        if len(events) == 0:
            return time_surface
        
        t_ref = events[-1][2]  # 참조 시간 (가장 최근)
        
        for x, y, t, p in events:
            if 0 <= x < self.width and 0 <= y < self.height:
                channel = 0 if p > 0 else 1
                time_surface[int(y), int(x), channel] = np.exp(
                    -(t_ref - t) / self.time_window
                )
        
        return time_surface
    
    def events_to_optical_flow(self, events, dt_us=10000):
        """
        이벤트에서 optical flow 추정 (contrast maximization 방식 간소화).
        
        핵심 아이디어: 올바른 optical flow로 이벤트를 시간적으로 정렬(warp)하면,
        이미지의 edge가 가장 선명해진다 (contrast가 최대화).
        """
        if len(events) < 100:
            return np.zeros((self.height, self.width, 2))
        
        # Contrast maximization:
        # argmax_{v} Var(I_warp(v))
        # 여기서 I_warp는 flow v로 warped된 이벤트 이미지
        
        # 간소화된 구현 (실제로는 최적화가 필요)
        best_flow = np.zeros(2)
        best_contrast = 0
        
        for vx in np.linspace(-2, 2, 20):
            for vy in np.linspace(-2, 2, 20):
                warped = np.zeros((self.height, self.width))
                
                t_ref = events[-1][2]
                for x, y, t, p in events:
                    dt = (t_ref - t) / 1e6  # 초 단위
                    wx = int(x + vx * dt)
                    wy = int(y + vy * dt)
                    
                    if 0 <= wx < self.width and 0 <= wy < self.height:
                        warped[wy, wx] += p
                
                contrast = np.var(warped)
                if contrast > best_contrast:
                    best_contrast = contrast
                    best_flow = np.array([vx, vy])
        
        flow = np.zeros((self.height, self.width, 2))
        flow[:, :] = best_flow
        return flow
```

---

## 13.6 4D Radar 퓨전

4D imaging radar는 센서 퓨전 분야에서 최근 주목받기 시작한 모달리티다. 전통적 automotive radar가 거리와 각도만 제공했다면, 4D radar는 거리(range), 방위각(azimuth), 고도(elevation), 도플러 속도(Doppler velocity)의 4차원 정보를 제공한다.

**FMCW radar의 거리/속도 측정 원리**: 4D radar의 대부분은 FMCW(Frequency-Modulated Continuous Wave) 방식을 사용한다. 송신 신호의 주파수를 시간에 따라 선형으로 증가(chirp)시키고, 반사 신호와의 비트 주파수(beat frequency)로 거리를, chirp 간 위상 변화로 속도를 측정한다:

$$R = \frac{c \cdot f_b}{2 \cdot S}$$

여기서 $R$은 타겟까지의 거리, $c$는 광속, $f_b$는 비트 주파수, $S$는 chirp의 주파수 변화율(Hz/s)이다.

$$v = \frac{\lambda \cdot \Delta\phi}{4\pi \cdot T_c}$$

여기서 $v$는 타겟의 radial velocity, $\lambda$는 캐리어 파장, $\Delta\phi$는 연속된 두 chirp 사이의 위상 변화, $T_c$는 chirp 주기이다.

### 13.6.1 악천후 Robustness

4D radar는 악천후에서도 작동한다. 이것이 이 센서의 핵심 가치다:

| 조건 | 카메라 | LiDAR | 4D Radar |
|------|--------|-------|----------|
| 맑은 날 | 최고 | 최고 | 양호 |
| 비 | 저하 | 약간 저하 | 정상 |
| 안개 | 심각 저하 | 심각 저하 | 정상 |
| 눈/먼지 | 심각 저하 | 심각 저하 | 정상 |
| 야간 | 심각 저하 | 정상 | 정상 |
| 직사광선 | 저하 | 정상 | 정상 |

Radar의 파장(밀리미터파)은 물방울, 안개 입자, 먼지보다 훨씬 크다. 이 입자들이 신호를 거의 산란시키지 못한다. LiDAR(근적외선)와 카메라(가시광선)가 갖는 한계를 여기서 보완한다.

### 13.6.2 4D Radar + Camera Fusion

4D radar와 카메라를 합치면 "악천후에서도 동작하는 저비용 인식 시스템"에 가까워진다:

**BEV 기반 퓨전**: 카메라 이미지에서 BEV feature를 추출하고 (LSS 또는 BEVFormer 방식), radar 포인트를 BEV 공간에 투영하여 결합한다.

**Radar의 Doppler 정보 활용**: 4D radar는 각 포인트의 시선 방향 속도(radial velocity)를 직접 측정한다. 이는 카메라나 LiDAR에는 없는 고유한 정보로:

- **동적 객체 분류**: 정적 배경과 움직이는 객체를 Doppler로 즉시 구분.
- **Ego-motion estimation**: 정적 포인트의 Doppler로 자차 속도를 추정 (IMU 없이도 가능).
- **Tracking 지원**: 객체의 속도 정보를 tracking에 직접 사용.

$$v_r = (\mathbf{v}_{\text{obj}} - \mathbf{v}_{\text{ego}}) \cdot \hat{\mathbf{r}}$$

여기서 $v_r$은 측정된 radial velocity, $\mathbf{v}_{\text{obj}}$는 객체 속도, $\mathbf{v}_{\text{ego}}$는 ego 속도, $\hat{\mathbf{r}}$은 radar → 타겟 방향 단위 벡터($\|\hat{\mathbf{r}}\| = 1$)다. 정적 객체($\mathbf{v}_{\text{obj}} = \mathbf{0}$)의 경우 $v_r = -\mathbf{v}_{\text{ego}} \cdot \hat{\mathbf{r}}$이 된다.

### 13.6.3 Radar Odometry의 최근 발전

Radar odometry는 2020년 이후 활발하게 연구되고 있다:

**FMCW radar odometry**: scanning FMCW radar (Navtech 등)에서의 odometry. Range-azimuth 이미지에서 특징점을 추출하고 매칭하여 ego-motion을 추정한다.

**4D radar odometry**: 4D radar 포인트 클라우드에서의 odometry. LiDAR odometry와 유사한 접근(ICP, feature matching)이 가능하지만, 해상도가 낮고 노이즈가 크다는 도전이 있다.

**Doppler 기반 ego-velocity estimation**: 정적 포인트의 Doppler 측정으로 ego-velocity를 직접 추정한다. RANSAC으로 동적 포인트를 제거하고, 정적 포인트의 Doppler로 $\mathbf{v}_{\text{ego}}$를 추정하는 방식:

$$v_r^{(k)} = -\mathbf{v}_{\text{ego}} \cdot \hat{\mathbf{r}}^{(k)} \quad \text{(정적 포인트)}$$

여기서 $k$는 포인트 인덱스. 최소 3개의 비공선(non-collinear) 포인트로 3D 속도 벡터를 추정할 수 있다.

```python
def estimate_ego_velocity_from_doppler(radar_points, doppler_values, 
                                        directions, ransac_threshold=0.3,
                                        max_iterations=100):
    """
    Radar Doppler 측정으로 ego-velocity 추정.
    
    정적 포인트의 Doppler: v_r = -v_ego · r_hat
    이는 선형 시스템으로 풀 수 있다.
    
    Args:
        radar_points: (N, 3) radar 포인트 좌표
        doppler_values: (N,) 각 포인트의 radial velocity
        directions: (N, 3) 각 포인트의 방향 단위 벡터
        ransac_threshold: RANSAC 인라이어 임계값 (m/s)
        max_iterations: RANSAC 최대 반복 수
        
    Returns:
        v_ego: (3,) ego-velocity 벡터
        inlier_mask: (N,) 정적 포인트 마스크
    """
    N = len(doppler_values)
    best_inliers = np.zeros(N, dtype=bool)
    best_v_ego = np.zeros(3)
    
    for _ in range(max_iterations):
        # 3개 포인트 랜덤 샘플링
        idx = np.random.choice(N, 3, replace=False)
        
        # 선형 시스템: v_r = -directions @ v_ego
        # A @ v_ego = b
        A = -directions[idx]  # (3, 3)
        b = doppler_values[idx]  # (3,)
        
        try:
            v_ego_candidate = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            continue
        
        # 모든 포인트에 대한 잔차 계산
        predicted_doppler = -directions @ v_ego_candidate
        residuals = np.abs(doppler_values - predicted_doppler)
        
        inliers = residuals < ransac_threshold
        
        if np.sum(inliers) > np.sum(best_inliers):
            best_inliers = inliers
            best_v_ego = v_ego_candidate
    
    # 인라이어로 재추정 (least squares)
    if np.sum(best_inliers) >= 3:
        A = -directions[best_inliers]
        b = doppler_values[best_inliers]
        best_v_ego = np.linalg.lstsq(A, b, rcond=None)[0]
    
    return best_v_ego, best_inliers


def separate_static_dynamic(radar_points, doppler_values, directions,
                             v_ego, threshold=0.5):
    """
    Ego-velocity를 이용하여 정적/동적 포인트 분류.
    
    Args:
        v_ego: 추정된 ego-velocity (3,)
        threshold: 정적/동적 분류 임계값 (m/s)
        
    Returns:
        static_mask: (N,) 정적 포인트
        dynamic_mask: (N,) 동적 포인트
        object_velocities: (N,) 각 포인트의 추정 객체 속도 (radial)
    """
    # 정적 가정 하의 예상 Doppler
    expected_doppler = -directions @ v_ego
    
    # 잔차 = 실제 Doppler - 예상 Doppler
    residuals = doppler_values - expected_doppler
    
    static_mask = np.abs(residuals) < threshold
    dynamic_mask = ~static_mask
    
    # 동적 포인트의 객체 속도 (radial component)
    object_velocities = residuals  # v_obj · r_hat
    
    return static_mask, dynamic_mask, object_velocities
```

### 13.6.4 대표 데이터셋과 벤치마크

| 데이터셋 | 센서 | 환경 | 특징 |
|----------|------|------|------|
| **[Boreas](https://arxiv.org/abs/2203.10168)** (Burnett et al. 2023) | Camera, LiDAR, Radar, GNSS/IMU | 도심 (다양한 날씨) | 1년간 동일 경로 반복, 악천후 포함 |
| **RadarScenes** | Radar, Camera, LiDAR | 도심 | 기존 automotive radar 포인트 + semantic labels (point-level annotation) |
| **nuScenes** | Camera, LiDAR, Radar | 도심 | 5개 radar 포함, 악천후 일부 |
| **View-of-Delft** | Camera, LiDAR, 4D Radar | 도심 | 4D radar + 3D annotation |

**최근 주요 발전 (2024~2025)**:

- **[Snail-Radar](https://arxiv.org/abs/2407.11705)** (2024): 4D radar 기반 SLAM 평가 벤치마크로, 핸드헬드·자전거·SUV 세 플랫폼에서 다양한 날씨/조명 조건으로 수집된 44개 시퀀스를 제공한다.
- **[4D Radar-Inertial Odometry](https://arxiv.org/abs/2412.13639)** (2024): 3D Gaussian 기반 radar scene representation과 multi-hypothesis scan matching을 제안하여 voxel 방식 대비 더 정밀한 radar odometry를 달성했다.

4D radar 퓨전은 아직 초기 단계다. 특히 Doppler 정보를 활용한 ego-motion estimation과 동적 객체 분류는 LiDAR나 카메라로는 대체할 수 없는 고유한 기능이다.

---

## 마무리

이 가이드는 센서 모델링(Ch.2)에서 출발하여 캘리브레이션(Ch.3), 상태 추정 이론(Ch.4), 특징점 매칭(Ch.5), VO/VIO(Ch.6), LiDAR odometry(Ch.7), 멀티센서 퓨전(Ch.8), Place Recognition(Ch.9), Loop Closure(Ch.10), 공간 표현(Ch.11), 실전 시스템(Ch.12), 그리고 연구 프런티어(Ch.13)까지 — 센서 퓨전의 전체 파이프라인을 단계별로 다루었다.

이 분야의 핵심 서사를 다시 한번 정리하면:

1. **전통 방법은 여전히 기반이다.** 칼만 필터, ICP, RANSAC, 팩터 그래프 — 수십 년 전에 제안된 이 방법들이 현대 시스템의 뼈대를 이루고 있다.
2. **딥러닝은 지각(perception)에서 전통 방법을 밀어냈다.** 특징점 매칭, 깊이 추정, Place Recognition 등 "무엇을 보는가"의 영역에서 학습 기반 방법이 전통을 압도한다.
3. **추론(inference)에서는 전통과 학습이 공존한다.** 상태 추정 backend는 여전히 최적화 기반이 지배적이지만, DROID-SLAM처럼 미분 가능 최적화로 경계를 허무는 시도가 진행 중이다.
4. **Foundation model이 파이프라인을 바꾸고 있다.** DINOv2, SAM 등 범용 모델의 표현력이 센서 퓨전 파이프라인 곳곳에 유입되고 있다.

센서 퓨전은 "센서 데이터를 합치는 기술"이 아니라, **불완전한 관측으로부터 세상을 이해하는 기술**이다. 이 가이드가 그 이해의 출발점이 되기를 바란다.
