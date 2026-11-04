# Ch.13 — Frontiers & Emerging Directions

---

## 13.1 Foundation Models for Spatial AI

Foundation model(FM)은 대규모 데이터로 미리 학습한 범용 모델이다(DINOv2, CLIP, SAM 등). 이 모델이 뽑은 표현을 센서 퓨전과 SLAM pipeline에 넣는 연구가 이어지고 있다. 이 절에서 다루는 연구는 대부분 이 표현을 retrieval, matching, semantic prior에 쓴다. MASt3R-SLAM(§13.1.3)처럼 dense geometry 예측까지 맡기는 시스템도 나왔다.

### 13.1.1 DINOv2/CLIP Feature를 SLAM에 활용

**DINOv2의 시각 특징**: [DINOv2](https://arxiv.org/abs/2304.07193)는 자기지도학습(self-supervised learning)으로 학습한 ViT다. Image patch마다 나오는 token을 dense feature로 쓴다. downstream 실험에서 드러난 성질은 아래와 같다.

- 일부 조명·계절 변화에서 같은 장소의 대응을 찾는 데 유용한 feature가 나타난다.
- 객체 category와 장면 semantics가 feature similarity에 반영될 수 있다.
- Patch 단위 feature를 correspondence나 segmentation 같은 downstream task에 쓸 수 있다.

**AnyLoc의 접근**: [AnyLoc](https://arxiv.org/abs/2308.00688) (Keetha et al., RA-L 2024)은 DINOv2의 dense feature를 VLAD로 모아 장소 전체를 나타내는 디스크립터 하나를 만든다. VLAD(Vector of Locally Aggregated Descriptors)는 각 feature와 가장 가까운 클러스터 중심의 차이를 클러스터별로 더하는 집계 방식이다.

- 논문은 이 디스크립터를 도심, 실내, 항공, 수중, 지하를 포함한 benchmark에서 VPR 전용 파인튜닝 없이 평가했다.
- 원 논문은 비교한 NetVLAD·CosPlace 계열보다 recall이 높다고 보고했고, 초록에는 기존 방법보다 최대 4배라고 적었다. 이 값은 논문이 쓴 dataset에서 나온 것이다.
- 논문 내 ablation에서는 31번째 layer의 value-facet dense feature가 전역 CLS token 디스크립터보다 좋았다. 개선 폭은 지표와 평가 대상에 따라 다르므로 원 논문의 ablation 표에서 지표 이름과 평균 범위를 함께 읽는다.

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
        이미지에서 패치 단위(1/14 해상도) dense feature를 추출.
        
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

**SLAM 파이프라인 모듈별로 넣는 자리**:

| 파이프라인 모듈 | 전통 방법 | 학습 feature를 넣는 예 |
|---------------|-----------|-------------|
| Feature detection | FAST, ORB | SuperPoint + DINOv2 hybrid |
| Feature matching | BF + ratio test | SuperGlue/LightGlue, LoFTR |
| Place recognition | DBoW2, Scan Context | AnyLoc (DINOv2 + VLAD) |
| Semantic segmentation | 전용 모델 학습 | [SAM](https://arxiv.org/abs/2304.02643), open-vocab segmentation |
| Depth estimation | Stereo matching | [Depth Anything](https://arxiv.org/abs/2401.10891)의 relative-depth prior |
| Loop closure candidate | DBoW2, Scan Context | FM descriptor similarity 후 geometric verification |

### 13.1.2 Open-Vocabulary 3D Understanding

CLIP은 이미지와 텍스트를 같은 공간에 놓도록 학습한 모델이다. 이 정렬을 3D 맵으로 넓히면 로봇에게 자연어로 물체를 찾게 할 수 있다. 동작 순서는 이렇다.

1. SLAM으로 3D 맵(point cloud, mesh, 3DGS)을 만든다.
2. 관측 이미지마다 영역별로 CLIP visual feature를 뽑는다.
3. 2D feature를 3D 맵의 대응 위치로 역투영해 붙인다.
4. 사용자가 "소화기를 찾아"라고 하면 CLIP text encoder로 그 문장을 벡터로 바꾸고, 3D 맵에서 유사도가 가장 높은 위치를 돌려준다.

[ConceptFusion](https://arxiv.org/abs/2302.07241) (Jatavallabhula et al. 2023)이 이 순서를 그대로 따른다. 픽셀 단위로 정렬한 foundation model feature를 SLAM과 다중 시점 융합으로 3D 맵에 합치고, 추가 학습 없이 텍스트·이미지·오디오로 맵을 검색한다. [LERF](https://arxiv.org/abs/2303.09553) (Kerr et al. 2023)는 feature를 점에 붙이는 대신 CLIP embedding을 NeRF 안에 language field로 학습한다. [OpenScene](https://arxiv.org/abs/2211.15654) (Peng et al. 2023)은 3D 점마다 CLIP 공간의 feature를 예측하는 3D 네트워크를 라벨 없이 학습한다. 세 시스템 모두 미리 정한 클래스 목록 없이 임의의 텍스트로 3D 장면을 검색한다.

**현재의 한계**:
- CLIP feature는 패치 단위라 공간 해상도가 낮아서 작은 물체의 위치를 정확히 잡기 어렵다.
- 같은 물체도 보는 시점에 따라 feature가 달라져 3D에서 일관성을 지키기 어렵다.
- 모든 이미지에서 FM feature를 뽑으면 계산량이 크다.

### 13.1.3 모듈별 근거와 경계

Foundation model을 넣었을 때 효과가 얼마나 입증됐는지는 과제와 벤치마크마다 다르다.

**논문 benchmark에서 활용 근거가 있는 영역**:
- **Visual place recognition**: AnyLoc은 DINOv2+VLAD를 여러 domain에서 평가해 높은 recall을 보고했다. DBoW2와는 입력 feature, 학습 조건, 계산량이 다르므로, DBoW2 자리에 쓰려면 목표 환경에서 false-positive rate와 latency를 함께 비교한다.
- **Feature matching**: LoFTR·RoMa는 texture와 viewpoint가 어려운 benchmark에서 좋은 결과를 보고한다. Classical sparse feature도 compute budget, calibration, repeatability 조건에 따라 알맞은 선택이 된다.
- **Monocular depth**: [Depth Anything](https://arxiv.org/abs/2401.10891)의 기본 출력은 relative depth다. Metric scale이 필요한 fusion에서는 metric depth용 파인튜닝이나 다른 sensor constraint가 필요하며, 출력 uncertainty도 따로 다뤄야 한다.

**별도의 estimator가 계속 필요한 영역**:
- **LiDAR odometry**: ICP·LOAM·FAST-LIO2 같은 geometry-based 방법과 learned method의 비교 결과는 sensor와 dataset에 따라 달라진다. Learned feature나 correspondence를 넣더라도 motion model과 registration 검증은 남는다.
- **IMU integration**: Preintegration은 kinematics와 noise model을 factor에 압축한다. 학습한 bias·noise 보정을 붙일 수는 있어도, preintegration의 관측 모델과 uncertainty 전파는 그대로 필요하다.
- **Backend optimization**: Factor graph와 iSAM2는 관측을 결합하는 estimator다. FM output을 factor로 넣으려면 covariance, outlier, correlation을 명시해야 한다.

**최근 시스템 (2024~2025)**:

- **[MASt3R-SLAM](https://arxiv.org/abs/2412.12392)** (Murai et al. CVPR 2025): MASt3R를 바탕으로 만든 단안 dense SLAM으로, 15 FPS로 전역 일관된 포즈와 dense geometry를 낸다고 보고했다(§11.3.2).
- **[Depth Anything V2](https://arxiv.org/abs/2406.09414)** (Yang et al. NeurIPS 2024): 라벨 있는 실사 이미지를 모두 합성 이미지로 바꿔 큰 teacher를 학습하고, teacher가 대규모 실사 이미지에 붙인 pseudo-label로 student를 학습한다. Metric depth 라벨로 파인튜닝한 모델도 따로 냈다. Fusion에 depth prior로 쓰려면 scale과 uncertainty를 따로 검증해야 한다.

---

## 13.2 End-to-End Learned SLAM

전통 SLAM은 feature extraction → matching → motion estimation → mapping → loop closure → optimization으로 모듈을 이어 붙인다. End-to-end 학습은 이 전체를 미분 가능한 시스템 하나로 만들고, 이미지나 센서 입력에서 포즈와 map까지 한 번에 학습한다.

### 13.2.1 현재의 대표 시스템

**[DROID-SLAM](https://arxiv.org/abs/2108.10869)** (Teed & Deng 2021)은 학습 기반 visual SLAM의 대표적인 공개 시스템이다. RAFT 기반 반복 업데이트 연산자가 correspondence를 다듬고, 미분 가능 Dense Bundle Adjustment(DBA) 레이어가 이를 카메라 포즈와 inverse depth 업데이트로 바꾼다. 계산 전체가 미분 가능하므로 역전파로 학습하고, 합성 데이터(TartanAir)로 학습한 모델 하나를 monocular/stereo/RGB-D 입력에 적용한다. Co-visibility 프레임 그래프에 멀리 떨어진 프레임 사이의 엣지를 더해, 따로 loop closure 모듈 없이 루프를 닫는다(§6.5.3).

### 13.2.2 Differentiable SLAM Components

SLAM 전체를 end-to-end로 바꾸지 않고 일부 컴포넌트만 미분 가능하게 만드는 방법도 있다.

**미분 가능 렌더링**: NeRF와 3DGS는 렌더링 자체가 미분 가능하다. 렌더링한 이미지와 실제 이미지의 photometric loss를 역전파해 포즈를 구한다.

$$\hat{\mathbf{T}}^* = \arg\min_{\hat{\mathbf{T}}} \| I_{\text{real}} - \text{Render}(\text{Map}, \hat{\mathbf{T}}) \|^2$$

$\text{Render}$ 함수가 미분 가능하므로 $\hat{\mathbf{T}}$에 대한 그래디언트를 바로 계산한다.

**미분 가능 ICP**: 전통 ICP의 nearest neighbor search와 SVD를 미분 가능하게 만들어 포인트 클라우드 정합을 학습 루프 안에 넣는다.

**미분 가능 pose graph optimization**: iSAM2 같은 최적화를 미분 가능하게 만들면 백엔드의 오차 신호로 프론트엔드(feature extraction, matching)를 학습할 수 있다.

### 13.2.3 현재의 한계와 가능성

**한계**:
- **일반화(generalization)**: 학습 분포 밖의 camera, motion, 조명, 동적 장면에서는 성능이 달라질 수 있다. DROID-SLAM의 원 논문은 synthetic-to-real generalization을 평가하지만 모든 환경과 sensor modality를 포괄하지는 않는다.
- **검증 범위**: 전통적 pipeline에서는 residual과 각 module을 따로 검사하기 쉽다. 학습된 update operator의 거동은 training distribution과 평가 protocol에 더 강하게 묶인다. 일반적인 비선형 SLAM 문제의 전역 수렴과 일관성은 두 방식 모두 별도로 검증해야 한다.
- **계산 비용(Computational cost)**: DROID-SLAM 같은 학습 기반 시스템은 GPU에서 돌리도록 만들어져, 임베디드 보드에서 실시간으로 돌리기 어렵다.
- **해석 가능성(Interpretability)**: 실패했을 때 원인을 찾기 어렵다. 전통 시스템에서는 어느 모듈에서 실패했는지 따라가 볼 수 있지만, end-to-end 시스템은 블랙박스에 가깝다.

**가능성**:
- LoFTR·RoMa 같은 학습 matcher는 어려운 benchmark에서 좋은 matching 결과를 보고했다(§13.1.3).
- 미분 가능 최적화 덕분에 전통 구조 안에 학습 모듈을 끼워 넣을 길이 열렸다.
- multi-task learning(다중 과제 학습): [Kendall et al. (CVPR 2018)](https://arxiv.org/abs/1705.07115)은 단안 이미지 하나로 depth, semantic segmentation, instance segmentation을 함께 학습한 모델이 과제별로 따로 학습한 모델보다 나았다고 보고했다.

---

## 13.3 Spatial Memory & Scene Graphs

공간 기억(spatial memory)에는 포인트 클라우드 말고도 계층, 관계, 시간 정보가 들어간다. 사람은 "주방에 냉장고가 있고, 그 안에 우유가 있었다"는 식으로 공간을 기억한다. 고수준 공간 기억 시스템은 로봇의 맵을 이런 구조로 짠다.

### 13.3.1 Persistent Spatial Memory

전통 SLAM 맵에는 지금 이 순간의 환경 상태만 담긴다. Persistent spatial memory(지속 공간 기억)는 환경이 시간에 따라 어떻게 바뀌었는지까지 남겨 두는 장기 공간 기억이다.

Persistent spatial memory의 과제:

1. **Episodic spatial memory(일화 공간 기억)**: "지난주 화요일에 여기 상자가 있었다"처럼 시간, 장소, 사건을 이어 둔다.
2. **Semantic persistence**: 벽·건물 같은 영구 요소와 사람·차량 같은 일시 요소를 갈라, 장기 맵이 흔들리지 않게 한다.
3. **Incremental forgetting**: 오래된 관측의 세부는 서서히 잊고, 핵심 구조는 남긴다.

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

[Hydra](https://arxiv.org/abs/2201.13360)가 만드는 것 같은 3D Scene Graph에 언어 명령과 LLM을 붙이는 연구가 이어졌다. 한 예로 [SayPlan (Rana et al., CoRL 2023)](https://arxiv.org/abs/2307.06135)은 3D scene graph를 LLM에 넣어 여러 층·여러 방이 있는 건물에서 작업 계획을 세운다. Hydra 자체는 계층을 만들고 계층 단위로 loop closure를 하는 데까지 다룬다.

**Scene Graph + Language**: Scene graph에 자연어 인터페이스를 붙이면 로봇에게 "거실 소파 옆 테이블 위에 있는 리모컨을 가져와" 같은 명령을 내릴 수 있다. 이 명령을 scene graph를 위에서부터 차례로 찾아 내려가는 탐색으로 바꾼다:

1. "거실" → Room 노드 탐색
2. "소파 옆의 테이블" → Room 내 Object 노드의 관계 탐색
3. "리모컨" → 그 Table 위에 있다는 관계를 만족하는 리모컨 Object 탐색
4. 경로 계획과 manipulation

**Scene Graph + LLM**: GPT-4 같은 LLM에 scene graph를 넣어 고수준 추론을 맡긴다. "이 방에서 사람이 넘어지면 가장 가까운 전화기는 어디 있는가?" 같은 질문에 답하게 하는 식이다.

**동적 Scene Graph**: Hydra의 agent 계층(§11.4.2의 계층 표)에는 로봇 자신의 궤적(pose graph)만 들어간다. 기하·의미 mesh와 place·room 계층 그리고 그 위의 최적화는 정적 구조를 가정한다. 동적 scene graph는 사람·차량 같은 움직이는 에이전트와 그 관계를 지도 구조 안에서 실시간으로 고쳐 나가려 한다. 이 정보는 사회적 내비게이션(social navigation)과 인간-로봇 상호작용(HRI, human-robot interaction)에 쓰인다.

### 13.3.3 시계열 공간 기억 관리

오래 운용하는 로봇에는 공간 기억을 언제 만들고, 얼마나 유지하고, 언제 지울지 정하는 전략이 필요하다.

**Hierarchical forgetting**: 정확한 텍스처나 개별 포인트 같은 세부 정보는 시간이 지날수록 해상도를 낮춘다. 방의 배치나 통로 연결 같은 구조 정보는 계속 남긴다.

**Event-triggered update**: 맵 전체를 주기적으로 갱신하지 않고, 변화를 감지한 영역만 고친다.

**Compression**: 시간이 지날수록 맵을 조금씩 압축한다. 예를 들어 dense point cloud를 sparse landmark로, 다시 topological graph로, 마지막에는 semantic description으로 줄인다.

---

## 13.4 Cross-Modal Representation

LiDAR 포인트 클라우드와 카메라 이미지는 같은 환경을 보지만 데이터 형태가 다르다. 종류가 다른 센서의 관측을 같은 표현 공간에 놓고 비교하려는 것이 cross-modal representation이다.

### 13.4.1 이종 센서 간 표현 정렬 문제

표현을 맞추기 어려운 이유:

- **차원 불일치**: LiDAR는 3D 포인트, 카메라는 2D 이미지, radar는 range-Doppler map을 낸다.
- **정보 비대칭**: LiDAR 점에는 텍스처가 없고, 카메라 한 대로는 절대 거리를 알 수 없다.
- **센서 특유의 아티팩트**: LiDAR의 motion distortion, 카메라의 rolling shutter, radar의 speckle noise처럼 센서마다 노이즈 패턴이 다르다.

### 13.4.2 Contrastive Learning for Cross-Modal Alignment

Contrastive learning(대조 학습)은 같은 장소나 물체를 다른 센서로 본 관측은 가깝게, 다른 장소나 물체의 관측은 멀게 놓이도록 표현을 학습한다.

$$\mathcal{L}_{\text{contrastive}} = -\log \frac{\exp(\text{sim}(f_L(\mathbf{x}_L), f_C(\mathbf{x}_C)) / \tau)}{\sum_{j} \exp(\text{sim}(f_L(\mathbf{x}_L), f_C(\mathbf{x}_C^j)) / \tau)}$$

여기서 $f_L$은 LiDAR encoder, $f_C$는 카메라 encoder, $\tau$는 temperature, $(\mathbf{x}_L, \mathbf{x}_C)$는 같은 장소의 LiDAR-카메라 쌍, $\mathbf{x}_C^j$는 분모의 후보 표본이다. 합은 양성 표본 $\mathbf{x}_C$와 negative sample을 모두 포함한다.

**Cross-modal place recognition**: LiDAR로 만든 맵 안에서 카메라만으로 위치를 찾는 경우다. 두 descriptor가 같은 공간에 있으면 카메라 이미지로 LiDAR 맵을 검색한다.

**LC$^2$** (Lee et al., RA-L 2023): LiDAR와 카메라 사이 cross-modal place recognition 방법이다. LiDAR를 range image로 투영하고 카메라 이미지도 disparity/depth 표현으로 바꾸어, 두 표현의 feature를 공통 공간으로 정렬한다(§9.4.3).

### 13.4.3 Knowledge Distillation

한 모달리티로 학습한 teacher 모델의 지식을 다른 모달리티의 student 모델에 옮긴다.

**LiDAR → Camera distillation**: LiDAR의 정확한 3D 정보로 학습한 모델의 지식을 카메라만 쓰는 모델에 옮긴다. 실제로 돌릴 때는 카메라만으로 LiDAR에 가까운 3D 정보를 얻으려는 것이다.

**Camera → LiDAR distillation**: 카메라가 주는 의미 정보를 LiDAR 처리 모델에 옮긴다. 예를 들어 CLIP feature를 LiDAR 포인트에 붙여 두면 텍스트로 LiDAR 맵을 검색할 수 있다.

### 13.4.4 아직 풀리지 않은 문제들

1. **모달리티에 매이지 않는(modality-agnostic) 표현**: LiDAR, 카메라, radar, event camera 어느 입력이든 같은 표현 공간에 넣는 범용 encoder를 만들 수 있을까?

2. **Temporal alignment**: 모달리티마다 관측 시각과 주기가 다르므로, 비동기 관측을 공통 표현으로 융합하는 방법이 필요하다.

3. **Partial observation**: 비 때문에 LiDAR가, 어둠 때문에 카메라가 잠시 제 몫을 못 할 때 남은 센서만으로도 표현이 어긋나지 않아야 한다.

---

## 13.5 Event Camera 기반 퓨전

이벤트 카메라(Event Camera, Dynamic Vision Sensor, DVS)는 프레임 카메라와 다르게 동작한다. 픽셀마다 따로 밝기 변화를 감지해, 변화가 생긴 순간에만 이벤트를 비동기로 내보낸다.

### 13.5.1 Event Camera의 원리와 장점

각 이벤트를 $(x, y, t, p)$로 나타낸다:

- $(x, y)$: 픽셀 좌표
- $t$: 마이크로초 단위 타임스탬프
- $p \in \{+1, -1\}$: 극성 (밝아짐 / 어두워짐)

이벤트가 생기는 조건:

$$|\log I(x, y, t) - \log I(x, y, t_{\text{last}})| \geq C$$

이때 극성(polarity)은 $p = \text{sign}(\log I(x, y, t) - \log I(x, y, t_{\text{last}}))$로 정해진다. 여기서 $I$는 밝기, $C$는 대비 임계값(contrast threshold), $t_{\text{last}}$는 해당 픽셀에서 마지막으로 이벤트가 생긴 시점이다.

**장점**:

| 특성 | 프레임 카메라 | Event 카메라 |
|------|------------|-------------|
| 시간 해상도 | 프레임 주기 단위 (예: 30 fps) | 마이크로초 단위 |
| 동적 범위 | ~60 dB | >120 dB ([Gallego et al., TPAMI 2022](https://arxiv.org/abs/1904.08405)) |
| 모션 블러 | 있음 | 거의 없음 |
| 데이터 출력 | 균일 프레임 | 비동기 이벤트 |
| 픽셀 밝기 변화 없음 | 프레임 정보 제공 | 이상적으로 이벤트 없음 |
| 전력 소비 | 장치별 차이가 큼 | 센서 자체는 활동량에 비례해 낮음. 판독·처리를 포함한 시스템 전력은 이벤트 발생률에 따라 커짐 |

Event camera는 고속 회전, 급격한 조명 변화(터널 진입/출구), 어두운 곳에서도 정보를 내므로 일반 카메라가 약한 상황을 메운다.

### 13.5.2 Event + Frame 퓨전

**Event-enhanced frame tracking**: 프레임과 프레임 사이의 빠른 움직임을 이벤트로 따라가, 프레임 기반 VO가 비워 두는 간격을 채운다. 카메라가 빠르게 움직일 때 추적이 덜 끊긴다. 대비가 부족해 이벤트가 적거나 반대로 이벤트가 포화해 대역폭이 막히면 여전히 끊길 수 있다.

**Event-aided HDR**: 이벤트의 넓은 동적 범위로 프레임 이미지에서 노출이 모자라거나 넘친(under/over-exposed) 영역의 정보를 메운다.

### 13.5.3 Event 단독 VO와 Event + IMU 퓨전

**[EVO](https://doi.org/10.1109/LRA.2016.2645143)** (Rebecq et al., RA-L 2017): Event-based Visual Odometry다. IMU 없이 이벤트만으로 6-DoF 카메라 포즈 추적과 semi-dense 3D 맵 작성을 병렬로 돌린다.

**[Ultimate SLAM](https://arxiv.org/abs/1709.06310)** (Vidal et al. 2018): Event camera, 프레임 카메라, IMU를 tightly-coupled 방식으로 묶었다. 원 논문은 Event Camera Dataset에서 이벤트만 쓴 pipeline보다 정확도가 130%, 프레임만 쓴 VIO보다 85% 좋아졌다고 보고했다. 센서별로 맡는 몫:
- 프레임 카메라: 움직임이 적은 장면에서 풍부한 텍스처
- Event camera: 빠르게 움직일 때도 끊기지 않는 추적
- IMU: 스케일 복원과 빠른 모션 예측

**현재 과제**:
- Event camera가 내는 비동기 이벤트 스트림은 프레임 기반 컴퓨터 비전 파이프라인에 바로 넣기 어렵다. 프레임으로 집계할 때 잃는 정보는 표현마다 다르다. 이벤트 수 histogram에서는 윈도 안의 시각 정보가 사라진다. time surface는 픽셀마다 최근 이벤트 시각을 남겨 순서를 상당 부분 보존한다. 로그 강도 감지에서 오는 높은 동적 범위는 집계해도 남는다.
- Event camera의 가격과 해상도는 모델별 차이가 크며, 같은 가격대의 frame camera보다 선택지가 제한적이다.
- 학습 데이터가 부족하다. [Gallego et al. 서베이](https://arxiv.org/abs/1904.08405)에 따르면 인식용 event 데이터셋은 기존 컴퓨터 비전 데이터셋보다 규모가 작고, 주석을 다는 표준 도구나 형식도 아직 없다.

**최근 시스템과 연구 (2024~2026)**:

- **[EvenNICER-SLAM](https://arxiv.org/abs/2410.03812)** (2024): NICE-SLAM에 event loss를 역전파하는 경로를 더했다. RGB-D 입력 빈도를 낮춘 조건에서 NICE-SLAM보다 tracking 결과가 좋았다고 보고했다.
- **Event-based 3D reconstruction survey** ([arxiv:2505.08438](https://arxiv.org/abs/2505.08438), Computational Visual Media 2026): 이벤트 카메라 3D 재구성만 다룬 서베이다. 입력에 따라 stereo, monocular, multimodal로 나누고, 다시 geometry 기반, 딥러닝, NeRF·3DGS 같은 neural rendering 기법으로 분류했다.

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
        
        for vx in np.linspace(-200, 200, 20):   # px/s. 탐색 범위는 예시값
            for vy in np.linspace(-200, 200, 20):
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

4D imaging radar는 거리(range), 방위각(azimuth), 고각(elevation), 도플러 속도(Doppler velocity)의 네 값을 잰다. 거리·방위각·도플러를 재던 기존 automotive radar에 고각 분해능을 더해, 3차원 위치와 시선 방향 속도를 함께 얻는다.

**FMCW radar의 거리/속도 측정 원리**: [Han et al.의 4D radar 서베이](https://arxiv.org/abs/2306.04242) 2절에 따르면 지금 나와 있는 4D radar는 대부분 FMCW 방식이다. 송신 주파수를 시간에 따라 선형으로 올리는 신호(chirp)를 쏘아 반사 신호와의 비트 주파수(beat frequency)로 거리를 재고(§2.5.1), chirp와 chirp 사이의 위상 변화로 속도를 잰다:

$$v = \frac{\lambda \cdot \Delta\phi}{4\pi \cdot T_c}$$

여기서 $v$는 타겟의 radial velocity, $\lambda$는 캐리어 파장, $\Delta\phi$는 연속된 두 chirp 사이의 위상 변화, $T_c$는 chirp 주기이다.

### 13.6.1 악천후 Robustness

4D imaging radar는 가시광·근적외선 센서의 성능이 저하되는 조건에서도 거리와 Doppler를 대체로 검출한다. 그래도 악천후에서는 성능이 떨어진다. 다음 표는 [Zhang et al. (ISPRS J. 2023)](https://doi.org/10.1016/j.isprsjprs.2022.12.021) 서베이의 센서별 기상 영향 표(Table 1)와 3.2~3.3절, radome 수막 측정([Arage et al.](https://duepublico2.uni-due.de/servlets/MCRFileNodeServlet/duepublico_derivate_00014694/Final_Papers/GM0022-F.pdf))에 근거한 상대 경향이다. 장치별 성능은 따로 검증한다.

| 조건 | 카메라의 경향 | LiDAR의 경향 | 4D Radar의 경향 | 주요 변수 |
|------|---------------|--------------|------------------|----------|
| 맑은 날 | 색·텍스처 정보가 풍부함 | 조밀한 기하 정보에 유리함 | 거리·Doppler를 주지만 각 해상도가 제한될 수 있음 | 타겟 RCS, 거리, 각 해상도 |
| 비 | 대비 저하와 렌즈 젖음 | 후방 산란과 감쇠 | 상대적으로 덜 저하될 수 있으나 rain clutter와 젖은 radome 영향 | 강우량, radome 상태, 주파수 |
| 안개 | 대비와 가시거리 저하 | 후방 산란과 감쇠 | 대체로 상대적 강건성이 높음 | 안개 농도, 거리, 타겟 RCS |
| 눈/먼지 | 가림과 대비 저하 | 입자 후방 산란 | clutter·multipath와 표면 적설 영향이 남음 | 입자 농도, 축적, 타겟 RCS |
| 야간 | 조명과 노출에 의존 | 능동 거리 측정 유지 | 능동 거리·Doppler 측정 유지 | 조명, 표면 반사율, 타겟 RCS |
| 직사광선 | glare와 포화 가능 | 수광부의 태양 배경광 영향 가능 | 영향이 거의 없음 | 수광부·필터, 장착 방향 |

야간은 기상 대신 조명에 관한 행이라, 출처가 [Han et al.의 4D radar 서베이](https://arxiv.org/abs/2306.04242) Table I의 조명 강건성 비교(RGB 카메라 낮음, LiDAR·4D radar 높음)다.

밀리미터파는 일반적으로 가시광·근적외선보다 안개 크기의 입자에 덜 민감하다. 그래도 강한 강수, 젖거나 얼어붙은 radome, multipath, RF 간섭, 낮은 RCS의 타겟, 제한된 각 해상도는 탐지 거리를 줄이거나 오경보를 늘릴 수 있다. 목표 기상 조건에서 탐지 거리, 오경보율, Doppler 오차를 장치별로 검증한다.

### 13.6.2 4D Radar + Camera Fusion

4D radar와 카메라는 실패하는 조건(failure mode)이 달라서, 함께 쓰면 서로의 약점을 메운다. 비용과 악천후 성능은 센서 모델, 장착, 처리 장치, 목표 데이터셋에서 따로 검증한다.

**BEV 기반 퓨전**: 카메라 이미지에서 LSS나 BEVFormer 방식으로 BEV feature를 뽑고, radar 포인트를 같은 BEV 공간에 투영해 합친다.

**Radar의 Doppler 정보 활용**: 4D radar는 각 포인트의 시선 방향 속도(radial velocity)를 직접 측정한다. 카메라와 pulsed time-of-flight LiDAR에는 없는 정보다. FMCW 방식 LiDAR는 같은 원리로 점별 속도를 재므로 예외다.

- **동적 객체 분류**: 자차 속도를 보상한 Doppler 잔차로 정적 배경과 움직이는 객체의 후보를 가른다. 시선에 직교해 움직이는 객체는 Doppler 기여가 0이어서 다른 단서가 필요하다. 임계값은 radar 속도 잡음과 자차 속도 추정 오차를 함께 반영해야 한다.
- **Ego-motion estimation**: 정적 포인트의 Doppler로 자차 속도를 추정한다. IMU가 없어도 된다.
- **Tracking 지원**: 물체의 속도를 tracking에 바로 쓴다.

$$v_r = (\mathbf{v}_{\text{obj}} - \mathbf{v}_{\text{ego}}) \cdot \hat{\mathbf{r}}$$

여기서 $v_r$은 측정된 radial velocity, $\mathbf{v}_{\text{obj}}$는 객체 속도, $\mathbf{v}_{\text{ego}}$는 ego 속도, $\hat{\mathbf{r}}$은 radar → 타겟 방향 단위 벡터($\|\hat{\mathbf{r}}\| = 1$)다. 정적 객체($\mathbf{v}_{\text{obj}} = \mathbf{0}$)의 경우 $v_r = -\mathbf{v}_{\text{ego}} \cdot \hat{\mathbf{r}}$이 된다.

### 13.6.3 Radar Odometry의 최근 발전

Radar odometry 연구는 다음과 같이 나눠 볼 수 있다:

**FMCW radar odometry**: Navtech 같은 회전식 FMCW radar로 하는 odometry다. Range-azimuth 이미지에서 특징점을 뽑아 맞추고 ego-motion을 추정한다.

**4D radar odometry**: 4D radar 포인트 클라우드로 하는 odometry다. ICP나 feature matching처럼 LiDAR odometry 방법을 가져다 쓰지만, 점이 성기고 노이즈가 커서 어렵다.

**Doppler 기반 ego-velocity estimation**: 정적 포인트의 Doppler 측정으로 ego-velocity를 바로 추정한다. RANSAC으로 동적 포인트를 걸러 내고, 남은 정적 포인트의 Doppler로 $\mathbf{v}_{\text{ego}}$를 구한다:

$$v_r^{(k)} = -\mathbf{v}_{\text{ego}} \cdot \hat{\mathbf{r}}^{(k)} \quad \text{(정적 포인트)}$$

여기서 $k$는 포인트 인덱스를 뜻한다. 3D 속도 벡터를 추정하려면 정적 포인트가 최소 3개 필요하고, 이들의 시선 방향 단위 벡터가 선형 독립이어야 한다.

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
| **RadarScenes** | Radar, Camera (참고용), Odometry | 도심 | 기존 automotive radar 포인트 + semantic labels (point-level annotation). LiDAR는 포함되지 않는다 |
| **nuScenes** | Camera, LiDAR, Radar | 도심 | 5개 radar 포함, 악천후 일부 |
| **[View-of-Delft](https://doi.org/10.1109/LRA.2022.3147324)** (Palffy et al. 2022) | Stereo camera, 64채널 LiDAR, 3+1D radar | 도심 (Delft) | 3+1D radar + 3D bounding box annotation |

**최근 시스템과 데이터셋 (2025~2026)**:

- **[SNAIL Radar](https://arxiv.org/abs/2407.11705)** (Huai et al., IJRR 2025): 4D radar 기반 SLAM 평가 벤치마크로, 손에 든 장비, 전기자전거, SUV로 모은 44개 시퀀스가 들어 있다.
- **[4D Radar-Inertial Odometry](https://arxiv.org/abs/2412.13639)** (RA-L 2026): Radar 장면을 voxel 대신 3D Gaussian 묶음으로 나타내고, 정합 가설 여러 개를 함께 최적화해 local optimum에 덜 빠지게 한다. 공개 4D radar 데이터셋에서 기존 정합 알고리즘과 비슷하거나 나은 결과를 보고했다.

---

## 마무리

센서 모델링(Ch.2)에서 출발한 흐름은 캘리브레이션(Ch.3), 상태 추정 이론(Ch.4), 특징점 매칭(Ch.5), VO/VIO(Ch.6), LiDAR odometry(Ch.7), 멀티센서 퓨전(Ch.8), Place Recognition(Ch.9), Loop Closure(Ch.10), 공간 표현(Ch.11), 실전 시스템(Ch.12), 연구 프런티어(Ch.13)까지 이어졌다.

- **전통 방법은 여전히 기반이다.** 칼만 필터, ICP, RANSAC, factor graph는 수십 년 전에 나왔지만 이 가이드에서 다룬 2025년까지의 시스템도 이 방법들로 짜여 있다.
- **지각(perception)에서는 학습 기반 방법이 높은 점수를 보고했다.** 특징점 매칭, 깊이 추정, Place Recognition에서는 학습 기반 방법이 여러 공개 벤치마크에서 전통 방법보다 높은 점수를 보고했다. 우위의 범위는 데이터셋과 평가 프로토콜에 따라 달라진다.
- **추론(inference)에서는 전통과 학습이 함께 쓰인다.** 이 가이드에서 다룬 상태 추정 backend는 필터나 최적화로 짜여 있고, DROID-SLAM은 미분 가능 최적화로 둘을 섞었다.
- **Foundation model이 파이프라인에 들어오고 있다.** 2023년에 나온 DINOv2, SAM 같은 범용 모델의 표현을 2024~2025년 연구들이 센서 퓨전 파이프라인 여러 단계에 넣었다(§13.1).
