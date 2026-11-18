# 13장 연구 전망

12장까지 다룬 실전 시스템과 벤치마크는 정교한 기하 모델과 확률적 추정을 토대로 이미 높은 완성도에 도달했다. 그럼에도 계절과 조명이 급변하는 장기 운용, 자연어 명령을 이해하는 고수준 추론, 그리고 기존 카메라나 LiDAR가 무력해지는 고속 기동·악천후 영역으로 발을 넓히면 전통 파이프라인의 한계가 드러난다. 이 장에서는 foundation model과 end-to-end 학습, 시계열 공간 기억, 그리고 event camera·4D radar 같은 신형 센서가 그 공백을 어떻게 메우려 하는지 현재까지 입증된 근거와 남아 있는 한계를 함께 짚어 본다.

---

## 13.1 Foundation Models for Spatial AI

DINOv2, CLIP, SAM처럼 대규모 데이터로 사전 학습한 범용 모델인 foundation model(FM)이 내놓는 표현을 센서 융합과 SLAM 파이프라인에 결합하는 연구가 활발하다. 대부분의 연구는 이 표현을 장소 검색(retrieval)이나 특징 매칭, 의미 사전 정보(semantic prior)로 활용하며, 최근에는 MASt3R-SLAM(§13.1.3)처럼 고밀도 기하(dense geometry) 예측 자체를 모델에 맡기는 시스템까지 등장했다.

### 13.1.1 DINOv2/CLIP Feature를 SLAM에 활용

**[DINOv2](https://arxiv.org/abs/2304.07193)의 시각 특징**: DINOv2는 별도 레이블 없이 자기지도 학습(self-supervised learning)으로 훈련한 ViT 모델로, 이미지를 패치로 나눈 뒤 각 패치에서 뽑아낸 token을 고밀도(dense) feature로 활용할 수 있다. 여러 downstream 실험에서 확인된 특성은 크게 세 가지다.

- 조명이나 계절이 다소 바뀌더라도 같은 장소 간 대응을 찾아내는 데 유용한 표현이 형성된다.
- 명시적 분류 학습 없이도 객체 범주와 장면의 의미적 맥락이 feature 유사도에 자연스럽게 반영된다.
- 패치 단위 feature를 그대로 가져와 특징점 대응이나 영상 분할 같은 다양한 downstream 과제에 재활용할 수 있다.

**AnyLoc의 접근**: [AnyLoc](https://arxiv.org/abs/2308.00688) (Keetha et al., RA-L 2024)은 이러한 DINOv2의 dense feature를 VLAD(Vector of Locally Aggregated Descriptors)로 응집해 장면 전체를 요약하는 전역 디스크립터를 만들고, VPR(Visual Place Recognition) 전용 파인튜닝 없이 도심·실내·항공·수중·지하 등 서로 다른 도메인의 벤치마크를 평가했다. 구체적으로 어떤 레이어와 facet을 썼는지, 그리고 recall 수치 차이는 §9.2.4에 정리해 두었다.

아래 `FoundationModelFeatureExtractor`는 DINOv2에서 패치 단위 dense feature를 추출한 뒤 VLAD 집계로 전역 장소 디스크립터를 만드는 AnyLoc 방식의 핵심 흐름을 개념적으로 구현한 코드다.

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

**SLAM 파이프라인 모듈별로 넣는 자리**: Foundation model이 기존 시각·기하 파이프라인 전체를 한꺼번에 대체하기보다는, 각 단계의 취약점을 메우는 형태로 자리 잡고 있다. 모듈별 대표 치환 사례는 다음과 같다.

| 파이프라인 모듈 | 전통 방법 | 학습 feature를 넣는 예 |
|---------------|-----------|-------------|
| Feature detection | FAST, ORB | SuperPoint + DINOv2 hybrid |
| Feature matching | BF + ratio test | SuperGlue/LightGlue, LoFTR |
| Place recognition | DBoW2, Scan Context | AnyLoc (DINOv2 + VLAD) |
| Semantic segmentation | 전용 모델 학습 | [SAM](https://arxiv.org/abs/2304.02643), open-vocab segmentation |
| Depth estimation | Stereo matching | [Depth Anything](https://arxiv.org/abs/2401.10891)의 relative-depth prior |
| Loop closure candidate | DBoW2, Scan Context | FM descriptor similarity 후 geometric verification |

### 13.1.2 Open-Vocabulary 3D Understanding

이미지와 텍스트를 같은 임베딩 공간에 정렬하는 CLIP의 성질을 3D 지도로 확장하면, 사전에 물체 클래스를 정해 두지 않아도 로봇이 자연어 질의로 공간을 탐색할 수 있다. 기본 동작 흐름은 네 단계로 요약된다.

1. SLAM으로 3D 맵(point cloud, mesh, 3DGS 등)을 구축한다.
2. 입력 이미지마다 영역별 CLIP visual feature를 추출한다.
3. 추출한 2D feature를 3D 맵의 대응 위치로 역투영해 결합한다.
4. 사용자가 "소화기를 찾아"라고 지시하면 CLIP text encoder로 문장을 임베딩 벡터로 바꾼 뒤, 3D 맵에서 유사도가 가장 높은 영역을 찾아낸다.

이 구조를 구현하는 방식은 3D 표현에 따라 갈린다. [ConceptFusion](https://arxiv.org/abs/2302.07241) (Jatavallabhula et al. 2023)은 위 절차를 충실히 따라 픽셀 정렬된 foundation model feature를 다중 시점 융합으로 3D SLAM 맵에 직접 얹고, 추가 학습 없이 텍스트·이미지·오디오로 맵을 검색한다. 이와 달리 [LERF](https://arxiv.org/abs/2303.09553) (Kerr et al. 2023)는 점 위에 feature를 붙이는 대신 CLIP 임베딩을 NeRF 내부의 연속적인 language field로 함께 학습하며, [OpenScene](https://arxiv.org/abs/2211.15654) (Peng et al. 2023)은 3D 포인트 클라우드에서 곧장 CLIP 공간의 feature를 예측하는 3D 네트워크를 별도 라벨 없이 증류·학습한다. 접근 방식은 다르지만, 세 시스템 모두 고정된 클래스 목록 없이 임의의 텍스트 질의로 3D 장면을 검색한다는 목표를 공유한다.

**현재의 한계**: 하지만 실전 배치에는 세 가지 제약이 남아 있다.
- CLIP feature는 패치 단위로 계산돼 공간 해상도가 거칠어, 작은 물체의 정확한 경계와 위치를 짚어 내기 어렵다.
- 같은 물체라도 바라보는 시점에 따라 visual feature가 흔들려 다중 시점 간 3D 일관성을 유지하기 까다롭다.
- 매 프레임마다 무거운 foundation model을 돌려 feature를 뽑으면 실시간 운용에 필요한 연산 예산을 넘기기 쉽다.

### 13.1.3 모듈별 근거와 경계

Foundation model을 SLAM 파이프라인에 투입했을 때 실제로 얼마나 효과가 검증됐는지는 모듈과 평가 조건에 따라 온도 차가 뚜렷하다.

**논문 벤치마크에서 활용 근거가 있는 영역**:
- **Visual place recognition**: AnyLoc은 별도 파인튜닝 없는 DINOv2+VLAD 조합만으로 다양한 도메인에서 높은 recall을 보고했다. 다만 DBoW2와는 입력 feature, 학습 조건, 추론 비용이 크게 다르므로, 실제 시스템에서 DBoW2를 대체하려면 목표 환경의 오탐률(false-positive rate)과 latency를 나란히 측정해 따져 봐야 한다.
- **Feature matching**: LoFTR·RoMa 같은 학습 기반 matcher는 텍스처가 빈약하거나 시점 차이가 큰 벤치마크에서 기존 기법보다 우수한 결과를 낸다. 반면 연산 예산이 빠듯하거나 캘리브레이션이 정밀하고 반복성(repeatability)이 중요한 환경에서는 전통적인 sparse feature가 여전히 합리적인 선택이다.
- **Monocular depth**: [Depth Anything](https://arxiv.org/abs/2401.10891)의 기본 출력은 절대 거리가 아닌 상대 깊이(relative depth)다. 따라서 미터 단위의 metric scale이 필수적인 센서 융합에 쓰려면 metric depth용 파인튜닝이나 다른 센서의 거리 제약이 동반돼야 하며, 예측 깊이의 불확실성(uncertainty) 모델링도 따로 챙겨야 한다.

**별도의 estimator가 계속 필요한 영역**:
- **LiDAR odometry**: ICP·LOAM·FAST-LIO2 같은 기하 기반 기법과 학습 기반 기법의 우열은 센서 특성과 데이터셋에 따라 갈린다. 설령 학습 기반 feature나 correspondence를 도입하더라도, 최종 단계에서 모션 모델과 정합 잔차를 검증하는 기하 절차가 빠져서는 안 된다.
- **IMU integration**: preintegration은 강체 운동학(kinematics)과 물리적 노이즈 모델을 factor로 압축한 수학적 구조다. 학습 모델로 바이어스나 노이즈 보정을 덧붙일 수는 있어도, preintegration의 관측 모델과 불확실성 전파 체계 자체는 그대로 유지돼야 한다.
- **Backend optimization**: Factor graph와 iSAM2는 이종 관측을 확률적으로 결합하는 추정기다. Foundation model 출력을 백엔드 factor로 통합하려면 해당 출력의 공분산, 이상치(outlier) 특성, 관측 간 상관관계를 명시적으로 규정할 수 있어야 한다.

**최근 시스템 (2024~2025)**: 최근에는 이러한 경계를 넓히는 시도도 빠르게 이어지고 있다.

- **[MASt3R-SLAM](https://arxiv.org/abs/2412.12392)** (Murai et al. CVPR 2025): 2-view 3D 재구성 모델인 MASt3R를 백본으로 삼은 단안 dense SLAM으로, 15 FPS 속도로 전역 일관성이 잡힌 포즈와 고밀도 3D 형상을 동시에 복원한다고 보고했다(§11.3.2).
- **[Depth Anything V2](https://arxiv.org/abs/2406.09414)** (Yang et al. NeurIPS 2024): 노이즈가 섞인 실사 라벨을 전면 합성 데이터로 대체해 거대 teacher 모델을 훈련한 뒤, 이 teacher가 대규모 비라벨 실사 영상에 생성한 pseudo-label로 student 모델을 학습시켰다. Metric depth 라벨로 파인튜닝한 모델도 별도로 배포했지만, 융합 파이프라인의 depth prior로 쓰려면 실제 환경에서의 스케일 일관성과 불확실성을 따로 검증해야 한다.

---

## 13.2 End-to-End Learned SLAM

전통적인 SLAM은 feature extraction → matching → motion estimation → mapping → loop closure → optimization에 이르는 각 단계를 수작업으로 설계해 이어 붙인다. End-to-end 학습은 이 파이프라인 전체를 하나의 미분 가능한 신경망 구조로 엮어, 원시 센서 입력에서 곧바로 카메라 포즈와 지도를 도출하도록 통째로 최적화한다.

### 13.2.1 현재의 대표 시스템

**[DROID-SLAM](https://arxiv.org/abs/2108.10869)** (Teed & Deng 2021)은 학습 기반 visual SLAM의 이정표로 꼽히는 공개 시스템이다. Optical flow 모델인 RAFT에서 따온 반복 업데이트 연산자(iterative update operator)와 미분 가능한 Dense Bundle Adjustment(DBA) 레이어를 결합해, 오차 신호가 프런트엔드까지 역전파되도록 전체 시스템을 학습시킨다. 흥미로운 점은 합성 데이터셋(TartanAir)만으로 학습한 단일 모델 가중치를 monocular·stereo·RGB-D 입력에 공통으로 적용하고, 별도의 loop closure 모듈 없이도 co-visibility 그래프의 장거리 엣지로 루프를 자연스럽게 닫는다는 점이다. 구체적인 내부 구조와 한계는 §6.5.3에서 다뤘다.

### 13.2.2 Differentiable SLAM Components

전체 파이프라인을 한 번에 end-to-end로 갈아엎기보다, 전통적인 기하 최적화 블록 일부를 미분 가능하게 만들어 학습 루프와 결합하는 절충안도 활발하다. 이렇게 하면 기하학적 해석 가능성을 유지하면서도 백엔드 오차로 프런트엔드 가중치를 개선할 수 있다.

**미분 가능 렌더링**: NeRF와 3DGS는 3D 장면에서 2D 이미지를 그리는 렌더링 과정 자체가 미분 가능하다. 따라서 현재 지도에서 렌더링한 영상과 실제 카메라 영상 사이의 photometric loss를 최소화하는 방향으로 포즈 $\hat{\mathbf{T}}$의 그래디언트를 직접 계산해 역전파로 위치를 추적한다.

$$\hat{\mathbf{T}}^* = \arg\min_{\hat{\mathbf{T}}} \| I_{\text{real}} - \text{Render}(\text{Map}, \hat{\mathbf{T}}) \|^2$$

여기서 $\text{Render}$ 연산이 미분 가능하므로 별도 특징점 매칭 없이도 픽셀 오차만으로 $\hat{\mathbf{T}}$를 정밀하게 조율할 수 있다.

**미분 가능 ICP**: 전통 ICP의 최근접 이웃 탐색(nearest neighbor search)을 soft-assignment로 완화하고 SVD 풀이를 미분 가능하게 구현해, 포인트 클라우드 정합 과정 자체를 신경망 학습 루프 안에 포함시킨다.

**미분 가능 pose graph optimization**: iSAM2 같은 백엔드 최적화를 미분 가능 레이어로 구현하면, 전역 궤적 오차 신호를 프런트엔드의 feature extraction이나 matching 네트워크까지 거슬러 올려 학습시킬 수 있다.

### 13.2.3 현재의 한계와 가능성

학습 기반 SLAM은 빠른 발전에도 불구하고 실전 도입 전 짚어야 할 한계가 분명하다.

**한계**:
- **일반화(generalization)**: 학습 데이터 분포를 벗어난 카메라 모델, 기동 패턴, 조명, 동적 장애물이 나타나면 성능이 크게 흔들릴 수 있다. DROID-SLAM 원 논문도 synthetic-to-real 일반화 성능을 평가했으나, 모든 운용 환경과 센서 모달리티를 아우르는 보증은 되지 않는다.
- **검증 범위**: 모듈형 전통 파이프라인에서는 각 단계의 잔차와 이상 동작을 개별적으로 떼어 검사하기 쉽다. 반면 학습된 update operator의 거동은 훈련 분포와 평가 프로토콜에 강하게 종속된다. 물론 비선형 SLAM 문제의 전역 수렴성과 일관성을 별도로 검증해야 한다는 점은 두 방식 모두 마찬가지다.
- **계산 비용(Computational cost)**: DROID-SLAM을 비롯한 고밀도 학습 시스템은 고성능 GPU를 전제로 설계돼, 전력과 메모리가 제한된 임베디드 보드에서 실시간으로 굴리기 어렵다.
- **해석 가능성(Interpretability)**: 시스템이 실패했을 때 원인을 추적하기 어렵다. 전통 파이프라인은 특징점 부족인지 오매칭인지 최적화 발산인지 단계별로 원인을 좁혀 갈 수 있지만, end-to-end 모델은 내부 거동이 블랙박스에 가까워 디버깅 비용이 높다.

**가능성**:
- 그럼에도 LoFTR·RoMa 같은 학습 기반 matcher가 까다로운 벤치마크에서 기존 방법을 뛰어넘는 매칭 성능을 입증했듯(§13.1.3), 데이터 기반 표현력의 이점은 분명하다.
- 미분 가능 최적화 기법이 성숙하면서 전통적인 기하 백엔드 안에 학습 모듈을 안전하게 끼워 넣는 하이브리드 설계가 한층 수월해졌다.
- **Multi-task learning(다중 과제 학습)**의 시너지도 주목할 만하다. 예컨대 [Kendall et al. (CVPR 2018)](https://arxiv.org/abs/1705.07115)은 단안 영상 한 장으로 depth와 semantic segmentation, instance segmentation을 공동 학습한 모델이 과제별로 따로 훈련한 모델보다 나은 성능을 낼 수 있음을 보였다.

---

## 13.3 Spatial Memory & Scene Graphs

11장에서 다룬 기하·의미 지도가 공간의 형상과 물체 배치를 담는다면, 공간 기억(spatial memory)은 여기에 계층과 관계, 시간의 흐름까지 엮어 넣는다. 사람이 "주방 냉장고 안에 우유가 있었다"는 식으로 장소를 기억하듯, 고수준 공간 기억 시스템은 로봇의 지도를 시간과 맥락이 살아 있는 구조로 조직한다.

### 13.3.1 Persistent Spatial Memory

전통적인 SLAM 지도는 대개 현재 스캔 시점의 정적인 스냅샷만 유지한다. 반면 persistent spatial memory(지속 공간 기억)는 환경이 시간에 따라 어떻게 변해 왔는지 이력까지 보존하는 장기 기억 체계다. 이를 구현하려면 다음 세 가지 과제를 풀어야 한다.

1. **Episodic spatial memory(일화 공간 기억)**: "지난주 화요일에 이 구석에 상자가 놓여 있었다"처럼 특정 시점, 장소, 사건을 연결해 저장하고 질의할 수 있어야 한다.
2. **Semantic persistence**: 벽이나 기둥 같은 영구 구조물과 사람·이동 차량 같은 일시적 요소를 구분해, 일시적 변화 때문에 장기 지도의 뼈대가 오염되지 않도록 해야 한다.
3. **Incremental forgetting**: 운용 기간이 길어질수록 모든 세부 관측을 쌓아 둘 수 없으므로, 오래된 세부 정보는 점진적으로 잊고 핵심 구조만 압축해 남긴다.

아래 `PersistentSpatialMemory` 클래스는 공간 격자 키마다 시간·의미·신뢰도를 함께 기록하고, 시간 감쇠(decay) 기반의 회상과 변화 감지, 우선순위 기반 기억 정리(consolidation)까지 수행하는 개념적 구현 예시다.

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

[Hydra](https://arxiv.org/abs/2201.13360) 같은 3D scene graph는 건물–방–장소–물체를 계층 그래프로 엮어, 그 자체로 계층적 구조 생성과 loop closure를 수행한다. 최근에는 이 그래프 위에 언어 인터페이스와 대규모 언어 모델(LLM)을 얹어 고수준 작업 계획으로 넓히는 연구가 이어졌다. 대표적으로 [SayPlan (Rana et al., CoRL 2023)](https://arxiv.org/abs/2307.06135)은 3D scene graph를 LLM에 입력해 여러 층과 방에 걸친 장기 작업 계획을 세운다.

**Scene Graph + Language**: scene graph에 자연어 인터페이스를 붙이면 "거실 소파 옆 테이블 위에 있는 리모컨을 가져와" 같은 복합 명령을 그래프의 위 계층에서 아래 계층으로 좁혀 가는 탐색 문제로 치환할 수 있다.

1. "거실" → Room 계층에서 해당 방 노드를 찾는다.
2. "소파 옆 테이블" → 방 내부 Object 노드들 사이의 공간 관계를 추적한다.
3. "리모컨" → 그 Table 노드 위에 놓였다는 관계를 만족하는 목표 Object를 찾는다.
4. 찾은 위치로 경로를 계획하고 조작(manipulation)을 수행한다.

**Scene Graph + LLM**: 한 걸음 더 나아가 GPT-4 같은 LLM에 scene graph 전체 구조를 넘기면, "이 방에서 사람이 넘어졌을 때 가장 가까운 전화기는 어디 있는가?"처럼 상식과 공간 맥락을 함께 요구하는 고수준 추론까지 맡길 수 있다.

**동적 Scene Graph**: 다만 Hydra의 agent 계층(§11.4.2 계층 표)은 로봇 자신의 궤적(pose graph)만을 다루며, 기하·의미 mesh와 place·room 계층 및 그래프 최적화는 정적 환경을 전제로 한다. 동적 scene graph 연구는 여기서 나아가 보행자나 차량처럼 움직이는 외부 에이전트와 그 상호작용 관계까지 그래프 안에서 실시간으로 갱신하려 하며, 이는 사회적 내비게이션(social navigation)과 인간-로봇 상호작용(HRI, human-robot interaction)의 핵심 토대가 된다.

### 13.3.3 시계열 공간 기억 관리

로봇을 몇 주, 몇 달씩 연속 운용하려면 공간 기억을 언제 생성하고 얼마나 오래 붙들며 어떤 기준으로 비울지 체계적인 관리 정책이 필요하다.

- **Hierarchical forgetting**: 벽면의 세밀한 텍스처나 개별 포인트 같은 말단 디테일은 시간이 지나면서 해상도를 낮추거나 버리되, 방 배치와 통로 연결 같은 골격 구조는 계속 보존한다.
- **Event-triggered update**: 매 주기마다 지도 전체를 다시 계산하지 않고, 변화 감지가 일어난 국소 영역만 선별적으로 갱신해 연산량을 아낀다.
- **Compression**: 시간이 흐를수록 지도의 표현 밀도를 단계적으로 압축한다. 예컨대 고밀도 포인트 클라우드를 희소 랜드마크로, 다시 위상 그래프(topological graph)로, 최종적으로는 간결한 의미 기술(semantic description)로 요약해 장기 메모리를 유지한다.

---

## 13.4 Cross-Modal Representation

LiDAR 포인트 클라우드와 카메라 이미지, radar 신호는 물리적으로 동일한 공간을 관측하지만 센서마다 내놓는 데이터 형태는 판이하다. 이처럼 서로 다른 모달리티의 관측을 하나의 공통 표현 공간에 매핑해 직접 비교하고 검색할 수 있게 만드는 연구가 cross-modal representation이다.

### 13.4.1 이종 센서 간 표현 정렬 문제

서로 다른 센서의 출력을 같은 공간에 정렬하기가 까다로운 데는 세 가지 물리적·구조적 이유가 있다.

- **차원 불일치**: LiDAR는 3D 포인트 집합을, 카메라는 2D 격자 이미지를, radar는 range-Doppler 맵을 생성하므로 입력 텐서의 기하 구조부터 다르다.
- **정보 비대칭**: LiDAR 포인트에는 색상과 표면 텍스처가 빠져 있고, 단안 카메라 이미지에는 절대 깊이 정보가 없다.
- **센서 특유의 아티팩트**: LiDAR의 motion distortion, 카메라의 rolling shutter 왜곡, radar의 speckle noise처럼 센서마다 고유한 잡음과 왜곡 양상을 띤다.

### 13.4.2 Contrastive Learning for Cross-Modal Alignment

이질적인 모달리티를 공통 임베딩 공간에 묶는 대표적인 학습 전략은 대조 학습(contrastive learning)이다. 같은 장소나 물체를 서로 다른 센서로 포착한 관측 쌍은 임베딩 공간에서 가까이 끌어당기고, 서로 다른 장소나 물체의 관측은 멀리 밀어내도록 각 센서의 인코더를 동시에 훈련한다. 이를 InfoNCE 형태의 손실 함수로 쓰면 다음과 같다.

$$\mathcal{L}_{\text{contrastive}} = -\log \frac{\exp(\text{sim}(f_L(\mathbf{x}_L), f_C(\mathbf{x}_C)) / \tau)}{\sum_{j} \exp(\text{sim}(f_L(\mathbf{x}_L), f_C(\mathbf{x}_C^j)) / \tau)}$$

여기서 $f_L$은 LiDAR encoder, $f_C$는 카메라 encoder, $\tau$는 분포의 날카로움을 조절하는 temperature 파라미터다. $(\mathbf{x}_L, \mathbf{x}_C)$는 같은 장소를 찍은 LiDAR-카메라 양성(positive) 쌍을 뜻하며, 분모의 합은 양성 표본 $\mathbf{x}_C$와 다수의 음성(negative) 후보 표본 $\mathbf{x}_C^j$를 모두 포괄한다.

이 공통 공간이 형성되면 한 센서로 만든 지도를 다른 센서만으로 검색하는 **cross-modal place recognition**이 가능해진다. 예컨대 사전에 LiDAR로 정밀 지도를 구축해 두고, 실제 운용 시에는 가벼운 카메라 영상만으로 LiDAR 지도 안의 현재 위치를 찾아내는 식이다. 대표적인 구현인 **LC$^2$** (Lee et al., RA-L 2023)는 LiDAR 포인트 클라우드를 range image로 투영하고 카메라 이미지는 disparity/depth 표현으로 변환해 기하 간극을 좁힌 뒤, 두 모달리티의 feature를 공통 임베딩 공간으로 정렬한다(§9.4.3).

### 13.4.3 Knowledge Distillation

공통 공간 정렬과 더불어, 한 모달리티에서 훈련한 teacher 모델의 지식을 다른 모달리티만 입력으로 받는 student 모델에 전수하는 knowledge distillation도 널리 쓰인다.

- **LiDAR → Camera distillation**: LiDAR의 정확한 3D 기하 정보로 학습한 teacher 모델의 공간 추론 능력을 카메라 전용 student 모델에 증류한다. 배포 단계에서는 비싼 LiDAR 없이 카메라만으로 LiDAR에 버금가는 3D 인지 성능을 끌어내려는 전략이다.
- **Camera → LiDAR distillation**: 반대로 카메라가 가진 풍부한 의미 정보를 LiDAR 네트워크에 옮기기도 한다. 예컨대 CLIP의 시각-언어 임베딩을 LiDAR 포인트 클라우드 표현에 증류해 두면, 카메라 없이 LiDAR 스캔만으로도 자연어 텍스트 질의로 맵을 검색할 수 있다.

### 13.4.4 아직 풀리지 않은 문제

Cross-modal 표현 학습은 빠르게 진전하고 있지만 세 가지 핵심 난제가 남아 있다.

1. **모달리티에 매이지 않는(modality-agnostic) 표현**: LiDAR, 카메라, radar, event camera 가운데 어느 입력이 들어오든 동일한 표현 공간으로 매핑하는 범용 인코더를 구축할 수 있을까?
2. **Temporal alignment**: 센서마다 샘플링 주기와 관측 시각이 제각각인 비동기 스트림을 정보 손실 없이 공통 표현으로 융합하는 방법이 필요하다.
3. **Partial observation**: 폭우로 LiDAR가 약해지거나 어둠 속에서 카메라가 제 몫을 못 해 일부 모달리티가 빠지더라도, 남은 센서만으로 표현의 일관성이 무너지지 않고 성능을 유지해야 한다.

---

## 13.5 Event Camera 기반 융합

기존 프레임 카메라는 일정한 노출 주기로 화면 전체의 밝기를 한 번에 읽어 들인다. 이 방식은 일상 환경에서는 잘 동작하지만, 기체가 고속으로 회전해 모션 블러가 번지거나 터널 입출구처럼 명암비가 극단적으로 바뀌는 순간에는 물리적 한계에 부딪힌다. 이벤트 카메라(Event Camera, Dynamic Vision Sensor, DVS)는 이러한 프레임을 버리고, 픽셀마다 독립적으로 밝기 변화를 감지해 변화가 일어난 순간에만 비동기 이벤트를 출력하는 전혀 다른 원리로 동작한다.

### 13.5.1 Event Camera의 원리와 장점

이벤트 카메라가 내보내는 각 이벤트는 $(x, y, t, p)$의 튜플로 표현된다.

- $(x, y)$: 밝기 변화가 감지된 픽셀 좌표
- $t$: 마이크로초 단위의 정밀 타임스탬프
- $p \in \{+1, -1\}$: 밝기 변화의 극성(polarity, 밝아짐 $+1$ / 어두워짐 $-1$)

픽셀 $(x, y)$에서 마지막 이벤트가 발생한 시점 $t_{\text{last}}$와 비교해 로그 밝기의 변화폭이 대비 임계값(contrast threshold) $C$를 넘어서면 새 이벤트가 발생한다.

$$|\log I(x, y, t) - \log I(x, y, t_{\text{last}})| \geq C$$

이때 극성은 $p = \text{sign}(\log I(x, y, t) - \log I(x, y, t_{\text{last}}))$로 결정되며, 여기서 $I$는 픽셀 밝기다.

로그 스케일 감지와 비동기 판독 방식 덕분에 이벤트 카메라는 프레임 카메라와 뚜렷하게 대비되는 물리적 특성을 갖는다. 두 센서의 차이를 나란히 비교하면 다음과 같다.

| 특성 | 프레임 카메라 | Event 카메라 |
|------|------------|-------------|
| 시간 해상도 | 프레임 주기 단위 (예: 30 fps) | 마이크로초 단위 |
| 동적 범위 | ~60 dB | >120 dB ([Gallego et al., TPAMI 2022](https://arxiv.org/abs/1904.08405)) |
| 모션 블러 | 있음 | 거의 없음 |
| 데이터 출력 | 균일 프레임 | 비동기 이벤트 |
| 픽셀 밝기 변화 없음 | 프레임 정보 제공 | 이상적으로 이벤트 없음 |
| 전력 소비 | 장치별 차이가 큼 | 센서 자체는 활동량에 비례해 낮음. 판독·처리를 포함한 시스템 전력은 이벤트 발생률에 따라 커짐 |

이러한 특성 덕분에 event camera는 고속 회전이나 터널 진입·탈출 같은 급격한 조명 변화, 극저조도 환경처럼 일반 카메라가 무너지는 구간에서 강점을 발휘한다.

### 13.5.2 Event + Frame 융합

실전 시스템에서는 이벤트 카메라를 단독으로 쓰기보다 프레임 카메라와 나란히 배치해 서로의 빈틈을 메운다.

- **Event-enhanced frame tracking**: 프레임 카메라가 30 fps 간격으로 장면을 찍는 사이, 기체가 빠르게 움직이면 프레임 간 간극이 벌어져 특징점 추적이 끊기기 쉽다. 이 프레임 사이의 연속적인 움직임을 마이크로초 단위 이벤트 스트림으로 이어 추적 안정성을 높인다(다만 대비가 너무 약해 이벤트가 거의 발생하지 않거나, 반대로 텍스처가 복잡하고 움직임이 극심해 이벤트가 포화하면서 판독 대역폭이 막히면 추적이 끊길 수 있다).
- **Event-aided HDR**: 이벤트 센서의 120 dB가 넘는 넓은 동적 범위를 활용해, 일반 프레임 이미지에서 노출이 부족해 검게 뭉개지거나 과노출로 하얗게 날아간 영역의 세부 구조를 복원한다.

### 13.5.3 Event 단독 VO와 Event + IMU 융합

프레임 카메라 없이 이벤트 스트림 자체로 포즈를 추정하거나 IMU와 결합하는 연구도 꾸준히 축적돼 왔다.

- **[EVO](https://doi.org/10.1109/LRA.2016.2645143)** (Rebecq et al., RA-L 2017): IMU 도움 없이 이벤트 카메라만으로 6-DoF 카메라 포즈 추적과 semi-dense 3D 맵 작성을 병렬로 수행하는 event-based visual odometry다.
- **[Ultimate SLAM](https://arxiv.org/abs/1709.06310)** (Vidal et al. 2018): event camera와 프레임 카메라, IMU 세 센서를 tightly-coupled 최적화로 결합했다. 원 논문은 Event Camera Dataset 평가에서 이벤트 단독 파이프라인보다 정확도가 130%, 프레임 기반 VIO보다 85% 향상됐다고 보고했다. 세 센서가 각자 맡는 역할이 뚜렷한데, 프레임 카메라는 움직임이 잔잔한 장면에서 풍부한 텍스처를 제공하고, event camera는 고속 기동 중에도 끊기지 않는 추적을 유지하며, IMU는 미터 단위 스케일 복원과 급격한 모션 예측을 뒷받침한다.

그럼에도 이벤트 카메라가 주류 센서로 자리 잡기까지 풀어야 할 현실적 과제가 남아 있다.

- 비동기 이벤트 스트림은 2D 이미지 격자를 전제로 한 기존 컴퓨터 비전 파이프라인에 곧바로 투입하기 어렵다. 이를 프레임 형태로 집계할 때 어떤 정보가 손실되는지는 표현 방식에 따라 다른데, 단순 이벤트 수 히스토그램(histogram)은 윈도 내부의 세부 시간 순서가 사라지는 반면, 픽셀마다 최근 이벤트 시각을 기록하는 time surface는 시간 순서를 상당 부분 보존한다. 한편 로그 강도 감지에서 비롯된 넓은 동적 범위는 프레임으로 집계하더라도 그대로 유지된다.
- 상용 event camera의 가격과 해상도는 제품마다 편차가 크며, 같은 가격대의 프레임 카메라에 비해 하드웨어 선택지가 아직 좁다.
- 학습용 데이터셋이 부족하다. [Gallego et al. 서베이](https://arxiv.org/abs/1904.08405)가 지적하듯 인식용 event 데이터셋은 기존 컴퓨터 비전 데이터셋보다 규모가 훨씬 작고, 어노테이션을 다는 표준 도구나 데이터 형식도 아직 정립되지 않았다.

**최근 시스템과 연구 (2024~2026)**: 최근에는 신경망 렌더링과 결합한 이벤트 SLAM 연구와 종합 서베이가 이어지고 있다.

- **[EvenNICER-SLAM](https://arxiv.org/abs/2410.03812)** (2024): neural implicit SLAM인 NICE-SLAM에 event loss를 역전파하는 경로를 추가해, RGB-D 입력 주기를 낮춘 조건에서도 기존 NICE-SLAM보다 우수한 추적 결과를 보고했다.
- **Event-based 3D reconstruction survey** ([arxiv:2505.08438](https://arxiv.org/abs/2505.08438), Computational Visual Media 2026): 이벤트 카메라 기반 3D 재구성 분야를 집대성한 서베이로, 입력 모달리티(stereo·monocular·multimodal)와 방법론(기하 기반·딥러닝·NeRF/3DGS 기반 neural rendering)의 두 축으로 기존 연구를 체계화했다.

아래 `EventProcessor`는 비동기 이벤트 스트림을 프레임 기반 알고리즘에 연결하기 위한 두 가지 대표 표현(event histogram과 time surface)과 contrast maximization 기반 optical flow 추정 원리를 구현한 예시다.

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

## 13.6 4D Radar 융합

비·안개·눈처럼 가시광과 근적외선이 막히는 환경에서는 카메라와 LiDAR가 함께 약해진다. 밀리미터파 대역을 쓰는 4D imaging radar는 이 악천후 공백을 메울 유력한 센서다. 거리(range), 방위각(azimuth), 도플러 속도(Doppler velocity)만 내놓던 기존 차량용 radar에 고각(elevation) 분해능을 더해, 3차원 공간 위치와 시선 방향 속도를 동시에 측정한다.

**FMCW radar의 거리/속도 측정 원리**: [Han et al.의 4D radar 서베이](https://arxiv.org/abs/2306.04242) 2절에 따르면 현재 상용·연구용 4D radar는 대부분 FMCW 방식을 채택한다. 시간에 따라 주파수가 선형으로 증가하는 송신 신호(chirp)를 쏘고, 반사되어 돌아온 신호와 섞어 얻은 비트 주파수(beat frequency)로 거리를 재며(§2.5.1), 연속된 두 chirp 사이의 위상 변화량으로 타겟의 시선 방향 속도(radial velocity)를 계산한다.

$$v = \frac{\lambda \cdot \Delta\phi}{4\pi \cdot T_c}$$

여기서 $v$는 타겟의 radial velocity, $\lambda$는 반송파 파장, $\Delta\phi$는 연속된 두 chirp 간의 위상 차이, $T_c$는 chirp 주기다.

### 13.6.1 악천후 Robustness

4D imaging radar는 가시광·근적외선 기반 센서가 급격히 열화되는 기상 조건에서도 거리와 Doppler 속도를 비교적 안정적으로 검출한다. 물론 악천후의 영향을 전혀 받지 않는 것은 아니다. 아래 표는 기상 조건별 영향에 대해 [Zhang et al. (ISPRS J. 2023)](https://doi.org/10.1016/j.isprsjprs.2022.12.021) 서베이(Table 1 및 3.2~3.3절)와 radome 수막 실험([Arage et al.](https://duepublico2.uni-due.de/servlets/MCRFileNodeServlet/duepublico_derivate_00014694/Final_Papers/GM0022-F.pdf))을 바탕으로 세 센서의 상대적 경향을 정리한 것이며, 실제 장치별 절대 성능은 개별 검증이 필요하다.

| 조건 | 카메라의 경향 | LiDAR의 경향 | 4D Radar의 경향 | 주요 변수 |
|------|---------------|--------------|------------------|----------|
| 맑은 날 | 색·텍스처 정보가 풍부함 | 조밀한 기하 정보에 유리함 | 거리·Doppler를 주지만 각 해상도가 제한될 수 있음 | 타겟 RCS, 거리, 각 해상도 |
| 비 | 대비 저하와 렌즈 젖음 | 후방 산란과 감쇠 | 상대적으로 덜 저하될 수 있으나 rain clutter와 젖은 radome 영향 | 강우량, radome 상태, 주파수 |
| 안개 | 대비와 가시거리 저하 | 후방 산란과 감쇠 | 대체로 상대적 강건성이 높음 | 안개 농도, 거리, 타겟 RCS |
| 눈/먼지 | 가림과 대비 저하 | 입자 후방 산란 | clutter·multipath와 표면 적설 영향이 남음 | 입자 농도, 축적, 타겟 RCS |
| 야간 | 조명과 노출에 의존 | 능동 거리 측정 유지 | 능동 거리·Doppler 측정 유지 | 조명, 표면 반사율, 타겟 RCS |
| 직사광선 | glare와 포화 가능 | 수광부의 태양 배경광 영향 가능 | 영향이 거의 없음 | 수광부·필터, 장착 방향 |

위 표에서 야간 행은 기상이 아니라 조명 조건에 해당하므로, [Han et al.의 4D radar 서베이](https://arxiv.org/abs/2306.04242) Table I의 조명 강건성 비교(RGB 카메라 낮음, LiDAR·4D radar 높음)를 따랐다. 밀리미터파는 파장이 길어 안개 입자 수준의 산란에는 가시광·근적외선보다 훨씬 둔감하다. 하지만 폭우나 물기·결빙이 낀 radome, 다중 반사(multipath), RF 간섭, RCS가 작은 비금속 타겟, 제한된 각 해상도가 겹치면 유효 탐지 거리가 줄거나 허위 경보가 늘어날 수 있다. 따라서 목표 운용 기상 조건에서 탐지 거리와 오경보율, Doppler 오차를 장치 단위로 직접 확인해야 한다.

### 13.6.2 4D Radar + Camera Fusion

4D radar와 카메라는 무너지는 조건(failure mode)이 서로 어긋나 있어 함께 묶었을 때 상보 효과가 크다. 물론 실제 차량 비용과 악천후 내구성은 센서 사양, 장착 위치, 연산 보드, 목표 데이터셋에서 별도로 따져야 한다.

**BEV 기반 융합**: 카메라 영상에서 LSS나 BEVFormer 같은 변환으로 BEV feature를 끌어올린 뒤, 4D radar 포인트를 같은 BEV 격자에 투영해 결합하는 방식이 널리 쓰인다.

**Radar의 Doppler 정보 활용**: 4D radar는 각 반사점마다 시선 방향 속도(radial velocity)를 직접 측정해 내놓는다. 카메라나 펄스 비행시간(pulsed ToF) 기반 LiDAR에서는 곧바로 얻을 수 없는 물리량이다(동일 원리로 점별 속도를 재는 FMCW LiDAR는 예외다). 이 속도 정보를 활용하면 인식과 상태 추정에 세 가지 이점이 생긴다.

- **동적 객체 분류**: 자차 움직임 성분을 뺀 Doppler 잔차만 보더라도 정적 배경과 움직이는 물체 후보를 빠르게 가려낼 수 있다. 다만 시선 방향과 수직으로 이동하는 물체는 radial velocity 기여가 0이므로 다른 단서를 보태야 하며, 정적·동적 판정 임계값도 radar 속도 잡음과 자차 속도 추정 오차를 함께 고려해 잡아야 한다.
- **Ego-motion estimation**: 정적 환경에서 반사된 포인트들의 Doppler 값을 역으로 풀면 IMU 없이도 자차의 순간 속도를 추정할 수 있다.
- **Tracking 지원**: 물체의 위치뿐 아니라 순시 속도 측정치를 multi-object tracking 필터에 곧바로 넣어 추적 안정성을 높인다.

타겟의 3차원 속도 $\mathbf{v}_{\text{obj}}$와 자차 속도 $\mathbf{v}_{\text{ego}}$가 측정된 radial velocity $v_r$로 투영되는 관계는 시선 방향 단위 벡터 $\hat{\mathbf{r}}$($\|\hat{\mathbf{r}}\| = 1$)과의 내적으로 깔끔하게 나타난다.

$$v_r = (\mathbf{v}_{\text{obj}} - \mathbf{v}_{\text{ego}}) \cdot \hat{\mathbf{r}}$$

특히 정지해 있는 배경 물체($\mathbf{v}_{\text{obj}} = \mathbf{0}$)라면 식이 $v_r = -\mathbf{v}_{\text{ego}} \cdot \hat{\mathbf{r}}$로 단순화되는데, 바로 이 관계가 다음 절에서 볼 Doppler 기반 ego-motion 추정의 출발점이 된다.

### 13.6.3 Radar Odometry의 최근 발전

Radar 신호로 자차 움직임을 추정하는 odometry 연구는 크게 세 갈래로 발전해 왔다.

- **FMCW radar odometry**: Navtech 같은 기계식 회전 FMCW radar가 출력하는 range-azimuth 이미지에서 특징점을 뽑아 정합함으로써 ego-motion을 계산한다.
- **4D radar odometry**: 4D radar 포인트 클라우드에 ICP나 특징 매칭 같은 LiDAR odometry 기법을 적용하는 접근이다. 다만 LiDAR에 비해 점이 성기고 다중 반사 노이즈가 많아 순수 기하 정합의 난도가 높다.
- **Doppler 기반 ego-velocity estimation**: 위의 한계를 보완하기 위해 정적 포인트의 Doppler 측정치를 직접 활용한다. RANSAC으로 동적 객체에서 나온 포인트를 걸러 내고, 남은 정적 포인트의 속도 제약으로 3차원 자차 속도 $\mathbf{v}_{\text{ego}}$를 푼다.

$$v_r^{(k)} = -\mathbf{v}_{\text{ego}} \cdot \hat{\mathbf{r}}^{(k)} \quad \text{(정적 포인트)}$$

여기서 $k$는 정적 포인트 인덱스다. 미지수가 3차원 속도 벡터 $\mathbf{v}_{\text{ego}}$이므로, 해를 유일하게 구하려면 시선 방향 단위 벡터 $\hat{\mathbf{r}}^{(k)}$가 선형 독립인 정적 포인트가 최소 3개 이상 확보돼야 한다.

아래 Python 코드는 이 선형 관계식과 RANSAC을 결합해 자차 속도 $\mathbf{v}_{\text{ego}}$를 추정(`estimate_ego_velocity_from_doppler`)하고, 추정한 자차 속도를 기준으로 정적·동적 포인트를 가려내는(`separate_static_dynamic`) 전체 절차를 보여 준다.

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

Radar odometry와 악천후 센서 융합 연구에서 널리 쓰이는 대표 데이터셋은 센서 세대와 라벨링 범위에 따라 다음과 같이 정리된다.

| 데이터셋 | 센서 | 환경 | 특징 |
|----------|------|------|------|
| **[Boreas](https://arxiv.org/abs/2203.10168)** (Burnett et al. 2023) | Camera, LiDAR, Radar, GNSS/IMU | 도심 (다양한 날씨) | 1년간 동일 경로 반복, 악천후 포함 |
| **RadarScenes** | Radar, Camera (참고용), Odometry | 도심 | 기존 automotive radar 포인트 + semantic labels (point-level annotation). LiDAR는 없음 |
| **nuScenes** | Camera, LiDAR, Radar | 도심 | 5개 radar 포함, 악천후 일부 |
| **[View-of-Delft](https://doi.org/10.1109/LRA.2022.3147324)** (Palffy et al. 2022) | Stereo camera, 64채널 LiDAR, 3+1D radar | 도심 (Delft) | 3+1D radar + 3D bounding box annotation |

**최근 시스템과 데이터셋 (2025~2026)**: 고각 정보를 포함한 4D radar 전용 벤치마크와 새로운 정합 표현도 속속 공개되고 있다.

- **[SNAIL Radar](https://arxiv.org/abs/2407.11705)** (Huai et al., IJRR 2025): 4D radar 기반 SLAM 평가를 위한 벤치마크로, 수집 플랫폼과 44개 시퀀스·날씨 조건의 세부 사항은 §12.4.1에 정리돼 있다.
- **[4D Radar-Inertial Odometry](https://arxiv.org/abs/2412.13639)** (RA-L 2026): 성긴 radar 장면을 고정 voxel 대신 3D Gaussian 분포 묶음으로 표현하고, 다중 정합 가설을 동시에 최적화해 local optimum에 빠질 위험을 줄였다. 공개 4D radar 데이터셋 평가에서 기존 정합 알고리즘과 대등하거나 앞선 성능을 보고했다.

---

## 요약

센서 융합과 SLAM 연구가 빠르게 확장되는 흐름 속에서도 실전 시스템을 설계할 때 붙잡아야 할 중심축은 분명하다.

- **전통 기하·확률 방법은 여전히 기반이다.** 칼만 필터, ICP, RANSAC, factor graph는 수십 년 전에 정립됐지만, 2025년 현재 공개되는 최신 시스템 역시 이 수학적 토대 위에서 구동된다.
- **지각(perception) 영역에서는 학습 기반 방법이 강세를 보인다.** 특징점 매칭과 단안 깊이 추정, place recognition에서는 학습 모델이 여러 공개 벤치마크에서 전통 기법보다 높은 점수를 기록해 왔다. 다만 그 우위의 폭은 데이터셋과 평가 프로토콜에 따라 달라진다.
- **추론(inference) 단계에서는 전통과 학습이 공존한다.** 상태 추정 백엔드는 여전히 필터나 비선형 최적화가 맡고 있으며, DROID-SLAM처럼 미분 가능 최적화를 신경망 내부에 녹여 두 세계를 결합하는 방향이 넓어지고 있다.
- **Foundation model이 파이프라인 곳곳으로 들어오고 있다.** 2023년에 등장한 DINOv2, SAM 같은 범용 모델의 표현을 2024~2025년 연구들이 장소 인식, 매칭, 의미 지도 구축 등 센서 융합 파이프라인 여러 단계에 결합하고 있다(§13.1).

결국 단일 센서의 물리적 한계를 보완해 신뢰할 수 있는 공간 인지를 구현한다는 본질은 변하지 않는다. 새로운 모델과 신형 센서가 더해질수록, 각 기법이 가정하는 관측 모델과 불확실성 전파, 그리고 목표 운용 환경에서의 정량적 검증을 함께 챙기는 엔지니어링 감각이 그 어느 때보다 중요하다.
