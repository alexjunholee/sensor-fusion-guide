# Feature Matching & Correspondence — 핵심 논문 요약

> **기술 계보 개관**
>
> 전통적 파이프라인은 **detect → describe → match**의 3단계를 독립 모듈로 수행했다 (SIFT, ORB 등).
> 딥러닝 시대에 이 패러다임은 두 갈래로 진화한다.
>
> 1. **Detect-then-Describe-then-Match의 딥러닝화**: SIFT → **SuperPoint** (학습 기반 검출+기술) → **SuperGlue** (학습 기반 매칭) → **LightGlue** (효율화)
> 2. **Detector-Free 패러다임**: 검출기를 아예 제거하고 밀집(dense) 매칭 수행 → **LoFTR** → **RoMa**
>
> 병렬로, **RAFT**는 optical flow 분야에서 all-pairs correlation + iterative refinement라는 아이디어를 제시하여 LoFTR·RoMa 등 detector-free 매칭의 설계 철학에 영향을 미쳤다.

---

## 1. SuperPoint: Self-Supervised Interest Point Detection and Description

- **저자/연도**: Daniel DeTone, Tomasz Malisiewicz, Andrew Rabinowitz (2018)
- **발표**: CVPR 2018 Workshop on Deep Learning for Visual SLAM
- **핵심 기여**: 자기지도학습(self-supervised learning)을 통해 키포인트 검출과 디스크립터 추출을 단일 네트워크로 통합한 최초의 실용적 딥러닝 파이프라인.

### 주요 내용

- **Homographic Adaptation**: 핵심 학습 전략. 하나의 이미지에 무작위 호모그래피를 반복 적용(100회 이상)하여 다양한 시점에서 검출된 포인트를 집계(aggregate)한다. 여러 변환에서 일관되게 검출되는 점만 pseudo ground-truth로 채택함으로써, 수작업 라벨 없이도 반복성(repeatability) 높은 키포인트 검출기를 학습한다.
- **MagicPoint → SuperPoint 2단계 학습**: 
  - 1단계: 합성 기하 도형(삼각형, 사각형, 선분 등)으로 구성된 Synthetic Shapes 데이터셋에서 코너/접합점 검출기(MagicPoint)를 사전 학습.
  - 2단계: MagicPoint를 MS-COCO 등 실제 이미지에 Homographic Adaptation과 함께 적용하여, 실제 장면에서의 pseudo ground-truth를 생성하고 이를 바탕으로 SuperPoint를 학습.
- **아키텍처**: VGG 스타일 인코더(공유 백본) → 두 개의 디코더 헤드로 분기.
  - **Interest Point Decoder**: 입력 이미지를 8×8 셀 그리드로 나누고, 각 셀에서 65채널(64 위치 + 1 "no interest point") softmax를 수행하여 픽셀 단위 키포인트 히트맵 생성. Sub-pixel 정밀도 없이 셀 내 위치를 직접 예측하는 방식.
  - **Descriptor Decoder**: 공유 백본의 feature map에서 256차원 디스크립터 맵을 출력하고, 검출된 키포인트 위치에서 bi-cubic interpolation으로 샘플링. L2 정규화 적용.
- **실시간 성능**: 단일 포워드 패스로 검출+기술을 동시 수행하므로 SIFT 대비 훨씬 빠름. 640×480 이미지에서 약 70 FPS (GPU 기준).
- **학습 손실**: 키포인트 검출에는 cross-entropy loss, 디스크립터에는 호모그래피로 대응점을 알고 있으므로 positive/negative pair에 대한 hinge loss 사용.

### 기술 계보에서의 위치

SIFT/ORB 같은 hand-crafted 특징점의 한계(조명 변화, 시점 변화에 대한 취약성)를 딥러닝으로 극복. 특히 **검출과 기술을 하나의 네트워크**로 합친 점이 핵심 전환점이다. 다만 매칭은 여전히 최근접 이웃 탐색(nearest neighbor) + ratio test라는 전통적 방식에 의존한다.

### 후속 영향

- SuperGlue의 직접적 입력 모듈로 활용되며, SuperPoint + SuperGlue 조합은 Visual Localization 벤치마크에서 사실상 표준이 됨.
- "학습 기반 키포인트"라는 연구 방향을 개척하여 D2-Net, R2D2, DISK 등 후속 연구에 영감.
- Homographic Adaptation 기법은 이후 다양한 self-supervised 특징 학습에 차용됨.

---

## 2. SuperGlue: Learning Feature Matching with Graph Neural Networks

- **저자/연도**: Paul-Edouard Sarlin, Daniel DeTone, Tomasz Malisiewicz, Andrew Rabinowitz (2020)
- **발표**: CVPR 2020
- **핵심 기여**: 키포인트 매칭을 **그래프 신경망(GNN)과 어텐션 메커니즘**으로 학습 가능한 문제로 재정의. 기존의 nearest-neighbor 매칭을 학습 기반 매칭으로 대체한 최초의 상용급 시스템.

### 주요 내용

- **문제 정의**: 두 이미지에서 추출된 키포인트 집합 사이의 **부분 할당(partial assignment)** 문제로 정의. 모든 키포인트가 대응점을 갖는 것은 아니므로, "매칭 없음(dustbin)"이라는 가상 노드를 추가하여 unmatchable 포인트를 명시적으로 처리.
- **Attentional Graph Neural Network**: 
  - 키포인트를 그래프의 노드로 표현. 각 노드의 초기 특징은 키포인트의 시각적 디스크립터 + 위치(x, y) + 검출 스코어를 MLP로 인코딩한 것.
  - **Keypoint Encoder**: 키포인트 위치 (x, y)와 검출 confidence를 MLP로 임베딩하여 디스크립터 벡터에 더함. 이로써 기하학적 정보가 특징에 내재됨.
  - **Self-Attention (intra-image)**: 같은 이미지 내 키포인트들 사이의 관계를 학습. 예: 건물의 코너들이 직선 위에 정렬되어 있다는 구조적 정보를 포착.
  - **Cross-Attention (inter-image)**: 두 이미지 간 키포인트 사이의 관계를 학습. "이 키포인트는 상대 이미지의 어떤 키포인트와 유사한가"를 어텐션으로 추론.
  - Self와 Cross attention을 교대로 L번(논문에서는 9회) 반복하여 점진적으로 매칭 정보를 정제.
- **Optimal Transport를 이용한 매칭**: 
  - 최종 매칭 점수 행렬을 GNN 출력의 내적으로 계산한 뒤, **Sinkhorn 알고리즘**(soft한 형태의 Hungarian algorithm)을 적용하여 최적 수송 문제를 풂.
  - Dustbin 행/열을 추가하여, 각 키포인트가 상대 이미지의 어떤 키포인트에도 매칭되지 않을 확률을 모델링.
  - Sinkhorn을 반복 적용(약 100회)하면 soft assignment matrix가 수렴하며, 이를 threshold하여 최종 매칭 결정.
- **학습**: ground-truth 대응점(호모그래피 또는 상대 포즈 + 깊이 맵에서 생성)에 대한 negative log-likelihood 최대화로 end-to-end 학습.

### 기술 계보에서의 위치

SuperPoint가 검출+기술을 학습화했다면, SuperGlue는 **매칭 단계를 학습화**했다. 이로써 detect-then-describe-then-match 파이프라인의 **세 단계 모두가 딥러닝으로 대체**되었다. 그러나 여전히 파이프라인의 직렬적 3단계 구조 자체는 유지된다.

### 후속 영향

- SuperPoint + SuperGlue 조합은 SfM, Visual Localization, SLAM에서 표준 baseline으로 자리잡음.
- Cross-attention을 이용한 매칭이라는 아이디어는 LoFTR에 직접적으로 차용됨.
- 그러나 GNN의 O(N²) 복잡도와 Sinkhorn 반복의 비용이 한계로 지적되어, 이를 해결하려는 LightGlue가 등장.

---

## 3. RAFT: Recurrent All-Pairs Field Transforms for Optical Flow

- **저자/연도**: Zachary Teed, Jia Deng (2020)
- **발표**: ECCV 2020 (Best Paper Award)
- **핵심 기여**: Optical flow 추정에 **4D correlation volume + 반복적 GRU 업데이트**라는 새로운 아키텍처 패러다임을 제시. Coarse-to-fine 피라미드 방식을 대체하고, 모든 픽셀 쌍의 상관관계를 한 번에 계산하는 all-pairs 접근법을 도입.

### 주요 내용

- **아키텍처 3단계**:
  1. **Feature Encoder**: 입력 두 이미지 각각에 CNN(ResNet 변형)을 적용하여 1/8 해상도의 특징 맵 추출. 별도의 Context Encoder가 첫 번째 이미지에서 GRU의 초기 hidden state와 context feature를 추출.
  2. **Correlation Volume 구성**: 두 특징 맵의 모든 픽셀 쌍에 대해 내적(dot product)을 계산하여 4D correlation volume (H×W×H×W) 생성. 이를 후반 두 차원에 대해 average pooling하여 4단계 correlation pyramid 구축 (1, 2, 4, 8 스케일). 핵심 통찰: **coarse-to-fine이 아닌, single resolution에서 multi-scale lookup**을 수행.
  3. **Iterative Update (GRU)**: ConvGRU가 현재 flow 추정치를 반복적으로 정제.
     - 각 반복에서: 현재 flow 추정에 따라 correlation pyramid에서 값을 lookup (현재 대응 위치 주변의 local window 참조) → correlation feature, 현재 flow, context feature를 결합 → ConvGRU가 hidden state 업데이트 → hidden state에서 flow 잔차(residual) 예측 → flow 업데이트.
     - 학습 시 12회, 추론 시 12~32회 반복. 반복 횟수를 늘리면 정확도가 향상되는 특성(test-time adaptability).
- **All-Pairs vs. Coarse-to-Fine**: 기존 PWC-Net, FlowNet 등은 피라미드에서 coarse flow를 먼저 추정하고 점진적으로 정밀화하는데, 이 과정에서 큰 변위가 coarse 레벨에서 놓치면 복구 불가. RAFT는 **전체 해상도에서 모든 상관관계를 한 번에 보유**하므로 큰 변위도 놓치지 않음.
- **학습**: 학습 시 모든 반복 단계의 예측에 대해 ground-truth flow와의 L1 loss를 적용하되, 후반 반복에 더 큰 가중치를 부여 (exponentially increasing weights). FlyingChairs → FlyingThings3D → Sintel/KITTI fine-tuning 순서.
- **성능**: Sintel (clean) EPE 1.43, KITTI-2015 F1-all 5.10%로 당시 SOTA를 큰 폭으로 갱신.

### 기술 계보에서의 위치

RAFT 자체는 optical flow 논문이지만, feature matching 분야에 두 가지 핵심 아이디어를 전파했다:
1. **All-pairs correlation**: 모든 위치 쌍의 유사도를 명시적으로 계산하는 접근법 → LoFTR의 coarse-level 전체 어텐션에 영향.
2. **Iterative refinement**: 한 번에 결과를 내는 것이 아니라 반복적으로 정제하는 패러다임 → RoMa 등 후속 dense matching 기법에서 차용.

### 후속 영향

- Optical flow 분야를 지배 (GMFlow, FlowFormer 등 후속 연구의 기반).
- RAFT의 all-pairs 사고방식은 dense matching / scene flow / stereo 등 관련 분야 전반에 확산.
- DROID-SLAM 등 SLAM 시스템에도 RAFT의 correlation + iterative update 구조가 직접 적용됨.

---

## 4. LoFTR: Detector-Free Local Feature Matching with Transformers

- **저자/연도**: Jiaming Sun, Zehong Shen, Yuang Wang, Hujun Bao, Xiaowei Zhou (2021)
- **발표**: CVPR 2021
- **핵심 기여**: **검출기 없이(detector-free)** 두 이미지 간 밀집 매칭을 수행하는 트랜스포머 기반 아키텍처. 텍스처가 부족한 영역이나 반복 패턴에서도 매칭 가능한 패러다임 전환을 이룸.

### 주요 내용

- **Detector-Free 패러다임의 동기**: 
  - 전통적 detect → describe → match 파이프라인은 검출기가 키포인트를 찾지 못하면 매칭 자체가 불가능. 텍스처 없는 벽, 바닥, 하늘 등에서 치명적.
  - LoFTR는 검출 단계를 없애고, **모든 위치를 잠재적 매칭 후보로 취급**하여 이 한계를 극복.
- **아키텍처**:
  1. **Local Feature CNN**: ResNet-18 기반 FPN으로 두 이미지에서 coarse (1/8 해상도) 및 fine (1/2 해상도) 특징 맵 추출.
  2. **Coarse-Level Matching (Transformer)**:
     - 1/8 특징 맵을 1D 시퀀스로 평탄화(flatten).
     - **Self-Attention + Cross-Attention**을 교대로 N번(보통 4회) 반복하는 트랜스포머 모듈(Linear Transformer 사용)을 적용.
     - 위치 인코딩: sinusoidal positional encoding으로 공간 정보 보존.
     - 출력 특징들 간의 내적으로 score matrix 계산 → dual-softmax(행 방향, 열 방향 softmax를 곱함)로 confidence matrix 생성 → threshold + mutual nearest neighbor 조건으로 coarse 매칭 추출.
  3. **Fine-Level Refinement**:
     - Coarse 매칭 각각에 대해, 1/2 해상도 특징 맵에서 해당 위치 주변 w×w 윈도우(5×5)를 crop.
     - 윈도우 내에서 다시 cross-attention 기반 correlation을 수행하여 sub-pixel 정밀도의 매칭 위치를 회귀(regression).
- **Linear Transformer**: 계산량 절감을 위해 standard softmax attention 대신 kernel-based linear attention 사용. O(N²) → O(N) 복잡도. 단, 후속 연구에서 standard attention이 정확도에서 우위라는 점이 밝혀짐.
- **학습**: ground-truth 포즈 + 깊이 맵에서 생성한 대응점을 supervision으로 사용. Coarse level에서는 cross-entropy loss, fine level에서는 L2 regression loss 적용. ScanNet(실내) 및 MegaDepth(실외) 데이터셋에서 학습.

### 기술 계보에서의 위치

**패러다임 전환점**. SuperPoint→SuperGlue로 이어진 detect-then-match 파이프라인과 결별하고, 검출기를 완전히 제거. 핵심 통찰은 **트랜스포머의 어텐션이 검출과 매칭을 동시에 수행할 수 있다**는 것이다. 이미지의 모든 위치가 상대 이미지의 모든 위치와 직접 소통(cross-attention)하므로, 별도의 검출 단계 없이도 매칭이 가능해졌다. SuperGlue의 cross-attention 아이디어를 검출기 없는 밀집 설정으로 확장한 것으로 볼 수 있다.

### 후속 영향

- Detector-free matching이라는 새로운 연구 방향을 개척. ASpanFormer, QuadTree Attention, TopicFM 등 수많은 후속 연구.
- Coarse-to-fine (coarse transformer + fine refinement) 전략이 이후 dense matching의 표준 설계가 됨.
- 그러나 linear attention의 정확도 한계, coarse 단계의 매칭 오류가 fine에서 복구 불가능한 점 등이 지적되어 후속 연구의 개선 대상이 됨.

---

## 5. LightGlue: Local Feature Matching at Light Speed

- **저자/연도**: Philipp Lindenberger, Paul-Edouard Sarlin, Marc Pollefeys (2023)
- **발표**: ICCV 2023
- **핵심 기여**: SuperGlue의 정확도를 유지하면서 **적응적 연산량 조절(adaptive computation)**로 속도를 대폭 개선. 쉬운 이미지 쌍에는 적은 연산, 어려운 쌍에는 많은 연산을 자동 할당.

### 주요 내용

- **SuperGlue의 문제점 진단**:
  - 항상 고정된 9개 GNN 레이어와 100회 Sinkhorn 반복을 수행 → 쉬운 매칭에도 불필요하게 많은 연산.
  - 키포인트 수 N에 대해 O(N²) attention이 반복되므로, 키포인트가 많을수록 급격히 느려짐.
- **아키텍처 개선**:
  1. **Self-Attention + Cross-Attention 구조 유지**: SuperGlue와 동일하게 교대 적용하되, 더 효율적으로 설계.
  2. **Adaptive Depth (레이어 조기 종료)**:
     - 각 레이어 이후에 경량 classifier(MLP)가 "이 키포인트 쌍의 매칭 확신도"를 예측.
     - 확신도가 충분히 높으면 해당 키포인트를 이후 레이어에서 제외(pruning).
     - 모든 키포인트가 확신에 도달하면 전체 네트워크를 조기 종료.
     - 결과: 쉬운 이미지 쌍은 2-3 레이어만으로 처리, 어려운 쌍은 전체 9 레이어 사용.
  3. **Adaptive Width (키포인트 pruning)**:
     - 매칭 불가능하다고 판단된 키포인트를 중간 레이어에서 제거.
     - Attention의 시퀀스 길이가 점진적으로 줄어들어 후반 레이어의 계산이 가벼워짐.
  4. **Sinkhorn 제거**: Optimal Transport 대신 단순한 dual-softmax + mutual nearest neighbor로 매칭. Sinkhorn의 반복 비용을 완전히 제거하면서도 성능 저하 거의 없음.
  5. **Mixed-Precision / FlashAttention 호환**: 현대 GPU의 효율적 어텐션 구현과 호환되도록 아키텍처를 설계하여 추가 속도 향상.
- **학습 전략**: 
  - 학습 시에는 적응적 종료를 사용하지 않고 전체 레이어를 통과시키되, 각 레이어의 출력에 supervision을 적용 (deep supervision).
  - 추론 시에만 적응적 종료를 활성화.
  - 다양한 로컬 특징 검출기(SuperPoint, DISK, ALIKED 등)와 호환되도록 범용 설계.
- **성능**: SuperGlue 대비 동등한 정확도에서 **3-5배 빠름**. 적응적 메커니즘 덕분에 쉬운 쌍에서는 최대 10배 이상 속도 향상.

### 기술 계보에서의 위치

Detect-then-match 패러다임 내에서의 **효율성 최적화**. SuperGlue의 구조적 핵심(교대 어텐션)은 유지하면서, 불필요한 연산을 제거하는 실용적 접근. 학술적으로 새로운 패러다임은 아니지만, 실용성 측면에서 SuperGlue를 사실상 대체.

### 후속 영향

- SuperPoint + LightGlue가 새로운 실용 표준으로 부상.
- 적응적 연산이라는 아이디어는 LoFTR 계열에도 적용 가능성이 탐색됨 (Efficient LoFTR).
- SLAM, SfM 등 실시간 응용에서 학습 기반 매칭의 실용적 채택을 가속화.

---

## 6. RoMa: Robust Dense Feature Matching

- **저자/연도**: Johan Edstedt, Qiyu Sun, Georg Bökman, Mårten Wadenbäck, Michael Felsberg (2024)
- **발표**: CVPR 2024
- **핵심 기여**: 사전학습된 DINOv2 특징을 활용하고, coarse-to-fine 밀집 매칭에 **robust regression(확률적 warp 분포 예측)**을 결합하여, detector-free 매칭의 정확도와 강건성을 동시에 끌어올림.

### 주요 내용

- **동기**: LoFTR 이후의 detector-free 매칭 연구들은 대부분 처음부터(scratch) 학습하는데, 이는 대규모 사전학습 모델의 풍부한 시각적 지식을 활용하지 못함. 또한 기존 방법들은 매칭의 불확실성을 명시적으로 모델링하지 않음.
- **아키텍처**:
  1. **Frozen DINOv2 Backbone**: 사전학습된 DINOv2 ViT-Large를 특징 추출기로 사용 (가중치 동결). DINOv2는 대규모 자기지도학습으로 학습된 범용 시각 특징으로, 이미 풍부한 의미적(semantic) 정보를 내포.
  2. **Coarse Matching (Warp Estimation)**:
     - DINOv2 특징에 cross-attention(transformer decoder) 적용.
     - 이미지 A의 각 위치에 대해, 이미지 B에서의 대응 위치 분포를 예측. 이때 단일 점이 아닌 **확률 분포(Gaussian mixture 또는 discretized distribution)**로 예측하여 모호한 매칭의 불확실성을 표현.
     - 1/14 해상도(DINOv2 patch size)에서 시작.
  3. **Fine Matching (Iterative Refinement)**:
     - Coarse warp를 초기값으로, CNN 기반의 fine-level 특징(별도 학습)을 사용하여 반복적으로 정밀화.
     - 각 정밀화 단계에서 해상도를 높이며, 이전 단계의 warp를 기반으로 local correlation을 계산하고 잔차를 예측.
     - RAFT와 유사한 iterative refinement 철학이지만, optical flow가 아닌 sparse-to-dense warp에 적용.
  4. **Certainty Estimation**: 각 매칭에 대해 신뢰도(certainty)를 함께 예측. 이를 통해 후처리 시 고신뢰 매칭만 선택적으로 사용 가능.
- **Robust Regression**: 
  - 매칭 위치를 단순히 L2 loss로 회귀하는 대신, 예측된 확률 분포와 ground-truth 간의 negative log-likelihood를 최적화.
  - Outlier에 강건: 잘못된 ground-truth 대응점이 있더라도 분포의 꼬리(tail)로 흡수되어 학습이 안정적.
- **학습 데이터**: MegaDepth에서 학습. ScanNet은 fine-tuning에 사용.
- **성능**: MegaDepth 및 ScanNet 벤치마크에서 LoFTR, ASpanFormer 등을 큰 폭으로 상회. 특히 넓은 baseline(큰 시점 변화)에서의 성능 향상이 두드러짐.

### 기술 계보에서의 위치

Detector-free 패러다임의 **성숙 단계**를 대표한다. 두 가지 핵심 진화:
1. **Foundation Model 활용**: LoFTR가 처음부터 학습한 것과 달리, DINOv2라는 대규모 사전학습 모델의 의미적 특징을 기반으로 함. 이는 "특징 학습은 범용 모델에 맡기고, 매칭 로직만 학습하면 된다"는 철학적 전환.
2. **확률적 매칭**: 결정론적 점 예측에서 확률 분포 예측으로의 전환. 불확실한 매칭을 명시적으로 다룸으로써 강건성 확보.

RAFT의 iterative refinement 아이디어와 LoFTR의 detector-free 사고방식을 결합하면서, DINOv2라는 강력한 사전학습 특징을 도입한 종합적 발전.

### 후속 영향

- Foundation model + task-specific head라는 설계 패턴이 feature matching 분야에 확산 (MASt3R 등).
- Dense matching의 실용적 정확도가 SfM/3D reconstruction에서 전통적 파이프라인을 위협하는 수준에 도달.
- 확률적 매칭 + 신뢰도 추정의 조합은 하류 태스크(포즈 추정, 3D 복원)에서 RANSAC과의 시너지를 강화.

---

## 기술 계보 요약

```
[Hand-Crafted]          [Learned Detect+Describe]       [Learned Matching]         [Efficiency]
SIFT (2004)       ──→   SuperPoint (2018)          ──→   SuperGlue (2020)     ──→   LightGlue (2023)
  검출+기술+매칭            학습 기반 검출+기술               GNN 기반 매칭               적응적 연산
  모두 수작업              매칭은 여전히 NN                  Optimal Transport           Sinkhorn 제거
                                                                                     
                          ┌──────────────────────────── 패러다임 전환 ────────────────────────────┐
                          │                                                                      │
                          │  [All-Pairs Flow]          [Detector-Free]           [Foundation +    │
                          │  RAFT (2020)          ──→   LoFTR (2021)        ──→   Dense Matching] │
                          │    4D correlation             Transformer 매칭          RoMa (2024)   │
                          │    반복적 정제                 coarse-to-fine             DINOv2 활용  │
                          │                               검출기 제거                확률적 매칭  │
                          └──────────────────────────────────────────────────────────────────────┘
```

**핵심 서사**: 전통적 3단계 파이프라인(detect → describe → match)을 딥러닝으로 각각 대체하는 흐름(SuperPoint → SuperGlue → LightGlue)과, 파이프라인 자체를 해체하고 end-to-end 밀집 매칭으로 전환하는 흐름(RAFT → LoFTR → RoMa)이 병렬로 진화하고 있다. 후자가 점점 우세해지는 추세이나, 전자의 효율성과 해석 가능성은 여전히 실용적 가치가 있다.
