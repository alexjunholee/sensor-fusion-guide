# Ch.13 — Frontiers & Emerging Directions

Sensor fusion includes established theory and deployed systems as well as research directions that are still being validated.

These directions include extending foundation models to spatial AI, end-to-end learned SLAM, scene-graph-based environmental understanding, cross-modal representation, and the fusion of event cameras and 4D radar.

---

## 13.1 Foundation Models for Spatial AI

Research increasingly uses representations from foundation models — general-purpose models pretrained on large-scale data, such as DINOv2, CLIP, and SAM — inside sensor-fusion and SLAM pipelines. Their features can support retrieval, matching, and semantic priors, but they do not automatically replace geometry or uncertainty handling.

### 13.1.1 Leveraging DINOv2/CLIP Features in SLAM

**Visual features of DINOv2**: [DINOv2](https://arxiv.org/abs/2304.07193) is a ViT trained with self-supervised learning. Tokens obtained for image patches can be used as dense features, and downstream experiments reveal the following properties.

- In some illumination and seasonal changes, the features help recover same-place correspondences.
- Object categories and scene semantics can affect feature similarity.
- Patch-level structure can support downstream correspondence and segmentation tasks.

**AnyLoc's approach**: [AnyLoc](https://arxiv.org/abs/2308.00688) (Keetha et al. 2023) aggregates DINOv2's dense features with VLAD to produce a global place descriptor. This descriptor:

- was evaluated without VPR-specific fine-tuning on paper benchmarks spanning urban, indoor, aerial, underwater, and subterranean settings.
- reports higher recall than the tested NetVLAD- and CosPlace-family baselines on several datasets, not a guarantee for every environment.
- includes an ablation in which value-facet dense features from layer 31 average 23% higher results than the CLS token.

```python
import numpy as np

class FoundationModelFeatureExtractor:
    """
    Conceptual implementation of DINOv2-based dense feature extraction.
    In practice, torch + the DINOv2 model is used.
    """
    
    def __init__(self, model_name='dinov2_vitg14', layer=31, facet='value'):
        """
        Args:
            model_name: DINOv2 model name
            layer: layer to extract from (31 in AnyLoc)
            facet: one of 'key', 'query', 'value' ('value' in AnyLoc)
        """
        self.model_name = model_name
        self.layer = layer
        self.facet = facet
        # In practice, load the torch model here
        # self.model = torch.hub.load('facebookresearch/dinov2', model_name)
        
    def extract_dense_features(self, image):
        """
        Extract pixel-level dense features from an image.
        
        Args:
            image: (H, W, 3) RGB image
            
        Returns:
            features: (H', W', D) dense feature map
                      H' = H/14, W' = W/14 (patch size = 14)
                      D = 1536 (feature dimension of ViT-G14)
        """
        # 1. Split the image into 14x14 patches
        # 2. Pass through the ViT to extract features from the specified layer
        # 3. Return in (num_patches, D) shape
        
        # placeholder
        H, W = image.shape[:2]
        h, w = H // 14, W // 14
        D = 1536
        features = np.random.randn(h, w, D).astype(np.float32)
        return features
    
    def extract_global_descriptor(self, image, num_clusters=64):
        """
        AnyLoc style: dense feature -> VLAD -> global descriptor.
        
        Args:
            image: (H, W, 3) RGB image
            num_clusters: number of VLAD clusters
            
        Returns:
            descriptor: (num_clusters * D,) global descriptor
        """
        dense_features = self.extract_dense_features(image)
        
        # Flatten spatial dimensions
        h, w, D = dense_features.shape
        features_flat = dense_features.reshape(-1, D)  # (N, D)
        
        # VLAD aggregation (simplified)
        # In practice, cluster centers are precomputed with k-means
        cluster_centers = np.random.randn(num_clusters, D)  # placeholder
        
        vlad = np.zeros((num_clusters, D))
        for feat in features_flat:
            # Hard assignment
            dists = np.linalg.norm(cluster_centers - feat, axis=1)
            closest = np.argmin(dists)
            vlad[closest] += feat - cluster_centers[closest]
        
        # L2 normalization (intra + inter)
        for i in range(num_clusters):
            norm = np.linalg.norm(vlad[i])
            if norm > 1e-8:
                vlad[i] /= norm
        
        descriptor = vlad.flatten()
        descriptor /= (np.linalg.norm(descriptor) + 1e-8)
        
        return descriptor
```

**Where foundation models fit in the SLAM pipeline**:

| Pipeline module | Traditional method | Example use of learned features |
|---------------|-----------|-------------|
| Feature detection | FAST, ORB | SuperPoint + DINOv2 hybrid |
| Feature matching | BF + ratio test | SuperGlue/LightGlue, LoFTR |
| Place recognition | DBoW2, Scan Context | AnyLoc (DINOv2 + VLAD) |
| Semantic segmentation | Task-specific model training | [SAM](https://arxiv.org/abs/2304.02643), open-vocabulary segmentation |
| Depth estimation | Stereo matching | Relative-depth prior from [Depth Anything](https://arxiv.org/abs/2401.10891) |
| Loop-closure candidate | DBoW2, Scan Context | FM descriptor similarity followed by geometric verification |

### 13.1.2 Open-Vocabulary 3D Understanding

Extending CLIP's vision-language alignment to 3D maps allows a robot to understand and navigate its environment via natural language.

The system operates as follows:

1. Build a 3D map (point cloud, mesh, 3DGS) with SLAM.
2. Extract CLIP visual features for each region of each observation image.
3. Back-project the 2D features to their corresponding locations in the 3D map and attach them.
4. When the user says "find the fire extinguisher," the CLIP text encoder encodes the text and returns the location in the 3D map with the highest similarity.

**ConceptFusion, LERF, OpenScene** are representative systems of this approach. They allow 3D space to be queried with arbitrary text without a predefined class set.

Limits remain:
- CLIP features have low spatial resolution (patch-level). Precise localization of small objects is difficult.
- 3D consistency is hard to guarantee — the same object can have different features from different viewpoints.
- Computational cost: extracting FM features from every image is expensive.

### 13.1.3 Evidence and Boundaries by Module

The evidence for foundation-model integration varies by task and benchmark. The following is not a replacement ranking; it identifies what must be validated for each use.

**Areas with evidence on published benchmarks**:
- **Visual place recognition**: AnyLoc reports strong recall for DINOv2+VLAD across several domains. Because its inputs, training assumptions, and compute differ from DBoW2, treat them as alternatives to test in the target environment and compare both false-positive rate and latency.
- **Feature matching**: LoFTR and RoMa report strong results on benchmarks with difficult texture and viewpoint changes. Classical sparse features remain viable when compute, calibration, or repeatability requirements favor them.
- **Monocular depth**: The base output of [Depth Anything](https://arxiv.org/abs/2401.10891) is relative depth. Fusion that requires metric scale needs a metric-depth fine-tune or constraints from another sensor, plus a separate treatment of output uncertainty.

**Areas that still require a separate estimator**:
- **LiDAR odometry**: Comparisons between geometry-based methods such as ICP, LOAM, and FAST-LIO2 and learned methods depend on the sensor and dataset. Learned features or correspondences do not remove motion modeling and registration validation.
- **IMU integration**: Preintegration compresses kinematics and a noise model into a factor. Learned bias or noise corrections can be added, but they do not automatically replace the observation model and uncertainty propagation.
- **Backend optimization**: Factor graphs and iSAM2 combine observations. An FM output used as a factor still needs explicit covariance, outlier handling, and correlation analysis.

A practical default is therefore a hybrid that retains the geometry and uncertainty structure of an estimator while adding learned features or semantic priors as testable modules.

**Recent systems (2024-2025)**:

- **[MASt3R-SLAM](https://arxiv.org/abs/2412.12392)** (Murai et al. CVPR 2025): This system integrates MASt3R's learned geometry into SLAM; the paper reports dense SLAM at 15 fps without a prespecified camera model.
- **[Depth Anything V2](https://arxiv.org/abs/2406.09414)** (Yang et al. NeurIPS 2024): This work trains a teacher on synthetic data and a student on large-scale pseudo-labels. Its benchmark results make it a candidate depth prior, but fusion still requires validation of scale and uncertainty.

---

## 13.2 End-to-End Learned SLAM

Traditional SLAM is built as a modular pipeline (feature extraction -> matching -> motion estimation -> mapping -> loop closure -> optimization). End-to-end learning turns this pipeline into a single differentiable system that maps directly from input (images/sensors) to output (pose, map).

### 13.2.1 Current Representative Systems

**[DROID-SLAM](https://arxiv.org/abs/2108.10869)** (Teed & Deng 2021) is a representative public system for learned visual SLAM.

DROID-SLAM has the following structure:

1. **RAFT-based iterative update operator**: A convolutional GRU iteratively refines the optical flow using features extracted from a correlation volume. This flow correction serves as a refinement of correspondences.

2. **Differentiable Dense Bundle Adjustment (DBA)**: The flow corrections are converted into camera pose (SE(3)) and per-pixel inverse-depth updates. Gauss-Newton is solved efficiently via the Schur complement, and the entire operation is differentiable, so it can be trained via backpropagation.

$$\begin{bmatrix} \mathbf{H}_{pp} & \mathbf{H}_{pd} \\ \mathbf{H}_{dp} & \mathbf{H}_{dd} \end{bmatrix} \begin{bmatrix} \Delta \boldsymbol{\xi} \\ \Delta \mathbf{d} \end{bmatrix} = \begin{bmatrix} \mathbf{b}_p \\ \mathbf{b}_d \end{bmatrix}$$

The Schur complement lets us solve for pose first:

$$(\mathbf{H}_{pp} - \mathbf{H}_{pd} \mathbf{H}_{dd}^{-1} \mathbf{H}_{dp}) \Delta \boldsymbol{\xi} = \mathbf{b}_p - \mathbf{H}_{pd} \mathbf{H}_{dd}^{-1} \mathbf{b}_d$$

Since $\mathbf{H}_{dd}$ is diagonal, each diagonal inverse costs $O(1)$ and processing the full depth block is linear in the number of depth variables. This structure makes the Schur complement efficient to compute.

3. **Frame-graph-based loop closure**: A frame graph is built dynamically based on co-visibility. On revisit, long-range edges are added to perform implicit loop closure.

4. **A single model supports monocular/stereo/RGB-D**: One model trained on synthetic TartanAir data is applied to all three input types. The paper reports higher accuracy and fewer catastrophic failures than the tested baselines on selected metrics across four benchmarks.

**Results reported for DROID-SLAM**:
- 62% error reduction over the previous best on TartanAir
- 82% reduction on EuRoC monocular
- These percentages are relative comparisons under the paper's specific splits, metrics, and baselines

### 13.2.2 Differentiable SLAM Components

Individual SLAM pipeline components can be made differentiable even without full end-to-end learning.

**Differentiable rendering**: NeRF and 3DGS are themselves differentiable rendering systems. In SLAM, pose estimation can be performed by backpropagating a photometric loss.

$$\hat{\mathbf{T}}^* = \arg\min_{\hat{\mathbf{T}}} \| I_{\text{real}} - \text{Render}(\text{Map}, \hat{\mathbf{T}}) \|^2$$

Because the $\text{Render}$ function is differentiable, the gradient with respect to $\hat{\mathbf{T}}$ can be computed directly.

**Differentiable ICP**: By making the nearest-neighbor search and SVD of traditional ICP differentiable, point cloud registration can be embedded inside a learning loop.

**Differentiable pose graph optimization**: If optimizers such as iSAM2 are made differentiable, the frontend (feature extraction, matching) can be trained with error signals from the backend. Optimization error becomes a training signal for the feature extractor.

### 13.2.3 Current Limitations and Possibilities

**Limitations**:
- **Generalization**: Performance can change with cameras, motion, lighting, and dynamic scenes outside the training distribution. The DROID-SLAM paper evaluates synthetic-to-real generalization, but not every environment or sensor modality.
- **Scope of verification**: Traditional pipelines make it easier to inspect residuals and modules separately, whereas a learned update operator is tied more strongly to its training distribution and evaluation protocol. Neither approach automatically guarantees global convergence or consistency for a general nonlinear SLAM problem.
- **Computational cost**: Most learning-based systems require a GPU. Real-time operation on embedded platforms is challenging.
- **Interpretability**: Failure analysis is difficult. Traditional systems allow tracing "which module failed," but end-to-end systems are closer to black boxes.

**Possibilities**:
- As FMs advance, the quality of feature extraction and matching continues to improve.
- As differentiable optimization techniques develop, learning-based modules can be integrated into traditional structures.
- Multi-task learning: jointly training pose estimation, depth estimation, and semantic segmentation produces mutual benefits.

---

## 13.3 Spatial Memory & Scene Graphs

Spatial memory includes hierarchical, relational, and temporal information as well as point clouds. Humans remember space in statements such as "there is a refrigerator in the kitchen, and there was milk inside it." High-level spatial memory systems apply this structure to a robot's map.

### 13.3.1 Persistent Spatial Memory

A traditional SLAM map reflects "the state of the environment at this moment." Persistent spatial memory is long-term spatial memory that also includes the history of how the environment changes over time.

Persistent spatial memory poses three challenges:

1. **Episodic spatial memory**: linking time, place, and event, as in "there was a box here last Tuesday."
2. **Semantic persistence**: distinguishing permanent elements (walls, buildings) from transient ones (people, vehicles) to preserve the stability of long-term maps.
3. **Incremental forgetting**: gradually forgetting the details of old observations while retaining core structure. This mirrors human memory.

```python
class PersistentSpatialMemory:
    """
    Temporal spatial memory -- a time series record of environment state.
    """
    
    def __init__(self, decay_rate=0.01):
        self.memories = {}  # {location_key: [MemoryEntry, ...]}
        self.decay_rate = decay_rate
    
    def record(self, location, observation, timestamp, semantic_class=None):
        """Record a new observation into spatial memory."""
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
        Recall relevant information from spatial memory.
        
        Args:
            location: query location
            time_query: query at a specific time (e.g., "3 days ago")
            semantic_query: semantic query (e.g., "chair")
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
        """Compare the current observation with memory to detect changes."""
        key = self._spatial_key(location)
        if key not in self.memories:
            return 'new_location'
        
        latest = self.memories[key][-1]
        
        # Compare observations (placeholder -- in practice, compare features)
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
        Memory consolidation: remove old, rarely accessed memories.
        Preserve essential structural information.
        """
        for key in self.memories:
            entries = self.memories[key]
            
            if len(entries) <= max_entries_per_location:
                continue
            
            # Priority: recency + access frequency + confidence
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

### 13.3.2 Scene-Graph-Based Environmental Understanding

The 3D Scene Graph of [Hydra](https://arxiv.org/abs/2201.13360) supports the following extensions.

**Scene Graph + Language**: Combining a scene graph with a natural language interface allows a robot to understand commands such as "bring me the remote on the table next to the sofa in the living room." This command is translated into a hierarchical traversal of the scene graph:

1. "living room" -> search the Room node
2. "table next to the sofa" -> search the relations among Object nodes within the Room
3. "remote" -> search Object nodes near that Table
4. Path planning and manipulation

**Scene Graph + LLM**: An LLM such as GPT-4 takes a scene graph as input and performs high-level reasoning. It can answer queries such as "if a person falls in this room, where is the nearest phone?"

**Dynamic scene graphs**: Hydra's current implementation assumes a static environment. Dynamic scene graphs include moving agents (people, vehicles) as nodes and update their relations in real time. Social navigation and human-robot interaction (HRI) can use this information.

### 13.3.3 Time-Series Spatial Memory Management

Robots operating over long time spans require strategies for the **creation, maintenance, and deletion** of spatial memory.

**Hierarchical forgetting**: The resolution of detail (exact texture, individual points) is reduced over time, while structural information (room layout, passage connectivity) is retained permanently. This parallels human spatial memory.

**Event-triggered update**: Instead of refreshing the entire map periodically, only regions where change is detected are updated selectively.

**Compression**: The map is progressively compressed over time. For example, dense point cloud -> sparse landmarks -> topological graph -> semantic description.

---

## 13.4 Cross-Modal Representation

One challenge in sensor fusion is comparing heterogeneous sensor observations in a **common representation space**. A LiDAR point cloud and a camera image have different data forms, yet they observe the same physical environment. Cross-modal representations reduce this "representation gap."

### 13.4.1 Aligning Representations Across Heterogeneous Sensors

Three differences make representation alignment difficult:

- **Dimensional mismatch**: LiDAR returns 3D points, a camera returns a 2D image, and radar returns a range-Doppler map. The data forms differ.
- **Information asymmetry**: LiDAR provides accurate range information but no texture. A camera provides rich texture but no absolute distance.
- **Sensor-specific artifacts**: LiDAR motion distortion, camera rolling shutter, radar speckle noise — each sensor has its own noise pattern.

### 13.4.2 Contrastive Learning for Cross-Modal Alignment

Contrastive learning learns a representation in which observations from different modalities of the same place/object are placed close together, while those from different places/objects are pushed apart.

$$\mathcal{L}_{\text{contrastive}} = -\log \frac{\exp(\text{sim}(f_L(\mathbf{x}_L), f_C(\mathbf{x}_C)) / \tau)}{\sum_{j} \exp(\text{sim}(f_L(\mathbf{x}_L), f_C(\mathbf{x}_C^j)) / \tau)}$$

Here $f_L$ is the LiDAR encoder, $f_C$ is the camera encoder, $\tau$ is the temperature, $(\mathbf{x}_L, \mathbf{x}_C)$ is a LiDAR-camera pair from the same place, and $\mathbf{x}_C^j$ is a negative sample.

**Application to cross-modal place recognition**: A camera-only system can localize against a map built with LiDAR. If the LiDAR descriptor and the camera descriptor occupy the same space, a camera query can retrieve locations from a LiDAR map.

**LC$^2$** (Lee et al. 2023): LiDAR-Camera cross-modal place recognition. It aligns features from a LiDAR BEV image and a camera image into a common space.

### 13.4.3 Knowledge Distillation

Knowledge distillation transfers information from one modality (teacher) to another (student).

**LiDAR -> Camera distillation**: The knowledge of a model trained with LiDAR's accurate 3D information is transferred to a camera-only model. At deployment, the camera alone can then approximate LiDAR-level 3D understanding.

**Camera -> LiDAR distillation**: Rich semantic information from cameras is transferred to LiDAR processing models. For example, attaching CLIP's semantic features to LiDAR points allows a LiDAR map to be queried by text.

### 13.4.4 Still-Open Questions

1. **Is a modality-agnostic representation possible?** Can a universal encoder map inputs from any sensor — LiDAR, camera, radar, event camera — into the same representation space?

2. **Temporal alignment**: Observations from different modalities are not perfectly synchronized in time. How should asynchronous observations be fused into a common representation?

3. **Partial observation**: When one sensor fails temporarily (LiDAR affected by rain, camera affected by darkness), how can a consistent representation be maintained from only the available modalities?

---

## 13.5 Event-Camera-Based Fusion

The event camera (Dynamic Vision Sensor, DVS) operates differently from traditional frame-based cameras. Each pixel independently senses brightness changes and asynchronously emits an **event** only when a change occurs.

### 13.5.1 Principles and Advantages of the Event Camera

Each event is represented as $(x, y, t, p)$:

- $(x, y)$: pixel coordinates
- $t$: microsecond-resolution timestamp
- $p \in \{+1, -1\}$: polarity (brighter / darker)

An event is triggered when:

$$|\log I(x, y, t) - \log I(x, y, t_{\text{last}})| \geq C$$

The polarity is $p = \text{sign}(\log I(x, y, t) - \log I(x, y, t_{\text{last}}))$. Here $I$ is brightness, $C$ is the contrast threshold, and $t_{\text{last}}$ is the most recent time at which the pixel fired an event.

**Advantages**:

| Property | Frame camera | Event camera |
|------|------------|-------------|
| Temporal resolution | 30-120 fps | microseconds |
| Dynamic range | ~60 dB | >120 dB |
| Motion blur | present | nearly none |
| Data output | uniform frames | asynchronous events |
| Static scene | provides information | no events (no information) |
| Power consumption | high | very low |

Event cameras provide information during fast rotations, abrupt illumination changes (entering/exiting tunnels), and low-light operation. They therefore complement traditional cameras and other sensors.

### 13.5.2 Event + Frame Fusion

Event cameras and traditional frame cameras are combined in the following ways:

**Event-enhanced frame tracking**: Fast motion between frames is tracked with events, filling the gap between frame-based VO frames. This maintains tracking even during fast camera motion.

**Event-aided HDR**: Using the event camera's high dynamic range, information in the under/over-exposed regions of frame images is recovered.

### 13.5.3 Event + IMU Fusion

**[EVO](https://doi.org/10.1109/LRA.2016.2645143)** (Rebecq et al. 2017): Event-based Visual Odometry. It estimates camera pose from events alone. Combining with an IMU recovers scale and improves accuracy.

**[Ultimate SLAM](https://arxiv.org/abs/1709.06310)** (Vidal et al. 2018): A system that combines an event camera, a frame camera, and an IMU. It exploits the complementarity of the three sensors:
- Frame camera: rich texture in static scenes
- Event camera: continuous tracking under fast motion
- IMU: scale recovery and fast motion prediction

**Current challenges**:
- The data format of event cameras (an asynchronous event stream) is not compatible with traditional computer vision pipelines (frame-based). Converting events to frames (event frame) sacrifices the advantages.
- Event-camera price and resolution vary by model, and the product range remains narrower than for conventional frame cameras.
- Training data is scarce. Most datasets are designed for frame cameras.

**Recent systems and research (2024-2025)**:

- **[EvenNICER-SLAM](https://arxiv.org/abs/2410.03812)** (2024): A system that integrates an event camera into neural implicit SLAM, using the high temporal resolution of events to improve tracking robustness under fast motion.
- **Event-based 3D reconstruction survey** ([arxiv:2505.08438](https://arxiv.org/abs/2505.08438), 2025): A survey of event-driven 3D reconstruction that organizes work on NeRF/3DGS-based event reconstruction, depth estimation, optical flow, and related topics.

```python
class EventProcessor:
    """
    Utilities for processing event camera data.
    """
    
    def __init__(self, width, height, time_window_us=33000):
        """
        Args:
            width, height: sensor resolution
            time_window_us: event accumulation window (microseconds)
        """
        self.width = width
        self.height = height
        self.time_window = time_window_us
    
    def events_to_frame(self, events, method='histogram'):
        """
        Convert an event stream into a frame.
        
        Args:
            events: [(x, y, t, p), ...] list of events
            method: 'histogram' or 'time_surface'
            
        Returns:
            frame: (H, W) or (H, W, 2) event frame
        """
        if method == 'histogram':
            return self._event_histogram(events)
        elif method == 'time_surface':
            return self._time_surface(events)
    
    def _event_histogram(self, events):
        """
        Event histogram: accumulate positive and negative events in separate channels.
        Simplest conversion, but loses temporal information.
        """
        frame = np.zeros((self.height, self.width, 2), dtype=np.float32)
        
        for x, y, t, p in events:
            if 0 <= x < self.width and 0 <= y < self.height:
                channel = 0 if p > 0 else 1
                frame[int(y), int(x), channel] += 1
        
        return frame
    
    def _time_surface(self, events):
        """
        Time Surface: records the most recent event time for each pixel.
        Preserves temporal information while converting to a frame-like form.
        """
        time_surface = np.zeros((self.height, self.width, 2), 
                                 dtype=np.float64)
        
        if len(events) == 0:
            return time_surface
        
        t_ref = events[-1][2]  # reference time (most recent)
        
        for x, y, t, p in events:
            if 0 <= x < self.width and 0 <= y < self.height:
                channel = 0 if p > 0 else 1
                time_surface[int(y), int(x), channel] = np.exp(
                    -(t_ref - t) / self.time_window
                )
        
        return time_surface
    
    def events_to_optical_flow(self, events, dt_us=10000):
        """
        Estimate optical flow from events (simplified contrast maximization).
        
        Key idea: temporally warping events with the correct optical flow
        maximizes the sharpness of image edges (maximal contrast).
        """
        if len(events) < 100:
            return np.zeros((self.height, self.width, 2))
        
        # Contrast maximization:
        # argmax_{v} Var(I_warp(v))
        # where I_warp is the event image warped by flow v
        
        # Simplified implementation (optimization is needed in practice)
        best_flow = np.zeros(2)
        best_contrast = 0
        
        for vx in np.linspace(-2, 2, 20):
            for vy in np.linspace(-2, 2, 20):
                warped = np.zeros((self.height, self.width))
                
                t_ref = events[-1][2]
                for x, y, t, p in events:
                    dt = (t_ref - t) / 1e6  # in seconds
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

## 13.6 4D Radar Fusion

4D imaging radar provides four-dimensional information: range, azimuth, elevation, and Doppler velocity. It adds elevation and Doppler velocity to the range and angle provided by traditional automotive radar.

**Principle of range/velocity measurement in FMCW radar**: Most 4D radars use FMCW (Frequency-Modulated Continuous Wave). The transmitted signal's frequency increases linearly over time (chirp); range is measured from the beat frequency of the reflected signal, and velocity from the phase change between chirps:

$$R = \frac{c \cdot f_b}{2 \cdot S}$$

Here $R$ is the range to the target, $c$ is the speed of light, $f_b$ is the beat frequency, and $S$ is the chirp's frequency slope (Hz/s).

$$v = \frac{\lambda \cdot \Delta\phi}{4\pi \cdot T_c}$$

Here $v$ is the radial velocity of the target, $\lambda$ is the carrier wavelength, $\Delta\phi$ is the phase change between two consecutive chirps, and $T_c$ is the chirp period.

### 13.6.1 Adverse-Weather Robustness

4D imaging radar often retains useful range and Doppler detections in conditions that degrade visible and near-infrared sensors, but it is not immune to adverse weather. The following table describes qualitative tendencies, not a performance guarantee for a particular device.

| Condition | Camera tendency | LiDAR tendency | 4D radar tendency | Main variables |
|------|-----------------|----------------|-------------------|----------------|
| Clear day | Rich color and texture information | Favors dense geometric detail | Provides range and Doppler, with potentially limited angular resolution | Target RCS, range, angular resolution |
| Rain | Contrast loss and lens wetting | Backscatter and attenuation | Often less degraded, but rain clutter and a wet radome still matter | Rain rate, radome condition, frequency |
| Fog | Reduced contrast and visibility | Backscatter and attenuation | Often relatively robust | Fog density, range, target RCS |
| Snow/dust | Occlusion and contrast loss | Particle backscatter | Clutter, multipath, and surface accumulation remain | Particle density, accumulation, target RCS |
| Night | Depends on illumination and exposure | Active ranging remains available | Active range and Doppler remain available | Illumination, surface reflectivity, target RCS |
| Direct sunlight | Glare and saturation are possible | Solar background can affect the receiver | Optically insensitive, but RF interference remains | Receiver/filter, RF interference, mounting direction |

Millimeter-wave radar is generally less sensitive to fog-sized particles than visible or near-infrared sensing, but the effect is not zero. Heavy precipitation, a wet or iced radome, multipath, RF interference, low-RCS targets, and limited angular resolution can reduce detection range or increase false alarms. Validate detection range, false-alarm rate, and Doppler error for each device under the target weather conditions.

### 13.6.2 4D Radar + Camera Fusion

Combining 4D radar with a camera can compensate for their different failure modes. Validate cost and adverse-weather performance for the sensor models, mounting, compute platform, and target dataset.

**BEV-based fusion**: BEV features are extracted from camera images (as in LSS or BEVFormer) and radar points are projected into the BEV space to be combined.

**Leveraging radar's Doppler information**: 4D radar directly measures the radial velocity along the line of sight for each point. This is information unique to radar, absent from cameras and LiDAR:

- **Dynamic object classification**: the Doppler channel immediately separates static background from moving objects.
- **Ego-motion estimation**: the Doppler of static points allows ego-velocity estimation (even without an IMU).
- **Tracking support**: the velocity information of an object can be used directly in tracking.

$$v_r = (\mathbf{v}_{\text{obj}} - \mathbf{v}_{\text{ego}}) \cdot \hat{\mathbf{r}}$$

Here $v_r$ is the measured radial velocity, $\mathbf{v}_{\text{obj}}$ is the object velocity, $\mathbf{v}_{\text{ego}}$ is the ego velocity, and $\hat{\mathbf{r}}$ is the unit direction vector from the radar to the target ($\|\hat{\mathbf{r}}\| = 1$). For a static object ($\mathbf{v}_{\text{obj}} = \mathbf{0}$), $v_r = -\mathbf{v}_{\text{ego}} \cdot \hat{\mathbf{r}}$.

### 13.6.3 Recent Developments in Radar Odometry

Radar odometry has advanced since 2020:

**FMCW radar odometry**: odometry on scanning FMCW radar (Navtech, etc.). Feature points are extracted and matched in range-azimuth images to estimate ego-motion.

**4D radar odometry**: odometry on 4D radar point clouds. Approaches similar to LiDAR odometry (ICP, feature matching) are feasible, but they face the challenges of low resolution and high noise.

**Doppler-based ego-velocity estimation**: Ego velocity is estimated directly from the Doppler measurements of static points. Dynamic points are removed with RANSAC, and $\mathbf{v}_{\text{ego}}$ is estimated from the Doppler of static points:

$$v_r^{(k)} = -\mathbf{v}_{\text{ego}} \cdot \hat{\mathbf{r}}^{(k)} \quad \text{(static point)}$$

Here $k$ is the point index. At least three non-collinear points are required to estimate the 3D velocity vector.

```python
def estimate_ego_velocity_from_doppler(radar_points, doppler_values, 
                                        directions, ransac_threshold=0.3,
                                        max_iterations=100):
    """
    Estimate ego-velocity from radar Doppler measurements.
    
    Doppler of static points: v_r = -v_ego . r_hat
    This is a linear system.
    
    Args:
        radar_points: (N, 3) radar point coordinates
        doppler_values: (N,) radial velocity for each point
        directions: (N, 3) unit direction vector for each point
        ransac_threshold: RANSAC inlier threshold (m/s)
        max_iterations: maximum number of RANSAC iterations
        
    Returns:
        v_ego: (3,) ego-velocity vector
        inlier_mask: (N,) static point mask
    """
    N = len(doppler_values)
    best_inliers = np.zeros(N, dtype=bool)
    best_v_ego = np.zeros(3)
    
    for _ in range(max_iterations):
        # Randomly sample 3 points
        idx = np.random.choice(N, 3, replace=False)
        
        # Linear system: v_r = -directions @ v_ego
        # A @ v_ego = b
        A = -directions[idx]  # (3, 3)
        b = doppler_values[idx]  # (3,)
        
        try:
            v_ego_candidate = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            continue
        
        # Compute residuals for all points
        predicted_doppler = -directions @ v_ego_candidate
        residuals = np.abs(doppler_values - predicted_doppler)
        
        inliers = residuals < ransac_threshold
        
        if np.sum(inliers) > np.sum(best_inliers):
            best_inliers = inliers
            best_v_ego = v_ego_candidate
    
    # Re-estimate with inliers (least squares)
    if np.sum(best_inliers) >= 3:
        A = -directions[best_inliers]
        b = doppler_values[best_inliers]
        best_v_ego = np.linalg.lstsq(A, b, rcond=None)[0]
    
    return best_v_ego, best_inliers


def separate_static_dynamic(radar_points, doppler_values, directions,
                             v_ego, threshold=0.5):
    """
    Use the ego-velocity to classify points as static or dynamic.
    
    Args:
        v_ego: estimated ego-velocity (3,)
        threshold: static/dynamic classification threshold (m/s)
        
    Returns:
        static_mask: (N,) static points
        dynamic_mask: (N,) dynamic points
        object_velocities: (N,) estimated object velocity for each point (radial)
    """
    # Expected Doppler under the static assumption
    expected_doppler = -directions @ v_ego
    
    # Residual = actual Doppler - expected Doppler
    residuals = doppler_values - expected_doppler
    
    static_mask = np.abs(residuals) < threshold
    dynamic_mask = ~static_mask
    
    # Radial component of dynamic point object velocity
    object_velocities = residuals  # v_obj . r_hat
    
    return static_mask, dynamic_mask, object_velocities
```

### 13.6.4 Representative Datasets and Benchmarks

| Dataset | Sensors | Environment | Features |
|----------|------|------|------|
| **[Boreas](https://arxiv.org/abs/2203.10168)** (Burnett et al. 2023) | Camera, LiDAR, Radar, GNSS/IMU | Urban (various weather) | Same route repeated for one year, including adverse weather |
| **RadarScenes** | Radar, Camera, LiDAR | Urban | Traditional automotive radar points + semantic labels (point-level annotation) |
| **nuScenes** | Camera, LiDAR, Radar | Urban | Includes 5 radars, some adverse weather |
| **View-of-Delft** | Camera, LiDAR, 4D Radar | Urban | 4D radar + 3D annotation |

**Recent systems and datasets (2024-2025)**:

- **[Snail-Radar](https://arxiv.org/abs/2407.11705)** (2024): A benchmark for evaluating 4D-radar-based SLAM, providing 44 sequences collected across three platforms (handheld, bicycle, SUV) under diverse weather and lighting conditions.
- **[4D Radar-Inertial Odometry](https://arxiv.org/abs/2412.13639)** (2024): Proposes a 3D Gaussian radar scene representation and multi-hypothesis scan matching, achieving more precise radar odometry than voxel-based methods.

The deployment scope of 4D radar fusion remains limited. Ego-motion estimation and dynamic object classification based on Doppler information provide capabilities that LiDAR and cameras do not.

---

## Closing

The sensor fusion pipeline runs from sensor modeling (Ch.2) through calibration (Ch.3), state estimation theory (Ch.4), feature matching (Ch.5), VO/VIO (Ch.6), LiDAR odometry (Ch.7), multi-sensor fusion (Ch.8), Place Recognition (Ch.9), Loop Closure (Ch.10), spatial representation (Ch.11), practical systems (Ch.12), and research frontiers (Ch.13).

Four trends emerge from this sequence:

1. **Traditional methods remain the foundation.** Kalman filters, ICP, RANSAC, factor graphs — methods proposed decades ago form the skeleton of modern systems.
2. **Deep learning is widely used in perception.** In feature matching, depth estimation, and Place Recognition, learned methods have reported higher scores than traditional baselines on several public benchmarks. The scope of that advantage depends on the dataset and evaluation protocol.
3. **In inference, traditional and learned methods coexist.** State estimation backends are still dominated by optimization-based methods, but efforts such as DROID-SLAM are dissolving the boundary through differentiable optimization.
4. **Foundation models are being integrated into the pipeline.** Sensor-fusion pipelines use representations from general-purpose models such as DINOv2 and SAM at several stages.

Sensor fusion combines incomplete observations to estimate the state of the environment. A system design should consider sensor models, estimators, and failure modes together.
