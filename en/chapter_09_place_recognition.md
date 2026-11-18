# Chapter 9. Place Recognition

Even tightly coupled multi-sensor odometry (Chapter 8) accumulates trajectory drift over long runs wherever absolute references such as GNSS are absent. Reversing that drift requires recognizing previously visited locations from current sensor streams and handing verified candidates to the loop closure pipeline (→ Chapter 10).

> Answering the core question: "Have I seen this place before?"
> Place recognition supplies candidates for loop closure while also supporting multi-session map merging and global relocalization.

---

## 9.1 Problem Definition

### 9.1.1 Place Recognition vs Loop Closure Detection

**Place Recognition (PR)** is a standalone retrieval problem: given a current sensor observation, which entry in a database depicts the same physical place? It can be posed and evaluated independently of any trajectory estimator. **Loop Closure Detection**, by contrast, operates inside a SLAM system to verify whether the robot has actually returned to a past pose. Place recognition retrieves visually or geometrically similar candidates, and the loop closure pipeline applies **geometric verification** before admitting valid matches into the pose graph:

```
Loop Closure Detection Pipeline:
┌─────────────┐    ┌────────────────┐    ┌───────────────────┐    ┌─────────────┐
│  Current    │───→│ Place          │───→│ Geometric         │───→│ Pose Graph  │
│  observation│    │ Recognition    │    │ Verification      │    │ Update      │
│  (query)    │    │ (candidate     │    │ (geometric        │    │ (optimize)  │
│             │    │  retrieval)    │    │  verification)    │    │             │
└─────────────┘    └────────────────┘    └───────────────────┘    └─────────────┘
```

Querying at scale makes retrieval complexity a primary design constraint. Comparing a query against all $N$ past frames costs $O(N)$ per query and $O(N^2)$ across an entire sequence. Global descriptors paired with an exact scan remain $O(N)$, whereas inverted, tree-based, and approximate-nearest-neighbor indexes reduce average query latency subject to data-distribution and recall trade-offs.

### 9.1.2 Retrieval Pipeline

Place recognition follows the standard four-stage architecture of information retrieval:

1. **Encoding**: Compress each observation—image, point cloud, or both—into a fixed-length **global descriptor**.
2. **Indexing**: Register database descriptors in a searchable index such as a kd-tree or FAISS.
3. **Retrieval**: Fetch the database entries closest to the query in descriptor space.
4. **Re-ranking & Verification**: Filter top candidates geometrically to eliminate perceptual aliasing before confirming a match.

At the core of the retrieval step is nearest-neighbor search across the database $\mathcal{D}$:

$$
q^* = \arg\min_{q \in \mathcal{D}} d(\mathbf{f}(\text{query}), \mathbf{f}(q))
$$

Here $\mathbf{f}(\cdot)$ maps an observation to its global descriptor, and $d(\cdot, \cdot)$ measures descriptor distance—typically L2 or cosine distance.

### 9.1.3 Evaluation Metrics

Evaluating a place recognition method requires balancing retrieval accuracy against false-positive risk and runtime latency. The primary accuracy metric is **Recall@N**, defined as the fraction of queries whose top-$N$ retrieved candidates include at least one true match:

$$
\text{Recall@N} = \frac{|\{q : \text{top-}N \text{ candidates contain the correct match for query } q\}|}{|\text{total queries}|}
$$

While **Recall@1** tests whether the single top-ranked candidate succeeds, a SLAM system typically passes top-$k$ candidates to geometric verification based on its compute budget and false-positive cost, so Recall@$k$, precision-recall curves, and query latency should all be reported together. Tracing a **Precision-Recall Curve** across similarity thresholds is especially revealing for SLAM, where even one accepted false-positive loop closure can corrupt the entire pose graph—favoring high precision even at the expense of some recall.

What counts as a "correct match" depends on the benchmark protocol—some rely on spatial distance, others on field-of-view overlap or relative pose thresholds. Several outdoor VPR benchmarks adopt a 25 m GPS radius, though that radius is not universal across indoor or dense urban datasets. The function below computes Recall@N from descriptor and ground-truth position matrices:

```python
import numpy as np
from scipy.spatial.distance import cdist

def compute_recall_at_n(query_descriptors, db_descriptors, 
                         query_positions, db_positions,
                         n_values=[1, 5, 10], threshold_m=25.0):
    """
    Compute Recall@N.
    
    query_descriptors: (Q, D) — query descriptors
    db_descriptors: (M, D) — database descriptors
    query_positions: (Q, 2 or 3) — GPS/GT positions of queries
    db_positions: (M, 2 or 3) — GPS/GT positions of database entries
    n_values: N values for Recall@N
    threshold_m: distance threshold (meters) to count as the same place
    """
    # Descriptor similarity matrix (L2 distance)
    desc_dists = cdist(query_descriptors, db_descriptors, metric='euclidean')
    
    # True distance matrix
    geo_dists = cdist(query_positions, db_positions, metric='euclidean')
    
    recalls = {}
    for n in n_values:
        correct = 0
        for q in range(len(query_descriptors)):
            # Top-N indices by descriptor distance
            top_n_indices = np.argsort(desc_dists[q])[:n]
            
            # Correct if any of the top-N is within the geographic threshold
            min_geo_dist = geo_dists[q, top_n_indices].min()
            if min_geo_dist < threshold_m:
                correct += 1
        
        recalls[f"Recall@{n}"] = correct / len(query_descriptors)
    
    return recalls
```

---

## 9.2 Visual Place Recognition (VPR)

Visual Place Recognition seeks to identify revisited locations from camera imagery alone, making it both the earliest and most extensively developed branch of place recognition.

### 9.2.1 Classical Method: Bag of Words (BoW)

Introduced in **[Video Google (Sivic & Zisserman, 2003)](https://www.robots.ox.ac.uk/~vgg/publications/2003/Sivic03/)**, the **Bag of Visual Words** model adapted document retrieval directly to image search. First, local descriptors such as SIFT or ORB extracted from a training corpus are clustered via k-means into $K$ centers called "visual words." Each image's descriptors are then hard-assigned to their nearest cluster centers to form a $K$-dimensional frequency histogram.

Because common visual features carry little discriminative power, raw counts are reweighted via TF-IDF (Term Frequency – Inverse Document Frequency):

$$
w_{i,d} = \underbrace{\frac{n_{i,d}}{n_d}}_{\text{TF}} \cdot \underbrace{\log \frac{N}{N_i}}_{\text{IDF}}
$$

Here $n_{i,d}$ counts occurrences of visual word $i$ in image $d$, $n_d$ is the total word count in image $d$, $N$ is the database size, and $N_i$ is the number of database images containing word $i$. High TF emphasizes words frequent in the current frame, while high IDF upweights rare, distinctive words across the database. Storing an **inverted index** from each visual word to the images containing it avoids exhaustive scans at query time; candidates sharing query words are then ranked by cosine similarity between TF-IDF vectors:

$$
\text{sim}(\mathbf{v}_q, \mathbf{v}_d) = \frac{\mathbf{v}_q \cdot \mathbf{v}_d}{\|\mathbf{v}_q\| \cdot \|\mathbf{v}_d\|}
$$

In real-time feature-based SLAM, **DBoW2** (used across the ORB-SLAM family) organizes the visual vocabulary into a hierarchical k-means tree for fast quantization and candidate scoring. The simplified class below illustrates BoW quantization, dynamic IDF updating, and inverted-index retrieval from scratch:

```python
import numpy as np
from collections import defaultdict

class SimpleBoW:
    """
    Simplified implementation of Bag of Visual Words.
    """
    def __init__(self, vocabulary):
        """
        vocabulary: (K, D) — descriptors of K visual words (k-means centers)
        """
        self.vocabulary = vocabulary  # (K, D)
        self.K = len(vocabulary)
        self.inverted_index = defaultdict(list)  # word_id -> [(img_id, tf)]
        self.idf = np.ones(self.K)
        self.N = 0  # number of images in the database
        self.db_words = {}  # img_id -> word_ids. TF-IDF vectors are recomputed at query time with the current IDF
    
    def quantize(self, descriptors):
        """Quantize local descriptors to the nearest visual word."""
        # (N_desc, D) vs (K, D) → (N_desc, K) distance matrix
        dists = np.linalg.norm(
            descriptors[:, None, :] - self.vocabulary[None, :, :], axis=2
        )
        word_ids = np.argmin(dists, axis=1)
        return word_ids
    
    def compute_bow_vector(self, descriptors):
        """Compute the image's BoW vector (with TF-IDF weighting)."""
        return self.bow_from_words(self.quantize(descriptors))
    
    def bow_from_words(self, word_ids):
        """Compute the TF-IDF vector from a list of quantized words using the current IDF."""
        
        # Term Frequency
        tf = np.zeros(self.K)
        for w in word_ids:
            tf[w] += 1
        tf /= len(word_ids)  # normalize
        
        # TF-IDF
        tfidf = tf * self.idf
        
        # L2 normalization
        norm = np.linalg.norm(tfidf)
        if norm > 0:
            tfidf /= norm
        
        return tfidf
    
    def add_to_database(self, img_id, descriptors):
        """Add an image to the database."""
        word_ids = self.quantize(descriptors)
        self.db_words[img_id] = word_ids   # store only the words, not the vector
        self.N += 1
        
        # Update inverted index
        unique_words = np.unique(word_ids)
        for w in unique_words:
            self.inverted_index[w].append(img_id)
        
        # Recompute IDF
        for w in range(self.K):
            n_w = len(set(self.inverted_index[w]))  # number of images containing this word
            self.idf[w] = np.log(self.N / (n_w + 1e-6))
    
    def query(self, descriptors, top_k=5):
        """Retrieve the database images most similar to the query image."""
        q_vector = self.compute_bow_vector(descriptors)
        
        # Narrow candidates with the inverted index: only images sharing at least one query word
        candidates = set()
        for w in np.unique(self.quantize(descriptors)):
            candidates.update(self.inverted_index[w])
        
        scores = {}
        for img_id in candidates:
            db_vector = self.bow_from_words(self.db_words[img_id])   # computed with the current IDF
            scores[img_id] = np.dot(q_vector, db_vector)
        
        # Sort by similarity
        sorted_results = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_results[:top_k]
```

Despite its speed, BoW suffers from heavy quantization loss—collapsing distinct local features onto identical visual words—sensitive dependence on vocabulary size $K$, and only whatever illumination or viewpoint invariance the underlying SIFT or ORB descriptors provide.

### 9.2.2 VLAD and Fisher Vector

To recover the detail discarded by BoW histograms, residual aggregation methods encode how each local descriptor deviates from its assigned cluster center. In **[VLAD (Vector of Locally Aggregated Descriptors, Jégou et al., 2010)](https://doi.org/10.1109/CVPR.2010.5540039)**, each cluster $k$ accumulates the vector residuals of descriptors assigned to it:

$$
\mathbf{V}_k = \sum_{i: \text{NN}(\mathbf{x}_i) = k} (\mathbf{x}_i - \mathbf{c}_k)
$$

Concatenating every cluster residual $\mathbf{V}_k$ yields $\mathbf{V} = [\mathbf{V}_1^T, \mathbf{V}_2^T, \ldots, \mathbf{V}_K^T]^T$ of dimension $K \times D$. **Fisher Vector** generalizes this idea by modeling visual words with a Gaussian Mixture Model (GMM) and encoding both first- and second-order statistics normalized by the square root of the Fisher Information Matrix, doubling dimensionality to $2KD$. Whether Fisher Vector or VLAD retrieves better depends on the metric, dataset, and normalization; VLAD can be viewed as a streamlined special case, and at equal dimensionality the two often perform comparably or favor VLAD.

### 9.2.3 NetVLAD: The Baseline for Learning-Based VPR

**[NetVLAD (Arandjelović et al., 2016)](https://arxiv.org/abs/1511.07247)** made VLAD end-to-end trainable by turning descriptor aggregation into a **differentiable CNN layer**. In classical VLAD, hard assignment (1 if $\text{NN}(\mathbf{x}_i) = k$, else 0) blocks gradient flow. NetVLAD replaces that step with a softmax soft assignment:

$$
\bar{a}_k(\mathbf{x}_i) = \frac{e^{\mathbf{w}_k^T \mathbf{x}_i + b_k}}{\sum_{k'} e^{\mathbf{w}_{k'}^T \mathbf{x}_i + b_{k'}}}
$$

Expanding Euclidean distance assignment $e^{-\alpha\|\mathbf{x}_i - \mathbf{c}_k\|^2}$ shows that setting $\mathbf{w}_k = 2\alpha\mathbf{c}_k$ and $b_k = -\alpha\|\mathbf{c}_k\|^2$ recovers soft assignment while allowing these parameters and cluster centers to adapt during training. Aggregating weighted residuals produces the NetVLAD matrix:

$$
\mathbf{V}(j, k) = \sum_{i=1}^{N} \bar{a}_k(\mathbf{x}_i) (x_i(j) - c_k(j))
$$

Intra-normalizing, flattening, and L2-normalizing this $D \times K$ matrix yields the final global descriptor.

Because nearby GPS coordinates do not guarantee overlapping camera viewpoints—images from opposite directions share a coordinate without sharing a scene—NetVLAD trains with **weakly supervised** triplet ranking on Google Street View Time Machine imagery. Geographically proximate frames form a candidate positive set from which the closest match in feature space is selected as $p^+$, while distant frames yield hard negatives $p^-$ under margin $m$:

$$
\mathcal{L} = \sum_{(q, p^+, p^-)} \max\left(0, m + d(\mathbf{f}(q), \mathbf{f}(p^+)) - d(\mathbf{f}(q), \mathbf{f}(p^-))\right)
$$

Built on a VGG-16 backbone, NetVLAD reached roughly 84.3% Recall@1 on Pitts250k and 86.3% on Pitts30k, outperforming the hand-crafted descriptors of its era and establishing the standard baseline for learned VPR.

### 9.2.4 AnyLoc: Foundation-Model-Based Universal VPR

**[AnyLoc (Keetha et al., 2023)](https://arxiv.org/abs/2308.00688)** investigates whether zero-shot place recognition can generalize across diverse environments without VPR-specific training by pairing self-supervised foundation-model features with classical unsupervised VLAD. Rather than taking the global CLS token, AnyLoc extracts **dense patch features** from an intermediate layer of DINOv2 ViT-G/14—preserving local spatial structure that a single token discards—clusters them with k-means into a visual vocabulary, and aggregates residuals with hard-assignment VLAD. The paper evaluates PCA groupings across Urban, Indoor, Aerial, SubT, Degraded, and Underwater domains using domain-specific vocabularies; both layer selection and vocabulary design depend on the target backbone and domain and should be validated on new deployments.

Self-supervised on 142 million images, [DINOv2 (Oquab et al., 2023)](https://arxiv.org/abs/2304.07193) learns representations that transfer well across appearance shifts without place labels, though they do not automatically solve repetitive architectural geometry, extreme viewpoint changes, or unfamiliar sensor modalities. Across the benchmarks and evaluation protocols reported in the AnyLoc paper, this zero-shot pipeline delivered notable relative gains:

- **Day-night changes**: Recall@1 up to 5–21 percentage points higher than tested MixVPR- and CosPlace-family baselines
- **Seasonal changes**: 8–9 percentage points higher in reported cases
- **Opposite viewpoint (180 degrees)**: 39–49 percentage points higher in reported cases
- **Unstructured environments (underwater and subterranean)**: up to roughly 4x the tested baseline
- **Compact compression**: PCA whitening from 49K down to 512 dimensions is also evaluated, though the resulting recall shift varies by dataset

The script below sketches the conceptual AnyLoc pipeline from patch extraction through VLAD aggregation and PCA whitening:

```python
import numpy as np

def anyloc_pipeline(images, dino_model, n_clusters=64, desc_dim=512):
    """
    Conceptual implementation of the AnyLoc pipeline.
    
    1. Extract dense features from DINOv2
    2. Build a visual vocabulary with k-means
    3. VLAD aggregation
    4. PCA dimensionality reduction
    """
    # Step 1: Dense feature extraction (ViT patch tokens)
    all_patch_features = []
    image_patch_features = []
    
    for img in images:
        # DINOv2 forward: patch tokens of shape (1, N_patches, D_feat)
        patch_tokens = dino_model.get_intermediate_layers(img, n=1)[0]
        # These are the output tokens of the last block. Using the value facet
        # of a specific layer — AnyLoc's key finding — requires hooking the
        # attention module.
        image_patch_features.append(patch_tokens)
        all_patch_features.append(patch_tokens)
    
    # Step 2: Build k-means visual vocabulary
    all_features = np.concatenate([p.reshape(-1, p.shape[-1]) for p in all_patch_features], axis=0)  # (N_total_patches, D_feat)
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(all_features)
    centers = kmeans.cluster_centers_  # (K, D_feat)
    
    # Step 3: Compute VLAD descriptor for each image
    vlad_descriptors = []
    D_feat = centers.shape[1]
    
    for patches in image_patch_features:
        # Hard assignment
        patches = patches.reshape(-1, patches.shape[-1])   # (1, N, D) → (N, D)
        assignments = kmeans.predict(patches)  # (N_patches,)
        
        # VLAD: sum residuals for each cluster
        vlad = np.zeros((n_clusters, D_feat))
        for i, patch_feat in enumerate(patches):
            k = assignments[i]
            vlad[k] += patch_feat - centers[k]
        
        # Intra-normalization (normalize each cluster vector individually)
        for k in range(n_clusters):
            norm = np.linalg.norm(vlad[k])
            if norm > 0:
                vlad[k] /= norm
        
        # Flatten + L2 normalization
        vlad_flat = vlad.flatten()
        vlad_flat /= (np.linalg.norm(vlad_flat) + 1e-10)
        
        vlad_descriptors.append(vlad_flat)
    
    vlad_descriptors = np.array(vlad_descriptors)
    
    # Step 4: PCA dimensionality reduction
    from sklearn.decomposition import PCA
    pca = PCA(n_components=desc_dim, whiten=True)
    reduced_descriptors = pca.fit_transform(vlad_descriptors)
    
    # Final L2 normalization
    norms = np.linalg.norm(reduced_descriptors, axis=1, keepdims=True)
    reduced_descriptors /= (norms + 1e-10)
    
    return reduced_descriptors  # (N_images, desc_dim)
```

### 9.2.5 EigenPlaces

**[EigenPlaces (Berton et al., 2023)](https://arxiv.org/abs/2308.10832)** builds on [CosPlace (Berton et al., 2022)](https://arxiv.org/abs/2204.02287) to make learned descriptors invariant across large viewpoint changes. Where CosPlace replaced triplet hard-negative mining with classification-based training, EigenPlaces structures those classes geometrically by running PCA on the training camera positions within each place cell rather than on the descriptor space. The first principal component generally follows the road direction, so EigenPlaces defines classes from multiple viewpoints aimed at focal points perpendicular to that axis (looking toward roadside facades), supplemented by a frontal class aimed along the road axis for forward-facing datasets.

### 9.2.6 Expanding the Use of Foundation Models

Research since AnyLoc explores several ways to build on foundation models for VPR. **CLIP** supports semantic place queries tied to natural language (such as "city street" or "forest trail"), though in AnyLoc's foundation-model comparison DINOv2 achieved stronger place discrimination under the authors' tested layers, facets, and datasets. **SAM (Segment Anything)** is being investigated for extracting segmentation masks as structural place signatures, while **DINOv2 with learned aggregation** replaces AnyLoc's unsupervised VLAD with trainable heads—SALAD (§9.7.4) casts feature-to-cluster assignment as optimal transport, and SelaVPR (Lu et al., ICLR 2024) adapts the pre-trained backbone directly for VPR.

### 9.2.7 SeqSLAM and Sequence Matching

When single images look indistinguishable—such as identical blocks along a suburban street—sequential context often remains unique (for instance, left turn followed by park and crosswalk). **[SeqSLAM (Milford & Wyeth, 2012)](https://doi.org/10.1109/ICRA.2012.6224623)** matches **image sequences** rather than single frames by reducing each frame to a compact low-resolution representation, forming a similarity matrix between query and database trajectories, and searching for optimal diagonal paths across constrained vehicle speeds:

$$
S_{\text{seq}}(q_s, d_s) = \min_{\text{path}} \sum_{i=0}^{L-1} d(\mathbf{f}(q_{s+i}), \mathbf{f}(d_{s+\delta(i)}))
$$

Here $L$ is the sequence length and $\delta(i)$ parameterizes the path step to accommodate velocity variations. Because temporal ordering persists even across day-to-night and summer-to-winter appearance shifts, sequence matching remains effective under severe visual change, and later work such as SeqNet (Garg et al., 2021) extends this formulation into a learned pipeline.

---

## 9.3 LiDAR Place Recognition

While visual place recognition struggles across day-night illumination shifts, 3D LiDAR point clouds capture direct scene geometry unaffected by ambient lighting—though they remain sensitive to seasonal foliage changes and large viewpoint offsets.

### 9.3.1 Handcrafted Method: Scan Context

**[Scan Context (Kim & Kim, 2018)](https://doi.org/10.1109/IROS.2018.8593953)** compresses a 3D LiDAR sweep into a **global matrix descriptor that preserves spatial layout directly** without quantizing features into a histogram or requiring learned weights. Because it needs no training, Scan Context serves as the loop-detection module in several LiDAR SLAM systems—note, however, that the public LIO-SAM repository defaults to radius search with ICP verification, while Scan Context integration is maintained in separate forks such as SC-LIO-SAM.

To construct the descriptor, Scan Context partitions the horizontal plane around the sensor into $N_s$ azimuthal sectors and $N_r$ radial rings, forming an $N_s \times N_r$ polar grid (commonly 60×20). Rather than averaging point elevations, each cell records the **maximum height** among points falling into that bin, forming $\mathbf{SC} \in \mathbb{R}^{N_s \times N_r}$:

$$
\mathbf{SC}(i, j) = \max_{p \in \text{bin}(i,j)} z(p)
$$

Taking the maximum height preserves tall protruding structures such as building facades, trees, and utility poles far better than mean elevation. Because the polar grid is centered on the sensor, revisiting the same position with a different heading merely **circularly shifts** the matrix rows along the sector axis, enabling rotation-invariant matching via row-shift search. Real reverse-direction traverses also introduce lateral lane offsets and altered occlusions, which a pure row shift cannot absorb; the original paper therefore augments stored scans via root shifting.

Comparing full matrices exhaustively against every past scan is slow, so retrieval proceeds in two stages. First, averaging each ring column along the sector axis produces a rotation-invariant **Ring Key** $\mathbf{k}_r = [\bar{h}_1, \bar{h}_2, \ldots, \bar{h}_{N_r}]$, which narrows candidates quickly in a kd-tree. Next, a rotation-variant **Sector Key**—aggregating along the ring axis per sector—estimates an initial yaw alignment so that full cosine-distance scoring need only test shifts near that offset (the formulation and code below sweep all shifts for clarity):

$$
d(\mathbf{SC}_q, \mathbf{SC}_d) = \min_{s \in [0, N_s)} \frac{1}{N_s}\sum_{i=1}^{N_s}\left(1 - \frac{\mathbf{c}^q_i \cdot \mathbf{c}^d_{i+s}}{\|\mathbf{c}^q_i\|\,\|\mathbf{c}^d_{i+s}\|}\right)
$$

The reference class below implements polar max-height binning alongside two-stage Ring Key filtering and circular row-shift scoring:

```python
import numpy as np

class ScanContext:
    """
    Scan Context descriptor generation and matching.
    """
    def __init__(self, n_sectors=60, n_rings=20, max_range=80.0):
        self.n_sectors = n_sectors
        self.n_rings = n_rings
        self.max_range = max_range
        self.database = []
        self.ring_keys = []
    
    def make_scan_context(self, point_cloud):
        """
        3D point cloud → Scan Context matrix.
        
        point_cloud: (N, 3) — x, y, z coordinates
        """
        sc = np.zeros((self.n_sectors, self.n_rings))
        
        # Polar conversion
        x, y, z = point_cloud[:, 0], point_cloud[:, 1], point_cloud[:, 2]
        ranges = np.sqrt(x**2 + y**2)
        angles = np.arctan2(y, x)  # [-pi, pi]
        angles = (angles + np.pi) / (2 * np.pi)  # normalize to [0, 1]
        
        # Remove out-of-range points
        valid = ranges < self.max_range
        ranges, angles, z = ranges[valid], angles[valid], z[valid]
        
        # Compute bin indices
        sector_idx = np.clip((angles * self.n_sectors).astype(int), 0, self.n_sectors - 1)
        ring_idx = np.clip((ranges / self.max_range * self.n_rings).astype(int), 0, self.n_rings - 1)
        
        # Max height per bin
        for i in range(len(z)):
            si, ri = sector_idx[i], ring_idx[i]
            sc[si, ri] = max(sc[si, ri], z[i])
        
        return sc
    
    def make_ring_key(self, sc):
        """Ring key: mean height of each ring."""
        return np.mean(sc, axis=0)  # (n_rings,)
    
    def add_to_database(self, point_cloud):
        """Add a scan to the database."""
        sc = self.make_scan_context(point_cloud)
        rk = self.make_ring_key(sc)
        self.database.append(sc)
        self.ring_keys.append(rk)
    
    def query(self, point_cloud, top_k=5, n_candidates=20):
        """
        Retrieve the database scans most similar to the query scan.
        """
        sc_query = self.make_scan_context(point_cloud)
        rk_query = self.make_ring_key(sc_query)
        
        # Stage 1: Candidate selection via ring key
        rk_dists = [np.linalg.norm(rk_query - rk) for rk in self.ring_keys]
        candidate_indices = np.argsort(rk_dists)[:n_candidates]
        
        # Stage 2: Scan Context row-shift matching
        scores = []
        for idx in candidate_indices:
            sc_db = self.database[idx]
            min_dist = float('inf')
            for shift in range(self.n_sectors):
                sc_shifted = np.roll(sc_db, shift, axis=0)
                # Mean of per-sector (row) cosine distances — the distance defined in the original paper
                num = np.sum(sc_query * sc_shifted, axis=1)
                den = np.linalg.norm(sc_query, axis=1) * np.linalg.norm(sc_shifted, axis=1) + 1e-9
                dist = np.mean(1.0 - num / den)
                min_dist = min(min_dist, dist)
            scores.append((idx, min_dist))
        
        scores.sort(key=lambda x: x[1])
        return scores[:top_k]
```

Building on this design, **[Scan Context++](https://arxiv.org/abs/2109.13494)** extends structural descriptors to handle larger heading rotations and lateral lane offsets when roll and pitch variations stay bounded, and pairs place retrieval with a 1-DOF relative alignment estimate.

### 9.3.2 M2DP and ESF

Before Scan Context, handcrafted 3D descriptors emphasized compactness or strict rotation invariance over explicit spatial grids. **M2DP (He et al., 2016)** projects a point cloud onto multiple 2D planes and compresses the resulting point-density distributions via SVD into a 192-dimensional rotation-invariant descriptor. **ESF (Ensemble of Shape Functions)** instead aggregates histograms of pairwise point distances, angles, and triangle area ratios. Although neither preserves 3D layout as explicitly as Scan Context, both offer compact or rotation-invariant signatures.

### 9.3.3 Learning-based: PointNetVLAD

**[PointNetVLAD (Uy & Lee, 2018)](https://arxiv.org/abs/1804.03492)** brought end-to-end learning to 3D point-cloud retrieval by pairing a PointNet feature extractor with a NetVLAD aggregation layer trained under lazy triplet and quadruplet losses:

$$
\mathcal{L} = \max(0, m + \max_{p^+} d(q, p^+) - \min_{p^-} d(q, p^-))
$$

Because PointNet processes points largely independently before global pooling, however, it struggles to model fine local neighborhood geometry and scales poorly to dense outdoor scans.

### 9.3.4 MinkLoc3D

**[MinkLoc3D (Komorowski, 2021)](https://arxiv.org/abs/2011.04530)** ([code](https://github.com/jac99/MinkLoc3D)) overcomes PointNet's lack of local neighborhood modeling by replacing it with a **Minkowski Convolutional Neural Network**. Sparse 3D convolutions capture multi-scale local geometry efficiently across occupied voxels, and Generalized Mean (GeM) pooling aggregates those features into a compact global descriptor.

### 9.3.5 OverlapTransformer: Range-Image-Based

Rather than operating on irregular 3D voxels or point sets, **[OverlapTransformer (Ma et al., 2022)](https://arxiv.org/abs/2203.03397)** projects each rotating LiDAR scan onto a structured 2D **range image** of size $(h, w)$, where $h$ equals the laser beam count, $w$ reflects horizontal angular resolution, and each pixel stores measured range.

A lightweight CNN extracts local range-image features, a Transformer encoder captures global structural context across columns, and a NetVLAD head aggregates the sequence into a global descriptor for fast inference. Its predecessor **[OverlapNet](https://github.com/PRBonn/OverlapNet)** similarly operates on range images, directly predicting scan overlap and relative yaw between scan pairs.

### 9.3.6 BEV-Based Methods

A parallel line of work projects 3D scans into a 2D Bird's Eye View (BEV) plane so standard 2D image retrieval architectures can process point clouds directly—for example, **BEVPlace**, which extracts NetVLAD descriptors from BEV projection images.

---

## 9.4 Cross-Modal Place Recognition

### 9.4.1 Why Cross-Modal PR Is Needed

In heterogeneous fleets and multi-stage mapping pipelines, **the sensor creating the map often differs from the sensor localizing against it**—for instance, a camera-only delivery robot or UAV localizing against a high-definition map previously built by a LiDAR-equipped vehicle, or cross-vehicle retrieval between a LiDAR+Camera car and a vision-only platform. Bridging those deployments requires **cross-modal place recognition** between LiDAR point clouds and camera images.

### 9.4.2 Domain Gap in Cross-Modal PR

Matching across sensing modalities is difficult because LiDAR sweeps and camera frames represent physical scenes in fundamentally different ways:

| Property | LiDAR point cloud | Camera image |
|------|---------------------|-------------|
| Data structure | Unstructured 3D point set | Structured 2D grid |
| Information | Geometry | Appearance |
| Illumination dependence | None | Very high |
| Texture | None | Rich |
| Density | Point spacing grows with range (density per unit surface area falls with the square of range and also depends on incidence angle) | Uniform in the image plane |

This structural discrepancy—the **domain gap**—pushes descriptors of the same physical place far apart whenever they originate from different sensors.

### 9.4.3 (LC)²: LiDAR-Camera Cross-Modal PR

**[(LC)² (Lee et al., 2023)](https://arxiv.org/abs/2304.08660)** bridges this domain gap by mapping LiDAR scans and camera images into a **shared embedding space**. First, both inputs are brought into a common range-image representation: LiDAR sweeps project directly to range images, while camera frames convert into disparity/depth images. Modality-specific encoders process each representation and train under a **contrastive loss** that pulls matching LiDAR-camera pairs together while pushing mismatched pairs apart beyond margin:

$$
\mathcal{L}_{\text{contrastive}} = \sum_{(l, c) \in \mathcal{P}^+} \| \mathbf{f}_L(l) - \mathbf{f}_C(c) \|^2 + \sum_{(l, c) \in \mathcal{P}^-} \max(0, m - \| \mathbf{f}_L(l) - \mathbf{f}_C(c) \|)^2
$$

Here $\mathbf{f}_L$ is the LiDAR encoder, $\mathbf{f}_C$ is the camera encoder, and $\mathcal{P}^+$ and $\mathcal{P}^-$ denote positive and negative cross-modal pairs.

### 9.4.4 ModaLink

**ModaLink** (Xie et al., IROS 2024) offers another cross-modal retrieval framework aimed specifically at querying LiDAR point-cloud maps from RGB camera images; sensor combinations incorporating radar fall outside its scope.

### 9.4.5 Modality-Agnostic Descriptor Approach

More broadly, **modality-agnostic descriptors** seek a single embedding invariant to which sensor observed the scene, pursuing three main strategies: **knowledge distillation** transfers representations from a multi-sensor teacher (such as LiDAR+Camera) into a single-modality student; **canonical representations** project every sensor stream into a shared geometric or semantic layout such as BEV before encoding; and **foundation-model alignment** investigates whether features extracted by models like DINOv2 from camera images align naturally with features extracted from rendered LiDAR views.

---

## 9.5 Multi-Session & Long-Term Place Recognition

### 9.5.1 Challenges of Long-Term VPR

Across repeated missions over days, months, or years, a single physical location can look dramatically different. Diurnal lighting cycles (day versus night), seasonal transitions (summer foliage versus snow cover), weather (clear skies versus dense fog), and physical modifications (construction, demolition, roadwork) all shift appearance descriptors and degrade retrieval recall.

### 9.5.2 Strategies for Seasonal/Time-of-Day/Weather Changes

Robust long-term pipelines generally combine four complementary defenses against appearance shift:

1. **Multi-condition data augmentation**: Training descriptors on repeat traverses under varied lighting and weather, exemplified by NetVLAD's use of Google Street View Time Machine imagery.
2. **Domain-invariant representations**: Encoding cues less tied to raw illumination, such as semantic segmentation layouts (buildings, roads, sky)—though when severe night or backlighting degrades the segmenter itself, the derived layout degrades along with it.
3. **Foundation-model features**: Leveraging self-supervised backbones such as DINOv2, whose augmentation-invariant training gives AnyLoc-style descriptors strong tolerance to seasonal and lighting change.
4. **Sequence matching**: Following SeqSLAM, matching short trajectory windows rather than isolated frames so temporal structure persists even when per-frame appearance shifts sharply.

### 9.5.3 Map Update Strategies

Descriptor robustness alone is not enough when environments physically evolve over months of operation; the database itself must adapt. **Experience-based maps** retain multiple historical observations per location and query against the experience closest to current conditions, trading steady storage growth for recall. **Adaptive descriptors** instead update stored embeddings incrementally—for instance via an exponential moving average—as new traverses arrive, while **change detection** flags structural edits such as demolished buildings so stale regions can be invalidated and rebuilt.

### 9.5.4 Lifelong Place Recognition

Sustaining place recognition across a robot's entire operational lifetime remains an open research frontier. A lifelong system must bound memory consumption as traverses accumulate, absorb ongoing environmental change, forget obsolete appearance safely, and incorporate newly explored spaces without catastrophic forgetting—connecting place retrieval directly to continual and incremental learning.

---

## 9.6 Geometric Verification & Re-ranking

Global descriptor similarity alone cannot rule out perceptual aliasing across repetitive corridors, intersections, or facades. Before a candidate enters the SLAM pose graph, **geometric verification** checks whether the two observations support a consistent 3D rigid transformation, filtering out false positives that could otherwise warp the map.

### 9.6.1 PnP + RANSAC (Visual)

For camera-based candidates, verification begins with local feature matching (such as SuperPoint+LightGlue or ORB with brute-force matching) between query and candidate images. Lifting those 2D-2D correspondences onto the candidate's associated 3D map points produces 2D-3D pairs for **Perspective-n-Point (PnP)** pose estimation inside a **RANSAC** loop that rejects outlier matches. When the 3D points reside in the global map frame, PnP recovers the query camera's map-frame pose directly, so the relative pose to the candidate follows by composing with the candidate's known pose. The function below implements this PnP+RANSAC check:

```python
import numpy as np

def geometric_verification_visual(query_keypoints, db_keypoints_3d, 
                                    K, ransac_threshold=5.0,
                                    min_inliers=30):
    """
    Visual geometric verification: PnP + RANSAC.
    
    query_keypoints: (N, 2) — 2D keypoints in the query image
    db_keypoints_3d: (N, 3) — matched 3D points from the database
    K: (3, 3) — camera intrinsic parameters
    """
    import cv2
    
    # Estimate camera pose via PnP + RANSAC
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
    
    # solvePnP's (rvec, tvec) is the transform from the 3D-point frame to the camera.
    # If the 3D points are in the map frame, T is the map→query-camera transform, and the relative
    # pose to the candidate must be obtained by composing it with the candidate's pose.
    R, _ = cv2.Rodrigues(rvec)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = tvec.flatten()
    
    return is_verified, T, n_inliers
```

### 9.6.2 3D-3D Registration (LiDAR)

For LiDAR candidates, verification shifts to 3D-3D point cloud registration: extracting local 3D descriptors (such as FPFH, FCGF, or learned transformers), estimating an initial rigid transform, and refining alignment coarse-to-fine with ICP. Learned correspondence models such as GeoTransformer (Qin et al., 2022) encode pairwise distances and triplet angles to perform local-to-global registration without an external RANSAC loop, reporting higher recall on low-overlap benchmarks such as 3DLoMatch and large runtime reductions relative to the compared registration stages (the widely cited 100× speedup reflects its specific implementation, hardware, and baseline stage). For classical pipelines without learned correspondences, the function below pairs FPFH feature matching and RANSAC with point-to-plane ICP refinement:

```python
def geometric_verification_lidar(query_cloud, db_cloud, 
                                   voxel_size=0.5, 
                                   distance_threshold=0.5,
                                   fitness_threshold=0.3):
    """
    LiDAR geometric verification: FPFH + RANSAC + ICP.
    """
    import open3d as o3d
    
    # Downsample
    q_down = query_cloud.voxel_down_sample(voxel_size)
    d_down = db_cloud.voxel_down_sample(voxel_size)
    
    # Normal estimation
    q_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size*2, max_nn=30))
    d_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size*2, max_nn=30))
    
    # FPFH feature extraction
    q_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        q_down, o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size*5, max_nn=100))
    d_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        d_down, o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size*5, max_nn=100))
    
    # RANSAC-based registration
    result_ransac = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        q_down, d_down, q_fpfh, d_fpfh,
        mutual_filter=True,
        max_correspondence_distance=distance_threshold * 2,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        ransac_n=3,
        criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999)
    )
    
    # ICP refinement
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

Geometric evidence can also re-rank an initial descriptor candidate list before final acceptance. When trajectory drift remains small, a **spatial proximity prior** favors candidates near the current pose estimate. After local feature matching, **inlier-count re-ranking** promotes candidates supported by stronger geometric consensus. Finally, a **pose consistency check** compares the recovered relative transform against accumulated odometry using Mahalanobis distance under the odometry covariance rather than raw Euclidean discrepancy—since genuine long-loop closures naturally exhibit large accumulated drift, rejecting on raw error magnitude would discard the very loops that fix the most drift.

---

## 9.7 Recent Developments

### 9.7.1 The Rise of Foundation-Model-Based PR

Visual place recognition now spans both domain-trained descriptors and zero-shot foundation-model pipelines, with AnyLoc demonstrating competitive recall across diverse datasets without VPR-specific labels. Current efforts push in two directions: compressing massive backbones such as ViT-G/14 (over 1B parameters) down to embedded-friendly ViT-S or ViT-B models paired with lightweight domain adaptation, and exploiting foundation models' dense patch correspondences directly for geometric re-ranking and relative pose estimation.

### 9.7.2 Semantic Place Recognition

**Semantic place recognition** abstracts away raw pixel appearance in favor of object-level spatial structure—encoding a scene such as "the large tree beside the red building" via segmented layout graphs, object-detection scene graphs, or CLIP-based language queries. Because semantic labels persist across seasonal foliage changes and moderate illumination shifts, the representation offers natural invariance, though it remains bounded by upstream segmentation accuracy and struggles to separate structurally repetitive neighborhoods.

### 9.7.3 4D Radar Place Recognition

As 4D imaging radar gains adoption for all-weather autonomy, radar place recognition is emerging along three fronts: Scan Context variants adapted to sparse radar point clouds, CNN global descriptors trained on Range-Doppler spectra, and radar-camera cross-modal retrieval. Although coarse angular resolution still makes fine place discrimination harder than with LiDAR or cameras, radar PR provides a valuable fallback in rain, snow, and fog as sensor resolution continues to improve.

### 9.7.4 Recent Research (2024-2025)

Recent descriptors combine DINOv2 backbones with sharper aggregation and tighter dimensionality budgets:

- **[SALAD (Izquierdo & Civera, 2024)](https://arxiv.org/abs/2311.15937)** reformulates NetVLAD's feature-to-cluster assignment as an **optimal transport** problem solved via Sinkhorn iterations atop a fine-tuned DINOv2 backbone, outperforming NetVLAD and CosPlace on its evaluated VPR benchmarks.
- **[EffoVPR (Taha et al., 2024)](https://arxiv.org/abs/2405.18065)** targets compact deployment of foundation-model features such as DINOv2, evaluating 128-dimensional descriptors to map the trade-off between retrieval recall and storage or query cost.

### 9.7.5 Technical Lineage Summary

Tracing the major visual, LiDAR, and cross-modal place recognition milestones together highlights how quantization gave way to residual learning, sparse 3D/range representations, and foundation-model embeddings:

```
Visual Place Recognition lineage:

Sivic (2003) Video Google [BoW]
    ↓ quantization → residual
Jégou (2010) VLAD
    ↓ hand-crafted → CNN end-to-end
Arandjelović (2016) NetVLAD [triplet, soft-assignment]
    ↓ triplet → classification
Berton (2022) CosPlace, (2023) EigenPlaces
    ↓ VPR-specific → Foundation Model
Keetha (2023) AnyLoc [DINOv2 + VLAD, zero-shot]
    ↓ assignment optimization + FM fine-tuning
Izquierdo (2024) SALAD [optimal transport + DINOv2]

LiDAR Place Recognition lineage:

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

## Summary

Place recognition retrieves candidate revisits to help bound odometry drift in SLAM. Visual PR evolved from BoW (Video Google) through VLAD and NetVLAD to zero-shot foundation-model pipelines such as AnyLoc and fine-tuned DINOv2 descriptors like SALAD. In parallel, LiDAR PR progressed from handcrafted polar grids (Scan Context) through PointNetVLAD and sparse-convolutional MinkLoc3D to fast range-image architectures like OverlapTransformer.

Extending retrieval across heterogeneous fleets introduces the cross-modal domain gap, motivating shared-embedding training and modality-agnostic representations, while long-term deployments require strategies for illumination, seasonal, and structural change. Geometric verification—via PnP or essential geometry for cameras and ICP-family or learned registration for LiDAR—filters descriptor candidates before graph insertion, though no verification stage eliminates false loop closures under every perceptual ambiguity. Lightweight foundation models, semantic scene graphs, and 4D radar retrieval continue to expand those capabilities.

Once place recognition and geometric verification confirm that the robot has returned to a past keyframe and produce a relative pose estimate, the SLAM backend must register that loop constraint safely, guard against residual outliers, and optimize the trajectory globally. Chapter 10 takes up that loop closure and global optimization pipeline.
