# Ch.9 — Place Recognition & Retrieval

The odometry/fusion systems covered in Ch.6-8 accumulate drift over time. To correct this drift, the robot must be able to recognize places it has visited before — this is the role of Place Recognition. The techniques covered in this chapter are used directly in the Loop Closure discussion of Ch.10.

> The problem of judging, "Have I seen this place before?"
> We systematically cover the techniques that form the core component of loop closure and the foundation for multi-session SLAM and relocalization.

---

## 9.1 Problem Definition

### 9.1.1 Place Recognition vs Loop Closure Detection

**Place Recognition (PR)** is the **retrieval problem** of judging, "Which place in the database does the current observation match?" It is a standalone problem that is defined even without SLAM.

**Loop Closure Detection** is, within a SLAM system, the detection of "Has the robot revisited a place it visited before?" Place recognition is the core component of loop closure detection, but loop closure is a broader pipeline that also includes **geometric verification** after PR.

```
Loop Closure Detection Pipeline:
┌─────────────┐    ┌────────────────┐    ┌───────────────────┐    ┌─────────────┐
│  Current    │───→│ Place          │───→│ Geometric         │───→│ Pose Graph  │
│  observation│    │ Recognition    │    │ Verification      │    │ Update      │
│  (query)    │    │ (candidate     │    │ (geometric        │    │ (optimize)  │
│             │    │  retrieval)    │    │  verification)    │    │             │
└─────────────┘    └────────────────┘    └───────────────────┘    └─────────────┘
```

**Why PR matters**: In SLAM, without loop closure, drift keeps accumulating. Yet brute-force comparison of the current frame against all past frames is $O(N^2)$ and infeasible. Place recognition is the key technique that reduces this comparison to **sub-linear** time (typically $O(\log N)$ or $O(1)$).

### 9.1.2 Retrieval Pipeline

The general place recognition pipeline follows the information retrieval paradigm:

1. **Encoding**: Encode each observation (image, point cloud, or both) into a fixed-length **global descriptor**.
2. **Indexing**: Store all database descriptors in a searchable index structure (e.g., kd-tree, FAISS).
3. **Retrieval**: Retrieve the database descriptors most similar to the query descriptor.
4. **Re-ranking & Verification**: Geometrically verify the candidates to confirm the final match.

$$
q^* = \arg\min_{q \in \mathcal{D}} d(\mathbf{f}(\text{query}), \mathbf{f}(q))
$$

Here $\mathbf{f}(\cdot)$ is the function that converts an observation into a global descriptor, $d(\cdot, \cdot)$ is a distance function (typically L2 or cosine distance), and $\mathcal{D}$ is the database.

### 9.1.3 Evaluation Metrics

PR system performance is evaluated with the following metrics:

**Recall@N**: The fraction of queries for which the correct match is among the top $N$ candidates. This is the most universal metric.

$$
\text{Recall@N} = \frac{|\{q : \text{top-}N \text{ candidates contain the correct match for query } q\}|}{|\text{total queries}|}
$$

**Recall@1** is especially important because in real-time SLAM we typically can only afford to verify the single most similar candidate.

**Precision-Recall Curve**: Varying a threshold on descriptor similarity traces out the relationship between precision and recall. High precision (minimizing false positives) is particularly important in SLAM — a false-positive loop closure can destructively distort the map.

**Definition of "correct match"**: Typically defined as being within 25 m by GPS distance, though this varies by dataset.

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

Visual Place Recognition is the problem of recognizing a place from images alone. It is the oldest and most actively studied branch of PR.

### 9.2.1 Classical Method: Bag of Words (BoW)

The **Bag of Visual Words** model proposed in **[Video Google (Sivic & Zisserman, 2003)](https://www.robots.ox.ac.uk/~vgg/publications/2003/Sivic03/)** is the origin point of VPR. The key idea is to directly apply the methodology of text retrieval to visual retrieval.

**Pipeline**:

1. **Visual Vocabulary Construction**: Extract local features (SIFT, ORB, etc.) from a large corpus of images, and group these feature descriptors into $K$ clusters with k-means. Each cluster center becomes a "visual word."

2. **Image representation**: Assign features extracted from each image to the nearest visual word (hard assignment) and count the occurrence frequency of each visual word, representing the image as a $K$-dimensional histogram.

3. **TF-IDF weighting**: Apply TF-IDF (Term Frequency – Inverse Document Frequency) weighting borrowed from text retrieval:

$$
w_{i,d} = \underbrace{\frac{n_{i,d}}{n_d}}_{\text{TF}} \cdot \underbrace{\log \frac{N}{N_i}}_{\text{IDF}}
$$

Here $n_{i,d}$ is the count of visual word $i$ in image $d$, $n_d$ is the total number of visual words in image $d$, $N$ is the total number of images in the database, and $N_i$ is the number of images containing visual word $i$.

- **TF**: Higher weight for words that appear frequently in this image → distinctive features of this image
- **IDF**: Higher weight for rare words across the entire database → more discriminative features

4. **Inverted Index**: Pre-build a list of images containing each visual word so that, at query time, rather than scanning the entire database, only images containing the relevant word are retrieved quickly.

5. **Similarity ranking**: Rank by cosine similarity between the TF-IDF vectors of the query image and the database images:

$$
\text{sim}(\mathbf{v}_q, \mathbf{v}_d) = \frac{\mathbf{v}_q \cdot \mathbf{v}_d}{\|\mathbf{v}_q\| \cdot \|\mathbf{v}_d\|}
$$

**DBoW2**: The BoW implementation used in the ORB-SLAM series. It uses a hierarchical k-means tree to scale vocabulary size substantially while keeping quantization fast. It was effectively the de facto standard for loop closure detection in real-time SLAM.

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
        self.db_vectors = {}  # img_id -> tf-idf vector
    
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
        word_ids = self.quantize(descriptors)
        
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
        bow_vector = self.compute_bow_vector(descriptors)
        self.db_vectors[img_id] = bow_vector
        self.N += 1
        
        # Update inverted index
        word_ids = self.quantize(descriptors)
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
        
        scores = {}
        for img_id, db_vector in self.db_vectors.items():
            scores[img_id] = np.dot(q_vector, db_vector)
        
        # Sort by similarity
        sorted_results = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_results[:top_k]
```

**Limitations of BoW**:
- Substantial information loss in quantization (all similar features collapse into the same word).
- Invariance to illumination and viewpoint changes is inherited from the underlying features (SIFT, ORB).
- The choice of vocabulary size $K$ has a large effect on performance.

### 9.2.2 VLAD and Fisher Vector

To overcome the limitations of BoW, aggregation methods that preserve **residual information** rather than simple frequency histograms emerged.

**[VLAD (Vector of Locally Aggregated Descriptors, Jégou et al., 2010)](https://doi.org/10.1109/CVPR.2010.5540039)**:

Whereas BoW records only "which visual words appeared how many times," VLAD records "in which direction and how large the residual is between each visual word and the actual descriptor."

For each cluster $k$, sum the residuals between the descriptors assigned to that cluster and the cluster center:

$$
\mathbf{V}_k = \sum_{i: \text{NN}(\mathbf{x}_i) = k} (\mathbf{x}_i - \mathbf{c}_k)
$$

The final VLAD descriptor is the concatenation of all $\mathbf{V}_k$: $\mathbf{V} = [\mathbf{V}_1^T, \mathbf{V}_2^T, \ldots, \mathbf{V}_K^T]^T$. Its dimensionality is $K \times D$.

**Fisher Vector**: A richer representation than VLAD. Visual words are modeled as a Gaussian Mixture Model (GMM), and first- and second-order statistics for each Gaussian component are normalized by the square root of the Fisher Information Matrix. The dimensionality is $2KD$, twice that of VLAD, but it typically yields higher performance.

### 9.2.3 NetVLAD: The Baseline for Learning-Based VPR

**[NetVLAD (Arandjelović et al., 2016)](https://arxiv.org/abs/1511.07247)** is a seminal work that reformulated VLAD as a **differentiable CNN layer**, enabling end-to-end training.

**Core problem**: In classical VLAD, hard-assigning each descriptor to its nearest cluster is non-differentiable. Backpropagation requires this process to be differentiable.

**Solution: Soft Assignment**:

Replace the hard assignment (1 if $\text{NN}(\mathbf{x}_i) = k$, else 0) with a softmax:

$$
\bar{a}_k(\mathbf{x}_i) = \frac{e^{\mathbf{w}_k^T \mathbf{x}_i + b_k}}{\sum_{k'} e^{\mathbf{w}_{k'}^T \mathbf{x}_i + b_{k'}}}
$$

Expanding the original VLAD's Euclidean-distance-based assignment $e^{-\alpha\|\mathbf{x}_i - \mathbf{c}_k\|^2}$, we can interpret $\mathbf{w}_k = 2\alpha\mathbf{c}_k$ and $b_k = -\alpha\|\mathbf{c}_k\|^2$. Making these parameters learnable lets the cluster centers adapt to the data.

**NetVLAD layer output**:

$$
\mathbf{V}(j, k) = \sum_{i=1}^{N} \bar{a}_k(\mathbf{x}_i) (x_i(j) - c_k(j))
$$

L2-normalizing this $D \times K$ matrix and flattening it into a vector yields the final global descriptor.

**Training strategy**: **Weakly supervised learning** using Google Street View Time Machine data. Images of the same GPS coordinate from different times are used as positive pairs, and images from distant GPS coordinates as negatives. Triplet ranking loss:

$$
\mathcal{L} = \sum_{(q, p^+, p^-)} \max\left(0, m + d(\mathbf{f}(q), \mathbf{f}(p^+)) - d(\mathbf{f}(q), \mathbf{f}(p^-))\right)
$$

Here $m$ is the margin, $p^+$ is a positive image, and $p^-$ is a hard-negative image.

**Performance**: Approximately 84.3% Recall@1 on Pitts250k and about 86.3% Recall@1 on Pitts30k — a substantial improvement over the hand-crafted methods of the time. VGG-16 backbone.

### 9.2.4 AnyLoc: Foundation-Model-Based Universal VPR

**[AnyLoc (Keetha et al., 2023)](https://arxiv.org/abs/2308.00688)** fundamentally shifted the paradigm of VPR. The central question is "Is universal place recognition possible without VPR-specific training?" and the answer is "Yes, using features from a Foundation Model like DINOv2."

**Approach**:

1. **DINOv2 feature extraction**: Extract **dense features** from a middle layer (the 31st layer) of DINOv2 ViT-G14. Use the features of all patches, not the CLS token (which summarizes the whole image into a single vector).

   Why dense features? The CLS token captures the semantic content of the entire image, but it may miss the fine-grained structural differences that distinguish places. Dense features preserve the local information of each patch, enabling more precise place discrimination (on average a 23% improvement).

2. **VLAD aggregation**: Cluster the dense features with k-means to build a visual vocabulary, and produce global descriptors with hard-assignment VLAD. Unlike NetVLAD, this is unsupervised VLAD without any training.

3. **Domain-specific vocabularies**: PCA projections reveal six domains in an unsupervised manner — Urban, Indoor, Aerial, SubT (subterranean), Degraded (adverse conditions), and Underwater — and using domain-specific vocabularies yields up to a further 19% improvement.

**Why does a Foundation Model work?**:

[DINOv2 (Oquab et al., 2023)](https://arxiv.org/abs/2304.07193) was trained with self-supervised learning on 142 million images. In this process, the model learns universal **structural and semantic features** of scenes. Even though it was never trained for a specific place recognition task, these universal features are sufficient for discriminating places.

**Performance**:
- Day-night changes: 5-21% Recall@1 improvement over prior SOTA (MixVPR, CosPlace)
- Seasonal changes: 8-9% improvement
- Opposite viewpoint (180 degrees): 39-49% improvement
- Unstructured environments (underwater, subterranean): up to 4x prior performance
- PCA-Whitening compresses 49K dimensions to 512 (100x compression) while maintaining SOTA performance

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
        # Use the value facet of layer 31 (AnyLoc's key finding)
        image_patch_features.append(patch_tokens)
        all_patch_features.append(patch_tokens)
    
    # Step 2: Build k-means visual vocabulary
    all_features = np.vstack(all_patch_features)  # (N_total_patches, D_feat)
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(all_features)
    centers = kmeans.cluster_centers_  # (K, D_feat)
    
    # Step 3: Compute VLAD descriptor for each image
    vlad_descriptors = []
    D_feat = centers.shape[1]
    
    for patches in image_patch_features:
        # Hard assignment
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

**[EigenPlaces (Berton et al., 2023)](https://arxiv.org/abs/2308.10832)** is the follow-up to [CosPlace (Berton et al., 2022)](https://arxiv.org/abs/2204.02287), integrating PCA-based dimensionality reduction into the training process. Building on CosPlace — which replaced the cumbersome hard-negative mining of triplet loss with classification-based training — EigenPlaces goes further and optimizes the feature-space structure from a PCA perspective.

### 9.2.6 Expanding the Use of Foundation Models

Beyond DINOv2, various Foundation Models are being used for VPR:

- **CLIP**: With text-image correspondence training, it can be used for semantic-level place recognition such as "city street" or "forest trail." However, it lags behind DINOv2 in distinguishing fine-grained structural differences.
- **SAM (Segment Anything)**: Research is underway on using segmentation masks as a structural representation of places.
- **DINOv2 + NetVLAD**: Follow-up work shows that attaching a trained NetVLAD layer to DINOv2 features, instead of AnyLoc's unsupervised VLAD, yields further performance gains.

### 9.2.7 SeqSLAM and Sequence Matching

**[SeqSLAM (Milford & Wyeth, 2012)](https://doi.org/10.1109/ICRA.2012.6224623)** proposes matching **entire image sequences** instead of individual images.

**Key intuition**: Places that are hard to distinguish from a single image (e.g., visually similar residential areas) can have unique patterns across consecutive images (left turn → park → crosswalk).

**Method**:
1. Convert each image into an extremely simplified representation (low-resolution patch or a simple descriptor)
2. Construct a **sequence similarity matrix** between the query sequence and the database sequence
3. Search for diagonal paths within a constrained velocity range over the matrix to find the best sequence match

$$
S_{\text{seq}}(q_s, d_s) = \min_{\text{path}} \sum_{i=0}^{L-1} d(\mathbf{f}(q_{s+i}), \mathbf{f}(d_{s+\delta(i)}))
$$

Here $L$ is the sequence length and $\delta(i)$ is a path function that admits velocity variations.

**Advantages**: Robust under dramatic appearance changes (day → night, summer → winter) because the sequence pattern is preserved.

**Follow-up**: SeqNet (Garg et al., 2021) performs sequence matching with a learning-based approach.

---

## 9.3 LiDAR Place Recognition

This is the problem of recognizing places from 3D LiDAR point clouds. Unlike cameras, it is not affected by illumination changes, but it is vulnerable to seasonal vegetation changes and viewpoint changes.

### 9.3.1 Handcrafted Method: Scan Context

**[Scan Context (Kim & Kim, 2018)](https://doi.org/10.1109/IROS.2018.8593953)** converts a 3D LiDAR scan into a **global descriptor that directly preserves spatial structure** without a histogram. No training is needed, and it is widely adopted as a loop closure module in major systems such as LIO-SAM.

**Descriptor generation**:

1. **Polar partitioning**: Partition the 2D plane viewed from the sensor center into $N_s$ azimuth sectors and $N_r$ range rings, forming an $N_s \times N_r$ grid (typically 60x20).

2. **Max-height encoding**: For each bin, record the **maximum height (max height)** among the 3D points falling into it. This yields the Scan Context matrix $\mathbf{SC} \in \mathbb{R}^{N_s \times N_r}$.

$$
\mathbf{SC}(i, j) = \max_{p \in \text{bin}(i,j)} z(p)
$$

**Why maximum height?** Max height captures protruding structures like buildings, trees, and poles better than mean height does.

3. **Advantage of an egocentric representation**: Because it is expressed in the sensor-centered frame, revisiting the same place from the **opposite direction** merely circularly shifts the rows (sector axis) of the descriptor. This is handled via row-shift matching.

**Search strategy — Ring Key & Sector Key**:

Directly comparing Scan Context matrices against the entire database is inefficient. A two-stage search is used for efficiency:

1. **Ring Key**: For each ring (column) of the SC matrix, take the mean over the sector direction to produce a vector — $\mathbf{k}_r = [\bar{h}_1, \bar{h}_2, \ldots, \bar{h}_{N_r}]$. This vector is used for fast kd-tree search to narrow down candidates.

2. **Sector Key**: For each candidate, try row shifts (sector axis) of the SC matrix to find the best match:

$$
d(\mathbf{SC}_q, \mathbf{SC}_d) = \min_{s \in [0, N_s)} \left\| \mathbf{SC}_q - \text{shift}(\mathbf{SC}_d, s) \right\|_F
$$

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
        
        # Stage 2: Scan Context column-shift matching
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

**Follow-up — Scan Context++**: Adds semantic segmentation, encoding semantic labels (buildings, roads, vegetation, etc.) instead of height. Semantic information is more robust to seasonal changes.

### 9.3.2 M2DP and ESF

Handcrafted LiDAR descriptors beyond Scan Context:

- **M2DP (He et al., 2016)**: Projects the point cloud onto several 2D planes and compresses the density distribution of each projection with SVD to produce a 192-dimensional descriptor. Rotation invariant.
- **ESF (Ensemble of Shape Functions)**: A descriptor combining histograms of pairwise distances, angles, and area ratios among point pairs.

These are weaker than Scan Context in preserving spatial structure, but have advantages in rotation invariance or compactness.

### 9.3.3 Learning-based: PointNetVLAD

**[PointNetVLAD (Uy & Lee, 2018)](https://arxiv.org/abs/1804.03492)** is the first work to apply the NetVLAD idea directly to 3D point clouds.

**Architecture**:
1. Extract local features from the point cloud with a **PointNet** backbone
2. Aggregate local features into a global descriptor with a **NetVLAD layer**
3. Train with lazy triplet loss

$$
\mathcal{L} = \max(0, m + \max_{p^+} d(q, p^+) - \min_{p^-} d(q, p^-))
$$

**Limitations**: PointNet does not sufficiently model inter-point interactions and is inefficient for processing large-scale point clouds.

### 9.3.4 MinkLoc3D

**[MinkLoc3D (Komorowski, 2021)](https://github.com/jac99/MinkLoc3D)** addresses the limitations of PointNetVLAD by using a **Minkowski Convolutional Neural Network** as its backbone. Sparse 3D convolutions effectively capture local point-cloud structure, and GeM (Generalized Mean) pooling produces the global descriptor.

### 9.3.5 OverlapTransformer: Range-Image-Based

**[OverlapTransformer (Ma et al., 2022)](https://arxiv.org/abs/2203.03397)** is an approach that converts LiDAR point clouds into **range images** to leverage 2D image processing pipelines.

**Range Image**: A rotating LiDAR scan is converted into a 2D image of size $(h, w)$. Each pixel value is the range in that direction. $h$ corresponds to the number of laser beams and $w$ to the horizontal resolution.

**Architecture**:
1. Process the range image with a lightweight CNN to extract a feature map
2. Produce a global descriptor with a **NetVLAD** layer
3. Incorporate overall context with a Transformer encoder

**Advantages**: 2D CNN processing is much faster than 3D point cloud processing, and existing image-network architectures can be directly leveraged.

### 9.3.6 BEV-Based Methods

Methods that project point clouds into a Bird's Eye View (BEV) 2D map and perform 2D image-based retrieval:

- **OverlapNet**: Directly predicts the degree of overlap between BEV projection images
- **BEVPlace**: Extracts NetVLAD descriptors from BEV images

---

## 9.4 Cross-Modal Place Recognition

### 9.4.1 Why Cross-Modal PR Is Needed

In real-world robot systems, **the sensor used during mapping may differ from the sensor used during localization**. Examples:
- Localizing with a camera only on a map built with LiDAR
- Place recognition between an autonomous vehicle (LiDAR+Camera) and a delivery robot (Camera only)
- A drone (Camera) localizing itself on a map built by a vehicle (LiDAR)

In these scenarios, **place recognition between LiDAR observations and camera observations** — that is, cross-modal PR — is needed.

### 9.4.2 The Fundamental Difficulty of Cross-Modal PR: Domain Gap

LiDAR point clouds and camera images are fundamentally different representations:

| Property | LiDAR point cloud | Camera image |
|------|---------------------|-------------|
| Data structure | Unstructured 3D point set | Structured 2D grid |
| Information | Geometry | Appearance |
| Illumination dependence | None | Very high |
| Texture | None | Rich |
| Density | Inversely proportional to range | Uniform |

This fundamental difference is called the **domain gap**, and it is the reason that descriptors of the same place observed in different modalities are far apart in descriptor space.

### 9.4.3 (LC)²: LiDAR-Camera Cross-Modal PR

**[(LC)² (Lee et al., 2023)](https://arxiv.org/abs/2304.08660)** proposes a method to map LiDAR point clouds and camera images into a **shared embedding space**.

**Approach**:
1. Convert LiDAR point clouds into range images / BEV images, unifying them as a 2D representation
2. Process camera images and LiDAR projection images with CNNs
3. Train with **contrastive learning** so that LiDAR-Camera pairs from the same place are close in the embedding space and pairs from different places are far apart

$$
\mathcal{L}_{\text{contrastive}} = \sum_{(l, c) \in \mathcal{P}^+} \| \mathbf{f}_L(l) - \mathbf{f}_C(c) \|^2 + \sum_{(l, c) \in \mathcal{P}^-} \max(0, m - \| \mathbf{f}_L(l) - \mathbf{f}_C(c) \|)^2
$$

Here $\mathbf{f}_L$ is the LiDAR encoder, $\mathbf{f}_C$ is the camera encoder, and $\mathcal{P}^+$/$\mathcal{P}^-$ are positive/negative pairs.

### 9.4.4 ModaLink

**ModaLink** aims at a more general cross-modal framework than (LC)², handling various modality combinations such as LiDAR, Camera, and Radar.

### 9.4.5 Modality-Agnostic Descriptor Approach

The ultimate goal is a **modality-agnostic descriptor** — regardless of which sensor is used, the same place produces the same descriptor. Approaches toward this:

- **Knowledge Distillation**: Use descriptors from an information-rich modality (LiDAR+Camera) as teacher and a single modality as student
- **Canonical Representation**: Convert to a modality-neutral representation such as BEV or semantic layout before comparison
- **Foundation-Model-based**: Research exploring whether the features a Foundation Model like DINOv2 extracts from an image lie in a similar space to those extracted from a rendered LiDAR image

---

## 9.5 Multi-Session & Long-Term Place Recognition

### 9.5.1 Challenges of Long-Term VPR

The same place can undergo **dramatic appearance changes over time**:

- **Illumination changes**: Day vs. night, clear vs. overcast
- **Seasonal changes**: Green trees vs. bare branches vs. snow-covered scenery
- **Weather changes**: Clear vs. rain vs. fog
- **Structural changes**: New construction/demolition, parked vehicles, road work

Such changes cause descriptors of the same place to drift over time, degrading PR performance.

### 9.5.2 Strategies for Seasonal/Time-of-Day/Weather Changes

1. **Data Augmentation based**: Include images under diverse conditions during training. NetVLAD's use of Google Street View Time Machine is a canonical example.

2. **Learning Domain-Invariant Features**: Learn features that are invariant to appearance changes. For example, semantic segmentation results (the arrangement of buildings, roads, sky) are invariant to illumination.

3. **Leveraging Foundation Models**: As shown by AnyLoc, DINOv2 features exhibit strong robustness to illumination/seasonal changes. This is because the self-supervised training process learns features invariant to diverse augmentations.

4. **Sequence matching**: The SeqSLAM approach. Even when the appearance of individual images changes, the pattern across an image sequence tends to be preserved.

### 9.5.3 Map Update Strategies

For long-term systems, the map itself must be updated:

- **Experience-Based Map**: Store observations of the same place from multiple time periods and match against the experience most similar to the current conditions. The drawback is that map size grows with time.

- **Adaptive descriptors**: Incrementally update the descriptor of an existing place with new observations, e.g., using an exponential moving average.

- **Change Detection**: Detect structural changes (e.g., building demolition) to invalidate and rebuild the affected map regions.

### 9.5.4 Lifelong Place Recognition

The ultimate goal is **place recognition for a robot operated over its lifetime**. This requires:

- Compressing/managing information so the map does not grow unboundedly
- Graceful forgetting of old observations
- Continuous adaptation to environmental change
- Learning new environments without catastrophic forgetting

This area is still an open research problem and is closely related to continual / incremental learning.

---

## 9.6 Geometric Verification & Re-ranking

Once place recognition has identified candidates, it is necessary to **verify geometrically whether they are truly the same place**. Without this step, false-positive loop closures can occur and destroy the map.

### 9.6.1 PnP + RANSAC (Visual)

For camera-based PR candidates:

1. **Local feature matching** between the query and candidate images (SuperPoint+LightGlue, ORB+BF, etc.)
2. **PnP (Perspective-n-Point)**: Estimate the relative camera pose from 2D-3D correspondences
3. **RANSAC**: Estimate the pose while rejecting outliers

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
    
    # Relative pose
    R, _ = cv2.Rodrigues(rvec)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = tvec.flatten()
    
    return is_verified, T, n_inliers
```

### 9.6.2 3D-3D Registration (LiDAR)

For LiDAR-based PR candidates:

1. Find **3D-3D correspondences** between the two point clouds (FPFH, FCGF, GeoTransformer, etc.)
2. Estimate the rigid transformation via **RANSAC** or a **RANSAC-free** approach like GeoTransformer
3. **Refine with ICP** (coarse-to-fine)

GeoTransformer (Qin et al., 2022) enables robust registration without RANSAC. It is a geometric transformer that encodes pairwise distances and triplet angles and learns features invariant to rigid transformation, remaining robust in low-overlap scenarios. It estimates the transformation directly from superpoint-level correspondences and is 100x faster than RANSAC.

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

Strategies for **re-ranking** the candidate list — originally ordered by descriptor similarity — using geometric information:

1. **Spatial proximity prior**: In SLAM, add a bonus to candidates near the current pose estimate. This is valid only when drift is small.

2. **Inlier-count-based re-ranking**: After feature matching, promote candidates with more inliers.

3. **Pose consistency check**: Verify that the estimated relative pose is consistent with the accumulated odometry pose. Reject as false positive if there is a large discrepancy.

---

## 9.7 Recent Developments

### 9.7.1 The Rise of Foundation-Model-Based PR

**Paradigm shift**: VPR is transitioning from "environment-specific dedicated training" to "universal zero-shot recognition." As AnyLoc showed, the universal features of a Foundation Model achieve SOTA in most environments without any VPR-specific training.

**Future directions**:
- The VPR performance of next-generation FMs beyond DINOv2 (Vision Foundation Model v2, RADIO, etc.)
- **Lightweighting** FM features: Current ViT-G14 has 1B+ parameters, limiting embedded deployment. Combinations of lightweight FMs (ViT-S/B) + domain adaptation are being actively studied
- **Leveraging fine-grained spatial information** from FM features: Directly using patch-level correspondences (dense correspondence) provided by FMs for re-ranking or relative pose estimation

### 9.7.2 Semantic Place Recognition

An approach that recognizes a place by its **semantic structure**:

"The big tree next to the red building" → represents the place by the spatial layout of the building (red) and the tree (big)

**Approaches**:
- Semantic segmentation → semantic layout comparison
- Object detection → object graph (scene graph) comparison
- Open-vocabulary: natural-language-description-based place retrieval with CLIP

**Advantages**: Semantic structure is robust to illumination/seasonal changes. Even if the color of a tree's leaves changes, the semantic label "tree" is preserved.

**Limitations**: Depends on the accuracy of semantic segmentation, and it is difficult to distinguish places with similar semantic structure (e.g., similarly structured residential areas).

### 9.7.3 4D Radar Place Recognition

With the advent of 4D imaging radar, radar-based PR is beginning to be studied:

- Encoding radar point clouds with Scan Context variants
- Producing global descriptors by processing Range-Doppler images with CNNs
- Radar-Camera cross-modal PR

**Advantages**: PR systems that operate in adverse weather. When LiDAR PR and visual PR fail in rain, snow, or fog, radar PR can serve as a backup.

**Current status**: Still in early stages, and discriminability falls short of LiDAR and Visual PR due to radar's low resolution. However, 4D radar resolution is improving rapidly, making this a promising area for the future.

### 9.7.4 Recent Research (2024-2025)

**[SALAD (Izquierdo & Civera, 2024)](https://arxiv.org/abs/2311.15937)**: Redefines NetVLAD's feature-to-cluster assignment as an **optimal transport** problem and fine-tunes DINOv2 as the backbone. The Sinkhorn algorithm is used to optimize soft assignment, achieving SOTA on numerous benchmarks over NetVLAD/CosPlace.

**[EffoVPR (Taha et al., 2024)](https://arxiv.org/abs/2405.18065)**: A framework for efficiently leveraging the features of Foundation Models like DINOv2. It maintains SOTA performance even with descriptors compressed to 128 dimensions, presenting a lightweight VPR suitable for embedded deployment.

### 9.7.5 Technical Lineage Summary

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

## Chapter 9 Summary

Place Recognition is the core component of loop closure that corrects drift in SLAM systems. Visual PR has evolved from BoW (Video Google) → VLAD → NetVLAD → AnyLoc, and universal zero-shot recognition based on Foundation Models (DINOv2) has recently become the paradigm. LiDAR PR has progressed from Scan Context (handcrafted) → PointNetVLAD → MinkLoc3D → OverlapTransformer, with range-image-based methods drawing attention for their efficiency.

Cross-modal PR faces the fundamental difficulty of the domain gap, and shared-embedding-space learning and modality-agnostic descriptors are active research directions. Long-term PR must cope with seasonal/illumination/structural changes, and the robust features of Foundation Models offer a promising solution to this problem.

Geometric verification is the final verification stage for PR candidates, preventing false positives and protecting the integrity of SLAM. PnP+RANSAC (visual) and ICP/GeoTransformer (LiDAR) are the standard methods.

Recent trends include the lightweighting of Foundation-Model-based PR, semantic PR, and 4D radar PR, all of which are active areas of research.

Place Recognition answers the question "Have I seen this place before?", but the process of converting that answer into global consistency for a SLAM system remains. The next chapter covers **Loop Closure and global optimization**, which integrate PR results into the pose graph to correct drift.
