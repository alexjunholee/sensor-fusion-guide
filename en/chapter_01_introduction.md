# Ch.1 — Why Sensor Fusion?

This guide is an in-depth reference that systematically covers the theory and practice of sensor fusion. It is aimed at readers who are newcomers to robotics but have a working foundation in linear algebra and probability. The first chapter draws a map of the sensor fusion landscape as a whole — why a single sensor is insufficient, in what ways sensors can be combined, and what roles classical methods and deep-learning-based methods each play.

---

## 1.1 Limitations of a Single Sensor

When designing real-world robotic systems — autonomous vehicles, drones, service robots — the first question one encounters is, "Which sensor will we use to perceive the world?" Each sensor is grounded in a physical principle that lets it observe a particular facet of the environment, and that same physical principle imposes its intrinsic limitations.

### Camera Limitations

Cameras provide rich visual information about the environment, yet several fundamental limitations exist.

**Illumination dependence.** A camera is a passive sensor that detects light reflected from objects. Performance therefore degrades sharply under poor lighting conditions such as nighttime, tunnels, or backlight. Auto-exposure mitigates this to some extent, but in scenes that exceed the sensor's dynamic range, saturation or underexposure is unavoidable.

**Scale ambiguity.** A monocular camera projects the 3D world onto a 2D image and in doing so loses depth information. A 1 m object at 2 m distance and a 10 m object at 20 m distance can appear at identical size in the image. This scale ambiguity is the root cause of why monocular visual odometry cannot recover absolute scale. Without stereo cameras or fusion with other sensors, metric distance estimation in meters is fundamentally impossible.

**Textureless environments.** In environments lacking visual features — white walls, long corridors, wide paved roads — feature point extraction and tracking fail. Direct visual odometry methods face the same problem when the photometric gradient is insufficient.

**Motion blur.** During high-speed motion or sharp rotation, images blur, and feature point extraction and matching performance degrade significantly.

### LiDAR Limitations

LiDAR (Light Detection And Ranging) is an active sensor that emits laser pulses and measures the time-of-flight of the reflected wave to provide precise 3D range information. However, it has the following limitations.

**Absence of texture information.** LiDAR captures geometry precisely but provides no color or texture information about objects (some LiDARs report reflection intensity, but this is extremely limited compared with a camera image). As a result, place recognition becomes difficult in structurally similar places — for example, a street with repeated buildings of identical shape.

**Weather and environmental sensitivity.** In rain, fog, snow, dust, and similar conditions, the laser beam scatters, producing large numbers of ghost points or drastically reducing detection range. Measurements are also unstable on black objects or highly reflective surfaces (glass, metal).

**Low resolution and cost.** Mechanical spinning LiDARs have limited vertical resolution (e.g., 16-channel, 32-channel). High-resolution LiDARs cost from several thousand to tens of thousands of dollars. Solid-state LiDARs have recently reduced cost, but they trade off a narrower field of view (FoV).

### IMU Limitations

An inertial measurement unit (IMU) is a sensor that measures acceleration and angular velocity, providing high-frequency (typically 100 Hz–1 kHz) proprioceptive data. Its greatest advantage is independence from the external environment, but it has a critical limitation.

**Drift.** When IMU measurements are integrated to compute velocity and position, sensor bias and noise accumulate over time. Integrating acceleration twice to obtain position causes the error to grow in proportion to $t^2$. Even navigation-grade high-end IMUs exhibit substantial position error within minutes, while the MEMS-grade IMUs commonly used in robotics incur meter-level errors within seconds.

$$\delta \mathbf{p}(t) \approx \frac{1}{2} \mathbf{b}_a \, t^2 + \frac{1}{\sqrt{3}} \sigma_a \, t^{3/2}$$

Here $\mathbf{b}_a$ is the accelerometer bias and $\sigma_a$ is the acceleration noise density. A bias of only $0.01\,\text{m/s}^2$ yields a position error of 0.5 m after 10 seconds.

**Absence of an absolute reference.** The IMU measures only relative changes and provides no information about absolute position or absolute heading. Roll and pitch can be extracted from the gravity direction, but yaw is unobservable without a magnetometer (an observability issue).

### GNSS Limitations

A global navigation satellite system (GNSS) provides absolute global position, but has the following limitations.

**Obstructed environments.** Indoors, in tunnels, and between tall buildings in dense urban settings (urban canyons), satellite signals are blocked or reflected via multipath, producing errors of tens of meters.

**Update rate.** GNSS typically has a low update rate of 1–10 Hz, making it difficult to track fast dynamic motion.

**Precision limits.** The precision of standard single-point positioning is at the meter level. RTK (Real-Time Kinematic) improves this to centimeter level, but requires a base station and takes time for initial convergence.

### Complementarity of Sensor Limitations

Organizing the sensor limitations surveyed above in a table makes it clear that the weaknesses of one sensor can be compensated by the strengths of another.

| Property | Camera | LiDAR | IMU | GNSS |
|------|--------|-------|-----|------|
| Absolute position | ✗ | ✗ | ✗ | ✓ |
| Relative motion (short-term) | ✓ | ✓ | ✓ | ✗ |
| Relative motion (long-term) | △ (drift) | △ (drift) | ✗ (diverges) | ✓ |
| High-frequency motion capture | ✗ | ✗ | ✓ | ✗ |
| Illumination independence | ✗ | ✓ | ✓ | ✓ |
| Weather robustness | △ | ✗ | ✓ | ✓ |
| 3D geometric information | △ (depth ambiguous) | ✓ | ✗ | ✗ |
| Texture / semantic information | ✓ | ✗ | ✗ | ✗ |
| Indoor operation | ✓ | ✓ | ✓ | ✗ |
| Cost | Low | High | Medium~Low | Medium |

The message of this table is clear: **no single sensor is sufficient in all situations.** Sensor fusion is the systematic answer to this problem.

---

## 1.2 Taxonomy of Sensor Fusion

Sensor fusion is the technique of combining information from multiple sensors to achieve accuracy, robustness, and completeness beyond what any individual sensor can deliver. Depending on how the sensors are combined, we can classify sensor fusion into three categories.

### Complementary Fusion

In this form, sensors that measure different physical quantities compensate for each other's shortcomings. Each sensor observes a different subset of the full state, and combining them allows us to estimate the complete state — one that no single sensor alone can observe.

**Representative example: Camera + IMU (Visual-Inertial Odometry)**

- The camera provides relative changes of a 6-DoF pose, but its scale is ambiguous and it fails under high-speed motion.
- The IMU provides high-frequency acceleration/angular velocity, interpolating fast motion between camera frames and recovering scale from the gravity direction.
- The IMU supplies what the camera lacks — scale and high-frequency motion — while the camera supplies what the IMU lacks — drift correction.

**Representative example: GNSS + IMU**

- GNSS provides low-frequency (1–10 Hz) absolute position.
- The IMU provides high-frequency (hundreds of Hz) relative motion.
- When GNSS is lost inside a tunnel, the IMU carries navigation in the short term, and when GNSS returns, the accumulated IMU drift is corrected.

The key point of this type is the **extension of observability**. State variables that are unobservable to one sensor become observable through another sensor's observations.

### Competitive Fusion

In this form, sensors measuring the same physical quantity are deployed redundantly to improve reliability and accuracy.

**Representative example: multi-camera systems**

- Two cameras pointing in the same direction independently track feature points.
- Even if one camera is contaminated (lens fouling, failure), the system continues to operate with the remaining camera.
- Combining the two estimates yields lower variance than either individual estimate.

**Statistical foundation.** Optimally combining two independent observations $z_1 \sim \mathcal{N}(\mu, \sigma_1^2)$ and $z_2 \sim \mathcal{N}(\mu, \sigma_2^2)$ gives:

$$\hat{\mu} = \frac{\sigma_2^2 z_1 + \sigma_1^2 z_2}{\sigma_1^2 + \sigma_2^2}, \quad \sigma_{\text{fused}}^2 = \frac{\sigma_1^2 \sigma_2^2}{\sigma_1^2 + \sigma_2^2}$$

The variance of the fused estimate is always smaller than that of either individual estimate: $\sigma_{\text{fused}}^2 < \min(\sigma_1^2, \sigma_2^2)$. This is exactly the same principle as the update step of a Kalman filter.

**Representative example: multi-LiDAR systems**

In autonomous vehicles, it is common to arrange 4–6 LiDARs around the vehicle to secure a 360° field of view while also using redundant observations in overlapping regions to increase reliability.

### Cooperative Fusion

In this form, raw data from each sensor is combined to generate **new forms of information** that no single sensor could produce.

**Representative example: Stereo Vision**

- The left and right camera images are combined to compute disparity.
- Disparity yields 3D depth information.
- A single camera cannot determine depth, but the cooperation of two cameras generates a new physical quantity (depth).

**Representative example: Camera + LiDAR → colored point cloud**

- LiDAR points are projected onto the camera image to assign a color to each 3D point.
- The resulting colored point cloud carries both the precise geometry of LiDAR and the rich texture of the camera — something no individual sensor can produce.
- Systems such as [R3LIVE (Lin et al., 2022)](https://arxiv.org/abs/2109.07982) perform this in real time.

**Representative example: Radar + Camera → adverse-weather perception**

- Doppler measurements from radar and visual information from cameras are combined to simultaneously recognize the velocity and class of moving objects even in fog or rain.

The three categories are not mutually exclusive. Real-world systems often use all three simultaneously. For example, an autonomous vehicle's sensor fusion system includes cooperative fusion of camera + LiDAR (colored point cloud), complementary fusion of camera + IMU (VIO), and competitive fusion across multiple LiDARs (redundant coverage).

---

## 1.3 Classification by Coupling Level: Loosely vs Tightly vs Ultra-tightly Coupled

Another important axis for classifying sensor fusion is the **level at which sensor data is combined**. This classification directly affects a system's accuracy, complexity, and robustness.

### Loosely Coupled

In this approach, each sensor subsystem independently produces its own estimate, and these outputs are combined at a higher level.

**Structure:**
```
Sensor A → [Subsystem A] → Estimate A ─┐
                                        ├─→ [Fusion] → Final estimate
Sensor B → [Subsystem B] → Estimate B ─┘
```

**Representative example: Independent VO + independent LiDAR odometry → EKF fusion**

- Visual odometry independently outputs a pose.
- LiDAR odometry independently outputs a pose.
- A higher-level EKF combines the two poses to generate the final estimate.

**Advantages:**
- Modularity: each subsystem can be developed, tested, and replaced independently.
- Simplicity: the design of the fusion layer is relatively simple.
- Partial failure handling: even if one subsystem fails, the system can continue to operate using the outputs of the others.

**Disadvantages:**
- Information loss: each subsystem internally summarizes (compresses) its observations before outputting, so raw-observation detail (e.g., the individual uncertainty of each feature point) is lost.
- Ignoring correlations: when independent subsystems use a shared observation (e.g., the same IMU data), the two estimates become correlated, and ignoring this correlation leads to overconfidence. This is known as the "double counting" problem.
- Loss of optimality: because information is summarized, the overall system is not optimal in the information-theoretic sense.

### Tightly Coupled

In this approach, the **raw measurements** of all sensors are fed directly into a single estimation framework (a single estimator).

**Structure:**
```
Sensor A → raw measurement A ─┐
                                ├─→ [Single Estimator] → Final estimate
Sensor B → raw measurement B ─┘
```

**Representative example: [VINS-Mono (Qin et al., 2018)](https://arxiv.org/abs/1708.03852)**

- The camera's raw feature point observations and the IMU's raw acceleration/angular velocity measurements are placed together into a single nonlinear optimization (sliding-window optimization).
- The cost function of the optimization simultaneously minimizes the reprojection error and the IMU preintegration residual.

$$\min_{\mathcal{X}} \left\{ \sum_{(i,j) \in \mathcal{B}} \| \mathbf{r}_{\text{IMU}}(\mathbf{x}_i, \mathbf{x}_j) \|^2_{\mathbf{P}_{ij}} + \sum_{(i,l) \in \mathcal{C}} \| \mathbf{r}_{\text{cam}}(\mathbf{x}_i, \mathbf{f}_l) \|^2_{\mathbf{\Sigma}_l} \right\}$$

Here $\mathbf{r}_{\text{IMU}}$ is the IMU preintegration residual, $\mathbf{r}_{\text{cam}}$ is the visual reprojection residual, and $\mathcal{X}$ is the full state (pose, velocity, bias, landmarks).

**Representative example: [FAST-LIO2 (Xu et al., 2022)](https://arxiv.org/abs/2107.06829)**

- Individual LiDAR points and raw IMU measurements are fed directly into a single iterated error-state Kalman filter.

**Representative example: [LIO-SAM (Shan et al., 2020)](https://arxiv.org/abs/2007.00258)**

- LiDAR feature points, IMU preintegration, and GNSS position observations are jointly optimized in a single factor graph.

**Advantages:**
- Maximum information usage: all information in the raw observations is used, enabling information-theoretically superior estimation.
- Cross-calibration: inter-sensor cross-calibration happens naturally. For instance, camera observations contribute to IMU bias estimation, and IMU data stabilize camera feature point tracking.
- Graceful degradation: even when observations from one sensor drop (e.g., in feature-poor environments), observations from the others continue to support the estimate.

**Disadvantages:**
- Complexity: every sensor's observation model must be implemented in the single estimator, making system design and debugging complex.
- Computational cost: the state vector grows and the number of observations increases, raising computation.
- Sensor coupling: removing or replacing a specific sensor requires modifying the entire estimator.

### Ultra-tightly Coupled

In this approach, fusion occurs at the **signal level** of the sensors. This term is used mainly in GNSS/INS integration.

**Representative example: GNSS/INS ultra-tight integration**

- In typical tightly coupled schemes, the GNSS receiver outputs pseudoranges, which are fed into the navigation filter.
- In ultra-tight, the INS's predicted velocity is fed back into the code/carrier tracking loop inside the GNSS receiver.
- This narrows the receiver's tracking loop bandwidth, increasing noise immunity and maintaining satellite tracking in severely jammed or weak-signal environments.

**Analogous concepts in vision:**

In visual-inertial systems, the counterpart to ultra-tight coupling is using the IMU prediction to constrain the feature point search region of the camera, or to directly correct motion blur using IMU data. Setting the initial value of feature point tracking via the IMU prediction in VINS-Mono comes close to this.

### Comparison of Coupling Levels

| Property | Loosely Coupled | Tightly Coupled | Ultra-tightly Coupled |
|------|----------------|-----------------|----------------------|
| Fusion level | Output | Measurement | Signal |
| Information utilization | Low | High | Highest |
| Implementation complexity | Low | Medium~High | Very high |
| Modularity | High | Low | Very low |
| Partial failure handling | Easy | Requires design | Difficult |
| Representative systems | Independent VO + LO → EKF | VINS-Mono, LIO-SAM, FAST-LIO2, ORB-SLAM3 | GNSS/INS deep integration |

In modern robotics, the **tightly coupled** approach is mainstream. Loosely coupled is simple to implement but loses accuracy due to information loss, while ultra-tightly coupled requires access to specialized hardware and is therefore limited in scope. VINS-Mono, FAST-LIO2, LIO-SAM, and the other most widely used open-source systems today all adopt a tightly coupled architecture. Most recently, [FAST-LIVO2 (Zheng et al., 2024)](https://arxiv.org/abs/2408.14035), which tightly couples LiDAR, inertial, and visual sensors in a single framework, has shown results that substantially surpass existing systems in both accuracy and real-time performance.

---

## 1.4 Classical vs Learning-based: What Deep Learning Changed and What It Did Not

For decades, the field of sensor fusion was dominated by **classical** approaches grounded in probabilistic estimation theory (Kalman filter, factor graph) and geometric methods (epipolar geometry, ICP). Since the mid-2010s, as deep learning revolutionized almost every area of computer vision, learning-based methods have rapidly permeated sensor fusion as well. However, the extent of that penetration differs greatly by area.

### What Deep Learning Changed

**Feature extraction and matching.** Traditionally, handcrafted feature descriptors such as SIFT and ORB were used. [SuperPoint (DeTone et al., 2018)](https://arxiv.org/abs/1712.07629) performs keypoint detection and description jointly via self-supervised learning, greatly improving robustness to illumination and viewpoint changes. [SuperGlue (Sarlin et al., 2020)](https://arxiv.org/abs/1911.11763) revolutionized feature point matching using graph neural networks (GNNs) and attention mechanisms. Most recently, **detector-free** methods such as [LoFTR (Sun et al., 2021)](https://arxiv.org/abs/2104.00680) and [RoMa (Edstedt et al., 2024)](https://arxiv.org/abs/2305.15404) directly find dense correspondences without keypoints, succeeding at matching even in texture-scarce environments.

In this area, learning-based methods clearly surpass traditional methods — one can speak of a **paradigm shift**.

**Place Recognition.** The transition from Bag of Words (DBoW2) to [NetVLAD (Arandjelović et al., 2016)](https://arxiv.org/abs/1511.07247) was dramatic. CNN-based global descriptors enabled place recognition that is far more robust to illumination, seasonal, and viewpoint changes. More recently, [AnyLoc (Keetha et al., 2023)](https://arxiv.org/abs/2308.00688) leverages features from foundation models such as DINOv2 to deliver place recognition that works universally across diverse environments without any additional training.

**Monocular depth estimation.** Estimating depth from a single image is a task that is impossible with classical methods (geometric cues are insufficient). Models such as [Depth Anything (Yang et al., 2024)](https://arxiv.org/abs/2401.10891) have achieved remarkable levels of monocular depth estimation through training on large-scale data. Its successor, [Depth Anything V2 (Yang et al., 2024)](https://arxiv.org/abs/2406.09414), pushed precision further through synthetic-data training and large-scale pseudo-labeling, and [Metric3D v2 (Hu et al., 2024)](https://arxiv.org/abs/2404.15506) enables zero-shot absolute-scale depth estimation, opening the possibility of using metric depth information in sensor fusion without LiDAR. This technology has the potential to replace or complement LiDAR in sensor fusion.

**Map representations.** NeRF and 3D Gaussian Splatting opened a new paradigm for representing scenes with neural networks. NeRF-SLAM, Gaussian Splatting SLAM, and related systems provide photorealistic map representations that go beyond traditional point maps or voxel grids.

**Event cameras.** Event cameras, also called neuromorphic vision sensors, asynchronously detect brightness changes at each pixel, providing extremely high temporal resolution (microsecond level) and wide dynamic range. As a recent [event camera survey (Huang et al., 2024)](https://arxiv.org/abs/2408.13627) summarizes, event-based VIO and SLAM research is developing rapidly, and fusion with conventional frame-based cameras is opening new possibilities in high-speed motion and low-light environments.

### What Deep Learning Did Not Change

**State estimation backends.** Probabilistic estimation frameworks such as Kalman filters and factor graph optimization have not been replaced by deep learning. The reasons are clear:

1. **Rigorous propagation of uncertainty**: Kalman filters and factor graphs mathematically and rigorously track and propagate the uncertainty of observations. Deep learning models struggle to provide comparably calibrated uncertainty.
2. **Guaranteed physical laws**: physical laws are directly encoded into the state transition model (dynamics, kinematics), preventing physically impossible estimates. Learning-based methods cannot guarantee such hard constraints.
3. **Data efficiency**: probabilistic frameworks operate without any data, given only a sensor noise model and a system model. Learning-based methods require large-scale training data.
4. **Generalization**: learning-based odometry (e.g., DeepVO) tends to drop sharply in performance in environments that differ from the training data. Geometric methods are agnostic to the environment.

**LiDAR odometry.** Since [LOAM (Zhang & Singh, 2014)](https://frc.ri.cmu.edu/~zhangji/publications/RSS_2014.pdf), LiDAR odometry has remained overwhelmingly dominated by traditional methods. Point cloud registration methods such as ICP, GICP, and NDT are mathematically well understood, deliver excellent real-time performance, and apply immediately to new environments. Learning-based LiDAR odometry (DeepLO-family) has not yet reached the accuracy and generalization of traditional methods.

**Calibration.** For camera intrinsic calibration, the checkerboard method of [Zhang (2000)](https://doi.org/10.1109/34.888718) remains the standard. Learning-based methods for targetless calibration are under active research, but they have not yet surpassed target-based methods in precision.

### Hybrid Approach: The Current Mainstream

The most successful systems today adopt a **hybrid** structure that combines a learning-based frontend with a classical backend.

```
[Learning-based frontend]         [Classical backend]
 SuperPoint/SuperGlue          Factor Graph / EKF
 Mono Depth Estimation    →    Nonlinear Optimization
 Semantic Segmentation         Kalman Filtering
 Place Recognition              Pose Graph SLAM
```

- At the frontend, deep learning extracts high-level features (feature points, depth, semantic labels) from raw sensor data.
- At the backend, geometric/probabilistic frameworks integrate these features into temporally consistent state estimates.

[DROID-SLAM (Teed & Deng, 2021)](https://arxiv.org/abs/2108.10869) is a good example of this hybrid approach. It uses learned feature extraction and correspondence finding, while performing the final pose estimation with differentiable bundle adjustment.

### Technical Lineage Summary

This guide consistently shows, for each topic, the flow of **traditional methods → what deep learning enabled → where tradition is still needed**. The table below summarizes the technical lineage that runs through the entire guide.

| Area | Classical | Learning-based | Current mainstream |
|------|-----------|---------------|----------|
| Feature matching | SIFT/ORB + BF/FLANN | SuperPoint+SuperGlue → LoFTR → RoMa | Hybrid |
| Visual odometry | Feature-based (ORB) / Direct (DSO) | DROID-SLAM, DPV-SLAM | Tradition leads, learning catching up |
| LiDAR odometry | ICP/LOAM | DeepLO-family | Tradition overwhelmingly leads |
| Place recognition | BoW/VLAD | NetVLAD → AnyLoc | Learning leads |
| Depth estimation | Stereo matching | Mono depth (Depth Anything) | Learning leads (mono) |
| Calibration | Target-based | Targetless + learning | Tradition leads |
| Map representation | Occupancy/TSDF | NeRF / 3DGS | Coexistence |
| State estimation backend | KF / Factor Graph | End-to-end attempts | Tradition overwhelmingly leads |

A noteworthy pattern in this table is that **the closer to perception, the stronger learning is; the closer to inference/estimation, the stronger tradition is**. This pattern provides an important criterion for deciding where to invest deep learning and where to retain traditional methods when designing a sensor fusion system.

So far we have surveyed the classification of sensor fusion and the division of roles between classical and learning-based methods. We now introduce how this guide treats this broad field and in what order the reader might best proceed.

---

## 1.5 Scope and Organization of This Guide

### Relation to robotics-practice

This guide is the advanced companion to the robotics-practice guide. Whereas robotics-practice is an introductory survey covering Spatial AI broadly, this guide is an in-depth reference focused on **sensor fusion, localization, and retrieval**.

- Where robotics-practice introduces EKF/PF in one or two pages, this guide treats each topic at chapter depth — from the derivation of the Kalman filter through ESKF, IMU preintegration, and factor graph optimization.
- Where robotics-practice introduces sensors at a general level, this guide derives each sensor's **noise model and observation equations** in equation form.
- Overlapping introductory content is replaced by references to robotics-practice; this guide covers only the additional depth.

### Guide Organization

This guide follows the organization below:

1. **Ch.1 — Introduction** (this chapter): motivation and classification of sensor fusion
2. **Ch.2 — Sensor Modeling**: observation models and noise characteristics of each sensor (equation-focused)
3. **Ch.3 — Calibration**: calibration theory and practice for various sensor combinations
4. **Ch.4 — State Estimation Theory**: Bayesian filtering, the Kalman filter family, particle filters, factor graphs
5. **Ch.5 — Feature Matching & Correspondence**: technical lineage from SIFT to RoMa
6. **Ch.6 — Visual Odometry & VIO**: internal architecture of VO/VIO systems
7. **Ch.7 — LiDAR Odometry & LIO**: LiDAR-based odometry and LiDAR-inertial fusion
8. **Ch.8 — Multi-Sensor Fusion Architectures**: design principles for multi-sensor integration
9. **Ch.9 — Place Recognition & Retrieval**: from BoW to foundation models
10. **Ch.10 — Loop Closure & Global Optimization**: loop closure and global optimization
11. **Ch.11 — Spatial Representations**: from point maps to neural maps
12. **Ch.12 — Practical Systems & Benchmarks**: real applications and evaluation
13. **Ch.13 — Frontiers**: recent developments and open problems

### Target Audience

The intended audience of this guide is a robotics newcomer with the following background:

- **Linear algebra**: understanding of matrix operations, eigendecomposition, and SVD
- **Probability**: understanding of probability distributions, conditional probability, Bayes' theorem, and the Gaussian distribution
- **Basic optimization**: familiarity with least squares and gradient descent
- **Python**: ability to read and run code built on numpy and scipy

Each chapter of this guide follows the flow **intuition → equations → code/examples**. We first provide an intuitive understanding of a concept, then derive it rigorously in mathematics, and finally implement it in Python code to verify.

---

## 1.6 Cross-Cutting Themes of This Guide

Throughout this guide, the reader will repeatedly encounter several core questions:

1. **"Why was this traditional method important?"** — Understand the fundamental problem each traditional method solved and the elegance of its solution.
2. **"What did deep learning change?"** — See concretely which limitations of traditional methods learning-based methods overcame.
3. **"Where is tradition still needed?"** — Analyze the areas that deep learning has not replaced and why.
4. **"Where is the gap between theory and practice?"** — Examine the differences between papers and real systems, and the engineering issues encountered in practice.

Keeping these questions in mind while reading each chapter will allow the reader to see not only individual algorithms but also the full landscape of sensor fusion as a field.

In the next chapter, we treat the starting point of sensor fusion — **the observation model of each sensor** — mathematically. Precisely expressing in equations how a sensor "sees" the world is the foundation of every fusion algorithm.
