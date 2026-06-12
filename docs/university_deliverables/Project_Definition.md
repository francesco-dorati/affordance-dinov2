# Project Definition

**Project Title:** Multi-Task Semantic-Geometric Fusion for Robotic Part Affordance Extraction  
**Team Members:** Francesco Dorati  


## 1. Problem Statement and Motivation
For autonomous robots to interact meaningfully with unconstrained environments, they must not only identify objects but understand *how* to manipulate them. Grasping unknown tools presents a dual challenge: the robot must identify the semantic functional region (e.g., the handle of a knife, mask estimation) and determine the physical 3D orientation for the gripper's approach (surface normal estimation). 

Traditional robotics rely heavily on depth sensors, which often fail on reflective or dark surfaces. This project addresses this limitation by developing a deep learning model that leverages rich semantic priors to predict both 2D affordance masks and 3D surface geometry, creating a robust, sensor-fault-tolerant perception system for robotic grasping.

## 2. Proposed CV Techniques & Methods
The proposed architecture employs a highly efficient, multi-task encoder-decoder design incorporating modern Vision Transformers (ViT):

* **Semantic Backbone (Encoder):** Meta's pre-trained DINOv2 (ViT-Small). The backbone will remain frozen to act as a robust semantic feature extractor. DINOv2 slices 448x448 inputs into 14x14 patches, generating highly descriptive spatial embeddings without requiring massive local compute for fine-tuning.
* **Spatial Decoder:** A custom Multi-Task Convolutional Feature Pyramid. The decoder upsamples the DINOv2 grid back to its original spatial resolution using transposed convolutions and skip-connections.
* **Multi-Task Heads:** 
   - *Head A (Affordance):* Uses a Sigmoid activation to predict a binary mask [0,1] identifying graspable regions (handles). Evaluated via Binary Cross Entropy (BCE) Loss.
   - *Head B (Geometry):* Regresses the 3D surface normal vector (Nx, Ny, Nz) for every pixel, followed by L2 normalization. Evaluated via Cosine Similarity Loss.

## 3. Dataset Design & Test Strategy

### Dataset(s) to be Used
The primary data source is the **UMD Part Affordance Dataset**, containing approximately 28,000 RGB-D images of tools and clutter. 
* **Data Processing:** Images are dynamically center-cropped to 448x448 (optimized for DINOv2's patch resolution).
* **Ground Truth Generation:** Affordance masks are derived from dataset annotations mapping to "grasp" and "wrap-grasp". Ground truth surface normals are computed dynamically (on-the-fly) from the raw depth maps using camera intrinsics, eliminating the need to store massive precomputed normal arrays.
* **Custom Test Set:** A small, supplementary dataset of highly diverse objects (mugs, tools) will be captured using a distinct office depth camera to test Sim-to-Real generalization.

### Test Design
To prove the model has learned generalized geometry rather than simply memorizing specific tools, the network will be evaluated on two fronts:
1.	**Quantitative (Instance Split)**: The UMD dataset will be split by object instance (e.g., train on knife_01, test on knife_03).
   * **Mask Metrics:** Intersection over Union (IoU) and Pixel Accuracy.
   * **Normal Metrics:** Mean Angular Error (MAE) in degrees between the predicted vector and the ground truth vector.
2. **Qualitative (Real-World Generalization)**: Passing the custom office dataset through the trained model to visually verify handle isolation and geometric grounding on completely unseen hardware and environments.

## 4. Project Pipeline Design

| Phase | Tasks & Components |
| :--- | :--- |
| **Phase 1: Data Engineering** | Parse UMD dataset labels. Implement dynamic 448x448 center cropping. Develop robust mathematical functions to compute ground-truth surface normals from raw depth maps using exact camera intrinsics. |
| **Phase 2: Architecture** | Integrate DINOv2 backbone. Develop the Multi-Task Decoder architecture. Ensure spatial dimensions correctly align from the encoded features back to 448x448 masks. |
| **50% Milestone** | Data pipeline functional, DINOv2 integration complete, and a preliminary end-to-end baseline training loop is capable of running for at least one epoch. |
| **Phase 3: Training** | Deploy to cloud compute (Google Colab). Execute the primary RGB-only training loop with weighted multi-task loss. |
| **Phase 4: Evaluation** | Run instance-split testing. Generate qualitative visualizations on the custom office dataset. Compile final quantitative results. |

## 5. Expected Outcomes and Deliverables
1. A functional PyTorch training and inference pipeline, version-controlled via GitHub.
2. A fully trained set of model weights capable of extracting grasp affordances and surface normals.
3. A comprehensive final technical report documenting the architecture and quantitative validation results.

---

## 6. Iterative Development Strategy
Based on architectural reviews, the project will follow an iterative deployment strategy regarding input modalities:
- **Iteration 1 (Pure RGB Inference):** The primary objective is to train the network using purely RGB inputs, utilizing Depth strictly offline to generate ground-truth normals. This preserves DINOv2's 3-channel pre-trained weights and creates a highly robust, sensor-fault-tolerant system.
- **Iteration 2 (RGB-D Fallback):** If the geometric loss plateaus indicating that Pure RGB lacks sufficient spatial context for normal reconstruction, the architecture will be expanded to inject raw Depth maps directly into the CNN Decoder via skip connections.
