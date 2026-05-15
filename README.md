# Geometric-Semantic Fusion for Autonomous Robotic Affordance

## 1. Project Overview
**Title:** Spatial Resolution Recovery and Multi-Task Geometric Estimation for Robotic Affordance Perception.

**Objective:** To develop a high-precision perception pipeline that identifies "affordances" (actionable regions) on unseen objects. The system bridges the gap between high-level semantic understanding (what an object is) and low-level robotic execution (where and how to grab it) by directly predicting pixel-perfect masks and surface normals.

**Motivation:**
Most robotic manipulation systems rely on pre-defined 3D CAD models. This project aims to enable "Zero-Shot" interaction. When a robot encounters a novel tool or object, it must identify the specific affordance region (e.g., the handle for grasping) and determine the correct approach orientation (surface normal) directly from sensory data, without relying on complex, real-time 3D point cloud reconstruction.

---

## 2. High-Level System Architecture
The system accepts multi-modal sensory data and uses a multi-task neural network to produce a deterministic robotic approach packet. 

### Global Inputs (Sensory Layer)
* **RGB Image:** $448 \times 448 \times 3$ (Color/Texture features).
* **Depth Map:** $448 \times 448 \times 1$ (Used strictly for generating Ground Truth normals, and optionally evaluated as a direct network input in an RGB-D variant).

### Global Outputs (Action Layer)
* **2D Affordance Mask:** A high-resolution segmentation map ($448 \times 448 \times 1$) identifying the actionable part.
* **Dense Surface Normal Map:** A 3D unit vector map ($448 \times 448 \times 3$) representing the orientation ($N_x, N_y, N_z$) of the surface for a collision-free approach.
* **3D Approach Centroid:** The $(X, Y, Z)$ coordinate of the target, derived from the center of the predicted mask and its corresponding depth value.

---

## 3. Detailed Pipeline Stages

### Stage 1: Semantic Feature Extraction (The Sensor)
**Description:** Utilizes a frozen Vision Transformer (DINOv2) to extract global semantic features. Because DINOv2 is trained on a massive diverse dataset, it recognizes functional parts across novel objects.
* **Input:** RGB Image ($448 \times 448 \times 3$).
* **Process:** ViT Patch Embedding and Transformer encoding.
* **Output:** Semantic Feature Tokens ($32 \times 32 \times d$), where $d$ is the embedding dimension.

### Stage 2: Multi-Task Convolutional Refinement (The Learning Core)
**Description:** DINOv2 outputs low-resolution tokens. The CNN Decoder performs "Spatial Resolution Recovery" by fusing high-level semantic tokens with low-level spatial features. It simultaneously predicts the affordance mask and the local geometry. Two input variants (Pure RGB vs. RGB-D) are evaluated for optimal robustness.
* **Input:** Semantic Feature Tokens + Skip Connections.
* **Process:** Transposed Convolutions expanding into a dual-head output.
* **Output:** A combined $448 \times 448 \times 4$ tensor (Channel 1: Affordance Probability, Channels 2-4: Surface Normal Vectors).

### Stage 3: Actionable Inference (The Robotics Core)
**Description:** Extracts the final robotic commands from the network's output tensors, translating pixel predictions into physical coordinates using the camera intrinsics.
* **Process:** Calculate the 2D centroid $(u, v)$ of the predicted affordance mask. Sample the depth $Z$ at $(u, v)$ and use Inverse Perspective Mapping to find $(X, Y, Z)$. Sample the predicted Normal Map at $(u, v)$ to get the approach vector.
* **Output:** The final robotic pose $(X, Y, Z, N_x, N_y, N_z)$.

---

## 4. Datasets & Technical Requirements

**Primary Dataset:** UMD Part Affordance Dataset. Features real-world RGB-D captures from a Kinect sensor, containing 105 kitchen, workshop, and gardening tools. Labeling includes explicit verb-based affordance labels (e.g., 1 = grasp).

**In-The-Wild Test Set:** Custom dataset captured using a modern depth camera in an office environment. This contains completely novel objects (mugs, tools) under varied lighting to strictly evaluate the model's qualitative Sim-to-Real generalization.

**Framework:** PyTorch.

**Evaluation Metrics:** IoU (Intersection over Union) for 2D mask accuracy, and Mean Angular Error (Cosine Similarity) for surface normal vector accuracy.

---

## 5. Workflow

**Phase 1: Data Engineering (The Foundation)**
* **Label Extraction:** Load the `.mat` label files and isolate the `grasp` affordance to create binary target masks.
* **On-the-Fly Normal Generation:** Compute true physical surface normals dynamically during data loading by back-projecting the depth map using precise camera intrinsics and cross-product algebra (bypassing heavy offline storage).
* **Data Loader:** Create a PyTorch Dataset class that outputs: `(RGB_Crop, Depth_Crop, Mask_Crop, Target_Normals)`.

**Phase 2: Neural Perception (The AI Brain)**
* **Input Modality Testing:** Train and evaluate a Pure RGB baseline against an RGB-D variant to determine the optimal balance of geometric accuracy vs. sensor-fault robustness.
* **Multi-Task Decoder:** Build the CNN decoder with two output heads.
* **Training Loop:** Train the network using a weighted loss function (BCE Loss for segmentation + Cosine Similarity Loss for active mask geometry).

**Phase 3: Evaluation & Synthesis (The Result)**
* **Quantitative Benchmarking:** Measure the Mean IoU and Angular Error on an instance-split testing strategy using the UMD dataset.
* **Qualitative Generalization:** Run inference on the custom "In-the-Wild" office dataset to prove the pipeline correctly grounds 3D geometry on completely unseen real-world objects.

---

## 6. Code Structure

```text
robotic_affordance_project/
│
├── data/                       # All datasets live here (Ignored in git)
│   ├── raw_umd/                # The raw extracted UMD dataset (tools/, clutter/)
│   └── custom_test_set/        # Office depth camera captures for final validation
│
├── models/                     # Phase 2: The AI Brain
│   ├── __init__.py
│   ├── backbone.py             # Script to load and freeze DINOv2
│   └── multi_task_cnn.py       # Custom CNN Decoder (Mask + Normal heads)
│
├── utils/                      # Helper scripts and math functions
│   ├── __init__.py
│   ├── dataset.py              # PyTorch Dataset class (On-the-fly math & cropping)
│   ├── geometry.py             # Exact Camera Intrinsics & Inverse Perspective Mapping
│   └── metrics.py              # IoU, Angular Error, and Loss functions
│
├── scripts/                    # Executable pipeline scripts
│   ├── 01_validate_umd.py      # Dataset inspection and extraction sanity check
│   ├── 02_train.py             # The main training loop (Handles RGB vs RGB-D testing)
│   ├── 03_evaluate.py          # Testing the model on unseen UMD tools
│   └── 04_inference.py         # Pass custom office images through the trained system
│
├── README.md                   # Project documentation
└── requirements.txt            # Python dependencies