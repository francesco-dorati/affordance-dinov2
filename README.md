# Project Documentation: Geometric-Semantic Fusion for Autonomous Robotic Affordance

## 1. Project Overview
**Title:** Spatial Resolution Recovery and Multi-Task Geometric Estimation for Robotic Affordance Perception.

**Objective:** To develop a high-precision perception pipeline that identifies "affordances" (actionable regions) on unseen objects. The system bridges the gap between high-level semantic understanding (what an object is) and low-level robotic execution (where and how to grab it) by directly predicting pixel-perfect masks and surface normals.

**Motivation:**
Most robotic manipulation systems rely on pre-defined 3D CAD models. This project aims to enable "Zero-Shot" interaction. When a robot encounters a novel tool or object, it must identify the specific affordance region (e.g., the handle for grasping) and determine the correct approach orientation (surface normal) directly from RGB-D sensory data, without relying on complex, real-time 3D point cloud reconstruction.

---

## 2. High-Level System Architecture
The system accepts multi-modal sensory data and uses a multi-task neural network to produce a deterministic robotic approach packet.

### **Global Inputs (Sensory Layer)**
* **RGB Image:** $H \times W \times 3$ (Color/Texture features).
* **Depth Map:** $H \times W \times 1$ (Per-pixel metric distance).

### **Global Outputs (Action Layer)**
* **2D Affordance Mask:** A high-resolution segmentation map ($H \times W \times 1$) identifying the actionable part.
* **Dense Surface Normal Map:** A 3D unit vector map ($H \times W \times 3$) representing the orientation ($N_x, N_y, N_z$) of the surface for a collision-free approach.
* **3D Approach Centroid:** The $(X, Y, Z)$ coordinate of the target, derived from the center of the predicted mask and its corresponding depth value.

---

## 3. Detailed Pipeline Stages

### **Stage 1: Semantic Feature Extraction (The Sensor)**
**Description:**
Utilizes a frozen Vision Transformer (DINOv2) to extract global semantic features. Because DINOv2 is trained on a massive diverse dataset, it recognizes functional parts across novel objects.
* **Input:** RGB Image ($224 \times 224 \times 3$).
* **Process:** ViT Patch Embedding and Transformer encoding.
* **Output:** Semantic Feature Tokens ($14 \times 14 \times d$), where $d$ is the embedding dimension.

### **Stage 2: Multi-Task Convolutional Refinement (The Learning Core)**
**Description:**
DINOv2 outputs low-resolution tokens ($14 \times 14$). The CNN Decoder is the primary custom-trained component. It performs "Spatial Resolution Recovery" by fusing high-level semantic tokens with low-level spatial features from the RGB-D input. Instead of a single output, it simultaneously predicts the affordance mask and the local geometry.
* **Input:** Semantic Feature Tokens + RGB-D Skip Connections.
* **Process:** Transposed Convolutions expanding into a dual-head output.
* **Output:** A combined $224 \times 224 \times 4$ tensor:
    * Channel 1: Affordance Probability $[0, 1]$.
    * Channels 2-4: Surface Normal Vectors $[-1, 1]$.

### **Stage 3: Actionable Inference (The Robotics Core)**
**Description:**
Extracts the final robotic commands from the network's output tensors, translating pixel predictions into physical coordinates using the camera intrinsics.
* **Process:**
    * Calculate the 2D centroid $(u, v)$ of the predicted affordance mask.
    * Sample the depth $Z$ at $(u, v)$ and use Inverse Perspective Mapping to find $(X, Y, Z)$.
    * Sample the predicted Normal Map at $(u, v)$ to get the approach vector.
* **Output:** The final robotic pose $(X, Y, Z, N_x, N_y, N_z)$.

---

## 4. Dataset & Technical Requirements

* **Dataset:** UMD Part Affordance Dataset.
    * **Source:** Real-world RGB-D captures from a Kinect sensor.
    * **Content:** 105 kitchen, workshop, and gardening tools (knives, mugs, hammers, etc.).
    * **Labeling:** Explicit verb-based affordance labels (e.g., `1 = grasp`, `2 = cut`, `3 = scoop`).
* **Framework:** PyTorch.
* **Evaluation Metrics:**
    1.  **IoU (Intersection over Union):** Accuracy of the 2D mask against ground truth.
    2.  **Cosine Similarity Error:** Accuracy of the predicted surface normal vectors against ground-truth geometry.

---

## 5. Workflow

#### Phase 1: Data Engineering (The Foundation)
**Goal:** Prepare the UMD dataset for multi-task training by generating geometric ground truths offline.
* **1.1 Label Extraction:** Load the `.mat` label files and isolate the `grasp` (1) affordance to create binary target masks.
* **1.2 Offline Normal Generation:** Compute depth gradients (using Sobel filters) across the `_depth.png` files to calculate ground-truth surface normals ($N_x, N_y, N_z$) for every image.
* **1.3 Data Loader:** Create a PyTorch Dataset class that outputs: `(RGB_Image, Depth_Map, Target_Mask, Target_Normals)`.

#### Phase 2: Neural Perception (The AI Brain)
**Goal:** Design and train the system to identify actionable parts and their geometry simultaneously.
* **2.1 Semantic Extraction:** Pass the RGB image through the Frozen DINOv2 backbone.
* **2.2 Multi-Task Decoder:** Build the CNN decoder with two output heads (Mask Head and Normal Head).
* **2.3 Training Loop:** Train the network using a combined loss function:
    * **Mask Loss:** Dice Loss or BCE Loss for affordance segmentation.
    * **Geometry Loss:** Cosine Similarity Loss applied only to the normals within the active mask region.

#### Phase 3: Evaluation & Synthesis (The Result)
**Goal:** Prove the streamlined project works and prepare for reporting.
* **3.1 Quantitative Benchmarking:** Measure the Mean IoU for 2D affordances and average Cosine Error for surface normal predictions.
* **3.2 Sim-to-Real Application:** Demonstrate how the output tensor $(X, Y, Z, N_x, N_y, N_z)$ can be passed directly to a robotic path planner (e.g., MoveIt) for execution.
* **3.3 Reporting:** Document the architecture shift, noting how the Multi-Task CNN approach optimized the pipeline by replacing expensive real-time Point Cloud processing.

## 6. Code Structure
```
robotic_affordance_project/
│
├── data/                       # Phase 1: All your datasets live here (DO NOT push to GitHub)
│   ├── raw_umd/                # The raw extracted UMD dataset (tools/, clutter/, etc.)
│   └── processed_umd/          # Where we will save the generated _normal.npy files
│
├── models/                     # Phase 2: The AI Brain
│   ├── __init__.py
│   ├── backbone.py             # Script to load and freeze DINOv2
│   └── multi_task_cnn.py       # Your custom CNN Decoder (Mask + Normal heads)
│
├── utils/                      # Helper scripts and math functions
│   ├── __init__.py
│   ├── dataset.py              # The PyTorch Dataset class (loads RGB, Depth, Mask, Normals)
│   ├── geometry.py             # Depth-to-Normal math, Sobel filters, back-projection
│   ├── losses.py               # Dice Loss (Masks) and Cosine Similarity Loss (Normals)
│   └── metrics.py              # IoU and Angular Error calculations
│
├── scripts/                    # Executable scripts to run the pipeline
│   ├── 01_validate_umd.py      # The inspection script I just gave you
│   ├── 02_generate_normals.py  # The offline script to batch-process the whole dataset
│   ├── 03_train.py             # The main training loop
│   ├── 04_evaluate.py          # Testing the model on unseen tools
│   └── 05_inference.py         # Pass a single new image through the system (Sim-to-Real)
│
├── notebooks/                  # For messy experimentation and plotting
│   └── data_exploration.ipynb  # Jupyter notebooks go here
│
├── README.md                   # The project documentation we just generated
└── requirements.txt            # List of dependencies (torch, opencv-python, scipy, etc.)
```
