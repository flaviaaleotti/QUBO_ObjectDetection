# QUBO_ObjectDetection

This repo is forked from *"Towards Quantum Approaches for Object Detection using QUBO Formulations"*, a research project (Master's thesis) that reformulates the **Non-Maximum Suppression (NMS)** post-processing step in object detection as a **Quadratic Unconstrained Binary Optimization (QUBO)** problem and solves it using classical, simulated, and D-Wave quantum annealing hardware.

The main branch corresponds to the original code, while this 'multiclass1' branch extends the code capabilities beyond the 'person' class of objects. 

---

## Overview

In object detection pipelines, NMS is the step that removes redundant overlapping bounding boxes after inference. The classical approach uses greedy thresholding; this project investigates whether QUBO-based optimization can achieve superior results.

The system targets **person detection** on the **COCO 2017 validation set**, using a pre-trained Faster R-CNN as the detection backbone and comparing four solver backends across four QUBO formulations.

---

## Problem Formulation

Given a set of $N$ candidate bounding boxes with confidence scores, we define a binary vector $\mathbf{x} \in \{0,1\}^N$ where $x_i = 1$ means box $i$ is kept. The objective is to **maximize**:

$E(\mathbf{x}) = \mathbf{x}^T Q \mathbf{x} = \alpha \sum_i c_i x_i - (1-\alpha) \sum_{i<j} \text{overlap}(i,j)\, x_i x_j$

- The **diagonal** of $Q$ rewards selecting high-confidence boxes (weighted by $\alpha$)
- The **off-diagonal** terms penalize selecting pairs of overlapping boxes (weighted by $1-\alpha$)
- $\alpha \in [0.5, 0.7]$ is a hyperparameter tuned per experiment

### Spatial Penalty Metrics

Three overlap/similarity measures are defined and combined in different QUBO variants:

| Metric | Formula | Description |
|---|---|---|
| **IoU** | $\frac{\|A \cap B\|}{\|A \cup B\|}$ | Standard Intersection over Union |
| **IoM** | $\frac{\|A \cap B\|}{\min(\|A\|, \|B\|)}$ | Intersection over Minimum area |
| **Spatial Feature** | $\frac{\|A \cap B\|}{\sqrt{\|A\| \cdot \|B\|}}$ | Intersection over geometric mean of areas |

### Four QUBO Cases

| Case | Penalty Matrix $P$ | Description |
|---|---|---|
| **Case 1** | $\text{IoU}(i,j)$ | Baseline: IoU-only penalty |
| **Case 2** | $0.7 \cdot \text{IoU}(i,j) + 0.3 \cdot \text{IoM}(i,j)$ | Combined IoU + IoM penalty |
| **Case 3** | $\frac{1-\alpha}{2} \cdot \text{IoU}(i,j) + \frac{1-\alpha}{2} \cdot \text{Spatial}(i,j)$ | IoU + Spatial Feature, equal weights |
| **Case 4** | $\frac{1-\alpha}{2} \cdot (0.7\cdot\text{IoU}+0.3\cdot\text{IoM}) + \frac{1-\alpha}{2} \cdot \text{Spatial}(i,j)$ | Combined IoU+IoM and Spatial Feature |

---

## Evaluation Metrics

| Metric | Description |
|---|---|
| **mAP (std)** | COCO-standard mean Average Precision at IoU=0.50:0.95 |
| **mAP@50** | Average Precision at IoU=0.50 |
| **mAR** | Mean Average Recall (at max 10 and 100 detections) |
| **Precision** | TP / (TP + FP) at IoU=0.5 |
| **Recall** | TP / (TP + FN) at IoU=0.5 |
| **F1** | Harmonic mean of Precision and Recall |
| **MAE** | Mean Absolute Error on predicted vs. ground truth box count |
| **RMSE** | Root Mean Squared Error on predicted vs. ground truth box count |

---

## Solution Approaches

### Gurobi
Formulates the QUBO as a Binary Integer Program and solves it exactly using branch-and-bound. Deterministic but scales poorly with $N$.

### Simulated Annealing (D-Wave `neal`)
Uses simulated annealing library with 1000 reads. Probabilistic; used as a quantum-adjacent baseline. The Q matrix is negaated (it minimizes).

### Quantum Annealing (D-Wave QPU)
Runs on real D-Wave Advantage quantum hardware via `EmbeddingComposite(DWaveSampler())`. The Q matrix is negated (D-Wave minimizes) and converted to triangular form. Experiments use 500, 1000, and 3000 reads. Top-10 solutions per image are saved to JSON.

### Brute Force
Two implementations:
- **CPU**: Gray code enumeration (based on arxiv:2310.19373)
- **GPU (CUDA)**: Numba-based kernel evaluating all $2^N$ energy values in parallel

Used to verify correctness of other solvers on small instances.

---

## System Architecture

```
[COCO 2017 val set]
        |
        v
[Faster R-CNN (ResNet-50 + FPN)]   <-- NMS threshold raised to 0.9 to expose all candidates
        |
        v  (boxes with score >= 0.6, person class only)
[QUBO Matrix Builder]   <-- 4 variants (Cases 1–4)
        |
        v
[QUBO Solver] -----> Gurobi
               |---> Simulated Annealing (neal, 1000 reads)
               |---> Quantum Annealing (D-Wave QPU, 500/1000/3000 reads)
               `---> Brute Force (CPU / CUDA GPU, exact)
        |
        v  (binary solution vector)
[Filtered Bounding Boxes]
        |
        v
[Evaluation: mAP, F1, MAE, RMSE via COCOeval + pycocotools]
```

**Key design choice:** the Faster R-CNN NMS threshold is intentionally raised to **0.9** (instead of the default ~0.5) so that the QUBO optimizer receives all candidate boxes, rather than a pre-filtered set.

---

## Repository Structure

```
.
├── RCNN.py                     # Faster R-CNN inference wrapper
├── IoU.py                      # IoU computation
├── IoM.py                      # IoM computation
├── spatial_feature.py          # Spatial feature computation
├── build_qubo_matrix.py        # QUBO builder — Case 1 (IoU only)
├── build_qubo_matrix2.py       # QUBO builder — Case 2 (IoU + IoM)
├── build_qubo_matrix3.py       # QUBO builder — Case 3 (IoU + Spatial)
├── build_qubo_matrix4.py       # QUBO builder — Case 4 (IoU+IoM + Spatial)
├── qubo_solver.py              # Gurobi solver
├── brute_force.py              # CPU and CUDA GPU brute-force solvers
├── metrics.py                  # Precision, Recall, F1 at IoU=0.5 computation
│
├── best_alpha_gurobi.py        # Alpha hyperparameter using Gurobi
├── best_alpha_sa.py            # Alpha hyperparameter using SA
│
├── main_gurobi.py              # Scalability experiment — Gurobi (6 images)
├── main_sa.py                  # Scalability experiment — SA (6 images)
├── main_qa.py                  # Scalability experiment — QA (6 images)
├── main_qa_291img.py           # Large-scale QA experiment (291 images)
├── main_brute.py               # Brute force experiment (small instances)
│
├── plots_for_6img.py           # Plots for the 6-image experiments (Gurobi, SA, QA)
├── plot_for_291img.py          # Plots for the 291-image experiment (Gurobi, SA, QA)
└── draw_boxes_on_images.py     # Visual box comparison 
│
├── main_QUBO.py  # multiclass solver (Gurobi, SA, QA)
```

---

## Experiments

### Hyperparameter Tuning
`best_alpha_gurobi.py` and `best_alpha_sa.py` sweep $\alpha \in [0.5, 0.7]$ (11 values, step 0.02) over up to 300 COCO images for all 4 QUBO cases. The alpha that maximizes mAP is selected. Discovered optimal range: **$\alpha \approx 0.58$–$0.62$**.

### 6-Image Scalability Test
Six COCO images with **5, 23, 42, 62, 87, and 99** candidate bounding boxes are used to compare all solvers side by side across all 4 QUBO cases. Results include solve times and detection metrics (mAP, F1, MAE).

### 291-Image Large-Scale QA Experiment
300 COCO images are filtered to those with 2–90 candidate boxes (yielding ~291 usable images). Results are stratified into three density groups:

| Group | Box Count Range |
|---|---|
| Low | 2–20 |
| Medium | 21–60 |
| High | 61–90 |

The D-Wave QPU is run with 900 reads. Results are saved incrementally to JSON to handle QPU connection instability.

---

## Technologies and Libraries

| Library | Role |
|---|---|
| **PyTorch** + **torchvision** | Faster R-CNN inference (ResNet-50 + FPN, COCO pre-trained) |
| **OpenCV** (`cv2`) | Image loading, BGR/RGB conversion, visualization |
| **NumPy** | QUBO matrix construction and numerical operations |
| **Gurobi** (`gurobipy`) | Binary Integer Programming solver |
| **D-Wave Ocean SDK** (`dwave.system`, `neal`) | Quantum and simulated annealing |
| **Numba** (`numba.cuda`) | CUDA GPU kernels for brute-force QUBO |
| **pycocotools** | COCO dataset loading and standard mAP evaluation |
| **Matplotlib** | Result plots and metric curves |
| **PrettyTable** | Console-formatted result tables |

**Dataset**: COCO 2017 validation set (`val2017/`, `instances_val2017.json`).

---

## Key Implementation Notes

- **OpenMP conflict workaround**: `KMP_DUPLICATE_LIB_OK=TRUE` is set to resolve conflicts between Gurobi's and PyTorch/NumPy's OpenMP libraries.
- **Q matrix rounding**: we rounded Q values to 6 decimal places for numerical stability.
- **matrix negation**: the Q matrix is negated when using SA and QA solvers.
- **Warm-up runs**: Dummy QUBO problems are solved before real measurements to eliminate cold-start bias.
- **Incremental saving**: for each image, the QA experiment results are saved to avoid data loss on QPU timeout.
- **Brute-force attribution**: The CPU brute-force algorithm is based on [arxiv:2310.19373](https://arxiv.org/abs/2310.19373) and the GPU brute-force on [QUBOBrute](https://github.com/AlexanderNenninger/QUBOBrute) repository.

---

## External Dependencies (not in repository)

- COCO 2017 dataset (env v ariable: `COCO_DATASET`)
- Gurobi license
- D-Wave API credentials and QPU access

---

## Multiclass Extension: User Guide

Before using any script in the repository, the COCO dataset must be downloaded, and its path saved in an env variable
````
export COCO_DATASET=/path/to/coco2017
````


Each class of objects to detect must have a corresponding list of best alpha values to be used when building the QUBO matrices corresponding to the various penalty cases.

The script `main_QUBO.py` already contains the alpha optimal values for the 'person' and 'car' classes.

For any additional class, the user must perform a scan over the images in the COCO validation dataset that contain objects of the target class, in order to evaluate the best alpha value for each penalty case. To do this, the script `best_alpha_gurobi.py` or `best_alpha_sa.py` must be edited to set the variable `TARGET_CATEGORY` equal to the new category name (currently set to `car`).
Running the script will then print the best alpha values on screen.

The obtained alpha values for penalty cases 1, 2, 3 and 4 must be added as list to the dictionary `best_alpha` in `main_QUBO.py`, using the new. category name as key.

Additionally, the `SOLVER` variable must be set equal to `gurobi`, `sa` or `qa` to use the desires QUBO solver, and the `TARGET_CATEGORIES` must be edited to contain all desired category names (lowercase).

Eventually, the script can be run as
````
python3 main_QUBO.py
````
processing times (per image, per penalty case) as well as accuracy metrics (per image, per penalty case) will be printed on screen.


