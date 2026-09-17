# IMPORT + INITIALIZATION
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import numpy as np
import os
import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
import time
import sys
from prettytable import PrettyTable
import matplotlib.pyplot as plt

# SOLUTION TO KERNEL CRASH: ignore conflicts between OpenMP libraries (Gurobi vs. PyTorch/NumPy)
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import build_qubo_matrix 
import build_qubo_matrix2
import build_qubo_matrix3
import build_qubo_matrix4
import metrics
import RCNN
import logging

# Set solver for QUBO ['gurobi', 'sa', 'qa']
SOLVER = 'qa'
# full names to use  for result printout
SOLVER_STRING = {'gurobi' : 'Gurobi',
                 'sa'     : 'Simulated Annealing',
                 'qa'     : 'Quantum Annealing'}

if SOLVER == 'gurobi':
    import qubo_solver
elif SOLVER == 'sa':
    import neal
    # Simulated Annealing initialization
    sa_sampler = neal.SimulatedAnnealingSampler()
elif SOLVER == 'qa':
    from dwave.system import DWaveSampler, EmbeddingComposite
    import json
    # SAMPLER INITIALIZATION
    # Default quantum sampler
    sampler = EmbeddingComposite(DWaveSampler()) 
    qa_num_reads = 500
else:
    raise SystemExit("Unidentified QUBO solver. Please use a solver in ['gurobi', 'sa', 'qa']")

# silenced torchvision/pytorch
logging.getLogger("torchvision").setLevel(logging.ERROR)
logging.getLogger("torch").setLevel(logging.ERROR)

# path to the file containing the ground truths for each image (called with an ID)
instances_file = os.path.join(os.environ["COCO_DATASET"], "annotations/instances_val2017.json")
coco = COCO(instances_file) # initialization

# Set target categories
TARGET_CATEGORIES = ['person', 'car']
TARGET_CATEGORIES_IDX = coco.getCatIds(catNms=TARGET_CATEGORIES)

# ========================================================
# HELPER FUNCTIONS
# ========================================================

def process_solution(best_sample, boxes, scores, image_id, category_id):
    """Extracts bounding boxes from best solution"""
    # convert best sample in NumPy array
    N = len(boxes)
    QUBO_sol = np.zeros(N, dtype=int)
    
    for idx, val in best_sample.items():
        QUBO_sol[idx] = val

    kept_indices = np.where(QUBO_sol == 1)[0]
    image_predictions = []
    
    if len(kept_indices) > 0:
        kept_boxes = boxes[kept_indices]
        kept_scores = scores[kept_indices]

        for k in range(len(kept_boxes)):
            prediction = {
                "image_id": int(image_id),
                "category_id": int(category_id),
                "bbox": kept_boxes[k].tolist(),
                "score": float(kept_scores[k])
            }
            image_predictions.append(prediction)
            
    return image_predictions, QUBO_sol

def evaluate_metrics(preds, cat_ids, sub_ids):
    """Runs COCOeval + compute_metrics for one case/image, given a list
    of category ids to evaluate (single-element for per-class, full list for global)."""
    if len(preds) == 0:
        return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    original_stdout = sys.stdout

    sys.stdout = open(os.devnull, 'w')
    coco_dt = coco.loadRes(preds)
    coco_eval = COCOeval(coco, coco_dt, 'bbox')
    coco_eval.params.imgIds = sub_ids
    coco_eval.params.catIds = cat_ids
    
    try:
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        mAP_std = coco_eval.stats[0]
        mAP_50 = coco_eval.stats[1]
        mAR_10 = coco_eval.stats[7]
        mAR_100 = coco_eval.stats[8]
    except Exception:
        mAP_std = mAP_50 = mAR_10 = mAR_100 = 0.0
    finally:
        sys.stdout = original_stdout

    precision, recall, f1 = metrics.compute_metrics(coco, preds, sub_ids, cat_ID=cat_ids)

    return (precision, recall, f1, mAP_std, mAP_50, mAR_10, mAR_100)

# SIMULATED ANNELAING
def conversion_Q_sa(Q):
    """Converts the QUBO matrix in a dictionary"""
    Q_annealing = {}
    N = len(Q)
    for j in range(N):
        for k in range(N):
            if Q[j, k] != 0:
                Q_annealing[(j, k)] = -Q[j, k]

    return Q_annealing

# QUANTUM ANNEALING
def conversion_Q_qa(Q):
    """Converts the QUBO matrix into triangular minimization form for D-Wave"""
    Q_dict = {}
    n = len(Q)

    for i in range(n):
        # Diagonal terms
        if Q[i, i] != 0:
            Q_dict[(i, i)] = -Q[i, i]
        
        for j in range(i + 1, n):
            val = -2 * Q[i, j]
            if val != 0: 
                Q_dict[(i, j)] = val

    return Q_dict

def extract_top_10(sampleset, N):
    """Extracts the top 10 solutions found by Quantum Annealer"""
    top_10_list = []
    limit = min(10, len(sampleset))
    
    for i, datum in enumerate(sampleset.data(['sample', 'energy', 'num_occurrences'])):
        if i >= limit:
            break
            
        sol_vector = [0] * N
        for idx, val in datum.sample.items():
            sol_vector[idx] = int(val)
            
        top_10_list.append({
            "rank": i + 1,
            "vector": sol_vector,
            "energy": float(datum.energy),
            "occurrences": int(datum.num_occurrences)
        })
        
    return top_10_list

# ========================================================
# list for scalability (6 images)
image_IDs = [532481, 458755, 147740, 57597]#, 172946, 214539] # 5, 23, 42, 62, 87, 99 boxes
# ========================================================

# DEVICE
if torch.cuda.is_available():
    device = torch.device("cuda") # NVIDIA GPU
elif torch.backends.mps.is_available():
    device = torch.device("mps") # Apple Silicon GPU
else:
    device = torch.device("cpu")
print(f"Using {device}")

# LOADING MODEL
# COCO weights
weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT  # DEFAULT means that it must use skills already developed with COCO's train 
model = fasterrcnn_resnet50_fpn(weights=weights)

# NMS parameters (let's keep all boxes for QUBO problem)
model.roi_heads.nms_thresh = 0.9  # raise the threshold to 0.9 to keep overlaps
model.roi_heads.score_thresh = 0.6 # minimum confidence score: 60%

model.to(device)
model.eval() # evaluation mode

print("INITIALIZATION COMPLETED")

if SOLVER == 'sa':
    # ========================================================
    # SIMULATED ANNEALING WARM-UP (DUMMY RUN)
    # ========================================================
    print("\n--- SIMULATED ANNEALING WARM-UP ---")
    Q_dummy = {(0, 0): 1.0, (1, 1): 1.0, (0, 1): -2.0}
    _ = sa_sampler.sample_qubo(Q_dummy, num_reads=10)
    print("Warm-up completed. Cold start avoided.")
elif SOLVER == 'gurobi':
    # ========================================================
    # GUROBI WARM-UP (DUMMY RUN)
    # ========================================================
    Q = np.array([[1.0, -2.0], [-2.0, 1.0]])
    _ = qubo_solver.qubo(Q)

# BOUNDING BOX ACQUISITION WITH GPU
# data list initialization
gpu_data = []
valid_count = 0

for i, img_id in enumerate(image_IDs):
    
    # load image data
    img_info = coco.loadImgs(img_id)[0]
    file_name = img_info['file_name']

    # BOUNDING BOXES
    image_path = os.path.join(os.environ["COCO_DATASET"], "val2017", file_name)

    t_rcnn_start = time.perf_counter()
    raw_boxes, scores, _, labels = RCNN.faster_rcnn(image_path, model, device, TARGET_CATEGORIES_IDX)
    
    if device.type == 'cuda':
        torch.cuda.synchronize()
    elif device.type == 'mps':
        torch.mps.synchronize()
    t_rcnn_end = time.perf_counter()

    valid_count += 1

    boxes_xywh  = {i:[] for i in TARGET_CATEGORIES_IDX}
    scores_dict = {i:[] for i in TARGET_CATEGORIES_IDX}

    for box in range(len(raw_boxes)):
        x1, y1, x2, y2 = raw_boxes[box]
        w = x2 - x1  
        h = y2 - y1  
        boxes_xywh[labels[box]].append([x1, y1, w, h])
        scores_dict[labels[box]].append(scores[box])
    
    print(f"[{valid_count}] Id imm: {img_id} | Nome imm: {file_name} | ", end="")
    for i in range(len(TARGET_CATEGORIES_IDX)):
        print(f"Box trovate per '{TARGET_CATEGORIES[i]}': {len(boxes_xywh[TARGET_CATEGORIES_IDX[i]])} | ", end="")
    print(f"Tempo RCNN: {t_rcnn_end - t_rcnn_start:.3f}s")

    gpu_data.append({
        'image_id': img_id,
        'boxes': {cat: np.array(boxes_xywh[cat]) for cat in boxes_xywh},
        'scores': {cat: np.array(scores_dict[cat]) for cat in scores_dict},
        'file_name': file_name
    })

# sort gpu_data in order of increasing number of boxes detected (global number across all categories)
gpu_data.sort(key=lambda x: sum(len(v) for v in x['boxes'].values()))

# Ground truths extraction
ground_truths = {}
for i, data in enumerate(gpu_data):

    annIds = coco.getAnnIds(imgIds=data['image_id'], catIds=TARGET_CATEGORIES_IDX)
    anns = coco.loadAnns(annIds) 
    
    gt_boxes = []
    gt_labels = []
    for ann in anns:
        gt_boxes.append(ann['bbox']) 
        gt_labels.append(ann['category_id']) 
    
    ground_truths[data['image_id']] = {
        'boxes': np.array(gt_boxes),
        'labels': np.array(gt_labels),
        'file_name': data['file_name']
    }
print("GROUND TRUTHS AND PREDICTIONS LOADING COMPLETED")

# best alpha computed using best_alpha_gurobi.py
best_alpha = {'person': [0.58, 0.60, 0.60, 0.62],
              'car'   : [0.62, 0.58, 0.58, 0.62]}

# lists for times
times = {cn : [[] for i in range(4)] for cn in TARGET_CATEGORIES}

# lists for predictions
detected_obj = {cn : [[] for i in range(4)] for cn in TARGET_CATEGORIES}

# lists for pred_count and gt_count for MAE
n_detected_obj = {cn : [[] for i in range(4)] for cn in TARGET_CATEGORIES}

# list of top ten results
top_10_list = {cn : [[] for i in range(4)] for cn in TARGET_CATEGORIES}

# we analyze every image only once per category
for catID in TARGET_CATEGORIES_IDX:

    cat_name = coco.loadCats(catID)[0]['name']
    print(f"\n=== CATEGORY: '{cat_name}' ===")

    for i, data in enumerate(gpu_data):
        
        RCNN_boxes = data['boxes'][catID]
        RCNN_scores = data['scores'][catID]
        image_id = data['image_id']

        # GT of this img
        mask = ground_truths[image_id]['labels'] == catID # mask out results from other categories
        gt_count = len(ground_truths[image_id]['boxes'][mask])

        RCNN_boxes = RCNN_boxes.astype(np.float64)
        RCNN_scores = RCNN_scores.astype(np.float64)
        N = len(RCNN_boxes) 
        if N == 0:
            # no bounding boxes for this category, append null results and skip QUBO solving
            for case in range(4):
                n_detected_obj[cat_name][case].append((0, gt_count))
                times[cat_name][case].append(0.0)
            continue

        print(f"\n--- Image ID {image_id} ({N} predicted box for '{cat_name}') ---")

        # loop over penalty cases
        for case in range(1,5):
            # build Q matrix
            if case == 1:
                # --- CASE 1 (IoU) ---
                L, P = build_qubo_matrix.qubo_matrices(RCNN_boxes, RCNN_scores)
                Q = best_alpha[cat_name][0] * L - (1 - best_alpha[cat_name][0]) * P
            elif case == 2:
                # --- CASE 2 (IoU + IoM) ---
                L, P = build_qubo_matrix2.qubo_matrices(RCNN_boxes, RCNN_scores)
                Q = best_alpha[cat_name][1] * L - (1 - best_alpha[cat_name][1]) * P
            elif case == 3:
                # --- CASE 3 (IoU + Sp) ---
                L, P1, P2 = build_qubo_matrix3.qubo_matrices(RCNN_boxes, RCNN_scores)
                beta = (1 - best_alpha[cat_name][2]) / 2
                Q = best_alpha[cat_name][2] * L - beta * P1 - beta * P2
            elif case == 4:
                # --- CASE 4 (IoU+IoM + Sp) ---
                L, P1, P2 = build_qubo_matrix4.qubo_matrices(RCNN_boxes, RCNN_scores)
                beta = (1 - best_alpha[cat_name][3]) / 2
                Q = best_alpha[cat_name][3] * L - beta * P1 - beta * P2

            Q = np.round(Q, decimals=6)
            if SOLVER == 'sa':
                Q = conversion_Q_sa(Q)
            elif SOLVER == 'qa':
                Q = conversion_Q_qa(Q)

            t_start = time.perf_counter()

            if SOLVER == 'gurobi':
                # solve QUBO
                sol = qubo_solver.qubo(Q)
                duration = time.perf_counter() - t_start

                # find out indices of boxes selected by final QUBO solution
                kept_indices = np.where(np.array(sol) == 1)[0]

                # build tuple (n°_boxes_QUBO_solution, n°_boxes_GT) and append it to corresponding list of n_detected_obj
                n_detected_obj[cat_name][case-1].append((len(kept_indices), gt_count))
            
                # list of boxes in solution
                image_predictions = []
                if len(kept_indices) > 0:
                    kept_boxes = RCNN_boxes[kept_indices]
                    kept_scores = RCNN_scores[kept_indices]
                    # one dict per box
                    for k in range(len(kept_boxes)):
                        image_predictions.append({
                            "image_id": int(image_id),
                            "category_id": catID, 
                            "bbox": kept_boxes[k].tolist(), 
                            "score": float(kept_scores[k])
                        })

                # append the selected boxes to corresponding list
                detected_obj[cat_name][case-1].extend(image_predictions)

                print(f"Case {case} Sol: {sol.tolist()}")

            elif SOLVER == 'sa':
                # solve QUBO
                sampleset = sa_sampler.sample_qubo(Q, num_reads=1000)
                duration = time.perf_counter() - t_start

                # retain best solution among sa reads
                best_sample = sampleset.first.sample
                preds, sol_sa = process_solution(best_sample, RCNN_boxes, RCNN_scores, image_id, catID)

                # append the selected boxes to corresponding list
                detected_obj[cat_name][case-1].extend(preds)

                # build tuple (n°_boxes_QUBO_solution, n°_boxes_GT) and append it to corresponding list of n_detected_obj
                n_detected_obj[cat_name][case-1].append((len(preds), gt_count))
            
                print(f"Case {case} Sol: {sol_sa.tolist()}")

            else:
                # solve QUBO
                sampleset = sampler.sample_qubo(Q, num_reads=qa_num_reads)
                duration = sampleset.info['timing']['qpu_access_time'] / 1000000.0

                # extract top 10 results and append them to corresponding list
                top_10 = extract_top_10(sampleset, N)
                top_10_list[cat_name][case-1].append({"image_id": int(image_id), "solutions": top_10})

                # retain best solution among qa reads
                try:
                    best_solution = sampleset.first.sample
                except ValueError:
                    best_solution = {}
                preds, sol_qa = process_solution(best_solution, RCNN_boxes, RCNN_scores, image_id, catID)

                # append the selected boxes to corresponding list
                detected_obj[cat_name][case-1].extend(preds)

                # build tuple (n°_boxes_QUBO_solution, n°_boxes_GT) and append it to corresponding list of n_detected_obj
                n_detected_obj[cat_name][case-1].append((len(preds), gt_count))
            
                # chain break best sol
                best_cbf = sampleset.record['chain_break_fraction'][0]
            
                # mean chain break
                mean_cbf = np.average(sampleset.record['chain_break_fraction'], weights=sampleset.record['num_occurrences'])
            
                print(f"Img ID {image_id} Case {case} - CBF Best Sol: {best_cbf * 100:.2f}% | Mean CBF (500 reads): {mean_cbf * 100:.2f}%")

            times[cat_name][case-1].append(duration)

        print(f"'{cat_name}': Processed {i + 1} / {len(gpu_data)} images...")

    if SOLVER == 'qa':
        # ========================================================
        # SAVING JSON RESULTS
        # ========================================================
        output_dir = f"qa_top10_results_6imm_500reads_{cat_name}"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        for case in range(1,5):
            with open(f"{output_dir}/top10_case_{case}.json", 'w') as f: json.dump(top_10_list[cat_name][case-1], f, indent=4)
        
        print(f"JSON files saved to {output_dir}\n\n")


# ================================================
# COMPUTE METRICS AND PRINT TABLES
# ================================================

# Prepare data
img_ids = [data['image_id'] for data in gpu_data]

# FINAL SUMMARY TABLE (TIME PER IMAGE)
table_time = PrettyTable()
table_time.title = f"{SOLVER_STRING[SOLVER]} - Times per Image"
table_time.field_names = ["Image ID", "Num Boxes", "Case 1 (s)", "Case 2 (s)", "Case 3 (s)", "Case 4 (s)"]

# compute and print global processing times per image (sum on all object categories)
global_box_counts = [sum(len(data['boxes'][cat]) for cat in TARGET_CATEGORIES_IDX) for data in gpu_data]

for i in range(len(img_ids)):
    table_time.add_row([
        img_ids[i], 
        global_box_counts[i],
        f"{sum(times[cat_name][0][i] for cat_name in TARGET_CATEGORIES):.4f}", 
        f"{sum(times[cat_name][1][i] for cat_name in TARGET_CATEGORIES):.4f}", 
        f"{sum(times[cat_name][2][i] for cat_name in TARGET_CATEGORIES):.4f}",
        f"{sum(times[cat_name][3][i] for cat_name in TARGET_CATEGORIES):.4f}"
    ])
table_time.add_divider()

# print processing times per category per image
for cat in TARGET_CATEGORIES_IDX:
    cat_name = coco.loadCats(cat)[0]['name']
    table_time.add_row([cat_name] + ["" for t in range(len(table_time.field_names) - 1)], divider=True)

    box_counts = [len(data['boxes'][cat]) for data in gpu_data]

    for i in range(len(img_ids)):
        table_time.add_row([
            img_ids[i], 
            box_counts[i],
            f"{times[cat_name][0][i]:.4f}", 
            f"{times[cat_name][1][i]:.4f}", 
            f"{times[cat_name][2][i]:.4f}", 
            f"{times[cat_name][3][i]:.4f}"
        ])
    table_time.add_divider()

print("\n")
print(table_time)
print("\n")

# FINAL SUMMARY TABLE (METRICS PER IMAGE)

'''dictionary to store the metrics (global and per category) per image and per penalty case

        metrics_dict[category][img_ID][penalty_case] = (precision, recall, f1, mAP_std, mAP_50, mAR_10, mAR_100, mae, rmse)

    - category = [str] 'global' or one element of TARGET_CATEGORIES
    - img_ID = [int] ID of the image
    - penalty_case = [int] 1, 2, 3 or 4
'''
metrics_dict = {'global' : {i: {} for i in img_ids}}
for cat in TARGET_CATEGORIES:
    metrics_dict[cat] = {i: {} for i in img_ids}

# ----------------------------------------------------------------------------------------
# Compute metrics (global and per category, per image, per case) and store in metrics_dict
# ----------------------------------------------------------------------------------------

for i in range(len(img_ids)):
    img_id = img_ids[i]

    for case in range(1, 5):
        # pool predictions and MAE across both categories, for this image+case only
        preds = []
        n_obj = 0
        n_GT = 0
        for c in range(len(TARGET_CATEGORIES)):
            preds_local = [p for p in detected_obj[TARGET_CATEGORIES[c]][case-1] if p['image_id'] == img_id]
            preds.extend(preds_local)
            p_c, g_c = n_detected_obj[TARGET_CATEGORIES[c]][case-1][i]
            n_obj += p_c
            n_GT += g_c
            mae = rmse = abs(p_c - g_c)
            precision, recall, f1, mAP_std, mAP_50, mAR_10, mAR_100 = evaluate_metrics(preds_local, [TARGET_CATEGORIES_IDX[c]], [img_id])
            metrics_dict[TARGET_CATEGORIES[c]][img_id][case] = (precision, recall, f1, mAP_std, mAP_50, mAR_10, mAR_100, mae, rmse)


        mae = rmse = abs(n_obj - n_GT) # for single image MAE and RMSE are identical
        precision, recall, f1, mAP_std, mAP_50, mAR_10, mAR_100 = evaluate_metrics(preds, TARGET_CATEGORIES_IDX, [img_id])
        metrics_dict['global'][img_id][case] = (precision, recall, f1, mAP_std, mAP_50, mAR_10, mAR_100, mae, rmse)

# ------------------------------------------------------------------
# PRINT: one table, GLOBAL block first, then per-category blocks
# ------------------------------------------------------------------
table_metrics = PrettyTable()
table_metrics.title = f"{SOLVER_STRING[SOLVER]} - Metrics per Image"
table_metrics.field_names = ["Image ID (Boxes)", "Case", "Prec", "Rec", "F1", "mAP(std)", "mAP(.50)", "mAR(10)", "mAR(100)", "MAE", "RMSE"]

# --- GLOBAL block ---
table_metrics.add_row(["GLOBAL"] + ["" for _ in range(len(table_metrics.field_names) - 1)], divider=True)

for i in range(len(img_ids)):
    img_id = img_ids[i]

    for case, res in metrics_dict['global'][img_id].items():
        if case == 1:
            label = f"ID {img_ids[i]}"
        elif case == 2:
            label = f"({global_box_counts[i]} box)"
        else:
            label = ""
        prec, rec, f1, mAP_std, mAP_50, mAR_10, mAR_100, mae, rmse = res
        case_label = f"Case {case}"
        table_metrics.add_row([label, case_label, f"{prec:.4f}", f"{rec:.4f}", f"{f1:.4f}", f"{mAP_std:.4f}", f"{mAP_50:.4f}", f"{mAR_10:.4f}", f"{mAR_100:.4f}", f"{mae:.4f}", f"{rmse:.4f}"])
    if i != len(img_ids)-1:
        table_metrics.add_row(["-"*18, "-"*8, "-"*6, "-"*6, "-"*6, "-"*8, "-"*8, "-"*7, "-"*8, "-"*6, "-"*6])
table_metrics.add_divider()

# --- Per-category blocks ---
for cat, cat_name in zip(TARGET_CATEGORIES_IDX, TARGET_CATEGORIES):
    cat_box_counts = [len(data['boxes'][cat]) for data in gpu_data]

    table_metrics.add_row([cat_name] + ["" for _ in range(len(table_metrics.field_names) - 1)], divider=True)

    for i in range(len(img_ids)):
        img_id = img_ids[i]
    
        for case, res in metrics_dict[cat_name][img_id].items():
            if case == 1:
                label = f"ID {img_ids[i]}"
            elif case == 2:
                label = f"({cat_box_counts[i]} box)"
            else:
                label = ""
            prec, rec, f1, mAP_std, mAP_50, mAR_10, mAR_100, mae, rmse = res
            case_label = f"Case {case}"
            table_metrics.add_row([label, case_label, f"{prec:.4f}", f"{rec:.4f}", f"{f1:.4f}", f"{mAP_std:.4f}", f"{mAP_50:.4f}", f"{mAR_10:.4f}", f"{mAR_100:.4f}", f"{mae:.4f}", f"{rmse:.4f}"])
        if i != len(img_ids)-1:
            table_metrics.add_row(["-"*18, "-"*8, "-"*6, "-"*6, "-"*6, "-"*8, "-"*8, "-"*7, "-"*8, "-"*6, "-"*6])
    table_metrics.add_divider()

print(table_metrics)
print("\n")
