# %%
# IMPORT + INITIALIZATION
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import numpy as np
import sys
import os
import cv2
import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
import time
from prettytable import PrettyTable
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# SOLUTION TO KERNEL CRASH: ignore conflicts between OpenMP libraries (Gurobi vs. PyTorch/NumPy)
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import build_qubo_matrix 
import build_qubo_matrix2
import build_qubo_matrix3
import build_qubo_matrix4
import metrics
import RCNN
import logging
import neal

# silenced torchvision/pytorch
logging.getLogger("torchvision").setLevel(logging.ERROR)
logging.getLogger("torch").setLevel(logging.ERROR)

# Simulated Annealing initialization
sa_sampler = neal.SimulatedAnnealingSampler()

# path to the file containing the ground truths for each image (called with an ID)
instances_file = '../../coco2017/annotations/instances_val2017.json'
coco = COCO(instances_file) #initialization

# select only images that have people (and that have people as ground truth)
catIds = coco.getCatIds(catNms=['car'])
image_IDs = coco.getImgIds(catIds=catIds) # Image IDs

# we reduce the analysis to 300 images
if len(image_IDs) > 300:
    image_IDs = image_IDs[:300] 

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

# values that alpha can assume
val_alpha = np.linspace(0.5,0.7,11) #step size of 0.02


print("INITIALIZATION COMPLETED")

# BOUNDING BOX ACQUISITION WITH GPU

# data list initialization
gpu_data = []

# count images
valid_count = 0

for i, img_id in enumerate(image_IDs):
    
    # load image data
    img_info = coco.loadImgs(img_id)[0]
    file_name = img_info['file_name']

    # BOUNDING BOXES
    # image upload
    image_path = f"../../coco2017/val2017/{file_name}"

    # starting time for RCNN
    t_rcnn_start = time.perf_counter()

    # now we use GPU to run the model (if available)
    raw_boxes, scores, original_image = RCNN.faster_rcnn(image_path, model, device) # !!! boxes are in the format [x1, y1, x2, y2]
    
    # cuda synchronization
    if device.type == 'cuda':
        torch.cuda.synchronize()
    elif device.type == 'mps':
        torch.mps.synchronize()
    t_rcnn_end = time.perf_counter()

    num_boxes = len(raw_boxes)

    # keep images that have at least 2 boxes, but less than 90
    if num_boxes < 2 or num_boxes > 90:
        continue
    
    # we count this image
    valid_count +=1

    # we convert the boxes from [x1, y1, x2, y2] to [x, y, w, h] format
    boxes_xywh = []
    for box in raw_boxes:
        x1, y1, x2, y2 = box
        w = x2 - x1  # width
        h = y2 - y1  # height
        boxes_xywh.append([x1, y1, w, h]) 
    
    # print boxes and times
    print(f"[{valid_count}] Id imm: {img_id} | Nome imm: {file_name} | Box trovate: {len(raw_boxes)} | Tempo RCNN: {t_rcnn_end - t_rcnn_start:.3f}s")

    # saving data for QUBO
    gpu_data.append({
        'image_id': img_id,
        'boxes': np.array(boxes_xywh),
        'scores': scores,
        'file_name': file_name
    })

# images sort by number of boxes (using the list gpu_data)
gpu_data.sort(key=lambda x: len(x['boxes']))

# ID list update
valid_image_IDs = [data['image_id'] for data in gpu_data]

# compute sclicing indices
# we divide in 3 categories: LOW (2-20 boxes), MEDIUM (21-60 boxes), HIGH (61-90 boxes)
idx_cut_low = 0
idx_cut_med = 0

for i, data in enumerate(gpu_data):
    n_boxes = len(data['boxes'])
    
    if n_boxes <= 20:
        idx_cut_low = i + 1 
    
    if n_boxes <= 60:
        idx_cut_med = i + 1

# GROUND TRUTH ACQUISITION only for the "survived" images

# ground truths dictionary
ground_truths = {}

for i, data in enumerate(gpu_data):
    # extracting data
    image_id = data['image_id']
    boxes = data['boxes']
    scores = data['scores']
    file_name = data['file_name']

    # get the annotation IDs for this image
    annIds = coco.getAnnIds(imgIds=img_id)
    anns = coco.loadAnns(annIds) # load the annotations
    
    gt_boxes = []
    gt_labels = []
    for ann in anns:
        gt_boxes.append(ann['bbox']) # !!! boxes are in the format [x, y, w, h]
        gt_labels.append(ann['category_id']) # class (person, car, etc.)
    
    # Save to dictionary: Key = Image ID, Value = List of true boxes
    ground_truths[image_id] = {
        'boxes': np.array(gt_boxes),
        'labels': np.array(gt_labels),
        'file_name': file_name
    }

print("GT AND PREDICTIONS LOADING IS COMPLETED")
print(f"SOGLIE INDIVIDUATE SU {len(gpu_data)} IMMAGINI:")
print(f"- Low Density (2-20 box): da 0 a {idx_cut_low} ({idx_cut_low} immagini)")
print(f"- Med Density (21-60 box): da {idx_cut_low} a {idx_cut_med} ({idx_cut_med - idx_cut_low} immagini)")
print(f"- High Density (>60 box): da {idx_cut_med} alla fine ({len(gpu_data) - idx_cut_med} immagini)")


# %%
# BOX SUPPRESSION + mAP

#1) QUBO P = IoU
# !!! hyperparameters are alpha and beta = 1 - alpha

# initialize the results in a dictionary: Key = value of alpha, List = where to append results of each image
results = {alpha : [] for alpha in val_alpha} 
    
# initialize MAE e RMSE in a dictionary: Key = value of alpha, List = where to append MAE e RMSE per image
mae_rmse_stats = {alpha: {'abs_errors': [], 'sq_errors': []} for alpha in val_alpha}

# dictionary for execution time (sa) Key = value of alpha
times_sa = {alpha : [] for alpha in val_alpha} 

# loop using gpu_data
for i, data in enumerate(gpu_data):
    image_id = data['image_id']
    boxes = data['boxes']
    scores = data['scores']
    file_name = data['file_name']
    
    # we use gt dictionary
    gt_info = ground_truths[image_id]
    gt_boxes = gt_info['boxes']
    
    # number of gt
    gt_count = len(gt_boxes)

    # double precision for solver
    boxes = boxes.astype(np.float64)
    scores = scores.astype(np.float64)

    # compute L and P matrices only once per image
    L, P = build_qubo_matrix.qubo_matrices(boxes, scores)
    
    # iterate over alpha
    for alpha in val_alpha:

        beta = 1 - alpha 
        
        # Q matrix calculation   
        Q = alpha * L - beta * P
        Q = np.round(Q, decimals=6)
        
        # --- SIMULATED ANNEALING ---
        # converting Q matrix in {(j,k): value} remembering Q is symmetric
        # we insert a minus cause neal looks for MINIMUM
        Q_annealing = {}
        N = len(Q)
        for j in range(N):
            for k in range(N):
                if Q[j, k] != 0:
                    Q_annealing[(j, k)] = -Q[j, k]

        # measuring only sim annealing
        t_sa_start = time.perf_counter()

        # sampling 1000 times
        sampleset = sa_sampler.sample_qubo(Q_annealing, num_reads=1000)

        # best sol
        best_sample = sampleset.first.sample

        duration_sa = time.perf_counter() - t_sa_start
        times_sa[alpha].append(duration_sa)

        # conversion best sample in Array NumPy
        sol_sa = np.zeros(N, dtype=int)
        for idx, val in best_sample.items():
            sol_sa[idx] = val

        # collect indices of the kept boxes
        kept_indices = np.where(sol_sa == 1)[0]
        
        # MAE/RMSE calculation
        pred_count = len(kept_indices) # number of predictions
        diff = pred_count - gt_count
        
        # saving MAE and RMSE in the dictionary
        mae_rmse_stats[alpha]['abs_errors'].append(abs(diff)) # |pred - groundtr|
        mae_rmse_stats[alpha]['sq_errors'].append(diff ** 2) # (pred - groundtr)**2

        # I save the boxes, scores and labels corresponding to the indexes, if I don't have any boxes we skip this step
        if len(kept_indices) > 0:
            kept_boxes = boxes[kept_indices]
            kept_scores = scores[kept_indices]

            for k in range(len(kept_boxes)):
                prediction = {
                    "image_id": int(image_id),
                    "category_id": 1, # 1 is the "person" class
                    "bbox": kept_boxes[k].tolist(),     # [x, y, w, h]
                    "score": float(kept_scores[k])
                }
                # append the predictions to the list corresponding to alpha
                results[alpha].append(prediction)
        
        
    # Feedback every 10 images
    if (i+1) % 10 == 0:
        print(f"Processate {i+1} immagini...")


# TABLE WITH Precision, Recall, F1 score, mAP(standard), mAP(.50), mAR(10), mAR(100), Mean Absolute Error, Root Mean Squared Error
# + PLOTS

table = PrettyTable()

# column names
table.field_names = ["alpha", "Prec", "Rec", "F1", "mAP(standard)", "mAP(.50)", "mAR(10)", "mAR(100)", "MAE", "RMSE"]

# silenced print
original_stdout = sys.stdout

# lists for plots
f1_list = []
map_std_list = []
mar_std_list = []

for alpha, total_predictions in results.items():
    if len(total_predictions) == 0:
        continue
    
    # MAE and RMSE calculation (mean)
    mae = np.mean(mae_rmse_stats[alpha]['abs_errors']) # mean |pred - groundtr|
    rmse = np.sqrt(np.mean(mae_rmse_stats[alpha]['sq_errors'])) # sqrt(mean (pred - groundtr)**2)

    # metrics calculation
    # upload the results
    coco_dt = coco.loadRes(total_predictions)
    
    coco_eval = COCOeval(coco, coco_dt, 'bbox')
    
    #!!!! compare only ground truths boxes OF THE PROCESSED IMAGES 
    coco_eval.params.imgIds = valid_image_IDs
    
    # calculate only for the "person" class [1]
    coco_eval.params.catIds = [1]
    
    # silenced print
    sys.stdout = open(os.devnull, 'w')
    
    try:
        # silenced print
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize() 
    except Exception as e:
        # if there is an error, reset the print to see it
        sys.stdout = original_stdout
        print(f"Error during evaluation: {e}")
        break
    finally:
        # print restoration
        sys.stdout = original_stdout
    
    # mAP standard (average of IoU 0.50:0.95) using stats[0]
    mAP_standard = coco_eval.stats[0] 
    
    # mAP (IoU 0.50) using stats[1]
    mAP_50 = coco_eval.stats[1] 

    # mAR (max 10 detections per image) using stats[7]
    mAR_10 = coco_eval.stats[7] 
    
    # mAR (max 100 detections per image) using stats[8]
    mAR_100 = coco_eval.stats[8] 
 
    # Precision Recall and F1 score
    precision, recall, f1 = metrics.compute_metrics(coco, total_predictions, valid_image_IDs)

    # add rows
    table.add_row([
        f"{alpha:.2f}", 
        f"{precision:.4f}", 
        f"{recall:.4f}", 
        f"{f1:.4f}", 
        f"{mAP_standard:.4f}", 
        f"{mAP_50:.4f}",
        f"{mAR_10:.4f}", 
        f"{mAR_100:.4f}",
        f"{mae:.3f}",
        f"{rmse:.3f}"
    ])

    # list append for plots
    f1_list.append(f1)
    map_std_list.append(mAP_standard)
    mar_std_list.append(mAR_100)

print(" ")
print("SUMMARY TABLE CASE 1")
print(" ")
print(table)

# BEST alpha WITH BEST mAP
best_idx = np.argmax(map_std_list)
best_map = map_std_list[best_idx]
best_alpha_1 = val_alpha[best_idx]

print(f"\n FOR P = IoU THE BEST alpha IS {best_alpha_1:.2f} WITH mAP {best_map:.4f}.")   

# METRICS PLOTS (val_alpha on x axis)
fig1 = plt.figure(figsize=(12, 6))
plt.plot(val_alpha, f1_list, 'o-', label='F1 Score', linewidth=2, color='blue')
plt.plot(val_alpha, map_std_list, 's-', label='AP (Standard)', color='green')
plt.plot(val_alpha, mar_std_list, 'd-', label='AR@100 (Standard)', color='red')

plt.xticks(val_alpha, rotation=45) # rotation=45 for overlapping numbers

# highlight best alpha with a vertical line
plt.axvline(x=best_alpha_1, color='black', linestyle='--', alpha=0.4, label=f'Best $\\alpha$ ({best_alpha_1:.2f})')

# highlight F1, mAP and mAR related to best alpha with an horizontal line
best_f1 = f1_list[best_idx]
best_mar = mar_std_list[best_idx]
plt.axhline(y=best_f1, color='blue', linestyle='--', alpha=0.3)
plt.text(val_alpha[0], best_f1, f' F1: {best_f1:.4f}', color='blue', va='bottom', fontweight='bold', fontsize=9)
plt.axhline(y=best_map, color='green', linestyle='--', alpha=0.3)
plt.text(val_alpha[0], best_map, f' AP: {best_map:.4f}', color='green', va='bottom', fontsize=9)
plt.axhline(y=best_mar, color='red', linestyle='--', alpha=0.3)
plt.text(val_alpha[0], best_mar, f' AR: {best_mar:.4f}', color='red', va='bottom', fontsize=9)

plt.title('Metrics: F1, AP and AR')
plt.xlabel('$\\alpha$')
plt.ylabel('Metric value')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

fig1.savefig('metrics_sa_case1.png', dpi=300)
plt.close(fig1)


# TIMES PLOTS SA with slicing
# average time for each density
avg_low = []
avg_med = []
avg_high = []

for o in val_alpha:
    # list of this alpha
    t_list = times_sa[o]
    
    # low density's list
    avg_low.append(np.mean(t_list[0 : idx_cut_low]))
    
    # medium density's list
    avg_med.append(np.mean(t_list[idx_cut_low : idx_cut_med]))
    
    # high density's list
    avg_high.append(np.mean(t_list[idx_cut_med : ]))

# total time for sa
total_sa = [np.sum(times_sa[o]) for o in val_alpha]

fig2 = plt.figure(figsize=(14, 6))

# average times
plt.subplot(1, 2, 1)
plt.plot(val_alpha, avg_low, 'v-', label=f'Low (2-20 box) [{idx_cut_low} img]', color='blue', alpha=0.8)
plt.plot(val_alpha, avg_med, 'o-', label=f'Med (21-60 box) [{idx_cut_med - idx_cut_low} img]', color='green', alpha=0.8)
plt.plot(val_alpha, avg_high, 's-', label=f'High (61-90 box) [{len(gpu_data) - idx_cut_med} img]', color='red', linewidth=2)
plt.xticks(val_alpha, rotation=45)

# highlight best alpha
plt.axvline(x=best_alpha_1, color='black', linestyle='--', alpha=0.4, label=f'Best $\\alpha$ ({best_alpha_1:.2f})')

# highlight avg time related to best alpha
best_low = avg_low[best_idx]
plt.axhline(y=best_low, color='blue', linestyle='--', alpha=0.4)
plt.text(val_alpha[0], best_low, f' {best_low:.4f}s', color='blue', va='bottom', fontweight='bold', fontsize=9)
best_med = avg_med[best_idx]
plt.axhline(y=best_med, color='green', linestyle='--', alpha=0.4)
plt.text(val_alpha[0], best_med, f' {best_med:.4f}s', color='green', va='bottom', fontweight='bold', fontsize=9)
best_high = avg_high[best_idx]
plt.axhline(y=best_high, color='red', linestyle='--', alpha=0.4)
plt.text(val_alpha[0], best_high, f' {best_high:.4f}s', color='red', va='bottom', fontweight='bold', fontsize=9)

plt.title('Average Time SA per Image (3 Densities)')
plt.xlabel('$\\alpha$')
plt.ylabel('Seconds')
plt.legend()
plt.grid(True, alpha=0.3)

# total times
plt.subplot(1, 2, 2)
plt.plot(val_alpha, total_sa, 'o-', label='Tot SA (All Images)', color='black')
plt.xticks(val_alpha, rotation=45)

# highlight best alpha
plt.axvline(x=best_alpha_1, color='black', linestyle='--', alpha=0.4, label=f'Best $\\alpha$ ({best_alpha_1:.2f})')

# highlight tot time related to best alpha
best_tot = total_sa[best_idx]
plt.axhline(y=best_tot, color='black', linestyle='--', alpha=0.4)
plt.text(val_alpha[0], best_tot, f' {best_tot:.2f}s', color='black', va='bottom', fontweight='bold', fontsize=9)

plt.title('Total Execution Time SA')
plt.xlabel('$\\alpha$')
plt.ylabel('Seconds')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
fig2.savefig('times_sa_case1.png', dpi=300)
plt.close(fig2)


# mAP density analysis
print("\n mAP analysis")

# predictions of best alpha
best_preds = results[best_alpha_1]

# sliced lists
ids_low  = valid_image_IDs[0 : idx_cut_low]
ids_med  = valid_image_IDs[idx_cut_low : idx_cut_med]
ids_high = valid_image_IDs[idx_cut_med : ]

# mAP computation on sliced list
def evaluate_subset(subset_ids, subset_name):
    if len(subset_ids) == 0:
        print(f"{subset_name}: Nessuna immagine.")
        return 0.0
        
    coco_dt = coco.loadRes(best_preds)
    coco_eval_sub = COCOeval(coco, coco_dt, 'bbox')
    coco_eval_sub.params.imgIds = subset_ids
    coco_eval_sub.params.catIds = [1] # class 'person'
    
    # silenced output
    original_stdout = sys.stdout
    sys.stdout = open(os.devnull, 'w')
    try:
        coco_eval_sub.evaluate()
        coco_eval_sub.accumulate()
        coco_eval_sub.summarize()
        mAP = coco_eval_sub.stats[0] # mAP 0.50:0.95
    except:
        mAP = 0.0
    finally:
        sys.stdout = original_stdout
        
    return mAP

# mAP evaluation
map_low  = evaluate_subset(ids_low, "Low Density")
map_med  = evaluate_subset(ids_med, "Medium Density")
map_high = evaluate_subset(ids_high, "High Density")

table_dens = PrettyTable()
table_dens.field_names = ["Density", "Range Box", "N. Images", "mAP @ Best alpha with Sim Ann"]
table_dens.add_row(["Low", "2-20", len(ids_low), f"{map_low:.4f}"])
table_dens.add_row(["Medium", "21-60", len(ids_med), f"{map_med:.4f}"])
table_dens.add_row(["High", "61-90", len(ids_high), f"{map_high:.4f}"])

print(table_dens)

# %%
#2) QUBO P = 0.7*IoU + 0.3*IoM
# !!! hyperparameters are alpha and beta = 1 - alpha

# initialize the results in a dictionary: Key = value of alpha, List = where to append results of each image
results = {alpha : [] for alpha in val_alpha} 
    
# initialize MAE e RMSE in a dictionary: Key = value of alpha, List = where to append MAE e RMSE per image
mae_rmse_stats = {alpha: {'abs_errors': [], 'sq_errors': []} for alpha in val_alpha}

# dictionary for execution time (sa) Key = value of alpha
times_sa = {alpha : [] for alpha in val_alpha} 

# loop using gpu_data
for i, data in enumerate(gpu_data):
    image_id = data['image_id']
    boxes = data['boxes']
    scores = data['scores']
    file_name = data['file_name']
    
    # we use gt dictionary
    gt_info = ground_truths[image_id]
    gt_boxes = gt_info['boxes']
    
    # number of gt
    gt_count = len(gt_boxes)

    # double precision for solver
    boxes = boxes.astype(np.float64)
    scores = scores.astype(np.float64)

    # compute L and P matrices only once per image
    L, P = build_qubo_matrix2.qubo_matrices(boxes, scores)
    
    # iterate over alpha
    for alpha in val_alpha:

        beta = 1 - alpha 
        
        # Q matrix calculation   
        Q = alpha * L - beta * P
        Q = np.round(Q, decimals=6)
        
        # --- SIMULATED ANNEALING ---
        # converting Q matrix in {(j,k): value} remembering Q is symmetric
        # we insert a minus cause neal looks for MINIMUM
        Q_annealing = {}
        N = len(Q)
        for j in range(N):
            for k in range(N):
                if Q[j, k] != 0:
                    Q_annealing[(j, k)] = -Q[j, k]

        # measuring only sim annealing
        t_sa_start = time.perf_counter()

        # sampling 1000 times
        sampleset = sa_sampler.sample_qubo(Q_annealing, num_reads=1000)

        # best sol
        best_sample = sampleset.first.sample

        duration_sa = time.perf_counter() - t_sa_start
        times_sa[alpha].append(duration_sa)

        # conversion best sample in Array NumPy
        sol_sa = np.zeros(N, dtype=int)
        for idx, val in best_sample.items():
            sol_sa[idx] = val

        # collect indices of the kept boxes
        kept_indices = np.where(sol_sa == 1)[0]
        
        # MAE/RMSE calculation
        pred_count = len(kept_indices) # number of predictions
        diff = pred_count - gt_count
        
        # saving MAE and RMSE in the dictionary
        mae_rmse_stats[alpha]['abs_errors'].append(abs(diff)) # |pred - groundtr|
        mae_rmse_stats[alpha]['sq_errors'].append(diff ** 2) # (pred - groundtr)**2

        # I save the boxes, scores and labels corresponding to the indexes, if I don't have any boxes we skip this step
        if len(kept_indices) > 0:
            kept_boxes = boxes[kept_indices]
            kept_scores = scores[kept_indices]

            for k in range(len(kept_boxes)):
                prediction = {
                    "image_id": int(image_id),
                    "category_id": 1, # 1 is the "person" class
                    "bbox": kept_boxes[k].tolist(),     # [x, y, w, h]
                    "score": float(kept_scores[k])
                }
                # append the predictions to the list corresponding to alpha
                results[alpha].append(prediction)


    # Feedback every 10 images
    if (i+1) % 10 == 0:
        print(f"Processate {i+1} immagini...")


# TABLE WITH Precision, Recall, F1 score, mAP(standard), mAP(.50), mAR(10), mAR(100), Mean Absolute Error, Root Mean Squared Error
# + PLOTS

table = PrettyTable()

# column names
table.field_names = ["alpha", "Prec", "Rec", "F1", "mAP(standard)", "mAP(.50)", "mAR(10)", "mAR(100)", "MAE", "RMSE"]

# silenced print
original_stdout = sys.stdout

# lists for plots
f1_list = []
map_std_list = []
mar_std_list = []

for alpha, total_predictions in results.items():
    if len(total_predictions) == 0:
        continue
    
    # MAE and RMSE calculation (mean)
    mae = np.mean(mae_rmse_stats[alpha]['abs_errors']) # mean |pred - groundtr|
    rmse = np.sqrt(np.mean(mae_rmse_stats[alpha]['sq_errors'])) # sqrt(mean (pred - groundtr)**2)

    # metrics calculation
    # upload the results
    coco_dt = coco.loadRes(total_predictions)
    
    coco_eval = COCOeval(coco, coco_dt, 'bbox')
    
    #!!!! compare only ground truths boxes OF THE PROCESSED IMAGES 
    coco_eval.params.imgIds = valid_image_IDs
    
    # calculate only for the "person" class [1]
    coco_eval.params.catIds = [1]
    
    # silenced print
    sys.stdout = open(os.devnull, 'w')
    
    try:
        # silenced print
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize() 
    except Exception as e:
        # if there is an error, reset the print to see it
        sys.stdout = original_stdout
        print(f"Error during evaluation: {e}")
        break
    finally:
        # print restoration
        sys.stdout = original_stdout
    
    # mAP standard (average of IoU 0.50:0.95) using stats[0]
    mAP_standard = coco_eval.stats[0] 
    
    # mAP (IoU 0.50) using stats[1]
    mAP_50 = coco_eval.stats[1] 

    # mAR (max 10 detections per image) using stats[7]
    mAR_10 = coco_eval.stats[7] 
    
    # mAR (max 100 detections per image) using stats[8]
    mAR_100 = coco_eval.stats[8] 
 
    # Precision Recall and F1 score
    precision, recall, f1 = metrics.compute_metrics(coco, total_predictions, valid_image_IDs)

    # add rows
    table.add_row([
        f"{alpha:.2f}", 
        f"{precision:.4f}", 
        f"{recall:.4f}", 
        f"{f1:.4f}", 
        f"{mAP_standard:.4f}", 
        f"{mAP_50:.4f}",
        f"{mAR_10:.4f}", 
        f"{mAR_100:.4f}",
        f"{mae:.3f}",
        f"{rmse:.3f}"
    ])

    # list append for plots
    f1_list.append(f1)
    map_std_list.append(mAP_standard)
    mar_std_list.append(mAR_100)

print(" ")
print("SUMMARY TABLE CASE 2")
print(" ")
print(table)

# BEST alpha WITH BEST mAP
best_idx = np.argmax(map_std_list)
best_map = map_std_list[best_idx]
best_alpha_2 = val_alpha[best_idx]

print(f"\n FOR P = 0.7*IoU + 0.3*IoM THE BEST alpha IS {best_alpha_2:.2f} WITH mAP {best_map:.4f}.")   

# METRICS PLOTS (val_alpha on x axis)
fig1 = plt.figure(figsize=(12, 6))
plt.plot(val_alpha, f1_list, 'o-', label='F1 Score', linewidth=2, color='blue')
plt.plot(val_alpha, map_std_list, 's-', label='AP (Standard)', color='green')
plt.plot(val_alpha, mar_std_list, 'd-', label='AR@100 (Standard)', color='red')

plt.xticks(val_alpha, rotation=45) # rotation=45 for overlapping numbers

# highlight best alpha with a vertical line
plt.axvline(x=best_alpha_2, color='black', linestyle='--', alpha=0.4, label=f'Best $\\alpha$ ({best_alpha_2:.2f})')

# highlight F1, mAP and mAR related to best alpha with an horizontal line
best_f1 = f1_list[best_idx]
best_mar = mar_std_list[best_idx]
plt.axhline(y=best_f1, color='blue', linestyle='--', alpha=0.3)
plt.text(val_alpha[0], best_f1, f' F1: {best_f1:.4f}', color='blue', va='bottom', fontweight='bold', fontsize=9)
plt.axhline(y=best_map, color='green', linestyle='--', alpha=0.3)
plt.text(val_alpha[0], best_map, f' AP: {best_map:.4f}', color='green', va='bottom', fontsize=9)
plt.axhline(y=best_mar, color='red', linestyle='--', alpha=0.3)
plt.text(val_alpha[0], best_mar, f' AR: {best_mar:.4f}', color='red', va='bottom', fontsize=9)

plt.title('Metrics: F1, AP and AR')
plt.xlabel('$\\alpha$')
plt.ylabel('Metric value')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

fig1.savefig('metrics_sa_case2.png', dpi=300)
plt.close(fig1)

# TIMES PLOTS SA with slicing
# average time for each density
avg_low = []
avg_med = []
avg_high = []

for o in val_alpha:
    # list of this alpha
    t_list = times_sa[o]
    
    # low density's list
    avg_low.append(np.mean(t_list[0 : idx_cut_low]))
    
    # medium density's list
    avg_med.append(np.mean(t_list[idx_cut_low : idx_cut_med]))
    
    # high density's list
    avg_high.append(np.mean(t_list[idx_cut_med : ]))

# total time for sa
total_sa = [np.sum(times_sa[o]) for o in val_alpha]

fig2 = plt.figure(figsize=(14, 6))

# average times
plt.subplot(1, 2, 1)
plt.plot(val_alpha, avg_low, 'v-', label=f'Low (2-20 box) [{idx_cut_low} img]', color='blue', alpha=0.8)
plt.plot(val_alpha, avg_med, 'o-', label=f'Med (21-60 box) [{idx_cut_med - idx_cut_low} img]', color='green', alpha=0.8)
plt.plot(val_alpha, avg_high, 's-', label=f'High (61-90 box) [{len(gpu_data) - idx_cut_med} img]', color='red', linewidth=2)
plt.xticks(val_alpha, rotation=45)

# highlight best alpha
plt.axvline(x=best_alpha_2, color='black', linestyle='--', alpha=0.4, label=f'Best $\\alpha$ ({best_alpha_2:.2f})')

# highlight avg time related to best alpha
best_low = avg_low[best_idx]
plt.axhline(y=best_low, color='blue', linestyle='--', alpha=0.4)
plt.text(val_alpha[0], best_low, f' {best_low:.4f}s', color='blue', va='bottom', fontweight='bold', fontsize=9)
best_med = avg_med[best_idx]
plt.axhline(y=best_med, color='green', linestyle='--', alpha=0.4)
plt.text(val_alpha[0], best_med, f' {best_med:.4f}s', color='green', va='bottom', fontweight='bold', fontsize=9)
best_high = avg_high[best_idx]
plt.axhline(y=best_high, color='red', linestyle='--', alpha=0.4)
plt.text(val_alpha[0], best_high, f' {best_high:.4f}s', color='red', va='bottom', fontweight='bold', fontsize=9)

plt.title('Average Time SA per Image (3 Densities)')
plt.xlabel('$\\alpha$')
plt.ylabel('Seconds')
plt.legend()
plt.grid(True, alpha=0.3)

# total times
plt.subplot(1, 2, 2)
plt.plot(val_alpha, total_sa, 'o-', label='Tot SA (All Images)', color='black')
plt.xticks(val_alpha, rotation=45)

# highlight best alpha
plt.axvline(x=best_alpha_2, color='black', linestyle='--', alpha=0.4, label=f'Best $\\alpha$ ({best_alpha_2:.2f})')

# highlight tot time related to best alpha
best_tot = total_sa[best_idx]
plt.axhline(y=best_tot, color='black', linestyle='--', alpha=0.4)
plt.text(val_alpha[0], best_tot, f' {best_tot:.2f}s', color='black', va='bottom', fontweight='bold', fontsize=9)

plt.title('Total Execution Time SA')
plt.xlabel('$\\alpha$')
plt.ylabel('Seconds')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
fig2.savefig('times_sa_case2.png', dpi=300)
plt.close(fig2)


# mAP density analysis
print("\n mAP analysis")

# predictions of best alpha
best_preds = results[best_alpha_2]

# sliced lists
ids_low  = valid_image_IDs[0 : idx_cut_low]
ids_med  = valid_image_IDs[idx_cut_low : idx_cut_med]
ids_high = valid_image_IDs[idx_cut_med : ]

# mAP computation on sliced list
def evaluate_subset(subset_ids, subset_name):
    if len(subset_ids) == 0:
        print(f"{subset_name}: Nessuna immagine.")
        return 0.0
        
    coco_dt = coco.loadRes(best_preds)
    coco_eval_sub = COCOeval(coco, coco_dt, 'bbox')
    coco_eval_sub.params.imgIds = subset_ids
    coco_eval_sub.params.catIds = [1] # class 'person'
    
    # silenced output
    original_stdout = sys.stdout
    sys.stdout = open(os.devnull, 'w')
    try:
        coco_eval_sub.evaluate()
        coco_eval_sub.accumulate()
        coco_eval_sub.summarize()
        mAP = coco_eval_sub.stats[0] # mAP 0.50:0.95
    except:
        mAP = 0.0
    finally:
        sys.stdout = original_stdout
        
    return mAP

# mAP evaluation
map_low  = evaluate_subset(ids_low, "Low Density")
map_med  = evaluate_subset(ids_med, "Medium Density")
map_high = evaluate_subset(ids_high, "High Density")

table_dens = PrettyTable()
table_dens.field_names = ["Density", "Range Box", "N. Images", "mAP @ Best alpha with Sim Ann"]
table_dens.add_row(["Low", "2-20", len(ids_low), f"{map_low:.4f}"])
table_dens.add_row(["Medium", "21-60", len(ids_med), f"{map_med:.4f}"])
table_dens.add_row(["High", "61-90", len(ids_high), f"{map_high:.4f}"])

print(table_dens)
# %%
#3) QUBO P1 = IoU P2 = sp feat
# !!! hyperparameters are alpha, beta1 = (1 - alpha) / 2, beta2 = beta1

# initialize the results in a dictionary: Key = value of alpha, List = where to append results of each image
results = {alpha : [] for alpha in val_alpha} 
    
# initialize MAE e RMSE in a dictionary: Key = value of alpha, List = where to append MAE e RMSE per image
mae_rmse_stats = {alpha: {'abs_errors': [], 'sq_errors': []} for alpha in val_alpha}

# dictionary for execution time (sa) Key = value of alpha
times_sa = {alpha : [] for alpha in val_alpha} 

# loop using gpu_data
for i, data in enumerate(gpu_data):
    image_id = data['image_id']
    boxes = data['boxes']
    scores = data['scores']
    file_name = data['file_name']
    
    # we use gt dictionary
    gt_info = ground_truths[image_id]
    gt_boxes = gt_info['boxes']
    
    # number of gt
    gt_count = len(gt_boxes)

    # double precision for solver
    boxes = boxes.astype(np.float64)
    scores = scores.astype(np.float64)

    # compute L and P matrices only once per image
    L, P1, P2 = build_qubo_matrix3.qubo_matrices(boxes, scores)
    
    # iterate over alpha
    for alpha in val_alpha:

        beta1 = (1 - alpha) / 2
        beta2 = (1 - alpha) / 2
        
        # Q matrix calculation   
        Q = alpha * L - beta1 * P1 - beta2 * P2
        Q = np.round(Q, decimals=6)
        
        # --- SIMULATED ANNEALING ---
        # converting Q matrix in {(j,k): value} remembering Q is symmetric
        # we insert a minus cause neal looks for MINIMUM
        Q_annealing = {}
        N = len(Q)
        for j in range(N):
            for k in range(N):
                if Q[j, k] != 0:
                    Q_annealing[(j, k)] = -Q[j, k]

        # measuring only sim annealing
        t_sa_start = time.perf_counter()

        # sampling 1000 times
        sampleset = sa_sampler.sample_qubo(Q_annealing, num_reads=1000)

        # best sol
        best_sample = sampleset.first.sample

        duration_sa = time.perf_counter() - t_sa_start
        times_sa[alpha].append(duration_sa)

        # conversion best sample in Array NumPy
        sol_sa = np.zeros(N, dtype=int)
        for idx, val in best_sample.items():
            sol_sa[idx] = val

        # collect indices of the kept boxes
        kept_indices = np.where(sol_sa == 1)[0]
        
        # MAE/RMSE calculation
        pred_count = len(kept_indices) # number of predictions
        diff = pred_count - gt_count
        
        # saving MAE and RMSE in the dictionary
        mae_rmse_stats[alpha]['abs_errors'].append(abs(diff)) # |pred - groundtr|
        mae_rmse_stats[alpha]['sq_errors'].append(diff ** 2) # (pred - groundtr)**2

        # I save the boxes, scores and labels corresponding to the indexes, if I don't have any boxes we skip this step
        if len(kept_indices) > 0:
            kept_boxes = boxes[kept_indices]
            kept_scores = scores[kept_indices]

            for k in range(len(kept_boxes)):
                prediction = {
                    "image_id": int(image_id),
                    "category_id": 1, # 1 is the "person" class
                    "bbox": kept_boxes[k].tolist(),     # [x, y, w, h]
                    "score": float(kept_scores[k])
                }
                # append the predictions to the list corresponding to alpha
                results[alpha].append(prediction)

        
    # Feedback every 10 images
    if (i+1) % 10 == 0:
        print(f"Processate {i+1} immagini...")


# TABLE WITH Precision, Recall, F1 score, mAP(standard), mAP(.50), mAR(10), mAR(100), Mean Absolute Error, Root Mean Squared Error
# + PLOTS

table = PrettyTable()

# column names
table.field_names = ["alpha", "Prec", "Rec", "F1", "mAP(standard)", "mAP(.50)", "mAR(10)", "mAR(100)", "MAE", "RMSE"]

# silenced print
original_stdout = sys.stdout

# lists for plots
f1_list = []
map_std_list = []
mar_std_list = []

for alpha, total_predictions in results.items():
    if len(total_predictions) == 0:
        continue
    
    # MAE and RMSE calculation (mean)
    mae = np.mean(mae_rmse_stats[alpha]['abs_errors']) # mean |pred - groundtr|
    rmse = np.sqrt(np.mean(mae_rmse_stats[alpha]['sq_errors'])) # sqrt(mean (pred - groundtr)**2)

    # metrics calculation
    # upload the results
    coco_dt = coco.loadRes(total_predictions)
    
    coco_eval = COCOeval(coco, coco_dt, 'bbox')
    
    #!!!! compare only ground truths boxes OF THE PROCESSED IMAGES 
    coco_eval.params.imgIds = valid_image_IDs
    
    # calculate only for the "person" class [1]
    coco_eval.params.catIds = [1]
    
    # silenced print
    sys.stdout = open(os.devnull, 'w')
    
    try:
        # silenced print
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize() 
    except Exception as e:
        # if there is an error, reset the print to see it
        sys.stdout = original_stdout
        print(f"Error during evaluation: {e}")
        break
    finally:
        # print restoration
        sys.stdout = original_stdout
    
    # mAP standard (average of IoU 0.50:0.95) using stats[0]
    mAP_standard = coco_eval.stats[0] 
    
    # mAP (IoU 0.50) using stats[1]
    mAP_50 = coco_eval.stats[1] 

    # mAR (max 10 detections per image) using stats[7]
    mAR_10 = coco_eval.stats[7] 
    
    # mAR (max 100 detections per image) using stats[8]
    mAR_100 = coco_eval.stats[8] 
 
    # Precision Recall and F1 score
    precision, recall, f1 = metrics.compute_metrics(coco, total_predictions, valid_image_IDs)

    # add rows
    table.add_row([
        f"{alpha:.2f}", 
        f"{precision:.4f}", 
        f"{recall:.4f}", 
        f"{f1:.4f}", 
        f"{mAP_standard:.4f}", 
        f"{mAP_50:.4f}",
        f"{mAR_10:.4f}", 
        f"{mAR_100:.4f}",
        f"{mae:.3f}",
        f"{rmse:.3f}"
    ])

    # list append for plots
    f1_list.append(f1)
    map_std_list.append(mAP_standard)
    mar_std_list.append(mAR_100)

print(" ")
print("SUMMARY TABLE CASE 3")
print(" ")
print(table)

# BEST alpha WITH BEST mAP
best_idx = np.argmax(map_std_list)
best_map = map_std_list[best_idx]
best_alpha_3 = val_alpha[best_idx]

print(f"\n FOR P1 = IoU P2 = sp feat THE BEST alpha IS {best_alpha_3:.2f} WITH mAP {best_map:.4f}.")   

# METRICS PLOTS (val_alpha on x axis)
fig1 = plt.figure(figsize=(12, 6))
plt.plot(val_alpha, f1_list, 'o-', label='F1 Score', linewidth=2, color='blue')
plt.plot(val_alpha, map_std_list, 's-', label='AP (Standard)', color='green')
plt.plot(val_alpha, mar_std_list, 'd-', label='AR@100 (Standard)', color='red')

plt.xticks(val_alpha, rotation=45) # rotation=45 for overlapping numbers

# highlight best alpha with a vertical line
plt.axvline(x=best_alpha_3, color='black', linestyle='--', alpha=0.4, label=f'Best $\\alpha$ ({best_alpha_3:.2f})')

# highlight F1, mAP and mAR related to best alpha with an horizontal line
best_f1 = f1_list[best_idx]
best_mar = mar_std_list[best_idx]
plt.axhline(y=best_f1, color='blue', linestyle='--', alpha=0.3)
plt.text(val_alpha[0], best_f1, f' F1: {best_f1:.4f}', color='blue', va='bottom', fontweight='bold', fontsize=9)
plt.axhline(y=best_map, color='green', linestyle='--', alpha=0.3)
plt.text(val_alpha[0], best_map, f' AP: {best_map:.4f}', color='green', va='bottom', fontsize=9)
plt.axhline(y=best_mar, color='red', linestyle='--', alpha=0.3)
plt.text(val_alpha[0], best_mar, f' AR: {best_mar:.4f}', color='red', va='bottom', fontsize=9)

plt.title('Metrics: F1, AP and AR')
plt.xlabel('$\\alpha$')
plt.ylabel('Metric value')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

fig1.savefig('metrics_sa_case3.png', dpi=300)
plt.close(fig1)

# TIMES PLOTS SA with slicing
# average time for each density
avg_low = []
avg_med = []
avg_high = []

for o in val_alpha:
    # list of this alpha
    t_list = times_sa[o]
    
    # low density's list
    avg_low.append(np.mean(t_list[0 : idx_cut_low]))
    
    # medium density's list
    avg_med.append(np.mean(t_list[idx_cut_low : idx_cut_med]))
    
    # high density's list
    avg_high.append(np.mean(t_list[idx_cut_med : ]))

# total time for sa
total_sa = [np.sum(times_sa[o]) for o in val_alpha]

fig2 = plt.figure(figsize=(14, 6))

# average times
plt.subplot(1, 2, 1)
plt.plot(val_alpha, avg_low, 'v-', label=f'Low (2-20 box) [{idx_cut_low} img]', color='blue', alpha=0.8)
plt.plot(val_alpha, avg_med, 'o-', label=f'Med (21-60 box) [{idx_cut_med - idx_cut_low} img]', color='green', alpha=0.8)
plt.plot(val_alpha, avg_high, 's-', label=f'High (61-90 box) [{len(gpu_data) - idx_cut_med} img]', color='red', linewidth=2)
plt.xticks(val_alpha, rotation=45)

# highlight best alpha
plt.axvline(x=best_alpha_3, color='black', linestyle='--', alpha=0.4, label=f'Best $\\alpha$ ({best_alpha_3:.2f})')

# highlight avg time related to best alpha
best_low = avg_low[best_idx]
plt.axhline(y=best_low, color='blue', linestyle='--', alpha=0.4)
plt.text(val_alpha[0], best_low, f' {best_low:.4f}s', color='blue', va='bottom', fontweight='bold', fontsize=9)
best_med = avg_med[best_idx]
plt.axhline(y=best_med, color='green', linestyle='--', alpha=0.4)
plt.text(val_alpha[0], best_med, f' {best_med:.4f}s', color='green', va='bottom', fontweight='bold', fontsize=9)
best_high = avg_high[best_idx]
plt.axhline(y=best_high, color='red', linestyle='--', alpha=0.4)
plt.text(val_alpha[0], best_high, f' {best_high:.4f}s', color='red', va='bottom', fontweight='bold', fontsize=9)

plt.title('Average Time SA per Image (3 Densities)')
plt.xlabel('$\\alpha$')
plt.ylabel('Seconds')
plt.legend()
plt.grid(True, alpha=0.3)

# total times
plt.subplot(1, 2, 2)
plt.plot(val_alpha, total_sa, 'o-', label='Tot SA (All Images)', color='black')
plt.xticks(val_alpha, rotation=45)

# highlight best alpha
plt.axvline(x=best_alpha_3, color='black', linestyle='--', alpha=0.4, label=f'Best $\\alpha$ ({best_alpha_3:.2f})')

# highlight tot time related to best alpha
best_tot = total_sa[best_idx]
plt.axhline(y=best_tot, color='black', linestyle='--', alpha=0.4)
plt.text(val_alpha[0], best_tot, f' {best_tot:.2f}s', color='black', va='bottom', fontweight='bold', fontsize=9)

plt.title('Total Execution Time SA')
plt.xlabel('$\\alpha$')
plt.ylabel('Seconds')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
fig2.savefig('times_sa_case3.png', dpi=300)
plt.close(fig2)


# mAP density analysis
print("\n mAP analysis")

# predictions of best alpha
best_preds = results[best_alpha_3]

# sliced lists
ids_low  = valid_image_IDs[0 : idx_cut_low]
ids_med  = valid_image_IDs[idx_cut_low : idx_cut_med]
ids_high = valid_image_IDs[idx_cut_med : ]

# mAP computation on sliced list
def evaluate_subset(subset_ids, subset_name):
    if len(subset_ids) == 0:
        print(f"{subset_name}: Nessuna immagine.")
        return 0.0
        
    coco_dt = coco.loadRes(best_preds)
    coco_eval_sub = COCOeval(coco, coco_dt, 'bbox')
    coco_eval_sub.params.imgIds = subset_ids
    coco_eval_sub.params.catIds = [1] # class 'person'
    
    # silenced output
    original_stdout = sys.stdout
    sys.stdout = open(os.devnull, 'w')
    try:
        coco_eval_sub.evaluate()
        coco_eval_sub.accumulate()
        coco_eval_sub.summarize()
        mAP = coco_eval_sub.stats[0] # mAP 0.50:0.95
    except:
        mAP = 0.0
    finally:
        sys.stdout = original_stdout
        
    return mAP

# mAP evaluation
map_low  = evaluate_subset(ids_low, "Low Density")
map_med  = evaluate_subset(ids_med, "Medium Density")
map_high = evaluate_subset(ids_high, "High Density")

table_dens = PrettyTable()
table_dens.field_names = ["Density", "Range Box", "N. Images", "mAP @ Best alpha with Sim Ann"]
table_dens.add_row(["Low", "2-20", len(ids_low), f"{map_low:.4f}"])
table_dens.add_row(["Medium", "21-60", len(ids_med), f"{map_med:.4f}"])
table_dens.add_row(["High", "61-90", len(ids_high), f"{map_high:.4f}"])

print(table_dens)
# %%
#4) QUBO P1 = 0.7*IoU + 0.3*IoM  P2 = sp feat
# !!! hyperparameters are alpha, beta1 = (1 - alpha) / 2, beta2 = beta1

# initialize the results in a dictionary: Key = value of alpha, List = where to append results of each image
results = {alpha : [] for alpha in val_alpha} 
    
# initialize MAE e RMSE in a dictionary: Key = value of alpha, List = where to append MAE e RMSE per image
mae_rmse_stats = {alpha: {'abs_errors': [], 'sq_errors': []} for alpha in val_alpha}

# dictionary for execution time (sa) Key = value of alpha 
times_sa = {alpha : [] for alpha in val_alpha} 

# loop using gpu_data
for i, data in enumerate(gpu_data):
    image_id = data['image_id']
    boxes = data['boxes']
    scores = data['scores']
    file_name = data['file_name']
    
    # we use gt dictionary
    gt_info = ground_truths[image_id]
    gt_boxes = gt_info['boxes']
    
    # number of gt
    gt_count = len(gt_boxes)

    # double precision for solver
    boxes = boxes.astype(np.float64)
    scores = scores.astype(np.float64)

    # compute L and P matrices only once per image
    L, P1, P2 = build_qubo_matrix4.qubo_matrices(boxes, scores)
    
    # iterate over alpha
    for alpha in val_alpha:

        beta1 = (1 - alpha) / 2
        beta2 = (1 - alpha) / 2
        
        # Q matrix calculation   
        Q = alpha * L - beta1 * P1 - beta2 * P2
        Q = np.round(Q, decimals=6)
        
        # --- SIMULATED ANNEALING ---
        # converting Q matrix in {(j,k): value} remembering Q is symmetric
        # we insert a minus cause neal looks for MINIMUM
        Q_annealing = {}
        N = len(Q)
        for j in range(N):
            for k in range(N):
                if Q[j, k] != 0:
                    Q_annealing[(j, k)] = -Q[j, k]

        # measuring only sim annealing
        t_sa_start = time.perf_counter()

        # sampling 1000 times
        sampleset = sa_sampler.sample_qubo(Q_annealing, num_reads=1000)

        # best sol
        best_sample = sampleset.first.sample

        duration_sa = time.perf_counter() - t_sa_start
        times_sa[alpha].append(duration_sa)

        # conversion best sample in Array NumPy
        sol_sa = np.zeros(N, dtype=int)
        for idx, val in best_sample.items():
            sol_sa[idx] = val
        # collect indices of the kept boxes
        kept_indices = np.where(sol_sa == 1)[0]
        
        # MAE/RMSE calculation
        pred_count = len(kept_indices) # number of predictions
        diff = pred_count - gt_count
        
        # saving MAE and RMSE in the dictionary
        mae_rmse_stats[alpha]['abs_errors'].append(abs(diff)) # |pred - groundtr|
        mae_rmse_stats[alpha]['sq_errors'].append(diff ** 2) # (pred - groundtr)**2

        # I save the boxes, scores and labels corresponding to the indexes, if I don't have any boxes we skip this step
        if len(kept_indices) > 0:
            kept_boxes = boxes[kept_indices]
            kept_scores = scores[kept_indices]

            for k in range(len(kept_boxes)):
                prediction = {
                    "image_id": int(image_id),
                    "category_id": 1, # 1 is the "person" class
                    "bbox": kept_boxes[k].tolist(),     # [x, y, w, h]
                    "score": float(kept_scores[k])
                }
                # append the predictions to the list corresponding to alpha
                results[alpha].append(prediction)

        
    # Feedback every 10 images
    if (i+1) % 10 == 0:
        print(f"Processate {i+1} immagini...")


# TABLE WITH Precision, Recall, F1 score, mAP(standard), mAP(.50), mAR(10), mAR(100), Mean Absolute Error, Root Mean Squared Error
# + PLOTS

table = PrettyTable()

# column names
table.field_names = ["alpha", "Prec", "Rec", "F1", "mAP(standard)", "mAP(.50)", "mAR(10)", "mAR(100)", "MAE", "RMSE"]

# silenced print
original_stdout = sys.stdout

# lists for plots
f1_list = []
map_std_list = []
mar_std_list = []

for alpha, total_predictions in results.items():
    if len(total_predictions) == 0:
        continue
    
    # MAE and RMSE calculation (mean)
    mae = np.mean(mae_rmse_stats[alpha]['abs_errors']) # mean |pred - groundtr|
    rmse = np.sqrt(np.mean(mae_rmse_stats[alpha]['sq_errors'])) # sqrt(mean (pred - groundtr)**2)

    # metrics calculation
    # upload the results
    coco_dt = coco.loadRes(total_predictions)
    
    coco_eval = COCOeval(coco, coco_dt, 'bbox')
    
    #!!!! compare only ground truths boxes OF THE PROCESSED IMAGES 
    coco_eval.params.imgIds = valid_image_IDs
    
    # calculate only for the "person" class [1]
    coco_eval.params.catIds = [1]
    
    # silenced print
    sys.stdout = open(os.devnull, 'w')
    
    try:
        # silenced print
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize() 
    except Exception as e:
        # if there is an error, reset the print to see it
        sys.stdout = original_stdout
        print(f"Error during evaluation: {e}")
        break
    finally:
        # print restoration
        sys.stdout = original_stdout
    
    # mAP standard (average of IoU 0.50:0.95) using stats[0]
    mAP_standard = coco_eval.stats[0] 
    
    # mAP (IoU 0.50) using stats[1]
    mAP_50 = coco_eval.stats[1] 

    # mAR (max 10 detections per image) using stats[7]
    mAR_10 = coco_eval.stats[7] 
    
    # mAR (max 100 detections per image) using stats[8]
    mAR_100 = coco_eval.stats[8] 
 
    # Precision Recall and F1 score
    precision, recall, f1 = metrics.compute_metrics(coco, total_predictions, valid_image_IDs)

    # add rows
    table.add_row([
        f"{alpha:.2f}", 
        f"{precision:.4f}", 
        f"{recall:.4f}", 
        f"{f1:.4f}", 
        f"{mAP_standard:.4f}", 
        f"{mAP_50:.4f}",
        f"{mAR_10:.4f}", 
        f"{mAR_100:.4f}",
        f"{mae:.3f}",
        f"{rmse:.3f}"
    ])

    # list append for plots
    f1_list.append(f1)
    map_std_list.append(mAP_standard)
    mar_std_list.append(mAR_100)

print(" ")
print("SUMMARY TABLE CASE 4")
print(" ")
print(table)

# BEST alpha WITH BEST mAP
best_idx = np.argmax(map_std_list)
best_map = map_std_list[best_idx]
best_alpha_4 = val_alpha[best_idx]

print(f"\n FOR P1 = 0.7*IoU + 0.3*IoM  P2 = sp feat THE BEST alpha IS {best_alpha_4:.2f} WITH mAP {best_map:.4f}.")   

# METRICS PLOTS (val_alpha on x axis)
fig1 = plt.figure(figsize=(12, 6))
plt.plot(val_alpha, f1_list, 'o-', label='F1 Score', linewidth=2, color='blue')
plt.plot(val_alpha, map_std_list, 's-', label='AP (Standard)', color='green')
plt.plot(val_alpha, mar_std_list, 'd-', label='AR@100 (Standard)', color='red')

plt.xticks(val_alpha, rotation=45) # rotation=45 for overlapping numbers

# highlight best alpha with a vertical line
plt.axvline(x=best_alpha_4, color='black', linestyle='--', alpha=0.4, label=f'Best $\\alpha$ ({best_alpha_4:.2f})')

# highlight F1, mAP and mAR related to best alpha with an horizontal line
best_f1 = f1_list[best_idx]
best_mar = mar_std_list[best_idx]
plt.axhline(y=best_f1, color='blue', linestyle='--', alpha=0.3)
plt.text(val_alpha[0], best_f1, f' F1: {best_f1:.4f}', color='blue', va='bottom', fontweight='bold', fontsize=9)
plt.axhline(y=best_map, color='green', linestyle='--', alpha=0.3)
plt.text(val_alpha[0], best_map, f' AP: {best_map:.4f}', color='green', va='bottom', fontsize=9)
plt.axhline(y=best_mar, color='red', linestyle='--', alpha=0.3)
plt.text(val_alpha[0], best_mar, f' AR: {best_mar:.4f}', color='red', va='bottom', fontsize=9)

plt.title('Metrics: F1, AP and AR')
plt.xlabel('$\\alpha$')
plt.ylabel('Metric value')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

fig1.savefig('metrics_sa_case4.png', dpi=300)
plt.close(fig1)


# TIMES PLOTS SA with slicing
# average time for each density
avg_low = []
avg_med = []
avg_high = []

for o in val_alpha:
    # list of this alpha
    t_list = times_sa[o]
    
    # low density's list
    avg_low.append(np.mean(t_list[0 : idx_cut_low]))
    
    # medium density's list
    avg_med.append(np.mean(t_list[idx_cut_low : idx_cut_med]))
    
    # high density's list
    avg_high.append(np.mean(t_list[idx_cut_med : ]))

# total time for sa
total_sa = [np.sum(times_sa[o]) for o in val_alpha]

fig2 = plt.figure(figsize=(14, 6))

# average times
plt.subplot(1, 2, 1)
plt.plot(val_alpha, avg_low, 'v-', label=f'Low (2-20 box) [{idx_cut_low} img]', color='blue', alpha=0.8)
plt.plot(val_alpha, avg_med, 'o-', label=f'Med (21-60 box) [{idx_cut_med - idx_cut_low} img]', color='green', alpha=0.8)
plt.plot(val_alpha, avg_high, 's-', label=f'High (61-90 box) [{len(gpu_data) - idx_cut_med} img]', color='red', linewidth=2)
plt.xticks(val_alpha, rotation=45)

# highlight best alpha
plt.axvline(x=best_alpha_4, color='black', linestyle='--', alpha=0.4, label=f'Best $\\alpha$ ({best_alpha_4:.2f})')

# highlight avg time related to best alpha
best_low = avg_low[best_idx]
plt.axhline(y=best_low, color='blue', linestyle='--', alpha=0.4)
plt.text(val_alpha[0], best_low, f' {best_low:.4f}s', color='blue', va='bottom', fontweight='bold', fontsize=9)
best_med = avg_med[best_idx]
plt.axhline(y=best_med, color='green', linestyle='--', alpha=0.4)
plt.text(val_alpha[0], best_med, f' {best_med:.4f}s', color='green', va='bottom', fontweight='bold', fontsize=9)
best_high = avg_high[best_idx]
plt.axhline(y=best_high, color='red', linestyle='--', alpha=0.4)
plt.text(val_alpha[0], best_high, f' {best_high:.4f}s', color='red', va='bottom', fontweight='bold', fontsize=9)

plt.title('Average Time SA per Image (3 Densities)')
plt.xlabel('$\\alpha$')
plt.ylabel('Seconds')
plt.legend()
plt.grid(True, alpha=0.3)

# total times
plt.subplot(1, 2, 2)
plt.plot(val_alpha, total_sa, 'o-', label='Tot SA (All Images)', color='black')
plt.xticks(val_alpha, rotation=45)

# highlight best alpha
plt.axvline(x=best_alpha_4, color='black', linestyle='--', alpha=0.4, label=f'Best $\\alpha$ ({best_alpha_4:.2f})')

# highlight tot time related to best alpha
best_tot = total_sa[best_idx]
plt.axhline(y=best_tot, color='black', linestyle='--', alpha=0.4)
plt.text(val_alpha[0], best_tot, f' {best_tot:.2f}s', color='black', va='bottom', fontweight='bold', fontsize=9)

plt.title('Total Execution Time SA')
plt.xlabel('$\\alpha$')
plt.ylabel('Seconds')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
fig2.savefig('times_sa_case4.png', dpi=300)
plt.close(fig2)


# mAP density analysis
print("\n mAP analysis")

# predictions of best alpha
best_preds = results[best_alpha_4]

# sliced lists
ids_low  = valid_image_IDs[0 : idx_cut_low]
ids_med  = valid_image_IDs[idx_cut_low : idx_cut_med]
ids_high = valid_image_IDs[idx_cut_med : ]

# mAP computation on sliced list
def evaluate_subset(subset_ids, subset_name):
    if len(subset_ids) == 0:
        print(f"{subset_name}: Nessuna immagine.")
        return 0.0
        
    coco_dt = coco.loadRes(best_preds)
    coco_eval_sub = COCOeval(coco, coco_dt, 'bbox')
    coco_eval_sub.params.imgIds = subset_ids
    coco_eval_sub.params.catIds = [1] # class 'person'
    
    # silenced output
    original_stdout = sys.stdout
    sys.stdout = open(os.devnull, 'w')
    try:
        coco_eval_sub.evaluate()
        coco_eval_sub.accumulate()
        coco_eval_sub.summarize()
        mAP = coco_eval_sub.stats[0] # mAP 0.50:0.95
    except:
        mAP = 0.0
    finally:
        sys.stdout = original_stdout
        
    return mAP

# mAP evaluation
map_low  = evaluate_subset(ids_low, "Low Density")
map_med  = evaluate_subset(ids_med, "Medium Density")
map_high = evaluate_subset(ids_high, "High Density")

table_dens = PrettyTable()
table_dens.field_names = ["Density", "Range Box", "N. Images", "mAP @ Best alpha with Sim Ann"]
table_dens.add_row(["Low", "2-20", len(ids_low), f"{map_low:.4f}"])
table_dens.add_row(["Medium", "21-60", len(ids_med), f"{map_med:.4f}"])
table_dens.add_row(["High", "61-90", len(ids_high), f"{map_high:.4f}"])

print(table_dens)
# %%