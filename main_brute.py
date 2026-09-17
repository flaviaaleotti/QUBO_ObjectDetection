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

# SOLUTION TO KERNEL CRASH: ignore conflicts between OpenMP libraries (Gurobi vs. PyTorch/NumPy)
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import build_qubo_matrix 
import build_qubo_matrix2
import build_qubo_matrix3
import build_qubo_matrix4
import metrics
import RCNN
import logging
import brute_force

# silenced torchvision/pytorch
logging.getLogger("torchvision").setLevel(logging.ERROR)
logging.getLogger("torch").setLevel(logging.ERROR)


import cv2
# helper to visualize results vs ground truth
def visualize_boxes(image_path, gt, pred, output_filename):
    """
    Draw boxes for ground truth in red and prediction in green
    """
    image = cv2.imread(image_path)

    # colors are BGR (Blue, Green, Red), not RGB
    red = (0, 0, 255)     # red->gt
    green = (0, 255, 0)   # green->box

    for box in gt:
        x, y, w, h = [int(b) for b in box]
        cv2.rectangle(image, (x, y), (x + w, y + h), red, 2)

    for box in pred:
        x, y, w, h = [int(b) for b in box]
        cv2.rectangle(image, (x, y), (x + w, y + h), green, 2)

    # image saving
    cv2.imwrite(output_filename, image)

# path to the file containing the ground truths for each image (called with an ID)
instances_file = os.path.join(os.environ["COCO_DATASET"], "annotations/instances_val2017.json")
coco = COCO(instances_file) #initialization

TARGET_CATEGORIES = ['person', 'car']
TARGET_CATEGORIES_IDX = coco.getCatIds(catNms=TARGET_CATEGORIES)

# select only images that have people (and that have people as ground truth)
image_IDs = coco.getImgIds(catIds=TARGET_CATEGORIES_IDX) # Image IDs

print(f"Number of images with", end = ' ')
for i in range(len(TARGET_CATEGORIES)):
    if i == len(TARGET_CATEGORIES) - 1:
        print(TARGET_CATEGORIES[i] + ': ')
    else:
        print(TARGET_CATEGORIES[i], end=' and ')
print(len(image_IDs))

# ========================================================
# image ID we like to analyze
image_IDs = [258793, 361506, 315187, 74058, 436738, 97679, 414510] #, 495146] --> last image only with GPU

# ORIGINAL SET FROM THESIS
# !!! use the instance 213035 (31 box) only with GPU
#image_IDs = [532481, 270908, 458755, 213035] # 5, 14, 23, 31 boxes
'''
# FULL SET OF IMAGES FROM COCO VALIDATION SET CONTAINING PERSON AND CAR (359 images)
image_IDs = [532481, 184324, 546823, 393226, 102411, 169996, 67616, 397351, 555050, 477227, 
             284725, 157756, 507975, 204871, 356424, 301135, 231508, 505942, 98392, 442456, 
             30828, 342128, 127092, 319607, 346232, 391290, 292997, 309391, 243867, 194716, 
             192670, 32941, 278705, 334006, 446651, 303305, 84170, 86220, 192716, 426203, 
             278749, 424162, 276707, 157928, 135410, 313588, 57597, 200961, 356612, 213255, 
             160012, 147725, 198928, 100624, 147740, 426268, 127263, 411938, 184611, 383289, 
             565563, 579902, 301376, 278848, 209222, 57672, 74058, 545100, 567640, 475484, 
             336232, 260470, 442746, 526728, 577932, 31118, 18837, 102805, 151962, 121242, 
             428454, 156071, 135604, 563653, 33221, 303566, 86483, 176606, 334309, 111086, 
             295420, 68093, 137727, 436738, 84492, 277005, 395801, 514586, 45596, 94751, 
             334371, 191013, 553511, 492077, 512564, 408120, 221754, 287291, 39484, 227898, 
             324158, 555597, 113235, 213593, 297562, 504415, 213605, 172648, 373353, 297578, 
             230008, 436883, 215723, 414385, 363188, 361142, 283318, 338624, 338625, 553669, 
             303818, 309964, 47828, 211674, 334555, 266981, 320232, 258793, 242411, 432898, 
             336658, 480021, 506656, 449312, 369442, 414510, 273198, 25393, 17207, 289594, 
             488251, 260925, 580418, 568147, 158548, 506707, 254814, 549738, 463730, 295809, 
             82821, 314251, 326541, 537506, 293794, 396200, 5037, 7088, 521141, 461751, 11197, 
             144333, 517069, 183246, 515025, 476119, 160728, 394206, 33759, 568290, 463849, 
             437239, 566282, 64523, 273420, 424975, 381971, 158744, 148508, 361506, 162858, 
             521259, 396338, 433204, 429109, 33854, 367680, 179265, 365642, 42070, 281687, 
             410712, 377946, 511076, 119911, 87144, 564336, 226417, 27768, 441468, 142472, 
             289938, 40083, 369812, 468124, 457884, 382111, 38048, 58539, 165039, 343218, 
             193717, 453841, 167122, 169169, 17627, 115946, 146667, 163057, 142585, 410880, 
             394510, 345361, 378139, 296224, 181542, 322864, 122166, 269632, 54593, 570688, 
             21839, 130386, 560474, 210273, 161128, 357737, 433515, 468332, 460147, 128372, 
             171382, 85376, 290179, 138639, 97679, 146831, 398742, 183709, 153011, 490936, 
             105912, 357816, 132544, 105923, 343496, 224724, 384468, 499181, 226802, 81394, 
             478721, 136715, 392722, 513567, 208423, 495146, 407083, 196141, 44590, 181816, 
             460347, 220732, 230983, 245320, 228942, 421455, 419408, 32334, 458325, 521819, 
             26204, 212573, 259690, 427655, 130699, 67213, 206487, 9891, 349860, 157365, 
             54967, 423617, 177861, 569030, 296649, 491213, 155341, 542423, 138979, 493286, 
             134886, 69356, 538364, 575243, 177934, 511760, 507667, 46872, 284445, 380706, 
             550691, 165681, 376625, 315187, 350003, 378673, 65350, 350023, 520009, 198489, 
             139099, 313182, 261982, 274272, 55150, 124798, 55167, 393093, 188296, 319369, 
             470924, 329614, 507797, 6040, 567197, 341921, 176037, 343976, 255917, 464824, 
             350148, 425925, 491464, 344029, 274411, 417779, 413689, 511999]
'''
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
    raw_boxes, scores, _, labels= RCNN.faster_rcnn(image_path, model, device, TARGET_CATEGORIES_IDX)
    
    if device.type == 'cuda':
        torch.cuda.synchronize()
    elif device.type == 'mps':
        torch.mps.synchronize()
    t_rcnn_end = time.perf_counter()
    
    valid_count +=1

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

ground_truths = {}

for i, data in enumerate(gpu_data):
    image_id = data['image_id']
    boxes = data['boxes']
    scores = data['scores']
    file_name = data['file_name']

    annIds = coco.getAnnIds(imgIds=image_id, catIds=TARGET_CATEGORIES_IDX)
    anns = coco.loadAnns(annIds) 
    
    gt_boxes = []
    gt_labels = []
    for ann in anns:
        gt_boxes.append(ann['bbox']) 
        gt_labels.append(ann['category_id'])
    
    ground_truths[image_id] = {
        'boxes': np.array(gt_boxes),
        'labels': np.array(gt_labels),
        'file_name': file_name
    }

print("GT AND PREDICTIONS LOADING IS COMPLETED")

# best alpha computed using best_alpha_gurobi.py
best_alpha = {'person': [0.58, 0.60, 0.60, 0.62],
              'car'   : [0.62, 0.58, 0.58, 0.62]}

# lists for times
brute_times = {'person' : [[] for i in range(4)],
               'car'    : [[] for i in range(4)]}

# lists for predictions
brute_results = {'person' : [[] for i in range(4)],
                 'car'    : [[] for i in range(4)]}

# lists for pred_count and gt_count for MAE
counts = {'person' : [[] for i in range(4)],
          'car'    : [[] for i in range(4)]}

# WARM-UP
dummy_Q = np.random.rand(5, 5).astype(np.float64)

if device.type == 'cuda':
    _, _ = brute_force.qubo_brute_gpu(dummy_Q, device)
    torch.cuda.synchronize()
elif device.type == 'mps':
    _, _ = brute_force.qubo_brute_gpu(dummy_Q, device)    
    torch.mps.synchronize()
else:
    _, _ = brute_force.qubo_brute_gpu(dummy_Q, device)

# Create folder to store image results
os.makedirs("./images", exist_ok=True)
for cat in TARGET_CATEGORIES:
    os.makedirs(f"./images/{cat}", exist_ok=True)
    for case in range(1,5):
        os.makedirs(f"./images/{cat}/case_{case}", exist_ok=True)

# we analyze every image only once
for i, data in enumerate(gpu_data):
    for cat in TARGET_CATEGORIES_IDX:
        cat_name = coco.loadCats(cat)[0]['name']
        boxes = data['boxes'][cat]
        scores = data['scores'][cat]
        image_id = data['image_id']

        # retrieve info to draw boxes on the current image (predictions vs GT)
        img_info = coco.loadImgs(image_id)[0]
        file_name = img_info['file_name']
        src_path = os.path.join(os.environ["COCO_DATASET"], "val2017", file_name)
        
        # GT of this img
        mask = ground_truths[image_id]['labels'] == cat # mask out results from other categories
        gt_count = len(ground_truths[image_id]['boxes'][mask])
    
        boxes = boxes.astype(np.float64)
        scores = scores.astype(np.float64)
        N = len(boxes) 
    
        print(f"\n--- Image ID {image_id} ({N} predicted box for '{cat_name}') ---")

        # loop over penalty cases
        for case in range(4):

            # build Q matrix
            if case == 0:
                # --- CASE 1 (IoU) ---
                L, P = build_qubo_matrix.qubo_matrices(boxes, scores)
                Q = best_alpha[cat_name][0] * L - (1 - best_alpha[cat_name][0]) * P
            elif case == 1:
                # --- CASE 2 (IoU + IoM) ---
                L, P = build_qubo_matrix2.qubo_matrices(boxes, scores)
                Q = best_alpha[cat_name][1] * L - (1 - best_alpha[cat_name][1]) * P
            elif case == 2:
                # --- CASE 3 (IoU + Sp) ---
                L, P1, P2 = build_qubo_matrix3.qubo_matrices(boxes, scores)
                beta = (1 - best_alpha[cat_name][2]) / 2
                Q = best_alpha[cat_name][2] * L - beta * P1 - beta * P2
            elif case == 3:
                # --- CASE 4 (IoU+IoM + Sp) ---
                L, P1, P2 = build_qubo_matrix4.qubo_matrices(boxes, scores)
                beta = (1 - best_alpha[cat_name][3]) / 2
                Q = best_alpha[cat_name][3] * L - beta * P1 - beta * P2

            Q = np.round(Q, decimals=6)

            t_start = time.perf_counter()
            if device.type == 'cuda':
                sol, val = brute_force.qubo_brute_gpu(Q,device)
                torch.cuda.synchronize()
            elif device.type == 'mps':
                sol, val = brute_force.qubo_brute_gpu(Q,device)
                torch.mps.synchronize()
            else:
                sol, val = brute_force.qubo_brute_gpu(Q, device)

            brute_times[cat_name][case].append(time.perf_counter() - t_start)
            
            sol = np.array(sol)
    
            # print sol and energy(with -) case 1
            print(f"Case {case + 1} Sol: {sol.tolist()} with energy: {-val:.6f}")
    
            # collect indices of the kept boxes
            kept_indices = np.where(sol == 1)[0]
            counts[cat_name][case].append((len(kept_indices), gt_count)) # for MAE
            
            # I save the boxes, scores and labels corresponding to the indexes, if I don't have any boxes we skip this step
            image_predictions = []
            if len(kept_indices) > 0:
                kept_boxes = boxes[kept_indices]
                kept_scores = scores[kept_indices]
                for k in range(len(kept_boxes)):
                    image_predictions.append({
                        "image_id": int(image_id),
                        "category_id": cat, 
                        "bbox": kept_boxes[k].tolist(),
                        "score": float(kept_scores[k])
                    })
            brute_results[cat_name][case].extend(image_predictions)

            # draw GT and predicted boxes on the current image (and save it to dst_path)
            dst_path = f"./images/{cat_name}/case_{case+1}/{file_name}"
            visualize_boxes(src_path, ground_truths[image_id]['boxes'][mask], kept_boxes, dst_path)
    
        print(f"Brute force: Processed {i + 1} / {len(gpu_data)} images...")

# --- FINAL SUMMARY TIME TABLE ---
table_brute = PrettyTable()
table_brute.title = "Brute Force - Execution Times"

header_times = ["Config."] + [f"ID_{data['image_id']} (s)" for data in gpu_data]


table_brute.field_names = header_times

for cat in TARGET_CATEGORIES_IDX:
    cat_name = coco.loadCats(cat)[0]['name']
    table_brute.add_row([cat_name] + ["" for t in brute_times[cat_name][0]], divider=True)
    table_brute.add_row(["Case 1"] + [f"{t:.4f}" for t in brute_times[cat_name][0]])
    table_brute.add_row(["Case 2"] + [f"{t:.4f}" for t in brute_times[cat_name][1]])
    table_brute.add_row(["Case 3"] + [f"{t:.4f}" for t in brute_times[cat_name][2]])
    table_brute.add_row(["Case 4"] + [f"{t:.4f}" for t in brute_times[cat_name][3]])
    table_brute.add_divider()

print(table_brute)
print("\n")


# --- FINAL SUMMARY METRICS TABLE (per category) ---
table_metrics = PrettyTable()
table_metrics.title = "Brute Force - Metrics"
table_metrics.field_names = ["n. boxes", "Config.", "F1", "mAP(std)", "mAP(.50)", "mAR(10)", "mAR(100)", "MAE"]

original_stdout = sys.stdout

all_img_ids = [d['image_id'] for d in gpu_data]

# ============================================================
# GLOBAL ANALYSIS (all categories)
# ============================================================

cases_info = [
    ("Case 1", {cn: brute_results[cn][0] for cn in TARGET_CATEGORIES}, {cn: counts[cn][0] for cn in TARGET_CATEGORIES}),
    ("Case 2", {cn: brute_results[cn][1] for cn in TARGET_CATEGORIES}, {cn: counts[cn][1] for cn in TARGET_CATEGORIES}),
    ("Case 3", {cn: brute_results[cn][2] for cn in TARGET_CATEGORIES}, {cn: counts[cn][2] for cn in TARGET_CATEGORIES}),
    ("Case 4", {cn: brute_results[cn][3] for cn in TARGET_CATEGORIES}, {cn: counts[cn][3] for cn in TARGET_CATEGORIES})
]

for i, data in enumerate(gpu_data):
    img_id = data['image_id']
    n_boxes = sum(len(data['boxes'][cat]) for cat in TARGET_CATEGORIES_IDX)  # total boxes, both categories

    for case_idx, (case_name, preds_by_cat, counts_by_cat) in enumerate(cases_info):

        # pool predictions from both categories, for this image only
        preds_img = []
        for cn in TARGET_CATEGORIES:
            preds_img.extend([p for p in preds_by_cat[cn] if p['image_id'] == img_id])

        # pool MAE across both categories, for this image only
        total_ae = 0
        for cn in TARGET_CATEGORIES:
            kept_boxes, gt_boxes = counts_by_cat[cn][i]
            total_ae += abs(kept_boxes - gt_boxes)
        mae = total_ae / len(TARGET_CATEGORIES)

        if len(preds_img) == 0:
            f1, mAP_std, mAP_50, mAR_10, mAR_100 = 0.0, 0.0, 0.0, 0.0, 0.0
        else:
            coco_dt = coco.loadRes(preds_img)
            coco_eval = COCOeval(coco, coco_dt, 'bbox')
            coco_eval.params.imgIds = [img_id]
            coco_eval.params.catIds = TARGET_CATEGORIES_IDX  # both categories, this image only

            sys.stdout = open(os.devnull, 'w')
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

            precision, recall, f1 = metrics.compute_metrics(
                coco, preds_img, [img_id], cat_ID=TARGET_CATEGORIES_IDX
            )

        col_n_boxes = str(n_boxes) if case_idx == 0 else ""

        table_metrics.add_row([
            col_n_boxes,
            case_name,
            f"{f1:.4f}",
            f"{mAP_std:.4f}",
            f"{mAP_50:.4f}",
            f"{mAR_10:.4f}",
            f"{mAR_100:.4f}",
            f"{mae:.3f}"
        ])

    if i < len(gpu_data) - 1:
        table_metrics.add_row(["-"*8, "-"*8, "-"*6, "-"*8, "-"*8, "-"*8, "-"*8, "-"*6])

table_metrics.add_divider()

# ============================================================
# PER-CATEGORY ANALYSIS
# ============================================================

for cat in TARGET_CATEGORIES_IDX:
    cat_name = coco.loadCats(cat)[0]['name']

    table_metrics.add_row([cat_name] + ["" for t in range(len(table_metrics.field_names)-1)], divider=True)

    cases_info = [
        ('Case 1', brute_results[cat_name][0], counts[cat_name][0]),
        ('Case 2', brute_results[cat_name][1], counts[cat_name][1]),
        ('Case 3', brute_results[cat_name][2], counts[cat_name][2]),
        ('Case 4', brute_results[cat_name][3], counts[cat_name][3])
    ]
    
    original_stdout = sys.stdout
    
    for i, data in enumerate(gpu_data):
        n_boxes = len(data['boxes'][cat])
        img_id = data['image_id']


        for case_idx, (case_name, preds_all, counts_all) in enumerate(cases_info):
            
            preds_img = [p for p in preds_all if p['image_id'] == img_id]
            
            kept_boxes, gt_boxes = counts_all[i]
            mae = abs(kept_boxes - gt_boxes)
            
            if len(preds_img) == 0:
                f1, mAP_std, mAP_50, mAR_10, mAR_100 = 0.0, 0.0, 0.0, 0.0, 0.0
            else:
                coco_dt = coco.loadRes(preds_img)
                coco_eval = COCOeval(coco, coco_dt, 'bbox')
                coco_eval.params.imgIds = [img_id]
                coco_eval.params.catIds = [cat]
                
                sys.stdout = open(os.devnull, 'w') # silence print
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
                    
                precision, recall, f1 = metrics.compute_metrics(coco, preds_img, [img_id], cat_ID=[cat])
    
            col_n_boxes = str(n_boxes) if case_idx == 0 else ""
    
            table_metrics.add_row([
                col_n_boxes,
                case_name,
                f"{f1:.4f}",
                f"{mAP_std:.4f}",
                f"{mAP_50:.4f}",
                f"{mAR_10:.4f}",
                f"{mAR_100:.4f}",
                f"{mae:.3f}"
            ])
            
        if i < len(gpu_data) - 1:
            table_metrics.add_row(["-"*8, "-"*8, "-"*6, "-"*8, "-"*8, "-"*8, "-"*8, "-"*6])

    table_metrics.add_divider()

print(table_metrics)
print("\n")