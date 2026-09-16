import IoU
import numpy as np


def compute_metrics(coco_gt, predictions_list, image_ids_list, iou_threshold=0.5, cat_ID=[1]):
    """
    Compute Precision, Recall, F1 (IoU=0.5)
    """
    # dictionary of predictions for every image_id
    preds_by_img = {img_id: [] for img_id in image_ids_list}

    for p in predictions_list:
        if p['image_id'] in preds_by_img:
            preds_by_img[p['image_id']].append(p)

    # initialization global precision/recall
    global_tp = 0
    global_fp = 0
    total_ground_truth = 0 #TP + FN

    # iteration per image
    for img_id in image_ids_list:

        # Ground Truths
        ann_ids = coco_gt.getAnnIds(imgIds=img_id, catIds=cat_ID)
        anns = coco_gt.loadAnns(ann_ids)
        gt_boxes = [a['bbox'] for a in anns] # [x,y,w,h]
        
        # total gt update
        total_ground_truth += len(gt_boxes)

        # list of predictions
        current_preds = preds_by_img[img_id]

        # sorting predictions by confidence score
        current_preds.sort(key=lambda x: x['score'], reverse=True)

        # sorted predictions 
        pred_boxes = [p['bbox'] for p in current_preds]

        # we need to control every gt that has already matched to a prediction
        gt_matched = [False] * len(gt_boxes)
        
        # iteration on predictions
        for p_box in pred_boxes:
            best_iou = 0
            best_gt_idx = -1
            
            # find gt with highest IoU which is not in gt_matched
            for i, gt_box in enumerate(gt_boxes):
                if not gt_matched[i]:
                    iou = IoU.compute_iou(p_box, gt_box)
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = i
            
            # when we find a match we choose whether it is TP or not
            if best_iou >= iou_threshold:
                global_tp += 1
                gt_matched[best_gt_idx] = True
            else:
                global_fp += 1
        

    # Precision = TP / (TP + FP)
    precision = global_tp / (global_tp + global_fp) if (global_tp + global_fp) > 0 else 0
    
    # Recall = TP / TotalGT
    recall = global_tp / total_ground_truth if total_ground_truth > 0 else 0
    
    # F1 Score
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return precision, recall, f1