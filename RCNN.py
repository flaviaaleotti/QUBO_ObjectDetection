import torch
import cv2
import numpy as np

def faster_rcnn(image_path, model, device, target_labels=[1]): 
    """
    Extract bounding boxes using Faster R-CNN (ResNet-50 + FPN)
    
    R-CNN stands for Region-based Convolutional Neural Network
    and Faster R-CNN is a set of multiple neural networks working in chain
    
    Backbone (ResNet-50) extracts ke image features by inserting them into a feature map
    FPN = Feature Pyramid Network to detect both large and small objects
    
    Moreover there are a prompter (RPN) and a classifier (Head) that find regions and label them

    target_labels: list of int (default = [1], person)
    """
    
    # image upload
    img_cv = cv2.imread(image_path)
    
    # convert BGR in RGB + normalization
    img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
    img_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 255.0
    
    # move the tensor to GPU or CPU
    img_tensor = img_tensor.unsqueeze(0).to(device) 

    # run model
    with torch.no_grad():
        predictions = model(img_tensor)

    # results are shifted to CPU to work with Numpy
    pred_boxes = predictions[0]['boxes'].cpu().numpy()
    pred_scores = predictions[0]['scores'].cpu().numpy()
    pred_labels = predictions[0]['labels'].cpu().numpy()

    # !!! extract only desired category (via target_label)
    # "person" is category 1
    target_indices = np.where(np.isin(pred_labels, target_labels))[0]
    boxes = pred_boxes[target_indices] # [x1, y1, x2, y2] format
    scores = pred_scores[target_indices]
    labels = pred_labels[target_indices]

    # we convert the boxes from [x1, y1, x2, y2] to [x, y, w, h] format
    final_boxes = []
    for box in boxes:
        x1, y1, x2, y2 = box
        w = x2 - x1
        h = y2 - y1
        final_boxes.append([int(x1), int(y1), int(w), int(h)])
    
    # return array numpy, scores and original image + labels for multiclass version
    return np.array(boxes), scores, img_cv, labels


