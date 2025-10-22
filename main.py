from nanoDETR import nanoDETR

import torch
import torchvision.models as models
from torchvision import datasets, transforms as T
from torchvision.datasets import wrap_dataset_for_transforms_v2
from torchvision.models import ResNet50_Weights  # <-- import this
from torchvision.transforms import v2
from torchvision.utils import draw_bounding_boxes
from torchvision.ops import generalized_box_iou_loss # same loss / paper ref. as in original DETR paper
from torchvision.ops import box_convert
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.optimize import linear_sum_assignment
import math
import utils


# imagenet stats here: https://docs.pytorch.org/vision/main/models/generated/torchvision.models.resnet50.html#torchvision.models.ResNet50_Weights
data_filepath = '/Users/veb/ms/nanoDETR/data'
transform = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
dataset = datasets.VOCDetection(root = data_filepath, 
                                year = '2012', 
                                image_set = 'train', 
                                download = False,
                                transform = transform) # len 5717, consisten with data
dataset = wrap_dataset_for_transforms_v2(dataset) 

    
def match_loss(logits, bbox, gt_label, gt_bbox, l1_lambda, IoU_lambda):
    # class loss
    class_loss = F.cross_entropy(input = logits, target = gt_label)
    class_loss = -torch.exp(-class_loss) # original paper want raw probability
    class_loss = 0.1 * class_loss if gt_label.item() == 20 else class_loss

    # bbox loss
    bbox = box_convert(boxes = bbox, in_fmt = 'xywh', out_fmt = 'xyxy')
    IoU_loss = generalized_box_iou_loss(bbox, gt_bbox, reduction = 'mean')
    l1_loss = F.l1_loss(bbox, gt_bbox)
    bbox_loss = (IoU_lambda * IoU_loss) + (l1_lambda  * l1_loss)
    
    return class_loss + bbox_loss

def match_log_loss(logits, bbox, gt_label, gt_bbox, l1_lambda, IoU_lambda):
    # class loss
    class_loss = F.cross_entropy(input = logits, target = gt_label)
    class_loss = 0.1 * class_loss if gt_label.item() == 20 else class_loss

    # bbox loss
    bbox = box_convert(boxes = bbox, in_fmt = 'xywh', out_fmt = 'xyxy')
    IoU_loss = generalized_box_iou_loss(bbox, gt_bbox, reduction = 'mean')
    l1_loss = F.l1_loss(bbox, gt_bbox)
    bbox_loss = (IoU_lambda * IoU_loss) + (l1_lambda  * l1_loss)
    
    return class_loss + bbox_loss


def loss(l1_lambda, IoU_lambda, logits, boxes, gt_labels, gt_boxes):
    
    # loss matrix
     
    loss_matrix = np.zeros((len(gt_labels),100)) # row_i = gt_i
    for i, (gt_label, gt_box) in enumerate(zip(gt_labels, gt_boxes)):
        for j, (logit, box) in enumerate(zip(logits, boxes)):
            loss_matrix[i,j] = match_loss(logit.detach(), box.detach(), gt_label, gt_box, l1_lambda, IoU_lambda)

    loss = 0
    opt_matrix_indicies = linear_sum_assignment(loss_matrix, maximize=False)
    row, col = opt_matrix_indicies
    for i, j in zip(row.tolist(), col.tolist()):
        loss += match_log_loss(logits[j], boxes[j], gt_labels[i], gt_boxes[i], l1_lambda, IoU_lambda)

    for i in range(100):
        if i not in col:
            loss += F.cross_entropy(logits[i], torch.tensor(20))
    
    return loss













if __name__ == "__main__":

    

    resnet50 = models.resnet50(weights = ResNet50_Weights.IMAGENET1K_V2)
    detr = nanoDETR(resnet50 = resnet50)    

    # predictions
    img, target = dataset[0]
    _, H, W = img.shape
    frac = torch.tensor([W,H,W,H])
    gt_boxes = target['boxes'].data / frac # 0-1 scale
    gt_labels = target['labels']
    

    class_out, bbox_out = detr(img)
    logits = class_out.squeeze(0)
    boxes = bbox_out.squeeze(0)

    # loss 
    l1_lambda = 1
    IoU_lambda = 1
    total_loss = loss(l1_lambda, IoU_lambda, logits, boxes, gt_labels, gt_boxes) 
    print("total loss:", total_loss.item())

    # training loop
    optimizer = torch.optim.AdamW(detr.parameters(), lr = 3e-4, weight_decay = 1e-4)
    detr.train()
    for i in range(200):
        optimizer.zero_grad()
        class_out, bbox_out = detr(img)
        logits = class_out.squeeze(0)
        boxes = bbox_out.squeeze(0)
        total_loss = loss(l1_lambda, IoU_lambda, logits, boxes, gt_labels, gt_boxes) 
        total_loss.backward()
        optimizer.step()
        
        pred_classes = logits.argmax(dim=-1)
        noobj_ratio = (pred_classes == 20).sum()
        print(
            f"step {i:03d} | loss: {total_loss.item():.2f} | "
            f"no-object classes: {noobj_ratio:.2f} | "
        )

        savepath = f"./img_training/iter {i}.png"
        utils.plot_pred(img, logits, boxes, savepath=savepath)