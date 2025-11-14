from nanoDETR import nanoDETR

import torch
import torchvision.models as models
from torchvision import datasets
from torchvision.datasets import wrap_dataset_for_transforms_v2
from torchvision.models import ResNet50_Weights  
from torchvision.transforms import v2
from torchvision.ops import generalized_box_iou_loss # same loss / paper ref. as in original DETR paper
from torchvision.ops import box_convert
import torch.nn.functional as F
import numpy as np
from scipy.optimize import linear_sum_assignment
from torch.optim.lr_scheduler import StepLR
import matplotlib.pyplot as plt
from datetime import datetime
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

import logging

# Configure basic logging once, at top-level
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[             # to console
        logging.FileHandler("training.log", mode="w")  # to file
    ],
)

logger = logging.getLogger(__name__)




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



def match_loss(logits, bbox, gt_label, gt_bbox, l1_lambda, IoU_lambda, log_prob=False):
    # class loss
    class_loss = -logits.softmax(dim=-1)[gt_label] # original paper want raw probability
  
    # bbox loss
    bbox = box_convert(boxes = bbox, in_fmt = 'cxcywh', out_fmt = 'xyxy')
    IoU_loss = generalized_box_iou_loss(bbox, gt_bbox, reduction = 'mean') # reduction not needed since single box, but keep for consistency
    l1_loss = F.l1_loss(bbox, gt_bbox)
    bbox_loss = (IoU_lambda * IoU_loss) + (l1_lambda  * l1_loss)
    
    total_loss = class_loss + bbox_loss
    return total_loss.item()




def loss(l1_lambda, IoU_lambda, logits, boxes, gt_labels, gt_boxes):
    
    # Step 1: Hungarian matching to find optimal permutation / assignment
    # construct loss matrix A where A_ij = L_match(y_i, y_hat_j)
    # size of matrix = (num_gt_boxes, 100)
    ntargets = gt_boxes.shape[0]
    loss_matrix = np.zeros((ntargets, 100)) 
    for i, (gt_label, gt_box) in enumerate(zip(gt_labels, gt_boxes)):
        for j, (logit, box) in enumerate(zip(logits, boxes)):
            loss_matrix[i,j] = match_loss(logit.detach(), box.detach(), gt_label, gt_box, l1_lambda, IoU_lambda) 
    
    opt_matrix_indicies = linear_sum_assignment(loss_matrix, maximize=False) 
    row, col = opt_matrix_indicies
    
    # Step 2: compute loss for optimal assignment
    # nll loss for classification
    targets = torch.ones(100, dtype=torch.uint8) * 20  # 20 is no-object class
    targets[col] = gt_labels[row].to(targets.dtype)
    weights = torch.ones(21)
    weights[-1] = 0.1  # lower weight for no-object class
    nll_loss = F.cross_entropy(logits, targets, weight= weights, reduction='mean') 

    # bbox loss for matched boxes
    perm_boxes = boxes[col]
    perm_boxes = box_convert(boxes = perm_boxes, in_fmt = 'cxcywh', out_fmt = 'xyxy')
    IoU_loss = generalized_box_iou_loss(perm_boxes, gt_boxes, reduction = 'sum') # now reduction needed
    l1_loss = F.l1_loss(perm_boxes, gt_boxes, reduction='sum')
    bbox_loss = (IoU_lambda * IoU_loss) + (l1_lambda  * l1_loss)
    bbox_loss /= ntargets

    # total loss
    total_loss = nll_loss + bbox_loss

   
    return total_loss, nll_loss.item(), bbox_loss.item()

    

# --- hyperparams
l1_lambda = 5 # loss. l1 and iou hyperparams from original detr paper
IoU_lambda = 2

nepochs = 200
batch_size = 2


def collate_fn(batch):
    return batch

small_dataset = Subset(dataset, indices=[0, 21]) # for overfitting
dl = DataLoader(small_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)


if __name__ == "__main__":

    
    resnet50 = models.resnet50(weights = ResNet50_Weights.IMAGENET1K_V2)
    detr = nanoDETR(resnet50 = resnet50)    
    #detr = torch.load('saved_models/epoch7_20251107_153657.pth', weights_only=False)  
    total, trainable = 0, 0
    for n,p in detr.named_parameters():
        total += p.numel()
        if p.requires_grad:
            trainable += p.numel()
    print(f"Trainable params: {trainable}/{total}")

    # training setup
    optimizer = torch.optim.AdamW(detr.parameters(), lr = 3e-4, weight_decay = 1e-4)
    scheduler = StepLR(optimizer, step_size=75, gamma=0.1)  
    detr.train()

    total_loss_arr = []
    class_loss_arr = []
    box_loss_arr = []
    for iepoch in range(0, nepochs):
        
        for batch in dl:

            epoch_total_loss, epoch_nll_loss, epoch_box_loss = 0,0,0

            for img, target in batch:    
                _, H, W = img.shape
                frac = torch.tensor([W,H,W,H])
                gt_boxes = target['boxes'].data / frac # 0-1 scale
                gt_labels = target['labels']
        
                class_out, bbox_out = detr(img)
                logits = class_out.squeeze(0)
                boxes = bbox_out.squeeze(0)
                total_loss, nll_loss, box_loss = loss(l1_lambda, IoU_lambda, logits, boxes, gt_labels, gt_boxes) 
                epoch_total_loss += total_loss
                epoch_nll_loss += nll_loss
                epoch_box_loss += box_loss

            epoch_total_loss /= len(batch); epoch_nll_loss /= len(batch); epoch_box_loss /= len(batch)
            optimizer.zero_grad()
            epoch_total_loss.backward()
            optimizer.step()

            logger.info(f"Epoch {iepoch + 1} Loss {epoch_total_loss:.3f} nll_loss {epoch_nll_loss:.3f} box_loss {epoch_box_loss:.3f}")    

        scheduler.step()
        
        total_loss_arr.append(epoch_total_loss.item()); class_loss_arr.append(epoch_nll_loss); box_loss_arr.append(epoch_box_loss)
        print(f"epoch {iepoch:03}, total_loss: {epoch_total_loss.item():.3f}, class loss: {epoch_nll_loss:.3f}, bbox loss: {epoch_box_loss:.3f}")
        
        #torch.save(detr, f'saved_models/epoch{iepoch}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pth')
    
    print("Training complete. Model saved.")
    
   

