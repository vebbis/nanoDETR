from torchvision.transforms import v2 
import torch
from torchvision.utils import draw_bounding_boxes
from torchvision.ops import box_convert

def v2show(img):
    mean= torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
    std=torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)
    img = img.data
    img = img*std + mean

    converter = v2.ToPILImage()
    pil_img = converter(img)
    pil_img.show()


# haven't checked if these are official. website down
itol = [
    "aeroplane", "bicycle", "bird", "boat", "bottle",
    "bus", "car", "cat", "chair", "cow",
    "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor"
]
def plot(sample):
    # img is normalized, so have to unnormalize
    mean= torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
    std=torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)
    img, target = sample
    img = img.data
    img = img*std + mean
    toimg = v2.ToPILImage()
    labels = [str(itol[i-1]) for i in target['labels']]
    toimg(draw_bounding_boxes(img, target['boxes'].data, width = 3, labels = labels)).show()

def plot_pred(img, logits, boxes, savepath=None):
    # img is normalized, so have to unnormalize
    mean= torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
    std=torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)
    img = img.data
    img = img*std + mean
    raw_img = img.clone()
    toimg = v2.ToPILImage()
        
    logits = logits.argmax(dim = -1, keepdim = True)
    pred_boxes = []
    pred_labels = []
    for i in range(100):
        predicted_label = logits[i]
        if predicted_label != 20:
            pred_labels.append(predicted_label)
            pred_boxes.append(boxes[i])
    
    if len(pred_boxes) == 0:
        print("no pred boxes")
        if savepath is not None:
            raw_img = toimg(raw_img)
            raw_img.save(savepath)
        return
    
    pred_boxes = torch.stack(pred_boxes, dim = 0)
    pred_boxes = box_convert(boxes = pred_boxes, in_fmt = 'cxcywh', out_fmt = 'xyxy')
    _, H, W = img.shape
    frac = torch.tensor([W,H,W,H])
    pred_boxes = pred_boxes * frac # scale to image size
    labels = [str(itol[i-1]) for i in pred_labels]
    img = toimg(draw_bounding_boxes(img, pred_boxes, width = 3, labels = labels))
    
    if savepath is not None:
        img.save(savepath)
    else:
        img.show()