from torchvision.transforms import v2 
import torch

def v2show(img):
    mean= torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
    std=torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)
    img = img.data
    img = img*std + mean

    converter = v2.ToPILImage()
    pil_img = converter(img)
    pil_img.show()