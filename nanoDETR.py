import torch
import torchvision.models as models
from torchvision import datasets, transforms as T
from torchvision.datasets import wrap_dataset_for_transforms_v2
from torchvision.models import ResNet50_Weights  # <-- import this
from torchvision.transforms import v2
from torchvision.utils import draw_bounding_boxes
import torch.nn as nn
import torch.nn.functional as F
import math
import utils




class EncoderHead(nn.Module):
    def __init__(self, hidden_dim = 256, nhead = 8):
        super().__init__()
        self.head_size = hidden_dim // nhead # 32
        self.query = nn.Linear(hidden_dim, self.head_size, bias = False) # 256 -> 32
        self.key = nn.Linear(hidden_dim, self.head_size, bias = False)
        self.value = nn.Linear(hidden_dim, self.head_size, bias = False)
        
    def forward(self, x, positional_encoding):
        # project into separate spaces
        q = self.query(x + positional_encoding) # (B, H*W, head_size) 
        k = self.key(x + positional_encoding)
        v = self.value(x)
        
        # attention scores
        scores = q @ k.transpose(-2,-1)
        scores /= math.sqrt(self.head_size) # print(scores.std()) will be ish 0.4 => breaks gaussian assumption
        scores = F.softmax(scores, dim = -1)
        out = scores @ v # (B, H*W, head_size) 

        return out

class EncoderLayer(nn.Module):
    def __init__(self, hidden_dim = 256, nhead = 8):
        super().__init__()
        self.encoder_heads = nn.ModuleList([EncoderHead() for _ in range(nhead)])
        self.cat_proj = nn.Linear(hidden_dim, hidden_dim, bias = False)

    def forward(self, x, positional_encoding):
        cat = torch.cat([head(x, positional_encoding) for head in self.encoder_heads], dim = -1) # (B, HW, head_size) -> (B, HW, hidden_dim)
        out = self.cat_proj(cat)
        return out

class EncoderBlock(nn.Module):
    def __init__(self, hidden_dim = 256, nhead = 8):
        super().__init__()
        self.encoder_layer = EncoderLayer(hidden_dim= hidden_dim, nhead = nhead)

        # layer norm
        self.gamma = nn.Parameter(torch.ones(1, 1, hidden_dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.eps = 1e-5

        # ffn
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim*4)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim*4, hidden_dim)

    def forward(self, x, positional_encoding):
        # attention layer
        x_res = x
        mean = x_res.mean(dim = -1, keepdim = True) # for layernorm
        var = x_res.var(dim = -1, correction = 0, keepdim = True)
        x_res = x_res - mean
        x_res = x_res / torch.sqrt(var + self.eps)
        x_res = x_res*self.gamma + self.beta
        x_res = self.encoder_layer(x_res, positional_encoding) # (B, HW, hidden_dim)
        x = x + x_res # residual connection

        # compute layer
        x_res = x
        x_res = self.layer_norm(x_res)
        x_res = self.fc1(x_res)
        x_res = self.relu(x_res)
        x_res = self.fc2(x_res)
        x = x + x_res

        return x

class Encoder(nn.Module):
    def __init__(self, num_encoder_layers = 6, hidden_dim = 256, nhead = 8):
        super().__init__()
        self.layers = nn.ModuleList([EncoderBlock(hidden_dim=hidden_dim, nhead=nhead) for _ in range(num_encoder_layers)])

    def forward(self, x, positional_encoding):
        for layer in self.layers:
            x = layer(x, positional_encoding)
        return x
        
    
class MHAttention(nn.Module):
    def __init__(self, embed_dim, nhead):
        super().__init__()
        self.nhead = nhead
        self.head_size = embed_dim // nhead
        self.query = nn.Linear(embed_dim, embed_dim, bias = False)
        self.key = nn.Linear(embed_dim, embed_dim, bias = False)
        self.value = nn.Linear(embed_dim, embed_dim, bias = False)
        self.projection = nn.Linear(embed_dim, embed_dim, bias = False)
        

    def forward(self, query, key, value):
        
        B, qT, C = query.shape
        _, kT, _ = key.shape
        
        q = self.query(query)
        k = self.key(key)
        v = self.value(value)
        
        # TODO: go over pen and paper for reshaping q, k, v
        q = q.view(B, qT, self.nhead, self.head_size).transpose(1,2) # (B, nhead, qT, head_size)
        k = k.view(B, kT, self.nhead, self.head_size).transpose(1,2) # (B, nhead, kT, head_size)
        v = v.view(B, kT, self.nhead, self.head_size).transpose(1,2)
            
        scores = q @ k.transpose(-2,-1) # (B, nhead, qT, head_size) @ (B, nhead, head_size, kT) ---> # (B, nhead, qT, kT)
        scores /= math.sqrt(self.head_size)
        scores = F.softmax(scores, dim = -1)
        out = scores @ v # (B, nhead, qT, kT) @ (B, nhead, kT, head_size) ---> (B, nhead, qT, head_size)
        out = out.transpose(1,2).flatten(2) # (B, nhead, qT, head_size) ---> (B, qT, nhead, head_size) ---> (B, qT, hidden_dim)
        out = self.projection(out)
        
        return out


class DecoderBlock(nn.Module):
    def __init__(self, hidden_dim = 256, nhead = 8):
        super().__init__()

        self.layer_norm_1 = nn.LayerNorm(hidden_dim)
        self.multihead_selfattention = MHAttention(embed_dim = hidden_dim, nhead = nhead)
        self.layer_norm_2 = nn.LayerNorm(hidden_dim)
        self.multihead_crossattention = MHAttention(embed_dim = hidden_dim, nhead = nhead)
        self.layer_norm_3 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(nn.Linear(hidden_dim, hidden_dim*4), nn.ReLU(), nn.Linear(hidden_dim*4, hidden_dim)) 

        
    def forward(self, x, memory, memory_pos_encoding, query_pos_encoding):
        # self-attention
        x_res = self.layer_norm_1(x)
        q = x_res + query_pos_encoding
        k = x_res + query_pos_encoding
        v = x_res 
        x_res = self.multihead_selfattention(query = q, key = k, value = v)
        x = x + x_res

        # cross-attention
        x_res = self.layer_norm_2(x)
        q = x_res + query_pos_encoding
        k = memory + memory_pos_encoding
        v = memory
        x_res = self.multihead_crossattention(query = q, key = k, value = v)
        x = x + x_res

        #ffn
        x_res = self.layer_norm_3(x)
        x_res = self.ffn(x_res)
        x = x + x_res

        return x



class nanoDETR(nn.Module):
    def __init__(self, resnet50, ntokens = 224, nlayers = 6, nhead = 8, hidden_dim = 256, nqueries = 100, num_classes = 20, batch_size = 1):
        super().__init__()
        self.batch_size = batch_size
        self.nlayers = nlayers
        self.nhead = nhead
        self.hidden_dim = hidden_dim
        
        # backbone
        self.resnet50 = resnet50
        self.backbone = nn.Sequential(*list(resnet50.children())[:-2])
        self.project = nn.Conv2d(in_channels = 2048, out_channels = hidden_dim, kernel_size = 1, bias=False)
        
        # build encoder
        self.encoder_pos_encode = nn.Parameter(torch.randn(ntokens, hidden_dim)) # (nqueries, hidden_dim) -> will be broadcasted to (B, nqueries, hidden_dim)
        self.encoder = Encoder(num_encoder_layers = nlayers, hidden_dim = hidden_dim, nhead = nhead)

        # build decoder
        self.nqueries = nqueries
        self.query_pos_encoding = nn.Parameter(torch.randn((nqueries, hidden_dim))) # (nqueries, hidden_dim) -> will be broadcasted to (B, nqueries, hidden_dim)
        self.layers = nn.ModuleList([DecoderBlock(hidden_dim = self.hidden_dim, nhead = self.nhead) for _ in range(self.nlayers)])

        # prediction heads
        self.class_ffn = nn.Linear(hidden_dim, num_classes + 1) # +1 for no-object class
        self.bbox_ffn = nn.Sequential(nn.Linear(hidden_dim, hidden_dim),
                                      nn.ReLU(),
                                      nn.Linear(hidden_dim, hidden_dim),
                                      nn.ReLU(),
                                      nn.Linear(hidden_dim, 4),
                                      nn.Sigmoid()) # normalize bbox to [0,1]
        

    def forward(self, x):
        # expect x to be raw img from resnet50, e.g. x.shape = (B, C, H, W) = (1, 2048, 14, 16)
        # we assume ntokens when we overfit to one img to check for bugs. It will be made general => cannot use fixed-size feat.dim embedding
        assert x.shape == (3, 442, 500), f'image shape is {x.shape}, but should be (3, 442, 500)'

        # backbone
        with torch.no_grad():
            x = self.backbone(x.unsqueeze(0))
        x = self.project(x)
        x = x.flatten(2) # (B, hidden_dim, T)
        x = x.transpose(-2,-1) # (B, T, hidden_dim)
        x = x.contiguous() # transpose will be permanent

        # encoder
        memory = self.encoder(x, self.encoder_pos_encode)

        # decoder
        queries = torch.zeros((self.batch_size, self.nqueries, self.hidden_dim), device=x.device, dtype=x.dtype)
        for layer in self.layers:
            queries = layer(x = queries, 
                            memory = memory, 
                            memory_pos_encoding = self.encoder_pos_encode, 
                            query_pos_encoding = self.query_pos_encoding)

        # prediction
        # TODO: check pre-activation distributions
        class_out = self.class_ffn(queries) # (B, nqueries, num_classes + 1)
        bbox_out = self.bbox_ffn(queries)   # (B, nqueries, 4)
        return class_out, bbox_out
        

# img,_ = dataset[0]
# detr = nanoDETR(resnet50 = resnet50)
# detr(img).shape

if __name__ == "__main__":
    resnet50 = models.resnet50(weights = ResNet50_Weights.IMAGENET1K_V2)
    detr = nanoDETR(resnet50 = resnet50)
    img = torch.randn(3, 442, 500) 
    class_out, bbox_out = detr(img)
    print("class_out shape:", class_out.shape) # (B, nqueries, num_classes + 1)
    print("bbox_out shape:", bbox_out.shape)   # (B, nqueries, 4)
        


        
    