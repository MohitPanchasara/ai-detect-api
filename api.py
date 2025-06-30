from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from io import BytesIO
from PIL import Image
import base64, torch, torchvision.transforms.functional as TF
import numpy as np
import random, math, torch.nn as nn, torch.nn.functional as F
import matplotlib.pyplot as plt

app = FastAPI()
app.add_middleware(
  CORSMiddleware,
  allow_origins=["*"],
  allow_methods=["POST"],
  allow_headers=["*"],
)

# << paste here your extract_random_patches, texture_diversity,
#    smash_and_reconstruct, FingerprintExtractor, SimpleDetector >>
# ------------------------------------------------------------------------------
# 1) Smash & Reconstruct (fixed to handle arbitrary #patches)
# ------------------------------------------------------------------------------

import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.transforms import functional as TF
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch

def extract_random_patches(img, patch_size, num_patches):
    W, H = img.size
    out = []
    for _ in range(num_patches):
        x = random.randint(0, W - patch_size)
        y = random.randint(0, H - patch_size)
        out.append(TF.crop(img, y, x, patch_size, patch_size))
    return out

def texture_diversity(patch):
    x = torch.from_numpy(np.array(patch).astype(np.float32))
    if x.dim()==3:
        x = 0.299*x[...,0] + 0.587*x[...,1] + 0.114*x[...,2]
    M = x.shape[0]
    s = (x[:M-1,:] - x[1:M,:]).abs().sum()
    s += (x[:,:M-1] - x[:,1:M]).abs().sum()
    s += (x[:M-1,:M-1] - x[1:M,1:M]).abs().sum()
    s += (x[1:M,:M-1] - x[:M-1,1:M]).abs().sum()
    return s.item()

import math

def smash_and_reconstruct(img, patch_size=64, grid_size=8):
    # get the image’s true size
    W, H = img.size

    # clamp patch_size so that at least one full patch fits
    patch_size = min(patch_size, W, H)

    # now we can safely ask for grid_size**2 patches of size patch_size
    N = grid_size*grid_size
    patches = extract_random_patches(img, patch_size, N)
    scored  = sorted(((texture_diversity(p),p) for p in patches),
                     key=lambda x:x[0])
    half    = N//2
    poor_list = [p for _,p in scored[:half]]
    rich_list = [p for _,p in scored[half:]]

    # now tile exactly floor(sqrt(N))×floor(sqrt(N)) of them
    def tile(plist):
        n = len(plist)
        g = int(math.sqrt(n))
        # shrink to a perfect square
        plist = plist[:g*g]
        rows = []
        it   = iter(plist)
        for _ in range(g):
            row = torch.cat([TF.to_tensor(next(it)) for __ in range(g)], dim=2)
            rows.append(row)
        return torch.cat(rows, dim=1).unsqueeze(0)

    poor_img = tile(poor_list)
    rich_img = tile(rich_list)
    return poor_img, rich_img, patches


# ------------------------------------------------------------------------------
# 2) Fingerprint extractor (unchanged)
# ------------------------------------------------------------------------------
HP_FILTERS = [
    torch.tensor([[ 0,  0,  0],
                  [ 0,  1, -1],
                  [ 0, -1,  1]], dtype=torch.float32),
    torch.tensor([[ -1,  2, -1],
                  [  2, -4,  2],
                  [ -1,  2, -1]], dtype=torch.float32),
]
HP_FILTERS = [k.unsqueeze(0).unsqueeze(0) for k in HP_FILTERS]

class FingerprintExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        weight = torch.cat(HP_FILTERS, dim=0)
        self.register_buffer('weight', weight)
        self.conv = nn.Conv2d(len(HP_FILTERS), len(HP_FILTERS), 3, padding=1, bias=False)
        self.bn   = nn.BatchNorm2d(len(HP_FILTERS))
        self.ht   = nn.Hardtanh(-1,1)
    def forward(self, x_rich, x_poor):
        x_r = x_rich.mean(1,True)
        x_p = x_poor.mean(1,True)
        f_r = F.conv2d(x_r, self.weight, padding=1)
        f_p = F.conv2d(x_p, self.weight, padding=1)
        f_r = self.ht(self.bn(self.conv(f_r)))
        f_p = self.ht(self.bn(self.conv(f_p)))
        return f_r - f_p

# ------------------------------------------------------------------------------
# 3) Simple CNN detector
# ------------------------------------------------------------------------------
class SimpleDetector(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch,64,3,padding=1), nn.ReLU(),
            nn.Conv2d(64,64,3,padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.fc = nn.Linear(64,2)
    def forward(self, x):
        x = self.net(x).view(x.size(0),-1)
        return self.fc(x)

# Initialize once
device = 'cpu'
fp_ext = FingerprintExtractor().to(device).eval()
det   = SimpleDetector(in_ch=len(HP_FILTERS)).to(device).eval()

@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    # 1) load image
    data = await file.read()
    img  = Image.open(BytesIO(data)).convert('RGB')

    # 2) smash/reconstruct
    poor, rich, patches = smash_and_reconstruct(img, patch_size=64, grid_size=8)
    poor, rich = poor.to(device), rich.to(device)

    # 3) fingerprint
    fingerprint = fp_ext(rich, poor)
    # plot fingerprint to PNG
    fig,ax = plt.subplots()
    ax.imshow(fingerprint.squeeze(0).mean(0).cpu().detach().numpy(), cmap='gray')
    ax.axis('off')
    buf = BytesIO(); plt.savefig(buf, format='png'); buf.seek(0)
    fp_b64 = base64.b64encode(buf.read()).decode()

    # 4) detector probs
    logits = det(fingerprint).detach()
    probs = F.softmax(logits, dim=1)[0].cpu().numpy().tolist()

    # 5) return
    return {
      "prob_real": probs[0],
      "prob_fake": probs[1],
      "fingerprint": fp_b64,
    }
