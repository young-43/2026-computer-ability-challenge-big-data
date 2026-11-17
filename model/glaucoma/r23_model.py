"""Lightweight ImageNet-initialized heatmap locator; no glaucoma weights."""
import torch
from torch import nn
from torchvision.models import mobilenet_v3_small

class DiscLocator(nn.Module):
    def __init__(self,weights_path=None):
        super().__init__()
        base=mobilenet_v3_small(weights=None)
        if weights_path is not None:
            state=torch.load(weights_path,map_location="cpu",weights_only=True)
            base.load_state_dict(state,strict=True)
        self.backbone=base.features[:9]
        self.head=nn.Sequential(nn.Conv2d(48,32,3,padding=1),nn.GroupNorm(4,32),nn.Hardswish(),nn.Conv2d(32,1,1))
    def train(self,mode=True):
        super().train(mode)
        for m in self.backbone.modules():
            if isinstance(m,nn.BatchNorm2d):m.eval()
        return self
    def forward(self,x):
        logits=self.head(self.backbone(x)).squeeze(1)
        if tuple(logits.shape[1:])!=(16,16):raise ValueError("Input256/stride16 contract required")
        return logits
def center_from_logits(logits):
    """Subpixel mean over a 3x3 neighborhood of the argmax; edge coordinates."""
    probabilities=logits.flatten(1).softmax(1).reshape_as(logits)
    predictions=[]
    for p in probabilities:
        i=int(p.argmax());y,x=divmod(i,16)
        y0,y1=max(0,y-1),min(16,y+2);x0,x1=max(0,x-1),min(16,x+2)
        patch=p[y0:y1,x0:x1];patch=patch/patch.sum()
        yy,xx=torch.meshgrid(torch.arange(y0,y1,device=p.device,dtype=p.dtype),torch.arange(x0,x1,device=p.device,dtype=p.dtype),indexing="ij")
        predictions.append(torch.stack((((xx+.5)*patch).sum()*16,((yy+.5)*patch).sum()*16)))
    return torch.stack(predictions),probabilities.amax((1,2))
