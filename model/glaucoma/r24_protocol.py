"""Frozen locator preprocessing and integer ROI replacement, no model dependency."""
import math
import numpy as np
from PIL import Image
from .r23_geometry import letterbox_geometry

def locator_input(decoded):
    if decoded.mode!="RGB":raise ValueError("Shared RGB decode required")
    g=letterbox_geometry(decoded.size,256)
    small=decoded.resize((g["width"],g["height"]),Image.Resampling.BICUBIC)
    out=Image.new("RGB",(256,256));out.paste(small,(g["ox"],g["oy"]))
    return out,g

def locator_array(pil):
    if pil.mode!="RGB" or pil.size!=(256,256):raise ValueError("Exact locator RGB256 required")
    a=np.asarray(pil,dtype=np.float32)/255
    a=(a-np.array([.485,.456,.406],np.float32))/np.array([.229,.224,.225],np.float32)
    return a.transpose(2,0,1).copy()

def replace_roi(parent,normalized):
    r=dict(parent)
    for k in ("decoded_width","decoded_height","px0","py0","px1","py1"):r[k]=int(r[k])
    w,h=r["decoded_width"],r["decoded_height"]
    if not (0<=r["px0"]<r["px1"]<=w and 0<=r["py0"]<r["py1"]<=h):raise ValueError("Parent dimensions/bounds required")
    fallback=r["fallback"]
    if fallback not in (True,False,"True","False"):raise ValueError("Explicit parent fallback required")
    if fallback is True or fallback=="True":
        r.update(fallback=True,fallback_reason="parent_global_only",locator_valid=False,locator_x=None,locator_y=None)
        return r
    valid=normalized is not None and len(normalized)==2 and all(math.isfinite(float(x)) and 0<=float(x)<=1 for x in normalized)
    if not valid:
        r.update(px0=0,py0=0,px1=w,py1=h,fallback=True,fallback_reason="invalid_locator_global_only",locator_valid=False,locator_x=None,locator_y=None)
        return r
    x,y=map(float,normalized);bw,bh=r["px1"]-r["px0"],r["py1"]-r["py0"]
    x0=max(0,min(w-bw,round(x*w-bw/2)));y0=max(0,min(h-bh,round(y*h-bh/2)))
    r.update(px0=x0,py0=y0,px1=x0+bw,py1=y0+bh,fallback=False,fallback_reason="none",locator_valid=True,locator_x=x,locator_y=y)
    return r

def validate_recipe(s):
    expected=dict(round="R24-LOCATORROI-01",seed=20261026,local_epochs=15,lr_horizon=20,batch_size=32,fusion_global_weight=.5,threshold=.5,
                  early_stopping=False,total_seconds=None,official_valid=False,test=False,confidence_threshold=None)
    if any(s.get(k)!=v for k,v in expected.items()):raise ValueError("Frozen single comparison recipe changed")
