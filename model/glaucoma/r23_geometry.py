"""Numpy-only, edge-coordinate geometry for the small fit-only locator."""
import math
import numpy as np
def letterbox_geometry(dimensions,size=256):
    w,h=dimensions
    if min(w,h,size)<=0:raise ValueError("Positive dimensions required")
    scale=min(size/w,size/h);nw=max(1,round(w*scale));nh=max(1,round(h*scale))
    return dict(width=nw,height=nh,ox=(size-nw)//2,oy=(size-nh)//2,size=size)
def map_center(normalized,g,hflip=False,vflip=False,translate=(0,0)):
    x,y=normalized
    if not all(math.isfinite(v) and 0<=v<=1 for v in (x,y)):raise ValueError("Finite normalized point required")
    px=x*g["width"]+g["ox"];py=y*g["height"]+g["oy"]
    if hflip:px=g["size"]-px
    if vflip:py=g["size"]-py
    return px+translate[0],py+translate[1]
def unmap_center(center,g):return ((center[0]-g["ox"])/g["width"],(center[1]-g["oy"])/g["height"])
def heatmap_target(center,size=256,grid=16,sigma=1.5):
    x,y=center
    if not all(math.isfinite(v) and 0<=v<size for v in (x,y)):raise ValueError("Target must remain in the input image")
    yy,xx=np.mgrid[:grid,:grid];cx=x/size*grid-.5;cy=y/size*grid-.5
    target=np.exp(-((xx-cx)**2+(yy-cy)**2)/(2*sigma**2));target/=target.sum()
    return target.astype(np.float32)
def validate_partition(rows):
    seen=set();sides={"loc_train":set(),"loc_check":set()}
    for row in rows:
        if row["role"]!="r21_fit" or row["localization_role"] not in sides or row["image_id"] in seen:raise ValueError("Unique fit-only localization roles required")
        seen.add(row["image_id"]);sides[row["localization_role"]].add(row["sha256"])
    if not all(sides.values()) or sides["loc_train"]&sides["loc_check"]:raise ValueError("Train/check source hash overlap or empty side")
