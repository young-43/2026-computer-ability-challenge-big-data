"""CAM-free fixed source-field geometry. PIL supplies black outside-source pixels."""
import math
from PIL import ImageOps, Image

def uniform_roi(original_size,decoded_size,field,center,fraction=.30):
    if fraction!=.30:raise ValueError('Frozen fraction is 0.30')
    ow,oh=map(int,original_size);dw,dh=map(int,decoded_size)
    if min(ow,oh,dw,dh)<=0:raise ValueError('Positive dimensions required')
    fx0,fy0,fx1,fy1=map(int,field)
    field_valid=0<=fx0<fx1<=dw and 0<=fy0<fy1<=dh
    point_valid=center is not None and len(center)==2 and all(math.isfinite(float(v)) and 0<=float(v)<=1 for v in center)
    base=dict(original_width=ow,original_height=oh,decoded_width=dw,decoded_height=dh,field=list(map(int,field)),fraction=.30)
    if not field_valid or not point_valid:
        box=(fx0,fy0,fx1,fy1) if field_valid else (0,0,dw,dh)
        return dict(base,px0=box[0],py0=box[1],px1=box[2],py1=box[3],fallback=True,fallback_reason='invalid_geometry_global_only',padding_fraction=0.)
    side=.30*max((fx1-fx0)*ow/dw,(fy1-fy0)*oh/dh)
    bw=max(1,round(side*dw/ow));bh=max(1,round(side*dh/oh))
    x0=round(float(center[0])*dw-bw/2);y0=round(float(center[1])*dh-bh/2)
    inside=max(0,min(dw,x0+bw)-max(0,x0))*max(0,min(dh,y0+bh)-max(0,y0))
    return dict(base,px0=x0,py0=y0,px1=x0+bw,py1=y0+bh,fallback=False,fallback_reason='none',
                locator_x=float(center[0]),locator_y=float(center[1]),source_side=side,padding_fraction=1-inside/(bw*bh))

def crop_uniform(decoded,row,size=384):
    if decoded.size!=(int(row['decoded_width']),int(row['decoded_height'])):raise ValueError('Decoded dimensions changed')
    box=tuple(int(row[k]) for k in ('px0','py0','px1','py1'))
    if box[0]>=box[2] or box[1]>=box[3]:raise ValueError('Nonempty crop required')
    return ImageOps.pad(decoded.crop(box),(size,size),method=Image.Resampling.BICUBIC,color=(0,0,0))
