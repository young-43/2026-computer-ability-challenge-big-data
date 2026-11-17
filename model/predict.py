"""Frozen R27 source-field30 original-JPEG serial B1 inference, no CAM or cache."""
import hashlib,json,math,time
from pathlib import Path

def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(2**20),b''):h.update(block)
    return h.hexdigest()

def validate_config(c):
    expected=dict(fraction=.30,fusion_global_weight=.5,threshold=.5,global_epoch=7,local_epoch=12,locator_epoch=50,
                  roi_policy='R23_center_uniform_sourcefield30',padding='black_keep_center_and_source_scale')
    if any(c.get(k)!=v for k,v in expected.items()):raise ValueError('Frozen R27 inference config changed')

def fuse_probability(global_logit,local_logit,fallback):
    if not all(math.isfinite(v) for v in (global_logit,local_logit)):raise ValueError('Nonfinite classifier output')
    z=global_logit if fallback else .5*(global_logit+local_logit)
    return 1/(1+math.exp(-z)) if z>=0 else math.exp(z)/(1+math.exp(z))

def validate_classifier_checkpoint(ck,key,epoch):
    # Original R16 raw7 predates partial_epoch; its exact SHA is checked separately.
    partial=ck.get('partial_epoch',False) if key=='global' else ck.get('partial_epoch',True)
    if ck['epoch']!=epoch or partial or ck['branch']!='raw' or ck['config']['use_age']:raise ValueError('Wrong frozen classifier')
    if key=='local' and ck['config']['roi_policy']!='R23_center_uniform_sourcefield30':raise ValueError('Wrong local crop policy')

def verify_package(root):
    root=Path(root).resolve();m=json.loads((root/'manifest.json').read_text(encoding='utf-8'));total=0
    for name,v in m.items():
        p=(root/name).resolve()
        if not p.is_relative_to(root) or not p.is_file() or p.stat().st_size!=v['bytes'] or digest(p)!=v['sha256']:raise ValueError('Package identity mismatch: '+name)
        total+=v['bytes']
    return total

class Inference:
    def __init__(self,root=None,device='cuda'):
        import torch
        from glaucoma.models import build_model
        from glaucoma.r23_model import DiscLocator
        self.root=Path(root or Path(__file__).resolve().parent);verify_package(self.root)
        self.config=json.loads((self.root/'config.json').read_text(encoding='utf-8'));validate_config(self.config)
        self.device=torch.device(device);torch.set_num_threads(8);torch.backends.cudnn.benchmark=False
        for key,epoch in (('global',7),('local',12)):
            e=self.config['weights'][key];p=self.root/e['path']
            if digest(p)!=e['sha256']:raise ValueError('Frozen weight changed')
            ck=torch.load(p,map_location='cpu',weights_only=True)
            validate_classifier_checkpoint(ck,key,epoch)
            m=build_model(ck['config'],False);m.load_state_dict(ck['state_dict'],strict=True)
            setattr(self,key+'_model',m.to(self.device).eval().requires_grad_(False))
        e=self.config['weights']['locator'];p=self.root/e['path']
        if digest(p)!=e['sha256']:raise ValueError('Frozen locator changed')
        ck=torch.load(p,map_location='cpu',weights_only=True)
        if ck['epoch']!=50 or ck['updates']!=300:raise ValueError('Wrong frozen locator')
        self.locator=DiscLocator();self.locator.load_state_dict(ck['model'],strict=True)
        self.locator=self.locator.to(self.device).eval().requires_grad_(False)

    def predict(self,path):
        import torch
        from PIL import Image
        from glaucoma.local_region import prepare_shared,image_tensor
        from glaucoma.r23_model import center_from_logits
        from glaucoma.r23_geometry import unmap_center
        from glaucoma.r24_protocol import locator_input,locator_array
        from glaucoma.r27_geometry import uniform_roi,crop_uniform
        cuda=self.device.type=='cuda'
        if cuda:torch.cuda.synchronize(self.device)
        begin=time.perf_counter()
        with torch.inference_mode():
            with Image.open(path) as im:
                ow,oh=im.size
                if im.getexif().get(274,1) in (5,6,7,8):ow,oh=oh,ow
            full,g,decoded=prepare_shared(path)
            field=tuple(round(v*s) for v,s in zip(g['field'],(decoded.width,decoded.height,decoded.width,decoded.height)))
            with torch.backends.cudnn.flags(enabled=True,benchmark=False,deterministic=False):
                with torch.autocast(self.device.type,dtype=torch.float16,enabled=cuda):
                    gl=float(self.global_model(image_tensor(full)[None].to(self.device),torch.zeros((1,2),device=self.device)).float().item())
            pil,lg=locator_input(decoded)
            with torch.backends.cudnn.flags(enabled=True,benchmark=False,deterministic=True):
                with torch.autocast(self.device.type,enabled=False):
                    heat=self.locator(torch.from_numpy(locator_array(pil))[None].to(self.device));points,_=center_from_logits(heat)
            point=points[0].cpu().tolist();center=unmap_center(point,lg)
            roi=uniform_roi((ow,oh),decoded.size,field,center);crop=crop_uniform(decoded,roi)
            with torch.backends.cudnn.flags(enabled=True,benchmark=False,deterministic=False):
                with torch.autocast(self.device.type,dtype=torch.float16,enabled=cuda):
                    ll=float(self.local_model(image_tensor(crop)[None].to(self.device),torch.zeros((1,2),device=self.device)).float().item())
            p=fuse_probability(gl,ll,roi['fallback'])
            if cuda:torch.cuda.synchronize(self.device)
            return dict(global_logit=gl,local_logit=ll,fused_logit=gl if roi['fallback'] else .5*(gl+ll),probability=p,label=int(p>=.5),
                        roi_box=[roi[k] for k in ('px0','py0','px1','py1')],decoded_size=list(decoded.size),original_size=[ow,oh],field=list(field),
                        locator_input_center=point,locator_center=list(center),padding_fraction=roi['padding_fraction'],fraction=.30,
                        fallback=roi['fallback'],fallback_reason=roi['fallback_reason'],crop_rgb_sha256=hashlib.sha256(crop.tobytes()).hexdigest(),
                        elapsed_ms=1000*(time.perf_counter()-begin))
