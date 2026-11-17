"""Official material entry: unchanged original R27 engine and fixed precision."""
from pathlib import Path
import argparse,importlib.util,json,os,random,sys
ROOT=Path(__file__).resolve().parents[1]
def configure_runtime():
    import numpy as np
    import torch
    if os.environ.get('NVIDIA_TF32_OVERRIDE') not in (None,'1'):raise RuntimeError('External TF32 override changes original R27 runtime')
    if os.environ.get('CUBLAS_WORKSPACE_CONFIG'):raise RuntimeError('Remove external CUBLAS_WORKSPACE_CONFIG to preserve original exported runtime')
    random.seed(20261026);np.random.seed(20261026);torch.manual_seed(20261026)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(20261026)
    torch.set_num_threads(8);torch.use_deterministic_algorithms(False)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=True
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=False
def load_model(device='cuda'):
    configure_runtime();sys.path.insert(0,str(ROOT/'model'))
    spec=importlib.util.spec_from_file_location('original_r27_frozen',ROOT/'model/predict.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    return m.Inference(ROOT/'model',device)
def predict_one(engine,path):return engine.predict(Path(path))
def main():
    from utils import load_images
    from submission_io import write_submission
    p=argparse.ArgumentParser(description='Original R27 global/local50:50; raw7/raw12/R23raw50; no expert')
    p.add_argument('--images-dir',required=True);p.add_argument('--output',required=True)
    p.add_argument('--metadata',required=True);p.add_argument('--expected-count',type=int,required=True)
    p.add_argument('--device',default='cuda',choices=('cuda','cpu'));a=p.parse_args()
    if Path(a.output).exists():raise FileExistsError('Never overwrite existing CSV')
    names,paths=load_images(a.images_dir,a.metadata,a.expected_count);engine=load_model(a.device)
    predictions=[]
    for i,(n,path) in enumerate(zip(names,paths),1):
        predictions.append(dict(image_id=n,**predict_one(engine,path)))
        if i==1 or i%100==0 or i==len(names):print(json.dumps(dict(done=i,total=len(names))),flush=True)
    write_submission(a.output,names,predictions,a.expected_count)
if __name__=='__main__':main()
