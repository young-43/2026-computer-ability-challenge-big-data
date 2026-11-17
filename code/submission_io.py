"""Official three-column serialization; never changes frozen model outputs."""
import csv,math
from pathlib import Path

def write_submission(path,names,predictions,expected_count=5855):
    if len(names)!=expected_count or len(predictions)!=expected_count or len(set(n.casefold() for n in names))!=expected_count:raise ValueError('Full unique metadata required')
    rows=[]
    for name,r in zip(names,predictions):
        if not name or r['image_id']!=name:raise ValueError('Metadata order/ID mismatch')
        p=float(r['probability'])
        if not math.isfinite(p) or not 0<=p<=1:raise ValueError('Finite continuous probability required')
        rows.append((name,format(p,'.17g'),int(p>=.5)))
    with Path(path).open('x',encoding='utf-8',newline='') as f:
        w=csv.writer(f,lineterminator='\n');w.writerow(('Image_name','Probability','Pred_Label'));w.writerows(rows)
