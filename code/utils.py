"""Portable original image-ID mapping; no fitting or label lookup."""
from pathlib import Path
import csv
def image_files(folder,expected):
    folder=Path(folder).resolve();paths=sorted((p for p in folder.iterdir() if p.is_file() and p.suffix.casefold() in ('.jpg','.jpeg')),key=lambda p:p.name.casefold())
    if len(paths)!=expected or len({p.name.casefold() for p in paths})!=expected:raise ValueError(f'Expected exactly {expected} unique JPEG images')
    if any(not p.resolve().is_relative_to(folder) for p in paths):raise ValueError('Image outside input folder')
    return paths
def load_images(folder,metadata,expected):
    with Path(metadata).open(encoding='utf-8-sig',newline='') as f:
        reader=csv.DictReader(f)
        if 'Image_name' not in (reader.fieldnames or []):raise ValueError('Metadata requires Image_name')
        names=[r['Image_name'] for r in reader]
    if len(names)!=expected or len({n.casefold() for n in names})!=expected:raise ValueError('Full unique metadata required')
    folder=Path(folder).resolve();available={}
    for p in folder.iterdir():
        if p.is_file() and p.suffix.casefold() in ('.jpg','.jpeg'):
            if p.name.casefold() in available:raise ValueError('Ambiguous image filename')
            available[p.name.casefold()]=p
    paths=[]
    for n in names:
        if not n or '/' in n or '\\' in n or Path(n).name!=n:raise ValueError('Image_name must be basename')
        p=available.get(n.casefold())
        if p is None or not p.resolve().is_relative_to(folder):raise ValueError('Image missing or outside folder: '+n)
        paths.append(p)
    return names,paths
