"""DeepWeeds dataset/pipeline utilities for Google Colab."""
from __future__ import annotations
from pathlib import Path
import random
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
import math

NUM_CLASSES = 9
CLASS_NAMES = [
    "Chinee Apple","Lantana","Parkinsonia","Parthenium","Prickly Acacia",
    "Rubber Vine","Siam Weed","Snake Weed","Negatives",
]
IMAGENET_MEAN=(0.485,0.456,0.406)
IMAGENET_STD=(0.229,0.224,0.225)

def load_split(labels_dir, fold=0):
    labels_dir=Path(labels_dir)
    paths=[labels_dir/f"{s}_subset{fold}.csv" for s in ("train","val","test")]
    missing=[str(p) for p in paths if not p.exists()]
    if missing: raise FileNotFoundError("Missing CSV: "+", ".join(missing))
    dfs=tuple(pd.read_csv(p) for p in paths)
    for name,df in zip(("train","val","test"),dfs):
        need={"Filename","Label","Species"}
        if not need.issubset(df.columns): raise ValueError(f"{name} thiếu cột {need-set(df.columns)}")
        if df["Filename"].duplicated().any(): raise ValueError(f"{name} có Filename trùng")
        if not df["Label"].between(0,NUM_CLASSES-1).all(): raise ValueError(f"{name} có Label ngoài 0..8")
    return dfs

def check_split(train_df,val_df,test_df,images_dir):
    dfs={"train":train_df,"val":val_df,"test":test_df}
    sets={k:set(v["Filename"]) for k,v in dfs.items()}
    overlap={f"{a}_{b}":len(sets[a]&sets[b]) for a,b in (("train","val"),("train","test"),("val","test"))}
    if any(overlap.values()): raise ValueError(f"Overlap giữa split: {overlap}")
    union=set.union(*sets.values())
    if len(union)!=17509: raise ValueError(f"Hợp 3 split phải có 17,509 ảnh, thực tế {len(union)}")
    images_dir=Path(images_dir)
    missing=[fn for fn in union if not (images_dir/fn).exists()]
    if missing: raise FileNotFoundError(f"{len(missing)} ảnh trong CSV không tồn tại; ví dụ {missing[:5]}")
    per_class={}
    for k,df in dfs.items():
        counts=df["Label"].value_counts().reindex(range(NUM_CLASSES),fill_value=0)
        per_class[k]={CLASS_NAMES[i]:int(counts.iloc[i]) for i in range(NUM_CLASSES)}
    n={k:len(v) for k,v in dfs.items()}
    result={"n":n,"per_class":per_class,"overlap":overlap,"union":len(union)}
    print("Split sizes:",n); print("Overlap:",overlap); print("Union:",len(union))
    print(pd.DataFrame(per_class).T)
    return result

def build_transforms(train=True,img_size=224,aug="basic"):
    if train:
        ops=[transforms.RandomResizedCrop(img_size,scale=(0.7,1.0),ratio=(0.9,1.1)),
             transforms.RandomHorizontalFlip()]
        if aug=="color":
            ops.append(transforms.ColorJitter(brightness=.25,contrast=.25,saturation=.25,hue=.05))
        elif aug=="trivial":
            ops.append(transforms.TrivialAugmentWide())
        elif aug=="randaug":
            ops.append(transforms.RandAugment(num_ops=2,magnitude=9))
        elif aug not in ("basic",None):
            raise ValueError(f"Unknown aug={aug}")
    else:
        ops=[transforms.Resize(img_size+32),transforms.CenterCrop(img_size)]
    ops += [transforms.ToTensor(),transforms.Normalize(IMAGENET_MEAN,IMAGENET_STD)]
    return transforms.Compose(ops)

class DeepWeedsDataset(Dataset):
    def __init__(self,df,images_dir,transform=None):
        self.df=df.reset_index(drop=True).copy()
        self.images_dir=Path(images_dir); self.transform=transform
    def __len__(self): return len(self.df)
    def __getitem__(self,i):
        row=self.df.iloc[i]; fn=str(row["Filename"])
        with Image.open(self.images_dir/fn) as im:
            image=im.convert("RGB")
        if self.transform: image=self.transform(image)
        return image,int(row["Label"]),fn

def make_loader(df,images_dir,transform,batch_size,train,sampler=None,num_workers=2):
    ds=DeepWeedsDataset(df,images_dir,transform)
    kwargs=dict(batch_size=batch_size,num_workers=num_workers,pin_memory=torch.cuda.is_available(),
                persistent_workers=(num_workers>0),drop_last=(train and sampler is None))
    if num_workers>0:
        kwargs["prefetch_factor"]=2
    if sampler=="balanced":
        counts=df["Label"].value_counts().reindex(range(NUM_CLASSES),fill_value=0).values
        weights=df["Label"].map({i:1.0/max(counts[i],1) for i in range(NUM_CLASSES)}).to_numpy()
        ws=WeightedRandomSampler(torch.as_tensor(weights,dtype=torch.double),len(weights),replacement=True)
        kwargs["sampler"]=ws
    else:
        kwargs["shuffle"]=train
    return DataLoader(ds,**kwargs)

def preview_batch(loader, n=9):
    import matplotlib.pyplot as plt
    mean=np.array(IMAGENET_MEAN); std=np.array(IMAGENET_STD)
    x,y,names=next(iter(loader))
    n=min(n,len(x)); cols=3; rows=math.ceil(n/cols)
    fig,axs=plt.subplots(rows,cols,figsize=(10,3*rows)); axs=np.array(axs).reshape(-1)
    for i in range(n):
        im=(x[i].permute(1,2,0).numpy()*std+mean).clip(0,1)
        axs[i].imshow(im); axs[i].set_title(CLASS_NAMES[int(y[i])]); axs[i].axis("off")
    for ax in axs[n:]: ax.axis("off")
    plt.tight_layout(); return fig
