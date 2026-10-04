"""Inference/TTA/calibration helpers."""
from __future__ import annotations
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def predict_logits(model,loader,device,view=None):
    model.eval(); names=[]; ys=[]; outs=[]
    with torch.inference_mode():
        for x,y,fn in loader:
            x=x.to(device,non_blocking=True)
            if view is not None: x=view(x)
            with torch.autocast(device_type=device.type,dtype=torch.float16,enabled=(device.type=="cuda")):
                z=model(x)
            names.extend(list(fn)); ys.append(y.numpy()); outs.append(z.float().cpu().numpy())
    return names,np.concatenate(ys),np.concatenate(outs)

def view_identity(x): return x
def view_hflip(x): return torch.flip(x,dims=[3])

def views_multicrop(x,crop):
    H,W=x.shape[-2:]
    if crop>H or crop>W: raise ValueError("crop lớn hơn ảnh")
    coords=[(0,0),(0,W-crop),(H-crop,0),(H-crop,W-crop),((H-crop)//2,(W-crop)//2)]
    return [x[:,:,a:a+crop,b:b+crop] for a,b in coords]

def views_multiscale(x,sizes):
    return [F.interpolate(x,size=(s,s),mode="bilinear",align_corners=False) for s in sizes]

def _softmax_np(z):
    z=z-z.max(axis=1,keepdims=True); e=np.exp(z); return e/e.sum(axis=1,keepdims=True)

def aggregate_views(logits_per_view,space="prob"):
    if space=="prob":
        p=np.mean([_softmax_np(z) for z in logits_per_view],axis=0)
        return p/p.sum(1,keepdims=True)
    if space=="logit": return _softmax_np(np.mean(logits_per_view,axis=0))
    raise ValueError(space)

def ensemble_probs(list_of_probs):
    p=np.mean(np.stack(list_of_probs,axis=0),axis=0)
    return p/p.sum(1,keepdims=True)

def fit_temperature(val_logits,val_labels):
    z=torch.as_tensor(val_logits,dtype=torch.float32)
    y=torch.as_tensor(val_labels,dtype=torch.long)
    logT=torch.zeros(1,requires_grad=True)
    opt=torch.optim.LBFGS([logT],lr=0.1,max_iter=100,line_search_fn="strong_wolfe")
    def closure():
        opt.zero_grad(); T=logT.exp().clamp(0.05,20)
        loss=F.cross_entropy(z/T,y); loss.backward(); return loss
    opt.step(closure)
    return float(logT.exp().clamp(0.05,20).item())

def apply_temperature(logits,T):
    return _softmax_np(np.asarray(logits)/float(T))

def fuse_conv_bn(model):
    """Fuse direct Conv2d -> BatchNorm2d sibling pairs where possible."""
    from torch.nn.utils.fusion import fuse_conv_bn_eval
    model=copy.deepcopy(model).eval()
    def rec(parent):
        children=list(parent.named_children())
        for name,mod in children: rec(mod)
        for i in range(len(children)-1):
            n1,m1=children[i]; n2,m2=children[i+1]
            if isinstance(m1,nn.Conv2d) and isinstance(m2,nn.BatchNorm2d):
                try:
                    fused=fuse_conv_bn_eval(m1,m2)
                    setattr(parent,n1,fused); setattr(parent,n2,nn.Identity())
                except Exception:
                    pass
    rec(model)
    return model
