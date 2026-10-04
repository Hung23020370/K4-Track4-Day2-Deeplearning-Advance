"""Losses, class weighting, Mixup and CutMix."""
from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def build_criterion(kind="ce",**kw):
    kind=kind.lower()
    if kind=="ce": return nn.CrossEntropyLoss()
    if kind=="ls": return LabelSmoothingCE(kw.get("smoothing",0.1))
    if kind=="focal": return FocalLoss(kw.get("gamma",2.0),kw.get("alpha"))
    if kind=="ce_weighted": return nn.CrossEntropyLoss(weight=kw["weight"])
    raise ValueError(f"Unknown loss {kind}")

class LabelSmoothingCE(nn.Module):
    def __init__(self,smoothing=0.1):
        super().__init__(); self.smoothing=float(smoothing)
    def forward(self,logits,target):
        return F.cross_entropy(logits,target,label_smoothing=self.smoothing)

class FocalLoss(nn.Module):
    def __init__(self,gamma=2.0,alpha=None):
        super().__init__(); self.gamma=float(gamma)
        if alpha is None: self.alpha=None
        else: self.register_buffer("alpha",torch.as_tensor(alpha,dtype=torch.float32))
    def forward(self,logits,target):
        logp=F.log_softmax(logits,dim=1); logpt=logp.gather(1,target[:,None]).squeeze(1)
        pt=logpt.exp(); loss=-(1-pt).pow(self.gamma)*logpt
        if self.alpha is not None: loss=loss*self.alpha.to(logits.device)[target]
        return loss.mean()

def class_weights(counts,beta=0.0):
    n=torch.as_tensor(counts,dtype=torch.float32)
    if (n<=0).any(): raise ValueError("counts phải >0")
    if beta is None: beta=0.0
    if beta==0: w=1.0/n
    else:
        b=float(beta); w=(1-b)/(1-torch.pow(torch.tensor(b),n))
    w=w/(w.mean())
    return w

def mix_batch(x,y,alpha=1.0,mode="cutmix"):
    if alpha<=0: raise ValueError("alpha phải >0")
    lam=float(np.random.beta(alpha,alpha)); perm=torch.randperm(x.size(0),device=x.device)
    y_a,y_b=y,y[perm]
    if mode=="mixup":
        return lam*x+(1-lam)*x[perm],(y_a,y_b,lam)
    if mode!="cutmix": raise ValueError(mode)
    _,_,H,W=x.shape
    cut_rat=np.sqrt(1-lam); ch=max(1,int(H*cut_rat)); cw=max(1,int(W*cut_rat))
    cy=int(np.random.randint(H)); cx=int(np.random.randint(W))
    y1=max(cy-ch//2,0); y2=min(cy+ch//2,H); x1=max(cx-cw//2,0); x2=min(cx+cw//2,W)
    xmix=x.clone(); xmix[:,:,y1:y2,x1:x2]=x[perm,:,y1:y2,x1:x2]
    lam=1-((y2-y1)*(x2-x1)/(H*W))
    return xmix,(y_a,y_b,float(lam))

def mixed_loss(criterion,logits,targets):
    y_a,y_b,lam=targets
    return float(lam)*criterion(logits,y_a)+(1-float(lam))*criterion(logits,y_b)

def self_check_focal():
    torch.manual_seed(0); z=torch.randn(32,9); y=torch.randint(0,9,(32,))
    a=FocalLoss(gamma=0)(z,y); b=F.cross_entropy(z,y)
    assert abs(float(a-b))<1e-6
    return float(a),float(b)
