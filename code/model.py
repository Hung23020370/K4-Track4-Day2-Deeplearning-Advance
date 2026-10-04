"""Model utilities built around timm."""
from __future__ import annotations
import copy
import torch
import torch.nn as nn

SUGGESTED_BACKBONES={
"resnet50":"resnet50","resnext50":"resnext50_32x4d","convnext_tiny":"convnext_tiny",
"deit_small":"deit_small_patch16_224","swin_tiny":"swin_tiny_patch4_window7_224",
"efficientnet_b0":"efficientnet_b0","mobilenetv3":"mobilenetv3_large_100"}

def _classifier_param_ids(model):
    try:
        head=model.get_classifier()
    except Exception:
        head=None
    if head is None or not isinstance(head,nn.Module):
        names=("fc","classifier","head","head_dist")
        mods=[getattr(model,n,None) for n in names]
        head=next((m for m in mods if isinstance(m,nn.Module)),None)
    if head is None: raise ValueError("Không tìm thấy classifier/head")
    return {id(p) for p in head.parameters()}

def build_model(name,pretrained=True,num_classes=9,drop_rate=0.0,init="finetune"):
    import timm
    if init not in ("scratch","frozen","finetune"): raise ValueError(init)
    use_pretrained=(pretrained and init!="scratch")
    model=timm.create_model(name,pretrained=use_pretrained,num_classes=num_classes,drop_rate=drop_rate)
    model._lab_head_param_ids=_classifier_param_ids(model)
    model._lab_frozen=(init=="frozen")
    if init=="frozen": freeze_backbone(model)
    cfg=getattr(model,"pretrained_cfg",{}) or {}
    model._lab_pretrained_tag=cfg.get("architecture",name)
    return model

def freeze_backbone(model):
    head_ids=_classifier_param_ids(model)
    for p in model.parameters(): p.requires_grad=(id(p) in head_ids)
    model._lab_frozen=True
    model._lab_head_param_ids=head_ids

def _is_head_param(name,p,head_ids):
    return id(p) in head_ids

def param_groups(model,lr_backbone,lr_head,weight_decay):
    head_ids=getattr(model,"_lab_head_param_ids",_classifier_param_ids(model))
    groups=[{"params":[],"lr":lr_backbone,"weight_decay":weight_decay},
            {"params":[],"lr":lr_backbone,"weight_decay":0.0},
            {"params":[],"lr":lr_head,"weight_decay":weight_decay}]
    for name,p in model.named_parameters():
        if not p.requires_grad: continue
        if id(p) in head_ids: groups[2]["params"].append(p)
        elif p.ndim<=1: groups[1]["params"].append(p)
        else: groups[0]["params"].append(p)
    return [g for g in groups if g["params"]]

def count_params(model):
    return sum(p.numel() for p in model.parameters())/1e6

def count_gmacs(model,img_size=224):
    try:
        from fvcore.nn import FlopCountAnalysis
        x=torch.zeros(1,3,img_size,img_size,device=next(model.parameters()).device)
        with torch.no_grad():
            flops=FlopCountAnalysis(model,x).total()
        return float(flops/1e9)
    except Exception:
        try:
            from thop import profile
            device=next(model.parameters()).device
            x=torch.zeros(1,3,img_size,img_size,device=device)
            macs,_=profile(model,inputs=(x,),verbose=False)
            return float(macs/1e9)
        except Exception as e:
            return float("nan")
