"""Single configurable DeepWeeds training pipeline."""
from __future__ import annotations
import argparse, json, os, random, sys, time
from dataclasses import dataclass, asdict, fields
from pathlib import Path
import numpy as np, pandas as pd
import torch
from torch.cuda.amp import GradScaler, autocast

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from eval import compute_metrics, save_predictions
from dataset import load_split, check_split, build_transforms, make_loader, NUM_CLASSES
from model import build_model, param_groups, count_params, count_gmacs
from losses import build_criterion, class_weights, mix_batch, mixed_loss

@dataclass
class Config:
    exp_id:str="T00"; seed:int=0; fold:int=0
    backbone:str="resnet50"; init:str="finetune"; drop_rate:float=0.0
    img_size:int=224; aug:str="basic"; sampler:str|None=None
    mix:str|None=None; mix_alpha:float=1.0
    loss:str="ce"; label_smoothing:float=0.0; focal_gamma:float=2.0
    class_weight_beta:float|None=None
    epochs:int=12; batch_size:int=64; lr_backbone:float=1e-4; lr_head:float=1e-3
    weight_decay:float=0.05; warmup_epochs:float=1.0; ema_decay:float|None=None
    amp:bool=True; num_workers:int=2
    images_dir:str="data/images"; labels_dir:str="data/labels"
    out_dir:str="runs"; pred_dir:str="predictions"; save_test_predictions:bool=False

def run_dir(cfg): return Path(cfg.out_dir)/cfg.exp_id/f"seed{cfg.seed}"
def pred_path(cfg,split): return Path(cfg.pred_dir)/f"{cfg.exp_id}_seed{cfg.seed}_{split}.csv"

def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark=False
    os.environ["PYTHONHASHSEED"]=str(seed)

def build_optimizer(model,cfg):
    return torch.optim.AdamW(param_groups(model,cfg.lr_backbone,cfg.lr_head,cfg.weight_decay))

def build_scheduler(optimizer,cfg,steps_per_epoch):
    total=max(1,cfg.epochs*steps_per_epoch); warm=max(1,int(cfg.warmup_epochs*steps_per_epoch))
    def f(step):
        if step < warm: return max(1e-8,(step+1)/warm)
        progress=(step-warm)/max(1,total-warm)
        return 0.5*(1+np.cos(np.pi*min(1.0,progress)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer,f)

class EMA:
    def __init__(self,model,decay):
        self.decay=float(decay)
        self.shadow={k:v.detach().clone() for k,v in model.named_parameters() if v.requires_grad}
        self.backup=None
    def update(self,model):
        with torch.no_grad():
            for k,v in model.named_parameters():
                if k in self.shadow: self.shadow[k].mul_(self.decay).add_(v.detach(),alpha=1-self.decay)
    def copy_to(self,model):
        self.backup={}
        with torch.no_grad():
            for k,v in model.named_parameters():
                if k in self.shadow:
                    self.backup[k]=v.detach().clone(); v.copy_(self.shadow[k])
    def restore(self,model):
        if self.backup is None:return
        with torch.no_grad():
            for k,v in model.named_parameters():
                if k in self.backup:v.copy_(self.backup[k])
        self.backup=None

def _keep_frozen_bn_eval(model):
    if not getattr(model,"_lab_frozen",False): return
    for m in model.modules():
        if isinstance(m,torch.nn.modules.batchnorm._BatchNorm):
            # If no trainable parameter belongs to this BN, freeze its running statistics.
            if not any(p.requires_grad for p in m.parameters(recurse=False)): m.eval()

def train_one_epoch(model,loader,criterion,optimizer,scheduler,scaler,cfg,device,ema=None):
    model.train(); _keep_frozen_bn_eval(model)
    total=0.; n=0
    for x,y,_ in loader:
        x=x.to(device,non_blocking=True); y=y.to(device,non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        mixed=None
        if cfg.mix:
            x,mixed=mix_batch(x,y,cfg.mix_alpha,cfg.mix)
        with autocast(enabled=(cfg.amp and device.type=="cuda")):
            z=model(x)
            loss=mixed_loss(criterion,z,mixed) if mixed is not None else criterion(z,y)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(),5.0)
        scaler.step(optimizer); scaler.update(); scheduler.step()
        if ema: ema.update(model)
        total += float(loss.detach())*len(y); n+=len(y)
    return {"train_loss":total/max(n,1),"lr":max(g["lr"] for g in optimizer.param_groups)}

def evaluate(model,loader,criterion,device):
    model.eval(); names=[]; ys=[]; zs=[]; total=0.; n=0
    with torch.inference_mode():
        for x,y,fn in loader:
            x=x.to(device,non_blocking=True); ydev=y.to(device,non_blocking=True)
            with autocast(enabled=(device.type=="cuda")): z=model(x)
            loss=criterion(z,ydev)
            total+=float(loss)*len(y); n+=len(y)
            names.extend(list(fn)); ys.append(y.numpy()); zs.append(z.float().cpu().numpy())
    return names,np.concatenate(ys),np.concatenate(zs),total/max(n,1)

def plot_curves(history,path,title):
    import matplotlib.pyplot as plt
    h=pd.DataFrame(history)
    fig,ax1=plt.subplots(figsize=(9,5))
    ax1.plot(h["epoch"],h["train_loss"],label="train loss")
    ax1.plot(h["epoch"],h["val_loss"],label="val loss")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss")
    ax2=ax1.twinx(); ax2.plot(h["epoch"],h["val_macro_f1"],label="val macro-F1",linestyle="--")
    ax2.set_ylabel("Val macro-F1"); ax2.set_ylim(0,1)
    ax1.set_title(title); ax1.legend(loc="upper left"); ax2.legend(loc="upper right")
    fig.tight_layout(); Path(path).parent.mkdir(parents=True,exist_ok=True); fig.savefig(path,dpi=160); plt.close(fig)

def _save_ckpt(path,model,epoch,cfg):
    torch.save({"model":model.state_dict(),"epoch":epoch,"config":asdict(cfg)},path)

def run(cfg):
    set_seed(cfg.seed); device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rd=run_dir(cfg); rd.mkdir(parents=True,exist_ok=True); Path(cfg.pred_dir).mkdir(parents=True,exist_ok=True)
    (rd/"config.json").write_text(json.dumps(asdict(cfg),indent=2),encoding="utf-8")
    train_df,val_df,test_df=load_split(cfg.labels_dir,cfg.fold)
    split_info=check_split(train_df,val_df,test_df,cfg.images_dir)
    (rd/"split_check.json").write_text(json.dumps(split_info,indent=2),encoding="utf-8")
    tr=make_loader(train_df,cfg.images_dir,build_transforms(True,cfg.img_size,cfg.aug),cfg.batch_size,True,cfg.sampler,cfg.num_workers)
    va=make_loader(val_df,cfg.images_dir,build_transforms(False,cfg.img_size,cfg.aug),cfg.batch_size,False,None,cfg.num_workers)
    te=make_loader(test_df,cfg.images_dir,build_transforms(False,cfg.img_size,cfg.aug),cfg.batch_size,False,None,cfg.num_workers) if cfg.save_test_predictions else None
    model=build_model(cfg.backbone,True,NUM_CLASSES,cfg.drop_rate,cfg.init).to(device)
    criterion_kw={}
    if cfg.loss=="ls": criterion_kw["smoothing"]=cfg.label_smoothing
    if cfg.loss=="focal": criterion_kw["gamma"]=cfg.focal_gamma
    if cfg.loss=="ce_weighted":
        counts=train_df["Label"].value_counts().reindex(range(NUM_CLASSES),fill_value=0).values
        criterion_kw["weight"]=class_weights(counts,cfg.class_weight_beta or 0).to(device)
    criterion=build_criterion(cfg.loss,**criterion_kw).to(device)
    opt=build_optimizer(model,cfg); sch=build_scheduler(opt,cfg,len(tr))
    scaler=GradScaler(enabled=(cfg.amp and device.type=="cuda"))
    ema=EMA(model,cfg.ema_decay) if cfg.ema_decay else None
    best=-1.; best_epoch=0; history=[]; t0=time.time()
    for epoch in range(cfg.epochs):
        e0=time.time()
        trm=train_one_epoch(model,tr,criterion,opt,sch,scaler,cfg,device,ema)
        if ema: ema.copy_to(model)
        names,y,z,vl=evaluate(model,va,criterion,device)
        m=compute_metrics(y,z.argmax(1),torch.softmax(torch.as_tensor(z),1).numpy())
        if ema: ema.restore(model)
        rec={"epoch":epoch+1,"train_loss":trm["train_loss"],"val_loss":vl,
             "val_macro_f1":float(m["macro_f1"]),"val_top1":float(m["top1"]),
             "lr":trm["lr"],"epoch_sec":time.time()-e0}
        history.append(rec)
        print(f"epoch {epoch+1:02d}/{cfg.epochs} loss={rec['train_loss']:.4f} valF1={rec['val_macro_f1']:.4f} valAcc={rec['val_top1']:.4f}")
        if rec["val_macro_f1"]>best:
            best=rec["val_macro_f1"]; best_epoch=epoch+1
            if ema: ema.copy_to(model)
            _save_ckpt(rd/"best.pt",model,epoch+1,cfg)
            if ema: ema.restore(model)
    pd.DataFrame(history).to_csv(rd/"history.csv",index=False)
    plot_curves(history,Path("curves")/f"{cfg.exp_id}_{cfg.backbone}.png",f"{cfg.exp_id} — {cfg.backbone}")
    ck=torch.load(rd/"best.pt",map_location=device); model.load_state_dict(ck["model"])
    names,y,z,vl=evaluate(model,va,criterion,device)
    probs=torch.softmax(torch.from_numpy(z),1).numpy()
    np.save(rd/"val_logits.npy",z); np.save(rd/"val_y.npy",y)
    (rd/"val_filenames.json").write_text(json.dumps(names),encoding="utf-8")
    save_predictions(pred_path(cfg,"val"),names,y,probs)
    result={"exp_id":cfg.exp_id,"seed":cfg.seed,"best_epoch":best_epoch,"macro_f1_val":float(compute_metrics(y,probs.argmax(1),probs)["macro_f1"]),
            "top1_val":float((probs.argmax(1)==y).mean()),"params_m":count_params(model),"gmac":count_gmacs(model,cfg.img_size),
            "train_sec":time.time()-t0,"device":str(device)}
    if cfg.save_test_predictions:
        names,y,z,_=evaluate(model,te,criterion,device); probs=torch.softmax(torch.from_numpy(z),1).numpy()
        np.save(rd/"test_logits.npy",z); np.save(rd/"test_y.npy",y)
        (rd/"test_filenames.json").write_text(json.dumps(names),encoding="utf-8")
        save_predictions(pred_path(cfg,"test"),names,y,probs)
    return result

def parse_overrides(pairs):
    hints={f.name:f.type for f in fields(Config)}
    # resolve postponed annotations
    from typing import get_type_hints
    hints=get_type_hints(Config)
    out={}
    for item in pairs:
        if "=" not in item: raise ValueError(f"Override phải có KEY=VALUE: {item}")
        k,v=item.split("=",1)
        if k not in hints: raise KeyError(f"Config không có field {k}")
        typ=hints[k]
        if v.lower()=="none": out[k]=None; continue
        origin=getattr(typ,"__origin__",None)
        if typ is bool: out[k]=v.lower() in ("1","true","yes","y","on")
        elif typ is int: out[k]=int(v)
        elif typ is float: out[k]=float(v)
        elif typ is str: out[k]=v
        elif origin is not None and str(origin).endswith("Union"):
            args=getattr(typ,"__args__",())
            base=next((a for a in args if a is not type(None)),str)
            out[k]=float(v) if base is float else int(v) if base is int else v
        else: out[k]=v
    return out

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--set",nargs="*",default=[])
    args=ap.parse_args(); cfg=Config(**parse_overrides(args.set))
    print(json.dumps(run(cfg),indent=2))

if __name__=="__main__": main()
