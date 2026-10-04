"""Correct GPU latency benchmark."""
from __future__ import annotations
import time, numpy as np, torch

def bench(fn,warmup=10,iters=100,sync=None):
    sync=sync or (torch.cuda.synchronize if torch.cuda.is_available() else None)
    for _ in range(warmup): fn()
    if sync: sync()
    vals=[]
    for _ in range(iters):
        if sync: sync()
        t0=time.perf_counter(); fn()
        if sync: sync()
        vals.append((time.perf_counter()-t0)*1000)
    a=np.asarray(vals)
    return {"p50":float(np.percentile(a,50)),"p95":float(np.percentile(a,95)),
            "p99":float(np.percentile(a,99)),"mean":float(a.mean()),"n":int(iters)}

def latency_report(model,batch_size,img_size,dtype="fp32",device="cuda",warmup=10,iters=100):
    dev=torch.device(device); model=model.to(dev).eval()
    if dtype=="fp16": model=model.half()
    x=torch.randn(batch_size,3,img_size,img_size,device=dev)
    if dtype=="fp16": x=x.half()
    enabled=(dtype=="amp" and dev.type=="cuda")
    def fn():
        with torch.inference_mode(),torch.autocast(device_type=dev.type,dtype=torch.float16,enabled=enabled):
            _=model(x)
    r=bench(fn,warmup,iters,torch.cuda.synchronize if dev.type=="cuda" else None)
    r.update({"gpu":torch.cuda.get_device_name(0) if dev.type=="cuda" else "CPU",
              "dtype":dtype,"batch":batch_size,"img_size":img_size,
              "images_per_s":batch_size/(r["p50"]/1000),"torch":torch.__version__})
    return r

def tta_latency(model,k_views,**kw):
    base=latency_report(model,**kw)
    base["tta_views"]=k_views
    base["p50_tta_approx"]=base["p50"]*k_views
    return base
