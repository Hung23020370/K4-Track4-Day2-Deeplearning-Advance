# %% [markdown]
# ## Bước 3 (bổ sung) — thêm phương pháp suy luận, CHỈ trên VAL
# Dán vào notebook Kaggle SAU ô "11. Temperature scaling" và TRƯỚC ô final. Chưa được kiểm thử trên máy của tôi
# (tôi không có GPU/ảnh/checkpoint): nếu lỗi tên hàm/tham số, sửa theo `code/` thật của bạn.
# Yêu cầu: runs/F01/seed{0,1,2}/{best.pt,val_logits.npy,val_y.npy} đã có (ô final đã tạo).
# KHÔNG chạy test ở đây. Chỉ khi một phương pháp thắng I00 trên val vượt nhiễu (std ~0.002) mới cân nhắc
# chạy test MỘT lần cho mỗi seed và ghi rõ trong báo cáo.

# %%
import time, json
import numpy as np, pandas as pd, torch, torch.nn.functional as F
from sklearn.metrics import f1_score

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEEDS = [0, 1, 2]

def load_val_logits(exp="F01"):
    L = [np.load(f"{RUNS_DIR}/{exp}/seed{s}/val_logits.npy") for s in SEEDS]
    y = np.load(f"{RUNS_DIR}/{exp}/seed0/val_y.npy")
    return L, y

def ece15(p, y):
    conf, pred = p.max(1), p.argmax(1); ok = (pred == y); e = 0.0
    edges = np.linspace(0, 1, 16)
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any(): e += m.mean() * abs(ok[m].mean() - conf[m].mean())
    return e

def row(name, p, y, K, cost):
    pred = p.argmax(1)
    return dict(method=name, K=K, macro_f1_val=f1_score(y, pred, average="macro"),
                top1_val=(pred == y).mean(), ece_val=ece15(p, y), relative_cost=cost)

sm = lambda z: torch.softmax(torch.from_numpy(z).float(), 1).numpy()
val_logits, val_y = load_val_logits()
rows = []

# I00: 1-view, từng seed (mốc)
for s in SEEDS:
    rows.append(row(f"I00 1-view seed{s}", sm(val_logits[s]), val_y, 1, 1.0))

# I03/I04: ensemble 3 seed, gộp xác suất vs gộp logit (không cần forward lại)
p_prob = np.mean([sm(z) for z in val_logits], 0)
p_logit = sm(np.mean(val_logits, 0))
rows.append(row("I03 ensemble 3 seed (mean prob)", p_prob, val_y, 3, 3.0))
rows.append(row("I04 ensemble 3 seed (mean logit)", p_logit, val_y, 3, 3.0))

# %% [markdown]
# ### I05/I06: TTA (cần forward lại model trên VAL)

# %%
val_loader = make_loader(val_df, IMAGES_DIR, build_transforms(False, 224, "basic"),
                         batch_size=64, train=False, sampler=None, num_workers=2)

@torch.inference_mode()
def logits_with_view(model, loader, view):
    # train.py đánh giá bằng autocast (FP16) nên logit đã lưu là FP16: forward lại cũng dùng autocast cho khớp
    out = []
    for x, y, fn in loader:                       # dataset trả (tensor, label, filename)
        x = x.to(DEVICE, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.float16, enabled=(DEVICE.type == "cuda")):
            out.append(model(view(x)).float().cpu())
    return torch.cat(out).numpy()

flip = lambda x: torch.flip(x, dims=[3])
def crop5(x, c=192):                              # 4 góc + giữa, rồi đưa model (ConvNeXt nhận kích thước khác 224)
    H, W = x.shape[-2:]
    offs = [(0, 0), (0, W - c), (H - c, 0), (H - c, W - c), ((H - c) // 2, (W - c) // 2)]
    return [x[..., i:i + c, j:j + c] for i, j in offs]

tta = {}
for s in SEEDS:
    model = build_model("convnext_tiny", pretrained=False, num_classes=9, init="finetune").to(DEVICE)
    ck = torch.load(f"{RUNS_DIR}/F01/seed{s}/best.pt", map_location=DEVICE)
    model.load_state_dict(ck["model"]); model.eval()
    z_id = logits_with_view(model, val_loader, lambda x: x)
    d = np.abs(z_id - val_logits[s]); agree = (z_id.argmax(1) == val_logits[s].argmax(1)).mean()
    print(f"seed{s}: max|Δlogit|={d.max():.4f} mean={d.mean():.5f} argmax khớp={agree:.4f}")
    if agree < 0.99:   # lệch lớn => sai transform/thứ tự/checkpoint; chạy cell chẩn đoán trước khi tin kết quả TTA
        raise RuntimeError("Forward lại không tái tạo logit đã lưu: kiểm tra val transform, thứ tự file, epoch của best.pt")
    z_fl = logits_with_view(model, val_loader, flip)
    z_c5 = [logits_with_view(model, val_loader, (lambda x, k=k: crop5(x)[k])) for k in range(5)]
    tta[s] = dict(id=z_id, flip=z_fl, c5=z_c5)

def agg(zs, space):
    return sm(np.mean(zs, 0)) if space == "logit" else np.mean([sm(z) for z in zs], 0)

# macro-F1 = trung bình 3 seed; ECE chỉ tính cho seed 0
for space in ["prob", "logit"]:
    f1s = [f1_score(val_y, agg([tta[s]["id"], tta[s]["flip"]], space).argmax(1), average="macro") for s in SEEDS]
    rows.append(dict(method=f"I05 hflip TTA ({space})", K=2, macro_f1_val=float(np.mean(f1s)),
                     top1_val=np.nan, ece_val=ece15(agg([tta[0]["id"], tta[0]["flip"]], space), val_y), relative_cost=2.0))
    f1s = [f1_score(val_y, agg([tta[s]["id"]] + tta[s]["c5"], space).argmax(1), average="macro") for s in SEEDS]
    rows.append(dict(method=f"I06 1-view + 5-crop192 ({space})", K=6, macro_f1_val=float(np.mean(f1s)),
                     top1_val=np.nan, ece_val=ece15(agg([tta[0]["id"]] + tta[0]["c5"], space), val_y), relative_cost=6.0))

# %% [markdown]
# ### Độ trễ batch 1 của TTA (đo thật, có warmup + synchronize)

# %%
def bench_ms(fn, warmup=10, iters=100):
    for _ in range(warmup): fn()
    torch.cuda.synchronize(); t = []
    for _ in range(iters):
        torch.cuda.synchronize(); t0 = time.perf_counter(); fn(); torch.cuda.synchronize()
        t.append((time.perf_counter() - t0) * 1000)
    return dict(p50=np.percentile(t, 50), p95=np.percentile(t, 95), p99=np.percentile(t, 99), n=iters)

x1 = torch.randn(1, 3, 224, 224, device=DEVICE)
@torch.inference_mode()
def f_1view(): model(x1)
@torch.inference_mode()
def f_hflip(): model(x1); model(torch.flip(x1, dims=[3]))
@torch.inference_mode()
def f_5crop():
    model(x1)
    for c in crop5(x1): model(c)
lat = {k: bench_ms(f) for k, f in [("I00", f_1view), ("I05", f_hflip), ("I06", f_5crop)]}
print(json.dumps(lat, indent=1, default=float))

# %%
res = pd.DataFrame(rows).sort_values("macro_f1_val", ascending=False)
display(res)
res.to_csv(f"{WORK_DIR}/inference_val_summary.csv", index=False)
json.dump(lat, open(f"{WORK_DIR}/inference_latency_tta.json", "w"), default=float)
# So sánh Δ với std macro-F1 val qua seed của I00 (~0.002); chỉ phương pháp vượt nhiễu mới đáng chọn.
