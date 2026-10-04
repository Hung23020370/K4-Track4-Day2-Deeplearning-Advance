# %% [markdown]
# ## Các cell còn thiếu — chạy trên Kaggle (cần GPU/ảnh). Chưa kiểm thử trên máy của tôi.
# Dán sau ô "6. Sanity check loss" (cell 1, 2) và sau ô final (cell 3). KHÔNG đưa kết quả cell 1 vào bảng thí nghiệm.

# %% [markdown]
# ### Cell 1 — Overfit một batch nhỏ (rubric A: 3 điểm)
# Kỳ vọng: loss bắt đầu ≈ ln 9 = 2.197 và giảm gần 0, acc batch → 1.0 sau vài chục step.

# %%
import torch, math, numpy as np
from dataset import make_loader, build_transforms
from model import build_model
from losses import build_criterion

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(0)
loader = make_loader(train_df, IMAGES_DIR, build_transforms(False, 224, "basic"),   # không augmentation ngẫu nhiên
                     batch_size=16, train=False, sampler=None, num_workers=2)
xb, yb, _ = next(iter(loader)); xb, yb = xb.to(dev), yb.to(dev)
print("nhãn trong batch:", yb.tolist())
m = build_model("convnext_tiny", pretrained=True, num_classes=9, init="finetune").to(dev).train()
opt = torch.optim.AdamW(m.parameters(), lr=1e-4, weight_decay=0.0)
crit = build_criterion("ce")
log = []
for step in range(60):
    opt.zero_grad(set_to_none=True)
    loss = crit(m(xb), yb); loss.backward(); opt.step()
    if step % 10 == 0 or step == 59:
        acc = (m(xb).argmax(1) == yb).float().mean().item()
        log.append((step, float(loss), acc)); print(f"step {step:02d} loss={float(loss):.4f} acc={acc:.2f}")
print(f"loss đầu = {log[0][1]:.3f} (ln9 = {math.log(9):.3f}); loss cuối = {log[-1][1]:.4f}")
del m, opt; torch.cuda.empty_cache()

# %% [markdown]
# ### Cell 2 — Ghi version thư viện và tag trọng số timm (rubric A: 3 điểm; B: công bằng)

# %%
import timm, torchvision, sklearn, platform, json
print("python", platform.python_version(), "| torch", torch.__version__, "| torchvision", torchvision.__version__,
      "| timm", timm.__version__, "| sklearn", sklearn.__version__)
tags = {}
for exp, name in BACKBONES.items():
    mm = timm.create_model(name, pretrained=False)          # chỉ đọc cấu hình, không tải trọng số
    cfg = mm.pretrained_cfg
    tags[exp] = dict(backbone=name, tag=cfg.get("tag"), architecture=cfg.get("architecture"), hf_hub_id=cfg.get("hf_hub_id"))
    print(exp, tags[exp]); del mm
json.dump(tags, open(f"{WORK_DIR}/weights_tags.json", "w"), indent=1)

# %% [markdown]
# ### Cell 3 — Xem ảnh bị đoán sai (rubric G: phân tích lỗi, 3 điểm)
# Dùng file `results/eval/misclassified_F01_test.csv` (đã tạo từ predictions). Đây là xem ảnh để phân tích lỗi SAU khi
# đã chốt cấu hình, không dùng để chọn lại cấu hình.

# %%
import pandas as pd, matplotlib.pyplot as plt
from PIL import Image
w = pd.read_csv("misclassified_F01_test.csv")           # upload file này vào Kaggle working
cnt = w.groupby("Filename").size().rename("n_seed_sai")
pair = w[((w.true == "Chinee apple") & (w.pred == "Snake weed")) | ((w.true == "Snake weed") & (w.pred == "Chinee apple"))]
pair = pair.drop_duplicates("Filename").merge(cnt, on="Filename")
print("ảnh Chinee↔Snake bị sai:", len(pair), "| sai ở cả 3 seed:", int((cnt == 3).sum()), "trên", len(cnt), "ảnh sai ít nhất 1 seed")
def show(df, title, n=12):
    df = df.head(n); cols = 4; rows = (len(df) + cols - 1) // cols
    fig, axs = plt.subplots(rows, cols, figsize=(14, 3.4 * rows)); axs = axs.ravel()
    for a, (_, r) in zip(axs, df.iterrows()):
        a.imshow(Image.open(f"{IMAGES_DIR}/{r.Filename}").convert("RGB")); a.axis("off")
        a.set_title(f"thật: {r.true}\nđoán: {r.pred} ({r.n_seed_sai}/3 seed)", fontsize=8)
    for a in axs[len(df):]: a.axis("off")
    fig.suptitle(title); plt.tight_layout(); return fig
show(pair, "Chinee apple ↔ Snake weed bị đoán sai").savefig(f"{WORK_DIR}/errors_chinee_snake.png", dpi=130)
neg = w[(w.true == "Negative")].drop_duplicates("Filename").merge(cnt, on="Filename").sort_values("n_seed_sai", ascending=False)
show(neg, "Negative bị đoán thành cỏ dại (sai nhiều seed nhất)").savefig(f"{WORK_DIR}/errors_negative.png", dpi=130)
miss = w[(w.pred == "Negative")].drop_duplicates("Filename").merge(cnt, on="Filename").sort_values("n_seed_sai", ascending=False)
show(miss, "Cỏ dại bị đoán thành Negative (sai nhiều seed nhất)").savefig(f"{WORK_DIR}/errors_weed_as_negative.png", dpi=130)
plt.show()
# Khi xem, ghi vào báo cáo: ảnh nhỏ/xa/che khuất? nền giống cây? nhãn có vẻ sai? Chỉ xác nhận hoặc bác bỏ giả thuyết mục 6.3.

# %% [markdown]
# ### Cell 4 (tuỳ chọn) — Lưới ảnh mẫu mỗi lớp cho EDA (rubric A: "ảnh mẫu")
# Hiện ảnh mẫu trong báo cáo mới có 4/9 lớp (augmentation_preview). Cell này xuất 3 ảnh/lớp cho đủ 9 lớp.

# %%
import matplotlib.pyplot as plt
from PIL import Image
fig, axs = plt.subplots(9, 3, figsize=(7.5, 22))
for c in range(9):
    sample = train_df[train_df.Label == c].sample(3, random_state=0)
    for j, fn in enumerate(sample.Filename):
        axs[c, j].imshow(Image.open(f"{IMAGES_DIR}/{fn}").convert("RGB")); axs[c, j].axis("off")
        if j == 0: axs[c, j].set_title(CLASS_NAMES[c], fontsize=9, loc="left")
plt.tight_layout(); plt.savefig(f"{WORK_DIR}/eda_samples_per_class.png", dpi=110); plt.show()
