# Bài làm Lab Day 2 — Backbone, công thức huấn luyện và suy luận trên DeepWeeds

## Chạy lại (Kaggle, GPU Tesla T4)
1. Notebook đã chạy (có output): `code/kaggle_run_notebook.ipynb`. **Link Kaggle: https://www.kaggle.com/code/hongkung/notebook8ff4ad9abf** (đặt notebook và dataset code ở chế độ Public hoặc chia sẻ cho giảng viên).
2. Add Input: (a) DeepWeeds (`images/`, `labels/` gồm `labels.csv`, `*_subset0.csv`), (b) dataset chứa repo này (thư mục có `code/` và `eval.py`). Sửa `LAB_DIR`/`REPO_DIR`/`IMAGES_DIR`/`LABELS_DIR` ở hai cell đầu cho khớp.
3. Thứ tự chạy: cài môi trường → đường dẫn → `load_split` + `check_split` → EDA → kiểm tra augmentation/loss ban đầu/overfit 1 batch (`code/todo_kaggle_cells.py`) → backbone B01–B05 → ablation T00–T06 → final (`T00R` ResNet-50 mốc và `F01` ConvNeXt-Tiny, seed 0/1/2) → temperature scaling → `eval.py score` → suy luận bổ sung (`code/extra_inference_cells.py`, chỉ val).
4. Lưu ý: mã trong `code/kaggle_run_notebook.ipynb` đã sửa sau khi chạy (mốc `T00R` với `backbone="resnet50"`, `timm==1.0.30`, bỏ ô lỗi cũ "3. Kiểm tra dataset" đã được ô "3. Load split" thay thế) và bổ sung các ô: overfit 1 batch, ảnh mẫu mỗi lớp, tag timm, chấm lại F01 sau TS, suy luận bổ sung (chỉ val), phân tích lỗi, `eval.py grade`, đóng gói đầy đủ. Output đã lưu ở một số ô cũ là của lần chạy gốc (mốc tên `T00`); có ô ghi chú giải thích trong notebook.
5. Chấm lại: `python eval.py score --pred "results/predictions/F01_seed*_test.csv" --test-csv <test_subset0.csv> --labels <labels.csv> --tag F01 --out results/eval/eval_F01` rồi `python eval.py grade ...`.

## Phiên bản
Python 3.13.15, torch 2.11.0+cu128 (CUDA 12.8), torchvision 0.26.0+cu128, timm 1.0.30, scikit-learn 1.6.1, GPU Tesla T4. Tag trọng số (mặc định của timm 1.0.30, xem `results/eval/weights_tags.json`): resnet50.a1_in1k, resnext50_32x4d.a1h_in1k, convnext_tiny.in12k_ft_in1k, deit_small_patch16_224.fb_in1k, efficientnet_b0.ra_in1k. Để tái lập, cài `pip install timm==1.0.30` (notebook hiện dùng `-U timm`, bản mới hơn có thể đổi tag mặc định).

## Cấu trúc
`code/` mã nguồn (gồm `todo_kaggle_cells.py`: overfit 1 batch, tag timm, xem ảnh sai, lưới ảnh mẫu) · `results/results.xlsx` · `results/curves/` · `results/predictions/` · `results/eval/` · `figures/` (EDA, augmentation, confusion, reliability, backbone, ablation, suy luận, ảnh lỗi) · `report .md`.
Không commit ảnh dataset hay checkpoint (`.gitignore` đã chặn `*.pt`, `images/`, `runs/`).

## Kết quả chính (test, 3 seed)
F01 (ConvNeXt-Tiny): macro-F1 0,9680 ± 0,0017, top-1 0,9751 ± 0,0012. Mốc ResNet-50 (T00R): macro-F1 0,8065 ± 0,0103. Chi tiết trong `report .md`.
