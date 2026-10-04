# Báo cáo Lab Day 2 — Backbone, công thức huấn luyện và suy luận trên DeepWeeds

> Mọi số trong báo cáo đến từ `results.xlsx`, `eval/` (kết quả `eval.py score/grade`) hoặc được tính lại từ `predictions/`.
> Mọi số trong báo cáo đã có bằng chứng trong thư mục nộp. Chỉ còn một chỗ đánh dấu **[CẦN ĐIỀN]**: link Kaggle công khai.

## 1. Tóm tắt

Bài toán: phân loại 9 lớp (8 loài cỏ dại + Negative) trên DeepWeeds, fold 0 chia sẵn (10.501 / 3.501 / 3.507 ảnh), chỉ số chính là macro-F1.
Đã so sánh 5 backbone (B01–B05), 6 biến thể công thức huấn luyện của ConvNeXt-Tiny (T01–T06) và chạy cấu hình chung kết **F01 = ConvNeXt-Tiny + finetune + augmentation cơ bản + CE + 1-view + temperature scaling** với 3 seed.

- **Kết quả test F01 (3 seed, mean ± std): top-1 = 97,51 ± 0,12 %, macro-F1 = 0,9680 ± 0,0017**, ECE = 0,0057 ± 0,0009 (trước temperature scaling: 0,0177 ± 0,0012).
- Recall hai lớp khó: Chinee apple 94,0 %, Snake weed 95,3 % (bài báo gốc: 88,5 % và 88,8 %; chỉ để tham khảo, khác điều kiện huấn luyện).
- Mốc ResNet-50 + công thức nền (T00R, 3 seed): macro-F1 test 0,8065 ± 0,0103. F01 hơn **+0,161** macro-F1.
- **Phần lớn mức cải thiện đến từ việc đổi backbone** (ResNet-50 → ConvNeXt-Tiny). Trong 12 epoch, các thay đổi về augmentation và loss không cải thiện so với công thức nền (Δ ≤ 0 và nhỏ hơn hoặc xấp xỉ nhiễu giữa các seed), và temperature scaling chỉ cải thiện hiệu chuẩn, không đổi nhãn dự đoán.
- Độ trễ ConvNeXt-Tiny FP32, batch 1, Tesla T4: p95 = 6,4 ms, dư rất nhiều so với ngân sách 100 ms.
- Suy luận (chỉ val): ensemble 3 seed là phương pháp duy nhất vượt nhiễu (+0,003–0,004 macro-F1, 3× chi phí); hflip TTA không phân biệt được khỏi nhiễu; 5-crop 192 làm giảm điểm (xem mục 5).

## 2. Dữ liệu và thiết lập

**Dữ liệu.** DeepWeeds, fold 0 nguyên bản từ GitHub của tác giả (`labels/`), không sửa. Kiểm tra (`eval/split_checks.json`, sheet `SplitChecks`):

| Tập | Số ảnh | Ghi chú |
|---|---|---|
| train | 10.501 | ≈ 60,0 % |
| val | 3.501 | ≈ 20,0 % |
| test | 3.507 | ≈ 20,0 % |

Giao train∩val = train∩test = val∩test = 0; hợp ba tập = 17.509 = số dòng `labels.csv`. Các file dự đoán test (F01, F01uncal, T00R) phủ đúng toàn bộ 3.507 ảnh test, file val phủ đúng 3.501 ảnh val.
Notebook Kaggle (`code/kaggle_run_notebook.ipynb`) in ra đúng các số trên (kích thước 10.501/3.501/3.507, giao rỗng, hợp 17.509) trước mỗi lần chạy. Kiểm tra "mọi file CSV tồn tại trong thư mục ảnh" nằm trong `check_split(..., IMAGES_DIR)` (`code/dataset.py`): hàm raise `FileNotFoundError` nếu thiếu ảnh, nên chạy qua không lỗi nghĩa là đủ ảnh (đã xác nhận từ mã nguồn). Ghi chú: `train/val/test_subset0.csv` của bản Kaggle không có cột `Species`, nên `load_split` được vá để lấy `Species` từ `labels.csv` (map theo `Filename`); các CSV fold 0 không bị sửa.

**Kiểm tra pipeline (trước khi train).** (i) Xem ảnh sau augmentation (`figures/augmentation_preview.png`, 9 ảnh của một batch train, augmentation cơ bản): ảnh sau RandomResizedCrop(scale 0,7–1)/lật vẫn giữ nội dung cây/đất, màu sau khi khử chuẩn hoá hợp lý, nhãn hiển thị khớp nội dung (batch gồm 4 Negative, 2 Prickly acacia, 2 Snake weed, 1 Rubber vine). (ii) CE với logit đều trên 9 lớp = 2,19722 so với ln 9 = 2,19722 (kiểm tra bằng logit giả, chưa phải loss ban đầu của model thật). (iii) Focal γ=0 khớp CE (hai giá trị bằng nhau, 2,6467 trên dữ liệu thử). (iv) *Overfit một batch nhỏ* (`code/todo_kaggle_cells.py`, Cell 1; ConvNeXt-Tiny pretrained, 16 ảnh đầu của train, không augmentation, AdamW lr 1e-4, 60 step, seed 0): loss 2,351 ở step 0 (ln 9 = 2,197; cao hơn một chút vì head khởi tạo ngẫu nhiên và batch chỉ 16 ảnh) → 0,0060 ở step 10 → 0,0001 ở step 59, acc batch 0,50 → 1,00 từ step 10. Pipeline học được. Hạn chế: batch chỉ có 6 trong 9 lớp (8/16 ảnh là Negative). Kết quả này không đưa vào bảng thí nghiệm.

**EDA.** `figures/eda_class_distribution.png`, `eval/class_counts.csv`. Lớp Negative chiếm 5.463/10.501 ≈ 52 % tập train; 8 lớp cỏ dại mỗi lớp khoảng 605–675 ảnh nên mất cân bằng giữa các loài cỏ dại nhẹ (tỉ lệ lớn nhất/nhỏ nhất ≈ 1,12), còn giữa Negative và loài cỏ dại thì rất rõ (≈ 9 lần so với lớp nhỏ nhất). Vì vậy top-1 bị lớp Negative kéo cao, nên dùng macro-F1 làm chỉ số chính.
Đối chiếu với Table 1 (`README.md` mục 2): số đếm từ ba tập cộng lại là Chinee apple 1.126, Lantana 1.063, các lớp còn lại khớp đúng (Parkinsonia 1.031, Parthenium 1.022, Prickly acacia 1.062, Rubber vine 1.009, Siam weed 1.074, Snake weed 1.016, Negative 9.106), tổng 17.509. Chỉ lệch 1 ảnh giữa Chinee apple (Table 1: 1.125) và Lantana (Table 1: 1.064); mình chưa xác định được nguyên nhân, nên ghi nhận như một sai khác nhỏ. Ảnh mẫu: `figures/eda_samples_per_class.png` (3 ảnh mỗi lớp, đủ 9 lớp), `figures/augmentation_preview.png` (9 ảnh, 4 lớp) và các ảnh trong mục 6.3 (`figures/errors_*.png`). Trong lưới 27 ảnh, độ biến thiên trong cùng một lớp rất lớn (ví dụ ảnh Snake weed ngả xanh lam/tím, Rubber vine ngả đỏ), cây mục tiêu thường chỉ chiếm một phần nhỏ khung hình (Parkinsonia, Prickly acacia, Siam weed), và có ít nhất một ảnh (Prickly acacia, ảnh thứ ba) chỉ thấy sỏi/đất và một dây đen, không thấy cây rõ ràng, gợi ý nhãn nhiễu trong tập train (quan sát trên 27 ảnh, chưa kiểm tra toàn bộ). Ảnh chụp sát mặt đất, 256×256, ánh sáng thay đổi mạnh (bóng đổ, ảnh ngả hồng/tím hoặc xanh lam trong một số ảnh), nền đa dạng (đất trơ, lá khô, đá, cỏ). Lớp Negative không đồng nhất, gồm cả cỏ và cây không thuộc 8 loài mục tiêu, nên có thể trông giống cỏ dại mục tiêu.

**Công thức nền (T00).** Theo ghi chú trong notebook: ImageNet-pretrained, fine-tune toàn bộ, ảnh 224 px, augmentation cơ bản, CE, AdamW, 12 epoch, batch 64, chọn checkpoint theo macro-F1 val cao nhất. Theo markdown của notebook: LR backbone/head = 1e-4/1e-3, weight decay 0,05, warmup + cosine, AMP. Phần cứng huấn luyện và đo độ trễ: Kaggle, **Tesla T4**, torch 2.11.0+cu128, CUDA 12.8. Seed: 0 (backbone, ablation), 0/1/2 (chung kết và mốc). Phiên bản (in từ notebook): Python 3.13.15, torch 2.11.0+cu128, torchvision 0.26.0+cu128, timm 1.0.30, scikit-learn 1.6.1. Tag trọng số timm (mặc định của `timm.create_model` ở timm 1.0.30, lưu ở `results/eval/weights_tags.json`): resnet50 `a1_in1k`, resnext50_32x4d `a1h_in1k`, convnext_tiny `in12k_ft_in1k`, deit_small_patch16_224 `fb_in1k`, efficientnet_b0 `ra_in1k`. **Lưu ý công bằng:** trọng số ConvNeXt-Tiny được tiền huấn luyện trên ImageNet-12k rồi tinh chỉnh trên ImageNet-1k, còn bốn backbone kia chỉ dùng ImageNet-1k; xem mục 3.

**Quy trình val/test.** Backbone, công thức và T được chọn/khớp trên val. Nhiệt độ T khớp riêng cho mỗi seed trên val (tôi kiểm tra lại: T = 2,013 / 2,152 / 2,164 cho seed 0/1/2, đúng bằng T đã áp lên file test). Trong notebook, `save_test_predictions=True` chỉ được bật ở ô chung kết (cell 12) và ô này chạy đúng một lần cho mỗi (exp, seed); các run B và T01–T06 không lưu dự đoán test. Dự đoán `F01uncal`/`F01` được sinh từ cùng `test_logits.npy` của một lần forward, nên không có forward test thứ hai.

## 3. So sánh backbone (val, seed 0)

Cùng công thức nền, cùng seed, cùng split. Hình: `figures/backbone_tradeoff.png`.

| exp_id | Backbone | Tag trọng số | Params (M) | GMAC | macro-F1 val | top-1 val | s/epoch (suy ra) |
|---|---|---|---|---|---|---|---|
| B01 | resnet50 | a1_in1k | 23,5 | 4,11 | 0,8139 | 0,8652 | 73,9 |
| B02 | resnext50_32x4d | a1h_in1k | 23,0 | 4,26 | 0,8612 | 0,8963 | 64,7 |
| **B03** | **convnext_tiny** | **in12k_ft_in1k** | 27,8 | 4,47 | **0,9672** | **0,9751** | 57,8 |
| B04 | deit_small_patch16_224 | fb_in1k | 21,7 | 4,25 | 0,9529 | 0,9654 | 40,6 |
| B05 | efficientnet_b0 | ra_in1k | 4,0 | 0,40 | 0,8584 | 0,8960 | 35,6 |

Nhận xét:
- ConvNeXt-Tiny (0,9672) và DeiT-S (0,9529) vượt xa ba mạng còn lại (0,81–0,86) ở cùng công thức nền, chênh lệch hơn 0,09 macro-F1, lớn hơn rất nhiều nhiễu seed (≈ 0,002–0,007). Đây là kết luận chắc chắn dù chỉ có 1 seed cho mỗi backbone.
- Với ResNet-50/ResNeXt/EfficientNet, công thức nền (LR, chuẩn hoá, augmentation cơ bản) có thể chưa phù hợp bằng với ConvNeXt/DeiT (công thức này có thể thiên vị kiến trúc hiện đại, nên không nên đọc thứ hạng này như thứ hạng "tuyệt đối" của kiến trúc). Đây là giả thuyết, chưa kiểm chứng.
- FLOPs không dự đoán thời gian: EfficientNet-B0 chỉ 0,40 GMAC nhưng thời gian mỗi epoch xấp xỉ DeiT-S (4,25 GMAC).
- Độ trễ batch-1 và tag trọng số của từng backbone chưa được ghi lại (ô trống trong `Backbones`).
- **Biến nhiễu về trọng số tiền huấn luyện (quan trọng).** Tag cho thấy ConvNeXt-Tiny dùng `in12k_ft_in1k` (tiền huấn luyện thêm trên ImageNet-12k), trong khi ResNet-50 (`a1_in1k`), ResNeXt-50, DeiT-S và EfficientNet-B0 chỉ ImageNet-1k. Vì vậy khoảng cách ConvNeXt so với các backbone còn lại **gồm cả ảnh hưởng của dữ liệu tiền huấn luyện lớn hơn, không chỉ kiến trúc**; so sánh này công bằng về split, công thức, seed nhưng **không công bằng về trọng số khởi tạo**. Muốn tách riêng cần chạy thêm ConvNeXt-Tiny với tag ImageNet-1k (ví dụ `convnext_tiny.fb_in1k`), chưa làm. Ngoài ra tag `a1_in1k`/`a1h_in1k` của ResNet được huấn luyện bằng công thức riêng (BCE), có thể chưa phù hợp với tinh chỉnh bằng CE (giả thuyết, chưa kiểm chứng).
- **Chọn đi tiếp: B03 ConvNeXt-Tiny** vì macro-F1 val cao nhất (cách DeiT-S 0,014 là chênh lệch lớn so với nhiễu seed ≈ 0,002); DeiT-S (cùng nhóm ImageNet-1k) là phương án thay thế gần nhất và nhanh hơn khi train (40,6 so với 57,8 s/epoch; thời gian train trên Kaggle dao động giữa các lần chạy, ví dụ ResNet-50 từ 49,5 lên 73,9 s/epoch, nên chỉ mang tính tham khảo). Quyết định chọn dựa vào F1 val, kèm lưu ý về trọng số ở trên; độ trễ batch-1 các backbone khác chưa đo nên không dùng để chọn.

Biểu đồ training của từng backbone: `curves/B01…B05_*.png`.

## 4. Công thức huấn luyện (ConvNeXt-Tiny, val, seed 0)

Mỗi dòng khác T00 đúng một yếu tố. Thiết kế tham lam theo trục (A. khởi tạo, B. augmentation, C. loss), mỗi trục có ≥ 2 giá trị. Hình: `figures/ablation_delta.png`.
Ngưỡng nhiễu: std macro-F1 val qua 3 seed của đúng công thức nền = **0,0022** (từ F01 val; temperature scaling không đổi nhãn nên giống T00).

| exp_id | Khác T00 | macro-F1 val | Δ so với T00 | F1 Chinee apple | F1 Snake weed | ECE val |
|---|---|---|---|---|---|---|
| T00 | (nền, = B03) | 0,9672 | — | 0,9455 | 0,9216 | 0,0182 |
| T01 | init từ đầu | 0,3131 | −0,654 | 0,2375 | 0,1973 | 0,0227 |
| T02 | đóng băng backbone | 0,8554 | −0,112 | 0,8222 | 0,7835 | 0,0379 |
| T03 | + color jitter | 0,9644 | −0,0028 | 0,9258 | 0,9177 | 0,0189 |
| T04 | RandAugment | 0,9659 | −0,0013 | 0,9352 | 0,9181 | 0,0195 |
| T05 | label smoothing 0,1 | 0,9664 | −0,0007 | 0,9254 | 0,9426 | 0,0790 |
| T06 | focal γ=2 | 0,9668 | −0,0004 | 0,9431 | 0,9273 | 0,0078 |

Nhận xét:
- **Khởi tạo là yếu tố quan trọng nhất**: huấn luyện từ đầu chưa hội tụ trong 12 epoch (0,31), đóng băng backbone mất 0,11. Chênh lệch này vượt xa nhiễu.
- **Augmentation và loss: không phân biệt được với công thức nền.** T04, T05, T06 nằm trong ±1 std (0,0022) so với T00; T03 thấp hơn 0,0028 (≈ 1,3 std, với ước lượng std chỉ từ 3 seed và T03 chỉ có 1 seed). Không có bằng chứng rằng chúng giúp ích; cũng không đủ bằng chứng rằng chúng gây hại. Một giải thích khả dĩ (chưa kiểm chứng) là với chỉ 12 epoch và ConvNeXt pretrained đã đạt ~0,967, còn ít chỗ để regularization giúp.
- Focal loss (T06) là cấu hình tốt nhất về **ECE val** (0,0078 so với 0,0182) mà không mất F1; label smoothing (T05) làm ECE val **tệ hơn** nhiều (0,079). Cả hai chỉ có 1 seed.
- Một số F1 theo lớp thay đổi nhiều hơn macro-F1 (ví dụ Snake weed: T05 0,943 so với 0,922 của T00), nhưng chỉ dựa trên 1 seed và vài chục ảnh/lớp nên không kết luận.
- Chưa thử kết hợp các yếu tố tốt (không có yếu tố nào tốt hơn nền để kết hợp), chưa thử trục sampler/class-weight và Mixup/CutMix.

## 5. Suy luận và hiệu chuẩn

Tất cả chỉ chạy trên **val** (không chạy test cho các phương pháp suy luận này). GPU Tesla T4, batch 1, 224 px, FP32 (trừ I01), warmup 10, `cuda.synchronize` trước/sau mỗi lần đo, 100 lần. Hình: `figures/inference_tradeoff.png`. Mã: `code/extra_inference_cells.py`.

**Kiểm tra tính đúng đắn.** Forward lại bằng autocast với view `identity` tái tạo đúng logit đã lưu (max|Δlogit| = 0, argmax khớp 100 % cả 3 seed), nên pipeline val/transform/checkpoint của các số TTA là đáng tin.

| exp_id | Phương pháp | K | macro-F1 val | ECE val | p50 / p95 / p99 (ms) | Chi phí vs I00 |
|---|---|---|---|---|---|---|
| I00 | 1-view FP32 (mốc, mean 3 seed) | 1 | 0,9652 | 0,0182 (seed 0) | 5,71 / 6,45 / 7,24 | 1,0 |
| I01 | AMP FP16 | 1 | chưa đo | — | 7,49 / 7,94 / 8,71 (`benchmark.latency_report`) | 1,29 |
| I02 | Temperature scaling | 1 | không đổi | 0,0077 (in-sample) | ≈ I00 | ≈ 1 |
| I03 | Ensemble 3 seed, gộp xác suất | 3 | 0,9687 | 0,0102 | chưa đo (ước ≈ 17) | 3 |
| I04 | Ensemble 3 seed, gộp logit | 3 | **0,9695** | 0,0169 | chưa đo (ước ≈ 17) | 3 |
| I05 | Lật ngang TTA (prob / logit) | 2 | 0,9663 / 0,9661 | 0,0157 / 0,0167 | 11,30 / 11,55 / 12,66 | 2,0 |
| I06 | 1-view + 5-crop 192 (prob / logit) | 6 | 0,9619 / 0,9615 | 0,0154 / 0,0185 | 32,76 / 34,48 / 35,50 | 5,7 |

Ghi chú đo: macro-F1 của I00, I05, I06 là trung bình 3 seed; ECE của I05/I06 chỉ tính seed 0 (so với I00 seed 0 = 0,0182). I03/I04 là một ensemble duy nhất. Độ trễ I00 đo bằng `benchmark.latency_report` là 5,82 / 6,38 / 6,55 ms, còn bằng `bench_ms` trong ô TTA là 5,71 / 6,45 / 7,24 ms; hai phép đo khác nhau nhẹ do dao động giữa các lần đo, nên chi phí tương đối được tính trong cùng một lần đo.

Nhận xét (so với nhiễu std seed của I00 ≈ 0,002):
- **Ensemble là phương pháp duy nhất vượt nhiễu**, nhưng vừa phải: +0,0034 (prob) đến +0,0043 (logit) so với mốc trung bình 3 seed (≈ 1,5–2 std); so với seed tốt nhất (0,9672) chỉ còn +0,0015 đến +0,0024. Chênh giữa gộp prob và gộp logit (0,0009) nằm trong nhiễu nên không chọn theo đó; gộp prob có ECE tốt hơn rõ (0,0102 so với 0,0169) và là cách gộp chuẩn. Độ trễ ensemble chưa đo, con số ≈ 17 ms chỉ là ước lượng 3 × I00.
- **hflip TTA không có lợi ích phân biệt được khỏi nhiễu** (+0,0011, nửa std) mà tốn gấp đôi độ trễ.
- **5-crop 192 làm giảm điểm** (−0,0033 đến −0,0037, hơn 1 std). Giả thuyết (chưa kiểm chứng): model train ở 224 full-frame, crop 192 cắt nội dung và lệch scale so với lúc train. Kết quả âm hợp lệ.
- AMP **chậm hơn** FP32 ở batch 1 trên T4 (1,29×); giả thuyết chi phí khởi chạy kernel/chuyển kiểu lấn át phần tính toán, chưa profile.
- Temperature scaling (T khớp riêng mỗi seed trên val, áp lên test): ECE test 0,0177 → 0,0057 (mean 3 seed), `figures/reliability_F01.png`; không đổi nhãn nên macro-F1 và top-1 giữ nguyên.

**Đánh đổi chính xác – độ trễ.** Hợp thời gian thực (≤ 100 ms): I00 (5,7 ms), I05 (11,3 ms), I06 (32,8 ms) đều dưới ngân sách, nhưng chỉ I00 là hợp lý vì hai phương pháp kia không tăng F1. Hợp ngoại tuyến/xử lý lô: ensemble 3 seed, chấp nhận 3× chi phí và bộ nhớ để lấy ≈ +0,003–0,004 F1 và ECE tốt hơn. Nếu dùng ensemble cho kết quả cuối, cần chạy test **đúng một lần mỗi seed** và chốt trước cách gộp.

Phương pháp đo độ trễ: `bench_ms` (warmup 10, `cuda.synchronize` trước/sau mỗi lần đo, 100 lần, input ngẫu nhiên, chưa tính tiền xử lý/đọc ảnh) cho I00/I05/I06 (xem cell trong `code/extra_inference_cells.py`); I01 đo bằng `benchmark.latency_report` ở lần trước (`code/benchmark.py` có warmup, `cuda.synchronize` trước/sau mỗi lần đo và `inference_mode`, đã xác nhận từ mã nguồn). Lưu ý các số độ trễ TTA đo bằng model của seed cuối vòng lặp, FP32, không autocast.

Số phương pháp suy luận ngoài mốc: I01–I06 (6 mục, trong đó I03–I06 có macro-F1 val). **Chưa thực hiện**: dò độ phân giải, EMA/soup, gộp BN, đo độ trễ ensemble.

## 6. Cấu hình tốt nhất và phân tích lỗi

### 6.1 Bảng chung kết (test, 3 seed)

| Cấu hình | macro-F1 test | top-1 test | ECE test |
|---|---|---|---|
| **F01** ConvNeXt-Tiny + CE + 1-view + TS | **0,9680 ± 0,0017** | **0,9751 ± 0,0012** | **0,0057 ± 0,0009** |
| F01uncal (như F01, chưa TS) | 0,9680 ± 0,0017 | 0,9751 ± 0,0012 | 0,0177 ± 0,0012 |
| T00R ResNet-50 + công thức nền + 1-view | 0,8065 ± 0,0103 | 0,8570 ± 0,0058 | 0,0321 ± 0,0048 |

(Chi tiết từng seed: sheet `Final`; đã xác minh bằng `eval.py score`.)

**Về mốc.** Theo GUIDE 1.4 mốc mặc định là ResNet-50, nên mốc chính ở đây là T00R (ResNet-50). Cần lưu ý: mốc cùng backbone với F01 (ConvNeXt + công thức nền + 1-view) chính là F01uncal, và nó **trùng macro-F1 với F01** (Δ = 0, vì temperature scaling không đổi nhãn). Tức là: so với mốc ResNet-50 cải thiện +0,161 (≫ std 0,010), nhưng toàn bộ phần này do đổi backbone; so với mốc cùng backbone, công thức huấn luyện và suy luận đã thử không đem lại cải thiện về F1.

**Nguồn gốc nhầm lẫn (đã xác minh từ notebook).** Ở ô chung kết (cell 12), `T00` được tạo bằng `Config(exp_id="T00", ...)` không truyền `backbone` (chỉ F01 nhận `FINAL_CONFIG`), nên T00 dùng backbone mặc định của `Config`, là **ResNet-50**; log in ra `params_m = 23,5` và `macro_f1_val = 0,8139`, trùng B01. Vì vậy các file `T00_*` trong bản trước thực chất là ResNet-50; đã đổi tên thành `T00R`. Lần chạy T00 của ConvNeXt (ô ablation, seed 0) cùng thư mục `runs/T00/seed0` đã bị ghi đè bởi lần chạy ResNet-50. Khi đó F01 seed 0 là bản tái lập chính xác (macro-F1 val 0,9672 giống hệt B03), sheet `Training` trước đây ghi nhầm F1 của ResNet-50 vào dòng T00 ConvNeXt.

### 6.2 Từng lớp (F01, test, mean 3 seed)

| Lớp | Precision | Recall | F1 |
|---|---|---|---|
| Chinee apple | 0,971 | 0,940 | 0,955 |
| Lantana | 0,951 | 0,975 | 0,963 |
| Parkinsonia | 0,979 | 0,976 | 0,977 |
| Parthenium | 0,985 | 0,966 | 0,975 |
| Prickly acacia | 0,923 | 0,970 | 0,946 |
| Rubber vine | 0,982 | 0,977 | 0,979 |
| Siam weed | 0,971 | 0,984 | 0,978 |
| Snake weed | 0,956 | 0,953 | 0,954 |
| Negative | 0,985 | 0,982 | 0,984 |

### 6.3 Ma trận nhầm lẫn và lỗi chính

`figures/confusion_F01_test.png` (cộng 3 seed). Các nhầm lẫn nhiều nhất (số ảnh trên tổng 3 seed):

- Negative → Prickly acacia: 33; Negative → Lantana: 24; Negative → Siam weed: 17 (tổng 97/5.466 ≈ 1,8 % ảnh Negative bị đoán thành cỏ dại; đây là nguồn làm thấp precision của Prickly acacia, 0,923).
- Chinee apple → Negative: 21 (3,1 % recall bị mất); Snake weed → Negative: 14; Rubber vine → Negative: 13.
- Cặp Chinee apple ↔ Snake weed: Chinee→Snake 15/678 = 2,2 %; Snake→Chinee 7/612 = 1,1 % (bài báo: 3,4 % và 4,1 %).
- Parthenium → Prickly acacia: 10/615; Parkinsonia → Prickly acacia: 8/621 (bài báo cũng nêu Parkinsonia ↔ Prickly acacia).

Thống kê lỗi (`results/eval/misclassified_F01_test.csv`, tính trên file dự đoán test của 3 seed): 262 lượt sai, trong đó 139 ảnh khác nhau bị sai ở ít nhất một seed (4,0 % trong 3.507 ảnh test) và **45 ảnh (1,3 %) sai ở cả 3 seed**; trong 45 ảnh đó có 20 ảnh Negative, 5 Snake weed, 4 Chinee apple, 4 Parthenium, 3 Lantana, 3 Prickly acacia, 3 Rubber vine, 2 Parkinsonia, 1 Siam weed. Phân tích dưới đây làm **sau khi** đã chốt cấu hình và mở test, chỉ để giải thích lỗi, không dùng để chọn lại cấu hình.

**Quan sát bằng mắt** trên các ảnh bị sai nhiều seed nhất (không xem toàn bộ 139 ảnh): `figures/errors_weed_as_negative.png` (12 ảnh), `figures/errors_negative.png` (12 ảnh), `figures/errors_chinee_snake.png` (16 ảnh, gồm 11 ảnh Chinee→Snake và 5 ảnh Snake→Chinee; 2 ảnh sai 3/3 seed, 5 ảnh 2/3, 9 ảnh 1/3).

- **Cỏ dại → Negative (80 lượt, 47 ảnh; 14 ảnh sai ở cả 3 seed).** Một số ảnh cây mục tiêu nhỏ, thưa hoặc lẫn trong lá khô/đá (Rubber vine, Siam weed), hoặc nằm trong vùng bóng tối (Lantana); điều này phù hợp giả thuyết "cỏ nhỏ/bị che khuất". Nhưng cũng có ảnh cây phủ kín khung hình (ba ảnh Snake weed, một ảnh Parkinsonia) vẫn bị đoán Negative ở cả 3 seed, nên giả thuyết "nhỏ/xa" **chỉ giải thích được một phần**; các ảnh này có thể là nhãn nhiễu hoặc phân biệt quá tinh giữa loài và cây khác (chưa kiểm chứng).
- **Negative → cỏ dại (97 lượt, 48 ảnh; 20 ảnh sai ở cả 3 seed).** Trong 12 ảnh hiển thị, 9 ảnh bị đoán là Prickly acacia, khớp với ma trận nhầm lẫn (33/97). Nhiều ảnh trong số đó là đất trơ có cành mảnh, gai và vài cụm lá xanh lam nhỏ, trông giống cây non hoặc cây bụi; khả năng có thực vật mục tiêu trong ảnh nhưng gán nhãn Negative, hoặc vùng thực vật rất nhỏ khiến nhãn mơ hồ. Một số ảnh Negative khác chứa cỏ hoặc lá xanh giữa lá khô bị đoán là Lantana/Rubber vine. Đây là giả thuyết từ việc xem ảnh, chưa được chuyên gia xác nhận.
- **Chinee apple ↔ Snake weed (22 lượt, 16 ảnh).** Phần lớn ảnh là thảm lá dày chiếm hết khung hình, lá rộng xen kẽ, tương phản mạnh do bóng đổ, nhiều ảnh bị ngả hồng/tím hoặc quá tối; hai loài trông tương tự ở độ phân giải 224 px. Chỉ 2/16 ảnh sai ở cả 3 seed, 9/16 chỉ sai ở một seed, nghĩa là phần lớn nhầm lẫn nằm ở vùng biên quyết định không ổn định giữa các seed, không phải lỗi hệ thống. So với bài báo gốc, tỉ lệ nhầm cặp này thấp hơn (2,2 % và 1,1 % so với 3,4 % và 4,1 %), nhưng điều kiện huấn luyện khác nhau nên chỉ để tham khảo.
- **Yếu tố ánh sáng/màu.** Nhiều ảnh sai có ánh sáng bất thường (bóng đổ lớn, ngả màu hồng/tím, xanh lam). Chưa có thí nghiệm riêng về lệch phân phối màu/độ sáng, nên đây chỉ là gợi ý cho công việc tiếp theo (ví dụ augmentation màu mạnh hơn hoặc rà soát nhãn).

**Khuyến nghị từ phân tích lỗi:** rà soát thủ công 45 ảnh sai ở cả 3 seed (đặc biệt 20 ảnh Negative) để xác định nhãn nhiễu trước khi kết luận về giới hạn của model.

## 7. Kết luận và khuyến nghị

- **Cấu hình tốt nhất:** F01 (ConvNeXt-Tiny, finetune, augmentation cơ bản, CE, 1-view, temperature scaling): macro-F1 test 0,9680 ± 0,0017. Hơn mốc ResNet-50 +0,161 (vượt xa nhiễu 0,010); không hơn mốc ConvNeXt cùng công thức (Δ = 0).
- **Yếu tố đóng góp nhiều nhất: backbone kèm trọng số tiền huấn luyện** (+0,15 val trên cùng công thức; không tách được kiến trúc khỏi tag `in12k_ft_in1k`, xem mục 3). Tiếp theo là khởi tạo bằng pretrained (so với đóng băng/từ đầu: +0,11 / +0,65). Augmentation, loss và các bước suy luận đã thử cho thay đổi macro-F1 nhỏ hơn hoặc xấp xỉ nhiễu.
- **Thời gian thực (30–100 ms/khung):** ConvNeXt-Tiny FP32 trên T4 đạt p95 = 6,4 ms, FP32 nhanh hơn AMP trên thiết lập này. Lưu ý đây là GPU T4 chứ không phải thiết bị nhúng của robot; cần đo lại trên phần cứng đích.
- **Suy luận:** giữ 1-view cho thời gian thực; ensemble 3 seed chỉ đáng cân nhắc cho xử lý ngoại tuyến (cần chạy test một lần mỗi seed và chốt cách gộp trước); TTA lật/5-crop không đáng.
- **Hiệu chuẩn:** nên áp temperature scaling (ECE test 0,0177 → 0,0057, chi phí gần 0).
- **Tự chấm phần I (`eval.py grade`, đề xuất; giảng viên xác nhận):** 20/20 theo ngưỡng tạm thời: I1 7/7 (top-1 97,51 %), I2 5/5 (Δ = +0,1614 so với mốc ResNet-50 T00R, s = 0,0103), I3 4/4 (Chinee apple 94,0 %, Snake weed 95,3 %), I4a 1/1 (ECE 0,0177 → 0,0057), I4b 1/1 (chênh macro-F1 val/test 0,0027), I5 2/2 (p95 = 6,45 ms, đo đúng cách). **Lưu ý:** I2 chỉ đạt 5/5 khi mốc là ResNet-50; nếu mốc là ConvNeXt-Tiny cùng công thức thì Δ = 0 và I2 = 0/5. Lệnh được chạy trên file `predictions/` với `--test-csv` dựng lại từ cột `y_true` (không có `test_subset0.csv`/`labels.csv` trong thư mục này), nên cần chạy lại trên Kaggle với file CSV gốc để kiểm tra độ phủ ảnh.

## 8. Hạn chế

- Chỉ một fold, chia ngẫu nhiên không theo địa điểm; điểm test có thể lạc quan khi gặp địa điểm/mùa mới.
- Ablation (T01–T06) và backbone (B01–B05) chỉ chạy 1 seed; chỉ cấu hình chung kết và mốc ResNet-50 có 3 seed. Nhiễu của ablation được ước lượng từ 3 seed của công thức nền.
- So sánh backbone không kiểm soát được trọng số tiền huấn luyện: ConvNeXt-Tiny dùng `in12k_ft_in1k`, các backbone khác chỉ ImageNet-1k.
- Huấn luyện chỉ 12 epoch (bài báo ~100 epoch), nên kết luận "augmentation/loss không giúp" chỉ áp dụng cho ngân sách ngắn này.
- Chưa làm: dò độ phân giải, EMA/soup, gộp BN, đo độ trễ ensemble, đo độ trễ cho các backbone khác, độ trễ trên phần cứng đích, kết hợp các yếu tố, trục sampler/class-weight/Mixup/CutMix, thử nghiệm lệch phân phối.
- Các số độ trễ chỉ đo cho F01 trên Tesla T4; các phương pháp suy luận I03–I06 chỉ đánh giá trên val, chưa chạy test.
- Thí nghiệm "thất bại" đáng ghi nhận: T01 (từ đầu) không hội tụ trong 12 epoch.

## 9. Phụ lục: danh sách exp_id

| exp_id | Nội dung | Ảnh biểu đồ | Dự đoán |
|---|---|---|---|
| B01–B05 | 5 backbone, công thức nền, seed 0 | `curves/B0x_*.png` | `predictions/B0x_seed0_val.csv` |
| T00 | ConvNeXt-Tiny công thức nền (= B03) | `curves/T00_convnext_tiny.png` | `B03_seed0_val.csv` |
| T00R | ResNet-50 công thức nền, seed 0–2 (mốc) | `curves/T00R_resnet50.png` | `T00R_seed{0,1,2}_{val,test}.csv` |
| T01–T06 | ablation ConvNeXt-Tiny, seed 0 | `curves/T0x_convnext_tiny.png` | `T0x_seed0_val.csv` |
| F01 | chung kết, seed 0–2 | `curves/F01_convnext_tiny.png` | `F01_seed{0,1,2}_{val,test}.csv`, `F01uncal_seed*_test.csv` |
| I00–I06 | suy luận (mốc, AMP, TS, ensemble, hflip, 5-crop) | `figures/inference_tradeoff.png` | tính từ `val_logits.npy` (không có file riêng) |

Notebook Kaggle đã chạy: `code/kaggle_run_notebook.ipynb` (có output). Link Kaggle công khai: **[CẦN ĐIỀN]** (notebook đọc code từ Kaggle dataset riêng `hongkung/fgfgfgf`, người khác cần dataset này hoặc repo).

**Lưu ý về notebook:** ô "12. Final runs" bản gốc tạo `T00` không truyền `backbone` nên ra ResNet-50. Mã trong `code/kaggle_run_notebook.ipynb` đã được sửa sau khi chạy (mốc `T00R`, `backbone="resnet50"`; `timm==1.0.30`; ô lỗi cũ "3. Kiểm tra dataset" được loại bỏ vì đã có ô "3. Load split" thay thế) và bổ sung các ô còn thiếu; có ô ghi chú. Output đã lưu ở các ô cũ là của lần chạy gốc, trong đó mốc tên `T00`.

**Chạy lại toàn bộ.** Notebook cuối được chạy tuần tự từ đầu trong một phiên Kaggle mới (Tesla T4, 30 ô, không ô nào lỗi). Kết quả huấn luyện và test trùng khớp với lần chạy trước ở mọi thí nghiệm (F01 test macro-F1 0,9680 ± 0,0017; T00R 0,8065 ± 0,0103; ablation T01–T06 không đổi), **trừ DeiT-S (B04)**: macro-F1 val 0,9505 → 0,9529 (có thể do phép tính attention không xác định hoàn toàn; chưa kiểm chứng). Do đó chỉ B04 và các số độ trễ/thời gian train cập nhật theo lần chạy mới. Vì test đã được forward ở lần chạy trước, lần chạy lại chỉ có ý nghĩa tái lập; không có quyết định cấu hình nào được đổi dựa trên test. Các file trong `results/predictions/` và `results/eval/` đã được thay bằng bản xuất từ lần chạy này (khác lần trước chỉ ở `B04_seed0_val.csv`; thêm `T00_seed0_val.csv`, file val đúng của ablation T00 ConvNeXt, macro-F1 0,9672, trước đó bị ResNet-50 ghi đè). Riêng thư mục `results/curves/`: `train.py` lưu đường cong theo đường dẫn tương đối nên ô đóng gói không lấy được; `B04_deit_small_patch16_224.png` đã được thay bằng bản của lần chạy này (F1 val 0,9529), 13 đường cong còn lại lấy từ lần chạy trước và không đổi vì kết quả trùng khớp ở các thí nghiệm đó.
