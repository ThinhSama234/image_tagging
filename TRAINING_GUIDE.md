# Hướng dẫn chi tiết: Metrics, Loss & Training Step-by-Step

> Tài liệu bổ sung cho `DOCUMENTATION.md`  
> Dự án: **Recognize Anything (RAM / RAM++ / Tag2Text)**

---

## Mục lục

1. [Giải thích chi tiết các Loss Functions](#1-giải-thích-chi-tiết-các-loss-functions)
2. [Giải thích chi tiết các Metrics đánh giá](#2-giải-thích-chi-tiết-các-metrics-đánh-giá)
3. [Hướng dẫn tải & chuẩn bị dữ liệu](#3-hướng-dẫn-tải--chuẩn-bị-dữ-liệu)
4. [Hướng dẫn Train Step-by-Step](#4-hướng-dẫn-train-step-by-step)

---

## 1. Giải thích chi tiết các Loss Functions

### 1.1 Asymmetric Loss (ASL) — Dùng cho Tag Recognition

**File:** `ram/models/utils.py` → class `AsymmetricLoss`

#### Vấn đề cần giải quyết

Trong image tagging, mỗi ảnh có thể có **nhiều tag đúng** (multi-label). Nhưng trong 4585 tags, chỉ có ~5-20 tag đúng, còn lại ~4565 tag sai → **class imbalance cực lớn** (negative >> positive). Nếu dùng Binary Cross-Entropy thông thường, model sẽ thiên về dự đoán "không có tag" cho mọi thứ.

#### Công thức từng bước

**Bước 1: Tính xác suất qua sigmoid**

```
p = σ(x) = 1 / (1 + e^(-x))

Trong đó x là logit đầu ra của FC layer.
```

- `p` gần 1 → model nghĩ tag này **có** trong ảnh
- `p` gần 0 → model nghĩ tag này **không có** trong ảnh

**Bước 2: Asymmetric Clipping (chỉ áp dụng cho negative)**

```
p_neg = max(1 - p - m, 0)

Trong đó m = 0.05 (margin)
```

Ý nghĩa: Nếu model đã khá tự tin rằng tag không có (p nhỏ → 1-p lớn), thì clip bớt để không "phạt" quá mức. Điều này giúp model tập trung vào các **hard negatives** (negative mà model chưa chắc chắn).

**Bước 3: Tính Cross-Entropy cơ bản**

```
Với positive (y=1):  L_pos = -log(p)
Với negative (y=0):  L_neg = -log(p_neg)     ← dùng p_neg đã clip
```

**Bước 4: Asymmetric Focusing (giống Focal Loss nhưng bất đối xứng)**

```
Với positive:  weight_pos = (1 - p)^γ⁺     với γ⁺ = 0  → weight = 1 (không down-weight)
Với negative:  weight_neg = (p_neg)^γ⁻      với γ⁻ = 7  → down-weight mạnh easy negatives
```

- `γ⁺ = 0`: **Không** giảm trọng số positive nào cả → mọi positive đều quan trọng
- `γ⁻ = 7`: Nếu model đã tự tin negative đúng (p_neg lớn, tức p nhỏ), thì `p_neg^7` rất lớn nhưng `log(p_neg)` gần 0 → loss gần 0. Chỉ khi model **nhầm** (p lớn cho negative) thì loss mới cao.

**Bước 5: Tổng hợp**

```
L = -Σ_k [ y_k × (1-p_k)^γ⁺ × log(p_k) + (1-y_k) × (p_neg_k)^γ⁻ × log(p_neg_k) ]

Tổng trên tất cả K = 4585 tags.
```

#### Ví dụ số cụ thể

Giả sử 1 ảnh có 3 tags, model dự đoán:

| Tag | Ground Truth (y) | Logit (x) | p = σ(x) | Loại |
|-----|-------------------|-----------|-----------|------|
| dog | 1 | 2.0 | 0.88 | Positive |
| cat | 0 | -3.0 | 0.05 | Easy negative |
| car | 0 | 1.0 | 0.73 | Hard negative |

**Tag "dog" (positive, y=1):**
```
L_dog = -(1-0.88)^0 × log(0.88) = -1 × (-0.128) = 0.128
```

**Tag "cat" (easy negative, y=0):**
```
p_neg = max(1 - 0.05 - 0.05, 0) = 0.90
weight = 0.90^7 = 0.478
L_cat = -0.478 × log(0.90) = -0.478 × (-0.105) = 0.050  ← loss rất nhỏ (tốt!)
```

**Tag "car" (hard negative, y=0):**
```
p_neg = max(1 - 0.73 - 0.05, 0) = 0.22
weight = 0.22^7 = 0.000025
L_car = -0.000025 × log(0.22) = -0.000025 × (-1.514) = 0.000038

Hmm, weight quá nhỏ? Thực ra khi p=0.73 cho negative, p_neg=0.22 nhỏ,
nên log(0.22) lớn (âm), nhưng weight cũng nhỏ.
Tuy nhiên so với BCE thông thường, ASL vẫn phạt hard negative nhưng
KHÔNG phạt easy negative → gradient tập trung vào positive + hard negative.
```

#### Tham số trong code

| Dùng cho | γ⁺ | γ⁻ | clip (m) |
|----------|-----|-----|----------|
| Tag recognition (`tagging_loss_function`) | 0 | 7 | 0.05 |
| Text alignment - RAM++ (`text_alignment_loss_function`) | 0 | 4 | 0.05 |

---

### 1.2 L1 Loss — CLIP Distillation

**Dùng trong:** RAM và RAM++ (không có trong Tag2Text)

```
loss_dis = L1Loss(image_cls_embeds, clip_feature)
         = mean(|image_cls_embeds - clip_feature|)
```

#### Ý nghĩa

- `image_cls_embeds`: CLS token từ visual encoder (Swin) → project xuống 512-dim
- `clip_feature`: Image feature từ CLIP ViT-B/16 (frozen, 512-dim)

Mục đích: Ép visual encoder của model **học được representation giống CLIP**. Vì CLIP đã được train trên hàng trăm triệu cặp image-text, nên representation của nó rất tốt cho open-vocabulary understanding.

#### Tại sao dùng L1 thay vì L2?

L1 loss ít nhạy với outlier hơn L2, giúp training ổn định hơn khi distill từ model lớn.

---

### 1.3 Cross-Entropy Loss — Text Generation (Captioning)

**Dùng trong:** RAM và Tag2Text (không có trong RAM++)

```
loss_t2t = CrossEntropyLoss(decoder_output_logits, decoder_targets)
```

#### Cơ chế

1. Caption gốc: `"a picture of a dog running on grass"`
2. Tokenize → `decoder_input_ids`
3. Đặt `[BOS]` token ở đầu
4. Mask prompt tokens (`"a picture of "`) bằng `-100` → không tính loss cho prompt
5. Text decoder (BertLMHeadModel) dự đoán token tiếp theo
6. Cross-entropy giữa predicted token và actual next token

---

### 1.4 Tổng hợp Loss cho từng model

#### RAM++

```
total_loss = loss_tag + loss_dis + loss_alignment

loss_tag:       AsymmetricLoss(γ⁻=7)  — nhận diện tag từ image
loss_dis:       L1Loss                 — distill từ CLIP
loss_alignment: AsymmetricLoss(γ⁻=4)  — align caption text với image
```

Ba loss này cộng trực tiếp, **không có trọng số**.

#### RAM

```
total_loss = loss_t2t + loss_tag / (loss_tag / loss_t2t).detach() + loss_dis

Đơn giản hóa:
= loss_t2t + loss_t2t.detach() + loss_dis
           ↑ loss_tag được normalize về cùng magnitude với loss_t2t
```

Giải thích `loss_tag / (loss_tag / loss_t2t).detach()`:
- `(loss_tag / loss_t2t).detach()` = tỉ lệ giữa 2 loss, **không có gradient**
- Chia loss_tag cho tỉ lệ này → scale loss_tag về **cùng magnitude** với loss_t2t
- Gradient vẫn chảy qua loss_tag bình thường (chỉ detach phần tỉ lệ)

#### Tag2Text

```
total_loss = loss_t2t + loss_tag / (loss_tag / loss_t2t).detach()

Tương tự RAM nhưng KHÔNG có loss_dis (không dùng CLIP distillation).
```

---

## 2. Giải thích chi tiết các Metrics đánh giá

Tất cả metrics được tính trong `ram/utils/metrics.py` và `batch_inference.py`.

### 2.1 Precision (Độ chính xác)

> "Trong những tag mà model dự đoán, bao nhiêu % là đúng?"

```
                    Số tag dự đoán đúng (TP)
Precision(k) = ─────────────────────────────────
                Tổng tag model dự đoán (TP + FP)
```

**Ví dụ cụ thể:**

Ảnh thật có tags: `{dog, grass, outdoor}`

Model dự đoán: `{dog, cat, grass, car}`

- TP = 2 (dog, grass đúng)
- FP = 2 (cat, car sai)
- Precision = 2 / (2+2) = **0.50** (50%)

→ Model dự đoán đúng 50% trong số những gì nó đoán.

### 2.2 Recall (Độ phủ)

> "Trong những tag thật sự có, model tìm được bao nhiêu %?"

```
                 Số tag dự đoán đúng (TP)
Recall(k) = ──────────────────────────────────
              Tổng tag thật sự có (TP + FN)
```

**Tiếp ví dụ trên:**

- TP = 2 (dog, grass)
- FN = 1 (outdoor bị bỏ sót)
- Recall = 2 / (2+1) = **0.67** (67%)

→ Model tìm được 67% tags thật sự có trong ảnh.

### 2.3 Cách code tính Precision/Recall theo từng CLASS

Code trong `get_PR()` tính **per-class** trên toàn bộ dataset:

```
Với tag k (ví dụ "dog"):
  - Duyệt qua TẤT CẢ ảnh trong dataset
  - TP_k = số ảnh mà model đoán "dog" VÀ ảnh thật có "dog"
  - FP_k = số ảnh mà model đoán "dog" NHƯNG ảnh thật KHÔNG có "dog"
  - FN_k = số ảnh mà model KHÔNG đoán "dog" NHƯNG ảnh thật CÓ "dog"

  Precision_k = TP_k / (TP_k + FP_k)
  Recall_k    = TP_k / (TP_k + FN_k)
```

### 2.4 CP và CR

```
CP = mean(Precision_k)  cho tất cả k = 1..K
CR = mean(Recall_k)     cho tất cả k = 1..K
```

Đây là **macro-average**: trung bình trên tất cả classes, mỗi class có trọng số bằng nhau bất kể class đó xuất hiện nhiều hay ít.

### 2.5 Average Precision (AP) — Cho 1 class

> AP đo chất lượng **ranking** của model, không phụ thuộc threshold.

**Cách tính step-by-step:**

Giả sử có 6 ảnh, tag "dog", model cho score:

| Ảnh | Score | Ground Truth |
|-----|-------|-------------|
| A   | 0.95  | 1 (có dog)  |
| B   | 0.80  | 0 (không)   |
| C   | 0.75  | 1 (có dog)  |
| D   | 0.60  | 1 (có dog)  |
| E   | 0.40  | 0 (không)   |
| F   | 0.20  | 0 (không)   |

Đã sort theo score giảm dần. Tổng positive = 3.

Đi từ trên xuống, tại mỗi vị trí positive, tính Precision@i:

| Vị trí i | Ảnh | GT | #Positive tới i | Precision@i | Là positive? |
|-----------|-----|-----|-----------------|-------------|-------------|
| 1 | A | 1 | 1 | 1/1 = **1.00** | ✅ cộng |
| 2 | B | 0 | 1 | 1/2 = 0.50 | ❌ bỏ qua |
| 3 | C | 1 | 2 | 2/3 = **0.67** | ✅ cộng |
| 4 | D | 1 | 3 | 3/4 = **0.75** | ✅ cộng |
| 5 | E | 0 | 3 | 3/5 = 0.60 | ❌ bỏ qua |
| 6 | F | 0 | 3 | 3/6 = 0.50 | ❌ bỏ qua |

```
AP = (1.00 + 0.67 + 0.75) / 3 = 2.42 / 3 = 0.807
```

#### Ý nghĩa trực quan

- AP = 1.0: Model xếp **tất cả** positive lên top, không có negative nào xen vào
- AP thấp: Positive bị xen lẫn hoặc nằm dưới negative trong ranking

### 2.6 mAP (mean Average Precision)

```
mAP = (AP_tag1 + AP_tag2 + ... + AP_tagK) / K
```

Trung bình AP trên tất cả K classes.

### 2.7 So sánh mAP vs CP/CR

| Metric | Phụ thuộc threshold? | Đo cái gì? |
|--------|---------------------|------------|
| **mAP** | ❌ Không | Chất lượng ranking (score ordering) |
| **CP** | ✅ Có | Precision sau khi chọn threshold |
| **CR** | ✅ Có | Recall sau khi chọn threshold |

→ **mAP** là metric "công bằng" nhất vì không phụ thuộc threshold. CP/CR hữu ích khi bạn đã chọn threshold cụ thể cho production.

---

## 3. Hướng dẫn tải & chuẩn bị dữ liệu

### 3.1 Tải pretrained model weights

Bạn cần tải checkpoint tương ứng model muốn dùng:

| Model | File | Tải từ |
|-------|------|--------|
| RAM++ | `ram_plus_swin_large_14m.pth` | [HuggingFace](https://huggingface.co/xinyu1205/recognize-anything-plus-model) |
| RAM | `ram_swin_large_14m.pth` | [HuggingFace](https://huggingface.co/xinyu1205/recognize_anything_model) |
| Tag2Text | `tag2text_swin_14m.pth` | [HuggingFace](https://huggingface.co/xinyu1205/recognize_anything_model) |

Đặt vào thư mục `pretrained/`:
```
recognize-anything/
└── pretrained/
    ├── ram_plus_swin_large_14m.pth
    ├── ram_swin_large_14m.pth
    └── tag2text_swin_14m.pth
```

### 3.2 Chuẩn bị dữ liệu training

#### Bước 1: Tổ chức thư mục ảnh

```
D:/data/my_dataset/
├── images/
│   ├── img_0001.jpg
│   ├── img_0002.jpg
│   ├── img_0003.jpg
│   └── ...
```

#### Bước 2: Xác định tag-id từ tag list

Mở file `ram/data/ram_tag_list.txt`. Mỗi dòng là 1 tag, **số dòng (0-based) = tag-id**.

```
Dòng 0:   "airport terminal"     → tag_id = 0
Dòng 1:   "aisle"                → tag_id = 1
...
Dòng 123: "dog"                  → tag_id = 123
...
Dòng 4584: "zoom"               → tag_id = 4584
```

Script Python để tìm tag-id:

```python
# Đọc tag list
with open('ram/data/ram_tag_list.txt', 'r', encoding='utf-8') as f:
    tag_list = [line.strip() for line in f]

# Tạo mapping text → id
tag_to_id = {tag: idx for idx, tag in enumerate(tag_list)}

# Tìm id
print(tag_to_id.get("dog"))      # ví dụ: 123
print(tag_to_id.get("grass"))    # ví dụ: 456
print(tag_to_id.get("outdoor"))  # ví dụ: 789
```

#### Bước 3: Tạo file annotation JSON

Tạo file `datasets/train/my_train.json`:

```json
[
  {
    "image_path": "images/img_0001.jpg",
    "caption": [
      "a brown dog running on green grass",
      "a dog playing outdoors in a park"
    ],
    "union_label_id": [123, 456, 789, 1024],
    "parse_label_id": [
      [123, 456],
      [123, 789, 1024]
    ]
  },
  {
    "image_path": "images/img_0002.jpg",
    "caption": [
      "a red car parked on the street"
    ],
    "union_label_id": [200, 350, 2100],
    "parse_label_id": [
      [200, 350, 2100]
    ]
  }
]
```

**Giải thích từng field:**

| Field | Bắt buộc? | Ý nghĩa |
|-------|-----------|---------|
| `image_path` | ✅ | Đường dẫn ảnh **tương đối** so với `image_path_root` trong config |
| `caption` | ✅ | List caption mô tả ảnh. Training sẽ chọn ngẫu nhiên 1 caption mỗi iteration |
| `union_label_id` | ✅ (RAM/RAM++) | Tất cả tag-ids liên quan ảnh (union). Dùng làm **ground truth cho tagging loss** |
| `parse_label_id` | ✅ | Mỗi sub-list chứa tag-ids **tương ứng 1 caption**. Dùng cho text generation loss |

#### Bước 4: Nếu bạn KHÔNG có caption

Nếu dataset chỉ có ảnh + tags (không có caption), bạn có thể **tự tạo caption từ tags**:

```python
import json

# Giả sử bạn có data dạng: [{image, tags: ["dog", "grass", ...]}]
my_data = [...]

with open('ram/data/ram_tag_list.txt', 'r', encoding='utf-8') as f:
    tag_list = [line.strip() for line in f]
tag_to_id = {tag: idx for idx, tag in enumerate(tag_list)}

annotations = []
for item in my_data:
    image_path = item["image"]
    tags = item["tags"]

    # Map text tags → ids (bỏ qua tag không có trong list)
    tag_ids = [tag_to_id[t] for t in tags if t in tag_to_id]

    if not tag_ids:
        continue

    # Tạo caption đơn giản từ tags
    caption = "a photo of " + ", ".join(tags)

    annotations.append({
        "image_path": image_path,
        "caption": [caption],
        "union_label_id": tag_ids,
        "parse_label_id": [tag_ids]  # parse = union vì chỉ có 1 caption
    })

with open('datasets/train/my_train.json', 'w') as f:
    json.dump(annotations, f, indent=2)

print(f"Created {len(annotations)} annotations")
```

#### Bước 5: Tạo file config YAML

Copy và sửa `ram/configs/finetune.yaml`:

Tạo file `ram/configs/my_finetune.yaml`:

```yaml
train_file: [
  'datasets/train/my_train.json',
]
image_path_root: "D:/data/my_dataset"

# Backbone: swin_l cho RAM/RAM++, swin_b cho Tag2Text
vit: 'swin_l'
vit_grad_ckpt: False
vit_ckpt_layer: 0

image_size: 384
batch_size: 8          # Giảm nếu GPU nhỏ (8GB → batch 2-4, 24GB → batch 8-16)

# Optimizer
weight_decay: 0.05
init_lr: 5e-06         # Learning rate nhỏ cho finetune
min_lr: 0
max_epoch: 5           # Tăng nếu dataset nhỏ

class_num: 4585        # Giữ nguyên, đây là số tags trong ram_tag_list.txt
```

**Lưu ý quan trọng về `image_path_root`:**

Code sẽ nối: `os.path.join(image_path_root, ann['image_path'])` để ra đường dẫn đầy đủ.

Ví dụ:
```
image_path_root = "D:/data/my_dataset"
image_path      = "images/img_0001.jpg"
→ Full path    = "D:/data/my_dataset/images/img_0001.jpg"
```

---

## 4. Hướng dẫn Train Step-by-Step

### Step 1: Cài đặt môi trường

```bash
# Tạo virtual environment (khuyến nghị)
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Linux/Mac

# Cài dependencies
cd recognize-anything
pip install -r requirements.txt

# Kiểm tra CUDA
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

**Yêu cầu phần cứng:**
| Config | GPU RAM tối thiểu |
|--------|-------------------|
| RAM++ Swin-L, batch=2, image=384 | ~12 GB |
| RAM++ Swin-L, batch=8, image=384 | ~24 GB |
| Tag2Text Swin-B, batch=8, image=384 | ~16 GB |
| Pretrain, image=224 | Ít hơn ~30% so với 384 |

### Step 2: Kiểm tra dữ liệu

```python
# Chạy script kiểm tra nhanh
import json
from PIL import Image
import os

config_root = "D:/data/my_dataset"  # image_path_root
with open("datasets/train/my_train.json") as f:
    data = json.load(f)

print(f"Total samples: {len(data)}")

# Kiểm tra 5 sample đầu
for i, ann in enumerate(data[:5]):
    full_path = os.path.join(config_root, ann["image_path"])
    exists = os.path.exists(full_path)
    img = Image.open(full_path) if exists else None
    print(f"[{i}] path={ann['image_path']}, exists={exists}, "
          f"size={img.size if img else 'N/A'}, "
          f"captions={len(ann['caption'])}, "
          f"union_tags={len(ann.get('union_label_id', []))}, "
          f"parse_tags={len(ann['parse_label_id'])}")
```

### Step 3: Chạy Finetune (1 GPU)

#### Finetune RAM++

```bash
python finetune.py ^
  --model-type ram_plus ^
  --config ram/configs/my_finetune.yaml ^
  --checkpoint pretrained/ram_plus_swin_large_14m.pth ^
  --output-dir output/finetune_ram_plus ^
  --device cuda
```

#### Finetune RAM

```bash
python finetune.py ^
  --model-type ram ^
  --config ram/configs/my_finetune.yaml ^
  --checkpoint pretrained/ram_swin_large_14m.pth ^
  --output-dir output/finetune_ram ^
  --device cuda
```

#### Finetune Tag2Text

Lưu ý: Tag2Text dùng `swin_b`, sửa config `vit: 'swin_b'`

```bash
python finetune.py ^
  --model-type tag2text ^
  --config ram/configs/my_finetune_tag2text.yaml ^
  --checkpoint pretrained/tag2text_swin_14m.pth ^
  --output-dir output/finetune_tag2text ^
  --device cuda
```

### Step 4: Chạy Finetune (Multi-GPU với DDP)

```bash
python -m torch.distributed.launch --nproc_per_node=2 finetune.py ^
  --model-type ram_plus ^
  --config ram/configs/my_finetune.yaml ^
  --checkpoint pretrained/ram_plus_swin_large_14m.pth ^
  --output-dir output/finetune_ram_plus
```

Hoặc dùng `torchrun` (PyTorch >= 1.10):

```bash
torchrun --nproc_per_node=2 finetune.py ^
  --model-type ram_plus ^
  --config ram/configs/my_finetune.yaml ^
  --checkpoint pretrained/ram_plus_swin_large_14m.pth ^
  --output-dir output/finetune_ram_plus
```

**Lưu ý DDP:**
- `batch_size` trong config là **per-GPU**. Với 2 GPU, batch_size=8 → effective batch = 16.
- Nếu chạy 1 GPU mà không dùng launcher, code tự detect và set `distributed=False`.

### Step 5: Theo dõi training

#### Log file

Mỗi epoch, file `output/finetune_ram_plus/log.txt` ghi 1 dòng JSON:

```json
{"train_lr": "0.000005", "train_loss_tag": "12.345", "train_loss_dis": "0.234", "train_loss_alignment": "1.567", "epoch": 0}
{"train_lr": "0.000003", "train_loss_tag": "8.901", "train_loss_dis": "0.189", "train_loss_alignment": "1.123", "epoch": 1}
```

#### Checkpoint files

```
output/finetune_ram_plus/
├── config.yaml          # Config đã dùng (để reproduce)
├── log.txt              # Training log
├── checkpoint_00.pth    # Checkpoint epoch 0
├── checkpoint_01.pth    # Checkpoint epoch 1
└── ...
```

Mỗi checkpoint chứa:
```python
{
    'model': model.state_dict(),
    'optimizer': optimizer.state_dict(),
    'config': config,
    'epoch': epoch
}
```

#### Đọc log bằng Python

```python
import json
import matplotlib.pyplot as plt

losses = {'loss_tag': [], 'loss_dis': [], 'loss_alignment': []}
with open('output/finetune_ram_plus/log.txt') as f:
    for line in f:
        entry = json.loads(line)
        for key in losses:
            train_key = f'train_{key}'
            if train_key in entry:
                losses[key].append(float(entry[train_key]))

for key, vals in losses.items():
    plt.plot(vals, label=key)
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.savefig('training_loss.png')
plt.show()
```

### Step 6: Inference với model đã finetune

```python
import torch
from PIL import Image
from ram.models import ram_plus
from ram import inference_ram, get_transform

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load model từ checkpoint finetune
checkpoint_path = 'output/finetune_ram_plus/checkpoint_01.pth'

model = ram_plus(pretrained=checkpoint_path,
                 image_size=384,
                 vit='swin_l')
model.eval().to(device)

# Transform và inference
transform = get_transform(image_size=384)
image = transform(Image.open('test_image.jpg')).unsqueeze(0).to(device)

tags_en, tags_cn = inference_ram(image, model)
print("Tags (EN):", tags_en)
print("Tags (CN):", tags_cn)
```

### Step 7: Đánh giá model (Evaluation)

Sau khi finetune, đánh giá trên benchmark:

```bash
python batch_inference.py ^
  --model-type ram_plus ^
  --checkpoint output/finetune_ram_plus/checkpoint_01.pth ^
  --dataset openimages_common_214 ^
  --input-size 384 ^
  --batch-size 64 ^
  --output-dir outputs/eval_finetune_ram_plus
```

Kết quả sẽ in ra:
```
mAP: 86.5        ← Chất lượng ranking tổng thể
CP: 82.3         ← Precision trung bình
CR: 75.1         ← Recall trung bình
```

---

## Phụ lục: Checklist trước khi train

- [ ] **GPU CUDA** hoạt động (`torch.cuda.is_available() == True`)
- [ ] **Dependencies** đã cài (`pip install -r requirements.txt`)
- [ ] **Pretrained checkpoint** đã tải và đặt đúng path
- [ ] **File JSON annotation** đúng format (có `image_path`, `caption`, `union_label_id`, `parse_label_id`)
- [ ] **Ảnh tồn tại** tại `os.path.join(image_path_root, image_path)`
- [ ] **Tag-ids** nằm trong range `[0, 4584]` (tổng 4585 tags)
- [ ] **Config YAML** đã sửa `train_file`, `image_path_root`, `batch_size` phù hợp GPU
- [ ] **Thư mục output** sẽ được tạo tự động

## Phụ lục: Troubleshooting

| Lỗi | Nguyên nhân | Cách sửa |
|-----|-------------|----------|
| `CUDA out of memory` | batch_size quá lớn | Giảm `batch_size` trong config YAML |
| `FileNotFoundError` cho ảnh | `image_path_root` + `image_path` sai | Kiểm tra lại đường dẫn, chạy script kiểm tra ở Step 2 |
| `KeyError: 'union_label_id'` | JSON thiếu field | Thêm field `union_label_id` vào mỗi annotation |
| `Not using distributed mode` | Chạy 1 GPU không qua launcher | Bình thường, code tự xử lý. Nếu muốn DDP thì dùng `torch.distributed.launch` |
| `model_clip` not defined (Tag2Text) | Tag2Text không dùng CLIP | Dùng `--model-type tag2text` với config `finetune_tag2text.yaml` |
| Loss không giảm | LR quá nhỏ/lớn hoặc data sai | Kiểm tra data, thử `init_lr: 1e-5` |
