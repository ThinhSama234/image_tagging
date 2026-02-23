# Lộ trình 3 ngày: Làm chủ dự án Recognize-Anything

> **Mục tiêu cuối:** Hiểu kiến trúc, chạy inference, finetune được, và có web demo hoạt động.  
> **Tài liệu tham khảo:** `DOCUMENTATION.md` (kiến trúc + API) | `TRAINING_GUIDE.md` (loss/metric/train chi tiết)

---

## Ngày 1: Chạy được + Hiểu input/output

### Buổi sáng (2-3h): Setup môi trường & chạy demo

#### 1.1 Cài đặt (30 phút)

```bash
# Clone/mở project
cd recognize-anything

# Tạo virtual environment
python -m venv venv
venv\Scripts\activate            # Windows
# source venv/bin/activate       # Linux/Mac

# Cài dependencies
pip install -r requirements.txt

# Verify
python -c "import torch; print('CUDA:', torch.cuda.is_available())"
```

#### 1.2 Tải pretrained weights (15 phút)

Tải 1 trong 3 (hoặc cả 3) vào thư mục `pretrained/`:

| Model | Link | Size |
|-------|------|------|
| RAM++ | [HuggingFace](https://huggingface.co/xinyu1205/recognize-anything-plus-model) | ~5.6 GB |
| RAM | [HuggingFace](https://huggingface.co/xinyu1205/recognize_anything_model) | ~5.6 GB |
| Tag2Text | [HuggingFace](https://huggingface.co/xinyu1205/recognize_anything_model) | ~3.8 GB |

#### 1.3 Chạy inference từng model (1h)

**RAM++ (tagging):**
```bash
python inference_ram_plus.py --image images/demo/demo1.jpg --pretrained pretrained/ram_plus_swin_large_14m.pth
```
→ Quan sát: `Image Tags: dog, grass, outdoor, ...`

**RAM (tagging):**
```bash
python inference_ram.py --image images/demo/demo1.jpg --pretrained pretrained/ram_swin_large_14m.pth
```
→ So sánh tags với RAM++.

**Tag2Text (tagging + captioning):**
```bash
python inference_tag2text.py --image images/demo/demo1.jpg --pretrained pretrained/tag2text_swin_14m.pth
```
→ Quan sát thêm: `Image Caption: ...`

**Open-set (nhận tag chưa từng train):**
```bash
python inference_ram_openset.py --image images/openset_example.jpg --pretrained pretrained/ram_swin_large_14m.pth
```

**Thử ảnh của bạn:** thay `--image` bằng ảnh bất kỳ, xem model nhận diện ra gì.

#### 1.4 Chạy web demo (30 phút)

```bash
cd demo_app
pip install gradio
python app.py
```
→ Mở browser, upload ảnh, xem tags + caption trực quan.

**✅ Checkpoint Ngày 1 sáng:**
- [ ] 3 model inference chạy được
- [ ] Hiểu output: RAM/RAM++ → tags, Tag2Text → tags + caption
- [ ] Web demo chạy local

---

### Buổi chiều (2-3h): Đọc hiểu pipeline code

#### 1.5 Đọc luồng dữ liệu (1h)

Đọc theo thứ tự, mỗi file chỉ cần nắm ý chính:

| Thứ tự | File | Cần hiểu |
|--------|------|----------|
| 1 | `ram/data/dataset.py` | Dataset trả về gì: `(image, image_224, caption, image_tag, parse_tag)` |
| 2 | `ram/data/__init__.py` | Augmentation pipeline: RandomResizedCrop → RandomAugment → Normalize |
| 3 | `ram/transform.py` | Inference transform (đơn giản hơn training) |

#### 1.6 Đọc model architecture (1h)

Chọn 1 model bạn quan tâm nhất (khuyến nghị RAM++):

| File | Cần hiểu |
|------|----------|
| `ram/models/ram_plus.py` | `forward()` → 3 loss. `generate_tag()` → inference |
| `ram/models/utils.py` | `AsymmetricLoss` (công thức xem `TRAINING_GUIDE.md` mục 1.1) |
| `ram/configs/q2l_config.json` | Tagging head: 2 layers, cross-attention only |

#### 1.7 Đọc training script (30 phút)

| File | Cần hiểu |
|------|----------|
| `finetune.py` | `train_ram_plus()`: 1 batch → CLIP encode → model forward → loss → backward |
| `ram/configs/finetune.yaml` | Các hyperparameter: lr, batch_size, epoch, image_size |

**✅ Checkpoint Ngày 1 chiều:**
- [ ] Vẽ được sơ đồ: Image → Swin → Tagging Head → Tags
- [ ] Biết 3 loss của RAM++: tag (ASL) + distill (L1) + alignment (ASL)
- [ ] Biết config YAML chứa gì

---

## Ngày 2: Chuẩn bị data & Finetune

### Buổi sáng (2-3h): Tạo dataset + config

#### 2.1 Tạo toy dataset (1h)

Chuẩn bị 50-200 ảnh. Tạo JSON annotation:

```python
import json

# Đọc tag list để map text → id
with open('ram/data/ram_tag_list.txt', 'r', encoding='utf-8') as f:
    tag_list = [line.strip() for line in f]
tag_to_id = {tag: idx for idx, tag in enumerate(tag_list)}

# Tạo annotations (thay bằng data thật của bạn)
annotations = []
my_data = [
    {"image": "img_001.jpg", "tags": ["dog", "grass", "outdoor"]},
    {"image": "img_002.jpg", "tags": ["car", "road", "building"]},
    # ... thêm ảnh
]

for item in my_data:
    tag_ids = [tag_to_id[t] for t in item["tags"] if t in tag_to_id]
    if not tag_ids:
        continue
    caption = "a photo of " + ", ".join(item["tags"])
    annotations.append({
        "image_path": item["image"],
        "caption": [caption],
        "union_label_id": tag_ids,
        "parse_label_id": [tag_ids]
    })

with open('datasets/train/my_train.json', 'w') as f:
    json.dump(annotations, f, indent=2)
print(f"Created {len(annotations)} samples")
```

#### 2.2 Verify dataset (30 phút)

```python
import json, os
from PIL import Image

root = "D:/data/my_images"  # sửa theo thư mục ảnh của bạn
with open('datasets/train/my_train.json') as f:
    data = json.load(f)

errors = 0
for ann in data:
    path = os.path.join(root, ann["image_path"])
    if not os.path.exists(path):
        print(f"MISSING: {path}")
        errors += 1
    for tid in ann["union_label_id"]:
        if tid < 0 or tid > 4584:
            print(f"BAD TAG ID: {tid} in {ann['image_path']}")
            errors += 1

print(f"\nTotal: {len(data)} samples, {errors} errors")
```

#### 2.3 Tạo config YAML (15 phút)

Tạo `ram/configs/my_finetune.yaml`:

```yaml
train_file: ['datasets/train/my_train.json']
image_path_root: "D:/data/my_images"

vit: 'swin_l'
vit_grad_ckpt: False
vit_ckpt_layer: 0

image_size: 384
batch_size: 4          # GPU 8GB → 2, GPU 24GB → 8-16

weight_decay: 0.05
init_lr: 5e-06
min_lr: 0
max_epoch: 3
warmup_steps: 100

class_num: 4585
```

**✅ Checkpoint Ngày 2 sáng:**
- [ ] File JSON annotation tạo xong, verify 0 errors
- [ ] Config YAML sửa đúng path

---

### Buổi chiều (2-3h): Chạy finetune + theo dõi

#### 2.4 Chạy finetune (1-2h chờ train)

```bash
python finetune.py ^
  --model-type ram_plus ^
  --config ram/configs/my_finetune.yaml ^
  --checkpoint pretrained/ram_plus_swin_large_14m.pth ^
  --output-dir output/my_finetune ^
  --device cuda
```

Nếu gặp `CUDA out of memory`: giảm `batch_size` trong YAML.

#### 2.5 Theo dõi loss (trong khi chờ)

```python
# Chạy trong terminal khác
import json, time

while True:
    try:
        with open('output/my_finetune/log.txt') as f:
            lines = f.readlines()
        for line in lines[-3:]:
            print(json.loads(line))
    except:
        print("Waiting for log...")
    time.sleep(30)
```

Loss tốt: `loss_tag` giảm dần qua mỗi epoch.

#### 2.6 Test model sau finetune (30 phút)

```bash
python inference_ram_plus.py ^
  --image test_image.jpg ^
  --pretrained output/my_finetune/checkpoint_02.pth
```

So sánh tags trước/sau finetune.

**✅ Checkpoint Ngày 2 chiều:**
- [ ] Finetune chạy hết ≥ 2 epoch không crash
- [ ] Loss giảm
- [ ] Inference với checkpoint mới cho kết quả hợp lý

---

## Ngày 3: Evaluation + Web Demo + Tổng kết

### Buổi sáng (2-3h): Evaluation & Open-set

#### 3.1 Chạy batch evaluation (1h)

```bash
python batch_inference.py ^
  --model-type ram_plus ^
  --checkpoint output/my_finetune/checkpoint_02.pth ^
  --dataset openimages_common_214 ^
  --input-size 384 ^
  --batch-size 64 ^
  --output-dir outputs/eval_my_model
```

Đọc kết quả:
```bash
type outputs/eval_my_model/summary.txt
# mAP: xx.x
# CP: xx.x
# CR: xx.x
```

Xem `TRAINING_GUIDE.md` mục 2 để hiểu ý nghĩa từng metric.

#### 3.2 Thử open-set (30 phút)

```bash
python inference_ram_plus_openset.py ^
  --image images/openset_example.jpg ^
  --pretrained output/my_finetune/checkpoint_02.pth ^
  --llm_tag_des datasets/openimages_rare_200/openimages_rare_200_llm_tag_descriptions.json
```

#### 3.3 So sánh model gốc vs finetune (30 phút)

Chạy evaluation cho cả model gốc:
```bash
python batch_inference.py ^
  --model-type ram_plus ^
  --checkpoint pretrained/ram_plus_swin_large_14m.pth ^
  --dataset openimages_common_214 ^
  --input-size 384 ^
  --output-dir outputs/eval_original
```

So sánh `summary.txt` của 2 runs.

**✅ Checkpoint Ngày 3 sáng:**
- [ ] Có số liệu mAP/CP/CR cho model gốc và model finetune
- [ ] Hiểu open-set hoạt động

---

### Buổi chiều (2-3h): Web Demo + Tổng kết

#### 3.4 Chạy web demo với model finetune (30 phút)

```bash
cd demo_app
python app.py --checkpoint ../output/my_finetune/checkpoint_02.pth
```

Upload ảnh → xem tags + caption + confidence scores.

#### 3.5 Tổng kết kiến thức (1h)

Đọc lại 3 file tài liệu:
1. `DOCUMENTATION.md` — kiến trúc tổng, so sánh 3 model
2. `TRAINING_GUIDE.md` — loss/metric chi tiết, data format
3. `ROADMAP.md` — lộ trình bạn đã đi qua

#### 3.6 Checklist hoàn thành

- [ ] Chạy inference 3 model (RAM++, RAM, Tag2Text)
- [ ] Hiểu kiến trúc: Swin → Tagging Head (Q2L) → Tags
- [ ] Hiểu 3 loss: ASL (tagging), L1 (CLIP distill), CE (text gen)
- [ ] Hiểu metrics: mAP, CP, CR
- [ ] Tạo dataset JSON đúng format
- [ ] Finetune thành công ≥ 1 model
- [ ] Evaluation có số liệu
- [ ] Web demo chạy được
- [ ] Hiểu open-set recognition

---

## Tech Stack tổng hợp

| Thành phần | Công nghệ |
|------------|-----------|
| **Deep Learning** | PyTorch, torchvision |
| **Vision Backbone** | Swin Transformer (timm) |
| **Text Encoder/Decoder** | BERT (HuggingFace transformers) |
| **Knowledge Distillation** | OpenAI CLIP (ViT-B/16) |
| **Training** | PyTorch DDP (distributed) |
| **Data Augmentation** | Custom RandomAugment (OpenCV) |
| **Web Demo** | Gradio |
| **Config** | YAML + JSON |

## Tech Stack Web Demo

| Thành phần | Lý do chọn |
|------------|-----------|
| **Gradio** | 1 file Python, không cần frontend, có UI upload ảnh + hiển thị kết quả sẵn |
| **Pillow** | Xử lý ảnh |
| **PyTorch** | Load model + inference |

> Gradio là lựa chọn tối ưu cho demo ML vì: zero frontend code, auto tạo UI đẹp, hỗ trợ share link public, tích hợp HuggingFace Spaces.
