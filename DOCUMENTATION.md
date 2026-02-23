# Recognize Anything - Tài liệu kỹ thuật tổng hợp

> **Tác giả source code:** Xinyu Huang  
> **Paper:** RAM++ - "Open-Set Image Tagging with Multi-Grained Text Supervision" ([arXiv:2310.15200](https://arxiv.org/abs/2310.15200))  
> **Paper:** RAM - "Recognize Anything: A Strong Image Tagging Model" ([recognize-anything.github.io](https://recognize-anything.github.io/))  
> **Paper:** Tag2Text - "Tag2Text: Guiding Vision-Language Model via Image Tagging" ([arXiv:2303.05657](https://arxiv.org/abs/2303.05657))

---

## Mục lục

1. [Tổng quan kiến trúc](#1-tổng-quan-kiến-trúc)
2. [Các mô hình sử dụng](#2-các-mô-hình-sử-dụng)
3. [Xử lý dữ liệu (Data Processing)](#3-xử-lý-dữ-liệu-data-processing)
4. [Hướng dẫn Pretrain](#4-hướng-dẫn-pretrain)
5. [Hướng dẫn Finetune](#5-hướng-dẫn-finetune)
6. [Hướng dẫn Inference (Chạy dự đoán)](#6-hướng-dẫn-inference-chạy-dự-đoán)
7. [Batch Inference & Evaluation](#7-batch-inference--evaluation)
8. [Cấu trúc thư mục dự án](#8-cấu-trúc-thư-mục-dự-án)
9. [Cấu hình & Hyperparameters](#9-cấu-hình--hyperparameters)
10. [Loss Functions](#10-loss-functions)
11. [Open-Set Recognition](#11-open-set-recognition)

---

## 1. Tổng quan kiến trúc

### Sơ đồ kiến trúc tổng thể

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        INPUT IMAGE (384x384)                            │
└──────────────────────────────┬──────────────────────────────────────────┘
                               │
                    ┌──────────▼──────────┐
                    │   Visual Encoder     │
                    │  (Swin Transformer   │
                    │   hoặc ViT)          │
                    └──────────┬──────────┘
                               │
                        image_embeds
                               │
              ┌────────────────┼────────────────┐
              │                │                │
    ┌─────────▼─────────┐     │     ┌──────────▼──────────┐
    │  Image Projection  │     │     │   CLS Token Embed    │
    │  (Linear → 512)    │     │     │   (image_cls_embeds)  │
    └─────────┬─────────┘     │     └──────────┬──────────┘
              │                │                │
              │                │     ┌──────────▼──────────┐
              │                │     │  CLIP Distillation    │
              │                │     │  L1 Loss (loss_dis)   │
              │                │     └─────────────────────┘
              │                │
    ┌─────────▼─────────────────▼─────────────┐
    │         Tagging Head (Query2Label)        │
    │   BertModel (cross-attention only)        │
    │   Input: label_embed + image_embeds       │
    │   Output: tagging logits                  │
    └─────────────────┬───────────────────────┘
                      │
            ┌─────────▼─────────┐
            │   FC Layer → σ     │
            │   (Sigmoid)        │
            └─────────┬─────────┘
                      │
            ┌─────────▼─────────┐
            │  Predicted Tags    │
            │  (4585 categories) │
            └───────────────────┘
```

### Luồng xử lý chính

1. **Image Encoding**: Ảnh đầu vào được encode bởi Swin Transformer (hoặc ViT) thành image embeddings
2. **Image Projection**: Project image embeddings xuống không gian 512 chiều
3. **CLIP Distillation**: CLS token của image embedding được align với CLIP image feature (L1 loss)
4. **Tag Recognition**: Label embeddings (frozen) được dùng làm query, cross-attend với image embeddings qua Tagging Head (Query2Label decoder)
5. **Classification**: FC layer + sigmoid để dự đoán xác suất cho mỗi tag

---

## 2. Các mô hình sử dụng

### 2.1 RAM++ (Recognize Anything Plus)

**File:** `ram/models/ram_plus.py` → class `RAM_plus`

**Đặc điểm chính:**
- Sử dụng **Multi-Grained Text Supervision** với LLM-generated tag descriptions
- Mỗi tag có **51 descriptions** (1 mặc định + 50 từ LLM) → `label_embed` shape: `(4585 × 51, 512)`
- **Description Reweighting**: Dùng image CLS embedding để tính attention weight cho các descriptions, sau đó weighted sum thành 1 embedding/tag
- Có thêm **Image-Text Alignment Loss**: Align caption text embedding (từ CLIP) với image qua tagging head

**Các thành phần:**
| Thành phần | Mô tả |
|---|---|
| `visual_encoder` | Swin Transformer Large/Base hoặc ViT |
| `image_proj` | Linear(vision_width → 512) |
| `tagging_head` | BertModel (Q2L config, 2 layers, cross-attention only) |
| `wordvec_proj` | Linear(512 → hidden_size) hoặc Identity |
| `fc` | Linear(hidden_size → 1) |
| `label_embed` | Frozen, shape (4585×51, 512) |
| `reweight_scale` | Learnable temperature parameter |

**3 Loss functions trong training:**
- `loss_tag`: AsymmetricLoss(γ⁻=7, γ⁺=0, clip=0.05) — tag recognition
- `loss_dis`: L1 Loss — CLIP distillation
- `loss_alignment`: AsymmetricLoss(γ⁻=4, γ⁺=0, clip=0.05) — image-text alignment

**Total loss:** `loss = loss_tag + loss_dis + loss_alignment`

---

### 2.2 RAM (Recognize Anything Model)

**File:** `ram/models/ram.py` → class `RAM`

**Đặc điểm chính:**
- Kiến trúc encoder-decoder: **Image-Tag Interaction Encoder** + **Image-Tag-Text Decoder**
- Label embedding đơn giản hơn RAM++: 1 embedding/tag → shape `(4585, 512)`
- Có thêm **Text Generation** (captioning) qua BertLMHeadModel decoder
- **Weight Sharing**: 2 layer thấp nhất của Tag Encoder được share với Tagging Head

**Các thành phần:**
| Thành phần | Mô tả |
|---|---|
| `visual_encoder` | Swin Transformer Large/Base hoặc ViT |
| `image_proj` | Linear(vision_width → 512) |
| `tag_encoder` | BertModel (12 layers, cross-attention) — image-tag interaction |
| `text_decoder` | BertLMHeadModel (12 layers) — text generation |
| `tagging_head` | BertModel (Q2L, 2 layers, cross-attention only) |
| `wordvec_proj` | Linear(512 → hidden_size) hoặc Identity |
| `fc` | Linear(hidden_size → 1) |
| `label_embed` | Frozen, shape (4585, 512) |

**3 Loss functions trong training:**
- `loss_tag`: AsymmetricLoss — tag recognition
- `loss_dis`: L1 Loss — CLIP distillation
- `loss_t2t`: Cross-entropy — text generation (captioning)

**Total loss:** `loss = loss_t2t + loss_tag / (loss_tag / loss_t2t).detach() + loss_dis`

> **Lưu ý:** `loss_tag` được normalize bằng tỉ lệ `loss_tag/loss_t2t` (detach gradient) để cân bằng magnitude giữa 2 loss.

---

### 2.3 Tag2Text

**File:** `ram/models/tag2text.py` → class `Tag2Text`

**Đặc điểm chính:**
- Mô hình tiền nhiệm, **không sử dụng CLIP distillation**
- Kiến trúc encoder-decoder tương tự RAM nhưng đơn giản hơn
- Label embedding dùng `nn.Embedding` (learnable, không frozen)
- FC layer dùng `GroupWiseLinear` thay vì Linear đơn giản
- Hỗ trợ **user-specified tags** cho controllable captioning

**Các thành phần:**
| Thành phần | Mô tả |
|---|---|
| `visual_encoder` | Swin Transformer Base hoặc ViT |
| `tag_encoder` | BertModel — image-tag interaction encoder |
| `text_decoder` | BertLMHeadModel — text generation decoder |
| `tagging_head` | BertModel (Q2L, 2 layers) |
| `label_embed` | nn.Embedding (learnable) |
| `fc` | GroupWiseLinear(num_class, hidden_size) |

**2 Loss functions:**
- `loss_tag`: AsymmetricLoss — tag recognition
- `loss_t2t`: Cross-entropy — text generation

**Total loss:** `loss = loss_t2t + loss_tag / (loss_tag / loss_t2t).detach()`

---

### 2.4 Backbone: Swin Transformer

**File:** `ram/models/swin_transformer.py`

Hỗ trợ 2 variants:
| Variant | embed_dim | depths | num_heads | vision_width |
|---|---|---|---|---|
| **Swin-B** | 128 | [2,2,18,2] | [4,8,16,32] | 1024 |
| **Swin-L** | 192 | [2,2,18,2] | [6,12,24,48] | 1536 |

- Pretrained weights từ ImageNet-22K
- Hỗ trợ image size 224 và 384
- Window size: 12 (cho 384), 7 (cho 224)

### 2.5 Backbone: ViT (Vision Transformer)

**File:** `ram/models/vit.py`

| Variant | embed_dim | depth | num_heads | patch_size |
|---|---|---|---|---|
| **ViT-Base** | 768 | 12 | 12 | 16 |
| **ViT-Large** | 1024 | 24 | 16 | 16 |

### 2.6 CLIP Model (Frozen)

- Sử dụng **OpenAI CLIP ViT-B/16**
- **Hoàn toàn frozen** trong quá trình training
- Dùng cho:
  - Encode image → CLIP image feature (cho distillation loss)
  - Encode text → text embedding (cho alignment loss trong RAM++)

### 2.7 Text Encoder/Decoder (BERT-based)

**File:** `ram/models/bert.py`

- Dựa trên **BertModel** và **BertLMHeadModel** từ HuggingFace
- Tokenizer: `bert-base-uncased` với special tokens: `[DEC]` (bos), `[ENC]` (encoder)
- Config MED (Mixture of Encoder-Decoder): 12 layers, hidden_size=768, 12 attention heads
- Config Q2L (Query2Label): 2 layers, hidden_size=768, 4 attention heads

---

## 3. Xử lý dữ liệu (Data Processing)

### 3.1 Định dạng dữ liệu đầu vào

Dữ liệu training được lưu dưới dạng **JSON**, mỗi file chứa list các annotation objects:

```json
[
  {
    "image_path": "path/to/image.jpg",
    "caption": ["caption 1", "caption 2", ...],
    "union_label_id": [id1, id2, ...],
    "parse_label_id": [[ids_for_caption1], [ids_for_caption2], ...]
  }
]
```

| Field | Mô tả |
|---|---|
| `image_path` | Đường dẫn tương đối tới ảnh |
| `caption` | List các caption mô tả ảnh |
| `union_label_id` | List tag IDs (union của tất cả captions) — dùng cho image-level tagging |
| `parse_label_id` | List of list tag IDs — mỗi sub-list tương ứng 1 caption — dùng cho caption-level parsing |

### 3.2 Dataset Classes

**File:** `ram/data/dataset.py`

#### `pretrain_dataset`
- Dùng cho **pretrain** (`pretrain.py`)
- Trả về: `(image, caption, image_tag, parse_tag)`
- `image`: Ảnh đã transform (config image_size, mặc định 224)
- `image_tag`: Binary vector (4585 dims), 1 tại vị trí union_label_id
- `parse_tag`: Binary vector (4585 dims), 1 tại vị trí parse_label_id của caption được chọn ngẫu nhiên
- Caption được chọn ngẫu nhiên từ list captions

#### `finetune_dataset`
- Dùng cho **finetune** (`finetune.py`)
- Trả về: `(image, image_224, caption, image_tag, parse_tag)`
- Thêm `image_224`: Ảnh resize về 224×224 — **dùng cho CLIP model** (vì CLIP ViT-B/16 nhận input 224)
- `image`: Ảnh ở config image_size (mặc định 384)

### 3.3 Data Augmentation

**File:** `ram/data/__init__.py` + `ram/data/randaugment.py`

#### Transform cho training:
```python
transforms.Compose([
    RandomResizedCrop(image_size, scale=(0.2, 1.0), interpolation=BICUBIC),
    RandomHorizontalFlip(),
    RandomAugment(N=2, M=5, augs=[
        'Identity', 'AutoContrast', 'Brightness', 'Sharpness',
        'Equalize', 'ShearX', 'ShearY', 'TranslateX', 'TranslateY', 'Rotate'
    ]),
    ToTensor(),
    Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
              std=[0.26862954, 0.26130258, 0.27577711])
])
```

#### Transform cho inference:
```python
transforms.Compose([
    convert_to_rgb,
    Resize((image_size, image_size)),
    ToTensor(),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
```

> **Lưu ý:** Normalization values khác nhau giữa training (CLIP stats) và inference (ImageNet stats).

#### RandomAugment
- Chọn ngẫu nhiên **N=2** augmentation operations
- Mỗi operation có probability **0.5** được áp dụng
- Magnitude level **M=5** (trên thang 10)

### 3.4 Caption Preprocessing

**File:** `ram/data/utils.py` → `pre_caption()`

```python
# Loại bỏ ký tự đặc biệt: . ! " ( ) * # : ; ~
# Chuyển lowercase
# Loại bỏ khoảng trắng thừa
# Truncate tối đa 30 từ
```

### 3.5 Tag List

- **RAM/RAM++**: 4585 tag categories (`ram/data/ram_tag_list.txt`)
- **Tag2Text**: Dùng tag list riêng (`ram/data/tag2text_ori_tag_list.txt`)
- Hỗ trợ tag tiếng Trung (`ram/data/ram_tag_list_chinese.txt`)
- Per-class threshold tuning (`ram/data/ram_tag_list_threshold.txt`)

### 3.6 Datasets cho Pretrain

Cấu hình trong `ram/configs/pretrain.yaml`:
| Dataset | File |
|---|---|
| COCO | `coco_train_rmcocodev_ram.json` |
| Visual Genome | `vg_ram.json` |
| SBU Captions | `sbu_ram.json` |
| CC3M (train) | `cc3m_train_ram.json` |
| CC3M (val) | `cc3m_val_ram.json` |
| CC12M | `cc12m_ram.json` |

### 3.7 Datasets cho Finetune

Cấu hình trong `ram/configs/finetune.yaml`:
| Dataset | File |
|---|---|
| COCO | `coco_train_rmcocodev_ram.json` |

---

## 4. Hướng dẫn Pretrain

### 4.1 Chuẩn bị

1. **Cài đặt dependencies:**
```bash
pip install -r requirements.txt
```

Dependencies chính:
- `timm==0.4.12`
- `transformers>=4.25.1,<=4.55.4`
- `fairscale==0.4.4`
- `torch`, `torchvision`
- `clip` (OpenAI CLIP)
- `Pillow`, `scipy`

2. **Chuẩn bị dữ liệu:** Tạo các file JSON annotation theo format ở mục 3.1

3. **Tải Swin Transformer pretrained weights:**
   - Swin-L: `swin_large_patch4_window12_384_22k.pth`
   - Swin-B: `swin_base_patch4_window7_224_22k.pth`
   - Đặt vào thư mục `pretrain_model/`

4. **Tải frozen tag embeddings:**
   - RAM++: `ram/data/frozen_tag_embedding/ram_plus_tag_embedding_class_4585_des_51.pth`
   - RAM: `ram/data/frozen_tag_embedding/ram_tag_embedding_class_4585.pth`

### 4.2 Chạy Pretrain

**RAM++:**
```bash
python -m torch.distributed.launch --nproc_per_node=8 pretrain.py \
    --model-type ram_plus \
    --config ram/configs/pretrain.yaml \
    --output-dir output/pretrain_ram_plus
```

**RAM:**
```bash
python -m torch.distributed.launch --nproc_per_node=8 pretrain.py \
    --model-type ram \
    --config ram/configs/pretrain.yaml \
    --output-dir output/pretrain_ram
```

**Tag2Text:**
```bash
python -m torch.distributed.launch --nproc_per_node=8 pretrain.py \
    --model-type tag2text \
    --config ram/configs/pretrain_tag2text.yaml \
    --output-dir output/pretrain_tag2text
```

### 4.3 Pretrain Config

| Parameter | RAM/RAM++ | Tag2Text |
|---|---|---|
| `vit` | swin_l | swin_b |
| `image_size` | 224 | 224 |
| `batch_size` | 52 | 80 |
| `init_lr` | 1e-4 | 1e-4 |
| `min_lr` | 5e-7 | 5e-7 |
| `warmup_lr` | 5e-7 | 5e-7 |
| `lr_decay_rate` | 0.9 | 0.9 |
| `max_epoch` | 5 | 5 |
| `warmup_steps` | 3000 | 3000 |
| `weight_decay` | 0.05 | 0.05 |
| `class_num` | 4585 | 4585 |

### 4.4 Learning Rate Schedule (Pretrain)

- **Epoch 0:** Warmup linear từ `warmup_lr` → `init_lr` trong `warmup_steps` iterations
- **Epoch 1+:** Step decay: `lr = max(min_lr, init_lr × decay_rate^epoch)`

### 4.5 Resume Training

```bash
python -m torch.distributed.launch --nproc_per_node=8 pretrain.py \
    --model-type ram_plus \
    --config ram/configs/pretrain.yaml \
    --checkpoint output/pretrain_ram_plus/checkpoint_02.pth \
    --output-dir output/pretrain_ram_plus
```

Checkpoint chứa: `model state_dict`, `optimizer state_dict`, `config`, `epoch`

---

## 5. Hướng dẫn Finetune

### 5.1 Khác biệt so với Pretrain

| Aspect | Pretrain | Finetune |
|---|---|---|
| Image size | 224 | **384** |
| CLIP input | Dùng chung image (224) | Thêm **image_224** riêng cho CLIP |
| LR schedule | Warmup + Step decay | **Cosine annealing** |
| Init LR | 1e-4 | **5e-6** |
| Max epoch | 5 | **2** |
| Checkpoint | Không bắt buộc | **Bắt buộc** (từ pretrain) |
| Model init | `stage='train_from_scratch'` | Load pretrained weights |

### 5.2 Chạy Finetune

**RAM++:**
```bash
python -m torch.distributed.launch --nproc_per_node=8 finetune.py \
    --model-type ram_plus \
    --config ram/configs/finetune.yaml \
    --checkpoint pretrained/ram_plus_swin_large_14m.pth \
    --output-dir output/finetune_ram_plus
```

**RAM:**
```bash
python -m torch.distributed.launch --nproc_per_node=8 finetune.py \
    --model-type ram \
    --config ram/configs/finetune.yaml \
    --checkpoint pretrained/ram_swin_large_14m.pth \
    --output-dir output/finetune_ram
```

**Tag2Text:**
```bash
python -m torch.distributed.launch --nproc_per_node=8 finetune.py \
    --model-type tag2text \
    --config ram/configs/finetune_tag2text.yaml \
    --checkpoint pretrained/tag2text_swin_14m.pth \
    --output-dir output/finetune_tag2text
```

### 5.3 Finetune Config

| Parameter | RAM/RAM++ | Tag2Text |
|---|---|---|
| `vit` | swin_l | swin_b |
| `image_size` | 384 | 384 |
| `batch_size` | 26 | 36 |
| `init_lr` | 5e-6 | 5e-6 |
| `min_lr` | 0 | 0 |
| `max_epoch` | 2 | 2 |
| `weight_decay` | 0.05 | 0.05 |

### 5.4 Learning Rate Schedule (Finetune)

**Cosine Annealing:**
```
lr = (init_lr - min_lr) × 0.5 × (1 + cos(π × epoch / max_epoch)) + min_lr
```

### 5.5 Frozen Parameters

Trong cả pretrain và finetune:
- **CLIP model**: Hoàn toàn frozen (`requires_grad = False`)
- **label_embed**: Frozen (`requires_grad = False`) — giữ nguyên semantic embedding từ CLIP/LLM
- **Optimizer**: AdamW chỉ update các parameters có `requires_grad = True`

---

## 6. Hướng dẫn Inference (Chạy dự đoán)

### 6.1 RAM++ Inference

```bash
python inference_ram_plus.py \
    --image images/demo/demo1.jpg \
    --pretrained pretrained/ram_plus_swin_large_14m.pth \
    --image-size 384
```

**Output:** Image Tags (English) + 图像标签 (Chinese)

### 6.2 RAM Inference

```bash
python inference_ram.py \
    --image images/demo/demo1.jpg \
    --pretrained pretrained/ram_swin_large_14m.pth \
    --image-size 384
```

### 6.3 Tag2Text Inference

```bash
python inference_tag2text.py \
    --image images/1641173_2291260800.jpg \
    --pretrained pretrained/tag2text_swin_14m.pth \
    --image-size 384 \
    --thre 0.68 \
    --specified-tags "None"
```

**Output:** Model Identified Tags + User Specified Tags + Image Caption

### 6.4 Open-Set Inference (RAM)

Nhận diện các category **chưa từng thấy** trong training:

```bash
python inference_ram_openset.py \
    --image images/openset_example.jpg \
    --pretrained pretrained/ram_swin_large_14m.pth \
    --image-size 384
```

**Cơ chế:** Thay thế `label_embed` bằng CLIP text embeddings của unseen categories (dùng multiple prompt templates).

### 6.5 Open-Set Inference (RAM++)

Sử dụng **LLM-generated descriptions** cho open-set:

```bash
python inference_ram_plus_openset.py \
    --image images/openset_example.jpg \
    --pretrained pretrained/ram_plus_swin_large_14m.pth \
    --image-size 384 \
    --llm_tag_des datasets/openimages_rare_200/openimages_rare_200_llm_tag_descriptions.json
```

### 6.6 Inference Pipeline (Code)

```python
from ram.models import ram_plus
from ram import inference_ram, get_transform
from PIL import Image
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 1. Transform
transform = get_transform(image_size=384)

# 2. Load model
model = ram_plus(pretrained='pretrained/ram_plus_swin_large_14m.pth',
                 image_size=384, vit='swin_l')
model.eval().to(device)

# 3. Inference
image = transform(Image.open('test.jpg')).unsqueeze(0).to(device)
tags_en, tags_cn = inference_ram(image, model)
print("Tags:", tags_en)
```

---

## 7. Batch Inference & Evaluation

**File:** `batch_inference.py`

### 7.1 Chạy Batch Inference

```bash
python batch_inference.py \
    --model-type ram_plus \
    --checkpoint pretrained/ram_plus_swin_large_14m.pth \
    --dataset openimages_common_214 \
    --input-size 384 \
    --batch-size 128 \
    --num-workers 4 \
    --output-dir outputs/ram_plus_common214
```

### 7.2 Datasets đánh giá

| Dataset | Mô tả |
|---|---|
| `openimages_common_214` | 214 common categories từ OpenImages |
| `openimages_rare_200` | 200 rare/unseen categories từ OpenImages |

### 7.3 Open-Set Evaluation

```bash
python batch_inference.py \
    --model-type ram_plus \
    --checkpoint pretrained/ram_plus_swin_large_14m.pth \
    --dataset openimages_rare_200 \
    --open-set \
    --input-size 384 \
    --output-dir outputs/ram_plus_openset
```

### 7.4 Output Files

| File | Nội dung |
|---|---|
| `pred.txt` | Tag predictions cho mỗi ảnh |
| `ap.txt` | Average Precision per tag |
| `pr.txt` | Precision & Recall per tag |
| `summary.txt` | mAP, CP (mean Precision), CR (mean Recall) |
| `logits.pth` | Raw logits (cached, dùng cho threshold tuning) |

### 7.5 Metrics

- **mAP** (mean Average Precision): Tính AP cho mỗi class, rồi lấy trung bình
- **CP** (Category Precision): Trung bình Precision trên tất cả categories
- **CR** (Category Recall): Trung bình Recall trên tất cả categories

---

## 8. Cấu trúc thư mục dự án

```
recognize-anything/
├── finetune.py                    # Script finetune chính
├── pretrain.py                    # Script pretrain chính
├── utils.py                       # Training utilities (LR schedulers, metrics, distributed)
├── inference_ram.py               # Inference RAM đơn ảnh
├── inference_ram_plus.py          # Inference RAM++ đơn ảnh
├── inference_ram_openset.py       # Inference RAM open-set
├── inference_ram_plus_openset.py  # Inference RAM++ open-set
├── inference_tag2text.py          # Inference Tag2Text
├── batch_inference.py             # Batch inference + evaluation
├── generate_tag_des_llm.py        # Generate tag descriptions bằng GPT-3.5
├── requirements.txt               # Python dependencies
│
├── ram/                           # Core package
│   ├── __init__.py                # Exports: inference functions, get_transform
│   ├── inference.py               # Inference wrappers (inference_ram, inference_tag2text, ...)
│   ├── transform.py               # Inference image transform
│   │
│   ├── models/                    # Model definitions
│   │   ├── ram_plus.py            # RAM++ model
│   │   ├── ram.py                 # RAM model
│   │   ├── tag2text.py            # Tag2Text model
│   │   ├── bert.py                # Modified BERT (encoder-decoder)
│   │   ├── swin_transformer.py    # Swin Transformer backbone
│   │   ├── vit.py                 # Vision Transformer backbone
│   │   └── utils.py               # Model utilities (load checkpoint, AsymmetricLoss, ...)
│   │
│   ├── data/                      # Data processing
│   │   ├── __init__.py            # create_dataset, create_sampler, create_loader
│   │   ├── dataset.py             # pretrain_dataset, finetune_dataset
│   │   ├── utils.py               # pre_caption, save_result, coco_caption_eval
│   │   ├── randaugment.py         # RandomAugment implementation
│   │   ├── ram_tag_list.txt       # 4585 tag categories (English)
│   │   ├── ram_tag_list_chinese.txt # Tag categories (Chinese)
│   │   ├── ram_tag_list_threshold.txt # Per-class thresholds
│   │   └── tag2text_ori_tag_list.txt  # Tag2Text tag list
│   │
│   ├── configs/                   # Configuration files
│   │   ├── pretrain.yaml          # Pretrain config (RAM/RAM++)
│   │   ├── pretrain_tag2text.yaml # Pretrain config (Tag2Text)
│   │   ├── finetune.yaml          # Finetune config (RAM/RAM++)
│   │   ├── finetune_tag2text.yaml # Finetune config (Tag2Text)
│   │   ├── med_config.json        # BERT encoder-decoder config (12 layers)
│   │   ├── q2l_config.json        # Query2Label decoder config (2 layers)
│   │   └── swin/                  # Swin Transformer configs
│   │       ├── config_swinB_224.json
│   │       ├── config_swinB_384.json
│   │       ├── config_swinL_224.json
│   │       └── config_swinL_384.json
│   │
│   └── utils/                     # Evaluation utilities
│       ├── metrics.py             # mAP, Precision, Recall
│       └── openset_utils.py       # Open-set label embedding builders
│
├── datasets/                      # Evaluation datasets
│   ├── openimages_common_214/
│   ├── openimages_rare_200/
│   ├── hico/
│   └── imagenet_multi/
│
├── pretrained/                    # Pretrained model weights
└── images/                        # Demo images
```

---

## 9. Cấu hình & Hyperparameters

### 9.1 BERT Encoder-Decoder (MED Config)

```json
{
    "hidden_size": 768,
    "num_hidden_layers": 12,
    "num_attention_heads": 12,
    "intermediate_size": 3072,
    "hidden_act": "gelu",
    "hidden_dropout_prob": 0.1,
    "attention_probs_dropout_prob": 0.1,
    "max_position_embeddings": 512,
    "encoder_width": 768,
    "add_cross_attention": true
}
```

### 9.2 Query2Label Decoder (Q2L Config)

```json
{
    "hidden_size": 768,
    "num_hidden_layers": 2,
    "num_attention_heads": 4,
    "intermediate_size": 3072,
    "encoder_width": 768,
    "add_cross_attention": true,
    "add_tag_cross_attention": false
}
```

> **Tối ưu:** Q2L decoder chỉ có 2 layers và **xóa self-attention** (chỉ giữ cross-attention), giảm đáng kể computation.

### 9.3 Arguments dòng lệnh

| Argument | Mô tả | Default |
|---|---|---|
| `--config` | Path tới YAML config | `./configs/pretrain.yaml` |
| `--model-type` | Loại model: `ram_plus`, `ram`, `tag2text` | (required) |
| `--output-dir` | Thư mục lưu checkpoint | `output/Pretrain` |
| `--checkpoint` | Path tới pretrained checkpoint | `''` |
| `--device` | Device | `cuda` |
| `--seed` | Random seed | `42` |
| `--world_size` | Số distributed processes | `1` |
| `--dist_url` | URL cho distributed training | `env://` |
| `--distributed` | Bật distributed training | `True` |

---

## 10. Loss Functions

### 10.1 Asymmetric Loss (ASL)

**File:** `ram/models/utils.py` → class `AsymmetricLoss`

Được thiết kế cho **multi-label classification** với class imbalance lớn (nhiều negative hơn positive).

```
L = -Σ [ y × (1-p)^γ⁺ × log(p) + (1-y) × p̂^γ⁻ × log(1-p̂) ]

Trong đó:
  p̂ = max(1-p-m, 0)    # Asymmetric clipping (m = 0.05)
  γ⁺ = 0                # Không down-weight positive samples
  γ⁻ = 7                # Mạnh mẽ down-weight easy negatives
```

**Tham số cho tag recognition:**
- `gamma_neg=7`, `gamma_pos=0`, `clip=0.05`

**Tham số cho text alignment (RAM++):**
- `gamma_neg=4`, `gamma_pos=0`, `clip=0.05`

### 10.2 L1 Loss (CLIP Distillation)

```python
loss_dis = F.l1_loss(image_cls_embeds, clip_feature)
```

Align CLS token embedding của visual encoder với CLIP image feature.

### 10.3 Cross-Entropy Loss (Text Generation)

Dùng cho text decoder trong RAM và Tag2Text. Prompt tokens (`"a picture of "`) được mask (`-100`) để không tính loss.

---

## 11. Open-Set Recognition

### 11.1 Cơ chế hoạt động

Open-set recognition cho phép nhận diện các category **không có trong 4585 tag training set** bằng cách thay thế `label_embed`:

1. **RAM Open-Set**: Dùng CLIP text encoder + multiple prompt templates (66 templates) để tạo label embedding cho unseen categories
2. **RAM++ Open-Set**: Dùng **LLM (GPT-3.5)** để generate descriptions cho mỗi category, rồi encode bằng CLIP

### 11.2 Generate Tag Descriptions bằng LLM

**File:** `generate_tag_des_llm.py`

```bash
python generate_tag_des_llm.py \
    --openai_api_key sk-xxxxx \
    --output_file_path datasets/openimages_rare_200/openimages_rare_200_llm_tag_descriptions.json
```

Mỗi tag được generate 51 descriptions:
- 1 default: `"a photo of a {tag}."`
- 50 từ LLM (5 prompts × 10 responses mỗi prompt)

**5 LLM Prompts:**
1. `"Describe concisely what a(n) {tag} looks like:"`
2. `"How can you identify a(n) {tag} concisely?"`
3. `"What does a(n) {tag} look like concisely?"`
4. `"What are the identifying characteristics of a(n) {tag}:"`
5. `"Please provide a concise description of the visual characteristics of {tag}:"`

### 11.3 Build Open-Set Label Embedding

**RAM (CLIP templates):**
```python
from ram.utils import build_openset_label_embedding
label_embed, categories = build_openset_label_embedding(custom_categories)
model.label_embed = nn.Parameter(label_embed.float())
model.num_class = len(categories)
model.class_threshold = torch.ones(model.num_class) * 0.5  # lower threshold
```

**RAM++ (LLM descriptions):**
```python
from ram.utils import build_openset_llm_label_embedding
label_embed, categories = build_openset_llm_label_embedding(llm_tag_des)
model.label_embed = nn.Parameter(label_embed.float())
model.num_class = len(categories)
model.class_threshold = torch.ones(model.num_class) * 0.5
```

> **Lưu ý:** Threshold cho unseen categories thường thấp hơn (0.5) so với seen categories (0.68).

---

## Tổng kết so sánh 3 mô hình

| Feature | Tag2Text | RAM | RAM++ |
|---|---|---|---|
| **Backbone mặc định** | Swin-B | Swin-L | Swin-L |
| **CLIP Distillation** | ❌ | ✅ | ✅ |
| **Text Generation** | ✅ | ✅ | ❌ |
| **Label Embedding** | Learnable (nn.Embedding) | Frozen CLIP (1 embed/tag) | Frozen CLIP+LLM (51 embeds/tag) |
| **Description Reweighting** | ❌ | ❌ | ✅ |
| **Image-Text Alignment** | ❌ | ❌ | ✅ |
| **Open-Set Support** | ❌ | ✅ (CLIP templates) | ✅ (LLM descriptions) |
| **FC Layer** | GroupWiseLinear | Linear(hidden→1) | Linear(hidden→1) |
| **Weight Sharing** | Tag Encoder ↔ Tagging Head | Tag Encoder ↔ Tagging Head | ❌ |
| **Losses** | tag + t2t | tag + t2t + dis | tag + dis + alignment |
| **Pretrained weights** | `tag2text_swin_14m.pth` | `ram_swin_large_14m.pth` | `ram_plus_swin_large_14m.pth` |
