Môn học: Khoa học máy tính nâng cao \\
Nhóm: KHMT15
Thành viên:
- Trần Quốc Thịnh - 25C11065
- Nguyễn Trường Thịnh - 25C11066
# RAM — Recognize Anything Model

Code original: [xinyu1205/recognize-anything](https://github.com/xinyu1205/recognize-anything)

**Model:** RAM (Recognize Anything Model) &nbsp;|&nbsp; **Backbone:** Swin Transformer Large &nbsp;|&nbsp; **Tags:** 4585 categories

---

## Muc luc

- [Cai dat](#cai-dat)
- [Chay demo](#chay-demo)
- [Chuan bi du lieu](#chuan-bi-du-lieu)
- [Fine-tune](#fine-tune)
- [Evaluation](#evaluation)
- [Kaggle Notebooks](#kaggle-notebooks)
- [Tuy chinh tag list](#tuy-chinh-tag-list)
- [Cau truc thu muc](#cau-truc-thu-muc)

---

## Cai dat

```bash
# Clone repo
git clone https://github.com/ThinhSama234/image_tagging.git
cd image_tagging

# Tao moi truong
conda create -n ram python=3.8 -y
conda activate ram

# Cai dat dependencies
pip install -r requirements.txt
pip install -e .
```

### Download checkpoint

Tao folder `pretrained/` va tai checkpoint vao:

| Model | Backbone | Link |
|-------|----------|------|
| RAM | Swin-L | [ram_swin_large_14m.pth](https://huggingface.co/spaces/xinyu1205/Recognize_Anything-Tag2Text/blob/main/ram_swin_large_14m.pth) |

---

## Chay demo

### Web UI (Gradio)

```bash
cd demo_app

# Chay demo RAM
python app.py --model-type ram --checkpoint ../pretrained/ram_swin_large_14m.pth

# Tao public link (Kaggle/Colab)
python app.py --model-type ram --checkpoint ../pretrained/ram_swin_large_14m.pth --share
```

Mo trinh duyet tai `http://localhost:7860`. Upload anh → nhan tags tieng Anh/Trung + confidence scores.

### CLI inference

```bash
python inference_ram.py \
    --image images/demo/demo1.jpg \
    --pretrained pretrained/ram_swin_large_14m.pth
```

Output:
```
Image Tags: armchair | blanket | lamp | carpet | couch | dog | floor | furniture | ...
```

---

## Chuan bi du lieu

Dung module `batch_image/` de chuyen du lieu tho thanh JSON annotation cho fine-tune.

### Ho tro 4 loai adapter

| Adapter | Input | Lenh |
|---------|-------|------|
| `csv` | File CSV (image_path, tags) | `--adapter csv --csv-path data/labels.csv` |
| `folder` | Thu muc anh + optional labels.txt | `--adapter folder --image-dir data/images/` |
| `flickr30k` | Flickr30k dataset | `--adapter flickr30k --csv-path tags.csv --captions-csv captions.csv --image-dir images/` |
| `voc` | Pascal VOC | `--adapter voc --voc-root data/ --voc-year 2012` |

### Vi du: Tu CSV

```bash
python -m batch_image.preprocess \
    --adapter csv \
    --csv-path data/labels.csv \
    --tag-sep "|" \
    --output-dir datasets/custom \
    --train-ratio 0.9
```

### Vi du: Tu folder

**Layout A** — folder anh phang + file labels:
```
data/images/
  img001.jpg
  img002.jpg
data/labels.txt     # img001.jpg|dog|cat
```

```bash
python -m batch_image.preprocess \
    --adapter folder \
    --image-dir data/images/ \
    --labels-file data/labels.txt \
    --output-dir datasets/custom
```

**Layout B** — subfolder = ten tag:
```
data/images/
  dog/
    img001.jpg
  cat/
    img002.jpg
```

```bash
python -m batch_image.preprocess \
    --adapter folder \
    --image-dir data/images/ \
    --output-dir datasets/custom
```

### Output

```
datasets/custom/
  train.json            # Annotation cho finetune
  val.json              # Validation set
  finetune_config.yaml  # Config san sang de train
```

Moi entry trong JSON:
```json
{
    "image_path": "data/images/img001.jpg",
    "caption": "a photo of dog and cat",
    "image_tag": [234, 567],
    "parse_tag": [234, 567]
}
```

---

## Fine-tune

### Single GPU

```bash
python finetune.py \
    --model-type ram \
    --config datasets/custom/finetune_config.yaml \
    --checkpoint pretrained/ram_swin_large_14m.pth \
    --output-dir output/custom_finetune \
    --distributed False
```

### Multi-GPU (DDP)

```bash
python -m torch.distributed.run --nproc_per_node=2 finetune.py \
    --model-type ram \
    --config datasets/custom/finetune_config.yaml \
    --checkpoint pretrained/ram_swin_large_14m.pth \
    --output-dir output/custom_finetune
```

### Config YAML

```yaml
train_file:
  - 'datasets/custom/train.json'

image_path_root: ""           # prefix cho image_path trong JSON
vit: 'swin_l'
vit_grad_ckpt: True           # gradient checkpointing (tiet kiem VRAM)
vit_ckpt_layer: 0

image_size: 384               # hoac 224 de tiet kiem VRAM
batch_size: 2
gradient_accumulation_steps: 2  # effective batch = 2*2 = 4
use_amp: true                 # mixed precision training (xem giai thich ben duoi)

weight_decay: 0.05
init_lr: 5e-06
min_lr: 0
max_epoch: 5
warmup_steps: 500
class_num: 4585
```

### Mixed Precision Training (`use_amp: true`)

Khi bat `use_amp`, forward pass chay o **FP16** (16-bit) thay vi FP32 (32-bit) mac dinh:

- **FP32** (float32): 1 so = 32 bit, pham vi ~$10^{-38}$ den ~$10^{38}$
- **FP16** (float16): 1 so = 16 bit, pham vi ~$10^{-8}$ den ~$65504$

Loi ich: moi gia tri chi ton **2 bytes thay vi 4 bytes** → giam ~40% VRAM, tang toc tinh toan tren GPU (Tensor Cores).

**Van de:** gradient nho (vd `0.00001`) co the bi **underflow thanh 0** o FP16 vi pham vi qua hep. Giai phap: `GradScaler`.

```
Forward (FP16) → Loss → GradScaler nhan loss len (vd x1024)
    → Backward (gradient lon, khong bi underflow)
    → GradScaler chia gradient lai (vd /1024)
    → Kiem tra inf/NaN: co → bo qua update, giam scale
                          khong → optimizer.step() binh thuong
    → GradScaler tu dieu chinh scale cho batch tiep theo
```

Code tuong ung trong `finetune.py`:
```python
scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

with torch.amp.autocast('cuda', enabled=use_amp):  # FP16 forward
    loss = model(image, caption, image_tag, ...)

scaler.scale(loss).backward()   # scale loss truoc backward
scaler.step(optimizer)          # unscale + kiem tra + update
scaler.update()                 # dieu chinh scale factor
```

> **Luu y:** Chi forward pass chay FP16 (thong qua `autocast`). Trong so model (`master weights`) van luu o FP32 de dam bao do chinh xac khi cap nhat.

### Freeze ViT Backbone (`--freeze-vit`)

Finetune co 2 option:

| Option | Lenh | Train gi | VRAM | Khi nao dung |
|--------|------|----------|------|--------------|
| **Full finetune** | (mac dinh) | Toan bo model (Swin + BERT + tagging head) | ~12-16 GB | Du VRAM, muon ket qua tot nhat |
| **Freeze ViT** | `--freeze-vit` | Chi BERT decoder + tagging head (dong bang Swin) | ~4-6 GB | Kaggle free GPU (T4 16GB), it data |

Khi `--freeze-vit`, Swin Transformer giu nguyen trong so pretrained, chi cap nhat phan decoder:

```bash
python finetune.py \
    --model-type ram \
    --config datasets/custom/finetune_config.yaml \
    --checkpoint pretrained/ram_swin_large_14m.pth \
    --output-dir output/custom_finetune \
    --freeze-vit \
    --distributed False
```

> **Trade-off:** Freeze ViT hoc nhanh hon, it overfit voi dataset nho, nhung khong the adapt feature extraction cho domain moi (vd anh y te, anh ve tinh).

### Chay tren Kaggle

Kaggle free GPU chi co **~20 GB disk** va **16 GB VRAM (T4)**. Script `finetune.py` da duoc adapt:

**Tiet kiem disk:**
- Chi luu 2 checkpoint: `checkpoint_latest.pth` (ghi de moi epoch) + `checkpoint_best.pth` (loss thap nhat)
- Khong luu optimizer state → moi checkpoint chi ~1.5 GB thay vi ~3 GB
- Tu dong xoa checkpoint cu khi disk con < 4 GB

**Tiet kiem VRAM:**
- `use_amp: true` — giam ~40% VRAM
- `vit_grad_ckpt: True` — gradient checkpointing cho Swin (doi VRAM lay thoi gian tinh)
- `image_size: 224` thay vi 384 — giam ~65% VRAM cho image encoder
- `--freeze-vit` — phuong an cuoi khi van OOM

Config khuyen dung cho Kaggle T4:
```yaml
image_size: 224
batch_size: 2
gradient_accumulation_steps: 2
use_amp: true
vit_grad_ckpt: True
```

### Tips

- **Resume sau khi Kaggle restart:** `--resume output/custom_finetune/checkpoint_latest.pth`
- Output: `checkpoint_latest.pth` (moi epoch, ghi de) + `checkpoint_best.pth` (loss thap nhat)

---

## Evaluation

### Tren custom dataset (JSON format)

```bash
# Danh gia 1 model
python evaluate.py \
    --checkpoint output/custom_finetune/checkpoint_best.pth \
    --val-json datasets/custom/val.json \
    --image-root /path/to/images \
    --model-type ram

# So sanh pretrained vs finetuned
python evaluate.py \
    --checkpoint pretrained/ram_swin_large_14m.pth \
    --checkpoint-b output/custom_finetune/checkpoint_best.pth \
    --val-json datasets/custom/val.json \
    --image-root /path/to/images \
    --model-type ram
```

### Tren Pascal VOC 2012

```bash
# Tu dong download VOC va danh gia
python evaluate_voc.py \
    --checkpoint pretrained/ram_swin_large_14m.pth \
    --model-type ram

# So sanh pretrained vs finetuned
python evaluate_voc.py \
    --checkpoint pretrained/ram_swin_large_14m.pth \
    --checkpoint-b output/custom_finetune/checkpoint_best.pth \
    --model-type ram \
    --output results.json
```

Output:
```
======================================================================
  Model A (ram_swin_large_14m.pth) — VOC 2012 Val Results
======================================================================
  Class              Prec  Recall      F1    TP    FP    FN    GT
  ----------------------------------------------------------------
  aeroplane          0.912  0.879   0.895   ...
  bicycle            0.834  0.791   0.812   ...
  ...
  MEAN               0.857  0.823   0.839
======================================================================
```

### Ket qua thuc nghiem

#### Flickr30k

| Model | Precision | Recall | F1 | Ghi chu |
|-------|-----------|--------|----|---------|
| RAM pretrained | — | — | — | |
| RAM finetuned (full) | — | — | — | |
| RAM finetuned (freeze-vit) | — | — | — | |

#### Pascal VOC 2012

| Model | mPrecision | mRecall | mF1 | Ghi chu |
|-------|------------|---------|-----|---------|
| RAM pretrained | — | — | — | |
| RAM finetuned | — | — | — | |

> **TODO:** Dien ket qua sau khi chay experiment.

---

## Kaggle Notebooks

3 notebook chinh cua nhom, chay tren Kaggle GPU T4:

### 1. `finetune_flickr30k.ipynb` — Fine-tune RAM tren Flickr30k
### 2. `finetune_tam_pascal_voc.ipynb` — Fine-tune + Evaluate tren Pascal VOC 2012

### 3. `ram-evaluation.ipynb` — Evaluate RAM tren nhieu dataset

---

## Tuy chinh tag list

Model RAM co 4585 tag, luu trong 2 file song song:

```
ram/data/ram_tag_list.txt            ← ten tag (moi dong 1 tag)
ram/data/ram_tag_list_threshold.txt  ← nguong tuong ung (moi dong 1 so)
```

Vi du dong thu 1 trong ca 2 file:
```
ram_tag_list.txt:           3D CG rendering
ram_tag_list_threshold.txt: 0.65
```
→ Nghia la: tag "3D CG rendering" chi duoc nhan dien khi confidence >= 0.65.

### Dieu chinh threshold

Khi model du doan, moi tag co 1 confidence score tu 0 den 1. Tag chi duoc output neu score >= threshold cua no.

- **Tang threshold** (vd 0.65 → 0.85): it tag hon, nhung chinh xac hon (giam false positive)
- **Giam threshold** (vd 0.65 → 0.4): nhieu tag hon, nhung co the sai nhieu hon (giam false negative)

Chi can sua so trong `ram_tag_list_threshold.txt` tai dong tuong ung, **khong can fine-tune lai**.

### Them/xoa tag

Neu muon them hoac xoa tag khoi danh sach 4585 tag:

1. Sua `ram/data/ram_tag_list.txt` — them/xoa dong
2. Sua `ram/data/ram_tag_list_threshold.txt` — them/xoa dong tuong ung (giu cung so dong)
3. Sua `class_num` trong config YAML cho khop so tag moi
4. **Bat buoc fine-tune lai** model vi kich thuoc output layer thay doi

---

## Cau truc thu muc

```
.
├── batch_image/              # Data preprocessing adapters
│   ├── preprocess.py         # Main pipeline: raw data → JSON annotation
│   ├── base_adapter.py       # BaseAdapter ABC + RawEntry dataclass
│   ├── csv_adapter.py        # CSV input adapter
│   ├── folder_adapter.py     # Folder structure adapter
│   ├── flickr30k_adapter.py  # Flickr30k adapter
│   └── voc_adapter.py        # Pascal VOC adapter
├── demo_app/
│   └── app.py                # Gradio web demo
├── ram/
│   ├── models/
│   │   ├── ram.py            # RAM model (SwinTransformer + BERT)
│   │   └── swin_transformer.py
│   ├── data/
│   │   ├── ram_tag_list.txt  # 4585 tags
│   │   └── ram_tag_list_threshold.txt
│   └── configs/
│       ├── finetune.yaml
│       └── pretrain.yaml
├── finetune.py               # Fine-tuning script (single/multi-GPU)
├── evaluate.py               # Evaluate on custom JSON dataset
├── evaluate_voc.py           # Evaluate on Pascal VOC 2012
├── batch_inference.py        # Batch inference + eval on OpenImages
└── inference_ram.py           # CLI inference RAM
```

---

## Citation

```bibtex
@article{zhang2023recognize,
  title={Recognize Anything: A Strong Image Tagging Model},
  author={Zhang, Youcai and Huang, Xinyu and Ma, Jinyu and others},
  journal={arXiv preprint arXiv:2306.03514},
  year={2023}
}
```

## Acknowledgements

Based on [BLIP](https://github.com/salesforce/BLIP). Original repo: [xinyu1205/recognize-anything](https://github.com/xinyu1205/recognize-anything).
