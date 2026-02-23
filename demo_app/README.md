# Recognize-Anything Web Demo

Web demo đơn giản cho RAM++ / RAM / Tag2Text sử dụng **Gradio**.

## Tech Stack

| Thành phần | Công nghệ | Lý do |
|------------|-----------|-------|
| **UI Framework** | Gradio 4.x | Zero frontend code, auto UI đẹp, hỗ trợ share link |
| **Backend** | Python + PyTorch | Tận dụng trực tiếp code inference có sẵn |
| **Image Processing** | Pillow | Load/convert ảnh |
| **Model** | RAM++ / RAM / Tag2Text | Từ project chính |

## Cài đặt

```bash
cd demo_app
pip install -r requirements.txt
```

## Chạy

### RAM++ (mặc định)
```bash
python app.py
```

### RAM
```bash
python app.py --model-type ram
```

### Tag2Text (có captioning)
```bash
python app.py --model-type tag2text
```

### Với checkpoint tùy chọn (ví dụ sau finetune)
```bash
python app.py --checkpoint ../output/my_finetune/checkpoint_02.pth
```

### Tạo link public (share qua internet)
```bash
python app.py --share
```

## Tham số

| Tham số | Mặc định | Mô tả |
|---------|----------|-------|
| `--model-type` | `ram_plus` | `ram_plus`, `ram`, hoặc `tag2text` |
| `--checkpoint` | auto-detect | Đường dẫn tới file `.pth` |
| `--image-size` | `384` | Kích thước ảnh đầu vào |
| `--port` | `7860` | Port server |
| `--share` | `False` | Tạo public link |

## Tính năng

- Upload ảnh → nhận tags (EN + CN)
- Tag2Text: thêm captioning + specified tags
- RAM/RAM++: xem confidence scores dạng bar chart
- Example images từ thư mục `images/`
- Auto-detect checkpoint trong `pretrained/`
