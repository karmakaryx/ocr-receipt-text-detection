![banner_ocr](./assets/banner_ocr.jpg)

## **💻 Project Overview**
### Environment
- **OS:** Linux Ubuntu 20.04.6 LTS
- **GPU:** NVIDIA GeForce RTX 3090 (24GB)
- **NVIDIA Driver Version:** 535.86.10
- **CUDA Version:** 12.2 (Runtime: 11.8)
- **Tool:** VS Code (SSH) / Google Colab
- **Language:** Python 3.10.13

### Requirements

## **📋 Competiton Info**
### 일정 (Timeline)
- 2026.05.04 09:00 ~ 2026.05.14 18:00 (Competition)
- 2026.05.15 15:00 ~ 2026.05.15 17:00 (Seminar)

## **⚙️ Components**
### Directory
```
├── assets/...
├── code/
│   ├── configs/
│   │   ├── preset/
│   │   │   ├── datasets/
│   │   │   │   └── db.yaml
│   │   │   ├── lightning_modules/
│   │   │   │   └── base.yaml
│   │   │   ├── models/
│   │   │   │   ├── decoder/
│   │   │   │   │   └── unet.yaml
│   │   │   │   ├── encoder/
│   │   │   │   │   └── timm_backbone.yaml
│   │   │   │   ├── head/
│   │   │   │   │   └── db_head.yaml
│   │   │   │   ├── loss/
│   │   │   │   │   └── db_loss.yaml
│   │   │   │   └── model_example.yaml
│   │   │   ├── base.yaml
│   │   │   └── example.yaml
│   │   ├── predict.yaml
│   │   ├── test.yaml
│   │   └── train.yaml
│   ├── ocr/
│   │   ├── datasets/
│   │   │   ├── base.py
│   │   │   ├── db_collate_fn.py
│   │   │   └── transforms.py
│   │   ├── lightning_modules/
│   │   │   ├── callbacks/
│   │   │   └── ocr_pl.py
│   │   ├── metrics/
│   │   │   ├── box_types.py
│   │   │   ├── cleval_metric.py
│   │   │   ├── data.py
│   │   │   ├── eval_functions.py
│   │   │   └── utils.py
│   │   ├── models/
│   │   │   ├── decoder/
│   │   │   │   └── unet.py
│   │   │   ├── encoder/
│   │   │   │   └── timm_backbone.py
│   │   │   ├── head/
│   │   │   │   ├── db_head.py
│   │   │   │   └── db_postprocess.py
│   │   │   ├── loss/
│   │   │   │   ├── bce_loss.py
│   │   │   │   ├── db_loss.py
│   │   │   │   ├── dice_loss.py
│   │   │   │   └── l1_loss.py
│   │   │   └── architecture.py
│   │   └── utils/
│   │       ├── convert_submission.py
│   │       └── ocr_utils.py
│   ├── outputs/
│   │   ├── ocr_training/
│   │   │   ├── .hydra/...
│   │   │   ├── checkpoints/...
│   │   │   ├── logs/...
│   │   │   └── submissions/...
│   │   └── submission.csv
│   ├── runners/
│   │   ├── predict.py
│   │   ├── test.py
│   │   └── train.py
│   ├── wandb/...
│   ├── baseline.ipynb
│   └── eda.ipynb
├── data/
│   └── datasets/
│       ├── images/
│       │   ├── test/...
│       │   ├── train/...
│       │   └── val/...
│       ├── jsons/
│       │   ├── test.json
│       │   ├── train.json
│       │   └── val.json
│       └── sample_submission.csv
├── .gitignore
├── README.md
└── requirements.txt
```
