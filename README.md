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
```
albumentations==1.3.1                             Polygon3==3.0.9.1
autopep8==2.0.4                                   pyclipper==1.3.0.post5
better-exceptions==0.3.3                          PyYAML==6.0.1
easydict==1.11                                    safetensors==0.4.1
editdistance==0.6.2                               setuptools==69.0.3
flake8==6.1.0                                     scikit-image==0.22.0
huggingface-hub==0.19.4                           scikit-learn==1.3.2
hydra-core==1.3.2                                 scipy==1.11.4
imageio==2.33.0                                   seaborn==0.13.0
lightning==2.1.3                                  shapely==2.0.2
pytorch-lightning==2.1.3                          tensorboard==2.15.1
matplotlib==3.8.2                                 tensorboard-data-server==0.7.2
numpy==1.26.2                                     timm==0.9.12
numba==0.58.1                                     torchmetrics==1.2.1
opencv-python==4.8.1.78                           tqdm==4.66.1
pandas==2.1.4                                     wandb==0.16.1
pathlib==1.0.1                                    torch==2.1.2+cu118
Pillow==10.1.0                                    torchvision==0.16.2+cu118
```

---

## **📋 Competition Info**
### 일정 (Timeline)
- 2026.05.04 09:00 ~ 2026.05.14 18:00 (Competition)
- 2026.05.15 15:00 ~ 2026.05.15 17:00 (Seminar)

### 영수증 글자 검출 대회: 영수증 사진에서 글자 위치를 정확하게 추출하는 태스크 수행
- 목표: 모델이 더욱 강건한 성능을 낼 수 있도록 generalization과 optimization을 모두 높이면서도, 그 사이의 최적점 찾기
- 각각의 영수증마다 평균 100여개의 text region이 있으며 polygon 좌표로 labeling 되어 있음
- 한 이미지 당 최대 글자 영역은 500개까지이며, 500개를 초과하는 글자 영역은 평가 대상에서 제외

### 데이터셋 정보 (Dataset Info)
- 학습 데이터: 3,273장
- 검증 데이터: 404장
- 평가 데이터: 413장
- 라벨 정보: 각 text word 별 좌표 정보 (CSV 형식의 결과 데이터를 파일로 제출)

### 규정 (Rule)
- 학습셋과 검증셋은 구분되어 있지만, 다른 기준으로 재분류 하거나 검증셋을 학습에 사용해도 무방
- 저작권 및 사용권에 문제가 없는 공개 데이터셋과 사전학습 가중치에 대해서 자유롭게 사용 가능
- 평가 데이터셋 시각화와 TTA(Test Time Augmentaion), SSL(Self-Supervised Learning) 등은 데이터 분석 및 학습에 활용 가능
- 자동화된 기법이 아닌 인위적인 labeling을 통한 학습은 절대 불가

### 평가지표 (Evaluation Metric)
- CLEval (Character Level Evaluation)
- 리더보드 순위는 H-Mean(Higher is better)으로 순위 결정 (소수점 4번째자리까지)
- Public 과 Private의 비율은 50:50 이며, 이미지 당 평균 단어 수 균등하게 분배

![equation](./assets/equation.png)

### 유의사항 (Evaluation Guidelines)
- 이번 대회는 Text Detection이 목적이므로 detection 결과에 대해서만 평가
- Ground Truth와 Prediction 모두 transcription 정보는 사용안함
- Ground Truth의 문자 영역에 대한 labeling은 polygon 기준이므로, CLEval 평가도 QUAD가 아닌 POLY방식으로 평가
- polygon의 좌표는 4점 이상을 대상으로 하며, 3점 이하의 영역은 무시되니 주의

---

## **⚙️ Components**
### Directory
```
├── assets/...                              # README images & PDF
├── code/
│   ├── configs/
│   │   ├── preset/
│   │   │   ├── datasets/
│   │   │   │   └── db.yaml                 # Dataset, Transform 등 데이터에 관련된 설정값
│   │   │   ├── lightning_modules/
│   │   │   │   └── base.yaml               # PyTorch Lightning 실행에 관련된 설정값
│   │   │   ├── models/                     # 모델 구성에 필요한 각각의 모듈에 관련된 설정값
│   │   │   │   ├── decoder/
│   │   │   │   │   └── unet.yaml
│   │   │   │   ├── encoder/
│   │   │   │   │   └── timm_backbone.yaml
│   │   │   │   ├── head/
│   │   │   │   │   └── db_head.yaml
│   │   │   │   ├── loss/
│   │   │   │   │   └── db_loss.yaml
│   │   │   │   └── model_example.yaml      # 각 모델 모듈의 설정 파일 및 Optimizer 지정
│   │   │   ├── base.yaml
│   │   │   └── example.yaml                # 각 모듈의 설정 파일 지정
│   │   ├── predict.yaml                    # Runner를 실행할 때 필요한 설정값
│   │   ├── test.yaml                       # Runner를 실행할 때 필요한 설정값
│   │   └── train.yaml                      # Runner를 실행할 때 필요한 설정값
│   ├── ocr/                                # 각 디렉토리마다 __init__.py 존재
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
│   │   │   │   ├── fpnc.py
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
│   ├── outputs/                            # (이하 GitHub 관리안함)
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
│   ├── wandb/...                           # (이하 GitHub 관리안함)
│   ├── baseline.ipynb                      # (GitHub 관리안함)
│   └── eda.ipynb                           # EDA
├── data/                                   # (이하 GitHub 관리안함)
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

---

## **💾 Data Description**
### EDA (Exploratory Data Analysis)
#### 1. 학습 JSON 구조
```
images:
  └─ drp.en_ko.in_house.selectstar_nnnnnn.jpg
    └─ words
      └─ nnnn: 이미지마다 검출된 words의 index 번호 (0으로 채운 4자리 정수값)
        └─ points
          └─ List 4개: X Position, Y Position (검출한 text region의 이미지상 좌표)
```
#### 2. 평가 feature 구성
> 헤더행: filename,polygons<br>
> 데이터행: IMAGE_FILENAME,X Y X Y X Y X Y|X Y X Y X Y X Y|...<br>
> polygon은 최소가 4점이고 실제 추론결과는 정교한 다각형

### Data Preprocessing (or postprocessing)

---

## **🧠 Modeling**
### Model Architecture
#### 1. DBNet
![dbnet1](./assets/dbnet1.png)

#### 2. DBNet++
![dbnet2](./assets/dbnet2.png)

### Model Description
#### 1. ResNet-18
#### 2. ResNet-50
#### 3. HRNet

---

## **🕵️‍♀️ Hypothesis Testing**
#### 1. 요약문 스타일 통일
- **가설:** "~합니다." "~한다." "~함." 등의 불규칙한 동사 어미를 일치시키면 ROUGE가 오르지 않을까?
- **결과:** 동일 코드에 동사 어미만 격식체로 일관화 시켰음에도 리더보드 점수 오히려 하락

---

## **💡 Insights from Trial and Error**
#### EDA & Data Preprocessing
- **시도:** 요약 양쪽의 과다한 공백을 제거하기 위해 EOS 토큰 생성 유도 및 후처리 로직 최적화
- **결과:** 리더보드에 0.003점 정도 기여? 😑

---

## **📊 Experiment Logger**
<table>
  <thead>
    <tr>
      <th align="center">NO.</th>
      <th align="center">DATE</th>
      <th align="center">MODEL</th>
      <th align="center">H-Mean(CV)</th>
      <th align="center">Precision</th>
      <th align="center">Recall</th>
      <th align="center">H-Mean(LB)</th>
      <th align="center">Precision</th>
      <th align="center">Recall</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td align="center">01</td>
      <td align="center">260504</td>
      <td>DBNet_ResNet18 (img 640)</td>
      <td align="center">0.8726</td>
      <td align="center">0.9581</td>
      <td align="center">0.8106</td>
      <td align="center"><b>0.8818</b></td>
      <td align="center"><b>0.9651</b></td>
      <td align="center"><b>0.8194</b></td>
    </tr>
    <tr>
      <td align="center">02</td>
      <td align="center">260504</td>
      <td>DBNet_ResNet18</td>
      <td align="center">0.9493</td>
      <td align="center">0.9775</td>
      <td align="center">0.9266</td>
      <td align="center"><b>0.9489</b></td>
      <td align="center"><b>0.9832</b></td>
      <td align="center"><b>0.9206</b></td>
    </tr>
    <tr>
      <td align="center">03</td>
      <td align="center">260505</td>
      <td>DBNet_ResNet18</td>
      <td align="center">0.9012</td>
      <td align="center">0.9765</td>
      <td align="center">0.8433</td>
      <td align="center"><b>0.9466</b></td>
      <td align="center"><b>0.9772</b></td>
      <td align="center"><b>0.9227</b></td>
    </tr>
    <tr>
      <td align="center">04</td>
      <td align="center">260505</td>
      <td>DBNet_ResNet50 (img 1024)</td>
      <td align="center">0.9331</td>
      <td align="center">0.9593</td>
      <td align="center">0.9175</td>
      <td align="center"><b>0.9564</b></td>
      <td align="center"><b>0.9686</b></td>
      <td align="center"><b>0.9509</b></td>
    </tr>
  </tbody>
</table>
<br>

![wandb_01](./assets/wandb_01.png)
<br>

---

## **🚀 Result**
### Champion Model Info
- **Version:** V04 (DBNet / ResNet-50)
- **Training Time:** 1h 39m
- **Time per Epoch:** 3m 18s
- **Selected CKPT:** Epoch 21
- **Accuracy:** 0.9564

### Leaderboard Rank: No. 1 🏆
![leaderboard mid](./assets/leaderboard_mid.png)
![leaderboard final](./assets/leaderboard_final.png)

### Presentation
- [[PDF] OCR Seminar Presentation](https://github.com/karmakaryx/nlp-dialogue-summarization/blob/main/assets/semiar_nlp.pdf)

---

## **📜 Version Log**
> V01: epoch=8-step=1845.ckpt
- dataset_base_path 변경, train wandb 사용

> V02: epoch=29-step=6150.ckpt
- GPU 활용 최적화를 위한 hyperparameter tuning

> V03: epoch=22-step=4715.ckpt
- optimizer: AdamW, weight_decay: 0.01

> V04: epoch=21-step=4510.ckpt
- weight_decay: 0.0001 (rollback), lr: 0.0001
- val/hmean 기준 checkpoint 생성

---

## **🛠️ etc.**
### Reference
- [[arXiv] Real-time Scene Text Detection with Differentiable Binarization](https://arxiv.org/pdf/1911.08947.pdf)
- [[GitHub] DBNet](https://github.com/MhLiao/DB)
- [[Docs] Hydra](https://hydra.cc/docs/intro/)
- [[Docs] PyTorch Lightning](https://lightning.ai/docs/pytorch/stable/)
- [[arXiv] Character-Level Evaluation for Text Detection and Recognition Tasks](https://arxiv.org/abs/2006.06244)
- [[GitHub] CLEval](https://github.com/clovaai/CLEval)

### Project Retrospective

<br>
