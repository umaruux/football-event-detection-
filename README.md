# Automated Detection and Counting of Free Kicks and Penalties
## Introduction to Data Science — Group Project

**Group Members:** Ahmed Umar Mirza (01-134232-028), Bisma Rauf (01-134232-047)  
**Dataset:** SoccerNet (https://www.soccer-net.org/data)

---

## Project Structure

```
football_event_detection/
├── 01_download_data.py        # SoccerNet data download
├── 02_extract_features.py     # Frame extraction + YOLO + feature engineering
├── 03_train_model.py          # EDA, training, evaluation
├── 04_predict.py              # Run on new video
├── requirements.txt
└── README.md
```

## Setup

```bash
pip install -r requirements.txt
```

## Run Order

```bash
python 01_download_data.py
python 02_extract_features.py
python 03_train_model.py
python 04_predict.py --video path/to/match.mkv
```
