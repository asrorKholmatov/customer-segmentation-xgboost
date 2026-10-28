# SegmentIQ — Customer Segmentation with XGBoost

Predict which marketing segment (**A, B, C or D**) a customer belongs to, from nine basic profile fields.
The trained XGBoost model runs **entirely in the browser** — the web app is a static site with no backend, and customer data never leaves the user's device.

**Live demo:** _LIVE_URL_

![stack](https://img.shields.io/badge/model-XGBoost-4f46e5) ![python](https://img.shields.io/badge/python-3.12-blue) ![deploy](https://img.shields.io/badge/deploy-Netlify-00c7b7)

## What it does

| Feature | Description |
|---|---|
| Single prediction | Fill in a form, get the segment, confidence for each class and a recommended marketing approach — updates live as you type. |
| Batch scoring | Drop a CSV of customers, get the segment distribution and a downloadable CSV with `Predicted_Segment`, `Confidence`, `P_A…P_D`. |
| Segment profiles | Who each segment is: age, marital status, education, top profession, spending. |
| Model report | Accuracy, F1, confusion matrix and feature importance. |
| REST API | `FastAPI` service for integrating with a CRM or data pipeline. |

## Dataset

[Kaggle — Customer Segmentation (vetrirah/customer)](https://www.kaggle.com/datasets/vetrirah/customer): 8,068 customers of an automobile company, each assigned to one of four segments by the sales team.

Features: `Gender, Ever_Married, Age, Graduated, Profession, Work_Experience, Spending_Score, Family_Size, Var_1` → target `Segmentation`.

## Results (held-out 20% test set, 1,614 customers)

| Metric | Value |
|---|---|
| Accuracy | **55.1%** |
| Macro F1 | 0.534 |
| 5-fold CV accuracy | 53.4% |
| Majority-class baseline | 28.1% |

| Segment | Precision | Recall | F1 |
|---|---|---|---|
| A | 0.45 | 0.45 | 0.45 |
| B | 0.46 | 0.35 | 0.39 |
| C | 0.59 | 0.59 | 0.59 |
| D | 0.65 | 0.78 | 0.71 |

The segments overlap heavily — customers with identical profiles often belong to different segments — so this is a hard problem; public solutions on this dataset reach roughly 50–55%. The model is about **2× better than guessing**, and most reliable for segment D. The confidence score is useful: high-confidence predictions are noticeably more accurate.

Top drivers (XGBoost gain): profession (Artist, Healthcare), spending score, age, graduation.

## Segments

| | Persona | Profile |
|---|---|---|
| **A** | Creative Individualists | ~45 y/o, artists & entertainers, small households, 69% low spenders |
| **B** | Established Professionals | ~48 y/o, 73% married, 72% graduates, balanced spending |
| **C** | Affluent Family Builders | ~49 y/o, 79% married, 82% graduates, highest share of average/high spenders |
| **D** | Young Starters | ~33 y/o, 71% never married, healthcare-heavy, 87% low spenders |

## Project structure

```
├── src/
│   ├── dataset.py     # download (Kaggle CLI or mirror), load, feature encoding
│   ├── train.py       # CV tuning, early stopping, evaluation, export
│   └── utils.py       # exports trees to compact JSON + verifies parity with XGBoost
├── app/
│   ├── main.py        # FastAPI endpoints
│   └── model.py       # model loading & scoring
├── models/xgb_model.json   # trained XGBoost model
├── web/               # static website (deployed to Netlify)
│   ├── index.html, styles.css, app.js
│   ├── model.json     # exported trees, metrics, segment profiles
│   └── sample_customers.csv  # 150 held-out test customers
└── netlify.toml
```

## Run it yourself

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

python -m src.train          # downloads data, trains, writes models/ and web/model.json
python -m http.server 8000 --directory web   # open http://localhost:8000
```

> macOS: XGBoost needs OpenMP — `brew install libomp` (or `conda install -c conda-forge llvm-openmp`).

### API

```bash
uvicorn app.main:app --reload
curl -X POST localhost:8000/predict -H 'Content-Type: application/json' -d '{
  "Gender": "Male", "Ever_Married": "No", "Age": 25, "Graduated": "No",
  "Profession": "Healthcare", "Work_Experience": 1, "Spending_Score": "Low",
  "Family_Size": 4, "Var_1": "Cat_4"}'
# {"segment":"D","confidence":0.6078,"probabilities":{"A":0.0848,"B":0.1265,"C":0.181,"D":0.6078}}
```

Interactive docs at `/docs`. `POST /predict/batch` accepts a list of customers.

## How the in-browser model works

1. `src/train.py` dumps every XGBoost tree and flattens it into arrays: `[feature, threshold/leaf, yes, no, missing]`.
2. The per-class intercept is derived by comparing XGBoost's raw margins with the tree sums, and checked to be constant.
3. `web/app.js` encodes the input exactly like `src/dataset.py::encode`, walks all 1,396 trees, adds each leaf to its class, and applies softmax.

Browser predictions match `predict_proba` from Python to ~5 decimal places.

## Modelling choices

- Binary fields → 0/1, `Spending_Score` → ordinal (Low < Average < High), `Profession` and `Var_1` → one-hot.
- Missing values are left as `NaN`; XGBoost learns the best default branch for them instead of imputing.
- Hyper-parameters (`max_depth`, `learning_rate`, `min_child_weight`) chosen by 5-fold stratified CV; number of trees by early stopping on a validation split; the test set is used once.
