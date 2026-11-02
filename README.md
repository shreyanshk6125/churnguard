# ChurnGuard — Customer Churn Prediction

ChurnGuard is a machine-learning project that predicts whether a telecom customer is likely to leave a service provider. The current version focuses on a reproducible analysis workflow, baseline model comparison, churn-focused evaluation, and interpretation of model coefficients.

## Project status

**Current stage:** notebook-based ML workflow completed; interactive application is planned but is not included yet.

## Problem statement

Customer churn affects recurring revenue and makes customer retention more expensive. The goal of this project is to identify customers who may churn so that a business can prioritize retention outreach.

This is a prediction task, not a causal analysis: model coefficients describe learned associations and should not be interpreted as proof that a feature causes churn.

## Dataset

The notebook loads the public IBM Telco Customer Churn CSV from:

<https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv>

The dataset contains 7,043 customer records and 21 columns, including the `Churn` target. The notebook excludes `customerID` from the model features and converts `TotalCharges` to numeric. The 11 blank `TotalCharges` entries are set to zero after conversion.

The dataset is downloaded by the notebook; the raw CSV is not committed to this repository.

## Workflow

1. Load and inspect the dataset.
2. Explore churn distribution, contract type, and customer tenure.
3. Clean `TotalCharges`.
4. Separate features and target; use a stratified 80/20 train-test split (`random_state=42`).
5. Standardize numerical features and one-hot encode categorical features.
6. Train Logistic Regression and Random Forest classifiers.
7. Compare models and test Logistic Regression decision thresholds of 0.5, 0.4, and 0.3.
8. Inspect Logistic Regression coefficients.
9. Evaluate the selected threshold and save the fitted preprocessing + model pipeline.

## Results

The held-out test set contains 1,409 customers, including 374 churners.

### Logistic Regression at the default threshold (0.50)

| Metric | Score |
|---|---:|
| Accuracy | 80.55% |
| Churn precision | 65.7% |
| Churn recall | 55.9% |
| Churn F1-score | 60.4% |

### Logistic Regression at the selected threshold (0.30)

| Metric | Score |
|---|---:|
| Accuracy | 74.95% |
| Churn precision | 51.9% |
| Churn recall | 75.4% |
| Churn F1-score | 61.5% |
| ROC-AUC | 84.21% |
| PR-AUC (average precision) | 63.43% |

At threshold 0.30, the confusion matrix (rows = actual `No`, `Yes`; columns = predicted `No`, `Yes`) is:

| | Predicted No | Predicted Yes |
|---|---:|---:|
| Actual No | 774 | 261 |
| Actual Yes | 92 | 282 |

The 0.30 threshold is selected as an initial recall-oriented operating point: it identifies more actual churners than the 0.50 threshold, at the cost of more false positives. It is a project decision for further validation, not a universally optimal threshold. A real deployment should choose a threshold using retention-team capacity and the business costs of false positives and false negatives.

### Model comparison at the default threshold

| Model | Accuracy | Churn precision | Churn recall | Churn F1 |
|---|---:|---:|---:|---:|
| Logistic Regression | 80.55% | 65.7% | 55.9% | 60.4% |
| Random Forest | 77.86% | 60.5% | 47.9% | 53.4% |

## Repository structure

```text
churnguard/
├── README.md
├── requirements.txt
├── .gitignore
├── notebooks/
│   └── churn_analysis.ipynb
└── models/
    └── .gitkeep
```

The notebook creates `models/churnguard_model.joblib` when its save-model cell is run. The model artifact is generated locally and is not included in this initial repository bundle.

## Setup and run

Python 3.10+ is recommended.

```bash
git clone https://github.com/shreyanshk6125/churnguard.git
cd churnguard

python -m venv .venv
source .venv/bin/activate        # macOS / Linux
# .venv\\Scripts\\activate     # Windows

pip install -r requirements.txt
jupyter notebook
```

Open `notebooks/churn_analysis.ipynb` and run the cells from top to bottom. An internet connection is required to download the dataset.

## Next steps

- Add a prediction module that loads the saved pipeline.
- Build an interactive Streamlit application for single-customer churn predictions.
- Add application screenshots and usage documentation.
- Validate the selected threshold against explicit business costs.

## Disclaimer

This project is for educational and portfolio purposes. Model performance is based on one stratified train-test split and should not be treated as production validation.
