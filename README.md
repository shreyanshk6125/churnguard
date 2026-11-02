# 🛡️ ChurnGuard — Customer Churn Prediction

An end-to-end machine learning project that predicts customer churn and helps identify customers who may be at risk of leaving a service.

**Live Demo:** [Try ChurnGuard](https://churnguard-2on5zbxi9hmegqpaeyndyj.streamlit.app/)

**GitHub:** [shreyanshk6125/churnguard](https://github.com/shreyanshk6125/churnguard)

---

## 📌 Project Overview

Customer churn is a major challenge for subscription-based businesses. Identifying customers who are likely to leave can help businesses make better customer-retention decisions.

ChurnGuard uses machine learning to estimate the probability of customer churn based on customer demographics, account information, subscription details, and service usage.

The project combines predictive modelling with an interactive Streamlit dashboard to make churn analysis accessible and practical.

## ✨ Features

- **Customer Churn Prediction:** Predict whether a customer is likely to leave.
- **Churn Probability:** Display the estimated probability of churn.
- **Risk Classification:** Categorise customers into Low, Moderate, and High risk.
- **Business Overview:** Display total customers, churned customers, churn rate, and average monthly charges.
- **Interactive Analytics:** Explore churn distribution, churn by contract type, and churn across customer tenure.
- **Interactive Dashboard:** Navigate between analytics and individual customer predictions.

## 📊 Dataset

The project uses the IBM Telco Customer Churn dataset.

- **Total records:** 7,043 customers
- **Total original features:** 20 input features and 1 target column
- **Target variable:** `Churn`
- **Classes:** Yes / No

The dataset contains information about customer demographics, services, contract types, tenure, billing, and payment methods.

**Dataset source:** [IBM Telco Customer Churn Dataset](https://github.com/IBM/telco-customer-churn-on-icp4d)

The deployed application downloads the dataset directly from IBM's public repository.

## ⚙️ Machine Learning Workflow

1. **Data Understanding:** Explored the dataset, feature types, missing values, and churn distribution.
2. **Data Cleaning:** Converted `TotalCharges` into a numeric feature and handled missing values.
3. **Exploratory Data Analysis:** Investigated relationships between churn, contract type, tenure, and customer characteristics.
4. **Preprocessing:** Used `StandardScaler` for numerical features and `OneHotEncoder` for categorical features.
5. **Model Training:** Trained a Logistic Regression baseline and evaluated a Random Forest alternative.
6. **Model Evaluation:** Compared performance using accuracy, precision, recall, F1-score, and confusion matrices.
7. **Threshold Selection:** Selected a classification threshold of 0.30 to improve the balance of churn-class precision and recall.
8. **Deployment:** Saved the preprocessing and prediction pipeline using Joblib and integrated it into Streamlit.

## 📈 Model Performance

The selected Logistic Regression model uses a classification threshold of **0.30**.

| Metric | Result |
|---|---:|
| Accuracy | 74.9% |
| Churn-class Precision | 52% |
| Churn-class F1-score | 0.615 |

The classification threshold was adjusted from the default 0.50 to 0.30 to improve churn detection.

The lower threshold increases the model's willingness to flag potential churners, which can be useful when identifying customers for retention follow-up.

*Note: These results are from the held-out test set. The model provides risk estimates, not certainty about future customer behaviour.*

## 🖥️ Application

The Streamlit application provides two main sections:

### 1. Overview
- Customer and churn KPIs
- Customer churn distribution
- Churn rate by contract type
- Churn rate across customer tenure
- Key business insights

### 2. Predict Churn
- Enter individual customer details
- Generate churn probability
- View predicted churn status
- View customer risk category

## 🛠️ Tech Stack

| Technology | Purpose |
|---|---|
| Python | Programming language |
| Pandas | Data manipulation |
| NumPy | Numerical operations |
| Scikit-learn | Preprocessing and machine learning |
| Joblib | Model persistence |
| Plotly | Interactive visualisations |
| Streamlit | Web application and deployment |
| Jupyter Notebook | Data analysis and experimentation |
| Git & GitHub | Version control |

## 📁 Project Structure

```text
churnguard/
│
├── .streamlit/
│   └── config.toml
│
├── models/
│   └── churnguard_model.joblib
│
├── notebooks/
│   └── 01_customer_churn_analysis.ipynb
│
├── app.py
├── requirements.txt
├── README.md
└── .gitignore
```

## 🚀 Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/shreyanshk6125/churnguard.git
cd churnguard
```

### 2. Create a virtual environment (optional)

```bash
python -m venv venv
```

Activate it:

**macOS / Linux**

```bash
source venv/bin/activate
```

**Windows**

```bash
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the application

```bash
streamlit run app.py
```

The application will open locally in your browser.

## ☁️ Deployment

ChurnGuard is deployed using Streamlit Community Cloud.

**Live application:** https://churnguard-2on5zbxi9hmegqpaeyndyj.streamlit.app/

## ⚠️ Limitations

- The model is trained on a specific telecom customer dataset and may not generalise to other industries without retraining.
- Model predictions are estimates and should not be treated as definitive customer behaviour.
- The selected classification threshold involves a trade-off between precision and recall.
- The model's performance depends on the quality and representativeness of the input data.

## 🔮 Future Improvements

- Experiment with advanced models such as XGBoost and Gradient Boosting.
- Improve hyperparameter tuning and model evaluation.
- Add explainability using SHAP.
- Develop customer-level retention recommendations.
- Add model monitoring and performance tracking.

## 👨‍💻 Author

**Shreyansh Khatri**

Computer Science Engineering Student

[GitHub](https://github.com/shreyanshk6125)

---

If you find this project useful, consider giving the repository a ⭐.
