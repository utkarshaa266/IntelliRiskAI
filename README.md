# IntelliRiskAI 🛡️

### Intelligent Fraud Detection & Risk Analysis using Machine Learning

IntelliRiskAI is a machine learning project designed to identify potentially fraudulent financial transactions by analyzing transaction patterns, customer behavior, transaction timing, and terminal activity.

The project focuses on building a realistic end-to-end fraud detection pipeline — from data generation and preprocessing to feature engineering, model training, and evaluation.

---

## 🎯 Problem Statement

Financial fraud can cause significant losses to businesses and customers. Traditional rule-based systems may struggle to identify new or unusual transaction patterns.

**IntelliRiskAI** aims to detect suspicious transactions by learning behavioral patterns from historical transaction data.

The system analyzes factors such as:

* Transaction amount
* Customer's previous transaction behavior
* Transaction frequency
* Transaction timing
* Customer spending patterns
* Terminal activity
* Changes in transaction amount

---

## 🚀 Project Objectives

* Generate realistic transaction data for experimentation
* Create customer and terminal profiles
* Simulate fraudulent transaction scenarios
* Perform time-aware feature engineering
* Build a machine learning fraud detection model
* Handle highly imbalanced fraud data
* Evaluate the model using fraud-focused metrics
* Develop an extensible ML pipeline for future improvements

---

## 🧠 Machine Learning Approach

The current system uses a **Random Forest Classifier** as the baseline model.

### Pipeline

```text
Customer Profiles
       ↓
Terminal Profiles
       ↓
Customer-Terminal Relationships
       ↓
Transaction Generation
       ↓
Fraud Scenario Generation
       ↓
Historical Feature Engineering
       ↓
Time-Based Train/Test Split
       ↓
Random Forest Model
       ↓
Fraud Prediction
       ↓
Model Evaluation
```

---

## 📊 Features

The current model uses the following features:

| Feature                      | Description                                                |
| ---------------------------- | ---------------------------------------------------------- |
| `TX_AMOUNT`                  | Transaction amount                                         |
| `TRANSACTION_HOUR`           | Hour when transaction occurred                             |
| `DAY_OF_WEEK`                | Day of the week                                            |
| `IS_WEEKEND`                 | Whether transaction occurred on weekend                    |
| `IS_NIGHT`                   | Whether transaction occurred during night                  |
| `CUSTOMER_PREVIOUS_AVG`      | Historical average transaction amount of customer          |
| `CUSTOMER_PREVIOUS_TX_COUNT` | Number of previous transactions by customer                |
| `AMOUNT_TO_CUSTOMER_AVG`     | Current amount compared with customer's historical average |
| `TERMINAL_PREVIOUS_TX_COUNT` | Historical transaction activity of terminal                |

---

## 🗂️ Project Structure

```text
IntelliRiskAI/
│
├── data/
│   ├── raw/
│   │   ├── customer_profiles.csv
│   │   ├── terminal_profiles.csv
│   │   ├── customer_terminal_links.csv
│   │   ├── transactions.csv
│   │   └── transactions_with_fraud.csv
│   │
│   └── processed/
│       └── fraud_features.csv
│
├── notebooks/
│
├── src/
│   ├── data_generation/
│   │   ├── generate_customers.py
│   │   ├── generate_terminals.py
│   │   ├── generate_customer_terminal_links.py
│   │   ├── generate_transactions.py
│   │   └── generate_fraud.py
│   │
│   ├── preprocessing/
│   │   └── feature_engineering.py
│   │
│   └── models/
│       └── train_model.py
│
├── README.md
└── requirements.txt
```

---

## 🛠️ Technologies Used

* **Python**
* **Pandas**
* **NumPy**
* **Scikit-learn**
* **Machine Learning**
* **Data Preprocessing**
* **Feature Engineering**
* **Random Forest**
* **Git & GitHub**

---

## 📦 Dataset

Instead of using a pre-trained model or a ready-made prediction system, IntelliRiskAI generates its own experimental transaction dataset.

The dataset currently contains:

* **5,000 customers**
* **500 terminals**
* **50,000 transactions**
* **500 simulated fraudulent transactions**
* Approximately **1% fraud rate**

Fraud scenarios currently include:

1. Abnormally high transaction amounts
2. Unusual transaction timing
3. Unusual terminal activity
4. Rapid transaction activity

---

## 🔍 Data Processing

The project uses chronological feature engineering to calculate historical customer behavior.

For example:

```text
Previous Transactions
        ↓
Customer Historical Average
        ↓
Compare Current Transaction
        ↓
Behavioral Feature
        ↓
Fraud Detection Model
```

Historical features are calculated using previous transactions rather than future information.

This helps reduce data leakage and makes the evaluation closer to a real-world fraud detection scenario.

---

## 🤖 Model

### Random Forest Classifier

The initial baseline model uses:

```text
n_estimators = 200
class_weight = balanced
random_state = 42
```

A **time-based 80/20 train-test split** is used instead of a random split.

This better represents a real-world scenario where the model learns from past transactions and is evaluated on later transactions.

---

## 📈 Initial Results

The current baseline model produced:

| Metric          | Result |
| --------------- | -----: |
| Accuracy        |   ~99% |
| Fraud Precision |   1.00 |
| Fraud Recall    |   0.19 |
| Fraud F1-score  |   0.32 |
| ROC-AUC         |  0.673 |

### Important Note

Accuracy alone is not a suitable metric for fraud detection because fraudulent transactions represent only a small percentage of the dataset.

Therefore, IntelliRiskAI focuses more heavily on:

* Precision
* Recall
* F1-score
* ROC-AUC
* Confusion Matrix

The current baseline demonstrates that there is significant room for improvement, particularly in detecting more fraudulent transactions.

---

## 🔄 Future Improvements

The project is actively being developed.

Planned improvements include:

* Advanced customer behavioral features
* Time since previous transaction
* Transaction amount changes
* Customer spending variability
* Terminal-customer relationships
* Better fraud scenario simulation
* Feature selection
* Hyperparameter tuning
* XGBoost / LightGBM comparison
* Anomaly detection models
* Probability-based risk scoring
* Explainable AI
* Real-time fraud prediction
* Interactive fraud monitoring dashboard

---

## 🎓 Learning Goals

This project is being developed from scratch to gain practical experience in:

* Data generation
* Data preprocessing
* Exploratory data analysis
* Feature engineering
* Handling imbalanced datasets
* Supervised machine learning
* Model evaluation
* Time-based validation
* ML pipeline development
* Git/GitHub project management

---

## 👨‍💻 Author

**Utkarsha Athare**

B.Tech — Artificial Intelligence & Data Science

GitHub: `@utkarshaa266`

---

## ⭐ Project Status

**Status: 🚧 Under Development**

IntelliRiskAI is currently in the **baseline model and feature improvement stage**. The model and dataset will continue to evolve as additional behavioral features and machine learning techniques are implemented.
