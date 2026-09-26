import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score
)

# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

df = pd.read_csv("data/processed/fraud_features.csv")

df["TX_DATETIME"] = pd.to_datetime(df["TX_DATETIME"])

# --------------------------------------------------
# FEATURES
# --------------------------------------------------

FEATURES = [
    "TX_AMOUNT",
    "TRANSACTION_HOUR",
    "DAY_OF_WEEK",
    "IS_WEEKEND",
    "IS_NIGHT",
    "CUSTOMER_PREVIOUS_AVG",
    "CUSTOMER_PREVIOUS_TX_COUNT",
    "AMOUNT_TO_CUSTOMER_AVG",
    "TERMINAL_PREVIOUS_TX_COUNT"
]

TARGET = "FRAUD"

# --------------------------------------------------
# TIME-BASED TRAIN / TEST SPLIT
# --------------------------------------------------

# First 80% = training
# Last 20% = testing

split_index = int(len(df) * 0.80)

train_df = df.iloc[:split_index]
test_df = df.iloc[split_index:]

X_train = train_df[FEATURES]
y_train = train_df[TARGET]

X_test = test_df[FEATURES]
y_test = test_df[TARGET]

print("Time-based split completed!")

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))

print("\nTraining fraud distribution:")
print(y_train.value_counts())

print("\nTesting fraud distribution:")
print(y_test.value_counts())

# --------------------------------------------------
# RANDOM FOREST
# --------------------------------------------------

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight="balanced",
    n_jobs=-1
)

print("\nTraining Random Forest...")

model.fit(X_train, y_train)

print("Training completed!")

# --------------------------------------------------
# PREDICTION
# --------------------------------------------------

y_pred = model.predict(X_test)

y_probability = model.predict_proba(X_test)[:, 1]

# --------------------------------------------------
# EVALUATION
# --------------------------------------------------

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nROC-AUC Score:")
print(roc_auc_score(y_test, y_probability))

# --------------------------------------------------
# FEATURE IMPORTANCE
# --------------------------------------------------

importance = pd.DataFrame({
    "Feature": FEATURES,
    "Importance": model.feature_importances_
}).sort_values(
    "Importance",
    ascending=False
)

print("\nFeature Importance:")
print(importance)