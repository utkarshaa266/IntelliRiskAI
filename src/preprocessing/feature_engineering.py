import pandas as pd
import numpy as np

# Load transaction data
df = pd.read_csv("data/raw/transactions_with_fraud.csv")

# Convert datetime
df["TX_DATETIME"] = pd.to_datetime(df["TX_DATETIME"])

# Sort transactions chronologically
df = df.sort_values("TX_DATETIME").reset_index(drop=True)

# --------------------------------------------------
# TIME FEATURES
# --------------------------------------------------

df["TRANSACTION_HOUR"] = df["TX_DATETIME"].dt.hour

df["DAY_OF_WEEK"] = df["TX_DATETIME"].dt.dayofweek

df["IS_WEEKEND"] = (
    df["DAY_OF_WEEK"] >= 5
).astype(int)

df["IS_NIGHT"] = (
    (df["TRANSACTION_HOUR"] < 6) |
    (df["TRANSACTION_HOUR"] >= 23)
).astype(int)

# --------------------------------------------------
# HISTORICAL CUSTOMER FEATURES
# --------------------------------------------------

# Previous transaction amount of the same customer
df["PREVIOUS_CUSTOMER_AMOUNT"] = (
    df.groupby("CUSTOMER_ID")["TX_AMOUNT"]
    .shift(1)
)

# Historical average amount BEFORE current transaction
df["CUSTOMER_PREVIOUS_AVG"] = (
    df.groupby("CUSTOMER_ID")["TX_AMOUNT"]
    .transform(
        lambda x: x.shift(1).expanding().mean()
    )
)

# Number of previous transactions by customer
df["CUSTOMER_PREVIOUS_TX_COUNT"] = (
    df.groupby("CUSTOMER_ID")
    .cumcount()
)

# Replace missing historical values
df["CUSTOMER_PREVIOUS_AVG"] = (
    df["CUSTOMER_PREVIOUS_AVG"]
    .fillna(df["TX_AMOUNT"])
)

# --------------------------------------------------
# AMOUNT BEHAVIOR
# --------------------------------------------------

df["AMOUNT_TO_CUSTOMER_AVG"] = (
    df["TX_AMOUNT"] /
    (df["CUSTOMER_PREVIOUS_AVG"] + 0.01)
)

# --------------------------------------------------
# HISTORICAL TERMINAL FEATURE
# --------------------------------------------------

# Number of previous transactions at this terminal
df["TERMINAL_PREVIOUS_TX_COUNT"] = (
    df.groupby("TERMINAL_ID")
    .cumcount()
)

# --------------------------------------------------
# SAVE
# --------------------------------------------------

output_path = "data/processed/fraud_features.csv"

df.to_csv(output_path, index=False)

print("Historical feature engineering completed!")

print("\nDataset shape:")
print(df.shape)

print("\nFeatures:")
print(df.columns.tolist())

print("\nFirst 5 rows:")
print(df.head())