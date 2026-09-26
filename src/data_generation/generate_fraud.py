import pandas as pd
import numpy as np

np.random.seed(42)

# Load data
transactions = pd.read_csv("data/raw/transactions.csv")
customers = pd.read_csv("data/raw/customer_profiles.csv")
links = pd.read_csv("data/raw/customer_terminal_links.csv")

# Start with every transaction as legitimate
transactions["FRAUD"] = 0

# Number of fraudulent transactions
NUM_FRAUD = 500

# Select transactions to modify
fraud_indices = np.random.choice(
    transactions.index,
    size=NUM_FRAUD,
    replace=False
)

# Split fraud transactions into scenarios
scenario_size = NUM_FRAUD // 4

high_amount_indices = fraud_indices[:scenario_size]
night_indices = fraud_indices[scenario_size:2 * scenario_size]
unusual_terminal_indices = fraud_indices[2 * scenario_size:3 * scenario_size]
rapid_indices = fraud_indices[3 * scenario_size:]

# --------------------------------------------------
# Scenario 1: Abnormally high transaction amount
# --------------------------------------------------

transactions.loc[
    high_amount_indices,
    "TX_AMOUNT"
] *= np.random.uniform(
    10,
    30,
    size=len(high_amount_indices)
)

transactions.loc[
    high_amount_indices,
    "FRAUD"
] = 1


# --------------------------------------------------
# Scenario 2: Unusual transaction time
# --------------------------------------------------

transactions.loc[
    night_indices,
    "TX_DATETIME"
] = pd.to_datetime(
    transactions.loc[
        night_indices,
        "TX_DATETIME"
    ]
).dt.normalize() + pd.to_timedelta(
    np.random.randint(
        0,
        5 * 60 * 60,
        size=len(night_indices)
    ),
    unit="s"
)

transactions.loc[
    night_indices,
    "FRAUD"
] = 1


# --------------------------------------------------
# Scenario 3: Unusual terminal
# --------------------------------------------------

all_terminals = terminals = links["TERMINAL_ID"].unique()

transactions.loc[
    unusual_terminal_indices,
    "TERMINAL_ID"
] = np.random.choice(
    all_terminals,
    size=len(unusual_terminal_indices)
)

transactions.loc[
    unusual_terminal_indices,
    "FRAUD"
] = 1


# --------------------------------------------------
# Scenario 4: Rapid transaction activity
# --------------------------------------------------

rapid_times = pd.to_datetime(
    transactions.loc[
        rapid_indices,
        "TX_DATETIME"
    ]
)

transactions.loc[
    rapid_indices,
    "TX_DATETIME"
] = rapid_times.dt.floor("min")

transactions.loc[
    rapid_indices,
    "FRAUD"
] = 1


# Save final dataset
output_path = "data/raw/transactions_with_fraud.csv"

transactions.to_csv(
    output_path,
    index=False
)

print("Fraud scenarios generated successfully!")

print(f"Total transactions: {len(transactions)}")
print(f"Fraud transactions: {transactions['FRAUD'].sum()}")
print(
    f"Legitimate transactions: "
    f"{(transactions['FRAUD'] == 0).sum()}"
)

print("\nFraud distribution:")
print(transactions["FRAUD"].value_counts())
