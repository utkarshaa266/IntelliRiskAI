import pandas as pd
import numpy as np

np.random.seed(42)

# Number of transactions
NUM_TRANSACTIONS = 50000

# Load existing data
customers = pd.read_csv("data/raw/customer_profiles.csv")
links = pd.read_csv("data/raw/customer_terminal_links.csv")

# Generate transaction IDs
transaction_ids = np.arange(1, NUM_TRANSACTIONS + 1)

# Select customers
customer_ids = np.random.choice(
    customers["CUSTOMER_ID"],
    size=NUM_TRANSACTIONS
)

# Generate timestamps
start_date = pd.Timestamp("2026-01-01")

tx_seconds = np.sort(
    np.random.randint(
        0,
        60 * 60 * 24 * 90,
        size=NUM_TRANSACTIONS
    )
)

tx_datetime = start_date + pd.to_timedelta(
    tx_seconds,
    unit="s"
)

# Generate transaction amounts
customer_avg = customers.set_index(
    "CUSTOMER_ID"
)["AVG_AMOUNT"]

avg_amounts = customer_avg.loc[customer_ids].values

tx_amounts = np.round(
    np.random.lognormal(
        mean=np.log(avg_amounts),
        sigma=0.5
    ),
    2
)

# Select one of the customer's normal terminals
customer_terminal_map = (
    links.groupby("CUSTOMER_ID")["TERMINAL_ID"]
    .apply(list)
    .to_dict()
)

terminal_ids = [
    np.random.choice(customer_terminal_map[c])
    for c in customer_ids
]

# Create transaction dataset
transactions = pd.DataFrame({
    "TRANSACTION_ID": transaction_ids,
    "TX_DATETIME": tx_datetime,
    "CUSTOMER_ID": customer_ids,
    "TERMINAL_ID": terminal_ids,
    "TX_AMOUNT": tx_amounts
})

# Save
output_path = "data/raw/transactions.csv"

transactions.to_csv(
    output_path,
    index=False
)

print("Transactions generated successfully!")
print(f"Number of transactions: {len(transactions)}")
print("\nFirst 5 transactions:")
print(transactions.head())