import pandas as pd
import numpy as np

# Make results reproducible
np.random.seed(42)

# Number of customers
NUM_CUSTOMERS = 5000

# Generate customer IDs
customer_ids = np.arange(1, NUM_CUSTOMERS + 1)

# Generate customer profiles
customers = pd.DataFrame({
    "CUSTOMER_ID": customer_ids,

    # Average transaction amount
    "AVG_AMOUNT": np.round(
        np.random.lognormal(mean=3.5, sigma=0.8, size=NUM_CUSTOMERS),
        2
    ),

    # Typical number of transactions per day
    "TX_FREQUENCY": np.random.poisson(
        lam=3,
        size=NUM_CUSTOMERS
    )
})

# Save the data
output_path = "data/raw/customer_profiles.csv"
customers.to_csv(output_path, index=False)

print("Customer profiles generated successfully!")
print(f"Number of customers: {len(customers)}")
print("\nFirst 5 customers:")
print(customers.head())