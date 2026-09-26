import pandas as pd
import numpy as np

np.random.seed(42)

# Load customer and terminal profiles
customers = pd.read_csv("data/raw/customer_profiles.csv")
terminals = pd.read_csv("data/raw/terminal_profiles.csv")

# Number of terminals each customer normally uses
NUM_TERMINALS_PER_CUSTOMER = 3

links = []

for customer_id in customers["CUSTOMER_ID"]:
    selected_terminals = np.random.choice(
        terminals["TERMINAL_ID"],
        size=NUM_TERMINALS_PER_CUSTOMER,
        replace=False
    )

    for terminal_id in selected_terminals:
        links.append({
            "CUSTOMER_ID": customer_id,
            "TERMINAL_ID": terminal_id
        })

customer_terminal_links = pd.DataFrame(links)

# Save
output_path = "data/raw/customer_terminal_links.csv"

customer_terminal_links.to_csv(
    output_path,
    index=False
)

print("Customer-terminal links generated successfully!")
print(f"Number of links: {len(customer_terminal_links)}")
print("\nFirst 10 links:")
print(customer_terminal_links.head(10))