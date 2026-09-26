import pandas as pd
import numpy as np

np.random.seed(42)

NUM_TERMINALS = 500

terminal_ids = np.arange(1, NUM_TERMINALS + 1)

terminals = pd.DataFrame({
    "TERMINAL_ID": terminal_ids,

    "TERMINAL_TYPE": np.random.choice(
        ["ATM", "POS", "ONLINE"],
        size=NUM_TERMINALS,
        p=[0.2, 0.6, 0.2]
    )
})

output_path = "data/raw/terminal_profiles.csv"

terminals.to_csv(output_path, index=False)

print("Terminal profiles generated successfully!")
print(f"Number of terminals: {len(terminals)}")
print("\nFirst 5 terminals:")
print(terminals.head())