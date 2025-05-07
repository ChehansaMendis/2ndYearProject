import pandas as pd

# Load dataset
df_fake = pd.read_csv("Fake.csv")
df_real = pd.read_csv("True.csv")

# Add labels (1 = Real, 0 = Fake)
df_fake["label"] = 0
df_real["label"] = 1

# Combine both datasets
df = pd.concat([df_fake, df_real])

# Shuffle data
df = df.sample(frac=1).reset_index(drop=True)

print(df.head())  # Show sample data
