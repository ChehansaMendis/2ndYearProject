import pandas as pd

# Load datasets
fake = pd.read_csv("Fake.csv")
real = pd.read_csv("True.csv")

# Add labels
fake["label"] = 0  # Fake news → 0
real["label"] = 1  # Real news → 1

# Combine both datasets
data = pd.concat([fake, real], axis=0)

# Shuffle the dataset
data = data.sample(frac=1, random_state=42).reset_index(drop=True)

# Save cleaned dataset
data.to_csv("news_dataset.csv", index=False)

print("✅ Data preprocessing complete! Saved as news_dataset.csv.")
