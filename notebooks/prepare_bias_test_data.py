"""
Prepare test data with sensitive attributes for bias detection
"""
import pandas as pd
import json

print("=" * 60)
print("Preparing Bias Test Dataset")
print("=" * 60)

# Load the test data
df = pd.read_csv('data/test.csv')

print(f"\n✅ Loaded test data: {len(df)} samples")
print(f"Columns: {list(df.columns)}")

# Rename loan_approved to true_label for clarity
df = df.rename(columns={'loan_approved': 'true_label'})

# Sample 100 records for testing (mix of both genders and ethnicities)
# Ensure balanced representation
sample_df = df.groupby(['gender', 'ethnicity'], group_keys=False).apply(
    lambda x: x.sample(min(len(x), 5), random_state=42)
).reset_index(drop=True)

print(f"\n✅ Sampled {len(sample_df)} records")
print(f"\nDistribution by gender:")
print(sample_df['gender'].value_counts())
print(f"\nDistribution by ethnicity:")
print(sample_df['ethnicity'].value_counts())
print(f"\nApproval rate by gender:")
print(sample_df.groupby('gender')['true_label'].mean())
print(f"\nApproval rate by ethnicity:")
print(sample_df.groupby('ethnicity')['true_label'].mean())

# Save as JSON for API testing
bias_test_data = sample_df.to_dict('records')

with open('data/bias_test_data.json', 'w') as f:
    json.dump(bias_test_data, f, indent=2)

print(f"\n✅ Saved bias test data to data/bias_test_data.json")

# Create a smaller sample for quick testing (20 samples)
small_sample = sample_df.head(20).to_dict('records')

with open('data/bias_test_small.json', 'w') as f:
    json.dump(small_sample, f, indent=2)

print(f"✅ Saved small sample to data/bias_test_small.json")

print("\n" + "=" * 60)
print("✅ Bias Test Data Ready!")
print("=" * 60)