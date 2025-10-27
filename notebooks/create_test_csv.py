"""
Create test CSV files for bias detection upload
"""
import pandas as pd

print("=" * 60)
print("Creating Test CSV Files for Bias Detection")
print("=" * 60)

# Load test data
df = pd.read_csv('data/test.csv')

# Rename for consistency
df = df.rename(columns={'loan_approved': 'true_label'})

# Create small test CSV (20 samples)
small_sample = df.head(20)
small_sample.to_csv('data/bias_test_upload_small.csv', index=False)
print(f"\n✅ Created: data/bias_test_upload_small.csv ({len(small_sample)} rows)")
print(f"   Columns: {list(small_sample.columns)}")

# Create medium test CSV (50 samples)
medium_sample = df.head(50)
medium_sample.to_csv('data/bias_test_upload_medium.csv', index=False)
print(f"\n✅ Created: data/bias_test_upload_medium.csv ({len(medium_sample)} rows)")

# Create large test CSV (100 samples) with balanced groups
large_sample = df.groupby(['gender', 'ethnicity'], group_keys=False).apply(
    lambda x: x.sample(min(len(x), 7), random_state=42)
).reset_index(drop=True)
large_sample.to_csv('data/bias_test_upload_large.csv', index=False)
print(f"\n✅ Created: data/bias_test_upload_large.csv ({len(large_sample)} rows)")

print(f"\n📊 Distribution in large CSV:")
print(f"\nBy Gender:")
print(large_sample['gender'].value_counts())
print(f"\nBy Ethnicity:")
print(large_sample['ethnicity'].value_counts())
print(f"\nApproval Rate by Gender:")
print(large_sample.groupby('gender')['true_label'].mean())

print("\n" + "=" * 60)
print("✅ Test CSV Files Ready!")
print("=" * 60)