import pandas as pd
import os

# --- CONFIGURATION ---
# Make sure this matches where your images actually downloaded!
IMAGE_DIR = 'polyvore_images'
CSV_FILE = 'polyvore_triplets.csv'

print(f"Reading {CSV_FILE}...")
df = pd.read_csv(CSV_FILE)
initial_count = len(df)

def images_exist(row):
    # Check if all three images actually exist on your hard drive
    anchor_exists = os.path.exists(os.path.join(IMAGE_DIR, str(row['anchor_image'])))
    positive_exists = os.path.exists(os.path.join(IMAGE_DIR, str(row['positive_image'])))
    negative_exists = os.path.exists(os.path.join(IMAGE_DIR, str(row['negative_image'])))
    
    return anchor_exists and positive_exists and negative_exists

print(f"Auditing {initial_count} triplets against your hard drive. Please wait...")
# Keep only the rows where all 3 images exist
valid_df = df[df.apply(images_exist, axis=1)]

final_count = len(valid_df)
valid_df.to_csv('polyvore_triplets_fixed.csv', index=False)

print(f"\n--- Audit Complete ---")
print(f"Removed: {initial_count - final_count} triplets with missing/dead images.")
print(f"Saved {final_count} perfectly valid, guaranteed-to-work triplets to 'polyvore_triplets_fixed.csv'")