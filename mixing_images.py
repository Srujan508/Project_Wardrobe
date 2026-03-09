import pandas as pd
import os

# Point this to your image folder
IMAGE_DIR = 'test_images'
CSV_FILE = 'deepfashion_triplets.csv'

print(f"Reading {CSV_FILE}...")
df = pd.read_csv(CSV_FILE)
initial_count = len(df)

def images_exist(row):
    # Check if all three images actually exist on your hard drive
    anchor_exists = os.path.exists(os.path.join(IMAGE_DIR, row['anchor_image']))
    positive_exists = os.path.exists(os.path.join(IMAGE_DIR, row['positive_image']))
    negative_exists = os.path.exists(os.path.join(IMAGE_DIR, row['negative_image']))
    
    return anchor_exists and positive_exists and negative_exists

print(f"Auditing {initial_count} triplets. This might take a few seconds...")
# Keep only the rows where the function returns True
valid_df = df[df.apply(images_exist, axis=1)]

final_count = len(valid_df)
valid_df.to_csv('deepfashion_triplets_fixed.csv', index=False)

print(f"--- Audit Complete ---")
print(f"Removed: {initial_count - final_count} missing triplets")
print(f"Saved {final_count} perfectly valid triplets to 'deepfashion_triplets_fixed.csv'")