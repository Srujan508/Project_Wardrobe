import pandas as pd
import random
import os

# --- 1. SET YOUR FOLDER HERE ---
# If you combined your images, update this path!
IMAGE_DIR = 'test_images' 

print("Loading master catalog...")
df = pd.read_csv('deepfashion_cleaned.csv')

# --- 2. THE REALITY CHECK ---
print(f"Scanning '{IMAGE_DIR}' for available images...")
# Check the hard drive to see if the file actually exists
df['exists'] = df['image_path'].apply(lambda x: os.path.exists(os.path.join(IMAGE_DIR, x)))

# Drop all rows where the image is missing from your folder
available_df = df[df['exists'] == True].copy()
print(f"Found {len(available_df)} available images to build outfits from.")

# --- 3. BUILD THE TRIPLETS ---
tops = available_df[available_df['category'].str.contains('Tees|Tanks|Blouses|Shirts|Sweaters|Jackets', case=False)]
bottoms = available_df[available_df['category'].str.contains('Shorts|Pants|Denim|Skirts', case=False)]

triplets = []
print("Generating 10,000 guaranteed valid Triplets...")

# We loop 10,000 times to create a robust dataset
for i in range(10000):
    # Randomly select a top to be the Anchor
    top = tops.sample(1).iloc[0]
    top_weather = top['weather_suitability']
    
    possible_positives = bottoms[bottoms['weather_suitability'] == top_weather]
    possible_negatives = bottoms[bottoms['weather_suitability'] != top_weather]
    
    # Only make the triplet if we have valid matching and clashing options available
    if not possible_positives.empty and not possible_negatives.empty:
        positive_item = possible_positives.sample(1).iloc[0]
        negative_item = possible_negatives.sample(1).iloc[0]
        
        triplets.append({
            'anchor_image': top['image_path'],
            'positive_image': positive_item['image_path'],
            'negative_image': negative_item['image_path']
        })

triplet_df = pd.DataFrame(triplets)
triplet_df.to_csv('deepfashion_triplets_fixed.csv', index=False)

print(f"Success! Saved {len(triplet_df)} valid triplets to 'deepfashion_triplets_fixed.csv'")