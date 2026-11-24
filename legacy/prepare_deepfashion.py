import pandas as pd

# 1. Define your files
files = ['train_ann_file.txt', 'val_ann_file.txt', 'test_ann_file.txt']
all_dataframes = []

# 2. Load and stitch the files together
print("Loading all DeepFashion annotations...")
for file in files:
    print(f" -> Reading {file}...")
    # '\s+' tells Pandas that the columns are separated by spaces
    df_temp = pd.read_csv(file, sep='\s+', header=None)
    all_dataframes.append(df_temp)

# Combine them into one giant DataFrame
df = pd.concat(all_dataframes, ignore_index=True)
print(f"\nSuccessfully combined! Total items found: {len(df)}")

# Rename the first column to 'image_path'
df.rename(columns={0: 'image_path'}, inplace=True)

# 3. Extract Category from the Filename
# Filenames look like: WOMEN-Blouses_Shirts-id_...
df['gender'] = df['image_path'].apply(lambda x: str(x).split('-')[0])
df['category'] = df['image_path'].apply(lambda x: str(x).split('-')[1].replace('_', ' '))

# 4. Apply Your Travel & Weather Logic
def apply_travel_logic(category):
    category = category.lower()
    
    # Defaults
    weather = "All-Season"
    waterproof = False
    warmth = 2

    # Winter / Heavy
    if any(word in category for word in ['jackets', 'coats', 'sweaters', 'sweatshirts']):
        weather = "Winter"
        warmth = 4
        if 'jackets' in category: waterproof = True 
        
    # Summer / Light
    elif any(word in category for word in ['shorts', 'tees', 'tanks', 'rompers']):
        weather = "Summer"
        warmth = 1

    # Denim/Pants are usually all-season, warmth 3
    elif 'denim' in category or 'pants' in category:
        weather = "All-Season"
        warmth = 3

    return pd.Series([weather, waterproof, warmth])

# Apply the logic to create new columns
print("Applying Travel Intelligence to the combined dataset...")
df[['weather_suitability', 'is_waterproof', 'warmth_level']] = df['category'].apply(apply_travel_logic)

# Keep only the columns we need
final_df = df[['image_path', 'gender', 'category', 'weather_suitability', 'is_waterproof', 'warmth_level']]

# Save to a clean CSV
final_df.to_csv('deepfashion_cleaned.csv', index=False)
print("\nData cleaned and saved to 'deepfashion_cleaned.csv'!")
print(final_df.head())