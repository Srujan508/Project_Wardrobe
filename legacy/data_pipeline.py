import pandas as pd
import sqlite3

# 1. Create Mock Raw Data (Simulating a downloaded dataset)
mock_data = {
    'image_path': ['images/item1.jpg', 'images/item2.jpg', 'images/item3.jpg', 'images/item4.jpg'],
    'category': ['Sweater', 'T-shirt', 'Jacket', 'Shorts'],
    'material': ['Wool', 'Cotton', 'Nylon', 'Denim'],
    'color': ['Navy', 'White', 'Black', 'Blue']
}
df_raw = pd.DataFrame(mock_data)
print("Raw Data Loaded:")
print(df_raw.head(), "\n")

# 2. The Intelligence Layer: Add Travel Tags
def add_travel_tags(row):
    category = row['category'].lower()
    material = row['material'].lower()
    
    # Defaults
    weather = "All-Season"
    waterproof = False
    warmth = 2

    # Logic Rules
    if 'wool' in material or 'sweater' in category:
        weather = "Winter"
        warmth = 4
    elif 'nylon' in material or 'jacket' in category:
        weather = "Rainy"
        waterproof = True
        warmth = 3
    elif 'shorts' in category or 'cotton' in material:
        weather = "Summer"
        warmth = 1

    return pd.Series([weather, waterproof, warmth])

# Apply the rules to create new columns
df_raw[['weather_suitability', 'is_waterproof', 'warmth_level']] = df_raw.apply(add_travel_tags, axis=1)
print("Travel Tags Added:")
print(df_raw[['category', 'weather_suitability', 'is_waterproof', 'warmth_level']], "\n")

# 3. Push to SQLite Database
conn = sqlite3.connect('wardrobe.db')
# We use if_exists='append' so we don't overwrite the table structure we made in Phase 1
df_raw.to_sql('Clothing_Items', conn, if_exists='append', index=False)
conn.close()

print("Successfully injected enriched data into the database!")