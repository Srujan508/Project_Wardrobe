import sqlite3
import pandas as pd
import random

# Fetch items from our database
conn = sqlite3.connect('wardrobe.db')
df = pd.read_sql_query("SELECT * FROM Clothing_Items", conn)
conn.close()

triplets = []

# Simple logic to generate Triplets (In reality, you'd use Polyvore's outfit lists for this)
tops = df[df['category'].isin(['T-shirt', 'Sweater', 'Jacket'])]
bottoms = df[df['category'] == 'Shorts']

for _, top in tops.iterrows():
    # If the top is a Summer item, a Summer bottom is a POSITIVE match
    if top['weather_suitability'] == 'Summer':
        positive_match = bottoms[bottoms['weather_suitability'] == 'Summer'].sample(1).iloc[0]
        negative_match = df[df['weather_suitability'] == 'Winter'].sample(1).iloc[0] # Winter bottoms clash with summer tops
        
        triplets.append({
            'anchor_id': top['item_id'],
            'positive_id': positive_match['item_id'],
            'negative_id': negative_match['item_id']
        })

triplet_df = pd.DataFrame(triplets)
triplet_df.to_csv('data/processed/training_triplets.csv', index=False)

print("Triplet training data generated and saved to data/processed/training_triplets.csv")
print(triplet_df.head())