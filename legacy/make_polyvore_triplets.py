import json
import pandas as pd
import random

with open('polyvore/train_no_dup.json', 'r') as f:
    outfits = json.load(f)

def get_image_name(item_dict, set_id):
    return f"{set_id}/{item_dict['index']}.jpg"

triplets = []
for outfit in outfits:
    items = outfit['items']
    
    if len(items) >= 2:
        anchor_item, positive_item = random.sample(items, 2)
        
        random_outfit = random.choice(outfits)
        while random_outfit['set_id'] == outfit['set_id']:
            random_outfit = random.choice(outfits)
            
        negative_item = random.choice(random_outfit['items'])
        
        triplets.append({
            'anchor_image': get_image_name(anchor_item, outfit['set_id']),
            'positive_image': get_image_name(positive_item, outfit['set_id']),
            'negative_image': get_image_name(negative_item, random_outfit['set_id'])
        })

triplet_df = pd.DataFrame(triplets)
triplet_df.to_csv('polyvore_triplets.csv', index=False)
print(f"Generated {len(triplet_df)} Polyvore triplets!")