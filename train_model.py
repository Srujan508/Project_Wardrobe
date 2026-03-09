import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import pandas as pd
import os
from models import StyleExtractor, CompatibilityEngine

# --- CONFIGURATION ---
TRAINING_PHASE = 'polyvore' # Change to 'polyvore' for the second run

if TRAINING_PHASE == 'deepfashion':
    CSV_FILE = 'deepfashion_triplets_fixed.csv'
    IMAGE_DIR = 'test_images' # Update path!
    EPOCHS = 6
else:
    CSV_FILE = 'polyvore_triplets_fixed.csv' 
    IMAGE_DIR = 'polyvore_images/archive/Re-PolyVore/Re-PolyVore/Re-PolyVore/'             # Update path!
    EPOCHS = 20

BATCH_SIZE = 32

class TripletWardrobeDataset(Dataset):
    def __init__(self, csv_file, image_dir):
        self.data = pd.read_csv(csv_file)
        self.image_dir = image_dir
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_a = Image.open(os.path.join(self.image_dir, row['anchor_image'])).convert('RGB')
        img_p = Image.open(os.path.join(self.image_dir, row['positive_image'])).convert('RGB')
        img_n = Image.open(os.path.join(self.image_dir, row['negative_image'])).convert('RGB')
        return self.transform(img_a), self.transform(img_p), self.transform(img_n)

def train_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Booting up on: {device} | Phase: {TRAINING_PHASE.upper()}")

    extractor = StyleExtractor().to(device)
    engine = CompatibilityEngine().to(device)
    
    if TRAINING_PHASE == 'polyvore' and os.path.exists("wardrobe_model_base.pth"):
        engine.load_state_dict(torch.load("wardrobe_model_base.pth"))
        print("Loaded DeepFashion base knowledge.")

    optimizer = optim.Adam(list(engine.parameters()) + list(extractor.parameters()), lr=0.0001)
    loss_function = nn.TripletMarginLoss(margin=1.0, p=2)

    dataset = TripletWardrobeDataset(CSV_FILE, IMAGE_DIR)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

    engine.train()
    
    for epoch in range(EPOCHS):
        total_loss = 0
        for batch_idx, (img_a, img_p, img_n) in enumerate(dataloader):
            img_a, img_p, img_n = img_a.to(device), img_p.to(device), img_n.to(device)
            
            with torch.no_grad():
                vec_a = extractor.extractor(img_a).squeeze()
                vec_p = extractor.extractor(img_p).squeeze()
                vec_n = extractor.extractor(img_n).squeeze()

            optimizer.zero_grad()
            out_a, out_p, out_n = engine(vec_a, vec_p, vec_n)
            loss = loss_function(out_a, out_p, out_n)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            if batch_idx % 50 == 0:
                print(f"Epoch {epoch+1} | Batch {batch_idx}/{len(dataloader)} | Loss: {loss.item():.4f}")

        print(f"--- Epoch {epoch+1} Complete | Average Loss: {total_loss/len(dataloader):.4f} ---")

    save_name = "wardrobe_model_base.pth" if TRAINING_PHASE == 'deepfashion' else "wardrobe_model_final.pth"
    torch.save(engine.state_dict(), save_name)
    print(f"Training Complete! Saved as {save_name}")

if __name__ == "__main__":
    train_model()