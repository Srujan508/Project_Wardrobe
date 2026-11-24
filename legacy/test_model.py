import torch
import torch.nn.functional as F
import os
from models import StyleExtractor, CompatibilityEngine

# 1. Wake up the AI
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Waking up AI on {device}...")

extractor = StyleExtractor().to(device)
engine = CompatibilityEngine().to(device)

# 2. Load your newly trained Polyvore brain
try:
    engine.load_state_dict(torch.load("wardrobe_model_final.pth", map_location=device, weights_only=True))
    engine.eval() 
    extractor.eval()
    print("Brain loaded successfully!\n")
except FileNotFoundError:
    print("Error: wardrobe_model_final.pth not found. Did you finish training?")
    exit()

def get_outfit_score(top_path, bottom_path):
    print(f"Analyzing Outfit...")
    print(f" Top: {top_path}")
    print(f" Bottom: {bottom_path}")
    
    # Check if files actually exist before doing math
    if not os.path.exists(top_path) or not os.path.exists(bottom_path):
        print("\nERROR: One of those image paths doesn't exist. Check your spelling!")
        return

    # Extract the math vectors
    vec_top = extractor(top_path).to(device)
    vec_bottom = extractor(bottom_path).to(device)
    
    with torch.no_grad(): 
        # Pass vectors through the Siamese Engine
        out_top = engine.process_one_item(vec_top.unsqueeze(0))
        out_bottom = engine.process_one_item(vec_bottom.unsqueeze(0))
        
        # Calculate Cosine Similarity
        similarity = F.cosine_similarity(out_top, out_bottom)
        
        # Convert the math (-1 to 1) to a standard 0 to 100 score
        score = ((similarity.item() + 1) / 2) * 100
        
    print(f"\n--- AI VERDICT ---")
    print(f"Compatibility Score: {score:.1f} / 100")
    
    if score >= 75:
        print("Verdict: The AI highly approves. Great match!")
    elif score >= 50:
        print("Verdict: It's okay, but you could probably do better.")
    else:
        print("Verdict: The AI says these clash. Do not wear these together!")

if __name__ == "__main__":
    # --- UPDATE THESE PATHS TO REAL IMAGES ON YOUR COMPUTER ---
    my_top = "/home/smirkmachine/Downloads/test_top.jpg" 
    my_bottom = "/home/smirkmachine/Downloads/test_top2.jpg"
    
    get_outfit_score(my_top, my_bottom)