import torch
import torch.nn.functional as F
from models import StyleExtractor, CompatibilityEngine

# 1. Initialize the Models
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Waking up AI on {device}...")

extractor = StyleExtractor().to(device)
engine = CompatibilityEngine().to(device)

# 2. Load Your Trained Brain
try:
    # weights_only=True is a security best practice in modern PyTorch
    engine.load_state_dict(torch.load("wardrobe_model_base.pth", map_location=device, weights_only=True))
    engine.eval() # Crucial: Tells PyTorch we are testing, not training
    print("Brain loaded successfully!\n")
except FileNotFoundError:
    print("Error: wardrobe_model_base.pth not found. Make sure it's in this folder.")
    exit()

def get_outfit_score(top_path, bottom_path):
    print(f"Analyzing:")
    print(f" 1. {top_path}")
    print(f" 2. {bottom_path}...")
    
    # Extract the 2048-dimensional style vectors using ResNet50
    vec1 = extractor(top_path).to(device)
    vec2 = extractor(bottom_path).to(device)
    
    with torch.no_grad(): # Don't track gradients during testing to save memory
        # Pass vectors through your trained Siamese Engine
        out1 = engine.process_one_item(vec1.unsqueeze(0))
        out2 = engine.process_one_item(vec2.unsqueeze(0))
        
        # Calculate Cosine Similarity (-1 to 1)
        similarity = F.cosine_similarity(out1, out2)
        
        # Convert the math to a standard 0 to 100 score
        score = ((similarity.item() + 1) / 2) * 100
        
    return score

if __name__ == "__main__":
    # --- CHANGE THESE PATHS ---
    # Pick two actual images from your DeepFashion test folder
    top_image = "test_images/testtop.jpg" 
    bottom_image = "test_images/testbottom.jpg"
    
    try:
        match_score = get_outfit_score(top_image, bottom_image)
        
        print(f"\n--- AI VERDICT ---")
        print(f"Compatibility Score: {match_score:.1f} / 100")
        
        if match_score >= 75:
            print("Verdict: The AI highly approves. Great match!")
        elif match_score >= 50:
            print("Verdict: It's okay, but you could probably do better.")
        else:
            print("Verdict: The AI says these clash. Do not pack these together!")
            
    except Exception as e:
        print(f"\nOops! Something went wrong: {e}")
        print("Did you remember to update the image paths to real files on your computer?")