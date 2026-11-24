import torch
import torch.nn.functional as F
import pandas as pd
import os
from tqdm import tqdm
from models import StyleExtractor, CompatibilityEngine

# --- CONFIGURATION ---
CSV_FILE = 'polyvore_triplets_fixed.csv'
IMAGE_DIR = 'polyvore_images/archive/Re-PolyVore/Re-PolyVore/Re-PolyVore/'
TEST_SAMPLE_SIZE = 500  # We will test 500 random outfits to get a solid accuracy rating

def evaluate_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Booting up Evaluation Engine on {device}...")

    # 1. Load the Models
    extractor = StyleExtractor().to(device)
    engine = CompatibilityEngine().to(device)

    try:
        engine.load_state_dict(torch.load("wardrobe_model_final.pth", map_location=device, weights_only=True))
        engine.eval()
        extractor.eval()
        print("Model 'wardrobe_model_final.pth' loaded successfully!\n")
    except FileNotFoundError:
        print("Error: Could not find wardrobe_model_final.pth")
        return

    # 2. Load the Data
    print(f"Loading {TEST_SAMPLE_SIZE} random outfits for the exam...")
    df = pd.read_csv(CSV_FILE)
    
    # Shuffle and pick a sample so we don't test the exact same ones every time
    test_df = df.sample(n=TEST_SAMPLE_SIZE, random_state=42)

    correct_predictions = 0
    valid_tests = 0

    # 3. The Testing Loop
    print("Running the AI Exam...")
    for index, row in tqdm(test_df.iterrows(), total=TEST_SAMPLE_SIZE):
        anchor_path = os.path.join(IMAGE_DIR, row['anchor_image'])
        positive_path = os.path.join(IMAGE_DIR, row['positive_image'])
        negative_path = os.path.join(IMAGE_DIR, row['negative_image'])

        # Double check files exist to prevent crashes
        if not (os.path.exists(anchor_path) and os.path.exists(positive_path) and os.path.exists(negative_path)):
            continue
            
        valid_tests += 1

        try:
            # Extract vectors
            vec_a = extractor(anchor_path).to(device)
            vec_p = extractor(positive_path).to(device)
            vec_n = extractor(negative_path).to(device)

            with torch.no_grad():
                out_a = engine.process_one_item(vec_a.unsqueeze(0))
                out_p = engine.process_one_item(vec_p.unsqueeze(0))
                out_n = engine.process_one_item(vec_n.unsqueeze(0))

                # Calculate similarities
                sim_positive = F.cosine_similarity(out_a, out_p).item()
                sim_negative = F.cosine_similarity(out_a, out_n).item()

                # Did the AI score the real outfit higher than the fake outfit?
                if sim_positive > sim_negative:
                    correct_predictions += 1
        except Exception as e:
            continue

    # 4. Calculate Final Score
    if valid_tests == 0:
        print("\nError: No valid images found to test. Check your IMAGE_DIR path.")
        return

    accuracy = (correct_predictions / valid_tests) * 100

    print("\n" + "="*40)
    print(" 📊 FINAL AI EXAM RESULTS 📊 ")
    print("="*40)
    print(f"Total Outfits Tested: {valid_tests}")
    print(f"Correct Predictions:  {correct_predictions}")
    print(f"Model Accuracy:       {accuracy:.2f}%")
    print("-" * 40)
    
    # The Verdict
    if accuracy >= 85.0:
        print("Verdict: EXCELLENT. Your AI is basically a professional stylist.")
    elif accuracy >= 70.0:
        print("Verdict: GOOD. The AI understands fashion rules and makes solid recommendations.")
    elif accuracy >= 60.0:
        print("Verdict: OKAY. It gets it right most of the time, but might make a few weird choices.")
    else:
        print("Verdict: POOR. The AI is practically guessing. We need to train it longer.")
    print("="*40 + "\n")

if __name__ == "__main__":
    evaluate_model()