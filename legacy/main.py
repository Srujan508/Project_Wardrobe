from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import torch
import torch.nn.functional as F
import os
import shutil
from models import StyleExtractor, CompatibilityEngine

app = FastAPI(title="AI Wardrobe Stylist")

# Mount folders so the browser can see the HTML and the uploaded images
app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")

# --- 1. WAKE UP THE AI ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Loading AI Stylist on {device}...")

extractor = StyleExtractor().to(device)
engine = CompatibilityEngine().to(device)

try:
    engine.load_state_dict(torch.load("wardrobe_model_final.pth", map_location=device, weights_only=True))
    engine.eval()
    extractor.eval()
    print("AI Stylist is online and ready!")
except FileNotFoundError:
    print("WARNING: wardrobe_model_final.pth not found!")

# --- 2. UPLOAD ENDPOINT ---
@app.post("/upload-item")
async def upload_item(category: str = Form(...), file: UploadFile = File(...)):
    if category not in ["tops", "bottoms"]:
        raise HTTPException(status_code=400, detail="Category must be tops or bottoms")
    
    # Save the file to the correct folder
    file_path = f"uploads/{category}/{file.filename}"
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    return {"message": f"Successfully added {file.filename} to your closet!"}

# --- 3. THE STYLIST ENDPOINT ---
@app.get("/get-suggestion")
async def get_suggestion():
    tops = [os.path.join("uploads/tops", f) for f in os.listdir("uploads/tops") if f.endswith(('.png', '.jpg', '.jpeg'))]
    bottoms = [os.path.join("uploads/bottoms", f) for f in os.listdir("uploads/bottoms") if f.endswith(('.png', '.jpg', '.jpeg'))]
    
    if not tops or not bottoms:
        raise HTTPException(status_code=400, detail="You need at least one top and one bottom in your closet!")

    best_score = -1.0
    best_outfit = {"top": None, "bottom": None}

    print(f"Analyzing {len(tops)} tops and {len(bottoms)} bottoms...")

    with torch.no_grad():
        # The AI tries on every single combination in your closet
        for top_path in tops:
            vec_top = extractor(top_path).to(device)
            out_top = engine.process_one_item(vec_top.unsqueeze(0))
            
            for bottom_path in bottoms:
                vec_bottom = extractor(bottom_path).to(device)
                out_bottom = engine.process_one_item(vec_bottom.unsqueeze(0))
                
                similarity = F.cosine_similarity(out_top, out_bottom).item()
                score = ((similarity + 1) / 2) * 100
                
                # Keep track of the absolute best match
                if score > best_score:
                    best_score = score
                    best_outfit = {"top": top_path, "bottom": bottom_path}

    return {
        "top_image": "/" + best_outfit["top"],
        "bottom_image": "/" + best_outfit["bottom"],
        "score": round(best_score, 1),
        "message": "Here is the ultimate outfit combination from your closet."
    }

# --- 4. MY WARDROBE ENDPOINT ---
@app.get("/get-wardrobe")
async def get_wardrobe():
    try:
        # Get all valid images in the folders
        tops = ["/uploads/tops/" + f for f in os.listdir("uploads/tops") if f.endswith(('.png', '.jpg', '.jpeg', '.webp'))]
        bottoms = ["/uploads/bottoms/" + f for f in os.listdir("uploads/bottoms") if f.endswith(('.png', '.jpg', '.jpeg', '.webp'))]
        return {"tops": tops, "bottoms": bottoms}
    except Exception as e:
        return {"tops": [], "bottoms": []}

# --- 5. ROOT ENDPOINT ---
# Send users to the frontend when they visit the root URL
@app.get("/")
async def serve_frontend():
    return FileResponse("static/index.html")