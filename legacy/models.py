import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

# --- Concept 1: The Extractor ---
class StyleExtractor(nn.Module):
    def __init__(self):
        super(StyleExtractor, self).__init__()
        # Load the base model
        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.extractor = nn.Sequential(*list(resnet.children())[:-1])
        
        # --- THE UPGRADE: UNFREEZE THE LAST LAYER ---
        # We lock the early layers (basic shapes/edges)
        # But we unlock 'layer4' so it learns advanced fashion textures
        for name, param in self.extractor.named_parameters():
            if "layer4" in name:
                param.requires_grad = True 
            else:
                param.requires_grad = False
                
        self.extractor.eval()
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def forward(self, image_path):
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.transform(image).unsqueeze(0) 
        
        device = next(self.extractor.parameters()).device
        image_tensor = image_tensor.to(device)
        
        # REMOVE the torch.no_grad() block here so the ResNet can actually learn!
        feature_vector = self.extractor(image_tensor)
        return feature_vector.squeeze()
# --- Concept 2: The Siamese Engine ---
class CompatibilityEngine(nn.Module):
    def __init__(self, input_size=2048, hidden_size=512, embedding_size=128):
        super(CompatibilityEngine, self).__init__()
        # Compress the 2048 dimensions into 128 "fashion" dimensions
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, embedding_size)

    def process_one_item(self, x):
        """Passes a single vector through the network."""
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        # Normalize to a length of 1 so we only measure angles
        return F.normalize(x, p=2, dim=1)

    def forward(self, anchor, positive, negative):
        """Processes the Triplet simultaneously."""
        out_a = self.process_one_item(anchor)
        out_p = self.process_one_item(positive)
        out_n = self.process_one_item(negative)
        return out_a, out_p, out_n