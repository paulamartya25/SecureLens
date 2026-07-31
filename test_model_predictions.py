import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from cloud_server.train_model_fhe_compatible import SecureLensNet
import os

print("="*60)
print("Testing Model Predictions on Real Images")
print("="*60)

# Load model
model = SecureLensNet(num_classes=2)
model.load_state_dict(torch.load('cloud_server/models/best_model.pth', map_location='cpu'))
model.eval()
print("✅ Model loaded successfully\n")

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

def test_image(img_path, expected_class):
    if not os.path.exists(img_path):
        print(f"⚠️  Skipping {img_path} (not found)")
        return None

    img = Image.open(img_path).convert('RGB')
    img_tensor = transform(img).unsqueeze(0)

    with torch.no_grad():
        output = model(img_tensor)
        probs = torch.softmax(output, dim=1)[0]

    predicted_class = "Normal" if probs[0] > probs[1] else "Pneumonia"
    confidence = max(probs[0], probs[1]).item()

    status = "✅" if predicted_class == expected_class else "❌"
    print(f"{status} {os.path.basename(img_path)}")
    print(f"   Expected: {expected_class:10s} | Predicted: {predicted_class:10s} ({confidence:.1%})")
    print(f"   Probs: Normal={probs[0]:.4f}, Pneumonia={probs[1]:.4f}")
    print(f"   Logits: [{output[0,0]:.4f}, {output[0,1]:.4f}]\n")

    return predicted_class == expected_class

# Test Normal images
print("="*60)
print("Testing NORMAL images")
print("="*60 + "\n")
normal_images = [
    'data/chest_xray/test/NORMAL/IM-0001-0001.jpeg',
    'data/chest_xray/test/NORMAL/IM-0003-0001.jpeg',
    'data/chest_xray/test/NORMAL/IM-0005-0001.jpeg',
]

normal_results = [test_image(img, "Normal") for img in normal_images]
normal_correct = sum(1 for r in normal_results if r is True)

# Test Pneumonia images
print("="*60)
print("Testing PNEUMONIA images")
print("="*60 + "\n")
pneumonia_images = [
    'data/chest_xray/test/PNEUMONIA/person100_bacteria_475.jpeg',
    'data/chest_xray/test/PNEUMONIA/person100_bacteria_477.jpeg',
    'data/chest_xray/test/PNEUMONIA/person101_bacteria_483.jpeg',
]

pneumonia_results = [test_image(img, "Pneumonia") for img in pneumonia_images]
pneumonia_correct = sum(1 for r in pneumonia_results if r is True)

print("="*60)
print(f"Results: Normal {normal_correct}/3, Pneumonia {pneumonia_correct}/3")
print("="*60)

if normal_correct == 0:
    print("\n❌ All Normal images predicted as Pneumonia - LABELS MAY BE SWAPPED!")
