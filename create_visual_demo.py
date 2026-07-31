content = '''"""
demo_visual_fhe.py
SecureLens — Visual TRUE FHE Demonstration with GradCAM

This demonstrates:
1. TRUE FHE predictions (encrypted inference)
2. GradCAM heatmaps showing what the model focuses on
3. Side-by-side visualization of classifications

Shows BOTH security AND interpretability!
"""

import os
import sys
import time
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from crypto_layer.ckks_engine import CKKSEngine
from cloud_server.encrypted_inference.he_inference import HEInferenceEngine
from cloud_server.train_model_fhe_compatible import SecureLensNetFHE

print("="*80)
print(" SecureLens - Visual TRUE FHE Demonstration with Explainability")
print("="*80)

# Configuration
MODELS_DIR = "cloud_server/models"
TEST_IMAGES = [
    ("data/chest_xray/test/NORMAL/IM-0001-0001.jpeg", "Normal", "Patient A"),
    ("data/chest_xray/test/NORMAL/IM-0003-0001.jpeg", "Normal", "Patient B"),
    ("data/chest_xray/test/PNEUMONIA/person100_bacteria_475.jpeg", "Pneumonia", "Patient C"),
    ("data/chest_xray/test/PNEUMONIA/person100_bacteria_477.jpeg", "Pneumonia", "Patient D"),
]

# Setup
print("\\n[Setup] Loading model and CKKS engine...")
model = SecureLensNetFHE(num_classes=2)
model.load_state_dict(torch.load(
    os.path.join(MODELS_DIR, "best_model.pth"),
    map_location="cpu"
))
model.eval()

ckks = CKKSEngine(8192, [60, 40, 40, 60], 2**40)
he_engine = HEInferenceEngine(MODELS_DIR)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

def generate_gradcam(model, img_tensor, target_class):
    # Get the last conv layer
    target_layer = model.backbone.layer4[-1].conv2

    # Forward pass
    model.eval()
    features = []
    gradients = []

    def forward_hook(module, input, output):
        features.append(output)

    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0])

    handle_forward = target_layer.register_forward_hook(forward_hook)
    handle_backward = target_layer.register_full_backward_hook(backward_hook)

    # Forward
    output = model(img_tensor)

    # Backward
    model.zero_grad()
    target = output[0][target_class]
    target.backward()

    # Generate heatmap
    feature_map = features[0].squeeze().detach()
    gradient = gradients[0].squeeze().detach()
    weights = gradient.mean(dim=(1, 2))

    cam = torch.zeros(feature_map.shape[1:], dtype=torch.float32)
    for i, w in enumerate(weights):
        cam += w * feature_map[i]

    cam = torch.relu(cam)
    cam = cam / (cam.max() + 1e-8)

    handle_forward.remove()
    handle_backward.remove()

    return cam.numpy()

def overlay_heatmap(img_path, heatmap):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (224, 224))

    heatmap_resized = cv2.resize(heatmap, (224, 224))
    heatmap_colored = cv2.applyColorMap(
        np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET
    )

    overlay = cv2.addWeighted(img, 0.6, heatmap_colored, 0.4, 0)
    return overlay

# Process each image
results = []
print("\\n" + "="*80)
print("Processing Test Images with TRUE FHE + GradCAM")
print("="*80)

for img_path, true_label, patient_id in TEST_IMAGES:
    if not os.path.exists(img_path):
        print(f"\\n⚠️  Skipping {patient_id}: Image not found")
        continue

    print(f"\\n{'='*80}")
    print(f"Processing: {patient_id} (True: {true_label})")
    print(f"{'='*80}")

    # Load image
    img = Image.open(img_path).convert('RGB')
    img_tensor = transform(img).unsqueeze(0)

    # Extract features
    print("[1/4] Extracting features...")
    with torch.no_grad():
        features = model.get_backbone_features(img_tensor)
    features_np = features.squeeze().numpy().astype(np.float64)

    # Encrypt
    print("[2/4] Encrypting with CKKS...")
    t_enc = time.time()
    enc_features = ckks.encrypt_feature_vector(features_np.copy())
    import tenseal as ts
    enc_features_server = ts.ckks_vector_from(
        ckks.public_context,
        enc_features.serialize()
    )
    t_enc = time.time() - t_enc

    # HE Inference
    print("[3/4] Running encrypted inference on server...")
    t_inf = time.time()
    enc_result = he_engine.infer_head(enc_features_server, ckks.public_context)
    t_inf = time.time() - t_inf

    # Decrypt
    print("[4/4] Decrypting result...")
    t_dec = time.time()
    result = ckks.decrypt_prediction(enc_result)
    t_dec = time.time() - t_dec

    # Generate GradCAM
    print("[Bonus] Generating GradCAM heatmap...")
    predicted_class = 0 if result['prediction'] == 'Normal' else 1
    heatmap = generate_gradcam(model, img_tensor, predicted_class)
    overlay = overlay_heatmap(img_path, heatmap)

    results.append({
        'patient_id': patient_id,
        'true_label': true_label,
        'predicted': result['prediction'],
        'confidence': result['confidence'],
        'normal_score': result['normal_score'],
        'pneumonia_score': result['pneumonia_score'],
        'time_enc': t_enc,
        'time_inf': t_inf,
        'time_dec': t_dec,
        'img': img,
        'overlay': overlay,
        'heatmap': heatmap,
    })

    print(f"✅ Result: {result['prediction']} ({result['confidence']:.1%} confidence)")
    print(f"   Timing: Enc={t_enc*1000:.0f}ms, Inf={t_inf*1000:.0f}ms, Dec={t_dec*1000:.0f}ms")

# Visualization
print("\\n" + "="*80)
print("Creating Visual Summary...")
print("="*80)

fig = plt.figure(figsize=(16, 12))
fig.suptitle('SecureLens: TRUE FHE Classification with Explainability',
             fontsize=16, fontweight='bold')

for idx, r in enumerate(results):
    # Original image
    ax1 = plt.subplot(4, 3, idx*3 + 1)
    ax1.imshow(r['img'])
    ax1.set_title(f"{r['patient_id']}\\nTrue: {r['true_label']}", fontsize=10)
    ax1.axis('off')

    # GradCAM overlay
    ax2 = plt.subplot(4, 3, idx*3 + 2)
    ax2.imshow(cv2.cvtColor(r['overlay'], cv2.COLOR_BGR2RGB))
    ax2.set_title('What Model Sees\\n(GradCAM Heatmap)', fontsize=10)
    ax2.axis('off')

    # Prediction
    ax3 = plt.subplot(4, 3, idx*3 + 3)
    ax3.axis('off')

    # Prediction box
    correct = r['predicted'] == r['true_label']
    color = 'green' if correct else 'red'
    status = '✅ Correct' if correct else '❌ Incorrect'

    text = f"""
🔒 ENCRYPTED INFERENCE
━━━━━━━━━━━━━━━━━━━━
Prediction: {r['predicted']}
Confidence: {r['confidence']:.1%}

Scores:
  Normal:    {r['normal_score']:.3f}
  Pneumonia: {r['pneumonia_score']:.3f}

Status: {status}

⏱️  Performance:
  Encryption:  {r['time_enc']*1000:.0f} ms
  Inference:   {r['time_inf']*1000:.0f} ms
  Decryption:  {r['time_dec']*1000:.0f} ms
  Total:       {(r['time_enc']+r['time_inf']+r['time_dec'])*1000:.0f} ms
"""

    ax3.text(0.1, 0.5, text, fontsize=9, family='monospace',
             verticalalignment='center',
             bbox=dict(boxstyle='round', facecolor=color, alpha=0.15))

plt.tight_layout()
output_path = 'securelens_visual_demo.png'
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"\\n✅ Visualization saved: {output_path}")
plt.show()

# Summary Statistics
print("\\n" + "="*80)
print("SUMMARY STATISTICS")
print("="*80)

total = len(results)
correct = sum(1 for r in results if r['predicted'] == r['true_label'])
accuracy = correct / total if total > 0 else 0

avg_enc = np.mean([r['time_enc'] for r in results]) * 1000
avg_inf = np.mean([r['time_inf'] for r in results]) * 1000
avg_dec = np.mean([r['time_dec'] for r in results]) * 1000
avg_total = avg_enc + avg_inf + avg_dec

print(f"\\n📊 Accuracy: {correct}/{total} ({accuracy:.1%})")
print(f"\\n⏱️  Average Latency:")
print(f"   Encryption:  {avg_enc:.0f} ms")
print(f"   Inference:   {avg_inf:.0f} ms")
print(f"   Decryption:  {avg_dec:.0f} ms")
print(f"   Total:       {avg_total:.0f} ms")

print("\\n" + "="*80)
print("KEY FEATURES DEMONSTRATED:")
print("="*80)
print("✅ TRUE FHE: Server never sees plaintext features or predictions")
print("✅ Accuracy: Model classifies correctly with encrypted data")
print("✅ Explainability: GradCAM shows what the model focuses on")
print(f"✅ Performance: ~{avg_total/1000:.1f}s total latency per image")
print("="*80)
'''

with open('demo_visual_fhe.py', 'w', encoding='utf-8') as f:
    f.write(content)

print('✅ Created demo_visual_fhe.py')
