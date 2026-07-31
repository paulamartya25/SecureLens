import sys
import json

print("="*60)
print("SecureLens System Diagnostics")
print("="*60)

# 1. Check NumPy version
print("\n[1] NumPy Version Check")
try:
    import numpy as np
    print(f"   NumPy version: {np.__version__}")
    if np.__version__.startswith('2.'):
        print("   ❌ ERROR: NumPy 2.x detected")
        print("   Required: NumPy 1.24+ but <2.0")
        print("   Fix: pip install \"numpy>=1.24,<2.0\"")
    else:
        print("   ✅ NumPy 1.x (compatible)")
except Exception as e:
    print(f"   ❌ NumPy import error: {e}")

# 2. Check OpenCV
print("\n[2] OpenCV Check")
try:
    import cv2
    print(f"   OpenCV version: {cv2.__version__}")
    print("   ✅ OpenCV available")
except Exception as e:
    print(f"   ❌ OpenCV error: {e}")

# 3. Check model weights
print("\n[3] Model Weights Check")
try:
    import torch
    model_path = 'cloud_server/models/best_model.pth'
    state_dict = torch.load(model_path, map_location='cpu')

    if 'fc.weight' in state_dict:
        fc_weights = state_dict['fc.weight']
        print(f"   FC layer shape: {fc_weights.shape}")
        print(f"   FC weight range: [{fc_weights.min():.4f}, {fc_weights.max():.4f}]")
        print(f"   FC weight std: {fc_weights.std():.4f}")

        if fc_weights.std() < 0.001:
            print("   ❌ WARNING: Weights too small (nearly zero)")
        else:
            print("   ✅ Weights look reasonable")
    else:
        print("   ℹ️  No 'fc.weight' key found")
        print(f"   Available keys: {list(state_dict.keys())[:5]}...")
except Exception as e:
    print(f"   ❌ Model loading error: {e}")

# 4. Check JSON weights
print("\n[4] JSON Weight Files Check")
try:
    with open('cloud_server/models/linear_weights.json') as f:
        lw = json.load(f)
    W = lw['W']
    b = lw['b']
    print(f"   Linear weights shape: {len(W)}x{len(W[0])}")
    print(f"   Bias shape: {len(b)}")

    import numpy as np
    W_arr = np.array(W)
    print(f"   Weight range: [{W_arr.min():.4f}, {W_arr.max():.4f}]")
    print(f"   Weight std: {W_arr.std():.4f}")

    if W_arr.std() < 0.001:
        print("   ❌ WARNING: JSON weights nearly zero")
    else:
        print("   ✅ JSON weights look reasonable")
except Exception as e:
    print(f"   ❌ JSON weights error: {e}")

# 5. Test model prediction
print("\n[5] Model Prediction Test")
try:
    import torch
    from cloud_server.train_model import SecureLensNet

    model = SecureLensNet(2)
    model.load_state_dict(torch.load('cloud_server/models/best_model.pth', map_location='cpu'))
    model.eval()

    # Test with random input
    test_input = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        output = model(test_input)
        probs = torch.softmax(output, dim=1)

    print(f"   Output logits: {output[0].tolist()}")
    print(f"   Probabilities: {probs[0].tolist()}")

    max_prob = probs.max().item()
    if 0.45 < max_prob < 0.55:
        print("   ❌ WARNING: Model outputs ~50% (random predictions)")
        print("   → Model may need retraining")
    else:
        print(f"   ✅ Model produces confident predictions ({max_prob:.1%})")
except Exception as e:
    print(f"   ❌ Prediction test error: {e}")

print("\n" + "="*60)
print("Diagnosis Complete")
print("="*60)
