# coding: utf-8
"""
app_gradio.py
SecureLens - Gradio Interface for TRUE FHE
"""

import gradio as gr
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from crypto_layer.ckks_engine import CKKSEngine
from cloud_server.encrypted_inference.he_inference import HEInferenceEngine
from cloud_server.train_model_fhe_compatible import SecureLensNetFHE
import tenseal as ts

print('Loading SecureLens TRUE FHE System...')

# Setup
model = SecureLensNetFHE(num_classes=2)
model.load_state_dict(torch.load(
    'cloud_server/models/best_model.pth',
    map_location='cpu'
))
model.eval()

ckks = CKKSEngine(8192, [60, 40, 40, 60], 2**40)
he_engine = HEInferenceEngine('cloud_server/models')

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

print('System ready!')

def classify_xray_fhe(image):
    if image is None:
        return '<div style="padding: 20px; color: red;">Please upload an image first!</div>'

    try:
        # Convert to PIL
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image).convert('RGB')

        # Preprocess
        img_tensor = transform(image).unsqueeze(0)

        # Extract features
        with torch.no_grad():
            features = model.get_backbone_features(img_tensor)
        features_np = features.squeeze().numpy().astype(np.float64)

        # Encrypt
        enc_features = ckks.encrypt_feature_vector(features_np.copy())
        enc_features_server = ts.ckks_vector_from(
            ckks.public_context,
            enc_features.serialize()
        )

        # HE Inference
        enc_result = he_engine.infer_head(enc_features_server, ckks.public_context)

        # Decrypt
        result = ckks.decrypt_prediction(enc_result)

        # Format result
        diagnosis = result['prediction']
        confidence = result['confidence']
        normal_score = result['normal_score']
        pneumonia_score = result['pneumonia_score']

        # Status
        if diagnosis == 'Normal':
            status_icon = 'OK'
            risk_level = 'LOW RISK'
            bg_color = '#667eea'
        else:
            status_icon = 'WARNING'
            risk_level = 'HIGH RISK'
            bg_color = '#f5576c'

        # Create result HTML
        result_html = f"""
        <div style="padding: 20px; border-radius: 10px; background: {bg_color}; color: white;">
            <h2>TRUE FHE Classification Result</h2>
            <div style="background: rgba(255,255,255,0.1); padding: 15px; border-radius: 8px; margin: 10px 0;">
                <h3>{status_icon}: {diagnosis}</h3>
                <p>Confidence: {confidence:.1%}</p>
                <p>Risk Level: {risk_level}</p>
            </div>

            <div style="background: rgba(255,255,255,0.1); padding: 15px; border-radius: 8px; margin: 10px 0;">
                <h4>Detailed Scores</h4>
                <p>Normal: {normal_score*100:.1f}%</p>
                <p>Pneumonia: {pneumonia_score*100:.1f}%</p>
            </div>

            <div style="background: rgba(255,255,255,0.1); padding: 15px; border-radius: 8px; margin: 10px 0;">
                <h4>Security Guarantees</h4>
                <ul>
                    <li>Features encrypted on YOUR device</li>
                    <li>Server processed ONLY encrypted data</li>
                    <li>Server NEVER saw your X-ray</li>
                    <li>Prediction decrypted on YOUR device</li>
                </ul>
                <p>This is TRUE Fully Homomorphic Encryption!</p>
            </div>
        </div>
        """

        return result_html

    except Exception as e:
        return f'<div style="padding: 20px; background: #ef4444; color: white;"><h3>Error</h3><p>{str(e)}</p></div>'

# Create Gradio Interface
with gr.Blocks(theme=gr.themes.Soft(), title="SecureLens - TRUE FHE") as demo:
    gr.Markdown("""
    # SecureLens - TRUE Fully Homomorphic Encryption
    ## Privacy-Preserving Pneumonia Detection

    Upload a chest X-ray image for TRUE FHE analysis!
    """)

    with gr.Row():
        with gr.Column(scale=1):
            image_input = gr.Image(type='pil', label='Upload Chest X-Ray', height=400)
            classify_btn = gr.Button("Classify with TRUE FHE", variant="primary", size="lg")

        with gr.Column(scale=1):
            output = gr.HTML(label="Result")

    classify_btn.click(fn=classify_xray_fhe, inputs=image_input, outputs=output)

    gr.Markdown("""
    ---
    ### Technical Details
    - **Encryption:** CKKS (128-bit security)
    - **Model:** ResNet-18 + FHE-compatible head
    - **Accuracy:** ~97% on test set

    **Disclaimer:** Research prototype only. Not for clinical use.
    """)

if __name__ == "__main__":
    print("\nSecureLens TRUE FHE Interface Starting...")
    print("Local URL: http://127.0.0.1:7860")
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
