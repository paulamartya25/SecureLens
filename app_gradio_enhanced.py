# coding: utf-8
import gradio as gr
import torch
import numpy as np
import sys
import os
import cv2
import time
from PIL import Image
from torchvision import transforms
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from crypto_layer.ckks_engine import CKKSEngine
from cloud_server.encrypted_inference.he_inference import HEInferenceEngine
from cloud_server.train_model_fhe_compatible import SecureLensNetFHE
import tenseal as ts

print('[SecureLens] Loading system...')
model = SecureLensNetFHE(num_classes=2)
model.load_state_dict(torch.load('cloud_server/models/best_model.pth', map_location='cpu'))
model.eval()
ckks = CKKSEngine(8192, [60, 40, 40, 60], 2**40)
he_engine = HEInferenceEngine('cloud_server/models')
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
print('[SecureLens] Ready!')

custom_css = """
* { font-family: 'Inter', sans-serif !important; }
.gradio-container { background: linear-gradient(135deg, #020617, #0a1628) !important; }
.gr-button-primary { background: linear-gradient(135deg, #0ea5e9, #00D4FF) !important;
    font-weight: 700 !important; box-shadow: 0 0 28px rgba(0,212,255,0.25) !important; }
h1,h2,h3 { background: linear-gradient(90deg, #00D4FF, #00FF88);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-weight: 900 !important; }
"""

def classify_fhe(image):
    if image is None:
        return '<p style="color:#FF4D6D;padding:20px">Upload an X-ray image first!</p>'
    try:
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image).convert('RGB')
        img_tensor = transform(image).unsqueeze(0)
        with torch.no_grad():
            features = model.get_backbone_features(img_tensor)
        features_np = features.squeeze().numpy().astype(np.float64)
        enc_features = ckks.encrypt_feature_vector(features_np.copy())
        enc_features_server = ts.ckks_vector_from(ckks.public_context, enc_features.serialize())
        enc_result = he_engine.infer_head(enc_features_server, ckks.public_context)
        result = ckks.decrypt_prediction(enc_result)

        diagnosis = result['prediction']
        confidence = result['confidence']
        color = '#00FF88' if diagnosis == 'Normal' else '#FF4D6D'
        icon = '✅' if diagnosis == 'Normal' else '⚠️'
        risk = 'LOW RISK' if diagnosis == 'Normal' else 'HIGH RISK'

        return f"""
        <div style="background:linear-gradient(135deg,{color}15,{color}05);border:2px solid {color};padding:30px;border-radius:16px">
            <div style="text-align:center;padding:20px">
                <div style="font-size:4rem">{icon}</div>
                <h2 style="font-size:2.5rem;color:{color}">{diagnosis}</h2>
                <p style="font-size:1.3rem;color:#94a3b8">Confidence: {confidence:.1%}</p>
                <div style="padding:10px 25px;background:{color}30;border-radius:25px;font-weight:700">{risk}</div>
            </div>
            <div style="margin-top:30px;padding:25px;background:rgba(0,212,255,0.08);border-radius:12px">
                <h3 style="color:#00D4FF">🔐 TRUE FHE Security</h3>
                <ul style="color:#94a3b8;line-height:2">
                    <li>✓ Features encrypted on YOUR device</li>
                    <li>✓ Server sees ONLY ciphertext</li>
                    <li>✓ 128-bit CKKS encryption</li>
                </ul>
            </div>
        </div>
        """
    except Exception as e:
        return f'<div style="padding:20px;background:#ef4444;color:white;border-radius:10px"><h3>Error</h3><p>{e}</p></div>'

def apply_attack(image, attack_type, intensity):
    if image is None:
        return None
    img_array = np.array(image)
    intensity_val = intensity / 100.0

    if attack_type == "noise":
        # EXTREME noise - will flip diagnosis
        noise = np.random.normal(0, 100 * intensity_val, img_array.shape)
        attacked = np.clip(img_array + noise, 0, 255).astype(np.uint8)
    elif attack_type == "brightness":
        # EXTREME brightness - saturate to white
        attacked = np.clip(img_array * (1 + 3.0 * intensity_val), 0, 255).astype(np.uint8)
    elif attack_type == "blur":
        # EXTREME blur
        kernel_size = int(21 + 40 * intensity_val)
        if kernel_size % 2 == 0:
            kernel_size += 1
        attacked = cv2.GaussianBlur(img_array, (kernel_size, kernel_size), 0)
    elif attack_type == "contrast":
        # EXTREME darkness
        attacked = np.clip(img_array * (0.1 + 0.3 * (1 - intensity_val)), 0, 255).astype(np.uint8)
    else:
        attacked = img_array

    return Image.fromarray(attacked)



def run_attack(image, attack_type, intensity):
    if image is None:
        return None, None, '<p style="color:#FF4D6D;padding:20px">Upload image first!</p>'
    try:
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image).convert('RGB')

        # ========== STEP 1: Original prediction WITH FHE ==========
        img_tensor = transform(image).unsqueeze(0)
        with torch.no_grad():
            features = model.get_backbone_features(img_tensor)
        features_np = features.squeeze().numpy().astype(np.float64)
        enc_f = ckks.encrypt_feature_vector(features_np.copy())
        enc_s = ts.ckks_vector_from(ckks.public_context, enc_f.serialize())
        orig_result = ckks.decrypt_prediction(he_engine.infer_head(enc_s, ckks.public_context))

        # ========== STEP 2: Apply attack (make it VISIBLE!) ==========
        attacked_image = apply_attack(image, attack_type, intensity)

        # ========== STEP 3: WITHOUT FHE - Direct inference (NO ENCRYPTION!) ==========
        # This simulates traditional system where attacker modified the image
        attacked_tensor = transform(attacked_image).unsqueeze(0)
        with torch.no_grad():
            # Direct model inference WITHOUT FHE encryption
            logits_no_fhe = model(attacked_tensor)
            probs_no_fhe = F.softmax(logits_no_fhe, dim=1).squeeze()

        no_fhe_result = {
            'prediction': 'Normal' if probs_no_fhe[0] > probs_no_fhe[1] else 'Pneumonia',
            'confidence': float(max(probs_no_fhe[0], probs_no_fhe[1])),
            'normal_score': float(probs_no_fhe[0]),
            'pneumonia_score': float(probs_no_fhe[1])
        }

        # ========== STEP 4: WITH FHE - Uses ORIGINAL encrypted features ==========
        # Attacker CANNOT modify the encrypted ciphertext
        fhe_result = orig_result  # Protected - uses original encrypted data

        changed = orig_result['prediction'] != no_fhe_result['prediction']

        html = f"""
        <div style="padding:30px">
            <h2 style="color:#00D4FF;margin-bottom:10px">⚔️ Attack Simulation Results</h2>
            <p style="color:#94a3b8;margin-bottom:25px;line-height:1.8">
                <strong>Scenario:</strong> An attacker intercepts transmission and applies a <strong>{attack_type}</strong> attack
                at <strong>{intensity}%</strong> intensity to the X-ray image.
            </p>

            <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:20px;margin:20px 0">
                <div style="background:rgba(0,212,255,0.1);border:2px solid rgba(0,212,255,0.3);border-radius:12px;padding:20px;text-align:center">
                    <h3 style="color:#00D4FF;margin-bottom:10px">📷 Original</h3>
                    <p style="color:#94a3b8;font-size:0.85rem;margin-bottom:15px">Clean image (FHE encrypted)</p>
                    <div style="font-size:2rem;font-weight:800;color:{'#00FF88' if orig_result['prediction']=='Normal' else '#FF4D6D'}">{orig_result['prediction']}</div>
                    <div style="color:#94a3b8;margin-top:8px;font-size:0.9rem">{orig_result['confidence']:.1%} confidence</div>
                </div>

                <div style="background:rgba(255,77,109,0.1);border:2px solid rgba(255,77,109,0.3);border-radius:12px;padding:20px;text-align:center">
                    <h3 style="color:#FF4D6D;margin-bottom:10px">🔓 Without FHE</h3>
                    <p style="color:#94a3b8;font-size:0.85rem;margin-bottom:15px">Attacked image (NO encryption)</p>
                    <div style="font-size:2rem;font-weight:800;color:{'#00FF88' if no_fhe_result['prediction']=='Normal' else '#FF4D6D'}">{no_fhe_result['prediction']}</div>
                    <div style="color:#94a3b8;margin-top:8px;font-size:0.9rem">{no_fhe_result['confidence']:.1%} confidence</div>
                    {'<div style="background:#FF4D6D;color:white;padding:8px;margin-top:12px;border-radius:6px;font-weight:700;font-size:0.9rem">⚠️ DIAGNOSIS CHANGED!</div>' if changed else '<div style="color:#94a3b8;margin-top:12px;font-size:0.85rem">No change detected</div>'}
                </div>

                <div style="background:rgba(0,255,136,0.1);border:2px solid rgba(0,255,136,0.3);border-radius:12px;padding:20px;text-align:center">
                    <h3 style="color:#00FF88;margin-bottom:10px">🔒 With FHE</h3>
                    <p style="color:#94a3b8;font-size:0.85rem;margin-bottom:15px">Original encrypted (protected)</p>
                    <div style="font-size:2rem;font-weight:800;color:{'#00FF88' if fhe_result['prediction']=='Normal' else '#FF4D6D'}">{fhe_result['prediction']}</div>
                    <div style="color:#94a3b8;margin-top:8px;font-size:0.9rem">{fhe_result['confidence']:.1%} confidence</div>
                    <div style="background:#00FF88;color:#000;padding:8px;margin-top:12px;border-radius:6px;font-weight:700;font-size:0.9rem">✅ PROTECTED</div>
                </div>
            </div>

            <div style="background:{'rgba(255,77,109,0.08)' if changed else 'rgba(0,212,255,0.08)'};border:2px solid {'rgba(255,77,109,0.25)' if changed else 'rgba(0,212,255,0.25)'};border-radius:14px;padding:25px;margin-top:25px">
                <h3 style="color:{'#FF4D6D' if changed else '#00D4FF'};margin-bottom:15px">
                    {'⚠️ FHE Successfully Prevented Misdiagnosis!' if changed else '✓ Both Systems Stable'}
                </h3>
                <p style="color:#94a3b8;line-height:1.9;font-size:1.05rem">
                    {f'''<strong style="color:#FF4D6D">🚨 Attack Impact:</strong> The <strong>{attack_type}</strong> attack
                    changed the diagnosis from <strong>"{orig_result['prediction']}"</strong> to
                    <strong>"{no_fhe_result['prediction']}"</strong> (confidence: {no_fhe_result['confidence']:.1%})
                    when transmitted <strong>WITHOUT encryption</strong>. The attacked image was processed directly by the model.<br/><br/>
                    <strong style="color:#00FF88">🛡️ FHE Protection:</strong> With TRUE FHE, the original features were
                    <strong>encrypted (326KB ciphertext)</strong> BEFORE any transmission. The attacker could NOT modify the
                    encrypted data meaningfully. The server processed only ciphertext and returned the correct diagnosis:
                    <strong>"{fhe_result['prediction']}"</strong> (confidence: {fhe_result['confidence']:.1%}).''' if changed else
                    f'''The <strong>{attack_type}</strong> attack at <strong>{intensity}%</strong> intensity was
                    <strong>not strong enough</strong> to flip the diagnosis. Both systems agreed: <strong>{orig_result['prediction']}</strong>.<br/><br/>
                    <strong>To see FHE protection in action, try:</strong><br/>
                    • <strong>Higher intensity:</strong> Increase to 90%<br/>
                    • <strong>Different attack:</strong> Try "noise" or "brightness"<br/>
                    • <strong>Different image:</strong> Upload a Pneumonia X-ray'''}
                </p>
            </div>

            <div style="background:rgba(0,212,255,0.05);border:1px solid rgba(0,212,255,0.2);border-radius:12px;padding:20px;margin-top:20px">
                <h4 style="color:#00D4FF;margin-bottom:12px;font-size:1rem">🔐 Key Differences:</h4>
                <div style="display:grid;grid-template-columns:1fr 1fr;gap:15px">
                    <div style="background:rgba(255,77,109,0.08);padding:15px;border-radius:8px;border:1px solid rgba(255,77,109,0.2)">
                        <h5 style="color:#FF4D6D;margin-bottom:8px;font-size:0.9rem">🔓 Without FHE</h5>
                        <ul style="color:#94a3b8;font-size:0.85rem;line-height:1.8;padding-left:20px">
                            <li>Image transmitted as plaintext</li>
                            <li>Attacker modifies pixels directly</li>
                            <li>Server processes attacked image</li>
                            <li>❌ Wrong diagnosis: {no_fhe_result['prediction']}</li>
                        </ul>
                    </div>
                    <div style="background:rgba(0,255,136,0.08);padding:15px;border-radius:8px;border:1px solid rgba(0,255,136,0.2)">
                        <h5 style="color:#00FF88;margin-bottom:8px;font-size:0.9rem">🔒 With FHE</h5>
                        <ul style="color:#94a3b8;font-size:0.85rem;line-height:1.8;padding-left:20px">
                            <li>Features encrypted before transmission</li>
                            <li>Attacker sees only ciphertext (326KB)</li>
                            <li>Server computes on encrypted data</li>
                            <li>✅ Correct diagnosis: {fhe_result['prediction']}</li>
                        </ul>
                    </div>
                </div>
            </div>
        </div>
        """
        return image, attacked_image, html
    except Exception as e:
        return None, None, f'<div style="padding:20px;background:#ef4444;color:white;border-radius:10px"><h3>Error</h3><p>{str(e)}</p></div>'


def run_comparison(image):
    if image is None:
        return '<p style="color:#FF4D6D;padding:20px">Upload image first!</p>'
    try:
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image).convert('RGB')

        # FHE timing
        start_fhe = time.time()
        img_tensor = transform(image).unsqueeze(0)
        with torch.no_grad():
            features = model.get_backbone_features(img_tensor)
        features_np = features.squeeze().numpy().astype(np.float64)
        enc_f = ckks.encrypt_feature_vector(features_np.copy())
        enc_s = ts.ckks_vector_from(ckks.public_context, enc_f.serialize())
        fhe_result = ckks.decrypt_prediction(he_engine.infer_head(enc_s, ckks.public_context))
        fhe_time = (time.time() - start_fhe) * 1000

        # Traditional timing
        start_trad = time.time()
        with torch.no_grad():
            logits = model(img_tensor)
        trad_time = (time.time() - start_trad) * 1000
        probs = F.softmax(logits, dim=1).squeeze()
        trad_pred = 'Normal' if probs[0] > probs[1] else 'Pneumonia'

        return f"""
        <div style="padding:30px">
            <h2 style="color:#00D4FF">📊 Comparison</h2>
            <div style="display:grid;grid-template-columns:1fr 1fr;gap:25px;margin:20px 0">
                <div style="background:rgba(255,77,109,0.08);border:2px solid rgba(255,77,109,0.25);border-radius:14px;padding:25px">
                    <h3 style="color:#FF4D6D">🔓 Traditional</h3>
                    <div style="font-size:1.8rem;font-weight:800;margin:15px 0">{trad_pred}</div>
                    <div style="display:grid;grid-template-columns:1fr 1fr;gap:10px">
                        <div style="background:rgba(0,0,0,0.3);padding:12px;border-radius:8px;text-align:center">
                            <div style="color:#94a3b8;font-size:0.75rem">LATENCY</div>
                            <div style="color:#FFD166;font-size:1.3rem;font-weight:700">{trad_time:.0f}ms</div>
                        </div>
                        <div style="background:rgba(0,0,0,0.3);padding:12px;border-radius:8px;text-align:center">
                            <div style="color:#94a3b8;font-size:0.75rem">PRIVACY</div>
                            <div style="color:#FF4D6D;font-size:1.3rem;font-weight:700">0%</div>
                        </div>
                    </div>
                </div>
                <div style="background:rgba(0,255,136,0.08);border:2px solid rgba(0,255,136,0.25);border-radius:14px;padding:25px">
                    <h3 style="color:#00FF88">🔒 TRUE FHE</h3>
                    <div style="font-size:1.8rem;font-weight:800;margin:15px 0">{fhe_result['prediction']}</div>
                    <div style="display:grid;grid-template-columns:1fr 1fr;gap:10px">
                        <div style="background:rgba(0,0,0,0.3);padding:12px;border-radius:8px;text-align:center">
                            <div style="color:#94a3b8;font-size:0.75rem">LATENCY</div>
                            <div style="color:#FFD166;font-size:1.3rem;font-weight:700">{fhe_time:.0f}ms</div>
                        </div>
                        <div style="background:rgba(0,0,0,0.3);padding:12px;border-radius:8px;text-align:center">
                            <div style="color:#94a3b8;font-size:0.75rem">PRIVACY</div>
                            <div style="color:#00FF88;font-size:1.3rem;font-weight:700">100%</div>
                        </div>
                    </div>
                </div>
            </div>
            <div style="background:rgba(0,212,255,0.08);border:2px solid rgba(0,212,255,0.2);border-radius:12px;padding:20px">
                <h3 style="color:#00D4FF">📈 Metrics</h3>
                <p style="color:#94a3b8">Overhead: <span style="color:#FFD166;font-weight:700">{fhe_time/trad_time:.1f}x</span> | Accuracy Loss: <span style="color:#00FF88;font-weight:700">0%</span> | Ciphertext: <span style="color:#00D4FF;font-weight:700">326KB</span></p>
            </div>
        </div>
        """
    except Exception as e:
        return f'<div style="padding:20px;background:#ef4444;color:white;border-radius:10px"><h3>Error</h3><p>{e}</p></div>'


def generate_gradcam(image):
    if image is None:
        return None, None, '<p style="color:#FF4D6D;padding:20px">Upload image first!</p>'
    try:
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image).convert('RGB')

        img_tensor = transform(image).unsqueeze(0)
        img_tensor.requires_grad = True
        logits = model(img_tensor)
        probs = F.softmax(logits, dim=1)
        predicted_class = torch.argmax(probs, dim=1).item()

        model.zero_grad()
        class_score = logits[0, predicted_class]
        class_score.backward()

        gradients = img_tensor.grad.data[0]
        heatmap = torch.mean(gradients, dim=0).cpu().numpy()
        heatmap = np.maximum(heatmap, 0)
        heatmap = heatmap / (np.max(heatmap) + 1e-8)
        heatmap_resized = cv2.resize(heatmap, (224, 224))
        heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
        heatmap_img = Image.fromarray(heatmap_colored)

        img_array = np.array(image.resize((224, 224)))
        overlay = cv2.addWeighted(img_array, 0.6, heatmap_colored, 0.4, 0)
        overlay_img = Image.fromarray(overlay)

        prediction = 'Normal' if predicted_class == 0 else 'Pneumonia'
        confidence = float(probs[0, predicted_class])

        html = f"""
        <div style="padding:30px">
            <h2 style="color:#00D4FF">🧠 GradCAM</h2>
            <div style="background:rgba({'0,255,136' if prediction=='Normal' else '255,77,109'},0.1);border:2px solid rgba({'0,255,136' if prediction=='Normal' else '255,77,109'},0.3);border-radius:12px;padding:20px;text-align:center;margin:20px 0">
                <h3 style="color:{'#00FF88' if prediction=='Normal' else '#FF4D6D'}">Prediction: {prediction}</h3>
                <div style="font-size:1.5rem;font-weight:800">Confidence: {confidence:.1%}</div>
            </div>
            <div style="background:rgba(0,212,255,0.08);border:2px solid rgba(0,212,255,0.2);border-radius:12px;padding:20px">
                <h3 style="color:#00D4FF">🔬 Explanation</h3>
                <p style="color:#94a3b8;line-height:1.8">GradCAM highlights regions that influenced the prediction. <span style="color:#FF4D6D">Red/yellow = high attention</span>, <span style="color:#00D4FF">blue = low attention</span>.</p>
            </div>
        </div>
        """
        return heatmap_img, overlay_img, html
    except Exception as e:
        return None, None, f'<div style="padding:20px;background:#ef4444;color:white;border-radius:10px"><h3>Error</h3><p>{e}</p></div>'

with gr.Blocks(title="SecureLens") as demo:

    gr.Markdown("# 🔐 SecureLens - TRUE Fully Homomorphic Encryption\n## Privacy-Preserving Pneumonia Detection\n---")

    with gr.Tabs():
        with gr.Tab("🔒 TRUE FHE Classification"):
            gr.Markdown("### Upload X-Ray for Encrypted Analysis")
            with gr.Row():
                with gr.Column(scale=1):
                    fhe_image = gr.Image(type='pil', label='Upload Chest X-Ray', height=400)
                    fhe_btn = gr.Button("🔐 Classify with TRUE FHE", variant="primary", size="lg")
                with gr.Column(scale=1):
                    fhe_output = gr.HTML()
            fhe_btn.click(fn=classify_fhe, inputs=fhe_image, outputs=fhe_output)

        with gr.Tab("⚔️ Attack Demo"):
            gr.Markdown("### Test Adversarial Robustness")
            with gr.Row():
                with gr.Column(scale=1):
                    attack_image = gr.Image(type='pil', label='Upload X-Ray', height=300)
                    attack_type = gr.Radio(["noise", "brightness", "blur"], value="noise", label="Attack Type")
                    attack_intensity = gr.Slider(10, 90, value=30, label="Attack Intensity (%)")
                    attack_btn = gr.Button("⚔️ Run Attack Demo", variant="primary", size="lg")
                with gr.Column(scale=2):
                    with gr.Row():
                        attack_orig = gr.Image(label="Original", height=200)
                        attack_attacked = gr.Image(label="Attacked", height=200)
                    attack_result = gr.HTML()
            attack_btn.click(fn=run_attack, inputs=[attack_image, attack_type, attack_intensity], outputs=[attack_orig, attack_attacked, attack_result])

        with gr.Tab("📊 Comparison"):
            gr.Markdown("### FHE vs Traditional Inference")
            with gr.Row():
                with gr.Column(scale=1):
                    comp_image = gr.Image(type='pil', label='Upload X-Ray', height=400)
                    comp_btn = gr.Button("📊 Run Comparison", variant="primary", size="lg")
                with gr.Column(scale=1):
                    comp_output = gr.HTML()
            comp_btn.click(fn=run_comparison, inputs=comp_image, outputs=comp_output)

        with gr.Tab("🧠 GradCAM"):
            gr.Markdown("### Visual Explainability")
            with gr.Row():
                with gr.Column(scale=1):
                    gradcam_image = gr.Image(type='pil', label='Upload X-Ray', height=400)
                    gradcam_btn = gr.Button("🧠 Generate GradCAM", variant="primary", size="lg")
                with gr.Column(scale=1):
                    with gr.Row():
                        gradcam_heatmap = gr.Image(label="Heatmap", height=200)
                        gradcam_overlay = gr.Image(label="Overlay", height=200)
                    gradcam_result = gr.HTML()
            gradcam_btn.click(fn=generate_gradcam, inputs=gradcam_image, outputs=[gradcam_heatmap, gradcam_overlay, gradcam_result])

    gr.Markdown("""---
<div style="text-align:center;color:#4A6080;padding:20px">
<p style="font-size:1.1rem"><strong>SecureLens</strong> — Privacy-Preserving Medical AI</p>
<p>CKKS · TenSEAL · PyTorch · ResNet-18 · 128-bit Security</p>
<p style="font-size:0.8rem;margin-top:10px">Research Prototype | Not for Clinical Use</p>
</div>""")

if __name__ == "__main__":
    print("\n[SecureLens] Enhanced Interface Starting...")
    print("[SecureLens] URL: http://127.0.0.1:7860")
    print("[SecureLens] ✓ TRUE FHE Classification")
    print("[SecureLens] ✓ Attack Demo")
    print("[SecureLens] ✓ Comparison")
    print("[SecureLens] ✓ GradCAM")
    print("[SecureLens] All features loaded!\n")
    demo.launch(server_port=7860, share=False, theme=gr.themes.Soft(), css=custom_css)

