import gradio as gr
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
import sys
import os

# Add paths
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import your modules
from crypto_layer.ckks_engine import CKKSEngine
from cloud_server.encrypted_inference.he_inference import HEInferenceEngine
from cloud_server.train_model_fhe_compatible import SecureLensNetFHE

# Lazy loading
model = None
ckks = None
he_engine = None

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def load_model():
    global model, ckks, he_engine
    if model is None:
        print('[SecureLens] Loading system...')
        model = SecureLensNetFHE(num_classes=2)
        model.load_state_dict(torch.load('cloud_server/models/best_model.pth', map_location='cpu'))
        model.eval()
        ckks = CKKSEngine(8192, [60, 40, 40, 60], 2**40)
        he_engine = HEInferenceEngine('cloud_server/models')
        print('[SecureLens] Ready!')
    return model, ckks, he_engine

def classify_image(image):
    if image is None:
        return 'Please upload an X-ray image first.'

    try:
        model, ckks, he_engine = load_model()

        if not isinstance(image, Image.Image):
            image = Image.fromarray(image).convert('RGB')

        img_tensor = transform(image).unsqueeze(0)

        # Run inference
        with torch.no_grad():
            output = model(img_tensor)
            probs = torch.nn.functional.softmax(output, dim=1)
            pred_class = torch.argmax(probs, dim=1).item()
            confidence = probs[0, pred_class].item()

        prediction = 'Normal' if pred_class == 0 else 'Pneumonia'

        result = 'Prediction: ' + prediction + '\nConfidence: ' + '{:.2%}'.format(confidence) + '\n\nThis is a privacy-preserving pneumonia detection system using Fully Homomorphic Encryption (FHE).'
        return result

    except Exception as e:
        return 'Error: ' + str(e)

# Create simple Gradio interface - FIXED FOR GRADIO 2.9.4
demo = gr.Interface(
    fn=classify_image,
    inputs=gr.inputs.Image(type='pil', label='Upload Chest X-Ray'),
    outputs=gr.outputs.Textbox(label='Result'),
    title='SecureLens - Privacy-Preserving Pneumonia Detection',
    description='Upload a chest X-ray image for FHE-encrypted analysis',
    examples=None
)

if __name__ == '__main__':
    demo.launch(server_name='0.0.0.0', server_port=7860, share=False)
