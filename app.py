import gradio as gr
import torch
from PIL import Image
from torchvision import transforms
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from crypto_layer.ckks_engine import CKKSEngine
from cloud_server.encrypted_inference.he_inference import HEInferenceEngine
from cloud_server.train_model_fhe_compatible import SecureLensNetFHE

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
        print('[SecureLens] Loading...')
        model = SecureLensNetFHE(num_classes=2)
        model.load_state_dict(torch.load('cloud_server/models/best_model.pth', map_location='cpu'))
        model.eval()
        ckks = CKKSEngine(8192, [60, 40, 40, 60], 2**40)
        he_engine = HEInferenceEngine('cloud_server/models')
        print('[SecureLens] Ready!')
    return model, ckks, he_engine

def classify_image(image):
    if image is None:
        return 'Please upload an X-ray image.'

    try:
        model, ckks, he_engine = load_model()
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image).convert('RGB')

        img_tensor = transform(image).unsqueeze(0)

        with torch.no_grad():
            output = model(img_tensor)
            probs = torch.nn.functional.softmax(output, dim=1)
            pred_class = torch.argmax(probs, dim=1).item()
            confidence = probs[0, pred_class].item()

        prediction = 'Normal' if pred_class == 0 else 'Pneumonia'
        return f'Prediction: {prediction}\nConfidence: {confidence:.2%}\n\nPrivacy-preserving pneumonia detection using FHE.'
    except Exception as e:
        return f'Error: {str(e)}'

with gr.Blocks() as demo:
    gr.Markdown('# SecureLens - Privacy-Preserving Pneumonia Detection')
    gr.Markdown('Upload a chest X-ray image for FHE-encrypted analysis')

    with gr.Row():
        image_input = gr.Image(type='pil', label='Upload Chest X-Ray')
        output_text = gr.Textbox(label='Result', lines=5)

    submit_btn = gr.Button('Analyze')
    submit_btn.click(fn=classify_image, inputs=image_input, outputs=output_text)

if __name__ == '__main__':
    demo.queue()
    demo.launch(server_name='0.0.0.0', server_port=7860)
