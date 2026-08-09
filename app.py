# SecureLens - Hugging Face Spaces Entry Point
# Launches the enhanced 5-tab interface with TRUE FHE

import os
import sys

# Ensure the project root is in the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import the enhanced demo from the full app
from app_gradio_enhanced_FOR_HF import demo

if __name__ == "__main__":
    demo.queue()
    demo.launch(server_name="0.0.0.0", server_port=7860)
