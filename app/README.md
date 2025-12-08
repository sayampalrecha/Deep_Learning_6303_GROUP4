# LipReading Demo App

The app performs three main tasks:

1. Preprocess the uploaded video into 75 grayscale lip-region frames  
2. Display sample frames and a looping GIF preview  
3. Run our model and output the predicted transcription

## Requirements

This app requires **Python 3.10 or 3.11**.  
TensorFlow does not support Python 3.12 and the model will not load correctly on 3.12+.

Recommended Setup:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

**IMPORTANT:**
Model was too large to upload to GitHub. Please download the "models" folder from [Google Drive](https://drive.google.com/drive/folders/18nZdxe_P3KdJYRGunXz2ue2N0yOJd-7I?usp=drive_link) and add within the "app" folder before running the Streamlit app.