# Lip Reading Mini-Stack

This directory bundles everything needed for a lightweight end-to-end lip-reading experiment inspired by LipNet. It covers preprocessing raw talking-head videos, training a PyTorch model that blends 3D convolutions with BiLSTMs + CTC, and a minimal demo app for inference.

## What’s inside
- `data-preprocessing.py`: converts raw video clips into cropped, normalized mouth ROIs and paired text transcripts. Outputs `.npy` frame tensors and `.txt` transcripts.
- `Train_LipReader.py`: PyTorch training loop with dataset loading, tokenizer creation, CTC-aware model, CER/WER tracking, checkpointing, and GradScaler support.
- `app/`: lightweight Streamlit/Gradio-style demo (see folder for details) that loads a trained checkpoint and runs inference on short clips.

## Pipeline at a glance
1. **Data ingestion** – read each video, convert to grayscale or RGB, stabilize frame counts (default 75), and store shape `(time, H, W, C)`.
2. **Mouth ROI extraction** – crop to lips using the landmarks logic inside `data-preprocessing.py` (dlib/OpenCV), then normalize pixel intensities.
3. **Serialization** – save aligned frame stacks as `*_frames.npy` and matching lowercase transcripts as `*_text.txt`.
4. **Training** – `Train_LipReader.py` scans `processed_data/` folders listed in `load_all_manifest_from_folders()`, pads videos to 75 frames, encodes characters with a tokenizer, and feeds batches through the CNN + BiLSTM + attention + CTC head.
5. **Monitoring** – prints CER/WER every epoch, performs early stopping with patience, and writes weights plus `char_tokenizer.pkl` under `models/`.

## Quick start
```bash
# create env (optional)
python -m venv .venv
source .venv/bin/activate

# install core deps
pip install -r requirements.txt

# preprocess raw videos (adjust input/output paths inside the script)
python3 data-preprocessing.py

# train model (expects processed_data folders defined at top of script)
python3 Train_LipReader.py
```

## Demo app
Once a checkpoint exists in `models/`, open the app folder (Streamlit example):
```bash
cd app 
streamlit run app.py
```

## References
- LipNet: “LipNet: End-to-End Sentence-level Lipreading” (Chung et al., 2016) – https://arxiv.org/abs/1611.01599
- CTC tutorial: https://distill.pub/2017/ctc


