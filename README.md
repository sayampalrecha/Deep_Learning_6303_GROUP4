# Deep_Learning_6303_GROUP4
A lightweight implementation of LipNet, a deep-learning model for lip-reading from video. <br>
This version focuses on simplicity by using less speaker than original paper while demonstrating core concepts:
video preprocessing, spatiotemporal feature extraction, and sequence prediction.

# Input data
![pbao9s](https://github.com/user-attachments/assets/4aed833a-993f-4500-a9ab-45aadbf80d04) ![lwip5p](https://github.com/user-attachments/assets/3bf27b62-f0bc-4970-ad18-bc70b751d84f)

# Preprocessing steps
1. Input the data
2. Convert it to grayscale
3. Crop the mouth features
4. Normalization of data
<img width="824" height="201" alt="image" src="https://github.com/user-attachments/assets/9e52e0eb-8e1c-4925-9fda-ccdf7af5aa7b" /> <br>
This is a frame-frame grayscaled image of the mouth-features

# LipNet Research Paper
[LipNet Paper](https://arxiv.org/abs/1611.01599)


# Model Architecture
<img width="2438" height="346" alt="Untitled diagram-2025-12-08-170612" src="https://github.com/user-attachments/assets/3cd2862e-195e-4eda-b7cc-db82e29fcafb" /><br>

Input Shape: (75,46,140,1)<br>
Kernel Size: 3x3x3 <br>
Regularization: Dropout (0.5) and Batch Normalization <br>

## Quick start (Code folder)
Everything needed to preprocess data, train the lip-reader, and launch the demo lives under `Code/`. From the project root:

```bash
cd Code

#create virtual env
python3 -m venv .venv
source .venv/bin/activate

# install dependencies listed by the team
pip install -r requirements.txt

# preprocess raw videos into mouth-only frame tensors + transcripts
python3 data-preprocessing.py  # edit paths inside as needed

# train the LipNet-style model (expects processed_data/*_frames.npy)
python3 Train_LipReader.py

# run the interactive demo app once a checkpoint exists in models/
cd app
streamlit run app.py
```




