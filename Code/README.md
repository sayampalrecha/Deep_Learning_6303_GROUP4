
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

## Demo app
Once a checkpoint exists in `models/`, open the app folder (Streamlit example):
```bash
cd app 
streamlit run app.py
```

## References
- LipNet: “LipNet: End-to-End Sentence-level Lipreading” (Chung et al., 2016) – https://arxiv.org/abs/1611.01599
- CTC tutorial: https://distill.pub/2017/ctc

