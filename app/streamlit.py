import os
import cv2
import numpy as np
import tempfile
import imageio
import tensorflow as tf
import streamlit as st

# config
vocab = [x for x in "abcdefghijklmnopqrstuvwxyz'?!123456789 "]

char_to_num = tf.keras.layers.StringLookup(vocabulary=vocab, oov_token="")
num_to_char = tf.keras.layers.StringLookup(
    vocabulary=char_to_num.get_vocabulary(), oov_token="", invert=True
)

# model
def build_lipnet_model() -> tf.keras.Model:
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import (
        Conv3D,
        LSTM,
        Dense,
        Dropout,
        Bidirectional,
        MaxPool3D,
        Activation,
        TimeDistributed,
        Flatten,
    )

    model = Sequential()
    model.add(Conv3D(128, 3, input_shape=(75, 46, 140, 1), padding="same"))
    model.add(Activation("relu"))
    model.add(MaxPool3D((1, 2, 2)))

    model.add(Conv3D(256, 3, padding="same"))
    model.add(Activation("relu"))
    model.add(MaxPool3D((1, 2, 2)))

    model.add(Conv3D(75, 3, padding="same"))
    model.add(Activation("relu"))
    model.add(MaxPool3D((1, 2, 2)))

    model.add(TimeDistributed(Flatten()))

    model.add(
        Bidirectional(
            LSTM(128, kernel_initializer="Orthogonal", return_sequences=True)
        )
    )
    model.add(Dropout(0.5))

    model.add(
        Bidirectional(
            LSTM(128, kernel_initializer="Orthogonal", return_sequences=True)
        )
    )
    model.add(Dropout(0.5))

    model.add(
        Dense(
            char_to_num.vocabulary_size() + 1,
            kernel_initializer="he_normal",
            activation="softmax",
            dtype="float32",
        )
    )

    return model


@st.cache_resource
def load_lipnet_model() -> tf.keras.Model:
    model = build_lipnet_model()
    weights_path = os.path.join("models", "checkpoint_fast.weights.h5")
    model.load_weights(weights_path)
    return model


def load_video_for_model(path: str) -> tf.Tensor:
    cap = cv2.VideoCapture(path)
    frames = []

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    for _ in range(frame_count):
        ret, frame = cap.read()
        if not ret or frame is None:
            continue
        frame = tf.image.rgb_to_grayscale(frame)
        frames.append(frame[190:236, 80:220, :])
    cap.release()

    if len(frames) == 0:
        frames = [tf.zeros((46, 140, 1), dtype=tf.float32) for _ in range(75)]
    elif len(frames) < 75:
        while len(frames) < 75:
            frames.append(frames[-1])
    elif len(frames) > 75:
        frames = frames[:75]

    frames = tf.stack(frames)

    mean = tf.math.reduce_mean(frames)
    std = tf.math.reduce_std(tf.cast(frames, tf.float32))
    std = tf.maximum(std, 1e-8)
    frames = tf.cast((frames - mean), tf.float32) / std

    return frames


def decode_prediction(y_pred: tf.Tensor) -> str:
    input_len = tf.fill([tf.shape(y_pred)[0]], tf.shape(y_pred)[1])
    decoded, _ = tf.keras.backend.ctc_decode(
        y_pred, input_length=input_len, greedy=False
    )
    decoded = decoded[0][0]

    text = tf.strings.reduce_join(num_to_char(decoded)).numpy().decode("utf-8")
    return text


def predict_text_from_video(video_path: str) -> str:
    frames = load_video_for_model(video_path)
    x = tf.expand_dims(frames, axis=0)

    model = load_lipnet_model()
    y_pred = model(x, training=False)

    return decode_prediction(y_pred)


# looping GIF and sample frames
def create_preview_gif(frames: np.ndarray, output_path: str,
                       num_loops: int = 8, fps: int = 10, scale: int = 2):

    frames_uint8 = (
        (frames - frames.min()) / (frames.max() - frames.min() + 1e-8) * 255
    ).astype(np.uint8)

    frames_uint8 = np.squeeze(frames_uint8, axis=-1)

    t, h, w = frames_uint8.shape
    upscaled_frames = [
        cv2.resize(frame, (w * scale, h * scale), interpolation=cv2.INTER_NEAREST)
        for frame in frames_uint8
    ]

    all_frames = upscaled_frames * num_loops
    imageio.mimsave(output_path, all_frames, fps=fps, loop=0)


def tensor_to_sample_frames(frames: tf.Tensor, num_display: int = 5):

    frames_np = frames.numpy()
    T = frames_np.shape[0]

    num_display = min(num_display, T)
    step = max(1, T // num_display)
    indices = list(range(0, T, step))[:num_display]

    display_frames = []
    for idx in indices:
        frame = frames_np[idx, :, :, 0]
        f = (frame - frame.min()) / (frame.max() - frame.min() + 1e-8) * 255
        f = f.astype(np.uint8)
        f = cv2.resize(f, (140 * 2, 46 * 2), interpolation=cv2.INTER_NEAREST)
        display_frames.append(f)

    return display_frames, indices

# streamlit
def main():
    st.title("Lipreading Prediction Demo")

    st.markdown(
        """
        1. Upload a video of a speaking face.
        2. Click **Let's Start!** to:
           - preprocess the video into lip-region frames
           - show sample grayscale frames and looping GIF
           - run model and display the predicted output text
        """
    )

    uploaded_file = st.file_uploader(
        "Upload a video",
        type=["mpg", "mp4", "avi", "mov", "mpeg"],
    )

    if uploaded_file is not None:
        suffix = os.path.splitext(uploaded_file.name)[1]
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        tfile.write(uploaded_file.read())
        tfile.flush()
        video_path = tfile.name

        if st.button("Let's Start!"):
            with st.spinner("Processing video and running LipNet model..."):
                frames = load_video_for_model(video_path)
                display_frames, indices = tensor_to_sample_frames(frames)
                frames_np = frames.numpy()
                gif_temp = tempfile.NamedTemporaryFile(delete=False, suffix=".gif")
                create_preview_gif(frames_np, gif_temp.name, scale=2)

                with open(gif_temp.name, "rb") as f:
                    gif_bytes = f.read()

                try:
                    predicted_text = predict_text_from_video(video_path)
                except Exception as ex:
                    predicted_text = f"(Prediction failed: {ex})"

            st.success("Processing complete!")

            st.markdown("### Sample Cropped Lip Frames (Grayscale)")
            cols = st.columns(len(display_frames))
            for i, frame in enumerate(display_frames):
                cols[i].image(
                    frame,
                    caption=f"Frame {indices[i] + 1}",
                    use_container_width=True,
                )

            st.markdown("### Grayscale Preview GIF & Predicted Output")
            gif_col, text_col = st.columns([1, 1])

            with gif_col:
                st.image(
                    gif_bytes,
                    caption="Lip Region GIF Preview",
                    width=320,
                )

            with text_col:
                st.markdown("**Predicted Output**")
                st.text_area(
                    "Model transcription:",
                    value=predicted_text,
                    height=150,
                )


if __name__ == "__main__":
    main()