import streamlit as st
import cv2
import numpy as np
import tempfile
import mediapipe as mp
import os
import imageio

MAX_FRAMES_PREVIEW = 25
FRAME_WIDTH, FRAME_HEIGHT = 128, 128


@st.cache_resource
def load_face_mesh():
    mp_face_mesh = mp.solutions.face_mesh
    return mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )


def extract_lip_region(frame, face_mesh, debug_info, frame_count):
    try:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    except Exception as ex:
        debug_info.append(f"Frame {frame_count}: Failed to convert to RGB - {ex}")
        return None

    try:
        results = face_mesh.process(rgb_frame)
    except Exception as ex:
        debug_info.append(f"Frame {frame_count}: FaceMesh processing failed - {ex}")
        return None

    if not results.multi_face_landmarks:
        debug_info.append(f"Frame {frame_count}: No face landmarks detected")
        return None

    h, w, _ = frame.shape
    landmarks = results.multi_face_landmarks[0].landmark

    # Lip landmark indices (outer + inner lips) for MediaPipe FaceMesh
    lip_indices = [
        61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 78,
        95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 415, 310
    ]

    xs, ys = [], []
    for idx in lip_indices:
        x = int(landmarks[idx].x * w)
        y = int(landmarks[idx].y * h)
        xs.append(x)
        ys.append(y)

    if not xs or not ys:
        debug_info.append(f"Frame {frame_count}: Lip landmarks empty")
        return None

    x_min, x_max = max(min(xs) - 10, 0), min(max(xs) + 10, w)
    y_min, y_max = max(min(ys) - 10, 0), min(max(ys) + 10, h)

    if x_min >= x_max or y_min >= y_max:
        debug_info.append(f"Frame {frame_count}: Invalid lip bounding box")
        return None

    lip_region = frame[y_min:y_max, x_min:x_max]
    if lip_region.size == 0:
        debug_info.append(f"Frame {frame_count}: Lip region is empty")
        return None

    try:
        # Better interpolation for nicer-looking crops
        lip_region = cv2.resize(
            lip_region, (FRAME_WIDTH, FRAME_HEIGHT),
            interpolation=cv2.INTER_CUBIC
        )
        lip_region = cv2.cvtColor(lip_region, cv2.COLOR_BGR2RGB)
    except Exception as ex:
        debug_info.append(f"Frame {frame_count}: Failed to resize/convert lip region - {ex}")
        return None

    debug_info.append(f"Frame {frame_count}: Lip region extracted successfully")
    return lip_region


def process_preview_frames(video_path, max_frames=MAX_FRAMES_PREVIEW):

    debug_info = []
    lip_frames = []
    lips_found = False

    face_mesh = load_face_mesh()

    if not os.path.exists(video_path):
        debug_info.append(f"Video path does not exist: {video_path}")
        return lip_frames, lips_found, debug_info

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        debug_info.append(f"Failed to open video: {video_path}")
        return lip_frames, lips_found, debug_info

    debug_info.append(f"Opened video: {video_path}")

    frame_count = 0

    while frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            debug_info.append(f"Frame {frame_count}: Failed to read (end of video or error)")
            break

        frame_count += 1
        debug_info.append(f"Frame {frame_count}: Read successfully")

        lip_region = extract_lip_region(frame, face_mesh, debug_info, frame_count)
        if lip_region is not None:
            lip_frames.append(lip_region)
            lips_found = True

    cap.release()

    if lips_found:
        debug_info.append(f"Lips detected in {len(lip_frames)} frame(s).")
    else:
        debug_info.append("No lips detected in any of the preview frames.")

    return lip_frames, lips_found, debug_info


def create_preview_gif(frames, output_path, num_loops=8, fps=10, scale=2):
    
    # Normalize to 0–255 and convert to uint8
    frames_uint8 = (
        (frames - frames.min()) /
        (frames.max() - frames.min() + 1e-8) * 255
    ).astype(np.uint8)

    # (T, H, W, 1) -> (T, H, W)
    frames_uint8 = np.squeeze(frames_uint8, axis=-1)

    # Upscale each frame so the GIF appears larger
    t, h, w = frames_uint8.shape
    upscaled_frames = [
        cv2.resize(frame, (w * scale, h * scale), interpolation=cv2.INTER_NEAREST)
        for frame in frames_uint8
    ]

    # Repeat the short sequence multiple times for a longer looping animation
    all_frames = upscaled_frames * num_loops

    # loop=0 = infinite loop
    imageio.mimsave(output_path, all_frames, fps=fps, loop=0)


def main():
    st.title("Lip Reading Preview")

    st.markdown(
        f"""
        1. Upload a video of a speaking face (**mp4 / avi / mov / mpg / mpeg**).
        2. Click **Let's Start!** to extract up to **{MAX_FRAMES_PREVIEW}** lip frames.
        3. View an animated black-and-white GIF preview and sample frames.
        4. Observe the predicted text output (placeholder for now).
        """
    )

    uploaded_file = st.file_uploader(
        "Upload a video",
        type=["mp4", "avi", "mov", "mpg", "mpeg"]
    )

    if uploaded_file is not None:
        # Save uploaded file to a temporary location
        suffix = os.path.splitext(uploaded_file.name)[1]
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        tfile.write(uploaded_file.read())
        tfile.flush()
        video_path = tfile.name

        if st.button("Let's Start!"):
            with st.spinner("Processing frames..."):
                lip_frames, lips_found, debug_info = process_preview_frames(video_path)

            if lips_found and len(lip_frames) > 0:
                st.success(f"Cropped lips from {len(lip_frames)} frames!")

                st.markdown("### Sample Cropped Lip Frames")

                num_display = min(5, len(lip_frames))
                step = max(1, len(lip_frames) // num_display)
                display_frames = [
                    lip_frames[i] for i in range(0, len(lip_frames), step)
                ][:num_display]

                cols = st.columns(len(display_frames))
                for i, frame in enumerate(display_frames):
                    cols[i].image(
                        frame,
                        caption=f"Frame {i * step + 1}",
                        use_container_width=True,
                    )

                st.markdown("### Grayscale Preview GIF & Predicted Output")

                # Convert lip_frames (RGB) -> grayscale float32, shape (T, H, W, 1)
                frames_gray = [
                    cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY) for frame in lip_frames
                ]
                frames_gray = np.array(frames_gray, dtype=np.float32)
                frames_gray = frames_gray[..., np.newaxis]

                gif_temp = tempfile.NamedTemporaryFile(delete=False, suffix=".gif")
                create_preview_gif(frames_gray, gif_temp.name, scale=2)

                with open(gif_temp.name, "rb") as f:
                    gif_bytes = f.read()

                # Placeholder
                predicted_text = "Predicted text will appear here once the model/output is available."

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
                        "Predicted transcription:",
                        value=predicted_text,
                        height=150,
                    )

            else:
                st.warning("No lips detected in the video.")

            with st.expander("Debug Information"):
                st.text("\n".join(debug_info))


if __name__ == "__main__":
    main()