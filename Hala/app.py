import streamlit as st
import cv2
import numpy as np
import tempfile
import mediapipe as mp
import os

MAX_FRAMES_PREVIEW = 3
FRAME_WIDTH, FRAME_HEIGHT = 64, 64


@st.cache_resource
def load_face_mesh():
    mp_face_mesh = mp.solutions.face_mesh
    return mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.3
    )


face_mesh = load_face_mesh()


def process_preview_frames(video_path, max_frames=MAX_FRAMES_PREVIEW):
    cap = cv2.VideoCapture(video_path)
    frames = []
    lips_found = False
    debug_info = []

    if not cap.isOpened():
        st.error("Cannot open video file.")
        return np.array([]), False, debug_info

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    debug_info.append(f"Video: {total_frames} frames, {fps:.2f} fps, {width}x{height}")

    frame_count = 0
    max_attempts = 50

    while cap.isOpened() and len(frames) < max_frames and frame_count < max_attempts:
        ret, frame = cap.read()

        frame_count += 1

        if not ret or frame is None:
            debug_info.append(f"Frame {frame_count}: Read failed")
            if not ret:
                break
            continue

        try:
            frame = frame.copy()

            h, w = frame.shape[:2]
            if w > 640:
                new_w = 640
                new_h = int(h * (640 / w))
                frame = cv2.resize(frame, (new_w, new_h))

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            debug_info.append(f"Frame {frame_count}: Converted to RGB successfully")

        except Exception as ex:
            debug_info.append(f"Frame {frame_count}: Conversion failed")
            continue

        try:
            results = face_mesh.process(rgb_frame)
        except Exception as ex:
            debug_info.append(f"Frame {frame_count}: FaceMesh failed")
            continue

        if not results.multi_face_landmarks:
            debug_info.append(f"Frame {frame_count}: No face detected")
            continue

        debug_info.append(f"Frame {frame_count}: Face detected!")

        face_landmarks = results.multi_face_landmarks[0]
        h, w = frame.shape[:2]

        lip_idx = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291,
                   78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308]

        xs = [int(face_landmarks.landmark[i].x * w) for i in lip_idx]
        ys = [int(face_landmarks.landmark[i].y * h) for i in lip_idx]

        padding = 20
        min_x = max(min(xs) - padding, 0)
        max_x = min(max(xs) + padding, w)
        min_y = max(min(ys) - padding, 0)
        max_y = min(max(ys) + padding, h)

        if min_x >= max_x or min_y >= max_y:
            debug_info.append(f"Frame {frame_count}: Invalid crop bounds")
            continue

        crop = frame[min_y:max_y, min_x:max_x]
        if crop.size == 0:
            debug_info.append(f"Frame {frame_count}: Empty crop")
            continue

        crop = cv2.resize(crop, (FRAME_WIDTH, FRAME_HEIGHT))
        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        frames.append(crop_rgb.astype(np.float32) / 255.0)
        lips_found = True
        debug_info.append(f"Frame {frame_count}: SUCCESS!")

    cap.release()

    return np.array(frames, dtype=np.float32), lips_found, debug_info


st.title("Lip Detection App")
st.header("Where Vision Meets Voice")

st.header("Step 1: Upload a Video")
uploaded_file = st.file_uploader("Upload a video", type=["mp4", "mov", "avi"])
video_path = None

if uploaded_file is not None:
    st.success("Upload successful!")

    suffix = os.path.splitext(uploaded_file.name)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix, mode='wb') as tmp_video:
        tmp_video.write(uploaded_file.read())
        tmp_video.flush()
        os.fsync(tmp_video.fileno())
        video_path = tmp_video.name

    st.video(video_path)

st.header("Step 2: Analyze the video.")
if video_path is not None:
    if st.button("Let's Start!"):
        with st.spinner("Processing frames..."):
            lip_frames, lips_found, debug_info = process_preview_frames(video_path)

        if lips_found and len(lip_frames) > 0:
            st.success(f"Cropped lips from {len(lip_frames)} frames!")

            cols = st.columns(len(lip_frames))
            for i, frame in enumerate(lip_frames):
                cols[i].image(frame, caption=f"Frame {i + 1}", use_container_width=True)
        else:
            st.warning("No lips detected in the video.")

        with st.expander("Debug Information"):
            st.text("\n".join(debug_info))