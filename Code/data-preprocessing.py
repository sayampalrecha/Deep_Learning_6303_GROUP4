import os
import glob
import cv2
import mediapipe as mp
import numpy as np
import pickle

BASE_PATH = os.path.expanduser("~/DL/DL Project/full_dataset")
OUTPUT_DIR = os.path.join(os.getcwd(), "processed_data")
os.makedirs(OUTPUT_DIR, exist_ok=True)

FRAME_HEIGHT = 64
FRAME_WIDTH = 64
MAX_FRAMES = 75

mp_face_mesh = mp.solutions.face_mesh

def load_alignment_file(align_path):
    words = []
    try:
        with open(align_path, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 3:
                    word = parts[2].strip()
                    words.append(word)
        if words:
            return ' '.join(words).lower()
        return None
    except Exception as e:
        print(f"Error reading {align_path}: {e}")
        return None


def load_data_pairs():
    data_pairs = []
    folders = tuple(f"s{i}_processed" for i in range(3, 6))

    for folder in folders:
        video_folder_path = os.path.join(BASE_PATH, folder)
        if not os.path.isdir(video_folder_path):
            continue

        video_files = glob.glob(os.path.join(video_folder_path, '*.mpg'))
        align_folder_path = os.path.join(video_folder_path, 'align')
        align_files = glob.glob(os.path.join(align_folder_path, '*.align')) if os.path.isdir(align_folder_path) else []

        align_map = {os.path.splitext(os.path.basename(a))[0]: a for a in align_files}

        for video_file in video_files:
            vid_basename = os.path.splitext(os.path.basename(video_file))[0]
            align_file = align_map.get(vid_basename)
            if align_file:
                text = load_alignment_file(align_file)
                if text:
                    data_pairs.append((video_file, align_file, text))

    print(f"Loaded {len(data_pairs)} video-alignment pairs")
    return data_pairs

def process_video(video_path, max_frames=MAX_FRAMES):
    cap = cv2.VideoCapture(video_path)
    frames = []

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames == 0:
        total_frames = max_frames

    with mp_face_mesh.FaceMesh(static_image_mode=True) as face_mesh:
        frame_idx = 0
        while cap.isOpened() and len(frames) < max_frames:
            ret, frame = cap.read()
            if not ret or frame is None:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb = rgb.astype(np.uint8)

            try:
                results = face_mesh.process(rgb)
            except Exception as e:
                print(f"Skipping frame {frame_idx+1} due to MediaPipe error: {e}")
                continue

            if results.multi_face_landmarks:
                h, w, _ = frame.shape
                xs = [int(lm.x * w) for lm in results.multi_face_landmarks[0].landmark[61:88]]
                ys = [int(lm.y * h) for lm in results.multi_face_landmarks[0].landmark[61:88]]

                min_x, max_x = max(min(xs) - 5, 0), min(max(xs) + 5, w)
                min_y, max_y = max(min(ys) - 5, 0), min(max(ys) + 5, h)

                crop = frame[min_y:max_y, min_x:max_x]
                if crop.size == 0:
                    continue

                crop = cv2.resize(crop, (FRAME_WIDTH, FRAME_HEIGHT))
                frames.append(crop.astype(np.float32) / 255.0)

            frame_idx += 1
            print(f"\rProcessing frame {frame_idx}/{total_frames} ({(frame_idx/total_frames)*100:.1f}%)", end="")

    cap.release()
    print()
    return np.array(frames, dtype=np.float32)

def main():
    data_pairs = load_data_pairs()
    manifest = []

    for idx, (video_path, align_path, text) in enumerate(data_pairs):
        print(f"\nProcessing video {idx+1}/{len(data_pairs)}: {os.path.basename(video_path)}")
        frames = process_video(video_path)
        if frames is not None and len(frames) > 0:
            frames_file = os.path.join(OUTPUT_DIR, f'video_{idx}_frames.npy')
            np.save(frames_file, frames)

            text_file = os.path.join(OUTPUT_DIR, f'video_{idx}_text.txt')
            with open(text_file, 'w') as f:
                f.write(text)

            manifest.append((frames_file, text_file))

    manifest_path = os.path.join(OUTPUT_DIR, 'manifest3.pkl')
    with open(manifest_path, 'wb') as f:
        pickle.dump(manifest, f, protocol=5)

    print(f"\nProcessed {len(manifest)} videos. Manifest saved to {manifest_path}")

if __name__ == "__main__":
    main()
