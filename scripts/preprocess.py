import os
import cv2
import numpy as np
from tqdm import tqdm

# ---------- CONFIG ----------
DATA_DIR = "data"
OUTPUT_DIR = "data/processed"
FRAME_SIZE = (224, 224)
FRAMES_PER_CLIP = 16
# ----------------------------

def extract_frames(video_path, num_frames=FRAMES_PER_CLIP):
    import warnings
    warnings.filterwarnings('ignore')  # Suppress OpenCV warnings for corrupted files

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    # Read all frames (MKV files may be corrupted/incomplete)
    all_frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # Only keep valid frames
        if frame is not None and frame.size > 0:
            all_frames.append(frame)

    cap.release()

    total_frames = len(all_frames)
    if total_frames < num_frames:
        raise ValueError(f"Video has only {total_frames} frames, need at least {num_frames}")

    # Sample evenly spaced frames
    frame_idxs = np.linspace(0, total_frames - 1, num_frames).astype(int)
    sampled_frames = []

    for idx in frame_idxs:
        frame = all_frames[idx]
        frame = cv2.resize(frame, FRAME_SIZE)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = frame / 255.0
        sampled_frames.append(frame)

    return np.array(sampled_frames, dtype=np.float32)

def process_split(split):
    input_path = os.path.join(DATA_DIR, split)
    output_path = os.path.join(OUTPUT_DIR, split)
    os.makedirs(output_path, exist_ok=True)

    # Updated labels to match your data structure
    for label in ["shoplifting", "normal"]:
        input_folder = os.path.join(input_path, label)
        output_folder = os.path.join(output_path, label)
        os.makedirs(output_folder, exist_ok=True)

        if not os.path.exists(input_folder):
            print(f"Warning: {input_folder} does not exist, skipping...")
            continue

        # Recursively find all video files
        video_files = []
        for root, dirs, files in os.walk(input_folder):
            for file in files:
                if file.endswith(('.avi', '.mp4', '.mkv')):
                    video_files.append(os.path.join(root, file))

        if len(video_files) == 0:
            print(f"Warning: No videos found in {input_folder}")
            continue

        success_count = 0
        fail_count = 0

        for video_path in tqdm(video_files, desc=f"{split}/{label}"):
            try:
                frames = extract_frames(video_path)
                # Use unique filename based on original path
                video_name = os.path.basename(video_path)
                output_name = video_name.replace(".avi", ".npy").replace(".mp4", ".npy").replace(".mkv", ".npy")
                np.save(os.path.join(output_folder, output_name), frames)
                success_count += 1
            except Exception as e:
                print(f"\n⚠ Error processing {os.path.basename(video_path)}: {e}")
                fail_count += 1

        print(f"{split}/{label}: {success_count} succeeded, {fail_count} failed")

if __name__ == "__main__":
    for split in ["train", "test"]:
        process_split(split)
    print("Preprocessing complete.")
