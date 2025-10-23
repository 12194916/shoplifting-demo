import os
import shutil
from sklearn.model_selection import train_test_split

# ---------- CONFIG ----------
DATA_DIR = "data"
TRAIN_SPLIT = 0.8  # 80% train, 20% test
RANDOM_SEED = 42
# ----------------------------

def split_dataset():
    """
    Split the dataset into train and test sets

    Original structure:
        data/normal/*.mp4
        data/shoplifting/*.mp4

    New structure:
        data/train/normal/*.mp4
        data/train/shoplifting/*.mp4
        data/test/normal/*.mp4
        data/test/shoplifting/*.mp4
    """

    for label in ["normal", "shoplifting"]:
        source_dir = os.path.join(DATA_DIR, label)

        if not os.path.exists(source_dir):
            print(f"Warning: {source_dir} does not exist, skipping...")
            continue

        # Get all video files
        video_files = [f for f in os.listdir(source_dir) if f.endswith(('.mp4', '.avi', '.mkv'))]

        if len(video_files) == 0:
            print(f"Warning: No videos found in {source_dir}")
            continue

        # Split into train and test
        train_files, test_files = train_test_split(
            video_files,
            train_size=TRAIN_SPLIT,
            random_state=RANDOM_SEED,
            shuffle=True
        )

        print(f"\n{label.upper()}:")
        print(f"  Total: {len(video_files)} videos")
        print(f"  Train: {len(train_files)} videos")
        print(f"  Test:  {len(test_files)} videos")

        # Create directories
        train_dir = os.path.join(DATA_DIR, "train", label)
        test_dir = os.path.join(DATA_DIR, "test", label)
        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(test_dir, exist_ok=True)

        # Copy files to train directory
        for filename in train_files:
            src = os.path.join(source_dir, filename)
            dst = os.path.join(train_dir, filename)
            shutil.copy2(src, dst)

        # Copy files to test directory
        for filename in test_files:
            src = os.path.join(source_dir, filename)
            dst = os.path.join(test_dir, filename)
            shutil.copy2(src, dst)

    print("\n" + "="*50)
    print("Dataset split complete!")
    print(f"Train data: {os.path.join(DATA_DIR, 'train')}")
    print(f"Test data:  {os.path.join(DATA_DIR, 'test')}")
    print("="*50)

if __name__ == "__main__":
    print("="*50)
    print("Splitting dataset into train/test sets...")
    print("="*50)
    split_dataset()
