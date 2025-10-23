import os
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

# ---------- CONFIG ----------
DATA_DIR = "data/processed"
BATCH_SIZE = 8
# ----------------------------

class ShopliftingDataset(Dataset):
    def __init__(self, split="train"):
        self.data = []
        self.labels = []
        split_dir = os.path.join(DATA_DIR, split)

        # Updated labels: "normal" = 0, "shoplifting" = 1
        for label_name, label_value in [("normal", 0), ("shoplifting", 1)]:
            folder = os.path.join(split_dir, label_name)
            if os.path.exists(folder):
                for file_name in os.listdir(folder):
                    if file_name.endswith(".npy"):
                        self.data.append(os.path.join(folder, file_name))
                        self.labels.append(label_value)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        clip = np.load(self.data[idx])  # shape: (frames, H, W, C)
        clip = np.transpose(clip, (3, 0, 1, 2))  # to (C, T, H, W)
        clip = torch.tensor(clip, dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return clip, label


def get_loaders(batch_size=BATCH_SIZE):
    train_set = ShopliftingDataset(split="train")
    test_set = ShopliftingDataset(split="test")

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader


if __name__ == "__main__":
    train_loader, test_loader = get_loaders()
    print(f"Train batches: {len(train_loader)}, Test batches: {len(test_loader)}")

    # test one batch
    for clips, labels in train_loader:
        print("Batch shape:", clips.shape)  # [B, C, T, H, W]
        print("Labels:", labels)
        break
