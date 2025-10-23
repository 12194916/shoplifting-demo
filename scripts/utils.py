import torch
import numpy as np

def preprocess_clip(frames):
    frames = np.stack(frames, axis=0)
    frames = frames.transpose((3, 0, 1, 2))  # (C, T, H, W)
    frames = frames / 255.0
    clip = torch.tensor(frames, dtype=torch.float32)
    return clip
