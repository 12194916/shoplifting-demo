# Shoplifting Detection System Demo 

A deep learning-based video classification system for detecting shoplifting behavior in surveillance footage using 3D Convolutional Neural Networks (R3D-18).

## 🎯 Overview

This project uses a pretrained R3D-18 model (3D ResNet) to analyze surveillance video footage and classify behaviors as either **shoplifting** or **normal** activity. The system processes video clips frame-by-frame and provides real-time predictions.

## 🎥 Demo Videos

Here are sample detection results:

### Shoplifting Detection Example
![Shoplifting Detection](gifs/shoplifting.gif)


> **Note:** Demo videos showcase the model's real-time inference capabilities with overlaid predictions and confidence scores.

## 🏗️ Architecture

- **Model**: R3D-18 (3D ResNet-18) pretrained on Kinetics-400
- **Input**: Video clips (16 frames, 224x224 resolution)
- **Output**: Binary classification (shoplifting vs. non-shoplifting)
- **Framework**: PyTorch with CUDA support

## 📁 Project Structure

```
shop-lifting/
├── data/                   # Raw video files (not tracked in git)
│   ├── normal/             # 90 normal behavior videos
│   ├── shoplifting/        # 93 shoplifting behavior videos
│   ├── train/              # Created by split_data.py
│   │   ├── normal/
│   │   └── shoplifting/
│   ├── test/               # Created by split_data.py
│   │   ├── normal/
│   │   └── shoplifting/
│   └── processed/          # Created by preprocess.py
│       ├── train/
│       │   ├── normal/     # .npy files
│       │   └── shoplifting/
│       └── test/
│           ├── normal/
│           └── shoplifting/
├── models/                 # Trained model checkpoints
│   └── best_shoplifting_detection.pth
├── scripts/
│   ├── split_data.py       # Split dataset into train/test
│   ├── preprocess.py       # Video preprocessing
│   ├── dataloader.py       # PyTorch dataset loader
│   ├── model.py            # Model definition
│   ├── train.py            # Training script
│   ├── test.py             # Evaluation script
│   ├── inference.py        # Real-time inference
│   └── utils.py            # Utility functions
├── .gitignore
├── requirements.txt
└── README.md
```

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended)

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/your-username/shop-lifting.git
cd shop-lifting
```

2. **Create virtual environment**
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate
```

3. **Install dependencies**
```bash
# Install PyTorch with CUDA support (for CUDA 11.8)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.x, use:
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# For CPU only:
# pip install torch torchvision

# Install other dependencies
pip install -r requirements.txt
```

4. **Prepare your dataset**

   Place your video files in the following structure:
   ```
   data/
   ├── normal/*.mp4        # Normal behavior videos
   └── shoplifting/*.mp4   # Shoplifting behavior videos
   ```

## 📊 Complete Training Workflow

### Step 1: Split Dataset into Train/Test

First, split your videos into training (80%) and testing (20%) sets:

```bash
cd scripts
python split_data.py
```

**What it does:**
- Randomly splits videos into 80% train / 20% test
- Creates `data/train/` and `data/test/` directories
- Maintains class balance in both splits

**Output structure:**
```
data/
├── train/
│   ├── normal/      (~72 videos)
│   └── shoplifting/ (~74 videos)
└── test/
    ├── normal/      (~18 videos)
    └── shoplifting/ (~19 videos)
```

> **Note:** After splitting, you can optionally delete the original `data/normal/` and `data/shoplifting/` folders to save space.

### Step 2: Preprocess Videos

Convert raw video files to preprocessed numpy arrays:

```bash
python preprocess.py
```

**What it does:**
- Extracts 16 evenly-spaced frames from each video
- Resizes frames to 224x224 pixels
- Normalizes pixel values to [0, 1]
- Saves as `.npy` files in `data/processed/`

**Processing time:** ~5-10 seconds per video (depends on video length and hardware)

### Step 3: Train Model

```bash
python train.py
```

**Training features:**
- Uses pretrained R3D-18 weights from Kinetics-400
- Logs training & validation accuracy per epoch
- Saves only the best model based on validation accuracy
- Automatically uses GPU if available

**Output:**
```
Starting training on cuda...
Train batches: 18, Test batches: 5
Loading pretrained R3D-18 model...

Epoch [1/50] - Loss: 0.4521 | Train Acc: 78.50% | Val Acc: 82.30% ✓ NEW BEST!
Epoch [2/50] - Loss: 0.3214 | Train Acc: 85.20% | Val Acc: 88.45% ✓ NEW BEST!
Epoch [3/50] - Loss: 0.2891 | Train Acc: 87.10% | Val Acc: 87.20%
...
==================================================
Training Complete!
Best Validation Accuracy: 95.50%
Best model saved to: models/best_shoplifting_detection.pth
==================================================
```

## 🔍 Inference

### Quick Test

Test on a single video with visualization:

```bash
python test.py
```

Edit the `video_path` in [test.py](scripts/test.py:16) to test different videos.

### Full Inference

Run inference on any video file and save the output:

```bash
# Basic usage - saves to output.mp4
python inference.py path/to/video.mp4

# Example with test video
python inference.py ../data/test/shoplifting/shoplifting-1.mp4

# Custom output filename
python inference.py video.mp4 --save result.mp4

# Use specific model
python inference.py video.mp4 --model ../models/best_shoplifting_detection.pth
```

**Supported inputs:**
- Video files (`.mp4`, `.avi`, `.mkv`)
- RTSP camera streams
- Webcam (use `0` as video path)

**Output:**
- Video with overlaid predictions
- Color-coded labels (red = SHOPLIFTING, green = NORMAL)
- Real-time confidence percentage displayed

## 📝 Model Details

### Preprocessing
- Frame extraction: 16 frames per clip
- Frame resolution: 224x224 pixels
- Normalization: pixel values scaled to [0, 1]
- Format: (C, T, H, W) tensor

### Training Configuration
- **Epochs**: 50
- **Learning rate**: 1e-4
- **Optimizer**: Adam
- **Loss function**: Cross-entropy
- **Batch size**: 8

### Data Format
- **Input**: Preprocessed `.npy` files containing frame arrays
- **Labels**: 0 = normal, 1 = shoplifting

## 🛠️ Scripts Overview

| Script | Description |
|--------|-------------|
| `split_data.py` | **[Step 1]** Split raw videos into train/test sets (80/20) |
| `preprocess.py` | **[Step 2]** Extract and preprocess frames from videos |
| `train.py` | **[Step 3]** Training loop with validation and checkpointing |
| `inference.py` | **[Step 4]** Real-time inference on videos/streams |
| `test.py` | Quick testing on single video with visualization |
| `dataloader.py` | PyTorch Dataset and DataLoader for training |
| `model.py` | R3D-18 model architecture definition |
| `utils.py` | Helper functions for preprocessing |

## ⚙️ Configuration

Key parameters can be modified in respective scripts:

**Data Split (`split_data.py`):**
- `TRAIN_SPLIT`: Train/test ratio (default: 0.8)
- `RANDOM_SEED`: For reproducible splits (default: 42)

**Preprocessing (`preprocess.py`):**
- `FRAME_SIZE`: Frame dimensions (default: 224x224)
- `FRAMES_PER_CLIP`: Frames per video clip (default: 16)

**Training (`train.py`):**
- `EPOCHS`: Number of training epochs (default: 50)
- `LR`: Learning rate (default: 1e-4)
- `BATCH_SIZE`: In dataloader.py (default: 8)
- `DEVICE`: Automatically selects "cuda" or "cpu"

**Inference (`inference.py`):**
- `MODEL_PATH`: Path to trained model
- `CLIP_LENGTH`: Number of frames per prediction (default: 16)

## 📦 Requirements

```
torch>=2.0.0
torchvision>=0.15.0
opencv-python>=4.8.0
numpy>=1.24.0
tqdm>=4.65.0
scikit-learn>=1.3.0
```

Install all at once:
```bash
pip install -r requirements.txt
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

## 📄 License

This project is available for educational and research purposes.

## 🙏 Acknowledgments

- Pretrained R3D-18 model from [torchvision](https://pytorch.org/vision/stable/models.html)
- Kinetics-400 dataset for pretraining

## 📧 Contact

For questions or feedback, please open an issue on GitHub.

## 📈 Dataset Information

**Current Dataset:**
- Total videos: 183 (90 normal + 93 shoplifting)
- After 80/20 split:
  - Training: ~146 videos (72 normal + 74 shoplifting)
  - Testing: ~37 videos (18 normal + 19 videos)

**Data not tracked in git:**
- Raw video files are excluded via `.gitignore` due to large file sizes
- Preprocessed `.npy` files are also excluded
- Only scripts and trained model are version controlled

---

**Note:** Raw video data is not included in this repository due to size constraints. You need to provide your own dataset in the `data/normal/` and `data/shoplifting/` folders.
