import torch
import cv2
import numpy as np
import argparse
from torchvision.models.video import r3d_18, R3D_18_Weights

# ---------- CONFIG ----------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = "models/best_shoplifting_detection.pth"
FRAME_SIZE = (224, 224)
CLIP_LENGTH = 16
# ----------------------------

def load_model(model_path):
    """Load trained model"""
    model = r3d_18(weights=R3D_18_Weights.DEFAULT)
    model.fc = torch.nn.Linear(model.fc.in_features, 2)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model = model.to(DEVICE)
    model.eval()
    print(f"✓ Model loaded from {model_path}")
    return model

def preprocess_clip(frames):
    """Convert frames to model input format"""
    # frames: list of numpy arrays (H, W, C)
    frames = np.stack(frames, axis=0)  # (T, H, W, C)
    frames = frames / 255.0  # normalize
    frames = frames.transpose((3, 0, 1, 2))  # (C, T, H, W)
    clip = torch.tensor(frames, dtype=torch.float32)
    return clip

def run_inference(video_source, model, save_output=None):
    """
    Run real-time inference on video

    Args:
        video_source: video file path or camera index (0 for webcam)
        model: loaded PyTorch model
        save_output: path to save output video (optional)
    """
    cap = cv2.VideoCapture(video_source)

    if not cap.isOpened():
        print(f"✗ Error: Cannot open video source: {video_source}")
        return

    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"✓ Video opened: {width}x{height} @ {fps} FPS")

    # Video writer for saving output
    writer = None
    if save_output:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(save_output, fourcc, fps, (width, height))
        print(f"✓ Saving output to {save_output}")

    frame_buffer = []
    frame_count = 0
    prediction = "UNKNOWN"
    confidence = 0.0

    print("\nProcessing video... Press 'q' to quit")
    print("-" * 50)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # Resize frame for model input
        resized_frame = cv2.resize(frame, FRAME_SIZE)
        frame_buffer.append(resized_frame)

        # Keep only last CLIP_LENGTH frames
        if len(frame_buffer) > CLIP_LENGTH:
            frame_buffer.pop(0)

        # Run inference when we have enough frames
        if len(frame_buffer) == CLIP_LENGTH:
            clip = preprocess_clip(frame_buffer)

            with torch.no_grad():
                output = model(clip.unsqueeze(0).to(DEVICE))
                probs = torch.softmax(output, dim=1)
                pred_idx = torch.argmax(probs, dim=1).item()
                confidence = probs[0][pred_idx].item()

            prediction = "SHOPLIFTING" if pred_idx == 1 else "NORMAL"

        # Draw prediction on frame
        if len(frame_buffer) == CLIP_LENGTH:
            # Background box for text
            cv2.rectangle(frame, (10, 10), (450, 80), (0, 0, 0), -1)

            # Prediction text
            color = (0, 0, 255) if prediction == "SHOPLIFTING" else (0, 255, 0)
            cv2.putText(frame, f"Status: {prediction}", (20, 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)

            # Confidence
            cv2.putText(frame, f"Confidence: {confidence*100:.1f}%", (20, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        else:
            # Loading message
            cv2.putText(frame, f"Loading... {len(frame_buffer)}/{CLIP_LENGTH}", (20, 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

        # Save frame if writer is active
        if writer:
            writer.write(frame)

    # Cleanup
    cap.release()
    if writer:
        writer.release()

    print(f"\n✓ Processed {frame_count} frames")
    if save_output:
        print(f"✓ Output saved to {save_output}")

def main():
    parser = argparse.ArgumentParser(description="Real-time shoplifting detection")
    parser.add_argument("video", type=str, help="Video file path or RTSP URL")
    parser.add_argument("--model", type=str, default=MODEL_PATH,
                       help="Path to model weights")
    parser.add_argument("--save", type=str, default="output.mp4",
                       help="Save output video to path (default: output.mp4)")

    args = parser.parse_args()

    print("=" * 50)
    print("  Shoplifting Detection - Inference")
    print("=" * 50)
    print(f"Device: {DEVICE}")
    print(f"Video source: {args.video}")

    # Load model
    model = load_model(args.model)

    # Run inference
    run_inference(args.video, model, save_output=args.save)

    print("\n✓ Done!")

if __name__ == "__main__":
    main()
