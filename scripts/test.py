import torch
import cv2
from torchvision.models.video import r3d_18
from utils import preprocess_clip

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load the model exactly like training
model = r3d_18(weights=None)
model.fc = torch.nn.Linear(model.fc.in_features, 2)
model.load_state_dict(torch.load("models/best_shoplifting_detection.pth", map_location=device))
model = model.to(device)
model.eval()

# Update this path to a video from your test set
video_path = "data/test/shoplifting/shoplifting-1.mp4"
cap = cv2.VideoCapture(video_path)

frames = []
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frames.append(cv2.resize(frame, (112, 112)))
    if len(frames) == 16:
        clip = preprocess_clip(frames)
        with torch.no_grad():
            output = model(clip.unsqueeze(0).to(device))
            pred = torch.argmax(output).item()
        label = "SHOPLIFTING" if pred == 1 else "NON-SHOPLIFTING"

        # Print to terminal
        print(label)

        # Show on video
        cv2.putText(frame, label, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
        cv2.imshow("Result", frame)

        # Slow down display: 200 ms per frame
        if cv2.waitKey(200) & 0xFF == ord('q'):
            break

        frames = []

cap.release()
cv2.destroyAllWindows()
