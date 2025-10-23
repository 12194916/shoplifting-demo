import os
import torch
import torch.nn as nn
import torch.optim as optim
from dataloader import get_loaders
from torchvision.models.video import r3d_18, R3D_18_Weights

# ---------- CONFIG ----------
EPOCHS = 10
LR = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SAVE_DIR = "models"
os.makedirs(SAVE_DIR, exist_ok=True)
# ----------------------------

print(f"Starting training on {DEVICE}...")

# Load data
train_loader, test_loader = get_loaders()
print(f"Train batches: {len(train_loader)}, Test batches: {len(test_loader)}")

# Model with pretrained weights
print("Loading pretrained R3D-18 model...")
model = r3d_18(weights=R3D_18_Weights.DEFAULT)  # use pretrained weights
model.fc = nn.Linear(model.fc.in_features, 2)  # 2 classes: shoplifting / non-shoplifting
model = model.to(DEVICE)
print("Pretrained model loaded successfully!")

# Loss & optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

# Track best model
best_val_acc = 0.0
best_model_path = os.path.join(SAVE_DIR, "best_shoplifting_detection.pth")

# ---------- TRAIN ----------
for epoch in range(EPOCHS):
    # Training phase
    model.train()
    total_loss = 0
    train_correct = 0
    train_total = 0

    for clips, labels in train_loader:
        clips, labels = clips.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(clips)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        train_total += labels.size(0)
        train_correct += (predicted == labels).sum().item()

    avg_loss = total_loss / len(train_loader)
    train_acc = train_correct / train_total

    # Validation phase
    model.eval()
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for clips, labels in test_loader:
            clips, labels = clips.to(DEVICE), labels.to(DEVICE)
            outputs = model(clips)
            _, predicted = torch.max(outputs, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()

    val_acc = val_correct / val_total

    # Save best model only
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), best_model_path)
        print(f"Epoch [{epoch+1}/{EPOCHS}] - Loss: {avg_loss:.4f} | Train Acc: {train_acc*100:.2f}% | Val Acc: {val_acc*100:.2f}% ✓ NEW BEST!")
    else:
        print(f"Epoch [{epoch+1}/{EPOCHS}] - Loss: {avg_loss:.4f} | Train Acc: {train_acc*100:.2f}% | Val Acc: {val_acc*100:.2f}%")

# ---------- FINAL EVALUATION ----------
print("\n" + "="*50)
print("Training Complete!")
print(f"Best Validation Accuracy: {best_val_acc*100:.2f}%")
print(f"Best model saved to: {best_model_path}")
print("="*50)
