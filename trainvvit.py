import os
import random
import torch
import decord
import torch.nn as nn
import timm
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# Dataset Class
class VideoDataset(Dataset):
    def __init__(self, video_folder, classes, clip_length=16, transform=None):
        self.video_paths = []
        self.labels = []
        self.clip_length = clip_length
        self.classes = classes
        self.class_to_idx = {cls: idx for idx, cls in enumerate(classes)}

        for cls in classes:
            cls_folder = os.path.join(video_folder, cls)
            for video_file in os.listdir(cls_folder):
                if video_file.endswith(('.mp4', '.avi')):
                    self.video_paths.append(os.path.join(cls_folder, video_file))
                    self.labels.append(self.class_to_idx[cls])

        # Transformation to resize frames to consistent size (224x224)
        if transform is None:
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),
                transforms.ToTensor()
            ])
        else:
            self.transform = transform

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        label = self.labels[idx]

        # Load video using Decord
        vr = decord.VideoReader(video_path)
        frame_count = len(vr)

        # Handle the case where video has fewer frames than required clip length
        if frame_count < self.clip_length:
            indices = list(range(frame_count))  # Use all available frames
            # Repeat frames to reach the required clip length
            while len(indices) < self.clip_length:
                indices.extend(indices[:self.clip_length - len(indices)])
        else:
            # Randomly select clip_length frames from the video
            indices = sorted(random.sample(range(frame_count), self.clip_length))

        frames = [vr[i].asnumpy() for i in indices]
        frames = [self.transform(frame) for frame in frames]
        frames = torch.stack(frames)

        # Average over the temporal dimension to create a single image
        averaged_frame = torch.mean(frames, dim=0)

        return averaged_frame, label

# Set paths to dataset folders
train_folder = "vivit_dataset/train"
val_folder = "vivit_dataset/validation"
test_folder = "vivit_dataset/test"

# Define classes
classes = ['real', 'fake']

# Create dataset loaders
train_dataset = VideoDataset(train_folder, classes)
val_dataset = VideoDataset(val_folder, classes)
test_dataset = VideoDataset(test_folder, classes)

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)

# Load pre-trained ViT model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=len(classes))
model = model.to(device)

# Define loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# Training Loop
num_epochs = 1

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (videos, labels) in enumerate(train_loader):
        videos, labels = videos.to(device), labels.to(device)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(videos)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # Calculate statistics
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        if batch_idx % 10 == 0:
            print(f"Epoch [{epoch + 1}/{num_epochs}], Step [{batch_idx}/{len(train_loader)}], Loss: {loss.item():.4f}")

    train_accuracy = 100 * correct / total
    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(train_loader):.4f}, Train Accuracy: {train_accuracy:.2f}%")

    # Validation phase
    model.eval()
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for videos, labels in val_loader:
            videos, labels = videos.to(device), labels.to(device)
            outputs = model(videos)
            _, predicted = torch.max(outputs.data, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()

    val_accuracy = 100 * val_correct / val_total
    print(f"Validation Accuracy: {val_accuracy:.2f}%")

# Save the trained model
torch.save(model.state_dict(), 'vivit_real_fake_classification.pth')

# Testing the Model
model.load_state_dict(torch.load('vivit_real_fake_classification.pth'))
model.eval()

# Test model
test_correct = 0
test_total = 0
with torch.no_grad():
    for videos, labels in test_loader:
        videos, labels = videos.to(device), labels.to(device)
        outputs = model(videos)
        _, predicted = torch.max(outputs.data, 1)
        test_total += labels.size(0)
        test_correct += (predicted == labels).sum().item()

test_accuracy = 100 * test_correct / test_total
print(f"Test Accuracy: {test_accuracy:.2f}%")
