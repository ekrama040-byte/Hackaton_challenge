import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split
import torchvision.models as models

def execute_network_training(epochs=5, batch_size=32, learning_rate=0.001):
    
    data_dir = "dataset/train"

    save_dir = "models"
    save_path = os.path.join(save_dir, "plant_disease_model.pth")
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    transform_pipeline = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    if not os.path.exists(data_dir) or len(os.listdir(data_dir)) == 0:
        print(f"❌ Error: Place your class subdirectories inside: '{data_dir}' before running.")
        return

    dataset = ImageFolder(root=data_dir, transform=transform_pipeline)
    num_classes = len(dataset.classes)
    print(f"📚 Dataset compiled successfully. Found {len(dataset)} images across {num_classes} folders.")

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_set, val_set = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Initializing Backpropagation Loop on device hardware: {device}")

    weights = models.MobileNet_V3_Small_Weights.DEFAULT
    model = models.mobilenet_v3_small(weights=weights)
    
   
       # Fetch features from the classifier block using your defined variable name
    in_features = model.classifier[0].in_features

    
    model.classifier = nn.Sequential(
        nn.Linear(in_features, 1024),
        nn.Hardswish(),
        nn.Dropout(p=0.2, inplace=True),
        nn.Linear(1024, num_classes)
    )

    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    best_acc = 0.0
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_acc = correct / total
        epoch_loss = running_loss / len(train_set)
        print(f"Epoch [{epoch+1}/{epochs}] - Loss: {epoch_loss:.4f} - Val Accuracy: {val_acc*100:.2f}%")

        if val_acc >= best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), save_path)
            print(f"🎯 State matrix weights exported successfully to {save_path}")

    print("🏁 Neural network optimization cycle completed.")

if __name__ == "__main__":
    execute_network_training()

