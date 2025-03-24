import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
import torch.multiprocessing as mp
import timm
import sys
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader
from PIL import Image
from torchvision import datasets
import random
#  Avoid garbled output
sys.stdout.reconfigure(encoding='utf-8')
cudnn.benchmark = True  #  cuDNN acceleration

#  Set device (GPU or CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def mixup_data(x, y, alpha=0.5):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0.0:
        lam = random.betavariate(alpha, alpha)
    else:
        lam = 1.0

    batch_size = x.size(0)
    index = torch.randperm(batch_size).cuda()  # 隨機打亂索引

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    '''Calculates the mixup loss using the weighted labels'''
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

#  Custom ImageFolder to ensure RGB format
class RGBImageFolder(datasets.ImageFolder):
    def __getitem__(self, index):
        path, target = self.samples[index]
        image = Image.open(path).convert("RGB")  
        if self.transform:
            image = self.transform(image)
        return image, target

#  Set hyperparameters
num_epochs = 60
max_lr = 0.00015
patience = 10 #  Early Stopping patience value
verbose = True

#  Train function
def train(model, train_loader, criterion, optimizer, device, scaler):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    """
    for images, labels in train_loader:
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)

        optimizer.zero_grad()
        with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
            outputs = model(images)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)
        """
    for images, labels in train_loader:
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        # Apply mixup
        mixed_images, targets_a, targets_b, lam = mixup_data(images, labels, 0.5)
        optimizer.zero_grad()
        
        with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
            outputs = model(mixed_images)
            loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        correct += predicted.eq(targets_a).sum().item()  # We use targets_a as the ground truth
        total += labels.size(0)
        

    accuracy = 100 * correct / total
    return running_loss / len(train_loader), accuracy

#  Evaluation function
def evaluate(model, val_loader, criterion, device):
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    all_labels = []
    all_preds = []
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())
    accuracy = 100 * correct / total
    return val_loss / len(val_loader), accuracy, all_labels, all_preds

def plot_confusion_matrix(cm, classes, save_path):
    # Plot the confusion matrix
    plt.figure(figsize=(20, 20))
    sns.heatmap(cm, annot=False, fmt='g', cmap="Blues", xticklabels=range(len(classes)), yticklabels=range(len(classes)))
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.title('Confusion Matrix')
    
    # Save the confusion matrix as a file
    plt.savefig(save_path)  # Save as an image file
    plt.close()  # Close the plot to free up memory

if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)

    #  Data Augmentation
    train_transform = transforms.Compose([  # Data augmentation
        transforms.RandomResizedCrop(400, scale=(0.5, 1.0)),  # Increase range of scaling
        transforms.Resize((256, 256)),
        transforms.ColorJitter(brightness=0.4, contrast=0.5, saturation=0.5, hue=0.3),  # Stronger color jitter
        transforms.RandomHorizontalFlip(p=0.5),  # Increase flip probability
        transforms.RandomRotation(45),  # Increased rotation range
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=10),  # Add affine transform for slight changes in perspective
        transforms.RandomPerspective(distortion_scale=0.3, p=0.5, interpolation=3),  # Add perspective distortion for more variation
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.4, scale=(0.02, 0.25))  # Increase probability and area for random erasing
    ])

    val_transform = transforms.Compose([
        transforms.Resize((320, 320)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = RGBImageFolder(root="data_temp/train", transform=train_transform)
    val_dataset = RGBImageFolder(root="data_temp/val", transform=val_transform)

    num_workers = 10
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0, pin_memory=True)

    num_classes = len(train_dataset.classes)

    #  Load pretrained model from timm library
    model = timm.create_model('seresnet152d.ra2_in1k', pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(num_ftrs, num_classes)
    )
    """ for continue training
    state_dict = torch.load("best_resnet_model_epoch_21.pth", map_location=device, weights_only=True) 
    """
    model = model.to(device)

    #  Set loss function and optimizer
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=max_lr, weight_decay=1e-2)

    scaler = torch.amp.GradScaler()
    
    best_val_loss = 0.0
    best_model_state = None
    bestcm = None
    
    early_stop_counter = 0
    no_improvement_count = 0

    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs} | LR: {optimizer.param_groups[0]['lr']:.9f}")
        train_loss, train_acc = train(model, train_loader, criterion, optimizer, device, scaler)
        val_loss, val_acc, all_labels, all_preds = evaluate(model, val_loader, criterion, device)

        print(f"Epoch {epoch+1}/{num_epochs} | Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}% | Val Loss: {val_loss:.5f}")

        # Save best model weights
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict()  # Save only the weights
            torch.save(best_model_state, f"best_resnet_model_epoch_{epoch+1}.pth")
            bestcm = confusion_matrix(all_labels, all_preds)
            save_path = f"confusion_matrix{epoch+1}.png"  # Save confusion matrix as a PNG file
            plot_confusion_matrix(bestcm, classes=train_dataset.classes, save_path=save_path)
            print(f" Confusion Matrix saved at: {save_path}")
            early_stop_counter = 0
        else:
            early_stop_counter += 1

        # Early stopping
        if early_stop_counter >= patience:
            print("Early Stopping Activated!")
            break
        # Manually adjust the learning rate if no improvement
        if(early_stop_counter%4==0 and early_stop_counter!=0):
            optimizer.param_groups[0]['lr'] = optimizer.param_groups[0]['lr'] * 0.6

    # After training, save the best model
    if best_model_state:
        torch.save(best_model_state, "best_resnet_model.pth")
        print(f"\n  Best model weights saved! Validation Acc: {best_val_acc:.2f}%")

        # Generate and plot confusion matrix after training
    save_path = "confusion_matrix.png"  # Save confusion matrix as a PNG file
    plot_confusion_matrix(bestcm, classes=train_dataset.classes, save_path=save_path)
    print(f" Confusion Matrix saved at: {save_path}")
