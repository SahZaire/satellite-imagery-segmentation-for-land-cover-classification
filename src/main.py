import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms as T
import segmentation_models_pytorch as smp
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
import gc
import cv2
from patchify import patchify
import albumentations as album
import shutil
import logging
from segmentation_models_pytorch import utils
from tiff_creator import TiffCreator
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader

torch.backends.cudnn.benchmark = True

logging.basicConfig(level=logging.INFO)

scaler = GradScaler()

def iou_score(outputs, targets):
    smooth = 1e-6
    outputs = outputs.sigmoid().data.cpu().numpy()
    targets = targets.data.cpu().numpy()
    np.seterr(divide='ignore', invalid='ignore')
    intersection = (outputs > 0.5) & (targets > 0.5)
    union = (outputs > 0.5) | (targets > 0.5)
    intersection = intersection.sum((1, 2))
    union = union.sum((1, 2))
    iou = (intersection + smooth) / (union + smooth)
    thresholded = np.ceil(np.clip(20 * (iou - 0.5), 0, 10)) / 10
    return thresholded.mean()

import os
import matplotlib.pyplot as plt
import torch
import numpy as np

def save_comparison_images(model, train_loader, device, save_dir=r'C:\A Drive\Machine Learning\ML Quick Projects for ISRO\Image Segmentation of Satellite Imagery for Land Cover Classification\Embedding\generated_images', num_images=5):
    """
    Save a canvas of original images, ground truth masks, and predicted masks for the train dataset.
    
    Args:
    model (torch.nn.Module): Trained model
    train_loader (DataLoader): DataLoader for train dataset
    device (torch.device): Device to run the model on
    save_dir (str): Directory to save the comparison images
    num_images (int): Number of images to save
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    model.eval()
    with torch.no_grad():
        for i, (images, masks) in enumerate(train_loader):
            if i >= num_images:
                break
            
            images = images.to(device)
            masks = masks.to(device)
            
            outputs = model(images)
            
            # Convert tensors to numpy arrays
            image = images[0].cpu().numpy().transpose(1, 2, 0)
            true_mask = masks[0].cpu().numpy().transpose(1, 2, 0)
            pred_mask = outputs[0].cpu().numpy().transpose(1, 2, 0)
            
            # Normalize image and masks for display
            image = (image - image.min()) / (image.max() - image.min())
            true_mask = np.argmax(true_mask, axis=2).astype(float)
            true_mask = (true_mask - true_mask.min()) / (true_mask.max() - true_mask.min())
            pred_mask = np.argmax(pred_mask, axis=2).astype(float)
            pred_mask = (pred_mask - pred_mask.min()) / (pred_mask.max() - pred_mask.min())
            
            # Create a figure with three subplots side by side
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
            
            # Plot original image
            ax1.imshow(image)
            ax1.set_title('Original Image')
            ax1.axis('off')
            
            # Plot ground truth mask
            ax2.imshow(true_mask, cmap='jet')
            ax2.set_title('Ground Truth')
            ax2.axis('off')
            
            # Plot predicted mask
            ax3.imshow(pred_mask, cmap='jet')
            ax3.set_title('Prediction')
            ax3.axis('off')
            
            # Save the figure
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f'comparison_{i}.png'))
            plt.close()

    print(f"Saved {num_images} comparison images in {save_dir}")

# Set device and environment variables
torch.backends.cudnn.benchmark = True
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Dataset paths
dataset_path = r'C:\A Drive\Machine Learning\ML Quick Projects for ISRO\Image Segmentation of Satellite Imagery for Land Cover Classification\Dataset'
train_dir = os.path.join(dataset_path, 'train')
test_dir = os.path.join(dataset_path, 'test')
valid_dir = os.path.join(dataset_path, 'valid')

def create_dataset_df(directory, has_masks=True):
    image_files = [f for f in os.listdir(directory) if f.endswith('_sat.jpg')]
    df = pd.DataFrame({
        'sat_image_path': [os.path.join(directory, f) for f in image_files],
    })
    
    if has_masks:
        df['mask_path'] = df['sat_image_path'].apply(lambda x: x.replace('_sat.jpg', '_mask.png'))
    else:
        df['mask_path'] = None
    
    return df

# Step 2: Load class dictionary
class_dict = pd.read_csv(os.path.join(dataset_path, 'class_dict.csv'))
class_names = class_dict['name'].tolist()
class_rgb_values = class_dict[['r', 'g', 'b']].values.tolist()

# Step 3: Helper functions for visualization and one-hot encoding/decoding
def one_hot_encode(label, label_values):
    """
    Convert a segmentation image label array to one-hot format
    """
    semantic_map = []
    for colour in label_values:
        equality = np.equal(label, colour)
        class_map = np.all(equality, axis=-1)
        semantic_map.append(class_map)
    semantic_map = np.stack(semantic_map, axis=-1)
    return semantic_map

def reverse_one_hot(image):
    """
    Transform a 2D array in one-hot format to a 2D array with class keys
    """
    x = np.argmax(image, axis=-1)
    return x

def colour_code_segmentation(image, label_values):
    """
    Given a 1-channel array of class keys, colour code the segmentation results
    """
    colour_codes = np.array(label_values)
    x = colour_codes[image.astype(int)]
    return x

# Step 4: Visualize Sample Image and Mask
def visualize(**images):
    """
    Plot images in one row
    """
    n_images = len(images)
    plt.figure(figsize=(20, 8))
    for idx, (name, image) in enumerate(images.items()):
        plt.subplot(1, n_images, idx + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(name.replace('_', ' ').title(), fontsize=20)
        plt.imshow(image)
    plt.show()

# Step 5: Defining Augmentations
def get_training_augmentation():
    train_transform = [
        album.RandomCrop(height=256, width=256, always_apply=True),
        album.HorizontalFlip(p=0.5),
        album.VerticalFlip(p=0.5),
    ]
    return album.Compose(train_transform)

def get_validation_augmentation():
    val_transform = [
        album.CenterCrop(height=256, width=256, always_apply=True),
    ]
    return album.Compose(val_transform)

def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype('float32')

def get_preprocessing(preprocessing_fn=None):
    _transform = []
    if preprocessing_fn:
        _transform.append(album.Lambda(image=preprocessing_fn))
    _transform.append(album.Lambda(image=to_tensor, mask=to_tensor))
    return album.Compose(_transform)

# Step 6: Dataset Class
class SatelliteDataset(Dataset):
    def __init__(self, images_dir, masks_dir, class_rgb_values, augmentation=None, preprocessing=None):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.class_rgb_values = class_rgb_values
        self.augmentation = augmentation
        self.preprocessing = preprocessing
        self.images_list = os.listdir(images_dir)

    def __len__(self):
        return len(self.images_list)

    def __getitem__(self, idx):
        image_name = self.images_list[idx]
        image_path = os.path.join(self.images_dir, image_name)
        mask_path = os.path.join(self.masks_dir, image_name)

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if os.path.exists(mask_path):
            mask = cv2.imread(mask_path)
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
        else:
            # If mask doesn't exist, create a blank mask
            mask = np.zeros_like(image)

        # One-hot encode the mask
        mask = one_hot_encode(mask, self.class_rgb_values).astype('float')

        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        return image, mask

# Step 7: Model Definition (DeepLabV3+)
def get_model(num_classes):
    model = smp.DeepLabV3Plus(
        encoder_name="resnet50",
        encoder_weights="imagenet",
        classes=num_classes,
        activation="sigmoid",
    )
    return model

# Step 8: Set Hyperparameters
ENCODER = 'resnet50'
ENCODER_WEIGHTS = 'imagenet'
CLASSES = class_names
ACTIVATION = 'sigmoid'
DEVICE = 'cuda'
BATCH_SIZE = 64
EPOCHS = 1
LEARNING_RATE = 0.0075
WORKERS = 16

# Step 9: Training and Evaluation Functions
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device):
    best_iou_score = 0.0
    train_logs_list, valid_logs_list = [], []

    for epoch in range(num_epochs):
        print(f'\nEpoch: {epoch+1}')
        model.train()
        train_loss = 0
        for images, masks in tqdm(train_loader):
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)

            optimizer.zero_grad()
            
            with autocast(device_type=device.type):
                outputs = model(images)
                loss = criterion(outputs, masks)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()

        train_loss /= len(train_loader)
        print(f'Train Loss: {train_loss:.4f}')

        model.eval()
        val_loss = 0
        val_iou_score = 0
        with torch.no_grad():
            for images, masks in tqdm(val_loader):
                images = images.to(device, non_blocking=True)
                masks = masks.to(device, non_blocking=True)

                with autocast(device_type=device.type):
                    outputs = model(images)
                    loss = criterion(outputs, masks)
                val_loss += loss.item()

                val_iou_score += iou_score(outputs, masks)

        val_loss /= len(val_loader)
        val_iou_score /= len(val_loader)
        print(f'Val Loss: {val_loss:.4f}, Val IoU Score: {val_iou_score:.4f}')

        train_logs_list.append(train_loss)
        valid_logs_list.append(val_loss)

        if val_iou_score > best_iou_score:
            best_iou_score = val_iou_score
            torch.save(model.state_dict(), 'best_model.pth')
            print('Model saved!')

    return train_logs_list, valid_logs_list

# Step 10: Main Function
def create_dataset_df(directory, has_masks=True):
    image_files = [f for f in os.listdir(directory) if f.endswith('_sat.jpg')]
    df = pd.DataFrame({
        'sat_image_path': [os.path.join(directory, f) for f in image_files],
    })
    
    if has_masks:
        df['mask_path'] = df['sat_image_path'].apply(lambda x: x.replace('_sat.jpg', '_mask.png'))
    else:
        df['mask_path'] = None
    
    return df

def main():
    # Create DataFrames for train, test, and valid sets
    train_df = create_dataset_df(train_dir, has_masks=True)
    test_df = create_dataset_df(test_dir, has_masks=False)
    valid_df = create_dataset_df(valid_dir, has_masks=False)

    # Create TIFF files for training set
    tiff_creator = TiffCreator(output_dir='tiff_files', patch_size=256)
    tiff_creator.create_tiff_files(train_df)

    # Prepare datasets
    train_dataset = SatelliteDataset(
        images_dir=tiff_creator.images_dir,
        masks_dir=tiff_creator.masks_dir,
        class_rgb_values=class_rgb_values,
        augmentation=get_training_augmentation(),
        preprocessing=get_preprocessing(),
    )

    val_dataset = SatelliteDataset(
        images_dir=tiff_creator.images_dir,
        masks_dir=tiff_creator.masks_dir,
        class_rgb_values=class_rgb_values,
        augmentation=get_validation_augmentation(),
        preprocessing=get_preprocessing(),
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=WORKERS, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=WORKERS, pin_memory=True)

    # Initialize model
    model = get_model(num_classes=len(CLASSES))
    model = model.to(DEVICE)

    # Define loss function and optimizer
    criterion = smp.utils.losses.DiceLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Train the model
    train_logs, valid_logs = train_model(model, train_loader, val_loader, criterion, optimizer, EPOCHS, device)

    # Plot training and validation loss
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, EPOCHS + 1), train_logs, label='Train Loss')
    plt.plot(range(1, EPOCHS + 1), valid_logs, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Losses')
    plt.legend()
    plt.savefig('loss_plot.png')
    plt.close()

    # Save comparison images
    save_comparison_images(model, train_loader, device, save_dir='comparison_images', num_images=10)

    # Evaluate on test set
    test_dataset = SatelliteDataset(
        images_dir='images256',
        masks_dir='masks256',
        class_rgb_values=class_rgb_values,
        augmentation=get_validation_augmentation(),
        preprocessing=get_preprocessing(),
    )
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=16, pin_memory=True)

    model.eval()
    test_iou_score = 0
    with torch.no_grad():
        for images, masks in tqdm(test_loader):
            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)
            test_iou_score += iou_score(outputs, masks).item()

    test_iou_score /= len(test_loader)
    print(f'Test IoU Score: {test_iou_score:.4f}')

if __name__ == '__main__':
    main()