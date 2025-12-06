import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from sklearn.model_selection import train_test_split
import cv2
import os
import matplotlib.pyplot as plt
from datetime import datetime
import mediapipe as mp
from tqdm import tqdm

class HandGestureDataset(Dataset):
    """PyTorch Dataset for hand gesture images."""
    
    def __init__(self, images, labels, transform=None):
        """
        Initialize dataset.
        
        Args:
            images: Numpy array of images
            labels: Numpy array of labels
            transform: Optional transforms to apply
        """
        self.images = images
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        
        # Convert to PIL Image for transforms
        if self.transform:
            image = self.transform(image)
        else:
            # Default: convert to tensor and normalize
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        
        return image, torch.tensor(label, dtype=torch.float32)


class MediaPipeHandGestureCNN:
    """Fine-tune EfficientNet with MediaPipe hand preprocessing using PyTorch."""
    
    def __init__(self, img_size=(224, 224), use_hand_crop=True, device=None):
        """
        Initialize with pretrained EfficientNet and MediaPipe Hands.
        
        Args:
            img_size: Input image size (width, height)
            use_hand_crop: Whether to crop and focus on detected hands
            device: Device to use (cuda/cpu)
        """
        self.img_size = img_size
        self.use_hand_crop = use_hand_crop
        self.model = None
        self.history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [],
                       'train_precision': [], 'train_recall': [], 
                       'val_precision': [], 'val_recall': []}
        self.class_names = ['unpinched', 'pinched']
        
        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        print(f"Using device: {self.device}")
        
        # Initialize MediaPipe Hands for preprocessing
        if use_hand_crop:
            self.mp_hands = mp.solutions.hands
            self.hands = self.mp_hands.Hands(
                static_image_mode=True,
                max_num_hands=1,
                min_detection_confidence=0.5
            )
            print("MediaPipe Hands initialized for hand detection and cropping")
    
    def preprocess_with_mediapipe(self, image):
        """
        Use MediaPipe to detect and crop hand region.
        
        Args:
            image: Input image (BGR format)
            
        Returns:
            Cropped hand image or original if no hand detected
        """
        if not self.use_hand_crop:
            return image
        
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Process image with MediaPipe
        results = self.hands.process(image_rgb)
        
        if results.multi_hand_landmarks:
            # Get hand bounding box
            hand_landmarks = results.multi_hand_landmarks[0]
            
            h, w, _ = image.shape
            x_coords = [landmark.x * w for landmark in hand_landmarks.landmark]
            y_coords = [landmark.y * h for landmark in hand_landmarks.landmark]
            
            # Add padding around hand
            padding = 30
            x_min = max(0, int(min(x_coords)) - padding)
            x_max = min(w, int(max(x_coords)) + padding)
            y_min = max(0, int(min(y_coords)) - padding)
            y_max = min(h, int(max(y_coords)) + padding)
            
            # Crop hand region
            hand_crop = image[y_min:y_max, x_min:x_max]
            
            if hand_crop.size > 0:
                return hand_crop
        
        return image
    
    def load_dataset(self, data_dir='data'):
        """
        Load images with MediaPipe hand preprocessing.
        
        Args:
            data_dir: Root directory containing class folders
            
        Returns:
            Tuple of (images, labels)
        """
        images = []
        labels = []
        
        print("Loading dataset with MediaPipe hand detection...")
        
        for label, class_name in enumerate(self.class_names):
            class_dir = os.path.join(data_dir, class_name)
            
            if not os.path.exists(class_dir):
                print(f"Warning: Directory {class_dir} not found")
                continue
            
            image_files = [f for f in os.listdir(class_dir) 
                          if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            
            print(f"Processing {len(image_files)} images from {class_name}...")
            
            for img_file in tqdm(image_files, desc=f"Loading {class_name}"):
                img_path = os.path.join(class_dir, img_file)
                img = cv2.imread(img_path)
                
                if img is None:
                    continue
                
                # Preprocess with MediaPipe to focus on hand
                img = self.preprocess_with_mediapipe(img)
                
                # Resize to model input size
                img = cv2.resize(img, self.img_size)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                images.append(img)
                labels.append(label)
        
        images = np.array(images, dtype=np.float32)
        labels = np.array(labels, dtype=np.int32)
        
        print(f"\nTotal images loaded: {len(images)}")
        print(f"Unpinched: {np.sum(labels == 0)}, Pinched: {np.sum(labels == 1)}")
        
        return images, labels
    
    def build_model(self, freeze_base=True):
        """
        Build model with EfficientNet-B0 base and custom head.
        
        Args:
            freeze_base: Whether to freeze base model weights initially
        """
        # Load pretrained EfficientNet-B0
        base_model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        
        # Freeze base model if requested
        if freeze_base:
            for param in base_model.parameters():
                param.requires_grad = False
        
        # Replace classifier with custom head
        num_features = base_model.classifier[1].in_features
        
        base_model.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(num_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        
        self.model = base_model.to(self.device)
        
        print(f"\nModel built with EfficientNet-B0 base")
        print(f"MediaPipe hand preprocessing: {'Enabled' if self.use_hand_crop else 'Disabled'}")
        print(f"Base model frozen: {freeze_base}")
        
        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        
        return self.model
    
    def unfreeze_base_model(self, layers_to_unfreeze=40):
        """
        Unfreeze base model layers for fine-tuning.
        
        Args:
            layers_to_unfreeze: Number of layers to unfreeze from the end
        """
        if self.model is None:
            print("Build model first")
            return
        
        # Get all parameters except classifier
        features_params = list(self.model.features.parameters())
        
        # Unfreeze last N layers
        for param in features_params[-layers_to_unfreeze:]:
            param.requires_grad = True
        
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Unfroze last {layers_to_unfreeze} layers")
        print(f"Trainable parameters: {trainable_params:,}")
    
    def get_transforms(self, augment=True):
        """Get data transforms for training and validation."""
        if augment:
            train_transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(20),
                transforms.RandomResizedCrop(self.img_size, scale=(0.8, 1.0)),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                   std=[0.229, 0.224, 0.225])
            ])
        else:
            train_transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                   std=[0.229, 0.224, 0.225])
            ])
        
        val_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        return train_transform, val_transform
    
    def calculate_metrics(self, outputs, labels):
        """Calculate precision and recall."""
        preds = (outputs > 0.5).float()
        
        tp = ((preds == 1) & (labels == 1)).sum().item()
        fp = ((preds == 1) & (labels == 0)).sum().item()
        fn = ((preds == 0) & (labels == 1)).sum().item()
        
        precision = tp / (tp + fp + 1e-7)
        recall = tp / (tp + fn + 1e-7)
        
        return precision, recall
    
    def train(self, X_train, y_train, X_val, y_val, 
              epochs=30, batch_size=32, augment=True, learning_rate=0.001):
        """
        Train the model.
        
        Args:
            X_train: Training images
            y_train: Training labels
            X_val: Validation images
            y_val: Validation labels
            epochs: Number of training epochs
            batch_size: Batch size for training
            augment: Whether to use data augmentation
            learning_rate: Learning rate
        """
        if self.model is None:
            self.build_model()
        
        # Get transforms
        train_transform, val_transform = self.get_transforms(augment)
        
        # Create datasets
        train_dataset = HandGestureDataset(X_train, y_train, transform=train_transform)
        val_dataset = HandGestureDataset(X_val, y_val, transform=val_transform)
        
        # Create dataloaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, 
                                 shuffle=True, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, 
                               shuffle=False, num_workers=0)
        
        # Loss and optimizer
        criterion = nn.BCELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', 
                                                         factor=0.5, patience=5)
        
        print("\nStarting training...")
        print(f"Training samples: {len(X_train)}")
        print(f"Validation samples: {len(X_val)}")
        print(f"Batch size: {batch_size}")
        print(f"Learning rate: {learning_rate}")
        
        best_val_loss = float('inf')
        patience_counter = 0
        patience = 10
        
        for epoch in range(epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            train_precision_sum = 0
            train_recall_sum = 0
            
            train_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs} [Train]')
            for inputs, labels in train_bar:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                labels = labels.unsqueeze(1)
                
                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                preds = (outputs > 0.5).float()
                train_correct += (preds == labels).sum().item()
                train_total += labels.size(0)
                
                precision, recall = self.calculate_metrics(outputs, labels)
                train_precision_sum += precision
                train_recall_sum += recall
                
                train_bar.set_postfix({'loss': loss.item(), 
                                      'acc': train_correct / train_total})
            
            # Validation phase
            self.model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            val_precision_sum = 0
            val_recall_sum = 0
            
            with torch.no_grad():
                val_bar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{epochs} [Val]')
                for inputs, labels in val_bar:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    labels = labels.unsqueeze(1)
                    
                    outputs = self.model(inputs)
                    loss = criterion(outputs, labels)
                    
                    val_loss += loss.item()
                    preds = (outputs > 0.5).float()
                    val_correct += (preds == labels).sum().item()
                    val_total += labels.size(0)
                    
                    precision, recall = self.calculate_metrics(outputs, labels)
                    val_precision_sum += precision
                    val_recall_sum += recall
                    
                    val_bar.set_postfix({'loss': loss.item(), 
                                        'acc': val_correct / val_total})
            
            # Calculate epoch metrics
            train_loss = train_loss / len(train_loader)
            train_acc = train_correct / train_total
            val_loss = val_loss / len(val_loader)
            val_acc = val_correct / val_total
            
            train_precision = train_precision_sum / len(train_loader)
            train_recall = train_recall_sum / len(train_loader)
            val_precision = val_precision_sum / len(val_loader)
            val_recall = val_recall_sum / len(val_loader)
            
            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['train_precision'].append(train_precision)
            self.history['train_recall'].append(train_recall)
            self.history['val_precision'].append(val_precision)
            self.history['val_recall'].append(val_recall)
            
            print(f'\nEpoch {epoch+1}/{epochs}:')
            print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}')
            print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')
            print(f'Val Precision: {val_precision:.4f}, Val Recall: {val_recall:.4f}')
            
            # Learning rate scheduling
            scheduler.step(val_loss)
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(self.model.state_dict(), 'best_model_checkpoint.pth')
                print('Saved best model')
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f'\nEarly stopping triggered after {epoch+1} epochs')
                    break
        
        # Load best model
        self.model.load_state_dict(torch.load('best_model_checkpoint.pth'))
        print('\nLoaded best model weights')
    
    def fine_tune(self, X_train, y_train, X_val, y_val,
                  epochs=20, batch_size=16):
        """
        Fine-tune with unfrozen base layers.
        """
        print("\n" + "="*60)
        print("FINE-TUNING PHASE")
        print("="*60)
        
        # Unfreeze some base model layers
        self.unfreeze_base_model(layers_to_unfreeze=40)
        
        # Continue training with lower learning rate
        self.train(X_train, y_train, X_val, y_val,
                  epochs=epochs, batch_size=batch_size, 
                  augment=True, learning_rate=0.00005)
    
    def evaluate(self, X_test, y_test, batch_size=32):
        """Evaluate the model on test data."""
        self.model.eval()
        
        val_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        test_dataset = HandGestureDataset(X_test, y_test, transform=val_transform)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, 
                                shuffle=False, num_workers=0)
        
        criterion = nn.BCELoss()
        test_loss = 0.0
        test_correct = 0
        test_total = 0
        precision_sum = 0
        recall_sum = 0
        
        with torch.no_grad():
            for inputs, labels in tqdm(test_loader, desc='Testing'):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                labels = labels.unsqueeze(1)
                
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                
                test_loss += loss.item()
                preds = (outputs > 0.5).float()
                test_correct += (preds == labels).sum().item()
                test_total += labels.size(0)
                
                precision, recall = self.calculate_metrics(outputs, labels)
                precision_sum += precision
                recall_sum += recall
        
        test_loss = test_loss / len(test_loader)
        test_acc = test_correct / test_total
        precision = precision_sum / len(test_loader)
        recall = recall_sum / len(test_loader)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-7)
        
        print(f"\nTest Results:")
        print(f"Loss: {test_loss:.4f}")
        print(f"Accuracy: {test_acc:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        
        return test_loss, test_acc, precision, recall
    
    def plot_training_history(self, save_path='training_history.png'):
        """Plot training history with all metrics."""
        if len(self.history['train_loss']) == 0:
            print("No training history available")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Plot accuracy
        axes[0, 0].plot(self.history['train_acc'], label='Train')
        axes[0, 0].plot(self.history['val_acc'], label='Validation')
        axes[0, 0].set_title('Model Accuracy')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Plot loss
        axes[0, 1].plot(self.history['train_loss'], label='Train')
        axes[0, 1].plot(self.history['val_loss'], label='Validation')
        axes[0, 1].set_title('Model Loss')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Plot precision
        axes[1, 0].plot(self.history['train_precision'], label='Train')
        axes[1, 0].plot(self.history['val_precision'], label='Validation')
        axes[1, 0].set_title('Model Precision')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Precision')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Plot recall
        axes[1, 1].plot(self.history['train_recall'], label='Train')
        axes[1, 1].plot(self.history['val_recall'], label='Validation')
        axes[1, 1].set_title('Model Recall')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Recall')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Training history saved to {save_path}")
        plt.show()
    
    def save_model(self, filepath='hand_gesture_mediapipe.pth'):
        """Save the trained model."""
        if self.model is None:
            print("No model to save")
            return
        
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'history': self.history,
            'img_size': self.img_size,
        }, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath='hand_gesture_mediapipe.pth'):
        """Load a trained model."""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        if self.model is None:
            self.build_model(freeze_base=False)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.history = checkpoint.get('history', self.history)
        print(f"Model loaded from {filepath}")
    
    def cleanup(self):
        """Clean up MediaPipe resources."""
        if self.use_hand_crop and self.hands:
            self.hands.close()


def main():
    """Main training pipeline with MediaPipe + EfficientNet in PyTorch."""
    print("=" * 60)
    print("PyTorch MediaPipe Hand Gesture Fine-Tuning Pipeline")
    print("Using EfficientNet-B0 + MediaPipe Hand Detection")
    print("=" * 60)
    
    # Initialize model with MediaPipe preprocessing
    cnn = MediaPipeHandGestureCNN(img_size=(224, 224), use_hand_crop=True)
    
    # Load dataset
    print("\n1. Loading dataset with MediaPipe hand detection...")
    images, labels = cnn.load_dataset(data_dir='data')
    
    if len(images) == 0:
        print("Error: No images found. Please capture training data first.")
        return
    
    # Normalize images (will be further normalized by transforms)
    images = images / 255.0
    
    # Split dataset
    print("\n2. Splitting dataset...")
    X_train, X_temp, y_train, y_temp = train_test_split(
        images, labels, test_size=0.3, random_state=42, stratify=labels
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )
    
    print(f"Training: {len(X_train)}, Validation: {len(X_val)}, Test: {len(X_test)}")
    
    # Build model
    print("\n3. Building model with frozen base...")
    cnn.build_model(freeze_base=True)
    
    # Phase 1: Train with frozen base
    print("\n4. Phase 1: Training classification head...")
    print("   (Base model frozen, only training new layers)")
    cnn.train(X_train, y_train, X_val, y_val, epochs=20, batch_size=32)
    
    # Phase 2: Fine-tune with unfrozen layers
    print("\n5. Phase 2: Fine-tuning...")
    print("   (Unfreezing last layers of base model)")
    cnn.fine_tune(X_train, y_train, X_val, y_val, epochs=15, batch_size=16)
    
    # Evaluate model
    print("\n6. Evaluating model...")
    cnn.evaluate(X_test, y_test)
    
    # Plot training history
    print("\n7. Saving training history...")
    cnn.plot_training_history()
    
    # Save model
    print("\n8. Saving model...")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = f'hand_gesture_mediapipe_{timestamp}.pth'
    cnn.save_model(model_path)
    cnn.save_model('proposed_hand_gesture_model.pth')
    
    # Cleanup
    cnn.cleanup()
    
    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)
    print(f"\nModel saved as: {model_path}")
    print("This model uses MediaPipe hand detection + EfficientNet-B0")


if __name__ == "__main__":
    main()