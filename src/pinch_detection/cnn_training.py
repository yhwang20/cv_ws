import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
import cv2
import os
import matplotlib.pyplot as plt
from datetime import datetime

class HandGestureCNN:
    """CNN model for hand gesture classification."""
    
    def __init__(self, img_size=(128, 128)):
        """
        Initialize the CNN model.
        
        Args:
            img_size: Input image size (width, height)
        """
        self.img_size = img_size
        self.model = None
        self.history = None
        self.class_names = ['unpinched', 'pinched']
        
    def load_dataset(self, data_dir='data'):
        """
        Load images from pinched and unpinched directories.
        
        Args:
            data_dir: Root directory containing class folders
            
        Returns:
            Tuple of (images, labels)
        """
        images = []
        labels = []
        
        for label, class_name in enumerate(self.class_names):
            class_dir = os.path.join(data_dir, class_name)
            
            if not os.path.exists(class_dir):
                print(f"Warning: Directory {class_dir} not found")
                continue
            
            image_files = [f for f in os.listdir(class_dir) 
                          if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            
            print(f"Loading {len(image_files)} images from {class_name}...")
            
            for img_file in image_files:
                img_path = os.path.join(class_dir, img_file)
                img = cv2.imread(img_path)
                
                if img is None:
                    continue
                
                img = cv2.resize(img, self.img_size)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                images.append(img)
                labels.append(label)
        
        images = np.array(images, dtype=np.float32) / 255.0
        labels = np.array(labels, dtype=np.int32)
        
        print(f"\nTotal images loaded: {len(images)}")
        print(f"Unpinched: {np.sum(labels == 0)}, Pinched: {np.sum(labels == 1)}")
        
        return images, labels
    
    def build_model(self):
        """Build the CNN architecture."""
        model = keras.Sequential([
            # First convolutional block
            layers.Conv2D(32, (3, 3), activation='relu', 
                         input_shape=(*self.img_size, 3)),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Second convolutional block
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Third convolutional block
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Fourth convolutional block
            layers.Conv2D(256, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Flatten and dense layers
            layers.Flatten(),
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(128, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(1, activation='sigmoid')  # Binary classification
        ])
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        self.model = model
        return model
    
    def train(self, X_train, y_train, X_val, y_val, 
              epochs=50, batch_size=32):
        """
        Train the model.
        
        Args:
            X_train: Training images
            y_train: Training labels
            X_val: Validation images
            y_val: Validation labels
            epochs: Number of training epochs
            batch_size: Batch size for training
        """
        if self.model is None:
            self.build_model()
        
        # Callbacks
        early_stopping = keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )
        
        reduce_lr = keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7
        )
        
        # Data augmentation
        data_augmentation = keras.Sequential([
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.1),
            layers.RandomZoom(0.1),
            layers.RandomBrightness(0.2),
        ])
        
        # Create augmented dataset
        augmented_train = data_augmentation(X_train, training=True)
        
        print("\nStarting training...")
        print(f"Training samples: {len(X_train)}")
        print(f"Validation samples: {len(X_val)}")
        
        self.history = self.model.fit(
            augmented_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stopping, reduce_lr],
            verbose=1
        )
        
        return self.history
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate the model on test data.
        
        Args:
            X_test: Test images
            y_test: Test labels
        """
        test_loss, test_accuracy = self.model.evaluate(X_test, y_test, verbose=0)
        print(f"\nTest Loss: {test_loss:.4f}")
        print(f"Test Accuracy: {test_accuracy:.4f}")
        return test_loss, test_accuracy
    
    def plot_training_history(self, save_path='training_history.png'):
        """Plot training history."""
        if self.history is None:
            print("No training history available")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Plot accuracy
        ax1.plot(self.history.history['accuracy'], label='Train')
        ax1.plot(self.history.history['val_accuracy'], label='Validation')
        ax1.set_title('Model Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        ax1.grid(True)
        
        # Plot loss
        ax2.plot(self.history.history['loss'], label='Train')
        ax2.plot(self.history.history['val_loss'], label='Validation')
        ax2.set_title('Model Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Training history saved to {save_path}")
        plt.show()
    
    def save_model(self, filepath='hand_gesture_model.keras'):
        """Save the trained model."""
        if self.model is None:
            print("No model to save")
            return
        
        self.model.save(filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath='hand_gesture_model.keras'):
        """Load a trained model."""
        self.model = keras.models.load_model(filepath)
        print(f"Model loaded from {filepath}")


def main():
    """Main training pipeline."""
    print("=" * 60)
    print("Hand Gesture CNN Training Pipeline")
    print("=" * 60)
    
    # Initialize model
    cnn = HandGestureCNN(img_size=(128, 128))
    
    # Load dataset
    print("\n1. Loading dataset...")
    images, labels = cnn.load_dataset(data_dir='data')
    
    if len(images) == 0:
        print("Error: No images found. Please capture training data first.")
        return
    
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
    print("\n3. Building model...")
    cnn.build_model()
    cnn.model.summary()
    
    # Train model
    print("\n4. Training model...")
    cnn.train(X_train, y_train, X_val, y_val, epochs=50, batch_size=32)
    
    # Evaluate model
    print("\n5. Evaluating model...")
    cnn.evaluate(X_test, y_test)
    
    # Plot training history
    print("\n6. Saving training history...")
    cnn.plot_training_history()
    
    # Save model
    print("\n7. Saving model...")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = f'hand_gesture_model_{timestamp}.keras'
    cnn.save_model(model_path)
    
    # Also save as default name for easy loading
    cnn.save_model('hand_gesture_model.keras')
    
    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()