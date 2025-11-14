import cv2
import numpy as np
import os
from sklearn.model_selection import train_test_split
from typing import Tuple, List
import matplotlib.pyplot as plt

class HandGestureDataset:
    """Load and prepare hand gesture training dataset."""
    
    def __init__(self, data_dir: str = 'data', img_size: Tuple[int, int] = (128, 128)):
        """
        Initialize dataset loader.
        
        Args:
            data_dir: Root directory containing 'pinched' and 'unpinched' folders
            img_size: Target size for images (width, height)
        """
        self.data_dir = data_dir
        self.img_size = img_size
        self.images = []
        self.labels = []
        self.class_names = ['unpinched', 'pinched']
        
    def load_images(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load all images from pinched and unpinched directories.
        
        Returns:
            Tuple of (images, labels) as numpy arrays
        """
        for label, class_name in enumerate(self.class_names):
            class_dir = os.path.join(self.data_dir, class_name)
            
            if not os.path.exists(class_dir):
                print(f"Warning: Directory {class_dir} not found")
                continue
            
            image_files = [f for f in os.listdir(class_dir) 
                          if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            
            print(f"Loading {len(image_files)} images from {class_name}...")
            
            for img_file in image_files:
                img_path = os.path.join(class_dir, img_file)
                
                # Read image
                img = cv2.imread(img_path)
                
                if img is None:
                    print(f"Warning: Could not read {img_path}")
                    continue
                
                # Resize image
                img = cv2.resize(img, self.img_size)
                
                # Convert BGR to RGB
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                self.images.append(img)
                self.labels.append(label)
        
        # Convert to numpy arrays
        self.images = np.array(self.images, dtype=np.float32)
        self.labels = np.array(self.labels, dtype=np.int32)
        
        print(f"\nTotal images loaded: {len(self.images)}")
        print(f"Image shape: {self.images.shape}")
        print(f"Unpinched: {np.sum(self.labels == 0)}, Pinched: {np.sum(self.labels == 1)}")
        
        return self.images, self.labels
    
    def normalize_images(self, images: np.ndarray) -> np.ndarray:
        """
        Normalize images to [0, 1] range.
        
        Args:
            images: Array of images
            
        Returns:
            Normalized images
        """
        return images / 255.0
    
    def get_train_test_split(self, test_size: float = 0.2, 
                            random_state: int = 42) -> Tuple[np.ndarray, np.ndarray, 
                                                              np.ndarray, np.ndarray]:
        """
        Split dataset into training and testing sets.
        
        Args:
            test_size: Proportion of dataset for testing
            random_state: Random seed for reproducibility
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        if len(self.images) == 0:
            raise ValueError("No images loaded. Call load_images() first.")
        
        # Normalize images
        X = self.normalize_images(self.images)
        y = self.labels
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        print(f"\nTraining set: {len(X_train)} images")
        print(f"Testing set: {len(X_test)} images")
        
        return X_train, X_test, y_train, y_test
    
    def visualize_samples(self, num_samples: int = 8):
        """
        Visualize random samples from the dataset.
        
        Args:
            num_samples: Number of samples to display
        """
        if len(self.images) == 0:
            print("No images loaded. Call load_images() first.")
            return
        
        # Select random samples
        indices = np.random.choice(len(self.images), 
                                  size=min(num_samples, len(self.images)), 
                                  replace=False)
        
        fig, axes = plt.subplots(2, 4, figsize=(12, 6))
        axes = axes.ravel()
        
        for idx, ax in zip(indices, axes):
            ax.imshow(self.images[idx].astype(np.uint8))
            ax.set_title(self.class_names[self.labels[idx]])
            ax.axis('off')
        
        plt.tight_layout()
        plt.savefig('dataset_samples.png', dpi=150, bbox_inches='tight')
        print("Sample visualization saved as 'dataset_samples.png'")
        plt.show()
    
    def get_class_distribution(self):
        """Print class distribution statistics."""
        if len(self.labels) == 0:
            print("No labels loaded. Call load_images() first.")
            return
        
        unique, counts = np.unique(self.labels, return_counts=True)
        
        print("\nClass Distribution:")
        print("-" * 30)
        for class_idx, count in zip(unique, counts):
            percentage = (count / len(self.labels)) * 100
            print(f"{self.class_names[class_idx]}: {count} ({percentage:.1f}%)")
        print("-" * 30)


def main():
    """Example usage of the HandGestureDataset class."""
    
    # Initialize dataset
    dataset = HandGestureDataset(data_dir='data', img_size=(128, 128))
    
    # Load images
    images, labels = dataset.load_images()
    
    # Show class distribution
    dataset.get_class_distribution()
    
    # Get train/test split
    X_train, X_test, y_train, y_test = dataset.get_train_test_split(test_size=0.2)
    
    # Visualize samples
    dataset.visualize_samples(num_samples=8)
    
    print("\nDataset ready for training!")
    print(f"X_train shape: {X_train.shape}")
    print(f"y_train shape: {y_train.shape}")
    
    # Example: Save processed data for later use
    np.save('X_train.npy', X_train)
    np.save('X_test.npy', X_test)
    np.save('y_train.npy', y_train)
    np.save('y_test.npy', y_test)
    print("\nProcessed data saved as .npy files")


if __name__ == "__main__":
    main()