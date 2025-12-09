import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import pandas as pd
from datetime import datetime
import os
import cv2
import mediapipe as mp

from MP_COTS import CustomMediapipeGestureRecognizer

# Assuming HandGestureDataset is available from your training script
class HandGestureDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        else:
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        
        return image, torch.tensor(label, dtype=torch.float32)


class ModelComparison:
    """Compare multiple trained models on the same test dataset."""
    
    def __init__(self, device=None):
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        print(f"Using device: {self.device}")
        
        self.results = []
        
        # Initialize MediaPipe Hands for preprocessing
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=True,
            max_num_hands=1,
            min_detection_confidence=0.5
        )
        print("MediaPipe Hands initialized for optional preprocessing")
        
    def preprocess_with_mediapipe(self, image):
        """
        Use MediaPipe to detect and crop hand region.
        
        Args:
            image: Input image (RGB format, numpy array)
            
        Returns:
            Cropped hand image or original if no hand detected
        """
        # Ensure image is in correct format
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)
        
        # Process image with MediaPipe
        results = self.hands.process(image)
        
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
    
    def load_model_from_checkpoint(self, model_path, model_architecture):
        """Load a model from checkpoint."""
        checkpoint = torch.load(model_path, map_location=self.device)
        model = model_architecture.to(self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        return model
    
    def calculate_metrics(self, outputs, labels):
        """Calculate comprehensive metrics."""
        preds = (outputs > 0.5).float()
        
        tp = ((preds == 1) & (labels == 1)).sum().item()
        tn = ((preds == 0) & (labels == 0)).sum().item()
        fp = ((preds == 1) & (labels == 0)).sum().item()
        fn = ((preds == 0) & (labels == 1)).sum().item()
        
        accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-7)
        precision = tp / (tp + fp + 1e-7)
        recall = tp / (tp + fn + 1e-7)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-7)
        specificity = tn / (tn + fp + 1e-7)
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'specificity': specificity,
            'tp': tp,
            'tn': tn,
            'fp': fp,
            'fn': fn
        }
    
    def evaluate_model(self, model, test_loader, model_name, use_mediapipe=False):
        """Evaluate a single model on test data.
        
        Args:
            model: The model to evaluate
            test_loader: DataLoader for test data
            model_name: Name of the model
            use_mediapipe: Whether to apply MediaPipe hand cropping
        """
        print(f"\nEvaluating {model_name}...")
        if use_mediapipe:
            print("  Using MediaPipe hand cropping preprocessing")
        
        criterion = nn.BCELoss()
        total_loss = 0.0
        all_outputs = []
        all_labels = []
        
        # If using MediaPipe, we need to reprocess images
        if use_mediapipe:
            # Get raw images from the dataset
            raw_images = test_loader.dataset.images
            labels_array = test_loader.dataset.labels
            
            # Get target size from the original images
            target_size = (raw_images[0].shape[1], raw_images[0].shape[0])  # (width, height)
            
            # Apply MediaPipe preprocessing
            processed_images = []
            print("  Applying MediaPipe preprocessing...")
            for img in tqdm(raw_images, desc="  Processing"):
                processed_img = self.preprocess_with_mediapipe(img)
                # Resize to consistent size after cropping
                processed_img = cv2.resize(processed_img, target_size)
                processed_images.append(processed_img)
            
            # Convert to numpy array (now all images have same shape)
            processed_images = np.array(processed_images, dtype=np.uint8)
            
            # Create new dataset with processed images
            processed_dataset = HandGestureDataset(
                processed_images, 
                labels_array, 
                transform=test_loader.dataset.transform
            )
            test_loader = DataLoader(
                processed_dataset, 
                batch_size=test_loader.batch_size,
                shuffle=False, 
                num_workers=0
            )
        
        with torch.no_grad():
            for inputs, labels in tqdm(test_loader, desc=f'Testing {model_name}'):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                labels = labels.unsqueeze(1)
                
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                total_loss += loss.item()
                all_outputs.append(outputs.cpu())
                all_labels.append(labels.cpu())
        
        # Concatenate all batches
        all_outputs = torch.cat(all_outputs)
        all_labels = torch.cat(all_labels)
        
        # Calculate metrics
        metrics = self.calculate_metrics(all_outputs, all_labels)
        metrics['loss'] = total_loss / len(test_loader)
        metrics['model_name'] = model_name
        metrics['use_mediapipe'] = use_mediapipe
        
        print(f"Results for {model_name}:")
        print(f"  MediaPipe:   {'Enabled' if use_mediapipe else 'Disabled'}")
        print(f"  Accuracy:    {metrics['accuracy']:.4f}")
        print(f"  Precision:   {metrics['precision']:.4f}")
        print(f"  Recall:      {metrics['recall']:.4f}")
        print(f"  F1 Score:    {metrics['f1']:.4f}")
        print(f"  Specificity: {metrics['specificity']:.4f}")
        print(f"  Loss:        {metrics['loss']:.4f}")
        
        return metrics
    
    def evaluate_cots_model(self, cots_model, X_test, y_test, model_name):
        """
        Evaluate a MediaPipeGestureRecognition COTS model on test images,
        returning metrics in the same format as evaluate_model.
        """
        print(f"\nEvaluating {model_name} (COTS MediaPipe)...")

        all_outputs = []
        all_labels = []

        for img, label in tqdm(zip(X_test, y_test), total=len(X_test), desc=f'Testing {model_name}'):
            # Ensure image is uint8 RGB
            if img.dtype != np.uint8:
                img = (img * 255).astype(np.uint8)

            # Run MediaPipe gesture recognition
            result = cots_model(img)

            # Binary prediction for metrics: Closed_Fist -> 1, Open_Palm -> 0
            pred = float(result['is_pinching'])
            all_outputs.append(torch.tensor([pred], dtype=torch.float32))
            all_labels.append(torch.tensor([label], dtype=torch.float32))

            # Optionally store extra outputs
            if not hasattr(self, "_cots_outputs"):
                self._cots_outputs = []
            self._cots_outputs.append({
                'pred': pred,
                'label': label,
                'gesture_name': result['gesture_name'],
                'gesture_confidence': result['gesture_confidence'],
                'handedness': result['handedness'],
                'palm_centroid_pixel': result['palm_centroid_pixel'],
                'normalized_spread': result['normalized_spread'],
                'annotated_image': result['annotated_image']
            })

        # Concatenate tensors
        all_outputs = torch.cat(all_outputs)
        all_labels = torch.cat(all_labels)

        # Compute metrics
        tp = ((all_outputs == 1) & (all_labels == 1)).sum().item()
        tn = ((all_outputs == 0) & (all_labels == 0)).sum().item()
        fp = ((all_outputs == 1) & (all_labels == 0)).sum().item()
        fn = ((all_outputs == 0) & (all_labels == 1)).sum().item()

        accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-7)
        precision = tp / (tp + fp + 1e-7)
        recall = tp / (tp + fn + 1e-7)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-7)
        specificity = tn / (tn + fp + 1e-7)

        metrics = {
            'model_name': model_name,
            'use_mediapipe': True,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'specificity': specificity,
            'loss': 0.0, 
            'tp': tp,
            'tn': tn,
            'fp': fp,
            'fn': fn
        }

        print(f"Results for {model_name}:")
        print(f"  Accuracy:    {metrics['accuracy']:.4f}")
        print(f"  Precision:   {metrics['precision']:.4f}")
        print(f"  Recall:      {metrics['recall']:.4f}")
        print(f"  F1 Score:    {metrics['f1']:.4f}")
        print(f"  Specificity: {metrics['specificity']:.4f}")
        print(f"  Loss:        {metrics['loss']:.4f}")

        return metrics

    
    def compare_models(self, models_list, X_test, y_test, batch_size=32, img_size=(224, 224)):
        """
        Compare multiple models on the same test dataset.
        
        Args:
            models_list: List of tuples (model_name, model_path, model_architecture, use_mediapipe)
                        use_mediapipe: bool indicating whether to preprocess with MediaPipe
            X_test: Test images (numpy array, RGB format, 0-255 range)
            y_test: Test labels
            batch_size: Batch size for evaluation
            img_size: Target image size (width, height)
        """
        # Prepare test data
        val_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        # Ensure images are in correct format (0-255 range, uint8)
        if X_test.dtype == np.float32 or X_test.dtype == np.float64:
            if X_test.max() <= 1.0:
                X_test = (X_test * 255).astype(np.uint8)
            else:
                X_test = X_test.astype(np.uint8)
        
        test_dataset = HandGestureDataset(X_test, y_test, transform=val_transform)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, 
                                shuffle=False, num_workers=0)
        
        print("="*60)
        print("MODEL COMPARISON")
        print("="*60)
        print(f"Test samples: {len(X_test)}")
        print(f"Number of models: {len(models_list)}")
        print(f"Image size: {img_size}")
        
        # Evaluate each model
        for model_info in models_list:
            model_name, model_path, model_architecture, use_mediapipe, COTS = model_info
            if COTS == None:
                try:
                    model = self.load_model_from_checkpoint(model_path, model_architecture)
                    metrics = self.evaluate_model(model, test_loader, model_name, use_mediapipe)
                    self.results.append(metrics)
                except Exception as e:
                    print(f"Error evaluating {model_name}: {e}")
                    import traceback
                    traceback.print_exc()
                    continue
            else:
                try: 
                    metrics = self.evaluate_cots_model(COTS, X_test, y_test, model_name)
                    self.results.append(metrics)
                except Exception as e:
                    print(f"Error evaluating {model_name}: {e}")
                    import traceback
                    traceback.print_exc()
                    continue
        
        return self.results
    
    def create_comparison_plots(self, save_dir='comparison_plots',test_name="UNSPECIFIED"):
        """Create publication-quality comparison plots."""
        if len(self.results) == 0:
            print("No results to plot")
            return
        
        os.makedirs(save_dir, exist_ok=True)
        
        # Convert results to DataFrame
        df = pd.DataFrame(self.results)
        model_names = df['model_name'].tolist()
        
        # Set style for publication
        sns.set_style("whitegrid")
        plt.rcParams['font.size'] = 11
        plt.rcParams['font.family'] = 'serif'
        
        # 1. Bar chart comparing all metrics
        fig, ax = plt.subplots(figsize=(12, 6))
        metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1', 'specificity']
        x = np.arange(len(model_names))
        width = 0.15
        
        colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#6A994E']
        
        for i, metric in enumerate(metrics_to_plot):
            values = df[metric].tolist()
            ax.bar(x + i*width, values, width, label=metric.capitalize(), 
                   color=colors[i], alpha=0.8)
        
        ax.set_xlabel('Model', fontsize=12, fontweight='bold')
        ax.set_ylabel('Score', fontsize=12, fontweight='bold')
        ax.set_title(f'Model Performance Comparison on {test_name}', fontsize=14, fontweight='bold')
        ax.set_xticks(x + width * 2)
        ax.set_xticklabels(model_names, rotation=45, ha='right')
        ax.legend(loc='lower right', framealpha=0.9)
        ax.set_ylim([0, 1.05])
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{save_dir}/metrics_comparison_bar.png', dpi=300, bbox_inches='tight')
        print(f"Saved: {save_dir}/metrics_comparison_bar.png")
        plt.close()
        
        # 2. Radar chart for each model
        fig, axes = plt.subplots(1, len(model_names), 
                                figsize=(5*len(model_names), 5), 
                                subplot_kw=dict(projection='polar'))
        
        if len(model_names) == 1:
            axes = [axes]
        
        categories = ['Accuracy', 'Precision', 'Recall', 'F1', 'Specificity']
        N = len(categories)
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
        angles += angles[:1]
        
        for idx, (ax, model_name) in enumerate(zip(axes, model_names)):
            values = df.iloc[idx][metrics_to_plot].tolist()
            values += values[:1]
            
            ax.plot(angles, values, 'o-', linewidth=2, color=colors[idx % len(colors)])
            ax.fill(angles, values, alpha=0.25, color=colors[idx % len(colors)])
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(categories, size=10)
            ax.set_ylim(0, 1)
            ax.set_title(model_name, size=12, fontweight='bold', pad=20)
            ax.grid(True)
        
        fig.suptitle(f"Radar Chart by model on {test_name}", fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig(f'{save_dir}/metrics_radar.png', dpi=300, bbox_inches='tight')
        print(f"Saved: {save_dir}/metrics_radar.png")
        plt.close()
        
        # 3. Heatmap of all metrics
        fig, ax = plt.subplots(figsize=(10, len(model_names)*0.8))
        
        heatmap_data = df[metrics_to_plot].values
        im = ax.imshow(heatmap_data, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
        
        ax.set_xticks(np.arange(len(metrics_to_plot)))
        ax.set_yticks(np.arange(len(model_names)))
        ax.set_xticklabels([m.capitalize() for m in metrics_to_plot])
        ax.set_yticklabels(model_names)
        
        # Add text annotations
        for i in range(len(model_names)):
            for j in range(len(metrics_to_plot)):
                text = ax.text(j, i, f'{heatmap_data[i, j]:.3f}',
                             ha="center", va="center", color="black", fontsize=10)
        
        fig.suptitle(f'Model Performance Heatmap on {test_name}', fontsize=14, fontweight='bold', y=1.02)
        
        # Create colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Score', rotation=270, labelpad=20)
        
        plt.tight_layout()
        plt.savefig(f'{save_dir}/metrics_heatmap.png', dpi=300, bbox_inches='tight')
        print(f"Saved: {save_dir}/metrics_heatmap.png")
        plt.close()
        
        # 4. Confusion matrix comparison (if we have TP, TN, FP, FN)
        fig, axes = plt.subplots(1, len(model_names), 
                                figsize=(5*len(model_names), 4))
        
        if len(model_names) == 1:
            axes = [axes]
        
        for idx, (ax, model_name) in enumerate(zip(axes, model_names)):
            cm = np.array([[df.iloc[idx]['tn'], df.iloc[idx]['fp']],
                          [df.iloc[idx]['fn'], df.iloc[idx]['tp']]])
            
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                       xticklabels=['Unpinched', 'Pinched'],
                       yticklabels=['Unpinched', 'Pinched'],
                       cbar=False)
            ax.set_title(model_name, fontweight='bold')
            ax.set_ylabel('True Label')
            ax.set_xlabel('Predicted Label')
        
        fig.suptitle(f"Confusion Matrices by model on {test_name}", fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig(f'{save_dir}/confusion_matrices.png', dpi=300, bbox_inches='tight')
        print(f"Saved: {save_dir}/confusion_matrices.png")
        plt.close()
        
        # 5. Summary statistics table plot
        fig, ax = plt.subplots(figsize=(14, len(model_names)*0.8 + 1))
        ax.axis('tight')
        ax.axis('off')
        
        table_data = []
        for _, row in df.iterrows():
            mp_indicator = "True" if row.get('use_mediapipe', False) else "False"
            table_data.append([
                row['model_name'],
                mp_indicator,
                f"{row['accuracy']:.4f}",
                f"{row['precision']:.4f}",
                f"{row['recall']:.4f}",
                f"{row['f1']:.4f}",
                f"{row['specificity']:.4f}",
                f"{row['loss']:.4f}"
            ])
        
        table = ax.table(cellText=table_data,
                        colLabels=['Model', 'MP', 'Accuracy', 'Precision', 'Recall', 
                                  'F1', 'Specificity', 'Loss'],
                        cellLoc='left',
                        loc='center',
                        colWidths=[0.28, 0.08, 0.11, 0.11, 0.11, 0.11, 0.11, 0.09])
        
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 2.2)
        
        # Style the header
        for i in range(8):
            table[(0, i)].set_facecolor('#2E86AB')
            table[(0, i)].set_text_props(weight='bold', color='white')
            if i == 0:  # Model name column header
                table[(0, i)].set_text_props(weight='bold', color='white', ha='left')
        
        # Style data rows
        for i in range(1, len(table_data) + 1):
            # Model name column - left aligned
            table[(i, 0)].set_text_props(ha='left')
            table[(i, 0)].PAD = 0.05
            
            # Alternate row colors
            for j in range(8):
                if i % 2 == 0:
                    table[(i, j)].set_facecolor('#E8E8E8')
        
        plt.title(f'Model Comparison Summary on {test_name}', 
                 fontsize=14, fontweight='bold', pad=20)
        plt.savefig(f'{save_dir}/summary_table.png', dpi=300, bbox_inches='tight')
        print(f"Saved: {save_dir}/summary_table.png")
        plt.close()
        
        print(f"\nAll plots saved to {save_dir}/")
    
    def export_results_to_csv(self, filepath='model_comparison_results.csv'):
        """Export results to CSV for further analysis."""
        if len(self.results) == 0:
            print("No results to export")
            return
        
        df = pd.DataFrame(self.results)
        df.to_csv(filepath, index=False)
        print(f"Results exported to {filepath}")
    
    def print_summary(self):
        """Print a summary of all models."""
        if len(self.results) == 0:
            print("No results available")
            return
        
        df = pd.DataFrame(self.results)
        
        print("\n" + "="*60)
        print("COMPARISON SUMMARY")
        print("="*60)
        
        print("\nBest performing models by metric:")
        metrics = ['accuracy', 'precision', 'recall', 'f1', 'specificity']
        for metric in metrics:
            best_idx = df[metric].idxmax()
            mp_status = "with MediaPipe" if df.iloc[best_idx].get('use_mediapipe', False) else "without MediaPipe"
            print(f"  {metric.capitalize()}: {df.iloc[best_idx]['model_name']} "
                  f"({df.iloc[best_idx][metric]:.4f}) {mp_status}")
        
        print("\nLowest loss:")
        best_idx = df['loss'].idxmin()
        mp_status = "with MediaPipe" if df.iloc[best_idx].get('use_mediapipe', False) else "without MediaPipe"
        print(f"  {df.iloc[best_idx]['model_name']} ({df.iloc[best_idx]['loss']:.4f}) {mp_status}")
        
        # Analyze MediaPipe impact if we have models with and without
        if 'use_mediapipe' in df.columns:
            mp_true = df[df['use_mediapipe'] == True]
            mp_false = df[df['use_mediapipe'] == False]
            
            if len(mp_true) > 0 and len(mp_false) > 0:
                print("\nMediaPipe Impact Analysis:")
                print(f"  Models with MediaPipe: {len(mp_true)}")
                print(f"  Models without MediaPipe: {len(mp_false)}")
                print(f"\n  Average Accuracy:")
                print(f"    With MediaPipe:    {mp_true['accuracy'].mean():.4f}")
                print(f"    Without MediaPipe: {mp_false['accuracy'].mean():.4f}")
                print(f"  Average F1 Score:")
                print(f"    With MediaPipe:    {mp_true['f1'].mean():.4f}")
                print(f"    Without MediaPipe: {mp_false['f1'].mean():.4f}")
    
    def cleanup(self):
        """Clean up MediaPipe resources."""
        if self.hands:
            self.hands.close()
            print("MediaPipe resources cleaned up")


# Example usage function
def main():
    """Example of how to use the ModelComparison class."""
    from torchvision import models
    from sklearn.model_selection import train_test_split

    # Initialize comparison
    comparator = ModelComparison()
    
    def get_model_architecture():
        base_model = models.efficientnet_b0(weights=None)
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
        return base_model

    from pathlib import Path
    # Use gesture_recognizer.task from unified_vision_node/weights
    gesture_model_path = Path(__file__).parent.parent / "unified_vision_node" / "weights" / "gesture_recognizer.task"
    gesture_model_path = gesture_model_path.resolve()  # Convert to absolute path
    
    # Verify file exists before passing to MediaPipe
    if not gesture_model_path.exists():
        raise FileNotFoundError(f"Gesture recognizer model not found at: {gesture_model_path}")
    
    MP_COTS = CustomMediapipeGestureRecognizer(model_path=str(gesture_model_path))

    # List of models to compare: (name, path, architecture)
    models_to_compare = [
        ("Method 1 - Baseline Model", "hand_gesture_model_no_crop_v01.pth", get_model_architecture(),False,None),
        ("Method 2 - MediaPipe Enhanced v01", "hand_gesture_model_v01.pth", get_model_architecture(),True,None),
        ("Method 2 - MediaPipe Enhanced v02", "hand_gesture_model_v02.pth", get_model_architecture(),True,None),
        ("Method 3 - COTS Finger Pinch", None, None, False, MP_COTS)
    ]
    
    #Setup Data: 
    from finetuned_training import MediaPipeHandGestureCNN
    cnn = MediaPipeHandGestureCNN(img_size=(224, 224), use_hand_crop=False)
    X, y = cnn.load_dataset(data_dir='test_data/dark_cluttered_test')
    
    # Compare models (you need to provide X - images and y - labels)
    results = comparator.compare_models(models_to_compare, X, y)
    
    # Create plots
    comparator.create_comparison_plots("comparison_plots/dark_cluttered","Dark Cluttered")
    
    # Export results
    comparator.export_results_to_csv()
    
    # Print summary
    comparator.print_summary()
    comparator.cleanup()
    

if __name__ == "__main__":
    main()