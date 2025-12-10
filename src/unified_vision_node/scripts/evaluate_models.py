#!/usr/bin/env python3

"""
Model Evaluation Script
Tests YOLO and MediaPipe models on categorized test images.
Measures detection robustness, inference time, and accuracy across environments.
"""

import os
import sys
import cv2
import numpy as np
from pathlib import Path
import time
from collections import defaultdict
import json

# Add package to path
package_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, package_dir)

from unified_vision_node.models.yolo_shoulder_wrist import YoloShoulderWrist
from unified_vision_node.models.mediapipe_shoulder_wrist import MediapipeShoulderWrist


class ModelEvaluator:
    """Evaluates pose detection models on test images."""
    
    def __init__(self, test_data_root='/home/mariana/final_ws/test_data'):
        self.test_data_root = Path(test_data_root)
        
        # Categories mapping
        self.categories = {
            'bright_clean_test': 'bright_clean',
            'bright_cluttered_test': 'bright_cluttered',
            'bright_distant_test': 'bright_distant',
            'dark_clean_test': 'dark_clean',
            'dark_cluttered_test': 'dark_cluttered',
            'dark_distant_test': 'dark_distant'
        }
        
        # Initialize results storage
        self.results = {
            'yolo': defaultdict(lambda: {
                'detection_success': 0,
                'detection_failure': 0,
                'inference_times': [],
                'wrist_detected': 0,
                'shoulder_detected': 0,
                'both_detected': 0,
                'total_images': 0
            }),
            'mediapipe': defaultdict(lambda: {
                'detection_success': 0,
                'detection_failure': 0,
                'inference_times': [],
                'wrist_detected': 0,
                'shoulder_detected': 0,
                'both_detected': 0,
                'total_images': 0
            })
        }
        
        print("Initializing models...")
        
        # Initialize YOLO model
        try:
            self.yolo_model = YoloShoulderWrist(
                model_path='yolo11n-pose.pt',
                hand_side='RIGHT',
                depth_window_size=1,
                confidence_threshold=0.5
            )
            print("✓ YOLO model initialized")
        except Exception as e:
            print(f"✗ Failed to initialize YOLO: {e}")
            self.yolo_model = None
        
        # Initialize MediaPipe model
        try:
            self.mediapipe_model = MediapipeShoulderWrist(
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
                model_complexity=2,
                hand_side='RIGHT',
                depth_window_size=1
            )
            print("✓ MediaPipe model initialized")
        except Exception as e:
            print(f"✗ Failed to initialize MediaPipe: {e}")
            self.mediapipe_model = None
    
    def load_image(self, image_path):
        """Load and convert image to RGB."""
        img = cv2.imread(str(image_path))
        if img is None:
            return None
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    def evaluate_single_image(self, model, model_name, image_path, category):
        """Run inference on a single image and record metrics."""
        # Load image
        rgb_image = self.load_image(image_path)
        if rgb_image is None:
            print(f"Failed to load: {image_path}")
            return
        
        # Run inference with timing
        start_time = time.perf_counter()
        result = model(rgb_image)  # No depth or camera_info needed
        inference_time = (time.perf_counter() - start_time) * 1000  # Convert to ms
        
        # Record results
        results_cat = self.results[model_name][category]
        results_cat['total_images'] += 1
        results_cat['inference_times'].append(inference_time)
        
        # Check detection success
        wrist_detected = result['wrist_pixel'] is not None
        shoulder_detected = result['shoulder_pixel'] is not None
        
        if wrist_detected:
            results_cat['wrist_detected'] += 1
        if shoulder_detected:
            results_cat['shoulder_detected'] += 1
        if wrist_detected and shoulder_detected:
            results_cat['both_detected'] += 1
            results_cat['detection_success'] += 1
        else:
            results_cat['detection_failure'] += 1
    
    def evaluate_category(self, category_dir, category_name, sample_size=None):
        """Evaluate all images in a category."""
        print(f"\n{'='*60}")
        print(f"Evaluating: {category_name}")
        print(f"{'='*60}")
        
        # Collect all images from pinched and unpinched subdirectories
        image_paths = []
        for subdir in ['pinched', 'unpinched']:
            subdir_path = category_dir / subdir
            if subdir_path.exists():
                image_paths.extend(list(subdir_path.glob('*.jpg')))
        
        total_images = len(image_paths)
        print(f"Found {total_images} images")
        
        # Sample if requested
        if sample_size and sample_size < total_images:
            import random
            random.seed(42)
            image_paths = random.sample(image_paths, sample_size)
            print(f"Sampling {sample_size} images")
        
        # Evaluate with YOLO
        if self.yolo_model:
            print(f"\nRunning YOLO on {len(image_paths)} images...")
            for i, img_path in enumerate(image_paths):
                if (i + 1) % 100 == 0:
                    print(f"  Progress: {i+1}/{len(image_paths)}")
                self.evaluate_single_image(
                    self.yolo_model, 'yolo', img_path, category_name
                )
        
        # Evaluate with MediaPipe
        if self.mediapipe_model:
            print(f"\nRunning MediaPipe on {len(image_paths)} images...")
            for i, img_path in enumerate(image_paths):
                if (i + 1) % 100 == 0:
                    print(f"  Progress: {i+1}/{len(image_paths)}")
                self.evaluate_single_image(
                    self.mediapipe_model, 'mediapipe', img_path, category_name
                )
    
    def run_evaluation(self, sample_size=None):
        """Run evaluation on all categories."""
        print("\n" + "="*60)
        print("STARTING MODEL EVALUATION")
        print("="*60)
        
        for category_dir_name, category_name in self.categories.items():
            category_path = self.test_data_root / category_dir_name
            if category_path.exists():
                self.evaluate_category(category_path, category_name, sample_size)
            else:
                print(f"Warning: Category not found: {category_path}")
        
        print("\n" + "="*60)
        print("EVALUATION COMPLETE")
        print("="*60)
    
    def compute_statistics(self):
        """Compute final statistics for all models and categories."""
        stats = {}
        
        for model_name in ['yolo', 'mediapipe']:
            stats[model_name] = {}
            
            for category, results in self.results[model_name].items():
                if results['total_images'] == 0:
                    continue
                
                # Calculate metrics
                detection_rate = (results['both_detected'] / results['total_images']) * 100
                wrist_rate = (results['wrist_detected'] / results['total_images']) * 100
                shoulder_rate = (results['shoulder_detected'] / results['total_images']) * 100
                
                avg_time = np.mean(results['inference_times'])
                std_time = np.std(results['inference_times'])
                fps = 1000.0 / avg_time if avg_time > 0 else 0
                
                stats[model_name][category] = {
                    'total_images': results['total_images'],
                    'detection_rate_both': detection_rate,
                    'detection_rate_wrist': wrist_rate,
                    'detection_rate_shoulder': shoulder_rate,
                    'avg_inference_time_ms': avg_time,
                    'std_inference_time_ms': std_time,
                    'fps': fps,
                    'success_count': results['both_detected'],
                    'failure_count': results['detection_failure']
                }
        
        return stats
    
    def print_results(self):
        """Print formatted results."""
        stats = self.compute_statistics()
        
        print("\n" + "="*80)
        print("EVALUATION RESULTS")
        print("="*80)
        
        for model_name in ['yolo', 'mediapipe']:
            if not stats[model_name]:
                continue
            
            print(f"\n{'='*80}")
            print(f"{model_name.upper()} MODEL RESULTS")
            print(f"{'='*80}")
            
            # Overall statistics
            all_times = []
            total_images = 0
            total_success = 0
            
            for category, cat_stats in stats[model_name].items():
                all_times.extend(self.results[model_name][category]['inference_times'])
                total_images += cat_stats['total_images']
                total_success += cat_stats['success_count']
            
            if all_times:
                overall_avg_time = np.mean(all_times)
                overall_fps = 1000.0 / overall_avg_time
                overall_detection_rate = (total_success / total_images) * 100 if total_images > 0 else 0
                
                print(f"\nOVERALL PERFORMANCE:")
                print(f"  Total Images Tested: {total_images}")
                print(f"  Overall Detection Rate: {overall_detection_rate:.2f}%")
                print(f"  Average Inference Time: {overall_avg_time:.2f} ms")
                print(f"  Average FPS: {overall_fps:.2f}")
            
            # Per-category results
            print(f"\nPER-CATEGORY RESULTS:")
            print(f"\n{'Category':<20} {'Images':<8} {'Detect%':<10} {'Wrist%':<10} {'Shoulder%':<12} {'Time(ms)':<12} {'FPS':<8}")
            print("-" * 90)
            
            for category in ['bright_clean', 'bright_cluttered', 'bright_distant',
                            'dark_clean', 'dark_cluttered', 'dark_distant']:
                if category in stats[model_name]:
                    s = stats[model_name][category]
                    print(f"{category:<20} {s['total_images']:<8} "
                          f"{s['detection_rate_both']:>7.2f}%  "
                          f"{s['detection_rate_wrist']:>7.2f}%  "
                          f"{s['detection_rate_shoulder']:>9.2f}%  "
                          f"{s['avg_inference_time_ms']:>9.2f}  "
                          f"{s['fps']:>7.2f}")
        
        # Comparison summary
        print(f"\n{'='*80}")
        print("MODEL COMPARISON SUMMARY")
        print(f"{'='*80}")
        print(f"\n{'Metric':<35} {'YOLO':<20} {'MediaPipe':<20}")
        print("-" * 80)
        
        # Compare overall metrics
        for model_name in ['yolo', 'mediapipe']:
            if stats[model_name]:
                all_times = []
                for category in stats[model_name].values():
                    all_times.extend(self.results[model_name][list(stats[model_name].keys())[0]]['inference_times'])
                break
        
        metrics_to_compare = [
            ('Average Inference Time (ms)', 'avg_inference_time_ms'),
            ('Average FPS', 'fps'),
            ('Detection Rate - Both (%)', 'detection_rate_both'),
            ('Detection Rate - Wrist (%)', 'detection_rate_wrist'),
            ('Detection Rate - Shoulder (%)', 'detection_rate_shoulder')
        ]
        
        for metric_name, metric_key in metrics_to_compare:
            yolo_vals = [s[metric_key] for s in stats.get('yolo', {}).values()]
            mp_vals = [s[metric_key] for s in stats.get('mediapipe', {}).values()]
            
            yolo_avg = np.mean(yolo_vals) if yolo_vals else 0
            mp_avg = np.mean(mp_vals) if mp_vals else 0
            
            print(f"{metric_name:<35} {yolo_avg:>18.2f}  {mp_avg:>18.2f}")
        
        return stats
    
    def save_results(self, output_file='evaluation_results.json'):
        """Save results to JSON file."""
        stats = self.compute_statistics()
        output_path = self.test_data_root / output_file
        
        with open(output_path, 'w') as f:
            json.dump(stats, f, indent=2)
        
        print(f"\nResults saved to: {output_path}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate YOLO and MediaPipe models on test images')
    parser.add_argument('--test-data-root', type=str, 
                       default='/home/mariana/final_ws/test_data',
                       help='Path to test data root directory')
    parser.add_argument('--sample-size', type=int, default=None,
                       help='Number of images to sample per category (default: all)')
    parser.add_argument('--output', type=str, default='evaluation_results.json',
                       help='Output JSON file name')
    
    args = parser.parse_args()
    
    # Create evaluator
    evaluator = ModelEvaluator(test_data_root=args.test_data_root)
    
    # Run evaluation
    evaluator.run_evaluation(sample_size=args.sample_size)
    
    # Print and save results
    evaluator.print_results()
    evaluator.save_results(output_file=args.output)


if __name__ == '__main__':
    main()

