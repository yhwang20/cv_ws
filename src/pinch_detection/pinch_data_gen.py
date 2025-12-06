import cv2
import os
from datetime import datetime
import time

def create_directories():
    """Create directories for storing pinched and unpinched hand images."""
    dirs = ['data/pinched', 'data/unpinched']
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)
    return dirs

def capture_gestures():
    """Capture hand gesture images for training data."""
    create_directories()
    
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open camera")
        return
    
    # Set camera properties for better quality
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    mode = 'unpinched'  # Start with unpinched mode
    photo_count = {'pinched': 0, 'unpinched': 0}
    countdown = 0
    countdown_start = 0
    
    # Continuous capture mode variables
    continuous_mode = False
    continuous_start_time = 0
    continuous_delay_active = False
    continuous_files = []  # Store all captured files during continuous mode
    last_capture_time = 0
    
    print("Hand Gesture Training Data Collector")
    print("=" * 50)
    print("Controls:")
    print("  'p' - Switch to PINCHED mode")
    print("  'u' - Switch to UNPINCHED mode")
    print("  SPACE - Start 3-second countdown to capture")
    print("  ENTER - Capture instantly")
    print("  'c' - Start/Stop continuous capture (3s delay, then every 0.1s)")
    print("  'q' - Quit")
    print("=" * 50)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame")
            break
        
        # Flip frame horizontally for mirror effect
        frame = cv2.flip(frame, 1)
        
        # Create display frame
        display_frame = frame.copy()
        
        # Add mode indicator
        color = (0, 255, 0) if mode == 'unpinched' else (0, 165, 255)
        cv2.putText(display_frame, f"Mode: {mode.upper()}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        
        # Add photo count
        cv2.putText(display_frame, f"Pinched: {photo_count['pinched']}", (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
        cv2.putText(display_frame, f"Unpinched: {photo_count['unpinched']}", (10, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Handle continuous capture mode
        if continuous_mode or continuous_delay_active:
            elapsed = time.time() - continuous_start_time
            
            if continuous_delay_active:
                # Show 3-second delay countdown
                remaining = 3 - int(elapsed)
                if remaining > 0:
                    cv2.putText(display_frame, f"Starting in: {remaining}", (180, 200),
                               cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 165, 0), 3)
                else:
                    # Delay complete, start continuous capture
                    continuous_delay_active = False
                    continuous_mode = True
                    continuous_start_time = time.time()
                    last_capture_time = time.time()
                    print("\nContinuous capture started!")
            
            elif continuous_mode:
                # Continuous capture active
                cv2.putText(display_frame, "CONTINUOUS CAPTURE", (150, 200),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
                cv2.putText(display_frame, "Press 'c' to stop", (200, 240),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                
                # Capture every 0.1 seconds
                current_time = time.time()
                if current_time - last_capture_time >= 0.1:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                    filename = f"data/{mode}/{mode}_{timestamp}.jpg"
                    cv2.imwrite(filename, frame)
                    continuous_files.append((filename, current_time))  # Store filename with timestamp
                    photo_count[mode] += 1
                    last_capture_time = current_time
        
        # Handle single capture countdown
        elif countdown > 0:
            elapsed = time.time() - countdown_start
            remaining = countdown - int(elapsed)
            
            if remaining > 0:
                # Display countdown
                cv2.putText(display_frame, str(remaining), (280, 240),
                           cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 0, 255), 8)
            else:
                # Capture photo
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                filename = f"data/{mode}/{mode}_{timestamp}.jpg"
                cv2.imwrite(filename, frame)
                photo_count[mode] += 1
                print(f"Captured: {filename}")
                countdown = 0
        
        # Add instructions
        cv2.putText(display_frame, "SPACE: countdown | ENTER: instant | c: continuous", (10, 450),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        cv2.imshow('Hand Gesture Capture', display_frame)
        
        # Handle keyboard input
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            break
        elif key == ord('p'):
            mode = 'pinched'
            print(f"\nSwitched to PINCHED mode")
        elif key == ord('u'):
            mode = 'unpinched'
            print(f"\nSwitched to UNPINCHED mode")
        elif key == ord(' ') and countdown == 0 and not continuous_mode and not continuous_delay_active:
            countdown = 3
            countdown_start = time.time()
            print(f"\nCountdown started for {mode} gesture...")
        elif key == 13 and not continuous_mode and not continuous_delay_active:  # Enter key
            # Capture photo instantly
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            filename = f"data/{mode}/{mode}_{timestamp}.jpg"
            cv2.imwrite(filename, frame)
            photo_count[mode] += 1
            print(f"Captured instantly: {filename}")
        elif key == ord('c'):
            if continuous_mode:
                # Stop continuous capture and delete ONLY last 3 seconds of images
                current_time = time.time()
                cutoff_time = current_time - 3.0  # 3 seconds ago
                
                files_to_delete = []
                files_to_keep = []
                
                for filename, capture_time in continuous_files:
                    if capture_time >= cutoff_time:
                        files_to_delete.append(filename)
                    else:
                        files_to_keep.append((filename, capture_time))
                
                print(f"\nStopping continuous capture.")
                print(f"Keeping {len(files_to_keep)} images, deleting last {len(files_to_delete)} images (last 3 seconds)...")
                
                deleted_count = 0
                for filename in files_to_delete:
                    try:
                        if os.path.exists(filename):
                            os.remove(filename)
                            photo_count[mode] -= 1
                            deleted_count += 1
                    except Exception as e:
                        print(f"Error deleting {filename}: {e}")
                
                print(f"Deleted {deleted_count} images from the last 3 seconds")
                print(f"Kept {len(files_to_keep)} images")
                continuous_mode = False
                continuous_files.clear()
            elif not continuous_delay_active:
                # Start continuous capture mode with 3-second delay
                continuous_delay_active = True
                continuous_start_time = time.time()
                continuous_files.clear()
                print(f"\nStarting continuous capture in 3 seconds...")
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    
    print("\n" + "=" * 50)
    print("Capture session complete!")
    print(f"Total pinched images: {photo_count['pinched']}")
    print(f"Total unpinched images: {photo_count['unpinched']}")
    print("=" * 50)

if __name__ == "__main__":
    capture_gestures()