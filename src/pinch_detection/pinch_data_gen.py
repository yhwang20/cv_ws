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
    
    print("Hand Gesture Training Data Collector")
    print("=" * 50)
    print("Controls:")
    print("  'p' - Switch to PINCHED mode")
    print("  'u' - Switch to UNPINCHED mode")
    print("  SPACE - Start 3-second countdown to capture")
    print("  ENTER - Capture instantly")
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
        
        # Handle countdown
        if countdown > 0:
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
        cv2.putText(display_frame, "SPACE: countdown | ENTER: instant", (10, 450),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
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
        elif key == ord(' ') and countdown == 0:
            countdown = 3
            countdown_start = time.time()
            print(f"\nCountdown started for {mode} gesture...")
        elif key == 13:  # Enter key
            # Capture photo instantly
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            filename = f"data/{mode}/{mode}_{timestamp}.jpg"
            cv2.imwrite(filename, frame)
            photo_count[mode] += 1
            print(f"Captured instantly: {filename}")
    
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