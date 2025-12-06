import rclpy
import cv2
import numpy as np
import torch
import torch.nn as nn
from collections import deque
from cv_bridge import CvBridge
from torchvision import transforms, models

from rclpy.node import Node

from std_msgs.msg import Bool
from sensor_msgs.msg import Image


class PinchPublisher(Node):

    def __init__(self):
        super().__init__('minimal_publisher')
        
        #SETUP ROS FUNCTIONALITY
        self.publisher = self.create_publisher(Bool, 'is_pinching', 10)
        self.subscriber = self.create_subscription(Image,'rgb_image',self.image_callback,10)
        timer_period = 0.1
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.i = 0
        
        #SETUP INTERFACE
        self.bridge = CvBridge()
        self.last_frame = None
        
        #SETUP PYTORCH MODEL
        self.img_size = (224, 224) #TODO
        self.class_names = ['UNPINCHED', 'PINCHED']
        self.model = None
        self.prediction_history = deque(maxlen=10) 
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.load_model('../hand_gesture_model.pth')
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
    ########################################   
    #           ROS FUNCTIONS              #
    ########################################
    
    def image_callback(self, msg):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            self.last_frame = frame
        except Exception as e:
            self.get_logger().error(f"CV Bridge error: {e}")

    def timer_callback(self):
        if self.last_frame is None:
            self.get_logger().warn("No frame received yet.")
            return

        frame = self.last_frame
    
        prediction, confidence, raw_value, smoothed_bool = self.predict(frame) #TODO: GET ROS INPUT (Subscribe to a publisher?)
        
        self.publisher.publish(smoothed_bool)
        self.get_logger().info(f'Pinch Class: "{prediction}"\nPublishing: Bool={smoothed_bool}')
        self.i += 1
    
    ########################################   
    #  PyTorch Classification Functions    # 
    ########################################
    
    def preprocess_frame(self, frame):
        # Resize to model input size
        img = cv2.resize(frame, self.img_size)
        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Apply transforms (normalization)
        img = self.transform(img)
        
        # Add batch dimension
        img = img.unsqueeze(0)
        
        return img
        
    def predict(self, frame):
        preprocessed = self.preprocess_frame(frame)
        preprocessed = preprocessed.to(self.device)
        
        with torch.no_grad():
            prediction = self.model(preprocessed).cpu().numpy()[0][0]
        
        # Add to history for smoothing
        self.prediction_history.append(prediction)
        
        # Use average of recent predictions for stability
        smoothed_prediction = np.mean(self.prediction_history)
        
        # Determine class
        class_idx = 1 if smoothed_prediction > 0.5 else 0
        confidence = smoothed_prediction if class_idx == 1 else 1 - smoothed_prediction
        smoothed_bool = (smoothed_prediction > .5)
        
        return self.class_names[class_idx], confidence, smoothed_prediction, smoothed_bool

    def load_model(self, model_path):
        # Build model architecture
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
        
        # Load weights
        checkpoint = torch.load(model_path, map_location=self.device)
        base_model.load_state_dict(checkpoint['model_state_dict'])
        base_model.eval()
        
        self.model = base_model.to(self.device)
        
        # Get image size from checkpoint if available
        if 'img_size' in checkpoint:
            self.img_size = checkpoint['img_size']
        
        print(f"Model loaded successfully from {model_path}")
        print(f"Using input size: {self.img_size}")



def main(args=None):
    rclpy.init(args=args)

    minimal_publisher = PinchPublisher()

    rclpy.spin(minimal_publisher)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    minimal_publisher.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()