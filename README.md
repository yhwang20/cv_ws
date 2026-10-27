# cv_ws

Hand tracking & gesture recognition for teleoperating a UR5 robotic arm, built
for Johns Hopkins' EN.601.661 Computer Vision course.

A camera watches an operator's arm; the system tracks hand/wrist/arm keypoints
(custom ResNet+Transformer, YOLO-Pose, and MediaPipe implementations were all
evaluated) and a gesture classifier detects an open or closed hand, driving a
UR5 arm's motion and gripper in real time through ROS2 and MoveIt.

**Team:** Yusoo Hwang, Ethan Oh, Owen Reed, Mariana Smith.

This is a copy of the team's original repository, pushed here (with the
team's OK) so the code is visible from [Yusoo's portfolio](https://github.com/yhwang20).
The original lives under [mariana-smith/cv_ws](https://github.com/mariana-smith/cv_ws).
