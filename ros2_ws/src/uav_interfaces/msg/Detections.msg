# header.stamp is the CAPTURE time of the frame (copied from the image), not the publish
# time: the mission synchronises the IMU to it and uses it as its clock (sim time works).
std_msgs/Header header
uint32 width        # frame size the pixel coordinates refer to
uint32 height
Detection[] detections
