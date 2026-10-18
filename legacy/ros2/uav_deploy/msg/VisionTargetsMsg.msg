# header.stamp MUST carry the camera-grab timestamp (from /camera/image_raw.header.stamp).
# mission_control_node uses it to look up the IMU sample matching frame capture
# time via MissionStateMachine._get_synced_imu(); drifted stamps defeat the
# 70 ms camera-latency compensation and cause feedforward oscillation.
std_msgs/Header header
VisionTarget[] targets
