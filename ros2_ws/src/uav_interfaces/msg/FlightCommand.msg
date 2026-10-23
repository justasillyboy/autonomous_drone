# Velocity setpoint in body FRD: +vx forward, +vy right, +vz DOWN, +yaw_rate clockwise.
# Convert with uav_core.frames.body_frd_to_flu before giving it to MAVROS (ADR-0003).
std_msgs/Header header
string phase
string action
float32 vx            # m/s
float32 vy            # m/s
float32 vz            # m/s
float32 yaw_rate_dps  # deg/s
bool yaw_lock
bool drop_ball        # release one ball now
