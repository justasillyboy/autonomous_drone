# FlightCommandMsg.msg
# Message definition for UAV flight commands

float32 vx        # Forward velocity (m/s)
float32 vy        # Lateral velocity (m/s)
float32 vz        # Vertical velocity (m/s, positive = descend)
float32 yaw_rate  # Rotation rate (deg/s, positive = clockwise)
bool yaw_lock     # True = hold current heading
bool drop_ball    # True = trigger servo/relay this frame
string phase      # Current phase name for logging
string action     # Human-readable action for HUD/logging