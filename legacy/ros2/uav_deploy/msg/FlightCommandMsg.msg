# Velocity command emitted by mission_control_node, consumed by action_mavros_node.
# Body-frame NED convention: +vx forward, +vy right, +vz down (descend),
# +yaw_rate clockwise. action_mavros_node MUST convert yaw_rate deg/s → rad/s
# via np.deg2rad() before publishing /mavros/setpoint_velocity/cmd_vel_unstamped.
std_msgs/Header header
float32 vx           # m/s, body-frame forward (+x = forward)
float32 vy           # m/s, body-frame right   (+y = right)
float32 vz           # m/s, body-frame down    (+vz = descend)
float32 yaw_rate     # deg/s, clockwise positive (action_mavros_node converts to rad/s)
bool    yaw_lock     # True = maintain current heading
bool    drop_ball    # True = release servo this frame
string  phase        # TAKEOFF / SEARCHING / TARGETING / DROPPING / FIND_HELIPAD / LANDING / COMPLETE / RECOVERY
string  action       # ALIGN / DESCEND / DROP / ASCEND / EDGE_CLIP / etc.
int32   score        # MissionStateMachine.score (not part of FlightCommand dataclass)
int8    balls_dropped  # MissionStateMachine.balls_dropped (not part of FlightCommand dataclass)
