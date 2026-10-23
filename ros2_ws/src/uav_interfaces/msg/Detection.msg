# One perceived object, frame pixels (origin top-left). Mirrors uav_core.types.Detection.
string cls          # "circle" | "semicircle" | "H_landing"
string color        # "red" | "yellow" | "blue" | "" (unknown or not a circle)
int32 cx
int32 cy
int32 radius
int32[4] bbox       # x1, y1, x2, y2
float32 fill_ratio
float32 conf
bool edge_clipped   # touches the frame border: radius not trusted for altitude
