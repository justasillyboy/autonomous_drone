# Kiến trúc ROS 2 — UAV Fire-Fighting Vision & Control (2026)

---

## 1. Sơ đồ Kiến trúc ROS 2 Tổng quan (Graph)

```
[camera_node] ──(sensor_msgs/Image)──────────────────────────────► [perception_node]
[camera_node] ──(sensor_msgs/CameraInfo)─────────────────────────► [perception_node]
                                                                          │
                                                            (uav_interfaces/VisionTargetsMsg)
                                                                          │
[bno080_imu_node] ──(sensor_msgs/Imu)──────────────────────────────────► ▼
[MAVROS] ─────────(mavros_msgs/RangeFinder, Odometry, State)──► [mission_control_node]
                                                                          │
                                                           (uav_interfaces/FlightCommandMsg)
                                                                          │
                                                                          ▼
                                                              [action_mavros_node]
                                                                          │
                                                           (geometry_msgs/Twist → FC)
                                                           (MAVROS CommandLong → Servo)
```

**Lỗi trong bản gốc:** Đồ thị cũ thiếu mũi tên từ `perception_node` → `mission_control_node`,
`bno080_imu_node` → `mission_control_node`, và `CameraInfo` từ `camera_node`. Đã bổ sung đầy đủ.

---

## 2. Chi tiết Từng Node

### Node 1: `camera_node` (Thu thập & Tiền xử lý ảnh)

**Nhiệm vụ:**
- Đọc frame từ `/dev/video0` (IMX219).
- Lock AWB và AEC bằng V4L2 (giống `hsv_calibrator.py`).
- Load `calibration.npz` và áp dụng `cv2.undistort()`.
- Publish:
  - `sensor_msgs/msg/Image` → `/camera/image_raw` (QoS: SensorData / Best Effort)
  - `sensor_msgs/msg/CameraInfo` → `/camera/camera_info`
    - Trường `K[0]` (fx) và `K[4]` (fy) chứa focal length tính từ `calibration.npz`.
    - **Đây là nguồn duy nhất cho f_px** — `perception_node` subscribe topic này để lấy
      focal length thay vì hardcode `DEFAULT_F_PX = 500.0`.

> **Bổ sung quan trọng (thiếu trong bản gốc):**
> `camera_node` phải publish `CameraInfo`. Nếu không có topic này,
> `perception_node` và `mission_control_node` sẽ dùng fallback `DEFAULT_F_PX = 500.0`
> — sai với bất kỳ camera thực nào và vi phạm nguyên tắc "Danger Zones" trong `CLAUDE.md`.

---

### Node 2: `perception_node` (Thị giác máy tính)

> **Giữ nguyên nguyên tắc gốc:** Tuyệt đối **không** chạy YOLO ở một node rồi publish mask
> sang node khác. Toàn bộ pipeline suy luận phải nằm trong một node duy nhất.

**Subscribe:**
- `/camera/image_raw`
- `/camera/camera_info` ← **mới** (để lấy f_px cho z_est_raw)

**Nhiệm vụ (theo thứ tự):**
1. Chạy `YOLO11n-seg` qua Hailo-8L (HEF model, `hailo_platform` API — xem ghi chú §3).
2. Gọi `collect_h_landing_bboxes()` để lấy danh sách bbox H_landing.
3. Với mỗi mask:
   a. Gọi `_extract_geometry()` → `cx, cy, radius, fill_ratio, cnt_frame, edge_clipped`.
   b. Lọc sớm (không publish nếu thất bại):
      - `fill_ratio < FILL_RATIO_MIN_CIRCLE (0.40)` → bỏ qua.
      - `2*radius / frame_w > MAX_FIRE_DIAMETER_RATIO (0.55)` → bỏ qua.
      - `overlaps_h_landing_bbox(cx, cy, radius, h_bboxes) = True` → bỏ qua.
   c. Gọi `color_of_mask()` → `color` (string: "red" / "yellow" / "blue" / None).
   d. Tính `z_est_raw = f_px * D_real / (2 * radius)` với `D_real` tra từ class/color.
      - **Quan trọng:** `z_est_raw` từ perception_node là **ước tính thô** — chỉ dùng khi
        không có FC altitude hay ToF. `mission_control_node` sẽ áp dụng altitude priority ladder.

**Publish:**
- `uav_interfaces/msg/VisionTargetsMsg` → `/perception/targets`
  (QoS: Reliable, depth=1, keep-last)
- `sensor_msgs/msg/Image` → `/perception/debug_image` (QoS: Best Effort, 5 Hz max, chỉ debug)

**Định nghĩa `VisionTargetsMsg.msg`:**

```
# uav_interfaces/msg/VisionTargetsMsg.msg
std_msgs/Header header          # header.stamp = thời điểm grab frame (quan trọng cho time-sync IMU)
uav_interfaces/Detection[] detections

# uav_interfaces/msg/Detection.msg
uint8   cls_id          # 0=circle, 1=semicircle, 2=H_landing
string  color           # "red" | "yellow" | "blue" | ""
float32 cx              # pixels — frame centre = (frame_w/2, frame_h/2)
float32 cy
float32 radius          # pixels
float32 fill_ratio      # mask pixels / solid circle area (đã qua lọc ≥ 0.40)
float32 z_est_raw       # metres — vision-only estimate (dùng làm fallback)
float32 conf            # YOLO confidence score
bool    edge_clipped    # True nếu circle chạm biên frame (radius untrustworthy)
```

> **Bổ sung quan trọng (thiếu trong bản gốc):**
> - `header.stamp` phải là timestamp lúc **grab frame** (không phải lúc publish).
>   `mission_control_node` dùng trường này để đồng bộ IMU qua `_get_synced_imu()`.
>   Nếu thiếu, toàn bộ cơ chế time-sync 70ms buffer bị vô hiệu.
> - `fill_ratio` cần publish để `mission_control_node` có thể log/debug, dù lọc đã xảy ra.
> - `conf` cần publish để tuning threshold sau bay thử.
> - Các bộ lọc FILL_RATIO, MAX_FIRE_DIAMETER, H_LANDING_OVERLAP **phải** chạy trong
>   `perception_node` (không phải `mission_control_node`) để giảm lưu lượng topic.

---

### Node 3: `sensor_bridge_node` (Tổng hợp phần cứng)

**Nhiệm vụ:**
- Đọc BNO080 ở tần số cao (100 Hz) qua I2C/UART.
- Publish `sensor_msgs/msg/Imu` → `/bno080/imu`:
  - `orientation` (quaternion) → Roll, Pitch, **Yaw** và Angular Rates.
  - `angular_velocity` → `pitch_rate_dps`, `roll_rate_dps` cho feedforward guard.

> **Bổ sung quan trọng (thiếu trong bản gốc):**
> `yaw_deg` là trường bắt buộc trong `SensorData` của `mission.py`.
> Hàm `_world_vel_to_body()` dùng `yaw_deg` để xoay vector vận tốc world→body.
> Hàm quản lý geofence dùng `(yaw_deg - _yaw_0)` để rotate optical flow.
> `sensor_bridge_node` phải publish đầy đủ quaternion để `mission_control_node`
> extract được `yaw_deg = euler_from_quaternion(msg.orientation).yaw`.

**MTF-01 (ToF + Optical Flow):** Dùng MAVROS (không cần node mới):
- `/mavros/distance_sensor/rangefinder_sub` → `tof_altitude_m`
- `/mavros/local_position/velocity_body` → `of_vx_ms`, `of_vy_ms`

---

### Node 4: `mission_control_node` (Máy trạng thái & PID)

**Subscribe:**
- `/perception/targets` (`VisionTargetsMsg`)
- `/bno080/imu` (`sensor_msgs/Imu`)
- `/mavros/distance_sensor/rangefinder_sub` → `tof_altitude_m`
- `/mavros/local_position/velocity_body` → `of_vx_ms`, `of_vy_ms`
- `/mavros/global_position/rel_alt` → FC altitude (ưu tiên cao nhất)
- `/mavros/state` → **mới** (detect ARM event để bắt đầu match timer)

**Parameters (ROS 2 params, set trong launch file):**
```yaml
takeoff_alt: 3.3
is_finals: false
hsv_calib: ""           # đường dẫn hsv_calib.json — load vào perception_node (xem ghi chú)
f_px_fallback: 500.0    # DEFAULT_F_PX — chỉ dùng nếu /camera/camera_info chưa nhận được
config_yaml: ""         # đường dẫn config.yaml cho tuning on-field — xem §2 Node 4 ghi chú mới
fps_hint: 20.0          # FPS thực tế của pipeline — bắt buộc để timing gates đúng
```

> **Bắt buộc load `config.yaml` trước khi khởi tạo `MissionStateMachine`.**
> Trong `mission_control_node.__init__()`, thứ tự phải là:
> ```python
> from mission import MissionStateMachine, load_mission_config_yaml
> if self.get_parameter('config_yaml').value:
>     load_mission_config_yaml(self.get_parameter('config_yaml').value)
> self.mission = MissionStateMachine(
>     f_px=..., is_finals=self.get_parameter('is_finals').value,
>     takeoff_alt=self.get_parameter('takeoff_alt').value,
> )
> self.mission.fps_hint = float(self.get_parameter('fps_hint').value)
> ```
> Gọi `load_mission_config_yaml()` **sau** khi đã tạo `MissionStateMachine` sẽ
> không cập nhật các biến instance như `_tof_spike_threshold_m`,
> `_ff_history_len`, `_drop_cooldown` (đọc từ module globals tại `__init__`).
> `fps_hint` sai → tất cả frame-count thresholds (altitude penalty, drop cooldown,
> recovery timer, tilt relax counter) bị lệch theo tỉ lệ — xem README.md §fps_hint.

**Nhiệm vụ:**
1. **Time-sync IMU:** Mỗi khi nhận `/bno080/imu`, gọi `push_sensor_data()` với
   `timestamp = msg.header.stamp`. Khi nhận `/perception/targets`, gọi `_get_synced_imu()`
   với `frame_timestamp = msg.header.stamp` để lấy IMU reading khớp với frame.

   > **Thay thế `message_filters::ApproximateTimeSynchronizer`:**
   > Cơ chế `deque` trong `_get_synced_imu()` của `mission.py` đã đủ chính xác và
   > nhẹ hơn trên Pi 5. **Giữ nguyên** cơ chế này — không cần ApproximateTimeSynchronizer.

2. **Match timer:** Bắt đầu đếm khi nhận `mavros_msgs/State.armed == True`.
   **Không** bắt đầu từ lúc node khởi động (drone có thể đã arm trước đó).
   Khi `time_elapsed_s >= LAND_EARLY_AT_S (285 s)` → chuyển sang `FIND_HELIPAD`.

3. **Altitude priority ladder** (áp dụng mỗi frame):
   ```
   fc_altitude  (từ /mavros/global_position/rel_alt)   ← ưu tiên cao nhất
       ↓ nếu None
   tof_altitude_m  (từ /mavros/distance_sensor/…)
       ↓ nếu None
   z_est_raw  (từ VisionTargetsMsg.detections[i].z_est_raw)   ← chỉ fallback
   ```

4. **Watchdog timeout:** Nếu không nhận được `/perception/targets` trong vòng 0.5 s
   (tương đương 10 frame ở 20 Hz), chuyển sang phase `RECOVERY` và publish lệnh hover
   (`vx=vy=vz=0`). Ngăn drone bay mất kiểm soát khi camera failure.

5. **Publish:** `uav_interfaces/msg/FlightCommandMsg` → `/mission/flight_command`
   (QoS: **Reliable**, depth=1 — **không** dùng Best Effort cho lệnh điều khiển)

**ROS 2 Services (mới — thiếu trong bản gốc):**

```
# Referee retry signal — gọi khi trọng tài cho phép retry
/mission/reset   [std_srvs/srv/Trigger]
    → gọi mission.reset() + mission.skip_takeoff() nếu drone đã ở độ cao

# Emergency hover — kill switch phần mềm
/mission/e_stop  [std_srvs/srv/Trigger]
    → publish FlightCommandMsg(vx=0,vy=0,vz=0,yaw_rate=0,drop_ball=False)
      và tạm dừng state machine cho đến khi /mission/reset được gọi
```

**Định nghĩa `FlightCommandMsg.msg`:**

```
# uav_interfaces/msg/FlightCommandMsg.msg
std_msgs/Header header
float32 vx           # m/s, body-frame forward (NED: +x = forward)
float32 vy           # m/s, body-frame right
float32 vz           # m/s, body-frame down (+vz = descend)
float32 yaw_rate     # deg/s, clockwise positive — convert with np.deg2rad() in action_mavros_node before publishing to /cmd_vel
bool    yaw_lock     # True = maintain current heading
bool    drop_ball    # True = release servo
string  phase        # TAKEOFF / SEARCHING / TARGETING / DROPPING / FIND_HELIPAD / LANDING / COMPLETE
string  action       # ALIGN / DESCEND / DROP / ASCEND / EDGE_CLIP etc. — từ FlightCommand.action
int32   score        # current competition score
int8    balls_dropped  # từ mission.balls_dropped (MissionStateMachine), không phải FlightCommand
```

---

### Node 5: `action_mavros_node` (Thực thi phần cứng)

> **Sửa lỗi typo:** Tên gốc viết "action_marvos_node" (sai) → đúng là `action_mavros_node`.

**Subscribe:** `/mission/flight_command`

**Nhiệm vụ:**
1. **Kiểm tra GUIDED mode** trước mỗi lần gửi lệnh:
   Subscribe `/mavros/state`. Nếu `state.mode != "GUIDED"`, không gửi velocity command.
   Log cảnh báo để field operator biết FC không ở đúng mode.

2. Chuyển `vx, vy, vz, yaw_rate` → `geometry_msgs/msg/Twist` publish vào
   `/mavros/setpoint_velocity/cmd_vel_unstamped`.

3. Khi `drop_ball == True`: Gọi MAVROS `CommandLong` service để kích hoạt servo AUX.

4. Lắng nghe `/mavros/state` để detect `LAND_COMPLETE`; khi phát hiện, publish
   `std_msgs/msg/Bool` → `/action/land_complete` để `mission_control_node` chuyển sang
   phase `COMPLETE`.

> **Bổ sung quan trọng (thiếu trong bản gốc):**
> GUIDED mode check là bắt buộc. Nếu FC ở STABILIZE hoặc ALT_HOLD, velocity setpoint
> không có tác dụng nhưng servo vẫn có thể kích hoạt — dẫn đến thả bóng ngoài vùng target.

---

## 3. Ghi chú Kỹ thuật Quan trọng

### 3a. Hailo-8L Python API trong ROS 2

Bản gốc chưa đề cập cách gọi Hailo-8L trong node. Quy trình:

```python
# Bước 1: Export model (chạy 1 lần trước thi)
# python export.py --weights best.pt --format onnx
# hailomz compile yolo11n_seg --ckpt best.onnx  → sinh ra best.hef

# Bước 2: Trong perception_node.__init__()
from hailo_platform import HEF, VDevice, HailoStreamInterface, InferVStreams, ConfigureParams
hef = HEF("best.hef")
target = VDevice()
network_group = target.configure(hef)[0]
# ... khởi tạo input/output vstreams

# Bước 3: Trong callback inference
# Chạy inference qua Hailo SDK, lấy output masks
# Sau đó gọi _extract_geometry() như bình thường
```

**Lưu ý:** `mask_ratio = 4` (trong `config.yaml`) là ràng buộc phần cứng PCIe Pi 5 ↔ Hailo-8L.
Không đổi thành 2 — sẽ bão hòa bus và FPS giảm xuống dưới 20 Hz.

### 3b. f_px và CameraInfo

- `perception_node` lấy `f_px` từ `/camera/camera_info` (trường `K[0]`).
- Nếu chưa nhận được `CameraInfo` (node vừa khởi động), dùng `DEFAULT_F_PX = 500.0` làm
  fallback tạm thời và log cảnh báo.
- **Không hardcode** `500.0` ở bất kỳ đâu trong logic chính — vi phạm Danger Zones trong
  `CLAUDE.md`.

### 3c. HSV Calibration

`hsv_calib.json` được load trong **`perception_node`** (nơi `color_of_mask()` chạy),
**không phải** `mission_control_node`. Truyền đường dẫn qua ROS 2 parameter:

```python
# perception_node.__init__()
hsv_calib = self.get_parameter('hsv_calib').value
if hsv_calib:
    from mission import load_hsv_calibration
    try:
        load_hsv_calibration(hsv_calib)
    except (FileNotFoundError, ValueError, KeyError) as e:
        self.get_logger().warn(f"HSV calib load failed: {e} — using built-in defaults")
```

### 3d. IPC / Zero-Copy cho Image topic

Để bật Zero-Copy thực sự giữa `camera_node` và `perception_node`:
- Dùng **Composable Nodes** (`rclcpp_components` hoặc tương đương rclpy) để chạy
  cả hai node trong **cùng một process**.
- Cấu hình `rmw_fastrtps` với Shared Memory transport trong `fastdds_config.xml`.
- Nếu không dùng Composable Nodes, Zero-Copy không áp dụng dù bật IPC.

### 3e. QoS Profile

| Topic | QoS |
|---|---|
| `/camera/image_raw` | Best Effort, depth=1 (tránh lag dồn frame) |
| `/camera/camera_info` | Reliable, transient local (latched) |
| `/perception/targets` | Reliable, depth=1, keep-last |
| `/perception/debug_image` | Best Effort, depth=1 |
| `/bno080/imu` | Best Effort, depth=5 (tốc độ cao, loss ok) |
| `/mission/flight_command` | **Reliable**, depth=1 ← **QUAN TRỌNG** |

> **Lỗi tiềm ẩn trong bản gốc:** Nếu `/mission/flight_command` dùng Best Effort,
> lệnh `drop_ball=True` có thể bị mất gói — servo không kích hoạt dù state machine đã ra lệnh.

---

## 4. Khuyến nghị Tối ưu (bổ sung và hiệu chỉnh)

### 4a. Time Synchronization

Giữ nguyên cơ chế `push_sensor_data()` + `_get_synced_imu()` từ `mission.py`.
Dùng `msg.header.stamp` (ROS time) làm timestamp thay cho `time.perf_counter()`.
**Không** cần `ApproximateTimeSynchronizer` — deque lookup đủ chính xác và nhẹ hơn trên Pi 5.

### 4b. ROS 2 Bag + logger.py

Dùng `ros2 bag record` để lưu raw topics:
```bash
ros2 bag record /mission/flight_command /perception/targets /bno080/imu \
                /mavros/distance_sensor/rangefinder_sub /mavros/global_position/rel_alt
```

**Giữ `logger.py`** làm post-processing script chạy trên bag file sau bay.
`logger.summary()` cung cấp các chỉ số thi đấu (score, balls_dropped, phase transitions,
mean FPS, drop events) mà `ros2 bag` không tự tổng hợp.

### 4c. Launch File (`bringup.launch.py`)

```python
# Thứ tự khởi động quan trọng:
# 1. MAVROS (phải connect FC trước)
# 2. camera_node
# 3. sensor_bridge_node
# 4. perception_node  (sau khi CameraInfo sẵn sàng)
# 5. mission_control_node  (sau khi tất cả publisher sẵn sàng)
# 6. action_mavros_node

# Parameters bắt buộc:
launch_args = [
    DeclareLaunchArgument('takeoff_alt',  default_value='3.3'),
    DeclareLaunchArgument('is_finals',    default_value='false'),
    DeclareLaunchArgument('hsv_calib',    default_value='/home/pi/hsv_calib.json'),
    DeclareLaunchArgument('calib_npz',    default_value='/home/pi/calibration.npz'),
    DeclareLaunchArgument('weights_hef',  default_value='/home/pi/best.hef'),
    DeclareLaunchArgument('log_csv',      default_value='/home/pi/run_001.csv'),
    DeclareLaunchArgument('config_yaml',  default_value='/home/pi/config.yaml'),  # on-field tuning
    DeclareLaunchArgument('fps_hint',     default_value='20.0'),                  # match actual pipeline FPS
]
```

---

## 5. Lộ trình Triển khai (giữ nguyên, bổ sung bước 0)

### Bước 0: Định nghĩa Interfaces (làm trước tất cả)

```bash
ros2 pkg create uav_interfaces --build-type ament_cmake
# Tạo: VisionTargetsMsg.msg, Detection.msg, FlightCommandMsg.msg
colcon build --packages-select uav_interfaces
```

### Giai đoạn 1: Implementation

(Như bản gốc — porting `_extract_geometry`, `color_of_mask`, `MissionStateMachine`)

**Bổ sung:** Thêm 2 ROS 2 services `/mission/reset` và `/mission/e_stop` vào
`mission_control_node` (xem §2 Node 4 ở trên).

### Giai đoạn 2: Kiểm thử Node-level

(Như bản gốc)

**Bổ sung:**
- Kiểm tra `/camera/camera_info` → `perception_node` nhận đúng `f_px`.
- Kiểm tra `header.stamp` trong `/perception/targets` có khớp với thời điểm grab frame.
- Kiểm tra `/mavros/state.armed` → match timer bắt đầu đúng lúc.

### Giai đoạn 3: Ground Testing (tháo cánh quạt)

(Như bản gốc)

**Bổ sung:**
- Test `/mission/reset` service: kích hoạt trong khi state machine đang ở TARGETING,
  xác nhận score = 0, phase = TAKEOFF, balls_dropped = 0.
- Test `/mission/e_stop` service: xác nhận `vx=vy=vz=0` được publish ngay lập tức.
- Kiểm tra GUIDED mode gate: đặt FC vào STABILIZE, xác nhận `action_mavros_node`
  không gửi velocity command (HUD trên debug_image log cảnh báo).

### Giai đoạn 4: Bay thử nghiệm và Tuning

(Như bản gốc)

**Bổ sung:**
- Xác nhận HUD debug_image hiển thị `[FC]` hoặc `[TOF]` — **không bao giờ `[VIS]`** khi bay.
- Sau mỗi chuyến bay, chạy `python logger.py --bag <path>` để xem summary thi đấu.

---

## 6. Tổng kết các Lỗi/Thiếu sót trong Bản gốc

| # | Vấn đề | Mức độ | File gốc liên quan |
|---|---|---|---|
| 1 | Đồ thị thiếu mũi tên sensor_bridge→mission, perception→mission | Trung bình | Sơ đồ §1 |
| 2 | Typo "action_marvos_node" | Nhỏ | §2 Node 5 |
| 3 | Không publish `CameraInfo` → f_px không đến được perception_node | **Nghiêm trọng** | §2 Node 1 |
| 4 | `VisionTargetsMsg` thiếu `header.stamp` → IMU time-sync bị vô hiệu | **Nghiêm trọng** | §2 Node 2 |
| 5 | `yaw_deg` chưa được đề cập rõ — bắt buộc cho geofence và _world_vel_to_body | Cao | §2 Node 3 |
| 6 | HSV calib phải load trong `perception_node`, không phải `mission_control_node` | Cao | §2 Node 2/4 |
| 7 | Không có service `/mission/reset` và `/mission/e_stop` | Cao | §2 Node 4 |
| 8 | Match timer phải bắt đầu từ ARM event, không phải node startup | Cao | §2 Node 4 |
| 9 | Watchdog timeout nếu perception_node ngừng publish | Cao | §2 Node 4 |
| 10 | `/mission/flight_command` phải dùng QoS Reliable | **Nghiêm trọng** | §2 Node 4/5 |
| 11 | Không check GUIDED mode trước khi gửi velocity command | Cao | §2 Node 5 |
| 12 | Không đề cập Hailo-8L HEF API | Trung bình | §3a |
| 13 | Composable Nodes là điều kiện cho Zero-Copy thực sự | Trung bình | §3d |
| 14 | `logger.py` cần giữ lại làm bag post-processor | Nhỏ | §4b |
| 15 | `load_hsv_calibration()` cần try/except (đã fix trong code) | Nhỏ | §3c |
| 16 | Thiếu `config_yaml` parameter → `config.yaml` không áp dụng runtime (chỉ dùng defaults) | **Nghiêm trọng** | §2 Node 4 |
| 17 | `fps_hint` phải set từ FPS camera thực tế — sai giá trị làm tất cả frame-count threshold lệch | Cao | §2 Node 4 |
