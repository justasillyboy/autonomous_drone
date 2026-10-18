# Thiết kế nâng cấp: Autonomous Fire-Fighting UAV

- **Ngày:** 2026-10-18
- **Trạng thái:** Chờ chủ dự án duyệt
- **Repo:** `github.com/justasillyboy/autonomous_drone` (clone local: `~/dev/autonomous_drone`)
- **Nguồn code cũ:** `~/Documents/AI/UAV_Competition_NCKH/` (giữ nguyên cho tới mốc M6)

---

## 1. Mục tiêu

Biến dự án thi "UAV Cứu Hỏa Cứu Hộ 2026" thành một dự án portfolio cá nhân, public, chứng minh được
năng lực ở **cả ba mảng**: Robotics/Autonomy, Computer Vision/ML, Edge AI. Mỗi mảng phải có **một câu
khẳng định đi kèm số đo thật**, không có khẳng định nào thiếu bằng chứng:

| Mảng | Câu khẳng định mục tiêu | Bằng chứng |
|---|---|---|
| Robotics | Nhiệm vụ tự hành trọn vẹn (cất cánh → 5 lần thả → hạ cánh) chạy kín vòng trong Gazebo + ArduPilot SITL, tỉ lệ thành công X% trên N ≥ 50 lượt ngẫu nhiên hoá | Referee độc lập trong sim, file kết quả JSON từng lượt, video demo |
| CV / ML | Nghiên cứu sim-to-real và ablation loss trên tập test ảnh thật đóng băng | Bảng mAP / Boundary F1 / Hausdorff, có seed và khoảng tin cậy |
| Edge AI | Model INT8 chạy trên Hailo-8L, vòng end-to-end ≥ 20 Hz, có bảng latency từng tầng | Benchmark p50/p95/p99 đo trên Pi 5, có cả chế độ HIL |

### Ràng buộc đã chốt

- Dự án **cá nhân**, repo **public**.
- Thời gian **hơn 6 tuần** (kế hoạch 8 tuần, mục 13).
- Phần cứng: **Raspberry Pi 5 + Hailo-8L** và Mac M1 Pro 16 GB. **Không có drone**, nên phần bay
  được chứng minh bằng mô phỏng và hardware-in-the-loop.
- **Hướng A:** Gazebo Harmonic + ArduPilot SITL + ROS 2, có chế độ HIL với Pi 5.
- File PDF thể lệ và bản dịch thể lệ của BTC **không** đưa lên repo public. README chỉ tóm tắt luật.
- Docs hướng ra ngoài (README, báo cáo, ADR) viết **tiếng Anh**, vì nhà tuyển dụng đọc chúng. Spec và
  plan nội bộ viết tiếng Việt.

### Ngoài phạm vi (YAGNI)

Bay thật; hỗ trợ PX4; điều khiển học bằng RL; nhiều drone; web dashboard; nền tảng MLOps
(chỉ dùng config trong git + CSV kết quả, W&B là tuỳ chọn bật bằng flag).

---

## 2. Hiện trạng (khảo sát ngày 2026-10-18)

**Điểm mạnh, giữ lại:**
- Pipeline CV đầy đủ: bộ sinh ảnh synthetic (`dataset.py`), 286 ảnh thật có nhãn (Roboflow),
  YOLO11n-seg với boundary/Dice loss, đánh giá bằng Boundary F1 và Hausdorff, export ONNX.
- State machine 8 phase: lọc median ToF, đồng bộ IMU theo timestamp frame, feedforward cho F2
  di động, attitude gate khi thả, geofence dead-reckoning.
- Kiến trúc ROS 2 năm node: QoS có cân nhắc, watchdog, e-stop, GUIDED-mode gate.

**Lỗi / nợ kỹ thuật cần xử lý:**

| # | Vấn đề | Hậu quả |
|---|---|---|
| D1 | Chưa từng chạy end-to-end: không có `.hef`, `calibration.npz`, `hsv_calib.json` | Không có số đo nào |
| D2 | `uav_deploy` là package `ament_python` nhưng chứa `msg/` | rosidl không sinh message, nên `from uav_deploy.msg import …` lỗi. Package fallback `uav_interfaces` không tồn tại |
| D3 | `mission_control_node` vẽ lại mask giả (`_ShimResult`) để gọi `mission.update()` | Hình học bị tính hai lần, mission node phải nhận cả ảnh, hai node phụ thuộc ngược |
| D4 | `mission.py` dài 3.095 dòng: `_fire_phase` 547 dòng, `update` 435 dòng, mọi phase nhận `frame` + YOLO `result` | Perception lẫn với ra quyết định, không unit test được |
| D5 | Thời gian tính bằng số frame (`fps_hint`) | Sai khi FPS dao động; hỏng khi sim chạy chậm hơn thời gian thực |
| D6 | Config chỉnh qua việc ghi đè biến global của module; phải load **trước** khi khởi tạo | Bẫy thứ tự gọi, phải đồng bộ tay giữa `config.yaml`, README và code |
| D7 | **Nghi vấn:** `action_mavros_node` đẩy `vx, vy, vz, yaw_rate` theo body-NED thẳng vào `/mavros/setpoint_velocity/cmd_vel_unstamped`. MAVROS nhận theo quy ước ROS (ENU/FLU), mặc định ở khung world | Có thể đảo dấu z/y/yaw (lệnh hạ thành bay lên) và sai khung body/world. **Xác minh trong SITL ở M2** |
| D8 | `data.yaml` trỏ tới `valid/`, `test/` nhưng chỉ có `train/` | Không có tập ảnh thật để đánh giá độc lập |
| D9 | 3 bản sao code (chỉ khác khoảng trắng) cùng 1 bản cũ hơn, không có git; `.venv` Linux/CUDA 5.2 GB nằm trong `~/Documents` đang sync iCloud | Lộn xộn, tốn đĩa, lệnh shell bị treo |

**Nguồn sự thật cho code cũ:** `fixed_yolo_circle_seg/files/` (giống hệt bản trong `deploy_files_…`
và `UAV_node_ROS2/src/uav_deploy/main/` sau khi bỏ khác biệt khoảng trắng), cộng thêm package
`UAV_node_ROS2/src/uav_deploy/` (các node, 2026-04-21). Bản `yolo_circle_seg/` (2026-04-13) đã bị
thay thế, chỉ lấy asset riêng của nó. Folder Google Drive "NodeRos2_TuHanh" là bản sao của
`deploy_files_drone_competition` (đã đối chiếu từng file node).

---

## 3. Cấu trúc repo đích

```
autonomous_drone/
├── README.md                 # giới thiệu, GIF demo, bảng kết quả, sơ đồ kiến trúc (EN)
├── LICENSE                   # Apache-2.0 (khớp package.xml cũ)
├── pyproject.toml            # uv workspace: uav_core, ml, edge, tools
├── .pre-commit-config.yaml   # ruff, ruff-format, end-of-file, check-added-large-files (5 MB)
├── docs/
│   ├── architecture.md       # sơ đồ ROS graph, data flow, ngân sách thời gian
│   ├── competition-rules.md  # tóm tắt luật (EN), không kèm PDF BTC
│   ├── config-reference.md   # SINH TỰ ĐỘNG từ schema config (mục 5); CI kiểm tra luôn khớp
│   ├── decisions/            # ADR-0001…, chuyển từ "Architecture Decisions" trong CLAUDE.md cũ
│   ├── report/               # báo cáo kỹ thuật (mục 12)
│   └── superpowers/{specs,plans}/
├── legacy/                   # code cũ, nguyên trạng; bị xoá ở cuối M1 (mục 4)
├── uav_core/                 # Python thuần, KHÔNG import ROS / cv2-GUI / torch
│   ├── src/uav_core/
│   │   ├── config.py         # schema pydantic: MissionConfig, RulebookConfig
│   │   ├── types.py          # Detection, SensorSnapshot, FlightCommand, MissionStatus
│   │   ├── frames.py         # chuyển đổi body-FRD ↔ ROS-FLU / ENU, một nơi duy nhất
│   │   ├── perception/       # geometry.py, color.py, filters.py, altitude_vision.py
│   │   ├── estimation/       # altitude.py (thứ tự ưu tiên), tof_filter.py, imu_sync.py, target_motion.py
│   │   ├── control/          # pid.py, feedforward.py, drop_gate.py
│   │   ├── mission/          # phases/*.py, state_machine.py, scoring.py, geofence.py, timers.py
│   │   └── viz/hud.py        # vẽ HUD (cv2), tách khỏi phần ra quyết định
│   └── tests/
├── ml/
│   ├── src/uav_ml/           # synth/ (từ dataset.py), train.py, loss.py, evaluate.py, export.py
│   ├── configs/              # mỗi thí nghiệm một file YAML
│   ├── experiments/          # script E1–E4 + kết quả CSV + script vẽ hình
│   └── tests/
├── edge/
│   ├── hailo/                # notebook compile cho Colab (x86), script trích frame calib, alls/yaml
│   ├── runtime/              # backend: hailo.py, onnx.py, ultralytics.py (cùng một interface)
│   └── benchmarks/           # script đo + kết quả
├── ros2_ws/src/
│   ├── uav_interfaces/       # ament_cmake: Detection.msg, Detections.msg, FlightCommand.msg, MissionStatus.msg
│   ├── uav_perception/       # perception_node (+ camera v4l2 tích hợp), camera_node, hud_node (đều tuỳ chọn)
│   ├── uav_mission/          # mission_node: lớp vỏ mỏng quanh uav_core
│   ├── uav_hardware/         # bno080_node, action_node (MAVROS)
│   ├── uav_sim/              # sim_referee, ball_dropper, target_mover, scenario_runner
│   └── uav_bringup/          # launch real / sim / hil + file params
├── sim/
│   ├── worlds/competition_field.sdf.jinja   # template, các kịch bản điền vào
│   ├── models/               # fire_circle_{red,yellow,blue}, tz_gate, helipad, ball, iris_downcam
│   └── scenarios/            # schema ngẫu nhiên hoá + seed
├── tools/                    # camera_calibration.py, hsv_calibrator.py, bag_report.py (từ logger.py)
├── docker/                   # sim.Dockerfile, pi.Dockerfile, compose.yaml, cyclonedds.xml
├── scripts/                  # fetch_assets.sh, run_monte_carlo.sh
└── .github/workflows/        # ci.yml, sim-smoke.yml (chạy tay / theo lịch)
```

**Quy tắc phụ thuộc** (CI kiểm tra bằng `import-linter`):
- `uav_core` không phụ thuộc gì trong repo.
- `ml`, `edge`, `tools` → `uav_core`.
- Các package ROS → `uav_core` (+ `edge.runtime` cho perception).
- Không ai được import ngược vào `legacy/`. `legacy/` chỉ được dùng trong characterization test.

---

## 4. Chuyển code cũ sang repo

1. **Commit `chore: import legacy code as-is`.** Chép code cũ vào `legacy/` giữ nguyên bố cục phẳng:
   `legacy/vision/` lấy từ `fixed_yolo_circle_seg/files/*.py|*.yaml|*.md|*.sh`,
   `legacy/ros2/uav_deploy/` lấy từ `src/uav_deploy` (bỏ `build/ install/ log/ __pycache__`).
   Không sửa một byte nào, để các commit refactor sau đó có diff sạch.
2. **Không đưa file nhị phân vào git.** `.gitignore` chặn `*.pt *.onnx *.hef *.mp4 *.zip *.npz`,
   `data/`, `runs/`, `.venv/`, `docs/private/`. Các asset upload lên **GitHub Release
   `legacy-assets-v0`**; `scripts/fetch_assets.sh` tải về và kiểm sha256:
   - `best.pt` (5.7 MB), `best.onnx` (38.6 MB)
   - `dataset.zip` (đóng gói lại, bỏ `__MACOSX`)
   - `real_domain_images_new.zip` (300 frame thật chưa gán nhãn)
   - `hailo_calib.zip`
   - các video kết quả khác nhau: `output_{2,3,7,8}`, `final_flight_result{,_new}`, `ket_qua`, `result{,_new}`
3. **Chỉ giữ local:** PDF thể lệ và `Thể lệ_dịch.docx` → `docs/private/` (gitignore).
4. **Thay thế dần:** mỗi module mới của `uav_core` hoặc `ml` thay một phần của `legacy/`. Hết M1,
   `legacy/vision/mission.py` và `legacy/ros2/` bị xoá; hết M3, các script ML trong legacy bị xoá.
   Lịch sử git là bằng chứng của quá trình nâng cấp.
5. **Dọn `~/Documents`** (mốc M0, cần chủ dự án duyệt danh sách): chuyển vào **Thùng rác**, không xoá
   vĩnh viễn, sau khi so sha256 với bản trong repo/Release:
   - `.venv` 5.2 GB
   - `UAV_node_ROS2.zip` 2.8 GB (đã có trên Drive)
   - các zip trùng, `__MACOSX`, `build/install/log/__pycache__`

   Việc này cũng giải phóng đĩa cho image Docker, vì ổ hiện chỉ còn ~33 GB trống.

---

## 5. Config: một nguồn sự thật duy nhất

- `uav_core.config.MissionConfig`: model pydantic v2 `frozen=True`. Giá trị mặc định là các hằng số
  hiện tại (`DROP_ALTITUDE_MIN=3.05`, `ALIGN_PIXEL_THRESH=25`…).
- `RulebookConfig` tách riêng, chỉ đọc: bảng điểm vòng loại/chung kết, kích thước sân, đường kính
  vòng tròn.
- Load từ YAML **một lần ở biên** (launch hoặc CLI), rồi **truyền tường minh** vào constructor.
  Không ghi đè biến global, nên bẫy "phải load trước khi khởi tạo" (D6) biến mất.
- **Validator fail-fast:** `drop_altitude_min < 3.05` gây lỗi lúc khởi động, không tự sửa ngầm giữa
  trận như REQ-NF-04 cũ. Lý do: config sai phải bị phát hiện trên mặt đất.
- `docs/config-reference.md` sinh từ schema (tên, mặc định, đơn vị, mô tả, nguồn luật). CI fail nếu
  file lệch với schema. Cơ chế này thay cho quy tắc tay "sửa cả config.yaml lẫn README" của CLAUDE.md cũ.

---

## 6. `uav_core`: tách `mission.py`

### Hợp đồng dữ liệu (`types.py`, đều là dataclass `frozen`)

```python
Detection(cls: TargetClass, color: Color | None, cx: float, cy: float, radius: float,
          bbox: tuple[float, float, float, float], fill_ratio: float, conf: float,
          edge_clipped: bool, z_vision_m: float | None)
SensorSnapshot(stamp_s: float, fc_alt_m, tof_alt_m, vel_body_ms: tuple | None,
               rpy_deg: tuple | None, rates_dps: tuple | None)   # mọi trường có thể None
FlightCommand(vx, vy, vz, yaw_rate_dps, yaw_lock, drop_ball, phase, action)  # body FRD
MissionStatus(phase, score, penalties, balls_dropped, targets_done, alt_source, elapsed_s)
```

### API cốt lõi

```python
sm = MissionStateMachine(cfg: MissionConfig, rules: RulebookConfig, round: Round)
out = sm.step(t_s: float, detections: Sequence[Detection], sensors: SensorSnapshot,
              frame_size: tuple[int, int]) -> StepOutput(command, status)
sm.reset(); sm.skip_takeoff(); sm.signal_landed(); sm.start_match(t_s)
```

- `step()` **không nhận ảnh** và không vẽ gì. HUD chuyển sang `uav_core.viz.hud`, dùng bởi
  `hud_node` (tuỳ chọn) hoặc tool offline.
- **Thời gian tính bằng giây** theo timestamp đầu vào (D5). Bỏ `fps_hint`. Mọi cửa sổ (phạt độ cao
  5 s, cooldown thả 2 s, recovery, nới tilt 1.5 s) là timer theo `t_s`. Chạy đúng cả với sim time.
- Mỗi phase là một module (`phases/takeoff.py`, `search.py`, `targeting.py`, `drop.py`,
  `recovery.py`, `helipad.py`, `landing.py`). Hàm thuần dạng
  `(ctx, inputs) -> (command, transition | None)`.
- Trạng thái nội bộ có thể thay đổi nằm gọn trong `MissionStateMachine`; đầu vào và đầu ra luôn
  bất biến.
- Hình học (`_extract_geometry`, fill ratio, chặn trùng H_landing, màu HSV) chuyển sang
  `uav_core.perception`. **Chỉ perception_node** gọi các hàm này, nên hết tính hai lần (D3).

### Giữ nguyên hành vi: characterization test **trước** khi refactor

1. Viết harness chạy `legacy/vision/mission.py` trên **các kịch bản có kịch bản sẵn**: frame render
   bằng bộ synthetic của `dataset.py` cùng chuỗi cảm biến giả lập (cất cánh, căn chỉnh, thả, mất
   mục tiêu, ToF nhiễu đột biến, F2 di động, hết giờ). Ghi lại chuỗi `FlightCommand` thành golden file.
2. Code mới phải tái tạo golden (sai số nhỏ cho float), **trừ** những thay đổi có chủ đích được
   ghi trong ADR: thời gian theo giây, config fail-fast. Với các thay đổi này, golden được cập nhật
   kèm lý do trong commit.
3. Unit test cho từng module. Coverage `uav_core` ≥ 80%. Dùng `hypothesis` cho `frames.py` (chuyển
   đổi qua lại ra lại đúng giá trị ban đầu) và geometry (mask tròn tổng hợp với nhiều kích thước,
   có cắt biên).

---

## 7. ROS 2

- **Phiên bản:** ROS 2 **Jazzy** + Gazebo **Harmonic**. Đây là cặp được hỗ trợ chính thức, có
  `ros_gz`, `ardupilot_gazebo` và MAVROS bản binary ổn định.
- **`uav_interfaces` (ament_cmake)** sửa D2. Gồm `Detection.msg`, `Detections.msg`
  (`std_msgs/Header` chứa stamp lúc chụp frame), `FlightCommand.msg`, `MissionStatus.msg`. Service
  dùng `std_srvs/Trigger`.
- **`perception_node`:**
  - Nguồn ảnh `input: topic | v4l2`. Chế độ `v4l2` tự đọc camera trên Pi để tránh copy ảnh qua DDS.
  - Undistort bằng map tính sẵn (`initUndistortRectifyMap`).
  - Backend `hailo | onnx | ultralytics` qua `edge.runtime`.
  - Gọi `uav_core.perception` rồi publish `/perception/detections` (Reliable, depth 1) và
    `/perception/debug_image` (giới hạn tần số, chỉ khi có subscriber).
  - Lấy `f_px` từ `CameraInfo`.
- **`mission_node`:**
  - Subscribe `/perception/detections`, IMU, MAVROS (rel_alt, rangefinder, velocity_body, state,
    extended_state).
  - Đưa cảm biến vào `imu_sync` theo stamp. Gọi `sm.step()`. Publish `/mission/command`
    (Reliable) và `/mission/status`.
  - **Không subscribe ảnh.** Giữ watchdog 0.5 s, `/mission/reset`, `/mission/e_stop`.
  - Đồng hồ trận bắt đầu ở cạnh lên ARM.
  - `use_sim_time` hoạt động đúng nhờ thời gian theo stamp.
- **`action_node`:**
  - Chuyển `FlightCommand` (body FRD) sang lệnh MAVROS bằng **một hàm duy nhất trong
    `uav_core.frames`**. Gửi qua `/mavros/setpoint_raw/local` (`PositionTarget`, khung body,
    type_mask chỉ dùng vận tốc + yaw rate), để khung tọa độ được khai báo rõ thay vì phụ thuộc
    tham số `mav_frame`.
  - Giữ GUIDED gate. Servo thả qua `MAV_CMD_DO_SET_SERVO`.
  - Xử lý D7: viết test cho phép đổi khung, rồi xác nhận bằng SITL. Nếu SITL chứng minh code cũ sai
    dấu, ghi vào báo cáo như một bug thật mà mô phỏng đã bắt được.
- **`bno080_node`:** giữ như cũ, mặc định 50 Hz (theo feedback cũ về tải I2C).
- **Launch:** `real.launch.py`, `sim.launch.py`, `hil.launch.py`. Cùng một bộ node, chỉ khác params.
- **Test:** `launch_testing` cho chuỗi detections giả → mission_node → command. Test chuyển khung cho
  action_node. Dùng `colcon test` trong CI.

---

## 8. Mô phỏng (Gazebo Harmonic + ArduPilot SITL)

- **Môi trường:** `docker/sim.Dockerfile` (Ubuntu 24.04, multi-arch arm64/amd64): ROS 2 Jazzy,
  Gazebo Harmonic, ArduCopter SITL (pin một tag Copter-4.x ổn định lúc bootstrap),
  `ardupilot_gazebo`, MAVROS. `compose.yaml` gồm các service `sim`, `stack`, `novnc` (tuỳ chọn, để
  quay video). Chạy lockstep: render chậm chỉ làm hệ số thời gian thực (RTF) giảm, vòng điều khiển
  vẫn đúng.
- **World `competition_field.sdf.jinja`:**
  - Sàn 24 × 12 m, Safe Zone 12 × 3.5 m có helipad H.
  - 5 vòng tròn mục tiêu theo luật: đỏ 1.0 m ×2, vàng 0.6 m ×2, xanh 0.4 m ×1, đặt trên bục 0.1 m.
  - Cổng TZ: vòng trong 1.6 m, tâm cao 1.7 m, có contact sensor.
  - Ánh sáng và vật liệu là tham số.
- **Drone:** `iris_downcam` = iris của `ardupilot_gazebo` cộng camera nhìn xuống 640 × 480, FOV và hệ
  số méo lấy gần với IMX219 góc rộng. Như vậy pipeline undistort cũng được kiểm tra trong sim.
- **Node trong `uav_sim`:**
  - `target_mover`: F2 di chuyển 0.1 m/s ở chế độ chung kết, qua service `set_pose`.
  - `ball_dropper`: theo dõi `/mavros/rc/out` trên kênh servo; khi PWM vượt ngưỡng thì spawn quả bóng
    có vật lý tại vị trí drone. Nhờ đó `action_node` giữ nguyên hành vi như khi bay thật.
  - `sim_referee`: chấm điểm **độc lập** với điểm tự báo của mission, dựa trên pose thật từ Gazebo:
    - cất cánh ≥ 3.0 m
    - bóng rơi trong bán kính vòng
    - phạt độ cao < 3.0 m quá 5 s
    - ra khỏi biên quá 5 s
    - va chạm TZ
    - hạ cánh trong helipad
    - giới hạn 300 s

    Ghi `results/<run_id>.json`.
  - `scenario_runner`: từ seed, sinh vị trí mục tiêu không chồng lên nhau, ánh sáng (hướng/cường độ),
    tông màu sàn, gió (`SIM_WIND_SPD/DIR`), nhiễu cảm biến (tham số SITL), vòng loại hoặc chung kết.
- **Monte Carlo:** `scripts/run_monte_carlo.sh` chạy N lượt headless. Báo cáo gồm tỉ lệ thành công
  (đủ 5 lần thả và hạ cánh trong ≤ 300 s), phân bố điểm (tối đa 200), phân loại lỗi (mất mục tiêu,
  thả trượt, phạt độ cao, hết giờ…), và so điểm referee với điểm mission tự báo.

---

## 9. Hardware-in-the-loop (Pi 5 + Hailo-8L)

- **Mac:** Gazebo + SITL. **Pi 5:** MAVROS + perception (backend Hailo) + mission + action. Cách chia
  này giống hệt lúc bay thật, khi Pi nói chuyện với FC qua MAVLink. MAVROS trên Pi nối tới SITL qua UDP.
- **Ảnh từ Gazebo sang Pi:** `image_transport` dạng compressed (JPEG) qua **cáp Ethernet trực tiếp**.
  Cyclone DDS cấu hình unicast peer. Đo **riêng** độ trễ thêm do truyền ảnh để không lẫn vào latency
  của perception.
- **Pi:** Raspberry Pi OS Bookworm làm host (Hailo hỗ trợ chính thức) và `docker/pi.Dockerfile`
  (`ros:jazzy` + HailoRT), truyền `--device /dev/hailo0` vào container. **Phiên bản HailoRT trong
  container phải khớp chính xác driver PCIe trên host**; pin cả hai.

---

## 10. CV / ML

### Dữ liệu và giao thức đánh giá (sửa D8)

- **Ba miền dữ liệu:**
  1. synthetic procedural (`uav_ml.synth`, từ `dataset.py`)
  2. **render từ Gazebo, tự gán nhãn** từ pose đã biết (thêm vào `uav_sim`)
  3. ảnh thật
- **Ảnh thật:** gán nhãn thêm 300 frame trong `real_domain_images_new/`. Model hiện tại gợi ý nhãn,
  sau đó sửa tay trên Roboflow.
- **Chia train/val/test theo phiên quay/video nguồn**, không chia ngẫu nhiên theo frame, để các frame
  liền nhau gần giống hệt không rơi vào cả train lẫn test. Tập test được **đóng băng**: danh sách
  file và sha256 commit trong `ml/configs/splits/`.
- **Chỉ số:** mask mAP50-95, Boundary F1 (2 px), Hausdorff, circularity, độ chính xác phân loại màu.
  Có **bootstrap 95% CI**, vì tập test thật nhỏ.

### Các thí nghiệm

| ID | Câu hỏi | Thiết kế |
|---|---|---|
| E1 | Cần bao nhiêu nhãn thật? (sim-to-real) | Train theo các tổ hợp {synth, gazebo, synth+gazebo} × {0, 25, 50, 100 %} lượng ảnh thật, thêm pretrain-synth → fine-tune-real. Vẽ đường cong theo lượng nhãn |
| E2 | Boundary/Dice loss tự viết có giúp ích không? | Baseline / +boundary / +Dice / +cả hai, mỗi cấu hình 3 seed, báo mean ± std. Báo cáo cả kết quả âm |
| E3 | Model nào cho đánh đổi tốt nhất trên Hailo-8L? | YOLO11n-seg, YOLO11s-seg, YOLOv8n-seg (Hailo Model Zoo hỗ trợ chính thức), và model seg mới nhất của Ultralytics lúc làm nếu compile được. Vẽ Pareto accuracy–latency |
| E4 | Kiểm chứng "Danger Zones" cũ | INT8 calibrate bằng frame thật vs frame synthetic; `mask_ratio` 4 vs 2 (FPS và băng thông PCIe). Biến các giả định trong CLAUDE.md cũ thành số đo |

- **Compute:** smoke test trên MPS của Mac. Các lượt train nhiều seed chạy trên GPU Colab/Kaggle.
  Mọi lượt chạy tái tạo được từ `ml/configs/*.yaml` + seed; kết quả là CSV, hình vẽ sinh bằng script.

---

## 11. Edge (Hailo-8L)

- **Compile:** ONNX → Hailo DFC (parse → optimize với tập calib thật → compile `.hef`) trong
  notebook `edge/hailo/compile.ipynb` chạy trên **Colab x86**. DFC không chạy được trên Mac M1.
  **Chủ dự án phải tự tải wheel DFC** từ Hailo Developer Zone (cần tài khoản).
  - Dự phòng: nếu YOLO11-seg dính op không hỗ trợ, dùng YOLOv8n-seg (đã nằm sẵn trong E3).
- **Runtime:** `edge.runtime.hailo` dùng HailoRT async `InferModel`. Hậu xử lý mask (proto ×
  coeffs, mask_ratio 4) bằng numpy trên CPU. Ba backend dùng cùng một interface
  `infer(frame) -> list[RawInstance]`.
- **Benchmark trên Pi 5:**
  - Latency từng tầng p50/p95/p99: chụp, tiền xử lý, NPU, hậu xử lý, hình học, `sm.step`, publish.
  - FPS end-to-end; %CPU.
  - Nhiệt độ và throttling khi chạy liên tục 10 phút.
  - So Hailo INT8 với ONNX CPU trên chính Pi 5.
  - Điện năng: tuỳ chọn, nếu có đồng hồ đo USB-C.
- **Kiểm chứng số liệu cũ:** con số "~2 ms inference" trong README cũ chưa từng được đo; thay bằng số đo.

---

## 12. Tài liệu và trình bày

- **README (EN):**
  - một đoạn giới thiệu
  - GIF demo (sim + HIL)
  - **bảng kết quả** với các con số của mục 1
  - sơ đồ kiến trúc (mermaid)
  - quickstart `docker compose up sim`
  - bản đồ repo
  - mục **Limitations** viết trung thực (không bay thật; tập test thật nhỏ)
- **Báo cáo kỹ thuật** `docs/report/` (Markdown → PDF, khoảng 8–10 trang): bài toán, hệ thống, E1–E4,
  kết quả vòng kín và HIL, hạn chế. Có thể đăng thành blog hoặc arXiv sau.
- **ADR:** chuyển 10 "Architecture Decisions" của CLAUDE.md cũ thành `docs/decisions/`, cộng các
  quyết định mới (thời gian theo giây, config fail-fast, chia khung trong `frames.py`, Jazzy/Harmonic, MAVROS).
- `CLAUDE.md` của repo viết lại theo cấu trúc mới.
- Video demo quay qua noVNC/Gazebo GUI, đăng YouTube và nhúng link.

---

## 13. Các mốc (8 tuần, một người; có chồng lấn)

Spec này quá lớn cho một plan duy nhất. Mỗi mốc có **plan riêng** trong `docs/superpowers/plans/`,
viết khi bắt đầu mốc đó. Plan đầu tiên gộp M0 + M1, vì mọi mốc sau đều phụ thuộc vào `uav_core` và
`uav_interfaces`.

| Mốc | Tuần | Đầu ra | Hoàn thành khi |
|---|---|---|---|
| M0 Bootstrap | 1 | import legacy, `.gitignore`, Release assets + `fetch_assets.sh`, pre-commit, CI (ruff + pytest), dọn `~/Documents` (sau khi được duyệt) | CI xanh trên `main`, asset tải về khớp sha256 |
| M1 Core | 1–3 | characterization golden → `uav_core` + config + frames + test; `uav_interfaces` + node mới + launch_testing; xoá `legacy/vision/mission.py` và `legacy/ros2` | golden khớp; coverage ≥ 80%; `colcon build && colcon test` xanh |
| M2 Sim | 2–4 | image Docker, world, iris_downcam, ball_dropper, referee, lượt kín vòng đầu tiên; xử lý D7 | 1 lượt vòng loại đi hết từ cất cánh tới hạ cánh trong sim, có JSON referee |
| M3 ML | 3–5 | gán nhãn + chia tập, ảnh Gazebo tự gán nhãn, E1–E3 | tập test đóng băng; bảng E1–E3 có CI |
| M4 Edge | 4–6 | `.hef`, backend Hailo, benchmark, E4 | bảng latency p50/p95/p99 trên Pi 5; ≥ 20 Hz hoặc ghi rõ lý do không đạt |
| M5 HIL + MC | 6–7 | `hil.launch.py`, Ethernet + DDS, Monte Carlo N ≥ 50 (sim) và N ≥ 10 (HIL) | tỉ lệ thành công + phân loại lỗi trong `results/` |
| M6 Trình bày | 7–8 | README, báo cáo, video, ADR, xoá các phần legacy còn lại | README có bảng số thật; báo cáo PDF; video công khai |

---

## 14. Rủi ro và cách giảm

| Rủi ro | Cách giảm |
|---|---|
| Render camera của Gazebo trong Docker trên M1 quá chậm (không có GPU) | Lockstep vẫn đúng; giảm độ phân giải/tần số camera; dự phòng chạy MC trên VM x86 cloud (dùng chung với Hailo DFC) |
| Hailo DFC cần x86 + tài khoản Hailo | Notebook Colab; chủ dự án tự tải wheel |
| YOLO11-seg dùng op mà Hailo không hỗ trợ | YOLOv8n-seg làm phương án dự phòng (nằm sẵn trong E3) |
| HailoRT trong container lệch phiên bản driver host | Pin phiên bản cả hai; script kiểm tra lúc khởi động container |
| Tập test thật nhỏ (khoảng vài trăm ảnh) | Bootstrap CI; chia theo video; ghi rõ trong Limitations |
| Phiên bản SITL/Gazebo/MAVROS thay đổi | Pin trong Dockerfile; tag image theo ngày |
| Đĩa Mac còn ~33 GB, image sim khoảng 8–10 GB | Dọn `~/Documents` ở M0; `docker system prune` định kỳ |
| Refactor làm hỏng hành vi đã có | Characterization golden test làm trước (mục 6) |

---

## 15. Tiêu chí hoàn thành toàn dự án

1. CI xanh trên `main`: ruff, import-linter, pytest (coverage `uav_core` ≥ 80%), `colcon test`,
   kiểm tra `config-reference.md` khớp schema.
2. Sim: N ≥ 50 lượt ngẫu nhiên, báo tỉ lệ thành công và điểm trung bình **do referee độc lập chấm**.
3. HIL: ít nhất 10 lượt với perception chạy trên Hailo-8L; latency vòng kín được đo và báo cáo.
4. ML: tập test thật đóng băng; E1–E4 có kết quả kèm seed/CI, kể cả kết quả âm.
5. README có GIF demo và bảng số thật; báo cáo kỹ thuật PDF; không còn khẳng định nào chưa được đo.
