# Jetson ROS 2 LiDAR 차량 제어

Jetson 기반 소형 차량에서 ROS 2와 Slamtec LiDAR를 사용해 전방 장애물을 감지하고, PCA9685를 통해 ESC를 제어하는 작업공간입니다. 카메라 영상 발행, PWM 상태 확인, 조향 및 구동 시험 코드도 함께 들어 있습니다.

## LiDAR 장애물 회피 주행

`run_lidar_avoidance_rviz.sh`는 Slamtec C1, RViz, 장애물 회피 노드를 함께 실행합니다. 차량 전방은 장착 방향에 맞춰 LiDAR 180도를 사용합니다. 장애물을 만나면 여유가 큰 쪽으로 조향하고, 전방이 확보되면 즉시 중앙 조향으로 돌아가 직진합니다.

- 전방 0.80 m 이내: 최소 회피 공간 제한 없이 좌우 중 넓은 쪽으로 회피
- 선택한 통로가 열려 있으면 시간 제한 없이 계속 회피 주행
- 전방 0.95 m 이상이 연속 3회 확인되면 즉시 중앙 조향으로 직진
- 회피 중 통로가 막히거나 새 장애물이 나타나면 좌우 공간을 다시 비교
- 회피 및 중앙 조향 복귀 중 별도의 중립 대기 없이 계속 주행
- 전방에 장애물이 없으면 측면 거리와 관계없이 중앙 조향으로 계속 직진
- 장애물 거리로 정지하지 않고 좌우 회피 조향과 주행을 동시에 유지
- `/scan` 데이터 오류·단절·시작 전에도 정지하지 않고 중앙 조향으로 계속 직진
- 주행 속도: PCA9685 채널 8에 1565 us
- 조향: 왼쪽 1880 us, 중앙 1640 us, 오른쪽 1400 us
- `LEFT`, `RIGHT`, `STRAIGHT` 상태에서는 조향 중에도 주행 1565 us 유지

처음에는 반드시 바퀴를 띄워 조향 방향과 ESC 진행 방향을 확인하세요. 확인 후 충분히 넓고 사람이 없는 저속 시험 공간에서 실행합니다.

```bash
cd ~/ros2_ws
./run_lidar_avoidance_rviz.sh
```

실제 구동 전에 LiDAR와 판단만 확인하려면 다음처럼 실행합니다. RViz에 `/scan`이 표시되고 터미널에 `앞`, `좌`, `우` 거리와 `STRAIGHT`, `LEFT`, `RIGHT` 상태가 출력됩니다.

```bash
DRY_RUN=1 ./run_lidar_avoidance_rviz.sh
```

LiDAR 드라이버가 이미 실행 중이면 회피 코드만 실행할 수 있습니다.

```bash
cd ~/ros2_ws
source /opt/ros/jazzy/setup.bash
source install/setup.bash
./lidar_avoid.py --drive --left-us 1880 --right-us 1400
```

`lidar_avoid.py`를 단독 실행하면 저장소의 LiDAR RViz 설정으로 RViz도 자동
실행되며, 터미널에는 0.5초마다 전방·왼쪽·오른쪽 거리와 현재 주행 상태가
표시됩니다. RViz가 이미 실행 중이면 `--no-rviz`를 추가해 중복 실행을 막을 수
있습니다. RViz 표시와 거리 출력을 위해 LiDAR 드라이버의 `/scan` 토픽은 먼저
실행되어 있어야 합니다.

`lidar_avoid.py`는 시간 기반 S자 회피를 시험하는 별도 코드입니다. 기본 실행은
PWM을 출력하지 않고 판단 로그만 표시하며, 실제 구동은 검증된 조향값과 함께
`--drive`를 명시해야 합니다.

```bash
source /opt/ros/jazzy/setup.bash
source ~/ros2_ws/install/setup.bash
./lidar_avoid.py

# 바퀴를 띄우고 조향 방향과 중립값을 확인한 뒤에만 실행
./lidar_avoid.py --drive --left-us 1880 --right-us 1400
```

거리 판단은 바뀌는데 바퀴 조향이 보이지 않으면 차량을 들어 올린 상태에서 다음
명령으로 조향 장치만 확인합니다. ESC는 중립을 유지하며 왼쪽, 오른쪽, 중앙을
각 1초씩 출력합니다.

```bash
./lidar_avoid.py --drive --steering-test
```

터미널의 `PCA9685 조향 출력: CH9, ... us`는 코드가 실제 I2C 출력 함수를
호출했다는 뜻입니다. 로그는 바뀌는데 바퀴가 움직이지 않으면 채널 9 배선, 서보
전원 또는 MUX 연결을 확인해야 합니다.

## 핵심 동작

메인 실행 파일은 `run_lidar_stop_rviz.sh`입니다. 이 스크립트는 다음 프로그램을 함께 실행합니다.

1. Slamtec C1 LiDAR 드라이버
2. `/scan` 데이터를 표시하는 RViz
3. `lidar_stop_drive.py` 주행 및 정지 제어

`lidar_stop_drive.py`의 현재 동작은 다음과 같습니다.

- LiDAR 토픽: `/scan`
- 검사 방향: 차량 정면에 해당하는 LiDAR `180도` 기준 좌우 `30도`
- 주행 출력: PCA9685 채널 8에 `1565 us`
- 중립 출력: `1500 us`
- 조향 중앙: PCA9685 채널 9에 `1640 us`
- 주행 시간: 장애물 감지 또는 LiDAR 이상이 발생할 때까지 계속 주행
- 거리 정지 조건: 정면 최근접 거리가 `0.20 m 이상 0.30 m 이하`
- 장애물 정지 후 재출발: 최근접 거리가 `0.30 m 초과`가 되면 자동 주행 재개
- 센서 단절 정지: `/scan`이 `0.5초` 넘게 끊기면 중립 출력
- `/scan` 단절 또는 유효 거리값 오류로 정지한 경우에는 프로그램을 다시 실행해야 함

> 주의: 현재 요청된 설정에 따라 `0.20 m 미만`의 측정값은 새로운 거리 정지 조건에서 제외됩니다. 단, 장애물 때문에 이미 정지한 뒤 거리가 `0.20 m 미만`으로 가까워지면 정지를 유지하며 `0.30 m 초과`가 되어야 재출발합니다. 반드시 저속 및 안전한 시험 공간에서 사용해야 합니다.

## 사용 하드웨어와 환경

현재 코드는 아래 구성을 기준으로 작성되어 있습니다.

- NVIDIA Jetson 계열 보드
- ROS 2 Jazzy
- Slamtec C1 LiDAR (`/dev/ttyUSB0`, 460800 baud)
- PCA9685 PWM 컨트롤러 (`I2C bus 7`, 주소 `0x40`, 50 Hz)
- ESC 및 조향 서보
- RC/자동 입력 전환용 MUX
- Jetson 물리 핀 15: MUX 선택 PWM, 50 Hz, duty 10%
- USB 카메라: `/dev/video0`

실제 배선에 따라 PCA9685 채널과 PWM 값이 달라질 수 있습니다. 특히 `pca9685_drive_test.py`의 기본 채널은 조향 8, 스로틀 9이지만, LiDAR 주행 코드는 스로틀 채널 8과 `test2.py` 기준 조향 채널 9를 사용합니다. 실행 전에 실제 배선과 각 파일의 상수를 확인하세요.

## 저장소 구조

| 경로 | 역할 |
| --- | --- |
| `lidar_obstacle_avoidance.py` | 전방 장애물을 감지해 여유가 큰 좌우 통로로 저속 회피 |
| `lidar_avoidance_logic.py` | 하드웨어와 분리된 LiDAR 구간 분석 및 회피 상태 로직 |
| `run_lidar_avoidance_rviz.sh` | Slamtec C1, RViz, S자 장애물 회피 코드를 한 번에 실행 |
| `lidar_avoid.py` | 기본은 판단 로그만 출력하는 시간 기반 S자 회피 시험 코드 |
| `lidar_stop_drive.py` | `/scan`을 구독하면서 장애물에 정지하고 장애물이 사라지면 자동 재출발 |
| `run_lidar_stop_rviz.sh` | Slamtec C1, RViz, LiDAR 주행 코드를 한 번에 실행하고 함께 종료 |
| `lidar_distance_monitor.py` | 주행 없이 정면 최근접 장애물 거리를 1초마다 출력 |
| `run_lidar_distance_monitor.sh` | LiDAR와 거리 모니터만 한 번에 실행 |
| `pca9685_drive_test.py` | PCA9685 조향·속도 대화형 시험. `--dry-run` 지원 |
| `test3.py` | PCA9685 채널 8 ESC 속도를 단계적으로 높이는 시험 |
| `test3_straight.py` | 설정 속도로 계속 직진하며 `Ctrl+C`에서 중립 정지 |
| `test2.py` | PCA9685 조향을 키보드 명령으로 시험 |
| `steering.py`, `test.py` | Jetson sysfs PWM 기반 조향 반복 시험 |
| `straight.py`, `smooth_straight.py` | sysfs PWM 기반 직진 및 속도 시험 |
| `pwm_test.py` | sysfs PWM 조향 좌우 시험 |
| `sel_test.py` | MUX 선택 핀 HIGH/LOW 전환 시험 |
| `src/camera_node` | USB 카메라 영상을 `/camera/image_raw`로 발행하는 ROS 2 패키지 |
| `src/pwm_reader` | sysfs PWM duty를 읽어 `/pwm_duty`로 발행하는 ROS 2 패키지 |
| `src/sllidar_ros2` | Slamtec 공식 ROS 2 드라이버 Git submodule |
| `HARDWARE_CHANGELOG.txt` | 배선, PWM, MUX, LiDAR 설정 변경 기록 |

`build/`, `install/`, `log/`는 빌드 시 생성되므로 GitHub에 포함되지 않습니다.

## 내려받기

Slamtec 드라이버가 submodule로 연결되어 있으므로 다음과 같이 복제합니다.

```bash
git clone --recurse-submodules https://github.com/huis2-2/ros2_ws.git
cd ros2_ws
```

일반 `git clone`을 이미 사용했다면 submodule을 별도로 받습니다.

```bash
git submodule update --init --recursive
```

## 필요한 소프트웨어

- ROS 2 Jazzy 및 `colcon`
- ROS 패키지: `rclpy`, `sensor_msgs`, `std_msgs`, `cv_bridge`
- Python 패키지: `smbus2`, `Jetson.GPIO`, OpenCV
- Slamtec 드라이버가 요구하는 ROS 2 의존성

가능한 ROS 의존성은 작업공간 루트에서 다음 명령으로 설치할 수 있습니다.

```bash
source /opt/ros/jazzy/setup.bash
rosdep install --from-paths src --ignore-src -r -y
```

직렬 장치, I2C, GPIO 접근 권한도 필요합니다. 그룹을 변경한 뒤에는 로그아웃 후 다시 로그인하거나 재부팅해야 현재 세션에 반영됩니다.

```bash
sudo usermod -aG dialout,i2c,gpio "$USER"
```

## 빌드

```bash
cd ~/ros2_ws
source /opt/ros/jazzy/setup.bash
colcon build --symlink-install
source install/setup.bash
```

## LiDAR 장애물 정지 주행

차량 바퀴를 띄우거나 충분한 안전 공간을 확보한 후 실행합니다.

```bash
cd ~/ros2_ws
./run_lidar_stop_rviz.sh
```

LiDAR가 다른 직렬 장치로 연결되었다면 환경 변수로 지정할 수 있습니다.

```bash
LIDAR_PORT=/dev/ttyUSB1 ./run_lidar_stop_rviz.sh
```

종료는 실행한 터미널에서 `Ctrl+C`를 누릅니다. 종료 처리에서 ESC에 중립값 `1500 us`를 출력합니다.

### 개별 실행

LiDAR만 실행하려면 다음 명령을 사용합니다.

```bash
source /opt/ros/jazzy/setup.bash
source ~/ros2_ws/install/setup.bash
ros2 launch sllidar_ros2 sllidar_c1_launch.py serial_port:=/dev/ttyUSB0
```

다른 터미널에서 `/scan` 발행을 확인할 수 있습니다.

```bash
source /opt/ros/jazzy/setup.bash
ros2 topic hz /scan
ros2 topic echo /scan --once
```

주행 코드만 실행하려면 다음 명령을 사용합니다. 이 경우 LiDAR 드라이버가 이미 `/scan`을 발행하고 있어야 합니다.

```bash
cd ~/ros2_ws
source /opt/ros/jazzy/setup.bash
source install/setup.bash
python3 lidar_stop_drive.py
```

## 주행 없이 거리만 확인

다음 명령은 ESC와 조향 장치에 접근하지 않습니다. Slamtec C1 드라이버를 실행하고 차량 정면에 해당하는 LiDAR 180도 기준 좌우 30도 안의 최근접 장애물 거리를 1초마다 터미널에 출력합니다. 현재 장착 방향에서는 LiDAR 0도가 차량 뒤쪽입니다.

```bash
cd ~/ros2_ws
./run_lidar_distance_monitor.sh
```

종료하려면 `Ctrl+C`를 누릅니다. LiDAR 드라이버가 이미 실행 중이라면 모니터만 별도로 실행할 수도 있습니다.

```bash
cd ~/ros2_ws
source /opt/ros/jazzy/setup.bash
source install/setup.bash
python3 lidar_distance_monitor.py
```

## ROS 2 보조 노드

### 카메라

`/dev/video0`의 640x480, 30 FPS 영상을 `/camera/image_raw`로 발행하고 OpenCV 창에도 표시합니다.

```bash
ros2 run camera_node camera_node
```

### 카메라 흰 차선 주행

`camera_lane_follow.py`는 `/camera/image_raw`에서 흰색과 노란색 마스크를 각각
표시하고 흰 선만 검출해 따라갑니다. 기존
실차 보정값인 속도 PCA9685 CH8과 조향 CH9를 사용합니다. 차선이 보일 때만
3프레임 연속 감지를 확인한 뒤 기본 `1650 us`로 주행하며 차선 또는 카메라
영상을 잃으면 즉시 ESC 중립 `1500 us`, 조향 중앙 `1640 us`로 정지합니다.
차선 위치 목표는 BEV 영상의 중앙이며, 위치 오차와 차선 기울기를 함께 사용해
조향합니다.

- 흰색 기본 HSV: lower `[0, 0, 255]`, upper `[0, 106, 255]`
- 노란색 기본 HSV: lower `[19, 0, 197]`, upper `[64, 153, 255]`

먼저 한 터미널에서 카메라 노드를 실행합니다.

```bash
source /opt/ros/jazzy/setup.bash
source ~/ros2_ws/install/setup.bash
ros2 run camera_node camera_node
```

다른 터미널에서 실제 PWM을 출력하지 않는 인식 시험을 실행합니다. `--show`를
붙이면 BEV 차선 영상, 조향에 사용하는 `white mask`, 확인 전용 `yellow mask`를
각각 별도 창에서 볼 수 있습니다.

```bash
cd ~/ros2_ws
source /opt/ros/jazzy/setup.bash
python3 camera_lane_follow.py --show
```

인식 위치와 좌우 조향 방향을 확인한 후, 처음에는 반드시 바퀴를 띄우고 실제
차선 주행을 시험합니다.

```bash
python3 camera_lane_follow.py --drive --show
```

카메라 토픽과 속도는 각각 `--camera-topic`, `--speed-us`로 바꿀 수 있습니다.
단독 차선 주행 속도는 중립을 포함한 `1500~1650 us` 범위만 허용합니다.

### 카메라 차선 주행 + LiDAR 장애물 회피

`camera_lidar_follow.py`는 전방이 깨끗할 때 카메라의 흰 차선을 따라가고,
LiDAR가 장애물을 감지하면 좌우 중 여유가 큰 통로로 회피합니다. LiDAR 데이터가
끊기거나 장애물이 너무 가깝거나 좌우 통로가 모두 막히면 즉시 중립 정지합니다.
전방이 깨끗한 상태에서 카메라 영상 또는 흰 차선을 잃어도 정지합니다.
통합 실행 스크립트는 RViz에 `/scan`을 표시하며, 터미널에는 0.5초마다 LiDAR
중앙·좌·우 거리와 현재 LiDAR 상태를 출력합니다.
중앙 `0.70 m` 이하에서 회피를 시작하고, 회피 후 중앙 거리가 `0.75 m` 이상으로
확보되면 LiDAR 회피 조향을 해제하고 다시 카메라 흰 차선을 따라갑니다.

카메라와 LiDAR를 포함한 판단 로그 시험은 다음 명령 하나로 시작합니다. 이
명령은 실제 PWM을 출력하지 않습니다.

```bash
cd ~/ros2_ws
./run_camera_lidar_follow.sh --show
```

화면과 터미널에서 차선 방향 및 LiDAR 회피 방향을 확인한 다음, 처음에는 반드시
바퀴를 띄우고 실제 주행을 실행합니다.

```bash
./run_camera_lidar_follow.sh --drive --show
```

통합 노드는 기본 차선 속도와 회피 속도 모두 `1565 us`를 사용합니다.
각각 `--lane-speed-us`, `--avoid-speed-us`로 변경할 수 있으며 안전상
`1500~1565 us` 범위만 허용합니다. 종료는 `Ctrl+C`입니다.

### PWM 상태 읽기

세 sysfs PWM 경로의 duty 비율을 50 Hz로 읽어 `/pwm_duty`에 `[PWM1, PWM5, PWM7]` 순서로 발행합니다.

```bash
ros2 run pwm_reader pwm_reader_node
```

## 하드웨어 없이 제어 코드 확인

`pca9685_drive_test.py`는 `--dry-run`에서 I2C와 GPIO에 접근하지 않고 출력 예정 값을 보여줍니다.

```bash
python3 pca9685_drive_test.py --dry-run
```

명령 키는 다음과 같습니다.

- `a`, `d`, `c`: 왼쪽, 오른쪽, 중앙
- `j`, `l`: 조향 미세 조정
- `w`, `s`: 속도 증가 및 감소
- `x`: 즉시 속도 중립
- `p`: 현재 상태 표시
- `q`: 종료

## 문제 해결

### `/scan`이 없음

```bash
ls -l /dev/ttyUSB0
ros2 node list
ros2 topic list -t
```

LiDAR 장치는 보이지만 열 수 없다면 현재 사용자가 `dialout` 그룹에 반영됐는지 `id`로 확인합니다.

### I2C 장치를 열 수 없음

```bash
ls -l /dev/i2c-7
id
```

PCA9685 주소와 연결 상태는 시스템에 `i2cdetect`가 설치된 경우 다음과 같이 확인할 수 있습니다.

```bash
i2cdetect -y 7
```

### 프로그램은 실행되지만 차량이 움직이지 않음

- `/scan`이 실제로 발행되는지 확인합니다.
- 시작 시 ESC 중립 안정화 시간 3초를 기다립니다.
- MUX가 Jetson/PCA9685 입력을 선택했는지 확인합니다.
- 코드의 PCA9685 채널이 실제 ESC 배선과 같은지 확인합니다.
- ESC 전원, 중립값, 구동 PWM 값을 확인합니다.

## 안전 주의사항

- 처음 시험할 때는 구동 바퀴를 지면에서 띄우세요.
- 사람이 있는 장소나 도로에서 시험하지 마세요.
- ESC 중립값과 진행 방향을 확인한 뒤 속도를 올리세요.
- LiDAR만을 유일한 충돌 방지 장치로 사용하지 마세요.
- 프로그램 비정상 종료에 대비해 물리 전원 차단 수단을 준비하세요.

## 라이선스

이 저장소 자체에는 아직 통합 라이선스가 지정되어 있지 않습니다. `src/sllidar_ros2`에는 해당 upstream 프로젝트의 라이선스가 별도로 적용됩니다.
