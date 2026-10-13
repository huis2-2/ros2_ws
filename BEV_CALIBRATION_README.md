# BEV 캘리브레이션 (Homography)

ROS 2 카메라 토픽 `/camera/image_raw` 영상에서 바닥의 기준 직사각형 네 점을 클릭해
카메라 영상 → 차량 기준 바닥 좌표(m) Homography를 구하고 BEV 영상을 만듭니다.
이 도구는 카메라 토픽만 구독하며 **모터, ESC, PWM은 전혀 사용하지 않습니다.**

| 파일 | 역할 |
| --- | --- |
| `bev_calibrate.py` | OpenCV 창, 마우스 클릭, 캘리브레이션 계산·저장·불러오기 |
| `bev_transform.py` | Homography, BEV 변환, `pixel_to_vehicle()` / `vehicle_to_pixel()` |
| `bev_calibration.yaml` | 저장 결과. 처음에는 `calibrated: false` 빈 템플릿 |
| `test_bev_transform.py` | 가상 카메라로 수식을 확인하는 단위 테스트 |

## 좌표계

- 원점: 뒷바퀴 차축 중앙 바로 아래 바닥
- X: 차량 오른쪽 +, Y: 차량 전방 +, 단위 m
- BEV 영상: 위쪽 = 전방, 오른쪽 = 차량 오른쪽, 차량 중심선(X=0)은 세로선
- 축척 `pixels_per_meter = 1000` (10 px/cm). 기본 표시 범위 X −0.40~+0.40 m,
  Y −0.05~0.90 m → 출력 800×950 px, 원점 BEV 픽셀 (400, 900)

```
u = (x - x_min) * pixels_per_meter
v = (y_max - y) * pixels_per_meter
```

기준 직사각형 (가까운 기준선 거리 `Y_near`, 좌우 중심 `X_center`, 휴대폰 71.6×147.6 mm):

```
A = (X_center - 0.0358, Y_near)            차량 쪽 왼쪽
B = (X_center + 0.0358, Y_near)            차량 쪽 오른쪽
C = (X_center + 0.0358, Y_near + 0.1476)   먼 쪽 오른쪽
D = (X_center - 0.0358, Y_near + 0.1476)   먼 쪽 왼쪽
```

카메라 기울기(30~35°)는 사용하지 않습니다. 실제 기준점 네 개만으로 Homography를
계산하므로 장착 각도 오차가 결과에 들어가지 않습니다.

## 준비: 휴대폰 배치

1. 차량을 평평한 바닥에 두고, 뒷바퀴 차축 중심 바로 아래 바닥에 표시를 합니다.
2. 휴대폰 긴 변(147.6 mm)이 차량 진행 방향과 평행하도록 카메라 앞 바닥에 놓습니다.
   카메라 영상 아래쪽 1/3~중앙 부근, 가능하면 차량 중심선 위가 좋습니다.
3. **권장:** 휴대폰 네 모서리를 바닥에 펜·테이프로 표시한 다음 휴대폰을 치우고
   바닥 표시점을 클릭합니다.
4. 줄자로 측정합니다.
   - `near Y`: 뒷바퀴 차축 중심 → 가까운 쪽 변(A-B) 까지의 전방 거리 (mm)
   - `center X`: 차량 중심선 → 사각형 중심의 좌우 거리 (오른쪽 +, 기본 0 mm)

### 휴대폰 윗면 모서리를 직접 클릭하는 경우

휴대폰 윗면은 바닥보다 7.8 mm 높습니다. 카메라(높이 116 mm)에서 보면 높은 점은
같은 시선의 더 먼 바닥점처럼 보이므로 116 / (116 − 7.8) ≈ 1.072배, 즉 카메라로부터의
수평거리가 약 7 % 멀리 계산됩니다. 예를 들어 카메라 앞 30 cm에 있는 모서리는 약
2 cm 멀리 잡혀 전체 축척이 틀어집니다.

그래서 바닥 표시점을 권장합니다. 부득이하게 윗면을 클릭했다면
`--reference-height-mm 7.8`을 주면 카메라 위치(전방 320 mm, 높이 116 mm)를
이용해 같은 시선의 바닥점으로 보정합니다. 이 보정은 카메라 위치 실측값에
의존하므로 바닥 표시점보다 정확도가 떨어집니다.

## 실행

카메라 노드를 먼저 실행합니다(이미 실행 중이면 생략).

```bash
cd ~/ros2_ws
source /opt/ros/jazzy/setup.bash
source install/setup.bash
ros2 run camera_node camera_node
```

다른 터미널에서:

```bash
cd ~/ros2_ws
source /opt/ros/jazzy/setup.bash
python3 bev_calibrate.py --near-y-mm 450 --center-x-mm 0
```

SSH로 접속했다면 OpenCV 창을 띄울 수 있도록 Jetson 모니터의 데스크톱
터미널에서 실행하거나 `DISPLAY=:0`을 지정합니다.

주요 옵션:

| 옵션 | 기본값 | 설명 |
| --- | --- | --- |
| `--near-y-mm` | 500 | 차축 중심 → A-B 변 전방 거리 |
| `--center-x-mm` | 0 | 사각형 중심 좌우 위치 (오른쪽 +) |
| `--reference-height-mm` | 0 | 클릭한 점의 높이 (휴대폰 윗면이면 7.8) |
| `--rect-width-mm`, `--rect-length-mm` | 71.6, 147.6 | 다른 기준판을 쓸 때 |
| `--px-per-cm` | 10 | BEV 축척 |
| `--x-range-m`, `--y-range-m` | `-0.4,0.4`, `-0.05,0.9` | BEV 표시 범위 |
| `--check-point X_MM,Y_MM` | 없음 | 줄자로 잰 검증 지점 표시 (여러 번 가능) |
| `--calibration` | `bev_calibration.yaml` | 저장·불러오기 파일 |
| `--no-autoload` | | 시작할 때 저장값을 불러오지 않음 |
| `--undistort FILE` | | 선택: 렌즈 왜곡 보정 파일 |
| `--topic` | `/camera/image_raw` | 카메라 토픽 |
| `--config FILE` | | 위 옵션을 YAML로 지정 |

설정 파일 예 (`bev_settings.yaml`, 키는 옵션 이름과 같음):

```yaml
near_y_mm: 452
center_x_mm: -3
reference_height_mm: 0
check_point: [[0, 800], [200, 700]]
```

```bash
python3 bev_calibrate.py --config bev_settings.yaml
```

명령줄 옵션이 설정 파일 값보다 우선합니다.

## 창과 조작

**Camera Calibration** (원본 영상)

- 클릭 순서: **1:A 차량 쪽 왼쪽 → 2:B 차량 쪽 오른쪽 → 3:C 먼 쪽 오른쪽 → 4:D 먼 쪽 왼쪽**
  (영상에서 가까운 쪽이 아래입니다. 영상 기준 왼쪽 아래 → 오른쪽 아래 → 오른쪽 위 → 왼쪽 위)
- 점마다 번호·문자·색(빨강 A, 주황 B, 초록 C, 파랑 D)과 연결선이 표시됩니다.
- 커서 주변 4배 확대창이 나타나 모서리를 정확히 클릭할 수 있습니다.
- 위쪽 슬라이더 `near Y mm`, `center X mm +500`으로 실측 거리를 실시간 수정합니다.
  `center X` 슬라이더 값에서 500을 뺀 값이 실제 값입니다 (500 = 0 mm).
- 네 점 이후 추가 클릭은 **검증점**: 차량 좌표(cm)와 직전 검증점까지 거리가 표시됩니다.
- 보정이 끝나면 10 cm 바닥 격자(청록)와 차량 중심선(빨강)이 원본 영상에 겹쳐집니다.

**BEV Preview**

- 10 cm 격자와 거리 라벨, 차량 중심선(빨강), 원점(흰 점)과 +X(빨강)·+Y(초록) 축
- 앞바퀴 차축(257 mm), 카메라 위치(320 mm) 표시선, 기준 사각형(초록)
- 클릭하면 그 지점의 차량 좌표가 표시됩니다.
- 창 크기는 마우스로 조절할 수 있습니다.

| 키 | 동작 |
| --- | --- |
| `r` | 기준점·검증점 초기화 |
| `u` | 마지막 검증점 또는 기준점 취소 |
| `c` | 검증점 지우기 |
| `s` | 캘리브레이션 저장 |
| `l` | 저장된 캘리브레이션 불러오기 |
| `q` / `ESC` | 종료 |

## 저장과 불러오기

`s`를 누르면 `bev_calibration.yaml`에 다음을 저장합니다: 원본 해상도, A~D 픽셀 좌표,
대응 바닥 좌표, `homography_image_to_ground`(픽셀→m),
`homography_image_to_bev`(픽셀→BEV 픽셀), BEV 출력 크기, `pixels_per_meter`,
원점 BEV 픽셀, 점별 재투영 오차(mm, px), 날짜, 입력 거리, 렌즈 보정 설정.

다음 경우에는 저장하지 않고 터미널에 원인을 출력합니다.

- 네 점이 모두 없거나 영상 범위 밖
- 순서가 틀림(예: A, B가 바뀜, 먼 쪽을 먼저 클릭), 오목·꼬인 사각형, 너무 작은 사각형
- Homography가 특이 행렬이거나 지평선이 기준점 사이를 지남

다음 실행 시 파일이 있으면 자동으로 불러옵니다. 실행 중 `l`로 다시 불러올 수 있습니다.
현재 카메라 해상도가 저장된 해상도와 다르면 경고를 출력하고 **사용하지 않습니다.**

## 보정이 맞는지 확인하는 방법

4점 Homography는 같은 4점을 정확히 지나도록 만들어지므로 그 4점의 재투영 오차는
항상 거의 0입니다. 이 값만으로 정확도를 판단하지 마세요. 다음으로 확인합니다.

1. **BEV 모양 확인**: 바닥의 직선 테이프, 차선, 타일 줄눈이 BEV에서 곧고, 평행한 선은
   평행하게, 직각은 직각으로 보여야 합니다. 차량과 평행한 테이프는 세로선이어야 합니다.
2. **줄자 검증점**: 휴대폰과 떨어진 곳(예: 중심선 위 80 cm, 오른쪽 20 cm·전방 70 cm)에
   표시를 하고 실측합니다. `--check-point 0,800 --check-point 200,700`으로 실행하면
   두 창에 노란 `+`가 표시됩니다. 바닥 표시와 노란 `+`가 겹쳐야 합니다.
   또는 바닥 표시를 Camera 창에서 클릭하면 계산된 좌표가 나오므로 실측값과 비교합니다.
   10 cm 격자 기준으로 근거리 ±5 mm, 80 cm 부근 ±1~2 cm 정도면 양호합니다.
3. **알려진 길이**: 길이를 아는 물체(자, A4 용지 297 mm 등)의 양 끝을 연속 클릭하면
   두 검증점 사이 거리가 표시됩니다.
4. **반복성**: `r`로 초기화하고 다시 클릭해 같은 검증점 좌표가 1 cm 이내로 나오는지 봅니다.
5. 오차가 멀리 갈수록 커지면 `near Y` 측정, 클릭 위치(특히 C·D), 휴대폰 윗면 사용 여부를
   확인하세요. 더 큰 기준 사각형(예: 바닥 테이프로 만든 30×50 cm)을 쓰면
   `--rect-width-mm 300 --rect-length-mm 500`으로 정확도를 높일 수 있습니다.

## 렌즈 왜곡 보정 (선택)

광각 렌즈로 직선이 휘어 보이면 먼저 `camera_calibration` 패키지 등으로 카메라
내부 파라미터를 구한 뒤 `--undistort ~/.ros/camera_info/camera.yaml`로 실행합니다.
ROS `camera_info` YAML(`camera_matrix.data`, `distortion_coefficients.data`)과
OpenCV `FileStorage` YAML(`camera_matrix`, `dist_coeffs`)을 지원합니다.
설정은 `bev_calibration.yaml`에 함께 저장되고, 불러올 때 자동 적용됩니다.
왜곡 보정을 켜고 끄면 반드시 다시 캘리브레이션하세요.

## 차선 주행 코드에서 사용하기

```python
from bev_transform import CalibrationError, load_calibration

calibration = load_calibration('bev_calibration.yaml')  # calibrated: false면 예외

def image_callback(image):
    try:
        bev = calibration.warp(image)   # 해상도가 다르면 CalibrationError
    except CalibrationError as exc:
        print(exc)
        return
    # HSV + Sliding Window를 bev에서 수행해 차선 픽셀 (u, v)를 얻은 뒤
    x, y = calibration.pixel_to_vehicle(u, v)          # m, 뒷바퀴 차축 기준
    points_m = calibration.pixels_to_vehicle(lane_pixels)  # (N, 2) 배열 변환
    u, v = calibration.vehicle_to_pixel(0.0, 0.5)      # 디버그 표시용
```

- Pure Pursuit의 기준점이 뒷바퀴 차축 중앙이므로 반환값을 그대로 사용할 수 있습니다.
  look-ahead 점 (x, y), `L_d = hypot(x, y)`, `alpha = atan2(x, y)`,
  조향각 `delta = atan(2 * 0.257 * sin(alpha) / L_d)` (오른쪽 +).
  좌 25°, 우 30° 제한은 주행 코드에서 적용합니다.
- 원본 영상에서 차선을 찾는다면 `calibration.image_to_vehicle(u, v)`로 바로 변환합니다.
- BEV 범위나 축척을 바꾸려면 `calibration.set_config(BevConfig(...))`를 사용합니다.
  바닥 Homography는 그대로 유지됩니다.

## 테스트

카메라 없이 수식만 확인합니다.

```bash
cd ~/ros2_ws
python3 -m unittest test_bev_transform -v
```
