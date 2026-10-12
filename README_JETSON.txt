Jetson Orin Nano DL lane follower V3
====================================

현재 장치의 소스 빌드 준비 (JetPack 7.2.1 / CUDA 13.2 / SM87)
---------------------------------------------------------
PyPI onnxruntime-gpu 1.31.0은 이 Orin에서 실제 추론 시
cudaErrorNoKernelImageForDevice가 발생했다. Provider 목록만으로 GPU 호환성을
판단하지 않는다. ONNX Runtime v1.23.0 소스는 다음 경로에 준비했다.
  /home/tofu/.cache/onnxruntime-orin-v1.23.0
실행 환경:
  source /home/tofu/.venvs/lane-onnx/bin/activate
빌드/설치/GPU 모델 시험:
  cd /home/tofu/ros2_ws
  bash build_onnxruntime_orin.sh
RAM+가용 스왑이 12 GB 미만이면 스크립트가 중단된다.
16 GB 스왑을 활성화한 후 실행한다. 빌드는 수 시간이
걸릴 수 있으며, 로그는 소스 폴더의 build-orin.log에 저장된다.
실제 차량 주행 성공 여부는 GPU 모델 시험과 별도로 현장에서 검증해야 한다.
이 빌드는 SM87 전용이며 CUDA NHWC, 전통 ML, Flash/Memory-efficient Attention
최적화 커널을 제외한다. lane_model.onnx의 연산은 유지한다.

GPU 시험 성공 후 간단 실행 (카메라 노드는 별도로 실행):
  bash run_dl_lane_follow.sh
실제 구동 옵션:
  bash run_dl_lane_follow.sh --drive --allow-drive --speed-us 1600
위 실행 스크립트는 ROS와 ~/.venvs/lane-onnx 환경을 자동으로 불러온다.

필요한 파일
  camera_lane_follow_dl_jetson_v3.py
  lane_model.onnx

모델 형식
  입력: RGB, float32 / 255, NCHW 320x256
  출력 클래스: 0 배경, 1 흰 선, 2 노란 선
  실제 조향은 흰 선만 사용한다.

필수 환경
  ROS 2 Jazzy: rclpy, cv_bridge, sensor_msgs
  numpy, OpenCV
  JetPack 버전에 맞는 GPU용 ONNX Runtime
  실제 구동 시에만 smbus2, Jetson.GPIO

1. GPU provider 확인
--------------------

  cd ~/ros2_ws
  python3 -c "import onnxruntime as ort; print(ort.__version__); print(ort.get_available_providers())"

TensorrtExecutionProvider가 있으면 --backend tensorrt를 사용한다.
TensorRT가 없고 CUDAExecutionProvider만 있으면 --backend cuda를 사용한다.
CPUExecutionProvider만 보이면 GPU용 ONNX Runtime 설치가 먼저 필요하다.
--backend cpu는 영상 진단용이며 실제 주행에는 권장하지 않는다.

2. 카메라 노드 실행 (터미널 1)
-----------------------------

  cd ~/ros2_ws
  source /opt/ros/jazzy/setup.bash
  source install/setup.bash
  ros2 run camera_node camera_node

이미 /camera/image_raw가 발행되고 있으면 이 단계는 생략한다. 확인 명령:

  ros2 topic hz /camera/image_raw

3. TensorRT 화면 시험 (터미널 2)
--------------------------------

바퀴를 구동하지 않는 시험이다.

  cd ~/ros2_ws
  source /opt/ros/jazzy/setup.bash
  source install/setup.bash
  python3 -u camera_lane_follow_dl_jetson_v3.py \
    --model ./lane_model.onnx \
    --backend tensorrt \
    --show \
    --target-x 0.50 \
    --crop-top 0.0 \
    --view-scale 0.85 \
    --debug-every 2

첫 TensorRT 실행은 엔진을 만드는 동안 오래 걸릴 수 있다. 생성된 엔진과 타이밍
캐시는 ~/.cache/ros2_lane_follow/trt 에 저장되며 다음 실행부터 재사용된다.
시작 로그에서 backend=tensorrt와 활성 provider, 입력 크기, warmup 시간을 확인한다.

CUDA만 사용 가능한 경우 위 명령의 한 부분만 바꾼다.

  --backend cuda

설치된 GPU provider 중 가장 빠른 것을 자동 선택하려면 다음을 사용할 수 있다.

  --backend auto

4. 실제 차량 구동
-----------------

먼저 바퀴를 띄운 상태에서 흰 선 방향과 조향 방향을 확인한다.

  cd ~/ros2_ws
  source /opt/ros/jazzy/setup.bash
  source install/setup.bash
  python3 -u camera_lane_follow_dl_jetson_v3.py \
    --model ./lane_model.onnx \
    --backend tensorrt \
    --show \
    --drive \
    --allow-drive \
    --target-x 0.50 \
    --crop-top 0.0 \
    --speed-us 1600 \
    --view-scale 0.85 \
    --debug-every 2

종료는 실행한 터미널에서 Ctrl+C를 누른다. 종료 시 스로틀 1500 us와 중앙 조향
1640 us를 출력한다.

5. 화면과 로그 읽기
-------------------

  1 Camera        원본 카메라 영상
  2 White Mask    모델이 예측한 흰 선. 실제 조향 입력
  3 Yellow Mask   모델이 예측한 노란 선. 확인용
  4 Lane Decision 선택된 흰 선, 슬라이딩 윈도우, 목표점과 PWM

TRACKING이 유지되고 FRAME AGE가 150 ms 이하인지 확인한다. 자주 멈추면 터미널의
STOP_REASON을 먼저 확인한다. 주요 값은 LANE_LOST, FRAME_STALE_AFTER_INFER,
COMMAND_TIMEOUT, CAMERA_TIMEOUT, INFERENCE_ERROR다.

주요 옵션
---------

  --target-x 0.50
      시작할 때 선택할 흰 선의 가로 위치 비율. 다른 흰 선으로 자동 점프하지 않는다.

  --crop-top 0.0
      원본 전체 구도를 모델에 입력한다. 학습 영상이 위 10%를 제거한 구도였다면
      0.10으로 시험한다.

  --debug-every 2
      두 프레임마다 화면을 갱신한다. 추론이 느리면 3 또는 4로 높인다.

  --max-frame-age-ms 150
      수신 후 150 ms가 지난 프레임으로는 구동하지 않는다.

  --watchdog-sec 0.35
      처리나 카메라 입력이 멈추면 중립으로 복귀하는 시간이다.

  --trt-cache ~/.cache/ros2_lane_follow/trt
      TensorRT 엔진과 타이밍 캐시 위치다.

  --no-trt-fp16
      FP16에서 모델 결과에 문제가 있을 때만 FP32로 시험한다.

  --speed-us 1600
      이 DL 코드에서 허용한 최대값이다.

Jetson 성능 확인
---------------

다른 터미널에서 사용률과 온도를 확인할 수 있다.

  sudo tegrastats

전원 모드를 확인하고 시험 중 클럭 변동을 줄이려면 다음을 사용할 수 있다.

  sudo nvpmodel -q
  sudo jetson_clocks

검증 범위
---------

코드 문법, 차선 추적 수학 함수, 명령행 도움말은 작업공간에서 확인했다. 현재
작업공간에는 onnxruntime이 설치되어 있지 않아 TensorRT/CUDA 실추론과 Jetson GPIO
구동은 여기서 실행하지 못했다. 실제 구동 전 화면 시험과 바퀴를 띄운 시험이 필요하다.
