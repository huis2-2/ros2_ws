import time
import Jetson.GPIO as GPIO
from smbus2 import SMBus

# =========================
# 설정
# =========================

I2C_BUS = 7
PCA_ADDR = 0x40
PCA_FREQ = 50

STEERING_CH = 9

MUX_PIN = 15
MUX_FREQ = 50
MUX_DUTY = 10.0   # 50Hz에서 2ms

# 조향값
STEER_RIGHT = 1400
STEER_CENTER = 1640
STEER_LEFT = 1880

# =========================
# PCA9685 레지스터
# =========================

MODE1 = 0x00
PRESCALE = 0xFE
LED0_ON_L = 0x06

bus = SMBus(I2C_BUS)

def init_pca9685(freq=50):
    prescale_val = 25000000.0 / (4096.0 * freq) - 1.0
    prescale = int(round(prescale_val))

    old_mode = bus.read_byte_data(PCA_ADDR, MODE1)

    sleep_mode = (old_mode & 0x7F) | 0x10
    bus.write_byte_data(PCA_ADDR, MODE1, sleep_mode)

    bus.write_byte_data(PCA_ADDR, PRESCALE, prescale)

    bus.write_byte_data(PCA_ADDR, MODE1, old_mode)
    time.sleep(0.005)

    bus.write_byte_data(PCA_ADDR, MODE1, old_mode | 0xA1)
    time.sleep(0.01)

def set_pwm_us(channel, pulse_us):
    ticks = int(pulse_us * PCA_FREQ * 4096 / 1_000_000)

    ticks = max(0, min(4095, ticks))

    reg = LED0_ON_L + 4 * channel

    bus.write_byte_data(PCA_ADDR, reg, 0)
    bus.write_byte_data(PCA_ADDR, reg + 1, 0)

    bus.write_byte_data(PCA_ADDR, reg + 2, ticks & 0xFF)
    bus.write_byte_data(PCA_ADDR, reg + 3, (ticks >> 8) & 0x0F)

# =========================
# 부드러운 조향
# =========================

current_steering = STEER_CENTER

def smooth_steer(target):
    global current_steering

    step = 5

    if target > current_steering:
        direction = step
    else:
        direction = -step

    value = current_steering

    while True:
        if direction > 0 and value >= target:
            break

        if direction < 0 and value <= target:
            break

        value += direction
        set_pwm_us(STEERING_CH, value)
        time.sleep(0.02)

    set_pwm_us(STEERING_CH, target)
    current_steering = target

# =========================
# 실행
# =========================

try:
    # MUX SEL PWM
    GPIO.setmode(GPIO.BOARD)
    GPIO.setup(MUX_PIN, GPIO.OUT)

    mux_pwm = GPIO.PWM(MUX_PIN, MUX_FREQ)
    mux_pwm.start(MUX_DUTY)

    # PCA9685 초기화
    init_pca9685(PCA_FREQ)

    # 시작 시 중앙
    set_pwm_us(STEERING_CH, STEER_CENTER)

    print("조향 테스트")
    print("a : 왼쪽")
    print("d : 오른쪽")
    print("c : 중앙")
    print("q : 종료")

    while True:
        cmd = input("명령 > ").strip().lower()

        if cmd == "a":
            smooth_steer(STEER_LEFT)

        elif cmd == "d":
            smooth_steer(STEER_RIGHT)

        elif cmd == "c":
            smooth_steer(STEER_CENTER)

        elif cmd == "q":
            break

finally:
    try:
        set_pwm_us(STEERING_CH, STEER_CENTER)
        time.sleep(0.5)
    except:
        pass

    try:
        mux_pwm.stop()
    except:
        pass

    GPIO.cleanup()
    bus.close()