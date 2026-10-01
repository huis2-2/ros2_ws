import time
import Jetson.GPIO as GPIO
from smbus2 import SMBus

# =========================
# 설정
# =========================

I2C_BUS = 7
PCA_ADDR = 0x40
PCA_FREQ = 50

# ESC 채널
THROTTLE_CH = 8

# MUX SEL
MUX_PIN = 15
MUX_FREQ = 50
MUX_DUTY = 10.0

# ESC PWM 값 (us)
NEUTRAL = 1500
START_SPEED = 1565
MAX_SPEED = 1750

STEP = 3
STEP_TIME = 0.3


# =========================
# PCA9685
# =========================

MODE1 = 0x00
PRESCALE = 0xFE
LED0_ON_L = 0x06

bus = SMBus(I2C_BUS)


def init_pca9685(freq=50):
    prescale = int(round(25000000.0 / (4096.0 * freq) - 1.0))

    old_mode = bus.read_byte_data(PCA_ADDR, MODE1)

    bus.write_byte_data(
        PCA_ADDR,
        MODE1,
        (old_mode & 0x7F) | 0x10
    )

    bus.write_byte_data(PCA_ADDR, PRESCALE, prescale)
    bus.write_byte_data(PCA_ADDR, MODE1, old_mode)

    time.sleep(0.005)

    bus.write_byte_data(
        PCA_ADDR,
        MODE1,
        old_mode | 0xA1
    )

    time.sleep(0.01)


def set_pwm_us(channel, pulse_us):
    ticks = int(
        pulse_us * PCA_FREQ * 4096 / 1_000_000
    )

    ticks = max(0, min(4095, ticks))

    reg = LED0_ON_L + (4 * channel)

    bus.write_byte_data(PCA_ADDR, reg, 0)
    bus.write_byte_data(PCA_ADDR, reg + 1, 0)

    bus.write_byte_data(
        PCA_ADDR,
        reg + 2,
        ticks & 0xFF
    )

    bus.write_byte_data(
        PCA_ADDR,
        reg + 3,
        (ticks >> 8) & 0x0F
    )


# =========================
# 실행
# =========================

try:
    GPIO.setmode(GPIO.BOARD)
    GPIO.setup(MUX_PIN, GPIO.OUT)

    mux_pwm = GPIO.PWM(
        MUX_PIN,
        MUX_FREQ
    )

    mux_pwm.start(MUX_DUTY)

    init_pca9685(PCA_FREQ)

    # ESC 중립
    print("ESC 중립")
    set_pwm_us(THROTTLE_CH, NEUTRAL)

    # ESC 초기화 기다리기
    time.sleep(3)

    print("속도 증가 시작")

    speed = START_SPEED

    while speed <= MAX_SPEED:

        print(f"{speed} us")

        set_pwm_us(
            THROTTLE_CH,
            speed
        )

        time.sleep(STEP_TIME)

        speed += STEP

    print("최대 테스트 속도 도달")
    print("3초 후 정지")

    time.sleep(3)

    set_pwm_us(
        THROTTLE_CH,
        NEUTRAL
    )

    print("정지")

finally:
    try:
        set_pwm_us(
            THROTTLE_CH,
            NEUTRAL
        )
        time.sleep(0.5)
    except:
        pass

    try:
        mux_pwm.stop()
    except:
        pass

    GPIO.cleanup()
    bus.close()