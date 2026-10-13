#!/usr/bin/env python3
import math
import signal
import time

from smbus2 import SMBus
import Jetson.GPIO as GPIO

# 기존 차량 설정
I2C_BUS = 7
PCA_ADDR = 0x40
FREQ = 50

THROTTLE_CH = 8       # 속도
STEERING_CH = 9       # 조향
MUX_PIN = 15          # Jetson BOARD 핀 번호

NEUTRAL_US = 1500
SPEED_US = 1565
CENTER_US = 1640

AMPLITUDE_US = 80     # 조향 폭
PERIOD_SEC = 4.0      # 좌우 한 번 왕복하는 시간


def set_pwm_us(bus, channel, pulse_us):
    ticks = int(pulse_us * FREQ * 4096 / 1_000_000)
    reg = 0x06 + 4 * channel

    values = (0, 0, ticks & 0xFF, (ticks >> 8) & 0x0F)
    for offset, value in enumerate(values):
        bus.write_byte_data(PCA_ADDR, reg + offset, value)


def init_pca(bus):
    old = bus.read_byte_data(PCA_ADDR, 0x00)
    awake = old & ~0x10

    bus.write_byte_data(PCA_ADDR, 0x00, (awake & 0x7F) | 0x10)
    prescale = round(25_000_000 / (4096 * FREQ) - 1)
    bus.write_byte_data(PCA_ADDR, 0xFE, prescale)
    bus.write_byte_data(PCA_ADDR, 0x00, awake)
    time.sleep(0.005)
    bus.write_byte_data(PCA_ADDR, 0x00, awake | 0xA1)
    time.sleep(0.01)


def interrupt(signum, frame):
    raise KeyboardInterrupt


def main():
    bus = None
    mux = None
    gpio_ready = False
    signal.signal(signal.SIGTERM, interrupt)

    try:
        bus = SMBus(I2C_BUS)
        init_pca(bus)

        # 중립 및 조향 중앙
        set_pwm_us(bus, THROTTLE_CH, NEUTRAL_US)
        set_pwm_us(bus, STEERING_CH, CENTER_US)

        GPIO.setwarnings(False)
        GPIO.setmode(GPIO.BOARD)
        gpio_ready = True
        GPIO.setup(MUX_PIN, GPIO.OUT)

        mux = GPIO.PWM(MUX_PIN, FREQ)
        mux.start(10.0)

        print("조종기 전원을 켜두세요. ESC 중립 안정화 3초")
        time.sleep(3)

        print("지그재그 연속 주행 시작! Ctrl+C로 정지")
        start = time.monotonic()
        last_steering = None
        last_speed = None

        while True:
            elapsed = time.monotonic() - start

            # 중앙 → 왼쪽 → 중앙 → 오른쪽 → 중앙 반복
            steering = round(
                CENTER_US
                + AMPLITUDE_US
                * math.sin(2 * math.pi * elapsed / PERIOD_SEC)
            )

            if steering != last_steering:
                set_pwm_us(bus, STEERING_CH, steering)
                last_steering = steering

            # 첫 1초 동안 전진값까지 서서히 올린 뒤 유지
            speed = round(
                NEUTRAL_US
                + (SPEED_US - NEUTRAL_US) * min(elapsed, 1.0)
            )

            if speed != last_speed:
                set_pwm_us(bus, THROTTLE_CH, speed)
                last_speed = speed

            time.sleep(0.02)

    except KeyboardInterrupt:
        print("정지 요청")

    finally:
        actions = []

        if bus is not None:
            actions.extend([
                lambda: set_pwm_us(bus, THROTTLE_CH, NEUTRAL_US),
                lambda: set_pwm_us(bus, STEERING_CH, CENTER_US),
                lambda: time.sleep(0.2),
            ])

        if mux is not None:
            actions.append(mux.stop)
        if gpio_ready:
            actions.append(lambda: GPIO.cleanup(MUX_PIN))
        if bus is not None:
            actions.append(bus.close)

        for action in actions:
            try:
                action()
            except Exception as exc:
                print(f"종료 처리 실패: {exc}")

        print("종료")


if __name__ == "__main__":
    main()