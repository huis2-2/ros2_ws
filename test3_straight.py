#!/usr/bin/env python3
"""test3.py 설정으로 초기 속도를 유지하며 계속 직진한다."""

import time

import Jetson.GPIO as GPIO
from smbus2 import SMBus


# test3.py와 동일한 하드웨어 설정
I2C_BUS = 7
PCA_ADDR = 0x40
PCA_FREQ = 50

THROTTLE_CH = 8

MUX_PIN = 15
MUX_FREQ = 50
MUX_DUTY = 10.0

# test3.py의 중립값과 초기 속도
NEUTRAL = 1500
STRAIGHT_SPEED = 1600

MODE1 = 0x00
PRESCALE = 0xFE
LED0_ON_L = 0x06


def init_pca9685(bus, freq=PCA_FREQ):
    prescale = int(round(25_000_000.0 / (4096.0 * freq) - 1.0))
    old_mode = bus.read_byte_data(PCA_ADDR, MODE1)

    # SLEEP 상태에서 주파수를 설정한 뒤 다시 깨운다.
    awake_mode = old_mode & ~0x10
    bus.write_byte_data(PCA_ADDR, MODE1, (awake_mode & 0x7F) | 0x10)
    bus.write_byte_data(PCA_ADDR, PRESCALE, prescale)
    bus.write_byte_data(PCA_ADDR, MODE1, awake_mode)
    time.sleep(0.005)
    bus.write_byte_data(PCA_ADDR, MODE1, awake_mode | 0xA1)
    time.sleep(0.01)


def set_pwm_us(bus, channel, pulse_us):
    ticks = int(pulse_us * PCA_FREQ * 4096 / 1_000_000)
    ticks = max(0, min(4095, ticks))
    reg = LED0_ON_L + (4 * channel)

    bus.write_byte_data(PCA_ADDR, reg, 0)
    bus.write_byte_data(PCA_ADDR, reg + 1, 0)
    bus.write_byte_data(PCA_ADDR, reg + 2, ticks & 0xFF)
    bus.write_byte_data(PCA_ADDR, reg + 3, (ticks >> 8) & 0x0F)


def main():
    bus = None
    mux_pwm = None

    try:
        bus = SMBus(I2C_BUS)
        init_pca9685(bus)

        # MUX를 자동 주행 입력으로 전환하기 전에 ESC를 중립으로 둔다.
        set_pwm_us(bus, THROTTLE_CH, NEUTRAL)

        GPIO.setwarnings(False)
        GPIO.setmode(GPIO.BOARD)
        GPIO.setup(MUX_PIN, GPIO.OUT)
        mux_pwm = GPIO.PWM(MUX_PIN, MUX_FREQ)
        mux_pwm.start(MUX_DUTY)

        print("ESC 중립 신호 안정화 중 (3초)")
        time.sleep(3)

        set_pwm_us(bus, THROTTLE_CH, STRAIGHT_SPEED)
        print(f"직진 시작: {STRAIGHT_SPEED} us")
        print("속도를 유지합니다. 정지는 Ctrl+C")

        while True:
            time.sleep(1)

    except KeyboardInterrupt:
        print("\n정지 요청")
    finally:
        if bus is not None:
            try:
                set_pwm_us(bus, THROTTLE_CH, NEUTRAL)
                time.sleep(0.5)
                print("ESC 중립: 1500 us")
            except OSError as exc:
                print(f"ESC 중립 출력 실패: {exc}")

        if mux_pwm is not None:
            mux_pwm.stop()
        GPIO.cleanup()

        if bus is not None:
            bus.close()


if __name__ == "__main__":
    main()
