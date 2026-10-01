#!/usr/bin/env python3
"""PCA9685 CH8/CH9 drive test with MUX select on Jetson header pin 15.

The script changes only the autonomous inputs of the external MUX:
  PCA9685 CH8 -> MUX S1 -> steering
  PCA9685 CH9 -> MUX S2 -> throttle/ESC

Receiver ST/TH are connected directly to MUX M1/M2, so their pulse widths
cannot be measured by the Jetson with the wiring above. Jetson header pin 15
outputs the 50 Hz selection pulse for the MUX.
"""

import argparse
import sys
import termios
import time
import tty


PCA9685_MODE1 = 0x00
PCA9685_MODE2 = 0x01
PCA9685_LED0_ON_L = 0x06
PCA9685_PRESCALE = 0xFE


class PCA9685:
    """Minimal PCA9685 driver using python3-smbus or smbus2."""

    def __init__(self, bus_number, address, frequency_hz):
        try:
            from smbus2 import SMBus
        except ImportError:
            try:
                from smbus import SMBus
            except ImportError as exc:
                raise RuntimeError(
                    "smbus 모듈이 없습니다. "
                    "sudo apt install python3-smbus 로 설치하세요."
                ) from exc

        self.bus = SMBus(bus_number)
        self.address = address
        self.frequency_hz = frequency_hz
        self._set_frequency(frequency_hz)

    def _set_frequency(self, frequency_hz):
        if not 24 <= frequency_hz <= 1526:
            raise ValueError("PCA9685 주파수는 24~1526 Hz 범위여야 합니다.")

        prescale = round(25_000_000 / (4096 * frequency_hz)) - 1
        old_mode = self.bus.read_byte_data(self.address, PCA9685_MODE1)
        # A previous program or the board itself may leave SLEEP set. Always
        # derive the restored mode from an explicitly awake state.
        awake_mode = old_mode & ~0x10
        sleep_mode = (awake_mode & 0x7F) | 0x10

        self.bus.write_byte_data(self.address, PCA9685_MODE1, sleep_mode)
        self.bus.write_byte_data(self.address, PCA9685_PRESCALE, prescale)
        self.bus.write_byte_data(self.address, PCA9685_MODE1, awake_mode)
        time.sleep(0.005)
        # RESTART + auto increment. MODE2 OUTDRV gives a conventional totem-pole output.
        self.bus.write_byte_data(self.address, PCA9685_MODE1, awake_mode | 0xA1)
        mode2 = self.bus.read_byte_data(self.address, PCA9685_MODE2)
        self.bus.write_byte_data(self.address, PCA9685_MODE2, mode2 | 0x04)

    def set_pulse_us(self, channel, pulse_us):
        if not 0 <= channel <= 15:
            raise ValueError("PCA9685 채널은 0~15 범위여야 합니다.")
        if not 500 <= pulse_us <= 2500:
            raise ValueError("PWM 펄스폭은 안전상 500~2500 us 범위만 허용합니다.")

        ticks = round(pulse_us * self.frequency_hz * 4096 / 1_000_000)
        ticks = max(1, min(4095, ticks))
        register = PCA9685_LED0_ON_L + 4 * channel
        data = [0, 0, ticks & 0xFF, (ticks >> 8) & 0x0F]
        self.bus.write_i2c_block_data(self.address, register, data)

    def close(self):
        self.bus.close()


class DryRunPCA9685:
    """Hardware-free output used to check the program on another computer."""

    def set_pulse_us(self, channel, pulse_us):
        print(f"[DRY-RUN] PCA9685 CH{channel} <- {pulse_us} us")

    def close(self):
        pass


class MuxSelectOutput:
    """Drive the MUX select input from a Jetson physical header pin."""

    def __init__(self, board_pin, mode, frequency_hz, duty_percent):
        try:
            import Jetson.GPIO as GPIO
        except ImportError as exc:
            raise RuntimeError(
                "Jetson.GPIO 모듈이 없습니다. "
                "MUX 출력을 생략하려면 --no-mux를 사용하세요."
            ) from exc

        self.gpio = GPIO
        self.board_pin = board_pin

        GPIO.setwarnings(False)
        GPIO.setmode(GPIO.BOARD)
        GPIO.setup(board_pin, GPIO.OUT, initial=GPIO.LOW)
        self.pwm = None
        try:
            if mode == "high":
                GPIO.output(board_pin, GPIO.HIGH)
            elif mode == "low":
                GPIO.output(board_pin, GPIO.LOW)
            else:
                self.pwm = GPIO.PWM(board_pin, frequency_hz)
                self.pwm.start(duty_percent)
        except Exception:
            GPIO.cleanup(board_pin)
            raise

    def close(self):
        if self.pwm is not None:
            self.pwm.ChangeDutyCycle(0)
            time.sleep(0.03)
            self.pwm.stop()
        else:
            self.gpio.output(self.board_pin, self.gpio.LOW)
        self.gpio.cleanup(self.board_pin)


class DryRunMuxSelectOutput:
    def __init__(self, board_pin, mode, frequency_hz, duty_percent):
        if mode == "pwm":
            value = f"{frequency_hz} Hz, {duty_percent:.1f}%"
        else:
            value = mode.upper()
        print(f"[DRY-RUN] Jetson 물리 {board_pin} MUX SEL <- {value}")

    def close(self):
        pass


def clamp(value, low, high):
    return max(low, min(high, value))


def move_toward(value, target, step):
    if value < target:
        return min(value + step, target)
    return max(value - step, target)


def read_command():
    """Read one key immediately on a terminal, or one line from piped input."""
    if not sys.stdin.isatty():
        return input("> ").strip().lower()

    fd = sys.stdin.fileno()
    previous_settings = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        print("> ", end="", flush=True)
        command = sys.stdin.read(1).lower()
        print(command)
        return command
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, previous_settings)


def print_status(args, steering_us, throttle_us):
    if abs(steering_us - args.steering_center) <= 10:
        steering_text = "중앙"
    elif (
        (steering_us - args.steering_center)
        * (args.steering_left - args.steering_center)
        > 0
    ):
        steering_text = "왼쪽"
    else:
        steering_text = "오른쪽"

    speed_range = max(1, args.throttle_max - args.throttle_neutral)
    speed_command = 100 * (throttle_us - args.throttle_neutral) / speed_range
    speed_command = clamp(speed_command, 0, 100)

    if args.no_mux:
        mux_text = "OFF (--no-mux)"
    elif args.mux_mode != "pwm":
        mux_text = f"디지털 {args.mux_mode.upper()}"
    else:
        mux_pulse_us = 1_000_000 * (args.mux_duty / 100) / args.mux_frequency
        if mux_pulse_us >= 1700:
            selection = "Jetson/PCA 선택"
        elif mux_pulse_us <= 1300:
            selection = "RC 수신기 선택"
        else:
            selection = "선택 중간값"
        mux_text = (
            f"{args.mux_frequency} Hz / {mux_pulse_us:.0f} us "
            f"({selection})"
        )
    print(
        f"PCA CH{args.steering_channel} 조향: "
        f"{steering_us:4d} us ({steering_text}) | "
        f"PCA CH{args.throttle_channel} 속도: "
        f"{throttle_us:4d} us (명령 {speed_command:3.0f}%) | "
        f"Jetson 물리 {args.mux_pin} MUX SEL: "
        f"{mux_text}"
    )


def build_parser():
    parser = argparse.ArgumentParser(
        description="PCA9685 CH8/CH9 조향·속도 시험 및 Jetson 15번 MUX 선택"
    )
    parser.add_argument(
        "--i2c-bus",
        type=int,
        default=7,
        help="Jetson Orin Nano 40핀 헤더 3/5번의 기본 버스",
    )
    parser.add_argument("--address", type=lambda value: int(value, 0), default=0x40)
    parser.add_argument("--frequency", type=int, default=50)
    parser.add_argument("--mux-pin", type=int, default=15, help="물리 헤더 핀 번호")
    parser.add_argument(
        "--mux-mode",
        choices=("high", "low", "pwm"),
        default="pwm",
        help="SEL 구동 방식(기본: RC 방식 PWM)",
    )
    parser.add_argument("--mux-frequency", type=float, default=50.0)
    parser.add_argument(
        "--mux-duty",
        type=float,
        default=10.0,
        help="기본 10%% = 50 Hz에서 약 2000 us",
    )
    parser.add_argument(
        "--no-mux",
        "--no-aux",
        dest="no_mux",
        action="store_true",
        help="Jetson 15번 MUX 출력을 사용하지 않음",
    )
    parser.add_argument("--dry-run", action="store_true")

    parser.add_argument("--steering-channel", type=int, default=8)
    parser.add_argument("--throttle-channel", type=int, default=9)
    parser.add_argument("--steering-right", type=int, default=1600)
    parser.add_argument("--steering-center", type=int, default=1500)
    parser.add_argument("--steering-left", type=int, default=1400)
    parser.add_argument("--steering-step", type=int, default=25)
    parser.add_argument("--throttle-neutral", type=int, default=1500)
    parser.add_argument("--throttle-max", type=int, default=1600)
    parser.add_argument("--throttle-step", type=int, default=10)
    return parser


def validate_args(parser, args):
    if args.i2c_bus < 0:
        parser.error("i2c-bus는 0 이상의 번호여야 합니다.")
    if not 0x03 <= args.address <= 0x77:
        parser.error("I2C 주소는 0x03~0x77 범위여야 합니다.")
    if not 24 <= args.frequency <= 1526:
        parser.error("PCA9685 주파수는 24~1526 Hz 범위여야 합니다.")
    if args.mux_frequency <= 0:
        parser.error("MUX 주파수는 0보다 커야 합니다.")
    if not 0 <= args.mux_duty <= 100:
        parser.error("MUX duty는 0~100% 범위여야 합니다.")
    steering_low = min(args.steering_right, args.steering_left)
    steering_high = max(args.steering_right, args.steering_left)
    if not steering_low < args.steering_center < steering_high:
        parser.error("조향 center 값은 left와 right 값 사이여야 합니다.")
    if args.throttle_max <= args.throttle_neutral:
        parser.error("throttle-max는 throttle-neutral보다 커야 합니다.")
    if args.steering_step <= 0 or args.throttle_step <= 0:
        parser.error("조절 step 값은 0보다 커야 합니다.")
    for channel in (args.steering_channel, args.throttle_channel):
        if not 0 <= channel <= 15:
            parser.error("PCA9685 채널은 0~15 범위여야 합니다.")
    for pulse in (
        args.steering_right,
        args.steering_center,
        args.steering_left,
        args.throttle_neutral,
        args.throttle_max,
    ):
        if not 500 <= pulse <= 2500:
            parser.error("모든 PWM 값은 500~2500 us 범위여야 합니다.")


def main():
    parser = build_parser()
    args = parser.parse_args()
    validate_args(parser, args)

    pca = None
    mux_output = None
    stage = "PCA9685 I2C 초기화"
    steering_us = args.steering_center
    throttle_us = args.throttle_neutral

    try:
        pca = (
            DryRunPCA9685()
            if args.dry_run
            else PCA9685(args.i2c_bus, args.address, args.frequency)
        )
        # Set safe PCA9685 values before connecting them through the MUX.
        stage = "PCA9685 CH8/CH9 안전값 출력"
        pca.set_pulse_us(args.steering_channel, steering_us)
        pca.set_pulse_us(args.throttle_channel, throttle_us)
        if not args.no_mux:
            stage = f"Jetson 물리 {args.mux_pin}번 MUX SEL 출력"
            mux_class = DryRunMuxSelectOutput if args.dry_run else MuxSelectOutput
            mux_output = mux_class(
                args.mux_pin,
                args.mux_mode,
                args.mux_frequency,
                args.mux_duty,
            )
        stage = "대화형 조향/속도 제어"
        if not args.dry_run:
            print("ESC 중립 신호 안정화 중 (2초)...")
            time.sleep(2.0)

        print("\n명령: a=왼쪽 끝, d=오른쪽 끝, c=중앙")
        print("      j=왼쪽 미세조정, l=오른쪽 미세조정")
        print("      w=속도 증가, s=속도 감소, x=즉시 중립, p=상태, q=종료")
        print("      키는 Enter 없이 즉시 적용됩니다.")
        print_status(args, steering_us, throttle_us)

        while True:
            command = read_command()
            if command == "q":
                break
            if command == "a":
                steering_us = args.steering_left
            elif command == "d":
                steering_us = args.steering_right
            elif command == "c":
                steering_us = args.steering_center
            elif command == "j":
                steering_us = move_toward(
                    steering_us,
                    args.steering_left,
                    args.steering_step,
                )
            elif command == "l":
                steering_us = move_toward(
                    steering_us,
                    args.steering_right,
                    args.steering_step,
                )
            elif command == "w":
                throttle_us = clamp(
                    throttle_us + args.throttle_step,
                    args.throttle_neutral,
                    args.throttle_max,
                )
            elif command == "s":
                throttle_us = clamp(
                    throttle_us - args.throttle_step,
                    args.throttle_neutral,
                    args.throttle_max,
                )
            elif command == "x":
                throttle_us = args.throttle_neutral
            elif command != "p":
                print("알 수 없는 명령입니다.")
                continue

            pca.set_pulse_us(args.steering_channel, steering_us)
            pca.set_pulse_us(args.throttle_channel, throttle_us)
            print_status(args, steering_us, throttle_us)

    except (KeyboardInterrupt, EOFError):
        print("\n종료 요청")
    except (OSError, RuntimeError) as exc:
        print(f"하드웨어 오류 ({stage}): {exc}")
        return 1
    finally:
        if pca is not None:
            try:
                # Keep the vehicle stopped and steering centered on every exit path.
                pca.set_pulse_us(args.throttle_channel, args.throttle_neutral)
                pca.set_pulse_us(args.steering_channel, args.steering_center)
                if not args.dry_run:
                    time.sleep(0.2)
            except OSError as exc:
                print(f"종료 중 중립 출력 실패: {exc}")
            pca.close()
        if mux_output is not None:
            try:
                mux_output.close()
            except RuntimeError as exc:
                print(f"MUX SEL 정리 경고: {exc}")
        if pca is not None:
            print("속도 중립 / 조향 중앙으로 종료했습니다.")
        else:
            print("PCA9685 PWM 초기화 전에 종료했습니다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
