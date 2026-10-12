#!/usr/bin/env python3
"""Find the steering pulse limits with the wheels raised.

Uses the same PCA9685 channels as ujin_versin.py. Throttle stays at neutral
the whole time. Keys:
  a / d   : steering +10 us (left) / -10 us (right)
  A / D   : steering +2 us / -2 us for fine adjustment
  c       : back to center
  q       : quit (steering centered, throttle neutral)
"""

import sys
import termios
import tty

from ujin_versin import (
    NEUTRAL_US,
    STEERING_CENTER_US,
    STEERING_LEFT_US,
    STEERING_RIGHT_US,
    DriveHardware,
)

# Hard bounds so a held key cannot drive the servo far past its stop.
ABSOLUTE_MIN_US = 1100
ABSOLUTE_MAX_US = 2200

STEPS = {'a': 10, 'd': -10, 'A': 2, 'D': -2}


def read_key():
    """Read one key press without waiting for Enter."""
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        return sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


def main():
    print(__doc__)
    print(f'현재 코드 값: 왼쪽 {STEERING_LEFT_US} / 중앙 {STEERING_CENTER_US} '
          f'/ 오른쪽 {STEERING_RIGHT_US} us')
    print('바퀴를 띄운 상태인지 확인하세요. 서보가 끝에 닿아 소리가 나면 '
          '바로 반대쪽으로 2~3번 돌리세요.\n')
    hardware = DriveHardware()
    steering = STEERING_CENTER_US
    try:
        hardware.set_speed(NEUTRAL_US)
        hardware.set_steering(steering)
        while True:
            print(f'\r조향 = {steering} us   ', end='', flush=True)
            key = read_key()
            if key in ('q', '\x03'):
                break
            if key == 'c':
                steering = STEERING_CENTER_US
            elif key in STEPS:
                steering = min(ABSOLUTE_MAX_US,
                               max(ABSOLUTE_MIN_US, steering + STEPS[key]))
            else:
                continue
            hardware.set_steering(steering)
    finally:
        print(f'\n마지막 조향 값: {steering} us. 중앙으로 복귀합니다.')
        hardware.close()


if __name__ == '__main__':
    main()
