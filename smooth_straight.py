import Jetson.GPIO as GPIO
import time

GPIO.setwarnings(False)
GPIO.setmode(GPIO.BOARD)

# 15번 : MUX 자동모드 선택
SEL_PIN = 15

# 32번 : 조향
STEER_PWM = "/sys/class/pwm/pwmchip2/pwm0"

# 33번 : ESC
ESC_PWM = "/sys/class/pwm/pwmchip3/pwm0"

# 조향 중앙
CENTER = 1600000

# ESC
NEUTRAL = 1500000
START_SPEED = 1565000
MAX_TEST_SPEED = 1700000
STEP = 10000           # 한 번에 0.01ms씩 증가
STEP_TIME = 0.5        # 0.5초마다 증가


def write_pwm(path, name, value):
    with open(f"{path}/{name}", "w") as f:
        f.write(str(value))


# -------------------------
# MUX 자동모드
# -------------------------
GPIO.setup(SEL_PIN, GPIO.OUT)

sel_pwm = GPIO.PWM(SEL_PIN, 50)

# B = 자동 모드
sel_pwm.start(10.0)


# -------------------------
# 조향 : 중앙 고정
# -------------------------
try:
    write_pwm(STEER_PWM, "enable", 0)
except:
    pass

write_pwm(STEER_PWM, "period", 20000000)
write_pwm(STEER_PWM, "duty_cycle", CENTER)
write_pwm(STEER_PWM, "enable", 1)


# -------------------------
# ESC 초기화
# -------------------------
try:
    write_pwm(ESC_PWM, "enable", 0)
except:
    pass

write_pwm(ESC_PWM, "period", 20000000)
write_pwm(ESC_PWM, "duty_cycle", NEUTRAL)
write_pwm(ESC_PWM, "enable", 1)

print("ESC 중립")
time.sleep(2)


try:
    speed = START_SPEED

    while speed <= MAX_TEST_SPEED:

        # 항상 직진
        write_pwm(STEER_PWM, "duty_cycle", CENTER)

        # 속도 증가
        write_pwm(ESC_PWM, "duty_cycle", speed)

        print("현재 속도 PWM:", speed)

        time.sleep(STEP_TIME)

        speed += STEP

    # 최고 테스트 속도 유지
    print("1800000 유지")

    while True:
        write_pwm(STEER_PWM, "duty_cycle", CENTER)
        write_pwm(ESC_PWM, "duty_cycle", MAX_TEST_SPEED)
        time.sleep(1)


except KeyboardInterrupt:
    print("정지")


finally:
    # ESC 정지
    write_pwm(ESC_PWM, "duty_cycle", NEUTRAL)

    # 바퀴 중앙
    write_pwm(STEER_PWM, "duty_cycle", CENTER)

    time.sleep(1)

    sel_pwm.stop()
    GPIO.cleanup()