import Jetson.GPIO as GPIO
import time

GPIO.setwarnings(False)
GPIO.setmode(GPIO.BOARD)

# -------------------------
# 15번 : MUX 자동모드 선택
# -------------------------
SEL_PIN = 15

# -------------------------
# PWM 경로
# -------------------------
STEER_PWM = "/sys/class/pwm/pwmchip2/pwm0"   # 32번
ESC_PWM   = "/sys/class/pwm/pwmchip3/pwm0"   # 33번

# -------------------------
# 조향 값
# -------------------------
CENTER = 1600000

# -------------------------
# ESC 값
# 실제 차량에 따라 조정 필요
# -------------------------
NEUTRAL = 1500000
SLOW    = 1562000
FAST    = 1700000


def write_pwm(path, name, value):
    with open(f"{path}/{name}", "w") as f:
        f.write(str(value))


# =========================
# 15번 MUX 자동모드
# =========================
GPIO.setup(SEL_PIN, GPIO.OUT)

sel_pwm = GPIO.PWM(SEL_PIN, 50)

# B = 자동모드
sel_pwm.start(10.0)


# =========================
# 32번 조향 초기화
# =========================
try:
    write_pwm(STEER_PWM, "enable", 0)
except:
    pass

write_pwm(STEER_PWM, "period", 20000000)
write_pwm(STEER_PWM, "duty_cycle", CENTER)
write_pwm(STEER_PWM, "enable", 1)


# =========================
# 33번 ESC 초기화
# =========================
try:
    write_pwm(ESC_PWM, "enable", 0)
except:
    pass

write_pwm(ESC_PWM, "period", 20000000)
write_pwm(ESC_PWM, "duty_cycle", NEUTRAL)
write_pwm(ESC_PWM, "enable", 1)

print("ESC 중립 대기")
time.sleep(2)


try:
    while True:

        # 직진 느리게
        print("직진 - 느리게")
        write_pwm(STEER_PWM, "duty_cycle", CENTER)
        write_pwm(ESC_PWM, "duty_cycle", SLOW)
        time.sleep(3)

        # 잠깐 정지
        print("정지")
        write_pwm(ESC_PWM, "duty_cycle", NEUTRAL)
        time.sleep(2)

        # 직진 조금 빠르게
        print("직진 - 조금 빠르게")
        write_pwm(STEER_PWM, "duty_cycle", CENTER)
        write_pwm(ESC_PWM, "duty_cycle", FAST)
        time.sleep(3)

        # 정지
        print("정지")
        write_pwm(ESC_PWM, "duty_cycle", NEUTRAL)
        time.sleep(2)


except KeyboardInterrupt:
    print("종료")


finally:
    # 정지 + 중앙
    write_pwm(ESC_PWM, "duty_cycle", NEUTRAL)
    write_pwm(STEER_PWM, "duty_cycle", CENTER)

    time.sleep(1)

    try:
        write_pwm(STEER_PWM, "enable", 0)
        write_pwm(ESC_PWM, "enable", 0)
    except:
        pass

    sel_pwm.stop()
    GPIO.cleanup()