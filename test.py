import Jetson.GPIO as GPIO
import time

GPIO.setwarnings(False)
GPIO.setmode(GPIO.BOARD)

# -------------------------
# 15번 : MUX 자동/수동 선택 PWM
# -------------------------
SEL_PIN = 15

# -------------------------
# 32번 : 조향 PWM
# 네 환경에서 32번이 pwmchip2라고 가정
# 만약 안 되면 pwmchip3로 바꿔
# -------------------------
STEER_PWM = "/sys/class/pwm/pwmchip2/pwm0"

# 조향 값
RIGHT = 1400000
CENTER = 1600000
LEFT = 1800000

def write_pwm(name, value):
    with open(f"{STEER_PWM}/{name}", "w") as f:
        f.write(str(value))

# -------------------------
# 15번 SEL PWM 설정
# -------------------------
GPIO.setup(SEL_PIN, GPIO.OUT)

sel_pwm = GPIO.PWM(SEL_PIN, 50)

# B 모드 = 자동 모드
# 50Hz에서 duty 10% ≈ 2ms
sel_pwm.start(10.0)

# -------------------------
# 32번 조향 PWM 설정
# -------------------------
try:
    write_pwm("enable", 0)
except:
    pass

write_pwm("period", 20000000)       # 50Hz
write_pwm("duty_cycle", CENTER)
write_pwm("enable", 1)

print("자동 모드 시작")

try:
    while True:

        print("왼쪽")
        write_pwm("duty_cycle", LEFT)
        time.sleep(2)

        print("중앙")
        write_pwm("duty_cycle", CENTER)
        time.sleep(2)

        print("오른쪽")
        write_pwm("duty_cycle", RIGHT)
        time.sleep(2)

        print("중앙")
        write_pwm("duty_cycle", CENTER)
        time.sleep(2)

except KeyboardInterrupt:
    print("종료")

finally:
    # 종료할 때 중앙
    write_pwm("duty_cycle", CENTER)
    time.sleep(0.5)

    try:
        write_pwm("enable", 0)
    except:
        pass

    sel_pwm.stop()
    GPIO.cleanup()