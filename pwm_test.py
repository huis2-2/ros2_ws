import Jetson.GPIO as GPIO
import time

# MUX SEL
SEL_PIN = 15

# ★ 32번 핀에 해당하는 실제 PWM 경로로 수정
PWM_PATH = "/sys/class/pwm/pwmchip0/pwm0"

# 서보 PWM 값 (ns)
CENTER = 1500000
LEFT   = 1200000
RIGHT  = 1800000

GPIO.setmode(GPIO.BOARD)

# 15번 → MUX SEL
GPIO.setup(SEL_PIN, GPIO.OUT)

# 자동 모드 선택
# HIGH/LOW가 반대라면 GPIO.LOW로 바꾸기
GPIO.output(SEL_PIN, GPIO.LOW)

def steering(value):
    with open(PWM_PATH + "/duty_cycle", "w") as f:
        f.write(str(value))

try:
    while True:
        print("왼쪽")
        steering(LEFT)
        time.sleep(2)

        print("오른쪽")
        steering(RIGHT)
        time.sleep(2)

except KeyboardInterrupt:
    print("종료")

    # 종료할 때 바퀴 중앙
    steering(CENTER)

finally:
    GPIO.cleanup()
