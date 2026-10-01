import Jetson.GPIO as GPIO
import time

GPIO.setwarnings(False)
GPIO.setmode(GPIO.BOARD)

SEL_PIN = 15
STEER_PWM = "/sys/class/pwm/pwmchip2/pwm0"

RIGHT  = 1400000
CENTER = 1600000
LEFT   = 1800000

def write_pwm(name, value):
    with open(f"{STEER_PWM}/{name}", "w") as f:
        f.write(str(value))

GPIO.setup(SEL_PIN, GPIO.OUT)
sel_pwm = GPIO.PWM(SEL_PIN, 50)
sel_pwm.start(10.0)   # 자동 모드

try:
    write_pwm("enable", 0)
except:
    pass

write_pwm("period", 20000000)
write_pwm("duty_cycle", CENTER)
write_pwm("enable", 1)

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
    write_pwm("duty_cycle", CENTER)

finally:
    sel_pwm.stop()
    GPIO.cleanup()