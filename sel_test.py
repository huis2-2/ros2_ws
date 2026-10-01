import Jetson.GPIO as GPIO
import time

GPIO.setwarnings(False)
GPIO.setmode(GPIO.BOARD)

SEL = 15
GPIO.setup(SEL, GPIO.OUT)

while True:
    GPIO.output(SEL, GPIO.HIGH)
    print("SEL HIGH")
    time.sleep(2)

    GPIO.output(SEL, GPIO.LOW)
    print("SEL LOW")
    time.sleep(2)