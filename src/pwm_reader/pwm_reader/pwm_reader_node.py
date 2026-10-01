import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray


class PWMReader(Node):

    def __init__(self):
        super().__init__('pwm_reader')

        self.publisher = self.create_publisher(
            Float32MultiArray,
            '/pwm_duty',
            10
        )

        self.timer = self.create_timer(0.02, self.read_pwm)

        self.pwm_paths = {
            'PWM1': '/sys/class/pwm/pwmchip0/pwm0',
            'PWM5': '/sys/class/pwm/pwmchip2/pwm0',
            'PWM7': '/sys/class/pwm/pwmchip3/pwm0',
        }

    def get_duty_percent(self, path):
        try:
            with open(f'{path}/period', 'r') as f:
                period = int(f.read().strip())

            with open(f'{path}/duty_cycle', 'r') as f:
                duty_cycle = int(f.read().strip())

            with open(f'{path}/enable', 'r') as f:
                enabled = int(f.read().strip())

            if enabled == 0 or period == 0:
                return 0.0

            return (duty_cycle / period) * 100.0

        except Exception as e:
            self.get_logger().warning(
                f'PWM read error: {path} -> {e}'
            )
            return 0.0

    def read_pwm(self):

        duty_pwm1 = self.get_duty_percent(
            self.pwm_paths['PWM1']
        )

        duty_pwm5 = self.get_duty_percent(
            self.pwm_paths['PWM5']
        )

        duty_pwm7 = self.get_duty_percent(
            self.pwm_paths['PWM7']
        )

        msg = Float32MultiArray()

        msg.data = [
            duty_pwm1,
            duty_pwm5,
            duty_pwm7
        ]

        self.publisher.publish(msg)

        self.get_logger().info(
            f'PWM1: {duty_pwm1:.2f}% | '
            f'PWM5: {duty_pwm5:.2f}% | '
            f'PWM7: {duty_pwm7:.2f}%'
        )


def main(args=None):
    rclpy.init(args=args)

    node = PWMReader()

    try:
        rclpy.spin(node)

    except KeyboardInterrupt:
        pass

    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
