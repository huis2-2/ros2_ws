import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2


class CameraNode(Node):

    def __init__(self):
        super().__init__('camera_node')

        # ROS 2 이미지 토픽 Publisher
        self.publisher = self.create_publisher(
            Image,
            '/camera/image_raw',
            10
        )

        # OpenCV ↔ ROS 2 변환
        self.bridge = CvBridge()

        # USB 카메라 열기
        self.cap = cv2.VideoCapture(
            '/dev/video0',
            cv2.CAP_V4L2
        )

        if not self.cap.isOpened():
            self.get_logger().error('카메라를 열 수 없습니다.')
            return

        # 카메라 해상도 설정
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        # FPS 설정
        self.cap.set(cv2.CAP_PROP_FPS, 30)

        # 30 FPS로 이미지 발행
        self.timer = self.create_timer(
            1.0 / 30.0,
            self.publish_image
        )

        self.get_logger().info('카메라 노드가 시작되었습니다.')

    def publish_image(self):

        # 카메라에서 프레임 읽기
        ret, frame = self.cap.read()

        if not ret:
            self.get_logger().warning(
                '카메라 프레임을 읽을 수 없습니다.'
            )
            return

        # OpenCV → ROS 2 Image 메시지 변환
        msg = self.bridge.cv2_to_imgmsg(
            frame,
            encoding='bgr8'
        )

        # ROS 2 토픽으로 발행
        self.publisher.publish(msg)

        # OpenCV 화면에 카메라 영상 표시
        cv2.imshow('Camera', frame)

        # 키 입력 처리
        cv2.waitKey(1)

    def destroy_node(self):

        # 카메라 종료
        if hasattr(self, 'cap'):
            self.cap.release()

        # OpenCV 창 종료
        cv2.destroyAllWindows()

        super().destroy_node()


def main(args=None):

    rclpy.init(args=args)

    node = CameraNode()

    try:
        rclpy.spin(node)

    except (KeyboardInterrupt, ExternalShutdownException):
        pass

    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
