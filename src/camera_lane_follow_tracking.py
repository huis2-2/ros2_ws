#!/usr/bin/env python3
"""White-lane follower with sliding windows, curve fitting and temporal tracking.

Dry-run by default. Use --show to inspect, --drive only with wheels lifted first.
The white line is the steering target, consistent with camera_lane_follow.py.
"""
import argparse
import time
from dataclasses import dataclass

import cv2
import numpy as np

# Unchanged from the repository's camera_lane_follow.py
I2C_BUS, PCA_ADDR, PCA_FREQ = 7, 0x40, 50
THROTTLE_CH, STEERING_CH = 8, 9
MUX_PIN, MUX_FREQ, MUX_DUTY = 15, 50, 10.0
NEUTRAL_US, SLOW_SPEED_US, MAX_SPEED_US = 1500, 1600, 1600
STEERING_CENTER_US, STEERING_LEFT_US, STEERING_RIGHT_US = 1640, 1880, 1400
MODE1, PRESCALE, LED0_ON_L = 0x00, 0xFE, 0x06
CAMERA_TOPIC, IMAGE_TIMEOUT_SEC, LANE_CONFIRM_FRAMES = '/camera/image_raw', 0.5, 3
KP_LATERAL, KP_HEADING = 0.0042, 0.15
DEADBAND_PX, MAX_ANGULAR, SMOOTH_ALPHA, MAX_DELTA_ANGULAR = 24.0, 3.20, 0.60, 2.35
WHITE_LOWER_HSV, WHITE_UPPER_HSV = (0, 0, 252), (0, 116, 255)
YELLOW_LOWER_HSV, YELLOW_UPPER_HSV = (7, 21, 201), (179, 255, 255)

# Adjustable vision parameters (pixel parameters scale with image width).
WINDOW_COUNT = 8
WINDOW_MARGIN_AT_640 = 75
MIN_PIXELS_REPOSITION_AT_640 = 18
MIN_FIT_PIXELS_AT_640 = 100
MAX_FIT_RMSE_AT_640 = 28.0
MAX_TRACK_JUMP_AT_640 = 105.0
TRACK_SEARCH_MARGIN_AT_640 = 65
MAX_MISSED_FRAMES = 2         # Visualization only: NO driving on predicted frames.
FIT_SMOOTH_ALPHA = 0.55


def bird_eye_view(image):
    h, w = image.shape[:2]
    src = np.float32([(w*.30,h*.55),(w*.70,h*.55),(w*.96,h*.96),(w*.04,h*.96)])
    bev_width = int(w*.60)
    x_min = int((w-bev_width)/2)
    dst = np.float32([(x_min,0),(x_min+bev_width,0),(x_min+bev_width,h),(x_min,h)])
    return cv2.warpPerspective(image, cv2.getPerspectiveTransform(src,dst), (w,h))


def lane_color_mask(image, low, high):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv,np.array(low,np.uint8),np.array(high,np.uint8))
    h,w = mask.shape
    roi = np.zeros_like(mask)
    polygon = np.array([[(int(w*.08),int(h*.58)),(int(w*.92),int(h*.58)),
                         (int(w*.98),int(h*.98)),(int(w*.02),int(h*.98))]],np.int32)
    cv2.fillPoly(roi,polygon,255)
    kernel = np.ones((5,5),np.uint8)
    mask = cv2.morphologyEx(cv2.bitwise_and(mask,roi),cv2.MORPH_OPEN,kernel)
    return cv2.morphologyEx(mask,cv2.MORPH_CLOSE,kernel)


@dataclass
class LaneObservation:
    fit: np.ndarray
    center_x: float
    angle_deg: float
    rmse: float
    count: int
    source: str


class LaneTracker:
    def __init__(self):
        self.fit = None
        self.missed = 0

    def reset(self):
        self.fit = None
        self.missed = 0

    def _fit(self, xs, ys, w, h, source):
        scale = w/640.0
        if xs.size < max(40,int(MIN_FIT_PIXELS_AT_640*scale)):
            return None
        try:
            fit = np.polyfit(ys.astype(float),xs.astype(float),2)
        except (ValueError, np.linalg.LinAlgError):
            return None
        if not np.all(np.isfinite(fit)):
            return None
        residual = xs-np.polyval(fit,ys)
        rmse = float(np.sqrt(np.mean(residual*residual)))
        y_near, y_far = h*.90, h*.62
        x_near, x_far = float(np.polyval(fit,y_near)),float(np.polyval(fit,y_far))
        if rmse > MAX_FIT_RMSE_AT_640*scale:
            return None
        if not (-w*.1 <= x_near <= w*1.1 and -w*.1 <= x_far <= w*1.1):
            return None
        # x changes with y; 90 degrees is a perfectly vertical line.
        slope = 2*fit[0]*h*.78+fit[1]
        angle = 90.0-np.degrees(np.arctan(slope))
        return LaneObservation(fit,x_near,float(angle),rmse,len(xs),source)

    def _windows(self, mask):
        h,w = mask.shape
        ys,xs = mask.nonzero()
        start,end = int(h*.58),int(h*.98)
        valid = (ys>=start)&(ys<end)
        ys,xs = ys[valid],xs[valid]
        if not xs.size:
            return xs,ys
        # Base position: prefer prior estimate; else dominant histogram peak.
        if self.fit is not None:
            center = int(np.polyval(self.fit,h*.92))
        else:
            base = ys>=int(h*.83)
            hist = np.bincount(xs[base],minlength=w) if np.any(base) else np.bincount(xs,minlength=w)
            hist = np.convolve(hist,np.ones(max(5,int(w*.035))),mode='same')
            center = int(np.argmax(hist))
        selected = []
        margin = max(12,int(WINDOW_MARGIN_AT_640*w/640))
        minimum = max(5,int(MIN_PIXELS_REPOSITION_AT_640*w/640))
        step = max(1,(end-start)//WINDOW_COUNT)
        for i in range(WINDOW_COUNT):
            lo,hi=end-(i+1)*step,end-i*step
            indices=np.where((ys>=lo)&(ys<hi)&(np.abs(xs-center)<margin))[0]
            if indices.size:
                selected.append(indices)
            if indices.size>=minimum:
                center=int(np.median(xs[indices]))
        if not selected:
            return np.empty(0,dtype=int),np.empty(0,dtype=int)
        idx=np.concatenate(selected)
        return xs[idx],ys[idx]

    def _previous_search(self,mask):
        ys,xs=mask.nonzero()
        h,w=mask.shape
        valid=(ys>=int(h*.58))&(ys<int(h*.98))
        margin=max(12,int(TRACK_SEARCH_MARGIN_AT_640*w/640))
        valid &= np.abs(xs-np.polyval(self.fit,ys))<margin
        return xs[valid],ys[valid]

    def detect(self,mask):
        h,w=mask.shape
        observation=None
        if self.fit is not None:
            xs,ys=self._previous_search(mask)
            observation=self._fit(xs,ys,w,h,'previous')
        if observation is None:
            xs,ys=self._windows(mask)
            observation=self._fit(xs,ys,w,h,'windows')
        if observation is not None and self.fit is not None:
            jump=abs(observation.center_x-float(np.polyval(self.fit,h*.90)))
            if jump>MAX_TRACK_JUMP_AT_640*w/640:
                observation=None
        if observation is None:
            self.missed+=1
            if self.missed>MAX_MISSED_FRAMES:
                self.reset()
            return None
        self.missed=0
        if self.fit is None:
            self.fit=observation.fit.copy()
        else:
            self.fit=FIT_SMOOTH_ALPHA*observation.fit+(1-FIT_SMOOTH_ALPHA)*self.fit
        observation.fit=self.fit.copy()
        observation.center_x=float(np.polyval(self.fit,h*.90))
        slope=2*self.fit[0]*h*.78+self.fit[1]
        observation.angle_deg=float(90.0-np.degrees(np.arctan(slope)))
        return observation


class LaneController:
    def __init__(self):
        self.previous_command=0.0

    def reset(self):
        self.previous_command=0.0

    def steering_pulse(self,center_x,angle_deg,image_width):
        scale=image_width/640.0
        lateral=center_x-image_width/2.0
        heading=angle_deg-90.0
        if abs(lateral)<DEADBAND_PX*scale: lateral=0.0
        if abs(heading)<2.0: heading=0.0
        cmd=float(np.clip(-KP_LATERAL/scale*lateral-KP_HEADING*heading,-MAX_ANGULAR,MAX_ANGULAR))
        cmd=SMOOTH_ALPHA*cmd+(1-SMOOTH_ALPHA)*self.previous_command
        cmd=float(np.clip(cmd,self.previous_command-MAX_DELTA_ANGULAR,
                          self.previous_command+MAX_DELTA_ANGULAR))
        if cmd>=0:
            pulse=STEERING_CENTER_US+cmd/MAX_ANGULAR*(STEERING_LEFT_US-STEERING_CENTER_US)
        else:
            pulse=STEERING_CENTER_US+(-cmd)/MAX_ANGULAR*(STEERING_RIGHT_US-STEERING_CENTER_US)
        self.previous_command=cmd
        return int(round(np.clip(pulse,STEERING_RIGHT_US,STEERING_LEFT_US)))


def init_pca9685(bus):
    prescale=int(round(25_000_000.0/(4096.0*PCA_FREQ)-1.0))
    old=bus.read_byte_data(PCA_ADDR,MODE1)
    awake=old & ~0x10
    bus.write_byte_data(PCA_ADDR,MODE1,(awake & 0x7F)|0x10)
    bus.write_byte_data(PCA_ADDR,PRESCALE,prescale)
    bus.write_byte_data(PCA_ADDR,MODE1,awake)
    time.sleep(.005)
    bus.write_byte_data(PCA_ADDR,MODE1,awake|0xA1)
    time.sleep(.01)


def set_pwm_us(bus,channel,pulse):
    ticks=max(0,min(4095,int(pulse*PCA_FREQ*4096/1_000_000)))
    register=LED0_ON_L+4*channel
    for offset,value in enumerate((0,0,ticks&255,(ticks>>8)&15)):
        bus.write_byte_data(PCA_ADDR,register+offset,value)


class DriveHardware:
    def __init__(self):
        from smbus2 import SMBus
        import Jetson.GPIO as GPIO
        self.bus=None
        self.gpio=GPIO
        self.mux_pwm=None
        self.speed_us=None
        self.steering_us=None
        try:
            self.bus=SMBus(I2C_BUS)
            init_pca9685(self.bus)
            self.set_speed(NEUTRAL_US)
            self.set_steering(STEERING_CENTER_US)
            GPIO.setwarnings(False)
            GPIO.setmode(GPIO.BOARD)
            GPIO.setup(MUX_PIN,GPIO.OUT)
            self.mux_pwm=GPIO.PWM(MUX_PIN,MUX_FREQ)
            self.mux_pwm.start(MUX_DUTY)
        except Exception:
            self.close()
            raise

    def set_speed(self,pulse):
        if self.bus is not None and pulse!=self.speed_us:
            set_pwm_us(self.bus,THROTTLE_CH,pulse)
            self.speed_us=pulse

    def set_steering(self,pulse):
        if self.bus is not None and pulse!=self.steering_us:
            set_pwm_us(self.bus,STEERING_CH,pulse)
            self.steering_us=pulse

    def stop(self):
        self.set_speed(NEUTRAL_US)
        self.set_steering(STEERING_CENTER_US)

    def close(self):
        if self.bus is not None:
            try:
                self.stop()
                time.sleep(.2)
            except OSError as exc:
                print(f'Safe output failed: {exc}')
        if self.mux_pwm is not None: self.mux_pwm.stop()
        if self.gpio is not None: self.gpio.cleanup()
        if self.bus is not None: self.bus.close()


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--drive',action='store_true',help='enable vehicle hardware (after bench testing)')
    p.add_argument('--show',action='store_true',help='show camera result and masks')
    p.add_argument('--camera-topic',default=CAMERA_TOPIC)
    p.add_argument('--speed-us',type=int,default=SLOW_SPEED_US)
    opts,ros_args=p.parse_known_args(argv)
    if not NEUTRAL_US<=opts.speed_us<=MAX_SPEED_US:
        p.error(f'--speed-us must be {NEUTRAL_US}..{MAX_SPEED_US}')
    try:
        import rclpy
        from rclpy.node import Node
        from rclpy.qos import qos_profile_sensor_data
        from sensor_msgs.msg import Image
        from cv_bridge import CvBridge
    except ModuleNotFoundError as exc:
        print(f'Missing ROS 2 module: {exc}; source /opt/ros/jazzy/setup.bash')
        return 2

    class CameraLaneFollower(Node):
        def __init__(self, hardware):
            super().__init__('camera_lane_follower_tracking')
            self.hardware=hardware
            self.bridge=CvBridge()
            self.tracker=LaneTracker()
            self.controller=LaneController()
            self.started=time.monotonic()
            self.last_image=None
            self.last_log=0.0
            self.last_key=None
            self.confirm=0
            self.timeout_reported=False
            self.create_subscription(Image,opts.camera_topic,self.callback,qos_profile_sensor_data)
            self.create_timer(.1,self.watchdog)
            self.get_logger().info(f'Mode: {"DRIVE" if hardware else "DRY RUN"}, topic={opts.camera_topic}')

        def apply(self,pulse,moving):
            if self.hardware is not None:
                if moving:
                    self.hardware.set_steering(pulse)
                    self.hardware.set_speed(opts.speed_us)
                else:
                    self.hardware.stop()

        def report(self,key,message,warning=False):
            now=time.monotonic()
            if key!=self.last_key or now-self.last_log>=1.0:
                (self.get_logger().warning if warning else self.get_logger().info)(message)
                self.last_log=now
                self.last_key=key

        def callback(self,msg):
            self.last_image=time.monotonic()
            self.timeout_reported=False
            t0=time.perf_counter()
            try:
                frame=self.bridge.imgmsg_to_cv2(msg,desired_encoding='bgr8')
                bev=bird_eye_view(frame)
                white=lane_color_mask(bev,WHITE_LOWER_HSV,WHITE_UPPER_HSV)
                yellow=lane_color_mask(bev,YELLOW_LOWER_HSV,YELLOW_UPPER_HSV)
                obs=self.tracker.detect(white)
            except Exception as exc:
                self.tracker.reset()
                self.controller.reset()
                self.confirm=0
                self.apply(STEERING_CENTER_US,False)
                self.report('error',f'Image error, stopped: {exc}',True)
                return
            elapsed=(time.perf_counter()-t0)*1000
            if obs is None:
                self.controller.reset()
                self.confirm=0
                pulse=STEERING_CENTER_US
                moving=False
                self.apply(pulse,False)
                self.report('lost',f'LOST / stopped, vision={elapsed:.1f} ms',True)
            else:
                pulse=self.controller.steering_pulse(obs.center_x,obs.angle_deg,frame.shape[1])
                self.confirm+=1
                moving=self.confirm>=LANE_CONFIRM_FRAMES
                self.apply(pulse,moving)
                label='TRACKING' if moving else 'CONFIRMING'
                self.report(label,f'{label}: x={obs.center_x:.1f} angle={obs.angle_deg:.1f} '
                            f'rmse={obs.rmse:.1f} n={obs.count} source={obs.source} '
                            f'steer={pulse} us vision={elapsed:.1f} ms')
            if opts.show:
                display=bev.copy()
                h,w=display.shape[:2]
                if self.tracker.fit is not None:
                    ys=np.arange(int(h*.58),int(h*.98),3)
                    xs=np.polyval(self.tracker.fit,ys)
                    valid=(xs>=0)&(xs<w)
                    points=np.stack((xs[valid],ys[valid]),axis=-1).astype(np.int32)
                    if len(points)>1:
                        cv2.polylines(display,[points],False,(0,255,0),3)
                if obs is not None:
                    cv2.circle(display,(int(obs.center_x),int(h*.90)),7,(0,0,255),-1)
                label='DRIVE' if moving else ('CONFIRM' if obs is not None else 'LOST/STOP')
                cv2.putText(display,f'{label} steer={pulse}us vision={elapsed:.1f}ms',(12,32),
                            cv2.FONT_HERSHEY_SIMPLEX,.55,(0,255,255),2)
                cv2.imshow('tracked white lane',display)
                cv2.imshow('white mask - input',white)
                cv2.imshow('yellow mask - preview only',yellow)
                cv2.waitKey(1)

        def watchdog(self):
            if time.monotonic()-(self.last_image or self.started)<=IMAGE_TIMEOUT_SEC:
                return
            self.tracker.reset()
            self.controller.reset()
            self.confirm=0
            self.apply(STEERING_CENTER_US,False)
            if not self.timeout_reported:
                self.get_logger().error(f'Camera timeout > {IMAGE_TIMEOUT_SEC}s; stopped')
                self.timeout_reported=True

    hardware=None
    node=None
    ros_started=False
    try:
        if opts.drive:
            hardware=DriveHardware()
            print('ESC neutral stabilization (3 s)')
            time.sleep(3)
        rclpy.init(args=ros_args)
        ros_started=True
        node=CameraLaneFollower(hardware)
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as exc:
        print(f'Runtime error: {exc}')
        return 1
    finally:
        if hardware is not None: hardware.close()
        if node is not None: node.destroy_node()
        cv2.destroyAllWindows()
        if ros_started and rclpy.ok(): rclpy.shutdown()
    return 0

if __name__=='__main__':
    raise SystemExit(main())
