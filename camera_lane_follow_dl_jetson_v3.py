#!/usr/bin/env python3
"""ROS2/MP4 lane follower: trained 3-class ONNX segmentation + sliding windows.
Default is diagnostic only. --drive can actuate only after explicit --allow-drive.
Classes: 0 background, 1 white, 2 yellow. Model input RGB /255, NCHW 320x256.
"""
import argparse
import time
from dataclasses import dataclass
from pathlib import Path
import cv2
import numpy as np

# Existing vehicle calibration (camera_lane_follow.py)
I2C_BUS, PCA_ADDR, PCA_FREQ = 7, 0x40, 50
THROTTLE_CH, STEERING_CH = 8, 9
MUX_PIN, MUX_FREQ, MUX_DUTY = 15, 50, 10.0
NEUTRAL_US, SPEED_US, MAX_SPEED_US = 1500, 1600, 1600
STEERING_CENTER_US, STEERING_LEFT_US, STEERING_RIGHT_US = 1640, 1880, 1400
CAMERA_TOPIC = '/camera/image_raw'
KP_LATERAL, KP_HEADING = 0.0042, 0.15
DEADBAND_PX, MAX_ANGULAR = 24.0, 3.20
SMOOTH_ALPHA, MAX_DELTA_ANGULAR = 0.60, 2.35

@dataclass
class Lane:
    coeff: np.ndarray
    x: float
    angle: float
    support: int
    span: float
    error: float
    confidence: float
    pixels: np.ndarray


def bev_matrix(w, h):
    src = np.float32([(w*.30,h*.55),(w*.70,h*.55),(w*.96,h*.96),(w*.04,h*.96)])
    bw=w*.60; left=(w-bw)/2
    dst = np.float32([(left,0),(left+bw,0),(left+bw,h),(left,h)])
    return cv2.getPerspectiveTransform(src,dst)

class Segmenter:
    """ONNX Runtime inference with a GPU-first provider configuration."""
    def __init__(self, path, backend='auto', crop_top=0.10,
                 trt_cache='~/.cache/ros2_lane_follow/trt', trt_fp16=True):
        import onnxruntime as ort
        self.crop_top = float(crop_top)
        self.backend = backend
        model_path = Path(path).expanduser().resolve()
        if not model_path.is_file():
            raise FileNotFoundError(f'ONNX model not found: {model_path}')
        available = ort.get_available_providers()
        trt = 'TensorrtExecutionProvider'
        cuda = 'CUDAExecutionProvider'
        if backend == 'auto':
            backend = 'tensorrt' if trt in available else 'cuda' if cuda in available else None
            if backend is None:
                raise RuntimeError(
                    f'GPU provider unavailable; installed providers={available}. '
                    'Use --backend cpu only for offline diagnostics.'
                )
        providers = []
        if backend == 'tensorrt':
            if trt not in available:
                raise RuntimeError(f'{trt} unavailable; installed providers={available}')
            cache = Path(trt_cache).expanduser().resolve()
            cache.mkdir(parents=True, exist_ok=True)
            providers.append((trt, {
                'trt_engine_cache_enable': '1',
                'trt_engine_cache_path': str(cache),
                'trt_timing_cache_enable': '1',
                'trt_timing_cache_path': str(cache),
                'trt_fp16_enable': '1' if trt_fp16 else '0',
            }))
            # TensorRT can delegate unsupported GPU nodes to CUDA.
            if cuda in available:
                providers.append(cuda)
        elif backend == 'cuda':
            if cuda not in available:
                raise RuntimeError(f'{cuda} unavailable; installed providers={available}')
            providers.append((cuda, {
                'cudnn_conv_algo_search': 'HEURISTIC',
                'do_copy_in_default_stream': '1',
            }))
        else:
            providers.append('CPUExecutionProvider')
        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        options.log_severity_level = 3
        self.sess = ort.InferenceSession(
            str(model_path), sess_options=options, providers=providers,
        )
        active = self.sess.get_providers()
        expected = trt if backend == 'tensorrt' else cuda if backend == 'cuda' else 'CPUExecutionProvider'
        if not active or active[0] != expected:
            raise RuntimeError(f'GPU provider activation failed: {active}')
        self.backend = backend
        self.key = self.sess.get_inputs()[0].name
        self.output_name = self.sess.get_outputs()[0].name
        shape = self.sess.get_inputs()[0].shape
        if len(shape) != 4 or (isinstance(shape[1], int) and shape[1] != 3):
            raise RuntimeError(f'Unexpected model input {shape}; expected NCHW RGB')
        self.input_width = shape[3] if isinstance(shape[3], int) else 320
        self.input_height = shape[2] if isinstance(shape[2], int) else 256
        self.blob = np.empty((1, 3, self.input_height, self.input_width), dtype=np.float32)
        self.last_infer_ms = 0.0
        warmup_start = time.perf_counter()
        self.sess.run([self.output_name], {self.key: np.zeros_like(self.blob)})
        self.warmup_ms = (time.perf_counter() - warmup_start) * 1000.0

    def predict(self, frame):
        h,w=frame.shape[:2]
        y0 = int(h*self.crop_top)
        crop=frame[y0:]
        resized=cv2.resize(crop,(self.input_width,self.input_height),interpolation=cv2.INTER_LINEAR)
        rgb=cv2.cvtColor(resized,cv2.COLOR_BGR2RGB)
        np.divide(rgb.transpose(2,0,1),255.0,out=self.blob[0],casting='unsafe')
        t0=time.perf_counter()
        logits=self.sess.run([self.output_name],{self.key:self.blob})[0]
        self.last_infer_ms=(time.perf_counter()-t0)*1000.
        if logits.ndim!=4 or logits.shape[1]!=3:
            raise RuntimeError(f'Unexpected model output {logits.shape}')
        pred=np.argmax(logits,axis=1)[0].astype('uint8')
        pred=cv2.resize(pred,(w,h-y0),interpolation=cv2.INTER_NEAREST)
        full=np.zeros((h,w),np.uint8)
        full[y0:]=pred
        return np.uint8(full==1)*255, np.uint8(full==2)*255


def windows(mask, anchor, nw=9, margin=30, minpix=8):
    """Search only inside each window ROI; avoid repeated full-mask scans."""
    h,w=mask.shape
    center=float(anchor); chunks=[]; boxes=[]
    top=int(h*.50); bottom=int(h*.98)
    winh=max(8,int((bottom-top)/nw))
    misses=0
    for i in range(nw):
        yhi=bottom-i*winh
        ylo=max(top,yhi-winh)
        if yhi<=ylo: break
        lo=max(0,int(center-margin)); hi=min(w,int(center+margin))
        if lo>=hi: break
        boxes.append((lo,ylo,hi,yhi))
        local_y,local_x=np.nonzero(mask[ylo:yhi,lo:hi])
        if local_x.size>=minpix:
            abs_x=local_x+lo
            abs_y=local_y+ylo
            center=float(np.median(abs_x))
            chunks.append(np.column_stack((abs_x,abs_y)))
            misses=0
        else:
            misses+=1
            if misses>=3:break
    if not chunks:return None, boxes
    pts=np.concatenate(chunks,axis=0)
    if len(pts)<38:return None,boxes
    return pts,boxes


def fit_lane(pts,w,h):
    if pts is None or len(pts)<38:return None
    xx=pts[:,0].astype('float64'); yy=pts[:,1].astype('float64')
    keep=np.ones(len(xx),bool)
    for _ in range(4):
        if keep.sum()<38:return None
        c=np.polyfit(yy[keep],xx[keep],2)
        r=np.abs(xx-np.polyval(c,yy)); nxt=r<max(4., min(12.,float(np.median(r[keep])*2.5+2)))
        if np.array_equal(nxt,keep):break
        keep=nxt
    if keep.sum()<38:return None
    ygood=yy[keep]; xgood=xx[keep]
    span=(ygood.max()-ygood.min())/h
    err=float(np.sqrt(np.mean((xgood-np.polyval(c,ygood))**2)))
    if span<.17 or err>9:return None
    ref=h*.82
    x=float(np.polyval(c,ref))
    if not (w*.02<x<w*.98):return None
    slope=float(np.polyval(np.polyder(c),ref))
    angle=float(np.degrees(np.arctan2(1.,slope)))
    conf=float(np.clip(.65*span/.38+.35*(1.-err/9.),0,1))
    return Lane(c,x,angle,int(keep.sum()),float(span),err,conf,pts[keep])


class LaneTracker:
    def __init__(self,initial_x=.50):
        self.initial_x=initial_x
        self.locked=False
        self.last=None
        self.misses=0
        self.confirm=0
        self.last_at=0.

    def update(self,mask,now):
        h,w=mask.shape
        # bottom-row histogram, local maxima with suppression.
        crop=mask[int(h*.73):int(h*.96)]>0
        hist=np.count_nonzero(crop,axis=0).astype('float32')
        smooth=np.convolve(hist,np.ones(13)/13,'same')
        if not self.locked:
            # Restrict first lock near explicit starting target; never relock automatically.
            guess=self.initial_x*w
            candidates=np.where((np.arange(w)>guess-w*.13)&(np.arange(w)<guess+w*.13))[0]
            # local peak suppression: choose a few separated histogram anchors
            order=candidates[np.argsort(smooth[candidates])[::-1]]
            cand=[]
            for pos in order:
                if smooth[pos]<2:break
                if all(abs(int(pos)-int(previous))>=18 for previous in cand):
                    cand.append(int(pos))
                if len(cand)>=6:break
        else:
            guess=self.last.x if self.last is not None else self.initial_x*w
            candidates=np.where((np.arange(w)>guess-w*.13)&(np.arange(w)<guess+w*.13))[0]
            # local peak suppression: choose a few separated histogram anchors
            order=candidates[np.argsort(smooth[candidates])[::-1]]
            cand=[]
            for pos in order:
                if smooth[pos]<2:break
                if all(abs(int(pos)-int(previous))>=18 for previous in cand):
                    cand.append(int(pos))
                if len(cand)>=6:break
        fits=[]
        for x in cand:
            if smooth[x]<2:break
            if any(abs(x-old)>-1 and abs(x-old)<18 for old in [q[0] for q in fits]):continue
            pts,boxes=windows(mask,x,margin=max(12,int(w*.075)),minpix=5)
            lane=fit_lane(pts,w,h)
            if lane is not None:fits.append((x,lane,boxes))
            if len(fits)>=4:break
        # Existing track must be geometrically continuous, and never jump to another lane.
        if self.last is None:
            good=[f for f in fits if abs(f[1].x-self.initial_x*w)<w*.13 and f[1].confidence>.45]
            best=min(good,key=lambda f:abs(f[1].x-self.initial_x*w)) if good else None
        else:
            good=[f for f in fits if abs(f[1].x-self.last.x)<w*.095 and
                  abs(f[1].angle-self.last.angle)<27 and f[1].confidence>.42]
            best=min(good,key=lambda f:abs(f[1].x-self.last.x)) if good else None
        if best is None:
            self.misses+=1; self.confirm=0
            # DO NOT move from prediction alone: hold visualization, not throttle.
            # Preserve last physical target identity. Never automatically choose another white line.
            return None, fits, []
        lane=best[1]
        self.misses=0
        if not self.locked:
            self.confirm+=1
            if self.confirm>=3:self.locked=True
        self.last=lane; self.last_at=now
        return lane if self.locked else None, fits, best[2]


def pulse_for_lane(lane,w,previous):
    scale=w/640.
    lateral=lane.x-w/2
    if abs(lateral)<DEADBAND_PX*scale:lateral=0.
    heading=lane.angle-90
    if abs(heading)<2:heading=0.
    cmd=float(np.clip(-KP_LATERAL/scale*lateral-KP_HEADING*heading,-MAX_ANGULAR,MAX_ANGULAR))
    cmd=SMOOTH_ALPHA*cmd+(1-SMOOTH_ALPHA)*previous
    cmd=float(np.clip(cmd,previous-MAX_DELTA_ANGULAR,previous+MAX_DELTA_ANGULAR))
    if cmd>=0: pulse=STEERING_CENTER_US+cmd/MAX_ANGULAR*(STEERING_LEFT_US-STEERING_CENTER_US)
    else:pulse=STEERING_CENTER_US+(-cmd)/MAX_ANGULAR*(STEERING_RIGHT_US-STEERING_CENTER_US)
    return int(np.clip(round(pulse),STEERING_RIGHT_US,STEERING_LEFT_US)),cmd


def draw_lane(img,lane,col,thickness=3):
    h,w=img.shape[:2]
    ys=np.arange(int(h*.50),int(h*.98),3)
    xs=np.polyval(lane.coeff,ys)
    valid=(xs>=0)&(xs<w)
    arr=np.column_stack((xs[valid],ys[valid])).astype('int32')
    if len(arr)>1:cv2.polylines(img,[arr],False,col,thickness)


def decision_view(bev,lane,fits,boxes,status,pulse,tms):
    result=bev.copy(); h,w=result.shape[:2]
    for _,ln,_ in fits:
        draw_lane(result,ln,(255,255,0),1)
    for x0,y0,x1,y1 in boxes:
        cv2.rectangle(result,(x0,y0),(x1,y1),(160,95,40),1)
    if lane is not None:
        draw_lane(result,lane,(0,255,0),3)
        cv2.circle(result,(int(lane.x),int(h*.82)),5,(0,0,255),-1)
    cv2.rectangle(result,(0,0),(w,52),(0,0,0),-1)
    cv2.putText(result,f'{status} PWM={pulse}',(6,21),cv2.FONT_HERSHEY_SIMPLEX,.45,(255,255,255),1)
    cv2.putText(result,f'TOTAL={tms:.1f}ms',(6,43),cv2.FONT_HERSHEY_SIMPLEX,.45,(255,255,255),1)
    return result

def debug_panel(frame,white,yellow,bev,lane,fits,boxes,status,pulse,tms):
    result=bev.copy(); h,w=result.shape[:2]
    for _,ln,_ in fits:
        for x,y in ln.pixels[::max(1,len(ln.pixels)//160)]:
            cv2.circle(result,(int(x),int(y)),1,(255,180,30),-1)
        draw_lane(result,ln,(255,255,0),1)
    for x0,y0,x1,y1 in boxes:cv2.rectangle(result,(x0,y0),(x1,y1),(160,95,40),1)
    if lane is not None:
        draw_lane(result,lane,(0,255,0),3)
        cv2.circle(result,(int(lane.x),int(h*.82)),6,(0,0,255),-1)
    cv2.rectangle(result,(0,0),(w,62),(0,0,0),-1)
    cv2.putText(result,f'{status}  STEER {pulse}us',(7,24),cv2.FONT_HERSHEY_SIMPLEX,.55,(255,255,255),1)
    cv2.putText(result,f'PROCESS {tms:.1f}ms',(7,48),cv2.FONT_HERSHEY_SIMPLEX,.52,(255,255,255),1)
    sz=(384,312)
    def pane(im,title):
        im=cv2.resize(im,sz)
        if im.ndim==2:im=cv2.cvtColor(im,cv2.COLOR_GRAY2BGR)
        cv2.rectangle(im,(0,0),(384,28),(0,0,0),-1)
        cv2.putText(im,title,(7,20),cv2.FONT_HERSHEY_SIMPLEX,.55,(255,255,255),1)
        return im
    return np.vstack((np.hstack((pane(frame,'1 Camera'),pane(white,'2 White Mask'))),
                      np.hstack((pane(yellow,'3 Yellow Mask'),pane(result,'4 Lane Decision')))))


# PCA9685 / MUX follows existing project's channels and settings.
def init_pca(bus):
    a=0x00; p=0xFE
    prescale=int(round(25_000_000/(4096*PCA_FREQ)-1))
    old=bus.read_byte_data(PCA_ADDR,a); awake=old&~0x10
    bus.write_byte_data(PCA_ADDR,a,(awake&0x7f)|0x10)
    bus.write_byte_data(PCA_ADDR,p,prescale)
    bus.write_byte_data(PCA_ADDR,a,awake);time.sleep(.005)
    bus.write_byte_data(PCA_ADDR,a,awake|0xA1);time.sleep(.01)

def pwm(bus,ch,microseconds):
    ticks=int(microseconds*PCA_FREQ*4096/1_000_000)
    base=0x06+4*ch
    for i,v in enumerate([0,0,ticks&255,(ticks>>8)&15]):bus.write_byte_data(PCA_ADDR,base+i,v)

class Hardware:
    def __init__(self):
        from smbus2 import SMBus
        import Jetson.GPIO as GPIO
        self.bus=None;self.gpio=GPIO;self.mux=None
        self.last_throttle=None;self.last_steering=None
        try:
            self.bus=SMBus(I2C_BUS);init_pca(self.bus)
            self._throttle(NEUTRAL_US);self._steering(STEERING_CENTER_US)
            GPIO.setwarnings(False);GPIO.setmode(GPIO.BOARD);GPIO.setup(MUX_PIN,GPIO.OUT)
            self.mux=GPIO.PWM(MUX_PIN,MUX_FREQ);self.mux.start(MUX_DUTY)
        except Exception:self.close();raise
    def _throttle(self,value):
        if value != self.last_throttle:
            pwm(self.bus,THROTTLE_CH,value);self.last_throttle=value
    def _steering(self,value):
        if value != self.last_steering:
            pwm(self.bus,STEERING_CH,value);self.last_steering=value
    def command(self,steer,move,speed):
        # Write throttle neutral BEFORE changing steering when stopping.
        if not move:self._throttle(NEUTRAL_US)
        self._steering(steer if move else STEERING_CENTER_US)
        if move:self._throttle(speed)
    def close(self):
        if self.bus is not None:
            try:self._throttle(NEUTRAL_US);self._steering(STEERING_CENTER_US)
            except Exception:pass
        if self.mux is not None:self.mux.stop()
        if self.gpio is not None:
            try:self.gpio.cleanup()
            except Exception:pass
        if self.bus is not None:self.bus.close()


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--model',default=str(Path(__file__).with_name('lane_model.onnx')))
    p.add_argument('--video',help='offline MP4 (never outputs PWM)')
    p.add_argument('--camera-topic',default=CAMERA_TOPIC)
    p.add_argument('--target-x',type=float,default=.50,help='initial white lane location, fraction 0..1')
    p.add_argument('--show',action='store_true')
    p.add_argument('--save-video',default='')
    p.add_argument('--drive',action='store_true')
    p.add_argument('--allow-drive',action='store_true',help='explicit hardware test acknowledgment')
    p.add_argument('--speed-us',type=int,default=SPEED_US)
    p.add_argument('--backend',choices=['auto','tensorrt','cuda','cpu'],default='auto')
    p.add_argument('--trt-cache',default='~/.cache/ros2_lane_follow/trt',
                   help='TensorRT engine/timing cache directory')
    p.add_argument('--trt-fp16',action=argparse.BooleanOptionalAction,default=True,
                   help='enable TensorRT FP16 optimization (default: enabled)')
    p.add_argument('--crop-top',type=float,default=.10,
                   help='top fraction excluded before inference (default .10); validate against original video')
    p.add_argument('--max-frame-age-ms',type=float,default=150.)
    p.add_argument('--debug-every',type=int,default=2,
                   help='render debug windows every N processed frames')
    p.add_argument('--watchdog-sec',type=float,default=.35,
                   help='stop motor if no successful fresh processed command by this age')
    p.add_argument('--view-scale',type=float,default=1.0,
                   help='scale diagnostic windows only; does not alter inference resolution')
    a,rosargs=p.parse_known_args()
    if not 0<=a.crop_top<=.30:p.error('--crop-top must be 0..0.30')
    if not .3<=a.view_scale<=2.0:p.error('--view-scale must be .3..2.0')
    if a.debug_every<1:p.error('--debug-every must be >=1')
    if not 70<=a.max_frame_age_ms<=1000:p.error('--max-frame-age-ms must be 70..1000')
    if not .15<=a.watchdog_sec<=1.0:p.error('--watchdog-sec must be 0.15..1.0')
    if not 0.1<=a.target_x<=0.9:p.error('--target-x must be between 0.1 and 0.9')
    if not NEUTRAL_US<=a.speed_us<=MAX_SPEED_US:p.error('speed pulse exceeds original tested limit')
    if a.drive and not a.allow_drive:p.error('for hardware actuation, add --allow-drive after wheels-up verification')
    if a.drive and a.video:p.error('--video may not be combined with --drive')
    segmenter=Segmenter(a.model,a.backend,a.crop_top,a.trt_cache,a.trt_fp16)
    print(f'backend={segmenter.backend} providers={segmenter.sess.get_providers()} '
          f'warmup={segmenter.warmup_ms:.1f}ms input={segmenter.input_width}x{segmenter.input_height}',
          flush=True)
    tracker=LaneTracker(a.target_x)
    state={'prev':0.,'writer':None,'count':0,'bev_shape':None,'bev_matrix':None}
    def process(frame,timestamp):
        t0=time.perf_counter();h,w=frame.shape[:2]
        wm,ym=segmenter.predict(frame)
        if state['bev_shape'] != (w,h):
            state['bev_shape']=(w,h);state['bev_matrix']=bev_matrix(w,h)
        mat=state['bev_matrix']
        bev=cv2.warpPerspective(frame,mat,(w,h))
        # Work on segmentation masks, not interpolated RGB.
        wb=cv2.warpPerspective(wm,mat,(w,h),flags=cv2.INTER_NEAREST)
        lane,fits,boxes=tracker.update(wb,timestamp)
        if lane is None:
            pulse=STEERING_CENTER_US;state['prev']=0.;status='LOST / NO DRIVE'; moving=False
        else:
            pulse,state['prev']=pulse_for_lane(lane,w,state['prev'])
            status='DIRECT WHITE';moving=True
        ms=(time.perf_counter()-t0)*1000
        return pulse,moving,ms,status,(frame,wm,ym,bev,lane,fits,boxes)

    def render_debug(data,status,pulse,ms,drive_status,age_ms):
        if not (a.show or a.save_video):return
        state['count']+=1
        if a.show and state['count']%a.debug_every!=0 and not a.save_video:return
        frame,wm,ym,bev,lane,fits,boxes=data
        h,w=frame.shape[:2]
        decision=decision_view(bev,lane,fits,boxes,status,pulse,ms)
        cv2.rectangle(decision,(0,62),(w,115),(0,0,0),-1)
        cv2.putText(decision,f'DRIVE={drive_status}',(5,83),cv2.FONT_HERSHEY_SIMPLEX,.48,(255,255,255),1)
        cv2.putText(decision,f'FRAME AGE={age_ms:.1f}ms INFER={segmenter.last_infer_ms:.1f}ms',
                    (5,105),cv2.FONT_HERSHEY_SIMPLEX,.38,(255,255,255),1)
        if a.show:
            sz=(max(1,int(w*a.view_scale)),max(1,int(h*a.view_scale)))
            cv2.imshow('1 Camera',cv2.resize(frame,sz))
            cv2.imshow('2 White Mask',cv2.resize(wm,sz,interpolation=cv2.INTER_NEAREST))
            cv2.imshow('3 Yellow Mask',cv2.resize(ym,sz,interpolation=cv2.INTER_NEAREST))
            cv2.imshow('4 Lane Decision',cv2.resize(decision,sz))
            cv2.waitKey(1)
        if a.save_video:
            sz=(384,312)
            def tile(im,label):
                t=cv2.resize(im,sz,interpolation=cv2.INTER_NEAREST if im.ndim==2 else cv2.INTER_LINEAR)
                if t.ndim==2:t=cv2.cvtColor(t,cv2.COLOR_GRAY2BGR)
                cv2.rectangle(t,(0,0),(384,25),(0,0,0),-1)
                cv2.putText(t,label,(7,18),cv2.FONT_HERSHEY_SIMPLEX,.50,(255,255,255),1)
                return t
            tiled=np.vstack((np.hstack((tile(frame,'1 Camera'),tile(wm,'2 White Mask'))),
                             np.hstack((tile(ym,'3 Yellow Mask'),tile(decision,'4 Lane Decision')))))
            if state['writer'] is None:
                state['writer']=cv2.VideoWriter(a.save_video,cv2.VideoWriter_fourcc(*'mp4v'),24,(768,624))
            state['writer'].write(tiled)
    if a.video:
        cap=cv2.VideoCapture(a.video)
        if not cap.isOpened():raise RuntimeError('Cannot open video')
        count=0
        try:
            while True:
                ok,frame=cap.read()
                if not ok:break
                pulse,moving,ms,status,dbg=process(frame,time.monotonic());count+=1
                render_debug(dbg,status,pulse,ms,'VIDEO-ONLY',0.)
                if count%100==0:print(f'frame={count} {status} process_ms={ms:.1f}',flush=True)
        finally:cap.release()
    else:
        import rclpy
        from cv_bridge import CvBridge
        from rclpy.node import Node
        from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
        from sensor_msgs.msg import Image
        import threading
        hw=Hardware() if a.drive else None
        if hw:time.sleep(3)
        rclpy.init(args=rosargs)
        class Follower(Node):
            def __init__(self):
                super().__init__('camera_lane_follow_dl_fast')
                self.bridge=CvBridge()
                self.lock=threading.Lock()
                self.latest=None
                self.received=time.monotonic()
                self.last_command=time.monotonic()
                self.last_good=0.
                self.stopped=True
                self.last_reason='STARTING'
                self.worker_running=True
                self.hw_lock=threading.Lock()
                self.worker=threading.Thread(target=self.loop,daemon=True)
                qos=QoSProfile(
                    depth=1,
                    reliability=ReliabilityPolicy.BEST_EFFORT,
                    history=HistoryPolicy.KEEP_LAST,
                )
                self.sub=self.create_subscription(Image,a.camera_topic,self.cb,qos)
                self.timer=self.create_timer(.05,self.watchdog)
                self.worker.start()
                self.get_logger().info('GPU DL follower; '+('PWM ENABLED' if hw else 'DISPLAY ONLY'))
            def cb(self,msg):
                # Do not infer in subscription callback. Keep only freshest frame.
                with self.lock:
                    self.latest=(msg,time.monotonic())
                    self.received=time.monotonic()
            def loop(self):
                while self.worker_running:
                    with self.lock:
                        item=self.latest
                        self.latest=None
                    if item is None:
                        time.sleep(.002)
                        continue
                    msg,rx=item
                    # Drop messages already stale on receipt/queue; Jetson monotonic age.
                    if (time.monotonic()-rx)*1000>a.max_frame_age_ms:
                        self.safe_stop('FRAME_STALE_BEFORE_INFER')
                        continue
                    try:
                        frame=self.bridge.imgmsg_to_cv2(msg,desired_encoding='bgr8')
                        pulse,moving,ms,status,dbg=process(frame,time.monotonic())
                        with self.hw_lock:
                            # Shutdown may occur while GPU inference is in flight.
                            if not self.worker_running:
                                return
                            age=(time.monotonic()-rx)*1000.
                            reason=('LANE_LOST' if not moving else
                                    'FRAME_STALE_AFTER_INFER' if age>a.max_frame_age_ms else 'TRACKING')
                            can_drive=moving and age<=a.max_frame_age_ms
                            if hw:
                                hw.command(pulse if can_drive else STEERING_CENTER_US,
                                           can_drive,a.speed_us)
                            self.last_command=time.monotonic()
                            self.stopped=not can_drive
                        # Render AFTER applying PWM; GUI must never delay motor control.
                        render_debug(dbg,status,pulse,ms,
                                     ('DRIVE' if hw else 'DRY RUN') if can_drive else 'STOP:'+reason,age)
                        if not can_drive and self.last_reason!=reason:
                            self.get_logger().warning(
                                f'STOP_REASON={reason} lane={status} age={age:.1f}ms inference={segmenter.last_infer_ms:.1f}ms'
                            )
                        self.last_reason=reason
                        if ms>60:
                            self.get_logger().warning(
                                f'infer={segmenter.last_infer_ms:.1f}ms total={ms:.1f}ms queue_age={age:.1f}ms'
                            )
                    except Exception as exc:
                        with self.hw_lock:
                            if not self.worker_running:
                                return
                            if hw:
                                hw.command(STEERING_CENTER_US,False,a.speed_us)
                        self.stopped=True
                        self.last_reason='INFERENCE_ERROR'
                        self.get_logger().error(f'inference failure: {exc}')
            def safe_stop(self,reason):
                with self.hw_lock:
                    if not self.worker_running:
                        return
                    if hw:
                        hw.command(STEERING_CENTER_US,False,a.speed_us)
                self.stopped=True
                if reason != self.last_reason:
                    self.get_logger().warning('STOP_REASON='+reason)
                self.last_reason=reason
            def watchdog(self):
                # Even when fresh frames arrive, a stalled inference must not
                # leave the throttle command latched indefinitely.
                if time.monotonic()-self.last_command>a.watchdog_sec:
                    self.safe_stop('COMMAND_TIMEOUT')
                elif time.monotonic()-self.received>a.watchdog_sec:
                    self.safe_stop('CAMERA_TIMEOUT')
            def destroy_node(self):
                with self.hw_lock:
                    self.worker_running=False
                    if hw:
                        hw.command(STEERING_CENTER_US,False,a.speed_us)
                self.worker.join(timeout=1.0)
                return super().destroy_node()
        node=None
        try:
            node=Follower();rclpy.spin(node)
        finally:
            if node:node.destroy_node()
            if hw:hw.close()
            if rclpy.ok():rclpy.shutdown()
    if state['writer'] is not None:state['writer'].release()
    cv2.destroyAllWindows()

if __name__=='__main__':main()
