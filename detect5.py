import argparse
import os
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from numpy import random

from models.experimental import attempt_load
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression,  scale_coords,  strip_optimizer, set_logging
from utils.plots import plot_one_box
from utils.torch_utils import select_device,  time_synchronized


def detect(save_img=False):

    weights, view_img, save_txt, imgsz =  opt.weights, opt.view_img, opt.save_txt, opt.img_size
    # Directories

    # Initialize
    set_logging()
    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    way=1
    if(weights=='Frepv5lite-e.pt'):
        way=0
    model = attempt_load(weights, map_location=device,imgsz=imgsz,way=way)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size
    if half:
        model.half()  # to FP16

    view_img = check_imshow()
    cudnn.benchmark = True  # set True to speed up constant image size inference
    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    t0 = time.time()
    color = [random.randint(0, 255) for _ in range(3)]


    video_capture = cv2.VideoCapture(0)
    while 1:
        ret, im0 = video_capture.read()
        if ret:
            img = letterbox(im0, opt.img_size, auto=True, stride=32)[0]
            t0 = time.time()
            img = torch.from_numpy(img).to(device)
            img = img.half() if half else img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            # Letterbox
            img = img.unsqueeze(0)
            # Stack
            img = np.stack(img.cpu(), 0)

            # Convert
            img = img[:, :, :, ::-1].transpose(0, 3, 1, 2)  # BGR to RGB, to bsx3x416x416
            img = np.ascontiguousarray(img)
            img = torch.from_numpy(img).to(device)
            #img = img.transpose(0, 2)

            # Inference
            t1 = time_synchronized()
            pred = model(img)[0]
            t2 = time_synchronized()

            print(f'Done. ({1 / (t2 - t1):.3f}fps)')
            # print(f'wDone. ({(t2 - t1)}s->{1 / (t2 - t1):.3f}fps)')
            # Apply NMS
            pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes,
                                       agnostic=opt.agnostic_nms)
            # Process detections
            for i, det in enumerate(pred):  # detections per image
                s, im0 =  '%g: ' % i, im0.copy()

                s += '%gx%g ' % img.shape[2:]  # print string
                if len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                    # xyxys = det[:, :4]
                    # Write results
                    for *xyxy, conf, cls in reversed(det):
                        if save_img or view_img:  # Add bbox to image
                            label = f'Hand {conf:.2f}'
                            plot_one_box(xyxy, im0, label=label, color=color, line_thickness=3)

                print(f'Done. ({(t2 - t1)}s->{1 / (t2 - t1):.3f}fps)')

                if view_img:
                    cv2.imshow('1', im0)
                    cv2.waitKey(1)  # 1 millisecond



    for path, img, im0s, vid_cap in dataset:
        t0 = time.time()
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = time_synchronized()
        pred = model(img, augment=opt.augment)[0]
        t2 = time_synchronized()

        #print(f'{s}Done. ({1 / (t2 - t1):.3f}fps)')
        #print(f'wDone. ({(t2 - t1)}s->{1 / (t2 - t1):.3f}fps)')
        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        # Process detections
        for i, det in enumerate(pred):  # detections per image
            p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count

            p = Path(p)  # to Path
            s += '%gx%g ' % img.shape[2:]  # print string
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                #xyxys = det[:, :4]
                # Write results
                for *xyxy, conf, cls in reversed(det):
                    if save_img or view_img:  # Add bbox to image
                        label = f'Hand {conf:.2f}'
                        plot_one_box(xyxy, im0, label=label, color=color, line_thickness=3)

            print(f'wDone. ({(t2 - t1)}s->{1 / (t2 - t1):.3f}fps)')

            if view_img:
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond



    print(f'Done. ({time.time() - t0:.3f}s)')


def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img, ratio, (dw, dh)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='weights/best_5s.pt', help='model.pt path(s)')
    #parser.add_argument('--weights', nargs='+', type=str, default='weights/v5lite-s.pt', help='model.pt path(s)')
    #parser.add_argument('--weights', nargs='+', type=str, default='weights/v5lite-c.pt', help='model.pt path(s)')
    #parser.add_argument('--weights', nargs='+', type=str, default='Frepv5lite-e.pt', help='model.pt path(s)')
    #parser.add_argument('--source', type=str, default='C:\Pro\pythonProject\Yolo-reim\hand128', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--source', type=str, default='0', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=320, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.6, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='IOU threshold for NMS')
    parser.add_argument('--device', default='cpu', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    #parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    opt = parser.parse_args()
    print(opt)
    check_requirements(exclude=('pycocotools', 'thop'))

    cpu_num = 1  # 这里设置成你想运行的CPU个数
    # cpu_num = os.cpu_count()   # 这里设置成你想运行的CPU个数
    os.environ['OMP_NUM_THREADS'] = str(cpu_num)
    os.environ['OPENBLAS_NUM_THREADS'] = str(cpu_num)
    os.environ['MKL_NUM_THREADS'] = str(cpu_num)
    os.environ['VECLIB_MAXIMUM_THREADS'] = str(cpu_num)
    os.environ['NUMEXPR_NUM_THREADS'] = str(cpu_num)
    torch.set_num_threads(cpu_num)

    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ['yolov5s.pt', 'yolov5m.pt', 'yolov5l.pt', 'yolov5x.pt']:
                detect()
                strip_optimizer(opt.weights)
        else:
            detect()
