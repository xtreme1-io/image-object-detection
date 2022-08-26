import argparse

import flask
import base64
import re
from flask import Flask, request
from flask_cors import CORS

import numpy as np
import cv2
import json
import tornado
import tornado.wsgi
import tornado.httpserver
import torch
import torch.backends.cudnn as cudnn

from models.experimental import attempt_load
from utils.general import (check_img_size, non_max_suppression, 
                           scale_coords, obtain_image_from_url)
from utils.torch_utils import select_device, time_synchronized
from utils.metrics import calculate_iou


MAX_IMAGES_PER_BATCH = 100
SUPER_CATEGORY_SEP = u"_@_"
app = Flask(__name__)
CORS(app, supports_credentials=True)


def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, auto_size=32):
    # Resize image to a 32-pixel-multiple rectangle https://github.com/ultralytics/yolov3/issues/232
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
        dw, dh = np.mod(dw, auto_size), np.mod(dh, auto_size)  # wh padding
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


CORS(app, resources=r'/*')
@app.route('/hello', methods=['GET'])
def greeting():
    return json.dumps({u"code":u"OK", u"message":u"", u"data":[u"hi, nice to meet you !"]})


def parse_request_params(request_data):
    try:
        request = json.loads(request_data.decode('utf-8'))
    except ValueError as err:
        response_str = json.dumps({u"code":u"ERROR", u"message":u"json decode error", u"data":[]})
        return response_str
    
    if u"datas" not in request or u"params" not in request \
        or not isinstance(request[u"datas"], (list, tuple)):
        response_str = json.dumps({u"code":u"ERROR", u"message":u"invalid parameters", u"data":[]})
        return response_str
    
    if len(request[u"datas"]) > MAX_IMAGES_PER_BATCH:
        response_str = json.dumps({u"code":u"ERROR", u"message":u"too many images in a single request", u"data":[]})
        return response_str
     
    param_info = []   
    for record in request[u"datas"]:
        if u"image_id" not in record \
            or u"img_url" not in record or not record[u"img_url"].strip().startswith(u"http"):
            response_str = json.dumps({u"code":u"ERROR", u"message":u"invalid parameters", u"data":[]})
            return response_str
        
        image = obtain_image_from_url(record[u"img_url"].strip())
        if image is None:
            response_str = json.dumps({u"code":u"ERROR", u"message":u"image unavailable", u"data":[]})
            return response_str
        
        rec = {u"height":image.shape[0], u"width":image.shape[1], 
                      u"image_id":record[u"image_id"].strip(), u"image":image}
        if u"region" in record:
            rec[u"region"] = record[u"region"]
        
        param_info.append(rec)
    
    return param_info


CORS(app, resources=r'/*')
@app.route('/basic_predict', methods=['POST'])
def basic_predict():
    request_data = flask.request.get_data()
    param_info = parse_request_params(request_data)
    if isinstance(param_info, str):
        return param_info
    
    det_res = []
    for record in param_info:
        org_img = record[u"image"]
        img = letterbox(org_img, new_shape=opt.img_size, auto_size=64)[0]#resize and padding
    
        # Convert
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416（在1280模式下，为[3, 768, 1280]）
        img = np.ascontiguousarray(img)
    
        #type convert and normalize
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
    
        # Inference
        pred = model(img, augment=False)[0]
        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=None, agnostic=False)
    
        print (u"file :", record[u"image_id"], u" image shape :", org_img.shape)
        rec_res = []
    
        # Process detections
        for i, det in enumerate(pred):  # detections per image
    
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], org_img.shape).round()
                
                for *xyxy, conf, cls in reversed(det):
                    xyxy_list = [e.item() for e in xyxy]
                    cn_items = names[int(cls)].split(SUPER_CATEGORY_SEP)
                    rec_res.append({"clsid": int(cls)+1, 
                                    "class": names[int(cls)].split(SUPER_CATEGORY_SEP)[0], 
                                    "super_class": re.sub(u"其他障碍物", u"other-barrier", cn_items[1] if len(cn_items)==2 else u""), 
                                    "score": float(conf.item()), 
                                    "left": max(0., xyxy_list[0]/record[u"width"]), 
                                    "right": min(1., xyxy_list[2]/record[u"width"]), 
                                    "top": max(0., xyxy_list[1]/record[u"height"]), 
                                    "bottom": min(1., xyxy_list[3]/record[u"height"]),
                                    "leftTopX":xyxy_list[0],
                                    "leftTopY":xyxy_list[1],
                                    "rightBottomX":xyxy_list[2],
                                    "rightBottomY":xyxy_list[3]
                                    })
                
        det_res.append({u"det_res":rec_res, u"image_id":record[u"image_id"]})
    return json.dumps({u"code":u"OK", u"message":u"", u"data":det_res})


CORS(app, resources=r'/*')
@app.route('/region_predict', methods=['POST'])
def region_predict():
    request_data = flask.request.get_data()
    param_info = parse_request_params(request_data)
    if isinstance(param_info, str):
        return param_info
    
    det_res = []
    for record in param_info:
        org_img = record[u"image"]
        img_region = record[u"region"]
        img = letterbox(org_img, new_shape=opt.img_size, auto_size=64)[0]#resize and padding
    
        # Convert
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416（在1280模式下，为[3, 768, 1280]）
        img = np.ascontiguousarray(img)
    
        #type convert and normalize
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
    
        # Inference
        pred = model(img, augment=False)[0]
        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=None, agnostic=False)
    
        print (u"file :", record[u"image_id"], u" image shape :", org_img.shape)
        rec_res = []
    
        # Process detections
        for i, det in enumerate(pred):  # detections per image
    
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], org_img.shape).round()
                
                for *xyxy, conf, cls in reversed(det):
                    xyxy_list = [e.item() for e in xyxy]
                    cn_items = names[int(cls)].split(SUPER_CATEGORY_SEP)
                    rec_res.append({"clsid": int(cls)+1, 
                                    "class": names[int(cls)].split(SUPER_CATEGORY_SEP)[0], 
                                    "super_class": re.sub(u"其他障碍物", u"other-barrier", cn_items[1] if len(cn_items)==2 else u""),
                                    "score": float(conf.item()), 
                                    "left": max(0., xyxy_list[0]/record[u"width"]), 
                                    "right": min(1., xyxy_list[2]/record[u"width"]), 
                                    "top": max(0., xyxy_list[1]/record[u"height"]), 
                                    "bottom": min(1., xyxy_list[3]/record[u"height"]),
                                    "leftTopX":xyxy_list[0],
                                    "leftTopY":xyxy_list[1],
                                    "rightBottomX":xyxy_list[2],
                                    "rightBottomY":xyxy_list[3]
                                    })
            
        max_iou = 0
        max_iou_id = -1    
        for idx, d in enumerate(rec_res):
            iou = calculate_iou(d[u"leftTopX"], d[u"leftTopY"], 
                                d[u"rightBottomX"], d[u"rightBottomY"], 
                                img_region[u"leftTopX"], img_region[u"leftTopY"], 
                                img_region[u"rightBottomX"], img_region[u"rightBottomY"])
            if max_iou < iou:
                max_iou = iou
                max_iou_id = idx
            
        salient_obj = rec_res[max_iou_id]
        salient_obj[u"keypointX"] = max(min((salient_obj[u"leftTopX"]+salient_obj[u"rightBottomX"])/2, 
                                            img_region[u"rightBottomX"]), 
                                        img_region[u"leftTopX"])
        salient_obj[u"keypointY"] = max(min((salient_obj[u"leftTopY"]+salient_obj[u"rightBottomY"])/2, 
                                            img_region[u"rightBottomY"]), 
                                        img_region[u"leftTopY"])
                
        det_res.append({u"det_res":[salient_obj], u"image_id":record[u"image_id"]})
    return json.dumps({u"code":u"OK", u"message":u"", u"data":det_res})


if __name__ == u"__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolor-p6.pt', help='model.pt path(s)')
    parser.add_argument('--img-size', type=int, default=1280, help='inference size (pixels)')
    parser.add_argument('--port', type=int, default=18883, help='service port')
    parser.add_argument('--conf-thres', type=float, default=0.5, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    opt = parser.parse_args()
    print(opt)
    
    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(opt.weights, map_location=device)  # load FP32 model
    imgsz = check_img_size(opt.img_size, s=model.stride.max())  # check img_size
    
    if half:
        model.half()  # to FP16

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    print (u"class names :", json.dumps({idx+1:n for idx, n in enumerate(names)}, ensure_ascii=False))

    img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
    _ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once
    
    server = tornado.httpserver.HTTPServer(tornado.wsgi.WSGIContainer(app), 
                                           max_buffer_size=10485760, 
                                           body_timeout=1000.)
    server.bind(opt.port)
    server.start(1)
    print('Tornado server starting on port {}'.format(opt.port), flush=True)
    tornado.ioloop.IOLoop.current().start()


