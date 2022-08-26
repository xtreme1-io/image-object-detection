#-*- encoding:utf-8 -*-

import base64
import json
import requests
import traceback
import os, cv2


def load_raw_b64(fn):
    with open(fn, 'rb') as f:
        raw = f.read()
    res = base64.b64encode(raw).decode('ascii')
    return res


def post_data(request_url, data):
    data_str = json.dumps(data).encode('ascii')
    headers = {'Content-Type': 'application/json'}
    resp = None
    try:
        resp = requests.post(request_url, data=data_str, headers=headers)
    except Exception as err:
        traceback.print_exc()
    return resp


def model_call(data_dic):
    request_url = u"http://127.0.0.1:18881/basic_predict"
    data = data_dic
    resp = post_data(request_url, data)
    return resp


def test_images():
    output_dir = u"./inference/output/"

    req_params = {
                    u"datas":[
                                {
                                    u"img_url":u"https://inews.gtimg.com/newsapp_bt/0/14253645180/1000", 
                                    u"image_id":u"14253645180"
                                },
                                {
                                    u"img_url":u"https://inews.gtimg.com/newsapp_bt/0/14253644835/1000",
                                    u"image_id":u"14253644835"
                                }
                            ], 
                    u"params":{}
                  }
    
    def obtain_image_from_url(img_url):
        import numpy as np
        html = requests.get(img_url)
        img = cv2.imdecode(np.fromstring(html.content, np.uint8), 1) 
        return img#[H,W,C]
    
    resp = model_call(req_params)
    print (resp.text)
    print (json.dumps(json.loads(resp.text), ensure_ascii=False, indent=2))
    
    img_urls = {rec[u"image_id"]:rec[u"img_url"] for rec in req_params[u"datas"]}
    for rec in json.loads(resp.text)[u"data"]:
        url = img_urls[rec[u"image_id"]]
        img = obtain_image_from_url(url)
        h,w = img.shape[:2]
        boxes = rec[u"det_res"]
        for res in boxes:
            c1 = (int(res[u"left"]*w), int(res[u"top"]*h))
            c2 = (int(res[u"right"]*w), int(res[u"bottom"]*h))
            img = cv2.rectangle(img, c1, c2, [255,255,255], thickness=1, lineType=cv2.LINE_AA)
            img = cv2.putText(img, res[u"class"], (c1[0], c1[1] - 2), 0, 1., [225, 255, 255], thickness=1, lineType=cv2.LINE_AA)
        cv2.imwrite(os.path.join(output_dir, rec[u"image_id"]+u".jpg"), img)
    
    print(u"test finish")


if __name__ == '__main__':
    test_images()
