import os
import json
import codecs


def get_categories(cat_file_path):
    with codecs.open(cat_file_path, u"rb", u"utf-8") as f:
        gt_coco = json.load(f)
        if u"categories" in gt_coco:#提取字典中的列表
            thing_classes   = [item[u"name"] for item in gt_coco[u"categories"]]
            supercategories   = [item[u"supercategory"] for item in gt_coco[u"categories"]]
            thing_ids       = [item[u"id"] for item in gt_coco[u"categories"]]
        else:#直接从类别列表读取
            thing_classes   = [item[u"name"] for item in gt_coco]
            supercategories   = [item[u"supercategory"] for item in gt_coco]
            thing_ids       = [item[u"id"] for item in gt_coco]
        
    return thing_classes, supercategories, thing_ids


def get_basic_obj_det_dicts(img_dir, ana_file_path):
    
    assert os.path.exists(img_dir) and os.path.isdir(img_dir) \
            and os.path.exists(ana_file_path), \
            u"lack of necessary file !!!"
    
    with codecs.open(ana_file_path, u"rb", u"utf-8") as f:
        gt_coco = json.load(f)
    
    img_info = {}#收集该子集中的所有imgID和文件名
    for img in gt_coco[u"images"]:
        if u"folder" not in img:
            img[u"folder"] = u""
        if len(img[u"folder"])>0 and img[u"folder"][0] == u"/":
            img[u"folder"] = img[u"folder"][1:]
            
        record = {u"image_id":img[u"id"], 
                  u"file_name":os.path.join(img_dir, img[u"folder"], img[u"file_name"]),
                  u"height":img[u"height"],
                  u"width":img[u"width"],
                  u"annotations":[]}
        img_info[img[u"id"]] = record
            
    for ana in gt_coco[u"annotations"]:
        
        if ana[u"image_id"] not in img_info:
            continue
        
        if isinstance(ana[u"segmentation"], list):
            seg = ana[u"segmentation"][0]
            bbox = [seg[0], seg[1], seg[4], seg[5]]
        elif isinstance(ana[u"segmentation"], dict):
            bbox = ana[u"bbox"]
            seg = ana[u"segmentation"]
            ana[u"segmentation"] = {u"size":seg[u"size"], u"counts":bytes(seg[u"counts"], encoding='utf-8')}
            
        obj = {
            u"bbox": bbox, #左上角、右下角的点坐标
            u"bbox_mode": 0,
            u"segmentation": ana[u"segmentation"],
            u"category_id": ana[u"category_id"] - 1,
        }
        if u"area" in ana:
            obj[u"area"] = ana[u"area"]
        img_info[ana[u"image_id"]]["annotations"].append(obj)

    print (u"number of images in dataset :", len(img_info))
    return [img_info[key] for key in img_info if len(img_info[key]["annotations"])>0]