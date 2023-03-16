import json
import os
import codecs


MINOVERLAP = 0.5 # default value (defined in the PASCAL VOC2012 challenge)
"""
 Create a ".temp_files/" and "output/" directory
"""
TEMP_FILES_PATH = ".temp_files"
if not os.path.exists(TEMP_FILES_PATH): # if it doesn't exist already
    os.makedirs(TEMP_FILES_PATH)
    
    
"""
 Calculate the AP given the recall and precision array
    1st) We compute a version of the measured precision/recall curve with
         precision monotonically decreasing
    2nd) We compute the AP as the area under this curve by numerical integration.
"""
def voc_ap(rec, prec):
    """
    --- Official matlab code VOC2012---
    mrec=[0 ; rec ; 1];
    mpre=[0 ; prec ; 0];
    for i=numel(mpre)-1:-1:1
            mpre(i)=max(mpre(i),mpre(i+1));
    end
    i=find(mrec(2:end)~=mrec(1:end-1))+1;
    ap=sum((mrec(i)-mrec(i-1)).*mpre(i));
    """
    rec.insert(0, 0.0) # insert 0.0 at begining of list
    rec.append(1.0) # insert 1.0 at end of list
    mrec = rec[:]
    prec.insert(0, 0.0) # insert 0.0 at begining of list
    prec.append(0.0) # insert 0.0 at end of list
    mpre = prec[:]
    """
     This part makes the precision monotonically decreasing
        (goes from the end to the beginning)
        matlab: for i=numel(mpre)-1:-1:1
                    mpre(i)=max(mpre(i),mpre(i+1));
    """
    for i in range(len(mpre)-2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i+1])
    """
     This part creates a list of indexes where the recall changes
        matlab: i=find(mrec(2:end)~=mrec(1:end-1))+1;
    """
    i_list = []
    for i in range(1, len(mrec)):
        if mrec[i] != mrec[i-1]:
            i_list.append(i) # if it was matlab would be i + 1
    """
     The Average Precision (AP) is the area under the curve
        (numerical integration)
        matlab: ap=sum((mrec(i)-mrec(i-1)).*mpre(i));
    """
    ap = 0.0
    for i in i_list:
        ap += ((mrec[i]-mrec[i-1])*mpre[i])
    return ap, mrec, mpre

    
def load_gt(gt_lines):
    gt_counter_per_class = {}
    file_ids = []
    
    for line in gt_lines:
        rec = json.loads(line.strip())
        file_id = str(rec[u"id"])
        bounding_boxes = []
        for item in rec[u"objects"]:
            class_name = u"unknown" if u"className" not in item else item[u"className"]
            left, top, right, bottom = item[u"leftTopX"], item[u"leftTopY"], item[u"rightBottomX"], item[u"rightBottomY"]
            bbox = str(left) + " " + str(top) + " " + str(right) + " " + str(bottom)
            bounding_boxes.append({"class_name":class_name, "bbox":bbox, "used":False})
            # count that object
            if class_name in gt_counter_per_class:
                gt_counter_per_class[class_name] += 1
            else:
                # if class didn't exist yet
                gt_counter_per_class[class_name] = 1
                
        file_ids.append(file_id)
        new_temp_file = TEMP_FILES_PATH + "/" + file_id + "_ground_truth.json"
        with codecs.open(new_temp_file, 'wb', u"utf-8") as outfile:
            json.dump(bounding_boxes, outfile)
        
    return file_ids, gt_counter_per_class

       
def load_pred(gt_classes, pred_lines):
    
    for class_name in gt_classes:
        bounding_boxes = []
        for rec in pred_lines:
            file_id = str(rec[u"id"])
            for item in rec[u"objects"]:
                tmp_class_name = u"unknown" if u"className" not in item else item[u"className"]
                confidence, left, top, right, bottom =  item[u"confidence"], \
                                                        item[u"leftTopX"], item[u"leftTopY"], \
                                                        item[u"rightBottomX"], item[u"rightBottomY"]
                if tmp_class_name == class_name:
                    bbox = str(left) + " " + str(top) + " " + str(right) + " " + str(bottom)
                    bounding_boxes.append({"confidence":confidence, "file_id":file_id, "bbox":bbox})
            
        # sort detection-results by decreasing confidence
        bounding_boxes.sort(key=lambda x:float(x['confidence']), reverse=True)
        with open(TEMP_FILES_PATH + "/" + class_name + "_dr.json", 'w') as outfile:
            json.dump(bounding_boxes, outfile)

    return            


"""
 Calculate the AP for each class
"""
def cal_mAP(n_classes, gt_classes, gt_counter_per_class):
    sum_AP = 0.0
    count_true_positives = {}
    for class_name in gt_classes:
        count_true_positives[class_name] = 0
        """
         Load detection-results of that class
        """
        dr_file = TEMP_FILES_PATH + "/" + class_name + "_dr.json"
        dr_data = json.load(open(dr_file))

        """
         Assign detection-results to ground-truth objects
        """
        nd = len(dr_data)
        tp = [0] * nd # creates an array of zeros of size nd
        fp = [0] * nd
        for idx, detection in enumerate(dr_data):
            file_id = detection["file_id"]
            # assign detection-results to ground truth object if any
            # open ground-truth with that file_id
            gt_file = TEMP_FILES_PATH + "/" + file_id + "_ground_truth.json"
            ground_truth_data = json.load(open(gt_file))
            
            ovmax = -1
            gt_match = -1
            # load detected object bounding-box
            bb = [ float(x) for x in detection["bbox"].split() ]
            for obj in ground_truth_data:
                # look for a class_name match
                if obj["class_name"] == class_name:
                    bbgt = [ float(x) for x in obj["bbox"].split() ]
                    bi = [max(bb[0],bbgt[0]), max(bb[1],bbgt[1]), min(bb[2],bbgt[2]), min(bb[3],bbgt[3])]
                    iw = bi[2] - bi[0] + 1
                    ih = bi[3] - bi[1] + 1
                    if iw > 0 and ih > 0:
                        # compute overlap (IoU) = area of intersection / area of union
                        ua = (bb[2] - bb[0] + 1) * (bb[3] - bb[1] + 1) + (bbgt[2] - bbgt[0]
                                        + 1) * (bbgt[3] - bbgt[1] + 1) - iw * ih
                        ov = iw * ih / ua
                        if ov > ovmax:
                            ovmax = ov
                            gt_match = obj

            if ovmax >= MINOVERLAP:
                if "difficult" not in gt_match:
                        if not bool(gt_match["used"]):
                            # true positive
                            tp[idx] = 1
                            gt_match["used"] = True
                            count_true_positives[class_name] += 1
                            # update the ".json" file
                            with open(gt_file, 'w') as f:
                                    f.write(json.dumps(ground_truth_data))
                        else:
                            # false positive (multiple detection)
                            fp[idx] = 1
            else:
                # false positive
                fp[idx] = 1

        # compute precision/recall
        cumsum = 0
        for idx, val in enumerate(fp):
            fp[idx] += cumsum
            cumsum += val
        cumsum = 0
        for idx, val in enumerate(tp):
            tp[idx] += cumsum
            cumsum += val

        rec = tp[:]
        for idx, val in enumerate(tp):
            rec[idx] = float(tp[idx]) / gt_counter_per_class[class_name]
#         print(rec)
        prec = tp[:]
        for idx, val in enumerate(tp):
            prec[idx] = float(tp[idx]) / (fp[idx] + tp[idx])
        #print(prec)

        ap, mrec, mprec = voc_ap(rec[:], prec[:])
        sum_AP += ap
        text = "{0:.2f}%".format(ap*100) + " = " + class_name + " AP " #class_name + " AP = {0:.2f}%".format(ap*100)
        
        print(text)

    mAP = sum_AP / n_classes
    text = "mAP = {0:.2f}%".format(mAP*100)
    print(text)
    return mAP


if __name__==u"__main__":
    with codecs.open(r"./groundTruth.json", u"rb", u"utf-8") as f:
        gt_lines = f.readlines()
    file_ids, gt_counter_per_class = load_gt(gt_lines)
    
    gt_classes = list(gt_counter_per_class.keys())
    gt_classes = sorted(gt_classes)
    n_classes = len(gt_classes)

    with codecs.open(r"./modelRun.json", u"rb", u"utf-8") as f:
        pred_lines = [json.loads(line.strip()) for line in f]
    load_pred(gt_classes, pred_lines)
    mAP = cal_mAP(n_classes, gt_classes, gt_counter_per_class)




