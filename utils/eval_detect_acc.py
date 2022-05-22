# -*- coding:utf8 -*-

import torch
import json
import numpy as np

# 获取检测结果的水平框的召回率
# 判断每一个gt_bbox是否都能在detect bbox中找到
def bbox_iou(box1, box2, x1y1x2y2=True):
    """
    Returns the IoU of two bounding boxes
    """
    if x1y1x2y2:
        # Get the coordinates of bounding boxes
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]
    else:
        # Transform from center and width to exact coordinates
        b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
        b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
        b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
        b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2

    # get the coordinates of the intersection rectangle
    inter_rect_x1 = torch.max(b1_x1, b2_x1)
    inter_rect_y1 = torch.max(b1_y1, b2_y1)
    inter_rect_x2 = torch.min(b1_x2, b2_x2)
    inter_rect_y2 = torch.min(b1_y2, b2_y2)
    # Intersection area
    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1, 0) * torch.clamp(inter_rect_y2 - inter_rect_y1, 0)
    # Union Area
    b1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
    b2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)

    # print(box1, box1.shape)
    # print(box2, box2.shape)
    return inter_area / (b1_area + b2_area - inter_area + 1e-16)

def isSameBBox(one_gt_bbox, det_bboxes):
    det_num = len(det_bboxes)
    gt_bboxes = one_gt_bbox * det_num
    gt_bboxes = np.array(gt_bboxes, dtype=np.float32).reshape(-1, 4)
    gt_bboxes = torch.from_numpy(gt_bboxes).float().cuda()

    det_bboxes = np.array(det_bboxes, dtype=np.float32).reshape(-1, 4)
    det_bboxes = torch.from_numpy(det_bboxes).float().cuda()

    iou_list = bbox_iou(gt_bboxes, det_bboxes, x1y1x2y2=False)
    total_cnt = (iou_list>0.5).sum()
    total_cnt = total_cnt.cpu().numpy().tolist()
    return total_cnt>0

def eval_det_recall():
    gt_json_path = 'data/RSVG/instances_rsvg.json'
    with open(gt_json_path, 'r') as f:
        gt_json_data = json.load(f)
    
    gt_image_list = gt_json_data['images']
    gt_ann_list = gt_json_data['annotations']
    gt_categories = gt_json_data['categories']
    baseball_id = -1
    for one_cat in gt_categories:
        if one_cat['name'] == 'baseball-diamond':
            baseball_id = one_cat['id']

    gt_imgid2name = {}
    for one_img in gt_image_list:
        gt_imgid2name[one_img['id']] = one_img['file_name']
    
    det_json_path = 'data/RSVG/det_instances_rsvg.json'
    with open(det_json_path, 'r') as f:
        det_json_data = json.load(f)

    recall_cnt1, recall_cnt2 = 0, 0
    for one_ann in gt_ann_list:
        assert one_ann['image_id'] in gt_imgid2name
        imgname = gt_imgid2name[one_ann['image_id']]
        assert imgname in det_json_data
        det_bboxes = det_json_data[imgname]['det_bboxes'] #xywh

        gt_bbox = one_ann['bbox']
        ret_data = isSameBBox(gt_bbox, det_bboxes)
        if ret_data or one_ann['category_id'] == baseball_id:
            recall_cnt1 = recall_cnt1 + 1
        if ret_data:
            recall_cnt2 = recall_cnt2 + 1

    print('eval hbb: recall_count/all_count (all baseball are recalled)')
    recall = recall_cnt1*1.0/len(gt_ann_list)
    print(str(recall_cnt1) + "/" + str(len(gt_ann_list)) + " = " + str(recall) )
    
    print('eval hbb: recall_count/all_count')
    recall = recall_cnt2*1.0/len(gt_ann_list)
    print(str(recall_cnt2) + "/" + str(len(gt_ann_list)) + " = " + str(recall) )

if __name__ == '__main__':
    eval_det_recall()