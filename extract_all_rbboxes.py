# -*- coding:utf8 -*-

from mmdet.apis import init_detector, inference_detector, show_result, draw_poly_detections
import mmcv
from mmcv import Config
from mmdet.datasets import get_dataset
import cv2
import os
import numpy as np
import DOTA_devkit.polyiou as polyiou
import math
import random
import torch

def py_cpu_nms_poly_fast_np(dets, thresh):
    obbs = dets[:, 0:-1]
    x1 = np.min(obbs[:, 0::2], axis=1)
    y1 = np.min(obbs[:, 1::2], axis=1)
    x2 = np.max(obbs[:, 0::2], axis=1)
    y2 = np.max(obbs[:, 1::2], axis=1)
    scores = dets[:, 8]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)

    polys = []
    for i in range(len(dets)):
        tm_polygon = polyiou.VectorDouble([dets[i][0], dets[i][1],
                                            dets[i][2], dets[i][3],
                                            dets[i][4], dets[i][5],
                                            dets[i][6], dets[i][7]])
        polys.append(tm_polygon)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        ovr = []
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        hbb_inter = w * h
        hbb_ovr = hbb_inter / (areas[i] + areas[order[1:]] - hbb_inter)
        h_inds = np.where(hbb_ovr > 0)[0]
        tmp_order = order[h_inds + 1]
        for j in range(tmp_order.size):
            iou = polyiou.iou_poly(polys[i], polys[tmp_order[j]])
            hbb_ovr[h_inds[j]] = iou

        try:
            if math.isnan(ovr[0]):
                pdb.set_trace()
        except:
            pass
        inds = np.where(hbb_ovr <= thresh)[0]
        order = order[inds + 1]
    return keep

class DetectorModel():
    def __init__(self,
                 config_file,
                 checkpoint_file):
        # init RoITransformer
        self.config_file = config_file
        self.checkpoint_file = checkpoint_file
        self.cfg = Config.fromfile(self.config_file)
        self.data_test = self.cfg.data['test']
        self.dataset = get_dataset(self.data_test)
        self.classnames = self.dataset.CLASSES
        self.model = init_detector(config_file, checkpoint_file, device='cuda:0')

    def inference_single(self, imagname, img_size):
        img = mmcv.imread(imagname)
        height, width, channel = img.shape
        assert height==img_size[0] and width==img_size[1]
        # TODO: check the corner case
        # import pdb; pdb.set_trace()
        total_detections = [np.zeros((0, 9)) for _ in range(len(self.classnames))]
        total_detections = inference_detector(self.model, img)
        
        return total_detections

    def inference_single_vis(self, srcpath, dstpath, img_size):
        detections = self.inference_single(srcpath, img_size)
        img = draw_poly_detections(srcpath, detections.copy(), self.classnames, scale=1, threshold=0.3)
        cv2.imwrite(dstpath, img)
        return detections

if __name__ == '__main__':
    assert False # 防止误执行

    seed = 13
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed+2)
    torch.cuda.manual_seed_all(seed+3)

    roitransformer = DetectorModel(r'configs/DOTA/faster_rcnn_RoITrans_r50_fpn_1x_dota.py',
                  r'work_dirs/faster_rcnn_RoITrans_r50_fpn_1x_dota/epoch_12.pth')
    
    input_dir = 'data/RSVG/images'
    det_results_path='data/RSVG/det_obb_rsvg.txt'
    output_dir = 'demo/RSVG'
    f=open(det_results_path, 'w')
    #for filename in open('diff_all_img.txt'):
    for filename in os.listdir(input_dir):
        filename = filename.strip()
        print(filename)
        in_path = os.path.join(input_dir, filename)
        detections = roitransformer.inference_single_vis(in_path, os.path.join(output_dir, filename), (1024, 1024))
        class_names = roitransformer.classnames
        for j, name in enumerate(class_names):
            dets = detections[j]
            if dets.shape[0] == 0:
                continue
            for det in dets:
                f.write(filename + ' ' + ' '.join([str(one) for one in det]) + ' ' + name + "\n")
    f.close()
