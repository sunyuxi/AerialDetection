# -*- coding:utf8 -*-

from mmdet.apis import init_detector, inference_detector
import mmcv
from mmcv import Config
from mmdet.datasets import get_dataset
import cv2
import os
import numpy as np
import math
import argparse
import json
import random
import torch

def polygonToRotRectangle_batch(bbox_ori, with_module=True):
    """
    :param bbox: The polygon stored in format [x1, y1, x2, y2, x3, y3, x4, y4]
            shape [num_boxes, 8]
    :return: Rotated Rectangle in format [cx, cy, w, h, theta]
            shape [num_rot_recs, 5]
    """
    bbox_ori = np.array(bbox_ori,dtype=np.float32)
    
    bbox = np.reshape(bbox_ori,newshape=(-1, 2, 4),order='F')
    # angle = math.atan2(-(bbox[0,1]-bbox[0,0]),bbox[1,1]-bbox[1,0])
    angle = np.arctan2(-(bbox[:, 0,1]-bbox[:, 0,0]),bbox[:, 1,1]-bbox[:, 1,0])
    
    # center = [[0],[0]] ## shape [2, 1]
    # print('angle: ', angle)
    center = np.zeros((bbox.shape[0], 2, 1))
    for i in range(4):
        center[:, 0, 0] += bbox[:, 0,i]
        center[:, 1, 0] += bbox[:, 1,i]

    center = np.array(center,dtype=np.float32)/4.0

    # R = np.array([[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]], dtype=np.float32)
    R = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]], dtype=np.float32)

    normalized = np.matmul(R.transpose((2, 1, 0)),bbox-center)
    xmin = np.min(normalized[:, 0, :], axis=1)
    # print('diff: ', (xmin - normalized[:, 0, 3]))
    # assert sum((abs(xmin - normalized[:, 0, 3])) > eps) == 0
    xmax = np.max(normalized[:, 0, :], axis=1)
    # assert sum(abs(xmax - normalized[:, 0, 1]) > eps) == 0
    # print('diff2: ', xmax - normalized[:, 0, 1])
    ymin = np.min(normalized[:, 1, :], axis=1)
    # assert sum(abs(ymin - normalized[:, 1, 3]) > eps) == 0
    # print('diff3: ', ymin - normalized[:, 1, 3])
    ymax = np.max(normalized[:, 1, :], axis=1)
    # assert sum(abs(ymax - normalized[:, 1, 1]) > eps) == 0
    # print('diff4: ', ymax - normalized[:, 1, 1])

    w = xmax - xmin + 1
    h = ymax - ymin + 1

    w = w[:, np.newaxis]
    h = h[:, np.newaxis]
    # TODO: check it
    if with_module:
        angle = angle[:, np.newaxis] % ( 2 * np.pi)
    else:
        angle = angle[:, np.newaxis]
    dboxes = np.concatenate((center[:, 0].astype(np.float), center[:, 1].astype(np.float), w, h, angle), axis=1)
    return dboxes

def TuplePoly2Poly(poly):
    outpoly = [poly[0][0], poly[0][1],
                       poly[1][0], poly[1][1],
                       poly[2][0], poly[2][1],
                       poly[3][0], poly[3][1]
                       ]
    return outpoly

def cal_line_length(point1, point2):
    return math.sqrt( math.pow(point1[0] - point2[0], 2) + math.pow(point1[1] - point2[1], 2))

def get_best_begin_point_single(coordinate):
    x1 = coordinate[0][0]
    y1 = coordinate[0][1]
    x2 = coordinate[1][0]
    y2 = coordinate[1][1]
    x3 = coordinate[2][0]
    y3 = coordinate[2][1]
    x4 = coordinate[3][0]
    y4 = coordinate[3][1]
    xmin = min(x1, x2, x3, x4)
    ymin = min(y1, y2, y3, y4)
    xmax = max(x1, x2, x3, x4)
    ymax = max(y1, y2, y3, y4)
    combinate = [[[x1, y1], [x2, y2], [x3, y3], [x4, y4]], [[x2, y2], [x3, y3], [x4, y4], [x1, y1]],
                 [[x3, y3], [x4, y4], [x1, y1], [x2, y2]], [[x4, y4], [x1, y1], [x2, y2], [x3, y3]]]
    dst_coordinate = [[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]]
    force = 100000000.0
    force_flag = 0
    for i in range(4):
        temp_force = cal_line_length(combinate[i][0], dst_coordinate[0]) + cal_line_length(combinate[i][1],
                                                                                           dst_coordinate[
                                                                                               1]) + cal_line_length(
            combinate[i][2], dst_coordinate[2]) + cal_line_length(combinate[i][3], dst_coordinate[3])
        if temp_force < force:
            force = temp_force
            force_flag = i
    if force_flag != 0:
        pass
        # print("choose one direction!")
    return  combinate[force_flag]

def convertRBoox2XYWHA(ori_ptns):
    ori_ptns = np.array([ori_ptns], dtype=np.float32).reshape(4, 2)
    bp_ptns = get_best_begin_point_single(ori_ptns)
    bp_ptns = TuplePoly2Poly(bp_ptns)
    bp_ptns = [np.stack(list(bp_ptns))]
    out_gt_ptns = polygonToRotRectangle_batch(bp_ptns)[0]
    out_gt_ptns[4] = out_gt_ptns[4] % np.pi
    return list(out_gt_ptns)

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

    def inference_single(self, imagname, img_size, info_bboxes):
        img = mmcv.imread(imagname)
        height, width, channel = img.shape
        assert height==img_size[0] and width==img_size[1]
        # TODO: check the corner case
        # import pdb; pdb.set_trace()
        return inference_detector(self.model, (img, info_bboxes))
        
    def extract_roi_features_dets(self, args):
        img_size = [int(one) for one in args.image_size.split(',')]
        input_filepath, images_dir, output_dir = args.det_json_path, args.images_dir, args.output_dir
        buffix='det_%s_%s_%s' % (args.net_name, args.imdb_name, args.tag)

        with open(input_filepath, 'r') as f:
            json_data = json.load(f)
        
        for key, one_dets in json_data.items():
            srcpath = os.path.join(images_dir, key)
            det_ids, det_rbboxes, det_bboxes = one_dets['det_rbbox_ids'], one_dets['det_rbboxes_xywha'], one_dets['det_bboxes']
            
            ret_result = self.inference_single(srcpath, img_size, (buffix, output_dir, det_ids, det_rbboxes, det_bboxes))
            assert ret_result == True
    def extract_roi_features_dets(self, args):
        img_size = [int(one) for one in args.image_size.split(',')]
        input_filepath, images_dir, output_dir = args.det_json_path, args.images_dir, args.det_output_dir
        buffix='det_%s_%s_%s' % (args.net_name, args.imdb_name, args.tag)

        with open(input_filepath, 'r') as f:
            json_data = json.load(f)
        
        for key, one_dets in json_data.items():
            srcpath = os.path.join(images_dir, key)
            det_ids, det_rbboxes, det_bboxes = one_dets['det_rbbox_ids'], one_dets['det_rbboxes_xywha'], one_dets['det_bboxes']
            
            ret_result = self.inference_single(srcpath, img_size, (buffix, output_dir, det_ids, det_rbboxes, det_bboxes))
            assert ret_result == True

    def extract_roi_features_gt(self, args):
        img_size = [int(one) for one in args.image_size.split(',')]
        input_filepath, images_dir, output_dir = args.gt_json_path, args.images_dir, args.gt_output_dir
        buffix='gt_%s_%s_%s' % (args.net_name, args.imdb_name, args.tag)
        
        with open(input_filepath, 'r') as f:
            gt_json_data = json.load(f)
        image_list = gt_json_data['images']
        ann_list = gt_json_data['annotations']
        
        dict_imgid2anns = {} # imgid to bboxes/rbboxes
        for one_ann in ann_list:
            gt_anns=None
            if one_ann['image_id'] not in dict_imgid2anns:
                # gt_bbox_ids, gt_rbboxes_xywha, gt_bboxes
                gt_anns = [list(), list(), list()] # gt_bboxes:xywh
                dict_imgid2anns[one_ann['image_id']] = gt_anns
            else:
                gt_anns = dict_imgid2anns[one_ann['image_id']]
            rbbox_xywha = convertRBoox2XYWHA(one_ann['segmentation'])
            gt_anns[0].append(one_ann['id'])
            gt_anns[1].append(rbbox_xywha)
            gt_anns[2].append(one_ann['bbox'])

        for one_img in image_list:
            imgname, imgid = one_img['file_name'], one_img['id']
            assert imgid in dict_imgid2anns
            srcpath = os.path.join(images_dir, imgname)
            gt_anns = dict_imgid2anns[imgid]
            gt_ids, gt_rbboxes, gt_bboxes = gt_anns[0], gt_anns[1], gt_anns[2]
            
            ret_result = self.inference_single(srcpath, img_size, (buffix, output_dir, gt_ids, gt_rbboxes, gt_bboxes))
            assert ret_result == True

if __name__ == '__main__':
    seed = 13
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed+2)
    torch.cuda.manual_seed_all(seed+3)

    parser = argparse.ArgumentParser()
    parser.add_argument('--image_size', default='1024,1024', type=str, help='image size')
    parser.add_argument('--det_json_path', default='data/RSVG/det_instances_rsvg.json', type=str, help='detected bboxes per image')
    parser.add_argument('--det_output_dir', default='data/RSVG/hbb_obb_features_det', type=str, help='output dir')
    parser.add_argument('--gt_json_path', default='data/RSVG/instances_rsvg.json', type=str, help='ground truth bboxes')
    parser.add_argument('--gt_output_dir', default='data/RSVG/hbb_obb_features_gt', type=str, help='output dir')
    parser.add_argument('--images_dir', default='data/RSVG/images', type=str, help='images dir')
    
    parser.add_argument('--imdb_name', default='dota_v1_0', help='image databased trained on.')
    parser.add_argument('--net_name', default='res50')
    parser.add_argument('--tag', default='RoITransformer')

    args = parser.parse_args()

    roitransformer = DetectorModel(r'configs/DOTA/faster_rcnn_RoITrans_r50_fpn_1x_dota.py',
                  r'work_dirs/faster_rcnn_RoITrans_r50_fpn_1x_dota/epoch_12.pth')
    
    #用来抽取检测框的roi特征，包括roi_feats和rrois_feats，即水平框和旋转框
    roitransformer.extract_roi_features_dets(args)
    #用来抽取ground truth框的roi特征，包括水平框和旋转框的
    #roitransformer.extract_roi_features_gt(args)

