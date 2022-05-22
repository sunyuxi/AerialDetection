# -*-coding:utf-8 -*-

import json
import random
import math
import numpy as np

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

def xyxy2xywh(x):  # Convert bounding box format from [x1, y1, x2, y2] to [x, y, w, h]
    y=[0.0, 0.0, 0.0, 0.0]
    y[0] = (x[0] + x[2]) / 2.0
    y[1] = (x[1] + x[3]) / 2.0
    y[2] = x[2] - x[0]
    y[3] = x[3] - x[1]
    return y

def getHBB(poly_coords):
    x_list, y_list = [poly_coords[0], poly_coords[2], poly_coords[4], poly_coords[6]], [poly_coords[1], poly_coords[3], poly_coords[5], poly_coords[7]]
    xmin, ymin, xmax, ymax = min(x_list), min(y_list), max(x_list), max(y_list)

    return xmin,ymin, xmax, ymax

def convertDet2Json():
    det_filepath = 'data/RSVG/det_obb_rsvg.txt'
    instances_cocostyle_path='data/RSVG/instances_rsvg.json'
    json_det_path = 'data/RSVG/det_instances_rsvg.json'

    dict_imgname2id = {}
    with open(instances_cocostyle_path, 'r') as f:
        json_data = json.load(f)['images']
        for one_data in json_data:
            dict_imgname2id[one_data['file_name']] = one_data['id']

    dict_img2dets={}
    start_detid=0
    for line in open(det_filepath):
        arr = line.strip().split()
        imgname = arr[0]
        catname = arr[-1]
        det_rbboxes_ptns = [float(one) for one in arr[1:-2]]
        det_rbboxes_xywha = convertRBoox2XYWHA(det_rbboxes_ptns)
        det_score = float(arr[-2])
        one_dets = None
        assert imgname in dict_imgname2id
        imgid = dict_imgname2id[imgname]
        if imgname not in dict_img2dets:
            one_dets = {'file_name':imgname, 'image_id':dict_imgname2id[imgname], 'det_rbbox_ids':list(), \
            'det_rbboxes_ptns':list(), 'det_rbboxes_xywha':list(), 'det_bboxes':list(), \
            'det_scores':list(), 'det_categories':list()}
            dict_img2dets[imgname] = one_dets
        else:
            one_dets = dict_img2dets[imgname]
        #deal rbbox
        x1, y1, x2, y2 = getHBB(det_rbboxes_ptns)
        hbb_tmp = [round(x1,2), round(y1,2), round(x2,2), round(y2,2)]
        hbb = xyxy2xywh(hbb_tmp)
        one_dets['det_rbbox_ids'].append(start_detid)
        start_detid += 1
        one_dets['det_rbboxes_ptns'].append(det_rbboxes_ptns)
        one_dets['det_rbboxes_xywha'].append(det_rbboxes_xywha)
        one_dets['det_bboxes'].append(hbb)
        one_dets['det_scores'].append(det_score)
        one_dets['det_categories'].append(catname)

    with open(json_det_path, 'w') as f:
        json.dump(dict_img2dets, f)

if __name__ == '__main__':
    assert False #避免被失误执行

    seed = 13
    random.seed(seed)
    np.random.seed(seed)

    convertDet2Json()