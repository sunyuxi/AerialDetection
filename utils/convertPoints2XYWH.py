import numpy as np
import math

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

#original points1:
#angle1:
#original points1:
#angle1:
#original points1:
#angle1:
#original points1:
#angle1:
def test_points():
    ori_ptns = np.array([[700., 758.], [730., 758.], [730., 783.], [700., 783.]], dtype=np.float32)
    #xy,wh,angle
    gt_ptns = [715., 770.5, 26.00000191, 31.00000191, 4.71238899]
    bp_ptns = get_best_begin_point_single(ori_ptns)
    bp_ptns = TuplePoly2Poly(bp_ptns)
    bp_ptns = [np.stack(list(bp_ptns))]
    out_gt_ptns = polygonToRotRectangle_batch(bp_ptns)
    print(out_gt_ptns)
    print('--------------')

    #top-left->bottom-right
    ori_ptns = np.array([[ 187.7814 ,  995.7177 ], [ 235.46129,  958.8042 ], [ 285.10086, 1022.92194], [ 237.42097, 1059.8354 ]], dtype=np.float32)
    #xy,wh,angle
    gt_ptns = [236.44113159, 1009.31982422, 82.08743286, 61.29911041, 4.05358219]
    bp_ptns = get_best_begin_point_single(ori_ptns)
    bp_ptns = TuplePoly2Poly(bp_ptns)
    bp_ptns = [np.stack(list(bp_ptns))]
    out_gt_ptns = polygonToRotRectangle_batch(bp_ptns)
    print(out_gt_ptns)
    print('--------------')

    #bottom-left->top-right
    ori_ptns = np.array([[ 995.8026 ,  576.3831 ], [1022.9071 ,  567.70966], [1030.1176 ,  590.24225], [1003.01306,  598.9157 ]], dtype=np.float32)
    #xy,wh,angle
    gt_ptns = [1012.96008301, 583.31268311, 24.65815735, 29.45843506, 4.40268517]
    bp_ptns = get_best_begin_point_single(ori_ptns)
    bp_ptns = TuplePoly2Poly(bp_ptns)
    bp_ptns = [np.stack(list(bp_ptns))]
    out_gt_ptns = polygonToRotRectangle_batch(bp_ptns)
    print(out_gt_ptns)
    print('--------------')

def test_points2():
    #top-left->bottom-right
    ori_ptns = np.array([[ 187.7814 ,  995.7177 ], [ 235.46129,  958.8042 ], [ 285.10086, 1022.92194], [ 237.42097, 1059.8354 ]], dtype=np.float32)
    print(ori_ptns.shape)
    ori_ptns = np.array([248.4104 , 295.6511 , 216.75797, 308.3276 , 205.96313, 281.3737 , 237.61557, 268.69717 ], dtype=np.float32).reshape(4,2)
    #xy,wh,angle
    gt_ptns = [236.44113159, 1009.31982422, 82.08743286, 61.29911041, 4.05358219]
    bp_ptns = get_best_begin_point_single(ori_ptns)
    bp_ptns = TuplePoly2Poly(bp_ptns)
    bp_ptns = [np.stack(list(bp_ptns))]
    out_gt_ptns = polygonToRotRectangle_batch(bp_ptns)[0]
    out_gt_ptns[4] = out_gt_ptns[4]*180/np.pi % 180
    #if out_gt_ptns[2]<out_gt_ptns[3]:
    #    tmp=out_gt_ptns[2]
    #    out_gt_ptns[2]=out_gt_ptns[3]
    #    out_gt_ptns[3]=tmp
    #    out_gt_ptns[4] = (out_gt_ptns[4] + np.pi/2) % np.pi
    print(out_gt_ptns)
    print('--------------')

if __name__ == '__main__':
    test_points2()