import numpy as np
import torch
import math
import DOTA_devkit.polyiou as polyiou
import pdb

def pesudo_nms_poly(dets, iou_thr):
    keep = torch.range(0, len(dets))
    return dets, keep

def py_cpu_nms_poly_fast(dets, iou_thr):
    # TODO: check the type numpy()
    if dets.shape[0] == 0:
        keep = dets.new_zeros(0, dtype=torch.long)
        keep = keep.cpu().numpy()
        device = dets.device
        if isinstance(dets, torch.Tensor):
            dets = dets.cpu().numpy().astype(np.float64)
        if isinstance(iou_thr, torch.Tensor):
            iou_thr = iou_thr.cpu().numpy().astype(np.float64)
    else:
        device = dets.device
        if isinstance(dets, torch.Tensor):
            dets = dets.cpu().numpy().astype(np.float64)
        if isinstance(iou_thr, torch.Tensor):
            iou_thr = iou_thr.cpu().numpy().astype(np.float64)
        obbs = dets[:, 0:-1]
        # pdb.set_trace()
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
            # if order.size == 0:
            #     break
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            # w = np.maximum(0.0, xx2 - xx1 + 1)
            # h = np.maximum(0.0, yy2 - yy1 + 1)
            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            hbb_inter = w * h
            hbb_ovr = hbb_inter / (areas[i] + areas[order[1:]] - hbb_inter)
            # h_keep_inds = np.where(hbb_ovr == 0)[0]
            h_inds = np.where(hbb_ovr > 0)[0]
            tmp_order = order[h_inds + 1]
            for j in range(tmp_order.size):
                iou = polyiou.iou_poly(polys[i], polys[tmp_order[j]])
                hbb_ovr[h_inds[j]] = iou
                # ovr.append(iou)
                # ovr_index.append(tmp_order[j])

            # ovr = np.array(ovr)
            # ovr_index = np.array(ovr_index)
            # print('ovr: ', ovr)
            # print('thresh: ', thresh)
            try:
                if math.isnan(ovr[0]):
                    pdb.set_trace()
            except:
                pass
            inds = np.where(hbb_ovr <= iou_thr)[0]

            # order_obb = ovr_index[inds]
            # print('inds: ', inds)
            # order_hbb = order[h_keep_inds + 1]
            order = order[inds + 1]
            # pdb.set_trace()
            # order = np.concatenate((order_obb, order_hbb), axis=0).astype(np.int)
    return torch.from_numpy(dets[keep, :]).to(device), torch.from_numpy(np.array(keep)).to(device)

def py_cpu_nms_poly_fast_np(dets, thresh):
    try:
        obbs = dets[:, 0:-1]
    except:
        print('fail index')
        pdb.set_trace()
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
        # if order.size == 0:
        #     break
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        # w = np.maximum(0.0, xx2 - xx1 + 1)
        # h = np.maximum(0.0, yy2 - yy1 + 1)
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        hbb_inter = w * h
        hbb_ovr = hbb_inter / (areas[i] + areas[order[1:]] - hbb_inter)
        # h_keep_inds = np.where(hbb_ovr == 0)[0]
        h_inds = np.where(hbb_ovr > 0)[0]
        tmp_order = order[h_inds + 1]
        for j in range(tmp_order.size):
            iou = polyiou.iou_poly(polys[i], polys[tmp_order[j]])
            hbb_ovr[h_inds[j]] = iou
            # ovr.append(iou)
            # ovr_index.append(tmp_order[j])

        # ovr = np.array(ovr)
        # ovr_index = np.array(ovr_index)
        # print('ovr: ', ovr)
        # print('thresh: ', thresh)
        try:
            if math.isnan(ovr[0]):
                pdb.set_trace()
        except:
            pass
        inds = np.where(hbb_ovr <= thresh)[0]

        # order_obb = ovr_index[inds]
        # print('inds: ', inds)
        # order_hbb = order[h_keep_inds + 1]
        order = order[inds + 1]
        # pdb.set_trace()
        # order = np.concatenate((order_obb, order_hbb), axis=0).astype(np.int)
    return keep

def py_cpu_nms(dets, thresh):
    """Pure Python NMS baseline."""
    #print('dets:', dets)
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    ## index for dets
    order = scores.argsort()[::-1]


    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return keep

def bbox_poly2hbb(boxes):
    """
    with label
    :param boxes: (x1, y1, ... x4, y4, score) [n, 9]
    :return: hbb: (xmin, ymin, xmax, ymax, score) [n, 5]
    """
    n = boxes.shape[0]
    hbbs = np.zeros((n, 4))

    xs = np.reshape(boxes[:, : -1], (n, 4, 2))[:, :, 0]
    ys = np.reshape(boxes[:, : -1], (n, 4, 2))[:, :, 1]
    # pdb.set_trace()
    hbbs[:, 0] = np.min(xs, axis=1)
    hbbs[:, 1] = np.min(ys, axis=1)
    hbbs[:, 2] = np.max(xs, axis=1)
    hbbs[:, 3] = np.max(ys, axis=1)
    hbbs = np.hstack((hbbs, boxes[:, -1, np.newaxis]))
    return hbbs

def obb_HNMS(dets, iou_thr=0.5):
    """
    do nms on obbs by corresponding hbbs
    :param dets: shape (n, 9) (x1, y1, ..., score)
    :param iou_thr:
    :return:
    """
    if dets.shape[0] == 0:
        # TODO: use warp of function to reimplement it
        keep = dets.new_zeros(0, dtype=torch.long)
        keep = keep.cpu().numpy()
        device = dets.device
        if isinstance(dets, torch.Tensor):
            dets = dets.cpu().numpy().astype(np.float64)
    else:
        device = dets.device
        if isinstance(dets, torch.Tensor):
            dets = dets.cpu().numpy().astype(np.float64)
        if isinstance(iou_thr, torch.Tensor):
            iou_thr = iou_thr.cpu().numpy().astype(np.float64)
        h_dets = bbox_poly2hbb(dets)
        keep = py_cpu_nms(h_dets, iou_thr)

    return torch.from_numpy(dets[keep, :]).to(device), torch.from_numpy(np.array(keep)).to(device)

def obb_hybrid_NMS(dets, thresh_hbb=0.5, thresh_obb=0.3):
    """
    do nms on obbs by 1. corresponding hbbs on relative high thresh 2. then nms by obbs on obbs
    :param dets:
    :param thresh:
    :return:
    """
    if dets.shape[0] == 0:
        keep = dets.new_zeros(0, dtype=torch.long)
        keep = keep.cpu().numpy()
        device = dets.device
        if isinstance(dets, torch.Tensor):
            dets = dets.cpu().numpy().astype(np.float64)
    else:
        device = dets.device
        if isinstance(dets, torch.Tensor):
            dets = dets.cpu().numpy().astype(np.float64)
        if isinstance(thresh_hbb, torch.Tensor):
            thresh_hbb = thresh_hbb.cpu().numpy().astype(np.float64)
        if isinstance(thresh_obb, torch.Tensor):
            thresh_obb = thresh_obb.cpu().numpy().astype(np.float64)
    h_dets = bbox_poly2hbb(dets)
    h_keep = py_cpu_nms(h_dets, thresh_hbb)

    keeped_o_dets = dets[h_keep, :]
    o_keep = py_cpu_nms_poly_fast(keeped_o_dets, thresh_obb)

    final_keep = h_keep[o_keep]

    return torch.from_numpy(dets[final_keep, :]).to(device), torch.from_numpy(np.array(final_keep)).to(device)


