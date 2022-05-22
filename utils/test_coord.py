from mmdet.core.bbox import gt_mask_bp_obbs_list,gt_mask_bp_obbs
import cv2

def test_coord():
    ptn_list = [[937.0, 913.0, 921.0, 912.0, 923.0, 874.0, 940.0, 875.0]]

    #ret_list = gt_mask_bp_obbs(ptn_list[0])
    contours, hierarchy = cv2.findContours(ptn_list[0], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    print(ret_list)

if __name__ == '__main__':
    test_coord()