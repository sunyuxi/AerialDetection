import os

def convert_files1():
    input_dir = '/mnt/A/sunyuxi/objectdetection/AerialDetection/work_dirs/faster_rcnn_RoITrans_r50_fpn_1x_dota/Task1_results'
    output_dir = '/mnt/A/sunyuxi/objectdetection/AerialDetection/work_dirs/faster_rcnn_RoITrans_r50_fpn_1x_dota/Task1_results_new'

    for filename in os.listdir(input_dir):
        filepath = os.path.join(input_dir, filename)
        with open(os.path.join(output_dir, 'Task1_'+filename), 'w') as f:
            for line in open(filepath):
                arr=line.strip().split(' ')
                det_arr = [float(one) for one in arr[1:]]
                imgname_arr = arr[0].split('__')
                left = float(imgname_arr[2])
                up = float(imgname_arr[-1].split('.')[0].split('_')[-1])
                det_arr_final = [det_arr[0]] + [val+left if idx%2==0 else val+up for idx,val in enumerate(det_arr[1:])]
                f.write(imgname_arr[0] + ' ' + ' '.join([str(one) for one in det_arr_final]) + "\n")

def convert_files2():
    input_dir = '/mnt/A/sunyuxi/objectdetection/AerialDetection/work_dirs/faster_rcnn_RoITrans_r50_fpn_1x_dota/Task1_results_merge'
    output_dir = '/mnt/A/sunyuxi/objectdetection/AerialDetection/work_dirs/faster_rcnn_RoITrans_r50_fpn_1x_dota/Task1_results_merge_new'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    for filename in os.listdir(input_dir):
        filepath = os.path.join(input_dir, filename)
        with open(os.path.join(output_dir, 'Task1_'+filename), 'w') as f:
            for line in open(filepath):
                f.write(line)

if __name__ == '__main__':
    convert_files2()