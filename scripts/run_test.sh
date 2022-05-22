python tools/test.py configs/DOTA/faster_rcnn_RoITrans_r50_fpn_1x_dota.py \
	work_dirs/faster_rcnn_RoITrans_r50_fpn_1x_dota/epoch_12.pth \
	--out work_dirs/faster_rcnn_RoITrans_r50_fpn_1x_dota/results.pkl \
	--log_dir logs

echo "next: sh run_test_results.sh"
echo "python DOTA_devkit/ResultMerge.py"
echo "python utils/convertTest.py"
