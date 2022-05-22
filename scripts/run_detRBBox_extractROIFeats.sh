#抽取所有的rbbox
python extract_roi_features.py
#将抽取结果转换成json格式
python utils/formatDetResults.py

#抽取detection bbox的roi特征
python extract_roi_features.py # 需要注释掉roitransformer.extract_roi_features_gt(args)函数
#抽取ground truth bbox的roi特征
python extract_roi_features.py # 需要注释掉roitransformer.extract_roi_features_dets(args)函数
