##!/usr/bin/env sh

rm *_hash.txt
rm *_label.txt

TOOLS=../caffe/build/tools \

QUERY=image
RETRIEVAL=bows
GPU=3
DATA=iapr
MODEL=/data2/zhangxi/models/iapr-tc12/your_model_name.caffemodel
$TOOLS/extract_features_txt.bin $MODEL ${DATA}_extract_features_retrieval.prototxt label,${RETRIEVAL}/classifier_hash retrieval_label.txt,retrieval_hash.txt 18000 GPU ${GPU}
$TOOLS/extract_features_txt.bin $MODEL ${DATA}_extract_features_test.prototxt label,${QUERY}/classifier_hash test_label.txt,test_hash.txt 2000 GPU ${GPU}

#mir-25k:
#retrieval:18049
#train:9841
#test:1966

#nus-wide:
#retrieval:195834
#train:10500
#test:2100

#iapr-tc12:
#retrieval:18000
#train:10000
#test:2000
