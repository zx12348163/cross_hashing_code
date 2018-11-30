#!/usr/bin/env sh

GPU=1

../caffe/build/tools/caffe train \
    --solver=./iapr_solver.prototxt --weights=/data2/pre-train-models/VGG_ILSVRC_19_layers.caffemodel --gpu=${GPU}
