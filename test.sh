#!/bin/sh
PARTITION=Segmentation

dataset=$1
exp_name=$2
exp_dir=exp/${dataset}/${exp_name}
model_dir=${exp_dir}/model
result_dir=${exp_dir}/result
config=CELP_PFENet/config/${dataset}/${dataset}_${exp_name}.yaml

mkdir -p ${model_dir} ${result_dir}
now=$(date +"%Y%m%d_%H%M%S")
cp test.sh test_pfenet.py ${config} ${exp_dir}

python3 -u test_pfenet.py --config=${config} 2>&1 | tee ${result_dir}/test-$now.log
