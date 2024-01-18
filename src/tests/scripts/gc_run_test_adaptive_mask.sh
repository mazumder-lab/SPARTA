#!/bin/bash 
# Loading the required module
source activate pruning

cd ..
cd ..

CUDA_VISIBLE_DEVICES=0 python3 -m test_adaptive_mask
