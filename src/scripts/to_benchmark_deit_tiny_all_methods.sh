#!/bin/bash

# Loading the required module
source /etc/profile

module load anaconda/2023a 
source activate pruning

echo $LLSUB_RANK
echo $LLSUB_SIZE

# Run the script
python to_benchmark_deit_tiny_all_methods.py $LLSUB_RANK $LLSUB_SIZE