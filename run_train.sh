# bin/bash

# pcres -> use 4-7 (0-3 for ww-flow) 
taskset -c 4-7 python train.py -w &> record &
