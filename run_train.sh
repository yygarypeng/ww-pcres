# bin/bash

# pcres -> use 4-7 (0-3 for ww-flow) 
taskset -c 0-5 python train.py -w &> record &
