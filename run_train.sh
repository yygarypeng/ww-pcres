#!/bin/bash

taskset -c 12-15 python train.py -w &> record &
