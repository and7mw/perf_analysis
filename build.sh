#!/bin/bash

g++ -O2 conv_nchw.cpp -c
g++ -O2 conv_nhwc.cpp -c
g++ -O2 reorder_nchw_nhwc.cpp -c
g++ -O2 main.cpp conv_nchw.o conv_nhwc.o reorder_nchw_nhwc.o -o run.out
