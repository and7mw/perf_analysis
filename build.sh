#!/bin/bash

g++ -g -O0 conv_nchw.cpp -c
g++ -g -O0 conv_nhwc.cpp -c
g++ -g -O0 reorder_nchw_nhwc.cpp -c

g++ -g -O0 main.cpp -DSIMD_IMPL conv_nchw.o conv_nhwc.o reorder_nchw_nhwc.o -o run-simd.out
g++ -g -O0 main.cpp -DNHWC_IMPL conv_nchw.o conv_nhwc.o reorder_nchw_nhwc.o -o run-nhwc.out
g++ -g -O0 main.cpp conv_nchw.o conv_nhwc.o reorder_nchw_nhwc.o -o run-nchw.out
