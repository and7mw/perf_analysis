#!/bin/bash

g++ -O2 conv_nchw.cpp -c
g++ -O2 conv_nhwc.cpp -c
g++ -O2 -mavx2 -mfma conv_nhwc_simd.cpp -c
g++ -O2 reorder_nchw_nhwc.cpp -c

g++ -O2 main.cpp -DSIMD_IMPL conv_nchw.o conv_nhwc.o reorder_nchw_nhwc.o conv_nhwc_simd.o -o run-simd.out
g++ -O2 main.cpp -DNHWC_IMPL conv_nchw.o conv_nhwc.o reorder_nchw_nhwc.o conv_nhwc_simd.o -o run-nhwc.out
g++ -O2 main.cpp conv_nchw.o conv_nhwc.o reorder_nchw_nhwc.o conv_nhwc_simd.o -o run-nchw.out
