#!/bin/bash

g++ -O2 conv_nchw.cpp -c
g++ -O2 main.cpp conv_nchw.o -o run.out
