NVCC:=nvcc
# Multi-arch for RunPod: A100(sm_80), A10/L40(sm_86), RTX3090(sm_86), T4(sm_75), RTX4070/4090(sm_89), RTX5090(sm_100)
GPU_CFLAGS:=-gencode arch=compute_75,code=sm_75 -gencode arch=compute_80,code=sm_80 -gencode arch=compute_86,code=sm_86 -gencode arch=compute_89,code=sm_89 -gencode arch=compute_90,code=sm_90 -gencode arch=compute_100,code=sm_100
# No maxrregcount: 0 spills, 207 regs — best throughput (~16M cps RTX 4070)
CFLAGS_release:=--ptxas-options=-v $(GPU_CFLAGS) -O3 --use_fast_math -Xcompiler "-Wall -Werror -fPIC -Wno-strict-aliasing"
CFLAGS_debug:=$(CFLAGS_release) -g
CFLAGS:=$(CFLAGS_$V)
