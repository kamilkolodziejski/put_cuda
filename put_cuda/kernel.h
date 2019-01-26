#ifndef KERNEL_H
#define KERNEL_H

#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <device_functions.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

cudaError_t multiplyMatrix(float *a, float *b, float *c, const int size, float *msecTotal);

int cudaCopy(float *dest, float *src, size_t size, cudaMemcpyKind copyKind);
int cudaAllocate(void **devPtr, size_t size);
int cudaCall(cudaError_t(*cudaFunc)(), char *msg);
int cudaStartTimer(cudaEvent_t *start, cudaEvent_t *stop);
int cudaStopTimer(cudaEvent_t *start, cudaEvent_t *stop, float *msecTotal);

void printMatrix(float *arr, const int arrSize);

const int SIZE = 5;

#endif // !KERNEL_H

