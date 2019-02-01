#ifndef KERNEL_H
#define KERNEL_H

#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime_api.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

cudaError_t multiplyMatrix(float *a, float *b, float *c, const int size, float & msecTotal);

cudaError_t cudaCopy(float *dest, float *src, size_t size, cudaMemcpyKind copyKind);
cudaError_t cudaAllocate(void **devPtr, size_t size);
int cudaCall(cudaError_t(*cudaFunc)(), char *msg);
cudaError_t cudaStartTimer(cudaEvent_t *start, cudaEvent_t *stop);
cudaError_t cudaStopTimer(cudaEvent_t *start, cudaEvent_t *stop, float & msecTotal);

void printMatrix(float *arr, const int arrSize);

const int SIZE = 4;
const int SUB_SIZE = 2;

#endif // !KERNEL_H

