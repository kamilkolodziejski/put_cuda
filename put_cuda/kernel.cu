
#include "cuda_runtime.h"
#include <cuda.h>
#include <cuda_runtime_api.h>

#include <device_functions.h>
#include "device_launch_parameters.h"

#include <stdio.h>

const int size = 5;

cudaError_t multiplyMatrix(float *a, float *b, float *c, const int size);

void printMatrix(float *arr, const int arrSize);

__global__ void addKernel(int *c, const int *a, const int *b)
{
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
}

__global__ void addMatrixKernel(float *a, float *b, float *c, int size)
{
	int i = threadIdx.x;
	c[i] = a[i] + b[i];
}

__global__ void multiplyKernet(float *a, float *b, float *c, int size)
{
	extern __shared__ float t_c[];
	int i = threadIdx.x;
	int j = threadIdx.y;
	float loc_c = 0.0;
	__syncthreads();
	for (int k = 0; k < size; ++k)
	{
		loc_c += a[i*k+j] * b[j*k+i];
	}
	c[i*size + j] = loc_c;
	__syncthreads();
}

int main()
{
    float a[size*size];
    float b[size*size];
    float c[size*size];
	for (int i = 0; i < size*size; ++i)
	{
		a[i] = 1.0;// (float)rand() / RAND_MAX;;
		b[i] = 2.0;// (float)rand() / RAND_MAX;;
	}

	printMatrix(a, size);
	printf("*\n");
	printMatrix(b, size);
	
    // Add vectors in parallel.
    cudaError_t cudaStatus = multiplyMatrix((float*)a, (float*)b, (float*)c, size);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "multiplyMatrix failed!");
        return 1;
    }

	printf("=\n");
	//printMatrix(arrayC, size);
	for (int i = 0; i < size; ++i)
	{
		for (int j = 0; j < size; ++j)
		{
			printf("%.5f, ", c[size*i + j]);
		}
		printf("\n");
	}

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    return 0;
}

cudaError_t multiplyMatrix(float *a, float *b, float *c, const int size)
{
	float *dev_a;
	float *dev_b;
	float *dev_c;
	int dev_size;
	cudaError_t cudaStatus;

	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_a, size*size*sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_b, size*size * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_c, size*size * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		goto Error;
	}

	cudaStatus = cudaMemcpy(dev_a, a, size * size * sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(dev_b, b, size * size * sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}
	
	dim3 blockDim(size, size);
	multiplyKernet <<<1, blockDim, size*size >>> (dev_a, dev_b, dev_c, size);

	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching multiplyKernet!\n", cudaStatus);
		goto Error;
	}

	// Copy output vector from GPU buffer to host memory.
	cudaStatus = cudaMemcpy(c, dev_c, size * size * sizeof(float), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

Error:
	cudaFree(dev_a);
	cudaFree(dev_b);
	cudaFree(dev_c);
	return cudaStatus;
}

void printMatrix(float *arr, const int arrSize)
{
	for (int i = 0; i < arrSize; ++i)
	{
		for (int j = 0; j < arrSize; ++j)
		{
			printf("%.5f, ", arr[i*size + j]);
		}
		printf("\n");
	}
}
