#include "kernel.h"

#include <memory>

__global__ void multiplyKernet(float *a, float *b, float *c)
{
	extern __shared__ float shared_a[SUB_SIZE*SUB_SIZE];
	extern __shared__ float shared_b[SUB_SIZE*SUB_SIZE];

	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	float loc_c = 0.0;

	for (int m = 0; m < SIZE / SUB_SIZE; ++m)
	{
		shared_a[(ty * SUB_SIZE) + tx] = a[(row*SIZE) + (m*SUB_SIZE + tx)];
		shared_b[(ty * SUB_SIZE) + tx] = b[(m*SUB_SIZE + ty) * SIZE + col];
		__syncthreads();
		for (int k = 0; k < SUB_SIZE; ++k)
			loc_c += shared_a[ty * SUB_SIZE + k] * shared_b[k * SUB_SIZE + tx];
		__syncthreads();
	}
	c[row*SIZE + col] = loc_c;
}


cudaError_t cudaCopy(float *dest, float *src, size_t size, cudaMemcpyKind copyKind)
{
	cudaError_t cudaStatus = cudaMemcpy(dest, src, size, copyKind);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed: %s", cudaGetErrorString(cudaStatus));
	}
	return cudaStatus;
}

cudaError_t cudaAllocate(void **devPtr, size_t size)
{
	cudaError_t cudaStatus = cudaMalloc(devPtr, size);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed: %s", cudaGetErrorString(cudaStatus));
	}

	return cudaStatus;
}

cudaError_t cudaCall(cudaError_t(*cudaFunc)(), const char *msg)
{
	cudaError_t cudaStatus = cudaFunc();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "%s: %s\n", msg, cudaGetErrorString(cudaStatus));
	}
	return cudaStatus;
}

cudaError_t cudaStartTimer(cudaEvent_t *start, cudaEvent_t *stop)
{
	cudaError_t cudaStatus = cudaEventCreate(start);
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaCEventCreate(start) failed (errorCode: %s)!\n", cudaGetErrorString(cudaStatus));
		return cudaStatus;
	}
	cudaStatus = cudaEventCreate(stop);
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaEventCreate(stop) failed (errorCode: %s)!\n", cudaGetErrorString(cudaStatus));
		return cudaStatus;
	}
	cudaStatus = cudaEventRecord(*start, 0);
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaEventRecord(start) failed (errorCode: %s)!\n", cudaGetErrorString(cudaStatus));
		return cudaStatus;
	}
	return cudaSuccess;
}

cudaError_t cudaStopTimer(cudaEvent_t *start, cudaEvent_t *stop, float & msecTotal)
{
	cudaError_t cudaStatus = cudaEventRecord(*stop, 0);
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaEventRecord(stop) failed !\n");
		return cudaStatus;
	}
	cudaStatus = cudaEventSynchronize(*stop);
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaEventSynchronize failed !\n");
		return cudaStatus;
	}
	msecTotal = 0.0f;
	cudaStatus = cudaEventElapsedTime(&msecTotal, *start, *stop);
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaEventElapsedTime failed !\n");
		return cudaStatus;
	}

	cudaStatus = cudaEventDestroy(*start);
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaEventDestroy failed !\n");
		return cudaStatus;
	}

	cudaStatus = cudaEventDestroy(*stop);
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaEventDestroy failed !\n");
		return cudaStatus;
	}

	return cudaSuccess;
}

int main()
{
    float a[SIZE*SIZE];
    float b[SIZE*SIZE];
    float c[SIZE*SIZE];
	float msecTotal = 0.0f;
	for (int i = 0; i < SIZE*SIZE; ++i)
	{
		a[i] = i+1;// (float)rand() / RAND_MAX;;
		b[i] = (i+1)*2;// (float)rand() / RAND_MAX;;
	}

	printMatrix(a, SIZE);
	printf("*\n");
	printMatrix(b, SIZE);
	
    // Add vectors in parallel.

	cudaError_t cudaStatus = multiplyMatrix((float*)a, (float*)b, (float*)c, SIZE, msecTotal);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "multiplyMatrix failed!");
        return 1;
    }

	printf("=\n");
	printMatrix(c, SIZE);

	printf("\nTotal miliseconds: %.5f\n", msecTotal);

	if (0 < cudaCall(cudaDeviceReset, "cudaDeviceReset failed!")) return 1;

    return 0;
}

#define CUDA_CHECK(FUN)				\
{									\
	cudaError_t status;				\
	status = FUN;					\
	if (status != cudaSuccess)		\
		return status;				\
}
		

cudaError_t multiplyMatrix(float *a, float *b, float *c, const int size, float & msecTotal)
{
	const size_t totalSize = size * size * sizeof(float);

	auto cudaMemoryDeleter = [&](float* ptr) { cudaFree(ptr); };
	
	std::shared_ptr<float> dev_a(new float(), cudaMemoryDeleter);
	std::shared_ptr<float> dev_b(new float(), cudaMemoryDeleter);
	std::shared_ptr<float> dev_c(new float(), cudaMemoryDeleter);

	cudaError_t cudaStatus;

	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		return cudaStatus;
	}

	CUDA_CHECK(cudaAllocate((void**)&dev_a, totalSize));
	CUDA_CHECK(cudaAllocate((void**)&dev_b, totalSize));
	CUDA_CHECK(cudaAllocate((void**)&dev_c, totalSize));

	CUDA_CHECK(cudaCopy(dev_a.get(), a, totalSize, cudaMemcpyHostToDevice));
	CUDA_CHECK(cudaCopy(dev_b.get(), b, totalSize, cudaMemcpyHostToDevice));

	dim3 gridDim(SUB_SIZE, SUB_SIZE);
	dim3 blockDim(SIZE / SUB_SIZE, SIZE / SUB_SIZE);

	cudaEvent_t start, stop;

	CUDA_CHECK(cudaStartTimer(&start, &stop));

	multiplyKernet <<< gridDim, blockDim>>> (dev_a.get(), dev_b.get(), dev_c.get());

	CUDA_CHECK(cudaStopTimer(&start, &stop, msecTotal));

	CUDA_CHECK(cudaCall(cudaGetLastError, "addKernet lauch failed"));
	CUDA_CHECK(cudaCall(cudaDeviceSynchronize, "cudaDeviceSynchronize returned error code"));
	CUDA_CHECK(cudaCopy(c, dev_c.get(), totalSize, cudaMemcpyDeviceToHost));
	
	return cudaSuccess;
}

void printMatrix(float *arr, const int size)
{
	for (int i = 0; i < size; ++i)
	{
		for (int j = 0; j < size; ++j)
		{
			printf("%.5f, ", arr[i*size + j]);
		}
		printf("\n");
	}
}
