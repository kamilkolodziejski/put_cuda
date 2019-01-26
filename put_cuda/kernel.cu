#include "kernel.h"

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


int cudaCopy(float *dest, float *src, size_t size, cudaMemcpyKind copyKind)
{
	cudaError_t cudaStatus = cudaMemcpy(dest, src, size, copyKind);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed: %s", cudaGetErrorString(cudaStatus));
		return -1;
	}
	return 0;
}

int cudaAllocate(void **devPtr, size_t size)
{
	cudaError_t cudaStatus = cudaMalloc(devPtr, size);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed: %s", cudaGetErrorString(cudaStatus));
		return -1;
	}
	return 0;
}

int cudaCall(cudaError_t(*cudaFunc)(), char *msg)
{
	cudaError_t cudaStatus = cudaFunc();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "%s: %s\n", msg, cudaGetErrorString(cudaStatus));
		return -1;
	}
	return 0;
}

int cudaStartTimer(cudaEvent_t *start, cudaEvent_t *stop)
{
	cudaError_t cudaStatus = cudaEventCreate(start);
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaCEventCreate(start) failed (errorCode: %s)!\n", cudaGetErrorString(cudaStatus));
		return -1;
	}
	cudaStatus = cudaEventCreate(stop);
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaEventCreate(stop) failed (errorCode: %s)!\n", cudaGetErrorString(cudaStatus));
		return -1;
	}
	cudaStatus = cudaEventRecord(*start, 0);
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaEventRecord(start) failed (errorCode: %s)!\n", cudaGetErrorString(cudaStatus));
		return -1;
	}
	return 0;
}

int cudaStopTimer(cudaEvent_t *start, cudaEvent_t *stop, float *msecTotal)
{
	cudaError_t cudaStatus = cudaEventRecord(*stop, 0);
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaEventRecord(stop) failed !\n");
		return -1;
	}
	cudaStatus = cudaEventSynchronize(*stop);
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaEventSynchronize failed !\n");
		return -1;
	}
	*msecTotal = 0.0f;
	cudaStatus = cudaEventElapsedTime(msecTotal, *start, *stop);
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaEventElapsedTime failed !\n");
		return -1;
	}
	cudaEventDestroy(*start);
	cudaEventDestroy(*stop);
	return 0;
}

int main()
{
    float a[SIZE*SIZE];
    float b[SIZE*SIZE];
    float c[SIZE*SIZE];
	float msecTotal = 0.0f;
	for (int i = 0; i < SIZE*SIZE; ++i)
	{
		a[i] = 1.0;// (float)rand() / RAND_MAX;;
		b[i] = 2.0;// (float)rand() / RAND_MAX;;
	}

	printMatrix(a, SIZE);
	printf("*\n");
	printMatrix(b, SIZE);
	
    // Add vectors in parallel.

	cudaError_t cudaStatus = multiplyMatrix((float*)a, (float*)b, (float*)c, SIZE, &msecTotal);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "multiplyMatrix failed!");
        return 1;
    }

	printf("=\n");
	printMatrix(c, SIZE);

	printf("\nTotal miliseconds: %.5f\n", msecTotal);

	if (0 < cudaCall(cudaDeviceReset, "cudaDeviceReset failed!")) return 1;
    //cudaStatus = cudaDeviceReset();
    //if (cudaStatus != cudaSuccess) {
    //    fprintf(stderr, "cudaDeviceReset failed!");
    //    return 1;
    //}

    return 0;
}

cudaError_t multiplyMatrix(float *a, float *b, float *c, const int size, float *msecTotal)
{
	float *dev_a;
	float *dev_b;
	float *dev_c;
	cudaError_t cudaStatus;

	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		goto Error;
	}

	if (0 > cudaAllocate((void**)&dev_a, size*size * sizeof(float))) goto Error;
	if (0 > cudaAllocate((void**)&dev_b, size*size * sizeof(float))) goto Error;
	if (0 > cudaAllocate((void**)&dev_c, size*size * sizeof(float))) goto Error;

	if (0 > cudaCopy(dev_a, a, size*size * sizeof(float), cudaMemcpyHostToDevice)) goto Error;
	if (0 > cudaCopy(dev_b, b, size*size * sizeof(float), cudaMemcpyHostToDevice)) goto Error;
		
	dim3 blockDim(size, size);
	cudaEvent_t start, stop;

	if(0 > cudaStartTimer(&start, &stop)) goto Error;

	multiplyKernet <<<1, blockDim, size*size >>> (dev_a, dev_b, dev_c, size);

	if(0 > cudaStopTimer(&start, &stop, msecTotal)) goto Error;
	
	if (0 > cudaCall(cudaGetLastError, "addKernet lauch failed") ) goto Error;
	if (0 > cudaCall(cudaDeviceSynchronize, "cudaDeviceSynchronize returned error code")) goto Error;
	if (0 > cudaCopy(c, dev_c, size * size * sizeof(float), cudaMemcpyDeviceToHost)) goto Error;
	//	fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching multiplyKernet!\n", cudaStatus);

Error:
	cudaFree(dev_a);
	cudaFree(dev_b);
	cudaFree(dev_c);
	return cudaStatus;
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
