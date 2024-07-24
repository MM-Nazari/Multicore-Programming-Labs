/*
*	In His Exalted Name
*	Vector Addition - Sequential Code
*	Ahmad Siavashi, Email: siavashi@aut.ac.ir
*	21/05/2018
*/
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

/* case 02 & 03 */
#define NUM_BLOCKS 1
#define NUM_TASK_PER_THREAD 1

void fillVector(int * v, size_t n);
void addVector(int * a, int *b, int *c, size_t n);
void printVector(int * v, size_t n);
cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size);

__global__ void add2Kernel(int *c, const int *a, const int *b)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	for (int j = 0; j < NUM_TASK_PER_THREAD; j++) {
		c[i] = a[i] + b[i];
		i += 1024;
	}
}


int main()
{


// use for n block and 1 task per thread
const int vectorSize1 = 1024 * NUM_BLOCKS;
// use for 1 block and n task per thread
const int vectorSize2 = 1024 * NUM_TASK_PER_THREAD;

int a[vectorSize1], b[vectorSize1], c[vectorSize1];


fillVector(a, vectorSize1);
fillVector(b, vectorSize1);

addWithCuda(c, a, b, vectorSize1);

//addVector(a, b, c, vectorSize1);

//printVector(c, vectorSize1);

return EXIT_SUCCESS;
}

// Fills a vector with data
void fillVector(int * v, size_t n) {
	int i;
	for (i = 0; i < n; i++) {
		v[i] = i;
	}
}

// Adds two vectors
void addVector(int * a, int *b, int *c, size_t n) {
	int i;
	for (i = 0; i < n; i++) {
		c[i] = a[i] + b[i];
	}
}

// Prints a vector to the stdout.
void printVector(int * v, size_t n) {
	int i;
	printf("[-] Vector elements: ");
	for (i = 0; i < n; i++) {
		printf("%d, ", v[i]);
	}
	printf("\b\b  \n");
}

cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size) {
	int *dev_a = 0;
	int *dev_b = 0;
	int *dev_c = 0;
	cudaError_t cudaStatus;

	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		printf("cudaSetDevice failed! Do you have a CUDA-capable GPU installed?");
	}

	cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		printf("cudaMalloc failed!");
	}

	cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		printf("cudaMalloc failed!");
	}

	cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		printf("cudaMalloc failed!");
	}

	cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		printf("cudaMemcpy failed!");
	}

	cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		printf("cudaMemcpy failed!");
	}

	
	clock_t start2 = clock();
	int num_blocks = NUM_BLOCKS;
	add2Kernel << <num_blocks, 1024 >> > (dev_c, dev_a, dev_b);

	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		printf("addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
	}

	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		printf("cudaDeviceSynchronize returned error code %d after launching addKernel!\n",
			cudaStatus);
	}

	
	double time2 = (double)(clock() - start2) / CLOCKS_PER_SEC;

	printf(" time is %f", time2);

	cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		printf("cudaMemcpy failed!");
	}

	cudaFree(dev_c);
	cudaFree(dev_a);
	cudaFree(dev_b);
	return cudaStatus;

}



/* case 04 
__global__ void ThreadDetails(int *block, int *warp, int *local_index) {

	int global_index = blockDim.x * blockIdx.x + threadIdx.x;
	block[global_index] = blockIdx.x;
	warp[global_index] = threadIdx.x / warpSize;
	local_index[global_index] = threadIdx.x;
}

int main(int argc, char **argv) {

	int size = 64 * 2;

	int *block, *warp, *local_index;
	cudaMallocManaged(&block, size * sizeof(int));
	cudaMallocManaged(&warp, size * sizeof(int));
	cudaMallocManaged(&local_index, size * sizeof(int));

	ThreadDetails <<<2, 64>>>(block, warp, local_index);
	cudaDeviceSynchronize();

	for (int i = 0; i < size; i++) {
		printf("Calculated Thread: %d,\tBlock: %d,\tWarp %d,\tThread %d\n", i, block[i], warp[i], local_index[i]);
	}

	return 0;
}
*/
