/*
*				In His Exalted Name
*	Title:	Prefix Sum Sequential Code
*	Author: Ahmad Siavashi, Email: siavashi@aut.ac.ir
*	Date:	29/04/2018
*/

// Let it be.
#define _CRT_SECURE_NO_WARNINGS

#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <cuda_runtime.h>
#include "device_launch_parameters.h"


#define NUM_BLOCKS 1
#define NUM_TASK_PER_THREAD 4
#define SIZE 4096


void fill_array(int *a, size_t n);
void prefix_sum(int *a, size_t n);
void print_array(int *a, size_t n);
cudaError_t prefixWithCuda1(int *c, const int *a, unsigned int size, int *d);
cudaError_t prefixWithCuda2(float *c, float *a, unsigned int size);

__global__ void prefix1_1Kernel(int *c, const int *a, int *d, size_t n)
{
	
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= n) {
		return;
	}
	for (int j = 0; j < NUM_TASK_PER_THREAD; j++) {
		if (j == 0) {
			c[i * NUM_TASK_PER_THREAD + j] = a[i * NUM_TASK_PER_THREAD + j];
		
		}
		else {
			c[i * NUM_TASK_PER_THREAD + j] = a[i * NUM_TASK_PER_THREAD + j] + c[i * NUM_TASK_PER_THREAD + j - 1];
		}
	}
	d[i] = c[NUM_TASK_PER_THREAD * i + NUM_TASK_PER_THREAD - 1];

}

__global__ void prefix1_2Kernel(int *c, int* d)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int sum = 0;
	for (int j = 0; j < i; j++) {
		sum += d[j];

	}

	for (int j = 0; j < NUM_TASK_PER_THREAD; j++) {

		c[i * NUM_TASK_PER_THREAD + j] += sum;
	}
}




__global__ void prefix2Kernel(float *c, float *a, int iterate, size_t n) {


	int temp = 1;
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i <= n - 1) {
		c[i] = a[i];
	}


	for (int j = 0; j < iterate; j++) {

		if (i >= temp) {
			c[i] = a[i] + a[i - temp];
		}

		temp *= 2;


		a[i] = c[i];


	}


/*
	 __shared__ int shared_A[SIZE];
	 __shared__ int shared_C[SIZE];


	int temp = 1;
	int i = blockIdx.x * blockDim.x + threadIdx.x;




	if (i <= n - 1) {
		shared_A[i] = a[i];
		shared_C[i] = c[i];
		shared_C[i] = shared_A[i];
	}

	__syncthreads();

	for (int j = 0; j < iterate; j++) {

		if (i >= temp && i <= n-1) {
			shared_C[i] = shared_A[i] + shared_A[i - temp];
		}

		temp *= 2;

		if (i <= n - 1) {
			shared_A[i] = shared_C[i];
		}
		__syncthreads();
	}

	if (i <= n - 1) {
		 a[i] = shared_A[i];
	}

	__syncthreads();
*/

}


int main(int argc, char *argv[]) {
	// Input N
	size_t n = 0;
	printf("[-] Please enter N: ");
	scanf("%uld\n", &n);

	//clock_t tStart = clock();
	// Allocate memory for array
	int * a = (int *)malloc(n * sizeof a);
	int * c = (int *)malloc(n * sizeof a);
	int * d = (int *)malloc((n / 4) * sizeof d);
	// Fill array with numbers 1..n
	fill_array(a, n);
	// Print array
	// print_array(a, n);
	// Compute prefix sum
	//prefix_sum(a, n);
	prefixWithCuda1(c, a, n, d);
	//prefixWithCuda2(c, a, n);
	// Print array
	print_array(a, n);
	// Free allocated memory
	free(a);
	free(c);
	free(d);
	//printf("Elapsed time in sec %lf", (double)(clock() - tStart) / CLOCKS_PER_SEC);
	return EXIT_SUCCESS;
}

void prefix_sum(int *a, size_t n) {
	int i;
	// serial 
	
		for (i = 1; i < n; ++i) {
		a[i] = a[i] + a[i - 1];
	}
	
}

cudaError_t prefixWithCuda1(int *c, int *a, unsigned int size, int *d) {
	int *dev_a = 0;
	int *dev_c = 0;
	int *dev_d = 0;
	
	cudaError_t cudaStatus;

	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		printf("cudaSetDevice failed! Do you have a CUDA-capable GPU installed?");
	}

	cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		printf("cudaMalloc failed!");
	}

	cudaStatus = cudaMalloc((void**)&dev_d, size * sizeof(int)/4);
	if (cudaStatus != cudaSuccess) {
		printf("cudaMalloc failed!");
	}

	cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		printf("cudaMalloc failed!");
	}


	cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		printf("cudaMemcpy failed!");
	}

	clock_t start2 = clock();
	int num_blocks = NUM_BLOCKS;
	prefix1_1Kernel <<<num_blocks, 1024 >>> (dev_c, dev_a, dev_d, size);

	cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		printf("cudaMemcpy failed!");
	}

	cudaStatus = cudaMemcpy(d, dev_d, size * sizeof(int)/4, cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		printf("cudaMemcpy d failed!");
	}

	//print_array(d, size / 4);
	//print_array(c, size);
	
	prefix1_2Kernel <<<num_blocks, 1024 >>> (dev_c, dev_d);

	cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		printf("cudaMemcpy failed!");
	}

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
	cudaFree(dev_d);
	return cudaStatus;

}

cudaError_t prefixWithCuda2(float *c, float *a, unsigned int size) {
	
	float *dev_a = 0;
	float *dev_c = 0;
	int iterate = log2(size);

	cudaError_t cudaStatus;

	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		printf("cudaSetDevice failed! Do you have a CUDA-capable GPU installed?");
	}

	cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		printf("cudaMalloc failed!");
	}


	cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		printf("cudaMalloc failed!");
	}


	cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		printf("cudaMemcpy failed!");
	}

	clock_t start2 = clock();
	int num_blocks = NUM_BLOCKS;
	prefix2Kernel <<<num_blocks, 1024 >>> (dev_c, dev_a, iterate, size);

	/*
	
		
	cudaStatus = cudaMemcpy(a, dev_a, size * sizeof(int), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		printf("cudaMemcpy failed!");
	}

	*/
	cudaStatus = cudaMemcpy(a, dev_a, size * sizeof(float), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		printf("cudaMemcpy failed!");
	}




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

	cudaStatus = cudaMemcpy(a, dev_a, size * sizeof(float), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		printf("cudaMemcpy failed!");
	}

	cudaFree(dev_c);
	cudaFree(dev_a);

	return cudaStatus;

}

void print_array(int *a, size_t n) {
	int i;
	printf("[-] array: ");
	for (i = 0; i < n; ++i) {
		printf("%f, ", a[i]);
	}
	printf("\b\b \n");
}

void fill_array(int *a, size_t n) {
	int i;
	for (i = 0; i < n; ++i) {
		a[i] = i + 1;
	}
}
