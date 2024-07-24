/*
*				In His Exalted Name
*	Title:	Prefix Sum Sequential Code
*	Author: Ahmad Siavashi, Email: siavashi@aut.ac.ir
*	Date:	29/04/2018
*/

// Let it be.
#define _CRT_SECURE_NO_WARNINGS
#define NUMTHREADS 4

#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <omp.h>
#include <math.h>

void omp_check();
void fill_array(int *a, size_t n);
void prefix_sum(int *a, size_t n);
void print_array(int *a, size_t n);

int main(int argc, char *argv[]) {
	// Check for correct compilation settings
	omp_check();
	// Input N
	size_t n = 0;
	printf("[-] Please enter N: ");
	scanf("%uld\n", &n);
	// Allocate memory for array
	int * a = (int *)malloc(n * sizeof a);
	// Fill array with numbers 1..n
	fill_array(a, n);
	//print_array(a, n);
	// Compute prefix sum
	double start = omp_get_wtime();
	prefix_sum(a, n);
	double duration = omp_get_wtime() - start;
	//print_array(a, n);
	printf("Duration: %f\n", duration);
	// Free allocated memory
	free(a);
	system("PAUSE");
	return EXIT_SUCCESS;
}

void prefix_sum(int *a, size_t n) {
	int i;
	// serial 
	for (i = 1; i < n; ++i) {
		a[i] = a[i] + a[i - 1];
	}
	
	// parallel 1
	/*
	#pragma omp parallel for num_threads(NUMTHREADS) schedule(static, n/NUMTHREADS)
	for (i = 1; i < n; ++i) {
	
		int id = omp_get_thread_num();
		
		if (i % (n/NUMTHREADS) != 0) {
		
			a[i] = a[i] + a[i - 1];
			
		}
		
	}
	
	for (int j = (n / NUMTHREADS); j < n; j += n / NUMTHREADS) {
	
		int id = omp_get_thread_num();
		
		for (int k = j; k < j + n / NUMTHREADS; k++) {
		
			a[k] += a[j - 1];
			
		}
		
	}
	*/
	// parallel 2
	/*
	int iterate = log2(n);
	int temp = 1;
	int * t = (int *)malloc(n * sizeof a);
	
	for (int k = 0; k < n; k++) {
		t[k] = a[k];
	}
	
	#pragma omp parallel for num_threads(NUMTHREADS)// private(temp) //schedule(static,n/NUMTHREADS)
	for (int j = 0; j < iterate; j++) {

		for (i = temp; i < n; i++) {
			t[i] = a[i] + a[i - temp];
		}
		
		temp *= 2;
		
		for (int k = 0; k < n; k++) {
			a[k] = t[k];
		}
		
	}
	*/
}
	void print_array(int *a, size_t n) {
		int i;
		printf("[-] array: ");
		for (i = 0; i < n; ++i) {
			printf("%d, ", a[i]);
		}
		printf("\b\b  \n");
	}

	void fill_array(int *a, size_t n) {
		int i;
		for (i = 0; i < n; ++i) {
			a[i] = i + 1;
		}
	}

	void omp_check() {
		printf("------------ Info -------------\n");
#ifdef _DEBUG
		printf("[!] Configuration: Debug.\n");
#pragma message ("Change configuration to Release for a fast execution.")
#else
		printf("[-] Configuration: Release.\n");
#endif // _DEBUG
#ifdef _M_X64
		printf("[-] Platform: x64\n");
#elif _M_IX86 
		printf("[-] Platform: x86\n");
#pragma message ("Change platform to x64 for more memory.")
#endif // _M_IX86 
#ifdef _OPENMP
		printf("[-] OpenMP is on.\n");
		printf("[-] OpenMP version: %d\n", _OPENMP);
#else
		printf("[!] OpenMP is off.\n");
		printf("[#] Enable OpenMP.\n");
#endif // _OPENMP
		printf("[-] Maximum threads: %d\n", omp_get_max_threads());
		printf("[-] Nested Parallelism: %s\n", omp_get_nested() ? "On" : "Off");
#pragma message("Enable nested parallelism if you wish to have parallel region within parallel region.")
		printf("===============================\n");
	}
