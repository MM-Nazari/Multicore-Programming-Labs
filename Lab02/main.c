/*
*	In His Exalted Name
*	Matrix Addition - Sequential Code
*	Ahmad Siavashi, Email: siavashi@aut.ac.ir
*	15/04/2018
*/

// Let it be.
#define _CRT_SECURE_NO_WARNINGS

#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <omp.h>

typedef struct {
    int *A, *B, *C;
    int n, m;
} DataSet;

void fillDataSet(DataSet *dataSet);
void printDataSet(DataSet dataSet);
void closeDataSet(DataSet dataSet);
void add(DataSet dataSet);
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

int main(int argc, char *argv[]) {
    omp_check();
    DataSet dataSet;
    if (argc < 3) {
        printf("[-] Invalid No. of arguments.\n");
        printf("[-] Try -> <n> <m> \n");
        printf(">>> ");
        scanf("%d %d", &dataSet.n, &dataSet.m);
    }
    else {
        dataSet.n = atoi(argv[1]);
        dataSet.m = atoi(argv[2]);
    }
    fillDataSet(&dataSet);
    printDataSet(dataSet);
    add(dataSet);
    printDataSet(dataSet);
    closeDataSet(dataSet);
    system("PAUSE");
    return EXIT_SUCCESS;
}

void fillDataSet(DataSet *dataSet) {
    int i, j;

    dataSet->A = (int *)malloc(sizeof(int) * dataSet->n * dataSet->m);
    dataSet->B = (int *)malloc(sizeof(int) * dataSet->n * dataSet->m);
    dataSet->C = (int *)malloc(sizeof(int) * dataSet->n * dataSet->m);

    srand(time(NULL));

    for (i = 0; i < dataSet->n; i++) {
        for (j = 0; j < dataSet->m; j++) {
            dataSet->A[i*dataSet->m + j] = rand() % 100;
            dataSet->B[i*dataSet->m + j] = rand() % 100;
        }
    }
}

void printDataSet(DataSet dataSet) {
    int i, j;

    printf("[-] Matrix A\n");
    for (i = 0; i < dataSet.n; i++) {
        for (j = 0; j < dataSet.m; j++) {
            printf("%-4d", dataSet.A[i*dataSet.m + j]);
        }
        putchar('\n');
    }

    printf("[-] Matrix B\n");
    for (i = 0; i < dataSet.n; i++) {
        for (j = 0; j < dataSet.m; j++) {
            printf("%-4d", dataSet.B[i*dataSet.m + j]);
        }
        putchar('\n');
    }

    printf("[-] Matrix C\n");
    for (i = 0; i < dataSet.n; i++) {
        for (j = 0; j < dataSet.m; j++) {
            printf("%-8d", dataSet.C[i*dataSet.m + j]);
        }
        putchar('\n');
    }
}

void closeDataSet(DataSet dataSet) {
    free(dataSet.A);
    free(dataSet.B);
    free(dataSet.C);
}

void add(DataSet dataSet) {
    int i, j;
    int x=4, y=4, r, k;
    double start = omp_get_wtime();
    /*
#pragma omp parallel for num_threads(2) private(i, j)
    for (i = 0; i < dataSet.n; i++) { // i+x
        for (j = 0; j < dataSet.m; j++) {
            dataSet.C[i * dataSet.m + j] = dataSet.A[i * dataSet.m + j] + dataSet.B[i * dataSet.m + j];
        }
    }
     */

    for (i = 0; i < dataSet.n; i+=x) {
        #pragma omp parallel for num_threads(2) private(j,k,r)
        for (j = 0; j < dataSet.m; j+=y) {
                for (k=i; k<i+x ; k++) {
                    for (r = j ; r<j+y ; r++)
                         dataSet.C[k * dataSet.m + r] = dataSet.A[k * dataSet.m + r] + dataSet.B[k * dataSet.m + r];
                }
            }
    }
    double end = omp_get_wtime() - start;
    printf("Time sarf shode : %f\n", end);

}