#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <omp.h>

const long int VERYBIG = 50000;
// ***********************************************************************
int main(void)
{
    #ifndef _OPENMP
        printf("OpenMP is not supported – sorry!\n" );
        getchar();
        return 0;
    #endif
    int i;
	long int j, k, sum;
	double sumx, sumy, total;
	double starttime, elapsedtime;
	double elapsedtime0, elapsedtime1, elapsedtime2, elapsedtime3, elapsedtime4;
	// -----------------------------------------------------------------------
	// Output a start message
	printf("Serial Timings for %ld iterations\n\n", VERYBIG);
	// repeat experiment several times
	for (i = 0; i<10; i++)
	{
		// get starting time56 x CHAPTER 3 PARALLEL STUDIO XE FOR THE IMPATIENT
		starttime = omp_get_wtime();
		// reset check sum & running total
		sum = 0;
		total = 0.0;
        int num_thread = 0;
		// Work Loop, do some work by looping VERYBIG times
        #pragma omp parallel for private(k, sumx, sumy) reduction(+:total, sum) num_threads(4)
		for (j = 0; j<VERYBIG; j++)
		{
            if(num_thread == 0){
                printf("Number of Threads = %d\t", omp_get_num_threads());
                num_thread++;
            }
            elapsedtime0 = omp_get_wtime();
			// increment check sum
            //#pragma omp critical
            //#pragma omp atomic
			sum += 1;
            elapsedtime1 = omp_get_wtime();
			// Calculate first arithmetic series
			sumx = 0.0;
			for (k = 0; k<j; k++)
				sumx = sumx + (double)k;
            elapsedtime2 = omp_get_wtime();
			// Calculate second arithmetic series
			sumy = 0.0;
			for (k = j; k>0; k--)
				sumy = sumy + (double)k;
            elapsedtime3 = omp_get_wtime();
            //#pragma omp critical
            //{
			    if (sumx > 0.0){
                    //#pragma omp atomic
                    total = total + 1.0 / sqrt(sumx);
                }
			    if (sumy > 0.0){
                    //#pragma omp atomic
                    total = total + 1.0 / sqrt(sumy);
                }
            //}
            elapsedtime4 = omp_get_wtime();
		}
		// get ending time and use it to determine elapsed time
		elapsedtime = omp_get_wtime() - starttime;
		double elapsedtime01 = elapsedtime1 - elapsedtime0;
		double elapsedtime02 = elapsedtime2 - elapsedtime1;
		double elapsedtime03 = elapsedtime3 - elapsedtime2;
		double elapsedtime04 = elapsedtime4 - elapsedtime3;
		// report elapsed time
		printf("Time Elapsed: %f Secs, Total = %lf, Check Sum = %ld\n",elapsedtime, total, sum);
		printf("Time Elapsed Part 1 : %f , Time Elapsed Part 2 : %f , Time Elapsed Part 3 : %f , Time Elapsed Part 4 : %f \n"
               ,elapsedtime01, elapsedtime02, elapsedtime03, elapsedtime04);
	}
	// return integer as required by function header
	getchar();
	return 0;
}
