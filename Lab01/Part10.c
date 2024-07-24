//
// Created by MMNazari1380 on 4/2/2023.
//

#include <stdio.h>
#include <math.h>
#include <omp.h>

const long int VERYBIG = 50000;
// ***********************************************************************
int main(void)
{
    int i;
    long int j, k, sum;
    double sumx, sumy, total;
    double starttime, elapsedtime;
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
        // Work Loop, do some work by looping VERYBIG times
        int id, sum2[2] = {0};
        double total2[2] = {0.0};
        #pragma omp parallel for private(sumx, sumy, k)
        for (j = 0; j<VERYBIG; j++)
        {
            id = omp_get_thread_num();
            // increment check sum
            if(id == 0){
                sum2[0] += 1;
            }
            if(id == 1){
                sum2[1] += 1;
            }

            // Calculate first arithmetic series
            sumx = 0.0;
            for (k = 0; k<j; k++)
                sumx = sumx + (double)k;
            // Calculate second arithmetic series
            sumy = 0.0;
            for (k = j; k>0; k--)
                sumy = sumy + (double)k;
            if (sumx > 0.0) {
                if (id == 0) {
                    total2[0] = total2[0] + 1.0 / sqrt(sumx);
                }
                if (id == 1) {
                    total2[1] = total2[1] + 1.0 / sqrt(sumx);
                }
            }
            if (sumy > 0.0){
                if(id == 0){
                    total2[0] = total2[0] + 1.0 / sqrt(sumy);
                }
                if(id == 1){
                    total2[1] = total2[1] + 1.0 / sqrt(sumy);
                }
            }
        }
        // get ending time and use it to determine elapsed time
        elapsedtime = omp_get_wtime() - starttime;
        sum = sum2[0] + sum2[1];
        total = total2[0] + total2[1];
        // report elapsed time
        printf("Time Elapsed: %f Secs, Total = %lf, Check Sum = %ld\n",
               elapsedtime, total, sum);
    }
    // return integer as required by function header
    getchar();
    return 0;
}


