/**
 *  @file szx_utility.c
 *  @author Sheng Di
 *  @date Feb, 2022
 *  @brief 
 *  (C) 2022 by Mathematics and Computer Science (MCS), Argonne National Laboratory.
 *      See COPYRIGHT in top-level directory.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "szx_utility.h"

struct timeval sz_costStart; /*only used for recording the cost*/
double sz_totalCost = 0;

void sz_cost_start()
{
	sz_totalCost = 0;
	gettimeofday(&sz_costStart, NULL);
}

void sz_cost_end()
{
	double elapsed;
	struct timeval costEnd;
	gettimeofday(&costEnd, NULL);
	elapsed = ((costEnd.tv_sec*1000000+costEnd.tv_usec)-(sz_costStart.tv_sec*1000000+sz_costStart.tv_usec))/1000000.0;
	sz_totalCost += elapsed;
}

void sz_cost_end_msg(char *msg)
{
    double elapsed;
    struct timeval costEnd;
    gettimeofday(&costEnd, NULL);
    elapsed = ((costEnd.tv_sec*1000000+costEnd.tv_usec)-(sz_costStart.tv_sec*1000000+sz_costStart.tv_usec))/1000000.0;
    sz_totalCost += elapsed;
    printf("timecost=%f, %s\n", elapsed, msg);
}
