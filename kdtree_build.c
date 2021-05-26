#include <stdio.h>
#include <stdlib.h>
#include "kdtree.h"

int main(void)
{
    srand(42);
    double ran;
    DATA* data = (DATA *) malloc(1000 * sizeof(DATA));
    for(int i = 0; i<3 * 1000; i++){
        data[i].x[0] = 1000 * (double) rand() / (double) RAND_MAX;
        data[i].x[1] = 1000 * (double) rand() / (double) RAND_MAX;
        data[i].x[2] = 1000 * (double) rand() / (double) RAND_MAX;
        data[i].w = 1;
        data[i].s = data[i].x[0]*data[i].x[0] + data[i].x[1]*data[i].x[1] + data[i].x[2]*data[i].x[2];
        printf("%f %f %f %f %f \n", data[i].x[0], data[i].x[1], data[i].x[2], data[i].w, data[i].s);
        //ran = (double) rand() / (double) RAND_MAX;
        
    }
    DATA buf;
    KDT* tree;
    int err = 0;
    
    tree = kdtree_build(data, (size_t) 1000, &buf, &err);


    return 0;
}