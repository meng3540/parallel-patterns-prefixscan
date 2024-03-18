#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#define N 1024
#define MAX 9
#define P 12


__global__ void scan2(int* X, int* Y, int InputSize) {
    __shared__ int XY[N];
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < InputSize) {
        XY[threadIdx.x] = X[i];
    }
    for (int j = 1; j < blockDim.x; j *= 2) {
        __syncthreads();
        if (threadIdx.x >= j) {
            XY[threadIdx.x] += XY[threadIdx.x - j];
        }
        Y[i] = XY[threadIdx.x];
    }
}


int main(int argc, char** argv) {

    int* data, * h_obstacle, * h_prefixSum; //Host pointers
    int* d_obstacle, * d_prefixSum2; //Device pointers

    //Allocate and initialize matrices
    data = (int*)malloc(N * sizeof(int));
    h_obstacle = (int*)malloc(N * sizeof(int));
    h_prefixSum = (int*)malloc(N * sizeof(int));

    //Populate with distances
    for (int i = 0; i < N; i++) {
        int j = 8;
        if (i % j == 0) {
            data[i] = 3;
            //if (i+1 <N) { //this is used to make the obstacles wider in the array.
                //data[i+1]=3;
                //i++;
            //}
        }
        else {
            data[i] = 9;
        }
    }

    //Check if obstacle detected
    for (int i = 0; i < N; i++) {
        if (data[i] < MAX) {
            h_obstacle[i] = 1;
        }
        else {
            h_obstacle[i] = 0;
        }
    }

    //@@Display result
    printf("Array Size = %d\n", N);
    printf("\nData Array:\n");
    for (int i = 0; i < P; i++) {
        printf("%d  ", data[i]);
    }
    printf(" . . . ");
    for (int i = N - P; i < N; i++) {
        printf("  %d", data[i]);
    }

    printf("\nObstacle Array:\n");
    for (int i = 0; i < P; i++) {
        printf("%d  ", h_obstacle[i]);
    }
    printf(" . . . ");
    for (int i = N - P; i < N; i++) {
        printf("  %d", h_obstacle[i]);
    }



    //@@Allocate GPU Memory
    cudaMalloc((void**)&d_obstacle, N * sizeof(int));

    cudaMalloc((void**)&d_prefixSum2, N * sizeof(int));


    //@@Copy memory to GPU
    cudaMemcpy(d_obstacle, h_obstacle, N * sizeof(int), cudaMemcpyHostToDevice);

    //@@Initialize the grid and block dimensions here
    dim3 blockSize(N);
    dim3 gridSize((int)ceil((float)N / blockSize.x));

    cudaEvent_t start2, stop2;

    cudaEventCreate(&start2);
    cudaEventCreate(&stop2);
    float t2 = 0;

    //@@Launch GPU kernel
    cudaEventRecord(start2);
    scan2 << <gridSize, blockSize >> > (d_obstacle, d_prefixSum2, N);
    cudaDeviceSynchronize();
    cudaEventRecord(stop2);

    //@@Copy GPU memory back to CPU
    cudaMemcpy(h_prefixSum, d_prefixSum2, N * sizeof(int), cudaMemcpyDeviceToHost);

    cudaEventSynchronize(stop2);
    cudaEventElapsedTime(&t2, start2, stop2);
    printf("\n\nTime (Kogge-Stone): %f ms", t2);

    printf("\nPrefix Sum Array:\n");
    for (int i = 0; i < P; i++) {
        printf("%3.0d", h_prefixSum[i]);
    }
    printf("  . . . ");
    for (int i = N - P; i < N; i++) {
        printf("%3.0d", h_prefixSum[i]);
    }
    printf("\n%d obstacles detected", h_prefixSum[N - 1]);

    //@@Free GPU memory
    cudaFree(d_obstacle);
    cudaFree(d_prefixSum2);

    //@@Free host memory
    free(h_obstacle);
    free(h_prefixSum);
}
