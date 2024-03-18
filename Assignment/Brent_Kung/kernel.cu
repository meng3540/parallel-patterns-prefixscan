#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#define N 1024
#define MAX 9
#define P 12


__global__ void scan3(int* X, int* Y, int InputSize) {
    __shared__ int XY[N];
    int i = 2 * blockIdx.x * blockDim.x + threadIdx.x;
    if (i < InputSize) {
        XY[threadIdx.x] = X[i];
    }
    if (i + blockDim.x < InputSize) {
        XY[threadIdx.x + blockDim.x] = X[i + blockDim.x];
    }

    for (int j = 1; j <= blockDim.x; j *= 2) {
        __syncthreads();
        int index = (threadIdx.x + 1) * 2 * j - 1;
        if (index < N) {
            XY[index] += XY[index - j];
        }
    }

    for (int j = N / 4; j > 0; j /= 2) {
        __syncthreads();
        int index = (threadIdx.x + 1) * j * 2 - 1;
        if (index + j < N) {
            XY[index + j] += XY[index];
        }
    }

    __syncthreads();
    if (i < InputSize) {
        Y[i] = XY[threadIdx.x];
    }
    if (i + blockDim.x < InputSize) {
        Y[i + blockDim.x] = XY[threadIdx.x + blockDim.x];
    }
}

int main(int argc, char** argv) {

    int* data, * h_obstacle, * h_prefixSum; //Host pointers
    int* d_obstacle, * d_prefixSum3; //Device pointers

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
    cudaMalloc((void**)&d_prefixSum3, N * sizeof(int));

    //@@Copy memory to GPU
    cudaMemcpy(d_obstacle, h_obstacle, N * sizeof(int), cudaMemcpyHostToDevice);

    //@@Initialize the grid and block dimensions here
    dim3 blockSize(N);
    dim3 gridSize((int)ceil((float)N / blockSize.x));

    cudaEvent_t start3, stop3;
    cudaEventCreate(&start3);
    cudaEventCreate(&stop3);
    float t3 = 0;


    //@@Launch GPU kernel
    cudaEventRecord(start3);
    scan3 << <gridSize, blockSize >> > (d_obstacle, d_prefixSum3, N);
    cudaDeviceSynchronize();
    cudaEventRecord(stop3);

    //@@Copy GPU memory back to CPU
    cudaMemcpy(h_prefixSum, d_prefixSum3, N * sizeof(int), cudaMemcpyDeviceToHost);

    cudaEventSynchronize(stop3);
    cudaEventElapsedTime(&t3, start3, stop3);
    printf("\n\nTime (Brent-Kung): %f ms", t3);

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
    cudaFree(d_prefixSum3);

    //@@Free host memory
    free(h_obstacle);
    free(h_prefixSum);
}
