#include <iostream>
#include <sys/time.h>
#include <cuda.h>
using namespace std;

__global__ void d_kernel1(int *A, int *B, int *R1, int p, int q, int r)
{
    extern __shared__ int localSum[];
    if (threadIdx.x == 0)
    {
        for (int k = 0; k < r; k++)
        {
            localSum[k] = 0;
        }
    }
    __syncthreads();

    for (int i = 0; i < q; i++)
    {
        localSum[threadIdx.x] += A[blockIdx.x * q + i] * B[i * r + threadIdx.x];
        __syncthreads();
    }
    R1[blockIdx.x * r + threadIdx.x] = localSum[threadIdx.x];
}

__global__ void d_kernel2(int *C, int *D, int *R2, int p, int q, int r)
{
    extern __shared__ int localSum[];
    if (threadIdx.x == 0)
    {
        for (int k = 0; k < r; k++)
        {
            localSum[k] = 0;
        }
    }
    __syncthreads();
    for (int i = 0; i < q; i++)
    {
        localSum[threadIdx.x] += C[blockIdx.x * q + i] * D[threadIdx.x * q + i];
        __syncthreads();
    }

    R2[blockIdx.x * r + threadIdx.x] = localSum[threadIdx.x];
}

__global__ void d_kernel3(int *R1, int *R2, int *E)
{
    unsigned id = blockIdx.x * blockDim.x + threadIdx.x;
    E[id] = R1[id] + R2[id];
}

// function to compute the output matrix
void computE(int p, int q, int r, int *h_matrixA, int *h_matrixB,
             int *h_matrixC, int *h_matrixD, int *h_matrixE)
{
    // Device variables declarations...
    int *d_matrixA, *d_matrixB, *d_matrixC, *d_matrixD, *d_matrixE;
    int *d_matrixR1, *d_matrixR2;
    // allocate memory...
    cudaMalloc(&d_matrixA, p * q * sizeof(int));
    cudaMalloc(&d_matrixB, q * r * sizeof(int));
    cudaMalloc(&d_matrixC, p * q * sizeof(int));
    cudaMalloc(&d_matrixD, r * q * sizeof(int));
    cudaMalloc(&d_matrixE, p * r * sizeof(int));
    // gpu array to store result of AB
    cudaMalloc(&d_matrixR1, p * r * sizeof(int));
    // gpu array to store result of CDT
    cudaMalloc(&d_matrixR2, p * r * sizeof(int));

    // copy the values...
    cudaMemcpy(d_matrixA, h_matrixA, p * q * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_matrixB, h_matrixB, q * r * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_matrixC, h_matrixC, p * q * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_matrixD, h_matrixD, r * q * sizeof(int), cudaMemcpyHostToDevice);

    /* ****************************************************************** */
    /* Write your code here */
    /* Configure and launch kernels */
    // kernel 1 to compute AB
    dim3 grid1(p, 1, 1);
    dim3 block1(r, 1, 1);
    d_kernel1<<<grid1, block1, r * sizeof(int)>>>(d_matrixA, d_matrixB, d_matrixR1, p, q, r);

    // kernel 2 to compute CDT
    dim3 grid2(p, 1, 1);
    dim3 block2(r, 1, 1);
    d_kernel2<<<grid2, block2, r * sizeof(int)>>>(d_matrixC, d_matrixD, d_matrixR2, p, q, r);

    // kernel 3 to compute AB + CDT
    dim3 grid3(p, 1, 1);
    dim3 block3(r, 1, 1);
    d_kernel3<<<grid3, block3>>>(d_matrixR1, d_matrixR2, d_matrixE);
    /* ****************************************************************** */

    // copy the result back...
    cudaMemcpy(h_matrixE, d_matrixE, p * r * sizeof(int), cudaMemcpyDeviceToHost);

    // deallocate the memory...
    cudaFree(d_matrixA);
    cudaFree(d_matrixB);
    cudaFree(d_matrixC);
    cudaFree(d_matrixD);
    cudaFree(d_matrixE);
}

// function to read the input matrices from the input file
void readMatrix(FILE *inputFilePtr, int *matrix, int rows, int cols)
{
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            fscanf(inputFilePtr, "%d", &matrix[i * cols + j]);
        }
    }
}

// function to write the output matrix into the output file
void writeMatrix(FILE *outputFilePtr, int *matrix, int rows, int cols)
{
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            fprintf(outputFilePtr, "%d ", matrix[i * cols + j]);
        }
        fprintf(outputFilePtr, "\n");
    }
}

int main(int argc, char **argv)
{
    // variable declarations
    int p, q, r;
    int *matrixA, *matrixB, *matrixC, *matrixD, *matrixE;
    struct timeval t1, t2;
    double seconds, microSeconds;

    // get file names from command line
    char *inputFileName = argv[1];
    char *outputFileName = argv[2];

    // file pointers
    FILE *inputFilePtr, *outputFilePtr;

    inputFilePtr = fopen(inputFileName, "r");
    if (inputFilePtr == NULL)
    {
        printf("Failed to open the input file.!!\n");
        return 0;
    }

    // read input values
    fscanf(inputFilePtr, "%d %d %d", &p, &q, &r);

    // allocate memory and read input matrices
    matrixA = (int *)malloc(p * q * sizeof(int));
    matrixB = (int *)malloc(q * r * sizeof(int));
    matrixC = (int *)malloc(p * q * sizeof(int));
    matrixD = (int *)malloc(r * q * sizeof(int));
    readMatrix(inputFilePtr, matrixA, p, q);
    readMatrix(inputFilePtr, matrixB, q, r);
    readMatrix(inputFilePtr, matrixC, p, q);
    readMatrix(inputFilePtr, matrixD, r, q);

    // allocate memory for output matrix
    matrixE = (int *)malloc(p * r * sizeof(int));

    // call the compute function
    gettimeofday(&t1, NULL);
    computE(p, q, r, matrixA, matrixB, matrixC, matrixD, matrixE);
    cudaDeviceSynchronize();
    gettimeofday(&t2, NULL);

    // print the time taken by the compute function
    seconds = t2.tv_sec - t1.tv_sec;
    microSeconds = t2.tv_usec - t1.tv_usec;
    printf("Time taken (ms): %.3f\n", 1000 * seconds + microSeconds / 1000);

    // store the result into the output file
    outputFilePtr = fopen(outputFileName, "w");
    writeMatrix(outputFilePtr, matrixE, p, r);

    // close files
    fclose(inputFilePtr);
    fclose(outputFilePtr);

    // deallocate memory
    free(matrixA);
    free(matrixB);
    free(matrixC);
    free(matrixD);
    free(matrixE);

    return 0;
}
