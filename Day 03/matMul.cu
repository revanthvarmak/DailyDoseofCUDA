#include <iostream>
#include <cuda_runtime.h>

__global__ void matMul(float* dA, float* dB, float* dC, int N){
    // Each thread has a unique row, col which will compute a unique C[row][col]
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if(row < N && col < N){
        float sum = 0.0f;
        for(int k = 0; k < N; ++k){
            // dA has coalesced memory access, but dB memory access is strided, not very efficient, may load from 
            // global memory everytime
            sum += dA[row * N + k] * dB[k * N + col];
        }
        dC[row * N + col] = sum;
    }
}

int main(){
    int N = 1024;
    size_t bytes = N * N * sizeof(float);
    float *hA, *hB, *hC, *dA, *dB, *dC;

    // This allocates pinned memory on the Host side, which is non-pageable and faster
    // For allocating pageable memory, use malloc
    cudaMallocHost(&hA, bytes);
    cudaMallocHost(&hB, bytes);
    cudaMallocHost(&hC, bytes);

    for(int i = 0; i < N * N; ++i){
        hA[i] = 1;
        hB[i] = 1;
    }

    // This allocates memory on the devide/CUDA 
    cudaMalloc(&dA, bytes);
    cudaMalloc(&dB, bytes);
    cudaMalloc(&dC, bytes);

    cudaMemcpy(dA, hA, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hB, bytes, cudaMemcpyHostToDevice);


    // dim3 is a struct with 3 integers, can be used to make each grid and block, either 1D, 2D or 3D
    dim3 block(16, 16);
    dim3 grid((N + block.x - 1) / block.x, (N + block.y - 1) / block.y); 

    matMul<<<grid, block>>>(dA, dB, dC, N);

    cudaDeviceSynchronize();

    cudaMemcpy(hC, dC, bytes, cudaMemcpyDeviceToHost);

    for(int i = 0; i < 10; ++i){
        std::cout << hC[i] << std::endl;
    }

    cudaFree(dA);
    cudaFree(dB); 
    cudaFree(dC);

    cudaFreeHost(hA);
    cudaFreeHost(hB);
    cudaFreeHost(hC);

}