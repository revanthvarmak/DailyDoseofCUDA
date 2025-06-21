#include <iostream>
#include <cuda_runtime.h>



__global__ void addVectors(float* dA, float* dB, float* dC, int N){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    // Even after allocating block size and grid size to match value of N, there are cases where there can be few additional
    // threads allocated (Since block size can only be a multiple of 32), so we need this check
    if(tid < N){
        dC[tid] = dA[tid] + dB[tid];
    }
}


int main(){
    int N = 1 << 20;
    size_t bytes = N * sizeof(float);
    float *hA, *hB, *hC, *dA, *dB, *dC;

    // This allocates pinned memory on the Host side, which is non-pageable and faster
    // For allocating pageable memory, use malloc
    cudaMallocHost(&hA, bytes);
    cudaMallocHost(&hB, bytes);
    cudaMallocHost(&hC, bytes);

    for(int i = 0; i < N; ++i){
        hA[i] = i;
        hB[i] = i + 7;
    }

    // This allocates memory on the devide/CUDA 
    cudaMalloc(&dA, bytes);
    cudaMalloc(&dB, bytes);
    cudaMalloc(&dC, bytes);

    cudaMemcpy(dA, hA, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hB, bytes, cudaMemcpyHostToDevice);

    int blockSize = 256;
    int gridSize = (blockSize + N - 1) / blockSize;

    addVectors<<<gridSize, blockSize>>>(dA, dB, dC, N);

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