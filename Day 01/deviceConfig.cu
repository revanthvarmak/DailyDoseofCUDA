#include <iostream>
#include <cuda_runtime.h>

int main(){
    int numDevices = 0;
    cudaGetDeviceCount(&numDevices);

    if(numDevices == 0){
        std::cout << "There are no CUDA devices" << std::endl;
        return 1;
    }

    for(int d = 0; d < numDevices; ++d){
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, d);
        std::cout << "Device " << d << " : " << prop.name << std::endl;
        std::cout << "Total Global Memory (MB) :" << prop.totalGlobalMem / (1024 * 1024) << std::endl;
        std::cout << "Shared Memory per SM (KB) :" << prop.sharedMemPerMultiprocessor / (1024) << std::endl;
        std::cout << "Shared Memory per Block (KB) :" << prop.sharedMemPerBlock / 1024 << std::endl;
        std::cout << "Number of SM :" << prop.multiProcessorCount << std::endl;
        std::cout << "Warp size : " << prop.warpSize << std::endl;
        std::cout << "Max number of threads per block : " << prop.maxThreadsPerBlock << std::endl;
        std::cout << "Max number of threads per SM : " << prop.maxThreadsPerMultiProcessor << std::endl;
    }

    return 0;
}