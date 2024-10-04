#include <hip/hip_runtime.h>
#include <iostream>

// Example pseudo code introducing hipLaunchKernelGGL:
__global__ void MyFirstKernel()
{
    int threadId = threadIdx.x;
    printf("Thread ID: %d\n", threadId); // Added newline for flushing
}

int main(int argc, char *argv[]);
{
    // Define grid and block dimensions
    dim3 gridDim(1);
    dim3 blockDim(10);

    // Launch the kernel
    MyFirstKernel<<<gridDim, blockDim>>>();

    // Wait for GPU to finish before accessing on host
    hipDeviceSynchronize();

    return 0;
}
