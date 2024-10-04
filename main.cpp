#include <hip/hip_runtime.h>
#include <iostream>
#include <algorithm>
#include <stdio.h>
#include <vector>
#include <cmath>

__global__ void MyFirstKernel_a()
{
    int threadId = threadIdx.x;
    printf("Thread ID: %d\n", threadId);
}

__global__ void MyFirstKernel_b(int* a, int* b, int* c, unsize arr_size)
{
    
    int arr_id = blockdim.x * blockIdx.x + threadIdx.x;
    if (arr_id < arr_size) {
        c[arr_id] = b[arr_id] + a[arr_id];
    }
}

void matadd_naive(std::vector a, std::vector b, std::vector result_c) {
    for (int i = 0; i < a.size(); i ++) {
        c[i] = a[i] + b[i]
    }
}

void compare_reuslt(std::vector c, std::vector result_c) {
    for (int i = 0; i < a.size(); i ++) {
        assert(c[i] == result_c[i] && "addition error");
    }
}

int main(int argc, char *argv[])
{
    std::random_device rnd_device;
    std::mt19937 mersenne_engine {rnd_device()};
    std::uniform_int_distribution<int> dist {1, 52};
    auto gen = [&](){
                   return dist(mersenne_engine);
               };
    
    unsize arr_size = argv[1];

    int BLOCK_SIZE = 1024;

    dim3 blockDim(BLOCK_SIZE);
    dim3 gridDim(ceil(div(arr_size, BLOCK_SIZE)));

    std::vector<int> a(arr_size);
    std::vector<int> b(arr_size);
    std::vector<int> c(arr_size);
    std::vector<int> result_c(arr_size);
    
    std::generate(a.begin(), a.end(), gen);
    std::generate(b.begin(), b.end(), gen);

    int* d_a;
    int* d_b;
    int* d_c;
    HIPCHECK(hipMalloc(&d_a, arr_size * sizeof(int)));
    HIPCHECK(hipMalloc(&d_a, arr_size * sizeof(int)));
    HIPCHECK(hipMalloc(&d_c, arr_size * sizeof(int)));
    
    HIPCHECK(hipMemcpy(d_a, a, arr_size * sizeof(int), hipMemcpyHostToDevice)); 
    HIPCHECK(hipMemcpy(d_b, b, arr_size * sizeof(int), hipMemcpyHostToDevice)); 

    hipLaunchKernelGGL(MyFirstKernel_b, blockDim, gridDim, 0, 0, d_a, d_b, d_c, arr_size);
    HIPCHECK(hipDeviceSynchronize());

    hipMemcpy(c, d_c, arr_size * sizeof(int), hipMemcpyDeviceToHost); 

    matadd_naive(a, b, result_c);
    compare_reuslt(c, result_c);

    return 0;
}
