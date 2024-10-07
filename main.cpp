#include <hip/hip_runtime.h>
#include <hip/hip_common.h>
#include <iostream>
#include <algorithm>
#include <stdio.h>
#include <vector>
#include <cmath>
#include <random>
#include <cassert>

__global__ void MyFirstKernel_a()
{
    int threadId = threadIdx.x;
    printf("Thread ID: %d\n", threadId);
}

__global__ void MyFirstKernel_b(int* a, int* b, int* c, int arr_size)
{
    int arr_id = blockDim.x * blockIdx.x + threadIdx.x;
    if (arr_id < arr_size) {
        c[arr_id] = b[arr_id] + a[arr_id];
    }
}

int div_ceil(int x, int y) {
    return (x + y - 1) / y;
}

void matadd_naive(const std::vector<int>& a, const std::vector<int>& b, std::vector<int>& result_c) {
    for (size_t i = 0; i < a.size(); i++) {
        result_c[i] = a[i] + b[i];
    }
}

void compare_result(const std::vector<int>& c, const std::vector<int>& result_c) {
    for (size_t i = 0; i < result_c.size(); i++) {
        assert(c[i] == result_c[i] && "addition error");
    }
}

int main(int argc, char *argv[])
{
    if (argc != 2) {
        std::cerr << "Usage: ./program array_size" << std::endl;
        return -1;
    }

    std::random_device rnd_device;
    std::mt19937 mersenne_engine{rnd_device()};
    std::uniform_int_distribution<int> dist{1, 52};
    auto gen = [&dist, &mersenne_engine]() {
        return dist(mersenne_engine);
    };

    int BLOCK_SIZE = 1024;
    int arr_size = atoi(argv[1]);
    int n_block = div_ceil(arr_size, BLOCK_SIZE);

    dim3 blockDim(BLOCK_SIZE);
    dim3 gridDim(n_block);

    std::vector<int> a(arr_size);
    std::vector<int> b(arr_size);
    std::vector<int> c(arr_size);
    std::vector<int> result_c(arr_size);

    std::generate(a.begin(), a.end(), gen);
    std::generate(b.begin(), b.end(), gen);

    int* d_a;
    int* d_b;
    int* d_c;
    hipMalloc(&d_a, arr_size * sizeof(int));
    hipMalloc(&d_b, arr_size * sizeof(int));
    hipMalloc(&d_c, arr_size * sizeof(int));

    hipMemcpy(d_a, a.data(), arr_size * sizeof(int), hipMemcpyHostToDevice);
    hipMemcpy(d_b, b.data(), arr_size * sizeof(int), hipMemcpyHostToDevice);

    MyFirstKernel_b<<<gridDim, blockDim>>>(d_a, d_b, d_c, arr_size);

    // hipLaunchKernelGGL(MyFirstKernel_b, gridDim, blockDim, 0, 0, d_a, d_b, d_c, arr_size);
    hipDeviceSynchronize();

    hipMemcpy(c.data(), d_c, arr_size * sizeof(int), hipMemcpyDeviceToHost);

    matadd_naive(a, b, result_c);
    compare_result(c, result_c);

    // Free device memory
    hipFree(d_a);
    hipFree(d_b);
    hipFree(d_c);

    return 0;
}
