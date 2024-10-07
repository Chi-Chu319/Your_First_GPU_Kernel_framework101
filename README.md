# Your_First_GPU_Kernel_framework101

to run
```
docker build . -t first_kernel
docker run -it  --device=/dev/kfd --device=/dev/dri first_kernel 
```