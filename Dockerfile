# Dockerfile for HIP programming on ROCm environment

FROM rocm/tensorflow:latest

USER root

# Set working directory
WORKDIR /workspace

# Copy the current directory contents into the container at /workspace
COPY . /workspace

# Compile the HIP source code (assuming the main file is main.cpp)
# Replace 'main.cpp' with the name of your HIP source file if different
RUN /opt/rocm/bin/hipcc --amdgpu-target=gfx942 -o main main.cpp

# Set the command to run your HIP program
# CMD ["./main"]
CMD ["./main", "2000"]