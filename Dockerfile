# Dockerfile for HIP programming on ROCm environment

FROM rocm/rocm-terminal:latest


# Set working directory
WORKDIR /workspace

# Copy the current directory contents into the container at /workspace
COPY . /workspace

# Compile the HIP source code (assuming the main file is main.cpp)
# Replace 'main.cpp' with the name of your HIP source file if different
RUN /opt/rocm/bin/hipcc -o main main.cpp

# Set the command to run your HIP program
CMD ["sh", "-c", "./main > output.log 2>&1"]