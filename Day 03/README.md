# Day 3: Matrix Multiplication

## Key Learnings
I learned how to write a CUDA kernel to perform matrix multiplication. Learned how to use 2D thread indexing with row and col, and launched the kernel using dim3 for grid and block sizes. 

Each thread computes one output element in dC. So each threads does N + N = 2N global memory reads for each computation, which for a warp results in 2N * 32 = 64N global memory reads, which is not very efficient