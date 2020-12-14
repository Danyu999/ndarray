#include <iostream>

/// This is what the add.ptx is compiled from
/// "nvcc add.cu --ptx"
extern "C" __global__ void sum(const float* x, const float* y, float* out, int count) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < count; i += blockDim.x * gridDim.x) {
        out[i] = x[i] + y[i];
    }
}

/// From this PDF: https://www.fz-juelich.de/SharedDocs/Downloads/IAS/JSC/EN/slides/cuda/05-cuda-mm.pdf?__blob=publicationFile
/// Also, see this similar guide: https://www.tutorialspoint.com/cuda/cuda_matrix_multiplication.htm
/// It's called like so:
/// mm_kernel<<<dimGrid, dimBlock>>> (d_a, d_b, d_c, n);
/// Naive. Can this be done using shared memory?
/// Also, this accesses global memory (A and B) twice per loop. See: https://www.tutorialspoint.com/cuda/cuda_performance_considerations.htm
extern "C" __global__ void mm_kernel(float* A, float* B, float* C, int n) {
    // Grid stride: https://developer.nvidia.com/blog/cuda-pro-tip-write-flexible-kernels-grid-stride-loops/
    for (int col = blockIdx.x * blockDim.x + threadIdx.x; col < n; col += blockDim.x * gridDim.x) {
        for (int row = blockIdx.y * blockDim.y + threadIdx.y; row < n; row += blockDim.y * gridDim.y) {
            float sum = 0.0f;
            for (int i = 0; i < n; ++i) {
                sum += A[row * n + i] * B[i * n + col];
            }
            C[row * n + col] = sum;
        }
    }
}

/// Inspiration: https://stackoverflow.com/questions/18997773/non-square-matrix-multiplication-in-cuda
#define TILE_DIM 16
extern "C" __global__ void mm_noshared(float* A, float* B, float* C, int ARows, int ACols, int BRows, int BCols, int CRows, int CCols) {

    float CValue = 0;

    int Row = blockIdx.y*TILE_DIM + threadIdx.y;
    int Col = blockIdx.x*TILE_DIM + threadIdx.x;

    for (int k = 0; k < (TILE_DIM + ACols - 1)/TILE_DIM; k++) {

        for (int n = 0; n < TILE_DIM; ++n)
            if ((k*TILE_DIM + n < ACols && Row < ARows) && (k*TILE_DIM + n < BRows && Col < BCols))
                CValue += A[Row*ACols + k*TILE_DIM + n] * B[(k*TILE_DIM + n)*BCols + Col];

    }

    if (Row < CRows && Col < CCols) C[((blockIdx.y * blockDim.y + threadIdx.y)*CCols)+(blockIdx.x*blockDim.x)+threadIdx.x]=CValue;
}

/// From: https://stackoverflow.com/questions/18815489/cuda-tiled-matrix-matrix-multiplication-with-shared-memory-and-matrix-size-whic
/// This does use shared memory!!!
/// Untested.
__global__ void MatMul(float* A, float* B, float* C, int ARows, int ACols, int BRows,
                       int BCols, int CRows, int CCols)
{
    float CValue = 0;

    int Row = blockIdx.y*TILE_DIM + threadIdx.y;
    int Col = blockIdx.x*TILE_DIM + threadIdx.x;

    __shared__ float As[TILE_DIM][TILE_DIM];
    __shared__ float Bs[TILE_DIM][TILE_DIM];

    for (int k = 0; k < (TILE_DIM + ACols - 1)/TILE_DIM; k++) {

        if (k*TILE_DIM + threadIdx.x < ACols && Row < ARows)
            As[threadIdx.y][threadIdx.x] = A[Row*ACols + k*TILE_DIM + threadIdx.x];
        else
            As[threadIdx.y][threadIdx.x] = 0.0;

        if (k*TILE_DIM + threadIdx.y < BRows && Col < BCols)
            Bs[threadIdx.y][threadIdx.x] = B[(k*TILE_DIM + threadIdx.y)*BCols + Col];
        else
            Bs[threadIdx.y][threadIdx.x] = 0.0;

        __syncthreads();

        for (int n = 0; n < TILE_DIM; ++n)
            CValue += As[threadIdx.y][n] * Bs[n][threadIdx.x];

        __syncthreads();
    }

    if (Row < CRows && Col < CCols)
        C[((blockIdx.y * blockDim.y + threadIdx.y)*CCols) +
          (blockIdx.x * blockDim.x)+ threadIdx.x] = CValue;
}

/// TEST -- NOT EXPORTED
/// CUDA kernel to add elements of two arrays
__global__ void add(int n, float *x, float *y) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < n; i += stride)
        y[i] = x[i] + y[i];
}

/// See: https://developer.nvidia.com/blog/even-easier-introduction-cuda/
int main(void) {
    int shape = 200;
    int N = shape * shape;
    float *x, *y, *out;

    // Allocate Unified Memory -- accessible from CPU or GPU
    // See: https://developer.nvidia.com/blog/unified-memory-cuda-beginners/
    cudaMallocManaged(&x, N*sizeof(float));
    cudaMallocManaged(&y, N*sizeof(float));
    cudaMallocManaged(&out, N*sizeof(float));

    // initialize x and y arrays on the host
    for (int i = 0; i < N; i++) {
        x[i] = 1.0f;
        y[i] = 2.0f;
    }

    // Launch kernel on 1M elements on the GPU
    dim3 numBlocks(64, 64);
    dim3 dimBlock(16, 16);
    mm_kernel<<<numBlocks, dimBlock>>>(x, y, out, shape);

    // Wait for GPU to finish before accessing on host
    cudaDeviceSynchronize();

    // Print out the matrix
    for (int i = 0; i < N; ++i) {
        std::cout << out[i] << " ";
    }

    // Free memory
    cudaFree(x);
    cudaFree(y);

    return 0;
}