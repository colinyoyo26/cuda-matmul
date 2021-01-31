#include <cuda.h>
#include <stdint.h>
#include <stdio.h>
#include <time.h>

#define MAXN 8192
#define MAXMAT MAXN *MAXN

#ifdef TILING
#define BLKW 16
#define BLKH BLKW
#define BLKDIM dim3(BLKW, BLKH)
#define GRIDDIM dim3(N / BLKW, N / BLKH)
#define TILEN BLKW
__global__ void matmul(uint32_t A[], uint32_t B[], uint32_t C[], uint32_t N)
{
    int blkc = blockIdx.x, blkr = blockIdx.y;
    int tidc = threadIdx.x, tidr = threadIdx.y; 
    int row = blkr * TILEN + tidr, col = blkc * TILEN + tidc;
    __shared__ uint32_t packA[TILEN][TILEN], packB[TILEN][TILEN];
    uint32_t sum = 0; 
    for (int i = 0; i < N / TILEN; i++) { 
        packA[tidr][tidc] = A[row * N + i * TILEN + tidc];
        packB[tidr][tidc] = B[(i * TILEN + tidr) * N + col];
        __syncthreads();
#pragma unroll
        for (int j = 0; j < TILEN; j++)
            sum += packA[tidr][j] * packB[j][tidc];
        __syncthreads();
    }
    C[row * N + col] = sum;
}

#elif ELEPACK
#define BLKSZ 64
#define BLKDIM dim3(BLKSZ)
#define GRIDDIM dim3(N *N / BLKSZ)
__global__ void matmul(uint32_t A[], uint32_t B[], uint32_t C[], uint32_t N)
{
    int tid = threadIdx.x, blksz = blockDim.x, idx = blockIdx.x * blksz + tid;
    int row = idx / N, col = idx % N;
    __shared__ uint32_t packA[MAXN];
    for (int i = tid; i < N; i += blksz)
        packA[i] = A[row * N + i];
    __syncthreads();
    uint32_t sum = 0; 
    for (int i = 0; i < N; i++) 
        sum += packA[i] * B[i * N + col];
    C[idx] = sum;
}

#elif ELE
#define BLKSZ 256
#define BLKDIM dim3(BLKSZ)
#define GRIDDIM dim3(N *N / BLKSZ)
__global__ void matmul(uint32_t A[], uint32_t B[], uint32_t C[], uint32_t N)
{
    int tid = threadIdx.x, blksz = blockDim.x, idx = blockIdx.x * blksz + tid;
    int row = idx / N, col = idx % N;
    uint32_t sum = 0; 
    for (int i = 0; i < N; i++) 
        sum += A[row * N + i] * B[i * N + col];
    C[idx] = sum;
}

#elif ROWPACK
#define BLKSZ 64
#define BLKDIM dim3(BLKSZ)
#define GRIDDIM dim3(N / BLKSZ)
__global__ void matmul(uint32_t A[], uint32_t B[], uint32_t C[], uint32_t N)
{
    int tid = threadIdx.x, blksz = blockDim.x, row = blockIdx.x * blksz + tid;
    __shared__ uint32_t packB[MAXN];
    for (int i = 0; i < N; i++) {
        for (int j = tid; j < N; j += blksz)
            packB[j] = B[j * N + i];
        __syncthreads();
        uint32_t sum = 0; 
        for (int j = 0; j < N; j++) 
            sum += A[row * N + j] * packB[j];
        C[row * N + i] = sum;
    }
}

#else
#define BLKSZ 64
#define BLKDIM dim3(BLKSZ)
#define GRIDDIM dim3(N / BLKSZ)
__global__ void matmul(uint32_t A[], uint32_t B[], uint32_t C[], uint32_t N)
{
    int tid = threadIdx.x, blksz = blockDim.x, row = blockIdx.x * blksz + tid;
    for (int i = 0; i < N; i++) {
        uint32_t sum = 0; 
        for (int j = 0; j < N; j++) 
            sum += A[row * N + j] * B[j * N + i];
        C[row * N + i] = sum;
    }
}
#endif

__global__ void print_sig(uint32_t A[], uint32_t N)
{
    uint32_t h = 0; 
    for (int i = 0; i < N * N; i++)
        h = (h + A[i]) * 2654435761LU;
    printf("Result: \n%u\n", h);
}

static inline void rand_gen(uint32_t c, int N, uint32_t A[])
{
    uint32_t x = 2, mod = N * N;
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++) { 
            x = (x * x + c + i + j) & (mod - 1);
            A[i * N + j] = x;
        }
}


static inline void init_all(uint32_t &N,
                            uint32_t &S1,
                            uint32_t &S2,
                            uint32_t A[],
                            uint32_t B[],
                            uint32_t **cuA,
                            uint32_t **cuB,
                            uint32_t **cuC)
{
#define SIZE (size_t)(N * N * sizeof(uint32_t))
    if (scanf("%u %u %u", &N, &S1, &S2) != 3)
        exit(1);
    rand_gen(S1, N, A);
    rand_gen(S2, N, B);
    cudaMalloc(cuA, SIZE);
    cudaMalloc(cuB, SIZE);
    cudaMalloc(cuC, SIZE);
    cudaMemcpy(*cuA, A, SIZE, cudaMemcpyHostToDevice);
    cudaMemcpy(*cuB, B, SIZE, cudaMemcpyHostToDevice);
    cudaMemset(*cuC, 0, SIZE);
#undef SIZE
}

static inline long get_time()
{
    struct timespec t;
    clock_gettime(CLOCK_REALTIME, &t);
    return (long) (t.tv_sec * 1e9 + t.tv_nsec);
}

static uint32_t A[MAXMAT], B[MAXMAT];
int main()
{
    uint32_t N, S1, S2, *cuA, *cuB, *cuC;
    init_all(N, S1, S2, A, B, &cuA, &cuB, &cuC);

    long t_start = get_time();
    matmul<<<GRIDDIM, BLKDIM>>>(cuA, cuB, cuC, N);
    long t_total = get_time() - t_start;
    print_sig<<<dim3(1), dim3(1)>>>(cuC, N);
    fprintf(stderr, "matmul time: \n%ld ns\n", t_total);
    cudaFree(cuA);
    cudaFree(cuB);
    cudaFree(cuC);
    return 0;
}
