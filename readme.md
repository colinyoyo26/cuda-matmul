# CUDA-MATMUL

## INPUT

Three integer `N`, `S1`, `S2`,  then generate `N` by `N` matrix `A`, `B` from random seed `S1`, `S2` respectively.

`N` should be multiple of 64.

## OUTPUT

Signature of `C`, where `C = AB`

> hash elements of `C` to a integer.

## Test

`$ ./test.sh X`

> `X` is TILING or ELE or ELEPACK or ROW or ROWPACK

## Implement

Following are 6 different version of matmul implementation.

### ROW

Each thread calculate a row of `C`.

``` cpp
__global__ void matmul(uint32_t A[], uint32_t B[], uint32_t C[], uint32_t N)
{
    int tid = threadIdx.x, blksz = blockDim.x, row = blockIdx.x * blksz + tid;
    for (int i = 0; i < N; i++){
        for (int j = 0; j < N; j++) 
            C[row * N + i] += A[row * N + j] * B[j * N + i];
}
```

Cache `sum` in register may speed up a lot. 

```cpp
__global__ void matmul(uint32_t A[], uint32_t B[], uint32_t C[], uint32_t N)
{
    int tid = threadIdx.x, blksz = blockDim.x, row = blockIdx.x * blksz + tid;
    for (int i = 0; i < N; i++){
        uint32_t sum = 0; 
        for (int j = 0; j < N; j++) 
            sum += A[row * N + j] * B[j * N + i];
        C[row * N + i] = sum;
    }
}
```

### ROWPACK

Threads in a block access the same column of `B`, so we can cache it to shared memory.

```cpp
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
```

### ELE

Each thread calculate a element of `C`
```cpp
__global__ void matmul(uint32_t A[], uint32_t B[], uint32_t C[], uint32_t N)
{
    int tid = threadIdx.x, blksz = blockDim.x, idx = blockIdx.x * blksz + tid;
    int row = idx / N, col = idx % N;
    uint32_t sum = 0;
    for (int i = 0; i < N; i++)
        sum += A[row * N + i] * B[i * N + col];
    C[idx] = sum;
}
```

### ELEPACK

Threads in a block access the same row of `A`, so we can cache it to shared memory.

```cpp
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
```

### TILING 

Tiling make it possible to share both column and row among threads in the same block.

```cpp
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
```
