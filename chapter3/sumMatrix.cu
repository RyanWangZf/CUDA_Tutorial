#include <cuda_runtime.h>
#include <stdio.h>
#include <sys/time.h>

double seconds(){
    struct timeval tp;
    struct timezone tzp;
    gettimeofday(&tp,&tzp);
    return((double)tp.tv_sec + (double)tp.tv_usec*1.e-6);
    }

void initialData(float *ip,int size){
    time_t t;
    srand((unsigned int) time(&t));
    for (int i=0;i<size;i++){
        ip[i] = (float)(rand() & 0xFF)/10.0f;
        }
    }

void checkResult(float *hostRef,  float *gpuRef, const int N){
    double epsilon = 1.e-8;
    bool match = 1;
    for (int i=0;i<N;i++){
        if (abs(hostRef[i]-gpuRef[i])>epsilon){
            match = 0;
            printf("Matrix do not match! \n");
            printf("host %5.2f gpu % 5.2f at current %d",hostRef[i],gpuRef[i],i);
            break;
            }
        }
    if (match) printf("Matrix match.\n\n");
    }

void sumMatrixOnHost(float *A, float *B, float *C, int NX, int NY){
    float *ia = A;
    float *ib = B;
    float *ic = C;

    for (int iy = 0;iy < NY;iy ++){
        for (int ix = 0;ix < NX;ix++){
            ic[ix] = ia[ix] + ib[ix];
            }
        ia += NX; ib += NX; ic += NX;
        }
}

__global__ void sumMatrixOnGPU2D(float *A, float *B, float *C, int NX, int NY)
{
    unsigned int ix = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int iy = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int idx = iy * NX + ix;

    if (ix < NX && iy < NY)
    {
        C[idx] = A[idx] + B[idx];
    }
}

int main(int argc, char **argv)
{
    // set up device
    int dev = 0;
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, dev);
    cudaSetDevice(dev);

    // set up data size of matrix
    int nx = 1 << 13;
    int ny = 1 << 13;

    int nxy = nx * ny;
    int nBytes = nxy * sizeof(float);

    // malloc host memory
    float *h_A, *h_B, *hostRef, *gpuRef;
    h_A = (float *)malloc(nBytes);
    h_B = (float *)malloc(nBytes);
    hostRef = (float *)malloc(nBytes);
    gpuRef = (float *)malloc(nBytes);

    // initialize data at host side
    double iStart = seconds();
    initialData(h_A, nxy);
    initialData(h_B, nxy);
    double iElaps = seconds() - iStart;
    printf("initialize data elapsed %f sec\n",iElaps);

    memset(hostRef, 0, nBytes);
    memset(gpuRef, 0, nBytes);

    // add matrix at host side for result checks
    iStart = seconds();
    sumMatrixOnHost (h_A, h_B, hostRef, nx, ny);
    iElaps = seconds() - iStart;
    printf("sumMatrixOnHost elapsed %f sec \n ",iElaps);

    // malloc device global memory
    float *d_MatA, *d_MatB, *d_MatC;
    cudaMalloc((void **)&d_MatA, nBytes);
    cudaMalloc((void **)&d_MatB, nBytes);
    cudaMalloc((void **)&d_MatC, nBytes);

    // transfer data from host to device
    cudaMemcpy(d_MatA, h_A, nBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_MatB, h_B, nBytes, cudaMemcpyHostToDevice);

    // invoke kernel at host side
    int dimx = 32;
    int dimy = 32;

    if(argc > 2)
    {
        dimx = atoi(argv[1]);
        dimy = atoi(argv[2]);
    }

    dim3 block(dimx, dimy);
    dim3 grid((nx + block.x - 1) / block.x, (ny + block.y - 1) / block.y);

    // execute the kernel
    // CHECK(cudaDeviceSynchronize());
    iStart = seconds();
    sumMatrixOnGPU2D<<<grid, block>>>(d_MatA, d_MatB, d_MatC, nx, ny);
    cudaDeviceSynchronize();
    iElaps = seconds() - iStart;
    printf("sumMatrixOnGPU2D <<<(%d,%d), (%d,%d)>>> elapsed %f sec\n", grid.x,
    grid.y,block.x, block.y, iElaps);
    cudaGetLastError();

    // copy kernel result back to host side
    cudaMemcpy(gpuRef, d_MatC, nBytes, cudaMemcpyDeviceToHost);

    // check device results
    checkResult(hostRef, gpuRef, nxy);

    // free device global memory
    cudaFree(d_MatA);
    cudaFree(d_MatB);
    cudaFree(d_MatC);

    // free host memory
    free(h_A);
    free(h_B);
    free(hostRef);
    free(gpuRef);

    // reset device
    cudaDeviceReset();

    return EXIT_SUCCESS;
}




