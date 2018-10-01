#include <cuda_runtime.h>
#include <stdio.h>
#include <sys/time.h>

#define LEN 1 << 22

// define Array of Struct (AoS)
struct innerStruct{
    float x;
    float y;
    };

// define Struct of Array (SoA)
struct innerArray{
    float x[LEN];
    float y[LEN];
    };


double seconds(){
    struct timeval tp;
    struct timezone tzp;
    int i = gettimeofday(&tp,&tzp);
    return ((double)tp.tv_sec+(double)tp.tv_usec*1.e-6);
    }

void initialInnerStruct(innerStruct *ip, int size){
    // initialize a array of struct (AoS)
    for (int i=0;i<size;i++){
        ip[i].x = (float)(rand() & 0xFF)/100.0f;
        ip[i].y = (float)(rand() & 0xFF)/100.0f;
        }
    return;
    }

void checkInnerStruct(innerStruct *hostRef, innerStruct *gpuRef, const int N){
    double epsilon = 1.0e-8;
    bool match = 1;

    for(int i = 0; i < N; i++){
        if (abs(hostRef[i].x - gpuRef[i].x)>epsilon){
            match = 0;
            printf("different on %dth element: host %f gpu %f\n",i,
                hostRef[i].x,gpuRef[i].x);
            break;
            }
        if (abs(hostRef[i].y - gpuRef[i].y)>epsilon){
            match = 0;
            printf("different on %dth element: host %f gpu %f\n",i,
                hostRef[i].y,gpuRef[i].y);
            break;
            }
        }
    if (!match) printf("Arrays do not match! \n\n");
    }

void testInnerStructHost(innerStruct *A,innerStruct *C,const int n){
    for (int idx = 0;idx < n; idx++){
        C[idx].x = A[idx].x + 10.f;
        C[idx].y = A[idx].y + 20.f;
        }
    return;
    }

__global__ void testInnerStruct(innerStruct *data, innerStruct *result, const int n){
    // test the array of struct (AoS)
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i<n){
        innerStruct tmp = data[i];
        tmp.x += 10.f;
        tmp.y += 20.f;
        result[i] = tmp;
        }
    }

__global__ void warmup(innerStruct *data, innerStruct *result, const int n){
    // warmup kernel function
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i<n){
        innerStruct tmp = data[i];
        tmp.x += 10.f;
        tmp.y += 20.f;
        result[i] = tmp;
        }
    }

int main(int argc,char **argv){
    // set up device
    int dev = 0;
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp,dev);
    printf("%s test struct of array at ",argv[0]);
    printf("device %d: %s \n",dev,deviceProp.name);
    cudaSetDevice(dev);
    
    // allocate host memory
    int nElem = LEN;
    size_t nBytes = nElem * sizeof(innerStruct);
    innerStruct *h_A = (innerStruct *)malloc(nBytes);
    innerStruct *hostRef = (innerStruct *)malloc(nBytes);
    innerStruct *gpuRef = (innerStruct *)malloc(nBytes);

    // initialize host array
    initialInnerStruct(h_A,nElem);
    testInnerStructHost(h_A,hostRef,nElem);

    // allocate device memory
    innerStruct *d_A,*d_C;
    cudaMalloc((innerStruct**)&d_A,nBytes);
    cudaMalloc((innerStruct**)&d_C,nBytes);

    // copy data from host to device
    cudaMemcpy(d_A,h_A,nBytes,cudaMemcpyHostToDevice);

    int blocksize = 128;
    if (argc > 1) blocksize = atoi(argv[1]);

    // execution config
    dim3 block(blocksize,1);
    dim3 grid((nElem+block.x-1)/block.x,1);

    // kernel 1: warmup
    double iStart = seconds();
    warmup<<<grid,block>>>(d_A,d_C,nElem);
    cudaDeviceSynchronize();
    double iElaps = seconds() - iStart;
    printf("warmup kernel <<< %3d, %3d >>> elapsed %f sec\n",grid.x,block.x,iElaps);
    
    cudaMemcpy(gpuRef,d_C,nBytes,cudaMemcpyDeviceToHost);
    checkInnerStruct(hostRef,gpuRef,nElem);
    cudaGetLastError();

    // kernel 2: testInnerStruct
    iStart = seconds();
    testInnerStruct<<<grid,block>>>(d_A,d_C,nElem);
    cudaDeviceSynchronize();
    iElaps = seconds() - iStart;
    printf("innerstruct <<< %3d, %3d >>> elapsed %f sec\n",grid.x,block.x,iElaps);
    cudaMemcpy(gpuRef,d_C,nBytes,cudaMemcpyDeviceToHost);
    checkInnerStruct(hostRef,gpuRef,nElem);
    cudaGetLastError();

    // free memories
    cudaFree(d_A);
    cudaFree(d_C);
    free(h_A);
    free(hostRef);
    free(gpuRef);

    // reset devices
    cudaDeviceReset();
    return EXIT_SUCCESS;
    }

