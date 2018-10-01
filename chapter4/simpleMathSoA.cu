#include <cuda_runtime.h>
#include <stdio.h>
#include <sys/time.h>

#define LEN 1<<22

double seconds(){
    struct timeval tp;
    struct timezone tzp;
    int i = gettimeofday(&tp,&tzp);
    return ((double)tp.tv_sec+(double)tp.tv_usec*1.e-6);
    }

struct InnerArray{
    float x[LEN];
    float y[LEN];
    };

void initialInnerArray(InnerArray *ip, int size){
    for (int i = 0; i < size; i++){
        ip->x[i] = (float)(rand() & 0xFF)/100.0f;
        ip->y[i] = (float)(rand() & 0xFF)/100.0f;
        }
    return;
    }

void testInnerArrayHost(InnerArray *A, InnerArray *C, const int n){
    for (int idx = 0; idx < n; idx ++){
        C->x[idx] = A->x[idx] + 10.f;
        C->y[idx] = A->y[idx] + 20.f;
        }
    return;
    }

void testInnerArrayHost2(InnerArray *A,InnerArray *C, const int n){
    // used for testify that ip->x[i] is equal to (*ip).x[i]
    for (int idx = 0;idx < n; idx++){
        (*C).x[idx] = (*A).x[idx] + 10.f;
        (*C).y[idx] = (*A).y[idx] + 20.f;
        }
    return;
    }

void printfHostResult(InnerArray *C, const int n){
    for (int idx = 0; idx < n; idx ++){
        printf("printout idx %d: x %f y %f \n",idx,C->x[idx],C->y[idx]);
        }
    return;
    }

void checkInnerArray(InnerArray *hostRef, InnerArray *gpuRef, const int N){
    double epsilon = 1.0e-8;
    bool match = 1;

    for(int i=0; i<N; i++){
        if (abs(hostRef->x[i] - gpuRef->x[i])>epsilon){
            match = 0;
            printf("different on x %dth element: host %f gpu %f \n",i,hostRef->x[i],gpuRef->x[i]);
            break;
            }

        if (abs(hostRef->y[i] - gpuRef->y[i])>epsilon){
            match = 0;
            printf("different on y %dth element: host %f gpu %f \n",i,hostRef->y[i],gpuRef->y[i]);
            break;
            }
        }
    if (!match) printf("Arrays do not match.\n\n");
    }

__global__ void testInnerArray(InnerArray *data, InnerArray *result, const int n){
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i<n){
        float tmpx = data->x[i];
        float tmpy = data->y[i];

        tmpx += 10.0f;
        tmpy += 20.0f;
        result->x[i] = tmpx;
        result->y[i] = tmpy;
        }
    }

__global__ void warmup(InnerArray *data, InnerArray *result, const int n){
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i<n){
        float tmpx = data->x[i];
        float tmpy = data->y[i];

        tmpx += 10.0f;
        tmpy += 20.0f;
        result->x[i] = tmpx;
        result->y[i] = tmpy;
        }
    }

int main(int argc, char ** argv){
    // set up device
    int dev = 0;
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp,dev);
    printf("%s test struct of array at ",argv[0]);
    printf("device %d: %s \n\n",dev,deviceProp.name);
    cudaSetDevice(dev);

    // allocate host memory
    int nElem = LEN;
    size_t nBytes = sizeof(InnerArray);
    InnerArray *h_A = (InnerArray *)malloc(nBytes);
    InnerArray *hostRef = (InnerArray *)malloc(nBytes);
    InnerArray *gpuRef = (InnerArray *)malloc(nBytes);
    InnerArray *hostRef2 = (InnerArray *)malloc(nBytes);

    // initialize host array
    initialInnerArray(h_A,nElem);
    testInnerArrayHost(h_A,hostRef,nElem);
    testInnerArrayHost(h_A,hostRef2,nElem);

    checkInnerArray(hostRef,hostRef2,nElem);

    // allocate memory on device
    InnerArray *d_A,*d_C;
    cudaMalloc((InnerArray**)&d_A,nBytes);
    cudaMalloc((InnerArray**)&d_C,nBytes);
    
    // copy data from host to device
    cudaMemcpy(d_A,h_A,nBytes,cudaMemcpyHostToDevice);

    // set up blocksize
    int blocksize = 128;

    if (argc>1) blocksize = atoi(argv[1]);

    // execution config
    dim3 block (blocksize,1);
    dim3 grid((nElem+block.x-1)/block.x,1);

    // kernel 1
    double iStart = seconds();
    warmup<<<grid,block>>>(d_A,d_C,nElem);
    cudaDeviceSynchronize();
    double iElaps = seconds() - iStart;
    printf("warmup <<<%3d,%3d>>> elapsed %f sec \n",grid.x,block.x,iElaps);
    
    // kernel 2
    iStart = seconds();
    testInnerArray<<<grid,block>>>(d_A,d_C,nElem);
    cudaDeviceSynchronize();
    iElaps = seconds() - iStart;
    printf("innerarray <<<%3d,%3d>>> elapsed %f sec \n",grid.x,block.x,iElaps);
    cudaMemcpy(gpuRef,d_C,nBytes,cudaMemcpyDeviceToHost);
    checkInnerArray(hostRef,gpuRef,nElem);
    cudaGetLastError();

    // free memories
    cudaFree(d_A);
    cudaFree(d_C);
    free(h_A);
    free(hostRef);
    free(hostRef2);
    free(gpuRef);

    cudaDeviceReset();
    return EXIT_SUCCESS;
    }



    



