#include "../common/common.h"
#include <cuda_runtime.h>
#include <stdio.h>

//sum 2 arrays with GPU

void checkResult(float *hostRef, float *gpuRef, const int N)
{
    double epsilon = 1.0E-8;
    bool match = 1;

    for (int i = 0; i < N; i++)
    {
        if (abs(hostRef[i] - gpuRef[i]) > epsilon)
        {
            match = 0;
            printf("Arrays do not match!\n");
            printf("host %5.2f gpu %5.2f at current %d\n", hostRef[i],
                   gpuRef[i], i);
            break;
        }
    }

    if (match) printf("Arrays match.\n\n");

    return;
}

void initialData(float *ip,int size){
	//generate different seed for random number
	time_t t;
	srand((unsigned int) time(&t));
	
	for (int i=0;i<size;i++){
		ip[i] = (float)(rand() & 0xFF) / 10.0f;
		}
	}

void sumArraysOnHost(float *A,float *B,float *C,const int N){
    for (int idx=0;idx<N;idx++){
		C[idx] = A[idx] + B[idx];}
	}

__global__ void sumArraysOnGPU(float *A,float *B,float *C,const int N){
    int i = threadIdx.x;
    if (i<N) C[i] = A[i] + B[i];
    }

int main(int argc, char **argv){
	// set up device
    int dev = 0;
    CHECK(cudaSetDevice(dev));
    
    int nElem = 1 << 5; // 2**5
    
    //size_t equals long int on 64bit machines, for count
	size_t nBytes = nElem * sizeof(float);
    float *h_A,*h_B,*hostRef,*gpuRef;
    
    // allocate mem on CPU
    h_A = (float*)malloc(nBytes);
    h_B = (float*)malloc(nBytes);
    hostRef = (float*)malloc(nBytes);
    gpuRef = (float*)malloc(nBytes);

    initialData(h_A,nElem);
    initialData(h_B,nElem);
    
    memset(hostRef,0,nBytes);
    memset(gpuRef,0,nBytes);

    // allocate mem on GPU
	float *d_A,*d_B,*d_C;
    CHECK(cudaMalloc((float **) &d_A, nBytes));
    CHECK(cudaMalloc((float **) &d_B, nBytes));
    CHECK(cudaMalloc((float **) &d_C, nBytes));
    
    // copy mem from host to dev
    cudaMemcpy(d_A,h_A,nBytes,cudaMemcpyHostToDevice);
    cudaMemcpy(d_B,h_B,nBytes,cudaMemcpyHostToDevice);
    cudaMemcpy(d_C,gpuRef,nBytes,cudaMemcpyHostToDevice);
    
    // ivoke kernel at host side
    dim3 block(nElem);
    dim3 grid(1);
    sumArraysOnGPU<<<grid,block>>>(d_A,d_B,d_C,nElem);
    printf("Execution configure <<<%d,%d>>>\n",grid.x,block.x);

    // GPU result from d_C to host
    CHECK(cudaMemcpy(gpuRef,d_C,nBytes,cudaMemcpyDeviceToHost));
     
	// compute at the host side
	sumArraysOnHost(h_A,h_B,hostRef,nElem);
	
    checkResult(hostRef,gpuRef,nElem);
	free(h_A);
	free(h_B);
	free(hostRef);
    free(gpuRef);

	cudaFree(d_A);  
	cudaFree(d_B);
	cudaFree(d_C);

    return(0);
	}

		
