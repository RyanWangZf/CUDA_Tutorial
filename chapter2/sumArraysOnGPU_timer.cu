#include "../common/common.h"
#include <cuda_runtime.h>
#include <stdio.h>
#include <time.h>

/*
// get CPU now seconds, tv_sec is the seconds and tv_usec is the microseconds
double seconds(){
    struct timeval tp;
    struct timezone tzp;
    int i = gettimeofday(&tp,&tzp);
    return ((double)tp.tv_sec + (double)tp.tv_usec*1.e-6);
    }
*/

// sum 2 arrays with GPU
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
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i<N) C[i] = A[i] + B[i];
    }

int main(int argc, char **argv){
    printf("%s Starting..\n",argv[0]);

	// set up device
    int dev = 0;
    cudaDeviceProp deviceProp;
    CHECK(cudaGetDeviceProperties(&deviceProp,dev));
    printf("Using Device %d: %s\n",dev,deviceProp.name);
    CHECK(cudaSetDevice(dev));
    
    // set up data size of vectors
    int nElem = 1 << 24; // 2**24
    printf("Vector size %d \n",nElem);

    //size_t equals long int on 64bit machines, for count
	size_t nBytes = nElem * sizeof(float);
    float *h_A,*h_B,*hostRef,*gpuRef;
    
    // allocate mem on CPU
    h_A = (float*)calloc(nElem,sizeof(float));
    h_B = (float*)calloc(nElem,sizeof(float));
    hostRef = (float*)calloc(nElem,sizeof(float));
    gpuRef = (float*)calloc(nElem,sizeof(float));
    
    // timer data 
    double iStart,iElaps;
    
    // intialize data at host side
    initialData(h_A,nElem);
    initialData(h_B,nElem);
	
    // compute at the host side
	iStart = seconds();
    sumArraysOnHost(h_A,h_B,hostRef,nElem);
    iElaps = seconds() - iStart;
    printf("Execution on the Host side, cost %f sec\n",iElaps); 
    
    // allocate mem on GPU
	float *d_A,*d_B,*d_C;
    CHECK(cudaMalloc((float **) &d_A, nBytes));
    CHECK(cudaMalloc((float **) &d_B, nBytes));
    CHECK(cudaMalloc((float **) &d_C, nBytes));
    
    // copy mem from host to dev
    cudaMemcpy(d_A,h_A,nBytes,cudaMemcpyHostToDevice);
    cudaMemcpy(d_B,h_B,nBytes,cudaMemcpyHostToDevice);
    cudaMemcpy(d_C,gpuRef,nBytes,cudaMemcpyHostToDevice);
    
    // invoke kernel at host side
    int iLen = 512;
    dim3 block(iLen);
    dim3 grid((nElem+ block.x-1)/block.x);
    
    // count the GPU time elaps
    iStart = seconds();
    sumArraysOnGPU<<<grid,block>>>(d_A,d_B,d_C,nElem);
    cudaDeviceSynchronize();
    iElaps = seconds() - iStart;
    
    printf("Execution configure <<<%d,%d>>>\n Time elapsed %f"\
        "sec\n",grid.x,block.x,iElaps);
             
    // GPU result from d_C to host
    CHECK(cudaMemcpy(gpuRef,d_C,nBytes,cudaMemcpyDeviceToHost));
    checkResult(hostRef,gpuRef,nElem);
    
	// free host memory
    free(h_A);
	free(h_B);
	free(hostRef);
    free(gpuRef);
    
    // free device memory
	cudaFree(d_A);  
	cudaFree(d_B);
	cudaFree(d_C);

    return(0);
	}

		
