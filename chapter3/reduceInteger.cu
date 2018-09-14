#include <stdio.h>
#include <cuda_runtime.h>
#include <sys/time.h>

double seconds(){
    struct timeval tp;
    struct timezone tzp;
    int i = gettimeofday(&tp,&tzp);
    return ((double)tp.tv_sec+(double)tp.tv_usec*1.e-6);
    }

int recursiveReduce(int *data, int const size){
    // terminate check
    if (size == 1) return data[0];

    // renew stride
    int const stride = size / 2;

    // in-place reduction
    for (int i = 0; i < stride; i++){
        data[i] += data[i+stride];
        }

    return recursiveReduce(data,stride);
    }

// kernel 1
__global__ void reduceNeighbored(int *g_idata, int *g_odata, unsigned int n){
    /*
    Do calculations on each block, save partly reduced result on the series of g_odata.
    After it, sum g_odata up for getting the last result on the host side.
    */
    
    // set thread ID
    unsigned int tid = threadIdx.x;
    unsigned int idx = threadIdx.x + blockDim.x * blockIdx.x;

    // convert global data pointer to the local pointer of this block
    int *idata = g_idata + blockIdx.x * blockDim.x;

    // boundary check
    if (idx >= n) return;

    // in-place reduction in global memory on each block
    for (int stride = 1; stride < blockDim.x; stride *= 2){
        if ((tid % (2 * stride)) == 0){
            idata[tid] += idata[tid+stride];
            }
        // synchronize within block
        __syncthreads();
        }

    // write result for this block to global memory
    if (tid == 0) g_odata[blockIdx.x] = idata[0];
    }

// kernel 2
__global__ void reduceNeighboredLess(int *g_idata,int *g_odata,unsigned int n){
    // set threadID
    unsigned int tid = threadIdx.x;
    unsigned int idx = threadIdx.x + blockDim.x * blockIdx.x;

    // convert global data pointer to the local pointer of this block
    int *idata = g_idata + blockIdx.x*blockDim.x;

    // boundary check
    if (idx >= n) return;

    // in-place reduction
    for (int stride = 1; stride < blockDim.x; stride *= 2){
        // convert tid into local array index
        int index = 2 * stride * tid;
        if (index < blockDim.x){
            idata[index] += idata[index+stride];
            }
        // synchronize within threadblock
        __syncthreads();
        }
     // write the result for this block to global memory
     if (tid == 0) g_odata[blockIdx.x] = idata[0];
     }

// kernel 3
__global__ void reduceInterleaved (int *g_idata, int *g_odata, unsigned int n){
    // interleaved pair implementation with less divergence
    unsigned int tid = threadIdx.x;
    unsigned int idx = threadIdx.x + blockDim.x * blockIdx.x;
    // convert global data pointer to the local pointer of this block
    int *idata = g_idata + blockIdx.x*blockDim.x;
    // boundary check
    if (idx >= n) return;
    // in-place reduction in global memory
    for (int stride = blockDim.x/2; stride > 0; stride >>= 1){
        if (tid < stride) idata[tid] += idata[tid+stride];
        __syncthreads();
        }
    // write result for this block to global memory
    if (tid == 0) g_odata[blockIdx.x] = idata[0];
}

// kernel 4
__global__ void reduceUnrolling2(int *g_idata,int *g_odata, unsigned int n){
    // set thread ID, one threadblock operates on two datablocks
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x * 2 + threadIdx.x;    
    // convert global data pointer to the local pointer of this block
    int *idata = g_idata + blockIdx.x * blockDim.x * 2;
    // unrolling 2 data blocks
    if (idx + blockDim.x < n) g_idata[idx] += g_idata[idx+blockDim.x];
    __syncthreads();
    // inplace reduction in global memory
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1){
        if (tid < stride){
            idata[tid] += idata[tid+stride];
            }
        // synchronize within threadBlock
        __syncthreads();
        }
    // write result for this block to global memory
    if (tid ==0) g_odata[blockIdx.x] = idata[0];
    }

int main(int argc, char** argv){
    // set up device
    int dev = 0;
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp,dev);
    printf("%s starting reduction at ",argv[0]);
    printf("device %d: %s ",dev,deviceProp.name);
    cudaSetDevice(dev);

    bool bResult = false;

    // initialization
    int size = 1 << 24; // total number of elementse to reduce
    printf("reduce with array size %d ",size);

    // execution configuration
    int blocksize = 512;
    if (argc > 1) blocksize = atoi(argv[1]);
    
    dim3 block(blocksize,1);
    dim3 grid ((size+block.x-1)/block.x,1);
    printf("grid %d block %d\n",grid.x,block.x);
    
    // allocate host memory
    size_t bytes = size * sizeof(int);
    int *h_idata = (int *) malloc(bytes);
    int *h_odata = (int *) malloc(grid.x*sizeof(int));
    int *tmp     = (int *) malloc(bytes);

    // initialize the array
    for (int i = 0; i <size; i++){
        // mask off high 2 bytes to force max number to 255
        h_idata[i] = (int)(rand() & 0xFF);
        }
    memcpy(tmp,h_idata,bytes);
    
    double iStart,iElaps;
    int gpu_sum = 0;

    // allocate device memory
    int *d_idata = NULL;
    int *d_odata = NULL;
    cudaMalloc((void **) &d_idata,bytes);
    cudaMalloc((void **) &d_odata,grid.x*sizeof(int));

    // cpu reduction
    iStart = seconds();
    int cpu_sum = recursiveReduce(tmp,size);
    iElaps = seconds() - iStart;
    printf("cpu reduce elapsed %f sec cpu_sum: %d \n",iElaps,cpu_sum);

    // kernel 1: reduceNeighbored
    cudaMemcpy(d_idata,h_idata,bytes,cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    iStart = seconds();
    reduceNeighbored<<<grid,block>>>(d_idata,d_odata,size);
    cudaDeviceSynchronize();
    iElaps = seconds() - iStart;
    cudaMemcpy(h_odata,d_odata,grid.x * sizeof(int),cudaMemcpyDeviceToHost);
    for (int i=0; i <grid.x; i++) gpu_sum += h_odata[i];
    printf("gpu Neighbored elapsed %f sec gpu_sum: %d <<<grid %d block %d>>>\n",
        iElaps,gpu_sum,grid.x,block.x);
    
    // kernel 2: reduceNeighbored with less divergence
    cudaMemcpy(d_idata,h_idata,bytes,cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    iStart = seconds();
    reduceNeighboredLess<<<grid,block>>>(d_idata,d_odata,size);
    cudaDeviceSynchronize();
    iElaps = seconds() - iStart;
    cudaMemcpy(h_odata,d_odata,grid.x * sizeof(int),cudaMemcpyDeviceToHost);
    gpu_sum = 0;
    for(int i = 0;i<grid.x;i++) gpu_sum += h_odata[i];
    printf("gpu Neighbored2 elapsed %f sec gpu_sum: %d <<<grid %d block %d>>>\n",
        iElaps,gpu_sum,grid.x,block.x);
    
    // kernel 3: reduceInterleaved
    cudaMemcpy(d_idata,h_idata,bytes,cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    iStart = seconds();
    reduceInterleaved<<<grid,block>>>(d_idata,d_odata,size);
    cudaDeviceSynchronize();
    iElaps = seconds() - iStart;
    cudaMemcpy(h_odata,d_odata,grid.x*sizeof(int),cudaMemcpyDeviceToHost);
    gpu_sum = 0;
    for(int i=0;i<grid.x;i++) gpu_sum += h_odata[i];
    printf("gpu Interleaved elaped %f sec gpu_sum %d <<<grid %d block %d>>>\n",
        iElaps,gpu_sum,grid.x,block.x);
    
    // kernel 4: reduceUnrolling2
    cudaMemcpy(d_idata,h_idata,bytes,cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    iStart = seconds();
    reduceUnrolling2<<<grid.x/2,block>>>(d_idata,d_odata,size);
    cudaDeviceSynchronize();
    iElaps = seconds() - iStart;
    cudaMemcpy(h_odata,d_odata,(grid.x/2)*sizeof(int),cudaMemcpyDeviceToHost);
    gpu_sum = 0;
    for(int i=0;i<grid.x/2;i++) gpu_sum += h_odata[i];
    printf("gpu Unrolling2 elaped %f sec gpu_sum %d <<<grid %d block %d>>>\n",
        iElaps,gpu_sum,grid.x/2,block.x);

    free(h_idata);
    free(h_odata);
    cudaFree(d_idata);
    cudaFree(d_odata);

    cudaDeviceReset();
    

    // check the results
    bResult = (gpu_sum == cpu_sum);
    if (!bResult) printf("Test Failed!\n");
    return EXIT_SUCCESS;
    }






