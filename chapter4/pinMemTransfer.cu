#include <cuda_runtime.h>
#include <stdio.h>

int main(int argc,char **argv){
    
    // set up device
    int dev = 0;
    cudaSetDevice(dev);

    // memory size
    unsigned int isize = 1<<22;
    unsigned int nbytes = isize * sizeof(float);

    // get device information
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp,dev);

    printf("%s starting at ",argv[0]);
    printf("device %d: %s memory size %d nbyte %5.2f MB\n",dev,deviceProp.name,
        isize,nbytes/(1024.0f*1024.0f));

    // allocate pinned host memory
    float *h_a;
    cudaMallocHost((float**)&h_a,nbytes);

    // allocate the device memory
    // Attention the different definition format between the malloc & cuadMalloc
    float *d_a;
    cudaMalloc((float **)&d_a,nbytes);

    // initialize the host memory
    for(unsigned int i = 0; i < isize; i++) h_a[i] = .5f;

    // transfer data from the host to the device
    cudaMemcpy(d_a,h_a,nbytes,cudaMemcpyHostToDevice);

    // transfer data from device to the host
    cudaMemcpy(h_a,d_a,nbytes,cudaMemcpyDeviceToHost);

    // free
    cudaFree(d_a);
    cudaFreeHost(h_a);

    // reset
    cudaDeviceReset();
    return EXIT_SUCCESS;
    }

    
