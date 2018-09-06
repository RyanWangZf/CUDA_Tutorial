#include <stdio.h>
#include <sys/time.h>

// toolbox
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
    
// main functions
void sumMatrixOnHost(float *A,float *B,float *C,const int nx,const int ny){
    float *ia = A;
    float *ib = B;
    float *ic = C;

    for (int iy=0; iy<ny; iy++){
        for (int ix=0; ix<nx; ix++){
            ic[ix] = ia[ix] + ib[ix];
            }
            ia += nx; ib += nx; ic+=nx;
        }
    }

__global__ void sumMatrixOnGPU2D(float *MatA,float *MatB,float *MatC,int nx,int ny){
    unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int iy = threadIdx.y + blockIdx.y * blockDim.y;
    unsigned int idx = iy*nx + ix; // global linear memory location
    
    if (ix < nx && iy < ny) MatC[idx] = MatA[idx] + MatB[idx];
    }

int main(int argc, char **argv){

    printf("%s Starting...\n",argv[0]);
    // set up device
    int dev = 0;
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp,dev);
    printf("Using Device %d: %s \n",dev,deviceProp.name);
    cudaSetDevice(dev);

    // initialize the data
    int nx = 1<<11; // 2**11
    int ny = 1<<11;
    int nxy = nx * ny;
    int nBytes = nxy * sizeof(float);
    printf("Matrix size: nx %d ny %d \n",nx,ny);
    
    // malloc host memory
    float *h_A,*h_B,*hostRef,*gpuRef;
    h_A =(float *)malloc(nBytes);
    h_B =(float *)malloc(nBytes);
    hostRef =(float *)malloc(nBytes);
    gpuRef =(float *)malloc(nBytes);
    
    initialData(h_A,nxy);
    initialData(h_B,nxy);
    
    memset(hostRef,0,nBytes);
    memset(gpuRef,0,nBytes);

    printf("Matrix size: nx %d ny %d \n",nx,ny);
    
    double iStart,iElaps;
    
    // add matrix at the host side
    iStart = seconds();
    sumMatrixOnHost(h_A,h_B,hostRef,nx,ny);
    iElaps = seconds() - iStart;
    printf("sumMatrixOnHost cost %f sec \n",iElaps);
    
    // add matrix at the device side
    float *d_MatA,*d_MatB,*d_MatC;
    cudaMalloc((void **) &d_MatA,nBytes);
    cudaMalloc((void **) &d_MatB,nBytes);
    cudaMalloc((void **) &d_MatC,nBytes);
    
    cudaMemcpy(d_MatA,h_A,nBytes,cudaMemcpyHostToDevice);
    cudaMemcpy(d_MatB,h_B,nBytes,cudaMemcpyHostToDevice);
    
    int dimx = 32;
    int dimy = 32;
    dim3 block(dimx,dimy);
    dim3 grid((nx+block.x-1)/block.x,(ny+block.y-1)/block.y);
    
    iStart = seconds();
    sumMatrixOnGPU2D <<<grid,block>>> (d_MatA,d_MatB,d_MatC,nx,ny);
    cudaDeviceSynchronize();
    iElaps = seconds() - iStart;
    printf("sumMatrixOnGPU2D <<<(%d,%d),(%d,%d)>>> elapsed %f sec\n",
        grid.x,grid.y,block.x,block.y,iElaps);

    cudaMemcpy(gpuRef,d_MatC,nBytes,cudaMemcpyDeviceToHost);
    
    // check the result
    checkResult(hostRef,gpuRef,nxy);
    
    // free the global device & host memory
    cudaFree(d_MatA);
    cudaFree(d_MatB);
    cudaFree(d_MatC);
    free(h_A);
    free(h_B);
    free(hostRef);
    free(gpuRef);
    
    cudaDeviceReset();
    return(0);
    }



    

    
    
    
    
    
    
    



