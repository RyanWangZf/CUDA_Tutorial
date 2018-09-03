// My first CUDA program! 
// 2018.9.1

#include <stdio.h>

__global__ void helloFromGPU(void)
{
	printf("Hello GPU! from thread \n");
	
}

int main(void)
{
	printf("Hello cPU! \n");
	
	helloFromGPU <<<1,10>>>();
	
	//cudaDeviceReset();
	cudaDeviceSynchronize();

	return 0;
}

