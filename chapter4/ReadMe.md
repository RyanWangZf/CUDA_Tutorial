# Chapter4 全局内存  
## 内存层次结构  
1) 局部性原则: 时间局部性和空间局部性.  
2) 层次结构: 磁盘存储器 --> 主存 --> 缓存 --> 寄存器  
## CUDA内存模型  
1) 存储器类型: 分为可编程和不可编程两类, CUDA内存模型提出了多种可编程内存类型: 寄存器, 共享内存, 本地内存, 常量内存, 纹理内存, 全局内存(见P240图4-2).  
2) 寄存器: 对于每个线程是私有的, 如果一个核函数使用了超过硬件限制数量的寄存器,则会用本地内存替代多占用的寄存器,会给性能带来不利影响. 可在代码中显式加上  
<__global__ void"  
__launch__ bounds__(maxThreadsPerblock,minBlocksPerMultiprocessor)  
kernel(...){//body}>  
其中maxThreadsPerblock指出每个线程块最大线程数, minBlockPerMultiprocessor指明每个SM中预期最小常驻线程块数量.  
也可以采用编译选项 --maxrregcount=32 指定代码里所有核函数使用的寄存器最大数量.  
寄存器和核函数生命周期一致.  
3) 本地内存: 寄存器中内存溢出.  
4) 共享内存: 核函数中使用 >>__shared__ 修饰的变量放在共享内存中. 每个SM有一定共享内存, 过度使用会限制活跃线程束的数量.  
共享内存生命周期伴随整个线程块, 是线程通信的基本方式, 访问共享内存必须调用 >>__syncthreads();  
5) 常量内存: 驻留在设备内存中, 在每个SM专用的常量缓存中缓存. 常量变量必须在全局空间和所有核函数之外声明, 用 >>__constant__ 来修饰,最大为64KB, 同一编译单元内的所有核函数可见.  
核函数只能从常量内存中读取数据, 所以常量内存必须在主机端使用  
>>cudaError_t cudaMemcpyToSymbol(const void* symbol, const void* src, size_t count);  
函数初始化, 即将count个字节从src指向的内存复制到symbol指向的内存中.  
常量内存适合所有线程从相同的内存地址读取数据的情况, 比如公式乘以某一个系数.  
6) 纹理内存: 亦驻留于设备内存中, 线程束里使用纹理内存访问二维数据的线程可以达到最优性能.  
7) 全局内存: 是GPU中最大/延迟最高且经常使用的内存. 使用 __device__ 可以静态声明一个变量.  
8) GPU缓存: 不可编程, 有四种: 一级/二级缓存, 只读常量/只读纹理缓存. 每个SM有一个一级缓存, 所有的SM共享一个二级缓存.  
GPU中只有内存加载操作可以被缓存, 内存存储操作不能被缓存. 每个SM也有一个只读常量和只读纹理缓存.  
9) 静态全局内存: 使用 >>__device__ float devData; 定义. 使用  
>>cudaMemcpyToSymblo(devData,&value,sizeof(float)); 将value传至设备内存. 不能使用  
>>cudaMemcpy(&devData,&value,sizeof(float),cudaMemcpyHostToDevice);  
因为devData是GPU上的位置符号, 主机端&devData无法访问. 可以通过  
>>float *dptr = NULL; cudaGetSymbolAddress((void **)&dptr,devData); 获得devData的地址, 再通过  
>>cudaMemcpy(dptr,&value,sizeof(float),cudaMemcpyHostToDevice);  
将value传入设备内存. 由于CUDA_runtime可以访问主机和设备变量, 尤其需要注意给函数提供的变量来自主机还是设备, 否则有可能造成程序崩溃, 或者出现问题而不报错,难以检查.  
## 内存管理
1) 内存分配和释放: 主机上使用  
>>cudaMalloc(void **devPtr, size_t count);  
分配全局内存, 该函数分配count字节的全局内存, 并用devPtr指针返回该内存的地址.  
>>cudaMemset(void *devPtr,int value,size_t count);  
可以使用存储在变量value上的值填充从地址devPtr处开始的count字节.  
>>cudaFree(void *devPtr);  
释放由devPtr指向的全局内存, 且该内存必须已被cudaMalloc或cudaMemset过, 否则会返回cudaErrorInvalidDevicePointer错误.  
2) 内存传输: 全局内存分配之后, 使用  
>>cudaMemcpy(void *dst,const void *src,size_t count,enum cudaMemcoyKind kind);  
从内存位置src复制了count字节到内存位置dst,kind指定复制方向,可以取cudaMemcpyHostToDevice等...  
CPU与GPU之间通过PCIe Gen2总线相连,带宽为8GB/s, GPU和GPU内存带宽则>100GB/s,所以应尽可能减少主机与设备之间的内存传输.  
3) 固定内存: malloc分配的主机内存默认是pageable的,GPU不能在pageable主机内存上安全访问数据,所以CUDA驱动程序会分配pageable锁定或固定的主机内存,将主机源数据复制到固定内存中,然后传输, 使用  
>>cudaMallocHost(void **devPtr,size_t count);  
可以直接分配固定主机内存,能被设备直接访问, 所以可用高的多的带宽进行读写. 固定主机内存须用  
>>cudaFreeHost(void *ptr)  
来释放. 比较malloc和cudaMallocHost两种情况的数据传输速度, 分别编写脚本memTransfer.cu和pinMemTransfer.cu.  
前者传输耗时5.2ms,而后者仅2.6ms.  
4) 零拷贝内存: 有一个例外,即主机和设备均可以访问零拷贝内存. 在CUDA核函数中使用零拷贝内存的优点有:  
当设备内存不足时可利用主机内存; 避免主机和设备间的显式数据传输; 提高PCIe传输率.  
零拷贝内存是固定内存,通过  
>>cudaHostAlloc(void **pHost,size_t count,unsigned int flags);  
创建到固定内存的映射, 必须用cudaFreeHost函数释放. flags用于特殊属性的配置,详见p259.  
可以使用  
>>cudaHostGetDevicePointer(void **pDevice,void *pHost,unsigned int flags);  
返回在pDevice中的设备指针,该指针可以在设备上引用以访问映射得到的固定主机内存.  
通过sumArrayZerocopy.cu计算时间,sumArraysZeroCopy 5.0us,而sumArraysOnGPU 2.9us.  
