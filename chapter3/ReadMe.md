# Chapter3  
## 并行性  
### 1. 活跃线程束(warp)  
可实现占用率(achieved_occupancy):  
活跃线程束的平均数量与SM支持最大数量的比值  
nvprof --metrics achieved_occupancy ./sumMatrix 32 32  
### 2. 加载吞吐量(throughput)
全局内核的内存吞吐量  
nvprof --metrics gld_throughput ./sumMatrix 32 32  
全局内存加载效率  
请求的吞吐量与所需吞吐量的比值  
nvprof --metrics gld_efficiency ./sumMatrix 32 32  
### 3. 增大并行性  
增大并行性即增加线程块数量是提高性能的一个重要因素, 其中线程块最内层的维度大小(block.x)对性能起到关键作用  
## 分支分化  
### 1.并行归约  
任意在向量中执行满足交换律和结合律的运算  
类似recursive和reduce的结合, 比如并行求和, 包括相邻配对与交错配对  
1) 简单思路: 在设备端大量并行归约生成线性序列,最后返回主机端求和  
2) 改善分化: 更改 tid%(2*stride)==0 语句,该语句导致每次迭代都只有1/2的线程活跃(偶数编号),而所有线程都必须被调度.
可以通过组织每个线程的数组索引来强制ID相邻的线程执行求和:  
int index = 2 * stride * tid;  
if (index < blockDim.x);  
在每轮归约中只调用一半线程束,剩下的空闲,则总指令数降低.  
测试结果显示,在每个warp里执行的指令数: NeighboredLess(138.875),Neighbored(405.75),运行时间则分别为0.015s与0.033s,提升巨大. (nvprof --metrics inst_per_warp ./reduceInteger)  
3) 交错配对: 两个元素间的跨度被初始为线程块维度的一半,然后在每次循环减小一半直到tid >= stride  
运行时间为0.013s,略有提升.其性能提升来源于函数的全局内存加载/存储模式,在chapter4中会涉及到.  
### 2.展开循环:  
将循环由for i in range(100): a[i] + b[i] 展开为 for i in np.linspace(0,99,50): a[i]+b[i];a[i+1] + b[i+1]  
这种提升来源于编译器循环展开时减少指令消耗,增加更多的独立调度命令. 增加流水线上的并发操作, 增大内存带宽.  
1) 手动展开: kernel4为reduceUnrolling2,每个线程里手动展开两个数据块,这样理论上是由一个线程块处理两个数据块, 在一个线程中有更多的独立内存加载/存储操作会产生更好的性能,这样可以更好地隐藏内存延迟,提高内存吞吐量.  
reduceUnrolling2的处理速度为0.0068sec,相比reduceInterleaved再次提高一倍性能. 而reduceUnrolling4的处理速度为0.0035sec, 再次提高几乎一倍. 可以使用nvprof --metrics dram_read_throughput ./reduceInteger 查看设备内存读取吞吐量.  
2) 线程束的展开: 减少内存阻塞. 原有的__syncthreads被用于块内同步,它用来确保在线程在进入下一轮前,每一轮的所有线程已经将局部结果写入全局内存里了. 而当剩下的线程小于32个时, 每条指令会存在隐式的线程束内同步过程. 因此归约的最后6个迭代可以展开如 vmem[tid] += vmem[tid+32]; ... 等所示.  
线程束展开能避免执行循环控制和线程同步逻辑. 笔者验证速度为0.0026sec,速度再次提升.  
内存阻塞程度可由nvprof --metrics stall_sync ./reduceInteger来查看, 经验证UnrollWarps4为34.58% 而 Unrolling4为52.32%.  
3) 循环完全展开: 已知一个循环中的迭代次数, 就可以把循环完全展开. 书中叙述该循环展开kernel较上述UnrollWarps4更快, 但经笔者本地实验, 完全展开反而略微降低了速度. 查看阻塞程度, CompleteUnrollWarps4为35.61%, 略高于UnrollWarps4的34.46%.  
4) 模板函数: 使用模板参数替代块大小作为if语句的判断来展开循环, 好处在于编译时如果if语句条件为False会被直接删除,使内循环更有效率.  
笔者实验证明其运行速度为0.0025sec,略有提升. 可以使用 nvprof --metrics gld_efficiecny,gst_efficiency ./reduceInteger查看内存的加载和存储效率.  
### 3.动态并行:  
动态并行是指在GPU端直接创建和同步新的GPU内核. 动态地利用GPU硬件调度器和加载平衡器,并且减少主机和设备之间的数据传输.  
1) 嵌套执行: 内核分为parent和child,parent启动子网格时先与child显式同步,child才能执行; 只有在child全部完后,parent才会完成.共享内存和局部内存对于线程块或线程是私有的,在parent和child之间不一致.  
编写第一个嵌套helloworld,需要注意的是编译时需要加上nvcc -arch=sm_50 -rdc=true nestedHelloWorld.cu -o nestedHelloWorld -lcudadevrt, 其中-rdc=true即强制生成可重定位的设备代码,这是动态并行的一个要求,-lcudadevrt则为动态并行需要的设备运行时库的明确链接.  
2) 嵌套归约: 归约可以表示为一个递归函数,如cpuRecursiveReduce函数就是利用cpu进行归约计算. 和cpu归约类似, gpu也可以采用嵌套网格的方式进行归约计算. 但是计算速度特别慢0.25sec, 相比之下cpu函数为0.0028sec. 原因可能是大量的内核调用(子网格启动)和同步(__syncthreads).  
去除函数内部所有的同步过程, 因为每个子线程只需要父线程的数值来指导部分归约. 速度为0.033sec,提高很多但是还是很慢.  
进一步优化线程调用, 上述调用方式为每个线程块均调动子网格, 最初有2048个线程块, 每个线程块执行8次递归, 总共会创造2048*8=16384个子线程,资源消耗巨大. 因此采用第二种子网格调用模式, 即网格中第一个线程块的第一个线程在每一步嵌套时都调用子网格(图流程见书P225 fig3-29/3-30), 这样调用的核函数个数等于递归次数8, 节省了大量的计算资源. 速度为0.000872sec,超过了cpu的计算速度. 可以使用nvprof ./nestedReduce 查看calls(device)项验证上述说法.  












