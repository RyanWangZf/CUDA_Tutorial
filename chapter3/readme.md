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











