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
1)手动展开: kernel4为reduceUnrolling2,每个线程里手动展开两个数据块,这样理论上是由一个线程块处理两个数据块,
在一个线程中有更多的独立内存加载/存储操作会产生更好的性能,这样可以更好地隐藏内存延迟,提高内存吞吐量. reduceUnrolling2的处理速度为0.0068sec,相比reduceInterleaved再次提高一倍性能.  








