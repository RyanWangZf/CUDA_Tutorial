#Chapter3  
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


