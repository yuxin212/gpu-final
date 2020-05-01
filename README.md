# Final Project for GPU Parallel Processing

This project is using CUDA to build a simple convolutional neural network. It includes training and testing functions. 

## Usage

Before use, make sure CUDA is installed. 

First make the program

```
nvcc -lcuda -lcublas main.cu -o cnn
````

Then run the executable file `./cnn`. 

## Test Performance

It is recommended to reduce the number of epoch before test performance. 

```
sudo nvprof --metrics dram_read_throughput,dram_write_throughput,flop_count_sp ./cnn
```

or 

```
sudo nv-nsight-cu-cli --metrics dram_read_throughput,dram_write_throughput,flop_count_sp cnn
```

If previous one doesn't work. 
