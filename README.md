# Final Project for GPU Parallel Processing

## Introduction

This project is using CUDA to build a simple convolutional neural network. It has two parts. The first part is the main program. It includes training and testing functions. The second part is for benchmark. It calculates computational performance and memory bandwidth. 

## Usage

Before use, make sure CUDA is installed. 

### For main program: 

First make the program

```
nvcc -lcuda -lcublas main.cu
````

Then run the executable file `./a.out`. 

### For benchmark: 

