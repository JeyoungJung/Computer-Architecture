#include <stdio.h>
#define N 5

__global__ void add(float *r, float *g, float *b, float *out, float * coefficient) {
      int tid = blockIdx.x * 64 + threadIdx.x;
    
      __shared__ float cache[3]; //shared memory
      if (threadIdx.x < 3){ //shared memory is shared within block, thus for each block, we have a exclusive copy
        cache[threadIdx.x] = coefficient[threadIdx.x];
      }
      __syncthreads(); // Synchronize (ensure all the data is available)
    
      if(tid < N) {
        out[tid] = 1;
        //  cache[0]*r[tid] + cache[1]*g[tid] + cache[2]*b[tid];
      }
}
  
int main(int argc, char *argv[]) {
    float *r, *g, *b, *out, *coefficient; //memory on host memory
    float *dev_r, *dev_g, *dev_b, *dev_out, *dev_coefficient; //memory on device memory
    
    r = (float*)malloc(N*sizeof(float)); //CPU malloc
    g = (float*)malloc(N*sizeof(float)); //CPU malloc
    b = (float*)malloc(N*sizeof(float)); //CPU malloc
    out = (float*)malloc(N*sizeof(float)); //CPU malloc
    coefficient = (float*)malloc(N*sizeof(float)); //CPU malloc
  
    r[0] = 10.0; r[1] = 11.0; r[2] = 12.0; r[3] = 13.0; r[4] = 14.0;
    g[0] = 15.0; g[1] = 16.0; g[2] = 17.0; g[3] = 18.0; g[4] = 19.0;
    b[0] = 19.0; b[1] = 20.0; b[2] = 21.0; b[3] = 22.0; b[4] = 23.0;

    coefficient[0] = 0.21;
    coefficient[1] = 0.72;
    coefficient[2] = 0.07;

    cudaMalloc((void**)&dev_r, N * sizeof(float)); //GPU malloc
    cudaMalloc((void**)&dev_g, N * sizeof(float)); //GPU malloc
    cudaMalloc((void**)&dev_b, N * sizeof(float)); //GPU malloc
    cudaMalloc((void**)&dev_out, N * sizeof(float)); //GPU malloc
    cudaMalloc((void**)&dev_coefficient, 3 * sizeof(float)); //GPU malloc
  
    cudaMemcpy(dev_r, r, N * sizeof(float), cudaMemcpyHostToDevice); //copy: Host 2 Device
    cudaMemcpy(dev_g, g, N * sizeof(float), cudaMemcpyHostToDevice); //copy: Host 2 Device
    cudaMemcpy(dev_b, b, N * sizeof(float), cudaMemcpyHostToDevice); //copy: Host 2 Device
    cudaMemcpy(dev_out, out, N * sizeof(float), cudaMemcpyHostToDevice); //copy: Host 2 Device
    cudaMemcpy(dev_coefficient, coefficient, 3 * sizeof(float), cudaMemcpyHostToDevice); //copy: Host 2 Device
  
    add<<<1,64>>>(dev_r, dev_g, dev_b, dev_out, dev_coefficient);
    cudaMemcpy(out, dev_out, N * sizeof(float), cudaMemcpyDeviceToHost); //copy: Device 2 Host
    for (int i = 0; i < N ;i++){
        printf("%f ",out[i]);
    } 
    printf("\n"); 
}
  