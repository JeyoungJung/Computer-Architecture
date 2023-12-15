#include <stdio.h>
#include <iostream>
#include <string>
#include <fstream>

using namespace std;

__global__ void add(float *r, float *g, float *b, float *out, float *coefficient) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    __shared__ float cache[3];
    if (threadIdx.x < 3) {
        cache[threadIdx.x] = coefficient[threadIdx.x];
    }
    __syncthreads();

    if(tid < blockDim.x * gridDim.x) {
        out[tid] = cache[0]*r[tid] + cache[1]*g[tid] + cache[2]*b[tid];
    }
}

void initColorData(string file, float **data, int *sizeX, int *sizeY) {
    int x;
    int y;
    long long i = 0;
    cout << "Reading " << file << "... \n";
    ifstream myfile(file);
    if (myfile.is_open()) {
        myfile >> x;
        myfile >> y;

        float *temp = new float[x * y * 3];
        for(i = 0; i < x * y * 3; i++){
            myfile >> temp[(int)i];
        }
        myfile.close();
        *data = temp;
        *sizeX = x;
        *sizeY = y;
    }
    else {
        cout << "ERROR: File " << file << " not found!\n";
        exit(0);
    }
    cout << i << " entries imported\n";
}

void saveResult(string file, float data[], int sizeX, int sizeY) {
    long long i = 0;
    cout << "Saving data to " << file << "... \n";
    ofstream myfile(file, std::ofstream::out);
    if (myfile.is_open()) {
        myfile << sizeX << "\n";
        myfile << sizeY << "\n";
        for (i = 0; i < sizeX * sizeY; i++){
            myfile << data[i] << "\n";
        }
        myfile.close();
    }
    else {
        cout << "ERROR: Cannot save to " << file << "!\n";
        exit(0);
    }
    cout << i << " entries saved\n";
}

int main(int argc, char *argv[]) {
    float *data, *r, *g, *b, *out, *coefficient;
    float *dev_r, *dev_g, *dev_b, *dev_out, *dev_coefficient;
    int sizeX, sizeY;

    initColorData("image_color.txt", &data, &sizeX, &sizeY);
    int total_pixels = sizeX * sizeY;

    r = new float[total_pixels];
    g = new float[total_pixels];
    b = new float[total_pixels];
    out = new float[total_pixels];
    coefficient = new float[3];

    for (int i = 0; i < total_pixels; i++) {
        r[i] = data[3 * i];
        g[i] = data[3 * i + 1];
        b[i] = data[3 * i + 2];
    }

    coefficient[0] = 0.21;
    coefficient[1] = 0.72;
    coefficient[2] = 0.07;

    cudaMalloc((void**)&dev_r, total_pixels * sizeof(float));
    cudaMalloc((void**)&dev_g, total_pixels * sizeof(float));
    cudaMalloc((void**)&dev_b, total_pixels * sizeof(float));
    cudaMalloc((void**)&dev_out, total_pixels * sizeof(float));
    cudaMalloc((void**)&dev_coefficient, 3 * sizeof(float));

    cudaMemcpy(dev_r, r, total_pixels * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_g, g, total_pixels * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, b, total_pixels * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_out, out, total_pixels * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_coefficient, coefficient, 3 * sizeof(float), cudaMemcpyHostToDevice);

    int blockSize = 256;
    int gridSize = (total_pixels + blockSize - 1) / blockSize;
    add<<<gridSize, blockSize>>>(dev_r, dev_g, dev_b, dev_out, dev_coefficient);

    cudaMemcpy(out, dev_out, total_pixels * sizeof(float), cudaMemcpyDeviceToHost);

    saveResult("image_gray.txt", out, sizeX, sizeY);

    delete[] data;
    delete[] r;
    delete[] g;
    delete[] b;
    delete[] out;
    delete[] coefficient;

    cudaFree(dev_r);
    cudaFree(dev_g);
    cudaFree(dev_b);
    cudaFree(dev_out);
    cudaFree(dev_coefficient);

    return 0;
}
