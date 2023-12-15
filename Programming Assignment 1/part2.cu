#include <iostream>
#include <string>
#include <fstream>
#include <chrono>

#include <stdio.h>
#include <stdlib.h>

using namespace std;

//We may change this value!!
const int FILTER_WIDTH = 7;

//We may change this value!!!
float FILTER[FILTER_WIDTH*FILTER_WIDTH] = {
    1,4,7,10,7,4,1,
    4,12,26,33,26,12,4,
    7,26,55,71,55,26,7,
    10,33,71,91,71,33,10,
    7,26,55,71,55,26,7,
    4,12,26,33,26,12,4,
    1,4,7,10,7,4,1
};

void normalize(float *FILTER) {
	// Calculate the sum of all elements in the filter
	float sum = 0;
	for (int i = 0; i < FILTER_WIDTH*FILTER_WIDTH; i++) {
			sum += FILTER[i];
	}

	// If the sum is not zero, divide each element by the sum
	if (sum != 1) {
			for (int i = 0; i < FILTER_WIDTH*FILTER_WIDTH; i++) {
					FILTER[i] /= sum;
			}
	}
}

// Display the first and last 10 items
// For debug only
void displayResult(const int original[], const int result[], int size) {
	cout << "Display result: ";
	cout << "(original -> result)\n";

	for (int i = 0; i < 10; i++) {
		cout << original[i] << " -> " << result[i] << "\n";
	}
	cout << ".\n.\n.\n";

	for (int i = size - 10; i < size; i++) {
		cout << original[i] << " -> " << result[i] << "\n";
	}
}

void initColorData(string file, int **data, int *sizeX, int *sizeY) {
	int x;
	int y;
	long long i = 0;
	cout << "Reading "<< file << "... \n";
	ifstream myfile(file);
	if (myfile.is_open()) {
		myfile >> x;
		myfile >> y;

		int *temp = new int[x * y * 3];
		for( i=0; i < x * y * 3; i++){
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

void saveResult(string file, int data[], int sizeX, int sizeY) {
	long long i = 0;
	cout << "Saving data to "<< file <<"... \n";
	ofstream myfile(file, std::ofstream::out);
	if (myfile.is_open()) {
		myfile << sizeX << "\n";
		myfile << sizeY << "\n";
		for (i = 0; i < sizeX * sizeY; i++){
			myfile << data[3* i] << " " << data[3* i + 1] << " " << data[3* i+ 2]<< "\n";
		}
		myfile.close();
	}
	else {
		cout << "ERROR: Cannot save to " << file << "!\n";
		exit(0);
	}
	cout << i << " entries saved\n";
}

// TODO: implement the kneral function for 2D smoothing 
__global__ void GPU_Kernel(int *data, int *result, float *filter, int sizeX, int sizeY) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < sizeX && y < sizeY) {
        for (int c = 0; c < 3; c++) {
            int sum = 0;
            for (int fy = 0; fy < FILTER_WIDTH; fy++) {
                for (int fx = 0; fx < FILTER_WIDTH; fx++) {
                    int imageX = x + fx - FILTER_WIDTH / 2;
                    int imageY = y + fy - FILTER_WIDTH / 2;
                    if (imageX >= 0 && imageX < sizeX && imageY >= 0 && imageY < sizeY) {
                        sum += data[(imageY * sizeX + imageX) * 3 + c] * filter[fy * FILTER_WIDTH + fx];
                    }
                }
            }
						sum = sum < 0 ? 0 : sum;
            sum = sum > 255 ? 255 : sum;
            result[(y * sizeX + x) * 3 + c] = sum;
        }
    }
}


// GPU implementation
void GPU_Test(int data[], int result[], int sizeX, int sizeY) {
		normalize(FILTER);
    int size = sizeX * sizeY * 3;
    int *d_data, *d_result;
		float *d_filter;

    // allocate device memory
    cudaMalloc((void**)&d_data, size * sizeof(int));
    cudaMalloc((void**)&d_result, size * sizeof(int));
    cudaMalloc((void**)&d_filter, FILTER_WIDTH * FILTER_WIDTH * sizeof(float));

    // copy data onto the device
    cudaMemcpy(d_data, data, size * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_filter, FILTER, FILTER_WIDTH * FILTER_WIDTH * sizeof(float), cudaMemcpyHostToDevice);

    // define block and grid sizes
    dim3 blockSize(32, 32);
    dim3 gridSize((sizeX + blockSize.x - 1) / blockSize.x, (sizeY + blockSize.y - 1) / blockSize.y);

    // start timer for kernel
    auto startKernel = chrono::steady_clock::now();

    // call the kernel function
    GPU_Kernel<<<gridSize, blockSize>>>(d_data, d_result, d_filter, sizeX, sizeY);

    // synchronize device
    cudaDeviceSynchronize(); 

    // end timer for kernel and display kernel time
    auto endKernel = chrono::steady_clock::now();
    cout << "Kernel Elapsed time: " << chrono::duration <double, milli>(endKernel - startKernel).count() << "ms\n";

    // copy result from device to host
    cudaMemcpy(result, d_result, size * sizeof(int), cudaMemcpyDeviceToHost);

    // free device memory
    cudaFree(d_data);
    cudaFree(d_result);
    cudaFree(d_filter);
}

// CPU implementation
void CPU_Test(int data[], int result[], int sizeX, int sizeY) {
	// input:
	//	int data[] - int array holding the flattened original image
	//	int sizeX - the width of the image
	//	int sizeY - the height of the image
	// output:
	//	int result[] - int array holding the image

	// TODO: smooth the image with filter size = FILTER_WIDTH
	//       apply zero padding for the border
	normalize(FILTER);
	for (int y = 0; y < sizeY; y++) {
			for (int x = 0; x < sizeX; x++) {
					for (int c = 0; c < 3; c++) { // for each color channel
							int sum = 0;
							for (int fy = 0; fy < FILTER_WIDTH; fy++) {
									for (int fx = 0; fx < FILTER_WIDTH; fx++) {
											int imageX = x + fx - FILTER_WIDTH / 2;
											int imageY = y + fy - FILTER_WIDTH / 2;
											if (imageX >= 0 && imageX < sizeX && imageY >= 0 && imageY < sizeY) {
													sum += data[(imageY * sizeX + imageX) * 3 + c] * FILTER[fy * FILTER_WIDTH + fx];
											} // for 'out of bounds' pixels, no addition operation is performed, equivalent to adding zero.
									}
							}
							sum = sum < 0 ? 0 : sum;
            	sum = sum > 255 ? 255 : sum;
							result[(y * sizeX + x) * 3 + c] = sum;
					}
			}
	}
}

// The image is flattened into a text file of pixel values.
int main(int argc, char *argv[]) {
	string inputFile = (argc == 1) ? "image_color.txt" : argv[1];

	int sizeX;
	int sizeY;
	int *dataForCPUTest;
	int *dataForGPUTest;	

	initColorData(inputFile, &dataForCPUTest, &sizeX, &sizeY);
	initColorData(inputFile, &dataForGPUTest, &sizeX, &sizeY);

	int size = sizeX * sizeY * 3;
	int *resultForCPUTest = new int[size];
	int *resultForGPUTest = new int[size];

	cout << "\n";

	cout << "CPU Implementation\n";

	auto startCPU = chrono::steady_clock::now();
	CPU_Test(dataForCPUTest, resultForCPUTest, sizeX, sizeY);
	auto endCPU = chrono::steady_clock::now();

	cout << "Elapsed time: " << chrono::duration <double, milli>(endCPU - startCPU).count() << "ms\n";

	// displayResult(dataForCPUTest, resultForCPUTest, size);

	saveResult("color_result_CPU.txt",resultForCPUTest, sizeX, sizeY);

	cout << "\n";
	cout << "GPU Implementation\n";

	auto startGPU = chrono::steady_clock::now();
	GPU_Test(dataForGPUTest, resultForGPUTest, sizeX, sizeY);
	auto endGPU = chrono::steady_clock::now();

	cout << "Elapsed time: " << chrono::duration <double, milli>(endGPU - startGPU).count() << "ms\n";

	// displayResult(dataForGPUTest, resultForGPUTest, size);
	saveResult("color_result_GPU.txt",resultForGPUTest, sizeX, sizeY);

	return 0;
}
