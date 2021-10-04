#include <iostream>
#include <math.h>
#include <map>
#include <string>
#include <vector>
#include <stdbool.h>
#include <tuple>
#include <unistd.h>
#include <sys/types.h>
#include <dirent.h>
#include <stdio.h>

#include "../external/lodepng/lodepng.cpp"

#define sqr(a) ((a)*(a))

using namespace std;

int maxGridDimX;
int maxGridDimY;

vector<string> listDir(const char* name) {
    DIR* dir;
    struct dirent* entry;
    vector<string> files;

    if (!(dir = opendir(name)))
        return files;

    while ((entry = readdir(dir)) != NULL) {
        if (strcmp(entry->d_name, ".") && strcmp(entry->d_name, "..")) {
            files.push_back(entry->d_name);
        }
    }
    closedir(dir);

    return files;
}

vector<vector<float> > kernels = {
    vector<float> { // gaussian 3x3
        1.0/16, 2.0/16, 1.0/16,
        2.0/16, 4.0/16, 2.0/16,
        1.0/16, 2.0/16, 1.0/16
    },
    vector<float> { // gaussian 5x5
        1.0/256, 4.0/256,   6.0/256,  4.0/256, 1.0/256,
        4.0/256, 16.0/256, 24.0/256, 16.0/256, 4.0/256,
        6.0/256, 24.0/256, 36.0/256, 24.0/256, 6.0/256,
        4.0/256, 16.0/256, 24.0/256, 16.0/256, 4.0/256,
        1.0/256, 4.0/256,   6.0/256,  4.0/256, 1.0/256
    },
    vector<float> { // edge detect 3x3
        -1, -1, -1,
        -1,  8, -1,
        -1, -1, -1
    }
};

void getError(cudaError_t err) {
    if (err != cudaSuccess) {
        printf("CUDA error - %s\n", cudaGetErrorString(err));
    }
}

__global__ void apply_kernel_device(
    unsigned char* input_image,
    unsigned char* output_image,
    int width,
    int height,
    float* kernel,
    char kernel_dim
) {
    const unsigned int linearX = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int linearY = blockIdx.y * blockDim.y + threadIdx.y;

    if (linearX >= width || linearY >= height) {
        return;
    }

    float r = 0, g = 0, b = 0;

    // Ядро 3х3, отсекаем рамку из 1 пикселя
    if (kernel_dim == 3 && linearX > 0 && linearX < width - 1 && linearY > 0 && linearY < height - 1) {
        for (int i = -1; i < 2; i++) {
            for (int j = -1; j < 2; j++) {
                r += input_image[3 * ((linearY + i) * width + (linearX + j))] * kernel[3 * (i + 1) + j + 1];
                g += input_image[3 * ((linearY + i) * width + (linearX + j)) + 1] * kernel[3 * (i + 1) + j + 1];
                b += input_image[3 * ((linearY + i) * width + (linearX + j)) + 2] * kernel[3 * (i + 1) + j + 1];
            }
        }

        output_image[3 * (linearY * width + linearX)] = ceil(r);
        output_image[3 * (linearY * width + linearX) + 1] = ceil(g);
        output_image[3 * (linearY * width + linearX) + 2] = ceil(b);

    // Ядро 5х5, отсекаем рамку из 2 пикселей
    } else if ((kernel_dim == 5 && linearX > 1 && linearX < width - 2 && linearY > 2 && linearY < height - 2)) {
        for (int i = -2; i < 3; i++) {
            for (int j = -2; j < 3; j++) {
                r += input_image[3 * ((linearY + i) * width + (linearX + j))] * kernel[3 * (i + 2) + j + 2];
                g += input_image[3 * ((linearY + i) * width + (linearX + j)) + 1] * kernel[3 * (i + 2) + j + 2];
                b += input_image[3 * ((linearY + i) * width + (linearX + j)) + 2] * kernel[3 * (i + 2) + j + 2];
            }
        }

        output_image[3 * (linearY * width + linearX)] = ceil(r);
        output_image[3 * (linearY * width + linearX) + 1] = ceil(g);
        output_image[3 * (linearY * width + linearX) + 2] = ceil(b);

    // То что попало в рамки (соотв. из 1 и из 2 пикселей)
    } else {
        output_image[3 * (linearY * width + linearX)] = input_image[3 * (linearY * width + linearX)];
        output_image[3 * (linearY * width + linearX) + 1] = input_image[3 * (linearY * width + linearX) + 1];
        output_image[3 * (linearY * width + linearX) + 2] = input_image[3 * (linearY * width + linearX) + 2];
    }
}

void apply_kernel(unsigned char* input_image, unsigned char* output_image, int width, int height, int kernelId, float& calcTime, float& totalTime) {
    unsigned char* dev_input;
    unsigned char* dev_output;
    float* dev_kernel;

    int size = kernels[kernelId].size();
    float* kernel = kernels[kernelId].data();

    float ms_outer = 0;
    float ms_inner = 0;
    cudaEvent_t start_outer;
    cudaEvent_t stop_outer;
    cudaEvent_t start_inner;
    cudaEvent_t stop_inner;
    cudaEventCreate(&start_outer);
    cudaEventCreate(&stop_outer);
    cudaEventCreate(&start_inner);
    cudaEventCreate(&stop_inner);

    cudaEventRecord(start_outer);
    cudaEventSynchronize(start_outer);

    getError(cudaMalloc((void **)&dev_input, 3 * width * height * sizeof(unsigned char)));
    getError(cudaMemcpy(dev_input, input_image, 3 * width * height * sizeof(unsigned char), cudaMemcpyHostToDevice));

    getError(cudaMalloc((void **)&dev_kernel, size * sizeof(float)));
    getError(cudaMemcpy(dev_kernel, kernel, size * sizeof(float), cudaMemcpyHostToDevice));

    getError(cudaMalloc((void **)&dev_output, 3 * width * height * sizeof(unsigned char)));

    int blockDim = 32;
    int gridDimX = ceil(1.0 * width / blockDim);
    int gridDimY = ceil(1.0 * height / blockDim);

    if (gridDimX > maxGridDimX || gridDimY > maxGridDimY) {
        throw runtime_error("Error! Image too big");
    }

    printf("Sizes: block size %d, grid x-dim %d, grid y-dim %d\n", blockDim, gridDimX, gridDimY);

    dim3 blockDims(blockDim, blockDim, 1);
    dim3 gridDims(gridDimX, gridDimY, 1);

    cudaEventRecord(start_inner);
    cudaEventSynchronize(start_inner);

    apply_kernel_device<<<gridDims, blockDims>>>(dev_input, dev_output, width, height, dev_kernel, (int)(sqrt(size)));

    cudaEventRecord(stop_inner);
    cudaEventSynchronize(stop_inner);
    cudaEventElapsedTime(&ms_inner, start_inner, stop_inner);

    getError(cudaMemcpy(output_image, dev_output, width * height * 3 * sizeof(unsigned char), cudaMemcpyDeviceToHost));

    getError(cudaFree(dev_input));
    getError(cudaFree(dev_output));
    getError(cudaFree(dev_kernel));

    cudaEventRecord(stop_outer);
    cudaEventSynchronize(stop_outer);
    cudaEventElapsedTime(&ms_outer, start_outer, stop_outer);

    printf("GPU calculation time: %g ms\n", ms_inner);
    printf("GPU total time: %g ms\n", ms_outer);

    calcTime = ms_inner;
    totalTime = ms_outer;
}

void loadCudaSettings() {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);

    maxGridDimX = prop.maxGridSize[0];
    maxGridDimY = prop.maxGridSize[1];
}

void processImage(char* inFile, char* outFile, int kernel, float& calcTime, float& totalTime) {
    vector<unsigned char> rgbaIn;
    unsigned int width, height;

    unsigned error = lodepng::decode(rgbaIn, width, height, inFile);
    if (error) {
        cout << "decoder error " << error << ": " << lodepng_error_text(error) << endl;
    }

    unsigned char *rgbIn = new unsigned char[(rgbaIn.size() * 3) / 4];
    unsigned char *rgbOut = new unsigned char[(rgbaIn.size() * 3) / 4];
    int inp_iterator = 0;
    for (int i = 0; i < rgbaIn.size(); ++i) {
        if ((i + 1) % 4 != 0) { // 3,7... - под альфа пропускаем
            rgbIn[inp_iterator] = rgbaIn.at(i);
            rgbOut[inp_iterator] = 255;
            inp_iterator++;
        }
    }

    printf("Image size - %dx%d\n", width, height);

    apply_kernel(rgbIn, rgbOut, width, height, kernel, calcTime, totalTime);

    int out_iterator = 0;
    vector<unsigned char> rgbaOut(rgbaIn.size());
    for (int i = 0; i < width * height * 3; ++i) {
        rgbaOut[out_iterator] = rgbOut[i];
        out_iterator++;
        if ((i + 1) % 3 == 0) { // в конец каждой тройки rgb пишем альфа
            rgbaOut[out_iterator] = 255;
            out_iterator++;
        }
    }

    error = lodepng::encode(outFile, rgbaOut, width, height);

    if (error) {
        printf("Encoder error: %s\n", lodepng_error_text(error));
    }

    delete[] rgbIn;
    delete[] rgbOut;
}

void parseArgs(int argc, char** argv, int* kernel, char** imgType) {
    if (argc != 3) {
        cout << "2 arguments required" << endl;
        exit(0);
    }

    *kernel = atoi(argv[1]);
    *imgType = argv[2];

    if (*kernel < 0 || *kernel > 2) {
        cout << "Kernel idx must be in range [0,2]" << endl;
        exit(0);
    }
}

int main(int argc, char** argv) {
    vector<pair<char*, char*> > images;
    char* imgType;
    int kernel;

    parseArgs(argc, argv, &kernel, &imgType);

    if (!strcmp(imgType,"big")) {
        images.push_back({ "in/big.png", "out/big.png" });

    } else if (!strcmp(imgType,"small")) {
        vector<string> files = listDir("in/small/");
        for (string name: files) {
            string strIn = "in/small/" + name;
            string strOut = "out/" + name;
            char *in = new char[strIn.length() + 1];
            char *out = new char[strOut.length() + 1];
            strcpy(in, strIn.c_str());
            strcpy(out, strOut.c_str());
            images.push_back({ in, out });
        }
    
    } else {
        cout << "Invalid arguments" << endl;
        return 0;
    }

    loadCudaSettings();

    float calcTimeSum = 0, totalTimeSum = 0;

    for (int i = 0; i < images.size(); i++) {
        printf("Started processing image %s\n", images[i].first);
        float calcTime = 0, totalTime = 0;
        processImage(images[i].first, images[i].second, kernel, calcTime, totalTime);
        calcTimeSum += calcTime;
        totalTimeSum += totalTime;
        printf("Finised. Output was written to %s\n", images[i].second);
    }

    printf("Sum of calculation times - %g\n", calcTimeSum);
    printf("Sum of total times - %g\n", totalTimeSum);
    
    return 0;
}