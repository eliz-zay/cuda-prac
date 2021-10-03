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

vector<string> listdir(const char *name, int indent) {
    DIR *dir;
    struct dirent *entry;
    vector<string> files;

    if (!(dir = opendir(name)))
        return files;

    while ((entry = readdir(dir)) != NULL) {
        if (entry->d_type == DT_DIR) {
            char path[1024];
            if (strcmp(entry->d_name, ".") == 0 || strcmp(entry->d_name, "..") == 0)
                continue;
            snprintf(path, sizeof(path), "%s/%s", name, entry->d_name);
            files.push_back(entry->d_name);
            listdir(path, indent + 2);
        } else {
            files.push_back(entry->d_name);
        }
    }
    closedir(dir);

    return files;
}

const map<string, pair<char, float* > > kernels {
    {
        "box-blur",
        {
            3,
            new float[9] {
                1.0/9, 1.0/9, 1.0/9,
                1.0/9, 1.0/9, 1.0/9,
                1.0/9, 1.0/9, 1.0/9
            }
        }
    },
    {
        "gaussian-blur-3",
        {
            3,
            new float[9] {
                1.0/16, 2.0/16, 1.0/16,
                2.0/16, 4.0/16, 2.0/16,
                1.0/16, 2.0/16, 1.0/16
            }
        }
    },
    {
        "gaussian-blur-5",
        {
            5,
            new float[25] {
                1.0/256, 4.0/256,   6.0/256,  4.0/256, 1.0/256,
                4.0/256, 16.0/256, 24.0/256, 16.0/256, 4.0/256,
                6.0/256, 24.0/256, 36.0/256, 24.0/256, 6.0/256,
                4.0/256, 16.0/256, 24.0/256, 16.0/256, 4.0/256,
                1.0/256, 4.0/256,   6.0/256,  4.0/256, 1.0/256
            }
        }
    },
    {
        "edge-detect",
        {
            3,
            new float[9] {
                -1, -1, -1,
                -1,  8, -1,
                -1, -1, -1
            }
        }
    },
    {
        "emboss",
        {
            3,
            new float[9] {
                -2, -1, 0,
                -1,  1, 1,
                 0,  1, 2
            }
        }
    }
};

int maxBlockSize;
int maxBlockDimX;
int maxBlockDimY;
int maxGridDimX;
int maxGridDimY;

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

    float r;
    float g;
    float b;

    /**
     * Case when kernel dimension is 3. In this case process all pixels but first edge
     */
    if (kernel_dim == 3 && linearX > 0 && linearX < width - 1 && linearY > 0 && linearY < height - 1) {
        r = 0;
        g = 0;
        b = 0;

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
    }
    /**
     * Case when kernel dimension is 5. In this case process all pixels but first two edges
     */
    else if ((kernel_dim == 5 && linearX > 1 && linearX < width - 2 && linearY > 2 && linearY < height - 2)) {
        r = 0;
        g = 0;
        b = 0;

        for (int i = -2; i < 3; i++) {
            for (int j = -2; j < 3; j++) {
                r += input_image[3 * ((linearY + i) * width + (linearX + j))] * kernel[3 * (i + 1) + j + 1];
                g += input_image[3 * ((linearY + i) * width + (linearX + j)) + 1] * kernel[3 * (i + 1) + j + 1];
                b += input_image[3 * ((linearY + i) * width + (linearX + j)) + 2] * kernel[3 * (i + 1) + j + 1];
            }
        }

        output_image[3 * (linearY * width + linearX)] = ceil(r);
        output_image[3 * (linearY * width + linearX) + 1] = ceil(g);
        output_image[3 * (linearY * width + linearX) + 2] = ceil(b);
    }
    /**
     * Case when pixel is on the edge
     */
    else {
        output_image[3 * (linearY * width + linearX)] = input_image[3 * (linearY * width + linearX)];
        output_image[3 * (linearY * width + linearX) + 1] = input_image[3 * (linearY * width + linearX) + 1];
        output_image[3 * (linearY * width + linearX) + 2] = input_image[3 * (linearY * width + linearX) + 2];
    }
}

void apply_kernel(unsigned char* input_image, unsigned char* output_image, int width, int height, string filter) {
    unsigned char* dev_input;
    unsigned char* dev_output;
    float* dev_kernel;

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

    getError(cudaMalloc((void **)&dev_kernel, sqr(kernels.at(filter).first) * sizeof(float)));
    getError(cudaMemcpy(dev_kernel, kernels.at(filter).second, sqr(kernels.at(filter).first) * sizeof(float), cudaMemcpyHostToDevice));

    getError(cudaMalloc((void **)&dev_output, 3 * width * height * sizeof(unsigned char)));

    int blockDim = min(static_cast<int>(floor(sqrt(maxBlockSize))), min(maxBlockDimX, maxBlockDimY));
    int gridDimX = ceil(1.0 * width / blockDim);
    int gridDimY = ceil(1.0 * height / blockDim);

    if (gridDimX > maxGridDimX || gridDimY > maxGridDimY) {
        throw runtime_error("Too big image");
    }

    printf("Device params: block size %d, grid x-dim %d, grid y-dim %d\n", blockDim, gridDimX, gridDimY);

    dim3 blockDims(blockDim, blockDim, 1);
    dim3 gridDims(gridDimX, gridDimY, 1);

    cudaEventRecord(start_inner);
    cudaEventSynchronize(start_inner);

    apply_kernel_device<<<gridDims, blockDims>>>(dev_input, dev_output, width, height, dev_kernel, kernels.at(filter).first);

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
    printf("GPU calculation + transport time: %g ms\n", ms_outer);
}

void loadCudaSettings() {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);

    maxBlockSize = prop.maxThreadsPerBlock;
    maxBlockDimX = prop.maxThreadsDim[0];
    maxBlockDimY = prop.maxThreadsDim[1];
    maxGridDimX = prop.maxGridSize[0];
    maxGridDimY = prop.maxGridSize[1];

    printf("CUDA block max size - %d\n", prop.maxThreadsPerBlock);
    printf("CUDA block max dimensions - %d, %d, %d\n", prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
    printf("CUDA grid max dimensions - %d, %d, %d\n", prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
}

void processImage(char* input_file, char* output_file, string filter) {
    vector<unsigned char> in_image;
    unsigned int width, height;

    unsigned error = lodepng::decode(in_image, width, height, input_file);
    if (error) {
        cout << "decoder error " << error << ": " << lodepng_error_text(error) << endl;
    }

    unsigned char *input_image = new unsigned char[(in_image.size() * 3) / 4];
    unsigned char *output_image = new unsigned char[(in_image.size() * 3) / 4];
    int inp_iterator = 0;
    for (int i = 0; i < in_image.size(); ++i) {
        if ((i + 1) % 4 != 0) {
            input_image[inp_iterator] = in_image.at(i);
            output_image[inp_iterator] = 255;
            inp_iterator++;
        }
    }

    printf("Image size - %dx%d\n", width, height);

    apply_kernel(input_image, output_image, width, height, filter);

    int out_iterator = 0;
    vector<unsigned char> out_image(in_image.size());
    for (int i = 0; i < width * height * 3; ++i) {
        out_image[out_iterator] = output_image[i];
        out_iterator++;
        if ((i + 1) % 3 == 0) {
            out_image[out_iterator] = 255;
            out_iterator++;
        }
    }

    error = lodepng::encode(output_file, out_image, width, height);

    if (error) {
        printf("Encoder error: %s\n", lodepng_error_text(error));
    }

    delete[] input_image;
    delete[] output_image;
}

void parseArgs(int argc, char** argv, char** filter, char** imgType) {
    if (argc != 3) {
        printf("2 arguments required");
        exit(0);
    }

    *filter = argv[1];
    *imgType = argv[2];
}

int main(int argc, char** argv) {
    vector<pair<char*, char*> > images;
    char* kernel, *imgType;

    parseArgs(argc, argv, &kernel, &imgType);

    if (!strcmp(imgType,"big")) {
        images.push_back({ "in/big.png", "out/big.png" });

    } else if (!strcmp(imgType,"small")) {
        vector<string> files = listdir("in/small/", 0);
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

    string str(kernel);

    for (int i = 0; i < images.size(); i++) {
        printf("Started processing image %s\n", images[i].first);
        processImage(images[i].first, images[i].second, str);
        printf("Finised. Output was written to %s\n", images[i].second);
    }
    
    return 0;
}