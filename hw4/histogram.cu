#define CUDA_DEBUG

#ifdef CUDA_DEBUG

#define CUDA_CHECK_ERROR(err)           \
if (err != cudaSuccess) {          \
printf("Cuda error: %s\n", cudaGetErrorString(err));    \
printf("Error in file: %s, line: %i\n", __FILE__, __LINE__);  \
}                 \

#else

#define CUDA_CHECK_ERROR(err)

#endif
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <stdint.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

__global__ void Blur(int height, int width,  uint8_t *d_image, unsigned int *d_res)
{
        int globalidx = threadIdx.x + blockDim.x * blockIdx.x;
	long int size = height * width;	
	// uint8_t tid = threadIdx.x;

	if (globalidx < size)
	{

		unsigned char value = d_image[globalidx];
		int bin = value % 256;
		atomicAdd(&d_res[bin], 1);
		//__syncthreads();				
	}
}

int main(int argc, char **argv)
{
	int width, height, bpp, size;
	FILE *fp;
	fp = fopen("hist.txt", "w");

	// Opening an image

	uint8_t* h_image_init = stbi_load("baby_yoda_resized.png", &width, &height, &bpp, 3);		
	size = height * width;

	uint8_t* h_image = (uint8_t *) malloc(sizeof(uint8_t) * size);
	for (int i = 0; i < width; i++)
	{
		for (int j = 0; j < height; j++)
		{
			h_image[j*width + i] = (h_image_init[j*width*3 + i*3] + \
						h_image_init[j*width*3 + i*3 + 1] + \
						h_image_init[j*width*3 + i*3 + 2]) / 3.;		
		}
	}

	uint8_t *d_image;
	unsigned int *d_res;
	unsigned int *h_res = (unsigned int *) malloc(sizeof(unsigned int) * 256);
	cudaMalloc(&d_image, sizeof(uint8_t) * size);
	cudaMalloc(&d_res, sizeof(unsigned int) * 256);
	cudaMemset(d_res, 0, sizeof(unsigned int) * 256);
	
	cudaMemcpy(d_image, h_image, sizeof(uint8_t) * size, cudaMemcpyHostToDevice);

	 // Sizes of blocks of threads and grid of blocks depending on N 
        int block_size, grid_size;
        if ( size % 256 != 0)
        {
		printf("Error: resize image. Size should be divided by 256.\n");
		exit(-1);
	}
        else
        {
		block_size = 256;
                grid_size = size / 256;
        }

	printf("block_size = %d, grid_size = %d\n", block_size, grid_size);
        dim3 dimBlock(block_size);
        dim3 dimGrid(grid_size);
	
	//Starting time
	cudaEvent_t start;
        cudaEvent_t stop;
    
    	// Events for synchronication and checking time
    	CUDA_CHECK_ERROR(cudaEventCreate(&start));
    	CUDA_CHECK_ERROR(cudaEventCreate(&stop));

    	//Starting time of calculations on GPU
    	cudaEventRecord(start, 0);

	// Blurring
	Blur<<<dimGrid, dimBlock>>>(height, width, d_image, d_res);
	cudaDeviceSynchronize();	

	// Check for errors
        cudaError_t error = cudaGetLastError();
        if (error != cudaSuccess)
        {
	        printf("CUDA error: %s\n", cudaGetErrorString(error));
                exit(-1);
        }
	
	
    	//The end of calculations on GPU
    	cudaEventRecord(stop, 0);

    	float time = 0;
    	//Synchronization 
    	cudaEventSynchronize(stop);
    	//Elapsed time GPU
    	cudaEventElapsedTime(&time, start, stop);
        printf("GPU compute time: %.2f\n", time);
	
	//double *h_buf_res = (double *)malloc(sizeof(double) * size);
	cudaMemcpy(h_res, d_res, sizeof(unsigned int) * 256, cudaMemcpyDeviceToHost);
	
	for (int i = 0; i < 256; i++)
	{
		fprintf(fp, "%d\t", h_res[i]);
	}
	fprintf(fp, "\n");

	free(h_image);
	free(h_res);
	cudaFree(d_image);
	cudaFree(d_res);
	fclose(fp);
	//stbi_image_free(image);
	return 0;	
}
