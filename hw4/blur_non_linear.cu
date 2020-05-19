// Number of channels are equal to 3
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

__global__ void Blur(int height, int width,  double *d_image, double *d_image_res)
{
        int globalidx = threadIdx.x + blockDim.x * blockIdx.x;

	int j = globalidx / width / 3;
	int i = globalidx / 3 - j * width;
	int ch = globalidx - i * 3 - j * width * 3; 
	long int size = height * width * 3;	
	double colorarray[121];

	if (globalidx < size)
	{
		if (i < 4 || j < 4 || i > width - 5 || j > height - 5)
		{
			d_image_res[j*width*3 + i*3 + ch] =  d_image[j*width*3 + i*3 + ch];
		}
		else
		{
			
			int count = 0;
			for (int indi = -5; indi < 6; indi++)
			{
				for (int indj = -5; indj < 6; indj++)
				{
					colorarray[count] = d_image[(j + indj)*width*3 + (i+indi)*3 + ch];
					count++;	
				}
			}
			
			double w = 0;
			for (int indi = 0; indi < 120; indi++)
			{
				for (int indj = indi + 1; indj < 121; indj++)
				{
					if (colorarray[indi] < colorarray[indj])
					{
						w = colorarray[indi];
						colorarray[indi] = colorarray[indj];
						colorarray[indj] = w;
					}
				}

			}


			d_image_res[j*width*3 + i*3 + ch] = colorarray[60];
		}
	}
}

int main(int argc, char **argv)
{
	int width, height, bpp, size;

	// Opening an image
	uint8_t* h_image = stbi_load("baby_yoda.png", &width, &height, &bpp, 3);	
	size = height * width * 3;
	double * h_buf = (double *) malloc(sizeof(double) * size);

	double *d_image;
	double *d_image_res;
	cudaMalloc(&d_image, sizeof(double) * size);
	cudaMalloc(&d_image_res, sizeof(double) * size);

	for (int i = 0; i < size; i++)
	{
		h_buf[i] = (double) h_image[i];
	}
	
	
	cudaMemcpy(d_image, h_buf, sizeof(double) * size, cudaMemcpyHostToDevice);

	 // Sizes of blocks of threads and grid of blocks depending on N 
        int block_size, grid_size;
        if ( height < 1024 )
        {
                block_size = height;
                grid_size = size / height;
        }
        else
        {
                // not for all sizes would work
		block_size = 1024;
                grid_size = size/1024;
        }
	// printf("block_size = %d, grid_size = %d\n", block_size, grid_size);
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
	Blur<<<dimGrid, dimBlock>>>(height, width, d_image, d_image_res);
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
	
	double *h_buf_res = (double *)malloc(sizeof(double) * size);
	cudaMemcpy(h_buf_res, d_image_res, sizeof(double) * size, cudaMemcpyDeviceToHost);
	for (int i = 0; i < size; i++)
        {
                h_image[i] = uint8_t (h_buf_res[i]);
		// printf("%d\n", h_image[i]);
        }
	
	//Printing to file blurred image
	stbi_write_png("changed_baby_yoda.png", width, height, 3, h_image, width * 3);

	free(h_image);
	free(h_buf);
	cudaFree(d_image);
	cudaFree(d_image_res);
	//stbi_image_free(image);
	return 0;
	
}
