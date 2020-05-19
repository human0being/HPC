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

__global__ void Blur(int height, int width, double *kernel, double *d_image, double *d_image_res)
{
        int globalidx = threadIdx.x + blockDim.x * blockIdx.x;

	int j = globalidx / width / 3;
	int i = globalidx / 3 - j * width;
	int ch = globalidx - i * 3 - j * width * 3; 
	long int size = height * width * 3;	

	if (globalidx < size)
	{
		if (i == 0 || j == 0 || i == width - 1 || j == height - 1)
		{
			d_image_res[j*width*3 + i*3 + ch] =  d_image[j*width*3 + i*3 + ch];
		}
		else
		{
			d_image_res[j*width*3 + i*3 + ch] =  (d_image[j*width*3 + i*3 + ch]*kernel[4] + \
                                                        d_image[(j + 1) *width * 3 + (i - 1) * 3 + ch]*kernel[0] + \
                                                        d_image[(j + 1) *width * 3 + (i + 1) * 3 + ch]*kernel[8] + \
                                                        d_image[(j - 1) *width * 3 + (i - 1) * 3 + ch]*kernel[6] + \
                                                        d_image[(j - 1) *width * 3 + (i + 1) * 3 + ch]*kernel[2] + \
                                                        d_image[(j + 1) *width * 3 + i * 3 + ch]*kernel[3] + \
                                                        d_image[j *width * 3 + (i - 1) * 3 + ch]*kernel[1] + \
                                                        d_image[(j - 1) *width * 3 + i * 3 + ch]*kernel[5] + \
                                                        d_image[j * width * 3 + (i + 1)*3 + ch]*kernel[7]); 
		}
	
	if (d_image_res[j*width*3 +i*3 + ch] < 0) { d_image_res[j*width*3 + i*3 + ch] = 0;}
		
	}

}

int main(int argc, char **argv)
{
	int width, height, bpp, size;
	double *kernel = (double *) calloc(sizeof(double), 9);
 	double *d_kernel;
	
	// Check if kernel name is specified
	if (argc != 2)
	{
		printf("Error: please, specify kernel name. ('gaussian' or 'edge_detection' or 'sharpen')\n");
		exit(-1);	
	}

	// Specify kernel for blurring	
	char *kernel_name;
	kernel_name = (char *) malloc(sizeof(char) * (strlen(argv[1] + 1)));
	kernel_name = argv[1];

	if (strcmp(kernel_name, "gaussian") != 0 && strcmp(kernel_name, "edge") != 0 && strcmp(kernel_name, "sharpen"))
	{
		printf("Error: wrong kernel name. Try 'gaussian' or 'edge' or 'sharpen'");
		exit(-1);
	}	
	
	if (strcmp(kernel_name, "edge") == 0)
	{
		
		kernel[0] = kernel[6] = kernel[2] = kernel[8] = -1;
                kernel[1] = kernel[3] = kernel[7] = kernel[5] = -1;
                kernel[4] = 8;
	}
	else
	{
		if (strcmp(kernel_name, "sharpen") == 0)
		{
			kernel[0] = kernel[6] = kernel[2] = kernel[8] = 0;
                        kernel[1] = kernel[3] = kernel[7] = kernel[5] = -1;
                        kernel[4] = 5;	
		}
		else
		{
			kernel[0] = kernel[6] = kernel[2] = kernel[8] = 1 / 16.;
			kernel[1] = kernel[3] = kernel[7] = kernel[5] = 2 / 16.;
			kernel[4] = 4 / 16.;
		}
	}

	// Transfering kernel to device
	cudaMalloc(&d_kernel, sizeof(double)*9);
	cudaMemcpy(d_kernel, kernel, sizeof(double) * 9, cudaMemcpyHostToDevice);	

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
	Blur<<<dimGrid, dimBlock>>>(height, width, d_kernel, d_image, d_image_res);
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
