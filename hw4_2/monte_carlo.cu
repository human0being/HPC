// first task
// nvcc monte_carlo.cu
// ./a.out
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

__global__ void scan(const int n, float * d_in, float * d_out)
{
	int idx = threadIdx.x;
	extern __shared__ float temp[];

	int pout = 0, pin = 1;

	temp[idx] = d_in[idx];
	__syncthreads();

	for(int offset = 1; offset < n; offset = offset * 2)
	{
		pout = 1 - pout;
		pin = 1 - pout;

		if(idx>=offset)
		{
			// scan algo
			temp[pout*n+idx] = temp[pin*n+idx-offset]+temp[pin*n+idx];
		}
		else
		{
			temp[pout*n+idx]=temp[pin*n+idx];
		}
		__syncthreads();
	}
	
	d_out[idx] = temp[pout*n+idx]; 
}


int main(void)
{
        //number of discretization points
        const int ARRAY_SIZE = pow(2, 10);
        printf("Number of discretization points: %d\n", ARRAY_SIZE);
        const int ARRAY_BYTES = ARRAY_SIZE * sizeof(float);

        //filling an array with elements
        float *h_in = (float *) malloc(ARRAY_BYTES);
	float x_rand, y_rand, sum=0;

        for (int i=0; i < ARRAY_SIZE; i++)
        {
		srand(i);
		x_rand = (float) rand() / RAND_MAX * 4 - 2;
		srand(i + 123467890);
		y_rand = (float) rand() / RAND_MAX * 4 - 2;
		//printf("%f\t%f\n", x_rand, y_rand);
                h_in[i] = exp(- x_rand*x_rand - y_rand*y_rand);
                sum += h_in[i];
        }

        printf("Integral sequential: %.5f\n", sum*16/ARRAY_SIZE);

        float *h_out = (float *) malloc(ARRAY_BYTES);
        float * d_in;
        float * d_out;

        cudaMalloc(&d_in, ARRAY_BYTES);
        cudaMalloc(&d_out, ARRAY_BYTES);
	cudaMemcpy(d_in, h_in, ARRAY_BYTES, cudaMemcpyHostToDevice);
	// kernel call

	scan<<<1, ARRAY_SIZE, ARRAY_BYTES*2>>>(ARRAY_SIZE, d_in, d_out);
	
	// Check for errors
        cudaError_t error = cudaGetLastError();
        if (error != cudaSuccess)
        {
              printf("CUDA error: %s\n", cudaGetErrorString(error));
              exit(-1);
        }

	cudaDeviceSynchronize();
	cudaMemcpy(h_out, d_out, ARRAY_BYTES, cudaMemcpyDeviceToHost);
	
	printf("Integral cuda: %.5f\n", h_out[ARRAY_SIZE-1]*16/ARRAY_SIZE);
	
	free(h_in);
	free(h_out);
	cudaFree(d_in);
	cudaFree(d_out);
	return 0;
}
