#include <stdio.h>
#include <math.h>

// Size
const int N = 64;

__global__ void Step(long int n, double *d_a, double *d_res)
{
        int globalidx = threadIdx.x + blockDim.x * blockIdx.x;
	
        double left_y, right_y; 
	int x = globalidx / n;
        double left_x, right_x; 
	int y = globalidx - n * x;

 	if (globalidx < n*n) 
	{
		left_y = y - 1 < 0 ? 0 : d_a[y - 1 + x*n];
		left_x = x - 1 < 0 ? 0 : d_a[y + (x - 1)*n];
		right_y = y + 1 > n - 1 ? 0 : d_a[y + 1 + x*n];
		right_x = x + 1 > n - 1 ? 0 : d_a[y + (x + 1)*n];

		d_res[y + x*n] = (left_x + left_y + right_x + right_y) / 4.;

		// Boundary conditions
		if (x == 0 || y == n - 1 || x == n - 1) {d_res[y + x*n] = 0;}	
	}

	// Boundary conditions
  	__syncthreads();
	if (y == 0) d_res[x*n] = 1;

  	__syncthreads();
	d_a[y + x*n] = d_res[y + x*n];      
	
}

void print_matrix_to_file(FILE *fp, double *matrix, long int n)
{
	for (int i = 0; i < n; i++)
	{
		for (int j = 0; j < n; j++)
		{
			fprintf(fp, "%f\t", matrix[i*N + j]);
		}
	}
	fprintf(fp, "\n");
}

void print_matrix(double *matrix, long int N)
{	
	if (N < 10)
	{
		// Printing matrix
		for (int i = 0; i < N; i++)
		{
			for (int j = 0; j < N; j++)
			{	
				printf("%f\t", matrix[i*N + j]);
			}
			printf("\n");
		}	
	}
}

int main(void)
{
	int steps = 100;
	FILE *fp;
	fp = fopen("out.txt", "w");

	// Initial conditions
	double *h_a = (double *) calloc(sizeof(double), N * N);
	double *h_res = (double *) malloc(sizeof(double) * N * N);

	double *d_a;
	double *d_res;
	cudaMalloc(&d_a, sizeof(double) * N * N);
	cudaMalloc(&d_res, sizeof(double) * N * N);

	for (int i = 0; i < N; i++)
	{
		h_a[i*N] = 1;		
	}
	
	print_matrix(h_a, N);	
	cudaMemcpy(d_a, h_a, sizeof(double) * N * N, cudaMemcpyHostToDevice);
	

	// Sizes of blocks of threads and grid of blocks depending on N	
	int block_size, grid_size;
	if (N * N > 1024)
	{
		block_size = 1024;
		grid_size = N * N / 1024;
	}
	else
	{
		block_size = N * N;
		grid_size = 1;
	}

	dim3 dimBlock(block_size);
	dim3 dimGrid(grid_size);

        // Loop
	for (int k = 0; k < steps; k++)
	{
		Step<<<dimGrid, dimBlock>>>(N, d_a, d_res);

		// Check for errors
		cudaError_t error = cudaGetLastError();
		if (error != cudaSuccess)
		{
			printf("CUDA error: %s\n", cudaGetErrorString(error));
			exit(-1);
		}

		cudaMemcpy(h_a, d_a, sizeof(double)*N*N, cudaMemcpyDeviceToHost);
		print_matrix_to_file(fp, h_a, N);		
		print_matrix(h_a, N);
	}
	
	//print_matrix(h_res, N);
	cudaDeviceSynchronize();

	free(h_a);
	free(h_res);
	cudaFree(d_a);
	cudaFree(d_res);
	fclose(fp);

	return 0;
	
}
