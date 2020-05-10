/*
 INPUT: mask in decimal system
	number of elements in the array to iterate
	
binary mask for the next arangement:
111 110 101 100 011 011 010 001 000

*/

#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <time.h>
#include <string.h>

void matrix_print(int *a, int N)
{
	for (int i = 0; i < N; i++)
	{
		printf("%d\t", a[i]);
		if ((i + 1) % 10 == 0 || i == (N - 1))
		{
			printf("\n");
		}
	}
}

void create_binary_mask(int *mask, int mask_dec)
{
	int i = 0;

	while(mask_dec != 0)	
	{
		if (mask_dec % 2 == 1)
		{
			mask[7 - i] = 1;
		}

		mask_dec = mask_dec / 2;
		i++;

		if (i > 8)
		{
			printf("Error: please, enter lower number < 256.\n");
			exit(-1);
		}
	}
	
}

void matrix_print_file(int *matrix, int N, FILE *fp)
{
	for (int i = 0; i < N; i++)
	{
		fprintf(fp, "%d\t", matrix[i]);
	}

	fprintf(fp, "\n");
}

void step_const(int *a, int N, int* mask, int* a_next)
{
	int ind;
	a_next[0] = a[0];
	a_next[N - 1] = a[N - 1];

	for (int i = 1; i < (N - 1); i++)
	{
		ind = 7 - a[i - 1]*4 - a[i]*2 - a[i + 1];
		a_next[i] = mask[ind];
	}	
	
	//copy a_next to a
	for (int i = 0; i < N; i++)
	{
		a[i] = a_next[i];
	}
}

int main(int argc, char** argv)
{
	int N, mask_dec, n_steps = 1000, ierr;
	int *a = (int *)calloc(sizeof(int), N);
	int *a_next = (int *)malloc(sizeof(int)*N);
	int *mask = (int *)calloc(sizeof(int), 8);
	double start, end;

	int psize, prank;
	MPI_Status status;
	MPI_Request request;
		
	ierr = MPI_Init(&argc, &argv);
        ierr = MPI_Comm_rank(MPI_COMM_WORLD, &prank);
        ierr = MPI_Comm_size(MPI_COMM_WORLD, &psize);
	
	mask_dec = atoi(argv[1]);
	N = atoi(argv[2]);

	if (psize > N)
	{
		printf("Error: number of elements should be larger than number of processes.\n");
		exit(-1);
	}	
	if (argc < 2)
	{
		printf("Error: wrong number of arguments.\n");
		exit(-1);
	}

	create_binary_mask(mask, mask_dec);
	
	int my_left = N / psize * prank;
	int my_right =  N / psize * (prank + 1);
	int *size = (int *)calloc(sizeof(int), psize);
	int *displs = (int *)calloc(sizeof(int), psize);
	
	for (int k = 0; k < psize; k++)
	{
		size[k] = (int) N / (double) psize;
		displs[k] = k * size[k];
	}

	size[psize - 1] += N % psize;

	if (prank == (psize - 1))
	{
		my_right = N; 
	}

	int right = (prank + 1) % psize;
	int left = (prank - 1) < 0 ? psize - 1 : prank - 1;
	int my_size = my_right - my_left + 2;	

	int* my_a = (int *)calloc(sizeof(int), my_size);
	int* my_a_next = (int *)malloc(sizeof(int) * my_size);
	int* my_a_to_send = (int *)malloc(sizeof(int) * (my_size - 2));
	
	unsigned int seed = (prank + 1)*123456789;
	for (int i = 0; i < my_size; i++)
	{
		my_a[i] = rand_r(&seed) % 2;
	}
	
	if (prank == 0)
	{
		start = MPI_Wtime();
		int *a = (int *)malloc(sizeof(int) * N);
	}
	for (int i = 0; i < n_steps; i++)	
	{
		MPI_Barrier(MPI_COMM_WORLD);

		MPI_Send(&my_a[1], 1, MPI_INT, left, 0, MPI_COMM_WORLD);
		MPI_Send(&my_a[my_size - 2], 1, MPI_INT, right, 1, MPI_COMM_WORLD);
		MPI_Recv(&my_a[my_size - 1], 1, MPI_INT, right, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		MPI_Recv(&my_a[0], 1, MPI_INT, left, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

		step_const(my_a, my_size, mask, my_a_next);
	}
	
	if (prank == 0)
	{
		end = MPI_Wtime();
		printf("%d\t%f\n", psize, (double) (end - start));
		free(a);
	}

	free(my_a);
	free(mask);
	MPI_Finalize();

	return 0;
}
