/*
 INPUT: mask in decimal system
	condition "periodic" or "constant"
	
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
	int N = 225, mask_dec, n_steps = 225, ierr;
	int *a = (int *)calloc(sizeof(int), N);
	int *a_next = (int *)malloc(sizeof(int)*N);
	int *mask = (int *)calloc(sizeof(int), 8);
	char *condition;
	double start, end;

	int psize, prank;
	MPI_Status status;
	MPI_Request request;
		
	ierr = MPI_Init(&argc, &argv);
        ierr = MPI_Comm_rank(MPI_COMM_WORLD, &prank);
        ierr = MPI_Comm_size(MPI_COMM_WORLD, &psize);
	
	if (psize > N)
	{
		printf("Error: number of elements should be larger than number of processes.\n");
		exit(-1);
	}	
	if (argc < 3)
	{
		printf("Error: wrong number of arguments.\n");
		exit(-1);
	}
	
	mask_dec = atoi(argv[1]);

	condition = (char *)malloc(strlen(argv[2] + 1)*sizeof(char));
	condition = argv[2];

	if ((strcmp(condition, "periodic") != 0) && strcmp(condition, "constant") != 0)
	{
		printf("Error: wrong condition. Try 'periodic' or 'constant'\n");
		exit(-1);
	}
	

	FILE *f;
	f = fopen("out_auto.txt", "a+");

	if (prank == 0)
	{
		printf("Decimal system: %d\n", mask_dec);
		printf("Conditions: %s\n", condition);

		int *a = (int *)malloc(sizeof(int) * N);
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
	printf("%d\t", my_size - 2);
	int* my_a = (int *)calloc(sizeof(int), my_size);
	int* my_a_next = (int *)malloc(sizeof(int) * my_size);
	int* my_a_to_send = (int *)malloc(sizeof(int) * (my_size - 2));

	printf("I'm process %d, my_left is %d, my_right is %d, [left %d, right %d]\n", prank, my_left, my_right, left, right);
	
	unsigned int seed = (prank + 1)*123456789;
	for (int i = 0; i < my_size; i++)
	{
		my_a[i] = rand_r(&seed) % 2;
	}
	
	for (int i = 0; i < n_steps; i++)	
	{
		MPI_Barrier(MPI_COMM_WORLD);

		MPI_Send(&my_a[1], 1, MPI_INT, left, 0, MPI_COMM_WORLD);
		MPI_Send(&my_a[my_size - 2], 1, MPI_INT, right, 1, MPI_COMM_WORLD);
		MPI_Recv(&my_a[my_size - 1], 1, MPI_INT, right, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		MPI_Recv(&my_a[0], 1, MPI_INT, left, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

		step_const(my_a, my_size, mask, my_a_next);
		if (i % 1 == 0)
		{
			for (int j = 1; j < (my_size - 1); j++)
			{
				my_a_to_send[j - 1] = my_a[j];
			}

			MPI_Barrier(MPI_COMM_WORLD);	
			MPI_Gatherv(my_a_to_send, (my_size - 2), MPI_INT, a, size, displs, MPI_INT, 0, MPI_COMM_WORLD);

			if (prank == 0)
			{
				matrix_print_file(a, N, f);
			}
		}
	}
	
	if (prank == 0)
	{
		fclose(f);	
		free(a);
	}

	free(my_a);
	free(mask);
	MPI_Finalize();

	return 0;
}
