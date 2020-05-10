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
	
	printf("Binary mask: \n");
	matrix_print(mask, 8);
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

void step_periodic(int *a, int N, int* mask, int* a_next)
{
	int ind, i_pred, i_next;
	
	for (int i = 0; i < N; i++)
	{
		i_pred = (i - 1) < 0 ? (N - 1) : (i - 1);
		i_next = (i + 1) > (N - 1) ? 0 : (i + 1);
 
		ind = 7 - a[i_pred]*4 - a[i]*2 - a[i_next];
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
	int N = 100, mask_dec, n_steps = 100;
	int *a = (int *)calloc(sizeof(int), N);
	int *a_next = (int *)malloc(sizeof(int)*N);
	int *mask = (int *)calloc(sizeof(int), 8);
	char *condition;
	FILE *fp;
	
	fp = fopen("out_auto.txt", "a+");

	srand(time(NULL));
	
	if (argc < 3)
	{
		printf("Error: wrong number of arguments.");
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
	
	printf("Decimal system: %d\n", mask_dec);
	printf("Conditions: %s\n", condition);
	
	// initial conditions	
	if (mask_dec == 110)
	{
		a[N - 1] = 1;	
	}
	else 
	{
		if (mask_dec == 90)
		{
			a[N / 2] = 1;
		}
		else
		{
			if (mask_dec == 30)
			{
				a[0] = 1;
			}
			else
			{
				for (int i = 0; i < N; i++)
				{
					a[i] = rand() % 2;
				}
			}	
		}
	}
	
	// matrix_print(a, N);
	create_binary_mask(mask, mask_dec);

	for (int i = 0; i < N; i++)	
	{
		if (strcmp(condition, "periodic") == 0)
		{
			step_periodic(a, N, mask, a_next);
		}
		else
		{
			step_const(a, N, mask, a_next);
		}
	//	matrix_print(a, N);
		matrix_print_file(a, N, fp);
	}
	
	fclose(fp);	
	free(a);
	free(mask);
	return 0;
}
