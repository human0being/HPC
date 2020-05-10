// INPUT: A word which processes will be transfering to each other during ~20 sec
// note: receiver should have been written like that
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h> 
#include <string.h>

char* concat(char *s1, char *s2)
{
    char *result =(char *) malloc(strlen(s1) + strlen(s2) + 1);

    strcpy(result, s1);
    strcat(result, s2);

    return result;
}

int main(int argc, char ** argv)
{
	int psize, prank;
	MPI_Status status;
	MPI_Request request;

	int number, flag=0, count=0;
	int ierr, N = 15;
	int sender = 0, reciever;
	double start, end;
	char rank_buff[5];
	char *name;

	name = concat(argv[1], argv[1]);
	ierr = MPI_Init(&argc, &argv);
	ierr = MPI_Comm_rank(MPI_COMM_WORLD, &prank);
	ierr = MPI_Comm_size(MPI_COMM_WORLD, &psize);	
	
	start = MPI_Wtime();
	
	while (1)
	{
		if (prank == sender)
		{
			do
			{
				reciever = rand() % psize;
			} while(sender == reciever);
			
			//printf("I'm process %d, send data to %d\n", sender, reciever);
			MPI_Ssend(name, strlen(name) + 1, MPI_CHAR, reciever, 0, MPI_COMM_WORLD);
			MPI_Send(&count, 1, MPI_INT, reciever, 1, MPI_COMM_WORLD);
			//printf("Sent count %d.\n", count);
           		sender = reciever;
		}
		else
		{
			while(!flag)
			{
				MPI_Iprobe(MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &flag, &status);
			//	sleep(1);
			}

			MPI_Recv(name, strlen(name) + 1, MPI_CHAR, status.MPI_SOURCE, 0, MPI_COMM_WORLD, &status);
			MPI_Recv(&count, 1, MPI_INT, status.MPI_SOURCE, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			count++;
			// printf("Process %d received message %s\n", prank, name);
			flag = 0;
			sender = prank;
			end = MPI_Wtime();
			
			if ((end - start) > 20)
			{
				//printf("Time: %f.\n", end - start);
				//printf("Number of throws: %d.\n", count);
				//printf("I'm process %d, want to kill ALL.\n", prank);

				printf("%lu\t%d\t%f\t%f\t%f\n", strlen(name), count, end - start, (double) (end - start) / (double) count, (double) count * strlen(name) / (end - start) / 1024. / 1024.) ;
				MPI_Abort(MPI_COMM_WORLD, 911);
			}

		}
	}
	
	return 0;
}
