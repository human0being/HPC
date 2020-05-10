//N is the number of throws

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

	int number, flag=0, count;
	int ierr, N = 15;
	int sender = 0, reciever;
	double time_elapsed;
	char rank_buff[5];
	char *name = (char *)malloc(sizeof(char)*10);
	char *my_name = (char *)malloc(sizeof(char)*15);
	
	ierr = MPI_Init(&argc, &argv);
	ierr = MPI_Comm_rank(MPI_COMM_WORLD, &prank);
	ierr = MPI_Comm_size(MPI_COMM_WORLD, &psize);
	
	time_elapsed = MPI_Wtime();
	
	sprintf(rank_buff, "%d", prank);
	my_name = concat((char *)"ananas_", rank_buff);
	
	while(1)
	{
		if (prank == sender)
		{
			number = -1;
			do
			{
				reciever = rand() % psize;
			} while(sender == reciever);
			
			//name = concat(name, my_name);
			printf("I'm process %d, send data to %d\n", sender, reciever);
			//MPI_Send(&number, 1, MPI_INT, reciever, 0, MPI_COMM_WORLD);
			MPI_Send(my_name, strlen(my_name) + 1, MPI_CHAR, reciever, 0, MPI_COMM_WORLD);
           		sender = reciever;
		}
		else
		{
			while(!flag)
			{
				MPI_Iprobe(MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &flag, &status);
				sleep(1);
			}
	
			//MPI_Recv(&number, 1, MPI_INT, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &status);

			MPI_Get_count(&status, MPI_CHAR, &count);
			char *name = (char *)malloc(sizeof(char)*(count + 1));
			MPI_Recv(name, count + 1, MPI_CHAR, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &status);
			my_name = concat(name, my_name);
			printf("Process %d received message %s\n", prank, name);
			flag = 0;
			sender = prank;
			}

			if (strlen(name) / strlen(my_name) == N)
			{	
					
						time_elapsed = MPI_Wtime() - time_elapsed;
						printf("The length of the message is %lu.\n", strlen(name)+1);
						printf("Total number of throws is %lu.\n", strlen(name) / strlen(my_name));
						printf("Process %d wants to KILL ALL.\n", prank);
						free(name);
						free(my_name);
						printf("Time %.5f\n", time_elapsed);
						MPI_Abort(MPI_COMM_WORLD, 911);
						break;
			}
		
	}
//	free(name);
//	free(my_name);	
	ierr = MPI_Finalize();
	
	return 0;
}
