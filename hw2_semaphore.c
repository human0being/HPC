/*
INPUT:
	numberOfPoints - double, number of discretization points;
	num_threads - int, number of threads;
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <pthread.h>
//#include <semaphore.h>
#include <sys/time.h>
#include <dispatch/dispatch.h>

#define MAX_THREADS 8
#define TIME 1000000.0
#define NUM_STEPS 200000

double sum = 0.0, a=0, b=1;
double width;
//int flag=0;

/*create semaphore*/
dispatch_semaphore_t semaphore;

/*create mutex for integral*/
//pthread_mutex_t integral_mutex;

struct intervals{
	int id;
	int left;
	int right;
};

double f(double x){
	return sin(x)/2.;
}

void * PartCalc(void *info_thread)
{	
	struct intervals * info = (struct intervals *) info_thread;
	double part_sum = 0;

	
	printf("Hello. My number:  %d, left: %d, right: %d\n",  info->id, info->left, info->right);
	
	for (int i=info->left; i<info->right; i++){
		double x_back = a + width*(i - 1);
		double x_forward = a + width*(i + 1);
		part_sum += pow(width*width + pow((f(x_back) - f(x_forward))/2.,2), 0.5);	
	}
	
	
	// Locking and unlocking controlling the access
	
	dispatch_semaphore_wait(semaphore, DISPATCH_TIME_FOREVER);
			
	sum += part_sum;
	
	dispatch_semaphore_signal(semaphore);

	//flag++;

	// could return data with this function
	pthread_exit(NULL);
}


int main(int argc, char *argv[])
{
	FILE *fp;

	/*open the file for writing*/
	fp = fopen ("./out.txt", "a+");
	
//	fprintf (fp, "znachok chto tu durachok");
	
	pthread_t *threads;
	double time_start, time_end;	
	double analytic_sol, diff;
	struct timeval tv;
	struct timezone tz;
	double numberOfPoints;
	int  num_threads;
	struct intervals *threads_info;
	int pthread_out;
		
	/*Initialize a mutex for integral*/
	//int pthread_out = pthread_mutex_init(&integral_mutex, NULL);

	//Initialize semaphore
	//sem_init(&mutex, 0, 1);
	semaphore = dispatch_semaphore_create(1);
	
	if (argc != 3)
	{
		printf("Wrong number of arguements.\n");
		return 1;
	}
	else
	{
		numberOfPoints = atoi(argv[1]);
		num_threads = atoi(argv[2]);
	}	
	
	printf("Number of intervals : %f\n", numberOfPoints);
	printf("Number of threads : %d\n", num_threads);
	
	if (num_threads >  MAX_THREADS || numberOfPoints == 0)
	{
		printf("Number of intervals or threads is inappropriate.\n");
		exit(-1);
	}
	
	if (num_threads > numberOfPoints){
		num_threads = numberOfPoints;
	}
	
	//discretization
	width = (b - a)/numberOfPoints;

	// calculation of start time
	gettimeofday(&tv, &tz);
	time_start = (double)tv.tv_sec + (double)tv.tv_usec / TIME;
	
	threads = (pthread_t *)malloc(sizeof(pthread_t) * num_threads);
	threads_info = (struct intervals *)malloc(sizeof(struct intervals) * num_threads);

	// Computation of integral
	for (int i = 0; i < num_threads; i++)
	{	
		threads_info[i].id = i;
		threads_info[i].left = i*numberOfPoints/num_threads;
		threads_info[i].right = (i + 1)*numberOfPoints/num_threads;
		
		pthread_out = pthread_create(&threads[i], NULL, PartCalc, &threads_info[i]);
		if (pthread_out)
		{
			printf("Error, pthread_create() doesn't work.\n");
			exit(-1);
		}
	}
	
	// block until all the threads complete
	for (int i = 0; i < num_threads; i++)
	{
		pthread_out =  pthread_join(threads[i], NULL);
		if (pthread_out)
		{
			printf("Error, pthread_join() doesn't work.\n");
			exit(-1);
		}
		
	}	

		
	sum += 1/2.*(pow(width*width + pow((f(a + width) - f(a)), 2), 0.5) \
						+ pow(width*width + pow((f(b) - f(b - width)),2), 0.5));



	// calculation of end time
	gettimeofday(&tv, &tz);
	time_end = (double)tv.tv_sec + (double)tv.tv_usec / TIME;
	diff = time_end - time_start;

	printf("Time of execution: %f\n", diff);
	printf("The integral: %f\n", sum);
	fprintf(fp, "%d\t%f\n", num_threads, diff);

	//sem_destroy(&mutex);
	dispatch_release(semaphore);
	free(threads_info);
	free(threads);	
	fclose(fp);
	return 0;	
}
