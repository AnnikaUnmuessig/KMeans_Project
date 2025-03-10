/*
 * k-Means clustering algorithm
 *
 * Reference sequential version (Do not modify this code)
 *
 * Parallel computing (Degree in Computer Engineering)
 * 2022/2023
 *
 * Version: 1.0
 *
 * (c) 2022 Diego García-Álvarez, Arturo Gonzalez-Escribano
 * Grupo Trasgo, Universidad de Valladolid (Spain)
 *
 * This work is licensed under a Creative Commons Attribution-ShareAlike 4.0 International License.
 * https://creativecommons.org/licenses/by-sa/4.0/
 */
#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include <float.h>
#include <omp.h>
#include <mpi.h>

#define MAXLINE 2000
#define MAXCAD 200

//Macros
#define MIN(a,b) ((a) < (b) ? (a) : (b))
#define MAX(a,b) ((a) > (b) ? (a) : (b))

/* 
Function showFileError: It displays the corresponding error during file reading.
*/
void showFileError(int error, char* filename)
{
	printf("Error\n");
	switch (error)
	{
		case -1:
			fprintf(stderr,"\tFile %s has too many columns.\n", filename);
			fprintf(stderr,"\tThe maximum number of columns has been exceeded. MAXLINE: %d.\n", MAXLINE);
			break;
		case -2:
			fprintf(stderr,"Error reading file: %s.\n", filename);
			break;
		case -3:
			fprintf(stderr,"Error writing file: %s.\n", filename);
			break;
	}
	fflush(stderr);	
}

/* 
Function readInput: It reads the file to determine the number of rows and columns.
*/
int readInput(char* filename, int *lines, int *samples)
{
    FILE *fp;
    char line[MAXLINE] = "";
    char *ptr;
    const char *delim = "\t";
    int contlines, contsamples = 0;
    
    contlines = 0;

    if ((fp=fopen(filename,"r"))!=NULL)
    {
        while(fgets(line, MAXLINE, fp)!= NULL) 
		{
			if (strchr(line, '\n') == NULL)
			{
				return -1;
			}
            contlines++;       
            ptr = strtok(line, delim);
            contsamples = 0;
            while(ptr != NULL)
            {
            	contsamples++;
				ptr = strtok(NULL, delim);
	    	}	    
        }
        fclose(fp);
        *lines = contlines;
        *samples = contsamples;  
        return 0;
    }
    else
	{
    	return -2;
	}
}

/* 
Function readInput2: It loads data from file.
*/
int readInput2(char* filename, float* data)
{
    FILE *fp;
    char line[MAXLINE] = "";
    char *ptr;
    const char *delim = "\t";
    int i = 0;
    
    if ((fp=fopen(filename,"rt"))!=NULL)
    {
        while(fgets(line, MAXLINE, fp)!= NULL)
        {         
            ptr = strtok(line, delim);
            while(ptr != NULL)
            {
            	data[i] = atof(ptr);
            	i++;
				ptr = strtok(NULL, delim);
	   		}
	    }
        fclose(fp);
        return 0;
    }
    else
	{
    	return -2; //No file found
	}
}

/* 
Function writeResult: It writes in the output file the cluster of each sample (point).
*/
int writeResult(int *classMap, int lines, const char* filename)
{	
    FILE *fp;
    
    if ((fp=fopen(filename,"wt"))!=NULL)
    {
        for(int i=0; i<lines; i++)
        {
        	fprintf(fp,"%d\n",classMap[i]);
        }
        fclose(fp);  
   
        return 0;
    }
    else
	{
    	return -3; //No file found
	}
}

/*

Function initCentroids: This function copies the values of the initial centroids, using their 
position in the input data structure as a reference map.
*/
void initCentroids(const float *data, float* centroids, int* centroidPos, int samples, int K)
{
	int i;
	int idx;
	for(i=0; i<K; i++)
	{
		idx = centroidPos[i];
		memcpy(&centroids[i*samples], &data[idx*samples], (samples*sizeof(float)));
	}
}

/*
Function euclideanDistance: Euclidean distance
This function could be modified
*/
float euclideanDistance(float *point, float *center, int samples)
{
	float dist=0.0;
	for(int i=0; i<samples; i++) 
	{
		dist+= (point[i]-center[i])*(point[i]-center[i]);
	}
	dist = sqrt(dist);
	return(dist);
}

/*
Function zeroFloatMatriz: Set matrix elements to 0
This function could be modified
*/
void zeroFloatMatriz(float *matrix, int rows, int columns)
{
	int i,j;
	for (i=0; i<rows; i++)
		for (j=0; j<columns; j++)
			matrix[i*columns+j] = 0.0;	
}

/*
Function zeroIntArray: Set array elements to 0
This function could be modified
*/
void zeroIntArray(int *array, int size)
{
	int i;
	for (i=0; i<size; i++)
		array[i] = 0;	
}



int main(int argc, char* argv[])
{

	//START CLOCK***************************************
	clock_t start, end;
	start = clock();
	//**************************************************
	/*
	* PARAMETERS
	*
	* argv[1]: Input data file
	* argv[2]: Number of clusters
	* argv[3]: Maximum number of iterations of the method. Algorithm termination condition.
	* argv[4]: Minimum percentage of class changes. Algorithm termination condition.
	*          If between one iteration and the next, the percentage of class changes is less than
	*          this percentage, the algorithm stops.
	* argv[5]: Precision in the centroid distance after the update.
	*          It is an algorithm termination condition. If between one iteration of the algorithm 
	*          and the next, the maximum distance between centroids is less than this precision, the
	*          algorithm stops.
	* argv[6]: Output file. Class assigned to each point of the input file.
	* */
	if(argc !=  7)
	{
		fprintf(stderr,"EXECUTION ERROR K-MEANS: Parameters are not correct.\n");
		fprintf(stderr,"./KMEANS [Input Filename] [Number of clusters] [Number of iterations] [Number of changes] [Threshold] [Output data file]\n");
		fflush(stderr);
		exit(-1);
	}

	// Reading the input data
	// lines = number of points; samples = number of dimensions per point
	int lines = 0, samples= 0;  
	
	int error = readInput(argv[1], &lines, &samples);
	if(error != 0)
	{
		showFileError(error,argv[1]);
		exit(error);
	}
	
	float *data = (float*)calloc(lines*samples,sizeof(float));
	if (data == NULL)
	{
		fprintf(stderr,"Memory allocation error.\n");
		exit(-4);
	}
	error = readInput2(argv[1], data);
	if(error != 0)
	{
		showFileError(error,argv[1]);
		exit(error);
	}

	// Parameters
	int K=atoi(argv[2]); 
	int maxIterations=atoi(argv[3]);
	int minChanges= (int)(lines*atof(argv[4])/100.0);
	float maxThreshold=atof(argv[5]);

	int *centroidPos = (int*)calloc(K,sizeof(int));
	float *centroids = (float*)calloc(K*samples,sizeof(float));
	int *classMap = (int*)calloc(lines,sizeof(int));

    if (centroidPos == NULL || centroids == NULL || classMap == NULL)
	{
		fprintf(stderr,"Memory allocation error.\n");
		exit(-4);
	}

	// Initial centrodis
	srand(0);
	int i;
	for(i=0; i<K; i++) 
		centroidPos[i]=rand()%lines;
	
	// Loading the array of initial centroids with the data from the array data
	// The centroids are points stored in the data array.
	initCentroids(data, centroids, centroidPos, samples, K);


	printf("\n\tData file: %s \n\tPoints: %d\n\tDimensions: %d\n", argv[1], lines, samples);
	printf("\tNumber of clusters: %d\n", K);
	printf("\tMaximum number of iterations: %d\n", maxIterations);
	printf("\tMinimum number of changes: %d [%g%% of %d points]\n", minChanges, atof(argv[4]), lines);
	printf("\tMaximum centroid precision: %f\n", maxThreshold);
	
	//END CLOCK*****************************************
	end = clock();
	printf("\nMemory allocation: %f seconds\n", (double)(end - start) / CLOCKS_PER_SEC);
	fflush(stdout);
	//**************************************************
	//START CLOCK***************************************
	start = clock();
	//**************************************************
	char *outputMsg = (char *)calloc(10000,sizeof(char));
	char line[100];

	int j;
	int class;
	float dist, minDist;
	int it=0;
	int changes = 0;
	float maxDist;

	//pointPerClass: number of points classified in each class
	//auxCentroids: mean of the points in each class
	int *pointsPerClass = (int *)malloc(K*sizeof(int));
	float *auxCentroids = (float*)malloc(K*samples*sizeof(float));
	float *distCentroids = (float*)malloc(K*sizeof(float)); 
	if (pointsPerClass == NULL || auxCentroids == NULL || distCentroids == NULL)
	{
		fprintf(stderr,"Memory allocation error.\n");
		exit(-4);
	}

/*
 *
 * START HERE: DO NOT CHANGE THE CODE ABOVE THIS POINT
 *
 */
int provided;
int size;
int rank;
int remainder;


// Initialize MPI environment
MPI_Init_thread(&argc , &argv, MPI_THREAD_FUNNELED, &provided);
MPI_Comm_size(MPI_COMM_WORLD, &size);
MPI_Comm_rank(MPI_COMM_WORLD, &rank);

if(provided != MPI_THREAD_FUNNELED){
    printf("WARNING: MPI_THREAD_FUNNELED not provided!\n");
}

// Divide lines by MPI processes
int lines_per_process = lines / size; 
remainder = lines % size;

int *send_counts = (int*)malloc(size * sizeof(int));  // Number of elements each process will receive
int *displacements = (int*)malloc(size * sizeof(int));  // Starting index for each process

int offset = 0;
for (int i = 0; i < size; i++) {
    send_counts[i] = (i < remainder) ? lines_per_process + 1 : lines_per_process; // Assign extra lines to the first 'remainder' processes
    displacements[i] = offset;
    offset += send_counts[i]; // Update offset for the next process
}

// Allocate memory for local data based on send_counts for each process
int local_lines = send_counts[rank];
float *local_data = (float*)malloc(local_lines * samples * sizeof(float)); // Each process gets its portion of data
int *local_classMap = (int*)malloc(local_lines * sizeof(int)); // Class assignments for each data point
float *local_aux_centroids = (float*)malloc(K * samples * sizeof(float)); // Sum of data points for each cluster
int *localPointsPerClass = (int*)malloc(K * sizeof(int)); // Local points per class for each process

// Initialize arrays to zero
zeroIntArray(local_classMap, local_lines); // Initialize local class map to zero
zeroFloatMatriz(local_aux_centroids, K, samples); // Initialize auxiliary centroids to zero
zeroIntArray(localPointsPerClass, K); // Initialize points per class to zero

if (!local_data || !local_classMap || !local_aux_centroids || !localPointsPerClass) {
    fprintf(stderr, "Memory allocation failed\n");
    MPI_Finalize();
    return -1;
}

if (rank == 0) {
    printf("Data distribution: \n");
    for (int i = 0; i < size; i++) {
        printf("Process %d will receive %d elements\n", i, send_counts[i]);
    }
}

// Scatter data among MPI processes using MPI_Scatterv
MPI_Scatterv(data, send_counts, displacements, MPI_FLOAT, local_data, local_lines * samples, MPI_FLOAT, 0, MPI_COMM_WORLD);
//samples = columns


//Start of the KMeans Loop (every process executes this with subdata)
	do{
		changes = 0;
		it++;
	
	//1. Calculate the distance from each point to the centroid
	//Assign each point to the nearest centroid.
    // OpenMP parallelization within each process
    //#pragma omp parallel for reduction(+:changes) //preventing race conditions when threads could accesss simultaneously the var changes

	//iterates over all local lines
    for (i = 0; i < local_lines; i++) {
        int global_index = displacements[rank] + i;  //get original index of point in dataset
        class = 1; //stores cluster assignment
        minDist = FLT_MAX; //stores smallest distance
        for (j = 0; j < K; j++) { //K: number of clusters
            dist = euclideanDistance(&data[global_index * samples], &centroids[j * samples], samples); //computes eucl dist between point centroid

            if (dist < minDist) {
                minDist = dist;
                class = j + 1;
            } //update class to cluster index
        }

        if (local_classMap[i] != class) {
            changes++;
        } //if class assignment has changed, increment changes
        local_classMap[i] = class; //updates local_classMap with cluster assignment
    }
	printf("Process %d, Changes: %d\n", rank, changes);

	zeroIntArray(localPointsPerClass, K);  // Zero out local points per class
    zeroFloatMatriz(local_aux_centroids, K, samples);  // Zero out local centroids

		// 2. Recalculates the centroids: calculates the mean within each cluster
		// Parallelize the centroid recalculation using OpenMP
		//#pragma omp parallel for

		//iterates over each local line, gets cluster assignment, increments count of points in that class
		for (i = 0; i < local_lines; i++) {
			class = local_classMap[i];
			localPointsPerClass[class - 1] += 1;
			for (j = 0; j < samples; j++) {
				#pragma omp atomic
				local_aux_centroids[(class - 1) * samples + j] += local_data[i * samples + j]; //adds each data point value to local_aux_centroids
			}
		}
	
		// Reduce the centroids within each process and send results to root process
       MPI_Reduce(local_aux_centroids, auxCentroids, K * samples, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);
       MPI_Reduce(localPointsPerClass, pointsPerClass, K, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);


	
		if (rank == 0) {
			// Compute the new centroids by averaging the sums from all processes using OpenMP
			//#pragma omp parallel for private(i, j)
			printf("New Centroids:\n");
			for (i = 0; i < K; i++) {
				printf("Centroid %d: ", i);
				for (j = 0; j < samples; j++) {
					if (pointsPerClass[i] != 0) {  // Avoid division by zero
						//Divides summed values by the number of points to compute the new centroid positions
						printf("%f ", auxCentroids[i * samples + j]);
						auxCentroids[i * samples + j] /= pointsPerClass[i];
					}
					printf("\n");
				}
			}
			
			maxDist = FLT_MIN;
			//#pragma omp parallel for reduction(max:maxDist)
			//computes how much centroids moved
			for (i = 0; i < K; i++) {
				distCentroids[i] = euclideanDistance(&centroids[i * samples], &auxCentroids[i * samples], samples);
				if (distCentroids[i] > maxDist) {
					maxDist = distCentroids[i];
				}
			}
			
			// Update centroids with new values
			memcpy(centroids, auxCentroids, K * samples * sizeof(float));
		}
		
		// Broadcast updated centroids to all processes
		MPI_Bcast(centroids, K * samples, MPI_FLOAT, 0, MPI_COMM_WORLD);

		
		if (rank == 0) {
			sprintf(line, "\n[%d] Cluster changes: %d\tMax. centroid distance: %f", it, changes, maxDist);
			outputMsg = strcat(outputMsg, line);
		} 
	
	
	printf("Changes: %d, MaxDist: %f, Iteration: %d\n", changes, maxDist, it);


	} while((changes>minChanges) && (it<maxIterations) && (maxDist>maxThreshold));

	free(local_data);
    free(local_classMap);
    free(local_aux_centroids);
    free(localPointsPerClass);
    free(auxCentroids);
    free(pointsPerClass);
 
	MPI_Finalize();

/*
 *
 * STOP HERE: DO NOT CHANGE THE CODE BELOW THIS POINT
 *
 */
	// Output and termination conditions
	printf("%s",outputMsg);	

	//END CLOCK*****************************************
	end = clock();
	printf("\nComputation: %f seconds", (double)(end - start) / CLOCKS_PER_SEC);
	fflush(stdout);
	//**************************************************
	//START CLOCK***************************************
	start = clock();
	//**************************************************

	

	if (changes <= minChanges) {
		printf("\n\nTermination condition:\nMinimum number of changes reached: %d [%d]", changes, minChanges);
	}
	else if (it >= maxIterations) {
		printf("\n\nTermination condition:\nMaximum number of iterations reached: %d [%d]", it, maxIterations);
	}
	else {
		printf("\n\nTermination condition:\nCentroid update precision reached: %g [%g]", maxDist, maxThreshold);
	}	

	// Writing the classification of each point to the output file.
	error = writeResult(classMap, lines, argv[6]);
	if(error != 0)
	{
		showFileError(error, argv[6]);
		exit(error);
	}

	//Free memory
	free(data);
	free(classMap);
	free(centroidPos);
	free(centroids);
	free(distCentroids);
	free(pointsPerClass);
	free(auxCentroids);

	//END CLOCK*****************************************
	end = clock();
	printf("\n\nMemory deallocation: %f seconds\n", (double)(end - start) / CLOCKS_PER_SEC);
	fflush(stdout);
	//***************************************************/
	return 0;
}
