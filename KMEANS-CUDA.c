/*
 * k-Means clustering algorithm
 *
 * CUDA version
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
#include <cuda.h>


#define MAXLINE 2000
#define MAXCAD 200

//Macros
#define MIN(a,b) ((a) < (b) ? (a) : (b))
#define MAX(a,b) ((a) > (b) ? (a) : (b))

/*
 * Macros to show errors when calling a CUDA library function,
 * or after launching a kernel
 */
#define CUDA_CHECK( a )	{ \
	cudaError_t ok = a; \
	if ( ok != cudaSuccess ) \
		fprintf(stderr, "-- Error CUDA call in line %d: %s\n", __LINE__, cudaGetErrorString( ok ) ); \
	}
#define CHECK_CUDA_LAST()	{ \
	cudaError_t ok = cudaGetLastError(); \
	if ( ok != cudaSuccess ) \
		fprintf(stderr, "-- Error CUDA last in line %d: %s\n", __LINE__, cudaGetErrorString( ok ) ); \
	}

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

// Kernel for calculating the distance from each point to the centroids
__global__ void calculateDistancesKernel(float* d_data, float* d_centroids, int* d_classMap, int lines, int samples, int K) {
    int i = blockIdx.x * blockDim.x + threadIdx.x; //computes global thread index
    if (i < lines) {
        float minDist = FLT_MAX;
        int class = -1;

        for (int j = 0; j < K; j++) {
            float dist = 0;
            for (int k = 0; k < samples; k++) {
                float diff = d_data[i * samples + k] - d_centroids[j * samples + k];
                dist += diff * diff;
            }
            dist = sqrtf(dist);

            if (dist < minDist) {
                minDist = dist;
                class = j;
            }
        }

        d_classMap[i] = class;
    }
}

// Kernel for updating centroids
__global__ void updateCentroidsKernel(float* d_data, int* d_classMap, float* d_auxCentroids, int* d_pointsPerClass, int lines, int samples, int K) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < lines) {
        int class = d_classMap[i];
        atomicAdd(&d_pointsPerClass[class], 1);

        for (int j = 0; j < samples; j++) {
            atomicAdd(&d_auxCentroids[class * samples + j], d_data[i * samples + j]);
        }
    }
}

// Kernel for finalizing centroids (averaging the sums)
__global__ void finalizeCentroidsKernel(float* d_auxCentroids, int* d_pointsPerClass, float* d_centroids, int K, int samples) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < K) { 
        int count = d_pointsPerClass[i];
        for (int j = 0; j < samples; j++) {
            if (count > 0) {
                d_centroids[i * samples + j] = d_auxCentroids[i * samples + j] / count;
            }
        }
    }
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

float *d_data, *d_centroids, *d_auxCentroids;
int *d_classMap, *d_pointsPerClass;

// Allocate memory on GPU
CUDA_CHECK(cudaMalloc(&d_data, lines * samples * sizeof(float)));
CUDA_CHECK(cudaMalloc(&d_centroids, K * samples * sizeof(float)));
CUDA_CHECK(cudaMalloc(&d_classMap, lines * sizeof(int)));
CUDA_CHECK(cudaMalloc(&d_pointsPerClass, K * sizeof(int)));
CUDA_CHECK(cudaMalloc(&d_auxCentroids, K * samples * sizeof(float)));

// Memory transfer between host and device memory
CUDA_CHECK(cudaMemcpy(d_data, data, lines * samples * sizeof(float), cudaMemcpyHostToDevice));
CUDA_CHECK(cudaMemcpy(d_centroids, centroids, K * samples * sizeof(float), cudaMemcpyHostToDevice));
CUDA_CHECK(cudaMemcpy(d_classMap, classMap, lines * sizeof(int), cudaMemcpyHostToDevice));

dim3 blockSize(256);
dim3 gridSizeDist((lines + blockSize.x - 1) / blockSize.x);
dim3 gridSizeCentroids((K + blockSize.x - 1) / blockSize.x);

do {
    it++;

    // Step 1: Assign each point to the nearest centroid
    calculateDistancesKernel<<<gridSizeDist, blockSize>>>(d_data, d_centroids, d_classMap, lines, samples, K);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy updated classMap back to check for changes
    CUDA_CHECK(cudaMemcpy(classMap, d_classMap, lines * sizeof(int), cudaMemcpyDeviceToHost));

    changes = 0;
    for (i = 0; i < lines; i++) {
        if (classMap[i] != class) {
            changes++;
        }
    }

    // Step 2: Reset auxiliary data for new centroid calculations
    CUDA_CHECK(cudaMemset(d_pointsPerClass, 0, K * sizeof(int)));
    CUDA_CHECK(cudaMemset(d_auxCentroids, 0, K * samples * sizeof(float)));

    // Step 3: Sum up the points assigned to each centroid
    updateCentroidsKernel<<<gridSizeDist, blockSize>>>(d_data, d_classMap, d_auxCentroids, d_pointsPerClass, lines, samples, K);
    CUDA_CHECK(cudaDeviceSynchronize());

	//need to copy d_auxCentroids or d_pointsPerClass back to host??

    // Step 4: Compute new centroid positions
    finalizeCentroidsKernel<<<gridSizeCentroids, blockSize>>>(d_auxCentroids, d_pointsPerClass, d_centroids, K, samples);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Step 5: Transfer new centroids back to host
    CUDA_CHECK(cudaMemcpy(centroids, d_centroids, K * samples * sizeof(float), cudaMemcpyDeviceToHost));

    // Compute max distance moved by centroids for convergence check
    maxDist = FLT_MIN;
    for (i = 0; i < K; i++) {
        distCentroids[i] = euclideanDistance(&centroids[i * samples], &auxCentroids[i * samples], samples);
        if (distCentroids[i] > maxDist) {
            maxDist = distCentroids[i];
        }
    }

    sprintf(line, "\n[%d] Cluster changes: %d\tMax. centroid distance: %f", it, changes, maxDist);
    outputMsg = strcat(outputMsg, line);

} while ((changes > minChanges) && (it < maxIterations) && (maxDist > maxThreshold)); // Termination conditions

// Free GPU memory
cudaFree(d_data);
cudaFree(d_centroids);
cudaFree(d_classMap);
cudaFree(d_pointsPerClass);
cudaFree(d_auxCentroids);


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

