mpicc -fopenmp -o KMEANS-MPI_and_OpenMP KMEANS-MPI_and_OpenMP.c -lm for compilation
mpirun -np 4 ./KMEANS-MPI_and_OpenMP test_files/input2D.inp 4 100 1 0.01 output.txt for running code MPI, OpenMP Implementation
