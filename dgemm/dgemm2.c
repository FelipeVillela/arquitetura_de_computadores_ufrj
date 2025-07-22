#include <time.h>      
#include <stdio.h>     
#include <stdlib.h>    


void dgemm (int n, double* A, double* B, double* C)
{
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
        {
            double cij = C[i+j*n]; /* cij = C[i][j] */
            for( int k = 0; k < n; k++ )
                cij += A[i+k*n] * B[k+j*n]; /* cij += A[i][k]*B[k][j] */
            C[i+j*n] = cij; /* C[i][j] = cij */
        }
}


void generate_matrices(int n, double* A, double* B, double* Cm) {
    srand(1);  
    int total_elements = n * n;
    for (int i = 0; i < total_elements; ++i) {
        A[i] = (rand() % 100) * 0.1;
        B[i] = (rand() % 100) * 0.1;

        if (Cm != NULL) {
            Cm[i] = 0.0;  
        }
    }
}

double get_time_seconds() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec / 1e9;
}

int main(int argc, char* argv[]){
    if (argc < 2) {
        return 1;
    }

    int n = atoi(argv[1]);

    if (n <= 0) {
        return 1;
    }

    size_t matrix_size = (size_t)n * n;
    double* A = (double*)malloc(matrix_size * sizeof(double));
    double* B = (double*)malloc(matrix_size * sizeof(double));
    double* C = (double*)malloc(matrix_size * sizeof(double));

    if (A == NULL || B == NULL || C == NULL) {
        free(A);
        free(B);
        free(C);
        return 1;
    }

    generate_matrices(n, A, B, C);
    double start_time = get_time_seconds();
    dgemm(n, A, B, C);
    double end_time = get_time_seconds();
    double elapsed = end_time - start_time;
    printf("%f \n", elapsed);

    free(A);
    free(B);
    free(C);

    return 0;
}
