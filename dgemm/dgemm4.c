#include <x86intrin.h>
#include <time.h>
#include <stdio.h>
#include <stdlib.h>

#define UNROLL (4)

void dgemm (int n, double* A, double* B, double* C)
{
    for (int i = 0; i < n; i+=UNROLL*8)
        for (int j = 0; j < n; ++j){
            __m512d c[UNROLL];
            for (int r=0;r<UNROLL;r++)
                c[r] = _mm512_load_pd(C+i+r*8+j*n);

            for (int k = 0; k < n; k++ )
            {
                __m512d bb = _mm512_broadcastsd_pd(_mm_load_sd(B+j*n+k)); 
                for (int r=0;r<UNROLL;r++)
                    c[r] = _mm512_fmadd_pd(_mm512_load_pd(A+n*k+r*8+i), bb, c[r]); 
            }

            for (int r=0;r<UNROLL;r++)
                _mm512_store_pd(C+i+r*8+j*n, c[r]); 
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

    // Valida formato da matriz
    if (n % (UNROLL * 8) != 0) {
        return 1;
    }

    size_t matrix_size = (size_t)n * n;
    double* A = (double*)_mm_malloc(matrix_size * sizeof(double), 64);
    double* B = (double*)_mm_malloc(matrix_size * sizeof(double), 64);
    double* C = (double*)_mm_malloc(matrix_size * sizeof(double), 64);

    if (A == NULL || B == NULL || C == NULL) {
        _mm_free(A);
        _mm_free(B);
        _mm_free(C);
        return 1;
    }

    generate_matrices(n, A, B, C);
    double start_time = get_time_seconds();
    dgemm(n, A, B, C);
    double end_time = get_time_seconds();
    double elapsed = end_time - start_time;
    printf("%f \n", elapsed);

    _mm_free(A);
    _mm_free(B);
    _mm_free(C);

    return 0;
}
