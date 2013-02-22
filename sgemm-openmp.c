#include <stdlib.h>
#include <stdio.h>
#include <nmmintrin.h>

void sgemm(int m, int n, float *A, float *C) {
    int ii, jj, i, j, k;
    int block_size = 2048/n;
    int last_block = (m/block_size*block_size)/16*16;
    int unroll16 = m/16*16;
    int unroll4 = m/4*4;
    float *A_col, *C_col1, *C_col2, *C_col3, *C_col4;
    __m128 c1a, c2a, c3a, c4a, r1a, r1b, r1c, r1d;
#pragma omp parallel private(ii, jj, i, j, k, A_col, C_col1, C_col2, C_col3, C_col4, c1a, c2a, c3a, c4a, r1a, r1b, r1c, r1d)
{
    //handle everything that can be put in a block
    #pragma omp for
    for (jj = 0; jj < last_block; jj += block_size) {
        for (ii = 0; ii < last_block; ii += block_size) {
            for (k = 0; k < n; k += 1) {
                A_col = A + k*m;
                for (j = 0; j < block_size; j += 4) {
                    C_col1 = C + (jj + j) * m;
                    C_col2 = C + (jj + j + 1) * m;
                    C_col3 = C + (jj + j + 2) * m;
                    C_col4 = C + (jj + j + 3) * m;
                    c1a = _mm_load1_ps(A_col + jj + j);
                    c2a = _mm_load1_ps(A_col + jj + j + 1);
                    c3a = _mm_load1_ps(A_col + jj + j + 2);
                    c4a = _mm_load1_ps(A_col + jj + j + 3);
                    for (i = 0; i < block_size; i += 16) {
                        r1a = _mm_loadu_ps(A_col + ii + i);
                        r1b = _mm_loadu_ps(A_col + ii + i + 4);
                        r1c = _mm_loadu_ps(A_col + ii + i + 8);
                        r1d = _mm_loadu_ps(A_col + ii + i + 12);

                        _mm_storeu_ps(C_col1 + ii + i, _mm_add_ps(_mm_loadu_ps(C_col1 + ii + i), _mm_mul_ps(r1a, c1a)));
                        _mm_storeu_ps(C_col1 + ii + i + 4, _mm_add_ps(_mm_loadu_ps(C_col1 + ii + i + 4), _mm_mul_ps(r1b, c1a)));
                        _mm_storeu_ps(C_col1 + ii + i + 8, _mm_add_ps(_mm_loadu_ps(C_col1 + ii + i + 8), _mm_mul_ps(r1c, c1a)));
                        _mm_storeu_ps(C_col1 + ii + i + 12, _mm_add_ps(_mm_loadu_ps(C_col1 + ii + i + 12), _mm_mul_ps(r1d, c1a)));

                        _mm_storeu_ps(C_col2 + ii + i, _mm_add_ps(_mm_loadu_ps(C_col2 + ii + i), _mm_mul_ps(r1a, c2a)));
                        _mm_storeu_ps(C_col2 + ii + i + 4, _mm_add_ps(_mm_loadu_ps(C_col2 + ii + i + 4), _mm_mul_ps(r1b, c2a)));
                        _mm_storeu_ps(C_col2 + ii + i + 8, _mm_add_ps(_mm_loadu_ps(C_col2 + ii + i + 8), _mm_mul_ps(r1c, c2a)));
                        _mm_storeu_ps(C_col2 + ii + i + 12, _mm_add_ps(_mm_loadu_ps(C_col2 + ii + i + 12), _mm_mul_ps(r1d, c2a)));

                        _mm_storeu_ps(C_col3 + ii + i, _mm_add_ps(_mm_loadu_ps(C_col3 + ii + i), _mm_mul_ps(r1a, c3a)));
                        _mm_storeu_ps(C_col3 + ii + i + 4, _mm_add_ps(_mm_loadu_ps(C_col3 + ii + i + 4), _mm_mul_ps(r1b, c3a)));
                        _mm_storeu_ps(C_col3 + ii + i + 8, _mm_add_ps(_mm_loadu_ps(C_col3 + ii + i + 8), _mm_mul_ps(r1c, c3a)));
                        _mm_storeu_ps(C_col3 + ii + i + 12, _mm_add_ps(_mm_loadu_ps(C_col3 + ii + i + 12), _mm_mul_ps(r1d, c3a)));

                        _mm_storeu_ps(C_col4 + ii + i, _mm_add_ps(_mm_loadu_ps(C_col4 + ii + i), _mm_mul_ps(r1a, c4a)));
                        _mm_storeu_ps(C_col4 + ii + i + 4, _mm_add_ps(_mm_loadu_ps(C_col4 + ii + i + 4), _mm_mul_ps(r1b, c4a)));
                        _mm_storeu_ps(C_col4 + ii + i + 8, _mm_add_ps(_mm_loadu_ps(C_col4 + ii + i + 8), _mm_mul_ps(r1c, c4a)));
                        _mm_storeu_ps(C_col4 + ii + i + 12, _mm_add_ps(_mm_loadu_ps(C_col4 + ii + i + 12), _mm_mul_ps(r1d, c4a)));
                    }
                }
            }
        }
    }

if (m/block_size*block_size != m) {
    for (k = 0; k < n; k += 1) {
        A_col = A + k*m;
        // begin handling the right-most edge case
        #pragma omp for
        for (j = jj; j < unroll4; j += 4) {
            C_col1 = C + j * m;
            C_col2 = C + (j + 1) * m;
            C_col3 = C + (j + 2) * m;
            C_col4 = C + (j + 3) * m;
            c1a = _mm_load1_ps(A_col + j);
            c2a = _mm_load1_ps(A_col + j + 1);
            c3a = _mm_load1_ps(A_col + j + 2);
            c4a = _mm_load1_ps(A_col + j + 3);
            for (i = 0; i < unroll16; i += 16) {                
                r1a = _mm_loadu_ps(A_col + i);
                r1b = _mm_loadu_ps(A_col + i + 4);
                r1c = _mm_loadu_ps(A_col + i + 8);
                r1d = _mm_loadu_ps(A_col + i + 12);

                _mm_storeu_ps(C_col1 + i, _mm_add_ps(_mm_loadu_ps(C_col1 + i), _mm_mul_ps(r1a, c1a)));
                _mm_storeu_ps(C_col1 + i + 4, _mm_add_ps(_mm_loadu_ps(C_col1 + i + 4), _mm_mul_ps(r1b, c1a)));
                _mm_storeu_ps(C_col1 + i + 8, _mm_add_ps(_mm_loadu_ps(C_col1 + i + 8), _mm_mul_ps(r1c, c1a)));
                _mm_storeu_ps(C_col1 + i + 12, _mm_add_ps(_mm_loadu_ps(C_col1 + i + 12), _mm_mul_ps(r1d, c1a)));

                _mm_storeu_ps(C_col2 + i, _mm_add_ps(_mm_loadu_ps(C_col2 + i), _mm_mul_ps(r1a, c2a)));
                _mm_storeu_ps(C_col2 + i + 4, _mm_add_ps(_mm_loadu_ps(C_col2 + i + 4), _mm_mul_ps(r1b, c2a)));
                _mm_storeu_ps(C_col2 + i + 8, _mm_add_ps(_mm_loadu_ps(C_col2 + i + 8), _mm_mul_ps(r1c, c2a)));
                _mm_storeu_ps(C_col2 + i + 12, _mm_add_ps(_mm_loadu_ps(C_col2 + i + 12), _mm_mul_ps(r1d, c2a)));
    
                _mm_storeu_ps(C_col3 + i, _mm_add_ps(_mm_loadu_ps(C_col3 + i), _mm_mul_ps(r1a, c3a)));
                _mm_storeu_ps(C_col3 + i + 4, _mm_add_ps(_mm_loadu_ps(C_col3 + i + 4), _mm_mul_ps(r1b, c3a)));
                _mm_storeu_ps(C_col3 + i + 8, _mm_add_ps(_mm_loadu_ps(C_col3 + i + 8), _mm_mul_ps(r1c, c3a)));
                _mm_storeu_ps(C_col3 + i + 12, _mm_add_ps(_mm_loadu_ps(C_col3 + i + 12), _mm_mul_ps(r1d, c3a)));
    
                _mm_storeu_ps(C_col4 + i, _mm_add_ps(_mm_loadu_ps(C_col4 + i), _mm_mul_ps(r1a, c4a)));
                _mm_storeu_ps(C_col4 + i + 4, _mm_add_ps(_mm_loadu_ps(C_col4 + i + 4), _mm_mul_ps(r1b, c4a)));
                _mm_storeu_ps(C_col4 + i + 8, _mm_add_ps(_mm_loadu_ps(C_col4 + i + 8), _mm_mul_ps(r1c, c4a)));
                _mm_storeu_ps(C_col4 + i + 12, _mm_add_ps(_mm_loadu_ps(C_col4 + i + 12), _mm_mul_ps(r1d, c4a)));
            }
            for ( ; i < unroll4; i += 4) {
                _mm_storeu_ps(C_col1 + i, _mm_add_ps(_mm_loadu_ps(C_col1 + i), _mm_mul_ps(_mm_loadu_ps(A_col + i), c1a)));
                _mm_storeu_ps(C_col1 + i + 4, _mm_add_ps(_mm_loadu_ps(C_col1 + i + 4), _mm_mul_ps(_mm_loadu_ps(A_col + i + 4), c1a)));
                _mm_storeu_ps(C_col1 + i + 8, _mm_add_ps(_mm_loadu_ps(C_col1 + i + 8), _mm_mul_ps(_mm_loadu_ps(A_col + i + 8), c1a)));
                _mm_storeu_ps(C_col1 + i + 12, _mm_add_ps(_mm_loadu_ps(C_col1 + i + 12), _mm_mul_ps(_mm_loadu_ps(A_col + i + 12), c1a)));
            }
            for ( ; i < m; i += 1) {
                *(C_col1 + i) += *(A_col + j + 1) * *(A_col + i);
                *(C_col2 + i) += *(A_col + j + 2) * *(A_col + i);
                *(C_col3 + i) += *(A_col + j + 3) * *(A_col + i);
                *(C_col4 + i) += *(A_col + j + 4) * *(A_col + i);
            }
        }
        for ( ; j < m; j += 1) {
            C_col1 = C + j * m;
            c1a = _mm_load1_ps(A_col + j);
            for (i = 0; i < unroll16; i += 16) {
                _mm_storeu_ps(C_col1 + i, _mm_add_ps(_mm_loadu_ps(C_col1 + i), _mm_mul_ps(_mm_loadu_ps(A_col + i), c1a)));
                _mm_storeu_ps(C_col1 + i + 4, _mm_add_ps(_mm_loadu_ps(C_col1 + i + 4), _mm_mul_ps(_mm_loadu_ps(A_col + i + 4), c1a)));
                _mm_storeu_ps(C_col1 + i + 8, _mm_add_ps(_mm_loadu_ps(C_col1 + i + 8), _mm_mul_ps(_mm_loadu_ps(A_col + i + 8), c1a)));
                _mm_storeu_ps(C_col1 + i + 12, _mm_add_ps(_mm_loadu_ps(C_col1 + i + 12), _mm_mul_ps(_mm_loadu_ps(A_col + i + 12), c1a)));
            }
            for ( ; i < unroll4; i += 4) {
                _mm_storeu_ps(C_col1 + i, _mm_add_ps(_mm_loadu_ps(C_col1 + i), _mm_mul_ps(_mm_loadu_ps(A_col + i), c1a)));
            }
            for ( ; i < m; i += 1) {
                *(C+i + j*m) += *(A_col + i) * *(A_col + j);
            }
        }
        //begin handling the bottom edge-case
        #pragma omp for
        for (j = 0; j < jj; j += 4) {
            C_col1 = C + j * m;
            C_col2 = C + (j + 1) * m;
            C_col3 = C + (j + 2) * m;
            C_col4 = C + (j + 3) * m;
            c1a = _mm_load1_ps(A_col + j);
            c2a = _mm_load1_ps(A_col + j + 1);
            c3a = _mm_load1_ps(A_col + j + 2);
            c4a = _mm_load1_ps(A_col + j + 3);
            for (i = ii; i < unroll16; i += 16) {                
                r1a = _mm_loadu_ps(A_col + i);
                r1b = _mm_loadu_ps(A_col + i + 4);
                r1c = _mm_loadu_ps(A_col + i + 8);
                r1d = _mm_loadu_ps(A_col + i + 12);

                _mm_storeu_ps(C_col1 + i, _mm_add_ps(_mm_loadu_ps(C_col1 + i), _mm_mul_ps(r1a, c1a)));
                _mm_storeu_ps(C_col1 + i + 4, _mm_add_ps(_mm_loadu_ps(C_col1 + i + 4), _mm_mul_ps(r1b, c1a)));
                _mm_storeu_ps(C_col1 + i + 8, _mm_add_ps(_mm_loadu_ps(C_col1 + i + 8), _mm_mul_ps(r1c, c1a)));
                _mm_storeu_ps(C_col1 + i + 12, _mm_add_ps(_mm_loadu_ps(C_col1 + i + 12), _mm_mul_ps(r1d, c1a)));

                _mm_storeu_ps(C_col2 + i, _mm_add_ps(_mm_loadu_ps(C_col2 + i), _mm_mul_ps(r1a, c2a)));
                _mm_storeu_ps(C_col2 + i + 4, _mm_add_ps(_mm_loadu_ps(C_col2 + i + 4), _mm_mul_ps(r1b, c2a)));
                _mm_storeu_ps(C_col2 + i + 8, _mm_add_ps(_mm_loadu_ps(C_col2 + i + 8), _mm_mul_ps(r1c, c2a)));
                _mm_storeu_ps(C_col2 + i + 12, _mm_add_ps(_mm_loadu_ps(C_col2 + i + 12), _mm_mul_ps(r1d, c2a)));
    
                _mm_storeu_ps(C_col3 + i, _mm_add_ps(_mm_loadu_ps(C_col3 + i), _mm_mul_ps(r1a, c3a)));
                _mm_storeu_ps(C_col3 + i + 4, _mm_add_ps(_mm_loadu_ps(C_col3 + i + 4), _mm_mul_ps(r1b, c3a)));
                _mm_storeu_ps(C_col3 + i + 8, _mm_add_ps(_mm_loadu_ps(C_col3 + i + 8), _mm_mul_ps(r1c, c3a)));
                _mm_storeu_ps(C_col3 + i + 12, _mm_add_ps(_mm_loadu_ps(C_col3 + i + 12), _mm_mul_ps(r1d, c3a)));
    
                _mm_storeu_ps(C_col4 + i, _mm_add_ps(_mm_loadu_ps(C_col4 + i), _mm_mul_ps(r1a, c4a)));
                _mm_storeu_ps(C_col4 + i + 4, _mm_add_ps(_mm_loadu_ps(C_col4 + i + 4), _mm_mul_ps(r1b, c4a)));
                _mm_storeu_ps(C_col4 + i + 8, _mm_add_ps(_mm_loadu_ps(C_col4 + i + 8), _mm_mul_ps(r1c, c4a)));
                _mm_storeu_ps(C_col4 + i + 12, _mm_add_ps(_mm_loadu_ps(C_col4 + i + 12), _mm_mul_ps(r1d, c4a)));
            }
            for ( ; i < unroll4; i += 4) {
                _mm_storeu_ps(C_col1 + i, _mm_add_ps(_mm_loadu_ps(C_col1 + i), _mm_mul_ps(_mm_loadu_ps(A_col + i), c1a)));
                _mm_storeu_ps(C_col1 + i + 4, _mm_add_ps(_mm_loadu_ps(C_col1 + i + 4), _mm_mul_ps(_mm_loadu_ps(A_col + i + 4), c1a)));
                _mm_storeu_ps(C_col1 + i + 8, _mm_add_ps(_mm_loadu_ps(C_col1 + i + 8), _mm_mul_ps(_mm_loadu_ps(A_col + i + 8), c1a)));
                _mm_storeu_ps(C_col1 + i + 12, _mm_add_ps(_mm_loadu_ps(C_col1 + i + 12), _mm_mul_ps(_mm_loadu_ps(A_col + i + 12), c1a)));
            }
            for ( ; i < m; i += 1) {
                *(C_col1 + i) += *(A_col + j + 1) * *(A_col + i);
                *(C_col2 + i) += *(A_col + j + 2) * *(A_col + i);
                *(C_col3 + i) += *(A_col + j + 3) * *(A_col + i);
                *(C_col4 + i) += *(A_col + j + 4) * *(A_col + i);
            }
        }
    }
}
}
}