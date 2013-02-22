#include <emmintrin.h>

void transpose(int m, int n, float *src, float *dst) {
    int i, j;
    for (i = 0; i < m; i += 1) {
        for (j = 0; j < n; j += 1) {
            dst[j + i * n] = src[i + j * m];
        }
    }
}

void sgemm( int m, int n, float *A, float *C ) {
    int i, j, k, jtn, cieling;
    float B[n * m];
    float buf[2];
    __m128d sum, ab, cd, ef, AB, CD, EF;
    transpose(m, n, A, B);
    for (i = 0; i < m; i += 1) {
        for (j = 0; j < m; j += 1) {
            jtn = j * n;
            for (k = 0, cieling = n - 5; k < cieling; k += 6) {
                ab = _mm_load1_pd(A + i + k * m);
                cd = _mm_load1_pd(A + i + (k + 2) * m);
                ef = _mm_load1_pd(A + i + (k + 4) * m);
                AB = _mm_loadu_pd(B + k + jtn);
                CD = _mm_loadu_pd(B + k + 2 + jtn);
                EF = _mm_loadu_pd(B + k + 4 + jtn);
                sum = _mm_add_pd(sum, _mm_mul_sd(ab, AB));
                sum = _mm_add_pd(sum, _mm_mul_sd(cd, CD));
                sum = _mm_add_pd(sum, _mm_mul_sd(ef, EF));
            }
            _mm_storeu_pd(buf, sum);
            C[i + j * m] = buf[0];
            if (n % 6 != 0) {
                for ( ; k < n; k += 1) {
                    C[i + j * m] += A[i + k * m] * A[k + jtn];
                }
            }
        }
    }
}
