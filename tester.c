#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <float.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>
#include <cblas.h>

/* Your function must have the following signature: */

void sgemm( int m, int n, float *A, float *C );

/* The benchmarking program */

int main( int argc, char **argv )
{
	srand(time(NULL));
	
	if(argc != 3)
	{
		printf("Incorrect number of arguments!\n");
		printf("Usage: ./bench-test M N\n");
		exit(0);
	}	

	int m  = atoi(argv[1]);
	int n  = atoi(argv[2]);
	
	// Change these values to modify range checking
	if(m < 1000 || m > 10000 || n < 32 || n > 10000)
	{
		printf("Dimensions provided are out of the range of matrices we will be testing!  Change tester.c to modify range checking.\n");
		exit(0);
	}

	/* Allocate and fill 2 random matrices A, C */
	float *A = (float*) malloc( m * n * sizeof(float) );
	float *C = (float*) malloc( m * m * sizeof(float) );
	
	for( int i = 0; i < m*n; i++ ) A[i] = 2 * drand48() - 1;
	for( int i = 0; i < m*m; i++ ) C[i] = 2 * drand48() - 1;
	
	/* measure Gflop/s rate; time a sufficiently long sequence of calls to eliminate noise */
	double Gflop_s, seconds = -1.0;
	for( int n_iterations = 1; seconds < 0.1; n_iterations *= 2 ) 
	{
		/* warm-up */
		sgemm( m, n, A, C );
		  
		/* measure time */
		struct timeval start, end;
		gettimeofday( &start, NULL );
		for( int i = 0; i < n_iterations; i++ )
			sgemm( m,n, A, C );
		gettimeofday( &end, NULL );
		seconds = (end.tv_sec - start.tv_sec) + 1.0e-6 * (end.tv_usec - start.tv_usec);
		  
		/* compute Gflop/s rate */
		Gflop_s = 2e-9 * n_iterations * m * m * n / seconds;
	}
	
	printf( "%d by %d matrix \t %g Gflop/s\n", m, n, Gflop_s );
	
	/* Ensure that error does not exceed the theoretical error bound */
		
	/* Set initial C to 0 and do matrix multiply of A*B */
	memset( C, 0, sizeof( float ) * m * m );
	sgemm( m,n, A, C );
	
	/* Subtract A*B from C using standard sgemm (note that this should be 0 to within machine roundoff) */
	cblas_sgemm( CblasColMajor,CblasNoTrans,CblasTrans, m,m,n, -1, A,m, A,m, 1, C,m );
	
	/* Subtract the maximum allowed roundoff from each element of C */
	for( int i = 0; i < m*n; i++ ) A[i] = fabs( A[i] );
	for( int i = 0; i < m*m; i++ ) C[i] = fabs( C[i] );
	cblas_sgemm( CblasColMajor,CblasNoTrans,CblasTrans, m,m,n, -3.0*FLT_EPSILON*n, A,m, A,m, 1, C,m );
	
	/* After this test if any element in C is still positive something went wrong in square_sgemm */
	for( int i = 0; i < m * m; i++ )
	  	if( C[i] > 0 ) {
			printf( "FAILURE: error in matrix multiply exceeds an acceptable margin\n" );
		return -1;
	}
	
	/* release memory */
	free( C );
	free( A );
	return 0;
}
