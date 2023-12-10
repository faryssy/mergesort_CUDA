#include <stdio.h>
#include <stdlib.h>
#include <math.h>

typedef struct
{
int x;
int y;
} Point;

// Function that catches the error 
void testCUDA(cudaError_t error, const char *file, int line)  {

	if (error != cudaSuccess) {
	   printf("There is an error in file %s at line %d\n", file, line);
       exit(EXIT_FAILURE);
	} 
}

// Has to be defined in the compilation in order to get the correct value 
// of the macros __FILE__ and __LINE__

#define testCUDA(error) (testCUDA(error, __FILE__ , __LINE__))

__global__ void mergeSmall_k(int *A, int *B, int *C, int n)
{   
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int offset = 0;
    Point Q = {0,0};
    Point K = {0,0};
    Point P = {0,0};

    if (idx > sizeof(A))
    {
       K = {idx - (int)sizeof(A), (int)sizeof(A)};
       P = {(int)sizeof(A), idx - (int)sizeof(A)};
    }
    else
    {
        K = {0, idx};
        P = {idx, 0};
    }

    while (true) 
    {
        offset = abs(K.y - P.y)/2;
        Q.x = K.x + offset;
        Q.y = K.y - offset;
        
        if (Q.y >= 0 && Q.x <= sizeof(B) && 
            (Q.y == sizeof(A) || Q.x == 0 || A[Q.y] > B[Q.x - 1]))
            {
                if (Q.x == sizeof(B) || Q.y == 0 || A[Q.y - 1] <= B[Q.x])
                {
                    if (Q.y < sizeof(A) && (Q.x == sizeof(B) || A[Q.y] <= B[Q.x]))
                    {
                        C[idx] = A[Q.y];
                    }

                    else
                    {
                        C[idx] = B[Q.x];
                    }
                    break;
                }
                else
                {
                    K.x = Q.x + 1;
                    K.y = Q.y - 1;
                }
            }
        else 
        {
            P.x = Q.x - 1;
            P.y = Q.y + 1;
        }
    }   
    __syncthreads();
}

int main(void)
{
   // GPU timer instructions
    float TimeExec;									
	cudaEvent_t start, stop;						
	testCUDA(cudaEventCreate(&start));
    testCUDA(cudaEventCreate(&stop));	

    // size of A and B
    const int n = 8;

    // host arrays
    int *A, *B, *C;

    // device arrays
    int *d_A, *d_B, *d_C;

    // Allocate memory for host arrays
    A = (int*)malloc(n*sizeof(int));
    B = (int*)malloc(n*sizeof(int));
    C = (int*)malloc(2*n*sizeof(int));

    // Allocate memory for device arrays
    cudaMalloc(&d_A, n*sizeof(int));
    cudaMalloc(&d_B, n*sizeof(int));
    cudaMalloc(&d_C, 2*n*sizeof(int));

    // Initialize arrays on host
    for (int i = 0; i < n*2+1; i++)
    {
      if (i%2)
      {
        A[i/2] = i;
      }
      else
      {
        B[(i-1)/2] = i;
      }
    }
    
    testCUDA(cudaEventRecord(start, 0));

    // Copy host vectors to device
    cudaMemcpy( d_A, A, n*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy( d_B, B, n*sizeof(int), cudaMemcpyHostToDevice);
    // Executing the kernel function
    mergeSmall_k<<<1, 2*n>>>(d_A, d_B, d_C, n);

    // Copy array back to host
    cudaMemcpy( C, d_C, 2*n*sizeof(int), cudaMemcpyDeviceToHost);
    testCUDA(cudaEventRecord(stop,0));
    testCUDA(cudaEventSynchronize(stop));			
    testCUDA(cudaEventElapsedTime(&TimeExec, start, stop));							
    testCUDA(cudaEventDestroy(start));				
    testCUDA(cudaEventDestroy(stop));

    // print C = mergeSmall_k(A,B)
    printf("Mergesort of A and B \n");
    printf("A : ");
    for(int i = 0; i < n; i++)
        printf("%d ", A[i]);
    printf("\nB : ");
    for(int i = 0; i < n; i++)
        printf("%d ", B[i]);
    
    printf("\n");
    for(int i = 0; i < 2*n; i++)
        printf("%d ", C[i]);
    
    printf("\nGPU time execution for Merge Sort: %f ms\n", TimeExec);

    // Release device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    // Release host memory
    free(A);
    free(B);
    free(C);
 
    return 0;
}
