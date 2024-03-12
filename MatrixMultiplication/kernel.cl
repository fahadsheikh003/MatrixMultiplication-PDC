__kernel void matrixMultiplication(__global int* A, __global int* B, __global int* C, int size) { 
    int i = get_global_id(0);
    int j = get_global_id(1);

    int sum = 0;
    for (int k = 0; k < size; k++)
	    sum += A[j * size + k] * B[k * size + i];

    //printf("Value of Sum: %d \n", sum);
    //printf("Value of A: %d \n", A[j * size + i]);
    //printf("Value of B: %d \n", B[j * size + i]);

    C[j * size + i] = sum;
}