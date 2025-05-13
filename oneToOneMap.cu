#include <stdio.h>
#include <stdlib.h>
#include <bits/stdc++.h>
#include <cuda_runtime.h>
#include <math.h>

// Use "module load mpi devtoolset/9 Cuda11.4" on huff

// salloc --partition GPU --qos gpu --nodes 1 --ntasks-per-node 1 --cpus-per-task 1 --mem 64G --exclude huff44 ./test 3 > output.txt

// nvcc -std=c++11 -o map fullyWork.cu (maple)

// nvcc -o map oneToOneMap.cu (local)
// ./map passLen > output.txt 
// all code needs to be ran inside of powershell not cmd

float kernel_runtime;
float kernel_memcpy_runtime;

__device__ unsigned char inputChars[] = "abcdefghijklmnopqrstuvwxyz""ABCDEFGHIJKLMNOPQRSTUVWXYZ""0123456789";

__device__ int alphabetSize = 62;

__device__ unsigned long long totThr = 0;

// Input Length Combinations (62^length)
// This is without special characters " !"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"
// 1: 62
// 2: 3,844
// 3: 238,328
// 4: 14,776,336
// 5: 916,132,832
// 6: 56,800,235,584
// 7: 3,521,614,606,208

//-------------------------------------------------------------------------------------------------------------------------------------------------------------------

// Works on password length 3
// 3: 238,328

// __global__ void kernel_MD2_brute(int passLen) {

//     // 2D
//     int tid = blockDim.x * blockIdx.x + threadIdx.x;

//     int totalComb = pow((double) alphabetSize, (double) passLen);

//     // 3D
//     // int tid = threadIdx.x + threadIdx.y + threadIdx.z * blockDim.x * blockDim.y;

//     atomicAdd(&totThr, 1);

//     // printf("TID: %d\n", tid);

//     if (tid < totalComb) {

//         unsigned char output[3];

//         if (tid < (totalComb/alphabetSize)) {

//             int firstIndex = tid/(totalComb/alphabetSize);
//             int subCombo = tid;
//             int secondIndex = (subCombo/alphabetSize);
//             int thirdIndex = tid % alphabetSize;

//             output[0] = inputChars[firstIndex];
//             output[1] = inputChars[secondIndex];
//             output[2] = inputChars[thirdIndex];

//             // printf("Here: %s\n", output);

//         }
//         else {

//             int firstIndex = tid/(totalComb/alphabetSize);
//             int subCombo = tid - ((totalComb/alphabetSize) * (firstIndex));
//             int secondIndex = (subCombo/alphabetSize);
//             int thirdIndex = tid % alphabetSize;

//             output[0] = inputChars[firstIndex];
//             output[1] = inputChars[secondIndex];
//             output[2] = inputChars[thirdIndex];

//             // printf("Tid: %d Middle Index: %d First Index: %d\n", tid, secondIndex, firstIndex);
//             // if (output[0] == '9'){
//             //     printf("Here: %s\n", output);
//             // }
//             // printf("Here: %s\n", output);

//         }
//         // free(output);
//     }
// }

//-------------------------------------------------------------------------------------------------------------------------------------------------------------------

// Works on password length 4
// 4: 14,776,336

// __global__ void kernel_MD2_brute(int passLen) {

//     // 2D
//     int tid = blockDim.x * blockIdx.x + threadIdx.x;

//     int totalComb = pow((double) alphabetSize, (double) passLen);

//     // 3D
//     // int tid = threadIdx.x + threadIdx.y + threadIdx.z * blockDim.x * blockDim.y;
    
//     atomicAdd(&totThr, 1);

//     // printf("TID: %d\n", tid);

//     if (tid < totalComb) {

//         // unsigned char* output = (unsigned char*) malloc(passLen * sizeof(unsigned char));

//         unsigned char output[4];

//         // cudaMalloc((void**)&output, passLen*sizeof(unsigned char));

//         if (tid < (totalComb/alphabetSize)) {

//             int firstIndex = tid/(totalComb/alphabetSize);
//             int subCombo = tid;
//             // int subCombo = tid - ((totalComb/alphabetSize) * (firstIndex));
//             // int middleIndex = (tid/(tid/(totalComb/alphabetSize)))/alphabetSize;
//             // int middleIndex = (subCombo/alphabetSize);
//             int secondIndex = (subCombo/(alphabetSize*alphabetSize));
//             int thirdIndex = (tid/alphabetSize) % alphabetSize;
//             int lastIndex = tid % alphabetSize;

//             output[0] = inputChars[firstIndex];
//             output[1] = inputChars[secondIndex];
//             output[2] = inputChars[thirdIndex];
//             output[3] = inputChars[lastIndex];

//             // printf("Here: %s\n", output);

//             // if (output[1] == '9'){
//             //     printf("Here: %s\n", output);
//             // }
//             // free(output);
//         }
//         else {

//             int firstIndex = tid/(totalComb/alphabetSize);
//             int subCombo = tid - ((totalComb/alphabetSize) * (firstIndex));
//             // int middleIndex = (tid/(tid/(totalComb/alphabetSize)))/alphabetSize;
//             // int middleIndex = (subCombo/alphabetSize);
//             int secondIndex = (subCombo/(alphabetSize*alphabetSize));
//             int thirdIndex = (subCombo/alphabetSize) % alphabetSize;
//             int lastIndex = tid % alphabetSize;

//             output[0] = inputChars[firstIndex];
//             output[1] = inputChars[secondIndex];
//             output[2] = inputChars[thirdIndex];
//             output[3] = inputChars[lastIndex];

//             // printf("Tid: %d Middle Index: %d First Index: %d\n", tid, middleIndex, firstIndex);
//             // if (output[0] == 'b' && output[1] == '9'){
//             //     printf("Here: %s\n", output);
//             // }
//             // free(output);
//         }
//         // free(output);
//     }
// }

//-------------------------------------------------------------------------------------------------------------------------------------------------------------------

// Works on password length 5
// 5: 916,132,832

// __global__ void kernel_MD2_brute(int passLen) {

//     // 2D
//     int tid = blockDim.x * blockIdx.x + threadIdx.x;

//     int totalComb = pow((double) alphabetSize, (double) passLen);

//     // 3D
//     // int tid = threadIdx.x + threadIdx.y + threadIdx.z * blockDim.x * blockDim.y;
    
//     atomicAdd(&totThr, 1);

//     // printf("TID: %d\n", tid);

//     if (tid < totalComb) {

//         // unsigned char* output = (unsigned char*) malloc(passLen * sizeof(unsigned char));

//         unsigned char output[5];

//         if (tid < (totalComb/alphabetSize)) {

//             int firstIndex = tid/(totalComb/alphabetSize);
//             int secondIndex = (tid/(alphabetSize*alphabetSize*alphabetSize));
//             int thirdIndex = (tid/(alphabetSize*alphabetSize)) % alphabetSize;
//             int fourthIndex = (tid/(alphabetSize)) % (alphabetSize);
//             int lastIndex = tid % alphabetSize;

//             output[0] = inputChars[firstIndex];
//             output[1] = inputChars[secondIndex];
//             output[2] = inputChars[thirdIndex];
//             output[3] = inputChars[fourthIndex];
//             output[4] = inputChars[lastIndex];


//             // printf("Here: %s\n", output);

//             // if (output[0] == 'a' && output[1] == 'b' && output[2] == 'c'){
//             //     printf("Here: %s\n", output);
//             // }

//         }
//         else {

//             int firstIndex = tid/(totalComb/alphabetSize);
//             int subCombo = tid - ((totalComb/alphabetSize) * (firstIndex));
//             int secondIndex = (subCombo/(alphabetSize*alphabetSize*alphabetSize));
//             int thirdIndex = (subCombo/(alphabetSize*alphabetSize)) % alphabetSize;
//             int fourthIndex = (subCombo/(alphabetSize)) % (alphabetSize);
//             int lastIndex = tid % alphabetSize;

//             output[0] = inputChars[firstIndex];
//             output[1] = inputChars[secondIndex];
//             output[2] = inputChars[thirdIndex];
//             output[3] = inputChars[fourthIndex];
//             output[4] = inputChars[lastIndex];

//             // printf("Tid: %d Middle Index: %d First Index: %d\n", tid, middleIndex, firstIndex);
//             // if (output[0] == '9' && output[1] == '9'){
//             //     printf("Here: %s\n", output);
//             // }

//             // if (output[0] == '9' && output[1] == '9' && output[2] == '9'){
//             //     printf("Here: %s\n", output);
//             // }
//         }
//         // free(output);
//     }
// }

//-------------------------------------------------------------------------------------------------------------------------------------------------------------------

// Works on password length 6
// 6: 56,800,235,584

__global__ void kernel_MD2_brute(int passLen) {

    // 2D
    // int tid = blockDim.x * blockIdx.x + threadIdx.x;

    // int totalComb = pow((double) alphabetSize, (double) passLen);

    // 3D
    
    unsigned long long int blockId = blockIdx.x + blockIdx.y * gridDim.x + gridDim.x * gridDim.y * blockIdx.z;
    
    unsigned long long int tid = blockId * (blockDim.x * blockDim.y * blockDim.z) + (threadIdx.z * (blockDim.x * blockDim.y)) + (threadIdx.y * blockDim.x) + threadIdx.x;

    unsigned long long int totalComb = pow((double) alphabetSize, (double) passLen);

    atomicAdd(&totThr, 1);

    // printf("Total Combinations: %llu\n", totalComb);

    if (tid < totalComb) {

        // atomicAdd(&totThr, 1);

        // printf("TID: %llu\n", tid);

        // unsigned char* output = (unsigned char*) malloc(passLen * sizeof(unsigned char));

        unsigned char output[6];

        if (tid < (totalComb/alphabetSize)) {

            int firstIndex = tid/(totalComb/alphabetSize);
            int secondIndex = (tid/(alphabetSize*alphabetSize*alphabetSize*alphabetSize));
            int thirdIndex = (tid/(alphabetSize*alphabetSize*alphabetSize)) % alphabetSize;
            int fourthIndex = (tid/(alphabetSize*alphabetSize)) % (alphabetSize);
            int fifthIndex = (tid/(alphabetSize)) % (alphabetSize);
            int lastIndex = tid % alphabetSize;

            output[0] = inputChars[firstIndex];
            output[1] = inputChars[secondIndex];
            output[2] = inputChars[thirdIndex];
            output[3] = inputChars[fourthIndex];
            output[4] = inputChars[fifthIndex];
            output[5] = inputChars[lastIndex];

            // printf("Here: %s\n", output);

            // if (output[0] == 'a' && output[1] == 'a' && output[2] == 'c'){
            //     printf("Here: %s\n", output);
            // }

        }
        else {

            int firstIndex = tid/(totalComb/alphabetSize);
            int subCombo = tid - ((totalComb/alphabetSize) * (firstIndex));
            int secondIndex = (subCombo/(alphabetSize*alphabetSize*alphabetSize*alphabetSize));
            int thirdIndex = (subCombo/(alphabetSize*alphabetSize*alphabetSize)) % alphabetSize;
            int fourthIndex = (subCombo/(alphabetSize*alphabetSize)) % (alphabetSize);
            int fifthIndex = (tid/(alphabetSize)) % (alphabetSize);
            int lastIndex = tid % alphabetSize;

            output[0] = inputChars[firstIndex];
            output[1] = inputChars[secondIndex];
            output[2] = inputChars[thirdIndex];
            output[3] = inputChars[fourthIndex];
            output[4] = inputChars[fifthIndex];
            output[5] = inputChars[lastIndex];

            // printf("Here: %s\n", output);

            // printf("Tid: %d Middle Index: %d First Index: %d\n", tid, middleIndex, firstIndex);
            // if (output[0] == '9' && output[1] == '9'){
            //     printf("Here: %s\n", output);
            // }

            // if (output[0] == 'T' && output[1] == 'u' && output[2] == 'c' && output[3] == 'k' && output[4] == 'e' && output[5] == 'r'){
            //     printf("Here: %s\n", output);
            // }

            // if (output[0] == '9' && output[1] == '9' && output[2] == '9' && output[3] == '9'){
            //     printf("Here: %s\n", output);
            // }
        }
        // free(output);
    }
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------------

// this will be for length 7 (currently working on the kernel call of creating the threads for this password length)
// 7: 3,521,614,606,208

// __global__ void kernel_MD2_brute(int passLen) {

//     // 2D
//     // int tid = blockDim.x * blockIdx.x + threadIdx.x;

//     // 3D
//     // unsigned long long int blockId = blockIdx.x + blockIdx.y * gridDim.x + gridDim.x * gridDim.y * blockIdx.z; 

//     // unsigned long long int tid = blockId * blockDim.x + threadIdx.x;

//     // unsigned long long int totalComb = pow((double) alphabetSize, (double) passLen);

//     unsigned long long int blockId = blockIdx.x + blockIdx.y * gridDim.x + gridDim.x * gridDim.y * blockIdx.z;
    
//     unsigned long long int tid = blockId * (blockDim.x * blockDim.y * blockDim.z) + (threadIdx.z * (blockDim.x * blockDim.y)) + (threadIdx.y * blockDim.x) + threadIdx.x;

//     unsigned long long int totalComb = pow((double) alphabetSize, (double) passLen);

//     // atomicAdd(&totThr, 1);

//     // printf("Total Combinations: %llu\n", totalComb);

//     // unsigned long long int test = 56800235580;
//     // 3,521,614,606,208
//     // 3,521,614,729,216

//     // length 6
//     // Grid : {57721, 62, 62} blocks. Blocks : {256, 1, 1} threads.

//     // length 7 
//     // Grid : {3578644, 62, 62} blocks. Blocks : {256, 1, 1} threads

//     if (tid < totalComb) {
//         // unsigned char* output = (unsigned char*) malloc(passLen * sizeof(unsigned char));

//         // unsigned char output[7];

//         // atomicAdd(&totThr, 1);

//         // printf("TID: %llu\n", tid);

//         if (tid < (totalComb/alphabetSize)) {

//             atomicAdd(&totThr, 1);

//             // int firstIndex = tid/(totalComb/alphabetSize);
//             // int secondIndex = (tid/(alphabetSize*alphabetSize*alphabetSize*alphabetSize*alphabetSize));
//             // int thirdIndex = (tid/(alphabetSize*alphabetSize*alphabetSize*alphabetSize)) % alphabetSize;
//             // int fourthIndex = (tid/(alphabetSize*alphabetSize*alphabetSize)) % (alphabetSize);
//             // int fifthIndex = (tid/(alphabetSize*alphabetSize)) % (alphabetSize);
//             // int sixthIndex = (tid/(alphabetSize)) % (alphabetSize);
//             // int lastIndex = tid % alphabetSize;

//             // output[0] = inputChars[firstIndex];
//             // output[1] = inputChars[secondIndex];
//             // output[2] = inputChars[thirdIndex];
//             // output[3] = inputChars[fourthIndex];
//             // output[4] = inputChars[fifthIndex];
//             // output[5] = inputChars[sixthIndex];
//             // output[6] = inputChars[lastIndex];

//             // printf("Here: %s\n", output);

//             // if (output[0] == 'a' && output[1] == 'a' && output[2] == 'a' && output[3] == 'a' && output[4] == 'c'){
//             //     printf("Here: %s\n", output);
//             // }

//         }
//         else {

//             // int firstIndex = tid/(totalComb/alphabetSize);
//             // int subCombo = tid - ((totalComb/alphabetSize) * (firstIndex));
//             // int secondIndex = (subCombo/(alphabetSize*alphabetSize*alphabetSize*alphabetSize*alphabetSize));
//             // int thirdIndex = (subCombo/(alphabetSize*alphabetSize*alphabetSize*alphabetSize)) % alphabetSize;
//             // int fourthIndex = (subCombo/(alphabetSize*alphabetSize*alphabetSize)) % (alphabetSize);
//             // int fifthIndex = (tid/(alphabetSize*alphabetSize)) % (alphabetSize);
//             // int sixthIndex = (tid/(alphabetSize)) % (alphabetSize);
//             // int lastIndex = tid % alphabetSize;

//             // output[0] = inputChars[firstIndex];
//             // output[1] = inputChars[secondIndex];
//             // output[2] = inputChars[thirdIndex];
//             // output[3] = inputChars[fourthIndex];
//             // output[4] = inputChars[fifthIndex];
//             // output[5] = inputChars[sixthIndex];
//             // output[6] = inputChars[lastIndex];

//             // printf("Tid: %d Middle Index: %d First Index: %d\n", tid, middleIndex, firstIndex);
//             // if (output[0] == '9' && output[1] == '9'){
//             //     printf("Here: %s\n", output);
//             // }

//             // if (output[0] == '9' && output[1] == '9' && output[2] == '9' && output[3] == '9' && output[4] == '9' && output[5] == '9'){
//             //     printf("Here: %s\n", output);
//             // }
//         }
//         // free(output);
//     }
// }

//-------------------------------------------------------------------------------------------------------------------------------------------------------------------

void BFS(int passLen) {   

    // Declare timers
    cudaEvent_t startEvent, stopEvent;
    cudaEvent_t startEventKernel, stopEventKernel;
    cudaEventCreate(&startEvent);           cudaEventCreate(&stopEvent);
    cudaEventCreate(&startEventKernel);     cudaEventCreate(&stopEventKernel);

    // Start timer measuring kernel + memory copy times
    cudaEventRecord(startEvent, 0);
    
    // Start timer measuring kernel time only
    cudaEventRecord(startEventKernel, 0);

    int alphabetSize = 62;

//-------------------------------------------------------------------------------------------------------------------------------------------------------------------
    // this needs to be the total combination of passwords (total comb * 62)

    // this is for length 6 or less
    int totalComb = pow((double) alphabetSize, (double) passLen - 1);

    // what is needed for length 6 and less
    kernel_MD2_brute<<<totalComb,alphabetSize>>>(passLen);

//-------------------------------------------------------------------------------------------------------------------------------------------------------------------

    // Maximum grid size is: (2147483647	65535	65535	)
    // Maximum block dim is: (1024	1024	64	)

    // length 7 and up math
    //testing a new totalComb

    // unsigned long long int totalComb = ((pow((double) alphabetSize, (double) passLen - 2)) / 256) + 1;
    // int gridYZ = 62;

    // // (2,147,483,647	65,535	65,535	)
    // // (1,024	1,024	64	)

    // // 3,521,614,606,208
    // // 3,521,614,729,216

    // dim3 gridDim(totalComb, gridYZ, gridYZ);
    // dim3 blockDim(256);

    // printf("Grid : {%d, %d, %d} blocks. Blocks : {%d, %d, %d} threads.\n", gridDim.x, gridDim.y, gridDim.z, blockDim.x, blockDim.y, blockDim.z);

    // // currently testing for length 7
    // kernel_MD2_brute<<<gridDim, blockDim>>>(passLen);

//-------------------------------------------------------------------------------------------------------------------------------------------------------------------

    // Wait for kernel completion
    cudaDeviceSynchronize();

    int devNo = 0;
    cudaDeviceProp iProp;
    cudaGetDeviceProperties(&iProp, devNo);
    printf("Maximum grid size is: (");
    for (int i = 0; i < 3; i++)
        printf("%d\t", iProp.maxGridSize[i]);
    printf(")\n");
    printf("Maximum block dim is: (");
    for (int i = 0; i < 3; i++)
        printf("%d\t", iProp.maxThreadsDim[i]);
    printf(")\n");
    printf("Max threads per block: %d\n", iProp.maxThreadsPerBlock);

    cudaError_t err = cudaGetLastError();

    if ( err != cudaSuccess ) {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));

        // Possibly: exit(-1) if program cannot continue....
    }

    unsigned long long total;
    cudaMemcpyFromSymbol(&total, totThr, sizeof(unsigned long long));
    printf("Total threads counted: %llu\n", total);

    // Stop timer measuring kernel time only
    cudaEventRecord(stopEventKernel, 0);

    // Stop timer measuring kernel + memory copy times
    cudaEventRecord(stopEvent, 0);
    
    // Calculate elapsed time
    cudaEventSynchronize(stopEvent);
    cudaEventSynchronize(stopEventKernel);
    cudaEventElapsedTime(&kernel_memcpy_runtime, startEvent, stopEvent);
    cudaEventElapsedTime(&kernel_runtime, startEventKernel, stopEventKernel);

    // cudaFree(output);
}

int main(int argc, char *argv[]) {
    if(argc != 2)
    {
        printf("Usage: ./program passLen\n");
        exit(0);
    }

    int passLen = strtol(argv[1], NULL, 10);
    
    // Initialize time measurement
    float time_difference;
    cudaEvent_t startEvent, stopEvent;
    cudaEventCreate(&startEvent);
    cudaEventCreate(&stopEvent);
    cudaEventRecord(startEvent, 0);
    
    BFS(passLen);

    // Stop time measurement
    cudaEventRecord(stopEvent, 0);
    cudaEventSynchronize(stopEvent);
    cudaEventElapsedTime(&time_difference, startEvent, stopEvent);
    
    printf("Single GPU: Required %f ms total time. Kernel and memcpy required %f ms. Kernel only required %f ms.\n", time_difference, kernel_memcpy_runtime, kernel_runtime);
}