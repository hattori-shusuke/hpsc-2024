#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cuda_runtime.h>

__global__ void Bucket_init(int *bucket, int range){
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx<range){
    bucket[idx] = 0;
  }
}

__global__ void Bucket_increment(int *bucket,int *key,int n){
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx<n){
    atomicAdd(&bucket[key[idx]],1);
  }
}

__global__ void Sort(int *bucket,int *key,int range){
  int index = 0;
  for(int i=0;i<range;i++){
    for(int count=bucket[i];count>0;count--) {
      key[index++] = i;
    }
  }
}

int main() {
  int n = 50;
  int range = 5;
  int *bucket,*key;

  cudaMallocManaged(&key,n*sizeof(int));
  cudaMallocManaged(&bucket,range*sizeof(int));

  for (int i=0;i<n;i++){
    key[i] = rand() % range;
    printf("%d ", key[i]);
  }
  printf("\n");

  Bucket_init<<<(range + 255) / 256, 256>>>(bucket,range);
  cudaDeviceSynchronize();
  Bucket_increment<<<(n + 255) / 256, 256>>>(bucket, key, n);
  cudaDeviceSynchronize();
  Sort<<<1, 1>>>(bucket, key, range);
  cudaDeviceSynchronize();

  for (int i=0; i<n; i++) {
    printf("%d ",key[i]);
  }
  printf("\n");
  
  cudaFree(bucket);
  cudaFree(key);
}