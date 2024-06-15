//sample.cpp
#include "mpi.h"
#include <stdio.h>

int main(int argc, char *argv[])
{
    int N = 21;
    int array2d[N][N];

    int rank, size;
    MPI_Init(&argc, &argv);

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int begin;
    int end;
    /*各プロセスの担当数をなるべく均等に*/
    int num_per_proc = N/size;/* プロセス当たりの基本担当数 4プロセスなら10*/
    int extra_num = N % size;/* 足りないぶん 4プロセスなら１*/
    if(rank<extra_num){ /*余りがあるとき、最初のextra_numプロセスが追加で一個ずつ担当する*/
        begin = rank*(num_per_proc+1);
        end = begin + num_per_proc + 1;
    }
    else{
        begin = rank*num_per_proc + extra_num;
        end = begin + num_per_proc;
    }

    for(int i=begin;i<end;i++){
        for (int j=0;j<N;j++){
            array2d[i][j] = rank+1;
        }
    }


    int* recvcounts = (int*)malloc(size * sizeof(int));
    int* displs = (int*)malloc(size*sizeof(int));

    for (int i = 0; i < size; i++) {
        recvcounts[i] = (i < extra_num) ? (num_per_proc + 1) * N : num_per_proc * N;
        displs[i] = (i < extra_num) ? i * (num_per_proc + 1) * N : (i * num_per_proc + extra_num) * N;
    }


    MPI_Allgatherv(&array2d[begin],(end-begin)*N,MPI_INT,array2d,recvcounts,displs,MPI_INT,MPI_COMM_WORLD);
    free(displs);
    free(recvcounts);

    for(int proc=0;proc<size;proc++){
        MPI_Barrier(MPI_COMM_WORLD);
        if(rank==proc){
            printf("process %d printing:\n",rank);
            for(int i=0;i<N;i++){
                for(int j=0;j<N;j++){
                    printf("%d ",array2d[i][j]);
                }
                printf("\n");
            }
            printf("\n");
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }

    for (int j=begin;j<end;j++){
        array2d[j][N-1]=0;
    }

    recvcounts = (int*)malloc(size * sizeof(int));
    displs = (int*)malloc(size*sizeof(int));

    for (int i = 0; i < size; i++) {
        recvcounts[i] = (i < extra_num) ? (num_per_proc + 1) * N : num_per_proc * N;
        displs[i] = (i < extra_num) ? i * (num_per_proc + 1) * N : (i * num_per_proc + extra_num) * N;
    }


    MPI_Allgatherv(&array2d[begin],(end-begin)*N,MPI_INT,array2d,recvcounts,displs,MPI_INT,MPI_COMM_WORLD);
    free(displs);
    free(recvcounts);

    for(int proc=0;proc<size;proc++){
        MPI_Barrier(MPI_COMM_WORLD);
        if(rank==proc){
            printf("process %d printing:\n",rank);
            for(int i=0;i<N;i++){
                for(int j=0;j<N;j++){
                    printf("%d ",array2d[i][j]);
                }
                printf("\n");
            }
            printf("\n");
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }
    MPI_Finalize();
    return 0;
}
