#include <cstdlib>
#include <cstdio>
#include <fstream>
#include <vector>
#include <time.h>
#include "mpi.h"
using namespace std;
typedef vector<vector<float>> matrix;


int main(int argc,char** argv) {
  clock_t start_time = clock();
  int nx = 41;
  int ny = 41;
  int nt = 500;
  int nit = 50;
  double dx = 2. / (nx - 1);
  double dy = 2. / (ny - 1);
  double dt = .01;
  double rho = 1.;
  double nu = .02;
/*
  matrix u(ny,vector<float>(nx));
  matrix v(ny,vector<float>(nx));
  matrix p(ny,vector<float>(nx));
  matrix b(ny,vector<float>(nx));
  matrix un(ny,vector<float>(nx));
  matrix vn(ny,vector<float>(nx));
  matrix pn(ny,vector<float>(nx));
  */
  float u[nx][ny];
  float v[nx][ny];
  float p[nx][ny];
  float b[nx][ny];
  float un[nx][ny];
  float vn[nx][ny];
  float pn[nx][ny];

  MPI_Init(&argc,&argv);
  int size,rank;
  MPI_Comm_size(MPI_COMM_WORLD,&size);
  MPI_Comm_rank(MPI_COMM_WORLD,&rank);

  int begin,end;

  /*各プロセスの担当数をなるべく均等に*/
  int num_per_proc = nx/size;/* プロセス当たりの基本担当数 4プロセスなら10*/
  int extra_num = nx % size;/* 足りないぶん 4プロセスなら１*/
  if(rank<extra_num){ /*余りがあるとき、最初のextra_numプロセスが追加で一個ずつ担当する*/
    begin = rank*(num_per_proc+1);
    end = begin + num_per_proc + 1;
  }
  else{
    begin = rank*num_per_proc + extra_num;
    end = begin + num_per_proc;
  }

  for (int j=begin;j<end;j++){
    for (int i=0;i<nx;i++){
      u[j][i]=0;
      v[j][i]=0;
      p[j][i]=0;
      b[j][i]=0;
    }
  }

    int* recvcounts = (int*)malloc(size*sizeof(int));
    int* displs = (int*)malloc(size*sizeof(int));
      for (int i = 0; i < size; i++) {
        recvcounts[i] = (i < extra_num) ? (num_per_proc + 1) * nx : num_per_proc * nx;
        displs[i] = (i < extra_num) ? i * (num_per_proc + 1) * nx : (i * num_per_proc + extra_num) * nx;
      }

  MPI_Allgatherv(&u[begin],(end-begin)*nx,MPI_FLOAT,u,recvcounts,displs,MPI_FLOAT,MPI_COMM_WORLD);
  MPI_Allgatherv(&v[begin],(end-begin)*nx,MPI_FLOAT,v,recvcounts,displs,MPI_FLOAT,MPI_COMM_WORLD);
  MPI_Allgatherv(&p[begin],(end-begin)*nx,MPI_FLOAT,p,recvcounts,displs,MPI_FLOAT,MPI_COMM_WORLD);
  MPI_Allgatherv(&b[begin],(end-begin)*nx,MPI_FLOAT,b,recvcounts,displs,MPI_FLOAT,MPI_COMM_WORLD);
  free(displs);
  free(recvcounts);
  

  ofstream ufile("u.dat");
  ofstream vfile("v.dat");
  ofstream pfile("p.dat");

  for (int n=0; n<nt; n++) {

     begin = rank*((nx-1)/size);
     end = (rank+1)*((nx-1)/size);

    for (int j=begin; j<end; j++) {
      if(j==0) continue;
      for (int i=1; i<nx-1; i++) {
        // Compute b[j][i]
        b[j][i] = rho * (1 / dt *
                ((u[j][i+1] - u[j][i-1]) / (2. * dx) + (v[j+1][i] - v[j-1][i]) / (2. * dy)) -
                ((u[j][i+1] - u[j][i-1]) / (2. * dx)) * ((u[j][i+1] - u[j][i-1]) / (2. * dx)) -
                2. * ((u[j+1][i] - u[j-1][i]) / (2. * dy) *
                     (v[j][i+1] - v[j][i-1]) / (2. * dx)) -
                ((v[j+1][i] - v[j-1][i]) / (2. * dy)) * ((v[j+1][i] - v[j-1][i]) / (2. * dy)));
      }
    }
    MPI_Allgather(&b[begin],(end-begin)*nx,MPI_FLOAT,b,(end-begin)*nx,MPI_FLOAT,MPI_COMM_WORLD);

    
    /*各プロセスの担当数をなるべく均等に*/
    int num_per_proc = nx/size;/* プロセス当たりの基本担当数 4プロセスなら10*/
    int extra_num = nx % size;/* 足りないぶん 4プロセスなら１*/
    if(rank<extra_num){ /*余りがあるとき、最初のextra_numプロセスが追加で一個ずつ担当する*/
      begin = rank*(num_per_proc+1);
      end = begin + num_per_proc + 1;
    }
    else{
      begin = rank*num_per_proc + extra_num;
      end = begin + num_per_proc;
    }
    for (int it=0; it<nit; it++) {
      for (int j=begin; j<end; j++){
        for (int i=0; i<nx; i++){
	        pn[j][i] = p[j][i];
        }
      }

      int *recvcounts = (int*)malloc(size*sizeof(int));
      int *displs = (int*)malloc(size*sizeof(int));
      for (int i = 0; i < size; i++) {
        recvcounts[i] = (i < extra_num) ? (num_per_proc + 1) * nx : num_per_proc * nx;
        displs[i] = (i < extra_num) ? i * (num_per_proc + 1) * nx : (i * num_per_proc + extra_num) * nx;
      }


      MPI_Allgatherv(&pn[begin],(end-begin)*nx,MPI_FLOAT,pn,recvcounts,displs,MPI_FLOAT,MPI_COMM_WORLD);
      free(displs);
      free(recvcounts);

       begin = rank*((nx-1)/size);
       end = (rank+1)*((nx-1)/size);

      for (int j=begin; j<end; j++) {
        if(j==0) continue;
        for (int i=1; i<nx-1; i++) {
	        // Compute p[j][i]
          p[j][i] = (dy*dy * (pn[j][i+1] + pn[j][i-1]) + dx*dx * (pn[j+1][i] + pn[j-1][i]) -
           b[j][i] * dx*dx * dy*dy) / (2. * (dx*dx + dy*dy));
	      }
      }
      MPI_Allgather(&p[begin],(end-begin)*nx,MPI_FLOAT,p,(end-begin)*nx,MPI_FLOAT,MPI_COMM_WORLD);

      
      /*各プロセスの担当数をなるべく均等に*/
      int num_per_proc = nx/size;/* プロセス当たりの基本担当数 4プロセスなら10*/
      int extra_num = nx % size;/* 足りないぶん 4プロセスなら１*/
      if(rank<extra_num){ /*余りがあるとき、最初のextra_numプロセスが追加で一個ずつ担当する*/
        begin = rank*(num_per_proc+1);
        end = begin + num_per_proc + 1;
      }
      else{
        begin = rank*num_per_proc + extra_num;
        end = begin + num_per_proc;
      }

      for (int j=begin; j<end; j++) {
        // Compute p[j][0] and p[j][nx-1]
        p[j][0]=p[j][1];
        p[j][nx-1]=p[j][nx-2];
      }

      recvcounts = (int*)malloc(size*sizeof(int));
      displs = (int*)malloc(size*sizeof(int));
      for (int i = 0; i < size; i++) {
        recvcounts[i] = (i < extra_num) ? (num_per_proc + 1) * nx : num_per_proc * nx;
        displs[i] = (i < extra_num) ? i * (num_per_proc + 1) * nx : (i * num_per_proc + extra_num) * nx;
      }

      MPI_Allgatherv(&p[begin],(end-begin)*nx,MPI_FLOAT,p,recvcounts,displs,MPI_FLOAT,MPI_COMM_WORLD);
      free(recvcounts);
      free(displs);

      for (int i=0; i<nx; i++) {
	      // Compute p[0][i] and p[ny-1][i]
        p[0][i]=p[1][i];
        p[ny-1][i]=0;
      }

    }

    for (int j=0; j<ny; j++) {
      for (int i=0; i<nx; i++) {
        un[j][i] = u[j][i];
	      vn[j][i] = v[j][i];
      }
    }
    for (int j=1; j<ny-1; j++) {
      for (int i=1; i<nx-1; i++) {
	// Compute u[j][i] and v[j][i]
    u[j][i] = (un[j][i] - un[j][i] * dt / dx * (un[j][i] - un[j][i - 1]) -
           un[j][i] * dt / dy * (un[j][i] - un[j - 1][i]) -
           dt / (2 * rho * dx) * (p[j][i+1] - p[j][i-1]) +
           nu * dt / (dx*dx) * (un[j][i+1] - 2 * un[j][i] + un[j][i-1]) +
           nu * dt / (dy*dy) * (un[j+1][i] - 2 * un[j][i] + un[j-1][i]));

    v[j][i] = (vn[j][i] - vn[j][i] * dt / dx * (vn[j][i] - vn[j][i - 1]) -
           vn[j][i] * dt / dy * (vn[j][i] - vn[j - 1][i]) -
           dt / (2 * rho * dy) * (p[j+1][i] - p[j-1][i]) +
           nu * dt / (dx*dx) * (vn[j][i+1] - 2 * vn[j][i] + vn[j][i-1]) +
           nu * dt / (dy*dy) * (vn[j+1][i] - 2 * vn[j][i] + vn[j-1][i]));

      }
    }
    for (int j=0; j<ny; j++) {
      // Compute u[j][0], u[j][nx-1], v[j][0], v[j][nx-1]
      u[j][0] =0;
      u[j][nx-1]=0;
      v[j][0]=0;
      v[j][nx-1]=0; 
    }
    for (int i=0; i<nx; i++) {
      // Compute u[0][i], u[ny-1][i], v[0][i], v[ny-1][i]
      u[0][i]=0;
      v[0][i]==0;
      v[ny-1][i]=0;
      u[ny-1][i]=1;
    }
    if (n % 10 == 0) {
      /*各プロセスの担当数をなるべく均等に*/
      int num_per_proc = nx/size;/* プロセス当たりの基本担当数 4プロセスなら10*/
      int extra_num = nx % size;/* 足りないぶん 4プロセスなら１*/
      if(rank<extra_num){ /*余りがあるとき、最初のextra_numプロセスが追加で一個ずつ担当する*/
        begin = rank*(num_per_proc+1);
        end = begin + num_per_proc + 1;
      }
      else{
        begin = rank*num_per_proc + extra_num;
        end = begin + num_per_proc;
      }

      for (int j=begin; j<end; j++)
        for (int i=0; i<nx; i++)
          ufile << u[j][i] << " ";
      ufile << "\n";
      for (int j=begin; j<end; j++)
        for (int i=0; i<nx; i++)
          vfile << v[j][i] << " ";
      vfile << "\n";
      for (int j=begin; j<end; j++)
        for (int i=0; i<nx; i++)
          pfile << p[j][i] << " ";
      pfile << "\n";
    }
  }
  ufile.close();
  vfile.close();
  pfile.close();
  MPI_Finalize();
  clock_t end_time = clock();
  const double excutetime = static_cast<double>(end_time - start_time) / CLOCKS_PER_SEC * 1000.0;
  printf("time %lf[ms]\n", excutetime);
}
