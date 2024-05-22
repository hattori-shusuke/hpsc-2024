#include <cassert>
#include <cstdio>
#include <chrono>
#include <vector>
#include "hdf5.h"
#include <mpi.h>
using namespace std;

int main (int argc, char** argv) {
  const int NX = 10000, NY = 10000;
  const int block_size = 2500;
  hsize_t dim[2] = {2, 2};
  int mpisize, mpirank;
  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &mpisize);
  MPI_Comm_rank(MPI_COMM_WORLD, &mpirank);
  assert(mpisize == dim[0] * dim[1]);
  hsize_t N[2] = {NX, NY};
  hsize_t Nlocal[2] = {block_size, block_size};
  hsize_t count[2] = {1, 1};
  hsize_t stride[2] = {1, 1};

  hsize_t offset[2];
  int row_factor = mpirank / 2;
  int col_factor = mpirank % 2;
  offset[0] = row_factor * 5000;
  offset[1] = col_factor * 5000;

  vector<int> buffer(Nlocal[0] * Nlocal[1], mpirank);

  // HDF5ファイルの作成と設定
  hid_t plist = H5Pcreate(H5P_FILE_ACCESS);
  H5Pset_fapl_mpio(plist, MPI_COMM_WORLD, MPI_INFO_NULL);
  hid_t file = H5Fcreate("data.h5", H5F_ACC_TRUNC, H5P_DEFAULT, plist);
  hid_t globalspace = H5Screate_simple(2, N, NULL);
  hid_t localspace = H5Screate_simple(2, Nlocal, NULL);
  hid_t dataset = H5Dcreate(file, "dataset", H5T_NATIVE_INT, globalspace,
                            H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

  for (int i = 0; i < 4; ++i) {
    int sub_row = i / 2;
    int sub_col = i % 2;
    hsize_t sub_offset[2];
    sub_offset[0] = offset[0] + sub_row * block_size;
    sub_offset[1] = offset[1] + sub_col * block_size;
    H5Sselect_hyperslab(globalspace, H5S_SELECT_SET, sub_offset, stride, count, Nlocal);
    plist = H5Pcreate(H5P_DATASET_XFER);
    H5Pset_dxpl_mpio(plist, H5FD_MPIO_COLLECTIVE);
    auto tic = chrono::steady_clock::now();
    H5Dwrite(dataset, H5T_NATIVE_INT, localspace, globalspace, plist, &buffer[0]);
    auto toc = chrono::steady_clock::now();
    H5Pclose(plist);
  }

  H5Dclose(dataset);
  H5Sclose(localspace);
  H5Sclose(globalspace);
  H5Fclose(file);

  MPI_Finalize();
}
