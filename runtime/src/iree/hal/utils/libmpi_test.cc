// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/utils/libmpi.h"

#include <iostream>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace iree {
namespace hal {
namespace utils {

using ::iree::StatusCode;
using ::iree::testing::status::StatusIs;

const int MPI_SUCCESS = 0;

class LibmpiTest : public ::testing::Test {
 protected:
  static void SetUpTestSuite() {
    iree_status_t status =
        iree_hal_mpi_library_load(iree_allocator_system(), &library, &symbols);
    if (!iree_status_is_ok(status)) {
      iree_status_fprint(stderr, status);
      iree_status_ignore(status);
    } else {
      EXPECT_EQ(symbols.MPI_Init(NULL, NULL), MPI_SUCCESS);
    }
  }

  void SetUp() override {
    if (!library) GTEST_SKIP() << "No MPI library available. Skipping suite.";

    IREE_EXPECT_OK(MPI_RESULT_TO_STATUS(
        &symbols, MPI_Comm_size(IREE_MPI_COMM_WORLD(&symbols), &world_size),
        "MPI_Comm_size"));

    IREE_EXPECT_OK(MPI_RESULT_TO_STATUS(
        &symbols, MPI_Comm_rank(IREE_MPI_COMM_WORLD(&symbols), &rank),
        "MPI_Comm_rank"));

    EXPECT_LT(rank, world_size);
  }

  void TearDown() override {}

  static void TearDownTestSuite() {
    if (!library) return;

    IREE_EXPECT_OK(
        MPI_RESULT_TO_STATUS(&symbols, MPI_Finalize(), "MPI_Finalize"));

    memset(&symbols, 0, sizeof(symbols));
    if (library) iree_dynamic_library_release(library);
  }

 protected:
  static iree_dynamic_library_t* library;
  static iree_hal_mpi_dynamic_symbols_t symbols;
  int rank = 0;
  int world_size = 0;
};

iree_dynamic_library_t* LibmpiTest::library = NULL;
iree_hal_mpi_dynamic_symbols_t LibmpiTest::symbols = {0};

// An MPI "hello world" program to test library loading.
TEST_F(LibmpiTest, HelloWorld) {
  std::cout << "Hello world! "
            << "I'm " << rank << " of " << world_size << std::endl;
}

TEST_F(LibmpiTest, MPIErrorToIREEStatus) {
  IREE_EXPECT_OK(iree_hal_mpi_result_to_status(NULL, 0, __FILE__, __LINE__));

  const int MPI_ERR_UNKNOWN = 14;
  EXPECT_THAT(
      iree_hal_mpi_result_to_status(NULL, MPI_ERR_UNKNOWN, __FILE__, __LINE__),
      StatusIs(StatusCode::kInternal));

  const int MPI_ERR_ACCESS = 20;
  EXPECT_THAT(iree_hal_mpi_result_to_status(&symbols, MPI_ERR_ACCESS, __FILE__,
                                            __LINE__),
              StatusIs(StatusCode::kInternal));
}

TEST_F(LibmpiTest, Bcast) {
  const int N = 100;
  char array[N];
  int root = 1;

  ASSERT_GT(world_size, 1)
      << "This test requires mpirun with more than one node";

  if (rank == root) memset(array, 'r', N);

  IREE_EXPECT_OK(
      MPI_RESULT_TO_STATUS(&symbols,
                           MPI_Bcast(array, N, IREE_MPI_BYTE(&symbols), root,
                                     IREE_MPI_COMM_WORLD(&symbols)),
                           "MPI_Bcast"));

  for (int i = 0; i < N; i++)
    EXPECT_EQ(array[i], 'r') << "[" << rank << "] failed at index " << i;
}

TEST_F(LibmpiTest, Gather) {
  const int N = 100;
  char sendarray[N];
  int root = 1;
  char *recvbuf = NULL;

  ASSERT_GT(world_size, 1)
      << "This test requires mpirun with more than one node";

  memset(sendarray, (char)(rank + 1), N);

  if (rank == root) {
    recvbuf = (char *)malloc(world_size * N * sizeof(char));
    ASSERT_NE(recvbuf, nullptr);
  }
  IREE_EXPECT_OK(MPI_RESULT_TO_STATUS(
      &symbols,
      MPI_Gather(sendarray, N, IREE_MPI_BYTE(&symbols), recvbuf, N,
                 IREE_MPI_BYTE(&symbols), root, IREE_MPI_COMM_WORLD(&symbols)),
      "MPI_Gather"));

  if (rank == root) {
    for (int i = 0; i < N * world_size; i++) {
      EXPECT_EQ(recvbuf[i], (char)((i / N) + 1))
          << "[" << rank << "] failed at index " << i;
    }
    free(recvbuf);
  }
}

TEST_F(LibmpiTest, Scatter) {
  const int N = 100;
  char recvbuf[N];
  int root = 1;
  char *sendbuf = NULL;

  ASSERT_GT(world_size, 1)
      << "This test requires mpirun with more than one node";

  if (rank == root) {
    sendbuf = (char *)malloc(world_size * N * sizeof(char));
    ASSERT_NE(sendbuf, nullptr);
    memset(sendbuf, 's', world_size * N * sizeof(char));
  }

  IREE_EXPECT_OK(MPI_RESULT_TO_STATUS(
      &symbols,
      MPI_Scatter(sendbuf, N, IREE_MPI_BYTE(&symbols), recvbuf, N,
                  IREE_MPI_BYTE(&symbols), root, IREE_MPI_COMM_WORLD(&symbols)),
      "MPI_Scatter"));

  for (int i = 0; i < N; i++) {
    EXPECT_EQ(recvbuf[i], 's') << "[" << rank << "] failed at index " << i;
  }

  if (rank == root) {
    free(sendbuf);
  }
}

TEST_F(LibmpiTest, Allgather) {
  const int N = 100;
  char sendarray[N];
  char *recvbuf = NULL;

  ASSERT_GT(world_size, 1)
      << "This test requires mpirun with more than one node";

  memset(sendarray, (char)(rank + 1), N);
  recvbuf = (char *)malloc(world_size * N * sizeof(char));
  ASSERT_NE(recvbuf, nullptr);

  IREE_EXPECT_OK(MPI_RESULT_TO_STATUS(
      &symbols,
      MPI_Allgather(sendarray, N, IREE_MPI_BYTE(&symbols), recvbuf, N,
                    IREE_MPI_BYTE(&symbols), IREE_MPI_COMM_WORLD(&symbols)),
      "MPI_Allgather"));

  for (int i = 0; i < N * world_size; i++) {
    EXPECT_EQ(recvbuf[i], (char)((i / N) + 1))
        << "[" << rank << "] failed at index " << i;
  }
  free(recvbuf);
}

TEST_F(LibmpiTest, Alltoall) {
  const int N = 100;
  char *sendbuf = NULL;
  char *recvbuf = NULL;

  ASSERT_GT(world_size, 1)
      << "Thist test requires mpirun with more than one node";

  sendbuf = (char *)malloc(world_size * N * sizeof(char));
  ASSERT_NE(sendbuf, nullptr);
  memset(sendbuf, (char)('A' + rank), N * world_size);

  recvbuf = (char *)malloc(world_size * N * sizeof(char));
  ASSERT_NE(recvbuf, nullptr);

  IREE_EXPECT_OK(MPI_RESULT_TO_STATUS(
      &symbols,
      MPI_Alltoall(sendbuf, N, IREE_MPI_BYTE(&symbols), recvbuf, N,
                   IREE_MPI_BYTE(&symbols), IREE_MPI_COMM_WORLD(&symbols)),
      "MPI_Alltoall"));

  for (int i = 0; i < N * world_size; i++) {
    EXPECT_EQ(recvbuf[i], (char)((i / N) + 'A'))
        << "[" << rank << "] failed at index " << i;
  }
  free(sendbuf);
  free(recvbuf);
}

TEST_F(LibmpiTest, ReduceSum) {
  const int N = 100;
  int sendbuf[N];
  int recvbuf[N];
  int root = 1;

  ASSERT_GT(world_size, 1)
      << "Thist test requires mpirun with more than one node";

  for (int i = 0; i < N; i++) sendbuf[i] = i + 1;

  IREE_EXPECT_OK(MPI_RESULT_TO_STATUS(
      &symbols,
      MPI_Reduce(sendbuf, recvbuf, N, IREE_MPI_INT(&symbols),
                 IREE_MPI_OP_SUM(&symbols), root,
                 IREE_MPI_COMM_WORLD(&symbols)),
      "MPI_Reduce"));

  if (rank == root) {
    for (int i = 0; i < N; i++) {
      EXPECT_EQ(recvbuf[i], world_size * (i + 1))
          << "[" << rank << "] failed at index " << i;
    }
  }
}

TEST_F(LibmpiTest, AllReduce) {
  const int N = 100;
  int sendbuf[N];
  int recvbuf[N];

  ASSERT_GT(world_size, 1)
      << "Thist test requires mpirun with more than one node";

  for (int i = 0; i < N; i++) sendbuf[i] = i + 1;

  IREE_EXPECT_OK(MPI_RESULT_TO_STATUS(
      &symbols,
      MPI_Allreduce(sendbuf, recvbuf, N, IREE_MPI_INT(&symbols),
                    IREE_MPI_OP_SUM(&symbols), IREE_MPI_COMM_WORLD(&symbols)),
      "MPI_Allreduce"));

  for (int i = 0; i < N; i++) {
    EXPECT_EQ(recvbuf[i], world_size * (i + 1))
        << "[" << rank << "] failed at index " << i;
  }
}

}  // namespace utils
}  // namespace hal
}  // namespace iree
