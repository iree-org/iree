// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/utils/libmpi.h"

#include <iostream>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

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

}  // namespace
