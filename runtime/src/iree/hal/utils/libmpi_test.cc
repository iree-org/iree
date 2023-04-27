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
    bool has_symbols = false;
    unsigned char *syms_ptr = (unsigned char *)&symbols;
    for (int i = 0; i < sizeof(symbols); i++) {
      if (*(syms_ptr + i) != 0) {
        has_symbols = true;
        break;
      }
    }
    if (!has_symbols)
      GTEST_SKIP() << "MPI library failed to load symbols. Skipping suite";

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
  static iree_dynamic_library_t *library;
  static iree_hal_mpi_dynamic_symbols_t symbols;
  int rank;
  int world_size;
};

iree_dynamic_library_t *LibmpiTest::library = NULL;
iree_hal_mpi_dynamic_symbols_t LibmpiTest::symbols = {0};

// an MPI hello_world program to test library loading
TEST_F(LibmpiTest, HelloWorld) {
  std::cout << "Hello world! "
            << "I'm " << rank << " of " << world_size << std::endl;
}

TEST_F(LibmpiTest, MPI_error_to_IREE_status) {
  iree_status_t status;
  IREE_EXPECT_OK(iree_hal_mpi_result_to_status(NULL, 0, __FILE__, __LINE__));

  const int MPI_ERR_UNKNOWN = 14;
  status =
      iree_hal_mpi_result_to_status(NULL, MPI_ERR_UNKNOWN, __FILE__, __LINE__);
  EXPECT_TRUE(iree_status_is_internal(status));
  char *buffer = NULL;
  iree_host_size_t length = 0;
  iree_allocator_t allocator = iree_allocator_system();
  if (iree_status_to_string(status, &allocator, &buffer, &length)) {
    EXPECT_THAT(buffer, testing::HasSubstr("MPI library symbols not loaded"));
    iree_allocator_free(allocator, buffer);
  }

  const int MPI_ERR_ACCESS = 20;
  status = iree_hal_mpi_result_to_status(&symbols, MPI_ERR_ACCESS, __FILE__,
                                         __LINE__);
  EXPECT_TRUE(iree_status_is_internal(status));
  ASSERT_TRUE(iree_status_to_string(status, &allocator, &buffer, &length));
  EXPECT_THAT(buffer, testing::HasSubstr("MPI_ERR_ACCESS"));
  iree_allocator_free(allocator, buffer);
}

}  // namespace
