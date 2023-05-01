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

TEST(libmpi, DynamicLoadLibraryAndSymbols) {
  iree_dynamic_library_t* library = NULL;
  iree_hal_mpi_dynamic_symbols_t symbols = {0};
  iree_status_t status =
      iree_hal_mpi_library_load(iree_allocator_system(), &library, &symbols);
  if (!iree_status_is_ok(status)) {
    iree_status_fprint(stderr, status);
    iree_status_ignore(status);
    std::cerr << "Symbols cannot be loaded, skipping test.";
    GTEST_SKIP();
  }

  IREE_EXPECT_OK(
      MPI_RESULT_TO_STATUS(&symbols, MPI_Init(NULL, NULL), "MPI_Init"));

  int size = 0;
  IREE_EXPECT_OK(MPI_RESULT_TO_STATUS(
      &symbols, MPI_Comm_size(IREE_MPI_COMM_WORLD(&symbols), &size),
      "MPI_Comm_size"));

  int rank = 0;
  IREE_EXPECT_OK(MPI_RESULT_TO_STATUS(
      &symbols, MPI_Comm_rank(IREE_MPI_COMM_WORLD(&symbols), &rank),
      "MPI_Comm_rank"));

  EXPECT_LT(rank, size);

  std::cout << "Hello world! "
            << "I'm " << rank << " of " << size << std::endl;

  IREE_EXPECT_OK(
      MPI_RESULT_TO_STATUS(&symbols, MPI_Finalize(), "MPI_Finalize"));

  iree_dynamic_library_release(library);
}

}  // namespace
