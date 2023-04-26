// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/utils/libmpi.h"

#include <iostream>

#include "iree/testing/gtest.h"

namespace iree {
namespace hal {
namespace utils {

TEST(libmpi, DynamicLoadLibraryAndSymbols) {
  iree_dynamic_library_t *library;
  iree_hal_mpi_dynamic_symbols_t *symbols;

  iree_status_t status =
      iree_hal_mpi_library_load(iree_allocator_system(), &library, &symbols);

  if (!iree_status_is_ok(status)) {
    iree_status_fprint(stderr, status);
    iree_status_ignore(status);
    std::cerr << "Symbols cannot be loaded, skipping test.";
    GTEST_SKIP();
  }

  // an MPI hello_world program
  int rank;
  int world_size;

  status = MPI_RESULT_TO_STATUS(symbols, MPI_Init(NULL, NULL), "MPI_Init");
  EXPECT_TRUE(iree_status_is_ok(status));

  status = MPI_RESULT_TO_STATUS(
      symbols, MPI_Comm_size((void *)symbols->ompi_mpi_comm_world, &world_size),
      "MPI_Comm_size");
  EXPECT_TRUE(iree_status_is_ok(status));

  status = MPI_RESULT_TO_STATUS(
      symbols, MPI_Comm_rank((void *)symbols->ompi_mpi_comm_world, &rank),
      "MPI_Comm_rank");
  EXPECT_TRUE(iree_status_is_ok(status));

  EXPECT_TRUE(rank < world_size);

  std::cout << "Hello world! "
            << "I'm " << rank << " of " << world_size << std::endl;

  status = MPI_RESULT_TO_STATUS(symbols, MPI_Finalize(), "MPI_Finalize");
  EXPECT_TRUE(iree_status_is_ok(status));

  iree_dynamic_library_release(library);
  if (symbols) {
    memset(symbols, 0, sizeof(*symbols));
    symbols = NULL;
  }
}

}  // namespace utils
}  // namespace hal
}  // namespace iree
