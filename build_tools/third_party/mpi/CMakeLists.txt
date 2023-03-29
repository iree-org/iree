# Copyright 2023 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

find_package(MPI REQUIRED)

external_cc_library(
  PACKAGE
    mpi
  NAME
    headers_and_libs
  ROOT
    "${IREE_SOURCE_DIR}/third_party/mpi/"
  INCLUDES
    "${MPI_CXX_INCLUDE_PATH}"
  DEPS
    "${MPI_CXX_LIBRARIES}"
  PUBLIC
)
