# Copyright 2019 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Just import if IREE_HOST_BIN_DIR is set (e.g. when cross-compiling).
if(IREE_HOST_BIN_DIR)
  iree_import_binary(NAME iree-c-embed-data)
  install(IMPORTED_RUNTIME_ARTIFACTS iree-c-embed-data
          COMPONENT iree-c-embed-data
          RUNTIME DESTINATION bin
          BUNDLE DESTINATION bin)
  return()
endif()

add_executable(iree-c-embed-data)
target_sources(iree-c-embed-data PRIVATE iree-c-embed-data-main.cc)
set_target_properties(iree-c-embed-data PROPERTIES
  OUTPUT_NAME iree-c-embed-data
  RUNTIME_OUTPUT_DIRECTORY "${IREE_BINARY_DIR}/tools"
)

install(TARGETS iree-c-embed-data
        COMPONENT IREETools-Runtime
        RUNTIME DESTINATION bin
        BUNDLE DESTINATION bin)

iree_c_embed_data(
  NAME
    "testembed1"
  SRCS
    "file1.bin"
    "data/file2.bin"
  C_FILE_OUTPUT
    "testembed1.c"
  H_FILE_OUTPUT
    "testembed1.h"
  FLATTEN
  TESTONLY
)

iree_c_embed_data(
  NAME
    "testembed2"
  SRCS
    "data/file3.bin"
  C_FILE_OUTPUT
    "testembed2.c"
  H_FILE_OUTPUT
    "testembed2.h"
  FLATTEN
  TESTONLY
)

iree_cc_test(
  NAME
    "iree_c_embed_data_test"
  SRCS
    "iree_c_embed_data_test.cc"
  DEPS
    ::testembed1
    ::testembed2
    iree::testing::gtest
    iree::testing::gtest_main
)
