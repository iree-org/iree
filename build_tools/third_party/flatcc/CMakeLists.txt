# Copyright 2020 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

set(FLATCC_ROOT "${IREE_ROOT_DIR}/third_party/flatcc/")

external_cc_library(
  PACKAGE
    flatcc
  NAME
    runtime
  ROOT
    ${FLATCC_ROOT}
  INCLUDES
    "${FLATCC_ROOT}/include"
  SRCS
    "src/runtime/builder.c"
    "src/runtime/emitter.c"
    "src/runtime/json_parser.c"
    "src/runtime/json_printer.c"
    "src/runtime/refmap.c"
    "src/runtime/verifier.c"
  HDRS
    "include/flatcc/flatcc_accessors.h"
    "include/flatcc/flatcc_alloc.h"
    "include/flatcc/flatcc_assert.h"
    "include/flatcc/flatcc_builder.h"
    "include/flatcc/flatcc_emitter.h"
    "include/flatcc/flatcc_endian.h"
    "include/flatcc/flatcc_epilogue.h"
    "include/flatcc/flatcc_flatbuffers.h"
    "include/flatcc/flatcc_identifier.h"
    "include/flatcc/flatcc_iov.h"
    "include/flatcc/flatcc_json_parser.h"
    "include/flatcc/flatcc_json_printer.h"
    "include/flatcc/flatcc_portable.h"
    "include/flatcc/flatcc_prologue.h"
    "include/flatcc/flatcc_refmap.h"
    "include/flatcc/flatcc_rtconfig.h"
    "include/flatcc/flatcc_types.h"
    "include/flatcc/flatcc_unaligned.h"
    "include/flatcc/flatcc_verifier.h"
    "include/flatcc/reflection/flatbuffers_common_builder.h"
    "include/flatcc/reflection/flatbuffers_common_reader.h"
  PUBLIC
)

# Define our own binary target for the CLI, for host builds only.
# flatcc's `flatcc_cli` target renames itself to `flatcc` or `flatcc_d`
# depending on the build configuration, so we prefer to just set our own name.
if(NOT CMAKE_CROSSCOMPILING)
  add_executable(iree-flatcc-cli
    "${FLATCC_ROOT}/src/cli/flatcc_cli.c"
  )

  target_link_libraries(iree-flatcc-cli
    flatcc
  )

  target_include_directories(iree-flatcc-cli SYSTEM
    PUBLIC
      "${FLATCC_ROOT}/include"
      "${FLATCC_ROOT}/config"
  )
endif()
