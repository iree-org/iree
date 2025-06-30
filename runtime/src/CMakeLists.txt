# Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

iree_setup_c_src_root(
  DEFAULT_EXPORT_SET Runtime
  DEFAULT_INSTALL_COMPONENT IREEDevLibraries-Runtime
  IMPLICIT_DEFS_TARGET iree_defs
)

# Enable LLVM coverage for the runtime libraries.
# All object archives under runtime/src/ will contain coverage information and
# binaries will be will produce profraw files when requested.
#
# TODO(benvanik): ideally we have iree_runtime_cc_library/binary and do this
# there - today we use the same macros for both compiler and runtime and this
# is the best way to scope coverage codegen to only the runtime.
if(IREE_ENABLE_RUNTIME_COVERAGE)
  message(WARNING
    "IREE_ENABLE_RUNTIME_COVERAGE enabling coverage in all runtime libraries. "
    "All runtime binaries are instrumented and should not be used for"
    "benchmarking."
  )
  add_compile_options(
    "-fprofile-instr-generate"
    "-fcoverage-mapping"
  )
  add_link_options(
    "-fprofile-instr-generate"
    "-fcoverage-mapping"
  )
endif(IREE_ENABLE_RUNTIME_COVERAGE)

add_subdirectory(iree)
