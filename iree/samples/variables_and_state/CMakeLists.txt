# Copyright 2021 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

set(_NAME "iree_samples_variables_and_state")
add_executable(${_NAME} "")
target_sources(${_NAME}
  PRIVATE
    main.c
)

set_target_properties(${_NAME} PROPERTIES OUTPUT_NAME "variables-and-state")

target_link_libraries(${_NAME}
  iree_runtime_runtime
)
