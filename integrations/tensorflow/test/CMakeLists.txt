# Copyright 2021 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception


if(${IREE_ENABLE_NEW_INTEGRATION_TESTS})
  add_subdirectory(tf_integration)
endif()

if(${IREE_ENABLE_TFLITE_INTEGRATION_TESTS})
  add_subdirectory(tflite_integration)
endif()
