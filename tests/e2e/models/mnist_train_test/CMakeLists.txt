# Copyright 2023 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

include(iree_macros)

iree_py_test(
  NAME
    mnist_train_test_cpu
  SRCS
    "mnist_train_test.py"
  ARGS
    "--target_backend=llvm-cpu"
    "--driver=local-task"
  LABELS
    "driver=local-task"
)

iree_py_test(
  NAME
    mnist_train_test_cuda
  SRCS
    "mnist_train_test.py"
  ARGS
    "--target_backend=cuda"
    "--driver=cuda"
  LABELS
    "requires-gpu-nvidia"
    "driver=cuda"
)
iree_package_path(PKG_PATH)
set_tests_properties(
  ${PKG_PATH}/mnist_train_test_cuda
  PROPERTIES WILL_FAIL TRUE)

iree_py_test(
  NAME
    mnist_train_test_vulkan
  SRCS
    "mnist_train_test.py"
  ARGS
    "--target_backend=vulkan-spirv"
    "--driver=vulkan"
  LABELS
    "driver=vulkan"
)
