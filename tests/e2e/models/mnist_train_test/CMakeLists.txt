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

# Mark the CUDA test as WILL_FAIL if it was defined.
# TODO(#13007): Thread this through the test rule instead of custom logic.
iree_package_path(PKG_PATH)
set(CUDA_TEST_NAME ${PKG_PATH}/mnist_train_test_cuda)
get_directory_property(TESTS_LIST TESTS)
if(${CUDA_TEST_NAME} IN_LIST TESTS_LIST)
  set_tests_properties(
    ${CUDA_TEST_NAME}
    PROPERTIES WILL_FAIL TRUE)
endif()

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
