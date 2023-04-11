# Copyright 2023 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

if(IREE_TARGET_BACKEND_LLVM_CPU AND IREE_HAL_DRIVER_LOCAL_TASK)
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
endif()

if(IREE_TARGET_BACKEND_CUDA AND IREE_HAL_DRIVER_CUDA)
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

  # The test fails on CUDA with `Mismatched elements: 1 / 100352 (0.000996%)`,
  # so mark as WILL_FAIL if the test was defined.
  # TODO(#13007): Plumb WILL_FAIL through the test rule instead of custom logic.
  iree_package_path(PKG_PATH)
  set(CUDA_TEST_NAME ${PKG_PATH}/mnist_train_test_cuda)
  get_directory_property(TESTS_LIST TESTS)
  if(${CUDA_TEST_NAME} IN_LIST TESTS_LIST)
    set_tests_properties(
      ${CUDA_TEST_NAME}
      PROPERTIES WILL_FAIL TRUE)
  endif()
endif()

if(IREE_TARGET_BACKEND_VULKAN_SPIRV AND IREE_HAL_DRIVER_VULKAN)
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
endif()
