# Copyright 2020 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

load("//iree:build_defs.oss.bzl", "iree_cmake_extra_content")

package(
    default_visibility = ["//visibility:public"],
    features = ["layering_check"],
    licenses = ["notice"],  # Apache 2.0
)

iree_cmake_extra_content(
    content = """
if(NOT ${IREE_HAL_DRIVER_DYLIB})
  return()
endif()

# TODO(#4298): Separate HAL driver registration from threading.
if(NOT ${IREE_ENABLE_THREADING})
  return()
endif()
""",
    inline = True,
)

cc_library(
    name = "registration",
    srcs = ["driver_module.c"],
    hdrs = ["driver_module.h"],
    defines = [
        "IREE_HAL_HAVE_DYLIB_DRIVER_MODULE=1",
    ],
    deps = [
        "//iree/base",
        "//iree/base/internal:flags",
        "//iree/hal",
        "//iree/hal/local",
        "//iree/hal/local:task_driver",
        "//iree/hal/local/loaders:embedded_library_loader",
        "//iree/hal/local/loaders:legacy_library_loader",
        "//iree/task:api",
    ],
)

cc_library(
    name = "sync",
    srcs = ["driver_module_sync.c"],
    hdrs = ["driver_module_sync.h"],
    defines = [
        "IREE_HAL_HAVE_DYLIB_SYNC_DRIVER_MODULE=1",
    ],
    deps = [
        "//iree/base",
        "//iree/hal",
        "//iree/hal/local",
        "//iree/hal/local:sync_driver",
        "//iree/hal/local/loaders:legacy_library_loader",
    ],
)
