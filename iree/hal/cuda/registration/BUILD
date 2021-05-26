# Copyright 2021 The IREE Authors
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
if(${IREE_HAL_DRIVER_CUDA})
""",
    inline = True,
)

cc_library(
    name = "registration",
    srcs = ["driver_module.c"],
    hdrs = ["driver_module.h"],
    defines = [
        "IREE_HAL_HAVE_CUDA_DRIVER_MODULE=1",
    ],
    deps = [
        "//iree/base:core_headers",
        "//iree/base:status",
        "//iree/base:tracing",
        "//iree/hal",
        "//iree/hal/cuda",
    ],
)

iree_cmake_extra_content(
    content = """
endif()
""",
    inline = True,
)
