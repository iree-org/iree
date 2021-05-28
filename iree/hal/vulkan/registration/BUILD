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
if(${IREE_HAL_DRIVER_VULKAN})
""",
    inline = True,
)

cc_library(
    name = "registration",
    srcs = ["driver_module.cc"],
    hdrs = ["driver_module.h"],
    defines = [
        "IREE_HAL_HAVE_VULKAN_DRIVER_MODULE=1",
    ],
    deps = [
        "//iree/base:core_headers",
        "//iree/base:status",
        "//iree/base:tracing",
        "//iree/base/internal:flags",
        "//iree/hal",
        "//iree/hal/vulkan",
    ],
)

iree_cmake_extra_content(
    content = """
endif()
""",
    inline = True,
)
