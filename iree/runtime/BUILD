# Copyright 2021 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

package(
    default_visibility = ["//visibility:public"],
    features = ["layering_check"],
    licenses = ["notice"],  # Apache 2.0
)

#===------------------------------------------------------------------------===#
# Public API
#===------------------------------------------------------------------------===#

cc_library(
    name = "runtime",
    hdrs = [
        "api.h",
    ],
    deps = [
        ":impl",
        "//iree/base",
    ],
)

#===------------------------------------------------------------------------===#
# Implementation
#===------------------------------------------------------------------------===#

cc_library(
    name = "impl",
    srcs = [
        "call.c",
        "instance.c",
        "session.c",
    ],
    hdrs = [
        "call.h",
        "instance.h",
        "session.h",
    ],
    deps = [
        "//iree/base",
        "//iree/base:core_headers",
        "//iree/base:tracing",
        "//iree/base/internal",
        "//iree/base/internal:file_io",
        "//iree/hal",
        "//iree/hal/drivers",
        "//iree/modules/hal",
        "//iree/vm",
        "//iree/vm:bytecode_module",
    ],
)
