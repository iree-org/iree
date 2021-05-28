# Copyright 2019 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Test utilities for HAL-specific code.

package(
    default_visibility = ["//visibility:public"],
    features = ["layering_check"],
    licenses = ["notice"],  # Apache 2.0
)

cc_library(
    name = "driver_registry",
    testonly = True,
    hdrs = ["driver_registry.h"],
    deps = [
        "//iree/hal",
        "//iree/hal/drivers",
    ],
)
