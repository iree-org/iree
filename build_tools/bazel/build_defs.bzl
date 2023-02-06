# Copyright 2021 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

def iree_pjrt_cc_library(copts = [], **kwargs):
    """Used for cc_library targets within the project."""
    native.cc_library(
        copts = [
            "-fvisibility=hidden",
        ] + copts,
        **kwargs
    )

def iree_pjrt_plugin_dylib(
        name,
        deps = [],
        defines = [],
        linkopts = [],
        **kwargs):
    native.cc_binary(
        name = name + ".so",
        linkshared = 1,
        linkstatic = 1,
        defines = [
            "PJRT_PLUGIN_BUILDING_LIBRARY",
        ] + defines,
        linkopts = [
            "-Wl,--no-undefined",
        ] + linkopts,
        deps = [
            "//iree/integrations/pjrt/common:impl",
            "//iree/integrations/pjrt/common:dylib_platform",
        ] + deps,
        **kwargs,
    )
