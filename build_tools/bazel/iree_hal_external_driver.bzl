# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Rules for registering HAL drivers that live outside of the core runtime.

This is the Bazel equivalent of the CMake `IREE_EXTERNAL_HAL_DRIVERS` support
in runtime/src/iree/hal/drivers/CMakeLists.txt. External drivers are selected
with the `//runtime/src/iree/hal/drivers:external_drivers` label flag, which
must point at an `iree_hal_external_driver` target:

  bazel build \\
    --//runtime/src/iree/hal/drivers:external_drivers=//path/to:my_driver \\
    //tools:iree-run-module

Multiple drivers can be enabled at once by pointing the flag at an
`iree_hal_external_driver` target whose deps are other
`iree_hal_external_driver` targets.
"""

load("@rules_cc//cc/common:cc_common.bzl", "cc_common")
load("@rules_cc//cc/common:cc_info.bzl", "CcInfo")

IreeHalExternalDriverInfo = provider(
    doc = "Registration functions of external HAL drivers.",
    fields = {
        "register_fns": "List of registration function names with the " +
                        "signature `iree_status_t fn(iree_hal_driver_registry_t*)`.",
    },
)

def _iree_hal_external_driver_impl(ctx):
    register_fns = list(ctx.attr.register_fns)
    for dep in ctx.attr.deps:
        if IreeHalExternalDriverInfo in dep:
            register_fns.extend(dep[IreeHalExternalDriverInfo].register_fns)
    return [
        IreeHalExternalDriverInfo(register_fns = register_fns),
        cc_common.merge_cc_infos(cc_infos = [dep[CcInfo] for dep in ctx.attr.deps]),
    ]

iree_hal_external_driver = rule(
    implementation = _iree_hal_external_driver_impl,
    doc = """Declares one or more external HAL drivers.

Provides the registration functions of the drivers along with the CcInfo of
the libraries implementing them.
""",
    attrs = {
        "deps": attr.label_list(
            doc = "Libraries implementing the registration functions and " +
                  "other iree_hal_external_driver targets.",
            providers = [CcInfo],
        ),
        "register_fns": attr.string_list(
            doc = "Names of the registration functions to call.",
        ),
    },
    provides = [IreeHalExternalDriverInfo, CcInfo],
)

def _iree_hal_external_driver_init_c_impl(ctx):
    register_fns = ctx.attr.drivers[IreeHalExternalDriverInfo].register_fns
    ctx.actions.expand_template(
        template = ctx.file.template,
        output = ctx.outputs.out,
        substitutions = {
            "${_INIT_EXTERNAL_REGISTER_DECLS}": "".join([
                "extern iree_status_t {}(iree_hal_driver_registry_t* registry);\n".format(fn)
                for fn in register_fns
            ]),
            "${_INIT_EXTERNAL_REGISTER_CALLS}": "".join([
                "IREE_RETURN_IF_ERROR({}(registry));\n".format(fn)
                for fn in register_fns
            ]),
        },
    )

iree_hal_external_driver_init_c = rule(
    implementation = _iree_hal_external_driver_init_c_impl,
    doc = "Generates the C source calling all external driver registrations.",
    attrs = {
        "drivers": attr.label(
            mandatory = True,
            providers = [IreeHalExternalDriverInfo],
        ),
        "out": attr.output(mandatory = True),
        "template": attr.label(
            mandatory = True,
            allow_single_file = True,
        ),
    },
)
