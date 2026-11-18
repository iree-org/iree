# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Rule for generating IREE HAL driver init C files."""

def _iree_hal_driver_init_c_generator_impl(ctx):
    output_file = ctx.outputs.out
    template = ctx.file.template

    substitutions = {
        "${_INIT_EXTERNAL_REGISTER_DECLS}": "\n".join([
            "extern iree_status_t {}(iree_hal_driver_registry_t* registry);".format(f)
            for f in ctx.attr.registration_functions
        ]),
        "${_INIT_EXTERNAL_REGISTER_CALLS}": "\n".join([
            "IREE_RETURN_IF_ERROR({}(registry));".format(f)
            for f in ctx.attr.registration_functions
        ]),
    }

    ctx.actions.expand_template(
        template = template,
        output = output_file,
        substitutions = substitutions,
    )

iree_hal_driver_init_c_generator = rule(
    implementation = _iree_hal_driver_init_c_generator_impl,
    attrs = {
        "out": attr.output(mandatory = True),
        "template": attr.label(
            mandatory = True,
            allow_single_file = True,
        ),
        "registration_functions": attr.string_list(),
    },
)
