# Copyright 2024 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Handles the messy affair of deriving options for targeting machines."""

import argparse

from iree.build.args import (
    expand_cl_arg_defaults,
    register_arg_parser_callback,
    cl_arg_ref,
)


class TargetMachine:
    def __init__(
        self,
        target_spec: str,
        *,
        iree_compile_device_type: str | None = None,
        extra_flags: list[str] | None = None,
    ):
        self.target_spec = target_spec
        self.iree_compile_device_type = iree_compile_device_type
        self.extra_flags = extra_flags

    @property
    def flag_list(self) -> list[str]:
        # This is just a hard-coded machine model using a single IREE device
        # type alias in the default configuration.
        flags = []
        if self.iree_compile_device_type is not None:
            flags.append(f"--iree-hal-target-device={self.iree_compile_device_type}")
        if self.extra_flags:
            flags.extend(self.extra_flags)
        return flags

    def __repr__(self):
        r = f"TargetMachine({self.target_spec}, "
        if self.iree_compile_device_type is not None:
            r += f"iree_compile_device_type='{self.iree_compile_device_type}', "
        if self.extra_flags:
            r += f"extra_flags={self.extra_flags}, "
        r += ")"
        return r


################################################################################
# Handling of --iree-hal-target-device from flags
################################################################################


HAL_TARGET_DEVICES_FROM_FLAGS_HANDLERS = {}


def handle_hal_target_devices_from_flags(*mnemonics: str):
    def decorator(f):
        for mn in mnemonics:
            HAL_TARGET_DEVICES_FROM_FLAGS_HANDLERS[mn] = f
        return f

    return decorator


def handle_unknown_hal_target_device(mnemonic: str) -> list[TargetMachine]:
    return [TargetMachine(mnemonic, iree_compile_device_type=mnemonic)]


@handle_hal_target_devices_from_flags("hip")
@expand_cl_arg_defaults
def amdgpu_hal_target_from_flags(
    mnemonic: str, *, hip_target=cl_arg_ref("iree_hip_target")
) -> list[TargetMachine]:
    if not hip_target:
        raise RuntimeError(
            "No HIP targets specified. Pass a chip to target as "
            "--iree-hip-target=gfx..."
        )
    return [
        TargetMachine(
            f"hip-{hip_target}",
            extra_flags=[
                "--iree-hal-target-device=hip",
                f"--iree-hip-target={hip_target}",
            ],
        )
    ]


@handle_hal_target_devices_from_flags("llvm-cpu", "cpu")
@expand_cl_arg_defaults
def cpu_hal_target_from_flags(
    mnemonic: str,
    *,
    cpu=cl_arg_ref("iree_llvmcpu_target_cpu"),
    features=cl_arg_ref("iree_llvmcpu_target_cpu_features"),
) -> list[TargetMachine]:
    target_spec = "cpu"
    extra_flags = [
        "--iree-hal-target-device=local",
        "--iree-hal-local-target-device-backends=llvm-cpu",
    ]
    if cpu:
        target_spec += f"-{cpu}"
        extra_flags.append(f"--iree-llvmcpu-target-cpu={cpu}")
    if features:
        target_spec += f":{features}"
        extra_flags.append(f"--iree-llvmcpu-target-cpu-features={features}")

    return [
        TargetMachine(
            f"cpu-{cpu or 'generic'}",
            extra_flags=extra_flags,
        )
    ]


################################################################################
# Flag definition
################################################################################


@register_arg_parser_callback
def _(p: argparse.ArgumentParser):
    g = p.add_argument_group(
        title="IREE Target Machine Options",
        description="Global options controlling invocation of iree-compile",
    )
    g.add_argument(
        "--iree-hal-target-device",
        help="Compiles with a single machine model and a single specified device"
        " (mutually exclusive with other ways to set the machine target). This "
        "emulates the simple case of device targeting if invoking `iree-compile` "
        "directly and is mostly a pass-through which also enforces other flags "
        "depending on the value given. Supported options (or any supported by the "
        "compiler): "
        f"{', '.join(HAL_TARGET_DEVICES_FROM_FLAGS_HANDLERS.keys() - 'default')}",
        nargs="*",
    )

    hip_g = p.add_argument_group(
        title="IREE HIP Target Options",
        description="Options controlling explicit targeting of HIP devices",
    )
    hip_g.add_argument(
        "--iree-hip-target",
        help="HIP target selection (i.e. 'gfxYYYY')",
    )

    cpu_g = p.add_argument_group(
        title="IREE CPU Target Options",
        description="These are mostly pass-through. See `iree-compile --help` for "
        "full information. Advanced usage will require an explicit machine config "
        "file",
    )
    cpu_g.add_argument(
        "--iree-llvmcpu-target-cpu",
        help="'generic', 'host', or an explicit CPU name. See iree-compile help.",
    )
    cpu_g.add_argument(
        "--iree-llvmcpu-target-cpu-features",
        help="Comma separated list of '+' prefixed CPU features. See iree-compile help.",
    )


################################################################################
# Global flag dispatch
################################################################################


@expand_cl_arg_defaults
def compute_target_machines_from_flags(
    *,
    explicit_hal_target_devices: list[str]
    | None = cl_arg_ref("iree_hal_target_device"),
) -> list[TargetMachine]:
    if explicit_hal_target_devices is not None:
        # Most basic default case for setting up compilation.
        machines = []
        for explicit_hal_target_device in explicit_hal_target_devices:
            handler = (
                HAL_TARGET_DEVICES_FROM_FLAGS_HANDLERS.get(explicit_hal_target_device)
                or handle_unknown_hal_target_device
            )
            machines.extend(handler(explicit_hal_target_device))
        return machines

    raise RuntimeError(
        "iree-compile target information is required but none was provided. "
        "See flags: --iree-hal-target-device"
    )
