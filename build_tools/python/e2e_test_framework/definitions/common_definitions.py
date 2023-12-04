# Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Common classes for benchmark definitions."""

import dataclasses
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Sequence

from e2e_test_framework import serialization, unique_ids


class ArchitectureType(Enum):
    """Type of architecture."""

    CPU = "cpu"
    GPU = "gpu"


@dataclass(frozen=True)
class _ArchitectureInfo(object):
    """Architecture information."""

    type: ArchitectureType
    architecture: str
    microarchitecture: str = ""
    vendor: str = ""


class DeviceArchitecture(_ArchitectureInfo, Enum):
    """Predefined architecture/microarchitecture."""

    # VMVX virtual machine
    VMVX_GENERIC = (ArchitectureType.CPU, "vmvx", "generic")

    # x86_64 CPUs
    X86_64_CASCADELAKE = (ArchitectureType.CPU, "x86_64", "cascadelake")

    # ARM CPUs
    ARMV8_2_A_GENERIC = (ArchitectureType.CPU, "armv8.2-a", "generic")
    ARMV9_A_GENERIC = (ArchitectureType.CPU, "armv9-a", "generic")

    # RISC-V CPUs
    RV64_GENERIC = (ArchitectureType.CPU, "riscv_64", "generic")
    RV32_GENERIC = (ArchitectureType.CPU, "riscv_32", "generic")

    # Vulkan GPUs
    QUALCOMM_ADRENO = (ArchitectureType.GPU, "adreno", "", "qualcomm")
    ARM_VALHALL = (ArchitectureType.GPU, "valhall", "", "arm")
    NVIDIA_AMPERE = (ArchitectureType.GPU, "ampere", "", "nvidia")
    NVIDIA_PASCAL = (ArchitectureType.GPU, "pascal", "", "nvidia")

    # CUDA GPUs
    CUDA_SM70 = (ArchitectureType.GPU, "cuda", "sm_70")
    CUDA_SM80 = (ArchitectureType.GPU, "cuda", "sm_80")

    # Starting from 3.11, enum members are defined before the subclasses (don't
    # follow MRO, see https://docs.python.org/3/whatsnew/3.11.html#enum).
    # Therefore __str__ is defined here instead of in _ArchitectureInfo to
    # override the default one.
    def __str__(self):
        parts = [self.vendor, self.architecture, self.microarchitecture]
        return "-".join(part for part in parts if part != "")


@dataclass(frozen=True)
class _HostEnvironmentInfo(object):
    """Environment information of a host.

    The definitions and terms here matches the macros in
    `runtime/src/iree/base/target_platform.h`.

    Note that this is the environment where the runtime "runs". For example:
    ```
    {
      "platform": "linux",
      "architecture": "x86_64"
    }
    ```
    means the runtime will run on a Linux x86_64 host. The runtime might dispatch
    the workloads on GPU or it can be a VM to run workloads compiled in another
    ISA, but those are irrelevant to the information here.
    """

    platform: str
    architecture: str


class HostEnvironment(_HostEnvironmentInfo, Enum):
    """Predefined host environment."""

    LINUX_X86_64 = ("linux", "x86_64")
    LINUX_RISCV_64 = ("linux", "riscv_64")
    LINUX_RISCV_32 = ("linux", "riscv_32")
    ANDROID_ARMV8_2_A = ("android", "armv8.2-a")


class ModelSourceType(Enum):
    """Type of model source."""

    # Exported Linalg MLIR file.
    EXPORTED_LINALG_MLIR = "exported_linalg_mlir"
    # Exported Stable HLO file.
    EXPORTED_STABLEHLO_MLIR = "exported_stablehlo_mlir"
    # Exported TFLite model file.
    EXPORTED_TFLITE = "exported_tflite"


@serialization.serializable
@dataclass(frozen=True)
class CPUParameters:
    """Describes CPU related parameters."""

    # CPU cores to pin at, ordered from the slowest to the fastest.
    pinned_cores: List[int]


@serialization.serializable
@dataclass(frozen=True)
class DeviceParameters:
    """Describes device parameters."""

    cpu_params: Optional[CPUParameters] = None


@serialization.serializable(type_key="device_specs")
@dataclass(frozen=True)
class DeviceSpec(object):
    """Benchmark device specification."""

    id: str

    # Unique name of the device spec.
    name: str

    # Device name. E.g., Pixel-6.
    device_name: str

    # Tags to describe the device spec.
    tags: List[str]

    # Host environment where the IREE runtime is running. For CPU device type,
    # this is usually the same as the device that workloads are dispatched to.
    # With a separate device, such as a GPU, however, the runtime and dispatched
    # workloads will run on different platforms.
    host_environment: HostEnvironment

    # Architecture of the target device.
    architecture: DeviceArchitecture

    # Device-specific parameters. E.g., 2-big-cores, 4-little-cores.
    # This is for modeling the spec of a heterogeneous processor. Depending on
    # which cores you run, the device has a different spec. Benchmark machines use
    # these parameters to set up the devices. E.g. set CPU mask.
    device_parameters: DeviceParameters

    def __str__(self):
        return self.name

    @classmethod
    def build(
        cls,
        id: str,
        device_name: str,
        host_environment: HostEnvironment,
        architecture: DeviceArchitecture,
        device_parameters: DeviceParameters = DeviceParameters(),
        tags: Sequence[str] = (),
    ):
        tag_part = ",".join(tags)
        # Format: <device_name>[<tag>,...]
        name = f"{device_name}[{tag_part}]"
        return cls(
            id=id,
            name=name,
            tags=list(tags),
            device_name=device_name,
            host_environment=host_environment,
            architecture=architecture,
            device_parameters=device_parameters,
        )


@serialization.serializable(type_key="models")
@dataclass(frozen=True)
class Model(object):
    """Model to be benchmarked."""

    id: str
    # Friendly unique name.
    name: str
    # Tags that describe the model characteristics.
    tags: List[str]
    source_type: ModelSourceType
    source_url: str
    entry_function: str
    # Input types. E.g., ["100x100xf32", "200x200x5xf32"].
    input_types: List[str]
    # URL to fetch input data tgz. The archive should contain
    # "input_{0,1,...}.npy" for each input.
    input_url: Optional[str] = None
    # URL to fetch expected output tgz. The archive should contain
    # "ouptut_0.npy".
    expected_output_url: Optional[str] = None
    # Parameters for iree-run-module to control the tolerance.
    # For example: --expected_f32_threshold=0.0001
    verify_params: List[str] = dataclasses.field(default_factory=list)

    def __str__(self):
        return self.name


@serialization.serializable(type_key="model_input_data")
@dataclass(frozen=True)
class ModelInputData(object):
    """Input data to benchmark the model."""

    id: str
    # Associated model.
    name: str

    def __str__(self):
        return self.name


# Get input from model input_url if available; otherwise use all zeros.
DEFAULT_INPUT_DATA = ModelInputData(
    id=unique_ids.MODEL_INPUT_DATA_ZEROS,
    name="default",
)


@dataclass(frozen=True)
class CpuBenchmarkConfig(object):
    model: Model
    threads: List[int]
