# Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Common classes for benchmark definitions."""

from dataclasses import dataclass
from enum import Enum
from typing import List
from e2e_test_framework import serialization, unique_ids
import dataclasses


class ArchitectureType(Enum):
  """Type of architecture."""
  CPU = "cpu"
  GPU = "gpu"


@dataclass(frozen=True)
class ArchitectureInfo(object):
  """Architecture information."""
  type: ArchitectureType
  architecture: str
  microarchitecture: str


class DeviceArchitecture(Enum):
  """Predefined architecture/microarchitecture."""

  # VMVX virtual machine
  VMVX_GENERIC = ArchitectureInfo(ArchitectureType.CPU, "vmvx", "generic")

  # x86_64 CPUs
  X86_64_CASCADELAKE = ArchitectureInfo(ArchitectureType.CPU, "x86_64",
                                        "cascadelake")

  # ARM CPUs
  ARMV8_2_A_GENERIC = ArchitectureInfo(ArchitectureType.CPU, "armv8.2-a",
                                       "generic")
  ARMV9_A_GENERIC = ArchitectureInfo(ArchitectureType.CPU, "armv9-a", "generic")

  # RISC-V CPUs
  RV64_GENERIC = ArchitectureInfo(ArchitectureType.CPU, "rv64", "generic")
  RV32_GENERIC = ArchitectureInfo(ArchitectureType.CPU, "rv32", "generic")

  # Mobile GPUs
  MALI_VALHALL = ArchitectureInfo(ArchitectureType.GPU, "mali", "valhall")
  ADRENO_GENERIC = ArchitectureInfo(ArchitectureType.GPU, "adreno", "generic")

  # CUDA GPUs
  CUDA_SM70 = ArchitectureInfo(ArchitectureType.GPU, "cuda", "sm_70")
  CUDA_SM80 = ArchitectureInfo(ArchitectureType.GPU, "cuda", "sm_80")


@dataclass(frozen=True)
class PlatformInfo(object):
  """Platform information of a device."""
  os: str


class DevicePlatform(Enum):
  """Predefined device platform information."""

  GENERIC_LINUX = PlatformInfo("linux")
  GENERIC_ANDROID = PlatformInfo("android")


class ModelSourceType(Enum):
  """Type of model source."""
  # Exported Linalg MLIR file.
  EXPORTED_LINALG_MLIR = "exported_linalg_mlir"
  # Exported TFLite model file.
  EXPORTED_TFLITE = "exported_tflite"
  # Exported SavedModel from Tensorflow.
  EXPORTED_TF = "exported_tf"


class InputDataFormat(Enum):
  """Model input data format."""
  ZEROS = "zeros"
  NUMPY_NPY = "numpy_npy"


@serialization.serializable(type_key="device_specs")
@dataclass(frozen=True)
class DeviceSpec(object):
  """Benchmark device specification."""
  id: str
  # Device vendor name. E.g., Pixel-6.
  vendor_name: str
  architecture: DeviceArchitecture
  platform: DevicePlatform
  # Device-specific parameters. E.g., 2-big-cores, 4-little-cores.
  # This is for modeling the spec of a heterogeneous processor. Depending on
  # which cores you run, the device has a different spec. Benchmark machines use
  # these parameters to set up the devices. E.g. set CPU mask.
  device_parameters: List[str] = dataclasses.field(default_factory=list)


@serialization.serializable(type_key="models")
@dataclass(frozen=True)
class Model(object):
  """Model to be benchmarked."""
  id: str
  # Friendly name.
  name: str
  # Tags that describe the model characteristics.
  tags: List[str]
  source_type: ModelSourceType
  source_url: str
  entry_function: str
  # Input types. E.g., ["100x100xf32", "200x200x5xf32"].
  input_types: List[str]


@serialization.serializable(type_key="model_input_data")
@dataclass(frozen=True)
class ModelInputData(object):
  """Input data to benchmark the model."""
  id: str
  # Associated model.
  model_id: str
  # Friendly name.
  name: str
  # Tags that describe the data characteristics.
  tags: List[str]
  data_format: InputDataFormat
  source_url: str


# All-zeros dummy input data. Runners will generate the zeros input with proper
# shapes.
ZEROS_MODEL_INPUT_DATA = ModelInputData(id=unique_ids.MODEL_INPUT_DATA_ZEROS,
                                        model_id="",
                                        name="zero_dummy_input",
                                        tags=[],
                                        data_format=InputDataFormat.ZEROS,
                                        source_url="")
