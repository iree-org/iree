# Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Common classes for benchmark definitions."""

from dataclasses import dataclass
from enum import Enum
from typing import List


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
  # x86_64 CPUs
  X86_64_CASCADELAKE = ArchitectureInfo(ArchitectureType.CPU, "x86_64",
                                        "cascadelake")

  # ARM CPUs
  ARMV8_2_A_GENERIC = ArchitectureInfo(ArchitectureType.CPU, "armv8.2-a",
                                       "generic")

  # RISC-V CPUs
  RV64_GENERIC = ArchitectureInfo(ArchitectureType.CPU, "rv64", "generic")
  RV32_GENERIC = ArchitectureInfo(ArchitectureType.CPU, "rv32", "generic")

  # Mobile GPUs
  VALHALL_GENERIC = ArchitectureInfo(ArchitectureType.GPU, "valhall", "generic")
  ADRENO_GENERIC = ArchitectureInfo(ArchitectureType.GPU, "adreno", "generic")

  # CUDA GPUs
  CUDA_SM70 = ArchitectureInfo(ArchitectureType.GPU, "cuda", "sm_70")
  CUDA_SM80 = ArchitectureInfo(ArchitectureType.GPU, "cuda", "sm_80")


class DevicePlatform(Enum):
  """OS platform and ABI."""
  LINUX_GNU = "linux-gnu"
  LINUX_ANDROID29 = "linux-android29"


class ModelSourceType(Enum):
  """Type of model source."""
  # Exported MLIR file.
  EXPORTED_MLIR = "exported_mlir"
  # Exported TFLite model file.
  EXPORTED_TFLITE = "exported_tflite"
  # Exported SavedModel from Tensorflow.
  EXPORTED_TF = "exported_tf"


class InputDataFormat(Enum):
  """Model input data format."""
  RANDOM = "random"
  NUMPY_NPY = "numpy_npy"


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
  device_parameters: List[str]


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


# Random dummy input data. The benchmark runner generates the random inputs with
# a proper shape.
RANDOM_MODEL_INPUT_DATA = ModelInputData(id="random",
                                         model_id="",
                                         name="random_dummy_input",
                                         tags=[],
                                         data_format=InputDataFormat.RANDOM,
                                         source_url="")
