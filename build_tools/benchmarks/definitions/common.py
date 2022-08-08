# Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from dataclasses import dataclass
from enum import Enum
from typing import List


@dataclass(frozen=True)
class ArchitectureInfo(object):
  architecture: str
  microarchitecture: str


class DeviceArchitecture(Enum):
  """Predefined architecture/microarchitecture."""
  X86_64_CASCADELAKE = ArchitectureInfo("x86_64", "CascadeLake")


class DevicePlatform(Enum):
  """OS platform and ABI."""
  LINUX_GNU = "linux-gnu"


class ModelSourceType(Enum):
  # Exported TFLite model file.
  EXPORTED_TFLITE = "exported_tflite"
  # Exported SavedModel from Tensorflow.
  EXPORTED_TF = "exported_tf"


class InputDataFormat(Enum):
  ZEROS = "zeros"
  NUMPY_NPY = "numpy_npy"


@dataclass(frozen=True)
class DeviceSpec(object):
  id: str
  vendor_name: str
  architecture: DeviceArchitecture
  platform: DevicePlatform
  # The device-specific configurations. E.g., big-cores.
  # Benchmark machines use these configs to set up the devices.
  configs: List[str]


@dataclass(frozen=True)
class Model(object):
  id: str
  name: str
  tags: List[str]
  source_type: ModelSourceType
  source_uri: str
  entry_function: str
  input_types: List[str]


@dataclass(frozen=True)
class ModelInputData(object):
  id: str
  model_id: str
  name: str
  tags: List[str]
  data_format: InputDataFormat
  source_uri: str


ZEROS_MODEL_INPUT_DATA = ModelInputData(id="zeros",
                                        model_id="any",
                                        name="zeros_dummy_input",
                                        tags=[],
                                        data_format=InputDataFormat.ZEROS,
                                        source_uri="")
