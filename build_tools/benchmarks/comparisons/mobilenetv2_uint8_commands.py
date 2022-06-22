# Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import os
from typing import Optional

from common.benchmark_command import *
from common.benchmark_command_factory import BenchmarkCommandFactory

_DEFAULT_NUM_BENCHMARK_RUNS = 50
_DEFAULT_NUM_THREADS = 1


class TfliteMobilenetV2UINT8(TFLiteBenchmarkCommand):
  """Specializes the benchmark command to use TFLite."""

  def __init__(self,
               benchmark_binary: str,
               model_name: str,
               model_path: str,
               test_data_dir: str,
               driver: str = "cpu",
               num_threads: int = _DEFAULT_NUM_THREADS,
               num_runs: int = _DEFAULT_NUM_BENCHMARK_RUNS,
               taskset: Optional[str] = None):
    super().__init__(benchmark_binary,
                     model_name,
                     model_path,
                     num_threads,
                     num_runs,
                     taskset=taskset)
    self.driver = driver
    self.args.append("--input_layer=input")
    self.args.append("--input_layer_shape=1,224,224,3")


class IreeMobilenetV2UINT8(IreeBenchmarkCommand):
  """Specializes the benchmark command to use IREE."""

  def __init__(self,
               benchmark_binary: str,
               model_name: str,
               model_path: str,
               driver: str = "local-task",
               num_threads: int = _DEFAULT_NUM_THREADS,
               num_runs: int = _DEFAULT_NUM_BENCHMARK_RUNS,
               taskset: Optional[str] = None):
    super().__init__(benchmark_binary,
                     model_name,
                     model_path,
                     num_threads,
                     num_runs,
                     taskset=taskset)
    self.driver = driver
    self.args.append("--entry_function=main")
    self.args.append('--function_input="1x224x224x3xui8"')


class MobilenetV2UINT8CommandFactory(BenchmarkCommandFactory):
  """Generates `BenchmarkCommand` objects specific to running MobileNet."""

  def __init__(self, base_dir: str):
    self._model_name = "mobilenet_v2_224_1.0_uint8"
    self._base_dir = base_dir
    self._iree_benchmark_binary_path = os.path.join(base_dir,
                                                    "iree-benchmark-module")
    self._tflite_benchmark_binary_path = os.path.join(base_dir,
                                                      "benchmark_model")
    self._tflite_model_path = os.path.join(self._base_dir, "models", "tflite",
                                           self._model_name + ".tflite")
    # Required to be set, but no test data used yet.
    self._tflite_test_data_dir = os.path.join(self._base_dir, "test_data",
                                              "squad")

  def generate_benchmark_commands(self, device: str,
                                  driver: str) -> list[BenchmarkCommand]:
    if device == "desktop" and driver == "cpu":
      return self._generate_cpu(device)
    elif device == "desktop" and driver == "gpu":
      return self._generate_gpu("cuda")
    elif device == "mobile" and driver == "cpu":
      return self._generate_cpu(device)
    elif device == "mobile" and driver == "gpu":
      return self._generate_gpu("vulkan")
    else:
      print("Warning! Not a valid configuration.")
      return []

  def _generate_cpu(self, device: str):
    # Generate TFLite benchmarks.
    tflite_mobilenet = TfliteMobilenetV2UINT8(
        self._tflite_benchmark_binary_path,
        self._model_name,
        self._tflite_model_path,
        self._tflite_test_data_dir,
        driver="cpu")

    # Generate IREE benchmarks.
    driver = "local-task"
    iree_model_path = os.path.join(self._base_dir, "models", "iree", driver,
                                   self._model_name + ".vmfb")
    iree_mobilenet = IreeMobilenetV2UINT8(self._iree_benchmark_binary_path,
                                          self._model_name,
                                          iree_model_path,
                                          driver=driver)
    commands = [tflite_mobilenet, iree_mobilenet]
    return commands

  def _generate_gpu(self, driver: str):
    tflite_mobilenet = TfliteMobilenetV2UINT8(
        self._tflite_benchmark_binary_path,
        self._model_name,
        self._tflite_model_path,
        self._tflite_test_data_dir,
        driver="gpu")
    tflite_mobilenet.args.append("--gpu_precision_loss_allowed=true")

    iree_model_path = os.path.join(self._base_dir, "models", "iree", driver,
                                   self._model_name + ".vmfb")
    iree_mobilenet = IreeMobilenetV2UINT8(self._iree_benchmark_binary_path,
                                          self._model_name,
                                          iree_model_path,
                                          driver=driver)
    return [tflite_mobilenet, iree_mobilenet]
