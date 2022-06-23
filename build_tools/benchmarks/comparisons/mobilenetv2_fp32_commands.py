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


class TfliteMobilenetV2FP32(TFLiteBenchmarkCommand):
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


class IreeMobilenetV2FP32(IreeBenchmarkCommand):
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
    self.args.append('--function_input="1x224x224x3xf32"')


class MobilenetV2FP32CommandFactory(BenchmarkCommandFactory):
  """Generates `BenchmarkCommand` objects specific to running MobileNet."""

  def __init__(self, base_dir: str):
    self._model_name = "mobilenet_v2_1.0_224"
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
    tflite_mobilenet = TfliteMobilenetV2FP32(self._tflite_benchmark_binary_path,
                                             self._model_name,
                                             self._tflite_model_path,
                                             self._tflite_test_data_dir,
                                             driver="cpu")

    # Generate IREE benchmarks.
    driver = "local-task"
    iree_model_path = os.path.join(self._base_dir, "models", "iree", driver,
                                   self._model_name + ".vmfb")
    iree_mobilenet = IreeMobilenetV2FP32(self._iree_benchmark_binary_path,
                                         self._model_name,
                                         iree_model_path,
                                         driver=driver)
    commands = [tflite_mobilenet, iree_mobilenet]

    # Test mmt4d only on mobile.
    if device == "mobile":
      model_mmt4d_name = self._model_name + "_mmt4d"
      iree_mmt4d_model_path = os.path.join(self._base_dir, "models", "iree",
                                           driver, model_mmt4d_name + ".vmfb")
      iree_mmt4d_mobilenet = IreeMobilenetV2FP32(
          self._iree_benchmark_binary_path,
          model_mmt4d_name,
          iree_mmt4d_model_path,
          driver=driver)
      commands.append(iree_mmt4d_mobilenet)

    return commands

  def _generate_gpu(self, driver: str):
    tflite_mobilenet = TfliteMobilenetV2FP32(self._tflite_benchmark_binary_path,
                                             self._model_name,
                                             self._tflite_model_path,
                                             self._tflite_test_data_dir,
                                             driver="gpu")
    tflite_mobilenet.args.append("--gpu_precision_loss_allowed=false")

    tflite_mobilenet_fp16 = TfliteMobilenetV2FP32(
        self._tflite_benchmark_binary_path,
        self._model_name + "_fp16",
        self._tflite_model_path,
        self._tflite_test_data_dir,
        driver="gpu")
    tflite_mobilenet_fp16.args.append("--gpu_precision_loss_allowed=true")

    iree_model_path = os.path.join(self._base_dir, "models", "iree", driver,
                                   self._model_name + ".vmfb")
    iree_mobilenet = IreeMobilenetV2FP32(self._iree_benchmark_binary_path,
                                         self._model_name,
                                         iree_model_path,
                                         driver=driver)
    return [tflite_mobilenet, tflite_mobilenet_fp16, iree_mobilenet]
