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


class TfliteWrapper(TFLiteBenchmarkCommand):
  """Specializes the benchmark command to use TFLite."""

  def __init__(self,
               benchmark_binary: str,
               model_name: str,
               model_path: str,
               input_layer: Optional[str] = None,
               input_shape: Optional[str] = None,
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
    if input_layer and input_shape:
      self.args.append("--input_layer=%s" % input_layer)
      self.args.append("--input_layer_shape=%s" % input_shape)


class IreeWrapper(IreeBenchmarkCommand):
  """Specializes the benchmark command to use IREE."""

  def __init__(self,
               benchmark_binary: str,
               model_name: str,
               model_path: str,
               function_input: str,
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
    self.args.append("--function=main")
    self.args.append('--input="%s"' % function_input)


class SimpleCommandFactory(BenchmarkCommandFactory):
  """
  Generates `BenchmarkCommand` objects specific to running series of simple models.

  A model is considered simple if its inputs can be generically generated based
  on expected signature only without affecting behavior.
  """

  def __init__(self,
               base_dir: str,
               model_name: str,
               function_input: str,
               input_name: Optional[str] = None,
               input_layer: Optional[str] = None):
    self._model_name = model_name
    self._function_input = function_input
    self._input_name = input_name
    self._input_layer = input_layer
    self._base_dir = base_dir
    self._iree_benchmark_binary_path = os.path.join(base_dir,
                                                    "iree-benchmark-module")
    self._tflite_benchmark_binary_path = os.path.join(base_dir,
                                                      "benchmark_model")
    # Required to be set, but no test data used yet.
    self._tflite_test_data_dir = os.path.join(self._base_dir, "test_data")

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
    commands = []
    # Generate TFLite benchmarks.
    tflite_model_path = os.path.join(self._base_dir, "models", "tflite",
                                     self._model_name + ".tflite")
    tflite = TfliteWrapper(self._tflite_benchmark_binary_path,
                           self._model_name,
                           tflite_model_path,
                           self._input_name,
                           driver="cpu")
    commands.append(tflite)

    tflite_noxnn = TfliteWrapper(self._tflite_benchmark_binary_path,
                                 self._model_name + "_noxnn",
                                 tflite_model_path,
                                 self._input_name,
                                 driver="cpu")
    tflite_noxnn.args.append("--use_xnnpack=false")
    commands.append(tflite_noxnn)

    # Generate IREE benchmarks.
    driver = "local-task"
    backend = "llvm-cpu"

    iree_model_path = os.path.join(self._base_dir, "models", "iree", backend,
                                   self._model_name + ".vmfb")
    iree = IreeWrapper(self._iree_benchmark_binary_path,
                       self._model_name,
                       iree_model_path,
                       self._function_input,
                       driver=driver)
    commands.append(iree)

    model_padfuse_name = self._model_name + "_padfuse"
    iree_padfuse_model_path = os.path.join(self._base_dir, "models", "iree",
                                           backend,
                                           model_padfuse_name + ".vmfb")
    iree_padfuse = IreeWrapper(self._iree_benchmark_binary_path,
                               model_padfuse_name,
                               iree_padfuse_model_path,
                               self._function_input,
                               driver=driver)
    commands.append(iree_padfuse)

    # Test mmt4d only on mobile.
    if device == "mobile":
      model_mmt4d_name = self._model_name + "_mmt4d"
      iree_mmt4d_model_path = os.path.join(self._base_dir, "models", "iree",
                                           backend, model_mmt4d_name + ".vmfb")
      iree_mmt4d = IreeWrapper(self._iree_benchmark_binary_path,
                               model_mmt4d_name,
                               iree_mmt4d_model_path,
                               self._function_input,
                               driver=driver)
      commands.append(iree_mmt4d)

      model_im2col_mmt4d_name = self._model_name + "_im2col_mmt4d"
      iree_im2col_mmt4d_model_path = os.path.join(
          self._base_dir, "models", "iree", backend,
          model_im2col_mmt4d_name + ".vmfb")
      iree_im2col_mmt4d = IreeWrapper(self._iree_benchmark_binary_path,
                                      model_im2col_mmt4d_name,
                                      iree_im2col_mmt4d_model_path,
                                      self._function_input,
                                      driver=driver)
      commands.append(iree_im2col_mmt4d)

    return commands

  def _generate_gpu(self, driver: str):
    commands = []
    tflite_model_path = os.path.join(self._base_dir, "models", "tflite",
                                     self._model_name + ".tflite")
    tflite = TfliteWrapper(self._tflite_benchmark_binary_path,
                           self._model_name,
                           tflite_model_path,
                           self._input_name,
                           self._input_layer,
                           driver="gpu")
    tflite.args.append("--gpu_precision_loss_allowed=false")
    commands.append(tflite)

    tflite_noxnn = TfliteWrapper(self._tflite_benchmark_binary_path,
                                 self._model_name + "_noxnn",
                                 tflite_model_path,
                                 self._input_name,
                                 self._input_layer,
                                 driver="gpu")
    tflite.args.append("--use_xnnpack=false")
    commands.append(tflite_noxnn)

    tflite_fp16 = TfliteWrapper(self._tflite_benchmark_binary_path,
                                self._model_name + "_fp16",
                                tflite_model_path,
                                self._input_name,
                                self._input_layer,
                                driver="gpu")
    tflite.args.append("--gpu_precision_loss_allowed=true")
    commands.append(tflite_fp16)

    iree_model_path = os.path.join(self._base_dir, "models", "iree", driver,
                                   self._model_name + ".vmfb")
    iree = IreeWrapper(self._iree_benchmark_binary_path,
                       self._model_name,
                       iree_model_path,
                       self._function_input,
                       driver=driver)
    commands.append(iree)

    iree_model_path = os.path.join(self._base_dir, "models", "iree", driver,
                                   self._model_name + "_fp16.vmfb")
    iree = IreeWrapper(self._iree_benchmark_binary_path,
                       self._model_name + "_fp16",
                       iree_model_path,
                       self._function_input,
                       driver=driver)
    commands.append(iree)

    iree_model_path = os.path.join(self._base_dir, "models", "iree", driver,
                                   self._model_name + "_padfuse.vmfb")
    iree = IreeWrapper(self._iree_benchmark_binary_path,
                       self._model_name + "_padfuse",
                       iree_model_path,
                       self._function_input,
                       driver=driver)
    commands.append(iree)
    return commands
