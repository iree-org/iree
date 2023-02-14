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


class TfliteMobilebertInt8(TFLiteBenchmarkCommand):
  """ Specializes the benchmark command to use TFLite. """

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
    self.args.append("--input_layer=input_ids,segment_ids,input_mask")
    self.args.append("--input_layer_value_files=input_ids:" + test_data_dir +
                     "/input_word_id.bin,segment_ids:" + test_data_dir +
                     "/input_type_id.bin,input_mask:" + test_data_dir +
                     "/input_mask.bin")
    self.args.append("--input_layer_shape=1,384:1,384:1,384")


class IreeMobilebertInt8(IreeBenchmarkCommand):
  """ Specializes the benchmark command to use IREE. """

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
    self.args.append("--function=main")
    self.args.append(
        '--input="1x384xi32=101 2129 2116 19576 2015 2106 3854 4679 2486 1029 102 1996 14169 2165 2019 2220 2599 1999 3565 4605 2753 1998 2196 11145 1012 8446 2001 3132 2011 7573 1005 1055 3639 1010 2029 14159 2032 2698 2335 1998 3140 2032 2046 2093 20991 2015 1010 2164 1037 19576 2029 2027 6757 2005 1037 7921 1012 7573 15674 3854 4679 2001 2315 3565 4605 12041 1010 3405 2274 3948 10455 1010 1016 13714 14918 1010 1998 2048 3140 19576 2015 1012 102 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0"'
    )
    self.args.append(
        '--input="1x384xi32=0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0"'
    )
    self.args.append(
        '--input="1x384xi32=1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0"'
    )


class MobilebertInt8CommandFactory(BenchmarkCommandFactory):
  """ Generates `BenchmarkCommand` objects specific to running MobileBert."""

  def __init__(self, base_dir: str):
    self._model_name = "mobilebert-baseline-tf2-quant"
    self._base_dir = base_dir
    self._iree_benchmark_binary_path = os.path.join(base_dir,
                                                    "iree-benchmark-module")
    self._tflite_benchmark_binary_path = os.path.join(base_dir,
                                                      "benchmark_model")
    self._tflite_model_path = os.path.join(self._base_dir, "models", "tflite",
                                           self._model_name + ".tflite")
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
    tflite_mobilebert = TfliteMobilebertInt8(self._tflite_benchmark_binary_path,
                                             self._model_name,
                                             self._tflite_model_path,
                                             self._tflite_test_data_dir,
                                             driver="cpu")

    tflite_mobilebert_noxnn = TfliteMobilebertInt8(
        self._tflite_benchmark_binary_path,
        self._model_name + "_noxnn",
        self._tflite_model_path,
        self._tflite_test_data_dir,
        driver="cpu")
    tflite_mobilebert_noxnn.args.append("--use_xnnpack=false")

    # Generate IREE benchmarks.
    driver = "local-task"
    backend = "llvm-cpu"
    iree_model_path = os.path.join(self._base_dir, "models", "iree", backend,
                                   self._model_name + ".vmfb")
    iree_mobilebert = IreeMobilebertInt8(self._iree_benchmark_binary_path,
                                         self._model_name,
                                         iree_model_path,
                                         driver=driver)
    commands = [tflite_mobilebert, tflite_mobilebert_noxnn, iree_mobilebert]

    # Test mmt4d only on mobile.
    if device == "mobile":
      model_mmt4d_name = self._model_name + "_mmt4d"
      iree_mmt4d_model_path = os.path.join(self._base_dir, "models", "iree",
                                           backend, model_mmt4d_name + ".vmfb")
      iree_mmt4d_mobilebert = IreeMobilebertInt8(
          self._iree_benchmark_binary_path,
          model_mmt4d_name,
          iree_mmt4d_model_path,
          driver=driver)
      commands.append(iree_mmt4d_mobilebert)

      model_im2col_mmt4d_name = self._model_name + "_im2col_mmt4d"
      iree_im2col_mmt4d_model_path = os.path.join(
          self._base_dir, "models", "iree", backend,
          model_im2col_mmt4d_name + ".vmfb")
      iree_im2col_mmt4d_mobilebert = IreeMobilebertInt8(
          self._iree_benchmark_binary_path,
          model_im2col_mmt4d_name,
          iree_im2col_mmt4d_model_path,
          driver=driver)
      commands.append(iree_im2col_mmt4d_mobilebert)

    return commands

  def _generate_gpu(self, driver: str):
    tflite_mobilebert = TfliteMobilebertInt8(self._tflite_benchmark_binary_path,
                                             self._model_name,
                                             self._tflite_model_path,
                                             self._tflite_test_data_dir,
                                             driver="gpu")
    tflite_mobilebert.args.append("--gpu_precision_loss_allowed=false")

    tflite_mobilebert_noxnn = TfliteMobilebertInt8(
        self._tflite_benchmark_binary_path,
        self._model_name + "_noxnn",
        self._tflite_model_path,
        self._tflite_test_data_dir,
        driver="gpu")
    tflite_mobilebert_noxnn.args.append("--gpu_precision_loss_allowed=false")
    tflite_mobilebert_noxnn.args.append("--use_xnnpack=false")

    iree_model_path = os.path.join(self._base_dir, "models", "iree", driver,
                                   self._model_name + ".vmfb")
    iree_mobilebert = IreeMobilebertInt8(self._iree_benchmark_binary_path,
                                         self._model_name,
                                         iree_model_path,
                                         driver=driver)

    iree_padfuse_model_path = os.path.join(self._base_dir, "models", "iree",
                                           driver,
                                           self._model_name + "_padfuse.vmfb")
    iree_padfuse_mobilebert = IreeMobilebertInt8(
        self._iree_benchmark_binary_path,
        self._model_name + "_padfuse",
        iree_padfuse_model_path,
        driver=driver)
    return [
        tflite_mobilebert, tflite_mobilebert_noxnn, iree_mobilebert,
        iree_padfuse_mobilebert
    ]
