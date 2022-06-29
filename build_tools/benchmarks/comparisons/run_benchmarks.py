# Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
""" Runs benchmarks and saves results to a .csv file

Expects a directory structure of:
<root-benchmark-dir>/
  └── ./benchmark_model (TFLite benchmark binary)
      ./iree-benchmark-module (IREE benchmark binary)
  ├── setup/
        ├── set_adreno_gpu_scaling_policy.sh
        ├── set_android_scaling_governor.sh
        └── set_pixel6_gpu_scaling_policy.sh
  ├── test_data/
  └── models/
        ├── tflite/*.tflite
        └── iree/
              └── <driver>/*.vmfb e.g. dylib, vulkan, cuda.

"""

import argparse
import os

from common.benchmark_runner import *
from common.utils import *
from mobilebert_fp32_commands import *
from simple_commands import *


def benchmark_desktop_cpu(device_name: str,
                          command_factories: list[BenchmarkCommandFactory],
                          results_path: str):
  benchmarks = []
  for factory in command_factories:
    benchmarks.extend(factory.generate_benchmark_commands("desktop", "cpu"))

  for num_threads in [1, 2, 4, 8]:
    for benchmark in benchmarks:
      results_array = [
          device_name, benchmark.model_name, benchmark.runtime,
          benchmark.driver, num_threads
      ]
      benchmark.num_threads = num_threads
      results_array.extend(run_command(benchmark))
      write_benchmark_result(results_array, results_path)


def benchmark_desktop_gpu(device_name: str,
                          command_factories: list[BenchmarkCommandFactory],
                          results_path: str):
  benchmarks = []
  for factory in command_factories:
    benchmarks.extend(factory.generate_benchmark_commands("desktop", "gpu"))
  for benchmark in benchmarks:
    results_array = [
        device_name, benchmark.model_name, benchmark.runtime, benchmark.driver,
        benchmark.num_threads
    ]
    results_array.extend(run_command(benchmark))
    write_benchmark_result(results_array, results_path)


def benchmark_mobile_cpu(device_name: str,
                         command_factories: list[BenchmarkCommandFactory],
                         results_path: str):
  benchmarks = []
  for factory in command_factories:
    benchmarks.extend(factory.generate_benchmark_commands("mobile", "cpu"))

  for _, tuple in enumerate([("80", 1), ("C0", 2), ("F0", 4), ("0F", 4),
                             ("FF", 8)]):
    taskset = tuple[0]
    num_threads = tuple[1]
    for benchmark in benchmarks:
      results_array = [
          device_name, benchmark.model_name, benchmark.runtime,
          benchmark.driver, taskset, num_threads
      ]
      benchmark.taskset = taskset
      benchmark.num_threads = num_threads
      results_array.extend(run_command(benchmark))
      write_benchmark_result(results_array, results_path)


def benchmark_mobile_gpu(device_name: str,
                         command_factories: list[BenchmarkCommandFactory],
                         results_path: str):
  benchmarks = []
  for factory in command_factories:
    benchmarks.extend(factory.generate_benchmark_commands("mobile", "gpu"))

  taskset = "80"
  num_threads = 1
  for benchmark in benchmarks:
    results_array = [
        device_name, benchmark.model_name, benchmark.runtime, benchmark.driver,
        taskset, num_threads
    ]
    benchmark.taskset = taskset
    benchmark.num_threads = num_threads
    results_array.extend(run_command(benchmark))
    write_benchmark_result(results_array, results_path)


def main(args):
  # Create factories for all models to be benchmarked.
  command_factory = []
  command_factory.append(MobilebertFP32CommandFactory(args.base_dir))
  command_factory.append(
      SimpleCommandFactory(args.base_dir, "mobilenet_v2_1.0_224",
                           "1x224x224x3xf32"))
  command_factory.append(
      SimpleCommandFactory(args.base_dir, "mobilenet_v2_224_1.0_uint8",
                           "1x224x224x3xui8", "input", "1,224,224,3"))

  if args.mode == "desktop":
    results_path = os.path.join(args.output_dir, "results.csv")
    with open(results_path, "w") as f:
      f.write(
          "device,model,runtime,driver/delegate,threads,latency (ms),vmhwm (KB),vmrss (KB),rssfile (KB)\n"
      )

    if not args.disable_cpu:
      benchmark_desktop_cpu(args.device_name, command_factory, results_path)
    if not args.disable_gpu:
      benchmark_desktop_gpu(args.device_name, command_factory, results_path)
  else:
    assert (args.mode == "mobile")
    results_path = os.path.join(args.output_dir, "results.csv")
    with open(results_path, "w") as f:
      f.write(
          "device,model,runtime,driver/delegate,taskset,threads,latency (ms),vmhwm (KB),vmrss (KB),rssfile (KB)\n"
      )
    if not args.disable_cpu:
      benchmark_mobile_cpu(args.device_name, command_factory, results_path)
    if not args.disable_gpu:
      benchmark_mobile_gpu(args.device_name, command_factory, results_path)


def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument(
      "--device_name",
      type=str,
      default=None,
      help="The name of the device the benchmark is running on e.g. Pixel 6")
  parser.add_argument(
      "--base_dir",
      type=str,
      default=None,
      help="The directory where all benchmarking artifacts are located.")
  parser.add_argument("--output_dir",
                      type=str,
                      default=None,
                      help="The directory to save output artifacts into.")
  parser.add_argument(
      "--mode",
      type=str,
      choices=("desktop", "mobile"),
      default="desktop",
      help="The benchmarking mode to use. If mode is `mobile`, uses tasksets.")
  parser.add_argument("--disable_cpu",
                      action="store_true",
                      help="Disables running benchmarks on CPU.")
  parser.add_argument("--disable_gpu",
                      action="store_true",
                      help="Disables running benchmarks on GPU.")
  return parser.parse_args()


if __name__ == '__main__':
  main(parse_args())
