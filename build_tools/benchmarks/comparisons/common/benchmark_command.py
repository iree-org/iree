# Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import abc
import re

from typing import Optional


class BenchmarkCommand(abc.ABC):
  """Abstracts a benchmark command."""

  def __init__(self,
               benchmark_binary: str,
               model_name: str,
               num_threads: int,
               num_runs: int,
               driver: Optional[str] = None,
               taskset: Optional[str] = None):
    self.benchmark_binary = benchmark_binary
    self.model_name = model_name
    self.taskset = taskset
    self.num_threads = num_threads
    self.num_runs = num_runs
    self.driver = driver
    self.args = []

  @property
  @abc.abstractmethod
  def runtime(self):
    pass

  @abc.abstractmethod
  def parse_latency_from_output(self, output: str) -> float:
    pass

  def generate_benchmark_command(self) -> list[str]:
    """Returns a list of strings that correspond to the command to be run."""
    command = []
    if self.taskset:
      command.append("taskset")
      command.append(str(self.taskset))
    command.append(self.benchmark_binary)
    command.extend(self.args)
    return command


class TFLiteBenchmarkCommand(BenchmarkCommand):
  """Represents a TFLite benchmark command."""

  def __init__(self,
               benchmark_binary: str,
               model_name: str,
               model_path: str,
               num_threads: int,
               num_runs: int,
               taskset: Optional[str] = None):
    super().__init__(benchmark_binary,
                     model_name,
                     num_threads,
                     num_runs,
                     taskset=taskset)
    self.args.append("--graph=" + model_path)
    self._latency_large_regex = re.compile(
        r".*?Inference \(avg\): (\d+.?\d*e\+?\d*).*")
    self._latency_regex = re.compile(r".*?Inference \(avg\): (\d+).*")

  @property
  def runtime(self):
    return "tflite"

  def parse_latency_from_output(self, output: str) -> float:
    # First match whether a large number has been recorded e.g. 1.18859e+06.
    matches = self._latency_large_regex.search(output)
    if not matches:
      # Otherwise, regular number e.g. 71495.6.
      matches = self._latency_regex.search(output)

    latency_ms = 0
    if matches:
      latency_ms = float(matches.group(1)) / 1000
    else:
      print("Warning! Could not parse latency. Defaulting to 0ms.")
    return latency_ms

  def generate_benchmark_command(self) -> list[str]:
    command = super().generate_benchmark_command()
    if self.driver == "gpu":
      command.append("--use_gpu=true")
    command.append("--num_threads=" + str(self.num_threads))
    command.append("--num_runs=" + str(self.num_runs))
    return command


class IreeBenchmarkCommand(BenchmarkCommand):
  """Represents an IREE benchmark command."""

  def __init__(self,
               benchmark_binary: str,
               model_name: str,
               model_path: str,
               num_threads: int,
               num_runs: int,
               taskset: Optional[str] = None):
    super().__init__(benchmark_binary,
                     model_name,
                     num_threads,
                     num_runs,
                     taskset=taskset)
    self.args.append("--module=" + model_path)
    self._latency_regex = re.compile(
        r".*?BM_main/process_time/real_time_mean\s+(.*?) ms.*")

  @property
  def runtime(self):
    return "iree"

  def parse_latency_from_output(self, output: str) -> float:
    matches = self._latency_regex.search(output)
    latency_ms = 0
    if matches:
      latency_ms = float(matches.group(1))
    else:
      print("Warning! Could not parse latency. Defaulting to 0ms.")
    return latency_ms

  def generate_benchmark_command(self) -> list[str]:
    command = super().generate_benchmark_command()
    command.append("--device=" + self.driver)
    command.append("--task_topology_max_group_count=" + str(self.num_threads))
    command.append("--benchmark_repetitions=" + str(self.num_runs))
    return command
