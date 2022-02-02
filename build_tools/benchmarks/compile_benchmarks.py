#!/usr/bin/env python3
# Copyright 2021 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Script for getting compilation statistics.

This file provides utility to compile benchmarks and gather static compilation utilities.
Allows either compiling a simple model or all the models in the benchmark_suite.

Examples usages

  # Compiling a single model.
  python3 compile_benchmarks.py \
    --model MobileBert
    --mlir $HOME/tmp/mobilebert.mlir
    --flagfile $HOME/tmp/mobilebert.compilation_flagfile
    --config_name cpu_ir_dump
    --mode print
    --tmp_dir $HOME/tmp/cpu/
    $HOME/iree/build


  # Compiling all the models in the benchmark suite.
  python3 compile_benchmarks.py \
    --tmp_dir $HOME/tmp/iree_benchmarks
    --mode stats
    $HOME/iree/build
"""

import argparse
import os
from statistics import mode
from pathlib import Path

from typing import Dict, List, Optional, Set, Tuple

from common.benchmark_definition import (execute_cmd,
                                         execute_cmd_and_get_output_and_error)

# All benchmarks' relative path against root build directory.
BENCHMARK_SUITE_REL_PATH = "benchmark_suites"
VMFB_REL_PATH = "vmfb"

# The compilation flagfiles name
MODEL_COMPILATION_FLAGFILE_NAME = "compilation_flagfile"

# Dictionary from Model directory name to the .mlir file
MODEL_TO_MLIR = {
    "DeepLabV3-fp32": "deeplabv3.tflite.mlir",
    "MobileBertSquad-fp32": "mobilebertsquad.tflite.mlir",
    "MobileNetV2-fp32,imagenet": "mobilenet_v2_1.0_224.tflite.mlir",
    "MobileNetV3Small-fp32,imagenet": "MobileNetV3SmallStaticBatch.tflite.mlir",
    "MobileSSD-fp32": "mobile_ssd_v2_float_coco.tflite.mlir",
    "PoseNet-fp32": "mobile_ssd_v2_float_coco.tflite.mlir"
}


class BenchmarkCompilationInfo:
  """An object to describe the compilation modes for a benchmark.

  It has the following fields
  - model_name : Name of the model being compiled.
  - mlir_file : Path to the mlir file to compile.
  - compilation_modes : Dictionary where each entry has a list of flags used
    to compile the model. The key for the entry is a name for the configuration.
  """
  model_name: str
  mlir_file: str
  compilation_modes: Dict[str, List[str]]

  def __repr__(self):
    s = f"Model='{self.model_name}'\n"
    s += f"\tInput='{self.mlir_file}'\n"
    for config_name, flags in self.compilation_modes.items():
      s += f"\t\tConfig='{config_name}, Flags={flags}\n"
    return s

  def __init__(self, name: str, file: str):
    self.model_name = name
    self.mlir_file = file
    self.compilation_modes = {}

  # Add list of flags for compiling a model and associate a name with it.
  def add_compilation(self, name: str, flags: List[str]):
    self.compilation_modes[name] = flags


def parse_flag_file(flagfile: str) -> List[str]:
  """Parses a flag file.

  Given path to the flagfile reads the compilation flags from it. Expects
  each flag to be in a separate line.
  """
  compilation_flags = []
  with open(flagfile, "r") as fp:
    for line in fp.readlines():
      compilation_flags += [line.strip()]
  return compilation_flags


def get_single_benchmark_compilation_info(
    model_name: str, mlir_file: str, config_name: str,
    flagfile: str) -> List[BenchmarkCompilationInfo]:
  """Gets the compilation info for a single benchmark.

  For the model specified in the mlir_file, parses the flag file to build the
  compilation info for the benchmark
  """
  model_compilation_info = BenchmarkCompilationInfo(model_name, mlir_file)
  compilation_flags = parse_flag_file(flagfile)
  model_compilation_info.add_compilation(config_name, compilation_flags)
  return [model_compilation_info]


def get_all_benchmarks_compilation_info(
    root_build_dir: str,
    verbose: bool = False) -> List[BenchmarkCompilationInfo]:
  """Get the compilation info for the benchmark suite.

  Builds the list of configurations to compile based on the benchmark_suite.
  Expects the benchmark_suites directory to have this structure
  <root-build-dir>/benchmark_suites
  └── <benchmark-category> (e.g., TFLite)
      ├── <benchmark-suite> (e.g., MobileBertSquad-fp32)
      │   ├── <benchmark-case> (e.g., iree-vulkan__GPU-Mali-Valhall__kernel-execution)
      │   │   └── flagfile
      │   ├── ...
      │   │   └── flagfile
      │   └── <benchmark_case>
      │       └── flagfile
      └── vmfb
  """
  root_benchmark_dir = os.path.join(root_build_dir, BENCHMARK_SUITE_REL_PATH)

  benchmarks = []
  # Go over all the benchmarks and collect the compilation flagfiles.
  for directory in sorted(os.listdir(root_benchmark_dir)):
    benchmark_category_dir = os.path.join(root_benchmark_dir, directory)

    for model, mlir_file in MODEL_TO_MLIR.items():
      if model == VMFB_REL_PATH:
        continue

      # Check if MLIR file is known.
      if model not in MODEL_TO_MLIR:
        continue

      model_dir = os.path.join(benchmark_category_dir, model)
      mlir_file = os.path.join(benchmark_category_dir, MODEL_TO_MLIR[model])
      if verbose:
        print(f"Found model : {model}, mlir file : {mlir_file}")

      model_compilation_info = BenchmarkCompilationInfo(model.replace(",", "_"),
                                                        mlir_file)
      for config in next(os.walk(model_dir))[1]:
        if verbose:
          print(f"Compilation Config : {config}")
        flagfile = os.path.join(model_dir, config,
                                MODEL_COMPILATION_FLAGFILE_NAME)
        if verbose:
          print(f"Compilation flag file : {flagfile}")

        compilation_flags = parse_flag_file(flagfile)
        model_compilation_info.add_compilation(config.replace(",", "_"),
                                               compilation_flags)

      benchmarks += [model_compilation_info]

  return benchmarks


def compile_benchmarks(benchmarks: List[BenchmarkCompilationInfo],
                       root_build_dir: str,
                       tmp_dir: str,
                       mode: str,
                       extra_flags: List[str],
                       verbose: bool = False):
  """Filters and compiles benchmakrs in all categories

  Given a list of benchmarks with their compilation modes, compile the model
  and generate artifacts based on the mode specified. For example
  - mode == stats generates statistics
  - mode == print collects the dump of IR after all.
  The artifacts are placed in the tmp_dir.
  """
  translate_tool = os.path.join(root_build_dir, "iree", "tools",
                                "iree-translate")

  for benchmark in benchmarks:
    for config_name, flags in benchmark.compilation_modes.items():
      compilation_flags = [translate_tool] + flags
      prefix = benchmark.model_name + config_name

      if mode == "stats":
        # Add flags to collect statistics
        statistics_flag = ["-iree-scheduling-dump-statistics-format=csv"]
        stats_file_name = prefix + ".csv"
        stats_file_path = os.path.join(tmp_dir, stats_file_name)
        compilation_flags += statistics_flag
        compilation_flags += [
            "-iree-scheduling-dump-statistics-file=" + stats_file_path
        ]
      elif mode == "print":
        #Add flags to dump IR after all
        print_flags = [
            "-print-ir-after-all", "-mlir-disable-threading",
            "-mlir-elide-elementsattrs-if-larger=16"
        ]
        compilation_flags += print_flags

      # Get output as MLIR file
      compilation_flags += ["-iree-vm-bytecode-module-output-format=mlir-text"]

      # Extra flags passed in
      compilation_flags += extra_flags

      # Specify the mlir file
      compilation_flags += [benchmark.mlir_file]

      # Compile the model
      pipes = execute_cmd_and_get_output_and_error(compilation_flags, verbose)

      # Write the output
      if pipes.stdout:
        out_file_name = prefix + ".mlir"
        out_file_path = os.path.join(tmp_dir, out_file_name)
        if (verbose):
          print(f"Output file : {out_file_path}")
        output_file = open(out_file_path, "w")
        output_file.write(pipes.stdout.decode("utf-8"))
        output_file.close()

      # Write the error
      if pipes.stderr :
        err_file_name = prefix + ".err"
        err_file_path = os.path.join(tmp_dir, err_file_name)
        if (verbose):
          print(f"Error file : {err_file_path}")
        error_file = open(err_file_path, "w")
        error_file.write(pipes.stderr.decode("utf-8"))
        error_file.close()

  return


def parse_arguments():
  """Parses command-line options."""

  def check_dir_path(path):
    if os.path.isdir(path):
      return path
    else:
      raise argparse.ArgumentTypeError(path)

  def check_exe_path(path):
    if os.access(path, os.X_OK):
      return path
    else:
      raise argparse.ArgumentTypeError(f"'{path}' is not an executable")

  parser = argparse.ArgumentParser()
  parser.add_argument(
      "build_dir",
      metavar="<build-dir>",
      type=check_dir_path,
      help="Path to the build directory containing benchmark suites")
  parser.add_argument(
      "--model",
      default="",
      help="Name of the model being compiled. If left unspecified all the models"
      " from the iree benchmark_suites are compiled.")
  parser.add_argument(
      "--mlir",
      type=Path,
      help="Path to mlir file to compiler. Required when --model is specified, "
      "ignored when it is not")
  parser.add_argument(
      "--flagfile",
      type=Path,
      help="Path to flag file used for compiling a model. The flag file is"
      " expected to have one flag per line. Required when --model is specified,"
      " ignored if it isnt")
  parser.add_argument(
      "--config_name",
      default="",
      help="Optional name for the configuration in which the model is compiled. "
      "Used only if model is specified")
  parser.add_argument(
      "--mode",
      help="Mode in which the compilation is done. Valid values are stats, print"
  )
  parser.add_argument(
      "--print-before",
      nargs="+",
      help=
      "List of passes before which IR is to be printed. Only valid IREE pass names are accepted"
  )
  parser.add_argument(
      "--print-after",
      nargs="+",
      help=
      "List of passes after which IR is to be printed. Only valid IREE pass names are accepted"
  )
  parser.add_argument(
      "--tmp_dir",
      "--tmp-dir",
      "--tmpdir",
      default="/tmp/iree-benchmarks",
      help="Base directory in which to store temporary files. A subdirectory"
      " with a name matching the git commit hash will be created.")
  parser.add_argument("--verbose",
                      action="store_true",
                      help="Print internal information during execution")

  args = parser.parse_args()

  return args


def main(args):
  # Create temporary directory to put the statistics in.
  if (args.verbose):
    print(args.tmp_dir)
  os.makedirs(args.tmp_dir, exist_ok=True)

  if args.model != "":
    benchmarks = get_single_benchmark_compilation_info(args.model,
                                                       str(args.mlir),
                                                       args.config_name,
                                                       str(args.flagfile))
  else:
    benchmarks = get_all_benchmarks_compilation_info(args.build_dir,
                                                     args.verbose)

  # Some arguments are processed before hand.
  extra_flags = []
  if args.print_before or args.print_after:
    extra_flags += ["-mlir-elide-elementsattrs-if-larger=16"]
    if args.print_before:
      for pass_name in args.print_before:
        extra_flags += [f"-print-ir-before={pass_name}"]
    if args.print_after:
      for pass_name in args.print_after:
        extra_flags += [f"-print-ir-after={pass_name}"]

  compile_benchmarks(benchmarks=benchmarks,
                     root_build_dir=args.build_dir,
                     tmp_dir=args.tmp_dir,
                     mode=args.mode,
                     extra_flags=extra_flags,
                     verbose=args.verbose)


if __name__ == "__main__":
  main(parse_arguments())
