#!/usr/bin/env python3
# Copyright 2023 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Miscellaneous tool to help work with benchmark suite and benchmark CI."""

import pathlib
import sys

# Add build_tools python dir to the search path.
sys.path.insert(0, str(pathlib.Path(__file__).parent.with_name("python")))

import argparse
import json
from typing import List

from e2e_test_framework import serialization
from e2e_test_artifacts import model_artifacts, iree_artifacts
from e2e_test_framework.definitions import iree_definitions

IREE_COMPILER_NAME = "iree-compile"


def _dump_cmds_of_generation_config(
    gen_config: iree_definitions.ModuleGenerationConfig,
    root_path: pathlib.PurePath = pathlib.PurePath()):

  imported_model = gen_config.imported_model
  imported_model_path = iree_artifacts.get_imported_model_path(
      imported_model=imported_model, root_path=root_path)
  module_path = iree_artifacts.get_module_dir_path(
      module_generation_config=gen_config,
      root_path=root_path) / iree_artifacts.MODULE_FILENAME
  compile_cmds = [
      IREE_COMPILER_NAME,
      str(imported_model_path), "-o",
      str(module_path)
  ]
  compile_cmds += gen_config.materialize_compile_flags()

  if imported_model.import_config.tool == iree_definitions.ImportTool.NONE:
    import_cmds = ["(Source model is already in MLIR)"]
  else:
    source_model_path = model_artifacts.get_model_path(
        model=imported_model.model, root_path=root_path)
    import_cmds = [
        imported_model.import_config.tool.value,
        str(source_model_path), "-o",
        str(imported_model_path)
    ]
    import_cmds += imported_model.import_config.materialize_import_flags(
        model=imported_model.model)

  # TODO(#12215): Print benchmark name to make them searchable by keywords.
  # Insert a blank line after each command to help read with line wrap.
  return [
      f'Compile Module: {" ".join(compile_cmds)}', "",
      f'Import Model: {" ".join(import_cmds)}', ""
  ]


def _dump_cmds_from_run_config(
    run_config: iree_definitions.E2EModelRunConfig,
    root_path: pathlib.PurePath = pathlib.PurePath()):

  gen_config = run_config.module_generation_config
  module_path = iree_artifacts.get_module_dir_path(
      module_generation_config=gen_config,
      root_path=root_path) / iree_artifacts.MODULE_FILENAME

  run_cmds = [run_config.tool.value, f"--module={module_path}"]
  run_cmds += run_config.materialize_run_flags()
  # TODO(#12215): Include benchmark name to make them searchable by keywords.
  # Insert a blank line after the command to help read with line wrap.
  lines = [f'Run Module: {" ".join(run_cmds)}', ""]
  lines += _dump_cmds_of_generation_config(gen_config=gen_config,
                                           root_path=root_path)
  return lines


def _dump_cmds_handler(args: argparse.Namespace):
  root_path = args.e2e_test_artifacts_dir
  benchmark_id = args.benchmark_id
  lines = []

  if args.execution_benchmark_config is not None:
    benchmark_groups = json.loads(args.execution_benchmark_config.read_text())
    for target_device, benchmark_group in benchmark_groups.items():
      run_configs = serialization.unpack_and_deserialize(
          data=benchmark_group["run_configs"],
          root_type=List[iree_definitions.E2EModelRunConfig])
      for run_config in run_configs:
        if benchmark_id is not None and benchmark_id != run_config.composite_id:
          continue

        lines.append("################")
        lines.append("")
        lines.append(f"Execution Benchmark ID: {run_config.composite_id}")
        lines.append(f"Target Device: {target_device}")
        lines.append("")
        lines += _dump_cmds_from_run_config(run_config=run_config,
                                            root_path=root_path)

  if args.compilation_benchmark_config is not None:
    benchmark_config = json.loads(args.compilation_benchmark_config.read_text())
    gen_configs = serialization.unpack_and_deserialize(
        data=benchmark_config["generation_configs"],
        root_type=List[iree_definitions.ModuleGenerationConfig])
    for gen_config in gen_configs:
      if benchmark_id is not None and benchmark_id != gen_config.composite_id:
        continue

      lines.append("################")
      lines.append("")
      lines.append(f"Compilation Benchmark ID: {gen_config.composite_id}")
      lines.append("")
      lines += _dump_cmds_of_generation_config(gen_config=gen_config,
                                               root_path=root_path)

  print(*lines, sep="\n")


def _parse_arguments() -> argparse.Namespace:
  parser = argparse.ArgumentParser(
      description=
      "Miscellaneous tool to help work with benchmark suite and benchmark CI.")

  subparser = parser.add_subparsers(required=True, title="operation")
  dump_cmds_parser = subparser.add_parser(
      "dump-cmds",
      help="Dump the commands to compile and run benchmarks manually.")
  dump_cmds_parser.add_argument(
      "--e2e_test_artifacts_dir",
      type=pathlib.PurePath,
      default=pathlib.Path(),
      help="E2E test artifacts root path used in the outputs of artifact paths")
  dump_cmds_parser.add_argument("--benchmark_id",
                                type=str,
                                help="Only dump the benchmark with this id")
  dump_cmds_parser.add_argument(
      "--execution_benchmark_config",
      type=pathlib.Path,
      help="Config file exported from export_benchmark_config.py execution")
  dump_cmds_parser.add_argument(
      "--compilation_benchmark_config",
      type=pathlib.Path,
      help="Config file exported from export_benchmark_config.py compilation")
  dump_cmds_parser.set_defaults(handler=_dump_cmds_handler)

  args = parser.parse_args()
  if (args.execution_benchmark_config is None and
      args.compilation_benchmark_config is None):
    parser.error("At least one of --execution_benchmark_config or "
                 "--compilation_benchmark_config must be set.")

  return args


def main(args: argparse.Namespace):
  args.handler(args)


if __name__ == "__main__":
  main(_parse_arguments())
