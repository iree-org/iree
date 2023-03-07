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


def dump_flags_of_generation_config(
    module_generation_config: iree_definitions.ModuleGenerationConfig,
    root_path: pathlib.PurePath = pathlib.PurePath()):

  imported_model = module_generation_config.imported_model
  imported_model_path = iree_artifacts.get_imported_model_path(
      imported_model=imported_model, root_path=root_path)
  source_model_path = model_artifacts.get_model_path(model=imported_model.model,
                                                     root_path=root_path)

  compile_flags = module_generation_config.materialize_compile_flags() + [
      str(imported_model_path)
  ]
  import_flags = imported_model.import_config.materialize_import_flags(
      model=imported_model.model) + [str(source_model_path)]
  # TODO(#12215): Include benchmark name to make them searchable by keywords.
  return {
      "composite_id": module_generation_config.composite_id,
      "compile_flags": compile_flags,
      "import_tool": imported_model.import_config.tool.value,
      "import_flags": import_flags
  }


def dump_flags_from_run_config(
    e2e_model_run_config: iree_definitions.E2EModelRunConfig,
    root_path: pathlib.PurePath = pathlib.PurePath()):

  gen_config = e2e_model_run_config.module_generation_config
  module_path = iree_artifacts.get_module_dir_path(
      module_generation_config=gen_config,
      root_path=root_path) / iree_artifacts.MODULE_FILENAME

  run_flags = e2e_model_run_config.materialize_run_flags() + [
      f"--module={module_path}"
  ]
  # TODO(#12215): Include benchmark name to make them searchable by keywords.
  return {
      "composite_id":
          e2e_model_run_config.composite_id,
      "run_flags":
          run_flags,
      "module_generation_config":
          dump_flags_of_generation_config(module_generation_config=gen_config,
                                          root_path=root_path)
  }


def _dump_flags_handler(args: argparse.Namespace):
  dump_configs = {}
  root_path = args.e2e_test_artifacts_dir
  if args.execution_benchmark_config is not None:
    benchmark_groups = json.loads(args.execution_benchmark_config.read_text())
    for target_device, benchmark_group in benchmark_groups.items():
      run_configs = serialization.unpack_and_deserialize(
          data=benchmark_group["run_configs"],
          root_type=List[iree_definitions.E2EModelRunConfig])
      dump_configs[target_device] = dict(
          (run_config.composite_id,
           dump_flags_from_run_config(e2e_model_run_config=run_config,
                                      root_path=root_path))
          for run_config in run_configs)

  elif args.compilation_benchmark_config is not None:
    benchmark_config = json.loads(args.compilation_benchmark_config.read_text())
    gen_configs = serialization.unpack_and_deserialize(
        data=benchmark_config["generation_configs"],
        root_type=List[iree_definitions.ModuleGenerationConfig])
    dump_configs = dict(
        (gen_config.composite_id,
         dump_flags_of_generation_config(module_generation_config=gen_config,
                                         root_path=root_path))
        for gen_config in gen_configs)

  else:
    raise AssertionError("No benchmark config is set.")

  print(json.dumps(dump_configs, indent=2))


def _parse_arguments() -> argparse.Namespace:
  parser = argparse.ArgumentParser(
      description=
      "Miscellaneous tool to help work with benchmark suite and benchmark CI.")

  subparser = parser.add_subparsers(required=True, title="operation")
  dump_flags_parser = subparser.add_parser(
      "dump-flags",
      help="Dump the flags to compile and run benchmarks manually.")
  dump_flags_parser.add_argument(
      "--e2e_test_artifacts_dir",
      type=pathlib.PurePath,
      default=pathlib.Path(),
      help="E2E test artifacts root path used in the outputs of artifact paths")
  dump_flags_input_parser = dump_flags_parser.add_mutually_exclusive_group(
      required=True)
  dump_flags_input_parser.add_argument(
      "--execution_benchmark_config",
      type=pathlib.Path,
      help="Config file exported from export_benchmark_config.py execution")
  dump_flags_input_parser.add_argument(
      "--compilation_benchmark_config",
      type=pathlib.Path,
      help="Config file exported from export_benchmark_config.py compilation")
  dump_flags_parser.set_defaults(handler=_dump_flags_handler)

  return parser.parse_args()


def main(args: argparse.Namespace):
  args.handler(args)


if __name__ == "__main__":
  main(_parse_arguments())
