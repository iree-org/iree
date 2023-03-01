#!/usr/bin/env python3
# Copyright 2023 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Miscellaneous tool to help work with benchmark suite and benchmark CI."""

import sys
import pathlib

# Add build_tools python dir to the search path.
sys.path.insert(0, str(pathlib.Path(__file__).parent.with_name("python")))

import argparse
import dataclasses
import json
from typing import Any, List, Type, Callable

from benchmark_suites.iree import export_definitions
from e2e_test_framework import serialization
from e2e_test_framework.definitions import iree_definitions


def _filter_and_serialize_into_plain_object(data: Any, root_type: Type,
                                            filter: Callable[[Any], bool]):
  # Re-pack into JSON serializable format but don't use references.
  configs = serialization.unpack_and_deserialize(data=data, root_type=root_type)
  configs = [config for config in configs if filter(config)]
  return serialization.serialize_and_pack(obj=configs, use_ref=False)


def _dump_config_handler(args: argparse.Namespace):
  benchmark_id = args.benchmark_id
  if benchmark_id is None:
    filter = lambda _: True
  else:
    # Run and generation config have the same id field.
    filter = lambda config: config.composite_id == benchmark_id

  if args.execution_config is not None:
    raw_data = json.loads(args.execution_config.read_text())
    exported_config = export_definitions.ExecutionBenchmarkConfig(**raw_data)
    plain_run_configs = _filter_and_serialize_into_plain_object(
        data=exported_config.run_configs,
        root_type=List[iree_definitions.E2EModelRunConfig],
        filter=filter)
    plain_exported_config = dataclasses.replace(exported_config,
                                                run_configs=plain_run_configs)

  elif args.compilation_config is not None:
    raw_data = json.loads(args.compilation_config.read_text())
    exported_config = export_definitions.CompilationBenchmarkConfig(**raw_data)
    plain_generation_configs = _filter_and_serialize_into_plain_object(
        data=exported_config.generation_configs,
        root_type=List[iree_definitions.ModuleGenerationConfig],
        filter=filter)
    plain_exported_config = dataclasses.replace(
        exported_config, generation_configs=plain_generation_configs)

  else:
    raise AssertionError("No config file is specified.")

  # Config is packed back into their original format so it can be still feed
  # into the benchmark tools.
  print(json.dumps(dataclasses.asdict(plain_exported_config), indent=2))


def _parse_arguments() -> argparse.Namespace:
  parser = argparse.ArgumentParser(
      description=
      "Miscellaneous tool to work with benchmark suite and benchmark CI.")

  subparser = parser.add_subparsers(required=True, title="operation")
  dump_config_parser = subparser.add_parser(
      "dump-config",
      help="Dump the serialized benchmark config into plain JSON format.")
  dump_config_parser.add_argument(
      "--benchmark_id",
      type=str,
      help="Only dump the config for the specified benchmark ID.")
  dump_config_input_parser = dump_config_parser.add_mutually_exclusive_group(
      required=True)
  dump_config_input_parser.add_argument(
      "--execution_config",
      type=pathlib.Path,
      help="Config file exported from export_benchmark_config.py execution")
  dump_config_input_parser.add_argument(
      "--compilation_config",
      type=pathlib.Path,
      help="Config file exported from export_benchmark_config.py compilation")
  dump_config_parser.set_defaults(handler=_dump_config_handler)

  return parser.parse_args()


def main(args: argparse.Namespace):
  args.handler(args)


if __name__ == "__main__":
  main(_parse_arguments())
