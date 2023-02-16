#!/usr/bin/env python3
## Copyright 2023 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Tool for users to work with benchmark CI."""

import sys
import pathlib

# Add build_tools python dir to the search path.
sys.path.insert(0, str(pathlib.Path(__file__).parent.with_name("python")))

import argparse
import pprint
from typing import Any, Dict, List, Optional
import re
import requests

from e2e_test_artifacts import iree_artifacts
from e2e_test_artifacts.cmake_generator import iree_rule_generator
from e2e_test_framework.definitions import iree_definitions
from e2e_model_tests import run_module_utils
from e2e_test_framework import serialization

GCS_HTTPS_URL_PREFIX = "https://storage.googleapis.com/"
GCS_E2E_TEST_ARTIFACTS_DIR = "e2e-test-artifacts"
LOCAL_E2E_TEST_ARTIFACTS_DIR = "e2e_test_artifacts"
BENCHMARK_CONFIG_FILENAME = "benchmark-config.json"
COMPILE_CONFIG_FILENAME = "compilation-config.json"


def _fetch_artifacts(gcs_url: str,
                     gen_configs: List[iree_definitions.ModuleGenerationConfig],
                     root_dir: pathlib.Path):
  artifacts_dir = root_dir / LOCAL_E2E_TEST_ARTIFACTS_DIR
  artifacts_dir.mkdir(exist_ok=True)
  fetched_urls = set()
  for gen_config in gen_configs:
    imported_model_path = iree_artifacts.get_imported_model_path(
        gen_config.imported_model)
    imported_model_url = f"{gcs_url}/{GCS_E2E_TEST_ARTIFACTS_DIR}/{imported_model_path}"
    if imported_model_url not in fetched_urls:
      print(f"Downloading imported model: {imported_model_url}")
      resp = requests.get(imported_model_url)
      (artifacts_dir / imported_model_path).write_bytes(resp.content)
      fetched_urls.add(imported_model_url)

    module_dir_path = iree_artifacts.get_module_dir_path(gen_config)
    (artifacts_dir / module_dir_path).mkdir(exist_ok=True)

    module_path = module_dir_path / iree_artifacts.MODULE_FILENAME
    module_url = f"{gcs_url}/{GCS_E2E_TEST_ARTIFACTS_DIR}/{module_path}"
    if module_url not in fetched_urls:
      print(f"Downloading module: {module_url}")
      resp = requests.get(module_url)
      (artifacts_dir / module_path).write_bytes(resp.content)
      fetched_urls.add(module_url)


def _print_gen_config(gen_config: iree_definitions.ModuleGenerationConfig,
                      root_dir: pathlib.Path):
  model = gen_config.imported_model.model
  print("Model info:")
  pprint.pprint(model)
  print("")

  builder = iree_rule_generator.IreeRuleBuilder(package_name="")
  compile_config = gen_config.compile_config
  compile_flags = builder._generate_compile_flags(
      compile_config=compile_config,
      mlir_dialect_type=gen_config.imported_model.import_config.dialect_type.
      value) + compile_config.extra_flags
  compile_flags.append(
      str(
          iree_artifacts.get_imported_model_path(gen_config.imported_model,
                                                 root_path=root_dir /
                                                 LOCAL_E2E_TEST_ARTIFACTS_DIR)))
  print("Compile flags:")
  print(" ".join(compile_flags) + "\n")


def _explain_handler(gcs_url: str, root_dir: pathlib.Path,
                     benchmark_config: Dict[str, Any],
                     compile_config: Dict[str, Any], args: argparse.Namespace):
  benchmark_id_filter = args.benchmark_id

  gen_configs = serialization.unpack_and_deserialize(
      data=compile_config["generation_configs"],
      root_type=List[iree_definitions.ModuleGenerationConfig])
  matched_gen_configs = []
  dep_gen_configs = {}
  for gen_config in gen_configs:
    gen_config_id = gen_config.composite_id()
    if gen_config_id != benchmark_id_filter:
      continue
    matched_gen_configs.append(gen_config)
    dep_gen_configs[gen_config_id] = gen_config

  matched_run_configs = []
  for _, data in benchmark_config.items():
    run_configs = serialization.unpack_and_deserialize(
        data["run_configs"], root_type=List[iree_definitions.E2EModelRunConfig])
    for run_config in run_configs:
      run_config_id = run_config.composite_id()
      if run_config_id != benchmark_id_filter:
        continue
      matched_run_configs.append(run_config)
      gen_config = run_config.module_generation_config
      dep_gen_configs[gen_config.composite_id()] = gen_config

  _fetch_artifacts(gcs_url=gcs_url,
                   gen_configs=list(dep_gen_configs.values()),
                   root_dir=root_dir)

  for gen_config in matched_gen_configs:
    print(
        f"\n--------\nExplaining compilation benchmark: {gen_config.composite_id()}\n"
    )
    _print_gen_config(gen_config, root_dir)

  for run_config in matched_run_configs:
    print(
        f"\n--------\nExplaining execution benchmark: {run_config.composite_id()}\n"
    )

    _print_gen_config(run_config.module_generation_config, root_dir)

    print("Run flags:")
    module_dir_path = iree_artifacts.get_module_dir_path(
        run_config.module_generation_config)
    run_flags = [
        f"--module={root_dir / LOCAL_E2E_TEST_ARTIFACTS_DIR / module_dir_path / iree_artifacts.MODULE_FILENAME}"
    ]
    run_flags += run_module_utils.build_run_flags_for_model(
        model=run_config.module_generation_config.imported_model.model,
        model_input_data=run_config.input_data)
    run_flags += run_module_utils.build_run_flags_for_execution_config(
        run_config.module_execution_config)
    print(" ".join(run_flags) + "\n")

    print("Target device:")
    pprint.pprint(run_config.target_device_spec)


def parse_arguments() -> argparse.Namespace:
  parser = argparse.ArgumentParser()

  # Makes global options come *after* command.
  # See https://stackoverflow.com/q/23296695
  subparser_base = argparse.ArgumentParser(add_help=False)
  subparser_base.add_argument("--root_dir", type=pathlib.Path)

  subparser = parser.add_subparsers(required=True, title="export type")
  explain_parser = subparser.add_parser("explain",
                                        parents=[subparser_base],
                                        help="Fetch and explain a benchmark.")
  explain_parser.set_defaults(handler=_explain_handler)
  explain_parser.add_argument("--gcs_url", type=str, required=True)
  explain_parser.add_argument("--benchmark_id",
                              type=str,
                              help="Benchmark ID.",
                              required=True)

  return parser.parse_args()


def main(args):
  match_results = re.findall(r"^gs://(.+)$", args.gcs_url)
  if len(match_results) == 0:
    raise ValueError("Invalid GCS URL.")
  gcs_path = match_results[0].strip("/")
  gcs_url = GCS_HTTPS_URL_PREFIX + gcs_path

  if args.root_dir is not None:
    root_dir = args.root_dir
  else:
    root_dir = pathlib.Path(gcs_path.replace("/", "-"))

  root_dir.mkdir(exist_ok=True)

  resp = requests.get(f"{gcs_url}/{BENCHMARK_CONFIG_FILENAME}")
  (root_dir / BENCHMARK_CONFIG_FILENAME).write_text(resp.text)
  benchmark_config = resp.json()

  resp = requests.get(f"{gcs_url}/{COMPILE_CONFIG_FILENAME}")
  (root_dir / COMPILE_CONFIG_FILENAME).write_text(resp.text)
  compile_config = resp.json()

  args.handler(gcs_url, root_dir, benchmark_config, compile_config, args)


if __name__ == "__main__":
  main(parse_arguments())
