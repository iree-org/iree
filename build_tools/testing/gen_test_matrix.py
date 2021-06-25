#!/usr/bin/env python
# Copyright 2021 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Generates runners for a matrix of test configurations.

This tool is used for sources under test which should be run in a matrix of
different real-world configurations. It is aimed at integration tests which
typically need a greater degree of flexibility and less "framework-ness"
compared to (for example), lit-tests or more classic unit tests.

As with all modern things, it starts with directories that contain
test_matrix.yaml files, which describe tests.

Conceptually, the test_matrix.yaml file declares a list of test_groups
at the top-level. Each test group has an interpolated "id" which is unique
in the scope of all tests. Example:

  test_groups:
    - id: "math_real_{backend}_vs_{ref_backend}_{function}_dyndims_{dyndims}"

Interpolation is done relative to placeholders declared in a "matrix", where
one concrete test is instantiated for the cartesian product of all combinations
in the matrix. Here is an example matrix:

  matrix:
    backend: *BACKENDS
    ref_backend: *REF_BACKENDS
    function: *MATH_FUNCTIONS
    dyndims: ["true", "false"]

Each key in the matrix expands to a list (or list of lists that are flattened)
of string values. Note in the above that we have used YAML references to
include fragments defined elsewhere in the file.

Each concrete test is then generated according to one or more runners, which
are responsible for creating actual generated test sources. Example:

  runner:
    - type: tfhost
      main: math_test.py
      args:
        - "--functions={function}"
        - "--target_backends={backend}"
        - "--reference_backend={ref_backend}"
        - "--dynamic_dims={dyndims}"

The type of runner is a hard-coded feature of this script. See the
implementations at the end.

By default, every expanded combination is included in the test suite and is
expected to pass. This can be changed by including 'xfail' and/or 'xpass'
lists. A test will be marked expected-failing if it matches any of the
predicates in xfail and none of the predicates in xpass.

Example:
  xfail:
    # Vulkan is currently structurally broken with dyndims.
    - matrix.backend: iree_vulkan
      matrix.dyndims: "true"
  xpass:
    # Some dynamic dim functions pass on vulkan regardless of overall
    # broken support.
    - matrix.backend: iree_vulkan
      matrix.function: *VULKAN_DYNDIM_PASS_MATH_FUNCTIONS
      matrix.dyndims: "true"

Note that each entry in xfail/xpass is a mapping of key/value pairs. Any keys
that start with 'matrix.' are taken to match an expanded matrix placeholder.
A mapping predicate is evaluated as a product-of-sums where the overall
predicate is true if all of the listed keys have a match for any of their
values.

TODO: Add support for skipping combinations.
TODO: Add support for annotating combinations with 'requires' for env feature
  checking.
"""

from typing import Set, Tuple

import argparse
from contextlib import contextmanager
import os
import shutil

try:
  import yaml
except ModuleNotFoundError as e:
  raise RuntimeError(
      f"PyYAML is not installed. Typically: 'python -m pip install PyYAML"
  ) from e

################################################################################
# Base classes and types
################################################################################


class Environment:
  """Runtime environment for processing a directory."""

  def __init__(self, args, root_dir: str, output_dir: str):
    self.args = args
    self.root_dir = root_dir
    self.output_dir = output_dir
    # Set of directories containing purely generated files.
    self.gen_dirs = set()  # type: Set[str]
    # Set of (gen_dir, file_name) for all files in a given directory that have
    # been generated.
    self.gen_files = set()  # type: Set[Tuple[str, str]]

  def remember_gen_file(self, gen_file_path: str):
    gen_dir = os.path.dirname(gen_file_path)
    gen_file = os.path.basename(gen_file_path)
    self.gen_dirs.add(gen_dir)
    self.gen_files.add((gen_dir, gen_file))

  def prune_gen_files(self):
    found_gen_files = set()
    for gen_dir in self.gen_dirs:
      dir_listing = os.listdir(gen_dir)
      for fname in dir_listing:
        found_gen_files.add((gen_dir, fname))
    obsolete_gen_files = found_gen_files - self.gen_files
    if obsolete_gen_files:
      for gen_dir, fname in obsolete_gen_files:
        obsolete_path = os.path.join(gen_dir, fname)
        log(f"Removing obsolete file {obsolete_path}")
        if os.path.isdir(obsolete_path):
          shutil.rmtree(obsolete_path)
        else:
          os.remove(obsolete_path)


class Runner:
  """Base class for a runner."""
  RUNNER_IDENT = None

  def __init__(self, env: Environment, test_id: str):
    self.env = env
    self.test_id = test_id
    self.gen_dir = os.path.join(self.env.output_dir, "generated")
    self.xfail = False

  @property
  def runner_ident(self) -> str:
    assert self.RUNNER_IDENT, "Must define RUNNER_IDENT"
    return self.RUNNER_IDENT

  def create_gen_file(self, file_name: str, mode: str = "wt"):
    os.makedirs(self.gen_dir, exist_ok=True)
    full_path = os.path.join(self.gen_dir, file_name)
    handle = open(full_path, mode)
    self.env.remember_gen_file(full_path)
    return handle

  def link_file(self, from_path: str, to_path: str):
    if from_path == to_path:
      return
    from_path = os.path.realpath(from_path)
    os.makedirs(os.path.dirname(to_path), exist_ok=True)
    if os.path.exists(to_path):
      os.remove(to_path)
    os.symlink(from_path, to_path)

  def generate(self):
    raise NotImplementedError(f"Generate not implemented for {self.__class__}")


################################################################################
# Main logic
################################################################################


def parse_arguments():
  parser = argparse.ArgumentParser(description="Test matrix generator")
  parser.add_argument("--dir",
                      required=True,
                      type=str,
                      help="Directory to process")
  parser.add_argument("--output_dir",
                      required=True,
                      type=str,
                      help="Output directory")
  args = parser.parse_args()
  return args


def main(args):
  env = Environment(args, args.dir, args.output_dir)
  process_directory(env)


def process_directory(env: Environment):
  dir = os.path.realpath(env.root_dir)
  try:
    config_sections = read_directory_config(dir)
  except Exception as e:
    raise RuntimeError(f"Could not read configuration from {dir}") from e
  for section in config_sections:
    require_mapping(section)
    for config_key, config_value in section.items():
      if config_key == "lists":
        # Ignore: a place to stash anchors and references.
        pass
      elif config_key == "test_groups":
        require_list(config_value)
        for test_group in config_value:
          require_mapping(test_group)
          process_test_group(env, test_group)
      else:
        raise ValueError(f"Unexpected top-level section {config_key}")

  env.prune_gen_files()


def process_test_group(env: Environment, test_group):
  group_id = get_mapping_key(test_group, "id", require_str)
  matrix = generate_matrix(
      get_mapping_key(test_group, "matrix", require_mapping))
  matrix_id_map = {group_id.format(**m): m for m in matrix}
  for runner_map in get_mapping_key(test_group, "runner", require_list):
    for matrix_id, matrix_map in matrix_id_map.items():
      runner = create_runner(env, matrix_id, runner_map, matrix_map)
      runner.xfail = (evaluate_xfail(test_group, matrix_map) and
                      not evaluate_xpass(test_group, matrix_map))
      runner.generate()


def evaluate_xfail(test_group, matrix_map) -> bool:
  try:
    xfail_list = flatten_lists(require_list(test_group["xfail"]))
  except KeyError:
    return False
  for xfail_group in xfail_list:
    if evaluate_matrix_map_predicate(matrix_map, xfail_group):
      return True
  return False


def evaluate_xpass(test_group, matrix_map) -> bool:
  try:
    xpass_list = flatten_lists(require_list(test_group["xpass"]))
  except KeyError:
    return False
  for xpass_group in xpass_list:
    if evaluate_matrix_map_predicate(matrix_map, xpass_group):
      return True
  return False


def evaluate_matrix_map_predicate(matrix_map, predicate_group) -> bool:
  # Each key is something like 'matrix.<key>' which are and'ed
  # together. Each value is either a literal or a list that is
  # or'd together.
  for pred_key, pred_value in predicate_group.items():
    match_value = None
    if pred_key.startswith("matrix."):
      try:
        match_value = matrix_map[pred_key[len("matrix."):]]
      except KeyError:
        raise ValueError(
            f"Could not match matrix predicate to matrix value: {pred_key}")
    else:
      raise ValueError(
          f"Expected a matrix predicate (i.e. matrix.) but got {pred_key}")
    # Match list (OR) or literal (==)
    if isinstance(pred_value, list):
      if match_value not in flatten_lists(pred_value):
        return False
    else:
      if pred_value != match_value:
        return False
  return True


################################################################################
# Utilities
################################################################################


def generate_matrix(matrix_map):
  # List of (key, [value, value, ...])
  matrix_entries = [(k, flatten_lists(v)) for k, v in matrix_map.items()]
  # Permute.
  permuted = []

  def accumulate(prior: dict, i: int):
    if i == len(matrix_entries):
      permuted.append(prior)
      return
    next_key, next_values = matrix_entries[i]
    for next_value in next_values:
      current = dict(prior)
      current[next_key] = next_value
      accumulate(current, i + 1)

  accumulate({}, 0)
  return permuted


def read_directory_config(dir: str) -> list:
  sections = []
  matrix_path = os.path.join(dir, "test_matrix.yaml")
  with open(matrix_path, "r") as stream:
    for section in yaml.safe_load_all(stream):
      sections.append(section)
  return sections


INDENT = 0


def log(msg: str):
  print("  " * INDENT + msg)


@contextmanager
def indent():
  global INDENT
  INDENT += 1
  yield
  INDENT -= 1


def flatten_lists(l):
  results = list()
  for item in l:
    if isinstance(item, list):
      results.extend(flatten_lists(item))
    else:
      results.append(item)
  return results


def require_mapping(v):
  if isinstance(v, dict):
    return v
  raise ValueError(f"Expected a YAML mapping for {v}")


def require_list(v):
  if isinstance(v, list):
    return v
  raise ValueError(f"Expected YAML list for {v}")


def require_str(v):
  if isinstance(v, str):
    return v
  raise ValueError(f"Expected str for {v}")


def get_mapping_key(mapping, key: str, checker=None):
  if key not in mapping:
    raise ValueError(f"Expected key '{key}' in {mapping}")
  value = mapping[key]
  if checker:
    checker(value)
  return value


################################################################################
# Runners
################################################################################

PYRUNNER_STUB = r"""
import importlib
import os
import sys
resolved_imports = False
for _ in range(2):
  try:
    for impname in REQUIRE_IMPORTS:
      importlib.import_module(impname)
    resolved_imports = True
  except ModuleNotFoundError as e:
    if os.path.exists(os.path.join(os.getcwd(), "CMakeCache.txt")):
      d = os.path.join(os.getcwd(), "bindings", "python")
      if os.path.exists(d):
        print(f"Added {d} to sys.path", file=sys.stderr)
        sys.path.append(d)
if not resolved_imports:
  raise Exception(f"Cannot find required imports: {REQUIRE_IMPORTS}\n"
                  f"If running interactively, ensure that you are in the "
                  f"build directory or have packages installed")

sys.argv = [sys.argv[0]] + ARGS + sys.argv[1:]
with open(MAIN, "rt") as f:
  script = f.read()
FAILED = False
try:
  exec(script, globals())
except SystemExit as exitex:
  # unittest like to sys.exit() itself. Catch that here so we can
  # process XFAIL properly.
  if exitex.code: FAILED = True
except:
  FAILED = True
  import traceback
  traceback.print_exc()
  if not XFAIL:
    raise

if XFAIL:
  if FAILED:
    print("=== TEST FAILED AS EXPECTED ===", file=sys.stderr)
    sys.exit(0)
  else:
    print("=== TEST PASSED BUT WAS EXPECTED TO FAIL ===", file=sys.stderr)
    sys.exit(1)

if FAILED:
  sys.exit(1)
"""


class TfHostRunner(Runner):
  """Runner for tf e2e host tests."""
  RUNNER_IDENT = "tfhost"

  def __init__(self, env: Environment, test_id: str, runner_map: dict,
               matrix_map: dict):
    super().__init__(env=env, test_id=test_id)
    self.main_file = get_mapping_key(runner_map, "main", require_str)
    raw_arg_list = get_mapping_key(runner_map, "args", require_list)
    self.args = [
        require_str(raw_arg).format(**matrix_map) for raw_arg in raw_arg_list
    ]

  def generate(self):
    # Generate the runner script.
    file_name = (
        f"{'XFAIL_' if self.xfail else ''}{self.test_id}_{self.runner_ident}.py"
    )
    with self.create_gen_file(file_name) as f:
      parts = [
          "import os",
          "import sys",
          "REQUIRE_IMPORTS = ['iree.tf.support.tf_utils', 'iree.tf.support.tf_test_utils']",
          f"ARGS = {repr(self.args)}",
          f"MAIN = os.path.join(os.path.dirname(__file__), '..', {repr(self.main_file)})",
          f"XFAIL = {self.xfail}",
          PYRUNNER_STUB,
      ]
      f.write("\n".join(parts))

    # Copy/link the main file.
    main_file_src_path = os.path.join(self.env.root_dir, self.main_file)
    main_file_dst_path = os.path.join(self.env.output_dir, self.main_file)
    if not os.path.exists(main_file_src_path):
      raise RuntimeError(
          f"Referenced main file '{main_file_src_path}' does not exist")
    self.link_file(main_file_src_path, main_file_dst_path)


RUNNER_CLASSES = {
    "tfhost": TfHostRunner,
}


def create_runner(env: Environment, test_id: str, runner_map: dict,
                  matrix_map: dict):
  runner_type = get_mapping_key(runner_map, "type", require_str)
  try:
    runner_class = RUNNER_CLASSES[runner_type]
  except KeyError:
    raise ValueError(f"Unknown runner type '{runner_type}'")
  return runner_class(env=env,
                      test_id=test_id,
                      runner_map=runner_map,
                      matrix_map=matrix_map)


if __name__ == "__main__":
  main(parse_arguments())
