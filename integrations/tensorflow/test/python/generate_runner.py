# Copyright 2021 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Generates a ".run" file in the corresponding test tree for a file under
# this directory. This is a convenience for bootstrapping new test files.
# Usage:
#   python generate_runner.py llvmcpu "--target_backends=iree_llvmcpu" \
#     iree_tf_tests/uncategorized/batch_norm_test.py[:batch_norm_alias]
#
# The first argument is the lit feature that this test is gated on and will
# be prepended to the test name as "{variant}__testfile".
# The second argument is the flag string to include when running the test.
# All remaining arguments are relative paths to python files under this
# directory that should have a .run file created for them.

import os
import sys


def main(args):
  variant = args[0]
  flags = args[1]
  src_file_specs = args[2:]
  src_files = [
      transform_src_file_spec_to_src_file(spec) for spec in src_file_specs
  ]
  module_names = [transform_src_file_to_module(f) for f in src_files]
  run_files = [
      transform_src_file_spec_to_run_file(spec, variant)
      for spec in src_file_specs
  ]
  for module, run_file in zip(module_names, run_files):
    if os.path.exists(run_file):
      print(f"SKIPPING (exists): {run_file}")
      continue
    print(f"CREATE RUN FILE: {module} -> {run_file}")
    os.makedirs(os.path.dirname(run_file), exist_ok=True)
    with open(run_file, "wt") as f:
      print(f"# REQUIRES: {variant}", file=f)
      print(f"# RUN: %PYTHON -m {module} {flags}", file=f)


def transform_src_file_spec_to_src_file(spec: str):
  try:
    colon_pos = spec.index(":")
  except ValueError:
    return spec
  return spec[0:colon_pos]


def transform_src_file_to_module(file_name):
  module_name = file_name.replace("/", ".")
  if (module_name.endswith(".py")):
    module_name = module_name[0:-3]
  return module_name


def transform_src_file_spec_to_run_file(spec, variant):
  # Transform path:alias, defaulting to the basename if the alias is not
  # specified.
  file_path = spec
  file_name = os.path.basename(file_path)
  colon_pos = -1
  try:
    colon_pos = spec.index(":")
  except ValueError:
    pass
  if colon_pos > -1:
    # Explicit alias.
    file_path = spec[0:colon_pos]
    file_name = spec[colon_pos + 1:]
    print(f"FILE PATH = {file_path}")
  else:
    # Auto detect the alias from the basename.
    file_name = os.path.basename(file_path)
    if file_name.endswith(".py"):
      file_name = file_name[0:-3]
    if file_name.endswith("_test"):
      file_name = file_name[0:-5]

  main_test_dir = os.path.join(os.path.dirname(__file__), "..")
  parent_path = os.path.dirname(file_path)

  file_name = f"{variant}__{file_name}.run"
  run_file = os.path.join(main_test_dir, parent_path, file_name)
  return run_file


if __name__ == "__main__":
  main(sys.argv[1:])
