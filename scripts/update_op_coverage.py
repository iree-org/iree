#!/usr/bin/env python3

# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Updates op coverage table.

Example usage: ./update_op_coverage.py IREE_BUILD_DIR
"""

import argparse
import collections
import os
import subprocess

# The symbols to show in the table if the operation is supported or not.
SUCCESS_ELEMENT = '<span class="success-table-element">✓</span>'
FAILURE_ELEMENT = '<span class="failure-table-element">✗</span>'

E2E_XLA_OPS_PATH = 'iree/test/e2e/xla_ops'

# TODO(scotttodd): LLVM AOT (dylib-llvm-aot) HAL target(s)
OP_COVERAGE_DESCRIPTION = """# XLA HLO Op Coverage
There are three backend [targets](https://github.com/google/iree/tree/main/iree/compiler/Dialect/HAL/Target) in IREE:

- vmla
- llvm-ir
- vulkan-spirv

The table shows the supported XLA HLO ops on each backend. It is auto-generated
from IREE's test status.

"""


def parse_arguments():
  """Parses command-line options."""
  parser = argparse.ArgumentParser(
      description='Generates Markdown files for op coverage table')
  parser.add_argument(
      'build_dir', metavar='BUILD_PATH', type=str, help='Base build directory.')

  parsed_args = parser.parse_args()
  if not os.path.isdir(parsed_args.build_dir):
    raise parser.error('expected path to a directory')

  return parsed_args


def get_backend_op_pair(test):
  """Returns the target backend and operation pair of the test."""
  test_suite_backends = {
      'check_vmla_vmla': 'vmla',
      'check_llvm-ir_llvm': 'llvm-ir',
      'check_vulkan-spirv_vulkan': 'vulkan-spirv'
  }
  for (test_suite, backend) in test_suite_backends.items():
    if test_suite in test:
      # Format: ...TEST_SUITE_OP.mlir
      start_idx = test.index(test_suite) + len(test_suite) + 1
      return backend, test[start_idx:-len('.mlir')]
  raise LookupError(f'Can not find a backend to match {test}')


def get_tested_ops_for_backends(build_dir):
  """Parses current op tests for each backend."""

  ctest_output = subprocess.check_output(
      ['ctest', '-N', '-L', E2E_XLA_OPS_PATH], cwd=build_dir)
  tests = ctest_output.decode('ascii').strip().split('\n')
  res = collections.defaultdict(list)
  for t in tests:
    if not t.endswith('.mlir'):
      continue
    backend, op = get_backend_op_pair(t)
    res[backend].append(op)
  return res


def create_markdown_table(rows):
  """Converts a 2D array to a Markdown table."""
  return '\n'.join([' | '.join(row) for row in rows])


def generate_table(build_dir):
  """Generates an op coverage Markdown table for each backend."""
  backend_ops = get_tested_ops_for_backends(build_dir)
  backends = list(backend_ops.keys())

  all_ops = []
  for ops in backend_ops.values():
    all_ops.extend(ops)
  all_ops = list(set(all_ops))
  all_ops.sort()

  first_row = ['op'] + backends
  second_row = [':-:' for _ in first_row]
  rows = [first_row, second_row]
  for op in all_ops:
    row = [op]
    for backend in backends:
      row.append(
          SUCCESS_ELEMENT if (op in backend_ops[backend]) else FAILURE_ELEMENT)
    rows.append(row)
  return create_markdown_table(rows)


if __name__ == '__main__':
  args = parse_arguments()
  content = generate_table(args.build_dir)
  table_path = os.path.join(args.build_dir, 'doc', 'xla_op_coverage.md')
  with open(table_path, 'w', encoding='utf-8') as f:
    f.write(OP_COVERAGE_DESCRIPTION)
    f.write(content)
