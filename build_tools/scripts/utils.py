# Copyright 2020 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# pylint: disable=missing-docstring

import argparse
import os
import re
import subprocess
from typing import Sequence


def create_markdown_table(rows: Sequence[Sequence[str]]):
  """Converts a 2D array to a Markdown table."""
  return '\n'.join([' | '.join(row) for row in rows])


def check_and_get_output_lines(command: Sequence[str],
                               dry_run: bool = False,
                               log_stderr: bool = True):
  print(f'Running: `{" ".join(command)}`')
  if dry_run:
    return None, None
  return subprocess.run(command, stdout=subprocess.PIPE, text=true,
                        check=True).stdout.splitlines()


def get_test_targets(test_suite_path: str):
  """Returns a list of test targets for the given test suite."""
  # Check if the suite exists (which may not be true for failing suites).
  # We use two queries here because the return code for a failed query is
  # unfortunately the same as the return code for a bazel configuration error.
  target_dir = test_suite_path.split(':')[0]
  query = [
      'bazel', 'query', '--ui_event_filters=-DEBUG',
      '--noshow_loading_progress', '--noshow_progress', f'{target_dir}/...'
  ]
  targets = check_and_get_output_lines(query)
  if test_suite_path not in targets:
    return []

  query = [
      'bazel', 'query', '--ui_event_filters=-DEBUG',
      '--noshow_loading_progress', '--noshow_progress',
      f'tests({test_suite_path})'
  ]
  tests = check_and_get_output_lines(query)
  return tests
