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

import argparse
import os
import re
import subprocess
from typing import List, Tuple, Union


def parse_arguments():
  parser = argparse.ArgumentParser(description="run_lit.py helper.")
  parser.add_argument("--test_file", help="Path of the test file to execute.",
                      required=True)
  args = parser.parse_args()
  return args


def execute_in_shell(command: str,
                     shell: str = '/bin/bash') -> Tuple[Union[str, int]]:
  """Executes command and returns its stdout, stderr and returncode."""
  p = subprocess.Popen([shell, command], shell=True, stdout=subprocess.PIPE,
                       stderr=subprocess.PIPE)
  stdout, stderr = p.communicate()
  return stdout, stderr, p.returncode


def find_runfiles(runfiles_dir: str) -> List[str]:
  runfiles = []
  # Iterate through all files in runfiles_dir and its subdirectories.
  for (path, _, files) in os.walk(runfiles_dir):
    for file_name in files:
      file_path = os.path.join(path, file_name)

      # Check each file to see if the user can execute it.
      if os.access(file_path, os.X_OK):
        runfiles.append(file_path)
  return runfiles


def get_runline_commands(test_file: str) -> List[str]:
  """Returns all of the commands listed in the given test file."""
  with open(test_file, 'r') as f:
    lines = f.readlines()
  commands = []
  for i, line in enumerate(lines):
    match = re.match('^// RUN: *', line)
    if match:
      commands.append(line[match.span()[-1]:])
      if not commands[-1]:
        print(f'ERROR: Encountered an empty runline on line {i + 1} of '
              f'{test_file}')
        exit(1)
  return commands


def main(args):
  """Executes all of the RUN lines in the given test file."""
  if 'RUNFILES_DIR' in os.environ:
    runfiles_dir = os.environ['RUNFILES_DIR']  # Usually set by bazel.
  else:
    # Some versions of bazel do not set RUNFILES_DIR. Instead they just cd
    # into the directory.
    runfiles_dir = os.getcwd()

  # Bazel helpfully puts all data deps in the RUNFILES_DIR, but
  # it unhelpfully preserves the nesting with no way to reason about
  # it generically. run_lit expects that anything passed in the runfiles
  # can be found on the path for execution.
  # Currently we iterate through all of the files in RUNFILES_DIR and its
  # subdirectories to add all the parent directories of files the user can
  # execute to the path.
  subpath = [os.path.dirname(file) for file in find_runfiles(runfiles_dir)]
  subpath = list(set(subpath))  # Remove duplicate paths.
  os.environ['PATH'] = f'{":".join(subpath)}:{os.environ["PATH"]}'

  print(f'Running run_lit.py on {args.test_file}')
  print(f'RUNFILES_DIR: {runfiles_dir}')
  print(f'Current dir:  {os.getcwd()}\n')

  # Extract all of the RUN lines in the given test file and ensure there are
  # tests to run.
  commands = get_runline_commands(args.test_file)
  if len(commands) == 0:
    print(f'!!! No RUN lines found in {args.test_file}')
    exit(1)
  # Run all of the commands on the shell.
  for command in commands:
    # Substitute any embedded '%s' with the file name.
    command = command.replace('%s', args.test_file)
    print(f'RUNNING TEST: {command}', end='')
    print('-' * 80)

    stdout, stderr, returncode = execute_in_shell(command)
    print(stdout.decode('utf-8'), end='')
    print(stderr.decode('utf-8'), end='')
    if returncode:
      print(f'!!! ERROR EVALUATING: {command}', end='')
      exit(1)
    print('--- COMPLETE ---' + '-' * 64 + '\n')

if __name__ == "__main__":
  main(parse_arguments())