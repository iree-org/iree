<<<<<<< HEAD
#!/bin/bash
=======
#!/usr/bin/env python3
>>>>>>> d4286e6b1e5939d9de64c676b09f416c36f834bd

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
<<<<<<< HEAD
from typing import List, Tuple, Union


def execute_in_shell(command: str, 
                     shell='/bin/bash': str) -> Tuple[Union[str, int]]:
  """Executes command and returns its stdout, stderr and returncode."""
  p = subprocess.Popen([shell, command], shell=True, stdout=subprocess.PIPE,
                       stderr=subprocess.PIPE)
  stdout, stderr = p.communicate()
  return stdout, stderr, p.returncode
=======
from typing import List
>>>>>>> d4286e6b1e5939d9de64c676b09f416c36f834bd


def parse_arguments():
  parser = argparse.ArgumentParser(description="run_lit.py helper.")
  parser.add_argument("--test_file", help="Path of the test file to execute.",
                      required=True)
  args = parser.parse_args()
  return args


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
<<<<<<< HEAD

=======
  
>>>>>>> d4286e6b1e5939d9de64c676b09f416c36f834bd
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


def windows_convert_subpath(subpath: List[str]) -> List[str]:
  """Converts each directory in subpath using cygpath if on Windows."""
  # TODO: Remove this function if it is not needed.
<<<<<<< HEAD
  # TODO: Determine if calling '/bin/bash' explicitly through subprocess works
  #       as expected on Windows.
  # Attempt to find the path of cygpath on windows. If empty, not on windows.
  cygpath, _, _ = execute_in_shell('which cygpath')
  if cygpath:
    for i, path in enumerate(subpath):
      subpath[i], _, _ = execute_in_shell(f'{cygpath} -u {path}')
=======
  # Attempt to find the path of cygpath on windows. If empty, not on windows
  cygpath, _ = subprocess.Popen('which cygpath', shell=True,
                                stdout=subprocess.PIPE).communicate()
  if cygpath:
    for i, path in enumerate(subpath):
      subpath[i], _ = subprocess.Popen(f'{cygpath} -u {path}', shell=True,
                                       stdout=subprocess.PIPE).communicate()
>>>>>>> d4286e6b1e5939d9de64c676b09f416c36f834bd
  return subpath


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
<<<<<<< HEAD
  # subdirectories to add all the parent directories of files the user can
=======
  # subdirectories to add all the parent directories of files the user can 
>>>>>>> d4286e6b1e5939d9de64c676b09f416c36f834bd
  # execute to the path.
  subpath = [os.path.dirname(file) for file in find_runfiles(runfiles_dir)]
  # TODO: Figure out if windows path conversion is necessary in python, or if
  #       os handles it already.
  # subpath = convert_subpath(subpath)
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
<<<<<<< HEAD

=======
  
>>>>>>> d4286e6b1e5939d9de64c676b09f416c36f834bd
  # Run all of the commands on the shell.
  for command in commands:
    # Substitute any embedded '%s' with the file name.
    command = command.replace('%s', args.test_file)

    print(f'RUNNING TEST: {command}', end='')
    print('-' * 80)
<<<<<<< HEAD

    stdout, stderr, returncode = execute_in_shell(command)
    print(stdout.decode('utf-8'), end='')
    print(stderr.decode('utf-8'), end='')
    if returncode:
=======
    p = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, 
                         stderr=subprocess.PIPE)
    out, err = p.communicate()
    print(out.decode('utf-8'), end='')
    print(err.decode('utf-8'), end='')
    if p.returncode:
>>>>>>> d4286e6b1e5939d9de64c676b09f416c36f834bd
      print(f'!!! ERROR EVALUATING: {command}', end='')
      exit(1)
    print('--- COMPLETE ---' + '-' * 64 + '\n')

if __name__ == "__main__":
  main(parse_arguments())