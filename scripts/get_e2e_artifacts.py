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
"""Runs all E2E TensorFlow tests and extracts their benchmarking artifacts.

Example usage:
  python3 get_e2e_artifacts.py
"""

import fileinput
import os
import re
import subprocess
import tempfile
from zipfile import ZipFile

import utils

from absl import app
from absl import flags

FLAGS = flags.FLAGS

flags.DEFINE_bool(
    'dry_run', False,
    'Run without extracting files. Useful for quickly checking for artifact '
    'collisions.')
flags.DEFINE_string(
    'artifacts_dir', None,
    'Directory to transfer the benchmarking artifacts to. Defaults to '
    '/tmp/iree/modules/')
flags.DEFINE_bool('run_test_suites', True, 'Run any specified test suites.')
flags.DEFINE_list(
    'test_suites', ['e2e_tests', 'keras_tests', 'vision_external_tests'],
    'Any combination of `e2e_tests`, `keras_tests`, and `vision_external_tests`')

SUITE_NAME_TO_TARGET = {
    'e2e_tests':
        '//integrations/tensorflow/e2e:e2e_tests',
    'keras_tests':
        '//integrations/tensorflow/e2e/keras:keras_tests',
    'vision_external_tests':
        '//integrations/tensorflow/e2e/keras:vision_external_tests',
}

EXPECTED_COLLISIONS = [
    '/tf_ref/', 'tf_input.mlir', 'iree_input.mlir', '/saved_model/'
]
WRITTEN_PATHS = set()
PATHS_TO_TESTS = dict()


def get_test_paths_and_names(test_suite_path: str):
  """Get the paths Bazel stores test outputs in and the matching test names."""
  targets = utils.get_test_targets(test_suite_path)
  # Convert the test target into the path where Bazel stores the artifacts we
  # put in `TEST_UNDECLARED_OUTPUTS_DIR`.
  test_paths = [target.replace('//', '') for target in targets]
  test_paths = [path.replace(':', os.sep) for path in test_paths]
  test_paths = [os.path.join('bazel-testlogs', path) for path in test_paths]

  # Get test_name from `suite_name_test_name__tf__backend_name`
  test_names = [target.split('__')[0] for target in targets]
  test_names = [name.replace(f'{test_suite_path}_', '') for name in test_names]
  return test_paths, test_names


def collision_check(filename: str, test_name: str):
  """Check that we aren't overwriting files unless we expect to."""
  # Note: We can't use a check that the files have identical contents because
  # tf_input.mlir can have random numbers appended to its function names.

  expected_collision = any([name in filename for name in EXPECTED_COLLISIONS])
  if filename in WRITTEN_PATHS and not expected_collision:
    print()  # Clear the unterminated counter line before raising.
    raise ValueError(f'Collision found on {filename} between {test_name}.py '
                     f'and {PATHS_TO_TESTS[filename]}.py')
  else:
    WRITTEN_PATHS.add(filename)
    PATHS_TO_TESTS[filename] = test_name


def update_path(archive_path: str, artifacts_dir: str):
  """Update the --input_file flag with the new location of the compiled.vmfb"""
  backend_path = archive_path.split('traces')[0]  # 'ModuleName/backend_name'.
  compiled_path = os.path.join(artifacts_dir, backend_path, 'compiled.vmfb')
  flagfile_path = os.path.join(artifacts_dir, archive_path)
  for line in fileinput.input(files=[flagfile_path], inplace=True):
    if line.strip().startswith('--input_file'):
      print(f'--input_file={compiled_path}\n', end='')
    else:
      print(line, end='')


def extract_artifacts(test_path: str, test_name: str, artifacts_dir: str):
  """Unzips all of the benchmarking artifacts for a given test and backend."""
  outputs = os.path.join(test_path, 'test.outputs', 'outputs.zip')
  archive = ZipFile(outputs)
  # Filter out directory names.
  filenames = [name for name in archive.namelist() if name[-1] != os.sep]

  for filename in filenames:
    # Check for collisions.
    collision_check(filename, test_name)

    # Extract and update flagfile path.
    if not FLAGS.dry_run:
      archive.extract(filename, artifacts_dir)
      if filename.endswith('flagfile'):
        update_path(filename, artifacts_dir)


def main(argv):
  del argv  # Unused.

  # Get the artifacts dir or default to `/tmp/iree/modules/`
  artifacts_dir = FLAGS.artifacts_dir
  if artifacts_dir is None:
    artifacts_dir = os.path.join(tempfile.gettempdir(), 'iree', 'modules')

  # Convert test suite shorthands to full test suite targets.
  test_suites = [SUITE_NAME_TO_TARGET[suite] for suite in FLAGS.test_suites]

  for test_suite in test_suites:
    if FLAGS.run_test_suites and not FLAGS.dry_run:
      subprocess.check_output(['bazel', 'test', test_suite, '--color=yes'])
      print()

    # Extract all of the artifacts for this test suite.
    test_paths, test_names = get_test_paths_and_names(test_suite)
    for i, (test_path, test_name) in enumerate(zip(test_paths, test_names)):
      print(f'\rTransfering {test_suite} {i + 1}/{len(test_paths)}', end='')
      extract_artifacts(test_path, test_name, artifacts_dir)
    print('\n')


if __name__ == '__main__':
  app.run(main)
