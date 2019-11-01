# Copyright 2019 Google LLC
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

# Lint as: python3
"""Utilities for running tests from TensorFlow models."""

import contextlib
import io
import os
import subprocess
import sys
import tempfile
import traceback

from absl import app
from absl import flags
import tensorflow.compat.v2 as tf
import pyiree

flags.DEFINE_string("filecheck_binary", "filecheck",
                    "Location of the filecheck binary.")
flags.DEFINE_bool("disable_filecheck", False,
                  "Disables filecheck redirection (for debugging).")
FLAGS = flags.FLAGS

ALL_TEST_DICTS = []


def add_test(**kwargs):
  assert "test_name" in kwargs, "'test_name' is a required argument"
  ALL_TEST_DICTS.append(kwargs)


def _run_test(test_dict):
  """Runs an individual test dict."""
  tf_module_builder_lambda = test_dict["tf_module_builder"]
  tf_module = tf_module_builder_lambda()
  ctx = pyiree.CompilerContext()
  with tempfile.TemporaryDirectory() as sm_path:
    options = tf.saved_model.SaveOptions(save_debug_info=True)
    tf.saved_model.save(tf_module, sm_path, options=options)
    input_module = pyiree.tf_load_saved_model(ctx, sm_path)

  passes = test_dict.get("passes")
  expect_pass_failure = test_dict.get("expect_pass_failure")
  if passes:
    try:
      input_module.run_pass_pipeline(passes)
    except:  # pylint: disable=bare-except
      if not expect_pass_failure:
        print(
            "UNEXPECTED PASS FAILURE (INTERMEDIATE ASM FOLLOWS ON STDERR):",
            file=sys.stderr)
        print(input_module.to_asm(), file=sys.stderr)
      raise

  # Print the input module ASM.
  if test_dict.get("print_input_module"):
    print(input_module.to_asm())


def _internal_run_tests():
  """Main function that runs all tests."""
  test_count = 0
  for test_dict in ALL_TEST_DICTS:
    test_count += 1
    test_name = test_dict["test_name"]
    print("RUN_TEST:", test_name)
    try:
      _run_test(test_dict)
      print("FINISH_TEST:", test_name)
    except:  # pylint: disable=bare-except
      # Error goes to stdout for FileCheck.
      traceback.print_exc(file=sys.stdout)
      print("FINISH_TEST_WITH_EXCEPTION:", test_name)

  print("FINISHED: RAN", test_count, "TESTS", file=sys.stderr)


def _find_filecheck():
  filecheck_binary = FLAGS.filecheck_binary
  if os.path.isabs(filecheck_binary):
    return filecheck_binary
  # TODO(laurenzo): Why is this runfiles resolution so hard and undocumented.
  # Talk to bazel team.
  runfiles_dir = os.environ.get("RUNFILES_DIR")
  if runfiles_dir:
    workspace_name = os.environ.get("TEST_WORKSPACE")
    if workspace_name:
      runfiles_dir = os.path.join(runfiles_dir, workspace_name)
    filecheck_binary = os.path.join(runfiles_dir, filecheck_binary)
  # Convert forward slash version to platform default (Windows).
  filecheck_binary = filecheck_binary.replace("/", os.path.sep)
  return filecheck_binary


def run_tests(main_file, with_filecheck=True):
  """Main entry point."""

  def internal_main(unused_argv):
    """App main."""
    # In case if running with a version prior to v2 defaulting.
    tf.enable_v2_behavior()
    if with_filecheck and not FLAGS.disable_filecheck:
      # Capture and run through filecheck.
      filecheck_capture_io = io.StringIO()
      with contextlib.redirect_stdout(filecheck_capture_io):
        _internal_run_tests()
      filecheck_capture_io.flush()
      filecheck_input = filecheck_capture_io.getvalue()
      # Convert forward slash version to platform default (Windows).
      filecheck_binary = _find_filecheck()
      filecheck_args = [filecheck_binary, main_file, "--dump-input=fail"]
      print("LAUNCHING FILECHECK:", filecheck_args, file=sys.stderr)
      p = subprocess.Popen(filecheck_args, stdin=subprocess.PIPE)
      p.communicate(filecheck_input.encode("UTF-8"))
      sys.exit(p.returncode)
    else:
      # Just run directly.
      _internal_run_tests()

  app.run(internal_main)
