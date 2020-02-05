#!/usr/bin/env python3
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

# Usage:
#   python3 colab/start_colab_kernel.py
#
# Note that in the case that multiple python interpreters are present on
# your path, it is best to not risk it: use an explicit one.
#
# This will build the python bindings and start a colab kernel with them
# on the path. It takes some care to ensure that the build is running with
# the same python interpreter as is used to launch this script.
#
# Pre-requisites:
# Install Jupyter (from https://jupyter.org/install)
#   python3 -m pip install --upgrade pip
#   python3 -m pip install jupyter
# Setup colab (https://research.google.com/colaboratory/local-runtimes.html)
#   python3 -m pip install jupyter_http_over_ws
#   jupyter serverextension enable --py jupyter_http_over_ws
# If you plan on using TensorFlow, enable the TensorFlow parts of IREE's
# compiler by adding a define to your user.bazelrc file:
#   build --define=iree_tensorflow=true

import os
import subprocess
import shutil
import sys

repo_root = None
bazel_env = dict(os.environ)
bazel_bin = None
bazel_exe = None


def setup_environment():
  """Sets up some environment globals."""
  global bazel_bin
  global repo_root
  global bazel_exe

  # Determine the repository root (two dir-levels up).
  repo_root = os.path.dirname(
      os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
  print("Repository root: %s" % (repo_root,))

  # Use 'bazelisk' instead of 'bazel' if it exists on the path.
  # Bazelisk is an optional utility that pick versions of Bazel to use and
  # passes through all command-line arguments to the real Bazel binary:
  # https://github.com/bazelbuild/bazelisk
  bazel_exe = "bazelisk" if shutil.which("bazelisk") else "bazel"
  print("Using bazel executable: %s" % (bazel_exe))

  # Detect python and query bazel for its output.
  print("Setting Bazel PYTHON_BIN=%s" % (sys.executable,))
  bazel_env["PYTHON_BIN"] = sys.executable
  bazel_bin = subprocess.check_output([bazel_exe, "info", "bazel-bin"],
                                      cwd=repo_root,
                                      env=bazel_env).decode("utf-8")
  bazel_bin = bazel_bin.splitlines()[0]
  # Bazel always reports the path with '/'. On windows, switch it
  # since we need native path manipulation code below to have it the
  # right way.
  if os.path.sep == "\\":
    bazel_bin = bazel_bin.replace("/", "\\")
  print("Found Bazel bin: %s" % (bazel_bin))


def build():
  """Builds the python bundle."""
  print("Building python bindings...")
  subprocess.check_call([bazel_exe, "build", "//colab:everything_for_colab"],
                        cwd=repo_root,
                        env=bazel_env)


def run():
  """Runs the Jupyter notebook."""
  runfiles_suffix = ".runfiles"
  if os.path.sep == "\\":
    runfiles_suffix = ".exe.runfiles"  # Windows uses a special name

  runfiles_dir = os.path.join(bazel_bin, "colab",
                              "everything_for_colab" + runfiles_suffix)
  # Top level directories under the runfiles get added to the sys path.
  extra_python_path = []
  # The iree_core/bindings/python directory under runfiles needs to come
  # first on the path.
  extra_python_path.append(
      os.path.join(runfiles_dir, "iree_core", "bindings", "python"))
  extra_python_path.append(
      os.path.join(runfiles_dir, "iree_core", "integrations", "tensorflow",
                   "bindings", "python"))
  for python_module in os.listdir(runfiles_dir):
    python_module_path = os.path.join(runfiles_dir, python_module)
    if os.path.isdir(python_module_path):
      extra_python_path.append(python_module_path)

  print("Augmented Python sys.path:")
  for p in extra_python_path:
    print(" ", p)
  launch_jupyter(extra_python_path)


def launch_jupyter(python_path):
  """Launches Jupyter with a python path."""
  try:
    from jupyter_core.command import main as jupyter_main  # pylint: disable=g-import-not-at-top
  except ImportError:
    show_install_instructions()
    sys.exit(1)

  # Override the PYTHONPATH, which Jupyter propagates to its kernels.
  path_sep = ":"
  if os.path.sep == "\\":
    path_sep = ";"  # Windows
  os.environ["PYTHONPATH"] = path_sep.join(python_path)

  # Launch jupyter (this is all the "jupyter" shell command does).
  sys.argv = [
      "jupyter", "notebook",
      "--NotebookApp.allow_origin='https://colab.research.google.com'",
      "--port=8888", "--NotebookApp.port_retries=0"
  ]
  sys.exit(jupyter_main())


def show_install_instructions():
  """Prints some install instructions."""
  print("ERROR: Unable to load Jupyter. Ensure that it is installed:")
  print("  %s -m pip install --upgrade pip" % (sys.executable,))
  print("  %s -m pip install jupyter" % (sys.executable,))
  print("  %s -m pip install jupyter_http_over_ws" % (sys.executable,))
  print("  jupyter serverextension enable --py jupyter_http_over_ws")


if __name__ == "__main__":
  setup_environment()
  build()
  run()
