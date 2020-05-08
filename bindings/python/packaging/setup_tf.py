#!/usr/bin/python3

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

# Build platform specific wheel files for the pyiree.tf packages.
# Built artifacts are per-platform and build out of the build tree.
# Usage:
# ------
#  bazel build -c opt //integrations/tensorflow/bindings/python/packaging:all_tf_packages
#  python3 ./setup_tf.py bdist_wheel
#
# Tips:
# Optionally add: --define=PYIREE_TF_DISABLE_KERNELS=1
# to build a 'thin' (less functional) version without TensorFlow kernels.
# This should not be done for released binaries but can help while developing.
#
# Note that this script violates our general policy of keeping TensorFlow
# things in the integrations/tensorflow directory because it is more convenient
# to have all packaging scripts in one place, and this will not grow further
# dependencies.

import os
import setuptools
import sys

# Ensure that path starts here for execution as a script.
sys.path.insert(0, os.path.dirname(__file__))
import common_setup


def find_bazel_runfiles_dir():
  bazel_bin = os.path.abspath(
      os.path.join(os.path.dirname(__file__), "..", "..", "..", "bazel-bin"))
  if not os.path.isdir(bazel_bin):
    print("ERROR: Could not find bazel-bin:", bazel_bin)
    sys.exit(1)
  # Find the path to the runfiles of the built target:
  #   //integrations/tensorflow/bindings/python/packaging:all_tf_packages
  runfiles_dir = os.path.join(bazel_bin, "integrations", "tensorflow",
                              "bindings", "python", "packaging",
                              "all_tf_packages.runfiles")
  if not os.path.isdir(runfiles_dir):
    print("ERROR: Could not find build target 'all_tf_packages':", runfiles_dir)
    print(
        "Make sure to build target",
        "//integrations/tensorflow/bindings/python/packaging:all_tf_packages")
    sys.exit(1)
  # And finally seek into the corresponding path in the runfiles dir.
  # Aren't bazel paths fun???
  # Note that the "iree_core" path segment corresponds to the workspace name.
  package_path = os.path.join(runfiles_dir, "iree_core", "integrations",
                              "tensorflow", "bindings", "python")
  if not os.path.isdir(package_path):
    print("ERROR: Could not find built python package:", package_path)
    sys.exit(1)
  return package_path


def run():
  package_dir = find_bazel_runfiles_dir()
  packages = setuptools.find_namespace_packages(
      package_dir,
      include=[
          "pyiree.tf.compiler", "pyiree.tf.compiler.*", "pyiree.tf.support",
          "pyiree.tf.support.*"
      ],
      exclude=["*.CMakeFiles"])
  print("Found packages:", packages)
  if not packages:
    print("ERROR: Did not find packages under", package_dir)
    sys.exit(1)
  setup_kwargs = common_setup.get_setup_defaults(
      sub_project="tf",
      description="IREE TensorFlow Compiler",
      package_dir=package_dir)
  common_setup.setup(packages=packages, **setup_kwargs)


if __name__ == "__main__":
  run()
