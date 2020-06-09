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

import os
import platform
import setuptools
import sys

# Ensure that path starts here for execution as a script.
sys.path.insert(0, os.path.dirname(__file__))
import common_setup


def run():
  package_dir = common_setup.get_package_dir(
      prefix=("integrations", "tensorflow", "bindings", "python"))
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
