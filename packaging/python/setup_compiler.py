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

# Build platform specific wheel files for the pyiree.rt package.
# Built artifacts are per-platform and build out of the build tree.
# Usage:
# ------
# Windows with CMake:
#   export CMAKE_BUILD_ROOT='D:\src\build-iree'  # Must be native path
#   python ./setup_compiler.py bdist_wheel

import os
import setuptools
import sys

# Ensure that path starts here for execution as a script.
sys.path.insert(0, os.path.dirname(__file__))
import common_setup


def run():
  packages = setuptools.find_namespace_packages(
      common_setup.get_package_dir(),
      include=["pyiree.compiler", "pyiree.compiler.*"],
      exclude=["*.CMakeFiles"])
  print("Found packages:", packages)
  setup_kwargs = common_setup.get_setup_defaults(
      sub_project="compiler", description="IREE Generic Compiler")
  common_setup.setup(
      packages=packages,
      ext_modules=[
          setuptools.Extension(name="pyiree.compiler.binding", sources=[]),
      ],
      **setup_kwargs)


if __name__ == "__main__":
  run()
