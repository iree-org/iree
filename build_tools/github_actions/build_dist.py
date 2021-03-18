# Copyright 2021 Google LLC
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
"""Builds the main distribution package.

This script runs as the CIBW_BEFORE_BUILD command within cibuildwheel:
  - Main distribution .tar.bz2 file (the result of `ninja install`).
  - The python_packages/iree_compiler wheel, which is python version
    independent but platform specific.
  - Installable tests.

It uses cibuildwheel for all of this as a convenience since it already knows
how to arrange for the cross platform part of the build, including using
an appropriate manylinux image, etc.

This is expected to be run from the project directory, containing the
following sub-directories:
  - main_checkout/ : Main IREE repository checkout.
  - bindist/ : Directory where binary distribution artifacts are written.
  - main_checkout/version_info.json : Version config information.

Within the build environment (which may be the naked runner or a docker image):
  - iree-build/ : The build tree.
  - iree-install/ : The install tree.

Environment variables:
  - BINDIST_DIR : If set, then this overrides the default bindist/ directory.
    Should be set if running in a mapped context like a docker container.

Testing this script:
It is not recommended to run cibuildwheel locally. However, this script can
be executed as if running within such an environment. To do so, create
a directory and:
  ln -s /path/to/iree main_checkout
  python -m venv .venv
  source .venv/bin/activate

  python ./main_checkout/build_tools/github_actions/build_dist.py main-dist
  python ./main_checkout/build_tools/github_actions/build_dist.py py-runtime-pkg
  python ./main_checkout/build_tools/github_actions/build_dist.py py-xla-compiler-tools-pkg
  python ./main_checkout/build_tools/github_actions/build_dist.py py-tflite-compiler-tools-pkg
  python ./main_checkout/build_tools/github_actions/build_dist.py py-tf-compiler-tools-pkg


That is not a perfect approximation but is close.
"""

import json
import os
import platform
import shutil
import subprocess
import sys
import sysconfig
import tarfile

# Setup.
WORK_DIR = os.path.realpath(os.path.curdir)
BUILD_DIR = os.path.join(WORK_DIR, "iree-build")
INSTALL_DIR = os.path.join(WORK_DIR, "iree-install")
IREESRC_DIR = os.path.join(WORK_DIR, "main_checkout")
TF_INTEGRATIONS_DIR = os.path.join(IREESRC_DIR, "integrations/tensorflow")
BINDIST_DIR = os.environ.get("BINDIST_DIR")
if BINDIST_DIR is None:
  BINDIST_DIR = os.path.join(WORK_DIR, "bindist")
THIS_DIR = os.path.realpath(os.path.dirname(__file__))
CMAKE_CI_SCRIPT = os.path.join(THIS_DIR, "cmake_ci.py")
BUILD_REQUIREMENTS_TXT = os.path.join(IREESRC_DIR, "bindings", "python",
                                      "build_requirements.txt")
CONFIGURE_BAZEL_PY = os.path.join(IREESRC_DIR, "configure_bazel.py")
INSTALL_TARGET = ("install"
                  if platform.system() == "Windows" else "install/strip")


# Load version info.
def load_version_info():
  with open(os.path.join(IREESRC_DIR, "version_info.json"), "rt") as f:
    return json.load(f)


try:
  version_info = load_version_info()
except FileNotFoundError:
  print("version_info.json found. Using defaults")
  version_info = {
      "package-version": "0.1dev1",
      "package-suffix": "-dev",
  }


def remove_cmake_cache():
  cache_file = os.path.join(BUILD_DIR, "CMakeCache.txt")
  if os.path.exists(cache_file):
    print(f"Removing {cache_file}")
    os.remove(cache_file)
  else:
    print(f"Not removing cache file (does not exist): {cache_file}")


def install_python_requirements():
  print("Installing python requirements...")
  subprocess.check_call(
      [sys.executable, "-m", "pip", "install", "-r", BUILD_REQUIREMENTS_TXT])


def configure_bazel():
  print("Generating configured.bazelrc...")
  subprocess.check_call([sys.executable, CONFIGURE_BAZEL_PY])


def build_main_dist():
  """Builds the main distribution binaries.

  Also builds the iree-install/python_packages/iree_compiler package, ready
  for wheel building.
  """
  install_python_requirements()

  # Clean up install and build trees.
  shutil.rmtree(INSTALL_DIR, ignore_errors=True)
  remove_cmake_cache()

  # CMake configure.
  print("*** Configuring ***")
  subprocess.run([
      sys.executable,
      CMAKE_CI_SCRIPT,
      f"-B{BUILD_DIR}",
      f"-DCMAKE_INSTALL_PREFIX={INSTALL_DIR}",
      f"-DCMAKE_BUILD_TYPE=Release",
      f"-DIREE_BUILD_COMPILER=ON",
      f"-DIREE_BUILD_PYTHON_BINDINGS=ON",
      f"-DIREE_BUILD_SAMPLES=OFF",
  ],
                 check=True)

  print("*** Building ***")
  subprocess.run([
      sys.executable,
      CMAKE_CI_SCRIPT,
      "--build",
      BUILD_DIR,
      "--target",
      INSTALL_TARGET,
  ],
                 check=True)

  print("*** Packaging ***")
  dist_entries = [
      "bin",
      "tests",
  ]
  dist_archive = os.path.join(
      BINDIST_DIR, f"iree-dist{version_info['package-suffix']}"
      f"-{version_info['package-version']}"
      f"-{sysconfig.get_platform()}.tar.xz")
  print(f"Creating archive {dist_archive}")
  os.makedirs(os.path.dirname(dist_archive), exist_ok=True)
  with tarfile.open(dist_archive, mode="w:xz") as tf:
    for entry in dist_entries:
      print(f"Adding entry: {entry}")
      tf.add(os.path.join(INSTALL_DIR, entry), arcname=entry, recursive=True)


def build_py_runtime_pkg():
  """Builds the iree-install/python_packages/iree_runtime package.

  This includes native, python-version dependent code and is designed to
  be built multiple times.
  """
  install_python_requirements()

  # Clean up install and build trees.
  shutil.rmtree(INSTALL_DIR, ignore_errors=True)
  remove_cmake_cache()

  # CMake configure.
  print("*** Configuring ***")
  subprocess.run([
      sys.executable,
      CMAKE_CI_SCRIPT,
      f"-B{BUILD_DIR}",
      f"-DCMAKE_INSTALL_PREFIX={INSTALL_DIR}",
      f"-DCMAKE_BUILD_TYPE=Release",
      f"-DIREE_BUILD_COMPILER=OFF",
      f"-DIREE_BUILD_PYTHON_BINDINGS=ON",
      f"-DIREE_BUILD_SAMPLES=OFF",
      f"-DIREE_BUILD_TESTS=OFF",
  ],
                 check=True)

  print("*** Building ***")
  subprocess.run([
      sys.executable,
      CMAKE_CI_SCRIPT,
      "--build",
      BUILD_DIR,
      "--target",
      "install-IreePythonPackage-rt-stripped",
  ],
                 check=True)


def bazel_build_tf_binary(target):
  """Builds a binary in the IREE-TF Workspace and returns the filepath."""
  install_python_requirements()
  configure_bazel()

  # Builds a runnable target and returns the path to the executable. Yes this is
  # really the best Bazel gives us.
  # See https://github.com/bazelbuild/bazel/issues/8739
  cmd = [
      "bazel",
      "run",
      "--run_under=echo",
      "--config=release",
      "--keep_going",
      target,
  ]
  process = subprocess.run(cmd,
                           cwd=TF_INTEGRATIONS_DIR,
                           check=True,
                           stdout=subprocess.PIPE,
                           universal_newlines=True)

  if len(process.stdout.splitlines()) != 1:
    raise RuntimeError(
        f"Unexpected output from `{' '.join(cmd)}`:\n{process.stdout}")
  bin_path = process.stdout.strip()
  if not os.path.isfile(bin_path):
    raise RuntimeError("{bin_path} is not a file.")

  return bin_path


def build_py_xla_compiler_tools_pkg():
  """Builds the iree-install/python_packages/iree_tools_xla package."""
  install_python_requirements()
  configure_bazel()

  # Clean up install and build trees.
  shutil.rmtree(INSTALL_DIR, ignore_errors=True)
  remove_cmake_cache()

  print("*** Building XLA import tool with Bazel ***")
  binpath = bazel_build_tf_binary("//iree_tf_compiler:iree-import-xla")

  # CMake configure.
  print("*** Configuring ***")
  subprocess.run([
      sys.executable,
      CMAKE_CI_SCRIPT,
      f"-B{BUILD_DIR}",
      f"-DIREE_TF_TOOLS_ROOT={os.path.dirname(binpath)}",
      f"-DCMAKE_INSTALL_PREFIX={INSTALL_DIR}",
      f"-DCMAKE_BUILD_TYPE=Release",
      f"-DIREE_BUILD_XLA_COMPILER=ON",
      f"-DIREE_BUILD_PYTHON_BINDINGS=ON",
      f"-DIREE_BUILD_SAMPLES=OFF",
      f"-DIREE_BUILD_TESTS=OFF",
  ],
                 check=True)

  print("*** Building ***")
  subprocess.run([
      sys.executable,
      CMAKE_CI_SCRIPT,
      "--build",
      BUILD_DIR,
      "--target",
      "install-IreePythonPackage-tools-xla-stripped",
  ],
                 check=True)


def build_py_tflite_compiler_tools_pkg():
  """Builds the iree-install/python_packages/iree_tools_tflite package."""
  install_python_requirements()
  configure_bazel()

  # Clean up install and build trees.
  shutil.rmtree(INSTALL_DIR, ignore_errors=True)
  remove_cmake_cache()

  print("*** Building TFLite import tool with Bazel ***")
  binpath = bazel_build_tf_binary("//iree_tf_compiler:iree-import-tflite")

  # CMake configure.
  print("*** Configuring ***")
  subprocess.run([
      sys.executable,
      CMAKE_CI_SCRIPT,
      f"-B{BUILD_DIR}",
      f"-DIREE_TF_TOOLS_ROOT={os.path.dirname(binpath)}",
      f"-DCMAKE_INSTALL_PREFIX={INSTALL_DIR}",
      f"-DCMAKE_BUILD_TYPE=Release",
      f"-DIREE_BUILD_TFLITE_COMPILER=ON",
      f"-DIREE_BUILD_PYTHON_BINDINGS=ON",
      f"-DIREE_BUILD_SAMPLES=OFF",
      f"-DIREE_BUILD_TESTS=OFF",
  ],
                 check=True)

  print("*** Building ***")
  subprocess.run([
      sys.executable,
      CMAKE_CI_SCRIPT,
      "--build",
      BUILD_DIR,
      "--target",
      "install-IreePythonPackage-tools-tflite-stripped",
  ],
                 check=True)


def build_py_tf_compiler_tools_pkg():
  """Builds the iree-install/python_packages/iree_tools_tf package."""
  install_python_requirements()
  configure_bazel()

  # Clean up install and build trees.
  shutil.rmtree(INSTALL_DIR, ignore_errors=True)
  remove_cmake_cache()

  print("*** Building TF import tool with Bazel ***")
  binpath = bazel_build_tf_binary("//iree_tf_compiler:iree-tf-import")

  # CMake configure.
  print("*** Configuring ***")
  subprocess.run([
      sys.executable,
      CMAKE_CI_SCRIPT,
      f"-B{BUILD_DIR}",
      f"-DIREE_TF_TOOLS_ROOT={os.path.dirname(binpath)}",
      f"-DCMAKE_INSTALL_PREFIX={INSTALL_DIR}",
      f"-DCMAKE_BUILD_TYPE=Release",
      f"-DIREE_BUILD_TENSORFLOW_COMPILER=ON",
      f"-DIREE_BUILD_PYTHON_BINDINGS=ON",
      f"-DIREE_BUILD_SAMPLES=OFF",
      f"-DIREE_BUILD_TESTS=OFF",
  ])

  print("*** Building ***")
  subprocess.run([
      sys.executable,
      CMAKE_CI_SCRIPT,
      "--build",
      BUILD_DIR,
      "--target",
      "install-IreePythonPackage-tools-tf-stripped",
  ],
                 check=True)


command = sys.argv[1]
if command == "main-dist":
  build_main_dist()
elif command == "py-runtime-pkg":
  build_py_runtime_pkg()
elif command == "py-xla-compiler-tools-pkg":
  build_py_xla_compiler_tools_pkg()
elif command == "py-tflite-compiler-tools-pkg":
  build_py_tflite_compiler_tools_pkg()
elif command == "py-tf-compiler-tools-pkg":
  build_py_tf_compiler_tools_pkg()
else:
  print(f"Unrecognized command: {command}")
