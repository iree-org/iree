# Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# This builds just the runtime API and is relatively quick to build.
# To install:
#   pip install .
# To build a wheel:
#   pip wheel .
#
# It is recommended to build with Ninja and ccache. To do so, set environment
# variables by prefixing to above invocations:
#   CMAKE_C_COMPILER_LAUNCHER=ccache CMAKE_CXX_COMPILER_LAUNCHER=ccache
#
# On CIs, it is often advantageous to re-use/control the CMake build directory.
# This can be set with the IREE_RUNTIME_API_CMAKE_BUILD_DIR env var.
#
# A custom package suffix can be specified with the environment variable:
#   IREE_RUNTIME_CUSTOM_PACKAGE_SUFFIX
#
# Select CMake options are available from environment variables:
#   IREE_HAL_DRIVER_CUDA
#   IREE_HAL_DRIVER_VULKAN
#   IREE_ENABLE_RUNTIME_TRACING
#   IREE_BUILD_TRACY
#   IREE_ENABLE_CPUINFO

import json
import os
import platform
import re
import shutil
import subprocess
import sys
import sysconfig
from distutils.command.build import build as _build
from gettext import install
from multiprocessing.spawn import prepare

from setuptools import Extension, find_namespace_packages, setup
from setuptools.command.build_ext import build_ext as _build_ext
from setuptools.command.build_py import build_py as _build_py


def check_pip_version():
  from packaging import version

  # Pip versions < 22.0.3 default to out of tree builds, which is quite
  # incompatible with what we do (and has other issues). Pip >= 22.0.4
  # removed this option entirely and are only in-tree builds. Since the
  # old behavior can silently produce unworking installations, we aggressively
  # suppress it.
  try:
    import pip
  except ModuleNotFoundError:
    # If pip not installed, we are obviously not trying to package via pip.
    pass
  else:
    if (version.parse(pip.__version__) < version.parse("21.3")):
      print("ERROR: pip version >= 21.3 required")
      print("Upgrade: pip install pip --upgrade")
      sys.exit(2)


check_pip_version()

# This file can be run directly from the source tree or it can be CMake
# configured so it can run from the build tree with an already existing
# build tree. We detect the difference based on whether the following
# are expanded by CMake.
CONFIGURED_SOURCE_DIR = "@IREE_SOURCE_DIR@"
CONFIGURED_BINARY_DIR = "@IREE_BINARY_DIR@"

IREE_SOURCE_DIR = None
IREE_BINARY_DIR = None

# We must do the intermediate installation to a fixed location that agrees
# between what we pass to setup() and cmake. So hard-code it here.
# Note that setup() needs a relative path (to the setup.py file).
SETUPPY_DIR = os.path.realpath(os.path.dirname(__file__))
CMAKE_INSTALL_DIR_REL = os.path.join("build", "cmake_install")
CMAKE_INSTALL_DIR_ABS = os.path.join(SETUPPY_DIR, CMAKE_INSTALL_DIR_REL)

IS_CONFIGURED = CONFIGURED_SOURCE_DIR[0] != "@"
if IS_CONFIGURED:
  IREE_SOURCE_DIR = CONFIGURED_SOURCE_DIR
  IREE_BINARY_DIR = CONFIGURED_BINARY_DIR
  print(
      f"Running setup.py from build tree: "
      f"SOURCE_DIR = {IREE_SOURCE_DIR} "
      f"BINARY_DIR = {IREE_BINARY_DIR}",
      file=sys.stderr)
else:
  IREE_SOURCE_DIR = os.path.join(SETUPPY_DIR, "..")
  IREE_BINARY_DIR = os.getenv("IREE_RUNTIME_API_CMAKE_BUILD_DIR")
  if not IREE_BINARY_DIR:
    # Note that setuptools always builds into a "build" directory that
    # is a sibling of setup.py, so we just colonize a sub-directory of that
    # by default.
    IREE_BINARY_DIR = os.path.join(SETUPPY_DIR, "build", "cmake_build")
  print(
      f"Running setup.py from source tree: "
      f"SOURCE_DIR = {IREE_SOURCE_DIR} "
      f"BINARY_DIR = {IREE_BINARY_DIR}",
      file=sys.stderr)

# Setup and get version information.
VERSION_INFO_FILE = os.path.join(IREE_SOURCE_DIR, "version_info.json")


def load_version_info():
  with open(VERSION_INFO_FILE, "rt") as f:
    return json.load(f)


def find_git_versions():
  revisions = {}
  try:
    revisions["IREE"] = subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=IREE_SOURCE_DIR).decode("utf-8").strip()
  except subprocess.SubprocessError as e:
    print(f"ERROR: Could not get IREE revision: {e}", file=sys.stderr)
  return revisions


def find_git_submodule_revision(submodule_path):
  try:
    data = subprocess.check_output(["git", "ls-tree", "HEAD", submodule_path],
                                   cwd=IREE_SOURCE_DIR).decode("utf-8").strip()
    columns = re.split("\\s+", data)
    return columns[2]
  except Exception as e:
    print(
        f"ERROR: Could not get submodule revision for {submodule_path}"
        f" ({e})",
        file=sys.stderr)
    return ""


try:
  version_info = load_version_info()
except FileNotFoundError:
  print("version_info.json not found. Using defaults", file=sys.stderr)
  version_info = {}
git_versions = find_git_versions()

PACKAGE_SUFFIX = version_info.get("package-suffix") or ""
PACKAGE_VERSION = version_info.get("package-version")
if not PACKAGE_VERSION:
  PACKAGE_VERSION = f"0.dev0+{git_versions.get('IREE') or '0'}"


def maybe_nuke_cmake_cache():
  # From run to run under pip, we can end up with different paths to ninja,
  # which isn't great and will confuse cmake. Detect if the location of
  # ninja changes and force a cache flush.
  ninja_path = ""
  try:
    import ninja
  except ModuleNotFoundError:
    pass
  else:
    ninja_path = ninja.__file__
  expected_stamp_contents = f"{sys.executable}\n{ninja_path}"

  # In order to speed things up on CI and not rebuild everything, we nuke
  # the CMakeCache.txt file if the path to the Python interpreter changed.
  # Ideally, CMake would let us reconfigure this dynamically... but it does
  # not (and gets very confused).
  PYTHON_STAMP_FILE = os.path.join(IREE_BINARY_DIR, "python_stamp.txt")
  if os.path.exists(PYTHON_STAMP_FILE):
    with open(PYTHON_STAMP_FILE, "rt") as f:
      actual_stamp_contents = f.read()
      if actual_stamp_contents == expected_stamp_contents:
        # All good.
        return

  # Mismatch or not found. Clean it.
  cmake_cache_file = os.path.join(IREE_BINARY_DIR, "CMakeCache.txt")
  if os.path.exists(cmake_cache_file):
    print("Removing CMakeCache.txt because Python version changed",
          file=sys.stderr)
    os.remove(cmake_cache_file)

  # Also clean the install directory. This avoids version specific pileups
  # of binaries that can occur with repeated builds against different
  # Python versions.
  if os.path.exists(CMAKE_INSTALL_DIR_ABS):
    print(
        f"Removing CMake install dir because Python version changed: "
        f"{CMAKE_INSTALL_DIR_ABS}",
        file=sys.stderr)
    shutil.rmtree(CMAKE_INSTALL_DIR_ABS)

  # And write.
  with open(PYTHON_STAMP_FILE, "wt") as f:
    f.write(expected_stamp_contents)


def get_env_cmake_option(name: str, default_value: bool = False) -> str:
  svalue = os.getenv(name)
  if not svalue:
    svalue = "ON" if default_value else "OFF"
  return f"-D{name}={svalue}"


def add_env_cmake_setting(args, env_name: str, cmake_name=None) -> str:
  svalue = os.getenv(env_name)
  if svalue is not None:
    if not cmake_name:
      cmake_name = env_name
    args.append(f"-D{cmake_name}={svalue}")


def prepare_installation():
  subprocess.check_call(["cmake", "--version"])
  version_py_content = generate_version_py()
  print(f"Generating version.py:\n{version_py_content}", file=sys.stderr)

  if not IS_CONFIGURED:
    # Build from source tree.
    os.makedirs(IREE_BINARY_DIR, exist_ok=True)
    maybe_nuke_cmake_cache()
    print(f"CMake build dir: {IREE_BINARY_DIR}", file=sys.stderr)
    print(f"CMake install dir: {CMAKE_INSTALL_DIR_ABS}", file=sys.stderr)
    cfg = "Release"
    cmake_args = [
        "-GNinja",
        "--log-level=VERBOSE",
        "-DIREE_BUILD_PYTHON_BINDINGS=ON",
        "-DIREE_BUILD_COMPILER=OFF",
        "-DIREE_BUILD_SAMPLES=OFF",
        "-DIREE_BUILD_TESTS=OFF",
        "-DPython3_EXECUTABLE={}".format(sys.executable),
        "-DCMAKE_BUILD_TYPE={}".format(cfg),
        get_env_cmake_option("IREE_HAL_DRIVER_CUDA"),
        get_env_cmake_option("IREE_HAL_DRIVER_VULKAN",
                             "OFF" if platform.system() == "Darwin" else "ON"),
        get_env_cmake_option("IREE_ENABLE_RUNTIME_TRACING"),
        get_env_cmake_option("IREE_BUILD_TRACY"),
        get_env_cmake_option("IREE_ENABLE_CPUINFO", "ON"),
    ]

    # These usually flow through the environment, but we add them explicitly
    # so that they show clearly in logs (getting them wrong can have bad
    # outcomes).
    add_env_cmake_setting(cmake_args, "CMAKE_OSX_ARCHITECTURES")
    add_env_cmake_setting(cmake_args, "MACOSX_DEPLOYMENT_TARGET",
                          "CMAKE_OSX_DEPLOYMENT_TARGET")

    # Only do a from-scratch configure if not already configured.
    cmake_cache_file = os.path.join(IREE_BINARY_DIR, "CMakeCache.txt")
    if not os.path.exists(cmake_cache_file):
      print(f"Configuring with: {cmake_args}", file=sys.stderr)
      subprocess.check_call(["cmake", IREE_SOURCE_DIR] + cmake_args,
                            cwd=IREE_BINARY_DIR)
    else:
      print(f"Not re-configuring (already configured)", file=sys.stderr)

    # Build. Since we have restricted to just the runtime, build everything
    # so as to avoid fragility with more targeted selection criteria.
    subprocess.check_call(["cmake", "--build", "."], cwd=IREE_BINARY_DIR)
    print("Build complete.", file=sys.stderr)

  # Install the component we care about.
  install_args = [
      "-DCMAKE_INSTALL_DO_STRIP=ON",
      f"-DCMAKE_INSTALL_PREFIX={CMAKE_INSTALL_DIR_ABS}/",
      f"-DCMAKE_INSTALL_COMPONENT=IreePythonPackage-runtime",
      "-P",
      os.path.join(IREE_BINARY_DIR, "cmake_install.cmake"),
  ]
  print(f"Installing with: {install_args}", file=sys.stderr)
  subprocess.check_call(["cmake"] + install_args, cwd=IREE_BINARY_DIR)

  # Write version.py directly into install dir.
  version_py_file = os.path.join(CMAKE_INSTALL_DIR_ABS, "python_packages",
                                 "iree_runtime", "iree", "runtime",
                                 "version.py")
  os.makedirs(os.path.dirname(version_py_file), exist_ok=True)
  with open(version_py_file, "wt") as f:
    f.write(version_py_content)

  print(f"Installation prepared: {CMAKE_INSTALL_DIR_ABS}", file=sys.stderr)


class CMakeBuildPy(_build_py):

  def run(self):
    # It is critical that the target directory contain all built extensions,
    # or else setuptools will helpfully compile an empty binary for us
    # (this is the **worst** possible thing it could do). We just copy
    # everything. What's another hundred megs between friends?
    target_dir = os.path.abspath(self.build_lib)
    print(f"Building in target dir: {target_dir}", file=sys.stderr)
    os.makedirs(target_dir, exist_ok=True)
    print("Copying install to target.", file=sys.stderr)
    if os.path.exists(target_dir):
      shutil.rmtree(target_dir)
    shutil.copytree(os.path.join(CMAKE_INSTALL_DIR_ABS, "python_packages",
                                 "iree_runtime"),
                    target_dir,
                    symlinks=False)
    print("Target populated.", file=sys.stderr)


class CustomBuild(_build):

  def run(self):
    self.run_command("build_py")
    self.run_command("build_ext")
    self.run_command("build_scripts")


class CMakeExtension(Extension):

  def __init__(self, name, sourcedir=""):
    Extension.__init__(self, name, sources=[])
    self.sourcedir = os.path.abspath(sourcedir)


class NoopBuildExtension(_build_ext):

  def __init__(self, *args, **kwargs):
    assert False

  def build_extension(self, ext):
    pass


def generate_version_py():
  return f"""# Auto-generated version info.
PACKAGE_SUFFIX = "{PACKAGE_SUFFIX}"
VERSION = "{PACKAGE_VERSION}"
REVISIONS = {json.dumps(git_versions)}
"""


prepare_installation()

packages = find_namespace_packages(where=os.path.join(CMAKE_INSTALL_DIR_ABS,
                                                      "python_packages",
                                                      "iree_runtime"),
                                   include=[
                                       "iree._runtime",
                                       "iree.runtime",
                                       "iree.runtime.*",
                                   ])
print(f"Found runtime packages: {packages}")

with open(
    os.path.join(IREE_SOURCE_DIR, "runtime", "bindings", "python", "iree",
                 "runtime", "README.md"), "rt") as f:
  README = f.read()

custom_package_suffix = os.getenv("IREE_RUNTIME_CUSTOM_PACKAGE_SUFFIX")
if not custom_package_suffix:
  custom_package_suffix = ""

setup(
    name=f"iree-runtime{custom_package_suffix}{PACKAGE_SUFFIX}",
    version=f"{PACKAGE_VERSION}",
    author="IREE Authors",
    author_email="iree-discuss@googlegroups.com",
    description="IREE Python Runtime Components",
    long_description=README,
    long_description_content_type="text/markdown",
    license="Apache-2.0",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    url="https://github.com/openxla/iree",
    python_requires=">=3.7",
    ext_modules=[
        CMakeExtension("iree._runtime"),
    ],
    cmdclass={
        "build": CustomBuild,
        "built_ext": NoopBuildExtension,
        "build_py": CMakeBuildPy,
    },
    zip_safe=False,
    package_dir={
        # Note: Must be relative path, so we line this up with the absolute
        # path built above. Note that this must exist prior to the call.
        "": f"{CMAKE_INSTALL_DIR_REL}/python_packages/iree_runtime",
    },
    packages=packages,
    # Matching the native extension as a data file keeps setuptools from
    # "building" it (i.e. turning it into a static binary).
    package_data={
        "": [
            f"*{sysconfig.get_config_var('EXT_SUFFIX')}",
            "iree-run-module*",
            "iree-run-trace*",
            "iree-benchmark-module*",
            "iree-benchmark-trace*",
            "iree-tracy-capture*",
        ],
    },
    entry_points={
        "console_scripts": [
            "iree-run-module = iree.runtime.scripts.iree_run_module.__main__:main",
            "iree-run-trace = iree.runtime.scripts.iree_run_trace.__main__:main",
            "iree-benchmark-module = iree.runtime.scripts.iree_benchmark_module.__main__:main",
            "iree-benchmark-trace = iree.runtime.scripts.iree_benchmark_trace.__main__:main",
            "iree-tracy-capture = iree.runtime.scripts.iree_tracy_capture.__main__:main",
        ],
    },
    install_requires=[
        "numpy",
        "PyYAML",
    ],
)
