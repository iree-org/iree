# Copyright 2021 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Build/install the iree-compiler-backend python package.
# Note that this includes a relatively large build of LLVM (~2400 C++ files)
# and can take a considerable amount of time, especially with defaults.
# To install:
#   pip install . --use-feature=in-tree-build
# To build a wheel:
#   pip wheel . --use-feature=in-tree-build
#
# It is recommended to build with Ninja and ccache. To do so, set environment
# variables by prefixing to above invocations:
#   CMAKE_C_COMPILER_LAUNCHER=ccache CMAKE_CXX_COMPILER_LAUNCHER=ccache
#
# On CIs, it is often advantageous to re-use/control the CMake build directory.
# This can be set with the IREE_COMPILER_API_CMAKE_BUILD_DIR env var.
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import sysconfig

from distutils.command.build import build as _build
from setuptools import find_namespace_packages, setup, Extension
from setuptools.command.build_ext import build_ext as _build_ext
from setuptools.command.build_py import build_py as _build_py

# Setup and get version information.
THIS_DIR = os.path.realpath(os.path.dirname(__file__))
IREESRC_DIR = os.path.join(THIS_DIR, "..", "..")
VERSION_INFO_FILE = os.path.join(IREESRC_DIR, "version_info.json")


def load_version_info():
  with open(VERSION_INFO_FILE, "rt") as f:
    return json.load(f)


try:
  version_info = load_version_info()
except FileNotFoundError:
  print("version_info.json not found. Using defaults")
  version_info = {}

PACKAGE_SUFFIX = version_info.get("package-suffix") or ""
PACKAGE_VERSION = version_info.get("package-version") or "0.1dev1"


class CustomBuild(_build):

  def run(self):
    self.run_command("build_py")
    self.run_command("build_ext")
    self.run_command("build_scripts")


class CMakeExtension(Extension):

  def __init__(self, name, sourcedir=""):
    Extension.__init__(self, name, sources=[])
    self.sourcedir = os.path.abspath(sourcedir)


class CMakeBuildPy(_build_py):

  def run(self):
    version_py_content = generate_version_py()
    print(f"Generating version.py:\n{version_py_content}", file=sys.stderr)
    subprocess.check_call(["cmake", "--version"])

    target_dir = os.path.abspath(self.build_lib)
    print(f"Building in target dir: {target_dir}", file=sys.stderr)
    os.makedirs(target_dir, exist_ok=True)
    cmake_build_dir = os.getenv("IREE_COMPILER_API_CMAKE_BUILD_DIR")
    if not cmake_build_dir:
      cmake_build_dir = os.path.join(target_dir, "..", "cmake_build")
    os.makedirs(cmake_build_dir, exist_ok=True)
    cmake_build_dir = os.path.abspath(cmake_build_dir)
    print(f"CMake build dir: {cmake_build_dir}", file=sys.stderr)
    cmake_install_dir = os.path.abspath(
        os.path.join(target_dir, "..", "cmake_install"))
    print(f"CMake install dir: {cmake_install_dir}", file=sys.stderr)
    src_dir = os.path.abspath(os.path.dirname(__file__))
    cfg = "Release"
    cmake_args = [
        "-GNinja",
        "--log-level=VERBOSE",
        "-DCMAKE_INSTALL_PREFIX={}".format(cmake_install_dir),
        "-DPython3_EXECUTABLE={}".format(sys.executable),
        "-DPython3_INCLUDE_DIRS={}".format(sysconfig.get_path("include")),
        "-DIREE_VERSION_INFO={}".format(self.distribution.get_version()),
        "-DCMAKE_BUILD_TYPE={}".format(cfg),
    ]

    # Enable CUDA if specified.
    cuda_target_option = os.getenv("IREE_TARGET_BACKEND_CUDA")
    if cuda_target_option:
      cmake_args.append(f"-DIREE_TARGET_BACKEND_CUDA={cuda_target_option}")

    build_args = []
    if os.path.exists(cmake_install_dir):
      shutil.rmtree(cmake_install_dir)
    cmake_cache_file = os.path.join(cmake_build_dir, "CMakeCache.txt")
    if os.path.exists(cmake_cache_file):
      os.remove(cmake_cache_file)
    install_target = "install/strip"
    if platform.system() == "Windows":
      install_target = "install"
    print(f"Configuring with: {cmake_args}", file=sys.stderr)
    subprocess.check_call(["cmake", src_dir] + cmake_args, cwd=cmake_build_dir)
    subprocess.check_call(
        ["cmake", "--build", ".", "--target", install_target] + build_args,
        cwd=cmake_build_dir)
    print("Build complete.", file=sys.stderr)
    if os.path.exists(target_dir):
      shutil.rmtree(target_dir)
    print("Copying install to target.", file=sys.stderr)
    shutil.copytree(os.path.join(cmake_install_dir, "python_package"),
                    target_dir,
                    symlinks=False)
    print("Target populated.", file=sys.stderr)

    # Write version.py.
    version_py_file = os.path.join(target_dir, "iree", "compiler", "version.py")
    os.makedirs(os.path.dirname(version_py_file), exist_ok=True)
    with open(version_py_file, "wt") as f:
      f.write(version_py_content)


class NoopBuildExtension(_build_ext):

  def __init__(self, *args, **kwargs):
    assert False

  def build_extension(self, ext):
    pass


def generate_version_py():
  return f"""# Auto-generated version info.
PACKAGE_SUFFIX = "{PACKAGE_SUFFIX}"
VERSION = "{PACKAGE_VERSION}"
REVISIONS = {json.dumps(find_git_versions())}
"""


def find_git_versions():
  revisions = {}
  try:
    revisions["IREE"] = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=IREESRC_DIR).decode("utf-8").strip()
  except subprocess.SubprocessError as e:
    print(f"ERROR: Could not get IREE revision: {e}", file=sys.stderr)
  revisions["LLVM_PROJECT"] = find_git_submodule_revision(
      "third_party/llvm-project")
  revisions["TENSORFLOW"] = find_git_submodule_revision(
      "third_party/tensorflow")
  revisions["MLIR_HLO"] = find_git_submodule_revision("third_party/mlir-hlo")
  return revisions


def find_git_submodule_revision(submodule_path):
  try:
    data = subprocess.check_output(["git", "ls-tree", "HEAD", submodule_path],
                                   cwd=IREESRC_DIR).decode("utf-8").strip()
    columns = re.split("\\s+", data)
    return columns[2]
  except Exception as e:
    print(
        f"ERROR: Could not get submodule revision for {submodule_path}"
        f" ({e})",
        file=sys.stderr)
    return ""


setup(
    name=f"iree-compiler{PACKAGE_SUFFIX}",
    version=f"{PACKAGE_VERSION}",
    author="IREE Authors",
    author_email="iree-discuss@googlegroups.com",
    description="IREE Compiler API",
    long_description="",
    license="Apache-2.0",
    classifiers=[
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
    ext_modules=[
        CMakeExtension("iree.compiler._mlir_libs._mlir"),
        CMakeExtension("iree.compiler._mlir_libs._ireeDialects"),
        CMakeExtension("iree.compiler._mlir_libs._ireecTransforms"),
        CMakeExtension("iree.compiler._mlir_libs._mlirHlo"),
        CMakeExtension("iree.compiler._mlir_libs._mlirLinalgPasses"),
    ],
    cmdclass={
        "build": CustomBuild,
        "built_ext": NoopBuildExtension,
        "build_py": CMakeBuildPy,
    },
    zip_safe=False,
    packages=find_namespace_packages(include=[
        "iree.compiler",
        "iree.compiler.*",
    ],),
    entry_points={
        "console_scripts": [
            "ireec = iree.compiler.tools.scripts.ireec.__main__:main",
            # Transitional note: iree-translate resolves to ireec.
            "iree-translate = iree.compiler.tools.scripts.ireec.__main__:main",
        ],
    },
    install_requires=[
        "numpy",
        "PyYAML",
    ],
)
