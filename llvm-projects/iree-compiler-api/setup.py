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
#   CMAKE_GENERATOR=Ninja CMAKE_C_COMPILER_LAUNCHER=ccache CMAKE_CXX_COMPILER_LAUNCHER=ccache
#
# On CIs, it is often advantageous to re-use/control the CMake build directory.
# This can be set with the IREE_COMPILER_API_CMAKE_BUILD_DIR env var.
import os
import shutil
import subprocess
import sys

from distutils.command.build import build as _build
from setuptools import find_namespace_packages, setup, Extension
from setuptools.command.build_ext import build_ext as _build_ext
from setuptools.command.build_py import build_py as _build_py


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
    subprocess.check_call(["cmake", "--version"])

    target_dir = self.build_lib
    cmake_build_dir = os.getenv("IREE_COMPILER_API_CMAKE_BUILD_DIR")
    if not cmake_build_dir:
      cmake_build_dir = os.path.join(target_dir, "..", "cmake_build")
    cmake_install_dir = os.path.join(target_dir, "..", "cmake_install")
    src_dir = os.path.abspath(os.path.dirname(__file__))
    cfg = "Release"
    cmake_args = [
        "-DCMAKE_INSTALL_PREFIX={}".format(os.path.abspath(cmake_install_dir)),
        "-DPython3_EXECUTABLE={}".format(sys.executable),
        "-DIREE_VERSION_INFO={}".format(self.distribution.get_version()),
        "-DCMAKE_BUILD_TYPE={}".format(cfg),
    ]
    build_args = []
    os.makedirs(cmake_build_dir, exist_ok=True)
    if os.path.exists(cmake_install_dir):
      shutil.rmtree(cmake_install_dir)
    cmake_cache_file = os.path.join(cmake_build_dir, "CMakeCache.txt")
    if os.path.exists(cmake_cache_file):
      os.remove(cmake_cache_file)
    subprocess.check_call(["cmake", src_dir] + cmake_args, cwd=cmake_build_dir)
    subprocess.check_call(
        ["cmake", "--build", ".", "--target", "install/strip"] + build_args,
        cwd=cmake_build_dir)
    shutil.copytree(os.path.join(cmake_install_dir, "python_package"),
                    target_dir,
                    symlinks=False,
                    dirs_exist_ok=True)


class NoopBuildExtension(_build_ext):

  def __init__(self, *args, **kwargs):
    assert False

  def build_extension(self, ext):
    pass


setup(
    name="iree-compiler-api",
    version="0.0.1",
    author="IREE Authors",
    author_email="iree-discuss@googlegroups.com",
    description="IREE Compiler API",
    long_description="",
    ext_modules=[
        CMakeExtension("iree.compiler._mlir_libs._mlir"),
        CMakeExtension("iree.compiler._mlir_libs._ireeDialects"),
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
)
