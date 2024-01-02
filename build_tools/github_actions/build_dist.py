# Copyright 2021 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
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
  - c/ : Main IREE repository checkout.
  - bindist/ : Directory where binary distribution artifacts are written.
  - c/version_info.json : Version config information.

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
  ln -s /path/to/iree c
  python -m venv .venv
  source .venv/bin/activate

  python ./c/build_tools/github_actions/build_dist.py main-dist
  python ./c/build_tools/github_actions/build_dist.py py-tflite-compiler-tools-pkg
  python ./c/build_tools/github_actions/build_dist.py py-tf-compiler-tools-pkg


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
IREESRC_DIR = os.path.join(WORK_DIR, "c")
TF_INTEGRATIONS_DIR = os.path.join(IREESRC_DIR, "integrations/tensorflow")
BINDIST_DIR = os.environ.get("BINDIST_DIR")
if BINDIST_DIR is None:
    BINDIST_DIR = os.path.join(WORK_DIR, "bindist")
THIS_DIR = os.path.realpath(os.path.dirname(__file__))
CMAKE_CI_SCRIPT = os.path.join(THIS_DIR, "cmake_ci.py")
BUILD_REQUIREMENTS_TXT = os.path.join(
    IREESRC_DIR,
    "runtime",
    "bindings",
    "python",
    "iree",
    "runtime",
    "build_requirements.txt",
)
CI_REQUIREMENTS_TXT = os.path.join(THIS_DIR, "ci_requirements.txt")
CONFIGURE_BAZEL_PY = os.path.join(IREESRC_DIR, "configure_bazel.py")


# Load version info.
def load_version_info():
    with open(os.path.join(IREESRC_DIR, "version_info.json"), "rt") as f:
        return json.load(f)


try:
    version_info = load_version_info()
except FileNotFoundError:
    print("version_info.json not found. Using defaults")
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
        [sys.executable, "-m", "pip", "install", "-r", BUILD_REQUIREMENTS_TXT]
    )
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "-r", CI_REQUIREMENTS_TXT]
    )


def configure_bazel():
    print("Generating configured.bazelrc...")
    subprocess.check_call([sys.executable, CONFIGURE_BAZEL_PY])


def build_main_dist():
    """Builds the main distribution binaries.

    Additional packages that are installable as part of a full build and do not
    benefit from a more restricted build can be added here.
    """
    install_python_requirements()

    # Clean up install and build trees.
    shutil.rmtree(INSTALL_DIR, ignore_errors=True)
    remove_cmake_cache()

    # CMake configure.
    print("*** Configuring ***")
    subprocess.run(
        [
            sys.executable,
            CMAKE_CI_SCRIPT,
            f"-B{BUILD_DIR}",
            "--log-level=VERBOSE",
            f"-DCMAKE_INSTALL_PREFIX={INSTALL_DIR}",
            # On some distributions, this will install to lib64. We would like
            # consistency in built packages, so hard-code it.
            "-DCMAKE_INSTALL_LIBDIR=lib",
            f"-DCMAKE_BUILD_TYPE=Release",
            f"-DIREE_BUILD_COMPILER=ON",
            f"-DIREE_BUILD_PYTHON_BINDINGS=OFF",
            f"-DIREE_BUILD_SAMPLES=OFF",
            # cpuinfo is set to be removed and is problematic from an
            # installed/bundled library perspective.
            f"-DIREE_ENABLE_CPUINFO=OFF",
        ],
        check=True,
    )

    print("*** Building ***")
    subprocess.run(
        [
            sys.executable,
            CMAKE_CI_SCRIPT,
            "--build",
            BUILD_DIR,
            "--target",
            "all",
        ],
        check=True,
    )

    # TODO: Get proper dependency management on install targets so we don't
    # have to build all first.
    subprocess.run(
        [
            sys.executable,
            CMAKE_CI_SCRIPT,
            "--build",
            BUILD_DIR,
            "--target",
            "iree-install-dist-stripped",
        ],
        check=True,
    )

    print("*** Packaging ***")
    dist_entries = [
        "bin",
        "lib",
        "include",
    ]
    dist_archive = os.path.join(
        BINDIST_DIR,
        f"iree-dist{version_info['package-suffix']}"
        f"-{version_info['package-version']}"
        f"-{sysconfig.get_platform()}.tar.xz",
    )
    print(f"Creating archive {dist_archive}")
    os.makedirs(os.path.dirname(dist_archive), exist_ok=True)
    with tarfile.open(dist_archive, mode="w:xz") as tf:
        for entry in dist_entries:
            print(f"Adding entry: {entry}")
            tf.add(os.path.join(INSTALL_DIR, entry), arcname=entry, recursive=True)


def build_py_tf_compiler_tools_pkg():
    """Builds iree-install/python_packages/iree_tools_[tf, tflite] packages."""
    install_python_requirements()
    configure_bazel()

    # Clean up install and build trees.
    shutil.rmtree(INSTALL_DIR, ignore_errors=True)
    remove_cmake_cache()

    os.makedirs(BINDIST_DIR, exist_ok=True)

    for project in ["iree_tflite", "iree_tf"]:
        print(f"*** Building wheel for {project} ***")
        subprocess.run(
            [
                sys.executable,
                "-m",
                "pip",
                "wheel",
                os.path.join(TF_INTEGRATIONS_DIR, "python_projects", project),
            ],
            cwd=BINDIST_DIR,
            check=True,
        )


command = sys.argv[1]
if command == "main-dist":
    build_main_dist()
elif command == "py-tf-compiler-tools-pkg":
    build_py_tf_compiler_tools_pkg()
else:
    print(f"Unrecognized command: {command}")
