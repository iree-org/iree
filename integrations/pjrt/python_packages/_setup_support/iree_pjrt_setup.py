# Copyright 2023 The OpenXLA Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from distutils.command.install import install
from distutils.command.build import build as _build
from setuptools.command.build_py import build_py as _build_py

THIS_DIR = os.path.realpath(os.path.dirname(__file__))
IREE_PJRT_DIR = os.path.join(THIS_DIR, "..", "..")
VERSION_INFO_FILE = os.path.join(IREE_PJRT_DIR, "version_info.json")


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

# Parse some versions out of the project level requirements.txt
# so that we get our pins setup properly.
install_requires = []
requirements_path = Path(IREE_PJRT_DIR) / "requirements.txt"
with requirements_path.open() as requirements_txt:
    # Filter for just pinned versions.
    pin_pairs = [line.strip().split("==") for line in requirements_txt if "==" in line]
    pin_versions = dict(pin_pairs)
    print(f"requirements.txt pins: {pin_versions}")
    # Convert pinned versions to >= for install_requires.
    for pin_name in ("iree-compiler", "jaxlib"):
        pin_version = pin_versions[pin_name]
        install_requires.append(f"{pin_name}>={pin_version}")

# Force platform specific wheel.
# https://stackoverflow.com/questions/45150304
try:
    from wheel.bdist_wheel import bdist_wheel as _bdist_wheel

    class bdist_wheel(_bdist_wheel):
        def finalize_options(self):
            _bdist_wheel.finalize_options(self)
            self.root_is_pure = False

        def get_tag(self):
            python, abi, plat = _bdist_wheel.get_tag(self)
            # We don't contain any python extensions so are version agnostic
            # but still want to be platform specific.
            python, abi = "py3", "none"
            return python, abi, plat

except ImportError:
    bdist_wheel = None


# Force installation into platlib.
# Since this is a pure-python library with platform binaries, it is
# mis-detected as "pure", which fails audit. Usually, the presence of an
# extension triggers non-pure install. We force it here.
class platlib_install(install):
    def finalize_options(self):
        install.finalize_options(self)
        self.install_lib = self.install_platlib


def get_env_cmake_option(name: str, default_value: bool = False) -> str:
    svalue = os.getenv(name)
    if not svalue:
        svalue = "ON" if default_value else "OFF"
    return f"-D{name}={svalue}"


def get_env_cmake_list(name: str, default_value: str = "") -> str:
    svalue = os.getenv(name)
    if not svalue:
        if not default_value:
            return f"-U{name}"
        svalue = default_value
    return f"-D{name}={svalue}"


def add_env_cmake_setting(args, env_name: str, cmake_name=None) -> str:
    svalue = os.getenv(env_name)
    if svalue is not None:
        if not cmake_name:
            cmake_name = env_name
        args.append(f"-D{cmake_name}={svalue}")


# We need some directories to exist before setup.
def populate_built_package(abs_dir):
    """Makes sure that a directory and __init__.py exist.

    This needs to unfortunately happen before any of the build process
    takes place so that setuptools can plan what needs to be built.
    We do this for any built packages (vs pure source packages).
    """
    os.makedirs(abs_dir, exist_ok=True)
    with open(os.path.join(abs_dir, "__init__.py"), "wt"):
        pass


class PjrtPluginBuild(_build):
    def run(self):
        self.run_command("build_py")


class BaseCMakeBuildPy(_build_py):
    def run(self):
        self.build_default_configuration()
        # The super-class handles the pure python build.
        # It must come after we have populated package data (binaries)
        # above.
        super().run()

    def build_configuration(self, cmake_build_dir, extra_cmake_args=()):
        subprocess.check_call(["cmake", "--version"])

        cfg = os.getenv("IREE_CMAKE_BUILD_TYPE", "Release")

        # Build from source tree.
        os.makedirs(cmake_build_dir, exist_ok=True)
        print(f"CMake build dir: {cmake_build_dir}", file=sys.stderr)
        cmake_args = [
            "-GNinja",
            "--log-level=VERBOSE",
            "-DIREE_BUILD_COMPILER=OFF",
            "-DIREE_BUILD_SAMPLES=OFF",
            "-DIREE_BUILD_TESTS=OFF",
            "-DIREE_HAL_DRIVER_DEFAULTS=OFF",
            "-DPython3_EXECUTABLE={}".format(sys.executable),
            "-DPython_EXECUTABLE={}".format(sys.executable),
            "-DCMAKE_BUILD_TYPE={}".format(cfg),
            get_env_cmake_option("IREE_ENABLE_CPUINFO", "ON"),
        ] + list(extra_cmake_args)
        add_env_cmake_setting(cmake_args, "IREE_TRACING_PROVIDER")
        add_env_cmake_setting(cmake_args, "IREE_TRACING_PROVIDER_H")

        # These usually flow through the environment, but we add them explicitly
        # so that they show clearly in logs (getting them wrong can have bad
        # outcomes).
        add_env_cmake_setting(cmake_args, "CMAKE_OSX_ARCHITECTURES")
        add_env_cmake_setting(
            cmake_args, "MACOSX_DEPLOYMENT_TARGET", "CMAKE_OSX_DEPLOYMENT_TARGET"
        )

        # Only do a from-scratch configure if not already configured.
        cmake_cache_file = os.path.join(cmake_build_dir, "CMakeCache.txt")
        if not os.path.exists(cmake_cache_file):
            print(f"Configuring with: {cmake_args}", file=sys.stderr)
            subprocess.check_call(
                ["cmake", IREE_PJRT_DIR] + cmake_args, cwd=cmake_build_dir
            )
        else:
            print(f"Not re-configuring (already configured)", file=sys.stderr)

        # Build. Since we have restricted to just the runtime, build everything
        # so as to avoid fragility with more targeted selection criteria.
        subprocess.check_call(["cmake", "--build", "."], cwd=cmake_build_dir)
        print("Build complete.", file=sys.stderr)
