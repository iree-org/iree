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
#   IREE_HAL_DRIVER_VULKAN
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


def getenv_bool(key, default_value="OFF"):
    value = os.getenv(key, default_value)
    return value.upper() in ["ON", "1", "TRUE"]


def combine_dicts(*ds):
    result = {}
    for d in ds:
        result.update(d)
    return result


ENABLE_TRACY = getenv_bool("IREE_RUNTIME_BUILD_TRACY", "ON")
if ENABLE_TRACY:
    print(
        "*** Enabling Tracy instrumented runtime (disable with IREE_RUNTIME_BUILD_TRACY=OFF)",
        file=sys.stderr,
    )
else:
    print(
        "*** Tracy instrumented runtime not enabled (enable with IREE_RUNTIME_BUILD_TRACY=ON)",
        file=sys.stderr,
    )
ENABLE_TRACY_TOOLS = getenv_bool("IREE_RUNTIME_BUILD_TRACY_TOOLS")
if ENABLE_TRACY_TOOLS:
    print("*** Enabling Tracy tools (may error if missing deps)", file=sys.stderr)
else:
    print(
        "*** Tracy tools not enabled (enable with IREE_RUNTIME_BUILD_TRACY_TOOLS=ON)",
        file=sys.stderr,
    )


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
        if version.parse(pip.__version__) < version.parse("21.3"):
            print("ERROR: pip version >= 21.3 required")
            print("Upgrade: pip install pip --upgrade")
            sys.exit(2)


check_pip_version()

# We must do the intermediate installation to a fixed location that agrees
# between what we pass to setup() and cmake. So hard-code it here.
# Note that setup() needs a relative path (to the setup.py file).
# We keep the path short ('i' instead of 'install') for platforms like Windows
# that have file length limits.
SETUPPY_DIR = os.path.realpath(os.path.dirname(__file__))
CMAKE_INSTALL_DIR_REL = os.path.join("build", "i", "d")
CMAKE_INSTALL_DIR_ABS = os.path.join(SETUPPY_DIR, CMAKE_INSTALL_DIR_REL)
CMAKE_TRACY_INSTALL_DIR_REL = os.path.join("build", "i", "t")
CMAKE_TRACY_INSTALL_DIR_ABS = os.path.join(SETUPPY_DIR, CMAKE_TRACY_INSTALL_DIR_REL)

IREE_SOURCE_DIR = os.path.join(SETUPPY_DIR, "..")
# Note that setuptools always builds into a "build" directory that
# is a sibling of setup.py, so we just colonize a sub-directory of that
# by default.
BASE_BINARY_DIR = os.getenv(
    "IREE_RUNTIME_API_CMAKE_BUILD_DIR", os.path.join(SETUPPY_DIR, "build", "b")
)
IREE_BINARY_DIR = os.path.join(BASE_BINARY_DIR, "d")
IREE_TRACY_BINARY_DIR = os.path.join(BASE_BINARY_DIR, "t")
print(
    f"Running setup.py from source tree: "
    f"SOURCE_DIR = {IREE_SOURCE_DIR} "
    f"BINARY_DIR = {IREE_BINARY_DIR}",
    file=sys.stderr,
)

# Setup and get version information.
VERSION_INFO_FILE = os.path.join(IREE_SOURCE_DIR, "version_info.json")


def load_version_info():
    with open(VERSION_INFO_FILE, "rt") as f:
        return json.load(f)


def find_git_versions():
    revisions = {}
    try:
        revisions["IREE"] = (
            subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=IREE_SOURCE_DIR)
            .decode("utf-8")
            .strip()
        )
    except subprocess.SubprocessError as e:
        print(f"ERROR: Could not get IREE revision: {e}", file=sys.stderr)
    return revisions


def find_git_submodule_revision(submodule_path):
    try:
        data = (
            subprocess.check_output(
                ["git", "ls-tree", "HEAD", submodule_path], cwd=IREE_SOURCE_DIR
            )
            .decode("utf-8")
            .strip()
        )
        columns = re.split("\\s+", data)
        return columns[2]
    except Exception as e:
        print(
            f"ERROR: Could not get submodule revision for {submodule_path}" f" ({e})",
            file=sys.stderr,
        )
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


def maybe_nuke_cmake_cache(cmake_build_dir, cmake_install_dir):
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
    PYTHON_STAMP_FILE = os.path.join(cmake_build_dir, "python_stamp.txt")
    if os.path.exists(PYTHON_STAMP_FILE):
        with open(PYTHON_STAMP_FILE, "rt") as f:
            actual_stamp_contents = f.read()
            if actual_stamp_contents == expected_stamp_contents:
                # All good.
                return

    # Mismatch or not found. Clean it.
    cmake_cache_file = os.path.join(cmake_build_dir, "CMakeCache.txt")
    if os.path.exists(cmake_cache_file):
        print("Removing CMakeCache.txt because Python version changed", file=sys.stderr)
        os.remove(cmake_cache_file)

    # Also clean the install directory. This avoids version specific pileups
    # of binaries that can occur with repeated builds against different
    # Python versions.
    if os.path.exists(cmake_install_dir):
        print(
            f"Removing CMake install dir because Python version changed: "
            f"{cmake_install_dir}",
            file=sys.stderr,
        )
        shutil.rmtree(cmake_install_dir)

    # And write.
    with open(PYTHON_STAMP_FILE, "wt") as f:
        f.write(expected_stamp_contents)


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


def build_configuration(cmake_build_dir, cmake_install_dir, extra_cmake_args=()):
    subprocess.check_call(["cmake", "--version"])
    version_py_content = generate_version_py()
    print(f"Generating version.py:\n{version_py_content}", file=sys.stderr)

    cfg = os.getenv("IREE_CMAKE_BUILD_TYPE", "Release")
    strip_install = cfg == "Release"

    # Build from source tree.
    os.makedirs(cmake_build_dir, exist_ok=True)
    maybe_nuke_cmake_cache(cmake_build_dir, cmake_install_dir)
    print(f"CMake build dir: {cmake_build_dir}", file=sys.stderr)
    print(f"CMake install dir: {cmake_install_dir}", file=sys.stderr)
    cmake_args = [
        "-GNinja",
        "--log-level=VERBOSE",
        "-DIREE_BUILD_PYTHON_BINDINGS=ON",
        "-DIREE_BUILD_COMPILER=OFF",
        "-DIREE_BUILD_SAMPLES=OFF",
        "-DIREE_BUILD_TESTS=OFF",
        "-DPython3_EXECUTABLE={}".format(sys.executable),
        "-DCMAKE_BUILD_TYPE={}".format(cfg),
        get_env_cmake_option(
            "IREE_HAL_DRIVER_VULKAN",
            "OFF" if platform.system() == "Darwin" else "ON",
        ),
        get_env_cmake_list("IREE_EXTERNAL_HAL_DRIVERS", ""),
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
            ["cmake", IREE_SOURCE_DIR] + cmake_args, cwd=cmake_build_dir
        )
    else:
        print(f"Not re-configuring (already configured)", file=sys.stderr)

    # Build. Since we have restricted to just the runtime, build everything
    # so as to avoid fragility with more targeted selection criteria.
    subprocess.check_call(["cmake", "--build", "."], cwd=cmake_build_dir)
    print("Build complete.", file=sys.stderr)

    # Install the component we care about.
    install_args = [
        f"-DCMAKE_INSTALL_PREFIX={cmake_install_dir}/",
        f"-DCMAKE_INSTALL_COMPONENT=IreePythonPackage-runtime",
        "-P",
        os.path.join(cmake_build_dir, "cmake_install.cmake"),
    ]
    if strip_install:
        install_args.append("-DCMAKE_INSTALL_DO_STRIP=ON")
    # May have been deleted in a cleanup step.
    populate_built_package(
        os.path.join(
            cmake_install_dir,
            "python_packages",
            "iree_runtime",
            "iree",
            "_runtime_libs",
        )
    )
    print(f"Installing with: {install_args}", file=sys.stderr)
    subprocess.check_call(["cmake"] + install_args, cwd=cmake_build_dir)

    # Write version.py directly into install dir.
    version_py_file = os.path.join(
        cmake_install_dir,
        "python_packages",
        "iree_runtime",
        "iree",
        "_runtime_libs",
        "version.py",
    )
    os.makedirs(os.path.dirname(version_py_file), exist_ok=True)
    with open(version_py_file, "wt") as f:
        f.write(version_py_content)

    print(f"Installation prepared: {cmake_install_dir}", file=sys.stderr)


class CMakeBuildPy(_build_py):
    def run(self):
        # The super-class handles the pure python build.
        super().run()
        self.build_default_configuration()
        if ENABLE_TRACY:
            self.build_tracy_configuration()

    def build_default_configuration(self):
        print("*****************************", file=sys.stderr)
        print("* Building base runtime     *", file=sys.stderr)
        print("*****************************", file=sys.stderr)
        build_configuration(IREE_BINARY_DIR, CMAKE_INSTALL_DIR_ABS, extra_cmake_args=())
        # We only take the iree._runtime_libs from the default build.
        target_dir = os.path.join(
            os.path.abspath(self.build_lib), "iree", "_runtime_libs"
        )
        print(f"Building in target dir: {target_dir}", file=sys.stderr)
        os.makedirs(target_dir, exist_ok=True)
        print("Copying install to target.", file=sys.stderr)
        if os.path.exists(target_dir):
            shutil.rmtree(target_dir)
        shutil.copytree(
            os.path.join(
                CMAKE_INSTALL_DIR_ABS,
                "python_packages",
                "iree_runtime",
                "iree",
                "_runtime_libs",
            ),
            target_dir,
            symlinks=False,
        )
        print("Target populated.", file=sys.stderr)

    def build_tracy_configuration(self):
        print("*****************************", file=sys.stderr)
        print("* Building tracy runtime    *", file=sys.stderr)
        print("*****************************", file=sys.stderr)
        cmake_args = [
            "-DIREE_ENABLE_RUNTIME_TRACING=ON",
        ]
        if ENABLE_TRACY_TOOLS:
            cmake_args.append("-DIREE_BUILD_TRACY=ON")
        build_configuration(
            IREE_TRACY_BINARY_DIR,
            CMAKE_TRACY_INSTALL_DIR_ABS,
            extra_cmake_args=cmake_args,
        )
        # We only take the iree._runtime_libs from the default build.
        target_dir = os.path.join(
            os.path.abspath(self.build_lib), "iree", "_runtime_libs_tracy"
        )
        print(f"Building in target dir: {target_dir}", file=sys.stderr)
        os.makedirs(target_dir, exist_ok=True)
        print("Copying install to target.", file=sys.stderr)
        if os.path.exists(target_dir):
            shutil.rmtree(target_dir)
        shutil.copytree(
            os.path.join(
                CMAKE_TRACY_INSTALL_DIR_ABS,
                "python_packages",
                "iree_runtime",
                "iree",
                "_runtime_libs",
            ),
            target_dir,
            symlinks=False,
        )
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


packages = (
    find_namespace_packages(
        where=os.path.join(IREE_SOURCE_DIR, "runtime", "bindings", "python"),
        include=[
            "iree.runtime",
            "iree.runtime.*",
            "iree._runtime",
            "iree._runtime.*",
        ],
    )
    + [
        # Default libraries.
        "iree._runtime_libs",
    ]
    + (["iree._runtime_libs_tracy"] if ENABLE_TRACY else [])
)
print(f"Found runtime packages: {packages}")

with open(
    os.path.join(
        IREE_SOURCE_DIR, "runtime", "bindings", "python", "iree", "runtime", "README.md"
    ),
    "rt",
) as f:
    README = f.read()

custom_package_suffix = os.getenv("IREE_RUNTIME_CUSTOM_PACKAGE_SUFFIX", "")
custom_package_prefix = os.getenv("IREE_RUNTIME_CUSTOM_PACKAGE_PREFIX", "")


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


populate_built_package(
    os.path.join(
        CMAKE_INSTALL_DIR_ABS,
        "python_packages",
        "iree_runtime",
        "iree",
        "_runtime_libs",
    )
)
populate_built_package(
    os.path.join(
        CMAKE_TRACY_INSTALL_DIR_ABS,
        "python_packages",
        "iree_runtime",
        "iree",
        "_runtime_libs",
    )
)

setup(
    name=f"{custom_package_prefix}iree-runtime{custom_package_suffix}{PACKAGE_SUFFIX}",
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
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    url="https://github.com/openxla/iree",
    python_requires=">=3.9",
    ext_modules=(
        [
            CMakeExtension("iree._runtime_libs._runtime"),
        ]
        + (
            [CMakeExtension("iree._runtime_libs_tracy._runtime")]
            if ENABLE_TRACY
            else []
        )
    ),
    cmdclass={
        "build": CustomBuild,
        "built_ext": NoopBuildExtension,
        "build_py": CMakeBuildPy,
    },
    zip_safe=False,
    package_dir=combine_dicts(
        {
            # Note: Must be relative path, so we line this up with the absolute
            # path built above. Note that this must exist prior to the call.
            "iree.runtime": "bindings/python/iree/runtime",
            "iree._runtime": "bindings/python/iree/_runtime",
            "iree._runtime_libs": f"{CMAKE_INSTALL_DIR_REL}/python_packages/iree_runtime/iree/_runtime_libs",
        },
        {
            # Note that we do a switcheroo here by populating the
            # _runtime_libs_tracy package from the tracy-enabled build of
            # iree._runtime_libs. It is relocatable, and the Python side looks
            # for this stuff.
            "iree._runtime_libs_tracy": f"{CMAKE_TRACY_INSTALL_DIR_REL}/python_packages/iree_runtime/iree/_runtime_libs",
        }
        if ENABLE_TRACY
        else {},
    ),
    packages=packages,
    # Matching the native extension as a data file keeps setuptools from
    # "building" it (i.e. turning it into a static binary).
    package_data=combine_dicts(
        {
            "iree._runtime_libs": [
                f"*{sysconfig.get_config_var('EXT_SUFFIX')}",
                "iree-run-module*",
                "iree-run-trace*",
                "iree-benchmark-module*",
                "iree-benchmark-trace*",
                # These utilities are invariant wrt tracing and are only built for the default runtime.
                "iree-create-parameters*",
                "iree-convert-parameters*",
                "iree-dump-module*",
                "iree-dump-parameters*",
                "iree-cpuinfo*",
            ],
        },
        {
            "iree._runtime_libs_tracy": [
                f"*{sysconfig.get_config_var('EXT_SUFFIX')}",
                "iree-run-module*",
                "iree-run-trace*",
                "iree-benchmark-module*",
                "iree-benchmark-trace*",
            ]
            + (["iree-tracy-capture"] if ENABLE_TRACY_TOOLS else [])
        }
        if ENABLE_TRACY
        else {},
    ),
    entry_points={
        "console_scripts": [
            "iree-run-module = iree._runtime.scripts.iree_run_module.__main__:main",
            "iree-run-trace = iree._runtime.scripts.iree_run_trace.__main__:main",
            "iree-benchmark-module = iree._runtime.scripts.iree_benchmark_module.__main__:main",
            "iree-benchmark-trace = iree._runtime.scripts.iree_benchmark_trace.__main__:main",
            "iree-create-parameters = iree._runtime.scripts.iree_create_parameters.__main__:main",
            "iree-convert-parameters = iree._runtime.scripts.iree_convert_parameters.__main__:main",
            "iree-dump-module = iree._runtime.scripts.iree_dump_module.__main__:main",
            "iree-dump-parameters = iree._runtime.scripts.iree_dump_parameters.__main__:main",
            "iree-cpuinfo = iree._runtime.scripts.iree_cpuinfo.__main__:main",
        ]
        + (
            [
                "iree-tracy-capture = iree._runtime.scripts.iree_tracy_capture.__main__:main"
            ]
            if ENABLE_TRACY_TOOLS
            else []
        ),
    },
    install_requires=[
        "numpy",
        "PyYAML",
    ],
)
