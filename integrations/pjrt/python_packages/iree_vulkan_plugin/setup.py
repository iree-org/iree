#!/usr/bin/python3

# Copyright 2023 The OpenXLA Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Early splice the _setup_support directory onto the python path.
import os
from pathlib import Path
import sys

THIS_DIR = os.path.realpath(os.path.dirname(__file__))
sys.path.insert(0, os.path.join(THIS_DIR, "..", "_setup_support"))

import iree_pjrt_setup
from setuptools import setup, find_namespace_packages

README = r"""
OpenXLA PJRT Plugin for Vulkan
"""

# Setup and get version information.
CMAKE_BUILD_DIR_ABS = os.path.join(THIS_DIR, "build", "cmake")


class CMakeBuildPy(iree_pjrt_setup.BaseCMakeBuildPy):
    def build_default_configuration(self):
        print("*****************************", file=sys.stderr)
        print("* Building base runtime     *", file=sys.stderr)
        print("*****************************", file=sys.stderr)
        self.build_configuration(
            os.path.join(THIS_DIR, "build", "cmake"),
            extra_cmake_args=("-DIREE_HAL_DRIVER_VULKAN=ON",),
        )
        print("Target populated.", file=sys.stderr)


iree_pjrt_setup.populate_built_package(
    os.path.join(
        CMAKE_BUILD_DIR_ABS,
        "python",
        "iree",
        "_pjrt_libs",
        "vulkan",
    )
)


setup(
    name=f"iree-pjrt-plugin-vulkan{iree_pjrt_setup.PACKAGE_SUFFIX}",
    version=f"{iree_pjrt_setup.PACKAGE_VERSION}",
    author="The IREE Team",
    author_email="iree-discuss@googlegroups.com",
    license="Apache-2.0",
    description="IREE PJRT Plugin for Vulkan (generic)",
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/openxla/iree",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
    ],
    packages=[
        "jax_plugins.iree_vulkan",
        "iree._pjrt_libs.vulkan",
    ],
    package_dir={
        "jax_plugins.iree_vulkan": "jax_plugins/iree_vulkan",
        "iree._pjrt_libs.vulkan": "build/cmake/python/iree/_pjrt_libs/vulkan",
    },
    package_data={
        "iree._pjrt_libs.vulkan": ["pjrt_plugin_iree_vulkan.*"],
    },
    cmdclass={
        "build": iree_pjrt_setup.PjrtPluginBuild,
        "build_py": CMakeBuildPy,
        "bdist_wheel": iree_pjrt_setup.bdist_wheel,
        "install": iree_pjrt_setup.platlib_install,
    },
    zip_safe=False,  # Needs to reference embedded shared libraries.
    entry_points={
        # We must advertise which Python modules should be treated as loadable
        # plugins. This augments the path based scanning that Jax does, which
        # is not always robust to all packaging circumstances.
        "jax_plugins": [
            "iree-vulkan = jax_plugins.iree_vulkan",
        ],
    },
    install_requires=iree_pjrt_setup.install_requires,
)
