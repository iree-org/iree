#!/usr/bin/python3

# Copyright 2020 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Build platform specific wheel files for the iree.runtime package.
# Built artifacts are per-platform and build out of the build tree.

from distutils.command.install import install
import json
import os
import platform
from setuptools import setup, find_namespace_packages

README = r"""
TensorFlow TFLite Compiler Tools
"""

exe_suffix = ".exe" if platform.system() == "Windows" else ""

# Setup and get version information.
THIS_DIR = os.path.realpath(os.path.dirname(__file__))
IREESRC_DIR = os.path.join(THIS_DIR, "..", "..", "..", "..")
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

setup(
    name=f"iree-tools-tflite{PACKAGE_SUFFIX}",
    version=f"{PACKAGE_VERSION}",
    author="The IREE Team",
    author_email="iree-discuss@googlegroups.com",
    license="Apache-2.0",
    description="IREE TFLite Compiler Tools",
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/openxla/iree",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.9",
    packages=find_namespace_packages(
        include=[
            "iree.tools.tflite",
            "iree.tools.tflite.*",
        ]
    ),
    package_data={
        "iree.tools.tflite": [
            f"iree-import-tflite{exe_suffix}",
        ],
    },
    entry_points={
        "console_scripts": [
            "iree-import-tflite = iree.tools.tflite.scripts.iree_import_tflite.__main__:main",
        ],
    },
    zip_safe=False,  # This package is fine but not zipping is more versatile.
)
