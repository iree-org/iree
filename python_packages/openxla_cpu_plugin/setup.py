#!/usr/bin/python3

# Copyright 2023 The OpenXLA Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from distutils.command.install import install
import json
import os
from pathlib import Path
from setuptools import setup, find_namespace_packages

README = r"""
OpenXLA PJRT Plugin for CPU (generic)
"""

# Setup and get version information.
THIS_DIR = os.path.realpath(os.path.dirname(__file__))
REPO_DIR = os.path.join(THIS_DIR, "..", "..")
VERSION_INFO_FILE = os.path.join(REPO_DIR, "version_info.json")


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
install_requires=[]
requirements_path = Path(REPO_DIR) / "requirements.txt"
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
      python, abi = 'py3', 'none'
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


packages = find_namespace_packages(include=[
        "jax_plugins.openxla_cpu",
        "jax_plugins.openxla_cpu.*",
    ])

setup(
    name=f"openxla-pjrt-plugin-cpu{PACKAGE_SUFFIX}",
    version=f"{PACKAGE_VERSION}",
    author="The OpenXLA Team",
    author_email="openxla-discuss@googlegroups.com",
    license="Apache-2.0",
    description="OpenXLA PJRT Plugin for CPUs (generic)",
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/openxla/openxla-pjrt-plugin",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
    ],
    packages=packages,
    package_data={
        "jax_plugins.openxla_cpu": ["pjrt_plugin_iree_cpu.so"],
    },
    cmdclass={
        "bdist_wheel": bdist_wheel,
        "install": platlib_install,
    },
    zip_safe=False,  # Needs to reference embedded shared libraries.
    entry_points={
        # We must advertise which Python modules should be treated as loadable
        # plugins. This augments the path based scanning that Jax does, which
        # is not always robust to all packaging circumstances.
        "jax_plugins": [
          "openxla-cpu = jax_plugins.openxla_cpu",
        ],
    },
    install_requires=install_requires,
)
