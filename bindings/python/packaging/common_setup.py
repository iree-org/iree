# Copyright 2020 Google LLC
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

import os
import platform
import setuptools
import sys
from datetime import date


def get_package_dir():
  if "CMAKE_BUILD_ROOT" in os.environ:
    cmake_build_root = os.environ["CMAKE_BUILD_ROOT"]
    print("Using CMake build root:", cmake_build_root)
    pkg_dir = os.path.join(cmake_build_root, "bindings", "python")
  else:
    print("ERROR: No build directory specified. Set one of these variables:")
    print("  CMAKE_BUILD_ROOT=/path/to/cmake/build")
    sys.exit(1)

  if not os.path.exists(pkg_dir):
    print("ERROR: Package path does not exist:", pkg_dir)
    sys.exit(1)
  return pkg_dir


def get_default_date_version():
  today = date.today()
  return today.strftime("%Y%m%d")


def get_setup_defaults(sub_project, description):
  return {
      "name": "google-iree-%s-pkg" % (sub_project,),
      "version": get_default_date_version(),
      "author": "The IREE Team at Google",
      "author_email": "iree-discuss@googlegroups.com",
      "description": description,
      "long_description": description,
      "long_description_content_type": "text/plain",
      "url": "https://github.com/google/iree",
      "package_dir": {
          "": get_package_dir()
      },
      "classifiers": [
          "Programming Language :: Python :: 3",
          "License :: OSI Approved :: Apache License",
          "Operating System :: OS Independent",
          "Development Status :: 3 - Alpha",
      ],
      "python_requires": ">=3.6",
  }


def get_native_file_extension():
  if platform.system() == "Windows":
    return "pyd"
  elif platform.system() == "Darwin":
    return "dylib"
  else:
    return "so"


def setup(**kwargs):
  # See: https://stackoverflow.com/q/45150304
  try:
    from wheel.bdist_wheel import bdist_wheel as _bdist_wheel

    class bdist_wheel(_bdist_wheel):

      def finalize_options(self):
        _bdist_wheel.finalize_options(self)
        self.root_is_pure = False
  except ImportError:
    bdist_wheel = None

  # Need to include platform specific extensions binaries:
  #  Windows: .pyd
  #  MacOS: .dylib
  #  Other: .so
  # Unfortunately, bazel is imprecise and scatters .so files around, so
  # need to be specific.
  package_data = {
      "": ["*.%s" % (get_native_file_extension(),)],
  }
  setuptools.setup(
      package_data=package_data,
      cmdclass={"bdist_wheel": bdist_wheel},
      **kwargs)
