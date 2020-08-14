# Copyright 2019 Google LLC
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

# Generates the BUILD file for the python package.
# It includes special comment lines that instruct the hosting program
# on how to setup the filesystem.
# Debugging hint: Just runt his with python to see what it prints.
"""Generates a bazel BUILD file for the repo."""

import json
import os
import sys

from distutils import sysconfig

extra_srcs = []
exec_prefix = sys.base_exec_prefix

# Print some directives for the calling starlark program.
print("# SYMLINK: {abs}\t{ws}".format(
    abs=sysconfig.get_python_inc(), ws="include"))

# If running on Windows, find the import library, which is named
# libs/pythonXY.lib (where (X, Y) == (major, minor)).
# See: https://docs.python.org/3/extending/windows.html
# Note that while not strictly a "header" as the rule name implies,
# this is integral to linking on Windows and parsing the header
# will require it, so it is included.
if os.name == "nt":
  implib_basename = "python{major}{minor}.lib".format(
      major=sys.version_info[0], minor=sys.version_info[1])
  implib_abs_path = os.path.join(exec_prefix, "libs", implib_basename)
  if not os.path.exists(implib_abs_path):
    raise RuntimeError("Could not find Windows python import library: %s" %
                       (implib_abs_path,))
  implib_ws_path = "libs/" + implib_basename
  print("# SYMLINK: {abs}\t{ws}".format(abs=implib_abs_path, ws=implib_ws_path))
  extra_srcs.append(implib_ws_path)

print("""
package(default_visibility = ["//visibility:public"])

config_setting(
    name = "config_windows_any",
    constraint_values = [
        "@platforms//os:windows",
    ],
)

cc_library(
    name = "python_headers",
    hdrs = glob(["include/**/*.h"]),
    srcs = [{extra_srcs}],
    includes = ["include"],
    linkopts = [],
)

""".format(extra_srcs=",".join([json.dumps(s) for s in extra_srcs]),))
