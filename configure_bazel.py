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

import platform
import os
import subprocess
import sys


def detect_unix_platform_config(bazelrc):
  # This is hoaky. Ideally, bazel had any kind of rational way of selecting
  # options from within its environment (key word: "rational"), but sadly, it
  # is unintelligible to mere mortals. Why should a build system have a way for
  # people to condition their build options on what compiler they are using
  # (without descending down the hole of deciphering what a Bazel toolchain is)?
  # All I want to do is set a couple of project specific warning options!

  if platform.system() == "Darwin":
    print(f"build --config=macos_clang", file=bazelrc)
    print(f"build:release --config=macos_clang_release", file=bazelrc)
  else:

    # If the user specified a CXX environment var, bazel will later respect that,
    # so we just see if it says "clang".
    cxx = os.environ.get("CXX")
    cc = os.environ.get("CC")
    if (cxx is not None and cc is None) or (cxx is None and cc is not None):
      print("WARNING: Only one of CXX or CC is set, which can confuse bazel. "
            "Recommend: set both appropriately (or none)")
    if cc is not None and cxx is not None:
      # Persist the variables.
      print(f"build --action_env CC=\"{cc}\"", file=bazelrc)
      print(f"build --action_env CXX=\"{cxx}\"", file=bazelrc)
    else:
      print(
          "WARNING: CC and CXX are not set, which can cause mismatches between "
          "flag configurations and compiler. Recommend setting them explicitly."
      )

    if cxx is not None and "clang" in cxx:
      print(
          f"Choosing generic_clang config because CXX is set to clang ({cxx})")
      print(f"build --config=generic_clang", file=bazelrc)
      print(f"build:release --config=generic_clang_release", file=bazelrc)
    else:
      print(f"Choosing generic_gcc config by default because no CXX set or "
            f"not recognized as clang ({cxx})")
      print(f"build --config=generic_gcc", file=bazelrc)
      print(f"build:release --config=generic_gcc_release", file=bazelrc)


def write_platform(bazelrc):
  if platform.system() == "Windows":
    print(f"build --config=msvc", file=bazelrc)
    print(f"build:release --config=msvc_release", file=bazelrc)
  else:
    detect_unix_platform_config(bazelrc)
  if not (platform.system() == "Darwin"):
    print("common --config=non_darwin", file=bazelrc)


if len(sys.argv) > 1:
  local_bazelrc = sys.argv[1]
else:
  local_bazelrc = os.path.join(os.path.dirname(__file__), "configured.bazelrc")
with open(local_bazelrc, "wt") as bazelrc:
  write_platform(bazelrc)

print("Wrote", local_bazelrc)
