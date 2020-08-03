#!/usr/bin/python
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

# Given a runfiles directory from a bazel build, does surgery to extract
# a usable python package directory. In addition to the bazel directory
# structure being unnecessarily obtuse, it is also really hard to actually
# name files correctly. This affects python extension modules which must be
# named with a specific extension suffix. Bazel is extremely unflexible and
# we patch around it with this script. For the record, there are various ways
# to write custom rules to do this more natively, but it is all complicated
# and needless complexity. We opt for a script that is at least readable by
# mere mortals and in one place.
# Usage:
#   ./this_script <dest_dir> <path to bazel-bin>

import os
import platform
import shutil
import sys
import sysconfig

FILE_NAME_MAP = {
    "binding.so": "binding{}".format(sysconfig.get_config_var("EXT_SUFFIX")),
    "binding.pyd": False,
    "binding.dylib": False,
}


def get_exe_suffix():
  if platform.system() == "Windows":
    return ".exe"
  else:
    return ""


def copy_prefix(dest_dir, runfiles_dir, prefix):
  # And finally seek into the corresponding path in the runfiles dir.
  # Aren't bazel paths fun???
  # Note that the "iree_core" path segment corresponds to the workspace name.
  pkg_dir = os.path.join(runfiles_dir, "iree_core", *prefix)
  if not os.path.exists(pkg_dir):
    return
  dest_dir = os.path.join(dest_dir)
  for root, dirs, files in os.walk(pkg_dir):
    assert root.startswith(pkg_dir)
    dest_prefix = root[len(pkg_dir):]
    if dest_prefix.startswith(os.path.sep):
      dest_prefix = dest_prefix[1:]
    local_dest_dir = os.path.join(dest_dir, dest_prefix)
    os.makedirs(local_dest_dir, exist_ok=True)
    for file in files:
      copy_file(os.path.join(root, file), local_dest_dir)


def copy_file(src_file, dst_dir):
  basename = os.path.basename(src_file)
  dst_file = os.path.join(dst_dir, basename)
  mapped_name = FILE_NAME_MAP.get(basename)
  if mapped_name is False:
    # Skip.
    return
  elif mapped_name is not None:
    dst_file = os.path.join(dst_dir, mapped_name)
  shutil.copyfile(src_file, dst_file, follow_symlinks=True)


def main():
  # Parse args.
  dest_dir = sys.argv[1]
  bazel_bin = sys.argv[2] if len(sys.argv) > 2 else os.path.join(
      os.path.dirname(__file__), "..", "..", "bazel-bin")

  # Find the path to the runfiles of the built target:
  #   //bindings/python/packaging:all_pyiree_packages
  runfiles_dir = os.path.join(
      bazel_bin, "packaging", "python",
      "all_pyiree_packages%s.runfiles" % (get_exe_suffix(),))
  if not os.path.isdir(runfiles_dir):
    print("ERROR: Could not find build target 'all_pyiree_packages':",
          runfiles_dir)
    print("Make sure to build target", "//packaging/python:all_pyiree_packages")
    sys.exit(1)

  copy_prefix(dest_dir, runfiles_dir, ("bindings", "python"))
  copy_prefix(dest_dir, runfiles_dir,
              ("integrations", "tensorflow", "bindings", "python"))


if __name__ == "__main__":
  main()
