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


def normalize_path(p):
  if platform.system() == "Windows":
    # Sure. Good idea, bazel.
    return p.replace("\\", "/")
  return p


def write_platform(bazelrc):
  platform_config = "generic_clang"
  if platform.system() == "Windows":
    platform_config = "windows"
  print("build --config={}".format(platform_config), file=bazelrc)


def write_python_bin(bazelrc):
  python_bin = normalize_path(sys.executable)
  print("build --python_path=\"{}\"".format(python_bin), file=bazelrc)
  # IREE extension compilation requires PYTHON_BIN
  print("build --action_env PYTHON_BIN=\"{}\"".format(python_bin), file=bazelrc)
  # TensorFlow defines this one. No idea why.
  print(
      "build --action_env PYTHON_BIN_PATH=\"{}\"".format(python_bin),
      file=bazelrc)


def write_python_path(bazelrc):
  # For some reason, bazel doesn't always find the user site path, which
  # is typically where "pip install --user" libraries end up. Inject it.
  try:
    user_site = subprocess.check_output(
        [sys.executable, "-m", "site", "--user-site"]).decode("utf-8").strip()
    print("Found user site directory:", user_site)
  except OSError:
    print("Could not resolve user site directory")
    return
  print(
      "build --action_env PYTHONPATH=\"{}\"".format(normalize_path(user_site)),
      file=bazelrc)


local_bazelrc = os.path.join(os.path.dirname(__file__), "configured.bazelrc")
with open(local_bazelrc, "wt") as bazelrc:
  write_platform(bazelrc)
  write_python_bin(bazelrc)
  write_python_path(bazelrc)

print("Wrote", local_bazelrc)
