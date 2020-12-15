#!/usr/bin/python3

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

# Build platform specific wheel files for the pyiree.rt package.
# Built artifacts are per-platform and build out of the build tree.

import os
import subprocess
import sys

# Make this setup position independent and make it not conflict with
# parallel scripts.
this_dir = os.path.abspath(os.path.dirname(__file__))


def run_sub_setup(name):
  sub_path = os.path.join(this_dir, f"{name}.py")
  args = [sys.executable, sub_path] + sys.argv[1:]
  print(f"##### Running sub setup: {' '.join(args)}")
  subprocess.run(args, check=True)
  print("")


run_sub_setup("setup_compiler")
run_sub_setup("setup_runtime")
run_sub_setup("setup_tools_core")
if os.path.exists(os.path.join(this_dir, "pyiree/tools/tf")):
  run_sub_setup("setup_tools_tf")
