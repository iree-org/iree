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


def write_platform(bazelrc):
  platform_config = "generic_clang"
  if platform.system() == "Windows":
    platform_config = "msvc"
  print(f"build --config={platform_config}", file=bazelrc)
  print(f"build:release --config={platform_config}_release", file=bazelrc)
  if not (platform.system() == "Darwin"):
    print("common --config=non_darwin", file=bazelrc)


if len(sys.argv) > 1:
  local_bazelrc = sys.argv[1]
else:
  local_bazelrc = os.path.join(os.path.dirname(__file__), "configured.bazelrc")
with open(local_bazelrc, "wt") as bazelrc:
  write_platform(bazelrc)

print("Wrote", local_bazelrc)
