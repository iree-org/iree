#!/usr/bin/env python3
# Lint as: python3
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

import argparse
import os
import re
import subprocess
import sys


VERSIONS_FILE = ".gitmoduleversions"


def get_submodule_versions(repo_dir):
  raw_status = subprocess.check_output(["git", "submodule", "status"], cwd=repo_dir).decode("UTF-8")
  status_lines = []
  for line in raw_status.splitlines():
    # Format is a status char followed by revision, space and path.
    m = re.match(r"""^.([0-9a-z]+)\s+([^\s]+)""", line)
    if m:
      # Output as just the commit hash followed by space and path.
      status_lines.append(m.group(1) + " " + m.group(2))
  return "\n".join(status_lines) + "\n"


def write_submodule_versions(repo_dir):
  current_versions = get_submodule_versions(repo_dir)
  with open(os.path.join(repo_dir, VERSIONS_FILE), "w", encoding="UTF-8") as f:
    f.write(current_versions)


def parse_versions(versions_text):
  versions = dict()
  for line in versions_text.splitlines():
    comps = line.split(" ", maxsplit=2)
    if len(comps) != 2: continue
    versions[comps[1]] = comps[0]
  return versions


def get_diff_versions(repo_dir):
  current_versions = parse_versions(get_submodule_versions(repo_dir))
  with open(os.path.join(repo_dir, VERSIONS_FILE), "r", encoding="UTF-8") as f:
    written_versions = parse_versions(f.read())
  diff_versions = current_versions.items() ^ written_versions.items()
  return {
    k:(current_versions.get(k), written_versions.get(k)) for k, _ in diff_versions
  }


def update_versions(repo_dir):
  diff_versions = get_diff_versions(repo_dir)
  if not diff_versions:
    print("No submodule updates required")
    return
  for path, (current, written) in diff_versions.items():
    if current is None:
      print(("Warning: Submodule %s does not exist but is "
             "still in the version file") % (path,))
      continue
    if written is None:
      print("Warning: Submodule %s is not in the version file" % (current,))
      continue
    command = ["git", "update-index", "--cacheinfo", "160000", written, path]
    print("Updating", path, "to", written)
    print("+", " ".join(command))
    subprocess.check_call(command, cwd=repo_dir)
    subprocess.check_call(["git", "submodule", "update", path], cwd=repo_dir)


def check_submodule_versions(repo_dir):
  diff_versions = get_diff_versions(repo_dir) 
  if diff_versions:
    print("Submodule versions need to be written. Run (and update commit):")
    print("  ./build_tools/scripts/submodule_util write")
    for k, (current, written) in diff_versions.items():
      print("%s : actual=%s written=%s" % (k, current, written))
      return False
  return True


def parse_arguments():
  parser = argparse.ArgumentParser()
  parser.add_argument("--repo", help="Repository root directory")
  parser.add_argument("command", help="Command to run (show|write|check)")
  args = parser.parse_args()

  # Default repo path.
  if args.repo is None:
    args.repo = subprocess.check_output(["git", "rev-parse", "--show-toplevel"],
                        cwd=os.path.dirname(__file__)).strip().decode("UTF-8")
  return args


def main(args):
  if args.command == "show":
    print(get_submodule_versions(args.repo))
  elif args.command == "write":
    write_submodule_versions(args.repo)
  elif args.command == "check":
    if not check_submodule_versions(args.repo):
      sys.exit(1)
  elif args.command == "update":
    update_versions(args.repo)
  else:
    print("Unrecognized command:", args.command)
    sys.exit(1)


if __name__ == "__main__":
  main(parse_arguments())
