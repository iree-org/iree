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

# pylint: disable=missing-docstring
"""submodule_versions.

Synchronizes the tracked SUBMODULE_VERSIONS file with the submodule state
in git.

Typical usage:
--------------
Exporting current git submodule state to SUBMODULE_VERSIONS:
  Syntax: ./scripts/git/submodule_versions.py export

Importing versions in SUBMODULE_VERSIONS to git submodule state:
  Syntax: ./scripts/git/submodule_versions.py import

Checking whether SUBMODULE_VERSIONS and git state are in sync:
  Syntax: ./scripts/git/submodule_versions.py check
"""

import argparse
import os
import re
import sys

import utils

VERSIONS_FILE = "SUBMODULE_VERSIONS"


def get_submodule_versions(repo_dir):
  raw_status = utils.execute(["git", "submodule", "status"],
                             cwd=repo_dir,
                             silent=True,
                             capture_output=True).decode("UTF-8")
  status_lines = []
  for line in raw_status.splitlines():
    # Format is a status char followed by revision, space and path.
    m = re.match(r"""^.([0-9a-z]+)\s+([^\s]+)""", line)
    if m:
      # Output as just the commit hash followed by space and path.
      status_lines.append(m.group(1) + " " + m.group(2))
  return "\n".join(status_lines) + "\n"


def export_versions(repo_dir):
  current_versions = get_submodule_versions(repo_dir)
  versions_file_path = os.path.join(repo_dir, VERSIONS_FILE)
  print("*** Exporting current submodule versions to:", versions_file_path)
  with open(versions_file_path, "w", encoding="UTF-8") as f:
    f.write(current_versions)
  utils.execute(["git", "add", VERSIONS_FILE], cwd=repo_dir)


def parse_versions(versions_text):
  versions = dict()
  for line in versions_text.splitlines():
    comps = line.split(" ", maxsplit=2)
    if len(comps) != 2:
      continue
    versions[comps[1]] = comps[0]
  return versions


def get_diff_versions(repo_dir):
  current_versions = parse_versions(get_submodule_versions(repo_dir))
  with open(os.path.join(repo_dir, VERSIONS_FILE), "r", encoding="UTF-8") as f:
    written_versions = parse_versions(f.read())
  diff_versions = current_versions.items() ^ written_versions.items()
  return {
      k: (current_versions.get(k), written_versions.get(k))
      for k, _ in diff_versions
  }


def sync_and_update_submodules(repo_dir):
  print("*** Synchronizing/updating submodules")
  utils.execute(["git", "submodule", "sync"], cwd=repo_dir)
  utils.execute(["git", "submodule", "update"], cwd=repo_dir)


def import_versions(repo_dir):
  print("*** Importing versions to git submodule state")
  diff_versions = get_diff_versions(repo_dir)
  if not diff_versions:
    print("*** No submodule updates required")
    return
  for path, (current, written) in diff_versions.items():
    if current is None:
      print(("Warning: Submodule %s does not exist but is "
             "still in the version file") % (path,))
      continue
    if written is None:
      print("Warning: Submodule %s is not in the version file" % (current,))
      continue
    # Directly update the submodule commit hash in the index.
    # See: https://stackoverflow.com/questions/33514642
    command = ["git", "update-index", "--cacheinfo", "160000", written, path]
    print("Updating", path, "to", written)
    utils.execute(command, cwd=repo_dir)


def init_submodules(repo_dir):
  print("*** Initializing submodules")
  utils.execute(["git", "submodule", "init"], cwd=repo_dir)


def parallel_shallow_update_submodules(repo_dir):
  print("*** Making shallow clone of submodules")
  # TODO(gcmn) Figure out a way to quickly fetch submodules without relying on
  # target SHA being within 10000 commits of HEAD.
  magic_depth = 10000
  utils.execute([
      "git", "submodule", "update", "--jobs", "8", "--depth",
      str(magic_depth)
  ],
                cwd=repo_dir)


def check_submodule_versions(repo_dir):
  diff_versions = get_diff_versions(repo_dir)
  if diff_versions:
    print(
        "Submodule state differs from SUBMODULE_VERSIONS file. Run (and commit) one of:"
    )
    print(
        "  ./scripts/git/submodule_versions.py import # Use version in SUBMODULE_VERSIONS"
    )
    print(
        "  ./scripts/git/submodule_versions.py export # Use version in git state"
    )
    for k, (current, written) in diff_versions.items():
      print("%s : actual=%s written=%s" % (k, current, written))
    return False
  return True


def parse_arguments():
  parser = argparse.ArgumentParser()
  parser.add_argument("--repo", help="Repository root directory")
  parser.add_argument(
      "command", help="Command to run (show|import|export|check|init)")
  args = parser.parse_args()

  # Default repo path.
  if args.repo is None:
    args.repo = utils.find_git_toplevel()
  return args


def main(args):
  if args.command == "show":
    print(get_submodule_versions(args.repo))
  elif args.command == "export":
    sync_and_update_submodules(args.repo)
    export_versions(args.repo)
  elif args.command == "check":
    if not check_submodule_versions(args.repo):
      sys.exit(1)
  elif args.command == "import":
    import_versions(args.repo)
    sync_and_update_submodules(args.repo)
  elif args.command == "init":
    init_submodules(args.repo)
    # Redundant, since import_versions will only update if they differ,
    # but good to only print output about the import if it's actually
    # needed.
    if not check_submodule_versions(args.repo):
      print("Warning: git submodule state does not match SUBMODULE_VERSIONS. "
            "Using state in SUBMODULE_VERSIONS")
      import_versions(args.repo)
    parallel_shallow_update_submodules(args.repo)
  else:
    print("Unrecognized command:", args.command)
    sys.exit(1)


if __name__ == "__main__":
  main(parse_arguments())
