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
"""update_tf_llvm_submodules.

Updates the third_party/tensorflow and third_party/llvm-project submodules
to new commits. We have special conditions around these submodules since
upstream will only accept an llvm-project version that is sync'd with the
corresponding version that tensorflow depends on. In addition, some BUILD
files must be sync'd for the new version.

Typical usage:
  Syntax: ./git_scripts/update_tf_llvm_modules.py

  By default, this will update the tensorflow submodule to remote HEAD and
  update the llvm-project submodule to the corresponding version. It will
  also sync BUILD file changes as needed and export the version metadata.
"""

import argparse
import re
import os
import sys

import submodule_versions
import utils


def parse_arguments():
  parser = argparse.ArgumentParser()
  parser.add_argument("--repo", help="Repository root directory")
  parser.add_argument(
      "--tensorflow",
      help="Path to the tensorflow sources "
      "(default to third_party/tensorflow)",
      default=None)
  parser.add_argument(
      "--llvm",
      help="Path to the LLVM sources "
      "(defaults to third_party/llvm-project)",
      default=None)
  parser.add_argument(
      "--tensorflow_commit",
      help="Update TensorFlow to this commit (or 'KEEP', 'REMOTE')",
      default="REMOTE")
  parser.add_argument(
      "--llvm_commit",
      help="Update LLVM to this commit (or 'KEEP', 'REMOTE', 'TENSORFLOW')",
      default="TENSORFLOW")
  parser.add_argument(
      "--update_build_files",
      help="Updates the IREE LLVM build files from TensorFlow",
      type=utils.str2bool,
      nargs="?",
      default=False)
  args = parser.parse_args()

  # Default repo path.
  if args.repo is None:
    args.repo = utils.find_git_toplevel()

  # Set some defaults.
  if not args.tensorflow:
    args.tensorflow = os.path.join(args.repo, "third_party", "tensorflow")
  if not args.llvm:
    args.llvm = os.path.join(args.repo, "third_party", "llvm-project")
  return args


def main(args):
  print("IREE handy-dandy-LLVM-submodule-updater at your service...")
  print("  IREE Path :", args.repo)
  print("  LLVM Path :", args.llvm)
  print("  TensorFlow Path :", args.tensorflow)
  print("  Update Build files:", args.update_build_files)
  current_llvm_commit = get_commit(args.llvm)
  current_tensorflow_commit = get_commit(args.tensorflow)

  print("Current Commits: llvm =", current_llvm_commit, "tensorflow =",
        current_tensorflow_commit)

  # Update TensorFlow
  if args.tensorflow_commit == "KEEP":
    print("Not updating TensorFlow (--tensorflow_commit == 'KEEP')")
  else:
    print("\n*** Updating TensorFlow to", args.tensorflow_commit, "***")
    update_submodule(args.tensorflow, args.tensorflow_commit)
    stage_path(args.repo, "third_party/tensorflow")

  # Update LLVM.
  if args.llvm_commit == "TENSORFLOW":
    args.llvm_commit = find_tensorflow_llvm_commit(args.tensorflow)
    print("Found TensorFlow's LLVM commit:", args.llvm_commit)
    if args.update_build_files is None:
      print("Will update build files from TensorFlow",
            "because --update_build_files not specified")
      args.update_build_files = True
  if args.llvm_commit == "KEEP":
    print("Not updating LLVM (--llvm_commit == 'KEEP')")
  else:
    print("\n*** Updating LLVM to", args.llvm_commit, "***")
    update_submodule(args.llvm, args.llvm_commit)
    stage_path(args.repo, "third_party/llvm-project")

  # Update build files.
  if not args.update_build_files:
    print("Not updating build files (--update_build_files not specified)")
  else:
    print("\n*** Updating BUILD.bazel files ***")
    update_build_files_from_tensorflow(args.repo, args.tensorflow)

  # Export SUBMODULE_VERSIONS.
  print()  # Add line break.
  submodule_versions.export_versions(args.repo)


def get_commit(path, rev="HEAD"):
  return utils.execute(["git", "rev-parse", rev],
                       cwd=path,
                       silent=True,
                       capture_output=True).decode("ISO-8859-1").strip()


def update_submodule(path, commit, tracking="origin/master"):
  # Fetch.
  utils.execute(["git", "fetch"], cwd=path)
  # Determine commit.
  if commit == "REMOTE":
    commit = get_commit(path, rev=tracking)
    print("Resolved remote commit:", commit)

  # Rebase to commit (will fail if not fast-forward).
  utils.execute(["git", "checkout", commit], cwd=path)


def find_tensorflow_llvm_commit(tensorflow_path):
  # TensorFlow keeps its commit in workspace.bzl on a line like:
  # LLVM_COMMIT = "..."
  # Yeah. This is how we do it.
  workspace_path = os.path.join(tensorflow_path, "tensorflow", "workspace.bzl")
  pattern_text = r"""\s*LLVM_COMMIT\s*=\s*"(.+)"\s*"""
  pattern = re.compile(pattern_text, flags=re.MULTILINE)
  for line in open(workspace_path, "r", encoding="UTF-8"):
    m = re.match(pattern, line)
    if m:
      return m.group(1)

  print("ERROR: Could not find LLVM commit in %s." % workspace_path)
  print("Request an explicit commit via --llvm_commit (and file a bug)")
  print("Expected pattern match for:", pattern_text)
  sys.exit(1)


def update_build_files_from_tensorflow(repo_path, tensorflow_path):
  src_llvm_build = os.path.join(tensorflow_path, "third_party", "llvm",
                                "llvm.autogenerated.BUILD")
  # NOTE(laurenzo): These will probably move upstream.
  src_mlir_build = os.path.join(tensorflow_path, "third_party", "mlir", "BUILD")
  src_mlir_test_build = os.path.join(tensorflow_path, "third_party", "mlir",
                                     "test.BUILD")
  overlay_path = os.path.join(repo_path, "build_tools", "bazel",
                              "third_party_import", "llvm-project", "overlay")
  copy_text_file(repo_path, src_llvm_build,
                 os.path.join(overlay_path, "llvm", "BUILD.bazel"))
  copy_text_file(repo_path, src_mlir_build,
                 os.path.join(overlay_path, "mlir", "BUILD.bazel"))
  copy_text_file(repo_path, src_mlir_test_build,
                 os.path.join(overlay_path, "mlir", "test", "BUILD.bazel"))


def copy_text_file(repo_path, src_file, dst_file):
  print("+ cp %s %s" % (src_file, dst_file))
  with open(src_file, "r", encoding="UTF-8") as f:
    src_contents = f.read()

  if not os.path.exists(dst_file):
    print("WARNING: Destination file does not exist:", dst_file)
  with open(dst_file, "w", encoding="UTF-8") as f:
    f.write(src_contents)
  stage_path(repo_path, dst_file)


def stage_path(repo_path, to_stage):
  # TODO(laurenzo): Move to utils.py.
  utils.execute(["git", "add", to_stage], cwd=repo_path)


if __name__ == "__main__":
  main(parse_arguments())
