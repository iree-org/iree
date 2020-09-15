#!/usr/bin/env python3
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
"""Updates LLVM-dependent submodules based on the current LLVM commit.

Updates the third_party/tensorflow submodule to a new commit based on the commit
in the third_party/llvm-project submodule. We have special conditions around
these submodules since they are synced as part of the integration of LLVM into
Google's source repository. See
https://google.github.io/iree/developing-iree/repository-management#the-special-relationship-with-llvm-and-tensorflow.

In addition we currently copy LLVM Bazel BUILD files from TensorFlow.

Typical usage:
  Syntax: ./scripts/git/update_to_llvm_syncpoint.py

  By default, this will update the TensorFlow submodule to the most recent
  commit with an LLVM version that matches IREE's and copy over the LLVM
  BUILD file changes as needed.
"""

import argparse
import re
import os
import sys

import submodule_versions
import utils

REMOTE_HEAD_COMMIT = "REMOTE"
KEEP_COMMIT = "KEEP"
INTEGRATE_COMMIT = "INTEGRATE"
LATEST_MATCHING_COMMIT = "LATEST_MATCH"

COMMIT_OPTIONS = {
    REMOTE_HEAD_COMMIT:
        "Update to the HEAD commit on the remote repository default branch",
    KEEP_COMMIT:
        "Do not modify the current commit",
    INTEGRATE_COMMIT:
        "Update to the commit where the current version of LLVM was first integrated",
    LATEST_MATCHING_COMMIT:
        "Update to the most recent commit with a matching version of LLVM",
}


def parse_arguments():
  parser = argparse.ArgumentParser()
  parser.add_argument("--repo", help="Repository root directory")
  parser.add_argument("--tensorflow",
                      help="Path to the tensorflow sources "
                      "(default to third_party/tensorflow)",
                      default=None)
  parser.add_argument("--llvm",
                      help="Path to the LLVM sources "
                      "(defaults to third_party/llvm-project)",
                      default=None)
  parser.add_argument(
      "--tensorflow_commit",
      "--tf_commit",
      help=
      f"Update TensorFlow to this commit, or a named option: {COMMIT_OPTIONS}",
      default=LATEST_MATCHING_COMMIT)
  parser.add_argument(
      "--validate",
      help="Validate that the selected commits all match the LLVM commit",
      type=utils.str2bool,
      nargs="?",
      default=True,
  )

  parser.add_argument("--update_build_files",
                      help="Updates the IREE LLVM build files from TensorFlow.",
                      type=utils.str2bool,
                      nargs="?",
                      default=True)
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
  new_tf_commit = find_new_tf_commit(args.tensorflow, current_llvm_commit,
                                     args.tensorflow_commit)
  print("\n*** Updating TensorFlow to", new_tf_commit, "***")
  utils.execute(["git", "checkout", new_tf_commit], cwd=args.tensorflow)
  stage_path(args.repo, args.tensorflow)

  validate_tf_commit(current_llvm_commit,
                     args.tensorflow,
                     exit_on_failure=args.validate)

  if args.update_build_files:
    print("\n*** Updating BUILD.bazel files ***")
    update_build_files_from_tensorflow(args.repo, args.tensorflow)

  # Export SUBMODULE_VERSIONS.
  print()  # Add line break.
  submodule_versions.export_versions(args.repo)


def get_commit(path, rev="HEAD"):
  return utils.execute(["git", "rev-parse", rev],
                       cwd=path,
                       silent=True,
                       capture_output=True,
                       universal_newlines=True).strip()


def find_new_tf_commit(tensorflow_path, llvm_commit, tf_commit):
  utils.execute(["git", "fetch"], cwd=tensorflow_path)

  if tf_commit not in COMMIT_OPTIONS:
    return get_commit(tensorflow_path, rev=tf_commit)

  if tf_commit == KEEP_COMMIT:
    return get_commit(tensorflow_path)

  if tf_commit == REMOTE_HEAD_COMMIT:
    return get_commit(tensorflow_path, "origin/master")

  tf_integrate_commits = utils.execute([
      "git", "log", "--first-parent", "--format=%H", "-S", llvm_commit,
      "origin/master", "--", "tensorflow/workspace.bzl"
  ],
                                       capture_output=True,
                                       universal_newlines=True,
                                       cwd=tensorflow_path).split()
  if len(tf_integrate_commits) > 2:
    raise RuntimeError(
        f"Expected one or two TF commits to involve LLVM commit {llvm_commit},"
        f" but got {len(tf_integrate_commits)}")

  if not tf_integrate_commits:
    raise RuntimeError(
        f"TF does not have any references to LLVM commit {llvm_commit}."
        " Maybe TF export is behind?")

  if tf_commit == INTEGRATE_COMMIT:
    return tf_integrate_commits[-1]

  assert tf_commit == LATEST_MATCHING_COMMIT
  if len(tf_integrate_commits) == 1:
    # There hasn't been a subsequent integrate, use remote head.
    return get_commit(tensorflow_path, "origin/master")

  return get_commit(tensorflow_path, rev=f"{tf_integrate_commits[0]}^")


def validate_tf_commit(llvm_commit, tensorflow_path, exit_on_failure=True):
  tf_llvm_commit = find_tensorflow_llvm_commit(tensorflow_path)

  matches = tf_llvm_commit == llvm_commit
  if not matches:
    print("WARNING: LLVM commit in TF  does not match that in IREE"
          f" ({tf_llvm_commit} vs {llvm_commit})")
    if exit_on_failure:
      sys.exit(1)


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

  print(f"ERROR: Could not find LLVM commit in {workspace_path}.")
  print("Please file a bug)")
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
  print(f"+ cp {src_file} {dst_file}")
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
