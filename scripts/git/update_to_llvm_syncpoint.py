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

Updates the third_party/llvm-bazel and third_party/tensorflow submodules to
commits that match the LLVM commit in third_party/llvm-project submodule. We
have special conditions around these submodules since they are synced as part of
the integration of LLVM into Google's source repository. See
https://google.github.io/iree/developing-iree/repository-management#the-special-relationship-with-llvm-and-tensorflow.

Typical usage:
  Syntax: ./scripts/git/update_to_llvm_syncpoint.py

  By default, this will update llvm-bazel to the tag corresponding to the
  current LLVM commit and update TensorFlow to the most recent commit that has a
  matching LLVM commit.
"""

import argparse
import re
import os
import sys

import submodule_versions
import utils

REMOTE_HEAD_COMMIT_OPTION = "REMOTE"
KEEP_COMMIT_OPTION = "KEEP"
INTEGRATE_COMMIT_OPTION = "INTEGRATE"
LATEST_MATCHING_COMMIT_OPTION = "LATEST_MATCH"

COMMIT_OPTIONS = {
    REMOTE_HEAD_COMMIT_OPTION:
        "Update to the HEAD commit on the remote repository default branch",
    KEEP_COMMIT_OPTION:
        "Do not modify the current commit",
    INTEGRATE_COMMIT_OPTION:
        "Update to the commit where the current version of LLVM was first "
        "integrated",
    LATEST_MATCHING_COMMIT_OPTION:
        "Update to the most recent commit with a matching version of LLVM",
}


def parse_arguments():
  parser = argparse.ArgumentParser()
  parser.add_argument("--repo", help="Repository root directory")
  parser.add_argument("--llvm_path",
                      help="Path to the LLVM sources "
                      "(defaults to third_party/llvm-project)",
                      default=None)
  parser.add_argument("--llvm_bazel_path",
                      help="Path to the LLVM Bazel BUILD files"
                      "(defaults to third_party/llvm-bazel)",
                      default=None)
  parser.add_argument(
      "--llvm_bazel_rev",
      "--llvm_bazel_commit",
      help=("Update llvm-bazel to this git rev, or a named option:"
            f" {COMMIT_OPTIONS}."
            f" {LATEST_MATCHING_COMMIT_OPTION} and {INTEGRATE_COMMIT_OPTION}"
            " are equivalentfor this repository."),
      default=LATEST_MATCHING_COMMIT_OPTION)
  parser.add_argument("--tensorflow_path",
                      help="Path to the tensorflow sources "
                      "(default to third_party/tensorflow)",
                      default=None)
  parser.add_argument("--tensorflow_rev",
                      "--tf_rev",
                      "--tensorflow_commit",
                      "--tf_commit",
                      help=("Update TensorFlow to this rev, or a named option:"
                            f" {COMMIT_OPTIONS}"),
                      default=LATEST_MATCHING_COMMIT_OPTION)
  parser.add_argument(
      "--validate",
      help="Validate that the selected commits all match the LLVM commit",
      type=utils.str2bool,
      nargs="?",
      default=True,
  )

  args = parser.parse_args()

  # Default repo path.
  if args.repo is None:
    args.repo = utils.find_git_toplevel()

  # Set some defaults.
  if not args.tensorflow_path:
    args.tensorflow_path = os.path.join(args.repo, "third_party", "tensorflow")
  if not args.llvm_path:
    args.llvm_path = os.path.join(args.repo, "third_party", "llvm-project")
  if not args.llvm_bazel_path:
    args.llvm_bazel_path = os.path.join(args.repo, "third_party", "llvm-bazel")

  return args


def main(args):
  print("IREE handy-dandy-LLVM-submodule-updater at your service...")
  print(f"  IREE Path: {args.repo}")
  print(f"  LLVM Path: {args.llvm_path}")
  print(f"  LLVM Bazel Path: {args.llvm_bazel_path}")
  print(f"  TensorFlow Path: {args.tensorflow_path}")

  current_llvm_commit = parse_rev(args.llvm_path, "HEAD")
  current_llvm_bazel_commit = parse_rev(args.llvm_bazel_path, "HEAD")
  current_tf_commit = parse_rev(args.tensorflow_path, "HEAD")
  print("Current Commits:")
  print(f"  llvm = {current_llvm_commit}")
  print(f"  llvm_bazel = {current_llvm_bazel_commit}")
  print(f"  tensorflow = {current_tf_commit}")

  # Update TensorFlow
  new_tf_commit = find_new_tf_commit(args.tensorflow_path, current_llvm_commit,
                                     args.tensorflow_rev)
  print("\n*** Updating TensorFlow to", new_tf_commit, "***")
  utils.execute(["git", "checkout", new_tf_commit], cwd=args.tensorflow_path)
  stage_path(args.repo, args.tensorflow_path)

  validate_tf_commit(current_llvm_commit,
                     args.tensorflow_path,
                     exit_on_failure=args.validate)

  # Update LLVM-Bazel
  new_llvm_bazel_commit = find_new_llvm_bazel_commit(args.llvm_bazel_path,
                                                     current_llvm_commit,
                                                     args.llvm_bazel_rev)
  print(f"\n*** Updating LLVM Bazel to {new_llvm_bazel_commit} ***")
  utils.execute(["git", "checkout", new_llvm_bazel_commit],
                cwd=args.llvm_bazel_path)
  stage_path(args.repo, args.llvm_bazel_path)

  validate_llvm_bazel_commit(current_llvm_commit,
                             args.llvm_bazel_path,
                             exit_on_failure=args.validate)

  # Export SUBMODULE_VERSIONS.
  print()  # Add line break.
  submodule_versions.export_versions(args.repo)


def parse_rev(path, rev):
  return utils.execute(["git", "rev-parse", rev],
                       cwd=path,
                       silent=True,
                       capture_output=True).stdout.strip()


def find_new_llvm_bazel_commit(llvm_bazel_path, llvm_commit, llvm_bazel_rev):
  # Explicitly force-fetch tags. Tags in llvm-bazel are not guaranteed to be
  # stable.
  utils.execute(["git", "fetch", "--tags", "--force"], cwd=llvm_bazel_path)

  if llvm_bazel_rev not in COMMIT_OPTIONS:
    return parse_rev(llvm_bazel_path, llvm_bazel_rev)

  if llvm_bazel_rev == KEEP_COMMIT_OPTION:
    return parse_rev(llvm_bazel_path, "HEAD")

  if llvm_bazel_rev == REMOTE_HEAD_COMMIT_OPTION:
    return parse_rev(llvm_bazel_path, "origin/main")

  if (llvm_bazel_rev == INTEGRATE_COMMIT_OPTION or
      llvm_bazel_rev == LATEST_MATCHING_COMMIT_OPTION):
    return parse_rev(llvm_bazel_path, f"llvm-project-{llvm_commit}")


def validate_llvm_bazel_commit(llvm_commit,
                               llvm_bazel_path,
                               exit_on_failure=True):
  llvm_bazel_llvm_commit = find_llvm_bazel_llvm_commit(llvm_bazel_path)

  matches = llvm_bazel_llvm_commit == llvm_commit
  if not matches:
    print("WARNING: LLVM commit in llvm-bazel does not match that in IREE"
          f" ({llvm_bazel_llvm_commit} vs {llvm_commit})")
    if exit_on_failure:
      sys.exit(1)


def find_llvm_bazel_llvm_commit(llvm_bazel_path):
  return utils.execute(
      ["git", "submodule", "status", "third_party/llvm-project"],
      capture_output=True,
      cwd=llvm_bazel_path).stdout.split()[0].lstrip("+-")


def find_new_tf_commit(tensorflow_path, llvm_commit, tf_rev):
  utils.execute(["git", "fetch"], cwd=tensorflow_path)

  if tf_rev not in COMMIT_OPTIONS:
    return parse_rev(tensorflow_path, tf_rev)

  if tf_rev == KEEP_COMMIT_OPTION:
    return parse_rev(tensorflow_path, "HEAD")

  if tf_rev == REMOTE_HEAD_COMMIT_OPTION:
    return parse_rev(tensorflow_path, "origin/master")

  # Find commits where the number of occurrences of the given LLVM commit hash
  # changes. In normal cases, there should be at most two commits that match
  # this:
  # 1. The commit that first introduced the new hash in the TF workspace file.
  # 2. The commit that changed it to a new hash afterwards.
  tf_integrate_commit_options = utils.execute(
      [
          "git",
          "log",
          # Only follow the first parent of a merge commit. We don't want to go
          # off to some random PR.
          "--first-parent",
          # Just print the commit hash
          "--format=%H",
          # Look for commits where the number of occurrences of llvm_commit
          # changed.
          # https://git-scm.com/docs/git-log#Documentation/git-log.txt--Sltstringgt
          "-S",
          llvm_commit,
          # Search along the master branch
          "origin/master",
          # Only look in the TF workspace file where the llvm_commit is recorded
          "--",
          "tensorflow/workspace.bzl"
      ],
      capture_output=True,
      cwd=tensorflow_path).stdout.split()
  if len(tf_integrate_commit_options) > 2:
    raise RuntimeError(
        f"Expected one or two TF commits to involve LLVM commit {llvm_commit},"
        f" but got {len(tf_integrate_commit_options)}")

  if not tf_integrate_commit_options:
    raise RuntimeError(
        f"TF does not have any references to LLVM commit {llvm_commit}."
        " Maybe TF export is behind?")

  if tf_rev == INTEGRATE_COMMIT_OPTION:
    return tf_integrate_commit_options[-1]

  assert tf_rev == LATEST_MATCHING_COMMIT_OPTION
  if len(tf_integrate_commit_options) == 1:
    # There hasn't been a subsequent integrate, use remote head.
    return parse_rev(tensorflow_path, "origin/master")

  # Use the commit one before the one that changed away from this LLVM version.
  return parse_rev(tensorflow_path, f"{tf_integrate_commit_options[0]}^")


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


def stage_path(repo_path, to_stage):
  # TODO(laurenzo): Move to utils.py.
  utils.execute(["git", "add", to_stage], cwd=repo_path)


if __name__ == "__main__":
  main(parse_arguments())
