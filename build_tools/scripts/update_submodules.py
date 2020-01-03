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
import re
import os
import subprocess
import sys


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
      help="Updates the IREE LLVM build files from TensorFlow"
      " (Defaults to True if --llvm_commit=TENSORFLOW)",
      type=str2bool,
      nargs="?",
      default=None)
  args = parser.parse_args()

  # Default repo path.
  if args.repo is None:
    args.repo = execute(["git", "rev-parse", "--show-toplevel"],
                        cwd=os.path.dirname(__file__),
                        capture_output=True,
                        silent=True).strip().decode("UTF-8")

  # Set some defaults.
  if not args.tensorflow:
    args.tensorflow = os.path.join(args.repo, "third_party", "tensorflow")
  if not args.llvm:
    args.llvm = os.path.join(args.repo, "third_party", "llvm-project")
  return args


def str2bool(v):
  if v is None:
    return None
  if isinstance(v, bool):
    return v
  if v.lower() in ("yes", "true", "t", "y", "1"):
    return True
  elif v.lower() in ("no", "false", "f", "n", "0"):
    return False
  else:
    raise argparse.ArgumentTypeError("Boolean value expected.")


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

  # Update build files.
  if not args.update_build_files:
    print("Not updating build files (--update_build_files not specified)")
  else:
    print("\n*** Updating BUILD.bazel files ***")
    update_build_files_from_tensorflow(args.repo, args.tensorflow)


def execute(args, cwd, capture_output=False, silent=False, **kwargs):
  if not silent:
    print("+", " ".join(args), "  [from %s]" % cwd)
  if capture_output:
    return subprocess.check_output(args, cwd=cwd, **kwargs)
  else:
    return subprocess.check_call(args, cwd=cwd, **kwargs)


def get_commit(path, rev="HEAD"):
  return execute(["git", "rev-parse", rev],
                 cwd=path,
                 silent=True,
                 capture_output=True).decode("ISO-8859-1").strip()


def update_submodule(path, commit, tracking="origin/master"):
  # Fetch.
  execute(["git", "fetch"], cwd=path)
  # Determine commit.
  if commit == "REMOTE":
    commit = get_commit(path, rev=tracking)
    print("Resolved remote commit:", commit)

  # Rebase to commit (will fail if not fast-forward).
  execute(["git", "checkout", commit], cwd=path)


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
  copy_text_file(src_llvm_build,
                 os.path.join(overlay_path, "llvm", "BUILD.bazel"))
  copy_text_file(src_mlir_build,
                 os.path.join(overlay_path, "mlir", "BUILD.bazel"))
  copy_text_file(src_mlir_test_build,
                 os.path.join(overlay_path, "mlir", "test", "BUILD.bazel"))


def copy_text_file(src_file, dst_file, prepend_text=None):
  print("+ cp %s %s" % (src_file, dst_file),
        "  [with prepended text]" if prepend_text else "")
  with open(src_file, "r", encoding="UTF-8") as f:
    src_contents = f.read()

  if prepend_text:
    src_contents = prepend_text + src_contents
  if not os.path.exists(dst_file):
    print("WARNING: Destination file does not exist:", dst_file)
  with open(dst_file, "w", encoding="UTF-8") as f:
    f.write(src_contents)


if __name__ == "__main__":
  main(parse_arguments())
