# Copyright 2023 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""sync.py helper script.

This script prepares the repository for development by checking out and
syncing to compatible upstream revisions. It is used by CI and recommended
for use for normal development.
"""

from typing import Optional, Sequence, Tuple

import os
from pathlib import Path
import platform
import re
import subprocess
import sys

REPO_ROOT = Path(__file__).parent.parent
EXTERNAL_PATH = REPO_ROOT / "external"
CLONE_PATH = EXTERNAL_PATH / ".clones"


def pip_install(package_name: str, pip_extra_args: Sequence[str] = ()):
  args = [sys.executable, "-m", "pip"]
  args.extend(["install", package_name])
  args.extend(pip_extra_args)
  execute(args)


def sync_nightly():
  """Synchronizes to a nightly binary compiler release.

  Compiler binaries will be installed and corresponding sources for
  runtime deps will be checked out/synced in external.
  """
  # Install iree-compiler from the nightly repository.
  pip_install("iree-compiler",
              pip_extra_args=[
                  "--upgrade", "-f",
                  "https://iree-org.github.io/iree/pip-release-links.html"
              ])
  iree_version, iree_commit = probe_iree_compiler_version()
  log(f"Using IREE compiler binary version={iree_version} at {iree_commit}")
  dylib_path = probe_iree_compiler_dylib()
  log(f"Found IREE compiler dylib: {dylib_path}")
  iree_path = checkout_repo("iree",
                            clone_url="https://github.com/iree-org/iree.git",
                            commit=iree_commit)
  sync_iree_runtime_submodules(iree_path)
  jax_path = checkout_repo("jax", "https://github.com/google/jax.git")
  tf_commit = probe_jax_tensorflow_commit(jax_path)
  log(f"Jax is synced to tensorflow commit {tf_commit}")
  tf_path = checkout_repo("tensorflow",
                          "https://github.com/tensorflow/tensorflow.git",
                          commit=tf_commit)
  write_env(iree_compiler_dylib=dylib_path)


def main():
  args = sys.argv[1:]
  if not args:
    command = "nightly"
  else:
    command = args[0]
  if command == "nightly":
    sync_nightly()
  else:
    raise ValueError("Expected command to be 'nightly'")


def write_env(iree_compiler_dylib: Path):
  plugin_paths = []

  def add_plugin_path(name, rel_path):
    abs_path = REPO_ROOT / "bazel-bin" / rel_path
    plugin_paths.append(f"{name}{os.path.pathsep}{abs_path}")

  add_plugin_path("iree_cpu",
                  "iree/integrations/pjrt/cpu/pjrt_plugin_iree_cpu.so")
  add_plugin_path("iree_cuda",
                  "iree/integrations/pjrt/cuda/pjrt_plugin_iree_cuda.so")

  # Give the environment to bazel.
  with open(REPO_ROOT / "env.bazelrc", "wt") as env_bazelrc, open(
      REPO_ROOT / ".env", "wt") as dotenv, open(REPO_ROOT / ".env.sh",
                                                "wt") as envsh:
    envsh.write("# Source with: source .env.sh")

    def add_env(key, value):
      env_bazelrc.write(f"build --action_env {key}={value}\n")
      dotenv.write(f"{key}=\"{value}\"\n")
      envsh.write(f"export {key}=\"{value}\"\n")

    add_env("IREE_PJRT_COMPILER_LIB_PATH", iree_compiler_dylib)
    add_env("PJRT_NAMES_AND_LIBRARY_PATHS", ','.join(plugin_paths))
    add_env("JAX_USE_PJRT_C_API_ON_TPU", "1")  # TODO: Remove when ready


def log(msg: str):
  print(msg)


def execute(args: Sequence[str], cwd: Optional[str] = None):
  log(f"Running: {' '.join(args)}")
  subprocess.check_call(args, cwd=cwd)


def checkout_repo(repo_name,
                  clone_url: str,
                  commit: Optional[str] = None) -> Path:
  link_path = EXTERNAL_PATH / repo_name
  clone_path = CLONE_PATH / repo_name
  if not link_path.exists():
    if not clone_path.exists():
      log(f"Repository not checked out at {link_path}... "
          f"cloning from {clone_url}")
      execute(["git", "clone", clone_url, str(clone_path)])
    log(f"Creating symlink from {clone_path} -> {link_path}")
    link_path.symlink_to(clone_path)
  if commit:
    execute(["git", "checkout", commit], cwd=link_path)
  return link_path


def probe_iree_compiler_version() -> Tuple[str, str]:
  """Probes an installed iree.compiler and returns (version, commit)."""
  from iree.compiler import version
  return version.VERSION, version.REVISIONS["IREE"]


def probe_iree_compiler_dylib() -> str:
  """Probes an installed iree.compiler for the compiler dylib."""
  # TODO: Make this an API on iree.compiler itself.
  from iree.compiler import _mlir_libs
  dylib_basename = "libIREECompiler.so"
  system = platform.system()
  if system == "Darwin":
    dylib_basename = "libIREECompiler.dylib"
  elif system == "Windows":
    dylib_basename = "IREECompiler.dll"

  paths = _mlir_libs.__path__
  for p in paths:
    dylib_path = Path(p) / dylib_basename
    if dylib_path.exists():
      return dylib_path
  raise ValueError(f"Could not find {dylib_basename} in {paths}")


def probe_jax_tensorflow_commit(jax_path) -> str:
  with open(jax_path / "WORKSPACE", "rt") as f:
    contents = f.read()
  for m in re.finditer(
      "https://github.com/tensorflow/tensorflow/archive/([0-9a-f]+).tar.gz",
      contents):
    return m[1]
  raise ValueError(f"Unable to find tensorflow commit hash in {jax_path}")


def sync_iree_runtime_submodules(iree_path: Path):
  with open(
      iree_path / "build_tools" / "scripts" / "git" / "runtime_submodules.txt",
      "rt") as f:
    submodule_names = [n.strip() for n in f.readlines()]
  args = ["git", "submodule", "update", "--init", "--depth", "1"]
  args.extend(submodule_names)
  execute(args, cwd=str(iree_path))


if __name__ == "__main__":
  main()
