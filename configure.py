#!/usr/bin/env python
# Copyright 2023 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import argparse
from pathlib import Path
import platform
import os
import subprocess
import sys

REPO_ROOT = Path(__file__).resolve().parent


def parse_arguments():
  parser = argparse.ArgumentParser(description="configure.py")
  default_cc = os.environ.get("CC")
  parser.add_argument(
      "--cc",
      default=default_cc,
      help=
      f"Override the c compiler (or {default_cc if default_cc else 'auto-detect'} if omitted)"
  )
  default_cxx = os.environ.get("CXX")
  parser.add_argument(
      "--cxx",
      default=default_cxx,
      help=
      f"Override the c++ compiler (or {default_cxx if default_cxx else 'auto-detect'} if omitted)"
  )
  parser.add_argument(
      "--iree-compiler-dylib",
      default="installed",
      help=
      "Path to libIREECompiler.so (or 'installed' to probe for an installed version)"
  )
  default_cuda_sdk_dir = os.environ.get("IREE_CUDA_DEPS")
  parser.add_argument(
      "--cuda-sdk-dir",
      default=default_cuda_sdk_dir,
      help=f"Path to CUDA SDK (defaults to {default_cuda_sdk_dir})")
  args = parser.parse_args()
  return args


def main(args):
  iree_dir = REPO_ROOT / ".." / "iree"

  # Chain to the main IREE configuration script. Note that it presently
  # only operates on environment variables.
  env = dict(os.environ)
  if args.cc:
    env["CC"] = args.cc
  if args.cxx:
    env["CXX"] = args.cxx
  subprocess.run(
      [sys.executable, str(iree_dir / "configure_bazel.py")],
      check=True,
      env=env)

  if args.iree_compiler_dylib == "installed":
    print("Probing for a path to --iree-compiler-dylib because it was set to "
          "'installed'")
    try:
      args.iree_compiler_dylib = probe_iree_compiler_dylib()
    except:
      print(
          "Probing failed. Either specify an --iree-compiler-dylib argument "
          "or install via `pip install --upgrade -f "
          "https://openxla.github.io/iree/pip-release-links.html iree-compiler`"
      )
      raise

  write_configuration(args)


def write_configuration(args):
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
    envsh.write("# Source with: source .env.sh\n")

    def add_env(key, value):
      env_bazelrc.write(f"build --action_env {key}={value}\n")
      dotenv.write(f"{key}=\"{value}\"\n")
      envsh.write(f"export {key}=\"{value}\"\n")

    add_env("IREE_PJRT_COMPILER_LIB_PATH", args.iree_compiler_dylib)
    add_env("PJRT_NAMES_AND_LIBRARY_PATHS", ','.join(plugin_paths))
    add_env("JAX_USE_PJRT_C_API_ON_TPU", "1")  # TODO: Remove when ready
    if args.cuda_sdk_dir:
      print(f"Enabling CUDA SDK: {args.cuda_sdk_dir}")
      add_env("IREE_CUDA_DEPS", args.cuda_sdk_dir)
    else:
      print("Not enabling CUDA. Pass --cuda-sdk-dir= to enable")


def probe_iree_compiler_dylib() -> str:
  """Probes an installed iree.compiler for the compiler dylib."""
  # TODO: Make this an API on iree.compiler itself.
  from iree.compiler import _mlir_libs
  from iree.compiler import version
  print(f"Found installed iree-compiler package {version.VERSION}")
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
      print(f"Found --iree-compiler-dylib={dylib_path}")
      return dylib_path
  raise ValueError(f"Could not find {dylib_basename} in {paths}")


if __name__ == "__main__":
  main(parse_arguments())
