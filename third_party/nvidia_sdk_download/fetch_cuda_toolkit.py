#!/usr/bin/env python3
# Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Fetches components of the CUDA toolkit that we need to build.

Syntax:
  fetch_cuda_toolkit.py {output_dir}

This will download an appropriate toolkit (subset) and print the full path
to the resulting directory (which will be a sub-directory of the output_dir).
"""

from pathlib import Path
import platform
import shutil
import subprocess
import sys

VERSION = "11.6.2"
PRODUCT = "cuda"
COMPONENTS = ["cuda_nvcc", "cuda_cudart"]


def main(output_dir: Path):
  system = platform.system()
  if system == "Linux":
    os = "linux"
  elif system == "Windows":
    os = "windows"
  else:
    print("ERROR: Fetching CUDA toolkit only supported on windows and linux")
    sys.exit(1)

  arch = platform.machine()
  if arch == "AMD64":
    arch = "x86_64"

  target_dir = output_dir / VERSION
  arch_dir = target_dir / f"{os}-{arch}"
  touch_file = arch_dir / "cuda_toolkit.downloaded"
  if touch_file.exists():
    print(f"Not downloading because touch file exists: {touch_file}",
          file=sys.stderr)
  else:
    # Remove and create arch dir.
    if arch_dir.exists():
      shutil.rmtree(arch_dir)
    arch_dir.mkdir(parents=True, exist_ok=True)

    for component in COMPONENTS:
      print(f"Downloading component {component}", file=sys.stderr)
      subprocess.check_call([
          sys.executable,
          str(Path(__file__).resolve().parent / "parse_redist.py"),
          "--label",
          VERSION,
          "--product",
          PRODUCT,
          "--os",
          os,
          "--arch",
          arch,
          "--component",
          component,
          "--output",
          target_dir,
      ],
                            cwd=target_dir,
                            stdout=sys.stderr)

    # Touch the file to note done.
    with open(touch_file, "w") as f:
      pass

  # Report back.
  print(arch_dir)


if __name__ == "__main__":
  if len(sys.argv) != 2:
    print("ERROR: Expected output_dir", file=sys.stderr)
    sys.exit(1)
  main(Path(sys.argv[1]))
