# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Runs a Python script with bindings built by this checkout's Bazel configuration.

Install the Torchax wheel and requirements in a Python 3.12 environment first.
Without arguments, runs the Torchax suite. Otherwise accepts a script and its
arguments, for example: python experimental/torchax/run_bazel.py example.py.
"""

import argparse
import os
from pathlib import Path
import subprocess
import sys
import tempfile

TARGETS = [
    "//compiler/bindings/python:_mlir.so",
    "//compiler/bindings/python:_site_initialize_0.so",
    "//compiler/bindings/python:_ireeCompilerDialects.so",
    "//compiler/bindings/python:iree-compile",
    "//compiler/bindings/python:iree-opt",
    "//compiler/bindings/python:iree-lld",
    "//compiler/bindings/python:UtilOpsPyFiles",
    "//compiler/bindings/python:FlowOpsPyFiles",
    "//compiler/bindings/python:HALOpsPyFiles",
    "//runtime/bindings/python:_runtime.so",
    "@llvm-project//mlir/python:FuncPyFiles",
    "@llvm-project//mlir/python:ArithOpsPyFiles",
]


def link_python_files(source, destination):
    """Merge source/generated package trees, with later definitions taking priority."""
    for source_file in source.rglob("*.py"):
        target = destination / source_file.relative_to(source)
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.is_symlink():
            target.unlink()
        target.symlink_to(source_file.resolve())


def stage_bindings(repo, output, destination):
    compiler = destination / "iree/compiler"
    for source in (
        repo / "third_party/llvm-project/mlir/python/mlir",
        repo / "compiler/bindings/python/iree/compiler",
        output / "external/+_repo_rules+llvm-project/mlir/python/mlir",
        output / "compiler/bindings/python/iree/compiler",
    ):
        link_python_files(source, compiler)
    link_python_files(repo / "runtime/bindings/python/iree", destination / "iree")

    native = compiler / "_mlir_libs"
    native.mkdir(parents=True, exist_ok=True)
    for name in ("_mlir.so", "_site_initialize_0.so", "_ireeCompilerDialects.so"):
        (native / name).symlink_to(output / "compiler/bindings/python" / name)
    library_name = (
        "libIREECompiler.dylib" if sys.platform == "darwin" else "libIREECompiler.so"
    )
    (native / library_name).symlink_to(output / "lib" / library_name)
    runtime = destination / "iree/_runtime_libs"
    runtime.mkdir(parents=True, exist_ok=True)
    (runtime / "_runtime.so").symlink_to(output / "runtime/bindings/python/_runtime.so")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--no-build",
        action="store_true",
        help="reuse an already-built Python 3.12 configuration",
    )
    parser.add_argument("script", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if sys.version_info[:2] != (3, 12):
        parser.error("use the documented Python 3.12 Torchax environment")
    if sys.platform not in ("darwin", "linux"):
        parser.error("this staging runner supports macOS and Linux")
    repo = Path(__file__).resolve().parents[2]
    if not args.no_build:
        subprocess.run(
            [
                "bazelisk",
                "build",
                *TARGETS,
                "--@rules_python//python/config_settings:python_version=3.12",
            ],
            cwd=repo,
            check=True,
        )
    output = (repo / "bazel-bin").resolve()
    with tempfile.TemporaryDirectory(prefix="iree-torchax-bazel-") as temporary:
        stage = Path(temporary)
        stage_bindings(repo, output, stage)
        env = os.environ.copy()
        env["PYTHONPATH"] = os.pathsep.join(
            filter(None, [str(stage), env.get("PYTHONPATH")])
        )
        env["IREE_TOOL_PATH"] = str(output / "compiler/bindings/python")
        script = args.script or [str(Path(__file__).with_name("run_tests.py"))]
        return subprocess.run([sys.executable, *script], env=env).returncode


if __name__ == "__main__":
    sys.exit(main())
