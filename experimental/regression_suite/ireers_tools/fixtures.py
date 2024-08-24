# Copyright 2023 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import pytest

from typing import Dict, Sequence, Union
from pathlib import Path
import subprocess
import time
import os

from .artifacts import (
    Artifact,
    FetchedArtifact,
    ProducedArtifact,
)


IREE_COMPILE_QOL_FLAGS = [
    "--mlir-timing",
    "--mlir-timing-display=list",
    "--iree-consteval-jit-debug",
]


def fetch_source_fixture(url: str, *, group: str):
    @pytest.fixture
    def fetcher() -> FetchedArtifact:
        art = FetchedArtifact(url=url, group=group)
        art.start()
        return art

    return fetcher


def iree_compile(source: Artifact, flags: Sequence[str], vmfb_path: Path):
    if not os.path.exists(vmfb_path.parent):
        os.makedirs(vmfb_path.parent)
    sep = "\n  "
    print("**************************************************************")
    print(f"  {sep.join(flags)}")
    exec_args = (
        [
            "iree-compile",
            "-o",
            str(vmfb_path),
            str(source.path),
        ]
        + IREE_COMPILE_QOL_FLAGS
        + flags
    )
    print("Exec:", " ".join(exec_args))
    start_time = time.time()
    subprocess.run(exec_args, check=True, capture_output=True, cwd=vmfb_path.parent)
    run_time = time.time() - start_time
    print(f"Compilation succeeded in {run_time}s")
    print("**************************************************************")
    return vmfb_path


def iree_run_module(vmfb: Path, *, device, function, args: Sequence[str] = ()):
    exec_args = [
        "iree-run-module",
        f"--device={device}",
        f"--module={vmfb}",
        f"--function={function}",
    ]
    exec_args.extend(args)
    print("**************************************************************")
    print("Exec:", " ".join(exec_args))
    subprocess.run(exec_args, check=True, capture_output=True, cwd=vmfb.parent)


def iree_benchmark_module(vmfb: Path, *, device, function, args: Sequence[str] = ()):
    exec_args = [
        "iree-benchmark-module",
        f"--device={device}",
        f"--module={vmfb}",
        f"--function={function}",
    ]
    exec_args.extend(args)
    print("**************************************************************")
    print("Exec:", " ".join(exec_args))
    subprocess.check_call(exec_args, cwd=vmfb.parent)
