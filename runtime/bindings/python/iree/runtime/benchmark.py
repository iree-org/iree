# Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Provides utilities for benchmarking IREE modules.

Provides convenient methods for invoking IREE's benchmarking tooling from
python. This allows easy benchmarking results from within python.
"""

# pylint: disable=protected-access
# pylint: disable=unused-argument
# pylint: disable=g-explicit-length-test

# TODO(#4131) python>=3.7: Use postponed type annotations.

from collections import namedtuple

import iree.runtime
import numpy
import os
import subprocess

__all__ = [
    "benchmark_exe",
    "benchmark_module",
]

BenchmarkResult = namedtuple(
    "BenchmarkResult", "benchmark_name time cpu_time iterations user_counters"
)

DTYPE_TO_ABI_TYPE = {
    numpy.dtype(numpy.float32): "f32",
    numpy.dtype(numpy.int32): "i32",
    numpy.dtype(numpy.int64): "i64",
    numpy.dtype(numpy.float64): "f64",
    numpy.dtype(numpy.int16): "i16",
    numpy.dtype(numpy.float16): "f16",
    numpy.dtype(numpy.int8): "i8",
    numpy.dtype(numpy.bool_): "i1",
}


class BenchmarkToolError(Exception):
    """Benchmark exception that preserves the command line and error output."""

    def __init__(self, message):
        self.message = message
        super().__init__(self.message)


class BenchmarkTimeoutError(Exception):
    """Exception raised if the benchmark is cancelled by the user specified timeout."""

    pass


def benchmark_exe():
    return os.path.join(
        os.path.dirname(__file__), "..", "_runtime_libs", "iree-benchmark-module"
    )


def benchmark_module(module, entry_function=None, inputs=[], timeout=None, **kwargs):
    funcs = [a for a in module.function_names if a != "__init"]
    if entry_function is None:
        if len(funcs) > 1:
            raise ValueError(f"No function specified with multiple options {funcs}")
        entry_function = funcs[0]
    if entry_function not in funcs:
        raise ValueError(
            f"Attempted to benchmark unknown function {entry_function} of options {funcs}"
        )

    flatbuffer = module.stashed_flatbuffer_blob
    args = [iree.runtime.benchmark_exe()]
    args.append(f"--function={entry_function}")

    for k in kwargs:
        v = kwargs[k]
        args.append(f"--{k}={v}")

    for inp in inputs:
        if isinstance(inp, str):
            args.append(f"--input={inp}")
            continue
        shape = "x".join([str(d) for d in inp.shape])
        abitype = DTYPE_TO_ABI_TYPE[inp.dtype]
        values = inp.flatten()
        if numpy.all(values[0] == values):
            values = str(values[0])
        else:
            values = ",".join([str(v) for v in values])

        args.append(f"--input={shape}x{abitype}={values}")
    args.append(f"--module=-")

    try:
        benchmark_process = subprocess.run(
            args=args,
            input=flatbuffer,
            timeout=timeout,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except subprocess.TimeoutExpired:
        raise BenchmarkTimeoutError(f"Benchmark timed out after {timeout} seconds")
    out = benchmark_process.stdout
    err = benchmark_process.stderr

    err = err.decode()
    if "INVALID_ARGUMENT;" in err:
        raise ValueError("Invalid inputs specified for benchmarking")

    # In the event benchmarking runs but encounteres an internal error,
    # return the internal error instead of benchmark results.
    if "INTERNAL; CUDA driver error" in str(out):
        raise BenchmarkToolError(str(out))

    # Grab individual results by line (skip header lines)
    bench_lines = out.decode().split("\n")[3:]
    benchmark_results = []
    for line in bench_lines:
        split = line.split()
        if len(split) == 0:
            continue
        elif len(split) <= 5:
            raise ValueError(
                f"Benchmark standard output is not expected. Line: '{line}' should have at least 6 columns.\nDetails: {args=},\n{out=}"
            )

        benchmark_name = split[0]
        time = " ".join(split[1:3])
        cpu_time = " ".join(split[3:5])
        iterations = split[5]
        user_counters = None
        if len(split) > 5:
            user_counters = split[6]
        benchmark_results.append(
            BenchmarkResult(
                benchmark_name=benchmark_name,
                time=time,
                cpu_time=cpu_time,
                iterations=iterations,
                user_counters=user_counters,
            )
        )

    return benchmark_results
