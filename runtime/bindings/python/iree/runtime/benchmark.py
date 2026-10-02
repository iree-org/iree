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
from typing import Union
from os import PathLike

from . import VmModule
import json
import numpy
import os
import subprocess
import tempfile
from .dtypes import DTYPE_TO_ABI_TYPE

__all__ = [
    "benchmark_exe",
    "benchmark_module",
]

BenchmarkResult = namedtuple(
    "BenchmarkResult", "benchmark_name time cpu_time iterations report"
)
BenchmarkResult.__doc__ = """One benchmark or dispatch row reported by iree-benchmark-module.

time and cpu_time are "<value> <unit>" strings, using % for percentage
aggregates, and iterations is a string. report is the complete Google Benchmark
JSON row, including metadata and user counters with their original values.
"""


class BenchmarkToolError(Exception):
    """Failure of the tool or a reported benchmark, with its diagnostics."""

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


def _build_benchmark_args(
    module: Union[VmModule, PathLike],
    entry_function: str | None = None,
    inputs: list[Union[str, numpy.ndarray]] | None = None,
    executable: Union[str, PathLike, None] = None,
    **kwargs,
) -> tuple[list[str], bytes | None]:
    args = [os.fspath(executable) if executable is not None else benchmark_exe()]

    if isinstance(module, VmModule):
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
        args.append("--module=-")
    else:
        flatbuffer = None
        args.append(f"--module={module}")

    # Without --function the tool benchmarks every exported function that takes
    # no inputs.
    if entry_function is not None:
        args.append(f"--function={entry_function}")

    for k in kwargs:
        # A list or tuple repeats the flag once per value.
        values = kwargs[k] if isinstance(kwargs[k], (list, tuple)) else [kwargs[k]]
        for v in values:
            # IREE flags only parse lowercase booleans.
            if isinstance(v, bool):
                v = "true" if v else "false"
            args.append(f"--{k}={v}")

    for inp in inputs or []:
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

    return args, flatbuffer


def _run_benchmark(
    args: list[str], flatbuffer: bytes | None = None, timeout: float | None = None
) -> tuple[str, str]:
    """Runs a benchmark command line and returns its stdout and stderr.

    Raises BenchmarkTimeoutError if the timeout expires and BenchmarkToolError
    with both outputs if the tool fails.
    """
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
    out = benchmark_process.stdout.decode()
    err = benchmark_process.stderr.decode()

    if benchmark_process.returncode != 0:
        raise BenchmarkToolError(f"stderr:\n{err}\nstdout:\n{out}")
    return out, err


def benchmark_module(
    module: Union[VmModule, PathLike],
    entry_function: str | None = None,
    inputs: list[Union[str, numpy.ndarray]] | None = None,
    timeout: float | None = None,
    executable: Union[str, PathLike, None] = None,
    **kwargs,
) -> list[BenchmarkResult]:
    """Benchmarks a module with iree-benchmark-module.

    Keyword arguments are passed to the tool as --key=value flags, once per
    value of a list or tuple. Without an
    entry function a module file runs every exported function that takes no
    inputs. The tool defaults to the one in this package unless an executable
    is given.

    Returns a BenchmarkResult with the complete JSON report of every row,
    including aggregates and dispatch rows. Raises BenchmarkToolError if the
    tool exits unsuccessfully or a reported benchmark fails.
    """
    if "benchmark_out" in kwargs or "benchmark_out_format" in kwargs:
        raise ValueError("benchmark_module reads results from its own JSON file")
    with tempfile.TemporaryDirectory() as directory:
        # A results file keeps console and output processing text on stdout
        # out of the parsed report.
        report = os.path.join(directory, "results.json")
        args, flatbuffer = _build_benchmark_args(
            module=module,
            entry_function=entry_function,
            inputs=inputs,
            executable=executable,
            benchmark_out=report,
            benchmark_out_format="json",
            **kwargs,
        )
        _run_benchmark(args, flatbuffer, timeout)
        with open(report) as file:
            return _parse_benchmark_results(file.read())


def _parse_benchmark_results(report: str) -> list[BenchmarkResult]:
    """Parses a Google Benchmark JSON report into benchmark results."""
    # Google Benchmark leaves the report empty when no benchmarks match.
    if not report.strip():
        return []
    results = []
    for run in json.loads(report)["benchmarks"]:
        if run.get("error_occurred"):
            raise BenchmarkToolError(f"{run['name']}: {run['error_message']}")
        unit = run["time_unit"]
        real_time, cpu_time = run["real_time"], run["cpu_time"]
        if run.get("aggregate_unit") == "percentage":
            unit = "%"
            real_time *= 100
            cpu_time *= 100
        results.append(
            BenchmarkResult(
                benchmark_name=run["name"],
                time=f"{real_time} {unit}",
                cpu_time=f"{cpu_time} {unit}",
                iterations=str(run["iterations"]),
                report=run,
            )
        )
    return results
