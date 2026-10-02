# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Exercises iree-benchmark-module lifecycle contracts.

Covers execution models, batching, discovery, output ownership, capture and
failures through iree.runtime.benchmark.
"""

import os
from pathlib import Path
import re
import subprocess
import tempfile
import unittest
from unittest import mock

import iree.compiler
from iree._runtime import libs
from iree.runtime.benchmark import (
    _build_benchmark_args as build_benchmark_args,
    _run_benchmark as run_benchmark_command,
    benchmark_module,
    BenchmarkToolError,
)

# Two 64x64 matmuls and one 64x32 matmul: two dispatch functions, one of them
# called twice per invocation.
MATMUL = """
func.func @main(%lhs: tensor<64x64xf32>, %rhs: tensor<64x64xf32>,
                %narrow: tensor<64x32xf32>) -> tensor<64x32xf32> {
  %zero = arith.constant 0.0 : f32
  %empty = tensor.empty() : tensor<64x64xf32>
  %fill = linalg.fill ins(%zero : f32) outs(%empty : tensor<64x64xf32>) -> tensor<64x64xf32>
  %0 = linalg.matmul ins(%lhs, %rhs : tensor<64x64xf32>, tensor<64x64xf32>)
                     outs(%fill : tensor<64x64xf32>) -> tensor<64x64xf32>
  %1 = linalg.matmul ins(%0, %rhs : tensor<64x64xf32>, tensor<64x64xf32>)
                     outs(%fill : tensor<64x64xf32>) -> tensor<64x64xf32>
  %narrow_empty = tensor.empty() : tensor<64x32xf32>
  %narrow_fill = linalg.fill ins(%zero : f32) outs(%narrow_empty : tensor<64x32xf32>) -> tensor<64x32xf32>
  %2 = linalg.matmul ins(%1, %narrow : tensor<64x64xf32>, tensor<64x32xf32>)
                     outs(%narrow_fill : tensor<64x32xf32>) -> tensor<64x32xf32>
  return %2 : tensor<64x32xf32>
}
"""

# Two exported functions without inputs, for discovery without --function.
NO_INPUTS = """
func.func @one() -> tensor<4xf32> {
  %input = util.unfoldable_constant dense<-1.0> : tensor<4xf32>
  %result = math.absf %input : tensor<4xf32>
  return %result : tensor<4xf32>
}
func.func @two() -> tensor<4xf32> {
  %input = util.unfoldable_constant dense<-1.0> : tensor<4xf32>
  %result = math.absf %input : tensor<4xf32>
  return %result : tensor<4xf32>
}
"""

# Uses no HAL device.
PURE = """
func.func @main() -> i32 {
  %result = arith.constant 42 : i32
  return %result : i32
}
"""

# Adds the number of invocations so far to the absolute input.
STATEFUL = """
util.global private mutable @count = 0 : i32
func.func @main(%input: tensor<4xf32>) -> tensor<4xf32> {
  %old = util.global.load @count : i32
  %one = arith.constant 1 : i32
  %next = arith.addi %old, %one : i32
  util.global.store %next, @count : i32
  %count = arith.sitofp %next : i32 to f32
  %empty = tensor.empty() : tensor<4xf32>
  %counts = linalg.fill ins(%count : f32) outs(%empty : tensor<4xf32>) -> tensor<4xf32>
  %absolute = math.absf %input : tensor<4xf32>
  %result = arith.addf %absolute, %counts : tensor<4xf32>
  return %result : tensor<4xf32>
}
"""

# One dispatch, dumped as a generated dispatch benchmark.
ABSOLUTE = """
func.func @abs(%input: tensor<4xf32>) -> tensor<4xf32> {
  %result = math.absf %input : tensor<4xf32>
  return %result : tensor<4xf32>
}
"""

LOCAL_TARGET = [
    "--iree-hal-target-device=local",
    "--iree-hal-local-target-device-backends=llvm-cpu",
]
MATMUL_INPUTS = ["64x64xf32=1", "64x64xf32=1", "64x32xf32=1"]
DEVICE = "local-task"
# Flags of every run: four iterations on a device with two workers.
BENCHMARK_FLAGS = {
    "device": DEVICE,
    "task_topology_group_count": 2,
    "benchmark_min_time": "4x",
}
# The tool under test and the replay tools come from the same package.
TOOLS = Path(libs.library_path)
BENCHMARK_TOOL = TOOLS / "iree-benchmark-module"
TIMEOUT_SECONDS = 60


def setUpModule():
    # The checks must be independent of ambient Google Benchmark defaults.
    environment = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith("BENCHMARK_")
    }
    patch = mock.patch.dict(os.environ, environment, clear=True)
    patch.start()
    unittest.addModuleCleanup(patch.stop)


def compile_module(directory, name, source, *flags):
    """Compiles MLIR source to directory/name.vmfb and returns its path."""
    output = directory / f"{name}.vmfb"
    iree.compiler.compile_str(
        source, extra_args=[*LOCAL_TARGET, *flags], output_file=str(output)
    )
    return output


def benchmark_results(module, function="main", inputs=MATMUL_INPUTS, **flags):
    """Benchmarks a module with BENCHMARK_FLAGS and returns all result rows.

    Keyword flags are passed to the tool. Without a function every exported
    function that takes no inputs runs.
    """
    return benchmark_module(
        module,
        entry_function=function,
        inputs=inputs,
        timeout=TIMEOUT_SECONDS,
        executable=BENCHMARK_TOOL,
        **{**BENCHMARK_FLAGS, **flags},
    )


def run_benchmark(module, function="main", inputs=MATMUL_INPUTS, **flags):
    """Runs a module like benchmark_results and returns stdout and stderr."""
    args, _ = build_benchmark_args(
        module,
        entry_function=function,
        inputs=inputs,
        executable=BENCHMARK_TOOL,
        **{**BENCHMARK_FLAGS, **flags},
    )
    return run_benchmark_command(args, timeout=TIMEOUT_SECONDS)


def run_replay_tool(name, *args):
    """Runs a replay tool from the package and fails with its output."""
    result = subprocess.run(
        [TOOLS / name, *map(str, args)],
        capture_output=True,
        text=True,
        timeout=TIMEOUT_SECONDS,
    )
    if result.returncode != 0:
        raise AssertionError(f"{name} failed:\n{result.stderr}\n{result.stdout}")


def compile_dispatch_benchmark(directory):
    """Compiles the generated benchmark of ABSOLUTE's one dispatch.

    Returns the module and the name of its benchmark function.
    """
    sources = directory / "dispatches"
    compile_module(
        directory,
        "abs",
        ABSOLUTE,
        f"--iree-hal-dump-executable-benchmarks-to={sources}",
    )
    (source,) = sources.glob("*_benchmark.mlir")
    module = directory / "dispatch.vmfb"
    iree.compiler.compile_file(str(source), output_file=str(module))
    stdout, _ = run_benchmark(
        module, function=None, inputs=[], benchmark_list_tests=True
    )
    (name,) = re.findall(r"^BM_([^/\s]+)", stdout, re.MULTILINE)
    return module, name


class BenchmarkLifecycleTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        modules = tempfile.TemporaryDirectory(prefix="benchmark-lifecycle-test-")
        cls.addClassCleanup(modules.cleanup)
        directory = Path(modules.name)
        cls.sync = compile_module(directory, "sync", MATMUL)
        cls.async_module = compile_module(
            directory, "async", MATMUL, "--iree-execution-model=async-external"
        )
        cls.repeated = compile_module(
            directory,
            "repeated",
            MATMUL,
            "--iree-hal-benchmark-dispatch-repeat-count=4",
        )
        # Four iterations of each execution model: four sync invocations, one
        # batch of four async invocations, or one invocation that repeats its
        # dispatches four times.
        cls.batched = [(cls.sync, 1), (cls.async_module, 4), (cls.repeated, 4)]

    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="benchmark-lifecycle-test-")
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)

    def assertBenchmarkFails(self, message, module, **kwargs):
        """Checks that run_benchmark fails with message in its output."""
        with self.assertRaisesRegex(BenchmarkToolError, re.escape(message)):
            run_benchmark(module, **kwargs)

    def test_unregistered_flags_pass_through(self):
        # As in the other benchmark tools.
        run_benchmark(self.sync, not_a_registered_flag=1)

    def test_iterations_for_each_execution_model(self):
        for module, batch_size in self.batched:
            with self.subTest(module=module.stem):
                results = benchmark_results(module, batch_size=batch_size)
                self.assertEqual([result.iterations for result in results], ["4"])

    def test_output_processing_for_each_execution_model(self):
        for module, batch_size in self.batched:
            with self.subTest(module=module.stem):
                stdout, _ = run_benchmark(
                    module,
                    batch_size=batch_size,
                    enable_output_processing=True,
                    expected_output="64x32xf32=262144",
                )
                self.assertIn("[SUCCESS]", stdout)

    def test_batch_rounded_up_to_concurrency(self):
        # Three timelines run a batch of four as six invocations.
        results = benchmark_results(
            self.async_module, batch_size=4, batch_concurrency=3
        )
        self.assertEqual([result.iterations for result in results], ["6"])

    def test_discovery_prepares_async_arguments(self):
        # Discovery must also prepare empty arguments for async entry points.
        for model in ("async-internal", "async-external"):
            with self.subTest(model=model):
                module = compile_module(
                    self.directory,
                    model,
                    NO_INPUTS,
                    f"--iree-execution-model={model}",
                )
                results = benchmark_results(module, function=None, inputs=[])
                self.assertEqual(len(results), 2, results)

    def test_filtered_discovery_uses_time_unit(self):
        for model in ("async-internal", "async-external"):
            with self.subTest(model=model):
                module = compile_module(
                    self.directory,
                    model,
                    NO_INPUTS,
                    f"--iree-execution-model={model}",
                )
                results = benchmark_results(
                    module,
                    function=None,
                    inputs=[],
                    benchmark_filter="BM_two",
                    time_unit="ns",
                )
                self.assertEqual(len(results), 1, results)
                self.assertTrue(results[0].time.endswith(" ns"), results)

    def test_output_processing_requires_one_function(self):
        # Without --function both @one and @two run.
        module = compile_module(self.directory, "no-inputs", NO_INPUTS)
        self.assertBenchmarkFails(
            "output processing requires one function",
            module,
            function=None,
            inputs=[],
            enable_output_processing=True,
        )

    def test_output_processing_without_devices(self):
        module = compile_module(self.directory, "pure", PURE)
        stdout, _ = run_benchmark(
            module, inputs=[], enable_output_processing=True, output="-"
        )
        self.assertIn("i32=42", stdout)

    def test_outputs_of_last_measured_invocation(self):
        # The stateful function counts its invocations, and the outputs are
        # those of the fourth.
        for model in ("async-internal", "async-external"):
            with self.subTest(model=model):
                module = compile_module(
                    self.directory,
                    model,
                    STATEFUL,
                    f"--iree-execution-model={model}",
                )
                stdout, _ = run_benchmark(
                    module,
                    inputs=["4xf32=-1"],
                    enable_output_processing=True,
                    expected_output="4xf32=5",
                )
                self.assertIn("[SUCCESS]", stdout)

    def test_dispatch_benchmark_discovery_and_selection(self):
        # Automatic discovery and explicit selection use the same dispatch ABI:
        # two batches of three per four iterations.
        module, name = compile_dispatch_benchmark(self.directory)
        for function in (None, name):
            with self.subTest(function=function):
                results = benchmark_results(
                    module, function=function, inputs=[], batch_size=3
                )
                self.assertEqual([result.iterations for result in results], ["6"])

    def test_dispatch_benchmark_rejects_inputs(self):
        module, name = compile_dispatch_benchmark(self.directory)
        self.assertBenchmarkFails(
            "dispatch wrappers take --batch_size, not --input",
            module,
            function=name,
            inputs=["4xf32=1"],
        )

    def test_device_statistics(self):
        _, stderr = run_benchmark(self.sync, print_device_statistics=True)
        self.assertIn("main_dispatch_", stderr)

    def test_replay_capture(self):
        for module in (self.sync, self.async_module):
            with self.subTest(module=module.stem):
                capture = self.directory / f"{module.stem}.ireereplay"
                run_benchmark(module, device_replay_output=capture)
                run_replay_tool("iree-run-replay", f"--device={DEVICE}", capture)

    def test_replay_capture_finalized_on_error(self):
        capture = self.directory / "failed.ireereplay"
        self.assertBenchmarkFails(
            "input0 shape rank mismatch",
            self.sync,
            inputs=["1xf32=1", *MATMUL_INPUTS[1:]],
            device_replay_output=capture,
        )
        run_replay_tool("iree-dump-replay", capture)

    def test_batch_size_and_concurrency_must_be_positive(self):
        for flags in ({"batch_size": 0}, {"batch_concurrency": 0}):
            with self.subTest(**flags):
                self.assertBenchmarkFails(
                    "batch size and concurrency must be positive", self.sync, **flags
                )


if __name__ == "__main__":
    unittest.main()
