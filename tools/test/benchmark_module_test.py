# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Exercises benchmark lifecycle contracts through the command line tools.

Run by iree-benchmark-module-dispatch-statistics.mlir, which passes itself as
the matmul module.
"""

import argparse
import csv
import io
import json
import os
from pathlib import Path
import subprocess
import tempfile

# One dispatch on each of two devices.
MULTI_DEVICE = """
func.func public @multi_device_mul(
  %input_a: tensor<4xf32> {iree.abi.affinity = #hal.device.promise<@device_a>}
) -> (tensor<4xf32> {iree.abi.affinity = #hal.device.promise<@device_a>}) {
  %constant_a = arith.constant dense<[0.0, 1.0, 2.0, 3.0]> : tensor<4xf32>
  %transient_a = arith.mulf %input_a, %constant_a : tensor<4xf32>
  %transient_b = flow.tensor.transfer %transient_a : tensor<4xf32> to #hal.device.promise<@device_b>
  %constant_b = arith.constant dense<[4.0, 5.0, 6.0, 7.0]> : tensor<4xf32>
  %result_b = arith.mulf %transient_b, %constant_b : tensor<4xf32>
  %result_a = flow.tensor.transfer %result_b : tensor<4xf32> to #hal.device.promise<@device_a>
  func.return %result_a : tensor<4xf32>
}
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compiler", default="iree-compile")
    parser.add_argument("--benchmark", default="iree-benchmark-module")
    parser.add_argument("--run-replay", default="iree-run-replay")
    parser.add_argument("--dump-replay", default="iree-dump-replay")
    parser.add_argument("--matmul", type=Path, required=True)
    args = parser.parse_args()

    # The harness must be independent of ambient Google Benchmark defaults.
    env = {k: v for k, v in os.environ.items() if not k.startswith("BENCHMARK_")}

    def run(command, success=True, environment=env):
        result = subprocess.run(
            [str(x) for x in command],
            capture_output=True,
            text=True,
            env=environment,
            timeout=60,
        )
        assert result.returncode == 0 if success else result.returncode > 0, (
            command,
            result.returncode,
            result.stdout,
            result.stderr,
        )
        return result

    with tempfile.TemporaryDirectory(prefix="benchmark-module-test-") as temporary:
        root = Path(temporary)

        def compile_module(source, name, *options):
            output = root / (name + ".vmfb")
            run(
                [
                    args.compiler,
                    source,
                    "--iree-hal-target-device=local",
                    "--iree-hal-local-target-device-backends=llvm-cpu",
                    *options,
                    "-o",
                    output,
                ]
            )
            return output

        matmul = args.matmul
        sync = compile_module(matmul, "sync")
        async_module = compile_module(
            matmul, "async", "--iree-execution-model=async-external"
        )
        repeated = compile_module(
            matmul, "repeated", "--iree-hal-benchmark-dispatch-repeat-count=4"
        )
        inputs = ["--input=64x64xf32=1", "--input=64x64xf32=1", "--input=64x32xf32=1"]

        def benchmark(
            module=sync,
            *options,
            function="main",
            input_flags=inputs,
            success=True,
            environment=env
        ):
            return run(
                [
                    args.benchmark,
                    "--device=local-task",
                    "--task_topology_group_count=2",
                    "--module=" + str(module),
                    "--benchmark_min_time=4x",
                    *(["--function=" + function] if function else []),
                    *input_flags,
                    *options,
                ],
                success,
                environment,
            )

        def fails(expected, module=sync, *options, **kwargs):
            result = benchmark(module, *options, success=False, **kwargs)
            assert expected in result.stderr, (options, expected, result.stderr)
            return result

        def rows(module=sync, *options, **kwargs):
            return json.loads(
                benchmark(module, "--benchmark_format=json", *options, **kwargs).stdout
            )["benchmarks"]

        def dispatches(reports):
            return [r for r in reports if "calls" in r]

        seconds = {"ns": 1e-9, "us": 1e-6, "ms": 1e-3, "s": 1.0}

        def check_dispatches(reports, iterations=4, batches=1):
            samples = dispatches(reports)
            assert len(samples) == 2, reports
            assert sorted(r["calls"] for r in samples) == [1, 2], samples
            assert all(r["iterations"] == iterations for r in samples), samples
            assert abs(sum(r["percent"] for r in samples) - 100) < 1e-6, samples
            for r in samples:
                # Per-call counters are in seconds; the row time is the
                # per-iteration total of all calls in the row's time unit.
                total = r["real_time"] * seconds[r["time_unit"]]
                assert abs(r["mean"] * r["calls"] - total) <= 1e-6 * total, r
                # Clocks with microsecond ticks can time a short call as 0.
                assert 0 <= r["min"] <= r["mean"] <= r["max"] and r["max"] > 0, r
                # The spread of per-call means needs two profiled batches.
                if batches < 2:
                    assert "stddev" not in r, r
                else:
                    assert 0 <= r["stddev"] <= r["max"] - r["min"], r

        # Flags the build does not register pass through, as in the other
        # benchmark tools.
        benchmark(sync, "--not_a_registered_flag=1")
        # With --benchmark_min_time=4x a pass runs four loop iterations of one
        # invocation, or one of four batched invocations.
        for module, batch in ((sync, 1), (async_module, 4), (repeated, 4)):
            check_dispatches(
                rows(module, "--dispatch_statistics", "--batch_size=" + str(batch)),
                batches=4 // batch,
            )
            result = benchmark(
                module,
                "--batch_size=" + str(batch),
                "--enable_output_processing",
                "--expected_output=64x32xf32=262144",
            )
            assert "[SUCCESS]" in result.stdout, result.stdout
        check_dispatches(
            rows(
                async_module,
                "--dispatch_statistics",
                "--batch_size=4",
                "--batch_concurrency=3",
            ),
            iterations=6,
        )

        # Both destinations see the same measured and profiled results. Profiles
        # aggregate all repetitions using their own total iteration count.
        for aggregate_flag in ([], ["--benchmark_report_aggregates_only=true"]):
            output = root / "results.json"
            reports = rows(
                sync,
                "--dispatch_statistics",
                "--benchmark_repetitions=2",
                "--benchmark_out=" + str(output),
                *aggregate_flag
            )
            check_dispatches(reports, iterations=8, batches=8)
            assert reports == json.loads(output.read_text())["benchmarks"]
        result = benchmark(
            sync,
            "--dispatch_statistics",
            "--benchmark_repetitions=2",
            "--benchmark_format=csv",
        )
        reports = list(csv.DictReader(io.StringIO(result.stdout)))
        assert sorted(float(r["calls"]) for r in reports if r["calls"]) == [
            1,
            2,
        ], reports
        output = root / "environment.json"
        reports = rows(
            sync,
            "--dispatch_statistics",
            environment={
                **env,
                "BENCHMARK_OUT": str(output),
                "BENCHMARK_OUT_FORMAT": "json",
            },
        )
        assert reports == json.loads(output.read_text())["benchmarks"]
        output = root / "tabular.txt"
        benchmark(
            sync,
            "--dispatch_statistics",
            "--benchmark_out=" + str(output),
            "--benchmark_out_format=console",
            "--benchmark_counters_tabular",
        )
        assert "calls" in output.read_text() and "calls=" not in output.read_text()
        assert (
            "BM_main"
            in benchmark(sync, "--dispatch_statistics", "--benchmark_list_tests").stdout
        )

        # Discovery must also prepare empty arguments for async entry points.
        no_inputs = root / "no-inputs.mlir"
        no_inputs.write_text(
            "\n".join(
                """
func.func @%s() -> tensor<4xf32> {
  %%input = util.unfoldable_constant dense<-1.0> : tensor<4xf32>
  %%result = math.absf %%input : tensor<4xf32>
  return %%result : tensor<4xf32>
}
"""
                % name
                for name in ("one", "two")
            )
        )
        for model in ("async-internal", "async-external"):
            module = compile_module(
                no_inputs, "no-inputs-" + model, "--iree-execution-model=" + model
            )
            reports = rows(
                module, "--dispatch_statistics", function=None, input_flags=[]
            )
            assert len(reports) == 4 and len(dispatches(reports)) == 2, reports
            reports = rows(
                module,
                "--dispatch_statistics",
                "--benchmark_filter=BM_two",
                "--time_unit=ns",
                function=None,
                input_flags=[],
            )
            assert len(reports) == 2 and all(
                r["time_unit"] == "ns" for r in reports
            ), reports

        # Outputs of more than one function (@one and @two) cannot be processed.
        fails(
            "output processing requires one function",
            module,
            "--enable_output_processing",
            function=None,
            input_flags=[],
        )

        pure = root / "pure.mlir"
        pure.write_text(
            """
func.func @main() -> i32 {
  %result = arith.constant 42 : i32
  return %result : i32
}
"""
        )
        module = compile_module(pure, "pure")
        result = benchmark(
            module, "--enable_output_processing", "--output=-", input_flags=[]
        )
        assert "i32=42" in result.stdout, result.stdout
        fails(
            "--dispatch_statistics requires a module that uses HAL devices",
            module,
            "--dispatch_statistics",
            input_flags=[],
        )

        # A program that uses HAL devices but runs no dispatch still runs, and
        # warns that the breakdown is empty.
        splat = root / "splat.mlir"
        splat.write_text(
            """
func.func @main() -> tensor<1024xf32> {
  %result = util.unfoldable_constant dense<1.0> : tensor<1024xf32>
  return %result : tensor<1024xf32>
}
"""
        )
        module = compile_module(splat, "splat")
        result = benchmark(module, "--dispatch_statistics", input_flags=[])
        assert (
            "--dispatch_statistics: BM_main: 0 functions" in result.stderr
            and "timings may be incomplete" in result.stderr
        ), result.stderr

        # Outputs belong to the measured pass, although profiling invokes the
        # same stateful VM function afterwards.
        stateful = root / "stateful.mlir"
        stateful.write_text(
            """
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
        )
        for model in ("async-internal", "async-external"):
            module = compile_module(
                stateful, "stateful-" + model, "--iree-execution-model=" + model
            )
            result = benchmark(
                module,
                "--dispatch_statistics",
                "--enable_output_processing",
                "--expected_output=4xf32=5",
                input_flags=["--input=4xf32=-1"],
            )
            assert "[SUCCESS]" in result.stdout, result.stdout

        # Automatic discovery and explicit selection use the same dispatch ABI.
        absolute = root / "abs.mlir"
        absolute.write_text(
            """
func.func @abs(%input: tensor<4xf32>) -> tensor<4xf32> {
  %result = math.absf %input : tensor<4xf32>
  return %result : tensor<4xf32>
}
"""
        )
        artifacts = root / "dispatches"
        compile_module(
            absolute,
            "abs",
            "--iree-hal-dump-executable-benchmarks-to=" + str(artifacts),
        )
        sources = list(artifacts.glob("*_benchmark.mlir"))
        assert len(sources) == 1, sources
        module = root / "dispatch.vmfb"
        run([args.compiler, sources[0], "-o", module])
        automatic = rows(
            module,
            "--dispatch_statistics",
            "--batch_size=3",
            function=None,
            input_flags=[],
        )
        name = automatic[0]["name"].split("/")[0].removeprefix("BM_")
        explicit = rows(
            module,
            "--dispatch_statistics",
            "--batch_size=3",
            function=name,
            input_flags=[],
        )
        for reports in (automatic, explicit):
            assert reports[0]["iterations"] == 6, reports
            samples = dispatches(reports)
            assert len(samples) == 1 and samples[0]["calls"] == 1, reports
            assert samples[0]["iterations"] == 6, reports
        fails(
            "dispatch wrappers take --batch_size, not --input",
            module,
            function=name,
            input_flags=["--input=4xf32=1"],
        )

        # A profile must cover every participating device.
        multi_source = root / "multi.mlir"
        multi_source.write_text(MULTI_DEVICE)
        multi = root / "multi.vmfb"
        run(
            [
                args.compiler,
                multi_source,
                "--iree-execution-model=async-external",
                "--iree-hal-target-device=device_a=local[0]",
                "--iree-hal-target-device=device_b=local[1]",
                "--iree-hal-local-target-device-backends=llvm-cpu",
                "-o",
                multi,
            ]
        )
        reports = rows(
            multi,
            "--device=local-task",
            "--dispatch_statistics",
            function="multi_device_mul",
            input_flags=["--input=4xf32=10,11,12,13"],
        )
        samples = dispatches(reports)
        assert len(samples) == 2 and all(r["calls"] == 1 for r in samples), reports

        # Native captures finish before output processing. Replay files must be
        # finalized both after successful execution and on a runtime error.
        result = benchmark(sync, "--print_device_statistics")
        assert "main_dispatch_" in result.stderr, result.stderr
        for module in (sync, async_module):
            capture = root / (module.stem + ".ireereplay")
            benchmark(module, "--device_replay_output=" + str(capture))
            run([args.run_replay, "--device=local-task", capture])
        capture = root / "failed.ireereplay"
        fails(
            "input0 shape rank mismatch",
            sync,
            "--device_replay_output=" + str(capture),
            input_flags=["--input=1xf32=1", *inputs[1:]],
        )
        run([args.dump_replay, capture])
        for invalid in ("--batch_size=0", "--batch_concurrency=0"):
            fails("batch size and concurrency must be positive", sync, invalid)
        # Conflicts are rejected before any session or capture starts, so the
        # tool reports them rather than a backend lacking the capture provider.
        fails(
            "--dispatch_statistics cannot be combined with device profiling",
            sync,
            "--dispatch_statistics",
            "--device_capture_tool=bogus",
        )
        fails(
            "--dispatch_statistics cannot be combined with --device_replay_output",
            sync,
            "--dispatch_statistics",
            "--device_replay_output=" + str(root / "conflict.ireereplay"),
        )
    print("benchmark module lifecycle checks passed")


if __name__ == "__main__":
    main()
