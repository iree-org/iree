# Copyright 2024 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import os
from collections import namedtuple
import logging
from typing import Sequence
import subprocess
import json
from pathlib import Path
import tabulate
from pytest_check import check

vmfb_dir = os.getenv("TEST_OUTPUT_ARTIFACTS", default=Path.cwd())
benchmark_dir = os.path.dirname(os.path.realpath(__file__))
artifacts_dir = os.getenv("IREE_TEST_FILES", default=Path.cwd()) + "/artifacts"
artifacts_dir = Path(os.path.expanduser(artifacts_dir)).resolve()
prompt_encoder_dir = f"{artifacts_dir}/sdxl_clip"
scheduled_unet_dir = f"{artifacts_dir}/sdxl_unet"
vae_decode_dir = f"{artifacts_dir}/sdxl_vae"
prompt_encoder_dir_compile = f"{vmfb_dir}/sdxl_clip_vmfbs"
scheduled_unet_dir_compile = f"{vmfb_dir}/sdxl_unet_vmfbs"
vae_decode_dir_compile = f"{vmfb_dir}/sdxl_vae_vmfbs"


def run_iree_command(args: Sequence[str] = ()):
    command = "Exec:", " ".join(args)
    logging.getLogger().info(command)
    proc = subprocess.run(
        args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False
    )
    (
        stdout_v,
        stderr_v,
    ) = (
        proc.stdout,
        proc.stderr,
    )
    return_code = proc.returncode
    if return_code == 0:
        return 0, proc.stdout
    logging.getLogger().error(
        f"Command failed!\n"
        f"Stderr diagnostics:\n{proc.stderr}\n"
        f"Stdout diagnostics:\n{proc.stdout}\n"
    )
    return 1, proc.stdout


def run_sdxl_rocm_benchmark(rocm_chip):
    exec_args = [
        "iree-compile",
        f"{benchmark_dir}/sdxl_pipeline_bench_f16.mlir",
        "--iree-hal-target-backends=rocm",
        f"--iree-hip-target={rocm_chip}",
        "--iree-global-opt-propagate-transposes=true",
        "--iree-codegen-llvmgpu-use-vector-distribution",
        "--iree-codegen-gpu-native-math-precision=true",
        "--iree-hip-waves-per-eu=2",
        "--iree-opt-outer-dim-concat=true",
        "--iree-llvmgpu-enable-prefetch",
        "-o",
        f"{benchmark_dir}/sdxl_full_pipeline_fp16_rocm.vmfb",
    ]
    # iree compile command for full sdxl pipeline
    ret_value, stdout = run_iree_command(exec_args)
    if ret_value == 1:
        return 1, stdout
    exec_args = [
        "iree-benchmark-module",
        f"--device=hip",
        "--device_allocator=caching",
        f"--module={prompt_encoder_dir_compile}/model.rocm_{rocm_chip}.vmfb",
        f"--parameters=model={prompt_encoder_dir}/real_weights.irpa",
        f"--module={scheduled_unet_dir_compile}/model.rocm_{rocm_chip}.vmfb",
        f"--parameters=model={scheduled_unet_dir}/real_weights.irpa",
        f"--module={vae_decode_dir_compile}/model.rocm_{rocm_chip}.vmfb",
        f"--parameters=model={vae_decode_dir}/real_weights.irpa",
        f"--module={benchmark_dir}/sdxl_full_pipeline_fp16_rocm.vmfb",
        "--function=tokens_to_image",
        "--input=1x4x128x128xf16",
        "--input=1xf16",
        "--input=1x64xi64",
        "--input=1x64xi64",
        "--input=1x64xi64",
        "--input=1x64xi64",
        "--benchmark_repetitions=10",
        "--benchmark_min_warmup_time=3.0",
    ]
    # iree benchmark command for full sdxl pipeline
    return run_iree_command(exec_args)


def run_sdxl_unet_rocm_benchmark(rocm_chip):
    exec_args = [
        "iree-benchmark-module",
        f"--device=hip",
        "--device_allocator=caching",
        f"--module={scheduled_unet_dir_compile}/model.rocm_{rocm_chip}.vmfb",
        f"--parameters=model={scheduled_unet_dir}/real_weights.irpa",
        "--function=run_forward",
        "--input=1x4x128x128xf16",
        "--input=2x64x2048xf16",
        "--input=2x1280xf16",
        "--input=2x6xf16",
        "--input=1xf16",
        "--input=1xi64",
        "--benchmark_repetitions=10",
        "--benchmark_min_warmup_time=3.0",
    ]
    # iree benchmark command for full sdxl pipeline
    return run_iree_command(exec_args)


def run_sdxl_prompt_encoder_rocm_benchmark(rocm_chip):
    exec_args = [
        "iree-benchmark-module",
        f"--device=hip",
        "--device_allocator=caching",
        f"--module={prompt_encoder_dir_compile}/model.rocm_{rocm_chip}.vmfb",
        f"--parameters=model={prompt_encoder_dir}/real_weights.irpa",
        "--function=encode_prompts",
        "--input=1x64xi64",
        "--input=1x64xi64",
        "--input=1x64xi64",
        "--input=1x64xi64",
        "--benchmark_repetitions=10",
        "--benchmark_min_warmup_time=3.0",
    ]
    # iree benchmark command for full sdxl pipeline
    return run_iree_command(exec_args)


def run_sdxl_vae_decode_rocm_benchmark(rocm_chip):
    exec_args = [
        "iree-benchmark-module",
        f"--device=hip",
        "--device_allocator=caching",
        f"--module={vae_decode_dir_compile}/model.rocm_{rocm_chip}.vmfb",
        f"--parameters=model={vae_decode_dir}/real_weights.irpa",
        "--function=main",
        "--input=1x4x128x128xf16",
        "--benchmark_repetitions=10",
        "--benchmark_min_warmup_time=3.0",
    ]
    # iree benchmark command for full sdxl pipeline
    return run_iree_command(exec_args)


BenchmarkResult = namedtuple(
    "BenchmarkResult", "benchmark_name time cpu_time iterations user_counters"
)


def decode_output(bench_lines):
    benchmark_results = []
    for line in bench_lines:
        split = line.split()
        if len(split) == 0:
            continue
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


def job_summary_process(ret_value, output):
    if ret_value == 1:
        # Output should have already been logged earlier.
        logging.getLogger().error("Running SDXL ROCm benchmark failed. Exiting.")
        return

    bench_lines = output.decode().split("\n")[3:]
    benchmark_results = decode_output(bench_lines)
    logging.getLogger().info(benchmark_results)
    benchmark_mean_time = float(benchmark_results[10].time.split()[0])
    return benchmark_mean_time


def test_sdxl_rocm_benchmark(
    goldentime_rocm_e2e,
    goldentime_rocm_unet,
    goldentime_rocm_clip,
    goldentime_rocm_vae,
    rocm_chip,
    goldendispatch_rocm_unet,
    goldendispatch_rocm_clip,
    goldendispatch_rocm_vae,
    goldensize_rocm_unet,
    goldensize_rocm_clip,
    goldensize_rocm_vae,
):
    # e2e benchmark
    ret_value, output = run_sdxl_rocm_benchmark(rocm_chip)
    benchmark_e2e_mean_time = job_summary_process(ret_value, output)
    mean_line = (
        f"E2E Benchmark Time: {str(benchmark_e2e_mean_time)} ms"
        f" (golden time {goldentime_rocm_e2e} ms)"
    )
    logging.getLogger().info(mean_line)

    # unet benchmark
    ret_value, output = run_sdxl_unet_rocm_benchmark(rocm_chip)
    benchmark_unet_mean_time = job_summary_process(ret_value, output)
    mean_line = (
        f"Scheduled Unet Benchmark Time: {str(benchmark_unet_mean_time)} ms"
        f" (golden time {goldentime_rocm_unet} ms)"
    )
    logging.getLogger().info(mean_line)

    # unet compilation stats check
    with open(f"{scheduled_unet_dir_compile}/compilation_info.json", "r") as file:
        comp_stats = json.load(file)
    unet_dispatch_count = int(
        comp_stats["stream-aggregate"]["execution"]["dispatch-count"]
    )
    compilation_line = (
        f"Scheduled Unet Dispatch Count: {unet_dispatch_count}"
        f" (golden dispatch count {goldendispatch_rocm_unet})"
    )
    logging.getLogger().info(compilation_line)

    module_path = f"{scheduled_unet_dir_compile}/model.rocm_{rocm_chip}.vmfb"
    unet_binary_size = Path(module_path).stat().st_size
    compilation_line = (
        f"Scheduled Unet Binary Size: {unet_binary_size} bytes"
        f" (golden binary size {goldensize_rocm_unet} bytes)"
    )
    logging.getLogger().info(compilation_line)

    # prompt encoder benchmark
    ret_value, output = run_sdxl_prompt_encoder_rocm_benchmark(rocm_chip)
    benchmark_clip_mean_time = job_summary_process(ret_value, output)
    mean_line = (
        f"Prompt Encoder Benchmark Time: {str(benchmark_clip_mean_time)} ms"
        f" (golden time {goldentime_rocm_clip} ms)"
    )
    logging.getLogger().info(mean_line)

    # prompt encoder compilation stats check
    with open(f"{prompt_encoder_dir_compile}/compilation_info.json", "r") as file:
        comp_stats = json.load(file)
    clip_dispatch_count = int(
        comp_stats["stream-aggregate"]["execution"]["dispatch-count"]
    )
    compilation_line = (
        f"Prompt Encoder Dispatch Count: {clip_dispatch_count}"
        f" (golden dispatch count {goldendispatch_rocm_clip})"
    )
    logging.getLogger().info(compilation_line)

    module_path = f"{prompt_encoder_dir_compile}/model.rocm_{rocm_chip}.vmfb"
    clip_binary_size = Path(module_path).stat().st_size
    compilation_line = (
        f"Prompt Encoder Binary Size: {clip_binary_size} bytes"
        f" (golden binary size {goldensize_rocm_clip} bytes)"
    )
    logging.getLogger().info(compilation_line)

    # vae decode benchmark
    ret_value, output = run_sdxl_vae_decode_rocm_benchmark(rocm_chip)
    benchmark_vae_mean_time = job_summary_process(ret_value, output)
    mean_line = (
        f"VAE Decode Benchmark Time: {str(benchmark_vae_mean_time)} ms"
        f" (golden time {goldentime_rocm_vae} ms)"
    )
    logging.getLogger().info(mean_line)

    # vae decode compilation stats check
    with open(f"{vae_decode_dir_compile}/compilation_info.json", "r") as file:
        comp_stats = json.load(file)
    vae_dispatch_count = int(
        comp_stats["stream-aggregate"]["execution"]["dispatch-count"]
    )
    compilation_line = (
        f"VAE Decode Dispatch Count: {vae_dispatch_count}"
        f" (golden dispatch count {goldendispatch_rocm_vae})"
    )
    logging.getLogger().info(compilation_line)

    module_path = f"{vae_decode_dir_compile}/model.rocm_{rocm_chip}.vmfb"
    vae_binary_size = Path(module_path).stat().st_size
    compilation_line = (
        f"VAE Decode Binary Size: {vae_binary_size} bytes"
        f" (golden binary size {goldensize_rocm_vae} bytes)"
    )
    logging.getLogger().info(compilation_line)

    # Create mean time table's header and rows
    mean_time_header = ["Benchmark", "Current time (ms)", "Expected/golden time (ms)"]
    mean_time_rows = [
        ["E2E†", f"{benchmark_e2e_mean_time}", f"{goldentime_rocm_e2e}"],
        ["Scheduled Unet", f"{benchmark_unet_mean_time}", f"{goldentime_rocm_unet}"],
        ["Prompt Encoder", f"{benchmark_clip_mean_time}", f"{goldentime_rocm_clip}"],
        ["VAE Decode", f"{benchmark_vae_mean_time}", f"{goldentime_rocm_vae}"],
    ]

    # Create dispatch count table's header and rows
    dispatch_count_header = [
        "Benchmark",
        "Current dispatch count",
        "Expected/golden dispatch count",
    ]
    dispatch_count_rows = [
        ["Scheduled Unet", f"{unet_dispatch_count}", f"{goldendispatch_rocm_unet}"],
        ["Prompt Encoder", f"{clip_dispatch_count}", f"{goldendispatch_rocm_clip}"],
        ["VAE Decode", f"{vae_dispatch_count}", f"{goldendispatch_rocm_vae}"],
    ]

    # Create binary size table's header and rows
    binary_size_header = [
        "Benchmark",
        "Current binary size (bytes)",
        "Expected/golden binary size (bytes)",
    ]
    binary_size_rows = [
        ["Scheduled Unet", f"{unet_binary_size}", f"{goldensize_rocm_unet}"],
        ["Prompt Encoder", f"{clip_binary_size}", f"{goldensize_rocm_clip}"],
        ["VAE Decode", f"{vae_binary_size}", f"{goldensize_rocm_vae}"],
    ]

    # Create mean time table using tabulate
    mean_time_full = [mean_time_header] + mean_time_rows
    mean_time_table = tabulate.tabulate(
        mean_time_full, headers="firstrow", tablefmt="pipe"
    )

    # Create dispatch count table using tabulate
    dispatch_count_full = [dispatch_count_header] + dispatch_count_rows
    dispatch_count_table = tabulate.tabulate(
        dispatch_count_full, headers="firstrow", tablefmt="pipe"
    )

    # Create binary size of compiled artifacts table using tabulate
    binary_size_full = [binary_size_header] + binary_size_rows
    binary_size_table = tabulate.tabulate(
        binary_size_full, headers="firstrow", tablefmt="pipe"
    )

    # Write markdown tables to job summary file
    with open("job_summary.md", "w") as job_summary:
        print("SDXL Benchmark Summary:\n", file=job_summary)
        print(mean_time_table, file=job_summary)
        print("\n† E2E = Encode + Scheduled Unet * 3 + Decode\n", file=job_summary)
        print(dispatch_count_table, file=job_summary)
        print("\n", file=job_summary)
        print(binary_size_table, file=job_summary)

    # Check all values are either <= than golden values for times and == for compilation statistics.

    check.less_equal(
        benchmark_e2e_mean_time,
        goldentime_rocm_e2e,
        "SDXL e2e benchmark time should not regress",
    )
    check.less_equal(
        benchmark_unet_mean_time,
        goldentime_rocm_unet,
        "SDXL unet benchmark time should not regress",
    )
    check.equal(
        unet_dispatch_count,
        goldendispatch_rocm_unet,
        "SDXL scheduled unet dispatch count should not regress",
    )
    check.less_equal(
        unet_binary_size,
        goldensize_rocm_unet,
        "SDXL scheduled unet binary size should not get bigger",
    )
    check.less_equal(
        benchmark_clip_mean_time,
        goldentime_rocm_clip,
        "SDXL prompt encoder benchmark time should not regress",
    )
    check.equal(
        clip_dispatch_count,
        goldendispatch_rocm_clip,
        "SDXL prompt encoder dispatch count should not regress",
    )
    check.less_equal(
        clip_binary_size,
        goldensize_rocm_clip,
        "SDXL prompt encoder binary size should not get bigger",
    )
    check.less_equal(
        benchmark_vae_mean_time,
        goldentime_rocm_vae,
        "SDXL vae decode benchmark time should not regress",
    )
    check.equal(
        vae_dispatch_count,
        goldendispatch_rocm_vae,
        "SDXL vae decode dispatch count should not regress",
    )
    check.less_equal(
        vae_binary_size,
        goldensize_rocm_vae,
        "SDXL vae decode binary size should not get bigger",
    )
