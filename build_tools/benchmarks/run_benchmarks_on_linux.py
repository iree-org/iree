#!/usr/bin/env python3
# Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Runs all matched benchmark suites on a Linux device."""

import sys
import pathlib

# Add build_tools python dir to the search path.
sys.path.insert(0, str(pathlib.Path(__file__).parent.with_name("python")))

from typing import Any, List, Optional
import atexit
import json
import requests
import shutil
import subprocess
import tarfile
import urllib.parse

from common import benchmark_suite as benchmark_suite_module
from common.benchmark_driver import BenchmarkDriver
from common.benchmark_suite import BenchmarkCase, BenchmarkSuite
from common.benchmark_config import BenchmarkConfig
from common.benchmark_definition import (
    execute_cmd,
    execute_cmd_and_get_output,
    get_git_commit_hash,
    get_iree_benchmark_module_arguments,
    wait_for_iree_benchmark_module_start,
    parse_iree_benchmark_metrics,
)
from common.linux_device_utils import get_linux_device_info
from e2e_test_artifacts import iree_artifacts
from e2e_model_tests import run_module_utils

import common.common_arguments


class LinuxBenchmarkDriver(BenchmarkDriver):
    """Linux benchmark driver."""

    def __init__(self, gpu_id: str, *args, **kwargs):
        self.gpu_id = gpu_id
        super().__init__(*args, **kwargs)

    def run_benchmark_case(
        self,
        benchmark_case: BenchmarkCase,
        benchmark_results_filename: Optional[pathlib.Path],
        capture_filename: Optional[pathlib.Path],
    ) -> None:
        module_dir = benchmark_case.module_dir
        if isinstance(module_dir, pathlib.Path):
            case_tmp_dir = module_dir
            module_path = module_dir / iree_artifacts.MODULE_FILENAME
        else:
            module_rel_dir = iree_artifacts.get_module_dir_path(
                benchmark_case.run_config.module_generation_config
            )
            case_tmp_dir = self.config.tmp_dir / module_rel_dir
            case_tmp_dir.mkdir(parents=True, exist_ok=True)
            module_path = self.__fetch_file(
                uri=urllib.parse.urljoin(module_dir, iree_artifacts.MODULE_FILENAME),
                dest=case_tmp_dir / iree_artifacts.MODULE_FILENAME,
            )

        inputs_dir = None
        expected_output_dir = None
        if benchmark_case.input_uri:
            inputs_dir = self.__fetch_and_unpack_npy(
                uri=benchmark_case.input_uri, dest_dir=case_tmp_dir / "inputs_npy"
            )
        if benchmark_case.expected_output_uri:
            expected_output_dir = self.__fetch_and_unpack_npy(
                uri=benchmark_case.expected_output_uri,
                dest_dir=case_tmp_dir / "expected_outputs_npy",
            )

        if benchmark_results_filename:
            if self.config.normal_benchmark_tool_dir is None:
                raise ValueError("normal_benchmark_tool_dir can't be None.")

            if self.config.verify and expected_output_dir:
                if not inputs_dir:
                    raise ValueError(f"Input data is missing for {benchmark_case}.")
                self.__run_verify(
                    tool_dir=self.config.normal_benchmark_tool_dir,
                    benchmark_case=benchmark_case,
                    module_path=module_path,
                    inputs_dir=inputs_dir,
                    expected_outputs_dir=expected_output_dir,
                )

            self.__run_benchmark(
                tool_dir=self.config.normal_benchmark_tool_dir,
                benchmark_case=benchmark_case,
                module_path=module_path,
                results_filename=benchmark_results_filename,
            )

        if capture_filename:
            self.__run_capture(
                benchmark_case=benchmark_case,
                module_path=module_path,
                capture_filename=capture_filename,
            )

    def __build_tool_cmds(
        self,
        benchmark_case: BenchmarkCase,
        tool_path: pathlib.Path,
        module_path: pathlib.Path,
        inputs_dir: Optional[pathlib.Path] = None,
    ) -> List[Any]:
        run_config = benchmark_case.run_config
        cmds: List[Any] = run_module_utils.build_linux_wrapper_cmds_for_device_spec(
            run_config.target_device_spec
        )
        cmds.append(tool_path)

        cmds += [f"--module={module_path}"]
        cmds += run_config.materialize_run_flags(
            gpu_id=self.gpu_id,
            inputs_dir=inputs_dir,
        )

        return cmds

    def __fetch_and_unpack_npy(self, uri: str, dest_dir: pathlib.Path) -> pathlib.Path:
        out_dir = self.__unpack_file(
            src=self.__fetch_file(
                uri=uri,
                dest=dest_dir.with_suffix(".tgz"),
            ),
            dest=dest_dir,
        )
        return out_dir.absolute()

    def __fetch_file(self, uri: str, dest: pathlib.Path) -> pathlib.Path:
        """Check and fetch file if needed."""
        if dest.exists():
            return dest
        req = requests.get(uri, stream=True, timeout=60)
        if not req.ok:
            raise RuntimeError(f"Failed to fetch {uri}: {req.status_code} - {req.text}")
        with dest.open("wb") as dest_file:
            for data in req.iter_content(chunk_size=64 * 1024 * 1024):
                dest_file.write(data)
        return dest

    def __unpack_file(self, src: pathlib.Path, dest: pathlib.Path) -> pathlib.Path:
        """Unpack tar with/without compression."""
        if dest.exists():
            return dest
        with tarfile.open(src) as tar_file:
            tar_file.extractall(dest)
        return dest

    def __run_verify(
        self,
        tool_dir: pathlib.Path,
        benchmark_case: BenchmarkCase,
        module_path: pathlib.Path,
        inputs_dir: pathlib.Path,
        expected_outputs_dir: pathlib.Path,
    ):
        cmd = self.__build_tool_cmds(
            benchmark_case=benchmark_case,
            tool_path=tool_dir / "iree-run-module",
            module_path=module_path,
            inputs_dir=inputs_dir,
        )
        # Currently only support single output.
        cmd.append(f'--expected_output=@{expected_outputs_dir / "output_0.npy"}')
        cmd += benchmark_case.verify_params
        execute_cmd_and_get_output(cmd, verbose=self.verbose)

    def __run_benchmark(
        self,
        tool_dir: pathlib.Path,
        benchmark_case: BenchmarkCase,
        module_path: pathlib.Path,
        results_filename: pathlib.Path,
    ):
        tool_name = benchmark_case.benchmark_tool_name
        cmd = self.__build_tool_cmds(
            benchmark_case=benchmark_case,
            tool_path=tool_dir / tool_name,
            module_path=module_path,
        )

        if tool_name == "iree-benchmark-module":
            cmd.extend(
                get_iree_benchmark_module_arguments(
                    results_filename=str(results_filename),
                    driver_info=benchmark_case.driver_info,
                    benchmark_min_time=self.config.benchmark_min_time,
                )
            )

        benchmark_stdout, benchmark_stderr = execute_cmd_and_get_output(
            cmd, verbose=self.verbose
        )
        benchmark_metrics = parse_iree_benchmark_metrics(
            benchmark_stdout, benchmark_stderr
        )
        if self.verbose:
            print(benchmark_metrics)
        results_filename.write_text(json.dumps(benchmark_metrics.to_json_object()))

    def __run_capture(
        self,
        benchmark_case: BenchmarkCase,
        module_path: pathlib.Path,
        capture_filename: pathlib.Path,
    ):
        capture_config = self.config.trace_capture_config
        if capture_config is None:
            raise ValueError("capture_config can't be None.")

        tool_name = benchmark_case.benchmark_tool_name
        cmd = self.__build_tool_cmds(
            benchmark_case=benchmark_case,
            tool_path=capture_config.traced_benchmark_tool_dir / tool_name,
            module_path=module_path,
        )

        if tool_name == "iree-benchmark-module":
            cmd.extend(
                get_iree_benchmark_module_arguments(
                    driver_info=benchmark_case.driver_info,
                    benchmark_min_time=self.config.benchmark_min_time,
                    capture_mode=True,
                )
            )

        process = subprocess.Popen(
            cmd, env={"TRACY_NO_EXIT": "1"}, stdout=subprocess.PIPE, text=True
        )

        wait_for_iree_benchmark_module_start(process, self.verbose)

        capture_cmd = [capture_config.trace_capture_tool, "-f", "-o", capture_filename]
        stdout_redirect = None if self.verbose else subprocess.DEVNULL
        execute_cmd(capture_cmd, verbose=self.verbose, stdout=stdout_redirect)


def main(args):
    device_info = get_linux_device_info(
        args.device_model, args.cpu_uarch, args.gpu_id, args.verbose
    )
    if args.verbose:
        print(device_info)

    commit = get_git_commit_hash("HEAD")
    benchmark_config = BenchmarkConfig.build_from_args(args, commit)

    benchmark_groups = json.loads(args.execution_benchmark_config.read_text())
    run_configs = benchmark_suite_module.get_run_configs_by_target_and_shard(
        benchmark_groups, args.target_device_name, args.shard_index
    )

    benchmark_suite = BenchmarkSuite.load_from_run_configs(
        run_configs=run_configs, root_benchmark_dir=benchmark_config.root_benchmark_dir
    )

    benchmark_driver = LinuxBenchmarkDriver(
        gpu_id=args.gpu_id,
        device_info=device_info,
        benchmark_config=benchmark_config,
        benchmark_suite=benchmark_suite,
        benchmark_grace_time=1.0,
        verbose=args.verbose,
    )

    if args.pin_cpu_freq:
        raise NotImplementedError("CPU freq pinning is not supported yet.")
    if args.pin_gpu_freq:
        raise NotImplementedError("GPU freq pinning is not supported yet.")
    if not args.no_clean:
        atexit.register(shutil.rmtree, args.tmp_dir)

    benchmark_driver.run()

    benchmark_results = benchmark_driver.get_benchmark_results()
    if args.output is not None:
        with args.output.open("w") as f:
            f.write(benchmark_results.to_json_str())

    if args.verbose:
        print(benchmark_results.commit)
        print(benchmark_results.benchmarks)

    trace_capture_config = benchmark_config.trace_capture_config
    if trace_capture_config:
        # Put all captures in a tarball and remove the original files.
        with tarfile.open(trace_capture_config.capture_tarball, "w:gz") as tar:
            for capture_filename in benchmark_driver.get_capture_filenames():
                tar.add(capture_filename)

    benchmark_errors = benchmark_driver.get_benchmark_errors()
    if benchmark_errors:
        print("Benchmarking completed with errors", file=sys.stderr)
        raise RuntimeError(benchmark_errors)


def parse_argument():
    arg_parser = common.common_arguments.Parser()
    arg_parser.add_argument("--device_model", default="Unknown", help="Device model")
    arg_parser.add_argument(
        "--cpu_uarch", default=None, help="CPU microarchitecture, e.g., CascadeLake"
    )
    arg_parser.add_argument(
        "--gpu_id",
        type=str,
        default="0",
        help="GPU ID to run the benchmark, e.g., '0' or 'GPU-<UUID>'",
    )

    return arg_parser.parse_args()


if __name__ == "__main__":
    main(parse_argument())
