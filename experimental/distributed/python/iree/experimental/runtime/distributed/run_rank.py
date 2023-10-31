# Copyright 2023 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import argparse
import iree.runtime
from iree.runtime.array_interop import DeviceArray
from mpi4py import MPI
import utils
import datetime
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Run 1 shard.")
    parser.add_argument("--driver", type=str, default="local-task", help="Device URI.")
    parser.add_argument(
        "--module_filepath", type=str, required=True, help="Path to IREE module."
    )
    parser.add_argument(
        "--function", type=str, required=True, help="Name of function to call."
    )
    parser.add_argument(
        "--call_count",
        type=int,
        default=1,
        help="How many times to call the function during time measurement.",
    )
    parser.add_argument(
        "--measure_execution_time",
        action="store_true",
        default=False,
        help="Measure execution time in seconds f64 and append to results.",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=0,
        help="How many warmup calls to do before the actual call that generates the result.",
    )
    parser.add_argument(
        "--inputs",
        nargs="+",
        type=str,
        required=True,
        help="Path to IREE module inputs for all ranks in npy format.",
    )
    parser.add_argument(
        "--outputs",
        nargs="+",
        type=str,
        required=True,
        help="Path to IREE module outputs form all ranks in npy format.",
    )
    return parser.parse_args()


def run_module(
    device: iree.runtime.HalDevice,
    module_filepath: str,
    function: str,
    call_count: int,
    input_filepath: str,
    output_filepath: str,
    measure_execution_time: bool,
    warmup: int,
):
    config = iree.runtime.Config(device=device)
    with open(module_filepath, "rb") as f:
        vm_flatbuffer = f.read()
    vm_module = iree.runtime.VmModule.from_flatbuffer(config.vm_instance, vm_flatbuffer)
    bound_module = iree.runtime.load_vm_module(vm_module, config)
    input_args = utils.read_numpy_arrays_from_file(input_filepath)
    input_args_on_device = [
        iree.runtime.asdevicearray(device, arr) for arr in input_args
    ]
    for _ in range(warmup):
        getattr(bound_module, function)(*input_args_on_device)
    if measure_execution_time:
        # Sync all ranks
        MPI.COMM_WORLD.barrier()
        start_time = datetime.datetime.now()
    assert call_count > 0
    for _ in range(call_count):
        results = getattr(bound_module, function)(*input_args_on_device)
    if measure_execution_time:
        end_time = datetime.datetime.now()
    if isinstance(results, DeviceArray):
        results = [results]
    if measure_execution_time:
        if isinstance(results, tuple):
            results = list(results)
        results.append(
            np.array((end_time - start_time).total_seconds() / call_count, dtype=float)
        )
    utils.write_numpy_arrays_to_file(filepath=output_filepath, arrays=results)


def run_rank(
    driver: str,
    module_filepath: str,
    function: str,
    inputs: str,
    outputs: str,
    call_count: int,
    measure_execution_time: bool,
    warmup: int,
):
    rank = MPI.COMM_WORLD.Get_rank()
    hal_driver = iree.runtime.get_driver(driver)
    device_infos = hal_driver.query_available_devices()
    device = hal_driver.create_device(
        device_infos[rank % len(device_infos)]["device_id"]
    )
    run_module(
        device=device,
        module_filepath=module_filepath,
        function=function,
        call_count=call_count,
        input_filepath=inputs[rank],
        output_filepath=outputs[rank],
        measure_execution_time=measure_execution_time,
        warmup=warmup,
    )


if __name__ == "__main__":
    args = parse_args()
    run_rank(**vars(args))
