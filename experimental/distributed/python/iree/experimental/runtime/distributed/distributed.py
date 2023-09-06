# Copyright 2023 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import sys
import iree.runtime
from iree.runtime.array_interop import DeviceArray
import os
from numpy.typing import ArrayLike
from typing import List, Tuple
import tempfile
import subprocess
from . import utils


def prepare_shards_io_files(
    inputs: List[List[ArrayLike]], out_dir: str
) -> Tuple[List[str], List[str]]:
    input_filepaths = []
    output_filepaths = []
    for i in range(len(inputs)):
        input_filepath = os.path.join(out_dir, f"shard_{i}", "input.npy")
        input_filepaths.append(input_filepath)
        os.makedirs(os.path.dirname(input_filepath))
        utils.write_numpy_arrays_to_file(filepath=input_filepath, arrays=inputs[i])
        output_filepath = os.path.join(out_dir, f"shard_{i}", "output.npy")
        output_filepaths.append(output_filepath)
    return input_filepaths, output_filepaths


def run_ranks(
    num_ranks: int,
    module_filepath: str,
    function: str,
    inputs: List[List[ArrayLike]],
    driver: str,
    call_count: int = 1,
    measure_execution_time: bool = False,
    warmup: int = 0,
) -> List[List[ArrayLike]]:
    """
    Start all ranks with mpirun.
    On all ranks run the function |function| from the given module.
    Parameters
    ----------
    inputs : Function inputs for all ranks.
    Axis 0 is ranks. Axis 1 is arguments per rank.
    Returns
    -------
    The output of the function for all ranks.
    Axis 0 is ranks. Axis 1 is arguments per rank.
    """
    with tempfile.TemporaryDirectory() as out_dir:
        input_filepaths, output_filepaths = prepare_shards_io_files(
            inputs=inputs, out_dir=out_dir
        )
        hal_driver = iree.runtime.get_driver(driver)
        hal_driver.query_available_devices()
        subprocess.check_call(
            [
                "mpirun",
                "--oversubscribe",
                "-n",
                str(num_ranks),
                sys.executable,
                os.path.join(os.path.dirname(__file__), "run_rank.py"),
                f"--driver={driver}",
                f"--module_filepath={module_filepath}",
                f"--function={function}",
                f"--call_count={call_count}",
            ]
            + (["--measure_execution_time"] if measure_execution_time else [])
            + [
                f"--warmup={warmup}",
                "--inputs",
            ]
            + input_filepaths
            + ["--outputs"]
            + output_filepaths
        )
        return [
            utils.read_numpy_arrays_from_file(out_file) for out_file in output_filepaths
        ]
