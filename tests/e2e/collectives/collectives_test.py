# Copyright 2023 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import unittest
import iree.compiler
import argparse
import sys
import iree.runtime
from iree.runtime.array_interop import DeviceArray
import os
from typing import List, Tuple, TypeVar
import numpy as np
import tempfile
import subprocess
import test_utils

ArrayLike = TypeVar("ArrayLike")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target_backend", type=str, default="llvm-cpu")
    parser.add_argument("--driver", type=str, default="local-task")
    return parser.parse_known_args()


def prepare_shards_io_files(
    inputs: List[List[ArrayLike]], out_dir: str
) -> Tuple[List[str], List[str]]:
    input_filepaths = []
    output_filepaths = []
    for i in range(len(inputs)):
        input_filepath = os.path.join(out_dir, f"shard_{i}", "input.npy")
        input_filepaths.append(input_filepath)
        os.makedirs(os.path.dirname(input_filepath))
        test_utils.write_numpy_arrays_to_file(filepath=input_filepath, arrays=inputs[i])
        output_filepath = os.path.join(out_dir, f"shard_{i}", "output.npy")
        output_filepaths.append(output_filepath)
    return input_filepaths, output_filepaths


def run_ranks(
    num_ranks: int,
    module_filepath: str,
    function: str,
    inputs: List[List[ArrayLike]],
    driver: str,
) -> List[List[DeviceArray]]:
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
                "--inputs",
            ]
            + input_filepaths
            + ["--outputs"]
            + output_filepaths
        )
        return [
            test_utils.read_numpy_arrays_from_file(out_file)
            for out_file in output_filepaths
        ]


def run_test(
    mlir: str, inputs: List[List[ArrayLike]], expected_outputs: List[List[ArrayLike]]
):
    with tempfile.TemporaryDirectory() as tmp_dir:
        module_filepath = os.path.join(tmp_dir, "module.vmfb")
        iree.compiler.tools.compile_str(
            input_str=mlir,
            output_file=module_filepath,
            target_backends=[args.target_backend],
            input_type="stablehlo",
        )

        num_ranks = len(inputs)
        # Ranks on the 0th axis.
        outputs = run_ranks(
            num_ranks=num_ranks,
            function="all_reduce_sum",
            driver=args.driver,
            module_filepath=module_filepath,
            inputs=inputs,
        )
        for rank in range(num_ranks):
            np.testing.assert_allclose(
                actual=outputs[rank], desired=expected_outputs[rank]
            )


class SingleRank(unittest.TestCase):
    def test_stablehlo_all_reduce(self):
        """
        Test trivial case of all_reduce with one rank.
        all_reduce([1, 2, 3, 4]) == [1, 2, 3, 4].
        """
        stablehlo_mlir = """
            func.func @all_reduce_sum(%input : tensor<4xf32>) -> tensor<4xf32> {
            %out = "stablehlo.all_reduce"(%input) ({
                ^bb0(%arg0: tensor<f32>, %arg1: tensor<f32>):
                %sum = stablehlo.add %arg0, %arg1 : tensor<f32>
                stablehlo.return %sum : tensor<f32>
                }) {channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>,
                    replica_groups = dense<[[0]]> : tensor<1x1xi64>,
                    use_global_device_ids} : (tensor<4xf32>) -> tensor<4xf32>
            return %out : tensor<4xf32>
            }
        """
        inputs = [[np.array([1, 2, 3, 4], dtype=np.float32)]]
        expected_outputs = [[np.array([1, 2, 3, 4], dtype=np.float32)]]
        run_test(mlir=stablehlo_mlir, inputs=inputs, expected_outputs=expected_outputs)


class TwoRanks(unittest.TestCase):
    def test_stablehlo_all_reduce(self):
        """
        Test all_reduce([1, 2, 3, 4], [5, 6, 7, 8]) == [6, 8, 10, 12].
        """
        stablehlo_mlir = """
            func.func @all_reduce_sum(%input : tensor<4xf32>) -> tensor<4xf32> {
            %out = "stablehlo.all_reduce"(%input) ({
                ^bb0(%arg0: tensor<f32>, %arg1: tensor<f32>):
                %sum = stablehlo.add %arg0, %arg1 : tensor<f32>
                stablehlo.return %sum : tensor<f32>
                }) {channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>,
                    replica_groups = dense<[[0, 1]]> : tensor<1x2xi64>,
                    use_global_device_ids} : (tensor<4xf32>) -> tensor<4xf32>
            return %out : tensor<4xf32>
            }
        """
        inputs = [
            [np.array([1, 2, 3, 4], dtype=np.float32)],
            [np.array([5, 6, 7, 8], dtype=np.float32)],
        ]
        expected_outputs = [[np.array([6, 8, 10, 12], dtype=np.float32)]]
        run_test(mlir=stablehlo_mlir, inputs=inputs, expected_outputs=expected_outputs)


if __name__ == "__main__":
    args, remaining_args = parse_args()
    unittest.main(argv=[sys.argv[0]] + remaining_args)
