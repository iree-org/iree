# Copyright 2023 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import iree.compiler
import iree.runtime
import os
from . import run_ranks
import subprocess
from pathlib import Path
from jax._src.lib import xla_client
from jaxlib.xla_client import HloSharding
from typing import List, Tuple, Union
from numpy.typing import ArrayLike
import jax
from jax._src.sharding_impls import GSPMDSharding
import jax._src.interpreters.pxla as pxla
import numpy as np
from datetime import timedelta

xla_extension = xla_client._xla


def compile_mlir(mlir_filepath: str, output_filepath: str, use_cache: bool, **kwargs):
    if use_cache and os.path.exists(output_filepath):
        return
    os.makedirs(os.path.dirname(output_filepath), exist_ok=True)
    iree.compiler.compile_file(
        input_file=mlir_filepath, output_file=output_filepath, **kwargs
    )


def extract_args_sharding(
    xla_computation: xla_extension.XlaComputation,
) -> List[HloSharding]:
    return [
        HloSharding.from_proto(sharding)
        for sharding in xla_computation.get_hlo_module().spmd_parameters_shardings
    ]


def extract_results_sharding(
    xla_computation: xla_extension.XlaComputation,
) -> List[HloSharding]:
    sharding = HloSharding.from_proto(
        xla_computation.get_hlo_module().spmd_output_sharding
    )
    if len(sharding.tuple_elements()):
        return sharding.tuple_elements()
    else:
        return [sharding]


def shard_arg(arg: ArrayLike, sharding: HloSharding) -> List[ArrayLike]:
    gspmd_sharding = GSPMDSharding(devices=jax.local_devices(), op_sharding=sharding)
    indices = gspmd_sharding.devices_indices_map(arg.shape).values()
    sharded_array = pxla.shard_arg(
        arg, devices=jax.local_devices(), arg_indices=indices, sharding=gspmd_sharding
    )
    return [shard.data for shard in sharded_array.global_shards]


def shard_args(
    args: List[ArrayLike], shardings: List[HloSharding]
) -> List[List[ArrayLike]]:
    assert len(args) == len(shardings)
    return [shard_arg(arg, sharding) for arg, sharding in zip(args, shardings)]


def assemble_shards(shards: List[ArrayLike], sharding: HloSharding) -> ArrayLike:
    if sharding.is_replicated():
        return shards[0]
    else:
        raise NotImplementedError()


def propagate_shardings_and_spmd_partition(
    mlir_filepath: str,
    output_filepath: str,
    num_devices: int,
    use_cache: bool,
    allow_spmd_sharding_propagation_to_output: int = 1,
):
    res = subprocess.run(
        [
            "stablehlo-opt",
            (
                "--pass-pipeline=builtin.module(stablehlo-xla-sharding-propagation-and-spmd-partitioner{"
                "is_spmd=1 "
                f"allow_spmd_sharding_propagation_to_output={allow_spmd_sharding_propagation_to_output} "
                "allow_spmd_sharding_propagation_to_parameters=1 "
                f"num_partitions={num_devices} "
                "num_replicas=1})"
            ),
            mlir_filepath,
        ],
        check=True,
        stdout=subprocess.PIPE,
    )
    Path(output_filepath).parent.mkdir(parents=True, exist_ok=True)
    if use_cache and os.path.exists(output_filepath):
        return
    os.makedirs(os.path.dirname(output_filepath), exist_ok=True)
    with open(output_filepath, "wb") as f:
        f.write(res.stdout)


def swap_shard_axis(arrays: List[ArrayLike]) -> List[List[ArrayLike]]:
    """Swap axis 0 with 1."""
    if len(arrays) == 0:
        return []
    expected_shards = len(arrays[0])
    res = [[] for _ in range(expected_shards)]
    for arr in arrays:
        assert len(arr) == expected_shards
        for shard in range(expected_shards):
            res[shard].append(arr[shard])
    return res


def execute_distributed(
    num_ranks: int,
    mlir_filepath: str,
    iree_module_filepath: str,
    function: str,
    inputs: List[ArrayLike],
    driver: str,
    measure_execution_time: bool = False,
) -> Union[List[ArrayLike], Tuple[List[ArrayLike], timedelta]]:
    with open(mlir_filepath, "r") as f:
        mlir_str = f.read()
    xla_computation = xla_extension.mlir.mlir_module_to_xla_computation(
        mlir_module=mlir_str, use_tuple_args=False, return_tuple=False
    )
    args_sharding = extract_args_sharding(xla_computation)
    results_sharding = extract_results_sharding(xla_computation)
    sharded_args = shard_args(args=inputs, shardings=args_sharding)
    sharded_args = swap_shard_axis(sharded_args)
    sharded_results = run_ranks(
        num_ranks=num_ranks,
        module_filepath=iree_module_filepath,
        function=function,
        inputs=sharded_args,
        driver=driver,
    )
    sharded_results = swap_shard_axis(sharded_results)
    if measure_execution_time:
        sharded_results, execution_times = sharded_results
    res = [
        assemble_shards(shards=result_shards, sharding=sharding)
        for result_shards, sharding in zip(sharded_results, results_sharding)
    ]
    if measure_execution_time:
        res = res, timedelta(seconds=np.max(execution_times))
    return res


def validate_sharding_passes(
    mlir_filepath: str,
    mlir_with_sharding_annotations_filepath: str,
    inputs: List[ArrayLike],
    function: str,
    num_devices: int,
    use_cache: bool,
    driver: str,
    target_backend: str,
    output_prefix_path: str,
    allow_spmd_sharding_propagation_to_output: int = 1,
):
    # Single instance.
    iree_module_filepath = (
        f"{output_prefix_path}{os.path.basename(mlir_filepath)}.{driver}.vmfb"
    )
    compile_mlir(
        mlir_filepath=mlir_filepath,
        output_filepath=iree_module_filepath,
        use_cache=use_cache,
        target_backends=[target_backend],
    )
    iree_module = iree.runtime.load_vm_flatbuffer_file(
        path=iree_module_filepath, driver=driver
    )
    results = iree_module[function](*inputs)
    if isinstance(results, iree.runtime.DeviceArray):
        results = [results]

    # Distributed.
    spmd_mlir_filepath = f"{output_prefix_path}{os.path.basename(mlir_with_sharding_annotations_filepath)}.spmd.mlir"
    propagate_shardings_and_spmd_partition(
        mlir_filepath=mlir_with_sharding_annotations_filepath,
        output_filepath=spmd_mlir_filepath,
        num_devices=num_devices,
        use_cache=use_cache,
        allow_spmd_sharding_propagation_to_output=allow_spmd_sharding_propagation_to_output,
    )
    spmd_iree_module_filepath = f"{output_prefix_path}{os.path.basename(spmd_mlir_filepath)}.{target_backend}.vmfb"
    compile_mlir(
        mlir_filepath=spmd_mlir_filepath,
        output_filepath=spmd_iree_module_filepath,
        use_cache=use_cache,
        target_backends=[target_backend],
    )
    spmd_results = execute_distributed(
        num_ranks=num_devices,
        mlir_filepath=spmd_mlir_filepath,
        iree_module_filepath=spmd_iree_module_filepath,
        function=function,
        inputs=inputs,
        driver=driver,
    )

    assert len(results) == len(spmd_results)
    for result, spmd_result in zip(results, spmd_results):
        np.testing.assert_allclose(result, spmd_result, atol=1e-7)
