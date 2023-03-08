## Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Defines TFLite models."""

from e2e_test_framework import unique_ids
from e2e_test_framework.definitions import common_definitions

UB_MATMUL_2560x2560xF16 = common_definitions.Model(
    id=unique_ids.UB_MATMUL_2560x2560xF16,
    name="UB_MATMUL_2560x2560xF16",
    tags=["fp16"],
    source_type=common_definitions.ModelSourceType.EXPORTED_LINALG_MLIR,
    source_url=
    "https://storage.googleapis.com/iree-model-artifacts/matmul_2560x2560xf16.mlir",
    entry_function="matmul",
    input_types=[])

UB_MATMUL_3456x1024x2048xF16_CONFIG_128x128_32x5 = common_definitions.Model(
    id=unique_ids.UB_MATMUL_3456x1024x2048xF16_CONFIG_128x128_32x5,
    name="UB_MATMUL_3456x1024x2048xF16_CONFIG_128x128_32x5",
    tags=["fp16"],
    source_type=common_definitions.ModelSourceType.EXPORTED_LINALG_MLIR,
    source_url=
    "https://storage.googleapis.com/iree-model-artifacts/matmul_3456x1024x2048_f16t_f16t_f16t_tile_config_128x128_32x5_tensorcore.mlir",
    entry_function=
    "matmul_3456x1024x2048_f16t_f16t_f16t_tile_config_128x128_32x5_tensorcore",
    input_types=["3456x2048xf16", "2048x1024xf16", "3456x1024xf16"])

UB_MATMUL_3456x1024x2048xF16_CONFIG_128x128_64x4 = common_definitions.Model(
    id=unique_ids.UB_MATMUL_3456x1024x2048xF16_CONFIG_128x128_64x4,
    name="UB_MATMUL_3456x1024x2048xF16_CONFIG_128x128_64x4",
    tags=["fp16"],
    source_type=common_definitions.ModelSourceType.EXPORTED_LINALG_MLIR,
    source_url=
    "https://storage.googleapis.com/iree-model-artifacts/matmul_3456x1024x2048_f16t_f16t_f16t_tile_config_128x128_64x4_tensorcore.mlir",
    entry_function=
    "matmul_3456x1024x2048_f16t_f16t_f16t_tile_config_128x128_64x4_tensorcore",
    input_types=["3456x2048xf16", "2048x1024xf16", "3456x1024xf16"])
