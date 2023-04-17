## Copyright 2023 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Defines Matmul microbenchmarks."""

from e2e_test_framework import unique_ids
from e2e_test_framework.definitions import common_definitions

MATMUL_3456X1024X2048_FP16_MLIR = common_definitions.Model(
    id=unique_ids.MICRO_MATMUL_3456X1024X2048_FP16_MLIR,
    name="matmul_3456x1024x2048_f16t_tile_config_default",
    tags=["fp16", "ubench", "matmul"],
    source_type=common_definitions.ModelSourceType.EXPORTED_LINALG_MLIR,
    source_url=
    "https://storage.googleapis.com/iree-model-artifacts/microbenchmarks/matmul/20230410_1681181224/matmul_3456x1024x2048_f16t_f16t_f16t_tile_config_default.mlirbc",
    entry_function="matmul_3456x1024x2048_f16t_f16t_f16t_tile_config_default",
    input_types=["3456x2048xf16", "2048x1024xf16"])

MATMUL_3456X1024X2048_FP32_MLIR = common_definitions.Model(
    id=unique_ids.MICRO_MATMUL_3456X1024X2048_FP32_MLIR,
    name="matmul_3456x1024x2048_f32t_tile_config_default",
    tags=["fp32", "ubench", "matmul"],
    source_type=common_definitions.ModelSourceType.EXPORTED_LINALG_MLIR,
    source_url=
    "https://storage.googleapis.com/iree-model-artifacts/microbenchmarks/matmul/20230410_1681181224/matmul_3456x1024x2048_f32t_f32t_f32t_tile_config_default.mlirbc",
    entry_function="matmul_3456x1024x2048_f32t_f32t_f32t_tile_config_default",
    input_types=["3456x2048xf32", "2048x1024xf32"])

MATMUL_2560X2560X2560_FP16_MLIR = common_definitions.Model(
    id=unique_ids.MICRO_MATMUL_2560X2560X2560_FP16_MLIR,
    name="matmul_2560x2560x2560_f16t_tile_config_default",
    tags=["fp16", "ubench", "matmul"],
    source_type=common_definitions.ModelSourceType.EXPORTED_LINALG_MLIR,
    source_url=
    "https://storage.googleapis.com/iree-model-artifacts/microbenchmarks/matmul/20230410_1681181224/matmul_2560x2560x2560_f16t_f16t_f16t_tile_config_default.mlirbc",
    entry_function="matmul_2560x2560x2560_f16t_f16t_f16t_tile_config_default",
    input_types=["2560x2560xf16", "2560x2560xf16"])

MATMUL_2560X2560X2560_FP32_MLIR = common_definitions.Model(
    id=unique_ids.MICRO_MATMUL_2560X2560X2560_FP32_MLIR,
    name="matmul_2560x2560x2560_f32t_tile_config_default",
    tags=["fp32", "ubench", "matmul"],
    source_type=common_definitions.ModelSourceType.EXPORTED_LINALG_MLIR,
    source_url=
    "https://storage.googleapis.com/iree-model-artifacts/microbenchmarks/matmul/20230410_1681181224/matmul_2560x2560x2560_f32t_f32t_f32t_tile_config_default.mlirbc",
    entry_function="matmul_2560x2560x2560_f32t_f32t_f32t_tile_config_default",
    input_types=["2560x2560xf32", "2560x2560xf32"])

MATMUL_128X256X8192_FP16_MLIR = common_definitions.Model(
    id=unique_ids.MICRO_MATMUL_128X256X8192_FP16_MLIR,
    name="matmul_128x256x8192_f16t_tile_config_default",
    tags=["fp16", "ubench", "matmul", "splitk"],
    source_type=common_definitions.ModelSourceType.EXPORTED_LINALG_MLIR,
    source_url=
    "https://storage.googleapis.com/iree-model-artifacts/microbenchmarks/matmul/20230410_1681181224/matmul_128x256x8192_f16t_f16t_f16t_tile_config_default.mlirbc",
    entry_function="matmul_128x256x8192_f16t_f16t_f16t_tile_config_default",
    input_types=["128x8192xf16", "8192x256xf16"])

MATMUL_128X256X8192_FP32_MLIR = common_definitions.Model(
    id=unique_ids.MICRO_MATMUL_128X256X8192_FP32_MLIR,
    name="matmul_128x256x8192_f32t_tile_config_default",
    tags=["fp32", "ubench", "matmul", "splitk"],
    source_type=common_definitions.ModelSourceType.EXPORTED_LINALG_MLIR,
    source_url=
    "https://storage.googleapis.com/iree-model-artifacts/microbenchmarks/matmul/20230410_1681181224/matmul_128x256x8192_f32t_f32t_f32t_tile_config_default.mlirbc",
    entry_function="matmul_128x256x8192_f32t_f32t_f32t_tile_config_default",
    input_types=["128x8192xf32", "8192x256xf32"])
