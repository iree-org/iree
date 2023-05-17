## Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Defines JAX models."""

from e2e_test_framework import unique_ids
from e2e_test_framework.definitions import common_definitions

GCS_ARTIFACT_ROOT_DIR = "https://storage.googleapis.com/iree-model-artifacts/jax/jax_models_0.4.10_1684396752"

# Derived from https://huggingface.co/docs/transformers/model_doc/resnet#transformers.FlaxResNetModel.
RESNET50_TAGS = ["fp32", "cnn", "resnet"]

RESNET50_FP32_JAX_3X224X224XF32_BATCH1 = common_definitions.Model(
    id=unique_ids.MODEL_RESNET50_FP32_JAX_3X224X224XF32_BATCH1,
    name="RESNET50_FP32_JAX_3X224X224XF32_BATCH1",
    tags=RESNET50_TAGS + ["batch-1"],
    source_type=common_definitions.ModelSourceType.EXPORTED_STABLEHLO_MLIR,
    source_url=f"{GCS_ARTIFACT_ROOT_DIR}/RESNET50/batch_1/stablehlo.mlirbc",
    entry_function="main",
    input_types=["1x3x224x224xf32"])

RESNET50_FP32_JAX_3X224X224XF32_BATCH8 = common_definitions.Model(
    id=unique_ids.MODEL_RESNET50_FP32_JAX_3X224X224XF32_BATCH8,
    name="RESNET50_FP32_JAX_3X224X224XF32_BATCH8",
    tags=RESNET50_TAGS + ["batch-8"],
    source_type=common_definitions.ModelSourceType.EXPORTED_STABLEHLO_MLIR,
    source_url=f"{GCS_ARTIFACT_ROOT_DIR}/RESNET50/batch_8/stablehlo.mlirbc",
    entry_function="main",
    input_types=["8x3x224x224xf32"])

RESNET50_FP32_JAX_3X224X224XF32_BATCH64 = common_definitions.Model(
    id=unique_ids.MODEL_RESNET50_FP32_JAX_3X224X224XF32_BATCH64,
    name="RESNET50_FP32_JAX_3X224X224XF32_BATCH64",
    tags=RESNET50_TAGS + ["batch-64"],
    source_type=common_definitions.ModelSourceType.EXPORTED_STABLEHLO_MLIR,
    source_url=f"{GCS_ARTIFACT_ROOT_DIR}/RESNET50/batch_64/stablehlo.mlirbc",
    entry_function="main",
    input_types=["64x3x224x224xf32"])

RESNET50_FP32_JAX_3X224X224XF32_BATCH128 = common_definitions.Model(
    id=unique_ids.MODEL_RESNET50_FP32_JAX_3X224X224XF32_BATCH128,
    name="RESNET50_FP32_JAX_3X224X224XF32_BATCH128",
    tags=RESNET50_TAGS + ["batch-128"],
    source_type=common_definitions.ModelSourceType.EXPORTED_STABLEHLO_MLIR,
    source_url=f"{GCS_ARTIFACT_ROOT_DIR}/RESNET50/batch_128/stablehlo.mlirbc",
    entry_function="main",
    input_types=["128x3x224x224xf32"])

RESNET50_FP32_JAX_3X224X224XF32_BATCH256 = common_definitions.Model(
    id=unique_ids.MODEL_RESNET50_FP32_JAX_3X224X224XF32_BATCH256,
    name="RESNET50_FP32_JAX_3X224X224XF32_BATCH256",
    tags=RESNET50_TAGS + ["batch-256"],
    source_type=common_definitions.ModelSourceType.EXPORTED_STABLEHLO_MLIR,
    source_url=f"{GCS_ARTIFACT_ROOT_DIR}/RESNET50/batch_256/stablehlo.mlirbc",
    entry_function="main",
    input_types=["256x3x224x224xf32"])

RESNET50_FP32_JAX_3X224X224XF32_BATCH2048 = common_definitions.Model(
    id=unique_ids.MODEL_RESNET50_FP32_JAX_3X224X224XF32_BATCH2048,
    name="RESNET50_FP32_JAX_3X224X224XF32_BATCH2048",
    tags=RESNET50_TAGS + ["batch-2048"],
    source_type=common_definitions.ModelSourceType.EXPORTED_STABLEHLO_MLIR,
    source_url=f"{GCS_ARTIFACT_ROOT_DIR}/RESNET50/batch_2048/stablehlo.mlirbc",
    entry_function="main",
    input_types=["2048x3x224x224xf32"])

# Derived from https://huggingface.co/docs/transformers/model_doc/bert#transformers.FlaxBertModel.
BERT_LARGE_TAGS = ["fp32", "seqlen384", "jax", "bert-variant"]

BERT_LARGE_FP32_JAX_384XI32_BATCH1 = common_definitions.Model(
    id=unique_ids.MODEL_BERT_LARGE_FP32_JAX_384XI32_BATCH1,
    name="BERT_LARGE_JAX_384XI32_BATCH1",
    tags=BERT_LARGE_TAGS + ["batch-1"],
    source_type=common_definitions.ModelSourceType.EXPORTED_STABLEHLO_MLIR,
    source_url=f"{GCS_ARTIFACT_ROOT_DIR}/BERT_LARGE/batch_1/stablehlo.mlirbc",
    entry_function="main",
    input_types=["1x384xi32", "1x384xi32"])

BERT_LARGE_FP32_JAX_384XI32_BATCH16 = common_definitions.Model(
    id=unique_ids.MODEL_BERT_LARGE_FP32_JAX_384XI32_BATCH16,
    name="BERT_LARGE_JAX_384XI32_BATCH16",
    tags=BERT_LARGE_TAGS + ["batch-16"],
    source_type=common_definitions.ModelSourceType.EXPORTED_STABLEHLO_MLIR,
    source_url=f"{GCS_ARTIFACT_ROOT_DIR}/BERT_LARGE/batch_16/stablehlo.mlirbc",
    entry_function="main",
    input_types=["16x384xi32", "16x384xi32"])

BERT_LARGE_FP32_JAX_384XI32_BATCH24 = common_definitions.Model(
    id=unique_ids.MODEL_BERT_LARGE_FP32_JAX_384XI32_BATCH24,
    name="BERT_LARGE_JAX_384XI32_BATCH24",
    tags=BERT_LARGE_TAGS + ["batch-24"],
    source_type=common_definitions.ModelSourceType.EXPORTED_STABLEHLO_MLIR,
    source_url=f"{GCS_ARTIFACT_ROOT_DIR}/BERT_LARGE/batch_24/stablehlo.mlirbc",
    entry_function="main",
    input_types=["24x384xi32", "24x384xi32"])

BERT_LARGE_FP32_JAX_384XI32_BATCH32 = common_definitions.Model(
    id=unique_ids.MODEL_BERT_LARGE_FP32_JAX_384XI32_BATCH32,
    name="BERT_LARGE_JAX_384XI32_BATCH32",
    tags=BERT_LARGE_TAGS + ["batch-32"],
    source_type=common_definitions.ModelSourceType.EXPORTED_STABLEHLO_MLIR,
    source_url=f"{GCS_ARTIFACT_ROOT_DIR}/BERT_LARGE/batch_32/stablehlo.mlirbc",
    entry_function="main",
    input_types=["32x384xi32", "32x384xi32"])

BERT_LARGE_FP32_JAX_384XI32_BATCH48 = common_definitions.Model(
    id=unique_ids.MODEL_BERT_LARGE_FP32_JAX_384XI32_BATCH48,
    name="BERT_LARGE_JAX_384XI32_BATCH48",
    tags=BERT_LARGE_TAGS + ["batch-48"],
    source_type=common_definitions.ModelSourceType.EXPORTED_STABLEHLO_MLIR,
    source_url=f"{GCS_ARTIFACT_ROOT_DIR}/BERT_LARGE/batch_48/stablehlo.mlirbc",
    entry_function="main",
    input_types=["48x384xi32", "48x384xi32"])

BERT_LARGE_FP32_JAX_384XI32_BATCH64 = common_definitions.Model(
    id=unique_ids.MODEL_BERT_LARGE_FP32_JAX_384XI32_BATCH64,
    name="BERT_LARGE_JAX_384XI32_BATCH64",
    tags=BERT_LARGE_TAGS + ["batch-64"],
    source_type=common_definitions.ModelSourceType.EXPORTED_STABLEHLO_MLIR,
    source_url=f"{GCS_ARTIFACT_ROOT_DIR}/BERT_LARGE/batch_64/stablehlo.mlirbc",
    entry_function="main",
    input_types=["64x384xi32", "64x384xi32"])

BERT_LARGE_FP32_JAX_384XI32_BATCH512 = common_definitions.Model(
    id=unique_ids.MODEL_BERT_LARGE_FP32_JAX_384XI32_BATCH512,
    name="BERT_LARGE_JAX_384XI32_BATCH512",
    tags=BERT_LARGE_TAGS + ["batch-512"],
    source_type=common_definitions.ModelSourceType.EXPORTED_STABLEHLO_MLIR,
    source_url=f"{GCS_ARTIFACT_ROOT_DIR}/BERT_LARGE/batch_512/stablehlo.mlirbc",
    entry_function="main",
    input_types=["512x384xi32", "512x384xi32"])

BERT_LARGE_FP32_JAX_384XI32_BATCH1024 = common_definitions.Model(
    id=unique_ids.MODEL_BERT_LARGE_FP32_JAX_384XI32_BATCH1024,
    name="BertLargeJAXBatch1024",
    tags=BERT_LARGE_TAGS + ["batch-1024"],
    source_type=common_definitions.ModelSourceType.EXPORTED_STABLEHLO_MLIR,
    source_url=f"{GCS_ARTIFACT_ROOT_DIR}/BERT_LARGE/batch_1024/stablehlo.mlirbc",
    entry_function="main",
    input_types=["1024x384xi32", "1024x384xi32"])

BERT_LARGE_FP32_JAX_384XI32_BATCH1280 = common_definitions.Model(
    id=unique_ids.MODEL_BERT_LARGE_FP32_JAX_384XI32_BATCH1280,
    name="BertLargeJAXBatch1280",
    tags=BERT_LARGE_TAGS + ["batch-1280"],
    source_type=common_definitions.ModelSourceType.EXPORTED_STABLEHLO_MLIR,
    source_url=f"{GCS_ARTIFACT_ROOT_DIR}/BERT_LARGE/batch_1280/stablehlo.mlirbc",
    entry_function="main",
    input_types=["1280x384xi32", "1280x384xi32"])

# Derived from https://huggingface.co/docs/transformers/model_doc/t5#transformers.FlaxT5Model
T5_TAGS = ["fp32", "transformer-encoder", "transformer-decoder", "t5"]

T5_LARGE_FP32_JAX_512XI32_BATCH1 = common_definitions.Model(
    id=unique_ids.MODEL_T5_LARGE_FP32_JAX_512XI32_BATCH1,
    name="T5_LARGE_FP32_JAX_512XI32_BATCH1",
    tags=T5_TAGS + ["batch-1"],
    source_type=common_definitions.ModelSourceType.EXPORTED_STABLEHLO_MLIR,
    source_url=f"{GCS_ARTIFACT_ROOT_DIR}/T5_LARGE/batch_1/stablehlo.mlirbc",
    entry_function="main",
    input_types=["1x512xi32", "1x512xi32"])

T5_LARGE_FP32_JAX_512XI32_BATCH16 = common_definitions.Model(
    id=unique_ids.MODEL_T5_LARGE_FP32_JAX_512XI32_BATCH16,
    name="T5_LARGE_FP32_JAX_512XI32_BATCH16",
    tags=T5_TAGS + ["batch-16"],
    source_type=common_definitions.ModelSourceType.EXPORTED_STABLEHLO_MLIR,
    source_url=f"{GCS_ARTIFACT_ROOT_DIR}/T5_LARGE/batch_16/stablehlo.mlirbc",
    entry_function="main",
    input_types=["16x512xi32", "16x512xi32"])

T5_LARGE_FP32_JAX_512XI32_BATCH24 = common_definitions.Model(
    id=unique_ids.MODEL_T5_LARGE_FP32_JAX_512XI32_BATCH24,
    name="T5_LARGE_FP32_JAX_512XI32_BATCH24",
    tags=T5_TAGS + ["batch-24"],
    source_type=common_definitions.ModelSourceType.EXPORTED_STABLEHLO_MLIR,
    source_url=f"{GCS_ARTIFACT_ROOT_DIR}/T5_LARGE/batch_24/stablehlo.mlirbc",
    entry_function="main",
    input_types=["24x512xi32", "24x512xi32"])

T5_LARGE_FP32_JAX_512XI32_BATCH32 = common_definitions.Model(
    id=unique_ids.MODEL_T5_LARGE_FP32_JAX_512XI32_BATCH32,
    name="T5_LARGE_FP32_JAX_512XI32_BATCH32",
    tags=T5_TAGS + ["batch-32"],
    source_type=common_definitions.ModelSourceType.EXPORTED_STABLEHLO_MLIR,
    source_url=f"{GCS_ARTIFACT_ROOT_DIR}/T5_LARGE/batch_32/stablehlo.mlirbc",
    entry_function="main",
    input_types=["32x512xi32", "32x512xi32"])

T5_LARGE_FP32_JAX_512XI32_BATCH48 = common_definitions.Model(
    id=unique_ids.MODEL_T5_LARGE_FP32_JAX_512XI32_BATCH48,
    name="T5_LARGE_FP32_JAX_512XI32_BATCH48",
    tags=T5_TAGS + ["batch-48"],
    source_type=common_definitions.ModelSourceType.EXPORTED_STABLEHLO_MLIR,
    source_url=f"{GCS_ARTIFACT_ROOT_DIR}/T5_LARGE/batch_48/stablehlo.mlirbc",
    entry_function="main",
    input_types=["48x512xi32", "48x512xi32"])

T5_LARGE_FP32_JAX_512XI32_BATCH64 = common_definitions.Model(
    id=unique_ids.MODEL_T5_LARGE_FP32_JAX_512XI32_BATCH64,
    name="T5_LARGE_FP32_JAX_512XI32_BATCH64",
    tags=T5_TAGS + ["batch-64"],
    source_type=common_definitions.ModelSourceType.EXPORTED_STABLEHLO_MLIR,
    source_url=f"{GCS_ARTIFACT_ROOT_DIR}/T5_LARGE/batch_64/stablehlo.mlirbc",
    entry_function="main",
    input_types=["64x512xi32", "64x512xi32"])

T5_LARGE_FP32_JAX_512XI32_BATCH512 = common_definitions.Model(
    id=unique_ids.MODEL_T5_LARGE_FP32_JAX_512XI32_BATCH512,
    name="T5_LARGE_FP32_JAX_512XI32_BATCH512",
    tags=T5_TAGS + ["batch-512"],
    source_type=common_definitions.ModelSourceType.EXPORTED_STABLEHLO_MLIR,
    source_url=f"{GCS_ARTIFACT_ROOT_DIR}/T5_LARGE/batch_512/stablehlo.mlirbc",
    entry_function="main",
    input_types=["512x512xi32", "512x512xi32"])
