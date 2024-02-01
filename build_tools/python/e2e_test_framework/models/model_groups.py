## Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Defines the groups of models."""

from e2e_test_framework.definitions import common_definitions
from e2e_test_framework.models import (
    matmul,
    tflite_models,
    torch_models,
    tf_models,
    jax_models,
)

# x86 models, single batch.

# A list of models with thread configurations.
# Note `0` represents sync execution.
X86_64_BENCHMARK_CONFIG = [
    common_definitions.CpuBenchmarkConfig(
        model=jax_models.GPT2LMHEAD_FP32_JAX_512XI32_BATCHES[1], threads=[30]
    ),
    common_definitions.CpuBenchmarkConfig(
        model=jax_models.GPT2LMHEAD_FP32_JAX_512XI32_BATCHES[64], threads=[30]
    ),
    common_definitions.CpuBenchmarkConfig(
        model=jax_models.GPT2LMHEAD_FP32_JAX_512XI32_BATCHES[128], threads=[30]
    ),
    # Tiny models.
    common_definitions.CpuBenchmarkConfig(
        model=tflite_models.PERSON_DETECT_INT8, threads=[0, 1]
    ),
    common_definitions.CpuBenchmarkConfig(
        model=tflite_models.MOBILENET_V3SMALL, threads=[0, 1]
    ),
    # Small models.
    common_definitions.CpuBenchmarkConfig(
        model=tflite_models.DEEPLABV3_FP32, threads=[1, 8]
    ),
    common_definitions.CpuBenchmarkConfig(
        model=tflite_models.EFFICIENTNET_INT8, threads=[1, 8]
    ),
    common_definitions.CpuBenchmarkConfig(
        model=tflite_models.MOBILENET_V1, threads=[1, 8]
    ),
    common_definitions.CpuBenchmarkConfig(
        model=tflite_models.MOBILENET_V2, threads=[1, 8]
    ),
    common_definitions.CpuBenchmarkConfig(
        model=tflite_models.MOBILENET_V2_INT8, threads=[1, 8]
    ),
    common_definitions.CpuBenchmarkConfig(
        model=tflite_models.MOBILESSD_FP32, threads=[1, 8]
    ),
    common_definitions.CpuBenchmarkConfig(
        model=tflite_models.POSENET_FP32, threads=[1, 8]
    ),
    # Medium models.
    common_definitions.CpuBenchmarkConfig(
        model=tflite_models.MOBILEBERT_FP16, threads=[1, 15]
    ),
    common_definitions.CpuBenchmarkConfig(
        model=tflite_models.MOBILEBERT_FP32, threads=[1, 15]
    ),
    common_definitions.CpuBenchmarkConfig(
        model=tflite_models.MOBILEBERT_INT8, threads=[1, 15]
    ),
    common_definitions.CpuBenchmarkConfig(
        model=tf_models.EFFICIENTNET_V2_S_FP32, threads=[1, 15]
    ),
    common_definitions.CpuBenchmarkConfig(
        model=tf_models.MINILM_L12_H384_UNCASED_INT32_SEQLEN128, threads=[1, 15]
    ),
    common_definitions.CpuBenchmarkConfig(
        model=tf_models.GPT2_117M_1x4_FP32_TF, threads=[1, 15]
    ),
    common_definitions.CpuBenchmarkConfig(
        model=tf_models.GPT2_117M_1x1_FP32_TF, threads=[1, 15]
    ),
    # Large models.
    common_definitions.CpuBenchmarkConfig(
        model=tf_models.BERT_FOR_MASKED_LM_FP32_SEQLEN512, threads=[30]
    ),
    common_definitions.CpuBenchmarkConfig(
        model=tf_models.BERT_LARGE_TF_FP32_SEQLEN384, threads=[30]
    ),
    common_definitions.CpuBenchmarkConfig(
        model=torch_models.BERT_LARGE_384_FP32_TORCH_BATCHES[1], threads=[30]
    ),
    common_definitions.CpuBenchmarkConfig(
        model=torch_models.FALCON7B_1X100XI64_GPTQ_TORCH, threads=[30]
    ),
    common_definitions.CpuBenchmarkConfig(
        model=torch_models.FALCON7B_INT4_1X100XI64_GPTQ_TORCH, threads=[30]
    ),
]

X86_64_BENCHMARK_CONFIG_LARGE = [
    common_definitions.CpuBenchmarkConfig(
        model=jax_models.BERT_LARGE_FP32_JAX_384XI32_BATCHES[1], threads=[30]
    ),
    common_definitions.CpuBenchmarkConfig(
        model=jax_models.BERT_LARGE_FP32_JAX_384XI32_BATCHES[32], threads=[30]
    ),
    common_definitions.CpuBenchmarkConfig(
        model=jax_models.BERT_LARGE_FP32_JAX_384XI32_BATCHES[64], threads=[30]
    ),
    common_definitions.CpuBenchmarkConfig(
        model=torch_models.BERT_LARGE_384_FP32_TORCH_BATCHES[24], threads=[30]
    ),
    common_definitions.CpuBenchmarkConfig(
        model=torch_models.BERT_LARGE_384_FP32_TORCH_BATCHES[48], threads=[30]
    ),
    common_definitions.CpuBenchmarkConfig(
        model=jax_models.RESNET50_FP32_JAX_3X224X224XF32_BATCHES[1], threads=[30]
    ),
    common_definitions.CpuBenchmarkConfig(
        model=jax_models.RESNET50_FP32_JAX_3X224X224XF32_BATCHES[64], threads=[30]
    ),
    common_definitions.CpuBenchmarkConfig(
        model=jax_models.RESNET50_FP32_JAX_3X224X224XF32_BATCHES[128], threads=[30]
    ),
    common_definitions.CpuBenchmarkConfig(
        model=jax_models.T5_LARGE_FP32_JAX_512XI32_BATCHES[1], threads=[30]
    ),
    common_definitions.CpuBenchmarkConfig(
        model=jax_models.T5_LARGE_FP32_JAX_512XI32_BATCHES[16], threads=[30]
    ),
    common_definitions.CpuBenchmarkConfig(
        model=jax_models.T5_LARGE_FP32_JAX_512XI32_BATCHES[32], threads=[30]
    ),
]

# Microkernels.

MICRO_MATMUL = [
    matmul.MATMUL_3456X1024X2048_FP16_MLIR,
    matmul.MATMUL_3456X1024X2048_FP32_MLIR,
    matmul.MATMUL_2560X2560X2560_FP16_MLIR,
    matmul.MATMUL_2560X2560X2560_FP32_MLIR,
    matmul.MATMUL_2564x2564x2564_FP32_MLIR,
    matmul.MATMUL_2562x2564x2562_FP32_MLIR,
    matmul.MATMUL_2562x2561x2561_FP32_MLIR,
    matmul.MATMUL_123x2561x2561_FP32_MLIR,
]

MICRO_MATMUL_SPLITK = [
    matmul.MATMUL_128X256X8192_FP16_MLIR,
    matmul.MATMUL_128X256X8192_FP32_MLIR,
]

# GPU model groups.

CUDA_MODELS = [
    tf_models.EFFICIENTNET_V2_S_FP32,
    tf_models.MINILM_L12_H384_UNCASED_INT32_SEQLEN128,
    tf_models.BERT_FOR_MASKED_LM_FP32_SEQLEN512,
    tf_models.BERT_LARGE_TF_FP32_SEQLEN384,
    # PyTorch model are disabled due to https://github.com/openxla/iree/issues/14993.
    # torch_models.MODEL_CLIP_TEXT_SEQLEN64_FP32_TORCH,
    # torch_models.MODEL_UNET_2D_FP32_TORCH,
]

VULKAN_MODELS = [
    # PyTorch model are disabled due to https://github.com/openxla/iree/issues/14993.
    # torch_models.MODEL_CLIP_TEXT_SEQLEN64_FP32_TORCH,
    # torch_models.MODEL_UNET_2D_FP32_TORCH,
]
