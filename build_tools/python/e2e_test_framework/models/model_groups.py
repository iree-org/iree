## Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Defines the groups of models."""

from e2e_test_framework.definitions import common_definitions
from e2e_test_framework.models import matmul, tflite_models, torch_models, tf_models, jax_models

# x86 models, single batch.

# A list of models with thread configurations.
# Note `0` represents sync execution.
X86_64_BENCHMARK_CONFIG = [
    # Tiny models.
    common_definitions.CpuBenchmarkConfig(
        model=tflite_models.PERSON_DETECT_INT8, threads=[0, 1]),
    common_definitions.CpuBenchmarkConfig(model=tflite_models.MOBILENET_V3SMALL,
                                          threads=[0, 1]),
    # Small models.
    common_definitions.CpuBenchmarkConfig(model=tflite_models.DEEPLABV3_FP32,
                                          threads=[1, 8]),
    common_definitions.CpuBenchmarkConfig(model=tflite_models.EFFICIENTNET_INT8,
                                          threads=[1, 8]),
    common_definitions.CpuBenchmarkConfig(model=tflite_models.MOBILENET_V1,
                                          threads=[1, 8]),
    common_definitions.CpuBenchmarkConfig(model=tflite_models.MOBILENET_V2,
                                          threads=[1, 8]),
    common_definitions.CpuBenchmarkConfig(model=tflite_models.MOBILENET_V2_INT8,
                                          threads=[1, 8]),
    common_definitions.CpuBenchmarkConfig(model=tflite_models.MOBILESSD_FP32,
                                          threads=[1, 8]),
    common_definitions.CpuBenchmarkConfig(model=tflite_models.POSENET_FP32,
                                          threads=[1, 8]),
    # Medium models.
    # TODO: Add 13 threads once we move to new hardware.
    common_definitions.CpuBenchmarkConfig(model=tflite_models.MOBILEBERT_FP16,
                                          threads=[1, 8]),
    common_definitions.CpuBenchmarkConfig(model=tflite_models.MOBILEBERT_FP32,
                                          threads=[1, 8]),
    common_definitions.CpuBenchmarkConfig(model=tflite_models.MOBILEBERT_INT8,
                                          threads=[1, 8]),
    common_definitions.CpuBenchmarkConfig(
        model=tf_models.EFFICIENTNET_V2_S_FP32, threads=[1, 8]),
    common_definitions.CpuBenchmarkConfig(
        model=tf_models.MINILM_L12_H384_UNCASED_INT32_SEQLEN128, threads=[1,
                                                                          8]),
    common_definitions.CpuBenchmarkConfig(
        model=torch_models.EFFICIENTNET_V2_S_FP32_TORCH, threads=[1, 8]),
    # Large models.
    # TODO: These models should be running at 8, 13, 28 threads but we use 8 for now until new hardware becomes available.
    common_definitions.CpuBenchmarkConfig(
        model=tf_models.BERT_FOR_MASKED_LM_FP32_SEQLEN512, threads=[8]),
    common_definitions.CpuBenchmarkConfig(
        model=tf_models.BERT_LARGE_TF_FP32_SEQLEN384, threads=[8]),
    common_definitions.CpuBenchmarkConfig(
        model=torch_models.EFFICIENTNET_B7_FP32_TORCH, threads=[8]),
]

# A subset of `x86_64_MODELS_AND_THREADS`.
X86_64_BENCHMARK_CONFIG_EXPERIMENTAL = [
    # Tiny models.
    common_definitions.CpuBenchmarkConfig(
        model=tflite_models.PERSON_DETECT_INT8, threads=[1]),
    common_definitions.CpuBenchmarkConfig(model=tflite_models.MOBILENET_V3SMALL,
                                          threads=[1]),
    # Small models.
    common_definitions.CpuBenchmarkConfig(model=tflite_models.DEEPLABV3_FP32,
                                          threads=[8]),
    common_definitions.CpuBenchmarkConfig(model=tflite_models.EFFICIENTNET_INT8,
                                          threads=[8]),
    common_definitions.CpuBenchmarkConfig(model=tflite_models.MOBILENET_V2,
                                          threads=[8]),
    common_definitions.CpuBenchmarkConfig(model=tflite_models.MOBILENET_V2_INT8,
                                          threads=[8]),
    common_definitions.CpuBenchmarkConfig(model=tflite_models.MOBILESSD_FP32,
                                          threads=[8]),
    common_definitions.CpuBenchmarkConfig(model=tflite_models.POSENET_FP32,
                                          threads=[8]),
    # Medium models.
    common_definitions.CpuBenchmarkConfig(model=tflite_models.MOBILEBERT_FP32,
                                          threads=[8]),
    common_definitions.CpuBenchmarkConfig(model=tflite_models.MOBILEBERT_INT8,
                                          threads=[8]),
    common_definitions.CpuBenchmarkConfig(
        model=tf_models.EFFICIENTNET_V2_S_FP32, threads=[8]),
    common_definitions.CpuBenchmarkConfig(
        model=tf_models.MINILM_L12_H384_UNCASED_INT32_SEQLEN128, threads=[8]),
    # Disabled due to https://github.com/openxla/iree/issues/12772.
    # common_definitions.CpuBenchmarkConfig(model=torch_models.EFFICIENTNET_V2_S_FP32_TORCH, threads=[8]),
    # Large models.
    common_definitions.CpuBenchmarkConfig(
        model=tf_models.BERT_LARGE_TF_FP32_SEQLEN384, threads=[8]),
    # Disabled due to https://github.com/openxla/iree/issues/12772.
    # common_definitions.CpuBenchmarkConfig(model=torch_models.EFFICIENTNET_B7_FP32_TORCH, threads=[8]),
]

X86_64_BENCHMARK_CONFIG_LONG = [
    common_definitions.CpuBenchmarkConfig(
        model=tf_models.BERT_LARGE_384_FP32_TF_BATCHES[1], threads=[8]),
    common_definitions.CpuBenchmarkConfig(
        model=tf_models.BERT_LARGE_384_FP32_TF_BATCHES[32], threads=[8]),
    common_definitions.CpuBenchmarkConfig(
        model=tf_models.BERT_LARGE_384_FP32_TF_BATCHES[64], threads=[8]),
    common_definitions.CpuBenchmarkConfig(
        model=tf_models.RESNET50_3X224X224_FP32_TF_BATCHES[1], threads=[8]),
    common_definitions.CpuBenchmarkConfig(
        model=tf_models.RESNET50_3X224X224_FP32_TF_BATCHES[64], threads=[8]),
    common_definitions.CpuBenchmarkConfig(
        model=tf_models.RESNET50_3X224X224_FP32_TF_BATCHES[128], threads=[8]),
    common_definitions.CpuBenchmarkConfig(
        model=tf_models.T5_LARGE_512_FP32_TF_BATCHES[1], threads=[8]),
    common_definitions.CpuBenchmarkConfig(
        model=tf_models.T5_LARGE_512_FP32_TF_BATCHES[16], threads=[8]),
    common_definitions.CpuBenchmarkConfig(
        model=tf_models.T5_LARGE_512_FP32_TF_BATCHES[32], threads=[8]),
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
]

MICRO_MATMUL_SPLITK = [
    matmul.MATMUL_128X256X8192_FP16_MLIR,
    matmul.MATMUL_128X256X8192_FP32_MLIR,
]

# Batched Torch models.

BERT_LARGE_TORCH_BATCHES = list(
    torch_models.BERT_LARGE_384_FP32_TORCH_BATCHES.values())

BERT_LARGE_FP16_TORCH_BATCHES = [
    model for batch_size, model in
    torch_models.BERT_LARGE_384_FP16_TORCH_BATCHES.items()
    # Batchsize 1 is included seperately in CUDA_MODELS
    if batch_size != 1
]

RESNET50_TORCH_BATCHES = list(
    torch_models.RESNET50_3X224X224_FP32_TORCH_BATCHES.values())

RESNET50_FP16_TORCH_BATCHES = list(
    torch_models.RESNET50_3X224X224_FP16_TORCH_BATCHES.values())

# Batched Tensorflow models.

BERT_LARGE_TF_BATCHES = list(tf_models.BERT_LARGE_384_FP32_TF_BATCHES.values())

RESNET50_TF_BATCHES = list(
    tf_models.RESNET50_3X224X224_FP32_TF_BATCHES.values())

T5_LARGE_TF_BATCHES = [
    model
    for batch_size, model in tf_models.T5_LARGE_512_FP32_TF_BATCHES.items()
    # Disabled due to https://github.com/openxla/iree/issues/13801.
    if batch_size != 512
]

# Batched JAX models.

RESNET50_JAX_BATCHES = list(
    jax_models.RESNET50_FP32_JAX_3X224X224XF32_BATCHES.values())

BERT_LARGE_JAX_BATCHES = list(
    jax_models.BERT_LARGE_FP32_JAX_384XI32_BATCHES.values())

T5_LARGE_JAX_BATCHES = list(
    jax_models.T5_LARGE_FP32_JAX_512XI32_BATCHES.values())

# GPU model groups.

CUDA_MODELS = [
    tf_models.EFFICIENTNET_V2_S_FP32,
    tf_models.MINILM_L12_H384_UNCASED_INT32_SEQLEN128,
    tf_models.BERT_FOR_MASKED_LM_FP32_SEQLEN512,
    tf_models.BERT_LARGE_TF_FP32_SEQLEN384,
    torch_models.MODEL_CLIP_TEXT_SEQLEN64_FP32_TORCH,
    torch_models.MODEL_UNET_2D_FP32_TORCH,
    torch_models.EFFICIENTNET_B7_FP32_TORCH,
    torch_models.BERT_LARGE_384_FP16_TORCH_BATCHES[1],
    torch_models.EFFICIENTNET_V2_S_FP16_TORCH,
]

CUDA_MODELS_LONG = (RESNET50_TF_BATCHES + BERT_LARGE_TF_BATCHES +
                    T5_LARGE_TF_BATCHES + BERT_LARGE_TORCH_BATCHES +
                    RESNET50_TORCH_BATCHES + RESNET50_FP16_TORCH_BATCHES +
                    BERT_LARGE_FP16_TORCH_BATCHES + BERT_LARGE_JAX_BATCHES +
                    RESNET50_JAX_BATCHES + T5_LARGE_JAX_BATCHES)

VULKAN_MODELS = [
    torch_models.MODEL_CLIP_TEXT_SEQLEN64_FP32_TORCH,
    torch_models.MODEL_UNET_2D_FP32_TORCH,
]
