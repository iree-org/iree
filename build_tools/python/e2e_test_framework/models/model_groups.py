## Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Defines the groups of models."""

from e2e_test_framework.models import matmul, tflite_models, torch_models, tf_models

# A list of models with thread configurations.
# Note `0` represents sync execution.
x86_64_MODELS_AND_THREADS = [
    # Tiny models.
    (tflite_models.PERSON_DETECT_INT8, [0, 1]),
    (tflite_models.MOBILENET_V3SMALL, [0, 1]),
    # Small models.
    (tflite_models.DEEPLABV3_FP32, [1, 8]),
    (tflite_models.EFFICIENTNET_INT8, [1, 8]),
    (tflite_models.MOBILENET_V1, [1, 8]),
    (tflite_models.MOBILENET_V2, [1, 8]),
    (tflite_models.MOBILENET_V2_INT8, [1, 8]),
    (tflite_models.MOBILESSD_FP32, [1, 8]),
    (tflite_models.POSENET_FP32, [1, 8]),
    # Medium models.
    # TODO: Add 13 threads once we move to new hardware.
    (tflite_models.MOBILEBERT_FP16, [1, 8]),
    (tflite_models.MOBILEBERT_FP32, [1, 8]),
    (tflite_models.MOBILEBERT_INT8, [1, 8]),
    (tf_models.EFFICIENTNET_V2_S_FP32, [1, 8]),
    (tf_models.MINILM_L12_H384_UNCASED_INT32_SEQLEN128, [1, 8]),
    (tf_models.RESNET50_1X3X224X224_FP32_TF, [1, 8]),
    (torch_models.EFFICIENTNET_V2_S_FP32_TORCH, [1, 8]),
    # Large models.
    # TODO: These models should be running at 8, 13, 28 threads but we use 8 for now until new hardware becomes available.
    (tf_models.BERT_FOR_MASKED_LM_FP32_SEQLEN512, [8]),
    (tf_models.BERT_LARGE_TF_FP32_SEQLEN384, [8]),
    (torch_models.EFFICIENTNET_B7_FP32_TORCH, [8]),
]

# A subset of `x86_64_MODELS_AND_THREADS`.
x86_64_MODELS_AND_THREADS_EXPERIMENTAL = [
    # Tiny models.
    (tflite_models.PERSON_DETECT_INT8, [1]),
    (tflite_models.MOBILENET_V3SMALL, [1]),
    # Small models.
    (tflite_models.DEEPLABV3_FP32, [8]),
    (tflite_models.EFFICIENTNET_INT8, [8]),
    (tflite_models.MOBILENET_V2, [8]),
    (tflite_models.MOBILENET_V2_INT8, [8]),
    (tflite_models.MOBILESSD_FP32, [8]),
    (tflite_models.POSENET_FP32, [8]),
    # Medium models.
    (tflite_models.MOBILEBERT_FP32, [8]),
    (tflite_models.MOBILEBERT_INT8, [8]),
    (tf_models.EFFICIENTNET_V2_S_FP32, [8]),
    (tf_models.MINILM_L12_H384_UNCASED_INT32_SEQLEN128, [8]),
    # Disabled due to https://github.com/openxla/iree/issues/11174.
    # (tf_models.RESNET50_1X3X224X224_FP32_TF, [8]),
    # Disabled due to https://github.com/openxla/iree/issues/12772.
    # (torch_models.EFFICIENTNET_V2_S_FP32_TORCH, [8]),
    # Large models.
    (tf_models.BERT_LARGE_TF_FP32_SEQLEN384, [8]),
    # Disabled due to https://github.com/openxla/iree/issues/12772.
    # (torch_models.EFFICIENTNET_B7_FP32_TORCH, [8]),
]

# BERT-Large in various batch sizes.
BERT_LARGE_TORCH_BATCHES = [
    torch_models.BERT_LARGE_1X384_FP32_TORCH,
    torch_models.BERT_LARGE_16X384_FP32_TORCH,
    torch_models.BERT_LARGE_24X384_FP32_TORCH,
    torch_models.BERT_LARGE_32X384_FP32_TORCH,
    torch_models.BERT_LARGE_48X384_FP32_TORCH,
    torch_models.BERT_LARGE_64X384_FP32_TORCH,
    torch_models.BERT_LARGE_512X384_FP32_TORCH,
    torch_models.BERT_LARGE_1024X384_FP32_TORCH,
    torch_models.BERT_LARGE_1280X384_FP32_TORCH,
]

RESNET50_TORCH_BATCHES = [
    torch_models.RESNET50_1X3X224X224_FP32_TORCH,
    torch_models.RESNET50_8X3X224X224_FP32_TORCH,
    torch_models.RESNET50_64X3X224X224_FP32_TORCH,
    torch_models.RESNET50_128X3X224X224_FP32_TORCH,
    torch_models.RESNET50_256X3X224X224_FP32_TORCH,
    torch_models.RESNET50_2048X3X224X224_FP32_TORCH,
]

MICRO_MATMUL = [
    matmul.MATMUL_3456X1024X2048_FP16_MLIR,
    matmul.MATMUL_3456X1024X2048_FP32_MLIR,
    matmul.MATMUL_2560X2560X2560_FP16_MLIR,
    matmul.MATMUL_2560X2560X2560_FP32_MLIR,
]

MICRO_MATMUL_SPLITK = [
    matmul.MATMUL_128X256X8192_FP16_MLIR,
    matmul.MATMUL_128X256X8192_FP32_MLIR,
]

BERT_LARGE_TF_BATCHES = [
    tf_models.BERT_LARGE_1X384_FP32_TF,
    tf_models.BERT_LARGE_16X384_FP32_TF,
    tf_models.BERT_LARGE_24X384_FP32_TF,
    tf_models.BERT_LARGE_32X384_FP32_TF,
    tf_models.BERT_LARGE_48X384_FP32_TF,
    tf_models.BERT_LARGE_64X384_FP32_TF,
    tf_models.BERT_LARGE_512X384_FP32_TF,
    tf_models.BERT_LARGE_1024X384_FP32_TF,
    tf_models.BERT_LARGE_1280X384_FP32_TF,
]

RESNET50_TF_BATCHES = [
    tf_models.RESNET50_1X3X224X224_FP32_TF,
    tf_models.RESNET50_8X3X224X224_FP32_TF,
    tf_models.RESNET50_64X3X224X224_FP32_TF,
    tf_models.RESNET50_128X3X224X224_FP32_TF,
    tf_models.RESNET50_256X3X224X224_FP32_TF,
    tf_models.RESNET50_2048X3X224X224_FP32_TF,
]

T5_LARGE_TF_BATCHES = [
    tf_models.T5_LARGE_1x512_FP32_TF,
    tf_models.T5_LARGE_16x512_FP32_TF,
    tf_models.T5_LARGE_24x512_FP32_TF,
    tf_models.T5_LARGE_32x512_FP32_TF,
    tf_models.T5_LARGE_48x512_FP32_TF,
    tf_models.T5_LARGE_64x512_FP32_TF,
    tf_models.T5_LARGE_512x512_FP32_TF,
]

CUDA_MODELS = [
    tf_models.EFFICIENTNET_V2_S_FP32,
    tf_models.MINILM_L12_H384_UNCASED_INT32_SEQLEN128,
    tf_models.BERT_FOR_MASKED_LM_FP32_SEQLEN512,
    tf_models.BERT_LARGE_TF_FP32_SEQLEN384,
    torch_models.MODEL_CLIP_TEXT_SEQLEN64_FP32_TORCH,
    torch_models.MODEL_UNET_2D_FP32_TORCH,
    torch_models.EFFICIENTNET_B7_FP32_TORCH,
]

CUDA_MODELS_LONG = RESNET50_TF_BATCHES + BERT_LARGE_TF_BATCHES + T5_LARGE_TF_BATCHES
