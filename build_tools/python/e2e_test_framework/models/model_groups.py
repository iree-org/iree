## Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Defines the groups of models."""

from e2e_test_framework.models import tflite_models, torch_models

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
    (torch_models.EFFICIENTNET_V2_S_FP32_TORCH, [1, 8]),
    # Large models.
    # TODO: These models should be running at 8, 13, 28 threads but we use 8 for now until new hardware becomes available.
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
    # Disabled due to https://github.com/openxla/iree/issues/11174.
    # (tf_models.RESNET50_TF_FP32, [8]),
    # Disabled due to https://github.com/openxla/iree/issues/12772.
    # (torch_models.EFFICIENTNET_V2_S_FP32_TORCH, [8]),
    # Large models.
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
    # Disabled due to https://github.com/openxla/iree/issues/12774.
    #torch_models.BERT_LARGE_512X384_FP32_TORCH,
    #torch_models.BERT_LARGE_1024X384_FP32_TORCH,
    #torch_models.BERT_LARGE_1280X384_FP32_TORCH,
]

RESNET50_TORCH_BATCHES = [
    torch_models.RESNET50_1X3X224X224_FP32_TORCH,
    torch_models.RESNET50_8X3X224X224_FP32_TORCH,
    # Disabled due to https://github.com/openxla/iree/issues/12774.
    #torch_models.RESNET50_64X3X224X224_FP32_TORCH,
    #torch_models.RESNET50_128X3X224X224_FP32_TORCH,
    #torch_models.RESNET50_256X3X224X224_FP32_TORCH,
    #torch_models.RESNET50_2048X3X224X224_FP32_TORCH,
]

CUDA_MODELS = BERT_LARGE_TORCH_BATCHES + RESNET50_TORCH_BATCHES + [
    torch_models.MODEL_CLIP_TEXT_SEQLEN64_FP32_TORCH,
    torch_models.MODEL_UNET_2D_FP32_TORCH,
    torch_models.EFFICIENTNET_B7_FP32_TORCH,
]
