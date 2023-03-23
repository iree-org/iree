## Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Defines the groups of models."""

from e2e_test_framework.models import tf_models, tflite_models, torch_models

# Small models that require less computational resources.
SMALL = [
    tflite_models.DEEPLABV3_FP32,
    tflite_models.EFFICIENTNET_INT8,
    tflite_models.MOBILEBERT_FP16,
    tflite_models.MOBILEBERT_FP32,
    tflite_models.MOBILEBERT_INT8,
    tflite_models.MOBILENET_V1,
    tflite_models.MOBILENET_V2,
    tflite_models.MOBILENET_V2_INT8,
    tflite_models.MOBILENET_V3SMALL,
    tflite_models.MOBILESSD_FP32,
    tflite_models.PERSON_DETECT_INT8,
    tflite_models.POSENET_FP32,
    torch_models.EFFICIENTNET_V2_S_FP32_TORCH,
]

# Large models that require more computational resources.
LARGE = [
    tf_models.BERT_FOR_MASKED_LM_FP32_SEQLEN512,
    tf_models.BERT_LARGE_TF_FP32_SEQLEN384,
    tf_models.EFFICIENTNET_V2_S_FP32,
    tf_models.MINILM_L12_H384_UNCASED_INT32_SEQLEN128,
    tf_models.RESNET50_TF_FP32,
    torch_models.CLIP_TEXT_SEQLEN64_FP32_TORCH,
    torch_models.UNET_2D_FP32_TORCH,
    torch_models.VAE_FP32_TORCH,
    torch_models.EFFICIENTNET_B7_FP32_TORCH,
]

# BERT-Large in various batch sizes.
BERT_LARGE_TORCH_BATCHES = [
    torch_models.BERT_LARGE_1x384_FP32_TORCH,
    torch_models.BERT_LARGE_8x384_FP32_TORCH,
    torch_models.BERT_LARGE_16x384_FP32_TORCH,
    torch_models.BERT_LARGE_32x384_FP32_TORCH,
    torch_models.BERT_LARGE_64x384_FP32_TORCH,
    torch_models.BERT_LARGE_128x384_FP32_TORCH,
    # Disabled due to disk space.
    #torch_models.BERT_LARGE_256x384_FP32_TORCH,
    #torch_models.BERT_LARGE_512x384_FP32_TORCH,
    #torch_models.BERT_LARGE_1024x384_FP32_TORCH,
]

# ResNet50 in various batch sizes.
RESNET50_TORCH_BATCHES = [
    torch_models.RESNET50_1x3x224x224_FP32_TORCH,
    torch_models.RESNET50_8x3x224x224_FP32_TORCH,
    torch_models.RESNET50_16x3x224x224_FP32_TORCH,
    torch_models.RESNET50_32x3x224x224_FP32_TORCH,
    torch_models.RESNET50_64x3x224x224_FP32_TORCH,
    torch_models.RESNET50_128x3x224x224_FP32_TORCH,
    # Disabled due to disk space.
    #torch_models.RESNET50_256x3x224x224_FP32_TORCH,
    #torch_models.RESNET50_512x3x224x224_FP32_TORCH,
    #torch_models.RESNET50_1024x3x224x224_FP32_TORCH,
]

CUDA_MODELS = LARGE + BERT_LARGE_TORCH_BATCHES + RESNET50_TORCH_BATCHES + [
    torch_models.EFFICIENTNET_V2_S_FP32_TORCH
]

x86_64_MODELS = SMALL + LARGE

ALL = SMALL + LARGE
