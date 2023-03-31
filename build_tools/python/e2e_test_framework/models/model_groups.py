## Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Defines the groups of models."""

from e2e_test_framework.models import tf_models, tflite_models, torch_models

# yapf: disable
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
    tf_models.EFFICIENTNET_V2_S_FP32,
    tf_models.MINILM_L12_H384_UNCASED_INT32_SEQLEN128,
    torch_models.EFFICIENTNET_V2_S_FP32_TORCH
]
# yapf: enable

# Large models that require more computational resources.
LARGE = [
    tf_models.BERT_FOR_MASKED_LM_FP32_SEQLEN512,
    tf_models.BERT_LARGE_TF_FP32_SEQLEN384,
    tf_models.RESNET50_TF_FP32,
    torch_models.MODEL_CLIP_TEXT_SEQLEN64_FP32_TORCH,
    torch_models.MODEL_UNET_2D_FP32_TORCH,
    torch_models.EFFICIENTNET_B7_FP32_TORCH,
]

# BERT-Large in various batch sizes.
BERT_LARGE_TORCH_BATCHES = [
    torch_models.BERT_LARGE_1X384_FP32_TORCH,
    torch_models.BERT_LARGE_8X384_FP32_TORCH,
    torch_models.BERT_LARGE_16X384_FP32_TORCH,
    torch_models.BERT_LARGE_32X384_FP32_TORCH,
    torch_models.BERT_LARGE_64X384_FP32_TORCH,
    # Disabled due to https://github.com/openxla/iree/issues/12774.
    #torch_models.BERT_LARGE_128X384_FP32_TORCH,
    #torch_models.BERT_LARGE_256X384_FP32_TORCH,
    #torch_models.BERT_LARGE_512X384_FP32_TORCH,
    #torch_models.BERT_LARGE_1024X384_FP32_TORCH,
]

ALL = SMALL + LARGE + BERT_LARGE_TORCH_BATCHES
