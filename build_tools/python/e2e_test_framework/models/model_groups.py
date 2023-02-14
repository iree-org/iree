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
    tflite_models.MOBILENET_V3SMALL,
    tflite_models.MOBILESSD_FP32,
    tflite_models.PERSON_DETECT_INT8,
    tflite_models.POSENET_FP32,
]

# Large models that require more computational resources.
LARGE = [
    tf_models.BERT_FOR_MASKED_LM_FP32_SEQLEN512,
    tf_models.BERT_LARGE_TF_FP32_SEQLEN384,
    tf_models.EFFICIENTNET_V2_S_FP32,
    tf_models.MINILM_L12_H384_UNCASED_INT32_SEQLEN128,
    tf_models.RESNET50_TF_FP32,
    torch_models.MODEL_CLIP_TEXT_SEQLEN64_FP32_TORCH,
    # Disabled due to https://github.com/openxla/iree/issues/11447.
    #torch_models.MODEL_UNET_2D_FP32_TORCH,
]

ALL = SMALL + LARGE
