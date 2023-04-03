## Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Defines Tensorflow models."""

from e2e_test_framework import unique_ids
from e2e_test_framework.definitions import common_definitions

MINILM_L12_H384_UNCASED_INT32_SEQLEN128 = common_definitions.Model(
    id=unique_ids.MODEL_MINILM_L12_H384_UNCASED_INT32_SEQLEN128,
    name="MiniLML12H384Uncased",
    tags=["int32", "seqlen128"],
    source_type=common_definitions.ModelSourceType.EXPORTED_TF_V2,
    # Converted from https://huggingface.co/microsoft/MiniLM-L12-H384-uncased/commit/44acabbec0ef496f6dbc93adadea57f376b7c0ec
    source_url=
    "https://storage.googleapis.com/iree-model-artifacts/minilm-l12-h384-uncased-seqlen128-tf-model.tar.gz",
    entry_function="predict",
    input_types=["1x128xi32", "1x128xi32", "1x128xi32"])

BERT_FOR_MASKED_LM_FP32_SEQLEN512 = common_definitions.Model(
    id=unique_ids.MODEL_BERT_FOR_MASKED_LM_FP32_SEQLEN512_TF,
    name="BertForMaskedLMTF",
    tags=["fp32", "seqlen512", "tensorflow"],
    source_type=common_definitions.ModelSourceType.EXPORTED_TF_V2,
    # Converted from https://huggingface.co/transformers/v3.0.2/model_doc/bert.html#tfbertformaskedlm
    source_url=
    "https://storage.googleapis.com/iree-model-artifacts/bert-for-masked-lm-seq512-tf-model.tar.gz",
    entry_function="forward",
    input_types=["1x512xi32", "1x512xi32"])

EFFICIENTNET_V2_S_FP32 = common_definitions.Model(
    id=unique_ids.MODEL_EFFICIENTNET_V2_S_FP32_TF,
    name="EfficientNetV2STF",
    tags=["fp32", "cnn", "tensorflow"],
    source_type=common_definitions.ModelSourceType.EXPORTED_TF_V2,
    # Converted from https://github.com/keras-team/keras/blob/v2.10.0/keras/applications/efficientnet_v2.py
    source_url=
    "https://storage.googleapis.com/iree-model-artifacts/efficientnet-v2-s-tf-model.tar.gz",
    entry_function="forward",
    input_types=["1x384x384x3xf32"])

RESNET50_TF_FP32 = common_definitions.Model(
    id=unique_ids.MODEL_RESNET50_TF_FP32,
    name="Resnet50TF",
    tags=["fp32"],
    source_type=common_definitions.ModelSourceType.EXPORTED_TF_V2,
    # Derived from https://github.com/keras-team/keras/blob/v2.10.0/keras/applications/resnet.py.
    source_url=
    "https://storage.googleapis.com/iree-model-artifacts/resnet50-tf-model.tar.gz",
    entry_function="forward",
    input_types=["1x224x224x3xf32"])

# This is the model used in the MLPerf Inference Suite.
BERT_LARGE_TF_FP32_SEQLEN384 = common_definitions.Model(
    id=unique_ids.MODEL_BERT_LARGE_TF_FP32_SEQLEN384,
    name="BertLargeTF",
    tags=["fp32", "seqlen384", "tensorflow"],
    source_type=common_definitions.ModelSourceType.EXPORTED_TF_V1,
    # Derived from https://github.com/mlcommons/inference/tree/master/language/bert
    # Instructions on how to regenerate the model: https://gist.github.com/mariecwhite/e61ccebd979d98d097946ac7725bcc29
    source_url=
    "https://storage.googleapis.com/iree-model-artifacts/bert-large-seq384-tf-model.tar.gz",
    entry_function="serving_default",
    input_types=["1x384xi32", "1x384xi32", "1x384xi32"])

# Model derived from https://huggingface.co/docs/transformers/model_doc/t5#transformers.TFT5Model.
T5_LARGE_FP32_SEQLEN512_TF = common_definitions.Model(
    id=unique_ids.MODEL_T5_LARGE_FP32_SEQLEN512_TF,
    name="T5LargeTF",
    tags=["fp32", "seqlen512", "tensorflow"],
    source_type=common_definitions.ModelSourceType.EXPORTED_TF_V1,
    source_url=
    "https://storage.googleapis.com/iree-model-artifacts/tensorflow/tf_models_2.12.0_1680491395/T5_LARGE/batch_1/tf-model.tar.gz",
    entry_function="serving_default",
    input_types=["1x512xi32", "1x512xi32"])
