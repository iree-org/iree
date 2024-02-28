## Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Defines Tensorflow models."""

import string

from e2e_test_framework import unique_ids
from e2e_test_framework.definitions import common_definitions
import e2e_test_framework.models.utils as model_utils

TF_MODELS_MANUAL_ROOT_DIR = (
    "https://storage.googleapis.com/iree-model-artifacts/tensorflow/manual"
)

MINILM_L12_H384_UNCASED_INT32_SEQLEN128 = common_definitions.Model(
    id=unique_ids.MODEL_MINILM_L12_H384_UNCASED_INT32_SEQLEN128,
    name="MiniLML12H384Uncased",
    tags=["int32", "seqlen128"],
    source_type=common_definitions.ModelSourceType.EXPORTED_STABLEHLO_MLIR,
    # Converted from https://huggingface.co/microsoft/MiniLM-L12-H384-uncased/commit/44acabbec0ef496f6dbc93adadea57f376b7c0ec
    source_url=f"{TF_MODELS_MANUAL_ROOT_DIR}/MiniLML12H384Uncased_2023-05-07.timestamp_1683504734j.mlirbc",
    entry_function="predict",
    input_types=["1x128xi32", "1x128xi32", "1x128xi32"],
)

BERT_FOR_MASKED_LM_FP32_SEQLEN512 = common_definitions.Model(
    id=unique_ids.MODEL_BERT_FOR_MASKED_LM_FP32_SEQLEN512_TF,
    name="BertForMaskedLMTF",
    tags=["fp32", "seqlen512", "tensorflow"],
    source_type=common_definitions.ModelSourceType.EXPORTED_STABLEHLO_MLIR,
    # Converted from https://huggingface.co/transformers/v3.0.2/model_doc/bert.html#tfbertformaskedlm
    source_url=f"{TF_MODELS_MANUAL_ROOT_DIR}/BertForMaskedLMTF_2023-05-07.timestamp_1683504734j.mlirbc",
    entry_function="forward",
    input_types=["1x512xi32", "1x512xi32"],
)

EFFICIENTNET_V2_S_FP32 = common_definitions.Model(
    id=unique_ids.MODEL_EFFICIENTNET_V2_S_FP32_TF,
    name="EfficientNetV2STF",
    tags=["fp32", "cnn", "tensorflow"],
    source_type=common_definitions.ModelSourceType.EXPORTED_STABLEHLO_MLIR,
    # Converted from https://github.com/keras-team/keras/blob/v2.10.0/keras/applications/efficientnet_v2.py
    source_url=f"{TF_MODELS_MANUAL_ROOT_DIR}/EfficientNetV2STF_1af8c88f4e64e388a0c87bbeddcfb888084059df30cd631340d51794a0796e0f.mlirbc",
    entry_function="forward",
    input_types=["1x384x384x3xf32"],
)

# This is the model used in the MLPerf Inference Suite.
BERT_LARGE_TF_FP32_SEQLEN384 = common_definitions.Model(
    id=unique_ids.MODEL_BERT_LARGE_TF_FP32_SEQLEN384,
    name="BertLargeTF",
    tags=["fp32", "seqlen384", "tensorflow", "bert-variant", "batch-1"],
    source_type=common_definitions.ModelSourceType.EXPORTED_STABLEHLO_MLIR,
    # Derived from https://github.com/mlcommons/inference/tree/master/language/bert
    # Instructions on how to regenerate the model: https://gist.github.com/mariecwhite/e61ccebd979d98d097946ac7725bcc29
    source_url=f"{TF_MODELS_MANUAL_ROOT_DIR}/BertLargeTF_2023-05-07.timestamp_1683504734j.mlirbc",
    entry_function="serving_default",
    input_types=["1x384xi32", "1x384xi32", "1x384xi32"],
)

GPT2_117M_1x4_FP32_TF = common_definitions.Model(
    id=unique_ids.MODEL_GPT2_117M_1x4_FP32_TF,
    name="GPT2_117M_TF_1X4XI32",
    tags=["fp32", "tensorflow", "gpt2", "batch-1"],
    source_type=common_definitions.ModelSourceType.EXPORTED_STABLEHLO_MLIR,
    source_url="https://storage.googleapis.com/iree-shared-files/tf_gpt2/static_input_seqlen5/stablehlo.mlir",
    entry_function="forward",
    input_types=["1x4xi32", "1x4xi32"],
)

GPT2_117M_1x1_FP32_TF = common_definitions.Model(
    id=unique_ids.MODEL_GPT2_117M_1x1_FP32_TF,
    name="GPT2_117M_TF_1X1XI32",
    tags=["fp32", "tensorflow", "gpt2", "batch-1"],
    source_type=common_definitions.ModelSourceType.EXPORTED_STABLEHLO_MLIR,
    source_url="https://storage.googleapis.com/iree-shared-files/tf_gpt2/static_input_seqlen1/stablehlo.mlir",
    entry_function="forward",
    input_types=["1x1xi32", "12x2x1x12x4x64xf32"],
)

TF_MODELS_ROOT_DIR = "https://storage.googleapis.com/iree-model-artifacts/tensorflow/tf_models_2.15.0.dev20230817_1692333975j"

ID_FORMAT = string.Template("${model_id}-batch-${batch_size}")
NAME_FORMAT = string.Template("${name}Batch${batch_size}")
SOURCE_URL_FORMAT = string.Template(
    TF_MODELS_ROOT_DIR + "/${directory}_BATCH${batch_size}/stablehlo.mlirbc"
)

# Derived from https://huggingface.co/docs/transformers/model_doc/bert#transformers.TFBertModel.
BERT_LARGE_384_FP32_TF_BATCHES = model_utils.generate_batch_models(
    id_template=model_utils.partial_template_substitute(
        ID_FORMAT, model_id=unique_ids.MODEL_BERT_LARGE_384_FP32_TF
    ),
    name_template=model_utils.partial_template_substitute(
        NAME_FORMAT, name="BertLargeTF"
    ),
    tags=["fp32", "seqlen384", "tensorflow", "bert-variant"],
    source_type=common_definitions.ModelSourceType.EXPORTED_STABLEHLO_MLIR,
    source_url_template=model_utils.partial_template_substitute(
        SOURCE_URL_FORMAT, directory="BERT_LARGE_FP32_TF_384XI32"
    ),
    entry_function="forward",
    input_type_templates=[
        string.Template("${batch_size}x384xi32"),
        string.Template("${batch_size}x384xi32"),
    ],
    batch_sizes=[1, 16, 24, 32, 48, 64, 512, 1024, 1280],
)

# Converted from https://www.tensorflow.org/api_docs/python/tf/keras/applications/resnet50/ResNet50
RESNET50_3X224X224_FP32_TF_BATCHES = model_utils.generate_batch_models(
    id_template=model_utils.partial_template_substitute(
        ID_FORMAT, model_id=unique_ids.MODEL_RESNET50_3X224X224_FP32_TF
    ),
    name_template=model_utils.partial_template_substitute(
        NAME_FORMAT, name="Resnet50TF"
    ),
    tags=["fp32", "cnn"],
    source_type=common_definitions.ModelSourceType.EXPORTED_STABLEHLO_MLIR,
    source_url_template=model_utils.partial_template_substitute(
        SOURCE_URL_FORMAT, directory="RESNET50_FP32_TF_224X224X3XF32"
    ),
    entry_function="forward",
    input_type_templates=[string.Template("${batch_size}x3x224x224xf32")],
    batch_sizes=[1, 8, 64, 128, 256, 2048],
)

# Derived from https://huggingface.co/transformers/v3.0.2/model_doc/t5.html#tft5model.
T5_LARGE_512_FP32_TF_BATCHES = model_utils.generate_batch_models(
    id_template=model_utils.partial_template_substitute(
        ID_FORMAT, model_id=unique_ids.MODEL_T5_LARGE_512_FP32_TF
    ),
    name_template=model_utils.partial_template_substitute(
        NAME_FORMAT, name="T5LargeTF"
    ),
    tags=["fp32", "seqlen512", "tensorflow"],
    source_type=common_definitions.ModelSourceType.EXPORTED_STABLEHLO_MLIR,
    source_url_template=model_utils.partial_template_substitute(
        SOURCE_URL_FORMAT, directory="T5_LARGE_FP32_TF_512XI32"
    ),
    entry_function="forward",
    input_type_templates=[
        string.Template("${batch_size}x512xi32"),
        string.Template("${batch_size}x512xi32"),
    ],
    batch_sizes=[1, 16, 24, 32, 48, 64, 512],
)
