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

# This is the model used in the MLPerf Inference Suite.
BERT_LARGE_TF_FP32_SEQLEN384 = common_definitions.Model(
    id=unique_ids.MODEL_BERT_LARGE_TF_FP32_SEQLEN384,
    name="BertLargeTF",
    tags=["fp32", "seqlen384", "tensorflow", "bert-variant", "batch-1"],
    source_type=common_definitions.ModelSourceType.EXPORTED_TF_V1,
    # Derived from https://github.com/mlcommons/inference/tree/master/language/bert
    # Instructions on how to regenerate the model: https://gist.github.com/mariecwhite/e61ccebd979d98d097946ac7725bcc29
    source_url=
    "https://storage.googleapis.com/iree-model-artifacts/bert-large-seq384-tf-model.tar.gz",
    entry_function="serving_default",
    input_types=["1x384xi32", "1x384xi32", "1x384xi32"])

# Derived from https://huggingface.co/docs/transformers/model_doc/bert#transformers.TFBertModel.
BERT_LARGE_1X384_FP32_TF = common_definitions.Model(
    id=unique_ids.MODEL_BERT_LARGE_1X384_FP32_TF,
    name="BertLargeTFBatch1",
    tags=["fp32", "seqlen384", "tensorflow", "bert-variant", "batch-1"],
    source_type=common_definitions.ModelSourceType.EXPORTED_TF_V2,
    source_url=
    "https://storage.googleapis.com/iree-model-artifacts/tensorflow/tf_models_2.12.0_1680491395/BERT_LARGE/batch_1/tf-model.tar.gz",
    entry_function="forward",
    input_types=["1x384xi32", "1x384xi32"])

BERT_LARGE_16X384_FP32_TF = common_definitions.Model(
    id=unique_ids.MODEL_BERT_LARGE_16X384_FP32_TF,
    name="BertLargeTFBatch16",
    tags=["fp32", "seqlen384", "tensorflow", "bert-variant", "batch-16"],
    source_type=common_definitions.ModelSourceType.EXPORTED_TF_V2,
    source_url=
    "https://storage.googleapis.com/iree-model-artifacts/tensorflow/tf_models_2.12.0_1680491395/BERT_LARGE/batch_16/tf-model.tar.gz",
    entry_function="forward",
    input_types=["16x384xi32", "16x384xi32"])

BERT_LARGE_24X384_FP32_TF = common_definitions.Model(
    id=unique_ids.MODEL_BERT_LARGE_24X384_FP32_TF,
    name="BertLargeTFBatch24",
    tags=["fp32", "seqlen384", "tensorflow", "bert-variant", "batch-24"],
    source_type=common_definitions.ModelSourceType.EXPORTED_TF_V2,
    source_url=
    "https://storage.googleapis.com/iree-model-artifacts/tensorflow/tf_models_2.12.0_1680491395/BERT_LARGE/batch_24/tf-model.tar.gz",
    entry_function="forward",
    input_types=["24x384xi32", "24x384xi32"])

BERT_LARGE_32X384_FP32_TF = common_definitions.Model(
    id=unique_ids.MODEL_BERT_LARGE_32X384_FP32_TF,
    name="BertLargeTFBatch32",
    tags=["fp32", "seqlen384", "tensorflow", "bert-variant", "batch-32"],
    source_type=common_definitions.ModelSourceType.EXPORTED_TF_V2,
    source_url=
    "https://storage.googleapis.com/iree-model-artifacts/tensorflow/tf_models_2.12.0_1680491395/BERT_LARGE/batch_32/tf-model.tar.gz",
    entry_function="forward",
    input_types=["32x384xi32", "32x384xi32"])

BERT_LARGE_48X384_FP32_TF = common_definitions.Model(
    id=unique_ids.MODEL_BERT_LARGE_48X384_FP32_TF,
    name="BertLargeTFBatch48",
    tags=["fp32", "seqlen384", "tensorflow", "bert-variant", "batch-48"],
    source_type=common_definitions.ModelSourceType.EXPORTED_TF_V2,
    source_url=
    "https://storage.googleapis.com/iree-model-artifacts/tensorflow/tf_models_2.12.0_1680491395/BERT_LARGE/batch_48/tf-model.tar.gz",
    entry_function="forward",
    input_types=["48x384xi32", "48x384xi32"])

BERT_LARGE_64X384_FP32_TF = common_definitions.Model(
    id=unique_ids.MODEL_BERT_LARGE_64X384_FP32_TF,
    name="BertLargeTFBatch64",
    tags=["fp32", "seqlen384", "tensorflow", "bert-variant", "batch-64"],
    source_type=common_definitions.ModelSourceType.EXPORTED_TF_V2,
    source_url=
    "https://storage.googleapis.com/iree-model-artifacts/tensorflow/tf_models_2.12.0_1680491395/BERT_LARGE/batch_64/tf-model.tar.gz",
    entry_function="forward",
    input_types=["64x384xi32", "64x384xi32"])

BERT_LARGE_512X384_FP32_TF = common_definitions.Model(
    id=unique_ids.MODEL_BERT_LARGE_512X384_FP32_TF,
    name="BertLargeTFBatch512",
    tags=["fp32", "seqlen384", "tensorflow", "bert-variant", "batch-512"],
    source_type=common_definitions.ModelSourceType.EXPORTED_TF_V2,
    source_url=
    "https://storage.googleapis.com/iree-model-artifacts/tensorflow/tf_models_2.12.0_1680491395/BERT_LARGE/batch_512/tf-model.tar.gz",
    entry_function="forward",
    input_types=["512x384xi32", "512x384xi32"])

BERT_LARGE_1024X384_FP32_TF = common_definitions.Model(
    id=unique_ids.MODEL_BERT_LARGE_1024X384_FP32_TF,
    name="BertLargeTFBatch1024",
    tags=["fp32", "seqlen384", "tensorflow", "bert-variant", "batch-1024"],
    source_type=common_definitions.ModelSourceType.EXPORTED_TF_V2,
    source_url=
    "https://storage.googleapis.com/iree-model-artifacts/tensorflow/tf_models_2.12.0_1680491395/BERT_LARGE/batch_1024/tf-model.tar.gz",
    entry_function="forward",
    input_types=["1024x384xi32", "1024x384xi32"])

BERT_LARGE_1280X384_FP32_TF = common_definitions.Model(
    id=unique_ids.MODEL_BERT_LARGE_1280X384_FP32_TF,
    name="BertLargeTFBatch1280",
    tags=["fp32", "seqlen384", "tensorflow", "bert-variant", "batch-1280"],
    source_type=common_definitions.ModelSourceType.EXPORTED_TF_V2,
    source_url=
    "https://storage.googleapis.com/iree-model-artifacts/tensorflow/tf_models_2.12.0_1680491395/BERT_LARGE/batch_1280/tf-model.tar.gz",
    entry_function="forward",
    input_types=["1280x384xi32", "1280x384xi32"])

# Converted from https://www.tensorflow.org/api_docs/python/tf/keras/applications/resnet50/ResNet50
RESNET50_1X3X224X224_FP32_TF = common_definitions.Model(
    id=unique_ids.MODEL_RESNET50_1X3X224X224_FP32_TF,
    name="Resnet50TFBatch1",
    tags=["fp32", "cnn", "batch-1"],
    source_type=common_definitions.ModelSourceType.EXPORTED_TF_V2,
    source_url=
    "https://storage.googleapis.com/iree-model-artifacts/tensorflow/tf_models_2.12.0_1680486104/RESNET50/batch_1/tf-model.tar.gz",
    entry_function="forward",
    input_types=["1x224x224x3xf32"])

RESNET50_8X3X224X224_FP32_TF = common_definitions.Model(
    id=unique_ids.MODEL_RESNET50_8X3X224X224_FP32_TF,
    name="Resnet50TFBatch8",
    tags=["fp32", "cnn", "batch-8"],
    source_type=common_definitions.ModelSourceType.EXPORTED_TF_V2,
    source_url=
    "https://storage.googleapis.com/iree-model-artifacts/tensorflow/tf_models_2.12.0_1680486104/RESNET50/batch_8/tf-model.tar.gz",
    entry_function="forward",
    input_types=["8x224x224x3xf32"])

RESNET50_64X3X224X224_FP32_TF = common_definitions.Model(
    id=unique_ids.MODEL_RESNET50_64X3X224X224_FP32_TF,
    name="Resnet50TFBatch64",
    tags=["fp32", "cnn", "batch-64"],
    source_type=common_definitions.ModelSourceType.EXPORTED_TF_V2,
    source_url=
    "https://storage.googleapis.com/iree-model-artifacts/tensorflow/tf_models_2.12.0_1680486104/RESNET50/batch_64/tf-model.tar.gz",
    entry_function="forward",
    input_types=["64x224x224x3xf32"])

RESNET50_128X3X224X224_FP32_TF = common_definitions.Model(
    id=unique_ids.MODEL_RESNET50_128X3X224X224_FP32_TF,
    name="Resnet50TFBatch128",
    tags=["fp32", "cnn", "batch-128"],
    source_type=common_definitions.ModelSourceType.EXPORTED_TF_V2,
    source_url=
    "https://storage.googleapis.com/iree-model-artifacts/tensorflow/tf_models_2.12.0_1680486104/RESNET50/batch_128/tf-model.tar.gz",
    entry_function="forward",
    input_types=["128x224x224x3xf32"])

RESNET50_256X3X224X224_FP32_TF = common_definitions.Model(
    id=unique_ids.MODEL_RESNET50_256X3X224X224_FP32_TF,
    name="Resnet50TFBatch256",
    tags=["fp32", "cnn", "batch-256"],
    source_type=common_definitions.ModelSourceType.EXPORTED_TF_V2,
    source_url=
    "https://storage.googleapis.com/iree-model-artifacts/tensorflow/tf_models_2.12.0_1680486104/RESNET50/batch_256/tf-model.tar.gz",
    entry_function="forward",
    input_types=["256x224x224x3xf32"])

RESNET50_2048X3X224X224_FP32_TF = common_definitions.Model(
    id=unique_ids.MODEL_RESNET50_2048X3X224X224_FP32_TF,
    name="Resnet50TFBatch2048",
    tags=["fp32", "cnn", "batch-2048"],
    source_type=common_definitions.ModelSourceType.EXPORTED_TF_V2,
    source_url=
    "https://storage.googleapis.com/iree-model-artifacts/tensorflow/tf_models_2.12.0_1680486104/RESNET50/batch_2048/tf-model.tar.gz",
    entry_function="forward",
    input_types=["2048x224x224x3xf32"])

# Derived from https://huggingface.co/transformers/v3.0.2/model_doc/t5.html#tft5model.
T5_LARGE_1x512_FP32_TF = common_definitions.Model(
    id=unique_ids.MODEL_T5_LARGE_1x512_FP32_TF,
    name="T5LargeTFBatch1",
    tags=["fp32", "seqlen512", "tensorflow", "batch-1"],
    source_type=common_definitions.ModelSourceType.EXPORTED_TF_V2,
    source_url=
    "https://storage.googleapis.com/iree-model-artifacts/tensorflow/tf_models_2.12.0_1681193933/T5_LARGE/batch_1/tf-model.tar.gz",
    entry_function="forward",
    input_types=["1x512xi32", "1x512xi32"])

T5_LARGE_16x512_FP32_TF = common_definitions.Model(
    id=unique_ids.MODEL_T5_LARGE_16x512_FP32_TF,
    name="T5LargeTFBatch16",
    tags=["fp32", "seqlen512", "tensorflow", "batch-16"],
    source_type=common_definitions.ModelSourceType.EXPORTED_TF_V2,
    source_url=
    "https://storage.googleapis.com/iree-model-artifacts/tensorflow/tf_models_2.12.0_1681193933/T5_LARGE/batch_16/tf-model.tar.gz",
    entry_function="forward",
    input_types=["16x512xi32", "16x512xi32"])

T5_LARGE_24x512_FP32_TF = common_definitions.Model(
    id=unique_ids.MODEL_T5_LARGE_24x512_FP32_TF,
    name="T5LargeTFBatch24",
    tags=["fp32", "seqlen512", "tensorflow", "batch-24"],
    source_type=common_definitions.ModelSourceType.EXPORTED_TF_V2,
    source_url=
    "https://storage.googleapis.com/iree-model-artifacts/tensorflow/tf_models_2.12.0_1681193933/T5_LARGE/batch_24/tf-model.tar.gz",
    entry_function="forward",
    input_types=["24x512xi32", "24x512xi32"])

T5_LARGE_32x512_FP32_TF = common_definitions.Model(
    id=unique_ids.MODEL_T5_LARGE_32x512_FP32_TF,
    name="T5LargeTFBatch32",
    tags=["fp32", "seqlen512", "tensorflow", "batch-32"],
    source_type=common_definitions.ModelSourceType.EXPORTED_TF_V2,
    source_url=
    "https://storage.googleapis.com/iree-model-artifacts/tensorflow/tf_models_2.12.0_1681193933/T5_LARGE/batch_32/tf-model.tar.gz",
    entry_function="forward",
    input_types=["32x512xi32", "32x512xi32"])

T5_LARGE_48x512_FP32_TF = common_definitions.Model(
    id=unique_ids.MODEL_T5_LARGE_48x512_FP32_TF,
    name="T5LargeTFBatch48",
    tags=["fp32", "seqlen512", "tensorflow", "batch-48"],
    source_type=common_definitions.ModelSourceType.EXPORTED_TF_V2,
    source_url=
    "https://storage.googleapis.com/iree-model-artifacts/tensorflow/tf_models_2.12.0_1681193933/T5_LARGE/batch_48/tf-model.tar.gz",
    entry_function="forward",
    input_types=["48x512xi32", "48x512xi32"])

# Disabled due to https://github.com/openxla/iree/issues/13189.
#T5_LARGE_64x512_FP32_TF = common_definitions.Model(
#    id=unique_ids.MODEL_T5_LARGE_64x512_FP32_TF,
#    name="T5LargeTFBatch64",
#    tags=["fp32", "seqlen512", "tensorflow", "batch-64"],
#    source_type=common_definitions.ModelSourceType.EXPORTED_TF_V2,
#    source_url=
#    "https://storage.googleapis.com/iree-model-artifacts/tensorflow/tf_models_2.12.0_1681193933/T5_LARGE/batch_64/tf-model.tar.gz",
#    entry_function="forward",
#    input_types=["64x512xi32", "64x512xi32"])

# Disabled due to https://github.com/openxla/iree/issues/13189.
#T5_LARGE_512x512_FP32_TF = common_definitions.Model(
#    id=unique_ids.MODEL_T5_LARGE_512x512_FP32_TF,
#    name="T5LargeTFBatch512",
#    tags=["fp32", "seqlen512", "tensorflow", "batch-512"],
#    source_type=common_definitions.ModelSourceType.EXPORTED_TF_V2,
#    source_url=
#    "https://storage.googleapis.com/iree-model-artifacts/tensorflow/tf_models_2.12.0_1681193933/T5_LARGE/batch_512/tf-model.tar.gz",
#    entry_function="forward",
#    input_types=["512x512xi32", "512x512xi32"])
