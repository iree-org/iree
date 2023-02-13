# Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception


################################################################################
#                                                                              #
# Benchmark models from Tensorflow                                             #
#                                                                              #
# Each module specification should be a list containing alternating keys and   #
# values. The fields are: NAME, TAGS, SOURCE, ENTRY_FUNCTION, and              #
# FUNCTION_INPUTS. See the iree_benchmark_suite definition for details         #
# about these fields. Note that these must be quoted when used as arguments.   #
#                                                                              #
################################################################################

set(MINILM_L12_H384_UNCASED_INT32_SEQLEN128_MODULE
  NAME
    "MiniLML12H384Uncased"
  TAGS
    "int32,seqlen128"
  SOURCE
    # Converted from https://huggingface.co/microsoft/MiniLM-L12-H384-uncased/commit/44acabbec0ef496f6dbc93adadea57f376b7c0ec
    "https://storage.googleapis.com/iree-model-artifacts/minilm-l12-h384-uncased-seqlen128-tf-model.tar.gz"
  ENTRY_FUNCTION
    "predict"
  IMPORT_FLAGS
    "--tf-savedmodel-exported-names=predict"
  FUNCTION_INPUTS
    "1x128xi32,1x128xi32,1x128xi32"
)

set(RESNET50_TF_FP32_MODULE
  NAME
    "Resnet50Tf"
  TAGS
    "fp32"
  SOURCE
    # Derived from https://github.com/keras-team/keras/blob/v2.10.0/keras/applications/resnet.py.
    "https://storage.googleapis.com/iree-model-artifacts/resnet50-tf-model.tar.gz"
  ENTRY_FUNCTION
    "forward"
  IMPORT_FLAGS
    "--tf-savedmodel-exported-names=forward"
  FUNCTION_INPUTS
    "1x224x224x3xf32"
)

set(BERT_FOR_MASKED_LM_FP32_SEQLEN512_MODULE
  NAME
    "BertForMaskedLM"
  TAGS
    "fp32,seqlen512"
  SOURCE
    # Derived from https://huggingface.co/transformers/v3.0.2/model_doc/bert.html#tfbertformaskedlm.
    "https://storage.googleapis.com/iree-model-artifacts/bert-for-masked-lm-seq512-tf-model.tar.gz"
  ENTRY_FUNCTION
    "forward"
  IMPORT_FLAGS
    "--tf-savedmodel-exported-names=forward"
  FUNCTION_INPUTS
    "1x512xi32,1x512xi32"
)

# This is the model used in the MLPerf Inference Suite.
set(BERT_LARGE_TF_FP32_SEQLEN384_MODULE
  NAME
     "BertLargeTf"
  TAGS
     "fp32,seqlen384"
  SOURCE
    # Derived from https://github.com/mlcommons/inference/tree/master/language/bert
    # Instructions on how to regenerate the model: https://gist.github.com/mariecwhite/e61ccebd979d98d097946ac7725bcc29
    "https://storage.googleapis.com/iree-model-artifacts/bert-large-seq384-tf-model.tar.gz"
  ENTRY_FUNCTION
    "serving_default"
  IMPORT_FLAGS
    "--tf-import-type=savedmodel_v1"
    "--tf-savedmodel-exported-names=serving_default"
  FUNCTION_INPUTS
    "1x384xi32,1x384xi32,1x384xi32"
)

set(EFFICIENTNET_V2_S_TF_FP32_MODULE
  NAME
    "EfficientNetV2STF"
  TAGS
    "fp32"
  SOURCE
    # Derived from https://github.com/keras-team/keras/blob/v2.10.0/keras/applications/efficientnet_v2.py.
    "https://storage.googleapis.com/iree-model-artifacts/efficientnet-v2-s-tf-model.tar.gz"
  ENTRY_FUNCTION
    "forward"
  IMPORT_FLAGS
    "--tf-savedmodel-exported-names=forward"
  FUNCTION_INPUTS
    "1x384x384x3xf32"
)

################################################################################
# Add benchmarks for all platforms.                                            #
################################################################################
include(linux-x86_64.cmake)
include(linux-cuda.cmake)
