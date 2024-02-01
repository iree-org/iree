## Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Defines JAX models."""

import string

from e2e_test_framework import unique_ids
from e2e_test_framework.definitions import common_definitions
import e2e_test_framework.models.utils as model_utils

GCS_ARTIFACT_ROOT_DIR = "https://storage.googleapis.com/iree-model-artifacts/jax/jax_models_0.4.23_1705868085"

ID_FORMAT = string.Template("${model_id}-batch${batch_size}")
NAME_FORMAT = string.Template("${name}_BATCH${batch_size}")
SOURCE_URL_FORMAT = string.Template(
    GCS_ARTIFACT_ROOT_DIR + "/${directory}_BATCH${batch_size}/stablehlo.mlirbc"
)

# Derived from https://huggingface.co/docs/transformers/model_doc/resnet#transformers.FlaxResNetModel.
RESNET50_TAGS = ["fp32", "cnn", "resnet"]

RESNET50_FP32_JAX_3X224X224XF32_BATCHES = model_utils.generate_batch_models(
    id_template=model_utils.partial_template_substitute(
        ID_FORMAT, model_id=unique_ids.MODEL_RESNET50_FP32_JAX_3X224X224XF32
    ),
    name_template=model_utils.partial_template_substitute(
        NAME_FORMAT, name="RESNET50_FP32_JAX_3X224X224XF32"
    ),
    tags=RESNET50_TAGS,
    source_type=common_definitions.ModelSourceType.EXPORTED_STABLEHLO_MLIR,
    source_url_template=model_utils.partial_template_substitute(
        SOURCE_URL_FORMAT, directory="RESNET50_FP32_JAX_3X224X224XF32"
    ),
    entry_function="main",
    input_type_templates=[string.Template("${batch_size}x3x224x224xf32")],
    batch_sizes=[1, 8, 64, 128, 256, 2048],
)

# Derived from https://huggingface.co/docs/transformers/model_doc/bert#transformers.FlaxBertModel.
BERT_LARGE_TAGS = ["fp32", "seqlen384", "jax", "bert-variant"]

BERT_LARGE_FP32_JAX_384XI32_BATCHES = model_utils.generate_batch_models(
    id_template=model_utils.partial_template_substitute(
        ID_FORMAT, model_id=unique_ids.MODEL_BERT_LARGE_FP32_JAX_384XI32
    ),
    name_template=model_utils.partial_template_substitute(
        NAME_FORMAT, name="BERT_LARGE_JAX_384XI32"
    ),
    tags=BERT_LARGE_TAGS,
    source_type=common_definitions.ModelSourceType.EXPORTED_STABLEHLO_MLIR,
    source_url_template=model_utils.partial_template_substitute(
        SOURCE_URL_FORMAT, directory="BERT_LARGE_FP32_JAX_384XI32"
    ),
    entry_function="main",
    input_type_templates=[
        string.Template("${batch_size}x384xi32"),
        string.Template("${batch_size}x384xi32"),
    ],
    batch_sizes=[1, 16, 24, 32, 48, 64, 512, 1024, 1280],
)

# Derived from https://huggingface.co/docs/transformers/model_doc/t5#transformers.FlaxT5Model
T5_TAGS = ["fp32", "transformer-encoder", "transformer-decoder", "t5"]

T5_LARGE_FP32_JAX_512XI32_BATCHES = model_utils.generate_batch_models(
    id_template=model_utils.partial_template_substitute(
        ID_FORMAT, model_id=unique_ids.MODEL_T5_LARGE_FP32_JAX_512XI32
    ),
    name_template=model_utils.partial_template_substitute(
        NAME_FORMAT, name="T5_LARGE_FP32_JAX_512XI32"
    ),
    tags=T5_TAGS,
    source_type=common_definitions.ModelSourceType.EXPORTED_STABLEHLO_MLIR,
    source_url_template=model_utils.partial_template_substitute(
        SOURCE_URL_FORMAT, directory="T5_LARGE_FP32_JAX_512XI32"
    ),
    entry_function="main",
    input_type_templates=[
        string.Template("${batch_size}x512xi32"),
        string.Template("${batch_size}x512xi32"),
        string.Template("${batch_size}x512xi32"),
        string.Template("${batch_size}x512xi32"),
    ],
    batch_sizes=[1, 16, 24, 32, 48, 64, 512],
)

T5_4CG_TAGS = ["fp32", "transformer-encoder", "transformer-decoder", "t5_4cg"]
T5_4CG_LARGE_FP32_JAX_512XI32_BATCHES = model_utils.generate_batch_models(
    id_template=model_utils.partial_template_substitute(
        ID_FORMAT, model_id=unique_ids.MODEL_T5_4CG_LARGE_FP32_JAX_512XI32
    ),
    name_template=model_utils.partial_template_substitute(
        NAME_FORMAT, name="T5_4CG_LARGE_FP32_JAX_512XI32"
    ),
    tags=T5_4CG_TAGS,
    source_type=common_definitions.ModelSourceType.EXPORTED_STABLEHLO_MLIR,
    source_url_template=model_utils.partial_template_substitute(
        SOURCE_URL_FORMAT, directory="T5_4CG_LARGE_FP32_JAX_512XI32"
    ),
    entry_function="main",
    input_type_templates=[
        string.Template("${batch_size}x512xi32"),
        string.Template("${batch_size}x512xi32"),
        string.Template("${batch_size}x512xi32"),
        string.Template("${batch_size}x512xi32"),
    ],
    batch_sizes=[1, 16, 24, 32, 48],
)
