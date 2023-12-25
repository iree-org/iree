## Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Defines PyTorch models."""

import string

from e2e_test_framework import unique_ids
from e2e_test_framework.definitions import common_definitions
import e2e_test_framework.models.utils as model_utils

# Implementations of the models listed below can be found in
# https://github.com/iree-org/iree-samples/tree/main/iree-torch/importer.
# We import the PyTorch models offline and make the .mlir available here for benchmarking.
# If the mlir artifacts need to be updated, please run
# https://github.com/iree-org/iree-samples/blob/main/iree-torch/importer/update_torch_models.sh
# Then update the `source_url` below with the new paths.

# `ClipTextModel` encodes text into an embedding.
#
# Used in Stable Diffusion to convert a text prompt into an embedding for input to the `Unet2d` model.
#
# Converted from https://huggingface.co/docs/transformers/model_doc/clip#transformers.CLIPTextModel
MODEL_CLIP_TEXT_SEQLEN64_FP32_TORCH = common_definitions.Model(
    id=unique_ids.MODEL_CLIP_TEXT_SEQLEN64_FP32_TORCH,
    name="ClipTextSeqLen64PT",
    tags=["fp32", "seqlen64"],
    source_type=common_definitions.ModelSourceType.EXPORTED_LINALG_MLIR,
    source_url="https://storage.googleapis.com/iree-model-artifacts/pytorch/torch_models_20230307.103_1678163233/SD_CLIP_TEXT_MODEL_SEQLEN64/linalg.mlir",
    entry_function="forward",
    input_types=["1x77xi64", "1x77xi64"],
)

# `Unet2d` consists of `ResNet` encoder and decoder blocks with cross-attention layers.
#
# Takes 2 inputs:
#     i) a text embedding generated from a language model like `ClipTextModel`
#     ii) an image seed (either random noise or an actual image) in the latent space (compressed representation of an image).
# Outputs an image in the latent space.
#
# Used in Stable Diffusion to gradually subtract noise in the latent space.
# Usually run over multiple steps until the image is sufficiently de-noised.
#
# Once complete, output is upsampled using a Variation Auto Encoder (`VAE`).
#
# Converted from https://huggingface.co/docs/diffusers/api/models#diffusers.UNet2DConditionModel
MODEL_UNET_2D_FP32_TORCH = common_definitions.Model(
    id=unique_ids.MODEL_UNET_2D_FP32_TORCH,
    name="Unet2dPT",
    tags=["fp32"],
    source_type=common_definitions.ModelSourceType.EXPORTED_LINALG_MLIR,
    source_url="https://storage.googleapis.com/iree-model-artifacts/pytorch/torch_models_20230307.103_1678163233/SD_UNET_MODEL/linalg.mlir",
    entry_function="forward",
    input_types=["1x4x64x64xf32", "1x77x768xf32"],
)

# Converted from https://pytorch.org/vision/stable/models/generated/torchvision.models.efficientnet_v2_s.html#torchvision.models.efficientnet_v2_s
EFFICIENTNET_V2_S_FP32_TORCH = common_definitions.Model(
    id=unique_ids.MODEL_EFFICIENTNET_V2_S_FP32_TORCH,
    name="EfficientNetV2SPT",
    tags=["fp32", "cnn", "depthwise-conv"],
    source_type=common_definitions.ModelSourceType.EXPORTED_LINALG_MLIR,
    source_url="https://storage.googleapis.com/iree-model-artifacts/pytorch/torch_models_20230321.784_1679461251/EFFICIENTNET_V2_S/batch_1/linalg.mlir",
    entry_function="forward",
    input_types=["1x3x384x384xf32"],
)

# FP16
EFFICIENTNET_V2_S_FP16_TORCH = common_definitions.Model(
    id=unique_ids.MODEL_EFFICIENTNET_V2_S_FP16_TORCH,
    name="EfficientNetV2Sfp16PT",
    tags=["fp16", "cnn", "depthwise-conv"],
    source_type=common_definitions.ModelSourceType.EXPORTED_LINALG_MLIR,
    source_url="https://storage.googleapis.com/iree-model-artifacts/pytorch/torch_models_20230522.846_1684831160/EFFICIENTNET_V2_S_FP16/batch_1/linalg.mlir",
    entry_function="forward",
    input_types=["1x3x384x384xf16"],
)

# Converted from https://pytorch.org/vision/stable/models/generated/torchvision.models.efficientnet_b7.html#torchvision.models.efficientnet_b7
EFFICIENTNET_B7_FP32_TORCH = common_definitions.Model(
    id=unique_ids.MODEL_EFFICIENTNET_B7_FP32_TORCH,
    name="EfficientNetB7PT",
    tags=["fp32", "cnn", "depthwise-conv"],
    source_type=common_definitions.ModelSourceType.EXPORTED_LINALG_MLIR,
    source_url="https://storage.googleapis.com/iree-model-artifacts/pytorch/torch_models_20230321.784_1679461251/EFFICIENTNET_B7/batch_1/linalg.mlir",
    entry_function="forward",
    input_types=["1x3x600x600xf32"],
)

ID_FORMAT = string.Template("${model_id}-batch-${batch_size}")
NAME_FORMAT = string.Template("${name}Batch${batch_size}")
BERT_LARGE_FP32_URL = string.Template(
    "https://storage.googleapis.com/iree-model-artifacts/pytorch/pt_models_20231010.987_1696982151/BERT_LARGE_FP32_PT_384XI32_BATCH${batch_size}/linalg.mlirbc"
)
BERT_LARGE_FP16_URL = string.Template(
    "https://storage.googleapis.com/iree-model-artifacts/pytorch/pt_models_20230816.932_1692245822/BERT_LARGE_FP16_PT_384XI32_BATCH${batch_size}/linalg.mlirbc"
)

# Converted from https://huggingface.co/docs/transformers/v4.27.2/en/model_doc/bert#transformers.BertModel
BERT_LARGE_384_FP32_TORCH_BATCHES = model_utils.generate_batch_models(
    id_template=model_utils.partial_template_substitute(
        ID_FORMAT, model_id=unique_ids.MODEL_BERT_LARGE_384_FP32_TORCH
    ),
    name_template=model_utils.partial_template_substitute(
        NAME_FORMAT, name="BertLargePT"
    ),
    tags=["fp32", "transformer", "seqlen384"],
    source_type=common_definitions.ModelSourceType.EXPORTED_LINALG_MLIR,
    source_url_template=BERT_LARGE_FP32_URL,
    entry_function="forward",
    input_type_templates=[
        string.Template("${batch_size}x384xi64"),
        string.Template("${batch_size}x384xi64"),
    ],
    batch_sizes=[1, 16, 24, 32, 48, 64, 512, 1024, 1280],
)

FALCON7B_1X100XI64_GPTQ_TORCH = common_definitions.Model(
    id=unique_ids.MODEL_FALCON7B_1X100XI64_GPTQ_TORCH,
    name="Falcon7bGptqPT",
    tags=["gptq", "transformer"],
    source_type=common_definitions.ModelSourceType.EXPORTED_LINALG_MLIR,
    source_url="https://storage.googleapis.com/iree-model-artifacts/pytorch/manual/falcon_7b_gptq_linalg_1702432230.mlirbc",
    entry_function="forward",
    input_types=["1x100xi64"],
)

FALCON7B_INT4_1X100XI64_GPTQ_TORCH = common_definitions.Model(
    id=unique_ids.MODEL_FALCON7B_INT4_1X100XI64_GPTQ_TORCH,
    name="Falcon7bInt4GptqPT",
    tags=["gptq", "transformer", "int4"],
    source_type=common_definitions.ModelSourceType.EXPORTED_LINALG_MLIR,
    source_url="https://storage.googleapis.com/iree-model-artifacts/pytorch/manual/falcon_7b_gptq_linalg_int4_1702863828.mlirbc",
    entry_function="forward",
    input_types=["1x100xi64"],
)
