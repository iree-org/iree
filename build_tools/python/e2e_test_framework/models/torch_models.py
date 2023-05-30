## Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Defines PyTorch models."""

from e2e_test_framework import unique_ids
from e2e_test_framework.definitions import common_definitions

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
    source_url=
    "https://storage.googleapis.com/iree-model-artifacts/pytorch/torch_models_20230307.103_1678163233/SD_CLIP_TEXT_MODEL_SEQLEN64/linalg.mlir",
    entry_function="forward",
    input_types=["1x77xi64", "1x77xi64"])

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
    source_url=
    "https://storage.googleapis.com/iree-model-artifacts/pytorch/torch_models_20230307.103_1678163233/SD_UNET_MODEL/linalg.mlir",
    entry_function="forward",
    input_types=["1x4x64x64xf32", "1x77x768xf32"])

# Converted from https://pytorch.org/vision/stable/models/generated/torchvision.models.efficientnet_v2_s.html#torchvision.models.efficientnet_v2_s
EFFICIENTNET_V2_S_FP32_TORCH = common_definitions.Model(
    id=unique_ids.MODEL_EFFICIENTNET_V2_S_FP32_TORCH,
    name="EfficientNetV2SPT",
    tags=["fp32", "cnn", "depthwise-conv"],
    source_type=common_definitions.ModelSourceType.EXPORTED_LINALG_MLIR,
    source_url=
    "https://storage.googleapis.com/iree-model-artifacts/pytorch/torch_models_20230321.784_1679461251/EFFICIENTNET_V2_S/batch_1/linalg.mlir",
    entry_function="forward",
    input_types=["1x3x384x384xf32"])

# Converted from https://pytorch.org/vision/stable/models/generated/torchvision.models.efficientnet_b7.html#torchvision.models.efficientnet_b7
EFFICIENTNET_B7_FP32_TORCH = common_definitions.Model(
    id=unique_ids.MODEL_EFFICIENTNET_B7_FP32_TORCH,
    name="EfficientNetB7PT",
    tags=["fp32", "cnn", "depthwise-conv"],
    source_type=common_definitions.ModelSourceType.EXPORTED_LINALG_MLIR,
    source_url=
    "https://storage.googleapis.com/iree-model-artifacts/pytorch/torch_models_20230321.784_1679461251/EFFICIENTNET_B7/batch_1/linalg.mlir",
    entry_function="forward",
    input_types=["1x3x600x600xf32"])

# Converted from https://huggingface.co/docs/transformers/v4.27.2/en/model_doc/bert#transformers.BertModel
BERT_LARGE_1X384_FP32_TORCH = common_definitions.Model(
    id=unique_ids.MODEL_BERT_LARGE_1X384_FP32_TORCH,
    name="BertLargePTBatch1",
    tags=["fp32", "transformer", "seqlen384", "batch-1"],
    source_type=common_definitions.ModelSourceType.EXPORTED_LINALG_MLIR,
    source_url=
    "https://storage.googleapis.com/iree-model-artifacts/pytorch/torch_models_20230401.795_1680469670/BERT_LARGE/batch_1/linalg.mlir",
    entry_function="forward",
    input_types=["1x384xi64", "1x384xi64"])

BERT_LARGE_16X384_FP32_TORCH = common_definitions.Model(
    id=unique_ids.MODEL_BERT_LARGE_16X384_FP32_TORCH,
    name="BertLargePTBatch16",
    tags=["fp32", "transformer", "seqlen384", "batch-16"],
    source_type=common_definitions.ModelSourceType.EXPORTED_LINALG_MLIR,
    source_url=
    "https://storage.googleapis.com/iree-model-artifacts/pytorch/torch_models_20230401.795_1680469670/BERT_LARGE/batch_16/linalg.mlir",
    entry_function="forward",
    input_types=["16x384xi64", "16x384xi64"])

BERT_LARGE_24X384_FP32_TORCH = common_definitions.Model(
    id=unique_ids.MODEL_BERT_LARGE_24X384_FP32_TORCH,
    name="BertLargePTBatch24",
    tags=["fp32", "transformer", "seqlen384", "batch-24"],
    source_type=common_definitions.ModelSourceType.EXPORTED_LINALG_MLIR,
    source_url=
    "https://storage.googleapis.com/iree-model-artifacts/pytorch/torch_models_20230401.795_1680469670/BERT_LARGE/batch_24/linalg.mlir",
    entry_function="forward",
    input_types=["24x384xi64", "24x384xi64"])

BERT_LARGE_32X384_FP32_TORCH = common_definitions.Model(
    id=unique_ids.MODEL_BERT_LARGE_32X384_FP32_TORCH,
    name="BertLargePTBatch32",
    tags=["fp32", "transformer", "seqlen384", "batch-32"],
    source_type=common_definitions.ModelSourceType.EXPORTED_LINALG_MLIR,
    source_url=
    "https://storage.googleapis.com/iree-model-artifacts/pytorch/torch_models_20230401.795_1680469670/BERT_LARGE/batch_32/linalg.mlir",
    entry_function="forward",
    input_types=["32x384xi64", "32x384xi64"])

BERT_LARGE_48X384_FP32_TORCH = common_definitions.Model(
    id=unique_ids.MODEL_BERT_LARGE_48X384_FP32_TORCH,
    name="BertLargePTBatch48",
    tags=["fp32", "transformer", "seqlen384", "batch-48"],
    source_type=common_definitions.ModelSourceType.EXPORTED_LINALG_MLIR,
    source_url=
    "https://storage.googleapis.com/iree-model-artifacts/pytorch/torch_models_20230401.795_1680469670/BERT_LARGE/batch_48/linalg.mlir",
    entry_function="forward",
    input_types=["48x384xi64", "48x384xi64"])

BERT_LARGE_64X384_FP32_TORCH = common_definitions.Model(
    id=unique_ids.MODEL_BERT_LARGE_64X384_FP32_TORCH,
    name="BertLargePTBatch64",
    tags=["fp32", "transformer", "seqlen384", "batch-64"],
    source_type=common_definitions.ModelSourceType.EXPORTED_LINALG_MLIR,
    source_url=
    "https://storage.googleapis.com/iree-model-artifacts/pytorch/torch_models_20230401.795_1680469670/BERT_LARGE/batch_64/linalg.mlir",
    entry_function="forward",
    input_types=["64x384xi64", "64x384xi64"])

BERT_LARGE_512X384_FP32_TORCH = common_definitions.Model(
    id=unique_ids.MODEL_BERT_LARGE_512X384_FP32_TORCH,
    name="BertLargePTBatch512",
    tags=["fp32", "transformer", "seqlen384", "batch-512"],
    source_type=common_definitions.ModelSourceType.EXPORTED_LINALG_MLIR,
    source_url=
    "https://storage.googleapis.com/iree-model-artifacts/pytorch/torch_models_20230401.795_1680469670/BERT_LARGE/batch_512/linalg.mlir",
    entry_function="forward",
    input_types=["512x384xi64", "512x384xi64"])

BERT_LARGE_1024X384_FP32_TORCH = common_definitions.Model(
    id=unique_ids.MODEL_BERT_LARGE_1024X384_FP32_TORCH,
    name="BertLargePTBatch1024",
    tags=["fp32", "transformer", "seqlen384", "batch-1024"],
    source_type=common_definitions.ModelSourceType.EXPORTED_LINALG_MLIR,
    source_url=
    "https://storage.googleapis.com/iree-model-artifacts/pytorch/torch_models_20230401.795_1680469670/BERT_LARGE/batch_1024/linalg.mlir",
    entry_function="forward",
    input_types=["1024x384xi64", "1024x384xi64"])

BERT_LARGE_1280X384_FP32_TORCH = common_definitions.Model(
    id=unique_ids.MODEL_BERT_LARGE_1280X384_FP32_TORCH,
    name="BertLargePTBatch1280",
    tags=["fp32", "transformer", "seqlen384", "batch-1280"],
    source_type=common_definitions.ModelSourceType.EXPORTED_LINALG_MLIR,
    source_url=
    "https://storage.googleapis.com/iree-model-artifacts/pytorch/torch_models_20230401.795_1680469670/BERT_LARGE/batch_1280/linalg.mlir",
    entry_function="forward",
    input_types=["1280x384xi64", "1280x384xi64"])

# Converted from https://pytorch.org/vision/main/models/generated/torchvision.models.resnet50.html
RESNET50_1X3X224X224_FP32_TORCH = common_definitions.Model(
    id=unique_ids.MODEL_RESNET50_1X3X224X224_FP32_TORCH,
    name="Resnet50PTBatch1",
    tags=["fp32", "cnn", "batch-1"],
    source_type=common_definitions.ModelSourceType.EXPORTED_LINALG_MLIR,
    source_url=
    "https://storage.googleapis.com/iree-model-artifacts/pytorch/torch_models_20230401.795_1680469670/RESNET50/batch_1/linalg.mlir",
    entry_function="forward",
    input_types=["1x3x224x224xf32"])

RESNET50_8X3X224X224_FP32_TORCH = common_definitions.Model(
    id=unique_ids.MODEL_RESNET50_8X3X224X224_FP32_TORCH,
    name="Resnet50PTBatch8",
    tags=["fp32", "cnn", "batch-8"],
    source_type=common_definitions.ModelSourceType.EXPORTED_LINALG_MLIR,
    source_url=
    "https://storage.googleapis.com/iree-model-artifacts/pytorch/torch_models_20230401.795_1680469670/RESNET50/batch_8/linalg.mlir",
    entry_function="forward",
    input_types=["8x3x224x224xf32"])

RESNET50_64X3X224X224_FP32_TORCH = common_definitions.Model(
    id=unique_ids.MODEL_RESNET50_64X3X224X224_FP32_TORCH,
    name="Resnet50PTBatch64",
    tags=["fp32", "cnn", "batch-64"],
    source_type=common_definitions.ModelSourceType.EXPORTED_LINALG_MLIR,
    source_url=
    "https://storage.googleapis.com/iree-model-artifacts/pytorch/torch_models_20230401.795_1680469670/RESNET50/batch_64/linalg.mlir",
    entry_function="forward",
    input_types=["64x3x224x224xf32"])

RESNET50_128X3X224X224_FP32_TORCH = common_definitions.Model(
    id=unique_ids.MODEL_RESNET50_128X3X224X224_FP32_TORCH,
    name="Resnet50PTBatch128",
    tags=["fp32", "cnn", "batch-128"],
    source_type=common_definitions.ModelSourceType.EXPORTED_LINALG_MLIR,
    source_url=
    "https://storage.googleapis.com/iree-model-artifacts/pytorch/torch_models_20230401.795_1680469670/RESNET50/batch_128/linalg.mlir",
    entry_function="forward",
    input_types=["128x3x224x224xf32"])

RESNET50_256X3X224X224_FP32_TORCH = common_definitions.Model(
    id=unique_ids.MODEL_RESNET50_256X3X224X224_FP32_TORCH,
    name="Resnet50PTBatch256",
    tags=["fp32", "cnn", "batch-256"],
    source_type=common_definitions.ModelSourceType.EXPORTED_LINALG_MLIR,
    source_url=
    "https://storage.googleapis.com/iree-model-artifacts/pytorch/torch_models_20230401.795_1680469670/RESNET50/batch_256/linalg.mlir",
    entry_function="forward",
    input_types=["256x3x224x224xf32"])

RESNET50_2048X3X224X224_FP32_TORCH = common_definitions.Model(
    id=unique_ids.MODEL_RESNET50_2048X3X224X224_FP32_TORCH,
    name="Resnet50PTBatch2048",
    tags=["fp32", "cnn", "batch-2048"],
    source_type=common_definitions.ModelSourceType.EXPORTED_LINALG_MLIR,
    source_url=
    "https://storage.googleapis.com/iree-model-artifacts/pytorch/torch_models_20230401.795_1680469670/RESNET50/batch_2048/linalg.mlir",
    entry_function="forward",
    input_types=["2048x3x224x224xf32"])

# Converted from https://pytorch.org/vision/main/models/generated/torchvision.models.resnet50.html
RESNET50_1X3X224X224_FP16_TORCH = common_definitions.Model(
    id=unique_ids.MODEL_RESNET50_1X3X224X224_FP16_TORCH,
    name="Resnet50fp16PTBatch1",
    tags=["fp16", "cnn", "batch-1"],
    source_type=common_definitions.ModelSourceType.EXPORTED_LINALG_MLIR,
    source_url=
    "https://storage.googleapis.com/iree-model-artifacts/pytorch/torch_models_20230522.846_1684830698/RESNET50_FP16/batch_1/linalg.mlir",
    entry_function="forward",
    input_types=["1x3x224x224xf16"])

RESNET50_8X3X224X224_FP16_TORCH = common_definitions.Model(
    id=unique_ids.MODEL_RESNET50_8X3X224X224_FP16_TORCH,
    name="Resnet50fp16PT16Batch8",
    tags=["fp16", "cnn", "batch-8"],
    source_type=common_definitions.ModelSourceType.EXPORTED_LINALG_MLIR,
    source_url=
    "https://storage.googleapis.com/iree-model-artifacts/pytorch/torch_models_20230522.846_1684830698/RESNET50_FP16/batch_8/linalg.mlir",
    entry_function="forward",
    input_types=["8x3x224x224xf16"])

RESNET50_64X3X224X224_FP16_TORCH = common_definitions.Model(
    id=unique_ids.MODEL_RESNET50_64X3X224X224_FP16_TORCH,
    name="Resnet50fp16PTBatch64",
    tags=["fp16", "cnn", "batch-64"],
    source_type=common_definitions.ModelSourceType.EXPORTED_LINALG_MLIR,
    source_url=
    "https://storage.googleapis.com/iree-model-artifacts/pytorch/torch_models_20230522.846_1684830698/RESNET50_FP16/batch_64/linalg.mlir",
    entry_function="forward",
    input_types=["64x3x224x224xf16"])

RESNET50_128X3X224X224_FP16_TORCH = common_definitions.Model(
    id=unique_ids.MODEL_RESNET50_128X3X224X224_FP16_TORCH,
    name="Resnet50fp16PTBatch128",
    tags=["fp16", "cnn", "batch-128"],
    source_type=common_definitions.ModelSourceType.EXPORTED_LINALG_MLIR,
    source_url=
    "https://storage.googleapis.com/iree-model-artifacts/pytorch/torch_models_20230522.846_1684830698/RESNET50_FP16/batch_128/linalg.mlir",
    entry_function="forward",
    input_types=["128x3x224x224xf16"])

RESNET50_256X3X224X224_FP16_TORCH = common_definitions.Model(
    id=unique_ids.MODEL_RESNET50_256X3X224X224_FP16_TORCH,
    name="Resnet50fp16PTBatch256",
    tags=["fp16", "cnn", "batch-256"],
    source_type=common_definitions.ModelSourceType.EXPORTED_LINALG_MLIR,
    source_url=
    "https://storage.googleapis.com/iree-model-artifacts/pytorch/torch_models_20230522.846_1684830698/RESNET50_FP16/batch_256/linalg.mlir",
    entry_function="forward",
    input_types=["256x3x224x224xf16"])

RESNET50_2048X3X224X224_FP16_TORCH = common_definitions.Model(
    id=unique_ids.MODEL_RESNET50_2048X3X224X224_FP16_TORCH,
    name="Resnet50fp16PTBatch2048",
    tags=["fp16", "cnn", "batch-2048"],
    source_type=common_definitions.ModelSourceType.EXPORTED_LINALG_MLIR,
    source_url=
    "https://storage.googleapis.com/iree-model-artifacts/pytorch/torch_models_20230522.846_1684830698/RESNET50_FP16/batch_2048/linalg.mlir",
    entry_function="forward",
    input_types=["2048x3x224x224xf16"])
