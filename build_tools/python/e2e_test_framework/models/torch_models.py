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
CLIP_TEXT_SEQLEN64_FP32_TORCH = common_definitions.Model(
    id=unique_ids.MODEL_CLIP_TEXT_SEQLEN64_FP32_TORCH,
    name="ClipTextSeqLen64PT",
    tags=["fp32", "seqlen64", "transformer"],
    source_type=common_definitions.ModelSourceType.EXPORTED_LINALG_MLIR,
    source_url=
    "https://storage.googleapis.com/iree-model-artifacts/pytorch/torch_models_20230321.784_1679461251/SD_CLIP_TEXT_MODEL_SEQLEN64/batch_1/linalg.mlir",
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
UNET_2D_FP32_TORCH = common_definitions.Model(
    id=unique_ids.MODEL_UNET_2D_FP32_TORCH,
    name="Unet2dPT",
    tags=["fp32", "cnn", "unet"],
    source_type=common_definitions.ModelSourceType.EXPORTED_LINALG_MLIR,
    source_url=
    "https://storage.googleapis.com/iree-model-artifacts/pytorch/torch_models_20230321.784_1679461251/SD_UNET_MODEL/batch_1/linalg.mlir",
    entry_function="forward",
    input_types=["1x4x64x64xf32", "1x77x768xf32"])

# `VAE`: Variational auto-encoder.
#
# Compresses an input image into latent space using its encoder.
# Uncompresses latents into images using the decoder.
#
# Allows Stable Diffusion to perform diffusion in the latent space and convert to a higher resolution image using the `VAE` decoder.
#
# Converted from https://huggingface.co/docs/diffusers/api/models#diffusers.AutoencoderKL.
VAE_FP32_TORCH = common_definitions.Model(
    id=unique_ids.MODEL_VAE_FP32_TORCH,
    name="VaePT",
    tags=["fp32", "cnn"],
    source_type=common_definitions.ModelSourceType.EXPORTED_LINALG_MLIR,
    source_url=
    "https://storage.googleapis.com/iree-model-artifacts/pytorch/torch_models_20230321.784_1679461251/SD_VAE_MODEL/batch_1/linalg.mlir",
    entry_function="forward",
    input_types=["1x4x64x64xf32"])

# Converted from https://pytorch.org/vision/stable/models/generated/torchvision.models.efficientnet_b7.html#torchvision.models.efficientnet_b7
EFFICIENTNET_B7_FP32_TORCH = common_definitions.Model(
    id=unique_ids.MODEL_EFFICIENTNET_B7_FP32_TORCH,
    name="EfficientNetB7PT",
    tags=["fp32", "cnn", "depthwise-conv"],
    source_type=common_definitions.ModelSourceType.EXPORTED_LINALG_MLIR,
    source_url=
    "https://storage.googleapis.com/iree-model-artifacts/pytorch/torch_models_20230321.784_1679461251/EFFICIENTNET_B7/batch_1/linalg.mlir",
    entry_function="forward",
    input_types=["1x3x600x600xi64"])

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

# Converted from https://huggingface.co/docs/transformers/v4.27.2/en/model_doc/bert#transformers.BertModel
BERT_LARGE_1x384_FP32_TORCH = common_definitions.Model(
    id=unique_ids.MODEL_BERT_LARGE_1x384_FP32_TORCH,
    name="BertLargeBatch1PT",
    tags=["fp32", "transformer", "batch-1"],
    source_type=common_definitions.ModelSourceType.EXPORTED_LINALG_MLIR,
    source_url=
    "https://storage.googleapis.com/iree-model-artifacts/pytorch/torch_models_20230321.784_1679461251/BERT_LARGE/batch_1/linalg.mlir",
    entry_function="forward",
    input_types=["1x384xf32", "1x384xf32"])

BERT_LARGE_8x384_FP32_TORCH = common_definitions.Model(
    id=unique_ids.MODEL_BERT_LARGE_8x384_FP32_TORCH,
    name="BertLargeBatch8PT",
    tags=["fp32", "transformer", "batch-8"],
    source_type=common_definitions.ModelSourceType.EXPORTED_LINALG_MLIR,
    source_url=
    "https://storage.googleapis.com/iree-model-artifacts/pytorch/torch_models_20230321.784_1679461251/BERT_LARGE/batch_8/linalg.mlir",
    entry_function="forward",
    input_types=["8x384xf32", "8x384xf32"])

BERT_LARGE_16x384_FP32_TORCH = common_definitions.Model(
    id=unique_ids.MODEL_BERT_LARGE_16x384_FP32_TORCH,
    name="BertLargeBatch16PT",
    tags=["fp32", "transformer", "batch-16"],
    source_type=common_definitions.ModelSourceType.EXPORTED_LINALG_MLIR,
    source_url=
    "https://storage.googleapis.com/iree-model-artifacts/pytorch/torch_models_20230321.784_1679461251/BERT_LARGE/batch_16/linalg.mlir",
    entry_function="forward",
    input_types=["16x384xf32", "16x384xf32"])

BERT_LARGE_32x384_FP32_TORCH = common_definitions.Model(
    id=unique_ids.MODEL_BERT_LARGE_32x384_FP32_TORCH,
    name="BertLargeBatch32PT",
    tags=["fp32", "transformer", "batch-32"],
    source_type=common_definitions.ModelSourceType.EXPORTED_LINALG_MLIR,
    source_url=
    "https://storage.googleapis.com/iree-model-artifacts/pytorch/torch_models_20230321.784_1679461251/BERT_LARGE/batch_32/linalg.mlir",
    entry_function="forward",
    input_types=["32x384xf32", "32x384xf32"])

BERT_LARGE_64x384_FP32_TORCH = common_definitions.Model(
    id=unique_ids.MODEL_BERT_LARGE_64x384_FP32_TORCH,
    name="BertLargeBatch64PT",
    tags=["fp32", "transformer", "batch-64"],
    source_type=common_definitions.ModelSourceType.EXPORTED_LINALG_MLIR,
    source_url=
    "https://storage.googleapis.com/iree-model-artifacts/pytorch/torch_models_20230321.784_1679461251/BERT_LARGE/batch_64/linalg.mlir",
    entry_function="forward",
    input_types=["64x384xf32", "64x384xf32"])

BERT_LARGE_128x384_FP32_TORCH = common_definitions.Model(
    id=unique_ids.MODEL_BERT_LARGE_128x384_FP32_TORCH,
    name="BertLargeBatch128PT",
    tags=["fp32", "transformer", "batch-128"],
    source_type=common_definitions.ModelSourceType.EXPORTED_LINALG_MLIR,
    source_url=
    "https://storage.googleapis.com/iree-model-artifacts/pytorch/torch_models_20230321.784_1679461251/BERT_LARGE/batch_128/linalg.mlir",
    entry_function="forward",
    input_types=["128x384xf32", "128x384xf32"])

BERT_LARGE_256x384_FP32_TORCH = common_definitions.Model(
    id=unique_ids.MODEL_BERT_LARGE_256x384_FP32_TORCH,
    name="BertLargeBatch256PT",
    tags=["fp32", "transformer", "batch-256"],
    source_type=common_definitions.ModelSourceType.EXPORTED_LINALG_MLIR,
    source_url=
    "https://storage.googleapis.com/iree-model-artifacts/pytorch/torch_models_20230321.784_1679461251/BERT_LARGE/batch_256/linalg.mlir",
    entry_function="forward",
    input_types=["256x384xf32", "256x384xf32"])

BERT_LARGE_512x384_FP32_TORCH = common_definitions.Model(
    id=unique_ids.MODEL_BERT_LARGE_512x384_FP32_TORCH,
    name="BertLargeBatch512PT",
    tags=["fp32", "transformer", "batch-512"],
    source_type=common_definitions.ModelSourceType.EXPORTED_LINALG_MLIR,
    source_url=
    "https://storage.googleapis.com/iree-model-artifacts/pytorch/torch_models_20230321.784_1679461251/BERT_LARGE/batch_512/linalg.mlir",
    entry_function="forward",
    input_types=["512x384xf32", "512x384xf32"])

BERT_LARGE_1024x384_FP32_TORCH = common_definitions.Model(
    id=unique_ids.MODEL_BERT_LARGE_1024x384_FP32_TORCH,
    name="BertLargeBatch1024PT",
    tags=["fp32", "transformer", "batch-1024"],
    source_type=common_definitions.ModelSourceType.EXPORTED_LINALG_MLIR,
    source_url=
    "https://storage.googleapis.com/iree-model-artifacts/pytorch/torch_models_20230321.784_1679461251/BERT_LARGE/batch_1024/linalg.mlir",
    entry_function="forward",
    input_types=["1024x384xf32", "1024x384xf32"])

# Converted from https://pytorch.org/vision/main/models/generated/torchvision.models.resnet50.html
RESNET50_1x3x224x224_FP32_TORCH = common_definitions.Model(
    id=unique_ids.MODEL_RESNET50_1x3x224x224_FP32_TORCH,
    name="Resnet50Batch1PT",
    tags=["fp32", "cnn", "batch-1"],
    source_type=common_definitions.ModelSourceType.EXPORTED_LINALG_MLIR,
    source_url=
    "https://storage.googleapis.com/iree-model-artifacts/pytorch/torch_models_20230321.784_1679461251/RESNET50/batch_1/linalg.mlir",
    entry_function="forward",
    input_types=["1x3x224x224xf32"])

RESNET50_8x3x224x224_FP32_TORCH = common_definitions.Model(
    id=unique_ids.MODEL_RESNET50_8x3x224x224_FP32_TORCH,
    name="Resnet50Batch8PT",
    tags=["fp32", "cnn", "batch-8"],
    source_type=common_definitions.ModelSourceType.EXPORTED_LINALG_MLIR,
    source_url=
    "https://storage.googleapis.com/iree-model-artifacts/pytorch/torch_models_20230321.784_1679461251/RESNET50/batch_8/linalg.mlir",
    entry_function="forward",
    input_types=["8x3x224x224xf32"])

RESNET50_16x3x224x224_FP32_TORCH = common_definitions.Model(
    id=unique_ids.MODEL_RESNET50_16x3x224x224_FP32_TORCH,
    name="Resnet50Batch16PT",
    tags=["fp32", "cnn", "batch-16"],
    source_type=common_definitions.ModelSourceType.EXPORTED_LINALG_MLIR,
    source_url=
    "https://storage.googleapis.com/iree-model-artifacts/pytorch/torch_models_20230321.784_1679461251/RESNET50/batch_16/linalg.mlir",
    entry_function="forward",
    input_types=["16x3x224x224xf32"])

RESNET50_32x3x224x224_FP32_TORCH = common_definitions.Model(
    id=unique_ids.MODEL_RESNET50_32x3x224x224_FP32_TORCH,
    name="Resnet50Batch32PT",
    tags=["fp32", "cnn", "batch-32"],
    source_type=common_definitions.ModelSourceType.EXPORTED_LINALG_MLIR,
    source_url=
    "https://storage.googleapis.com/iree-model-artifacts/pytorch/torch_models_20230321.784_1679461251/RESNET50/batch_32/linalg.mlir",
    entry_function="forward",
    input_types=["32x3x224x224xf32"])

RESNET50_64x3x224x224_FP32_TORCH = common_definitions.Model(
    id=unique_ids.MODEL_RESNET50_64x3x224x224_FP32_TORCH,
    name="Resnet50Batch64PT",
    tags=["fp32", "cnn", "batch-64"],
    source_type=common_definitions.ModelSourceType.EXPORTED_LINALG_MLIR,
    source_url=
    "https://storage.googleapis.com/iree-model-artifacts/pytorch/torch_models_20230321.784_1679461251/RESNET50/batch_64/linalg.mlir",
    entry_function="forward",
    input_types=["64x3x224x224xf32"])

RESNET50_128x3x224x224_FP32_TORCH = common_definitions.Model(
    id=unique_ids.MODEL_RESNET50_128x3x224x224_FP32_TORCH,
    name="Resnet50Batch128PT",
    tags=["fp32", "cnn", "batch-128"],
    source_type=common_definitions.ModelSourceType.EXPORTED_LINALG_MLIR,
    source_url=
    "https://storage.googleapis.com/iree-model-artifacts/pytorch/torch_models_20230321.784_1679461251/RESNET50/batch_128/linalg.mlir",
    entry_function="forward",
    input_types=["128x3x224x224xf32"])

RESNET50_256x3x224x224_FP32_TORCH = common_definitions.Model(
    id=unique_ids.MODEL_RESNET50_256x3x224x224_FP32_TORCH,
    name="Resnet50Batch256PT",
    tags=["fp32", "cnn", "batch-256"],
    source_type=common_definitions.ModelSourceType.EXPORTED_LINALG_MLIR,
    source_url=
    "https://storage.googleapis.com/iree-model-artifacts/pytorch/torch_models_20230321.784_1679461251/RESNET50/batch_256/linalg.mlir",
    entry_function="forward",
    input_types=["256x3x224x224xf32"])

RESNET50_512x3x224x224_FP32_TORCH = common_definitions.Model(
    id=unique_ids.MODEL_RESNET50_512x3x224x224_FP32_TORCH,
    name="Resnet50Batch512PT",
    tags=["fp32", "cnn", "batch-512"],
    source_type=common_definitions.ModelSourceType.EXPORTED_LINALG_MLIR,
    source_url=
    "https://storage.googleapis.com/iree-model-artifacts/pytorch/torch_models_20230321.784_1679461251/RESNET50/batch_512/linalg.mlir",
    entry_function="forward",
    input_types=["512x3x224x224xf32"])

RESNET50_1024x3x224x224_FP32_TORCH = common_definitions.Model(
    id=unique_ids.MODEL_RESNET50_1024x3x224x224_FP32_TORCH,
    name="Resnet50Batch1024PT",
    tags=["fp32", "cnn", "batch-1024"],
    source_type=common_definitions.ModelSourceType.EXPORTED_LINALG_MLIR,
    source_url=
    "https://storage.googleapis.com/iree-model-artifacts/pytorch/torch_models_20230321.784_1679461251/RESNET50/batch_1024/linalg.mlir",
    entry_function="forward",
    input_types=["1024x3x224x224xf32"])
