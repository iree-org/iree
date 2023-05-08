## Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Defines TFLite models."""

from e2e_test_framework import unique_ids
from e2e_test_framework.definitions import common_definitions

DEEPLABV3_FP32 = common_definitions.Model(
    id=unique_ids.MODEL_DEEPLABV3_FP32,
    name="DeepLabV3_fp32",
    tags=["fp32"],
    source_type=common_definitions.ModelSourceType.EXPORTED_TFLITE,
    # Mirror of https://tfhub.dev/tensorflow/lite-model/deeplabv3/1/default/1
    source_url=
    "https://storage.googleapis.com/iree-model-artifacts/deeplabv3.tflite",
    entry_function="main",
    input_types=["1x257x257x3xf32"])

MOBILESSD_FP32 = common_definitions.Model(
    id=unique_ids.MODEL_MOBILESSD_FP32,
    name="MobileSSD_fp32",
    tags=["fp32"],
    source_type=common_definitions.ModelSourceType.EXPORTED_TFLITE,
    # Mirror of https://storage.googleapis.com/download.tensorflow.org/models/tflite/gpu/mobile_ssd_v2_float_coco.tflite
    source_url=
    "https://storage.googleapis.com/iree-model-artifacts/mobile_ssd_v2_float_coco.tflite",
    entry_function="main",
    input_types=["1x320x320x3xf32"])

POSENET_FP32 = common_definitions.Model(
    id=unique_ids.MODEL_POSENET_FP32,
    name="PoseNet_fp32",
    tags=["fp32"],
    source_type=common_definitions.ModelSourceType.EXPORTED_TFLITE,
    # Mirror of https://tfhub.dev/tensorflow/lite-model/posenet/mobilenet/float/075/1/default/1
    source_url=
    "https://storage.googleapis.com/iree-model-artifacts/posenet.tflite",
    entry_function="main",
    input_types=["1x353x257x3xf32"])

MOBILEBERT_FP32 = common_definitions.Model(
    id=unique_ids.MODEL_MOBILEBERT_FP32,
    name="MobileBertSquad_fp32",
    tags=["fp32"],
    source_type=common_definitions.ModelSourceType.EXPORTED_TFLITE,
    # Mirror of https://tfhub.dev/iree/lite-model/mobilebert/fp32/1
    source_url=
    "https://storage.googleapis.com/iree-model-artifacts/mobilebert-baseline-tf2-float.tflite",
    entry_function="main",
    input_types=["1x384xi32", "1x384xi32", "1x384xi32"])

MOBILEBERT_INT8 = common_definitions.Model(
    id=unique_ids.MODEL_MOBILEBERT_INT8,
    name="MobileBertSquad_int8",
    tags=["int8"],
    source_type=common_definitions.ModelSourceType.EXPORTED_TFLITE,
    # Mirror of https://tfhub.dev/iree/lite-model/mobilebert/int8/1
    source_url=
    "https://storage.googleapis.com/iree-model-artifacts/mobilebert-baseline-tf2-quant.tflite",
    entry_function="main",
    input_types=["1x384xi32", "1x384xi32", "1x384xi32"])

MOBILEBERT_FP16 = common_definitions.Model(
    id=unique_ids.MODEL_MOBILEBERT_FP16,
    name="MobileBertSquad_fp16",
    tags=["fp16"],
    source_type=common_definitions.ModelSourceType.EXPORTED_TFLITE,
    # Mirror of https://tfhub.dev/tensorflow/lite-model/mobilebert/1/default/1
    source_url=
    "https://storage.googleapis.com/iree-model-artifacts/mobilebertsquad.tflite",
    entry_function="main",
    input_types=["1x384xi32", "1x384xi32", "1x384xi32"])

MOBILENET_V1 = common_definitions.Model(
    id=unique_ids.MODEL_MOBILENET_V1,
    name="MobileNetV1_fp32",
    tags=["fp32", "imagenet"],
    source_type=common_definitions.ModelSourceType.EXPORTED_TFLITE,
    # Mirror of https://tfhub.dev/iree/lite-model/mobilenet_v1_100_224/fp32/1
    source_url=
    "https://storage.googleapis.com/iree-model-artifacts/mobilenet_v1_224_1.0_float.tflite",
    entry_function="main",
    input_types=["1x224x224x3xf32"])

MOBILENET_V2 = common_definitions.Model(
    id=unique_ids.MODEL_MOBILENET_V2,
    name="MobileNetV2_fp32",
    tags=["fp32", "imagenet"],
    source_type=common_definitions.ModelSourceType.EXPORTED_TFLITE,
    # Mirror of https://github.com/tensorflow/tflite-support/blob/master/tensorflow_lite_support/metadata/python/tests/testdata/image_classifier/mobilenet_v2_1.0_224.tflite
    source_url=
    "https://storage.googleapis.com/iree-model-artifacts/mobilenet_v2_1.0_224.tflite",
    entry_function="main",
    input_types=["1x224x224x3xf32"])

MOBILENET_V3SMALL = common_definitions.Model(
    id=unique_ids.MODEL_MOBILENET_V3SMALL,
    name="MobileNetV3Small_fp32",
    tags=["fp32", "imagenet"],
    source_type=common_definitions.ModelSourceType.EXPORTED_TFLITE,
    # https://tfhub.dev/google/imagenet/mobilenet_v3_small_100_224/classification/5
    # Manually exported to tflite with static batch dimension
    source_url=
    "https://storage.googleapis.com/iree-model-artifacts/MobileNetV3SmallStaticBatch.tflite",
    entry_function="main",
    input_types=["1x224x224x3xf32"])

PERSON_DETECT_INT8 = common_definitions.Model(
    id=unique_ids.MODEL_PERSON_DETECT_INT8,
    name="PersonDetect_int8",
    tags=["int8"],
    source_type=common_definitions.ModelSourceType.EXPORTED_TFLITE,
    # Mirror of https://github.com/tensorflow/tflite-micro/raw/aeac6f39e5c7475cea20c54e86d41e3a38312546/tensorflow/lite/micro/models/person_detect.tflite
    source_url=
    "https://storage.googleapis.com/iree-model-artifacts/person_detect.tflite",
    entry_function="main",
    input_types=["1x96x96x1xi8"])

EFFICIENTNET_INT8 = common_definitions.Model(
    id=unique_ids.MODEL_EFFICIENTNET_INT8,
    name="EfficientNet_int8",
    tags=["int8"],
    source_type=common_definitions.ModelSourceType.EXPORTED_TFLITE,
    # Mirror of https://tfhub.dev/tensorflow/lite-model/efficientnet/lite0/int8/2
    source_url=
    "https://storage.googleapis.com/iree-model-artifacts/efficientnet_lite0_int8_2.tflite",
    entry_function="main",
    input_types=["1x224x224x3xui8"])

MOBILENET_V2_INT8 = common_definitions.Model(
    name="MobileNetV2_int8",
    id=unique_ids.MODEL_MOBILENET_V2_INT8,
    tags=["int8", "imagenet"],
    source_type=common_definitions.ModelSourceType.EXPORTED_TFLITE,
    # Mirror of https://tfhub.dev/tensorflow/lite-model/mobilenet_v2_1.0_224_quantized/1/default/1
    source_url=
    "https://storage.googleapis.com/iree-model-artifacts/mobilenet_v2_1.0_224_quantized.tflite",
    entry_function="main",
    input_types=["1x224x224x3xui8"])
