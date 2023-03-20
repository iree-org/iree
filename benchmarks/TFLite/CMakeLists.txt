# Copyright 2021 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception


################################################################################
#                                                                              #
# Benchmark models from TFLite                                                 #
#                                                                              #
# Each module specification should be a list containing alternating keys and   #
# values. The fields are: NAME, TAGS, SOURCE, ENTRY_FUNCTION, and              #
# FUNCTION_INPUTS. See the iree_benchmark_suite definition for details         #
# about these fields. Note that these must be quoted when used as arguments.   #
#                                                                              #
################################################################################

set(DEEPLABV3_FP32_MODULE
  NAME
    "DeepLabV3"
  TAGS
    "fp32"
  SOURCE
    # Mirror of https://tfhub.dev/tensorflow/lite-model/deeplabv3/1/default/1
    "https://storage.googleapis.com/iree-model-artifacts/deeplabv3.tflite"
  ENTRY_FUNCTION
    "main"
  FUNCTION_INPUTS
    "1x257x257x3xf32"
)

set(MOBILESSD_FP32_MODULE
  NAME
    "MobileSSD"
  TAGS
    "fp32"
  SOURCE
    # Mirror of https://storage.googleapis.com/download.tensorflow.org/models/tflite/gpu/mobile_ssd_v2_float_coco.tflite
    "https://storage.googleapis.com/iree-model-artifacts/mobile_ssd_v2_float_coco.tflite"
  ENTRY_FUNCTION
    "main"
  FUNCTION_INPUTS
    "1x320x320x3xf32"
)

set(POSENET_FP32_MODULE
  NAME
    "PoseNet"
  TAGS
    "fp32"
  SOURCE
    # Mirror of https://tfhub.dev/tensorflow/lite-model/posenet/mobilenet/float/075/1/default/1
    "https://storage.googleapis.com/iree-model-artifacts/posenet.tflite"
  ENTRY_FUNCTION
    "main"
  FUNCTION_INPUTS
    "1x353x257x3xf32"
)

set(MOBILEBERT_FP32_MODULE
  NAME
    "MobileBertSquad"
  TAGS
    "fp32"
  SOURCE
    # Mirror of https://tfhub.dev/iree/lite-model/mobilebert/fp32/1
    "https://storage.googleapis.com/iree-model-artifacts/mobilebert-baseline-tf2-float.tflite"
  ENTRY_FUNCTION
    "main"
  FUNCTION_INPUTS
    "1x384xi32,1x384xi32,1x384xi32"
)

set(MOBILEBERT_INT8_MODULE
  NAME
    "MobileBertSquad"
  TAGS
    "int8"
  SOURCE
    # Mirror of https://tfhub.dev/iree/lite-model/mobilebert/int8/1
    "https://storage.googleapis.com/iree-model-artifacts/mobilebert-baseline-tf2-quant.tflite"
  ENTRY_FUNCTION
    "main"
  FUNCTION_INPUTS
    "1x384xi32,1x384xi32,1x384xi32"
)

set(MOBILEBERT_FP16_MODULE
  NAME
    "MobileBertSquad"
  TAGS
    "fp16"
  # This uses the same input MLIR source as fp32 to save download time.
  # It requires users to have "--iree-flow-demote-f32-to-f16".
  SOURCE
    # Mirror of https://tfhub.dev/tensorflow/lite-model/mobilebert/1/default/1
    "https://storage.googleapis.com/iree-model-artifacts/mobilebertsquad.tflite"
  ENTRY_FUNCTION
    "main"
  # The conversion done by "--iree-flow-demote-f32-to-f16" won't change the
  # original input signature.
  FUNCTION_INPUTS
    "1x384xi32,1x384xi32,1x384xi32"
)

set(MOBILENET_V1_MODULE
  NAME
    "MobileNetV1"
  TAGS
    "fp32,imagenet"
  SOURCE
    # Mirror of https://tfhub.dev/iree/lite-model/mobilenet_v1_100_224/fp32/1
    "https://storage.googleapis.com/iree-model-artifacts/mobilenet_v1_224_1.0_float.tflite"
  ENTRY_FUNCTION
    "main"
  FUNCTION_INPUTS
    "1x224x224x3xf32"
)

set(MOBILENET_V2_MODULE
  NAME
    "MobileNetV2"
  TAGS
    "fp32,imagenet"
  SOURCE
    # Mirror https://github.com/tensorflow/tflite-support/blob/master/tensorflow_lite_support/metadata/python/tests/testdata/image_classifier/mobilenet_v2_1.0_224.tflite
    "https://storage.googleapis.com/iree-model-artifacts/mobilenet_v2_1.0_224.tflite"
  ENTRY_FUNCTION
    "main"
  FUNCTION_INPUTS
    "1x224x224x3xf32"
)

set(MOBILENET_V3SMALL_MODULE
  NAME
    "MobileNetV3Small"
  TAGS
    "fp32,imagenet"
  SOURCE
    # https://tfhub.dev/google/imagenet/mobilenet_v3_small_100_224/classification/5
    # Manually exported to tflite with static batch dimension
    "https://storage.googleapis.com/iree-model-artifacts/MobileNetV3SmallStaticBatch.tflite"
  ENTRY_FUNCTION
    "main"
  FUNCTION_INPUTS
    "1x224x224x3xf32"
)

set(PERSON_DETECT_INT8_MODULE
  NAME
    "PersonDetect"
  TAGS
    "int8"
  SOURCE
    # Mirror of https://github.com/tensorflow/tflite-micro/raw/aeac6f39e5c7475cea20c54e86d41e3a38312546/tensorflow/lite/micro/models/person_detect.tflite
    "https://storage.googleapis.com/iree-model-artifacts/person_detect.tflite"
  ENTRY_FUNCTION
    "main"
  FUNCTION_INPUTS
    "1x96x96x1xi8"
)

set(EFFICIENTNET_INT8_MODULE
  NAME
    "EfficientNet"
  TAGS
    "int8"
  SOURCE
    # Mirror of https://tfhub.dev/tensorflow/lite-model/efficientnet/lite0/int8/2
    "https://storage.googleapis.com/iree-model-artifacts/efficientnet_lite0_int8_2.tflite"
  ENTRY_FUNCTION
    "main"
  FUNCTION_INPUTS
    "1x224x224x3xui8"
)

set(MOBILENET_V2_INT8_MODULE
  NAME
    "MobileNetV2"
  TAGS
    "int8"
  SOURCE
    # Mirror of https://tfhub.dev/tensorflow/lite-model/mobilenet_v2_1.0_224_quantized/1/default/1
    "https://storage.googleapis.com/iree-model-artifacts/mobilenet_v2_1.0_224_quantized.tflite"
  ENTRY_FUNCTION
    "main"
  FUNCTION_INPUTS
    "1x224x224x3xui8"
)

################################################################################
# Add benchmarks for all platforms.                                            #
################################################################################
include(android-arm64-v8a.cmake)
include(android-adreno.cmake)
include(android-mali.cmake)
include(linux-x86_64.cmake)
include(linux-riscv.cmake)
