## Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Defines TFLite models."""

from ..id_defs import MODEL_MOBILENET_V2_ID
from ..definitions.common import Model, ModelSourceType

# Mirror https://github.com/tensorflow/tflite-support/blob/master/tensorflow_lite_support/metadata/python/tests/testdata/image_classifier/mobilenet_v2_1.0_224.tflite
MOBILENET_V2_MODEL = Model(
    id=MODEL_MOBILENET_V2_ID,
    name="mobilenet_v2",
    tags=["f32", "imagenet"],
    source_type=ModelSourceType.EXPORTED_TFLITE,
    source_url=
    "https://storage.googleapis.com/iree-model-artifacts/mobilenet_v2_1.0_224.tflite",
    entry_function="main",
    input_types=["1x224x224x3xf32"])
