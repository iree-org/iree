## Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from benchmarks.definitions.common import Model, ModelSourceType

MOBILENET_V2_MODEL = Model(id="m1234",
                           name="mobilenet_v2",
                           tags=["f32", "imagenet"],
                           source_type=ModelSourceType.EXPORTED_TFLITE,
                           source_uri="https://example.com/mobilenet_v2.tflite",
                           entry_function="main",
                           input_types=["1x224x224x3xf32"])
